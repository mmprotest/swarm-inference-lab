"""First-class Colibri implementation of the Universal Worker backend ABI."""

from __future__ import annotations

import asyncio
import json
import platform
import time
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.backends.colibri.model import ColibriModelInspector, resolve_model_family
from swarm_inference.backends.colibri.plan import ColibriPlanTranslator
from swarm_inference.backends.colibri.probe import ColibriCapabilityProbe
from swarm_inference.backends.colibri.process import ColibriProcess
from swarm_inference.backends.colibri.replay import ColibriReplayRunner, ReplayTokenSequence
from swarm_inference.backends.colibri.schemas import ColibriMode, TelemetryLevel
from swarm_inference.backends.colibri.telemetry import (
    ColibriRouteTraceReader,
    ColibriTelemetryReader,
    ColibriUsageHistoryReader,
)
from swarm_inference.worker.abi import (
    BackendAdapter,
    ResultClassification,
    TokenPayload,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
)

_COLIBRI_JOB_TYPES = frozenset(
    {
        WorkerJobType.HEALTH_CHECK,
        WorkerJobType.CAPABILITY_PROBE,
        WorkerJobType.MODEL_INVENTORY,
        WorkerJobType.RESOURCE_PLAN,
        WorkerJobType.GENERATE,
        WorkerJobType.STREAM_GENERATE,
        WorkerJobType.TEACHER_FORCED_REPLAY,
        WorkerJobType.PROFILE,
        WorkerJobType.ROUTE_TRACE,
        WorkerJobType.PLACEMENT_EVALUATION,
        WorkerJobType.AUTOTUNE,
        WorkerJobType.SHUTDOWN,
    }
)


class ColibriBackend(BackendAdapter):
    """Control one local Colibri runtime without cross-backend fallback."""

    backend_id = "colibri"
    supported_jobs = _COLIBRI_JOB_TYPES

    def __init__(
        self,
        *,
        engine_directory: str | Path,
        model_path: str | Path,
        model_id: str,
        model_revision: str,
        source_directory: str | Path | None = None,
        build_manifest: str | Path | None = None,
        model_family: str | None = None,
        mode: ColibriMode = ColibriMode.BRIDGE,
        telemetry_level: TelemetryLevel = TelemetryLevel.SUMMARY,
        telemetry_path: str | Path | None = None,
        log_directory: str | Path | None = None,
        cap: int | None = None,
        environment: dict[str, str] | None = None,
        execution_profile_id: str | None = None,
        execution_profile_fingerprint: str | None = None,
        ram_safety_reserve_bytes: int = 8 * 1024**3,
    ) -> None:
        self.engine_directory = Path(engine_directory).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_family = model_family
        self.mode = mode
        self.telemetry_level = telemetry_level
        self.log_directory = (
            Path(log_directory).expanduser().resolve()
            if log_directory is not None
            else self.model_path / ".swarm_colibri_logs"
        )
        self.telemetry_path = (
            Path(telemetry_path).expanduser().resolve()
            if telemetry_path is not None
            else self.log_directory / "telemetry.ndjson"
        )
        self.cap = cap
        self.environment = dict(environment or {})
        self.execution_profile_id = execution_profile_id
        self.execution_profile_fingerprint = execution_profile_fingerprint
        self.ram_safety_reserve_bytes = ram_safety_reserve_bytes
        self.probe = ColibriCapabilityProbe(
            self.engine_directory,
            source_directory=source_directory,
            build_manifest=build_manifest,
            model_path=self.model_path,
        )
        self.inspector = ColibriModelInspector(self.engine_directory)
        self.plan_translator = ColibriPlanTranslator()
        self.process: ColibriProcess | None = None
        self._last_profile = WorkerBenchmarkProfile(
            model_revision=model_revision, model_load_seconds=0, warmup_seconds=0
        )
        self._inventory_cache: tuple[Any, Any, Any, Any] | None = None
        self._tokenizer: Any | None = None

    def _local_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_path,
                local_files_only=True,
            )
        return self._tokenizer

    def prompt_payload(
        self,
        prompt: str,
        *,
        tokenizer_hash: str | None = None,
    ) -> TokenPayload:
        """Create the backend-native prompt payload from the immutable local snapshot."""

        family = resolve_model_family(self._config(), self.model_family)
        if family != "olmoe":
            return TokenPayload(token_ids=[], text=prompt, tokenizer_hash=tokenizer_hash)
        encoded = self._local_tokenizer()(
            prompt,
            add_special_tokens=True,
            return_tensors=None,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        if not token_ids:
            raise ValueError("Colibri prompt tokenization produced no token IDs")
        return TokenPayload(
            token_ids=token_ids,
            text=prompt,
            tokenizer_hash=tokenizer_hash,
        )

    def decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return str(self._local_tokenizer().decode(token_ids, skip_special_tokens=False))

    def capabilities(self) -> WorkerCapabilities:
        report = self.probe.probe()
        memory = report.memory
        physical = int(report.cpu.get("physical_cores", 1))
        logical = int(report.cpu.get("logical_cores", physical))
        gpu = report.gpu_devices[0] if report.gpu_devices else None
        config = self._config()
        context = int(
            config.get("max_position_embeddings", 0)
            or config.get("max_sequence_length", 0)
            or config.get("seq_length", 0)
            or 4096
        )
        features = [
            name
            for name, enabled in {
                "expert_residency": report.supports_expert_residency,
                "route_trace": report.supports_route_trace,
                "usage_history": report.supports_usage_history,
                "expert_prefetch": report.supports_expert_prefetch,
                "native_mxfp4": report.supports_native_mxfp4,
                "exact_replay": report.supports_exact_replay,
                "prefill_decode_separation": report.supports_prefill_decode_separation,
            }.items()
            if enabled
        ]
        accelerator_memory = int(gpu.get("total_memory_bytes", 0)) if gpu else 0
        return WorkerCapabilities(
            architecture=report.architecture or platform.machine(),
            operating_system=report.platform,
            cpu_model=str(report.cpu.get("model", "unknown")),
            physical_cpu_cores=max(1, physical),
            logical_cpu_cores=max(1, logical),
            accelerator_type=(
                "cuda" if report.supports_cuda else "vulkan" if report.supports_vulkan else None
            ),
            accelerator_model=str(gpu.get("name")) if gpu else None,
            accelerator_memory_bytes=accelerator_memory,
            system_memory_bytes=int(memory.get("total_bytes", psutil.virtual_memory().total)),
            supported_weight_formats=report.quantization_formats,
            supported_activation_dtypes=["f32", "f16", "bf16"],
            supported_cache_dtypes=["f32"],
            maximum_weight_bytes=max(
                0, int(memory.get("available_bytes", 0)) - self.ram_safety_reserve_bytes
            ),
            maximum_cache_bytes=max(
                0, int(memory.get("available_bytes", 0)) - self.ram_safety_reserve_bytes
            ),
            maximum_batch_size=1,
            maximum_context_length=max(1, context),
            measured_network_upload_bps=0,
            measured_network_download_bps=0,
            coordinator_latency_ms=0,
            backend_features=features,
            backend_details=report.model_dump(mode="json"),
        )

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return self._last_profile

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        rejected = self.admission_result(job)
        if rejected is not None:
            return rejected
        try:
            return await asyncio.to_thread(self._execute_sync, job)
        except InterruptedError as error:
            return self._result(job, WorkerJobStatus.CANCELLED, detail=str(error))
        except MemoryError as error:
            return self._result(job, WorkerJobStatus.INSUFFICIENT_MEMORY, detail=str(error))
        except Exception as error:
            return self._result(
                job,
                WorkerJobStatus.BACKEND_FAILURE,
                detail=f"{type(error).__name__}: {error}",
            )

    async def cancel(self, request_id: str) -> bool:
        return self.process.cancel(request_id) if self.process is not None else False

    async def shutdown(self) -> None:
        if self.process is not None:
            await asyncio.to_thread(self.process.shutdown)
            self.process = None

    def _execute_sync(self, job: WorkerJob) -> WorkerJobResult:
        role = job.role
        if role == WorkerJobType.HEALTH_CHECK:
            health = self.process.health() if self.process is not None else {"status": "stopped"}
            return self._result(job, metrics={"health": health})
        if role == WorkerJobType.CAPABILITY_PROBE:
            return self._result(
                job, metrics={"capability_report": self.probe.probe().model_dump(mode="json")}
            )
        if role == WorkerJobType.MODEL_INVENTORY:
            inventory, tensors, experts, native = self._inventory()
            return self._result(
                job,
                metrics={
                    "model_inventory": inventory.model_dump(mode="json"),
                    "tensor_inventory": [item.model_dump(mode="json") for item in tensors],
                    "expert_inventory": [item.model_dump(mode="json") for item in experts],
                    "native_quantization_inventory": [
                        item.model_dump(mode="json") for item in native
                    ],
                },
            )
        if role == WorkerJobType.RESOURCE_PLAN:
            native = self._native_plan(job.metadata)
            inventory, tensors, experts, _ = self._inventory()
            plan = self.plan_translator.translate(
                native,
                hardware_fingerprint=str(job.metadata["hardware_fingerprint"]),
                tensors=tensors if job.metadata.get("reconcile_tensors", True) else None,
                experts=experts if job.metadata.get("reconcile_experts", True) else None,
                capabilities=self.probe.probe(),
            )
            return self._result(
                job,
                metrics={
                    "colibri_resource_plan": native,
                    "swarm_resource_plan": plan.model_dump(mode="json"),
                    "model_inventory": inventory.model_dump(mode="json"),
                },
            )
        if role in {WorkerJobType.GENERATE, WorkerJobType.STREAM_GENERATE}:
            return self._generate(job, stream=role == WorkerJobType.STREAM_GENERATE)
        if role == WorkerJobType.TEACHER_FORCED_REPLAY:
            return self._replay(job)
        if role == WorkerJobType.PROFILE:
            return self._profile(job)
        if role == WorkerJobType.ROUTE_TRACE:
            trace = Path(str(job.metadata["route_trace_path"]))
            selections = ColibriRouteTraceReader().read(trace)
            summary = ColibriRouteTraceReader.summarize(selections)
            return self._result(
                job,
                metrics={
                    "route_selections": [item.model_dump(mode="json") for item in selections],
                    "route_summary": summary,
                },
            )
        if role == WorkerJobType.PLACEMENT_EVALUATION:
            return self._placement_evaluation(job)
        if role == WorkerJobType.AUTOTUNE:
            raise ValueError(
                "autotune jobs require the experiment runner's bounded candidate matrix; "
                "use ColibriFixedReplayTuner"
            )
        if role == WorkerJobType.SHUTDOWN:
            if self.process is not None:
                self.process.shutdown()
                self.process = None
            return self._result(job, metrics={"clean_shutdown": True})
        raise ValueError(f"unhandled Colibri worker role {role.value}")

    def _generate(self, job: WorkerJob, *, stream: bool) -> WorkerJobResult:
        invocation_started_ns = time.time_ns()
        parameters = job.generation_parameters
        if parameters is None:
            raise ValueError("Colibri generation requires generation_parameters")
        family = resolve_model_family(self._config(), self.model_family)
        if family == "olmoe":
            if stream:
                raise NotImplementedError(
                    "Colibri v1.4.0 OLMoE has no persistent streaming mux; "
                    "the GLM/Inkling/Kimi gateway remains the streaming path"
                )
            if parameters.temperature != 0 or parameters.top_p != 1:
                raise ValueError(
                    "OLMoE one-shot adapter generation currently supports greedy decoding only"
                )
            payload = job.input_payload
            if not isinstance(payload, TokenPayload) or not payload.token_ids:
                raise ValueError("OLMoE one-shot generation requires explicit input token IDs")
            runner = ColibriReplayRunner(
                engine_directory=self.engine_directory,
                model_path=self.model_path,
                model_id=self.model_id,
                model_revision=self.model_revision,
                model_family=family,
                cap=self.cap or 16,
                ram_safety_reserve_bytes=self.ram_safety_reserve_bytes,
                timeout_seconds=max(1, job.remaining_deadline_ms / 1000),
                environment=self._one_shot_environment(),
            )
            report = self.probe.probe()
            execution = runner.generate_from_tokens(
                payload.token_ids,
                completion_tokens=parameters.max_new_tokens,
                candidate_id=str(job.metadata.get("candidate_id", "adapter_generate")),
                settings=dict(job.metadata.get("settings", {})),
                supported_settings=self.probe.supported_tuning_settings(
                    report, model_family=family
                ),
                route_trace_path=job.metadata.get("route_trace_path"),
                invocation_started_ns=invocation_started_ns,
            )
            if execution.return_code or execution.timed_out:
                raise RuntimeError(
                    f"Colibri OLMoE generation failed with exit {execution.return_code}: "
                    f"{execution.stderr[-2000:]}"
                )
            return self._result(
                job,
                output=TokenPayload(
                    token_ids=execution.output_token_ids,
                    tokenizer_hash=payload.tokenizer_hash,
                ),
                metrics={
                    "replay_execution": execution.model_dump(mode="json"),
                    "token_identity_observed": True,
                    "input_token_ids": payload.token_ids,
                    "output_token_ids": execution.output_token_ids,
                    "stop_reason": "length",
                },
            )
        process = self._ensure_process()
        prompt = self._prompt(job)
        chunks: list[str] = []
        if stream:
            generated = process.stream_generate(
                prompt=prompt,
                max_tokens=parameters.max_new_tokens,
                temperature=parameters.temperature,
                top_p=parameters.top_p,
                request_id=job.request_id,
                timeout_seconds=max(1, job.remaining_deadline_ms / 1000),
                on_text=chunks.append,
            )
        else:
            generated = process.generate(
                prompt=prompt,
                max_tokens=parameters.max_new_tokens,
                temperature=parameters.temperature,
                top_p=parameters.top_p,
                request_id=job.request_id,
                timeout_seconds=max(1, job.remaining_deadline_ms / 1000),
            )
        output = TokenPayload(
            token_ids=generated.output_token_ids or [],
            text=generated.text,
            tokenizer_hash=job.metadata.get("tokenizer_hash"),
        )
        metrics = generated.model_dump(mode="json")
        if stream:
            metrics["stream_chunks"] = chunks
        if generated.decode_tokens_per_second is not None:
            self._last_profile = WorkerBenchmarkProfile(
                model_revision=self.model_revision,
                decode_tokens_per_second=generated.decode_tokens_per_second,
                model_load_seconds=0,
                warmup_seconds=0,
            )
        return self._result(job, output=output, metrics=metrics)

    def _replay(self, job: WorkerJob) -> WorkerJobResult:
        replay = ReplayTokenSequence.model_validate(job.metadata["replay"])
        runner = ColibriReplayRunner(
            engine_directory=self.engine_directory,
            model_path=self.model_path,
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_family=self.model_family,
            cap=self.cap or 16,
            ram_safety_reserve_bytes=self.ram_safety_reserve_bytes,
            timeout_seconds=max(1, job.remaining_deadline_ms / 1000),
            environment=self._one_shot_environment(),
        )
        report = self.probe.probe()
        execution = runner.run(
            replay,
            candidate_id=str(job.metadata.get("candidate_id", "replay")),
            settings=dict(job.metadata.get("settings", {})),
            supported_settings=self.probe.supported_tuning_settings(
                report, model_family=runner.family
            ),
            route_trace_path=job.metadata.get("route_trace_path"),
        )
        if execution.return_code != 0 or execution.timed_out:
            raise RuntimeError(
                f"Colibri replay failed with exit {execution.return_code}: {execution.stderr[-2000:]}"
            )
        if execution.decode_tokens_per_second is None:
            raise RuntimeError("Colibri replay did not report decode throughput")
        return self._result(
            job,
            output=TokenPayload(
                token_ids=execution.output_token_ids,
                tokenizer_hash=replay.tokenizer_hash,
            ),
            metrics={"replay_execution": execution.model_dump(mode="json")},
        )

    def _profile(self, job: WorkerJob) -> WorkerJobResult:
        metrics: dict[str, Any] = {"benchmark_profile": self._last_profile.model_dump(mode="json")}
        if self.telemetry_path.is_file():
            events = ColibriTelemetryReader(self.telemetry_path).read()
            metrics["telemetry_events"] = len(events)
            metrics["telemetry_summary"] = ColibriTelemetryReader.summarize(events)
        usage_path = job.metadata.get("usage_history_path")
        if usage_path:
            metrics["usage_history"] = ColibriUsageHistoryReader().read(str(usage_path))
        return self._result(job, metrics=metrics)

    def _placement_evaluation(self, job: WorkerJob) -> WorkerJobResult:
        trace = Path(str(job.metadata["route_trace_path"]))
        selections = ColibriRouteTraceReader().read(trace)
        summary = ColibriRouteTraceReader.summarize(selections)
        requested = job.metadata.get("policy")
        if not isinstance(requested, dict):
            raise ValueError("placement evaluation requires a concrete policy object")
        return self._result(
            job,
            metrics={
                "policy": requested,
                "route_summary": summary,
                "measured_execution": False,
                "detail": "offline trace evaluation only; no performance value is fabricated",
            },
        )

    def _inventory(self) -> tuple[Any, Any, Any, Any]:
        if self._inventory_cache is None:
            report = self.probe.probe()
            self._inventory_cache = self.inspector.inspect(
                self.model_path,
                model_id=self.model_id,
                model_revision=self.model_revision,
                model_family=self.model_family,
                execution_backends=report.execution_backends,
            )
        return self._inventory_cache

    def _ensure_process(self) -> ColibriProcess:
        if self.process is None:
            self.process = ColibriProcess(
                engine_directory=self.engine_directory,
                model_path=self.model_path,
                model_id=self.model_id,
                model_revision=self.model_revision,
                model_family=self.model_family,
                mode=self.mode,
                telemetry_level=self.telemetry_level,
                telemetry_path=self.telemetry_path,
                log_directory=self.log_directory,
                cap=self.cap,
                environment=self.environment,
                ram_safety_reserve_bytes=self.ram_safety_reserve_bytes,
            )
            self.process.start()
        return self.process

    def _config(self) -> dict[str, Any]:
        value = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        nested = value.get("text_config") if isinstance(value, dict) else None
        return nested if isinstance(nested, dict) else value

    def _one_shot_environment(self) -> dict[str, str]:
        environment = dict(self.environment)
        if self.mode == ColibriMode.BRIDGE:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            environment.update(
                {
                    "COLI_SWARM_BRIDGE": "1",
                    "COLI_SWARM_BRIDGE_PATH": str(self.telemetry_path),
                    "COLI_SWARM_TELEMETRY": self.telemetry_level.value,
                    "COLI_MODEL_ID": self.model_id,
                    "COLI_MODEL_REVISION": self.model_revision,
                }
            )
        return environment

    @staticmethod
    def _prompt(job: WorkerJob) -> str:
        payload = job.input_payload
        if not isinstance(payload, TokenPayload):
            raise ValueError("Colibri generation accepts token/text payloads, not tensor payloads")
        if payload.text is None:
            raise ValueError(
                "the stock Colibri gateway accepts text prompts; token-ID input is available through replay"
            )
        return payload.text

    @staticmethod
    def _native_plan(metadata: dict[str, Any]) -> dict[str, Any]:
        if isinstance(metadata.get("colibri_plan"), dict):
            return dict(metadata["colibri_plan"])
        path_value = metadata.get("colibri_plan_path")
        if path_value is None:
            raise ValueError("resource_plan job requires colibri_plan or colibri_plan_path")
        value = json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Colibri resource plan must be a JSON object")
        return value

    @staticmethod
    def _classification() -> ResultClassification:
        return ResultClassification.MEASURED_X86_CPU

    def _result(
        self,
        job: WorkerJob,
        status: WorkerJobStatus = WorkerJobStatus.ACCEPTED,
        *,
        output: TokenPayload | None = None,
        detail: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> WorkerJobResult:
        recorded_metrics = dict(metrics or {})
        if self.execution_profile_id is not None:
            recorded_metrics.update(
                {
                    "execution_profile_id": self.execution_profile_id,
                    "execution_profile_fingerprint": self.execution_profile_fingerprint,
                    "routing_aware_placement": True,
                }
            )
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=status,
            output_payload=output,
            detail=detail,
            metrics=recorded_metrics,
            classification=self._classification() if status == WorkerJobStatus.ACCEPTED else None,
        )
