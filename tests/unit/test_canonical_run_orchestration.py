from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm_inference.cluster.models import ArtifactFile, ArtifactManifest
from swarm_inference.cluster.orchestrator import ClusterOrchestrator
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionPlan,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
    WorkerExecutionCapability,
)
from swarm_inference.engines.registry import ExecutionEngineRegistry
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.model.resolver import ModelResolution
from swarm_inference.security.identity import WorkerIdentity


class _Resolver:
    def __init__(self, descriptor: ResolvedModelDescriptor, acquired_path: Path) -> None:
        self.descriptor = descriptor
        self.acquired_path = acquired_path
        self.acquire_count = 0

    def inspect(self, *_args: Any, **_kwargs: Any) -> ModelResolution:
        return ModelResolution(descriptor=self.descriptor)

    async def acquire_async(
        self, _descriptor: ResolvedModelDescriptor
    ) -> tuple[Path, ...]:
        self.acquire_count += 1
        return (self.acquired_path,)


class _Engine:
    engine_id = "general-test-engine"

    def __init__(self) -> None:
        self.bound_paths: tuple[Path, ...] = ()
        self.prepare_count = 0
        self.submit_count = 0
        self.unload_count = 0

    def probe(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        workers = cluster.workers_for_engine(self.engine_id)
        if model.format != "gguf":
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_FORMAT,
                reason="test engine consumes GGUF",
            )
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason="immutable GGUF and pinned runtime are supported",
            supported_worker_ids=tuple(item.worker_id for item in workers),
        )

    async def candidate_plans(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> list[ExecutionPlan]:
        worker = cluster.workers_for_engine(self.engine_id)[0]
        roles = {worker.worker_id: "critical_path_stage"}
        return [
            ExecutionPlan(
                plan_id="plan-general",
                engine_id=self.engine_id,
                model_fingerprint=model.content_fingerprint,
                execution_identity="sha256:" + "c" * 64,
                objective=request.objective,
                topology="complete-model",
                worker_roles=roles,
                prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                predicted_ttft_ms=10,
                predicted_decode_tokens_s=20,
                predicted_aggregate_tokens_s=20,
                required_memory_bytes=model.weight_bytes,
                score=20,
            )
        ]

    def bind_acquired_model(
        self,
        _descriptor: ResolvedModelDescriptor,
        paths: tuple[Path, ...],
    ) -> None:
        self.bound_paths = paths

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        self.prepare_count += 1
        return Deployment(
            deployment_id="deployment-general",
            engine_id=self.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
        )

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        assert deployment.ready
        self.submit_count += 1
        yield InferenceEvent(
            event_type="started",
            request_id=request.request_id,
            sequence_number=0,
        )
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=1,
            token_id=42,
            text="answer",
        )
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=2,
            telemetry={
                "decode_tokens_per_second": 19.5,
                "cache_hits": 3,
                "cache_misses": 1,
                "exactness_verified": True,
            },
        )

    async def unload(self, _deployment: Deployment) -> None:
        self.unload_count += 1


class _Client:
    def __init__(self) -> None:
        capability = SimpleNamespace(
            worker_id="node-a/worker",
            supported_activation_dtypes=["float32"],
            control_endpoint="127.0.0.1:50000",
            endpoint="127.0.0.1:50000",
        )
        self.catalog = SimpleNamespace(
            workers=[
                SimpleNamespace(
                    healthy_registration=True,
                    capability=capability,
                    control_endpoint="127.0.0.1:50000",
                )
            ]
        )
        self.closed = False

    async def workers(self, _request: Any) -> Any:
        return self.catalog

    async def close(self) -> None:
        self.closed = True


class _State:
    def __init__(self, root: Path) -> None:
        self.paths = SimpleNamespace(
            artifacts=root / "artifacts",
            logs=root / "logs",
        )
        self.identity = WorkerIdentity.generate()

    def load_cluster(self) -> Any:
        return SimpleNamespace(coordinator_endpoint="coordinator:1")

    def load_or_create_node_identity(self) -> WorkerIdentity:
        return self.identity


class _WorkerLifecycle:
    def __init__(self) -> None:
        self.prepare_count = 0
        self.submit_count = 0
        self.unload_count = 0

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        self.prepare_count += 1
        return Deployment(
            deployment_id="worker-deployment-general",
            engine_id=plan.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
        )

    async def submit(
        self,
        _deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        self.submit_count += 1
        yield InferenceEvent(
            event_type="started",
            request_id=request.request_id,
            sequence_number=0,
        )
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=1,
            token_id=42,
            text="answer",
        )
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=2,
            telemetry={
                "decode_tokens_per_second": 19.5,
                "cache_hits": 3,
                "cache_misses": 1,
                "exactness_verified": True,
            },
        )

    async def unload(self, _deployment: Deployment) -> None:
        self.unload_count += 1


class _RecoveringWorkerLifecycle(_WorkerLifecycle):
    def __init__(self, *, divergent_replay: bool = False) -> None:
        super().__init__()
        self.divergent_replay = divergent_replay

    async def submit(
        self,
        _deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        self.submit_count += 1
        yield InferenceEvent(
            event_type="started",
            request_id=request.request_id,
            sequence_number=0,
        )
        if self.submit_count == 1:
            yield InferenceEvent(
                event_type="token",
                request_id=request.request_id,
                sequence_number=1,
                token_id=42,
                text="first ",
            )
            raise ConnectionError("injected engine process loss")
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=1,
            token_id=99 if self.divergent_replay else 42,
            text="duplicate ",
        )
        yield InferenceEvent(
            event_type="token",
            request_id=request.request_id,
            sequence_number=2,
            token_id=43,
            text="second",
        )
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=3,
        )


class _HarnessOrchestrator(ClusterOrchestrator):
    def __init__(self, *, capabilities: ClusterCapabilities, **kwargs: Any) -> None:
        self._test_capabilities = capabilities
        super().__init__(**kwargs)

    async def _wait_for_fresh_links(self, _client: Any, _workers: Any) -> None:
        return None

    async def _execution_capabilities(
        self, _client: Any, _workers: Any
    ) -> ClusterCapabilities:
        return self._test_capabilities

    async def _build_model_artifact(
        self,
        descriptor: ResolvedModelDescriptor,
        *,
        engine_id: str,
        node_id: str,
    ) -> tuple[Any, ArtifactManifest]:
        del node_id
        file = descriptor.files[0]
        manifest = ArtifactManifest(
            artifact_id="9" * 64,
            model_id=descriptor.model_id,
            model_revision=descriptor.revision,
            model_fingerprint=descriptor.content_fingerprint,
            tokenizer_revision=descriptor.tokenizer_identity,
            engine_id=engine_id,
            model_format=descriptor.format,
            artifact_kind="gguf",
            content_hash="9" * 64,
            files=[
                ArtifactFile(
                    relative_path=file.relative_path,
                    size_bytes=file.size_bytes,
                    sha256=file.sha256 or "8" * 64,
                )
            ],
            total_size_bytes=file.size_bytes,
            total_bytes=file.size_bytes,
        )
        return SimpleNamespace(), manifest


def _cluster() -> ClusterCapabilities:
    capability = ExecutionEngineCapability(
        engine_id="general-test-engine",
        enabled=True,
        runtime_revision="runtime-pinned",
        binary_hashes={"server": "sha256:" + "d" * 64},
        formats=("gguf",),
        devices=(
            ExecutionDevice(
                device_id="cpu",
                device_type="cpu",
                name="test CPU",
                total_memory_bytes=2**30,
                usable_memory_bytes=2**30,
                measured_decode_tokens_s=20,
            ),
        ),
        roles=("complete-model", "idle"),
    )
    return ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(
                worker_id="node-a/worker",
                node_id="node-a",
                engines=(capability,),
            ),
        )
    )


def _descriptor(path: Path) -> ResolvedModelDescriptor:
    return ResolvedModelDescriptor(
        model_id=str(path),
        revision="sha256:" + "a" * 64,
        content_fingerprint="sha256:" + "b" * 64,
        source_type="local",
        format="gguf",
        architecture="test",
        files=(
            ModelFileDescriptor(
                relative_path=path.name,
                size_bytes=path.stat().st_size,
                sha256="e" * 64,
            ),
        ),
        quantization="Q4_K_M",
        weight_bytes=path.stat().st_size,
        tokenizer_identity="sha256:" + "f" * 64,
        local_paths=(str(path),),
    )


@pytest.mark.asyncio
async def test_generic_dry_run_does_not_acquire_selected_weights(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    resolver = _Resolver(_descriptor(model), model)
    engine = _Engine()
    client = _Client()
    orchestrator = _HarnessOrchestrator(
        capabilities=_cluster(),
        state=_State(tmp_path),  # type: ignore[arg-type]
        client_factory=lambda _endpoint: client,
        model_source_resolver=resolver,  # type: ignore[arg-type]
        engine_registry=ExecutionEngineRegistry((engine,)),
    )

    summary = await orchestrator.run(
        model_id=str(model),
        prompt="safe",
        dry_run=True,
    )

    assert summary.status == "dry-run"
    assert summary.engine_id == engine.engine_id
    assert summary.model_fingerprint == "sha256:" + "b" * 64
    assert resolver.acquire_count == 0
    assert engine.prepare_count == 0
    assert client.closed


@pytest.mark.asyncio
async def test_selected_general_engine_is_prepared_and_submitted(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    resolver = _Resolver(_descriptor(model), model)
    engine = _Engine()
    lifecycle = _WorkerLifecycle()
    client = _Client()
    streamed: list[Any] = []
    orchestrator = _HarnessOrchestrator(
        capabilities=_cluster(),
        state=_State(tmp_path),  # type: ignore[arg-type]
        client_factory=lambda _endpoint: client,
        model_source_resolver=resolver,  # type: ignore[arg-type]
        engine_registry=ExecutionEngineRegistry((engine,)),
        worker_engine_lifecycle_factory=lambda **_kwargs: lifecycle,
        stream_sink=streamed.append,
    )

    summary = await orchestrator.run(
        model_id=str(model),
        prompt="safe",
        max_new_tokens=1,
    )

    assert summary.status == "completed"
    assert summary.output_token_ids == [42]
    assert summary.decoded_text == "answer"
    assert resolver.acquire_count == 1
    assert engine.bound_paths == (model.resolve(),)
    assert engine.prepare_count == 0
    assert engine.submit_count == 0
    assert engine.unload_count == 0
    assert lifecycle.prepare_count == 1
    assert lifecycle.submit_count == 1
    assert lifecycle.unload_count == 0
    assert [item.token_id for item in streamed] == [42]
    assert summary.engine_revision == "runtime-pinned"
    assert summary.execution_identity == "sha256:" + "c" * 64
    assert summary.telemetry is not None
    assert summary.telemetry.model_fingerprint == "sha256:" + "b" * 64
    assert summary.telemetry.worker_roles == {"node-a/worker": "critical_path_stage"}
    assert summary.telemetry.generated_tokens == 1
    assert summary.telemetry.cache_hits == 3
    assert summary.telemetry.cache_misses == 1
    assert summary.telemetry.exactness_verified
    assert summary.telemetry.metric_sources["ttft_ms"] == "client-observed"
    event_rows = [
        json.loads(line)
        for line in (tmp_path / "logs" / "product-events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(event_rows) == 1
    assert event_rows[0]["event_type"] == "inference_recorded"
    assert event_rows[0]["execution_identity"] == "sha256:" + "c" * 64
    assert event_rows[0]["engine_metrics"]["cache_hits"] == 3
    assert client.closed


@pytest.mark.asyncio
async def test_general_engine_restart_replays_and_suppresses_duplicate_tokens(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    resolver = _Resolver(_descriptor(model), model)
    engine = _Engine()
    lifecycle = _RecoveringWorkerLifecycle()
    client = _Client()
    streamed: list[Any] = []
    orchestrator = _HarnessOrchestrator(
        capabilities=_cluster(),
        state=_State(tmp_path),  # type: ignore[arg-type]
        client_factory=lambda _endpoint: client,
        model_source_resolver=resolver,  # type: ignore[arg-type]
        engine_registry=ExecutionEngineRegistry((engine,)),
        worker_engine_lifecycle_factory=lambda **_kwargs: lifecycle,
        stream_sink=streamed.append,
    )

    summary = await orchestrator.run(
        model_id=str(model),
        prompt="safe",
        max_new_tokens=2,
    )

    assert summary.status == "completed"
    assert summary.output_token_ids == [42, 43]
    assert summary.decoded_text == "first second"
    assert [item.token_id for item in streamed] == [42, 43]
    assert lifecycle.prepare_count == 2
    assert lifecycle.submit_count == 2
    assert lifecycle.unload_count == 1
    assert summary.telemetry is not None
    assert summary.telemetry.recoveries == 1
    assert summary.telemetry.engine_metrics["restart_replay"] == {
        "execution_identity_preserved": True,
        "recovery_count": 1,
        "verified_prefix_tokens": 1,
    }


@pytest.mark.asyncio
async def test_general_engine_replay_divergence_fails_closed(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    resolver = _Resolver(_descriptor(model), model)
    lifecycle = _RecoveringWorkerLifecycle(divergent_replay=True)
    client = _Client()
    streamed: list[Any] = []
    orchestrator = _HarnessOrchestrator(
        capabilities=_cluster(),
        state=_State(tmp_path),  # type: ignore[arg-type]
        client_factory=lambda _endpoint: client,
        model_source_resolver=resolver,  # type: ignore[arg-type]
        engine_registry=ExecutionEngineRegistry((_Engine(),)),
        worker_engine_lifecycle_factory=lambda **_kwargs: lifecycle,
        stream_sink=streamed.append,
    )

    summary = await orchestrator.run(
        model_id=str(model),
        prompt="safe",
        max_new_tokens=2,
    )

    assert summary.status == "failed"
    assert "exact replay failed closed" in summary.detail
    assert summary.output_token_ids == [42]
    assert [item.token_id for item in streamed] == [42]
    assert lifecycle.prepare_count == 2
    assert lifecycle.submit_count == 2
    assert lifecycle.unload_count == 2
    assert summary.telemetry is not None
    assert summary.telemetry.recoveries == 1
    assert summary.telemetry.engine_metrics["restart_replay"]["verified_prefix_tokens"] == 0
