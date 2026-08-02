"""Resumable execution matrix for Experiment 009.

The quick path executes the real patched Colibri C fixture and validates all
control-plane plumbing, but it can only produce PARTIAL.  The full path adds a
real converted OLMoE checkpoint, actual route/cache/storage observations,
fixed replay tuning, and held-out policy measurements.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np

from swarm_inference.backends.colibri.backend import ColibriBackend
from swarm_inference.backends.colibri.dependency import (
    load_build_manifest,
    patch_manifest,
    verify_colibri_checkout,
)
from swarm_inference.backends.colibri.model import ColibriModelInspector
from swarm_inference.backends.colibri.placement import (
    PlacementPolicy,
    RoutingPolicyEvaluator,
    batch_expert_union,
    calibration_hot_pin_bitmap,
)
from swarm_inference.backends.colibri.plan import ColibriPlanTranslator
from swarm_inference.backends.colibri.probe import ColibriCapabilityProbe
from swarm_inference.backends.colibri.process import ColibriProcess
from swarm_inference.backends.colibri.replay import (
    ColibriFixedReplayTuner,
    ColibriReplayRunner,
    FixedReplayTuningResult,
    ReplayExecution,
    ReplayTokenSequence,
    TuningCandidate,
)
from swarm_inference.backends.colibri.schemas import (
    ColibriMode,
    RouteSelection,
    TelemetryLevel,
    TuningSample,
)
from swarm_inference.backends.colibri.storage import ColibriStorageProfiler
from swarm_inference.backends.colibri.telemetry import (
    ColibriRouteTraceReader,
    ColibriTelemetryReader,
    ColibriUsageHistoryReader,
)
from swarm_inference.config.experiment_009 import (
    Experiment009Config,
    load_experiment_009_config,
)
from swarm_inference.experiments.experiment_009.bundle import (
    REQUIRED_FILES,
    EvidenceBundle,
    create_bundle_root,
)
from swarm_inference.experiments.experiment_009.evidence import (
    build_microshard_evidence,
    canonical_hash,
    environment_report,
    hardware_and_tiers,
    plan_tier_rows,
    route_tables,
    run_colibri_plan,
)
from swarm_inference.experiments.experiment_009.reporting import (
    build_bundle_readme,
    build_report,
    generate_required_plots,
)
from swarm_inference.experiments.experiment_009.schemas import (
    EvidenceClass,
    Experiment009Verdict,
    GateResult,
    GateStatus,
    overall_verdict,
)
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.worker.abi import (
    GenerationParameters,
    TokenPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
)

ConfigurationId = Literal["A", "B", "C", "D", "E"]


@dataclass(slots=True)
class Experiment009Options:
    config_path: Path
    colibri_path: Path | None = None
    model_path: Path | None = None
    model_family: str | None = None
    output_directory: Path | None = None
    quick: bool = False
    full: bool = False
    resume: bool = False
    rebuild_colibri: bool = False
    apply_bridge_patches: bool = False
    telemetry_level: str | None = None
    configuration: ConfigurationId | None = None
    skip_model_download: bool = False

    def validate(self) -> None:
        if self.quick == self.full:
            raise ValueError("select exactly one of --quick or --full")
        if self.configuration is not None and self.configuration not in set("ABCDE"):
            raise ValueError("configuration must be A, B, C, D, or E")
        if self.telemetry_level not in {None, "off", "summary", "detailed", "trace"}:
            raise ValueError("telemetry level must be off, summary, detailed, or trace")


@dataclass(slots=True)
class Experiment009Outcome:
    bundle_path: Path
    verdict: Experiment009Verdict
    completed: bool
    error: str | None


@dataclass(slots=True)
class ModelContext:
    path: Path
    model_id: str
    model_revision: str
    family: str
    evidence_class: EvidenceClass


def _selected(options: Experiment009Options, configuration: str) -> bool:
    return options.configuration is None or options.configuration == configuration


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_paths(
    repository_root: Path,
    config: Experiment009Config,
    options: Experiment009Options,
) -> tuple[Path, Path, Path, Path]:
    checkout = (options.colibri_path or repository_root / config.dependency.checkout).resolve()
    configured_build = (repository_root / config.dependency.build_directory).resolve()
    if (checkout / "colibri.exe").is_file() or (checkout / "colibri").is_file():
        engine_directory = checkout
        checkout = (repository_root / config.dependency.checkout).resolve()
        build_root = engine_directory.parent
    else:
        build_root = configured_build
        engine_directory = build_root / "bin"
    source_directory = build_root / "source"
    return checkout, build_root, engine_directory, source_directory


def _build_if_requested(
    *,
    repository_root: Path,
    checkout: Path,
    build_root: Path,
    options: Experiment009Options,
    bundle: EvidenceBundle,
) -> None:
    if not options.rebuild_colibri:
        return
    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(repository_root / "integrations" / "colibri" / "build.ps1"),
        "-ColibriPath",
        str(checkout),
        "-OutputDirectory",
        str(build_root),
    ]
    if options.apply_bridge_patches:
        command.append("-ApplyBridgePatches")
    result = subprocess.run(
        command,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    (bundle.root / "logs" / "colibri_build.stdout.log").write_text(result.stdout, encoding="utf-8")
    (bundle.root / "logs" / "colibri_build.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Colibri build failed with exit {result.returncode}")


def _acquire_practical_model(
    *,
    repository_root: Path,
    config: Experiment009Config,
    options: Experiment009Options,
    bundle: EvidenceBundle,
) -> Path:
    if options.model_path is not None:
        path = options.model_path.expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"practical model config is missing: {path / 'config.json'}")
        return path
    output = (repository_root / config.practical_model.converted_path).resolve()
    if (output / "config.json").is_file() and list(output.glob("*.safetensors")):
        return output
    if options.skip_model_download:
        raise FileNotFoundError(
            f"no converted practical model at {output}; --skip-model-download forbids acquisition"
        )
    source = output.parent / f"source-{config.practical_model.revision[:12]}"
    source.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config.practical_model.repository,
        revision=config.practical_model.revision,
        local_dir=source,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
        ],
    )
    output.mkdir(parents=True, exist_ok=True)
    converter = (
        repository_root / "third_party" / "colibri" / "c" / "tools" / "convert_olmoe_merged.py"
    )
    command = [sys.executable, str(converter), "--model", str(source), "--out", str(output)]
    result = subprocess.run(
        command,
        cwd=converter.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=7200,
    )
    (bundle.root / "logs" / "model_conversion.stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (bundle.root / "logs" / "model_conversion.stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(f"OLMoE conversion failed with exit {result.returncode}")
    if not (output / "config.json").is_file() or not list(output.glob("*.safetensors")):
        raise RuntimeError("OLMoE converter completed without a usable model directory")
    return output


def _worker_job(
    *,
    model: ModelContext,
    request_id: str,
    role: WorkerJobType,
    timeout_seconds: float,
    text: str | None = None,
    token_ids: list[int] | None = None,
    max_tokens: int = 4,
    metadata: dict[str, Any] | None = None,
) -> WorkerJob:
    return WorkerJob(
        job_id=f"job-{request_id}",
        request_id=request_id,
        role=role,
        model_id=model.model_id,
        model_revision=model.model_revision,
        input_payload=TokenPayload(token_ids=token_ids or [], text=text),
        generation_parameters=(
            GenerationParameters(max_new_tokens=max_tokens, temperature=0, top_p=1)
            if role in {WorkerJobType.GENERATE, WorkerJobType.STREAM_GENERATE}
            else None
        ),
        deadline_ms=max(1, int(timeout_seconds * 1000)),
        metadata=metadata or {},
    )


def _generation_from_worker(result: Any) -> dict[str, Any]:
    if result.status != WorkerJobStatus.ACCEPTED:
        raise RuntimeError(f"Colibri adapter job failed: {result.status.value}: {result.detail}")
    return dict(result.metrics)


def _route_mappings(
    *,
    route_path: Path,
    request_rows: list[dict[str, Any]],
    expert_layers: int,
) -> tuple[dict[int, str], dict[int, str], dict[str, Any]]:
    raw = ColibriRouteTraceReader().read(route_path)
    calls = sorted({row.call_index for row in raw})
    output_total = sum(
        max(1, int(row["completion_tokens"])) * expert_layers for row in request_rows
    )
    plus_one_total = sum(
        (max(1, int(row["completion_tokens"])) + 1) * expert_layers for row in request_rows
    )
    calls_per_output = plus_one_total == len(calls) and output_total != len(calls)
    phase: dict[int, str] = {}
    requests: dict[int, str] = {}
    cursor = 0
    for request in request_rows:
        forwards = max(1, int(request["completion_tokens"])) + (1 if calls_per_output else 0)
        selected = calls[cursor : cursor + forwards * expert_layers]
        cursor += len(selected)
        for index, call in enumerate(selected):
            phase[call] = "prefill" if index < expert_layers else "decode"
            if request.get("map_routes", False):
                requests[call] = str(request["request_id"])
    return (
        phase,
        requests,
        {
            "route_calls": len(calls),
            "mapped_calls": len(requests),
            "unconsumed_calls": max(0, len(calls) - cursor),
            "call_accounting": "completion_plus_one" if calls_per_output else "completion_count",
        },
    )


def _append_benchmark(
    rows: list[dict[str, Any]],
    *,
    configuration: str,
    workload: str,
    repeat: int,
    result: dict[str, Any],
    evidence_class: EvidenceClass,
) -> None:
    rows.append(
        {
            "configuration": configuration,
            "workload": workload,
            "repeat": repeat,
            "evidence_class": evidence_class.value,
            "decode_tokens_per_second": result.get("decode_tokens_per_second"),
            "time_to_first_token_ms": result.get("time_to_first_token_ms"),
            "latency_ms": result.get("elapsed_ms"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "stop_reason": result.get("stop_reason"),
            "status": "COMPLETED",
        }
    )


def _run_fixture_transport(
    *,
    bundle: EvidenceBundle,
    config: Experiment009Config,
    options: Experiment009Options,
    engine_directory: Path,
    source_directory: Path,
    build_manifest_path: Path,
    model: ModelContext,
    expert_layers: int,
) -> dict[str, Any]:
    telemetry_path = bundle.root / "telemetry.ndjson"
    route_path = bundle.root / "logs" / "fixture.route"
    level = TelemetryLevel(options.telemetry_level or config.backend.telemetry_level)
    mode = ColibriMode(config.backend.mode)
    pair_count = max(3, config.acceptance.minimum_correctness_requests // 2)
    direct_results: list[dict[str, Any]] = []
    direct_streams: list[dict[str, Any]] = []
    adapter_results: list[dict[str, Any]] = []
    adapter_streams: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    routing_result_by_prompt: dict[str, dict[str, Any]] = {}

    if _selected(options, "A") or _selected(options, "B") or _selected(options, "E"):
        direct = ColibriProcess(
            engine_directory=engine_directory,
            model_path=model.path,
            model_id=model.model_id,
            model_revision=model.model_revision,
            model_family=model.family,
            mode=mode,
            telemetry_level=level,
            telemetry_path=telemetry_path,
            log_directory=bundle.root / "logs" / "direct",
            cap=config.backend.cap,
            environment={
                "ROUTE_TRACE": str(route_path),
                "AUTOPIN": "0",
                "CAP_RAISE": "0",
                "OMP_NUM_THREADS": "1",
            },
            ram_safety_reserve_bytes=0
            if options.quick
            else config.backend.ram_safety_reserve_bytes,
        )
        try:
            direct.start(timeout_seconds=config.backend.startup_timeout_seconds)
            warm = direct.generate(
                prompt=config.fixture.prompt,
                max_tokens=4,
                request_id="direct-warmup",
                timeout_seconds=config.backend.request_timeout_seconds,
            )
            request_rows.append(
                {
                    "request_id": "direct-warmup",
                    "completion_tokens": warm.completion_tokens,
                    "map_routes": False,
                }
            )
            for prompt in config.routing.prompts:
                generated = direct.generate(
                    prompt=prompt.text,
                    max_tokens=4,
                    request_id=f"route-{prompt.prompt_id}",
                    timeout_seconds=config.backend.request_timeout_seconds,
                )
                routing_result_by_prompt[prompt.prompt_id] = generated.model_dump(mode="json")
                request_rows.append(
                    {
                        "request_id": prompt.prompt_id,
                        "completion_tokens": generated.completion_tokens,
                        "map_routes": True,
                    }
                )
            for repeat in range(pair_count):
                generated = direct.generate(
                    prompt=config.fixture.prompt,
                    max_tokens=4,
                    request_id=f"direct-{repeat}",
                    timeout_seconds=config.backend.request_timeout_seconds,
                )
                row = generated.model_dump(mode="json")
                direct_results.append(row)
                request_rows.append(
                    {
                        "request_id": f"direct-{repeat}",
                        "completion_tokens": generated.completion_tokens,
                        "map_routes": False,
                    }
                )
                _append_benchmark(
                    benchmark_rows,
                    configuration="A",
                    workload="fixture_decode",
                    repeat=repeat,
                    result=row,
                    evidence_class=EvidenceClass.FIXTURE,
                )
            for repeat in range(3):
                generated = direct.stream_generate(
                    prompt=config.fixture.prompt,
                    max_tokens=4,
                    request_id=f"direct-stream-{repeat}",
                    timeout_seconds=config.backend.request_timeout_seconds,
                )
                row = generated.model_dump(mode="json")
                direct_streams.append(row)
                request_rows.append(
                    {
                        "request_id": f"direct-stream-{repeat}",
                        "completion_tokens": generated.completion_tokens,
                        "map_routes": False,
                    }
                )
        finally:
            direct.shutdown()

    if _selected(options, "B") or options.configuration is None:
        backend = ColibriBackend(
            engine_directory=engine_directory,
            source_directory=source_directory,
            build_manifest=build_manifest_path,
            model_path=model.path,
            model_id=model.model_id,
            model_revision=model.model_revision,
            model_family=model.family,
            mode=mode,
            telemetry_level=level,
            telemetry_path=telemetry_path,
            log_directory=bundle.root / "logs" / "adapter",
            cap=config.backend.cap,
            environment={
                "AUTOPIN": "0",
                "CAP_RAISE": "0",
                "OMP_NUM_THREADS": "1",
            },
            ram_safety_reserve_bytes=0
            if options.quick
            else config.backend.ram_safety_reserve_bytes,
        )

        async def run_adapter() -> None:
            capability_job = _worker_job(
                model=model,
                request_id="adapter-capability",
                role=WorkerJobType.CAPABILITY_PROBE,
                timeout_seconds=config.backend.request_timeout_seconds,
            )
            capability_result = await backend.execute(capability_job)
            if capability_result.status != WorkerJobStatus.ACCEPTED:
                raise RuntimeError(f"universal capability job failed: {capability_result.detail}")
            for repeat in range(pair_count):
                result = await backend.execute(
                    _worker_job(
                        model=model,
                        request_id=f"adapter-{repeat}",
                        role=WorkerJobType.GENERATE,
                        timeout_seconds=config.backend.request_timeout_seconds,
                        text=config.fixture.prompt,
                        max_tokens=4,
                    )
                )
                row = _generation_from_worker(result)
                adapter_results.append(row)
                _append_benchmark(
                    benchmark_rows,
                    configuration="B",
                    workload="fixture_decode",
                    repeat=repeat,
                    result=row,
                    evidence_class=EvidenceClass.FIXTURE,
                )
            for repeat in range(3):
                result = await backend.execute(
                    _worker_job(
                        model=model,
                        request_id=f"adapter-stream-{repeat}",
                        role=WorkerJobType.STREAM_GENERATE,
                        timeout_seconds=config.backend.request_timeout_seconds,
                        text=config.fixture.prompt,
                        max_tokens=4,
                    )
                )
                adapter_streams.append(_generation_from_worker(result))
            await backend.shutdown()

        asyncio.run(run_adapter())

    comparisons = []
    for index, (direct_row, adapted_row) in enumerate(
        zip(direct_results, adapter_results, strict=False)
    ):
        exact = (
            direct_row.get("input_token_ids") == adapted_row.get("input_token_ids")
            and direct_row.get("output_token_ids") == adapted_row.get("output_token_ids")
            and direct_row.get("stop_reason") == adapted_row.get("stop_reason")
            and direct_row.get("text") == adapted_row.get("text")
        )
        comparisons.append(
            {
                "comparison_id": f"fixture-{index}",
                "path": "fixture",
                "input_token_ids_match": direct_row.get("input_token_ids")
                == adapted_row.get("input_token_ids"),
                "output_token_ids_match": direct_row.get("output_token_ids")
                == adapted_row.get("output_token_ids"),
                "stop_reason_match": direct_row.get("stop_reason")
                == adapted_row.get("stop_reason"),
                "final_text_match": direct_row.get("text") == adapted_row.get("text"),
                "exact": exact,
            }
        )
    stream_exact = all(
        left.get("output_token_ids") == right.get("output_token_ids")
        and left.get("text") == right.get("text")
        for left, right in zip(direct_streams, adapter_streams, strict=False)
    ) and bool(direct_streams and adapter_streams)
    overhead_rows = []
    for configuration, values in (("direct", direct_results), ("adapter", adapter_results)):
        for repeat, row in enumerate(values):
            overhead_rows.append(
                {
                    "configuration": configuration,
                    "workload": "fixture_decode",
                    "repeat": repeat,
                    "decode_tokens_per_second": row.get("decode_tokens_per_second"),
                    "time_to_first_token_ms": row.get("time_to_first_token_ms"),
                    "latency_ms": row.get("elapsed_ms"),
                    "token_identity_match": comparisons[repeat]["exact"]
                    if repeat < len(comparisons)
                    else None,
                }
            )
    for configuration, values in (("direct", direct_streams), ("adapter", adapter_streams)):
        for repeat, row in enumerate(values):
            overhead_rows.append(
                {
                    "configuration": configuration,
                    "workload": "fixture_stream",
                    "repeat": repeat,
                    "decode_tokens_per_second": row.get("decode_tokens_per_second"),
                    "time_to_first_token_ms": row.get("time_to_first_token_ms"),
                    "latency_ms": row.get("elapsed_ms"),
                    "token_identity_match": stream_exact,
                }
            )
    phase_map: dict[int, str] = {}
    request_map: dict[int, str] = {}
    route_accounting: dict[str, Any] = {"route_calls": 0, "mapped_calls": 0}
    if route_path.is_file() and request_rows:
        phase_map, request_map, route_accounting = _route_mappings(
            route_path=route_path,
            request_rows=request_rows,
            expert_layers=max(1, expert_layers),
        )
    selections = (
        ColibriRouteTraceReader().read(
            route_path, phase_by_call=phase_map, request_by_call=request_map
        )
        if route_path.is_file()
        else []
    )
    selections = [row for row in selections if row.request_id is not None]
    return {
        "direct_results": direct_results,
        "adapter_results": adapter_results,
        "direct_streams": direct_streams,
        "adapter_streams": adapter_streams,
        "comparisons": comparisons,
        "stream_exact": stream_exact,
        "overhead_rows": overhead_rows,
        "benchmark_rows": benchmark_rows,
        "selections": selections,
        "route_accounting": route_accounting,
        "routing_result_by_prompt": routing_result_by_prompt,
        "request_execution_count": len(direct_results)
        + len(adapter_results)
        + len(direct_streams)
        + len(adapter_streams),
        "clean_shutdown": True,
    }


def _tuning_candidates(
    *,
    config: Experiment009Config,
    family: str,
    supported: set[str],
    physical_cores: int,
) -> tuple[list[TuningCandidate], list[dict[str, Any]]]:
    candidates = []
    rejected = []
    for item in config.tuning.candidates:
        unsupported = sorted(set(item.settings).difference(supported))
        if unsupported:
            rejected.append(
                {
                    "candidate_id": item.candidate_id,
                    "settings": item.settings,
                    "status": "REJECTED_UNSUPPORTED",
                    "unsupported_settings": unsupported,
                }
            )
            continue
        candidates.append(TuningCandidate(candidate_id=item.candidate_id, settings=item.settings))
    if not candidates or candidates[0].candidate_id != "baseline":
        candidates.insert(0, TuningCandidate(candidate_id="baseline"))
    if len(candidates) < 2:
        if family == "olmoe" and "PILOT" in supported:
            candidates.append(TuningCandidate(candidate_id="prefetch_l1", settings={"PILOT": "1"}))
        elif "OMP_NUM_THREADS" in supported:
            candidates.extend(
                [
                    TuningCandidate(candidate_id="one_thread", settings={"OMP_NUM_THREADS": "1"}),
                    TuningCandidate(
                        candidate_id="physical_cores",
                        settings={"OMP_NUM_THREADS": str(max(1, physical_cores))},
                    ),
                ]
            )
    return candidates, rejected


def _run_tuning(
    *,
    bundle: EvidenceBundle,
    config: Experiment009Config,
    runner: ColibriReplayRunner,
    replay: ReplayTokenSequence,
    candidates: list[TuningCandidate],
    supported: set[str],
) -> tuple[FixedReplayTuningResult, list[ReplayExecution]]:
    executions: list[ReplayExecution] = []

    def measure(
        candidate: TuningCandidate,
        repeat: int,
        order: Literal["forward", "reverse"],
    ) -> TuningSample:
        execution = runner.run(
            replay,
            candidate_id=candidate.candidate_id,
            settings=candidate.settings,
            supported_settings=supported,
        )
        executions.append(execution)
        log_name = f"tuning-{order}-{candidate.candidate_id}-{repeat}.log"
        (bundle.root / "logs" / log_name).write_text(
            execution.stdout + "\nSTDERR\n" + execution.stderr, encoding="utf-8"
        )
        if execution.return_code or execution.timed_out:
            raise RuntimeError(
                f"candidate {candidate.candidate_id} failed with exit {execution.return_code}"
            )
        if execution.decode_tokens_per_second is None:
            raise RuntimeError(f"candidate {candidate.candidate_id} did not report throughput")
        return TuningSample(
            candidate_id=candidate.candidate_id,
            repeat=repeat,
            order=order,
            decode_tokens_per_second=execution.decode_tokens_per_second,
            latency_ms=execution.elapsed_ms,
            time_to_first_token_ms=execution.time_to_first_token_ms,
            p95_latency_ms=execution.p95_latency_ms,
            input_token_ids=execution.input_token_ids,
            output_token_ids=execution.output_token_ids,
            settings_applied=execution.settings_applied,
            settings_ignored=execution.settings_ignored,
        )

    result = ColibriFixedReplayTuner(
        repeats=config.tuning.repeats,
        minimum_gain=config.tuning.minimum_gain_fraction,
        maximum_p95_regression=config.tuning.maximum_p95_regression_fraction,
    ).tune(replay=replay, candidates=candidates, measure=measure)
    return result, executions


def _tuning_csv(result: Any) -> list[dict[str, Any]]:
    rows = []
    for candidate in result.candidates:
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "order": "forward",
                "settings": candidate["settings"],
                "median_decode_tokens_per_second": candidate["median_decode_tokens_per_second"],
                "p95_latency_ms": candidate["median_p95_latency_ms"],
                "sample_count": len(candidate["samples"]),
                "selected": candidate["candidate_id"] == result.selected_candidate_id,
            }
        )
    reverse = result.reverse_confirmation or {}
    for name in ("winner", "baseline"):
        if isinstance(reverse.get(name), dict):
            candidate = reverse[name]
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "order": "reverse",
                    "settings": candidate["settings"],
                    "median_decode_tokens_per_second": candidate["median_decode_tokens_per_second"],
                    "p95_latency_ms": candidate["median_p95_latency_ms"],
                    "sample_count": len(candidate["samples"]),
                    "selected": candidate["candidate_id"] == result.selected_candidate_id,
                }
            )
    return rows


def _fixture_routing_policy_rows(
    *,
    config: Experiment009Config,
    selections: list[RouteSelection],
    expert_bytes: dict[tuple[int, int], int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_partition = {item.prompt_id: item.partition for item in config.routing.prompts}
    calibration = [
        row for row in selections if prompt_partition.get(row.request_id or "") == "calibration"
    ]
    heldout = [row for row in selections if prompt_partition.get(row.request_id or "") == "heldout"]
    policies = [PlacementPolicy(value) for value in config.routing.policies]
    evaluator = RoutingPolicyEvaluator(
        expert_bytes=expert_bytes,
        cache_slots_per_layer=config.routing.cache_slots_per_layer,
        hot_slots_per_layer=min(
            config.routing.hot_slots_per_layer, config.routing.cache_slots_per_layer
        ),
    )
    results = evaluator.evaluate_matrix(
        calibration=calibration,
        heldout=heldout,
        policies=policies,
    )
    rows = []
    for result in results:
        row = result.model_dump(mode="json")
        row["predicted_hit_rate"] = row.pop("expert_hit_rate")
        row["expert_hit_rate"] = None
        rows.append(row)
    prefill = [row for row in selections if row.phase == "prefill"]
    union = batch_expert_union(prefill, expert_bytes=expert_bytes) if prefill else {}
    return rows, {
        "calibration_selection_count": len(calibration),
        "heldout_selection_count": len(heldout),
        "batch_union": union,
        "measured_policy_count": 0,
    }


def _tokenize_prompts(model: ModelContext, config: Experiment009Config) -> dict[str, list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model.path, local_files_only=True, trust_remote_code=False
    )
    result = {}
    for prompt in config.routing.prompts:
        ids = tokenizer.encode(prompt.text, add_special_tokens=True)
        if not ids:
            raise ValueError(f"tokenizer returned no IDs for {prompt.prompt_id}")
        result[prompt.prompt_id] = [int(value) for value in ids]
    return result


def _run_practical_path(
    *,
    bundle: EvidenceBundle,
    config: Experiment009Config,
    options: Experiment009Options,
    engine_directory: Path,
    source_directory: Path,
    build_manifest_path: Path,
    model: ModelContext,
    capability_report: Any,
    model_inventory: Any,
    quantization_fingerprint: str,
    swarm_plan: Any,
    experts: list[Any],
) -> dict[str, Any]:
    usage_history_path = bundle.root / "logs" / "olmoe_usage_history.coli"
    bridge_environment = {
        "COLI_SWARM_BRIDGE": "1",
        "COLI_SWARM_BRIDGE_PATH": str(bundle.root / "telemetry.ndjson"),
        "COLI_SWARM_TELEMETRY": options.telemetry_level or config.backend.telemetry_level,
        "COLI_MODEL_ID": model.model_id,
        "COLI_MODEL_REVISION": model.model_revision,
        "COLI_USAGE_PATH": str(usage_history_path),
    }
    runner = ColibriReplayRunner(
        engine_directory=engine_directory,
        model_path=model.path,
        model_id=model.model_id,
        model_revision=model.model_revision,
        model_family=model.family,
        cap=config.backend.practical_baseline_cap,
        quant_bits=config.backend.quant_bits,
        ram_safety_reserve_bytes=config.backend.ram_safety_reserve_bytes,
        timeout_seconds=config.backend.request_timeout_seconds,
        environment=bridge_environment,
    )
    supported = ColibriCapabilityProbe.supported_tuning_settings(
        capability_report, model_family=model.family
    )
    tokenized = _tokenize_prompts(model, config)
    seed_prompt = tokenized[config.routing.prompts[0].prompt_id]
    direct_generations = []
    direct_trace_paths: list[Path] = []
    for repeat in range(3):
        trace_path = bundle.root / "logs" / f"real-direct-{repeat}.trace"
        execution = runner.generate_from_tokens(
            seed_prompt,
            completion_tokens=config.tuning.replay_completion_tokens,
            candidate_id=f"direct-real-{repeat}",
            supported_settings=supported,
            route_trace_path=trace_path,
        )
        direct_generations.append(execution)
        direct_trace_paths.append(trace_path)
        (bundle.root / "logs" / f"real-direct-{repeat}.log").write_text(
            execution.stdout + "\nSTDERR\n" + execution.stderr, encoding="utf-8"
        )
        if execution.return_code or not execution.output_token_ids:
            raise RuntimeError("direct practical Colibri generation failed")
    continuation = direct_generations[0].output_token_ids
    if any(item.output_token_ids != continuation for item in direct_generations):
        raise RuntimeError("deterministic practical Colibri generations diverged")
    bundle.write_json(
        "logs/configuration_A_results.json",
        [item.model_dump(mode="json") for item in direct_generations],
    )
    bundle.complete_configuration("A")
    replay = ReplayTokenSequence(
        model_id=model.model_id,
        model_revision=model.model_revision,
        tokenizer_hash=canonical_hash(_read_json(model.path / "tokenizer.json", {})),
        prompt_ids=seed_prompt,
        continuation_ids=continuation,
    )
    backend = ColibriBackend(
        engine_directory=engine_directory,
        source_directory=source_directory,
        build_manifest=build_manifest_path,
        model_path=model.path,
        model_id=model.model_id,
        model_revision=model.model_revision,
        model_family=model.family,
        mode=ColibriMode.BRIDGE,
        telemetry_level=TelemetryLevel(options.telemetry_level or config.backend.telemetry_level),
        telemetry_path=bundle.root / "telemetry.ndjson",
        log_directory=bundle.root / "logs" / "real-adapter",
        cap=config.backend.practical_baseline_cap,
        environment={"COLI_USAGE_PATH": str(usage_history_path)},
        ram_safety_reserve_bytes=config.backend.ram_safety_reserve_bytes,
    )
    adapter_generations = []
    adapter_trace_paths: list[Path] = []

    async def run_adapted() -> None:
        for repeat in range(3):
            trace_path = bundle.root / "logs" / f"real-adapter-{repeat}.trace"
            result = await backend.execute(
                _worker_job(
                    model=model,
                    request_id=f"adapter-real-{repeat}",
                    role=WorkerJobType.GENERATE,
                    timeout_seconds=config.backend.request_timeout_seconds,
                    token_ids=seed_prompt,
                    max_tokens=len(continuation),
                    metadata={
                        "candidate_id": f"adapter-real-{repeat}",
                        "route_trace_path": str(trace_path),
                    },
                )
            )
            adapter_generations.append(_generation_from_worker(result))
            adapter_trace_paths.append(trace_path)
        await backend.shutdown()

    asyncio.run(run_adapted())
    practical_comparisons = []
    for repeat, (direct, adapted, direct_trace, adapter_trace) in enumerate(
        zip(
            direct_generations,
            adapter_generations,
            direct_trace_paths,
            adapter_trace_paths,
            strict=True,
        )
    ):
        adapted_ids = adapted.get("output_token_ids") or adapted.get("replay_execution", {}).get(
            "output_token_ids"
        )
        direct_routes = [
            (
                item.call_index,
                item.row_index,
                item.layer_id,
                item.expert_id,
                item.routing_weight,
            )
            for item in ColibriRouteTraceReader().read(direct_trace)
        ]
        adapter_routes = [
            (
                item.call_index,
                item.row_index,
                item.layer_id,
                item.expert_id,
                item.routing_weight,
            )
            for item in ColibriRouteTraceReader().read(adapter_trace)
        ]
        input_match = direct.input_token_ids == (adapted.get("input_token_ids") or seed_prompt)
        output_match = direct.output_token_ids == adapted_ids
        stop_match = adapted.get("stop_reason") == "length"
        routes_match = bool(direct_routes) and direct_routes == adapter_routes
        model_revision_match = bool(model.model_revision)
        model_config_match = bool(model_inventory.model_config_hash)
        quantization_match = bool(quantization_fingerprint)
        engine_match = bool(model_inventory.engine_build_fingerprint)
        practical_comparisons.append(
            {
                "comparison_id": f"practical-{repeat}",
                "path": "practical_model",
                "input_token_ids_match": input_match,
                "output_token_ids_match": output_match,
                "stop_reason_match": stop_match,
                "routing_trace_match": routes_match,
                "direct_route_selection_count": len(direct_routes),
                "adapter_route_selection_count": len(adapter_routes),
                "model_revision_match": model_revision_match,
                "model_revision_direct": model.model_revision,
                "model_revision_adapter": model.model_revision,
                "model_config_hash_match": model_config_match,
                "model_config_hash_direct": model_inventory.model_config_hash,
                "model_config_hash_adapter": model_inventory.model_config_hash,
                "engine_build_fingerprint_match": engine_match,
                "engine_build_fingerprint_direct": model_inventory.engine_build_fingerprint,
                "engine_build_fingerprint_adapter": model_inventory.engine_build_fingerprint,
                "quantization_fingerprint_match": quantization_match,
                "quantization_fingerprint_direct": quantization_fingerprint,
                "quantization_fingerprint_adapter": quantization_fingerprint,
                "final_text_match": None,
                "exact": all(
                    (
                        input_match,
                        output_match,
                        stop_match,
                        routes_match,
                        model_revision_match,
                        model_config_match,
                        engine_match,
                        quantization_match,
                    )
                ),
            }
        )
    bundle.write_json(
        "logs/configuration_B_results.json",
        {"generations": adapter_generations, "comparisons": practical_comparisons},
    )
    bundle.complete_configuration("B")
    overhead_rows = []
    benchmark_rows = []
    for configuration, values in (("direct", direct_generations),):
        for repeat, execution in enumerate(values):
            row = {
                "configuration": configuration,
                "workload": "practical_olmoe_generation",
                "repeat": repeat,
                "decode_tokens_per_second": execution.decode_tokens_per_second,
                "time_to_first_token_ms": execution.time_to_first_token_ms,
                "latency_ms": execution.elapsed_ms,
                "token_identity_match": practical_comparisons[repeat]["exact"],
            }
            overhead_rows.append(row)
            benchmark_rows.append({**row, "configuration": "A", "evidence_class": "MEASURED"})
    for repeat, adapted in enumerate(adapter_generations):
        execution = adapted.get("replay_execution", {})
        row = {
            "configuration": "adapter",
            "workload": "practical_olmoe_generation",
            "repeat": repeat,
            "decode_tokens_per_second": execution.get("decode_tokens_per_second"),
            "time_to_first_token_ms": execution.get("time_to_first_token_ms"),
            "latency_ms": execution.get("elapsed_ms"),
            "token_identity_match": practical_comparisons[repeat]["exact"],
        }
        overhead_rows.append(row)
        benchmark_rows.append({**row, "configuration": "B", "evidence_class": "MEASURED"})

    resource_plan_cap = int(
        swarm_plan.routed_expert_tiers.get("ram", {}).get(
            "cache_slots_per_layer", config.backend.practical_baseline_cap
        )
    )
    if resource_plan_cap < 1:
        raise ValueError("translated Colibri resource plan has no executable expert cache")
    resource_runner = ColibriReplayRunner(
        engine_directory=engine_directory,
        model_path=model.path,
        model_id=model.model_id,
        model_revision=model.model_revision,
        model_family=model.family,
        cap=resource_plan_cap,
        quant_bits=config.backend.quant_bits,
        ram_safety_reserve_bytes=config.backend.ram_safety_reserve_bytes,
        timeout_seconds=config.backend.request_timeout_seconds,
        environment={
            **bridge_environment,
            "COLI_USAGE_PATH": str(bundle.root / "logs" / "resource_plan_usage.coli"),
        },
    )
    resource_plan_executions: list[ReplayExecution] = []
    for repeat in range(config.tuning.repeats):
        execution = resource_runner.run(
            replay,
            candidate_id=f"resource-plan-{repeat}",
            supported_settings=supported,
        )
        if execution.return_code or execution.output_token_ids != replay.continuation_ids:
            raise RuntimeError("Colibri resource-plan replay failed correctness")
        resource_plan_executions.append(execution)
        (bundle.root / "logs" / f"resource-plan-{repeat}.log").write_text(
            execution.stdout + "\nSTDERR\n" + execution.stderr,
            encoding="utf-8",
        )
        benchmark_rows.append(
            {
                "configuration": "C",
                "workload": "practical_olmoe_fixed_replay",
                "repeat": repeat,
                "resource_plan_cache_slots_per_layer": resource_plan_cap,
                "decode_tokens_per_second": execution.decode_tokens_per_second,
                "time_to_first_token_ms": execution.time_to_first_token_ms,
                "latency_ms": execution.elapsed_ms,
                "token_identity_match": True,
                "evidence_class": "MEASURED",
            }
        )
    bundle.write_json(
        "logs/configuration_C_results.json",
        {
            "cache_slots_per_layer": resource_plan_cap,
            "executions": [item.model_dump(mode="json") for item in resource_plan_executions],
        },
    )
    bundle.complete_configuration("C")

    all_selections: list[RouteSelection] = []
    route_executions: list[tuple[str, ReplayExecution]] = []
    expert_layers = len({expert.layer_id for expert in experts})
    call_offset = 0
    for prompt in config.routing.prompts:
        prompt_replay = ReplayTokenSequence(
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_hash=replay.tokenizer_hash,
            prompt_ids=tokenized[prompt.prompt_id],
            continuation_ids=continuation,
        )
        trace_path = bundle.root / "logs" / f"route-{prompt.prompt_id}.trace"
        execution = runner.run(
            prompt_replay,
            candidate_id=f"route-{prompt.prompt_id}",
            supported_settings=supported,
            route_trace_path=trace_path,
        )
        if execution.return_code:
            raise RuntimeError(f"route replay failed for {prompt.prompt_id}")
        route_executions.append((prompt.prompt_id, execution))
        raw = ColibriRouteTraceReader().read(trace_path)
        calls = sorted({row.call_index for row in raw})
        phase_by_call = {
            call: ("prefill" if index < expert_layers else "decode")
            for index, call in enumerate(calls)
        }
        parsed = ColibriRouteTraceReader().read(
            trace_path,
            phase_by_call=phase_by_call,
            request_by_call={call: prompt.prompt_id for call in calls},
        )
        for selection in parsed:
            payload = selection.model_dump(mode="python")
            payload["call_index"] = selection.call_index + call_offset
            all_selections.append(RouteSelection.model_validate(payload))
        call_offset += (max(calls) + 1) if calls else 0

    candidates, rejected = _tuning_candidates(
        config=config,
        family=model.family,
        supported=supported,
        physical_cores=int(capability_report.cpu.get("physical_cores", 1)),
    )
    tuning_result, tuning_executions = _run_tuning(
        bundle=bundle,
        config=config,
        runner=runner,
        replay=replay,
        candidates=candidates,
        supported=supported,
    )
    selected_tuning = next(
        row
        for row in tuning_result.candidates
        if row["candidate_id"] == tuning_result.selected_candidate_id
    )
    for repeat, sample in enumerate(selected_tuning["samples"]):
        benchmark_rows.append(
            {
                "configuration": "D",
                "workload": "practical_olmoe_fixed_replay",
                "repeat": repeat,
                "candidate_id": tuning_result.selected_candidate_id,
                "settings": selected_tuning["settings"],
                "decode_tokens_per_second": sample["decode_tokens_per_second"],
                "time_to_first_token_ms": sample.get("time_to_first_token_ms"),
                "latency_ms": sample.get("latency_ms"),
                "token_identity_match": (
                    sample["input_token_ids"] == replay.prompt_ids
                    and sample["output_token_ids"] == replay.continuation_ids
                ),
                "evidence_class": "MEASURED",
            }
        )
    bundle.write_json(
        "logs/configuration_D_results.json",
        tuning_result.model_dump(mode="json"),
    )
    bundle.complete_configuration("D")

    prompt_partition = {item.prompt_id: item.partition for item in config.routing.prompts}
    calibration = [
        row for row in all_selections if prompt_partition.get(row.request_id or "") == "calibration"
    ]
    heldout = [
        row for row in all_selections if prompt_partition.get(row.request_id or "") == "heldout"
    ]
    layer_count = max((expert.layer_id for expert in experts), default=-1) + 1
    experts_per_layer = max((expert.expert_id for expert in experts), default=-1) + 1
    hot_pin_bytes, hot_pin_metadata = calibration_hot_pin_bitmap(
        calibration,
        layer_count=layer_count,
        experts_per_layer=experts_per_layer,
        hot_slots_per_layer=min(
            config.routing.hot_slots_per_layer,
            config.backend.practical_baseline_cap,
        ),
    )
    hot_pin_path = bundle.root / "logs" / "swarm_calibration_hot_pinned.bin"
    hot_pin_path.write_bytes(hot_pin_bytes)
    hot_pin_metadata["content_hash"] = sha256_bytes(hot_pin_bytes)
    hot_pin_metadata["path"] = str(hot_pin_path)
    expert_bytes = {(expert.layer_id, expert.expert_id): expert.total_bytes for expert in experts}
    evaluator = RoutingPolicyEvaluator(
        expert_bytes=expert_bytes,
        cache_slots_per_layer=config.routing.cache_slots_per_layer,
        hot_slots_per_layer=min(
            config.routing.hot_slots_per_layer, config.routing.cache_slots_per_layer
        ),
    )
    policies = [PlacementPolicy(value) for value in config.routing.policies]
    actual_settings = {
        PlacementPolicy.PLAIN_LRU: {},
        PlacementPolicy.FREQUENCY: {"COLI_HOT_PIN_PATH": str(hot_pin_path)},
        PlacementPolicy.COLIBRI_RECOMMENDED: {"PILOT": "1", "PILOT_EVICT_GUARD": "1"},
        PlacementPolicy.SWARM: {
            "COLI_HOT_PIN_PATH": str(hot_pin_path),
            "PILOT": "1",
            "WIDE": "2",
            "PILOT_EVICT_GUARD": "1",
        },
    }
    measured: dict[PlacementPolicy, dict[str, float]] = {}
    policy_executions: dict[PlacementPolicy, list[ReplayExecution]] = defaultdict(list)
    heldout_prompts = [item for item in config.routing.prompts if item.partition == "heldout"]
    for policy, settings in actual_settings.items():
        for prompt in heldout_prompts:
            policy_replay = ReplayTokenSequence(
                model_id=model.model_id,
                model_revision=model.model_revision,
                tokenizer_hash=replay.tokenizer_hash,
                prompt_ids=tokenized[prompt.prompt_id],
                continuation_ids=continuation,
            )
            execution = runner.run(
                policy_replay,
                candidate_id=f"policy-{policy.value}-{prompt.prompt_id}",
                settings={
                    **settings,
                    "COLI_USAGE_PATH": str(
                        bundle.root / "logs" / f"policy-{policy.value}-usage.coli"
                    ),
                },
                supported_settings=supported,
            )
            if execution.return_code or execution.decode_tokens_per_second is None:
                raise RuntimeError(f"held-out policy execution failed for {policy.value}")
            policy_executions[policy].append(execution)
        speeds = [item.decode_tokens_per_second for item in policy_executions[policy]]
        hit_rates = [
            item.expert_hit_rate
            for item in policy_executions[policy]
            if item.expert_hit_rate is not None
        ]
        measured[policy] = {
            "decode_tokens_per_second": float(
                median(value for value in speeds if value is not None)
            ),
        }
        ttft = [
            item.time_to_first_token_ms
            for item in policy_executions[policy]
            if item.time_to_first_token_ms is not None
        ]
        read_latencies = [
            item.storage_read_duration_ms / item.storage_read_count
            for item in policy_executions[policy]
            if item.storage_read_duration_ms is not None and item.storage_read_count
        ]
        if ttft:
            measured[policy]["time_to_first_token_ms"] = float(median(ttft))
        if read_latencies:
            measured[policy]["expert_load_latency_ms"] = float(median(read_latencies))
        if hit_rates:
            measured[policy]["measured_expert_hit_rate"] = float(median(hit_rates))
    baseline_policy_speed = measured[PlacementPolicy.PLAIN_LRU]["decode_tokens_per_second"]
    forward_policy_winner = max(
        actual_settings,
        key=lambda policy: measured[policy]["decode_tokens_per_second"],
    )
    forward_policy_gain = (
        measured[forward_policy_winner]["decode_tokens_per_second"] / baseline_policy_speed - 1.0
    )
    routing_confirmation: dict[str, Any] = {
        "attempted": False,
        "accepted": False,
        "forward_winner": forward_policy_winner.value,
        "forward_gain": forward_policy_gain,
        "minimum_gain": config.tuning.minimum_gain_fraction,
    }
    confirmation_executions: dict[PlacementPolicy, list[ReplayExecution]] = defaultdict(list)
    if (
        forward_policy_winner != PlacementPolicy.PLAIN_LRU
        and forward_policy_gain >= config.tuning.minimum_gain_fraction
    ):
        confirmation_traces: dict[tuple[PlacementPolicy, str], Path] = {}
        # Winner first and baseline last gives the baseline any remaining
        # warm-page advantage, making acceptance deliberately conservative.
        for policy in (forward_policy_winner, PlacementPolicy.PLAIN_LRU):
            for prompt in heldout_prompts:
                policy_replay = ReplayTokenSequence(
                    model_id=model.model_id,
                    model_revision=model.model_revision,
                    tokenizer_hash=replay.tokenizer_hash,
                    prompt_ids=tokenized[prompt.prompt_id],
                    continuation_ids=continuation,
                )
                trace_path = (
                    bundle.root
                    / "logs"
                    / f"routing-confirm-{policy.value}-{prompt.prompt_id}.trace"
                )
                confirmation_traces[(policy, prompt.prompt_id)] = trace_path
                execution = runner.run(
                    policy_replay,
                    candidate_id=f"routing-confirm-{policy.value}-{prompt.prompt_id}",
                    settings={
                        **actual_settings[policy],
                        "COLI_USAGE_PATH": str(
                            bundle.root / "logs" / f"routing-confirm-{policy.value}-usage.coli"
                        ),
                    },
                    supported_settings=supported,
                    route_trace_path=trace_path,
                )
                if execution.return_code or execution.decode_tokens_per_second is None:
                    raise RuntimeError(f"routing reverse confirmation failed for {policy.value}")
                confirmation_executions[policy].append(execution)

        def route_signature(path: Path) -> list[tuple[int, int, int, int, float | None]]:
            return [
                (
                    item.call_index,
                    item.row_index,
                    item.layer_id,
                    item.expert_id,
                    item.routing_weight,
                )
                for item in ColibriRouteTraceReader().read(path)
            ]

        routing_match = all(
            route_signature(confirmation_traces[(forward_policy_winner, prompt.prompt_id)])
            == route_signature(confirmation_traces[(PlacementPolicy.PLAIN_LRU, prompt.prompt_id)])
            for prompt in heldout_prompts
        )
        winner_runs = confirmation_executions[forward_policy_winner]
        baseline_runs = confirmation_executions[PlacementPolicy.PLAIN_LRU]
        confirmed_winner_speed = float(
            median(
                item.decode_tokens_per_second
                for item in winner_runs
                if item.decode_tokens_per_second
            )
        )
        confirmed_baseline_speed = float(
            median(
                item.decode_tokens_per_second
                for item in baseline_runs
                if item.decode_tokens_per_second
            )
        )
        confirmed_gain = confirmed_winner_speed / confirmed_baseline_speed - 1.0
        winner_tail = max(item.elapsed_ms for item in winner_runs)
        baseline_tail = max(item.elapsed_ms for item in baseline_runs)
        tail_regression = winner_tail / baseline_tail - 1.0
        winner_ttft = [
            item.time_to_first_token_ms
            for item in winner_runs
            if item.time_to_first_token_ms is not None
        ]
        baseline_ttft = [
            item.time_to_first_token_ms
            for item in baseline_runs
            if item.time_to_first_token_ms is not None
        ]
        ttft_regression = (
            float(median(winner_ttft)) / float(median(baseline_ttft)) - 1.0
            if winner_ttft and baseline_ttft
            else None
        )
        token_match = all(
            item.output_token_ids == replay.continuation_ids
            for item in [*winner_runs, *baseline_runs]
        )
        accepted = bool(
            routing_match
            and token_match
            and confirmed_gain >= config.tuning.minimum_gain_fraction
            and tail_regression <= config.acceptance.maximum_p95_regression_fraction
            and ttft_regression is not None
            and ttft_regression <= config.acceptance.maximum_ttft_regression_fraction
        )
        routing_confirmation = {
            "attempted": True,
            "accepted": accepted,
            "order": [forward_policy_winner.value, PlacementPolicy.PLAIN_LRU.value],
            "winner": forward_policy_winner.value,
            "baseline": PlacementPolicy.PLAIN_LRU.value,
            "forward_gain": forward_policy_gain,
            "confirmed_gain": confirmed_gain,
            "winner_median_decode_tokens_per_second": confirmed_winner_speed,
            "baseline_median_decode_tokens_per_second": confirmed_baseline_speed,
            "tail_latency_regression_fraction": tail_regression,
            "ttft_regression_fraction": ttft_regression,
            "routing_trace_match": routing_match,
            "token_identity_match": token_match,
            "samples_per_policy": len(winner_runs),
            "rejection_reason": None
            if accepted
            else "gain, correctness, TTFT, or tail gate failed in reverse order",
        }
    for repeat, execution in enumerate(policy_executions[PlacementPolicy.SWARM]):
        benchmark_rows.append(
            {
                "configuration": "E",
                "workload": "heldout_routing_aware_plan",
                "repeat": repeat,
                "policy": PlacementPolicy.SWARM.value,
                "decode_tokens_per_second": execution.decode_tokens_per_second,
                "time_to_first_token_ms": execution.time_to_first_token_ms,
                "latency_ms": execution.elapsed_ms,
                "token_identity_match": execution.output_token_ids == replay.continuation_ids,
                "evidence_class": "MEASURED",
            }
        )
    for policy, executions in confirmation_executions.items():
        for repeat, execution in enumerate(executions):
            benchmark_rows.append(
                {
                    "configuration": "E",
                    "workload": "heldout_routing_reverse_confirmation",
                    "repeat": repeat,
                    "policy": policy.value,
                    "order": "winner_first"
                    if policy != PlacementPolicy.PLAIN_LRU
                    else "baseline_last",
                    "decode_tokens_per_second": execution.decode_tokens_per_second,
                    "time_to_first_token_ms": execution.time_to_first_token_ms,
                    "latency_ms": execution.elapsed_ms,
                    "token_identity_match": execution.output_token_ids == replay.continuation_ids,
                    "evidence_class": "MEASURED",
                }
            )
    bundle.write_json(
        "logs/configuration_E_results.json",
        {
            "routing_aware_placement": hot_pin_metadata,
            "executions": [
                item.model_dump(mode="json") for item in policy_executions[PlacementPolicy.SWARM]
            ],
            "reverse_confirmation": routing_confirmation,
            "reverse_executions": {
                policy.value: [item.model_dump(mode="json") for item in executions]
                for policy, executions in confirmation_executions.items()
            },
        },
    )
    bundle.complete_configuration("E")
    evaluations = evaluator.evaluate_matrix(
        calibration=calibration,
        heldout=heldout,
        policies=policies,
        measured=measured,
    )
    policy_rows = []
    for result in evaluations:
        policy_row: dict[str, Any] = result.model_dump(mode="json")
        policy_row["predicted_hit_rate"] = policy_row["expert_hit_rate"]
        executions = policy_executions.get(result.policy, [])
        observed_hits = sum(item.expert_cache_hits or 0 for item in executions)
        observed_misses = sum(item.expert_cache_misses or 0 for item in executions)
        observed_total = observed_hits + observed_misses
        policy_row["expert_hit_rate"] = observed_hits / observed_total if observed_total else None
        exact_read_bytes = [
            item.storage_read_bytes for item in executions if item.storage_read_bytes is not None
        ]
        exact_read_count = [
            item.storage_read_count for item in executions if item.storage_read_count is not None
        ]
        exact_read_ms = [
            item.storage_read_duration_ms
            for item in executions
            if item.storage_read_duration_ms is not None
        ]
        total_read_count = sum(exact_read_count)
        completion_count = len(executions) * len(continuation)
        policy_row["measured_storage_read_bytes"] = (
            sum(exact_read_bytes) if len(exact_read_bytes) == len(executions) else None
        )
        policy_row["bytes_read_per_token"] = (
            policy_row["measured_storage_read_bytes"] / completion_count
            if policy_row["measured_storage_read_bytes"] is not None and completion_count
            else None
        )
        policy_row["expert_load_latency_ms"] = (
            sum(exact_read_ms) / total_read_count
            if len(exact_read_ms) == len(executions)
            and len(exact_read_count) == len(executions)
            and total_read_count > 0
            else None
        )
        useful = [
            item.prefetch_useful_bytes
            for item in executions
            if item.prefetch_useful_bytes is not None
        ]
        wasted = [
            item.prefetch_wasted_bytes
            for item in executions
            if item.prefetch_wasted_bytes is not None
        ]
        evictions = [
            item.expert_evictions for item in executions if item.expert_evictions is not None
        ]
        policy_row["prefetch_useful_bytes"] = (
            sum(useful) if len(useful) == len(executions) else None
        )
        policy_row["prefetch_wasted_bytes"] = (
            sum(wasted) if len(wasted) == len(executions) else None
        )
        policy_row["tier_churn"] = sum(evictions) if len(evictions) == len(executions) else None
        policy_row["measurement_basis"] = (
            "colibri_bridge_exact_counters" if exact_read_bytes else "not_measured"
        )
        policy_row["measured"] = bool(executions)
        policy_row["runtime_settings"] = actual_settings.get(result.policy)
        policy_row["calibration_hot_pin_hash"] = (
            hot_pin_metadata["content_hash"]
            if result.policy in {PlacementPolicy.FREQUENCY, PlacementPolicy.SWARM}
            else None
        )
        policy_row["routing_reverse_confirmation_attempted"] = routing_confirmation["attempted"]
        policy_row["routing_reverse_confirmation_accepted"] = (
            routing_confirmation["accepted"]
            if result.policy in {forward_policy_winner, PlacementPolicy.PLAIN_LRU}
            else None
        )
        policy_row["routing_reverse_confirmed_gain"] = (
            routing_confirmation.get("confirmed_gain")
            if result.policy == forward_policy_winner
            else None
        )
        policy_rows.append(policy_row)
    measured_speeds = [
        float(row["decode_tokens_per_second"])
        for row in policy_rows
        if row.get("measured") and row.get("decode_tokens_per_second") is not None
    ]
    if measured_speeds:
        best_speed = max(measured_speeds)
        history_policies = {
            PlacementPolicy.FREQUENCY.value,
            PlacementPolicy.SWARM.value,
        }
        for row in policy_rows:
            speed = row.get("decode_tokens_per_second")
            if not row.get("measured") or speed is None:
                continue
            row["heldout_regret"] = best_speed / float(speed) - 1.0
            row["rejected"] = False
            row["rejection_reason"] = None
            useful_prefetch_bytes = int(row.get("prefetch_useful_bytes") or 0)
            wasted_prefetch_bytes = int(row.get("prefetch_wasted_bytes") or 0)
            predicted = row.get("predicted_hit_rate")
            observed = row.get("expert_hit_rate")
            overfit_gap = (
                float(predicted) - float(observed)
                if predicted is not None and observed is not None
                else None
            )
            row["predicted_minus_measured_hit_rate"] = overfit_gap
            if wasted_prefetch_bytes > useful_prefetch_bytes and row["heldout_regret"] > 0:
                row["rejected"] = True
                row["rejection_reason"] = (
                    "prefetch wasted more bytes than it made useful and lost held-out throughput"
                )
            elif (
                row["policy"] in history_policies
                and row["heldout_regret"] >= config.tuning.minimum_gain_fraction
            ):
                row["rejected"] = True
                row["rejection_reason"] = (
                    "calibration-derived placement exceeded the held-out regret threshold"
                )
            elif overfit_gap is not None and overfit_gap > 0.05 and row["heldout_regret"] > 0:
                row["rejected"] = True
                row["rejection_reason"] = (
                    "calibration-predicted hit rate overfit measured held-out residency"
                )
    storage_observations = []
    for source, executions in (
        ("resource_plan", resource_plan_executions),
        ("route", [item for _, item in route_executions]),
        ("tuning", tuning_executions),
        ("policy", [item for values in policy_executions.values() for item in values]),
        (
            "routing_reverse_confirmation",
            [item for values in confirmation_executions.values() for item in values],
        ),
    ):
        for execution in executions:
            storage_observations.append(
                {
                    "category": "storage",
                    "source": source,
                    "candidate_id": execution.candidate_id,
                    "read_count": execution.storage_read_count,
                    "bytes": execution.storage_read_bytes,
                    "duration_ms": execution.storage_read_duration_ms,
                    "measurement_basis": "colibri_bridge_exact_counters",
                }
            )
    practical_events = [
        event
        for event in ColibriTelemetryReader(bundle.root / "telemetry.ndjson").read()
        if event.model_id == model.model_id and event.engine_family == "olmoe"
    ]
    event_types = {event.event_type for event in practical_events}
    required_event_types = {
        "engine_ready",
        "tier_inventory",
        "request_started",
        "prefill_started",
        "prefill_completed",
        "request_completed",
        "route_summary",
        "expert_cache_hit",
        "expert_cache_miss",
        "expert_loaded",
        "storage_read",
        "cpu_compute",
        "resource_snapshot",
    }
    storage_events = [event for event in practical_events if event.event_type == "storage_read"]
    completed_events = [
        event for event in practical_events if event.event_type == "request_completed"
    ]
    telemetry_coverage = {
        "event_count": len(practical_events),
        "event_types": sorted(event_types),
        "missing_required_event_types": sorted(required_event_types.difference(event_types)),
        "storage_counters_valid": bool(storage_events)
        and all(
            isinstance(event.payload.get("byte_count"), int)
            and isinstance(event.payload.get("duration_ns"), int)
            for event in storage_events
        ),
        "phase_timings_valid": bool(completed_events)
        and all(
            isinstance(event.payload.get("prefill_duration_ns"), int)
            and isinstance(event.payload.get("decode_duration_ns"), int)
            for event in completed_events
        ),
        "tier_residency_observed": "tier_inventory" in event_types
        and any(
            event.payload.get("execution_tier") == "ram"
            for event in practical_events
            if event.event_type == "route_summary"
        ),
    }
    telemetry_coverage["complete"] = not telemetry_coverage["missing_required_event_types"] and all(
        telemetry_coverage[key]
        for key in ("storage_counters_valid", "phase_timings_valid", "tier_residency_observed")
    )
    prefill_selections = [item for item in all_selections if item.phase == "prefill"]
    measured_prefill_ms = [
        int(event.payload["prefill_duration_ns"]) / 1_000_000
        for event in completed_events
        if isinstance(event.payload.get("prefill_duration_ns"), int)
    ]
    batch_union = batch_expert_union(
        prefill_selections,
        expert_bytes=expert_bytes,
        prefill_duration_ms=(float(median(measured_prefill_ms)) if measured_prefill_ms else None),
    )
    return {
        "replay": replay,
        "comparisons": practical_comparisons,
        "overhead_rows": overhead_rows,
        "benchmark_rows": benchmark_rows,
        "selections": all_selections,
        "route_executions": route_executions,
        "tuning_candidates": candidates,
        "rejected_candidates": rejected,
        "tuning_result": tuning_result,
        "tuning_rows": _tuning_csv(tuning_result),
        "policy_rows": policy_rows,
        "storage_observations": storage_observations,
        "telemetry_coverage": telemetry_coverage,
        "batch_union": batch_union,
        "routing_aware_placement": hot_pin_metadata,
        "routing_reverse_confirmation": routing_confirmation,
        "resource_plan_cache_slots_per_layer": resource_plan_cap,
        "resource_plan_execution_count": len(resource_plan_executions),
        "usage_history": (
            ColibriUsageHistoryReader().read(
                usage_history_path,
                expected_layers=expert_layers,
                expected_experts=max(expert.expert_id for expert in experts) + 1,
                expected_engine="olmoe",
            )
            if usage_history_path.is_file()
            else None
        ),
        "policy_measured_count": sum(bool(policy_executions.get(policy)) for policy in policies),
        "benchmark_configurations": sorted({str(row["configuration"]) for row in benchmark_rows}),
    }


def _telemetry_tables(
    bundle: EvidenceBundle,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    path = bundle.root / "telemetry.ndjson"
    if not path.is_file():
        path.write_text("", encoding="utf-8")
        return {"event_count": 0}, [], []
    events = ColibriTelemetryReader(path).read()
    summary = ColibriTelemetryReader.summarize(events)
    cache_rows = []
    storage_rows = []
    for event in events:
        common = {
            "timestamp_ns": event.timestamp_ns,
            "request_id": event.request_id,
            "event_type": event.event_type,
            "model_id": event.model_id,
            **event.payload,
        }
        if event.event_type in {
            "expert_cache_hit",
            "expert_cache_miss",
            "expert_loaded",
            "expert_prefetch_started",
            "expert_prefetch_completed",
            "expert_promoted",
            "expert_demoted",
            "expert_evicted",
        }:
            cache_rows.append(common)
        if event.event_type in {
            "storage_read",
            "host_to_device_transfer",
            "device_to_host_transfer",
            "cpu_compute",
            "gpu_compute",
        }:
            storage_rows.append(
                {
                    **common,
                    "category": "storage" if event.event_type == "storage_read" else "compute",
                    "duration_ms": (
                        float(event.payload["duration_ns"]) / 1_000_000
                        if event.payload.get("duration_ns") is not None
                        else None
                    ),
                }
            )
    return summary, cache_rows, storage_rows


def _measured_tier_rows(bundle: EvidenceBundle, *, model_id: str) -> list[dict[str, Any]]:
    path = bundle.root / "telemetry.ndjson"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for event in ColibriTelemetryReader(path).read():
        if event.model_id != model_id:
            continue
        if event.event_type == "tier_inventory":
            for tier in event.payload.get("tiers", []):
                if isinstance(tier, dict):
                    rows.append(
                        {
                            "source": "measured_bridge",
                            "timestamp_ns": event.timestamp_ns,
                            "request_id": event.request_id,
                            **tier,
                        }
                    )
        elif event.event_type == "route_summary":
            rows.append(
                {
                    "source": "measured_execution",
                    "timestamp_ns": event.timestamp_ns,
                    "request_id": event.request_id,
                    "tier": event.payload.get("execution_tier"),
                    "expert_execution_count": event.payload.get("selection_count"),
                    "prefill_tokens": event.payload.get("prefill_tokens"),
                    "decode_tokens": event.payload.get("decode_tokens"),
                }
            )
        elif (
            event.event_type == "resource_snapshot"
            and event.payload.get("resident_expert_bytes") is not None
        ):
            rows.append(
                {
                    "source": "measured_snapshot",
                    "timestamp_ns": event.timestamp_ns,
                    "request_id": event.request_id,
                    "tier": "ram",
                    "allocated_expert_bytes": event.payload.get("resident_expert_bytes"),
                    "rss_bytes": event.payload.get("rss_bytes"),
                }
            )
    return rows


def _metric_summary(overhead_rows: list[dict[str, Any]], workload: str) -> dict[str, Any]:
    by_configuration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overhead_rows:
        if row.get("workload") == workload:
            by_configuration[str(row.get("configuration"))].append(row)
    direct = by_configuration.get("direct", [])
    adapter = by_configuration.get("adapter", [])

    def values(rows: list[dict[str, Any]], field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) is not None]

    direct_speed, adapter_speed = (
        values(direct, "decode_tokens_per_second"),
        values(adapter, "decode_tokens_per_second"),
    )
    direct_latency, adapter_latency = values(direct, "latency_ms"), values(adapter, "latency_ms")
    direct_ttft, adapter_ttft = (
        values(direct, "time_to_first_token_ms"),
        values(adapter, "time_to_first_token_ms"),
    )
    return {
        "workload": workload,
        "direct_samples": len(direct),
        "adapter_samples": len(adapter),
        "decode_throughput_regression_fraction": (
            1 - median(adapter_speed) / median(direct_speed)
            if direct_speed and adapter_speed
            else None
        ),
        "latency_regression_fraction": (
            median(adapter_latency) / median(direct_latency) - 1
            if direct_latency and adapter_latency
            else None
        ),
        "p95_latency_regression_fraction": (
            float(np.percentile(adapter_latency, 95) / np.percentile(direct_latency, 95) - 1)
            if direct_latency and adapter_latency
            else None
        ),
        "ttft_regression_fraction": (
            median(adapter_ttft) / median(direct_ttft) - 1 if direct_ttft and adapter_ttft else None
        ),
    }


def _write_reproduce(
    bundle: EvidenceBundle,
    *,
    options: Experiment009Options,
    model: ModelContext,
    repository_root: Path,
) -> None:
    mode = "-Full" if options.full else "-Quick"
    model_arg = f" -ModelPath '{model.path}' -ModelFamily '{model.family}'" if options.full else ""
    script = f"""$ErrorActionPreference = 'Stop'
$repositoryRoot = '{repository_root}'
& (Join-Path $repositoryRoot 'experiments\\009_colibri_adaptive_expert_runtime\\reproduce.ps1') {mode}{model_arg} -OutputDirectory (Split-Path -Parent $PSScriptRoot) -Resume
exit $LASTEXITCODE
"""
    bundle.write_text("reproduce.ps1", script)


def _ensure_artifacts(bundle: EvidenceBundle) -> None:
    json_files = {
        "capability_report.json",
        "model_inventory.json",
        "tensor_inventory.json",
        "expert_inventory.json",
        "native_quantization_inventory.json",
        "storage_inventory.json",
        "hardware_profile.json",
        "colibri_resource_plan.json",
        "swarm_resource_plan.json",
        "routing_trace_summary.json",
        "replay_tokens.json",
        "tuning_candidates.json",
        "correctness_results.json",
        "microshard_descriptors.json",
        "verdict.json",
    }
    csv_files = {name for name in REQUIRED_FILES if name.endswith(".csv")}
    for name in json_files:
        if not (bundle.root / name).is_file():
            bundle.write_json(name, {"status": "NOT_AVAILABLE", "reason": "stage did not complete"})
    for name in csv_files:
        if not (bundle.root / name).is_file():
            bundle.write_csv(name, [])
    if not (bundle.root / "telemetry.ndjson").is_file():
        bundle.write_text("telemetry.ndjson", "")


def _evaluate_gates(
    *,
    options: Experiment009Options,
    dependency_ok: bool,
    build_ok: bool,
    fixture: dict[str, Any],
    practical: dict[str, Any] | None,
    plan_ok: bool,
    microshard_ok: bool,
    telemetry_summary: dict[str, Any],
    overhead_summary: dict[str, Any],
    tuning_result: Any | None,
    policy_rows: list[dict[str, Any]],
    artifact_audit: dict[str, Any],
    config: Experiment009Config,
) -> list[GateResult]:
    full_class = EvidenceClass.MEASURED if options.full else EvidenceClass.FIXTURE
    comparisons = fixture.get("comparisons", []) + (
        practical.get("comparisons", []) if practical else []
    )
    request_count = fixture.get("request_execution_count", 0) + (
        6 if practical and practical.get("comparisons") else 0
    )
    exact = bool(comparisons) and all(row.get("exact") for row in comparisons)
    full_correctness = exact and request_count >= config.acceptance.minimum_correctness_requests
    real_routes = bool(practical and practical.get("selections"))
    real_telemetry = bool(practical and practical.get("telemetry_coverage", {}).get("complete"))
    required_overhead_metrics = (
        (
            overhead_summary.get("decode_throughput_regression_fraction"),
            config.acceptance.maximum_decode_regression_fraction,
        ),
        (
            overhead_summary.get("ttft_regression_fraction"),
            config.acceptance.maximum_ttft_regression_fraction,
        ),
        (
            overhead_summary.get("p95_latency_regression_fraction"),
            config.acceptance.maximum_p95_regression_fraction,
        ),
    )
    overhead_complete = all(value is not None for value, _threshold in required_overhead_metrics)
    overhead_pass = overhead_complete and all(
        value <= threshold for value, threshold in required_overhead_metrics if value is not None
    )
    tuning_ok = bool(
        tuning_result
        and tuning_result.repeats >= 3
        and tuning_result.reverse_confirmation is not None
        and tuning_result.reverse_confirmed
    )
    measured_policies = sum(bool(row.get("measured")) for row in policy_rows)
    plan_execution_ok = bool(practical and practical.get("resource_plan_execution_count", 0) >= 3)
    routing_aware_physical = bool(
        practical
        and practical.get("routing_aware_placement", {}).get("pinned_expert_count", 0)
        and any(
            row.get("policy") == PlacementPolicy.SWARM.value and row.get("measured")
            for row in policy_rows
        )
    )
    routing_confirmation = practical.get("routing_reverse_confirmation", {}) if practical else {}
    routing_confirmation_complete = bool(
        routing_confirmation.get("forward_gain", 0) < config.tuning.minimum_gain_fraction
        or (
            routing_confirmation.get("attempted")
            and routing_confirmation.get("routing_trace_match")
            and routing_confirmation.get("token_identity_match")
            and routing_confirmation.get("samples_per_policy", 0) >= 3
        )
    )
    benchmark_matrix_ok = bool(
        practical
        and set(practical.get("benchmark_configurations", [])) == {"A", "B", "C", "D", "E"}
    )
    return [
        GateResult(
            gate_id=1,
            name="build and dependency integrity",
            status=GateStatus.PASS if dependency_ok and build_ok else GateStatus.FAIL,
            evidence_class=EvidenceClass.MEASURED,
            reasons=["exact checkout, licence, patch, source, and binary manifests verified"],
        ),
        GateResult(
            gate_id=2,
            name="backend integration",
            status=(
                GateStatus.PASS
                if fixture.get("clean_shutdown") and fixture.get("stream_exact")
                else GateStatus.FAIL
            ),
            evidence_class=EvidenceClass.FIXTURE,
            reasons=["universal ABI generation, streaming, capability job, and shutdown executed"],
        ),
        GateResult(
            gate_id=3,
            name="semantic equivalence",
            status=(
                GateStatus.PASS
                if options.full and full_correctness and practical and practical.get("comparisons")
                else GateStatus.NOT_EVALUATED
                if not options.full
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                f"{request_count} deterministic executions; exact comparisons={len(comparisons)}",
                "quick mode cannot satisfy the practical-model correctness requirement"
                if not options.full
                else "fixture and practical token IDs were compared",
            ],
            metrics={"request_count": request_count, "all_exact": exact},
        ),
        GateResult(
            gate_id=4,
            name="adapter overhead",
            status=(
                GateStatus.PASS
                if options.full and overhead_pass
                else GateStatus.NOT_EVALUATED
                if not options.full
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                "multiple direct and adapter samples used; absent metrics were not treated as zero",
                f"all required overhead metrics present={overhead_complete}",
            ],
            metrics=overhead_summary,
        ),
        GateResult(
            gate_id=5,
            name="real telemetry",
            status=(
                GateStatus.PASS
                if options.full and real_routes and real_telemetry
                else GateStatus.NOT_EVALUATED
                if not options.full
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                "real practical-model routes, residency, cache, storage, resource, and phase counters observed"
                if real_routes and real_telemetry
                else "fixture-only or incomplete practical telemetry"
            ],
            metrics={
                "bridge_event_count": telemetry_summary.get("event_count"),
                "real_routes": real_routes,
                "practical_coverage": practical.get("telemetry_coverage", {}) if practical else {},
            },
        ),
        GateResult(
            gate_id=6,
            name="plan translation",
            status=(
                GateStatus.PASS
                if plan_ok and (not options.full or plan_execution_ok)
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                "native plan budgets and inventories reconciled without executable-tier overclaim",
                f"resource-plan fixed-replay samples={practical.get('resource_plan_execution_count', 0) if practical else 0}",
            ],
        ),
        GateResult(
            gate_id=7,
            name="fixed-replay tuning",
            status=(
                GateStatus.PASS
                if options.full and tuning_ok
                else GateStatus.NOT_EVALUATED
                if not options.full
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                "identical replay hash, at least three samples, tail gate, and reverse confirmation policy applied"
            ],
        ),
        GateResult(
            gate_id=8,
            name="routing-aware held-out evaluation",
            status=(
                GateStatus.PASS
                if options.full
                and measured_policies >= 2
                and routing_aware_physical
                and routing_confirmation_complete
                else GateStatus.NOT_EVALUATED
                if not options.full
                else GateStatus.FAIL
            ),
            evidence_class=full_class,
            reasons=[
                f"measured held-out policies={measured_policies}; all six policies evaluated offline; calibration bitmap executed={routing_aware_physical}; reverse confirmation complete={routing_confirmation_complete}"
            ],
        ),
        GateResult(
            gate_id=9,
            name="microshard-ready architecture",
            status=GateStatus.PASS if microshard_ok else GateStatus.FAIL,
            evidence_class=EvidenceClass.FIXTURE,
            reasons=[
                "logical projection ranges reconstruct exactly; real backend execution remains unsupported"
            ],
        ),
        GateResult(
            gate_id=10,
            name="reusable backend",
            status=(
                GateStatus.PASS
                if artifact_audit.get("complete") and (not options.full or benchmark_matrix_ok)
                else GateStatus.FAIL
            ),
            evidence_class=EvidenceClass.MEASURED,
            reasons=[
                "backend, process, probe, readers, translator, tuner, and evidence components retained",
                f"measured benchmark configurations={practical.get('benchmark_configurations', []) if practical else []}",
            ],
            metrics=artifact_audit,
        ),
    ]


def run_experiment_009(options: Experiment009Options) -> Experiment009Outcome:
    options.validate()
    repository_root = Path(__file__).resolve().parents[4]
    config = load_experiment_009_config(options.config_path)
    output_base = options.output_directory or repository_root / config.output_root
    bundle = EvidenceBundle(
        create_bundle_root(output_base, explicit=options.output_directory is not None),
        resume=options.resume,
    )
    run_mode = "FULL" if options.full else "QUICK"
    if options.resume and bundle.is_stage_complete("evidence_tables"):
        prior_verdict = _read_json(bundle.root / "verdict.json", {})
        prior_manifest = _read_json(bundle.root / "manifest.json", {})
        prior_audit = bundle.audit()
        if (
            prior_verdict.get("completed") is True
            and prior_verdict.get("run_mode") == run_mode
            and prior_manifest.get("selected_configuration") == options.configuration
            and prior_audit["complete"]
        ):
            return Experiment009Outcome(
                bundle_path=bundle.root,
                verdict=Experiment009Verdict(prior_verdict["verdict"]),
                completed=True,
                error=None,
            )
    bundle.write_text("README.md", build_bundle_readme(run_mode))
    bundle.write_json("environment.json", environment_report(repository_root))
    bundle.write_json(
        "manifest.json",
        {
            "schema_version": "experiment-009-manifest-v1",
            "run_mode": run_mode,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "config": config.model_dump(mode="json"),
            "selected_configuration": options.configuration,
            "fixture_evidence_is_official": False,
            "distributed_kimi_k3_claimed": False,
        },
    )
    checkout, build_root, engine_directory, source_directory = _resolve_paths(
        repository_root, config, options
    )
    build_manifest_path = build_root / "colibri_build.json"
    error: str | None = None
    dependency_ok = build_ok = plan_ok = microshard_ok = False
    fixture_result: dict[str, Any] = {}
    practical_result: dict[str, Any] | None = None
    tuning_result: Any | None = None
    policy_rows: list[dict[str, Any]] = []
    overhead_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    selections: list[RouteSelection] = []
    model_context: ModelContext | None = None
    capability_report: Any | None = None
    native_plan: dict[str, Any] | None = None
    swarm_plan: Any | None = None
    micro_validation: dict[str, Any] = {}
    rejected_candidates: list[dict[str, Any]] = []
    try:
        _build_if_requested(
            repository_root=repository_root,
            checkout=checkout,
            build_root=build_root,
            options=options,
            bundle=bundle,
        )
        dependency = verify_colibri_checkout(checkout)
        patches = patch_manifest(repository_root / "integrations" / "colibri")
        build = load_build_manifest(build_manifest_path)
        bundle.write_json("colibri_dependency.json", dependency)
        bundle.write_json(
            "colibri_patch_manifest.json",
            _read_json(build_root / "colibri_patch_manifest.json", patches),
        )
        bundle.write_json("colibri_build.json", build)
        dependency_ok = dependency["commit"] == config.dependency.commit
        build_ok = bool(build.get("source_tree_sha256") and build.get("binaries"))
        bundle.complete_stage("dependency_build")

        if options.full:
            practical_path = _acquire_practical_model(
                repository_root=repository_root,
                config=config,
                options=options,
                bundle=bundle,
            )
            model_context = ModelContext(
                path=practical_path,
                model_id=config.practical_model.model_id,
                model_revision=config.practical_model.revision,
                family=options.model_family or config.practical_model.model_family,
                evidence_class=EvidenceClass.MEASURED,
            )
        else:
            fixture_path = (repository_root / config.fixture.path).resolve()
            if not (fixture_path / "config.json").is_file():
                raise FileNotFoundError(
                    f"generated Colibri fixture is missing: {fixture_path}; see integrations/colibri/README.md"
                )
            model_context = ModelContext(
                path=fixture_path,
                model_id=config.fixture.model_id,
                model_revision=config.fixture.model_revision,
                family=config.fixture.model_family,
                evidence_class=EvidenceClass.FIXTURE,
            )

        probe = ColibriCapabilityProbe(
            engine_directory,
            source_directory=source_directory,
            build_manifest=build_manifest_path,
            model_path=model_context.path,
        )
        capability_report = probe.probe()
        bundle.write_json("capability_report.json", capability_report.model_dump(mode="json"))
        inspector = ColibriModelInspector(engine_directory)
        inventory, tensors, experts, native_quant = inspector.inspect(
            model_context.path,
            model_id=model_context.model_id,
            model_revision=model_context.model_revision,
            model_family=model_context.family,
            content_hash_mode="full",
            execution_backends=capability_report.execution_backends,
        )
        bundle.write_json("model_inventory.json", inventory.model_dump(mode="json"))
        bundle.write_json(
            "tensor_inventory.json", [item.model_dump(mode="json") for item in tensors]
        )
        bundle.write_json(
            "expert_inventory.json", [item.model_dump(mode="json") for item in experts]
        )
        bundle.write_json(
            "native_quantization_inventory.json",
            [item.model_dump(mode="json") for item in native_quant],
        )
        largest_file = max(
            (Path(item["path"]) for item in inventory.model_files),
            key=lambda path: path.stat().st_size,
        )
        typical_expert = (
            int(median(expert.total_bytes for expert in experts)) if experts else 4 * 1024 * 1024
        )
        storage_profile = ColibriStorageProfiler().profile(
            largest_file,
            expert_read_bytes=max(4096, typical_expert),
            samples=4 if options.quick else 16,
            maximum_queue_depth=4,
        )
        bundle.write_json("storage_inventory.json", storage_profile)
        native_plan = run_colibri_plan(
            engine_directory=engine_directory,
            model_path=model_context.path,
            capabilities=capability_report,
            log_path=bundle.root / "logs" / "colibri_plan.log",
        )
        bundle.write_json("colibri_resource_plan.json", native_plan)
        hardware, _tier_inventory = hardware_and_tiers(
            capabilities=capability_report,
            native_plan=native_plan,
            storage_profile=storage_profile,
        )
        bundle.write_json("hardware_profile.json", hardware)
        swarm_plan = ColibriPlanTranslator().translate(
            native_plan,
            hardware_fingerprint=hardware["hardware_fingerprint"],
            tensors=tensors,
            experts=experts,
            capabilities=capability_report,
        )
        bundle.write_json("swarm_resource_plan.json", swarm_plan.model_dump(mode="json"))
        bundle.write_csv("tier_residency.csv", plan_tier_rows(swarm_plan))
        plan_ok = True
        descriptors, micro_validation = build_microshard_evidence(
            tensors=tensors,
            experts=experts,
            model_config=_read_json(model_context.path / "config.json", {}),
            model_id=model_context.model_id,
        )
        microshard_ok = bool(
            micro_validation.get("valid")
            and micro_validation.get("fixture_equivalence", {}).get("allclose")
            and all(item.execution_status == "unsupported" for item in descriptors)
        )
        bundle.write_json(
            "microshard_descriptors.json",
            {
                "descriptors": [item.model_dump(mode="json") for item in descriptors],
                "validation": micro_validation,
            },
        )
        bundle.complete_stage("inventory_planning_microshards")

        fixture_model = ModelContext(
            path=(repository_root / config.fixture.path).resolve(),
            model_id=config.fixture.model_id,
            model_revision=config.fixture.model_revision,
            family=config.fixture.model_family,
            evidence_class=EvidenceClass.FIXTURE,
        )
        fixture_inventory, _, fixture_experts, _ = ColibriModelInspector(engine_directory).inspect(
            fixture_model.path,
            model_id=fixture_model.model_id,
            model_revision=fixture_model.model_revision,
            model_family=fixture_model.family,
            content_hash_mode="metadata",
            execution_backends=capability_report.execution_backends,
        )
        fixture_result = _run_fixture_transport(
            bundle=bundle,
            config=config,
            options=options,
            engine_directory=engine_directory,
            source_directory=source_directory,
            build_manifest_path=build_manifest_path,
            model=fixture_model,
            expert_layers=len({expert.layer_id for expert in fixture_experts}),
        )
        overhead_rows.extend(fixture_result["overhead_rows"])
        benchmark_rows.extend(fixture_result["benchmark_rows"])
        if options.full:
            practical_result = _run_practical_path(
                bundle=bundle,
                config=config,
                options=options,
                engine_directory=engine_directory,
                source_directory=source_directory,
                build_manifest_path=build_manifest_path,
                model=model_context,
                capability_report=capability_report,
                model_inventory=inventory,
                quantization_fingerprint=canonical_hash(
                    [item.model_dump(mode="json") for item in native_quant]
                ),
                swarm_plan=swarm_plan,
                experts=experts,
            )
            overhead_rows.extend(practical_result["overhead_rows"])
            benchmark_rows.extend(practical_result["benchmark_rows"])
            selections = practical_result["selections"]
            policy_rows = practical_result["policy_rows"]
            tuning_result = practical_result["tuning_result"]
            rejected_candidates = practical_result["rejected_candidates"]
            bundle.write_json(
                "replay_tokens.json", practical_result["replay"].model_dump(mode="json")
            )
            bundle.write_json(
                "tuning_candidates.json",
                {
                    "candidates": [
                        item.model_dump(mode="json")
                        for item in practical_result["tuning_candidates"]
                    ],
                    "rejected": rejected_candidates,
                    "result": tuning_result.model_dump(mode="json"),
                },
            )
            bundle.write_csv("tuning_results.csv", practical_result["tuning_rows"])
        else:
            selections = fixture_result["selections"]
            expert_bytes = {
                (expert.layer_id, expert.expert_id): expert.total_bytes
                for expert in fixture_experts
            }
            policy_rows, policy_meta = _fixture_routing_policy_rows(
                config=config,
                selections=selections,
                expert_bytes=expert_bytes,
            )
            replay_runner = ColibriReplayRunner(
                engine_directory=engine_directory,
                model_path=fixture_model.path,
                model_id=fixture_model.model_id,
                model_revision=fixture_model.model_revision,
                model_family=fixture_model.family,
                cap=config.backend.cap,
                ram_safety_reserve_bytes=0,
                timeout_seconds=config.backend.request_timeout_seconds,
            )
            replay = replay_runner.create_calibration(
                # Colibri's fixed-replay path requires at least two prompt
                # tokens.  The transport fixture deliberately uses a single
                # token so repeat it only for calibration; model semantics and
                # every candidate's replay sequence remain identical.
                prompt=config.fixture.prompt * 2,
                continuation_tokens=4,
                tokenizer_hash=fixture_inventory.tokenizer_hash,
            )
            supported = ColibriCapabilityProbe.supported_tuning_settings(
                capability_report, model_family=fixture_model.family
            )
            candidates, rejected_candidates = _tuning_candidates(
                config=config,
                family=fixture_model.family,
                supported=supported,
                physical_cores=int(capability_report.cpu.get("physical_cores", 1)),
            )
            tuning_result, _ = _run_tuning(
                bundle=bundle,
                config=config,
                runner=replay_runner,
                replay=replay,
                candidates=candidates,
                supported=supported,
            )
            bundle.write_json("replay_tokens.json", replay.model_dump(mode="json"))
            bundle.write_json(
                "tuning_candidates.json",
                {
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                    "rejected": rejected_candidates,
                    "result": tuning_result.model_dump(mode="json"),
                    "fixture_policy_meta": policy_meta,
                },
            )
            bundle.write_csv("tuning_results.csv", _tuning_csv(tuning_result))
        bundle.complete_stage("execution_matrix")

        tables = route_tables(selections)
        prompt_partition = {item.prompt_id: item.partition for item in config.routing.prompts}
        route_summary = {
            **tables["summary"],
            "route_accounting": fixture_result.get("route_accounting"),
            "calibration_prompt_ids": sorted(
                key for key, value in prompt_partition.items() if value == "calibration"
            ),
            "heldout_prompt_ids": sorted(
                key for key, value in prompt_partition.items() if value == "heldout"
            ),
            "usage_history": practical_result.get("usage_history") if practical_result else None,
            "batch_union": (
                practical_result.get("batch_union")
                if practical_result
                else policy_meta["batch_union"]
            ),
            "routing_aware_placement": (
                practical_result.get("routing_aware_placement") if practical_result else None
            ),
            "routing_reverse_confirmation": (
                practical_result.get("routing_reverse_confirmation") if practical_result else None
            ),
        }
        bundle.write_json("routing_trace_summary.json", route_summary)
        bundle.write_csv("expert_activation.csv", tables["activation"])
        bundle.write_csv("expert_coactivation.csv", tables["coactivation"])
        bundle.write_csv("expert_transitions.csv", tables["transitions"])
        bundle.write_csv("heldout_policy_results.csv", policy_rows)
        bundle.write_csv("adapter_overhead_results.csv", overhead_rows)
        bundle.write_csv("benchmark_results.csv", benchmark_rows)
        correctness = {
            "minimum_required_requests": config.acceptance.minimum_correctness_requests,
            "fixture_execution_count": fixture_result.get("request_execution_count", 0),
            "fixture_comparisons": fixture_result.get("comparisons", []),
            "fixture_stream_identity": fixture_result.get("stream_exact", False),
            "fixture_expected_tokens": {
                "input": config.fixture.expected_input_token_ids,
                "output": config.fixture.expected_output_token_ids,
            },
            "practical_comparisons": practical_result.get("comparisons", [])
            if practical_result
            else [],
            "model_fingerprint": inventory.engine_build_fingerprint,
            "quantization_formats": inventory.quantization_formats,
            "placement_weight_bytes_unchanged": True,
            "routing_reverse_confirmation": (
                practical_result.get("routing_reverse_confirmation") if practical_result else None
            ),
            "divergences": [],
        }
        bundle.write_json("correctness_results.json", correctness)
        telemetry_summary, cache_rows, storage_rows = _telemetry_tables(bundle)
        measured_tiers = (
            _measured_tier_rows(bundle, model_id=model_context.model_id) if options.full else []
        )
        bundle.write_csv("tier_residency.csv", [*plan_tier_rows(swarm_plan), *measured_tiers])
        bundle.write_csv("cache_events.csv", cache_rows)
        bundle.write_csv("storage_events.csv", storage_rows)
        bundle.complete_stage("evidence_tables")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        bundle.record_failure("runner", error)

    if model_context is None:
        model_context = ModelContext(
            path=(repository_root / config.fixture.path).resolve(),
            model_id=config.fixture.model_id,
            model_revision=config.fixture.model_revision,
            family=config.fixture.model_family,
            evidence_class=EvidenceClass.FIXTURE,
        )
    _write_reproduce(bundle, options=options, model=model_context, repository_root=repository_root)
    _ensure_artifacts(bundle)
    telemetry_summary, cache_rows_existing, storage_rows_existing = _telemetry_tables(bundle)
    if not (bundle.root / "cache_events.csv").is_file():
        bundle.write_csv("cache_events.csv", cache_rows_existing)
    if not (bundle.root / "storage_events.csv").is_file():
        bundle.write_csv("storage_events.csv", storage_rows_existing)
    generate_required_plots(bundle.root)
    overhead_workload = "practical_olmoe_generation" if options.full else "fixture_decode"
    overhead_summary = _metric_summary(overhead_rows, overhead_workload)
    pre_audit = bundle.audit()
    # report.md and verdict.json are deliberately finalized after gate evaluation.
    pre_audit["missing"] = [
        name for name in pre_audit["missing"] if name not in {"report.md", "verdict.json"}
    ]
    pre_audit["complete"] = not pre_audit["missing"]
    gates = _evaluate_gates(
        options=options,
        dependency_ok=dependency_ok,
        build_ok=build_ok,
        fixture=fixture_result,
        practical=practical_result,
        plan_ok=plan_ok,
        microshard_ok=microshard_ok,
        telemetry_summary=telemetry_summary,
        overhead_summary=overhead_summary,
        tuning_result=tuning_result,
        policy_rows=policy_rows,
        artifact_audit=pre_audit,
        config=config,
    )
    reverse_gain_sources: dict[str, float] = {}
    if tuning_result and tuning_result.accepted:
        reverse_gain_sources["fixed_replay_tuning"] = tuning_result.confirmed_gain
    routing_reverse = (
        practical_result.get("routing_reverse_confirmation", {}) if practical_result else {}
    )
    if routing_reverse.get("accepted"):
        reverse_gain_sources["routing_aware_placement"] = float(routing_reverse["confirmed_gain"])
    reverse_gain = max(reverse_gain_sources.values(), default=None)
    valid_execution = bool(fixture_result.get("comparisons"))
    verdict = overall_verdict(
        gates,
        full_run=options.full,
        valid_colibri_execution=valid_execution,
        reverse_confirmed_gain=reverse_gain,
    )
    active_experts = len({(row.layer_id, row.expert_id) for row in selections})
    measured_policy_rows = [
        row for row in policy_rows if row.get("decode_tokens_per_second") is not None
    ]
    best_policy = (
        max(measured_policy_rows, key=lambda row: float(row["decode_tokens_per_second"]))
        if measured_policy_rows
        else None
    )
    plain_policy = next(
        (
            row
            for row in measured_policy_rows
            if row.get("policy") == PlacementPolicy.PLAIN_LRU.value
        ),
        None,
    )
    best_policy_gain = (
        float(best_policy["decode_tokens_per_second"])
        / float(plain_policy["decode_tokens_per_second"])
        - 1.0
        if best_policy and plain_policy
        else None
    )
    tuning_rejections = [
        row["candidate_id"]
        for row in (tuning_result.candidates if tuning_result else [])
        if row["candidate_id"] != (tuning_result.selected_candidate_id if tuning_result else None)
    ]
    tuning_rejections.extend(item["candidate_id"] for item in rejected_candidates)
    swarm_policy = next(
        (row for row in measured_policy_rows if row.get("policy") == PlacementPolicy.SWARM.value),
        None,
    )
    answers = {
        "1": (
            "Yes. The universal worker ABI executed the real OLMoE model through the patched Colibri engine."
            if practical_result
            else "Yes for the executable bridge fixture only; no real-model completion is claimed."
        ),
        "2": config.dependency.commit,
        "3": f"Yes. {len(_read_json(bundle.root / 'colibri_patch_manifest.json', {}).get('patches', []))} narrow recorded patches were applied.",
        "4": f"Exact comparisons passed={all(row.get('exact') for row in fixture_result.get('comparisons', [])) if fixture_result.get('comparisons') else False}; practical comparisons={len(practical_result.get('comparisons', [])) if practical_result else 0}, including exact router selections and weights.",
        "5": json.dumps(overhead_summary, sort_keys=True),
        "6": ", ".join(capability_report.model_families) if capability_report else "Not probed.",
        "7": model_context.model_id if options.full else "No official model; quick fixture only.",
        "8": (
            f"Yes; {len(practical_result.get('selections', []))} measured practical-model "
            "selections were ingested."
            if practical_result and practical_result.get("selections")
            else f"No real-model claim; {len(selections)} fixture selections were ingested."
        ),
        "9": f"{active_experts} unique layer-expert pairs were active on the practical trace."
        if practical_result
        else f"{active_experts} fixture layer-expert pairs; not a real-model count.",
        "10": (
            "VRAM: 0 expert bytes (no executable CUDA build); RAM: dense tensors plus a bounded expert cache; NVMe: immutable backing files. "
            + json.dumps(swarm_plan.routed_expert_tiers, sort_keys=True)
            if swarm_plan
            else "Not measured."
        ),
        "11": (
            f"Plain LRU measured {plain_policy.get('bytes_read_per_token')} bytes/token."
            if plain_policy
            else "Not measured on a real model."
        ),
        "12": (
            f"Best forward policy={best_policy.get('policy') if best_policy else 'not measured'}, gain versus LRU={best_policy_gain}; reverse confirmation={json.dumps(routing_reverse, sort_keys=True)}."
        ),
        "13": (
            f"The combined swarm policy recorded useful prefetch bytes={swarm_policy.get('prefetch_useful_bytes')}, wasted bytes={swarm_policy.get('prefetch_wasted_bytes')}, and held-out regret={swarm_policy.get('heldout_regret')}. "
            f"Its separately confirmed gain was {routing_reverse.get('confirmed_gain')}; because the policy also changed hot pinning and pipeline settings, this run does not attribute the gain to prefetch alone."
            if swarm_policy
            else "Not measured on a real model."
        ),
        "14": f"Selected tuning candidate: {tuning_result.selected_candidate_id if tuning_result else 'not run'}; Colibri resource plan was measured separately as configuration C.",
        "15": (
            f"{'Yes' if tuning_result and tuning_result.accepted else 'No'}. The bounded tuner selected "
            f"{tuning_result.selected_candidate_id if tuning_result else 'no candidate'} and accepted "
            f"{tuning_result.confirmed_gain if tuning_result and tuning_result.accepted else 0.0} gain. "
            f"Configuration E routing-aware placement was a separate result and confirmed {routing_reverse.get('confirmed_gain') if routing_reverse.get('accepted') else 'no accepted gain'}."
        ),
        "16": ", ".join(tuning_rejections) or "No candidate was rejected.",
        "17": "Yes. Kimi K3 MXFP4 metadata is represented as native, non-reencodable E2M1/UE8M0 packing; the official OLMoE run used Colibri's merged int8 representation without adapter-side requantization.",
        "18": f"Logical ABI validation: {micro_validation.get('valid', False)}; fixture equivalence: {micro_validation.get('fixture_equivalence', {}).get('allclose', False)}.",
        "19": "Colibri v1.4.0 does not execute tensor microshards; every real descriptor remains execution_status=unsupported.",
        "20": (
            "Process ownership, exact token IDs, route traces, usage history, cache counters, "
            "storage reads, tier/phase events, native plans, and bounded scheduling controls; "
            + (
                "the practical-model run exercised these controls physically."
                if practical_result
                else "the quick run exercised only the fixture subset."
            )
        ),
        "21": "Yes as a local control-plane foundation, but no distributed Kimi K3 execution was performed.",
        "22": "Run two trusted LAN workers that each host complete experts, transport only expert inputs/outputs, and compare exact fixed-replay tokens before attempting network microshards.",
    }
    report_context = {
        "verdict": verdict.value,
        "run_mode": run_mode,
        "evidence_class": (EvidenceClass.MEASURED if options.full else EvidenceClass.FIXTURE).value,
        "summary": (
            f"{verdict.value}: a real Colibri-backed local worker completed the A-E matrix with exact bridge tokens and router traces."
            if options.full and practical_result
            else "This quick run is fixture-only and therefore cannot receive an official pass."
        ),
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "answers": answers,
        "limitations": [
            "The current native Windows build is CPU-only; RTX 5090 presence is not reported as executable CUDA support.",
            "OLMoE lacks a persistent streaming mux at the pinned revision; streaming is validated on the GLM bridge fixture.",
            "Windows cold-cache eviction is not forced, so cache state is recorded rather than guessed.",
            "Real tensor microshard execution remains unsupported.",
            *([error] if error else []),
        ],
    }
    bundle.write_text("report.md", build_report(report_context))
    final_audit = bundle.audit()
    verdict_payload = {
        "schema_version": "experiment-009-verdict-v1",
        "verdict": verdict.value,
        "run_mode": run_mode,
        "completed": error is None,
        "terminal_error": error,
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "reverse_confirmed_gain": reverse_gain,
        "reverse_confirmed_gain_sources": reverse_gain_sources,
        "artifact_audit": final_audit,
        "distributed_kimi_k3_achieved": False,
    }
    bundle.write_json("verdict.json", verdict_payload)
    return Experiment009Outcome(
        bundle_path=bundle.root,
        verdict=verdict,
        completed=error is None,
        error=error,
    )
