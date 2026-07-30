"""One process-isolated real-Qwen3 fan-out session."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from transformers import AutoTokenizer

from swarm_inference.config.models import Backend, ModelManifest
from swarm_inference.config.worker_fanout import FanoutExperimentConfig
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.exceptions import IntegrityError, TransportError
from swarm_inference.experiments.fanout_lifecycle import (
    LifecycleRecorder,
    lifecycle_duration_seconds,
    pipeline_ready_time_seconds,
    read_lifecycle_events,
)
from swarm_inference.experiments.fanout_resources import ResourceSampler
from swarm_inference.experiments.real_model import _verify_worker_proof
from swarm_inference.host import stop_process
from swarm_inference.protocol.messages import RoutePlan, SubmitRequest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class WorkerHandle:
    worker_id: str
    stage_id: int
    process: subprocess.Popen[str]
    endpoint: str
    stdout_handle: Any
    stderr_handle: Any
    stdout_path: Path
    stderr_path: Path
    lifecycle_path: Path
    lifecycle_recorder: LifecycleRecorder
    identity_path: Path
    shutdown_path: Path
    shard_root: Path
    runtime_process_id: int | None = None


@dataclass(slots=True)
class FanoutSessionResult:
    worker_count: int
    phase: str
    repeat: int
    passed: bool
    runnable_generation: bool
    failure_type: str | None
    failure_message: str | None
    pipeline_ready_seconds: float | None
    request_results: list[dict[str, Any]] = field(default_factory=list)
    worker_lifecycle_rows: list[dict[str, Any]] = field(default_factory=list)
    resource_rows: list[dict[str, Any]] = field(default_factory=list)
    worker_memory_rows: list[dict[str, Any]] = field(default_factory=list)
    gpu_process_memory_rows: list[dict[str, Any]] = field(default_factory=list)
    health_rows: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    transport_metrics: dict[str, Any] = field(default_factory=dict)
    cleanup: dict[str, Any] = field(default_factory=dict)
    rejoin: dict[str, Any] | None = None


def _numeric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in after.items():
        prior = before.get(key)
        if isinstance(value, (int, float)) and isinstance(prior, (int, float)):
            result[key] = value - prior
    return result


def _request_statistics(
    *,
    response: Any,
    metric: dict[str, Any],
    reference_ids: list[int],
    submitted_ns: int,
    completed_ns: int,
    transport_delta: dict[str, Any],
) -> dict[str, Any]:
    output_ids = [int(value) for value in response.output_token_ids]
    decode_ms = [
        float(item["total_token_latency_ms"])
        for item in metric.get("token_steps", [])[1:]
        if item.get("total_token_latency_ms") is not None
    ]
    token_identity = output_ids == reference_ids[: len(output_ids)] and len(output_ids) == len(
        reference_ids
    )
    first_mismatch: int | None = None
    for index, (actual, expected) in enumerate(zip(output_ids, reference_ids, strict=False)):
        if actual != expected:
            first_mismatch = index
            break
    if first_mismatch is None and len(output_ids) != len(reference_ids):
        first_mismatch = min(len(output_ids), len(reference_ids))
    end_to_end = float(response.end_to_end_s or 0.0)
    return {
        "status": response.status,
        "response_detail": response.detail,
        "verified": bool(response.verified),
        "prompt_token_count": None,
        "output_token_count": len(output_ids),
        "distributed_token_ids": output_ids,
        "reference_token_ids": reference_ids,
        "token_identity": token_identity,
        "first_mismatching_token": first_mismatch,
        "ttft_seconds": response.time_to_first_token_s,
        "prefill_latency_seconds": (
            float(metric["token_steps"][0]["total_token_latency_ms"]) / 1000
            if metric.get("token_steps")
            else None
        ),
        "median_decode_token_latency_seconds": (
            statistics.median(decode_ms) / 1000 if decode_ms else None
        ),
        "p95_decode_token_latency_seconds": (
            float(np.percentile(np.asarray(decode_ms, dtype=np.float64), 95)) / 1000
            if decode_ms
            else None
        ),
        "end_to_end_latency_seconds": end_to_end,
        "output_tokens_per_second": (len(output_ids) / end_to_end if end_to_end > 0 else 0.0),
        "request_submitted_monotonic_ns": submitted_ns,
        "request_completed_monotonic_ns": completed_ns,
        "stage_execution_seconds": float(metric.get("stage_execution_s", 0.0)),
        "worker_queue_seconds": float(metric.get("queue_s", 0.0)),
        "direct_hop_transfer_seconds": float(metric.get("transport_s", 0.0)),
        "route_reservation_time_ms": float(metric.get("route_reservation_time_ms", 0.0)),
        "route_generation": int(metric.get("route_generation", 0)),
        "retry_count": int(metric.get("retry_count", 0)),
        "route_changes": int(metric.get("route_changes", 0)),
        "per_stage": metric.get("per_stage", []),
        "token_steps": metric.get("token_steps", []),
        "transport_delta": transport_delta,
        "passed": (response.status == "completed" and bool(response.verified) and token_identity),
    }


def _hardlink_boundary_alias(
    *,
    boundary_root: Path,
    reference_request_id: str,
    request_id: str,
) -> None:
    if request_id == reference_request_id:
        return
    source = boundary_root / reference_request_id
    destination = boundary_root / request_id
    if destination.exists():
        return
    destination.mkdir(parents=True)
    for source_file in source.glob("*.npy"):
        destination_file = destination / source_file.name
        try:
            os.link(source_file, destination_file)
        except OSError:
            shutil.copy2(source_file, destination_file)


async def run_fanout_session(
    *,
    config: FanoutExperimentConfig,
    experiment_id: str,
    origin_monotonic_ns: int,
    worker_count: int,
    phase: str,
    repeat: int,
    manifest: ModelManifest,
    architecture_config: dict[str, Any],
    model_path: Path,
    shard_root: Path,
    session_root: Path,
    requests: list[dict[str, Any]],
    reference_by_id: dict[str, dict[str, Any]],
    stage_local_warmup: bool,
    pipeline_warmup: bool,
    pipeline_warmup_request: dict[str, Any] | None,
    boundary_root: Path | None = None,
    boundary_enabled: bool = False,
    hot_idle_seconds: list[float] | None = None,
    shard_roots_by_stage: dict[int, Path] | None = None,
    acquisition_events_by_stage: dict[int, dict[str, Any]] | None = None,
    rejoin_stage_id: int | None = None,
    rejoin_after_tokens: int = 4,
) -> FanoutSessionResult:
    """Launch exactly one worker per stage, run requests, and tear everything down."""

    if len(manifest.stages) != worker_count:
        raise ValueError("session worker count must equal manifest stage count")
    session_root.mkdir(parents=True, exist_ok=True)
    logs_root = session_root / "logs"
    lifecycle_root = session_root / "lifecycle"
    logs_root.mkdir(parents=True, exist_ok=True)
    lifecycle_root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )
    largest_stage = max(stage.required_memory_bytes for stage in manifest.stages)
    if worker_count == 1:
        logical_weight_limit = largest_stage + 256 * 1024 * 1024
    else:
        available_gap = manifest.total_weight_bytes - largest_stage
        if available_gap <= 1:
            raise IntegrityError("multi-stage layout leaves no fail-closed worker memory cap")
        logical_weight_limit = largest_stage + max(
            1,
            min(128 * 1024 * 1024, available_gap // 2),
        )
        logical_weight_limit = min(logical_weight_limit, manifest.total_weight_bytes - 1)
    logical_total_limit = (
        max(
            stage.required_total_memory_bytes or stage.required_memory_bytes
            for stage in manifest.stages
        )
        + 512 * 1024 * 1024
    )
    runtime = config.runtime_config(
        worker_count=worker_count,
        model_layer_count=manifest.layer_count,
        model_hidden_size=manifest.hidden_size,
        logical_weight_limit_bytes=logical_weight_limit,
    )
    worker_ids = [
        f"fanout-{worker_count:02d}-{phase}-r{repeat:02d}-stage-{stage_id:03d}"
        for stage_id in range(worker_count)
    ]
    affinity = {worker_id: stage_id for stage_id, worker_id in enumerate(worker_ids)}
    core = CoordinatorCore(
        config=runtime,
        model_manifest=manifest,
        architecture_config=architecture_config,
        runtime_dtype=config.model.dtype,
        tokenizer=tokenizer,
        worker_stage_affinity=affinity,
    )
    server = CoordinatorRpcServer(core)
    port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{port}"
    handles: list[WorkerHandle] = []
    lifecycle_paths: list[Path] = []
    handle_by_stage: dict[int, WorkerHandle] = {}
    source_root = Path(__file__).resolve().parents[2]
    session_worker_origin_ns = time.monotonic_ns()
    acquisition_events_by_stage = acquisition_events_by_stage or {}
    shard_roots_by_stage = shard_roots_by_stage or {}

    def active_worker_process_ids() -> list[int]:
        process_ids: list[int] = []
        for handle in handles:
            if handle.process.poll() is not None:
                continue
            process_ids.append(handle.process.pid)
            if handle.runtime_process_id is None:
                events = read_lifecycle_events([handle.lifecycle_path])
                python_entry = next(
                    (
                        event
                        for event in events
                        if event.get("event_name") == "python_entry_started"
                    ),
                    None,
                )
                if python_entry is not None:
                    handle.runtime_process_id = int(python_entry["process_id"])
            if (
                handle.runtime_process_id is not None
                and handle.runtime_process_id != handle.process.pid
            ):
                process_ids.append(handle.runtime_process_id)
        return list(dict.fromkeys(process_ids))

    sampler = ResourceSampler(
        phase=phase,
        worker_count=worker_count,
        repeat=repeat,
        coordinator_pid=os.getpid(),
        worker_pids=active_worker_process_ids,
        interval_seconds=1.0,
    )
    sampler.start()
    event_loop_lag_ms: list[float] = []
    lag_monitor_stop = asyncio.Event()

    async def monitor_event_loop_lag() -> None:
        interval_seconds = 0.1
        while not lag_monitor_stop.is_set():
            before = asyncio.get_running_loop().time()
            await asyncio.sleep(interval_seconds)
            elapsed = asyncio.get_running_loop().time() - before
            event_loop_lag_ms.append(max(0.0, elapsed - interval_seconds) * 1000)

    lag_monitor_task = asyncio.create_task(
        monitor_event_loop_lag(),
        name=f"fanout-event-loop-lag-{worker_count}-{phase}-{repeat}",
    )
    request_results: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    failure_type: str | None = None
    failure_message: str | None = None
    pipeline_ready: float | None = None
    rejoin_metrics: dict[str, Any] | None = None
    pipeline_warmup_emitted = False

    async def spawn_worker(
        *,
        stage_id: int,
        worker_id: str,
        suffix: str = "",
        worker_shard_root: Path | None = None,
    ) -> WorkerHandle:
        stage = manifest.stages[stage_id]
        lifecycle_path = lifecycle_root / f"stage-{stage_id:03d}{suffix}.jsonl"
        lifecycle_paths.append(lifecycle_path)
        identity_path = logs_root / f"stage-{stage_id:03d}{suffix}.pem"
        shutdown_path = logs_root / f"stage-{stage_id:03d}{suffix}.shutdown"
        stdout_path = logs_root / f"stage-{stage_id:03d}{suffix}.stdout.log"
        stderr_path = logs_root / f"stage-{stage_id:03d}{suffix}.stderr.log"
        endpoint = f"127.0.0.1:{_free_port()}"
        override = acquisition_events_by_stage.get(stage_id)
        assignment_ns = int(override["assignment_created_ns"]) if override else time.monotonic_ns()
        acquisition_started_ns = (
            int(override["acquisition_started_ns"]) if override else time.monotonic_ns()
        )
        acquisition_completed_ns = (
            int(override["acquisition_completed_ns"]) if override else time.monotonic_ns()
        )
        spawn_started_ns = time.monotonic_ns()
        acquisition_details = (
            dict(override.get("details", {}))
            if override
            else {
                "node_state": "cached-cold",
                "source": "local-stage-cache",
                "measurement_class": "local-shard-cache-hit",
            }
        )
        pre_spawn_recorder = LifecycleRecorder(
            path=lifecycle_path,
            experiment_id=experiment_id,
            worker_id=worker_id,
            stage_id=stage_id,
            origin_monotonic_ns=origin_monotonic_ns,
            process_id=os.getpid(),
        )
        pre_spawn_recorder.emit(
            "assignment_created",
            monotonic_ns=assignment_ns,
            details={"phase": phase, "repeat": repeat},
        )
        pre_spawn_recorder.emit(
            "shard_acquisition_started",
            monotonic_ns=acquisition_started_ns,
            bytes_count=stage.required_memory_bytes,
            details=acquisition_details,
        )
        pre_spawn_recorder.emit(
            "shard_acquisition_completed",
            monotonic_ns=acquisition_completed_ns,
            duration_ns=max(0, acquisition_completed_ns - acquisition_started_ns),
            bytes_count=stage.required_memory_bytes,
            details=acquisition_details,
        )
        pre_spawn_recorder.emit(
            "process_spawn_started",
            monotonic_ns=spawn_started_ns,
            details={"process_id_pending": True},
        )
        environment = os.environ.copy()
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        environment["PYTHONPATH"] = (
            str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
        )
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["SWARM_EXPERIMENT_ID"] = experiment_id
        environment["SWARM_EXPERIMENT_ORIGIN_NS"] = str(origin_monotonic_ns)
        environment["SWARM_WORKER_ID"] = worker_id
        environment["SWARM_STAGE_ID"] = str(stage_id)
        environment["SWARM_LIFECYCLE_FILE"] = str(lifecycle_path)
        if boundary_root is not None and boundary_enabled:
            environment["SWARM_REFERENCE_BOUNDARY_ROOT"] = str(boundary_root)
            environment["SWARM_BOUNDARY_ATOL"] = str(config.correctness.boundary_atol)
            environment["SWARM_BOUNDARY_RTOL"] = str(config.correctness.boundary_rtol)
            environment["SWARM_BOUNDARY_MINIMUM_COSINE"] = str(
                config.correctness.minimum_cosine_similarity
            )
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "swarm_inference.worker.process_main",
            "--coordinator",
            coordinator_endpoint,
            "--listen",
            endpoint,
            "--advertise",
            endpoint,
            "--backend",
            Backend.TORCH_CUDA.value,
            "--memory-limit-bytes",
            str(logical_weight_limit),
            "--total-memory-limit-bytes",
            str(logical_total_limit),
            "--identity",
            str(identity_path),
            "--shutdown-file",
            str(shutdown_path),
            "--worker-id",
            worker_id,
            "--queue-capacity",
            "64",
            "--max-microbatch-size",
            "1",
            "--max-microbatch-wait-ms",
            "0",
            "--model-shard-root",
            str(worker_shard_root or shard_root),
            "--outbound-queue-capacity",
            "64",
            "--inbound-queue-capacity",
            "64",
            "--max-inflight-operations",
            "4",
            "--warmup-sequence-length",
            str(config.workloads.warm.input_tokens_approx),
        ]
        if stage_local_warmup:
            command.append("--stage-local-warmup")
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                creationflags=creation_flags,
            )
        except Exception as exc:
            pre_spawn_recorder.emit(
                "process_spawned",
                error=f"{type(exc).__name__}: {exc}",
                details={"spawn_succeeded": False, "endpoint": endpoint},
            )
            stdout_handle.close()
            stderr_handle.close()
            raise
        recorder = LifecycleRecorder(
            path=lifecycle_path,
            experiment_id=experiment_id,
            worker_id=worker_id,
            stage_id=stage_id,
            origin_monotonic_ns=origin_monotonic_ns,
            process_id=process.pid,
        )
        recorder.emit(
            "process_spawned",
            details={
                "spawn_method": "subprocess-spawn",
                "spawn_succeeded": True,
                "endpoint": endpoint,
            },
        )
        handle = WorkerHandle(
            worker_id=worker_id,
            stage_id=stage_id,
            process=process,
            endpoint=endpoint,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            lifecycle_path=lifecycle_path,
            lifecycle_recorder=recorder,
            identity_path=identity_path,
            shutdown_path=shutdown_path,
            shard_root=worker_shard_root or shard_root,
        )
        handles.append(handle)
        handle_by_stage[stage_id] = handle
        return handle

    async def wait_until_routable(
        expected_worker_ids: set[str],
        *,
        timeout_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            failed = [
                {
                    "worker_id": handle.worker_id,
                    "process_id": handle.process.pid,
                    "exit_code": handle.process.poll(),
                }
                for handle in handles
                if handle.worker_id in expected_worker_ids and handle.process.poll() is not None
            ]
            if failed:
                raise IntegrityError(f"worker exited before becoming routable: {failed}")
            events = read_lifecycle_events(
                handle.lifecycle_path
                for handle in handles
                if handle.worker_id in expected_worker_ids
            )
            by_worker = {handle.worker_id: handle for handle in handles}
            for event in events:
                if event.get("event_name") != "python_entry_started":
                    continue
                handle = by_worker.get(str(event.get("worker_id")))
                if handle is not None:
                    handle.runtime_process_id = int(event["process_id"])
            ready = {
                str(row["worker_id"])
                for row in events
                if row.get("event_name") == "worker_routable"
            }
            now_ns = time.monotonic_ns()
            for handle in handles:
                if handle.worker_id not in expected_worker_ids or handle.worker_id in ready:
                    continue
                spawn_started = next(
                    (
                        int(event["monotonic_timestamp_ns"])
                        for event in events
                        if event.get("worker_id") == handle.worker_id
                        and event.get("event_name") == "process_spawn_started"
                    ),
                    None,
                )
                if (
                    spawn_started is not None
                    and (now_ns - spawn_started) / 1_000_000_000
                    > config.resource_limits.max_worker_start_seconds
                ):
                    raise TimeoutError(
                        f"worker {handle.worker_id} exceeded "
                        f"{config.resource_limits.max_worker_start_seconds}s startup limit"
                    )
            if expected_worker_ids <= ready:
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(
            f"workers did not become routable within {timeout_seconds}s: "
            f"{sorted(expected_worker_ids)}"
        )

    async def health_snapshot() -> list[dict[str, Any]]:
        workers = sorted(core.registry.workers(), key=lambda item: item.worker_id)

        async def one(worker: Any) -> dict[str, Any] | None:
            if worker.endpoint is None:
                return None
            health = await core.transport.health(worker.endpoint)
            return {
                "worker_id": worker.worker_id,
                "endpoint": worker.endpoint,
                "loaded_stages": health.loaded_stages,
                "healthy": health.healthy,
                "proof": health.proof,
                "proof_verified": _verify_worker_proof(health.proof),
            }

        snapshots = await asyncio.gather(*(one(worker) for worker in workers))
        return [item for item in snapshots if item is not None]

    async def execute_request(spec: dict[str, Any]) -> dict[str, Any]:
        request_id = str(spec["request_id"])
        reference_id = str(spec["reference_id"])
        if boundary_root is not None and boundary_enabled:
            _hardlink_boundary_alias(
                boundary_root=boundary_root,
                reference_request_id=reference_id,
                request_id=request_id,
            )
        reference = reference_by_id[reference_id]
        expected_values = reference.get("generated_token_ids", reference.get("token_ids", []))
        expected = [int(value) for value in expected_values][: int(spec["max_new_tokens"])]
        before = dict(core.runtime_transport_metrics)
        submitted_ns = time.monotonic_ns()
        response = await core.submit(
            SubmitRequest(
                request_id=request_id,
                prompt_token_ids=[int(value) for value in spec["prompt_token_ids"]],
                max_new_tokens=int(spec["max_new_tokens"]),
                random_seed=0,
                model_id=manifest.model_id,
                model_revision=manifest.model_revision,
            )
        )
        completed_ns = time.monotonic_ns()
        after = dict(core.runtime_transport_metrics)
        metric = next(
            item for item in reversed(core.request_metrics) if item["request_id"] == request_id
        )
        result = _request_statistics(
            response=response,
            metric=metric,
            reference_ids=expected,
            submitted_ns=submitted_ns,
            completed_ns=completed_ns,
            transport_delta=_numeric_delta(after, before),
        )
        result.update(
            {
                "request_id": request_id,
                "reference_id": reference_id,
                "worker_count": worker_count,
                "phase": phase,
                "repeat": repeat,
                "variant": spec.get("variant"),
                "concurrency": int(spec.get("concurrency", 1)),
                "idle_seconds": spec.get("idle_seconds"),
                "prompt_token_count": len(spec["prompt_token_ids"]),
            }
        )
        route_event = next(
            (
                event
                for event in reversed(core.events)
                if event.get("event_type") == "route_reserved"
                and event.get("request_id") == request_id
            ),
            None,
        )
        current_lifecycle = read_lifecycle_events(handle.lifecycle_path for handle in handles)
        first_stage_started = min(
            (
                int(row["monotonic_timestamp_ns"])
                for row in current_lifecycle
                if row.get("event_name") == "request_stage_operation_started"
                and row.get("request_id") == request_id
                and int(row.get("stage_id", -1)) == 0
            ),
            default=None,
        )
        route_assignment_ns = (
            int(route_event["timestamp_monotonic_ns"])
            if isinstance(route_event, dict)
            and route_event.get("timestamp_monotonic_ns") is not None
            else None
        )
        result["route_assignment_monotonic_ns"] = route_assignment_ns
        result["first_stage_operation_started_monotonic_ns"] = first_stage_started
        result["hot_time_to_contribution_seconds"] = (
            (first_stage_started - route_assignment_ns) / 1_000_000_000
            if first_stage_started is not None
            and route_assignment_ns is not None
            and first_stage_started >= route_assignment_ns
            else None
        )
        first_step = metric.get("token_steps", [None])[0]
        if isinstance(first_step, dict):
            first_token_ns = int(first_step["token_produced_monotonic_ns"])
            final_worker = handle_by_stage[max(handle_by_stage)].worker_id
            for handle in handles:
                if handle.process.poll() is None:
                    worker_entry_ns = next(
                        (
                            int(row["monotonic_timestamp_ns"])
                            for row in current_lifecycle
                            if row.get("worker_id") == handle.worker_id
                            and row.get("event_name") == "python_entry_started"
                        ),
                        first_token_ns,
                    )
                    joined_after_first_token = worker_entry_ns > first_token_ns
                    handle.lifecycle_recorder.emit_once(
                        "first-token-produced",
                        "first_token_produced",
                        monotonic_ns=completed_ns if joined_after_first_token else first_token_ns,
                        details={
                            "request_id": request_id,
                            "producer_worker_id": final_worker,
                            "observed_pipeline_milestone": handle.stage_id != max(handle_by_stage),
                            "observed": not joined_after_first_token,
                            "not_applicable": joined_after_first_token,
                            "reason": (
                                "replacement joined after the pipeline's first output token"
                                if joined_after_first_token
                                else None
                            ),
                        },
                    )
        return result

    async def perform_pipeline_warmup() -> None:
        nonlocal pipeline_warmup_emitted
        started_ns = time.monotonic_ns()
        for handle in handles:
            if handle.process.poll() is None:
                handle.lifecycle_recorder.emit(
                    "pipeline_warmup_started",
                    monotonic_ns=started_ns,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": True,
                    },
                )
        if pipeline_warmup_request is None:
            raise ValueError("pipeline warmup was requested without a warmup request")
        warmup_result = await execute_request(pipeline_warmup_request)
        if not warmup_result["passed"]:
            raise IntegrityError("full-pipeline warmup did not match the independent reference")
        completed_ns = time.monotonic_ns()
        for handle in handles:
            if handle.process.poll() is None:
                handle.lifecycle_recorder.emit(
                    "pipeline_warmup_completed",
                    monotonic_ns=completed_ns,
                    duration_ns=completed_ns - started_ns,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": True,
                        "generated_tokens": int(pipeline_warmup_request["max_new_tokens"]),
                        "warmup_request_caches_cleared": True,
                    },
                )
        pipeline_warmup_emitted = True

    try:
        for stage_id, worker_id in enumerate(worker_ids):
            await spawn_worker(
                stage_id=stage_id,
                worker_id=worker_id,
                worker_shard_root=shard_roots_by_stage.get(stage_id),
            )
        await wait_until_routable(
            set(worker_ids),
            timeout_seconds=config.resource_limits.max_pipeline_ready_seconds,
        )
        await core.wait_for_coverage(
            minimum_replicas=1,
            timeout_s=config.resource_limits.max_pipeline_ready_seconds,
        )
        lifecycle_rows = read_lifecycle_events(handle.lifecycle_path for handle in handles)
        pipeline_ready = pipeline_ready_time_seconds(
            lifecycle_rows,
            experiment_worker_start_origin_ns=session_worker_origin_ns,
            required_worker_ids=worker_ids,
        )
        if not phase.startswith("cold_"):
            health_rows = await health_snapshot()
        if pipeline_warmup:
            await perform_pipeline_warmup()
        else:
            skipped_ns = time.monotonic_ns()
            for handle in handles:
                handle.lifecycle_recorder.emit(
                    "pipeline_warmup_started",
                    monotonic_ns=skipped_ns,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": False,
                    },
                )
                handle.lifecycle_recorder.emit(
                    "pipeline_warmup_completed",
                    monotonic_ns=skipped_ns,
                    duration_ns=0,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": False,
                    },
                )
            pipeline_warmup_emitted = True

        rejoin_triggered = False
        if rejoin_stage_id is not None:
            original = handle_by_stage[rejoin_stage_id]
            original_runtime_pid = original.runtime_process_id or original.process.pid
            rejoin_metrics = {
                "failure_stage_id": rejoin_stage_id,
                "old_worker_id": original.worker_id,
                "old_process_id": original_runtime_pid,
                "committed_tokens_before_failure": rejoin_after_tokens,
            }

            async def after_token(payload: dict[str, Any]) -> RoutePlan | None:
                nonlocal rejoin_triggered
                if rejoin_triggered or int(payload["output_position"]) + 1 != rejoin_after_tokens:
                    return None
                rejoin_triggered = True
                failure_issued_ns = time.monotonic_ns()
                original.lifecycle_recorder.emit(
                    "worker_shutdown_started",
                    monotonic_ns=failure_issued_ns,
                    details={
                        "shutdown_mode": "experiment-injected-process-termination",
                        "runtime_process_id": original_runtime_pid,
                        "graceful": False,
                    },
                )
                runtime_process = psutil.Process(original_runtime_pid)
                runtime_process.terminate()
                await asyncio.to_thread(runtime_process.wait, 30)
                await asyncio.to_thread(original.process.wait, 30)
                detected_ns = time.monotonic_ns()
                original.lifecycle_recorder.emit(
                    "worker_shutdown_completed",
                    monotonic_ns=detected_ns,
                    duration_ns=detected_ns - failure_issued_ns,
                    details={
                        "shutdown_mode": "experiment-injected-process-termination",
                        "runtime_process_id": original_runtime_pid,
                        "graceful": False,
                        "process_genuinely_terminated": True,
                    },
                )
                core.remove_worker(original.worker_id)
                replacement_id = original.worker_id + "-replacement"
                core.worker_stage_affinity[replacement_id] = rejoin_stage_id
                restart_started_ns = time.monotonic_ns()
                replacement = await spawn_worker(
                    stage_id=rejoin_stage_id,
                    worker_id=replacement_id,
                    suffix="-replacement",
                    worker_shard_root=shard_roots_by_stage.get(rejoin_stage_id),
                )
                await wait_until_routable(
                    {replacement_id},
                    timeout_seconds=config.resource_limits.max_worker_start_seconds,
                )
                await core.wait_for_coverage(
                    minimum_replicas=1,
                    timeout_s=config.resource_limits.max_pipeline_ready_seconds,
                )
                routable_ns = time.monotonic_ns()
                replacement.lifecycle_recorder.emit(
                    "pipeline_warmup_started",
                    monotonic_ns=routable_ns,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": False,
                        "not_applicable": True,
                        "reason": (
                            "replacement joined after full-pipeline warmup and reconstructs "
                            "state through cache replay"
                        ),
                    },
                )
                replacement.lifecycle_recorder.emit(
                    "pipeline_warmup_completed",
                    monotonic_ns=routable_ns,
                    duration_ns=0,
                    details={
                        "warmup_level": "full-pipeline",
                        "warmup_performed": False,
                        "not_applicable": True,
                        "reason": (
                            "replacement joined after full-pipeline warmup and reconstructs "
                            "state through cache replay"
                        ),
                    },
                )
                replacement_runtime_pid = replacement.runtime_process_id or replacement.process.pid
                rejoin_metrics.update(
                    {
                        "failure_issued_monotonic_ns": failure_issued_ns,
                        "failure_detected_monotonic_ns": detected_ns,
                        "failure_detection_seconds": (detected_ns - failure_issued_ns)
                        / 1_000_000_000,
                        "restart_started_monotonic_ns": restart_started_ns,
                        "replacement_routable_monotonic_ns": routable_ns,
                        "restart_to_routable_seconds": (routable_ns - restart_started_ns)
                        / 1_000_000_000,
                        "total_unavailable_seconds": (routable_ns - failure_issued_ns)
                        / 1_000_000_000,
                        "new_worker_id": replacement.worker_id,
                        "new_process_id": replacement_runtime_pid,
                        "new_pid": replacement_runtime_pid != original_runtime_pid,
                        "worker_process_genuinely_terminated": (
                            not psutil.pid_exists(original_runtime_pid)
                            and original.process.poll() is not None
                        ),
                    }
                )
                recovered_route = await core._recover_direct_route(
                    submission=payload["submission"],
                    request_state=payload["request_state"],
                    route=payload["route_plan"],
                    metrics=payload["metrics"],
                    committed_through_token_position=int(payload["token_position"]),
                    failure=TransportError("experiment-injected intermediate worker termination"),
                    known_failed_stage_ids=[rejoin_stage_id],
                )
                recovery_completed_ns = time.monotonic_ns()
                rejoin_metrics.update(
                    {
                        "recovery_completed_monotonic_ns": recovery_completed_ns,
                        "total_unavailable_seconds": (recovery_completed_ns - failure_issued_ns)
                        / 1_000_000_000,
                    }
                )
                return recovered_route

            core.after_token_hook = after_token

        if hot_idle_seconds is not None:
            if len(requests) != len(hot_idle_seconds):
                raise ValueError("hot-idle requests must match configured idle durations")
            for idle_seconds, request in zip(hot_idle_seconds, requests, strict=True):
                await asyncio.sleep(idle_seconds)
                result = await execute_request(request)
                result["idle_seconds"] = idle_seconds
                request_results.append(result)
        else:
            concurrency_groups: dict[str, list[dict[str, Any]]] = {}
            sequential: list[dict[str, Any]] = []
            for request in requests:
                group = request.get("concurrency_group")
                if group is None:
                    sequential.append(request)
                else:
                    concurrency_groups.setdefault(str(group), []).append(request)
            for request in sequential:
                request_results.append(await execute_request(request))
            for group, members in sorted(concurrency_groups.items()):
                group_started = time.monotonic_ns()
                group_results = await asyncio.gather(
                    *(execute_request(request) for request in members)
                )
                group_completed = time.monotonic_ns()
                aggregate_tokens = sum(
                    int(result["output_token_count"])
                    for result in group_results
                    if result["passed"]
                )
                aggregate_tps = (
                    aggregate_tokens / ((group_completed - group_started) / 1_000_000_000)
                    if group_completed > group_started
                    else 0.0
                )
                for result in group_results:
                    result["concurrency_group"] = group
                    result["aggregate_verified_tokens_per_second"] = aggregate_tps
                request_results.extend(group_results)
        health_rows = await health_snapshot()
        if rejoin_metrics is not None:
            recovered = [
                event for event in core.events if event.get("event_type") == "stage_recovered"
            ]
            if recovered:
                event = recovered[-1]
                rejoin_metrics.update(
                    {
                        "route_generation_before": event.get("old_route_generation"),
                        "route_generation_after": event.get("route_generation"),
                        "route_generation_incremented": int(event.get("route_generation", 0))
                        > int(event.get("old_route_generation", 0)),
                        "cache_replay_seconds": event.get("replay_duration_s"),
                        "cache_replay_bytes": event.get("replay_bytes"),
                        "cache_replay_occurred": int(event.get("replay_bytes", 0)) > 0,
                        "lost_work_tokens": 0,
                    }
                )
            rejoin_metrics["exact_token_identity"] = bool(
                request_results and all(item["token_identity"] for item in request_results)
            )
            request_metric = next(
                (
                    item
                    for item in reversed(core.request_metrics)
                    if item.get("request_id")
                    == (request_results[-1].get("request_id") if request_results else None)
                ),
                None,
            )
            if isinstance(request_metric, dict):
                token_steps = request_metric.get("token_steps", [])
                if len(token_steps) > rejoin_after_tokens:
                    resumed_ns = int(
                        token_steps[rejoin_after_tokens]["token_produced_monotonic_ns"]
                    )
                    rejoin_metrics["generation_resumed_monotonic_ns"] = resumed_ns
                    rejoin_metrics["resumption_time_seconds"] = (
                        resumed_ns
                        - int(rejoin_metrics.get("failure_issued_monotonic_ns", resumed_ns))
                    ) / 1_000_000_000
            rejoin_metrics["status"] = (
                "PASS"
                if rejoin_metrics.get("new_pid")
                and rejoin_metrics.get("worker_process_genuinely_terminated")
                and rejoin_metrics.get("route_generation_incremented")
                and rejoin_metrics.get("cache_replay_occurred")
                and rejoin_metrics["exact_token_identity"]
                else "FAIL"
            )
    except Exception as exc:
        failure_type = type(exc).__name__
        failure_message = str(exc)
        with suppress(Exception):
            health_rows = await health_snapshot()
    finally:
        if not pipeline_warmup_emitted:
            skipped_ns = time.monotonic_ns()
            for handle in handles:
                if handle.process.poll() is None:
                    handle.lifecycle_recorder.emit(
                        "pipeline_warmup_started",
                        monotonic_ns=skipped_ns,
                        details={"warmup_performed": False, "aborted_before_warmup": True},
                    )
                    handle.lifecycle_recorder.emit(
                        "pipeline_warmup_completed",
                        monotonic_ns=skipped_ns,
                        duration_ns=0,
                        details={"warmup_performed": False, "aborted_before_warmup": True},
                    )
        if not requests and failure_type is None:
            # Load-only sessions deliberately perform no request. Emit explicit,
            # machine-readable not-applicable milestones so complete lifecycle
            # schemas do not masquerade as observed model work.
            not_applicable_ns = time.monotonic_ns()
            for handle in handles:
                observed_names = {
                    str(event.get("event_name"))
                    for event in read_lifecycle_events([handle.lifecycle_path])
                }
                for event_name in (
                    "first_request_received",
                    "first_stage_operation_started",
                    "first_stage_operation_completed",
                    "first_token_produced",
                ):
                    if event_name not in observed_names:
                        handle.lifecycle_recorder.emit(
                            event_name,
                            monotonic_ns=not_applicable_ns,
                            duration_ns=0 if event_name.endswith("completed") else None,
                            details={
                                "observed": False,
                                "not_applicable": True,
                                "reason": "load-only phase intentionally runs no generation",
                            },
                        )
        for handle in handles:
            if handle.process.poll() is None:
                with suppress(OSError):
                    handle.shutdown_path.write_text("shutdown\n", encoding="utf-8")
        for handle in handles:
            if handle.process.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(
                        handle.process.wait,
                        config.resource_limits.max_worker_start_seconds,
                    )
        with suppress(Exception):
            await server.stop()
        lag_monitor_stop.set()
        with suppress(asyncio.CancelledError):
            await lag_monitor_task
        for handle in handles:
            if handle.process.poll() is None:
                stop_process(handle.process, terminate_timeout_s=15)
            handle.stdout_handle.close()
            handle.stderr_handle.close()
        sampler.stop()

    lifecycle_events = read_lifecycle_events(lifecycle_paths)
    worker_lifecycle_rows: list[dict[str, Any]] = []
    for handle in handles:
        rows = [row for row in lifecycle_events if row.get("worker_id") == handle.worker_id]
        runtime_process_id = next(
            (
                int(item["process_id"])
                for item in rows
                if item.get("event_name") == "python_entry_started"
            ),
            handle.runtime_process_id or handle.process.pid,
        )
        row: dict[str, Any] = {
            "worker_count": worker_count,
            "phase": phase,
            "repeat": repeat,
            "worker_id": handle.worker_id,
            "stage_id": handle.stage_id,
            "process_id": runtime_process_id,
            "launcher_process_id": handle.process.pid,
            "process_exit_code": handle.process.poll(),
        }
        for metric_name, start, end in (
            ("process_startup_seconds", "process_spawn_started", "python_entry_started"),
            ("python_import_seconds", "python_imports_started", "python_imports_completed"),
            (
                "cuda_initialisation_seconds",
                "cuda_initialisation_started",
                "cuda_initialisation_completed",
            ),
            (
                "shard_verification_seconds",
                "shard_verification_started",
                "shard_verification_completed",
            ),
            ("shard_read_seconds", "shard_read_started", "shard_read_completed"),
            ("weight_load_seconds", "weight_load_started", "weight_load_completed"),
            (
                "host_to_device_transfer_seconds",
                "host_to_device_transfer_started",
                "host_to_device_transfer_completed",
            ),
            (
                "stage_module_construction_seconds",
                "stage_module_construction_started",
                "stage_module_construction_completed",
            ),
            ("local_warmup_seconds", "local_warmup_started", "local_warmup_completed"),
            ("worker_ready_seconds", "process_spawn_started", "worker_routable"),
            (
                "cached_time_to_contribution_seconds",
                "process_spawn_started",
                "worker_routable",
            ),
            (
                "cold_time_to_contribution_seconds",
                "assignment_created",
                "worker_routable",
            ),
        ):
            row[metric_name] = lifecycle_duration_seconds(
                rows,
                start_event=start,
                end_event=end,
            )
        worker_lifecycle_rows.append(row)
    if rejoin_metrics is not None:
        replacement_lifecycle = next(
            (
                row
                for row in worker_lifecycle_rows
                if row.get("worker_id") == rejoin_metrics.get("new_worker_id")
            ),
            None,
        )
        if replacement_lifecycle is not None:
            rejoin_metrics.update(
                {
                    "replacement_process_startup_seconds": replacement_lifecycle.get(
                        "process_startup_seconds"
                    ),
                    "replacement_shard_read_seconds": replacement_lifecycle.get(
                        "shard_read_seconds"
                    ),
                    "replacement_weight_load_seconds": replacement_lifecycle.get(
                        "weight_load_seconds"
                    ),
                    "replacement_host_to_device_transfer_seconds": (
                        replacement_lifecycle.get("host_to_device_transfer_seconds")
                    ),
                    "replacement_local_warmup_seconds": replacement_lifecycle.get(
                        "local_warmup_seconds"
                    ),
                }
            )

    loaded_stages = [
        int(stage_id) for health in health_rows for stage_id in health.get("loaded_stages", [])
    ]
    one_stage_per_worker = bool(health_rows) and all(
        len(health.get("loaded_stages", [])) == 1 for health in health_rows
    )
    complete_coverage = sorted(loaded_stages) == list(range(worker_count))
    cache_count = 0
    coordinator_weight_bytes = 0
    proof_valid = bool(health_rows) and all(row.get("proof_verified") for row in health_rows)
    cache_validation: list[dict[str, Any]] = []
    cache_valid = True
    for health in health_rows:
        shards = health.get("proof", {}).get("shards", {})
        for stage_payload in shards.get("stages", {}).values():
            module_state = stage_payload.get("module_state", {})
            stage_id = int(module_state.get("stage_id", -1))
            stage = manifest.stages[stage_id] if 0 <= stage_id < len(manifest.stages) else None
            cache_count += int(module_state.get("cache_count", 0))
            history = module_state.get("cache_history", [])
            history_rows = history if isinstance(history, list) else []
            expected_layers = (
                list(range(stage.layer_start, stage.layer_end)) if stage is not None else []
            )
            history_valid = bool(history_rows) if request_results else True
            peak_cache_bytes = 0
            for cache in history_rows:
                peak_cache_bytes = max(peak_cache_bytes, int(cache.get("cache_bytes", 0)))
                layer_rows = cache.get("layers", [])
                actual_layers = [
                    int(layer.get("global_layer_index", -1))
                    for layer in layer_rows
                    if isinstance(layer, dict)
                ]
                shapes_valid = all(
                    isinstance(layer.get("key_shape"), list)
                    and isinstance(layer.get("value_shape"), list)
                    and len(layer["key_shape"]) == 4
                    and len(layer["value_shape"]) == 4
                    and layer.get("dtype") == "bfloat16"
                    for layer in layer_rows
                    if isinstance(layer, dict)
                )
                history_valid = history_valid and (
                    stage is not None
                    and int(cache.get("owned_layer_count", -1)) == len(expected_layers)
                    and int(cache.get("initialised_layer_count", -1)) == len(expected_layers)
                    and actual_layers == expected_layers
                    and shapes_valid
                    and int(cache.get("sequence_length", 0)) > 0
                    and cache.get("stale_after_operation") is False
                )
            finite_valid = (
                int(module_state.get("finite_output_checks", 0)) > 0
                and module_state.get("all_checked_outputs_finite") is True
                if request_results
                else True
            )
            stage_cache_valid = (
                history_valid and finite_valid and int(module_state.get("cache_count", -1)) == 0
            )
            cache_valid = cache_valid and stage_cache_valid
            cache_validation.append(
                {
                    "worker_id": health.get("worker_id"),
                    "stage_id": stage_id,
                    "history_record_count": len(history_rows),
                    "peak_kv_cache_bytes": peak_cache_bytes,
                    "cache_history_valid": history_valid,
                    "finite_output_checks": module_state.get("finite_output_checks"),
                    "all_checked_outputs_finite": finite_valid,
                    "stale_cache_count": module_state.get("cache_count"),
                    "valid": stage_cache_valid,
                }
            )
    transport: dict[str, Any] = {
        **core.runtime_transport_metrics,
        "data_plane": "direct",
        "coordinator_activation_bytes": int(
            core.runtime_transport_metrics.get("coordinator_activation_bytes", 0)
        ),
        "worker_to_worker_activation_bytes": int(
            core.runtime_transport_metrics.get("worker_to_worker_activation_bytes", 0)
        ),
        "coordinator_model_weight_bytes": coordinator_weight_bytes,
        "stage_local_kv_cache_valid": cache_valid,
        "cache_validation": cache_validation,
        "event_loop_lag_median_ms": (
            statistics.median(event_loop_lag_ms) if event_loop_lag_ms else None
        ),
        "event_loop_lag_p95_ms": (
            float(np.percentile(np.asarray(event_loop_lag_ms, dtype=np.float64), 95))
            if event_loop_lag_ms
            else None
        ),
        "event_loop_lag_maximum_ms": max(event_loop_lag_ms, default=None),
    }
    peer_snapshots = [health.get("proof", {}).get("peer_connections", {}) for health in health_rows]
    transport.update(
        {
            "peer_streams_created": sum(
                int(item.get("streams_created", 0)) for item in peer_snapshots
            ),
            "peer_channels_created": sum(
                int(item.get("channels_created", 0)) for item in peer_snapshots
            ),
            "peer_active_pairs": sum(
                int(item.get("active_peer_pairs", 0)) for item in peer_snapshots
            ),
        }
    )
    clean_shutdown = all(
        handle.process.poll() == 0
        for handle in handles
        if not (
            rejoin_metrics is not None
            and handle.stage_id == rejoin_stage_id
            and not handle.worker_id.endswith("-replacement")
        )
    )
    cleanup = {
        "worker_exit_codes": {handle.worker_id: handle.process.poll() for handle in handles},
        "all_workers_stopped": all(handle.process.poll() is not None for handle in handles),
        "clean_shutdown": clean_shutdown,
        "stale_worker_processes": [
            handle.process.pid for handle in handles if handle.process.poll() is None
        ],
        "session_root": str(session_root),
        "worker_stdout_paths": [str(handle.stdout_path) for handle in handles],
        "worker_stderr_paths": [str(handle.stderr_path) for handle in handles],
        "expected_terminated_worker_ids": (
            [str(rejoin_metrics["old_worker_id"])] if rejoin_metrics is not None else []
        ),
        "unexpected_worker_crashes": [
            {
                "worker_id": handle.worker_id,
                "exit_code": handle.process.poll(),
            }
            for handle in handles
            if handle.process.poll() not in {0, None}
            and not (
                rejoin_metrics is not None
                and handle.worker_id == rejoin_metrics.get("old_worker_id")
            )
        ],
    }
    if rejoin_metrics is not None:
        rejoin_metrics["clean_shutdown"] = clean_shutdown
        rejoin_metrics["unexpected_worker_crashes"] = cleanup["unexpected_worker_crashes"]
        rejoin_metrics["status"] = (
            "PASS"
            if rejoin_metrics.get("new_pid")
            and rejoin_metrics.get("worker_process_genuinely_terminated")
            and rejoin_metrics.get("route_generation_incremented")
            and rejoin_metrics.get("cache_replay_occurred")
            and rejoin_metrics.get("exact_token_identity")
            and clean_shutdown
            and not cleanup["unexpected_worker_crashes"]
            else "FAIL"
        )
    generated = bool(request_results)
    request_pass = all(bool(item.get("passed")) for item in request_results)
    direct_ok = transport["coordinator_activation_bytes"] == 0 and (
        not generated or worker_count == 1 or transport["worker_to_worker_activation_bytes"] > 0
    )
    passed: bool = bool(
        failure_type is None
        and one_stage_per_worker
        and complete_coverage
        and proof_valid
        and cache_valid
        and cache_count == 0
        and direct_ok
        and request_pass
        and cleanup["all_workers_stopped"]
        and clean_shutdown
    )
    return FanoutSessionResult(
        worker_count=worker_count,
        phase=phase,
        repeat=repeat,
        passed=passed,
        runnable_generation=generated and request_pass and direct_ok and complete_coverage,
        failure_type=failure_type,
        failure_message=failure_message,
        pipeline_ready_seconds=pipeline_ready,
        request_results=request_results,
        worker_lifecycle_rows=worker_lifecycle_rows,
        resource_rows=sampler.resource_rows,
        worker_memory_rows=sampler.worker_rows,
        gpu_process_memory_rows=sampler.gpu_process_rows,
        health_rows=health_rows,
        lifecycle_events=lifecycle_events,
        transport_metrics=transport,
        cleanup=cleanup,
        rejoin=rejoin_metrics,
    )
