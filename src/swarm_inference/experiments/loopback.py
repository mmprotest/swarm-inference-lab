"""Single-host loopback experiment with independently killable worker processes."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from swarm_inference.config.models import (
    Backend,
    ExecutionMode,
    ExperimentConfig,
    ModelManifest,
)
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.experiments.charts import generate_charts
from swarm_inference.experiments.metrics import (
    evaluate_experiment,
    project_acceptance_status,
)
from swarm_inference.experiments.reporting import render_html_report
from swarm_inference.experiments.runner import (
    ExperimentRun,
    _write_artifact_manifest,
    collect_environment,
)
from swarm_inference.host import qualifies_as_remote_physical_worker, stop_process
from swarm_inference.protocol.messages import SubmitRequest, SubmitResponse


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _submission(
    config: ExperimentConfig,
    *,
    request_id: str,
    seed: int,
    model_id: str,
    model_revision: str,
    prompt: str | None = None,
) -> SubmitRequest:
    return SubmitRequest(
        request_id=request_id,
        prompt=prompt,
        prompt_token_ids=(
            []
            if prompt is not None
            else [101 + (index % 997) for index in range(config.workload.prompt_tokens)]
        ),
        max_new_tokens=config.workload.output_tokens,
        random_seed=seed,
        workload_class=config.workload.workload_class.value,
        model_id=model_id,
        model_revision=model_revision,
    )


async def _wait_for_worker_count(
    core: CoordinatorCore,
    *,
    expected_worker_count: int,
    timeout_s: float,
    processes: list[subprocess.Popen[str]],
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        registered = len(core.registry.workers())
        if registered >= expected_worker_count:
            return
        failed = [process.pid for process in processes if process.poll() is not None]
        if failed:
            raise RuntimeError(f"local worker process(es) exited before registration: {failed}")
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"timed out after {timeout_s:.1f}s waiting for {expected_worker_count} workers; "
        f"registered={len(core.registry.workers())}"
    )


async def _run_sustained_requests(
    core: CoordinatorCore,
    *,
    config: ExperimentConfig,
    duration_s: float,
    prefix: str,
    model_id: str,
    model_revision: str,
    prompt: str | None,
) -> list[SubmitResponse]:
    if duration_s <= 0:
        raise ValueError("measured workload duration must be positive")
    deadline = time.perf_counter() + duration_s

    async def lane(lane_id: int) -> list[SubmitResponse]:
        responses: list[SubmitResponse] = []
        sequence = 0
        while time.perf_counter() < deadline or not responses:
            responses.append(
                await core.submit(
                    _submission(
                        config,
                        request_id=f"{prefix}-{lane_id:04d}-{sequence:08d}",
                        seed=config.seed + lane_id * 1_000_003 + sequence,
                        model_id=model_id,
                        model_revision=model_revision,
                        prompt=prompt,
                    )
                )
            )
            sequence += 1
        return responses

    lanes = await asyncio.gather(
        *(lane(index) for index in range(config.workload.concurrent_requests))
    )
    return [response for lane_responses in lanes for response in lane_responses]


async def _run_sustained_with_progress(
    core: CoordinatorCore,
    *,
    config: ExperimentConfig,
    duration_s: float,
    prefix: str,
    model_id: str,
    model_revision: str,
    prompt: str | None,
    label: str,
    progress_callback: Callable[[str], None] | None,
) -> list[SubmitResponse]:
    workload = asyncio.create_task(
        _run_sustained_requests(
            core,
            config=config,
            duration_s=duration_s,
            prefix=prefix,
            model_id=model_id,
            model_revision=model_revision,
            prompt=prompt,
        )
    )
    started = time.perf_counter()
    last_reported = -1
    while not workload.done():
        elapsed = min(duration_s, time.perf_counter() - started)
        whole_seconds = int(elapsed)
        if progress_callback is not None and whole_seconds != last_reported and whole_seconds > 0:
            progress_callback(f"{label}: {whole_seconds}/{int(duration_s)} seconds")
            last_reported = whole_seconds
        try:
            await asyncio.wait_for(asyncio.shield(workload), timeout=0.25)
        except TimeoutError:
            continue
    if progress_callback is not None and last_reported < int(duration_s):
        progress_callback(f"{label}: {int(duration_s)}/{int(duration_s)} seconds")
    return await workload


async def _collect_worker_proofs(core: CoordinatorCore) -> dict[str, dict[str, Any]]:
    replicas = core.registry.replicas()
    endpoint_by_worker = {
        replica.worker_id: replica.endpoint for replica in replicas if replica.endpoint is not None
    }

    async def one(worker_id: str, endpoint: str) -> tuple[str, dict[str, Any]]:
        health = await core.transport.health(endpoint)
        return worker_id, health.proof

    results = await asyncio.gather(
        *(one(worker_id, endpoint) for worker_id, endpoint in endpoint_by_worker.items()),
        return_exceptions=True,
    )
    proofs: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, BaseException):
            continue
        worker_id, proof = result
        proofs[worker_id] = proof
    return proofs


async def _profile_processes(
    *,
    worker_processes: list[subprocess.Popen[str]],
    interval_s: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    import psutil

    coordinator = psutil.Process()
    workers = [
        psutil.Process(process.pid) for process in worker_processes if process.poll() is None
    ]
    coordinator.cpu_percent(None)
    for worker in workers:
        worker.cpu_percent(None)
    expected = time.perf_counter() + interval_s
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            break
        except TimeoutError:
            pass
        now = time.perf_counter()
        worker_samples: dict[str, dict[str, float | int]] = {}
        for worker in workers:
            try:
                memory = worker.memory_info()
                worker_samples[str(worker.pid)] = {
                    "cpu_percent": worker.cpu_percent(None),
                    "rss_bytes": memory.rss,
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        coordinator_memory = coordinator.memory_info()
        samples.append(
            {
                "monotonic_s": now,
                "event_loop_lag_ms": max(0.0, (now - expected) * 1000),
                "coordinator_cpu_percent": coordinator.cpu_percent(None),
                "coordinator_rss_bytes": coordinator_memory.rss,
                "workers": worker_samples,
            }
        )
        expected = now + interval_s


async def run_loopback_experiment(
    config: ExperimentConfig,
    *,
    worker_count: int = 4,
    sustained: bool = False,
    duration_s: float | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ExperimentRun:
    """Launch local native workers for a burst or sustained loopback measurement."""

    return await _run_runtime_experiment(
        config,
        expected_worker_count=worker_count,
        local_worker_count=worker_count,
        listen_endpoint="127.0.0.1:0",
        startup_timeout_s=60.0,
        sustained=sustained,
        duration_s=duration_s,
        progress_callback=progress_callback,
    )


async def run_physical_experiment(
    config: ExperimentConfig,
    *,
    expected_worker_count: int,
    listen_endpoint: str = "0.0.0.0:50051",
    startup_timeout_s: float = 300.0,
    duration_s: float | None = None,
    ready_callback: Callable[[str, Path], None] | None = None,
    model_manifest: ModelManifest | None = None,
    architecture_config: dict[str, Any] | None = None,
    runtime_dtype: str | None = None,
    tokenizer: Any | None = None,
    prompt: str = "Explain why distributed inference is difficult.",
) -> ExperimentRun:
    """Wait for remote workers and record a real physical-network experiment.

    The synthetic backend measures the physical control/activation transport and
    deterministic stage execution. It is never described as real-model kernel
    performance.
    """

    return await _run_runtime_experiment(
        config,
        expected_worker_count=expected_worker_count,
        local_worker_count=0,
        listen_endpoint=listen_endpoint,
        startup_timeout_s=startup_timeout_s,
        sustained=True,
        duration_s=duration_s,
        ready_callback=ready_callback,
        model_manifest=model_manifest,
        architecture_config=architecture_config,
        runtime_dtype=runtime_dtype,
        tokenizer=tokenizer,
        prompt=prompt if model_manifest is not None else None,
    )


async def _run_runtime_experiment(
    config: ExperimentConfig,
    *,
    expected_worker_count: int,
    local_worker_count: int,
    listen_endpoint: str,
    startup_timeout_s: float,
    sustained: bool,
    duration_s: float | None = None,
    ready_callback: Callable[[str, Path], None] | None = None,
    model_manifest: ModelManifest | None = None,
    architecture_config: dict[str, Any] | None = None,
    runtime_dtype: str | None = None,
    tokenizer: Any | None = None,
    prompt: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ExperimentRun:
    if local_worker_count:
        if config.execution_mode != ExecutionMode.SINGLE_HOST_LOOPBACK:
            raise ValueError("local worker launch requires execution_mode=single-host-loopback")
    elif config.execution_mode not in {ExecutionMode.PHYSICAL_LAN, ExecutionMode.PHYSICAL_WAN}:
        raise ValueError("remote worker run requires execution_mode=physical-lan or physical-wan")
    worker_count = expected_worker_count
    stage_count = (
        len(model_manifest.stages) if model_manifest is not None else config.model.stage_count
    )
    if worker_count < stage_count:
        raise ValueError("worker_count cannot be smaller than the runtime model stage count")
    start = datetime.now(UTC)
    run_id = f"{start.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = Path(config.output_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "charts").mkdir()
    process_dir = run_dir / "processes"
    process_dir.mkdir()

    coordinator_affinity_before: list[int] = []
    coordinator_cpu: int | None = None
    worker_cpu_ids: list[int | None] = [None] * local_worker_count
    affinity_status = "disabled"
    if local_worker_count and config.synthetic_compute.cpu_affinity:
        try:
            import psutil

            current_process = psutil.Process()
            coordinator_affinity_before = list(current_process.cpu_affinity())
            if len(coordinator_affinity_before) >= local_worker_count + 1:
                coordinator_cpu = coordinator_affinity_before[0]
                worker_cpu_ids = [
                    int(value) for value in coordinator_affinity_before[1 : local_worker_count + 1]
                ]
                current_process.cpu_affinity([coordinator_cpu])
                affinity_status = "dedicated"
            elif len(coordinator_affinity_before) >= local_worker_count:
                worker_cpu_ids = [
                    int(value) for value in coordinator_affinity_before[:local_worker_count]
                ]
                affinity_status = "workers-dedicated-coordinator-shared"
            else:
                affinity_status = "insufficient-logical-cpus"
        except (AttributeError, OSError, ValueError, psutil.Error):
            affinity_status = "unsupported"

    core = CoordinatorCore(
        config=config,
        model_manifest=model_manifest,
        architecture_config=architecture_config,
        runtime_dtype=runtime_dtype,
        tokenizer=tokenizer,
    )
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start(listen_endpoint)
    listen_host = listen_endpoint.rsplit(":", 1)[0].strip("[]")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    if ready_callback is not None:
        display_host = listen_host if listen_host not in {"0.0.0.0", "::"} else "*"
        ready_callback(f"{display_host}:{coordinator_port}", run_dir)
    stages = core.stages
    largest_stage = max(stage.required_memory_bytes for stage in stages)
    full_model = (
        model_manifest.total_weight_bytes
        if model_manifest is not None
        else config.model.layer_count * config.model.bytes_per_layer
    )
    configured_memory_limit = config.worker.logical_memory_limit_bytes
    memory_limit = (
        configured_memory_limit
        if configured_memory_limit is not None
        else min(
            full_model - 1,
            largest_stage
            + max(
                64 * 1024 * 1024,
                config.model.cache_bytes_per_token_per_layer * 4096,
            ),
        )
    )
    if configured_memory_limit is not None and memory_limit <= largest_stage:
        raise ValueError(
            "worker logical memory limit must be larger than one stage: "
            f"limit={memory_limit}, largest_stage={largest_stage}"
        )
    if configured_memory_limit is None and memory_limit <= largest_stage:
        memory_limit = largest_stage
    if configured_memory_limit is not None and stage_count > 1 and memory_limit >= full_model:
        raise ValueError(
            "worker logical memory limit must remain below the full logical model: "
            f"limit={memory_limit}, full_model={full_model}"
        )

    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    worker_proofs_before: dict[str, dict[str, Any]] = {}
    worker_proofs_after: dict[str, dict[str, Any]] = {}
    profile_samples: list[dict[str, Any]] = []
    profile_stop = asyncio.Event()
    profile_task: asyncio.Task[None] | None = None
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    for thread_variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[thread_variable] = "1"
    try:
        for index in range(local_worker_count):
            port = _free_port()
            endpoint = f"127.0.0.1:{port}"
            worker_id = f"loopback-worker-{index:03d}"
            log_path = process_dir / f"{worker_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            log_handles.append(log_handle)
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
                Backend.SYNTHETIC.value,
                "--memory-limit-bytes",
                str(memory_limit),
                "--identity",
                str(process_dir / f"{worker_id}.pem"),
                "--worker-id",
                worker_id,
                "--queue-capacity",
                str(config.queue.capacity),
                "--max-microbatch-size",
                str(config.queue.max_microbatch_size),
                "--max-microbatch-wait-ms",
                str(config.queue.max_microbatch_wait_ms),
                "--outbound-queue-capacity",
                str(config.worker.outbound_queue_capacity),
                "--inbound-queue-capacity",
                str(config.worker.inbound_queue_capacity),
                "--max-inflight-operations",
                str(config.worker.max_inflight_operations),
                "--reconnect-attempts",
                str(config.transport.reconnect_attempts),
                "--reconnect-initial-backoff-ms",
                str(config.transport.reconnect_initial_backoff_ms),
                "--reconnect-max-backoff-ms",
                str(config.transport.reconnect_max_backoff_ms),
            ]
            if worker_cpu_ids[index] is not None:
                command.extend(["--cpu-affinity", str(worker_cpu_ids[index])])
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=Path.cwd(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        await _wait_for_worker_count(
            core,
            expected_worker_count=expected_worker_count,
            timeout_s=startup_timeout_s,
            processes=processes,
        )
        registered_workers = core.registry.workers()
        if not local_worker_count:
            remote_workers = [
                worker
                for worker in registered_workers
                if qualifies_as_remote_physical_worker(
                    worker_hostname=worker.hostname,
                    endpoint=worker.endpoint,
                )
            ]
            if not remote_workers:
                raise ValueError(
                    "physical execution requires at least one worker with a different "
                    "hostname and a non-local advertised IP; use single-host-loopback "
                    "for one-host tests"
                )
            if model_manifest is None and any(
                worker.backend != Backend.SYNTHETIC for worker in registered_workers
            ):
                raise ValueError(
                    "synthetic physical experiments require --backend synthetic so "
                    "NumPy stage work is not misreported as CPU/GPU model execution"
                )
            if model_manifest is not None and any(
                worker.backend == Backend.SYNTHETIC for worker in registered_workers
            ):
                raise ValueError("real-model physical experiments require torch workers")
        minimum_replicas = 2 if local_worker_count and worker_count >= stage_count * 2 else 1
        await core.wait_for_coverage(
            minimum_replicas=minimum_replicas,
            timeout_s=startup_timeout_s,
        )
        worker_proofs_before = await _collect_worker_proofs(core)
        if sustained:
            if config.warmup_s > 0:
                core.events.append(
                    {
                        "event_type": "warmup_started",
                        "duration_s": config.warmup_s,
                    }
                )
                await _run_sustained_with_progress(
                    core,
                    config=config,
                    duration_s=config.warmup_s,
                    prefix="warmup",
                    model_id=core.runtime_model_id,
                    model_revision=core.runtime_model_revision,
                    prompt=prompt,
                    label="Warm-up",
                    progress_callback=progress_callback,
                )
                core.request_metrics.clear()
                core.events.append({"event_type": "warmup_completed"})
            run_started = time.perf_counter()
            if config.profiling.enabled and processes:
                profile_task = asyncio.create_task(
                    _profile_processes(
                        worker_processes=processes,
                        interval_s=config.profiling.sample_interval_ms / 1000,
                        stop_event=profile_stop,
                        samples=profile_samples,
                    )
                )
            responses = await _run_sustained_with_progress(
                core,
                config=config,
                duration_s=duration_s or config.steady_state_s,
                prefix=("physical" if not local_worker_count else "loopback"),
                model_id=core.runtime_model_id,
                model_revision=core.runtime_model_revision,
                prompt=prompt,
                label="Measurement",
                progress_callback=progress_callback,
            )
            elapsed = time.perf_counter() - run_started
            if profile_task is not None:
                profile_stop.set()
                await profile_task
                profile_task = None
        else:
            run_started = time.perf_counter()
            if config.profiling.enabled and processes:
                profile_task = asyncio.create_task(
                    _profile_processes(
                        worker_processes=processes,
                        interval_s=config.profiling.sample_interval_ms / 1000,
                        stop_event=profile_stop,
                        samples=profile_samples,
                    )
                )
            submissions = [
                _submission(
                    config,
                    request_id=f"loopback-request-{index:05d}",
                    seed=config.seed + index,
                    model_id=core.runtime_model_id,
                    model_revision=core.runtime_model_revision,
                )
                for index in range(config.workload.concurrent_requests)
            ]
            responses = await asyncio.gather(*(core.submit(item) for item in submissions))
            elapsed = time.perf_counter() - run_started
            if profile_task is not None:
                profile_stop.set()
                await profile_task
                profile_task = None
        worker_proofs_after = await _collect_worker_proofs(core)
    finally:
        if profile_task is not None:
            profile_stop.set()
            await profile_task
        await server.stop()
        for process in processes:
            stop_process(process)
        for handle in log_handles:
            handle.close()
        if coordinator_affinity_before:
            try:
                import psutil

                psutil.Process().cpu_affinity(coordinator_affinity_before)
            except (AttributeError, OSError, ValueError, psutil.Error):
                pass

    end = datetime.now(UTC)
    actual_worker_count = len(core.registry.workers())
    concurrency = config.workload.concurrent_requests
    accepted_request_count = len(responses)
    run_key = f"n{actual_worker_count}-c{concurrency}"
    is_physical = config.execution_mode in {
        ExecutionMode.PHYSICAL_LAN,
        ExecutionMode.PHYSICAL_WAN,
    }
    verified = [response for response in responses if response.verified]
    verified_tokens = sum(len(response.output_token_ids) for response in verified)
    request_rows: list[dict[str, Any]] = []
    metric_by_id = {row["request_id"]: row for row in core.request_metrics}
    for response in responses:
        metric = metric_by_id[response.request_id]
        token_count = len(response.output_token_ids)
        ttft = response.time_to_first_token_s
        decode_duration = (
            response.end_to_end_s - ttft if ttft is not None else response.end_to_end_s
        )
        request_rows.append(
            {
                "run_key": run_key,
                "request_id": response.request_id,
                "status": response.status,
                "verification_state": "verified" if response.verified else "rejected",
                "committed_output_tokens": token_count,
                "decode_tokens_s": (
                    max(token_count - 1, 0) / decode_duration if decode_duration > 0 else 0.0
                ),
                "time_to_first_token_s": ttft,
                "end_to_end_s": response.end_to_end_s,
                "queueing_s": metric["queue_s"],
                "network_s": metric["transport_s"],
                "stage_execution_s": metric["stage_execution_s"],
                "replay_s": metric["replay_s"],
                "replay_bytes": metric["replay_bytes"],
                "retry_count": metric["retry_count"],
                "route_changes": metric["route_changes"],
                "admission_time_ms": metric["admission_time_ms"],
                "route_reservation_time_ms": metric["route_reservation_time_ms"],
                "route_id": metric["route_id"],
                "route_generation": metric["route_generation"],
                "data_plane_mode": metric["data_plane_mode"],
                "detail": response.detail,
            }
        )
    replicas = core.registry.replicas()
    all_operations = [
        {
            **operation,
            "request_id": metric["request_id"],
            "route_id": metric.get("route_id"),
        }
        for metric in core.request_metrics
        for operation in metric["per_stage"]
    ]
    replica_keys = [f"{replica.stage_id}:{replica.worker_id}" for replica in replicas]
    operations_by_replica = {
        key: sum(
            1
            for operation in all_operations
            if key == f"{int(operation['stage_id'])}:{operation['worker_id']}"
        )
        for key in replica_keys
    }
    tokens_by_replica = dict(operations_by_replica)
    busy_time_by_replica = {
        key: sum(
            float(operation["execution_ms"]) / 1000
            for operation in all_operations
            if key == f"{int(operation['stage_id'])}:{operation['worker_id']}"
        )
        for key in replica_keys
    }
    bytes_by_replica = {
        key: sum(
            int(operation["activation_bytes_sent"])
            for operation in all_operations
            if key == f"{int(operation['stage_id'])}:{operation['worker_id']}"
        )
        for key in replica_keys
    }
    route_assignments_by_replica = {
        key: len(
            {
                str(operation.get("route_id"))
                for operation in all_operations
                if key == f"{int(operation['stage_id'])}:{operation['worker_id']}"
                and operation.get("route_id") is not None
            }
        )
        for key in replica_keys
    }
    stage_rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_records = [
            operation for operation in all_operations if operation["stage_id"] == stage.stage_id
        ]
        busy = sum(float(item["execution_ms"]) / 1000 for item in stage_records)
        stage_replicas = [replica for replica in replicas if replica.stage_id == stage.stage_id]
        replica_count = len(stage_replicas)
        stage_operation_count = len(stage_records)
        distribution = {
            replica.worker_id: operations_by_replica[f"{stage.stage_id}:{replica.worker_id}"]
            for replica in stage_replicas
        }
        meaningful = {
            worker_id: count
            for worker_id, count in distribution.items()
            if stage_operation_count > 0 and count / stage_operation_count >= 0.05
        }
        meaningful_counts = list(meaningful.values())
        imbalance = (
            max(meaningful_counts) / min(meaningful_counts)
            if meaningful_counts and min(meaningful_counts) > 0
            else 0.0
        )
        admission_share_ms = sum(
            float(metric.get("admission_time_ms", 0.0)) for metric in core.request_metrics
        ) / max(stage_operation_count, 1)
        effective_service_ms: dict[str, float] = {}
        measured_service_rates: dict[str, float] = {}
        independent_service_rates: dict[str, float] = {}
        for replica in stage_replicas:
            replica_records = [
                operation
                for operation in stage_records
                if operation["worker_id"] == replica.worker_id
            ]
            operation_count = len(replica_records)
            measured_ms = elapsed * 1000 / operation_count if operation_count else float("inf")
            effective_service_ms[replica.worker_id] = measured_ms
            measured_service_rates[replica.worker_id] = (
                1000 / measured_ms if measured_ms > 0 and math.isfinite(measured_ms) else 0.0
            )
            exclusive_samples = [
                float(operation.get("execution_ms", 0.0))
                + float(operation.get("serialisation_ms", 0.0))
                + float(operation.get("deserialisation_ms", 0.0))
                + float(operation.get("integrity_validation_ms", 0.0))
                + float(operation.get("cache_update_ms", 0.0))
                + admission_share_ms
                for operation in replica_records
            ]
            exclusive_ms = (
                sum(exclusive_samples) / len(exclusive_samples)
                if exclusive_samples
                else float("inf")
            )
            independent_service_rates[replica.worker_id] = (
                1000 / exclusive_ms if exclusive_ms > 0 and math.isfinite(exclusive_ms) else 0.0
            )
        raw_stage_capacity = sum(independent_service_rates.values())
        predicted_stage_capacity = sum(measured_service_rates.values())
        measured_efficiency_factor = (
            min(1.0, predicted_stage_capacity / raw_stage_capacity)
            if raw_stage_capacity > 0
            else 0.0
        )
        component_names = (
            "execution_ms",
            "queue_ms",
            "serialisation_ms",
            "deserialisation_ms",
            "integrity_validation_ms",
            "cache_update_ms",
            "stream_queue_ms",
            "transfer_ms",
        )
        critical_path_components_ms = {
            name: (
                sum(float(record.get(name, 0.0)) for record in stage_records)
                / max(stage_operation_count, 1)
            )
            for name in component_names
        }
        critical_path_components_ms["coordinator_admission_ms"] = admission_share_ms
        stage_rows.append(
            {
                "run_key": run_key,
                "stage_id": stage.stage_id,
                "replica_count": replica_count,
                "aggregate_service_rate": predicted_stage_capacity,
                "measured_service_rates": measured_service_rates,
                "per_replica_effective_service_ms": effective_service_ms,
                "raw_stage_capacity": raw_stage_capacity,
                "measured_efficiency_factor": measured_efficiency_factor,
                "predicted_capacity": predicted_stage_capacity,
                "critical_path_components_ms": critical_path_components_ms,
                "queue_depth": 0,
                "utilisation": min(1.0, busy / max(elapsed * replica_count, 1e-12)),
                "operations": stage_operation_count,
                "operations_by_replica": distribution,
                "tokens_by_replica": {
                    worker_id: tokens_by_replica[f"{stage.stage_id}:{worker_id}"]
                    for worker_id in distribution
                },
                "busy_time_by_replica": {
                    worker_id: busy_time_by_replica[f"{stage.stage_id}:{worker_id}"]
                    for worker_id in distribution
                },
                "bytes_by_replica": {
                    worker_id: bytes_by_replica[f"{stage.stage_id}:{worker_id}"]
                    for worker_id in distribution
                },
                "meaningfully_used_replicas": len(meaningful),
                "meaningful_replica_fraction": (
                    len(meaningful) / replica_count if replica_count else 0.0
                ),
                "replica_imbalance_ratio": imbalance,
                "failure_count": sum(
                    1
                    for event in core.events
                    if event.get("event_type") == "stage_recovered"
                    and event.get("stage_id") == stage.stage_id
                ),
                "replay_overhead_s": sum(
                    float(metric["replay_s"]) for metric in core.request_metrics
                ),
                "route_distribution": distribution,
            }
        )
    reservation_snapshot = core.route_allocator.snapshot()
    worker_rows: list[dict[str, Any]] = []
    for worker in core.registry.workers():
        assigned_replica = next(
            (item for item in replicas if item.worker_id == worker.worker_id),
            None,
        )
        stage_id = assigned_replica.stage_id if assigned_replica is not None else None
        replica_key = f"{stage_id}:{worker.worker_id}" if stage_id is not None else None
        state = (
            reservation_snapshot["replicas"].get(replica_key, {}) if replica_key is not None else {}
        )
        proof = worker_proofs_after.get(worker.worker_id, {})
        peer = proof.get("peer_connections", {})
        operation_count = (
            operations_by_replica.get(replica_key, 0) if replica_key is not None else 0
        )
        busy_time = busy_time_by_replica.get(replica_key, 0.0) if replica_key is not None else 0.0
        worker_rows.append(
            {
                "run_key": run_key,
                "worker_id": worker.worker_id,
                "queue_depth": worker.current_queue_depth,
                "assigned_stage_id": stage_id,
                "memory_limit_bytes": worker.memory_limit_bytes,
                "profile_source": worker.profile_source,
                "cpu_affinity": worker.cpu_affinity,
                "single_thread_environment": worker.single_thread_environment,
                "operations": operation_count,
                "tokens_contributed": operation_count,
                "busy_time_s": busy_time,
                "reserved_work": route_assignments_by_replica.get(replica_key or "", 0)
                * config.workload.output_tokens,
                "in_flight_work": state.get("in_flight_stage_operations", 0),
                "bytes_sent": bytes_by_replica.get(replica_key or "", 0),
                "bytes_received": bytes_by_replica.get(replica_key or "", 0),
                "time_starved_s": max(0.0, elapsed - busy_time),
                "time_overloaded_s": sum(
                    float(operation.get("queue_ms", 0.0)) / 1000
                    for operation in all_operations
                    if operation["worker_id"] == worker.worker_id
                ),
                "peer_channels_created": int(peer.get("channels_created", 0)),
                "peer_streams_created": int(peer.get("streams_created", 0)),
                "peer_stream_reconnects": int(peer.get("stream_reconnects", 0)),
                "active_peer_pairs": int(peer.get("active_peer_pairs", 0)),
                "data_messages_sent": int(peer.get("messages_sent", 0)),
                "data_messages_received": int(peer.get("messages_received", 0)),
                "peer_payload_bytes": int(peer.get("payload_bytes", 0)),
                "peer_control_bytes": int(peer.get("control_bytes", 0)),
                "peer_queue_wait_ms": float(peer.get("queue_wait_ms", 0.0)),
                "peer_serialisation_ms": float(peer.get("serialisation_time_ms", 0.0)),
                "peer_deserialisation_ms": float(peer.get("deserialisation_time_ms", 0.0)),
                "peer_transfer_ms": float(peer.get("network_transfer_time_ms", 0.0)),
            }
        )
    if config.data_plane.value == "direct":
        network_rows = [
            {
                "run_key": run_key,
                "request_id": operation["request_id"],
                "source": operation.get("source_worker", operation["worker_id"]),
                "destination": operation.get("destination_worker", "coordinator"),
                "stage_id": operation["stage_id"],
                "token_position": operation["token_position"],
                "payload_bytes": operation["activation_bytes_sent"],
                "direction": (
                    "worker-to-worker"
                    if int(operation["stage_id"]) < len(stages) - 1
                    else "final-result"
                ),
                "measured_transport_elapsed_s": (float(operation.get("transfer_ms", 0.0)) / 1000),
                "transport_scope": "physical" if is_physical else "local",
                "emulated_delay_s": 0,
            }
            for operation in all_operations
        ]
    else:
        network_rows = [
            {
                "run_key": run_key,
                "request_id": operation["request_id"],
                "source": "coordinator",
                "destination": operation["worker_id"],
                "stage_id": operation["stage_id"],
                "token_position": operation["token_position"],
                "payload_bytes": operation["activation_bytes_sent"],
                "direction": "activation-input",
                "measured_transport_elapsed_s": operation["transport_elapsed_s"],
                "transport_scope": "physical" if is_physical else "local",
                "emulated_delay_s": 0,
            }
            for operation in all_operations
        ] + [
            {
                "run_key": run_key,
                "request_id": operation["request_id"],
                "source": operation["worker_id"],
                "destination": "coordinator",
                "stage_id": operation["stage_id"],
                "token_position": operation["token_position"],
                "payload_bytes": operation["activation_bytes_received"],
                "direction": "activation-result",
                "measured_transport_elapsed_s": operation["transport_elapsed_s"],
                "transport_scope": "physical" if is_physical else "local",
                "emulated_delay_s": 0,
            }
            for operation in all_operations
        ]
    predicted_throughput = min(
        (float(row["predicted_capacity"]) for row in stage_rows),
        default=0.0,
    )
    observed_throughput = verified_tokens / max(elapsed, 1e-12)
    prediction_error = abs(predicted_throughput - observed_throughput)
    prediction_error_fraction = (
        prediction_error / observed_throughput if observed_throughput > 0 else 1.0
    )
    peer_channels_created = sum(int(row["peer_channels_created"]) for row in worker_rows)
    peer_streams_created = sum(int(row["peer_streams_created"]) for row in worker_rows)
    peer_stream_reconnects = sum(int(row["peer_stream_reconnects"]) for row in worker_rows)
    active_peer_pairs = sum(int(row["active_peer_pairs"]) for row in worker_rows)
    peer_messages_sent = sum(int(row["data_messages_sent"]) for row in worker_rows)
    peer_messages_received = sum(int(row["data_messages_received"]) for row in worker_rows)
    assigned_replica_count = len(replicas)
    meaningfully_used_replica_count = sum(
        int(row["meaningfully_used_replicas"]) for row in stage_rows
    )
    meaningful_fraction = (
        meaningfully_used_replica_count / assigned_replica_count if assigned_replica_count else 0.0
    )
    replica_imbalance = max(
        (float(row["replica_imbalance_ratio"]) for row in stage_rows),
        default=0.0,
    )
    profile_components_ms = {
        "stage execution": sum(
            float(operation.get("execution_ms", 0.0)) for operation in all_operations
        ),
        "worker input queue": sum(
            float(operation.get("queue_ms", 0.0)) for operation in all_operations
        ),
        "serialisation": sum(
            float(operation.get("serialisation_ms", 0.0)) for operation in all_operations
        ),
        "deserialisation": sum(
            float(operation.get("deserialisation_ms", 0.0)) for operation in all_operations
        ),
        "stream queue": sum(
            float(operation.get("stream_queue_ms", 0.0)) for operation in all_operations
        ),
        "hop transfer": sum(
            float(operation.get("transfer_ms", 0.0)) for operation in all_operations
        ),
        "integrity validation": sum(
            float(operation.get("integrity_validation_ms", 0.0)) for operation in all_operations
        ),
        "cache update": sum(
            float(operation.get("cache_update_ms", 0.0)) for operation in all_operations
        ),
        "coordinator admission": sum(
            float(metric.get("admission_time_ms", 0.0)) for metric in core.request_metrics
        ),
    }
    top_five_wall_time = [
        {"source": name, "total_ms": total_ms}
        for name, total_ms in sorted(
            profile_components_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    ]
    summary_row: dict[str, Any] = {
        "execution_mode": config.execution_mode.value,
        "data_plane": config.data_plane.value,
        "backend": config.backend,
        "data_plane_mode": config.data_plane.value,
        "values": "measured",
        "model": core.runtime_model_id,
        "node_count": actual_worker_count,
        "concurrent_request_count": concurrency,
        "simulated_duration_s": elapsed,
        "measured_duration_s": elapsed,
        "aggregate_verified_output_tokens_s": observed_throughput,
        "verified_output_tokens": verified_tokens,
        "completed_verified_requests": len(verified),
        "accepted_requests": accepted_request_count,
        "completion_fraction": len(verified) / max(accepted_request_count, 1),
        "mean_request_tokens_s": sum(float(row["decode_tokens_s"]) for row in request_rows)
        / max(len(request_rows), 1),
        "mean_time_to_first_token_s": sum(
            float(row["time_to_first_token_s"] or 0) for row in request_rows
        )
        / max(len(request_rows), 1),
        "mean_end_to_end_s": sum(float(row["end_to_end_s"]) for row in request_rows)
        / max(len(request_rows), 1),
        "minimum_stage_utilisation": min(
            (float(row["utilisation"]) for row in stage_rows), default=0.0
        ),
        "mean_stage_utilisation": sum(float(row["utilisation"]) for row in stage_rows)
        / max(len(stage_rows), 1),
        "network_bytes": sum(int(row["payload_bytes"]) for row in network_rows),
        "network_lost_transmissions": 0,
        "capacity_imbalance": _capacity_imbalance(stage_rows),
        "failed_requests": accepted_request_count - len(verified),
        "recovered_route_changes": sum(int(row["route_changes"]) for row in request_rows),
        "replay_bytes": sum(int(row["replay_bytes"]) for row in request_rows),
        "replay_duration_s": sum(float(row["replay_s"]) for row in request_rows),
        "quarantined_workers": 0,
        "idle_workers": actual_worker_count - len(replicas),
        "coordinator_control_bytes": int(
            core.runtime_transport_metrics["coordinator_control_bytes"]
        ),
        "coordinator_activation_bytes": int(
            core.runtime_transport_metrics["coordinator_activation_bytes"]
        ),
        "worker_to_worker_activation_bytes": int(
            core.runtime_transport_metrics["worker_to_worker_activation_bytes"]
        ),
        "peer_channels_created": peer_channels_created,
        "peer_streams_created": peer_streams_created,
        "peer_stream_reconnects": peer_stream_reconnects,
        "active_peer_pairs": active_peer_pairs,
        "data_messages_sent": peer_messages_sent,
        "data_messages_received": peer_messages_received,
        "serialisation_time_ms": float(core.runtime_transport_metrics["serialisation_time_ms"])
        + sum(float(row["peer_serialisation_ms"]) for row in worker_rows),
        "deserialisation_time_ms": float(core.runtime_transport_metrics["deserialisation_time_ms"])
        + sum(float(row["peer_deserialisation_ms"]) for row in worker_rows),
        "stream_queue_time_ms": float(core.runtime_transport_metrics["stream_queue_time_ms"])
        + sum(float(row["peer_queue_wait_ms"]) for row in worker_rows),
        "hop_transfer_time_ms": float(core.runtime_transport_metrics["hop_transfer_time_ms"]),
        "stage_execution_time_ms": float(core.runtime_transport_metrics["stage_execution_time_ms"]),
        "admission_time_ms": float(core.runtime_transport_metrics["admission_time_ms"]),
        "route_reservation_time_ms": float(
            core.runtime_transport_metrics["route_reservation_time_ms"]
        ),
        "route_assignments_by_replica": route_assignments_by_replica,
        "operations_by_replica": operations_by_replica,
        "tokens_by_replica": tokens_by_replica,
        "busy_time_by_replica": busy_time_by_replica,
        "reservation_leaks": int(reservation_snapshot["reservation_leaks"]),
        "assigned_replicas": assigned_replica_count,
        "meaningfully_used_replicas": meaningfully_used_replica_count,
        "meaningful_replica_fraction": meaningful_fraction,
        "replica_imbalance_ratio": replica_imbalance,
        "predicted_throughput": predicted_throughput,
        "absolute_prediction_error": prediction_error,
        "prediction_error_fraction": prediction_error_fraction,
        "capacity_normalised_efficiency": (
            observed_throughput / predicted_throughput if predicted_throughput > 0 else 0.0
        ),
        "affinity_status": affinity_status,
        "coordinator_cpu_affinity": ([coordinator_cpu] if coordinator_cpu is not None else []),
        "worker_cpu_affinities": {
            worker.worker_id: worker.cpu_affinity for worker in core.registry.workers()
        },
        "measurement_categories": {
            "gpu_kernel_time": (
                "included in worker stage execution time; not separately isolated"
                if model_manifest is not None
                else "not applicable (synthetic NumPy backend)"
            ),
            "local_transport_time": (
                0 if is_physical else sum(float(row["network_s"]) for row in request_rows)
            ),
            "emulated_compute_delay": 0,
            "emulated_wan_delay": 0,
            "data_plane_mode": config.data_plane.value,
            "physical_network_delay": (
                sum(float(row["network_s"]) for row in request_rows) if is_physical else 0
            ),
        },
    }
    if config.profiling.enabled:
        coordinator_cpu_samples = [
            float(sample["coordinator_cpu_percent"]) for sample in profile_samples
        ]
        worker_cpu = [
            float(worker_sample["cpu_percent"])
            for sample in profile_samples
            for worker_sample in sample["workers"].values()
        ]
        profile_payload = {
            "sample_interval_ms": config.profiling.sample_interval_ms,
            "samples": profile_samples,
            "coordinator_cpu_percent": {
                "mean": (
                    sum(coordinator_cpu_samples) / len(coordinator_cpu_samples)
                    if coordinator_cpu_samples
                    else 0.0
                ),
                "max": max(coordinator_cpu_samples, default=0.0),
            },
            "worker_cpu_percent": {
                "mean": (sum(worker_cpu) / len(worker_cpu) if worker_cpu else 0.0),
                "max": max(worker_cpu, default=0.0),
            },
            "event_loop_lag_ms": {
                "mean": (
                    sum(float(sample["event_loop_lag_ms"]) for sample in profile_samples)
                    / len(profile_samples)
                    if profile_samples
                    else 0.0
                ),
                "max": max(
                    (float(sample["event_loop_lag_ms"]) for sample in profile_samples),
                    default=0.0,
                ),
            },
            "grpc_message_rate_s": (
                (peer_messages_sent + peer_messages_received) / max(elapsed, 1e-12)
            ),
            "peer_stream_utilisation_messages_per_stream": (
                peer_messages_sent / peer_streams_created if peer_streams_created else 0.0
            ),
            "wall_time_components_ms": profile_components_ms,
            "top_five_wall_time_sources": top_five_wall_time,
            "memory": {
                "coordinator_peak_rss_bytes": max(
                    (int(sample["coordinator_rss_bytes"]) for sample in profile_samples),
                    default=0,
                ),
                "worker_peak_rss_bytes": max(
                    (
                        int(worker_sample["rss_bytes"])
                        for sample in profile_samples
                        for worker_sample in sample["workers"].values()
                    ),
                    default=0,
                ),
            },
        }
        _json(run_dir / "profile.json", profile_payload)
        summary_row["profile_top_five_wall_time_sources"] = top_five_wall_time
    scaling_rows = [
        {
            "node_count": actual_worker_count,
            "concurrent_requests": concurrency,
            "throughput": summary_row["aggregate_verified_output_tokens_s"],
            "baseline_throughput": summary_row["aggregate_verified_output_tokens_s"],
            "throughput_gain": 1.0,
            "marginal_throughput": 0.0,
            "homogeneous_scaling_efficiency": 1.0,
            "predicted_ideal_throughput": predicted_throughput,
            "predicted_throughput": predicted_throughput,
            "absolute_prediction_error": prediction_error,
            "prediction_error_fraction": prediction_error_fraction,
            "capacity_normalised_efficiency": summary_row["capacity_normalised_efficiency"],
        }
    ]
    experiment_criteria = evaluate_experiment(
        config=config,
        summaries=[summary_row],
        scaling_rows=scaling_rows,
    )
    project_criteria = project_acceptance_status(
        config=config,
        experiment_criteria=experiment_criteria,
    )
    criteria = [item.to_dict() for item in experiment_criteria + project_criteria]
    summary = {
        "schema_version": "1",
        "run_id": run_id,
        "execution_mode": config.execution_mode.value,
        "data_plane": config.data_plane.value,
        "backend": config.backend,
        "values": (
            (
                "measured physical real-model execution and physical gRPC transport; "
                "no emulated delays"
                if model_manifest is not None
                else "measured physical synthetic execution and physical gRPC transport; "
                "no emulated delays"
            )
            if is_physical
            else "measured local synthetic execution and local gRPC transport; no emulated delays"
        ),
        "model_id": core.runtime_model_id,
        "model_revision": core.runtime_model_revision,
        "seed": config.seed,
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat(),
        "primary_metric_definition": (
            "committed output tokens from successfully completed and verified requests "
            "divided by measured steady-state experiment wall time"
        ),
        "primary_result": summary_row,
        "baseline_result": summary_row,
        "matrix_results": [summary_row],
        "transport_metrics": {
            **core.runtime_transport_metrics,
            "peer_channels_created": peer_channels_created,
            "peer_streams_created": peer_streams_created,
            "peer_stream_reconnects": peer_stream_reconnects,
            "active_peer_pairs": active_peer_pairs,
            "data_messages_sent": peer_messages_sent,
            "data_messages_received": peer_messages_received,
        },
        "affinity": {
            "status": affinity_status,
            "coordinator_cpu": coordinator_cpu,
            "workers": {
                worker.worker_id: worker.cpu_affinity for worker in core.registry.workers()
            },
        },
        "acceptance_criteria": criteria,
        "status": "PASS" if all(item["status"] == "PASS" for item in criteria) else "FAIL",
        "failed_acceptance_criteria": [
            item["name"] for item in criteria if item["status"] == "FAIL"
        ],
        "limitations": (
            (
                [
                    "One physical node-count point does not establish hardware scaling.",
                    "GPU kernel time is included in stage execution and is not separately "
                    "isolated by this runner.",
                    "Aggregate throughput is not single-request generation speed.",
                ]
                if model_manifest is not None
                else [
                    "Physical transport uses deterministic synthetic stage execution and "
                    "is not real-model kernel performance evidence.",
                    "One physical node-count point does not establish hardware scaling.",
                    "Aggregate throughput is not single-request generation speed.",
                ]
            )
            if is_physical
            else [
                "Single-host-loopback does not prove physical hardware scaling.",
                "Synthetic execution is not real-model performance evidence.",
                "Aggregate throughput is not single-request generation speed.",
            ]
        ),
    }

    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    _json(
        run_dir / "environment.json",
        collect_environment(
            config=config.model_copy(
                update={
                    "model_id": core.runtime_model_id,
                    "model_revision": core.runtime_model_revision,
                }
            ),
            start=start,
            end=end,
        ),
    )
    _json(
        run_dir / "model_manifest.json",
        (
            {
                **model_manifest.model_dump(mode="json"),
                "evidence_kind": "measured real-model execution",
            }
            if model_manifest is not None
            else {
                "model_id": core.runtime_model_id,
                "model_revision": core.runtime_model_revision,
                "architecture": "synthetic-dense",
                "total_weight_bytes": full_model,
                "stages": [stage.model_dump(mode="json") for stage in stages],
                "evidence_kind": "measured synthetic execution",
            }
        ),
    )
    _json(
        run_dir / "worker_manifest.json",
        {
            "workers": [worker.model_dump(mode="json") for worker in core.registry.workers()],
            "replicas": [replica.model_dump(mode="json") for replica in replicas],
            "data_plane_mode": config.data_plane.value,
            "affinity_status": affinity_status,
            "coordinator_cpu_affinity": ([coordinator_cpu] if coordinator_cpu is not None else []),
            "worker_proofs_before_measurement": worker_proofs_before,
            "worker_proofs_after_measurement": worker_proofs_after,
            "worker_memory_limits": {
                worker.worker_id: worker.memory_limit_bytes for worker in core.registry.workers()
            },
            "full_model_bytes": full_model,
        },
    )
    event_rows = [{"sequence": index, **event} for index, event in enumerate(core.events)]
    failure_rows = [
        row
        for row in event_rows
        if row.get("event_type") in {"request_failed", "stage_recovered", "assignment_failed"}
    ]
    _jsonl(run_dir / "events.jsonl", event_rows)
    _jsonl(run_dir / "requests.jsonl", request_rows)
    _jsonl(run_dir / "stage_metrics.jsonl", stage_rows)
    _jsonl(run_dir / "worker_metrics.jsonl", worker_rows)
    _jsonl(run_dir / "network_metrics.jsonl", network_rows)
    _csv(
        run_dir / "scaling.csv",
        scaling_rows,
        list(scaling_rows[0]),
    )
    _csv(
        run_dir / "latency.csv",
        request_rows,
        [
            "run_key",
            "request_id",
            "time_to_first_token_s",
            "decode_tokens_s",
            "end_to_end_s",
            "queueing_s",
            "network_s",
            "stage_execution_s",
            "replay_s",
        ],
    )
    _csv(
        run_dir / "failures.csv",
        failure_rows,
        ["sequence", "event_type", "request_id", "worker_id", "stage_id", "detail"],
    )
    _json(run_dir / "summary.json", summary)
    generate_charts(
        chart_dir=run_dir / "charts",
        scaling_rows=scaling_rows,
        requests=request_rows,
        stages=stage_rows,
        workers=worker_rows,
        network=network_rows,
        failures=failure_rows,
    )
    report = render_html_report(
        run_dir=run_dir,
        summary=summary,
        scaling_rows=scaling_rows,
        request_rows=request_rows,
    )
    _write_artifact_manifest(run_dir)
    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        report_path=report,
        passed=summary["status"] == "PASS",
        summary=summary,
    )


def _capacity_imbalance(stage_rows: list[dict[str, Any]]) -> float:
    capacities = [float(row["aggregate_service_rate"]) for row in stage_rows]
    if not capacities or max(capacities) <= 0:
        return 1.0
    return (max(capacities) - min(capacities)) / max(capacities)
