"""Single-host loopback experiment with independently killable worker processes."""

from __future__ import annotations

import asyncio
import csv
import json
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


async def run_loopback_experiment(
    config: ExperimentConfig,
    *,
    worker_count: int = 4,
) -> ExperimentRun:
    """Launch local native worker processes and label the result as loopback."""

    return await _run_runtime_experiment(
        config,
        expected_worker_count=worker_count,
        local_worker_count=worker_count,
        listen_endpoint="127.0.0.1:0",
        startup_timeout_s=60.0,
        sustained=False,
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
    memory_limit = min(
        full_model - 1,
        largest_stage + max(64 * 1024 * 1024, config.model.cache_bytes_per_token_per_layer * 4096),
    )
    if memory_limit <= largest_stage:
        memory_limit = largest_stage

    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
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
            ]
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
        if sustained:
            if config.warmup_s > 0:
                core.events.append(
                    {
                        "event_type": "warmup_started",
                        "duration_s": config.warmup_s,
                    }
                )
                await _run_sustained_requests(
                    core,
                    config=config,
                    duration_s=config.warmup_s,
                    prefix="warmup",
                    model_id=core.runtime_model_id,
                    model_revision=core.runtime_model_revision,
                    prompt=prompt,
                )
                core.request_metrics.clear()
                core.events.append({"event_type": "warmup_completed"})
            run_started = time.perf_counter()
            responses = await _run_sustained_requests(
                core,
                config=config,
                duration_s=duration_s or config.steady_state_s,
                prefix="physical",
                model_id=core.runtime_model_id,
                model_revision=core.runtime_model_revision,
                prompt=prompt,
            )
            elapsed = time.perf_counter() - run_started
        else:
            run_started = time.perf_counter()
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
    finally:
        await server.stop()
        for process in processes:
            stop_process(process)
        for handle in log_handles:
            handle.close()

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
                "detail": response.detail,
            }
        )
    replicas = core.registry.replicas()
    stage_rows: list[dict[str, Any]] = []
    for stage in stages:
        stage_records = [
            operation
            for metric in core.request_metrics
            for operation in metric["per_stage"]
            if operation["stage_id"] == stage.stage_id
        ]
        busy = sum(float(item["execution_ms"]) / 1000 for item in stage_records)
        replica_count = len([replica for replica in replicas if replica.stage_id == stage.stage_id])
        stage_rows.append(
            {
                "run_key": run_key,
                "stage_id": stage.stage_id,
                "replica_count": replica_count,
                "aggregate_service_rate": sum(
                    replica.measured_service_rate
                    for replica in replicas
                    if replica.stage_id == stage.stage_id
                ),
                "queue_depth": 0,
                "utilisation": min(1.0, busy / max(elapsed * replica_count, 1e-12)),
                "failure_count": sum(
                    1
                    for event in core.events
                    if event.get("event_type") == "stage_recovered"
                    and event.get("stage_id") == stage.stage_id
                ),
                "replay_overhead_s": sum(
                    float(metric["replay_s"]) for metric in core.request_metrics
                ),
                "route_distribution": {},
            }
        )
    worker_rows = [
        {
            "run_key": run_key,
            "worker_id": worker.worker_id,
            "queue_depth": worker.current_queue_depth,
            "assigned_stage_id": next(
                (replica.stage_id for replica in replicas if replica.worker_id == worker.worker_id),
                None,
            ),
            "memory_limit_bytes": worker.memory_limit_bytes,
            "profile_source": worker.profile_source,
            "bytes_sent": sum(
                int(operation["activation_bytes_received"])
                for metric in core.request_metrics
                for operation in metric["per_stage"]
                if operation["worker_id"] == worker.worker_id
            ),
            "bytes_received": sum(
                int(operation["activation_bytes_sent"])
                for metric in core.request_metrics
                for operation in metric["per_stage"]
                if operation["worker_id"] == worker.worker_id
            ),
        }
        for worker in core.registry.workers()
    ]
    network_rows = [
        {
            "run_key": run_key,
            "request_id": metric["request_id"],
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
        for metric in core.request_metrics
        for operation in metric["per_stage"]
    ] + [
        {
            "run_key": run_key,
            "request_id": metric["request_id"],
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
        for metric in core.request_metrics
        for operation in metric["per_stage"]
    ]
    summary_row = {
        "execution_mode": config.execution_mode.value,
        "values": "measured",
        "model": core.runtime_model_id,
        "node_count": actual_worker_count,
        "concurrent_request_count": concurrency,
        "simulated_duration_s": elapsed,
        "measured_duration_s": elapsed,
        "aggregate_verified_output_tokens_s": verified_tokens / max(elapsed, 1e-12),
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
            "physical_network_delay": (
                sum(float(row["network_s"]) for row in request_rows) if is_physical else 0
            ),
        },
    }
    scaling_rows = [
        {
            "node_count": actual_worker_count,
            "concurrent_requests": concurrency,
            "throughput": summary_row["aggregate_verified_output_tokens_s"],
            "baseline_throughput": summary_row["aggregate_verified_output_tokens_s"],
            "throughput_gain": 1.0,
            "marginal_throughput": 0.0,
            "homogeneous_scaling_efficiency": 1.0,
            "predicted_ideal_throughput": min(
                (float(row["aggregate_service_rate"]) for row in stage_rows),
                default=0.0,
            ),
            "capacity_normalised_efficiency": 0.0,
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
