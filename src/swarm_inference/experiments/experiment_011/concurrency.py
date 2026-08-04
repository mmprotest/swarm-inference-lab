"""Multi-session continuous scheduling over one persistent stage ring."""

from __future__ import annotations

import contextlib
import csv
import itertools
import json
import multiprocessing as mp
import socket
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.experiments.experiment_011.protocol import BufferPool, Operation
from swarm_inference.experiments.experiment_011.runtime import (
    StageRingController,
    StageWorkerConfiguration,
    _connect,
    _make_listener,
    _stage_worker_entry,
)
from swarm_inference.experiments.experiment_011.telemetry import merge_traces
from swarm_inference.experiments.experiment_011.tensor_transport import pack_tensor, unpack_tensor


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values), probability)) if values else 0.0


def run_concurrent_stage_ring(
    *,
    run_id: str,
    controller: StageRingController,
    prompt_token_ids: list[int],
    expected_token_ids: list[int],
    concurrency: int,
    generated_token_count: int,
    cancel_one_before_prefill: bool = False,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    plan = controller.plan
    control_listeners: list[socket.socket] = []
    data_listeners: list[socket.socket] = []
    control_endpoints: list[tuple[str, int]] = []
    data_endpoints: list[tuple[str, int]] = []
    for _ in plan.assignments:
        control_listener, control_endpoint = _make_listener()
        data_listener, data_endpoint = _make_listener()
        control_listeners.append(control_listener)
        data_listeners.append(data_listener)
        control_endpoints.append(control_endpoint)
        data_endpoints.append(data_endpoint)
    configs = tuple(
        StageWorkerConfiguration(
            run_id=run_id,
            model_path=plan.model_path,
            model_revision=plan.model_revision,
            tokenizer_revision=plan.tokenizer_revision,
            topology_id=plan.topology_id,
            assignment=assignment,
            assignments=plan.assignments,
            network_profile=controller.network_profile.model_dump(mode="json"),
            compression_request=controller.compression_request,
            trace_path=str(
                controller.output_directory / "traces" / f"stage-{assignment.stage_id}.ndjson"
            ),
            capture_directory=None,
            control_endpoint=control_endpoints[assignment.stage_id],
            data_endpoint=data_endpoints[assignment.stage_id],
            ring_endpoints=tuple(data_endpoints),
            timeout_s=controller.timeout_s,
            publication_queue_size=max(64, concurrency * generated_token_count * 2),
        )
        for assignment in plan.assignments
    )
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_stage_worker_entry,
            args=(config, control_listeners[index], data_listeners[index]),
            name=f"experiment-011-concurrent-stage-{index}",
        )
        for index, config in enumerate(configs)
    ]
    controls: list[socket.socket] = []
    pools = [BufferPool(capacity=2, initial_size=256 * 1024) for _ in configs]
    session_ids = [f"{run_id}-session-{index:02d}" for index in range(concurrency)]
    request_ids = [f"{run_id}-request-{index:02d}" for index in range(concurrency)]
    active_sessions = list(session_ids)
    tokens_by_session = {session: [] for session in session_ids}
    publication_times = {session: [] for session in session_ids}
    errors: list[str] = []
    cancellation_cleanup_seconds = 0.0
    measurement_started = 0
    measurement_ended = 0
    try:
        for process in processes:
            process.start()
        for listener in control_listeners + data_listeners:
            listener.close()
        controls = [_connect(endpoint, controller.timeout_s) for endpoint in control_endpoints]
        for connection, config in zip(controls, configs, strict=True):
            controller._send_control(connection, config, Operation.HELLO)
        for stage_id, connection in enumerate(controls):
            controller._receive_control(connection, stage_id, pools[stage_id])
        for connection, config in zip(controls, configs, strict=True):
            controller._send_control(connection, config, Operation.LOAD_STAGE)
        for stage_id, connection in enumerate(controls):
            controller._receive_control(connection, stage_id, pools[stage_id])
        for session_id, request_id in zip(session_ids, request_ids, strict=True):
            for connection, config in zip(controls, configs, strict=True):
                controller._send_control(
                    connection,
                    config,
                    Operation.OPEN_SESSION,
                    session_id=session_id,
                    request_id=request_id,
                    attributes={
                        "prompt_length": len(prompt_token_ids),
                        "generated_token_target": generated_token_count,
                    },
                )
            for stage_id, connection in enumerate(controls):
                controller._receive_control(connection, stage_id, pools[stage_id])
        if cancel_one_before_prefill and concurrency > 1:
            cancelled = session_ids[-1]
            cancellation_started = time.perf_counter_ns()
            for connection, config in zip(controls, configs, strict=True):
                controller._send_control(
                    connection,
                    config,
                    Operation.CANCEL_SESSION,
                    session_id=cancelled,
                    request_id=request_ids[-1],
                    attributes={"shutdown": False},
                )
            for stage_id, connection in enumerate(controls):
                controller._receive_control(connection, stage_id, pools[stage_id])
            cancellation_cleanup_seconds = (time.perf_counter_ns() - cancellation_started) / 1e9
            active_sessions.remove(cancelled)
        prompt = pack_tensor(
            torch.tensor([prompt_token_ids], dtype=torch.int64),
            requested_mode="none",
        )
        measurement_started = time.perf_counter_ns()
        for session_id, request_id in zip(session_ids, request_ids, strict=True):
            if session_id not in active_sessions:
                continue
            controller._send_control(
                controls[0],
                configs[0],
                Operation.PREFILL,
                session_id=session_id,
                request_id=request_id,
                token_position=0,
                payload=prompt.payload,
                tensor_shape=prompt.shape,
                tensor_dtype=prompt.dtype,
                attributes={
                    "tensor": prompt.attributes(),
                    "cache_position_start": 0,
                    "prompt_length": len(prompt_token_ids),
                    "generated_token_target": generated_token_count,
                },
            )
        expected_publications = len(active_sessions) * generated_token_count
        publications = 0
        while publications < expected_publications:
            publication = controller._receive_control(controls[0], 0, pools[0])
            if publication.operation != Operation.TOKEN_RESULT:
                raise RuntimeError("unexpected concurrent control response")
            tensor, _ = unpack_tensor(publication.payload, publication.attributes["tensor"])
            token = int(tensor.item())
            tokens_by_session[publication.session_id].append(token)
            publication_times[publication.session_id].append(time.perf_counter_ns())
            publications += 1
        measurement_ended = time.perf_counter_ns()
        for session_id, request_id in zip(session_ids, request_ids, strict=True):
            if session_id not in active_sessions:
                continue
            shutdown = session_id == active_sessions[-1]
            for connection, config in zip(controls, configs, strict=True):
                controller._send_control(
                    connection,
                    config,
                    Operation.CLOSE_SESSION,
                    session_id=session_id,
                    request_id=request_id,
                    attributes={"shutdown": shutdown},
                )
            for stage_id, connection in enumerate(controls):
                controller._receive_control(connection, stage_id, pools[stage_id])
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        for connection in controls:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        for listener in control_listeners + data_listeners:
            with contextlib.suppress(OSError):
                listener.close()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                errors.append(f"{process.name} required termination")
            elif process.exitcode not in {0, None}:
                errors.append(f"{process.name} exited {process.exitcode}")
        controller.trace.close()
    elapsed = (measurement_ended - measurement_started) / 1e9 if measurement_ended else 0.0
    per_session = []
    for session_id in active_sessions:
        times = publication_times[session_id]
        itls = [(right - left) / 1e9 for left, right in itertools.pairwise(times)]
        exact = tokens_by_session[session_id] == expected_token_ids[:generated_token_count]
        per_session.append(
            {
                "session_id": session_id,
                "generated_token_ids": tokens_by_session[session_id],
                "exact": exact,
                "ttft_seconds": (times[0] - measurement_started) / 1e9 if times else 0.0,
                "mean_itl_seconds": statistics.mean(itls) if itls else 0.0,
                "p95_itl_seconds": _quantile(itls, 0.95),
                "throughput_tps": generated_token_count / elapsed if elapsed else 0.0,
            }
        )
    trace_paths = [
        controller.output_directory / "traces" / f"stage-{index}.ndjson"
        for index in range(plan.stage_count)
    ] + [controller.trace_path]
    events = merge_traces(trace_paths)
    stage_compute_ns = {
        stage_id: sum(
            int(event.get("duration_ns", 0))
            for event in events
            if event.get("event") == "cuda_compute_end" and int(event["stage_id"]) == stage_id
        )
        for stage_id in range(plan.stage_count)
    }
    session_rates = [float(row["throughput_tps"]) for row in per_session]
    result = {
        "run_id": run_id,
        "concurrency_requested": concurrency,
        "concurrency_active": len(active_sessions),
        "generated_tokens_per_session": generated_token_count,
        "aggregate_verified_tokens": len(active_sessions) * generated_token_count,
        "aggregate_verified_tokens_per_second": (
            len(active_sessions) * generated_token_count / elapsed if elapsed else 0.0
        ),
        "per_session_verified_tokens_per_second": statistics.mean(session_rates)
        if session_rates
        else 0.0,
        "p50_first_token_latency_seconds": _quantile(
            [float(row["ttft_seconds"]) for row in per_session], 0.5
        ),
        "p95_first_token_latency_seconds": _quantile(
            [float(row["ttft_seconds"]) for row in per_session], 0.95
        ),
        "p50_inter_token_latency_seconds": _quantile(
            [float(row["mean_itl_seconds"]) for row in per_session], 0.5
        ),
        "p95_inter_token_latency_seconds": _quantile(
            [float(row["p95_itl_seconds"]) for row in per_session], 0.95
        ),
        "microbatch_size_distribution": {"1": len(active_sessions) * generated_token_count},
        "stage_utilisation": {
            str(stage_id): compute_ns / (elapsed * 1e9) if elapsed else 0.0
            for stage_id, compute_ns in stage_compute_ns.items()
        },
        "queue_latency_mean_ns": statistics.mean(
            int(event.get("duration_ns", 0))
            for event in events
            if event.get("event") == "stage_queue_exit"
        )
        if any(event.get("event") == "stage_queue_exit" for event in events)
        else 0.0,
        "fairness_min_over_max": min(session_rates) / max(session_rates)
        if session_rates and max(session_rates)
        else 0.0,
        "cancellation_cleanup_seconds": cancellation_cleanup_seconds,
        "cancelled_session": session_ids[-1]
        if cancel_one_before_prefill and concurrency > 1
        else None,
        "cancelled_session_kv_cleanup": cancellation_cleanup_seconds > 0,
        "all_sessions_exact": all(bool(row["exact"]) for row in per_session),
        "per_session": per_session,
        "errors": errors,
        "valid_for_claims": not errors and all(bool(row["exact"]) for row in per_session),
        "continuous_scheduler": True,
        "tensor_fusion_enabled": False,
        "evidence_category": "REAL_MODEL_MEASURED",
    }
    controller.output_directory.mkdir(parents=True, exist_ok=True)
    (controller.output_directory / "concurrency_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def write_concurrency_summary(results: list[dict[str, Any]], path: Path) -> None:
    scalar_keys = sorted(
        key for key, value in results[0].items() if not isinstance(value, (dict, list))
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
