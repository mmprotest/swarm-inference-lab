"""Measured persistent localhost worker protocol and coordinator scaling."""

from __future__ import annotations

import multiprocessing as mp
import statistics
import struct
import time
from typing import Any, Sequence

import numpy as np

from swarm_inference.experiments.experiment_019.physical import timing

_HEADER = struct.Struct("<QIIII")
_RESPONSE = struct.Struct("<QQQQQ")


def _worker(connection: Any) -> None:
    while True:
        receive_started = time.perf_counter_ns()
        frame = connection.recv_bytes()
        receive_finished = time.perf_counter_ns()
        if not frame:
            break
        dispatch_started = time.perf_counter_ns()
        sequence, rows, hidden, route_count, payload_bytes = _HEADER.unpack_from(frame)
        payload = memoryview(frame)[_HEADER.size :]
        if len(payload) != payload_bytes:
            raise RuntimeError("persistent worker received a malformed frame")
        checksum = int(sum(payload[: min(64, len(payload))])) & 0xFFFFFFFF
        dispatch_finished = time.perf_counter_ns()
        send_started = time.perf_counter_ns()
        connection.send_bytes(
            _RESPONSE.pack(
                sequence ^ checksum,
                receive_started,
                receive_finished,
                dispatch_started,
                dispatch_finished,
            )
        )
        _ = rows, hidden, route_count, send_started
    connection.close()


def benchmark_persistent_protocol(
    payload_sizes: Sequence[int],
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_worker, args=(child,), daemon=True)
    process.start()
    child.close()
    rows: list[dict[str, Any]] = []
    try:
        for payload_size in payload_sizes:
            serialization_values: list[float] = []
            framing_values: list[float] = []
            queue_send_values: list[float] = []
            receive_values: list[float] = []
            deserialization_values: list[float] = []
            dispatch_values: list[float] = []
            round_trip_values: list[float] = []
            for iteration in range(warmup + iterations):
                serialization_started = time.perf_counter_ns()
                payload = np.arange(payload_size, dtype=np.uint8).tobytes()
                serialization_ms = (time.perf_counter_ns() - serialization_started) / 1e6
                framing_started = time.perf_counter_ns()
                header = _HEADER.pack(iteration, 1, 7168, 16, len(payload))
                frame = header + payload
                framing_ms = (time.perf_counter_ns() - framing_started) / 1e6
                round_trip_started = time.perf_counter_ns()
                send_started = time.perf_counter_ns()
                parent.send_bytes(frame)
                send_ms = (time.perf_counter_ns() - send_started) / 1e6
                receive_started = time.perf_counter_ns()
                response = parent.recv_bytes()
                receive_ms = (time.perf_counter_ns() - receive_started) / 1e6
                round_trip_ms = (time.perf_counter_ns() - round_trip_started) / 1e6
                deserialization_started = time.perf_counter_ns()
                _sequence, _receive_start, _receive_finish, dispatch_start, dispatch_finish = (
                    _RESPONSE.unpack(response)
                )
                deserialization_ms = (
                    time.perf_counter_ns() - deserialization_started
                ) / 1e6
                if iteration >= warmup:
                    serialization_values.append(serialization_ms)
                    framing_values.append(framing_ms)
                    queue_send_values.append(send_ms)
                    receive_values.append(receive_ms)
                    deserialization_values.append(deserialization_ms)
                    dispatch_values.append((dispatch_finish - dispatch_start) / 1e6)
                    round_trip_values.append(round_trip_ms)
            rows.append(
                {
                    "payload_bytes": payload_size,
                    "iterations": iterations,
                    "serialization": timing(serialization_values),
                    "framing": timing(framing_values),
                    "queue_send": timing(queue_send_values),
                    "dispatch": timing(dispatch_values),
                    "receive": timing(receive_values),
                    "deserialization": timing(deserialization_values),
                    "round_trip": timing(round_trip_values),
                    "software_overhead_p50_ms": sum(
                        (
                            timing(serialization_values)["p50_ms"],
                            timing(framing_values)["p50_ms"],
                            timing(queue_send_values)["p50_ms"],
                            timing(dispatch_values)["p50_ms"],
                            timing(receive_values)["p50_ms"],
                            timing(deserialization_values)["p50_ms"],
                        )
                    ),
                }
            )
    finally:
        try:
            parent.send_bytes(b"")
        except (BrokenPipeError, EOFError, OSError):
            pass
        parent.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    return {
        "schema_version": "experiment-019-persistent-protocol-v1",
        "status": "PASS" if process.exitcode == 0 else "FAIL",
        "process_model": "one persistent coordinator process and one persistent worker process",
        "rows": rows,
    }


def benchmark_control_plane(
    worker_counts: Sequence[int] = (100, 250, 500, 1000, 2000),
    *,
    repetitions: int = 20,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for worker_count in worker_counts:
        values: list[float] = []
        descriptor_count = math_ceil_div(worker_count, 8)
        for repetition in range(repetitions):
            started = time.perf_counter_ns()
            descriptors = [
                {
                    "pod_id": index,
                    "request_id": repetition,
                    "chunk_id": repetition % 9,
                    "member_range": [index * 8, min(worker_count, (index + 1) * 8)],
                }
                for index in range(descriptor_count)
            ]
            if sum(end - start for start, end in (row["member_range"] for row in descriptors)) != worker_count:
                raise RuntimeError("control descriptors did not cover all workers")
            values.append((time.perf_counter_ns() - started) / 1e6)
        results.append(
            {
                "worker_count": worker_count,
                "coarse_descriptors": descriptor_count,
                "logical_expert_rpcs": 0,
                "coordinator_issue": timing(values),
                "p50_ms_per_worker": timing(values)["p50_ms"] / worker_count,
            }
        )
    return results


def math_ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


__all__ = ["benchmark_control_plane", "benchmark_persistent_protocol"]
