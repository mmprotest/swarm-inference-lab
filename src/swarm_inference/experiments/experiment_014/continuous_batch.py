"""Real Kimi continuous-batching certification for H014-031.

The scheduler in this module is intentionally small: it owns a bounded FIFO,
maps logical streams to persistent executor sessions, and performs no model
arithmetic.  The production stage executor remains the sole CUDA owner.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _process_snapshot,
    _timing,
    _warm_lifecycle_delta,
)
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import (
    _known_safe_fixture,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.persistent_stages import (
    MODEL_CONTENT_FINGERPRINT,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    _CaptureConnectionPool,
    _source_assignment,
    _stage_fixtures,
)
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

SCHEMA_VERSION = "experiment-014-k3-continuous-batch-v1"
MINIMUM_CAPACITY_GAIN = 1.7


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@dataclass(slots=True)
class _PendingDecode:
    stream_id: str
    session_id: str
    hidden_states: torch.Tensor
    position: int
    enqueued_ns: int
    sequence: int


class PersistentKimiContinuousBatchScheduler:
    """Bounded FIFO over persistent, independently stateful Kimi sessions."""

    def __init__(
        self,
        executor: PersistentKimiStageExecutor,
        *,
        maximum_batch: int = 8,
        maximum_active_streams: int = 8,
        session_prefix: str = "continuous",
    ) -> None:
        if maximum_batch not in executor.runtime.expert_supported_batches:
            raise ValueError("scheduler batch is absent from the native certified set")
        if maximum_batch > executor.lifecycle_snapshot()["batch_capacity"]:
            raise ValueError("scheduler batch exceeds the stage workspace capacity")
        if maximum_active_streams < maximum_batch:
            raise ValueError("active-stream capacity cannot be smaller than maximum batch")
        self.executor = executor
        self.maximum_batch = maximum_batch
        self.maximum_active_streams = maximum_active_streams
        self.session_prefix = session_prefix
        self._active: dict[str, str] = {}
        self._pending: deque[_PendingDecode] = deque()
        self._generation = 0
        self._sequence = 0
        self._dispatches = 0
        self._cancellations = 0
        self._slot_reuses = 0
        self._ever_full = False

    @property
    def active_stream_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def open_stream(self, stream_id: str, *, maximum_context: int) -> str:
        if not stream_id or stream_id in self._active:
            raise ValueError("continuous stream is empty or already active")
        if len(self._active) >= self.maximum_active_streams:
            raise RuntimeError("continuous scheduler has no free stream slot")
        reused = self._ever_full and len(self._active) < self.maximum_active_streams
        self._generation += 1
        session_id = f"{self.session_prefix}-{self._generation:04d}-{stream_id}"
        self.executor.open_session(
            session_id, maximum_context_override=maximum_context
        )
        self._active[stream_id] = session_id
        self._ever_full = self._ever_full or (
            len(self._active) == self.maximum_active_streams
        )
        self._slot_reuses += int(reused)
        return session_id

    def session_id(self, stream_id: str) -> str:
        try:
            return self._active[stream_id]
        except KeyError as exc:
            raise KeyError(f"unknown continuous stream {stream_id!r}") from exc

    def submit(
        self, stream_id: str, hidden_states: torch.Tensor, *, position: int
    ) -> int:
        session_id = self.session_id(stream_id)
        if any(item.stream_id == stream_id for item in self._pending):
            raise RuntimeError("continuous stream already has pending decode work")
        if hidden_states.dtype != torch.float32 or tuple(hidden_states.shape) != (
            1,
            9,
            self.executor.config.hidden,
        ):
            raise ValueError("continuous Kimi boundary must be float32 [1,9,7168]")
        self._sequence += 1
        self._pending.append(
            _PendingDecode(
                stream_id=stream_id,
                session_id=session_id,
                hidden_states=hidden_states,
                position=int(position),
                enqueued_ns=time.perf_counter_ns(),
                sequence=self._sequence,
            )
        )
        return self._sequence

    def dispatch_once(self) -> dict[str, Any]:
        if not self._pending:
            raise RuntimeError("continuous scheduler dispatch requested with an empty FIFO")
        items = [self._pending.popleft() for _ in range(min(self.maximum_batch, len(self._pending)))]
        started_ns = time.perf_counter_ns()
        hidden = torch.cat([item.hidden_states for item in items], dim=0)
        record = self.executor.execute_decode_batch(
            session_ids=tuple(item.session_id for item in items),
            hidden_states=hidden,
            cache_position_starts=tuple(item.position for item in items),
        )
        completed_ns = time.perf_counter_ns()
        # The executor has already built a fresh owned batch boundary.  Row
        # views keep that allocation alive, so a second 2 MiB host copy is not
        # needed for scheduler ownership or isolation.
        outputs = np.asarray(record["boundary_output"], dtype=np.float32)
        self._dispatches += 1
        rows = []
        for row, item in enumerate(items):
            rows.append(
                {
                    "stream_id": item.stream_id,
                    "session_id": item.session_id,
                    "position": item.position,
                    "sequence": item.sequence,
                    "enqueued_ns": item.enqueued_ns,
                    "started_ns": started_ns,
                    "completed_ns": completed_ns,
                    "queue_delay_ms": (started_ns - item.enqueued_ns) / 1e6,
                    "response_ms": (completed_ns - item.enqueued_ns) / 1e6,
                    "selected_expert_ids": record["selected_expert_ids"][row],
                    "selected_weights": record["selected_weights"][row],
                    "output": outputs[row],
                }
            )
        return {
            "batch_size": len(items),
            "positions": [item.position for item in items],
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "wall_ms": (completed_ns - started_ns) / 1e6,
            "formation_delay_ms": (
                started_ns - min(item.enqueued_ns for item in items)
            )
            / 1e6,
            "last_enqueue_to_start_ms": (
                started_ns - max(item.enqueued_ns for item in items)
            )
            / 1e6,
            "device_ms": float(record["device_ms"]),
            "routing": record["routing"],
            "rows": rows,
        }

    def cancel_stream(self, stream_id: str) -> dict[str, Any]:
        session_id = self.session_id(stream_id)
        retained: deque[_PendingDecode] = deque()
        removed = 0
        while self._pending:
            item = self._pending.popleft()
            if item.stream_id == stream_id:
                removed += 1
            else:
                retained.append(item)
        self._pending = retained
        released = self.executor.cancel_session(session_id)
        del self._active[stream_id]
        self._cancellations += 1
        return {
            "stream_id": stream_id,
            "session_id": session_id,
            "pending_requests_removed": removed,
            "released_state_bytes": released,
        }

    def close_stream(self, stream_id: str) -> int:
        if any(item.stream_id == stream_id for item in self._pending):
            raise RuntimeError("cannot close a stream with pending decode work")
        session_id = self.session_id(stream_id)
        released = self.executor.close_session(session_id)
        del self._active[stream_id]
        return released

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_batch": self.maximum_batch,
            "maximum_active_streams": self.maximum_active_streams,
            "active_streams": sorted(self._active),
            "pending_count": len(self._pending),
            "dispatches": self._dispatches,
            "cancellations": self._cancellations,
            "slot_reuses": self._slot_reuses,
            "worker_threads_created": 0,
            "async_tasks_created": 0,
            "connections_created": 0,
        }

    def close(self) -> None:
        if self._pending:
            raise RuntimeError("cannot close scheduler with pending work")
        for stream_id in list(self._active):
            self.close_stream(stream_id)


def _fixture_tensor(fixtures: list[np.ndarray], stream: int, step: int) -> torch.Tensor:
    return torch.from_numpy(fixtures[(stream + step) % len(fixtures)].copy())


def _close_direct_sessions(
    executor: PersistentKimiStageExecutor, session_ids: list[str]
) -> None:
    for session_id in session_ids:
        with suppress(KeyError):
            executor.close_session(session_id)


def _prime_session(
    executor: PersistentKimiStageExecutor,
    session_id: str,
    fixtures: list[np.ndarray],
    *,
    stream: int,
    steps: int,
) -> None:
    for position in range(steps):
        executor.execute_decode(
            session_id=session_id,
            hidden_states=_fixture_tensor(fixtures, stream, position),
            cache_position_start=position,
        )


def _position_aware_correctness(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    layer: int,
    batch: int,
) -> dict[str, Any]:
    scheduler = PersistentKimiContinuousBatchScheduler(
        executor,
        maximum_batch=batch,
        maximum_active_streams=batch,
        session_prefix=f"h014-031-l{layer}-correct",
    )
    controls: list[str] = []
    offsets = list(range(batch))
    comparisons: list[dict[str, Any]] = []
    try:
        for row, offset in enumerate(offsets):
            stream_id = f"stream-{row}"
            scheduled_session = scheduler.open_stream(stream_id, maximum_context=offset + 3)
            control = f"h014-031-l{layer}-control-{row}"
            executor.open_session(control, maximum_context_override=offset + 3)
            controls.append(control)
            _prime_session(
                executor, scheduled_session, fixtures, stream=row, steps=offset
            )
            _prime_session(executor, control, fixtures, stream=row, steps=offset)

        for round_index in range(3):
            expected: list[np.ndarray] = []
            expected_routes: list[list[int]] = []
            for row, control in enumerate(controls):
                position = offsets[row] + round_index
                result = executor.execute_decode(
                    session_id=control,
                    hidden_states=_fixture_tensor(fixtures, row, position),
                    cache_position_start=position,
                )
                expected.append(
                    result.stage_boundary_hidden_states.detach().cpu().numpy()[0].copy()
                )
                expected_routes.append(
                    list(executor.execution_records[-1]["selected_expert_ids"])
                )
                scheduler.submit(
                    f"stream-{row}",
                    _fixture_tensor(fixtures, row, position),
                    position=position,
                )
            dispatch = scheduler.dispatch_once()
            for row, observed in enumerate(dispatch["rows"]):
                metrics = _numerical_metrics(observed["output"], expected[row])
                comparisons.append(
                    {
                        "round": round_index,
                        "stream": row,
                        "position": offsets[row] + round_index,
                        "metrics": metrics,
                        "routes_equal": observed["selected_expert_ids"]
                        == expected_routes[row],
                        "observed_fingerprint": _array_fingerprint(observed["output"]),
                        "expected_fingerprint": _array_fingerprint(expected[row]),
                    }
                )

        state_rows = []
        for row, control in enumerate(controls):
            observed_state = executor.session_state_evidence(
                scheduler.session_id(f"stream-{row}")
            )
            expected_state = executor.session_state_evidence(control)
            state_rows.append(
                {
                    "stream": row,
                    "observed": observed_state,
                    "expected": expected_state,
                    "fingerprint_equal": observed_state["fingerprint"]
                    == expected_state["fingerprint"],
                }
            )
    finally:
        scheduler.close()
        _close_direct_sessions(executor, controls)

    maximum_error = max(
        float(row["metrics"]["relative_l2_error"]) for row in comparisons
    )
    passed = (
        maximum_error <= 1e-7
        and all(bool(row["routes_equal"]) for row in comparisons)
        and all(bool(row["fingerprint_equal"]) for row in state_rows)
    )
    return {
        "offsets": offsets,
        "rounds": 3,
        "comparisons": comparisons,
        "maximum_relative_l2_error": maximum_error,
        "exact_routes": all(bool(row["routes_equal"]) for row in comparisons),
        "exact_state_fingerprints": all(
            bool(row["fingerprint_equal"]) for row in state_rows
        ),
        "state_evidence": state_rows,
        "pass": passed,
    }


def _batch1_baseline(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    layer: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    session_id = f"h014-031-l{layer}-batch1-baseline"
    executor.open_session(session_id, maximum_context_override=warmup + iterations)
    device_ms: list[float] = []
    wall_ms: list[float] = []
    try:
        for position in range(warmup + iterations):
            started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=(session_id,),
                hidden_states=_fixture_tensor(fixtures, 0, position),
                cache_position_starts=(position,),
            )
            if position >= warmup:
                wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                device_ms.append(float(record["device_ms"]))
    finally:
        executor.close_session(session_id)
    return {
        "warmup_calls": warmup,
        "retained_calls": iterations,
        "device": _timing(device_ms),
        "wall": _timing(wall_ms),
        "aggregate_rows_per_second": 1000.0 / _timing(device_ms)["p50_ms"],
    }


def _measure_continuous(
    executor: PersistentKimiStageExecutor,
    runtime: PersistentStageRuntime,
    fixtures: list[np.ndarray],
    baseline: dict[str, Any],
    *,
    layer: int,
    batch: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], PersistentKimiContinuousBatchScheduler]:
    scheduler = PersistentKimiContinuousBatchScheduler(
        executor,
        maximum_batch=batch,
        maximum_active_streams=batch,
        session_prefix=f"h014-031-l{layer}-perf",
    )
    offsets = list(range(batch))
    for row, offset in enumerate(offsets):
        session = scheduler.open_stream(
            f"stream-{row}", maximum_context=offset + warmup + iterations + 4
        )
        _prime_session(executor, session, fixtures, stream=row, steps=offset)

    device_ms: list[float] = []
    wall_ms: list[float] = []
    formation_ms: list[float] = []
    last_enqueue_ms: list[float] = []
    queue_ms: list[float] = []
    response_ms: list[float] = []
    completion_ns: dict[str, list[int]] = {f"stream-{row}": [] for row in range(batch)}
    completion_counts = {f"stream-{row}": 0 for row in range(batch)}
    routing_rows: list[dict[str, Any]] = []
    before_retained: dict[str, Any] | None = None
    for round_index in range(warmup + iterations):
        if round_index == warmup:
            # Snapshot before requests enter the measured FIFO.  Taking this
            # after submit would charge process inspection to queue latency.
            before_retained = _process_snapshot(runtime, executor)
        for row, offset in enumerate(offsets):
            position = offset + round_index
            scheduler.submit(
                f"stream-{row}",
                _fixture_tensor(fixtures, row, position),
                position=position,
            )
        dispatch = scheduler.dispatch_once()
        if round_index >= warmup:
            device_ms.append(dispatch["device_ms"])
            wall_ms.append(dispatch["wall_ms"])
            formation_ms.append(dispatch["formation_delay_ms"])
            last_enqueue_ms.append(dispatch["last_enqueue_to_start_ms"])
            routing_rows.append(dispatch["routing"])
            for row in dispatch["rows"]:
                stream_id = row["stream_id"]
                queue_ms.append(row["queue_delay_ms"])
                response_ms.append(row["response_ms"])
                completion_ns[stream_id].append(row["completed_ns"])
                completion_counts[stream_id] += 1
    after_retained = _process_snapshot(runtime, executor)
    if before_retained is None:
        raise RuntimeError("continuous benchmark did not enter retained execution")

    cadence_by_stream: dict[str, dict[str, Any]] = {}
    for stream_id, timestamps in completion_ns.items():
        cadence = [
            (timestamps[index] - timestamps[index - 1]) / 1e6
            for index in range(1, len(timestamps))
        ]
        cadence_by_stream[stream_id] = _timing(cadence)
    device = _timing(device_ms)
    wall = _timing(wall_ms)
    gain = batch * float(baseline["device"]["p50_ms"]) / float(device["p50_ms"])
    lifecycle = _warm_lifecycle_delta(before_retained, after_retained)
    state_rows = [
        executor.session_state_evidence(scheduler.session_id(f"stream-{row}"))
        for row in range(batch)
    ]
    total_selections = sum(int(row["total_selections"]) for row in routing_rows)
    native_calls = sum(int(row["native_routed_expert_calls"]) for row in routing_rows)
    return (
        {
            "active_streams": batch,
            "position_offsets": offsets,
            "warmup_rounds": warmup,
            "retained_rounds": iterations,
            "retained_rows": batch * iterations,
            "device": device,
            "wall": wall,
            "aggregate_device_rows_per_second": batch * 1000.0 / device["p50_ms"],
            "aggregate_wall_rows_per_second": batch * 1000.0 / wall["p50_ms"],
            "per_row_device_service_ms": device["p50_ms"] / batch,
            "capacity_gain_vs_batch1": gain,
            "queue_delay": _timing(queue_ms),
            "response": _timing(response_ms),
            "batch_formation_delay": _timing(formation_ms),
            "last_enqueue_to_dispatch": _timing(last_enqueue_ms),
            "per_stream_cadence": cadence_by_stream,
            "fairness": {
                "completion_counts": completion_counts,
                "minimum_completions": min(completion_counts.values()),
                "maximum_completions": max(completion_counts.values()),
                "max_min_completion_ratio": max(completion_counts.values())
                / min(completion_counts.values()),
                "equal_completion_counts": len(set(completion_counts.values())) == 1,
            },
            "routing": {
                "total_selections": total_selections,
                "native_routed_expert_calls": native_calls,
                "effective_weight_reuse_rows_per_native_call": total_selections
                / native_calls,
                "mean_unique_experts": float(
                    np.mean([int(row["unique_experts"]) for row in routing_rows])
                ),
                "maximum_rows_for_one_expert": max(
                    int(row["maximum_rows_for_one_expert"]) for row in routing_rows
                ),
            },
            "state": {
                "bytes_each": executor.kv_cache_bytes(scheduler.session_id("stream-0")),
                "bytes_total": sum(int(row["bytes"]) for row in state_rows),
                "evidence": state_rows,
            },
            "lifecycle_delta": lifecycle,
            "lifecycle_deltas_zero": all(value == 0 for value in lifecycle.values()),
            "scheduler": scheduler.snapshot(),
        },
        scheduler,
    )


def _cancel_and_reuse(
    executor: PersistentKimiStageExecutor,
    scheduler: PersistentKimiContinuousBatchScheduler,
    fixtures: list[np.ndarray],
    *,
    layer: int,
    batch: int,
) -> dict[str, Any]:
    cancelled = "stream-3"
    cancelled_position = int(
        executor.session_state_evidence(scheduler.session_id(cancelled))[
            "cache_sequence_length"
        ]
    )
    scheduler.submit(
        cancelled,
        _fixture_tensor(fixtures, 3, cancelled_position),
        position=cancelled_position,
    )
    cancellation = scheduler.cancel_stream(cancelled)
    stale_submit_rejected = False
    try:
        scheduler.submit(
            cancelled,
            _fixture_tensor(fixtures, 3, cancelled_position),
            position=cancelled_position,
        )
    except KeyError:
        stale_submit_rejected = True

    replacement = "replacement-3"
    replacement_session = scheduler.open_stream(replacement, maximum_context=3)
    control = f"h014-031-l{layer}-replacement-control"
    executor.open_session(control, maximum_context_override=3)
    comparisons: list[dict[str, Any]] = []
    try:
        for position in range(3):
            control_result = executor.execute_decode(
                session_id=control,
                hidden_states=_fixture_tensor(fixtures, 3, position),
                cache_position_start=position,
            )
            control_output = (
                control_result.stage_boundary_hidden_states.detach().cpu().numpy()[0].copy()
            )
            control_routes = list(
                executor.execution_records[-1]["selected_expert_ids"]
            )
            scheduler.submit(
                replacement,
                _fixture_tensor(fixtures, 3, position),
                position=position,
            )
            observed = scheduler.dispatch_once()["rows"][0]
            comparisons.append(
                {
                    "position": position,
                    "metrics": _numerical_metrics(observed["output"], control_output),
                    "routes_equal": observed["selected_expert_ids"] == control_routes,
                }
            )
        observed_state = executor.session_state_evidence(replacement_session)
        control_state = executor.session_state_evidence(control)
    finally:
        executor.close_session(control)

    passed = (
        cancellation["pending_requests_removed"] == 1
        and cancellation["released_state_bytes"] > 0
        and stale_submit_rejected
        and max(float(row["metrics"]["relative_l2_error"]) for row in comparisons)
        <= 1e-7
        and all(bool(row["routes_equal"]) for row in comparisons)
        and observed_state["fingerprint"] == control_state["fingerprint"]
        and scheduler.active_stream_count == batch
        and scheduler.pending_count == 0
    )
    return {
        "cancellation": cancellation,
        "stale_submit_rejected": stale_submit_rejected,
        "replacement_session": replacement_session,
        "comparisons": comparisons,
        "replacement_state": observed_state,
        "control_state": control_state,
        "state_fingerprint_equal": observed_state["fingerprint"]
        == control_state["fingerprint"],
        "scheduler_after_reuse": scheduler.snapshot(),
        "pass": passed,
    }


async def _benchmark_continuous_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int,
    device: int,
    batch: int,
    warmup: int,
    iterations: int,
    cycle_id: str,
) -> dict[str, Any]:
    if batch != 8:
        raise ValueError("H014-031 certifies only the already safe complete-stage batch 8")
    if warmup < 3 or iterations < 20:
        raise ValueError("continuous benchmark requires >=3 warmup and >=20 retained rounds")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "identity_manifest": identity_manifest.resolve(),
        "graph_certification": graph_certification.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    fixture = graph.get("fixture", {})
    provenance = {
        "graph_status": graph.get("status"),
        "trace_matches": fixture.get("oracle_trace_sha256")
        == _sha256_file(paths["oracle_trace"]),
        "routes_matches": fixture.get("oracle_routes_sha256")
        == _sha256_file(paths["oracle_routes"]),
    }
    provenance["pass"] = (
        provenance["graph_status"] == "PASS"
        and provenance["trace_matches"]
        and provenance["routes_matches"]
    )
    if not provenance["pass"]:
        raise ValueError("continuous benchmark oracle provenance is not graph-certified")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "A position-aware FIFO can combine eight independent layer-89 decode "
                "streams exactly, retain >=1.7x aggregate device capacity, preserve "
                "fairness, and cancel/reuse one slot without state interference."
            ),
            "minimum_capacity_gain": MINIMUM_CAPACITY_GAIN,
        },
        "configuration": {
            "layer": layer,
            "device": device,
            "active_streams": batch,
            "maximum_batch": batch,
            "warmup_rounds": warmup,
            "retained_rounds": iterations,
            "position_offsets": list(range(batch)),
        },
        "implementation": {
            "scheduler": "bounded in-process FIFO, no threads or async tasks",
            "state": "one persistent CUDA-owned session per active stream",
            "formation": "oldest ready rows, up to certified batch 8",
            "positions": "one validated cache position per row",
            "cuda_arithmetic_change": False,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "oracle_provenance": provenance,
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    receipt["gpu_health_before"] = _health_snapshot(device)
    receipt["device_identity"] = _device_identity(device)
    retain("gpu_health_before", status=receipt["gpu_health_before"]["status"])
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before continuous batching")

    assignment = _source_assignment(checkpoint, layer=layer, device=f"native-cuda:{device}")
    fixtures, _ = _stage_fixtures(checkpoint, oracle_trace, layer=layer)
    runtime = PersistentStageRuntime(
        worker_id=f"{cycle_id.lower()}-worker-{layer:03d}",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=24,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=_CaptureConnectionPool(),  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=f"{cycle_id.lower()}-layer-{layer}",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=max(128, warmup + iterations + batch + 8),
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    scheduler: PersistentKimiContinuousBatchScheduler | None = None
    try:
        started = time.perf_counter_ns()
        response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        receipt["load"] = {
            "accepted": response.accepted,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            "resident_device_bytes": executor.resident_device_bytes,
            "weight_fingerprint": executor.weight_fingerprint,
            "lifecycle": executor.lifecycle_snapshot(),
        }
        retain("loaded_and_prepared", accepted=response.accepted)

        correctness = _position_aware_correctness(
            executor, fixtures, layer=layer, batch=batch
        )
        receipt["correctness"] = correctness
        retain(
            "position_aware_correctness",
            status="PASS" if correctness["pass"] else "FAIL",
            maximum_relative_l2_error=correctness["maximum_relative_l2_error"],
        )
        if not correctness["pass"]:
            raise RuntimeError("position-aware scheduler differs from serial execution")

        baseline = _batch1_baseline(
            executor,
            fixtures,
            layer=layer,
            warmup=warmup,
            iterations=iterations,
        )
        receipt["batch1_baseline"] = baseline
        retain("batch1_baseline", device_p50_ms=baseline["device"]["p50_ms"])

        measured, scheduler = _measure_continuous(
            executor,
            runtime,
            fixtures,
            baseline,
            layer=layer,
            batch=batch,
            warmup=warmup,
            iterations=iterations,
        )
        receipt["continuous"] = measured
        retain(
            "continuous_retained",
            device_p50_ms=measured["device"]["p50_ms"],
            capacity_gain=measured["capacity_gain_vs_batch1"],
            fairness=measured["fairness"]["equal_completion_counts"],
        )

        cancellation = _cancel_and_reuse(
            executor, scheduler, fixtures, layer=layer, batch=batch
        )
        receipt["cancellation_and_slot_reuse"] = cancellation
        retain("cancellation_and_slot_reuse", status="PASS" if cancellation["pass"] else "FAIL")

        scheduler.close()
        scheduler = None
        safe = _known_safe_fixture(executor, fixtures[0], layer=layer, batch=batch)
        executor.runtime.synchronize()
        health_after = _health_snapshot(device)
        receipt["post_run_checks"] = {
            "known_safe_fixture": safe,
            "cuda_synchronize": "PASS",
            "cuda_error_state_ok": executor.runtime.error_state_ok(),
            "free_vram_bytes": executor.runtime.mem_info()["free_bytes"],
            "nvidia_smi": health_after,
        }
        execution_pass = (
            correctness["pass"]
            and measured["fairness"]["equal_completion_counts"]
            and measured["lifecycle_deltas_zero"]
            and cancellation["pass"]
            and safe["pass"]
            and receipt["post_run_checks"]["cuda_error_state_ok"]
            and health_after["status"] == "MEASURED"
        )
        gate_evaluation = {
            "capacity_gain_at_least_1_7": measured["capacity_gain_vs_batch1"]
            >= MINIMUM_CAPACITY_GAIN,
            "formation_p99_below_0_5_ms": measured["batch_formation_delay"][
                "p99_ms"
            ]
            < 0.5,
            "response_p99_below_16_ms": measured["response"]["p99_ms"] < 16.0,
            "observed_formation_p99_ms": measured["batch_formation_delay"]["p99_ms"],
            "observed_response_p99_ms": measured["response"]["p99_ms"],
        }
        supported = execution_pass and all(
            bool(gate_evaluation[name])
            for name in (
                "capacity_gain_at_least_1_7",
                "formation_p99_below_0_5_ms",
                "response_p99_below_16_ms",
            )
        )
        receipt["execution_pass"] = execution_pass
        receipt["hypothesis_gate_evaluation"] = gate_evaluation
        receipt["hypothesis_supported"] = supported
        receipt["inspection"] = {
            "actual_bottleneck": (
                "complete-stage CUDA service; FIFO formation p99 is "
                f"{measured['batch_formation_delay']['p99_ms']:.6f} ms versus "
                f"{measured['device']['p99_ms']:.6f} ms device service"
            ),
            "fixture_limitation": (
                "Eight streams use the three retained real layer boundaries with "
                "different state positions; physical request-arrival jitter is not replayed."
            ),
        }
        receipt["decision"] = {
            "scheduler": "RETAIN" if supported else "RETAIN_WITH_MEASURED_TAIL",
            "production_batch": batch if execution_pass else None,
            "next_hypothesis": (
                "Separate decode from prefill and measure the retained production "
                "batch under explicit arrival-rate and context-length workloads."
            ),
        }
        receipt["status"] = "PASS" if execution_pass else "FAIL"
        retain("complete", status=receipt["status"])
        return receipt
    finally:
        if scheduler is not None:
            with suppress(Exception):
                scheduler.close()
        await runtime.close()


def benchmark_continuous_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    layer: int = 89,
    device: int = 0,
    batch: int = 8,
    warmup: int = 10,
    iterations: int = 50,
    cycle_id: str = "H014-031a",
) -> dict[str, Any]:
    """Run and atomically retain the position-aware continuous-batch test."""
    try:
        return asyncio.run(
            _benchmark_continuous_batch(
                checkpoint,
                cuda_library,
                oracle_trace,
                oracle_routes,
                identity_manifest,
                graph_certification,
                output_path,
                layer=layer,
                device=device,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
                cycle_id=cycle_id,
            )
        )
    except Exception as exc:
        if output_path.exists():
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle_id,
                "status": "RUNNING",
                "progress": [],
            }
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
