"""Structured lifecycle evidence for Experiment 003 worker processes."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_LIFECYCLE_EVENTS = (
    "assignment_created",
    "shard_acquisition_started",
    "shard_acquisition_completed",
    "shard_verification_started",
    "shard_verification_completed",
    "process_spawn_started",
    "process_spawned",
    "python_entry_started",
    "python_imports_started",
    "python_imports_completed",
    "cuda_initialisation_started",
    "cuda_initialisation_completed",
    "stage_module_construction_started",
    "stage_module_construction_completed",
    "shard_read_started",
    "shard_read_completed",
    "weight_load_started",
    "weight_load_completed",
    "host_to_device_transfer_started",
    "host_to_device_transfer_completed",
    "worker_registration_started",
    "worker_registered",
    "local_warmup_started",
    "local_warmup_completed",
    "pipeline_warmup_started",
    "pipeline_warmup_completed",
    "worker_routable",
    "first_request_received",
    "first_stage_operation_started",
    "first_stage_operation_completed",
    "first_token_produced",
    "worker_shutdown_started",
    "worker_shutdown_completed",
)

_START_END_PAIRS = {
    "shard_acquisition_started": "shard_acquisition_completed",
    "shard_verification_started": "shard_verification_completed",
    "process_spawn_started": "process_spawned",
    "python_imports_started": "python_imports_completed",
    "cuda_initialisation_started": "cuda_initialisation_completed",
    "stage_module_construction_started": "stage_module_construction_completed",
    "shard_read_started": "shard_read_completed",
    "weight_load_started": "weight_load_completed",
    "host_to_device_transfer_started": "host_to_device_transfer_completed",
    "worker_registration_started": "worker_registered",
    "local_warmup_started": "local_warmup_completed",
    "pipeline_warmup_started": "pipeline_warmup_completed",
    "first_stage_operation_started": "first_stage_operation_completed",
    "worker_shutdown_started": "worker_shutdown_completed",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LifecycleRecorder:
    """Append process-local JSONL events with a shared monotonic origin."""

    def __init__(
        self,
        *,
        path: str | Path,
        experiment_id: str,
        worker_id: str,
        stage_id: int,
        origin_monotonic_ns: int,
        process_id: int | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.worker_id = worker_id
        self.stage_id = stage_id
        self.origin_monotonic_ns = origin_monotonic_ns
        self.process_id = os.getpid() if process_id is None else process_id
        self._lock = threading.Lock()
        self._once: set[str] = set()

    def emit(
        self,
        event_name: str,
        *,
        monotonic_ns: int | None = None,
        wall_clock_utc: str | None = None,
        duration_ns: int | None = None,
        bytes_count: int | None = None,
        memory_metrics: Mapping[str, int | float | None] | None = None,
        error: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        timestamp = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        if duration_ns is not None and duration_ns < 0:
            raise ValueError("lifecycle duration cannot be negative")
        payload: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "worker_id": self.worker_id,
            "stage_id": self.stage_id,
            "process_id": self.process_id,
            "monotonic_timestamp_ns": timestamp,
            "experiment_elapsed_ns": timestamp - self.origin_monotonic_ns,
            "wall_clock_utc": wall_clock_utc or utc_now(),
            "event_name": event_name,
        }
        if duration_ns is not None:
            payload["duration_ns"] = int(duration_ns)
            payload["duration_seconds"] = duration_ns / 1_000_000_000
        if bytes_count is not None:
            payload["bytes"] = int(bytes_count)
        if memory_metrics is not None:
            payload["memory_metrics"] = dict(memory_metrics)
        if error is not None:
            payload["error"] = error
        if details:
            payload.update(details)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
        return payload

    def emit_once(
        self,
        key: str,
        event_name: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        with self._lock:
            if key in self._once:
                return None
            self._once.add(key)
        return self.emit(event_name, **kwargs)


_RECORDER: LifecycleRecorder | None = None


def configure_lifecycle_recorder(recorder: LifecycleRecorder | None) -> None:
    global _RECORDER
    _RECORDER = recorder


def lifecycle_recorder() -> LifecycleRecorder | None:
    return _RECORDER


def recorder_from_environment() -> LifecycleRecorder | None:
    path = os.environ.get("SWARM_LIFECYCLE_FILE")
    experiment_id = os.environ.get("SWARM_EXPERIMENT_ID")
    worker_id = os.environ.get("SWARM_WORKER_ID")
    stage_id = os.environ.get("SWARM_STAGE_ID")
    origin = os.environ.get("SWARM_EXPERIMENT_ORIGIN_NS")
    if not all((path, experiment_id, worker_id, stage_id is not None, origin)):
        return None
    return LifecycleRecorder(
        path=Path(str(path)),
        experiment_id=str(experiment_id),
        worker_id=str(worker_id),
        stage_id=int(str(stage_id)),
        origin_monotonic_ns=int(str(origin)),
    )


def read_lifecycle_events(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid lifecycle JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"lifecycle row is not an object at {path}:{line_number}")
            rows.append(payload)
    # Preserve append order. Sorting here would both hide backward-clock evidence
    # and reverse start/completion events that legitimately share a timestamp.
    return rows


def write_lifecycle_events(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def validate_lifecycle_events(
    rows: list[dict[str, Any]],
    *,
    require_complete_workers: bool = True,
) -> list[str]:
    """Return fail-closed validation errors for structured lifecycle evidence."""

    errors: list[str] = []
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        required_fields = {
            "experiment_id",
            "worker_id",
            "stage_id",
            "process_id",
            "monotonic_timestamp_ns",
            "wall_clock_utc",
            "event_name",
        }
        missing = sorted(required_fields - set(row))
        if missing:
            errors.append(f"lifecycle row missing fields {missing}: {row}")
            continue
        duration = row.get("duration_ns")
        if duration is not None and int(duration) < 0:
            errors.append(f"negative duration for {row['worker_id']}:{row['event_name']}")
        key = str(row["worker_id"])
        by_worker.setdefault(key, []).append(row)
    for worker_id, worker_rows in by_worker.items():
        timestamps = [int(row["monotonic_timestamp_ns"]) for row in worker_rows]
        if timestamps != sorted(timestamps):
            errors.append(f"monotonic timestamps moved backward for {worker_id}")
        names = [str(row["event_name"]) for row in worker_rows]
        if require_complete_workers:
            missing_events = sorted(set(REQUIRED_LIFECYCLE_EVENTS) - set(names))
            if missing_events:
                errors.append(f"worker {worker_id} missing lifecycle events {missing_events}")
        first_positions = {name: names.index(name) for name in set(names)}
        for started, completed in _START_END_PAIRS.items():
            if (
                started in first_positions
                and completed in first_positions
                and first_positions[completed] < first_positions[started]
            ):
                errors.append(f"worker {worker_id} emitted {completed} before {started}")
    return errors


def pipeline_ready_time_seconds(
    rows: list[dict[str, Any]],
    *,
    experiment_worker_start_origin_ns: int,
    required_worker_ids: Iterable[str],
) -> float:
    ready_by_worker: dict[str, int] = {}
    required = set(required_worker_ids)
    for row in rows:
        worker_id = str(row.get("worker_id", ""))
        if worker_id in required and row.get("event_name") == "worker_routable":
            ready_by_worker[worker_id] = max(
                ready_by_worker.get(worker_id, 0),
                int(row["monotonic_timestamp_ns"]),
            )
    missing = sorted(required - set(ready_by_worker))
    if missing:
        raise ValueError(f"pipeline readiness is missing workers: {missing}")
    return (max(ready_by_worker.values()) - int(experiment_worker_start_origin_ns)) / 1_000_000_000


def lifecycle_duration_seconds(
    rows: Iterable[Mapping[str, Any]],
    *,
    start_event: str,
    end_event: str,
) -> float | None:
    start: int | None = None
    end: int | None = None
    for row in rows:
        name = row.get("event_name")
        timestamp = int(row.get("monotonic_timestamp_ns", 0))
        if name == start_event and start is None:
            start = timestamp
        if name == end_event:
            end = timestamp
    if start is None or end is None or end < start:
        return None
    return (end - start) / 1_000_000_000
