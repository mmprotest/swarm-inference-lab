"""Structured lifecycle evidence for Experiment 003 worker processes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from swarm_inference.runtime.telemetry import (
    JsonlLifecycleObserver,
    configure_lifecycle_observer,
    lifecycle_observer,
    lifecycle_observer_from_environment,
)

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


LifecycleRecorder = JsonlLifecycleObserver


def configure_lifecycle_recorder(recorder: LifecycleRecorder | None) -> None:
    configure_lifecycle_observer(recorder)


def lifecycle_recorder() -> LifecycleRecorder | None:
    return cast(LifecycleRecorder | None, lifecycle_observer())


def recorder_from_environment() -> LifecycleRecorder | None:
    return lifecycle_observer_from_environment()


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
