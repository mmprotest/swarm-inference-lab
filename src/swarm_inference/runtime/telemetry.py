"""Structured monotonic tracing and dependency-based critical-path analysis."""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

PRODUCT_EVENT_NAMES = frozenset(
    {
        "worker_registered",
        "worker_unhealthy",
        "deployment_started",
        "deployment_ready",
        "session_opened",
        "token_accepted",
        "recovery_started",
        "replacement_selected",
        "route_generation_installed",
        "replay_token_verified",
        "recovery_completed",
        "recovery_failed",
        "session_cancelled",
        "session_closed",
        "stage_unloaded",
    }
)


class ProductTelemetry:
    """Canonical product event sink independent of experiment recorders."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, event_type: str, **details: Any) -> dict[str, Any]:
        if event_type not in PRODUCT_EVENT_NAMES:
            raise ValueError(f"unknown canonical product event {event_type!r}")
        row = {
            "event_type": event_type,
            "timestamp_unix_ns": time.time_ns(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            **details,
        }
        serialized = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            self.events.append(dict(row))
            if self.path is not None:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
        return row


class LifecycleObserver(Protocol):
    """Optional sink for generic worker lifecycle events."""

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
    ) -> object: ...

    def emit_once(self, key: str, event_name: str, **values: Any) -> object | None: ...


class JsonlLifecycleObserver:
    """Append process-local lifecycle events to a shared-origin JSONL file."""

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
        from datetime import UTC, datetime

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
            "wall_clock_utc": wall_clock_utc or datetime.now(UTC).isoformat(),
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


_LIFECYCLE_OBSERVER: LifecycleObserver | None = None


def configure_lifecycle_observer(observer: LifecycleObserver | None) -> None:
    """Install the optional process-wide observer used by product components."""

    global _LIFECYCLE_OBSERVER
    _LIFECYCLE_OBSERVER = observer


def lifecycle_observer() -> LifecycleObserver | None:
    return _LIFECYCLE_OBSERVER


def lifecycle_observer_from_environment() -> JsonlLifecycleObserver | None:
    """Create the compatibility JSONL observer only when all context is explicit."""

    path = os.environ.get("SWARM_LIFECYCLE_FILE")
    experiment_id = os.environ.get("SWARM_EXPERIMENT_ID")
    worker_id = os.environ.get("SWARM_WORKER_ID")
    stage_id = os.environ.get("SWARM_STAGE_ID")
    origin = os.environ.get("SWARM_EXPERIMENT_ORIGIN_NS")
    if not all((path, experiment_id, worker_id, stage_id is not None, origin)):
        return None
    return JsonlLifecycleObserver(
        path=Path(str(path)),
        experiment_id=str(experiment_id),
        worker_id=str(worker_id),
        stage_id=int(str(stage_id)),
        origin_monotonic_ns=int(str(origin)),
    )


@dataclass(frozen=True, slots=True)
class TraceContext:
    run_id: str
    session_id: str
    request_id: str
    token_position: int
    stage_id: int
    source_stage: int
    destination_stage: int
    message_type: str
    model_revision: str


class TraceWriter:
    """Thread-safe newline-delimited JSON trace writer."""

    def __init__(self, path: Path, *, base: TraceContext) -> None:
        self.path = path
        self.base = base
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def emit(self, event: str, **values: Any) -> dict[str, Any]:
        now_ns = time.perf_counter_ns()
        row = {
            **asdict(self.base),
            "event": event,
            "monotonic_ns": now_ns,
            "wall_time_ns": time.time_ns(),
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "payload_bytes": 0,
            "wire_bytes": 0,
            "tensor_shape": [],
            "tensor_dtype": "none",
            "compression_mode": "none",
            "sequence_number": -1,
            "status": "OK",
            **values,
        }
        with self._lock:
            self._handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        return row

    def callback(self, event: str, values: dict[str, Any]) -> None:
        self.emit(event, **values)

    @contextmanager
    def span(self, name: str, **values: Any) -> Iterator[None]:
        correlation_id = f"{os.getpid()}-{threading.get_ident()}-{time.perf_counter_ns()}"
        self.emit(f"{name}_start", correlation_id=correlation_id, **values)
        started = time.perf_counter_ns()
        status = "OK"
        error = None
        try:
            yield
        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.emit(
                f"{name}_end",
                correlation_id=correlation_id,
                duration_ns=time.perf_counter_ns() - started,
                status=status,
                error=error,
                **values,
            )

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def merge_traces(paths: Sequence[Path]) -> list[dict[str, Any]]:
    events = [event for path in paths for event in read_trace(path)]
    return sorted(events, key=lambda row: (int(row["monotonic_ns"]), int(row["process_id"])))


def _durations(events: Sequence[dict[str, Any]], event_name: str) -> list[int]:
    return [
        int(row.get("duration_ns", 0)) for row in events if row.get("event") == f"{event_name}_end"
    ]


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def reconstruct_critical_path(
    events: Sequence[dict[str, Any]], *, generated_tokens: int
) -> dict[str, Any]:
    """Reconstruct serial dependencies from linked receive/compute events.

    A receive counts only when marked token-critical and its
    ``unblocks_event_id`` resolves to a later required compute event. Aggregate
    message totals are never substituted for serial dependency counts.
    """

    if generated_tokens < 0:
        raise ValueError("generated token count cannot be negative")
    by_id = {
        str(row["event_id"]): row
        for row in events
        if row.get("event_id") and row.get("event") == "cuda_compute_start"
    }
    dependencies: list[dict[str, Any]] = []
    invalid_links: list[str] = []
    for row in events:
        if row.get("event") != "socket_receive_end" or not row.get("critical_dependency"):
            continue
        target_id = str(row.get("unblocks_event_id", ""))
        target = by_id.get(target_id)
        if (
            target is None
            or target.get("event") != "cuda_compute_start"
            or int(target["monotonic_ns"]) < int(row["monotonic_ns"])
        ):
            invalid_links.append(str(row.get("event_id", "missing-event-id")))
            continue
        dependencies.append(
            {
                "receive_event_id": row.get("event_id"),
                "compute_event_id": target_id,
                "token_position": int(row.get("dependency_token_position", row["token_position"])),
                "source_stage": int(row["source_stage"]),
                "destination_stage": int(row["destination_stage"]),
                "message_type": row["message_type"],
                "wait_ns": max(0, int(row.get("duration_ns", 0))),
            }
        )
    by_token: Counter[int] = Counter(dep["token_position"] for dep in dependencies)
    per_token = [by_token[position] for position in range(generated_tokens)]
    sends = [
        row
        for row in events
        if row.get("event") == "socket_send_end"
        and row.get("data_plane") == "ring"
        and row.get("message_type") in {"PREFILL", "DECODE", "VERIFY_CANDIDATES", "TOKEN_RESULT"}
    ]
    payload_bytes = sum(int(row.get("payload_bytes", 0)) for row in sends)
    wire_bytes = sum(int(row.get("wire_bytes", 0)) for row in sends)
    token_latencies = [
        int(row.get("duration_ns", 0)) / 1e9
        for row in events
        if row.get("event") == "token_step_end"
    ]
    stage_compute: defaultdict[int, int] = defaultdict(int)
    for row in events:
        if row.get("event") == "cuda_compute_end":
            stage_compute[int(row["stage_id"])] += int(row.get("duration_ns", 0))
    maximum_stage_compute = max(stage_compute.values(), default=0)
    minimum_stage_compute = min(stage_compute.values(), default=0)
    overlap_events = [row for row in events if row.get("event") == "communication_compute_overlap"]
    return {
        "definition": (
            "A network dependency on the token-critical path that must complete before "
            "the next required model computation can begin."
        ),
        "dependency_edges": dependencies,
        "invalid_dependency_links": invalid_links,
        "serial_waits_total": len(dependencies),
        "serial_waits_per_token": statistics.median(per_token) if per_token else 0.0,
        "serial_waits_by_token": per_token,
        "network_messages_total": len(sends),
        "messages_per_token": len(sends) / generated_tokens if generated_tokens else 0.0,
        "payload_bytes_total": payload_bytes,
        "payload_bytes_per_token": (payload_bytes / generated_tokens if generated_tokens else 0.0),
        "wire_bytes_total": wire_bytes,
        "wire_bytes_per_token": (wire_bytes / generated_tokens if generated_tokens else 0.0),
        "serialisation_ns_per_token": sum(
            int(row.get("duration_ns", 0))
            for row in events
            if row.get("event") == "serialization_end" and row.get("data_plane") == "ring"
        )
        / max(generated_tokens, 1),
        "compression_ns_per_token": sum(_durations(events, "compression"))
        / max(generated_tokens, 1),
        "socket_ns_per_token": sum(int(row.get("duration_ns", 0)) for row in sends)
        / max(generated_tokens, 1),
        "queue_ns_per_token": sum(
            int(row.get("duration_ns", 0))
            for row in events
            if row.get("event") == "stage_queue_exit"
        )
        / max(generated_tokens, 1),
        "model_compute_ns_per_token": sum(_durations(events, "cuda_compute"))
        / max(generated_tokens, 1),
        "coordinator_blocked_ns_per_token": sum(_durations(events, "coordinator_blocked"))
        / max(generated_tokens, 1),
        "communication_compute_overlap_ns": sum(
            int(row.get("duration_ns", 0)) for row in overlap_events
        ),
        "gpu_idle_ns": sum(_durations(events, "gpu_idle")),
        "stage_compute_ns": dict(sorted(stage_compute.items())),
        "stage_imbalance_ratio": (
            maximum_stage_compute / minimum_stage_compute if minimum_stage_compute > 0 else 0.0
        ),
        "end_to_end_token_latency_mean_s": (
            statistics.mean(token_latencies) if token_latencies else 0.0
        ),
        "time_to_first_token_s": token_latencies[0] if token_latencies else 0.0,
        "inter_token_latency_p50_s": _quantile(token_latencies[1:], 0.50),
        "inter_token_latency_p95_s": _quantile(token_latencies[1:], 0.95),
    }


__all__ = [
    "PRODUCT_EVENT_NAMES",
    "JsonlLifecycleObserver",
    "LifecycleObserver",
    "ProductTelemetry",
    "TraceContext",
    "TraceWriter",
    "configure_lifecycle_observer",
    "lifecycle_observer",
    "lifecycle_observer_from_environment",
    "merge_traces",
    "read_trace",
    "reconstruct_critical_path",
]
