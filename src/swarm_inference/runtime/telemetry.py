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
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel

if TYPE_CHECKING:
    from swarm_inference.engines.interfaces import ClusterCapabilities, ExecutionPlan
    from swarm_inference.model.descriptor import ResolvedModelDescriptor
    from swarm_inference.protocol.product import ProductStagePlan

PRODUCT_EVENT_NAMES = frozenset(
    {
        "worker_registration_rejected",
        "worker_registered",
        "worker_unhealthy",
        "deployment_started",
        "deployment_progress",
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
        "engine_process_started",
        "engine_process_stopped",
        "fast_path_profiled",
        "inference_recorded",
    }
)


class DeviceResidencyRecord(StrictModel):
    """One immutable snapshot of where an execution allocation resides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    device_id: str
    category: Literal[
        "weights",
        "kv-cache",
        "expert-cache",
        "cuda-graph",
        "compile-artifact",
        "gguf-rpc-cache",
    ]
    tier: Literal["vram", "ram", "mapped", "storage"]
    bytes: NonNegativeInt
    identity: str = ""


class DataMovementRecord(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    destination: str
    bytes: NonNegativeInt
    elapsed_ms: float = Field(ge=0)
    reason: str


class InferenceTelemetryRecord(StrictModel):
    """Complete engine-neutral product record for one terminal inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    request_id: str
    model_id: str
    model_revision: str
    model_fingerprint: str
    model_format: str
    quantization: str | None = None
    tokenizer_identity: str | None = None
    engine_id: str
    engine_revision: str | None = None
    engine_runtime_revisions: dict[str, str] = Field(default_factory=dict)
    execution_identity: str
    adapter_id: str | None = None
    fast_paths: dict[str, str] = Field(default_factory=dict)
    topology: str
    workers: tuple[str, ...]
    worker_roles: dict[str, str]
    stage_assignments: tuple[dict[str, Any], ...] = ()
    expert_assignments: tuple[dict[str, Any], ...] = ()
    microshard_assignments: tuple[dict[str, Any], ...] = ()
    residency: tuple[DeviceResidencyRecord, ...] = ()
    movement: tuple[DataMovementRecord, ...] = ()
    generated_tokens: NonNegativeInt = 0
    ttft_ms: float | None = Field(default=None, ge=0)
    prefill_tokens_s: float | None = Field(default=None, ge=0)
    decode_tokens_s: float | None = Field(default=None, ge=0)
    aggregate_tokens_s: float | None = Field(default=None, ge=0)
    inter_token_latency_ms: tuple[float, ...] = ()
    serial_waits_per_token: float | None = Field(default=None, ge=0)
    messages_per_token: float | None = Field(default=None, ge=0)
    payload_bytes_per_token: float | None = Field(default=None, ge=0)
    network_bytes: NonNegativeInt | None = None
    queue_time_ms: float | None = Field(default=None, ge=0)
    compute_time_ms: float | None = Field(default=None, ge=0)
    serialization_time_ms: float | None = Field(default=None, ge=0)
    cache_hits: NonNegativeInt | None = None
    cache_misses: NonNegativeInt | None = None
    bytes_loaded: NonNegativeInt | None = None
    bytes_evicted: NonNegativeInt | None = None
    prefetch_useful_bytes: NonNegativeInt | None = None
    prefetch_wasted_bytes: NonNegativeInt | None = None
    cache_stall_time_ms: float | None = Field(default=None, ge=0)
    fallbacks: tuple[str, ...] = ()
    recoveries: NonNegativeInt = 0
    exactness_verified: bool = False
    metric_sources: dict[str, str] = Field(default_factory=dict)
    engine_metrics: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "failed", "cancelled"]
    recorded_at_unix_ns: NonNegativeInt = Field(default_factory=time.time_ns)


def _nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result >= 0 else None


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = int(value)
    return result if result >= 0 else None


def _metric(mapping: Mapping[str, Any], *path: str) -> object:
    value: object = mapping
    for component in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(component)
    return value


def _first_float(mapping: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float | None:
    for path in paths:
        value = _nonnegative_float(_metric(mapping, *path))
        if value is not None:
            return value
    return None


def _first_int(mapping: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> int | None:
    for path in paths:
        value = _nonnegative_int(_metric(mapping, *path))
        if value is not None:
            return value
    return None


def _json_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a durable JSON representation without losing terminal evidence."""

    if not metrics:
        return {}
    loaded = json.loads(json.dumps(dict(metrics), sort_keys=True, default=str))
    if not isinstance(loaded, dict):  # pragma: no cover - JSON source is constructed above
        raise TypeError("runtime metrics must serialize to a JSON object")
    return dict(loaded)


def execution_runtime_revisions(
    cluster: ClusterCapabilities,
    plan: ExecutionPlan,
) -> dict[str, str]:
    active = {
        worker_id
        for worker_id, role in plan.worker_roles.items()
        if role not in {"idle", "storage_cache"}
    }
    revisions: dict[str, str] = {}
    for worker in cluster.workers:
        if worker.worker_id not in active:
            continue
        capability = worker.engine(plan.engine_id)
        if capability is not None and capability.runtime_revision:
            revisions[worker.worker_id] = capability.runtime_revision
    return dict(sorted(revisions.items()))


def _execution_layout(
    execution_plan: ExecutionPlan,
    deployed_plan: ExecutionPlan | ProductStagePlan,
) -> tuple[
    dict[str, str],
    dict[str, str],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    roles = dict(execution_plan.worker_roles)
    fast_paths = dict(execution_plan.fast_paths)
    stages: list[dict[str, Any]] = []
    experts: list[dict[str, Any]] = []
    microshards: list[dict[str, Any]] = []
    assignments = getattr(deployed_plan, "assignments", None)
    if assignments is not None:
        for item in assignments:
            roles[item.worker_id] = item.worker_role
            if item.fast_path_id:
                fast_paths[item.worker_id] = item.fast_path_id
            stages.append(
                {
                    "stage_id": item.stage_id,
                    "worker_id": item.worker_id,
                    "device": item.device,
                    "layer_start": item.assignment.layer_start,
                    "layer_end": item.assignment.layer_end,
                    "owns_embeddings": item.assignment.owns_embeddings,
                    "owns_final_norm": item.assignment.owns_final_norm,
                    "owns_output_projection": item.assignment.owns_output_projection,
                    "fast_path_id": item.fast_path_id,
                    "fast_path_mode": item.fast_path_mode,
                    "fast_path_profile_fingerprint": item.fast_path_profile_fingerprint,
                }
            )
        for stage in getattr(deployed_plan, "expert_plans", ()):
            for placement in stage.placements:
                row = {
                    "stage_id": stage.stage_id,
                    **placement.model_dump(
                        mode="json", exclude={"worker_endpoints", "microshards"}
                    ),
                }
                experts.append(row)
                role = (
                    "whole_expert" if placement.strategy == "whole-remote" else "expert_microshard"
                )
                if placement.strategy != "local":
                    for worker_id in placement.worker_ids:
                        roles.setdefault(worker_id, role)
                for shard in placement.microshards:
                    microshards.append({"stage_id": stage.stage_id, **dict(shard)})
        idle_workers = getattr(deployed_plan, "idle_workers", {})
    else:
        stages.extend(dict(item) for item in execution_plan.stage_assignments)
        idle_workers = execution_plan.idle_workers
    for worker_id in idle_workers:
        roles.setdefault(worker_id, "idle")
    return (
        dict(sorted(roles.items())),
        dict(sorted(fast_paths.items())),
        tuple(stages),
        tuple(experts),
        tuple(microshards),
    )


def build_inference_telemetry_record(
    *,
    request_id: str,
    model: ResolvedModelDescriptor,
    execution_plan: ExecutionPlan,
    deployed_plan: ExecutionPlan | ProductStagePlan,
    cluster: ClusterCapabilities,
    status: Literal["completed", "failed", "cancelled"],
    submitted_monotonic_s: float,
    completed_monotonic_s: float,
    token_monotonic_s: Sequence[float],
    terminal_metrics: Mapping[str, Any] | None = None,
    terminal_timing_metrics: Mapping[str, float] | None = None,
    per_token_expert_metrics: Sequence[Mapping[str, Any]] = (),
    recoveries: int = 0,
) -> InferenceTelemetryRecord:
    """Build one honest, engine-neutral record from observed terminal facts.

    Planner predictions are deliberately excluded. Metrics that were not
    observed or reported remain ``None`` instead of being represented as zero.
    Raw engine evidence is retained so new normalizers need not change the
    execution engines or discard historical measurements.
    """

    engine_metrics = _json_metrics(terminal_metrics)
    timing_metrics = _json_metrics(terminal_timing_metrics)
    if timing_metrics:
        engine_metrics = {**engine_metrics, "coordinator_timing": timing_metrics}
    roles, fast_paths, stages, experts, microshards = _execution_layout(
        execution_plan,
        deployed_plan,
    )
    runtime_revisions = execution_runtime_revisions(cluster, execution_plan)
    unique_revisions = sorted(set(runtime_revisions.values()))
    engine_revision = unique_revisions[0] if len(unique_revisions) == 1 else None

    observed_tokens = tuple(float(item) for item in token_monotonic_s)
    elapsed_s = max(0.0, completed_monotonic_s - submitted_monotonic_s)
    ttft_ms = (
        max(0.0, observed_tokens[0] - submitted_monotonic_s) * 1000 if observed_tokens else None
    )
    inter_token_ms = tuple(
        max(0.0, right - left) * 1000 for left, right in pairwise(observed_tokens)
    )
    decode_tokens_s = (
        (len(observed_tokens) - 1) / (observed_tokens[-1] - observed_tokens[0])
        if len(observed_tokens) > 1 and observed_tokens[-1] > observed_tokens[0]
        else None
    )
    aggregate_tokens_s = len(observed_tokens) / elapsed_s if elapsed_s > 0 else None
    sources: dict[str, str] = {}
    if ttft_ms is not None:
        sources["ttft_ms"] = "client-observed"
    if decode_tokens_s is not None:
        sources["decode_tokens_s"] = "client-observed"
    if aggregate_tokens_s is not None:
        sources["aggregate_tokens_s"] = "client-observed"
    if inter_token_ms:
        sources["inter_token_latency_ms"] = "client-observed"

    prefill_tokens_s = _first_float(
        engine_metrics,
        (
            ("prefill_tokens_s",),
            ("prompt_tokens_per_second",),
            ("timings", "prompt_per_second"),
        ),
    )
    if prefill_tokens_s is not None:
        sources["prefill_tokens_s"] = "engine-reported"
    reported_decode = _first_float(
        engine_metrics,
        (
            ("decode_tokens_per_second",),
            ("decode_tokens_s",),
            ("timings", "predicted_per_second"),
        ),
    )
    if decode_tokens_s is None and reported_decode is not None:
        decode_tokens_s = reported_decode
        sources["decode_tokens_s"] = "engine-reported"

    integer_metrics = {
        "network_bytes": (
            ("network_bytes",),
            ("bytes_transferred",),
        ),
        "cache_hits": (
            ("cache_hits",),
            ("expert_cache_hits",),
            ("telemetry_summary", "expert_cache_hits"),
        ),
        "cache_misses": (
            ("cache_misses",),
            ("expert_cache_misses",),
            ("telemetry_summary", "expert_cache_misses"),
        ),
        "bytes_loaded": (
            ("bytes_loaded",),
            ("storage_read_bytes",),
            ("telemetry_summary", "storage_read_bytes"),
        ),
        "bytes_evicted": (("bytes_evicted",),),
        "prefetch_useful_bytes": (("prefetch_useful_bytes",),),
        "prefetch_wasted_bytes": (("prefetch_wasted_bytes",),),
    }
    measured_ints = {
        name: _first_int(engine_metrics, paths) for name, paths in integer_metrics.items()
    }
    expert_network_bytes = sum(
        value
        for metrics in per_token_expert_metrics
        if (value := _nonnegative_int(metrics.get("bytes_transferred"))) is not None
    )
    if expert_network_bytes:
        measured_ints["network_bytes"] = (
            measured_ints["network_bytes"] or 0
        ) + expert_network_bytes
        sources["network_bytes"] = "coordinator-verified-expert-trace"
    for name, integer_value in measured_ints.items():
        if integer_value is not None and name not in sources:
            sources[name] = "engine-reported"

    float_metrics = {
        "serial_waits_per_token": (("serial_waits_per_token",),),
        "messages_per_token": (("messages_per_token",),),
        "payload_bytes_per_token": (("payload_bytes_per_token",),),
        "queue_time_ms": (("queue_time_ms",),),
        "compute_time_ms": (
            ("compute_time_ms",),
            ("cpu_compute_duration_ms",),
        ),
        "serialization_time_ms": (("serialization_time_ms",),),
        "cache_stall_time_ms": (("cache_stall_time_ms",),),
    }
    measured_floats = {
        name: _first_float(engine_metrics, paths) for name, paths in float_metrics.items()
    }
    for name, float_value in measured_floats.items():
        if float_value is not None:
            sources[name] = "engine-reported"

    fallbacks: list[str] = []
    reported_fallbacks = engine_metrics.get("fallbacks")
    if isinstance(reported_fallbacks, list):
        fallbacks.extend(str(item) for item in reported_fallbacks)
    expert_fallbacks = sum(
        value
        for metrics in per_token_expert_metrics
        if (value := _nonnegative_int(metrics.get("fallbacks"))) is not None
    )
    if expert_fallbacks:
        fallbacks.append(f"expert-local-fallback:{expert_fallbacks}")

    exactness_verified = any(
        engine_metrics.get(name) is True
        for name in ("exactness_verified", "exactness_passed", "correctness_passed")
    )
    if exactness_verified:
        sources["exactness_verified"] = "engine-reported"

    residency: list[DeviceResidencyRecord] = []
    raw_residency = engine_metrics.get("residency", [])
    if isinstance(raw_residency, list):
        for item in raw_residency:
            if isinstance(item, Mapping):
                residency.append(DeviceResidencyRecord.model_validate(item))
    movement: list[DataMovementRecord] = []
    raw_movement = engine_metrics.get("movement", [])
    if isinstance(raw_movement, list):
        for item in raw_movement:
            if isinstance(item, Mapping):
                movement.append(DataMovementRecord.model_validate(item))

    adapter = execution_plan.engine_parameters.get("adapter_id")
    if adapter is None:
        adapter = execution_plan.engine_parameters.get("model_family")
    return InferenceTelemetryRecord(
        request_id=request_id,
        model_id=model.model_id,
        model_revision=model.revision,
        model_fingerprint=model.content_fingerprint,
        model_format=model.format,
        quantization=model.quantization,
        tokenizer_identity=model.tokenizer_identity,
        engine_id=execution_plan.engine_id,
        engine_revision=engine_revision,
        engine_runtime_revisions=runtime_revisions,
        execution_identity=execution_plan.execution_identity,
        adapter_id=str(adapter) if adapter is not None else None,
        fast_paths=fast_paths,
        topology=(
            deployed_plan.topology_id
            if hasattr(deployed_plan, "topology_id")
            else execution_plan.topology
        ),
        workers=tuple(sorted(roles)),
        worker_roles=roles,
        stage_assignments=stages,
        expert_assignments=experts,
        microshard_assignments=microshards,
        residency=tuple(residency),
        movement=tuple(movement),
        generated_tokens=len(observed_tokens),
        ttft_ms=ttft_ms,
        prefill_tokens_s=prefill_tokens_s,
        decode_tokens_s=decode_tokens_s,
        aggregate_tokens_s=aggregate_tokens_s,
        inter_token_latency_ms=inter_token_ms,
        serial_waits_per_token=measured_floats["serial_waits_per_token"],
        messages_per_token=measured_floats["messages_per_token"],
        payload_bytes_per_token=measured_floats["payload_bytes_per_token"],
        network_bytes=measured_ints["network_bytes"],
        queue_time_ms=measured_floats["queue_time_ms"],
        compute_time_ms=measured_floats["compute_time_ms"],
        serialization_time_ms=measured_floats["serialization_time_ms"],
        cache_hits=measured_ints["cache_hits"],
        cache_misses=measured_ints["cache_misses"],
        bytes_loaded=measured_ints["bytes_loaded"],
        bytes_evicted=measured_ints["bytes_evicted"],
        prefetch_useful_bytes=measured_ints["prefetch_useful_bytes"],
        prefetch_wasted_bytes=measured_ints["prefetch_wasted_bytes"],
        cache_stall_time_ms=measured_floats["cache_stall_time_ms"],
        fallbacks=tuple(fallbacks),
        recoveries=recoveries,
        exactness_verified=exactness_verified,
        metric_sources=dict(sorted(sources.items())),
        engine_metrics=engine_metrics,
        status=status,
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

    def record_inference(self, record: InferenceTelemetryRecord) -> dict[str, Any]:
        """Persist a terminal record without importing any experiment recorder."""

        return self.emit("inference_recorded", **record.model_dump(mode="json"))


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
    "DataMovementRecord",
    "DeviceResidencyRecord",
    "InferenceTelemetryRecord",
    "JsonlLifecycleObserver",
    "LifecycleObserver",
    "ProductTelemetry",
    "TraceContext",
    "TraceWriter",
    "build_inference_telemetry_record",
    "configure_lifecycle_observer",
    "execution_runtime_revisions",
    "lifecycle_observer",
    "lifecycle_observer_from_environment",
    "merge_traces",
    "read_trace",
    "reconstruct_critical_path",
]
