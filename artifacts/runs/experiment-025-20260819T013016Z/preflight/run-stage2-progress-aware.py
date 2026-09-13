"""Run the existing five-machine Stage 2 canary with progress-aware supervision."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025 import stages
from swarm_inference.experiments.experiment_025.io import utc_now

RUN_ROOT = Path("artifacts/runs/experiment-025-20260819T013016Z").resolve()
STAGE_ROOT = RUN_ROOT / "rental" / "stage-2-sub-layer-canary"
PROGRESS_PATH = STAGE_ROOT / "bootstrap-progress.jsonl"
HARD_TTL_SECONDS = 40 * 60
NO_PROGRESS_SECONDS = 4 * 60
LOG_QUERY_AFTER_SECONDS = 2 * 60
LOG_QUERY_INTERVAL_SECONDS = 60
PROVIDER_POLL_SECONDS = 30
BOOTSTRAP_STAGE_ORDER = (
    "CONTAINER_BOOTSTRAP",
    "CONTAINER_RUNNING",
    "MODEL_DOWNLOAD",
    "PACKAGE_READY",
    "GPU_LOAD",
    "WORKER_LISTENING",
    "WORKER_READY",
)

_write_lock = threading.Lock()
_state_lock = threading.Lock()
_states: dict[int, dict[str, Any]] = {}
_observer_stop = threading.Event()

_original_create = stages.create_worker_instance
_original_failed_machine_exclusions = stages._failed_machine_exclusions
_original_wait_endpoint = stages.wait_for_public_endpoint
_original_wait_worker = stages.wait_for_worker

CREATE_ID_LOST_ATTEMPTS = tuple(
    sorted(
        (RUN_ROOT / "rental").glob(
            "stage-2-sub-layer-canary-attempt-*-create-id-lost-machine*"
        )
    )
)
PROGRESS_STALL_ATTEMPTS = tuple(
    sorted(
        (RUN_ROOT / "rental").glob(
            "stage-2-sub-layer-canary-attempt-*-*-stall-machine*"
        )
    )
)
SOFTWARE_BOOTSTRAP_ATTEMPT = (
    RUN_ROOT
    / "rental"
    / "stage-2-sub-layer-canary-attempt-004-parent-snapshot-reactivation-machine10134"
)


def emit(event: str, **fields: Any) -> None:
    payload = {
        "schema_version": "experiment-025-bootstrap-progress-v1",
        "timestamp_utc": utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "stage": "physical-stage-2",
        "event_type": event,
        **fields,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock, PROGRESS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def row_instance_id(row: dict[str, Any]) -> int:
    return int(row.get("id", row.get("instance_id", -1)))


def bootstrap_stage(row: dict[str, Any], log_text: str = "") -> str:
    lowered = log_text.lower()
    if "worker_ready" in lowered or "status\": \"ready" in lowered:
        return "WORKER_READY"
    if "listening" in lowered and "42525" in lowered:
        return "WORKER_LISTENING"
    if "[k3-persistent:load]" in lowered or "[cuda] device" in lowered:
        return "GPU_LOAD"
    if "activated snapshot" in lowered or "activation complete" in lowered:
        return "PACKAGE_READY"
    if (
        "download" in lowered
        or "materialize_worker_package" in lowered
        or row.get("inet_down_billed") not in (None, 0, 0.0, "0")
    ):
        return "MODEL_DOWNLOAD"
    if str(row.get("actual_status", "")).lower() in {"running", "loading"}:
        return "CONTAINER_RUNNING"
    return "CONTAINER_BOOTSTRAP"


def furthest_bootstrap_stage(*candidates: str | None) -> str:
    known = [candidate for candidate in candidates if candidate in BOOTSTRAP_STAGE_ORDER]
    if not known:
        return "CONTAINER_BOOTSTRAP"
    return max(known, key=BOOTSTRAP_STAGE_ORDER.index)


class CombinedAbort:
    def __init__(self, *events: Any) -> None:
        self.events = tuple(event for event in events if event is not None)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self.events)

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
        return True


def observed_create_worker_instance(**kwargs: Any) -> int:
    created_id = _original_create(**kwargs)
    now = time.monotonic()
    state = {
        "abort": threading.Event(),
        "worker_id": kwargs["worker_id"],
        "role": kwargs["role"],
        "machine_id": kwargs["offer"].machine_id,
        "last_signature": None,
        "last_progress": now,
        "last_log_query": 0.0,
        "last_log_digest": None,
        "bootstrap_stage": "CONTAINER_BOOTSTRAP",
        "ready": False,
    }
    with _state_lock:
        _states[created_id] = state
    emit(
        "INSTANCE_CREATED",
        instance_id=created_id,
        machine_id=state["machine_id"],
        worker_id=state["worker_id"],
        role=state["role"],
        hard_ttl_seconds=HARD_TTL_SECONDS,
        no_progress_timeout_seconds=NO_PROGRESS_SECONDS,
    )
    return created_id


def observed_wait_for_public_endpoint(**kwargs: Any) -> tuple[str, int]:
    with _state_lock:
        state = _states[kwargs["instance_id"]]
    kwargs["abort_event"] = CombinedAbort(
        kwargs.get("abort_event"), state["abort"]
    )
    return _original_wait_endpoint(**kwargs)


def observed_wait_for_worker(**kwargs: Any) -> Any:
    with _state_lock:
        state = _states[kwargs["instance_id"]]
    kwargs["abort_event"] = CombinedAbort(
        kwargs.get("abort_event"), state["abort"]
    )
    try:
        worker = _original_wait_worker(**kwargs)
    except BaseException as exc:
        reason = state.get("failure_reason")
        if reason:
            raise RuntimeError(str(reason)) from exc
        raise
    with _state_lock:
        state["ready"] = True
        state["last_progress"] = time.monotonic()
        state["bootstrap_stage"] = "WORKER_READY"
    emit(
        "WORKER_READY",
        instance_id=kwargs["instance_id"],
        machine_id=worker.machine_id,
        worker_id=worker.worker_id,
        role=worker.role,
        gpu_model=worker.gpu_name,
    )
    return worker


def progress_aware_failed_machine_exclusions(
    stage_root: Path, *, stage_prefix: str
) -> dict[str, Any]:
    """Exclude machines only for physical/provider failures attributable to them."""

    payload = _original_failed_machine_exclusions(
        stage_root, stage_prefix=stage_prefix
    )
    if stage_prefix != "stage-2-sub-layer-canary":
        return payload
    reclassified_attempts = {
        path.name
        for path in (
            *CREATE_ID_LOST_ATTEMPTS,
            *PROGRESS_STALL_ATTEMPTS,
            SOFTWARE_BOOTSTRAP_ATTEMPT,
        )
        if path.is_dir()
    }
    if not reclassified_attempts:
        return payload
    retained_sources: list[dict[str, Any]] = []
    for source in payload.get("sources", []):
        observations = [
            observation
            for observation in source.get("observations", [])
            if observation.get("stage_directory") not in reclassified_attempts
        ]
        if observations:
            retained_sources.append(
                {"machine_id": int(source["machine_id"]), "observations": observations}
            )
    for attempt in CREATE_ID_LOST_ATTEMPTS:
        ledger_entries = [
            json.loads(line)
            for line in (attempt / "instance-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for entry in ledger_entries:
            if (
                entry.get("event") != "CREATE_ID_LOST"
                or entry.get("machine_id") is None
            ):
                continue
            machine_id = int(entry["machine_id"])
            source = next(
                (
                    row
                    for row in retained_sources
                    if int(row["machine_id"]) == machine_id
                ),
                None,
            )
            observation = {
                "stage_directory": attempt.name,
                "instance_id": None,
                "offer_id": entry.get("offer_id"),
                "reason": (
                    "provider create returned no recoverable instance ID; use a "
                    "ranked alternate for the immediate retry"
                ),
                "physical_failure_receipt": None,
            }
            if source is None:
                retained_sources.append(
                    {"machine_id": machine_id, "observations": [observation]}
                )
            else:
                source["observations"].append(observation)
    for attempt in PROGRESS_STALL_ATTEMPTS:
        progress_entries = [
            json.loads(line)
            for line in (attempt / "bootstrap-progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        ledger_entries = [
            json.loads(line)
            for line in (attempt / "instance-ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for entry in progress_entries:
            if (
                entry.get("event_type") != "BOOTSTRAP_STALLED"
                or entry.get("machine_id") is None
            ):
                continue
            machine_id = int(entry["machine_id"])
            instance_id = entry.get("instance_id")
            observed_stage = furthest_bootstrap_stage(
                *(
                    row.get("bootstrap_stage")
                    for row in progress_entries
                    if row.get("instance_id") == instance_id
                )
            )
            ledger_entry = next(
                (
                    row
                    for row in ledger_entries
                    if row.get("event") == "CREATE_CONFIRMED"
                    and row.get("instance_id") == instance_id
                ),
                {},
            )
            source = next(
                (
                    row
                    for row in retained_sources
                    if int(row["machine_id"]) == machine_id
                ),
                None,
            )
            observation = {
                "stage_directory": attempt.name,
                "instance_id": instance_id,
                "offer_id": ledger_entry.get("offer_id"),
                "reason": (
                    f"{observed_stage} made no observable "
                    f"progress for {entry.get('no_progress_seconds')} seconds; use a "
                    "ranked alternate for the immediate retry"
                ),
                "physical_failure_receipt": "bootstrap-progress.jsonl",
            }
            if source is None:
                retained_sources.append(
                    {"machine_id": machine_id, "observations": [observation]}
                )
            else:
                source["observations"].append(observation)
    retained_sources.sort(key=lambda row: int(row["machine_id"]))
    payload["machine_ids"] = [int(row["machine_id"]) for row in retained_sources]
    payload["sources"] = retained_sources
    payload["progress_aware_reclassification"] = {
        "stage_directories": sorted(reclassified_attempts),
        "healthy_progressing_fragment_machines_blacklisted": False,
        "create_id_lost_machines_temporarily_excluded": True,
        "bootstrap_stalled_machines_temporarily_excluded": True,
        "software_bootstrap_failure_machines_blacklisted": False,
        "basis": (
            "For each CREATE_ID_LOST attempt, successfully created siblings were "
            "destroyed only because a separate provider create lost its ID. For each "
            "progress-aware stall, only the machine with a BOOTSTRAP_STALLED event is "
            "excluded; progressing siblings are retained as eligible. Attempt 004 was "
            "aborted for a reproducible snapshot-restart software defect, not a "
            "physical host failure."
        ),
    }
    return payload


def observe(client: Any) -> None:
    fatal_markers = (
        "native mxfp4 cuda tensor upload failed",
        "tensor allocation: out of memory",
        "no space left on device",
        "refusing to replace activated snapshot",
        "port is already allocated",
    )
    while not _observer_stop.wait(PROVIDER_POLL_SECONDS):
        try:
            rows = {row_instance_id(row): row for row in client.show_instances()}
        except BaseException as exc:
            emit(
                "OBSERVER_RETRY",
                operation="provider_progress_query",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
            continue
        with _state_lock:
            snapshots = list(_states.items())
        for created_id, state in snapshots:
            if state["ready"] or state["abort"].is_set():
                continue
            row = rows.get(created_id)
            if row is None:
                emit(
                    "PROVIDER_INSTANCE_MISSING",
                    instance_id=created_id,
                    worker_id=state["worker_id"],
                )
                continue
            now = time.monotonic()
            signature = (
                row.get("actual_status"),
                row.get("status_msg"),
                row.get("disk_usage"),
                row.get("inet_down_billed"),
                row.get("vmem_usage"),
            )
            row_stage = bootstrap_stage(row)
            with _state_lock:
                if signature != state["last_signature"]:
                    state["last_signature"] = signature
                    state["last_progress"] = now
                    state["bootstrap_stage"] = furthest_bootstrap_stage(
                        state["bootstrap_stage"], row_stage
                    )
                    changed = True
                else:
                    changed = False
                should_query_log = (
                    now - state["last_progress"] >= LOG_QUERY_AFTER_SECONDS
                    and now - state["last_log_query"] >= LOG_QUERY_INTERVAL_SECONDS
                )
                if should_query_log:
                    state["last_log_query"] = now
            if changed:
                emit(
                    "BOOTSTRAP_PROGRESS",
                    instance_id=created_id,
                    machine_id=row.get("machine_id"),
                    worker_id=state["worker_id"],
                    role=state["role"],
                    bootstrap_stage=row_stage,
                    provider_status=row.get("actual_status"),
                    provider_status_message=row.get("status_msg"),
                    disk_usage_gb=row.get("disk_usage"),
                    provider_billed_download_counter=row.get("inet_down_billed"),
                    gpu_memory_usage=row.get("vmem_usage"),
                    gpu_utilization=row.get("gpu_util"),
                )
            if should_query_log:
                try:
                    log_text = client.logs(created_id, tail=160)
                except BaseException as exc:
                    emit(
                        "OBSERVER_RETRY",
                        operation="bootstrap_progress_log_query",
                        instance_id=created_id,
                        error_type=type(exc).__name__,
                        error=str(exc)[:500],
                    )
                    log_text = ""
                lowered = log_text.lower()
                substantive = bool(log_text.strip()) and not any(
                    marker in lowered
                    for marker in (
                        "error response from daemon",
                        "no such container",
                    )
                )
                if substantive:
                    digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
                    log_stage = bootstrap_stage(row, log_text)
                    with _state_lock:
                        log_changed = (
                            state["last_log_digest"] is None
                            or digest != state["last_log_digest"]
                        )
                        state["last_log_digest"] = digest
                        if log_changed:
                            state["last_progress"] = time.monotonic()
                            state["bootstrap_stage"] = furthest_bootstrap_stage(
                                state["bootstrap_stage"], log_stage
                            )
                    emit(
                        "BOOTSTRAP_LOG_OBSERVED",
                        instance_id=created_id,
                        machine_id=row.get("machine_id"),
                        worker_id=state["worker_id"],
                        role=state["role"],
                        bootstrap_stage=log_stage,
                        log_changed=log_changed,
                        log_sha256=digest,
                        raw_log_retained=False,
                    )
                    fatal = next(
                        (marker for marker in fatal_markers if marker in lowered),
                        None,
                    )
                    if fatal is not None:
                        with _state_lock:
                            state["failure_reason"] = (
                                f"fatal bootstrap marker: {fatal}"
                            )
                            state["abort"].set()
                        emit(
                            "WORKER_UNHEALTHY",
                            instance_id=created_id,
                            machine_id=row.get("machine_id"),
                            worker_id=state["worker_id"],
                            bootstrap_stage=log_stage,
                            reason="fatal_bootstrap_marker",
                            marker=fatal,
                        )
                        continue
            with _state_lock:
                stalled_for = time.monotonic() - state["last_progress"]
                stalled_stage = state["bootstrap_stage"]
                if stalled_for >= NO_PROGRESS_SECONDS:
                    state["failure_reason"] = (
                        f"no bootstrap progress for {stalled_for:.1f} seconds"
                    )
                    state["abort"].set()
                    stalled = True
                else:
                    stalled = False
            if stalled:
                emit(
                    "BOOTSTRAP_STALLED",
                    instance_id=created_id,
                    machine_id=row.get("machine_id"),
                    worker_id=state["worker_id"],
                    bootstrap_stage=stalled_stage,
                    no_progress_seconds=stalled_for,
                )


def install_observer_after_client_construction() -> None:
    original_init = stages.VastClient.__init__

    def observed_init(client: Any, *args: Any, **kwargs: Any) -> None:
        original_init(client, *args, **kwargs)
        if not getattr(client, "_e025_progress_observer_started", False):
            client._e025_progress_observer_started = True
            thread = threading.Thread(
                target=observe,
                args=(client,),
                name="e025-stage2-bootstrap-observer",
                daemon=True,
            )
            thread.start()

    stages.VastClient.__init__ = observed_init


def main() -> int:
    if PROGRESS_PATH.is_file() and PROGRESS_PATH.stat().st_size:
        raise RuntimeError("refusing to replace Stage 2 bootstrap progress evidence")
    stages.SUB_LAYER_CANARY_TTL_SECONDS = HARD_TTL_SECONDS
    stages.create_worker_instance = observed_create_worker_instance
    stages._failed_machine_exclusions = progress_aware_failed_machine_exclusions
    stages.wait_for_public_endpoint = observed_wait_for_public_endpoint
    stages.wait_for_worker = observed_wait_for_worker
    install_observer_after_client_construction()
    sys.argv = [
        "experiment_025_sub_layer_canary.py",
        "--run-root",
        str(RUN_ROOT),
    ]
    try:
        runpy.run_path("scripts/experiment_025_sub_layer_canary.py", run_name="__main__")
    finally:
        _observer_stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
