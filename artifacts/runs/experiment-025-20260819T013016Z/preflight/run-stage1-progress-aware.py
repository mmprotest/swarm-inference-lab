"""Run the existing Stage 1 canary with progress-aware bootstrap supervision.

This wrapper does not alter worker behavior or model execution. It extends the
independent hard stop to 30 minutes while aborting a created host after four
minutes without provider- or log-observed bootstrap progress.
"""

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

from swarm_inference.experiments.experiment_025 import provisioning, stages
from swarm_inference.experiments.experiment_025.io import utc_now

RUN_ROOT = Path("artifacts/runs/experiment-025-20260819T013016Z").resolve()
STAGE_ROOT = RUN_ROOT / "rental" / "stage-1-backbone-canary"
PROGRESS_PATH = STAGE_ROOT / "bootstrap-progress.jsonl"
HARD_TTL_SECONDS = 30 * 60
NO_PROGRESS_SECONDS = 4 * 60
LOG_QUERY_AFTER_SECONDS = 2 * 60
LOG_QUERY_INTERVAL_SECONDS = 60
PROVIDER_POLL_SECONDS = 30

_progress_write_lock = threading.Lock()


def emit(event: str, **fields: Any) -> None:
    payload = {
        "schema_version": "experiment-025-bootstrap-progress-v1",
        "timestamp_utc": utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "stage": "physical-stage-1",
        "event_type": event,
        **fields,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _progress_write_lock, PROGRESS_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def instance_id(row: dict[str, Any]) -> int:
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


def progress_aware_provision_worker(**kwargs: Any) -> provisioning.LiveWorker:
    created_id = provisioning.create_worker_instance(
        client=kwargs["client"],
        run_id=kwargs["run_id"],
        worker_id=kwargs["worker_id"],
        role=kwargs["role"],
        layer=kwargs["layer"],
        worker_index=kwargs["worker_index"],
        offer=kwargs["offer"],
        image_reference=kwargs["image_reference"],
        image_digest=kwargs["image_digest"],
        disk_gb=kwargs["disk_gb"],
        material=kwargs["material"],
        watchdog_receipt=kwargs["watchdog_receipt"],
        go_receipt=kwargs["go_receipt"],
        maximum_context=kwargs["maximum_context"],
        expert_endpoints=kwargs.get("expert_endpoints"),
    )
    abort = threading.Event()
    stop = threading.Event()
    observer_failure: list[str] = []

    def observe() -> None:
        last_signature: tuple[Any, ...] | None = None
        last_progress = time.monotonic()
        last_log_query = 0.0
        last_log_digest: str | None = None
        emit(
            "INSTANCE_CREATED",
            instance_id=created_id,
            machine_id=kwargs["offer"].machine_id,
            worker_id=kwargs["worker_id"],
            hard_ttl_seconds=HARD_TTL_SECONDS,
            no_progress_timeout_seconds=NO_PROGRESS_SECONDS,
        )
        while not stop.is_set() and not abort.is_set():
            try:
                rows = kwargs["client"].show_instances()
                row = next(
                    (value for value in rows if instance_id(value) == created_id),
                    None,
                )
                now = time.monotonic()
                if row is None:
                    emit("PROVIDER_INSTANCE_MISSING", instance_id=created_id)
                else:
                    signature = (
                        row.get("actual_status"),
                        row.get("status_msg"),
                        row.get("disk_usage"),
                        row.get("inet_down_billed"),
                        row.get("vmem_usage"),
                    )
                    if signature != last_signature:
                        last_signature = signature
                        last_progress = now
                        emit(
                            "BOOTSTRAP_PROGRESS",
                            instance_id=created_id,
                            machine_id=row.get("machine_id"),
                            bootstrap_stage=bootstrap_stage(row),
                            provider_status=row.get("actual_status"),
                            provider_status_message=row.get("status_msg"),
                            disk_usage_gb=row.get("disk_usage"),
                            provider_billed_download_counter=row.get(
                                "inet_down_billed"
                            ),
                            gpu_memory_usage=row.get("vmem_usage"),
                            gpu_utilization=row.get("gpu_util"),
                        )
                    if (
                        now - last_progress >= LOG_QUERY_AFTER_SECONDS
                        and now - last_log_query >= LOG_QUERY_INTERVAL_SECONDS
                    ):
                        last_log_query = now
                        log_text = kwargs["client"].logs(created_id, tail=160)
                        digest = hashlib.sha256(log_text.encode("utf-8")).hexdigest()
                        changed = last_log_digest is None or digest != last_log_digest
                        last_log_digest = digest
                        lowered = log_text.lower()
                        substantive = bool(log_text.strip()) and not any(
                            marker in lowered
                            for marker in (
                                "error response from daemon",
                                "no such container",
                            )
                        )
                        if substantive and changed:
                            last_progress = time.monotonic()
                        emit(
                            "BOOTSTRAP_LOG_OBSERVED",
                            instance_id=created_id,
                            machine_id=row.get("machine_id"),
                            bootstrap_stage=bootstrap_stage(row, log_text),
                            log_changed=changed,
                            log_substantive=substantive,
                            log_sha256=digest,
                            raw_log_retained=False,
                        )
                        fatal = next(
                            (
                                marker
                                for marker in (
                                    "native mxfp4 cuda tensor upload failed",
                                    "tensor allocation: out of memory",
                                    "no space left on device",
                                    "refusing to replace activated snapshot",
                                    "port is already allocated",
                                )
                                if marker in lowered
                            ),
                            None,
                        )
                        if fatal is not None:
                            observer_failure.append(f"fatal bootstrap marker: {fatal}")
                            emit(
                                "WORKER_UNHEALTHY",
                                instance_id=created_id,
                                machine_id=row.get("machine_id"),
                                bootstrap_stage=bootstrap_stage(row, log_text),
                                reason="fatal_bootstrap_marker",
                                marker=fatal,
                            )
                            abort.set()
                            break
                stalled_for = time.monotonic() - last_progress
                if stalled_for >= NO_PROGRESS_SECONDS:
                    observer_failure.append(
                        f"no bootstrap progress for {stalled_for:.1f} seconds"
                    )
                    emit(
                        "BOOTSTRAP_STALLED",
                        instance_id=created_id,
                        no_progress_seconds=stalled_for,
                    )
                    abort.set()
                    break
            except BaseException as exc:
                emit(
                    "OBSERVER_RETRY",
                    instance_id=created_id,
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                )
            stop.wait(PROVIDER_POLL_SECONDS)

    observer = threading.Thread(
        target=observe,
        name="e025-stage1-bootstrap-observer",
        daemon=True,
    )
    observer.start()
    try:
        worker = provisioning.wait_for_worker(
            client=kwargs["client"],
            ledger=kwargs["ledger"],
            run_id=kwargs["run_id"],
            worker_id=kwargs["worker_id"],
            role=kwargs["role"],
            layer=kwargs["layer"],
            worker_index=kwargs["worker_index"],
            offer=kwargs["offer"],
            instance_id=created_id,
            credential=Path(kwargs["material"]["credential_path"]).read_bytes(),
            certificate=Path(kwargs["material"]["certificate_path"]),
            image_digest=kwargs["image_digest"],
            deadline_epoch=kwargs["deadline_epoch"],
            container_port=42525,
            gpu_slot=0,
            abort_event=abort,
        )
        emit(
            "WORKER_READY",
            instance_id=created_id,
            machine_id=worker.machine_id,
            worker_id=worker.worker_id,
            gpu_model=worker.gpu_name,
        )
        return worker
    except BaseException as exc:
        if observer_failure:
            raise RuntimeError(observer_failure[-1]) from exc
        raise
    finally:
        stop.set()
        observer.join(timeout=5)


def main() -> int:
    if PROGRESS_PATH.is_file() and PROGRESS_PATH.stat().st_size:
        raise RuntimeError("refusing to replace Stage 1 bootstrap progress evidence")
    stages.CANARY_TTL_SECONDS = HARD_TTL_SECONDS
    stages.provision_worker = progress_aware_provision_worker
    sys.argv = [
        "experiment_025_canary.py",
        "--run-root",
        str(RUN_ROOT),
    ]
    runpy.run_path("scripts/experiment_025_canary.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
