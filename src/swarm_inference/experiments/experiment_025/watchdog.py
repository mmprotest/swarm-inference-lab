"""Independent hard-TTL watchdog for one paid E025 rental stage."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .io import append_jsonl, atomic_write_json, read_json, utc_now
from .vast_lifecycle import destroy_all_from_ledger

SCHEMA_VERSION = "experiment-025-independent-watchdog-v1"
WATCHDOG_STARTUP_TIMEOUT_SECONDS = 60.0
WATCHDOG_RECEIPT_WRITE_ATTEMPTS = 20
WATCHDOG_RECEIPT_WRITE_RETRY_SECONDS = 0.1


def _receipt(
    *,
    run_id: str,
    stage: str,
    deadline_epoch: float,
    status: str,
    startup_nonce: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_id": run_id,
        "stage": stage,
        "watchdog_pid": os.getpid(),
        "deadline_epoch": deadline_epoch,
        "timestamp": utc_now(),
        "independent_process": True,
        "startup_nonce_sha256": hashlib.sha256(
            startup_nonce.encode("utf-8")
        ).hexdigest(),
    }


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    """Commit a receipt despite brief Windows reader/OneDrive locks."""

    for attempt in range(WATCHDOG_RECEIPT_WRITE_ATTEMPTS):
        try:
            atomic_write_json(path, value)
            return
        except PermissionError:
            if attempt + 1 == WATCHDOG_RECEIPT_WRITE_ATTEMPTS:
                raise
            time.sleep(WATCHDOG_RECEIPT_WRITE_RETRY_SECONDS)


def run_watchdog(
    *,
    run_id: str,
    stage: str,
    ledger_path: Path,
    deadline_epoch: float,
    receipt_path: Path,
    log_path: Path,
    trigger_path: Path,
    stop_path: Path,
    cleanup_receipt_path: Path,
    vast_executable: str,
    startup_nonce: str,
    poll_seconds: float = 5.0,
) -> int:
    _write_receipt(
        receipt_path,
        _receipt(
            run_id=run_id,
            stage=stage,
            deadline_epoch=deadline_epoch,
            status="RUNNING",
            startup_nonce=startup_nonce,
        ),
    )
    append_jsonl(
        log_path,
        {
            "event": "WATCHDOG_STARTED",
            **_receipt(
                run_id=run_id,
                stage=stage,
                deadline_epoch=deadline_epoch,
                status="RUNNING",
                startup_nonce=startup_nonce,
            ),
        },
    )
    reason: str | None = None
    while reason is None:
        now = time.time()
        if trigger_path.exists():
            reason = "MANUAL_TRIGGER"
        elif now >= deadline_epoch:
            reason = "TTL_EXCEEDED"
        elif stop_path.exists():
            cleanup = (
                read_json(cleanup_receipt_path)
                if cleanup_receipt_path.is_file()
                else {}
            )
            if cleanup.get("zero_live_e025_instances") is True:
                final = _receipt(
                    run_id=run_id,
                    stage=stage,
                    deadline_epoch=deadline_epoch,
                    status="STOPPED_AFTER_ZERO_LIVE_VERIFICATION",
                    startup_nonce=startup_nonce,
                )
                _write_receipt(receipt_path, final)
                append_jsonl(log_path, {"event": "WATCHDOG_STOPPED", **final})
                return 0
            append_jsonl(
                log_path,
                {
                    "event": "STOP_REJECTED_WITHOUT_ZERO_LIVE_PROOF",
                    "timestamp": utc_now(),
                    "run_id": run_id,
                    "stage": stage,
                },
            )
        if reason is None:
            # The launcher reads the startup receipt immediately.  Delay the
            # first replacement so Windows does not race that reader, then
            # retain the same bounded heartbeat cadence for the stage.
            time.sleep(max(1.0, min(poll_seconds, 30.0)))
            _write_receipt(
                receipt_path,
                _receipt(
                    run_id=run_id,
                    stage=stage,
                    deadline_epoch=deadline_epoch,
                    status="RUNNING",
                    startup_nonce=startup_nonce,
                ),
            )
    append_jsonl(
        log_path,
        {
            "event": "CLEANUP_TRIGGERED",
            "timestamp": utc_now(),
            "run_id": run_id,
            "stage": stage,
            "reason": reason,
        },
    )
    cleanup = destroy_all_from_ledger(
        ledger_path=ledger_path,
        run_id=run_id,
        reason=f"watchdog:{stage}:{reason}",
        executable=vast_executable,
        attempts=5,
    )
    atomic_write_json(cleanup_receipt_path, cleanup)
    final_status = (
        "TIMEOUT_CLEANUP_COMPLETE"
        if cleanup["zero_live_e025_instances"]
        else "TIMEOUT_CLEANUP_FAILED"
    )
    final = _receipt(
        run_id=run_id,
        stage=stage,
        deadline_epoch=deadline_epoch,
        status=final_status,
        startup_nonce=startup_nonce,
    )
    final["trigger_reason"] = reason
    final["cleanup"] = cleanup
    _write_receipt(receipt_path, final)
    append_jsonl(log_path, {"event": "WATCHDOG_FINISHED", **final})
    return 0 if cleanup["zero_live_e025_instances"] else 2


def start_watchdog(
    *,
    run_id: str,
    stage: str,
    ledger_path: Path,
    ttl_seconds: int,
    receipt_path: Path,
    log_path: Path,
    trigger_path: Path,
    stop_path: Path,
    cleanup_receipt_path: Path,
    vast_executable: str = "vastai",
    startup_timeout_seconds: float = WATCHDOG_STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if startup_timeout_seconds <= 0:
        raise ValueError("watchdog startup timeout must be positive")
    stale_control_paths = [
        path
        for path in (
            receipt_path,
            trigger_path,
            stop_path,
            cleanup_receipt_path,
        )
        if path.exists()
    ]
    if stale_control_paths:
        raise FileExistsError(
            "E025 watchdog refuses stale control files: "
            + ", ".join(str(path) for path in stale_control_paths)
        )
    startup_nonce = secrets.token_hex(32)
    startup_nonce_sha256 = hashlib.sha256(startup_nonce.encode("utf-8")).hexdigest()
    deadline = time.time() + ttl_seconds
    command = [
        sys.executable,
        "-m",
        "swarm_inference.experiments.experiment_025.watchdog",
        "run",
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--ledger",
        str(ledger_path.resolve()),
        "--deadline-epoch",
        str(deadline),
        "--receipt",
        str(receipt_path.resolve()),
        "--log",
        str(log_path.resolve()),
        "--trigger",
        str(trigger_path.resolve()),
        "--stop",
        str(stop_path.resolve()),
        "--cleanup-receipt",
        str(cleanup_receipt_path.resolve()),
        "--vast-executable",
        vast_executable,
        "--startup-nonce",
        startup_nonce,
    ]
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **kwargs)
    wait_deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < wait_deadline:
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if (
                receipt.get("status") == "RUNNING"
                and receipt.get("run_id") == run_id
                and receipt.get("stage") == stage
                and receipt.get("startup_nonce_sha256") == startup_nonce_sha256
            ):
                receipt["launcher_pid"] = process.pid
                receipt["watchdog_pid_matches_launcher"] = (
                    int(receipt.get("watchdog_pid", -1)) == process.pid
                )
                receipt["startup_identity"] = "per-launch SHA-256 nonce"
                return receipt
        if process.poll() is not None:
            raise RuntimeError("E025 independent watchdog exited during startup")
        time.sleep(0.1)
    atomic_write_json(
        trigger_path,
        {
            "timestamp": utc_now(),
            "reason": "WATCHDOG_STARTUP_RECEIPT_TIMEOUT",
            "watchdog_pid": process.pid,
            "startup_nonce_sha256": startup_nonce_sha256,
            "cleanup_required_if_process_starts_late": True,
        },
    )
    raise TimeoutError("E025 independent watchdog did not produce its RUNNING receipt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--deadline-epoch", type=float, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--trigger", type=Path, required=True)
    run.add_argument("--stop", type=Path, required=True)
    run.add_argument("--cleanup-receipt", type=Path, required=True)
    run.add_argument("--vast-executable", default="vastai")
    run.add_argument("--startup-nonce", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-id", required=True)
    start.add_argument("--stage", required=True)
    start.add_argument("--ledger", type=Path, required=True)
    start.add_argument("--ttl-seconds", type=int, required=True)
    start.add_argument("--receipt", type=Path, required=True)
    start.add_argument("--log", type=Path, required=True)
    start.add_argument("--trigger", type=Path, required=True)
    start.add_argument("--stop", type=Path, required=True)
    start.add_argument("--cleanup-receipt", type=Path, required=True)
    start.add_argument("--vast-executable", default="vastai")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        return run_watchdog(
            run_id=arguments.run_id,
            stage=arguments.stage,
            ledger_path=arguments.ledger,
            deadline_epoch=arguments.deadline_epoch,
            receipt_path=arguments.receipt,
            log_path=arguments.log,
            trigger_path=arguments.trigger,
            stop_path=arguments.stop,
            cleanup_receipt_path=arguments.cleanup_receipt,
            vast_executable=arguments.vast_executable,
            startup_nonce=arguments.startup_nonce,
        )
    receipt = start_watchdog(
        run_id=arguments.run_id,
        stage=arguments.stage,
        ledger_path=arguments.ledger,
        ttl_seconds=arguments.ttl_seconds,
        receipt_path=arguments.receipt,
        log_path=arguments.log,
        trigger_path=arguments.trigger,
        stop_path=arguments.stop,
        cleanup_receipt_path=arguments.cleanup_receipt,
        vast_executable=arguments.vast_executable,
    )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "run_watchdog", "start_watchdog"]
