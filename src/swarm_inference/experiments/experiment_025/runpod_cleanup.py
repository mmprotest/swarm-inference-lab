"""Append-only RunPod ledger, watchdog cleanup, and strict E025 attribution."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .io import append_jsonl, utc_now
from .providers.base import ProviderMutationPolicy, ProviderOperation

RUNPOD_WATCHDOG_SCHEMA = "experiment-025-runpod-independent-watchdog-v1"


class CleanupProvider(Protocol):
    def list_pods(self) -> Any: ...

    def delete_pod(self, pod_id: str) -> Any: ...


class AppendOnlyRunPodLedger:
    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = path.resolve()
        self.run_id = run_id

    def append(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.path,
            {
                "schema_version": "experiment-025-runpod-pod-ledger-v1",
                "timestamp": utc_now(),
                "provider": "runpod",
                "run_id": self.run_id,
                "event": event,
                **fields,
            },
        )

    def record_created(self, *, pod_id: str, pod_name: str, stage: str) -> None:
        if not pod_id or not safe_e025_name(pod_name, self.run_id):
            raise ValueError("refusing to ledger an unattributable RunPod Pod")
        self.append(
            "POD_CREATED",
            pod_id=pod_id,
            pod_name=pod_name,
            stage=stage,
            deletion_required=True,
        )


def safe_e025_name(name: str, run_id: str) -> bool:
    prefix = f"e025-rp-{run_id}-"
    if not name.startswith(prefix):
        return False
    suffix = name.removeprefix(prefix)
    return bool(suffix) and all(character.isalnum() or character == "-" for character in suffix)


def read_ledger(path: Path, *, run_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"RunPod ledger line {line_number} is not an object")
        if value.get("run_id") != run_id:
            raise ValueError(f"RunPod ledger line {line_number} has another run identity")
        rows.append(value)
    return rows


def ledgered_targets(path: Path, *, run_id: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for row in read_ledger(path, run_id=run_id):
        pod_id = str(row.get("pod_id", ""))
        pod_name = str(row.get("pod_name", ""))
        if row.get("event") == "POD_CREATED":
            if not pod_id or not safe_e025_name(pod_name, run_id):
                raise ValueError("RunPod ledger contains an unsafe creation target")
            targets[pod_id] = pod_name
    # Keep every created allocation attributable forever. RunPod deletion can be
    # asynchronous, so a DELETE response or ledger event is not proof that a Pod
    # is gone. Cleanup intersects this immutable attribution set with each fresh
    # provider list and retries until the Pod is actually absent.
    return targets


def _pod_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id", row.get("podId", row.get("pod_id", ""))))


def _pod_name(row: Mapping[str, Any]) -> str:
    return str(row.get("name", row.get("podName", row.get("pod_name", ""))))


def select_cleanup_targets(
    *,
    live_pods: Iterable[Mapping[str, Any]],
    ledger_path: Path,
    run_id: str,
) -> list[dict[str, str]]:
    ledgered = ledgered_targets(ledger_path, run_id=run_id)
    selected: list[dict[str, str]] = []
    for row in live_pods:
        pod_id = _pod_id(row)
        name = _pod_name(row)
        if pod_id in ledgered and ledgered[pod_id] == name and safe_e025_name(name, run_id):
            selected.append({"pod_id": pod_id, "pod_name": name})
    return sorted(selected, key=lambda row: row["pod_id"])


def cleanup_from_ledger(
    *,
    provider: CleanupProvider,
    ledger_path: Path,
    run_id: str,
    reason: str,
    attempts: int = 5,
    retry_seconds: float = 1.0,
) -> dict[str, Any]:
    ledger = AppendOnlyRunPodLedger(ledger_path, run_id=run_id)
    deleted: list[str] = []
    failures: list[dict[str, str]] = []
    for attempt in range(1, attempts + 1):
        live_value = provider.list_pods()
        live = live_value if isinstance(live_value, list) else live_value.get("pods", [])
        targets = select_cleanup_targets(
            live_pods=live,
            ledger_path=ledger_path,
            run_id=run_id,
        )
        if not targets:
            break
        failures.clear()
        for target in targets:
            try:
                provider.delete_pod(target["pod_id"])
                ledger.append(
                    "POD_DELETE_REQUESTED",
                    pod_id=target["pod_id"],
                    pod_name=target["pod_name"],
                    reason=reason,
                    permanent_delete=True,
                    attempt=attempt,
                )
                deleted.append(target["pod_id"])
            except Exception as exc:
                failures.append(
                    {
                        "pod_id": target["pod_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                )
        # Deletion is asynchronous on some provider paths. Give the control
        # plane time to converge even when the DELETE request itself succeeded.
        if attempt < attempts:
            time.sleep(retry_seconds)
    final_value = provider.list_pods()
    final_live = final_value if isinstance(final_value, list) else final_value.get("pods", [])
    remaining = select_cleanup_targets(
        live_pods=final_live,
        ledger_path=ledger_path,
        run_id=run_id,
    )
    remaining_ids = {row["pod_id"] for row in remaining}
    for pod_id in sorted(set(deleted) - remaining_ids):
        ledger.append(
            "POD_DELETION_CONFIRMED",
            pod_id=pod_id,
            reason=reason,
            confirmed_absent_from_provider_list=True,
        )
    return {
        "schema_version": "experiment-025-runpod-cleanup-receipt-v1",
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "reason": reason,
        "status": "PASS" if not remaining else "FAIL",
        "permanent_delete": True,
        "deleted_pod_ids": sorted(set(deleted)),
        "remaining_attributed_pods": remaining,
        "zero_live_attributed_pods": not remaining,
        "failures": failures,
        "unrelated_pods_touched": 0,
    }


def watchdog_receipt(
    *,
    run_id: str,
    stage: str,
    deadline_epoch: float,
    startup_nonce: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUNPOD_WATCHDOG_SCHEMA,
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "stage": stage,
        "deadline_epoch": deadline_epoch,
        "status": status,
        "watchdog_pid": os.getpid(),
        "independent_process": True,
        "startup_nonce_sha256": hashlib.sha256(startup_nonce.encode()).hexdigest(),
    }


def start_runpod_watchdog(
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
    allow_paid_run: bool,
    startup_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    policy = ProviderMutationPolicy.from_intent(allow_paid_run=allow_paid_run)
    policy.require(ProviderOperation.DELETE_POD)
    stale = [
        path
        for path in (receipt_path, trigger_path, stop_path, cleanup_receipt_path)
        if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "RunPod watchdog refuses stale control files: " + ", ".join(str(path) for path in stale)
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    startup_nonce = secrets.token_hex(32)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[4] / "scripts" / "experiment_025_runpod_watchdog.py"),
        "--run-id",
        run_id,
        "--stage",
        stage,
        "--ledger",
        str(ledger_path),
        "--deadline-epoch",
        str(deadline_epoch),
        "--receipt",
        str(receipt_path),
        "--log",
        str(log_path),
        "--trigger",
        str(trigger_path),
        "--stop",
        str(stop_path),
        "--cleanup-receipt",
        str(cleanup_receipt_path),
        "--startup-nonce",
        startup_nonce,
        "--allow-paid-run",
    ]
    creation_flags = 0
    if os.name == "nt":
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creation_flags,
    )
    expected_hash = hashlib.sha256(startup_nonce.encode()).hexdigest()
    deadline = time.monotonic() + startup_timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("independent RunPod watchdog exited during startup")
        if receipt_path.is_file():
            value = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                value.get("status") == "RUNNING"
                and value.get("startup_nonce_sha256") == expected_hash
                and int(value.get("watchdog_pid", -1)) == process.pid
            ):
                return value
        time.sleep(0.1)
    process.terminate()
    raise TimeoutError("independent RunPod watchdog did not publish startup receipt")


__all__ = [
    "RUNPOD_WATCHDOG_SCHEMA",
    "AppendOnlyRunPodLedger",
    "cleanup_from_ledger",
    "ledgered_targets",
    "read_ledger",
    "safe_e025_name",
    "select_cleanup_targets",
    "start_runpod_watchdog",
    "watchdog_receipt",
]
