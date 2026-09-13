"""E027-only Vast leases, cost accounting, and independent teardown watchdog."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from ..experiment_025.vast_lifecycle import Offer, VastClient, _json_output, _rows
from ..experiment_026.io import append_event, utc_now, write_once


ROOT = Path("artifacts/experiment-027/cost")
HARD_CEILING_USD = 10.0
SOFT_STOP_USD = 9.0
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
DISK_GB = 45
MODEL_BYTES = 18_973_870_432


def client() -> VastClient:
    return VastClient(executable="vastai")


def emit(event: str, **fields: Any) -> dict[str, Any]:
    row = {"timestamp": utc_now(), "event": event, **fields}
    append_event(ROOT / "ledger.jsonl", row)
    return row


def _events() -> list[dict[str, Any]]:
    path = ROOT / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _leases() -> dict[str, dict[str, Any]]:
    folder = ROOT / "leases"
    return {path.stem: json.loads(path.read_text()) for path in folder.glob("*.json")} if folder.exists() else {}


def _epoch(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def initialize() -> dict[str, Any]:
    ROOT.mkdir(parents=True, exist_ok=True)
    opening = ROOT / "opening.json"
    if not opening.exists():
        write_once(opening, {
            "timestamp": utc_now(),
            "budget": client().user_budget(),
            "hard_ceiling_usd": HARD_CEILING_USD,
            "soft_stop_usd": SOFT_STOP_USD,
        })
    return json.loads(opening.read_text())


def estimate(now: float | None = None) -> dict[str, float]:
    now = time.time() if now is None else now
    rows = _events()
    total = 0.0
    reserved = 0.0
    for label, lease in _leases().items():
        created = next((row for row in rows if row["event"] == "CREATE_ATTEMPT" and row.get("label") == label), None)
        if not created:
            continue
        succeeded = next((row for row in rows if row["event"] == "CREATE_SUCCEEDED"
                          and row.get("label") == label), None)
        last_present = next((row for row in reversed(rows) if row["event"] == "ACCOUNT_SNAPSHOT"
                             and any(item.get("label") == label for item in row["owned_instances"])), None)
        stopped = [row for row in rows if row["event"] == "ACCOUNT_SNAPSHOT" and succeeded and last_present
                   and row["timestamp"] > last_present["timestamp"]
                   and not any(item.get("label") == label for item in row["owned_instances"])]
        stop = _epoch(stopped[0]["timestamp"]) if stopped else now
        total += max(0.0, stop - _epoch(created["timestamp"])) / 3600 * lease["hourly_usd"]
        total += lease["network_reserve_usd"]
        if not stopped:
            reserved += max(0.0, lease["deadline_epoch"] - now) / 3600 * lease["hourly_usd"]
    return {"estimated_upper_cost_usd": total, "reserved_future_rental_usd": reserved}


def snapshot() -> dict[str, Any]:
    opening = initialize()
    api = client()
    budget = api.user_budget()
    instances = api.show_instances()
    costs = estimate()
    account_delta = max(0.0, opening["budget"]["conservative_available_usd"] - budget["conservative_available_usd"])
    owned_labels = set(_leases())
    row = {
        "budget": budget,
        **costs,
        "account_decrease_usd": account_delta,
        "conservative_spend_usd": max(account_delta, costs["estimated_upper_cost_usd"]),
        "owned_instances": [
            {key: item.get(key) for key in (
                "id", "label", "actual_status", "dph_total", "gpu_name", "machine_id",
                "geolocation", "public_ipaddr", "ports", "ssh_host", "ssh_port",
            )}
            for item in instances if item.get("label") in owned_labels
        ],
    }
    emit("ACCOUNT_SNAPSHOT", **row)
    return row


def _reconcile_absence(instances: list[dict[str, Any]]) -> None:
    present = {item.get("label") for item in instances}
    rows = _events()
    for label in _leases():
        attempted = any(row["event"] == "CREATE_SUCCEEDED" and row.get("label") == label for row in rows)
        confirmed = any(row["event"] == "ABSENCE_CONFIRMED" and row.get("label") == label for row in rows)
        if attempted and not confirmed and label not in present:
            emit("ABSENCE_CONFIRMED", label=label)


def destroy(label: str, instance_id: int, reason: str) -> None:
    if label not in _leases():
        raise ValueError("refusing to destroy an instance outside E027 leases")
    emit("DESTROY_ATTEMPT", label=label, instance_id=instance_id, reason=reason)
    result = client()._run(["destroy", "instance", str(instance_id), "--raw", "--yes"], timeout_seconds=90)
    emit("DESTROY_RESPONSE", label=label, instance_id=instance_id, returncode=result.returncode,
         stderr_tail=result.stderr[-1000:])
    if result.returncode:
        raise RuntimeError(f"Vast destruction failed for E027 instance {instance_id}")


def watch() -> None:
    emit("WATCHDOG_STARTED", pid=os.getpid(), hard_ceiling_usd=HARD_CEILING_USD,
         soft_stop_usd=SOFT_STOP_USD)
    failures = 0
    while True:
        append_event(ROOT / "heartbeat.jsonl", {"timestamp": utc_now(), "pid": os.getpid()})
        try:
            current = snapshot()
            failures = 0
            _reconcile_absence(current["owned_instances"])
            over = current["conservative_spend_usd"] >= SOFT_STOP_USD
            for item in current["owned_instances"]:
                lease = _leases().get(str(item.get("label")))
                if lease and (over or time.time() >= lease["deadline_epoch"]):
                    destroy(lease["label"], int(item["id"]), "BUDGET_GUARD" if over else "LEASE_EXPIRED")
        except Exception as error:  # fail closed if account visibility is lost
            failures += 1
            emit("WATCHDOG_ERROR", failures=failures, error=repr(error))
            if failures >= 2:
                for lease in _leases().values():
                    if lease.get("instance_id"):
                        try:
                            destroy(lease["label"], int(lease["instance_id"]), "ACCOUNT_VISIBILITY_LOST")
                        except Exception as destroy_error:
                            emit("WATCHDOG_DESTROY_ERROR", label=lease["label"], error=repr(destroy_error))
        time.sleep(20)


def watchdog_alive() -> bool:
    path = ROOT / "heartbeat.jsonl"
    if not path.exists():
        return False
    last = json.loads(path.read_text().splitlines()[-1])
    if time.time() - _epoch(last["timestamp"]) > 60:
        return False
    try:
        import psutil
        return "experiment_027.vast" in " ".join(psutil.Process(last["pid"]).cmdline())
    except Exception:
        return False


def start_watchdog() -> dict[str, Any]:
    initialize()
    if watchdog_alive():
        return {"status": "ALREADY_RUNNING"}
    log = (ROOT / "watchdog.log").open("ab")
    process = subprocess.Popen(
        [sys.executable, "-m", "swarm_inference.experiments.experiment_027.vast", "watch"],
        stdout=log,
        stderr=log,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    log.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not watchdog_alive():
        time.sleep(0.25)
    if not watchdog_alive():
        raise RuntimeError("E027 watchdog did not become healthy")
    return {"status": "RUNNING", "pid": process.pid}


def fresh_offer(offer_id: int) -> Offer:
    query = f"id={offer_id} gpu_ram>=16 num_gpus=1 rentable=True verified=True reliability>=0.98 direct_port_count>=1 cuda_vers>=13.0"
    result = client()._run(
        ["search", "offers", query, "--on-demand", "--limit", "1000", "--storage", str(DISK_GB), "--raw"],
        timeout_seconds=180,
    )
    if result.returncode:
        raise RuntimeError("Vast offer refresh failed")
    offer = next((Offer.from_raw(row) for row in _rows(_json_output(result.stdout))
                  if int(row.get("id", -1)) == offer_id), None)
    if offer is None:
        raise RuntimeError("selected Vast offer is no longer available")
    return offer


def create(offer_id: int, role: str, hours: float) -> dict[str, Any]:
    if not watchdog_alive():
        raise RuntimeError("E027 budget watchdog is not alive")
    if role not in {"b", "c"} or not (0 < hours <= 2.0):
        raise ValueError("E027 requires role b/c and a lease of at most two hours")
    active_labels = {item.get("label") for item in snapshot()["owned_instances"]}
    if any(lease["role"] == role and label in active_labels for label, lease in _leases().items()):
        raise RuntimeError(f"E027 role {role} already has a lease")
    offer = fresh_offer(offer_id)
    rate = offer.effective_rate_per_hour(DISK_GB)
    if not (offer.gpu_count == 1 and offer.verified and offer.rentable and offer.reliability >= 0.98
            and offer.gpu_ram_gib >= 16 and offer.cuda_max_version >= 13.0 and rate <= 1.5
            and max(offer.inet_down_cost_per_gb, offer.inet_up_cost_per_gb) <= 0.02):
        raise ValueError("selected offer is outside the E027 admission envelope")
    network_reserve = MODEL_BYTES / 1e9 * offer.inet_down_cost_per_gb + 0.25
    current = snapshot()
    projected = (current["conservative_spend_usd"] + current["reserved_future_rental_usd"]
                 + hours * rate + network_reserve + 0.5)
    if projected >= SOFT_STOP_USD:
        raise RuntimeError("projected E027 spend exceeds the soft stop")
    label = f"e027-{role}-{int(time.time())}"
    lease = {
        "label": label, "role": role, "offer_id": offer.offer_id,
        "machine_id": offer.machine_id, "gpu_name": offer.gpu_name,
        "region": offer.raw.get("geolocation"), "hourly_usd": rate,
        "network_reserve_usd": network_reserve, "deadline_epoch": time.time() + hours * 3600,
        "disk_gb": DISK_GB, "image": IMAGE,
    }
    lease_path = ROOT / "leases" / f"{label}.json"
    write_once(lease_path, lease)
    command = ["create", "instance", str(offer.offer_id), "--raw", "--image", IMAGE,
               "--disk", str(DISK_GB), "--ssh", "--direct", "--cancel-unavail", "--label", label]
    emit("CREATE_ATTEMPT", label=label, role=role, offer_id=offer.offer_id,
         machine_id=offer.machine_id, hourly_usd=rate)
    result = client()._run(command, timeout_seconds=150)
    payload = _json_output(result.stdout)
    if result.returncode or not isinstance(payload, dict) or not payload.get("success"):
        emit("CREATE_FAILED_OR_AMBIGUOUS", label=label, returncode=result.returncode,
             stderr_tail=result.stderr[-1000:])
        raise RuntimeError("Vast creation failed or was ambiguous; reconcile by E027 label")
    instance_id = int(payload["new_contract"])
    lease["instance_id"] = instance_id
    temporary = lease_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(lease, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, lease_path)
    emit("CREATE_SUCCEEDED", label=label, role=role, instance_id=instance_id)
    return {"label": label, "role": role, "instance_id": instance_id,
            "gpu_name": offer.gpu_name, "region": offer.raw.get("geolocation"),
            "hourly_usd": rate, "maximum_lease_hours": hours}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "status", "watch", "start-watchdog", "create", "destroy"))
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--role")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--label")
    parser.add_argument("--instance-id", type=int)
    args = parser.parse_args()
    if args.action == "init":
        print(json.dumps(initialize()))
    elif args.action == "status":
        print(json.dumps(snapshot()))
    elif args.action == "watch":
        watch()
    elif args.action == "start-watchdog":
        print(json.dumps(start_watchdog()))
    elif args.action == "create":
        print(json.dumps(create(args.offer_id, args.role, args.hours)))
    else:
        destroy(args.label, args.instance_id, "EXPERIMENT_COMPLETE")


if __name__ == "__main__":
    main()
