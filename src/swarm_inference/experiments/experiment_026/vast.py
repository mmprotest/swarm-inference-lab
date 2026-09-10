"""E026-only admission control, append-only cost ledger, and lease watchdog.

No mutation is permitted outside a recorded E026 lease. Provider account spend
is treated conservatively as E026 spend when attribution is uncertain.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..experiment_025.vast_lifecycle import Offer, VastClient, _json_output
from .io import append_event, utc_now, write_once

ROOT = Path("artifacts/experiment-026/cost")
CEILING = 38.0
SOFT_STOP = 34.0
RESERVE = 7.0
IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04"


def client():
    return VastClient(executable="vastai")


def events():
    path = ROOT / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def emit(event, **fields):
    row = {"timestamp": utc_now(), "event": event, **fields}
    append_event(ROOT / "ledger.jsonl", row)
    return row


def leases():
    return {p.stem: json.loads(p.read_text()) for p in (ROOT / "leases").glob("*.json")}


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def effective_deadline(label, lease, rows=None):
    rows=events() if rows is None else rows
    extensions=[r["new_deadline_epoch"] for r in rows
                if r["event"]=="LEASE_EXTENDED" and r.get("label")==label]
    return max([lease["deadline_epoch"],*extensions])


def estimate(now=None):
    now = time.time() if now is None else now
    rows = events()
    total, worst_remaining = 0.0, 0.0
    for label, lease in leases().items():
        starts = [r for r in rows if r["event"] == "CREATE_ATTEMPT" and r.get("label") == label]
        if not starts:
            continue
        start = epoch(starts[0]["timestamp"])
        stops = [r for r in rows if r["event"] == "ABSENCE_CONFIRMED" and r.get("label") == label]
        stop = epoch(stops[-1]["timestamp"]) if stops else now
        # Network reservation is charged in full even before all bytes move.
        total += max(0, stop - start) / 3600 * lease["hourly_usd"] + lease["network_reserve_usd"]
        if not stops:
            worst_remaining += max(0, effective_deadline(label,lease,rows) - now) / 3600 * lease["hourly_usd"]
    return {"estimated_upper_cost_usd": total, "reserved_future_rental_usd": worst_remaining}


def snapshot(api=None):
    api = api or client()
    with ThreadPoolExecutor(max_workers=2) as pool:
        budget_future = pool.submit(api.user_budget)
        instances_future = pool.submit(api.show_instances)
        budget, instances = budget_future.result(), instances_future.result()
    opening = json.loads((ROOT / "opening.json").read_text())["budget"]["conservative_available_usd"]
    estimate_row = estimate()
    attributed = max(0, opening - budget["conservative_available_usd"])
    row = {"budget": budget, **estimate_row, "account_decrease_usd": attributed,
           "conservative_spend_usd": max(attributed, estimate_row["estimated_upper_cost_usd"]),
           "instances": [{k: r.get(k) for k in ("id", "label", "actual_status", "dph_total", "start_date",
                        "ssh_host", "ssh_port", "public_ipaddr", "ports", "gpu_name", "machine_id", "geolocation",
                        "inet_down_cost", "inet_up_cost", "total_flops", "gpu_ram", "cpu_name", "cpu_ram",
                        "disk_usage", "disk_space", "extra_env", "status_msg")} for r in instances]}
    emit("ACCOUNT_SNAPSHOT", **row)
    return row


def destroy(label, instance_id, reason):
    if label not in leases():
        raise ValueError("Refusing destruction outside E026 leases")
    emit("DESTROY_ATTEMPT", label=label, instance_id=instance_id, reason=reason)
    result = client()._run(["destroy", "instance", str(instance_id), "--raw", "--yes"], timeout_seconds=60)
    emit("DESTROY_RESPONSE", label=label, instance_id=instance_id, returncode=result.returncode,
         stdout=result.stdout[-3000:], stderr=result.stderr[-1000:])
    if result.returncode or "Aborted" in result.stdout:
        raise RuntimeError(f"Vast destruction failed for owned instance {instance_id}")


def reconcile_absence(instances):
    present = {r["label"] for r in instances}
    rows = events()
    for label in leases():
        created = any(r["event"] == "CREATE_ATTEMPT" and r.get("label") == label for r in rows)
        absent = any(r["event"] == "ABSENCE_CONFIRMED" and r.get("label") == label for r in rows)
        attempted = any(r["event"] == "DESTROY_ATTEMPT" and r.get("label") == label for r in rows)
        if created and attempted and not absent and label not in present:
            emit("ABSENCE_CONFIRMED", label=label, stop_time=utc_now())


def watch():
    """Independent process; budgets and lease expiry fail closed on API faults."""
    emit("WATCHDOG_STARTED", pid=os.getpid(), hard_ceiling_usd=CEILING, soft_stop_usd=SOFT_STOP)
    failures = 0
    while True:
        append_event(ROOT / "heartbeat.jsonl", {"timestamp": utc_now(), "pid": os.getpid()})
        try:
            current = snapshot()
            failures = 0
            reconcile_absence(current["instances"])
            over = (current["conservative_spend_usd"] >= SOFT_STOP or
                    current["budget"]["conservative_available_usd"] <= RESERVE + 1)
            for instance in current["instances"]:
                label = instance.get("label")
                lease = leases().get(label)
                if lease and (over or time.time() >= effective_deadline(label,lease)):
                    destroy(label, instance["id"], "BUDGET_GUARD" if over else "LEASE_EXPIRED")
        except Exception as error:
            failures += 1
            emit("WATCHDOG_QUERY_ERROR", count=failures, error=str(error))
            if failures >= 2:
                # Recorded IDs permit safe teardown even when listing is unavailable.
                for record in events():
                    if record["event"] == "CREATE_SUCCEEDED":
                        try:
                            destroy(record["label"], record["instance_id"], "ACCOUNT_VISIBILITY_LOST")
                        except Exception as delete_error:
                            emit("WATCHDOG_DELETE_ERROR", error=str(delete_error))
        time.sleep(30)


def guard_alive():
    path = ROOT / "heartbeat.jsonl"
    if not path.exists():
        return False
    last = json.loads(path.read_text().splitlines()[-1])
    if time.time() - epoch(last["timestamp"]) > 100:
        return False
    try:
        import psutil
        return "experiment_026.vast" in " ".join(psutil.Process(last["pid"]).cmdline())
    except Exception:
        return False


def start_watchdog():
    if guard_alive():
        return {"status": "ALREADY_RUNNING"}
    log = (ROOT / "watchdog.log").open("ab")
    proc = subprocess.Popen([sys.executable, "-m", "swarm_inference.experiments.experiment_026.vast", "watch"],
                            stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    log.close()
    return {"watchdog_pid": proc.pid}


def create(offer: Offer, role: str, hours=3.0):
    if not guard_alive():
        raise RuntimeError("Independent budget watchdog is not alive")
    if not (0 < hours <= 4 and role in {"a", "b", "standby"}):
        raise ValueError("Bounded experimental lease required")
    disk_gb = 35
    rate = max(offer.dph_total, offer.effective_rate_per_hour(disk_gb))
    if not (offer.verified and offer.rentable and offer.gpu_count == 1 and offer.reliability >= .98
            and offer.gpu_ram_gib >= 11.5 and 0 < rate <= .35 and offer.cuda_max_version >= 13
            and max(offer.inet_down_cost_per_gb, offer.inet_up_cost_per_gb) <= .004):
        raise ValueError("Offer exceeds conservative E026 admission envelope")
    network = 100 * offer.inet_down_cost_per_gb + 50 * offer.inet_up_cost_per_gb
    current = snapshot()
    future = hours * rate + network + 1.0  # teardown/measurement-lag allowance
    if current["conservative_spend_usd"] + current["reserved_future_rental_usd"] + future > SOFT_STOP:
        raise RuntimeError("Projected cost exceeds soft budget with hard-ceiling reserve")
    if current["budget"]["conservative_available_usd"] - future < RESERVE:
        raise RuntimeError("Required follow-up reserve would be breached")
    if len([r for r in current["instances"] if r.get("label") in leases()]) >= 3:
        raise RuntimeError("Three paid nodes is the absolute E026 topology limit")
    label = f"e026-{role}-{int(time.time())}"
    lease = {"label": label, "role": role, "offer_id": offer.offer_id, "machine_id": offer.machine_id,
             "gpu_type": offer.gpu_name, "region": offer.raw.get("geolocation"), "disk_gb": disk_gb,
             "hourly_usd": rate, "storage_usd_per_gb_month": offer.storage_cost_per_gb_month,
             "ingress_usd_per_gb": offer.inet_down_cost_per_gb, "egress_usd_per_gb": offer.inet_up_cost_per_gb,
             "network_reserve_usd": network, "deadline_epoch": time.time() + hours * 3600,
             "image": IMAGE, "offer": offer.raw}
    write_once(ROOT / "leases" / f"{label}.json", lease)
    command = ["create", "instance", str(offer.offer_id), "--raw", "--image", IMAGE, "--disk", str(disk_gb),
               "--ssh", "--direct", "--cancel-unavail", "--label", label]
    emit("CREATE_ATTEMPT", label=label, start_time=utc_now(), command=command, **{k:lease[k] for k in ("hourly_usd", "gpu_type")})
    # No automatic retry: a timed-out POST may already have created a paid node.
    result = client()._run(command, timeout_seconds=150)
    payload = _json_output(result.stdout)
    emit("CREATE_RESPONSE", label=label, returncode=result.returncode, response=payload, stderr=result.stderr[-1500:])
    if result.returncode or not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError("Creation failed/ambiguous: inspect ledger and reconcile label before retry")
    instance_id = int(payload["new_contract"])
    emit("CREATE_SUCCEEDED", label=label, instance_id=instance_id)
    return {"label": label, "instance_id": instance_id, "maximum_lease_hours": hours, "hourly_usd": rate}


def extend(label, hours):
    if not guard_alive():raise RuntimeError("Independent budget watchdog is not alive")
    owned=leases()
    if label not in owned or not (0 < hours <= 4):raise ValueError("Bounded owned lease required")
    current=snapshot()
    if label not in {row.get("label") for row in current["instances"]}:
        raise RuntimeError("Cannot extend an absent instance")
    lease=owned[label]
    old=effective_deadline(label,lease)
    new=old+hours*3600
    added=hours*lease["hourly_usd"]
    projected=current["conservative_spend_usd"]+current["reserved_future_rental_usd"]+added+1.0
    if projected>SOFT_STOP:raise RuntimeError("Extension exceeds soft budget with teardown allowance")
    if current["budget"]["conservative_available_usd"]-(current["reserved_future_rental_usd"]+added+1.0)<RESERVE:
        raise RuntimeError("Extension would breach follow-up reserve")
    return emit("LEASE_EXTENDED",label=label,previous_deadline_epoch=old,new_deadline_epoch=new,
                added_hours=hours,added_rental_reserve_usd=added,projected_upper_with_allowance_usd=projected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "watch", "start-watchdog", "create", "extend", "destroy"))
    parser.add_argument("--offers", type=Path)
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--role")
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--label")
    parser.add_argument("--instance-id", type=int)
    args = parser.parse_args()
    if args.action == "watch":
        watch()
    elif args.action == "status":
        print(json.dumps(snapshot()))
    elif args.action == "start-watchdog":
        print(json.dumps(start_watchdog()))
    elif args.action == "create":
        data = json.loads(args.offers.read_text())
        offer = Offer.from_raw(next(row for row in data["offers"] if row["id"] == args.offer_id))
        if time.time() - epoch(data["timestamp"]) > 600:
            raise ValueError("Offer evidence is stale; refresh before admission")
        print(json.dumps(create(offer, args.role, args.hours)))
    elif args.action == "extend":
        print(json.dumps(extend(args.label,args.hours)))
    else:
        destroy(args.label, args.instance_id, "EXPERIMENT_NODE_NO_LONGER_REQUIRED")


if __name__ == "__main__":
    main()
