"""Verify E026 teardown and seal a privacy-minimized final Vast balance snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket

from vastai.api.billing import show_user
from vastai.api.client import VastClient as HttpClient
from vastai.sync.client import SyncClient

from .io import utc_now, write_once
from .vast import emit, events


TARGETS = {
    "e026-a-1789033163": 50478198,
    "e026-b-1789033167": 50478207,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-ip", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiment-026/cost/final-provider-snapshot.json"),
    )
    args = parser.parse_args()

    original = socket.getaddrinfo

    def resolve(host, port, *rest, **kwargs):
        return original(args.api_ip if host == "console.vast.ai" else host, port, *rest, **kwargs)

    socket.getaddrinfo = resolve
    sdk = SyncClient()
    active_ids = {instance.id for instance in sdk.show_instances()}
    user = show_user(HttpClient(api_key=sdk._api_key))
    credit = float(user.get("credit") or 0.0)
    balance = float(user.get("balance") or 0.0)
    presence = {label: instance_id in active_ids for label, instance_id in TARGETS.items()}
    row = {
        "timestamp": utc_now(),
        "target_presence": presence,
        "all_e026_instances_absent": not any(presence.values()),
        "budget": {
            "credit_usd": credit,
            "balance_usd": balance,
            "conservative_available_usd": max(0.0, credit + balance),
            "identity_fields_persisted": False,
            "api_key_persisted": False,
        },
        "api_dns_override": {"hostname": "console.vast.ai", "ip": args.api_ip, "scope": "this process"},
    }
    write_once(args.output, row)
    prior = events()
    if row["all_e026_instances_absent"]:
        for label in TARGETS:
            if not any(item["event"] == "ABSENCE_CONFIRMED" and item.get("label") == label for item in prior):
                emit("ABSENCE_CONFIRMED", label=label, stop_time=row["timestamp"], evidence=str(args.output))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
