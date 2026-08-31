from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from swarm_inference.experiments.experiment_025.io import (
    append_jsonl,
    atomic_write_json,
)
from swarm_inference.experiments.experiment_025.providers.base import (
    ProviderMutationPolicy,
    ProviderOperation,
)
from swarm_inference.experiments.experiment_025.providers.runpod import (
    RunPodProvider,
    UrllibRunPodTransport,
)
from swarm_inference.experiments.experiment_025.runpod_cleanup import (
    cleanup_from_ledger,
    watchdog_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent E025 RunPod hard-TTL watchdog")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--stop", type=Path, required=True)
    parser.add_argument("--cleanup-receipt", type=Path, required=True)
    parser.add_argument("--startup-nonce", required=True)
    parser.add_argument("--allow-paid-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    arguments = parser.parse_args()
    policy = ProviderMutationPolicy.from_intent(allow_paid_run=arguments.allow_paid_run)
    policy.require(ProviderOperation.DELETE_POD)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is absent from the watchdog environment")
    provider = RunPodProvider(
        transport=UrllibRunPodTransport(api_key=api_key),
        policy=policy,
    )
    running = watchdog_receipt(
        run_id=arguments.run_id,
        stage=arguments.stage,
        deadline_epoch=arguments.deadline_epoch,
        startup_nonce=arguments.startup_nonce,
        status="RUNNING",
    )
    atomic_write_json(arguments.receipt, running)
    append_jsonl(arguments.log, {"event": "WATCHDOG_STARTED", **running})
    reason: str | None = None
    while reason is None:
        if arguments.trigger.exists():
            reason = "MANUAL_TRIGGER"
        elif time.time() >= arguments.deadline_epoch:
            reason = "TTL_EXCEEDED"
        elif arguments.stop.exists():
            cleanup = (
                json.loads(arguments.cleanup_receipt.read_text(encoding="utf-8"))
                if arguments.cleanup_receipt.is_file()
                else {}
            )
            if cleanup.get("zero_live_attributed_pods") is True:
                stopped = watchdog_receipt(
                    run_id=arguments.run_id,
                    stage=arguments.stage,
                    deadline_epoch=arguments.deadline_epoch,
                    startup_nonce=arguments.startup_nonce,
                    status="STOPPED_AFTER_ZERO_LIVE_VERIFICATION",
                )
                atomic_write_json(arguments.receipt, stopped)
                append_jsonl(arguments.log, {"event": "WATCHDOG_STOPPED", **stopped})
                return 0
        if reason is None:
            time.sleep(max(1.0, min(arguments.poll_seconds, 30.0)))
            atomic_write_json(
                arguments.receipt,
                watchdog_receipt(
                    run_id=arguments.run_id,
                    stage=arguments.stage,
                    deadline_epoch=arguments.deadline_epoch,
                    startup_nonce=arguments.startup_nonce,
                    status="RUNNING",
                ),
            )
    cleanup = cleanup_from_ledger(
        provider=provider,
        ledger_path=arguments.ledger,
        run_id=arguments.run_id,
        reason=f"watchdog:{arguments.stage}:{reason}",
    )
    atomic_write_json(arguments.cleanup_receipt, cleanup)
    final = watchdog_receipt(
        run_id=arguments.run_id,
        stage=arguments.stage,
        deadline_epoch=arguments.deadline_epoch,
        startup_nonce=arguments.startup_nonce,
        status=(
            "TIMEOUT_CLEANUP_COMPLETE"
            if cleanup["zero_live_attributed_pods"]
            else "TIMEOUT_CLEANUP_FAILED"
        ),
    )
    final["trigger_reason"] = reason
    final["cleanup"] = cleanup
    atomic_write_json(arguments.receipt, final)
    append_jsonl(arguments.log, {"event": "WATCHDOG_FINISHED", **final})
    return 0 if cleanup["zero_live_attributed_pods"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
