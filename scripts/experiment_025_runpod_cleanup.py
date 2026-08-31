from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from swarm_inference.experiments.experiment_025.io import atomic_write_json
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
)
from swarm_inference.experiments.experiment_025.runpod_planning import RUN_ID


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permanently delete only ledgered E025 RunPod Pods"
    )
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--allow-paid-run", action="store_true")
    arguments = parser.parse_args()
    policy = ProviderMutationPolicy.from_intent(allow_paid_run=arguments.allow_paid_run)
    policy.require(ProviderOperation.DELETE_POD)
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY must be supplied through the process environment")
    provider = RunPodProvider(
        transport=UrllibRunPodTransport(api_key=api_key),
        policy=policy,
    )
    receipt = cleanup_from_ledger(
        provider=provider,
        ledger_path=arguments.ledger.resolve(),
        run_id=arguments.run_id,
        reason="explicit-emergency-cleanup",
    )
    if arguments.receipt:
        atomic_write_json(arguments.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["zero_live_attributed_pods"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
