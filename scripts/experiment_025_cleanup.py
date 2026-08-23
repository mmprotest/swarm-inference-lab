from __future__ import annotations

import argparse
import json
from pathlib import Path

from swarm_inference.experiments.experiment_025.io import atomic_write_json
from swarm_inference.experiments.experiment_025.vast_lifecycle import (
    destroy_all_from_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotently destroy ledger-owned E025 instances")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason", default="manual emergency cleanup")
    arguments = parser.parse_args()
    receipt = destroy_all_from_ledger(
        ledger_path=arguments.ledger.resolve(),
        run_id=arguments.run_id,
        reason=arguments.reason,
        attempts=6,
    )
    atomic_write_json(arguments.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["zero_live_e025_instances"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
