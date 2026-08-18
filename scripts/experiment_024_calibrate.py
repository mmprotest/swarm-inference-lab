"""Run fresh physical calibration for Experiment 024."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.service_calibration import (  # noqa: E402
    run_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("all", "dense", "whole", "p8", "fusion", "assemble"),
        default="all",
    )
    arguments = parser.parse_args()
    result = run_calibration(REPO_ROOT, arm=arguments.arm)
    statuses = [
        str(value.get("status", "PASS"))
        for value in result.values()
        if isinstance(value, dict)
    ]
    print(json.dumps({"arm": arguments.arm, "statuses": statuses}, indent=2))
    return 0 if all(status in {"PASS", "RUNNING"} for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
