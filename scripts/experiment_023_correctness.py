"""Run E023 compatibility or full correctness gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_023.correctness import (  # noqa: E402
    run_engine_compatibility,
    run_full_correctness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compatibility", "full"))
    parser.add_argument("--primary-attempt", default="deterministic-run-2")
    parser.add_argument("--timeout-seconds", type=float, default=14_400.0)
    arguments = parser.parse_args()
    try:
        if arguments.phase == "compatibility":
            rows = run_engine_compatibility(REPO_ROOT)
            result = {"status": "PASS", "case_count": len(rows)}
        else:
            rows = run_full_correctness(
                REPO_ROOT,
                primary_attempt=arguments.primary_attempt,
                timeout_seconds=arguments.timeout_seconds,
            )
            result = {"status": "PASS", "case_count": len(rows)}
    except Exception as exc:
        print(json.dumps({"status": "MODEL_INVALID", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
