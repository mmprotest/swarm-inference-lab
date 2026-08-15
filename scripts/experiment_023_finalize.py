"""Create the fail-closed Experiment 023 audit package and report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_023.finalize import (  # noqa: E402
    finalize_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chart-qa-status",
        choices=("PENDING_VISUAL_QA", "PASS_VISUAL_INSPECTION"),
        default="PENDING_VISUAL_QA",
    )
    arguments = parser.parse_args()
    try:
        result = finalize_experiment(
            REPO_ROOT,
            chart_qa_status=arguments.chart_qa_status,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FINALIZATION_FAILED", "reason": f"{type(exc).__name__}: {exc}"},
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
