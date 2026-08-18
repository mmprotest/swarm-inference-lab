"""Run deterministic Experiment 023 planning and serving evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_023.runner import (  # noqa: E402
    run_deterministic_attempt,
)


def _progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt",
        default="attempts/deterministic-run-2",
        help="path below artifacts/experiment-023",
    )
    parser.add_argument(
        "--inventory",
        action="append",
        dest="inventories",
        help="run only this frozen inventory (repeatable; diagnostic subsets only)",
    )
    arguments = parser.parse_args()
    attempt_root = (
        REPO_ROOT / "artifacts" / "experiment-023" / arguments.attempt
    ).resolve()
    experiment_root = (
        REPO_ROOT / "artifacts" / "experiment-023"
    ).resolve()
    if experiment_root not in attempt_root.parents:
        raise SystemExit("attempt output must remain below artifacts/experiment-023")
    try:
        result = run_deterministic_attempt(
            REPO_ROOT,
            attempt_root,
            inventory_ids=arguments.inventories,
            progress=_progress,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "MODEL_INVALID", "reason": f"{type(exc).__name__}: {exc}"},
                indent=2,
            ),
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "PASS",
                "inventory_count": len(result.inventory_results),
                "elapsed_seconds": result.elapsed_seconds,
                "attempt_root": str(attempt_root),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
