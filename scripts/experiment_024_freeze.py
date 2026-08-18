"""Run the Experiment 024 immutable-input gate and fail-closed finalizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.runner import run_phase0  # noqa: E402


def main() -> int:
    result = run_phase0(REPO_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
