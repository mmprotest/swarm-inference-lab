"""Verify and freeze immutable E022 inputs for Experiment 023."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_023.freeze import freeze_e023  # noqa: E402


def main() -> int:
    try:
        result = freeze_e023(REPO_ROOT)
    except Exception as exc:
        print(json.dumps({"status": "MODEL_INVALID", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
