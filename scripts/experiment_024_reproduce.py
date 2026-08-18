"""Run repaired E024 Stage A and frozen-placement Stage B reproducibility."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.authoritative_runner import (  # noqa: E402
    run_reproducibility_phase,
)


def main() -> int:
    result = run_reproducibility_phase(REPO_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
