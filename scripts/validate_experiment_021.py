"""Validate the completed Experiment 021 artifact and test bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_021.qa import (  # noqa: E402
    materialize_test_results,
)


def main() -> int:
    result = materialize_test_results(ROOT, ROOT / "artifacts" / "experiment-021")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
