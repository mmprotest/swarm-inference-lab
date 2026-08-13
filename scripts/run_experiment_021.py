"""Run the zero-rental independent-machine Experiment 021."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_021.runner import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
