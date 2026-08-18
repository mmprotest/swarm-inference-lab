"""Freeze repaired E024 code after calibration and before performance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.authoritative_runner import (  # noqa: E402
    run_code_freeze,
)


def main() -> int:
    result = run_code_freeze(REPO_ROOT)
    print(
        json.dumps(
            {
                "status": result["status"],
                "file_count": result["file_count"],
                "freeze_sha256": result["freeze_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
