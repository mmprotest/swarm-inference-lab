"""Enter E024 calibration only after the immutable-input gate passes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.freeze import (  # noqa: E402
    audit_immutable_inputs,
)


def main() -> int:
    audit = audit_immutable_inputs(REPO_ROOT)
    print(json.dumps(audit.as_dict(), indent=2, sort_keys=True))
    return 2 if audit.status != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
