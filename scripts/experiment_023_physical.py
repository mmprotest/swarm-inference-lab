"""Run E023's mandatory local RTX 5090 duplicate-group validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_023.physical import (  # noqa: E402
    run_physical_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    arguments = parser.parse_args()
    try:
        result = run_physical_validation(
            repo=REPO_ROOT,
            checkpoint=arguments.checkpoint,
        )
    except Exception as exc:
        print(json.dumps({"status": "MODEL_INVALID", "reason": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_gate": result["primary_gate"],
                "case_count": result["case_count"],
                "hedge_service_drift": result["hedge_service_drift"],
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
