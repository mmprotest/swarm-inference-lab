"""Run repaired E024 full two-token autoregressive correctness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.authoritative_runner import (  # noqa: E402
    run_two_token_phase,
)


def main() -> int:
    result = run_two_token_phase(REPO_ROOT)
    print(
        json.dumps(
            {
                "status": result["status"],
                "T1": result["step_1_token_t1"],
                "T2": result["step_2_token_t2"],
            },
            indent=2,
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
