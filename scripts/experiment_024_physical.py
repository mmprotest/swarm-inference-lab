"""Run repaired E024 physical D correctness and token-semantics audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.authoritative_runner import (  # noqa: E402
    run_physical_d_phase,
)


def main() -> int:
    result = run_physical_d_phase(REPO_ROOT)
    print(
        json.dumps(
            {
                "physical_d": result["physical_d"]["status"],
                "token_semantics": result["token_semantics"]["status"],
            },
            indent=2,
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
