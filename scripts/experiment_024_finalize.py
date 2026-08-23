"""Finalize the E024 fixed-anchor closure without rerunning performance."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.reporting import (  # noqa: E402
    finalize_closure,
)

TIMING_PATH = REPO_ROOT / "artifacts/experiment-024/closure/execution-timing.json"


def _elapsed() -> tuple[str, float]:
    timing = json.loads(TIMING_PATH.read_text(encoding="utf-8"))
    started_text = str(timing["started_at_utc"])
    started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    return started_text, (datetime.now(UTC) - started).total_seconds()


def main() -> int:
    started, elapsed = _elapsed()
    summary = finalize_closure(
        REPO_ROOT,
        started_at_utc=started,
        elapsed_seconds=elapsed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 2 if summary["final_verdict"] == "MODEL_INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
