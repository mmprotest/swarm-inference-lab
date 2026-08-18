"""Finalize repaired E024 economics, verdict, report, charts, and optional QA."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from swarm_inference.experiments.experiment_024.authoritative_runner import (  # noqa: E402
    finalize_commercial_gates,
)
from swarm_inference.experiments.experiment_024.reporting import (  # noqa: E402
    finalize_authoritative,
    record_authoritative_qa,
)

TIMING_PATH = REPO_ROOT / "artifacts/experiment-024/repair/execution-timing.json"


def _elapsed() -> tuple[str, float]:
    timing = json.loads(TIMING_PATH.read_text(encoding="utf-8"))
    started_text = str(timing["started_at_utc"])
    started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    return started_text, (datetime.now(UTC) - started).total_seconds()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path)
    args = parser.parse_args()
    started, elapsed = _elapsed()
    if args.qa_json is None:
        gates = finalize_commercial_gates(REPO_ROOT)
        result = finalize_authoritative(
            REPO_ROOT,
            started_at_utc=started,
            elapsed_seconds=elapsed,
        )
        payload = {"commercial_gates": gates, "summary": result}
    else:
        qa = json.loads(args.qa_json.read_text(encoding="utf-8"))
        payload = record_authoritative_qa(
            REPO_ROOT,
            compileall_status=str(qa["compileall_status"]),
            focused_tests=str(qa["focused_tests"]),
            full_repository_tests=str(qa["full_repository_tests"]),
            e024_ruff_findings=int(qa["e024_ruff_findings"]),
            repository_ruff_before=int(qa["repository_ruff_before"]),
            repository_ruff_after=int(qa["repository_ruff_after"]),
            chart_visual_qa=str(qa["chart_visual_qa"]),
            elapsed_seconds=elapsed,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
