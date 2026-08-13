"""Run and record the focused Experiment 022 test suite."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
# The E020 lifecycle regression launches a fresh Python worker process. Mirror the
# source-tree import path into that child instead of assuming an editable install.
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    value for value in (str(SOURCE_ROOT), existing_pythonpath) if value
)

from swarm_inference.experiments.experiment_022.io import (  # noqa: E402
    atomic_write_json,
    read_json,
)


class _Results:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when != "call":
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1


def main() -> int:
    artifact = REPO_ROOT / "artifacts" / "experiment-022"
    output = artifact / "test-results.json"
    existing = read_json(output) if output.is_file() else {
        "schema_version": "experiment-022-test-results-v1"
    }
    collector = _Results()
    started = time.perf_counter_ns()
    exit_code = pytest.main(
        [
            "-q",
            str(REPO_ROOT / "tests" / "test_experiment_022.py"),
            str(REPO_ROOT / "tests/unit/test_experiment_020_transport.py"),
            str(REPO_ROOT / "tests/unit/test_experiment_020_worker_lifecycle.py"),
            str(REPO_ROOT / "tests/test_experiment_021.py"),
            "--basetemp",
            str(artifact / "test-temp-final"),
        ],
        plugins=[collector],
    )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    existing["pytest"] = {
        "status": "PASS" if exit_code == pytest.ExitCode.OK else "FAIL",
        "exit_code": int(exit_code),
        "passed": collector.passed,
        "failed": collector.failed,
        "skipped": collector.skipped,
        "elapsed_seconds": elapsed,
        "command": "python scripts/experiment_022_test.py",
    }
    if exit_code != pytest.ExitCode.OK:
        existing["status"] = "FAIL"
    atomic_write_json(output, existing)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
