"""Run the focused Experiment 023 unit and completion tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = dict(os.environ)
    source = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_experiment_023.py",
        "tests/test_experiment_023_completion.py",
    )
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
