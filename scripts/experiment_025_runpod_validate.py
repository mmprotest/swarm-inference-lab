from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025.io import atomic_write_json, utc_now
from swarm_inference.experiments.experiment_025.runpod_planning import RUN_ID

NEW_SOURCE_FILES = [
    "src/swarm_inference/experiments/experiment_025/providers/__init__.py",
    "src/swarm_inference/experiments/experiment_025/providers/base.py",
    "src/swarm_inference/experiments/experiment_025/providers/runpod.py",
    "src/swarm_inference/experiments/experiment_025/runpod_cleanup.py",
    "src/swarm_inference/experiments/experiment_025/runpod_inventory.py",
    "src/swarm_inference/experiments/experiment_025/runpod_lifecycle.py",
    "src/swarm_inference/experiments/experiment_025/runpod_operator.py",
    "src/swarm_inference/experiments/experiment_025/runpod_planning.py",
    "src/swarm_inference/experiments/experiment_025/runpod_preparation.py",
    "src/swarm_inference/experiments/experiment_025/runpod_security.py",
    "scripts/experiment_025_runpod.py",
    "scripts/experiment_025_runpod_cleanup.py",
    "scripts/experiment_025_runpod_image_verify.py",
    "scripts/experiment_025_runpod_inventory.py",
    "scripts/experiment_025_runpod_secret_scan.py",
    "scripts/experiment_025_runpod_validate.py",
    "scripts/experiment_025_runpod_watchdog.py",
    "tests/test_experiment_025_runpod.py",
]


def _run(
    name: str,
    command: list[str],
    *,
    repo: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = utc_now()
    process = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=600,
    )
    return {
        "name": name,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "command": command,
        "returncode": process.returncode,
        "status": "PASS" if process.returncode == 0 else "FAIL",
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
    }


def main() -> int:
    repo = Path.cwd().resolve()
    output = (
        repo
        / "artifacts"
        / "runs"
        / f"experiment-025-{RUN_ID}"
        / "preflight"
        / "runpod"
        / "runpod-focused-tests.json"
    )
    temporary = repo / ".test-temp" / f"runpod-validation-{os.getpid()}"
    temporary.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo / "src")
    environment["TEMP"] = str(temporary)
    environment["TMP"] = str(temporary)
    python = sys.executable
    checks = [
        _run(
            "compile_and_import",
            [
                python,
                "-c",
                (
                    "import compileall; "
                    "assert compileall.compile_dir('src/swarm_inference/experiments/experiment_025', quiet=1); "
                    "import swarm_inference.experiments.experiment_025.runpod_operator; "
                    "import swarm_inference.experiments.experiment_025.runpod_preparation"
                ),
            ],
            repo=repo,
            environment=environment,
        ),
        _run(
            "focused_runpod_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_experiment_025_runpod.py",
                "--basetemp",
                str(temporary / "pytest-runpod"),
            ],
            repo=repo,
            environment=environment,
        ),
        _run(
            "existing_supervisor_tests",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_experiment_025.py::test_grouped_supervisor_rejects_duplicate_gpu_slots",
                "tests/test_experiment_025.py::test_single_worker_supervisor_preserves_parent_expert_endpoints",
                "--basetemp",
                str(temporary / "pytest-supervisor"),
            ],
            repo=repo,
            environment=environment,
        ),
        _run(
            "ruff_format_check",
            [python, "-m", "ruff", "format", "--check", *NEW_SOURCE_FILES],
            repo=repo,
            environment=environment,
        ),
        _run(
            "ruff_lint",
            [python, "-m", "ruff", "check", *NEW_SOURCE_FILES],
            repo=repo,
            environment=environment,
        ),
        _run(
            "mypy_targeted",
            [python, "-m", "mypy", *NEW_SOURCE_FILES[:-1]],
            repo=repo,
            environment=environment,
        ),
    ]
    passed = all(row["status"] == "PASS" for row in checks)
    receipt = {
        "schema_version": "experiment-025-runpod-focused-tests-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if passed else "FAIL",
        "provider_mode": "READ_ONLY_PREPARATION",
        "provider_network_calls": 0,
        "provider_mutations": [],
        "paid_resources_created": 0,
        "mutation_firewall_passed": next(
            row for row in checks if row["name"] == "focused_runpod_tests"
        )["status"]
        == "PASS",
        "worker_image_changed": False,
        "local_rtx_5090_recanary_required": False,
        "scope": "New provider/controller code and the existing multi-worker supervisor contract.",
        "checks": checks,
    }
    atomic_write_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
