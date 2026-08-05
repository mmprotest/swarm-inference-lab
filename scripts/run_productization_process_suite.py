"""Run the product process-suite repeatability gate with hard per-run timeouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-runs", type=int, default=3)
    parser.add_argument("--stage-runs", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance"))
    return parser


def _terminate_tree(process: subprocess.Popen[str]) -> None:
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    for child in descendants:
        with suppress(psutil.Error):
            child.terminate()
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        try:
            child.kill()
            child.wait(timeout=5)
        except psutil.Error:
            pass


def _run(
    *,
    repository_root: Path,
    output: Path,
    name: str,
    tests: list[str],
    timeout_s: float,
) -> dict[str, Any]:
    temp = output / "temp" / name
    temp.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "--basetemp",
        str(temp),
    ]
    environment = dict(os.environ)
    environment["TEMP"] = str(temp)
    environment["TMP"] = str(temp)
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_tree(process)
        stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    stdout_path = output / f"{name}.stdout.log"
    stderr_path = output / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    warning_fragments = (
        "leaked semaphore",
        "resource_tracker",
        "Task was destroyed but it is pending",
        "Unclosed client session",
        "unclosed transport",
    )
    combined_output = f"{stdout}\n{stderr}".lower()
    warnings = [fragment for fragment in warning_fragments if fragment.lower() in combined_output]
    passed = not timed_out and process.returncode == 0 and not warnings
    return {
        "name": name,
        "command": command,
        "duration_s": duration,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "resource_warnings": warnings,
        "status": "PASS" if passed else "FAIL",
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.full_runs < 1 or args.stage_runs < 1 or args.timeout_seconds <= 0:
        _parser().error("run counts and timeout must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.expanduser().resolve() / f"process-repeatability-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for index in range(args.full_runs):
        result = _run(
            repository_root=repository_root,
            output=output,
            name=f"full-{index + 1}",
            tests=["tests/integration", "tests/failure", "-m", "not gpu"],
            timeout_s=args.timeout_seconds,
        )
        results.append(result)
        print(
            f"{result['name']}={result['status']} duration_s={result['duration_s']:.3f}",
            flush=True,
        )
        if result["status"] != "PASS":
            break
    if all(result["status"] == "PASS" for result in results):
        for index in range(args.stage_runs):
            result = _run(
                repository_root=repository_root,
                output=output,
                name=f"stage-ring-{index + 1}",
                tests=["tests/integration/test_product_stage_ring.py"],
                timeout_s=args.timeout_seconds,
            )
            results.append(result)
            print(
                f"{result['name']}={result['status']} duration_s={result['duration_s']:.3f}",
                flush=True,
            )
            if result["status"] != "PASS":
                break
    overall = (
        "PASS"
        if len(results) == args.full_runs + args.stage_runs
        and all(result["status"] == "PASS" for result in results)
        else "FAIL"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "document_type": "swarm-process-repeatability",
                "format_version": 1,
                "status": overall,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"evidence={output}", flush=True)
    print(f"overall_status={overall}", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
