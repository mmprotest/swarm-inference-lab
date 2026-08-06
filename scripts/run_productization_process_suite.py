"""Run the product process-suite repeatability gate with hard per-run timeouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.acceptance.productization import (
    ACCEPTANCE_BUNDLE_VERSION,
    NON_GPU_PRODUCT_TEST_ARGUMENTS,
    NON_PRODUCT_SOURCE_AUDIT_TESTS,
    REPEATABILITY_SCHEMA_VERSION,
    REPEATABILITY_TEST_COMMAND_VERSION,
)

REQUIRED_FULL_RUNS = 3
REQUIRED_STAGE_RUNS = 5
WARNING_FRAGMENTS = (
    "resource_tracker",
    "leaked semaphore",
    "leaked shared_memory",
    "Task was destroyed but it is pending",
    "Unclosed client session",
    "unclosed transport",
    "unclosed event loop",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-runs", type=int, default=REQUIRED_FULL_RUNS)
    parser.add_argument("--stage-runs", type=int, default=REQUIRED_STAGE_RUNS)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("artifacts/acceptance"))
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_value(repository_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }


def _snapshot_descendants(
    process: subprocess.Popen[str],
    observed: dict[int, float],
) -> None:
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        return
    for child in descendants:
        with suppress(psutil.Error):
            observed[child.pid] = child.create_time()


def _still_alive(observed: dict[int, float]) -> list[psutil.Process]:
    alive: list[psutil.Process] = []
    for pid, created_at in observed.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created_at and process.is_running():
                alive.append(process)
        except psutil.Error:
            continue
    return alive


def _terminate_processes(processes: list[psutil.Process]) -> dict[str, int]:
    counts = {"terminate_count": 0, "kill_count": 0, "survivor_count": 0}
    for process in reversed(processes):
        try:
            process.terminate()
            counts["terminate_count"] += 1
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for process in alive:
        try:
            process.kill()
            counts["kill_count"] += 1
        except psutil.Error:
            pass
    _, survivors = psutil.wait_procs(alive, timeout=5)
    counts["survivor_count"] = len(survivors)
    return counts


def _terminate_tree(process: subprocess.Popen[str]) -> dict[str, int]:
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    counts = _terminate_processes(descendants)
    if process.poll() is None:
        process.terminate()
        counts["terminate_count"] += 1
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            counts["kill_count"] += 1
            process.wait(timeout=5)
    return counts


def _lifecycle_counts(path: Path) -> dict[str, int]:
    totals = {
        "graceful_shutdown_count": 0,
        "unexpected_terminate_count": 0,
        "unexpected_kill_count": 0,
        "expected_terminate_count": 0,
        "expected_kill_count": 0,
        "leaked_process_count": 0,
    }
    paths = ([path] if path.is_file() else []) + sorted(
        path.parent.glob(f"{path.name}.worker-*.json")
    )
    for lifecycle_path in paths:
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            for name in totals:
                totals[name] += int(payload.get(name, 0))
    return totals


def _not_run(name: str, command: list[str], reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "command": command,
        "started_at": None,
        "finished_at": None,
        "duration_s": 0.0,
        "exit_code": None,
        "status": "NOT_RUN",
        "reason": reason,
        "test_counts": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
        "warning_scan": {"status": "NOT_RUN", "matches": [], "ignore_list": []},
        "resource_warning_count": 0,
        "graceful_shutdown_count": 0,
        "unexpected_terminate_count": 0,
        "unexpected_kill_count": 0,
        "expected_terminate_count": 0,
        "expected_kill_count": 0,
        "leaked_process_count": 0,
        "stdout_log": None,
        "stderr_log": None,
        "lifecycle_log": None,
        "checksums": {},
    }


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
    junit_path = output / f"{name}.junit.xml"
    lifecycle_path = output / f"{name}.lifecycle.jsonl"
    command = [
        sys.executable,
        "-m",
        "pytest",
        *tests,
        "-q",
        "--basetemp",
        str(temp),
        "--junitxml",
        str(junit_path),
    ]
    environment = dict(os.environ)
    environment["TEMP"] = str(temp)
    environment["TMP"] = str(temp)
    environment["SWARM_PROCESS_LIFECYCLE_LOG"] = str(lifecycle_path)
    process = subprocess.Popen(
        command,
        cwd=repository_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = _utc_now()
    started = time.monotonic()
    deadline = started + timeout_s
    observed: dict[int, float] = {}
    timed_out = False
    external_force = {"terminate_count": 0, "kill_count": 0, "survivor_count": 0}
    while True:
        _snapshot_descendants(process, observed)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            external_force = _terminate_tree(process)
            stdout, stderr = process.communicate()
            break
        try:
            stdout, stderr = process.communicate(timeout=min(remaining, 0.25))
            break
        except subprocess.TimeoutExpired:
            continue
    duration = time.monotonic() - started
    finished_at = _utc_now()

    # Give descendants a brief natural-exit window, then clean any leak while
    # retaining a failing count in the evidence.
    leak_deadline = time.monotonic() + 2
    alive = _still_alive(observed)
    while alive and time.monotonic() < leak_deadline:
        time.sleep(0.02)
        alive = _still_alive(observed)
    leaked_process_count = len(alive)
    if alive:
        leak_cleanup = _terminate_processes(alive)
        external_force["terminate_count"] += leak_cleanup["terminate_count"]
        external_force["kill_count"] += leak_cleanup["kill_count"]
        external_force["survivor_count"] += leak_cleanup["survivor_count"]

    stdout_path = output / f"{name}.stdout.log"
    stderr_path = output / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    combined_output = f"{stdout}\n{stderr}".lower()
    warnings = [fragment for fragment in WARNING_FRAGMENTS if fragment.lower() in combined_output]
    lifecycle_error: str | None = None
    try:
        lifecycle = _lifecycle_counts(lifecycle_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        lifecycle = {
            "graceful_shutdown_count": 0,
            "unexpected_terminate_count": 0,
            "unexpected_kill_count": 0,
            "expected_terminate_count": 0,
            "expected_kill_count": 0,
            "leaked_process_count": 0,
        }
        lifecycle_error = f"{type(exc).__name__}: {exc}"
    lifecycle["unexpected_terminate_count"] += external_force["terminate_count"]
    lifecycle["unexpected_kill_count"] += external_force["kill_count"]
    lifecycle["leaked_process_count"] += leaked_process_count

    if timed_out:
        status = "TIMEOUT"
        reason = f"external timeout after {timeout_s:.1f}s"
    elif warnings:
        status = "WARNING_FAILURE"
        reason = "managed-resource warning signature was emitted"
    elif process.returncode != 0:
        status = "FAIL"
        reason = f"pytest exited with code {process.returncode}"
    elif lifecycle_error is not None:
        status = "FAIL"
        reason = f"process lifecycle telemetry is invalid: {lifecycle_error}"
    elif (
        lifecycle["unexpected_terminate_count"]
        or lifecycle["unexpected_kill_count"]
        or lifecycle["leaked_process_count"]
        or external_force["survivor_count"]
    ):
        status = "FAIL"
        reason = "managed process cleanup required force or leaked a child"
    else:
        status = "PASS"
        reason = "pytest passed with warning-free graceful process cleanup"

    cleanup_error: str | None = None
    try:
        shutil.rmtree(temp)
    except OSError as exc:
        cleanup_error = f"{type(exc).__name__}: {exc}"
        if status == "PASS":
            status = "FAIL"
            reason = "pytest temporary directory could not be removed"

    lifecycle_paths = ([lifecycle_path] if lifecycle_path.is_file() else []) + sorted(
        lifecycle_path.parent.glob(f"{lifecycle_path.name}.worker-*.json")
    )
    files = [stdout_path, stderr_path]
    if junit_path.is_file():
        files.append(junit_path)
    files.extend(lifecycle_paths)
    checksums = {path.name: f"sha256:{_sha256(path)}" for path in files}
    try:
        test_counts = _junit_counts(junit_path)
    except (OSError, ET.ParseError, ValueError):
        test_counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        if status == "PASS":
            status = "FAIL"
            reason = "JUnit execution counts are unavailable"
    if status == "PASS" and test_counts["skipped"]:
        status = "SKIP"
        reason = f"{test_counts['skipped']} required product tests skipped"
    return {
        "name": name,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration,
        "exit_code": process.returncode,
        "status": status,
        "reason": reason,
        "test_counts": test_counts,
        "warning_scan": {
            "status": "PASS" if not warnings else "FAIL",
            "matches": warnings,
            "ignore_list": [],
        },
        "resource_warning_count": len(warnings),
        **lifecycle,
        "stdout_log": stdout_path.name,
        "stderr_log": stderr_path.name,
        "lifecycle_log": lifecycle_path.name if lifecycle_path.is_file() else None,
        "lifecycle_logs": [item.name for item in lifecycle_paths],
        "lifecycle_parse_error": lifecycle_error,
        "temporary_cleanup_error": cleanup_error,
        "checksums": checksums,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.full_runs < 1 or args.stage_runs < 1 or args.timeout_seconds <= 0:
        _parser().error("run counts and timeout must be positive")
    repository_root = Path(__file__).resolve().parents[1]
    started_at = _utc_now()
    start_git_status = _git_value(repository_root, "status", "--porcelain")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.expanduser().resolve() / f"process-repeatability-{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    full_tests = list(NON_GPU_PRODUCT_TEST_ARGUMENTS)
    # Real-model GPU gates are evaluated separately.  Repeatability proves the
    # complete non-GPU product process module. Opt-in source-audit tests for an
    # already completed Experiment 007 run are explicitly excluded and retained
    # in the evidence contract; real-model GPU nodes remain separately gated.
    stage_tests = ["tests/integration/test_product_stage_ring.py", "-m", "not gpu"]
    specifications = [(f"full-{index + 1}", full_tests) for index in range(args.full_runs)] + [
        (f"stage-ring-{index + 1}", stage_tests) for index in range(args.stage_runs)
    ]
    results: list[dict[str, Any]] = []
    previous_failure = False
    for name, tests in specifications:
        command = [sys.executable, "-m", "pytest", *tests, "-q"]
        if previous_failure:
            result = _not_run(name, command, "a preceding required repeatability run failed")
        else:
            result = _run(
                repository_root=repository_root,
                output=output,
                name=name,
                tests=tests,
                timeout_s=args.timeout_seconds,
            )
            previous_failure = result["status"] != "PASS"
        results.append(result)
        print(
            f"{result['name']}={result['status']} duration_s={result['duration_s']:.3f} "
            f"warnings={result['resource_warning_count']} "
            f"unexpected_terminate={result['unexpected_terminate_count']} "
            f"unexpected_kill={result['unexpected_kill_count']}",
            flush=True,
        )

    finish_git_status = _git_value(repository_root, "status", "--porcelain")
    all_pass = all(result["status"] == "PASS" for result in results)
    provenance_stable = start_git_status == finish_git_status
    requested_is_required = (
        args.full_runs == REQUIRED_FULL_RUNS and args.stage_runs == REQUIRED_STAGE_RUNS
    )
    if all_pass and provenance_stable and requested_is_required:
        overall = "PASS"
    elif (
        any(result["status"] in {"FAIL", "TIMEOUT", "WARNING_FAILURE"} for result in results)
        or not provenance_stable
    ):
        overall = "FAIL"
    else:
        overall = "INCOMPLETE"

    acceptance_source = repository_root / "src/swarm_inference/acceptance/productization.py"
    payload: dict[str, Any] = {
        "document_type": "swarm-process-repeatability",
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "test_command_version": REPEATABILITY_TEST_COMMAND_VERSION,
        "acceptance_schema_version": ACCEPTANCE_BUNDLE_VERSION,
        "excluded_source_audit_tests": list(NON_PRODUCT_SOURCE_AUDIT_TESTS),
        "git_commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "git_dirty": start_git_status is None or bool(start_git_status),
        "git_status": start_git_status,
        "finish_git_status": finish_git_status,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "required_runs": {
            "full_process_suite": REQUIRED_FULL_RUNS,
            "stage_ring_module": REQUIRED_STAGE_RUNS,
        },
        "requested_runs": {
            "full_process_suite": args.full_runs,
            "stage_ring_module": args.stage_runs,
        },
        "commands": [result["command"] for result in results],
        "results": results,
        "warning_fragments": list(WARNING_FRAGMENTS),
        "warning_ignore_list": [],
        "process_runner_sha256": f"sha256:{_sha256(Path(__file__))}",
        "acceptance_source_sha256": f"sha256:{_sha256(acceptance_source)}",
        "overall_repeatability_status": overall,
    }
    payload["evidence_checksum"] = f"sha256:{_payload_checksum(payload)}"
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"evidence={output}", flush=True)
    print(f"summary_sha256=sha256:{_sha256(summary_path)}", flush=True)
    print(f"overall_status={overall}", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
