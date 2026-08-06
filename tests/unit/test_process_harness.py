from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.testing.process_harness import (
    ChildStartupError,
    ProcessCleanupError,
    ProductCluster,
)
from swarm_inference.worker.process_main import _write_managed_process_lifecycle_record


def _ready_until_stopped(ready: Any, stopped: Any) -> None:
    ready.put({"pid": os.getpid(), "ready": True})
    while not stopped.wait(0.02):
        pass


def _exit_before_ready() -> None:
    raise RuntimeError("intentional startup failure")


def _ignore_shutdown(ready: Any) -> None:
    ready.put({"pid": os.getpid(), "ready": True})
    while True:
        time.sleep(0.02)


def _write_isolated_lifecycle_record(path: str, index: int) -> None:
    _write_managed_process_lifecycle_record(
        path,
        {
            "schema_version": 1,
            "cluster_id": f"concurrent-{index}",
            "graceful_shutdown_count": 1,
            "unexpected_terminate_count": 0,
            "unexpected_kill_count": 0,
            "leaked_process_count": 0,
        },
    )


@pytest.mark.asyncio
async def test_product_cluster_joins_process_and_closes_queue() -> None:
    baseline = {process.pid for process in multiprocessing.active_children()}
    cluster = ProductCluster()
    ready = cluster.queue()
    stopped = cluster.event()
    process = cluster.process(
        "worker",
        target=_ready_until_stopped,
        args=(ready, stopped),
        shutdown=stopped.set,
    )
    cluster.start()
    payloads = await cluster.wait_ready(ready, count=1, timeout=10)
    child_pid = int(payloads[0]["pid"])
    assert process.pid == child_pid
    assert process.is_alive()

    await cluster.close()
    await cluster.close()
    assert child_pid not in {
        child.pid for child in multiprocessing.active_children() if child.pid not in baseline
    }
    assert cluster.live_children() == []
    assert cluster.lifecycle_counts() == {
        "graceful_shutdown_count": 1,
        "unexpected_terminate_count": 0,
        "unexpected_kill_count": 0,
        "expected_terminate_count": 0,
        "expected_kill_count": 0,
        "leaked_process_count": 0,
    }


@pytest.mark.asyncio
async def test_product_cluster_cleans_up_after_test_assertion_failure() -> None:
    cluster = ProductCluster()
    ready = cluster.queue()
    stopped = cluster.event()
    cluster.process(
        "worker-after-failure",
        target=_ready_until_stopped,
        args=(ready, stopped),
        shutdown=stopped.set,
    )
    try:
        cluster.start()
        payloads = await cluster.wait_ready(ready, count=1, timeout=10)
        assert payloads[0]["ready"]
        with pytest.raises(AssertionError, match="intentional assertion"):
            raise AssertionError("intentional assertion")
    finally:
        await cluster.close()
    assert cluster.live_children() == []


@pytest.mark.asyncio
async def test_product_cluster_propagates_startup_failure_and_still_closes() -> None:
    cluster = ProductCluster()
    ready = cluster.queue()
    cluster.process("failed-worker", target=_exit_before_ready)
    try:
        cluster.start()
        with pytest.raises(ChildStartupError, match="exited before readiness"):
            await cluster.wait_ready(ready, count=1, timeout=10)
    finally:
        await cluster.close()
    assert cluster.live_children() == []


@pytest.mark.asyncio
async def test_managed_queue_feeder_thread_is_joined() -> None:
    cluster = ProductCluster()
    managed_queue = cluster.queue()
    managed_queue.put("value")
    assert await asyncio.to_thread(managed_queue.get, True, 1) == "value"
    feeder = managed_queue._thread
    assert feeder is not None and feeder.is_alive()
    await cluster.close()
    assert not feeder.is_alive()
    with pytest.raises((ValueError, OSError, queue.Empty)):
        managed_queue.get_nowait()


@pytest.mark.asyncio
async def test_unexpected_force_termination_fails_cleanup_and_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This negative test deliberately exercises the failing fallback.  Keep
    # its expected local failure out of an enclosing acceptance run's normal
    # process-lifecycle aggregate.
    monkeypatch.delenv("SWARM_PROCESS_LIFECYCLE_LOG", raising=False)
    cluster = ProductCluster()
    ready = cluster.queue()
    cluster.process("stubborn-worker", target=_ignore_shutdown, args=(ready,))
    cluster.processes["stubborn-worker"].graceful_timeout_s = 0.05
    cluster.processes["stubborn-worker"].terminate_timeout_s = 1
    cluster.start()
    await cluster.wait_ready(ready, count=1, timeout=10)

    with pytest.raises(ProcessCleanupError, match="force cleanup was required"):
        await cluster.close()

    counts = cluster.lifecycle_counts()
    assert counts["unexpected_terminate_count"] == 1
    assert counts["unexpected_kill_count"] == 0
    assert counts["leaked_process_count"] == 0


@pytest.mark.asyncio
async def test_expected_crash_is_distinct_from_cleanup_failure() -> None:
    cluster = ProductCluster()
    ready = cluster.queue()
    stopped = cluster.event()
    cluster.process(
        "fault-target",
        target=_ready_until_stopped,
        args=(ready, stopped),
        shutdown=stopped.set,
    )
    cluster.start()
    await cluster.wait_ready(ready, count=1, timeout=10)

    await asyncio.to_thread(
        cluster.crash_process,
        "fault-target",
        reason="unit-test crash injection",
        timeout=2,
    )
    await cluster.close()

    counts = cluster.lifecycle_counts()
    assert counts["expected_terminate_count"] == 1
    assert counts["unexpected_terminate_count"] == 0
    assert counts["unexpected_kill_count"] == 0
    assert counts["leaked_process_count"] == 0


def test_concurrent_process_lifecycle_records_have_single_owners(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    base = tmp_path / "lifecycle.jsonl"
    processes = [
        context.Process(
            target=_write_isolated_lifecycle_record,
            args=(str(base), index),
        )
        for index in range(8)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
            process.close()

    records = sorted(tmp_path.glob("lifecycle.jsonl.worker-*.json"))
    assert len(records) == len(processes)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in records]
    assert {payload["cluster_id"] for payload in payloads} == {
        f"concurrent-{index}" for index in range(8)
    }


def test_representative_process_test_has_no_resource_warning(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/test_process_harness.py::test_product_cluster_joins_process_and_closes_queue",
        "-q",
        "--basetemp",
        str(tmp_path / "pytest-temp"),
    ]
    environment = dict(os.environ)
    environment["TEMP"] = str(tmp_path)
    environment["TMP"] = str(tmp_path)
    completed = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = f"{completed.stdout}\n{completed.stderr}".lower()
    warning_fragments = (
        "resource_tracker",
        "leaked semaphore",
        "leaked shared_memory",
        "task was destroyed but it is pending",
        "unclosed client session",
        "unclosed transport",
        "unclosed event loop",
    )
    assert completed.returncode == 0, combined
    assert [fragment for fragment in warning_fragments if fragment in combined] == []


def test_process_harness_is_repository_local_test_infrastructure() -> None:
    path = Path("src/swarm_inference/testing/process_harness.py")
    assert path.is_file()
    assert "os._exit" not in path.read_text(encoding="utf-8")
