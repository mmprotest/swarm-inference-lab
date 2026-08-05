from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.testing.process_harness import ChildStartupError, ProductCluster


def _ready_until_stopped(ready: Any, stopped: Any) -> None:
    ready.put({"pid": os.getpid(), "ready": True})
    while not stopped.wait(0.02):
        pass


def _exit_before_ready() -> None:
    raise RuntimeError("intentional startup failure")


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


def test_process_harness_is_repository_local_test_infrastructure() -> None:
    path = Path("src/swarm_inference/testing/process_harness.py")
    assert path.is_file()
    assert "os._exit" not in path.read_text(encoding="utf-8")
