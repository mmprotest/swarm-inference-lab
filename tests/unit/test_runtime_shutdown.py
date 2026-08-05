from __future__ import annotations

import asyncio

import pytest

from swarm_inference.runtime.shutdown import wait_for_service_shutdown


@pytest.mark.asyncio
async def test_shutdown_request_cancels_service_waiter() -> None:
    stop_event = asyncio.Event()
    waiter_cancelled = asyncio.Event()

    async def wait_forever() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            waiter_cancelled.set()

    task = asyncio.create_task(wait_for_service_shutdown(wait_forever(), stop_event))
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert waiter_cancelled.is_set()


@pytest.mark.asyncio
async def test_shutdown_callback_releases_service_waiter_before_cleanup() -> None:
    stop_event = asyncio.Event()
    shutdown_finished = asyncio.Event()
    termination_finished = asyncio.Event()

    async def wait_for_shutdown() -> None:
        await shutdown_finished.wait()
        termination_finished.set()

    async def shutdown() -> None:
        assert not termination_finished.is_set()
        shutdown_finished.set()

    task = asyncio.create_task(
        wait_for_service_shutdown(wait_for_shutdown(), stop_event, shutdown=shutdown)
    )
    await asyncio.sleep(0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert shutdown_finished.is_set()
    assert termination_finished.is_set()


@pytest.mark.asyncio
async def test_service_termination_error_propagates_and_cancels_stop_waiter() -> None:
    stop_event = asyncio.Event()

    async def fail() -> None:
        raise RuntimeError("server failed")

    with pytest.raises(RuntimeError, match="server failed"):
        await wait_for_service_shutdown(fail(), stop_event)

    current = asyncio.current_task()
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not current and task.get_name() == "service-shutdown-request"
    ]
    assert pending == []
