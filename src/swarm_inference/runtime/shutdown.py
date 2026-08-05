"""Signal-aware, bounded service shutdown helpers."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable
from contextlib import suppress


def install_shutdown_signal_handlers(
    stop_event: asyncio.Event,
) -> Callable[[], None]:
    """Request shutdown on SIGINT/SIGTERM and return an idempotent restore hook."""

    loop = asyncio.get_running_loop()
    signals = tuple(
        candidate
        for candidate in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
        if candidate is not None
    )
    loop_handlers: list[signal.Signals] = []
    synchronous_handlers: dict[signal.Signals, signal._HANDLER] = {}

    def request_shutdown(*_: object) -> None:
        loop.call_soon_threadsafe(stop_event.set)

    for signum in signals:
        try:
            loop.add_signal_handler(signum, request_shutdown)
            loop_handlers.append(signum)
            continue
        except (NotImplementedError, RuntimeError):
            pass
        try:
            synchronous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
        except (OSError, RuntimeError, ValueError):
            # Signal registration is restricted outside the main thread on some
            # platforms. The caller can still stop through its explicit event.
            continue

    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        for signum in loop_handlers:
            with suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signum)
        for signum, previous in synchronous_handlers.items():
            with suppress(OSError, RuntimeError, ValueError):
                signal.signal(signum, previous)

    return restore


async def wait_for_service_shutdown(
    termination: Awaitable[None],
    stop_event: asyncio.Event,
    *,
    shutdown: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Return on an explicit stop request or propagate service termination errors."""

    async def await_termination() -> None:
        await termination

    termination_task = asyncio.create_task(await_termination(), name="service-termination")
    stop_task = asyncio.create_task(stop_event.wait(), name="service-shutdown-request")
    try:
        done, _ = await asyncio.wait(
            {termination_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if termination_task in done:
            await termination_task
        elif shutdown is not None:
            # Stop the service before cancelling its termination waiter.  Some
            # AsyncIO servers share internal futures with wait_for_termination;
            # cancelling that waiter first can poison the subsequent graceful
            # stop operation.
            await shutdown()
            await termination_task
    finally:
        for task in (termination_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(termination_task, stop_task, return_exceptions=True)


__all__ = ["install_shutdown_signal_handlers", "wait_for_service_shutdown"]
