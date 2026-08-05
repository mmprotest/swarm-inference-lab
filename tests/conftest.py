from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _live_children_excluding(baseline: set[int | None]) -> list[multiprocessing.Process]:
    return [
        child
        for child in multiprocessing.active_children()
        if child.pid not in baseline and child.is_alive()
    ]


@pytest.fixture(autouse=True)
def process_test_child_guard(request: pytest.FixtureRequest) -> None:
    """Fail process/failure tests that return while a child they created is alive."""

    path = Path(str(request.node.path))
    guarded = path.parent.name in {"integration", "failure"}
    baseline = {child.pid for child in multiprocessing.active_children()} if guarded else set()
    yield
    if not guarded:
        return
    deadline = time.monotonic() + 1.0
    leaked = _live_children_excluding(baseline)
    while leaked and time.monotonic() < deadline:
        for child in leaked:
            child.join(timeout=0.05)
        leaked = _live_children_excluding(baseline)
    if leaked:
        details = [
            {"pid": child.pid, "name": child.name, "exitcode": child.exitcode} for child in leaked
        ]
        for child in leaked:
            child.terminate()
            child.join(timeout=2)
        pytest.fail(f"process test leaked managed child processes: {details}")


@pytest.fixture(scope="session", autouse=True)
def process_suite_resource_guard() -> None:
    """Report suite-wide child and multiprocessing queue-thread leaks."""

    baseline_children = {child.pid for child in multiprocessing.active_children()}
    baseline_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "QueueFeederThread" and thread.is_alive()
    }
    yield
    leaked_children = _live_children_excluding(baseline_children)
    leaked_threads = [
        {"name": thread.name, "ident": thread.ident}
        for thread in threading.enumerate()
        if thread.name == "QueueFeederThread"
        and thread.is_alive()
        and thread.ident not in baseline_threads
    ]
    if leaked_children or leaked_threads:
        child_details = [
            {"pid": child.pid, "name": child.name, "exitcode": child.exitcode}
            for child in leaked_children
        ]
        for child in leaked_children:
            child.terminate()
            child.join(timeout=2)
        pytest.fail(
            "process suite leaked resources: "
            f"children={child_details}, queue_threads={leaked_threads}"
        )
