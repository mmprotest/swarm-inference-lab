from __future__ import annotations

from pathlib import Path

import pytest

from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.runtime import CoordinatorRuntime
from swarm_inference.coordinator.service import CoordinatorCore


class _FakeServer:
    def __init__(self, core: CoordinatorCore, *, fail_start: bool = False) -> None:
        self.core = core
        self.fail_start = fail_start
        self.bound_port: int | None = None
        self.start_count = 0
        self.stop_count = 0
        self.terminated = False

    async def start(self, endpoint: str, *, advertised_endpoint: str | None = None) -> int:
        self.start_count += 1
        if self.fail_start:
            raise RuntimeError("injected coordinator startup failure")
        self.bound_port = 43123
        self.core.publication_endpoint = advertised_endpoint or "127.0.0.1:43123"
        return self.bound_port

    async def stop(self, grace_s: float = 2.0) -> None:
        del grace_s
        self.stop_count += 1
        self.bound_port = None
        self.terminated = True
        await self.core.close()

    async def wait_for_termination(self) -> None:
        while not self.terminated:
            import asyncio

            await asyncio.sleep(0)


def _core(path: Path) -> CoordinatorCore:
    return CoordinatorCore(
        product_config=ProductCoordinatorConfig(require_trusted_workers=False),
        state_directory=path,
    )


@pytest.mark.asyncio
async def test_coordinator_runtime_start_and_stop_are_idempotent(tmp_path: Path) -> None:
    server: _FakeServer | None = None

    def factory(core: CoordinatorCore) -> _FakeServer:
        nonlocal server
        server = _FakeServer(core)
        return server

    runtime = CoordinatorRuntime(
        core=_core(tmp_path / "coordinator"),
        listen_endpoint="127.0.0.1:0",
        advertised_endpoint="127.0.0.1:43123",
        server_factory=factory,
    )
    first = await runtime.start()
    second = await runtime.start()

    assert first == second
    assert first.state == "running"
    assert first.listen_endpoint == "127.0.0.1:43123"
    assert first.identity_fingerprint
    assert server is not None
    assert server.start_count == 1

    await runtime.stop()
    await runtime.stop()
    assert runtime.status.state == "stopped"
    assert server.stop_count == 1


@pytest.mark.asyncio
async def test_coordinator_runtime_rolls_back_partial_start(tmp_path: Path) -> None:
    server: _FakeServer | None = None

    def factory(core: CoordinatorCore) -> _FakeServer:
        nonlocal server
        server = _FakeServer(core, fail_start=True)
        return server

    runtime = CoordinatorRuntime(
        core=_core(tmp_path / "coordinator"),
        listen_endpoint="127.0.0.1:0",
        server_factory=factory,
    )
    with pytest.raises(RuntimeError, match="injected coordinator startup failure"):
        await runtime.start()

    assert runtime.status.state == "failed"
    assert runtime.status.last_error == "injected coordinator startup failure"
    assert server is not None
    assert server.stop_count == 1
