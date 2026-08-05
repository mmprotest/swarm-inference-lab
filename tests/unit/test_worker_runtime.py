from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.worker.runtime import WorkerRuntime, WorkerRuntimeConfig


def _capability(identity: WorkerIdentity) -> WorkerCapability:
    return WorkerCapability(
        worker_id="node-abcd1234/cpu-0",
        public_key=identity.public_key_b64,
        hostname="test-host",
        operating_system="test",
        architecture="x86_64",
        backend=Backend.TORCH_CPU,
        cpu_model="test",
        logical_cpu_count=1,
        total_ram_bytes=1024**3,
        available_ram_bytes=1024**3,
        supported_dtypes=["float32"],
        upload_bandwidth_bytes_s=0,
        download_bandwidth_bytes_s=0,
        coordinator_latency_ms=0,
        memory_limit_bytes=512 * 1024**2,
        endpoint="127.0.0.1:51001",
        control_endpoint="127.0.0.1:51001",
        data_plane_endpoint="127.0.0.1:51002",
        device_identifier="cpu",
        last_heartbeat=datetime.now(UTC),
    )


def _config(path: Path) -> WorkerRuntimeConfig:
    return WorkerRuntimeConfig(
        coordinator_endpoint="127.0.0.1:50051",
        listen_endpoint="127.0.0.1:51001",
        advertised_endpoint="127.0.0.1:51001",
        backend=Backend.TORCH_CPU,
        memory_limit_bytes=512 * 1024**2,
        identity_path=path,
        worker_id="node-abcd1234/cpu-0",
        stage_runtime_enabled=True,
        data_listen_endpoint="127.0.0.1:51002",
        data_advertised_endpoint="127.0.0.1:51002",
        device="cpu",
        dtype="float32",
        trusted_coordinator_fingerprint="0" * 64,
    )


@pytest.mark.asyncio
async def test_worker_runtime_start_wait_stop_are_idempotent(tmp_path: Path) -> None:
    identity = WorkerIdentity.load_or_create(tmp_path / "worker.json")
    starts = 0
    stops = 0

    async def runner(**kwargs: Any) -> None:
        nonlocal starts, stops
        starts += 1
        startup_future = kwargs["startup_future"]
        startup_future.set_result(_capability(identity))
        await kwargs["stop_event"].wait()
        stops += 1

    runtime = WorkerRuntime(config=_config(tmp_path / "worker.json"), runner=runner)
    first = await runtime.start()
    second = await runtime.start()

    assert first == second
    assert first.state == "running"
    assert first.worker_id == "node-abcd1234/cpu-0"
    assert first.identity_fingerprint == identity.public_key_fingerprint
    assert starts == 1

    await runtime.stop()
    await runtime.stop()
    assert runtime.status.state == "stopped"
    assert stops == 1


@pytest.mark.asyncio
async def test_worker_runtime_reports_start_failure_and_rolls_back(tmp_path: Path) -> None:
    stop_observed = False

    async def runner(**kwargs: Any) -> None:
        nonlocal stop_observed
        try:
            raise RuntimeError("injected worker startup failure")
        finally:
            stop_observed = kwargs["stop_event"].is_set()

    runtime = WorkerRuntime(config=_config(tmp_path / "worker.json"), runner=runner)
    with pytest.raises(RuntimeError, match="injected worker startup failure"):
        await runtime.start()

    assert runtime.status.state == "failed"
    assert runtime.status.last_error == "injected worker startup failure"
    # The canonical runner completed its own rollback before propagating.
    assert not stop_observed
