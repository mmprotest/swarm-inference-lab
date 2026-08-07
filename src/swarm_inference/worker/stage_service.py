"""Composition root for one control server and one persistent stage data server."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING

from swarm_inference.security.tls import TlsServerConfig
from swarm_inference.transport.grpc_transport import WorkerRpcServer
from swarm_inference.transport.stage_ring_server import StageRingServer
from swarm_inference.worker.agent import WorkerAgent

if TYPE_CHECKING:
    from swarm_inference.cluster.artifacts import ArtifactManager
    from swarm_inference.worker.engine_runtime import PersistentEngineRuntime
    from swarm_inference.worker.stage_runtime import PersistentStageRuntime


class PersistentStageWorkerService:
    """Extend the existing worker process without creating a second agent."""

    def __init__(
        self,
        *,
        agent: WorkerAgent,
        stage_runtime: PersistentStageRuntime | None,
        engine_runtime: PersistentEngineRuntime | None = None,
        artifact_manager: ArtifactManager | None = None,
        trusted_coordinator_fingerprint: str | None = None,
        model_shard_root: str | None = None,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        data_queue_capacity: int = 256,
        data_maximum_connections: int = 128,
        data_read_timeout_s: float = 30.0,
        data_write_timeout_s: float = 30.0,
        data_idle_timeout_s: float = 120.0,
        tls_server_config: TlsServerConfig | None = None,
    ) -> None:
        self.agent = agent
        self.stage_runtime = stage_runtime
        self.control_server = WorkerRpcServer(
            agent=agent,
            model_shard_root=model_shard_root,
            maximum_message_bytes=maximum_message_bytes,
            stage_runtime=stage_runtime,
            engine_runtime=engine_runtime,
            artifact_manager=artifact_manager,
            trusted_coordinator_fingerprint=trusted_coordinator_fingerprint,
            tls=tls_server_config,
        )
        self.data_server = (
            StageRingServer(
                handler=stage_runtime.handle_message,
                queue_capacity=data_queue_capacity,
                maximum_connections=data_maximum_connections,
                read_timeout_s=data_read_timeout_s,
                write_timeout_s=data_write_timeout_s,
                idle_timeout_s=data_idle_timeout_s,
                require_peer_authentication=lambda: stage_runtime.require_authenticated_peers,
                tls=tls_server_config,
            )
            if stage_runtime is not None
            else None
        )
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    def configure_artifact_trust(
        self,
        *,
        coordinator_public_key: str,
        coordinator_fingerprint: str,
    ) -> None:
        self.control_server.configure_artifact_trust(
            coordinator_public_key=coordinator_public_key,
            coordinator_fingerprint=coordinator_fingerprint,
        )

    def configure_engine_trust(
        self,
        *,
        coordinator_public_key: str,
        expected_fingerprint: str | None,
    ) -> None:
        self.control_server.configure_engine_trust(
            coordinator_public_key=coordinator_public_key,
            expected_fingerprint=expected_fingerprint,
        )

    async def start(
        self,
        *,
        control_listen_endpoint: str,
        data_listen_endpoint: str | None,
    ) -> tuple[int, int | None]:
        async with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("persistent worker service is already started")
            if (self.data_server is None) != (data_listen_endpoint is None):
                raise ValueError(
                    "data listen endpoint must be supplied exactly when stage runtime is enabled"
                )
            self._stopping = False
            data_port = None
            try:
                if self.stage_runtime is not None:
                    await self.stage_runtime.start()
                if self.data_server is not None and data_listen_endpoint is not None:
                    data_port = await self.data_server.start(data_listen_endpoint)
                control_port = await self.control_server.start(control_listen_endpoint)
            except BaseException:
                self._stopping = True
                if self.data_server is not None:
                    await self.data_server.stop()
                if self.stage_runtime is not None:
                    await self.stage_runtime.close()
                with suppress(Exception):
                    await self.agent.stop()
                raise
            self._started = True
            return control_port, data_port

    async def wait_for_termination(self) -> None:
        await self.control_server.wait_for_termination()

    async def stop(self, grace_s: float = 2.0) -> None:
        async with self._lifecycle_lock:
            if self._stopping and not self._started:
                return
            self._stopping = True
            if self.stage_runtime is not None:
                self.stage_runtime.begin_draining()
            try:
                if self.data_server is not None:
                    await self.data_server.stop()
            finally:
                try:
                    if self._started:
                        await self.control_server.stop(grace_s)
                    else:
                        if self.stage_runtime is not None:
                            await self.stage_runtime.close()
                        await self.agent.stop()
                finally:
                    self._started = False


__all__ = ["PersistentStageWorkerService"]
