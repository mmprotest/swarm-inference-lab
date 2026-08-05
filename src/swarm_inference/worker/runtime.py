"""Reusable lifecycle owner for the canonical persistent worker runtime."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, PositiveInt

from swarm_inference.config.models import (
    Backend,
    QueueConfig,
    StrictModel,
    WorkerCapability,
    WorkerRole,
)
from swarm_inference.security.identity import public_key_fingerprint

if TYPE_CHECKING:
    from swarm_inference.cluster.artifacts import ArtifactManager

WorkerRuntimeState = Literal[
    "starting",
    "running",
    "stopping",
    "stopped",
    "failed",
]


class WorkerRuntimeConfig(StrictModel):
    """Validated configuration consumed by both the CLI and node agent."""

    schema_version: Literal["1"] = "1"
    coordinator_endpoint: str
    listen_endpoint: str
    advertised_endpoint: str
    backend: Backend
    memory_limit_bytes: PositiveInt
    identity_path: Path
    total_memory_limit_bytes: PositiveInt | None = None
    worker_id: str | None = None
    model_shard_root: Path | None = None
    queue_config: QueueConfig = Field(default_factory=QueueConfig)
    outbound_queue_capacity: PositiveInt = 1024
    inbound_queue_capacity: PositiveInt = 1024
    max_inflight_operations: PositiveInt = 256
    reconnect_attempts: int = Field(default=5, ge=0, le=100)
    reconnect_initial_backoff_ms: float = Field(default=25.0, gt=0)
    reconnect_max_backoff_ms: float = Field(default=1000.0, gt=0)
    stage_runtime_enabled: bool = False
    data_listen_endpoint: str | None = None
    data_advertised_endpoint: str | None = None
    device: str | None = None
    dtype: str = "bfloat16"
    model_cache_dir: Path | None = None
    artifact_storage_limit_bytes: PositiveInt | None = None
    configured_model_path: Path | None = None
    allow_model_download: bool = False
    max_stage_sessions: PositiveInt = 256
    stage_execution_queue_capacity: PositiveInt = 256
    token_publication_queue_capacity: PositiveInt = 256
    upload_bandwidth_bytes_s: float = Field(default=0.0, ge=0)
    download_bandwidth_bytes_s: float = Field(default=0.0, ge=0)
    network_rates_measured: bool = False
    trusted_coordinator_fingerprint: str | None = None
    worker_roles: set[WorkerRole] = Field(default_factory=set)
    expert_manifest_path: Path | None = None
    expert_data_listen_endpoint: str | None = None
    expert_data_advertised_endpoint: str | None = None
    expert_residency_budget_bytes: int = Field(default=0, ge=0)
    expert_cache_budget_bytes: int = Field(default=0, ge=0)
    expert_queue_capacity: PositiveInt = 64
    expert_max_concurrent_requests: PositiveInt = 1
    service_mode: str = "foreground"
    platform_support_status: str = "unknown"


class WorkerRuntimeStatus(StrictModel):
    """Public, non-secret worker lifecycle status."""

    schema_version: Literal["1"] = "1"
    state: WorkerRuntimeState = "stopped"
    worker_id: str | None = None
    identity_fingerprint: str | None = None
    coordinator_endpoint: str
    control_endpoint: str
    data_endpoint: str | None = None
    backend: Backend
    device: str | None = None
    dtype: str
    process_id: int = os.getpid()
    service_mode: str = "foreground"
    last_error: str | None = None


WorkerRunner = Callable[..., Awaitable[None]]


class WorkerRuntime:
    """Start, wait for, and stop the existing persistent worker implementation."""

    def __init__(
        self,
        *,
        config: WorkerRuntimeConfig,
        startup_timeout_s: float = 120.0,
        shutdown_timeout_s: float = 15.0,
        runner: WorkerRunner | None = None,
        artifact_manager: ArtifactManager | None = None,
    ) -> None:
        if startup_timeout_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError("worker runtime timeouts must be positive")
        self.config = config
        self.startup_timeout_s = startup_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        if runner is None:
            # Resolve lazily so tests and embedding applications can replace the
            # canonical entrypoint before constructing the lifecycle owner.
            from swarm_inference.worker.service import run_worker

            runner = run_worker
        self._runner = runner
        self.artifact_manager = artifact_manager
        self._lifecycle_lock = asyncio.Lock()
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._startup_future: asyncio.Future[WorkerCapability] | None = None
        self._status = WorkerRuntimeStatus(
            coordinator_endpoint=config.coordinator_endpoint,
            control_endpoint=config.advertised_endpoint,
            data_endpoint=config.data_advertised_endpoint,
            backend=config.backend,
            device=config.device,
            dtype=config.dtype,
            service_mode=config.service_mode,
        )

    @property
    def status(self) -> WorkerRuntimeStatus:
        return self._status.model_copy(deep=True)

    def _runner_arguments(
        self,
        *,
        stop_event: asyncio.Event,
        startup_future: asyncio.Future[WorkerCapability],
    ) -> dict[str, Any]:
        config = self.config
        return {
            "coordinator_endpoint": config.coordinator_endpoint,
            "listen_endpoint": config.listen_endpoint,
            "advertised_endpoint": config.advertised_endpoint,
            "backend": config.backend,
            "memory_limit_bytes": config.memory_limit_bytes,
            "identity_path": config.identity_path,
            "total_memory_limit_bytes": config.total_memory_limit_bytes,
            "worker_id": config.worker_id,
            "model_shard_root": config.model_shard_root,
            "queue_config": config.queue_config,
            "stop_event": stop_event,
            "outbound_queue_capacity": config.outbound_queue_capacity,
            "inbound_queue_capacity": config.inbound_queue_capacity,
            "max_inflight_operations": config.max_inflight_operations,
            "reconnect_attempts": config.reconnect_attempts,
            "reconnect_initial_backoff_ms": config.reconnect_initial_backoff_ms,
            "reconnect_max_backoff_ms": config.reconnect_max_backoff_ms,
            "stage_runtime_enabled": config.stage_runtime_enabled,
            "data_listen_endpoint": config.data_listen_endpoint,
            "data_advertised_endpoint": config.data_advertised_endpoint,
            "device": config.device,
            "dtype": config.dtype,
            "model_cache_dir": config.model_cache_dir,
            "artifact_storage_limit_bytes": config.artifact_storage_limit_bytes,
            "artifact_manager": self.artifact_manager,
            "configured_model_path": config.configured_model_path,
            "allow_model_download": config.allow_model_download,
            "max_stage_sessions": config.max_stage_sessions,
            "stage_execution_queue_capacity": config.stage_execution_queue_capacity,
            "token_publication_queue_capacity": config.token_publication_queue_capacity,
            "upload_bandwidth_bytes_s": config.upload_bandwidth_bytes_s,
            "download_bandwidth_bytes_s": config.download_bandwidth_bytes_s,
            "network_rates_measured": config.network_rates_measured,
            "trusted_coordinator_fingerprint": config.trusted_coordinator_fingerprint,
            "worker_roles": config.worker_roles,
            "expert_manifest_path": config.expert_manifest_path,
            "expert_data_listen_endpoint": config.expert_data_listen_endpoint,
            "expert_data_advertised_endpoint": config.expert_data_advertised_endpoint,
            "expert_residency_budget_bytes": config.expert_residency_budget_bytes,
            "expert_cache_budget_bytes": config.expert_cache_budget_bytes,
            "expert_queue_capacity": config.expert_queue_capacity,
            "expert_max_concurrent_requests": config.expert_max_concurrent_requests,
            "startup_future": startup_future,
            "service_mode": config.service_mode,
            "platform_support_status": config.platform_support_status,
        }

    async def _run(
        self,
        stop_event: asyncio.Event,
        startup_future: asyncio.Future[WorkerCapability],
    ) -> None:
        try:
            await self._runner(
                **self._runner_arguments(
                    stop_event=stop_event,
                    startup_future=startup_future,
                )
            )
        except asyncio.CancelledError:
            if not startup_future.done():
                startup_future.cancel()
            raise
        except BaseException as exc:
            if not startup_future.done():
                startup_future.set_exception(exc)
            self._status = self._status.model_copy(
                update={"state": "failed", "last_error": str(exc)}
            )
            raise
        finally:
            if self._status.state in {"stopping", "running"}:
                self._status = self._status.model_copy(update={"state": "stopped"})

    async def _abort_start_locked(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=self.shutdown_timeout_s)
        except TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        except BaseException:
            # The startup failure is reported by ``start`` from the startup
            # future/task.  Cleanup exceptions must not replace it.
            pass

    async def start(self) -> WorkerRuntimeStatus:
        """Return only after registration and route trust configuration succeed."""

        async with self._lifecycle_lock:
            if self._task is not None and not self._task.done() and self._status.state == "running":
                return self.status
            self._status = self._status.model_copy(update={"state": "starting", "last_error": None})
            self._stop_event = asyncio.Event()
            self._startup_future = asyncio.get_running_loop().create_future()
            self._task = asyncio.create_task(
                self._run(self._stop_event, self._startup_future),
                name=f"worker-runtime:{self.config.worker_id or 'pending'}",
            )
            waiters: set[asyncio.Future[Any]] = {
                self._startup_future,
                self._task,
            }
            done, _ = await asyncio.wait(
                waiters,
                timeout=self.startup_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                error = TimeoutError(
                    f"worker startup exceeded {self.startup_timeout_s:.3f} seconds"
                )
                self._status = self._status.model_copy(
                    update={"state": "failed", "last_error": str(error)}
                )
                await self._abort_start_locked()
                raise error
            if self._startup_future not in done:
                assert self._task in done
                await self._task
                # A dependency-injected runner may be a finite lifecycle probe.
                # The canonical runner always signals registration readiness and
                # remains active until stopped.
                self._status = self._status.model_copy(update={"state": "stopped"})
                return self.status
            try:
                capability = self._startup_future.result()
            except BaseException:
                await self._abort_start_locked()
                raise
            if self._task.done():
                await self._task
                raise RuntimeError("worker terminated while completing startup")
            self._status = self._status.model_copy(
                update={
                    "state": "running",
                    "worker_id": capability.worker_id,
                    "identity_fingerprint": public_key_fingerprint(capability.public_key),
                    "control_endpoint": capability.control_endpoint
                    or capability.endpoint
                    or self.config.advertised_endpoint,
                    "data_endpoint": capability.data_plane_endpoint,
                    "backend": capability.backend,
                    "device": capability.device_identifier,
                    "last_error": None,
                }
            )
            return self.status

    async def wait(self) -> None:
        task = self._task
        if task is None:
            if self._status.state == "failed":
                raise RuntimeError(self._status.last_error or "worker runtime failed")
            return
        await asyncio.shield(task)

    async def stop(self) -> None:
        """Request canonical shutdown and enforce an upper bound."""

        async with self._lifecycle_lock:
            task = self._task
            if task is None or task.done():
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                if self._status.state != "failed":
                    self._status = self._status.model_copy(update={"state": "stopped"})
                return
            self._status = self._status.model_copy(update={"state": "stopping"})
            assert self._stop_event is not None
            self._stop_event.set()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self.shutdown_timeout_s)
            except TimeoutError as exc:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._status = self._status.model_copy(
                    update={"state": "failed", "last_error": str(exc)}
                )
                raise
            except BaseException as exc:
                self._status = self._status.model_copy(
                    update={"state": "failed", "last_error": str(exc)}
                )
                raise
            self._status = self._status.model_copy(update={"state": "stopped"})


__all__ = ["WorkerRuntime", "WorkerRuntimeConfig", "WorkerRuntimeStatus"]
