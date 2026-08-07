"""Reusable lifecycle owner for the canonical coordinator runtime.

The CLI, node agent, and tests all start the same :class:`CoordinatorCore` and
``CoordinatorRpcServer`` through this class.  It deliberately contains no
inference, planning, deployment, or transport implementation.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import yaml
from pydantic import NonNegativeInt

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.config.models import ModelManifest, StrictModel
from swarm_inference.config.product import load_product_config
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.host import format_endpoint, split_endpoint
from swarm_inference.security.tls import TlsServerConfig

CoordinatorRuntimeState = Literal[
    "starting",
    "running",
    "stopping",
    "stopped",
    "failed",
]


class CoordinatorRuntimeStatus(StrictModel):
    """Public, non-secret coordinator lifecycle status."""

    schema_version: Literal["1"] = "1"
    state: CoordinatorRuntimeState = "stopped"
    listen_endpoint: str
    advertised_endpoint: str | None = None
    bound_port: NonNegativeInt | None = None
    coordinator_id: str | None = None
    identity_fingerprint: str | None = None
    state_directory: str
    process_id: int = os.getpid()
    service_mode: str = "foreground"
    last_error: str | None = None


class _CoordinatorServer(Protocol):
    bound_port: int | None

    async def start(self, endpoint: str, *, advertised_endpoint: str | None = None) -> int: ...

    async def stop(self, grace_s: float = 2.0) -> None: ...

    async def wait_for_termination(self) -> None: ...


ServerFactory = Callable[[CoordinatorCore], _CoordinatorServer]


class CoordinatorRuntime:
    """Idempotent, bounded lifecycle for the canonical coordinator service."""

    def __init__(
        self,
        *,
        core: CoordinatorCore,
        listen_endpoint: str,
        advertised_endpoint: str | None = None,
        service_mode: str = "foreground",
        shutdown_timeout_s: float = 12.0,
        server_factory: ServerFactory | None = None,
        tls_server_config: TlsServerConfig | None = None,
    ) -> None:
        if shutdown_timeout_s <= 0:
            raise ValueError("coordinator runtime shutdown timeout must be positive")
        self.core = core
        self.listen_endpoint = listen_endpoint
        self.advertised_endpoint = advertised_endpoint
        self.shutdown_timeout_s = shutdown_timeout_s
        self._server = (
            server_factory(core)
            if server_factory is not None
            else CoordinatorRpcServer(core, tls=tls_server_config)
        )
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        coordinator_id = core.product_config.coordinator_id if core.product_config else None
        identity_fingerprint = (
            core.coordinator_identity.public_key_fingerprint
            if core.coordinator_identity is not None
            else None
        )
        self._status = CoordinatorRuntimeStatus(
            listen_endpoint=listen_endpoint,
            advertised_endpoint=advertised_endpoint,
            coordinator_id=coordinator_id,
            identity_fingerprint=identity_fingerprint,
            state_directory=str(core.state_directory),
            service_mode=service_mode,
        )

    @classmethod
    def from_config_path(
        cls,
        *,
        config_path: Path,
        listen_endpoint: str = "0.0.0.0:50051",
        advertised_endpoint: str | None = None,
        state_directory: Path = Path(".swarm/coordinator"),
        model_manifest_path: Path | None = None,
        model_path: Path | None = None,
        dtype: str | None = None,
        service_mode: str = "foreground",
    ) -> Self:
        """Build the existing coordinator core from the low-level CLI inputs."""

        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        product_configuration = None
        experiment = None
        if isinstance(raw_config, dict) and raw_config.get("kind") == "product-stage-ring":
            product_configuration = load_product_config(config_path)
        else:
            experiment = load_experiment_config(config_path)
        if product_configuration is not None and (
            model_manifest_path is not None or model_path is not None or dtype is not None
        ):
            raise ValueError(
                "product coordinator metadata is discovered from workers; do not pass legacy "
                "model-manifest, model-path, or dtype options"
            )
        if (model_manifest_path is None) != (model_path is None):
            raise ValueError("--model-manifest and --model-path must be supplied together")
        manifest = None
        architecture_config: dict[str, Any] | None = None
        tokenizer: Any | None = None
        if model_manifest_path is not None and model_path is not None:
            manifest = ModelManifest.model_validate_json(
                model_manifest_path.read_text(encoding="utf-8")
            )
            architecture_config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8")
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_path,
                local_files_only=True,
            )
        core = CoordinatorCore(
            config=experiment,
            product_config=product_configuration,
            state_directory=state_directory,
            model_manifest=manifest,
            architecture_config=architecture_config,
            runtime_dtype=dtype,
            tokenizer=tokenizer,
        )
        return cls(
            core=core,
            listen_endpoint=listen_endpoint,
            advertised_endpoint=advertised_endpoint,
            service_mode=service_mode,
        )

    @property
    def status(self) -> CoordinatorRuntimeStatus:
        return self._status.model_copy(deep=True)

    async def start(self) -> CoordinatorRuntimeStatus:
        """Start once; repeated calls while running return the same public status."""

        async with self._lifecycle_lock:
            if self._started:
                return self.status
            self._status = self._status.model_copy(update={"state": "starting", "last_error": None})
            try:
                bound_port = await self._server.start(
                    self.listen_endpoint,
                    advertised_endpoint=self.advertised_endpoint,
                )
                listen_host, requested_port = split_endpoint(self.listen_endpoint)
                effective_listen = format_endpoint(
                    listen_host,
                    bound_port if requested_port == 0 else requested_port,
                )
                published = self.core.publication_endpoint or self.advertised_endpoint
                self._started = True
                self._status = self._status.model_copy(
                    update={
                        "state": "running",
                        "listen_endpoint": effective_listen,
                        "advertised_endpoint": published,
                        "bound_port": bound_port,
                    }
                )
                return self.status
            except BaseException as exc:
                self._status = self._status.model_copy(
                    update={"state": "failed", "last_error": str(exc)}
                )
                # Preserve the startup error.  The canonical server attempts to
                # close its core even when the gRPC bind/start only partly worked.
                with suppress(BaseException):
                    await asyncio.wait_for(
                        self._server.stop(grace_s=0.0),
                        timeout=self.shutdown_timeout_s,
                    )
                raise

    async def wait(self) -> None:
        """Wait for service termination without installing process signal handlers."""

        if not self._started:
            if self._status.state == "failed":
                raise RuntimeError(self._status.last_error or "coordinator runtime failed")
            return
        await self._server.wait_for_termination()

    async def stop(self) -> None:
        """Stop all started resources once, with a hard upper bound."""

        async with self._lifecycle_lock:
            if not self._started:
                if self._status.state != "failed":
                    self._status = self._status.model_copy(update={"state": "stopped"})
                return
            self._status = self._status.model_copy(update={"state": "stopping"})
            try:
                await asyncio.wait_for(
                    self._server.stop(),
                    timeout=self.shutdown_timeout_s,
                )
            except BaseException as exc:
                self._status = self._status.model_copy(
                    update={"state": "failed", "last_error": str(exc)}
                )
                self._started = False
                raise
            self._started = False
            self._status = self._status.model_copy(update={"state": "stopped", "bound_port": None})


__all__ = ["CoordinatorRuntime", "CoordinatorRuntimeStatus"]
