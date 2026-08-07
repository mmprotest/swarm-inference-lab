"""Construct worker-owned engine runtimes from installer manifests."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from swarm_inference.backends.colibri.backend import ColibriBackend
from swarm_inference.backends.colibri.runtime_manifest import (
    load_colibri_runtime_manifest,
)
from swarm_inference.cluster.artifacts import ArtifactManager
from swarm_inference.engines.colibri import ColibriExecutionEngine, LocalColibriLifecycle
from swarm_inference.engines.installed import discover_installed_engine_manifests
from swarm_inference.engines.interfaces import ExecutionPlan
from swarm_inference.engines.llamacpp_rpc import (
    LlamaCppRpcEngine,
    LocalLlamaCppLifecycle,
    load_llamacpp_runtime_manifest,
)
from swarm_inference.host import split_endpoint
from swarm_inference.runtime.engine_processes import EngineProcessManager
from swarm_inference.worker.engine_runtime import PersistentEngineRuntime


def _colibri_factory(manifest_path: Path) -> Callable[[ExecutionPlan], ColibriBackend]:
    manifest = load_colibri_runtime_manifest(manifest_path)
    engine_directory = manifest.engine_directory
    if engine_directory is None or not engine_directory.is_dir():
        raise FileNotFoundError("Colibri runtime manifest engine_directory is unavailable")
    log_root = manifest_path.parent / "logs"

    def build(plan: ExecutionPlan) -> ColibriBackend:
        model_paths = [
            Path(item).expanduser().resolve()
            for item in plan.engine_parameters.get("model_paths", [])
        ]
        if not model_paths or not all(item.exists() for item in model_paths):
            raise FileNotFoundError("selected Colibri model artifact is not resident")
        common = Path(os.path.commonpath([str(item) for item in model_paths])).resolve()
        model_path = common if common.is_dir() else common.parent
        routing_enabled = bool(plan.optional_mechanisms.get("routing_aware_placement"))
        requested_profile = plan.engine_parameters.get("routing_profile_id")
        if routing_enabled != (requested_profile is not None):
            raise ValueError("Colibri routing plan and profile identity do not reconcile")
        profile = None
        if requested_profile is not None:
            profile = manifest.routing_profile(str(requested_profile))
            if not profile.admitted:
                raise ValueError("Colibri routing profile lacks positive exactness evidence")
            if profile.model_fingerprint != plan.model_fingerprint:
                raise ValueError("Colibri routing profile belongs to a different model artifact")
            if profile.adapter_id != str(plan.engine_parameters.get("model_family") or ""):
                raise ValueError("Colibri routing profile belongs to a different adapter")
        return ColibriBackend(
            engine_directory=engine_directory,
            model_path=model_path,
            model_id=str(plan.engine_parameters["model_id"]),
            model_revision=str(plan.engine_parameters["model_revision"]),
            source_directory=manifest.source_directory,
            build_manifest=manifest.build_manifest,
            model_family=str(plan.engine_parameters.get("model_family") or "") or None,
            log_directory=log_root / plan.plan_id,
            environment=profile.environment if profile is not None else None,
            execution_profile_id=profile.profile_id if profile is not None else None,
            execution_profile_fingerprint=(
                profile.content_fingerprint if profile is not None else None
            ),
        )

    return build


def build_worker_engine_runtime(
    *,
    worker_id: str,
    advertised_endpoint: str,
    identity_directory: Path,
    artifact_manager: ArtifactManager | None,
    llamacpp_runtime_manifest: str | Path | None,
    colibri_runtime_manifest: str | Path | None,
) -> PersistentEngineRuntime | None:
    """Instantiate only hash-checked engines explicitly installed on this worker."""

    engines = []
    bind_host, _ = split_endpoint(advertised_endpoint)
    manifests = discover_installed_engine_manifests(
        llamacpp=(
            Path(llamacpp_runtime_manifest).expanduser().resolve()
            if llamacpp_runtime_manifest is not None
            else None
        ),
        colibri=(
            Path(colibri_runtime_manifest).expanduser().resolve()
            if colibri_runtime_manifest is not None
            else None
        ),
    )
    if manifests.llamacpp is not None:
        manifest = load_llamacpp_runtime_manifest(
            manifests.llamacpp
        )
        processes = EngineProcessManager(identity_directory / "engine-logs" / "llamacpp")
        engines.append(
            LlamaCppRpcEngine(
                lifecycle=LocalLlamaCppLifecycle(
                    manifest=manifest,
                    processes=processes,
                    worker_id=worker_id,
                    bind_host=bind_host,
                )
            )
        )
    if manifests.colibri is not None:
        manifest_path = manifests.colibri
        engines.append(
            ColibriExecutionEngine(
                lifecycle=LocalColibriLifecycle(_colibri_factory(manifest_path))
            )
        )
    if not engines:
        return None
    return PersistentEngineRuntime(
        worker_id=worker_id,
        engines=tuple(engines),
        artifact_resolver=(artifact_manager.resolve if artifact_manager is not None else None),
    )


__all__ = ["build_worker_engine_runtime"]
