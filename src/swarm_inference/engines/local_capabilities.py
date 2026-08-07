"""Read-only discovery of engine/device facts on the invoking host."""

from __future__ import annotations

import platform
import socket
from pathlib import Path

import psutil

from swarm_inference.backends.colibri.runtime_manifest import (
    load_colibri_runtime_manifest,
)
from swarm_inference.engines.installed import discover_installed_engine_manifests
from swarm_inference.engines.interfaces import (
    AdapterFastPathCapability,
    ClusterCapabilities,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionProfileCapability,
    WorkerExecutionCapability,
)
from swarm_inference.model.adapter import default_native_adapter_registry
from swarm_inference.model.architecture import GGUF_ARCHITECTURE_IDENTIFIERS


def _native_capability() -> ExecutionEngineCapability:
    adapters = default_native_adapter_registry().adapters()
    adapter_ids = tuple(sorted(item.adapter_id for item in adapters))
    fast_paths = tuple(
        sorted(
            {
                mode
                for adapter in adapters
                for path in adapter.fast_paths()
                for mode in getattr(path, "candidate_modes", (path.fast_path_id,))
            }
        )
    )
    adapter_fast_paths = tuple(
        AdapterFastPathCapability(
            adapter_id=adapter.adapter_id,
            fast_path_id=path.fast_path_id,
            candidate_modes=tuple(getattr(path, "candidate_modes", (path.fast_path_id,))),
        )
        for adapter in adapters
        for path in adapter.fast_paths()
    )
    memory = psutil.virtual_memory()
    device = ExecutionDevice(
        device_id="cpu",
        device_type="cpu",
        name=platform.processor() or platform.machine() or "CPU",
        total_memory_bytes=int(memory.total),
        usable_memory_bytes=int(memory.available),
        features=("eager",),
    )
    runtime_revision: str | None = None
    detail = "PyTorch is unavailable"
    enabled = False
    try:
        import torch

        enabled = True
        runtime_revision = str(torch.__version__)
        detail = "local PyTorch native-stage runtime"
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(index)
            properties = torch.cuda.get_device_properties(index)
            device = ExecutionDevice(
                device_id=f"cuda:{index}",
                device_type="cuda",
                name=str(properties.name),
                uuid=str(getattr(properties, "uuid", "")) or None,
                total_memory_bytes=int(total),
                usable_memory_bytes=int(free),
                runtime_version=str(torch.version.cuda or torch.__version__),
                driver_version=str(torch.version.cuda or "unavailable"),
                features=fast_paths,
            )
        elif bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available():
            device = ExecutionDevice(
                device_id="mps",
                device_type="mps",
                name="Apple Metal Performance Shaders",
                total_memory_bytes=int(memory.total),
                usable_memory_bytes=int(memory.available),
                runtime_version=str(torch.__version__),
                features=("eager",),
            )
    except (ImportError, OSError, RuntimeError) as exc:
        detail = f"PyTorch native runtime unavailable: {type(exc).__name__}: {exc}"
    return ExecutionEngineCapability(
        engine_id="native-stage",
        enabled=enabled,
        runtime_revision=runtime_revision,
        formats=("safetensors",),
        devices=(device,) if enabled else (),
        adapters=adapter_ids,
        fast_paths=fast_paths,
        adapter_fast_paths=adapter_fast_paths,
        roles=(
            "stage",
            "whole-expert",
            "microshard",
            "background",
            "verification",
            "storage",
            "idle",
        ),
        detail=detail,
    )


def _llamacpp_capability(path: Path | None) -> ExecutionEngineCapability:
    if path is None:
        return ExecutionEngineCapability(
            engine_id="llamacpp-rpc",
            enabled=False,
            formats=("gguf",),
            detail="no installer-owned llama.cpp runtime manifest is configured",
        )
    from swarm_inference.engines.llamacpp_rpc import (
        load_llamacpp_runtime_manifest,
        probe_llamacpp_architectures,
    )

    try:
        manifest = load_llamacpp_runtime_manifest(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return ExecutionEngineCapability(
            engine_id="llamacpp-rpc",
            enabled=False,
            formats=("gguf",),
            detail=f"pinned runtime manifest is invalid: {type(exc).__name__}: {exc}",
        )
    memory = psutil.virtual_memory()
    device_types = {item.casefold() for item in manifest.device_support}
    devices: list[ExecutionDevice] = [
        ExecutionDevice(
            device_id="cpu",
            device_type="cpu",
            name=platform.processor() or platform.machine() or "CPU",
            total_memory_bytes=int(memory.total),
            usable_memory_bytes=int(memory.available),
            features=("gguf", "hybrid-cpu-gpu"),
        )
    ]
    if "cuda" in device_types:
        try:
            import torch

            if torch.cuda.is_available():
                index = torch.cuda.current_device()
                free, total = torch.cuda.mem_get_info(index)
                devices.append(
                    ExecutionDevice(
                        device_id=f"cuda:{index}",
                        device_type="cuda",
                        name=str(torch.cuda.get_device_properties(index).name),
                        total_memory_bytes=int(total),
                        usable_memory_bytes=int(free),
                        runtime_version=str(torch.version.cuda or "unavailable"),
                        features=("gguf", "rpc", "hybrid-cpu-gpu"),
                    )
                )
        except (ImportError, RuntimeError):
            pass
    roles = [
        "critical_path_stage",
        "background_replica",
        "storage_cache",
        "idle",
    ]
    if manifest.rpc_enabled:
        roles.insert(1, "tensor_rpc_compute")
    architecture_probe = probe_llamacpp_architectures(
        manifest,
        tuple(
            sorted(
                {
                    identifier
                    for identifiers in GGUF_ARCHITECTURE_IDENTIFIERS.values()
                    for identifier in identifiers
                }
            )
        ),
    )
    return ExecutionEngineCapability(
        engine_id="llamacpp-rpc",
        enabled=True,
        runtime_revision=manifest.commit,
        binary_hashes={
            "llama-server": manifest.server_sha256,
            "ggml-rpc-server": manifest.rpc_server_sha256,
        },
        formats=("gguf",),
        model_architectures=architecture_probe.supported_identifiers,
        required_features=("gguf-model-loader",),
        unsupported_features=manifest.unsupported_features,
        devices=tuple(devices),
        roles=tuple(roles),
        detail=(
            f"installer-owned pinned build {manifest.build_id}; architecture probe="
            f"{architecture_probe.mechanism}:" + ",".join(architecture_probe.supported_identifiers)
        ),
    )


def _colibri_capability(path: Path | None) -> ExecutionEngineCapability:
    if path is None:
        return ExecutionEngineCapability(
            engine_id="colibri",
            enabled=False,
            detail="no installer-owned Colibri runtime manifest is configured",
        )
    try:
        manifest = load_colibri_runtime_manifest(path)
        if manifest.engine_directory is None or not manifest.engine_directory.is_dir():
            raise FileNotFoundError("engine_directory is unavailable")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return ExecutionEngineCapability(
            engine_id="colibri",
            enabled=False,
            detail=f"Colibri manifest is invalid: {type(exc).__name__}: {exc}",
        )
    memory = psutil.virtual_memory()
    return ExecutionEngineCapability(
        engine_id="colibri",
        enabled=True,
        runtime_revision=manifest.runtime_revision,
        binary_hashes=manifest.binary_hashes,
        formats=manifest.formats,
        quantizations=manifest.quantizations,
        devices=(
            ExecutionDevice(
                device_id="cpu",
                device_type="cpu",
                name=platform.processor() or platform.machine() or "CPU",
                total_memory_bytes=int(memory.total),
                usable_memory_bytes=int(memory.available),
                features=manifest.fast_paths,
            ),
        ),
        adapters=manifest.model_families,
        fast_paths=manifest.fast_paths,
        adapter_fast_paths=tuple(
            AdapterFastPathCapability(
                adapter_id=adapter_id,
                fast_path_id="routing-aware-placement",
                candidate_modes=tuple(
                    profile.profile_id
                    for profile in manifest.routing_profiles
                    if profile.admitted and profile.adapter_id == adapter_id
                ),
            )
            for adapter_id in manifest.model_families
            if any(
                profile.admitted and profile.adapter_id == adapter_id
                for profile in manifest.routing_profiles
            )
        ),
        execution_profiles=tuple(
            ExecutionProfileCapability(
                profile_id=profile.profile_id,
                mechanism="routing_aware_placement",
                adapter_id=profile.adapter_id,
                model_fingerprint=profile.model_fingerprint,
                content_fingerprint=profile.content_fingerprint,
                exactness_passed=profile.exactness_passed,
                measured_utility=profile.measured_utility,
                evidence_fingerprint=profile.evidence_fingerprint,
            )
            for profile in manifest.routing_profiles
            if profile.admitted
        ),
        roles=("critical_path_stage", "storage_cache", "idle"),
        detail="pinned Colibri runtime",
    )


def discover_local_cluster_capabilities(
    *,
    llamacpp_manifest: Path | None = None,
    colibri_manifest: Path | None = None,
) -> ClusterCapabilities:
    """Return physical local facts without running a performance benchmark."""

    general = discover_configured_general_engine_capabilities(
        llamacpp_manifest=llamacpp_manifest,
        colibri_manifest=colibri_manifest,
    )
    worker_id = f"local/{socket.gethostname()}"
    engines = (
        _native_capability(),
        *general,
    )
    return ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(
                worker_id=worker_id,
                node_id="local",
                engines=engines,
                reliability=1.0,
                storage_available_bytes=int(psutil.disk_usage(str(Path.cwd())).free),
            ),
        )
    )


def discover_configured_general_engine_capabilities(
    *,
    llamacpp_manifest: Path | None = None,
    colibri_manifest: Path | None = None,
    install_root: Path | None = None,
) -> tuple[ExecutionEngineCapability, ...]:
    """Read installer-owned runtime manifests without probing model families."""

    manifests = discover_installed_engine_manifests(
        llamacpp=llamacpp_manifest,
        colibri=colibri_manifest,
        install_root=install_root,
    )
    return (
        _llamacpp_capability(manifests.llamacpp),
        _colibri_capability(manifests.colibri),
    )


__all__ = [
    "discover_configured_general_engine_capabilities",
    "discover_local_cluster_capabilities",
]
