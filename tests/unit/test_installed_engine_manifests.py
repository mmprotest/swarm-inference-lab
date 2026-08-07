from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swarm_inference.engines.installed import discover_installed_engine_manifests
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionRequest,
    WorkerExecutionCapability,
)
from swarm_inference.engines.llamacpp_rpc import LlamaCppRpcEngine
from swarm_inference.engines.local_capabilities import (
    discover_configured_general_engine_capabilities,
)
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.worker.engine_factory import build_worker_engine_runtime


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_llama_runtime(root: Path, *, rpc: bool) -> Path:
    runtime = root / "runtime" / "engines" / "llamacpp"
    runtime.mkdir(parents=True)
    server = runtime / "llama-server.exe"
    rpc_server = runtime / "rpc-server.exe"
    server.write_bytes(b"server")
    rpc_server.write_bytes(b"rpc")
    manifest = runtime / "runtime-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": "1" * 40,
                "build_id": "llama-pinned",
                "platform": "windows-x64",
                "server_binary": server.name,
                "server_sha256": _sha(server),
                "rpc_server_binary": rpc_server.name,
                "rpc_server_sha256": _sha(rpc_server),
                "build_flags": {"GGML_RPC": rpc},
                "device_support": ["CPU"],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _model() -> ResolvedModelDescriptor:
    return ResolvedModelDescriptor(
        model_id="org/model",
        revision="2" * 40,
        content_fingerprint="sha256:" + "3" * 64,
        source_type="huggingface",
        format="gguf",
        architecture="qwen3",
        files=(ModelFileDescriptor(relative_path="model.gguf", size_bytes=100),),
        weight_bytes=100,
    )


def _worker(
    worker_id: str,
    node_id: str,
    roles: tuple[str, ...],
    *,
    memory: int = 70,
) -> WorkerExecutionCapability:
    return WorkerExecutionCapability(
        worker_id=worker_id,
        node_id=node_id,
        engines=(
            ExecutionEngineCapability(
                engine_id="llamacpp-rpc",
                enabled=True,
                runtime_revision="1" * 40,
                binary_hashes={"llama-server": "4" * 64, "rpc-server": "5" * 64},
                formats=("gguf",),
                roles=roles,
                devices=(
                    ExecutionDevice(
                        device_id="cpu",
                        device_type="cpu",
                        name="CPU",
                        usable_memory_bytes=memory,
                        measured_decode_tokens_s=10,
                    ),
                ),
            ),
        ),
    )


def test_installer_owned_manifests_are_discovered_without_path_lookup(tmp_path: Path) -> None:
    manifest = _write_llama_runtime(tmp_path, rpc=True)

    discovered = discover_installed_engine_manifests(
        install_root=tmp_path,
        environment={},
    )

    assert discovered.llamacpp == manifest.resolve()
    assert discovered.colibri is None


def test_llamacpp_advertises_tensor_rpc_only_for_a_proven_rpc_build(tmp_path: Path) -> None:
    _write_llama_runtime(tmp_path, rpc=False)
    capability = discover_configured_general_engine_capabilities(install_root=tmp_path)[0]
    assert capability.enabled
    assert "critical_path_stage" in capability.roles
    assert "tensor_rpc_compute" not in capability.roles


def test_worker_runtime_automatically_owns_the_installed_llamacpp_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_llama_runtime(tmp_path, rpc=True)
    monkeypatch.setenv("SWARM_LLAMACPP_RUNTIME_MANIFEST", str(manifest))
    identity = tmp_path / "identity"
    identity.mkdir()

    runtime = build_worker_engine_runtime(
        worker_id="node-a/worker",
        advertised_endpoint="127.0.0.1:5001",
        identity_directory=identity,
        artifact_manager=None,
        llamacpp_runtime_manifest=None,
        colibri_runtime_manifest=None,
    )

    assert runtime is not None
    assert runtime.engine_ids == ("llamacpp-rpc",)


@pytest.mark.asyncio
async def test_llamacpp_distributed_plan_requires_explicit_rpc_roles() -> None:
    engine = LlamaCppRpcEngine()
    owner = _worker("owner", "node-a", ("critical_path_stage",))
    incapable = _worker("compute", "node-b", ("storage_cache",))

    assert not await engine.candidate_plans(
        _model(),
        ClusterCapabilities(workers=(owner, incapable)),
        ExecutionRequest(objective="capacity", require_distributed=True),
    )

    compute = _worker("compute", "node-b", ("tensor_rpc_compute",))
    plans = await engine.candidate_plans(
        _model(),
        ClusterCapabilities(workers=(owner, compute)),
        ExecutionRequest(objective="capacity", require_distributed=True),
    )
    assert len(plans) == 1
    assert plans[0].worker_roles == {
        "owner": "critical_path_stage",
        "compute": "tensor_rpc_compute",
    }


@pytest.mark.asyncio
async def test_llamacpp_identity_excludes_unselected_workers() -> None:
    engine = LlamaCppRpcEngine()
    owner = _worker("owner", "node-a", ("critical_path_stage",), memory=200)
    first = await engine.candidate_plans(
        _model(),
        ClusterCapabilities(workers=(owner,)),
        ExecutionRequest(objective="speed"),
    )
    unused = _worker("storage", "node-b", ("storage_cache",), memory=500)
    second = await engine.candidate_plans(
        _model(),
        ClusterCapabilities(workers=(owner, unused)),
        ExecutionRequest(objective="speed"),
    )

    assert len(first) == len(second) == 1
    assert first[0].execution_identity == second[0].execution_identity
