from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from swarm_inference.coordinator.expert_planner import (
    ExpertStrategy,
    ExpertStrategyCandidate,
    ExpertUtilityInputs,
    ExpertUtilityPlanner,
)
from swarm_inference.engines.colibri import ColibriExecutionEngine
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    EngineSupportStatus,
    ExecutionDevice,
    ExecutionEngineCapability,
    ExecutionRequest,
    WorkerExecutionCapability,
)
from swarm_inference.engines.llamacpp_rpc import (
    LlamaCppRpcEngine,
    LlamaCppRuntimeManifest,
    probe_llamacpp_architectures,
)
from swarm_inference.engines.native_stage import NativeStageEngine
from swarm_inference.engines.registry import ExecutionEngineRegistry, default_engine_registry
from swarm_inference.engines.topology import NetworkLinkProfile, TopologyDomain
from swarm_inference.model.adapter import NativeModelAdapterRegistry
from swarm_inference.model.architecture import (
    ModelArchitecture,
    architecture_from_config,
    architecture_from_gguf,
    normalize_model_architecture,
)
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.model.qwen3_moe import Qwen3MoeAdapter
from swarm_inference.model.resolver import ModelSourceResolver


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("OlmoeForCausalLM", ModelArchitecture.OLMOE),
        ("Qwen3ForCausalLM", ModelArchitecture.QWEN3_DENSE),
        ("qwen3_moe", ModelArchitecture.QWEN3_MOE),
        ("qwen3moe", ModelArchitecture.QWEN3_MOE),
        ("Qwen3MoeForCausalLM", ModelArchitecture.QWEN3_MOE),
        ("qwen35moe", ModelArchitecture.QWEN3_MOE),
        ("qwen3_5_moe", ModelArchitecture.QWEN3_MOE),
        ("qwen3_5_moe_text", ModelArchitecture.QWEN3_MOE),
        ("Qwen3_5MoeForConditionalGeneration", ModelArchitecture.QWEN3_MOE),
        ("FutureForCausalLM", "FutureForCausalLM"),
    ],
)
def test_architecture_aliases_are_conservative(raw: str, canonical: str) -> None:
    assert normalize_model_architecture(raw) == canonical


def test_hugging_face_and_gguf_architecture_sources_are_retained() -> None:
    hugging_face = architecture_from_config(
        {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}
    )
    gguf = architecture_from_gguf("qwen3moe", fallback=hugging_face)

    assert hugging_face.canonical == ModelArchitecture.QWEN3_MOE
    assert hugging_face.raw == "Qwen3MoeForCausalLM"
    assert hugging_face.source == "config.architectures"
    assert gguf.canonical == ModelArchitecture.QWEN3_MOE
    assert gguf.raw == "qwen3moe"
    assert gguf.source == "gguf.general.architecture"


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_metadata_only_gguf(path: Path, architecture: str) -> None:
    entries = (
        (_gguf_string("general.architecture"), 8, _gguf_string(architecture)),
        (_gguf_string(f"{architecture}.block_count"), 4, struct.pack("<I", 2)),
        (_gguf_string(f"{architecture}.embedding_length"), 4, struct.pack("<I", 16)),
    )
    payload = bytearray(b"GGUF")
    payload.extend(struct.pack("<IQQ", 3, 0, len(entries)))
    for key, value_type, value in entries:
        payload.extend(key)
        payload.extend(struct.pack("<I", value_type))
        payload.extend(value)
    path.write_bytes(bytes(payload))


def test_local_gguf_metadata_drives_qwen3_moe_detection(tmp_path: Path) -> None:
    model = tmp_path / "tiny-Q4_K_M.gguf"
    _write_metadata_only_gguf(model, "qwen3moe")

    descriptor = ModelSourceResolver(cache_directory=tmp_path / "cache").resolve(str(model))

    assert descriptor.architecture == ModelArchitecture.QWEN3_MOE
    assert descriptor.architecture_raw == "qwen3moe"
    assert descriptor.architecture_source == "gguf.general.architecture"
    assert descriptor.layer_count == 2
    assert descriptor.hidden_size == 16


def _model(
    *,
    architecture: str,
    model_format: str,
    raw: str | None = None,
    features: tuple[str, ...] = (),
) -> ResolvedModelDescriptor:
    suffix = "gguf" if model_format == "gguf" else "safetensors"
    return ResolvedModelDescriptor(
        model_id="fixture/model",
        revision="a" * 40,
        content_fingerprint="sha256:" + "b" * 64,
        source_type="huggingface",
        format=model_format,  # type: ignore[arg-type]
        architecture=architecture,
        architecture_raw=raw or architecture,
        architecture_source=(
            "gguf.general.architecture" if model_format == "gguf" else "config.architectures"
        ),
        files=(ModelFileDescriptor(relative_path=f"model.{suffix}", size_bytes=100),),
        weight_bytes=100,
        layer_count=4,
        hidden_size=16,
        activation_dtype_bytes=2,
        features=features,
    )


def _worker(engine: ExecutionEngineCapability, *, worker_id: str = "node-a/worker"):
    return WorkerExecutionCapability(
        worker_id=worker_id,
        node_id=worker_id.split("/", 1)[0],
        engines=(engine,),
    )


def _device(*, rate: float = 10) -> ExecutionDevice:
    return ExecutionDevice(
        device_id="cpu",
        device_type="cpu",
        name="fixture CPU",
        usable_memory_bytes=1_000,
        measured_decode_tokens_s=rate,
    )


def test_native_qwen3_moe_capability_and_unsupported_representation() -> None:
    capability = ExecutionEngineCapability(
        engine_id="native-stage",
        enabled=True,
        runtime_revision="native-v1",
        binary_hashes={"python": "sha256:runtime"},
        formats=("safetensors",),
        adapters=("qwen3_moe",),
        devices=(_device(),),
        roles=("contiguous-stage",),
    )
    engine = NativeStageEngine(adapters=NativeModelAdapterRegistry((Qwen3MoeAdapter(),)))
    cluster = ClusterCapabilities(workers=(_worker(capability),))

    supported = engine.probe_model_support(
        _model(
            architecture=ModelArchitecture.QWEN3_MOE,
            model_format="safetensors",
            raw="Qwen3MoeForCausalLM",
        ),
        cluster,
    )
    unsupported = engine.probe_model_support(
        _model(
            architecture=ModelArchitecture.QWEN3_MOE,
            model_format="safetensors",
            raw="qwen35moe",
        ),
        cluster,
    )

    assert supported.status == EngineSupportStatus.SUPPORTED
    assert supported.compatibility == "supported"
    assert supported.adapter_id == "qwen3_moe"
    assert unsupported.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert "Transformers qwen3_moe representation only" in unsupported.reason


def test_colibri_remains_selectable_and_rejects_unadvertised_architecture() -> None:
    capability = ExecutionEngineCapability(
        engine_id="colibri",
        enabled=True,
        runtime_revision="colibri-v1",
        binary_hashes={"colibri": "sha256:runtime"},
        formats=("safetensors",),
        adapters=("olmoe",),
        devices=(_device(),),
        roles=("complete-model",),
    )
    engine = ColibriExecutionEngine()
    cluster = ClusterCapabilities(workers=(_worker(capability),))
    olmoe = _model(architecture=ModelArchitecture.OLMOE, model_format="safetensors")
    qwen = _model(
        architecture=ModelArchitecture.QWEN3_MOE,
        model_format="safetensors",
        raw="Qwen3MoeForCausalLM",
    )

    report = engine.probe_model_support(olmoe, cluster)
    rejected = engine.probe_model_support(qwen, cluster)

    assert report.status == EngineSupportStatus.SUPPORTED
    assert report.adapter_id == "olmoe"
    assert rejected.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert "matching" in rejected.reason


@pytest.mark.asyncio
async def test_automatic_registry_selection_can_select_colibri() -> None:
    capability = ExecutionEngineCapability(
        engine_id="colibri",
        enabled=True,
        runtime_revision="colibri-v1",
        binary_hashes={"colibri": "sha256:runtime"},
        formats=("safetensors",),
        adapters=("olmoe",),
        devices=(_device(),),
        roles=("complete-model",),
    )
    cluster = ClusterCapabilities(workers=(_worker(capability),))
    registry = ExecutionEngineRegistry((ColibriExecutionEngine(),))

    result = await registry.compete(
        _model(architecture=ModelArchitecture.OLMOE, model_format="safetensors"),
        cluster,
        ExecutionRequest(),
    )

    assert result.selected.engine_id == "colibri"
    assert result.support[0].compatibility == "supported"


def test_all_canonical_engines_and_qwen3_moe_adapter_are_registered() -> None:
    assert tuple(engine.engine_id for engine in default_engine_registry().engines()) == (
        "colibri",
        "llamacpp-rpc",
        "native-stage",
    )
    assert "qwen3_moe" in {
        adapter.adapter_id for adapter in NativeStageEngine().adapters.adapters()
    }


def _llama_capability(
    *,
    architectures: tuple[str, ...] = ("qwen3moe",),
    unsupported_features: tuple[str, ...] = (),
) -> ExecutionEngineCapability:
    return ExecutionEngineCapability(
        engine_id="llamacpp-rpc",
        enabled=True,
        runtime_revision="llama-commit",
        binary_hashes={"llama-server": "sha256:server"},
        formats=("gguf",),
        model_architectures=architectures,
        unsupported_features=unsupported_features,
        devices=(_device(),),
        roles=("critical_path_stage", "tensor_rpc_compute"),
    )


def test_llamacpp_capability_is_architecture_and_feature_exact() -> None:
    model = _model(
        architecture=ModelArchitecture.QWEN3_MOE,
        model_format="gguf",
        raw="qwen3moe",
    )
    supported = LlamaCppRpcEngine().probe_model_support(
        model,
        ClusterCapabilities(workers=(_worker(_llama_capability()),)),
    )
    unsupported_architecture = LlamaCppRpcEngine().probe_model_support(
        model,
        ClusterCapabilities(workers=(_worker(_llama_capability(architectures=("qwen3",))),)),
    )
    unsupported_feature = LlamaCppRpcEngine().probe_model_support(
        model.model_copy(update={"features": ("vision",)}),
        ClusterCapabilities(
            workers=(_worker(_llama_capability(unsupported_features=("vision",))),)
        ),
    )
    unknown = LlamaCppRpcEngine().probe_model_support(
        _model(architecture="future_model", model_format="gguf"),
        ClusterCapabilities(workers=(_worker(_llama_capability()),)),
    )
    future_runtime_support = LlamaCppRpcEngine().probe_model_support(
        _model(architecture="future_model", model_format="gguf"),
        ClusterCapabilities(workers=(_worker(_llama_capability(architectures=("future_model",))),)),
    )
    unproven_gguf_metadata = LlamaCppRpcEngine().probe_model_support(
        model.model_copy(update={"architecture_source": "config.architectures"}),
        ClusterCapabilities(workers=(_worker(_llama_capability()),)),
    )
    qwen36_wrong_family_loader = LlamaCppRpcEngine().probe_model_support(
        model.model_copy(update={"architecture_raw": "qwen35moe"}),
        ClusterCapabilities(workers=(_worker(_llama_capability(architectures=("qwen3moe",))),)),
    )

    assert supported.status == EngineSupportStatus.SUPPORTED
    assert supported.required_features == ("qwen3moe",)
    assert unsupported_architecture.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert unsupported_feature.unsupported_features == ("vision",)
    assert unknown.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert future_runtime_support.status == EngineSupportStatus.SUPPORTED
    assert unproven_gguf_metadata.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert "exact GGUF general.architecture" in unproven_gguf_metadata.reason
    assert qwen36_wrong_family_loader.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert "qwen35moe" in qwen36_wrong_family_loader.reason


def test_llamacpp_binary_probe_is_hash_bound_and_not_filename_based(tmp_path: Path) -> None:
    server = tmp_path / "arbitrary-server-name.bin"
    rpc = tmp_path / "arbitrary-rpc-name.bin"
    server.write_bytes(b"prefix\x00qwen3\x00qwen3moe\x00suffix")
    rpc.write_bytes(b"rpc")

    def digest(path: Path) -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = LlamaCppRuntimeManifest(
        commit="pinned-commit",
        build_id="fixture-build",
        platform="fixture",
        server_binary=server,
        server_sha256=digest(server),
        rpc_server_binary=rpc,
        rpc_server_sha256=digest(rpc),
        build_flags={"GGML_RPC": True},
    )

    report = probe_llamacpp_architectures(manifest, ("qwen3", "qwen3moe", "unsupported"))

    assert report.supported_identifiers == ("qwen3", "qwen3moe")
    assert report.binary_sha256 == digest(server)
    assert report.mechanism == "bounded-binary-identifier-scan"


def _remote_candidate(domain: TopologyDomain) -> ExpertStrategyCandidate:
    return ExpertStrategyCandidate(
        candidate_id=f"remote-{domain.value}",
        strategy=ExpertStrategy.MICROSHARD_REMOTE,
        worker_ids=["node-b/worker"],
        utility=ExpertUtilityInputs(
            measured_local_expert_ms=10,
            measured_remote_expert_ms=1,
            serialization_ms=0,
            network_transfer_ms=0,
            queue_delay_ms=0,
            reduction_ms=0,
            cache_hit_rate=1,
        ),
        topology_domain=domain,
    )


def test_fine_grained_expert_execution_is_local_domain_only() -> None:
    local = ExpertStrategyCandidate(
        candidate_id="local",
        strategy=ExpertStrategy.LOCAL,
    )
    planner = ExpertUtilityPlanner()

    local_fast = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=[local, _remote_candidate(TopologyDomain.LOCAL_FAST)],
    )
    wan = planner.choose(
        stage_id=0,
        layer_id=0,
        expert_id=0,
        candidates=[local, _remote_candidate(TopologyDomain.WAN)],
    )

    assert local_fast.selected_strategy == ExpertStrategy.MICROSHARD_REMOTE
    assert wan.selected_strategy == ExpertStrategy.LOCAL
    assert any(
        "not admitted across a WAN" in reason for item in wan.rejected for reason in item.reasons
    )


@pytest.mark.asyncio
async def test_speed_excludes_a_weak_stage_but_capacity_can_include_it() -> None:
    model = _model(
        architecture=ModelArchitecture.QWEN3_MOE,
        model_format="safetensors",
        raw="Qwen3MoeForCausalLM",
    )
    strong_capability = ExecutionEngineCapability(
        engine_id="native-stage",
        enabled=True,
        runtime_revision="native-v1",
        binary_hashes={"python": "sha256:runtime"},
        formats=("safetensors",),
        adapters=("qwen3_moe",),
        devices=(_device(rate=20),),
        roles=("contiguous-stage",),
    )
    weak_capability = strong_capability.model_copy(update={"devices": (_device(rate=0.1),)})
    fast_link = NetworkLinkProfile(
        rtt_ms=0.1,
        bandwidth_bytes_s=1_000_000_000,
        jitter_ms=0.01,
        stability=1,
        sample_count=10,
        authenticated=True,
        provenance="fixture",
    )
    strong = _worker(strong_capability)
    strong = strong.model_copy(update={"network_links": {"node-b/worker": fast_link}})
    weak = _worker(weak_capability, worker_id="node-b/worker")
    cluster = ClusterCapabilities(workers=(strong, weak))
    engine = NativeStageEngine(adapters=NativeModelAdapterRegistry((Qwen3MoeAdapter(),)))

    speed = await ExecutionEngineRegistry((engine,)).compete(
        model, cluster, ExecutionRequest(objective="speed")
    )
    capacity = await engine.candidate_plans(model, cluster, ExecutionRequest(objective="capacity"))

    assert "node-b/worker" not in speed.selected.worker_roles
    assert any("node-b/worker" in plan.worker_roles for plan in capacity)


@pytest.mark.asyncio
async def test_qwen3_moe_synthetic_distributed_dry_run_uses_contiguous_wan_stages() -> None:
    model = _model(
        architecture=ModelArchitecture.QWEN3_MOE,
        model_format="safetensors",
        raw="Qwen3MoeForCausalLM",
    )
    capability = ExecutionEngineCapability(
        engine_id="native-stage",
        enabled=True,
        runtime_revision="native-v1",
        binary_hashes={"python": "sha256:runtime"},
        formats=("safetensors",),
        adapters=("qwen3_moe",),
        devices=(_device(rate=10),),
        roles=("contiguous-stage",),
    )
    wan_link = NetworkLinkProfile(
        rtt_ms=90,
        bandwidth_bytes_s=20_000_000,
        jitter_ms=8,
        stability=0.98,
        sample_count=10,
        authenticated=True,
        provenance="synthetic dry-run fixture",
    )
    first = _worker(capability).model_copy(update={"network_links": {"node-b/worker": wan_link}})
    second = _worker(capability, worker_id="node-b/worker")
    registry = ExecutionEngineRegistry(
        (NativeStageEngine(adapters=NativeModelAdapterRegistry((Qwen3MoeAdapter(),))),)
    )

    decision = await registry.compete(
        model,
        ClusterCapabilities(workers=(first, second)),
        ExecutionRequest(
            objective="capacity",
            requested_engine="native-stage",
            require_distributed=True,
        ),
    )

    assert decision.selected.topology == "direct-stage-ring-2"
    assert set(decision.selected.worker_roles) == {"node-a/worker", "node-b/worker"}
    assert decision.selected.number_of_wan_stage_boundaries == 1
    assert decision.selected.persistent_connections is True
    assignments = decision.selected.stage_assignments
    assert [(item["layer_start"], item["layer_end"]) for item in assignments] == [
        (0, 2),
        (2, 4),
    ]
