from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.engines.colibri import ColibriExecutionEngine
from swarm_inference.engines.compatibility import (
    ValidationEvidence,
    build_compatibility_registry,
    engine_runtime_fingerprint,
)
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    CompatibilityStatus,
    EngineSupportStatus,
    ExecutionDevice,
    ExecutionEngineCapability,
    WorkerExecutionCapability,
)
from swarm_inference.engines.llamacpp_rpc import LlamaCppRpcEngine
from swarm_inference.engines.native_stage import NativeStageEngine
from swarm_inference.model.architecture import (
    ShardReduction,
    TensorRole,
    architecture_from_config,
)
from swarm_inference.model.architecture_adapters import (
    default_architecture_adapter_registry,
)
from swarm_inference.model.descriptor import ModelFileDescriptor, ResolvedModelDescriptor
from swarm_inference.model.quantization import quantization_from_config


def _dense(model_type: str, architecture: str) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "architectures": [architecture],
        "num_hidden_layers": 4,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 128,
        "torch_dtype": "bfloat16",
    }


def _moe(
    model_type: str,
    architecture: str,
    *,
    experts: int = 8,
    top_k: int = 2,
) -> dict[str, Any]:
    return {
        **_dense(model_type, architecture),
        "num_experts": experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": 48,
    }


FAMILY_CONFIGS: tuple[tuple[str, dict[str, Any], str, str], ...] = (
    ("Qwen3", _dense("qwen3", "Qwen3ForCausalLM"), "qwen3_dense", "dense"),
    (
        "Qwen3 MoE",
        _moe("qwen3_moe", "Qwen3MoeForCausalLM"),
        "qwen3_moe",
        "moe",
    ),
    (
        "Qwen3.5",
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                **_dense("qwen3_5_text", "Qwen3_5ForCausalLM"),
                "layer_types": ["linear_attention", "full_attention"] * 2,
            },
            "vision_config": {"hidden_size": 16},
        },
        "qwen3_5_dense",
        "dense",
    ),
    (
        "Qwen3.6-35B-A3B",
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": {
                **_moe("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
                "layer_types": ["linear_attention", "full_attention"] * 2,
            },
            "vision_config": {"hidden_size": 16},
        },
        "qwen3_5_moe",
        "moe",
    ),
    (
        "Qwen3.5 MoE",
        {
            **_moe("qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
            "layer_types": ["linear_attention", "full_attention"] * 2,
        },
        "qwen3_5_moe",
        "moe",
    ),
    (
        "Qwen3.6 dense",
        {
            **_dense("qwen3_5", "Qwen3_5ForCausalLM"),
            "layer_types": ["linear_attention", "full_attention"] * 2,
        },
        "qwen3_5_dense",
        "dense",
    ),
    (
        "Kimi K2",
        {
            "model_type": "kimi_k2",
            "architectures": ["KimiK2ForCausalLM"],
            "text_config": {
                **_moe("deepseek_v3", "DeepseekV3ForCausalLM", experts=16, top_k=4),
                "n_routed_experts": 16,
                "n_shared_experts": 1,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 8,
                "v_head_dim": 8,
            },
        },
        "kimi_k2_moe",
        "moe",
    ),
    (
        "Kimi K2.5",
        {
            "model_type": "kimi_k25",
            "architectures": ["KimiK25ForConditionalGeneration"],
            "text_config": {
                **_moe("deepseek_v3", "DeepseekV3ForCausalLM", experts=16, top_k=4),
                "n_routed_experts": 16,
                "n_shared_experts": 1,
                "kv_lora_rank": 16,
                "quantization_config": {
                    "format": "pack-quantized",
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "targets": ["Linear"],
                            "weights": {
                                "type": "int",
                                "num_bits": 4,
                                "strategy": "group",
                                "group_size": 32,
                                "symmetric": True,
                                "dynamic": False,
                            },
                        }
                    },
                },
            },
        },
        "kimi_k2_moe",
        "moe",
    ),
    (
        "Kimi K3",
        {
            "model_type": "kimi_k3",
            "architectures": ["KimiK3ForConditionalGeneration"],
            "text_config": {
                **_moe("kimi_k3_text", "KimiK3ForCausalLM"),
                "num_experts_per_token": 2,
                "num_shared_experts": 1,
                "routed_expert_hidden_size": 32,
                "linear_attn_config": {"kda_layers": [0, 2]},
                "quantization_config": {"format": "mxfp4-pack-quantized"},
            },
        },
        "kimi_k3_moe",
        "moe",
    ),
    (
        "GLM-5 MoE",
        {
            **_moe("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "kv_lora_rank": 16,
        },
        "glm_moe",
        "moe",
    ),
    (
        "DeepSeek V3/R1",
        {
            **_moe("deepseek_v3", "DeepseekV3ForCausalLM", experts=16, top_k=4),
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "kv_lora_rank": 16,
            "n_group": 4,
            "topk_group": 2,
        },
        "deepseek_v3_moe",
        "moe",
    ),
    (
        "DeepSeek R1",
        {
            **_moe("deepseek_v3", "DeepseekV3ForCausalLM", experts=16, top_k=4),
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "kv_lora_rank": 16,
            "n_group": 4,
            "topk_group": 2,
        },
        "deepseek_v3_moe",
        "moe",
    ),
    (
        "DeepSeek V3.2",
        {
            **_moe("deepseek_v32", "DeepseekV32ForCausalLM", experts=16, top_k=4),
            "n_routed_experts": 16,
            "n_shared_experts": 1,
            "kv_lora_rank": 16,
            "q_lora_rank": 24,
            "n_group": 4,
            "topk_group": 2,
            "topk_method": "noaux_tc",
            "scoring_func": "sigmoid",
        },
        "deepseek_v3_moe",
        "moe",
    ),
    (
        "MiniMax",
        {
            **_moe("minimax_m2", "MiniMaxM2ForCausalLM", experts=16, top_k=4),
            "num_local_experts": 16,
            "num_shared_experts": 1,
            "layer_types": ["linear_attention", "full_attention"] * 2,
        },
        "minimax_moe",
        "moe",
    ),
    ("Llama", _dense("llama", "LlamaForCausalLM"), "llama_dense", "dense"),
    (
        "Llama 4",
        {
            **_moe("llama4", "Llama4ForCausalLM"),
            "num_local_experts": 8,
        },
        "llama4_moe",
        "moe",
    ),
    ("Mistral", _dense("mistral", "MistralForCausalLM"), "mistral_dense", "dense"),
    (
        "Mixtral",
        {
            **_moe("mixtral", "MixtralForCausalLM"),
            "num_local_experts": 8,
        },
        "mixtral_moe",
        "moe",
    ),
    (
        "Mistral 4 MoE",
        {
            "model_type": "mistral3",
            "architectures": ["Mistral3ForConditionalGeneration"],
            "text_config": {
                **_moe("mistral4", "Mistral4ForCausalLM"),
                "n_routed_experts": 8,
            },
        },
        "mistral4_moe",
        "moe",
    ),
    (
        "Gemma",
        {
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "text_config": _dense("gemma3_text", "Gemma3ForCausalLM"),
            "vision_config": {"hidden_size": 16},
        },
        "gemma_dense",
        "dense",
    ),
)


def _descriptor(
    config: dict[str, Any],
    *,
    model_format: str = "safetensors",
    raw_architecture: str | None = None,
    quantization: str | None = None,
) -> ResolvedModelDescriptor:
    identity = architecture_from_config(config)
    raw = raw_architecture or identity.raw
    source = "gguf.general.architecture" if model_format == "gguf" else identity.source
    model = ResolvedModelDescriptor(
        model_id="renamed-organization/derivative-checkpoint",
        revision="a" * 40,
        content_fingerprint="sha256:" + "b" * 64,
        source_type="local",
        format=model_format,  # type: ignore[arg-type]
        architecture=identity.canonical,
        architecture_raw=raw,
        architecture_source=source,
        files=(ModelFileDescriptor(relative_path=f"weights.{model_format}", size_bytes=1_000_000),),
        quantization=quantization,
        weight_bytes=1_000_000,
        configuration=config,
    )
    adapter = default_architecture_adapter_registry().resolve_model(model)
    assert adapter is not None
    return model.model_copy(update={"architecture_profile": adapter.inspect(model)})


@pytest.mark.parametrize(
    ("family", "config", "architecture_id", "dense_or_moe"),
    FAMILY_CONFIGS,
    ids=[item[0] for item in FAMILY_CONFIGS],
)
def test_major_family_profiles_are_metadata_driven(
    family: str,
    config: dict[str, Any],
    architecture_id: str,
    dense_or_moe: str,
) -> None:
    del family
    descriptor = _descriptor(config)
    profile = descriptor.architecture_profile
    assert descriptor.model_id == "renamed-organization/derivative-checkpoint"
    assert descriptor.architecture == architecture_id
    assert profile is not None
    assert profile.architecture_id == architecture_id
    assert profile.dense_or_moe == dense_or_moe
    assert profile.layer_count == 4
    assert profile.hidden_size == 32
    if dense_or_moe == "moe":
        assert profile.expert_count is not None
        assert profile.experts_per_token is not None
        assert "routed-experts" in profile.capabilities
    else:
        assert profile.expert_count is None


def test_tensor_interpretation_and_microshards_use_semantics_not_names_in_runtime() -> None:
    descriptor = _descriptor(dict(FAMILY_CONFIGS[7][1]))
    adapter = default_architecture_adapter_registry().resolve_model(descriptor)
    assert adapter is not None
    gate = adapter.interpret_tensor(
        "model.layers.1.mlp.experts.3.gate_proj.weight",
        shape=(48, 32),
        dtype="BF16",
        byte_size=3072,
    )
    down = adapter.interpret_tensor(
        "model.layers.1.mlp.experts.3.down_proj.weight",
        shape=(32, 48),
        dtype="BF16",
        byte_size=3072,
    )
    up = adapter.interpret_tensor(
        "model.layers.1.mlp.experts.3.up_proj.weight",
        shape=(48, 32),
        dtype="BF16",
        byte_size=3072,
    )
    shared = adapter.interpret_tensor(
        "model.layers.1.mlp.shared_experts.0.gate_proj.weight",
        shape=(48, 32),
        dtype="BF16",
        byte_size=3072,
    )
    experts = adapter.describe_experts((gate, down, up, shared), descriptor.architecture_profile)

    assert gate.role == TensorRole.ROUTED_EXPERT
    assert gate.shard_semantics[0].reduction == ShardReduction.CONCATENATE
    assert down.shard_semantics[0].reduction == ShardReduction.SUM
    assert {(item.expert_type, item.expert_index) for item in experts} == {
        ("routed", 3),
        ("shared", 0),
    }


def test_fused_expert_axis_expands_to_canonical_expert_descriptors() -> None:
    descriptor = _descriptor(
        dict(next(config for family, config, *_rest in FAMILY_CONFIGS if family == "Llama 4"))
    )
    adapter = default_architecture_adapter_registry().resolve_model(descriptor)
    assert adapter is not None
    tensors = tuple(
        adapter.interpret_tensor(name, shape=shape, dtype="BF16", byte_size=4096)
        for name, shape in (
            ("model.layers.0.mlp.experts.gate_up_proj.weight", (8, 96, 32)),
            ("model.layers.0.mlp.experts.down_proj.weight", (8, 32, 48)),
        )
    )
    experts = adapter.describe_experts(tensors, descriptor.architecture_profile)

    assert len(experts) == 8
    assert experts[5].routing_metadata["tensor_slices"] == {
        tensors[0].tensor_name: {"axis": 0, "index": 5},
        tensors[1].tensor_name: {"axis": 0, "index": 5},
    }


def test_structural_quantization_metadata_and_kimi_packed_tensor_mapping() -> None:
    kimi_k25 = next(config for family, config, *_rest in FAMILY_CONFIGS if family == "Kimi K2.5")
    kimi_k3 = next(config for family, config, *_rest in FAMILY_CONFIGS if family == "Kimi K3")
    k3_config = dict(kimi_k3)
    k3_text = dict(k3_config["text_config"])
    k3_text["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
    }
    k3_config["text_config"] = k3_text

    assert quantization_from_config(kimi_k25) == "int4-g32"
    assert quantization_from_config(k3_config) == "mxfp4"

    descriptor = _descriptor(dict(kimi_k25))
    adapter = default_colibri_adapter_registry().resolve_model(descriptor)
    assert adapter is not None
    prefix = "language_model.model.layers.17.mlp.experts.383.gate_proj"
    mapped = adapter.map_tensor_names(
        (
            (f"{prefix}.weight_packed", (2048, 896), "I32", 7_340_032),
            (f"{prefix}.weight_scale", (2048, 224), "BF16", 917_504),
            (f"{prefix}.weight_shape", (2,), "I32", 8),
        ),
        config=descriptor.configuration,
    )

    packed, scale, shape = mapped
    assert packed.tensor_role == "routed_expert_gate_projection"
    assert packed.logical_shape == (2048, 7168)
    assert packed.quantization_format == "int4-g32"
    assert packed.scale_group_size == 32
    assert packed.packing == "compressed-tensors-signed-offset-eight-nibbles-i32"
    assert scale.tensor_role == "routed_expert_gate_scale"
    assert shape.tensor_role == "routed_expert_gate_shape"


def _cluster(*capabilities: ExecutionEngineCapability) -> ClusterCapabilities:
    return ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(
                worker_id="node-a/worker",
                node_id="node-a",
                engines=capabilities,
            ),
        )
    )


def _device() -> ExecutionDevice:
    return ExecutionDevice(
        device_id="cuda:0",
        device_type="cuda",
        name="fixture GPU",
        usable_memory_bytes=2_000_000_000,
        measured_prefill_tokens_s=20,
        measured_decode_tokens_s=10,
    )


def test_qwen36_format_identity_and_engine_probes_are_independent() -> None:
    config = dict(FAMILY_CONFIGS[3][1])
    safetensors = _descriptor(config)
    gguf = _descriptor(
        config,
        model_format="gguf",
        raw_architecture="qwen35moe",
        quantization="Q4_K_M",
    )
    metadata_only_gguf = gguf.model_copy(
        update={"configuration": {"general.architecture": "qwen35moe"}}
    )
    cluster = _cluster(
        ExecutionEngineCapability(
            engine_id="colibri",
            enabled=True,
            runtime_revision="colibri-pinned",
            binary_hashes={"swarm_moe": "sha256:runtime"},
            formats=("safetensors",),
            adapters=("qwen3-5-moe",),
            devices=(_device(),),
            roles=("expert-execution",),
        ),
        ExecutionEngineCapability(
            engine_id="llamacpp-rpc",
            enabled=True,
            runtime_revision="llama-pinned",
            binary_hashes={"llama-server": "sha256:runtime"},
            formats=("gguf",),
            model_architectures=("qwen35moe",),
            devices=(_device(),),
            roles=("critical_path_stage", "tensor_rpc_compute"),
        ),
    )

    colibri_safe = ColibriExecutionEngine().probe(safetensors, cluster)
    colibri_gguf = ColibriExecutionEngine().probe(gguf, cluster)
    colibri_metadata_only_gguf = ColibriExecutionEngine().probe(metadata_only_gguf, cluster)
    llama_safe = LlamaCppRpcEngine().probe(safetensors, cluster)
    llama_gguf = LlamaCppRpcEngine().probe(gguf, cluster)
    native_gguf = NativeStageEngine().probe(gguf, cluster)
    no_runtime = _cluster(
        ExecutionEngineCapability(
            engine_id="colibri",
            enabled=False,
            formats=("safetensors",),
        )
    )
    colibri_safe_without_runtime = ColibriExecutionEngine().probe(safetensors, no_runtime)
    colibri_gguf_without_runtime = ColibriExecutionEngine().probe(gguf, no_runtime)

    assert safetensors.architecture == gguf.architecture == "qwen3_5_moe"
    assert colibri_safe.status == EngineSupportStatus.COMPONENT_SUPPORTED
    assert colibri_safe.support_scope == "component"
    assert not colibri_safe.supported
    assert colibri_gguf.status == EngineSupportStatus.UNSUPPORTED_FORMAT
    assert colibri_gguf.architecture_supported is True
    assert colibri_metadata_only_gguf.status == EngineSupportStatus.UNSUPPORTED_FORMAT
    assert colibri_metadata_only_gguf.architecture_supported is True
    assert colibri_metadata_only_gguf.format_supported is False
    assert llama_safe.status == EngineSupportStatus.UNSUPPORTED_FORMAT
    assert llama_gguf.status == EngineSupportStatus.SUPPORTED
    assert native_gguf.status == EngineSupportStatus.UNSUPPORTED_ARCHITECTURE
    assert native_gguf.reason == (
        "no registered native-stage adapter accepts architecture 'qwen3_5_moe'; "
        "native-stage accepts safetensors, not gguf"
    )
    assert colibri_safe_without_runtime.status == EngineSupportStatus.MISSING_RUNTIME
    assert colibri_safe_without_runtime.support_scope == "component"
    assert colibri_gguf_without_runtime.status == EngineSupportStatus.UNSUPPORTED_FORMAT
    assert colibri_gguf_without_runtime.support_scope == "component"

    registry = build_compatibility_registry(
        (safetensors, gguf),
        (ColibriExecutionEngine(), LlamaCppRpcEngine()),
        cluster,
    )
    records = registry.architectures["qwen3_5_moe"]
    assert records["safetensors"][0].engines["colibri"].support_status.value == (
        "SUPPORTED_WITH_LIMITATIONS"
    )
    assert records["gguf"][0].engines["llamacpp-rpc"].supported is True

    runtime_fingerprint = engine_runtime_fingerprint(llama_gguf)
    assert runtime_fingerprint is not None
    validated = build_compatibility_registry(
        (gguf,),
        (LlamaCppRpcEngine(),),
        cluster,
        validation_evidence=(
            ValidationEvidence(
                model_fingerprint=gguf.content_fingerprint,
                engine_id="llamacpp-rpc",
                runtime_fingerprint=runtime_fingerprint,
                status=CompatibilityStatus.SOFTWARE_VALIDATED,
                evidence_fingerprint="sha256:" + "a" * 64,
            ),
        ),
    )
    record = validated.architectures["qwen3_5_moe"]["gguf"][0]
    assert record.engines["llamacpp-rpc"].validation_status == (
        CompatibilityStatus.SOFTWARE_VALIDATED
    )


def test_colibri_adapter_registry_covers_major_moe_architectures() -> None:
    registry = default_colibri_adapter_registry()
    expected = {
        "qwen3_moe": "qwen3-moe",
        "qwen3_5_moe": "qwen3-5-moe",
        "kimi_k2_moe": "kimi-k2-moe",
        "kimi_k3_moe": "kimi-k3",
        "glm_moe": "glm-5.2",
        "deepseek_v3_moe": "deepseek-v3-moe",
        "minimax_moe": "minimax-moe",
        "mixtral_moe": "mixtral-moe",
    }
    for architecture, adapter_id in expected.items():
        config = next(item[1] for item in FAMILY_CONFIGS if item[2] == architecture)
        resolved = registry.resolve_model(_descriptor(dict(config)))
        assert resolved is not None
        assert resolved.adapter_id == adapter_id


def test_generic_product_layers_contain_no_model_family_dispatch() -> None:
    paths = (
        "src/swarm_inference/cluster/artifacts.py",
        "src/swarm_inference/cluster/orchestrator.py",
        "src/swarm_inference/coordinator/model_catalog.py",
        "src/swarm_inference/model/product.py",
        "src/swarm_inference/model/resolver.py",
        "src/swarm_inference/worker/agent.py",
        "src/swarm_inference/worker/execution.py",
        "src/swarm_inference/worker/shard_manager.py",
    )
    family_tokens = (
        "olmoe",
        "qwen",
        "kimi",
        "deepseek",
        "minimax",
        "mixtral",
        "mistral",
        "gemma",
        "glm",
    )
    violations = {
        path: token
        for path in paths
        for token in family_tokens
        if token in Path(path).read_text(encoding="utf-8").casefold()
    }
    assert violations == {}
