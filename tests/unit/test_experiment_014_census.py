from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from swarm_inference.experiments.experiment_014.census import (
    CheckpointCensusError,
    _verify_payloads_resumable,
    classify_tensor,
)


@pytest.fixture
def k3_config() -> dict[str, object]:
    return {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
            "model_type": "kimi_linear",
            "architectures": ["KimiLinearForCausalLM"],
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "hidden_size": 7168,
            "routed_expert_hidden_size": 3584,
            "moe_intermediate_size": 3072,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "vocab_size": 163840,
            "max_position_embeddings": 1048576,
            "linear_attn_config": {
                "kda_layers": [1, 2, 3],
                "full_attn_layers": [4],
            },
        },
    }


def test_classifies_all_material_k3_tensor_families(k3_config: dict[str, object]) -> None:
    embedding = classify_tensor("language_model.model.embed_tokens.weight", k3_config)
    final_residual = classify_tensor("language_model.model.output_attn_res_proj.weight", k3_config)
    kda = classify_tensor("language_model.model.layers.0.self_attn.A_log", k3_config)
    mla = classify_tensor(
        "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight", k3_config
    )
    packed = classify_tensor(
        "language_model.model.layers.1.block_sparse_moe.experts.895.w2.weight_packed",
        k3_config,
    )
    scale = classify_tensor(
        "language_model.model.layers.1.block_sparse_moe.experts.895.w2.weight_scale",
        k3_config,
    )
    shared = classify_tensor(
        "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
        k3_config,
    )
    vision = classify_tensor("vision_tower.encoder.layers.0.weight", k3_config)

    assert embedding.role == "token_embedding"
    assert final_residual.role == "final_attention_residual_score_projection"
    assert kda.attention_type == "kda" and kda.role == "kda_log_decay"
    assert mla.attention_type == "gated_mla"
    assert packed.quantization_format == "mxfp4-e2m1"
    assert packed.quantization_block_size == 32
    assert packed.related_tensor == scale.related_tensor.replace("_packed", "_scale")
    assert scale.quantization_format == "ue8m0"
    assert shared.shared_expert == "fused_shared_experts"
    assert vision.required_for_text_generation is False


def test_unknown_required_text_tensor_fails_closed(k3_config: dict[str, object]) -> None:
    with pytest.raises(CheckpointCensusError, match="unclassified required text tensor"):
        classify_tensor("language_model.model.layers.0.self_attn.mystery.weight", k3_config)


def test_attention_layer_lists_must_cover_the_model(k3_config: dict[str, object]) -> None:
    invalid = deepcopy(k3_config)
    text = invalid["text_config"]
    assert isinstance(text, dict)
    linear = text["linear_attn_config"]
    assert isinstance(linear, dict)
    linear["full_attn_layers"] = []
    with pytest.raises(CheckpointCensusError, match="cover every layer"):
        classify_tensor("language_model.model.layers.0.self_attn.A_log", invalid)


def test_payload_hash_receipt_is_atomic_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    checkpoint = tmp_path / "checkpoint"
    metadata = checkpoint / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    shard_names = ("model-00001.safetensors", "model-00002.safetensors")
    for index, name in enumerate(shard_names):
        payload = bytes([index + 1]) * 4096
        (checkpoint / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (metadata / f"{name}.metadata").write_text(f"{revision}\n{digest}\n0\n", encoding="utf-8")
    receipt = tmp_path / "integrity.json"
    first = _verify_payloads_resumable(
        checkpoint=checkpoint,
        shard_names=shard_names,
        revision=revision,
        config_sha256="b" * 64,
        index_sha256="c" * 64,
        receipt_path=receipt,
        workers=2,
    )
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["complete"] is True
    assert saved["verified_shard_count"] == 2
    assert set(first) == set(shard_names)

    def unexpected_hash(_path: Path) -> str:
        raise AssertionError("a matching receipt should prevent a second payload read")

    monkeypatch.setattr(
        "swarm_inference.experiments.experiment_014.census._sha256", unexpected_hash
    )
    resumed = _verify_payloads_resumable(
        checkpoint=checkpoint,
        shard_names=shard_names,
        revision=revision,
        config_sha256="b" * 64,
        index_sha256="c" * 64,
        receipt_path=receipt,
        workers=2,
    )
    assert resumed == first
