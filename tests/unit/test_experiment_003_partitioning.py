from __future__ import annotations

from pathlib import Path

from swarm_inference.model.adapter import (
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
)
from swarm_inference.model.shard_builder import build_manifest


def _qwen3_description(tmp_path: Path) -> ModelDescription:
    tensors = [
        TensorInfo(
            name="model.embed_tokens.weight",
            source_file="model.safetensors",
            dtype="BF16",
            shape=(128, 16),
            bytes=4096,
            component=ComponentRef(ComponentKind.EMBEDDING),
        ),
        TensorInfo(
            name="model.norm.weight",
            source_file="model.safetensors",
            dtype="BF16",
            shape=(16,),
            bytes=32,
            component=ComponentRef(ComponentKind.FINAL_NORM),
        ),
    ]
    for layer in range(28):
        tensors.append(
            TensorInfo(
                name=f"model.layers.{layer}.self_attn.q_proj.weight",
                source_file="model.safetensors",
                dtype="BF16",
                shape=(16, 16),
                bytes=512 + layer,
                component=ComponentRef(
                    ComponentKind.DECODER_LAYER,
                    layer_index=layer,
                ),
            )
        )
    return ModelDescription(
        model_id="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        model_path=tmp_path,
        config={
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_hidden_layers": 28,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 128,
            "max_position_embeddings": 512,
            "tie_word_embeddings": True,
        },
        tensors=tensors,
        source_file_hashes={"model.safetensors": "a" * 64},
    )


def test_every_arbitrary_stage_count_covers_qwen3_exactly_once(tmp_path: Path) -> None:
    description = _qwen3_description(tmp_path)
    source_names = {tensor.name for tensor in description.tensors}
    for count in range(1, 29):
        manifest = build_manifest(
            description,
            target_stage_bytes=100_000,
            maximum_stage_bytes=1_000_000,
            stage_count=count,
        )
        assert len(manifest.stages) == count
        assert all(stage.layer_end > stage.layer_start for stage in manifest.stages)
        layers = [
            layer
            for stage in manifest.stages
            for layer in range(stage.layer_start, stage.layer_end)
        ]
        assert layers == list(range(28))
        assert manifest.stages[0].owns_embeddings
        assert manifest.embedding_owner == 0
        assert manifest.stages[-1].owns_final_norm
        assert manifest.stages[-1].owns_output_head
        assert manifest.final_normalisation_owner == count - 1
        assert manifest.lm_head_owner == count - 1
        assert set(manifest.tensor_to_stages) == source_names
        assert all(stage.estimated_execution_ms for stage in manifest.stages)
    maximum = build_manifest(
        description,
        target_stage_bytes=100_000,
        maximum_stage_bytes=1_000_000,
        stage_count=28,
    )
    assert all(stage.layer_end - stage.layer_start == 1 for stage in maximum.stages)


def test_single_stage_tied_weight_is_not_declared_as_cross_stage_duplicate(
    tmp_path: Path,
) -> None:
    manifest = build_manifest(
        _qwen3_description(tmp_path),
        target_stage_bytes=100_000,
        maximum_stage_bytes=1_000_000,
        stage_count=1,
    )
    assert manifest.shared_tensors == {}
    assert manifest.total_sharded_weight_bytes == manifest.total_weight_bytes
    assert manifest.stages[0].owns_embeddings
    assert manifest.stages[0].owns_output_head
