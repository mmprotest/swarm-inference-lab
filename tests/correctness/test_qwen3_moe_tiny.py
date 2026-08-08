from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.qwen3 import Qwen3StageModule
from swarm_inference.model.qwen3_moe import Qwen3MoeAdapter
from swarm_inference.model.shard_builder import shard_model

torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_qwen3_moe(tmp_path: Path):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(29)
    model = transformers.Qwen3MoeForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)

    adapter = Qwen3MoeAdapter()
    description = adapter.describe(
        source,
        model_id="tiny-qwen3-moe",
        model_revision="fixture-revision",
    )
    total_weight_bytes = sum(item.bytes for item in description.tensors)
    output = tmp_path / "stages"
    manifest = shard_model(
        description,
        output=output,
        target_stage_bytes=max(1, total_weight_bytes // 2),
        maximum_stage_bytes=total_weight_bytes,
        maximum_output_file_bytes=1024 * 1024,
        stage_count=2,
    )
    modules = []
    for stage in manifest.stages:
        module = adapter.create_stage_module(
            config,
            stage,
            torch.device("cpu"),
            torch.float32,
        )
        adapter.load_stage_weights(
            module,
            output / f"stage-{stage.stage_id:03d}",
            manifest=manifest,
        )
        modules.append(module)
    return model, adapter, source, output, manifest, modules


def test_qwen3_moe_stage_ownership_keeps_experts_with_their_layer(
    tiny_qwen3_moe,
) -> None:
    _, _, _, _, manifest, _ = tiny_qwen3_moe

    assert [tuple(range(stage.layer_start, stage.layer_end)) for stage in manifest.stages] == [
        (0,),
        (1,),
    ]
    all_names = [name for stage in manifest.stages for name in stage.tensor_names]
    assert len(all_names) == len(set(all_names))
    for stage in manifest.stages:
        for name in stage.tensor_names:
            if ".mlp.experts." in name or ".mlp.gate." in name:
                assert any(
                    name.startswith(f"model.layers.{layer_id}.")
                    for layer_id in range(stage.layer_start, stage.layer_end)
                )


def test_qwen3_moe_stage_ring_matches_reference_greedy_and_isolates_kv(
    tiny_qwen3_moe,
) -> None:
    model, _, _, _, _, modules = tiny_qwen3_moe
    prompt = torch.tensor([[1, 5, 9]], dtype=torch.long)

    with torch.inference_mode():
        reference = model(input_ids=prompt, use_cache=True).logits
        current = prompt
        stage_shapes = []
        for module in modules:
            current = module.forward(
                current,
                request_id="request-a",
                token_position=0,
                use_cache=True,
            )
            stage_shapes.append(tuple(current.shape))

        assert stage_shapes[0] == (1, prompt.shape[1], model.config.hidden_size)
        assert stage_shapes[1] == (1, prompt.shape[1], model.config.vocab_size)
        assert torch.allclose(current, reference, atol=1e-5, rtol=1e-4)
        assert torch.equal(
            torch.argmax(current[:, -1, :], dim=-1),
            torch.argmax(reference[:, -1, :], dim=-1),
        )

        second_prompt = torch.tensor([[2, 4]], dtype=torch.long)
        current = second_prompt
        for module in modules:
            current = module.forward(
                current,
                request_id="request-b",
                token_position=0,
                use_cache=True,
            )
        assert all(module.state_summary()["cache_count"] == 2 for module in modules)
        for module in modules:
            module.cancel("request-a")
        assert all(module.state_summary()["cache_count"] == 1 for module in modules)


def test_qwen3_moe_adapter_validates_a_delegated_expert_component(
    tiny_qwen3_moe,
) -> None:
    _, adapter, source, _, manifest, _ = tiny_qwen3_moe
    stage = manifest.stages[0]
    remote_experts = {(0, 0)}
    description = adapter.describe(
        source,
        model_id="tiny-qwen3-moe",
        model_revision="fixture-revision",
    )
    delegated_bytes = sum(
        tensor.bytes
        for tensor in description.tensors
        if tensor.name.startswith("model.layers.0.mlp.experts.0.")
    )
    assert delegated_bytes > 0
    assignment = StageAssignment(
        stage_id=stage.stage_id,
        layer_start=stage.layer_start,
        layer_end=stage.layer_end,
        layer_ids=tuple(range(stage.layer_start, stage.layer_end)),
        weight_bytes=stage.required_memory_bytes - delegated_bytes,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=1,
        peak_temporary_bytes=0,
        activation_bytes=manifest.activation_bytes_per_stage_boundary,
        device="cpu",
        owns_embeddings=stage.owns_embeddings,
        owns_final_norm=stage.owns_final_norm,
        owns_output_projection=stage.owns_output_head,
    )
    metadata = adapter.validate_stage_assignment(
        source,
        assignment=assignment,
        stage_count=2,
        model_revision="fixture-revision",
        tokenizer_revision="fixture-tokenizer",
        remote_experts=remote_experts,
    )

    assert metadata.expert_count == 4
    selected = tuple(stage.tensor_names)
    excluded = adapter.exclude_delegated_tensor_names(selected, remote_experts)
    assert excluded
    assert all(name.startswith("model.layers.0.mlp.experts.0.") for name in excluded)


def test_qwen3_moe_colibri_component_matches_reference_logits_and_tokens(
    tiny_qwen3_moe,
) -> None:
    from swarm_inference.backends.colibri.torch_backend import ColibriMoeBackend
    from swarm_inference.execution.moe import HybridMoeBackend

    model, adapter, _, output, manifest, _ = tiny_qwen3_moe
    modules: list[Qwen3StageModule] = []
    component_backends: list[ColibriMoeBackend] = []
    for stage in manifest.stages:
        delegated = {
            (layer_id, expert_id)
            for layer_id in range(stage.layer_start, stage.layer_end)
            for expert_id in range(model.config.num_experts)
        }

        def backend_factory(
            _local_experts: dict[tuple[int, int], Any],
            *,
            artifact: Path = output / f"stage-{stage.stage_id:03d}",
            placements: set[tuple[int, int]] = delegated,
        ) -> HybridMoeBackend:
            colibri = ColibriMoeBackend(
                model_path=artifact,
                model_id="tiny-qwen3-moe",
                model_revision="fixture-revision",
                selected_experts=placements,
                device="cpu",
                cache_budget_bytes=1024**2,
            )
            component_backends.append(colibri)
            return HybridMoeBackend(
                colibri=colibri,
                placement={key: "colibri" for key in placements},
            )

        module = Qwen3StageModule(
            config=model.config,
            stage=stage,
            device=torch.device("cpu"),
            dtype=torch.float32,
            moe_backend_factory=backend_factory,
            remote_experts=delegated,
        )
        adapter.load_stage_weights(
            module,
            output / f"stage-{stage.stage_id:03d}",
            manifest=manifest,
        )
        module.open_expert_session("colibri-request")
        modules.append(module)

    prompt = torch.tensor([[1, 5, 9]], dtype=torch.long)
    with torch.inference_mode():
        reference = model(input_ids=prompt, use_cache=True).logits
        current = prompt
        for module in modules:
            current = module.forward(
                current,
                request_id="colibri-request",
                token_position=0,
                use_cache=True,
            )

    assert torch.allclose(current, reference, atol=1e-5, rtol=1e-4)
    assert torch.equal(
        torch.argmax(current[:, -1, :], dim=-1),
        torch.argmax(reference[:, -1, :], dim=-1),
    )
    assert all(backend.status()["colibri_expert_calls"] > 0 for backend in component_backends)
    assert all(
        backend.status()["coordinator_activation_bytes"] == 0
        for backend in component_backends
    )
    assert all(backend.status()["colibri_cache_misses"] > 0 for backend in component_backends)
    for module in modules:
        module.close_expert_session("colibri-request")
