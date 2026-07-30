from __future__ import annotations

import pytest

from swarm_inference.model.qwen3 import Qwen3Adapter
from swarm_inference.model.shard_builder import (
    ResolvedModel,
    inspect_qwen3_model,
    shard_model,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def tiny_qwen(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    source = tmp_path / "source"
    model.save_pretrained(source, safe_serialization=True)
    resolved = ResolvedModel(
        model_id="tiny-random-qwen3",
        revision="test",
        path=source,
        downloaded=False,
    )
    description = inspect_qwen3_model(resolved)
    average_layer = (
        sum(item.bytes for item in description.tensors if item.component.layer_index is not None)
        // 4
    )
    output = tmp_path / "shards"
    manifest = shard_model(
        description,
        output=output,
        target_stage_bytes=average_layer + 4096,
        maximum_stage_bytes=average_layer * 3,
        maximum_output_file_bytes=1024 * 1024,
    )
    return model, config, output, manifest


def _split_modules(config, output, manifest):
    adapter = Qwen3Adapter()
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
    return modules


def test_stage_created_from_serialised_config_selects_eager_attention(tiny_qwen) -> None:
    _, config, _, manifest = tiny_qwen
    serialised = config.to_dict()
    serialised.pop("_attn_implementation", None)
    module = Qwen3Adapter().create_stage_module(
        serialised,
        manifest.stages[0],
        torch.device("cpu"),
        torch.float32,
    )
    assert module.config._attn_implementation == "eager"


def test_no_stage_loads_full_model_and_tensor_union_is_exact(tiny_qwen) -> None:
    _, _, _, manifest = tiny_qwen
    assert len(manifest.stages) >= 2
    assert all(
        stage.required_memory_bytes < manifest.total_weight_bytes for stage in manifest.stages
    )
    assigned = [name for stage in manifest.stages for name in stage.tensor_names]
    duplicates = {name for name in assigned if assigned.count(name) > 1}
    assert duplicates == set(manifest.shared_tensors)


def test_split_prefill_decode_and_replay_match_unsplit(tiny_qwen) -> None:
    model, config, output, manifest = tiny_qwen
    modules = _split_modules(config, output, manifest)
    ids = torch.tensor([[1, 5, 9, 2]], dtype=torch.long)
    with torch.inference_mode():
        reference_prefill = model(input_ids=ids, use_cache=True)
        current = ids
        stage_inputs = []
        for module in modules:
            stage_inputs.append(current.detach().clone())
            current = module.forward(
                current,
                request_id="r",
                token_position=0,
                use_cache=True,
            )
        assert torch.allclose(current, reference_prefill.logits, atol=1e-5, rtol=1e-4)
        first = torch.argmax(current[:, -1, :], dim=-1, keepdim=True)
        reference_decode = model(
            input_ids=first,
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
        )

        adapter = Qwen3Adapter()
        replay_stage = min(1, len(modules) - 1)
        replacement = adapter.create_stage_module(
            config,
            manifest.stages[replay_stage],
            torch.device("cpu"),
            torch.float32,
        )
        adapter.load_stage_weights(
            replacement,
            output / f"stage-{replay_stage:03d}",
            manifest=manifest,
        )
        replacement.forward(
            stage_inputs[replay_stage],
            request_id="r",
            token_position=0,
            use_cache=True,
        )
        modules[replay_stage] = replacement
        current = first
        for module in modules:
            current = module.forward(
                current,
                request_id="r",
                token_position=ids.shape[1],
                use_cache=True,
            )
        assert torch.allclose(current, reference_decode.logits, atol=1e-5, rtol=1e-4)
        assert torch.equal(
            torch.argmax(current[:, -1, :], dim=-1),
            torch.argmax(reference_decode.logits[:, -1, :], dim=-1),
        )


def test_multiple_concurrent_prompt_caches_are_isolated(tiny_qwen) -> None:
    _, config, output, manifest = tiny_qwen
    modules = _split_modules(config, output, manifest)
    prompts = {
        "a": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "b": torch.tensor([[4, 5]], dtype=torch.long),
    }
    with torch.inference_mode():
        for request_id, ids in prompts.items():
            current = ids
            for module in modules:
                current = module.forward(
                    current,
                    request_id=request_id,
                    token_position=0,
                    use_cache=True,
                )
        assert all(module.state_summary()["cache_count"] == 2 for module in modules)
        for module in modules:
            module.cancel("a")
        assert all(module.state_summary()["cache_count"] == 1 for module in modules)
