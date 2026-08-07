from __future__ import annotations

import pytest

from swarm_inference.model.qwen3 import Qwen3Adapter, Qwen3StageModule
from swarm_inference.model.qwen3_runtime import Qwen3EngineOptions
from swarm_inference.model.shard_builder import (
    ResolvedModel,
    inspect_native_model,
    shard_model,
)
from swarm_inference.model.stage_module import (
    BatchExecutionMetadata,
    StageExecutionMetadata,
)

torch = pytest.importorskip("torch")


@pytest.fixture(params=[2, 4], ids=["two-stages", "four-stages"])
def tiny_qwen(tmp_path, request):
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
    description = inspect_native_model(resolved)
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
        stage_count=request.param,
    )
    assert len(manifest.stages) == request.param
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


def test_manual_graph_slots_reuse_position_buffers_and_retain_scrubbed_kv(
    tiny_qwen,
) -> None:
    _, config, _, manifest = tiny_qwen
    options = Qwen3EngineOptions.from_values(
        profile="qwen3_fast",
        attention_backend="sdpa",
        cache_backend="static",
        compile_mode="manual_cuda_graph",
        max_sequence_length=16,
        max_batch_size=1,
    )
    module = Qwen3StageModule(
        config=config,
        stage=manifest.stages[0],
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        engine_options=options,
    )
    module.model_revision = "immutable-test-revision"
    first = BatchExecutionMetadata(
        requests=(
            StageExecutionMetadata(
                request_id="graph-slot-a",
                token_position=0,
                sequence_length=1,
            ),
        )
    )
    second = BatchExecutionMetadata(
        requests=(
            StageExecutionMetadata(
                request_id="graph-slot-b",
                token_position=0,
                sequence_length=1,
            ),
        )
    )

    first_cache = module.begin_cuda_graph_decode(first)
    position_pointer = module._graph_position.data_ptr()
    second_cache = module.begin_cuda_graph_decode(second)

    assert first_cache is not second_cache
    assert module._graph_position.data_ptr() == position_pointer
    first_cache.prepare_append(token_position=0, query_length=1)
    first_cache.commit_append()
    logical_bytes = first_cache.used_bytes
    assert logical_bytes > 0
    assert module.reset_cuda_graph_slot("graph-slot-a") == logical_bytes
    assert first_cache.sequence_length == 0
    assert first_cache.deleted is False
    assert module.inspect_cache("graph-slot-a")[0]["reserved_bytes"] > 0


def test_stage_rotary_frequencies_follow_reference_cpu_initialisation(
    tiny_qwen, monkeypatch
) -> None:
    from transformers.models.qwen3 import modeling_qwen3

    _, config, _, manifest = tiny_qwen
    original = modeling_qwen3.Qwen3RotaryEmbedding
    construction_devices = []

    class RecordingRotaryEmbedding(original):
        def __init__(self, *args, device=None, **kwargs):
            construction_devices.append(device)
            super().__init__(*args, device=device, **kwargs)

    monkeypatch.setattr(
        modeling_qwen3,
        "Qwen3RotaryEmbedding",
        RecordingRotaryEmbedding,
    )
    Qwen3Adapter().create_stage_module(
        config,
        manifest.stages[0],
        torch.device("cpu"),
        torch.float32,
    )
    assert construction_devices == [None]


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


def test_tied_embedding_split_forward_matches_unsplit(tmp_path) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    model.tie_weights()
    source = tmp_path / "tied-source"
    model.save_pretrained(source, safe_serialization=True)
    description = inspect_native_model(
        ResolvedModel(
            model_id="tiny-tied-qwen3",
            revision="test",
            path=source,
            downloaded=False,
        )
    )
    output = tmp_path / "tied-shards"
    manifest = shard_model(
        description,
        output=output,
        target_stage_bytes=32 * 1024,
        maximum_stage_bytes=128 * 1024,
        maximum_output_file_bytes=1024 * 1024,
        stage_count=2,
    )
    assert manifest.shared_tensors == {"model.embed_tokens.weight": [0, 1]}
    modules = _split_modules(config, output, manifest)
    ids = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=True).logits
        current = ids
        for module in modules:
            current = module.forward(
                current,
                request_id="tied",
                token_position=0,
                use_cache=True,
            )
    assert torch.allclose(current, reference, atol=1e-5, rtol=1e-4)


def test_fast_batched_gpu_native_path_matches_reference_without_hot_reflection_or_numpy(
    tiny_qwen,
    monkeypatch,
) -> None:
    import swarm_inference.model.qwen3 as qwen3_module

    model, config, output, manifest = tiny_qwen
    options = Qwen3EngineOptions.from_values(
        profile="qwen3_fast",
        attention_backend="eager",
        cache_backend="static",
        max_sequence_length=16,
        max_batch_size=2,
        final_worker_sampling=True,
        boundary_diagnostics=False,
    )
    adapter = Qwen3Adapter()
    modules = []
    for stage in manifest.stages:
        module = Qwen3StageModule(
            config=config,
            stage=stage,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            engine_options=options,
        )
        adapter.load_stage_weights(
            module,
            output / f"stage-{stage.stage_id:03d}",
            manifest=manifest,
        )
        modules.append(module)

    def forbidden_reflection(*_args, **_kwargs):
        raise AssertionError("inspect.signature entered the measured decode path")

    monkeypatch.setattr(qwen3_module.inspect, "signature", forbidden_reflection)

    prompts = torch.tensor(
        [[1, 5, 9, 2], [4, 6, 8, 3]],
        dtype=torch.long,
    )
    request_ids = ("fast-a", "fast-b")
    output_tokens = 4
    fast_rows = torch.empty((2, output_tokens), dtype=torch.long)
    current = prompts
    prefill_metadata = BatchExecutionMetadata(
        requests=tuple(
            StageExecutionMetadata(
                request_id=request_id,
                token_position=0,
                sequence_length=prompts.shape[1],
            )
            for request_id in request_ids
        )
    )
    for module in modules:
        current = module.prefill_batch_cuda(current, prefill_metadata)
    sampled = modules[-1].sample_cuda(current, request_ids=request_ids)
    assert sampled.full_logits is None
    fast_rows[:, 0] = sampled.token_ids
    for output_index in range(1, output_tokens):
        current = fast_rows[:, output_index - 1 : output_index]
        metadata = BatchExecutionMetadata(
            requests=tuple(
                StageExecutionMetadata(
                    request_id=request_id,
                    token_position=prompts.shape[1] + output_index - 1,
                    sequence_length=1,
                )
                for request_id in request_ids
            )
        )
        for module in modules:
            current = module.decode_batch_cuda(current, metadata)
        sampled = modules[-1].sample_cuda(current, request_ids=request_ids)
        fast_rows[:, output_index] = sampled.token_ids

    reference_model = model.to(dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        reference = reference_model(input_ids=prompts, use_cache=True)
        reference_rows = torch.empty_like(fast_rows)
        reference_rows[:, 0] = torch.argmax(reference.logits[:, -1, :], dim=-1)
        cache = reference.past_key_values
        for output_index in range(1, output_tokens):
            reference = reference_model(
                input_ids=reference_rows[:, output_index - 1 : output_index],
                past_key_values=cache,
                use_cache=True,
            )
            cache = reference.past_key_values
            reference_rows[:, output_index] = torch.argmax(
                reference.logits[:, -1, :],
                dim=-1,
            )

    assert torch.equal(fast_rows, reference_rows)
    for module in modules:
        state = module.state_summary()
        assert state["fast_batch_forward_count"] == output_tokens
        assert state["caches"][0]["request_slots"] == {
            "fast-a": 0,
            "fast-b": 1,
        }
        module.cancel_batch(request_ids)
        assert module.state_summary()["cache_count"] == 0
    final_state = modules[-1].state_summary()
    assert final_state["full_logit_return_count"] == 0
    assert final_state["sampled_token_return_count"] == 2 * output_tokens


def test_fast_decode_method_has_no_reflection_numpy_or_explicit_synchronisation() -> None:
    import inspect

    source = inspect.getsource(Qwen3StageModule.decode_batch_cuda)
    forward_source = inspect.getsource(Qwen3StageModule._fast_forward_cuda)
    combined = source + forward_source
    assert "inspect.signature" not in combined
    assert "numpy" not in combined
    assert "np." not in combined
    assert ".synchronize(" not in combined
