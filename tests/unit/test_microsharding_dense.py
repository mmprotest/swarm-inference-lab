from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from swarm_inference.microsharding.builder import (
    build_microshards_from_description,
    validate_microshards,
)
from swarm_inference.microsharding.dense import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelQwenModel,
    distributed_argmax,
)
from swarm_inference.model.shard_builder import ResolvedModel, inspect_qwen3_model


@pytest.fixture(scope="module")
def tiny_microshards(tmp_path_factory: pytest.TempPathFactory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    root = tmp_path_factory.mktemp("microshards")
    torch.manual_seed(6006)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_dropout=0,
        rms_norm_eps=1e-6,
    )
    config._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(config).eval()
    reference.tie_weights()
    source = root / "source"
    reference.save_pretrained(source, safe_serialization=True)
    description = inspect_qwen3_model(
        ResolvedModel(
            model_id="tiny-qwen3-microsharding",
            revision="immutable-test",
            path=source,
            downloaded=False,
        )
    )
    paths: dict[tuple[int, int], Path] = {}
    for pipeline in (1, 4):
        for degree in (1, 2, 4, 8):
            output = root / f"pp{pipeline}-tp{degree}"
            result = build_microshards_from_description(
                description,
                pipeline_stage_count=pipeline,
                tensor_parallel_degree=degree,
                output=output,
                vocabulary_parallel=True,
            )
            assert result.validation["status"] == "PASS"
            paths[(pipeline, degree)] = output
    return reference, source, paths


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_column_parallel_linear_concatenation_matches_full(degree: int) -> None:
    torch.manual_seed(10)
    value = torch.randn(3, 16)
    weight = torch.randn(32, 16)
    bias = torch.randn(32)
    outputs = []
    for rank in range(degree):
        start = rank * (32 // degree)
        end = (rank + 1) * (32 // degree)
        module = ColumnParallelLinear(
            16,
            end - start,
            global_out_features=32,
            shard_start=start,
            shard_end=end,
            bias=True,
            dtype=torch.float32,
        )
        module.load_local(weight[start:end], bias[start:end])
        outputs.append(module(value))
    assert torch.allclose(
        torch.cat(outputs, dim=-1), torch.nn.functional.linear(value, weight, bias)
    )


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_row_parallel_linear_sum_and_single_bias_match_full(degree: int) -> None:
    torch.manual_seed(11)
    value = torch.randn(3, 32)
    weight = torch.randn(16, 32)
    bias = torch.randn(16)
    partials = []
    for rank in range(degree):
        start = rank * (32 // degree)
        end = (rank + 1) * (32 // degree)
        module = RowParallelLinear(
            end - start,
            16,
            global_in_features=32,
            shard_start=start,
            shard_end=end,
            bias=True,
            apply_bias=rank == 0,
            dtype=torch.float32,
        )
        module.load_local(weight[:, start:end], bias)
        partials.append(module(value[:, start:end]))
    actual = torch.stack(partials).sum(dim=0)
    expected = torch.nn.functional.linear(value, weight, bias)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_row_parallel_rejects_invalid_local_shape() -> None:
    module = RowParallelLinear(
        4,
        3,
        global_in_features=8,
        shard_start=0,
        shard_end=4,
        dtype=torch.float32,
    )
    with pytest.raises(Exception, match="shape"):
        module.load_local(torch.randn(3, 5))


def test_distributed_argmax_uses_lowest_token_for_exact_tie() -> None:
    value, token = distributed_argmax(
        [
            (torch.tensor([3.0]), torch.tensor([9])),
            (torch.tensor([3.0]), torch.tensor([4])),
            (torch.tensor([2.0]), torch.tensor([1])),
        ]
    )
    assert value.item() == 3.0
    assert token.item() == 4


@pytest.mark.parametrize(
    ("pipeline", "degree"),
    [(1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4), (4, 8)],
)
def test_full_tiny_model_exact_greedy_and_cache_cleanup(
    tiny_microshards, pipeline: int, degree: int
) -> None:
    reference, _, paths = tiny_microshards
    ids = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference.generate(ids, max_new_tokens=5, do_sample=False)[0, ids.shape[1] :]
    model = TensorParallelQwenModel(paths[(pipeline, degree)], device="cpu", dtype=torch.float32)
    actual = model.generate(ids, max_new_tokens=5, request_id="exact")
    assert actual == expected.tolist()
    assert model.cache.inspect("exact") == []
    assert model.validate_rank_local_matrices()["status"] == "PASS"
    assert sum(event.get("operation") == "all_reduce_sum" for event in model.collective_trace) > 0


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_full_decoder_boundaries_match_reference(tiny_microshards, degree: int) -> None:
    reference, _, paths = tiny_microshards
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference(input_ids=ids, use_cache=False, output_hidden_states=True)
    model = TensorParallelQwenModel(paths[(1, degree)], device="cpu", dtype=torch.float32)
    actual, captures = model.forward_hidden(
        ids,
        request_id="boundary",
        position_start=0,
        capture_layers={0, 2, 3},
    )
    assert torch.allclose(actual, expected.hidden_states[-1], atol=2e-4, rtol=2e-4)
    assert set(captures) == {0, 2, 3}
    assert all(capture.final_hidden.shape == (1, 5, 32) for capture in captures.values())
    model.cache.cleanup("boundary")


def test_kv_head_replication_is_explicit_and_cache_bytes_are_separate(tiny_microshards) -> None:
    _, _, paths = tiny_microshards
    model = TensorParallelQwenModel(paths[(1, 8)], device="cpu", dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model.forward_hidden(ids, request_id="kv", position_start=0)
    validation = model.cache.validate_ownership(
        request_id="kv",
        layer_id=2,
        global_kv_head_count=2,
    )
    assert validation["status"] == "PASS"
    assert set(validation["replicated_heads"]) == {0, 1}
    assert validation["actual_bytes"] > validation["unique_bytes"]
    assert validation["replicated_bytes"] > 0
    assert all(record["layer_id"] >= 0 for record in model.cache.inspect("kv"))
    model.cache.cleanup("kv")


def test_microshard_hash_validation_detects_modified_rank_file(
    tiny_microshards, tmp_path: Path
) -> None:
    _, source, paths = tiny_microshards
    corrupted = tmp_path / "corrupted"
    shutil.copytree(paths[(1, 2)], corrupted)
    weight_path = corrupted / "ranks" / "rank-000" / "weights.safetensors"
    state = load_file(weight_path)
    name = sorted(state)[0]
    state[name] = state[name].clone()
    state[name].view(-1)[0] += 1
    save_file(state, weight_path)
    validation = validate_microshards(corrupted, source_model=source)
    assert validation["status"] == "FAIL"
    assert any("hash mismatch" in failure for failure in validation["failures"])
