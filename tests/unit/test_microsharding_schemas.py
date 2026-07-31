from __future__ import annotations

import pytest

from swarm_inference.config.models import CacheSpec, StageDefinition, TensorSpec
from swarm_inference.microsharding.schemas import (
    ModelPartitionPlan,
    TensorShard,
    attention_head_ownership,
    balanced_ranges,
    build_dense_partition_plan,
    validate_tensor_shard_union,
)


def _shard(rank: int, start: int, end: int, *, replicated: bool = False) -> TensorShard:
    return TensorShard(
        tensor_name="weight",
        global_shape=(10, 4),
        local_shape=(10, 4) if replicated else (end - start, 4),
        shard_axis=None if replicated else 0,
        shard_start=0 if replicated else start,
        shard_end=10 if replicated else end,
        rank=rank,
        world_size=2,
        dtype="F32",
        source_file="model.safetensors",
        source_tensor_hash="source-hash",
        local_tensor_hash=f"local-{rank}",
        replicated=replicated,
        logical_bytes=(10 if replicated else end - start) * 4 * 4,
    )


def test_balanced_ranges_support_uneven_dimensions_without_gaps() -> None:
    assert balanced_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert balanced_ranges(24, 4, alignment=3) == [(0, 6), (6, 12), (12, 18), (18, 24)]
    with pytest.raises(ValueError, match="not divisible"):
        balanced_ranges(10, 2, alignment=4)


def test_head_aligned_ownership_and_explicit_kv_replication() -> None:
    query, kv, replication = attention_head_ownership(
        query_heads=16,
        kv_heads=8,
        tensor_parallel_degree=8,
    )
    assert set().union(*map(set, query.values())) == set(range(16))
    assert set().union(*map(set, kv.values())) == set(range(8))
    assert replication == {}

    query, kv, replication = attention_head_ownership(
        query_heads=16,
        kv_heads=8,
        tensor_parallel_degree=16,
    )
    assert all(len(heads) == 1 for heads in query.values())
    assert all(len(ranks) == 2 for ranks in replication.values())
    assert kv[0] == kv[1] == [0]
    with pytest.raises(ValueError, match="requires"):
        attention_head_ownership(query_heads=12, kv_heads=5, tensor_parallel_degree=2)


def test_shard_union_detects_overlap_and_missing_range() -> None:
    assert validate_tensor_shard_union([_shard(0, 0, 5), _shard(1, 5, 10)])["status"] == "PASS"
    overlap = validate_tensor_shard_union([_shard(0, 0, 6), _shard(1, 5, 10)])
    assert overlap["status"] == "FAIL"
    assert any("overlap" in failure for failure in overlap["failures"])
    missing = validate_tensor_shard_union([_shard(0, 0, 4), _shard(1, 5, 10)])
    assert missing["status"] == "FAIL"
    assert any("missing range" in failure for failure in missing["failures"])


def test_replicated_tensor_accounting_requires_every_rank() -> None:
    passed = validate_tensor_shard_union(
        [_shard(0, 0, 10, replicated=True), _shard(1, 0, 10, replicated=True)]
    )
    assert passed["status"] == "PASS"
    failed = validate_tensor_shard_union([_shard(0, 0, 10, replicated=True)])
    assert failed["status"] == "FAIL"
    assert "replicated rank coverage mismatch" in failed["failures"][0]


def test_dense_plan_represents_more_workers_than_layers() -> None:
    plan = build_dense_partition_plan(
        model_id="Qwen/Qwen3-0.6B",
        model_revision="immutable",
        layer_count=28,
        hidden_size=1024,
        query_heads=16,
        kv_heads=8,
        head_dimension=128,
        intermediate_size=3072,
        pipeline_stage_count=4,
        tensor_parallel_degree=8,
        vocabulary_parallel=True,
    )
    assert plan.logical_pipeline_rank_workers == 32
    assert plan.logical_layer_shards == 224
    assert plan.metadata["partition_count_not_bounded_by_layer_count"] is True
    assert all(
        len(layer.collective_operations) == 2
        for stage in plan.pipeline_stages
        for layer in stage.layer_plans
    )


def test_legacy_stage_definitions_round_trip_into_hierarchy() -> None:
    spec = TensorSpec(dtype="float32", shape=["batch", "sequence", 32])
    cache = CacheSpec(bytes_per_token=1)
    stages = [
        StageDefinition(
            stage_id=0,
            layer_start=0,
            layer_end=2,
            input_spec=spec,
            output_spec=spec,
            owns_embeddings=True,
            owns_final_norm=False,
            owns_output_head=False,
            required_memory_bytes=1,
            cache_spec=cache,
        ),
        StageDefinition(
            stage_id=1,
            layer_start=2,
            layer_end=4,
            input_spec=spec,
            output_spec=spec,
            owns_embeddings=False,
            owns_final_norm=True,
            owns_output_head=True,
            required_memory_bytes=1,
            cache_spec=cache,
        ),
    ]
    plan = ModelPartitionPlan.from_stage_definitions(
        model_id="tiny",
        model_revision="test",
        layer_count=4,
        stages=stages,
        hidden_size=32,
        query_heads=4,
        kv_heads=2,
        head_dimension=8,
        intermediate_size=64,
    )
    assert plan.logical_pipeline_rank_workers == 2
    assert plan.metadata["converted_from_layer_only_stage_definitions"] is True
