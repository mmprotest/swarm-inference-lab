"""Typed hierarchical partition and collective schemas for Experiment 006."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import Field, model_validator

from swarm_inference.config.models import StageDefinition, StrictModel, TensorSpec

CollectiveOperation = Literal[
    "broadcast",
    "all_reduce_sum",
    "all_gather",
    "reduce_scatter_sum",
    "all_to_all",
    "gather_to_leader",
    "distributed_argmax",
    "barrier",
]
CollectiveAlgorithm = Literal[
    "ring",
    "binary_tree",
    "recursive_doubling",
    "leader_gather_broadcast",
]


class TensorShard(StrictModel):
    """A precise rank-local view of one checkpoint tensor."""

    tensor_name: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    shard_axis: int | None
    shard_start: int = Field(ge=0)
    shard_end: int = Field(ge=0)
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    dtype: str
    source_file: str
    source_tensor_hash: str
    local_tensor_hash: str
    replicated: bool
    partition_mode: str = "tensor_parallel"
    logical_bytes: int = Field(ge=0)
    stage_id: int = Field(default=0, ge=0)
    logical_rank_id: str | None = None

    @model_validator(mode="after")
    def validate_slice(self) -> TensorShard:
        if not self.global_shape:
            raise ValueError("global_shape cannot be empty")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.replicated:
            if self.shard_axis is not None:
                raise ValueError("replicated tensors cannot declare a shard axis")
            if self.local_shape != self.global_shape:
                raise ValueError("replicated tensor local_shape must equal global_shape")
            if (self.shard_start, self.shard_end) != (0, self.global_shape[0]):
                raise ValueError("replicated tensor range must describe the full leading axis")
            return self
        if self.shard_axis is None:
            raise ValueError("non-replicated tensors require a shard axis")
        if not 0 <= self.shard_axis < len(self.global_shape):
            raise ValueError("shard_axis is outside the tensor rank")
        if not 0 <= self.shard_start < self.shard_end <= self.global_shape[self.shard_axis]:
            raise ValueError("invalid shard interval")
        expected = list(self.global_shape)
        expected[self.shard_axis] = self.shard_end - self.shard_start
        if tuple(expected) != self.local_shape:
            raise ValueError("local_shape does not match the declared slice")
        return self


class CollectivePlan(StrictModel):
    collective_id: str
    operation: CollectiveOperation
    group_id: str
    rank_ids: list[str]
    tensor_spec: TensorSpec
    algorithm: CollectiveAlgorithm = "ring"
    compression: str = "bfloat16"
    timeout_ms: int = Field(default=30_000, gt=0)
    exactness: str = "exact"
    layer_id: int | None = None
    phase: str | None = None

    @model_validator(mode="after")
    def validate_group(self) -> CollectivePlan:
        if not self.rank_ids:
            raise ValueError("collective rank_ids cannot be empty")
        if len(self.rank_ids) != len(set(self.rank_ids)):
            raise ValueError("collective rank_ids must be unique")
        return self


class CollectiveGroupPlan(StrictModel):
    group_id: str
    rank_ids: list[str]
    backend: str
    deterministic_rank_order: bool = True

    @model_validator(mode="after")
    def validate_ranks(self) -> CollectiveGroupPlan:
        if not self.rank_ids or len(self.rank_ids) != len(set(self.rank_ids)):
            raise ValueError("collective group ranks must be non-empty and unique")
        return self


class AttentionPartitionPlan(StrictModel):
    tensor_parallel_degree: int = Field(gt=0)
    query_head_ownership: dict[int, list[int]]
    kv_head_ownership: dict[int, list[int]]
    kv_replication_groups: dict[int, list[int]] = Field(default_factory=dict)
    head_dimension: int = Field(gt=0)
    rotary_dimension: int = Field(gt=0)
    grouped_query_ratio: int = Field(gt=0)
    q_projection_name: str
    k_projection_name: str
    v_projection_name: str
    output_projection_name: str


class DenseMLPPartitionPlan(StrictModel):
    tensor_parallel_degree: int = Field(gt=0)
    intermediate_ranges: dict[int, tuple[int, int]]
    gate_projection_name: str
    up_projection_name: str
    down_projection_name: str


class MoEPartitionPlan(StrictModel):
    layer_id: int = Field(ge=0)
    mode: Literal["expert_parallel", "expert_tensor_parallel", "replicated_reference"]
    router_mode: str
    router_owner_rank: int | None
    router_replicated: bool
    top_k: int = Field(gt=0)
    expert_parallel_degree: int = Field(gt=0)
    expert_tensor_parallel_degree: int = Field(gt=0)
    expert_ownership: dict[int, list[int]]
    shared_expert_plan: DenseMLPPartitionPlan | None = None
    collective_operations: list[CollectivePlan] = Field(default_factory=list)


class LayerPartitionPlan(StrictModel):
    layer_id: int = Field(ge=0)
    attention: AttentionPartitionPlan
    mlp: DenseMLPPartitionPlan | None
    moe: MoEPartitionPlan | None
    replicated_tensor_names: list[str] = Field(default_factory=list)
    collective_operations: list[CollectivePlan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_feed_forward(self) -> LayerPartitionPlan:
        if (self.mlp is None) == (self.moe is None):
            raise ValueError("a layer must contain exactly one of dense MLP or MoE")
        return self


class PipelineStagePlan(StrictModel):
    stage_id: int = Field(ge=0)
    layer_plans: list[LayerPartitionPlan]
    parallel_cell_id: str
    input_spec: TensorSpec
    output_spec: TensorSpec
    owns_embeddings: bool
    owns_final_norm: bool
    owns_lm_head: bool

    @model_validator(mode="after")
    def validate_layers(self) -> PipelineStagePlan:
        if not self.layer_plans:
            raise ValueError("pipeline stage must own at least one layer")
        layer_ids = [item.layer_id for item in self.layer_plans]
        if layer_ids != list(range(layer_ids[0], layer_ids[-1] + 1)):
            raise ValueError("pipeline stage layers must be contiguous")
        return self


class ParallelCellPlan(StrictModel):
    cell_id: str
    rank_ids: list[str]
    pipeline_stage_ids: list[int]
    expected_network_class: str
    tensor_parallel_degree: int = Field(gt=0)
    expert_parallel_degree: int = Field(gt=0)
    collective_backend: str

    @model_validator(mode="after")
    def validate_cell(self) -> ParallelCellPlan:
        if not self.rank_ids or len(self.rank_ids) != len(set(self.rank_ids)):
            raise ValueError("parallel cell ranks must be non-empty and unique")
        if not self.pipeline_stage_ids:
            raise ValueError("parallel cell must expose at least one pipeline stage")
        return self


class ModelPartitionPlan(StrictModel):
    schema_version: str = "experiment-006-v1"
    model_id: str
    model_revision: str
    pipeline_stages: list[PipelineStagePlan]
    tensor_shards: list[TensorShard]
    collective_groups: list[CollectiveGroupPlan]
    parallel_cells: list[ParallelCellPlan]
    layer_count: int = Field(gt=0)
    vocabulary_parallel: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hierarchy(self) -> ModelPartitionPlan:
        ordered = sorted(self.pipeline_stages, key=lambda item: item.stage_id)
        if [item.stage_id for item in ordered] != list(range(len(ordered))):
            raise ValueError("pipeline stage IDs must be contiguous from zero")
        layer_ids = [layer.layer_id for stage in ordered for layer in stage.layer_plans]
        if layer_ids != list(range(self.layer_count)):
            raise ValueError("pipeline stages must cover every layer exactly once")
        cells = {item.cell_id: item for item in self.parallel_cells}
        if set(cells) != {stage.parallel_cell_id for stage in ordered}:
            raise ValueError("every and only referenced parallel cells must be declared")
        groups = {item.group_id: item for item in self.collective_groups}
        for stage in ordered:
            cell = cells[stage.parallel_cell_id]
            if stage.stage_id not in cell.pipeline_stage_ids:
                raise ValueError("parallel cell omits one of its pipeline stages")
            for layer in stage.layer_plans:
                for collective in layer.collective_operations:
                    if collective.group_id not in groups:
                        raise ValueError(f"unknown collective group {collective.group_id}")
                    if collective.rank_ids != groups[collective.group_id].rank_ids:
                        raise ValueError("collective rank order differs from its group")
        return self

    @property
    def logical_pipeline_rank_workers(self) -> int:
        return sum(
            len(self.parallel_cell(stage.parallel_cell_id).rank_ids)
            for stage in self.pipeline_stages
        )

    @property
    def logical_layer_shards(self) -> int:
        return sum(
            layer.attention.tensor_parallel_degree
            for stage in self.pipeline_stages
            for layer in stage.layer_plans
        )

    def parallel_cell(self, cell_id: str) -> ParallelCellPlan:
        for cell in self.parallel_cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)

    @classmethod
    def from_stage_definitions(
        cls,
        *,
        model_id: str,
        model_revision: str,
        layer_count: int,
        stages: list[StageDefinition],
        hidden_size: int,
        query_heads: int,
        kv_heads: int,
        head_dimension: int,
        intermediate_size: int,
    ) -> ModelPartitionPlan:
        """Represent the legacy layer-only plan as TP1 hierarchical cells."""

        if query_heads % kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        pipeline: list[PipelineStagePlan] = []
        cells: list[ParallelCellPlan] = []
        groups: list[CollectiveGroupPlan] = []
        for legacy in sorted(stages, key=lambda item: item.stage_id):
            rank_id = f"stage-{legacy.stage_id:03d}-rank-000"
            cell_id = f"cell-{legacy.stage_id:03d}"
            group_id = f"tp-stage-{legacy.stage_id:03d}"
            groups.append(
                CollectiveGroupPlan(group_id=group_id, rank_ids=[rank_id], backend="legacy")
            )
            layers = [
                _dense_layer_plan(
                    layer_id=layer_id,
                    stage_id=legacy.stage_id,
                    rank_ids=[rank_id],
                    hidden_size=hidden_size,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dimension=head_dimension,
                    intermediate_size=intermediate_size,
                )
                for layer_id in range(legacy.layer_start, legacy.layer_end)
            ]
            pipeline.append(
                PipelineStagePlan(
                    stage_id=legacy.stage_id,
                    layer_plans=layers,
                    parallel_cell_id=cell_id,
                    input_spec=legacy.input_spec,
                    output_spec=legacy.output_spec,
                    owns_embeddings=legacy.owns_embeddings,
                    owns_final_norm=legacy.owns_final_norm,
                    owns_lm_head=legacy.owns_output_head,
                )
            )
            cells.append(
                ParallelCellPlan(
                    cell_id=cell_id,
                    rank_ids=[rank_id],
                    pipeline_stage_ids=[legacy.stage_id],
                    expected_network_class="same_gpu_logical",
                    tensor_parallel_degree=1,
                    expert_parallel_degree=1,
                    collective_backend="legacy",
                )
            )
        return cls(
            model_id=model_id,
            model_revision=model_revision,
            pipeline_stages=pipeline,
            tensor_shards=[],
            collective_groups=groups,
            parallel_cells=cells,
            layer_count=layer_count,
            metadata={"converted_from_layer_only_stage_definitions": True},
        )


def balanced_ranges(size: int, parts: int, *, alignment: int = 1) -> list[tuple[int, int]]:
    """Split a dimension without gaps; aligned mode rejects impossible shapes."""

    if size <= 0 or parts <= 0 or parts > size:
        raise ValueError("size and parts must be positive, with parts <= size")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    if size % alignment:
        raise ValueError(f"dimension {size} is not divisible by alignment {alignment}")
    units = size // alignment
    if parts > units:
        raise ValueError("partition would split an aligned unit")
    quotient, remainder = divmod(units, parts)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for rank in range(parts):
        width = quotient + (1 if rank < remainder else 0)
        end = cursor + width * alignment
        ranges.append((cursor, end))
        cursor = end
    if cursor != size:
        raise AssertionError("balanced range construction lost values")
    return ranges


def attention_head_ownership(
    *, query_heads: int, kv_heads: int, tensor_parallel_degree: int
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """Return query/KV ownership and explicit KV replication groups."""

    if query_heads % kv_heads:
        raise ValueError("Qwen grouped-query attention requires query_heads % kv_heads == 0")
    query_ranges = balanced_ranges(query_heads, tensor_parallel_degree, alignment=1)
    query: dict[int, list[int]] = {}
    kv: dict[int, list[int]] = {}
    users: dict[int, list[int]] = defaultdict(list)
    ratio = query_heads // kv_heads
    for rank, (start, end) in enumerate(query_ranges):
        query[rank] = list(range(start, end))
        required = sorted({head // ratio for head in query[rank]})
        kv[rank] = required
        for head in required:
            users[head].append(rank)
    replication = {head: ranks for head, ranks in users.items() if len(ranks) > 1}
    return query, kv, replication


def _dense_layer_plan(
    *,
    layer_id: int,
    stage_id: int,
    rank_ids: list[str],
    hidden_size: int,
    query_heads: int,
    kv_heads: int,
    head_dimension: int,
    intermediate_size: int,
) -> LayerPartitionPlan:
    degree = len(rank_ids)
    query, kv, replication = attention_head_ownership(
        query_heads=query_heads,
        kv_heads=kv_heads,
        tensor_parallel_degree=degree,
    )
    prefix = f"model.layers.{layer_id}"
    group_id = f"tp-stage-{stage_id:03d}"
    spec = TensorSpec(dtype="bfloat16", shape=["batch", "sequence", hidden_size])
    attention_collective = CollectivePlan(
        collective_id=f"layer-{layer_id:03d}-attention-reduce",
        operation="all_reduce_sum",
        group_id=group_id,
        rank_ids=rank_ids,
        tensor_spec=spec,
        layer_id=layer_id,
        phase="attention_output",
    )
    mlp_collective = CollectivePlan(
        collective_id=f"layer-{layer_id:03d}-mlp-reduce",
        operation="all_reduce_sum",
        group_id=group_id,
        rank_ids=rank_ids,
        tensor_spec=spec,
        layer_id=layer_id,
        phase="mlp_output",
    )
    return LayerPartitionPlan(
        layer_id=layer_id,
        attention=AttentionPartitionPlan(
            tensor_parallel_degree=degree,
            query_head_ownership=query,
            kv_head_ownership=kv,
            kv_replication_groups=replication,
            head_dimension=head_dimension,
            rotary_dimension=head_dimension,
            grouped_query_ratio=query_heads // kv_heads,
            q_projection_name=f"{prefix}.self_attn.q_proj.weight",
            k_projection_name=f"{prefix}.self_attn.k_proj.weight",
            v_projection_name=f"{prefix}.self_attn.v_proj.weight",
            output_projection_name=f"{prefix}.self_attn.o_proj.weight",
        ),
        mlp=DenseMLPPartitionPlan(
            tensor_parallel_degree=degree,
            intermediate_ranges={
                rank: interval
                for rank, interval in enumerate(balanced_ranges(intermediate_size, degree))
            },
            gate_projection_name=f"{prefix}.mlp.gate_proj.weight",
            up_projection_name=f"{prefix}.mlp.up_proj.weight",
            down_projection_name=f"{prefix}.mlp.down_proj.weight",
        ),
        moe=None,
        replicated_tensor_names=[
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.q_norm.weight",
            f"{prefix}.self_attn.k_norm.weight",
        ],
        collective_operations=[attention_collective, mlp_collective],
    )


def build_dense_partition_plan(
    *,
    model_id: str,
    model_revision: str,
    layer_count: int,
    hidden_size: int,
    query_heads: int,
    kv_heads: int,
    head_dimension: int,
    intermediate_size: int,
    pipeline_stage_count: int,
    tensor_parallel_degree: int,
    vocabulary_parallel: bool,
    dtype: str = "bfloat16",
) -> ModelPartitionPlan:
    """Build the hierarchy before source tensor slices are attached."""

    if not 1 <= pipeline_stage_count <= layer_count:
        raise ValueError("pipeline stage count must be between one and layer count")
    # Head ranges must never split a query head.
    attention_head_ownership(
        query_heads=query_heads,
        kv_heads=kv_heads,
        tensor_parallel_degree=tensor_parallel_degree,
    )
    intermediate_ranges = balanced_ranges(intermediate_size, tensor_parallel_degree)
    stage_ranges = balanced_ranges(layer_count, pipeline_stage_count)
    pipeline: list[PipelineStagePlan] = []
    groups: list[CollectiveGroupPlan] = []
    cells: list[ParallelCellPlan] = []
    boundary_spec = TensorSpec(dtype=dtype, shape=["batch", "sequence", hidden_size])
    for stage_id, (layer_start, layer_end) in enumerate(stage_ranges):
        rank_ids = [
            f"stage-{stage_id:03d}-rank-{rank:03d}" for rank in range(tensor_parallel_degree)
        ]
        cell_id = f"cell-{stage_id:03d}"
        group_id = f"tp-stage-{stage_id:03d}"
        groups.append(
            CollectiveGroupPlan(
                group_id=group_id,
                rank_ids=rank_ids,
                backend="single_device_logical",
            )
        )
        cells.append(
            ParallelCellPlan(
                cell_id=cell_id,
                rank_ids=rank_ids,
                pipeline_stage_ids=[stage_id],
                expected_network_class="same_gpu_logical",
                tensor_parallel_degree=tensor_parallel_degree,
                expert_parallel_degree=1,
                collective_backend="single_device_logical",
            )
        )
        layers = [
            _dense_layer_plan(
                layer_id=layer_id,
                stage_id=stage_id,
                rank_ids=rank_ids,
                hidden_size=hidden_size,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dimension=head_dimension,
                intermediate_size=intermediate_size,
            )
            for layer_id in range(layer_start, layer_end)
        ]
        # Keep construction honest if a future range implementation changes.
        for layer in layers:
            assert layer.mlp is not None
            layer.mlp.intermediate_ranges = {
                rank: interval for rank, interval in enumerate(intermediate_ranges)
            }
        pipeline.append(
            PipelineStagePlan(
                stage_id=stage_id,
                layer_plans=layers,
                parallel_cell_id=cell_id,
                input_spec=boundary_spec,
                output_spec=boundary_spec,
                owns_embeddings=stage_id == 0,
                owns_final_norm=stage_id == pipeline_stage_count - 1,
                owns_lm_head=stage_id == pipeline_stage_count - 1,
            )
        )
    return ModelPartitionPlan(
        model_id=model_id,
        model_revision=model_revision,
        pipeline_stages=pipeline,
        tensor_shards=[],
        collective_groups=groups,
        parallel_cells=cells,
        layer_count=layer_count,
        vocabulary_parallel=vocabulary_parallel,
        metadata={
            "pipeline_stage_count": pipeline_stage_count,
            "tensor_parallel_degree": tensor_parallel_degree,
            "logical_pipeline_rank_workers": pipeline_stage_count * tensor_parallel_degree,
            "logical_layer_shards": layer_count * tensor_parallel_degree,
            "partition_count_not_bounded_by_layer_count": (
                pipeline_stage_count * tensor_parallel_degree > layer_count
            ),
        },
    )


def validate_tensor_shard_union(shards: list[TensorShard]) -> dict[str, Any]:
    """Validate complete, non-overlapping source coverage and hash agreement."""

    grouped: dict[tuple[str, int, str], list[TensorShard]] = defaultdict(list)
    for shard in shards:
        grouped[(shard.tensor_name, shard.stage_id, shard.partition_mode)].append(shard)
    failures: list[str] = []
    tensor_results: dict[str, dict[str, Any]] = {}
    source_names: set[str] = set()
    for (name, stage_id, partition_mode), items in sorted(grouped.items()):
        source_names.add(name)
        display_name = f"{name}@stage-{stage_id}:{partition_mode}"
        shapes = {item.global_shape for item in items}
        source_hashes = {item.source_tensor_hash for item in items}
        if len(shapes) != 1:
            failures.append(f"{display_name}: inconsistent global shapes")
        if len(source_hashes) != 1:
            failures.append(f"{display_name}: inconsistent source hashes")
        replicated = {item.replicated for item in items}
        if len(replicated) != 1:
            failures.append(f"{display_name}: mixed replicated and sliced entries")
            continue
        if True in replicated:
            expected_ranks = set(range(items[0].world_size))
            actual_ranks = {item.rank for item in items}
            if actual_ranks != expected_ranks:
                failures.append(f"{display_name}: replicated rank coverage mismatch")
            tensor_results[display_name] = {
                "replicated": True,
                "rank_count": len(actual_ranks),
                "covered": not failures,
            }
            continue
        axes = {item.shard_axis for item in items}
        if len(axes) != 1:
            failures.append(f"{display_name}: inconsistent shard axes")
            continue
        axis = items[0].shard_axis
        assert axis is not None
        intervals = sorted((item.shard_start, item.shard_end, item.rank) for item in items)
        coverage_intervals = intervals
        if partition_mode == "kv_head_replication":
            coverage_intervals = sorted({(start, end, -1) for start, end, _ in intervals})
        cursor = 0
        for start, end, rank in coverage_intervals:
            if start != cursor:
                kind = "overlap" if start < cursor else "missing range"
                failures.append(f"{display_name}: {kind} before rank {rank}: {cursor}->{start}")
            cursor = max(cursor, end)
        expected_end = items[0].global_shape[axis]
        if cursor != expected_end:
            failures.append(f"{display_name}: coverage ends at {cursor}, expected {expected_end}")
        tensor_results[display_name] = {
            "replicated": False,
            "shard_axis": axis,
            "intervals": intervals,
            "covered_elements_on_axis": cursor,
            "expected_elements_on_axis": expected_end,
        }
    return {
        "status": "PASS" if not failures else "FAIL",
        "tensor_count": len(source_names),
        "placement_count": len(grouped),
        "shard_count": len(shards),
        "failures": failures,
        "tensors": tensor_results,
    }
