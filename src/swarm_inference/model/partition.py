"""Model-agnostic contiguous-layer partition records and algorithms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from swarm_inference.config.models import CacheSpec, StageDefinition, TensorSpec

PartitionMethod = Literal["equal", "balanced"]


@dataclass(frozen=True, slots=True)
class LayerCost:
    layer_id: int
    execution_ns: int
    weight_bytes: int
    kv_bytes_per_token: int
    peak_temporary_bytes: int
    activation_bytes: int
    measured: bool
    expert_weight_bytes: int = 0
    expert_execution_ns: int = 0

    @property
    def objective_cost(self) -> float:
        # Compute dominates; memory terms provide deterministic tie-breaking.
        return (
            float(self.execution_ns) + self.weight_bytes / 32.0 + self.peak_temporary_bytes / 128.0
        )


@dataclass(frozen=True, slots=True)
class StageAssignment:
    stage_id: int
    layer_start: int
    layer_end: int
    layer_ids: tuple[int, ...]
    weight_bytes: int
    estimated_compute_ns: int
    measured_compute_ns: int | None
    kv_cache_bytes_per_token: int
    peak_temporary_bytes: int
    activation_bytes: int
    device: str
    owns_embeddings: bool
    owns_final_norm: bool
    owns_output_projection: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["layer_ids"] = list(self.layer_ids)
        return value

    def to_stage_definition(
        self,
        *,
        input_spec: TensorSpec,
        output_spec: TensorSpec,
        tensor_names: tuple[str, ...] = (),
        cache_format: str = "dynamic-kv",
        shard_hash: str | None = None,
        required_total_memory_bytes: int | None = None,
    ) -> StageDefinition:
        """Convert this assignment to the existing product manifest type."""

        if self.weight_bytes <= 0:
            raise ValueError("StageDefinition requires a positive weight byte count")
        return StageDefinition(
            stage_id=self.stage_id,
            layer_start=self.layer_start,
            layer_end=self.layer_end,
            owns_embeddings=self.owns_embeddings,
            owns_final_norm=self.owns_final_norm,
            owns_output_head=self.owns_output_projection,
            required_memory_bytes=self.weight_bytes,
            estimated_execution_ms={"default": self.estimated_compute_ns / 1_000_000},
            input_spec=input_spec,
            output_spec=output_spec,
            cache_spec=CacheSpec(
                format=cache_format,
                bytes_per_token=self.kv_cache_bytes_per_token,
                reconstructable_by_replay=True,
            ),
            tensor_names=list(tensor_names),
            tensor_count=len(tensor_names),
            shard_hash=shard_hash,
            required_total_memory_bytes=required_total_memory_bytes,
        )

    @classmethod
    def from_stage_definition(
        cls,
        stage: StageDefinition,
        *,
        device: str,
        measured_compute_ns: int | None = None,
        peak_temporary_bytes: int = 0,
        activation_bytes: int = 0,
    ) -> StageAssignment:
        """Convert a manifest stage when the additional planning costs are known."""

        default_ms = float(stage.estimated_execution_ms.get("default", 0.0))
        return cls(
            stage_id=stage.stage_id,
            layer_start=stage.layer_start,
            layer_end=stage.layer_end,
            layer_ids=tuple(range(stage.layer_start, stage.layer_end)),
            weight_bytes=stage.required_memory_bytes,
            estimated_compute_ns=int(default_ms * 1_000_000),
            measured_compute_ns=measured_compute_ns,
            kv_cache_bytes_per_token=stage.cache_spec.bytes_per_token,
            peak_temporary_bytes=peak_temporary_bytes,
            activation_bytes=activation_bytes,
            device=device,
            owns_embeddings=stage.owns_embeddings,
            owns_final_norm=stage.owns_final_norm,
            owns_output_projection=stage.owns_output_head,
        )


def stage_assignment_to_definition(
    assignment: StageAssignment,
    *,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
    tensor_names: tuple[str, ...] = (),
    cache_format: str = "dynamic-kv",
    shard_hash: str | None = None,
    required_total_memory_bytes: int | None = None,
) -> StageDefinition:
    return assignment.to_stage_definition(
        input_spec=input_spec,
        output_spec=output_spec,
        tensor_names=tensor_names,
        cache_format=cache_format,
        shard_hash=shard_hash,
        required_total_memory_bytes=required_total_memory_bytes,
    )


def stage_assignment_from_definition(
    stage: StageDefinition,
    *,
    device: str,
    measured_compute_ns: int | None = None,
    peak_temporary_bytes: int = 0,
    activation_bytes: int = 0,
) -> StageAssignment:
    return StageAssignment.from_stage_definition(
        stage,
        device=device,
        measured_compute_ns=measured_compute_ns,
        peak_temporary_bytes=peak_temporary_bytes,
        activation_bytes=activation_bytes,
    )


@dataclass(frozen=True, slots=True)
class StagePlan:
    model_path: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    stage_count: int
    layer_count: int
    partition_method: PartitionMethod
    planner_objective: str
    memory_limit_bytes: int
    assignments: tuple[StageAssignment, ...]
    metadata_hash: str

    def validate(self) -> None:
        if self.stage_count < 1 or self.layer_count < 1:
            raise ValueError("stage and layer counts must be positive")
        if self.memory_limit_bytes <= 0:
            raise ValueError("stage plan memory limit must be positive")
        if not self.model_revision or not self.tokenizer_revision or not self.topology_id:
            raise ValueError("stage plan identities cannot be empty")
        if len(self.assignments) != self.stage_count:
            raise ValueError("stage plan assignment count is incomplete")
        ordered = [layer for stage in self.assignments for layer in stage.layer_ids]
        if ordered != list(range(self.layer_count)):
            raise ValueError("stage plan has missing, duplicate, overlapping, or reordered layers")
        for expected_stage, assignment in enumerate(self.assignments):
            if assignment.stage_id != expected_stage:
                raise ValueError("stage IDs must be consecutive")
            if not assignment.layer_ids:
                raise ValueError("every stage must own at least one layer")
            if assignment.layer_ids != tuple(range(assignment.layer_start, assignment.layer_end)):
                raise ValueError("stage assignment is not contiguous")
            if assignment.weight_bytes < 0:
                raise ValueError("stage assignment weight bytes cannot be negative")
            if assignment.weight_bytes > self.memory_limit_bytes:
                raise ValueError("stage assignment exceeds its memory limit")
            if assignment.owns_embeddings != (expected_stage == 0):
                raise ValueError("only stage zero may own token embeddings")
            final = expected_stage == self.stage_count - 1
            if assignment.owns_final_norm != final or assignment.owns_output_projection != final:
                raise ValueError("only the final stage may own normalisation and output projection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "topology_id": self.topology_id,
            "stage_count": self.stage_count,
            "layer_count": self.layer_count,
            "partition_method": self.partition_method,
            "planner_objective": self.planner_objective,
            "memory_limit_bytes": self.memory_limit_bytes,
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "metadata_hash": self.metadata_hash,
        }

    def write(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StagePlan:
        raw = dict(value)
        try:
            assignment_rows = raw.pop("assignments")
        except KeyError as exc:
            raise ValueError("stage plan is missing assignments") from exc
        if not isinstance(assignment_rows, list):
            raise ValueError("stage plan assignments must be an array")
        assignments = tuple(
            StageAssignment(
                **{
                    **row,
                    "layer_ids": tuple(int(layer) for layer in row["layer_ids"]),
                }
            )
            for row in assignment_rows
        )
        plan = cls(assignments=assignments, **raw)
        plan.validate()
        return plan

    @classmethod
    def read(cls, path: Path) -> StagePlan:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("stage plan JSON must contain an object")
        return cls.from_dict(raw)


@dataclass(frozen=True, slots=True)
class ModelPartitionMetadata:
    layer_costs: tuple[LayerCost, ...]
    embedding_weight_bytes: int
    final_weight_bytes: int
    dtype_bytes: int
    hidden_size: int
    model_revision: str
    tokenizer_revision: str
    metadata_hash: str
    expert_count: int = 0
    experts_per_token: int = 0
    expert_intermediate_size: int = 0
    model_fingerprint: str = ""
    quantization_fingerprint: str = ""


def equal_ranges(layer_count: int, stage_count: int) -> tuple[tuple[int, int], ...]:
    """Split layers into complete contiguous intervals with near-equal counts."""

    if stage_count < 1 or stage_count > layer_count:
        raise ValueError("stage count must be between one and the number of layers")
    quotient, remainder = divmod(layer_count, stage_count)
    ranges = []
    start = 0
    for stage_id in range(stage_count):
        count = quotient + (1 if stage_id < remainder else 0)
        ranges.append((start, start + count))
        start += count
    return tuple(ranges)


def _validate_layer_costs(costs: tuple[LayerCost, ...]) -> None:
    if [cost.layer_id for cost in costs] != list(range(len(costs))):
        raise ValueError("layer costs must be ordered and contiguous from layer zero")
    for cost in costs:
        if any(
            value < 0
            for value in (
                cost.execution_ns,
                cost.weight_bytes,
                cost.kv_bytes_per_token,
                cost.peak_temporary_bytes,
                cost.activation_bytes,
            )
        ):
            raise ValueError("layer costs cannot be negative")


def balanced_ranges(
    costs: tuple[LayerCost, ...],
    stage_count: int,
    *,
    memory_limit_bytes: int,
    stage_overhead_bytes: tuple[int, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Find a minimax contiguous partition using dynamic programming."""

    _validate_layer_costs(costs)
    layer_count = len(costs)
    if stage_count < 1 or stage_count > layer_count:
        raise ValueError("stage count must be between one and the number of layers")
    if memory_limit_bytes < 0:
        raise ValueError("memory limit cannot be negative")
    overheads = stage_overhead_bytes or (0,) * stage_count
    if len(overheads) != stage_count or any(value < 0 for value in overheads):
        raise ValueError("stage overheads must contain one non-negative value per stage")
    objective_prefix = [0.0]
    memory_prefix = [0]
    for cost in costs:
        objective_prefix.append(objective_prefix[-1] + cost.objective_cost)
        memory_prefix.append(memory_prefix[-1] + cost.weight_bytes)

    infinity = float("inf")
    dp = [[infinity] * (layer_count + 1) for _ in range(stage_count + 1)]
    split = [[-1] * (layer_count + 1) for _ in range(stage_count + 1)]
    dp[0][0] = 0.0
    for stages in range(1, stage_count + 1):
        for end in range(stages, layer_count + 1):
            for start in range(stages - 1, end):
                segment_memory = memory_prefix[end] - memory_prefix[start] + overheads[stages - 1]
                if segment_memory > memory_limit_bytes:
                    continue
                segment_cost = objective_prefix[end] - objective_prefix[start]
                candidate = max(dp[stages - 1][start], segment_cost)
                if candidate < dp[stages][end]:
                    dp[stages][end] = candidate
                    split[stages][end] = start
    if split[stage_count][layer_count] < 0:
        raise MemoryError("no contiguous stage plan satisfies the memory limit")
    result = []
    end = layer_count
    for stages in range(stage_count, 0, -1):
        start = split[stages][end]
        result.append((start, end))
        end = start
    return tuple(reversed(result))


def build_stage_plan(
    model_path: Path,
    *,
    metadata: ModelPartitionMetadata,
    stage_count: int,
    method: PartitionMethod,
    memory_limit_bytes: int,
    device: str = "cpu",
) -> StagePlan:
    """Build and validate a complete contiguous ownership plan."""

    _validate_layer_costs(metadata.layer_costs)
    if memory_limit_bytes <= 0:
        raise ValueError("memory limit must be positive")
    if not device:
        raise ValueError("stage device cannot be empty")
    if method == "equal":
        ranges = equal_ranges(len(metadata.layer_costs), stage_count)
        objective = "equal contiguous layer counts"
    elif method == "balanced":
        overheads = [0] * stage_count
        overheads[0] += metadata.embedding_weight_bytes
        overheads[-1] += metadata.final_weight_bytes
        ranges = balanced_ranges(
            metadata.layer_costs,
            stage_count,
            memory_limit_bytes=memory_limit_bytes,
            stage_overhead_bytes=tuple(overheads),
        )
        objective = "minimise maximum measured contiguous stage cost subject to memory"
    else:
        raise ValueError(f"unknown partition method {method!r}")
    topology_seed = {
        "model_revision": metadata.model_revision,
        "stage_count": stage_count,
        "method": method,
        "ranges": ranges,
        "metadata_hash": metadata.metadata_hash,
    }
    topology_hash = hashlib.sha256(
        json.dumps(topology_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assignments = []
    for stage_id, (start, end) in enumerate(ranges):
        selected = metadata.layer_costs[start:end]
        overhead = (metadata.embedding_weight_bytes if stage_id == 0 else 0) + (
            metadata.final_weight_bytes if stage_id == stage_count - 1 else 0
        )
        weight_bytes = sum(cost.weight_bytes for cost in selected) + overhead
        if weight_bytes > memory_limit_bytes:
            raise MemoryError(
                f"stage {stage_id} requires {weight_bytes} bytes, exceeding "
                f"the {memory_limit_bytes}-byte limit"
            )
        assignments.append(
            StageAssignment(
                stage_id=stage_id,
                layer_start=start,
                layer_end=end,
                layer_ids=tuple(range(start, end)),
                weight_bytes=weight_bytes,
                estimated_compute_ns=sum(cost.execution_ns for cost in selected),
                measured_compute_ns=(
                    sum(cost.execution_ns for cost in selected)
                    if all(cost.measured for cost in selected)
                    else None
                ),
                kv_cache_bytes_per_token=sum(cost.kv_bytes_per_token for cost in selected),
                peak_temporary_bytes=max(cost.peak_temporary_bytes for cost in selected),
                activation_bytes=max(cost.activation_bytes for cost in selected),
                device=device,
                owns_embeddings=stage_id == 0,
                owns_final_norm=stage_id == stage_count - 1,
                owns_output_projection=stage_id == stage_count - 1,
            )
        )
    plan = StagePlan(
        model_path=str(model_path.resolve()),
        model_revision=metadata.model_revision,
        tokenizer_revision=metadata.tokenizer_revision,
        topology_id=f"stage-ring-{stage_count}-{method}-{topology_hash[:16]}",
        stage_count=stage_count,
        layer_count=len(metadata.layer_costs),
        partition_method=method,
        planner_objective=objective,
        memory_limit_bytes=memory_limit_bytes,
        assignments=tuple(assignments),
        metadata_hash=metadata.metadata_hash,
    )
    plan.validate()
    return plan


__all__ = [
    "LayerCost",
    "ModelPartitionMetadata",
    "PartitionMethod",
    "StageAssignment",
    "StagePlan",
    "balanced_ranges",
    "build_stage_plan",
    "equal_ranges",
    "stage_assignment_from_definition",
    "stage_assignment_to_definition",
]
