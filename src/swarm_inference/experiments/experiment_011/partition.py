"""Generic contiguous-layer partition planning for Experiment 011."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from safetensors import safe_open
from transformers import AutoConfig

PartitionMethod = Literal["equal", "balanced"]

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True, slots=True)
class LayerCost:
    layer_id: int
    execution_ns: int
    weight_bytes: int
    kv_bytes_per_token: int
    peak_temporary_bytes: int
    activation_bytes: int
    measured: bool

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
    def read(cls, path: Path) -> StagePlan:
        raw = json.loads(path.read_text(encoding="utf-8"))
        assignments = tuple(
            StageAssignment(**{**row, "layer_ids": tuple(int(value) for value in row["layer_ids"])})
            for row in raw.pop("assignments")
        )
        plan = cls(assignments=assignments, **raw)
        plan.validate()
        return plan


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


def _tensor_nbytes(handle: Any, key: str) -> int:
    tensor_slice = handle.get_slice(key)
    shape = tensor_slice.get_shape()
    dtype = str(tensor_slice.get_dtype())
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * _DTYPE_BYTES[dtype]


def inspect_model_partition_metadata(
    model_path: Path,
    *,
    model_revision: str,
    tokenizer_revision: str,
    measured_layer_ns: dict[int, int] | None = None,
) -> ModelPartitionMetadata:
    model_path = model_path.resolve()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    sizes: dict[str, int] = {}
    for shard, keys in by_shard.items():
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                sizes[key] = _tensor_nbytes(handle, key)
    layer_count = int(config.num_hidden_layers)
    weight_bytes = [0] * layer_count
    embedding_weight_bytes = 0
    final_weight_bytes = 0
    for key, size in sizes.items():
        if key.startswith("model.layers."):
            layer_id = int(key.split(".")[2])
            weight_bytes[layer_id] += size
        elif key.startswith("model.embed_tokens."):
            embedding_weight_bytes += size
        elif key.startswith("model.norm.") or key.startswith("lm_head."):
            final_weight_bytes += size
    dtype_bytes = 2
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    kv_heads = int(config.num_key_value_heads)
    kv_bytes_per_layer_token = 2 * kv_heads * head_dim * dtype_bytes
    activation_bytes = int(config.hidden_size) * dtype_bytes
    costs = []
    for layer_id in range(layer_count):
        measured = measured_layer_ns is not None and layer_id in measured_layer_ns
        execution_ns = (
            int(measured_layer_ns[layer_id])
            if measured
            else max(1, int(weight_bytes[layer_id] / 4_000_000_000 * 1e9))
        )
        costs.append(
            LayerCost(
                layer_id=layer_id,
                execution_ns=execution_ns,
                weight_bytes=weight_bytes[layer_id],
                kv_bytes_per_token=kv_bytes_per_layer_token,
                peak_temporary_bytes=max(activation_bytes * 16, weight_bytes[layer_id] // 32),
                activation_bytes=activation_bytes,
                measured=measured,
            )
        )
    identity_payload = json.dumps(
        {
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "weight_map": weight_map,
            "layer_costs": [asdict(cost) for cost in costs],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ModelPartitionMetadata(
        layer_costs=tuple(costs),
        embedding_weight_bytes=embedding_weight_bytes,
        final_weight_bytes=final_weight_bytes,
        dtype_bytes=dtype_bytes,
        hidden_size=int(config.hidden_size),
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        metadata_hash=hashlib.sha256(identity_payload).hexdigest(),
    )


def equal_ranges(layer_count: int, stage_count: int) -> tuple[tuple[int, int], ...]:
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


def balanced_ranges(
    costs: tuple[LayerCost, ...], stage_count: int, *, memory_limit_bytes: int
) -> tuple[tuple[int, int], ...]:
    """Minimax contiguous partition via dynamic programming."""

    layer_count = len(costs)
    if stage_count < 1 or stage_count > layer_count:
        raise ValueError("stage count must be between one and the number of layers")
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
                segment_memory = memory_prefix[end] - memory_prefix[start]
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
    device: str = "cuda:0",
) -> StagePlan:
    if method == "equal":
        ranges = equal_ranges(len(metadata.layer_costs), stage_count)
        objective = "equal contiguous layer counts"
    elif method == "balanced":
        ranges = balanced_ranges(
            metadata.layer_costs, stage_count, memory_limit_bytes=memory_limit_bytes
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
