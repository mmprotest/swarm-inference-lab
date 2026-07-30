"""Synthetic model partitioning for simulator stages."""

from __future__ import annotations

from swarm_inference.config.models import (
    CacheSpec,
    StageDefinition,
    SyntheticModelConfig,
    TensorSpec,
)


def partition_layer_ranges(layer_count: int, stage_count: int) -> list[tuple[int, int]]:
    if layer_count <= 0 or stage_count <= 0:
        raise ValueError("layer_count and stage_count must be positive")
    if stage_count > layer_count:
        raise ValueError("stage_count cannot exceed layer_count")
    base, remainder = divmod(layer_count, stage_count)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for stage_id in range(stage_count):
        width = base + (1 if stage_id < remainder else 0)
        ranges.append((cursor, cursor + width))
        cursor += width
    return ranges


def build_synthetic_stages(config: SyntheticModelConfig) -> list[StageDefinition]:
    ranges = partition_layer_ranges(config.layer_count, config.stage_count)
    stages: list[StageDefinition] = []
    shape: list[int | str] = ["batch", "sequence", config.hidden_size]
    for stage_id, (start, end) in enumerate(ranges):
        layer_count = end - start
        stages.append(
            StageDefinition(
                stage_id=stage_id,
                layer_start=start,
                layer_end=end,
                owns_embeddings=stage_id == 0,
                owns_final_norm=stage_id == len(ranges) - 1,
                owns_output_head=stage_id == len(ranges) - 1,
                required_memory_bytes=layer_count * config.bytes_per_layer,
                estimated_execution_ms={},
                input_spec=TensorSpec(dtype=config.activation_dtype, shape=shape),
                output_spec=TensorSpec(dtype=config.activation_dtype, shape=shape),
                cache_spec=CacheSpec(
                    bytes_per_token=(config.cache_bytes_per_token_per_layer * layer_count)
                ),
                tensor_names=[f"synthetic.layers.{index}" for index in range(start, end)],
            )
        )
    return stages
