"""Deprecated compatibility exports for canonical contiguous partitioning.

New code must import :mod:`swarm_inference.model.partition` and the OLMoE
checkpoint inspector in :mod:`swarm_inference.model.olmoe` directly.
"""

from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.partition import (
    LayerCost,
    ModelPartitionMetadata,
    PartitionMethod,
    StageAssignment,
    StagePlan,
    balanced_ranges,
    build_stage_plan,
    equal_ranges,
    stage_assignment_from_definition,
    stage_assignment_to_definition,
)

# Deprecated spelling retained while Experiment 011 import paths are supported.
inspect_model_partition_metadata = inspect_olmoe_partition_metadata

__all__ = [
    "LayerCost",
    "ModelPartitionMetadata",
    "PartitionMethod",
    "StageAssignment",
    "StagePlan",
    "balanced_ranges",
    "build_stage_plan",
    "equal_ranges",
    "inspect_model_partition_metadata",
    "inspect_olmoe_partition_metadata",
    "stage_assignment_from_definition",
    "stage_assignment_to_definition",
]
