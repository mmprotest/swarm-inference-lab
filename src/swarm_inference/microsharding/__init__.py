"""Intra-layer tensor, vocabulary, and expert microsharding runtime.

Experiment 006 deliberately keeps logical rank ownership separate from physical
process placement.  The primary backend executes all ranks in one CUDA context;
the projection backend models independently provisioned ranks.
"""

from swarm_inference.microsharding.expert_abi import (
    ExpertMicroshardDescriptor,
    ExpertProjectionSlice,
    executable_microshard_equivalence,
    validate_expert_microshard_set,
)
from swarm_inference.microsharding.schemas import (
    AttentionPartitionPlan,
    CollectiveGroupPlan,
    CollectivePlan,
    DenseMLPPartitionPlan,
    LayerPartitionPlan,
    ModelPartitionPlan,
    MoEPartitionPlan,
    ParallelCellPlan,
    PipelineStagePlan,
    TensorShard,
)

__all__ = [
    "AttentionPartitionPlan",
    "CollectiveGroupPlan",
    "CollectivePlan",
    "DenseMLPPartitionPlan",
    "ExpertMicroshardDescriptor",
    "ExpertProjectionSlice",
    "LayerPartitionPlan",
    "MoEPartitionPlan",
    "ModelPartitionPlan",
    "ParallelCellPlan",
    "PipelineStagePlan",
    "TensorShard",
    "executable_microshard_equivalence",
    "validate_expert_microshard_set",
]
