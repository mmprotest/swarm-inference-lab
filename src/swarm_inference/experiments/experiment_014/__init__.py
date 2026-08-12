"""Experiment 014: Kimi K3 pre-cluster certification."""

from swarm_inference.experiments.experiment_014.census import (
    CheckpointCensusError,
    TensorClassification,
    build_checkpoint_census,
    classify_tensor,
)
from swarm_inference.experiments.experiment_014.compatibility import (
    CompatibilityError,
    build_compatibility_matrix,
)
from swarm_inference.experiments.experiment_014.deployment import (
    DeploymentPlanError,
    build_execution_plan,
    run_logical_rehearsal,
)
from swarm_inference.experiments.experiment_014.distribution import (
    DistributionError,
    build_distribution_manifest,
    materialize_worker_package,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    KimiCudaGraphRunner,
    benchmark_streamed_cuda_graph,
)
from swarm_inference.experiments.experiment_014.oracle import (
    SerialOracleError,
    run_serial_oracle,
)
from swarm_inference.experiments.experiment_014.placement import (
    MemoryPolicy,
    PlacementError,
    write_placement_artifacts,
)
from swarm_inference.experiments.experiment_014.support import (
    ModelSupportError,
    build_model_support_matrix,
)

__all__ = [
    "CheckpointCensusError",
    "CompatibilityError",
    "DeploymentPlanError",
    "DistributionError",
    "KimiCudaGraphRunner",
    "MemoryPolicy",
    "ModelSupportError",
    "PlacementError",
    "SerialOracleError",
    "TensorClassification",
    "benchmark_streamed_cuda_graph",
    "build_checkpoint_census",
    "build_compatibility_matrix",
    "build_distribution_manifest",
    "build_execution_plan",
    "build_model_support_matrix",
    "classify_tensor",
    "materialize_worker_package",
    "run_logical_rehearsal",
    "run_serial_oracle",
    "write_placement_artifacts",
]
