"""Deprecated import compatibility for frozen Experiment 010 dispatch."""

# No product functionality may be added through this compatibility module.
from swarm_inference.experiments.experiment_010.legacy_runtime.dispatch import (
    DispatchResult,
    ExpertDispatcher,
    FailureController,
    FailureEvent,
    LocalExecutor,
    RecoveryMetrics,
)

__all__ = [
    "DispatchResult",
    "ExpertDispatcher",
    "FailureController",
    "FailureEvent",
    "LocalExecutor",
    "RecoveryMetrics",
]
