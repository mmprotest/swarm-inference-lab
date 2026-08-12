"""Experiment 015: speculative hierarchical Kimi K3 architecture research."""

from swarm_inference.experiments.experiment_015.contracts import (
    EconomicsConfig,
    EvidenceClass,
    Experiment015Error,
)
from swarm_inference.experiments.experiment_015.speculation import (
    RequestState,
    SpeculativeSession,
    greedy_acceptance,
    stochastic_acceptance,
)

__all__ = [
    "EconomicsConfig",
    "EvidenceClass",
    "Experiment015Error",
    "RequestState",
    "SpeculativeSession",
    "greedy_acceptance",
    "stochastic_acceptance",
]
