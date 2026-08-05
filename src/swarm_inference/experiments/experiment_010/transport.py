"""Deprecated compatibility for frozen Experiment 010 shaped transport."""

# This remains for evidence reproduction, including experiment-only network
# shaping. Product stage-ring and expert transports live under transport/.
from swarm_inference.experiments.experiment_010.legacy_runtime.transport import (
    NETWORK_PROFILES,
    ExpertTransportClient,
    ExpertTransportMetrics,
    NetworkShaper,
    ShapedTransportError,
    ShaperMetrics,
    measured_network_profile,
)

__all__ = [
    "NETWORK_PROFILES",
    "ExpertTransportClient",
    "ExpertTransportMetrics",
    "NetworkShaper",
    "ShapedTransportError",
    "ShaperMetrics",
    "measured_network_profile",
]
