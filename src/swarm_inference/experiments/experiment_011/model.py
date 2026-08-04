"""Deprecated compatibility exports for canonical OLMoE stage execution.

New code must import :mod:`swarm_inference.execution.olmoe_stage` directly.
"""

from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.execution.olmoe_stage import (
    ContiguousOlmoeStage,
    SafeTensorRepository,
    StageSessionState,
)

__all__ = [
    "ContiguousOlmoeStage",
    "SafeTensorRepository",
    "StageExecutionResult",
    "StageSessionState",
    "WeightOwnership",
]
