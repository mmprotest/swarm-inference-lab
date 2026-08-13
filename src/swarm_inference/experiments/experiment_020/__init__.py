"""Experiment 020: full pre-spend swarm readiness.

The package is deliberately import-safe: importing it cannot contact Vast.ai,
start a worker, or allocate a GPU resource.
"""

from .vast import E020_RENTAL_FORBIDDEN, EXPERIMENT_020_READ_ONLY, VastMode

__all__ = ["E020_RENTAL_FORBIDDEN", "EXPERIMENT_020_READ_ONLY", "VastMode"]
