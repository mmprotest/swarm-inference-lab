"""Experiment 023: exact sparse expert optionality.

The physical evidence in this experiment is limited to local RTX 5090
production-native expert-group primitives.  The headline result is a
physically grounded multi-request model with shaped network resources; it is
not a physical distributed Kimi K3 run.
"""

from __future__ import annotations

E023_ZERO_RENTAL = True
E023_EXTERNAL_SWARM = False
E023_VAST_ACCESS = False

EVIDENCE_CLASS = "PHYSICALLY GROUNDED MODEL + SHAPED NETWORK"
PRIMARY_HYPOTHESIS = (
    "A heterogeneous resource pool is more useful when excess memory creates "
    "alternative exact execution paths than when every added worker creates "
    "another mandatory dependency."
)

__all__ = [
    "E023_EXTERNAL_SWARM",
    "E023_VAST_ACCESS",
    "E023_ZERO_RENTAL",
    "EVIDENCE_CLASS",
    "PRIMARY_HYPOTHESIS",
]
