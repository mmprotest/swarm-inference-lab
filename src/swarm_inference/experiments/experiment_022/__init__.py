"""Experiment 022: adaptive heterogeneous placement.

The experiment is deliberately local-only.  Physical measurements are taken on
the local reference device; heterogeneous nodes and network links are controlled
model inputs and are never described as a physical heterogeneous swarm.
"""

from __future__ import annotations

E022_ZERO_RENTAL = True
E022_EXTERNAL_SWARM = False
E022_VAST_ACCESS = False

EVIDENCE_CLASS = (
    "PHYSICALLY GROUNDED MODEL + CONTROLLED HETEROGENEITY + SHAPED NETWORK"
)

PRIMARY_HYPOTHESIS = (
    "For heterogeneous resource pools, giving the Swarm planner access to exact "
    "sub-layer partitioning produces materially better Kimi K3 placements than "
    "restricting the same planner to whole transformer layers."
)

__all__ = [
    "E022_EXTERNAL_SWARM",
    "E022_VAST_ACCESS",
    "E022_ZERO_RENTAL",
    "EVIDENCE_CLASS",
    "PRIMARY_HYPOTHESIS",
]
