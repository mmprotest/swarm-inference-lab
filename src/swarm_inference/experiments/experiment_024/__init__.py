"""Experiment 024: communication-avoiding Kimi K3 swarm economics.

Experiment 024 fails closed when an immutable input cannot support the frozen
93-layer, P8-only deployment.  A failed validity gate is evidence, not a
commercial-performance result.
"""

from __future__ import annotations

EXPERIMENT_ID = "024"
E024_ZERO_RENTAL = True
E024_EXTERNAL_SWARM = False
E024_VAST_ACCESS = False

__all__ = [
    "E024_EXTERNAL_SWARM",
    "E024_VAST_ACCESS",
    "E024_ZERO_RENTAL",
    "EXPERIMENT_ID",
]
