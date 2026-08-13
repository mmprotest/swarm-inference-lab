"""Experiment 021: independent-machine sub-layer Swarm thesis test.

The package is intentionally zero-rental.  It may inspect the Vast marketplace
through the audited read-only boundary, but it has no code path that can mutate
or rent a Vast resource.
"""

from __future__ import annotations

CANONICAL_SWARM_DEFINITION = (
    "A Swarm result is only a Swarm result if the model cannot be executed by "
    "assigning whole layers to the participating workers, and the reported "
    "performance emerges from sub-layer fragments distributed across "
    "independent machines."
)

E021_ZERO_RENTAL = True
E021_VAST_MUTATIONS_ALLOWED = False
MODEL_VALIDATION_GATES = {
    "median_absolute_percentage_error": 0.05,
    "p90_absolute_percentage_error": 0.10,
    "maximum_absolute_percentage_error": 0.15,
}

__all__ = [
    "CANONICAL_SWARM_DEFINITION",
    "E021_VAST_MUTATIONS_ALLOWED",
    "E021_ZERO_RENTAL",
    "MODEL_VALIDATION_GATES",
]
