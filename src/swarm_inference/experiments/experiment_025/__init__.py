"""Experiment 025: physical Kimi K3 consumer-GPU swarm.

The package is intentionally isolated from the read-only Vast boundaries used
by Experiments 020 and 021.  Nothing in this module performs a rental merely by
being imported.
"""

from .constants import EVIDENCE_CLASS, EXPERIMENT_ID, MODEL_ID, MODEL_REVISION

__all__ = ["EVIDENCE_CLASS", "EXPERIMENT_ID", "MODEL_ID", "MODEL_REVISION"]
