"""Experiment 011: communication-avoiding exact contiguous-stage decode."""

from __future__ import annotations

from swarm_inference.protocol.stage_ring import (
    STAGE_RING_PROTOCOL_VERSION as PROTOCOL_VERSION,
)

EXPERIMENT_ID = "experiment-011"
MODEL_REVISION = "pinned-b085b48888a88d9a1c00b151a9979774b72cdbfd"
TOKENIZER_REVISION = "sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea"

__all__ = ["EXPERIMENT_ID", "MODEL_REVISION", "PROTOCOL_VERSION", "TOKENIZER_REVISION"]
