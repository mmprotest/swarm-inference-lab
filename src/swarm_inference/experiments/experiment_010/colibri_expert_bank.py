"""Deprecated Experiment 010 names for the canonical Colibri expert-bank tools."""

from swarm_inference.backends.colibri.expert_bank import (
    ByteRange,
    OutputTensor,
    TensorLocation,
    _finish_target,
    build_coordinator_container,
    build_expert_bank,
    build_microshard_bank,
    main,
    model_fingerprint,
    require_bank_ownership,
    scan_safetensors,
    verify_bank,
    verify_coordinator_container,
    verify_microshard_reconstruction,
)

__all__ = [
    "ByteRange",
    "OutputTensor",
    "TensorLocation",
    "_finish_target",
    "build_coordinator_container",
    "build_expert_bank",
    "build_microshard_bank",
    "main",
    "model_fingerprint",
    "require_bank_ownership",
    "scan_safetensors",
    "verify_bank",
    "verify_coordinator_container",
    "verify_microshard_reconstruction",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
