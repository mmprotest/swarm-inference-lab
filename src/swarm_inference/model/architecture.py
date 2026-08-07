"""Canonical model-architecture identities shared by resolvers and engines.

The values in this module describe model families, not repositories.  Raw
configuration and GGUF identifiers are retained on the resolved descriptor so
an execution engine can prove support for the exact representation it will
open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class ModelArchitecture(StrEnum):
    OLMOE = "olmoe"
    QWEN3_DENSE = "qwen3_dense"
    QWEN3_MOE = "qwen3_moe"


ArchitectureSource = Literal[
    "config.architectures",
    "config.model_type",
    "gguf.general.architecture",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ArchitectureIdentity:
    canonical: str | None
    raw: str | None
    source: ArchitectureSource

    @property
    def known(self) -> bool:
        return self.canonical in {item.value for item in ModelArchitecture}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


# These are upstream metadata identifiers (Transformers class/model_type and
# GGUF general.architecture values), deliberately not model repository names.
_ALIASES: dict[str, ModelArchitecture] = {
    "olmoe": ModelArchitecture.OLMOE,
    "olmoeforcausallm": ModelArchitecture.OLMOE,
    "qwen3": ModelArchitecture.QWEN3_DENSE,
    "qwen3dense": ModelArchitecture.QWEN3_DENSE,
    "qwen3forcausallm": ModelArchitecture.QWEN3_DENSE,
    "qwen3moe": ModelArchitecture.QWEN3_MOE,
    "qwen3moeforcausallm": ModelArchitecture.QWEN3_MOE,
    # Newer Qwen 3.x sparse checkpoints use the same product family while
    # retaining their exact raw identifier for engine-specific validation.
    "qwen35moe": ModelArchitecture.QWEN3_MOE,
    "qwen35moeforcausallm": ModelArchitecture.QWEN3_MOE,
    "qwen35moeforconditionalgeneration": ModelArchitecture.QWEN3_MOE,
    "qwen35moetext": ModelArchitecture.QWEN3_MOE,
}


GGUF_ARCHITECTURE_IDENTIFIERS: dict[ModelArchitecture, tuple[str, ...]] = {
    ModelArchitecture.OLMOE: ("olmoe",),
    ModelArchitecture.QWEN3_DENSE: ("qwen3", "qwen35"),
    ModelArchitecture.QWEN3_MOE: ("qwen3moe", "qwen35moe"),
}


def normalize_model_architecture(value: str | None) -> str | None:
    """Return one product family ID while preserving unknown identities."""

    if value is None or not value.strip():
        return None
    match = _ALIASES.get(_key(value))
    return match.value if match is not None else value.strip()


def architecture_is_known(value: str | None) -> bool:
    return normalize_model_architecture(value) in {item.value for item in ModelArchitecture}


def architecture_from_config(config: dict[str, Any]) -> ArchitectureIdentity:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        raw = str(architectures[0])
        return ArchitectureIdentity(normalize_model_architecture(raw), raw, "config.architectures")
    if config.get("model_type"):
        raw = str(config["model_type"])
        return ArchitectureIdentity(normalize_model_architecture(raw), raw, "config.model_type")
    return ArchitectureIdentity(None, None, "unknown")


def architecture_from_gguf(
    raw_value: object,
    *,
    fallback: ArchitectureIdentity | None = None,
) -> ArchitectureIdentity:
    if raw_value is not None and str(raw_value).strip():
        raw = str(raw_value).strip()
        return ArchitectureIdentity(
            normalize_model_architecture(raw), raw, "gguf.general.architecture"
        )
    return fallback or ArchitectureIdentity(None, None, "unknown")


def gguf_identifiers_for(value: str | None) -> tuple[str, ...]:
    normalized = normalize_model_architecture(value)
    if normalized is None:
        return (str(value).strip(),) if value and str(value).strip() else ()
    try:
        architecture = ModelArchitecture(normalized)
    except ValueError:
        return (str(value).strip(),) if value and str(value).strip() else ()
    return GGUF_ARCHITECTURE_IDENTIFIERS[architecture]


__all__ = [
    "GGUF_ARCHITECTURE_IDENTIFIERS",
    "ArchitectureIdentity",
    "ArchitectureSource",
    "ModelArchitecture",
    "architecture_from_config",
    "architecture_from_gguf",
    "architecture_is_known",
    "gguf_identifiers_for",
    "normalize_model_architecture",
]
