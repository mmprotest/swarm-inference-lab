"""GGUF variant discovery and positive-feasibility selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from swarm_inference.model.descriptor import ModelFileDescriptor

_SPLIT_GGUF = re.compile(
    r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)
_QUANT = re.compile(
    r"(?i)(UD-[IQF]\d(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)"
)


@dataclass(frozen=True, slots=True)
class ModelVariant:
    variant_id: str
    quantization: str
    files: tuple[ModelFileDescriptor, ...]
    total_bytes: int
    quality_rank: float


@dataclass(frozen=True, slots=True)
class VariantCandidate:
    variant: ModelVariant
    feasible: bool
    reason: str
    score: float


@dataclass(frozen=True, slots=True)
class VariantSelection:
    selected: ModelVariant
    candidates: tuple[VariantCandidate, ...]


def _quantization(name: str) -> str:
    matches = _QUANT.findall(name.upper())
    return matches[-1].upper() if matches else "UNKNOWN"


def _quality_rank(quantization: str) -> float:
    value = quantization.upper()
    if value in {"F32", "BF16"}:
        return 16.0
    if value == "F16":
        return 15.0
    number = re.search(r"(\d+)", value)
    base = float(number.group(1)) if number else 0.0
    suffix = 0.0
    if value.endswith("_XL"):
        suffix = 0.45
    elif value.endswith("_L"):
        suffix = 0.3
    elif value.endswith("_M"):
        suffix = 0.2
    elif value.endswith("_S"):
        suffix = 0.1
    return base + suffix


def discover_gguf_variants(files: tuple[ModelFileDescriptor, ...]) -> tuple[ModelVariant, ...]:
    """Group split GGUF files atomically and expose one candidate per quantisation."""

    groups: dict[str, list[ModelFileDescriptor]] = {}
    expected_counts: dict[str, int] = {}
    for item in files:
        if not item.relative_path.lower().endswith(".gguf"):
            continue
        name = item.relative_path.rsplit("/", 1)[-1]
        split = _SPLIT_GGUF.match(name)
        if split:
            group = item.multipart_group or split.group("stem")
            part = int(split.group("part"))
            count = int(split.group("count"))
            expected_counts[group] = count
            item = item.model_copy(
                update={
                    "multipart_group": group,
                    "multipart_index": part,
                    "multipart_count": count,
                }
            )
        else:
            group = item.relative_path.removesuffix(".gguf").removesuffix(".GGUF")
            expected_counts[group] = 1
        groups.setdefault(group, []).append(item)

    complete: list[tuple[str, str, tuple[ModelFileDescriptor, ...]]] = []
    for group, members in sorted(groups.items()):
        count = expected_counts[group]
        if len(members) != count:
            # An incomplete split artifact is never a runnable candidate.
            continue
        ordered = tuple(sorted(members, key=lambda item: item.multipart_index or 1))
        if count > 1 and [item.multipart_index for item in ordered] != list(range(1, count + 1)):
            continue
        quantization = _quantization(group)
        complete.append((group, quantization, ordered))

    quantization_counts: dict[str, int] = {}
    for _, quantization, _ in complete:
        quantization_counts[quantization] = quantization_counts.get(quantization, 0) + 1
    variants: list[ModelVariant] = []
    for group, quantization, ordered in complete:
        variant_id = (
            quantization
            if quantization != "UNKNOWN" and quantization_counts[quantization] == 1
            else group.rsplit("/", 1)[-1]
        )
        variants.append(
            ModelVariant(
                variant_id=variant_id,
                quantization=quantization,
                files=ordered,
                total_bytes=sum(item.size_bytes for item in ordered),
                quality_rank=_quality_rank(quantization),
            )
        )
    return tuple(variants)


def select_variant(
    variants: tuple[ModelVariant, ...],
    *,
    objective: Literal["speed", "throughput", "capacity", "balanced"] = "balanced",
    aggregate_usable_memory_bytes: int,
    local_fast_memory_bytes: int = 0,
    requested_variant: str | None = None,
    requested_quantization: str | None = None,
    quality_preference: float = 0.6,
    memory_headroom_fraction: float = 0.05,
) -> VariantSelection:
    if not variants:
        raise ValueError("repository exposes no complete GGUF variants")
    if aggregate_usable_memory_bytes <= 0:
        raise ValueError("aggregate usable memory must be positive")
    if not 0 <= quality_preference <= 1:
        raise ValueError("quality preference must be in [0, 1]")
    capacity = int(aggregate_usable_memory_bytes / (1.0 + memory_headroom_fraction))
    candidates: list[VariantCandidate] = []
    for variant in variants:
        matches = True
        reasons: list[str] = []
        if requested_variant is not None and requested_variant.lower() not in {
            variant.variant_id.lower(),
            variant.quantization.lower(),
        }:
            matches = False
            reasons.append("does not match requested variant")
        if requested_quantization is not None and (
            variant.quantization.lower() != requested_quantization.lower()
        ):
            matches = False
            reasons.append("does not match requested quantization")
        if variant.total_bytes > capacity:
            matches = False
            reasons.append("exceeds aggregate usable memory with headroom")
        local_fraction = min(1.0, local_fast_memory_bytes / max(1, variant.total_bytes))
        size_efficiency = capacity / max(capacity, variant.total_bytes)
        quality = min(1.0, variant.quality_rank / 16.0)
        if objective == "speed":
            score = 0.65 * local_fraction + 0.25 * size_efficiency + 0.10 * quality
        elif objective == "throughput":
            score = 0.45 * local_fraction + 0.40 * size_efficiency + 0.15 * quality
        elif objective == "capacity":
            score = 0.65 * quality + 0.35 * size_efficiency
        else:
            score = quality_preference * quality + (1 - quality_preference) * (
                0.6 * local_fraction + 0.4 * size_efficiency
            )
        candidates.append(
            VariantCandidate(
                variant=variant,
                feasible=matches,
                reason="feasible" if matches else "; ".join(reasons),
                score=score if matches else float("-inf"),
            )
        )
    feasible = [item for item in candidates if item.feasible]
    if not feasible:
        requested = requested_variant or requested_quantization or "automatic selection"
        raise ValueError(f"no complete feasible GGUF variant satisfies {requested}")
    selected = max(
        feasible,
        key=lambda item: (item.score, item.variant.quality_rank, -item.variant.total_bytes),
    ).variant
    return VariantSelection(selected=selected, candidates=tuple(candidates))


__all__ = [
    "ModelVariant",
    "VariantCandidate",
    "VariantSelection",
    "discover_gguf_variants",
    "select_variant",
]
