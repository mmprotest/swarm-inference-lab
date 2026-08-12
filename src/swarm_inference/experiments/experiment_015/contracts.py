"""Shared Experiment 015 evidence and economics contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Experiment015Error(RuntimeError):
    """Raised when an Experiment 015 correctness or evidence invariant fails."""


class EvidenceClass(StrEnum):
    """The only scientific classifications admitted by Experiment 015."""

    MEASURED = "MEASURED"
    SHAPED = "SHAPED"
    VALIDATED_MODEL = "VALIDATED MODEL"
    PROJECTED = "PROJECTED"


@dataclass(frozen=True, slots=True)
class EconomicsConfig:
    """Hardware-independent serving economics."""

    gpu_hourly_price_usd: float = 0.15
    output_price_per_million_usd: float = 15.0
    target_gpu_margin_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.gpu_hourly_price_usd <= 0:
            raise ValueError("GPU hourly price must be positive")
        if self.output_price_per_million_usd <= 0:
            raise ValueError("output token price must be positive")
        if not 0 <= self.target_gpu_margin_fraction < 1:
            raise ValueError("target GPU margin must be in [0, 1)")

    @property
    def break_even_tok_s_per_paid_gpu(self) -> float:
        return (
            self.gpu_hourly_price_usd
            * 1_000_000.0
            / (3600.0 * self.output_price_per_million_usd)
        )

    @property
    def margin_tok_s_per_paid_gpu(self) -> float:
        return self.break_even_tok_s_per_paid_gpu / (
            1.0 - self.target_gpu_margin_fraction
        )

    def cost_per_million_output_tokens(
        self, aggregate_tok_s: float, paid_gpu_equivalents: float
    ) -> float:
        if aggregate_tok_s <= 0 or paid_gpu_equivalents <= 0:
            raise ValueError("throughput and paid GPU-equivalents must be positive")
        return (
            paid_gpu_equivalents
            * self.gpu_hourly_price_usd
            * 1_000_000.0
            / (aggregate_tok_s * 3600.0)
        )
