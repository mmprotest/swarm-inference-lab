"""Secondary stochastic-hedging contracts for Experiment 023.

The actual E023 run stopped at the mandatory negative-control gate before the
secondary hedging phase.  These utilities preserve the preregistered policy
boundary and make its non-clairvoyance testable without claiming a hedging
result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

HEDGE_SEEDS: Final = tuple(23_023_000 + index for index in range(32))


@dataclass(frozen=True, slots=True)
class HedgeDeadlineInputs:
    """Only prediction-time values visible to the hedge policy."""

    initial_dispatch_time_ms: float
    predicted_input_transfer_ms: float
    empirical_p95_compute_ms_scaled_for_node: float
    predicted_output_transfer_ms: float

    @property
    def deadline_ms(self) -> float:
        return (
            self.initial_dispatch_time_ms
            + self.predicted_input_transfer_ms
            + self.empirical_p95_compute_ms_scaled_for_node
            + self.predicted_output_transfer_ms
        )


@dataclass(frozen=True, slots=True)
class HedgeLaunchDecision:
    launch: bool
    deadline_ms: float
    contribution_arrived_by_deadline: bool


def decide_hedge_launch(
    prediction: HedgeDeadlineInputs,
    *,
    observed_contribution_arrival_ms: float | None,
) -> HedgeLaunchDecision:
    """Launch only from state observable at the deterministic deadline.

    There is deliberately no realized or future service-duration argument.
    An arrival later than the deadline is indistinguishable from no arrival at
    decision time and therefore triggers the exact duplicate.
    """

    deadline = prediction.deadline_ms
    arrived = (
        observed_contribution_arrival_ms is not None
        and observed_contribution_arrival_ms <= deadline
    )
    return HedgeLaunchDecision(
        launch=not arrived,
        deadline_ms=deadline,
        contribution_arrived_by_deadline=arrived,
    )


class PhysicalServiceSamplePool:
    """Bootstrap pool keyed only by KDA/MLA layer type and row count."""

    def __init__(self, samples: dict[tuple[str, int], tuple[float, ...]]) -> None:
        if not samples:
            raise ValueError("hedging sample pool cannot be empty")
        if any(not values for values in samples.values()):
            raise ValueError("every hedging sample cell must be populated")
        self._samples = dict(samples)

    @classmethod
    def from_csv(cls, path: Path) -> PhysicalServiceSamplePool:
        grouped: dict[tuple[str, int], list[float]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row["layer_type"]), int(row["rows"]))
                duration = float(row["cuda_ms"])
                if not np.isfinite(duration) or duration < 0.0:
                    raise ValueError("physical hedge sample is invalid")
                grouped.setdefault(key, []).append(duration)
        return cls({key: tuple(values) for key, values in grouped.items()})

    def draw_compute_ms(
        self,
        *,
        layer_type: str,
        rows: int,
        compute_multiplier: float,
        rng: np.random.Generator,
    ) -> float:
        """Draw after routing; the policy never receives this realized value."""

        if compute_multiplier <= 0.0 or compute_multiplier > 1.0:
            raise ValueError("compute multiplier must be in (0, 1]")
        values = self._samples[(layer_type, rows)]
        draw = values[int(rng.integers(0, len(values)))]
        return draw / compute_multiplier

    def empirical_p95_ms(
        self, *, layer_type: str, rows: int, compute_multiplier: float
    ) -> float:
        if compute_multiplier <= 0.0 or compute_multiplier > 1.0:
            raise ValueError("compute multiplier must be in (0, 1]")
        return float(
            np.quantile(
                self._samples[(layer_type, rows)], 0.95, method="linear"
            )
            / compute_multiplier
        )


def common_random_draw_indices(
    *, seed: int, task_count: int, sample_count: int
) -> tuple[int, ...]:
    """Create the common-random-number schedule for hedge/no-hedge arms."""

    if seed not in HEDGE_SEEDS:
        raise ValueError("seed is outside the frozen E023 hedge seed set")
    if task_count < 0 or sample_count <= 0:
        raise ValueError("invalid common-random-number dimensions")
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.integers(0, sample_count, task_count))


__all__ = [
    "HEDGE_SEEDS",
    "HedgeDeadlineInputs",
    "HedgeLaunchDecision",
    "PhysicalServiceSamplePool",
    "common_random_draw_indices",
    "decide_hedge_launch",
]
