"""Queueing statistics shared by simulation reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def utilisation(*, busy_time_s: float, elapsed_s: float, parallelism: int = 1) -> float:
    if elapsed_s <= 0 or parallelism <= 0:
        return 0.0
    return min(1.0, busy_time_s / (elapsed_s * parallelism))
