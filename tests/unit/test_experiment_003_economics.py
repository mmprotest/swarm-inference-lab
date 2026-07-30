from __future__ import annotations

import pytest

from swarm_inference.experiments.fanout_economics import (
    minimum_lease_duration,
    productive_fraction,
    productive_tokens,
)


def test_economics_short_zero_and_long_startup_cases() -> None:
    assert productive_fraction(30, 60) == 0
    assert productive_tokens(30, 60, 100) == 0
    assert productive_fraction(60, 0) == 1
    assert productive_tokens(60, 0, 2) == 120
    assert productive_fraction(86_400, 10) == pytest.approx((86_400 - 10) / 86_400)


@pytest.mark.parametrize("target", [0.50, 0.75, 0.90, 0.95])
def test_all_required_break_even_targets(target: float) -> None:
    startup = 12.5
    lease = minimum_lease_duration(startup, target)
    assert productive_fraction(lease, startup) == pytest.approx(target)
