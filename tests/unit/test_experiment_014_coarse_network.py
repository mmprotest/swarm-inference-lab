from __future__ import annotations

import pytest

from swarm_inference.experiments.experiment_014.coarse_network_analysis import (
    _select_coupled_admission,
)


def test_coupled_admission_uses_passing_bandwidth_at_the_selected_rtt() -> None:
    rows = [
        {
            "rtt_ms": 5.0,
            "bandwidth_gbps": 5.0,
            "capacity_retention_percent": 85.9,
            "useful_90_percent": False,
        },
        {
            "rtt_ms": 5.0,
            "bandwidth_gbps": 10.0,
            "capacity_retention_percent": 91.2,
            "useful_90_percent": True,
        },
        {
            "rtt_ms": 5.0,
            "bandwidth_gbps": 25.0,
            "capacity_retention_percent": 95.0,
            "useful_90_percent": True,
        },
    ]

    selected = _select_coupled_admission(rows, preferred_rtt_ms=5.0)

    assert selected["bandwidth_gbps"] == 10.0
    assert selected["useful_90_percent"] is True


def test_coupled_admission_fails_closed_without_a_passing_pair() -> None:
    rows = [
        {
            "rtt_ms": 5.0,
            "bandwidth_gbps": 5.0,
            "capacity_retention_percent": 85.9,
            "useful_90_percent": False,
        }
    ]

    with pytest.raises(ValueError, match=">=90%-capacity"):
        _select_coupled_admission(rows, preferred_rtt_ms=5.0)
