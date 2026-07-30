from __future__ import annotations

import os

import pytest


@pytest.mark.physical
@pytest.mark.skipif(
    os.environ.get("SWARM_RUN_PHYSICAL") != "1",
    reason="set SWARM_RUN_PHYSICAL=1 after provisioning at least two physical hosts",
)
def test_physical_lan_manual_gate() -> None:
    """The manual physical run is validated through its standard run artifacts."""

    artifact = os.environ.get("SWARM_PHYSICAL_RUN")
    assert artifact, "SWARM_PHYSICAL_RUN must point to a completed physical run"
    from swarm_inference.experiments.runner import validate_run

    assert validate_run(artifact) == []
