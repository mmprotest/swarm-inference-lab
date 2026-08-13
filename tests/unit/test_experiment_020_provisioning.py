from __future__ import annotations

import json

import pytest

from swarm_inference.experiments.experiment_020.fleet import FleetPolicy, plan_fleet
from swarm_inference.experiments.experiment_020.provisioning import (
    FAILURE_SCENARIOS,
    CostKillSwitch,
    FakeVastBackend,
    ProvisioningStateMachine,
    RentalLedger,
)


def _offers(count: int = 12) -> list[dict[str, object]]:
    return [
        {
            "id": 100 + index,
            "machine_id": 200 + index,
            "gpu_name": "NVIDIA GeForce RTX 3090",
            "gpu_ram": 24576,
            "num_gpus": 8,
            "reliability2": 0.99,
            "dph_total": 4.0 + index / 100,
            "inet_down": 1000,
            "inet_up": 500,
            "disk_space": 500,
            "cuda_max_good": "13.0",
            "driver_version": "580.65.06",
            "direct_port_count": 4,
            "verified": True,
        }
        for index in range(count)
    ]


def test_fleet_planner_requires_twelve_distinct_complete_p8_hosts() -> None:
    policy = FleetPolicy()
    plan = plan_fleet(_offers(), policy)
    assert plan["currently_feasible"] is True
    assert len(plan["assignments"]) == 12
    assert len({row["machine_id"] for row in plan["assignments"]}) == 12


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS)
def test_every_mock_failure_rolls_back_all_created_instances(scenario: str) -> None:
    result = ProvisioningStateMachine(FakeVastBackend(scenario)).run()
    assert result["status"] == "ABORTED_CLEAN"
    assert result["no_orphans"] is True
    assert result["ledger_valid"] is True
    assert result["created_instance_ids"] == result["destroyed_instance_ids"]


def test_unrelated_instance_is_protected() -> None:
    ledger = RentalLedger()
    with pytest.raises(PermissionError, match="UNRELATED"):
        ledger.append_action("DESTROYED", 999)


def test_cost_kill_switch_tears_down_at_budget_boundary_once() -> None:
    calls: list[str] = []
    switch = CostKillSwitch(10.0, lambda: calls.append("teardown"))
    assert switch.check([4.0, 6.0], 0.9) == 9.0
    assert not calls
    assert switch.check([4.0, 6.0], 1.0) == 10.0
    assert calls == ["teardown"]
    switch.check([4.0, 6.0], 2.0)
    assert calls == ["teardown"]


def test_state_machine_persists_valid_immutable_ledger(tmp_path) -> None:
    path = tmp_path / "rental-ledger.json"
    result = ProvisioningStateMachine(FakeVastBackend(), ledger_path=path).run()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert receipt["immutable"] is True
    assert receipt["valid"] is True
    assert len(receipt["records"]) == 12
    assert receipt["events"][-1]["action"] == "DESTROYED"
