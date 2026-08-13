from __future__ import annotations

from swarm_inference.experiments.experiment_020.fleet import (
    FleetPolicy,
    NormalizedOffer,
    offer_matches,
    plan_fleet,
    render_launch_commands,
)


def _offer(**overrides: object) -> NormalizedOffer:
    values = {
        "offer_id": 1,
        "machine_id": 10,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_ram_mib": 24576,
        "num_gpus": 8,
        "reliability": 0.99,
        "total_price_per_hour": 4.0,
        "internet_down_mbps": 1000.0,
        "internet_up_mbps": 500.0,
        "internet_down_cost_per_gb": 0.0,
        "internet_up_cost_per_gb": 0.0,
        "disk_space_gb": 1000.0,
        "disk_cost_per_gb_month": 0.0,
        "cuda_max": "13.0",
        "driver_version": "580.65.06",
        "direct_ports": 4,
        "location": "AU",
        "verified": True,
        "nvlink": None,
    }
    values.update(overrides)
    return NormalizedOffer(**values)  # type: ignore[arg-type]


def test_gpu_classes_are_not_substring_matched() -> None:
    policy = FleetPolicy(required_pods=1)
    assert offer_matches(_offer(), policy, "RTX 3090")
    assert not offer_matches(
        _offer(gpu_name="NVIDIA GeForce RTX 3090 Ti"), policy, "RTX 3090"
    )


def test_cuda_and_verified_fields_fail_closed() -> None:
    policy = FleetPolicy(required_pods=1)
    assert not offer_matches(_offer(cuda_max="12.7"), policy, "RTX 3090")
    assert not offer_matches(_offer(driver_version="580.64.99"), policy, "RTX 3090")
    parsed = NormalizedOffer.from_raw(
        {
            "id": 1,
            "machine_id": 1,
            "gpu_name": "RTX 3090",
            "verified": "false",
        }
    )
    assert parsed.verified is False


def test_one_offer_per_machine() -> None:
    policy = FleetPolicy(required_pods=2)
    plan = plan_fleet(
        [
            _offer(offer_id=1, machine_id=10),
            _offer(offer_id=2, machine_id=10, total_price_per_hour=3.0),
            _offer(offer_id=3, machine_id=11),
        ],
        policy,
    )
    assert plan["currently_feasible"] is True
    assert {row["machine_id"] for row in plan["assignments"]} == {10, 11}


def test_render_is_complete_args_mode_and_contains_no_secret_value() -> None:
    plan = plan_fleet([_offer()], FleetPolicy(required_pods=2))
    commands = render_launch_commands(
        plan,
        image="registry/image@sha256:placeholder",
        disk_gb=200,
        run_id="test-run",
    )
    assert len(commands) == 2
    assert "--args controller-host-agent" in commands[0]
    assert "--args host-agent" in commands[1]
    assert "<REFRESH_REQUIRED_OFFER_ID_pod-001>" in commands[1]
    assert all("RUNTIME_SECRET_NOT_ARTIFACT" in command for command in commands)
    assert all("--onstart-cmd" not in command for command in commands)


def test_partial_plan_distinguishes_selected_cost_from_full_fleet_estimate() -> None:
    plan = plan_fleet([_offer(total_price_per_hour=3.0)], FleetPolicy(required_pods=2))

    assert plan["currently_feasible"] is False
    assert plan["currently_selected_hourly_price"] == 3.0
    assert plan["estimated_hourly_fleet_price"] == 6.0
