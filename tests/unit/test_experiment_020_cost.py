from __future__ import annotations

from swarm_inference.experiments.experiment_020.cost import estimate_e021_cost


def _offer(machine_id: int, *, cuda: str, driver: str, rate: float) -> dict[str, object]:
    return {
        "offer_id": machine_id,
        "machine_id": machine_id,
        "gpu_name": "RTX 3090",
        "gpu_ram_mib": 24576,
        "num_gpus": 8,
        "reliability": 0.99,
        "total_price_per_hour": rate,
        "internet_down_mbps": 1000.0,
        "internet_up_mbps": 500.0,
        "internet_down_cost_per_gb": 0.0,
        "internet_up_cost_per_gb": 0.0,
        "disk_space_gb": 500.0,
        "disk_cost_per_gb_month": 0.0,
        "cuda_max": cuda,
        "driver_version": driver,
        "direct_ports": 4,
        "location": "AU",
        "verified": True,
        "nvlink": None,
    }


def test_cost_model_uses_the_same_cuda_driver_policy_as_fleet_planner() -> None:
    snapshot = {
        "snapshot_at": "test",
        "offers": [
            _offer(1, cuda="12.6", driver="560.35.03", rate=0.1),
            _offer(2, cuda="13.0", driver="580.65.06", rate=2.0),
        ],
    }
    bundles = {
        "bundles": [{"disk_requirement_bytes": 1_000_000_000}],
        "total_fleet_transfer_bytes": 12_000_000_000,
    }

    result = estimate_e021_cost(snapshot, bundles)

    assert result["offer_filter_matches_fleet_policy"] is True
    assert result["observed_suitable_p8_hosts"] == 1
    assert result["observed_median_rate_per_pod_hour"] == 2.0
    assert result["expected_fleet_rate_per_hour"] == 24.0
    assert result["observed_rate_basis"].startswith("median policy-matching")
