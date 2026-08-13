"""Pre-spend E021 budget model; never performs a rental."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from .fleet import FleetPolicy, NormalizedOffer, offer_matches


def estimate_e021_cost(
    snapshot: Mapping[str, Any],
    pod_bundles: Mapping[str, Any],
    *,
    required_pods: int = 12,
    maximum_policy_rate_per_pod_hour: float = 8.0,
    disk_gb_per_pod: float = 200.0,
) -> dict[str, Any]:
    policy = FleetPolicy(
        required_pods=required_pods,
        minimum_disk_gb=disk_gb_per_pod,
        maximum_price_per_pod_hour=maximum_policy_rate_per_pod_hour,
    )
    matching = [
        offer
        for row in snapshot.get("offers", [])
        if offer_matches(
            offer := NormalizedOffer(**row),
            policy,
            policy.preferred_gpu_class,
        )
    ]
    best_by_machine: dict[int, NormalizedOffer] = {}
    for offer in matching:
        prior = best_by_machine.get(offer.machine_id)
        if prior is None or offer.total_price_per_hour < prior.total_price_per_hour:
            best_by_machine[offer.machine_id] = offer
    suitable = list(best_by_machine.values())
    observed_rates = [row.total_price_per_hour for row in suitable]
    observed_rate = (
        statistics.median(observed_rates)
        if observed_rates
        else maximum_policy_rate_per_pod_hour / 2
    )
    observed_rate_basis = (
        "median policy-matching complete RTX 3090 hosts"
        if observed_rates
        else "no matching host: explicit midpoint-of-policy scenario assumption"
    )
    maximum_pod_bytes = max(
        float(bundle["disk_requirement_bytes"])
        for bundle in pod_bundles.get("bundles", [])
    )
    advertised_down = min(
        (row.internet_down_mbps for row in suitable), default=500.0
    )
    download_hours = maximum_pod_bytes * 8 / (advertised_down * 1_000_000) / 3600
    disk_monthly = statistics.median(
        [row.disk_cost_per_gb_month for row in suitable]
    ) if suitable else 0.20
    down_cost = statistics.median(
        [row.internet_down_cost_per_gb for row in suitable]
    ) if suitable else 0.005
    transfer_gb = float(pod_bundles["total_fleet_transfer_bytes"]) / 1e9

    phase_hours = {
        "controller_and_worker_rental_startup": 0.25,
        "model_download": download_hours,
        "bootstrap_and_hash_verification": 0.25,
        "network_qualification": 0.25,
        "warmup": 0.25,
        "physical_benchmark": 1.25,
        "artifact_collection_and_teardown": 0.25,
    }
    expected_hours = sum(phase_hours.values())
    conservative_hours = expected_hours * 1.5 + 0.5
    worst_hours = 6.0

    def total(rate: float, hours: float, retry_multiplier: float) -> dict[str, float]:
        gpu = rate * required_pods * hours * retry_multiplier
        disk = disk_monthly * disk_gb_per_pod * required_pods / (30 * 24) * hours
        transfer = down_cost * transfer_gb * retry_multiplier
        return {
            "gpu_rental_usd": gpu,
            "disk_usd": disk,
            "data_transfer_usd": transfer,
            "total_usd": gpu + disk + transfer,
        }

    expected = total(observed_rate, expected_hours, 1.0)
    conservative_rate = max(observed_rate * 2.0, 4.0)
    conservative = total(conservative_rate, conservative_hours, 1.25)
    worst = total(maximum_policy_rate_per_pod_hour, worst_hours, 1.5)
    # A dollar cap is intentionally rounded upward to a user-readable gate.
    worst_cap = float((int(worst["total_usd"] / 50) + 1) * 50)
    return {
        "schema_version": "experiment-020-e021-cost-estimate-v1",
        "status": "ESTIMATE_ONLY_NO_CHARGES",
        "currency": "USD",
        "snapshot_at": snapshot.get("snapshot_at"),
        "required_pods": required_pods,
        "gpus_per_pod": 8,
        "total_gpus": required_pods * 8,
        "observed_suitable_p8_hosts": len(suitable),
        "offer_filter_matches_fleet_policy": True,
        "observed_median_rate_per_pod_hour": observed_rate,
        "expected_fleet_rate_per_hour": observed_rate * required_pods,
        "observed_rate_basis": observed_rate_basis,
        "conservative_rate_per_pod_hour": conservative_rate,
        "conservative_fleet_rate_per_hour": conservative_rate * required_pods,
        "maximum_policy_rate_per_pod_hour": maximum_policy_rate_per_pod_hour,
        "maximum_pod_model_bytes": int(maximum_pod_bytes),
        "advertised_download_mbps_used": advertised_down,
        "estimated_model_download_hours": download_hours,
        "phase_hours": phase_hours,
        "expected_elapsed_hours": expected_hours,
        "conservative_elapsed_hours": conservative_hours,
        "expected_cost": expected,
        "conservative_cost": conservative,
        "worst_case": worst,
        "worst_case_budget_cap_usd": worst_cap,
        "budget_requires_explicit_future_user_approval": True,
        "automatic_cost_kill_switch_required": True,
        "charge_incurred": False,
        "caveats": [
            "Offer IDs and rates are ephemeral and must be refreshed immediately before E021.",
            "Model downloads are concurrent across pods; elapsed time uses the slowest pod, while transfer fees use fleet bytes.",
            "The worst-case cap includes a 1.5x failed-attempt allowance and must not be treated as approved.",
        ],
    }


__all__ = ["estimate_e021_cost"]
