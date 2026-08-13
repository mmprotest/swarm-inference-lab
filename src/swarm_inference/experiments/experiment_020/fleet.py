"""Read-only Vast offer normalization, fleet planning, and command rendering."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .vast import plan_digest


def _first(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "verified"}


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", value)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


@dataclass(frozen=True, slots=True)
class NormalizedOffer:
    offer_id: int
    machine_id: int
    gpu_name: str
    gpu_ram_mib: int
    num_gpus: int
    reliability: float
    total_price_per_hour: float
    internet_down_mbps: float
    internet_up_mbps: float
    internet_down_cost_per_gb: float
    internet_up_cost_per_gb: float
    disk_space_gb: float
    disk_cost_per_gb_month: float
    cuda_max: str
    driver_version: str
    direct_ports: int
    location: str
    verified: bool
    nvlink: str | None

    @classmethod
    def from_raw(cls, row: Mapping[str, Any]) -> NormalizedOffer:
        return cls(
            offer_id=_int(_first(row, "id", "ask_contract_id", "offer_id"), -1),
            machine_id=_int(_first(row, "machine_id", "machine", "host_id"), -1),
            gpu_name=str(_first(row, "gpu_name", "gpu", default="UNKNOWN")),
            gpu_ram_mib=_int(_first(row, "gpu_ram", "gpu_ram_mib", "gpu_mem")),
            num_gpus=_int(_first(row, "num_gpus", "gpu_count")),
            reliability=_float(_first(row, "reliability2", "reliability")),
            total_price_per_hour=_float(
                _first(row, "dph_total", "total_price", "price_per_hour")
            ),
            internet_down_mbps=_float(_first(row, "inet_down", "internet_down")),
            internet_up_mbps=_float(_first(row, "inet_up", "internet_up")),
            internet_down_cost_per_gb=_float(
                _first(row, "inet_down_cost", "download_cost")
            ),
            internet_up_cost_per_gb=_float(
                _first(row, "inet_up_cost", "upload_cost")
            ),
            disk_space_gb=_float(_first(row, "disk_space", "disk_space_gb")),
            disk_cost_per_gb_month=_float(
                _first(row, "storage_cost", "disk_cost", "disk_cost_per_gb_month")
            ),
            cuda_max=str(_first(row, "cuda_max_good", "cuda_max", default="UNKNOWN")),
            driver_version=str(_first(row, "driver_version", default="UNKNOWN")),
            direct_ports=_int(
                _first(row, "direct_port_count", "direct_ports", "ports")
            ),
            location=str(
                _first(row, "geolocation", "location", "country", default="UNKNOWN")
            ),
            verified=_bool(_first(row, "verified", "verification", default=False)),
            nvlink=(
                str(_first(row, "nvlink_bw", "nvlink", "interconnect"))
                if _first(row, "nvlink_bw", "nvlink", "interconnect") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class FleetPolicy:
    preferred_gpu_class: str = "RTX 3090"
    fallback_gpu_classes: tuple[str, ...] = (
        "RTX 3090 Ti",
        "RTX A5000",
        "RTX A6000",
        "RTX 4090",
    )
    minimum_vram_gib: float = 20.0
    exact_gpus_per_pod: int = 8
    required_pods: int = 12
    minimum_reliability: float = 0.95
    minimum_disk_gb: float = 200.0
    minimum_internet_down_mbps: float = 500.0
    minimum_internet_up_mbps: float = 200.0
    minimum_direct_ports: int = 1
    # The frozen image is based on CUDA 13.0.1.  Vast's cuda_max field must
    # advertise at least that compatibility level; accepting 12.x hosts would
    # make bootstrap failure predictable before rental.
    minimum_cuda: str = "13.0"
    minimum_driver: str = "580.65.06"
    maximum_price_per_pod_hour: float = 8.0
    verified_only: bool = True
    homogeneous_primary_fleet: bool = True

    @property
    def total_gpus(self) -> int:
        return self.required_pods * self.exact_gpus_per_pod


def _name_matches(actual: str, desired: str) -> bool:
    def normalize(value: str) -> str:
        normalized = "".join(character for character in value.lower() if character.isalnum())
        for prefix in ("nvidiageforce", "nvidia", "geforce"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized

    # Exact class separation matters: an RTX 3090 Ti cannot silently enter the
    # homogeneous RTX 3090 headline fleet (and vice versa).
    return normalize(desired) == normalize(actual)


def offer_matches(offer: NormalizedOffer, policy: FleetPolicy, gpu_class: str) -> bool:
    return (
        offer.offer_id >= 0
        and offer.machine_id >= 0
        and _name_matches(offer.gpu_name, gpu_class)
        and offer.gpu_ram_mib >= int(policy.minimum_vram_gib * 1024)
        and offer.num_gpus >= policy.exact_gpus_per_pod
        and offer.reliability >= policy.minimum_reliability
        and offer.disk_space_gb >= policy.minimum_disk_gb
        and offer.internet_down_mbps >= policy.minimum_internet_down_mbps
        and offer.internet_up_mbps >= policy.minimum_internet_up_mbps
        and offer.direct_ports >= policy.minimum_direct_ports
        and _version_tuple(offer.cuda_max) >= _version_tuple(policy.minimum_cuda)
        and _version_tuple(offer.driver_version)
        >= _version_tuple(policy.minimum_driver)
        and offer.total_price_per_hour <= policy.maximum_price_per_pod_hour
        and (offer.verified or not policy.verified_only)
    )


def plan_fleet(
    offers: Iterable[Mapping[str, Any] | NormalizedOffer],
    policy: FleetPolicy,
    *,
    gpu_class: str | None = None,
) -> dict[str, Any]:
    selected_class = gpu_class or policy.preferred_gpu_class
    normalized = [
        row if isinstance(row, NormalizedOffer) else NormalizedOffer.from_raw(row)
        for row in offers
    ]
    matching = [row for row in normalized if offer_matches(row, policy, selected_class)]
    # At most one offer per physical machine; a pod must stay within one host.
    best_by_machine: dict[int, NormalizedOffer] = {}
    for row in matching:
        prior = best_by_machine.get(row.machine_id)
        if prior is None or row.total_price_per_hour < prior.total_price_per_hour:
            best_by_machine[row.machine_id] = row
    candidates = sorted(
        best_by_machine.values(),
        key=lambda row: (-row.reliability, row.total_price_per_hour, row.offer_id),
    )
    chosen = candidates[: policy.required_pods]
    assignments = [
        {
            "pod_id": f"pod-{index:03d}",
            "offer_id": row.offer_id,
            "machine_id": row.machine_id,
            "gpu_name": row.gpu_name,
            "gpu_count": policy.exact_gpus_per_pod,
            "available_gpu_count": row.num_gpus,
            "hourly_rate": row.total_price_per_hour,
            "location": row.location,
            "reliability": row.reliability,
        }
        for index, row in enumerate(chosen)
    ]
    feasible = len(chosen) >= policy.required_pods
    selected_hourly_price = sum(row.total_price_per_hour for row in chosen)
    candidate_prices = sorted(row.total_price_per_hour for row in candidates)
    projected_full_fleet_price = (
        candidate_prices[len(candidate_prices) // 2] * policy.required_pods
        if candidate_prices
        else None
    )
    body: dict[str, Any] = {
        "schema_version": "experiment-020-vast-dry-run-plan-v1",
        "experiment_id": "experiment-021",
        "approved": False,
        "read_only": True,
        "gpu_class": selected_class,
        "homogeneous": True,
        "required_pods": policy.required_pods,
        "gpus_per_pod": policy.exact_gpus_per_pod,
        "total_gpus": policy.total_gpus,
        "matching_p8_hosts": len(candidates),
        "assignments": assignments,
        "currently_feasible": feasible,
        "availability": "YES" if feasible else "PARTIAL" if chosen else "NO",
        "currently_selected_hourly_price": selected_hourly_price,
        "estimated_hourly_fleet_price": projected_full_fleet_price,
        "estimated_hourly_fleet_price_basis": (
            "median matching complete-host rate times required pods"
            if candidate_prices
            else "unavailable: no matching complete host"
        ),
        "policy": dataclasses.asdict(policy),
    }
    body["plan_sha256"] = plan_digest(body)
    return body


def render_launch_commands(
    plan: Mapping[str, Any],
    *,
    image: str,
    disk_gb: int,
    run_id: str,
    controller_port: int = 7443,
) -> list[str]:
    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a non-empty shell-safe identifier")
    assignments = {str(row["pod_id"]): row for row in plan.get("assignments", [])}
    commands = []
    for index in range(int(plan["required_pods"])):
        pod_id = f"pod-{index:03d}"
        assignment = assignments.get(pod_id)
        offer_id = (
            str(int(assignment["offer_id"]))
            if assignment is not None
            else f"<REFRESH_REQUIRED_OFFER_ID_{pod_id}>"
        )
        label = f"swarm-e021-{run_id}-{pod_id}"
        common_environment = (
            f"-e SWARM_RUN_ID={run_id} -e SWARM_POD_ID={pod_id} "
            "-e SWARM_RUN_CREDENTIAL_B64=<RUNTIME_SECRET_NOT_ARTIFACT> "
            "-e SWARM_WORKER_MANIFEST_ROOT=/opt/swarm/manifests"
        )
        if index == 0:
            environment = common_environment + f" -p {controller_port}:{controller_port}/tcp"
            arguments = (
                "controller-host-agent --workers-per-pod 8 "
                f"--listen-host 0.0.0.0 --port {controller_port}"
            )
        else:
            environment = (
                common_environment
                + " -e SWARM_CONTROLLER=<DISCOVERED_CONTROLLER_ADDRESS>"
            )
            arguments = "host-agent --workers-per-pod 8"
        commands.append(
            "vastai create instance "
            f"{offer_id} --image {image} --disk {int(disk_gb)} "
            f"--label {label} --cancel-unavail --env \"{environment}\" "
            f"--args {arguments}"
        )
    return commands


def market_summary(
    offers: Sequence[NormalizedOffer], policy: FleetPolicy, gpu_class: str
) -> dict[str, Any]:
    matching = [offer for offer in offers if offer_matches(offer, policy, gpu_class)]
    prices = sorted(offer.total_price_per_hour for offer in matching)
    return {
        "gpu_class": gpu_class,
        "matching_offers": len(matching),
        "matching_p8_hosts": len({offer.machine_id for offer in matching}),
        "suitable_gpus": sum(
            min(offer.num_gpus, policy.exact_gpus_per_pod) for offer in matching
        ),
        "price_per_host_hour_min": prices[0] if prices else None,
        "price_per_host_hour_median": (
            prices[len(prices) // 2] if prices else None
        ),
        "price_per_host_hour_max": prices[-1] if prices else None,
        "locations": sorted({offer.location for offer in matching}),
        "reliability_min": min((offer.reliability for offer in matching), default=None),
        "reliability_max": max((offer.reliability for offer in matching), default=None),
    }


__all__ = [
    "FleetPolicy",
    "NormalizedOffer",
    "market_summary",
    "offer_matches",
    "plan_fleet",
    "render_launch_commands",
]
