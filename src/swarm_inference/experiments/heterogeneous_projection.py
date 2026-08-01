"""Event-driven network and node-availability replay for Experiment 007."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swarm_inference.config.models import JitterDistribution, NetworkProfile
from swarm_inference.planner import NodeRole
from swarm_inference.simulation.clock import SimClock
from swarm_inference.simulation.network import NetworkEmulator
from swarm_inference.worker.abi import ResultClassification

NETWORK_PROFILES: dict[str, NetworkProfile] = {
    "localhost": NetworkProfile(
        name="localhost",
        base_latency_ms=0.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=20_000_000_000,
        download_bandwidth_bytes_s=20_000_000_000,
        measured=True,
    ),
    "home_lan": NetworkProfile(
        name="1 Gbps LAN, 1 ms",
        base_latency_ms=1.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=125_000_000,
        download_bandwidth_bytes_s=125_000_000,
    ),
    "residential_fast": NetworkProfile(
        name="100 Mbps residential, 10 ms",
        base_latency_ms=10.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=12_500_000,
        download_bandwidth_bytes_s=12_500_000,
    ),
    "residential_slow": NetworkProfile(
        name="50 Mbps residential, 20 ms",
        base_latency_ms=20.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=6_250_000,
        download_bandwidth_bytes_s=6_250_000,
    ),
    "regional_wan": NetworkProfile(
        name="20 Mbps WAN, 50 ms",
        base_latency_ms=50.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=2_500_000,
        download_bandwidth_bytes_s=2_500_000,
    ),
    "constrained_wan": NetworkProfile(
        name="10 Mbps WAN, 100 ms",
        base_latency_ms=100.0,
        jitter_distribution=JitterDistribution.NONE,
        upload_bandwidth_bytes_s=1_250_000,
        download_bandwidth_bytes_s=1_250_000,
    ),
}


@dataclass(frozen=True, slots=True)
class MeasuredRoleTrace:
    role: NodeRole
    request_payload_bytes: int
    response_payload_bytes: int
    measured_compute_ms: float
    verified_tokens: float
    baseline_service_ms: float
    measured_marginal_verified_tps_gain: float = 0.0
    failure_recovery_ms: float = 0.0
    verification_ms: float = 0.0
    availability_requirement: float = 1.0


def replay_measured_role(
    trace: MeasuredRoleTrace,
    profile: NetworkProfile,
    *,
    availability_fraction: float = 1.0,
) -> dict[str, Any]:
    """Replay measured bytes/compute with events; never sleep or rerun a model."""

    if not 0 <= availability_fraction <= 1:
        raise ValueError("availability fraction must be between zero and one")
    clock = SimClock()
    network = NetworkEmulator(profile, seed=7007)
    timeline: list[dict[str, float | str]] = []
    completed_at = 0.0

    def request_arrived() -> None:
        nonlocal completed_at
        timeline.append({"event": "request_arrived", "time_s": clock.now_s})

        def compute_complete() -> None:
            timeline.append({"event": "compute_complete", "time_s": clock.now_s})
            returned = network.transmit(
                source="worker",
                destination="coordinator",
                now_s=clock.now_s,
                payload_bytes=trace.response_payload_bytes,
            )

            def response_arrived() -> None:
                nonlocal completed_at
                completed_at = clock.now_s
                timeline.append({"event": "response_arrived", "time_s": clock.now_s})

            clock.schedule_at(
                returned.completed_at_s,
                response_arrived,
                name="response_arrived",
            )

        clock.schedule_in(
            trace.measured_compute_ms / 1000,
            compute_complete,
            name="compute_complete",
        )

    sent = network.transmit(
        source="coordinator",
        destination="worker",
        now_s=0.0,
        payload_bytes=trace.request_payload_bytes,
    )
    clock.schedule_at(sent.completed_at_s, request_arrived, name="request_arrived")
    clock.run()
    total_ms = completed_at * 1000 + trace.failure_recovery_ms + trace.verification_ms
    effective_tokens = trace.verified_tokens * availability_fraction
    effective_tps = effective_tokens / max(total_ms / 1000, 1e-12)
    measured_local_service_ms = (
        trace.measured_compute_ms + trace.failure_recovery_ms + trace.verification_ms
    )
    measured_role_tps = trace.verified_tokens / max(measured_local_service_ms / 1000, 1e-12)
    network_and_availability_loss_tps = max(0.0, measured_role_tps - effective_tps)
    projected_marginal_gain = (
        trace.measured_marginal_verified_tps_gain - network_and_availability_loss_tps
    )
    baseline_tps = trace.verified_tokens / max(trace.baseline_service_ms / 1000, 1e-12)
    return {
        "classification": ResultClassification.EMULATED_NETWORK.value,
        "role": trace.role.value,
        "network_profile": profile.name,
        "one_way_latency_ms": profile.base_latency_ms,
        "bandwidth_mbps": min(
            profile.upload_bandwidth_bytes_s,
            profile.download_bandwidth_bytes_s,
        )
        * 8
        / 1_000_000,
        "request_payload_bytes": trace.request_payload_bytes,
        "response_payload_bytes": trace.response_payload_bytes,
        "measured_compute_ms": trace.measured_compute_ms,
        "projected_end_to_end_ms": total_ms,
        "projected_verified_tokens_per_second": effective_tps,
        "baseline_tokens_per_second": baseline_tps,
        "measured_marginal_verified_tps_gain": trace.measured_marginal_verified_tps_gain,
        "network_and_availability_loss_tps": network_and_availability_loss_tps,
        "projected_marginal_verified_tps_gain": projected_marginal_gain,
        "projected_throughput_change_fraction": projected_marginal_gain / max(baseline_tps, 1e-12),
        "availability_fraction": availability_fraction,
        "availability_requirement": trace.availability_requirement,
        "viable": projected_marginal_gain > 0,
        "timeline": timeline,
    }


def replay_network_matrix(
    traces: list[MeasuredRoleTrace],
    profile_names: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in profile_names:
        try:
            profile = NETWORK_PROFILES[name]
        except KeyError as exc:
            raise ValueError(f"unknown Experiment 007 network profile {name!r}") from exc
        rows.extend(replay_measured_role(trace, profile) for trace in traces)
    return rows


def speculative_break_even_rows(
    *,
    acceptance_rate: float,
    mean_accepted_length: float,
    target_tokens_per_second: float,
    target_verification_ms: float,
    draft_length: int,
    request_payload_bytes: int,
    response_payload_bytes: int,
) -> list[dict[str, Any]]:
    if not 0 <= acceptance_rate <= 1:
        raise ValueError("acceptance rate must be between zero and one")
    rows: list[dict[str, Any]] = []
    committed_per_block = max(1.0, mean_accepted_length + (1.0 - acceptance_rate))
    for draft_tps in (2, 5, 10, 20, 40, 80):
        draft_ms = draft_length / draft_tps * 1000
        for latency_ms in (0, 1, 5, 10, 20, 50, 100):
            network_ms = latency_ms * 2
            total_ms = draft_ms + target_verification_ms + network_ms
            projected_tps = committed_per_block / max(total_ms / 1000, 1e-12)
            speedup = projected_tps / target_tokens_per_second - 1
            rows.append(
                {
                    "classification": ResultClassification.EMULATED_NETWORK.value,
                    "draft_speed_tokens_per_second": draft_tps,
                    "one_way_latency_ms": latency_ms,
                    "draft_length": draft_length,
                    "acceptance_rate": acceptance_rate,
                    "mean_accepted_length": mean_accepted_length,
                    "accepted_tokens_per_verification": committed_per_block,
                    "request_payload_bytes": request_payload_bytes,
                    "response_payload_bytes": response_payload_bytes,
                    "target_verification_ms": target_verification_ms,
                    "projected_tokens_per_second": projected_tps,
                    "single_request_speedup_fraction": speedup,
                    "draft_utilisation": draft_ms / max(total_ms, 1e-12),
                    "target_utilisation": target_verification_ms / max(total_ms, 1e-12),
                    "useful": speedup > 0,
                }
            )
    return rows


def availability_economics_rows(
    traces: list[MeasuredRoleTrace],
    *,
    acquisition_seconds: dict[NodeRole, float],
    conversion_seconds: dict[NodeRole, float],
    load_seconds: dict[NodeRole, float],
    warmup_seconds: dict[NodeRole, float],
    lease_durations_seconds: tuple[int, ...] | list[int],
    artifact_residency: str = "prevalidated_local_cache",
    all_roles: list[NodeRole] | tuple[NodeRole, ...] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        readiness = (
            acquisition_seconds.get(trace.role, 0.0)
            + conversion_seconds.get(trace.role, 0.0)
            + load_seconds.get(trace.role, 0.0)
            + warmup_seconds.get(trace.role, 0.0)
        )
        steady_tps = trace.verified_tokens / max(trace.measured_compute_ms / 1000, 1e-12)
        marginal_tps = trace.measured_marginal_verified_tps_gain
        for lease in lease_durations_seconds:
            productive_seconds = max(0.0, lease - readiness)
            productive_fraction = productive_seconds / lease
            contributed = productive_seconds * steady_tps
            marginal_contributed = productive_seconds * marginal_tps
            rows.append(
                {
                    "classification": ResultClassification.PROJECTED_DEVICE_PROFILE.value,
                    "role": trace.role.value,
                    "role_measurement_status": "measured_compute_trace",
                    "lease_duration_seconds": lease,
                    "model_acquisition_seconds": acquisition_seconds.get(trace.role, 0.0),
                    "model_conversion_seconds": conversion_seconds.get(trace.role, 0.0),
                    "model_load_seconds": load_seconds.get(trace.role, 0.0),
                    "warmup_seconds": warmup_seconds.get(trace.role, 0.0),
                    "artifact_residency": artifact_residency,
                    "time_to_first_useful_work_seconds": readiness,
                    "minimum_useful_lease_duration_seconds": (
                        readiness if marginal_tps > 0 else None
                    ),
                    "productive_fraction": productive_fraction,
                    "verified_tokens_contributed": contributed,
                    "measured_steady_state_marginal_verified_tps_gain": marginal_tps,
                    "marginal_verified_tokens_contributed": marginal_contributed,
                    "positive_for_lease": productive_seconds > 0 and marginal_contributed > 0,
                }
            )
    measured_roles = {trace.role for trace in traces}
    for role in all_roles or ():
        if role in measured_roles:
            continue
        readiness = (
            acquisition_seconds.get(role, 0.0)
            + conversion_seconds.get(role, 0.0)
            + load_seconds.get(role, 0.0)
            + warmup_seconds.get(role, 0.0)
        )
        for lease in lease_durations_seconds:
            rows.append(
                {
                    "classification": ResultClassification.PROJECTED_DEVICE_PROFILE.value,
                    "role": role.value,
                    "role_measurement_status": (
                        "idle" if role == NodeRole.IDLE else "unsupported_or_unmeasured"
                    ),
                    "lease_duration_seconds": lease,
                    "model_acquisition_seconds": acquisition_seconds.get(role, 0.0),
                    "model_conversion_seconds": conversion_seconds.get(role, 0.0),
                    "model_load_seconds": load_seconds.get(role, 0.0),
                    "warmup_seconds": warmup_seconds.get(role, 0.0),
                    "artifact_residency": artifact_residency,
                    "time_to_first_useful_work_seconds": readiness,
                    "minimum_useful_lease_duration_seconds": None,
                    "productive_fraction": 0.0,
                    "verified_tokens_contributed": 0.0,
                    "measured_steady_state_marginal_verified_tps_gain": 0.0,
                    "marginal_verified_tokens_contributed": 0.0,
                    "positive_for_lease": False,
                }
            )
    return rows
