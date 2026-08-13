"""Corrected worker-level E020 performance and accounting model."""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

from swarm_inference.experiments.experiment_019.events import (
    DeterministicMicroworkerEngine,
    MicroworkerTask,
    NetworkProfile,
)
from swarm_inference.experiments.experiment_019.finalize import _physical_service
from swarm_inference.experiments.experiment_019.placement import PlacementSpec
from swarm_inference.experiments.experiment_019.simulation import (
    NETWORK_PROFILES,
    MeasuredShardService,
    SimulationConfiguration,
    build_worker_dag,
    simulate,
)


def measured_service(
    attention: Mapping[str, Any],
    legacy_expert: Mapping[str, Any],
    grouped_expert: Mapping[str, Any],
    other: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    degree: int,
    rows: int,
) -> MeasuredShardService:
    """Build post-fusion service inputs without changing other measured arms."""

    base = _physical_service(
        attention,
        legacy_expert,
        other,
        protocol,
        degree=degree,
        rows=rows,
    )
    result = next(
        row
        for row in grouped_expert["results"]
        if int(row["rows"]) == rows and int(row["stripe_degree"]) == degree
    )
    workers = sorted(result["workers"], key=lambda row: int(row["stripe_index"]))
    return replace(
        base,
        expert_worker_ms=tuple(
            float(worker["e020_grouped"]["wall"]["p50_ms"]) for worker in workers
        ),
        source=(
            "RTX 5090 post-fusion physical worker services: refreshed attention/"
            "projection/shared/endpoint/reduction plus exact grouped top-16 experts"
        ),
    )


def accounting_reconciliation(
    service: MeasuredShardService,
    configuration: SimulationConfiguration,
) -> dict[str, Any]:
    tasks = build_worker_dag(service, configuration)
    result, run = simulate(service, configuration)
    measured_compute = sum(
        task.service_time_ms for task in tasks if isinstance(task, MicroworkerTask)
    )
    trace_compute = sum(
        record.duration_ms for record in run.records if record.resource_type == "microworker"
    )
    network_duration = sum(
        record.duration_ms for record in run.records if record.resource_type == "network"
    )
    network_components = {
        "base_network_latency_ms": sum(record.base_network_latency_ms for record in run.records),
        "serialization_ms": sum(record.serialization_ms for record in run.records),
        "software_protocol_overhead_ms": sum(
            record.software_overhead_ms for record in run.records
        ),
    }
    record_by_id = {record.task_id: record for record in run.records}
    queue_wait = 0.0
    state_wait = 0.0
    for record in run.records:
        dependency_finish = max(
            (record_by_id[value].finish_time for value in record.dependency_ids),
            default=0.0,
        )
        wait = max(0.0, record.start_time - dependency_finish)
        queue_wait += wait
        if record.state_refs:
            state_wait += wait
    collective_records = [
        record for record in run.records if record.collective_algorithm is not None
    ]
    tolerance = max(1e-9, measured_compute * 1e-12)
    compute_delta = trace_compute - measured_compute
    network_delta = network_duration - sum(network_components.values())
    return {
        "schema_version": "experiment-020-accounting-reconciliation-v1",
        "configuration": {
            **asdict(configuration.placement),
            "block": configuration.block,
            "chunk": configuration.chunk,
            "local_profile": configuration.local_profile.name,
            "inter_pod_profile": configuration.inter_pod_profile.name,
        },
        "measured_bottom_up_worker_compute_work_ms": measured_compute,
        "sum_worker_compute_durations_ms": trace_compute,
        "event_run_total_compute_work_ms": run.total_compute_work_ms,
        "compute_delta_ms": compute_delta,
        "accounting_tolerance_ms": tolerance,
        "compute_invariant_pass": abs(compute_delta) <= tolerance
        and abs(run.total_compute_work_ms - measured_compute) <= tolerance,
        "network_event_count": sum(
            record.resource_type == "network" for record in run.records
        ),
        "network_duration_ms": network_duration,
        **network_components,
        "network_component_delta_ms": network_delta,
        "network_invariant_pass": abs(network_delta) <= max(1e-9, network_duration * 1e-12),
        "collective_event_count": len(collective_records),
        "collective_steps": sum(record.collective_steps for record in collective_records),
        "state_wait_ms": state_wait,
        "queue_wait_ms": queue_wait,
        "launch_overhead_policy": "physical worker wall service includes CUDA launch and Python/native call overhead",
        "result": result,
        "status": (
            "PASS"
            if abs(compute_delta) <= tolerance
            and abs(network_delta) <= max(1e-9, network_duration * 1e-12)
            else "FAIL"
        ),
    }


def candidate_projection(
    service: MeasuredShardService,
    *,
    depth_span: int,
    block: int = 16,
    chunk: int = 1,
    compute_slowdown: float = 1.0,
    local_profile: NetworkProfile | None = None,
    inter_pod_profile: NetworkProfile | None = None,
    heterogeneity: str = "homogeneous",
    jitter_fraction: float = 0.0,
) -> tuple[dict[str, Any], Any]:
    placement = PlacementSpec(
        memory_cap_gib=20,
        stripe_degree=service.degree,
        depth_span=depth_span,
        chunk_rows=chunk,
        hardware_class="PREDICTED_FROM_RTX_5090_SHARD_SERVICE_NOT_SM86",
    )
    configuration = SimulationConfiguration(
        placement=placement,
        block=block,
        chunk=chunk,
        local_profile=local_profile or NETWORK_PROFILES["canonical_fast_local"],
        inter_pod_profile=inter_pod_profile or NETWORK_PROFILES["canonical_inter_pod"],
        compute_slowdown=compute_slowdown,
        heterogeneity=heterogeneity,
        jitter_fraction=jitter_fraction,
    )
    return simulate(service, configuration)


def uncertainty_envelope(
    service: MeasuredShardService,
    *,
    depth_span: int,
    blocks: Sequence[int] = (7, 12, 16),
    chunks: Sequence[int] = (1, 2, 4),
) -> list[dict[str, Any]]:
    scenarios = {
        "optimistic": {
            "compute": 0.9,
            "local": NetworkProfile("optimistic_local", 0.05, 100.0, 0.01),
            "inter": NetworkProfile("optimistic_inter", 1.0, 25.0, 0.02),
            "jitter": 0.0,
        },
        "nominal": {
            "compute": 1.0,
            "local": NetworkProfile("nominal_local", 0.25, 25.0, 0.02),
            "inter": NetworkProfile("nominal_inter", 5.0, 10.0, 0.04),
            "jitter": 0.0,
        },
        "conservative": {
            "compute": 1.3,
            "local": NetworkProfile("conservative_local", 1.0, 10.0, 0.05),
            "inter": NetworkProfile("conservative_inter", 10.0, 2.0, 0.10),
            "jitter": 0.20,
        },
    }
    rows = []
    for block in blocks:
        for chunk in chunks:
            if chunk > block + 1:
                continue
            for name, values in scenarios.items():
                result, _run = candidate_projection(
                    service,
                    depth_span=depth_span,
                    block=block,
                    chunk=chunk,
                    compute_slowdown=float(values["compute"]),
                    local_profile=values["local"],
                    inter_pod_profile=values["inter"],
                    jitter_fraction=float(values["jitter"]),
                )
                rows.append(
                    {
                        "scenario": name,
                        "block": block,
                        "chunk": chunk,
                        "predicted": True,
                        "target_pass_ms": result["target_pass_ms"],
                        "tok_s_per_user": result["exact_tok_s_per_user"],
                        "compute_multiplier": values["compute"],
                        "local_rtt_ms": values["local"].rtt_ms,
                        "local_bandwidth_gbps": values["local"].bandwidth_gbps,
                        "inter_pod_rtt_ms": values["inter"].rtt_ms,
                        "inter_pod_bandwidth_gbps": values["inter"].bandwidth_gbps,
                        "network_jitter_fraction": values["jitter"],
                    }
                )
    return rows


def slowdown_sensitivity(
    service: MeasuredShardService,
    *,
    depth_span: int,
    block: int = 16,
    chunk: int = 1,
) -> list[dict[str, Any]]:
    rows = []
    for degradation in (0.0, 0.10, 0.20, 0.30, 0.50):
        result, _run = candidate_projection(
            service,
            depth_span=depth_span,
            block=block,
            chunk=chunk,
            compute_slowdown=1.0 + degradation,
        )
        rows.append(
            {
                "compute_degradation_percent": degradation * 100,
                "compute_multiplier": 1.0 + degradation,
                "predicted_tok_s_per_user": result["exact_tok_s_per_user"],
                "predicted_target_pass_ms": result["target_pass_ms"],
                "passes_5_tok_s": result["exact_tok_s_per_user"] >= 5.0,
            }
        )
    return rows


def _prefixed_tasks(tasks: Sequence[Any], request_index: int) -> list[Any]:
    prefix = f"r{request_index:03d}."
    output = []
    for task in tasks:
        output.append(
            dataclasses.replace(
                task,
                task_id=prefix + task.task_id,
                dependency_ids=tuple(prefix + value for value in task.dependency_ids),
                request_id=f"concurrent-request-{request_index:03d}",
            )
        )
    return output


def concurrency_sweep(
    service: MeasuredShardService,
    *,
    depth_span: int,
    requests: Sequence[int] = (1, 2, 4, 8, 16),
    block: int = 16,
    chunk: int = 1,
) -> list[dict[str, Any]]:
    placement = PlacementSpec(20, service.degree, depth_span, chunk)
    configuration = SimulationConfiguration(
        placement=placement,
        block=block,
        chunk=chunk,
        local_profile=NETWORK_PROFILES["canonical_fast_local"],
        inter_pod_profile=NETWORK_PROFILES["canonical_inter_pod"],
    )
    base = build_worker_dag(service, configuration)
    baseline_run = DeterministicMicroworkerEngine().run(_prefixed_tasks(base, 0))
    baseline_latency = baseline_run.makespan_ms
    rows = []
    for count in requests:
        tasks = [task for request in range(count) for task in _prefixed_tasks(base, request)]
        run = DeterministicMicroworkerEngine().run(tasks)
        completions = []
        for request in range(count):
            request_id = f"concurrent-request-{request:03d}"
            completions.append(
                max(
                    record.finish_time
                    for record in run.records
                    if record.request_id == request_id
                )
            )
        accepted = block + 1
        per_user_rates = [accepted / (value / 1000.0) for value in completions]
        rows.append(
            {
                "concurrent_requests": count,
                "single_user_latency_ms_p50": statistics.median(completions),
                "per_user_tok_s_p50": statistics.median(per_user_rates),
                "aggregate_tok_s": count * accepted / (run.makespan_ms / 1000.0),
                "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                "queueing_ms_p50": statistics.median(
                    max(0.0, value - baseline_latency) for value in completions
                ),
                "active_worker_seconds_per_token": run.total_compute_work_ms
                / 1000.0
                / (accepted * count),
                "predicted": True,
            }
        )
    return rows


def straggler_sweep(
    service: MeasuredShardService,
    *,
    depth_span: int,
    block: int = 16,
    chunk: int = 1,
) -> list[dict[str, Any]]:
    placement = PlacementSpec(20, service.degree, depth_span, chunk)
    configuration = SimulationConfiguration(
        placement=placement,
        block=block,
        chunk=chunk,
        local_profile=NETWORK_PROFILES["canonical_fast_local"],
        inter_pod_profile=NETWORK_PROFILES["canonical_inter_pod"],
    )
    base_tasks = build_worker_dag(service, configuration)

    def run_with_slow_workers(slow: set[str], factor: float) -> Any:
        transformed = [
            dataclasses.replace(
                task,
                service_time_ms=task.service_time_ms * factor,
            )
            if isinstance(task, MicroworkerTask) and task.worker_id in slow
            else task
            for task in base_tasks
        ]
        return DeterministicMicroworkerEngine().run(transformed)

    workers = sorted(
        {task.worker_id for task in base_tasks if isinstance(task, MicroworkerTask)}
    )
    pods: dict[str, list[str]] = defaultdict(list)
    for worker in workers:
        pods[worker.split(".worker-")[0]].append(worker)
    ten_percent = set(workers[: math.ceil(len(workers) * 0.10)])
    one_per_pod = {values[0] for values in pods.values()}
    degraded_pod = set(pods[sorted(pods)[0]])
    baseline_run = DeterministicMicroworkerEngine().run(base_tasks)
    accepted = block + 1
    rows = []
    scenarios = (
        ("homogeneous", set(), 1.0, None),
        ("10_percent_workers_25_percent_slower", ten_percent, 1.25, None),
        ("10_percent_workers_50_percent_slower", ten_percent, 1.5, None),
        ("one_slow_worker_per_pod", one_per_pod, 1.5, None),
        ("network_jitter_20_percent", set(), 1.0, 0.2),
        ("one_degraded_pod", degraded_pod, 1.5, None),
    )
    for name, slow_workers, factor, jitter in scenarios:
        if jitter is not None:
            result, run = candidate_projection(
                service,
                depth_span=depth_span,
                block=block,
                chunk=chunk,
                jitter_fraction=jitter,
            )
            makespan = result["target_pass_ms"]
            rate = result["exact_tok_s_per_user"]
        else:
            run = run_with_slow_workers(slow_workers, factor)
            makespan = run.makespan_ms
            rate = accepted / (makespan / 1000.0)
        rows.append(
            {
                "scenario": name,
                "affected_worker_count": len(slow_workers),
                "affected_worker_fraction": len(slow_workers) / len(workers),
                "affected_compute_multiplier": factor,
                "network_jitter_fraction": jitter or 0.0,
                "predicted_tok_s_per_user": rate,
                "predicted_target_pass_ms": makespan,
                "critical_path_amplification": makespan / baseline_run.makespan_ms,
                "predicted": True,
            }
        )
    return rows


def topology_gate(
    service: MeasuredShardService,
    *,
    depth_span: int,
    link: str,
) -> dict[str, Any]:
    if link not in {"intra_pod", "inter_pod"}:
        raise ValueError("link must be intra_pod or inter_pod")
    accepted = []
    for rtt in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        for bandwidth in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0):
            local = NetworkProfile("gate_local", 0.25, 25.0, 0.05)
            inter = NetworkProfile("gate_inter", 5.0, 10.0, 0.10)
            if link == "intra_pod":
                local = replace(local, rtt_ms=rtt, bandwidth_gbps=bandwidth)
            else:
                inter = replace(inter, rtt_ms=rtt, bandwidth_gbps=bandwidth)
            result, _ = candidate_projection(
                service,
                depth_span=depth_span,
                block=16,
                chunk=1,
                compute_slowdown=1.2,
                local_profile=local,
                inter_pod_profile=inter,
                jitter_fraction=0.1,
            )
            if result["exact_tok_s_per_user"] >= 5.0:
                accepted.append((rtt, bandwidth, result["exact_tok_s_per_user"]))
    if not accepted:
        return {"link": link, "status": "NO_ACCEPTABLE_PROFILE"}
    max_rtt = max(value[0] for value in accepted)
    at_max = [value for value in accepted if value[0] == max_rtt]
    minimum_bandwidth = min(value[1] for value in at_max)
    rate = next(value[2] for value in at_max if value[1] == minimum_bandwidth)
    return {
        "link": link,
        "status": "PREREGISTERED",
        "maximum_rtt_ms": max_rtt,
        "minimum_bandwidth_gbps_at_maximum_rtt": minimum_bandwidth,
        "predicted_tok_s_at_gate": rate,
        "compute_degradation_percent": 20,
        "jitter_fraction": 0.1,
        "decision": "TOPOLOGY_ACCEPTED when every required measured path satisfies both thresholds; otherwise TOPOLOGY_REJECTED",
    }


def work_inflation_breakdown(
    run: Any,
    canonical_equivalent_compute_ms: float,
) -> list[dict[str, Any]]:
    categories: dict[str, float] = defaultdict(float)
    for record in run.records:
        if record.resource_type != "microworker":
            continue
        operator = record.operator.lower()
        if "kda" in operator:
            category = "KDA"
        elif "mla" in operator or "attention" in operator:
            category = "MLA"
        elif "expert_stripe" in operator:
            category = "experts"
        elif "shared" in operator:
            category = "shared_experts"
        elif "projection" in operator or "latent" in operator:
            category = "projections"
        elif "endpoint" in operator or "embedding" in operator or "lm_head" in operator:
            category = "endpoint"
        elif "reduce" in operator or "accumulation" in operator:
            category = "reduction"
        else:
            category = "other"
        categories[category] += record.duration_ms
    total = sum(categories.values())
    overall = total / canonical_equivalent_compute_ms
    return [
        {
            "category": category,
            "bounded_worker_compute_ms": value,
            "share_of_bounded_compute": value / total,
            "contribution_to_overall_inflation": value / canonical_equivalent_compute_ms,
            "overall_compute_work_inflation": overall,
            "category_specific_canonical_denominator_available": False,
        }
        for category, value in sorted(categories.items())
    ]


def validate_single_resource_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [
        abs(float(row["predicted_sharded_wall_ms"]) - float(row["actual_sharded_wall_ms"]))
        / float(row["actual_sharded_wall_ms"])
        for row in rows
    ]
    ordered = sorted(errors)
    median = statistics.median(errors)
    p90 = ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]
    maximum = max(errors)
    return {
        "median_error": median,
        "p90_error": p90,
        "maximum_error": maximum,
        "targets": {"median": 0.05, "p90": 0.10, "maximum": 0.15},
        "normalization_applied": False,
        "post_hoc_multiplier": None,
        "status": "PASS" if median <= 0.05 and p90 <= 0.10 and maximum <= 0.15 else "FAIL",
    }


__all__ = [
    "accounting_reconciliation",
    "candidate_projection",
    "concurrency_sweep",
    "measured_service",
    "slowdown_sensitivity",
    "straggler_sweep",
    "topology_gate",
    "uncertainty_envelope",
    "validate_single_resource_rows",
    "work_inflation_breakdown",
]
