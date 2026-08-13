"""Independent-machine diagnostic sweeps for Experiment 021.

The simulator is deliberately downstream of ordered physical validation.  It
will still emit diagnostic rows after a validation failure so the failed model
can be inspected, but those rows are explicitly inadmissible as performance
evidence and can never drive the E021 outcome.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_019.events import (
    CommunicationEdge,
    DeterministicMicroworkerEngine,
    MicroworkerTask,
    NetworkProfile,
    NetworkTask,
)
from swarm_inference.experiments.experiment_019.finalize import _physical_service
from swarm_inference.experiments.experiment_019.placement import PlacementResult, PlacementSpec
from swarm_inference.experiments.experiment_019.simulation import (
    MeasuredShardService,
    SimulationConfiguration,
    build_worker_dag,
    simulate,
)
from swarm_inference.experiments.experiment_020.simulation import (
    accounting_reconciliation,
    measured_service,
)

from .io import atomic_write_json, write_csv

REGIMES: dict[str, tuple[str, float, float]] = {
    "A": ("very_fast_independent_hosts", 0.25, 25.0),
    "B": ("fast_lan", 1.0, 10.0),
    "C": ("regional_independent_machines", 5.0, 1.0),
    "D": ("consumer_wan_like", 20.0, 0.1),
    "E": ("wider_wan_sensitivity", 50.0, 0.1),
}
BLOCKS = (7, 12, 16)
CHUNKS = (1, 2, 4)
SOFTWARE_OVERHEAD_MS = 0.04245
EVIDENCE_INVALID = (
    "PHYSICAL_SHARD_EXECUTION inputs + SHAPED_NETWORK + "
    "UNVALIDATED_INDEPENDENT_MACHINE_MODEL_DIAGNOSTIC_ONLY"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs(repo: Path) -> dict[str, dict[str, Any]]:
    e019 = repo / "artifacts" / "experiment-019"
    e020 = repo / "artifacts" / "experiment-020"
    return {
        "attention": _read(e020 / "physical" / "attention-raw.json"),
        "expert": _read(e019 / "physical" / "expert-stripe-raw.json"),
        "grouped": _read(e020 / "physical" / "expert-grouped-robust-calibration.json"),
        "other": _read(e019 / "physical" / "other-shards-raw.json"),
        "protocol": _read(e020 / "runtime" / "protocol-heldout-raw.json"),
    }


def measured_worker_service(
    repo: Path,
    *,
    degree: int,
    rows: int,
) -> MeasuredShardService:
    values = _inputs(repo)
    if degree == 8:
        return measured_service(
            values["attention"],
            values["expert"],
            values["grouped"],
            values["other"],
            values["protocol"],
            degree=degree,
            rows=rows,
        )
    service = _physical_service(
        values["attention"],
        values["expert"],
        values["other"],
        values["protocol"],
        degree=degree,
        rows=rows,
    )
    return dataclasses.replace(
        service,
        source=(
            "RTX 5090 exact legacy expert-stripe physical worker services; "
            "grouped top-16 bank execution was not physically characterized "
            f"at P={degree}"
        ),
    )


def independent_profile(regime: str, *, rtt_ms: float | None = None, bandwidth_gbps: float | None = None) -> NetworkProfile:
    name, default_rtt, default_bandwidth = REGIMES[regime]
    return NetworkProfile(
        name=f"independent_machine_{name}",
        rtt_ms=default_rtt if rtt_ms is None else rtt_ms,
        bandwidth_gbps=default_bandwidth if bandwidth_gbps is None else bandwidth_gbps,
        software_overhead_ms=SOFTWARE_OVERHEAD_MS,
    )


def _equivalent_unsharded_work_ms(repo: Path, *, block: int, chunk: int) -> float:
    historical = _read(repo / "artifacts" / "experiment-018" / "physical" / "service-raw.json")
    remaining = block + 1
    equivalent = 0.0
    while remaining:
        actual = min(chunk, remaining)
        kda = float(historical["layers"]["89"]["service"][str(actual)]["wall"]["p50_ms"])
        mla = float(historical["layers"]["91"]["service"][str(actual)]["wall"]["p50_ms"])
        equivalent += 69 * kda + 24 * mla
        remaining -= actual
    return equivalent


def _wait_accounting(run: Any) -> tuple[float, float]:
    by_id = {row.task_id: row for row in run.records}
    queue_wait = 0.0
    state_wait = 0.0
    for row in run.records:
        dependency_finish = max(
            (by_id[value].finish_time for value in row.dependency_ids),
            default=0.0,
        )
        wait = max(0.0, row.start_time - dependency_finish)
        queue_wait += wait
        if row.state_refs:
            state_wait += wait
    return queue_wait, state_wait


def _decorate(
    repo: Path,
    result: dict[str, Any],
    run: Any,
    placement: PlacementResult,
    *,
    regime: str,
    validated: bool,
    grouped_experts: bool,
) -> dict[str, Any]:
    accepted = int(result["block"]) + 1
    queue_wait, state_wait = _wait_accounting(run)
    denominator = _equivalent_unsharded_work_ms(
        repo,
        block=int(result["block"]),
        chunk=int(result["chunk"]),
    )
    network_records = [row for row in run.records if row.resource_type == "network"]
    compute_records = [row for row in run.records if row.resource_type == "microworker"]
    all_edges_explicit = all(row.communication_edges for row in network_records)
    all_compute_owned = all(bool(row.worker_id) for row in compute_records)
    machine_count = len(placement.workers)
    return {
        **result,
        "schema_version": "experiment-021-independent-machine-sweep-v1",
        "regime": regime,
        "rtt_ms": independent_profile(regime).rtt_ms if regime in REGIMES else None,
        "bandwidth_gbps": (
            independent_profile(regime).bandwidth_gbps if regime in REGIMES else None
        ),
        "protocol_software_overhead_ms_per_network_event_step": SOFTWARE_OVERHEAD_MS,
        "worker_count": machine_count,
        "machine_count": machine_count,
        "one_compute_worker_per_machine": True,
        "logical_depth_group_count": placement.spec.pod_count,
        "logical_depth_groups_have_compute": False,
        "logical_depth_groups_have_memory": False,
        "same_host_pcie_nvlink_nccl_assumed": False,
        "all_worker_links_use_same_independent_machine_profile": True,
        "max_peak_memory_per_worker_bytes": placement.max_worker_peak_bytes,
        "max_peak_memory_per_worker_gib": placement.max_worker_peak_bytes / 1024**3,
        "total_resident_bytes": placement.total_resident_weight_bytes,
        "active_worker_seconds_per_accepted_token": result[
            "worker_seconds_per_accepted_token"
        ],
        "resident_worker_seconds_per_accepted_token": (
            machine_count * float(result["target_pass_ms"]) / 1000.0 / accepted
        ),
        "network_messages_per_accepted_token": len(network_records) / accepted,
        "network_event_count": len(network_records),
        "compute_event_count": len(compute_records),
        "queue_wait_ms_sum": queue_wait,
        "state_wait_ms_sum": state_wait,
        "serial_waits_per_accepted_token_ms": queue_wait / accepted,
        "reduction_collective_count": sum(
            row.collective_algorithm is not None for row in network_records
        ),
        "reduction_collective_steps": sum(row.collective_steps for row in network_records),
        "compute_work_inflation": float(result["total_physical_compute_work_ms"])
        / denominator,
        "equivalent_unsharded_compute_work_ms": denominator,
        "compute_work_denominator_source": (
            "E018 physical layer-89/91 controls; post-schedule diagnostic only"
        ),
        "compute_share_of_critical_path": float(result["critical_path_compute_ms"])
        / max(float(result["target_pass_ms"]), 1e-30),
        "network_share_of_critical_path": float(result["network_critical_path_ms"])
        / max(float(result["target_pass_ms"]), 1e-30),
        "all_compute_events_have_worker_id": all_compute_owned,
        "all_network_events_have_explicit_edges": all_edges_explicit,
        "grouped_top16_expert_service": grouped_experts,
        "model_validation_status": "PASS" if validated else "FAIL",
        "admissible": validated,
        "admissibility_reason": (
            "ordered physical replay passed every preregistered gate"
            if validated
            else "MODEL_INVALID: ordered physical replay did not validate the resident-worker event model"
        ),
        "evidence_class": (
            "VALIDATED_INDEPENDENT_MACHINE_MODEL + SHAPED_NETWORK"
            if validated
            else EVIDENCE_INVALID
        ),
    }


def _configuration(
    placement: PlacementResult,
    *,
    block: int,
    chunk: int,
    profile: NetworkProfile,
    heterogeneity: str = "homogeneous",
    jitter_fraction: float = 0.0,
) -> SimulationConfiguration:
    spec = PlacementSpec(
        placement.spec.memory_cap_gib,
        placement.spec.stripe_degree,
        placement.spec.depth_span,
        chunk,
        hardware_class=placement.spec.hardware_class,
        expert_allocation_overhead_factor=placement.spec.expert_allocation_overhead_factor,
    )
    return SimulationConfiguration(
        placement=spec,
        block=block,
        chunk=chunk,
        local_profile=profile,
        inter_pod_profile=profile,
        heterogeneity=heterogeneity,
        jitter_fraction=jitter_fraction,
    )


def _best(rows: Sequence[Mapping[str, Any]], *, cap: int, regime: str) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if int(row["memory_cap_gib"]) == cap and row["regime"] == regime
    ]
    return dict(
        max(
            candidates,
            key=lambda row: (
                float(row["exact_tok_s_per_user"]),
                -int(row["worker_count"]),
            ),
        )
    )


def run_main_sweep(
    repo: Path,
    placements: Mapping[int, PlacementResult],
    artifact_root: Path,
    *,
    validated: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _cap, placement in sorted(placements.items(), reverse=True):
        for chunk in CHUNKS:
            service = measured_worker_service(
                repo,
                degree=placement.spec.stripe_degree,
                rows=chunk,
            )
            for regime in REGIMES:
                profile = independent_profile(regime)
                for block in BLOCKS:
                    configuration = _configuration(
                        placement,
                        block=block,
                        chunk=chunk,
                        profile=profile,
                    )
                    result, run = simulate(service, configuration)
                    rows.append(
                        _decorate(
                            repo,
                            result,
                            run,
                            placement,
                            regime=regime,
                            validated=validated,
                            grouped_experts=placement.spec.stripe_degree == 8,
                        )
                    )
    write_csv(artifact_root / "simulation" / "sweep.csv", rows)
    best = {
        (cap, regime): _best(rows, cap=cap, regime=regime)
        for cap in placements
        for regime in REGIMES
    }
    curve = []
    utilization = []
    for (cap, regime), row in sorted(best.items(), key=lambda item: (-item[0][0], item[0][1])):
        curve.append(
            {
                "worker_memory_cap_gib": cap,
                "regime": regime,
                "rtt_ms": row["rtt_ms"],
                "bandwidth_gbps": row["bandwidth_gbps"],
                "diagnostic_tok_s_per_user": row["exact_tok_s_per_user"],
                "exact_tok_s_per_user": (
                    row["exact_tok_s_per_user"] if validated else ""
                ),
                "target_pass_ms": row["target_pass_ms"],
                "worker_count": row["worker_count"],
                "network_bytes_per_accepted_token": row[
                    "network_bytes_per_accepted_token"
                ],
                "active_worker_seconds_per_accepted_token": row[
                    "active_worker_seconds_per_accepted_token"
                ],
                "average_worker_utilization": row["average_worker_utilization"],
                "block": row["block"],
                "chunk": row["chunk"],
                "admissible": validated,
                "evidence_class": row["evidence_class"],
            }
        )
        utilization.append(
            {
                "worker_memory_cap_gib": cap,
                "regime": regime,
                "worker_count": row["worker_count"],
                "average_worker_utilization": row["average_worker_utilization"],
                "p50_worker_utilization": row["p50_worker_utilization"],
                "p95_worker_utilization": row["p95_worker_utilization"],
                "peak_simultaneous_workers": row["peak_simultaneous_workers"],
                "active_worker_seconds_per_accepted_token": row[
                    "active_worker_seconds_per_accepted_token"
                ],
                "resident_worker_seconds_per_accepted_token": row[
                    "resident_worker_seconds_per_accepted_token"
                ],
                "admissible": validated,
            }
        )
    write_csv(artifact_root / "simulation" / "memory-network-curve.csv", curve)
    write_csv(artifact_root / "simulation" / "worker-utilization.csv", utilization)

    selected = best[(8, "B")]
    selected_placement = placements[8]
    selected_service = measured_worker_service(
        repo,
        degree=selected_placement.spec.stripe_degree,
        rows=int(selected["chunk"]),
    )
    _selected_result, selected_run = simulate(
        selected_service,
        _configuration(
            selected_placement,
            block=int(selected["block"]),
            chunk=int(selected["chunk"]),
            profile=independent_profile("B"),
        ),
    )
    by_id = {row.task_id: row for row in selected_run.records}
    critical_records = [by_id[value] for value in selected_run.critical_path_task_ids]
    chunk_completions: dict[int, float] = {}
    chunk_starts: dict[int, float] = {}
    for row in selected_run.records:
        chunk_completions[row.chunk_id] = max(
            chunk_completions.get(row.chunk_id, 0.0), row.finish_time
        )
        chunk_starts[row.chunk_id] = min(
            chunk_starts.get(row.chunk_id, math.inf), row.start_time
        )
    completion_values = [value for _, value in sorted(chunk_completions.items())]
    periods = [right - left for left, right in itertools.pairwise(completion_values)]
    critical = {
        "schema_version": "experiment-021-critical-path-v1",
        "status": "PASS" if validated else "DIAGNOSTIC_MODEL_INVALID",
        "configuration": {
            "memory_cap_gib": 8,
            "regime": "B",
            "block": selected["block"],
            "chunk": selected["chunk"],
            "worker_count": selected["worker_count"],
        },
        "fill_ms": completion_values[0],
        "steady_state_period_ms_median": statistics.median(periods) if periods else 0.0,
        "drain_ms": selected_run.makespan_ms
        - max(chunk_starts.values(), default=selected_run.makespan_ms),
        "makespan_ms": selected_run.makespan_ms,
        "critical_path_compute_ms": selected_run.critical_path_compute_ms,
        "critical_path_network_ms": selected_run.network_critical_path_ms,
        "critical_path_task_count": len(critical_records),
        "critical_path": [
            {
                "task_id": row.task_id,
                "resource_type": row.resource_type,
                "worker_id": row.worker_id,
                "operator": row.operator,
                "layer": row.layer_id,
                "chunk": row.chunk_id,
                "start_ms": row.start_time,
                "finish_ms": row.finish_time,
                "duration_ms": row.duration_ms,
                "payload_bytes": row.payload_bytes,
                "network_profile": row.network_profile,
            }
            for row in critical_records
        ],
        "admissible": validated,
    }
    atomic_write_json(artifact_root / "simulation" / "critical-path.json", critical)
    return rows, best, critical


def run_network_envelope(
    repo: Path,
    placement: PlacementResult,
    best_row: Mapping[str, Any],
    artifact_root: Path,
    *,
    validated: bool,
) -> list[dict[str, Any]]:
    rtts = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    bandwidths = (0.1, 0.25, 0.5, 1.0, 2.5, 10.0, 25.0)
    block = int(best_row["block"])
    chunk = int(best_row["chunk"])
    service = measured_worker_service(
        repo,
        degree=placement.spec.stripe_degree,
        rows=chunk,
    )
    rows: list[dict[str, Any]] = []
    for rtt in rtts:
        for bandwidth in bandwidths:
            profile = NetworkProfile(
                name=f"independent_machine_envelope_{rtt:g}ms_{bandwidth:g}gbps",
                rtt_ms=rtt,
                bandwidth_gbps=bandwidth,
                software_overhead_ms=SOFTWARE_OVERHEAD_MS,
            )
            configuration = _configuration(
                placement,
                block=block,
                chunk=chunk,
                profile=profile,
            )
            result, run = simulate(service, configuration)
            row = _decorate(
                repo,
                result,
                run,
                placement,
                regime="CUSTOM_ENVELOPE",
                validated=validated,
                grouped_experts=True,
            )
            row["rtt_ms"] = rtt
            row["bandwidth_gbps"] = bandwidth
            row["diagnostic_passes_5_tok_s"] = (
                float(row["exact_tok_s_per_user"]) >= 5.0
            )
            row["passes_5_tok_s"] = (
                row["diagnostic_passes_5_tok_s"] if validated else False
            )
            rows.append(row)
    write_csv(artifact_root / "simulation" / "network-envelope.csv", rows)
    return rows


def run_accounting(
    repo: Path,
    placements: Mapping[int, PlacementResult],
    best: Mapping[tuple[int, str], Mapping[str, Any]],
    artifact_root: Path,
) -> dict[str, Any]:
    receipts = []
    for cap in sorted(placements, reverse=True):
        placement = placements[cap]
        selected = best[(cap, "B")]
        chunk = int(selected["chunk"])
        service = measured_worker_service(
            repo,
            degree=placement.spec.stripe_degree,
            rows=chunk,
        )
        receipt = accounting_reconciliation(
            service,
            _configuration(
                placement,
                block=int(selected["block"]),
                chunk=chunk,
                profile=independent_profile("B"),
            ),
        )
        receipt["worker_memory_cap_gib"] = cap
        receipt["one_worker_per_independent_machine"] = True
        receipt["all_links_independent_machine_profile"] = True
        receipts.append(receipt)
    result = {
        "schema_version": "experiment-021-accounting-reconciliation-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in receipts) else "FAIL",
        "no_cost_disappears_under_parallelism": all(
            row["compute_invariant_pass"] and row["network_invariant_pass"]
            for row in receipts
        ),
        "scheduler_overhead_policy": (
            "measured protocol software overhead is charged on every network step; "
            "worker service includes native launch and local scheduler overhead"
        ),
        "receipts": receipts,
    }
    atomic_write_json(artifact_root / "validation" / "accounting-reconciliation.json", result)
    atomic_write_json(artifact_root / "simulation" / "accounting-reconciliation.json", result)
    return result


def _prefix_tasks(tasks: Sequence[Any], request: int) -> list[Any]:
    prefix = f"request-{request:03d}."
    return [
        dataclasses.replace(
            task,
            task_id=prefix + task.task_id,
            dependency_ids=tuple(prefix + value for value in task.dependency_ids),
            request_id=f"request-{request:03d}",
        )
        for task in tasks
    ]


def run_concurrency(
    repo: Path,
    placement: PlacementResult,
    selected: Mapping[str, Any],
    artifact_root: Path,
    *,
    validated: bool,
) -> list[dict[str, Any]]:
    block = int(selected["block"])
    chunk = int(selected["chunk"])
    service = measured_worker_service(repo, degree=placement.spec.stripe_degree, rows=chunk)
    configuration = _configuration(
        placement,
        block=block,
        chunk=chunk,
        profile=independent_profile("B"),
    )
    base = build_worker_dag(service, configuration)
    baseline = DeterministicMicroworkerEngine().run(_prefix_tasks(base, 0))
    accepted = block + 1
    rows = []
    for count in (1, 2, 4, 8, 16):
        tasks = [task for request in range(count) for task in _prefix_tasks(base, request)]
        run = DeterministicMicroworkerEngine().run(tasks)
        completions = [
            max(
                row.finish_time
                for row in run.records
                if row.request_id == f"request-{request:03d}"
            )
            for request in range(count)
        ]
        rates = [accepted / (value / 1000.0) for value in completions]
        rows.append(
            {
                "concurrent_requests": count,
                "single_user_reference_tok_s": accepted / (baseline.makespan_ms / 1000.0),
                "per_user_tok_s_p50_diagnostic": statistics.median(rates),
                "per_user_tok_s_p50": statistics.median(rates) if validated else "",
                "aggregate_tok_s_diagnostic": count * accepted / (run.makespan_ms / 1000.0),
                "aggregate_tok_s": (
                    count * accepted / (run.makespan_ms / 1000.0) if validated else ""
                ),
                "makespan_ms": run.makespan_ms,
                "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                "queueing_ms_p50": statistics.median(
                    max(0.0, value - baseline.makespan_ms) for value in completions
                ),
                "active_worker_seconds_per_token": run.total_compute_work_ms
                / 1000.0
                / (accepted * count),
                "worker_count": len(placement.workers),
                "estimated_fleet_occupied_fraction": statistics.fmean(
                    run.worker_utilization.values()
                ),
                "economics_status": "MARKET_ESTIMATE_ONLY_NOT_ACTUAL_COST_PER_TOKEN",
                "admissible": validated,
                "evidence_class": (
                    "VALIDATED_INDEPENDENT_MACHINE_MODEL + SHAPED_NETWORK"
                    if validated
                    else EVIDENCE_INVALID
                ),
            }
        )
    write_csv(artifact_root / "simulation" / "concurrency.csv", rows)
    return rows


def run_heterogeneity(
    repo: Path,
    placement: PlacementResult,
    selected: Mapping[str, Any],
    artifact_root: Path,
    *,
    validated: bool,
) -> list[dict[str, Any]]:
    block = int(selected["block"])
    chunk = int(selected["chunk"])
    service = measured_worker_service(repo, degree=placement.spec.stripe_degree, rows=chunk)
    profile = independent_profile("B")
    scenarios = (
        ("homogeneous", "homogeneous", 0.0),
        ("random_plus_minus_20_percent_compute", "random_plus_minus_20", 0.0),
        ("ten_percent_workers_1p5x_slower", "ten_percent_1p5x", 0.0),
        ("ten_percent_workers_2x_slower", "ten_percent_2x", 0.0),
        ("network_jitter_plus_minus_20_percent", "homogeneous", 0.2),
    )
    raw: list[tuple[str, Any, dict[str, Any]]] = []
    for name, heterogeneity, jitter in scenarios:
        result, run = simulate(
            service,
            _configuration(
                placement,
                block=block,
                chunk=chunk,
                profile=profile,
                heterogeneity=heterogeneity,
                jitter_fraction=jitter,
            ),
        )
        raw.append((name, run, result))
    baseline_run = raw[0][1]
    baseline_tasks = build_worker_dag(
        service,
        _configuration(
            placement,
            block=block,
            chunk=chunk,
            profile=profile,
        ),
    )
    record_by_id = {row.task_id: row for row in baseline_run.records}
    slow_worker = next(
        record_by_id[value].worker_id
        for value in baseline_run.critical_path_task_ids
        if record_by_id[value].resource_type == "microworker"
    )
    transformed = [
        dataclasses.replace(task, service_time_ms=task.service_time_ms * 2.0)
        if isinstance(task, MicroworkerTask) and task.worker_id == slow_worker
        else task
        for task in baseline_tasks
    ]
    slow_run = DeterministicMicroworkerEngine().run(transformed)
    raw.append(
        (
            "single_2x_slow_worker_on_reduction_critical_path",
            slow_run,
            {
                "exact_tok_s_per_user": (block + 1) / (slow_run.makespan_ms / 1000.0),
                "target_pass_ms": slow_run.makespan_ms,
            },
        )
    )
    rows = []
    for name, run, result in raw:
        rows.append(
            {
                "scenario": name,
                "slow_critical_worker_id": slow_worker if name.startswith("single_") else "",
                "diagnostic_tok_s_per_user": result["exact_tok_s_per_user"],
                "exact_tok_s_per_user": result["exact_tok_s_per_user"] if validated else "",
                "target_pass_ms": result["target_pass_ms"],
                "straggler_amplification": run.makespan_ms / baseline_run.makespan_ms,
                "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                "admissible": validated,
                "evidence_class": (
                    "VALIDATED_INDEPENDENT_MACHINE_MODEL + SHAPED_NETWORK"
                    if validated
                    else EVIDENCE_INVALID
                ),
            }
        )
    write_csv(artifact_root / "simulation" / "heterogeneity.csv", rows)
    return rows


def run_whole_layer_control(repo: Path, artifact_root: Path) -> list[dict[str, Any]]:
    """Independent-machine whole-layer diagnostic at a relaxed 20 GiB tier."""

    historical = _read(repo / "artifacts" / "experiment-018" / "physical" / "service-raw.json")
    rows = []
    profile = independent_profile("B")
    for block in BLOCKS:
        for chunk in CHUNKS:
            tasks: list[Any] = []
            remaining = block + 1
            chunk_index = 0
            while remaining:
                actual = min(chunk, remaining)
                previous = None
                for layer in range(93):
                    worker = f"whole-layer-control-machine-{layer:03d}.worker"
                    attention = "89" if layer == 0 or layer % 4 != 3 else "91"
                    service_ms = float(
                        historical["layers"][attention]["service"][str(actual)]["wall"]["p50_ms"]
                    )
                    dependencies = []
                    if previous is not None:
                        dependencies.append(previous)
                    prior_state = (
                        f"whole.c{chunk_index - 1:03d}.l{layer:03d}.compute"
                        if chunk_index > 0
                        else None
                    )
                    if prior_state is not None:
                        dependencies.append(prior_state)
                    compute_id = f"whole.c{chunk_index:03d}.l{layer:03d}.compute"
                    tasks.append(
                        MicroworkerTask(
                            task_id=compute_id,
                            worker_id=worker,
                            pod_id=f"depth-label-{layer:03d}",
                            request_id="whole-layer-control",
                            block_id="verification",
                            chunk_id=chunk_index,
                            layer_id=layer,
                            operator="complete_transformer_layer_control",
                            shard_id="whole-layer",
                            dependency_ids=tuple(dependencies),
                            input_refs=(f"hidden.{chunk_index}.{layer}",),
                            state_refs=(f"state.{worker}",),
                            service_time_ms=service_ms,
                        )
                    )
                    if layer < 92:
                        network_id = f"whole.c{chunk_index:03d}.l{layer:03d}.activation"
                        tasks.append(
                            NetworkTask(
                                task_id=network_id,
                                request_id="whole-layer-control",
                                block_id="verification",
                                chunk_id=chunk_index,
                                layer_id=layer,
                                operator="whole_layer_activation_transfer",
                                dependency_ids=(compute_id,),
                                edges=(
                                    CommunicationEdge(
                                        worker,
                                        f"whole-layer-control-machine-{layer + 1:03d}.worker",
                                        actual * 7168 * 4,
                                    ),
                                ),
                                profile=profile,
                            )
                        )
                        previous = network_id
                    else:
                        previous = compute_id
                remaining -= actual
                chunk_index += 1
            run = DeterministicMicroworkerEngine().run(tasks)
            rows.append(
                {
                    "memory_cap_gib": 20,
                    "worker_count": 93,
                    "machine_count": 93,
                    "one_worker_per_machine": True,
                    "whole_layers_owned": True,
                    "complete_model_whole_layer_placement_possible": True,
                    "block": block,
                    "chunk": chunk,
                    "regime": "B",
                    "rtt_ms": profile.rtt_ms,
                    "bandwidth_gbps": profile.bandwidth_gbps,
                    "target_pass_ms": run.makespan_ms,
                    "exact_tok_s_per_user": (block + 1) / (run.makespan_ms / 1000.0),
                    "network_bytes_per_accepted_token": run.total_network_bytes / (block + 1),
                    "active_worker_seconds_per_accepted_token": run.total_compute_work_ms
                    / 1000.0
                    / (block + 1),
                    "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                    "evidence_class": "PHYSICAL_SINGLE_MACHINE service + SHAPED_NETWORK whole-layer control",
                    "headline_swarm_result": False,
                }
            )
    write_csv(artifact_root / "simulation" / "whole-layer-control.csv", rows)
    return rows


__all__ = [
    "BLOCKS",
    "CHUNKS",
    "REGIMES",
    "independent_profile",
    "measured_worker_service",
    "run_accounting",
    "run_concurrency",
    "run_heterogeneity",
    "run_main_sweep",
    "run_network_envelope",
    "run_whole_layer_control",
]
