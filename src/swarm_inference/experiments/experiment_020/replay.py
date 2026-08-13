"""Corrected single-resource validation of the physically sharded algorithm."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from swarm_inference.experiments.experiment_019.events import (
    MicroworkerTask,
    NetworkProfile,
    NetworkTask,
)
from swarm_inference.experiments.experiment_019.placement import PlacementSpec
from swarm_inference.experiments.experiment_019.simulation import (
    SimulationConfiguration,
    build_worker_dag,
)

from .simulation import measured_service, validate_single_resource_rows


def _attention_split(receipt: Mapping[str, Any], split: str) -> dict[str, Any]:
    value = copy.deepcopy(receipt)
    selected = copy.deepcopy(receipt[split])
    value["calibration"] = copy.deepcopy(selected)
    value["heldout"] = copy.deepcopy(selected)
    return value


def _software_overhead(protocol: Mapping[str, Any]) -> float:
    return statistics.median(
        float(row["software_overhead_p50_ms"])
        for row in protocol["protocol"]["rows"]
    )


def _tasks(service: Any, protocol: Mapping[str, Any]) -> list[Any]:
    loopback = NetworkProfile(
        "single_resource_replay_loopback",
        rtt_ms=0.0,
        bandwidth_gbps=1_000_000.0,
        software_overhead_ms=_software_overhead(protocol),
    )
    configuration = SimulationConfiguration(
        placement=PlacementSpec(20, service.degree, 8, service.rows),
        block=0,
        chunk=1,
        local_profile=loopback,
        inter_pod_profile=loopback,
    )
    return build_worker_dag(service, configuration)


def _task_wall(task: Any) -> float:
    if isinstance(task, MicroworkerTask):
        return float(task.service_time_ms)
    if isinstance(task, NetworkTask):
        return float(task.service_time_ms)
    raise TypeError(type(task))


def _span_wall(tasks: Sequence[Any], layers: set[int]) -> tuple[float, int, int]:
    selected = [task for task in tasks if int(task.layer_id) in layers]
    return (
        sum(_task_wall(task) for task in selected),
        sum(isinstance(task, MicroworkerTask) for task in selected),
        sum(isinstance(task, NetworkTask) for task in selected),
    )


def _operator_wall(tasks: Sequence[Any], layer: int, operator: str) -> float:
    return _task_wall(
        next(task for task in tasks if task.layer_id == layer and task.operator == operator)
    )


def _raw_worker_wall(
    receipt: Mapping[str, Any], degree: int, rows: int, operator: str
) -> float:
    result = next(
        row
        for row in receipt["results"]
        if int(row["degree"]) == degree and int(row["rows"]) == rows
    )
    timing = result["operators"][operator]["workers"][0]["duration"]
    return float(timing["p50_ms"])


def run_single_resource_replay(
    *,
    attention_calibration: Mapping[str, Any],
    attention_heldout: Mapping[str, Any],
    legacy_expert: Mapping[str, Any],
    grouped_calibration: Mapping[str, Any],
    grouped_heldout: Mapping[str, Any],
    other_calibration: Mapping[str, Any],
    other_heldout: Mapping[str, Any],
    protocol_calibration: Mapping[str, Any],
    protocol_heldout: Mapping[str, Any],
) -> dict[str, Any]:
    predicted_service = measured_service(
        _attention_split(attention_calibration, "calibration"),
        legacy_expert,
        grouped_calibration,
        other_calibration,
        protocol_calibration,
        degree=8,
        rows=1,
    )
    actual_service = measured_service(
        _attention_split(attention_heldout, "heldout"),
        legacy_expert,
        grouped_heldout,
        other_heldout,
        protocol_heldout,
        degree=8,
        rows=1,
    )
    predicted_tasks = _tasks(predicted_service, protocol_calibration)
    actual_tasks = _tasks(actual_service, protocol_heldout)

    definitions: list[tuple[str, set[int]]] = [
        ("complete_sharded_KDA_layer", {1}),
        ("complete_sharded_MLA_layer", {3}),
        ("two_layer_span", set(range(1, 3))),
        ("four_layer_span", set(range(1, 5))),
        ("eight_layer_span", set(range(1, 9))),
    ]
    rows: list[dict[str, Any]] = []
    operators = (
        ("expert_stripe", 1, "expert_stripe_local_top16_accumulation"),
        ("KDA_stripe", 1, "KDA_head_projection_stripe"),
        ("MLA_stripe", 3, "Gated_MLA_head_projection_stripe"),
        ("projection_stripe", 1, "latent_down_row_projection_stripe"),
    )
    for workload, layer, operator in operators:
        predicted = _operator_wall(predicted_tasks, layer, operator)
        actual = _operator_wall(actual_tasks, layer, operator)
        rows.append(
            {
                "workload": workload,
                "predicted_sharded_wall_ms": predicted,
                "actual_sharded_wall_ms": actual,
                "absolute_percentage_error": abs(predicted - actual) / actual,
                "compute_task_count": 1,
                "network_task_count": 0,
                "physical_actual_source": "independent heldout RTX 5090 worker wall",
            }
        )
    predicted = _raw_worker_wall(other_calibration, 8, 1, "shared_expert")
    actual = _raw_worker_wall(other_heldout, 8, 1, "shared_expert")
    rows.append(
        {
            "workload": "shared_expert_stripe",
            "predicted_sharded_wall_ms": predicted,
            "actual_sharded_wall_ms": actual,
            "absolute_percentage_error": abs(predicted - actual) / actual,
            "compute_task_count": 1,
            "network_task_count": 0,
            "physical_actual_source": "independent heldout RTX 5090 worker wall",
        }
    )
    for workload, layers in definitions:
        predicted, compute_count, network_count = _span_wall(predicted_tasks, layers)
        actual, actual_compute_count, actual_network_count = _span_wall(
            actual_tasks, layers
        )
        if (compute_count, network_count) != (
            actual_compute_count,
            actual_network_count,
        ):
            raise RuntimeError("calibration/heldout task graph topology drifted")
        rows.append(
            {
                "workload": workload,
                "predicted_sharded_wall_ms": predicted,
                "actual_sharded_wall_ms": actual,
                "absolute_percentage_error": abs(predicted - actual) / actual,
                "compute_task_count": compute_count,
                "network_task_count": network_count,
                "physical_actual_source": (
                    "single-resource sum of independently heldout physical shard calls, "
                    "physical reductions, and heldout protocol framing; no compute overlap"
                ),
            }
        )
    validation = validate_single_resource_rows(rows)
    return {
        "schema_version": "experiment-020-single-resource-replay-v1",
        "mode": "single_resource_replay",
        "physical_compute_resources": 1,
        "compute_overlap": False,
        "same_worker_task_graph": True,
        "operation_order_preserved": True,
        "real_local_software_overhead": True,
        "physical_equivalent_workload_executed": False,
        "methodology_classification": "FAIL_METHOD_INVALID",
        "methodology_failure": (
            "complete layer/span actuals are sums of independently held-out primitive "
            "walls rather than one physically executed ordered shard workload"
        ),
        "global_normalization_factor": None,
        "post_hoc_multiplier": None,
        "rows": rows,
        "validation": validation,
        "numerical_service_stability_status": validation["status"],
        "status": "FAIL",
    }


__all__ = ["run_single_resource_replay"]
