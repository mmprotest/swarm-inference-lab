"""Compatibility and exactness gates for Experiment 023."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)
from swarm_inference.experiments.experiment_022.evaluator import (
    PlacementEvaluator,
    common_endpoint_policy,
)
from swarm_inference.experiments.experiment_022.io import write_csv
from swarm_inference.experiments.experiment_022.model_graph import build_model_graph
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .baseline import clone_placement, load_frozen_placement
from .freeze import validate_e023_freeze
from .models import E023Plan, NetworkMode
from .serving_engine import ServingEngine

COMPATIBILITY_REPRESENTATIVES = (
    ("coarse-friendly-01", "A"),
    ("memory-fragmented-03", "B"),
    ("compute-heterogeneous-04", "A"),
    ("network-heterogeneous-01", "D"),
    ("full-mixed-01", "E"),
)


def _float_equal(left: float, right: float, tolerance: float = 1e-9) -> bool:
    absolute = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = absolute / scale if scale else 0.0
    return absolute <= tolerance or relative <= tolerance


def run_engine_compatibility(repo: Path) -> list[dict[str, Any]]:
    """Require the interval engine's legacy/no-replica path to reproduce E022."""

    root = repo.resolve()
    validate_e023_freeze(root)
    inventories, _audit = load_frozen_inventories(root)
    inventory_map = {value.inventory_id: value for value in inventories}
    model = build_model_graph(
        Path("F:/models/Kimi-K3"),
        whole_layer_service_csv=root
        / "artifacts/experiment-018/physical/layer-service.csv",
    )
    service = ResidentServiceModel.from_csv(
        root
        / "artifacts/experiment-022/completion/validation/repaired-resident-service.csv"
    )
    rows: list[dict[str, Any]] = []
    placement_root = root / "artifacts/experiment-022/completion/rerun/placements"
    for inventory_id, planner_level in COMPATIBILITY_REPRESENTATIVES:
        inventory = inventory_map[inventory_id]
        endpoint = common_endpoint_policy(model, inventory)
        if endpoint is None:
            raise RuntimeError(f"MODEL_INVALID: endpoint infeasible for {inventory_id}")
        plan = load_frozen_placement(
            placement_root / f"{inventory_id}-{planner_level}.json", inventory
        )
        e022_plan = clone_placement(plan)
        old = PlacementEvaluator(model, inventory, service, endpoint).evaluate(e022_plan)
        new = ServingEngine(
            model,
            inventory,
            service,
            endpoint,
            network_mode=NetworkMode.LEGACY_DIRECTED_LINK,
        ).run_single_pass(E023Plan(inventory_id, "U_STRONG", clone_placement(plan)))
        network_equal = new.network_bytes == old.total_network_bytes
        message_equal = new.messages == old.messages
        event_equal = new.event_count == len(old.records)
        worker_equal = set(new.participating_workers) == set(old.worker_utilization)
        critical_equal = _float_equal(new.critical_path_ms, old.makespan_ms)
        compute_equal = _float_equal(
            new.total_worker_compute_ms, old.total_compute_ms
        )
        network_time_equal = _float_equal(
            new.total_network_ms, old.total_network_ms
        )
        passed = all(
            (
                network_equal,
                message_equal,
                event_equal,
                worker_equal,
                critical_equal,
                compute_equal,
                network_time_equal,
            )
        )
        rows.append(
            {
                "case_id": f"{inventory_id}-{planner_level}",
                "inventory_id": inventory_id,
                "planner_level": planner_level,
                "network_mode": NetworkMode.LEGACY_DIRECTED_LINK.value,
                "concurrency": 1,
                "target_passes": 1,
                "replicas": 0,
                "e022_critical_path_ms": old.makespan_ms,
                "e023_critical_path_ms": new.critical_path_ms,
                "critical_path_absolute_error_ms": abs(
                    new.critical_path_ms - old.makespan_ms
                ),
                "e022_total_worker_compute_ms": old.total_compute_ms,
                "e023_total_worker_compute_ms": new.total_worker_compute_ms,
                "worker_compute_absolute_error_ms": abs(
                    new.total_worker_compute_ms - old.total_compute_ms
                ),
                "e022_total_network_ms": old.total_network_ms,
                "e023_total_network_ms": new.total_network_ms,
                "network_time_absolute_error_ms": abs(
                    new.total_network_ms - old.total_network_ms
                ),
                "e022_network_bytes": old.total_network_bytes,
                "e023_network_bytes": new.network_bytes,
                "e022_messages": old.messages,
                "e023_messages": new.messages,
                "e022_event_count": len(old.records),
                "e023_event_count": new.event_count,
                "e022_participating_workers": len(old.worker_utilization),
                "e023_participating_workers": len(new.participating_workers),
                "network_bytes_equal": network_equal,
                "messages_equal": message_equal,
                "event_count_equal": event_equal,
                "participating_workers_equal": worker_equal,
                "timing_equal_within_1e_9": (
                    critical_equal and compute_equal and network_time_equal
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
    write_csv(
        root / "artifacts/experiment-023/validation/engine-compatibility.csv",
        rows,
    )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("MODEL_INVALID: E023 legacy engine compatibility failed")
    return rows


__all__ = ["COMPATIBILITY_REPRESENTATIVES", "run_engine_compatibility"]
