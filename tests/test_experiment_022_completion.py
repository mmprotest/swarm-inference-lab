from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime
from swarm_inference.experiments.experiment_022.benchmark import headline_statistics
from swarm_inference.experiments.experiment_022.completion_finalize import _verdict
from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)
from swarm_inference.experiments.experiment_022.completion_physical import (
    _ledger_for_sample,
)
from swarm_inference.experiments.experiment_022.completion_rerun import (
    _placement_signature,
)
from swarm_inference.experiments.experiment_022.manifest_correctness import (
    _manifest_assignments,
)
from swarm_inference.experiments.experiment_022.models import PartitionKind
from swarm_inference.experiments.experiment_022.planner import SharedPlacementOptimizer

REPO = Path(__file__).resolve().parents[1]


def test_nested_cuda_runtime_close_does_not_shutdown_outer_worker() -> None:
    class Library:
        def __init__(self) -> None:
            self.freed: list[int] = []
            self.shutdowns = 0

        def coli_cuda_tensor_free(self, handle: int) -> None:
            self.freed.append(handle)

        def coli_cuda_shutdown(self) -> None:
            self.shutdowns += 1

    library = Library()
    runtime = object.__new__(_CudaRuntime)
    runtime._library = library
    runtime._tensors = [1, 2, 3]

    runtime.close(shutdown=False)

    assert library.freed == [3, 2, 1]
    assert library.shutdowns == 0
    assert runtime._tensors == []
    runtime.close()
    assert library.shutdowns == 1


def test_completion_loader_preserves_the_exact_frozen_27() -> None:
    inventories, audit = load_frozen_inventories(REPO)

    assert audit["status"] == "PASS"
    assert len(inventories) == audit["inventory_count"] == 27
    assert len({inventory.inventory_id for inventory in inventories}) == 27
    assert audit["suite_canonical_sha256"] == (
        "3e949a8eee0a71e128493f64e0be903bd373d3baad4be86a8869d87279d4bb49"
    )
    assert audit["regenerated"] is False
    assert len(audit["immutable_original_diagnostics"]) == 11
    assert len(audit["immutable_selected_manifests"]) == 5


def test_final_correctness_signature_includes_endpoint_without_treating_it_as_layer() -> None:
    manifest = (
        REPO
        / "artifacts"
        / "experiment-022"
        / "planner"
        / "placements"
        / "coarse-friendly-01-A.json"
    )

    assert len(_placement_signature(manifest)) == 64


def test_final_mixed_manifest_has_an_exact_supported_assignment_map() -> None:
    path = (
        REPO
        / "artifacts"
        / "experiment-022"
        / "completion"
        / "rerun"
        / "placements"
        / "memory-fragmented-02-E.json"
    )
    assignments = _manifest_assignments(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert set(assignments) == set(range(93))
    assert {row["partition_type"] for row in assignments.values()} == {
        "WHOLE_LAYER",
        "WHOLE_EXPERT",
        "EXPERT_SHARD",
        "FULL_MIXED_STRIPE",
    }
    assert all(
        len(row["pieces"])
        == (1 if row["partition_type"] == "WHOLE_LAYER" else row["degree"])
        for row in assignments.values()
    )


def _receipt(*, outer_ms: float, phase_wall: dict[str, float]) -> dict[str, object]:
    return {
        "layer": 45,
        "attention_type": "KDA",
        "rows": 4,
        "wall_samples_ms": [outer_ms],
        "operation_records": [
            [
                {
                    "phase": "router",
                    "operator": "router",
                    "duration_ms": 2.0,
                    "cuda_ms": 1.0,
                },
                {
                    "phase": "attention_workers_and_collective",
                    "operator": "KDA_attention_stripe",
                    "stripe_index": 0,
                    "duration_ms": 3.0,
                    "cuda_ms": 2.0,
                },
                {
                    "phase": "attention_workers_and_collective",
                    "operator": "KDA_attention_stripe",
                    "stripe_index": 1,
                    "duration_ms": 3.0,
                    "cuda_ms": 2.0,
                },
                {
                    "phase": "routed_shared_collective",
                    "operator": "routed_shared_native_reduction",
                    "duration_ms": 1.0,
                    "cuda_ms": 0.5,
                },
            ]
        ],
        "instrumentation": [
            {
                "phase_wall_ms": phase_wall,
                "h2d_wall_ms": 0.0,
                "d2h_wall_ms": 0.0,
                "launch_submit_ms": 0.0,
                "cuda_sync_wall_ms": 0.0,
            }
        ],
    }


def test_residual_ledger_has_one_owner_and_reconciles_outer_wall() -> None:
    receipt = _receipt(
        outer_ms=10.0,
        phase_wall={
            "router": 2.2,
            "attention_workers_and_collective": 6.5,
            "routed_shared_collective": 1.3,
        },
    )

    row, raw = _ledger_for_sample(receipt, 0)

    assert row["status"] == "PASS"
    assert row["unexplained_ms"] == pytest.approx(0.0)
    assert row["accounted_ms"] == pytest.approx(row["total_outer_wall_ms"])
    assert row["reconciliation_fraction"] == pytest.approx(0.0)
    assert row["single_gpu_serialization_artifact_ms"] == pytest.approx(3.0)
    assert row["worker_local_host_ms"] == pytest.approx(2.5)
    assert row["experimental_harness_ms"] == pytest.approx(0.5)
    assert raw["parallel_python_harness_ms"] == pytest.approx(0.5)
    assert raw["inconsistent_timer_boundaries"] is False


def test_residual_ledger_rejects_inconsistent_nested_timer_boundaries() -> None:
    receipt = _receipt(
        outer_ms=8.0,
        phase_wall={
            "router": 1.0,
            "attention_workers_and_collective": 5.7,
            "routed_shared_collective": 1.3,
        },
    )

    row, raw = _ledger_for_sample(receipt, 0)

    assert row["status"] == "FAIL"
    assert row["unexplained_fraction"] == pytest.approx(1.0)
    assert row["accounted_ms"] == pytest.approx(row["total_outer_wall_ms"])
    assert raw["inconsistent_timer_boundaries"] is True


def test_residual_ledger_assigns_validation_capture_only_to_experiment() -> None:
    receipt = _receipt(
        outer_ms=11.0,
        phase_wall={
            "router": 2.2,
            "attention_workers_and_collective": 7.5,
            "routed_shared_collective": 1.3,
        },
    )
    receipt["operation_records"][0].append(  # type: ignore[index]
        {
            "phase": "attention_workers_and_collective",
            "operator": "state_validation_capture",
            "duration_ms": 1.0,
            "cuda_ms": 0.0,
            "device_copy_ms": 0.4,
            "cost_classification": "EXPERIMENT_ONLY",
        }
    )
    receipt["instrumentation"][0]["d2h_wall_ms"] = 0.4  # type: ignore[index]

    row, raw = _ledger_for_sample(receipt, 0)

    assert row["status"] == "PASS"
    assert row["experimental_harness_ms"] == pytest.approx(1.5)
    assert row["device_copy_ms"] == pytest.approx(0.0)
    assert row["worker_local_host_ms"] == pytest.approx(2.5)
    assert row["accounted_ms"] == pytest.approx(row["total_outer_wall_ms"])
    assert raw["experiment_operation_wall_ms"] == pytest.approx(1.0)
    assert raw["experiment_device_copy_wall_ms"] == pytest.approx(0.4)
    assert raw["parallel_python_harness_ms"] == pytest.approx(0.5)
    assert raw["experiment_operation_count"] == 1


def test_headline_median_excludes_controls_but_dominance_does_not() -> None:
    analysis = {
        "throughput-uplift": [
            {
                "family": "coarse-friendly",
                "throughput_uplift_percent": -2.0,
            },
            {
                "family": "full-mixed",
                "throughput_uplift_percent": 10.0,
            },
            {
                "family": "memory-fragmented",
                "throughput_uplift_percent": 30.0,
            },
        ],
        "capacity-unlocks": [],
        "target-crossings": [],
    }

    result = headline_statistics(analysis)

    assert result["a_feasible_inventory_count"] == 3
    assert result["heterogeneous_a_feasible_inventory_count"] == 2
    assert result["median_uplift_percent"] == pytest.approx(20.0)
    assert result["wins_ge_20_percent"] == 1
    assert result["regressions_gt_1_percent"] == 1


def test_full_mixed_requires_both_expert_and_projection_runtime_support() -> None:
    inventories, _audit = load_frozen_inventories(REPO)
    node = inventories[0].nodes[0]
    expert_only = replace(node, runtime_capabilities=("EXPERT_SHARD",))
    fully_capable = replace(
        node,
        runtime_capabilities=("EXPERT_SHARD", "PROJECTION_SHARD"),
    )

    assert not SharedPlacementOptimizer._runtime_supported(
        expert_only, PartitionKind.FULL_MIXED_STRIPE
    )
    assert SharedPlacementOptimizer._runtime_supported(
        fully_capable, PartitionKind.FULL_MIXED_STRIPE
    )


def test_verdict_does_not_redefine_an_uncovered_frozen_threshold_interval() -> None:
    thresholds = {
        "strong": {
            "heterogeneous_median_uplift_percent_min": 20.0,
            "fraction_ge_20_percent_min": 1 / 3,
            "target_crossings_min": 1,
            "capacity_unlocks_min": 1,
            "coarse_control_whole_layer_percent_min": 80.0,
        },
        "supported": {"median_uplift_percent_min": 10.0},
        "capacity_only": {
            "median_performance_uplift_percent_max_exclusive": 10.0
        },
        "not_material": {
            "median_performance_uplift_percent_max_exclusive": 5.0,
            "target_crossings_allowed": 0,
        },
    }
    uplift = [{"family": "full-mixed", "throughput_uplift_percent": "7.0"}]
    adaptive = [
        {
            "family": "coarse-friendly",
            "feasible": "True",
            "whole_layer_percent": "100.0",
        }
    ]

    category, answer, details = _verdict(
        gates={"all": True},
        uplift=uplift,
        unlocks=[],
        crossings=[],
        adaptive=adaptive,
        thresholds=thresholds,
    )

    assert category == "MODEL_INVALID"
    assert answer == "MODEL INVALID"
    assert details["outcome_rule_matched"] is False
