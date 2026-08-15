"""Fail-closed Phase 0 freeze for Experiment 023."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)
from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)

EXPERIMENT_ID: Final = "023"
EXPECTED_E022_SUITE_DIGEST: Final = (
    "3e949a8eee0a71e128493f64e0be903bd373d3baad4be86a8869d87279d4bb49"
)
EXPECTED_FAILED_USEFUL_JOINS: Final = (
    "full-mixed-01",
    "full-mixed-02",
    "full-mixed-03",
    "full-mixed-05",
)

HEADLINE_INVENTORIES: Final = (
    "memory-fragmented-01",
    "memory-fragmented-03",
    "memory-fragmented-05",
    "compute-heterogeneous-01",
    "compute-heterogeneous-02",
    "compute-heterogeneous-03",
    "compute-heterogeneous-04",
    "compute-heterogeneous-05",
    "compute-heterogeneous-06",
    "network-heterogeneous-01",
    "network-heterogeneous-02",
    "network-heterogeneous-03",
    "network-heterogeneous-04",
    "network-heterogeneous-05",
    "network-heterogeneous-06",
    "full-mixed-01",
    "full-mixed-02",
    "full-mixed-03",
)
CONTROL_INVENTORIES: Final = (
    "coarse-friendly-01",
    "coarse-friendly-02",
    "coarse-friendly-03",
)
CAPACITY_INVENTORIES: Final = (
    "memory-fragmented-02",
    "memory-fragmented-04",
    "memory-fragmented-06",
    "full-mixed-04",
    "full-mixed-05",
    "full-mixed-06",
)

FROZEN_CONSTANTS: Final[dict[str, Any]] = {
    "experiment_id": EXPERIMENT_ID,
    "seed": 23023,
    "target_rows": 17,
    "admitted_partition_degree": 8,
    "replicable_partition_kind": "WHOLE_EXPERT",
    "maximum_copies_per_logical_expert_group": 2,
    "replica_count_options": [2, 4, 8],
    "concurrency_levels": [1, 8, 32, 64, 128],
    "planning_concurrency": 32,
    "planner_candidate_layers": 12,
    "planner_primary_groups_per_layer": 2,
    "planner_max_accepted_layer_actions": 6,
    "planner_minimum_relative_gain": 0.005,
    "planner_local_throughput_regression_limit": 0.01,
    "economic_fastest_throughput_floor": 0.95,
    "primary_latency_multiplier": 2.0,
    "latency_sensitivity_multipliers": [1.5, 2.0, 4.0],
    "whole_expert_relative_l2_gate": 2e-6,
    "replica_memory_error_percent_max": 5.0,
    "general_wedge_efficiency_median_percent_min": 20.0,
    "general_wedge_min_cases_ge_20": 12,
    "family_wedge_efficiency_median_percent_min": 20.0,
    "headline_regression_percent_max": 5.0,
    "hedge_seed_count": 32,
    "hedge_seed_start": 23023000,
    "hedge_extra_compute_percent_max": 15.0,
    "hedge_p95_latency_improvement_percent_min": 20.0,
}

_REQUIRED_E022_FILES: Final = (
    "artifacts/experiment-022/completion/frozen-inputs.json",
    "artifacts/experiment-022/completion/rerun/ablation-results.csv",
    "artifacts/experiment-022/completion/rerun/adaptive-results.csv",
    "artifacts/experiment-022/completion/rerun/dynamic-results.csv",
    "artifacts/experiment-022/completion/physical/chunk-1-services.csv",
    "artifacts/experiment-022/completion/physical/chunk-2-services.csv",
    "artifacts/experiment-022/completion/physical/chunk-4-services.csv",
    "artifacts/experiment-022/completion/physical/whole-expert-services.json",
    "artifacts/experiment-022/completion/correctness/final-headline-manifests.json",
    "artifacts/experiment-022/completion/completion-summary.json",
    "artifacts/experiment-022/completion/implementation/execute-shard-bindings.json",
    "artifacts/experiment-022/completion/physical/execute-shard-bindings-chunk-1.json",
    "artifacts/experiment-022/completion/physical/execute-shard-bindings-chunk-2.json",
    "artifacts/experiment-022/completion/physical/execute-shard-bindings-chunk-4.json",
    "artifacts/experiment-022/completion/validation/repaired-resident-service.csv",
    "artifacts/experiment-022/completion/validation/repaired-service-manifest.json",
    "artifacts/experiment-022/completion/validation/repaired-event-services.json",
    "artifacts/experiment-018/physical/layer-service.csv",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"E023_FREEZE_INVALID: expected object in {path}")
    return value


def _assert_binding_receipt(receipt: dict[str, Any], *, rows: int | None) -> None:
    required = (
        receipt.get("status") == "PASS"
        and receipt.get("all_six_bound") is True
        and receipt.get("all_authenticated") is True
        and receipt.get("all_production_native") is True
        and int(receipt.get("operation_count", -1)) == 6
        and int(receipt.get("checkpoint_reads_in_timed_region", -1)) == 0
        and int(receipt.get("whole_layer_fallback_count", -1)) == 0
    )
    if rows is not None:
        required = required and int(receipt.get("rows", -1)) == rows
        cases = list(receipt.get("cases", ()))
        required = required and len(cases) == 6 and all(
            row.get("status") == "PASS"
            and row.get("production_native_binding") is True
            and row.get("authenticated_frame_round_trip") is True
            and int(row.get("checkpoint_reads_in_timed_region", -1)) == 0
            and int(row.get("whole_layer_fallback_count", -1)) == 0
            for row in cases
        )
    if not required:
        suffix = "aggregate" if rows is None else f"rows={rows}"
        raise RuntimeError(f"E023_FREEZE_INVALID: EXECUTE_SHARD admission failed ({suffix})")


def _hash_rows(repo: Path, paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted({value.resolve() for value in paths}, key=lambda item: str(item)):
        if not path.is_file():
            raise RuntimeError(f"E023_FREEZE_INVALID: required E022 input missing: {path}")
        try:
            relative = path.relative_to(repo.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"E023_FREEZE_INVALID: input outside repository: {path}") from exc
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def freeze_e023(repo: Path) -> dict[str, Any]:
    """Verify immutable E022 evidence and write the preregistered E023 freeze."""

    root = repo.resolve()
    completion = root / "artifacts" / "experiment-022" / "completion"
    inventories, inventory_audit = load_frozen_inventories(root)
    if inventory_audit["suite_canonical_sha256"] != EXPECTED_E022_SUITE_DIGEST:
        raise RuntimeError("E023_FREEZE_INVALID: E022 inventory suite digest changed")

    inventory_ids = tuple(value.inventory_id for value in inventories)
    expected_ids = set(HEADLINE_INVENTORIES + CONTROL_INVENTORIES + CAPACITY_INVENTORIES)
    if len(inventory_ids) != 27 or set(inventory_ids) != expected_ids:
        raise RuntimeError("E023_FREEZE_INVALID: exact 27-inventory cohort changed")

    placement_root = completion / "rerun" / "placements"
    placement_paths = [
        placement_root / f"{inventory_id}-{planner}.json"
        for inventory_id in inventory_ids
        for planner in "ABCDE"
    ]
    missing = [path for path in placement_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"E023_FREEZE_INVALID: {len(missing)} frozen A/B/C/D/E placements missing"
        )

    summary = _read_object(completion / "completion-summary.json")
    if summary.get("verdict_category") != "MODEL_INVALID":
        raise RuntimeError("E023_FREEZE_INVALID: E022 verdict is no longer MODEL_INVALID")
    failed_joins = tuple(
        row.get("inventory_id") for row in summary.get("dynamic_failure_cases", ())
    )
    if failed_joins != EXPECTED_FAILED_USEFUL_JOINS:
        raise RuntimeError("E023_FREEZE_INVALID: E022 failed useful-node joins changed")

    aggregate_binding = _read_object(
        completion / "implementation" / "execute-shard-bindings.json"
    )
    _assert_binding_receipt(aggregate_binding, rows=None)
    for rows in (1, 2, 4):
        _assert_binding_receipt(
            _read_object(
                completion / "physical" / f"execute-shard-bindings-chunk-{rows}.json"
            ),
            rows=rows,
        )
    if not summary.get("gates", {}).get("chunk_2") or not summary.get("gates", {}).get(
        "chunk_4"
    ):
        raise RuntimeError("E023_FREEZE_INVALID: E022 chunk 2/4 admission changed")

    required_paths = [root / relative for relative in _REQUIRED_E022_FILES]
    required_paths.extend(placement_paths)
    required_paths.extend(root / row["path"] for row in inventory_audit["inventories"])
    suite_rows = [
        root / row["path"]
        for row in _read_object(completion / "frozen-inputs.json").get("core_files", ())
        if str(row.get("path", "")).replace("\\", "/").endswith(
            "/inventories/inventory-suite.json"
        )
    ]
    required_paths.extend(suite_rows)
    hash_rows = _hash_rows(root, required_paths)

    cohorts = {
        "schema_version": "experiment-023-headline-cohorts-v1",
        "experiment_id": EXPERIMENT_ID,
        "headline": list(HEADLINE_INVENTORIES),
        "negative_controls": list(CONTROL_INVENTORIES),
        "capacity_exploratory": list(CAPACITY_INVENTORIES),
        "counts": {"headline": 18, "negative_controls": 3, "capacity_exploratory": 6},
        "canonical_sha256": canonical_sha256(
            {
                "headline": list(HEADLINE_INVENTORIES),
                "negative_controls": list(CONTROL_INVENTORIES),
                "capacity_exploratory": list(CAPACITY_INVENTORIES),
            }
        ),
    }
    input_hashes = {
        "schema_version": "experiment-023-e022-input-hashes-v1",
        "status": "PASS",
        "source_experiment": "022",
        "source_verdict": "MODEL_INVALID",
        "inventory_suite_sha256": EXPECTED_E022_SUITE_DIGEST,
        "file_count": len(hash_rows),
        "files": hash_rows,
        "canonical_sha256": canonical_sha256(hash_rows),
    }
    frozen = {
        "schema_version": "experiment-023-frozen-inputs-v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "A heterogeneous resource pool is more useful when excess memory creates "
            "alternative execution paths than when every additional worker creates "
            "another mandatory dependency."
        ),
        "constants": FROZEN_CONSTANTS,
        "constants_sha256": canonical_sha256(FROZEN_CONSTANTS),
        "cohorts_sha256": cohorts["canonical_sha256"],
        "e022_input_hashes_sha256": input_hashes["canonical_sha256"],
        "e022_inventory_suite_sha256": EXPECTED_E022_SUITE_DIGEST,
        "e022_verdict": "MODEL_INVALID",
        "failed_useful_node_joins": list(EXPECTED_FAILED_USEFUL_JOINS),
        "inventory_count": 27,
        "placement_manifest_count": 135,
        "all_six_execute_shard_bindings_admitted": True,
        "chunk_2_physical_completion_admitted": True,
        "chunk_4_physical_completion_admitted": True,
        "thresholds_frozen_before_headline": True,
    }

    output = root / "artifacts" / "experiment-023" / "freeze"
    atomic_write_json(output / "headline-cohorts.json", cohorts)
    atomic_write_json(output / "e022-input-hashes.json", input_hashes)
    atomic_write_json(output / "e023-frozen-inputs.json", frozen)
    return frozen


def validate_e023_freeze(repo: Path) -> dict[str, Any]:
    """Re-run Phase 0 and require byte-stable preregistration values."""

    root = repo.resolve()
    freeze_path = root / "artifacts" / "experiment-023" / "freeze"
    existing = _read_object(freeze_path / "e023-frozen-inputs.json")
    if existing.get("constants") != FROZEN_CONSTANTS:
        raise RuntimeError("E023_FREEZE_INVALID: constants changed after freeze")
    if existing.get("constants_sha256") != canonical_sha256(FROZEN_CONSTANTS):
        raise RuntimeError("E023_FREEZE_INVALID: frozen threshold hash changed")
    regenerated = freeze_e023(root)
    if regenerated != existing:
        raise RuntimeError("E023_FREEZE_INVALID: E022 inputs changed after E023 freeze")
    return regenerated


__all__ = [
    "CAPACITY_INVENTORIES",
    "CONTROL_INVENTORIES",
    "EXPERIMENT_ID",
    "FROZEN_CONSTANTS",
    "HEADLINE_INVENTORIES",
    "freeze_e023",
    "validate_e023_freeze",
]
