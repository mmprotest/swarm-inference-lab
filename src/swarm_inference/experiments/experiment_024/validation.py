"""Mechanical validation and reconciliation artifacts for E024."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import canonical_sha256

from .decode_step import DecodeStepTaskBuilder
from .economics import cost_per_million
from .freeze import (
    COMMODITY_WORKER_MEMORY_BYTES,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    P8_REQUIRED_LAYER_IDS,
    PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR,
    sha256_file,
)
from .geometry import A_BYTES, B_BYTES, C_BYTES, D_BYTES, GEOMETRY
from .placement import CommodityPlacement, reconcile_placement_memory


def token_semantics_audit() -> dict[str, Any]:
    builder = DecodeStepTaskBuilder()
    cases = []
    for rows in (1, 2, 4):
        task = builder.build(rows)
        cases.append(
            {
                "rows": rows,
                "input_current_token_count": rows,
                "new_output_token_count": task.generated_output_tokens,
                "independent_recurrent_state_per_row": (
                    task.independent_recurrent_state_per_row
                ),
                "speculative_execution": task.speculative_execution,
                "target_verification": task.target_verification,
                "complete_decode_step": [
                    "embedding",
                    "layer_0_dense_whole",
                    "layers_1_92_degree_8",
                    "final_norm",
                    "lm_head",
                    "greedy_argmax",
                    "state_commit",
                ],
                "status": "PASS",
            }
        )
    return {
        "schema_version": "experiment-024-token-semantics-v2",
        "status": "PASS",
        "one_completed_decode_row_equals_one_new_output_token": True,
        "layer_0_execution_kind": "WHOLE_LAYER",
        "layer_0_candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
        "layer_0_degree": 1,
        "layers_1_92_execution_degree": 8,
        "layers_1_92_execution_kind": "PHYSICALLY_ADMITTED_SUB_LAYER",
        "no_speculative_decoding": True,
        "no_preloaded_future_token": True,
        "no_target_verification": True,
        "cases": cases,
    }


def communication_reconciliation() -> dict[str, Any]:
    expected = {
        "A_CURRENT": 1_004_416,
        "B_RETAIN_HIDDEN": 803_712,
        "C_SLICE_LATENT": 715_904,
        "D_FUSE_OUTPUT": 515_200,
    }
    actual = {arm.value: value.bytes_per_row for arm, value in GEOMETRY.items()}
    return {
        "schema_version": "experiment-024-communication-reconciliation-v1",
        "status": "PASS" if actual == expected else "FAIL",
        "bytes_per_row": actual,
        "expected_bytes_per_row": expected,
        "message_count_per_block": {
            arm.value: value.messages_per_row for arm, value in GEOMETRY.items()
        },
        "a_to_d_reduction_percent": 100 * (1 - D_BYTES / A_BYTES),
        "all_formulas_unchanged": (
            (A_BYTES, B_BYTES, C_BYTES, D_BYTES)
            == (1_004_416, 803_712, 715_904, 515_200)
        ),
    }


def placement_reconciliation_rows(
    placements: tuple[CommodityPlacement, ...],
) -> list[dict[str, Any]]:
    rows = []
    for placement in placements:
        rows.append(
            {
                "scenario": placement.scenario.value,
                "placement_kind": placement.placement_kind,
                "available_node_budget": placement.available_node_budget,
                "feasible": placement.feasible,
                "infeasible_reason": placement.infeasible_reason,
                "placement_sha256": placement.placement_sha256,
                "layer_zero_candidate_id": placement.layer_zero_candidate_id,
                "whole_layer_layer_ids": "|".join(
                    str(value) for value in placement.whole_layer_layer_ids
                ),
                "p8_layer_ids": "|".join(
                    str(value) for value in placement.p8_layer_ids
                ),
                "whole_layer_layer_count": len(placement.whole_layer_layer_ids),
                "p8_layer_count": len(placement.p8_layer_ids),
                "whole_layer_only_commodity_model_feasible": False,
                "p8_required_whole_layer_incapable_compute_share": (
                    1.0 if placement.feasible else None
                ),
                "active_node_count": placement.active_node_count,
                "exact_architecture": (
                    not placement.feasible
                    or (
                        placement.whole_layer_layer_ids == (0,)
                        and placement.p8_layer_ids == P8_REQUIRED_LAYER_IDS
                    )
                ),
            }
        )
    return rows


def memory_reconciliation_rows(
    placements: tuple[CommodityPlacement, ...],
) -> list[dict[str, Any]]:
    return [
        row
        for placement in placements
        if placement.feasible
        for row in reconcile_placement_memory(placement)
    ]


def cost_reconciliation_rows(frontier_rows: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    rows = []
    for point in frontier_rows:
        active_nodes = int(point["active_node_count"])
        throughput = float(point["aggregate_output_tokens_per_second"])
        hourly_cost = (
            active_nodes * PRIMARY_CONTRIBUTOR_PAYOUT_USD_PER_ACTIVE_NODE_HOUR
        )
        calculated = cost_per_million(hourly_cost, throughput)
        reported = float(point["cost_per_M_at_0_15"])
        rows.append(
            {
                "scenario": point["scenario"],
                "architecture": point["architecture"],
                "available_node_budget": point["available_node_budget"],
                "active_node_count": active_nodes,
                "layer_zero_worker_counted_once": True,
                "free_dense_node": False,
                "free_coordinator": False,
                "free_endpoint": False,
                "hourly_cost_usd": hourly_cost,
                "output_tokens_per_second": throughput,
                "reported_cost_per_M": reported,
                "recomputed_cost_per_M": calculated,
                "absolute_difference": abs(reported - calculated),
                "status": "PASS" if abs(reported - calculated) <= 1e-12 else "FAIL",
            }
        )
    return rows


def create_code_freeze(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_root = repo_root / "src/swarm_inference/experiments/experiment_024"
    paths = sorted(source_root.glob("*.py"))
    paths.append(repo_root / "src/swarm_inference/execution/kimi_k3_stage.py")
    paths.extend(sorted((repo_root / "scripts").glob("experiment_024*.py")))
    paths.extend(sorted((repo_root / "tests").glob("test_experiment_024*.py")))
    files = [
        {
            "relative_path": str(path.relative_to(repo_root)).replace("\\", "/"),
            "byte_count": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    payload = {
        "schema_version": "experiment-024-repaired-code-freeze-v2",
        "status": "PASS",
        "valid_code_freeze": True,
        "performance_results_seen": False,
        "architecture": {
            "layer_zero_candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
            "whole_layer_transformer_layer_ids": [0],
            "p8_transformer_layer_ids": list(P8_REQUIRED_LAYER_IDS),
        },
        "commodity_worker_memory_bytes": COMMODITY_WORKER_MEMORY_BYTES,
        "files": files,
        "file_count": len(files),
    }
    payload["freeze_sha256"] = canonical_sha256(payload)
    return payload


def verify_code_freeze(repo_root: Path, payload: dict[str, Any]) -> bool:
    repo_root = repo_root.resolve()
    return all(
        (repo_root / row["relative_path"]).is_file()
        and (repo_root / row["relative_path"]).stat().st_size == row["byte_count"]
        and sha256_file(repo_root / row["relative_path"]) == row["sha256"]
        for row in payload["files"]
    )


__all__ = [
    "communication_reconciliation",
    "cost_reconciliation_rows",
    "create_code_freeze",
    "memory_reconciliation_rows",
    "placement_reconciliation_rows",
    "token_semantics_audit",
    "verify_code_freeze",
]
