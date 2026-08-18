"""Frozen-placement execution reproducibility checks for E024."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import canonical_sha256

from .models import Architecture
from .placement import CommodityPlacement
from .service import E024ServiceTable
from .stage_a import run_stage_a
from .stage_b import _execute_commodity


def stage_a_reproducibility(
    service: E024ServiceTable,
    authoritative_rows: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    repeated = run_stage_a(service)
    authoritative_hash = canonical_sha256(list(authoritative_rows))
    repeated_hash = canonical_sha256(list(repeated.rows))
    return {
        "schema_version": "experiment-024-stage-a-reproducibility-v1",
        "status": "PASS" if authoritative_hash == repeated_hash else "FAIL",
        "authoritative_sha256": authoritative_hash,
        "repeated_sha256": repeated_hash,
        "row_count": len(authoritative_rows),
        "exact_match": authoritative_hash == repeated_hash,
        "placement_rerun": False,
        "deterministic_event_model": True,
    }


def stage_b_reproducibility(
    repo_root: Path,
    service: E024ServiceTable,
    canonical_rows: tuple[dict[str, Any], ...],
    frozen_placements: tuple[CommodityPlacement, ...],
) -> dict[str, Any]:
    lookup = {
        (
            placement.scenario.value,
            placement.placement_kind,
            placement.available_node_budget,
        ): placement
        for placement in frozen_placements
    }
    cases = []
    for authoritative in canonical_rows:
        architecture = Architecture(str(authoritative["architecture"]))
        placement_kind = (
            "CURRENT_PLACEMENT"
            if architecture is Architecture.SWARM_CURRENT_OPT
            else "D_PLACEMENT"
        )
        key = (
            str(authoritative["scenario"]),
            placement_kind,
            int(authoritative["available_node_budget"]),
        )
        placement = lookup[key]
        repeated = _execute_commodity(
            repo_root,
            service,
            placement,
            int(authoritative["selected_slo_concurrency"]),
        )
        fields = (
            "placement_sha256",
            "task_graph_sha256",
            "aggregate_output_tokens_per_second",
            "p50_token_latency_ms",
            "p95_token_latency_ms",
            "network_bytes",
            "worker_compute_ms",
            "layer_zero_candidate_id",
            "whole_layer_layer_count",
            "p8_layer_count",
            "p8_required_whole_layer_incapable_compute_share",
        )
        exact = all(repeated[field] == authoritative[field] for field in fields)
        cases.append(
            {
                "scenario": authoritative["scenario"],
                "architecture": authoritative["architecture"],
                "available_node_budget": authoritative["available_node_budget"],
                "concurrency": authoritative["selected_slo_concurrency"],
                "placement_sha256": placement.placement_sha256,
                "placement_loaded_from_freeze": True,
                "placement_rerun": False,
                "exact_fields": list(fields),
                "exact_match": exact,
                "status": "PASS" if exact else "FAIL",
            }
        )
    return {
        "schema_version": "experiment-024-stage-b-reproducibility-v1",
        "status": "PASS" if all(row["exact_match"] for row in cases) else "FAIL",
        "frozen_placement_execution_only": True,
        "placement_determinism_tested_before_freeze": True,
        "cases": cases,
    }


__all__ = ["stage_a_reproducibility", "stage_b_reproducibility"]
