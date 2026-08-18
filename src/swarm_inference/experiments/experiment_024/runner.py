"""Experiment 024 phase orchestration and fail-closed preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import atomic_write_json

from .freeze import audit_immutable_inputs, frozen_constants
from .geometry import A_BYTES, B_BYTES, C_BYTES, D_BYTES


def run_phase0(repo_root: Path, *, pre_ruff_findings: int = 725) -> dict[str, Any]:
    """Run and persist the repaired immutable-input/architecture gate."""

    del pre_ruff_findings
    repo_root = repo_root.resolve()
    root = repo_root / "artifacts/experiment-024"
    audit = audit_immutable_inputs(repo_root)
    phase0 = {
        "schema_version": "experiment-024-repaired-phase0-v2",
        "experiment_id": "024",
        "status": audit.status,
        "final_verdict": "NOT_YET_DETERMINED" if audit.status == "PASS" else "MODEL_INVALID",
        "performance_results_seen": False,
        "phase0_audit": audit.as_dict(),
    }
    atomic_write_json(root / "freeze/e024-frozen-inputs.json", {
        "schema_version": "experiment-024-frozen-inputs-v2",
        "experiment_id": "024",
        "status": "PHASE_0_PASS_PRE_PERFORMANCE" if audit.status == "PASS" else "MODEL_INVALID_DURING_PHASE_0",
        "performance_results_seen": False,
        "constants": frozen_constants(),
        "phase0_audit": audit.as_dict(),
        "communication_formulas": {
            "A_BYTES": "7 * ((H + R) + L + H + H + L + H)",
            "B_BYTES": "7 * (H + R + L + H + L + H)",
            "C_BYTES": "7 * (H + R + L + H + L_SLICE + H)",
            "D_BYTES": "7 * (H + R + L + L_SLICE + H)",
        },
        "communication_values": {
            "A_BYTES": A_BYTES,
            "B_BYTES": B_BYTES,
            "C_BYTES": C_BYTES,
            "D_BYTES": D_BYTES,
        },
    })
    atomic_write_json(
        root / "validation/commodity-whole-layer-feasibility.json",
        {
            "schema_version": "experiment-024-whole-layer-feasibility-v1",
            "status": "PASS" if audit.status == "PASS" else "FAIL",
            "evidence_class": "FROZEN_CANDIDATE_CATALOG_MEMORY_AUDIT",
            "commodity_worker_memory_bytes": audit.commodity_worker_memory_bytes,
            "whole_layer_resident_bytes_by_layer": audit.whole_layer_resident_bytes_by_layer,
            "whole_layer_feasible_transformer_layer_ids": list(
                audit.whole_layer_feasible_layer_ids_on_commodity
            ),
            "whole_layer_infeasible_transformer_layer_ids": list(
                audit.whole_layer_infeasible_layer_ids_on_commodity
            ),
            "whole_layer_only_complete_k3_placement": "INFEASIBLE",
            "whole_layer_only_commodity_model_feasible": (
                audit.whole_layer_only_commodity_model_feasible
            ),
            "derived_from_chosen_placement": False,
        },
    )
    atomic_write_json(
        root / "failure-log.json",
        {
            "schema_version": "experiment-024-failure-log-v2",
            "failures": (
                []
                if audit.status == "PASS"
                else [
                    {
                        "failure_id": audit.mandatory_failure_id,
                        "phase": "R3",
                        "status": "MANDATORY_MODEL_INVALID",
                        "reason": audit.reason,
                        "result_driven_assumption_changes": 0,
                    }
                ]
            ),
        },
    )
    return phase0


__all__ = ["run_phase0"]
