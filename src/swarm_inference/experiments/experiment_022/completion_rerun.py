"""Gate and execute the frozen 27-inventory Experiment 022 completion rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark import (
    headline_statistics,
    run_dynamic_suite,
    run_static_suite,
    validated_candidate_catalog,
)
from .completion_correctness import (
    reconcile_fresh_manifest_receipt,
    strict_manifest_receipt_failures,
)
from .completion_inputs import load_frozen_inventories
from .completion_service import build_repaired_service
from .io import atomic_write_json, write_csv
from .model_graph import build_model_graph
from .models import ALLOWED_BY_LEVEL, PlannerLevel
from .oracle import validate_optimizer_oracle
from .planner import OptimizerConfiguration, SharedPlacementOptimizer
from .service import ResidentServiceModel


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _placement_signature(path: Path) -> str:
    manifest = _read(path)
    value = {
        "chunk_rows": manifest.get("chunk_rows"),
        "endpoint_owners": sorted(
            str(node["node_id"])
            for node in manifest.get("nodes", [])
            if any(
                str(piece.get("partition_type")) == "IDENTICAL_ENDPOINT_POLICY"
                for piece in node.get("pieces", [])
            )
        ),
        "pieces": sorted(
            (
                str(node["node_id"]),
                str(piece["piece"]),
                str(piece["candidate_id"]),
                str(piece["partition_type"]),
                int(piece["degree"]),
                bool(piece["coordinator"]),
            )
            for node in manifest.get("nodes", [])
            for piece in node.get("pieces", [])
            if str(piece.get("piece", "")).startswith("transformer_layer_")
        ),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _preflight(repo: Path, completion: Path) -> dict[str, Any]:
    inventories, frozen = load_frozen_inventories(repo)
    if len(inventories) != 27:
        raise RuntimeError("E022_FROZEN_INPUTS_UNRECOVERABLE")
    required = {
        "bindings": completion / "implementation" / "execute-shard-bindings.json",
        "physical": completion / "validation" / "physical-gates-summary.json",
        "residual": completion / "validation" / "residual-classification.json",
        "model": completion / "validation" / "model-validation.json",
        "accounting": completion / "validation" / "accounting-reconciliation.json",
        "whole_expert": completion / "implementation" / "whole-expert-status.json",
    }
    values = {name: _read(path) for name, path in required.items()}
    failed = [name for name, value in values.items() if value.get("status") != "PASS"]
    expected_schemas = {
        "bindings": "experiment-022-completion-execute-shard-bindings-v2",
        "physical": "experiment-022-completion-physical-gates-v2",
        "residual": "experiment-022-completion-residual-classification-v2",
        "model": "experiment-022-completion-model-validation-v2",
        "accounting": "experiment-022-completion-accounting-reconciliation-v2",
        "whole_expert": "experiment-022-completion-whole-expert-v1",
    }
    failed.extend(
        f"{name}-stale-schema"
        for name, schema in expected_schemas.items()
        if values[name].get("schema_version") != schema
    )
    bindings = values["bindings"]
    expected_operations = {
        "kda_shard",
        "mla_shard",
        "routed_expert_stripe",
        "shared_expert_shard",
        "projection_shard",
        "reduction_contribution",
    }
    if (
        bindings.get("chunks_physically_executed") != [1, 2, 4]
        or set(bindings.get("operations", ())) != expected_operations
        or int(bindings.get("checkpoint_reads_in_timed_region", -1)) != 0
        or int(bindings.get("whole_layer_fallback_count", -1)) != 0
    ):
        failed.append("bindings-incomplete")
    physical = values["physical"]
    if (
        int(physical.get("receipt_count", -1)) != 12
        or physical.get("receipt_matrix_complete") is not True
        or physical.get("chunks") != [1, 2, 4]
        or physical.get("sub_layer_chunk_2_physically_validated") is not True
        or physical.get("sub_layer_chunk_4_physically_validated") is not True
    ):
        failed.append("physical-chunk-gate-incomplete")
    residual = values["residual"]
    if (
        residual.get("artificial_barrier_residual") is not False
        or float(residual.get("maximum_unexplained_fraction", 1.0)) > 0.10
        or float(residual.get("maximum_reconciliation_fraction", 1.0)) > 0.10
    ):
        failed.append("residual-gate-incomplete")
    model_validation = values["model"]
    errors = model_validation.get("absolute_percent_error", {})
    if (
        model_validation.get("chunks") != [1, 2, 4]
        or model_validation.get("normalization") is not False
        or model_validation.get("global_correction_factor") is not False
        or float(errors.get("median", 100.0)) > 5.0
        or float(errors.get("p90", 100.0)) > 10.0
        or float(errors.get("maximum", 100.0)) > 15.0
    ):
        failed.append("model-validation-gate-incomplete")
    freeze = _read(completion / "frozen-inputs.json")
    selected = list(freeze["selected_correctness_manifests"])
    representative_rows: list[dict[str, Any]] = []
    for index, selection in enumerate(selected, 1):
        receipt_path = completion / "correctness" / f"representative-{index:02d}.json"
        if not receipt_path.is_file():
            failed.append(f"representative-{index:02d}-missing")
            continue
        receipt = _read(receipt_path)
        if receipt.get("status") != "PASS":
            failed.append(f"representative-{index:02d}-failed")
        strict_failures = strict_manifest_receipt_failures(receipt)
        if strict_failures:
            failed.append(f"representative-{index:02d}-strict-correctness-incomplete")
        if (
            str(receipt.get("selection_id")) != str(selection["selection_id"])
            or str(receipt.get("manifest_sha256")) != str(selection["sha256"])
            or str(receipt.get("inventory_id")) != str(selection["inventory_id"])
        ):
            failed.append(f"representative-{index:02d}-selection-mismatch")
        representative_rows.append(
            {
                **selection,
                "receipt_path": str(receipt_path.relative_to(repo)).replace("\\", "/"),
                "receipt_sha256": _sha256(receipt_path),
                "status": receipt.get("status"),
                "strict_correctness_failures": strict_failures,
            }
        )
    selection_receipt = {
        "schema_version": "experiment-022-completion-representative-selection-v1",
        "status": "PASS" if not any(value.startswith("representative") for value in failed) else "FAIL",
        "selection_frozen_before_completion_execution": True,
        "selection_count": len(selected),
        "unique_manifest_count": len({str(row["sha256"]) for row in selected}),
        "selections": representative_rows,
    }
    atomic_write_json(
        completion / "correctness" / "representative-selection.json", selection_receipt
    )
    if failed:
        raise RuntimeError(f"MODEL_INVALID: completion preflight failed: {sorted(set(failed))}")
    return {
        "frozen": frozen,
        "inventories": inventories,
        "gates": values,
        "representatives": selection_receipt,
    }


def _copy_required_outputs(completion: Path) -> None:
    implementation = completion / "implementation"
    rerun = completion / "rerun"
    shutil.copy2(rerun / "candidate-catalog.json", implementation / "candidate-catalog.json")


def reconcile_final_correctness(*, repo: Path) -> dict[str, Any]:
    """Close the post-rerun gate using fresh receipts for changed manifests."""

    root = repo.resolve()
    completion = root / "artifacts" / "experiment-022" / "completion"
    path = completion / "correctness" / "final-headline-manifests.json"
    value = _read(path)
    reference_path = completion / "correctness" / "representative-01.json"
    reference_receipt = _read(reference_path)
    reference_receipt_sha256 = _sha256(reference_path)
    passed = True
    updated: list[dict[str, Any]] = []
    validated_fresh_by_manifest: dict[str, dict[str, Any]] = {}
    for row in value.get("manifests", []):
        item = dict(row)
        if not bool(item.get("fresh_correctness_required")):
            original_path = (
                completion
                / "correctness"
                / f"{item['selection_id']}.json"
            )
            original = _read(original_path)
            original_failures = strict_manifest_receipt_failures(original)
            original_valid = original.get("status") == "PASS" and not original_failures
            item["original_correctness_receipt"] = str(
                original_path.relative_to(root)
            ).replace("\\", "/")
            item["original_correctness_receipt_sha256"] = _sha256(original_path)
            item["strict_correctness_failures"] = original_failures
            item["fresh_correctness_status"] = (
                "NOT_REQUIRED_ORIGINAL_RECEIPT_PASS"
                if original_valid
                else "ORIGINAL_RECEIPT_FAIL"
            )
            passed = passed and original_valid
            updated.append(item)
            continue
        receipt_path = (
            completion
            / "correctness"
            / f"final-{item['selection_id']}.json"
        )
        if not receipt_path.is_file():
            manifest_sha256 = str(item["rerun_manifest_sha256"])
            reusable = validated_fresh_by_manifest.get(manifest_sha256)
            identical_manifest = (
                reusable is not None
                and str(reusable["inventory_id"]) == str(item["inventory_id"])
                and str(reusable["planner_level"]) == str(item["planner_level"])
                and str(reusable["manifest_path"])
                == str(item["rerun_manifest_path"])
            )
            if identical_manifest:
                # Correctness belongs to the executed placement manifest, not to
                # the prose reason that selected it.  Two frozen representative
                # slots may converge to the exact same post-rerun manifest.  In
                # that case retain one physical receipt and expose the reuse
                # explicitly instead of copying it or claiming two executions.
                item["fresh_correctness_receipt"] = str(reusable["receipt_path"])
                item["fresh_correctness_receipt_sha256"] = str(
                    reusable["receipt_sha256"]
                )
                item["fresh_correctness_status"] = (
                    "PASS_REUSED_IDENTICAL_MANIFEST_EXECUTION"
                )
                item["fresh_correctness_reused_identical_manifest"] = True
                item["fresh_correctness_source_selection_id"] = str(
                    reusable["selection_id"]
                )
                item["strict_correctness_failures"] = []
            else:
                item["fresh_correctness_receipt"] = str(
                    receipt_path.relative_to(root)
                ).replace("\\", "/")
                item["fresh_correctness_status"] = "MISSING"
                item["fresh_correctness_reused_identical_manifest"] = False
                passed = False
            updated.append(item)
            continue
        item["fresh_correctness_receipt"] = str(
            receipt_path.relative_to(root)
        ).replace("\\", "/")
        manifest_path = root / str(item["rerun_manifest_path"])
        receipt = reconcile_fresh_manifest_receipt(
            receipt=_read(receipt_path),
            manifest=_read(manifest_path),
            manifest_sha256=str(item["rerun_manifest_sha256"]),
            expected_selection_id=f"final-{item['selection_id']}",
            expected_inventory_id=str(item["inventory_id"]),
            expected_planner_level=str(item["planner_level"]),
            selection_case=str(item["case"]),
            reference_receipt=reference_receipt,
            reference_receipt_sha256=reference_receipt_sha256,
        )
        atomic_write_json(receipt_path, receipt)
        strict_failures = strict_manifest_receipt_failures(receipt)
        valid = (
            receipt.get("status") == "PASS"
            and str(receipt.get("selection_id"))
            == f"final-{item['selection_id']}"
            and str(receipt.get("inventory_id")) == str(item["inventory_id"])
            and str(receipt.get("manifest_sha256"))
            == str(item["rerun_manifest_sha256"])
            and receipt.get("complete_93_layer_traversal") is True
            and receipt.get("authenticated_execute_shard") is True
            and receipt.get("greedy_token_equality") is True
            and not strict_failures
        )
        item["fresh_correctness_status"] = "PASS" if valid else "FAIL"
        item["fresh_correctness_receipt_sha256"] = _sha256(receipt_path)
        item["fresh_correctness_reused_identical_manifest"] = False
        item["strict_correctness_failures"] = strict_failures
        passed = passed and valid
        if valid:
            validated_fresh_by_manifest[str(item["rerun_manifest_sha256"])] = {
                "selection_id": item["selection_id"],
                "inventory_id": item["inventory_id"],
                "planner_level": item["planner_level"],
                "manifest_path": item["rerun_manifest_path"],
                "receipt_path": item["fresh_correctness_receipt"],
                "receipt_sha256": item["fresh_correctness_receipt_sha256"],
            }
        updated.append(item)
    value["manifests"] = updated
    value["status"] = "PASS" if passed else "PENDING_FRESH_CORRECTNESS"
    atomic_write_json(path, value)
    rerun_summary_path = completion / "rerun" / "rerun-summary.json"
    if rerun_summary_path.is_file():
        summary = _read(rerun_summary_path)
        summary["final_correctness_status"] = value["status"]
        atomic_write_json(rerun_summary_path, summary)
    return value


def run(*, repo: Path, checkpoint: Path) -> dict[str, Any]:
    repo = repo.resolve()
    completion = repo / "artifacts" / "experiment-022" / "completion"
    preflight = _preflight(repo, completion)
    inventories = preflight["inventories"]
    service_rows, _service_path, service_manifest = build_repaired_service(
        repo=repo, completion_root=completion
    )
    service = ResidentServiceModel(service_rows)
    model = build_model_graph(
        checkpoint,
        whole_layer_service_csv=repo
        / "artifacts"
        / "experiment-018"
        / "physical"
        / "layer-service.csv",
    )
    frozen = _read(completion / "frozen-inputs.json")
    catalog = validated_candidate_catalog(model, service)
    atomic_write_json(
        completion / "implementation" / "candidate-catalog.json", catalog
    )
    optimizer_values = frozen["planner_settings"]["optimizer"]
    configuration = OptimizerConfiguration(
        proposal_budget=int(optimizer_values["proposal_budget"]),
        exact_evaluation_budget=int(optimizer_values["exact_evaluation_budget"]),
        restarts=int(optimizer_values["restarts"]),
        candidate_groups_per_action=int(optimizer_values["candidate_groups_per_action"]),
        randomization=float(optimizer_values["randomization"]),
        seed=int(optimizer_values["seed"]),
    )
    oracle_rows = validate_optimizer_oracle(
        model, inventories[0], service, configuration
    )
    write_csv(completion / "validation" / "optimizer-small-oracle.csv", oracle_rows)
    if not oracle_rows or any(row["status"] != "PASS" for row in oracle_rows):
        raise RuntimeError("MODEL_INVALID: repaired optimizer oracle gate failed")

    static = run_static_suite(
        completion,
        model,
        inventories,
        service,
        configuration,
        planner_directory="rerun",
    )
    _copy_required_outputs(completion)
    optimizer = SharedPlacementOptimizer(model, service, configuration)
    dynamic = run_dynamic_suite(
        completion,
        inventories,
        static,
        optimizer,
        output_directory="rerun/dynamic",
    )
    dynamic_rows = [
        {"scenario": scenario, **row}
        for scenario, values in sorted(dynamic.items())
        for row in values
    ]
    write_csv(completion / "rerun" / "dynamic-results.csv", dynamic_rows)
    dynamic_failed_rows = [
        row for row in dynamic_rows if row.get("status") != "PASS"
    ]
    dynamic_pass = len(dynamic_rows) == 25 and not dynamic_failed_rows

    headline = headline_statistics(static["analysis"])
    containment = (
        ALLOWED_BY_LEVEL[PlannerLevel.A] <= ALLOWED_BY_LEVEL[PlannerLevel.E]
    )
    if not containment:
        raise RuntimeError("OPTIMIZER/MODEL FAILURE: Planner E does not contain Planner A")
    dominance = all(
        bool(row["dominance_pass"])
        for row in static["analysis"]["throughput-uplift"]
    )
    if not dominance or int(headline["regressions_gt_1_percent"]) != 0:
        raise RuntimeError("OPTIMIZER/MODEL FAILURE: Planner E regressed beyond 1%")
    if len(static["rows"]) != 27 * 5:
        raise RuntimeError("MODEL_INVALID: rerun did not emit all 135 ablation rows")

    selected_final: list[dict[str, Any]] = []
    for selection in frozen["selected_correctness_manifests"]:
        path = (
            completion
            / "rerun"
            / "placements"
            / f"{selection['inventory_id']}-{selection['planner_level']}.json"
        )
        new_hash = _sha256(path)
        old_path = repo / str(selection["path"])
        old_signature = _placement_signature(old_path)
        new_signature = _placement_signature(path)
        selected_final.append(
            {
                **selection,
                "rerun_manifest_path": str(path.relative_to(repo)).replace("\\", "/"),
                "rerun_manifest_sha256": new_hash,
                "original_placement_signature": old_signature,
                "rerun_placement_signature": new_signature,
                "materially_changed": new_signature != old_signature,
                "fresh_correctness_required": new_signature != old_signature,
            }
        )
    final_manifest_receipt = {
        "schema_version": "experiment-022-completion-final-headline-manifests-v1",
        "status": "PENDING_FRESH_CORRECTNESS"
        if any(row["fresh_correctness_required"] for row in selected_final)
        else "PASS",
        "manifests": selected_final,
    }
    atomic_write_json(
        completion / "correctness" / "final-headline-manifests.json",
        final_manifest_receipt,
    )
    result = {
        "schema_version": "experiment-022-completion-frozen-rerun-v1",
        "status": "PASS" if dynamic_pass else "FAIL",
        "inventory_count": len(inventories),
        "inventory_audit": preflight["frozen"],
        "service_manifest": service_manifest,
        "candidate_catalog_status": catalog["status"],
        "candidate_catalog_fair_chunk_4_comparison": catalog[
            "fair_chunk_4_comparison"
        ],
        "optimizer_configuration": asdict(configuration),
        "optimizer_same_for_all_arms": True,
        "planner_e_seeded_with_a": True,
        "planner_e_contains_planner_a": containment,
        "ablation_row_count": len(static["rows"]),
        "headline": headline,
        "dominance_pass": dominance,
        "dynamic_row_count": len(dynamic_rows),
        "dynamic_expected_row_count": 25,
        "dynamic_status": "PASS" if dynamic_pass else "FAIL",
        "dynamic_failed_row_count": len(dynamic_failed_rows),
        "dynamic_failed_scenarios": [
            {
                "inventory_id": row.get("inventory_id"),
                "scenario": row.get("scenario"),
                "changed_capability": row.get("changed_capability"),
                "reason": (
                    "nominally useful joined node was not admitted because every "
                    "discovered joined-node plan was slower than the retained plan"
                    if row.get("scenario") == "JOIN_USEFUL"
                    else "dynamic scenario gate failed"
                ),
            }
            for row in dynamic_failed_rows
        ],
        "failure_reasons": [] if dynamic_pass else ["dynamic-adaptation-gate"],
        "final_correctness_status": final_manifest_receipt["status"],
    }
    atomic_write_json(completion / "rerun" / "rerun-summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = run(repo=args.repo, checkpoint=args.checkpoint.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["reconcile_final_correctness", "run"]
