"""Strict state reconciliation for the five frozen E022 manifest executions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .io import atomic_write_json

LAYERS = 93
KDA_LAYERS = 69
MLA_LAYERS = 24
RELATIVE_L2_GATE = 2e-6


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


def _attention_states(
    receipt: dict[str, Any], key: str
) -> dict[int, tuple[str, int, bool]]:
    return {
        int(row["layer"]): (
            str(row["state_fingerprint"]),
            int(row["state_bytes"]),
            bool(row["state_finite"]),
        )
        for row in receipt.get(key, ())
    }


def _attnres_states(receipt: dict[str, Any]) -> dict[int, str]:
    return {
        int(row["layer"]): str(row["fingerprint"])
        for row in receipt.get("attnres_state_fingerprints", ())
    }


def _comparison(
    actual: dict[int, Any], reference: dict[int, Any]
) -> dict[str, Any]:
    missing = sorted(set(reference) - set(actual))
    unexpected = sorted(set(actual) - set(reference))
    mismatched = sorted(
        layer
        for layer in set(actual).intersection(reference)
        if actual[layer] != reference[layer]
    )
    exact = not missing and not unexpected and not mismatched
    return {
        "exact": exact,
        "actual_layer_count": len(actual),
        "reference_layer_count": len(reference),
        "missing_layers": missing,
        "unexpected_layers": unexpected,
        "mismatched_layers": mismatched,
    }


def strict_manifest_receipt_failures(receipt: dict[str, Any]) -> list[str]:
    """Return every failed correctness fact; an empty list is a strict PASS."""

    failures: list[str] = []
    execution_status = receipt.get(
        "execution_status_before_state_reconciliation", receipt.get("status")
    )
    checks = {
        "execution_status": execution_status == "PASS",
        "frozen_selection_exact": receipt.get("frozen_selection_match") is True,
        "tensor_assignment_coverage": receipt.get(
            "complete_tensor_assignment_coverage"
        )
        is True,
        "assigned_layers": int(receipt.get("assigned_layers", -1)) == LAYERS,
        "complete_traversal": receipt.get("complete_93_layer_traversal") is True,
        "route_equality": receipt.get("route_equality") is True,
        "ordered_expert_equality": receipt.get("ordered_expert_equality") is True,
        "all_state_finite": receipt.get("all_state_finite") is True,
        "kda_state_equality": receipt.get("kda_state_equality") is True,
        "mla_state_equality": receipt.get("mla_state_equality") is True,
        "attnres_state_equality": receipt.get("attnres_state_equality") is True,
        "state_equality_basis": receipt.get("state_equality_basis_valid") is True,
        "hidden_relative_l2": float(
            receipt.get("hidden_relative_l2_maximum", float("inf"))
        )
        <= RELATIVE_L2_GATE,
        "logit_relative_l2": float(
            receipt.get("logit_relative_l2", float("inf"))
        )
        <= RELATIVE_L2_GATE,
        "greedy_token_equality": receipt.get("greedy_token_equality") is True,
        "authenticated_execute_shard": receipt.get("authenticated_execute_shard")
        is True,
        "logical_worker_instantiation": receipt.get(
            "logical_worker_instantiation_exact"
        )
        is True,
        "persistent_expert_worker": receipt.get(
            "persistent_expert_worker_gate"
        )
        is True,
        "no_whole_layer_fallback": receipt.get(
            "whole_layer_fallback_for_split_layers"
        )
        is False,
        "complete_expert_bank": receipt.get(
            "full_expert_bank_resident_for_every_split_task"
        )
        is True,
        "zero_timed_checkpoint_reads": int(
            receipt.get("checkpoint_reads_in_expert_timed_regions", -1)
        )
        == 0,
        "kda_fingerprint_coverage": len(
            receipt.get("kda_state_fingerprints", ())
        )
        == KDA_LAYERS,
        "mla_fingerprint_coverage": len(
            receipt.get("mla_state_fingerprints", ())
        )
        == MLA_LAYERS,
        "attnres_fingerprint_coverage": len(
            receipt.get("attnres_state_fingerprints", ())
        )
        == LAYERS,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return failures


def _numerical_state_reference(
    receipt: dict[str, Any],
    *,
    expected_reference_hash: str,
) -> dict[str, Any]:
    validation = receipt.get("state_reference_validation", {})
    rows = list(validation.get("layers", ()))
    common = (
        validation.get("status") == "PASS"
        and validation.get("reference_selection_id") == "representative-01"
        and validation.get("reference_manifest_sha256")
        == expected_reference_hash
        and float(validation.get("relative_l2_gate", float("inf")))
        == RELATIVE_L2_GATE
        and int(validation.get("layer_count", -1)) == LAYERS
        and {int(row["layer"]) for row in rows} == set(range(LAYERS))
        and all(row.get("status") == "PASS" for row in rows)
    )
    kda_rows = [row for row in rows if row.get("attention_type") == "KDA"]
    mla_rows = [
        row for row in rows if row.get("attention_type") == "Gated_MLA"
    ]
    kda = common and len(kda_rows) == KDA_LAYERS and all(
        float(row["attention_state_relative_l2"]) <= RELATIVE_L2_GATE
        for row in kda_rows
    )
    mla = common and len(mla_rows) == MLA_LAYERS and all(
        float(row["attention_state_relative_l2"]) <= RELATIVE_L2_GATE
        for row in mla_rows
    )
    attnres = common and all(
        float(row["attnres_metrics"]["relative_l2_error"])
        <= RELATIVE_L2_GATE
        for row in rows
    )
    return {
        "valid": bool(common and kda and mla and attnres),
        "basis": (
            "numerical array comparison with independently executed frozen "
            "representative-01 whole-layer state trace"
        ),
        "relative_l2_gate": RELATIVE_L2_GATE,
        "reference_selection_id": validation.get("reference_selection_id"),
        "reference_manifest_sha256": validation.get(
            "reference_manifest_sha256"
        ),
        "layer_count": len(rows),
        "kda": bool(kda),
        "mla": bool(mla),
        "attnres": bool(attnres),
        "kda_relative_l2_maximum": max(
            (float(row["attention_state_relative_l2"]) for row in kda_rows),
            default=None,
        ),
        "mla_relative_l2_maximum": max(
            (float(row["attention_state_relative_l2"]) for row in mla_rows),
            default=None,
        ),
        "attnres_relative_l2_maximum": max(
            (
                float(row["attnres_metrics"]["relative_l2_error"])
                for row in rows
            ),
            default=None,
        ),
    }


def _manifest_active_workers(manifest: dict[str, Any]) -> set[str]:
    transformer_workers: set[str] = set()
    endpoint_workers: list[str] = []
    for node in manifest["nodes"]:
        worker_id = str(node["node_id"])
        for piece in node.get("pieces", ()):
            name = str(piece.get("piece", ""))
            if name.startswith("transformer_layer_"):
                transformer_workers.add(worker_id)
            if piece.get("partition_type") == "IDENTICAL_ENDPOINT_POLICY":
                endpoint_workers.append(worker_id)
    if not endpoint_workers:
        raise ValueError("frozen manifest has no endpoint owner")
    transformer_workers.add(sorted(endpoint_workers)[0])
    return transformer_workers


def _enrich_execution_facts(
    *,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Derive audit facts omitted by the first whole-only receipt schema."""

    expected_workers = _manifest_active_workers(manifest)
    actual_workers = {
        str(row["worker_id"])
        for row in receipt.get("layer_dispatch_receipts", ())
    }
    actual_workers.update(
        str(row["worker_id"])
        for row in receipt.get("expert_worker_receipts", ())
    )
    receipt.setdefault("logical_workers_expected", len(expected_workers))
    receipt.setdefault("logical_worker_ids_expected", sorted(expected_workers))
    receipt.setdefault("logical_worker_ids_instantiated", sorted(actual_workers))
    receipt.setdefault(
        "logical_worker_instantiation_exact", actual_workers == expected_workers
    )
    split_tasks = list(receipt.get("expert_worker_receipts", ()))
    if not split_tasks:
        receipt.setdefault(
            "persistent_expert_worker",
            {
                "required": False,
                "successful_startups": 0,
                "ready_status": "NOT_APPLICABLE",
                "close_status": "NOT_APPLICABLE",
                "pid": None,
                "exitcode": None,
            },
        )
        receipt.setdefault("persistent_expert_worker_gate", True)
    receipt.setdefault("runner_close_error", None)
    receipt.setdefault(
        "authentication_evidence",
        {
            "outer_expected_roundtrips": LAYERS + 1,
            "outer_receipts": len(receipt.get("layer_dispatch_receipts", ())),
            "expert_expected_roundtrips": len(split_tasks),
            "expert_dispatcher_audits": sum(
                len(row.get("dispatcher_audit", ())) for row in split_tasks
            ),
            "all_ordered_route_hashes_exact": all(
                row.get("route_ids_equal") is True
                and row.get("route_weights_equal") is True
                for row in split_tasks
            ),
            "derivation": (
                "persisted receipt rows emitted only after authenticated frame "
                "decode, native dispatch, authenticated result decode"
            ),
        },
    )
    receipt.setdefault(
        "execution_receipt_enrichment",
        {
            "scope": "mechanical facts omitted by v1 whole-layer receipt schema",
            "source": "persisted dispatch receipts plus frozen hashed manifest",
            "execution_replaced": False,
        },
    )


def reconcile_fresh_manifest_receipt(
    *,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    expected_selection_id: str,
    expected_inventory_id: str,
    expected_planner_level: str,
    selection_case: str,
    reference_receipt: dict[str, Any],
    reference_receipt_sha256: str,
) -> dict[str, Any]:
    """Apply the original strict state gate to a fresh rerun manifest receipt."""

    result = dict(receipt)
    result["selection_case"] = selection_case
    result["frozen_selection_match"] = (
        result.get("selection_id") == expected_selection_id
        and result.get("manifest_sha256") == manifest_sha256
        and result.get("inventory_id") == expected_inventory_id
        and result.get("planner_level") == expected_planner_level
    )
    result["final_rerun_selection"] = {
        "selection_id": expected_selection_id,
        "inventory_id": expected_inventory_id,
        "planner_level": expected_planner_level,
        "manifest_sha256": manifest_sha256,
        "case": selection_case,
    }
    _enrich_execution_facts(receipt=result, manifest=manifest)
    execution_status = result.get(
        "execution_status_before_state_reconciliation", result.get("status")
    )
    result["execution_status_before_state_reconciliation"] = execution_status
    numerical = _numerical_state_reference(
        result,
        expected_reference_hash=str(reference_receipt["manifest_sha256"]),
    )
    kda = _comparison(
        _attention_states(result, "kda_state_fingerprints"),
        _attention_states(reference_receipt, "kda_state_fingerprints"),
    )
    mla = _comparison(
        _attention_states(result, "mla_state_fingerprints"),
        _attention_states(reference_receipt, "mla_state_fingerprints"),
    )
    attnres = _comparison(
        _attnres_states(result), _attnres_states(reference_receipt)
    )
    result["state_reference_comparison"] = {
        "evidence_class": "PHYSICAL independent execution comparison",
        "reference_selection_id": reference_receipt["selection_id"],
        "reference_manifest_sha256": reference_receipt["manifest_sha256"],
        "reference_receipt_sha256": reference_receipt_sha256,
        "comparison_semantics": (
            "direct numerical comparison of every saved KDA, MLA, and AttnRes "
            "array with the independently executed whole-layer control; "
            "fingerprint differences remain visible diagnostics"
        ),
        "fingerprints": {"kda": kda, "mla": mla, "attnres": attnres},
        "numerical_arrays": numerical,
    }
    result["kda_state_fingerprint_equality"] = bool(kda["exact"])
    result["mla_state_fingerprint_equality"] = bool(mla["exact"])
    result["attnres_state_fingerprint_equality"] = bool(attnres["exact"])
    result["state_equality_basis"] = "NUMERICAL_INDEPENDENT_STATE_TRACE"
    result["kda_state_equality"] = bool(numerical["kda"])
    result["mla_state_equality"] = bool(numerical["mla"])
    result["attnres_state_equality"] = bool(numerical["attnres"])
    result["state_equality_basis_valid"] = bool(numerical["valid"])
    failures = strict_manifest_receipt_failures(result)
    result["strict_correctness_failures"] = failures
    result["status"] = "PASS" if not failures else "FAIL"
    return result


def reconcile_original_representatives(*, repo: Path) -> dict[str, Any]:
    """Compare all states with an independently executed whole-layer control."""

    root = repo.resolve()
    correctness = (
        root / "artifacts" / "experiment-022" / "completion" / "correctness"
    )
    paths = {
        index: correctness / f"representative-{index:02d}.json"
        for index in range(1, 6)
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing representative receipts: {missing}")
    receipts = {index: _read(path) for index, path in paths.items()}
    original_hashes = {index: _sha256(path) for index, path in paths.items()}
    frozen = _read(
        root
        / "artifacts"
        / "experiment-022"
        / "completion"
        / "frozen-inputs.json"
    )
    frozen_rows = {
        str(row["selection_id"]): row
        for row in frozen["selected_correctness_manifests"]
    }
    if set(frozen_rows) != {
        f"representative-{index:02d}" for index in range(1, 6)
    }:
        raise ValueError("frozen correctness selections are not exactly representatives 01-05")
    for index, receipt in receipts.items():
        selection_id = f"representative-{index:02d}"
        selected = frozen_rows[selection_id]
        manifest_path = root / selected["path"]
        manifest = _read(manifest_path)
        manifest_sha256 = _sha256(manifest_path)
        receipt["selection_case"] = selected["case"]
        receipt["frozen_selection_match"] = (
            receipt.get("selection_id") == selection_id
            and receipt.get("manifest_sha256") == selected["sha256"]
            and manifest_sha256 == selected["sha256"]
            and receipt.get("inventory_id") == selected["inventory_id"]
            and receipt.get("planner_level") == selected["planner_level"]
        )
        receipt["frozen_selection"] = selected
        _enrich_execution_facts(receipt=receipt, manifest=manifest)
    # Representative 01 and 02 are independently executed whole-layer controls
    # for one another.  Every heterogeneous/mixed receipt is compared with the
    # independently executed representative-01 whole-layer semantic reference.
    reference_for = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
    rows: list[dict[str, Any]] = []
    for index, receipt in receipts.items():
        reference_index = reference_for[index]
        reference = receipts[reference_index]
        kda = _comparison(
            _attention_states(receipt, "kda_state_fingerprints"),
            _attention_states(reference, "kda_state_fingerprints"),
        )
        mla = _comparison(
            _attention_states(receipt, "mla_state_fingerprints"),
            _attention_states(reference, "mla_state_fingerprints"),
        )
        attnres = _comparison(
            _attnres_states(receipt), _attnres_states(reference)
        )
        execution_status = receipt.get(
            "execution_status_before_state_reconciliation", receipt.get("status")
        )
        receipt["execution_status_before_state_reconciliation"] = execution_status
        numerical = _numerical_state_reference(
            receipt,
            expected_reference_hash=frozen_rows["representative-01"]["sha256"],
        )
        use_exact_control = index in (1, 2)
        receipt["state_reference_comparison"] = {
            "evidence_class": "PHYSICAL independent execution comparison",
            "reference_selection_id": reference["selection_id"],
            "reference_manifest_sha256": reference["manifest_sha256"],
            "reference_receipt_sha256_before_reconciliation": original_hashes[
                reference_index
            ],
            "comparison_semantics": (
                "whole-layer controls 01/02 require bitwise layer-indexed state "
                "fingerprints; mixed controls require direct numerical comparison "
                "of every saved KDA, MLA, and AttnRes array with control 01 while "
                "retaining fingerprint differences as visible diagnostics"
            ),
            "fingerprints": {"kda": kda, "mla": mla, "attnres": attnres},
            "numerical_arrays": numerical,
        }
        receipt["kda_state_fingerprint_equality"] = bool(kda["exact"])
        receipt["mla_state_fingerprint_equality"] = bool(mla["exact"])
        receipt["attnres_state_fingerprint_equality"] = bool(attnres["exact"])
        receipt["state_equality_basis"] = (
            "BITWISE_FINGERPRINT"
            if use_exact_control
            else "NUMERICAL_INDEPENDENT_STATE_TRACE"
        )
        receipt["kda_state_equality"] = bool(
            kda["exact"] if use_exact_control else numerical["kda"]
        )
        receipt["mla_state_equality"] = bool(
            mla["exact"] if use_exact_control else numerical["mla"]
        )
        receipt["attnres_state_equality"] = bool(
            attnres["exact"] if use_exact_control else numerical["attnres"]
        )
        receipt["state_equality_basis_valid"] = bool(
            (kda["exact"] and mla["exact"] and attnres["exact"])
            if use_exact_control
            else numerical["valid"]
        )
        failures = strict_manifest_receipt_failures(receipt)
        receipt["strict_correctness_failures"] = failures
        receipt["status"] = "PASS" if not failures else "FAIL"
        atomic_write_json(paths[index], receipt)
        rows.append(
            {
                "selection_id": receipt["selection_id"],
                "status": receipt["status"],
                "reference_selection_id": reference["selection_id"],
                "kda_state_equality": receipt["kda_state_equality"],
                "mla_state_equality": receipt["mla_state_equality"],
                "attnres_state_equality": receipt["attnres_state_equality"],
                "failures": failures,
            }
        )
    result = {
        "schema_version": "experiment-022-completion-state-reconciliation-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "reference_policy": (
            "independent whole-layer control; representative-01 and -02 cross-check "
            "one another, representatives -03/-04/-05 compare with -01"
        ),
        "representative_count": len(rows),
        "rows": rows,
    }
    atomic_write_json(correctness / "state-reconciliation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = reconcile_original_representatives(repo=args.repo)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "reconcile_fresh_manifest_receipt",
    "reconcile_original_representatives",
    "strict_manifest_receipt_failures",
]
