from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = REPOSITORY_ROOT / "benchmarks" / "canonical"
DISPOSITIONS = {
    "REQUIRED",
    "AVAILABLE_CONDITIONAL",
    "REJECTED_DEFAULT",
    "EVIDENCE_ONLY",
}
CONTRACT_FIELDS = {
    "contract_version",
    "source_experiment",
    "status",
    "evidence_bundle_identity",
    "promoted_mechanisms",
    "rejected_mechanisms",
    "reference_models",
    "reference_hardware_class",
    "correctness_gates",
    "performance_gates",
    "planner_gates",
    "required_telemetry",
}


def _document(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"{path.name} must contain one mapping"
    return value


def test_every_numbered_experiment_has_one_canonical_contract() -> None:
    expected = {f"experiment_{number:03d}_contract.yaml" for number in range(1, 12)}
    actual = {path.name for path in CANONICAL_ROOT.glob("experiment_*_contract.yaml")}
    assert actual == expected

    for number in range(1, 12):
        path = CANONICAL_ROOT / f"experiment_{number:03d}_contract.yaml"
        contract = _document(path)
        assert set(contract) == CONTRACT_FIELDS
        expected_source = "007-corrected" if number == 7 else f"{number:03d}"
        assert contract["source_experiment"] == expected_source
        for field in CONTRACT_FIELDS.difference(
            {"contract_version", "source_experiment", "status", "evidence_bundle_identity"}
        ):
            if field == "reference_hardware_class":
                assert isinstance(contract[field], (str, list))
            else:
                assert isinstance(contract[field], list), f"{path.name}: {field} must be a list"


def test_experiment_005_contract_makes_no_capability_claim() -> None:
    contract = _document(CANONICAL_ROOT / "experiment_005_contract.yaml")
    assert contract["status"] == "NO_COMPLETED_EXPERIMENT"
    assert contract["evidence_bundle_identity"] is None
    assert contract["promoted_mechanisms"] == []
    assert contract["rejected_mechanisms"] == []
    assert contract["correctness_gates"] == []
    assert contract["performance_gates"] == []
    assert contract["required_telemetry"] == []


def test_promotion_ledger_covers_contract_mechanisms_and_importable_product_owners() -> None:
    manifest = _document(CANONICAL_ROOT / "promotion_manifest.yaml")
    assert set(manifest["allowed_dispositions"]) == DISPOSITIONS
    assert set(manifest["experiments"]) == {f"{number:03d}" for number in range(1, 12)}
    mechanisms = manifest["mechanisms"]
    assert isinstance(mechanisms, list) and mechanisms
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in mechanisms:
        assert isinstance(row, dict)
        assert {
            "experiment",
            "mechanism",
            "disposition",
            "canonical_owner",
            "release_gate",
        }.issubset(row)
        assert row["disposition"] in DISPOSITIONS
        key = (str(row["experiment"]), str(row["mechanism"]))
        assert key not in by_key, f"duplicate promotion disposition: {key}"
        by_key[key] = row
        owner = str(row["canonical_owner"])
        if owner.startswith("swarm_inference.") and row["disposition"] != "EVIDENCE_ONLY":
            importlib.import_module(owner)

    for number in range(1, 12):
        experiment = f"{number:03d}"
        contract = _document(CANONICAL_ROOT / f"experiment_{experiment}_contract.yaml")
        for mechanism in contract["promoted_mechanisms"]:
            # A result can be promoted into the durable regression corpus while
            # remaining evidence-only (for example the calibrated simulator).
            assert by_key[(experiment, mechanism)]["disposition"] != "REJECTED_DEFAULT"
        for mechanism in contract["rejected_mechanisms"]:
            assert by_key[(experiment, mechanism)]["disposition"] in {
                "REJECTED_DEFAULT",
                "EVIDENCE_ONLY",
            }
