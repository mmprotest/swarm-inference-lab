from __future__ import annotations

import ast
import hashlib
import importlib
import types
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = REPOSITORY_ROOT / "benchmarks" / "canonical"
PRODUCT_ROOT = REPOSITORY_ROOT / "src" / "swarm_inference"
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


def _resolve_owner(owner: str) -> tuple[types.ModuleType | None, object]:
    if owner.startswith("benchmarks.canonical."):
        relative = owner.removeprefix("benchmarks.canonical.") + ".yaml"
        evidence_path = CANONICAL_ROOT / relative
        assert evidence_path.is_file(), f"canonical evidence owner does not exist: {owner}"
        return None, evidence_path

    parts = owner.split(".")
    for module_length in range(len(parts), 0, -1):
        module_name = ".".join(parts[:module_length])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                continue
            raise
        resolved: object = module
        for attribute in parts[module_length:]:
            assert hasattr(resolved, attribute), f"canonical owner does not exist: {owner}"
            resolved = getattr(resolved, attribute)
        return module, resolved
    raise AssertionError(f"canonical owner does not exist: {owner}")


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
        module, resolved = _resolve_owner(owner)
        assert resolved is not None
        if row["disposition"] != "EVIDENCE_ONLY":
            assert owner.startswith("swarm_inference.")
            assert not owner.startswith("swarm_inference.experiments")
            assert module is not None and module.__file__ is not None
            module_path = Path(module.__file__).resolve()
            assert module_path.is_relative_to(PRODUCT_ROOT)
            assert "experiments" not in module_path.relative_to(PRODUCT_ROOT).parts

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


def test_contract_evidence_has_durable_provenance_and_matches_local_archive() -> None:
    provenance = _document(CANONICAL_ROOT / "evidence_provenance.yaml")
    assert provenance["schema_version"] == 1
    assert provenance["hash_algorithm"] == "sha256"
    bundles = provenance["bundles"]
    assert isinstance(bundles, dict)
    assert set(bundles) == {f"{number:03d}" for number in range(1, 12)} - {"005"}

    for number in range(1, 12):
        experiment = f"{number:03d}"
        contract = _document(CANONICAL_ROOT / f"experiment_{experiment}_contract.yaml")
        identity = contract["evidence_bundle_identity"]
        if experiment == "005":
            assert identity is None
            continue
        record = bundles[experiment]
        assert record["source_experiment"] == contract["source_experiment"]
        assert record["identity"] == identity
        assert isinstance(record["verdict"], str) and record["verdict"]
        anchors = record["anchors"]
        assert isinstance(anchors, dict) and anchors
        for relative, expected in anchors.items():
            assert isinstance(relative, str) and relative
            assert isinstance(expected, str) and len(expected) == 64
            int(expected, 16)

        archive = REPOSITORY_ROOT / identity
        if archive.exists():
            assert archive.is_dir()
            for relative, expected in anchors.items():
                anchored_file = archive / relative
                assert anchored_file.is_file(), f"missing evidence anchor: {anchored_file}"
                assert hashlib.sha256(anchored_file.read_bytes()).hexdigest() == expected


def test_product_modules_do_not_import_experiment_implementations() -> None:
    violations: list[str] = []
    for path in PRODUCT_ROOT.rglob("*.py"):
        relative = path.relative_to(PRODUCT_ROOT)
        if relative.parts and relative.parts[0] == "experiments":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = (node.module,)
            for module_name in imported:
                if module_name == "swarm_inference.experiments" or module_name.startswith(
                    "swarm_inference.experiments."
                ):
                    violations.append(f"{relative.as_posix()}:{node.lineno}:{module_name}")
    assert not violations, "product modules import experiment implementations:\n" + "\n".join(
        violations
    )
