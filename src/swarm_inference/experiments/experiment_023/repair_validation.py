"""Read-only validation of the immutable E023 repair inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import (
    canonical_sha256,
    sha256_file,
)

from .freeze import (
    CAPACITY_INVENTORIES,
    CONTROL_INVENTORIES,
    EXPECTED_E022_SUITE_DIGEST,
    FROZEN_CONSTANTS,
    HEADLINE_INVENTORIES,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"MODEL_INVALID: expected JSON object in {path}")
    return value


def validate_repair_inputs_read_only(repo: Path) -> dict[str, Any]:
    """Validate the E023/E022/physical preregistration without writing it."""

    root = repo.resolve()
    experiment = root / "artifacts" / "experiment-023"
    freeze_root = experiment / "freeze"
    frozen = _read_object(freeze_root / "e023-frozen-inputs.json")
    constants_hash = canonical_sha256(FROZEN_CONSTANTS)
    if frozen.get("constants") != FROZEN_CONSTANTS:
        raise RuntimeError("MODEL_INVALID: frozen E023 constants changed")
    if frozen.get("constants_sha256") != constants_hash:
        raise RuntimeError("MODEL_INVALID: frozen E023 constants hash changed")
    if frozen.get("e022_inventory_suite_sha256") != EXPECTED_E022_SUITE_DIGEST:
        raise RuntimeError("MODEL_INVALID: frozen E022 inventory suite changed")

    cohorts = _read_object(freeze_root / "headline-cohorts.json")
    expected_cohorts = {
        "headline": list(HEADLINE_INVENTORIES),
        "negative_controls": list(CONTROL_INVENTORIES),
        "capacity_exploratory": list(CAPACITY_INVENTORIES),
    }
    for name, expected in expected_cohorts.items():
        if cohorts.get(name) != expected:
            raise RuntimeError(f"MODEL_INVALID: frozen {name} cohort changed")

    e022 = _read_object(freeze_root / "e022-input-hashes.json")
    mismatches: list[str] = []
    for row in e022.get("files", ()):
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            mismatches.append(str(row["path"]))
    if mismatches:
        raise RuntimeError(
            f"MODEL_INVALID: {len(mismatches)} immutable E022 inputs changed"
        )
    if e022.get("canonical_sha256") != frozen.get("e022_input_hashes_sha256"):
        raise RuntimeError("MODEL_INVALID: frozen E022 input manifest hash changed")

    protocol = _read_object(experiment / "repair" / "repair-protocol.json")
    if protocol.get("original_verdict") != "MODEL_INVALID":
        raise RuntimeError("MODEL_INVALID: original E023 verdict was rewritten")
    if protocol.get("source_attempt") != "deterministic-run-1":
        raise RuntimeError("MODEL_INVALID: original E023 attempt identity changed")
    if protocol.get("frozen_constants") != FROZEN_CONSTANTS:
        raise RuntimeError("MODEL_INVALID: repair protocol changed frozen constants")
    if (
        protocol.get("frozen_constants_canonical_sha256_from_run_1")
        != constants_hash
    ):
        raise RuntimeError("MODEL_INVALID: repair protocol constants hash changed")

    physical_mismatches: list[str] = []
    physical = protocol.get("physical_validation_reuse", {})
    for row in physical.get("files", ()):
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != str(row["sha256"])
        ):
            physical_mismatches.append(str(row["path"]))
    if physical_mismatches:
        raise RuntimeError(
            f"MODEL_INVALID: {len(physical_mismatches)} frozen physical artifacts changed"
        )
    physical_receipt = _read_object(
        experiment / "physical" / "duplicate-expert-group-correctness.json"
    )
    if (
        physical_receipt.get("status") != "PASS"
        or physical_receipt.get("primary_gate") != "PASS"
        or int(physical_receipt.get("case_count", -1)) != 12
    ):
        raise RuntimeError("MODEL_INVALID: frozen physical replica gate is not PASS")

    return {
        "status": "PASS",
        "constants_sha256": constants_hash,
        "e022_file_count": int(e022["file_count"]),
        "e022_input_hashes_sha256": str(e022["canonical_sha256"]),
        "e022_inventory_suite_sha256": EXPECTED_E022_SUITE_DIGEST,
        "physical_artifact_count": len(physical.get("files", ())),
        "physical_status": "PASS",
    }


__all__ = ["validate_repair_inputs_read_only"]
