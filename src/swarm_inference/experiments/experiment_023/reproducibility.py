"""Deterministic run-2/run-3 comparison for Experiment 023."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import atomic_write_json

from .freeze import CAPACITY_INVENTORIES, CONTROL_INVENTORIES, HEADLINE_INVENTORIES
from .models import Arm

INVENTORY_IDS = HEADLINE_INVENTORIES + CONTROL_INVENTORIES + CAPACITY_INVENTORIES
ARMS = tuple(value.value for value in Arm)
INTEGER_PATTERN = re.compile(r"^-?\d+$")
FLOAT_RELATIVE_TOLERANCE = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _compare_scalar(
    left: Any,
    right: Any,
    *,
    path: str,
    failures: list[str],
    maximum_relative: list[float],
) -> None:
    if left == right:
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            failures.append(f"INTEGER_MISMATCH:{path}:{left}!={right}")
            return
        relative = _relative_difference(float(left), float(right))
        maximum_relative[0] = max(maximum_relative[0], relative)
        if relative > FLOAT_RELATIVE_TOLERANCE:
            failures.append(f"FLOAT_MISMATCH:{path}:{relative:.17g}")
        return
    failures.append(f"VALUE_MISMATCH:{path}:{left!r}!={right!r}")


def _compare_structure(
    left: Any,
    right: Any,
    *,
    path: str,
    failures: list[str],
    maximum_relative: list[float],
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            failures.append(f"KEY_MISMATCH:{path}")
            return
        for key in sorted(left):
            _compare_structure(
                left[key],
                right[key],
                path=f"{path}.{key}",
                failures=failures,
                maximum_relative=maximum_relative,
            )
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(f"LENGTH_MISMATCH:{path}:{len(left)}!={len(right)}")
            return
        for index, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            _compare_structure(
                left_value,
                right_value,
                path=f"{path}[{index}]",
                failures=failures,
                maximum_relative=maximum_relative,
            )
        return
    _compare_scalar(
        left,
        right,
        path=path,
        failures=failures,
        maximum_relative=maximum_relative,
    )


def _compare_csv_file(
    left_path: Path,
    right_path: Path,
    *,
    failures: list[str],
    maximum_relative: list[float],
) -> None:
    left_fields, left_rows = _read_csv(left_path)
    right_fields, right_rows = _read_csv(right_path)
    label = left_path.relative_to(left_path.parents[2]).as_posix()
    if left_fields != right_fields:
        failures.append(f"CSV_SCHEMA_MISMATCH:{label}")
        return
    if len(left_rows) != len(right_rows):
        failures.append(
            f"CSV_ROW_COUNT_MISMATCH:{label}:{len(left_rows)}!={len(right_rows)}"
        )
        return
    for row_index, (left, right) in enumerate(zip(left_rows, right_rows, strict=True)):
        for field in left_fields:
            left_value = left[field]
            right_value = right[field]
            if left_value == right_value:
                continue
            path = f"{label}[{row_index}].{field}"
            if INTEGER_PATTERN.fullmatch(left_value) and INTEGER_PATTERN.fullmatch(
                right_value
            ):
                failures.append(
                    f"INTEGER_MISMATCH:{path}:{left_value}!={right_value}"
                )
                continue
            try:
                left_float = float(left_value)
                right_float = float(right_value)
            except ValueError:
                failures.append(
                    f"VALUE_MISMATCH:{path}:{left_value!r}!={right_value!r}"
                )
                continue
            if not math.isfinite(left_float) or not math.isfinite(right_float):
                failures.append(f"NONFINITE_VALUE:{path}")
                continue
            relative = _relative_difference(left_float, right_float)
            maximum_relative[0] = max(maximum_relative[0], relative)
            if relative > FLOAT_RELATIVE_TOLERANCE:
                failures.append(f"FLOAT_MISMATCH:{path}:{relative:.17g}")


def compare_deterministic_attempts(
    repo: Path,
    *,
    primary_attempt: str = "deterministic-run-2",
    reproducibility_attempt: str = "deterministic-run-3",
) -> dict[str, Any]:
    """Require deterministic equality across independently planned attempts."""

    root = repo.resolve()
    attempts = root / "artifacts/experiment-023/attempts"
    left_root = attempts / primary_attempt
    right_root = attempts / reproducibility_attempt
    failures: list[str] = []
    maximum_relative = [0.0]
    plan_hash_matches = 0
    u_strong_hash_matches = 0
    assignment_matches = 0
    replica_matches = 0
    action_sequence_matches = 0
    envelope_source_matches = 0
    for inventory_id in INVENTORY_IDS:
        for arm in ARMS:
            left = _read_json(left_root / "plans" / inventory_id / f"{arm}.json")
            right = _read_json(right_root / "plans" / inventory_id / f"{arm}.json")
            label = f"plans.{inventory_id}.{arm}"
            if left["canonical_plan_sha256"] == right["canonical_plan_sha256"]:
                plan_hash_matches += 1
                if arm == Arm.U_STRONG.value:
                    u_strong_hash_matches += 1
            else:
                failures.append(f"CANONICAL_PLAN_HASH_MISMATCH:{label}")
            left_assignments = [
                piece
                for node in left["nodes"]
                for piece in node["pieces"]
                if str(piece["piece"]).startswith("transformer_layer_")
            ]
            right_assignments = [
                piece
                for node in right["nodes"]
                for piece in node["pieces"]
                if str(piece["piece"]).startswith("transformer_layer_")
            ]
            before = len(failures)
            _compare_structure(
                left_assignments,
                right_assignments,
                path=f"{label}.primary_assignments",
                failures=failures,
                maximum_relative=maximum_relative,
            )
            assignment_matches += len(failures) == before
            before = len(failures)
            _compare_structure(
                left.get("replicas", []),
                right.get("replicas", []),
                path=f"{label}.replicas",
                failures=failures,
                maximum_relative=maximum_relative,
            )
            replica_matches += len(failures) == before
            before = len(failures)
            _compare_structure(
                left.get("planner_actions", []),
                right.get("planner_actions", []),
                path=f"{label}.planner_actions",
                failures=failures,
                maximum_relative=maximum_relative,
            )
            action_sequence_matches += len(failures) == before
            if arm == Arm.FLEX_POOL.value:
                left_source = left.get("metadata", {}).get(
                    "selected_final_flex_pool_envelope_source"
                )
                right_source = right.get("metadata", {}).get(
                    "selected_final_flex_pool_envelope_source"
                )
                if left_source == right_source:
                    envelope_source_matches += 1
                else:
                    failures.append(f"ENVELOPE_SOURCE_MISMATCH:{label}")

    csv_paths = (
        "baseline/baseline-envelope.csv",
        "baseline/baseline-relocations.csv",
        "baseline/unique-refinement-actions.csv",
        "serving/replica-actions.csv",
        "serving/arm-results.csv",
        "serving/saturation-summary.csv",
        "serving/replica-routing-summary.csv",
        "serving/resource-utilization.csv",
        "serving/network-summary.csv",
        "validation/memory-reconciliation.csv",
        "validation/cost-reconciliation.csv",
        "validation/flex-pool-superset.csv",
        "validation/flex-pool-search-coverage.csv",
    )
    for relative in csv_paths:
        _compare_csv_file(
            left_root / relative,
            right_root / relative,
            failures=failures,
            maximum_relative=maximum_relative,
        )

    left_summary = _read_json(left_root / "attempt-summary.json")
    right_summary = _read_json(right_root / "attempt-summary.json")
    receipt = {
        "schema_version": "experiment-023-reproducibility-v2",
        "experiment_id": "023",
        "status": "PASS" if not failures else "FAIL",
        "primary_attempt": primary_attempt,
        "reproducibility_attempt": reproducibility_attempt,
        "inventory_count": len(INVENTORY_IDS),
        "canonical_plan_hash_matches": plan_hash_matches,
        "canonical_plan_hash_expected": len(INVENTORY_IDS) * len(ARMS),
        "u_strong_hash_matches": u_strong_hash_matches,
        "u_strong_hash_expected": len(INVENTORY_IDS),
        "primary_assignment_matches": assignment_matches,
        "replicated_group_and_alternate_matches": replica_matches,
        "accepted_action_sequence_matches": action_sequence_matches,
        "selected_flex_pool_envelope_source_matches": envelope_source_matches,
        "integer_and_replica_selection_counters_exact": not any(
            failure.startswith("INTEGER_MISMATCH") for failure in failures
        ),
        "maximum_float_relative_difference": maximum_relative[0],
        "float_relative_tolerance": FLOAT_RELATIVE_TOLERANCE,
        "primary_wall_seconds": float(left_summary["elapsed_seconds"]),
        "reproducibility_wall_seconds": float(right_summary["elapsed_seconds"]),
        "wall_clock_runtime_compared": False,
        "failures": failures,
    }
    atomic_write_json(
        root / "artifacts/experiment-023/validation/reproducibility.json",
        receipt,
    )
    if failures:
        raise RuntimeError(
            f"MODEL_INVALID: deterministic reproducibility failed ({len(failures)} mismatches)"
        )
    return receipt


__all__ = ["compare_deterministic_attempts"]
