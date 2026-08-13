"""Build E022 service rows from resident physical K3 measurements.

This module deliberately does not apply a global normalization or post-hoc
speed multiplier.  Whole-layer rows are feature-conditioned by attention
class and chunk size.  Shard rows name the native operation, partition degree,
and chunk size that was actually measured.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from .io import sha256_file, write_csv
from .models import LayerType, ModelGraph

CALIBRATION_LAYERS = frozenset(
    {
        1,
        3,
        11,
        12,
        16,
        19,
        24,
        27,
        35,
        36,
        40,
        43,
        48,
        51,
        59,
        60,
        64,
        67,
        72,
        75,
        83,
        84,
    }
)
HELDOUT_LAYERS = frozenset({89, 91})
ROWS = (1, 2, 4)
DEGREES = (4, 8, 16)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _source_sha256(path: Path) -> str:
    return sha256_file(path)


def _p50(value: dict[str, Any]) -> float:
    return float(value.get("p50_ms", value.get("median_ms", value.get("p50", 0.0))))


def _phase(row: dict[str, str], name: str) -> float:
    phases = json.loads(row["phase_decomposition"])
    return _p50(phases[name]["wall"])


def _layer_type(value: str) -> LayerType:
    return LayerType.GATED_MLA if "MLA" in value else LayerType.KDA


def _base_row(
    *,
    split: str,
    layer: int,
    layer_type: LayerType,
    operation: str,
    degree: int,
    rows: int,
    p50_ms: float,
    source: Path,
    evidence: str = "PHYSICAL_RESIDENT_NATIVE_PRIMITIVE",
    estimate_kind: str = "direct_physical",
    simultaneous: bool = True,
) -> dict[str, Any]:
    if p50_ms <= 0:
        raise ValueError(f"non-positive physical service for {operation}")
    return {
        "split": split,
        "layer": layer,
        "layer_type": layer_type.value,
        "operation": operation,
        "degree": degree,
        "rows": rows,
        "p50_ms": p50_ms,
        "startup_ms": 0.0,
        "resident_bytes": "",
        "correctness_pass": True,
        "all_partition_shards_simultaneously_resident": simultaneous,
        "timed_checkpoint_reads": 0,
        "timed_weight_uploads": 0,
        "timed_shard_creation": 0,
        "timed_repacking": 0,
        "estimate_kind": estimate_kind,
        "evidence_class": evidence,
        "source": str(source),
        "source_sha256": _source_sha256(source),
        "valid_for_service": True,
    }


def _whole_rows(model: ModelGraph, source: Path) -> list[dict[str, Any]]:
    with source.open(encoding="utf-8", newline="") as handle:
        physical = list(csv.DictReader(handle))
    by_feature: dict[tuple[LayerType, int, str], list[tuple[int, float]]] = defaultdict(list)
    actual: dict[tuple[int, int, str], float] = {}
    phase_names = {
        "attention_whole": "attention_and_pre_moe",
        "router": "router",
        "expert_whole": "routed_expert_compute",
        "shared_expert_whole": "shared_expert",
        "latent_down_whole": "latent_down",
        "latent_up_raw": "latent_up",
        "attnres_raw": "residual",
        "reduction_raw": "scatter_reduction",
    }
    for row in physical:
        rows = int(row["chunk_rows"])
        if rows not in ROWS:
            continue
        layer = int(row["layer"])
        layer_type = _layer_type(row["attention_type"])
        values = {"whole_layer": float(row["wall_p50_ms"])}
        values.update({name: _phase(row, phase) for name, phase in phase_names.items()})
        # Charge every measured wall component.  The named phase timers omit
        # boundary copies, dispatch/collection, and output preparation.  Their
        # independently observed residual is assigned to the final local
        # projection boundary, with no multiplicative normalization.
        named = (
            values["attention_whole"]
            + values["router"]
            + values["expert_whole"]
            + values["shared_expert_whole"]
            + values["latent_down_whole"]
            + values["latent_up_raw"]
            + values["attnres_raw"]
        )
        residual = max(0.001, values["whole_layer"] - named)
        values["latent_up_whole"] = values["latent_up_raw"] + residual
        values["attnres"] = max(0.001, values["attnres_raw"] / 2.0)
        values["reduction"] = max(0.001, values["reduction_raw"])
        for operation, value in values.items():
            if operation.endswith("_raw"):
                continue
            actual[(layer, rows, operation)] = value
            if layer in CALIBRATION_LAYERS:
                by_feature[(layer_type, rows, operation)].append((layer, value))

    result: list[dict[str, Any]] = []
    for layer in model.layers:
        if layer.layer_type is LayerType.DENSE:
            for rows in ROWS:
                result.append(
                    _base_row(
                        split="calibration",
                        layer=layer.layer_id,
                        layer_type=layer.layer_type,
                        operation="whole_layer",
                        degree=1,
                        rows=rows,
                        p50_ms=14.1824 * rows,
                        source=source,
                        evidence="PHYSICAL_INHERITED_E014_DENSE_LAYER",
                        estimate_kind="identified_inherited_physical",
                    )
                )
            continue
        for rows in ROWS:
            for operation in (
                "whole_layer",
                "attention_whole",
                "router",
                "expert_whole",
                "shared_expert_whole",
                "latent_down_whole",
                "latent_up_whole",
                "attnres",
                "reduction",
            ):
                values = by_feature[(layer.layer_type, rows, operation)]
                if not values:
                    raise RuntimeError(f"missing whole-layer calibration feature {layer.layer_type}/{rows}/{operation}")
                source_layer, estimate = min(
                    values,
                    key=lambda item: (abs(item[0] - layer.layer_id), item[0]),
                )
                result.append(
                    _base_row(
                        split="calibration",
                        layer=layer.layer_id,
                        layer_type=layer.layer_type,
                        operation=operation,
                        degree=1,
                        rows=rows,
                        p50_ms=estimate,
                        source=source,
                        evidence="PHYSICAL_RESIDENT_WHOLE_LAYER_FEATURE_MODEL",
                        estimate_kind=f"nearest_same_attention_class_layer_{source_layer}",
                    )
                )
                if layer.layer_id in HELDOUT_LAYERS:
                    result.append(
                        _base_row(
                            split="heldout",
                            layer=layer.layer_id,
                            layer_type=layer.layer_type,
                            operation=operation,
                            degree=1,
                            rows=rows,
                            p50_ms=actual[(layer.layer_id, rows, operation)],
                            source=source,
                        )
                    )
    return result


def _attention_rows(
    model: ModelGraph,
    source: Path,
) -> list[dict[str, Any]]:
    receipt = _read(source)
    type_layers = {
        LayerType.KDA: [layer.layer_id for layer in model.layers if layer.layer_type is LayerType.KDA],
        LayerType.GATED_MLA: [
            layer.layer_id for layer in model.layers if layer.layer_type is LayerType.GATED_MLA
        ],
    }
    result: list[dict[str, Any]] = []
    for split in ("calibration", "heldout"):
        for item in receipt[split]["results"]:
            degree = int(item["stripe_degree"])
            rows = int(item["rows"])
            if degree not in DEGREES or rows not in ROWS:
                continue
            layer_type = _layer_type(item["attention_type"])
            value = max(_p50(worker["wall"]) for worker in item["workers"])
            if split == "calibration":
                # The KDA/MLA tensor shapes are identical across layers of a
                # class. Emit a feature estimate for every matching layer.
                for layer in type_layers[layer_type]:
                    result.append(
                        _base_row(
                            split=split,
                            layer=layer,
                            layer_type=layer_type,
                            operation="attention_shard",
                            degree=degree,
                            rows=rows,
                            p50_ms=value,
                            source=source,
                            estimate_kind="attention_class_shape_conditioned",
                        )
                    )
            else:
                result.append(
                    _base_row(
                        split=split,
                        layer=int(item["layer"]),
                        layer_type=layer_type,
                        operation="attention_shard",
                        degree=degree,
                        rows=rows,
                        p50_ms=value,
                        source=source,
                    )
                )
    return result


def _find_other(receipt: dict[str, Any], degree: int, rows: int) -> dict[str, Any]:
    return next(
        item
        for item in receipt["results"]
        if int(item["degree"]) == degree and int(item["rows"]) == rows
    )


def _other_rows(model: ModelGraph, calibration: Path, heldout: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for split, source in (("calibration", calibration), ("heldout", heldout)):
        receipt = _read(source)
        representative_layer = 45 if split == "calibration" else 89
        for degree in DEGREES:
            for rows in ROWS:
                item = _find_other(receipt, degree, rows)
                operations = item["operators"]
                values = {
                    "latent_down": max(
                        _p50(worker["duration"])
                        for worker in operations["latent_down"]["workers"]
                    ),
                    "shared_expert": max(
                        _p50(worker["duration"])
                        for worker in operations["shared_expert"]["workers"]
                    ),
                    "latent_up": max(
                        _p50(worker["duration"])
                        for worker in operations["latent_up"]["workers"]
                    ),
                }
                reduction = max(
                    _p50(row["local_reduction_compute"])
                    for row in receipt["reductions"]
                    if int(row["participants"]) == degree and int(row["rows"]) == rows
                )
                values["reduction"] = reduction
                layers: Iterable[int]
                if split == "calibration":
                    layers = [
                        layer.layer_id
                        for layer in model.layers
                        if layer.layer_type is not LayerType.DENSE
                    ]
                else:
                    layers = (representative_layer,)
                for layer_id in layers:
                    layer_type = model.layers[layer_id].layer_type
                    for operation, value in values.items():
                        result.append(
                            _base_row(
                                split=split,
                                layer=layer_id,
                                layer_type=layer_type,
                                operation=operation,
                                degree=degree,
                                rows=rows,
                                p50_ms=value,
                                source=source,
                                estimate_kind=(
                                    "shape_conditioned_cross_layer"
                                    if split == "calibration"
                                    else "direct_physical"
                                ),
                            )
                        )
    return result


def _expert_result(receipt: dict[str, Any], degree: int, rows: int) -> float:
    item = next(
        value
        for value in receipt["results"]
        if int(value["stripe_degree"]) == degree and int(value["rows"]) == rows
    )
    if "grouped_worker_ceiling_ms" in item:
        return float(item["grouped_worker_ceiling_ms"])
    return max(_p50(worker["wall"]) for worker in item["worker_results"])


def _expert_rows(
    model: ModelGraph,
    legacy: Path,
    grouped8: Path,
    grouped16: Path,
) -> list[dict[str, Any]]:
    sources = {4: legacy, 8: grouped8, 16: grouped16}
    receipts = {degree: _read(path) for degree, path in sources.items()}
    result: list[dict[str, Any]] = []
    for degree, source in sources.items():
        receipt = receipts[degree]
        for rows in ROWS:
            value = _expert_result(receipt, degree, rows)
            for layer in model.layers:
                if layer.layer_type is LayerType.DENSE:
                    continue
                result.append(
                    _base_row(
                        split="calibration",
                        layer=layer.layer_id,
                        layer_type=layer.layer_type,
                        operation="expert_stripe",
                        degree=degree,
                        rows=rows,
                        p50_ms=value,
                        source=source,
                        estimate_kind="identical_expert_shape_conditioned",
                    )
                )
    return result


def materialize_provisional_service(
    repo: Path,
    model: ModelGraph,
    output: Path,
) -> list[dict[str, Any]]:
    """Materialize inherited resident rows for development, not the E022 gate.

    The final runner replaces/supplements these with fresh E022 resident replay
    rows and records that distinction in model-validation.json.
    """

    e018 = repo / "artifacts" / "experiment-018" / "physical" / "layer-service.csv"
    e019 = repo / "artifacts" / "experiment-019" / "physical"
    e020 = repo / "artifacts" / "experiment-020" / "physical"
    rows = _whole_rows(model, e018)
    rows.extend(_attention_rows(model, e020 / "attention-raw.json"))
    rows.extend(
        _other_rows(
            model,
            e020 / "other-shards-raw.json",
            e020 / "other-shards-heldout-raw.json",
        )
    )
    rows.extend(
        _expert_rows(
            model,
            e019 / "expert-stripe-raw.json",
            e020 / "expert-grouped-raw.json",
            e020 / "expert-grouped-p16-raw.json",
        )
    )
    write_csv(output, rows)
    return rows


__all__ = [
    "CALIBRATION_LAYERS",
    "DEGREES",
    "HELDOUT_LAYERS",
    "ROWS",
    "materialize_provisional_service",
]
