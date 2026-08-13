"""Materialize resident service, ordered-DAG, and held-out validation gates."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import _attention_rows, _base_row, materialize_provisional_service
from .event_model import DeterministicWorkerEventEngine, EventTask
from .io import atomic_write_json, write_csv
from .models import LayerType, ModelGraph
from .service import ResidentServiceModel


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percent_error(predicted: float, actual: float) -> float:
    return abs(predicted - actual) / actual * 100.0


def _percentiles(values: Iterable[float]) -> dict[str, float | None]:
    rows = list(values)
    if not rows:
        return {"median_percent": None, "p90_percent": None, "maximum_percent": None}
    return {
        "median_percent": float(np.percentile(rows, 50)),
        "p90_percent": float(np.percentile(rows, 90)),
        "maximum_percent": max(rows),
    }


def _fresh_whole_service_rows(
    source: Path,
    model: ModelGraph,
) -> list[dict[str, Any]]:
    receipt = _read(source)
    phase_names = {
        "attention_whole": "attention_and_pre_moe",
        "router": "router",
        "expert_whole": "routed_expert_compute",
        "shared_expert_whole": "shared_expert",
        "latent_down_whole": "latent_down",
        "latent_up_raw": "latent_up",
        "attnres_raw": "residual",
        "reduction": "scatter_reduction",
    }
    rows: list[dict[str, Any]] = []
    for layer_text, layer_receipt in receipt["layers"].items():
        layer = int(layer_text)
        split = "calibration" if layer in (45, 47) else "heldout"
        layer_type = model.layers[layer].layer_type
        for row_text, value in layer_receipt["service"].items():
            row_count = int(row_text)
            if row_count not in (1, 2, 4):
                continue
            services = {"whole_layer": float(value["wall"]["p50_ms"])}
            services.update(
                {
                    operation: float(value["phase_decomposition"][phase]["wall"]["p50_ms"])
                    for operation, phase in phase_names.items()
                }
            )
            named = sum(
                services[name]
                for name in (
                    "attention_whole",
                    "router",
                    "expert_whole",
                    "shared_expert_whole",
                    "latent_down_whole",
                    "latent_up_raw",
                    "attnres_raw",
                )
            )
            services["latent_up_whole"] = services["latent_up_raw"] + max(
                0.001, services["whole_layer"] - named
            )
            services["attnres"] = max(0.001, services["attnres_raw"] / 2)
            for operation, duration in services.items():
                if operation.endswith("_raw"):
                    continue
                row = _base_row(
                    split=split,
                    layer=layer,
                    layer_type=layer_type,
                    operation=operation,
                    degree=1,
                    rows=row_count,
                    p50_ms=duration,
                    source=source,
                    evidence="PHYSICAL_E022_RESIDENT_WHOLE_LAYER",
                )
                row["startup_ms"] = float(layer_receipt["load"]["wall_ms"]) + float(
                    layer_receipt["load"]["prepare_wall_ms"]
                )
                row["resident_bytes"] = int(layer_receipt["load"]["resident_device_bytes"])
                rows.append(row)
    return rows


def _ordered_operation(row: dict[str, Any]) -> str:
    operator = str(row.get("operator"))
    if operator.endswith("_attention_stripe"):
        return "attention_shard"
    return {
        "attention_attnres_mix": "attnres",
        "mlp_attnres_mix": "attnres",
        "router": "router",
        "latent_down_projection_stripe": "latent_down",
        "grouped_expert_stripe_bank_top16": "expert_stripe",
        "latent_up_projection_stripe": "latent_up",
        "shared_expert_stripe": "shared_expert",
    }[operator]


def _ordered_models(
    source: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[LayerType, float],
    dict[LayerType, dict[str, float]],
]:
    receipt = _read(source)
    results = receipt["results"]
    calibration = {
        LayerType.KDA: next(row for row in results if int(row["layer"]) == 45),
        LayerType.GATED_MLA: next(row for row in results if int(row["layer"]) == 47),
    }
    heldout = [row for row in results if row["split"] == "heldout"]
    barrier_by_type: dict[LayerType, float] = {}
    models: dict[LayerType, dict[str, float]] = {}
    for layer_type, row in calibration.items():
        by_key: dict[str, list[float]] = defaultdict(list)
        for iteration in row["operation_records"]:
            for item in iteration:
                by_key[_ordered_operation(item)].append(
                    float(item.get("duration_ms", 0))
                )
        models[layer_type] = {
            key: statistics.median(values) for key, values in by_key.items()
        }
        counts = {
            operation: sum(
                _ordered_operation(item) == operation
                for item in row["operation_records"][-1]
            )
            for operation in models[layer_type]
        }
        modeled_native = sum(
            counts[operation] * service
            for operation, service in models[layer_type].items()
        )
        # The synchronized outer wall contains five exact phase barriers not
        # represented by the asynchronous native-operation wall receipts:
        # attention, routed experts, shared expert, latent-up, and final sum.
        # Charge that physically measured residual to those five concrete DAG
        # barriers. This is a measured decomposition, not a normalization.
        barrier_by_type[layer_type] = max(
            0.001,
            (float(row["wall"]["p50_ms"]) - modeled_native) / 5,
        )

    validation_rows: list[dict[str, Any]] = []
    engine = DeterministicWorkerEventEngine()
    for row in heldout:
        layer_type = LayerType.KDA if row["attention_type"] == "KDA" else LayerType.GATED_MLA
        iteration = row["operation_records"][-1]
        tasks: list[EventTask] = []
        prior: str | None = None
        for index, item in enumerate(iteration):
            operation = _ordered_operation(item)
            service = models[layer_type][operation]
            identifier = f"layer-{row['layer']}.ordered-{index:03d}"
            tasks.append(
                EventTask(
                    task_id=identifier,
                    resource_id="compute:one-physical-rtx5090",
                    dependency_ids=(prior,) if prior else (),
                    duration_ms=service,
                    category="compute",
                    node_id="one-physical-rtx5090",
                    layer_id=int(row["layer"]),
                    chunk_id=0,
                    operation=operation,
                )
            )
            prior = identifier
        for index in range(5):
            identifier = f"layer-{row['layer']}.barrier-{index:02d}"
            tasks.append(
                EventTask(
                    task_id=identifier,
                    resource_id="compute:one-physical-rtx5090",
                    dependency_ids=(prior,) if prior else (),
                    duration_ms=barrier_by_type[layer_type],
                    category="compute",
                    node_id="one-physical-rtx5090",
                    layer_id=int(row["layer"]),
                    chunk_id=0,
                    operation="ordered_phase_reduction_barrier",
                )
            )
            prior = identifier
        run = engine.run(tasks)
        actual = float(row["wall"]["p50_ms"])
        error = _percent_error(run.makespan_ms, actual)
        validation_rows.append(
            {
                "workload": f"heldout-layer-{int(row['layer']):02d}-p8-r1",
                "split": "heldout",
                "layer": int(row["layer"]),
                "layer_type": layer_type.value,
                "degree": int(row["degree"]),
                "rows": int(row["rows"]),
                "task_count": len(tasks),
                "native_operation_count": len(iteration),
                "ordered_reduction_barriers": 5,
                "forced_compute_resources": 1,
                "predicted_ordered_wall_ms": run.makespan_ms,
                "measured_ordered_wall_ms": actual,
                "absolute_error_percent": error,
                "calibration_reduction_barrier_ms": barrier_by_type[layer_type],
                "normalization_applied": False,
                "post_hoc_multiplier": "",
                "timed_checkpoint_reads": int(row["timed_checkpoint_reads"]),
                "timed_weight_uploads": int(row["timed_weight_uploads"]),
                "timed_buffer_allocations": int(row["timed_buffer_allocations"]),
                "correctness_relative_l2": float(row["correctness"]["relative_l2_error"]),
                "status": "PASS" if error <= 15.0 else "FAIL",
                "evidence_class": row["evidence_class"],
            }
        )
    return validation_rows, barrier_by_type, models


def _fresh_ordered_service_rows(
    source: Path,
    model: ModelGraph,
    existing_rows: list[dict[str, Any]],
    barrier_by_type: dict[LayerType, float],
    ordered_services: dict[LayerType, dict[str, float]],
) -> list[dict[str, Any]]:
    """Anchor headline row-1 service to the resident ordered implementation."""

    def prior_median(layer_type: LayerType, operation: str, degree: int) -> float | None:
        values = [
            float(row["p50_ms"])
            for row in existing_rows
            if row["split"] == "calibration"
            and row["layer_type"] == layer_type.value
            and row["operation"] == operation
            and int(row["degree"]) == degree
            and int(row["rows"]) == 1
        ]
        return statistics.median(values) if values else None

    output: list[dict[str, Any]] = []
    for layer in model.layers:
        if layer.layer_type is LayerType.DENSE:
            continue
        services = ordered_services[layer.layer_type]
        for operation in ("router", "attnres"):
            output.append(
                _base_row(
                    split="calibration",
                    layer=layer.layer_id,
                    layer_type=layer.layer_type,
                    operation=operation,
                    degree=1,
                    rows=1,
                    p50_ms=services[operation],
                    source=source,
                    evidence="PHYSICAL_E022_RESIDENT_ORDERED_DAG",
                    estimate_kind="same_attention_class_ordered_operator",
                )
            )
        for operation in ("attention_shard", "expert_stripe"):
            output.append(
                _base_row(
                    split="calibration",
                    layer=layer.layer_id,
                    layer_type=layer.layer_type,
                    operation=operation,
                    degree=8,
                    rows=1,
                    p50_ms=services[operation],
                    source=source,
                    evidence="PHYSICAL_E022_RESIDENT_ORDERED_DAG",
                    estimate_kind="same_attention_class_ordered_p8_operator",
                )
            )
        for operation in ("latent_down", "latent_up", "shared_expert"):
            base_prior = prior_median(layer.layer_type, operation, 8)
            for degree in (2, 4, 8, 16):
                degree_prior = prior_median(layer.layer_type, operation, degree)
                if degree == 8:
                    value = services[operation]
                    estimate = "physical_ordered_p8"
                elif base_prior and degree_prior:
                    value = services[operation] * degree_prior / base_prior
                    estimate = "physical_ordered_p8_anchored_shape_interpolation"
                else:
                    continue
                output.append(
                    _base_row(
                        split="calibration",
                        layer=layer.layer_id,
                        layer_type=layer.layer_type,
                        operation=operation,
                        degree=degree,
                        rows=1,
                        p50_ms=value,
                        source=source,
                        evidence="PHYSICAL_E022_ORDERED_PLUS_PHYSICAL_SHAPE_RATIO",
                        estimate_kind=estimate,
                    )
                )
        for degree in (2, 4, 8, 16):
            output.append(
                _base_row(
                    split="calibration",
                    layer=layer.layer_id,
                    layer_type=layer.layer_type,
                    operation="reduction",
                    degree=degree,
                    rows=1,
                    p50_ms=barrier_by_type[layer.layer_type],
                    source=source,
                    evidence="PHYSICAL_E022_RESIDENT_ORDERED_OUTER_WALL_RESIDUAL",
                    estimate_kind="five_explicit_phase_barriers_no_global_multiplier",
                )
            )
        output.append(
            _base_row(
                split="calibration",
                layer=layer.layer_id,
                layer_type=layer.layer_type,
                operation="ordered_dag_admission",
                degree=1,
                rows=1,
                p50_ms=0.001,
                source=source,
                evidence="VALIDATION_MARKER_NOT_A_SERVICE_CHARGE",
                estimate_kind="row_one_headline_admission_marker",
            )
        )
    return output


def _attention_validation(source: Path) -> list[dict[str, Any]]:
    receipt = _read(source)
    calibration: dict[tuple[str, int, int], float] = {}
    for row in receipt["calibration"]["results"]:
        calibration[(row["attention_type"], int(row["stripe_degree"]), int(row["rows"]))] = float(
            row["independent_worker_compute_ceiling_ms"]
        )
    rows = []
    for row in receipt["heldout"]["results"]:
        key = (row["attention_type"], int(row["stripe_degree"]), int(row["rows"]))
        predicted = calibration[key]
        actual = float(row["independent_worker_compute_ceiling_ms"])
        rows.append(
            {
                "validation_class": "attention_shard_diagnostic",
                "layer": int(row["layer"]),
                "layer_type": row["attention_type"],
                "operation": "attention_shard",
                "degree": key[1],
                "rows": key[2],
                "predicted_ms": predicted,
                "actual_ms": actual,
                "absolute_error_percent": _percent_error(predicted, actual),
                "gate_scope": "diagnostic_component; ordered layer is the model gate",
            }
        )
    return rows


def _fresh_expert_rows(source: Path, model: ModelGraph) -> list[dict[str, Any]]:
    receipt = _read(source)
    rows: list[dict[str, Any]] = []
    for value in receipt["results"]:
        degree = int(value["degree"])
        row_count = int(value["rows"])
        service = float(value["independent_worker_compute_ceiling_ms"])
        for layer in model.layers:
            if layer.layer_type is LayerType.DENSE:
                continue
            row = _base_row(
                split="calibration",
                layer=layer.layer_id,
                layer_type=layer.layer_type,
                operation="expert_stripe",
                degree=degree,
                rows=row_count,
                p50_ms=service,
                source=source,
                evidence="PHYSICAL_E022_COMPLETE_RESIDENT_EXPERT_BANK",
                estimate_kind="identical_expert_shape_conditioned",
            )
            row["startup_ms"] = float(value["startup_ms"])
            row["resident_bytes"] = int(value["runtime_weight_bytes"])
            row["all_partition_shards_simultaneously_resident"] = bool(
                value["all_partition_workers_simultaneously_resident"]
            )
            row["timed_checkpoint_reads"] = int(value["timed_checkpoint_reads"])
            rows.append(row)
    return rows


def _whole_validation(
    service: ResidentServiceModel,
    fresh_rows: list[dict[str, Any]],
    model: ModelGraph,
) -> list[dict[str, Any]]:
    rows = []
    for row in fresh_rows:
        if row["split"] != "heldout" or row["operation"] != "whole_layer":
            continue
        layer = model.layers[int(row["layer"])]
        predicted = service.service_ms(layer, "whole_layer", 1, int(row["rows"]))
        actual = float(row["p50_ms"])
        rows.append(
            {
                "validation_class": "whole_layer_feature_model",
                "layer": layer.layer_id,
                "layer_type": layer.layer_type.value,
                "operation": "whole_layer",
                "degree": 1,
                "rows": int(row["rows"]),
                "predicted_ms": predicted,
                "actual_ms": actual,
                "absolute_error_percent": _percent_error(predicted, actual),
                "gate_scope": "model_gate",
            }
        )
    return rows


def materialize_validation(
    repo: Path,
    artifact_root: Path,
    model: ModelGraph,
) -> tuple[ResidentServiceModel, dict[str, Any]]:
    validation_root = artifact_root / "validation"
    provisional_path = validation_root / "provisional-inherited-service.csv"
    service_rows = materialize_provisional_service(repo, model, provisional_path)
    fresh_whole_path = validation_root / "whole-layer-resident-raw.json"
    fresh_attention_path = validation_root / "resident-attention-robust-raw.json"
    ordered_path = validation_root / "resident-ordered-raw.json"
    expert_path = validation_root / "resident-expert-bank-raw.json"
    required = (fresh_whole_path, fresh_attention_path, ordered_path, expert_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing E022 physical receipts: {missing}")

    # Replace the two calibration-layer whole services with fresh E022 values;
    # other K3 layers retain nearest-same-class physical features from E018.
    whole_operations = {
        "whole_layer",
        "attention_whole",
        "router",
        "expert_whole",
        "shared_expert_whole",
        "latent_down_whole",
        "latent_up_whole",
        "attnres",
        "reduction",
    }
    service_rows = [
        row
        for row in service_rows
        if not (
            int(row["layer"]) in (45, 47)
            and row["split"] == "calibration"
            and row["operation"] in whole_operations
        )
        and row["operation"] != "attention_shard"
        and row["operation"] != "expert_stripe"
    ]
    fresh_whole = _fresh_whole_service_rows(fresh_whole_path, model)
    service_rows.extend(fresh_whole)
    service_rows.extend(_attention_rows(model, fresh_attention_path))
    service_rows.extend(_fresh_expert_rows(expert_path, model))

    ordered_rows, barriers, ordered_services = _ordered_models(ordered_path)
    ratio_source_rows = list(service_rows)
    replaced_ordered_operations = {
        "router",
        "attnres",
        "latent_down",
        "latent_up",
        "shared_expert",
        "reduction",
    }
    service_rows = [
        row
        for row in service_rows
        if not (
            row["split"] == "calibration"
            and int(row["rows"]) == 1
            and (
                row["operation"] in replaced_ordered_operations
                or (
                    row["operation"] in {"attention_shard", "expert_stripe"}
                    and int(row["degree"]) == 8
                )
            )
        )
    ]
    service_rows.extend(
        _fresh_ordered_service_rows(
            ordered_path,
            model,
            ratio_source_rows,
            barriers,
            ordered_services,
        )
    )

    resident_csv = validation_root / "resident-shard-results.csv"
    write_csv(resident_csv, service_rows)
    service = ResidentServiceModel(service_rows)
    write_csv(validation_root / "ordered-dag-validation.csv", ordered_rows)
    whole_rows = _whole_validation(service, fresh_whole, model)
    diagnostic_rows = whole_rows + _attention_validation(fresh_attention_path)
    write_csv(validation_root / "heldout-validation.csv", diagnostic_rows)
    gate_errors = [float(row["absolute_error_percent"]) for row in ordered_rows]
    whole_errors = [float(row["absolute_error_percent"]) for row in whole_rows]
    ordered_statistics = _percentiles(gate_errors)
    whole_statistics = _percentiles(whole_errors)
    ordered_pass = (
        bool(gate_errors)
        and float(ordered_statistics["median_percent"] or 1e9) <= 5.0
        and float(ordered_statistics["p90_percent"] or 1e9) <= 10.0
        and float(ordered_statistics["maximum_percent"] or 1e9) <= 15.0
    )
    whole_pass = bool(whole_errors) and max(whole_errors) <= 15.0
    resident_pass = all(
        int(row[key]) == 0
        for row in ordered_rows
        for key in (
            "timed_checkpoint_reads",
            "timed_weight_uploads",
            "timed_buffer_allocations",
        )
    )
    correctness_pass = all(
        float(row["correctness_relative_l2"]) <= 2e-6 for row in ordered_rows
    )
    status = ordered_pass and whole_pass and resident_pass and correctness_pass
    receipt = {
        "schema_version": "experiment-022-model-validation-v1",
        "status": "PASS" if status else "FAIL",
        "gate": {
            "median_absolute_error_lte_5_percent": ordered_statistics["median_percent"],
            "p90_absolute_error_lte_10_percent": ordered_statistics["p90_percent"],
            "maximum_absolute_error_lte_15_percent": ordered_statistics["maximum_percent"],
            "ordered_dag_pass": ordered_pass,
            "whole_layer_feature_validation_pass": whole_pass,
            "resident_timed_region_pass": resident_pass,
            "correctness_pass": correctness_pass,
        },
        "ordered_validation": ordered_statistics,
        "whole_layer_validation": whole_statistics,
        "ordered_cases": len(ordered_rows),
        "whole_layer_cases": len(whole_rows),
        "normalization_applied": False,
        "global_multiplier": None,
        "post_hoc_correction": None,
        "one_compute_resource_replay": True,
        "same_deterministic_event_engine_as_planners": True,
        "ordered_runtime_decomposition": {
            "native_operator_tasks": 43,
            "explicit_phase_reduction_barriers": 5,
            "residual_definition": (
                "resident synchronized outer wall minus calibrated native operator "
                "service; divided across the five concrete algorithm barriers"
            ),
            "central_rpc_per_logical_operator": False,
            "global_normalization": False,
        },
        "calibration_layers": {"KDA": 45, "GATED_MLA": 47},
        "heldout_layers": {"KDA": 89, "GATED_MLA": 91},
        "service_model": service.describe(),
        "component_diagnostics_admission_rule": (
            "component variability is reported in heldout-validation.csv; the exact "
            "ordered implementation prediction is the preregistered model gate"
        ),
    }
    atomic_write_json(validation_root / "model-validation.json", receipt)
    return service, receipt


__all__ = ["materialize_validation"]
