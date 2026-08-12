"""Preregister and validate the Experiment 014 bottleneck-aware timing model."""

from __future__ import annotations

import asyncio
import json
import statistics
import traceback
from pathlib import Path
from typing import Any

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import (
    _batch_boundaries,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _device_identity,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.performance import _profile_layer
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.prefill_stage import (
    _reference,
    _run_context,
)
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
    _request,
)

PREREGISTRATION_SCHEMA = "experiment-014-k3-performance-model-preregistration-v1"
VALIDATION_SCHEMA = "experiment-014-k3-performance-model-heldout-validation-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _linear_fit(points: list[tuple[float, float]]) -> dict[str, Any]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    x_mean = sum(row[0] for row in points) / len(points)
    y_mean = sum(row[1] for row in points) / len(points)
    denominator = sum((row[0] - x_mean) ** 2 for row in points)
    if denominator == 0.0:
        raise ValueError("linear fit inputs have no x variance")
    slope = sum(
        (row[0] - x_mean) * (row[1] - y_mean) for row in points
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual_sum_squares = sum(
        (row[1] - (intercept + slope * row[0])) ** 2 for row in points
    )
    total_sum_squares = sum((row[1] - y_mean) ** 2 for row in points)
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else 1.0
    )
    return {
        "form": "device_p50_ms = intercept_ms + slope_ms_per_unit * x",
        "intercept_ms": intercept,
        "slope_ms_per_unit": slope,
        "r_squared_on_training_points": r_squared,
        "training_points": [
            {"x": x, "device_p50_ms": y} for x, y in points
        ],
    }


def _predict(model: dict[str, Any], x: float) -> float:
    return float(model["intercept_ms"]) + float(model["slope_ms_per_unit"]) * x


def preregister_performance_model(
    depth_profile: Path,
    mla_context_profile: Path,
    cuda_library: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-035a",
    median_ape_gate_percent: float = 10.0,
) -> dict[str, Any]:
    """Freeze held-out predictions without executing or reading held-out slices."""
    paths = {
        "depth_profile": depth_profile.resolve(),
        "mla_context_profile": mla_context_profile.resolve(),
        "cuda_library": cuda_library.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    depth = _read_json(paths["depth_profile"])
    context = _read_json(paths["mla_context_profile"])
    if depth.get("status") != "PASS" or not depth.get("result", {}).get(
        "execution_pass"
    ):
        raise ValueError("depth profile is not passing")
    if context.get("status") != "PASS" or not context.get("execution_pass"):
        raise ValueError("MLA context profile is not passing")

    rows = depth["benchmark"]["layers"]
    kda_points = [
        (float(row["layer"]), float(row["device_p50_ms"]))
        for row in rows
        if row["attention_type"] == "KDA"
    ]
    mla_points = [
        (float(row["layer"]), float(row["device_p50_ms"]))
        for row in rows
        if row["attention_type"] == "Gated_MLA"
    ]
    required_contexts = (1024, 4096, 8192, 16384)
    context_points = [
        (
            float(tokens),
            float(
                context["contexts"][str(tokens)]["prefill"]["last_256_device"][
                    "p50_ms"
                ]
            ),
        )
        for tokens in required_contexts
    ]
    if [int(row[0]) for row in kda_points] != [1, 45, 89]:
        raise ValueError("unexpected KDA depth training layers")
    if [int(row[0]) for row in mla_points] != [3, 47, 91]:
        raise ValueError("unexpected Gated MLA depth training layers")
    if any(context["contexts"][str(tokens)]["status"] != "PASS" for tokens in required_contexts):
        raise ValueError("one or more MLA context training points are not passing")

    models = {
        "kda_depth": _linear_fit(kda_points),
        "gated_mla_depth": _linear_fit(mla_points),
        "gated_mla_context": _linear_fit(context_points),
    }
    predictions = {
        "kda_layer_65_batch1_device_p50_ms": _predict(models["kda_depth"], 65.0),
        "gated_mla_layer_67_batch1_device_p50_ms": _predict(
            models["gated_mla_depth"], 67.0
        ),
        "gated_mla_layer_91_context_2048_last256_device_p50_ms": _predict(
            models["gated_mla_context"], 2048.0
        ),
    }
    receipt = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "cycle_id": cycle_id,
        "status": "PASS",
        "hypothesis": (
            "Architecture-class depth interpolation plus an O(T) Gated-MLA context "
            "fit predicts three unseen real Kimi slices with median absolute "
            f"percentage error <= {median_ape_gate_percent:g}%."
        ),
        "implementation": (
            "Ordinary least-squares linear fits over immutable retained evidence; "
            "no held-out receipt is an input and the validator is forbidden to refit."
        ),
        "sources": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        },
        "training_scope": {
            "kda_depth_layers": [1, 45, 89],
            "gated_mla_depth_layers": [3, 47, 91],
            "gated_mla_context_tokens": list(required_contexts),
            "held_out_slices_not_executed_or_read": [
                "KDA layer 65 batch-1 warm production device p50",
                "Gated MLA layer 67 batch-1 warm production device p50",
                "Gated MLA layer 91 context 2048 last-256 device p50",
            ],
        },
        "models": models,
        "predictions": predictions,
        "acceptance_gate": {
            "statistic": "median absolute percentage error across the three slices",
            "maximum_percent": median_ape_gate_percent,
            "denominator": "measured held-out value",
            "execution_gates_must_pass": True,
        },
        "decision": "FREEZE_PREDICTIONS_THEN_EXECUTE_H014_035B",
    }
    _atomic_json(output_path, receipt)
    return receipt


def _ape_percent(predicted: float, measured: float) -> float:
    if measured == 0.0:
        raise ValueError("cannot calculate percentage error against zero")
    return 100.0 * abs(predicted - measured) / abs(measured)


async def _validate_performance_model_async(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    reference_receipt: Path,
    preregistration: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
    cycle_id: str,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError("held-out profiling requires >=3 warmups and >=20 calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "identity_manifest": identity_manifest.resolve(),
        "reference_receipt": reference_receipt.resolve(),
        "preregistration": preregistration.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    prereg = _read_json(paths["preregistration"])
    if prereg.get("schema_version") != PREREGISTRATION_SCHEMA or prereg.get(
        "status"
    ) != "PASS":
        raise ValueError("performance-model preregistration is not valid")
    cuda_sha = _sha256_file(paths["cuda_library"])
    if cuda_sha != prereg["sources"]["cuda_library"]["sha256"]:
        raise ValueError("held-out CUDA binary differs from preregistered binary")

    receipt: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": prereg["hypothesis"],
        "configuration": {
            "depth_layers": [65, 67],
            "context_layer": 91,
            "held_out_context_tokens": 2048,
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "telemetry_mode": "production",
            "device": device,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "preregistered_predictions": prereg["predictions"],
        "acceptance_gate": prereg["acceptance_gate"],
        "device_identity": _device_identity(device),
        "gpu_health_before": _health_snapshot(device),
        "progress": [{"phase": "preregistration_verified"}],
        "held_out": {},
    }
    _atomic_json(output_path, receipt)
    try:
        for layer, key in (
            (65, "kda_layer_65_batch1_device_p50_ms"),
            (67, "gated_mla_layer_67_batch1_device_p50_ms"),
        ):
            profile = await _profile_layer(
                paths["checkpoint"],
                paths["cuda_library"],
                paths["oracle_trace"],
                paths["oracle_routes"],
                paths["identity_manifest"],
                layer=layer,
                device=device,
                warmup=warmup,
                iterations=iterations,
                mode_order=("production",),
                cycle_id=cycle_id,
            )
            measured = float(profile["modes"]["production"]["device"]["p50_ms"])
            predicted = float(prereg["predictions"][key])
            health = _health_snapshot(device)
            receipt["held_out"][key] = {
                "prediction_ms": predicted,
                "measurement_ms": measured,
                "absolute_percentage_error": _ape_percent(predicted, measured),
                "execution_pass": profile["status"] == "PASS",
                "post_slice_nvidia_smi": health,
                "profile": profile,
            }
            receipt["progress"].append(
                {"phase": f"layer_{layer}", "status": profile["status"]}
            )
            _atomic_json(output_path, receipt)
            if profile["status"] != "PASS" or health["status"] != "MEASURED":
                raise RuntimeError(f"held-out layer {layer} failed closed")

        reference = _read_json(paths["reference_receipt"])
        expected_fingerprints, expected_routes = _reference(reference, 91)
        fixtures, _ = _stage_fixtures(
            paths["checkpoint"], paths["oracle_trace"], layer=91
        )
        boundaries = [
            _batch_boundaries(fixtures, batch=1, position=position)
            for position in range(3)
        ]
        request = _request(
            paths["checkpoint"],
            paths["cuda_library"],
            layer=91,
            device=device,
            cycle_id=cycle_id,
            maximum_context=2049,
        )
        executor: PersistentKimiStageExecutor | None = None
        try:
            executor = PersistentKimiStageExecutor(
                request=request,
                checkpoint=paths["checkpoint"],
                cuda_library=paths["cuda_library"],
                device=device,
            )
            context_result = _run_context(
                executor,
                boundaries,
                expected_fingerprints,
                expected_routes,
                context=2048,
                cycle_id=cycle_id,
                device=device,
            )
        finally:
            if executor is not None:
                executor.close()
        key = "gated_mla_layer_91_context_2048_last256_device_p50_ms"
        measured = float(context_result["prefill"]["last_256_device"]["p50_ms"])
        predicted = float(prereg["predictions"][key])
        receipt["held_out"][key] = {
            "prediction_ms": predicted,
            "measurement_ms": measured,
            "absolute_percentage_error": _ape_percent(predicted, measured),
            "execution_pass": context_result["status"] == "PASS",
            "context_result": context_result,
        }
        receipt["progress"].append(
            {"phase": "context_2048", "status": context_result["status"]}
        )
        _atomic_json(output_path, receipt)
        if context_result["status"] != "PASS":
            raise RuntimeError("held-out MLA context failed closed")

        errors = [
            float(row["absolute_percentage_error"])
            for row in receipt["held_out"].values()
        ]
        median_ape = statistics.median(errors)
        execution_pass = all(
            bool(row["execution_pass"]) for row in receipt["held_out"].values()
        )
        gate = float(prereg["acceptance_gate"]["maximum_percent"])
        hypothesis_supported = execution_pass and median_ape <= gate
        receipt["validation"] = {
            "absolute_percentage_errors": errors,
            "median_absolute_percentage_error": median_ape,
            "maximum_absolute_percentage_error": max(errors),
            "gate_percent": gate,
            "gate_pass": median_ape <= gate,
            "execution_pass": execution_pass,
            "hypothesis_supported": hypothesis_supported,
        }
        receipt["inspection"] = {
            "largest_residual": max(
                receipt["held_out"].items(),
                key=lambda item: float(item[1]["absolute_percentage_error"]),
            )[0],
            "model_refit_after_holdout": False,
            "remaining_physical_unknown": (
                "RTX 3090 timing and physical network contention are not measured here."
            ),
        }
        receipt["decision"] = (
            "RETAIN_FOR_TOPOLOGY_AND_CAPACITY_SEARCH"
            if hypothesis_supported
            else "REDESIGN_FALSIFIED_MODEL_COMPONENT"
        )
        receipt["status"] = "PASS" if hypothesis_supported else "FAIL"
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["progress"].append(
            {"phase": "complete", "status": receipt["status"]}
        )
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["gpu_health_after_failure"] = _health_snapshot(device)
        receipt["progress"].append({"phase": "failed", "status": "FAIL"})
    _atomic_json(output_path, receipt)
    return receipt


def validate_performance_model(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    reference_receipt: Path,
    preregistration: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 10,
    iterations: int = 50,
    cycle_id: str = "H014-035b",
) -> dict[str, Any]:
    """Execute the three frozen held-out Kimi slices without refitting."""
    return asyncio.run(
        _validate_performance_model_async(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            reference_receipt,
            preregistration,
            output_path,
            device=device,
            warmup=warmup,
            iterations=iterations,
            cycle_id=cycle_id,
        )
    )
