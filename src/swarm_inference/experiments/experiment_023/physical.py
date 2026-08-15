"""Local RTX 5090 duplicate whole-expert-group validation for E023."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_019.checkpoint import balanced_range
from swarm_inference.experiments.experiment_019.physical import LATENT, ROUTED_EXPERTS
from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    sha256_file,
    write_csv,
)
from swarm_inference.experiments.experiment_022.model_graph import build_model_graph
from swarm_inference.experiments.experiment_022.native_dispatch import (
    ShardRequest,
    ShardTaskType,
)
from swarm_inference.experiments.experiment_022.resident_primitives import _sha256_arrays
from swarm_inference.experiments.experiment_022.whole_expert import (
    PreparedWholeExpertGroup,
)
from swarm_inference.experiments.experiment_022.whole_expert_physical import (
    _fixture as e022_frozen_fixture,
)
from swarm_inference.experiments.experiment_022.whole_expert_physical import (
    _reference as e022_whole_expert_reference,
)

from .freeze import FROZEN_CONSTANTS, validate_e023_freeze
from .replica_memory import memory_error_percent, standalone_whole_expert_group_memory

LAYERS = ((89, "KDA"), (91, "Gated_MLA"))
GROUPS = (0, 7)
ROWS = (1, 2, 4)
WARMUP_EXECUTIONS = 20
TIMED_EXECUTIONS = 200


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _e022_services(repo: Path) -> dict[tuple[int, int], float]:
    receipt = _read_object(
        repo
        / "artifacts"
        / "experiment-022"
        / "completion"
        / "physical"
        / "whole-expert-services.json"
    )
    if receipt.get("status") != "PASS" or int(receipt.get("degree", -1)) != 8:
        raise RuntimeError("E023_PHYSICAL_INVALID: frozen E022 whole-expert service failed")
    return {
        (int(row["layer"]), int(row["rows"])): float(row["native_p50_ms"])
        for row in receipt.get("services", ())
        if row.get("operation") == "expert_whole_group"
    }


def _request(layer: int, group: int, rows: int, shape: tuple[int, ...]) -> ShardRequest:
    return ShardRequest(
        assignment_id=f"e023.layer-{layer}:WHOLE_EXPERT:p8:g{group}",
        task_type=ShardTaskType.WHOLE_EXPERT_GROUP,
        layer=layer,
        shard_index=group,
        degree=8,
        rows=rows,
        input_shape=shape,
        state_id=f"e023:stateless:{layer}:{group}",
    )


def _execute(
    primitive: PreparedWholeExpertGroup,
    packed: np.ndarray,
    *,
    layer: int,
    group: int,
    rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.ascontiguousarray(packed[:rows], dtype=np.float32)
    request = _request(layer, group, rows, tuple(int(value) for value in values.shape))
    started = time.perf_counter_ns()
    output = primitive(values, request)
    outer_elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    execution = dict(primitive.last_execution)
    return output, {
        "cuda_ms": float(execution["cuda_ms"]),
        "input_copy_ms": float(execution["input_copy_ms"]),
        "output_copy_ms": float(execution["output_copy_ms"]),
        "outer_elapsed_ms": outer_elapsed_ms,
        "checkpoint_reads_in_timed_region": int(
            execution["checkpoint_reads_in_timed_region"]
        ),
        "finite": bool(np.isfinite(output).all()),
        "whole_layer_fallback": bool(execution.get("whole_layer_fallback", False)),
        "input_route_ids_sha256": str(execution["input_route_ids_sha256"]),
        "input_route_weights_sha256": str(execution["input_route_weights_sha256"]),
        "route_order_preserved": bool(execution["route_order_preserved"]),
        "expert_id_start": int(execution["expert_id_start"]),
        "expert_id_stop_exclusive": int(execution["expert_id_stop_exclusive"]),
    }


def _group_reference(
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    *,
    layer: int,
    group: int,
    latent: np.ndarray,
    routes: np.ndarray,
    weights: np.ndarray,
) -> dict[int, np.ndarray]:
    owned = balanced_range(ROUTED_EXPERTS, 8, group)
    mask = (routes >= owned.start) & (routes < owned.stop)
    group_weights = np.ascontiguousarray(np.where(mask, weights, 0.0), dtype=np.float32)
    return e022_whole_expert_reference(
        checkpoint,
        cuda_library,
        grouped_library,
        layer=layer,
        latent=latent,
        routes=routes,
        weights=group_weights,
    )


def _instantiate_copy(
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    probe: _CudaRuntime,
    *,
    layer: int,
    group: int,
    copy_id: str,
    estimator_bytes: int,
    packed: np.ndarray,
    collect_distribution: bool,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    before = probe.mem_info()
    gpu_rows: list[dict[str, Any]] = [
        {
            "layer": layer,
            "group_index": group,
            "copy_id": copy_id,
            "phase": "before_instantiation",
            "free_bytes": int(before["free_bytes"]),
            "total_bytes": int(before["total_bytes"]),
            "delta_from_before_bytes": 0,
        }
    ]
    primitive = PreparedWholeExpertGroup(
        checkpoint,
        cuda_library,
        grouped_library,
        layer=layer,
        degree=8,
        shard_index=group,
        max_rows=max(ROWS),
        shutdown_runtime_on_close=False,
    )
    outputs: dict[int, np.ndarray] = {}
    receipts: dict[int, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    def gpu_sample(phase: str) -> None:
        value = probe.mem_info()
        gpu_rows.append(
            {
                "layer": layer,
                "group_index": group,
                "copy_id": copy_id,
                "phase": phase,
                "free_bytes": int(value["free_bytes"]),
                "total_bytes": int(value["total_bytes"]),
                "delta_from_before_bytes": int(before["free_bytes"] - value["free_bytes"]),
            }
        )

    gpu_sample("after_instantiation")
    try:
        for rows in ROWS:
            for _ in range(WARMUP_EXECUTIONS):
                _execute(primitive, packed, layer=layer, group=group, rows=rows)
            gpu_sample(f"after_warmup_rows_{rows}")
            count = TIMED_EXECUTIONS if collect_distribution else 1
            output = np.empty((rows, LATENT), dtype=np.float32)
            sample: dict[str, Any] = {}
            for sample_index in range(count):
                output, sample = _execute(
                    primitive, packed, layer=layer, group=group, rows=rows
                )
                if collect_distribution:
                    samples.append(
                        {
                            "layer": layer,
                            "layer_type": "KDA" if layer == 89 else "Gated_MLA",
                            "group_index": group,
                            "rows": rows,
                            "sample_index": sample_index,
                            "cuda_ms": sample["cuda_ms"],
                            "input_copy_ms": sample["input_copy_ms"],
                            "output_copy_ms": sample["output_copy_ms"],
                            "outer_elapsed_ms": sample["outer_elapsed_ms"],
                            "checkpoint_reads_in_timed_region": sample[
                                "checkpoint_reads_in_timed_region"
                            ],
                            "finite": sample["finite"],
                        }
                    )
            outputs[rows] = output.copy()
            receipts[rows] = sample
            gpu_sample(f"after_timed_rows_{rows}")
        resident_deltas = [
            int(row["delta_from_before_bytes"])
            for row in gpu_rows
            if row["phase"] != "before_instantiation"
        ]
        physical_resident_bytes = max(resident_deltas)
        error_percent = memory_error_percent(estimator_bytes, physical_resident_bytes)
        identity = {
            "copy_id": copy_id,
            "expert_id_start": primitive.expert_start,
            "expert_id_stop_exclusive": primitive.expert_stop,
            "expert_ids": list(primitive.expert_ids),
            "startup": dict(primitive.startup),
            "primitive_reported_resident_bytes": int(primitive.resident_bytes),
            "physical_resident_bytes": physical_resident_bytes,
            "estimated_resident_bytes": estimator_bytes,
            "memory_error_percent": error_percent,
            "persistent_state_bytes": int(primitive.startup["persistent_state_bytes"]),
            "native_primitive": primitive.native_primitive,
        }
    finally:
        primitive.close()
    after_close = probe.mem_info()
    gpu_rows.append(
        {
            "layer": layer,
            "group_index": group,
            "copy_id": copy_id,
            "phase": "after_close",
            "free_bytes": int(after_close["free_bytes"]),
            "total_bytes": int(after_close["total_bytes"]),
            "delta_from_before_bytes": int(before["free_bytes"] - after_close["free_bytes"]),
        }
    )
    return outputs, receipts, samples, gpu_rows, identity


def run_physical_validation(
    *,
    repo: Path,
    checkpoint: Path = Path("F:/models/Kimi-K3"),
    cuda_library: Path | None = None,
    quantizer_library: Path | None = None,
    grouped_library: Path | None = None,
    oracle_root: Path | None = None,
) -> dict[str, Any]:
    """Run the mandatory duplicate-residency exactness and memory gate."""

    root = repo.resolve()
    validate_e023_freeze(root)
    checkpoint = checkpoint.resolve()
    cuda_library = (
        cuda_library
        or root / "artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"
    ).resolve()
    quantizer_library = (
        quantizer_library
        or root / "artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"
    ).resolve()
    grouped_library = (
        grouped_library
        or root / "artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll"
    ).resolve()
    oracle_root = (
        oracle_root or root / "artifacts/experiment-014/oracle-full-93-idot0"
    ).resolve()
    required = (
        checkpoint,
        cuda_library,
        quantizer_library,
        grouped_library,
        oracle_root / "hidden-trace.f32",
        oracle_root / "routes.txt",
    )
    if any(not value.exists() for value in required):
        missing = [str(value) for value in required if not value.exists()]
        raise RuntimeError("E023_PHYSICAL_INVALID: missing immutable input: " + ",".join(missing))

    model = build_model_graph(
        checkpoint,
        whole_layer_service_csv=root
        / "artifacts/experiment-018/physical/layer-service.csv",
    )
    e022_service = _e022_services(root)
    service_samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    service_summary: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    route_input_hashes: dict[str, dict[str, str]] = {}
    for layer, layer_type in LAYERS:
        packed, latent, routes, weights = e022_frozen_fixture(
            checkpoint,
            cuda_library,
            quantizer_library,
            oracle_root,
            layer=layer,
        )
        route_input_hashes[str(layer)] = {
            "ids": _sha256_arrays((np.ascontiguousarray(routes, dtype=np.int32),)),
            "weights": _sha256_arrays(
                (np.ascontiguousarray(weights, dtype=np.float32),)
            ),
        }
        for group in GROUPS:
            memory = standalone_whole_expert_group_memory(model.layers[layer], group)
            # The E022 reference owns and closes its CUDA runtime. Keep the
            # memory probe outside that lifetime so a global runtime shutdown
            # cannot invalidate the probe.
            reference = _group_reference(
                checkpoint,
                cuda_library,
                grouped_library,
                layer=layer,
                group=group,
                latent=latent,
                routes=routes,
                weights=weights,
            )
            probe = _CudaRuntime(cuda_library, 0)
            probe.set_telemetry("minimal")
            try:
                a_outputs, a_receipts, samples, a_gpu, a_identity = _instantiate_copy(
                    checkpoint,
                    cuda_library,
                    grouped_library,
                    probe,
                    layer=layer,
                    group=group,
                    copy_id="A",
                    estimator_bytes=memory.resident_bytes,
                    packed=packed,
                    collect_distribution=True,
                )
                b_outputs, b_receipts, _unused, b_gpu, b_identity = _instantiate_copy(
                    checkpoint,
                    cuda_library,
                    grouped_library,
                    probe,
                    layer=layer,
                    group=group,
                    copy_id="B",
                    estimator_bytes=memory.resident_bytes,
                    packed=packed,
                    collect_distribution=False,
                )
            finally:
                probe.close()
                service_samples.extend(samples)
                gpu_samples.extend(a_gpu)
                gpu_samples.extend(b_gpu)
                ownership_equal = (
                    a_identity["expert_ids"] == b_identity["expert_ids"]
                    and a_identity["expert_id_start"] == b_identity["expert_id_start"]
                    and a_identity["expert_id_stop_exclusive"]
                    == b_identity["expert_id_stop_exclusive"]
                )
                for rows in ROWS:
                    ab = _numerical_metrics(a_outputs[rows], b_outputs[rows])
                    a_ref = _numerical_metrics(reference[rows], a_outputs[rows])
                    b_ref = _numerical_metrics(reference[rows], b_outputs[rows])
                    a_receipt = a_receipts[rows]
                    b_receipt = b_receipts[rows]
                    passed = (
                        ownership_equal
                        and a_receipt["input_route_ids_sha256"]
                        == b_receipt["input_route_ids_sha256"]
                        == _sha256_arrays(
                            (np.ascontiguousarray(routes[:rows], dtype=np.int32),)
                        )
                        and a_receipt["input_route_weights_sha256"]
                        == b_receipt["input_route_weights_sha256"]
                        == _sha256_arrays(
                            (np.ascontiguousarray(weights[:rows], dtype=np.float32),)
                        )
                        and a_receipt["route_order_preserved"]
                        and b_receipt["route_order_preserved"]
                        and a_receipt["finite"]
                        and b_receipt["finite"]
                        and math.isfinite(float(ab["relative_l2_error"]))
                        and float(ab["relative_l2_error"])
                        <= float(FROZEN_CONSTANTS["whole_expert_relative_l2_gate"])
                        and float(a_ref["relative_l2_error"])
                        <= float(FROZEN_CONSTANTS["whole_expert_relative_l2_gate"])
                        and float(b_ref["relative_l2_error"])
                        <= float(FROZEN_CONSTANTS["whole_expert_relative_l2_gate"])
                        and a_receipt["checkpoint_reads_in_timed_region"] == 0
                        and b_receipt["checkpoint_reads_in_timed_region"] == 0
                        and not a_receipt["whole_layer_fallback"]
                        and not b_receipt["whole_layer_fallback"]
                        and a_identity["persistent_state_bytes"] == 0
                        and b_identity["persistent_state_bytes"] == 0
                        and a_identity["memory_error_percent"]
                        <= float(FROZEN_CONSTANTS["replica_memory_error_percent_max"])
                        and b_identity["memory_error_percent"]
                        <= float(FROZEN_CONSTANTS["replica_memory_error_percent_max"])
                    )
                    cases.append(
                        {
                            "layer": layer,
                            "layer_type": layer_type,
                            "group_index": group,
                            "rows": rows,
                            "status": "PASS" if passed else "FAIL",
                            "ownership_equal": ownership_equal,
                            "ordered_input_route_ids_equal": (
                                a_receipt["input_route_ids_sha256"]
                                == b_receipt["input_route_ids_sha256"]
                            ),
                            "ordered_input_route_weights_equal": (
                                a_receipt["input_route_weights_sha256"]
                                == b_receipt["input_route_weights_sha256"]
                            ),
                            "input_route_ids_sha256": a_receipt[
                                "input_route_ids_sha256"
                            ],
                            "input_route_weights_sha256": a_receipt[
                                "input_route_weights_sha256"
                            ],
                            "a_b_relative_l2": float(ab["relative_l2_error"]),
                            "a_e022_reference_relative_l2": float(
                                a_ref["relative_l2_error"]
                            ),
                            "b_e022_reference_relative_l2": float(
                                b_ref["relative_l2_error"]
                            ),
                            "finite": bool(
                                a_receipt["finite"] and b_receipt["finite"]
                            ),
                            "copy_a": a_receipt,
                            "copy_b": b_receipt,
                        }
                    )
                for copy_id, identity, receipts in (
                    ("A", a_identity, a_receipts),
                    ("B", b_identity, b_receipts),
                ):
                    for rows in ROWS:
                        selected = [
                            row
                            for row in samples
                            if int(row["rows"]) == rows
                        ] if copy_id == "A" else []
                        service_summary.append(
                            {
                                "layer": layer,
                                "layer_type": layer_type,
                                "group_index": group,
                                "copy_id": copy_id,
                                "rows": rows,
                                "sample_count": len(selected) or 1,
                                "outer_elapsed_p50_ms": (
                                    statistics.median(
                                        float(row["outer_elapsed_ms"]) for row in selected
                                    )
                                    if selected
                                    else float(receipts[rows]["outer_elapsed_ms"])
                                ),
                                "cuda_p50_ms": (
                                    statistics.median(float(row["cuda_ms"]) for row in selected)
                                    if selected
                                    else float(receipts[rows]["cuda_ms"])
                                ),
                                "checkpoint_reads_in_timed_region": sum(
                                    int(row["checkpoint_reads_in_timed_region"])
                                    for row in selected
                                ) if selected else int(
                                    receipts[rows]["checkpoint_reads_in_timed_region"]
                                ),
                                "finite": bool(
                                    all(bool(row["finite"]) for row in selected)
                                    if selected
                                    else receipts[rows]["finite"]
                                ),
                                "estimated_resident_bytes": memory.resident_bytes,
                                "physical_resident_bytes": identity[
                                    "physical_resident_bytes"
                                ],
                                "memory_error_percent": identity[
                                    "memory_error_percent"
                                ],
                                "persistent_state_bytes": identity[
                                    "persistent_state_bytes"
                                ],
                            }
                        )
    drift_rows: list[dict[str, Any]] = []
    for layer, layer_type in LAYERS:
        for group in GROUPS:
            for rows in ROWS:
                observed = statistics.median(
                    float(row["outer_elapsed_ms"])
                    for row in service_samples
                    if int(row["layer"]) == layer
                    and int(row["group_index"]) == group
                    and int(row["rows"]) == rows
                )
                frozen = e022_service[(layer, rows)]
                drift = 100.0 * abs(observed - frozen) / frozen
                drift_rows.append(
                    {
                        "layer": layer,
                        "layer_type": layer_type,
                        "group_index": group,
                        "rows": rows,
                        "e022_native_p50_ms": frozen,
                        "e023_outer_p50_ms": observed,
                        "absolute_drift_percent": drift,
                        "status": "HEDGE_SERVICE_DRIFT" if drift > 10.0 else "PASS",
                    }
                )

    passed = len(cases) == 12 and all(row["status"] == "PASS" for row in cases)
    output_root = root / "artifacts" / "experiment-023" / "physical"
    write_csv(output_root / "duplicate-expert-group-services.csv", service_summary)
    write_csv(output_root / "service-samples.csv", service_samples)
    write_csv(output_root / "gpu-samples.csv", gpu_samples)
    receipt = {
        "schema_version": "experiment-023-duplicate-expert-group-correctness-v1",
        "status": "PASS" if passed else "FAIL",
        "primary_gate": "PASS" if passed else "MODEL_INVALID",
        "evidence_class": "PHYSICAL local RTX 5090 production-native primitive",
        "checkpoint": str(checkpoint),
        "input_hashes": {
            "checkpoint_index": sha256_file(checkpoint / "model.safetensors.index.json"),
            "hidden_trace": sha256_file(oracle_root / "hidden-trace.f32"),
            "routes": sha256_file(oracle_root / "routes.txt"),
            "cuda_library": sha256_file(cuda_library),
            "quantizer_library": sha256_file(quantizer_library),
            "grouped_library": sha256_file(grouped_library),
        },
        "layers": [layer for layer, _ in LAYERS],
        "groups": list(GROUPS),
        "rows": list(ROWS),
        "warmup_executions": WARMUP_EXECUTIONS,
        "timed_executions": TIMED_EXECUTIONS,
        "case_count": len(cases),
        "cases": cases,
        "copy_identity": [
            {
                "layer": int(row["layer"]),
                "group_index": int(row["group_index"]),
                "copy_id": str(row["copy_id"]),
                "estimated_resident_bytes": int(row["estimated_resident_bytes"]),
                "physical_resident_bytes": int(row["physical_resident_bytes"]),
                "memory_error_percent": float(row["memory_error_percent"]),
                "persistent_state_bytes": int(row["persistent_state_bytes"]),
            }
            for row in service_summary
            if int(row["rows"]) == 1
        ],
        "service_drift": drift_rows,
        "hedge_service_drift": any(
            row["status"] == "HEDGE_SERVICE_DRIFT" for row in drift_rows
        ),
        "hedging_conclusion_suppressed": any(
            row["status"] == "HEDGE_SERVICE_DRIFT" for row in drift_rows
        ),
        "reference": "unchanged E022 grouped whole-expert reference with exact group mask",
        "simultaneous_same_gpu_residency_required": False,
        "sequential_copy_instantiation": True,
        "replica_memory_formula": asdict(
            standalone_whole_expert_group_memory(model.layers[89], 0)
        ),
        "route_input_sha256": route_input_hashes,
    }
    atomic_write_json(output_root / "duplicate-expert-group-correctness.json", receipt)
    return receipt


__all__ = ["run_physical_validation"]
