"""Fresh physical service calibration for corrected Experiment 024."""

from __future__ import annotations

import csv
import gc
import json
import math
import statistics
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.execution.verification import VerificationBlock
from swarm_inference.experiments.experiment_014.full_cuda import _CheckpointReader
from swarm_inference.experiments.experiment_014.persistent_stages import (
    _stage_fixtures,
)
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _request,
)
from swarm_inference.experiments.experiment_016.benchmark import _inputs
from swarm_inference.experiments.experiment_022.completion_service import (
    _semantic_services,
)
from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    sha256_file,
    write_csv,
)
from swarm_inference.experiments.experiment_022.resident_replay import (
    replay_resident_layer,
)

from .correctness import ModelInvalidError, require_phase0
from .freeze import (
    CALIBRATION_ITERATIONS,
    CALIBRATION_WARMUP,
    CHECKPOINT,
    COMMODITY_WORKER_MEMORY_BYTES,
    DENSE_LAYER0_CALIBRATION_ITERATIONS,
    FUSION_ITERATIONS,
    FUSION_WARMUP,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    LAYER_ZERO_WHOLE_RESIDENT_BYTES,
    SERVICE_VALIDATION_MAX_ERROR_PERCENT_MAX,
    SERVICE_VALIDATION_MEDIAN_ERROR_PERCENT_MAX,
)

CUDA_LIBRARY_RELATIVE_PATH = Path(
    "artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"
)
SHARD_LIBRARY_RELATIVE_PATH = Path(
    "artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"
)
GROUPED_LIBRARY_RELATIVE_PATH = Path(
    "artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll"
)
ORACLE_ROOT_RELATIVE_PATH = Path("artifacts/experiment-014/oracle-full-93-idot0")
CATALOG_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/rerun/candidate-catalog.json"
)
REPAIRED_SERVICE_RELATIVE_PATH = Path(
    "artifacts/experiment-022/completion/validation/repaired-resident-service.csv"
)
BINDING_RECEIPT_TEMPLATE = (
    "artifacts/experiment-022/completion/physical/"
    "execute-shard-bindings-chunk-{rows}.json"
)


@dataclass(frozen=True, slots=True)
class CalibrationProtocol:
    calibration_layers: tuple[int, int] = (45, 47)
    heldout_layers: tuple[int, int] = (89, 91)
    rows: tuple[int, int, int] = (1, 2, 4)
    warmup: int = CALIBRATION_WARMUP
    iterations: int = CALIBRATION_ITERATIONS
    dense_layer0_iterations: int = DENSE_LAYER0_CALIBRATION_ITERATIONS
    fusion_warmup: int = FUSION_WARMUP
    fusion_iterations: int = FUSION_ITERATIONS


def calibration_preflight(repo_root: Path) -> CalibrationProtocol:
    require_phase0(repo_root)
    required = (
        CHECKPOINT,
        repo_root / CUDA_LIBRARY_RELATIVE_PATH,
        repo_root / SHARD_LIBRARY_RELATIVE_PATH,
        repo_root / GROUPED_LIBRARY_RELATIVE_PATH,
        repo_root / ORACLE_ROOT_RELATIVE_PATH / "hidden-trace.f32",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ModelInvalidError(f"missing physical calibration inputs: {missing}")
    return CalibrationProtocol()


def _timing(samples: Sequence[float]) -> dict[str, float]:
    if not samples or not all(math.isfinite(value) and value > 0 for value in samples):
        raise ModelInvalidError("physical service samples must be finite and positive")
    values = np.asarray(samples, dtype=np.float64)
    return {
        "minimum_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "maximum_ms": float(values.max()),
        "mean_ms": float(values.mean()),
        "standard_deviation_ms": float(values.std()),
    }


def _layer_type(layer: int, kda_layers: set[int]) -> str:
    if layer == 0:
        return "DENSE"
    return "KDA" if layer in kda_layers else "GATED_MLA"


def _isolated_request(
    checkpoint: Path,
    cuda_library: Path,
    *,
    layer: int,
    maximum_context: int,
) -> Any:
    request = _request(
        checkpoint,
        cuda_library,
        layer=layer,
        device=0,
        cycle_id="E024",
        maximum_context=maximum_context,
    )
    reader = _CheckpointReader(checkpoint)
    prefix = f"language_model.model.layers.{layer}."
    transformer_weight_bytes = sum(
        int(reader.array(name).nbytes)
        for name in reader.weight_map
        if name.startswith(prefix)
    )
    if transformer_weight_bytes <= 0:
        raise ModelInvalidError(
            f"checkpoint has no transformer weights for layer {layer}"
        )
    assignment = replace(
        request.assignment,
        weight_bytes=transformer_weight_bytes,
        owns_embeddings=False,
        owns_final_norm=False,
        owns_output_projection=False,
    )
    return request.model_copy(
        update={
            "assignment": assignment,
            "fast_path_mode": "verification-major",
            "fast_path_batch_bucket": 17,
        }
    )


def _dense_fixtures(checkpoint: Path) -> list[np.ndarray]:
    reader = _CheckpointReader(checkpoint)
    embedding = reader.array("language_model.model.embed_tokens.weight")
    values = []
    for token_id in (163584, 18699, 11):
        boundary = np.zeros((1, 9, 7168), dtype=np.float32)
        token_bits = np.asarray(embedding[token_id], dtype=np.uint16)
        boundary[0, 0] = (
            token_bits.astype(np.uint32) << np.uint32(16)
        ).view(np.float32)
        values.append(boundary)
    return values


def _dense_boundary(fixtures: list[np.ndarray], position: int) -> torch.Tensor:
    return torch.from_numpy(fixtures[position % len(fixtures)].copy())


def _measure_dense_layer_zero(
    repo_root: Path,
    *,
    protocol: CalibrationProtocol,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cuda_library = (repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve()
    fixtures = _dense_fixtures(CHECKPOINT)
    maximum_context = protocol.dense_layer0_iterations + protocol.warmup + 8
    request = _isolated_request(
        CHECKPOINT,
        cuda_library,
        layer=0,
        maximum_context=maximum_context,
    )
    started = time.perf_counter_ns()
    executor = PersistentKimiStageExecutor(
        request=request,
        checkpoint=CHECKPOINT,
        cuda_library=cuda_library,
        device=0,
    )
    try:
        load_ms = (time.perf_counter_ns() - started) / 1e6
        prepare_started = time.perf_counter_ns()
        executor.prepare_for_ready()
        prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6
        physical_resident_bytes = int(executor.resident_device_bytes)
        initial_free_bytes = int(executor.memory_before["free_bytes"])
        maximum_observed_device_bytes = max(
            physical_resident_bytes,
            initial_free_bytes - int(executor.runtime.mem_info()["free_bytes"]),
        )
        if physical_resident_bytes <= 0:
            raise ModelInvalidError("physical dense layer 0 reported no resident memory")
        if physical_resident_bytes > COMMODITY_WORKER_MEMORY_BYTES:
            raise ModelInvalidError("physical dense layer 0 exceeds commodity memory")
        samples: list[dict[str, Any]] = []
        services: dict[str, Any] = {}
        for rows in protocol.rows:
            session_ids = tuple(f"e024-dense-r{rows}-s{row}" for row in range(rows))
            for session_id in session_ids:
                executor.open_session(
                    session_id,
                    maximum_context_override=maximum_context,
                )
            maximum_observed_device_bytes = max(
                maximum_observed_device_bytes,
                initial_free_bytes - int(executor.runtime.mem_info()["free_bytes"]),
            )
            wall_values: list[float] = []
            all_finite = True
            production_native = True
            no_hot_reads = True
            try:
                for iteration in range(protocol.warmup + protocol.dense_layer0_iterations):
                    started = time.perf_counter_ns()
                    device_ms = 0.0
                    iteration_weight_loads = 0
                    iteration_materializations = 0
                    for row, session_id in enumerate(session_ids):
                        result = executor.execute_decode(
                            session_id=session_id,
                            hidden_states=_dense_boundary(fixtures, iteration + row),
                            cache_position_start=iteration,
                        )
                        record = executor.execution_records[-1]
                        device_ms += float(record.get("device_ms") or 0.0)
                        iteration_weight_loads += int(
                            record["weight_loads_during_execute"]
                        )
                        iteration_materializations += int(
                            record["materializations_during_execute"]
                        )
                        all_finite = all_finite and bool(
                            torch.isfinite(result.hidden_states).all().item()
                        )
                        production_native = production_native and (
                            result.expert_metrics.get("backend_identity")
                            == "nvidia_cuda_persistent_kimi_stage"
                        )
                    no_hot_reads = no_hot_reads and (
                        iteration_weight_loads == 0
                        and iteration_materializations == 0
                    )
                    maximum_observed_device_bytes = max(
                        maximum_observed_device_bytes,
                        initial_free_bytes
                        - int(executor.runtime.mem_info()["free_bytes"]),
                    )
                    wall_ms = (time.perf_counter_ns() - started) / 1e6
                    if iteration >= protocol.warmup:
                        measured_index = iteration - protocol.warmup
                        wall_values.append(wall_ms)
                        samples.append(
                            {
                                "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
                                "candidate_type": "WHOLE_LAYER",
                                "degree": 1,
                                "layer": 0,
                                "layer_type": "DENSE",
                                "rows": rows,
                                "sample_index": measured_index,
                                "wall_ms": wall_ms,
                                "device_ms": device_ms,
                                "finite_output": all_finite,
                                "production_native_dispatch": production_native,
                                "timed_checkpoint_reads": iteration_weight_loads,
                                "timed_materializations": iteration_materializations,
                                "resident_weights": True,
                                "resident_memory_bytes": LAYER_ZERO_WHOLE_RESIDENT_BYTES,
                                "physical_resident_device_bytes": physical_resident_bytes,
                                "commodity_worker_memory_bytes": COMMODITY_WORKER_MEMORY_BYTES,
                            }
                        )
            finally:
                for session_id in session_ids:
                    with suppress(KeyError):
                        executor.close_session(session_id)
            services[str(rows)] = {
                "physical_service": _timing(wall_values),
                "sample_count": len(wall_values),
                "all_outputs_finite": all_finite,
                "production_native_dispatch": production_native,
                "timed_checkpoint_reads": 0 if no_hot_reads else -1,
            }
            if not all_finite or not production_native or not no_hot_reads:
                raise ModelInvalidError(
                    f"dense layer 0 rows={rows} physical validity failed"
                )
        if maximum_observed_device_bytes > COMMODITY_WORKER_MEMORY_BYTES:
            raise ModelInvalidError(
                "physical dense layer 0 exceeded commodity memory during execution"
            )
        for sample in samples:
            sample["maximum_observed_device_bytes"] = maximum_observed_device_bytes
        payload = {
            "schema_version": "experiment-024-dense-layer0-whole-service-v1",
            "status": "PASS",
            "evidence_class": "PHYSICAL",
            "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
            "candidate_type": "WHOLE_LAYER",
            "degree": 1,
            "layer": 0,
            "warmup_iterations_per_rows": protocol.warmup,
            "measured_iterations_per_rows": protocol.dense_layer0_iterations,
            "resident_weights": True,
            "timed_checkpoint_reads": 0,
            "production_native_dispatch": True,
            "load_ms_excluded": load_ms,
            "prepare_ms_excluded": prepare_ms,
            "resident_memory_bytes": LAYER_ZERO_WHOLE_RESIDENT_BYTES,
            "catalog_resident_memory_bytes": LAYER_ZERO_WHOLE_RESIDENT_BYTES,
            "physical_resident_device_bytes": physical_resident_bytes,
            "maximum_observed_device_bytes": maximum_observed_device_bytes,
            "commodity_worker_memory_bytes": COMMODITY_WORKER_MEMORY_BYTES,
            "memory_feasible": (
                LAYER_ZERO_WHOLE_RESIDENT_BYTES <= COMMODITY_WORKER_MEMORY_BYTES
                and maximum_observed_device_bytes <= COMMODITY_WORKER_MEMORY_BYTES
            ),
            "service_by_rows": services,
            "compute_multiplier_scaling": "physical_p50_ms / compute_multiplier",
        }
        validate_dense_layer0_service(payload, samples)
        return samples, payload
    finally:
        executor.close()


def validate_dense_layer0_service(
    payload: dict[str, Any], samples: Sequence[dict[str, Any]]
) -> None:
    if payload.get("candidate_id") != LAYER_ZERO_WHOLE_CANDIDATE_ID:
        raise ModelInvalidError("dense calibration candidate ID changed")
    if payload.get("candidate_type") != "WHOLE_LAYER" or int(
        payload.get("degree", -1)
    ) != 1:
        raise ModelInvalidError("dense calibration is not the admitted whole candidate")
    if payload.get("production_native_dispatch") is not True:
        raise ModelInvalidError("dense calibration did not use production-native dispatch")
    if int(payload.get("timed_checkpoint_reads", -1)) != 0:
        raise ModelInvalidError("dense calibration timed checkpoint reads")
    if int(payload.get("resident_memory_bytes", -1)) != LAYER_ZERO_WHOLE_RESIDENT_BYTES:
        raise ModelInvalidError("dense calibration resident memory changed")
    physical_resident = int(payload.get("physical_resident_device_bytes", -1))
    maximum_observed = int(payload.get("maximum_observed_device_bytes", -1))
    if (
        physical_resident <= 0
        or maximum_observed < physical_resident
        or maximum_observed > COMMODITY_WORKER_MEMORY_BYTES
    ):
        raise ModelInvalidError("dense calibration physical memory evidence is invalid")
    if (
        int(payload.get("commodity_worker_memory_bytes", -1))
        != COMMODITY_WORKER_MEMORY_BYTES
        or payload.get("memory_feasible") is not True
    ):
        raise ModelInvalidError("dense calibration commodity memory gate changed")
    if payload.get("resident_weights") is not True:
        raise ModelInvalidError("dense calibration weights were not resident")
    for rows in (1, 2, 4):
        service = payload.get("service_by_rows", {}).get(str(rows))
        if not service or int(service.get("sample_count", -1)) != 200:
            raise ModelInvalidError(f"dense rows={rows} lacks 200 samples")
        p50 = float(service["physical_service"]["p50_ms"])
        if not math.isfinite(p50) or p50 <= 0:
            raise ModelInvalidError(f"dense rows={rows} service is invalid")
    if len(samples) != 600:
        raise ModelInvalidError("dense calibration sample matrix is incomplete")
    if {
        rows: sum(int(row.get("rows", -1)) == rows for row in samples)
        for rows in (1, 2, 4)
    } != {1: 200, 2: 200, 4: 200}:
        raise ModelInvalidError("dense calibration sample counts by rows changed")
    if any(
        not math.isfinite(float(row["wall_ms"]))
        or float(row["wall_ms"]) <= 0
        or row.get("finite_output") is not True
        or row.get("candidate_id") != LAYER_ZERO_WHOLE_CANDIDATE_ID
        or row.get("production_native_dispatch") is not True
        or int(row.get("timed_checkpoint_reads", -1)) != 0
        or int(row.get("resident_memory_bytes", -1))
        != LAYER_ZERO_WHOLE_RESIDENT_BYTES
        or int(row.get("physical_resident_device_bytes", -1)) <= 0
        or int(row.get("physical_resident_device_bytes", -1))
        > COMMODITY_WORKER_MEMORY_BYTES
        or int(row.get("maximum_observed_device_bytes", -1))
        < int(row.get("physical_resident_device_bytes", -1))
        or int(row.get("maximum_observed_device_bytes", -1))
        > COMMODITY_WORKER_MEMORY_BYTES
        or int(row.get("commodity_worker_memory_bytes", -1))
        != COMMODITY_WORKER_MEMORY_BYTES
        for row in samples
    ):
        raise ModelInvalidError("dense calibration contains an invalid sample")


def run_dense_calibration(repo_root: Path) -> dict[str, Any]:
    protocol = calibration_preflight(repo_root)
    root = repo_root.resolve() / "artifacts/experiment-024/calibration"
    samples, payload = _measure_dense_layer_zero(repo_root.resolve(), protocol=protocol)
    write_csv(root / "dense-layer0-whole-samples.csv", samples)
    atomic_write_json(root / "dense-layer0-whole-service.json", payload)
    return payload


def _measure_whole_moe_layer(
    repo_root: Path,
    *,
    layer: int,
    split: str,
    protocol: CalibrationProtocol,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cuda_library = (repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve()
    oracle_trace = (repo_root / ORACLE_ROOT_RELATIVE_PATH / "hidden-trace.f32").resolve()
    fixtures, _expected = _stage_fixtures(CHECKPOINT, oracle_trace, layer=layer)
    maximum_context = (protocol.warmup + protocol.iterations) * 4 + 32
    request = _isolated_request(
        CHECKPOINT,
        cuda_library,
        layer=layer,
        maximum_context=maximum_context,
    )
    started = time.perf_counter_ns()
    executor = PersistentKimiStageExecutor(
        request=request,
        checkpoint=CHECKPOINT,
        cuda_library=cuda_library,
        device=0,
    )
    try:
        load_ms = (time.perf_counter_ns() - started) / 1e6
        prepare_started = time.perf_counter_ns()
        executor.prepare_for_ready()
        prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6
        kda_layers = set(int(value) for value in executor.config.kda_layers)
        layer_type = _layer_type(layer, kda_layers)
        samples: list[dict[str, Any]] = []
        services: dict[str, Any] = {}
        for rows in protocol.rows:
            session_id = f"e024-whole-l{layer}-r{rows}"
            executor.open_session(
                session_id,
                maximum_context_override=(protocol.warmup + protocol.iterations) * rows,
            )
            wall_values: list[float] = []
            device_values: list[float] = []
            all_finite = True
            no_hot_reads = True
            try:
                position = 0
                for iteration in range(protocol.warmup + protocol.iterations):
                    block = VerificationBlock(session_id, position, rows, include_bonus_token=False)
                    started = time.perf_counter_ns()
                    record = executor.execute_verification_block(
                        block=block,
                        hidden_states=_inputs(fixtures, position=position, rows=rows),
                        expert_strategy="expert_major",
                    )
                    wall_ms = (time.perf_counter_ns() - started) / 1e6
                    boundary = np.asarray(record["boundary_output"])
                    all_finite = all_finite and bool(np.isfinite(boundary).all())
                    no_hot_reads = no_hot_reads and (
                        int(record["weight_loads_during_execute"]) == 0
                        and int(record["materializations_during_execute"]) == 0
                    )
                    if iteration >= protocol.warmup:
                        sample_index = iteration - protocol.warmup
                        device_ms = float(record.get("device_ms") or 0.0)
                        wall_values.append(wall_ms)
                        device_values.append(device_ms)
                        samples.append(
                            {
                                "split": split,
                                "candidate_id": f"layer-{layer:02d}:WHOLE_LAYER:p1",
                                "candidate_type": "WHOLE_LAYER",
                                "degree": 1,
                                "layer": layer,
                                "layer_type": layer_type,
                                "rows": rows,
                                "sample_index": sample_index,
                                "wall_ms": wall_ms,
                                "device_ms": device_ms,
                                "finite_output": bool(np.isfinite(boundary).all()),
                                "production_native_dispatch": True,
                                "timed_checkpoint_reads": 0,
                                "resident_weights": True,
                                "resident_memory_bytes": int(executor.resident_device_bytes),
                            }
                        )
                    position += rows
            finally:
                with suppress(KeyError):
                    executor.close_session(session_id)
            if not all_finite or not no_hot_reads:
                raise ModelInvalidError(f"whole layer {layer} rows={rows} validity failed")
            services[str(rows)] = {
                "wall": _timing(wall_values),
                "device": _timing(device_values),
                "sample_count": len(wall_values),
            }
        return samples, {
            "status": "PASS",
            "split": split,
            "layer": layer,
            "layer_type": layer_type,
            "candidate_id": f"layer-{layer:02d}:WHOLE_LAYER:p1",
            "degree": 1,
            "production_native_dispatch": True,
            "resident_weights": True,
            "timed_checkpoint_reads": 0,
            "load_ms_excluded": load_ms,
            "prepare_ms_excluded": prepare_ms,
            "resident_memory_bytes": int(executor.resident_device_bytes),
            "service_by_rows": services,
        }
    finally:
        executor.close()


def run_whole_layer_calibration(repo_root: Path) -> dict[str, Any]:
    protocol = calibration_preflight(repo_root)
    repo_root = repo_root.resolve()
    root = repo_root / "artifacts/experiment-024/calibration"
    all_samples: list[dict[str, Any]] = []
    layers: dict[str, Any] = {}
    for split, values in (
        ("calibration", protocol.calibration_layers),
        ("heldout", protocol.heldout_layers),
    ):
        for layer in values:
            samples, receipt = _measure_whole_moe_layer(
                repo_root,
                layer=layer,
                split=split,
                protocol=protocol,
            )
            all_samples.extend(samples)
            layers[str(layer)] = receipt
            write_csv(root / "whole-layer-calibration-samples.csv", all_samples)
            atomic_write_json(
                root / "whole-layer-service.json",
                {
                    "schema_version": "experiment-024-fresh-whole-layer-service-v1",
                    "status": "RUNNING",
                    "warmup": protocol.warmup,
                    "iterations": protocol.iterations,
                    "layers": layers,
                },
            )
            gc.collect()
    payload = {
        "schema_version": "experiment-024-fresh-whole-layer-service-v1",
        "status": "PASS",
        "evidence_class": "PHYSICAL",
        "warmup": protocol.warmup,
        "iterations": protocol.iterations,
        "layers": layers,
    }
    atomic_write_json(root / "whole-layer-service.json", payload)
    return payload


def run_p8_calibration(repo_root: Path) -> dict[str, Any]:
    protocol = calibration_preflight(repo_root)
    repo_root = repo_root.resolve()
    root = repo_root / "artifacts/experiment-024/calibration"
    cuda_library = (repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve()
    shard_library = (repo_root / SHARD_LIBRARY_RELATIVE_PATH).resolve()
    grouped_library = (repo_root / GROUPED_LIBRARY_RELATIVE_PATH).resolve()
    oracle_root = (repo_root / ORACLE_ROOT_RELATIVE_PATH).resolve()
    samples: list[dict[str, Any]] = []
    operation_services: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    for split, layers in (
        ("calibration", protocol.calibration_layers),
        ("heldout", protocol.heldout_layers),
    ):
        for layer in layers:
            for rows in protocol.rows:
                receipt = replay_resident_layer(
                    CHECKPOINT,
                    cuda_library,
                    shard_library,
                    grouped_library,
                    oracle_root,
                    layer=layer,
                    degree=8,
                    rows=rows,
                    warmup=protocol.warmup,
                    iterations=protocol.iterations,
                )
                if (
                    receipt["status"] != "PASS"
                    or int(receipt["timed_checkpoint_reads"]) != 0
                    or not receipt["all_output_values_finite"]
                ):
                    raise ModelInvalidError(
                        f"fresh P8 calibration failed layer={layer} rows={rows}"
                    )
                key = f"{layer}:{rows}"
                binding_receipt = json.loads(
                    (
                        repo_root
                        / BINDING_RECEIPT_TEMPLATE.format(rows=rows)
                    ).read_text(encoding="utf-8")
                )
                binding = {
                    str(case["operation"]): case
                    for case in binding_receipt["cases"]
                }
                reductions = {
                    str(case["operation"]).removeprefix("reduction_"): case
                    for case in binding_receipt["reduction_service_variants"]
                }
                projections = {
                    str(case["operation"]).removeprefix("projection_"): case
                    for case in binding_receipt["projection_service_variants"]
                }
                semantic = _semantic_services(
                    receipt,
                    binding,
                    reductions,
                    projections,
                )
                p8_operations = {
                    "attention_shard",
                    "attention_reduction",
                    "latent_down",
                    "expert_stripe",
                    "expert_reduction",
                    "latent_up",
                    "latent_up_reduction",
                    "shared_expert",
                    "shared_reduction",
                }
                for operation, p50_ms in sorted(semantic.items()):
                    if operation.endswith("_remote"):
                        continue
                    operation_services.append(
                        {
                            "split": split,
                            "layer": layer,
                            "layer_type": str(receipt["attention_type"]).upper(),
                            "rows": rows,
                            "operation": operation,
                            "degree": (
                                2
                                if operation == "routed_shared_reduction"
                                else 8
                                if operation in p8_operations
                                else 1
                            ),
                            "p50_ms": float(p50_ms),
                            "measurement_kind": "FRESH_PHYSICAL_NATIVE_OPERATION",
                            "production_native_dispatch": True,
                            "timed_checkpoint_reads": 0,
                        }
                    )
                receipts[key] = {
                    name: value
                    for name, value in receipt.items()
                    if name
                    not in {
                        "operation_records",
                        "instrumentation",
                        "layer_records",
                        "reduction_records",
                    }
                }
                receipts[key]["semantic_services_ms"] = semantic
                for index, wall_ms in enumerate(receipt["wall_samples_ms"]):
                    samples.append(
                        {
                            "split": split,
                            "candidate_id": f"layer-{layer:02d}:FULL_MIXED_STRIPE:p8",
                            "candidate_type": "FULL_MIXED_STRIPE",
                            "degree": 8,
                            "layer": layer,
                            "layer_type": receipt["attention_type"].upper(),
                            "rows": rows,
                            "sample_index": index,
                            "wall_ms": float(wall_ms),
                            "finite_output": True,
                            "production_native_dispatch": True,
                            "timed_checkpoint_reads": 0,
                            "resident_weights": True,
                        }
                    )
                write_csv(root / "p8-calibration-samples.csv", samples)
                write_csv(root / "p8-operation-services.csv", operation_services)
                atomic_write_json(
                    root / "p8-service.json",
                    {
                        "schema_version": "experiment-024-fresh-p8-service-v1",
                        "status": "RUNNING",
                        "warmup": protocol.warmup,
                        "iterations": protocol.iterations,
                        "receipts": receipts,
                    },
                )
                del receipt
                gc.collect()
    payload = {
        "schema_version": "experiment-024-fresh-p8-service-v1",
        "status": "PASS",
        "evidence_class": "PHYSICAL",
        "warmup": protocol.warmup,
        "iterations": protocol.iterations,
        "receipts": receipts,
        "operation_service_row_count": len(operation_services),
    }
    atomic_write_json(root / "p8-service.json", payload)
    return payload


def run_fusion_calibration(repo_root: Path) -> dict[str, Any]:
    protocol = calibration_preflight(repo_root)
    repo_root = repo_root.resolve()
    root = repo_root / "artifacts/experiment-024/calibration"
    cuda_library = (repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve()
    runtime = _CudaRuntime(cuda_library, 0)
    samples: list[dict[str, Any]] = []
    services: dict[str, Any] = {}
    maximum_count = 4 * 7168
    routed = runtime.allocate(maximum_count * 4)
    shared = runtime.allocate(maximum_count * 4)
    try:
        runtime.upload_activation(
            routed,
            np.full((4, 7168), np.float32(0.001), dtype=np.float32),
        )
        runtime.upload_activation(
            shared,
            np.full((4, 7168), np.float32(0.002), dtype=np.float32),
        )
        for rows in protocol.rows:
            wall_values: list[float] = []
            device_values: list[float] = []
            for iteration in range(protocol.fusion_warmup + protocol.fusion_iterations):
                runtime.profile_begin()
                started = time.perf_counter_ns()
                runtime.execute_add(routed, shared, rows * 7168)
                runtime.synchronize()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
                device_ms = runtime.profile_end()
                if iteration >= protocol.fusion_warmup:
                    index = iteration - protocol.fusion_warmup
                    wall_values.append(wall_ms)
                    device_values.append(device_ms)
                    samples.append(
                        {
                            "operation": "worker_local_routed_first_shared_second_fusion",
                            "rows": rows,
                            "sample_index": index,
                            "wall_ms": wall_ms,
                            "device_ms": device_ms,
                            "finite_output": True,
                            "production_native_dispatch": True,
                            "timed_checkpoint_reads": 0,
                        }
                    )
            output = runtime.download_activation(routed, (4, 7168))
            if not np.isfinite(output).all():
                raise ModelInvalidError("physical fusion produced a non-finite output")
            services[str(rows)] = {
                "wall": _timing(wall_values),
                "device": _timing(device_values),
                "sample_count": len(wall_values),
            }
    finally:
        runtime.free(shared)
        runtime.free(routed)
        runtime.close()
    write_csv(root / "fusion-samples.csv", samples)
    payload = {
        "schema_version": "experiment-024-local-fusion-service-v1",
        "status": "PASS",
        "evidence_class": "PHYSICAL",
        "warmup": protocol.fusion_warmup,
        "iterations": protocol.fusion_iterations,
        "production_native_dispatch": True,
        "timed_checkpoint_reads": 0,
        "service_by_rows": services,
    }
    atomic_write_json(root / "fusion-service.json", payload)
    return payload


def _read_samples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _p50_by_key(
    rows: Sequence[dict[str, str]],
    *keys: str,
) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        grouped.setdefault(key, []).append(float(row["wall_ms"]))
    return {key: statistics.median(values) for key, values in grouped.items()}


def assemble_service_table(repo_root: Path) -> dict[str, Any]:
    require_phase0(repo_root)
    repo_root = repo_root.resolve()
    root = repo_root / "artifacts/experiment-024/calibration"
    dense_path = root / "dense-layer0-whole-service.json"
    p8_samples_path = root / "p8-calibration-samples.csv"
    p8_operation_path = root / "p8-operation-services.csv"
    whole_samples_path = root / "whole-layer-calibration-samples.csv"
    fusion_path = root / "fusion-service.json"
    required = (
        dense_path,
        p8_samples_path,
        p8_operation_path,
        whole_samples_path,
        fusion_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ModelInvalidError(f"missing fresh E024 calibration outputs: {missing}")

    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    p8_samples = _read_samples(p8_samples_path)
    p8_operations = _read_samples(p8_operation_path)
    whole_samples = _read_samples(whole_samples_path)
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    p8_measured = _p50_by_key(p8_samples, "split", "layer_type", "rows")
    p8_operation_measured = {
        (row["split"], row["layer_type"], row["rows"], row["operation"]): float(
            row["p50_ms"]
        )
        for row in p8_operations
    }
    whole_measured = _p50_by_key(whole_samples, "split", "layer_type", "rows")

    catalog = json.loads(
        (repo_root / CATALOG_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    layer_type_by_id = {
        int(row["layer"]): str(row["layer_type"]).upper()
        for row in catalog["candidates"]
        if row["candidate_type"] == "WHOLE_LAYER"
    }
    calibration_layer = {"KDA": 45, "GATED_MLA": 47}
    heldout_layer = {"KDA": 89, "GATED_MLA": 91}
    service_rows: list[dict[str, Any]] = []
    for rows in (1, 2, 4):
        service_rows.append(
            {
                "layer": 0,
                "layer_type": "DENSE",
                "operation": "whole_layer",
                "degree": 1,
                "rows": rows,
                "p50_ms": float(
                    dense["service_by_rows"][str(rows)]["physical_service"]["p50_ms"]
                ),
                "source_layer": 0,
                "measurement_kind": "DIRECT_PHYSICAL",
                "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
                "evidence_class": "PHYSICAL",
            }
        )
    inherited_rows = _read_samples(repo_root / REPAIRED_SERVICE_RELATIVE_PATH)
    worker_protocol = {
        (row["layer_type"], row["rows"]): float(row["p50_ms"])
        for row in inherited_rows
        if row["split"] == "calibration" and row["operation"] == "worker_protocol"
    }
    p8_operation_names = sorted(
        {
            row["operation"]
            for row in p8_operations
            if row["split"] == "calibration"
        }
    )
    p8_degree_operations = {
        "attention_shard",
        "attention_reduction",
        "latent_down",
        "expert_stripe",
        "expert_reduction",
        "latent_up",
        "latent_up_reduction",
        "shared_expert",
        "shared_reduction",
    }
    for layer in range(1, 93):
        layer_type = layer_type_by_id[layer]
        for rows in (1, 2, 4):
            for operation in p8_operation_names:
                service_rows.append(
                    {
                        "layer": layer,
                        "layer_type": layer_type,
                        "operation": operation,
                        "degree": (
                            2
                            if operation == "routed_shared_reduction"
                            else 8
                            if operation in p8_degree_operations
                            else 1
                        ),
                        "rows": rows,
                        "p50_ms": p8_operation_measured[
                            ("calibration", layer_type, str(rows), operation)
                        ],
                        "source_layer": calibration_layer[layer_type],
                        "measurement_kind": "FRESH_PHYSICAL_TYPE_CONDITIONED",
                        "candidate_id": f"layer-{layer:02d}:FULL_MIXED_STRIPE:p8",
                        "evidence_class": "PHYSICALLY_GROUNDED_MODEL_INPUT",
                    }
                )
            service_rows.append(
                {
                    "layer": layer,
                    "layer_type": layer_type,
                    "operation": "worker_protocol",
                    "degree": 1,
                    "rows": rows,
                    "p50_ms": worker_protocol[(layer_type, str(rows))],
                    "source_layer": calibration_layer[layer_type],
                    "measurement_kind": "FROZEN_PHYSICAL_AUTHENTICATED_FRAME",
                    "candidate_id": f"layer-{layer:02d}:FULL_MIXED_STRIPE:p8",
                    "evidence_class": "PHYSICAL_INHERITED_E022",
                }
            )
            for operation, measured in (("whole_layer", whole_measured),):
                service_rows.append(
                    {
                        "layer": layer,
                        "layer_type": layer_type,
                        "operation": operation,
                        "degree": 8 if operation == "p8_ordered_layer" else 1,
                        "rows": rows,
                        "p50_ms": measured[("calibration", layer_type, str(rows))],
                        "source_layer": calibration_layer[layer_type],
                        "measurement_kind": "PHYSICAL_TYPE_CONDITIONED",
                        "candidate_id": (
                            f"layer-{layer:02d}:WHOLE_LAYER:p1"
                        ),
                        "evidence_class": "PHYSICALLY_GROUNDED_MODEL_INPUT",
                    }
                )
    for rows in (1, 2, 4):
        service_rows.append(
            {
                "layer": 0,
                "layer_type": "DENSE",
                "operation": "worker_protocol",
                "degree": 1,
                "rows": rows,
                "p50_ms": worker_protocol[("DENSE", str(rows))],
                "source_layer": 0,
                "measurement_kind": "FROZEN_PHYSICAL_AUTHENTICATED_FRAME",
                "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
                "evidence_class": "PHYSICAL_INHERITED_E022",
            }
        )
    for rows in (1, 2, 4):
        service_rows.append(
            {
                "layer": -1,
                "layer_type": "LOCAL_FUSION",
                "operation": "d_worker_local_fusion",
                "degree": 1,
                "rows": rows,
                "p50_ms": float(fusion["service_by_rows"][str(rows)]["wall"]["p50_ms"]),
                "source_layer": -1,
                "measurement_kind": "DIRECT_PHYSICAL",
                "candidate_id": "D:worker-local-fusion",
                "evidence_class": "PHYSICAL",
            }
        )

    validation_rows: list[dict[str, Any]] = []
    for layer_type in ("KDA", "GATED_MLA"):
        for rows in (1, 2, 4):
            for validation_class, measured in (
                ("p8_ordered_layer", p8_measured),
                ("whole_layer", whole_measured),
            ):
                predicted = measured[("calibration", layer_type, str(rows))]
                actual = measured[("heldout", layer_type, str(rows))]
                validation_rows.append(
                    {
                        "validation_class": validation_class,
                        "layer": heldout_layer[layer_type],
                        "layer_type": layer_type,
                        "rows": rows,
                        "predicted_ms": predicted,
                        "actual_ms": actual,
                        "absolute_error_percent": abs(predicted - actual) / actual * 100,
                        "normalization_or_correction_factor": False,
                    }
                )
    errors = [float(row["absolute_error_percent"]) for row in validation_rows]
    median_error = float(np.percentile(errors, 50))
    maximum_error = max(errors)
    status = (
        "PASS"
        if median_error <= SERVICE_VALIDATION_MEDIAN_ERROR_PERCENT_MAX
        and maximum_error <= SERVICE_VALIDATION_MAX_ERROR_PERCENT_MAX
        else "FAIL"
    )
    write_csv(root / "e024-service.csv", service_rows)
    write_csv(root / "heldout-validation.csv", validation_rows)
    summary = {
        "schema_version": "experiment-024-calibration-summary-v2",
        "status": status,
        "evidence_class": "PHYSICAL_CALIBRATION_AND_HELDOUT_VALIDATION",
        "dense_layer_zero_status": dense["status"],
        "dense_layer_zero_candidate_id": dense["candidate_id"],
        "dense_layer_zero_direct_physical_rows": [1, 2, 4],
        "p8_calibration_layers": [45, 47],
        "whole_layer_calibration_layers": [45, 47],
        "heldout_layers": [89, 91],
        "validation_case_count": len(validation_rows),
        "service_median_error_percent": median_error,
        "service_maximum_error_percent": maximum_error,
        "thresholds": {
            "median_percent": SERVICE_VALIDATION_MEDIAN_ERROR_PERCENT_MAX,
            "maximum_percent": SERVICE_VALIDATION_MAX_ERROR_PERCENT_MAX,
        },
        "normalization_applied": False,
        "post_hoc_correction_factor": None,
        "service_table": {
            "path": str((root / "e024-service.csv").relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(root / "e024-service.csv"),
            "row_count": len(service_rows),
        },
    }
    atomic_write_json(root / "calibration-summary.json", summary)
    if status != "PASS":
        raise ModelInvalidError(
            "fresh P8/whole-layer held-out service validation exceeded frozen thresholds"
        )
    return summary


def run_calibration(repo_root: Path, *, arm: str = "all") -> dict[str, Any]:
    """Run a resumable fresh calibration arm or the complete calibration suite."""

    if arm not in {"all", "dense", "whole", "p8", "fusion", "assemble"}:
        raise ValueError("unknown E024 calibration arm")
    results: dict[str, Any] = {}
    if arm in {"all", "dense"}:
        results["dense"] = run_dense_calibration(repo_root)
    if arm in {"all", "whole"}:
        results["whole"] = run_whole_layer_calibration(repo_root)
    if arm in {"all", "p8"}:
        results["p8"] = run_p8_calibration(repo_root)
    if arm in {"all", "fusion"}:
        results["fusion"] = run_fusion_calibration(repo_root)
    if arm in {"all", "assemble"}:
        results["summary"] = assemble_service_table(repo_root)
    return results


__all__ = [
    "CalibrationProtocol",
    "assemble_service_table",
    "calibration_preflight",
    "run_calibration",
    "run_dense_calibration",
    "run_fusion_calibration",
    "run_p8_calibration",
    "run_whole_layer_calibration",
    "validate_dense_layer0_service",
]
