"""Physical RTX 5090 shard primitives for Experiment 019.

This module deliberately exposes shard service, not a layer or pod service.
The complete-bank residency check loads one logical worker's stripe across all
896 experts.  Compute correctness executes every stripe worker sequentially on
the single physical GPU and reduces their one-partial-per-worker outputs.
"""

from __future__ import annotations

import ctypes
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _load_real_expert,
    _numerical_metrics,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    _pointer_offset,
    _quantize_bf16_grouped_int4,
)
from swarm_inference.experiments.experiment_018.analysis import (
    parse_oracle_route_weights,
    parse_oracle_routes,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer

LATENT = 3584
HIDDEN = 7168
TOPK = 16
ROUTED_EXPERTS = 896
PHYSICAL_DEGREES = (2, 4, 8, 16, 32)
PHYSICAL_ROWS = (1, 2, 4)
RELATIVE_L2_GATE = 2e-6


def timing(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("timing summary requires values")
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
        "mean_ms": statistics.fmean(float(value) for value in values),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "standard_deviation_ms": float(np.std(array)),
    }


@dataclass(slots=True)
class _ResidentHandles:
    runtime: _CudaRuntime
    handles: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]]
    runtime_bytes: int

    def close(self) -> None:
        for triple in self.handles.values():
            for handle in triple:
                self.runtime.release_tensor(handle)
        self.handles.clear()


def _upload_stripe_experts(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    *,
    layer: int,
    experts: Sequence[int],
    degree: int,
    stripe: int,
    worker_id: str,
) -> _ResidentHandles:
    handles: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]] = {}
    runtime_bytes = 0
    try:
        for expert in experts:
            shard = loader.expert_stripe(
                layer=layer,
                expert=int(expert),
                degree=degree,
                stripe=stripe,
                worker_id=worker_id,
            )
            triple = tuple(runtime.upload(tensor) for tensor in shard)
            handles[int(expert)] = triple  # type: ignore[assignment]
            runtime_bytes += sum(runtime.tensor_bytes(handle) for handle in triple)
        return _ResidentHandles(runtime, handles, runtime_bytes)
    except BaseException:
        for triple in handles.values():
            for handle in triple:
                runtime.release_tensor(handle)
        raise


def _upload_whole_active_experts(
    runtime: _CudaRuntime,
    checkpoint: Path,
    *,
    layer: int,
    experts: Sequence[int],
) -> _ResidentHandles:
    handles: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]] = {}
    runtime_bytes = 0
    try:
        for expert_id in experts:
            expert = _load_real_expert(checkpoint, layer, int(expert_id))
            triple = (
                runtime.upload(expert.gate),
                runtime.upload(expert.up),
                runtime.upload(expert.down),
            )
            handles[int(expert_id)] = triple
            runtime_bytes += sum(runtime.tensor_bytes(handle) for handle in triple)
        return _ResidentHandles(runtime, handles, runtime_bytes)
    except BaseException:
        for triple in handles.values():
            for handle in triple:
                runtime.release_tensor(handle)
        raise


def _execute_local_route_sum(
    runtime: _CudaRuntime,
    resident: _ResidentHandles,
    activations: np.ndarray,
    routes: np.ndarray,
    route_weights: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = int(activations.shape[0])
    if (
        activations.shape != (rows, LATENT)
        or routes.shape != (rows, TOPK)
        or route_weights.shape != (rows, TOPK)
    ):
        raise ValueError("expert stripe workload has invalid geometry")
    input_device = runtime.allocate(activations.nbytes)
    partial_device = runtime.allocate(rows * TOPK * LATENT * 4)
    weights_device = runtime.allocate(route_weights.nbytes)
    output_device = runtime.allocate(rows * LATENT * 4)
    runtime.upload_activation(input_device, activations)
    runtime.upload_activation(weights_device, route_weights)

    def execute_once() -> None:
        for row in range(rows):
            for slot in range(TOPK):
                expert = int(routes[row, slot])
                runtime.execute_resident(
                    resident.handles[expert],
                    _pointer_offset(partial_device, (row * TOPK + slot) * LATENT),
                    _pointer_offset(input_device, row * LATENT),
                    1,
                )
            runtime.execute_moe_reduction(
                _pointer_offset(output_device, row * LATENT),
                _pointer_offset(partial_device, row * TOPK * LATENT),
                _pointer_offset(weights_device, row * TOPK),
                count=TOPK,
                dimension=LATENT,
            )
        runtime.synchronize()

    wall_values: list[float] = []
    device_values: list[float] = []
    try:
        for _ in range(warmup):
            execute_once()
        for _ in range(iterations):
            started = time.perf_counter_ns()
            runtime.profile_begin()
            execute_once()
            device_values.append(runtime.profile_end())
            wall_values.append((time.perf_counter_ns() - started) / 1e6)
        output = runtime.download_activation(output_device, (rows, LATENT))
    finally:
        runtime.free(output_device)
        runtime.free(weights_device)
        runtime.free(partial_device)
        runtime.free(input_device)
    return (
        {
            "wall": timing(wall_values),
            "cuda": timing(device_values),
            "logical_expert_operations": rows * TOPK,
            "physical_expert_launches": rows * TOPK,
            "physical_reduction_launches": rows,
            "total_physical_launches": rows * (TOPK + 1),
            "logical_to_physical_expert_coalescing_factor": 1.0,
            "network_visible_partial_outputs": 1,
        },
        output,
    )


def real_route_workload(
    checkpoint: Path,
    oracle_trace: Path,
    routes_path: Path,
    *,
    layer: int,
    rows: int,
    loader: DirectShardLoader | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    catalog = loader.catalog if loader is not None else CheckpointCatalog(checkpoint)
    owned_loader = loader if loader is not None else DirectShardLoader(catalog)
    trace = np.memmap(
        oracle_trace,
        mode="r",
        dtype="<f4",
        shape=(3 * 94, HIDDEN),
    )
    token_ids = (163584, 18699, 11)
    fixtures: list[np.ndarray] = []
    for position, token_id in enumerate(token_ids):
        if layer:
            fixture = np.ascontiguousarray(trace[position * 94 + layer - 1][None, :])
        else:
            bits = owned_loader.load(
                "language_model.model.embed_tokens.weight",
                worker_id="embedding-fixture-owner",
                purpose="real_oracle_embedding_row",
                axis=0,
                start=token_id,
                stop=token_id + 1,
            ).astype(np.uint16, copy=False)
            fixture = np.ascontiguousarray(
                (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            )
        fixtures.append(fixture)
    route_map = parse_oracle_routes(routes_path)[layer]
    weight_map = parse_oracle_route_weights(routes_path)[layer]
    hidden = np.ascontiguousarray(
        np.concatenate([fixtures[index % len(fixtures)] for index in range(rows)]),
        dtype=np.float32,
    )
    routes = np.asarray([route_map[index % len(route_map)] for index in range(rows)], dtype=np.int32)
    weights = np.asarray(
        [weight_map[index % len(weight_map)] for index in range(rows)], dtype=np.float32
    )
    return hidden, routes, weights, {
        "source": "real K3 oracle boundaries and exact oracle routes cycled over requested rows",
        "fixture_positions_available": len(fixtures),
        "route_positions_available": len(route_map),
        "hidden_fingerprint": _array_fingerprint(hidden),
        "routes_fingerprint": _array_fingerprint(routes),
        "weights_fingerprint": _array_fingerprint(weights),
    }


def striped_latent_down(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    hidden: np.ndarray,
    *,
    layer: int,
    degree: int,
    warmup: int,
    iterations: int,
    quantizer: GpuShardQuantizer | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    name = (
        f"language_model.model.layers.{layer}.block_sparse_moe."
        "routed_expert_down_proj.weight"
    )
    outputs: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for stripe in range(degree):
        worker_id = f"latent-down.layer-{layer}.worker-{stripe:02d}"
        shard = balanced_range(LATENT, degree, stripe)
        source = loader.load(
            name,
            worker_id=worker_id,
            purpose="latent_down_row_projection_stripe",
            axis=0,
            start=shard.start,
            stop=shard.stop,
        )
        quantized = (
            quantizer.grouped_int4(source, owner=worker_id)
            if quantizer is not None
            else _quantize_bf16_grouped_int4(source)
        )
        handle = runtime.upload_grouped_int4(quantized)
        input_device = runtime.allocate(hidden.nbytes)
        output_shape = (hidden.shape[0], shard.stop - shard.start)
        output_device = runtime.allocate(math_prod(output_shape) * 4)
        runtime.upload_activation(input_device, hidden)
        wall: list[float] = []
        device: list[float] = []
        try:
            for iteration in range(warmup + iterations):
                started = time.perf_counter_ns()
                runtime.profile_begin()
                runtime.execute_dense(handle, output_device, input_device, hidden.shape[0])
                runtime.synchronize()
                elapsed_device = runtime.profile_end()
                if iteration >= warmup:
                    wall.append((time.perf_counter_ns() - started) / 1e6)
                    device.append(elapsed_device)
            output = runtime.download_activation(output_device, output_shape)
            outputs.append(output)
            records.append(
                {
                    "evidence_class": "PHYSICAL",
                    "worker_id": worker_id,
                    "layer": layer,
                    "stripe_degree": degree,
                    "stripe_index": stripe,
                    "rows": hidden.shape[0],
                    "output_range": [shard.start, shard.stop],
                    "checkpoint_bytes": source.nbytes,
                    "runtime_weight_bytes": runtime.tensor_bytes(handle),
                    "wall": timing(wall),
                    "cuda": timing(device),
                    "input_bytes": hidden.nbytes,
                    "output_bytes": output.nbytes,
                }
            )
        finally:
            runtime.free(output_device)
            runtime.free(input_device)
            runtime.release_tensor(handle)
    return np.concatenate(outputs, axis=1), records


def math_prod(shape: Sequence[int]) -> int:
    value = 1
    for dimension in shape:
        value *= int(dimension)
    return value


def benchmark_expert_stripes(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    routes_path: Path,
    quantizer_library: Path | None = None,
    *,
    layer: int = 89,
    degrees: Sequence[int] = PHYSICAL_DEGREES,
    rows_sweep: Sequence[int] = PHYSICAL_ROWS,
    device: int = 0,
    warmup: int = 2,
    iterations: int = 8,
    full_bank_residency_degrees: Sequence[int] = PHYSICAL_DEGREES,
) -> dict[str, Any]:
    catalog = CheckpointCatalog(checkpoint)
    loader = DirectShardLoader(catalog)
    runtime = _CudaRuntime(cuda_library, device)
    runtime.set_telemetry("minimal")
    runtime.set_fused_gate_up(True)
    quantizer = (
        GpuShardQuantizer(quantizer_library, device)
        if quantizer_library is not None
        else None
    )
    receipt: dict[str, Any] = {
        "schema_version": "experiment-019-expert-stripe-physical-v1",
        "status": "RUNNING",
        "configuration": {
            "layer": layer,
            "degrees": list(degrees),
            "rows_sweep": list(rows_sweep),
            "warmup": warmup,
            "iterations": iterations,
            "full_bank_residency_degrees": list(full_bank_residency_degrees),
        },
        "results": [],
        "bank_residency": [],
    }
    try:
        hidden_max, routes_max, weights_max, workload = real_route_workload(
            checkpoint,
            oracle_trace,
            routes_path,
            layer=layer,
            rows=max(rows_sweep),
            loader=loader,
        )
        latent_max, projection_records = striped_latent_down(
            runtime,
            loader,
            hidden_max,
            layer=layer,
            degree=max(degrees),
            warmup=warmup,
            iterations=iterations,
            quantizer=quantizer,
        )
        receipt["workload"] = workload
        receipt["block_route_contexts"] = []
        for block in (7, 16):
            _hidden, block_routes, block_weights, block_workload = real_route_workload(
                checkpoint,
                oracle_trace,
                routes_path,
                layer=layer,
                rows=block + 1,
                loader=loader,
            )
            receipt["block_route_contexts"].append(
                {
                    "block": block,
                    "rows": block + 1,
                    "total_assignments": (block + 1) * TOPK,
                    "active_experts": len(
                        {int(value) for value in block_routes.reshape(-1)}
                    ),
                    "routes_fingerprint": _array_fingerprint(block_routes),
                    "weights_fingerprint": _array_fingerprint(block_weights),
                    "workload": block_workload,
                }
            )
        receipt["latent_projection"] = projection_records
        unique = sorted({int(value) for value in routes_max.reshape(-1)})

        for degree in degrees:
            if degree in full_bank_residency_degrees:
                worker_id = f"expert-bank.layer-{layer}.p{degree}.worker-00"
                before = runtime.mem_info()
                started = time.perf_counter_ns()
                bank = _upload_stripe_experts(
                    runtime,
                    loader,
                    layer=layer,
                    experts=list(range(ROUTED_EXPERTS)),
                    degree=degree,
                    stripe=0,
                    worker_id=worker_id,
                )
                try:
                    after = runtime.mem_info()
                    receipt["bank_residency"].append(
                        {
                            "evidence_class": "PHYSICAL",
                            "worker_id": worker_id,
                            "stripe_degree": degree,
                            "stripe_index": 0,
                            "resident_expert_count": len(bank.handles),
                            "arbitrary_route_ready": len(bank.handles) == ROUTED_EXPERTS,
                            "runtime_weight_bytes": bank.runtime_bytes,
                            "measured_free_memory_delta_bytes": before["free_bytes"]
                            - after["free_bytes"],
                            "startup_load_ms": (time.perf_counter_ns() - started) / 1e6,
                            "startup_excluded_from_service": True,
                        }
                    )
                finally:
                    bank.close()

            for rows in rows_sweep:
                routes = routes_max[:rows]
                route_weights = weights_max[:rows]
                activations = latent_max[:rows]
                worker_results: list[dict[str, Any]] = []
                partials: list[np.ndarray] = []
                for stripe in range(degree):
                    worker_id = f"expert.layer-{layer}.p{degree}.worker-{stripe:02d}"
                    resident = _upload_stripe_experts(
                        runtime,
                        loader,
                        layer=layer,
                        experts=unique,
                        degree=degree,
                        stripe=stripe,
                        worker_id=worker_id,
                    )
                    try:
                        measured, partial = _execute_local_route_sum(
                            runtime,
                            resident,
                            activations,
                            routes,
                            route_weights,
                            warmup=warmup,
                            iterations=iterations,
                        )
                        worker_results.append(
                            {
                                "worker_id": worker_id,
                                "stripe_index": stripe,
                                "active_runtime_weight_bytes": resident.runtime_bytes,
                                **measured,
                            }
                        )
                        partials.append(partial)
                    finally:
                        resident.close()
                reduction_started = time.perf_counter_ns()
                reconstructed = np.zeros_like(partials[0], dtype=np.float32)
                for partial in partials:
                    reconstructed += partial
                reduction_ms = (time.perf_counter_ns() - reduction_started) / 1e6

                reference_resident = _upload_whole_active_experts(
                    runtime, checkpoint, layer=layer, experts=unique
                )
                try:
                    canonical_measurement, reference = _execute_local_route_sum(
                        runtime,
                        reference_resident,
                        activations,
                        routes,
                        route_weights,
                        warmup=warmup,
                        iterations=iterations,
                    )
                finally:
                    reference_resident.close()
                metrics = _numerical_metrics(reference, reconstructed)
                receipt["results"].append(
                    {
                        "evidence_class": "PHYSICAL sequential logical workers on one RTX 5090",
                        "stripe_degree": degree,
                        "rows": rows,
                        "active_experts": len(unique),
                        "input_bytes_per_worker": activations.nbytes,
                        "route_metadata_bytes_per_worker": routes.nbytes
                        + route_weights.nbytes,
                        "one_partial_output_bytes_per_worker": reconstructed.nbytes,
                        "network_visible_outputs_per_worker": 1,
                        "local_expert_intermediates_network_visible": False,
                        "worker_results": worker_results,
                        "naive_per_expert_network_control": {
                            "same_compute_measurement": True,
                            "network_visible_outputs_per_worker": TOPK,
                            "output_bytes_per_worker": rows * TOPK * LATENT * 4,
                            "rpc_count_per_layer": degree * TOPK,
                            "route_workload_identical": True,
                        },
                        "independent_worker_compute_ceiling_ms": max(
                            item["wall"]["p50_ms"] for item in worker_results
                        ),
                        "sequential_worker_compute_ms": sum(
                            item["wall"]["p50_ms"] for item in worker_results
                        ),
                        "local_host_reduction_ms": reduction_ms,
                        "canonical_whole_expert": canonical_measurement,
                        "metrics": metrics,
                        "reference_fingerprint": _array_fingerprint(reference),
                        "reconstructed_fingerprint": _array_fingerprint(reconstructed),
                        "pass": float(metrics["relative_l2_error"]) <= RELATIVE_L2_GATE,
                    }
                )
        receipt["checkpoint_read_audit"] = loader.audit
        receipt["startup_quantization_audit"] = (
            quantizer.audit if quantizer is not None else []
        )
        receipt["direct_loader_gate"] = {
            "headline_full_expert_materializations": sum(
                bool(row["full_source_tensor_materialized"])
                and row["purpose"].startswith("routed_expert")
                for row in loader.audit
            ),
            "bytes_read": sum(int(row["bytes_read"]) for row in loader.audit),
            "request_count": len(loader.audit),
        }
        receipt["status"] = (
            "PASS"
            if all(bool(row["pass"]) for row in receipt["results"])
            and receipt["direct_loader_gate"]["headline_full_expert_materializations"] == 0
            else "FAIL"
        )
        return receipt
    finally:
        runtime.close()


__all__ = [
    "PHYSICAL_DEGREES",
    "PHYSICAL_ROWS",
    "benchmark_expert_stripes",
    "real_route_workload",
    "striped_latent_down",
    "timing",
]
