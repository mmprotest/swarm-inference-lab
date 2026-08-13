"""Physical grouped top-16 expert-stripe benchmark for Experiment 020."""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.physical import (
    LATENT,
    RELATIVE_L2_GATE,
    TOPK,
    _execute_local_route_sum,
    _upload_stripe_experts,
    real_route_workload,
    striped_latent_down,
    timing,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer


class GroupedTop16Runtime:
    def __init__(self, library: Path) -> None:
        self._library = ctypes.CDLL(str(library))
        function = self._library.e020_kimi_grouped_top16
        function.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_double),
        ]
        function.restype = ctypes.c_int
        self._execute = function
        self._library.e020_kimi_grouped_physical_launches.restype = ctypes.c_int
        self._library.e020_kimi_grouped_release.argtypes = []
        self._library.e020_kimi_grouped_release.restype = None

    @property
    def physical_launches(self) -> int:
        return int(self._library.e020_kimi_grouped_physical_launches())

    def execute(
        self,
        resident: Any,
        routes: np.ndarray,
        output_device: ctypes.c_void_p,
        activation_device: ctypes.c_void_p,
        weights_device: ctypes.c_void_p,
    ) -> float:
        flat = [int(value) for value in routes.reshape(-1)]
        pointer_array = ctypes.c_void_p * len(flat)
        gates = pointer_array(*(resident.handles[value][0] for value in flat))
        ups = pointer_array(*(resident.handles[value][1] for value in flat))
        downs = pointer_array(*(resident.handles[value][2] for value in flat))
        elapsed = ctypes.c_double()
        status = self._execute(
            gates,
            ups,
            downs,
            output_device,
            activation_device,
            weights_device,
            routes.shape[0],
            ctypes.c_float(4.0),
            ctypes.c_float(25.0),
            ctypes.byref(elapsed),
        )
        if status != 1:
            raise RuntimeError("E020 grouped top-16 CUDA execution failed")
        return float(elapsed.value)

    def close(self) -> None:
        self._library.e020_kimi_grouped_release()


def _execute_grouped(
    runtime: _CudaRuntime,
    grouped: GroupedTop16Runtime,
    resident: Any,
    activation: np.ndarray,
    routes: np.ndarray,
    route_weights: np.ndarray,
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = int(activation.shape[0])
    input_device = runtime.allocate(activation.nbytes)
    weights_device = runtime.allocate(route_weights.nbytes)
    output_device = runtime.allocate(rows * LATENT * 4)
    runtime.upload_activation(input_device, activation)
    runtime.upload_activation(weights_device, route_weights)
    wall: list[float] = []
    cuda: list[float] = []
    try:
        for iteration in range(warmup + iterations):
            started = time.perf_counter_ns()
            cuda_ms = grouped.execute(
                resident, routes, output_device, input_device, weights_device
            )
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if iteration >= warmup:
                wall.append(elapsed)
                cuda.append(cuda_ms)
        output = runtime.download_activation(output_device, (rows, LATENT))
    finally:
        runtime.free(output_device)
        runtime.free(weights_device)
        runtime.free(input_device)
    launches = grouped.physical_launches
    return {
        "wall": timing(wall),
        "cuda": timing(cuda),
        "logical_expert_operations": rows * TOPK,
        "physical_expert_launches": launches,
        "physical_reduction_launches": 0,
        "total_physical_launches": launches,
        "logical_to_physical_expert_coalescing_factor": rows * TOPK / launches,
        "network_visible_partial_outputs": 1,
    }, output


def benchmark_grouped_expert_stripe(
    checkpoint: Path,
    cuda_library: Path,
    quantizer_library: Path,
    grouped_library: Path,
    oracle_trace: Path,
    routes_path: Path,
    *,
    layer: int = 89,
    degree: int = 8,
    rows_sweep: Sequence[int] = (1, 2, 4),
    warmup: int = 1,
    iterations: int = 7,
    device: int = 0,
) -> dict[str, Any]:
    catalog = CheckpointCatalog(checkpoint)
    loader = DirectShardLoader(catalog)
    runtime = _CudaRuntime(cuda_library, device)
    runtime.set_telemetry("minimal")
    runtime.set_fused_gate_up(True)
    quantizer = GpuShardQuantizer(quantizer_library, device)
    grouped = GroupedTop16Runtime(grouped_library)
    results: list[dict[str, Any]] = []
    try:
        hidden, routes, weights, workload = real_route_workload(
            checkpoint,
            oracle_trace,
            routes_path,
            layer=layer,
            rows=max(rows_sweep),
            loader=loader,
        )
        latent, projection = striped_latent_down(
            runtime,
            loader,
            hidden,
            layer=layer,
            degree=degree,
            warmup=warmup,
            iterations=iterations,
            quantizer=quantizer,
        )
        active = sorted({int(value) for value in routes.reshape(-1)})
        for rows in rows_sweep:
            old_partials: list[np.ndarray] = []
            grouped_partials: list[np.ndarray] = []
            workers: list[dict[str, Any]] = []
            memory_before = runtime.mem_info()
            for stripe in range(degree):
                worker = f"e020.expert.layer-{layer}.p{degree}.worker-{stripe:02d}"
                resident = _upload_stripe_experts(
                    runtime,
                    loader,
                    layer=layer,
                    experts=active,
                    degree=degree,
                    stripe=stripe,
                    worker_id=worker,
                )
                try:
                    old, old_partial = _execute_local_route_sum(
                        runtime,
                        resident,
                        latent[:rows],
                        routes[:rows],
                        weights[:rows],
                        warmup=warmup,
                        iterations=iterations,
                    )
                    new, new_partial = _execute_grouped(
                        runtime,
                        grouped,
                        resident,
                        latent[:rows],
                        routes[:rows],
                        weights[:rows],
                        warmup=warmup,
                        iterations=iterations,
                    )
                    metrics = _numerical_metrics(old_partial, new_partial)
                    workers.append(
                        {
                            "worker_id": worker,
                            "stripe_index": stripe,
                            "e019_existing": old,
                            "e020_grouped": new,
                            "metrics": metrics,
                            "runtime_weight_bytes": resident.runtime_bytes,
                            "pass": float(metrics["relative_l2_error"])
                            <= RELATIVE_L2_GATE,
                        }
                    )
                    old_partials.append(old_partial)
                    grouped_partials.append(new_partial)
                finally:
                    resident.close()
            old_output = np.sum(np.stack(old_partials), axis=0, dtype=np.float64).astype(
                np.float32
            )
            grouped_output = np.sum(
                np.stack(grouped_partials), axis=0, dtype=np.float64
            ).astype(np.float32)
            end_metrics = _numerical_metrics(old_output, grouped_output)
            memory_after = runtime.mem_info()
            results.append(
                {
                    "rows": rows,
                    "stripe_degree": degree,
                    "workers": workers,
                    "old_worker_ceiling_ms": max(
                        row["e019_existing"]["wall"]["p50_ms"] for row in workers
                    ),
                    "grouped_worker_ceiling_ms": max(
                        row["e020_grouped"]["wall"]["p50_ms"] for row in workers
                    ),
                    "grouped_physical_launches_per_worker": grouped.physical_launches,
                    "logical_fragments_per_worker": rows * TOPK,
                    "minimum_launch_coalescing": min(
                        row["e020_grouped"][
                            "logical_to_physical_expert_coalescing_factor"
                        ]
                        for row in workers
                    ),
                    "relative_l2_error": float(end_metrics["relative_l2_error"]),
                    "peak_memory_delta_bytes": max(
                        0,
                        int(memory_before["free_bytes"])
                        - int(memory_after["free_bytes"]),
                    ),
                    "pass": all(row["pass"] for row in workers)
                    and float(end_metrics["relative_l2_error"]) <= RELATIVE_L2_GATE,
                }
            )
        return {
            "schema_version": "experiment-020-grouped-expert-stripe-v1",
            "status": (
                "PASS"
                if all(row["pass"] for row in results)
                and min(row["minimum_launch_coalescing"] for row in results) >= 4.0
                else "FAIL"
            ),
            "evidence_class": "PHYSICAL RTX 5090 exact grouped worker stripe",
            "configuration": {
                "layer": layer,
                "degree": degree,
                "rows_sweep": list(rows_sweep),
                "warmup": warmup,
                "iterations": iterations,
            },
            "workload": workload,
            "latent_projection": projection,
            "results": results,
            "checkpoint_read_request_count": len(loader.audit),
            "startup_quantization_count": len(quantizer.audit),
        }
    finally:
        grouped.close()
        runtime.close()


__all__ = ["GroupedTop16Runtime", "benchmark_grouped_expert_stripe"]
