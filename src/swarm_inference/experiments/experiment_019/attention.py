"""Physical sub-layer KDA and Gated-MLA attention stripes."""

from __future__ import annotations

import ctypes
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _numerical_metrics,
    _QuantizedInt8Tensor,
    _quantize_bf16_rows_int8,
    _rmsnorm_reference,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    _pointer_offset,
    _quantize_bf16_grouped_int4,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.physical import timing
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer

HIDDEN = 7168
HEADS = 96
HEAD_DIMENSION = 128
KDA_PROJECTION = HEADS * HEAD_DIMENSION
QUERY_NOPE = 128
QUERY_ROPE = 64
VALUE_DIMENSION = 128
QUERY_LORA = 1536
KV_LORA = 512
MLA_QUERY_PER_HEAD = QUERY_NOPE + QUERY_ROPE
MLA_KV_B_PER_HEAD = QUERY_NOPE + VALUE_DIMENSION
RELATIVE_L2_GATE = 2e-6


class KdaShardKernel:
    """Experiment-owned KDA entry point without the inherited 96-head guard."""

    def __init__(self, path: Path, device: int) -> None:
        self.path = path.resolve()
        self.device = device
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._library = ctypes.CDLL(str(self.path))
        pointer = ctypes.c_void_p
        function = self._library.exp019_kda_shard_short_window_dev
        function.argtypes = [
            ctypes.c_int,
            *([pointer] * 17),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        function.restype = ctypes.c_int
        self._function = function

    def execute(
        self,
        output: ctypes.c_void_p,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        gate: ctypes.c_void_p,
        decay: ctypes.c_void_p,
        beta: ctypes.c_void_p,
        conv_q: ctypes.c_void_p,
        conv_k: ctypes.c_void_p,
        conv_v: ctypes.c_void_p,
        window_q: ctypes.c_void_p,
        window_k: ctypes.c_void_p,
        window_v: ctypes.c_void_p,
        state: ctypes.c_void_p,
        dt: ctypes.c_void_p,
        a: ctypes.c_void_p,
        output_norm: ctypes.c_void_p,
        *,
        rows: int,
        heads: int,
    ) -> None:
        status = self._function(
            self.device,
            output,
            q,
            k,
            v,
            gate,
            decay,
            beta,
            conv_q,
            conv_k,
            conv_v,
            window_q,
            window_k,
            window_v,
            state,
            dt,
            a,
            output_norm,
            rows,
            heads,
            HEAD_DIMENSION,
            4,
            ctypes.c_float(-5.0),
            ctypes.c_float(1e-5),
        )
        if status != 1:
            raise RuntimeError("Experiment 019 KDA shard kernel rejected execution")


def _bf16_f32(source: np.ndarray) -> np.ndarray:
    if source.dtype != np.dtype("<u2"):
        raise ValueError("expected BF16 bits")
    return np.ascontiguousarray(
        (source.astype(np.uint32) << np.uint32(16)).view(np.float32)
    )


@dataclass(slots=True)
class DeviceResources:
    runtime: _CudaRuntime
    handles: list[ctypes.c_void_p] = field(default_factory=list)
    allocations: list[ctypes.c_void_p] = field(default_factory=list)
    weight_bytes: int = 0

    def tensor(self, handle: ctypes.c_void_p) -> ctypes.c_void_p:
        self.handles.append(handle)
        self.weight_bytes += self.runtime.tensor_bytes(handle)
        return handle

    def allocate(self, elements: int) -> ctypes.c_void_p:
        pointer = self.runtime.allocate(elements * 4)
        self.allocations.append(pointer)
        return pointer

    def upload(self, values: np.ndarray) -> ctypes.c_void_p:
        source = np.ascontiguousarray(values, dtype=np.float32)
        pointer = self.allocate(source.size)
        self.runtime.upload_activation(pointer, source)
        return pointer

    def close(self) -> None:
        for pointer in reversed(self.allocations):
            self.runtime.free(pointer)
        self.allocations.clear()
        for handle in reversed(self.handles):
            self.runtime.release_tensor(handle)
        self.handles.clear()


def _int4_rows(
    runtime: _CudaRuntime,
    resources: DeviceResources,
    loader: DirectShardLoader,
    name: str,
    *,
    worker_id: str,
    start: int,
    stop: int,
    purpose: str,
    quantizer: GpuShardQuantizer | None = None,
) -> ctypes.c_void_p:
    source = loader.load(
        name,
        worker_id=worker_id,
        purpose=purpose,
        axis=0,
        start=start,
        stop=stop,
    )
    tensor = (
        quantizer.grouped_int4(source, owner=worker_id)
        if quantizer is not None
        else _quantize_bf16_grouped_int4(source)
    )
    return resources.tensor(runtime.upload_grouped_int4(tensor))


def _int4_columns(
    runtime: _CudaRuntime,
    resources: DeviceResources,
    loader: DirectShardLoader,
    name: str,
    *,
    worker_id: str,
    start: int,
    stop: int,
    purpose: str,
    quantizer: GpuShardQuantizer | None = None,
) -> ctypes.c_void_p:
    if start % 64 or stop % 64:
        raise ValueError("grouped-int4 column stripes must align to 64 values")
    source = loader.load(
        name,
        worker_id=worker_id,
        purpose=purpose,
        axis=1,
        start=start,
        stop=stop,
    )
    tensor = (
        quantizer.grouped_int4(source, owner=worker_id)
        if quantizer is not None
        else _quantize_bf16_grouped_int4(source)
    )
    return resources.tensor(runtime.upload_grouped_int4(tensor))


def _int8_rows(
    runtime: _CudaRuntime,
    resources: DeviceResources,
    loader: DirectShardLoader,
    name: str,
    *,
    worker_id: str,
    start: int,
    stop: int,
    purpose: str,
    quantizer: GpuShardQuantizer | None = None,
) -> ctypes.c_void_p:
    source = loader.load(
        name,
        worker_id=worker_id,
        purpose=purpose,
        axis=0,
        start=start,
        stop=stop,
    )
    tensor = (
        quantizer.row_int8(source, owner=worker_id)
        if quantizer is not None
        else _quantize_bf16_rows_int8(source)
    )
    return resources.tensor(runtime.upload_int8(tensor))


def _distributed_int8_columns(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    name: str,
    *,
    degree: int,
    worker_prefix: str,
    quantizer: GpuShardQuantizer | None = None,
) -> tuple[list[tuple[DeviceResources, ctypes.c_void_p, tuple[int, int]]], dict[str, Any]]:
    record = loader.catalog.record(name)
    if len(record.shape) != 2:
        raise ValueError("distributed int8 column quantization requires a matrix")
    output, inputs = record.shape
    maxima = np.zeros(output, dtype=np.float32)
    ranges: list[tuple[int, int]] = []
    for stripe in range(degree):
        shard = balanced_range(inputs, degree, stripe)
        worker = f"{worker_prefix}.worker-{stripe:02d}"
        source = loader.load(
            name,
            worker_id=worker,
            purpose="startup_int8_global_scale_local_max",
            axis=1,
            start=shard.start,
            stop=shard.stop,
        )
        local_maxima = (
            quantizer.row_maximum(source, owner=worker)
            if quantizer is not None
            else np.max(np.abs(_bf16_f32(source)), axis=1).astype(np.float32)
        )
        maxima = np.maximum(maxima, local_maxima)
        ranges.append((shard.start, shard.stop))
    scales = np.maximum(maxima / np.float32(127.0), np.float32(1e-20)).astype(np.float32)
    inverse = np.float32(1.0) / scales
    result: list[tuple[DeviceResources, ctypes.c_void_p, tuple[int, int]]] = []
    try:
        for stripe, (start, stop) in enumerate(ranges):
            worker = f"{worker_prefix}.worker-{stripe:02d}"
            source = loader.load(
                name,
                worker_id=worker,
                purpose="startup_int8_global_scale_quantized_column_stripe",
                axis=1,
                start=start,
                stop=stop,
            )
            if quantizer is not None:
                tensor = quantizer.row_int8_with_scales(
                    source, scales, owner=worker
                )
            else:
                values = _bf16_f32(source)
                quantized = np.clip(
                    np.rint(values * inverse[:, None]), -127, 127
                ).astype(np.int8)
                tensor = _QuantizedInt8Tensor(
                    weights=np.ascontiguousarray(quantized),
                    scales=np.ascontiguousarray(scales),
                    input_dimension=stop - start,
                    output_dimension=output,
                )
            resources = DeviceResources(runtime)
            handle = resources.tensor(runtime.upload_int8(tensor))
            result.append((resources, handle, (start, stop)))
        return result, {
            "algorithm": "two_pass_local_max_then_exact_fp32_max_allreduce",
            "participants": degree,
            "allreduce_payload_bytes": output * 4,
            "source_tensor_materialized_by_one_worker": False,
            "global_scale_fingerprint": _array_fingerprint(scales),
        }
    except BaseException:
        for resources, _handle, _range in result:
            resources.close()
        raise


def _measure(
    operation: Callable[[], None],
    *,
    runtime: _CudaRuntime,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    wall: list[float] = []
    cuda: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        runtime.profile_begin()
        operation()
        cuda.append(runtime.profile_end())
        wall.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "wall": timing(wall),
        "cuda": timing(cuda),
        "host_overhead": timing(
            [maximum - device for maximum, device in zip(wall, cuda, strict=True)]
        ),
    }


def normalized_real_inputs(
    catalog: CheckpointCatalog,
    loader: DirectShardLoader,
    oracle_trace: Path,
    *,
    layer: int,
    rows: int,
) -> np.ndarray:
    trace = np.memmap(oracle_trace, mode="r", dtype="<f4", shape=(3 * 94, HIDDEN))
    inputs = np.ascontiguousarray(
        np.stack([trace[(index % 3) * 94 + layer - 1] for index in range(rows)]),
        dtype=np.float32,
    )
    name = f"language_model.model.layers.{layer}.input_layernorm.weight"
    norm = _bf16_f32(
        loader.reviewed_small(name, worker_id=f"attention.layer-{layer}.worker-00", purpose="replicated_small_input_norm")
    )
    return np.ascontiguousarray(
        np.stack([_rmsnorm_reference(row, norm, 1e-5) for row in inputs]),
        dtype=np.float32,
    )


def execute_kda_attention(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    normalized: np.ndarray,
    *,
    layer: int,
    degree: int,
    warmup: int,
    iterations: int,
    shard_kernel: KdaShardKernel | None = None,
    quantizer: GpuShardQuantizer | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if shard_kernel is None:
        raise ValueError("KDA stripe execution requires the Experiment 019 shard kernel")
    prefix = f"language_model.model.layers.{layer}.self_attn"
    rows = normalized.shape[0]
    common = DeviceResources(runtime)
    workers: list[DeviceResources] = []
    try:
        common_worker = f"kda.layer-{layer}.p{degree}.worker-00"
        f_a_source = loader.reviewed_small(
            f"{prefix}.f_a_proj.weight",
            worker_id=common_worker,
            purpose="reviewed_small_kda_decay_low_projection",
        )
        f_a = common.tensor(runtime.upload_float32(_bf16_f32(f_a_source)))
        input_device = common.upload(normalized)
        decay_low = common.allocate(rows * HEAD_DIMENSION)
        runtime.execute_dense(f_a, decay_low, input_device, rows)
        runtime.synchronize()
        common_record = _measure(
            lambda: (runtime.execute_dense(f_a, decay_low, input_device, rows), runtime.synchronize()),
            runtime=runtime,
            warmup=warmup,
            iterations=iterations,
        )
        partials: list[np.ndarray] = []
        worker_records: list[dict[str, Any]] = []
        for stripe in range(degree):
            resources = DeviceResources(runtime)
            workers.append(resources)
            worker_id = f"kda.layer-{layer}.p{degree}.worker-{stripe:02d}"
            heads = balanced_range(HEADS, degree, stripe)
            projection_start = heads.start * HEAD_DIMENSION
            projection_stop = heads.stop * HEAD_DIMENSION
            local_heads = heads.stop - heads.start
            local_projection = local_heads * HEAD_DIMENSION
            weights: dict[str, ctypes.c_void_p] = {}
            for role in ("q", "k", "v", "g"):
                weights[role] = _int4_rows(
                    runtime,
                    resources,
                    loader,
                    f"{prefix}.{role}_proj.weight",
                    worker_id=worker_id,
                    start=projection_start,
                    stop=projection_stop,
                    purpose=f"kda_{role}_head_rows",
                    quantizer=quantizer,
                )
            f_b_source = loader.load(
                f"{prefix}.f_b_proj.weight",
                worker_id=worker_id,
                purpose="kda_decay_head_rows",
                axis=0,
                start=projection_start,
                stop=projection_stop,
            )
            weights["f_b"] = resources.tensor(runtime.upload_float32(_bf16_f32(f_b_source)))
            beta_source = loader.load(
                f"{prefix}.b_proj.weight",
                worker_id=worker_id,
                purpose="kda_beta_head_rows",
                axis=0,
                start=heads.start,
                stop=heads.stop,
            )
            weights["beta"] = resources.tensor(runtime.upload_float32(_bf16_f32(beta_source)))
            weights["output"] = _int4_columns(
                runtime,
                resources,
                loader,
                f"{prefix}.o_proj.weight",
                worker_id=worker_id,
                start=projection_start,
                stop=projection_stop,
                purpose="kda_output_projection_columns",
                quantizer=quantizer,
            )
            for role in ("q", "k", "v"):
                conv = loader.load(
                    f"{prefix}.{role}_conv1d.weight",
                    worker_id=worker_id,
                    purpose=f"kda_{role}_conv_channels",
                    axis=0,
                    start=projection_start,
                    stop=projection_stop,
                )
                weights[f"conv_{role}"] = resources.upload(_bf16_or_f32(conv).reshape(-1))
            dt = loader.load(
                f"{prefix}.dt_bias",
                worker_id=worker_id,
                purpose="kda_dt_head_channels",
                axis=0,
                start=projection_start,
                stop=projection_stop,
            )
            weights["dt"] = resources.upload(_bf16_or_f32(dt).reshape(-1))
            a_log = loader.load(
                f"{prefix}.A_log",
                worker_id=worker_id,
                purpose="kda_state_decay_heads",
                axis=0,
                start=heads.start,
                stop=heads.stop,
            )
            weights["a"] = resources.upload(np.exp(_bf16_or_f32(a_log)).astype(np.float32))
            output_norm = loader.reviewed_small(
                f"{prefix}.o_norm.weight",
                worker_id=worker_id,
                purpose="replicated_small_kda_output_norm",
            )
            weights["output_norm"] = resources.upload(_bf16_or_f32(output_norm).reshape(-1))
            local_input = resources.upload(normalized)
            q = resources.allocate(rows * local_projection)
            k = resources.allocate(rows * local_projection)
            v = resources.allocate(rows * local_projection)
            gate = resources.allocate(rows * local_projection)
            decay = resources.allocate(rows * local_projection)
            beta = resources.allocate(rows * local_heads)
            core = resources.allocate(rows * local_projection)
            output = resources.allocate(rows * HIDDEN)
            state = resources.allocate(local_heads * HEAD_DIMENSION * HEAD_DIMENSION)
            window_q = resources.allocate(local_projection * 4)
            window_k = resources.allocate(local_projection * 4)
            window_v = resources.allocate(local_projection * 4)
            zeros_state = np.zeros(
                (local_heads, HEAD_DIMENSION, HEAD_DIMENSION), dtype=np.float32
            )
            zeros_window = np.zeros((local_projection, 4), dtype=np.float32)

            def reset() -> None:
                runtime.upload_activation(state, zeros_state)
                runtime.upload_activation(window_q, zeros_window)
                runtime.upload_activation(window_k, zeros_window)
                runtime.upload_activation(window_v, zeros_window)

            def operation() -> None:
                reset()
                for role, target in (("q", q), ("k", k), ("v", v), ("g", gate)):
                    runtime.execute_dense(weights[role], target, local_input, rows)
                runtime.execute_dense(weights["f_b"], decay, decay_low, rows)
                runtime.execute_dense(weights["beta"], beta, local_input, rows)
                shard_kernel.execute(
                    core,
                    q,
                    k,
                    v,
                    gate,
                    decay,
                    beta,
                    weights["conv_q"],
                    weights["conv_k"],
                    weights["conv_v"],
                    window_q,
                    window_k,
                    window_v,
                    state,
                    weights["dt"],
                    weights["a"],
                    weights["output_norm"],
                    rows=rows,
                    heads=local_heads,
                )
                runtime.execute_dense(weights["output"], output, core, rows)
                runtime.synchronize()

            operation()
            partial = runtime.download_activation(output, (rows, HIDDEN))
            measurement = _measure(
                operation, runtime=runtime, warmup=warmup, iterations=iterations
            )
            state_parts = {
                "recurrent": runtime.download_activation(
                    state, (local_heads, HEAD_DIMENSION, HEAD_DIMENSION)
                ),
                "window_q": runtime.download_activation(
                    window_q, (local_projection, 4)
                ),
                "window_k": runtime.download_activation(
                    window_k, (local_projection, 4)
                ),
                "window_v": runtime.download_activation(
                    window_v, (local_projection, 4)
                ),
            }
            state_digest = hashlib.sha256()
            for state_name, state_values in state_parts.items():
                state_digest.update(state_name.encode("utf-8"))
                state_digest.update(np.ascontiguousarray(state_values).tobytes())
            partials.append(partial)
            worker_records.append(
                {
                    "worker_id": worker_id,
                    "stripe_index": stripe,
                    "head_start": heads.start,
                    "head_stop": heads.stop,
                    "runtime_weight_bytes": resources.weight_bytes,
                    "persistent_state_bytes": zeros_state.nbytes + 3 * zeros_window.nbytes,
                    "state_fingerprint": "sha256:" + state_digest.hexdigest(),
                    "wall": measurement["wall"],
                    "physical_launches": 8,
                    "network_visible_partial_outputs": 1,
                }
            )
        output = np.sum(
            np.stack(partials, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        return output, {
            "attention_type": "KDA",
            "layer": layer,
            "rows": rows,
            "stripe_degree": degree,
            "common_owner": common_worker,
            "common_projection": common_record,
            "workers": worker_records,
            "independent_worker_compute_ceiling_ms": max(
                item["wall"]["p50_ms"] for item in worker_records
            ),
            "sequential_worker_compute_ms": sum(
                item["wall"]["p50_ms"] for item in worker_records
            ),
            "one_layer_reduction": True,
            "kda_shard_library": str(shard_kernel.path),
            "kda_shard_library_sha256": shard_kernel.sha256,
            "output_fingerprint": _array_fingerprint(output),
        }
    finally:
        for resources in reversed(workers):
            resources.close()
        common.close()


def _bf16_or_f32(source: np.ndarray) -> np.ndarray:
    if source.dtype == np.dtype("<u2"):
        return _bf16_f32(source)
    return np.ascontiguousarray(source, dtype=np.float32)


def execute_mla_attention(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    normalized: np.ndarray,
    *,
    layer: int,
    degree: int,
    warmup: int,
    iterations: int,
    maximum_context: int = 256,
    quantizer: GpuShardQuantizer | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    prefix = f"language_model.model.layers.{layer}.self_attn"
    rows = normalized.shape[0]
    common = DeviceResources(runtime)
    workers: list[DeviceResources] = []
    column_resources: list[tuple[DeviceResources, ctypes.c_void_p, tuple[int, int]]] = []
    try:
        common_worker = f"mla.layer-{layer}.p{degree}.worker-00"
        q_a_source = loader.reviewed_small(
            f"{prefix}.q_a_proj.weight",
            worker_id=common_worker,
            purpose="reviewed_small_mla_query_low_projection",
        )
        kv_a_source = loader.reviewed_small(
            f"{prefix}.kv_a_proj_with_mqa.weight",
            worker_id=common_worker,
            purpose="reviewed_small_mla_kv_low_projection",
        )
        q_a_tensor = (
            quantizer.row_int8(q_a_source, owner=common_worker)
            if quantizer is not None
            else _quantize_bf16_rows_int8(q_a_source)
        )
        kv_a_tensor = (
            quantizer.row_int8(kv_a_source, owner=common_worker)
            if quantizer is not None
            else _quantize_bf16_rows_int8(kv_a_source)
        )
        q_a = common.tensor(runtime.upload_int8(q_a_tensor))
        kv_a = common.tensor(runtime.upload_int8(kv_a_tensor))
        input_device = common.upload(normalized)
        query_low = common.allocate(rows * QUERY_LORA)
        compressed = common.allocate(rows * (KV_LORA + QUERY_ROPE))
        query_norm = common.upload(
            _bf16_f32(
                loader.reviewed_small(
                    f"{prefix}.q_a_layernorm.weight",
                    worker_id=common_worker,
                    purpose="replicated_small_mla_query_norm",
                )
            )
        )

        def common_operation() -> None:
            runtime.execute_dense(q_a, query_low, input_device, rows)
            runtime.execute_rmsnorm(
                query_low,
                query_low,
                query_norm,
                batch=rows,
                dimension=QUERY_LORA,
                epsilon=1e-5,
            )
            runtime.execute_dense(kv_a, compressed, input_device, rows)
            runtime.synchronize()

        common_operation()
        common_record = _measure(
            common_operation,
            runtime=runtime,
            warmup=warmup,
            iterations=iterations,
        )
        query_low_host = runtime.download_activation(query_low, (rows, QUERY_LORA))
        compressed_host = runtime.download_activation(compressed, (rows, KV_LORA + QUERY_ROPE))
        column_resources, scale_collective = _distributed_int8_columns(
            runtime,
            loader,
            f"{prefix}.o_proj.weight",
            degree=degree,
            worker_prefix=f"mla.layer-{layer}.p{degree}",
            quantizer=quantizer,
        )
        partials: list[np.ndarray] = []
        worker_records: list[dict[str, Any]] = []
        for stripe, (output_resources, output_weight, column_range) in enumerate(column_resources):
            resources = DeviceResources(runtime)
            workers.append(resources)
            worker_id = f"mla.layer-{layer}.p{degree}.worker-{stripe:02d}"
            heads = balanced_range(HEADS, degree, stripe)
            local_heads = heads.stop - heads.start
            q_start = heads.start * MLA_QUERY_PER_HEAD
            q_stop = heads.stop * MLA_QUERY_PER_HEAD
            context_start = heads.start * VALUE_DIMENSION
            context_stop = heads.stop * VALUE_DIMENSION
            kv_start = heads.start * MLA_KV_B_PER_HEAD
            kv_stop = heads.stop * MLA_KV_B_PER_HEAD
            if column_range != (context_start, context_stop):
                raise RuntimeError("MLA output column stripe is not head-compatible")
            q_b = _int8_rows(
                runtime,
                resources,
                loader,
                f"{prefix}.q_b_proj.weight",
                worker_id=worker_id,
                start=q_start,
                stop=q_stop,
                purpose="mla_query_head_rows",
                quantizer=quantizer,
            )
            kv_b = _int8_rows(
                runtime,
                resources,
                loader,
                f"{prefix}.kv_b_proj.weight",
                worker_id=worker_id,
                start=kv_start,
                stop=kv_stop,
                purpose="mla_kv_b_head_rows",
                quantizer=quantizer,
            )
            gate = _int8_rows(
                runtime,
                resources,
                loader,
                f"{prefix}.g_proj.weight",
                worker_id=worker_id,
                start=context_start,
                stop=context_stop,
                purpose="mla_gate_head_rows",
                quantizer=quantizer,
            )
            kv_norm = resources.upload(
                _bf16_f32(
                    loader.reviewed_small(
                        f"{prefix}.kv_a_layernorm.weight",
                        worker_id=worker_id,
                        purpose="replicated_small_mla_kv_norm",
                    )
                )
            )
            query_low_device = resources.upload(query_low_host)
            compressed_device = resources.upload(compressed_host)
            query = resources.allocate(rows * local_heads * MLA_QUERY_PER_HEAD)
            mla_gate = resources.allocate(rows * local_heads * VALUE_DIMENSION)
            context = resources.allocate(rows * local_heads * VALUE_DIMENSION)
            latent_cache = resources.allocate(maximum_context * KV_LORA)
            rope_cache = resources.allocate(maximum_context * QUERY_ROPE)
            output = resources.allocate(rows * HIDDEN)
            zero_latent = np.zeros((maximum_context, KV_LORA), dtype=np.float32)
            zero_rope = np.zeros((maximum_context, QUERY_ROPE), dtype=np.float32)

            def operation() -> None:
                runtime.upload_activation(latent_cache, zero_latent)
                runtime.upload_activation(rope_cache, zero_rope)
                runtime.execute_dense(q_b, query, query_low_device, rows)
                runtime.execute_dense(gate, mla_gate, input_device, rows)
                for row in range(rows):
                    runtime.execute_mla_cache_append(
                        _pointer_offset(latent_cache, row * KV_LORA),
                        _pointer_offset(rope_cache, row * QUERY_ROPE),
                        _pointer_offset(compressed_device, row * (KV_LORA + QUERY_ROPE)),
                        kv_norm,
                        kv_lora=KV_LORA,
                        rope_dimension=QUERY_ROPE,
                        epsilon=1e-5,
                    )
                    runtime.execute_mla_absorb(
                        kv_b,
                        _pointer_offset(context, row * local_heads * VALUE_DIMENSION),
                        _pointer_offset(query, row * local_heads * MLA_QUERY_PER_HEAD),
                        latent_cache,
                        rope_cache,
                        heads=local_heads,
                        query_nope=QUERY_NOPE,
                        query_rope=QUERY_ROPE,
                        value_dimension=VALUE_DIMENSION,
                        kv_lora=KV_LORA,
                        context_length=row + 1,
                        attention_scale=1.0 / math.sqrt(QUERY_NOPE + QUERY_ROPE),
                    )
                    runtime.execute_mla_gate(
                        _pointer_offset(context, row * local_heads * VALUE_DIMENSION),
                        _pointer_offset(mla_gate, row * local_heads * VALUE_DIMENSION),
                        local_heads * VALUE_DIMENSION,
                    )
                runtime.execute_dense(output_weight, output, context, rows)
                runtime.synchronize()

            operation()
            partial = runtime.download_activation(output, (rows, HIDDEN))
            measurement = _measure(
                operation, runtime=runtime, warmup=warmup, iterations=iterations
            )
            state_parts = {
                "latent_cache": runtime.download_activation(
                    latent_cache, (maximum_context, KV_LORA)
                ),
                "rope_cache": runtime.download_activation(
                    rope_cache, (maximum_context, QUERY_ROPE)
                ),
            }
            state_digest = hashlib.sha256()
            for state_name, state_values in state_parts.items():
                state_digest.update(state_name.encode("utf-8"))
                state_digest.update(np.ascontiguousarray(state_values).tobytes())
            partials.append(partial)
            worker_records.append(
                {
                    "worker_id": worker_id,
                    "stripe_index": stripe,
                    "head_start": heads.start,
                    "head_stop": heads.stop,
                    "runtime_weight_bytes": resources.weight_bytes
                    + output_resources.weight_bytes,
                    "persistent_state_bytes": zero_latent.nbytes + zero_rope.nbytes,
                    "state_fingerprint": "sha256:" + state_digest.hexdigest(),
                    "wall": measurement["wall"],
                    "physical_launches": rows * 3 + 3,
                    "network_visible_partial_outputs": 1,
                }
            )
        output = np.sum(
            np.stack(partials, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        return output, {
            "attention_type": "Gated_MLA",
            "layer": layer,
            "rows": rows,
            "stripe_degree": degree,
            "common_owner": common_worker,
            "common_projection": common_record,
            "output_int8_scale_collective": scale_collective,
            "workers": worker_records,
            "independent_worker_compute_ceiling_ms": max(
                item["wall"]["p50_ms"] for item in worker_records
            ),
            "sequential_worker_compute_ms": sum(
                item["wall"]["p50_ms"] for item in worker_records
            ),
            "one_layer_reduction": True,
            "output_fingerprint": _array_fingerprint(output),
        }
    finally:
        for resources in reversed(workers):
            resources.close()
        for resources, _handle, _range in reversed(column_resources):
            resources.close()
        common.close()


def benchmark_attention_stripes(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    kda_shard_library: Path,
    *,
    kda_layer: int = 89,
    mla_layer: int = 91,
    degrees: Sequence[int] = (4, 8, 16, 32),
    rows_sweep: Sequence[int] = (1, 2, 4),
    warmup: int = 1,
    iterations: int = 5,
    device: int = 0,
) -> dict[str, Any]:
    catalog = CheckpointCatalog(checkpoint)
    loader = DirectShardLoader(catalog)
    runtime = _CudaRuntime(cuda_library, device)
    kda_shard_kernel = KdaShardKernel(kda_shard_library, device)
    quantizer = GpuShardQuantizer(kda_shard_library, device)
    runtime.set_telemetry("minimal")
    results: list[dict[str, Any]] = []
    try:
        for layer, executor in (
            (kda_layer, execute_kda_attention),
            (mla_layer, execute_mla_attention),
        ):
            for rows in rows_sweep:
                normalized = normalized_real_inputs(
                    catalog, loader, oracle_trace, layer=layer, rows=rows
                )
                reference: np.ndarray | None = None
                for degree in degrees:
                    arguments: dict[str, Any] = {}
                    if executor is execute_kda_attention:
                        arguments["shard_kernel"] = kda_shard_kernel
                    arguments["quantizer"] = quantizer
                    output, record = executor(
                        runtime,
                        loader,
                        normalized,
                        layer=layer,
                        degree=degree,
                        warmup=warmup,
                        iterations=iterations,
                        **arguments,
                    )
                    if reference is None:
                        reference = output
                    metrics = _numerical_metrics(reference, output)
                    record["reference_degree"] = int(degrees[0])
                    record["cross_degree_metrics"] = metrics
                    record["cross_degree_pass"] = (
                        float(metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                    )
                    results.append(record)
        return {
            "schema_version": "experiment-019-attention-stripe-physical-v1",
            "status": "PASS" if all(row["cross_degree_pass"] for row in results) else "FAIL",
            "results": results,
            "checkpoint_read_audit": loader.audit,
            "direct_loader_gate": {
                "full_material_tensors_over_review_threshold": sum(
                    bool(row["full_source_tensor_materialized"])
                    and int(row["source_tensor_bytes"]) > 32 * 1024 * 1024
                    for row in loader.audit
                ),
                "request_count": len(loader.audit),
                "bytes_read": sum(int(row["bytes_read"]) for row in loader.audit),
            },
        }
    finally:
        runtime.close()


__all__ = [
    "benchmark_attention_stripes",
    "execute_kda_attention",
    "execute_mla_attention",
    "KdaShardKernel",
    "normalized_real_inputs",
]
