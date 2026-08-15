"""Production-native resident Kimi K3 primitives used by ``EXECUTE_SHARD``.

The original E022 dispatcher authenticated and validated shard requests, but its
experiment runner registered identity callables.  This module binds that same
dispatcher boundary to the CUDA primitives already used by the canonical K3
execution graph.  Checkpoint slicing, quantization, handle creation, and buffer
allocation happen in constructors; ``__call__`` performs only steady-state
input transfer, native execution, state mutation, result transfer, and response
materialization.

These classes deliberately expose one physical shard.  They never call a whole
layer implementation and therefore cannot silently fall back to one.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime
from swarm_inference.execution.kimi_k3_graph_runtime import _pointer_offset
from swarm_inference.experiments.experiment_019.attention import (
    HEAD_DIMENSION,
    HEADS,
    HIDDEN,
    KV_LORA,
    MLA_KV_B_PER_HEAD,
    MLA_QUERY_PER_HEAD,
    QUERY_LORA,
    QUERY_NOPE,
    QUERY_ROPE,
    VALUE_DIMENSION,
    DeviceResources,
    KdaShardKernel,
    _bf16_f32,
    _bf16_or_f32,
    _distributed_int8_columns,
    _int4_columns,
    _int4_rows,
    _int8_rows,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.physical import (
    LATENT,
    ROUTED_EXPERTS,
    TOPK,
    _ResidentHandles,
    _upload_stripe_experts,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_020.expert_grouped import GroupedTop16Runtime
from swarm_inference.experiments.experiment_022.native_dispatch import ShardRequest
from swarm_inference.model.mxfp4 import MXFP4Tensor


def _sha256_arrays(parts: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        source = np.ascontiguousarray(part)
        digest.update(str(source.shape).encode("ascii"))
        digest.update(source.dtype.str.encode("ascii"))
        digest.update(source.tobytes())
    return "sha256:" + digest.hexdigest()


def prepare_complete_expert_stripe_banks(
    runtime: _CudaRuntime,
    loader: DirectShardLoader,
    *,
    layer: int,
    degree: int,
    experts: Sequence[int] = tuple(range(ROUTED_EXPERTS)),
    worker_prefix: str | None = None,
) -> tuple[_ResidentHandles, ...]:
    """Prepare every exact expert stripe with one checkpoint read per tensor.

    The old local correctness harness reopened the same expert tensor once for
    every stripe.  A production distributor reads immutable model state once
    and assigns disjoint tensor-axis views to workers.  This helper models that
    startup behavior honestly: it materializes one complete expert at a time,
    slices it on the same MXFP4 boundaries as ``DirectShardLoader.expert_stripe``,
    and uploads a distinct native handle triple for every worker.  No complete
    expert remains as a runtime handle and no preparation enters steady-state
    timing.
    """

    if degree not in (2, 4, 8, 16):
        raise ValueError("expert stripe-bank degree must be 2/4/8/16")
    prefix = worker_prefix or f"expert.layer-{layer}.p{degree}"
    handles_by_stripe: list[
        dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]]
    ] = [dict() for _ in range(degree)]
    bytes_by_stripe = [0 for _ in range(degree)]

    def complete(name: str) -> np.ndarray:
        record = loader.catalog.record(name)
        return loader.load(
            name,
            worker_id=f"{prefix}.startup-distributor",
            purpose="single_read_then_exact_resident_stripe_distribution",
            axis=0,
            start=0,
            stop=record.shape[0],
        )

    try:
        for expert_value in experts:
            expert = int(expert_value)
            tensor_prefix = (
                f"language_model.model.layers.{layer}.block_sparse_moe."
                f"experts.{expert}"
            )
            w1_packed = complete(f"{tensor_prefix}.w1.weight_packed")
            w1_scales = complete(f"{tensor_prefix}.w1.weight_scale")
            w2_packed = complete(f"{tensor_prefix}.w2.weight_packed")
            w2_scales = complete(f"{tensor_prefix}.w2.weight_scale")
            w3_packed = complete(f"{tensor_prefix}.w3.weight_packed")
            w3_scales = complete(f"{tensor_prefix}.w3.weight_scale")
            for stripe in range(degree):
                partition = balanced_range(3072, degree, stripe, quantum=32)
                local = partition.stop - partition.start
                gate = MXFP4Tensor(
                    packed=np.ascontiguousarray(
                        w1_packed[partition.start : partition.stop]
                    ),
                    scales=np.ascontiguousarray(
                        w1_scales[partition.start : partition.stop]
                    ),
                    input_dimension=LATENT,
                    output_dimension=local,
                )
                up = MXFP4Tensor(
                    packed=np.ascontiguousarray(
                        w3_packed[partition.start : partition.stop]
                    ),
                    scales=np.ascontiguousarray(
                        w3_scales[partition.start : partition.stop]
                    ),
                    input_dimension=LATENT,
                    output_dimension=local,
                )
                down = MXFP4Tensor(
                    packed=np.ascontiguousarray(
                        w2_packed[
                            :, partition.start // 2 : partition.stop // 2
                        ]
                    ),
                    scales=np.ascontiguousarray(
                        w2_scales[
                            :, partition.start // 32 : partition.stop // 32
                        ]
                    ),
                    input_dimension=local,
                    output_dimension=LATENT,
                )
                triple = tuple(runtime.upload(tensor) for tensor in (gate, up, down))
                handles_by_stripe[stripe][expert] = triple  # type: ignore[assignment]
                bytes_by_stripe[stripe] += sum(
                    runtime.tensor_bytes(handle) for handle in triple
                )
        banks = tuple(
            _ResidentHandles(runtime, handles, bytes_by_stripe[stripe])
            for stripe, handles in enumerate(handles_by_stripe)
        )
        if any(len(bank.handles) != len(experts) for bank in banks):
            raise RuntimeError("complete expert stripe-bank preparation lost experts")
        return banks
    except BaseException:
        for handles in handles_by_stripe:
            for triple in handles.values():
                for handle in triple:
                    runtime.release_tensor(handle)
            handles.clear()
        raise


class _PreparedPrimitive:
    """Shared accounting contract for a prepared native assignment."""

    native = True
    invocation_count = 0
    native_primitive = ""

    def __init__(self) -> None:
        self.invocation_count = 0
        self.startup: dict[str, Any] = {}
        self.last_execution: dict[str, Any] = {}
        self.resident_bytes = 0

    def _begin_startup(self) -> int:
        return time.perf_counter_ns()

    def _finish_startup(
        self,
        started: int,
        loader: DirectShardLoader,
        *,
        persistent_state_bytes: int,
        runtime_weight_bytes: int,
        buffer_bytes: int,
    ) -> None:
        self.resident_bytes = runtime_weight_bytes + persistent_state_bytes + buffer_bytes
        self.startup = {
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            "checkpoint_reads": len(loader.audit),
            "checkpoint_bytes": sum(int(row["bytes_read"]) for row in loader.audit),
            "runtime_weight_bytes": runtime_weight_bytes,
            "persistent_state_bytes": persistent_state_bytes,
            "reusable_buffer_bytes": buffer_bytes,
            "resident_bytes": self.resident_bytes,
        }

    def _record(
        self,
        *,
        started: int,
        cuda_ms: float,
        input_copy_ms: float,
        output_copy_ms: float,
        reads_before: int,
        loader: DirectShardLoader,
        launches: int,
        state_mutated: bool,
    ) -> None:
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        self.invocation_count += 1
        self.last_execution = {
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "host_ms": max(0.0, wall_ms - cuda_ms),
            "input_copy_ms": input_copy_ms,
            "output_copy_ms": output_copy_ms,
            "physical_launches": launches,
            "state_mutated": state_mutated,
            "checkpoint_reads_in_timed_region": len(loader.audit) - reads_before,
            "whole_layer_fallback": False,
        }


class PreparedKdaShard(_PreparedPrimitive):
    """One resident KDA head stripe backed by the E019 native shard kernel."""

    native_primitive = "exp019_kda_shard_short_window_dev"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        shard_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        precomputed_common: bool = False,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.precomputed_common = precomputed_common
        if precomputed_common:
            self.native_primitive = (
                "exp019_kda_shard_short_window_dev_after_common_projection"
            )
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.runtime.set_fused_gate_up(True)
        self.kernel = KdaShardKernel(shard_library, device)
        self.quantizer = GpuShardQuantizer(shard_library, device)
        self.resources = DeviceResources(self.runtime)
        prefix = f"language_model.model.layers.{layer}.self_attn"
        worker = f"kda.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        heads = balanced_range(HEADS, degree, shard_index)
        self.local_heads = heads.stop - heads.start
        start = heads.start * HEAD_DIMENSION
        stop = heads.stop * HEAD_DIMENSION
        self.local_projection = stop - start

        self.f_a: ctypes.c_void_p | None = None
        if not precomputed_common:
            source = self.loader.reviewed_small(
                f"{prefix}.f_a_proj.weight",
                worker_id=worker,
                purpose="resident_common_kda_decay_low_projection",
            )
            self.f_a = self.resources.tensor(
                self.runtime.upload_float32(_bf16_f32(source))
            )
        self.weights: dict[str, ctypes.c_void_p] = {}
        for role in ("q", "k", "v", "g"):
            self.weights[role] = _int4_rows(
                self.runtime,
                self.resources,
                self.loader,
                f"{prefix}.{role}_proj.weight",
                worker_id=worker,
                start=start,
                stop=stop,
                purpose=f"resident_kda_{role}_head_rows",
                quantizer=self.quantizer,
            )
        f_b = self.loader.load(
            f"{prefix}.f_b_proj.weight",
            worker_id=worker,
            purpose="resident_kda_decay_head_rows",
            axis=0,
            start=start,
            stop=stop,
        )
        self.weights["f_b"] = self.resources.tensor(
            self.runtime.upload_float32(_bf16_f32(f_b))
        )
        beta = self.loader.load(
            f"{prefix}.b_proj.weight",
            worker_id=worker,
            purpose="resident_kda_beta_head_rows",
            axis=0,
            start=heads.start,
            stop=heads.stop,
        )
        self.weights["beta"] = self.resources.tensor(
            self.runtime.upload_float32(_bf16_f32(beta))
        )
        self.weights["output"] = _int4_columns(
            self.runtime,
            self.resources,
            self.loader,
            f"{prefix}.o_proj.weight",
            worker_id=worker,
            start=start,
            stop=stop,
            purpose="resident_kda_output_projection_columns",
            quantizer=self.quantizer,
        )
        for role in ("q", "k", "v"):
            conv = self.loader.load(
                f"{prefix}.{role}_conv1d.weight",
                worker_id=worker,
                purpose=f"resident_kda_{role}_conv_channels",
                axis=0,
                start=start,
                stop=stop,
            )
            self.weights[f"conv_{role}"] = self.resources.upload(
                _bf16_or_f32(conv).reshape(-1)
            )
        dt = self.loader.load(
            f"{prefix}.dt_bias",
            worker_id=worker,
            purpose="resident_kda_dt_head_channels",
            axis=0,
            start=start,
            stop=stop,
        )
        self.weights["dt"] = self.resources.upload(_bf16_or_f32(dt).reshape(-1))
        a_log = self.loader.load(
            f"{prefix}.A_log",
            worker_id=worker,
            purpose="resident_kda_state_decay_heads",
            axis=0,
            start=heads.start,
            stop=heads.stop,
        )
        self.weights["a"] = self.resources.upload(
            np.exp(_bf16_or_f32(a_log)).astype(np.float32)
        )
        output_norm = self.loader.reviewed_small(
            f"{prefix}.o_norm.weight",
            worker_id=worker,
            purpose="resident_small_kda_output_norm",
        )
        self.weights["output_norm"] = self.resources.upload(
            _bf16_or_f32(output_norm).reshape(-1)
        )

        self.input = self.resources.allocate(max_rows * HIDDEN)
        self.decay_low = self.resources.allocate(max_rows * HEAD_DIMENSION)
        self.q = self.resources.allocate(max_rows * self.local_projection)
        self.k = self.resources.allocate(max_rows * self.local_projection)
        self.v = self.resources.allocate(max_rows * self.local_projection)
        self.gate = self.resources.allocate(max_rows * self.local_projection)
        self.decay = self.resources.allocate(max_rows * self.local_projection)
        self.beta = self.resources.allocate(max_rows * self.local_heads)
        self.core = self.resources.allocate(max_rows * self.local_projection)
        self.output = self.resources.allocate(max_rows * HIDDEN)
        self.state = self.resources.allocate(
            self.local_heads * HEAD_DIMENSION * HEAD_DIMENSION
        )
        self.window_q = self.resources.allocate(self.local_projection * 4)
        self.window_k = self.resources.allocate(self.local_projection * 4)
        self.window_v = self.resources.allocate(self.local_projection * 4)
        self._zero_state = np.zeros(
            (self.local_heads, HEAD_DIMENSION, HEAD_DIMENSION), dtype=np.float32
        )
        self._zero_window = np.zeros((self.local_projection, 4), dtype=np.float32)
        self.reset_state()
        state_bytes = self._zero_state.nbytes + 3 * self._zero_window.nbytes
        buffer_elements = max_rows * (
            HIDDEN * 2
            + HEAD_DIMENSION
            + self.local_projection * 6
            + self.local_heads
        )
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=state_bytes,
            runtime_weight_bytes=self.resources.weight_bytes,
            buffer_bytes=buffer_elements * 4,
        )

    def reset_state(self) -> None:
        self.runtime.upload_activation(self.state, self._zero_state)
        for pointer in (self.window_q, self.window_k, self.window_v):
            self.runtime.upload_activation(pointer, self._zero_window)
        self.runtime.synchronize()

    def state_fingerprint(self) -> str:
        return _sha256_arrays(
            [
                self.runtime.download_activation(
                    self.state,
                    (self.local_heads, HEAD_DIMENSION, HEAD_DIMENSION),
                ),
                self.runtime.download_activation(
                    self.window_q, (self.local_projection, 4)
                ),
                self.runtime.download_activation(
                    self.window_k, (self.local_projection, 4)
                ),
                self.runtime.download_activation(
                    self.window_v, (self.local_projection, 4)
                ),
            ]
        )

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("KDA request does not match resident assignment")
        expected_width = HIDDEN + HEAD_DIMENSION if self.precomputed_common else HIDDEN
        if source.shape != (rows, expected_width) or rows > self.max_rows:
            raise ValueError("KDA resident input geometry is invalid")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(
            self.input,
            np.ascontiguousarray(source[:, :HIDDEN], dtype=np.float32),
        )
        if self.precomputed_common:
            self.runtime.upload_activation(
                self.decay_low,
                np.ascontiguousarray(source[:, HIDDEN:], dtype=np.float32),
            )
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.runtime.profile_begin()
        if self.f_a is not None:
            self.runtime.execute_dense(self.f_a, self.decay_low, self.input, rows)
        for role, target in (("q", self.q), ("k", self.k), ("v", self.v), ("g", self.gate)):
            self.runtime.execute_dense(self.weights[role], target, self.input, rows)
        self.runtime.execute_dense(self.weights["f_b"], self.decay, self.decay_low, rows)
        self.runtime.execute_dense(self.weights["beta"], self.beta, self.input, rows)
        self.kernel.execute(
            self.core,
            self.q,
            self.k,
            self.v,
            self.gate,
            self.decay,
            self.beta,
            self.weights["conv_q"],
            self.weights["conv_k"],
            self.weights["conv_v"],
            self.window_q,
            self.window_k,
            self.window_v,
            self.state,
            self.weights["dt"],
            self.weights["a"],
            self.weights["output_norm"],
            rows=rows,
            heads=self.local_heads,
        )
        self.runtime.execute_dense(self.weights["output"], self.output, self.core, rows)
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, HIDDEN))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=8 if self.precomputed_common else 9,
            state_mutated=True,
        )
        return output

    def close(self) -> None:
        self.resources.close()
        self.runtime.close()


class PreparedMlaShard(_PreparedPrimitive):
    """One resident Gated-MLA head stripe with persistent KV/rope caches."""

    native_primitive = "kimi_cuda_mla_absorb_and_output_shard"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        quantizer_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        maximum_context: int = 256,
        precomputed_common: bool = False,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.maximum_context = maximum_context
        self.precomputed_common = precomputed_common
        if precomputed_common:
            self.native_primitive = (
                "kimi_cuda_mla_absorb_and_output_shard_after_common_projection"
            )
        self.position = 0
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.quantizer = GpuShardQuantizer(quantizer_library, device)
        self.resources = DeviceResources(self.runtime)
        prefix = f"language_model.model.layers.{layer}.self_attn"
        worker = f"mla.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        self.q_a: ctypes.c_void_p | None = None
        self.kv_a: ctypes.c_void_p | None = None
        self.query_norm: ctypes.c_void_p | None = None
        if not precomputed_common:
            q_a_source = self.loader.reviewed_small(
                f"{prefix}.q_a_proj.weight",
                worker_id=worker,
                purpose="resident_mla_query_low_projection",
            )
            kv_a_source = self.loader.reviewed_small(
                f"{prefix}.kv_a_proj_with_mqa.weight",
                worker_id=worker,
                purpose="resident_mla_kv_low_projection",
            )
            self.q_a = self.resources.tensor(
                self.runtime.upload_int8(
                    self.quantizer.row_int8(q_a_source, owner=worker)
                )
            )
            self.kv_a = self.resources.tensor(
                self.runtime.upload_int8(
                    self.quantizer.row_int8(kv_a_source, owner=worker)
                )
            )
            self.query_norm = self.resources.upload(
                _bf16_f32(
                    self.loader.reviewed_small(
                        f"{prefix}.q_a_layernorm.weight",
                        worker_id=worker,
                        purpose="resident_small_mla_query_norm",
                    )
                )
            )
        heads = balanced_range(HEADS, degree, shard_index)
        self.local_heads = heads.stop - heads.start
        q_start = heads.start * MLA_QUERY_PER_HEAD
        q_stop = heads.stop * MLA_QUERY_PER_HEAD
        context_start = heads.start * VALUE_DIMENSION
        context_stop = heads.stop * VALUE_DIMENSION
        kv_start = heads.start * MLA_KV_B_PER_HEAD
        kv_stop = heads.stop * MLA_KV_B_PER_HEAD
        self.local_context = context_stop - context_start
        self.q_b = _int8_rows(
            self.runtime,
            self.resources,
            self.loader,
            f"{prefix}.q_b_proj.weight",
            worker_id=worker,
            start=q_start,
            stop=q_stop,
            purpose="resident_mla_query_head_rows",
            quantizer=self.quantizer,
        )
        self.kv_b = _int8_rows(
            self.runtime,
            self.resources,
            self.loader,
            f"{prefix}.kv_b_proj.weight",
            worker_id=worker,
            start=kv_start,
            stop=kv_stop,
            purpose="resident_mla_kv_head_rows",
            quantizer=self.quantizer,
        )
        self.gate_weight = _int8_rows(
            self.runtime,
            self.resources,
            self.loader,
            f"{prefix}.g_proj.weight",
            worker_id=worker,
            start=context_start,
            stop=context_stop,
            purpose="resident_mla_gate_head_rows",
            quantizer=self.quantizer,
        )
        self.kv_norm = self.resources.upload(
            _bf16_f32(
                self.loader.reviewed_small(
                    f"{prefix}.kv_a_layernorm.weight",
                    worker_id=worker,
                    purpose="resident_small_mla_kv_norm",
                )
            )
        )
        output_resources, _collective = _distributed_int8_columns(
            self.runtime,
            self.loader,
            f"{prefix}.o_proj.weight",
            degree=degree,
            worker_prefix=f"mla.layer-{layer}.p{degree}",
            quantizer=self.quantizer,
        )
        selected: tuple[DeviceResources, ctypes.c_void_p, tuple[int, int]] | None = None
        for index, item in enumerate(output_resources):
            if index == shard_index:
                selected = item
            else:
                item[0].close()
        if selected is None or selected[2] != (context_start, context_stop):
            raise RuntimeError("MLA output stripe is not head compatible")
        selected_resources, selected_handle, _ = selected
        self.resources.handles.extend(selected_resources.handles)
        self.resources.weight_bytes += selected_resources.weight_bytes
        selected_resources.handles.clear()
        selected_resources.weight_bytes = 0
        selected_resources.close()
        self.output_weight = selected_handle

        self.input = self.resources.allocate(max_rows * HIDDEN)
        self.query_low = self.resources.allocate(max_rows * QUERY_LORA)
        self.compressed = self.resources.allocate(max_rows * (KV_LORA + QUERY_ROPE))
        self.query = self.resources.allocate(max_rows * self.local_heads * MLA_QUERY_PER_HEAD)
        self.mla_gate = self.resources.allocate(max_rows * self.local_context)
        self.context = self.resources.allocate(max_rows * self.local_context)
        self.latent_cache = self.resources.allocate(maximum_context * KV_LORA)
        self.rope_cache = self.resources.allocate(maximum_context * QUERY_ROPE)
        self.output = self.resources.allocate(max_rows * HIDDEN)
        self._zero_latent = np.zeros((maximum_context, KV_LORA), dtype=np.float32)
        self._zero_rope = np.zeros((maximum_context, QUERY_ROPE), dtype=np.float32)
        self.reset_state()
        state_bytes = self._zero_latent.nbytes + self._zero_rope.nbytes
        buffer_elements = max_rows * (
            HIDDEN * 2
            + QUERY_LORA
            + KV_LORA
            + QUERY_ROPE
            + self.local_heads * MLA_QUERY_PER_HEAD
            + 2 * self.local_context
        )
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=state_bytes,
            runtime_weight_bytes=self.resources.weight_bytes,
            buffer_bytes=buffer_elements * 4,
        )

    def reset_state(self) -> None:
        self.runtime.upload_activation(self.latent_cache, self._zero_latent)
        self.runtime.upload_activation(self.rope_cache, self._zero_rope)
        self.runtime.synchronize()
        self.position = 0

    def state_fingerprint(self) -> str:
        return _sha256_arrays(
            [
                self.runtime.download_activation(
                    self.latent_cache, (self.maximum_context, KV_LORA)
                ),
                self.runtime.download_activation(
                    self.rope_cache, (self.maximum_context, QUERY_ROPE)
                ),
            ]
        )

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("MLA request does not match resident assignment")
        common_width = QUERY_LORA + KV_LORA + QUERY_ROPE
        expected_width = HIDDEN + common_width if self.precomputed_common else HIDDEN
        if source.shape != (rows, expected_width) or rows > self.max_rows:
            raise ValueError("MLA resident input geometry is invalid")
        if self.position + rows > self.maximum_context:
            raise ValueError("MLA resident cache capacity exceeded")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(
            self.input,
            np.ascontiguousarray(source[:, :HIDDEN], dtype=np.float32),
        )
        if self.precomputed_common:
            query_stop = HIDDEN + QUERY_LORA
            self.runtime.upload_activation(
                self.query_low,
                np.ascontiguousarray(source[:, HIDDEN:query_stop], dtype=np.float32),
            )
            self.runtime.upload_activation(
                self.compressed,
                np.ascontiguousarray(source[:, query_stop:], dtype=np.float32),
            )
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.runtime.profile_begin()
        if self.q_a is not None and self.kv_a is not None and self.query_norm is not None:
            self.runtime.execute_dense(self.q_a, self.query_low, self.input, rows)
            self.runtime.execute_rmsnorm(
                self.query_low,
                self.query_low,
                self.query_norm,
                batch=rows,
                dimension=QUERY_LORA,
                epsilon=1e-5,
            )
            self.runtime.execute_dense(self.kv_a, self.compressed, self.input, rows)
        self.runtime.execute_dense(self.q_b, self.query, self.query_low, rows)
        self.runtime.execute_dense(self.gate_weight, self.mla_gate, self.input, rows)
        for row in range(rows):
            absolute = self.position + row
            self.runtime.execute_mla_cache_append(
                _pointer_offset(self.latent_cache, absolute * KV_LORA),
                _pointer_offset(self.rope_cache, absolute * QUERY_ROPE),
                _pointer_offset(self.compressed, row * (KV_LORA + QUERY_ROPE)),
                self.kv_norm,
                kv_lora=KV_LORA,
                rope_dimension=QUERY_ROPE,
                epsilon=1e-5,
            )
            self.runtime.execute_mla_absorb(
                self.kv_b,
                _pointer_offset(self.context, row * self.local_context),
                _pointer_offset(
                    self.query, row * self.local_heads * MLA_QUERY_PER_HEAD
                ),
                self.latent_cache,
                self.rope_cache,
                heads=self.local_heads,
                query_nope=QUERY_NOPE,
                query_rope=QUERY_ROPE,
                value_dimension=VALUE_DIMENSION,
                kv_lora=KV_LORA,
                context_length=absolute + 1,
                attention_scale=1.0 / math.sqrt(QUERY_NOPE + QUERY_ROPE),
            )
            self.runtime.execute_mla_gate(
                _pointer_offset(self.context, row * self.local_context),
                _pointer_offset(self.mla_gate, row * self.local_context),
                self.local_context,
            )
        self.runtime.execute_dense(self.output_weight, self.output, self.context, rows)
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, HIDDEN))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.position += rows
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=(3 if self.precomputed_common else 6) + rows * 3,
            state_mutated=True,
        )
        return output

    def close(self) -> None:
        self.resources.close()
        self.runtime.close()


class PreparedExpertStripe(_PreparedPrimitive):
    """One complete resident expert-bank stripe using grouped top-16 CUDA."""

    native_primitive = "e020_kimi_grouped_top16"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        grouped_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        experts: Sequence[int] = tuple(range(ROUTED_EXPERTS)),
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.runtime.set_fused_gate_up(True)
        self.grouped = GroupedTop16Runtime(grouped_library)
        self.resident: _ResidentHandles = _upload_stripe_experts(
            self.runtime,
            self.loader,
            layer=layer,
            experts=experts,
            degree=degree,
            stripe=shard_index,
            worker_id=f"expert.layer-{layer}.p{degree}.worker-{shard_index:02d}",
        )
        self.input = self.runtime.allocate(max_rows * LATENT * 4)
        self.route_weights = self.runtime.allocate(max_rows * TOPK * 4)
        self.output = self.runtime.allocate(max_rows * LATENT * 4)
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=self.resident.runtime_bytes,
            buffer_bytes=max_rows * (LATENT * 2 + TOPK) * 4,
        )

    def state_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(b"stateless-expert-stripe").hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        expected = LATENT + 2 * TOPK
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("expert request does not match resident assignment")
        if source.shape != (rows, expected) or rows > self.max_rows:
            raise ValueError("expert stripe input must pack latent, route ids, and weights")
        activation = np.ascontiguousarray(source[:, :LATENT])
        routes = np.rint(source[:, LATENT : LATENT + TOPK]).astype(np.int32)
        weights = np.ascontiguousarray(source[:, LATENT + TOPK :])
        if any(int(value) not in self.resident.handles for value in routes.reshape(-1)):
            raise ValueError("expert route is not resident on this assignment")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, activation)
        self.runtime.upload_activation(self.route_weights, weights)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        cuda_ms = self.grouped.execute(
            self.resident,
            routes,
            self.output,
            self.input,
            self.route_weights,
        )
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, LATENT))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=self.grouped.physical_launches,
            state_mutated=False,
        )
        self.last_execution.update(
            {
                "route_ids_sha256": _sha256_arrays((routes,)),
                "route_weights_sha256": _sha256_arrays((weights,)),
                "ordered_route_ids": routes.tolist(),
            }
        )
        return output

    def close(self) -> None:
        self.runtime.free(self.output)
        self.runtime.free(self.route_weights)
        self.runtime.free(self.input)
        self.resident.close()
        self.grouped.close()
        self.runtime.close()


class PreparedSharedExpertShard(_PreparedPrimitive):
    """One resident intermediate-width stripe of the shared expert MLP."""

    native_primitive = "coli_cuda_execute_resident_shared_expert_stripe"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        quantizer_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        intermediate: int = 6144,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.runtime.set_fused_gate_up(True)
        self.quantizer = GpuShardQuantizer(quantizer_library, device)
        self.resources = DeviceResources(self.runtime)
        shard = balanced_range(intermediate, degree, shard_index, quantum=64)
        worker = f"shared.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        prefix = f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts"
        sources = (
            self.loader.load(
                f"{prefix}.gate_proj.weight",
                worker_id=worker,
                purpose="resident_shared_gate_rows",
                axis=0,
                start=shard.start,
                stop=shard.stop,
            ),
            self.loader.load(
                f"{prefix}.up_proj.weight",
                worker_id=worker,
                purpose="resident_shared_up_rows",
                axis=0,
                start=shard.start,
                stop=shard.stop,
            ),
            self.loader.load(
                f"{prefix}.down_proj.weight",
                worker_id=worker,
                purpose="resident_shared_down_columns",
                axis=1,
                start=shard.start,
                stop=shard.stop,
            ),
        )
        self.handles = tuple(
            self.resources.tensor(
                self.runtime.upload_grouped_int4(
                    self.quantizer.grouped_int4(source, owner=worker)
                )
            )
            for source in sources
        )
        self.input = self.resources.allocate(max_rows * HIDDEN)
        self.output = self.resources.allocate(max_rows * HIDDEN)
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=self.resources.weight_bytes,
            buffer_bytes=max_rows * HIDDEN * 2 * 4,
        )

    def state_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(b"stateless-shared-expert").hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("shared expert request does not match resident assignment")
        if source.shape != (rows, HIDDEN) or rows > self.max_rows:
            raise ValueError("shared expert resident input geometry is invalid")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, source)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.runtime.profile_begin()
        self.runtime.execute_resident(self.handles, self.output, self.input, rows)
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, HIDDEN))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=1,
            state_mutated=False,
        )
        return output

    def close(self) -> None:
        self.resources.close()
        self.runtime.close()


class PreparedProjectionShard(_PreparedPrimitive):
    """Resident column stripe of K3's routed latent-up projection."""

    native_primitive = "coli_cuda_execute_dense_routed_latent_up_column_shard"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        quantizer_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.quantizer = GpuShardQuantizer(quantizer_library, device)
        self.resources = DeviceResources(self.runtime)
        self.shard = balanced_range(LATENT, degree, shard_index, quantum=64)
        worker = f"projection.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        source = self.loader.load(
            f"language_model.model.layers.{layer}.block_sparse_moe.routed_expert_up_proj.weight",
            worker_id=worker,
            purpose="resident_routed_latent_up_columns",
            axis=1,
            start=self.shard.start,
            stop=self.shard.stop,
        )
        self.weight = self.resources.tensor(
            self.runtime.upload_grouped_int4(
                self.quantizer.grouped_int4(source, owner=worker)
            )
        )
        self.local_input = self.resources.allocate(
            max_rows * (self.shard.stop - self.shard.start)
        )
        self.output = self.resources.allocate(max_rows * HIDDEN)
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=self.resources.weight_bytes,
            buffer_bytes=max_rows
            * (HIDDEN + self.shard.stop - self.shard.start)
            * 4,
        )

    def state_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(b"stateless-projection").hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        local = self.shard.stop - self.shard.start
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("projection request does not match resident assignment")
        if source.shape != (rows, local) or rows > self.max_rows:
            raise ValueError("projection resident input geometry is invalid")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.local_input, source)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.runtime.profile_begin()
        self.runtime.execute_dense(self.weight, self.output, self.local_input, rows)
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, HIDDEN))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=1,
            state_mutated=False,
        )
        return output

    def close(self) -> None:
        self.resources.close()
        self.runtime.close()


class PreparedLatentDownShard(_PreparedPrimitive):
    """Resident row stripe of K3's routed latent-down projection."""

    native_primitive = "coli_cuda_execute_dense_routed_latent_down_row_shard"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        quantizer_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.quantizer = GpuShardQuantizer(quantizer_library, device)
        self.resources = DeviceResources(self.runtime)
        self.shard = balanced_range(LATENT, degree, shard_index, quantum=64)
        worker = f"latent-down.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        source = self.loader.load(
            f"language_model.model.layers.{layer}.block_sparse_moe."
            "routed_expert_down_proj.weight",
            worker_id=worker,
            purpose="resident_routed_latent_down_rows",
            axis=0,
            start=self.shard.start,
            stop=self.shard.stop,
        )
        self.weight = self.resources.tensor(
            self.runtime.upload_grouped_int4(
                self.quantizer.grouped_int4(source, owner=worker)
            )
        )
        self.input = self.resources.allocate(max_rows * HIDDEN)
        self.output = self.resources.allocate(
            max_rows * (self.shard.stop - self.shard.start)
        )
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=self.resources.weight_bytes,
            buffer_bytes=max_rows
            * (HIDDEN + self.shard.stop - self.shard.start)
            * 4,
        )

    def state_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(b"stateless-latent-down").hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        local = self.shard.stop - self.shard.start
        if (
            request.layer != self.layer
            or request.degree != self.degree
            or request.shard_index != self.shard_index
        ):
            raise ValueError("latent-down request does not match resident assignment")
        if source.shape != (rows, HIDDEN) or rows > self.max_rows:
            raise ValueError("latent-down resident input geometry is invalid")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, source)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self.runtime.profile_begin()
        self.runtime.execute_dense(self.weight, self.output, self.input, rows)
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, local))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=1,
            state_mutated=False,
        )
        return output

    def close(self) -> None:
        self.resources.close()
        self.runtime.close()


class PreparedReductionContribution(_PreparedPrimitive):
    """Resident CUDA reduction over worker contributions."""

    native_primitive = "coli_cuda_copy_add_reduction"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int = 0,
        participants: int = 4,
        dimension: int = HIDDEN,
        max_rows: int = 4,
        device: int = 0,
    ) -> None:
        super().__init__()
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.participants = participants
        self.dimension = dimension
        self.max_rows = max_rows
        # Keep the same audit interface as weight-bearing primitives.  The
        # reduction owns no checkpoint state and therefore has zero reads.
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.input = self.runtime.allocate(participants * max_rows * dimension * 4)
        self.output = self.runtime.allocate(max_rows * dimension * 4)
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=0,
            buffer_bytes=(participants + 1) * max_rows * dimension * 4,
        )

    def state_fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(b"stateless-reduction").hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        if source.ndim != 3:
            raise ValueError("reduction input must be participants x rows x dimension")
        participants, rows, dimension = source.shape
        if request.layer != self.layer or request.degree != self.degree or request.shard_index != self.shard_index:
            raise ValueError("reduction request does not match resident assignment")
        if participants != self.participants or dimension != self.dimension or rows > self.max_rows:
            raise ValueError("reduction resident input geometry is invalid")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, source)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        elements = rows * dimension
        self.runtime.profile_begin()
        self.runtime.execute_copy(self.output, self.input, elements)
        for participant in range(1, participants):
            self.runtime.execute_add(
                self.output,
                _pointer_offset(self.input, participant * elements),
                elements,
            )
        self.runtime.synchronize()
        cuda_ms = self.runtime.profile_end()
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, dimension))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=participants,
            state_mutated=False,
        )
        return output

    def close(self) -> None:
        self.runtime.free(self.output)
        self.runtime.free(self.input)
        self.runtime.close()


__all__ = [
    "PreparedExpertStripe",
    "PreparedKdaShard",
    "PreparedLatentDownShard",
    "PreparedMlaShard",
    "PreparedProjectionShard",
    "PreparedReductionContribution",
    "PreparedSharedExpertShard",
]
