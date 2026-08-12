"""Streamed, no-fallback CUDA execution of the complete real Kimi K3 graph.

This module deliberately keeps the experiment's local full-graph runner narrow:
one CUDA context is reused, while one layer's immutable weights are resident at
a time.  Mathematical operations execute through the certified CUDA ABI; host
memory is used only to carry layer boundaries and checkpoint request state when
the full 93-layer model cannot fit on the local GPU at once.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    KimiCudaError,
    _array_fingerprint,
    _array_fingerprint_streaming,
    _CudaRuntime,
    _GroupedInt4Tensor,
    _numerical_metrics,
    _quantize_bf16_rows_int8,
)
from swarm_inference.model.mxfp4 import MXFP4Tensor

SCHEMA_VERSION = "experiment-014-k3-cuda-streamed-graph-v1"


@dataclass(frozen=True, slots=True)
class _KimiConfig:
    hidden: int
    layers: int
    vocab: int
    first_dense: int
    dense_intermediate: int
    heads: int
    query_lora: int
    kv_lora: int
    query_nope: int
    query_rope: int
    value_dimension: int
    experts: int
    topk: int
    moe_intermediate: int
    latent: int
    shared_experts: int
    residual_block: int
    situ_beta: float
    situ_linear_beta: float
    epsilon: float
    kda_heads: int
    kda_head_dimension: int
    convolution_width: int
    gate_lower_bound: float
    kda_layers: frozenset[int]
    bos_token_id: int

    @property
    def kda_projection(self) -> int:
        return self.kda_heads * self.kda_head_dimension

    @property
    def query_dimension(self) -> int:
        return self.heads * (self.query_nope + self.query_rope)

    @property
    def context_dimension(self) -> int:
        return self.heads * self.value_dimension

    @property
    def attention_scale(self) -> float:
        return 1.0 / math.sqrt(self.query_nope + self.query_rope)


class _CheckpointReader:
    """Range-oriented Safetensors reader with one parsed header per shard."""

    def __init__(self, checkpoint: Path) -> None:
        self.root = checkpoint.expanduser().resolve()
        index_path = self.root / "model.safetensors.index.json"
        config_path = self.root / "config.json"
        if not index_path.is_file() or not config_path.is_file():
            raise KimiCudaError("Kimi checkpoint is missing config or Safetensors index")
        self.index_path = index_path
        self.config_path = config_path
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        weight_map = self.index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise KimiCudaError("Kimi checkpoint index has no weight_map")
        self.weight_map: dict[str, str] = {
            str(name): str(shard) for name, shard in weight_map.items()
        }
        self._headers: dict[str, tuple[int, dict[str, Any]]] = {}
        self.config = self._parse_config()

    def _parse_config(self) -> _KimiConfig:
        text = self.raw_config["text_config"]
        linear = text["linear_attn_config"]
        kda_layers = frozenset(int(value) - 1 for value in linear["kda_layers"])
        config = _KimiConfig(
            hidden=int(text["hidden_size"]),
            layers=int(text["num_hidden_layers"]),
            vocab=int(text["vocab_size"]),
            first_dense=int(text["first_k_dense_replace"]),
            dense_intermediate=int(text["intermediate_size"]),
            heads=int(text["num_attention_heads"]),
            query_lora=int(text["q_lora_rank"]),
            kv_lora=int(text["kv_lora_rank"]),
            query_nope=int(text["qk_nope_head_dim"]),
            query_rope=int(text["qk_rope_head_dim"]),
            value_dimension=int(text["v_head_dim"]),
            experts=int(text["num_experts"]),
            topk=int(text["num_experts_per_token"]),
            moe_intermediate=int(text["moe_intermediate_size"]),
            latent=int(text["routed_expert_hidden_size"]),
            shared_experts=int(text["num_shared_experts"]),
            residual_block=int(text["attn_res_block_size"]),
            situ_beta=float(text["activation_situ_beta"]),
            situ_linear_beta=float(text["activation_situ_linear_beta"]),
            epsilon=float(text["rms_norm_eps"]),
            kda_heads=int(linear["num_heads"]),
            kda_head_dimension=int(linear["head_dim"]),
            convolution_width=int(linear["short_conv_kernel_size"]),
            gate_lower_bound=float(linear["gate_lower_bound"]),
            kda_layers=kda_layers,
            bos_token_id=int(text["bos_token_id"]),
        )
        geometry = (
            config.hidden,
            config.layers,
            config.heads,
            config.kda_heads,
            config.kda_head_dimension,
            config.experts,
            config.topk,
        )
        if geometry != (7168, 93, 96, 96, 128, 896, 16):
            raise KimiCudaError(f"unexpected Kimi K3 geometry {geometry}")
        return config

    def _header(self, shard_name: str) -> tuple[int, dict[str, Any]]:
        cached = self._headers.get(shard_name)
        if cached is not None:
            return cached
        shard_path = self.root / shard_name
        if not shard_path.is_file():
            raise KimiCudaError(f"missing checkpoint shard: {shard_path}")
        with shard_path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise KimiCudaError(f"truncated Safetensors header: {shard_path}")
            header_size = struct.unpack("<Q", raw_size)[0]
            header = json.loads(handle.read(header_size).decode("utf-8"))
        result = (8 + int(header_size), header)
        self._headers[shard_name] = result
        return result

    def array(self, name: str) -> np.memmap:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise KimiCudaError(f"checkpoint index has no tensor {name}")
        payload_offset, header = self._header(shard_name)
        metadata = header.get(name)
        if not isinstance(metadata, dict):
            raise KimiCudaError(f"tensor {name} is absent from {shard_name}")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        dtype_name = str(metadata.get("dtype"))
        dtypes = {"BF16": np.dtype("<u2"), "F32": np.dtype("<f4"), "U8": np.dtype("u1")}
        dtype = dtypes.get(dtype_name)
        if (
            dtype is None
            or not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise KimiCudaError(f"invalid tensor metadata for {name}")
        start, end = int(offsets[0]), int(offsets[1])
        count = int(np.prod(shape, dtype=np.int64))
        if end - start != count * dtype.itemsize:
            raise KimiCudaError(f"tensor byte count mismatch for {name}")
        return np.memmap(
            self.root / shard_name,
            mode="r",
            dtype=dtype,
            offset=payload_offset + start,
            shape=tuple(shape),
        )

    def f32(self, name: str) -> np.ndarray:
        source = self.array(name)
        if source.dtype == np.dtype("<u2"):
            bits = np.asarray(source, dtype=np.uint16)
            return np.ascontiguousarray(
                (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
            )
        if source.dtype == np.dtype("<f4"):
            return np.ascontiguousarray(source, dtype=np.float32)
        raise KimiCudaError(f"tensor {name} cannot be converted to float32")


def _quantize_bf16_grouped_int4(
    source: np.ndarray, *, chunk_rows: int = 256
) -> _GroupedInt4Tensor:
    if source.ndim != 2 or source.dtype != np.dtype("<u2") or source.shape[1] % 64:
        raise KimiCudaError(f"grouped-int4 source must be BF16 [O,I%64], got {source}")
    output_dimension, input_dimension = (int(source.shape[0]), int(source.shape[1]))
    packed = np.empty((output_dimension, input_dimension // 2), dtype=np.uint8)
    scales = np.empty((output_dimension, input_dimension // 64), dtype=np.float32)
    for start in range(0, output_dimension, chunk_rows):
        stop = min(output_dimension, start + chunk_rows)
        bits = np.asarray(source[start:stop], dtype=np.uint16)
        values = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        grouped = values.reshape(stop - start, -1, 64)
        maximum = np.max(np.abs(grouped), axis=2).astype(np.float32)
        chunk_scales = np.maximum(maximum / np.float32(7.0), np.float32(1e-20))
        # Match kimi_k3.c's load-time quantizer exactly: it rounds a multiply by
        # a separately rounded FP32 reciprocal, not a direct divide.  The two
        # forms disagree at a small but graph-visible number of half-way values.
        inverse = np.float32(1.0) / chunk_scales
        quantized = np.rint(grouped * inverse[:, :, None])
        quantized = np.clip(quantized, -8, 7).astype(np.int8)
        encoded = (quantized.astype(np.int16) + 8).astype(np.uint8)
        packed[start:stop] = (
            encoded[:, :, 0::2] | (encoded[:, :, 1::2] << np.uint8(4))
        ).reshape(stop - start, input_dimension // 2)
        scales[start:stop] = chunk_scales
    return _GroupedInt4Tensor(
        packed=np.ascontiguousarray(packed),
        scales=np.ascontiguousarray(scales),
        source=np.empty((0,), dtype=np.float32),
        input_dimension=input_dimension,
        output_dimension=output_dimension,
    )


def _pointer_offset(pointer: ctypes.c_void_p, elements: int) -> ctypes.c_void_p:
    value = int(pointer.value or 0)
    if not value:
        raise KimiCudaError("cannot offset a null CUDA pointer")
    return ctypes.c_void_p(value + elements * np.dtype(np.float32).itemsize)


def _digest_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def _parse_oracle_routes(path: Path | None) -> dict[int, dict[int, list[int]]]:
    if path is None:
        return {}
    source = path.expanduser().resolve()
    routes: dict[int, dict[int, list[int]]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        layer = int(fields[2])
        layer_routes = routes.setdefault(layer, {})
        # ROUTE_TRACE's row field is local to one MoE call.  The retained
        # oracle used K3_CHUNK=1, so it is zero on every line; the occurrence
        # ordinal for a given layer is the global token position and also
        # generalizes to larger chunks because rows are emitted in order.
        position = len(layer_routes)
        if layer < 0 or position < 0:
            raise KimiCudaError(f"invalid oracle route coordinates: {line}")
        if position in layer_routes:
            raise KimiCudaError(
                f"duplicate oracle route for layer {layer}, position {position}"
            )
        layer_routes[position] = [
            int(field.split(":", 1)[0]) for field in fields[3:]
        ]
    return routes


class _LayerResources:
    def __init__(self, runtime: _CudaRuntime) -> None:
        self.runtime = runtime
        self.allocations: list[ctypes.c_void_p] = []
        self.tensors: list[ctypes.c_void_p] = []
        self.resident_tensor_bytes = 0
        self.resident_vector_bytes = 0
        self.weight_digest = hashlib.sha256()
        self.tensor_names: list[str] = []

    def allocate(self, elements: int) -> ctypes.c_void_p:
        pointer = self.runtime.allocate(elements * np.dtype(np.float32).itemsize)
        self.allocations.append(pointer)
        return pointer

    def upload_vector(self, name: str, values: np.ndarray) -> ctypes.c_void_p:
        source = np.ascontiguousarray(values, dtype=np.float32)
        pointer = self.allocate(source.size)
        self.runtime.upload_activation(pointer, source)
        self.resident_vector_bytes += source.nbytes
        _digest_array(self.weight_digest, name, source)
        self.tensor_names.append(name)
        return pointer

    def upload_data(self, values: np.ndarray) -> ctypes.c_void_p:
        source = np.ascontiguousarray(values, dtype=np.float32)
        pointer = self.allocate(source.size)
        self.runtime.upload_activation(pointer, source)
        return pointer

    def track_tensor(
        self, name: str, handle: ctypes.c_void_p, arrays: tuple[np.ndarray, ...]
    ) -> ctypes.c_void_p:
        self.tensors.append(handle)
        self.resident_tensor_bytes += self.runtime.tensor_bytes(handle)
        for index, array in enumerate(arrays):
            _digest_array(self.weight_digest, f"{name}:{index}", array)
        self.tensor_names.append(name)
        return handle

    def close(self) -> None:
        for pointer in reversed(self.allocations):
            self.runtime.free(pointer)
        self.allocations.clear()
        for handle in reversed(self.tensors):
            self.runtime.release_tensor(handle)
        self.tensors.clear()


class KimiCudaGraphRunner:
    """Strict streamed graph runner used to promote CUDA component evidence."""

    def __init__(self, checkpoint: Path, cuda_library: Path, device: int = 0) -> None:
        self.reader = _CheckpointReader(checkpoint)
        self.config = self.reader.config
        self.runtime = _CudaRuntime(cuda_library.expanduser().resolve(), device)
        self.runtime.set_telemetry("minimal")
        self.device = device
        self.states: dict[int, dict[str, np.ndarray]] = {}
        self.operation_counts: dict[str, int] = {
            name: 0
            for name in (
                "embedding",
                "dense_projection",
                "KDA",
                "Gated_MLA",
                "attention_residual",
                "router",
                "MXFP4_routed_expert",
                "shared_experts",
                "MoE_reduction",
                "final_norm",
                "LM_head",
            )
        }

    def close(self) -> None:
        self.runtime.close()

    def _upload_int4(
        self, resources: _LayerResources, name: str
    ) -> ctypes.c_void_p:
        source = self.reader.array(name)
        tensor = _quantize_bf16_grouped_int4(source)
        handle = self.runtime.upload_grouped_int4(tensor)
        return resources.track_tensor(name, handle, (tensor.packed, tensor.scales))

    def _upload_int8(
        self, resources: _LayerResources, name: str
    ) -> ctypes.c_void_p:
        source = self.reader.array(name)
        tensor = _quantize_bf16_rows_int8(source)
        handle = self.runtime.upload_int8(tensor)
        return resources.track_tensor(name, handle, (tensor.weights, tensor.scales))

    def _upload_f32_matrix(
        self, resources: _LayerResources, name: str
    ) -> ctypes.c_void_p:
        matrix = self.reader.f32(name)
        handle = self.runtime.upload_float32(matrix)
        return resources.track_tensor(name, handle, (matrix,))

    def _upload_expert(
        self, resources: _LayerResources, layer: int, expert: int
    ) -> tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
        prefix = (
            f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
        )

        def matrix(role: str, stem: str, inputs: int, outputs: int) -> ctypes.c_void_p:
            packed_name = f"{prefix}.{stem}.weight_packed"
            scale_name = f"{prefix}.{stem}.weight_scale"
            packed = np.ascontiguousarray(self.reader.array(packed_name), dtype=np.uint8)
            scales = np.ascontiguousarray(self.reader.array(scale_name), dtype=np.uint8)
            tensor = MXFP4Tensor(
                packed=packed,
                scales=scales,
                input_dimension=inputs,
                output_dimension=outputs,
            )
            handle = self.runtime.upload(tensor)
            return resources.track_tensor(
                f"{prefix}:{role}", handle, (tensor.packed, tensor.scales)
            )

        gate = matrix("gate", "w1", self.config.latent, self.config.moe_intermediate)
        down = matrix("down", "w2", self.config.moe_intermediate, self.config.latent)
        up = matrix("up", "w3", self.config.latent, self.config.moe_intermediate)
        return gate, up, down

    def embed(self, token_ids: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
        table_name = "language_model.model.embed_tokens.weight"
        table = self.reader.array(table_name)
        ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        expected_shape = (self.config.vocab, self.config.hidden)
        if table.shape != expected_shape or any(
            token < 0 or token >= self.config.vocab for token in token_ids
        ):
            raise KimiCudaError("invalid Kimi embedding table or token IDs")
        before = self.runtime.mem_info()
        handle = self.runtime.upload_bf16_embedding(table)
        token_device = self.runtime.allocate(ids.nbytes)
        output_device = self.runtime.allocate(ids.size * self.config.hidden * 4)
        try:
            self.runtime.upload_bytes(token_device, ids)
            self.runtime.execute_embedding(handle, output_device, token_device, ids.size)
            self.runtime.synchronize()
            output = self.runtime.download_activation(
                output_device, (ids.size, self.config.hidden)
            )
            self.operation_counts["embedding"] += ids.size
            return output, {
                "token_ids": token_ids,
                "input_fingerprint": _array_fingerprint(ids),
                "weight_fingerprint": _array_fingerprint_streaming(table),
                "output_fingerprint": _array_fingerprint(output),
                "resident_bytes": self.runtime.tensor_bytes(handle),
                "memory_before": before,
                "memory_after": self.runtime.mem_info(),
            }
        finally:
            self.runtime.free(output_device)
            self.runtime.free(token_device)
            self.runtime.release_tensor(handle)

    def _load_layer_weights(
        self, resources: _LayerResources, layer: int
    ) -> dict[str, Any]:
        config = self.config
        prefix = f"language_model.model.layers.{layer}"
        weights: dict[str, Any] = {
            "input_norm": resources.upload_vector(
                f"{prefix}.input_layernorm.weight",
                self.reader.f32(f"{prefix}.input_layernorm.weight"),
            ),
            "post_norm": resources.upload_vector(
                f"{prefix}.post_attention_layernorm.weight",
                self.reader.f32(f"{prefix}.post_attention_layernorm.weight"),
            ),
        }

        def residual_score(stem: str) -> ctypes.c_void_p:
            norm_name = f"{prefix}.{stem}_res_norm.weight"
            projection_name = f"{prefix}.{stem}_res_proj.weight"
            norm = self.reader.f32(norm_name)
            projection = self.reader.f32(projection_name)
            resources.tensor_names.extend((norm_name, projection_name))
            _digest_array(resources.weight_digest, norm_name, norm)
            _digest_array(resources.weight_digest, projection_name, projection)
            return resources.upload_vector(
                f"{prefix}.{stem}_res_score_weight",
                np.ascontiguousarray(norm * projection, dtype=np.float32),
            )

        weights["attention_residual_score"] = residual_score("self_attention")
        weights["mlp_residual_score"] = residual_score("mlp")
        if layer in config.kda_layers:
            attention_prefix = f"{prefix}.self_attn"
            for role, suffix in (
                ("q", "q_proj.weight"),
                ("k", "k_proj.weight"),
                ("v", "v_proj.weight"),
                ("gate", "g_proj.weight"),
                ("output", "o_proj.weight"),
            ):
                weights[role] = self._upload_int4(
                    resources, f"{attention_prefix}.{suffix}"
                )
            for role, suffix in (
                ("decay_a", "f_a_proj.weight"),
                ("decay_b", "f_b_proj.weight"),
                ("beta", "b_proj.weight"),
            ):
                weights[role] = self._upload_f32_matrix(
                    resources, f"{attention_prefix}.{suffix}"
                )
            for role, suffix in (
                ("conv_q", "q_conv1d.weight"),
                ("conv_k", "k_conv1d.weight"),
                ("conv_v", "v_conv1d.weight"),
                ("dt", "dt_bias"),
                ("output_norm", "o_norm.weight"),
            ):
                name = f"{attention_prefix}.{suffix}"
                weights[role] = resources.upload_vector(name, self.reader.f32(name))
            a_name = f"{attention_prefix}.A_log"
            a_log = self.reader.f32(a_name).reshape(-1)
            if a_log.size < config.kda_heads:
                raise KimiCudaError(f"{a_name} does not cover all KDA heads")
            weights["a"] = resources.upload_vector(
                f"{a_name}:exp:first-{config.kda_heads}",
                np.exp(a_log[: config.kda_heads]).astype(np.float32),
            )
            weights["attention_type"] = "KDA"
        else:
            attention_prefix = f"{prefix}.self_attn"
            for role, suffix in (
                ("query_a", "q_a_proj.weight"),
                ("query_b", "q_b_proj.weight"),
                ("kv_a", "kv_a_proj_with_mqa.weight"),
                ("kv_b", "kv_b_proj.weight"),
                ("gate", "g_proj.weight"),
                ("output", "o_proj.weight"),
            ):
                weights[role] = self._upload_int8(
                    resources, f"{attention_prefix}.{suffix}"
                )
            for role, suffix in (
                ("query_norm", "q_a_layernorm.weight"),
                ("kv_norm", "kv_a_layernorm.weight"),
            ):
                name = f"{attention_prefix}.{suffix}"
                weights[role] = resources.upload_vector(name, self.reader.f32(name))
            weights["attention_type"] = "Gated_MLA"

        if layer < config.first_dense:
            mlp_prefix = f"{prefix}.mlp"
            weights["dense_mlp"] = tuple(
                self._upload_int4(resources, f"{mlp_prefix}.{suffix}")
                for suffix in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            )
            weights["mlp_type"] = "dense"
        else:
            moe_prefix = f"{prefix}.block_sparse_moe"
            router_name = f"{moe_prefix}.gate.weight"
            bias_name = f"{moe_prefix}.gate.e_score_correction_bias"
            norm_name = f"{moe_prefix}.routed_expert_norm.weight"
            weights["router"] = resources.upload_vector(
                router_name, self.reader.f32(router_name)
            )
            weights["router_bias"] = resources.upload_vector(
                bias_name, self.reader.f32(bias_name)
            )
            weights["routed_norm"] = resources.upload_vector(
                norm_name, self.reader.f32(norm_name)
            )
            weights["latent_down"] = self._upload_int4(
                resources, f"{moe_prefix}.routed_expert_down_proj.weight"
            )
            weights["latent_up"] = self._upload_int4(
                resources, f"{moe_prefix}.routed_expert_up_proj.weight"
            )
            shared_prefix = f"{moe_prefix}.shared_experts"
            weights["shared_mlp"] = tuple(
                self._upload_int4(resources, f"{shared_prefix}.{suffix}")
                for suffix in (
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            )
            weights["mlp_type"] = "moe"
        return weights

    def _prepare_attention_state(
        self,
        resources: _LayerResources,
        layer: int,
        *,
        maximum_context: int,
    ) -> dict[str, ctypes.c_void_p]:
        config = self.config
        retained = self.states.get(layer)
        if layer in config.kda_layers:
            shapes = {
                "state": (
                    config.kda_heads,
                    config.kda_head_dimension,
                    config.kda_head_dimension,
                ),
                "window_q": (config.kda_projection, config.convolution_width),
                "window_k": (config.kda_projection, config.convolution_width),
                "window_v": (config.kda_projection, config.convolution_width),
            }
        else:
            shapes = {
                "latent_cache": (maximum_context, config.kv_lora),
                "rope_cache": (maximum_context, config.query_rope),
            }
        state: dict[str, ctypes.c_void_p] = {}
        for name, shape in shapes.items():
            values = (
                np.zeros(shape, dtype=np.float32)
                if retained is None
                else np.ascontiguousarray(retained[name], dtype=np.float32)
            )
            if values.shape != shape:
                raise KimiCudaError(
                    f"layer {layer} retained {name} shape {values.shape} != {shape}"
                )
            state[name] = resources.upload_data(values)
        return state

    def _save_attention_state(
        self,
        layer: int,
        state: dict[str, ctypes.c_void_p],
        *,
        maximum_context: int,
    ) -> dict[str, Any]:
        config = self.config
        if layer in config.kda_layers:
            shapes = {
                "state": (
                    config.kda_heads,
                    config.kda_head_dimension,
                    config.kda_head_dimension,
                ),
                "window_q": (config.kda_projection, config.convolution_width),
                "window_k": (config.kda_projection, config.convolution_width),
                "window_v": (config.kda_projection, config.convolution_width),
            }
        else:
            shapes = {
                "latent_cache": (maximum_context, config.kv_lora),
                "rope_cache": (maximum_context, config.query_rope),
            }
        saved = {
            name: self.runtime.download_activation(state[name], shape)
            for name, shape in shapes.items()
        }
        self.states[layer] = saved
        digest = hashlib.sha256()
        for name, values in saved.items():
            _digest_array(digest, name, values)
        return {
            "bytes": sum(values.nbytes for values in saved.values()),
            "fingerprint": "sha256:" + digest.hexdigest(),
            "finite": all(bool(np.isfinite(values).all()) for values in saved.values()),
        }

    def _execute_attention(
        self,
        resources: _LayerResources,
        weights: dict[str, Any],
        state: dict[str, ctypes.c_void_p],
        normalized: ctypes.c_void_p,
        output: ctypes.c_void_p,
        *,
        layer: int,
        position: int,
        scratch: dict[str, ctypes.c_void_p],
    ) -> None:
        config = self.config
        if layer in config.kda_layers:
            for role in ("q", "k", "v", "gate"):
                self.runtime.execute_dense(
                    weights[role], scratch[role], normalized, 1
                )
            self.runtime.execute_dense(
                weights["decay_a"], scratch["decay_low"], normalized, 1
            )
            self.runtime.execute_dense(
                weights["decay_b"], scratch["decay"], scratch["decay_low"], 1
            )
            self.runtime.execute_dense(
                weights["beta"], scratch["beta"], normalized, 1
            )
            self.runtime.execute_kda_core(
                scratch["core"],
                scratch["q"],
                scratch["k"],
                scratch["v"],
                scratch["gate"],
                scratch["decay"],
                scratch["beta"],
                weights["conv_q"],
                weights["conv_k"],
                weights["conv_v"],
                state["window_q"],
                state["window_k"],
                state["window_v"],
                state["state"],
                weights["dt"],
                weights["a"],
                weights["output_norm"],
                heads=config.kda_heads,
                head_dimension=config.kda_head_dimension,
                convolution_width=config.convolution_width,
                gate_lower_bound=config.gate_lower_bound,
                epsilon=config.epsilon,
            )
            self.runtime.execute_dense(weights["output"], output, scratch["core"], 1)
            self.operation_counts["dense_projection"] += 8
            self.operation_counts["KDA"] += 1
            return
        self.runtime.execute_dense(weights["query_a"], scratch["query_low"], normalized, 1)
        self.runtime.execute_rmsnorm(
            scratch["query_low"],
            scratch["query_low"],
            weights["query_norm"],
            batch=1,
            dimension=config.query_lora,
            epsilon=config.epsilon,
        )
        self.runtime.execute_dense(
            weights["query_b"], scratch["query"], scratch["query_low"], 1
        )
        self.runtime.execute_dense(weights["kv_a"], scratch["compressed_kv"], normalized, 1)
        self.runtime.execute_mla_cache_append(
            _pointer_offset(state["latent_cache"], position * config.kv_lora),
            _pointer_offset(state["rope_cache"], position * config.query_rope),
            scratch["compressed_kv"],
            weights["kv_norm"],
            kv_lora=config.kv_lora,
            rope_dimension=config.query_rope,
            epsilon=config.epsilon,
        )
        self.runtime.execute_dense(weights["gate"], scratch["mla_gate"], normalized, 1)
        self.runtime.execute_mla_absorb(
            weights["kv_b"],
            scratch["context"],
            scratch["query"],
            state["latent_cache"],
            state["rope_cache"],
            heads=config.heads,
            query_nope=config.query_nope,
            query_rope=config.query_rope,
            value_dimension=config.value_dimension,
            kv_lora=config.kv_lora,
            context_length=position + 1,
            attention_scale=config.attention_scale,
        )
        self.runtime.execute_mla_gate(
            scratch["context"], scratch["mla_gate"], config.context_dimension
        )
        self.runtime.execute_dense(weights["output"], output, scratch["context"], 1)
        self.operation_counts["dense_projection"] += 5
        self.operation_counts["Gated_MLA"] += 1

    def _attention_scratch(
        self, resources: _LayerResources, layer: int
    ) -> dict[str, ctypes.c_void_p]:
        config = self.config
        if layer in config.kda_layers:
            projection = config.kda_projection
            return {
                "q": resources.allocate(projection),
                "k": resources.allocate(projection),
                "v": resources.allocate(projection),
                "gate": resources.allocate(projection),
                "decay_low": resources.allocate(config.kda_head_dimension),
                "decay": resources.allocate(projection),
                "beta": resources.allocate(config.kda_heads),
                "core": resources.allocate(projection),
            }
        return {
            "query_low": resources.allocate(config.query_lora),
            "query": resources.allocate(config.query_dimension),
            "compressed_kv": resources.allocate(config.kv_lora + config.query_rope),
            "mla_gate": resources.allocate(config.context_dimension),
            "context": resources.allocate(config.context_dimension),
        }

    def _execute_dense_mlp(
        self,
        weights: dict[str, Any],
        mlp_input: ctypes.c_void_p,
        output: ctypes.c_void_p,
    ) -> None:
        self.runtime.execute_resident(weights["dense_mlp"], output, mlp_input, 1)
        self.operation_counts["dense_projection"] += 3

    def _execute_sparse_mlp_rows(
        self,
        resources: _LayerResources,
        weights: dict[str, Any],
        mlp_inputs: ctypes.c_void_p,
        prefix_rows: ctypes.c_void_p,
        routes: list[dict[str, Any]],
        *,
        layer: int,
    ) -> None:
        config = self.config
        unique_experts = sorted(
            {expert for route in routes for expert in route["selected_expert_ids"]}
        )
        expert_handles = {
            expert: self._upload_expert(resources, layer, expert)
            for expert in unique_experts
        }
        expert_rows = resources.allocate(config.topk * config.latent)
        route_weights = resources.allocate(config.topk)
        latent_input = resources.allocate(config.latent)
        reduced = resources.allocate(config.latent)
        routed_output = resources.allocate(config.hidden)
        shared_output = resources.allocate(config.hidden)
        for row, route in enumerate(routes):
            source = _pointer_offset(mlp_inputs, row * config.hidden)
            prefix = _pointer_offset(prefix_rows, row * config.hidden)
            self.runtime.execute_dense(weights["latent_down"], latent_input, source, 1)
            for slot, expert in enumerate(route["selected_expert_ids"]):
                self.runtime.execute_resident(
                    expert_handles[expert],
                    _pointer_offset(expert_rows, slot * config.latent),
                    latent_input,
                    1,
                )
            selected_weights = np.ascontiguousarray(
                route["selected_weights"], dtype=np.float32
            )
            self.runtime.upload_activation(route_weights, selected_weights)
            self.runtime.execute_moe_reduction(
                reduced,
                expert_rows,
                route_weights,
                count=config.topk,
                dimension=config.latent,
            )
            self.runtime.execute_rmsnorm(
                reduced,
                reduced,
                weights["routed_norm"],
                batch=1,
                dimension=config.latent,
                epsilon=config.epsilon,
            )
            self.runtime.execute_dense(weights["latent_up"], routed_output, reduced, 1)
            self.runtime.execute_resident(
                weights["shared_mlp"], shared_output, source, 1
            )
            self.runtime.execute_add(routed_output, shared_output, config.hidden)
            self.runtime.execute_add(prefix, routed_output, config.hidden)
            self.operation_counts["MXFP4_routed_expert"] += config.topk
            self.operation_counts["MoE_reduction"] += 1
            self.operation_counts["shared_experts"] += 1
            self.operation_counts["dense_projection"] += 5

    def execute_layer(
        self,
        layer: int,
        hidden_rows: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
        positions: list[int],
        *,
        maximum_context: int,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        config = self.config
        values = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != config.hidden
            or values.shape[0] != len(positions)
        ):
            raise KimiCudaError("layer input does not match Kimi hidden geometry")
        if block_residuals.shape != (values.shape[0], 8, config.hidden):
            raise KimiCudaError("AttnRes snapshot buffer has the wrong shape")
        if not 0 <= layer < config.layers:
            raise KimiCudaError(f"layer index outside Kimi graph: {layer}")
        is_snapshot = layer % config.residual_block == 0
        next_block_count = block_count + int(is_snapshot)
        if next_block_count < 1 or next_block_count > 8:
            raise KimiCudaError("AttnRes block count is outside the certified geometry")

        resources = _LayerResources(self.runtime)
        memory_before = self.runtime.mem_info()
        load_started = time.perf_counter_ns()
        try:
            weights = self._load_layer_weights(resources, layer)
            state_before = self.states.get(layer)
            state_input_fingerprint = None
            if state_before is not None:
                digest = hashlib.sha256()
                for name, array in state_before.items():
                    _digest_array(digest, name, array)
                state_input_fingerprint = "sha256:" + digest.hexdigest()
            state = self._prepare_attention_state(
                resources, layer, maximum_context=maximum_context
            )
            attention_scratch = self._attention_scratch(resources, layer)
            input_rows = resources.upload_data(values)
            prefix_rows = resources.upload_data(values)
            hidden_scratch = resources.allocate(config.hidden)
            normalized = resources.allocate(config.hidden)
            attention_output = resources.allocate(config.hidden)
            mixed = resources.allocate(config.hidden)
            mlp_inputs = resources.allocate(values.shape[0] * config.hidden)
            mlp_output = resources.allocate(config.hidden)
            residual_scratch = resources.allocate(next_block_count * config.hidden)
            load_ms = (time.perf_counter_ns() - load_started) / 1e6
            memory_after_load = self.runtime.mem_info()

            routes: list[dict[str, Any]] = []
            execute_started = time.perf_counter_ns()
            for row, position in enumerate(positions):
                incoming = _pointer_offset(input_rows, row * config.hidden)
                prefix = _pointer_offset(prefix_rows, row * config.hidden)
                if block_count:
                    residual_values = np.ascontiguousarray(
                        block_residuals[row, :block_count], dtype=np.float32
                    )
                    self.runtime.upload_activation(residual_scratch, residual_values)
                    self.runtime.execute_attnres_mix(
                        hidden_scratch,
                        prefix,
                        residual_scratch,
                        weights["attention_residual_score"],
                        block_count=block_count,
                        dimension=config.hidden,
                        epsilon=config.epsilon,
                    )
                    attention_input = hidden_scratch
                    self.operation_counts["attention_residual"] += 1
                else:
                    attention_input = incoming
                if is_snapshot:
                    block_residuals[row, block_count] = values[row]
                self.runtime.execute_rmsnorm(
                    normalized,
                    attention_input,
                    weights["input_norm"],
                    batch=1,
                    dimension=config.hidden,
                    epsilon=config.epsilon,
                )
                self._execute_attention(
                    resources,
                    weights,
                    state,
                    normalized,
                    attention_output,
                    layer=layer,
                    position=position,
                    scratch=attention_scratch,
                )
                if is_snapshot:
                    self.runtime.execute_copy(prefix, attention_output, config.hidden)
                else:
                    self.runtime.execute_add(prefix, attention_output, config.hidden)
                residual_values = np.ascontiguousarray(
                    block_residuals[row, :next_block_count], dtype=np.float32
                )
                self.runtime.upload_activation(residual_scratch, residual_values)
                self.runtime.execute_attnres_mix(
                    mixed,
                    prefix,
                    residual_scratch,
                    weights["mlp_residual_score"],
                    block_count=next_block_count,
                    dimension=config.hidden,
                    epsilon=config.epsilon,
                )
                mlp_input = _pointer_offset(mlp_inputs, row * config.hidden)
                self.runtime.execute_rmsnorm(
                    mlp_input,
                    mixed,
                    weights["post_norm"],
                    batch=1,
                    dimension=config.hidden,
                    epsilon=config.epsilon,
                )
                self.operation_counts["attention_residual"] += 1
                if weights["mlp_type"] == "moe":
                    ids, selected_weights, effective = self.runtime.route(
                        mlp_input,
                        weights["router"],
                        weights["router_bias"],
                        hidden=config.hidden,
                        experts=config.experts,
                        topk=config.topk,
                    )
                    if effective != config.topk:
                        raise KimiCudaError(
                            f"layer {layer} route retained {effective}, expected {config.topk}"
                        )
                    routes.append(
                        {
                            "position": position,
                            "selected_expert_ids": [int(value) for value in ids],
                            "selected_weights": [float(value) for value in selected_weights],
                        }
                    )
                    self.operation_counts["router"] += 1

            if weights["mlp_type"] == "dense":
                for row in range(values.shape[0]):
                    mlp_input = _pointer_offset(mlp_inputs, row * config.hidden)
                    prefix = _pointer_offset(prefix_rows, row * config.hidden)
                    self._execute_dense_mlp(weights, mlp_input, mlp_output)
                    self.runtime.execute_add(prefix, mlp_output, config.hidden)
            else:
                self._execute_sparse_mlp_rows(
                    resources,
                    weights,
                    mlp_inputs,
                    prefix_rows,
                    routes,
                    layer=layer,
                )
            self.runtime.synchronize()
            output = self.runtime.download_activation(
                prefix_rows, (values.shape[0], config.hidden)
            )
            state_evidence = self._save_attention_state(
                layer, state, maximum_context=maximum_context
            )
            execute_ms = (time.perf_counter_ns() - execute_started) / 1e6
            memory_peak = self.runtime.mem_info()
            if not np.isfinite(output).all() or not state_evidence["finite"]:
                raise KimiCudaError(f"layer {layer} produced non-finite output or state")
            return output, next_block_count, {
                "layer": layer,
                "attention_type": weights["attention_type"],
                "mlp_type": weights["mlp_type"],
                "positions": positions,
                "input_fingerprint": _array_fingerprint(values),
                "weight_fingerprint": "sha256:" + resources.weight_digest.hexdigest(),
                "weight_tensor_count": len(resources.tensor_names),
                "weight_tensor_names": sorted(set(resources.tensor_names)),
                "output_fingerprint": _array_fingerprint(output),
                "state_input_fingerprint": state_input_fingerprint,
                "state_output": state_evidence,
                "routes": routes,
                "timing": {"load_ms": load_ms, "execute_ms": execute_ms},
                "memory": {
                    "before": memory_before,
                    "after_load": memory_after_load,
                    "after_experts": memory_peak,
                    "resident_weight_bytes": resources.resident_tensor_bytes
                    + resources.resident_vector_bytes,
                },
                "backend_identity": "nvidia_cuda_streamed_layer_no_cpu_math_fallback",
            }
        finally:
            resources.close()

    def execute_final_head(
        self,
        hidden_rows: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        config = self.config
        if block_count != 8:
            raise KimiCudaError(f"final Kimi AttnRes expected 8 blocks, got {block_count}")
        values = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        resources = _LayerResources(self.runtime)
        memory_before = self.runtime.mem_info()
        started = time.perf_counter_ns()
        try:
            norm_name = "language_model.model.output_attn_res_norm.weight"
            projection_name = "language_model.model.output_attn_res_proj.weight"
            norm_value = self.reader.f32(norm_name)
            projection_value = self.reader.f32(projection_name)
            resources.tensor_names.extend((norm_name, projection_name))
            _digest_array(resources.weight_digest, norm_name, norm_value)
            _digest_array(resources.weight_digest, projection_name, projection_value)
            score = resources.upload_vector(
                "language_model.model.output_attn_res_score_weight",
                np.ascontiguousarray(norm_value * projection_value, dtype=np.float32),
            )
            final_norm_name = "language_model.model.norm.weight"
            final_norm = resources.upload_vector(
                final_norm_name, self.reader.f32(final_norm_name)
            )
            head_name = "language_model.lm_head.weight"
            head = self._upload_int8(resources, head_name)
            prefix_rows = resources.upload_data(values)
            residual_scratch = resources.allocate(block_count * config.hidden)
            mixed = resources.allocate(config.hidden)
            normalized = resources.allocate(config.hidden)
            logits_device = resources.allocate(config.vocab)
            final_hidden = np.empty_like(values)
            logits = np.empty((values.shape[0], config.vocab), dtype=np.float32)
            load_ms = (time.perf_counter_ns() - started) / 1e6
            execute_started = time.perf_counter_ns()
            for row in range(values.shape[0]):
                residual_values = np.ascontiguousarray(
                    block_residuals[row, :block_count], dtype=np.float32
                )
                self.runtime.upload_activation(residual_scratch, residual_values)
                self.runtime.execute_attnres_mix(
                    mixed,
                    _pointer_offset(prefix_rows, row * config.hidden),
                    residual_scratch,
                    score,
                    block_count=block_count,
                    dimension=config.hidden,
                    epsilon=config.epsilon,
                )
                self.runtime.execute_rmsnorm(
                    normalized,
                    mixed,
                    final_norm,
                    batch=1,
                    dimension=config.hidden,
                    epsilon=config.epsilon,
                )
                self.runtime.execute_dense(head, logits_device, normalized, 1)
                self.runtime.synchronize()
                final_hidden[row] = self.runtime.download_activation(
                    normalized, (config.hidden,)
                )
                logits[row] = self.runtime.download_activation(
                    logits_device, (config.vocab,)
                )
                self.operation_counts["attention_residual"] += 1
                self.operation_counts["final_norm"] += 1
                self.operation_counts["LM_head"] += 1
            execute_ms = (time.perf_counter_ns() - execute_started) / 1e6
            return final_hidden, logits, {
                "input_fingerprint": _array_fingerprint(values),
                "weight_fingerprint": "sha256:" + resources.weight_digest.hexdigest(),
                "weight_tensor_names": sorted(set(resources.tensor_names)),
                "final_hidden_fingerprint": _array_fingerprint(final_hidden),
                "logits_fingerprint": _array_fingerprint(logits),
                "argmax_token_ids": [int(value) for value in np.argmax(logits, axis=1)],
                "timing": {"load_ms": load_ms, "execute_ms": execute_ms},
                "memory": {
                    "before": memory_before,
                    "after": self.runtime.mem_info(),
                    "resident_weight_bytes": resources.resident_tensor_bytes
                    + resources.resident_vector_bytes,
                },
                "backend_identity": "nvidia_cuda_streamed_final_head_no_cpu_math_fallback",
            }
        finally:
            resources.close()

    def execute_pass(
        self,
        token_ids: list[int],
        positions: list[int],
        *,
        layer_limit: int,
        maximum_context: int,
        oracle_trace: np.ndarray | None,
        oracle_layer_count: int,
        oracle_routes: dict[int, dict[int, list[int]]],
        oracle_logits: np.ndarray | None,
        progress_label: str,
    ) -> dict[str, Any]:
        if len(token_ids) != len(positions) or not token_ids:
            raise KimiCudaError("a CUDA graph pass requires aligned token IDs and positions")
        hidden, embedding = self.embed(token_ids)
        block_residuals = np.zeros((len(token_ids), 8, self.config.hidden), dtype=np.float32)
        block_count = 0
        layers: list[dict[str, Any]] = []
        maximum_relative_error = 0.0
        routing_equal = True
        for layer in range(layer_limit):
            layer_started = time.perf_counter_ns()
            hidden, block_count, evidence = self.execute_layer(
                layer,
                hidden,
                block_residuals,
                block_count,
                positions,
                maximum_context=maximum_context,
            )
            evidence["wall_ms"] = (time.perf_counter_ns() - layer_started) / 1e6
            if oracle_trace is not None:
                references = np.ascontiguousarray(
                    np.stack(
                        [
                            oracle_trace[position * (oracle_layer_count + 1) + layer]
                            for position in positions
                        ]
                    ),
                    dtype=np.float32,
                )
                metrics = _numerical_metrics(references, hidden)
                evidence["oracle_correctness"] = metrics
                evidence["oracle_correctness_by_position"] = [
                    {
                        "position": position,
                        **_numerical_metrics(reference, actual),
                    }
                    for position, reference, actual in zip(
                        positions, references, hidden, strict=True
                    )
                ]
                maximum_relative_error = max(
                    maximum_relative_error, float(metrics["relative_l2_error"])
                )
            if evidence["mlp_type"] == "moe" and oracle_routes:
                route_rows = []
                expected_rows = oracle_routes.get(layer, {})
                for observed in evidence["routes"]:
                    position = int(observed["position"])
                    expected = expected_rows.get(position, [])
                    equal = observed["selected_expert_ids"] == expected
                    route_rows.append(
                        {
                            "position": position,
                            "expected_expert_ids": expected,
                            "routing_equality": equal,
                        }
                    )
                    routing_equal = routing_equal and equal
                evidence["oracle_routing"] = route_rows
            layers.append(evidence)
            if (layer + 1) % 8 == 0 or layer + 1 == layer_limit:
                print(
                    f"[k3-cuda:{progress_label}] layer {layer + 1}/{layer_limit}; "
                    f"max_rel_l2={maximum_relative_error:.9g}; "
                    f"routing_equal={routing_equal}",
                    file=sys.stderr,
                    flush=True,
                )
        result: dict[str, Any] = {
            "token_ids": token_ids,
            "positions": positions,
            "embedding": embedding,
            "layers": layers,
            "block_count": block_count,
            "last_hidden_fingerprint": _array_fingerprint(hidden),
            "maximum_layer_relative_l2_error": maximum_relative_error,
            "routing_equality": routing_equal,
        }
        if layer_limit == self.config.layers:
            final_hidden, logits, head = self.execute_final_head(
                hidden, block_residuals, block_count
            )
            if oracle_trace is not None:
                references = np.ascontiguousarray(
                    np.stack(
                        [
                            oracle_trace[
                                position * (oracle_layer_count + 1)
                                + oracle_layer_count
                            ]
                            for position in positions
                        ]
                    ),
                    dtype=np.float32,
                )
                head["oracle_final_hidden"] = _numerical_metrics(
                    references, final_hidden
                )
                maximum_relative_error = max(
                    maximum_relative_error,
                    float(head["oracle_final_hidden"]["relative_l2_error"]),
                )
            if oracle_logits is not None and all(
                position < oracle_logits.shape[0] for position in positions
            ):
                expected_logits = np.ascontiguousarray(
                    np.stack([oracle_logits[position] for position in positions]),
                    dtype=np.float32,
                )
                head["oracle_logits"] = _numerical_metrics(expected_logits, logits)
            result["head"] = head
            result["sampled_token_id"] = int(np.argmax(logits[-1]))
            result["maximum_layer_relative_l2_error"] = maximum_relative_error
        return result


