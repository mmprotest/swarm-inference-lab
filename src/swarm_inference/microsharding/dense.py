"""Rank-local Megatron-style tensor parallelism for dense Qwen3."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from swarm_inference.exceptions import IntegrityError
from swarm_inference.microsharding.kv_cache import PartitionedKVCache
from swarm_inference.microsharding.schemas import ModelPartitionPlan


class ColumnParallelLinear(nn.Module):
    """Linear projection whose output features are rank-local."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        global_out_features: int,
        shard_start: int,
        shard_end: int,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if shard_end - shard_start != out_features:
            raise ValueError("local output width does not match the shard interval")
        if not 0 <= shard_start < shard_end <= global_out_features:
            raise ValueError("invalid column-parallel output interval")
        self.in_features = in_features
        self.out_features = out_features
        self.global_out_features = global_out_features
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def load_local(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if tuple(weight.shape) != tuple(self.weight.shape):
            raise IntegrityError(
                f"column-parallel weight shape {tuple(weight.shape)} != {tuple(self.weight.shape)}"
            )
        if (self.bias is None) != (bias is None):
            raise IntegrityError("column-parallel bias presence mismatch")
        with torch.no_grad():
            self.weight.copy_(weight.to(device=self.weight.device, dtype=self.weight.dtype))
            if self.bias is not None and bias is not None:
                if tuple(bias.shape) != tuple(self.bias.shape):
                    raise IntegrityError("column-parallel bias shape mismatch")
                self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear projection whose input features are rank-local.

    Every rank returns a hidden-state contribution.  The group applies a
    deterministic sum; a replicated bias is applied by exactly one rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        global_in_features: int,
        shard_start: int,
        shard_end: int,
        bias: bool = False,
        apply_bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if shard_end - shard_start != in_features:
            raise ValueError("local input width does not match the shard interval")
        if not 0 <= shard_start < shard_end <= global_in_features:
            raise ValueError("invalid row-parallel input interval")
        if apply_bias and not bias:
            raise ValueError("cannot apply an absent row-parallel bias")
        self.in_features = in_features
        self.out_features = out_features
        self.global_in_features = global_in_features
        self.shard_start = shard_start
        self.shard_end = shard_end
        self.apply_bias = apply_bias
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def load_local(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        if tuple(weight.shape) != tuple(self.weight.shape):
            raise IntegrityError(
                f"row-parallel weight shape {tuple(weight.shape)} != {tuple(self.weight.shape)}"
            )
        if (self.bias is None) != (bias is None):
            raise IntegrityError("row-parallel bias presence mismatch")
        with torch.no_grad():
            self.weight.copy_(weight.to(device=self.weight.device, dtype=self.weight.dtype))
            if self.bias is not None and bias is not None:
                if tuple(bias.shape) != tuple(self.bias.shape):
                    raise IntegrityError("row-parallel bias shape mismatch")
                self.bias.copy_(bias.to(device=self.bias.device, dtype=self.bias.dtype))

    def forward(self, local_value: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.apply_bias else None
        if self.in_features == self.global_in_features:
            return F.linear(local_value, self.weight, bias)
        return F.linear(
            local_value.float(),
            self.weight.float(),
            None if bias is None else bias.float(),
        )


class QwenRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(hidden_size, device=device, dtype=dtype), requires_grad=False
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        value = hidden_states.to(torch.float32)
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * value.to(input_dtype)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


class QwenRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dimension: int,
        *,
        theta: float,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        inv_frequency = 1.0 / (
            theta
            ** (torch.arange(0, head_dimension, 2, dtype=torch.int64).float() / head_dimension)
        )
        self.register_buffer("inv_freq", inv_frequency.to(device), persistent=False)

    @torch.no_grad()
    def forward(
        self, value: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        expanded = expanded.to(value.device)
        positions = position_ids[:, None, :].float()
        device_type = value.device.type if value.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (expanded.float() @ positions.float()).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cos = embedding.cos()
            sin = embedding.sin()
        return cos.to(dtype=value.dtype), sin.to(dtype=value.dtype)


class TensorParallelQwenAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        query_head_ids: list[int],
        kv_head_ids: list[int],
        global_query_heads: int,
        global_kv_heads: int,
        head_dimension: int,
        rms_norm_eps: float,
        attention_bias: bool,
        rank: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if global_query_heads % global_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if not query_head_ids or not kv_head_ids:
            raise ValueError("attention rank must own query and KV heads")
        self.hidden_size = hidden_size
        self.query_head_ids = tuple(query_head_ids)
        self.kv_head_ids = tuple(kv_head_ids)
        self.global_query_heads = global_query_heads
        self.global_kv_heads = global_kv_heads
        self.head_dimension = head_dimension
        self.rank = rank
        self.grouped_query_ratio = global_query_heads // global_kv_heads
        self.scaling = head_dimension**-0.5
        query_start = query_head_ids[0] * head_dimension
        query_end = (query_head_ids[-1] + 1) * head_dimension
        kv_start = kv_head_ids[0] * head_dimension
        kv_end = (kv_head_ids[-1] + 1) * head_dimension
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            query_end - query_start,
            global_out_features=global_query_heads * head_dimension,
            shard_start=query_start,
            shard_end=query_end,
            bias=attention_bias,
            device=device,
            dtype=dtype,
        )
        self.k_proj = ColumnParallelLinear(
            hidden_size,
            kv_end - kv_start,
            global_out_features=global_kv_heads * head_dimension,
            shard_start=kv_start,
            shard_end=kv_end,
            bias=attention_bias,
            device=device,
            dtype=dtype,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            kv_end - kv_start,
            global_out_features=global_kv_heads * head_dimension,
            shard_start=kv_start,
            shard_end=kv_end,
            bias=attention_bias,
            device=device,
            dtype=dtype,
        )
        self.o_proj = RowParallelLinear(
            query_end - query_start,
            hidden_size,
            global_in_features=global_query_heads * head_dimension,
            shard_start=query_start,
            shard_end=query_end,
            bias=attention_bias,
            apply_bias=attention_bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        self.q_norm = QwenRMSNorm(head_dimension, eps=rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = QwenRMSNorm(head_dimension, eps=rms_norm_eps, device=device, dtype=dtype)
        kv_to_local = {head: index for index, head in enumerate(kv_head_ids)}
        query_to_kv = [kv_to_local[head // self.grouped_query_ratio] for head in query_head_ids]
        self.register_buffer(
            "query_to_local_kv",
            torch.tensor(query_to_kv, device=device, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: PartitionedKVCache | None,
        request_id: str,
        layer_id: int,
        cache_generation: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states).view(
            *input_shape, len(self.query_head_ids), self.head_dimension
        )
        key = self.k_proj(hidden_states).view(
            *input_shape, len(self.kv_head_ids), self.head_dimension
        )
        value = self.v_proj(hidden_states).view(
            *input_shape, len(self.kv_head_ids), self.head_dimension
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary(query, key, cos, sin)
        if cache is not None:
            key, value = cache.append(
                request_id=request_id,
                layer_id=layer_id,
                tp_rank=self.rank,
                global_kv_head_ids=self.kv_head_ids,
                key=key,
                value=value,
                cache_generation=cache_generation,
            )
        query_key = key.index_select(1, self.query_to_local_kv)
        query_value = value.index_select(1, self.query_to_local_kv)
        attention_weights = torch.matmul(query, query_key.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask[..., : key.shape[-2]]
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attention_output = torch.matmul(attention_weights, query_value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attention_output), key, value


class TensorParallelQwenMLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        global_intermediate_size: int,
        intermediate_start: int,
        intermediate_end: int,
        rank: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        local_intermediate = intermediate_end - intermediate_start
        if local_intermediate <= 0:
            raise ValueError("MLP rank must own intermediate channels")
        self.intermediate_start = intermediate_start
        self.intermediate_end = intermediate_end
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            local_intermediate,
            global_out_features=global_intermediate_size,
            shard_start=intermediate_start,
            shard_end=intermediate_end,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            local_intermediate,
            global_out_features=global_intermediate_size,
            shard_start=intermediate_start,
            shard_end=intermediate_end,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            local_intermediate,
            hidden_size,
            global_in_features=global_intermediate_size,
            shard_start=intermediate_start,
            shard_end=intermediate_end,
            bias=False,
            apply_bias=False,
            device=device,
            dtype=dtype,
        )
        self.rank = rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)),
        )


class TensorParallelQwenDecoderLayer(nn.Module):
    """One rank's direct, local construction of a Qwen3 decoder layer."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        layer_id: int,
        rank: int,
        query_head_ids: list[int],
        kv_head_ids: list[int],
        intermediate_range: tuple[int, int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.rank = rank
        hidden_size = int(config["hidden_size"])
        query_heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads") or query_heads)
        head_dimension = int(config.get("head_dim") or hidden_size // query_heads)
        eps = float(config["rms_norm_eps"])
        self.input_layernorm = QwenRMSNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        self.self_attn = TensorParallelQwenAttention(
            hidden_size=hidden_size,
            query_head_ids=query_head_ids,
            kv_head_ids=kv_head_ids,
            global_query_heads=query_heads,
            global_kv_heads=kv_heads,
            head_dimension=head_dimension,
            rms_norm_eps=eps,
            attention_bias=bool(config.get("attention_bias", False)),
            rank=rank,
            device=device,
            dtype=dtype,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            hidden_size, eps=eps, device=device, dtype=dtype
        )
        self.mlp = TensorParallelQwenMLP(
            hidden_size=hidden_size,
            global_intermediate_size=int(config["intermediate_size"]),
            intermediate_start=intermediate_range[0],
            intermediate_end=intermediate_range[1],
            rank=rank,
            device=device,
            dtype=dtype,
        )

    def load_local_state(self, state: dict[str, torch.Tensor]) -> set[str]:
        prefix = f"model.layers.{self.layer_id}."
        attention_bias = self.self_attn.q_proj.bias is not None

        def required(suffix: str) -> torch.Tensor:
            name = prefix + suffix
            try:
                return state[name]
            except KeyError as exc:
                raise IntegrityError(f"rank {self.rank} is missing {name}") from exc

        consumed = {
            prefix + "input_layernorm.weight",
            prefix + "post_attention_layernorm.weight",
            prefix + "self_attn.q_norm.weight",
            prefix + "self_attn.k_norm.weight",
            prefix + "self_attn.q_proj.weight",
            prefix + "self_attn.k_proj.weight",
            prefix + "self_attn.v_proj.weight",
            prefix + "self_attn.o_proj.weight",
            prefix + "mlp.gate_proj.weight",
            prefix + "mlp.up_proj.weight",
            prefix + "mlp.down_proj.weight",
        }
        with torch.no_grad():
            self.input_layernorm.weight.copy_(
                required("input_layernorm.weight").to(
                    self.input_layernorm.weight.device, self.input_layernorm.weight.dtype
                )
            )
            self.post_attention_layernorm.weight.copy_(
                required("post_attention_layernorm.weight").to(
                    self.post_attention_layernorm.weight.device,
                    self.post_attention_layernorm.weight.dtype,
                )
            )
            self.self_attn.q_norm.weight.copy_(
                required("self_attn.q_norm.weight").to(
                    self.self_attn.q_norm.weight.device,
                    self.self_attn.q_norm.weight.dtype,
                )
            )
            self.self_attn.k_norm.weight.copy_(
                required("self_attn.k_norm.weight").to(
                    self.self_attn.k_norm.weight.device,
                    self.self_attn.k_norm.weight.dtype,
                )
            )
        q_bias = k_bias = v_bias = o_bias = None
        if attention_bias:
            q_bias = required("self_attn.q_proj.bias")
            k_bias = required("self_attn.k_proj.bias")
            v_bias = required("self_attn.v_proj.bias")
            o_bias = required("self_attn.o_proj.bias")
            consumed.update(
                {
                    prefix + "self_attn.q_proj.bias",
                    prefix + "self_attn.k_proj.bias",
                    prefix + "self_attn.v_proj.bias",
                    prefix + "self_attn.o_proj.bias",
                }
            )
        self.self_attn.q_proj.load_local(required("self_attn.q_proj.weight"), q_bias)
        self.self_attn.k_proj.load_local(required("self_attn.k_proj.weight"), k_bias)
        self.self_attn.v_proj.load_local(required("self_attn.v_proj.weight"), v_bias)
        self.self_attn.o_proj.load_local(required("self_attn.o_proj.weight"), o_bias)
        self.mlp.gate_proj.load_local(required("mlp.gate_proj.weight"))
        self.mlp.up_proj.load_local(required("mlp.up_proj.weight"))
        self.mlp.down_proj.load_local(required("mlp.down_proj.weight"))
        return consumed

    def attention_partial(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: PartitionedKVCache | None,
        request_id: str,
        cache_generation: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            self.self_attn(
                self.input_layernorm(hidden_states),
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                cache=cache,
                request_id=request_id,
                layer_id=self.layer_id,
                cache_generation=cache_generation,
            ),
        )

    def mlp_partial(self, post_attention_hidden: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.mlp(self.post_attention_layernorm(post_attention_hidden)))


@dataclass(slots=True)
class LayerBoundaryCapture:
    layer_id: int
    attention_output: torch.Tensor
    post_attention_hidden: torch.Tensor
    mlp_output: torch.Tensor
    final_hidden: torch.Tensor
    kv_by_rank: dict[int, tuple[torch.Tensor, torch.Tensor]]


class TensorParallelLayerGroup(nn.Module):
    def __init__(
        self,
        layers: list[TensorParallelQwenDecoderLayer],
        *,
        stage_id: int,
        trace: list[dict[str, Any]],
        measure_collectives: bool = False,
        communication_format: str = "bfloat16",
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("tensor-parallel layer group cannot be empty")
        ranks = [layer.rank for layer in layers]
        if ranks != list(range(len(layers))):
            raise ValueError("tensor-parallel layers must use deterministic contiguous rank order")
        self.ranks = nn.ModuleList(layers)
        self.layer_id = layers[0].layer_id
        self.stage_id = stage_id
        self.trace = trace
        self.measure_collectives = measure_collectives
        if communication_format not in {"bfloat16", "float16", "float32", "fp8", "int8"}:
            raise ValueError(f"unsupported communication format {communication_format}")
        self.communication_format = communication_format

    def _round_trip_payload(
        self, value: torch.Tensor, output_dtype: torch.dtype
    ) -> tuple[torch.Tensor, int, int, float, float]:
        """Return the dequantised value and exact wire accounting.

        Row-parallel kernels keep FP32 accumulators locally.  The primary path
        rounds those values to BF16 before the logical all-reduce, so the
        collective payload remains exact BF16 even though summation uses a
        stable FP32 accumulator.
        """

        started = time.perf_counter_ns()
        metadata_bytes = 0
        if self.communication_format == "bfloat16":
            encoded = value.to(torch.bfloat16)
            decoded = encoded.float()
        elif self.communication_format == "float16":
            encoded = value.to(torch.float16)
            decoded = encoded.float()
        elif self.communication_format == "float32":
            encoded = value.float()
            decoded = encoded
        elif self.communication_format == "fp8":
            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is None:
                raise RuntimeError("this PyTorch build does not provide float8_e4m3fn")
            maximum = value.float().abs().max().clamp_min(1e-12)
            scale = maximum / 448.0
            encoded = (value.float() / scale).clamp(-448, 448).to(fp8_dtype)
            metadata_bytes = 4
            decoded = encoded.float() * scale
        else:
            maximum = value.float().abs().max().clamp_min(1e-12)
            scale = maximum / 127.0
            encoded = torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)
            metadata_bytes = 4
            decoded = encoded.float() * scale
        quantisation_ms = (time.perf_counter_ns() - started) / 1_000_000
        dequantisation_started = time.perf_counter_ns()
        decoded = decoded.to(torch.float32)
        dequantisation_ms = (time.perf_counter_ns() - dequantisation_started) / 1_000_000
        payload_bytes = int(encoded.numel() * encoded.element_size())
        if self.communication_format == "bfloat16" and output_dtype != torch.bfloat16:
            decoded = decoded.to(output_dtype).float()
        return decoded, payload_bytes, metadata_bytes, quantisation_ms, dequantisation_ms

    def _reduce(self, partials: list[torch.Tensor], phase: str) -> torch.Tensor:
        if len(partials) != len(self.ranks):
            raise ValueError("one partial output is required per tensor rank")
        device = partials[0].device
        start_event = end_event = None
        started_ns = time.perf_counter_ns()
        if self.measure_collectives and device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start_event.record()
        output_dtype = cast(
            TensorParallelQwenDecoderLayer, self.ranks[0]
        ).input_layernorm.weight.dtype
        communicated: list[torch.Tensor] = []
        payload_bytes = metadata_bytes = 0
        quantisation_ms = dequantisation_ms = 0.0
        for partial in partials:
            decoded, encoded_bytes, scale_bytes, quant_ms, dequant_ms = self._round_trip_payload(
                partial, output_dtype
            )
            communicated.append(decoded)
            payload_bytes = encoded_bytes
            metadata_bytes = scale_bytes
            quantisation_ms += quant_ms
            dequantisation_ms += dequant_ms
        reduced = communicated[0]
        for partial in communicated[1:]:
            if tuple(partial.shape) != tuple(reduced.shape):
                raise ValueError("all-reduce partial shapes differ")
            reduced = reduced + partial
        reduced = reduced.to(output_dtype)
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        degree = len(partials)
        self.trace.append(
            {
                "classification": "logical_single_gpu_measurement",
                "event_type": "collective_complete",
                "operation": "all_reduce_sum",
                "algorithm": "deterministic_rank_order",
                "compression": self.communication_format,
                "layer_id": self.layer_id,
                "stage_id": self.stage_id,
                "phase": phase,
                "rank_ids": [
                    f"stage-{self.stage_id:03d}-rank-{rank:03d}" for rank in range(degree)
                ],
                "payload_bytes": payload_bytes,
                "scale_metadata_bytes": metadata_bytes,
                "quantisation_time_ms": quantisation_ms,
                "dequantisation_time_ms": dequantisation_ms,
                "bytes_sent_per_rank": 0
                if degree == 1
                else 2 * (degree - 1) * payload_bytes / degree,
                "aggregate_bytes": 0 if degree == 1 else 2 * (degree - 1) * payload_bytes,
                "actual_same_gpu_time_ms": elapsed_ms,
            }
        )
        return reduced

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: PartitionedKVCache | None,
        request_id: str,
        cache_generation: int = 0,
        capture: bool = False,
    ) -> tuple[torch.Tensor, LayerBoundaryCapture | None]:
        attention_results = [
            cast(TensorParallelQwenDecoderLayer, rank).attention_partial(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                cache=cache,
                request_id=request_id,
                cache_generation=cache_generation,
            )
            for rank in self.ranks
        ]
        attention_output = self._reduce([item[0] for item in attention_results], "attention_output")
        post_attention = hidden_states + attention_output
        mlp_output = self._reduce(
            [
                cast(TensorParallelQwenDecoderLayer, rank).mlp_partial(post_attention)
                for rank in self.ranks
            ],
            "mlp_output",
        )
        final = post_attention + mlp_output
        boundary = None
        if capture:
            boundary = LayerBoundaryCapture(
                layer_id=self.layer_id,
                attention_output=attention_output.detach().clone(),
                post_attention_hidden=post_attention.detach().clone(),
                mlp_output=mlp_output.detach().clone(),
                final_hidden=final.detach().clone(),
                kv_by_rank={
                    cast(TensorParallelQwenDecoderLayer, rank).rank: (
                        result[1].detach().clone(),
                        result[2].detach().clone(),
                    )
                    for rank, result in zip(self.ranks, attention_results, strict=True)
                },
            )
        return final, boundary


class VocabularyParallelEmbedding(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        vocabulary_start: int,
        vocabulary_end: int,
        device: torch.device | str,
        dtype: torch.dtype,
        shared_parameter: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if vocabulary_end - vocabulary_start != weight.shape[0]:
            raise ValueError("vocabulary slice does not match local embedding rows")
        self.vocabulary_start = vocabulary_start
        self.vocabulary_end = vocabulary_end
        if shared_parameter is None:
            self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        else:
            if tuple(shared_parameter.shape) != tuple(weight.shape):
                raise ValueError("shared vocabulary parameter shape mismatch")
            self.weight = shared_parameter

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids >= self.vocabulary_start) & (token_ids < self.vocabulary_end)
        local_ids = (token_ids - self.vocabulary_start).clamp(min=0, max=self.weight.shape[0] - 1)
        result = F.embedding(local_ids, self.weight)
        return result * mask.unsqueeze(-1).to(result.dtype)


class VocabularyParallelLMHead(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        vocabulary_start: int,
        vocabulary_end: int,
        device: torch.device | str,
        dtype: torch.dtype,
        shared_parameter: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if vocabulary_end - vocabulary_start != weight.shape[0]:
            raise ValueError("vocabulary slice does not match local LM-head rows")
        self.vocabulary_start = vocabulary_start
        self.vocabulary_end = vocabulary_end
        if shared_parameter is None:
            self.weight = nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        else:
            if tuple(shared_parameter.shape) != tuple(weight.shape):
                raise ValueError("shared vocabulary parameter shape mismatch")
            self.weight = shared_parameter

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight)

    def local_argmax(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(hidden_states)
        values, indices = torch.max(logits, dim=-1)
        return values, indices + self.vocabulary_start


def distributed_argmax(
    candidates: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact max across vocabulary shards, with torch.argmax's lowest-ID tie rule."""

    if not candidates:
        raise ValueError("distributed argmax needs at least one rank candidate")
    values = torch.stack([item[0] for item in candidates], dim=0)
    tokens = torch.stack([item[1] for item in candidates], dim=0)
    maximum = values.max(dim=0).values
    maximum_tokens = torch.iinfo(tokens.dtype).max
    eligible = torch.where(values == maximum.unsqueeze(0), tokens, maximum_tokens)
    selected = eligible.min(dim=0).values
    return maximum, selected


def _rank_weight_path(root: Path, *, stage_count: int, stage_id: int, rank: int) -> Path:
    if stage_count == 1:
        return root / "ranks" / f"rank-{rank:03d}" / "weights.safetensors"
    return (
        root
        / "stages"
        / f"stage-{stage_id:03d}"
        / "ranks"
        / f"rank-{rank:03d}"
        / "weights.safetensors"
    )


def _load_rank_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise IntegrityError(f"rank weights are missing: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name: handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118
        }


class TensorParallelQwenModel(nn.Module):
    """Complete Qwen3 assembled exclusively from rank-local microshards."""

    def __init__(
        self,
        shard_root: Path,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        measure_collectives: bool = False,
        communication_format: str | None = None,
    ) -> None:
        super().__init__()
        self.shard_root = shard_root.expanduser().resolve()
        self.device = torch.device(device)
        self.dtype = dtype
        if communication_format is None:
            communication_format = {
                torch.bfloat16: "bfloat16",
                torch.float16: "float16",
                torch.float32: "float32",
            }.get(dtype)
        if communication_format is None:
            raise ValueError(f"no exact communication format for model dtype {dtype}")
        self.plan = ModelPartitionPlan.model_validate_json(
            (self.shard_root / "parallel_plan.json").read_text(encoding="utf-8")
        )
        self.config = json.loads(
            (self.shard_root / "config" / "config.json").read_text(encoding="utf-8")
        )
        self.tensor_parallel_degree = int(self.plan.metadata["tensor_parallel_degree"])
        self.pipeline_stage_count = len(self.plan.pipeline_stages)
        self.collective_trace: list[dict[str, Any]] = []
        self.cache = PartitionedKVCache()
        self.rotary = QwenRotaryEmbedding(
            int(
                self.config.get("head_dim")
                or int(self.config["hidden_size"]) // int(self.config["num_attention_heads"])
            ),
            theta=float(self.config.get("rope_theta", 1_000_000.0)),
            device=self.device,
        )
        shard_lookup = {
            (item.stage_id, item.rank, item.tensor_name): item for item in self.plan.tensor_shards
        }
        layer_ranks: dict[int, list[TensorParallelQwenDecoderLayer]] = defaultdict(list)
        embeddings: list[VocabularyParallelEmbedding] = []
        lm_heads: list[VocabularyParallelLMHead] = []
        final_norms: list[QwenRMSNorm] = []
        consumed_by_rank: dict[tuple[int, int], set[str]] = defaultdict(set)
        expected_by_rank: dict[tuple[int, int], set[str]] = defaultdict(set)
        rank_states: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for stage in self.plan.pipeline_stages:
            for rank in range(self.tensor_parallel_degree):
                key = (stage.stage_id, rank)
                state = _load_rank_state(
                    _rank_weight_path(
                        self.shard_root,
                        stage_count=self.pipeline_stage_count,
                        stage_id=stage.stage_id,
                        rank=rank,
                    )
                )
                rank_states[key] = state
                expected_by_rank[key] = set(state)
                for layer_plan in stage.layer_plans:
                    assert layer_plan.mlp is not None
                    layer = TensorParallelQwenDecoderLayer(
                        config=self.config,
                        layer_id=layer_plan.layer_id,
                        rank=rank,
                        query_head_ids=layer_plan.attention.query_head_ownership[rank],
                        kv_head_ids=layer_plan.attention.kv_head_ownership[rank],
                        intermediate_range=layer_plan.mlp.intermediate_ranges[rank],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    consumed_by_rank[key].update(layer.load_local_state(state))
                    layer_ranks[layer_plan.layer_id].append(layer)
                if stage.owns_embeddings:
                    name = "model.embed_tokens.weight"
                    entry = shard_lookup[(stage.stage_id, rank, name)]
                    embeddings.append(
                        VocabularyParallelEmbedding(
                            state[name],
                            vocabulary_start=entry.shard_start,
                            vocabulary_end=entry.shard_end,
                            device=self.device,
                            dtype=self.dtype,
                        )
                    )
                    consumed_by_rank[key].add(name)
                if stage.owns_final_norm:
                    name = "model.norm.weight"
                    value = state[name]
                    norm = QwenRMSNorm(
                        int(self.config["hidden_size"]),
                        eps=float(self.config["rms_norm_eps"]),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    with torch.no_grad():
                        norm.weight.copy_(value.to(device=self.device, dtype=self.dtype))
                    final_norms.append(norm)
                    consumed_by_rank[key].add(name)
                if stage.owns_lm_head:
                    tied = bool(self.config.get("tie_word_embeddings", False))
                    name = (
                        "lm_head.weight"
                        if "lm_head.weight" in state
                        else "model.embed_tokens.weight"
                    )
                    entry = shard_lookup[(stage.stage_id, rank, name)]
                    shared: nn.Parameter | None = None
                    if stage.owns_embeddings and tied and name == "model.embed_tokens.weight":
                        shared = embeddings[rank].weight
                    lm_heads.append(
                        VocabularyParallelLMHead(
                            state[name],
                            vocabulary_start=entry.shard_start,
                            vocabulary_end=entry.shard_end,
                            device=self.device,
                            dtype=self.dtype,
                            shared_parameter=shared,
                        )
                    )
                    consumed_by_rank[key].add(name)
        for key, expected in expected_by_rank.items():
            missing = expected - consumed_by_rank[key]
            unexpected = consumed_by_rank[key] - expected
            if missing or unexpected:
                raise IntegrityError(
                    f"strict rank weight load failed for stage={key[0]} rank={key[1]}: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        self.layers = nn.ModuleDict()
        stage_by_layer = {
            layer.layer_id: stage.stage_id
            for stage in self.plan.pipeline_stages
            for layer in stage.layer_plans
        }
        self.stage_by_layer = stage_by_layer
        for layer_id in range(self.plan.layer_count):
            self.layers[str(layer_id)] = TensorParallelLayerGroup(
                layer_ranks[layer_id],
                stage_id=stage_by_layer[layer_id],
                trace=self.collective_trace,
                measure_collectives=measure_collectives,
                communication_format=communication_format,
            )
        self.embedding_ranks = nn.ModuleList(embeddings)
        self.final_norm_ranks = nn.ModuleList(final_norms)
        self.lm_head_ranks = nn.ModuleList(lm_heads)
        if len(self.embedding_ranks) != self.tensor_parallel_degree:
            raise IntegrityError("embedding ownership does not cover the first parallel cell")
        if len(self.final_norm_ranks) != self.tensor_parallel_degree:
            raise IntegrityError("final norm ownership does not cover the final parallel cell")
        if len(self.lm_head_ranks) != self.tensor_parallel_degree:
            raise IntegrityError("LM-head ownership does not cover the final parallel cell")
        self.eval()

    def _embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.plan.vocabulary_parallel:
            partials = [
                cast(VocabularyParallelEmbedding, rank)(token_ids) for rank in self.embedding_ranks
            ]
            result = partials[0]
            for partial in partials[1:]:
                result = result + partial
            return cast(torch.Tensor, result)
        return cast(
            torch.Tensor,
            cast(VocabularyParallelEmbedding, self.embedding_ranks[0])(token_ids),
        )

    def _causal_mask(
        self,
        *,
        batch_size: int,
        query_length: int,
        position_start: int,
    ) -> torch.Tensor | None:
        total_length = position_start + query_length
        if query_length == 1 and total_length >= 1:
            return None
        query_positions = torch.arange(
            position_start,
            position_start + query_length,
            device=self.device,
        )
        key_positions = torch.arange(total_length, device=self.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        minimum = torch.finfo(self.dtype).min
        mask = torch.full(
            (query_length, total_length), minimum, device=self.device, dtype=self.dtype
        )
        mask.masked_fill_(allowed, 0)
        return mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    @torch.inference_mode()
    def forward_hidden(
        self,
        token_ids: torch.Tensor,
        *,
        request_id: str,
        position_start: int,
        cache_generation: int = 0,
        capture_layers: set[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, LayerBoundaryCapture]]:
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        hidden = self._embedding(token_ids)
        position_ids = (
            torch.arange(
                position_start,
                position_start + token_ids.shape[1],
                device=self.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(token_ids.shape[0], -1)
        )
        cos, sin = self.rotary(hidden, position_ids)
        mask = self._causal_mask(
            batch_size=token_ids.shape[0],
            query_length=token_ids.shape[1],
            position_start=position_start,
        )
        captures: dict[int, LayerBoundaryCapture] = {}
        previous_stage = self.stage_by_layer[0]
        for layer_id in range(self.plan.layer_count):
            current_stage = self.stage_by_layer[layer_id]
            if current_stage != previous_stage:
                payload_bytes = int(hidden.numel() * hidden.element_size())
                self.collective_trace.extend(
                    [
                        {
                            "classification": "logical_single_gpu_measurement",
                            "event_type": "pipeline_hop_start",
                            "source_stage": previous_stage,
                            "destination_stage": current_stage,
                            "payload_bytes": payload_bytes,
                        },
                        {
                            "classification": "logical_single_gpu_measurement",
                            "event_type": "pipeline_hop_complete",
                            "source_stage": previous_stage,
                            "destination_stage": current_stage,
                            "payload_bytes": payload_bytes,
                            "actual_same_gpu_time_ms": 0.0,
                        },
                    ]
                )
                previous_stage = current_stage
            hidden, boundary = self.layers[str(layer_id)](
                hidden,
                cos=cos,
                sin=sin,
                attention_mask=mask,
                cache=self.cache,
                request_id=request_id,
                cache_generation=cache_generation,
                capture=capture_layers is not None and layer_id in capture_layers,
            )
            if boundary is not None:
                captures[layer_id] = boundary
        hidden = self.final_norm_ranks[0](hidden)
        return hidden, captures

    @torch.inference_mode()
    def greedy_token(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last = hidden_states[:, -1, :]
        if self.plan.vocabulary_parallel:
            candidates = [
                cast(VocabularyParallelLMHead, rank).local_argmax(last)
                for rank in self.lm_head_ranks
            ]
            _, tokens = distributed_argmax(candidates)
            payload_bytes = sum(
                int(values.numel() * values.element_size() + ids.numel() * ids.element_size())
                for values, ids in candidates
            )
            self.collective_trace.append(
                {
                    "classification": "logical_single_gpu_measurement",
                    "event_type": "collective_complete",
                    "operation": "distributed_argmax",
                    "phase": "lm_head",
                    "rank_ids": [f"rank-{rank:03d}" for rank in range(len(candidates))],
                    "payload_bytes": payload_bytes,
                    "aggregate_bytes": payload_bytes,
                    "actual_same_gpu_time_ms": 0.0,
                }
            )
            return tokens
        logits = cast(VocabularyParallelLMHead, self.lm_head_ranks[0])(last)
        return torch.argmax(logits, dim=-1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        request_id: str,
        cleanup: bool = True,
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the correctness generator accepts one request at a time")
        generated: list[int] = []
        hidden, _ = self.forward_hidden(
            input_ids,
            request_id=request_id,
            position_start=0,
        )
        next_token = self.greedy_token(hidden)
        generated.append(int(next_token.item()))
        position = int(input_ids.shape[1])
        for _ in range(max_new_tokens - 1):
            hidden, _ = self.forward_hidden(
                next_token.view(1, 1),
                request_id=request_id,
                position_start=position,
            )
            next_token = self.greedy_token(hidden)
            generated.append(int(next_token.item()))
            position += 1
        if cleanup:
            self.cache.cleanup(request_id)
        return generated

    @torch.inference_mode()
    def generate_concurrent(
        self,
        requests: list[tuple[str, torch.Tensor]],
        *,
        max_new_tokens: int,
        cleanup: bool = True,
    ) -> dict[str, list[int]]:
        """Interleave independent request caches in deterministic round-robin order."""

        if len(requests) < 2:
            raise ValueError("concurrent generation requires at least two requests")
        if len({request_id for request_id, _ in requests}) != len(requests):
            raise ValueError("concurrent request IDs must be unique")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        state: dict[str, tuple[torch.Tensor, int]] = {}
        result: dict[str, list[int]] = {}
        for request_id, input_ids in requests:
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("each concurrent request must have shape [1, sequence]")
            hidden, _ = self.forward_hidden(
                input_ids,
                request_id=request_id,
                position_start=0,
            )
            next_token = self.greedy_token(hidden)
            result[request_id] = [int(next_token.item())]
            state[request_id] = (next_token, int(input_ids.shape[1]))
        for _ in range(max_new_tokens - 1):
            for request_id, _ in requests:
                token, position = state[request_id]
                hidden, _ = self.forward_hidden(
                    token.view(1, 1),
                    request_id=request_id,
                    position_start=position,
                )
                next_token = self.greedy_token(hidden)
                result[request_id].append(int(next_token.item()))
                state[request_id] = (next_token, position + 1)
        if cleanup:
            for request_id, _ in requests:
                self.cache.cleanup(request_id)
        return result

    def logical_weight_bytes_by_rank(self) -> dict[str, int]:
        result: dict[str, int] = defaultdict(int)
        for shard in self.plan.tensor_shards:
            result[shard.logical_rank_id or f"stage-{shard.stage_id}-rank-{shard.rank}"] += (
                shard.logical_bytes
            )
        return dict(sorted(result.items()))

    def validate_rank_local_matrices(self) -> dict[str, Any]:
        matrix_markers = (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
            "lm_head.weight",
            "embed_tokens.weight",
        )
        violations = [
            shard.model_dump(mode="json")
            for shard in self.plan.tensor_shards
            if self.tensor_parallel_degree > 1
            and any(shard.tensor_name.endswith(marker) for marker in matrix_markers)
            and shard.local_shape == shard.global_shape
            and shard.partition_mode != "kv_head_replication"
        ]
        return {
            "status": "PASS" if not violations else "FAIL",
            "violations": violations,
            "rank_count": self.pipeline_stage_count * self.tensor_parallel_degree,
            "logical_layer_shards": self.plan.logical_layer_shards,
        }


def numerical_error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(actual.shape):
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "maximum_absolute_error": math.inf,
            "mean_absolute_error": math.inf,
            "maximum_relative_error": math.inf,
            "cosine_similarity": 0.0,
            "nan_count": int(torch.isnan(actual).sum().item()),
            "inf_count": int(torch.isinf(actual).sum().item()),
        }
    reference_float = reference.float()
    actual_float = actual.float()
    difference = (actual_float - reference_float).abs()
    relative = difference / reference_float.abs().clamp_min(1e-8)
    cosine = F.cosine_similarity(
        reference_float.reshape(1, -1), actual_float.reshape(1, -1), dim=-1
    )
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "maximum_relative_error": float(relative.max().item()),
        "cosine_similarity": float(cosine.item()),
        "nan_count": int(torch.isnan(actual).sum().item()),
        "inf_count": int(torch.isinf(actual).sum().item()),
    }
