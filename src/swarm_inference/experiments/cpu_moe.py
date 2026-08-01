"""Real Qwen3-MoE hybrid placement with selected experts on x86 CPU."""

from __future__ import annotations

import gc
import math
import random
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

from swarm_inference.microsharding.real_moe import RealMoEDownloadPlan, _load_layer_state

ExpertWeights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ExpertFormat = Literal["BF16", "INT8", "Q4"]
PlacementPolicy = Literal[
    "coldest_experts_on_cpu",
    "hottest_experts_on_cpu",
    "random_experts_on_cpu",
    "load_balanced",
    "predicted_next_experts",
]


@dataclass(slots=True)
class QuantizedWeight:
    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    bits: int
    quantisation_ms: float

    @property
    def storage_bytes(self) -> int:
        return int(self.values.numel() * self.values.element_size()) + int(
            self.scales.numel() * self.scales.element_size()
        )

    def dequantize(self, *, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
        started = time.perf_counter()
        if self.bits == 4:
            flat = self.values.flatten()
            low = (flat & 0x0F).to(torch.int8)
            high = ((flat >> 4) & 0x0F).to(torch.int8)
            unpacked = torch.empty(flat.numel() * 2, dtype=torch.int8)
            unpacked[0::2] = low
            unpacked[1::2] = high
            signed = unpacked[: math.prod(self.shape)].reshape(self.shape) - 8
        else:
            signed = self.values.reshape(self.shape)
        result = signed.float() * self.scales[:, None]
        return result.to(dtype), (time.perf_counter() - started) * 1000


def quantize_matrix(weight: torch.Tensor, bits: Literal[4, 8]) -> QuantizedWeight:
    if weight.ndim != 2:
        raise ValueError("expert quantisation requires a matrix")
    started = time.perf_counter()
    maximum = 7 if bits == 4 else 127
    source = weight.float().cpu()
    scales = source.abs().amax(dim=1).clamp_min(1e-12) / maximum
    quantized = torch.round(source / scales[:, None]).clamp(-maximum, maximum).to(torch.int8)
    if bits == 4:
        unsigned = (quantized + 8).to(torch.uint8).flatten()
        if unsigned.numel() % 2:
            unsigned = torch.cat((unsigned, torch.zeros(1, dtype=torch.uint8)))
        values = unsigned[0::2] | (unsigned[1::2] << 4)
    else:
        values = quantized
    return QuantizedWeight(
        values=values,
        scales=scales,
        shape=(int(source.shape[0]), int(source.shape[1])),
        bits=bits,
        quantisation_ms=(time.perf_counter() - started) * 1000,
    )


@dataclass(slots=True)
class StoredExpert:
    expert_id: int
    weight_format: ExpertFormat
    matrices: tuple[torch.Tensor | QuantizedWeight, ...]
    source_bytes: int
    storage_bytes: int
    quantisation_ms: float
    maximum_weight_error: float


class CpuExpertCache:
    """Bounded real-weight cache; misses quantise/load canonical expert weights."""

    def __init__(
        self,
        source: dict[int, ExpertWeights],
        *,
        capacity: int,
        weight_format: ExpertFormat,
    ) -> None:
        if capacity <= 0:
            raise ValueError("expert cache capacity must be positive")
        self.source = source
        self.capacity = capacity
        self.weight_format = weight_format
        self.entries: OrderedDict[int, StoredExpert] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.load_ms = 0.0

    def _load(self, expert_id: int) -> StoredExpert:
        started = time.perf_counter()
        source = self.source[expert_id]
        source_bytes = sum(int(item.numel() * item.element_size()) for item in source)
        quantisation_ms = 0.0
        maximum_error = 0.0
        if self.weight_format == "BF16":
            matrices: tuple[torch.Tensor | QuantizedWeight, ...] = tuple(
                item.to(dtype=torch.bfloat16, device="cpu").contiguous() for item in source
            )
            storage_bytes = source_bytes
        else:
            bits: Literal[4, 8] = 8 if self.weight_format == "INT8" else 4
            packed = tuple(quantize_matrix(item, bits) for item in source)
            matrices = packed
            storage_bytes = sum(item.storage_bytes for item in packed)
            quantisation_ms = sum(item.quantisation_ms for item in packed)
            for original, item in zip(source, packed, strict=True):
                restored, _ = item.dequantize(dtype=torch.float32)
                maximum_error = max(
                    maximum_error,
                    float((restored - original.float().cpu()).abs().max().item()),
                )
        self.load_ms += (time.perf_counter() - started) * 1000
        return StoredExpert(
            expert_id=expert_id,
            weight_format=self.weight_format,
            matrices=matrices,
            source_bytes=source_bytes,
            storage_bytes=storage_bytes,
            quantisation_ms=quantisation_ms,
            maximum_weight_error=maximum_error,
        )

    def get(self, expert_id: int) -> StoredExpert:
        cached = self.entries.pop(expert_id, None)
        if cached is not None:
            self.hits += 1
            self.entries[expert_id] = cached
            return cached
        self.misses += 1
        loaded = self._load(expert_id)
        self.entries[expert_id] = loaded
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.evictions += 1
        return loaded

    def prefetch(self, expert_ids: list[int]) -> float:
        started = time.perf_counter()
        for expert_id in expert_ids:
            self.get(expert_id)
        return (time.perf_counter() - started) * 1000

    def metrics(self) -> dict[str, Any]:
        accesses = self.hits + self.misses
        return {
            "expert_cache_hits": self.hits,
            "expert_cache_misses": self.misses,
            "expert_cache_hit_rate": self.hits / max(accesses, 1),
            "expert_cache_evictions": self.evictions,
            "expert_cache_load_ms": self.load_ms,
            "expert_cache_bytes": sum(item.storage_bytes for item in self.entries.values()),
        }


def select_cpu_experts(
    policy: PlacementPolicy,
    *,
    count: int,
    num_experts: int,
    routing_counts: dict[int, int],
    predicted_counts: dict[int, int] | None = None,
    seed: int = 7007,
) -> list[int]:
    if not 1 <= count < num_experts:
        raise ValueError("CPU must own at least one but not the complete expert set")
    ids = list(range(num_experts))
    if policy == "coldest_experts_on_cpu":
        ordered = sorted(ids, key=lambda item: (routing_counts.get(item, 0), item))
    elif policy == "hottest_experts_on_cpu":
        ordered = sorted(ids, key=lambda item: (-routing_counts.get(item, 0), item))
    elif policy == "random_experts_on_cpu":
        ordered = ids.copy()
        random.Random(seed).shuffle(ordered)
    elif policy == "load_balanced":
        mean = sum(routing_counts.values()) / max(num_experts, 1)
        ordered = sorted(ids, key=lambda item: (abs(routing_counts.get(item, 0) - mean), item))
    elif policy == "predicted_next_experts":
        predictions = predicted_counts or routing_counts
        ordered = sorted(ids, key=lambda item: (-predictions.get(item, 0), item))
    else:
        raise ValueError(f"unknown expert placement policy {policy!r}")
    return sorted(ordered[:count])


def _stored_expert_call(
    value: torch.Tensor,
    expert: StoredExpert,
) -> tuple[torch.Tensor, float]:
    matrices: list[torch.Tensor] = []
    dequantisation_ms = 0.0
    for item in expert.matrices:
        if isinstance(item, QuantizedWeight):
            matrix, duration = item.dequantize(dtype=value.dtype)
            matrices.append(matrix)
            dequantisation_ms += duration
        else:
            matrices.append(item.to(dtype=value.dtype))
    gate, up, down = matrices
    return F.linear(F.silu(F.linear(value, gate)) * F.linear(value, up), down), dequantisation_ms


@dataclass(slots=True)
class RealMoeFixture:
    plan: RealMoEDownloadPlan
    config: Any
    expert_input: torch.Tensor
    post_attention: torch.Tensor
    reference_expert_output: torch.Tensor
    reference_layer_output: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor
    expert_weights_cpu: dict[int, ExpertWeights]
    expert_weights_gpu: dict[int, ExpertWeights]
    common_component_ms: float
    baseline_expert_ms: float
    baseline_layer_ms: float
    baseline_gpu_memory_bytes: int

    def release(self) -> None:
        self.expert_weights_gpu.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def prepare_real_moe_fixture(
    plan: RealMoEDownloadPlan,
    files: list[Any],
    *,
    sequence_length: int = 64,
) -> RealMoeFixture:
    from pathlib import Path

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("real hybrid MoE measurement requires CUDA")
    config_payload = __import__("json").loads(Path(plan.config_path).read_text(encoding="utf-8"))
    config = Qwen3MoeConfig.from_dict(config_payload)
    config._attn_implementation = "eager"
    source = _load_layer_state(plan, [Path(item) for item in files])
    prefix = f"model.layers.{plan.selected_layer}."
    local_state = {name.removeprefix(prefix): value for name, value in source.items()}
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(7007)
    hidden = torch.randn(
        (1, sequence_length, int(config.hidden_size)),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    layer = Qwen3MoeDecoderLayer(config, plan.selected_layer).to(device=device, dtype=dtype)
    incompatible = layer.load_state_dict(local_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict real MoE layer load failed: {incompatible}")
    layer.eval()
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
    rotary = Qwen3MoeRotaryEmbedding(config=config).to(device)
    cos, sin = rotary(hidden, position_ids)
    minimum = torch.finfo(dtype).min
    causal = torch.full((sequence_length, sequence_length), minimum, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)[None, None, :, :]
    common_started = time.perf_counter()
    normalised = layer.input_layernorm(hidden)
    attention_output, _ = layer.self_attn(
        hidden_states=normalised,
        position_embeddings=(cos, sin),
        attention_mask=causal,
        position_ids=position_ids,
    )
    post_attention = hidden + attention_output
    expert_input = layer.post_attention_layernorm(post_attention)
    torch.cuda.synchronize()
    common_ms = (time.perf_counter() - common_started) * 1000
    expert_started = time.perf_counter()
    reference_expert, router_logits = layer.mlp(expert_input)
    torch.cuda.synchronize()
    expert_ms = (time.perf_counter() - expert_started) * 1000
    probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    routing_weights, selected = torch.topk(probabilities, int(config.num_experts_per_tok), dim=-1)
    if bool(config.norm_topk_prob):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(dtype)
    expert_weights_cpu = {
        expert_id: (
            source[f"{prefix}mlp.experts.{expert_id}.gate_proj.weight"].cpu(),
            source[f"{prefix}mlp.experts.{expert_id}.up_proj.weight"].cpu(),
            source[f"{prefix}mlp.experts.{expert_id}.down_proj.weight"].cpu(),
        )
        for expert_id in range(int(config.num_experts))
    }
    expert_weights_gpu: dict[int, ExpertWeights] = {
        expert_id: (
            weights[0].to(device=device, dtype=dtype),
            weights[1].to(device=device, dtype=dtype),
            weights[2].to(device=device, dtype=dtype),
        )
        for expert_id, weights in expert_weights_cpu.items()
    }
    baseline_memory = sum(
        int(item.numel() * item.element_size())
        for weights in expert_weights_gpu.values()
        for item in weights
    )
    del layer, rotary, source, local_state, normalised, attention_output
    gc.collect()
    torch.cuda.empty_cache()
    return RealMoeFixture(
        plan=plan,
        config=config,
        expert_input=expert_input,
        post_attention=post_attention,
        reference_expert_output=reference_expert,
        reference_layer_output=post_attention + reference_expert,
        routing_weights=routing_weights,
        selected_experts=selected,
        expert_weights_cpu=expert_weights_cpu,
        expert_weights_gpu=expert_weights_gpu,
        common_component_ms=common_ms,
        baseline_expert_ms=expert_ms,
        baseline_layer_ms=common_ms + expert_ms,
        baseline_gpu_memory_bytes=baseline_memory,
    )


def _gpu_expert(value: torch.Tensor, weights: ExpertWeights) -> torch.Tensor:
    gate, up, down = weights
    return F.linear(F.silu(F.linear(value, gate)) * F.linear(value, up), down)


@torch.inference_mode()
def run_hybrid_cpu_experts(
    fixture: RealMoeFixture,
    *,
    cpu_expert_ids: list[int],
    weight_format: ExpertFormat,
    prefetch_expert_ids: list[int] | None = None,
) -> dict[str, Any]:
    num_experts = int(fixture.config.num_experts)
    if not cpu_expert_ids or len(cpu_expert_ids) >= num_experts:
        raise ValueError("CPU rank may not own zero or the complete expert set")
    owned = set(cpu_expert_ids)
    cache = CpuExpertCache(
        fixture.expert_weights_cpu,
        capacity=len(cpu_expert_ids),
        weight_format=weight_format,
    )
    prefetch_ms = cache.prefetch(prefetch_expert_ids or cpu_expert_ids)
    selected = fixture.selected_experts
    routing_weights = fixture.routing_weights
    flat = fixture.expert_input.reshape(-1, int(fixture.config.hidden_size))
    combined = torch.zeros_like(flat)
    dispatch_bytes = 0
    return_bytes = 0
    cpu_compute_ms = 0.0
    gpu_compute_ms = 0.0
    cpu_transfer_ms = 0.0
    dequantisation_ms = 0.0
    quantisation_ms = sum(item.quantisation_ms for item in cache.entries.values())
    selected_cpu_calls = 0
    hybrid_started = time.perf_counter()
    assignments: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for token in range(int(selected.shape[0])):
        for slot in range(int(selected.shape[1])):
            assignments[int(selected[token, slot].item())].append((token, slot))
    for expert_id, token_slots in assignments.items():
        token_indices_gpu = torch.tensor(
            [item[0] for item in token_slots], device="cuda", dtype=torch.long
        )
        slots_gpu = torch.tensor([item[1] for item in token_slots], device="cuda", dtype=torch.long)
        local_input = flat.index_select(0, token_indices_gpu)
        if expert_id in owned:
            selected_cpu_calls += len(token_slots)
            transfer_started = time.perf_counter()
            local_cpu = local_input.to(device="cpu", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            cpu_transfer_ms += (time.perf_counter() - transfer_started) * 1000
            dispatch_bytes += int(local_cpu.numel() * local_cpu.element_size())
            cpu_started = time.perf_counter()
            expert_output_cpu, dequant_ms = _stored_expert_call(local_cpu, cache.get(expert_id))
            cpu_compute_ms += (time.perf_counter() - cpu_started) * 1000
            dequantisation_ms += dequant_ms
            transfer_started = time.perf_counter()
            expert_output = expert_output_cpu.to(device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            cpu_transfer_ms += (time.perf_counter() - transfer_started) * 1000
            return_bytes += int(expert_output_cpu.numel() * expert_output_cpu.element_size())
        else:
            gpu_started = time.perf_counter()
            expert_output = _gpu_expert(local_input, fixture.expert_weights_gpu[expert_id])
            torch.cuda.synchronize()
            gpu_compute_ms += (time.perf_counter() - gpu_started) * 1000
        weighted = expert_output * routing_weights[token_indices_gpu, slots_gpu, None]
        combined.index_add_(0, token_indices_gpu, weighted)
    torch.cuda.synchronize()
    hybrid_expert_ms = (time.perf_counter() - hybrid_started) * 1000
    hybrid_output = fixture.post_attention + combined.reshape_as(fixture.expert_input)
    error = (hybrid_output.float() - fixture.reference_layer_output.float()).abs()
    maximum_error = float(error.max().item())
    mean_error = float(error.mean().item())
    exact = bool(torch.equal(hybrid_output, fixture.reference_layer_output))
    cosine = float(
        F.cosine_similarity(
            hybrid_output.float().reshape(1, -1),
            fixture.reference_layer_output.float().reshape(1, -1),
        ).item()
    )
    cpu_memory = sum(item.storage_bytes for item in cache.entries.values())
    gpu_saved = sum(
        int(item.numel() * item.element_size())
        for expert_id in owned
        for item in fixture.expert_weights_gpu[expert_id]
    )
    hybrid_layer_ms = fixture.common_component_ms + hybrid_expert_ms
    retained = fixture.baseline_layer_ms / max(hybrid_layer_ms, 1e-12)
    routing_counts = Counter(int(item) for item in selected.flatten().tolist())
    cache_metrics = cache.metrics()
    tolerance = 0.02 if weight_format == "BF16" else (0.25 if weight_format == "INT8" else 0.75)
    passed = maximum_error <= tolerance and cosine >= 0.99
    return {
        "classification": "measured_mixed_backend",
        "model_id": fixture.plan.model_id,
        "revision": fixture.plan.revision,
        "layer_id": fixture.plan.selected_layer,
        "cpu_expert_ids": cpu_expert_ids,
        "cpu_expert_count": len(cpu_expert_ids),
        "routed_expert_count": num_experts,
        "cpu_rank_owns_complete_expert_set": False,
        "weight_format": weight_format,
        "status": "PASS" if passed else "FAIL",
        "comparison_reference": "full_gpu_real_layer",
        "comparison_performed": True,
        "exact_layer_output": exact,
        "bitwise_exact_layer_output": exact,
        "numerical_tolerance": tolerance,
        "numerical_tolerance_pass": passed,
        "maximum_absolute_error": maximum_error,
        "mean_absolute_error": mean_error,
        "cosine_similarity": cosine,
        "quantisation_time_ms": quantisation_ms,
        "dequantisation_time_ms": dequantisation_ms,
        "maximum_quantised_weight_error": max(
            (item.maximum_weight_error for item in cache.entries.values()), default=0.0
        ),
        "gpu_memory_saved_bytes": gpu_saved,
        "baseline_gpu_expert_memory_bytes": fixture.baseline_gpu_memory_bytes,
        "cpu_memory_bytes": cpu_memory,
        "dispatch_bytes": dispatch_bytes,
        "return_bytes": return_bytes,
        "cpu_expert_latency_ms": cpu_compute_ms,
        "cpu_gpu_transfer_ms": cpu_transfer_ms,
        "gpu_expert_latency_ms": gpu_compute_ms,
        "common_component_ms": fixture.common_component_ms,
        "baseline_layer_latency_ms": fixture.baseline_layer_ms,
        "hybrid_layer_latency_ms": hybrid_layer_ms,
        "baseline_layer_throughput": 1000 / fixture.baseline_layer_ms,
        "hybrid_layer_throughput": 1000 / hybrid_layer_ms,
        "throughput_retained_fraction": retained,
        "selected_cpu_expert_frequency": selected_cpu_calls / max(selected.numel(), 1),
        "selected_cpu_expert_calls": selected_cpu_calls,
        "router_imbalance": max(routing_counts.values(), default=0)
        / max(sum(routing_counts.values()) / max(len(routing_counts), 1), 1e-12),
        "expert_prefetch_ms": prefetch_ms,
        "expert_prefetch_useful": prefetch_ms < cpu_compute_ms and selected_cpu_calls > 0,
        **cache_metrics,
    }
