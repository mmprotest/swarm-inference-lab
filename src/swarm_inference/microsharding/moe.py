"""Deterministic Qwen-compatible expert and expert-tensor parallelism."""

from __future__ import annotations

import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from swarm_inference.config.models import TensorSpec
from swarm_inference.microsharding.dense import ColumnParallelLinear, RowParallelLinear
from swarm_inference.microsharding.schemas import (
    CollectivePlan,
    DenseMLPPartitionPlan,
    MoEPartitionPlan,
    balanced_ranges,
)


@dataclass(frozen=True, slots=True)
class TinyMoEConfig:
    hidden_size: int = 32
    expert_intermediate_size: int = 64
    num_experts: int = 8
    top_k: int = 2
    norm_topk_prob: bool = True
    shared_expert_intermediate_size: int | None = 48
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.expert_intermediate_size <= 0:
            raise ValueError("MoE dimensions must be positive")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between one and num_experts")


@dataclass(slots=True)
class MoEFixtureState:
    router: torch.Tensor
    experts: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    input_norm: torch.Tensor
    shared_expert: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


def deterministic_moe_state(config: TinyMoEConfig, *, seed: int = 6006) -> MoEFixtureState:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def weight(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.08

    experts = {
        expert_id: (
            weight(config.expert_intermediate_size, config.hidden_size),
            weight(config.expert_intermediate_size, config.hidden_size),
            weight(config.hidden_size, config.expert_intermediate_size),
        )
        for expert_id in range(config.num_experts)
    }
    shared = None
    if config.shared_expert_intermediate_size is not None:
        size = config.shared_expert_intermediate_size
        shared = (
            weight(size, config.hidden_size),
            weight(size, config.hidden_size),
            weight(config.hidden_size, size),
        )
    return MoEFixtureState(
        router=weight(config.num_experts, config.hidden_size),
        experts=experts,
        input_norm=torch.ones(config.hidden_size, dtype=torch.float32),
        shared_expert=shared,
    )


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = value.dtype
    normalised = value.float()
    normalised = normalised * torch.rsqrt(normalised.pow(2).mean(-1, keepdim=True) + eps)
    return weight.to(device=value.device, dtype=dtype) * normalised.to(dtype)


def gated_mlp(
    value: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    gate, up, down = (item.to(device=value.device, dtype=value.dtype) for item in weights)
    return F.linear(F.silu(F.linear(value, gate)) * F.linear(value, up), down)


@dataclass(slots=True)
class MoEOutput:
    output: torch.Tensor
    router_logits: torch.Tensor
    routing_weights: torch.Tensor
    selected_experts: torch.Tensor
    metrics: dict[str, Any]


class ReplicatedMoEReference:
    def __init__(self, config: TinyMoEConfig, state: MoEFixtureState) -> None:
        self.config = config
        self.state = state

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        routing_override: torch.Tensor | None = None,
    ) -> MoEOutput:
        residual = hidden_states
        normalised = rms_norm(hidden_states, self.state.input_norm, self.config.rms_norm_eps)
        flat = normalised.reshape(-1, self.config.hidden_size)
        router_logits = F.linear(flat, self.state.router.to(flat.device, flat.dtype))
        probabilities = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected = torch.topk(probabilities, self.config.top_k, dim=-1)
        if routing_override is not None:
            if tuple(routing_override.shape) != tuple(selected.shape):
                raise ValueError("routing override shape mismatch")
            selected = routing_override.to(device=flat.device, dtype=torch.long)
            routing_weights = probabilities.gather(1, selected)
        if self.config.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        combined = torch.zeros_like(flat)
        for expert_id in range(self.config.num_experts):
            token_indices, slots = torch.where(selected == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = gated_mlp(flat[token_indices], self.state.experts[expert_id])
            combined.index_add_(
                0,
                token_indices,
                expert_output * routing_weights[token_indices, slots, None],
            )
        if self.state.shared_expert is not None:
            combined = combined + gated_mlp(flat, self.state.shared_expert)
        output = residual + combined.reshape_as(hidden_states)
        return MoEOutput(
            output=output,
            router_logits=router_logits,
            routing_weights=routing_weights,
            selected_experts=selected,
            metrics={"mode": "replicated_reference"},
        )


def expert_ownership(
    *,
    num_experts: int,
    expert_parallel_degree: int,
    strategy: Literal["contiguous", "round_robin", "load_balanced_from_trace"],
    routing_counts: dict[int, int] | None = None,
) -> dict[int, list[int]]:
    if not 1 <= expert_parallel_degree <= num_experts:
        raise ValueError("expert parallel degree must be between one and expert count")
    ownership: dict[int, list[int]] = {rank: [] for rank in range(expert_parallel_degree)}
    if strategy == "contiguous":
        for rank, (start, end) in enumerate(balanced_ranges(num_experts, expert_parallel_degree)):
            ownership[rank] = list(range(start, end))
    elif strategy == "round_robin":
        for expert_id in range(num_experts):
            ownership[expert_id % expert_parallel_degree].append(expert_id)
    elif strategy == "load_balanced_from_trace":
        counts = routing_counts or {expert_id: 1 for expert_id in range(num_experts)}
        loads = [0] * expert_parallel_degree
        ordered_experts = sorted(range(num_experts), key=lambda item: (-counts.get(item, 0), item))
        # Seed every rank with one expert before applying longest-processing-time
        # placement.  A sparse routing trace commonly contains zero-count cold
        # experts; without this seed all of those experts repeatedly select the
        # first zero-load rank and can leave otherwise useful ranks empty.
        for rank, expert_id in enumerate(ordered_experts[:expert_parallel_degree]):
            ownership[rank].append(expert_id)
            loads[rank] = counts.get(expert_id, 0)
        for expert_id in ordered_experts[expert_parallel_degree:]:
            rank = min(range(expert_parallel_degree), key=lambda item: (loads[item], item))
            ownership[rank].append(expert_id)
            loads[rank] += counts.get(expert_id, 0)
        for experts in ownership.values():
            experts.sort()
    else:
        raise ValueError(f"unsupported expert ownership strategy {strategy}")
    if any(not experts for experts in ownership.values()):
        raise ValueError("every expert-parallel rank must own at least one expert")
    if set().union(*map(set, ownership.values())) != set(range(num_experts)):
        raise AssertionError("expert ownership does not cover every expert")
    return ownership


class LocalExpertShard(nn.Module):
    def __init__(
        self,
        *,
        expert_id: int,
        hidden_size: int,
        global_intermediate_size: int,
        intermediate_range: tuple[int, int],
        tensor_rank: int,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        start, end = intermediate_range
        local_size = end - start
        self.expert_id = expert_id
        self.tensor_rank = tensor_rank
        self.intermediate_start = start
        self.intermediate_end = end
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            local_size,
            global_out_features=global_intermediate_size,
            shard_start=start,
            shard_end=end,
            device=device,
            dtype=dtype,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            local_size,
            global_out_features=global_intermediate_size,
            shard_start=start,
            shard_end=end,
            device=device,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            local_size,
            hidden_size,
            global_in_features=global_intermediate_size,
            shard_start=start,
            shard_end=end,
            device=device,
            dtype=dtype,
        )
        gate, up, down = weights
        self.gate_proj.load_local(gate[start:end, :])
        self.up_proj.load_local(up[start:end, :])
        self.down_proj.load_local(down[:, start:end])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value)).to(value.dtype),
        )

    @property
    def weight_bytes(self) -> int:
        return sum(int(item.numel() * item.element_size()) for item in self.parameters())


def build_moe_partition_plan(
    *,
    config: TinyMoEConfig,
    expert_parallel_degree: int,
    expert_tensor_parallel_degree: int,
    ownership_strategy: Literal["contiguous", "round_robin", "load_balanced_from_trace"],
    routing_counts: dict[int, int] | None = None,
) -> MoEPartitionPlan:
    ownership = expert_ownership(
        num_experts=config.num_experts,
        expert_parallel_degree=expert_parallel_degree,
        strategy=ownership_strategy,
        routing_counts=routing_counts,
    )
    logical_ownership = {
        expert_id: [rank] for rank, experts in ownership.items() for expert_id in experts
    }
    rank_ids = [f"ep-rank-{rank:03d}" for rank in range(expert_parallel_degree)]
    tensor_spec = TensorSpec(dtype="float32", shape=["assignments", config.hidden_size])
    dispatch = CollectivePlan(
        collective_id="moe-dispatch",
        operation="all_to_all",
        group_id="moe-ep-group",
        rank_ids=rank_ids,
        tensor_spec=tensor_spec,
        algorithm="ring",
        phase="expert_dispatch",
    )
    returned = CollectivePlan(
        collective_id="moe-return",
        operation="all_to_all",
        group_id="moe-ep-group",
        rank_ids=rank_ids,
        tensor_spec=tensor_spec,
        algorithm="ring",
        phase="expert_return",
    )
    shared = None
    if config.shared_expert_intermediate_size is not None:
        ranges = balanced_ranges(
            config.shared_expert_intermediate_size, expert_tensor_parallel_degree
        )
        shared = DenseMLPPartitionPlan(
            tensor_parallel_degree=expert_tensor_parallel_degree,
            intermediate_ranges={rank: interval for rank, interval in enumerate(ranges)},
            gate_projection_name="shared_expert.gate_proj.weight",
            up_projection_name="shared_expert.up_proj.weight",
            down_projection_name="shared_expert.down_proj.weight",
        )
    return MoEPartitionPlan(
        layer_id=0,
        mode=("expert_tensor_parallel" if expert_tensor_parallel_degree > 1 else "expert_parallel"),
        router_mode="replicated",
        router_owner_rank=None,
        router_replicated=True,
        top_k=config.top_k,
        expert_parallel_degree=expert_parallel_degree,
        expert_tensor_parallel_degree=expert_tensor_parallel_degree,
        expert_ownership=logical_ownership,
        shared_expert_plan=shared,
        collective_operations=[dispatch, returned],
    )


class ExpertParallelMoE:
    """Logical all-to-all dispatch over rank-local, optionally TP experts."""

    def __init__(
        self,
        config: TinyMoEConfig,
        state: MoEFixtureState,
        *,
        expert_parallel_degree: int,
        expert_tensor_parallel_degree: int = 1,
        ownership_strategy: Literal[
            "contiguous", "round_robin", "load_balanced_from_trace"
        ] = "contiguous",
        routing_counts: dict[int, int] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        apply_input_norm: bool = True,
        add_residual: bool = True,
    ) -> None:
        self.config = config
        self.state = state
        self.device = torch.device(device)
        self.dtype = dtype
        self.ep_degree = expert_parallel_degree
        self.etp_degree = expert_tensor_parallel_degree
        self.apply_input_norm = apply_input_norm
        self.add_residual = add_residual
        self.ownership_by_rank = expert_ownership(
            num_experts=config.num_experts,
            expert_parallel_degree=expert_parallel_degree,
            strategy=ownership_strategy,
            routing_counts=routing_counts,
        )
        self.owner_by_expert = {
            expert_id: rank
            for rank, experts in self.ownership_by_rank.items()
            for expert_id in experts
        }
        self.plan = build_moe_partition_plan(
            config=config,
            expert_parallel_degree=expert_parallel_degree,
            expert_tensor_parallel_degree=expert_tensor_parallel_degree,
            ownership_strategy=ownership_strategy,
            routing_counts=routing_counts,
        )
        ranges = balanced_ranges(config.expert_intermediate_size, expert_tensor_parallel_degree)
        self.rank_experts: dict[tuple[int, int], nn.ModuleDict] = {}
        for ep_rank, expert_ids in self.ownership_by_rank.items():
            for tensor_rank, interval in enumerate(ranges):
                modules = nn.ModuleDict(
                    {
                        str(expert_id): LocalExpertShard(
                            expert_id=expert_id,
                            hidden_size=config.hidden_size,
                            global_intermediate_size=config.expert_intermediate_size,
                            intermediate_range=interval,
                            tensor_rank=tensor_rank,
                            weights=state.experts[expert_id],
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for expert_id in expert_ids
                    }
                )
                self.rank_experts[(ep_rank, tensor_rank)] = modules
        self.shared_shards: nn.ModuleList | None = None
        if state.shared_expert is not None:
            shared_size = config.shared_expert_intermediate_size
            assert shared_size is not None
            shared_ranges = balanced_ranges(shared_size, expert_tensor_parallel_degree)
            self.shared_shards = nn.ModuleList(
                [
                    LocalExpertShard(
                        expert_id=-1,
                        hidden_size=config.hidden_size,
                        global_intermediate_size=shared_size,
                        intermediate_range=interval,
                        tensor_rank=rank,
                        weights=state.shared_expert,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    for rank, interval in enumerate(shared_ranges)
                ]
            )

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        routing_override: torch.Tensor | None = None,
    ) -> MoEOutput:
        def synchronize() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        synchronize()
        started = time.perf_counter_ns()
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        residual = hidden_states
        normalised = (
            rms_norm(hidden_states, self.state.input_norm, self.config.rms_norm_eps)
            if self.apply_input_norm
            else hidden_states
        )
        flat = normalised.reshape(-1, self.config.hidden_size)
        router_started = time.perf_counter_ns()
        router_logits = F.linear(flat, self.state.router.to(device=self.device, dtype=self.dtype))
        probabilities = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected = torch.topk(probabilities, self.config.top_k, dim=-1)
        if routing_override is not None:
            if tuple(routing_override.shape) != tuple(selected.shape):
                raise ValueError("routing override shape mismatch")
            selected = routing_override.to(device=self.device, dtype=torch.long)
            routing_weights = probabilities.gather(1, selected)
        if self.config.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)
        synchronize()
        routing_ms = (time.perf_counter_ns() - router_started) / 1_000_000
        packing_started = time.perf_counter_ns()
        dispatch: dict[int, list[tuple[int, int, int]]] = {
            rank: [] for rank in range(self.ep_degree)
        }
        for token in range(selected.shape[0]):
            for slot in range(selected.shape[1]):
                expert_id = int(selected[token, slot].item())
                dispatch[self.owner_by_expert[expert_id]].append((token, slot, expert_id))
        packing_ms = (time.perf_counter_ns() - packing_started) / 1_000_000
        bytes_per_assignment = self.config.hidden_size * flat.element_size() + 16
        dispatch_bytes = sum(len(items) for items in dispatch.values()) * bytes_per_assignment
        combined = torch.zeros_like(flat)
        compute_by_rank: dict[int, float] = {}
        combination_ms = 0.0
        returned_assignments = 0
        for ep_rank, assignments in dispatch.items():
            synchronize()
            rank_started = time.perf_counter_ns()
            by_expert: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for token, slot, expert_id in assignments:
                by_expert[expert_id].append((token, slot))
            rank_outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for expert_id, token_slots in by_expert.items():
                token_indices = torch.tensor(
                    [item[0] for item in token_slots], device=self.device, dtype=torch.long
                )
                slots = torch.tensor(
                    [item[1] for item in token_slots], device=self.device, dtype=torch.long
                )
                local_input = flat.index_select(0, token_indices)
                partials = [
                    self.rank_experts[(ep_rank, tensor_rank)][str(expert_id)](local_input)
                    for tensor_rank in range(self.etp_degree)
                ]
                expert_output = partials[0]
                for partial in partials[1:]:
                    expert_output = expert_output + partial
                rank_outputs.append((token_indices, slots, expert_output))
                returned_assignments += len(token_slots)
            synchronize()
            compute_by_rank[ep_rank] = (time.perf_counter_ns() - rank_started) / 1_000_000
            synchronize()
            combination_started = time.perf_counter_ns()
            for token_indices, slots, expert_output in rank_outputs:
                weighted = expert_output * routing_weights[token_indices, slots, None]
                combined.index_add_(0, token_indices, weighted)
            synchronize()
            combination_ms += (time.perf_counter_ns() - combination_started) / 1_000_000
        shared_compute_ms = 0.0
        if self.shared_shards is not None:
            synchronize()
            shared_started = time.perf_counter_ns()
            shared_partials = [shard(flat) for shard in self.shared_shards]
            shared_output = shared_partials[0]
            for partial in shared_partials[1:]:
                shared_output = shared_output + partial
            synchronize()
            shared_compute_ms = (time.perf_counter_ns() - shared_started) / 1_000_000
            synchronize()
            shared_combination_started = time.perf_counter_ns()
            combined = combined + shared_output
            synchronize()
            combination_ms += (time.perf_counter_ns() - shared_combination_started) / 1_000_000
        return_bytes = returned_assignments * self.config.hidden_size * flat.element_size()
        expert_result = combined.reshape_as(hidden_states)
        output = residual + expert_result if self.add_residual else expert_result
        active_ranks = [rank for rank, assignments in dispatch.items() if assignments]
        counts = [len(assignments) for assignments in dispatch.values()]
        mean_count = sum(counts) / max(len(counts), 1)
        imbalance = max(counts, default=0) / mean_count if mean_count else 0.0
        synchronize()
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        metrics = {
            "classification": "logical_microsharding_correctness",
            "tokens_dispatched_per_rank": {
                str(rank): len(assignments) for rank, assignments in dispatch.items()
            },
            "experts_selected_per_token": self.config.top_k,
            "active_ranks_per_token": [
                len({self.owner_by_expert[int(expert)] for expert in row})
                for row in selected.tolist()
            ],
            "selected_rank_fanout": len(active_ranks),
            "dispatch_bytes": dispatch_bytes,
            "return_bytes": return_bytes,
            "local_expert_compute_time_ms": compute_by_rank,
            "shared_expert_compute_time_ms": shared_compute_ms,
            "routing_time_ms": routing_ms,
            "packing_time_ms": packing_ms,
            "all_to_all_time_ms": 0.0,
            "expert_combination_time_ms": combination_ms,
            "expert_imbalance": imbalance,
            "idle_rank_fraction": 1 - len(active_ranks) / self.ep_degree,
            "same_gpu_wall_clock_ms": total_ms,
            "collectives": [
                {"operation": "all_to_all", "phase": "dispatch", "bytes": dispatch_bytes},
                {"operation": "all_to_all", "phase": "return", "bytes": return_bytes},
            ],
        }
        return MoEOutput(
            output=output,
            router_logits=router_logits,
            routing_weights=routing_weights,
            selected_experts=selected,
            metrics=metrics,
        )

    def memory_report(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for (ep_rank, tensor_rank), experts in sorted(self.rank_experts.items()):
            expert_bytes = sum(
                parameter.numel() * parameter.element_size()
                for module in experts.values()
                for parameter in module.parameters()
            )
            rows.append(
                {
                    "ep_rank": ep_rank,
                    "expert_tensor_rank": tensor_rank,
                    "expert_ids": [int(item) for item in experts],
                    "expert_count": len(experts),
                    "expert_bytes": int(expert_bytes),
                    "router_bytes": int(
                        self.state.router.numel() * self.state.router.element_size()
                    ),
                    "owns_all_experts": len(experts) == self.config.num_experts,
                    "owns_complete_expert_matrix": self.etp_degree == 1,
                }
            )
        return {
            "status": (
                "PASS"
                if self.ep_degree == 1 or all(not row["owns_all_experts"] for row in rows)
                else "FAIL"
            ),
            "ranks": rows,
            "maximum_expert_bytes_per_rank": max((row["expert_bytes"] for row in rows), default=0),
            "shared_expert_bytes": (
                sum(
                    parameter.numel() * parameter.element_size()
                    for module in self.shared_shards
                    for parameter in module.parameters()
                )
                if self.shared_shards is not None
                else 0
            ),
        }


def adversarial_routing(
    distribution: str,
    *,
    token_count: int,
    top_k: int,
    num_experts: int,
    ownership_by_rank: dict[int, list[int]],
) -> torch.Tensor:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if distribution == "uniform":
        values = [
            [(token * top_k + slot) % num_experts for slot in range(top_k)]
            for token in range(token_count)
        ]
    elif distribution == "highly_skewed":
        values = [[0, *range(1, top_k)] for _ in range(token_count)]
    elif distribution == "one_hot_expert":
        # top-k requires unique indices, so expert zero is always selected and
        # the remaining slots stay on the earliest experts.
        values = [list(range(top_k)) for _ in range(token_count)]
    elif distribution == "alternating_experts":
        values = [
            [((token % 2) * top_k + slot) % num_experts for slot in range(top_k)]
            for token in range(token_count)
        ]
    elif distribution == "all_selected_experts_on_one_rank":
        candidates = max(ownership_by_rank.values(), key=lambda experts: (len(experts), experts))
        selected = list(candidates[:top_k])
        if len(selected) < top_k:
            # Some valid Cartesian fixture points make literal single-rank
            # placement impossible (for example 8 experts, top-k 4, EP8).
            # Exercise the maximally concentrated feasible routing instead:
            # keep every expert from the fullest rank, then add the minimum
            # number of distinct remote experts needed by top-k semantics.
            selected.extend(
                expert_id for expert_id in range(num_experts) if expert_id not in selected
            )
        values = [selected[:top_k] for _ in range(token_count)]
    elif distribution == "maximum_rank_fanout":
        heads = [experts[0] for _, experts in sorted(ownership_by_rank.items())]
        values = [
            [heads[(token + slot) % len(heads)] for slot in range(top_k)]
            for token in range(token_count)
        ]
    else:
        raise ValueError(f"unknown routing distribution {distribution}")
    return torch.tensor(values, dtype=torch.long)


@dataclass(frozen=True, slots=True)
class ExpertCacheProfile:
    expert_cache_capacity_bytes: int
    local_storage_bandwidth_mbps: float
    peer_transfer_bandwidth_mbps: float
    peer_transfer_latency_ms: float
    expert_load_time_ms: float


def project_expert_cache(
    expert_trace: list[list[int]],
    *,
    expert_bytes: dict[int, int],
    profile: ExpertCacheProfile,
    policy: Literal["LRU", "LFU", "routing-prediction-prefetch", "hot-expert-pinning"],
) -> dict[str, Any]:
    cache: OrderedDict[int, None] = OrderedDict()
    frequency: Counter[int] = Counter()
    used_bytes = 0
    hits = misses = evictions = bytes_loaded = 0
    load_stall_ms = 0.0
    flattened = [expert for token in expert_trace for expert in token]
    hot = Counter(flattened).most_common()
    pinned: set[int] = set()
    if policy == "hot-expert-pinning":
        total = 0
        for expert_id, _ in hot:
            size = expert_bytes[expert_id]
            if total + size > profile.expert_cache_capacity_bytes:
                break
            pinned.add(expert_id)
            total += size

    def evict_for(size: int) -> None:
        nonlocal used_bytes, evictions
        while cache and used_bytes + size > profile.expert_cache_capacity_bytes:
            candidates = [expert for expert in cache if expert not in pinned]
            if not candidates:
                break
            if policy == "LFU":
                victim = min(
                    candidates, key=lambda item: (frequency[item], list(cache).index(item))
                )
            else:
                victim = candidates[0]
            cache.pop(victim)
            used_bytes -= expert_bytes[victim]
            evictions += 1

    for token_index, selected in enumerate(expert_trace):
        if policy == "routing-prediction-prefetch" and token_index + 1 < len(expert_trace):
            for predicted in expert_trace[token_index + 1]:
                if predicted in cache:
                    continue
                size = expert_bytes[predicted]
                if size <= profile.expert_cache_capacity_bytes:
                    evict_for(size)
                    cache[predicted] = None
                    used_bytes += size
                    bytes_loaded += size
        for expert_id in selected:
            frequency[expert_id] += 1
            if expert_id in cache:
                hits += 1
                cache.move_to_end(expert_id)
                continue
            misses += 1
            size = expert_bytes[expert_id]
            bytes_loaded += size
            storage_ms = size * 8 / (profile.local_storage_bandwidth_mbps * 1_000_000) * 1_000
            peer_ms = profile.peer_transfer_latency_ms + (
                size * 8 / (profile.peer_transfer_bandwidth_mbps * 1_000_000) * 1_000
            )
            load_stall_ms += profile.expert_load_time_ms + min(storage_ms, peer_ms)
            if size <= profile.expert_cache_capacity_bytes:
                evict_for(size)
                cache[expert_id] = None
                used_bytes += size
    total_accesses = hits + misses
    elapsed_seconds = max(load_stall_ms / 1_000, 1e-9)
    return {
        "classification": "independent_rank_projection",
        "policy": policy,
        "expert_cache_hit_rate": hits / total_accesses if total_accesses else 1.0,
        "hits": hits,
        "misses": misses,
        "load_stalls_ms": load_stall_ms,
        "bytes_loaded": bytes_loaded,
        "evictions": evictions,
        "projected_tokens_per_second_during_load": len(expert_trace) / elapsed_seconds,
        "minimum_node_memory_bytes": profile.expert_cache_capacity_bytes,
        "availability_impact": misses / total_accesses if total_accesses else 0.0,
    }


def tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | int | bool]:
    difference = (reference.float() - actual.float()).abs()
    relative = difference / reference.float().abs().clamp_min(1e-8)
    cosine = F.cosine_similarity(reference.float().reshape(1, -1), actual.float().reshape(1, -1))
    return {
        "shape_match": tuple(reference.shape) == tuple(actual.shape),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "maximum_relative_error": float(relative.max().item()),
        "cosine_similarity": float(cosine.item()),
        "nan_count": int(torch.isnan(actual).sum().item()),
        "inf_count": int(torch.isinf(actual).sum().item()),
    }
