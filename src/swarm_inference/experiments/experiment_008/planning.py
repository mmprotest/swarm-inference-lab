"""Capability-aware cumulative plans and bounded baseline search for Experiment 008."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Any, Literal

from pydantic import Field

from swarm_inference.config.experiment_008 import Experiment008BaselineSearchConfig
from swarm_inference.config.models import StrictModel
from swarm_inference.experiments.experiment_008.schemas import (
    EvidenceClass,
    ExecutionStatus,
    PhasePlan,
    TechniqueDecision,
    TensorPlacement,
    TensorTile,
)


class BackendCapabilities(StrictModel):
    conventional_layer_offload: bool
    tensor_buffer_override: bool
    cpu_moe: bool
    asynchronous_backend_scheduler: bool
    operation_level_overlap_trace: bool
    expert_routing_trace: bool
    per_expert_dynamic_residency: bool
    expert_prefetch: bool
    separate_process_phase_plans: bool
    in_request_phase_switch: bool
    deterministic_greedy_tokens: bool
    final_logits: bool
    limitations: list[str] = Field(default_factory=list)


class BaselineCandidate(StrictModel):
    candidate_id: str
    gpu_layers: int | Literal["auto", "all"]
    cpu_threads: int
    batch_size: int
    microbatch_size: int
    memory_map: bool
    flash_attention: bool
    cpu_moe_layers: int
    backend_arguments: list[str]


def _candidate_id(payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"stock-{digest}"


def baseline_search_space(
    config: Experiment008BaselineSearchConfig, *, seed: int
) -> list[BaselineCandidate]:
    combinations = list(
        itertools.product(
            config.gpu_layers,
            config.cpu_threads,
            config.batch_sizes,
            config.microbatch_sizes,
            config.memory_map,
            config.flash_attention,
            config.cpu_moe_layers,
        )
    )
    valid = [item for item in combinations if item[3] <= item[2]]
    generator = random.Random(seed)
    generator.shuffle(valid)
    equal_cpu_moe = min(
        config.cpu_moe_layers,
        key=lambda value: abs(value - max(config.cpu_moe_layers) / 2),
    )
    largest_batch = max(config.batch_sizes)
    largest_ubatch = max(value for value in config.microbatch_sizes if value <= largest_batch)
    anchors = [
        (
            "auto",
            max(config.cpu_threads),
            largest_batch,
            largest_ubatch,
            True,
            True,
            cpu_moe_layers,
        )
        for cpu_moe_layers in (
            min(config.cpu_moe_layers),
            equal_cpu_moe,
            max(config.cpu_moe_layers),
        )
    ]
    dimensions = [
        set(config.gpu_layers),
        set(config.cpu_threads),
        set(config.batch_sizes),
        set(config.microbatch_sizes),
        set(config.memory_map),
        set(config.flash_attention),
        set(config.cpu_moe_layers),
    ]
    uncovered = {
        (dimension_index, value)
        for dimension_index, values in enumerate(dimensions)
        for value in values
    }
    selected: list[tuple[Any, ...]] = []
    for item in anchors:
        if item in valid and item not in selected:
            selected.append(item)
            uncovered.difference_update(enumerate(item))
    while len(selected) < config.maximum_candidates:
        remaining = [item for item in valid if item not in selected]
        if not remaining:
            break
        item = max(
            remaining,
            key=lambda candidate: sum(
                (index, value) in uncovered for index, value in enumerate(candidate)
            ),
        )
        selected.append(item)
        uncovered.difference_update(enumerate(item))
    rows: list[BaselineCandidate] = []
    for (
        gpu_layers,
        threads,
        batch,
        microbatch,
        memory_map,
        flash_attention,
        cpu_moe_layers,
    ) in selected:
        payload = {
            "gpu_layers": gpu_layers,
            "cpu_threads": threads,
            "batch_size": batch,
            "microbatch_size": microbatch,
            "memory_map": memory_map,
            "flash_attention": flash_attention,
            "cpu_moe_layers": cpu_moe_layers,
        }
        arguments = [
            "--n-gpu-layers",
            str(gpu_layers),
            "--threads",
            str(threads),
            "--threads-batch",
            str(threads),
            "--batch-size",
            str(batch),
            "--ubatch-size",
            str(microbatch),
            "--flash-attn",
            "on" if flash_attention else "off",
        ]
        if not memory_map:
            arguments.append("--no-mmap")
        if cpu_moe_layers > 0:
            arguments.extend(["--n-cpu-moe", str(cpu_moe_layers)])
        rows.append(
            BaselineCandidate(
                candidate_id=_candidate_id(payload), **payload, backend_arguments=arguments
            )
        )
    return rows


def select_best_stock_by_workload(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    completed = [
        row
        for row in rows
        if row.get("status") == "COMPLETED" and row.get("classification") == "MEASURED"
    ]

    def best(workload: str, metric: str, *, minimum: bool) -> dict[str, Any] | None:
        matching = [
            row
            for row in completed
            if row.get("workload") == workload and isinstance(row.get(metric), (int, float))
        ]
        if not matching:
            return None
        return (
            min(matching, key=lambda row: float(row[metric]))
            if minimum
            else max(matching, key=lambda row: float(row[metric]))
        )

    return {
        "decode": best("decode", "decode_tokens_per_second", minimum=False),
        "prefill_8k": best("prefill_8k", "time_to_first_token_ms", minimum=True),
        "prefill_32k": best("prefill_32k", "time_to_first_token_ms", minimum=True),
        "mixed": best("mixed", "combined_generated_tokens_per_second", minimum=False),
    }


_TECHNIQUE_BY_CONFIGURATION: dict[str, list[str]] = {
    "A": ["stock_offloading"],
    "B": ["stock_offloading", "tensor_granular_placement", "asymmetric_cpu_gpu_partition"],
    "C": [
        "stock_offloading",
        "tensor_granular_placement",
        "asymmetric_cpu_gpu_partition",
        "asynchronous_cpu_gpu_overlap",
    ],
    "D": [
        "stock_offloading",
        "tensor_granular_placement",
        "asymmetric_cpu_gpu_partition",
        "asynchronous_cpu_gpu_overlap",
        "activation_aware_expert_cache",
    ],
    "E": [
        "stock_offloading",
        "tensor_granular_placement",
        "asymmetric_cpu_gpu_partition",
        "asynchronous_cpu_gpu_overlap",
        "activation_aware_expert_cache",
        "predictive_expert_prefetch",
    ],
    "F": [
        "stock_offloading",
        "tensor_granular_placement",
        "asymmetric_cpu_gpu_partition",
        "asynchronous_cpu_gpu_overlap",
        "activation_aware_expert_cache",
        "predictive_expert_prefetch",
        "separate_prefill_decode_plans",
    ],
}


def _supported(technique: str, capabilities: BackendCapabilities) -> tuple[bool, str]:
    requirements = {
        "stock_offloading": capabilities.conventional_layer_offload,
        "tensor_granular_placement": capabilities.tensor_buffer_override,
        "asymmetric_cpu_gpu_partition": capabilities.cpu_moe,
        "asynchronous_cpu_gpu_overlap": (
            capabilities.asynchronous_backend_scheduler
            and capabilities.operation_level_overlap_trace
        ),
        "activation_aware_expert_cache": (
            capabilities.expert_routing_trace and capabilities.per_expert_dynamic_residency
        ),
        "predictive_expert_prefetch": (
            capabilities.expert_routing_trace
            and capabilities.per_expert_dynamic_residency
            and capabilities.expert_prefetch
        ),
        "separate_prefill_decode_plans": capabilities.separate_process_phase_plans,
    }
    supported = requirements[technique]
    return supported, (
        "backend capability is available"
        if supported
        else f"selected backend does not expose the hooks required for {technique}"
    )


_PERFORMANCE_CRITICAL_OVERRIDE = ".*ffn_gate_inp.*=CUDA0,.*norm.*=CUDA0"


def _placements(
    tiles: list[TensorTile], *, cpu_moe: bool, tensor_override: bool
) -> list[TensorPlacement]:
    totals: dict[tuple[str, str, str], int] = {}
    for tile in tiles:
        if cpu_moe and tile.tensor_role.startswith("routed_expert"):
            residency, device = "CPU", "CPU"
        elif tensor_override and tile.tensor_role in {"router", "normalisation"}:
            residency, device = "GPU", "GPU"
        else:
            residency, device = "BACKEND_MANAGED", "BACKEND_SELECTED"
        key = (tile.tensor_role, residency, device)
        totals[key] = totals.get(key, 0) + tile.byte_size
    return [
        TensorPlacement(
            tensor_pattern=role,
            tensor_role=role,
            residency=residency,  # type: ignore[arg-type]
            execution_device=device,  # type: ignore[arg-type]
            byte_size=byte_size,
            reason=(
                "routed experts are large and sparse; CPU placement is evaluated against transfer and CPU execution measurements"
                if residency == "CPU"
                else "small, per-token router or normalisation tensors are explicitly retained on GPU"
                if residency == "GPU"
                else "coarse layer residency remains under the measured stock backend plan; no unsupported exact placement is claimed"
            ),
        )
        for (role, residency, device), byte_size in sorted(totals.items())
    ]


def _replace_valued_option(arguments: list[str], option: str, value: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] == option:
            index += 2
            continue
        result.append(arguments[index])
        index += 1
    return [*result, option, value]


def build_phase_plan(
    *,
    configuration: Literal["A", "B", "C", "D", "E", "F", "G"],
    phase: Literal["prefill", "decode", "mixed"],
    capabilities: BackendCapabilities,
    tiles: list[TensorTile],
    stock_arguments: list[str],
    cpu_moe_layers: int,
    measured_utility_by_technique: dict[str, float | None] | None = None,
) -> PhasePlan:
    measured_utility_by_technique = measured_utility_by_technique or {}
    requested = (
        _TECHNIQUE_BY_CONFIGURATION[configuration]
        if configuration != "G"
        else list(_TECHNIQUE_BY_CONFIGURATION["F"])
    )
    decisions: list[TechniqueDecision] = []
    for technique in requested:
        supported, support_reason = _supported(technique, capabilities)
        measured = measured_utility_by_technique.get(technique)
        positive = measured is not None and measured > 0
        if configuration == "G":
            enabled = supported and (technique == "stock_offloading" or positive)
            if not supported:
                reason = support_reason
                status = ExecutionStatus.UNSUPPORTED
            elif technique == "stock_offloading":
                reason = "stock execution remains the reference foundation"
                status = ExecutionStatus.COMPLETED
            elif positive:
                reason = (
                    f"enabled because measured incremental utility was positive ({measured:.6f})"
                )
                status = ExecutionStatus.COMPLETED
            else:
                reason = (
                    "disabled because measured incremental utility was non-positive or unavailable"
                )
                status = (
                    ExecutionStatus.COMPLETED if measured is not None else ExecutionStatus.NOT_RUN
                )
        else:
            enabled = supported
            reason = support_reason
            status = ExecutionStatus.COMPLETED if supported else ExecutionStatus.UNSUPPORTED
        decisions.append(
            TechniqueDecision(
                technique=technique,
                enabled=enabled,
                execution_status=status,
                evidence_class=EvidenceClass.MEASURED if measured is not None else None,
                measured_utility=measured,
                reason=reason,
            )
        )
    enabled = {decision.technique for decision in decisions if decision.enabled}
    arguments = list(stock_arguments)
    use_tensor_override = "tensor_granular_placement" in enabled
    if use_tensor_override:
        arguments = _replace_valued_option(
            arguments, "--override-tensor", _PERFORMANCE_CRITICAL_OVERRIDE
        )
    use_cpu_moe = "asymmetric_cpu_gpu_partition" in enabled and cpu_moe_layers > 0
    if use_cpu_moe:
        arguments = _replace_valued_option(arguments, "--n-cpu-moe", str(cpu_moe_layers))
    objective = {
        "prefill": "minimum_time_to_first_token",
        "decode": "maximum_decode_throughput",
        "mixed": "maximum_mixed_verified_throughput",
    }[phase]
    return PhasePlan(
        plan_id=f"{configuration.lower()}-{phase}",
        configuration=configuration,
        phase=phase,
        objective=objective,  # type: ignore[arg-type]
        placements=_placements(
            tiles,
            cpu_moe=use_cpu_moe,
            tensor_override=use_tensor_override,
        ),
        techniques=decisions,
        backend_arguments=arguments,
        predicted_metrics={},
        constraints={
            "interactive_p95_maximum_increase_fraction": 0.05,
            "other_workload_maximum_regression_fraction": 0.10,
        },
        explanation=[
            *[f"{decision.technique}: {decision.reason}" for decision in decisions],
            *(
                [
                    f"measured stock search selected {cpu_moe_layers} leading MoE layers for CPU expert execution; zero is a valid positive-utility outcome"
                ]
                if "asymmetric_cpu_gpu_partition" in enabled
                else []
            ),
            *(
                [
                    "actual llama.cpp tensor override: routers and normalisation tensors -> CUDA0; all other tensor residency remains governed by the selected stock layer plan"
                ]
                if use_tensor_override
                else []
            ),
        ],
    )
