"""Small real-compute MoE fixture used only for software validation and CI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FixtureRun:
    tensor_reference: Any
    tensor_reconstructed: Any
    reference: Any
    split: Any
    cached: Any
    prefetched: Any
    phase_planned: Any
    expert_ids: Any


def run_tiny_moe_fixture(*, seed: int = 8008) -> FixtureRun:
    """Exercise split/cache/prefetch equivalence with actual PyTorch tensor kernels.

    This is intentionally classified as EMULATED by callers.  It validates the
    execution machinery but cannot contribute to the official experiment verdict.
    """

    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens, hidden, intermediate, experts, top_k = 9, 32, 48, 6, 2
    values = torch.randn(tokens, hidden, generator=generator, dtype=torch.float32)
    router = torch.randn(hidden, experts, generator=generator, dtype=torch.float32)
    up = torch.randn(experts, hidden, intermediate, generator=generator, dtype=torch.float32)
    gate = torch.randn(experts, hidden, intermediate, generator=generator, dtype=torch.float32)
    down = torch.randn(experts, intermediate, hidden, generator=generator, dtype=torch.float32)
    scores, selected = torch.topk(values @ router, k=top_k, dim=-1)
    weights = torch.softmax(scores, dim=-1)

    def expert_projection(expert_id: int, rows: Any) -> Any:
        activated = torch.nn.functional.silu(rows @ gate[expert_id]) * (rows @ up[expert_id])
        return activated @ down[expert_id]

    reference = torch.zeros_like(values)
    for token in range(tokens):
        for rank in range(top_k):
            expert_id = int(selected[token, rank])
            reference[token] += (
                weights[token, rank] * expert_projection(expert_id, values[token : token + 1])[0]
            )

    tensor_boundary = intermediate // 3
    tensor_reconstructed = torch.cat(
        (up[:, :, :tensor_boundary], up[:, :, tensor_boundary:]),
        dim=2,
    )

    # Valid projection microshards preserve the same intermediate range across
    # up/gate/down and reduce partial outputs after the activation product.
    split = torch.zeros_like(values)
    boundary = intermediate // 2
    for token in range(tokens):
        for rank in range(top_k):
            expert_id = int(selected[token, rank])
            partials = []
            for start, end in ((0, boundary), (boundary, intermediate)):
                up_part = values[token : token + 1] @ up[expert_id, :, start:end]
                gate_part = values[token : token + 1] @ gate[expert_id, :, start:end]
                partials.append(
                    (torch.nn.functional.silu(gate_part) * up_part) @ down[expert_id, start:end, :]
                )
            split[token] += weights[token, rank] * torch.stack(partials).sum(dim=0)[0]

    cache: dict[int, tuple[Any, Any, Any]] = {}
    cached = torch.zeros_like(values)
    for token in range(tokens):
        for rank in range(top_k):
            expert_id = int(selected[token, rank])
            cache.setdefault(expert_id, (up[expert_id], gate[expert_id], down[expert_id]))
            cached_up, cached_gate, cached_down = cache[expert_id]
            activated = torch.nn.functional.silu(values[token : token + 1] @ cached_gate) * (
                values[token : token + 1] @ cached_up
            )
            cached[token] += weights[token, rank] * (activated @ cached_down)[0]

    # Prefetch is represented by populating the same cache one token early.  It
    # must not alter routing or arithmetic.
    prefetched = torch.zeros_like(values)
    cache.clear()
    for token in range(tokens):
        if token + 1 < tokens:
            for expert_id_tensor in selected[token + 1]:
                expert_id = int(expert_id_tensor)
                cache.setdefault(expert_id, (up[expert_id], gate[expert_id], down[expert_id]))
        for rank in range(top_k):
            expert_id = int(selected[token, rank])
            cached_up, cached_gate, cached_down = cache.setdefault(
                expert_id, (up[expert_id], gate[expert_id], down[expert_id])
            )
            activated = torch.nn.functional.silu(values[token : token + 1] @ cached_gate) * (
                values[token : token + 1] @ cached_up
            )
            prefetched[token] += weights[token, rank] * (activated @ cached_down)[0]

    def execute_phase(start: int, end: int) -> Any:
        phase_output = torch.zeros_like(values[start:end])
        for local_token, token in enumerate(range(start, end)):
            for rank in range(top_k):
                expert_id = int(selected[token, rank])
                phase_output[local_token] += (
                    weights[token, rank]
                    * expert_projection(expert_id, values[token : token + 1])[0]
                )
        return phase_output

    phase_boundary = tokens // 2
    phase_planned = torch.cat(
        (execute_phase(0, phase_boundary), execute_phase(phase_boundary, tokens)),
        dim=0,
    )
    return FixtureRun(
        up,
        tensor_reconstructed,
        reference,
        split,
        cached,
        prefetched,
        phase_planned,
        selected,
    )


def validate_tiny_moe_fixture(*, seed: int = 8008) -> dict[str, Any]:
    import torch

    result = run_tiny_moe_fixture(seed=seed)

    def comparison(reference: Any, candidate: Any) -> dict[str, float | bool]:
        difference = (reference - candidate).abs()
        return {
            # Split matrix products can accumulate in a different order across
            # supported PyTorch/Python wheels. Match the tolerance used by the
            # real CPU/CUDA tensor fixture below while retaining error metrics.
            "allclose": bool(torch.allclose(reference, candidate, atol=2e-4, rtol=2e-4)),
            "maximum_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    reference.flatten(), candidate.flatten(), dim=0
                )
            ),
        }

    rows = {
        "tensor_tile_reconstruction": comparison(
            result.tensor_reference, result.tensor_reconstructed
        ),
        "expert_microshard_equivalence": comparison(result.reference, result.split),
        "cache_hit_and_miss_equivalence": comparison(result.reference, result.cached),
        "prefetch_enabled_disabled_equivalence": comparison(result.reference, result.prefetched),
        "separate_prefill_decode_plan_equivalence": comparison(
            result.reference, result.phase_planned
        ),
    }
    from swarm_inference.experiments.experiment_008.runtime import (
        validate_tensor_runtime_fixture,
    )

    runtime = validate_tensor_runtime_fixture(seed=seed)
    runtime_passed = runtime.get("status") == "UNSUPPORTED" or bool(runtime.get("passed"))
    return {
        "classification": "EMULATED",
        "scope": "tiny deterministic PyTorch MoE software fixture; not official model evidence",
        "seed": seed,
        "selected_expert_ids": result.expert_ids.tolist(),
        "checks": rows,
        "tensor_runtime": runtime,
        "passed": all(bool(row["allclose"]) for row in rows.values()) and runtime_passed,
    }
