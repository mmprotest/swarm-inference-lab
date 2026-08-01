"""Efficient real target/draft trace capture and lossless proposal replay."""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from swarm_inference.worker.abi import (
    BackendAdapter,
    GenerationParameters,
    TokenPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
)


@dataclass(frozen=True, slots=True)
class HeldOutPrompt:
    prompt_id: str
    category: str
    text: str
    token_ids: list[int]


async def _trace(
    adapter: BackendAdapter,
    *,
    prompt: HeldOutPrompt,
    role: WorkerJobType,
    model_id: str,
    revision: str,
    tokenizer_hash: str,
    maximum_output_tokens: int,
) -> tuple[list[int], dict[str, Any], float]:
    job = WorkerJob(
        job_id=uuid4().hex,
        request_id=f"exp007-{prompt.prompt_id}-{role.value}-{uuid4().hex[:8]}",
        role=role,
        model_id=model_id,
        model_revision=revision,
        input_payload=TokenPayload(token_ids=prompt.token_ids, tokenizer_hash=tokenizer_hash),
        generation_parameters=GenerationParameters(
            max_new_tokens=maximum_output_tokens,
            temperature=0.0,
            ignore_eos=False,
            seed=7007,
        ),
        deadline_ms=900_000,
        priority=100 if role == WorkerJobType.TARGET_DECODE else 25,
    )
    started = time.perf_counter()
    result = await adapter.execute(job)
    elapsed = time.perf_counter() - started
    if result.status != WorkerJobStatus.ACCEPTED:
        raise RuntimeError(f"{adapter.backend_id}: {result.status.value}: {result.detail}")
    if not isinstance(result.output_payload, TokenPayload):
        raise RuntimeError(f"{adapter.backend_id} returned no token trace")
    return result.output_payload.token_ids, result.metrics, elapsed


def replay_lossless_trace(
    *,
    target: list[int],
    draft: list[int],
    draft_length: int,
    prompt_id: str,
    category: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Commit only target-equal candidates; output is target by construction and check."""

    if draft_length not in {1, 2, 4, 8}:
        raise ValueError("draft length must be 1, 2, 4, or 8")
    committed: list[int] = []
    blocks: list[dict[str, Any]] = []
    accepted_total = 0
    proposed_total = 0
    full_rejections = 0
    accepted_lengths: list[int] = []
    first_mismatch: dict[str, Any] | None = None
    target_verification_tokens = 0
    cursor = 0
    while cursor < len(target):
        remaining_before = len(target) - cursor
        candidates = draft[cursor : cursor + draft_length]
        if not candidates:
            candidates = [-1]
        proposed_total += min(len(candidates), remaining_before)
        target_verification_tokens += min(len(candidates) + 1, remaining_before)
        accepted = 0
        mismatch_index: int | None = None
        for offset, candidate in enumerate(candidates):
            position = cursor + offset
            if position >= len(target) or candidate != target[position]:
                mismatch_index = offset
                break
            committed.append(candidate)
            accepted += 1
            accepted_total += 1
        cursor += accepted
        if cursor < len(target):
            if mismatch_index is None and accepted == len(candidates):
                committed.append(target[cursor])
                cursor += 1
            else:
                if accepted == 0:
                    full_rejections += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "output_position": cursor,
                        "candidate_token": (
                            candidates[mismatch_index] if mismatch_index is not None else None
                        ),
                        "target_token": target[cursor],
                    }
                committed.append(target[cursor])
                cursor += 1
        accepted_lengths.append(accepted)
        blocks.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "block": len(blocks),
                "draft_length": draft_length,
                "proposal_token_ids": candidates,
                "accepted_length": accepted,
                "mismatch_index": mismatch_index,
                "committed_tokens_after": cursor,
                "verification_source": "measured_sglang_greedy_trace",
            }
        )
    exact = committed == target
    if not exact:
        raise RuntimeError("lossless trace coordinator did not preserve target token IDs")
    return (
        {
            "prompt_id": prompt_id,
            "category": category,
            "draft_length": draft_length,
            "output_tokens": len(target),
            "proposed_tokens": proposed_total,
            "accepted_tokens": accepted_total,
            "acceptance_rate": accepted_total / max(proposed_total, 1),
            "mean_accepted_length": statistics.mean(accepted_lengths),
            "full_rejection_rate": full_rejections / max(len(blocks), 1),
            "verification_count": len(blocks),
            "accepted_tokens_per_verification": accepted_total / max(len(blocks), 1),
            "committed_tokens_per_verification": len(target) / max(len(blocks), 1),
            "target_work_per_committed_token": target_verification_tokens / max(len(target), 1),
            "exact_output_identity": exact,
            "first_mismatch": first_mismatch,
            "target_token_ids": target,
            "speculative_token_ids": committed,
        },
        blocks,
    )


async def capture_target_traces(
    prompts: list[HeldOutPrompt],
    *,
    target: BackendAdapter,
    model_id: str,
    revision: str,
    tokenizer_hash: str,
    maximum_output_tokens: int,
) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        tokens, metrics, elapsed = await _trace(
            target,
            prompt=prompt,
            role=WorkerJobType.TARGET_DECODE,
            model_id=model_id,
            revision=revision,
            tokenizer_hash=tokenizer_hash,
            maximum_output_tokens=maximum_output_tokens,
        )
        traces[prompt.prompt_id] = {
            "tokens": tokens,
            "metrics": metrics,
            "elapsed_seconds": elapsed,
        }
    return traces


async def capture_and_replay_draft_format(
    prompts: list[HeldOutPrompt],
    *,
    target_traces: dict[str, dict[str, Any]],
    draft: BackendAdapter,
    model_id: str,
    revision: str,
    tokenizer_hash: str,
    maximum_output_tokens: int,
    weight_format: str,
    draft_lengths: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    draft_measurements: list[dict[str, Any]] = []
    for prompt in prompts:
        tokens, metrics, elapsed = await _trace(
            draft,
            prompt=prompt,
            role=WorkerJobType.SPECULATIVE_DRAFT,
            model_id=model_id,
            revision=revision,
            tokenizer_hash=tokenizer_hash,
            maximum_output_tokens=maximum_output_tokens,
        )
        draft_measurements.append(
            {
                "prompt_id": prompt.prompt_id,
                "tokens": tokens,
                "elapsed_seconds": elapsed,
                "metrics": metrics,
            }
        )
        target_record = target_traces[prompt.prompt_id]
        target_tokens = list(target_record["tokens"])
        for length in draft_lengths:
            replay, blocks = replay_lossless_trace(
                target=target_tokens,
                draft=tokens,
                draft_length=length,
                prompt_id=prompt.prompt_id,
                category=prompt.category,
            )
            target_seconds = float(target_record["elapsed_seconds"])
            verification_scale = float(replay["target_work_per_committed_token"])
            projected_seconds = elapsed + target_seconds * verification_scale
            target_tps = len(target_tokens) / max(target_seconds, 1e-12)
            coordinated_tps = len(target_tokens) / max(projected_seconds, 1e-12)
            rows.append(
                {
                    **replay,
                    "classification": "emulated_network",
                    "proposal_classification": "measured_mixed_backend",
                    "timing_method": "measured_compute_trace_event_replay",
                    "weight_format": weight_format,
                    "draft_seconds": elapsed,
                    "target_only_seconds": target_seconds,
                    "target_verification_seconds": target_seconds * verification_scale,
                    "draft_tokens_per_second": len(tokens) / max(elapsed, 1e-12),
                    "target_only_tokens_per_second": target_tps,
                    "speculative_tokens_per_second": coordinated_tps,
                    "speedup_fraction": coordinated_tps / max(target_tps, 1e-12) - 1,
                    "positive_contribution_pass": coordinated_tps >= target_tps * 1.10,
                }
            )
            evidence.extend({**block, "weight_format": weight_format} for block in blocks)
    profile: dict[str, Any] = {
        "weight_format": weight_format,
        "prompt_count": len(prompts),
        "draft_tokens": sum(len(item["tokens"]) for item in draft_measurements),
        "draft_seconds": sum(float(item["elapsed_seconds"]) for item in draft_measurements),
    }
    profile["draft_tokens_per_second"] = float(profile["draft_tokens"]) / max(
        float(profile["draft_seconds"]), 1e-12
    )
    return rows, evidence, profile


def aggregate_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["weight_format"]), int(row["draft_length"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (weight_format, draft_length), members in sorted(groups.items()):
        proposed = sum(int(row["proposed_tokens"]) for row in members)
        accepted = sum(int(row["accepted_tokens"]) for row in members)
        tokens = sum(int(row["output_tokens"]) for row in members)
        target_seconds = sum(float(row["target_only_seconds"]) for row in members)
        draft_seconds = sum(float(row["draft_seconds"]) for row in members)
        target_verification_seconds = sum(
            float(row["target_verification_seconds"]) for row in members
        )
        speculative_seconds = sum(
            float(row["draft_seconds"]) + float(row["target_verification_seconds"])
            for row in members
        )
        target_tps = tokens / max(target_seconds, 1e-12)
        speculative_tps = tokens / max(speculative_seconds, 1e-12)
        verification_count = sum(int(row["verification_count"]) for row in members)
        output.append(
            {
                "classification": "emulated_network",
                "proposal_classification": "measured_mixed_backend",
                "timing_method": "measured_compute_trace_event_replay",
                "weight_format": weight_format,
                "draft_length": draft_length,
                "prompt_count": len(members),
                "all_exact": all(bool(row["exact_output_identity"]) for row in members),
                "acceptance_rate": accepted / max(proposed, 1),
                "mean_accepted_length": statistics.mean(
                    float(row["mean_accepted_length"]) for row in members
                ),
                "full_rejection_rate": statistics.mean(
                    float(row["full_rejection_rate"]) for row in members
                ),
                "accepted_tokens_per_verification": accepted / max(verification_count, 1),
                "committed_tokens_per_verification": tokens / max(verification_count, 1),
                "target_work_per_committed_token": sum(
                    float(row["target_work_per_committed_token"]) * int(row["output_tokens"])
                    for row in members
                )
                / max(tokens, 1),
                "draft_seconds": draft_seconds,
                "target_verification_seconds": target_verification_seconds,
                "speculative_seconds": speculative_seconds,
                "draft_tokens_per_second": sum(int(row["proposed_tokens"]) for row in members)
                / max(draft_seconds, 1e-12),
                "target_only_tokens_per_second": target_tps,
                "speculative_tokens_per_second": speculative_tps,
                "speedup_fraction": speculative_tps / max(target_tps, 1e-12) - 1,
                "positive_contribution_pass": speculative_tps >= target_tps * 1.10,
            }
        )
    return output


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)
