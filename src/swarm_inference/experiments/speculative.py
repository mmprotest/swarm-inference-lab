"""Lossless heterogeneous speculative coordinator for greedy decoding."""

from __future__ import annotations

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
class SpeculativePrompt:
    prompt_id: str
    category: str
    token_ids: list[int]


class SpeculativeExecutionError(RuntimeError):
    pass


async def _generate(
    adapter: BackendAdapter,
    *,
    role: WorkerJobType,
    prompt: SpeculativePrompt,
    prefix: list[int],
    output_tokens: int,
    model_id: str,
    model_revision: str,
    tokenizer_hash: str,
    deadline_ms: int,
    request_suffix: str,
) -> tuple[list[int], dict[str, Any]]:
    job = WorkerJob(
        job_id=uuid4().hex,
        request_id=f"{prompt.prompt_id}-{request_suffix}",
        role=role,
        model_id=model_id,
        model_revision=model_revision,
        input_payload=TokenPayload(
            token_ids=[*prompt.token_ids, *prefix],
            tokenizer_hash=tokenizer_hash,
        ),
        generation_parameters=GenerationParameters(
            max_new_tokens=output_tokens,
            temperature=0.0,
            ignore_eos=False,
        ),
        deadline_ms=deadline_ms,
        priority=100 if role != WorkerJobType.BACKGROUND_GENERATE else 1,
    )
    result = await adapter.execute(job)
    if result.status != WorkerJobStatus.ACCEPTED:
        raise SpeculativeExecutionError(
            f"{adapter.backend_id}: {result.status.value}: {result.detail}"
        )
    if not isinstance(result.output_payload, TokenPayload):
        raise SpeculativeExecutionError(f"{adapter.backend_id} returned a non-token result")
    return result.output_payload.token_ids, result.metrics


async def run_lossless_speculative_prompt(
    *,
    prompt: SpeculativePrompt,
    target: BackendAdapter,
    draft: BackendAdapter,
    target_model_id: str,
    target_revision: str,
    draft_model_id: str,
    draft_revision: str,
    tokenizer_hash: str,
    draft_length: int,
    maximum_output_tokens: int,
    deadline_ms: int = 300_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run target-only and coordinated greedy decode, then require token identity."""

    if draft_length not in {1, 2, 4, 8}:
        raise ValueError("Experiment 007 draft length must be 1, 2, 4, or 8")
    baseline_started = time.perf_counter()
    target_only, baseline_metrics = await _generate(
        target,
        role=WorkerJobType.TARGET_DECODE,
        prompt=prompt,
        prefix=[],
        output_tokens=maximum_output_tokens,
        model_id=target_model_id,
        model_revision=target_revision,
        tokenizer_hash=tokenizer_hash,
        deadline_ms=deadline_ms,
        request_suffix="target-only",
    )
    baseline_seconds = time.perf_counter() - baseline_started
    committed: list[int] = []
    evidence: list[dict[str, Any]] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_verification_tokens = 0
    full_rejections = 0
    accepted_lengths: list[int] = []
    draft_seconds = 0.0
    verification_seconds = 0.0
    draft_removed = False
    first_mismatch: dict[str, Any] | None = None
    speculative_started = time.perf_counter()
    block = 0
    while len(committed) < len(target_only):
        remaining = len(target_only) - len(committed)
        proposal_count = min(draft_length, remaining)
        try:
            draft_started = time.perf_counter()
            proposal, draft_metrics = await _generate(
                draft,
                role=WorkerJobType.SPECULATIVE_DRAFT,
                prompt=prompt,
                prefix=committed,
                output_tokens=proposal_count,
                model_id=draft_model_id,
                model_revision=draft_revision,
                tokenizer_hash=tokenizer_hash,
                deadline_ms=deadline_ms,
                request_suffix=f"draft-{draft_length}-{block}",
            )
            draft_seconds += time.perf_counter() - draft_started
        except SpeculativeExecutionError as exc:
            draft_removed = True
            evidence.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "block": block,
                    "event": "draft_node_removed",
                    "detail": str(exc),
                    "committed_before": len(committed),
                }
            )
            committed.extend(target_only[len(committed) :])
            break
        proposal = proposal[:proposal_count]
        if not proposal:
            raise SpeculativeExecutionError("draft backend returned no candidate tokens")
        proposed_tokens += len(proposal)
        verify_started = time.perf_counter()
        verification, target_metrics = await _generate(
            target,
            role=WorkerJobType.TARGET_DECODE,
            prompt=prompt,
            prefix=committed,
            output_tokens=min(len(proposal) + 1, remaining),
            model_id=target_model_id,
            model_revision=target_revision,
            tokenizer_hash=tokenizer_hash,
            deadline_ms=deadline_ms,
            request_suffix=f"verify-{draft_length}-{block}",
        )
        verification_seconds += time.perf_counter() - verify_started
        target_verification_tokens += len(verification)
        accepted = 0
        mismatch_index: int | None = None
        for index, candidate in enumerate(proposal):
            if index >= len(verification) or candidate != verification[index]:
                mismatch_index = index
                break
            committed.append(candidate)
            accepted += 1
            accepted_tokens += 1
            if len(committed) >= len(target_only):
                break
        if len(committed) < len(target_only):
            if mismatch_index is not None:
                target_token = verification[mismatch_index]
                committed.append(target_token)
                if accepted == 0:
                    full_rejections += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "prompt_id": prompt.prompt_id,
                        "block": block,
                        "candidate_index": mismatch_index,
                        "candidate_token": proposal[mismatch_index],
                        "target_token": target_token,
                        "absolute_token_position": len(prompt.token_ids) + len(committed) - 1,
                    }
            elif len(verification) > len(proposal):
                committed.append(verification[len(proposal)])
        accepted_lengths.append(accepted)
        evidence.append(
            {
                "prompt_id": prompt.prompt_id,
                "category": prompt.category,
                "block": block,
                "draft_length": draft_length,
                "prefix_output_tokens": len(committed) - accepted - (1 if len(committed) else 0),
                "proposal_token_ids": proposal,
                "target_verification_token_ids": verification,
                "accepted_length": accepted,
                "mismatch_index": mismatch_index,
                "draft_metrics": draft_metrics,
                "target_metrics": target_metrics,
            }
        )
        if committed != target_only[: len(committed)]:
            mismatch = next(
                index
                for index, (actual, expected) in enumerate(
                    zip(committed, target_only, strict=False)
                )
                if actual != expected
            )
            raise SpeculativeExecutionError(
                f"lossless invariant failed for {prompt.prompt_id} at output token {mismatch}"
            )
        block += 1
    speculative_seconds = time.perf_counter() - speculative_started
    identity = committed == target_only
    if not identity:
        raise SpeculativeExecutionError("speculative token IDs differ from target-only token IDs")
    output_count = len(committed)
    baseline_tps = output_count / max(baseline_seconds, 1e-12)
    speculative_tps = output_count / max(speculative_seconds, 1e-12)
    return (
        {
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "draft_length": draft_length,
            "target_only_token_ids": target_only,
            "speculative_token_ids": committed,
            "exact_output_identity": identity,
            "first_mismatch": first_mismatch,
            "proposed_tokens": proposed_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": accepted_tokens / max(proposed_tokens, 1),
            "mean_accepted_length": sum(accepted_lengths) / max(len(accepted_lengths), 1),
            "full_rejection_rate": full_rejections / max(len(accepted_lengths), 1),
            "target_work_per_committed_token": target_verification_tokens / max(output_count, 1),
            "draft_seconds": draft_seconds,
            "target_verification_seconds": verification_seconds,
            "target_only_seconds": baseline_seconds,
            "speculative_seconds": speculative_seconds,
            "target_only_tokens_per_second": baseline_tps,
            "speculative_tokens_per_second": speculative_tps,
            "speedup_fraction": speculative_tps / baseline_tps - 1,
            "draft_tokens_per_second": proposed_tokens / max(draft_seconds, 1e-12),
            "draft_node_removed": draft_removed,
            "target_only_metrics": baseline_metrics,
        },
        evidence,
    )


def aggregate_speculative_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty speculative result set")
    proposed = sum(int(row["proposed_tokens"]) for row in rows)
    accepted = sum(int(row["accepted_tokens"]) for row in rows)
    target_seconds = sum(float(row["target_only_seconds"]) for row in rows)
    speculative_seconds = sum(float(row["speculative_seconds"]) for row in rows)
    output_tokens = sum(len(row["speculative_token_ids"]) for row in rows)
    target_tps = output_tokens / max(target_seconds, 1e-12)
    speculative_tps = output_tokens / max(speculative_seconds, 1e-12)
    return {
        "prompt_count": len(rows),
        "all_exact": all(bool(row["exact_output_identity"]) for row in rows),
        "acceptance_rate": accepted / max(proposed, 1),
        "mean_accepted_length": sum(float(row["mean_accepted_length"]) for row in rows) / len(rows),
        "full_rejection_rate": sum(float(row["full_rejection_rate"]) for row in rows) / len(rows),
        "target_only_tokens_per_second": target_tps,
        "speculative_tokens_per_second": speculative_tps,
        "speedup_fraction": speculative_tps / target_tps - 1,
        "draft_tokens_per_second": proposed
        / max(sum(float(row["draft_seconds"]) for row in rows), 1e-12),
        "target_work_per_committed_token": sum(
            float(row["target_work_per_committed_token"]) * len(row["speculative_token_ids"])
            for row in rows
        )
        / max(output_tokens, 1),
    }
