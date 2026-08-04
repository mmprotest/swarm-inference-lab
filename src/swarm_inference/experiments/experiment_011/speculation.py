"""Real-model exact prompt-lookup candidate verification across contiguous stages."""

from __future__ import annotations

import csv
import gc
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from swarm_inference.execution.olmoe_stage import ContiguousOlmoeStage
from swarm_inference.experiments.experiment_011.drafting import (
    PromptLookupDraftProvider,
    verify_greedy_candidates,
)
from swarm_inference.model.partition import StagePlan


def _traverse(
    stages: list[ContiguousOlmoeStage],
    *,
    session_id: str,
    token_ids: list[int],
    cache_position_start: int,
) -> tuple[list[int], int]:
    result = stages[0].execute(
        session_id=session_id,
        cache_position_start=cache_position_start,
        input_ids=torch.tensor([token_ids], dtype=torch.int64),
    )
    compute_ns = result.compute_ns
    for stage in stages[1:]:
        result = stage.execute(
            session_id=session_id,
            cache_position_start=cache_position_start,
            hidden_states=result.hidden_states,
        )
        compute_ns += result.compute_ns
    if result.all_sampled_token_ids is None:
        raise RuntimeError("final stage did not produce authoritative candidate logits")
    return [int(value) for value in result.all_sampled_token_ids[0].tolist()], compute_ns


def _run_depth(
    stages: list[ContiguousOlmoeStage],
    *,
    prompt_token_ids: list[int],
    expected_token_ids: list[int],
    depth: int,
    generated_token_count: int,
) -> dict[str, Any]:
    session_id = f"speculation-depth-{depth}"
    for stage in stages:
        stage.open_session(session_id)
    provider = PromptLookupDraftProvider(minimum_match=2)
    started = time.perf_counter_ns()
    authoritative, compute_ns = _traverse(
        stages,
        session_id=session_id,
        token_ids=prompt_token_ids,
        cache_position_start=0,
    )
    generated = [authoritative[-1]]
    traversals = 1
    proposed_total = 0
    accepted_total = 0
    rejected_total = 0
    verification_compute_ns = 0
    candidate_payload_bytes = 0
    accepted_per_round: list[int] = []
    rounds: list[dict[str, Any]] = []
    while len(generated) < generated_token_count:
        history = prompt_token_ids + generated
        remaining = generated_token_count - len(generated)
        proposal = provider.propose(history, depth=min(depth, remaining)) if depth > 0 else []
        old_cache_length = len(prompt_token_ids) + len(generated) - 1
        if proposal:
            proposal = proposal[:remaining]
            block = [generated[-1], *proposal[:-1]]
            targets, round_compute_ns = _traverse(
                stages,
                session_id=session_id,
                token_ids=block,
                cache_position_start=old_cache_length,
            )
            verification_compute_ns += round_compute_ns
            verification = verify_greedy_candidates(proposal, targets)
            accepted = list(verification.accepted_tokens)
            committed = list(accepted)
            if verification.first_rejection_index is not None:
                committed.append(targets[verification.first_rejection_index])
            committed = committed[:remaining]
            proposed_total += len(proposal)
            accepted_total += len(accepted)
            rejected_total += len(proposal) - len(accepted)
            accepted_per_round.append(len(accepted))
            candidate_payload_bytes += len(block) * 8
            desired_cache_length = old_cache_length + len(committed)
            for stage in stages:
                stage.crop_session(session_id, desired_cache_length)
            generated.extend(committed)
            rounds.append(
                {
                    "history_length": len(history),
                    "proposed_tokens": proposal,
                    "authoritative_tokens": targets[: len(proposal)],
                    "accepted_tokens": accepted,
                    "committed_tokens": committed,
                    "cache_length_after_commit": desired_cache_length,
                    "exact_prefix": accepted == proposal[: len(accepted)],
                }
            )
        else:
            targets, round_compute_ns = _traverse(
                stages,
                session_id=session_id,
                token_ids=[generated[-1]],
                cache_position_start=old_cache_length,
            )
            verification_compute_ns += round_compute_ns
            generated.append(targets[-1])
            accepted_per_round.append(0)
            rounds.append(
                {
                    "history_length": len(history),
                    "proposed_tokens": [],
                    "authoritative_tokens": [targets[-1]],
                    "accepted_tokens": [],
                    "committed_tokens": [targets[-1]],
                    "cache_length_after_commit": old_cache_length + 1,
                    "exact_prefix": True,
                }
            )
        traversals += 1
    elapsed = (time.perf_counter_ns() - started) / 1e9
    kv_before_close = [stage.kv_cache_bytes(session_id) for stage in stages]
    released = [stage.close_session(session_id) for stage in stages]
    exact = generated == expected_token_ids[:generated_token_count]
    return {
        "speculation_provider": "prompt_lookup" if depth > 0 else "none",
        "speculation_depth": depth,
        "proposed_tokens": proposed_total,
        "accepted_tokens": accepted_total,
        "acceptance_rate": accepted_total / proposed_total if proposed_total else 0.0,
        "accepted_tokens_per_verification_round": (
            statistics.mean(accepted_per_round) if accepted_per_round else 0.0
        ),
        "accepted_tokens_per_network_traversal": generated_token_count / traversals,
        "rejected_compute_tokens": rejected_total,
        "candidate_payload_bytes": candidate_payload_bytes,
        "verification_compute_ns": verification_compute_ns,
        "total_compute_ns": compute_ns + verification_compute_ns,
        "network_traversals": traversals,
        "generated_tokens": generated_token_count,
        "generated_token_ids": generated,
        "expected_token_ids": expected_token_ids[:generated_token_count],
        "exact_token_identity": exact,
        "elapsed_seconds": elapsed,
        "throughput_tps": generated_token_count / elapsed,
        "kv_cache_bytes_before_close": kv_before_close,
        "kv_cache_bytes_released": released,
        "session_cleanup_complete": kv_before_close == released,
        "rounds": rounds,
        "oracle_proposals_used": False,
        "evidence_category": "REAL_MODEL_MECHANISM_ONLY",
        "valid_for_network_performance_claims": False,
        "network_limitation": "candidate verification used the real staged model but not socket transport",
    }


def run_real_prompt_lookup_speculation(
    *,
    plan: StagePlan,
    prompt_token_ids: list[int],
    expected_token_ids: list[int],
    generated_token_count: int,
    depths: Sequence[int],
    output_directory: Path,
) -> list[dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stages = [
        ContiguousOlmoeStage(
            model_path=Path(plan.model_path),
            assignment=assignment,
            stage_count=plan.stage_count,
            device="cuda:0",
        )
        for assignment in plan.assignments
    ]
    results = [
        _run_depth(
            stages,
            prompt_token_ids=prompt_token_ids,
            expected_token_ids=expected_token_ids,
            depth=0,
            generated_token_count=generated_token_count,
        )
    ]
    for depth in depths:
        results.append(
            _run_depth(
                stages,
                prompt_token_ids=prompt_token_ids,
                expected_token_ids=expected_token_ids,
                depth=int(depth),
                generated_token_count=generated_token_count,
            )
        )
    baseline_tps = float(results[0]["throughput_tps"])
    for result in results:
        result["throughput_multiple_vs_non_speculative"] = (
            float(result["throughput_tps"]) / baseline_tps if baseline_tps else 0.0
        )
        result["planner_enabled"] = (
            int(result["speculation_depth"]) > 0
            and bool(result["exact_token_identity"])
            and float(result["acceptance_rate"]) > 0
            and float(result["throughput_tps"]) > baseline_tps * 1.02
            and bool(result["valid_for_network_performance_claims"])
        )
        result["planner_reason"] = (
            "enabled from positive measured socket-path expected value"
            if result["planner_enabled"]
            else "disabled: no positive valid socket-path expected-value evidence"
        )
    with (output_directory / "speculation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        scalar_keys = sorted(
            key for key, value in results[0].items() if not isinstance(value, (list, dict))
        )
        writer = csv.DictWriter(handle, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    (output_directory / "speculation_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del stages
    gc.collect()
    torch.cuda.empty_cache()
    return results
