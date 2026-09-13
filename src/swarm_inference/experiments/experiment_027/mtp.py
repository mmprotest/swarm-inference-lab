"""Qwen native MTP candidates, accepted only by distributed target argmax."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .protocol import StageClient, StagePipeline
from .runner import Prompt, SpeculativeLane, token_hash


def generate_mtp_lane(pipeline: StagePipeline, draft: StageClient, prompt: Prompt,
                      prompt_tokens: Sequence[int], *, output_tokens: int,
                      block_size: int, top_k: int = 16) -> SpeculativeLane:
    pipeline.reset()
    draft.reset()
    prefill = pipeline.traverse(prompt_tokens, position=0, top_k=top_k, return_nextn=True)
    h = prefill.output.nextn
    assert h is not None
    draft.infer_mtp(list(prompt_tokens), np.concatenate((np.zeros_like(h[:1]), h[:-1])),
                    position=0)
    pending = int(prefill.output.greedy_ids[-1])
    pending_h = h[-1:]
    position = len(prompt_tokens)
    generated: list[int] = []
    traversals = []
    proposed = accepted = 0
    commit_counts, block_elapsed_ns = [], []
    rewind = -1
    begin = time.perf_counter_ns()
    while len(generated) < output_tokens:
        block_begin = time.perf_counter_ns()
        candidates: list[int] = []
        token, state = pending, pending_h
        count = min(block_size - 1, output_tokens - len(generated) - 1)
        for step in range(count):
            prediction = draft.infer_mtp([token], state, position=position + step,
                                         rewind_position=position if step == 0 else -1)
            token = int(prediction.greedy_ids[-1])
            state = prediction.nextn
            assert state is not None
            candidates.append(token)
        traversal = pipeline.traverse([pending, *candidates], position=position,
                                      rewind_position=rewind, top_k=top_k, return_nextn=True)
        traversals.append(traversal)
        proposed += len(candidates)
        accepted_now = 0
        for index, candidate in enumerate(candidates):
            if candidate != int(traversal.output.greedy_ids[index]):
                break
            accepted_now += 1
        accepted += accepted_now
        committed = [pending, *candidates[:accepted_now]]
        h = traversal.output.nextn
        assert h is not None
        # Replace draft KV from this block with accepted target-conditioned rows.
        draft.infer_mtp(committed, np.concatenate((pending_h, h[:accepted_now])),
                        position=position, rewind_position=position)
        pending_h = h[accepted_now:accepted_now + 1]
        pending = int(traversal.output.greedy_ids[accepted_now])
        position += len(committed)
        generated.extend(committed)
        rewind = position if accepted_now < len(candidates) else -1
        commit_counts.append(len(committed))
        block_elapsed_ns.append(time.perf_counter_ns() - block_begin)
    elapsed = time.perf_counter_ns() - begin
    return SpeculativeLane(prompt.prompt_id, block_size, tuple(generated), token_hash(generated),
                           tuple(traversals), elapsed, proposed, accepted,
                           tuple(commit_counts), tuple(block_elapsed_ns))
