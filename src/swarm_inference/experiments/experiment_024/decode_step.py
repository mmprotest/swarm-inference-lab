"""Actual autoregressive decode-unit semantics for Experiment 024."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodeStepTask:
    batch_size: int
    consumes_current_token_ids: bool = True
    independent_recurrent_state_per_row: bool = True
    transformer_layer_count: int = 93
    executes_embedding: bool = True
    executes_final_norm: bool = True
    executes_lm_head: bool = True
    executes_greedy_argmax: bool = True
    commits_recurrent_state: bool = True
    speculative_execution: bool = False
    target_verification: bool = False

    @property
    def generated_output_tokens(self) -> int:
        return self.batch_size


class DecodeStepTaskBuilder:
    """Build one real output-token step per active sequence row."""

    def build(self, batch_size: int) -> DecodeStepTask:
        if batch_size not in (1, 2, 4):
            raise ValueError("E024 decode microbatch size must be 1, 2, or 4")
        return DecodeStepTask(batch_size=batch_size)


__all__ = ["DecodeStepTask", "DecodeStepTaskBuilder"]
