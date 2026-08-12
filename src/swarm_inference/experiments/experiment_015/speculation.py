"""Lossless speculative acceptance and transactional Kimi request state."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_015.contracts import Experiment015Error


def _fingerprint(value: Any) -> str:
    def update(digest: Any, item: Any) -> None:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                update(digest, item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                update(digest, child)
        else:
            digest.update(repr(item).encode())

    result = hashlib.sha256()
    update(result, value)
    return "sha256:" + result.hexdigest()


@dataclass(slots=True)
class RequestState:
    """All state that must commit atomically after target verification."""

    position: int
    token_ids: list[int] = field(default_factory=list)
    kda: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    mla: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    attnres: dict[str, np.ndarray] = field(default_factory=dict)
    eos: bool = False
    cancelled: bool = False
    generation: int = 0

    def clone(self) -> RequestState:
        return copy.deepcopy(self)

    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "position": self.position,
                "token_ids": self.token_ids,
                "kda": self.kda,
                "mla": self.mla,
                "attnres": self.attnres,
                "eos": self.eos,
                "cancelled": self.cancelled,
                "generation": self.generation,
            }
        )


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    accepted_draft_tokens: tuple[int, ...]
    replacement_token: int | None
    first_rejection_index: int | None
    target_rows_consumed: int
    full_block_accepted: bool
    eos_reached: bool

    @property
    def output_tokens(self) -> tuple[int, ...]:
        suffix = () if self.replacement_token is None else (self.replacement_token,)
        return self.accepted_draft_tokens + suffix


def greedy_acceptance(
    draft_tokens: list[int],
    target_argmax_tokens: list[int],
    *,
    bonus_token: int | None = None,
    eos_token_id: int | None = None,
) -> AcceptanceResult:
    """Accept the longest exact draft prefix, then emit target replacement/bonus."""
    if len(target_argmax_tokens) < len(draft_tokens):
        raise ValueError("target verification must cover every draft position")
    accepted: list[int] = []
    rejection: int | None = None
    replacement: int | None = None
    eos = False
    for index, draft in enumerate(draft_tokens):
        target = int(target_argmax_tokens[index])
        if int(draft) != target:
            rejection = index
            replacement = target
            eos = eos_token_id is not None and replacement == eos_token_id
            break
        accepted.append(int(draft))
        if eos_token_id is not None and int(draft) == eos_token_id:
            eos = True
            break
    full = rejection is None and len(accepted) == len(draft_tokens) and not eos
    if full and bonus_token is not None:
        replacement = int(bonus_token)
        eos = eos_token_id is not None and replacement == eos_token_id
    return AcceptanceResult(
        accepted_draft_tokens=tuple(accepted),
        replacement_token=replacement,
        first_rejection_index=rejection,
        target_rows_consumed=len(draft_tokens) + int(bonus_token is not None),
        full_block_accepted=full,
        eos_reached=eos,
    )


def _categorical(probabilities: np.ndarray, uniform: float) -> int:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("categorical probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise Experiment015Error("categorical distribution has zero mass")
    threshold = min(max(uniform, 0.0), math.nextafter(1.0, 0.0)) * total
    return int(np.searchsorted(np.cumsum(values), threshold, side="right"))


def stochastic_acceptance(
    draft_tokens: list[int],
    draft_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    uniforms: list[float],
    *,
    eos_token_id: int | None = None,
) -> AcceptanceResult:
    """Leviathan rejection sampling preserving the target distribution exactly."""
    q = np.asarray(draft_probabilities, dtype=np.float64)
    p = np.asarray(target_probabilities, dtype=np.float64)
    if (
        q.ndim != 2
        or p.ndim != 2
        or q.shape[0] != len(draft_tokens)
        or p.shape != (len(draft_tokens) + 1, q.shape[1])
    ):
        raise ValueError(
            "target probabilities must contain one verification row per draft "
            "plus the bonus row"
        )
    if len(uniforms) < len(draft_tokens) + 1:
        raise ValueError("one acceptance uniform per draft plus one residual uniform required")
    accepted: list[int] = []
    for index, token_value in enumerate(draft_tokens):
        token = int(token_value)
        if not 0 <= token < q.shape[1]:
            raise ValueError("draft token is outside probability vocabulary")
        q_row = q[index] / q[index].sum()
        p_row = p[index] / p[index].sum()
        q_mass = float(q_row[token])
        ratio = 1.0 if q_mass == 0.0 else min(1.0, float(p_row[token]) / q_mass)
        if uniforms[index] <= ratio:
            accepted.append(token)
            if eos_token_id is not None and token == eos_token_id:
                return AcceptanceResult(
                    tuple(accepted), None, None, len(draft_tokens) + 1, False, True
                )
            continue
        residual = np.maximum(p_row - q_row, 0.0)
        replacement = _categorical(residual, uniforms[len(draft_tokens)])
        return AcceptanceResult(
            tuple(accepted),
            replacement,
            index,
            len(draft_tokens) + 1,
            False,
            eos_token_id is not None and replacement == eos_token_id,
        )
    bonus_distribution = p[len(draft_tokens)] / p[len(draft_tokens)].sum()
    bonus = _categorical(bonus_distribution, uniforms[len(draft_tokens)])
    return AcceptanceResult(
        tuple(accepted),
        bonus,
        None,
        len(draft_tokens) + 1,
        True,
        eos_token_id is not None and bonus == eos_token_id,
    )


class SpeculativeSession:
    """Versioned transaction around KDA, MLA and AttnRes speculative state."""

    def __init__(self, state: RequestState) -> None:
        self._committed = state.clone()
        self._working: RequestState | None = None
        self._verified_prefix_states: list[RequestState] = []
        self._base_fingerprint: str | None = None

    @property
    def committed(self) -> RequestState:
        return self._committed.clone()

    def begin(self) -> RequestState:
        if self._working is not None:
            raise Experiment015Error("a speculative transaction is already active")
        if self._committed.cancelled or self._committed.eos:
            raise Experiment015Error("a terminal request cannot begin speculation")
        self._base_fingerprint = self._committed.fingerprint()
        self._working = self._committed.clone()
        self._working.generation += 1
        self._verified_prefix_states = []
        return self._working

    def stage_verified_prefix(self) -> None:
        """Capture state after one target-verified output position."""
        if self._working is None:
            raise Experiment015Error("no speculative transaction is active")
        self._verified_prefix_states.append(self._working.clone())

    def commit(self, result: AcceptanceResult) -> RequestState:
        if self._working is None or self._base_fingerprint is None:
            raise Experiment015Error("no speculative transaction is active")
        if self._committed.fingerprint() != self._base_fingerprint:
            raise Experiment015Error("committed request state changed during verification")
        output = list(result.output_tokens)
        if len(self._verified_prefix_states) < len(output):
            raise Experiment015Error(
                "target state snapshots do not cover the committed output prefix"
            )
        selected = (
            self._committed.clone()
            if not output
            else self._verified_prefix_states[len(output) - 1].clone()
        )
        selected.token_ids = [*self._committed.token_ids, *output]
        selected.position = self._committed.position + len(output)
        selected.eos = result.eos_reached
        self._committed = selected
        self._working = None
        self._verified_prefix_states = []
        self._base_fingerprint = None
        return self.committed

    def rollback(self) -> RequestState:
        if self._working is None:
            raise Experiment015Error("no speculative transaction is active")
        self._working = None
        self._verified_prefix_states = []
        self._base_fingerprint = None
        return self.committed

    def cancel(self) -> RequestState:
        self._working = None
        self._verified_prefix_states = []
        self._base_fingerprint = None
        self._committed.cancelled = True
        self._committed.generation += 1
        return self.committed
