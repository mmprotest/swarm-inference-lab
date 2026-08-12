"""Exact compact-factor reference machinery for Kimi K3 KDA verification.

This module is intentionally a NumPy reference, not a production kernel.  It
captures the affine structure of the KDA recurrent update so experiments can
test prefix composition, accepted-state reconstruction, and replay without
conflating the mathematics with a CUDA implementation.

For one head, with state ``H`` stored as ``[key, value]``, Kimi K3 updates

``H' = (D - beta * k @ (D k).T) H + beta * k @ v.T``.

The transition is diagonal plus rank one and the additive term is rank one.
Products of these transitions remain diagonal plus low rank, with rank growing
by at most one per candidate token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _matrix(value: FloatArray, *, name: str) -> FloatArray:
    result = np.asarray(value)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    return result


def _vector(value: FloatArray, *, name: str) -> FloatArray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    return result


@dataclass(frozen=True)
class KDATokenFactor:
    """KDA-specific compact description of one exact affine transition."""

    alpha: FloatArray
    key: FloatArray
    value: FloatArray
    beta: float

    def __post_init__(self) -> None:
        alpha = _vector(self.alpha, name="alpha")
        key = _vector(self.key, name="key")
        value = _vector(self.value, name="value")
        if alpha.shape != key.shape:
            raise ValueError("alpha and key must have identical dimensions")
        if not np.isfinite(float(self.beta)):
            raise ValueError("beta must be finite")
        if not all(np.isfinite(item).all() for item in (alpha, key, value)):
            raise ValueError("KDA token factors must be finite")

    @property
    def key_dimension(self) -> int:
        return int(self.key.size)

    @property
    def value_dimension(self) -> int:
        return int(self.value.size)

    @property
    def compact_elements(self) -> int:
        # alpha, key, value, beta. Query vectors belong to output evaluation,
        # not to recurrent-state progression, and are therefore not counted.
        return int(self.alpha.size + self.key.size + self.value.size + 1)

    def expand(self) -> AffineKDAFactor:
        """Expand the KDA-specific tuple to a generic diagonal/low-rank factor."""

        dtype = np.result_type(self.alpha, self.key, self.value, np.float32)
        alpha = np.asarray(self.alpha, dtype=dtype)
        key = np.asarray(self.key, dtype=dtype)
        value = np.asarray(self.value, dtype=dtype)
        beta = np.asarray(self.beta, dtype=dtype)
        return AffineKDAFactor(
            diagonal=alpha,
            transition_left=(-beta * key)[:, None],
            transition_right=(alpha * key)[:, None],
            additive_left=(beta * key)[:, None],
            additive_right=value[:, None],
        )


@dataclass(frozen=True)
class AffineKDAFactor:
    """An exact affine map ``H -> (D + U V.T) H + L R.T``."""

    diagonal: FloatArray
    transition_left: FloatArray
    transition_right: FloatArray
    additive_left: FloatArray
    additive_right: FloatArray

    def __post_init__(self) -> None:
        diagonal = _vector(self.diagonal, name="diagonal")
        transition_left = _matrix(self.transition_left, name="transition_left")
        transition_right = _matrix(self.transition_right, name="transition_right")
        additive_left = _matrix(self.additive_left, name="additive_left")
        additive_right = _matrix(self.additive_right, name="additive_right")
        keys = int(diagonal.size)
        if transition_left.shape[0] != keys or transition_right.shape[0] != keys:
            raise ValueError("transition factors must match the key dimension")
        if transition_left.shape[1] != transition_right.shape[1]:
            raise ValueError("transition factor ranks must match")
        if additive_left.shape[0] != keys:
            raise ValueError("additive left factor must match the key dimension")
        if additive_left.shape[1] != additive_right.shape[1]:
            raise ValueError("additive factor ranks must match")
        arrays = (
            diagonal,
            transition_left,
            transition_right,
            additive_left,
            additive_right,
        )
        if not all(np.issubdtype(item.dtype, np.floating) for item in arrays):
            raise TypeError("affine KDA factors must use floating-point dtypes")
        if not all(np.isfinite(item).all() for item in arrays):
            raise ValueError("affine KDA factors must be finite")

    @classmethod
    def identity(
        cls,
        key_dimension: int,
        value_dimension: int,
        *,
        dtype: np.dtype | type[np.floating] = np.float64,
    ) -> AffineKDAFactor:
        if key_dimension < 1 or value_dimension < 1:
            raise ValueError("KDA factor dimensions must be positive")
        empty_key = np.empty((key_dimension, 0), dtype=dtype)
        empty_value = np.empty((value_dimension, 0), dtype=dtype)
        return cls(
            diagonal=np.ones(key_dimension, dtype=dtype),
            transition_left=empty_key.copy(),
            transition_right=empty_key.copy(),
            additive_left=empty_key.copy(),
            additive_right=empty_value,
        )

    @property
    def key_dimension(self) -> int:
        return int(self.diagonal.size)

    @property
    def value_dimension(self) -> int:
        return int(self.additive_right.shape[0])

    @property
    def transition_rank(self) -> int:
        return int(self.transition_left.shape[1])

    @property
    def additive_rank(self) -> int:
        return int(self.additive_left.shape[1])

    @property
    def stored_elements(self) -> int:
        return int(
            self.diagonal.size
            + self.transition_left.size
            + self.transition_right.size
            + self.additive_left.size
            + self.additive_right.size
        )

    def then(self, later: AffineKDAFactor) -> AffineKDAFactor:
        """Return ``later(self(H))`` without materializing an intermediate state."""

        if self.key_dimension != later.key_dimension:
            raise ValueError("composed KDA factors must have the same key dimension")
        if self.value_dimension != later.value_dimension:
            raise ValueError("composed KDA factors must have the same value dimension")

        # A2 A1, with Ai = Di + Ui Vi.T.
        first_left = later.diagonal[:, None] * self.transition_left
        second_right = self.diagonal[:, None] * later.transition_right + self.transition_right @ (
            self.transition_left.T @ later.transition_right
        )
        transition_left = np.concatenate((first_left, later.transition_left), axis=1)
        transition_right = np.concatenate((self.transition_right, second_right), axis=1)

        # A2 C1 + C2, with Ci = Li Ri.T.
        transformed_additive_left = later.diagonal[
            :, None
        ] * self.additive_left + later.transition_left @ (
            later.transition_right.T @ self.additive_left
        )
        additive_left = np.concatenate((transformed_additive_left, later.additive_left), axis=1)
        additive_right = np.concatenate((self.additive_right, later.additive_right), axis=1)
        return AffineKDAFactor(
            diagonal=later.diagonal * self.diagonal,
            transition_left=transition_left,
            transition_right=transition_right,
            additive_left=additive_left,
            additive_right=additive_right,
        )

    def apply(self, state: FloatArray) -> FloatArray:
        """Materialize the state produced by this factor exactly once."""

        state = _matrix(state, name="state")
        if state.shape != (self.key_dimension, self.value_dimension):
            raise ValueError("state dimensions do not match the KDA factor")
        result = self.diagonal[:, None] * state
        if self.transition_rank:
            result = result + self.transition_left @ (self.transition_right.T @ state)
        if self.additive_rank:
            result = result + self.additive_left @ self.additive_right.T
        return result

    def output(self, state: FloatArray, query: FloatArray) -> FloatArray:
        """Evaluate ``query.T @ Apply(state)`` without writing ``Apply(state)``."""

        state = _matrix(state, name="state")
        query = _vector(query, name="query")
        if state.shape != (self.key_dimension, self.value_dimension):
            raise ValueError("state dimensions do not match the KDA factor")
        if query.size != self.key_dimension:
            raise ValueError("query dimension does not match the KDA factor")
        effective_query = self.diagonal * query
        if self.transition_rank:
            effective_query = effective_query + self.transition_right @ (
                self.transition_left.T @ query
            )
        output = effective_query @ state
        if self.additive_rank:
            output = output + self.additive_right @ (self.additive_left.T @ query)
        return output


def token_factors(
    alphas: FloatArray,
    keys: FloatArray,
    values: FloatArray,
    betas: FloatArray,
) -> tuple[KDATokenFactor, ...]:
    """Build validated token factors from a verification window."""

    alphas = _matrix(alphas, name="alphas")
    keys = _matrix(keys, name="keys")
    values = _matrix(values, name="values")
    betas = _vector(betas, name="betas")
    rows = int(alphas.shape[0])
    if keys.shape != alphas.shape or values.shape[0] != rows or betas.size != rows:
        raise ValueError("KDA token-factor rows must match")
    return tuple(
        KDATokenFactor(alphas[row], keys[row], values[row], float(betas[row]))
        for row in range(rows)
    )


def serial_kda_block(
    state: FloatArray,
    factors: tuple[KDATokenFactor, ...],
    queries: FloatArray,
    *,
    retain_states: bool = False,
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, ...]]:
    """Execute the canonical serial recurrence for one KDA head."""

    state = _matrix(state, name="state").copy()
    queries = _matrix(queries, name="queries")
    if queries.shape[0] != len(factors):
        raise ValueError("query count must equal the number of KDA factors")
    outputs: list[FloatArray] = []
    states: list[FloatArray] = []
    for query, factor in zip(queries, factors, strict=True):
        if state.shape != (factor.key_dimension, factor.value_dimension):
            raise ValueError("state dimensions do not match the KDA token factor")
        decayed = factor.alpha[:, None] * state
        k_state = factor.key @ decayed
        delta = (factor.value - k_state) * factor.beta
        state = decayed + factor.key[:, None] * delta[None, :]
        outputs.append(query @ state)
        if retain_states:
            states.append(state.copy())
    output = np.stack(outputs) if outputs else np.empty((0, state.shape[1]), dtype=state.dtype)
    return output, state, tuple(states)


def factorized_kda_block(
    state: FloatArray,
    factors: tuple[KDATokenFactor, ...],
    queries: FloatArray,
    *,
    accepted_tokens: int | None = None,
) -> tuple[FloatArray, FloatArray, AffineKDAFactor]:
    """Evaluate prefix outputs and reconstruct only the accepted state."""

    state = _matrix(state, name="state")
    queries = _matrix(queries, name="queries")
    if queries.shape[0] != len(factors):
        raise ValueError("query count must equal the number of KDA factors")
    accepted = len(factors) if accepted_tokens is None else int(accepted_tokens)
    if not 0 <= accepted <= len(factors):
        raise ValueError("accepted token count is outside the verification window")
    if factors:
        key_dimension = factors[0].key_dimension
        value_dimension = factors[0].value_dimension
    else:
        key_dimension, value_dimension = map(int, state.shape)
    prefix = AffineKDAFactor.identity(key_dimension, value_dimension, dtype=state.dtype)
    accepted_prefix = prefix
    outputs: list[FloatArray] = []
    for index, (query, factor) in enumerate(zip(queries, factors, strict=True), start=1):
        prefix = prefix.then(factor.expand())
        outputs.append(prefix.output(state, query))
        if index == accepted:
            accepted_prefix = prefix
    if accepted == 0:
        accepted_prefix = AffineKDAFactor.identity(
            key_dimension, value_dimension, dtype=state.dtype
        )
    accepted_state = accepted_prefix.apply(state)
    output = np.stack(outputs) if outputs else np.empty((0, value_dimension), dtype=state.dtype)
    return output, accepted_state, accepted_prefix


def replay_kda_state(
    checkpoint: FloatArray,
    factors: tuple[KDATokenFactor, ...],
    *,
    accepted_tokens: int | None = None,
) -> FloatArray:
    """Reconstruct an accepted state by replaying compact token factors."""

    accepted = len(factors) if accepted_tokens is None else int(accepted_tokens)
    if not 0 <= accepted <= len(factors):
        raise ValueError("accepted token count is outside the replay window")
    state = _matrix(checkpoint, name="checkpoint").copy()
    for factor in factors[:accepted]:
        decayed = factor.alpha[:, None] * state
        k_state = factor.key @ decayed
        state = decayed + factor.key[:, None] * ((factor.value - k_state) * factor.beta)[None, :]
    return state


def kda_state_traffic_model(
    *,
    block_tokens: int,
    heads: int = 96,
    key_dimension: int = 128,
    value_dimension: int = 128,
    bytes_per_element: int = 4,
) -> dict[str, int | float]:
    """Return logical state/factor traffic bounds for one verification block."""

    if block_tokens < 1:
        raise ValueError("block_tokens must be positive")
    if min(heads, key_dimension, value_dimension, bytes_per_element) < 1:
        raise ValueError("KDA traffic-model dimensions must be positive")
    full_state_bytes = heads * key_dimension * value_dimension * bytes_per_element
    compact_factor_bytes = (
        heads * block_tokens * (2 * key_dimension + value_dimension + 1) * bytes_per_element
    )
    snapshot_bytes = full_state_bytes * block_tokens
    generic_prefix_peak_bytes = (
        heads
        * (key_dimension + block_tokens * (3 * key_dimension + value_dimension))
        * bytes_per_element
    )
    return {
        "block_tokens": block_tokens,
        "full_state_bytes": full_state_bytes,
        # The canonical kernel has two dense state loops: decay/k-state and
        # delta/output.  These are logical bytes; cache behavior is deliberately
        # not inferred without hardware counters.
        "serial_logical_state_read_bytes": 2 * snapshot_bytes,
        "serial_logical_state_write_bytes": 2 * snapshot_bytes,
        "speculative_full_snapshot_bytes": snapshot_bytes,
        "compact_token_factor_bytes": compact_factor_bytes,
        "generic_composed_factor_peak_bytes": generic_prefix_peak_bytes,
        "factor_bytes_vs_full_snapshots": compact_factor_bytes / snapshot_bytes,
        "snapshot_compression_ratio": snapshot_bytes / compact_factor_bytes,
        "ideal_factorized_committed_state_read_bytes": 2 * full_state_bytes,
        "ideal_factorized_committed_state_write_bytes": full_state_bytes,
        "accepted_state_copy_bytes": 0,
        "accepted_state_copy_launches": 0,
        "accepted_state_materializations_factorized": 1,
    }
