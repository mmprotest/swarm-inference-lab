"""Final-stage sampling that keeps vocabulary logits on the model device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SamplingParameters:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 0
    diagnostic_top_k: int = 0
    return_full_logits: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.diagnostic_top_k < 0:
            raise ValueError("diagnostic_top_k cannot be negative")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


@dataclass(slots=True)
class SamplingResult:
    token_ids: Any
    selected_logits: Any
    top_token_ids: Any | None = None
    top_logits: Any | None = None
    full_logits: Any | None = None

    def coordinator_payload_bytes(self) -> int:
        tensors = [
            self.token_ids,
            self.selected_logits,
            self.top_token_ids,
            self.top_logits,
            self.full_logits,
        ]
        return sum(
            int(value.numel() * value.element_size()) for value in tensors if value is not None
        )


class SamplingState:
    """Deterministic per-request CUDA random streams for future non-greedy use."""

    def __init__(self, torch_module: Any, device: Any) -> None:
        self.torch = torch_module
        self.device = device
        self._generators: dict[tuple[str, int], Any] = {}

    def generator(self, request_id: str, seed: int) -> Any:
        key = (request_id, seed)
        generator = self._generators.get(key)
        if generator is None:
            generator = self.torch.Generator(device=self.device)
            generator.manual_seed(seed)
            self._generators[key] = generator
        return generator

    def delete(self, request_id: str) -> None:
        for key in [key for key in self._generators if key[0] == request_id]:
            self._generators.pop(key)


def _apply_repetition_penalty(
    torch_module: Any,
    scores: Any,
    token_history: Any | None,
    penalty: float,
) -> Any:
    if token_history is None or penalty == 1.0:
        return scores
    adjusted = scores.clone()
    for batch_index in range(int(scores.shape[0])):
        unique_tokens = torch_module.unique(token_history[batch_index])
        selected = adjusted[batch_index].gather(0, unique_tokens)
        selected = torch_module.where(
            selected < 0,
            selected * penalty,
            selected / penalty,
        )
        adjusted[batch_index].scatter_(0, unique_tokens, selected)
    return adjusted


def _filter_top_k_top_p(
    torch_module: Any,
    scores: Any,
    *,
    top_k: int,
    top_p: float,
) -> Any:
    filtered = scores
    if top_k > 0 and top_k < int(scores.shape[-1]):
        threshold = torch_module.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch_module.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        probabilities = torch_module.softmax(sorted_scores, dim=-1)
        cumulative = torch_module.cumsum(probabilities, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        restored = torch_module.full_like(filtered, float("-inf"))
        filtered = restored.scatter(-1, sorted_indices, sorted_scores)
    return filtered


def sample_final_logits(
    torch_module: Any,
    logits: Any,
    *,
    parameters: SamplingParameters,
    request_ids: tuple[str, ...],
    state: SamplingState | None = None,
    token_history: Any | None = None,
) -> SamplingResult:
    """Select tokens without transferring the vocabulary vector to the host."""

    if logits.ndim == 3:
        scores = logits[:, -1, :]
    elif logits.ndim == 2:
        scores = logits
    else:
        raise ValueError(f"sampling requires [batch, vocab] logits, got {tuple(logits.shape)}")
    if len(request_ids) != int(scores.shape[0]):
        raise ValueError("sampling request IDs do not match logit batch size")
    if parameters.greedy:
        token_ids = torch_module.argmax(scores, dim=-1)
    else:
        scaled = scores / parameters.temperature
        scaled = _apply_repetition_penalty(
            torch_module,
            scaled,
            token_history,
            parameters.repetition_penalty,
        )
        scaled = _filter_top_k_top_p(
            torch_module,
            scaled,
            top_k=parameters.top_k,
            top_p=parameters.top_p,
        )
        probabilities = torch_module.softmax(scaled, dim=-1)
        sampling_state = state or SamplingState(torch_module, scores.device)
        sampled = [
            torch_module.multinomial(
                probabilities[index],
                num_samples=1,
                generator=sampling_state.generator(request_id, parameters.seed),
            )
            for index, request_id in enumerate(request_ids)
        ]
        token_ids = torch_module.cat(sampled, dim=0)
    selected_logits = scores.gather(1, token_ids[:, None]).squeeze(1)
    top_token_ids = None
    top_logits = None
    if parameters.diagnostic_top_k:
        count = min(parameters.diagnostic_top_k, int(scores.shape[-1]))
        top = torch_module.topk(scores, count, dim=-1)
        top_token_ids = top.indices
        top_logits = top.values
    return SamplingResult(
        token_ids=token_ids,
        selected_logits=selected_logits,
        top_token_ids=top_token_ids,
        top_logits=top_logits,
        full_logits=scores if parameters.return_full_logits else None,
    )
