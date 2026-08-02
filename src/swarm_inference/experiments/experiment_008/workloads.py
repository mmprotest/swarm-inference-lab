"""Deterministic, multi-domain workload construction for Experiment 008."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

Tokenizer = Callable[[str], list[int]]


@dataclass(frozen=True, slots=True)
class WorkloadPrompt:
    fixture_id: str
    workload: str
    domain: str
    text_sha256: str
    text: str
    token_ids: list[int]
    requested_output_tokens: int

    def manifest(self, *, include_text: bool = False) -> dict[str, object]:
        payload = asdict(self)
        if not include_text:
            payload.pop("text")
        payload["input_token_count"] = len(self.token_ids)
        return payload


_DECODE_SEEDS: list[tuple[str, str]] = [
    (
        "general_knowledge",
        "Explain how public libraries changed from book repositories into community information services. Distinguish established facts from plausible interpretation.",
    ),
    (
        "reasoning",
        "A museum has three connected halls with different closing times. Develop a logically ordered visitor route and state every assumption used.",
    ),
    (
        "mathematics",
        "Derive the sum of the first n odd positive integers, prove the result two ways, and check the cases n=1 through n=4.",
    ),
    (
        "coding",
        "Write a clear Python design for a bounded producer-consumer queue with cancellation. Explain invariants, failure modes, and tests.",
    ),
    (
        "summarisation",
        "Summarise the following topic for a non-specialist: why measurement uncertainty, calibration, and repeatability matter in experimental engineering.",
    ),
    (
        "long_form",
        "Compose a structured essay about the trade-off between local resilience and global efficiency in infrastructure planning, including counterarguments.",
    ),
    (
        "structured_output",
        "Return a JSON object that compares solar, wind, and hydro power using fields for strengths, constraints, deployment timescale, and uncertainty. Then validate its schema.",
    ),
]

_FILLER = (
    "Use precise language, make assumptions visible, avoid invented citations, and separate evidence "
    "from inference. Include a compact example, one limitation, and a way the conclusion could be "
    "tested. The response should remain self-contained and should not depend on hidden metadata. "
)

_LONG_SEEDS = [
    "A technical history of navigation methods, from coastal landmarks through satellite systems",
    "A comparative survey of urban water management under drought and flood conditions",
    "A software architecture review covering observability, fault isolation, and capacity planning",
    "A mathematical tutorial on probability calibration and proper scoring rules",
    "A policy-neutral synthesis of food supply chains, storage, transport, and waste reduction",
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fit_tokens(
    seed: str,
    *,
    tokenizer: Tokenizer,
    minimum: int,
    maximum: int,
    suffix_seed: int,
) -> tuple[str, list[int]]:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid workload token bounds")
    generator = random.Random(suffix_seed)
    pieces = [seed]
    tokens = tokenizer("\n\n".join(pieces))
    while len(tokens) < minimum:
        probe = tokenizer(f"Context note 000000: {_FILLER}")
        estimated_per_piece = max(len(probe), 1)
        needed = max(1, (minimum - len(tokens) + estimated_per_piece - 1) // estimated_per_piece)
        for _ in range(needed):
            marker = generator.randrange(1_000_000)
            pieces.append(f"Context note {marker}: {_FILLER}")
        tokens = tokenizer("\n\n".join(pieces))
    if len(tokens) > maximum:
        tokens = tokens[:maximum]
    return "\n\n".join(pieces), tokens


def build_decode_workload(
    tokenizer: Tokenizer,
    *,
    prompt_count: int,
    input_minimum: int,
    input_maximum: int,
    output_tokens: int,
    seed: int,
) -> list[WorkloadPrompt]:
    if prompt_count < 20:
        raise ValueError("official decode workload requires at least 20 prompts")
    prompts: list[WorkloadPrompt] = []
    for index in range(prompt_count):
        domain, base = _DECODE_SEEDS[index % len(_DECODE_SEEDS)]
        variant = index // len(_DECODE_SEEDS)
        text, token_ids = _fit_tokens(
            f"{base}\nVariant {variant}: approach the question independently.",
            tokenizer=tokenizer,
            minimum=input_minimum,
            maximum=input_maximum,
            suffix_seed=seed + index,
        )
        prompts.append(
            WorkloadPrompt(
                fixture_id=f"decode-{index:03d}",
                workload="decode",
                domain=domain,
                text_sha256=_hash(text),
                text=text,
                token_ids=token_ids,
                requested_output_tokens=output_tokens,
            )
        )
    return prompts


def build_long_context_workload(
    tokenizer: Tokenizer,
    *,
    target_tokens: int,
    prompt_count: int,
    output_tokens: int,
    seed: int,
) -> list[WorkloadPrompt]:
    if target_tokens < 1 or prompt_count < 1:
        raise ValueError("long-context workload dimensions must be positive")
    prompts: list[WorkloadPrompt] = []
    for index in range(prompt_count):
        topic = _LONG_SEEDS[index % len(_LONG_SEEDS)]
        text, token_ids = _fit_tokens(
            (
                f"Prepare {topic}. Build the analysis from the supplied neutral context, identify "
                "recurring themes, and finish with a concise synthesis."
            ),
            tokenizer=tokenizer,
            minimum=target_tokens,
            maximum=target_tokens,
            suffix_seed=seed + target_tokens + index,
        )
        prompts.append(
            WorkloadPrompt(
                fixture_id=f"prefill-{target_tokens}-{index:02d}",
                workload=f"prefill_{target_tokens // 1000}k",
                domain="long_context",
                text_sha256=_hash(text),
                text=text,
                token_ids=token_ids,
                requested_output_tokens=output_tokens,
            )
        )
    return prompts


def build_correctness_prompts(tokenizer: Tokenizer, *, count: int = 10) -> list[WorkloadPrompt]:
    prompts: list[WorkloadPrompt] = []
    for index in range(count):
        domain, text = _DECODE_SEEDS[index % len(_DECODE_SEEDS)]
        text = f"{text}\nCorrectness case {index}. Answer concisely."
        prompts.append(
            WorkloadPrompt(
                fixture_id=f"correctness-{index:03d}",
                workload="correctness",
                domain=domain,
                text_sha256=_hash(text),
                text=text,
                token_ids=tokenizer(text),
                requested_output_tokens=16,
            )
        )
    return prompts


def scheduling_features(
    prompt: WorkloadPrompt, *, batch_size: int, concurrency: int
) -> dict[str, int]:
    """Only policy-permitted features cross the planner boundary."""

    return {
        "prompt_length": len(prompt.token_ids),
        "requested_generation_length": prompt.requested_output_tokens,
        "batch_size": batch_size,
        "concurrency": concurrency,
    }
