"""Deterministic workload identifiers and fixed prompt sets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from swarm_inference.config.models import WorkloadClass

DEFAULT_PROMPTS = (
    "Explain why distributed inference is difficult.",
    "Describe a bounded queue in two sentences.",
    "What distinguishes throughput from latency?",
    "Give one reason deterministic simulation is useful.",
    "Explain cache reconstruction after a stage failure.",
    "Why can a slow node reduce pipeline performance?",
    "Define a falsifiable scaling claim.",
    "Summarise probabilistic redundant execution.",
)


@dataclass(frozen=True, slots=True)
class WorkloadItem:
    request_id: str
    prompt: str
    workload_class: WorkloadClass
    random_seed: int


def build_workload(
    *,
    count: int,
    seed: int,
    workload_class: WorkloadClass,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> list[WorkloadItem]:
    if count <= 0:
        raise ValueError("workload count must be positive")
    if not prompts:
        raise ValueError("prompt set cannot be empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(prompts), size=count)
    items: list[WorkloadItem] = []
    for index, prompt_index in enumerate(indices):
        request_id = hashlib.sha256(
            f"{seed}:{index}:{prompts[int(prompt_index)]}".encode()
        ).hexdigest()[:16]
        items.append(
            WorkloadItem(
                request_id=request_id,
                prompt=prompts[int(prompt_index)],
                workload_class=workload_class,
                random_seed=seed + index,
            )
        )
    return items
