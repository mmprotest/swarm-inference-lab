"""Routing-aware batching with hard queue-delay bounds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BatchingPolicy(StrEnum):
    NONE = "none"
    ARRIVAL_ORDER = "arrival_order"
    RANDOM = "random"
    EXPERT_OVERLAP = "expert_overlap"
    DOMAIN_INFORMED = "domain_informed"
    PLANNER = "planner"


@dataclass(frozen=True, slots=True)
class RoutedRequest:
    request_id: str
    arrival_ns: int
    expert_ids: tuple[int, ...]
    expert_bytes: int
    domain: str | None = None

    def __post_init__(self) -> None:
        if self.arrival_ns < 0 or not self.expert_ids or self.expert_bytes < 0:
            raise ValueError("routed request fields are invalid")


@dataclass(frozen=True, slots=True)
class RoutingBatch:
    batch_id: str
    requests: tuple[RoutedRequest, ...]
    released_ns: int
    policy: BatchingPolicy

    def metrics(self) -> dict[str, Any]:
        before = sum(len(item.expert_ids) for item in self.requests)
        unique = set().union(*(set(item.expert_ids) for item in self.requests))
        bytes_before = sum(item.expert_bytes * len(item.expert_ids) for item in self.requests)
        bytes_per_expert = max((item.expert_bytes for item in self.requests), default=0)
        bytes_after = len(unique) * bytes_per_expert
        queue_delays = [self.released_ns - item.arrival_ns for item in self.requests]
        return {
            "batch_id": self.batch_id,
            "policy": self.policy.value,
            "request_ids": [item.request_id for item in self.requests],
            "batch_size": len(self.requests),
            "queue_delay_ns_max": max(queue_delays, default=0),
            "queue_delay_ns_mean": sum(queue_delays) / len(queue_delays),
            "expert_selections_before_union": before,
            "unique_experts_after_union": len(unique),
            "deduplication_ratio": 1 - len(unique) / before if before else 0.0,
            "expert_bytes_before_union": bytes_before,
            "expert_bytes_after_union": bytes_after,
            "expert_ids_after_union": sorted(unique),
        }


def _overlap(left: RoutedRequest, right: RoutedRequest) -> float:
    a, b = set(left.expert_ids), set(right.expert_ids)
    return len(a & b) / len(a | b)


def make_routing_batches(
    requests: list[RoutedRequest],
    *,
    policy: BatchingPolicy | str,
    maximum_batch_size: int,
    maximum_queue_delay_ns: int,
    seed: int = 1010,
    planner_policy: BatchingPolicy | str = BatchingPolicy.EXPERT_OVERLAP,
) -> list[RoutingBatch]:
    if maximum_batch_size <= 0 or maximum_queue_delay_ns < 0:
        raise ValueError("batch size must be positive and queue delay non-negative")
    selected = BatchingPolicy(policy)
    if selected == BatchingPolicy.PLANNER:
        selected = BatchingPolicy(planner_policy)
        if selected == BatchingPolicy.PLANNER:
            raise ValueError("planner batching policy cannot recursively select planner")
    remaining = sorted(requests, key=lambda item: (item.arrival_ns, item.request_id))
    batches = []
    generator = random.Random(seed)
    while remaining:
        anchor = remaining.pop(0)
        if selected == BatchingPolicy.NONE:
            chosen = [anchor]
        else:
            eligible = [
                item
                for item in remaining
                if item.arrival_ns - anchor.arrival_ns <= maximum_queue_delay_ns
            ]
            if selected == BatchingPolicy.ARRIVAL_ORDER:
                ranked = eligible
            elif selected == BatchingPolicy.RANDOM:
                ranked = list(eligible)
                generator.shuffle(ranked)
            elif selected == BatchingPolicy.EXPERT_OVERLAP:
                ranked = sorted(
                    eligible,
                    key=lambda item: (_overlap(anchor, item), -item.arrival_ns, item.request_id),
                    reverse=True,
                )
            elif selected == BatchingPolicy.DOMAIN_INFORMED:
                ranked = sorted(
                    eligible,
                    key=lambda item: (
                        item.domain is not None and item.domain == anchor.domain,
                        -item.arrival_ns,
                        item.request_id,
                    ),
                    reverse=True,
                )
            else:  # pragma: no cover - exhaustive enum guard
                raise ValueError(f"unsupported batching policy {selected}")
            chosen = [anchor, *ranked[: maximum_batch_size - 1]]
            chosen_ids = {item.request_id for item in chosen}
            remaining = [item for item in remaining if item.request_id not in chosen_ids]
        release_ns = max(item.arrival_ns for item in chosen)
        if release_ns - min(item.arrival_ns for item in chosen) > maximum_queue_delay_ns:
            raise AssertionError("batching implementation violated the queue-delay bound")
        batches.append(
            RoutingBatch(
                batch_id=f"batch-{len(batches):05d}",
                requests=tuple(chosen),
                released_ns=release_ns,
                policy=selected,
            )
        )
    return batches


def batching_summary(batches: list[RoutingBatch]) -> dict[str, Any]:
    rows = [batch.metrics() for batch in batches]
    selections = sum(row["expert_selections_before_union"] for row in rows)
    unique = sum(row["unique_experts_after_union"] for row in rows)
    return {
        "batch_count": len(rows),
        "request_count": sum(row["batch_size"] for row in rows),
        "mean_batch_size": (sum(row["batch_size"] for row in rows) / len(rows) if rows else 0.0),
        "expert_selections_before_union": selections,
        "unique_experts_after_union": unique,
        "deduplication_ratio": 1 - unique / selections if selections else 0.0,
        "maximum_queue_delay_ns": max((row["queue_delay_ns_max"] for row in rows), default=0),
        "batches": rows,
    }


def batching_policy_inventory() -> list[dict[str, str]]:
    return [
        {"policy": policy.value, "input_fields": "arrival,route"}
        if policy == BatchingPolicy.EXPERT_OVERLAP
        else {"policy": policy.value, "input_fields": "arrival,domain"}
        if policy == BatchingPolicy.DOMAIN_INFORMED
        else {"policy": policy.value, "input_fields": "arrival"}
        for policy in BatchingPolicy
    ]
