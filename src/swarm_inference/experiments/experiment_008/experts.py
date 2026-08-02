"""Activation statistics, residency classes, cache accounting, and bounded predictors."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from pydantic import Field

from swarm_inference.config.models import StrictModel


class ExpertActivation(StrictModel):
    token_index: int = Field(ge=0)
    layer_id: int = Field(ge=0)
    phase: Literal["prefill", "decode"]
    expert_ids: list[int]


class ExpertMeasurement(StrictModel):
    layer_id: int = Field(ge=0)
    expert_id: int = Field(ge=0)
    weight_bytes: int = Field(gt=0)
    cpu_latency_ms: float = Field(gt=0)
    gpu_latency_ms: float = Field(gt=0)
    transfer_latency_ms: float = Field(ge=0)


class ExpertStatistics(StrictModel):
    layer_id: int
    expert_id: int
    activation_count: int
    activation_probability: float
    consecutive_token_reuse: int
    consecutive_token_reuse_probability: float
    prefill_activation_count: int
    decode_activation_count: int
    coactivation: dict[int, int]
    cpu_latency_ms: float | None
    gpu_latency_ms: float | None
    transfer_latency_ms: float | None
    weight_bytes: int | None


class ExpertResidencyDecision(StrictModel):
    layer_id: int
    expert_id: int
    residency_class: Literal["hot", "warm", "cold"]
    policy: Literal["gpu_resident", "cpu_prefetch", "cpu_execute", "load_on_demand"]
    expected_gpu_residency_value_ms: float
    explanation: list[str]


class PredictionEvent(StrictModel):
    token_index: int
    layer_id: int
    predictor: str
    predicted_experts: list[int]
    actual_experts: list[int]
    true_positives: int
    false_positives: int
    false_negatives: int
    bytes_prefetched: int
    useful_bytes: int
    wasted_bytes: int
    transfer_time_hidden_ms: float
    transfer_interference_ms: float
    visible_transfer_latency_removed_ms: float


def activation_statistics(
    events: list[ExpertActivation],
    measurements: list[ExpertMeasurement] | None = None,
) -> tuple[list[ExpertStatistics], list[dict[str, int]]]:
    measurement_map = {(item.layer_id, item.expert_id): item for item in (measurements or [])}
    counts: Counter[tuple[int, int]] = Counter()
    phase_counts: Counter[tuple[int, int, str]] = Counter()
    coactivation: Counter[tuple[int, int, int]] = Counter()
    reuse: Counter[tuple[int, int]] = Counter()
    opportunities: Counter[int] = Counter()
    previous: dict[int, set[int]] = {}
    for event in sorted(events, key=lambda item: (item.token_index, item.layer_id)):
        unique = sorted(set(event.expert_ids))
        opportunities[event.layer_id] += 1
        for expert_id in unique:
            counts[(event.layer_id, expert_id)] += 1
            phase_counts[(event.layer_id, expert_id, event.phase)] += 1
            if expert_id in previous.get(event.layer_id, set()):
                reuse[(event.layer_id, expert_id)] += 1
        for first, second in combinations(unique, 2):
            coactivation[(event.layer_id, first, second)] += 1
        previous[event.layer_id] = set(unique)
    keys = sorted(set(counts) | set(measurement_map))
    rows: list[ExpertStatistics] = []
    for layer_id, expert_id in keys:
        count = counts[(layer_id, expert_id)]
        measured = measurement_map.get((layer_id, expert_id))
        related: dict[int, int] = {}
        for (co_layer, first, second), value in coactivation.items():
            if co_layer == layer_id and expert_id in {first, second}:
                related[second if first == expert_id else first] = value
        rows.append(
            ExpertStatistics(
                layer_id=layer_id,
                expert_id=expert_id,
                activation_count=count,
                activation_probability=count / opportunities[layer_id]
                if opportunities[layer_id]
                else 0.0,
                consecutive_token_reuse=reuse[(layer_id, expert_id)],
                consecutive_token_reuse_probability=(
                    reuse[(layer_id, expert_id)] / count if count else 0.0
                ),
                prefill_activation_count=phase_counts[(layer_id, expert_id, "prefill")],
                decode_activation_count=phase_counts[(layer_id, expert_id, "decode")],
                coactivation=related,
                cpu_latency_ms=measured.cpu_latency_ms if measured else None,
                gpu_latency_ms=measured.gpu_latency_ms if measured else None,
                transfer_latency_ms=measured.transfer_latency_ms if measured else None,
                weight_bytes=measured.weight_bytes if measured else None,
            )
        )
    co_rows = [
        {"layer_id": layer, "expert_a": first, "expert_b": second, "count": count}
        for (layer, first, second), count in sorted(coactivation.items())
    ]
    return rows, co_rows


def classify_residency(
    statistics: list[ExpertStatistics],
    *,
    gpu_budget_bytes: int,
) -> list[ExpertResidencyDecision]:
    if gpu_budget_bytes < 0:
        raise ValueError("GPU expert budget cannot be negative")
    scored: list[tuple[float, ExpertStatistics]] = []
    for row in statistics:
        if None in (
            row.weight_bytes,
            row.cpu_latency_ms,
            row.gpu_latency_ms,
            row.transfer_latency_ms,
        ):
            value = float("-inf")
        else:
            avoided = max(
                float(row.cpu_latency_ms) - float(row.gpu_latency_ms),
                float(row.transfer_latency_ms),
                0.0,
            )
            value = row.activation_probability * avoided / max(int(row.weight_bytes), 1)
        scored.append((value, row))
    scored.sort(key=lambda item: (item[0], item[1].activation_count), reverse=True)
    used = 0
    decisions: list[ExpertResidencyDecision] = []
    for value, row in scored:
        weight = int(row.weight_bytes or 0)
        measurable = value != float("-inf")
        if measurable and value > 0 and used + weight <= gpu_budget_bytes:
            residency, policy = "hot", "gpu_resident"
            used += weight
        elif (
            measurable
            and row.transfer_latency_ms is not None
            and row.cpu_latency_ms is not None
            and row.gpu_latency_ms is not None
            and row.transfer_latency_ms + row.gpu_latency_ms < row.cpu_latency_ms
        ):
            residency, policy = "warm", "cpu_prefetch"
        elif (
            measurable
            and row.cpu_latency_ms is not None
            and row.transfer_latency_ms is not None
            and row.gpu_latency_ms is not None
            and row.cpu_latency_ms <= row.transfer_latency_ms + row.gpu_latency_ms
        ):
            residency, policy = "cold", "cpu_execute"
        else:
            residency, policy = "cold", "load_on_demand"
        decisions.append(
            ExpertResidencyDecision(
                layer_id=row.layer_id,
                expert_id=row.expert_id,
                residency_class=residency,
                policy=policy,
                expected_gpu_residency_value_ms=(0.0 if value == float("-inf") else value * weight),
                explanation=[
                    f"activation frequency: {row.activation_probability:.6f}",
                    f"weight bytes: {row.weight_bytes if row.weight_bytes is not None else 'unavailable'}",
                    f"transfer cost ms: {row.transfer_latency_ms if row.transfer_latency_ms is not None else 'unavailable'}",
                    f"CPU execution ms: {row.cpu_latency_ms if row.cpu_latency_ms is not None else 'unavailable'}",
                    f"GPU execution ms: {row.gpu_latency_ms if row.gpu_latency_ms is not None else 'unavailable'}",
                    f"selected policy: {policy}",
                ],
            )
        )
    return sorted(decisions, key=lambda item: (item.layer_id, item.expert_id))


class BoundedExpertPredictor:
    def __init__(self, *, window: int = 16, maximum_predictions: int = 2) -> None:
        if window <= 0 or maximum_predictions <= 0:
            raise ValueError("predictor window and bound must be positive")
        self.window = window
        self.maximum_predictions = maximum_predictions
        self.history: dict[int, deque[list[int]]] = defaultdict(lambda: deque(maxlen=window))
        self.adjacent: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        self.last_by_token_layer: dict[tuple[int, int], list[int]] = {}

    def predict(self, *, token_index: int, layer_id: int, mode: str) -> list[int]:
        if mode == "previous_token_reuse":
            return list(self.last_by_token_layer.get((token_index - 1, layer_id), []))[
                : self.maximum_predictions
            ]
        if mode == "sliding_window_frequency":
            counts = Counter(expert for selection in self.history[layer_id] for expert in selection)
            return [expert for expert, _ in counts.most_common(self.maximum_predictions)]
        if mode == "adjacent_layer_correlation":
            previous = self.last_by_token_layer.get((token_index, layer_id - 1), [])
            counts: Counter[int] = Counter()
            for expert in previous:
                counts.update(self.adjacent[(layer_id - 1, expert)])
            return [expert for expert, _ in counts.most_common(self.maximum_predictions)]
        raise ValueError(f"unknown expert predictor {mode}")

    def observe(self, event: ExpertActivation) -> None:
        actual = sorted(set(event.expert_ids))
        previous_layer = self.last_by_token_layer.get((event.token_index, event.layer_id - 1), [])
        for previous_expert in previous_layer:
            self.adjacent[(event.layer_id - 1, previous_expert)].update(actual)
        self.history[event.layer_id].append(actual)
        self.last_by_token_layer[(event.token_index, event.layer_id)] = actual


def evaluate_prediction(
    *,
    token_index: int,
    layer_id: int,
    predictor: str,
    predicted: Iterable[int],
    actual: Iterable[int],
    bytes_by_expert: dict[int, int],
    transfer_ms_by_expert: dict[int, float],
    overlap_available_ms: float,
    measured_interference_ms: float,
) -> PredictionEvent:
    predicted_set = set(predicted)
    actual_set = set(actual)
    useful = predicted_set & actual_set
    wasted = predicted_set - actual_set
    bytes_prefetched = sum(bytes_by_expert.get(expert, 0) for expert in predicted_set)
    useful_bytes = sum(bytes_by_expert.get(expert, 0) for expert in useful)
    transfer_hidden = sum(
        min(transfer_ms_by_expert.get(expert, 0.0), overlap_available_ms) for expert in useful
    )
    visible_removed = max(transfer_hidden - measured_interference_ms, 0.0)
    return PredictionEvent(
        token_index=token_index,
        layer_id=layer_id,
        predictor=predictor,
        predicted_experts=sorted(predicted_set),
        actual_experts=sorted(actual_set),
        true_positives=len(useful),
        false_positives=len(wasted),
        false_negatives=len(actual_set - predicted_set),
        bytes_prefetched=bytes_prefetched,
        useful_bytes=useful_bytes,
        wasted_bytes=bytes_prefetched - useful_bytes,
        transfer_time_hidden_ms=transfer_hidden,
        transfer_interference_ms=measured_interference_ms,
        visible_transfer_latency_removed_ms=visible_removed,
    )


@dataclass(frozen=True, slots=True)
class CacheAccess:
    key: tuple[int, int]
    hit: bool
    evicted: tuple[tuple[int, int], ...]


class ExpertLRUCache:
    """Byte-bounded metadata cache; tensor transfer is owned by the backend runtime."""

    def __init__(self, capacity_bytes: int, pinned: Iterable[tuple[int, int]] = ()) -> None:
        if capacity_bytes < 0:
            raise ValueError("cache capacity cannot be negative")
        self.capacity_bytes = capacity_bytes
        self.pinned = set(pinned)
        self.entries: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0

    def access(self, key: tuple[int, int], byte_size: int) -> CacheAccess:
        if byte_size <= 0:
            raise ValueError("expert cache entries must have positive size")
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return CacheAccess(key, True, ())
        self.misses += 1
        if byte_size > self.capacity_bytes:
            return CacheAccess(key, False, ())
        evicted: list[tuple[int, int]] = []
        while self.used_bytes + byte_size > self.capacity_bytes:
            victim = next((item for item in self.entries if item not in self.pinned), None)
            if victim is None:
                return CacheAccess(key, False, tuple(evicted))
            victim_size = self.entries.pop(victim)
            self.used_bytes -= victim_size
            evicted.append(victim)
        self.entries[key] = byte_size
        self.used_bytes += byte_size
        return CacheAccess(key, False, tuple(evicted))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
