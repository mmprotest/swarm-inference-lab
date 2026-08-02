"""Backend-neutral held-out routing-policy and batch-union evaluation."""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from swarm_inference.backends.colibri.schemas import RouteSelection
from swarm_inference.config.models import StrictModel

_WORKLOAD_GROUPS = frozenset(
    {"general_chat", "coding", "mathematics_reasoning", "multilingual_long_form"}
)


class PromptPartition(StrictModel):
    prompt_id: str
    workload_group: Literal[
        "general_chat", "coding", "mathematics_reasoning", "multilingual_long_form"
    ]
    partition: Literal["calibration", "heldout"]


class PlacementPolicy(StrEnum):
    PLAIN_LRU = "plain_lru"
    FREQUENCY = "frequency_hot_experts"
    RECENT_REUSE = "recent_token_reuse"
    TRANSITION = "transition_history"
    COLIBRI_RECOMMENDED = "colibri_recommended"
    SWARM = "swarm_planner"


def calibration_hot_pin_bitmap(
    selections: Iterable[RouteSelection],
    *,
    layer_count: int,
    experts_per_layer: int,
    hot_slots_per_layer: int,
) -> tuple[bytes, dict[str, Any]]:
    """Encode calibration-frequency placement in Colibri's native bitmap."""

    if layer_count <= 0 or experts_per_layer <= 0:
        raise ValueError("hot-pin geometry must be positive")
    if hot_slots_per_layer < 0 or hot_slots_per_layer > experts_per_layer:
        raise ValueError("hot-pin capacity must fit the expert geometry")
    frequency: Counter[tuple[int, int]] = Counter()
    for row in selections:
        if row.layer_id >= layer_count or row.expert_id >= experts_per_layer:
            raise ValueError("calibration route falls outside hot-pin geometry")
        frequency[(row.layer_id, row.expert_id)] += 1
    bitmap = bytearray(layer_count * experts_per_layer)
    selected: list[dict[str, int]] = []
    for layer in range(layer_count):
        ranked = sorted(
            (
                (expert, frequency[(layer, expert)])
                for expert in range(experts_per_layer)
                if frequency[(layer, expert)]
            ),
            key=lambda item: (-item[1], item[0]),
        )[:hot_slots_per_layer]
        for expert, count in ranked:
            bitmap[layer * experts_per_layer + expert] = 1
            selected.append({"layer_id": layer, "expert_id": expert, "activation_count": count})
    return bytes(bitmap), {
        "source": "calibration_route_frequency",
        "layer_count": layer_count,
        "experts_per_layer": experts_per_layer,
        "hot_slots_per_layer": hot_slots_per_layer,
        "pinned_expert_count": len(selected),
        "selected": selected,
        "prompt_text_or_labels_used": False,
    }


class PolicyEvaluation(StrictModel):
    policy: PlacementPolicy
    partition: Literal["calibration", "heldout"]
    measured: bool
    selection_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    expert_hits: int = Field(ge=0)
    expert_misses: int = Field(ge=0)
    expert_hit_rate: float | None = Field(default=None, ge=0, le=1)
    bytes_read: int = Field(ge=0)
    bytes_read_per_token: float | None = Field(default=None, ge=0)
    tier_churn: int = Field(ge=0)
    prefetch_useful_bytes: int = Field(ge=0)
    prefetch_wasted_bytes: int = Field(ge=0)
    decode_tokens_per_second: float | None = Field(default=None, ge=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    expert_load_latency_ms: float | None = Field(default=None, ge=0)
    heldout_regret: float | None = Field(default=None, ge=0)
    rejected: bool = False
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def reconcile(self) -> PolicyEvaluation:
        if self.expert_hits + self.expert_misses != self.selection_count:
            raise ValueError("placement hit and miss counts must reconcile to selections")
        expected = self.expert_hits / self.selection_count if self.selection_count else None
        if self.expert_hit_rate != expected:
            raise ValueError("placement hit rate does not reconcile to counts")
        expected_bytes = self.bytes_read / self.token_count if self.token_count else None
        if self.bytes_read_per_token != expected_bytes:
            raise ValueError("bytes-per-token does not reconcile to raw bytes and tokens")
        return self


def validate_prompt_partitions(rows: Iterable[PromptPartition]) -> dict[str, Any]:
    partitions = list(rows)
    identifiers = [row.prompt_id for row in partitions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration and held-out prompt IDs must be disjoint")
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in partitions:
        coverage[row.partition].add(row.workload_group)
    missing = {
        partition: sorted(_WORKLOAD_GROUPS.difference(groups))
        for partition, groups in coverage.items()
        if groups != _WORKLOAD_GROUPS
    }
    if set(coverage) != {"calibration", "heldout"}:
        missing["partitions"] = sorted({"calibration", "heldout"}.difference(coverage))
    if missing:
        raise ValueError(f"prompt split lacks required workload coverage: {missing}")
    return {
        "valid": True,
        "calibration_prompts": sum(row.partition == "calibration" for row in partitions),
        "heldout_prompts": sum(row.partition == "heldout" for row in partitions),
        "workload_groups": sorted(_WORKLOAD_GROUPS),
    }


class RoutingPolicyEvaluator:
    """Replay observed routes through bounded local expert-cache policies."""

    def __init__(
        self,
        *,
        expert_bytes: dict[tuple[int, int], int],
        cache_slots_per_layer: int,
        hot_slots_per_layer: int = 0,
    ) -> None:
        if cache_slots_per_layer <= 0:
            raise ValueError("cache capacity must be positive")
        if hot_slots_per_layer < 0 or hot_slots_per_layer > cache_slots_per_layer:
            raise ValueError("hot expert capacity must fit inside cache capacity")
        self.expert_bytes = expert_bytes
        self.cache_slots_per_layer = cache_slots_per_layer
        self.hot_slots_per_layer = hot_slots_per_layer

    def evaluate(
        self,
        *,
        policy: PlacementPolicy,
        calibration: Iterable[RouteSelection],
        heldout: Iterable[RouteSelection],
        measured_metrics: dict[str, float] | None = None,
    ) -> PolicyEvaluation:
        calibration_rows = list(calibration)
        heldout_rows = sorted(
            heldout,
            key=lambda row: (row.call_index, row.row_index, row.layer_id, row.expert_id),
        )
        frequency = Counter((row.layer_id, row.expert_id) for row in calibration_rows)
        transitions = self._transition_counts(calibration_rows)
        pinned = self._pinned(policy, frequency)
        cache: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        for layer, experts in pinned.items():
            for expert in experts:
                cache[layer][expert] = None
        hits = misses = bytes_read = churn = useful = 0
        prefetched: set[tuple[int, int]] = set()
        previous_by_layer: dict[int, int] = {}
        tokens = {(row.call_index, row.row_index) for row in heldout_rows}
        for row in heldout_rows:
            key = (row.layer_id, row.expert_id)
            layer_cache = cache[row.layer_id]
            if row.expert_id in layer_cache:
                hits += 1
                layer_cache.move_to_end(row.expert_id)
                if key in prefetched:
                    useful += self._bytes(key)
                    prefetched.discard(key)
            else:
                misses += 1
                bytes_read += self._bytes(key)
                self._insert(layer_cache, row.layer_id, row.expert_id, pinned)
                churn += 1
            predicted = self._prediction(policy, row, previous_by_layer, frequency, transitions)
            previous_by_layer[row.layer_id] = row.expert_id
            if predicted is not None and predicted not in cache[row.layer_id]:
                predicted_key = (row.layer_id, predicted)
                prefetched.add(predicted_key)
                self._insert(cache[row.layer_id], row.layer_id, predicted, pinned)
        wasted = sum(self._bytes(key) for key in prefetched)
        count = len(heldout_rows)
        token_count = len(tokens)
        return PolicyEvaluation(
            policy=policy,
            partition="heldout",
            measured=measured_metrics is not None,
            selection_count=count,
            token_count=token_count,
            expert_hits=hits,
            expert_misses=misses,
            expert_hit_rate=hits / count if count else None,
            bytes_read=bytes_read,
            bytes_read_per_token=bytes_read / token_count if token_count else None,
            tier_churn=churn,
            prefetch_useful_bytes=useful,
            prefetch_wasted_bytes=wasted,
            decode_tokens_per_second=(measured_metrics or {}).get("decode_tokens_per_second"),
            time_to_first_token_ms=(measured_metrics or {}).get("time_to_first_token_ms"),
            expert_load_latency_ms=(measured_metrics or {}).get("expert_load_latency_ms"),
        )

    def evaluate_matrix(
        self,
        *,
        calibration: Iterable[RouteSelection],
        heldout: Iterable[RouteSelection],
        policies: Iterable[PlacementPolicy],
        measured: dict[PlacementPolicy, dict[str, float]] | None = None,
    ) -> list[PolicyEvaluation]:
        calibration_rows, heldout_rows = list(calibration), list(heldout)
        results = [
            self.evaluate(
                policy=policy,
                calibration=calibration_rows,
                heldout=heldout_rows,
                measured_metrics=(measured or {}).get(policy),
            )
            for policy in policies
        ]
        measured_speeds = [
            row.decode_tokens_per_second
            for row in results
            if row.decode_tokens_per_second is not None
        ]
        if measured_speeds:
            best = max(measured_speeds)
            for row in results:
                if row.decode_tokens_per_second is not None:
                    row.heldout_regret = best / row.decode_tokens_per_second - 1.0
                    if (
                        row.prefetch_wasted_bytes > row.prefetch_useful_bytes
                        and row.heldout_regret > 0
                    ):
                        row.rejected = True
                        row.rejection_reason = (
                            "prefetch wasted more bytes and lost held-out throughput"
                        )
        return results

    def _bytes(self, key: tuple[int, int]) -> int:
        if key not in self.expert_bytes:
            raise ValueError(f"missing byte size for routed expert {key}")
        return self.expert_bytes[key]

    def _pinned(
        self,
        policy: PlacementPolicy,
        frequency: Counter[tuple[int, int]],
    ) -> dict[int, set[int]]:
        if policy == PlacementPolicy.PLAIN_LRU or self.hot_slots_per_layer == 0:
            return defaultdict(set)
        by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for (layer, expert), count in frequency.items():
            by_layer[layer].append((expert, count))
        return {
            layer: {
                expert
                for expert, _ in sorted(values, key=lambda item: (-item[1], item[0]))[
                    : self.hot_slots_per_layer
                ]
            }
            for layer, values in by_layer.items()
        }

    def _insert(
        self,
        cache: OrderedDict[int, None],
        layer: int,
        expert: int,
        pinned: dict[int, set[int]],
    ) -> bool:
        if expert in cache:
            cache.move_to_end(expert)
            return False
        while len(cache) >= self.cache_slots_per_layer:
            victim = next(
                (candidate for candidate in cache if candidate not in pinned[layer]), None
            )
            if victim is None:
                return False
            del cache[victim]
        cache[expert] = None
        return True

    @staticmethod
    def _transition_counts(
        rows: list[RouteSelection],
    ) -> Counter[tuple[int, int, int]]:
        result: Counter[tuple[int, int, int]] = Counter()
        last: dict[int, int] = {}
        for row in sorted(rows, key=lambda item: (item.call_index, item.row_index, item.layer_id)):
            if row.layer_id in last:
                result[(row.layer_id, last[row.layer_id], row.expert_id)] += 1
            last[row.layer_id] = row.expert_id
        return result

    @staticmethod
    def _prediction(
        policy: PlacementPolicy,
        row: RouteSelection,
        previous: dict[int, int],
        frequency: Counter[tuple[int, int]],
        transitions: Counter[tuple[int, int, int]],
    ) -> int | None:
        if policy == PlacementPolicy.RECENT_REUSE:
            return previous.get(row.layer_id)
        if policy in {PlacementPolicy.TRANSITION, PlacementPolicy.SWARM}:
            prior = previous.get(row.layer_id)
            if prior is not None:
                choices = [
                    (target, count)
                    for (layer, source, target), count in transitions.items()
                    if layer == row.layer_id and source == prior
                ]
                if choices:
                    return max(choices, key=lambda item: (item[1], -item[0]))[0]
        if policy in {PlacementPolicy.FREQUENCY, PlacementPolicy.COLIBRI_RECOMMENDED}:
            choices = [
                (expert, count)
                for (layer, expert), count in frequency.items()
                if layer == row.layer_id
            ]
            if choices:
                return max(choices, key=lambda item: (item[1], -item[0]))[0]
        return None


def batch_expert_union(
    selections: Iterable[RouteSelection],
    *,
    expert_bytes: dict[tuple[int, int], int],
    prefill_duration_ms: float | None = None,
) -> dict[str, Any]:
    rows = list(selections)
    raw = [(row.layer_id, row.expert_id) for row in rows]
    unique = sorted(set(raw))
    before = sum(expert_bytes[key] for key in raw)
    after = sum(expert_bytes[key] for key in unique)
    return {
        "raw_expert_selections": len(raw),
        "unique_expert_selections": len(unique),
        "deduplication_ratio": 1.0 - len(unique) / len(raw) if raw else None,
        "bytes_before_union": before,
        "bytes_after_union": after,
        "bytes_avoided": before - after,
        "prefill_duration_ms": prefill_duration_ms,
        "token_semantics_reordered": False,
    }
