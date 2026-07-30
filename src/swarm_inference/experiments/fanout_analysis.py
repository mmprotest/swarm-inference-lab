"""Sweep, stability, resume, and optimum logic for Experiment 003."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable, Mapping
from typing import Any

COUNT_POINT_PHASES = frozenset(
    {
        "cached_cold_load_only",
        "cold_no_stage_warmup",
        "cold_with_stage_warmup",
        "warm",
        "hot_standby",
    }
)


def initial_sweep_order(initial_counts: Iterable[int], maximum_worker_count: int) -> list[int]:
    if maximum_worker_count < 1:
        raise ValueError("maximum_worker_count must be positive")
    result: list[int] = []
    for value in initial_counts:
        count = int(value)
        if not 1 <= count <= maximum_worker_count:
            raise ValueError(f"worker count {count} is outside 1..{maximum_worker_count}")
        if count not in result:
            result.append(count)
    return result


def next_adaptive_count(last_success: int, first_failure: int) -> int | None:
    """Return the upper midpoint until the exact integer boundary is known."""

    if last_success < 0 or first_failure <= last_success:
        raise ValueError("adaptive search bounds are invalid")
    if first_failure - last_success <= 1:
        return None
    return (last_success + first_failure + 1) // 2


def adaptive_search_summary(
    *,
    initial_worker_counts: Iterable[int],
    adaptive_search_enabled: bool,
    attempts: Iterable[Mapping[str, Any]],
    maximum_worker_count: int,
) -> dict[str, Any]:
    """Summarise whether the exact runnable boundary has been established."""

    rows = [dict(row) for row in attempts]
    successes = [int(row["worker_count"]) for row in rows if bool(row.get("runnable"))]
    failures = [int(row["worker_count"]) for row in rows if not bool(row.get("runnable"))]
    last_success = max(successes, default=0)
    first_failure = min(
        (count for count in failures if count > last_success),
        default=None,
    )
    return {
        "initial_worker_counts": [int(value) for value in initial_worker_counts],
        "adaptive_search_enabled": bool(adaptive_search_enabled),
        "attempts": rows,
        "last_success": last_success or None,
        "first_failure": first_failure,
        "exact_boundary_identified": (
            last_success == int(maximum_worker_count) or first_failure == last_success + 1
        ),
        "maximum_runnable_worker_count": last_success,
    }


def coefficient_of_variation(values: Iterable[float]) -> float | None:
    samples = [float(value) for value in values]
    if not samples:
        return None
    mean = statistics.mean(samples)
    if mean == 0:
        return 0.0 if all(value == 0 for value in samples) else None
    return statistics.pstdev(samples) / mean


def count_is_runnable(session_rows: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        bool(row.get("runnable_generation"))
        and bool(row.get("exact_token_identity"))
        and bool(row.get("direct_data_plane"))
        and bool(row.get("clean_shutdown"))
        for row in session_rows
    )


def count_is_stable(
    session_rows: Iterable[Mapping[str, Any]],
    *,
    repeats: int,
    max_gpu_memory_fraction: float,
    max_system_memory_fraction: float,
    warm_throughput_cv_limit: float = 0.10,
) -> bool:
    rows = list(session_rows)
    warm = [row for row in rows if row.get("phase") == "warm"]
    if len(warm) != repeats:
        return False
    if any(
        not bool(row.get("passed"))
        or not bool(row.get("exact_token_identity"))
        or not bool(row.get("direct_data_plane"))
        or not bool(row.get("clean_shutdown"))
        or bool(row.get("worker_crash"))
        or bool(row.get("oom"))
        or bool(row.get("request_timeout"))
        or bool(row.get("stale_cache"))
        for row in rows
    ):
        return False
    if any(
        row.get("peak_gpu_memory_fraction") is None
        or float(row["peak_gpu_memory_fraction"]) > max_gpu_memory_fraction
        or row.get("peak_system_memory_fraction") is None
        or float(row["peak_system_memory_fraction"]) > max_system_memory_fraction
        for row in warm
    ):
        return False
    throughputs = [
        float(row["warm_output_tokens_per_second"])
        for row in warm
        if row.get("warm_output_tokens_per_second") is not None
    ]
    cv = coefficient_of_variation(throughputs)
    return len(throughputs) == repeats and cv is not None and cv <= warm_throughput_cv_limit


def performance_optima(
    count_rows: Iterable[Mapping[str, Any]],
) -> tuple[int, int]:
    eligible = [row for row in count_rows if bool(row.get("stable"))]
    if not eligible:
        return 0, 0
    latency = min(
        eligible,
        key=lambda row: (
            float(row.get("median_warm_end_to_end_seconds", float("inf"))),
            int(row["worker_count"]),
        ),
    )
    throughput = max(
        eligible,
        key=lambda row: (
            float(row.get("median_concurrency_4_verified_tps", float("-inf"))),
            -int(row["worker_count"]),
        ),
    )
    return int(latency["worker_count"]), int(throughput["worker_count"])


def config_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def session_evidence_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        [dict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def valid_completed_counts(
    state: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> set[int]:
    if state.get("config_fingerprint") != expected_fingerprint:
        return set()
    completed = state.get("completed_counts")
    if not isinstance(completed, dict):
        return set()
    session_rows = state.get("session_rows")
    count_rows = state.get("count_rows")
    if not isinstance(session_rows, list) or not isinstance(count_rows, dict):
        return set()
    point_rows = [
        row
        for row in session_rows
        if isinstance(row, dict) and row.get("phase") in COUNT_POINT_PHASES
    ]
    return {
        int(count)
        for count, evidence in completed.items()
        if isinstance(evidence, dict)
        and evidence.get("complete") is True
        and evidence.get("evidence_schema") == "experiment-003-count-v1"
        and str(count) in count_rows
        and int(evidence.get("session_count", -1))
        == sum(1 for row in point_rows if int(row.get("worker_count", -1)) == int(count))
        and evidence.get("session_evidence_sha256")
        == session_evidence_digest(
            row for row in point_rows if int(row.get("worker_count", -1)) == int(count)
        )
    }


def validate_rejoin_evidence(evidence: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required_true = (
        "worker_process_genuinely_terminated",
        "new_pid",
        "route_generation_incremented",
        "cache_replay_occurred",
        "exact_token_identity",
    )
    for field in required_true:
        if evidence.get(field) is not True:
            errors.append(f"rejoin evidence requires {field}=true")
    if evidence.get("old_process_id") == evidence.get("new_process_id"):
        errors.append("replacement process must have a new PID")
    if int(evidence.get("route_generation_after", 0)) <= int(
        evidence.get("route_generation_before", 0)
    ):
        errors.append("route generation did not increase")
    if float(evidence.get("cache_replay_seconds", 0)) <= 0:
        errors.append("cache replay duration was not positive")
    return errors
