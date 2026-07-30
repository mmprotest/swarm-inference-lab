from __future__ import annotations

from swarm_inference.experiments.fanout_analysis import (
    adaptive_search_summary,
    count_is_runnable,
    count_is_stable,
    initial_sweep_order,
    next_adaptive_count,
    performance_optima,
    session_evidence_digest,
    valid_completed_counts,
    validate_rejoin_evidence,
)


def _session(phase: str, throughput: float, *, passed: bool = True) -> dict[str, object]:
    return {
        "phase": phase,
        "passed": passed,
        "runnable_generation": phase == "warm" and passed,
        "exact_token_identity": passed,
        "direct_data_plane": passed,
        "clean_shutdown": passed,
        "worker_crash": False,
        "oom": False,
        "request_timeout": False,
        "stale_cache": False,
        "peak_gpu_memory_fraction": 0.8,
        "peak_system_memory_fraction": 0.5,
        "warm_output_tokens_per_second": throughput if phase == "warm" else None,
    }


def test_initial_and_adaptive_search_identify_exact_integer_maximum() -> None:
    assert initial_sweep_order([1, 2, 4, 7, 14, 21, 28], 28) == [
        1,
        2,
        4,
        7,
        14,
        21,
        28,
    ]
    lower, upper = 21, 28
    attempts = []
    while (candidate := next_adaptive_count(lower, upper)) is not None:
        attempts.append(candidate)
        if candidate <= 24:
            lower = candidate
        else:
            upper = candidate
    assert lower == 24
    assert upper == 25
    assert attempts


def test_adaptive_search_summary_treats_semantic_maximum_as_exact_boundary() -> None:
    attempts = [{"worker_count": count, "runnable": True} for count in (1, 2, 4, 7, 14, 21, 28)]
    summary = adaptive_search_summary(
        initial_worker_counts=(1, 2, 4, 7, 14, 21, 28),
        adaptive_search_enabled=True,
        attempts=attempts,
        maximum_worker_count=28,
    )
    assert summary["last_success"] == 28
    assert summary["first_failure"] is None
    assert summary["maximum_runnable_worker_count"] == 28
    assert summary["exact_boundary_identified"] is True


def test_runnable_and_stable_can_differ_and_require_three_repeats() -> None:
    rows = [
        _session("cached_cold_load_only", 0),
        _session("cold_no_stage_warmup", 0),
        _session("cold_with_stage_warmup", 0),
        _session("hot_standby", 0),
        _session("warm", 10.0),
        _session("warm", 10.5),
        _session("warm", 9.5),
    ]
    assert count_is_runnable(rows)
    assert count_is_stable(
        rows,
        repeats=3,
        max_gpu_memory_fraction=0.95,
        max_system_memory_fraction=0.90,
    )
    rows[-1]["peak_gpu_memory_fraction"] = 0.97
    assert not count_is_stable(
        rows,
        repeats=3,
        max_gpu_memory_fraction=0.95,
        max_system_memory_fraction=0.90,
    )
    assert count_is_runnable(rows)


def test_resume_skips_only_compatible_complete_points() -> None:
    sessions = [
        {"worker_count": 4, "phase": "warm", "passed": True},
        {"worker_count": 7, "phase": "warm", "passed": True},
        {"worker_count": 14, "phase": "warm", "passed": True},
        {
            "worker_count": 4,
            "phase": "unprovisioned_acquisition",
            "passed": True,
        },
    ]
    state = {
        "config_fingerprint": "expected",
        "session_rows": sessions,
        "count_rows": {"4": {}, "7": {}, "14": {}},
        "completed_counts": {
            "4": {
                "complete": True,
                "evidence_schema": "experiment-003-count-v1",
                "session_count": 1,
                "session_evidence_sha256": session_evidence_digest([sessions[0]]),
            },
            "7": {
                "complete": False,
                "evidence_schema": "experiment-003-count-v1",
                "session_count": 1,
                "session_evidence_sha256": session_evidence_digest([sessions[1]]),
            },
            "14": {
                "complete": True,
                "evidence_schema": "old",
                "session_count": 1,
                "session_evidence_sha256": session_evidence_digest([sessions[2]]),
            },
        },
    }
    assert valid_completed_counts(state, expected_fingerprint="expected") == {4}
    assert valid_completed_counts(state, expected_fingerprint="different") == set()


def test_performance_optima_only_consider_stable_counts() -> None:
    counts = [
        {
            "worker_count": 1,
            "stable": True,
            "median_warm_end_to_end_seconds": 1.0,
            "median_concurrency_4_verified_tps": 20.0,
        },
        {
            "worker_count": 2,
            "stable": True,
            "median_warm_end_to_end_seconds": 1.2,
            "median_concurrency_4_verified_tps": 25.0,
        },
        {
            "worker_count": 28,
            "stable": False,
            "median_warm_end_to_end_seconds": 0.1,
            "median_concurrency_4_verified_tps": 100.0,
        },
    ]
    assert performance_optima(counts) == (1, 2)


def test_rejoin_evidence_is_fail_closed() -> None:
    passing = {
        "worker_process_genuinely_terminated": True,
        "old_process_id": 100,
        "new_process_id": 200,
        "new_pid": True,
        "route_generation_before": 1,
        "route_generation_after": 2,
        "route_generation_incremented": True,
        "cache_replay_occurred": True,
        "cache_replay_seconds": 0.2,
        "exact_token_identity": True,
    }
    assert validate_rejoin_evidence(passing) == []
    passing["new_process_id"] = 100
    assert validate_rejoin_evidence(passing)
