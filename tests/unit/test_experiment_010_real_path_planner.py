from __future__ import annotations

from typing import Any

from swarm_inference.experiments.experiment_010.real_path_planner import (
    CANDIDATE_SPECS,
    REQUIRED_CANDIDATES,
    _select_context,
)
from swarm_inference.experiments.experiment_010.schemas import PlannerObjective


def _context_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        local = spec.candidate_id == "local_monolithic"
        capacity = spec.candidate_id == "capacity_isolated"
        idle = spec.candidate_id == "idle"
        measured = local or capacity or spec.candidate_id in {
            "whole_expert_direct_tcp",
            "whole_expert_fast_response",
        }
        exact = spec.candidate_id != "whole_expert_fast_response"
        rows.append(
            {
                "schema_version": "test",
                "phase": "decode",
                "workload_variant": "short_128",
                "network_profile": "loopback_unshaped",
                "candidate_id": spec.candidate_id,
                "strategy": spec.strategy.value,
                "role": spec.role,
                "workers": list(spec.workers),
                "measurement_status": "MEASURED" if measured else "NOT_APPLICABLE",
                "eligible": bool(measured and exact),
                "evidence_category": "REAL_MODEL_MEASURED" if measured else None,
                "sample_count": 5 if measured else 0,
                "throughput": 5.0 if local else 2.0 if measured else None,
                "throughput_confidence_interval_95": (
                    (4.9, 5.1) if local else (1.9, 2.1) if measured else None
                ),
                "ttft_ms": 10.0 if local else 20.0 if measured else None,
                "ttft_confidence_interval_95": None,
                "p95_latency_ms": 20.0 if local else 60.0 if capacity else 30.0,
                "p95_latency_confidence_interval_95": None,
                "network_bytes": 0.0 if local else 100.0 if measured else None,
                "network_bytes_confidence_interval_95": None,
                "capacity_bytes": 1_000 if capacity else 0,
                "correctness_gate": bool(exact and (measured or idle)),
                "reliability_gate": bool(measured or idle),
                "source_paths": ["measured.csv"] if measured else [],
                "explanation": ["test measurement"],
            }
        )
    return rows


def test_real_path_planner_candidate_complete() -> None:
    catalog = {spec.candidate_id for spec in CANDIDATE_SPECS}
    assert set(REQUIRED_CANDIDATES) <= catalog
    assert {
        "background_inference",
        "verification_only",
        "local_fallback",
        "alternate_worker_recovery",
        "idle",
    } <= catalog


def test_real_path_planner_regret() -> None:
    plan, evaluated = _select_context(
        _context_rows(), PlannerObjective.MAX_DECODE_THROUGHPUT
    )
    assert plan["selected_candidate_id"] == "local_monolithic"
    assert plan["measured_regret"]["regret_fraction"] == 0.0
    assert plan["measured_regret"]["passes"] is True
    fast = next(row for row in evaluated if row["candidate_id"] == "whole_expert_fast_response")
    assert fast["correctness_gate"] is False


def test_real_path_planner_selects_capacity_when_required() -> None:
    plan, _ = _select_context(
        _context_rows(), PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY
    )
    assert plan["selected_candidate_id"] == "capacity_isolated"
    assert plan["capacity_exception"] is True
