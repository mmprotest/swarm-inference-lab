from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.real_path_resilience import (
    CORRUPTION_KINDS,
    FAILURE_KINDS,
    _summarize_corruption,
    _summarize_failure,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _failure_summary(tmp_path: Path, kind: str) -> dict[str, Any]:
    run_root = tmp_path / kind
    _write_jsonl(
        run_root / "workers" / "worker-0" / "worker-telemetry.jsonl",
        [
            {
                "event": "native_expert_fault_injected",
                "request_id": "request-1",
                "wall_time_ns": 100,
            },
            {
                "event": "native_expert_request_completed",
                "request_id": "request-1",
                "duration_ns": 70,
            },
        ],
    )
    _write_jsonl(
        run_root / "coordinator-telemetry.jsonl",
        [
            {
                "event": "expert_rpc_timeout",
                "request_id": "request-1",
                "wall_time_ns": 130,
            },
            {
                "event": "expert_rpc_fallback",
                "request_id": "request-1",
                "wall_time_ns": 140,
                "expert_ids": [7],
                "token_position": 3,
            },
            {
                "event": "expert_rpc_recovery_completed",
                "request_id": "request-1",
                "wall_time_ns": 180,
            },
            {
                "event": "expert_rpc_bytes_sent",
                "request_id": "request-1",
                "byte_count": 80,
            },
            {
                "event": "expert_rpc_bytes_received",
                "request_id": "request-1",
                "byte_count": 120,
            },
            {
                "event": "expert_rpc_request_completed",
                "request_id": "request-1",
                "duration_ns": 90,
            },
        ],
    )
    scenario = {
        "name": f"{kind}-recovery",
        "fault": {"kind": kind},
        "strategy": "timeout_alternate_worker",
        "expected": "recover",
        "baseline_rpc_p95_ns": 50,
        "baseline_network_bytes": 150,
        "baseline_worker_execution_wall_ns_proxy": 40,
    }
    result = {
        "return_code": 0,
        "exact_token_identity": True,
        "matching_tokens": 32,
        "expected_tokens": 32,
        "actual_token_ids": list(range(32)),
        "expected_token_ids": list(range(32)),
        "forbidden_local_loads": 0,
        "elapsed_ns": 1_000,
    }
    row, _ = _summarize_failure(scenario, result, run_root)
    return row


def _corruption_summary(tmp_path: Path, kind: str) -> dict[str, Any]:
    run_root = tmp_path / kind
    _write_jsonl(
        run_root / "workers" / "worker-0" / "worker-telemetry.jsonl",
        [
            {
                "event": "native_expert_corruption_injected",
                "request_id": "injected",
                "wall_time_ns": 100,
            }
        ],
    )
    _write_jsonl(
        run_root / "coordinator-telemetry.jsonl",
        [
            {
                "event": "expert_rpc_corruption_detected",
                "request_id": "injected",
                "wall_time_ns": 150,
            },
            {
                "event": "expert_rpc_sampled_duplicate_passed",
                "request_id": "clean-control",
                "wall_time_ns": 175,
                "duration_ns": 20,
                "byte_count": 64,
            },
            {
                "event": "expert_rpc_request_completed",
                "request_id": "injected",
                "duration_ns": 80,
            },
        ],
    )
    result = {
        "elapsed_ns": 900,
        "exact_token_identity": True,
        "return_code": 0,
        "forbidden_local_loads": 0,
        "worker_quarantines": 1,
    }
    row, _ = _summarize_corruption(
        kind,
        result,
        run_root,
        1_000,
        {
            "rpc_p95_ns": 60,
            "network_bytes": 0,
            "worker_execution_wall_ns_proxy": 0,
        },
    )
    return row


def test_real_path_worker_termination_recovery(tmp_path: Path) -> None:
    row = _failure_summary(tmp_path, "worker_termination")
    assert row["passed"] is True
    assert row["failure_detection_latency_ns"] == 30
    assert row["recovery_latency_ns"] == 80


def test_real_path_worker_delay_recovery(tmp_path: Path) -> None:
    assert "fixed_delay" in FAILURE_KINDS
    assert "random_delay" in FAILURE_KINDS
    assert _failure_summary(tmp_path, "fixed_delay")["exact_token_identity"] is True


def test_real_path_network_outage_recovery(tmp_path: Path) -> None:
    row = _failure_summary(tmp_path, "network_outage")
    assert row["fallback_count"] == 1
    assert row["extra_network_bytes"] == 50
    assert row["extra_compute_ns_proxy"] == 30
    assert row["p95_impact_ns"] == 40


def test_real_path_bit_flip_detection(tmp_path: Path) -> None:
    row = _corruption_summary(tmp_path, "bit_flip")
    assert row["detection_rate"] == 1.0
    assert row["exact_token_identity"] is True


def test_real_path_zero_result_detection(tmp_path: Path) -> None:
    assert _corruption_summary(tmp_path, "zero_result")["detected_count"] == 1


def test_real_path_stale_result_detection(tmp_path: Path) -> None:
    assert _corruption_summary(tmp_path, "stale_result")["detected_count"] == 1


def test_real_path_wrong_revision_detection(tmp_path: Path) -> None:
    row = _corruption_summary(tmp_path, "wrong_model_revision")
    assert row["detected_count"] == 1
    assert row["quarantine_count"] == 1


def test_real_path_clean_false_positive_rate(tmp_path: Path) -> None:
    row = _corruption_summary(tmp_path, "plausible_random_perturbation")
    assert set(CORRUPTION_KINDS) >= {
        "wrong_expert_id",
        "wrong_layer_id",
        "lower_precision_result",
    }
    assert row["clean_control_count"] == 1
    assert row["false_positive_count"] == 0
    assert row["false_positive_rate"] == 0.0
