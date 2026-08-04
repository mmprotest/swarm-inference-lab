from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from swarm_inference.experiments.experiment_010 import real_path_simulator as simulator


def test_simulator_behavioral_cache_parity() -> None:
    cache = simulator.CacheState(budget_bytes=20, max_entries=2)
    cache.access((0, 1, None, None), 10)
    cache.access((0, 2, None, None), 10)
    cache.access((0, 1, None, None), 10)
    cache.access((0, 3, None, None), 10)

    assert (cache.hits, cache.misses, cache.evictions) == (1, 3, 1)
    assert tuple(cache.entries) == ((0, 1, None, None), (0, 3, None, None))


def test_simulator_behavioral_message_parity(tmp_path: Path) -> None:
    trace = tmp_path / "route.trace"
    trace.write_text(
        "0 0 0 1:0.6 2:0.4\n1 0 1 2:0.7 1:0.3\n",
        encoding="utf-8",
    )
    plan = {
        "workers": [
            {
                "worker_id": "left",
                "owned_experts": [
                    {"layer_id": 0, "expert_id": 1},
                    {"layer_id": 1, "expert_id": 1},
                ],
                "owned_microshards": [],
            },
            {
                "worker_id": "right",
                "owned_experts": [
                    {"layer_id": 0, "expert_id": 2},
                    {"layer_id": 1, "expert_id": 2},
                ],
                "owned_microshards": [],
            },
        ]
    }

    groups, selections = simulator._route_groups(trace, plan)

    assert selections == 4
    assert len(groups) == 2
    assert groups[0][2] == {"left": [1], "right": [2]}
    assert groups[1][2] == {"right": [2], "left": [1]}


def _measured_timing_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phases = ["decode"] * 12 + ["concurrent_decode"] * 6 + ["mixed_service"] * 5
    for index, phase in enumerate(phases):
        parallel = phase != "decode"
        concurrency = 1 if not parallel else 2 + index % 5
        micro = int(index % 2 == 0)
        prompt_tokens = 11.0
        output_tokens = 128.0
        verified_tokens = output_tokens * concurrency if parallel else output_tokens
        transport_ns = (8.0 + index) * 1e9
        if parallel:
            base_ns = 30e9 + 3e9 * concurrency - 4e9 * micro
        else:
            base_ns = 1e9 + 1e8 * output_tokens + 2e7 * prompt_tokens
        total_ns = transport_ns + base_ns
        compute_ns = total_ns * (2 if micro else 4) * 0.25
        rows.append(
            {
                "configuration_id": f"configuration-{index:02d}",
                "workload_id": "shared-ranking-workload",
                "phase": phase,
                "configuration": f"candidate-{index:02d}",
                "network_profile": f"profile-{index % 4}",
                "data_plane": "direct_tcp",
                "response_mode": "per_expert_exact",
                "shard_layout": "equal" if micro else "whole",
                "worker_subset": ("left", "right") if micro else ("w0", "w1", "w2", "w3"),
                "worker_count": 2 if micro else 4,
                "concurrency": concurrency,
                "is_microshard": micro,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "verified_tokens": verified_tokens,
                "critical_transport_ns": transport_ns,
                "first_token_transport_ns": 1e9 + index * 1e7,
                "rpc_compute_ns": compute_ns,
                "rpc_queue_ns": index * 1000.0,
                "rpc_raw_payload_bytes": 1_000_000.0 + index,
                "rpc_message_count": 100.0 + index,
                "measured_total_ns": total_ns,
                "measured_throughput": verified_tokens * 1e9 / total_ns,
                "measured_p95_ns": total_ns * (1.01 if parallel else 1.0),
                "measured_ttft_ns": 1e9 + index * 1e7 + 0.2e9 + 2e7 * prompt_tokens,
                "measured_network_bytes": 1_000_000.0 + index,
                "measured_queue_ns": index * 1000.0,
                "measured_worker_utilization": compute_ns / ((2 if micro else 4) * total_ns),
                "sample_count": 3,
                "run_ids": [f"run-{index}"],
                "source_paths": [f"measurement-{index}.json"],
                "evidence_category": "REAL_MODEL_MEASURED",
            }
        )
    prompt_tokens = 8192.0
    output_tokens = 64.0
    transport_ns = 900e9
    base_ns = 1e9 + 1e8 * output_tokens + 2e7 * prompt_tokens
    total_ns = transport_ns + base_ns
    rows.append(
        {
            **rows[0],
            "configuration_id": "prefill-singleton",
            "workload_id": "prefill-8192",
            "phase": "prefill",
            "configuration": "whole-prefill",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "verified_tokens": output_tokens,
            "critical_transport_ns": transport_ns,
            "first_token_transport_ns": 890e9,
            "rpc_compute_ns": 300e9,
            "measured_total_ns": total_ns,
            "measured_throughput": output_tokens * 1e9 / total_ns,
            "measured_p95_ns": total_ns,
            "measured_ttft_ns": 890e9 + 0.2e9 + 2e7 * prompt_tokens,
            "measured_worker_utilization": 300e9 / (4 * total_ns),
        }
    )
    return rows


@pytest.fixture
def calibrated_simulator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    output = tmp_path / "simulator"
    output.mkdir()
    (output / "simulator_behavioral_parity.json").write_text(
        json.dumps({"all_exact": True, "configuration_count": 24}),
        encoding="utf-8",
    )
    monkeypatch.setattr(simulator, "_load_timing_rows", lambda _path: _measured_timing_rows())
    monkeypatch.setattr(
        simulator,
        "_recovery_validation",
        lambda _path: (
            {"held_out_error_fraction": 0.01},
            [
                {
                    "row_type": "failure_recovery",
                    "configuration_id": "recovery-test",
                    "recovery_error_fraction": 0.01,
                }
            ],
        ),
    )
    return simulator.calibrate_real_path_timing(
        tmp_path / "analysis", tmp_path / "failures.csv", output
    )


def test_simulator_heldout_throughput_error(
    calibrated_simulator: dict[str, Any],
) -> None:
    validation = calibrated_simulator["validation"]
    assert validation["median_throughput_error_fraction"] <= 0.10
    assert calibrated_simulator["split"]["calibration_count"] == 17
    assert calibrated_simulator["split"]["validation_count"] == 7


def test_simulator_heldout_p95_error(calibrated_simulator: dict[str, Any]) -> None:
    assert calibrated_simulator["validation"]["p95_latency_error_fraction"] <= 0.15
    assert calibrated_simulator["validation"]["median_ttft_error_fraction"] <= 0.15


def test_simulator_heldout_ranking(calibrated_simulator: dict[str, Any]) -> None:
    validation = calibrated_simulator["validation"]
    assert validation["plan_ranking_agreement_fraction"] >= 0.80
    assert validation["planner_regret_fraction"] <= 0.05
    assert validation["all_gates_pass"] is True
