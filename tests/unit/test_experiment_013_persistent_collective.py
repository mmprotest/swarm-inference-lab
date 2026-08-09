from __future__ import annotations

import json

import pytest

from swarm_inference.experiments.experiment_013.persistent_harness import run_benchmark
from swarm_inference.experiments.experiment_013.persistent_protocol import (
    LEAN_ARCHITECTURE,
    make_collective_request,
)
from swarm_inference.experiments.experiment_013.scaling_analysis import _fit
from swarm_inference.microworker_protocol import NETWORK_PROFILES


def test_lean_envelope_requires_root_proof_and_omits_static_topology() -> None:
    arguments = {
        "collective_id": "collective",
        "topology_id": "topology",
        "route_lease_id": "lease",
        "route_generation": 13,
        "request_id": "request",
        "operation_id": "operation",
        "execution_generation": 7,
        "parent_worker": "stage-owner",
        "target_worker": "worker-0000",
        "deadline_unix_ns": 10**20,
        "profile": NETWORK_PROFILES["same_host_shaped"],
        "retry_policy": {"max_attempts": 0, "backoff_ms": 0.0},
        "fault_control": {},
        "aggregation": {
            "mode": "exact_int64_sum",
            "ordering": "deterministic_worker_key",
        },
        "payload_b64": "AA==",
        "architecture": LEAN_ARCHITECTURE,
    }
    with pytest.raises(ValueError, match="operation digest"):
        make_collective_request(**arguments)

    message = make_collective_request(**arguments, operation_digest="proof")
    assert message["od"] == "proof"
    assert message["collective_id"] == "collective"
    assert "topology_id" not in message
    assert "target_worker" not in message
    assert "network_profile" not in message


def test_linear_candidate_fit_beats_n_log_n_for_linear_observations() -> None:
    counts = [2, 8, 32, 73, 128, 512, 1000]
    values = [2.5 + 0.015 * count for count in counts]
    linear = _fit("n", counts, values)
    n_log_n = _fit("n_log_n", counts, values)
    assert linear["rss"] < 1e-20
    assert linear["aicc"] < n_log_n["aicc"]


def test_live_collective_reuses_processes_routes_tasks_and_connections(tmp_path) -> None:
    output = tmp_path / "persistent-live"
    output.mkdir()
    (output / "hypothesis.json").write_text(
        json.dumps(
            {
                "hypothesis_id": "H013-TEST",
                "prediction": "warm operations reuse all installed machinery",
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark(
        output_directory=output,
        worker_counts=(8,),
        branch_factor=4,
        payload_bytes=32,
        profile=NETWORK_PROFILES["same_host_shaped"],
        warmup_trials=1,
        measured_trials=3,
        operation_deadline_s=10.0,
        startup_deadline_s=60.0,
        mailbox_depth=2,
        hypothesis_id="H013-TEST",
        architecture=LEAN_ARCHITECTURE,
    )

    assert result["all_scales_completed"] is True
    assert result["all_operations_correct"] is True
    summary = result["summaries"][0]
    assert summary["successful_trials"] == 3
    assert summary["root_leaf_rpc_count_median"] == 0
    assert summary["new_task_creation_median"] == 0
    assert summary["topology_rebuilds_median"] == 0
    assert summary["total_connection_count_median"] == 0
    assert summary["worker_activations_median"] == 0
    assert summary["total_messages_median"] == 16
