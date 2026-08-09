from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm_inference.experiments.experiment_012.baseline_harness import (
    build_local_scheduler_topology,
    run_baselines,
)
from swarm_inference.microworker_protocol import (
    NETWORK_PROFILES,
    aggregate_digest,
    combine_aggregates,
    leaf_aggregate,
    make_leaf_request,
    validate_operation_envelope,
    worker_contribution,
)


def test_exact_aggregate_proof_is_order_independent_and_duplicate_sensitive() -> None:
    leaves = [
        leaf_aggregate(f"worker-{index:06d}", worker_contribution(index)) for index in range(8)
    ]
    forward = combine_aggregates(leaves)
    reverse = combine_aggregates(list(reversed(leaves)))
    duplicate = combine_aggregates([*leaves, leaves[0]])

    assert forward == reverse
    assert aggregate_digest(forward) == aggregate_digest(reverse)
    assert duplicate != forward
    assert duplicate["contribution_count"] == 9


def test_leaf_envelope_carries_predeclared_operation_identity() -> None:
    request = make_leaf_request(
        request_id="request-1",
        operation_id="operation-1",
        execution_generation=1,
        parent_worker="stage-owner",
        worker_id="worker-000000",
        worker_index=0,
        deadline_unix_ns=time.time_ns() + 10_000_000_000,
        route_lease_id="lease-1",
        ordering_key="worker-000000",
        payload_bytes=16,
        trace_id="trace-1",
        span_id="span-1",
    )

    validate_operation_envelope(request, allow_children=False)
    assert request["assigned_child_workers"] == []
    assert request["work_partition"]["partition_start"] == 0
    assert request["aggregation"]["mode"] == "exact_int64_sum"
    assert request["retry_policy"]["max_attempts"] == 0
    assert request["cancellation"]["cancelled"] is False

    request["assigned_child_workers"] = [{"worker_id": "worker-000001"}]
    with pytest.raises(ValueError, match="cannot carry delegated children"):
        validate_operation_envelope(request, allow_children=False)


def test_local_scheduler_topology_does_not_change_runtime_network_ownership() -> None:
    workers = [
        SimpleNamespace(worker_id=f"worker-{index:06d}", endpoint=f"127.0.0.1:{9000 + index}")
        for index in range(1000)
    ]
    topology = build_local_scheduler_topology(workers, branch_factor=8)  # type: ignore[arg-type]

    assert len(topology.root_children) <= 8
    assert topology.depth == 4
    assert topology.node_count > len(workers)


def test_two_process_baseline_uses_real_endpoints_and_retains_raw_rows(tmp_path: Path) -> None:
    output = tmp_path / "experiment-012-baseline-smoke"
    summary = run_baselines(
        output_directory=output,
        worker_counts=(2,),
        branch_factor=8,
        maximum_root_concurrency=2,
        payload_bytes=32,
        warmup_trials=0,
        measured_trials=1,
        network_profile=NETWORK_PROFILES["same_host_shaped"],
        operation_deadline_s=10.0,
        startup_deadline_s=30.0,
    )

    assert summary["failed_trials"] == []
    assert summary["attempts"][0]["status"] == "completed"
    assert summary["attempts"][0]["started_processes"] == 2
    assert len(summary["summaries"]) == 2
    for row in summary["summaries"]:
        assert row["root_messages_median"] == 4
        assert row["root_direct_degree_median"] == 2
        assert row["root_serial_waits_median"] == 2
    identities = list((output / "raw" / "workers-0002" / "processes").glob("*/ready.json"))
    assert len(identities) == 2
    assert (output / "raw" / "trials.jsonl").is_file()
    assert (output / "traces" / "root.jsonl").is_file()
