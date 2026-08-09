from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from swarm_inference.experiments.experiment_012.branch_factor_harness import run_h012_005
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DELEGATED_MODE,
    build_capacity_parent_topology,
    build_delegated_topology,
    run_h012_001,
)
from swarm_inference.experiments.experiment_012.parallel_harness import run_h012_002
from swarm_inference.experiments.experiment_012.topology_planner import (
    calibrate_from_summary,
    choose_branch_factor,
)
from swarm_inference.microworker_protocol import NETWORK_PROFILES


def _fake_workers(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            worker_id=f"worker-{index:06d}",
            worker_index=index,
            process_id=10_000 + index,
            endpoint=f"127.0.0.1:{20_000 + index}",
        )
        for index in range(count)
    ]


def test_delegated_topology_is_bounded_and_uses_every_worker_once() -> None:
    for count in (2, 8, 32, 128, 512, 1000):
        topology = build_delegated_topology(  # type: ignore[arg-type]
            _fake_workers(count),
            branch_factor=8,
            topology_id=f"topology-{count}",
            route_lease_id=f"lease-{count}",
        )

        assert len(topology.nodes) == count
        assert len({node.worker.worker_id for node in topology.nodes}) == count
        assert len(topology.root_children) <= 8
        assert all(node.children for node in topology.root_children)
        assert all(len(node.children) <= 8 for node in topology.nodes)
        assert topology.depth <= 2 + math.ceil(math.log(count, 8))


def test_two_process_delegation_is_real_and_exact(tmp_path: Path) -> None:
    output = tmp_path / "h012-001-smoke"
    output.mkdir(parents=True)
    (output / "hypothesis.json").write_text(
        json.dumps({"hypothesis_id": "H012-001", "criteria_locked": True}),
        encoding="utf-8",
    )
    summary = run_h012_001(
        output_directory=output,
        worker_counts=(2,),
        profiles=("same_host_shaped",),
        branch_factor=8,
        maximum_root_concurrency=8,
        payload_bytes=32,
        warmup_trials=0,
        measured_trials=1,
        operation_deadline_s=10.0,
        startup_deadline_s=30.0,
    )

    assert summary["failed_trials"] == []
    assert summary["attempts"][0]["status"] == "completed"
    delegated = next(row for row in summary["summaries"] if row["mode"] == DELEGATED_MODE)
    assert delegated["root_messages_median"] == 2
    assert delegated["root_direct_degree_median"] == 1
    assert delegated["root_leaf_rpc_count_median"] == 0
    assert delegated["hierarchy_depth_median"] == 2

    worker_events = []
    for path in (output / "raw" / "workers-0002" / "processes").glob("*/trace.jsonl"):
        worker_events.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    child_edges = [event for event in worker_events if event["event"] == "worker_child_round_trip"]
    assert len(child_edges) == 1
    assert child_edges[0]["sender_worker_id"] != "stage-owner"
    assert child_edges[0]["sender_process_id"] != child_edges[0]["receiver_process_id"]


def test_parallel_worker_dispatch_shortens_the_observed_sync_path(tmp_path: Path) -> None:
    output = tmp_path / "h012-002-smoke"
    output.mkdir(parents=True)
    (output / "hypothesis.json").write_text(
        json.dumps({"hypothesis_id": "H012-002", "criteria_locked": True}),
        encoding="utf-8",
    )
    summary = run_h012_002(
        output_directory=output,
        worker_counts=(18,),
        profiles=("same_host_shaped",),
        branch_factor=8,
        maximum_root_concurrency=8,
        payload_bytes=32,
        warmup_trials=0,
        measured_trials=1,
        operation_deadline_s=10.0,
        startup_deadline_s=30.0,
    )

    assert summary["failed_trials"] == []
    by_mode = {row["mode"]: row for row in summary["summaries"]}
    serial = by_mode["delegated_serial"]
    parallel = by_mode["delegated_parallel"]
    assert serial["critical_path_sync_points_median"] == 3
    assert parallel["critical_path_sync_points_median"] == 2
    assert parallel["hierarchy_depth_median"] == 2
    assert parallel["root_leaf_rpc_count_median"] == 0
    assert parallel["total_messages_median"] == 36


def test_persistent_hierarchy_reuses_every_edge_after_warmup(tmp_path: Path) -> None:
    output = tmp_path / "h012-004-smoke"
    output.mkdir(parents=True)
    (output / "hypothesis.json").write_text(
        json.dumps({"hypothesis_id": "H012-004", "criteria_locked": True}),
        encoding="utf-8",
    )
    summary = run_h012_002(
        output_directory=output,
        worker_counts=(18,),
        profiles=("same_host_shaped",),
        branch_factor=8,
        maximum_root_concurrency=8,
        payload_bytes=32,
        warmup_trials=1,
        measured_trials=2,
        operation_deadline_s=10.0,
        startup_deadline_s=30.0,
        modes=("delegated_parallel", "delegated_parallel_persistent"),
        hypothesis_id="H012-004",
    )

    assert summary["failed_trials"] == []
    rows = [
        json.loads(line)
        for line in (output / "raw" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    persistent = [
        row for row in rows if row["mode"] == "delegated_parallel_persistent" and not row["warmup"]
    ]
    assert len(persistent) == 2
    assert all(row["correctness"] for row in persistent)
    assert all(row["root_connection_count"] == 0 for row in persistent)
    assert all(row["total_connection_count"] == 0 for row in persistent)
    assert summary["attempts"][0]["shutdown"]["graceful"] == 18


def test_branch_factor_harness_reinstalls_real_worker_topology(tmp_path: Path) -> None:
    output = tmp_path / "h012-005-smoke"
    output.mkdir(parents=True)
    (output / "hypothesis.json").write_text(
        json.dumps({"hypothesis_id": "H012-005", "criteria_locked": True}),
        encoding="utf-8",
    )
    summary = run_h012_005(
        output_directory=output,
        worker_counts=(18,),
        branch_factors=(8, 4),
        profiles=("same_host_shaped",),
        payload_bytes=32,
        warmup_trials=0,
        measured_trials=1,
        operation_deadline_s=10.0,
        startup_deadline_s=30.0,
    )

    assert summary["failed_trials"] == []
    assert {row["branch_factor"] for row in summary["summaries"]} == {4, 8}
    for row in summary["summaries"]:
        assert row["root_direct_degree_median"] <= row["branch_factor"]
        assert row["root_leaf_rpc_count_median"] == 0
        assert row["total_messages_median"] == 36
    assert summary["attempts"][0]["memory"]["observed_workers"] == 18


def test_evidence_planner_uses_only_declared_calibration_and_changes_choice(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "h012-006-summary.json"
    calibration_path.write_text(
        json.dumps(
            {
                "hypothesis_id": "H012-006",
                "summaries": [
                    {
                        "network_profile": "same_host_shaped",
                        "worker_count": 128,
                        "branch_factor": 8,
                        "end_to_end_latency_p50_ms": 58.2887,
                        "hierarchy_depth_median": 3,
                    },
                    {
                        "network_profile": "same_host_shaped",
                        "worker_count": 128,
                        "branch_factor": 32,
                        "end_to_end_latency_p50_ms": 67.8021,
                        "hierarchy_depth_median": 2,
                    },
                    {
                        "network_profile": "moderate_wan_shaped",
                        "worker_count": 128,
                        "branch_factor": 8,
                        "end_to_end_latency_p50_ms": 1.0,
                        "hierarchy_depth_median": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    calibration = calibrate_from_summary(calibration_path)
    low = choose_branch_factor(calibration, NETWORK_PROFILES["holdout_2ms_shaped"])
    high = choose_branch_factor(calibration, NETWORK_PROFILES["holdout_40ms_shaped"])

    assert low["selected_branch_factor"] == 8
    assert high["selected_branch_factor"] == 32
    assert calibration["calibration_profile"] == "same_host_shaped"
    assert all(candidate["activation_ms"] > 50 for candidate in calibration["candidates"])


def test_capacity_topology_reserves_faster_workers_for_parent_roles() -> None:
    workers = _fake_workers(18)
    for index, worker in enumerate(workers):
        worker.runtime_profile = {
            "capacity_score": 10.0 if index < 10 else 0.1,
            "compute_delay_ms": 0.1 if index < 10 else 20.0,
        }
    topology = build_capacity_parent_topology(  # type: ignore[arg-type]
        workers,
        branch_factor=4,
        topology_id="capacity-topology",
        route_lease_id="capacity-lease",
    )

    parents = [node for node in topology.nodes if node.children]
    leaves = [node for node in topology.nodes if not node.children]
    assert parents
    assert leaves
    assert min(node.worker.runtime_profile["capacity_score"] for node in parents) >= max(
        node.worker.runtime_profile["capacity_score"] for node in leaves
    )
    root_partitions = [set(node.partition_worker_indices) for node in topology.root_children]
    assert set.union(*root_partitions) == set(range(18))
    assert sum(len(partition) for partition in root_partitions) == 18
