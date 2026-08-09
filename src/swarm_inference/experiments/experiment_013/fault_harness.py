"""Exercise correctness and recovery in a live Experiment 013 collective."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _append_jsonl,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DelegatedNode,
    _control_round_trip,
    build_delegated_topology,
    topology_record,
)
from swarm_inference.experiments.experiment_013.persistent_harness import (
    PersistentCollectiveRunner,
    _install_collective,
    _worker_statuses,
)
from swarm_inference.experiments.experiment_013.persistent_protocol import (
    LEAN_ARCHITECTURE,
)
from swarm_inference.microworker_protocol import (
    MAGIC,
    NETWORK_PROFILES,
    PROTOCOL_VERSION,
)

HYPOTHESIS_ID = "H013-008"


def _record_fault_row(
    row: dict[str, Any],
    *,
    scale_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    trials_path: Path,
) -> None:
    scale_rows.append(row)
    all_rows.append(row)
    _append_jsonl(trials_path, row)


RECOVERABLE_SCENARIOS = (
    "reordered_arrivals",
    "transient_leaf_failure",
    "transient_parent_failure",
    "dropped_response",
    "stale_response",
    "duplicate_response",
    "timeout",
    "recursive_cancellation",
)


def _descendants(node: DelegatedNode) -> list[DelegatedNode]:
    values = [node]
    for child in node.children:
        values.extend(_descendants(child))
    return values


def _targets(topology: Any) -> tuple[DelegatedNode, DelegatedNode]:
    parent = topology.root_children[0]
    leaf = next(
        node
        for node in reversed(_descendants(parent))
        if not node.children and node.parent_worker != "stage-owner"
    )
    return parent, leaf


def _send_bounded_cancellation(topology: Any, operation_id: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    lock = threading.Lock()

    def cancel(node: DelegatedNode) -> None:
        started_ns = time.perf_counter_ns()
        try:
            response, transport = _control_round_trip(
                node.worker,
                {
                    "magic": MAGIC,
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "cancel_operation",
                    "operation_id": operation_id,
                },
                timeout_s=3.0,
            )
            record = {
                "worker_id": node.worker.worker_id,
                "status": response.get("status"),
                "messages": 2,
                "bytes": int(transport["request_bytes"]) + int(transport["response_bytes"]),
                "elapsed_ns": int(transport["elapsed_ns"]),
                "propagated_workers": int(response.get("propagated_workers", 1)),
                "tree_messages": int(response.get("tree_messages", 0)),
                "cancellation_depth": int(response.get("cancellation_depth", 0)),
                "propagation_failures": int(response.get("propagation_failures", 0)),
            }
        except BaseException as error:
            record = {
                "worker_id": node.worker.worker_id,
                "status": "failed",
                "messages": 0,
                "bytes": 0,
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "propagated_workers": 0,
                "tree_messages": 0,
                "cancellation_depth": 0,
                "propagation_failures": 1,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        with lock:
            records.append(record)

    with ThreadPoolExecutor(max_workers=len(topology.root_children)) as executor:
        futures = [executor.submit(cancel, node) for node in topology.root_children]
        for future in futures:
            future.result()
    return {
        "operation_id": operation_id,
        "root_cancel_degree": len(topology.root_children),
        "root_cancel_messages": sum(int(record["messages"]) for record in records),
        "root_cancel_bytes": sum(int(record["bytes"]) for record in records),
        "elapsed_ms": max((int(record["elapsed_ns"]) for record in records), default=0) / 1_000_000,
        "propagated_workers": sum(int(record["propagated_workers"]) for record in records),
        "tree_messages": sum(int(record["tree_messages"]) for record in records),
        "cancellation_depth": 1
        + max((int(record["cancellation_depth"]) for record in records), default=0),
        "propagation_failures": sum(int(record["propagation_failures"]) for record in records),
        "records": sorted(records, key=lambda record: str(record["worker_id"])),
    }


def _control_for(scenario: str, parent: DelegatedNode, leaf: DelegatedNode) -> dict[str, Any]:
    if scenario == "reordered_arrivals":
        return {
            "kind": "slow_child",
            "target_worker_ids": [leaf.worker.worker_id],
            "one_shot": False,
            "delay_ms": 12.0,
        }
    kinds = {
        "transient_leaf_failure": ("worker_failure_once", leaf),
        "transient_parent_failure": ("worker_failure_once", parent),
        "dropped_response": ("drop_result_once", leaf),
        "stale_response": ("stale_generation_once", leaf),
        "duplicate_response": ("duplicate_response_once", leaf),
    }
    if scenario in kinds:
        kind, node = kinds[scenario]
        return {
            "kind": kind,
            "target_worker_ids": [node.worker.worker_id],
            "one_shot": True,
        }
    if scenario == "timeout":
        return {
            "kind": "slow_child",
            "target_worker_ids": [leaf.worker.worker_id],
            "one_shot": False,
            "delay_ms": 150.0,
        }
    if scenario == "recursive_cancellation":
        return {
            "kind": "slow_child",
            "target_worker_ids": [leaf.worker.worker_id],
            "one_shot": False,
            "delay_ms": 500.0,
        }
    raise ValueError(f"unknown persistent fault scenario {scenario!r}")


def _annotate(
    row: dict[str, Any],
    *,
    scenario: str,
    repetition: int,
    phase: str,
    expected_status: str,
    clean_latency_ms: float,
    cancellation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed_status = str(row["status"])
    passed = observed_status == expected_status
    if expected_status == "ok":
        passed = passed and bool(row["correctness"])
    row.update(
        {
            "fault_scenario": scenario,
            "fault_phase": phase,
            "repetition": repetition,
            "expected_status": expected_status,
            "gate_pass": passed,
            "latency_amplification": (
                float(row["end_to_end_latency_ms"]) / clean_latency_ms if clean_latency_ms else None
            ),
            "cancellation": cancellation,
        }
    )
    return row


def _scenario_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(str(row["fault_scenario"]), str(row["fault_phase"])) for row in rows})
    for scenario, phase in keys:
        selected = [
            row for row in rows if row["fault_scenario"] == scenario and row["fault_phase"] == phase
        ]
        latencies = [float(row["end_to_end_latency_ms"]) for row in selected]
        result.append(
            {
                "scenario": scenario,
                "phase": phase,
                "trials": len(selected),
                "passed": sum(bool(row["gate_pass"]) for row in selected),
                "median_latency_ms": statistics.median(latencies),
                "maximum_latency_ms": max(latencies),
                "median_amplification": statistics.median(
                    float(row["latency_amplification"])
                    for row in selected
                    if row["latency_amplification"] is not None
                ),
                "total_retries": sum(int(row["retries"]) for row in selected),
            }
        )
    return result


def run_fault_benchmark(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    repetitions: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
    hypothesis_id: str,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_directory / "hypothesis.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError("fault hypothesis must be preregistered")
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = Path(__file__).with_name("persistent_protocol.py").resolve()
    source_files = [
        Path(__file__).resolve(),
        protocol_script,
        Path(__file__).with_name("persistent_harness.py").resolve(),
        repo_root / "src" / "swarm_inference" / "microworker_protocol.py",
    ]
    raw_directory = output_directory / "raw"
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, snapshot_directory / source.name)
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
    )
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )

    all_rows: list[dict[str, Any]] = []
    scale_results: list[dict[str, Any]] = []
    trials_path = raw_directory / "fault-trials.jsonl"
    for worker_count in worker_counts:
        branch_factor = 16 if worker_count <= 256 else 32
        pool = WorkerPool(
            count=worker_count,
            directory=output_directory / "workers" / f"workers-{worker_count:04d}",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        runner: PersistentCollectiveRunner | None = None
        scale: dict[str, Any] = {
            "worker_count": worker_count,
            "branch_factor": branch_factor,
            "status": "failed",
        }
        generation = 1
        scale_rows: list[dict[str, Any]] = []

        record = partial(
            _record_fault_row,
            scale_rows=scale_rows,
            all_rows=all_rows,
            trials_path=trials_path,
        )

        try:
            workers = pool.start()
            topology = build_delegated_topology(
                workers,
                branch_factor=branch_factor,
                topology_id=f"{hypothesis_id.lower()}-fault-n{worker_count}",
                route_lease_id=f"{hypothesis_id.lower()}-fault-lease-n{worker_count}",
            )
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_record(topology),
            )
            collective_id = f"{hypothesis_id.lower()}-fault-collective-n{worker_count}"
            setup_started_ns = time.perf_counter_ns()
            installation = _install_collective(
                topology,
                collective_id=collective_id,
                architecture=LEAN_ARCHITECTURE,
                profile=NETWORK_PROFILES["same_host_shaped"],
                mailbox_depth=2,
            )
            runner = PersistentCollectiveRunner(
                topology=topology,
                collective_id=collective_id,
                output_directory=output_directory,
                payload_bytes=256,
                operation_deadline_s=operation_deadline_s,
                cycle_id=hypothesis_id,
                architecture=LEAN_ARCHITECTURE,
            )
            root_prepare = runner.prepare(NETWORK_PROFILES["same_host_shaped"])
            setup_ms = (time.perf_counter_ns() - setup_started_ns) / 1_000_000
            parent, leaf = _targets(topology)

            first = runner.run_trial(
                trial_index=0,
                warmup=True,
                execution_generation=generation,
                profile=NETWORK_PROFILES["same_host_shaped"],
            )
            generation += 1
            record(
                _annotate(
                    first,
                    scenario="warmup",
                    repetition=-1,
                    phase="operation",
                    expected_status="ok",
                    clean_latency_ms=float(first["end_to_end_latency_ms"]),
                )
            )
            clean_rows: list[dict[str, Any]] = []
            for repetition in range(repetitions):
                row = runner.run_trial(
                    trial_index=repetition,
                    warmup=False,
                    execution_generation=generation,
                    profile=NETWORK_PROFILES["same_host_shaped"],
                )
                generation += 1
                clean_rows.append(row)
            clean_latency_ms = statistics.median(
                float(row["end_to_end_latency_ms"]) for row in clean_rows
            )
            for repetition, row in enumerate(clean_rows):
                record(
                    _annotate(
                        row,
                        scenario="clean",
                        repetition=repetition,
                        phase="operation",
                        expected_status="ok",
                        clean_latency_ms=clean_latency_ms,
                    )
                )

            for scenario in RECOVERABLE_SCENARIOS:
                for repetition in range(repetitions):
                    control = _control_for(scenario, parent, leaf)
                    operation_id = (
                        f"{hypothesis_id.lower()}-n{worker_count}-{scenario}-"
                        f"r{repetition}-{uuid4().hex[:8]}"
                    )
                    cancellation: dict[str, Any] | None = None
                    if scenario == "recursive_cancellation":
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                runner.run_trial,
                                trial_index=repetition,
                                warmup=False,
                                execution_generation=generation,
                                profile=NETWORK_PROFILES["same_host_shaped"],
                                operation_id_override=operation_id,
                                retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                                fault_control=control,
                            )
                            time.sleep(0.04)
                            cancellation = _send_bounded_cancellation(topology, operation_id)
                            row = future.result(timeout=operation_deadline_s + 5.0)
                        expected_status = "failed"
                    elif scenario == "timeout":
                        original_deadline_s = runner.operation_deadline_s
                        runner.operation_deadline_s = 0.06
                        try:
                            row = runner.run_trial(
                                trial_index=repetition,
                                warmup=False,
                                execution_generation=generation,
                                profile=NETWORK_PROFILES["same_host_shaped"],
                                operation_id_override=operation_id,
                                retry_policy={"max_attempts": 0, "backoff_ms": 0.0},
                                fault_control=control,
                            )
                        finally:
                            runner.operation_deadline_s = original_deadline_s
                        expected_status = "failed"
                    else:
                        row = runner.run_trial(
                            trial_index=repetition,
                            warmup=False,
                            execution_generation=generation,
                            profile=NETWORK_PROFILES["same_host_shaped"],
                            operation_id_override=operation_id,
                            retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                            fault_control=control,
                        )
                        expected_status = "ok"
                    generation += 1
                    record(
                        _annotate(
                            row,
                            scenario=scenario,
                            repetition=repetition,
                            phase="injected_operation",
                            expected_status=expected_status,
                            clean_latency_ms=clean_latency_ms,
                            cancellation=cancellation,
                        )
                    )
                    probe = runner.run_trial(
                        trial_index=repetition,
                        warmup=False,
                        execution_generation=generation,
                        profile=NETWORK_PROFILES["same_host_shaped"],
                        retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                        operation_id_override=(f"{operation_id}-next-generation-{uuid4().hex[:8]}"),
                    )
                    generation += 1
                    record(
                        _annotate(
                            probe,
                            scenario=scenario,
                            repetition=repetition,
                            phase="next_generation_probe",
                            expected_status="ok",
                            clean_latency_ms=clean_latency_ms,
                        )
                    )

            for repetition in range(repetitions):
                duplicate_operation = (
                    f"{hypothesis_id.lower()}-n{worker_count}-duplicate-request-"
                    f"r{repetition}-{uuid4().hex[:8]}"
                )
                first_duplicate = runner.run_trial(
                    trial_index=repetition,
                    warmup=False,
                    execution_generation=generation,
                    profile=NETWORK_PROFILES["same_host_shaped"],
                    operation_id_override=duplicate_operation,
                    retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                )
                replay = runner.run_trial(
                    trial_index=repetition,
                    warmup=False,
                    execution_generation=generation,
                    profile=NETWORK_PROFILES["same_host_shaped"],
                    operation_id_override=duplicate_operation,
                    retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                )
                generation += 1
                record(
                    _annotate(
                        first_duplicate,
                        scenario="duplicate_request_replay",
                        repetition=repetition,
                        phase="original",
                        expected_status="ok",
                        clean_latency_ms=clean_latency_ms,
                    )
                )
                replay["identical_aggregate"] = (
                    replay["actual"] == first_duplicate["actual"]
                    and replay["actual_digest"] == first_duplicate["actual_digest"]
                )
                replay_row = _annotate(
                    replay,
                    scenario="duplicate_request_replay",
                    repetition=repetition,
                    phase="replay",
                    expected_status="ok",
                    clean_latency_ms=clean_latency_ms,
                )
                replay_row["gate_pass"] = bool(replay_row["gate_pass"]) and bool(
                    replay_row["identical_aggregate"]
                )
                record(replay_row)

                stale = runner.run_trial(
                    trial_index=repetition,
                    warmup=False,
                    execution_generation=generation - 1,
                    profile=NETWORK_PROFILES["same_host_shaped"],
                    operation_id_override=(
                        f"{hypothesis_id.lower()}-n{worker_count}-stale-request-"
                        f"r{repetition}-{uuid4().hex[:8]}"
                    ),
                    retry_policy={"max_attempts": 0, "backoff_ms": 0.0},
                )
                record(
                    _annotate(
                        stale,
                        scenario="stale_generation_request",
                        repetition=repetition,
                        phase="rejected",
                        expected_status="failed",
                        clean_latency_ms=clean_latency_ms,
                    )
                )

            lost_worker = leaf.worker
            loss_started_ns = time.perf_counter_ns()
            lost_worker.process.terminate()
            lost_worker.process.wait(timeout=3.0)
            loss = {
                "worker_id": lost_worker.worker_id,
                "process_id": lost_worker.process_id,
                "returncode": lost_worker.process.returncode,
                "subtree_worker_count": leaf.subtree_worker_count,
                "termination_elapsed_ms": (time.perf_counter_ns() - loss_started_ns) / 1_000_000,
            }
            permanent = runner.run_trial(
                trial_index=0,
                warmup=False,
                execution_generation=generation,
                profile=NETWORK_PROFILES["same_host_shaped"],
                retry_policy={"max_attempts": 1, "backoff_ms": 0.0},
                operation_id_override=(
                    f"{hypothesis_id.lower()}-n{worker_count}-permanent-loss-{uuid4().hex[:8]}"
                ),
            )
            record(
                _annotate(
                    permanent,
                    scenario="permanent_worker_loss",
                    repetition=0,
                    phase="fail_closed",
                    expected_status="failed",
                    clean_latency_ms=clean_latency_ms,
                )
            )
            statuses = _worker_statuses(
                type(topology)(
                    topology_id=topology.topology_id,
                    route_lease_id=topology.route_lease_id,
                    route_generation=topology.route_generation,
                    branch_factor=topology.branch_factor,
                    root_children=topology.root_children,
                    nodes=tuple(node for node in topology.nodes if node.worker is not lost_worker),
                )
            )
            scale.update(
                {
                    "status": "completed",
                    "hierarchy_depth": topology.depth,
                    "root_degree": len(topology.root_children),
                    "collective_setup_ms": setup_ms,
                    "installation": installation,
                    "root_prepare": root_prepare,
                    "clean_latency_ms": clean_latency_ms,
                    "recoverable_injections": repetitions * len(RECOVERABLE_SCENARIOS),
                    "permanent_loss": loss,
                    "collective_state_after_permanent_loss": "invalid_fail_closed_rebuild_required",
                    "surviving_worker_status_count": len(statuses),
                }
            )
        except BaseException as error:
            scale.update({"error_type": type(error).__name__, "error": str(error)})
        finally:
            if runner is not None:
                scale["root_teardown"] = runner.close()
            scale["worker_teardown"] = pool.stop()
        scale["scenario_summaries"] = _scenario_summary(scale_rows)
        scale["all_recorded_gates_pass"] = bool(scale_rows) and all(
            bool(row["gate_pass"]) for row in scale_rows
        )
        scale_results.append(scale)
        _write_json(raw_directory / f"fault-scale-n{worker_count:04d}.json", scale)

    recoverable_rows = [
        row
        for row in all_rows
        if row["fault_scenario"] in RECOVERABLE_SCENARIOS
        and row["fault_phase"] == "injected_operation"
    ]
    probes = [
        row
        for row in all_rows
        if row["fault_scenario"] in RECOVERABLE_SCENARIOS
        and row["fault_phase"] == "next_generation_probe"
    ]
    summary = {
        "experiment_id": "013",
        "hypothesis_id": hypothesis_id,
        "architecture": LEAN_ARCHITECTURE,
        "all_scales_completed": all(scale["status"] == "completed" for scale in scale_results),
        "all_recorded_gates_pass": all(bool(row["gate_pass"]) for row in all_rows),
        "recoverable_injections": len(recoverable_rows),
        "recoverable_injections_passed": sum(bool(row["gate_pass"]) for row in recoverable_rows),
        "next_generation_probes": len(probes),
        "next_generation_probes_passed": sum(bool(row["gate_pass"]) for row in probes),
        "permanent_losses_fail_closed": sum(
            bool(row["gate_pass"])
            for row in all_rows
            if row["fault_scenario"] == "permanent_worker_loss"
        ),
        "scale_results": [
            {key: value for key, value in scale.items() if key not in {"installation"}}
            for scale in scale_results
        ],
        "scenario_summaries": _scenario_summary(all_rows),
        "environment": _source_identity(repo_root, source_files),
    }
    _write_json(output_directory / "fault-summary.json", summary)
    return summary


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_ints, default=(73, 1000))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--operation-deadline-s", type=float, default=3.0)
    parser.add_argument("--startup-deadline-s", type=float, default=240.0)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    args = parser.parse_args(argv)
    result = run_fault_benchmark(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        repetitions=args.repetitions,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        hypothesis_id=args.hypothesis_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_scales_completed"] and result["all_recorded_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
