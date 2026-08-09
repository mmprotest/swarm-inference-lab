"""Benchmark a genuine persistent event-driven subtree collective."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    _append_jsonl,
    _percentile,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DelegatedNode,
    DelegatedTopology,
    _control_round_trip,
    build_delegated_topology,
    topology_record,
)
from swarm_inference.experiments.experiment_013.persistent_protocol import (
    ARCHITECTURE,
    ARCHITECTURES,
    BUFFERED_ARCHITECTURE,
    COLLECTIVE_PROTOCOL,
    COMPACT_ARCHITECTURE,
    LEAN_ARCHITECTURE,
    SELECTOR_ARCHITECTURE,
    PersistentFanout,
    decode_subtree_metrics,
    make_collective_request,
)
from swarm_inference.microworker_protocol import (
    MAGIC,
    NETWORK_PROFILES,
    PROTOCOL_VERSION,
    LinkProfile,
    PersistentChannel,
    RoundTripError,
    aggregate_digest,
    canonical_json_bytes,
    combine_aggregates,
    combine_latency_histograms,
    combine_operation_aggregates,
    combine_transport_attempts,
    latency_histogram_observe,
    latency_histogram_percentile_ms,
    leaf_aggregate,
    retry_attempt_timeout_s,
    retryable_round_trip_error,
    worker_contribution,
)

HYPOTHESIS_ID = "H013-001"
REQUIRED_COUNTS = (2, 8, 32, 73, 128, 512, 1000)


@dataclass(slots=True)
class _RootJob:
    message: dict[str, Any]
    profile: LinkProfile
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    started_ns: int = 0
    completed_ns: int = 0
    result: Any = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)


class _RootEdgeLoop:
    """A root-owned setup-created dispatcher for one immediate subtree."""

    def __init__(self, node: DelegatedNode, trace: Any) -> None:
        import queue

        self.node = node
        self.trace = trace
        self.jobs: queue.Queue[_RootJob | None] = queue.Queue(maxsize=2)
        self.channel = PersistentChannel(node.worker.endpoint, "stage-owner", node.worker.worker_id)
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"collective-root-edge-{node.worker.worker_id}",
        )
        self.thread.start()
        self.wakeups = 0
        self.queue_delay_ns = 0
        self.maximum_queue_depth = 0
        self.dynamic_envelope_allocations = 0
        self.connection_establishments = 0

    def submit(self, message: dict[str, Any], profile: LinkProfile) -> _RootJob:
        job = _RootJob(message=message, profile=profile)
        self.dynamic_envelope_allocations += 1
        self.maximum_queue_depth = max(self.maximum_queue_depth, self.jobs.qsize() + 1)
        self.jobs.put(job)
        return job

    @staticmethod
    def wait(job: _RootJob, deadline_unix_ns: int) -> tuple[dict[str, Any], dict[str, Any]]:
        remaining_s = max(0.0, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
        if not job.done.wait(timeout=remaining_s):
            raise TimeoutError("root persistent edge mailbox deadline elapsed")
        if job.error is not None:
            raise job.error
        response, transport = job.result
        return dict(response), dict(transport)

    def _round_trip(
        self, message: dict[str, Any], profile: LinkProfile
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        retry_policy = dict(message.get("retry_policy", {}))
        maximum_attempts = int(retry_policy.get("max_attempts", 0))
        backoff_ms = float(retry_policy.get("backoff_ms", 0.0))
        attempts: list[dict[str, Any]] = []
        response: dict[str, Any] | None = None
        final_error: BaseException | None = None
        for attempt in range(maximum_attempts + 1):
            try:
                timeout_s = retry_attempt_timeout_s(
                    deadline_unix_ns=int(message["deadline_unix_ns"]),
                    maximum_attempts=maximum_attempts,
                    attempt=attempt,
                )
                response, transport = self.channel.round_trip(
                    message=message,
                    profile=profile,
                    timeout_s=timeout_s,
                    attempt=attempt,
                )
                if not message.get("collective_ping"):
                    if (
                        response.get("request_id") != message.get("request_id")
                        or response.get("operation_id") != message.get("operation_id")
                        or int(response.get("execution_generation", 0))
                        != int(message.get("execution_generation", 0))
                    ):
                        raise RoundTripError(
                            "root received a stale persistent response",
                            metrics=transport,
                            response=response,
                        )
                    aggregate = dict(response["aggregate"])
                    if response.get("aggregate_digest") != aggregate_digest(aggregate):
                        raise RoundTripError(
                            "root received an invalid persistent aggregate digest",
                            metrics=transport,
                            response=response,
                        )
                attempts.append(transport)
                final_error = None
                break
            except BaseException as error:
                final_error = error
                failure = dict(getattr(error, "metrics", {}))
                failure.setdefault("attempt", attempt)
                attempts.append(failure)
                will_retry = attempt < maximum_attempts and retryable_round_trip_error(error)
                self.trace(
                    "root_persistent_round_trip_failed",
                    operation_id=message.get("operation_id"),
                    request_id=message.get("request_id"),
                    receiver_worker_id=self.node.worker.worker_id,
                    attempt=attempt,
                    will_retry=will_retry,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                if not will_retry:
                    break
                if backoff_ms:
                    time.sleep(backoff_ms / 1_000.0)
        combined = combine_transport_attempts(attempts)
        self.connection_establishments += int(combined.get("new_connection_count", 0))
        if final_error is not None or response is None:
            error = final_error or RuntimeError("root persistent retry exhausted")
            error.root_transport_metrics = combined
            raise error
        return response, combined

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            job.started_ns = time.perf_counter_ns()
            self.wakeups += 1
            self.queue_delay_ns += job.started_ns - job.enqueued_ns
            try:
                job.result = self._round_trip(job.message, job.profile)
            except BaseException as error:
                job.error = error
            finally:
                job.completed_ns = time.perf_counter_ns()
                job.done.set()

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=2.0)
        self.channel.close()


def _install_collective(
    topology: DelegatedTopology,
    *,
    collective_id: str,
    architecture: str,
    profile: LinkProfile,
    mailbox_depth: int,
) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    install_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lock = threading.Lock()

    def install(node: DelegatedNode) -> None:
        message = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "install_collective",
            "collective_protocol": COLLECTIVE_PROTOCOL,
            "architecture": architecture,
            "collective_id": collective_id,
            "topology_id": topology.topology_id,
            "route_lease_id": topology.route_lease_id,
            "route_generation": topology.route_generation,
            "mailbox_depth": mailbox_depth,
            "node": node.install_node(),
        }
        try:
            response, transport = _control_round_trip(node.worker, message, timeout_s=30.0)
            if (
                response.get("kind") != "collective_installed"
                or response.get("collective_id") != collective_id
                or response.get("topology_id") != topology.topology_id
            ):
                raise RuntimeError("worker acknowledged a different persistent collective")
            with lock:
                install_records.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "process_id": node.worker.process_id,
                        "child_count": len(node.children),
                        "response": response,
                        **transport,
                    }
                )
        except BaseException as error:
            with lock:
                errors.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "phase": "install",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    with ThreadPoolExecutor(max_workers=min(32, len(topology.nodes))) as executor:
        futures = [executor.submit(install, node) for node in topology.nodes]
        for future in futures:
            future.result()
    install_elapsed_ns = time.perf_counter_ns() - started_ns
    if errors or len(install_records) != len(topology.nodes):
        raise RuntimeError(f"persistent collective installation failed: {errors}")

    prepare_started_ns = time.perf_counter_ns()
    prepare_records: list[dict[str, Any]] = []

    def prepare(node: DelegatedNode) -> None:
        message = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "prepare_collective",
            "collective_id": collective_id,
            "topology_id": topology.topology_id,
            "route_lease_id": topology.route_lease_id,
            "route_generation": topology.route_generation,
            "deadline_unix_ns": time.time_ns() + 30_000_000_000,
            "network_profile": {
                "name": profile.name,
                "rtt_ms": profile.rtt_ms,
                "upload_mbps": profile.upload_mbps,
                "download_mbps": profile.download_mbps,
                "jitter_ms": profile.jitter_ms,
                "request_loss_rate": profile.request_loss_rate,
                "temporary_disconnect_rate": profile.temporary_disconnect_rate,
            },
        }
        try:
            response, transport = _control_round_trip(node.worker, message, timeout_s=30.0)
            if response.get("kind") != "collective_prepared":
                raise RuntimeError("worker did not prepare its persistent child edges")
            with lock:
                prepare_records.append(
                    {"worker_id": node.worker.worker_id, "response": response, **transport}
                )
        except BaseException as error:
            with lock:
                errors.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "phase": "prepare",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    with ThreadPoolExecutor(max_workers=min(32, len(topology.nodes))) as executor:
        futures = [executor.submit(prepare, node) for node in topology.nodes]
        for future in futures:
            future.result()
    prepare_elapsed_ns = time.perf_counter_ns() - prepare_started_ns
    if errors or len(prepare_records) != len(topology.nodes):
        raise RuntimeError(f"persistent collective preparation failed: {errors}")
    return {
        "status": "ok",
        "collective_id": collective_id,
        "architecture": architecture,
        "worker_count": len(topology.nodes),
        "route_installation_count": len(install_records),
        "root_setup_direct_degree": len(topology.nodes),
        "topology_rebuilds": 1,
        "worker_persistent_execution_loop_starts": sum(
            int(item["response"]["persistent_execution_loop_starts"]) for item in install_records
        ),
        "worker_persistent_dispatch_loop_starts": sum(
            int(item["response"]["persistent_dispatch_loop_starts"]) for item in install_records
        ),
        "worker_edge_connection_establishments": sum(
            int(item["response"]["new_connections"]) for item in prepare_records
        ),
        "setup_messages": 2 * (len(install_records) + len(prepare_records))
        + sum(int(item["response"]["messages"]) for item in prepare_records),
        "setup_bytes": sum(
            int(item["request_bytes"]) + int(item["response_bytes"])
            for item in install_records + prepare_records
        )
        + sum(int(item["response"]["bytes"]) for item in prepare_records),
        "install_elapsed_ns": install_elapsed_ns,
        "prepare_elapsed_ns": prepare_elapsed_ns,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "install_records": sorted(install_records, key=lambda item: item["worker_id"]),
        "prepare_records": sorted(prepare_records, key=lambda item: item["worker_id"]),
        "errors": errors,
    }


def _worker_statuses(topology: DelegatedTopology) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    lock = threading.Lock()

    def fetch(node: DelegatedNode) -> None:
        response, _transport = _control_round_trip(
            node.worker,
            {"magic": MAGIC, "protocol_version": PROTOCOL_VERSION, "kind": "status"},
            timeout_s=10.0,
        )
        with lock:
            statuses[node.worker.worker_id] = response

    with ThreadPoolExecutor(max_workers=min(64, len(topology.nodes))) as executor:
        futures = [executor.submit(fetch, node) for node in topology.nodes]
        for future in futures:
            future.result()
    return statuses


class PersistentCollectiveRunner:
    def __init__(
        self,
        *,
        topology: DelegatedTopology,
        collective_id: str,
        output_directory: Path,
        payload_bytes: int,
        operation_deadline_s: float,
        cycle_id: str,
        architecture: str = ARCHITECTURE,
    ) -> None:
        self.topology = topology
        self.collective_id = collective_id
        self.output_directory = output_directory
        self.operation_deadline_s = operation_deadline_s
        self.cycle_id = cycle_id
        self.architecture = architecture
        self.compact_wire = architecture in {
            COMPACT_ARCHITECTURE,
            BUFFERED_ARCHITECTURE,
            LEAN_ARCHITECTURE,
        }
        self.lean_wire = architecture == LEAN_ARCHITECTURE
        self.payload_b64 = base64.b64encode(bytes(payload_bytes)).decode("ascii")
        self.payload_bytes = payload_bytes
        self.trace_path = output_directory / "traces" / "root.jsonl"
        self.trace_lock = threading.Lock()
        self.expected = combine_aggregates(
            [
                leaf_aggregate(node.worker.worker_id, worker_contribution(node.worker.worker_index))
                for node in topology.nodes
            ]
        )
        self.root_selector: PersistentFanout | None = None
        if architecture in {
            SELECTOR_ARCHITECTURE,
            COMPACT_ARCHITECTURE,
            BUFFERED_ARCHITECTURE,
            LEAN_ARCHITECTURE,
        }:
            self.root_loops: dict[str, _RootEdgeLoop] = {}
            self.root_selector = PersistentFanout(
                [node.child_route() for node in topology.root_children],
                sender_id="stage-owner",
                trace=self._trace,
            )
        else:
            self.root_loops = {
                node.worker.worker_id: _RootEdgeLoop(node, self._trace)
                for node in topology.root_children
            }
        self.root_persistent_loop_starts = len(self.root_loops)
        self.root_reusable_buffer_allocations = 1 + len(topology.root_children)

    def _trace(self, event: str, **payload: Any) -> None:
        _append_jsonl(
            self.trace_path,
            {
                "event": event,
                "hypothesis_id": self.cycle_id,
                "collective_id": self.collective_id,
                "timestamp_unix_ns": time.time_ns(),
                "root_process_id": os.getpid(),
                **payload,
            },
            self.trace_lock,
        )

    def prepare(self, profile: LinkProfile) -> dict[str, Any]:
        deadline_unix_ns = time.time_ns() + 30_000_000_000
        jobs: list[tuple[DelegatedNode, _RootJob]] = []
        started_ns = time.perf_counter_ns()
        selector_pings: dict[str, dict[str, Any]] = {}
        for node in self.topology.root_children:
            message = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "delegated_execute",
                "collective_protocol": COLLECTIVE_PROTOCOL,
                "collective_ping": True,
                "collective_id": self.collective_id,
                "topology_id": self.topology.topology_id,
                "route_lease_id": self.topology.route_lease_id,
                "route_generation": self.topology.route_generation,
                "request_id": f"prepare:stage-owner:{node.worker.worker_id}",
                "parent_worker": "stage-owner",
                "target_worker": node.worker.worker_id,
                "deadline_unix_ns": deadline_unix_ns,
                "retry_policy": {"max_attempts": 0, "backoff_ms": 0.0},
                "connection_policy": "persistent",
            }
            if self.root_selector is not None:
                selector_pings[node.worker.worker_id] = message
            else:
                jobs.append((node, self.root_loops[node.worker.worker_id].submit(message, profile)))
        transports: list[dict[str, Any]] = []
        if self.root_selector is not None:
            selector_results = self.root_selector.round_trip_many(
                selector_pings, profile=profile, deadline_unix_ns=deadline_unix_ns
            )
            transports.extend(transport for _response, transport in selector_results.values())
        for node, job in jobs:
            response, transport = _RootEdgeLoop.wait(job, deadline_unix_ns)
            if (
                response.get("kind") != "collective_pong"
                or response.get("worker_id") != node.worker.worker_id
            ):
                raise RuntimeError("root edge prewarm returned the wrong worker")
            transports.append(transport)
        return {
            "root_edge_count": len(jobs) + len(selector_pings),
            "root_persistent_dispatch_loop_starts": len(jobs),
            "root_selector_loop_starts": int(self.root_selector is not None),
            "root_edge_connection_establishments": sum(
                int(item["new_connection_count"]) for item in transports
            ),
            "messages": sum(
                int(item["messages_sent"]) + int(item["messages_received"]) for item in transports
            ),
            "bytes": sum(
                int(item["request_bytes"]) + int(item["response_bytes"]) for item in transports
            ),
            "elapsed_ns": time.perf_counter_ns() - started_ns,
        }

    def run_trial(
        self,
        *,
        trial_index: int,
        warmup: bool,
        execution_generation: int,
        profile: LinkProfile,
        retry_policy: dict[str, Any] | None = None,
        fault_control: dict[str, Any] | None = None,
        operation_id_override: str | None = None,
        aggregation: dict[str, Any] | None = None,
        workload: dict[str, Any] | None = None,
        payload_b64: str | None = None,
    ) -> dict[str, Any]:
        operation_id = operation_id_override or (
            f"{self.cycle_id.lower()}-n{len(self.topology.nodes)}-"
            f"{'first' if warmup else 'warm'}-{trial_index}-{uuid4().hex[:8]}"
        )
        deadline_unix_ns = time.time_ns() + int(self.operation_deadline_s * 1_000_000_000)
        retry = dict(retry_policy or {"max_attempts": 0, "backoff_ms": 0.0})
        faults = dict(fault_control or {})
        aggregate_contract = dict(
            aggregation or {"mode": "exact_int64_sum", "ordering": "deterministic_worker_key"}
        )
        effective_payload_b64 = payload_b64 or self.payload_b64
        operation_digest = (
            hashlib.sha256(
                canonical_json_bytes(
                    {
                        "operation_id": operation_id,
                        "execution_generation": execution_generation,
                        "payload_b64": effective_payload_b64,
                        "aggregation": aggregate_contract,
                        "workload": workload,
                    }
                )
            ).hexdigest()
            if self.lean_wire
            else None
        )
        submitted_at_unix_ns = time.time_ns()
        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        dispatch_started_ns = time.perf_counter_ns()
        jobs: list[tuple[DelegatedNode, _RootJob]] = []
        selector_messages: dict[str, dict[str, Any]] = {}
        nodes_by_id = {node.worker.worker_id: node for node in self.topology.root_children}
        for node in self.topology.root_children:
            request_id = f"{operation_id}:{node.worker.worker_id}"
            message = make_collective_request(
                collective_id=self.collective_id,
                topology_id=self.topology.topology_id,
                route_lease_id=self.topology.route_lease_id,
                route_generation=self.topology.route_generation,
                request_id=request_id,
                operation_id=operation_id,
                execution_generation=execution_generation,
                parent_worker="stage-owner",
                target_worker=node.worker.worker_id,
                deadline_unix_ns=deadline_unix_ns,
                profile=profile,
                retry_policy=retry,
                fault_control=faults,
                aggregation=aggregate_contract,
                payload_b64=effective_payload_b64,
                workload=workload,
                architecture=self.architecture,
                operation_digest=operation_digest,
            )
            if self.root_selector is not None:
                selector_messages[node.worker.worker_id] = message
            else:
                jobs.append((node, self.root_loops[node.worker.worker_id].submit(message, profile)))
        root_dispatch_ns = time.perf_counter_ns() - dispatch_started_ns

        results: list[tuple[DelegatedNode, dict[str, Any], dict[str, Any]]] = []
        failures: list[dict[str, Any]] = []
        if self.root_selector is not None:
            try:
                selector_results = self.root_selector.round_trip_many(
                    selector_messages,
                    profile=profile,
                    deadline_unix_ns=deadline_unix_ns,
                )
                root_dispatch_ns = self.root_selector.last_dispatch_ns
                results.extend(
                    (nodes_by_id[child_id], response, transport)
                    for child_id, (response, transport) in selector_results.items()
                )
            except BaseException as error:
                failures.append(
                    {
                        "worker_id": str(
                            dict(getattr(error, "metrics", {})).get(
                                "receiver_worker_id", "selector-fanout"
                            )
                        ),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "transport_metrics": dict(getattr(error, "metrics", {})),
                    }
                )
        for node, job in jobs:
            try:
                response, transport = _RootEdgeLoop.wait(job, deadline_unix_ns)
                results.append((node, response, transport))
            except BaseException as error:
                failures.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "transport_metrics": dict(
                            getattr(error, "root_transport_metrics", getattr(error, "metrics", {}))
                        ),
                    }
                )
        root_cpu_ns = time.process_time_ns() - cpu_started_ns
        latency_ns = time.perf_counter_ns() - wall_started_ns
        status = "failed" if failures else "ok"
        actual = (
            combine_operation_aggregates(
                [dict(response["aggregate"]) for _node, response, _t in results],
                aggregate_contract,
            )
            if results
            else None
        )
        if (
            aggregate_contract["mode"] == "exact_int64_sum"
            and status == "ok"
            and actual != self.expected
        ):
            status = "incorrect"
            failures.append(
                {
                    "error_type": "CorrectnessError",
                    "error": "persistent aggregate differs from the exact reference",
                }
            )
        root_transports = [transport for _node, _response, transport in results] + [
            dict(item["transport_metrics"]) for item in failures
        ]
        subtree_metrics = [decode_subtree_metrics(response) for _n, response, _t in results]
        root_request_bytes = sum(int(item.get("request_bytes", 0)) for item in root_transports)
        root_response_bytes = sum(int(item.get("response_bytes", 0)) for item in root_transports)
        edge_histogram: dict[str, int] = {}
        leaf_histogram: dict[str, int] = {}
        for (_node, _response, transport), metrics in zip(results, subtree_metrics, strict=True):
            root_histogram: dict[str, int] = {}
            latency_histogram_observe(root_histogram, int(transport["elapsed_ns"]))
            edge_histogram = combine_latency_histograms(
                edge_histogram,
                root_histogram,
                dict(metrics.get("latency_histogram", {})),
            )
            leaf_histogram = combine_latency_histograms(
                leaf_histogram, dict(metrics.get("leaf_latency_histogram", {}))
            )
        root_degree = len(self.topology.root_children)
        total_messages = sum(
            int(item.get("messages_sent", 0)) + int(item.get("messages_received", 0))
            for item in root_transports
        ) + sum(int(item["total_messages"]) for item in subtree_metrics)
        total_bytes = (
            root_request_bytes
            + root_response_bytes
            + sum(int(item["total_bytes"]) for item in subtree_metrics)
        )
        row = {
            "schema_version": "1.0",
            "experiment_id": "013",
            "cycle_id": self.cycle_id,
            "architecture": self.architecture,
            "collective_protocol": COLLECTIVE_PROTOCOL,
            "mode": "persistent_collective",
            "worker_count": len(self.topology.nodes),
            "branch_factor": self.topology.branch_factor,
            "network_profile": profile.name,
            "payload_bytes": len(base64.b64decode(effective_payload_b64)),
            "trial_index": trial_index,
            "warmup": warmup,
            "operation_id": operation_id,
            "execution_generation": execution_generation,
            "route_generation": self.topology.route_generation,
            "submitted_at_unix_ns": submitted_at_unix_ns,
            "status": status,
            "correctness": status == "ok",
            "expected": self.expected if aggregate_contract["mode"] == "exact_int64_sum" else None,
            "actual": actual,
            "expected_digest": (
                aggregate_digest(self.expected)
                if aggregate_contract["mode"] == "exact_int64_sum"
                else None
            ),
            "actual_digest": aggregate_digest(actual) if actual is not None else None,
            "result_published": status == "ok",
            "failure_details": failures,
            "root_rpc_count": sum(int(item.get("attempt_count", 0)) for item in root_transports),
            "root_leaf_rpc_count": sum(not node.children for node, _r, _t in results),
            "root_messages_sent": sum(
                int(item.get("messages_sent", 0)) for item in root_transports
            ),
            "root_messages_received": sum(
                int(item.get("messages_received", 0)) for item in root_transports
            ),
            "root_messages_total": sum(
                int(item.get("messages_sent", 0)) + int(item.get("messages_received", 0))
                for item in root_transports
            ),
            "root_bytes_sent": root_request_bytes,
            "root_bytes_received": root_response_bytes,
            "root_bytes_total": root_request_bytes + root_response_bytes,
            "root_direct_degree": root_degree,
            "root_serial_waits": 1 if self.root_selector is not None else root_degree,
            "root_coordinator_waits": int(bool(root_degree)),
            "root_cpu_ns": root_cpu_ns,
            "root_dispatch_ns": root_dispatch_ns,
            "root_persistent_dispatch_loop_wakeups": (
                0 if self.root_selector is not None else root_degree
            ),
            "root_selector_loop_wakeups": int(self.root_selector is not None),
            "root_new_task_creation": 0,
            "root_topology_rebuilds": 0,
            "root_dynamic_envelope_allocations": root_degree,
            "root_reusable_buffer_uses": root_degree,
            "root_connection_count": sum(
                int(item.get("new_connection_count", 0)) for item in root_transports
            ),
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "worker_to_worker_rpc_count": sum(
                int(item["worker_to_worker_rpc_count"]) for item in subtree_metrics
            ),
            "leaf_rpc_count": (
                sum(not node.children for node in self.topology.nodes)
                if self.lean_wire
                else sum(int(item["leaf_rpc_count"]) for item in subtree_metrics)
            ),
            "hierarchy_depth": (
                self.topology.depth
                if self.lean_wire
                else max((int(item["hierarchy_depth"]) for item in subtree_metrics), default=0)
            ),
            "reduction_depth": (
                self.topology.depth
                if self.lean_wire
                else (
                    1
                    + max(
                        (int(item["reduction_depth"]) for item in subtree_metrics),
                        default=0,
                    )
                    if results
                    else 0
                )
            ),
            "intermediate_reductions": (
                1 + sum(bool(node.children) for node in self.topology.nodes)
                if self.lean_wire
                else 1 + sum(int(item["intermediate_reductions"]) for item in subtree_metrics)
            ),
            "end_to_end_latency_ns": latency_ns,
            "end_to_end_latency_ms": latency_ns / 1_000_000,
            "throughput_ops_s": 1_000_000_000 / latency_ns if latency_ns else 0.0,
            "edge_latency_p50_ms": latency_histogram_percentile_ms(edge_histogram, 50),
            "edge_latency_p95_ms": latency_histogram_percentile_ms(edge_histogram, 95),
            "edge_latency_p99_ms": latency_histogram_percentile_ms(edge_histogram, 99),
            "leaf_latency_p50_ms": latency_histogram_percentile_ms(leaf_histogram, 50),
            "leaf_latency_p95_ms": latency_histogram_percentile_ms(leaf_histogram, 95),
            "leaf_latency_p99_ms": latency_histogram_percentile_ms(leaf_histogram, 99),
            "total_connection_count": sum(
                int(item.get("new_connection_count", 0)) for item in root_transports
            )
            + sum(int(item["connection_count"]) for item in subtree_metrics),
            "total_connection_ns": sum(int(item.get("connect_ns", 0)) for item in root_transports)
            + sum(int(item["connection_ns"]) for item in subtree_metrics),
            "total_worker_cpu_ns": sum(int(item["worker_cpu_ns"]) for item in subtree_metrics),
            "persistent_execution_loop_wakeups": (
                len(self.topology.nodes)
                if self.compact_wire
                else sum(int(item["persistent_execution_loop_wakeups"]) for item in subtree_metrics)
            ),
            "persistent_receive_loop_wakeups": (
                len(self.topology.nodes)
                if self.compact_wire
                else sum(
                    int(item.get("persistent_receive_loop_wakeups", 0)) for item in subtree_metrics
                )
            ),
            "persistent_mailbox_loop_wakeups": (
                0
                if self.compact_wire
                else sum(
                    int(item.get("persistent_mailbox_loop_wakeups", 0)) for item in subtree_metrics
                )
            ),
            "persistent_dispatch_loop_wakeups": (
                0 if self.root_selector is not None else root_degree
            )
            + sum(int(item.get("persistent_dispatch_loop_wakeups", 0)) for item in subtree_metrics),
            "application_scheduler_events": (
                len(self.topology.nodes) + 1
                if self.compact_wire
                else int(self.root_selector is not None)
                + (0 if self.root_selector is not None else root_degree)
                + sum(int(item["application_scheduler_events"]) for item in subtree_metrics)
            ),
            "worker_activations": sum(
                int(item.get("worker_activations", 0)) for item in subtree_metrics
            ),
            "new_task_creation": sum(
                int(item.get("new_task_creation", 0)) for item in subtree_metrics
            ),
            "topology_rebuilds": sum(
                int(item.get("topology_rebuilds", 0)) for item in subtree_metrics
            ),
            "dynamic_envelope_allocations": (
                2 * len(self.topology.nodes)
                if self.compact_wire
                else root_degree
                + sum(int(item["dynamic_envelope_allocations"]) for item in subtree_metrics)
            ),
            "reusable_buffer_uses": (
                2 * len(self.topology.nodes)
                if self.compact_wire
                else root_degree
                + sum(int(item["reusable_buffer_uses"]) for item in subtree_metrics)
            ),
            "queue_delay_ns": sum(int(item["queue_delay_ns"]) for item in subtree_metrics),
            "intermediate_dispatch_elapsed_ns": sum(
                int(item["dispatch_elapsed_ns"]) for item in subtree_metrics
            ),
            "local_execution_ns": sum(int(item["local_execution_ns"]) for item in subtree_metrics),
            "reduction_ns": sum(int(item["reduction_ns"]) for item in subtree_metrics),
            "serialization_ns": sum(
                int(item.get("serialization_ns", 0)) for item in root_transports
            )
            + sum(int(item["serialization_ns"]) for item in subtree_metrics),
            "maximum_queue_depth": max(
                [loop.maximum_queue_depth for loop in self.root_loops.values()]
                + [int(item.get("maximum_queue_depth", 1)) for item in subtree_metrics]
                + [0]
            ),
            "retries": sum(int(item.get("retry_count", 0)) for item in root_transports)
            + sum(int(item["retries"]) for item in subtree_metrics),
            "failures": len(failures) + sum(int(item["failures"]) for item in subtree_metrics),
            "duplicated_work": sum(int(item["duplicated_work"]) for item in subtree_metrics),
            "retry_policy": retry,
            "fault_control": faults,
            "telemetry_delivery": (
                "lean_in_band_wire_counters_plus_buffered_worker_traces"
                if self.lean_wire
                else "in_band_subtree_metrics"
            ),
            "completion_unix_ns": time.time_ns(),
        }
        self._trace(
            "root_persistent_operation_completed"
            if status == "ok"
            else "root_persistent_operation_failed",
            operation_id=operation_id,
            execution_generation=execution_generation,
            worker_count=len(self.topology.nodes),
            root_degree=root_degree,
            root_leaf_rpc_count=row["root_leaf_rpc_count"],
            root_messages=row["root_messages_total"],
            total_messages=total_messages,
            end_to_end_latency_ns=latency_ns,
            status=status,
        )
        return row

    def close(self) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        for loop in self.root_loops.values():
            loop.close()
        if self.root_selector is not None:
            self.root_selector.close()
        return {
            "root_persistent_dispatch_loop_stops": len(self.root_loops),
            "root_selector_loop_stops": int(self.root_selector is not None),
            "elapsed_ns": time.perf_counter_ns() - started_ns,
        }


def _dispersion(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stddev": 0.0, "variance": 0.0, "mad": 0.0, "iqr": 0.0}
    median = statistics.median(values)
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "variance": statistics.variance(values) if len(values) > 1 else 0.0,
        "mad": statistics.median(abs(item - median) for item in values),
        "iqr": _percentile(values, 75) - _percentile(values, 25),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _summarize(
    rows: list[dict[str, Any]], setups: list[dict[str, Any]], attempts: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for worker_count in sorted({int(item["worker_count"]) for item in rows}):
        selected = [
            item
            for item in rows
            if int(item["worker_count"]) == worker_count
            and not item["warmup"]
            and item["status"] == "ok"
        ]
        all_measured = [
            item
            for item in rows
            if int(item["worker_count"]) == worker_count and not item["warmup"]
        ]
        first = next(
            item for item in rows if int(item["worker_count"]) == worker_count and item["warmup"]
        )
        setup = next(item for item in setups if int(item["worker_count"]) == worker_count)
        latencies = [float(item["end_to_end_latency_ms"]) for item in selected]
        summaries.append(
            {
                "worker_count": worker_count,
                "branch_factor": int(selected[0]["branch_factor"]) if selected else None,
                "architecture": selected[0]["architecture"] if selected else None,
                "network_profile": selected[0]["network_profile"] if selected else None,
                "trial_count": len(all_measured),
                "successful_trials": len(selected),
                "failed_trials": len(all_measured) - len(selected),
                "first_operation_latency_ms": float(first["end_to_end_latency_ms"]),
                "collective_setup_ms": float(setup["collective_setup_ns"]) / 1_000_000,
                "process_start_ms": float(setup["process_start_ns"]) / 1_000_000,
                "cold_latency_including_process_start_ms": float(
                    setup["cold_latency_including_process_start_ns"]
                )
                / 1_000_000,
                "warm_latency_p50_ms": statistics.median(latencies) if latencies else None,
                "warm_latency_p95_ms": _percentile(latencies, 95) if latencies else None,
                "warm_latency_p99_ms": _percentile(latencies, 99) if latencies else None,
                "warm_latency_dispersion_ms": _dispersion(latencies),
                "throughput_ops_s_median": statistics.median(
                    float(item["throughput_ops_s"]) for item in selected
                )
                if selected
                else None,
                **{
                    f"{field}_median": statistics.median(int(item[field]) for item in selected)
                    if selected
                    else None
                    for field in (
                        "root_messages_total",
                        "root_serial_waits",
                        "root_direct_degree",
                        "root_leaf_rpc_count",
                        "root_bytes_total",
                        "total_messages",
                        "total_bytes",
                        "hierarchy_depth",
                        "worker_activations",
                        "new_task_creation",
                        "topology_rebuilds",
                        "total_connection_count",
                        "persistent_execution_loop_wakeups",
                        "persistent_receive_loop_wakeups",
                        "persistent_mailbox_loop_wakeups",
                        "persistent_dispatch_loop_wakeups",
                        "application_scheduler_events",
                        "dynamic_envelope_allocations",
                    )
                },
                "memory_growth_bytes": int(setup["memory_after_total_worker_rss_bytes"])
                - int(setup["memory_before_total_worker_rss_bytes"]),
                "tail_statistical_note": (
                    "p99 is indicative only at this trial count"
                    if len(selected) < 100
                    else "p99 has at least 100 successful observations"
                ),
            }
        )
    return {
        "experiment_id": "013",
        "hypothesis_id": rows[0]["cycle_id"] if rows else HYPOTHESIS_ID,
        "architecture": rows[0]["architecture"] if rows else None,
        "run_kind": "measured_real_processes_tcp_loopback_single_host",
        "network_evidence": "physical loopback only; no physical LAN or WAN claim",
        "summaries": summaries,
        "attempts": attempts,
        "all_operations_correct": all(item["correctness"] for item in rows),
        "all_scales_completed": all(item["status"] == "completed" for item in attempts),
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[key for key in rows[0] if not isinstance(rows[0][key], dict)],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: value for key, value in row.items() if not isinstance(value, dict)}
            )


def run_benchmark(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    branch_factor: int,
    payload_bytes: int,
    profile: LinkProfile,
    warmup_trials: int,
    measured_trials: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
    mailbox_depth: int,
    hypothesis_id: str,
    architecture: str = ARCHITECTURE,
) -> dict[str, Any]:
    if warmup_trials != 1:
        raise ValueError("the lifecycle benchmark requires exactly one first-operation trial")
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = Path(__file__).with_name("persistent_protocol.py").resolve()
    source_files = [
        protocol_script,
        Path(__file__).resolve(),
        Path(__file__).parents[1] / "experiment_012" / "delegation_harness.py",
        repo_root / "src" / "swarm_inference" / "microworker_protocol.py",
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_directory / "hypothesis.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError("benchmark hypothesis must be preregistered before execution")
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
    snapshot = raw_directory / "source-snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        shutil.copy2(source, snapshot / source.name)

    rows: list[dict[str, Any]] = []
    setups: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    trials_path = raw_directory / "operations.jsonl"
    for worker_count in worker_counts:
        scale_started_ns = time.perf_counter_ns()
        scale_started_unix_ns = time.time_ns()
        pool = WorkerPool(
            count=worker_count,
            directory=output_directory / "workers" / f"workers-{worker_count:04d}",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        runner: PersistentCollectiveRunner | None = None
        shutdown: dict[str, Any] = {}
        attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "attempted_unix_ns": scale_started_unix_ns,
            "status": "failed",
        }
        try:
            process_start_started_ns = time.perf_counter_ns()
            workers = pool.start()
            process_start_ns = time.perf_counter_ns() - process_start_started_ns
            topology_started_ns = time.perf_counter_ns()
            topology = build_delegated_topology(
                workers,
                branch_factor=branch_factor,
                topology_id=f"{hypothesis_id.lower()}-persistent-n{worker_count}",
                route_lease_id=f"{hypothesis_id.lower()}-lease-n{worker_count}",
            )
            topology_construction_ns = time.perf_counter_ns() - topology_started_ns
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_record(topology),
            )
            collective_id = f"{hypothesis_id.lower()}-collective-n{worker_count}"
            collective_setup_started_ns = time.perf_counter_ns()
            installation = _install_collective(
                topology,
                collective_id=collective_id,
                architecture=architecture,
                profile=profile,
                mailbox_depth=mailbox_depth,
            )
            runner = PersistentCollectiveRunner(
                topology=topology,
                collective_id=collective_id,
                output_directory=output_directory,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                cycle_id=hypothesis_id,
                architecture=architecture,
            )
            root_prepare = runner.prepare(profile)
            collective_setup_ns = time.perf_counter_ns() - collective_setup_started_ns
            _write_json(
                output_directory / "collective-installation" / f"workers-{worker_count:04d}.json",
                {**installation, "root_prepare": root_prepare},
            )
            memory_before = pool.sample_memory_snapshot() or {
                "total_worker_rss_bytes": 0,
                "maximum_worker_rss_bytes": 0,
                "median_worker_rss_bytes": 0,
                "observed_workers": 0,
            }
            generation = 1
            first = runner.run_trial(
                trial_index=0,
                warmup=True,
                execution_generation=generation,
                profile=profile,
            )
            rows.append(first)
            _append_jsonl(trials_path, first)
            generation += 1
            for trial_index in range(measured_trials):
                row = runner.run_trial(
                    trial_index=trial_index,
                    warmup=False,
                    execution_generation=generation,
                    profile=profile,
                )
                rows.append(row)
                _append_jsonl(trials_path, row)
                generation += 1
            memory_after = pool.sample_memory_snapshot() or memory_before
            statuses = _worker_statuses(topology)
            _write_json(
                output_directory / "traces" / f"worker-status-n{worker_count:04d}.json",
                statuses,
            )
            setup = {
                "worker_count": worker_count,
                "branch_factor": branch_factor,
                "process_start_ns": process_start_ns,
                "worker_process_starts": len(workers),
                "topology_construction_ns": topology_construction_ns,
                "collective_setup_ns": collective_setup_ns,
                "route_installation_ns": installation["install_elapsed_ns"],
                "session_establishment_ns": installation["prepare_elapsed_ns"]
                + root_prepare["elapsed_ns"],
                "route_installation_count": installation["route_installation_count"],
                "topology_rebuilds": installation["topology_rebuilds"],
                "worker_persistent_execution_loop_starts": installation[
                    "worker_persistent_execution_loop_starts"
                ],
                "worker_persistent_dispatch_loop_starts": installation[
                    "worker_persistent_dispatch_loop_starts"
                ],
                "root_persistent_dispatch_loop_starts": root_prepare[
                    "root_persistent_dispatch_loop_starts"
                ],
                "root_selector_loop_starts": root_prepare["root_selector_loop_starts"],
                "total_persistent_loop_starts": installation[
                    "worker_persistent_execution_loop_starts"
                ]
                + installation["worker_persistent_dispatch_loop_starts"]
                + root_prepare["root_persistent_dispatch_loop_starts"]
                + root_prepare["root_selector_loop_starts"],
                "connection_establishments": installation["worker_edge_connection_establishments"]
                + root_prepare["root_edge_connection_establishments"],
                "reusable_buffer_allocations": worker_count
                + installation["worker_persistent_dispatch_loop_starts"]
                + runner.root_reusable_buffer_allocations,
                "session_resets": 0,
                "first_operation_latency_ns": first["end_to_end_latency_ns"],
                "cold_latency_including_process_start_ns": process_start_ns
                + topology_construction_ns
                + collective_setup_ns
                + first["end_to_end_latency_ns"],
                "memory_before_total_worker_rss_bytes": memory_before["total_worker_rss_bytes"],
                "memory_after_total_worker_rss_bytes": memory_after["total_worker_rss_bytes"],
                "memory_before": memory_before,
                "memory_after": memory_after,
                "setup_messages": installation["setup_messages"] + root_prepare["messages"],
                "setup_bytes": installation["setup_bytes"] + root_prepare["bytes"],
            }
            setups.append(setup)
            _write_json(output_directory / "raw" / f"lifecycle-n{worker_count:04d}.json", setup)
            attempt.update(
                {
                    "status": "completed",
                    "started_processes": len(workers),
                    "topology_depth": topology.depth,
                    "root_direct_degree": len(topology.root_children),
                    "root_leaf_rpc_count": 0,
                }
            )
        except BaseException as error:
            attempt.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_processes": len(pool.workers),
                }
            )
            _append_jsonl(
                raw_directory / "errors.jsonl",
                {**attempt, "timestamp_unix_ns": time.time_ns()},
            )
        finally:
            if runner is not None:
                attempt["root_teardown"] = runner.close()
            teardown_started_ns = time.perf_counter_ns()
            shutdown = pool.stop()
            attempt["worker_teardown_ns"] = time.perf_counter_ns() - teardown_started_ns
            attempt["shutdown"] = shutdown
            attempt["elapsed_ns"] = time.perf_counter_ns() - scale_started_ns
            attempts.append(attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)

    summary = _summarize(rows, setups, attempts)
    summary["environment"] = _source_identity(repo_root, source_files)
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_json(output_directory / "lifecycle-summary.json", setups)
    _write_summary_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_ints, default=REQUIRED_COUNTS)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--profile", choices=tuple(NETWORK_PROFILES), default="same_host_shaped")
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=20)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    parser.add_argument("--mailbox-depth", type=int, default=2)
    parser.add_argument("--hypothesis-id", default=HYPOTHESIS_ID)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default=ARCHITECTURE)
    args = parser.parse_args(argv)
    summary = run_benchmark(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        branch_factor=args.branch_factor,
        payload_bytes=args.payload_bytes,
        profile=NETWORK_PROFILES[args.profile],
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        mailbox_depth=args.mailbox_depth,
        hypothesis_id=args.hypothesis_id,
        architecture=args.architecture,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_scales_completed"] and summary["all_operations_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
