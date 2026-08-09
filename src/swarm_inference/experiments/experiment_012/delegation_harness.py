"""H012-001 real worker-to-worker delegation experiment harness."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import statistics
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from swarm_inference.experiments.experiment_012.baseline_harness import (
    BASELINE_MODES,
    BaselineRunner,
    WorkerPool,
    WorkerProcess,
    _append_jsonl,
    _percentile,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.microworker_protocol import (
    MAGIC,
    NETWORK_PROFILES,
    PROTOCOL_VERSION,
    LinkProfile,
    PersistentChannel,
    aggregate_digest,
    combine_aggregates,
    combine_latency_histograms,
    combine_transport_attempts,
    encode_message,
    latency_histogram_observe,
    latency_histogram_percentile_ms,
    leaf_aggregate,
    make_delegated_request,
    receive_message,
    retryable_round_trip_error,
    shaped_round_trip,
    worker_contribution,
)

HYPOTHESIS_ID = "H012-001"
DELEGATED_MODE = "delegated_serial"
DELEGATED_MODES = {
    "delegated_serial": "serial",
    "delegated_parallel": "parallel",
    "delegated_parallel_persistent": "parallel",
}
DELEGATED_CONNECTION_POLICIES = {
    "delegated_serial": "ephemeral",
    "delegated_parallel": "ephemeral",
    "delegated_parallel_persistent": "persistent",
}
DELEGATED_MODE_TAGS = {
    "delegated_serial": "m0",
    "delegated_parallel": "m1",
    "delegated_parallel_persistent": "m2",
}
DISCRIMINATING_COUNTS = (8, 32, 128)
DISCRIMINATING_PROFILES = ("same_host_shaped", "moderate_wan_shaped")


@dataclass(slots=True)
class DelegatedNode:
    worker: WorkerProcess
    parent_worker: str
    children: list[DelegatedNode] = field(default_factory=list)
    partition_start: int = 0
    partition_end: int = 0
    partition_worker_indices: tuple[int, ...] = ()
    subtree_worker_count: int = 1

    @property
    def ordering_key(self) -> str:
        return self.worker.worker_id

    @property
    def depth(self) -> int:
        return 1 + max((child.depth for child in self.children), default=0)

    def child_route(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker.worker_id,
            "worker_index": self.worker.worker_index,
            "process_id": self.worker.process_id,
            "endpoint": self.worker.endpoint,
            "ordering_key": self.ordering_key,
            "assigned_child_worker_ids": [child.worker.worker_id for child in self.children],
            "partition_start": self.partition_start,
            "partition_end": self.partition_end,
            "partition_worker_indices": list(self.partition_worker_indices),
            "subtree_worker_count": self.subtree_worker_count,
        }

    def install_node(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker.worker_id,
            "worker_index": self.worker.worker_index,
            "process_id": self.worker.process_id,
            "endpoint": self.worker.endpoint,
            "ordering_key": self.ordering_key,
            "parent_worker": self.parent_worker,
            "partition_start": self.partition_start,
            "partition_end": self.partition_end,
            "partition_worker_indices": list(self.partition_worker_indices),
            "subtree_worker_count": self.subtree_worker_count,
            "children": [child.child_route() for child in self.children],
        }


@dataclass(frozen=True, slots=True)
class DelegatedTopology:
    topology_id: str
    route_lease_id: str
    route_generation: int
    branch_factor: int
    root_children: tuple[DelegatedNode, ...]
    nodes: tuple[DelegatedNode, ...]

    @property
    def depth(self) -> int:
        return max((node.depth for node in self.root_children), default=0)


def _balanced_groups(items: list[WorkerProcess], count: int) -> list[list[WorkerProcess]]:
    if count <= 0 or count > len(items):
        raise ValueError("balanced group count is out of range")
    quotient, remainder = divmod(len(items), count)
    groups: list[list[WorkerProcess]] = []
    start = 0
    for index in range(count):
        width = quotient + (1 if index < remainder else 0)
        groups.append(items[start : start + width])
        start += width
    return groups


def build_delegated_topology(
    workers: list[WorkerProcess],
    *,
    branch_factor: int,
    topology_id: str,
    route_lease_id: str,
    route_generation: int = 1,
) -> DelegatedTopology:
    if len(workers) < 2:
        raise ValueError("delegated topology requires at least two workers")
    if branch_factor < 2 or branch_factor > 32:
        raise ValueError("branch factor must be between two and 32")
    ordered = sorted(workers, key=lambda worker: worker.worker_index)
    all_nodes: list[DelegatedNode] = []

    def build(group: list[WorkerProcess], parent_worker: str) -> DelegatedNode:
        worker = group[0]
        node = DelegatedNode(
            worker=worker,
            parent_worker=parent_worker,
            partition_start=min(item.worker_index for item in group),
            partition_end=max(item.worker_index for item in group) + 1,
            partition_worker_indices=tuple(sorted(item.worker_index for item in group)),
            subtree_worker_count=len(group),
        )
        all_nodes.append(node)
        remaining = group[1:]
        if remaining:
            child_count = min(branch_factor, len(remaining))
            node.children = [
                build(child_group, worker.worker_id)
                for child_group in _balanced_groups(remaining, child_count)
            ]
        return node

    root_degree = min(branch_factor, len(ordered) // 2)
    root_children = tuple(
        build(group, "stage-owner") for group in _balanced_groups(ordered, root_degree)
    )
    topology = DelegatedTopology(
        topology_id=topology_id,
        route_lease_id=route_lease_id,
        route_generation=route_generation,
        branch_factor=branch_factor,
        root_children=root_children,
        nodes=tuple(all_nodes),
    )
    if len(topology.nodes) != len(workers):
        raise AssertionError("delegated topology lost or duplicated a worker")
    if any(not node.children for node in topology.root_children):
        raise AssertionError("root must not contact a topology leaf")
    if len(topology.root_children) > branch_factor:
        raise AssertionError("root degree exceeds the branch factor")
    if any(len(node.children) > branch_factor for node in topology.nodes):
        raise AssertionError("intermediate degree exceeds the branch factor")
    return topology


def build_capacity_parent_topology(
    workers: list[WorkerProcess],
    *,
    branch_factor: int,
    topology_id: str,
    route_lease_id: str,
    route_generation: int = 1,
) -> DelegatedTopology:
    """Keep the balanced shape but assign faster declared workers to shallower roles."""

    topology = build_delegated_topology(
        workers,
        branch_factor=branch_factor,
        topology_id=topology_id,
        route_lease_id=route_lease_id,
        route_generation=route_generation,
    )
    scores = [float(worker.runtime_profile.get("capacity_score", 1.0)) for worker in workers]
    if len(set(scores)) <= 1:
        return topology

    breadth_first: list[DelegatedNode] = []
    level = list(topology.root_children)
    while level:
        breadth_first.extend(level)
        level = [child for node in level for child in node.children]
    assigned = sorted(
        workers,
        key=lambda worker: (
            -float(worker.runtime_profile.get("capacity_score", 1.0)),
            float(worker.runtime_profile.get("compute_delay_ms", 0.0)),
            worker.worker_id,
        ),
    )
    for node, worker in zip(breadth_first, assigned, strict=True):
        node.worker = worker

    def refresh(node: DelegatedNode, parent_worker: str) -> tuple[int, ...]:
        node.parent_worker = parent_worker
        indices = [node.worker.worker_index]
        for child in node.children:
            indices.extend(refresh(child, node.worker.worker_id))
        node.partition_worker_indices = tuple(sorted(indices))
        node.partition_start = min(indices)
        node.partition_end = max(indices) + 1
        node.subtree_worker_count = len(indices)
        return node.partition_worker_indices

    partitions = [refresh(node, "stage-owner") for node in topology.root_children]
    flattened = [index for partition in partitions for index in partition]
    if sorted(flattened) != sorted(worker.worker_index for worker in workers):
        raise AssertionError("capacity topology lost or duplicated a worker partition")
    return topology


def topology_record(topology: DelegatedTopology) -> dict[str, Any]:
    def value(node: DelegatedNode) -> dict[str, Any]:
        return {
            **node.install_node(),
            "children": [value(child) for child in node.children],
        }

    return {
        "topology_id": topology.topology_id,
        "route_lease_id": topology.route_lease_id,
        "route_generation": topology.route_generation,
        "branch_factor": topology.branch_factor,
        "root_direct_degree": len(topology.root_children),
        "root_leaf_rpc_count": sum(not node.children for node in topology.root_children),
        "hierarchy_depth": topology.depth,
        "worker_count": len(topology.nodes),
        "root_children": [value(node) for node in topology.root_children],
    }


def _control_round_trip(
    worker: WorkerProcess, message: dict[str, Any], timeout_s: float = 10.0
) -> tuple[dict[str, Any], dict[str, int]]:
    host, port_text = worker.endpoint.rsplit(":", 1)
    framed = encode_message(message)
    started_ns = time.perf_counter_ns()
    connection = socket.create_connection((host, int(port_text)), timeout=timeout_s)
    try:
        connection.settimeout(timeout_s)
        connection.sendall(framed)
        response, response_bytes = receive_message(connection)
    finally:
        connection.close()
    if response.get("kind") == "error":
        raise RuntimeError(str(response.get("error", "microworker control failed")))
    return response, {
        "request_bytes": len(framed),
        "response_bytes": response_bytes,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }


def install_topology(topology: DelegatedTopology) -> dict[str, Any]:
    started_ns = time.perf_counter_ns()
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lock = threading.Lock()

    def install(node: DelegatedNode) -> None:
        message = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "install_topology",
            "topology_id": topology.topology_id,
            "route_lease_id": topology.route_lease_id,
            "route_generation": topology.route_generation,
            "node": node.install_node(),
        }
        try:
            response, metrics = _control_round_trip(node.worker, message)
            if (
                response.get("topology_id") != topology.topology_id
                or int(response.get("route_generation", 0)) != topology.route_generation
            ):
                raise ValueError("worker acknowledged a different topology identity")
            with lock:
                records.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "process_id": node.worker.process_id,
                        "endpoint": node.worker.endpoint,
                        **metrics,
                    }
                )
        except BaseException as error:
            with lock:
                errors.append(
                    {
                        "worker_id": node.worker.worker_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    with ThreadPoolExecutor(max_workers=min(32, len(topology.nodes))) as executor:
        futures = [executor.submit(install, node) for node in topology.nodes]
        for future in futures:
            future.result()
    result = {
        "topology_id": topology.topology_id,
        "route_lease_id": topology.route_lease_id,
        "route_generation": topology.route_generation,
        "status": "ok" if not errors and len(records) == len(topology.nodes) else "failed",
        "worker_count": len(topology.nodes),
        "root_setup_direct_degree": len(topology.nodes),
        "root_setup_messages": 2 * len(records),
        "root_setup_bytes": sum(
            record["request_bytes"] + record["response_bytes"] for record in records
        ),
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "records": sorted(records, key=lambda record: record["worker_id"]),
        "errors": errors,
    }
    if result["status"] != "ok":
        raise RuntimeError(f"topology installation failed: {errors}")
    return result


class DelegatedRunner:
    def __init__(
        self,
        *,
        topology: DelegatedTopology,
        output_directory: Path,
        payload_bytes: int,
        operation_deadline_s: float,
        maximum_root_concurrency: int,
        cycle_id: str = HYPOTHESIS_ID,
    ) -> None:
        self.topology = topology
        self.output_directory = output_directory
        self.payload_bytes = payload_bytes
        self.operation_deadline_s = operation_deadline_s
        self.maximum_root_concurrency = maximum_root_concurrency
        self.cycle_id = cycle_id
        self.trace_path = output_directory / "traces" / "root.jsonl"
        self.observation_path = output_directory / "raw" / "delegated-root-observations.jsonl"
        self.trace_lock = threading.Lock()
        self.channels_lock = threading.Lock()
        self.channels: dict[str, PersistentChannel] = {}
        self.expected = combine_aggregates(
            [
                leaf_aggregate(node.worker.worker_id, worker_contribution(node.worker.worker_index))
                for node in topology.nodes
            ]
        )

    def _persistent_channel(self, node: DelegatedNode) -> PersistentChannel:
        with self.channels_lock:
            channel = self.channels.get(node.worker.worker_id)
            if channel is None:
                channel = PersistentChannel(
                    node.worker.endpoint, "stage-owner", node.worker.worker_id
                )
                self.channels[node.worker.worker_id] = channel
            return channel

    def close(self) -> None:
        with self.channels_lock:
            channels = list(self.channels.values())
            self.channels.clear()
        for channel in channels:
            channel.close()

    def _root_child_request(
        self,
        *,
        node: DelegatedNode,
        operation_id: str,
        execution_generation: int,
        profile: LinkProfile,
        deadline_unix_ns: int,
        dispatch_policy: str,
        connection_policy: str,
        retry_policy: dict[str, Any],
        fault_control: dict[str, Any],
    ) -> tuple[DelegatedNode, dict[str, Any], dict[str, Any]]:
        request_id = f"{operation_id}:{node.worker.worker_id}"
        request = make_delegated_request(
            request_id=request_id,
            operation_id=operation_id,
            execution_generation=execution_generation,
            parent_worker="stage-owner",
            worker_id=node.worker.worker_id,
            worker_index=node.worker.worker_index,
            assigned_child_workers=[child.worker.worker_id for child in node.children],
            partition_start=node.partition_start,
            partition_end=node.partition_end,
            partition_worker_indices=list(node.partition_worker_indices),
            subtree_worker_count=node.subtree_worker_count,
            deadline_unix_ns=deadline_unix_ns,
            route_lease_id=self.topology.route_lease_id,
            ordering_key=node.ordering_key,
            payload_bytes=self.payload_bytes,
            trace_id=operation_id,
            parent_span_id="root",
            span_id=f"root-to-{node.worker.worker_id}",
            network_profile=profile,
            dispatch_policy=dispatch_policy,
            connection_policy=connection_policy,
            route_generation=self.topology.route_generation,
            retry_policy=retry_policy,
            fault_control=fault_control,
        )
        maximum_attempts = int(retry_policy.get("max_attempts", 0))
        backoff_ms = float(retry_policy.get("backoff_ms", 0.0))
        attempt_metrics: list[dict[str, Any]] = []
        response: dict[str, Any] | None = None
        final_error: BaseException | None = None
        for attempt in range(maximum_attempts + 1):
            timeout_s = max(0.001, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
            try:
                if connection_policy == "persistent":
                    response, transport = self._persistent_channel(node).round_trip(
                        message=request, profile=profile, timeout_s=timeout_s, attempt=attempt
                    )
                else:
                    response, transport = shaped_round_trip(
                        endpoint=node.worker.endpoint,
                        message=request,
                        profile=profile,
                        timeout_s=timeout_s,
                        sender_id="stage-owner",
                        receiver_id=node.worker.worker_id,
                        attempt=attempt,
                    )
                if (
                    response.get("request_id") != request_id
                    or response.get("operation_id") != operation_id
                    or int(response.get("execution_generation", 0)) != execution_generation
                ):
                    raise RuntimeError("root received a stale or mismatched delegated response")
                aggregate = dict(response["aggregate"])
                if response.get("aggregate_digest") != aggregate_digest(aggregate):
                    raise RuntimeError("root received a delegated aggregate with an invalid digest")
                attempt_metrics.append(transport)
                final_error = None
                break
            except BaseException as error:
                final_error = error
                failure_transport = dict(getattr(error, "metrics", {}))
                failure_transport.setdefault("attempt", attempt)
                attempt_metrics.append(failure_transport)
                will_retry = attempt < maximum_attempts and retryable_round_trip_error(error)
                _append_jsonl(
                    self.trace_path,
                    {
                        "event": "root_delegated_round_trip_failed",
                        "hypothesis_id": self.cycle_id,
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "sender_worker_id": "stage-owner",
                        "receiver_worker_id": node.worker.worker_id,
                        "sender_process_id": os.getpid(),
                        "receiver_process_id": node.worker.process_id,
                        "receiver_is_leaf": not node.children,
                        "timestamp_unix_ns": time.time_ns(),
                        "attempt": attempt,
                        "will_retry": will_retry,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "transport_metrics": failure_transport,
                    },
                    self.trace_lock,
                )
                if not will_retry:
                    break
                _append_jsonl(
                    self.trace_path,
                    {
                        "event": "root_child_retry",
                        "hypothesis_id": self.cycle_id,
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "sender_worker_id": "stage-owner",
                        "receiver_worker_id": node.worker.worker_id,
                        "next_attempt": attempt + 1,
                        "timestamp_unix_ns": time.time_ns(),
                    },
                    self.trace_lock,
                )
                if backoff_ms:
                    time.sleep(backoff_ms / 1_000.0)
        transport = combine_transport_attempts(attempt_metrics)
        if final_error is not None or response is None:
            if final_error is None:
                final_error = RuntimeError("root child retry exhausted")
            final_error.root_transport_metrics = transport  # type: ignore[attr-defined]
            raise final_error
        aggregate = dict(response["aggregate"])
        event = {
            "event": "root_delegated_round_trip",
            "hypothesis_id": self.cycle_id,
            "dispatch_policy": dispatch_policy,
            "connection_policy": connection_policy,
            "operation_id": operation_id,
            "request_id": request_id,
            "sender_worker_id": "stage-owner",
            "receiver_worker_id": node.worker.worker_id,
            "sender_process_id": os.getpid(),
            "receiver_process_id": node.worker.process_id,
            "receiver_is_leaf": not node.children,
            "timestamp_unix_ns": time.time_ns(),
            **transport,
        }
        _append_jsonl(self.trace_path, event, self.trace_lock)
        return node, response, transport

    def run_trial(
        self,
        *,
        trial_index: int,
        warmup: bool,
        execution_generation: int,
        profile: LinkProfile,
        mode: str = DELEGATED_MODE,
        operation_id_override: str | None = None,
        retry_policy: dict[str, Any] | None = None,
        fault_control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if mode not in DELEGATED_MODES:
            raise ValueError(f"unsupported delegated mode {mode!r}")
        dispatch_policy = DELEGATED_MODES[mode]
        connection_policy = DELEGATED_CONNECTION_POLICIES[mode]
        operation_id = operation_id_override or (
            f"{self.cycle_id.lower()}-{DELEGATED_MODE_TAGS[mode]}-"
            f"n{len(self.topology.nodes)}-{profile.name}-"
            f"{'warmup' if warmup else 'trial'}-{trial_index}-{uuid4().hex[:8]}"
        )
        deadline_unix_ns = time.time_ns() + int(self.operation_deadline_s * 1_000_000_000)
        wall_started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        results: list[tuple[DelegatedNode, dict[str, Any], dict[str, Any]]] = []
        failures: list[dict[str, Any]] = []
        dispatch_started_ns = time.perf_counter_ns()
        with ThreadPoolExecutor(
            max_workers=min(self.maximum_root_concurrency, len(self.topology.root_children))
        ) as executor:
            futures: dict[Future[Any], DelegatedNode] = {
                executor.submit(
                    self._root_child_request,
                    node=node,
                    operation_id=operation_id,
                    execution_generation=execution_generation,
                    profile=profile,
                    deadline_unix_ns=deadline_unix_ns,
                    dispatch_policy=dispatch_policy,
                    connection_policy=connection_policy,
                    retry_policy=dict(retry_policy or {"max_attempts": 0, "backoff_ms": 0.0}),
                    fault_control=dict(fault_control or {}),
                ): node
                for node in self.topology.root_children
            }
            scheduler_dispatch_ns = time.perf_counter_ns() - dispatch_started_ns
            for future in as_completed(futures):
                node = futures[future]
                try:
                    results.append(future.result())
                except BaseException as error:
                    failures.append(
                        {
                            "worker_id": node.worker.worker_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "transport_metrics": dict(getattr(error, "root_transport_metrics", {})),
                        }
                    )
        root_cpu_ns = time.process_time_ns() - cpu_started_ns
        latency_ns = time.perf_counter_ns() - wall_started_ns
        status = "failed" if failures else "ok"
        actual = (
            combine_aggregates([dict(result[1]["aggregate"]) for result in results])
            if results
            else None
        )
        if status == "ok" and actual != self.expected:
            status = "incorrect"
            failures.append(
                {
                    "error_type": "CorrectnessError",
                    "error": "delegated aggregate or contribution proof differs from reference",
                }
            )
        failed_transports = [dict(item["transport_metrics"]) for item in failures]
        root_transports = [result[2] for result in results] + failed_transports
        root_request_bytes = sum(int(item.get("request_bytes", 0)) for item in root_transports)
        root_response_bytes = sum(int(item.get("response_bytes", 0)) for item in root_transports)
        subtree_metrics = [dict(result[1]["subtree_metrics"]) for result in results]
        all_edge_histogram: dict[str, int] = {}
        leaf_histogram: dict[str, int] = {}
        for result, metrics in zip(results, subtree_metrics, strict=True):
            root_histogram: dict[str, int] = {}
            latency_histogram_observe(root_histogram, int(result[2]["elapsed_ns"]))
            all_edge_histogram = combine_latency_histograms(
                all_edge_histogram,
                root_histogram,
                dict(metrics["latency_histogram"]),
            )
            leaf_histogram = combine_latency_histograms(
                leaf_histogram, dict(metrics["leaf_latency_histogram"])
            )
        system_messages = sum(
            int(result[2]["messages_sent"]) + int(result[2]["messages_received"])
            for result in results
        ) + sum(int(metrics["total_messages"]) for metrics in subtree_metrics)
        system_bytes = (
            root_request_bytes
            + root_response_bytes
            + sum(int(metrics["total_bytes"]) for metrics in subtree_metrics)
        )
        hierarchy_depth = max(
            (int(metrics["hierarchy_depth"]) for metrics in subtree_metrics), default=0
        )
        reduction_depth = (
            1 + max((int(metrics["reduction_depth"]) for metrics in subtree_metrics), default=0)
            if results
            else 0
        )
        critical_path_sync_points = (
            1
            + max(
                (int(metrics["critical_path_sync_points"]) for metrics in subtree_metrics),
                default=0,
            )
            if results
            else 0
        )
        row = {
            "schema_version": "1.0",
            "experiment_id": "012",
            "cycle_id": self.cycle_id,
            "mode": mode,
            "dispatch_policy": dispatch_policy,
            "connection_policy": connection_policy,
            "worker_count": len(self.topology.nodes),
            "branch_factor": self.topology.branch_factor,
            "network_profile": profile.name,
            "payload_bytes": self.payload_bytes,
            "trial_index": trial_index,
            "warmup": warmup,
            "operation_id": operation_id,
            "execution_generation": execution_generation,
            "route_generation": self.topology.route_generation,
            "status": status,
            "correctness": status == "ok",
            "expected": self.expected,
            "actual": actual,
            "expected_digest": aggregate_digest(self.expected),
            "actual_digest": aggregate_digest(actual) if actual is not None else None,
            "result_published": status == "ok",
            "partial_aggregate": actual if status != "ok" else None,
            "root_rpc_count": sum(int(item.get("attempt_count", 1)) for item in root_transports),
            "root_leaf_rpc_count": sum(not result[0].children for result in results),
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
            "root_direct_degree": len(self.topology.root_children),
            "root_serial_waits": len(self.topology.root_children),
            "root_coordinator_waits": 1 if self.topology.root_children else 0,
            "root_cpu_ns": root_cpu_ns,
            "scheduler_dispatch_ns": scheduler_dispatch_ns,
            "total_messages": system_messages,
            "total_bytes": system_bytes,
            "worker_to_worker_rpc_count": sum(
                int(metrics["worker_to_worker_rpc_count"]) for metrics in subtree_metrics
            ),
            "leaf_rpc_count": sum(int(metrics["leaf_rpc_count"]) for metrics in subtree_metrics),
            "hierarchy_depth": hierarchy_depth,
            "observed_network_depth": hierarchy_depth,
            "fanout_depth": hierarchy_depth,
            "reduction_depth": reduction_depth,
            "critical_path_sync_points": critical_path_sync_points,
            "worker_serial_waits": sum(int(metrics["serial_waits"]) for metrics in subtree_metrics),
            "worker_barrier_waits": sum(
                int(metrics["barrier_waits"]) for metrics in subtree_metrics
            ),
            "parallel_dispatch_nodes": sum(
                int(metrics["parallel_dispatch_nodes"]) for metrics in subtree_metrics
            ),
            "end_to_end_latency_ns": latency_ns,
            "end_to_end_latency_ms": latency_ns / 1_000_000,
            "throughput_ops_s": 1_000_000_000 / latency_ns if latency_ns else 0.0,
            "leaf_latency_p50_ms": latency_histogram_percentile_ms(leaf_histogram, 50),
            "leaf_latency_p95_ms": latency_histogram_percentile_ms(leaf_histogram, 95),
            "leaf_latency_p99_ms": latency_histogram_percentile_ms(leaf_histogram, 99),
            "edge_latency_p50_ms": latency_histogram_percentile_ms(all_edge_histogram, 50),
            "edge_latency_p95_ms": latency_histogram_percentile_ms(all_edge_histogram, 95),
            "edge_latency_p99_ms": latency_histogram_percentile_ms(all_edge_histogram, 99),
            "root_connection_count": sum(
                int(item.get("new_connection_count", 0)) for item in root_transports
            ),
            "total_connection_count": sum(
                int(result[2]["new_connection_count"]) for result in results
            )
            + sum(int(metrics["connection_count"]) for metrics in subtree_metrics),
            "total_connection_ns": sum(int(result[2]["connect_ns"]) for result in results)
            + sum(int(metrics["connection_ns"]) for metrics in subtree_metrics),
            "total_worker_cpu_ns": sum(
                int(metrics["worker_cpu_ns"]) for metrics in subtree_metrics
            ),
            "simulated_compute_delay_ms_sum": sum(
                float(metrics["simulated_compute_delay_ms_sum"]) for metrics in subtree_metrics
            ),
            "critical_path_compute_delay_ms": max(
                (float(metrics["critical_path_compute_delay_ms"]) for metrics in subtree_metrics),
                default=0.0,
            ),
            "worker_profile_counts": {
                name: sum(
                    int(dict(metrics["worker_profile_counts"]).get(name, 0))
                    for metrics in subtree_metrics
                )
                for name in sorted(
                    {
                        str(name)
                        for metrics in subtree_metrics
                        for name in dict(metrics["worker_profile_counts"])
                    }
                )
            },
            "intermediate_reductions": 1
            + sum(int(metrics["intermediate_reductions"]) for metrics in subtree_metrics),
            "maximum_queue_depth": max(
                (int(metrics["maximum_queue_depth"]) for metrics in subtree_metrics), default=0
            ),
            "retries": sum(int(item.get("retry_count", 0)) for item in root_transports)
            + sum(int(metrics["retries"]) for metrics in subtree_metrics),
            "failures": len(failures)
            + sum(int(metrics["failures"]) for metrics in subtree_metrics),
            "stragglers": sum(int(metrics["stragglers"]) for metrics in subtree_metrics),
            "duplicated_work": sum(int(metrics["duplicated_work"]) for metrics in subtree_metrics),
            "unexpected_serialization": dispatch_policy == "serial",
            "failure_details": failures,
            "retry_policy": dict(retry_policy or {"max_attempts": 0, "backoff_ms": 0.0}),
            "fault_control": dict(fault_control or {}),
            "measured_unix_ns": time.time_ns(),
        }
        for node, _response, transport in results:
            _append_jsonl(
                self.observation_path,
                {
                    "operation_id": operation_id,
                    "worker_count": len(self.topology.nodes),
                    "network_profile": profile.name,
                    "worker_id": node.worker.worker_id,
                    "worker_process_id": node.worker.process_id,
                    "warmup": warmup,
                    **transport,
                },
                self.trace_lock,
            )
        return row


def _summarize(
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    *,
    hypothesis_id: str = HYPOTHESIS_ID,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    profiles = sorted({str(row["network_profile"]) for row in rows})
    modes = sorted({str(row["mode"]) for row in rows})
    for profile in profiles:
        for worker_count in sorted({int(row["worker_count"]) for row in rows}):
            for mode in modes:
                selected = [
                    row
                    for row in rows
                    if row["network_profile"] == profile
                    and row["worker_count"] == worker_count
                    and row["mode"] == mode
                    and not row["warmup"]
                    and row["status"] == "ok"
                    and row["correctness"]
                ]
                if not selected:
                    continue
                summaries.append(
                    {
                        "network_profile": profile,
                        "worker_count": worker_count,
                        "mode": mode,
                        "successful_trials": len(selected),
                        "root_messages_median": statistics.median(
                            row["root_messages_total"] for row in selected
                        ),
                        "root_bytes_median": statistics.median(
                            row["root_bytes_total"] for row in selected
                        ),
                        "root_serial_waits_median": statistics.median(
                            row["root_serial_waits"] for row in selected
                        ),
                        "root_direct_degree_median": statistics.median(
                            row["root_direct_degree"] for row in selected
                        ),
                        "root_leaf_rpc_count_median": statistics.median(
                            row["root_leaf_rpc_count"] for row in selected
                        ),
                        "root_cpu_ms_median": statistics.median(
                            row["root_cpu_ns"] / 1_000_000 for row in selected
                        ),
                        "end_to_end_latency_p50_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 50
                        ),
                        "end_to_end_latency_p95_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 95
                        ),
                        "end_to_end_latency_p99_ms": _percentile(
                            [row["end_to_end_latency_ms"] for row in selected], 99
                        ),
                        "throughput_ops_s_median": statistics.median(
                            row["throughput_ops_s"] for row in selected
                        ),
                        "hierarchy_depth_median": statistics.median(
                            row["hierarchy_depth"] for row in selected
                        ),
                        "critical_path_sync_points_median": statistics.median(
                            row["critical_path_sync_points"] for row in selected
                        ),
                        "total_messages_median": statistics.median(
                            row["total_messages"] for row in selected
                        ),
                        "total_bytes_median": statistics.median(
                            row["total_bytes"] for row in selected
                        ),
                    }
                )
    return {
        "schema_version": "1.0",
        "hypothesis_id": hypothesis_id,
        "evidence_classification": "measured synthetic workload over independent loopback processes and shaped links",
        "attempts": attempts,
        "summaries": summaries,
        "failed_trials": [row for row in rows if row["status"] != "ok"],
        "generated_unix_ns": time.time_ns(),
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = list(summary["summaries"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_h012_001(
    *,
    output_directory: Path,
    worker_counts: tuple[int, ...],
    profiles: tuple[str, ...],
    branch_factor: int,
    maximum_root_concurrency: int,
    payload_bytes: int,
    warmup_trials: int,
    measured_trials: int,
    operation_deadline_s: float,
    startup_deadline_s: float,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    protocol_script = repo_root / "src" / "swarm_inference" / "microworker_protocol.py"
    baseline_harness = Path(__file__).with_name("baseline_harness.py")
    source_files = [protocol_script, baseline_harness, Path(__file__).resolve()]
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    trials_path = raw_directory / "trials.jsonl"
    errors_path = raw_directory / "errors.jsonl"
    _write_json(
        output_directory / "source-identity.json", _source_identity(repo_root, source_files)
    )
    snapshot_directory = raw_directory / "source-snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        shutil.copy2(source_file, snapshot_directory / source_file.name)
    hypothesis_path = output_directory / "hypothesis.json"
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )
    rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    execution_generation = 1
    for worker_count in worker_counts:
        scale_started_ns = time.perf_counter_ns()
        scale_directory = raw_directory / f"workers-{worker_count:04d}"
        pool = WorkerPool(
            count=worker_count,
            directory=scale_directory / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
        )
        attempt: dict[str, Any] = {
            "worker_count": worker_count,
            "status": "starting",
            "attempted_unix_ns": time.time_ns(),
        }
        try:
            workers = pool.start()
            topology = build_delegated_topology(
                workers,
                branch_factor=branch_factor,
                topology_id=f"h012-001-topology-n{worker_count}",
                route_lease_id=f"h012-001-lease-n{worker_count}",
            )
            topology_payload = topology_record(topology)
            _write_json(
                output_directory / "topologies" / f"workers-{worker_count:04d}.json",
                topology_payload,
            )
            installation = install_topology(topology)
            _write_json(
                output_directory / "topology-installation" / f"workers-{worker_count:04d}.json",
                installation,
            )
            delegated = DelegatedRunner(
                topology=topology,
                output_directory=output_directory,
                payload_bytes=payload_bytes,
                operation_deadline_s=operation_deadline_s,
                maximum_root_concurrency=maximum_root_concurrency,
            )
            for profile_name in profiles:
                profile = NETWORK_PROFILES[profile_name]
                controls = BaselineRunner(
                    workers=workers,
                    output_directory=output_directory,
                    branch_factor=branch_factor,
                    maximum_root_concurrency=maximum_root_concurrency,
                    payload_bytes=payload_bytes,
                    operation_deadline_s=operation_deadline_s,
                    network_profile=profile,
                    cycle_id=HYPOTHESIS_ID,
                )
                for warmup_index in range(warmup_trials):
                    for mode in (*BASELINE_MODES, DELEGATED_MODE):
                        row = (
                            delegated.run_trial(
                                trial_index=warmup_index,
                                warmup=True,
                                execution_generation=execution_generation,
                                profile=profile,
                            )
                            if mode == DELEGATED_MODE
                            else controls.run_trial(
                                mode=mode,
                                trial_index=warmup_index,
                                warmup=True,
                                generation=execution_generation,
                            )
                        )
                        execution_generation += 1
                        rows.append(row)
                        _append_jsonl(trials_path, row)
                        if row["status"] != "ok":
                            _append_jsonl(errors_path, row)
                for trial_index in range(measured_trials):
                    for mode in (*BASELINE_MODES, DELEGATED_MODE):
                        row = (
                            delegated.run_trial(
                                trial_index=trial_index,
                                warmup=False,
                                execution_generation=execution_generation,
                                profile=profile,
                            )
                            if mode == DELEGATED_MODE
                            else controls.run_trial(
                                mode=mode,
                                trial_index=trial_index,
                                warmup=False,
                                generation=execution_generation,
                            )
                        )
                        execution_generation += 1
                        rows.append(row)
                        _append_jsonl(trials_path, row)
                        if row["status"] != "ok":
                            _append_jsonl(errors_path, row)
            attempt.update(
                {
                    "status": "completed",
                    "started_processes": len(workers),
                    "peak_worker_rss_bytes": pool.sample_memory(),
                    "topology_depth": topology.depth,
                    "root_direct_degree": len(topology.root_children),
                    "root_leaf_rpc_count": sum(
                        not node.children for node in topology.root_children
                    ),
                }
            )
        except BaseException as error:
            attempt.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "started_processes": len(pool.workers),
                }
            )
            _append_jsonl(errors_path, {"event": "scale_failed", **attempt})
        finally:
            attempt["shutdown"] = pool.stop()
            attempt["elapsed_ms"] = (time.perf_counter_ns() - scale_started_ns) / 1_000_000
            attempts.append(attempt)
            _write_json(raw_directory / "scale-attempts.json", attempts)
    summary = _summarize(rows, attempts)
    _write_json(output_directory / "benchmark-summary.json", summary)
    _write_summary_csv(output_directory / "benchmark-summary.csv", summary)
    return summary


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item < 2 for item in parsed):
        raise argparse.ArgumentTypeError("counts must contain integers of at least two")
    return parsed


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(parsed) - NETWORK_PROFILES.keys()
    if not parsed or unknown:
        raise argparse.ArgumentTypeError(f"unknown network profiles: {sorted(unknown)}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_parse_csv_ints, default=DISCRIMINATING_COUNTS)
    parser.add_argument("--profiles", type=_parse_csv_strings, default=DISCRIMINATING_PROFILES)
    parser.add_argument("--branch-factor", type=int, default=8)
    parser.add_argument("--maximum-root-concurrency", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=256)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument("--operation-deadline-s", type=float, default=30.0)
    parser.add_argument("--startup-deadline-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    summary = run_h012_001(
        output_directory=args.output.resolve(),
        worker_counts=args.counts,
        profiles=args.profiles,
        branch_factor=args.branch_factor,
        maximum_root_concurrency=args.maximum_root_concurrency,
        payload_bytes=args.payload_bytes,
        warmup_trials=args.warmup_trials,
        measured_trials=args.measured_trials,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return (
        1
        if summary["failed_trials"]
        or any(attempt["status"] != "completed" for attempt in summary["attempts"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DELEGATED_CONNECTION_POLICIES",
    "DELEGATED_MODE",
    "DELEGATED_MODES",
    "DELEGATED_MODE_TAGS",
    "DISCRIMINATING_COUNTS",
    "DISCRIMINATING_PROFILES",
    "DelegatedNode",
    "DelegatedRunner",
    "DelegatedTopology",
    "build_capacity_parent_topology",
    "build_delegated_topology",
    "install_topology",
    "run_h012_001",
    "topology_record",
]
