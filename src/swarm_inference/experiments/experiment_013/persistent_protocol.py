"""Persistent event-driven microworker protocol for Experiment 013.

The transport remains real length-prefixed TCP between independent processes.
Unlike the Experiment 012 delegated path, topology installation starts one
long-lived execution mailbox and one long-lived dispatcher for every owned child
edge.  Warm operations enqueue work on those loops; they do not create threads,
executors, connections, or topology objects.

This file can be executed directly with ``python -S`` so the 1,000-process
synthetic study does not import a tensor runtime into every worker.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import queue
import select
import socket
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from swarm_inference.microworker_protocol import (
    MAGIC,
    PROTOCOL_VERSION,
    LinkProfile,
    MicroworkerServer,
    PersistentChannel,
    RoundTripError,
    _deterministic_fraction,
    _shape_delay_s,
    aggregate_digest,
    canonical_json_bytes,
    combine_latency_histograms,
    combine_operation_aggregates,
    combine_transport_attempts,
    encode_message,
    latency_histogram_observe,
    leaf_aggregate,
    parse_endpoint,
    receive_message,
    retry_attempt_timeout_s,
    retryable_round_trip_error,
    worker_contribution,
)

COLLECTIVE_PROTOCOL = "persistent-event-v1"
ARCHITECTURE = "mailbox-v1"
INLINE_ARCHITECTURE = "inline-receive-v2"
SELECTOR_ARCHITECTURE = "selector-fanout-v3"
COMPACT_ARCHITECTURE = "compact-selector-v4"
BUFFERED_ARCHITECTURE = "buffered-selector-v5"
LEAN_ARCHITECTURE = "lean-selector-v6"
ARCHITECTURES = (
    ARCHITECTURE,
    INLINE_ARCHITECTURE,
    SELECTOR_ARCHITECTURE,
    COMPACT_ARCHITECTURE,
    BUFFERED_ARCHITECTURE,
    LEAN_ARCHITECTURE,
)


def _uses_selector(architecture: str) -> bool:
    return architecture in {
        SELECTOR_ARCHITECTURE,
        COMPACT_ARCHITECTURE,
        BUFFERED_ARCHITECTURE,
        LEAN_ARCHITECTURE,
    }


def _uses_compact_wire(architecture: str) -> bool:
    return architecture in {
        COMPACT_ARCHITECTURE,
        BUFFERED_ARCHITECTURE,
        LEAN_ARCHITECTURE,
    }


def _uses_lean_wire(architecture: str) -> bool:
    return architecture == LEAN_ARCHITECTURE


COMPACT_METRIC_FIELDS = (
    "total_messages",
    "total_bytes",
    "worker_to_worker_rpc_count",
    "leaf_rpc_count",
    "hierarchy_depth",
    "reduction_depth",
    "intermediate_reductions",
    "connection_count",
    "connection_ns",
    "worker_cpu_ns",
    "retries",
    "failures",
    "duplicated_work",
    "serialization_ns",
    "queue_delay_ns",
    "local_execution_ns",
    "reduction_ns",
    "dispatch_elapsed_ns",
)

LEAN_METRIC_FIELDS = (
    "total_messages",
    "total_bytes",
    "worker_to_worker_rpc_count",
    "connection_count",
    "connection_ns",
    "retries",
    "failures",
    "duplicated_work",
)


def decode_subtree_metrics(response: dict[str, Any]) -> dict[str, Any]:
    metrics = response.get("subtree_metrics")
    if isinstance(metrics, dict):
        return dict(metrics)
    compact = response.get("m")
    if isinstance(compact, list) and len(compact) == len(COMPACT_METRIC_FIELDS):
        return dict(zip(COMPACT_METRIC_FIELDS, compact, strict=True))
    lean = response.get("lm")
    if isinstance(lean, list) and len(lean) == len(LEAN_METRIC_FIELDS):
        result = {field: 0 for field in COMPACT_METRIC_FIELDS}
        result.update(dict(zip(LEAN_METRIC_FIELDS, lean, strict=True)))
        return result
    raise ValueError("persistent response has no valid subtree telemetry")


def make_collective_request(
    *,
    collective_id: str,
    topology_id: str,
    route_lease_id: str,
    route_generation: int,
    request_id: str,
    operation_id: str,
    execution_generation: int,
    parent_worker: str,
    target_worker: str,
    deadline_unix_ns: int,
    profile: LinkProfile,
    retry_policy: dict[str, Any],
    fault_control: dict[str, Any],
    aggregation: dict[str, Any],
    payload_b64: str,
    workload: dict[str, Any] | None = None,
    architecture: str = ARCHITECTURE,
    operation_digest: str | None = None,
) -> dict[str, Any]:
    """Build the minimal warm envelope; static subtree routing is omitted."""

    message: dict[str, Any] = {
        "magic": MAGIC,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "delegated_execute",
        "collective_protocol": COLLECTIVE_PROTOCOL,
        "architecture": architecture,
        "collective_id": collective_id,
        "topology_id": topology_id,
        "route_lease_id": route_lease_id,
        "route_generation": route_generation,
        "request_id": request_id,
        "operation_id": operation_id,
        "execution_generation": execution_generation,
        "parent_worker": parent_worker,
        "target_worker": target_worker,
        "deadline_unix_ns": deadline_unix_ns,
        "network_profile": {
            "name": profile.name,
            "rtt_ms": profile.rtt_ms,
            "upload_mbps": profile.upload_mbps,
            "download_mbps": profile.download_mbps,
            "jitter_ms": profile.jitter_ms,
            "request_loss_rate": profile.request_loss_rate,
            "temporary_disconnect_rate": profile.temporary_disconnect_rate,
        },
        "retry_policy": dict(retry_policy),
        "cancellation": {"cancelled": False, "epoch": 0},
        "fault_control": dict(fault_control),
        "aggregation": dict(aggregation),
        "connection_policy": "persistent",
        "payload_b64": payload_b64,
    }
    if workload is not None:
        message["workload"] = dict(workload)
    if _uses_compact_wire(architecture):
        compact: dict[str, Any] = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "delegated_execute",
            "collective_protocol": COLLECTIVE_PROTOCOL,
            "collective_id": collective_id,
            "route_lease_id": route_lease_id,
            "route_generation": route_generation,
            "request_id": request_id,
            "operation_id": operation_id,
            "execution_generation": execution_generation,
            "parent_worker": parent_worker,
            "deadline_unix_ns": deadline_unix_ns,
            "connection_policy": "persistent",
            "payload_b64": payload_b64,
        }
        if retry_policy != {"max_attempts": 0, "backoff_ms": 0.0}:
            compact["retry_policy"] = dict(retry_policy)
        if fault_control:
            compact["fault_control"] = dict(fault_control)
        if aggregation != {
            "mode": "exact_int64_sum",
            "ordering": "deterministic_worker_key",
        }:
            compact["aggregation"] = dict(aggregation)
        if workload is not None:
            compact["workload"] = dict(workload)
        if _uses_lean_wire(architecture):
            if not operation_digest:
                raise ValueError("lean warm envelopes require an operation digest")
            compact["od"] = operation_digest
        return compact
    return message


@dataclass(slots=True)
class _MailboxJob:
    message: dict[str, Any]
    profile: LinkProfile | None = None
    enqueued_ns: int = field(default_factory=time.perf_counter_ns)
    started_ns: int = 0
    completed_ns: int = 0
    result: Any = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)


class PersistentFanout:
    """One inline event loop multiplexing persistent sockets to many children.

    The object is created during collective setup and is called only by its
    owning worker's persistent receive/execute loop.  No operation creates a
    thread, executor, task, or socket on the normal warm path.
    """

    def __init__(
        self,
        children: list[dict[str, Any]],
        *,
        sender_id: str,
        trace: Any | None = None,
        metrics_add: Any | None = None,
    ) -> None:
        self.children = {
            str(child["worker_id"]): dict(child)
            for child in sorted(children, key=lambda item: item["ordering_key"])
        }
        self.sender_id = sender_id
        self.trace = trace
        self.metrics_add = metrics_add
        self.connections: dict[str, socket.socket] = {}
        self.last_dispatch_ns = 0
        self.last_select_waits = 0

    def _connect(self, child_id: str, timeout_s: float) -> tuple[socket.socket, int, int]:
        existing = self.connections.get(child_id)
        if existing is not None:
            return existing, 0, 0
        host, port = parse_endpoint(str(self.children[child_id]["endpoint"]))
        started_ns = time.perf_counter_ns()
        connection = socket.create_connection((host, port), timeout=timeout_s)
        connection.settimeout(timeout_s)
        self.connections[child_id] = connection
        connect_ns = time.perf_counter_ns() - started_ns
        if self.metrics_add is not None:
            self.metrics_add("connection_establishments", 1)
        return connection, connect_ns, 1

    def _close_child(self, child_id: str) -> None:
        connection = self.connections.pop(child_id, None)
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()

    @staticmethod
    def _validate_response(
        message: dict[str, Any], response: dict[str, Any], transport: dict[str, Any]
    ) -> None:
        if response.get("kind") == "error":
            raise RoundTripError(
                str(response.get("error", "persistent child failed")),
                metrics=transport,
                response=response,
            )
        if message.get("collective_ping"):
            if response.get("kind") != "collective_pong":
                raise RoundTripError(
                    "persistent selector prewarm returned a non-pong response",
                    metrics=transport,
                    response=response,
                )
            return
        if (
            response.get("request_id") != message.get("request_id")
            or response.get("operation_id") != message.get("operation_id")
            or int(response.get("execution_generation", 0))
            != int(message.get("execution_generation", 0))
        ):
            raise RoundTripError(
                "persistent selector child response identity mismatch",
                metrics=transport,
                response=response,
            )
        aggregate = dict(response["aggregate"])
        if response.get("aggregate_digest") != aggregate_digest(aggregate):
            raise RoundTripError(
                "persistent selector child aggregate digest mismatch",
                metrics=transport,
                response=response,
            )

    def _attempt_many(
        self,
        messages: dict[str, dict[str, Any]],
        *,
        profile: LinkProfile,
        deadline_unix_ns: int,
        attempt: int,
    ) -> tuple[
        dict[str, tuple[dict[str, Any], dict[str, Any]]],
        dict[str, tuple[BaseException, dict[str, Any]]],
    ]:
        attempt_started_ns = time.perf_counter_ns()
        frames: dict[str, bytes] = {}
        serialization_ns: dict[str, int] = {}
        connect_ns: dict[str, int] = {}
        new_connections: dict[str, int] = {}
        sendable: dict[str, socket.socket] = {}
        failures: dict[str, tuple[BaseException, dict[str, Any]]] = {}
        request_delay_s = 0.0
        for child_id, message in messages.items():
            started_ns = time.perf_counter_ns()
            frames[child_id] = encode_message(message)
            serialization_ns[child_id] = time.perf_counter_ns() - started_ns
            seed = f"{message.get('request_id')}|{self.sender_id}|{child_id}|{attempt}"
            if _deterministic_fraction(seed, "loss") < profile.request_loss_rate:
                failure = TimeoutError("simulated shaped-link request loss")
                failures[child_id] = (failure, {"attempt": attempt})
                continue
            if _deterministic_fraction(seed, "disconnect") < profile.temporary_disconnect_rate:
                failure = ConnectionError("simulated shaped-link temporary disconnect")
                failures[child_id] = (failure, {"attempt": attempt})
                continue
            remaining_s = (deadline_unix_ns - time.time_ns()) / 1_000_000_000
            if remaining_s <= 0:
                failures[child_id] = (
                    TimeoutError("persistent selector deadline elapsed before connect"),
                    {"attempt": attempt},
                )
                continue
            try:
                connection, connected_ns, created = self._connect(child_id, remaining_s)
                connect_ns[child_id] = connected_ns
                new_connections[child_id] = created
                sendable[child_id] = connection
                request_delay_s = max(
                    request_delay_s,
                    _shape_delay_s(
                        profile,
                        byte_count=len(frames[child_id]),
                        direction="request",
                        seed=seed,
                    ),
                )
            except BaseException as error:
                failures[child_id] = (error, {"attempt": attempt})
        remaining_s = max(0.0, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
        if request_delay_s:
            time.sleep(min(request_delay_s, remaining_s))
        sent_at_ns = time.perf_counter_ns()
        active: dict[socket.socket, str] = {}
        for child_id, connection in list(sendable.items()):
            try:
                remaining_s = max(0.001, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
                connection.settimeout(remaining_s)
                connection.sendall(frames[child_id])
                active[connection] = child_id
            except BaseException as error:
                metrics = {
                    "request_bytes": 0,
                    "response_bytes": 0,
                    "messages_sent": 0,
                    "messages_received": 0,
                    "connect_ns": connect_ns.get(child_id, 0),
                    "new_connection_count": new_connections.get(child_id, 0),
                    "connection_reused": not bool(new_connections.get(child_id, 0)),
                    "elapsed_ns": time.perf_counter_ns() - attempt_started_ns,
                    "client_cpu_ns": 0,
                    "serialization_ns": serialization_ns[child_id],
                    "profile": profile.name,
                    "attempt": attempt,
                }
                failures[child_id] = (error, metrics)
                self._close_child(child_id)
        received: dict[str, tuple[dict[str, Any], int]] = {}
        select_waits = 0
        while active:
            remaining_s = (deadline_unix_ns - time.time_ns()) / 1_000_000_000
            if remaining_s <= 0:
                for connection, child_id in list(active.items()):
                    failures[child_id] = (
                        TimeoutError("persistent selector response deadline elapsed"),
                        {"attempt": attempt},
                    )
                    self._close_child(child_id)
                    active.pop(connection, None)
                break
            readable, _writable, exceptional = select.select(
                list(active), [], list(active), remaining_s
            )
            select_waits += 1
            for connection in exceptional:
                child_id = active.pop(connection)
                failures[child_id] = (
                    ConnectionError("persistent selector socket became exceptional"),
                    {"attempt": attempt},
                )
                self._close_child(child_id)
            for connection in readable:
                child_id = active.pop(connection)
                try:
                    remaining_s = max(0.001, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
                    connection.settimeout(remaining_s)
                    response, response_bytes = receive_message(connection)
                    received[child_id] = (response, response_bytes)
                except BaseException as error:
                    metrics = {
                        "request_bytes": len(frames[child_id]),
                        "response_bytes": 0,
                        "messages_sent": 1,
                        "messages_received": 0,
                        "connect_ns": connect_ns.get(child_id, 0),
                        "new_connection_count": new_connections.get(child_id, 0),
                        "connection_reused": not bool(new_connections.get(child_id, 0)),
                        "elapsed_ns": time.perf_counter_ns() - attempt_started_ns,
                        "client_cpu_ns": 0,
                        "serialization_ns": serialization_ns[child_id],
                        "profile": profile.name,
                        "attempt": attempt,
                    }
                    failures[child_id] = (error, metrics)
                    self._close_child(child_id)
        self.last_select_waits += select_waits
        response_delay_s = 0.0
        for child_id, (_response, response_bytes) in received.items():
            message = messages[child_id]
            seed = f"{message.get('request_id')}|{self.sender_id}|{child_id}|{attempt}"
            response_delay_s = max(
                response_delay_s,
                _shape_delay_s(
                    profile,
                    byte_count=response_bytes,
                    direction="response",
                    seed=seed,
                ),
            )
        remaining_s = max(0.0, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
        if response_delay_s:
            time.sleep(min(response_delay_s, remaining_s))
        completed_ns = time.perf_counter_ns()
        results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for child_id, (response, response_bytes) in received.items():
            transport = {
                "sender_worker_id": self.sender_id,
                "receiver_worker_id": child_id,
                "request_bytes": len(frames[child_id]),
                "response_bytes": response_bytes,
                "messages_sent": 1,
                "messages_received": 1,
                "connect_ns": connect_ns.get(child_id, 0),
                "new_connection_count": new_connections.get(child_id, 0),
                "connection_reused": not bool(new_connections.get(child_id, 0)),
                "elapsed_ns": completed_ns - attempt_started_ns,
                "client_cpu_ns": 0,
                "serialization_ns": serialization_ns[child_id],
                "profile": profile.name,
                "attempt": attempt,
                "select_waits": select_waits,
                "fanout_send_ns": sent_at_ns - attempt_started_ns,
            }
            try:
                self._validate_response(messages[child_id], response, transport)
                results[child_id] = (response, transport)
            except BaseException as error:
                failures[child_id] = (error, transport)
                # A stale or duplicated frame leaves stream alignment unknown.
                # Discard the session before a bounded retry so the next
                # generation cannot inherit unread bytes.
                self._close_child(child_id)
        self.last_dispatch_ns = sent_at_ns - attempt_started_ns
        return results, failures

    def round_trip_many(
        self,
        messages: dict[str, dict[str, Any]],
        *,
        profile: LinkProfile,
        deadline_unix_ns: int,
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        pending = dict(messages)
        attempt_metrics: dict[str, list[dict[str, Any]]] = {child_id: [] for child_id in messages}
        results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        final_errors: dict[str, BaseException] = {}
        maximum_attempts = max(
            (
                int(dict(message.get("retry_policy", {})).get("max_attempts", 0))
                for message in messages.values()
            ),
            default=0,
        )
        for attempt in range(maximum_attempts + 1):
            if not pending:
                break
            received, failures = self._attempt_many(
                pending,
                profile=profile,
                deadline_unix_ns=deadline_unix_ns,
                attempt=attempt,
            )
            next_pending: dict[str, dict[str, Any]] = {}
            for child_id, (response, transport) in received.items():
                attempt_metrics[child_id].append(transport)
                results[child_id] = (
                    response,
                    combine_transport_attempts(attempt_metrics[child_id]),
                )
                final_errors.pop(child_id, None)
            for child_id, (error, metrics) in failures.items():
                attempt_metrics[child_id].append(metrics)
                message = pending[child_id]
                allowed = int(dict(message.get("retry_policy", {})).get("max_attempts", 0))
                will_retry = attempt < allowed and retryable_round_trip_error(error)
                final_errors[child_id] = error
                if self.trace is not None:
                    self.trace(
                        "persistent_selector_round_trip_failed",
                        operation_id=message.get("operation_id"),
                        request_id=message.get("request_id"),
                        receiver_worker_id=child_id,
                        attempt=attempt,
                        will_retry=will_retry,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                if will_retry:
                    next_pending[child_id] = message
            pending = next_pending
            if pending:
                backoff_ms = max(
                    float(dict(message.get("retry_policy", {})).get("backoff_ms", 0.0))
                    for message in pending.values()
                )
                if backoff_ms:
                    time.sleep(backoff_ms / 1_000.0)
        if pending or final_errors:
            child_id = sorted(pending or final_errors)[0]
            error = final_errors[child_id]
            error.metrics = combine_transport_attempts(attempt_metrics[child_id])
            raise error
        return results

    def close(self) -> None:
        for child_id in list(self.connections):
            self._close_child(child_id)


class _PersistentEdgeLoop:
    """One setup-created child dispatcher and one reusable TCP channel."""

    def __init__(
        self,
        owner: PersistentCollectiveServer,
        child: dict[str, Any],
        *,
        asynchronous: bool = True,
    ) -> None:
        self.owner = owner
        self.child = child
        self.asynchronous = asynchronous
        self.jobs: queue.Queue[_MailboxJob | None] = queue.Queue(maxsize=2)
        self.channel = PersistentChannel(
            str(child["endpoint"]), owner.worker_id, str(child["worker_id"])
        )
        self.thread: threading.Thread | None = None
        if asynchronous:
            self.thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"collective-edge-{owner.worker_id}-{child['worker_id']}",
            )
            self.thread.start()
            owner._collective_add("persistent_dispatch_loop_starts", 1)

    def submit(self, message: dict[str, Any], profile: LinkProfile) -> _MailboxJob:
        if not self.asynchronous:
            raise RuntimeError("direct persistent edge does not own a dispatcher mailbox")
        job = _MailboxJob(message=message, profile=profile)
        self.owner._collective_add("dynamic_envelope_allocations", 1)
        self.owner._collective_max("maximum_dispatch_queue_depth", self.jobs.qsize() + 1)
        self.jobs.put(job)
        return job

    def round_trip_inline(
        self, message: dict[str, Any], profile: LinkProfile
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.asynchronous:
            raise RuntimeError("asynchronous persistent edge must use its mailbox")
        return self._round_trip(message, profile)

    @staticmethod
    def wait(job: _MailboxJob, deadline_unix_ns: int) -> tuple[dict[str, Any], dict[str, Any]]:
        remaining_s = max(0.0, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
        if not job.done.wait(timeout=remaining_s):
            raise TimeoutError("persistent edge mailbox deadline elapsed")
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
        deadline_unix_ns = int(message["deadline_unix_ns"])
        attempts: list[dict[str, Any]] = []
        response: dict[str, Any] | None = None
        final_error: BaseException | None = None
        for attempt in range(maximum_attempts + 1):
            try:
                timeout_s = retry_attempt_timeout_s(
                    deadline_unix_ns=deadline_unix_ns,
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
                            "persistent child response identity mismatch",
                            metrics=transport,
                            response=response,
                        )
                    child_aggregate = dict(response["aggregate"])
                    if response.get("aggregate_digest") != aggregate_digest(child_aggregate):
                        raise RoundTripError(
                            "persistent child aggregate digest mismatch",
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
                self.owner._trace(
                    "persistent_child_round_trip_failed",
                    collective_id=message.get("collective_id"),
                    operation_id=message.get("operation_id"),
                    request_id=message.get("request_id"),
                    receiver_worker_id=self.child["worker_id"],
                    attempt=attempt,
                    will_retry=will_retry,
                    error_type=type(error).__name__,
                    error=str(error),
                )
                if not will_retry:
                    break
                self.owner._collective_add("retry_wakeups", 1)
                if backoff_ms:
                    time.sleep(backoff_ms / 1_000.0)
        combined = combine_transport_attempts(attempts)
        self.owner._collective_add(
            "connection_establishments", int(combined.get("new_connection_count", 0))
        )
        if final_error is not None or response is None:
            error = final_error or RuntimeError("persistent child retry exhausted")
            error.metrics = combined
            raise error
        return response, combined

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                break
            job.started_ns = time.perf_counter_ns()
            self.owner._collective_add("persistent_dispatch_loop_wakeups", 1)
            self.owner._collective_add("dispatch_queue_delay_ns", job.started_ns - job.enqueued_ns)
            try:
                assert job.profile is not None
                job.result = self._round_trip(job.message, job.profile)
            except BaseException as error:
                job.error = error
            finally:
                job.completed_ns = time.perf_counter_ns()
                job.done.set()
        self.owner._collective_add("persistent_dispatch_loop_stops", 1)

    def close(self) -> None:
        if self.thread is not None:
            self.jobs.put(None)
            self.thread.join(timeout=2.0)
        self.channel.close()


class PersistentCollectiveServer(MicroworkerServer):
    """Microworker with setup-created mailbox and child edge loops."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._trace_buffer_enabled = False
        self._trace_buffer_lock = threading.Lock()
        self._trace_buffer: deque[dict[str, Any]] = deque(maxlen=4096)
        self._trace_buffer_dropped = 0
        super().__init__(config)
        self.synthetic_local_aggregate = leaf_aggregate(
            self.worker_id, worker_contribution(self.worker_index)
        )
        self.collective_id: str | None = None
        self.collective_architecture = ARCHITECTURE
        self.collective_profile: LinkProfile | None = None
        self.operation_queue: queue.Queue[_MailboxJob | None] | None = None
        self.execution_thread: threading.Thread | None = None
        self.edge_loops: dict[str, _PersistentEdgeLoop] = {}
        self.selector_fanout: PersistentFanout | None = None
        self.collective_metrics_lock = threading.Lock()
        self.collective_metrics: dict[str, int] = {
            "collective_installs": 0,
            "collective_prepares": 0,
            "collective_resets": 0,
            "topology_rebuilds": 0,
            "persistent_execution_loop_starts": 0,
            "persistent_execution_loop_stops": 0,
            "persistent_execution_loop_wakeups": 0,
            "persistent_receive_loop_starts": 0,
            "persistent_receive_loop_wakeups": 0,
            "persistent_mailbox_loop_wakeups": 0,
            "persistent_dispatch_loop_starts": 0,
            "persistent_dispatch_loop_stops": 0,
            "persistent_dispatch_loop_wakeups": 0,
            "connection_establishments": 0,
            "dynamic_envelope_allocations": 0,
            "reusable_buffer_allocations": 0,
            "reusable_buffer_uses": 0,
            "execution_queue_delay_ns": 0,
            "dispatch_queue_delay_ns": 0,
            "maximum_execution_queue_depth": 0,
            "maximum_dispatch_queue_depth": 0,
            "operations_completed": 0,
            "operations_failed": 0,
            "retry_wakeups": 0,
        }
        self.last_completed_generation = 0
        self.parent_session_prepared = False
        self.collective_operation_lock = threading.Lock()
        self.collective_response_cache: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        if self._trace_buffer_enabled and path == self.trace_path:
            with self._trace_buffer_lock:
                if len(self._trace_buffer) == self._trace_buffer.maxlen:
                    self._trace_buffer_dropped += 1
                self._trace_buffer.append(dict(payload))
            return
        super()._append_jsonl(path, payload)

    def _flush_trace_buffer(self) -> dict[str, int]:
        with self._trace_buffer_lock:
            buffered = list(self._trace_buffer)
            self._trace_buffer.clear()
            dropped = self._trace_buffer_dropped
            self._trace_buffer_dropped = 0
        if buffered:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8", newline="\n") as handle:
                for payload in buffered:
                    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return {"flushed_events": len(buffered), "dropped_events": dropped}

    def serve(self) -> int:
        try:
            return super().serve()
        finally:
            self._flush_trace_buffer()

    def _collective_add(self, field: str, value: int) -> None:
        with self.collective_metrics_lock:
            self.collective_metrics[field] = int(self.collective_metrics.get(field, 0)) + int(value)

    def _collective_max(self, field: str, value: int) -> None:
        with self.collective_metrics_lock:
            self.collective_metrics[field] = max(
                int(self.collective_metrics.get(field, 0)), int(value)
            )

    def _collective_snapshot(self) -> dict[str, int]:
        with self.collective_metrics_lock:
            return dict(self.collective_metrics)

    def _stop_collective(self) -> None:
        loops = list(self.edge_loops.values())
        self.edge_loops.clear()
        for loop in loops:
            loop.close()
        if self.selector_fanout is not None:
            self.selector_fanout.close()
            self.selector_fanout = None
        operation_queue = self.operation_queue
        thread = self.execution_thread
        self.operation_queue = None
        self.execution_thread = None
        if operation_queue is not None:
            operation_queue.put(None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self.collective_id = None

    def _start_collective(self, request: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        if self.collective_id is not None:
            self._collective_add("collective_resets", 1)
            self._stop_collective()
        installed = self._validate_topology_install(request)
        node = dict(installed["node"])
        self.collective_id = str(request["collective_id"])
        self.parent_session_prepared = False
        self.collective_architecture = str(request.get("architecture", ARCHITECTURE))
        if self.collective_architecture not in ARCHITECTURES:
            raise ValueError(
                f"unsupported collective architecture {self.collective_architecture!r}"
            )
        self._trace_buffer_enabled = self.collective_architecture in {
            BUFFERED_ARCHITECTURE,
            LEAN_ARCHITECTURE,
        }
        if self.collective_architecture == ARCHITECTURE:
            self.operation_queue = queue.Queue(maxsize=int(request.get("mailbox_depth", 2)))
            self.execution_thread = threading.Thread(
                target=self._execution_loop,
                daemon=True,
                name=f"collective-execute-{self.worker_id}",
            )
            self.execution_thread.start()
            self._collective_add("persistent_execution_loop_starts", 1)
        else:
            self.operation_queue = None
            self.execution_thread = None
        ordered_children = sorted(node["children"], key=lambda item: item["ordering_key"])
        if _uses_selector(self.collective_architecture):
            self.selector_fanout = PersistentFanout(
                [dict(child) for child in ordered_children],
                sender_id=self.worker_id,
                trace=self._trace,
                metrics_add=self._collective_add,
            )
        else:
            for child in ordered_children:
                self.edge_loops[str(child["worker_id"])] = _PersistentEdgeLoop(
                    self,
                    dict(child),
                    asynchronous=(
                        self.collective_architecture == ARCHITECTURE or len(node["children"]) > 1
                    ),
                )
        # One immutable payload holder models an equivalent distributed receive buffer.
        self._collective_add("reusable_buffer_allocations", 1 + len(ordered_children))
        self._collective_add("collective_installs", 1)
        self._trace(
            "persistent_collective_installed",
            collective_id=self.collective_id,
            architecture=self.collective_architecture,
            route_generation=installed["route_generation"],
            child_worker_ids=[str(child["worker_id"]) for child in ordered_children],
            persistent_execution_loops=int(self.collective_architecture == ARCHITECTURE),
            persistent_dispatch_loops=sum(
                int(loop.asynchronous) for loop in self.edge_loops.values()
            ),
            elapsed_ns=time.perf_counter_ns() - started_ns,
        )
        return installed

    def _prepare_collective(self, request: dict[str, Any]) -> dict[str, Any]:
        if str(request.get("collective_id", "")) != self.collective_id:
            raise ValueError("prepare targets a different persistent collective")
        profile = LinkProfile.from_dict(dict(request["network_profile"]))
        self.collective_profile = profile
        deadline_unix_ns = int(request["deadline_unix_ns"])
        with self.topology_lock:
            installed = self.topology
        if installed is None:
            raise RuntimeError("persistent collective preparation has no topology")
        node = dict(installed["node"])
        if self.selector_fanout is not None:
            pings = {
                str(child["worker_id"]): {
                    "magic": MAGIC,
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "delegated_execute",
                    "collective_protocol": COLLECTIVE_PROTOCOL,
                    "collective_ping": True,
                    "collective_id": self.collective_id,
                    "topology_id": request["topology_id"],
                    "route_lease_id": request["route_lease_id"],
                    "route_generation": request["route_generation"],
                    "request_id": f"prepare:{self.worker_id}:{child['worker_id']}",
                    "parent_worker": self.worker_id,
                    "target_worker": child["worker_id"],
                    "deadline_unix_ns": deadline_unix_ns,
                    "retry_policy": {"max_attempts": 0, "backoff_ms": 0.0},
                    "connection_policy": "persistent",
                }
                for child in node["children"]
            }
            results = self.selector_fanout.round_trip_many(
                pings, profile=profile, deadline_unix_ns=deadline_unix_ns
            )
            transports = [transport for _response, transport in results.values()]
            self._collective_add("collective_prepares", 1)
            return {
                "edge_count": len(pings),
                "new_connections": sum(int(item["new_connection_count"]) for item in transports),
                "messages": sum(
                    int(item["messages_sent"]) + int(item["messages_received"])
                    for item in transports
                ),
                "bytes": sum(
                    int(item["request_bytes"]) + int(item["response_bytes"]) for item in transports
                ),
                "elapsed_ns": max((int(item["elapsed_ns"]) for item in transports), default=0),
            }
        jobs: list[tuple[str, _MailboxJob]] = []
        direct_transports: list[dict[str, Any]] = []
        for child_id, loop in sorted(self.edge_loops.items()):
            ping = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "delegated_execute",
                "collective_protocol": COLLECTIVE_PROTOCOL,
                "collective_ping": True,
                "collective_id": self.collective_id,
                "topology_id": request["topology_id"],
                "route_lease_id": request["route_lease_id"],
                "route_generation": request["route_generation"],
                "request_id": f"prepare:{self.worker_id}:{child_id}",
                "parent_worker": self.worker_id,
                "target_worker": child_id,
                "deadline_unix_ns": deadline_unix_ns,
                "retry_policy": {"max_attempts": 0, "backoff_ms": 0.0},
                "connection_policy": "persistent",
            }
            if loop.asynchronous:
                jobs.append((child_id, loop.submit(ping, profile)))
            else:
                response, transport = loop.round_trip_inline(ping, profile)
                if (
                    response.get("kind") != "collective_pong"
                    or response.get("worker_id") != child_id
                ):
                    raise RuntimeError("collective direct edge prewarm returned the wrong identity")
                direct_transports.append(transport)
        transports: list[dict[str, Any]] = list(direct_transports)
        for child_id, job in jobs:
            response, transport = _PersistentEdgeLoop.wait(job, deadline_unix_ns)
            if response.get("kind") != "collective_pong" or response.get("worker_id") != child_id:
                raise RuntimeError("collective edge prewarm returned the wrong identity")
            transports.append(transport)
        self._collective_add("collective_prepares", 1)
        self._trace(
            "persistent_collective_prepared",
            collective_id=self.collective_id,
            child_count=len(jobs) + len(direct_transports),
            new_connections=sum(int(item["new_connection_count"]) for item in transports),
        )
        return {
            "edge_count": len(jobs) + len(direct_transports),
            "new_connections": sum(int(item["new_connection_count"]) for item in transports),
            "messages": sum(
                int(item["messages_sent"]) + int(item["messages_received"]) for item in transports
            ),
            "bytes": sum(
                int(item["request_bytes"]) + int(item["response_bytes"]) for item in transports
            ),
            "elapsed_ns": sum(int(item["elapsed_ns"]) for item in transports),
        }

    def _validate_collective_identity(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("magic") != MAGIC or request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("unsupported persistent collective protocol identity")
        if request.get("collective_protocol") != COLLECTIVE_PROTOCOL:
            raise ValueError("unsupported persistent collective envelope")
        if self.collective_id is None or request.get("collective_id") != self.collective_id:
            raise ValueError("operation targets an unavailable persistent collective")
        with self.topology_lock:
            installed = self.topology
        if installed is None:
            raise RuntimeError("persistent operation has no installed topology")
        node = dict(installed["node"])
        compact = _uses_compact_wire(self.collective_architecture)
        if (
            (not compact and request.get("topology_id") != installed["topology_id"])
            or request.get("route_lease_id") != installed["route_lease_id"]
            or int(request.get("route_generation", 0)) != int(installed["route_generation"])
            or (not compact and request.get("target_worker") != self.worker_id)
            or request.get("parent_worker") != node["parent_worker"]
        ):
            raise ValueError("persistent operation route, generation, target, or parent is stale")
        return node

    def _enqueue_operation(self, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_collective_identity(request)
        self._collective_add("persistent_receive_loop_wakeups", 1)
        if self.collective_architecture != ARCHITECTURE:
            # Recovery can reconnect while an earlier timed-out handler is
            # still unwinding. Preserve one ordered execution stream per
            # worker so generations cannot race through shared child sockets.
            lock_started_ns = time.perf_counter_ns()
            with self.collective_operation_lock:
                return self._execute_collective(
                    request,
                    queue_delay_ns=time.perf_counter_ns() - lock_started_ns,
                )
        operation_queue = self.operation_queue
        if operation_queue is None:
            raise RuntimeError("persistent execution mailbox is not running")
        job = _MailboxJob(message=dict(request))
        self._collective_add("dynamic_envelope_allocations", 1)
        self._collective_max("maximum_execution_queue_depth", operation_queue.qsize() + 1)
        operation_queue.put(job)
        remaining_s = max(0.0, (int(request["deadline_unix_ns"]) - time.time_ns()) / 1_000_000_000)
        if not job.done.wait(timeout=remaining_s):
            raise TimeoutError("persistent execution mailbox deadline elapsed")
        if job.error is not None:
            raise job.error
        return dict(job.result)

    def _execution_loop(self) -> None:
        while True:
            operation_queue = self.operation_queue
            if operation_queue is None:
                break
            job = operation_queue.get()
            if job is None:
                break
            job.started_ns = time.perf_counter_ns()
            queue_delay_ns = job.started_ns - job.enqueued_ns
            self._collective_add("persistent_execution_loop_wakeups", 1)
            self._collective_add("execution_queue_delay_ns", queue_delay_ns)
            try:
                job.result = self._execute_collective(job.message, queue_delay_ns=queue_delay_ns)
                self._collective_add("operations_completed", 1)
            except BaseException as error:
                job.error = error
                self._collective_add("operations_failed", 1)
            finally:
                job.completed_ns = time.perf_counter_ns()
                job.done.set()
        self._collective_add("persistent_execution_loop_stops", 1)

    def _empty_collective_metrics(self) -> dict[str, Any]:
        return {
            "total_messages": 0,
            "total_bytes": 0,
            "worker_to_worker_rpc_count": 0,
            "leaf_rpc_count": 0,
            "hierarchy_depth": 1,
            "reduction_depth": 0,
            "intermediate_reductions": 0,
            "connection_count": 0,
            "connection_ns": 0,
            "worker_cpu_ns": 0,
            "retries": 0,
            "failures": 0,
            "duplicated_work": 0,
            "latency_histogram": {},
            "leaf_latency_histogram": {},
            "persistent_execution_loop_wakeups": 1,
            "persistent_receive_loop_wakeups": 1,
            "persistent_mailbox_loop_wakeups": int(self.collective_architecture == ARCHITECTURE),
            "persistent_dispatch_loop_wakeups": 0,
            "application_scheduler_events": 1 + int(self.collective_architecture == ARCHITECTURE),
            "worker_activations": 0,
            "new_task_creation": 0,
            "topology_rebuilds": 0,
            "dynamic_envelope_allocations": 1,
            "reusable_buffer_uses": 1,
            "queue_delay_ns": 0,
            "dispatch_elapsed_ns": 0,
            "local_execution_ns": 0,
            "reduction_ns": 0,
            "serialization_ns": 0,
            "maximum_queue_depth": 1,
        }

    def _execute_collective(
        self, request: dict[str, Any], *, queue_delay_ns: int
    ) -> dict[str, Any]:
        node = self._validate_collective_identity(request)
        for field_name in ("request_id", "operation_id"):
            if not isinstance(request.get(field_name), str) or not request[field_name]:
                raise ValueError(f"persistent collective field {field_name!r} is required")
        generation = int(request.get("execution_generation", 0))
        if generation <= 0:
            raise ValueError("persistent execution generation must be positive")
        if int(request.get("deadline_unix_ns", 0)) <= time.time_ns():
            raise TimeoutError("persistent operation deadline elapsed")
        if dict(request.get("cancellation", {"cancelled": False})).get("cancelled") is not False:
            raise RuntimeError("persistent operation was cancelled")
        operation_id = str(request["operation_id"])
        request_id = str(request["request_id"])
        if operation_id in self.cancelled_operations:
            raise RuntimeError("persistent operation was cancelled at this worker")
        if _uses_lean_wire(self.collective_architecture):
            operation_digest = str(request.get("od", ""))
            if len(operation_digest) != 64:
                raise ValueError("lean operation digest is missing or malformed")
            request_hash = ":".join(
                (
                    operation_digest,
                    request_id,
                    str(request.get("parent_worker", "")),
                    str(generation),
                )
            )
        else:
            request_hash = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        with self.cache_lock:
            cached = self.collective_response_cache.get(request_id)
            if cached is not None:
                if cached[0] != request_hash:
                    raise ValueError("duplicate persistent request ID has different content")
                response = json.loads(json.dumps(cached[1]))
                if "m" in response:
                    response["m"][COMPACT_METRIC_FIELDS.index("duplicated_work")] += 1
                elif "lm" in response:
                    response["lm"][LEAN_METRIC_FIELDS.index("duplicated_work")] += 1
                else:
                    response["subtree_metrics"]["duplicated_work"] += 1
                with self.metrics_lock:
                    self.metrics["duplicate_requests"] += 1
                self._trace(
                    "duplicate_persistent_request_replayed",
                    collective_id=self.collective_id,
                    operation_id=operation_id,
                    request_id=request_id,
                    execution_generation=generation,
                )
                return response
        if generation <= self.last_completed_generation:
            raise ValueError("stale persistent execution generation")

        fault_control = dict(request.get("fault_control", {}))
        fault_kind = str(fault_control.get("kind", ""))
        fault_targets = {str(value) for value in fault_control.get("target_worker_ids", [])}

        def inject_once(phase: str) -> bool:
            if self.worker_id not in fault_targets:
                return False
            key = (operation_id, f"{fault_kind}:{phase}")
            if not bool(fault_control.get("one_shot", False)):
                return True
            with self.fault_lock:
                if key in self.injected_faults:
                    return False
                self.injected_faults.add(key)
                return True

        if fault_kind in {"worker_failure_once", "worker_failure_always"} and inject_once(
            "before_compute"
        ):
            self._trace(
                "persistent_controlled_fault_injected",
                operation_id=operation_id,
                request_id=request_id,
                fault_kind=fault_kind,
                phase="before_compute",
            )
            raise RuntimeError(f"controlled {fault_kind} at {self.worker_id}")

        if "network_profile" in request:
            profile = LinkProfile.from_dict(dict(request["network_profile"]))
        elif self.collective_profile is not None:
            profile = self.collective_profile
        else:
            raise RuntimeError("persistent collective network profile is not installed")
        payload = base64.b64decode(str(request.get("payload_b64", "")), validate=True)
        if len(payload) > int(self.runtime_profile["maximum_payload_bytes"]):
            raise MemoryError("persistent payload exceeds declared worker memory limit")
        metrics = self._empty_collective_metrics()
        metrics["queue_delay_ns"] = queue_delay_ns
        cpu_started_ns = time.process_time_ns()
        local_started_ns = time.perf_counter_ns()
        local_delay_ms = float(self.runtime_profile["compute_delay_ms"])
        if local_delay_ms:
            self._interruptible_delay(
                operation_id=operation_id,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                delay_ms=local_delay_ms,
            )
        if fault_kind == "slow_child" and inject_once("after_compute"):
            self._interruptible_delay(
                operation_id=operation_id,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                delay_ms=float(fault_control.get("delay_ms", 0.0)),
            )
        aggregation = dict(
            request.get(
                "aggregation",
                {"mode": "exact_int64_sum", "ordering": "deterministic_worker_key"},
            )
        )
        if aggregation.get("mode") == "vocabulary_argmax":
            aggregate_items = [self._real_model_local_aggregate(request)]
        else:
            aggregate_items = [self.synthetic_local_aggregate]
        metrics["local_execution_ns"] = time.perf_counter_ns() - local_started_ns

        children = sorted(node["children"], key=lambda item: item["ordering_key"])
        submitted: list[tuple[dict[str, Any], _MailboxJob]] = []
        selector_messages: dict[str, dict[str, Any]] = {}
        child_results: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        dispatch_started_ns = time.perf_counter_ns()
        for child in children:
            child_id = str(child["worker_id"])
            child_request = make_collective_request(
                collective_id=str(self.collective_id),
                topology_id=str(request.get("topology_id", "")),
                route_lease_id=str(request["route_lease_id"]),
                route_generation=int(request["route_generation"]),
                request_id=f"{request_id}:{child_id}",
                operation_id=operation_id,
                execution_generation=generation,
                parent_worker=self.worker_id,
                target_worker=child_id,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                profile=profile,
                retry_policy=dict(
                    request.get("retry_policy", {"max_attempts": 0, "backoff_ms": 0.0})
                ),
                fault_control=fault_control,
                aggregation=aggregation,
                payload_b64=str(request["payload_b64"]),
                workload=(dict(request["workload"]) if "workload" in request else None),
                architecture=self.collective_architecture,
                operation_digest=(
                    str(request["od"]) if _uses_lean_wire(self.collective_architecture) else None
                ),
            )
            if self.selector_fanout is not None:
                selector_messages[child_id] = child_request
            else:
                loop = self.edge_loops.get(child_id)
                if loop is None:
                    raise RuntimeError(f"persistent child edge {child_id!r} is not installed")
                if loop.asynchronous:
                    submitted.append((child, loop.submit(child_request, profile)))
                else:
                    response, transport = loop.round_trip_inline(child_request, profile)
                    child_results.append((child, response, transport))

        if self.selector_fanout is not None and selector_messages:
            selector_results = self.selector_fanout.round_trip_many(
                selector_messages,
                profile=profile,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
            )
            by_child = {str(child["worker_id"]): child for child in children}
            for child_id, (response, transport) in selector_results.items():
                child_results.append((by_child[child_id], response, transport))

        for child, job in submitted:
            response, transport = _PersistentEdgeLoop.wait(job, int(request["deadline_unix_ns"]))
            child_request_id = f"{request_id}:{child['worker_id']}"
            if (
                response.get("operation_id") != operation_id
                or response.get("request_id") != child_request_id
                or int(response.get("execution_generation", 0)) != generation
            ):
                raise RoundTripError(
                    "persistent child response identity mismatch",
                    metrics=transport,
                    response=response,
                )
            child_aggregate = dict(response["aggregate"])
            if response.get("aggregate_digest") != aggregate_digest(child_aggregate):
                raise RoundTripError(
                    "persistent child aggregate digest mismatch",
                    metrics=transport,
                    response=response,
                )
            child_results.append((child, response, transport))
        metrics["dispatch_elapsed_ns"] = time.perf_counter_ns() - dispatch_started_ns

        lean_wire = _uses_lean_wire(self.collective_architecture)
        child_depths: list[int] = []
        child_reduction_depths: list[int] = []
        for child, response, transport in child_results:
            child_metrics = decode_subtree_metrics(response)
            aggregate_items.append(dict(response["aggregate"]))
            metrics["total_messages"] += (
                int(transport["messages_sent"])
                + int(transport["messages_received"])
                + int(child_metrics["total_messages"])
            )
            metrics["total_bytes"] += (
                int(transport["request_bytes"])
                + int(transport["response_bytes"])
                + int(child_metrics["total_bytes"])
            )
            metrics["worker_to_worker_rpc_count"] += int(transport["attempt_count"]) + int(
                child_metrics["worker_to_worker_rpc_count"]
            )
            if not lean_wire:
                metrics["leaf_rpc_count"] += int(child_metrics["leaf_rpc_count"]) + int(
                    not child["assigned_child_worker_ids"]
                )
            metrics["connection_count"] += int(transport["new_connection_count"]) + int(
                child_metrics["connection_count"]
            )
            metrics["connection_ns"] += int(transport["connect_ns"]) + int(
                child_metrics["connection_ns"]
            )
            if not lean_wire:
                metrics["worker_cpu_ns"] += int(child_metrics["worker_cpu_ns"])
            metrics["retries"] += int(transport["retry_count"]) + int(child_metrics["retries"])
            metrics["failures"] += int(transport["retry_count"]) + int(child_metrics["failures"])
            aggregate_fields = (
                ("duplicated_work",)
                if lean_wire
                else (
                    "duplicated_work",
                    "persistent_execution_loop_wakeups",
                    "persistent_receive_loop_wakeups",
                    "persistent_mailbox_loop_wakeups",
                    "persistent_dispatch_loop_wakeups",
                    "application_scheduler_events",
                    "worker_activations",
                    "new_task_creation",
                    "topology_rebuilds",
                    "dynamic_envelope_allocations",
                    "reusable_buffer_uses",
                    "queue_delay_ns",
                    "local_execution_ns",
                    "reduction_ns",
                    "serialization_ns",
                )
            )
            for metric_field in aggregate_fields:
                metrics[metric_field] += int(child_metrics.get(metric_field, 0))
            edge_loop = self.edge_loops.get(str(child["worker_id"]))
            if edge_loop is not None and edge_loop.asynchronous:
                metrics["persistent_dispatch_loop_wakeups"] += 1
                metrics["application_scheduler_events"] += 1
            if not lean_wire:
                metrics["dynamic_envelope_allocations"] += 1
                metrics["reusable_buffer_uses"] += 1
                metrics["serialization_ns"] += int(transport.get("serialization_ns", 0))
                metrics["maximum_queue_depth"] = max(
                    int(metrics["maximum_queue_depth"]),
                    int(child_metrics.get("maximum_queue_depth", 1)),
                )
                edge_histogram: dict[str, int] = {}
                latency_histogram_observe(edge_histogram, int(transport["elapsed_ns"]))
                metrics["latency_histogram"] = combine_latency_histograms(
                    dict(metrics["latency_histogram"]),
                    edge_histogram,
                    dict(child_metrics.get("latency_histogram", {})),
                )
                leaf_histogram = dict(child_metrics.get("leaf_latency_histogram", {}))
                if not child["assigned_child_worker_ids"]:
                    latency_histogram_observe(leaf_histogram, int(transport["elapsed_ns"]))
                metrics["leaf_latency_histogram"] = combine_latency_histograms(
                    dict(metrics["leaf_latency_histogram"]), leaf_histogram
                )
                child_depths.append(int(child_metrics["hierarchy_depth"]))
                child_reduction_depths.append(int(child_metrics["reduction_depth"]))

        if operation_id in self.cancelled_operations:
            raise RuntimeError("persistent operation was cancelled before reduction")
        reduction_started_ns = time.perf_counter_ns()
        aggregate = combine_operation_aggregates(aggregate_items, aggregation)
        metrics["reduction_ns"] += time.perf_counter_ns() - reduction_started_ns
        if lean_wire:
            metrics["hierarchy_depth"] = 1
            metrics["reduction_depth"] = int(bool(children))
            metrics["intermediate_reductions"] = int(bool(children))
        else:
            metrics["hierarchy_depth"] = 1 + max(child_depths, default=0)
            metrics["reduction_depth"] = (
                1 + max(child_reduction_depths, default=0) if children else 0
            )
            metrics["intermediate_reductions"] = sum(
                int(decode_subtree_metrics(response)["intermediate_reductions"])
                for _child, response, _transport in child_results
            ) + int(bool(children))
        metrics["worker_cpu_ns"] += time.process_time_ns() - cpu_started_ns
        response = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "result",
            "collective_protocol": COLLECTIVE_PROTOCOL,
            "collective_id": self.collective_id,
            "request_id": request_id,
            "operation_id": operation_id,
            "execution_generation": generation,
            "worker_id": self.worker_id,
            "parent_worker": request["parent_worker"],
            "aggregate": aggregate,
            "aggregate_digest": aggregate_digest(aggregate),
            "status": "ok",
        }
        if lean_wire:
            response["lm"] = [metrics[field] for field in LEAN_METRIC_FIELDS]
        elif _uses_compact_wire(self.collective_architecture):
            response["m"] = [metrics[field] for field in COMPACT_METRIC_FIELDS]
        else:
            response["subtree_metrics"] = metrics
        with self.cache_lock:
            self.collective_response_cache[request_id] = (request_hash, response)
            self.collective_response_cache.move_to_end(request_id)
            while len(self.collective_response_cache) > 64:
                self.collective_response_cache.popitem(last=False)
        self.last_completed_generation = max(self.last_completed_generation, generation)
        with self.metrics_lock:
            self.metrics["delegated_requests"] += 1
            self.metrics["child_rpcs"] += len(children)
        self._trace(
            "persistent_subtree_reduced",
            collective_id=self.collective_id,
            operation_id=operation_id,
            request_id=request_id,
            execution_generation=generation,
            parent_worker=request["parent_worker"],
            child_worker_ids=[child["worker_id"] for child in children],
            contribution_count=aggregate["contribution_count"],
            aggregate_digest=response["aggregate_digest"],
            hierarchy_depth=metrics["hierarchy_depth"],
            execution_queue_delay_ns=queue_delay_ns,
            dispatch_elapsed_ns=metrics["dispatch_elapsed_ns"],
            local_execution_ns=metrics["local_execution_ns"],
            reduction_ns=metrics["reduction_ns"],
            new_task_creation=0,
            topology_rebuilds=0,
        )
        return response

    def _dispatch_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        kind = str(request.get("kind"))
        if kind == "install_collective":
            if request.get("collective_protocol") != COLLECTIVE_PROTOCOL:
                raise ValueError("unsupported collective install protocol")
            installed = self._start_collective(request)
            return kind, {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "collective_installed",
                "worker_id": self.worker_id,
                "collective_id": self.collective_id,
                "topology_id": installed["topology_id"],
                "route_generation": installed["route_generation"],
                "persistent_execution_loop_starts": 1,
                "persistent_dispatch_loop_starts": sum(
                    int(loop.asynchronous) for loop in self.edge_loops.values()
                ),
                "status": "ok",
            }
        if kind == "prepare_collective":
            prepared = self._prepare_collective(request)
            return kind, {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "collective_prepared",
                "worker_id": self.worker_id,
                "collective_id": self.collective_id,
                "status": "ok",
                **prepared,
            }
        if (
            kind == "delegated_execute"
            and request.get("collective_protocol") == COLLECTIVE_PROTOCOL
        ):
            if request.get("collective_ping") is True:
                self._validate_collective_identity(request)
                if not self.parent_session_prepared:
                    self.parent_session_prepared = True
                    self._collective_add("persistent_receive_loop_starts", 1)
                    if self.collective_architecture != ARCHITECTURE:
                        self._collective_add("persistent_execution_loop_starts", 1)
                return "delegated_execute", {
                    "magic": MAGIC,
                    "protocol_version": PROTOCOL_VERSION,
                    "kind": "collective_pong",
                    "worker_id": self.worker_id,
                    "collective_id": self.collective_id,
                    "status": "ok",
                }
            response = self._enqueue_operation(request)
            return "delegated_execute", self._apply_response_fault(request, response)
        if kind == "status":
            _base_kind, response = super()._dispatch_request(request)
            response["persistent_collective"] = {
                "collective_id": self.collective_id,
                "architecture": self.collective_architecture,
                "execution_loop_alive": bool(
                    self.collective_architecture != ARCHITECTURE
                    or (self.execution_thread is not None and self.execution_thread.is_alive())
                ),
                "dispatch_loops_alive": sum(
                    loop.thread is not None and loop.thread.is_alive()
                    for loop in self.edge_loops.values()
                ),
                "metrics": self._collective_snapshot(),
            }
            return kind, response
        if kind == "shutdown":
            self._stop_collective()
        return super()._dispatch_request(request)


def _serve_from_config(path: Path) -> int:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        return PersistentCollectiveServer(config).serve()
    except BaseException as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve_from_config(args.config.resolve())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
