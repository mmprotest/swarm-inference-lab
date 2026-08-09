"""Lightweight canonical protocol primitives for persistent microworkers.

The module intentionally uses only the Python standard library so a large
single-host process experiment does not pay the import and resident-memory
cost of a tensor runtime in every controlled protocol worker.  The wire is a
real length-prefixed TCP protocol; the controlled exact-integer operation is
an explicit synthetic workload, not model evidence.

Earlier Experiment 012 sources are retained in their cycle snapshots. The
current protocol includes the measured parallel hierarchy, persistent
parent-child sessions, exact reduction, bounded retries, and cancellation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "1.0"
MAGIC = "SWARMMW1"
MAX_FRAME_BYTES = 8 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_PROOF_MODULUS = 1 << 256


class RoundTripError(RuntimeError):
    """Transport or remote failure retaining the measured attempt."""

    def __init__(
        self,
        message: str,
        *,
        metrics: dict[str, Any],
        response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.response = response


class DropResponse(RuntimeError):
    """Internal sentinel used by a controlled fault to close without a response."""


@dataclass(frozen=True, slots=True)
class LinkProfile:
    name: str
    rtt_ms: float
    upload_mbps: float
    download_mbps: float
    jitter_ms: float = 0.0
    request_loss_rate: float = 0.0
    temporary_disconnect_rate: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LinkProfile:
        profile = cls(
            name=str(value["name"]),
            rtt_ms=float(value["rtt_ms"]),
            upload_mbps=float(value["upload_mbps"]),
            download_mbps=float(value["download_mbps"]),
            jitter_ms=float(value.get("jitter_ms", 0.0)),
            request_loss_rate=float(value.get("request_loss_rate", 0.0)),
            temporary_disconnect_rate=float(value.get("temporary_disconnect_rate", 0.0)),
        )
        if profile.rtt_ms < 0 or profile.jitter_ms < 0:
            raise ValueError("link latency and jitter must be non-negative")
        if profile.upload_mbps <= 0 or profile.download_mbps <= 0:
            raise ValueError("link bandwidth must be positive")
        if not 0 <= profile.request_loss_rate <= 1:
            raise ValueError("request loss rate must be between zero and one")
        if not 0 <= profile.temporary_disconnect_rate <= 1:
            raise ValueError("disconnect rate must be between zero and one")
        return profile


NETWORK_PROFILES: dict[str, LinkProfile] = {
    "same_host_shaped": LinkProfile("same_host_shaped", 0.10, 10_000.0, 10_000.0, 0.01),
    "fast_lan_shaped": LinkProfile("fast_lan_shaped", 0.50, 1_000.0, 1_000.0, 0.05),
    "slower_lan_shaped": LinkProfile("slower_lan_shaped", 5.0, 100.0, 100.0, 0.50),
    "metro_intercity_shaped": LinkProfile("metro_intercity_shaped", 20.0, 100.0, 100.0, 2.0),
    "moderate_wan_shaped": LinkProfile("moderate_wan_shaped", 80.0, 25.0, 50.0, 8.0),
    "intercontinental_wan_shaped": LinkProfile(
        "intercontinental_wan_shaped", 220.0, 10.0, 20.0, 20.0
    ),
    # H012-007 held-out profiles were added only after that cycle's criteria
    # were locked.  They intentionally sit between H012-006 calibration
    # points and must never be read by the planner before its decisions are
    # sealed.
    "holdout_2ms_shaped": LinkProfile("holdout_2ms_shaped", 2.0, 500.0, 500.0, 0.2),
    "holdout_10ms_shaped": LinkProfile("holdout_10ms_shaped", 10.0, 100.0, 100.0, 1.0),
    "holdout_40ms_shaped": LinkProfile("holdout_40ms_shaped", 40.0, 50.0, 50.0, 4.0),
}


def canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def encode_message(value: dict[str, Any]) -> bytes:
    payload = canonical_json_bytes(value)
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("microworker frame exceeds the maximum size")
    return _LENGTH.pack(len(payload)) + payload


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("microworker connection closed before frame completion")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket) -> tuple[dict[str, Any], int]:
    prefix = _recv_exact(connection, _LENGTH.size)
    (size,) = _LENGTH.unpack(prefix)
    if size <= 0 or size > MAX_FRAME_BYTES:
        raise ValueError("invalid microworker frame size")
    payload = _recv_exact(connection, size)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("microworker frame root must be an object")
    return decoded, len(prefix) + len(payload)


def send_message(connection: socket.socket, value: dict[str, Any]) -> int:
    framed = encode_message(value)
    connection.sendall(framed)
    return len(framed)


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, port = endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid endpoint {endpoint!r}")
    return host, int(port)


def worker_contribution(worker_index: int) -> int:
    """Return a stable signed contribution with no random state."""

    return ((worker_index + 1) * 7_919 % 2_000_003) - 1_000_001


def _leaf_proof_value(worker_id: str, contribution: int) -> int:
    encoded = f"{worker_id}:{contribution}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


def leaf_aggregate(worker_id: str, contribution: int) -> dict[str, Any]:
    proof = _leaf_proof_value(worker_id, contribution)
    return {
        "aggregate": contribution,
        "contribution_count": 1,
        "proof_xor": f"{proof:064x}",
        "proof_sum": f"{proof:064x}",
    }


def combine_aggregates(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("at least one aggregate is required")
    aggregate = 0
    count = 0
    proof_xor = 0
    proof_sum = 0
    for item in items:
        aggregate += int(item["aggregate"])
        count += int(item["contribution_count"])
        proof_xor ^= int(str(item["proof_xor"]), 16)
        proof_sum = (proof_sum + int(str(item["proof_sum"]), 16)) % _PROOF_MODULUS
    return {
        "aggregate": aggregate,
        "contribution_count": count,
        "proof_xor": f"{proof_xor:064x}",
        "proof_sum": f"{proof_sum:064x}",
    }


def aggregate_digest(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def combine_argmax_aggregates(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce vocabulary-shard maxima with deterministic global tie breaking."""

    if not items:
        raise ValueError("at least one argmax aggregate is required")
    shards: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    contribution_count = 0
    local_logit_count = 0
    for item in items:
        score = float(item["score"])
        token_id = int(item["token_id"])
        if not math.isfinite(score) or token_id < 0:
            raise ValueError("argmax aggregate contains an invalid candidate")
        candidates.append((score, token_id, item))
        contribution_count += int(item["contribution_count"])
        local_logit_count += int(item["local_logit_count"])
        shards.extend(dict(shard) for shard in item["shards"])
    if len({str(shard["worker_id"]) for shard in shards}) != len(shards):
        raise ValueError("argmax aggregate contains a duplicate worker shard")
    shards.sort(key=lambda shard: (int(shard["token_start"]), str(shard["worker_id"])))
    winner = max(candidates, key=lambda item: (item[0], -item[1]))[2]
    return {
        "mode": "vocabulary_argmax",
        "score": float(winner["score"]),
        "score_float32_hex": str(winner["score_float32_hex"]),
        "token_id": int(winner["token_id"]),
        "winner_worker_id": str(winner["winner_worker_id"]),
        "contribution_count": contribution_count,
        "local_logit_count": local_logit_count,
        "shards": shards,
    }


def combine_operation_aggregates(
    items: list[dict[str, Any]], aggregation: dict[str, Any]
) -> dict[str, Any]:
    mode = str(aggregation.get("mode", ""))
    if mode == "exact_int64_sum":
        return combine_aggregates(items)
    if mode == "vocabulary_argmax":
        return combine_argmax_aggregates(items)
    raise ValueError(f"unsupported microworker aggregation mode {mode!r}")


def latency_histogram_observe(histogram: dict[str, int], elapsed_ns: int) -> None:
    microseconds = max(1, math.ceil(elapsed_ns / 1_000))
    bucket = str(min(63, math.ceil(math.log2(microseconds))))
    histogram[bucket] = histogram.get(bucket, 0) + 1


def combine_latency_histograms(*histograms: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for histogram in histograms:
        for bucket, count in histogram.items():
            combined[str(int(bucket))] = combined.get(str(int(bucket)), 0) + int(count)
    return combined


def latency_histogram_percentile_ms(histogram: dict[str, int], percentile: float) -> float:
    total = sum(histogram.values())
    if total <= 0:
        return 0.0
    target = max(1, math.ceil(total * percentile / 100.0))
    cumulative = 0
    for bucket in sorted(int(value) for value in histogram):
        cumulative += histogram[str(bucket)]
        if cumulative >= target:
            return math.pow(2.0, bucket) / 1_000.0
    return math.pow(2.0, max(int(value) for value in histogram)) / 1_000.0


_OPERATION_REQUIRED_FIELDS = {
    "request_id",
    "operation_id",
    "execution_generation",
    "parent_worker",
    "assigned_child_workers",
    "work_partition",
    "aggregation",
    "deadline_unix_ns",
    "route_generation",
    "route_lease_id",
    "retry_policy",
    "deterministic_ordering_key",
    "cancellation",
    "trace_context",
    "connection_policy",
}


def validate_operation_envelope(message: dict[str, Any], *, allow_children: bool) -> None:
    if message.get("magic") != MAGIC or message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported microworker protocol identity")
    missing = sorted(_OPERATION_REQUIRED_FIELDS - message.keys())
    if missing:
        raise ValueError(f"microworker operation is missing fields: {missing}")
    for field in ("request_id", "operation_id", "parent_worker", "route_lease_id"):
        if not isinstance(message[field], str) or not message[field]:
            raise ValueError(f"microworker field {field!r} must be a non-empty string")
    if int(message["execution_generation"]) <= 0:
        raise ValueError("execution generation must be positive")
    if int(message["route_generation"]) <= 0:
        raise ValueError("route generation must be positive")
    if int(message["deadline_unix_ns"]) <= time.time_ns():
        raise TimeoutError("microworker operation deadline elapsed")
    children = message["assigned_child_workers"]
    if not isinstance(children, list):
        raise ValueError("assigned child workers must be a list")
    if children and not allow_children:
        raise ValueError("direct leaf execution cannot carry delegated children")
    aggregation = message["aggregation"]
    supported_aggregations = (
        {
            "mode": "exact_int64_sum",
            "ordering": "deterministic_worker_key",
        },
        {
            "mode": "vocabulary_argmax",
            "ordering": "maximum_score_then_lowest_token_id",
        },
    )
    if aggregation not in supported_aggregations:
        raise ValueError("unsupported microworker aggregation contract")
    if aggregation["mode"] == "vocabulary_argmax":
        workload = message.get("workload")
        if not isinstance(workload, dict) or workload.get("kind") != "real_model_lm_head":
            raise ValueError("real vocabulary argmax workload contract is missing")
    cancellation = message["cancellation"]
    if not isinstance(cancellation, dict) or cancellation.get("cancelled") is not False:
        raise RuntimeError("microworker operation was cancelled")
    retry = message["retry_policy"]
    if not isinstance(retry, dict) or int(retry.get("max_attempts", -1)) < 0:
        raise ValueError("invalid microworker retry policy")
    if message.get("dispatch_policy", "serial") not in {"serial", "parallel"}:
        raise ValueError("invalid delegated dispatch policy")
    if message["connection_policy"] not in {"ephemeral", "persistent"}:
        raise ValueError("invalid microworker connection policy")
    trace = message["trace_context"]
    if not isinstance(trace, dict) or not trace.get("trace_id") or not trace.get("span_id"):
        raise ValueError("microworker trace context is incomplete")


def make_leaf_request(
    *,
    request_id: str,
    operation_id: str,
    execution_generation: int,
    parent_worker: str,
    worker_id: str,
    worker_index: int,
    deadline_unix_ns: int,
    route_lease_id: str,
    ordering_key: str,
    payload_bytes: int,
    trace_id: str,
    span_id: str,
) -> dict[str, Any]:
    if payload_bytes < 0:
        raise ValueError("payload bytes must be non-negative")
    return {
        "magic": MAGIC,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "leaf_execute",
        "request_id": request_id,
        "operation_id": operation_id,
        "execution_generation": execution_generation,
        "parent_worker": parent_worker,
        "assigned_child_workers": [],
        "work_partition": {
            "worker_id": worker_id,
            "worker_index": worker_index,
            "partition_start": worker_index,
            "partition_end": worker_index + 1,
        },
        "aggregation": {
            "mode": "exact_int64_sum",
            "ordering": "deterministic_worker_key",
        },
        "deadline_unix_ns": deadline_unix_ns,
        "route_generation": 1,
        "route_lease_id": route_lease_id,
        "retry_policy": {"max_attempts": 0, "backoff_ms": 0.0},
        "deterministic_ordering_key": ordering_key,
        "cancellation": {"cancelled": False, "epoch": 0},
        "connection_policy": "ephemeral",
        "trace_context": {
            "trace_id": trace_id,
            "parent_span_id": "root",
            "span_id": span_id,
        },
        "payload_b64": base64.b64encode(bytes(payload_bytes)).decode("ascii"),
    }


def make_delegated_request(
    *,
    request_id: str,
    operation_id: str,
    execution_generation: int,
    parent_worker: str,
    worker_id: str,
    worker_index: int,
    assigned_child_workers: list[str],
    partition_start: int,
    partition_end: int,
    subtree_worker_count: int,
    deadline_unix_ns: int,
    route_lease_id: str,
    ordering_key: str,
    payload_bytes: int,
    trace_id: str,
    parent_span_id: str,
    span_id: str,
    network_profile: LinkProfile,
    dispatch_policy: str = "serial",
    connection_policy: str = "ephemeral",
    route_generation: int = 1,
    retry_policy: dict[str, Any] | None = None,
    cancellation: dict[str, Any] | None = None,
    partition_worker_indices: list[int] | None = None,
    fault_control: dict[str, Any] | None = None,
    aggregation: dict[str, str] | None = None,
    workload: dict[str, Any] | None = None,
    payload_b64: str | None = None,
) -> dict[str, Any]:
    if payload_bytes < 0:
        raise ValueError("payload bytes must be non-negative")
    if dispatch_policy not in {"serial", "parallel"}:
        raise ValueError("delegated dispatch policy must be serial or parallel")
    if connection_policy not in {"ephemeral", "persistent"}:
        raise ValueError("delegated connection policy must be ephemeral or persistent")
    if payload_b64 is None:
        encoded_payload = base64.b64encode(bytes(payload_bytes)).decode("ascii")
    else:
        decoded_payload = base64.b64decode(payload_b64, validate=True)
        if len(decoded_payload) != payload_bytes:
            raise ValueError("delegated encoded payload length differs from payload_bytes")
        encoded_payload = payload_b64
    message = {
        "magic": MAGIC,
        "protocol_version": PROTOCOL_VERSION,
        "kind": "delegated_execute",
        "request_id": request_id,
        "operation_id": operation_id,
        "execution_generation": execution_generation,
        "parent_worker": parent_worker,
        "assigned_child_workers": list(assigned_child_workers),
        "work_partition": {
            "worker_id": worker_id,
            "worker_index": worker_index,
            "partition_start": partition_start,
            "partition_end": partition_end,
            "partition_worker_indices": (
                list(partition_worker_indices)
                if partition_worker_indices is not None
                else list(range(partition_start, partition_end))
            ),
            "subtree_worker_count": subtree_worker_count,
        },
        "aggregation": aggregation
        or {"mode": "exact_int64_sum", "ordering": "deterministic_worker_key"},
        "deadline_unix_ns": deadline_unix_ns,
        "route_generation": route_generation,
        "route_lease_id": route_lease_id,
        "retry_policy": retry_policy or {"max_attempts": 0, "backoff_ms": 0.0},
        "deterministic_ordering_key": ordering_key,
        "cancellation": cancellation or {"cancelled": False, "epoch": 0},
        "trace_context": {
            "trace_id": trace_id,
            "parent_span_id": parent_span_id,
            "span_id": span_id,
        },
        "network_profile": {
            "name": network_profile.name,
            "rtt_ms": network_profile.rtt_ms,
            "upload_mbps": network_profile.upload_mbps,
            "download_mbps": network_profile.download_mbps,
            "jitter_ms": network_profile.jitter_ms,
            "request_loss_rate": network_profile.request_loss_rate,
            "temporary_disconnect_rate": network_profile.temporary_disconnect_rate,
        },
        "dispatch_policy": dispatch_policy,
        "connection_policy": connection_policy,
        "fault_control": dict(fault_control or {}),
        "payload_b64": encoded_payload,
    }
    if workload is not None:
        message["workload"] = dict(workload)
    return message


def _deterministic_fraction(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _shape_delay_s(
    profile: LinkProfile,
    *,
    byte_count: int,
    direction: str,
    seed: str,
) -> float:
    bandwidth_mbps = profile.upload_mbps if direction == "request" else profile.download_mbps
    bandwidth_s = byte_count * 8 / (bandwidth_mbps * 1_000_000)
    jitter_fraction = (_deterministic_fraction(seed, direction, "jitter") * 2.0) - 1.0
    one_way_ms = max(0.0, profile.rtt_ms / 2.0 + jitter_fraction * profile.jitter_ms / 2.0)
    return one_way_ms / 1_000.0 + bandwidth_s


def shaped_round_trip(
    *,
    endpoint: str,
    message: dict[str, Any],
    profile: LinkProfile,
    timeout_s: float,
    sender_id: str,
    receiver_id: str,
    attempt: int = 0,
    connection: socket.socket | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one real TCP request/response with deterministic link shaping."""

    if timeout_s <= 0:
        raise TimeoutError("microworker timeout elapsed before transport")
    framed = encode_message(message)
    seed = f"{message.get('request_id')}|{sender_id}|{receiver_id}|{attempt}"
    if _deterministic_fraction(seed, "loss") < profile.request_loss_rate:
        time.sleep(min(timeout_s, profile.rtt_ms / 2_000.0))
        raise TimeoutError("simulated shaped-link request loss")
    if _deterministic_fraction(seed, "disconnect") < profile.temporary_disconnect_rate:
        raise ConnectionError("simulated shaped-link temporary disconnect")

    started_ns = time.perf_counter_ns()
    cpu_started_ns = time.process_time_ns()
    request_delay_s = _shape_delay_s(
        profile, byte_count=len(framed), direction="request", seed=seed
    )
    if request_delay_s:
        time.sleep(min(request_delay_s, timeout_s))
    owns_connection = connection is None
    connect_ns = 0
    if connection is None:
        host, port = parse_endpoint(endpoint)
        connect_started_ns = time.perf_counter_ns()
        connection = socket.create_connection((host, port), timeout=timeout_s)
        connect_ns = time.perf_counter_ns() - connect_started_ns
    response: dict[str, Any] | None = None
    response_bytes = 0
    request_sent = False
    try:
        connection.settimeout(timeout_s)
        connection.sendall(framed)
        request_sent = True
        response, response_bytes = receive_message(connection)
    except BaseException as error:
        elapsed_ns = time.perf_counter_ns() - started_ns
        metrics = {
            "sender_worker_id": sender_id,
            "receiver_worker_id": receiver_id,
            "request_bytes": len(framed) if request_sent else 0,
            "response_bytes": 0,
            "messages_sent": int(request_sent),
            "messages_received": 0,
            "connect_ns": connect_ns,
            "new_connection_count": int(owns_connection),
            "connection_reused": not owns_connection,
            "elapsed_ns": elapsed_ns,
            "client_cpu_ns": time.process_time_ns() - cpu_started_ns,
            "profile": profile.name,
            "attempt": attempt,
        }
        raise RoundTripError(str(error), metrics=metrics) from error
    finally:
        if owns_connection:
            connection.close()
    assert response is not None
    response_delay_s = _shape_delay_s(
        profile, byte_count=response_bytes, direction="response", seed=seed
    )
    if response_delay_s:
        time.sleep(min(response_delay_s, timeout_s))
    elapsed_ns = time.perf_counter_ns() - started_ns
    if elapsed_ns / 1_000_000_000 > timeout_s:
        raise TimeoutError("microworker deadline elapsed during shaped transport")
    metrics = {
        "sender_worker_id": sender_id,
        "receiver_worker_id": receiver_id,
        "request_bytes": len(framed),
        "response_bytes": response_bytes,
        "messages_sent": 1,
        "messages_received": 1,
        "connect_ns": connect_ns,
        "new_connection_count": int(owns_connection),
        "connection_reused": not owns_connection,
        "elapsed_ns": elapsed_ns,
        "client_cpu_ns": time.process_time_ns() - cpu_started_ns,
        "profile": profile.name,
        "attempt": attempt,
    }
    if response.get("kind") == "error":
        raise RoundTripError(
            str(response.get("error", "microworker request failed")),
            metrics=metrics,
            response=response,
        )
    return response, metrics


def combine_transport_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine serial retry attempts without hiding failed transport work."""

    if not attempts:
        return {
            "request_bytes": 0,
            "response_bytes": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connect_ns": 0,
            "new_connection_count": 0,
            "elapsed_ns": 0,
            "client_cpu_ns": 0,
            "attempt_count": 0,
            "retry_count": 0,
            "attempt_metrics": [],
        }
    combined = dict(attempts[-1])
    for field in (
        "request_bytes",
        "response_bytes",
        "messages_sent",
        "messages_received",
        "connect_ns",
        "new_connection_count",
        "elapsed_ns",
        "client_cpu_ns",
    ):
        combined[field] = sum(int(item.get(field, 0)) for item in attempts)
    combined["attempt_count"] = len(attempts)
    combined["retry_count"] = len(attempts) - 1
    combined["connection_reused"] = all(
        bool(item.get("connection_reused", False)) for item in attempts
    )
    combined["attempt_metrics"] = attempts
    return combined


def retry_attempt_timeout_s(*, deadline_unix_ns: int, maximum_attempts: int, attempt: int) -> float:
    """Return a bounded per-attempt timeout while preserving deadline headroom.

    A retried child cannot consume the operation's entire remaining deadline on
    its first socket wait.  Equal retry slices plus one reserved slice leave
    time for the failure to reduce back through the hierarchy.
    """

    remaining_s = (deadline_unix_ns - time.time_ns()) / 1_000_000_000
    if remaining_s <= 0:
        raise TimeoutError("microworker operation deadline elapsed before retry")
    if maximum_attempts <= 0:
        return max(0.001, remaining_s)
    attempts_remaining = maximum_attempts - attempt + 1
    return max(0.001, remaining_s / (attempts_remaining + 1))


def retryable_round_trip_error(error: BaseException) -> bool:
    """Return whether retrying an identical delegated request is meaningful."""

    response = getattr(error, "response", None)
    response_error_type = ""
    response_error = ""
    if isinstance(response, dict):
        response_error_type = str(response.get("error_type", ""))
        response_error = str(response.get("error", ""))
    description = " ".join(
        (type(error).__name__, str(error), response_error_type, response_error)
    ).lower()
    return "cancel" not in description and "deadline" not in description


class PersistentChannel:
    """One synchronized reusable TCP request/response session."""

    def __init__(self, endpoint: str, sender_id: str, receiver_id: str) -> None:
        self.endpoint = endpoint
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.connection: socket.socket | None = None
        self.lock = threading.Lock()

    def round_trip(
        self,
        *,
        message: dict[str, Any],
        profile: LinkProfile,
        timeout_s: float,
        attempt: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        with self.lock:
            created = False
            connect_ns = 0
            if self.connection is None:
                host, port = parse_endpoint(self.endpoint)
                connect_started_ns = time.perf_counter_ns()
                self.connection = socket.create_connection((host, port), timeout=timeout_s)
                connect_ns = time.perf_counter_ns() - connect_started_ns
                created = True
            try:
                response, metrics = shaped_round_trip(
                    endpoint=self.endpoint,
                    message=message,
                    profile=profile,
                    timeout_s=timeout_s,
                    sender_id=self.sender_id,
                    receiver_id=self.receiver_id,
                    attempt=attempt,
                    connection=self.connection,
                )
            except RoundTripError as error:
                error.metrics["connect_ns"] = int(error.metrics["connect_ns"]) + connect_ns
                error.metrics["new_connection_count"] = int(created)
                error.metrics["connection_reused"] = not created
                error.metrics["elapsed_ns"] = time.perf_counter_ns() - started_ns
                error.metrics["client_cpu_ns"] = time.process_time_ns() - cpu_started_ns
                # A complete protocol error frame leaves a persistent stream
                # synchronized.  Transport failures and dropped responses do not.
                if error.response is None:
                    self.close()
                raise
            except BaseException:
                self.close()
                raise
        metrics["connect_ns"] = connect_ns
        metrics["new_connection_count"] = int(created)
        metrics["connection_reused"] = not created
        metrics["elapsed_ns"] = time.perf_counter_ns() - started_ns
        metrics["client_cpu_ns"] = time.process_time_ns() - cpu_started_ns
        return response, metrics

    def close(self) -> None:
        if self.connection is not None:
            with suppress(OSError):
                self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            self.connection = None


class MicroworkerServer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.worker_id = str(config["worker_id"])
        self.worker_index = int(config["worker_index"])
        self.host = str(config.get("host", "127.0.0.1"))
        self.port = int(config.get("port", 0))
        self.ready_path = Path(config["ready_path"])
        self.trace_path = Path(config["trace_path"])
        self.log_path = Path(config["log_path"])
        runtime_profile = dict(config.get("runtime_profile", {}))
        self.real_model_shard_config = dict(runtime_profile.get("real_model_shard", {}))
        self.real_model_weight: Any | None = None
        self.model_shard_proof: dict[str, Any] | None = None
        self.runtime_profile: dict[str, Any] = {
            "name": str(runtime_profile.get("name", "homogeneous")),
            "compute_delay_ms": float(runtime_profile.get("compute_delay_ms", 0.0)),
            "capacity_score": float(runtime_profile.get("capacity_score", 1.0)),
            "maximum_payload_bytes": int(
                runtime_profile.get("maximum_payload_bytes", MAX_FRAME_BYTES)
            ),
        }
        if self.runtime_profile["compute_delay_ms"] < 0:
            raise ValueError("worker compute delay must be non-negative")
        if self.runtime_profile["capacity_score"] <= 0:
            raise ValueError("worker capacity score must be positive")
        if self.runtime_profile["maximum_payload_bytes"] < 0:
            raise ValueError("worker maximum payload must be non-negative")
        self.shutdown_event = threading.Event()
        self.trace_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.topology_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.topology: dict[str, Any] | None = None
        self.child_channels: dict[str, PersistentChannel] = {}
        self.child_channels_lock = threading.Lock()
        self.cancelled_operations: set[str] = set()
        self.injected_faults: set[tuple[str, str]] = set()
        self.fault_lock = threading.Lock()
        self.response_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self.metrics = {
            "requests_completed": 0,
            "requests_rejected": 0,
            "duplicate_requests": 0,
            "delegated_requests": 0,
            "child_rpcs": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
            "maximum_active_requests": 0,
            "active_requests": 0,
            "persistent_sessions_accepted": 0,
            "persistent_reused_requests": 0,
        }
        if self.real_model_shard_config:
            self._load_real_model_shard()

    def _load_real_model_shard(self) -> None:
        """Load and verify one immutable vocabulary-parallel output-head slice."""

        import torch
        from safetensors import safe_open

        config = self.real_model_shard_config
        manifest_path = Path(config["manifest_path"]).resolve()
        weight_path = Path(config["weight_file"]).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tensor_name = str(config.get("tensor_name", "lm_head.weight"))
        matches = [
            dict(item)
            for item in manifest["tensors"]
            if str(item.get("tensor_name")) == tensor_name
        ]
        if len(matches) != 1:
            raise ValueError(f"rank manifest must declare one {tensor_name!r} tensor")
        tensor_record = matches[0]
        if str(manifest["model_id"]) != str(config["model_id"]):
            raise ValueError("real model shard ID differs from its worker configuration")
        if str(manifest["model_revision"]) != str(config["model_revision"]):
            raise ValueError("real model shard revision differs from its worker configuration")
        if int(manifest["rank"]) != self.worker_index:
            raise ValueError("real model rank must equal the microworker index")
        if str(tensor_record["partition_mode"]) != "vocabulary_lm_head":
            raise ValueError("real model shard is not a vocabulary output-head partition")
        torch.set_num_threads(int(config.get("torch_threads", 1)))
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            source_tensor = handle.get_tensor(tensor_name).contiguous()
        source_bytes = source_tensor.view(torch.uint8).numpy().tobytes()
        local_tensor_hash = hashlib.sha256(source_bytes).hexdigest()
        if local_tensor_hash != str(tensor_record["local_tensor_hash"]):
            raise ValueError("loaded real-model tensor hash differs from the rank manifest")
        local_shape = [int(value) for value in tensor_record["local_shape"]]
        if list(source_tensor.shape) != local_shape:
            raise ValueError("loaded real-model tensor shape differs from the rank manifest")
        global_shape = [int(value) for value in tensor_record["global_shape"]]
        token_start = int(tensor_record["shard_start"])
        token_end = int(tensor_record["shard_end"])
        if token_end - token_start != local_shape[0] or local_shape[1] != global_shape[1]:
            raise ValueError("real-model vocabulary shard geometry is inconsistent")
        self.real_model_weight = source_tensor.float().contiguous()
        self.model_shard_proof = {
            "model_id": str(manifest["model_id"]),
            "model_revision": str(manifest["model_revision"]),
            "logical_rank_id": str(manifest["logical_rank_id"]),
            "rank": int(manifest["rank"]),
            "tensor_name": tensor_name,
            "source_dtype": str(tensor_record["dtype"]),
            "compute_dtype": "float32",
            "local_shape": local_shape,
            "global_shape": global_shape,
            "token_start": token_start,
            "token_end": token_end,
            "local_tensor_hash": local_tensor_hash,
            "expected_local_tensor_hash": str(tensor_record["local_tensor_hash"]),
            "source_tensor_hash": str(tensor_record["source_tensor_hash"]),
            "weight_file": str(weight_path),
            "weight_file_hash": str(manifest["weight_file_hash"]),
            "rank_manifest": str(manifest_path),
            "loaded_source_tensor_bytes": len(source_bytes),
            "resident_compute_tensor_bytes": int(self.real_model_weight.numel() * 4),
            "complete_output_head_loaded": local_shape[0] == global_shape[0],
            "complete_model_loaded": False,
            "evidence_directory": str(Path(config["evidence_directory"]).resolve()),
        }
        self._trace("real_model_shard_loaded", proof=self.model_shard_proof)

    def _real_model_local_aggregate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Compute one real vocabulary shard and persist its tensor evidence."""

        import torch

        if self.real_model_weight is None or self.model_shard_proof is None:
            raise RuntimeError("real-model operation reached a worker without a model shard")
        workload = dict(request.get("workload", {}))
        proof = self.model_shard_proof
        for field in ("model_id", "model_revision", "tensor_name"):
            if str(workload.get(field)) != str(proof[field]):
                raise ValueError(f"real-model workload {field} differs from the loaded shard")
        if int(workload.get("hidden_size", -1)) != int(proof["global_shape"][1]):
            raise ValueError("real-model hidden size differs from the loaded output head")
        if int(workload.get("vocabulary_size", -1)) != int(proof["global_shape"][0]):
            raise ValueError("real-model vocabulary size differs from the loaded output head")
        if workload.get("hidden_dtype") != "float32-le":
            raise ValueError("real-model hidden payload must be little-endian float32")
        payload = base64.b64decode(str(request.get("payload_b64", "")), validate=True)
        expected_bytes = int(proof["global_shape"][1]) * 4
        if len(payload) != expected_bytes:
            raise ValueError("real-model hidden payload has the wrong byte length")
        hidden = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
        compute_started_ns = time.perf_counter_ns()
        logits = torch.mv(self.real_model_weight, hidden).contiguous()
        compute_elapsed_ns = time.perf_counter_ns() - compute_started_ns
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("real-model local logits contain non-finite values")
        local_index = int(torch.argmax(logits).item())
        score = float(logits[local_index].item())
        token_id = int(proof["token_start"]) + local_index
        logits_bytes = logits.view(torch.uint8).numpy().tobytes()
        logits_digest = hashlib.sha256(logits_bytes).hexdigest()
        evidence_directory = Path(str(proof["evidence_directory"]))
        evidence_directory.mkdir(parents=True, exist_ok=True)
        evidence_id = hashlib.sha256(
            f"{request['operation_id']}|{self.worker_id}".encode()
        ).hexdigest()[:24]
        tensor_path = evidence_directory / f"{evidence_id}.logits.f32"
        metadata_path = evidence_directory / f"{evidence_id}.json"
        tensor_path.write_bytes(logits_bytes)
        metadata = {
            "schema_version": "1.0",
            "operation_id": str(request["operation_id"]),
            "request_id": str(request["request_id"]),
            "worker_id": self.worker_id,
            "process_id": os.getpid(),
            "model_id": proof["model_id"],
            "model_revision": proof["model_revision"],
            "rank": proof["rank"],
            "token_start": proof["token_start"],
            "token_end": proof["token_end"],
            "logit_count": int(logits.numel()),
            "logits_dtype": "float32-le",
            "logits_sha256": logits_digest,
            "tensor_path": str(tensor_path),
            "token_id": token_id,
            "score": score,
            "compute_elapsed_ns": compute_elapsed_ns,
        }
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, metadata_path)
        shard = {
            "worker_id": self.worker_id,
            "process_id": os.getpid(),
            "rank": int(proof["rank"]),
            "token_start": int(proof["token_start"]),
            "token_end": int(proof["token_end"]),
            "logits_sha256": logits_digest,
            "local_tensor_hash": str(proof["local_tensor_hash"]),
            "loaded_tensor_bytes": int(proof["loaded_source_tensor_bytes"]),
        }
        aggregate = {
            "mode": "vocabulary_argmax",
            "score": score,
            "score_float32_hex": struct.pack("!f", score).hex(),
            "token_id": token_id,
            "winner_worker_id": self.worker_id,
            "contribution_count": 1,
            "local_logit_count": int(logits.numel()),
            "shards": [shard],
        }
        self._trace(
            "real_model_shard_computed",
            operation_id=request["operation_id"],
            request_id=request["request_id"],
            token_id=token_id,
            score=score,
            logits_sha256=logits_digest,
            logit_count=int(logits.numel()),
            compute_elapsed_ns=compute_elapsed_ns,
            evidence_metadata_path=str(metadata_path),
            evidence_tensor_path=str(tensor_path),
        )
        return aggregate

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.trace_lock, path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded + "\n")

    def _persistent_child_channel(self, child: dict[str, Any]) -> PersistentChannel:
        worker_id = str(child["worker_id"])
        with self.child_channels_lock:
            channel = self.child_channels.get(worker_id)
            if channel is None:
                channel = PersistentChannel(str(child["endpoint"]), self.worker_id, worker_id)
                self.child_channels[worker_id] = channel
            elif channel.endpoint != str(child["endpoint"]):
                channel.close()
                channel = PersistentChannel(str(child["endpoint"]), self.worker_id, worker_id)
                self.child_channels[worker_id] = channel
            return channel

    def _trace(self, event: str, **payload: Any) -> None:
        self._append_jsonl(
            self.trace_path,
            {
                "event": event,
                "worker_id": self.worker_id,
                "worker_index": self.worker_index,
                "process_id": os.getpid(),
                "timestamp_unix_ns": time.time_ns(),
                **payload,
            },
        )

    def _interruptible_delay(
        self, *, operation_id: str, deadline_unix_ns: int, delay_ms: float
    ) -> None:
        remaining_s = delay_ms / 1_000.0
        while remaining_s > 0:
            if operation_id in self.cancelled_operations:
                raise RuntimeError("delegated operation was cancelled during worker compute")
            if time.time_ns() >= deadline_unix_ns:
                raise TimeoutError("delegated deadline elapsed during worker compute")
            interval_s = min(0.01, remaining_s)
            time.sleep(interval_s)
            remaining_s -= interval_s

    def _leaf_response(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_operation_envelope(request, allow_children=False)
        partition = request["work_partition"]
        if str(partition.get("worker_id")) != self.worker_id:
            raise ValueError("work partition worker identity mismatch")
        if int(partition.get("worker_index", -1)) != self.worker_index:
            raise ValueError("work partition worker index mismatch")
        contribution = worker_contribution(self.worker_index)
        aggregate = leaf_aggregate(self.worker_id, contribution)
        return {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "result",
            "request_id": request["request_id"],
            "operation_id": request["operation_id"],
            "execution_generation": request["execution_generation"],
            "worker_id": self.worker_id,
            "parent_worker": request["parent_worker"],
            "aggregate": aggregate,
            "aggregate_digest": aggregate_digest(aggregate),
            "status": "ok",
        }

    def _validate_topology_install(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("magic") != MAGIC or request.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("unsupported topology-install protocol identity")
        topology_id = str(request.get("topology_id", ""))
        route_lease_id = str(request.get("route_lease_id", ""))
        route_generation = int(request.get("route_generation", 0))
        node = request.get("node")
        if (
            not topology_id
            or not route_lease_id
            or route_generation <= 0
            or not isinstance(node, dict)
        ):
            raise ValueError("topology install identity is incomplete")
        if (
            node.get("worker_id") != self.worker_id
            or int(node.get("worker_index", -1)) != self.worker_index
        ):
            raise ValueError("topology install targets the wrong worker")
        children = node.get("children")
        if not isinstance(children, list) or len(children) > 32:
            raise ValueError("topology children must be a bounded list")
        child_ids = [str(child.get("worker_id", "")) for child in children]
        if any(not child_id for child_id in child_ids) or len(set(child_ids)) != len(child_ids):
            raise ValueError("topology child identities must be unique and non-empty")
        for child in children:
            parse_endpoint(str(child.get("endpoint", "")))
            assigned = child.get("assigned_child_worker_ids")
            if not isinstance(assigned, list) or len(assigned) > 32:
                raise ValueError("child route must describe bounded assigned children")
            if int(child.get("subtree_worker_count", 0)) <= 0:
                raise ValueError("child route subtree count must be positive")
        parent_worker = str(node.get("parent_worker", ""))
        if not parent_worker:
            raise ValueError("topology node parent identity is required")
        if int(node.get("subtree_worker_count", 0)) <= 0:
            raise ValueError("topology node subtree count must be positive")
        partition_indices = node.get("partition_worker_indices")
        if (
            not isinstance(partition_indices, list)
            or len(partition_indices) != int(node["subtree_worker_count"])
            or len({int(index) for index in partition_indices}) != len(partition_indices)
        ):
            raise ValueError("topology partition must identify every subtree worker exactly once")
        installed = {
            "topology_id": topology_id,
            "route_lease_id": route_lease_id,
            "route_generation": route_generation,
            "node": node,
        }
        with self.topology_lock:
            current = self.topology
            if current is not None and route_generation < int(current["route_generation"]):
                raise ValueError("stale topology generation")
            self.topology = installed
        self._trace(
            "topology_installed",
            topology_id=topology_id,
            route_lease_id=route_lease_id,
            route_generation=route_generation,
            parent_worker=parent_worker,
            child_worker_ids=child_ids,
            subtree_worker_count=int(node["subtree_worker_count"]),
        )
        return installed

    @staticmethod
    def _empty_subtree_metrics() -> dict[str, Any]:
        return {
            "total_messages": 0,
            "total_bytes": 0,
            "worker_to_worker_rpc_count": 0,
            "leaf_rpc_count": 0,
            "serial_waits": 0,
            "barrier_waits": 0,
            "parallel_dispatch_nodes": 0,
            "critical_path_sync_points": 0,
            "hierarchy_depth": 1,
            "fanout_depth": 1,
            "reduction_depth": 0,
            "intermediate_reductions": 0,
            "connection_count": 0,
            "connection_ns": 0,
            "worker_cpu_ns": 0,
            "retries": 0,
            "failures": 0,
            "duplicated_work": 0,
            "stragglers": 0,
            "simulated_compute_delay_ms_sum": 0.0,
            "critical_path_compute_delay_ms": 0.0,
            "worker_profile_counts": {},
            "maximum_queue_depth": 1,
            "latency_histogram": {},
            "leaf_latency_histogram": {},
        }

    def _delegated_response(self, request: dict[str, Any]) -> dict[str, Any]:
        validate_operation_envelope(request, allow_children=True)
        with self.topology_lock:
            installed = self.topology
        if installed is None:
            raise RuntimeError("delegated operation has no installed topology")
        node = installed["node"]
        if (
            int(request["route_generation"]) != int(installed["route_generation"])
            or request["route_lease_id"] != installed["route_lease_id"]
            or request["parent_worker"] != node["parent_worker"]
        ):
            raise ValueError("delegated operation route, generation, or parent is stale")
        partition = request["work_partition"]
        if (
            partition.get("worker_id") != self.worker_id
            or int(partition.get("worker_index", -1)) != self.worker_index
            or int(partition.get("partition_start", -1)) != int(node["partition_start"])
            or int(partition.get("partition_end", -1)) != int(node["partition_end"])
            or list(partition.get("partition_worker_indices", []))
            != list(node["partition_worker_indices"])
            or int(partition.get("subtree_worker_count", -1)) != int(node["subtree_worker_count"])
        ):
            raise ValueError("delegated work partition differs from the installed subtree")
        children = sorted(node["children"], key=lambda child: child["ordering_key"])
        expected_child_ids = [child["worker_id"] for child in children]
        if request["assigned_child_workers"] != expected_child_ids:
            raise ValueError("delegated assigned children differ from the installed topology")
        operation_id = str(request["operation_id"])
        if operation_id in self.cancelled_operations:
            raise RuntimeError("delegated operation was cancelled at this worker")
        request_hash = hashlib.sha256(canonical_json_bytes(request)).hexdigest()
        with self.cache_lock:
            cached = self.response_cache.get(str(request["request_id"]))
            if cached is not None:
                if cached[0] != request_hash:
                    raise ValueError("duplicate delegated request ID has different content")
                with self.metrics_lock:
                    self.metrics["duplicate_requests"] += 1
                self._trace(
                    "duplicate_delegated_request_replayed",
                    operation_id=operation_id,
                    request_id=request["request_id"],
                )
                return dict(cached[1])

        cpu_started_ns = time.process_time_ns()
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
                "controlled_fault_injected",
                operation_id=operation_id,
                request_id=request["request_id"],
                fault_kind=fault_kind,
                phase="before_compute",
            )
            raise RuntimeError(f"controlled {fault_kind} at {self.worker_id}")
        aggregation = dict(request["aggregation"])
        aggregation_mode = str(aggregation["mode"])
        metrics = self._empty_subtree_metrics()
        child_depths: list[int] = []
        child_reduction_depths: list[int] = []
        critical_path_sync_points = 0
        profile = LinkProfile.from_dict(dict(request["network_profile"]))
        payload_bytes = len(base64.b64decode(str(request.get("payload_b64", "")), validate=True))
        if payload_bytes > int(self.runtime_profile["maximum_payload_bytes"]):
            raise MemoryError("delegated payload exceeds declared worker memory limit")
        local_compute_delay_ms = float(self.runtime_profile["compute_delay_ms"])
        if local_compute_delay_ms:
            self._interruptible_delay(
                operation_id=operation_id,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                delay_ms=local_compute_delay_ms,
            )
        if fault_kind == "slow_child" and inject_once("after_compute"):
            extra_delay_ms = float(fault_control.get("delay_ms", 0.0))
            if extra_delay_ms < 0:
                raise ValueError("controlled slow-child delay must be non-negative")
            self._trace(
                "controlled_fault_injected",
                operation_id=operation_id,
                request_id=request["request_id"],
                fault_kind=fault_kind,
                phase="after_compute",
                delay_ms=extra_delay_ms,
            )
            self._interruptible_delay(
                operation_id=operation_id,
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                delay_ms=extra_delay_ms,
            )
        profile_name = str(self.runtime_profile["name"])
        metrics["simulated_compute_delay_ms_sum"] = local_compute_delay_ms
        metrics["critical_path_compute_delay_ms"] = local_compute_delay_ms
        metrics["worker_profile_counts"] = {profile_name: 1}
        metrics["stragglers"] = int(profile_name == "straggler")
        if aggregation_mode == "vocabulary_argmax":
            aggregate_items = [self._real_model_local_aggregate(request)]
        else:
            aggregate_items = [
                leaf_aggregate(self.worker_id, worker_contribution(self.worker_index))
            ]
            self._trace(
                "simulated_worker_compute",
                operation_id=operation_id,
                request_id=request["request_id"],
                runtime_profile=profile_name,
                compute_delay_ms=local_compute_delay_ms,
                capacity_score=self.runtime_profile["capacity_score"],
            )
        trace_context = request["trace_context"]
        dispatch_policy = str(request.get("dispatch_policy", "serial"))
        if dispatch_policy not in {"serial", "parallel"}:
            raise ValueError("delegated dispatch policy must be serial or parallel")
        connection_policy = str(request["connection_policy"])

        def child_round_trip(
            child: dict[str, Any],
        ) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
            if time.time_ns() >= int(request["deadline_unix_ns"]):
                raise TimeoutError("delegated deadline elapsed before child dispatch")
            if operation_id in self.cancelled_operations:
                raise RuntimeError("delegated operation was cancelled during fanout")
            child_request_id = f"{request['request_id']}:{child['worker_id']}"
            child_request = make_delegated_request(
                request_id=child_request_id,
                operation_id=operation_id,
                execution_generation=int(request["execution_generation"]),
                parent_worker=self.worker_id,
                worker_id=child["worker_id"],
                worker_index=int(child["worker_index"]),
                assigned_child_workers=list(child["assigned_child_worker_ids"]),
                partition_start=int(child["partition_start"]),
                partition_end=int(child["partition_end"]),
                partition_worker_indices=list(child["partition_worker_indices"]),
                subtree_worker_count=int(child["subtree_worker_count"]),
                deadline_unix_ns=int(request["deadline_unix_ns"]),
                route_lease_id=str(request["route_lease_id"]),
                ordering_key=str(child["ordering_key"]),
                payload_bytes=payload_bytes,
                trace_id=str(trace_context["trace_id"]),
                parent_span_id=str(trace_context["span_id"]),
                span_id=f"{self.worker_id}-to-{child['worker_id']}",
                network_profile=profile,
                route_generation=int(request["route_generation"]),
                retry_policy=dict(request["retry_policy"]),
                cancellation=dict(request["cancellation"]),
                fault_control=fault_control,
                dispatch_policy=dispatch_policy,
                connection_policy=connection_policy,
                aggregation=aggregation,
                workload=(
                    dict(request["workload"]) if isinstance(request.get("workload"), dict) else None
                ),
                payload_b64=str(request["payload_b64"]),
            )
            child_started_ns = time.perf_counter_ns()
            maximum_attempts = int(request["retry_policy"].get("max_attempts", 0))
            backoff_ms = float(request["retry_policy"].get("backoff_ms", 0.0))
            attempt_metrics: list[dict[str, Any]] = []
            child_response: dict[str, Any] | None = None
            final_error: BaseException | None = None
            for attempt in range(maximum_attempts + 1):
                timeout_s = retry_attempt_timeout_s(
                    deadline_unix_ns=int(request["deadline_unix_ns"]),
                    maximum_attempts=maximum_attempts,
                    attempt=attempt,
                )
                try:
                    if connection_policy == "persistent":
                        child_response, transport = self._persistent_child_channel(
                            child
                        ).round_trip(
                            message=child_request,
                            profile=profile,
                            timeout_s=timeout_s,
                            attempt=attempt,
                        )
                    else:
                        child_response, transport = shaped_round_trip(
                            endpoint=str(child["endpoint"]),
                            message=child_request,
                            profile=profile,
                            timeout_s=timeout_s,
                            sender_id=self.worker_id,
                            receiver_id=str(child["worker_id"]),
                            attempt=attempt,
                        )
                    if (
                        child_response.get("operation_id") != operation_id
                        or child_response.get("request_id") != child_request_id
                        or int(child_response.get("execution_generation", 0))
                        != int(request["execution_generation"])
                    ):
                        raise RoundTripError(
                            "delegated child response identity mismatch",
                            metrics=transport,
                            response=child_response,
                        )
                    child_aggregate = dict(child_response["aggregate"])
                    if child_response.get("aggregate_digest") != aggregate_digest(child_aggregate):
                        raise RoundTripError(
                            "delegated child aggregate digest mismatch",
                            metrics=transport,
                            response=child_response,
                        )
                    attempt_metrics.append(transport)
                    final_error = None
                    break
                except BaseException as error:
                    final_error = error
                    failure_metrics = dict(getattr(error, "metrics", {}))
                    failure_metrics.setdefault("attempt", attempt)
                    attempt_metrics.append(failure_metrics)
                    will_retry = attempt < maximum_attempts and retryable_round_trip_error(error)
                    self._trace(
                        "worker_child_round_trip_failed",
                        operation_id=operation_id,
                        request_id=child_request_id,
                        sender_worker_id=self.worker_id,
                        receiver_worker_id=child["worker_id"],
                        attempt=attempt,
                        will_retry=will_retry,
                        error_type=type(error).__name__,
                        error=str(error),
                        elapsed_ns=time.perf_counter_ns() - child_started_ns,
                        transport_metrics=failure_metrics,
                    )
                    if not will_retry:
                        break
                    self._trace(
                        "worker_child_retry",
                        operation_id=operation_id,
                        request_id=child_request_id,
                        sender_worker_id=self.worker_id,
                        receiver_worker_id=child["worker_id"],
                        next_attempt=attempt + 1,
                    )
                    if backoff_ms:
                        self._interruptible_delay(
                            operation_id=operation_id,
                            deadline_unix_ns=int(request["deadline_unix_ns"]),
                            delay_ms=backoff_ms,
                        )
            transport = combine_transport_attempts(attempt_metrics)
            if final_error is not None or child_response is None:
                if final_error is None:
                    final_error = RuntimeError("delegated child retry exhausted")
                final_error.metrics = transport  # type: ignore[attr-defined]
                raise final_error
            if (
                child_response.get("operation_id") != operation_id
                or child_response.get("request_id") != child_request_id
                or int(child_response.get("execution_generation", 0))
                != int(request["execution_generation"])
            ):
                raise ValueError("delegated child response identity mismatch")
            child_aggregate = dict(child_response["aggregate"])
            if child_response.get("aggregate_digest") != aggregate_digest(child_aggregate):
                raise ValueError("delegated child aggregate digest mismatch")
            self._trace(
                "worker_child_round_trip",
                operation_id=operation_id,
                request_id=child_request_id,
                sender_process_id=os.getpid(),
                receiver_process_id=child["process_id"],
                aggregation=aggregation_mode,
                reduction_order=expected_child_ids,
                dispatch_policy=dispatch_policy,
                connection_policy=connection_policy,
                **transport,
            )
            return child, child_request_id, child_response, transport

        child_results: list[tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]] = []
        if dispatch_policy == "parallel" and len(children) > 1:
            child_errors: list[BaseException] = []
            result_lock = threading.Lock()

            def capture_child(child: dict[str, Any]) -> None:
                try:
                    result = child_round_trip(child)
                    with result_lock:
                        child_results.append(result)
                except BaseException as error:
                    with result_lock:
                        child_errors.append(error)

            child_threads = [
                threading.Thread(
                    target=capture_child,
                    args=(child,),
                    name=f"delegate-{self.worker_id}-{child['worker_id']}",
                )
                for child in children
            ]
            for thread in child_threads:
                thread.start()
            for thread in child_threads:
                thread.join()
            if child_errors:
                raise child_errors[0]
        else:
            child_results = [child_round_trip(child) for child in children]
        child_results.sort(key=lambda item: item[0]["ordering_key"])

        child_critical_paths: list[int] = []
        child_compute_paths: list[float] = []
        for child, _child_request_id, child_response, transport in child_results:
            child_aggregate = dict(child_response["aggregate"])
            child_metrics = dict(child_response["subtree_metrics"])
            aggregate_items.append(child_aggregate)
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
            metrics["leaf_rpc_count"] += int(child_metrics["leaf_rpc_count"]) + (
                1 if not child["assigned_child_worker_ids"] else 0
            )
            metrics["serial_waits"] += int(child_metrics["serial_waits"])
            metrics["barrier_waits"] += int(child_metrics["barrier_waits"])
            metrics["parallel_dispatch_nodes"] += int(child_metrics["parallel_dispatch_nodes"])
            metrics["connection_count"] += int(transport["new_connection_count"]) + int(
                child_metrics["connection_count"]
            )
            metrics["connection_ns"] += int(transport["connect_ns"]) + int(
                child_metrics["connection_ns"]
            )
            metrics["worker_cpu_ns"] += int(child_metrics["worker_cpu_ns"])
            metrics["retries"] += int(transport["retry_count"]) + int(child_metrics["retries"])
            metrics["failures"] += int(transport["retry_count"]) + int(child_metrics["failures"])
            metrics["duplicated_work"] += int(child_metrics["duplicated_work"])
            metrics["stragglers"] += int(child_metrics["stragglers"])
            metrics["simulated_compute_delay_ms_sum"] += float(
                child_metrics["simulated_compute_delay_ms_sum"]
            )
            for name, count in dict(child_metrics["worker_profile_counts"]).items():
                profile_counts = dict(metrics["worker_profile_counts"])
                profile_counts[str(name)] = profile_counts.get(str(name), 0) + int(count)
                metrics["worker_profile_counts"] = profile_counts
            metrics["intermediate_reductions"] += int(child_metrics["intermediate_reductions"])
            metrics["maximum_queue_depth"] = max(
                int(metrics["maximum_queue_depth"]),
                int(child_metrics["maximum_queue_depth"]),
            )
            histogram = dict(metrics["latency_histogram"])
            latency_histogram_observe(histogram, int(transport["elapsed_ns"]))
            metrics["latency_histogram"] = combine_latency_histograms(
                histogram, dict(child_metrics["latency_histogram"])
            )
            leaf_histogram = combine_latency_histograms(
                dict(metrics["leaf_latency_histogram"]),
                dict(child_metrics["leaf_latency_histogram"]),
            )
            if not child["assigned_child_worker_ids"]:
                latency_histogram_observe(leaf_histogram, int(transport["elapsed_ns"]))
            metrics["leaf_latency_histogram"] = leaf_histogram
            child_depths.append(int(child_metrics["hierarchy_depth"]))
            child_reduction_depths.append(int(child_metrics["reduction_depth"]))
            child_critical_paths.append(1 + int(child_metrics["critical_path_sync_points"]))
            child_compute_paths.append(float(child_metrics["critical_path_compute_delay_ms"]))

        if dispatch_policy == "parallel" and children:
            metrics["barrier_waits"] += 1
            metrics["parallel_dispatch_nodes"] += 1
            metrics["maximum_queue_depth"] = max(int(metrics["maximum_queue_depth"]), len(children))
            critical_path_sync_points = max(child_critical_paths, default=0)
        else:
            metrics["serial_waits"] += len(children)
            critical_path_sync_points = sum(child_critical_paths)
        if operation_id in self.cancelled_operations:
            raise RuntimeError("delegated operation was cancelled before reduction")
        aggregate = combine_operation_aggregates(aggregate_items, aggregation)
        metrics["critical_path_sync_points"] = critical_path_sync_points
        metrics["critical_path_compute_delay_ms"] = local_compute_delay_ms + max(
            child_compute_paths, default=0.0
        )
        metrics["hierarchy_depth"] = 1 + max(child_depths, default=0)
        metrics["fanout_depth"] = metrics["hierarchy_depth"]
        metrics["reduction_depth"] = 1 + max(child_reduction_depths, default=0) if children else 0
        metrics["intermediate_reductions"] += 1 if children else 0
        metrics["worker_cpu_ns"] += time.process_time_ns() - cpu_started_ns
        response = {
            "magic": MAGIC,
            "protocol_version": PROTOCOL_VERSION,
            "kind": "result",
            "request_id": request["request_id"],
            "operation_id": operation_id,
            "execution_generation": request["execution_generation"],
            "worker_id": self.worker_id,
            "parent_worker": request["parent_worker"],
            "aggregate": aggregate,
            "aggregate_digest": aggregate_digest(aggregate),
            "subtree_metrics": metrics,
            "status": "ok",
        }
        with self.cache_lock:
            self.response_cache[str(request["request_id"])] = (request_hash, response)
            if len(self.response_cache) > 256:
                self.response_cache.pop(next(iter(self.response_cache)))
        with self.metrics_lock:
            self.metrics["delegated_requests"] += 1
            self.metrics["child_rpcs"] += len(children)
        self._trace(
            "delegated_subtree_reduced",
            operation_id=operation_id,
            request_id=request["request_id"],
            parent_worker=request["parent_worker"],
            child_worker_ids=expected_child_ids,
            contribution_count=aggregate["contribution_count"],
            aggregate_digest=response["aggregate_digest"],
            hierarchy_depth=metrics["hierarchy_depth"],
            reduction_depth=metrics["reduction_depth"],
            serial_waits=metrics["serial_waits"],
            barrier_waits=metrics["barrier_waits"],
            dispatch_policy=dispatch_policy,
            connection_policy=connection_policy,
            runtime_profile=profile_name,
            local_compute_delay_ms=local_compute_delay_ms,
            critical_path_compute_delay_ms=metrics["critical_path_compute_delay_ms"],
            aggregation_mode=aggregation_mode,
        )
        return response

    def _apply_response_fault(
        self, request: dict[str, Any], response: dict[str, Any]
    ) -> dict[str, Any]:
        fault_control = dict(request.get("fault_control", {}))
        fault_kind = str(fault_control.get("kind", ""))
        targets = {str(value) for value in fault_control.get("target_worker_ids", [])}
        if self.worker_id not in targets or fault_kind not in {
            "drop_result_once",
            "stale_generation_once",
        }:
            return response
        key = (str(request.get("operation_id")), f"{fault_kind}:response")
        with self.fault_lock:
            if bool(fault_control.get("one_shot", False)) and key in self.injected_faults:
                return response
            self.injected_faults.add(key)
        self._trace(
            "controlled_fault_injected",
            operation_id=request.get("operation_id"),
            request_id=request.get("request_id"),
            fault_kind=fault_kind,
            phase="response",
        )
        if fault_kind == "drop_result_once":
            raise DropResponse(f"controlled dropped result at {self.worker_id}")
        stale = dict(response)
        stale["execution_generation"] = int(response["execution_generation"]) - 1
        return stale

    def _cancel_operation(self, operation_id: str) -> dict[str, Any]:
        self.cancelled_operations.add(operation_id)
        with self.topology_lock:
            installed = self.topology
        children = [] if installed is None else list(installed["node"]["children"])
        records: list[dict[str, Any]] = []
        records_lock = threading.Lock()

        def cancel_child(child: dict[str, Any]) -> None:
            started_ns = time.perf_counter_ns()
            request_bytes = 0
            response_bytes = 0
            try:
                host, port = parse_endpoint(str(child["endpoint"]))
                connection = socket.create_connection((host, port), timeout=1.0)
                try:
                    connection.settimeout(1.0)
                    request_bytes = send_message(
                        connection,
                        {
                            "magic": MAGIC,
                            "protocol_version": PROTOCOL_VERSION,
                            "kind": "cancel_operation",
                            "operation_id": operation_id,
                        },
                    )
                    response, response_bytes = receive_message(connection)
                finally:
                    connection.close()
                record = {
                    "worker_id": child["worker_id"],
                    "status": response.get("status"),
                    "messages": 2,
                    "bytes": request_bytes + response_bytes,
                    "propagated_workers": int(response.get("propagated_workers", 1)),
                    "tree_messages": 2 + int(response.get("tree_messages", 0)),
                    "depth": 1 + int(response.get("cancellation_depth", 0)),
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                }
            except BaseException as error:
                record = {
                    "worker_id": child["worker_id"],
                    "status": "failed",
                    "messages": int(request_bytes > 0),
                    "bytes": request_bytes + response_bytes,
                    "propagated_workers": 0,
                    "tree_messages": int(request_bytes > 0),
                    "depth": 0,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            self._trace(
                "worker_cancel_round_trip",
                operation_id=operation_id,
                sender_worker_id=self.worker_id,
                receiver_worker_id=child["worker_id"],
                **record,
            )
            with records_lock:
                records.append(record)

        if children:
            threads = [
                threading.Thread(
                    target=cancel_child,
                    args=(child,),
                    name=f"cancel-{self.worker_id}-{child['worker_id']}",
                )
                for child in children
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        return {
            "propagated_workers": 1 + sum(int(record["propagated_workers"]) for record in records),
            "tree_messages": sum(int(record["tree_messages"]) for record in records),
            "cancellation_depth": max((int(record["depth"]) for record in records), default=0),
            "propagation_failures": sum(record["status"] != "ok" for record in records),
        }

    def _dispatch_request(self, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        kind = str(request.get("kind"))
        if kind == "leaf_execute":
            response = self._leaf_response(request)
        elif kind == "delegated_execute":
            response = self._apply_response_fault(request, self._delegated_response(request))
        elif kind == "install_topology":
            installed = self._validate_topology_install(request)
            response = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "topology_installed",
                "worker_id": self.worker_id,
                "topology_id": installed["topology_id"],
                "route_generation": installed["route_generation"],
                "status": "ok",
            }
        elif kind == "cancel_operation":
            operation_id = str(request.get("operation_id", ""))
            if not operation_id:
                raise ValueError("cancel operation identity is required")
            cancellation = self._cancel_operation(operation_id)
            response = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "operation_cancelled",
                "worker_id": self.worker_id,
                "operation_id": operation_id,
                "status": "ok",
                **cancellation,
            }
        elif kind == "status":
            with self.metrics_lock:
                snapshot = dict(self.metrics)
            with self.topology_lock:
                topology = self.topology
            response = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "status_result",
                "worker_id": self.worker_id,
                "process_id": os.getpid(),
                "metrics": snapshot,
                "topology_id": topology["topology_id"] if topology is not None else None,
                "route_generation": topology["route_generation"] if topology is not None else None,
            }
        elif kind == "shutdown":
            response = {
                "magic": MAGIC,
                "protocol_version": PROTOCOL_VERSION,
                "kind": "shutdown_result",
                "worker_id": self.worker_id,
                "status": "ok",
            }
            self.shutdown_event.set()
        else:
            raise ValueError(f"unsupported microworker request kind {kind!r}")
        return kind, response

    def _handle(self, connection: socket.socket, address: tuple[str, int]) -> None:
        connection.settimeout(0.5)
        persistent_session = False
        handled_requests = 0
        try:
            while not self.shutdown_event.is_set():
                request_started_ns = time.perf_counter_ns()
                request: dict[str, Any] = {}
                try:
                    request, received = receive_message(connection)
                except TimeoutError:
                    if persistent_session and handled_requests:
                        if self.shutdown_event.is_set():
                            break
                        continue
                    raise
                except ConnectionError as error:
                    if persistent_session and handled_requests:
                        self._trace(
                            "persistent_session_closed",
                            peer_host=address[0],
                            peer_port=address[1],
                            handled_requests=handled_requests,
                            close_reason=type(error).__name__,
                        )
                        break
                    raise
                with self.metrics_lock:
                    self.metrics["active_requests"] += 1
                    self.metrics["maximum_active_requests"] = max(
                        self.metrics["maximum_active_requests"],
                        self.metrics["active_requests"],
                    )
                try:
                    kind, response = self._dispatch_request(request)
                    sent = send_message(connection, response)
                    persistent_request = (
                        kind == "delegated_execute"
                        and request.get("connection_policy") == "persistent"
                    )
                    with self.metrics_lock:
                        self.metrics["requests_completed"] += 1
                        self.metrics["bytes_received"] += received
                        self.metrics["bytes_sent"] += sent
                        if persistent_request and persistent_session:
                            self.metrics["persistent_reused_requests"] += 1
                        elif persistent_request:
                            self.metrics["persistent_sessions_accepted"] += 1
                    handled_requests += 1
                    self._trace(
                        "server_request_completed",
                        request_id=request.get("request_id"),
                        operation_id=request.get("operation_id"),
                        request_kind=kind,
                        connection_policy=request.get("connection_policy", "ephemeral"),
                        persistent_reuse=persistent_request and persistent_session,
                        peer_host=address[0],
                        peer_port=address[1],
                        bytes_received=received,
                        bytes_sent=sent,
                        elapsed_ns=time.perf_counter_ns() - request_started_ns,
                    )
                    persistent_session = persistent_request
                except BaseException as error:
                    with self.metrics_lock:
                        self.metrics["requests_rejected"] += 1
                    if isinstance(error, DropResponse):
                        self._trace(
                            "server_response_dropped",
                            request_id=request.get("request_id"),
                            operation_id=request.get("operation_id"),
                            error=str(error),
                            elapsed_ns=time.perf_counter_ns() - request_started_ns,
                        )
                        break
                    failure = {
                        "magic": MAGIC,
                        "protocol_version": PROTOCOL_VERSION,
                        "kind": "error",
                        "request_id": request.get("request_id"),
                        "operation_id": request.get("operation_id"),
                        "worker_id": self.worker_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    sent = 0
                    with suppress(OSError):
                        sent = send_message(connection, failure)
                    persistent_request = (
                        request.get("kind") == "delegated_execute"
                        and request.get("connection_policy") == "persistent"
                    )
                    retain_session = persistent_request and sent > 0
                    if retain_session:
                        with self.metrics_lock:
                            self.metrics["bytes_received"] += received
                            self.metrics["bytes_sent"] += sent
                            if persistent_session:
                                self.metrics["persistent_reused_requests"] += 1
                            else:
                                self.metrics["persistent_sessions_accepted"] += 1
                        handled_requests += 1
                        persistent_session = True
                    self._trace(
                        "server_request_failed",
                        request_id=request.get("request_id"),
                        operation_id=request.get("operation_id"),
                        error_type=type(error).__name__,
                        error=str(error),
                        persistent_session_retained=retain_session,
                        elapsed_ns=time.perf_counter_ns() - request_started_ns,
                    )
                    if not retain_session:
                        break
                finally:
                    with self.metrics_lock:
                        self.metrics["active_requests"] -= 1
                if not persistent_session:
                    break
        finally:
            connection.close()

    def serve(self) -> int:
        self.ready_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
            server.listen(64)
            server.settimeout(0.2)
            actual_host, actual_port = server.getsockname()[:2]
            ready = {
                "worker_id": self.worker_id,
                "worker_index": self.worker_index,
                "process_id": os.getpid(),
                "endpoint": f"{actual_host}:{actual_port}",
                "protocol_version": PROTOCOL_VERSION,
                "started_unix_ns": time.time_ns(),
                "runtime_profile": self.runtime_profile,
                "model_shard_proof": self.model_shard_proof,
            }
            temporary = self.ready_path.with_suffix(self.ready_path.suffix + ".tmp")
            temporary.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.ready_path)
            self._trace(
                "worker_ready", endpoint=ready["endpoint"], runtime_profile=self.runtime_profile
            )
            threads: list[threading.Thread] = []
            while not self.shutdown_event.is_set():
                try:
                    connection, address = server.accept()
                except TimeoutError:
                    continue
                thread = threading.Thread(
                    target=self._handle,
                    args=(connection, address),
                    daemon=True,
                    name=f"microworker-{self.worker_id}",
                )
                thread.start()
                threads.append(thread)
                threads = [item for item in threads if item.is_alive()]
            for thread in threads:
                thread.join(timeout=2.0)
            self._trace("worker_stopped")
            return 0
        finally:
            server.close()


def _serve_from_config(path: Path) -> int:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        return MicroworkerServer(config).serve()
    except BaseException as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Swarm persistent microworker protocol")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve_from_config(args.config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAGIC",
    "NETWORK_PROFILES",
    "PROTOCOL_VERSION",
    "LinkProfile",
    "PersistentChannel",
    "aggregate_digest",
    "canonical_json_bytes",
    "combine_aggregates",
    "combine_argmax_aggregates",
    "combine_latency_histograms",
    "combine_operation_aggregates",
    "encode_message",
    "latency_histogram_observe",
    "latency_histogram_percentile_ms",
    "leaf_aggregate",
    "make_delegated_request",
    "make_leaf_request",
    "receive_message",
    "retry_attempt_timeout_s",
    "retryable_round_trip_error",
    "send_message",
    "shaped_round_trip",
    "validate_operation_envelope",
    "worker_contribution",
]
