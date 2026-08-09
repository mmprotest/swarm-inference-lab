"""Canonical expert role hosted by the normal persistent worker process."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from swarm_inference.exceptions import IntegrityError
from swarm_inference.execution.expert import (
    ExpertStore,
    validate_expert_content_hash,
)
from swarm_inference.execution.microshard import (
    MicroshardRange,
    validate_resident_microshard,
)
from swarm_inference.host import format_endpoint
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.expert import (
    SUPPORTED_EXPERT_PROTOCOL_VERSIONS,
    DelegatedMicroshardNode,
    DelegatedMicroshardOperation,
    ExpertExecutionMetadata,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    ExpertPeerHandshake,
    ExpertProtocolVersion,
    ResultIntegrity,
    SignedExpertRouteLease,
    expert_route_lease_hash,
    sign_expert_peer_handshake,
    verify_expert_peer_handshake,
    verify_expert_route_lease,
)
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import WorkerIdentity, public_key_fingerprint
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.security.tls import (
    TlsClientConfig,
    TlsServerConfig,
    require_tls_for_endpoint,
)
from swarm_inference.transport.expert import (
    ExpertPacket,
    ExpertTransportClient,
    decode_packet,
    decode_request,
    encode_packet,
    encode_response,
    frame_with_length,
    read_length_frame,
)


@dataclass(slots=True)
class ExpertServiceTelemetry:
    requests_completed: int = 0
    duplicate_requests: int = 0
    rejected_requests: int = 0
    cancelled_requests: int = 0
    deadline_failures: int = 0
    queue_rejections: int = 0
    remote_whole_expert_calls: int = 0
    remote_microshard_calls: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    compute_ns: int = 0
    whole_expert_compute_ns: int = 0
    microshard_compute_ns: int = 0
    queue_ns: int = 0
    delegated_parent_calls: int = 0
    delegated_child_calls: int = 0
    delegated_reductions: int = 0
    delegated_retries: int = 0
    delegated_failures: int = 0
    delegated_cancellations: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "requests_completed": self.requests_completed,
            "duplicate_requests": self.duplicate_requests,
            "rejected_requests": self.rejected_requests,
            "cancelled_requests": self.cancelled_requests,
            "deadline_failures": self.deadline_failures,
            "queue_rejections": self.queue_rejections,
            "remote_whole_expert_calls": self.remote_whole_expert_calls,
            "remote_microshard_calls": self.remote_microshard_calls,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
            "compute_ns": self.compute_ns,
            "whole_expert_compute_ns": self.whole_expert_compute_ns,
            "microshard_compute_ns": self.microshard_compute_ns,
            "queue_ns": self.queue_ns,
            "delegated_parent_calls": self.delegated_parent_calls,
            "delegated_child_calls": self.delegated_child_calls,
            "delegated_reductions": self.delegated_reductions,
            "delegated_retries": self.delegated_retries,
            "delegated_failures": self.delegated_failures,
            "delegated_cancellations": self.delegated_cancellations,
        }


@dataclass(frozen=True, slots=True)
class _CachedResponse:
    request_hash: str
    response: ExpertExecutionResponse
    output: np.ndarray


class ExpertWorkerRuntime:
    """Identity-bound, bounded execution service for expert worker roles."""

    def __init__(
        self,
        *,
        worker_id: str,
        identity: WorkerIdentity,
        model_id: str,
        model_revision: str,
        model_fingerprint: str,
        quantization_fingerprint: str,
        store: ExpertStore,
        roles: set[str] | None = None,
        owned_microshards: list[dict[str, Any]] | None = None,
        maximum_queue_depth: int = 64,
        maximum_concurrent_requests: int = 1,
        response_cache_size: int = 256,
        require_authenticated_routes: bool = False,
        trusted_coordinators: dict[str, str] | set[str] | None = None,
        peer_tls: TlsClientConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if maximum_queue_depth <= 0 or maximum_concurrent_requests <= 0:
            raise ValueError("expert queue and concurrency limits must be positive")
        if response_cache_size <= 0:
            raise ValueError("expert response cache size must be positive")
        self.worker_id = worker_id
        self.identity = identity
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_fingerprint = model_fingerprint
        self.quantization_fingerprint = quantization_fingerprint
        self.store = store
        self.roles = roles or {"whole-expert"}
        allowed_roles = {"whole-expert", "expert-microshard", "reducer"}
        if not self.roles or not self.roles <= allowed_roles:
            raise ValueError("expert worker has an unsupported role")
        self.owned_microshards = list(owned_microshards or [])
        self.maximum_queue_depth = maximum_queue_depth
        self._slots = asyncio.Semaphore(maximum_concurrent_requests)
        self._queue_lock = asyncio.Lock()
        self._request_lock_guard = asyncio.Lock()
        self._request_locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._queued = 0
        self._active_request_ids: set[str] = set()
        self._active_session_by_request: dict[str, str] = {}
        self._cancelled_request_ids: set[str] = set()
        self._cancelled_session_ids: set[str] = set()
        self._responses: OrderedDict[str, _CachedResponse] = OrderedDict()
        self._response_cache_size = response_cache_size
        self.require_authenticated_routes = require_authenticated_routes
        self._trusted_coordinators: dict[str, str] | set[str] = trusted_coordinators or set()
        if self.require_authenticated_routes and not self._trusted_coordinators:
            raise ValueError("authenticated expert routes require a trusted coordinator")
        self.route_lease: SignedExpertRouteLease | None = None
        self._last_route_generation: int | None = None
        self._route_nonce_cache = BoundedNonceCache(capacity=4096)
        self._peer_nonce_cache = BoundedNonceCache(capacity=4096)
        self.peer_tls = peer_tls
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self._active_delegated_children: dict[str, dict[str, DelegatedMicroshardNode]] = {}
        self._delegation_events: list[dict[str, Any]] = []
        self.telemetry = ExpertServiceTelemetry()

        microshard_keys: set[tuple[int, int]] = set()
        for item in self.owned_microshards:
            try:
                layer_id = int(item["layer_id"])
                expert_id = int(item["expert_id"])
                hidden_start = int(item["hidden_start"])
                hidden_end = int(item["hidden_end"])
                logical_width = int(item["logical_intermediate_dimension"])
                content_hash = str(item["content_hash"])
                group_value = item.get("quantization_group_size")
                group_size = int(group_value) if group_value is not None else None
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("owned microshard descriptor is incomplete") from exc
            ownership = MicroshardRange(
                worker_id=self.worker_id,
                layer_id=layer_id,
                expert_id=expert_id,
                hidden_start=hidden_start,
                hidden_end=hidden_end,
                logical_intermediate_dimension=logical_width,
                content_hash=content_hash,
                quantization_group_size=group_size,
            )
            if ownership.owns_full_expert:
                raise ValueError("a native microshard worker cannot retain the full expert")
            key = (layer_id, expert_id)
            if key in microshard_keys:
                raise ValueError("one worker may own only one physical slice of an expert")
            microshard_keys.add(key)
            if key not in self.store.owned:
                raise ValueError("owned microshard has no corresponding resident store entry")
            resident = self.store.loader(layer_id, expert_id)
            validate_expert_content_hash(resident, content_hash)
            validate_resident_microshard(resident, ownership)
            if resident.byte_size > self.store.residency_budget_bytes:
                raise MemoryError("resident microshard exceeds the worker residency budget")
            if (
                resident.scale_group_size is not None
                and resident.scale_group_size != ownership.quantization_group_size
            ):
                raise ValueError(
                    "resident microshard quantisation group does not match its descriptor"
                )
        if "expert-microshard" in self.roles and not microshard_keys:
            raise ValueError("microshard role requires physical microshard ownership")
        whole_keys = self.store.owned - microshard_keys
        if "whole-expert" in self.roles and not whole_keys:
            raise ValueError("whole-expert role requires physical whole-expert ownership")

    def install_route(
        self,
        lease: SignedExpertRouteLease,
        trusted_coordinators: dict[str, str] | set[str] | None = None,
    ) -> None:
        trusted = trusted_coordinators or self._trusted_coordinators
        if not trusted:
            raise IntegrityError("expert route trust is not configured")
        verify_expert_route_lease(
            lease,
            trusted,
            last_route_generation=self._last_route_generation,
            nonce_cache=self._route_nonce_cache,
        )
        participant = next(
            (item for item in lease.participants if item.worker_id == self.worker_id), None
        )
        if participant is None:
            raise IntegrityError("expert worker is absent from the route lease")
        if participant.worker_public_key != self.identity.public_key_b64:
            raise IntegrityError("expert route worker identity mismatch")
        if (
            participant.model_fingerprint != self.model_fingerprint
            or participant.quantization_fingerprint != self.quantization_fingerprint
            or lease.model_id != self.model_id
            or lease.model_revision != self.model_revision
        ):
            raise IntegrityError("expert route model identity mismatch")
        self.route_lease = lease
        self._last_route_generation = lease.route_generation

    def configure_route_trust(
        self,
        *,
        coordinator_identity: str,
        coordinator_public_key: str,
        expected_fingerprint: str | None = None,
    ) -> None:
        fingerprint = public_key_fingerprint(coordinator_public_key)
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise IntegrityError(
                "coordinator fingerprint does not match expert trust configuration"
            )
        if not isinstance(self._trusted_coordinators, dict):
            self._trusted_coordinators = {}
        existing = self._trusted_coordinators.get(coordinator_identity)
        if existing is not None and existing != coordinator_public_key:
            raise IntegrityError("coordinator identity is already pinned to another key")
        self._trusted_coordinators[coordinator_identity] = coordinator_public_key
        self.require_authenticated_routes = True

    def remove_route(self, *, topology_id: str, route_generation: int) -> bool:
        lease = self.route_lease
        if lease is None:
            return False
        if lease.topology_id != topology_id or lease.route_generation != route_generation:
            raise IntegrityError("expert route removal identity mismatch")
        self.route_lease = None
        return True

    def _validate_route(self, request: ExpertExecutionRequest) -> str | None:
        lease = self.route_lease
        if lease is None:
            if self.require_authenticated_routes:
                raise IntegrityError("expert request has no installed authenticated route")
            return None
        if lease.lease_expiry_unix_ns <= time.time_ns():
            raise IntegrityError("installed expert route lease has expired")
        if (
            request.topology_id != lease.topology_id
            or request.route_generation != lease.route_generation
            or request.model_id != lease.model_id
            or request.model_revision != lease.model_revision
            or request.quantization_fingerprint != lease.quantization_fingerprint
        ):
            raise IntegrityError("stale or mismatched expert route identity")
        return self.validate_peer_authentication(request.authentication)

    def validate_peer_authentication(self, authentication: dict[str, Any] | None) -> str | None:
        """Validate one stage/reducer peer handshake for a stateful operation."""

        lease = self.route_lease
        if lease is None:
            if self.require_authenticated_routes:
                raise IntegrityError("expert operation has no installed authenticated route")
            return None
        if authentication is None:
            raise IntegrityError("expert operation is missing peer authentication")
        handshake = ExpertPeerHandshake.model_validate(authentication)
        verify_expert_peer_handshake(
            handshake,
            lease,
            expected_worker_id=handshake.worker_id,
            expected_peer_worker_id=self.worker_id,
            nonce_cache=self._peer_nonce_cache,
        )
        peer = next(
            (item for item in lease.participants if item.worker_id == handshake.worker_id), None
        )
        if peer is None or not set(peer.roles).intersection({"contiguous-stage", "reducer"}):
            raise IntegrityError("expert request peer is not authorized to dispatch stage work")
        if handshake.route_lease_hash != expert_route_lease_hash(lease):
            raise IntegrityError("expert handshake is not bound to the installed route")
        return handshake.worker_id

    def _peer_authentication(self, peer_worker_id: str) -> dict[str, Any]:
        lease = self.route_lease
        if lease is None:
            raise IntegrityError("delegated worker dispatch requires an installed route")
        peer = next((item for item in lease.participants if item.worker_id == peer_worker_id), None)
        if peer is None or not set(peer.roles).intersection({"expert-microshard", "reducer"}):
            raise IntegrityError("delegated child is not authorized by the installed route")
        unsigned = ExpertPeerHandshake(
            protocol_versions=list(SUPPORTED_EXPERT_PROTOCOL_VERSIONS),
            selected_version=ExpertProtocolVersion.V1,
            worker_id=self.worker_id,
            public_key_fingerprint=self.identity.public_key_fingerprint,
            topology_id=lease.topology_id,
            route_generation=lease.route_generation,
            peer_worker_id=peer_worker_id,
            model_revision=lease.model_revision,
            quantization_fingerprint=lease.quantization_fingerprint,
            nonce=uuid4().hex,
            timestamp_unix_ns=time.time_ns(),
            route_lease_hash=expert_route_lease_hash(lease),
        )
        return sign_expert_peer_handshake(unsigned, self.identity).model_dump(mode="json")

    @staticmethod
    def _node_descriptor(node: DelegatedMicroshardNode) -> dict[str, Any]:
        return {
            "worker_id": node.worker_id,
            "layer_id": node.layer_id,
            "expert_id": node.expert_id,
            "hidden_start": node.hidden_start,
            "hidden_end": node.hidden_end,
            "logical_intermediate_dimension": node.logical_intermediate_dimension,
            "content_hash": node.content_hash,
        }

    def _validate_delegation(
        self,
        request: ExpertExecutionRequest,
        authenticated_parent: str | None,
    ) -> DelegatedMicroshardOperation | None:
        raw = request.metadata.get("delegation")
        if raw is None:
            return None
        operation = DelegatedMicroshardOperation.model_validate(raw)
        lease = self.route_lease
        if lease is None:
            raise IntegrityError("delegated microshards require an installed signed route")
        if authenticated_parent != operation.parent_worker_id:
            raise IntegrityError("delegated microshard parent does not match authentication")
        if (
            operation.operation_id not in request.request_id
            or operation.execution_generation != request.route_generation
            or operation.deadline_ns != request.deadline_ns
            or operation.route_lease_identity != expert_route_lease_hash(lease)
        ):
            raise IntegrityError("delegated operation identity is stale or mismatched")
        node = operation.node
        if (
            node.worker_id != self.worker_id
            or node.layer_id != request.layer_id
            or node.expert_id not in request.all_expert_ids
            or node.hidden_start != request.hidden_start
            or node.hidden_end != request.hidden_end
        ):
            raise IntegrityError("delegated root node does not match the receiving worker")

        participants = {item.worker_id: item for item in lease.participants}
        seen: set[str] = set()
        ordered_ranges: list[tuple[int, int, str]] = []

        def visit(current: DelegatedMicroshardNode) -> None:
            if current.worker_id in seen:
                raise IntegrityError("delegated subtree contains a duplicate worker")
            seen.add(current.worker_id)
            if len(current.children) > operation.branch_factor:
                raise IntegrityError("delegated subtree exceeds its bounded branch factor")
            if (
                current.layer_id != request.layer_id
                or current.expert_id not in request.all_expert_ids
                or current.logical_intermediate_dimension != node.logical_intermediate_dimension
            ):
                raise IntegrityError("delegated subtree work partition is inconsistent")
            participant = participants.get(current.worker_id)
            if participant is None:
                raise IntegrityError("delegated subtree contains an unleased worker")
            if participant.endpoint != current.endpoint:
                raise IntegrityError("delegated child endpoint differs from the signed route")
            roles = set(participant.roles)
            if "expert-microshard" not in roles or (current.children and "reducer" not in roles):
                raise IntegrityError("delegated worker lacks its required route role")
            expected = self._node_descriptor(current)
            matches = [
                item
                for item in participant.owned_microshards
                if all(
                    item.get(key) == value for key, value in expected.items() if key != "worker_id"
                )
            ]
            if len(matches) != 1:
                raise IntegrityError("delegated work is not owned by the signed participant")
            ordered_ranges.append((current.hidden_start, current.hidden_end, current.worker_id))
            for child in current.children:
                visit(child)

        visit(node)
        ordered_ranges.sort()
        cursor = ordered_ranges[0][0]
        for start, end, _worker_id in ordered_ranges:
            if start != cursor:
                raise IntegrityError("delegated subtree ranges overlap or contain a gap")
            cursor = end
        return operation

    def _verify_child_response(
        self,
        *,
        child: DelegatedMicroshardNode,
        request: ExpertExecutionRequest,
        response: ExpertExecutionResponse,
        result: np.ndarray,
    ) -> None:
        lease = self.route_lease
        if lease is None:
            raise IntegrityError("delegated child response has no installed route")
        participant = next(
            (item for item in lease.participants if item.worker_id == child.worker_id), None
        )
        if participant is None or response.worker_id != child.worker_id:
            raise IntegrityError("delegated child response identity mismatch")
        result_hash = "sha256:" + sha256_bytes(np.ascontiguousarray(result).tobytes())
        if response.integrity.result_hash != result_hash:
            raise IntegrityError("delegated child result hash mismatch")
        payload = canonical_json_bytes(
            {
                "worker_id": child.worker_id,
                "request_id": request.request_id,
                "session_id": request.session_id,
                "token_position": request.token_position,
                "route_generation": request.route_generation,
                "model_fingerprint": self.model_fingerprint,
                "result_hash": result_hash,
            }
        )
        verify_signature(
            participant.worker_public_key,
            payload,
            response.integrity.worker_signature,
        )

    async def _dispatch_delegated_child(
        self,
        *,
        parent_request: ExpertExecutionRequest,
        source: np.ndarray,
        operation: DelegatedMicroshardOperation,
        child: DelegatedMicroshardNode,
    ) -> tuple[DelegatedMicroshardNode, np.ndarray, ExpertExecutionMetadata, dict[str, Any], int]:
        child_operation = operation.model_copy(
            update={
                "parent_worker_id": self.worker_id,
                "parent_span_id": operation.node.ordering_key,
                "node": child,
            },
            deep=True,
        )
        metadata = dict(parent_request.metadata)
        metadata["delegation"] = child_operation.model_dump(mode="json")
        child_request = parent_request.model_copy(
            update={
                "request_id": f"{operation.operation_id}:worker:{child.worker_id}",
                "hidden_start": child.hidden_start,
                "hidden_end": child.hidden_end,
                "expert_hashes": {child.expert_id: child.content_hash},
                "metadata": metadata,
            },
            deep=True,
        )
        failures = 0
        last_error: BaseException | None = None
        for attempt in range(operation.retry_max_attempts):
            child_request = child_request.model_copy(
                update={"authentication": self._peer_authentication(child.worker_id)},
                deep=True,
            )
            client = ExpertTransportClient(
                child.endpoint,
                timeout_s=max(
                    0.001,
                    (parent_request.deadline_ns - time.time_ns()) / 1_000_000_000,
                ),
                tls=self.peer_tls,
                allow_plaintext_loopback=self.allow_plaintext_loopback,
            )
            try:
                response, result, transport = await asyncio.to_thread(
                    client.execute,
                    child_request,
                    source,
                )
                self._verify_child_response(
                    child=child,
                    request=child_request,
                    response=response,
                    result=result,
                )
                self.telemetry.delegated_child_calls += 1
                self.telemetry.delegated_retries += attempt
                self.telemetry.delegated_failures += failures
                return (
                    child,
                    np.ascontiguousarray(result, dtype=np.float32),
                    (response.execution_metadata),
                    transport,
                    attempt,
                )
            except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
                failures += 1
                last_error = exc
                if attempt + 1 >= operation.retry_max_attempts:
                    break
                if time.time_ns() >= parent_request.deadline_ns:
                    break
        self.telemetry.delegated_failures += failures
        assert last_error is not None
        raise last_error

    async def _execute_delegated(
        self,
        *,
        request: ExpertExecutionRequest,
        source: np.ndarray,
        down_accumulators: np.ndarray | None,
        operation: DelegatedMicroshardOperation,
    ) -> tuple[np.ndarray, dict[str, Any], int]:
        children = {item.worker_id: item for item in operation.node.children}
        self._active_delegated_children[request.session_id] = children
        child_tasks = [
            asyncio.create_task(
                self._dispatch_delegated_child(
                    parent_request=request,
                    source=source,
                    operation=operation,
                    child=child,
                )
            )
            for child in operation.node.children
        ]
        compute_started = time.perf_counter_ns()
        try:
            local_output, metrics = await asyncio.to_thread(
                self.store.execute,
                request,
                source,
                down_accumulators,
            )
            child_results = await asyncio.gather(*child_tasks)
        except BaseException:
            for task in child_tasks:
                task.cancel()
            await asyncio.gather(*child_tasks, return_exceptions=True)
            raise
        finally:
            current = self._active_delegated_children.get(request.session_id)
            if current is children:
                self._active_delegated_children.pop(request.session_id, None)
        compute_ns = time.perf_counter_ns() - compute_started
        contributions = [(operation.node.ordering_key, local_output)] + [
            (child.ordering_key, output)
            for child, output, _metadata, _transport, _attempt in child_results
        ]
        contributions.sort(key=lambda item: item[0])
        output = np.zeros_like(np.asarray(local_output), dtype=np.float32)
        for _key, contribution in contributions:
            if contribution.shape != output.shape:
                raise ValueError("delegated child result geometry mismatch")
            output += np.asarray(contribution, dtype=np.float32)

        child_metadata = [item[2] for item in child_results]
        child_transport = [item[3] for item in child_results]
        immediate_edges = [
            {
                "operation_id": operation.operation_id,
                "parent_worker_id": self.worker_id,
                "child_worker_id": child.worker_id,
                "parent_process_id": None,
                "child_endpoint": child.endpoint,
                "attempts": attempt + 1,
            }
            for child, _output, _metadata, _transport, attempt in child_results
        ]
        descendant_edges = [edge for item in child_metadata for edge in item.delegated_edges]
        worker_ids = [self.worker_id] + [
            worker_id for item in child_metadata for worker_id in item.delegated_worker_ids
        ]
        child_request_bytes = sum(int(item.get("request_bytes", 0)) for item in child_transport)
        child_response_bytes = sum(int(item.get("response_bytes", 0)) for item in child_transport)
        metrics.update(
            {
                "delegation_mode": "worker_tree",
                "delegated_worker_ids": worker_ids,
                "delegated_edges": immediate_edges + descendant_edges,
                "delegated_worker_count": len(worker_ids),
                "delegated_worker_messages": 2 * len(child_results)
                + sum(item.delegated_worker_messages for item in child_metadata),
                "delegated_request_bytes": child_request_bytes
                + sum(item.delegated_request_bytes for item in child_metadata),
                "delegated_response_bytes": child_response_bytes
                + sum(item.delegated_response_bytes for item in child_metadata),
                "delegated_depth": 1
                + max((item.delegated_depth for item in child_metadata), default=0),
                "delegated_reduction_depth": (
                    1 + max((item.delegated_reduction_depth for item in child_metadata), default=0)
                    if child_results
                    else 0
                ),
                "delegated_intermediate_reductions": (
                    (1 if child_results else 0)
                    + sum(item.delegated_intermediate_reductions for item in child_metadata)
                ),
                "delegated_retries": sum(item[4] for item in child_results)
                + sum(item.delegated_retries for item in child_metadata),
                "delegated_failures": sum(item[4] for item in child_results)
                + sum(item.delegated_failures for item in child_metadata),
            }
        )
        event = {
            "event": "delegated_microshard_reduced",
            "operation_id": operation.operation_id,
            "worker_id": self.worker_id,
            "process_id": __import__("os").getpid(),
            "parent_worker_id": operation.parent_worker_id,
            "child_worker_ids": [item.worker_id for item in operation.node.children],
            "subtree_worker_ids": worker_ids,
            "subtree_depth": metrics["delegated_depth"],
            "intermediate_reduction": bool(child_results),
            "retries": sum(item[4] for item in child_results),
            "result_hash": "sha256:" + sha256_bytes(output.tobytes()),
        }
        self._delegation_events.append(event)
        del self._delegation_events[:-256]
        self.telemetry.delegated_parent_calls += 1
        self.telemetry.delegated_reductions += int(bool(child_results))
        return output, metrics, compute_ns

    @staticmethod
    def _request_hash(
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None,
    ) -> str:
        digest = hashlib.sha256(
            canonical_json_bytes(
                request.model_dump(
                    mode="json",
                    exclude={"activations", "authentication"},
                )
            )
        )
        digest.update(np.ascontiguousarray(activation).tobytes())
        if down_accumulators is not None:
            digest.update(np.ascontiguousarray(down_accumulators).tobytes())
        return digest.hexdigest()

    def _validate_identity(self, request: ExpertExecutionRequest) -> None:
        if request.model_id != self.model_id or request.model_revision != self.model_revision:
            raise ValueError("expert request model identity does not match worker")
        if request.topology_id != "legacy" and not request.model_fingerprint:
            raise ValueError("product expert request requires an exact model fingerprint")
        if request.model_fingerprint and request.model_fingerprint != self.model_fingerprint:
            raise ValueError("expert request model fingerprint does not match worker")
        if request.quantization_fingerprint != self.quantization_fingerprint:
            raise ValueError("expert request quantization fingerprint does not match worker")
        if request.execution_mode.value == "whole_expert":
            if "whole-expert" not in self.roles:
                raise ValueError("worker does not advertise whole-expert execution")
        else:
            if "expert-microshard" not in self.roles:
                raise ValueError("worker does not advertise microshard execution")
            matches = [
                item
                for item in self.owned_microshards
                if int(item.get("layer_id", -1)) == request.layer_id
                and int(item.get("expert_id", -1)) in request.all_expert_ids
                and int(item.get("hidden_start", -1)) == request.hidden_start
                and int(item.get("hidden_end", -1)) == request.hidden_end
            ]
            if len(matches) != len(request.all_expert_ids):
                raise ValueError("requested microshard is not physically owned by this worker")

    async def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        bytes_received: int,
        decode_ns: int,
        accepted_ns: int | None = None,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[ExpertExecutionResponse, np.ndarray]:
        """Serialize identical request IDs so concurrent retries cannot execute twice."""

        async with self._request_lock_guard:
            existing = self._request_locks.get(request.request_id)
            request_lock, users = existing if existing is not None else (asyncio.Lock(), 0)
            self._request_locks[request.request_id] = (request_lock, users + 1)
        try:
            async with request_lock:
                return await self._execute_once(
                    request,
                    activation,
                    bytes_received=bytes_received,
                    decode_ns=decode_ns,
                    accepted_ns=accepted_ns,
                    down_accumulators=down_accumulators,
                )
        finally:
            async with self._request_lock_guard:
                current = self._request_locks.get(request.request_id)
                if current is not None and current[0] is request_lock:
                    if current[1] == 1:
                        self._request_locks.pop(request.request_id, None)
                    else:
                        self._request_locks[request.request_id] = (
                            request_lock,
                            current[1] - 1,
                        )

    async def _execute_once(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        bytes_received: int,
        decode_ns: int,
        accepted_ns: int | None = None,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[ExpertExecutionResponse, np.ndarray]:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        if (request.down_accumulators is None) != (down_accumulators is None):
            raise ValueError("microshard accumulator metadata and payload do not agree")
        if (
            request.request_id in self._cancelled_request_ids
            or request.session_id in self._cancelled_session_ids
        ):
            self.telemetry.cancelled_requests += 1
            raise asyncio.CancelledError("expert request or session was cancelled")
        if time.time_ns() >= request.deadline_ns:
            self.telemetry.deadline_failures += 1
            raise TimeoutError("expert request deadline elapsed before queue admission")
        self._validate_identity(request)
        authenticated_parent = self._validate_route(request)
        delegated_operation = self._validate_delegation(request, authenticated_parent)
        request_hash = self._request_hash(request, source, down_accumulators)
        cached = self._responses.get(request.request_id)
        if cached is not None:
            if cached.request_hash != request_hash:
                raise IntegrityError("duplicate expert request ID has different content")
            self._responses.move_to_end(request.request_id)
            self.telemetry.duplicate_requests += 1
            return cached.response, cached.output.copy()
        async with self._queue_lock:
            if self._queued >= self.maximum_queue_depth:
                self.telemetry.queue_rejections += 1
                raise OverflowError("expert worker queue is full")
            self._queued += 1
        queued_at = time.perf_counter_ns()
        try:
            async with self._slots:
                queue_ns = time.perf_counter_ns() - queued_at
                if accepted_ns is not None:
                    queue_ns = max(queue_ns, time.perf_counter_ns() - accepted_ns)
                if time.time_ns() >= request.deadline_ns:
                    self.telemetry.deadline_failures += 1
                    raise TimeoutError("expert request deadline elapsed in the worker queue")
                if (
                    request.request_id in self._cancelled_request_ids
                    or request.session_id in self._cancelled_session_ids
                ):
                    self.telemetry.cancelled_requests += 1
                    raise asyncio.CancelledError("expert request or session was cancelled")
                self._active_request_ids.add(request.request_id)
                self._active_session_by_request[request.request_id] = request.session_id
                if delegated_operation is not None:
                    output, metrics, compute_ns = await self._execute_delegated(
                        request=request,
                        source=source,
                        down_accumulators=down_accumulators,
                        operation=delegated_operation,
                    )
                else:
                    compute_started = time.perf_counter_ns()
                    output, metrics = await asyncio.to_thread(
                        self.store.execute,
                        request,
                        source,
                        down_accumulators,
                    )
                    compute_ns = time.perf_counter_ns() - compute_started
                if time.time_ns() >= request.deadline_ns:
                    self.telemetry.deadline_failures += 1
                    raise TimeoutError("expert request deadline elapsed during execution")
                if (
                    request.request_id in self._cancelled_request_ids
                    or request.session_id in self._cancelled_session_ids
                ):
                    self.telemetry.cancelled_requests += 1
                    raise asyncio.CancelledError("expert request was cancelled during execution")
        finally:
            self._active_request_ids.discard(request.request_id)
            self._active_session_by_request.pop(request.request_id, None)
            async with self._queue_lock:
                self._queued -= 1
        result_hash = "sha256:" + sha256_bytes(np.ascontiguousarray(output).tobytes())
        signature_payload = canonical_json_bytes(
            {
                "worker_id": self.worker_id,
                "request_id": request.request_id,
                "session_id": request.session_id,
                "token_position": request.token_position,
                "route_generation": request.route_generation,
                "model_fingerprint": self.model_fingerprint,
                "result_hash": result_hash,
            }
        )
        response = ExpertExecutionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            token_position=request.token_position,
            sequence_id=request.sequence_id,
            route_generation=request.route_generation,
            worker_id=self.worker_id,
            model_revision=self.model_revision,
            quantization_fingerprint=self.quantization_fingerprint,
            layer_id=request.layer_id,
            result={"codec": request.compression.value},
            execution_metadata=ExpertExecutionMetadata(
                **metrics,
                bytes_received=bytes_received,
                bytes_sent=int(output.nbytes),
                queue_ns=queue_ns,
                transfer_ns=0,
                serialisation_ns=decode_ns,
                backend="canonical-numpy",
                device="cpu",
            ),
            integrity=ResultIntegrity(
                result_hash=result_hash,
                model_fingerprint=self.model_fingerprint,
                worker_signature=self.identity.sign(signature_payload),
                expert_hashes=dict(request.expert_hashes),
            ),
        )
        self.telemetry.requests_completed += 1
        self.telemetry.bytes_received += bytes_received
        self.telemetry.bytes_sent += int(output.nbytes)
        self.telemetry.compute_ns += compute_ns
        self.telemetry.queue_ns += queue_ns
        if request.execution_mode.value == "whole_expert":
            self.telemetry.remote_whole_expert_calls += 1
            self.telemetry.whole_expert_compute_ns += compute_ns
        else:
            self.telemetry.remote_microshard_calls += 1
            self.telemetry.microshard_compute_ns += compute_ns
        cached_response = _CachedResponse(request_hash, response, output.copy())
        self._responses[request.request_id] = cached_response
        while len(self._responses) > self._response_cache_size:
            self._responses.popitem(last=False)
        return response, output

    def cancel_request(self, request_id: str) -> bool:
        active = request_id in self._active_request_ids
        self._cancelled_request_ids.add(request_id)
        return active

    def cancel_session(self, session_id: str) -> int:
        self._cancelled_session_ids.add(session_id)
        matching = {
            request_id
            for request_id, active_session_id in self._active_session_by_request.items()
            if active_session_id == session_id
        }
        self._cancelled_request_ids.update(matching)
        return len(matching)

    async def propagate_cancel_session(self, session_id: str) -> dict[str, Any]:
        children = list(self._active_delegated_children.get(session_id, {}).values())

        async def cancel_child(child: DelegatedMicroshardNode) -> tuple[str, dict[str, Any]]:
            client = ExpertTransportClient(
                child.endpoint,
                timeout_s=5.0,
                tls=self.peer_tls,
                allow_plaintext_loopback=self.allow_plaintext_loopback,
            )
            response = await asyncio.to_thread(
                client.control,
                "cancel_session",
                session_id=session_id,
                authentication=self._peer_authentication(child.worker_id),
            )
            return child.worker_id, response

        results = await asyncio.gather(
            *(cancel_child(child) for child in children),
            return_exceptions=True,
        )
        successful = [item for item in results if not isinstance(item, BaseException)]
        failures = [item for item in results if isinstance(item, BaseException)]
        descendants = sum(int(item[1].get("delegated_workers_contacted", 0)) for item in successful)
        contacted = len(successful) + descendants
        self.telemetry.delegated_cancellations += contacted
        event = {
            "event": "delegated_session_cancelled",
            "worker_id": self.worker_id,
            "process_id": __import__("os").getpid(),
            "session_id": session_id,
            "child_worker_ids": [item.worker_id for item in children],
            "delegated_workers_contacted": contacted,
            "failures": [f"{type(item).__name__}: {item}" for item in failures],
        }
        self._delegation_events.append(event)
        del self._delegation_events[:-256]
        return event

    def status(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "roles": sorted(self.roles),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "quantization_fingerprint": self.quantization_fingerprint,
            "owned_experts": [list(item) for item in sorted(self.store.owned)],
            "owned_microshards": self.owned_microshards,
            "queue_depth": self._queued,
            "maximum_queue_depth": self.maximum_queue_depth,
            "route_generation": (
                self.route_lease.route_generation if self.route_lease is not None else None
            ),
            "delegation_events": list(self._delegation_events),
            "active_delegated_sessions": sorted(self._active_delegated_children),
            **self.store.status(),
            **self.telemetry.to_dict(),
        }


class ExpertWorkerServer:
    """Bounded SWARMEX1 server colocated with the persistent worker."""

    def __init__(
        self,
        runtime: ExpertWorkerRuntime,
        *,
        host: str,
        port: int,
        tls: TlsServerConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.tls = tls
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self.server: asyncio.Server | None = None

    @property
    def endpoint(self) -> tuple[str, int]:
        if self.server is None or not self.server.sockets:
            raise RuntimeError("expert worker server is not started")
        address = self.server.sockets[0].getsockname()
        return str(address[0]), int(address[1])

    async def start(self) -> tuple[str, int]:
        require_tls_for_endpoint(
            format_endpoint(self.host, self.port),
            tls_configured=self.tls is not None,
            allow_plaintext_loopback=self.allow_plaintext_loopback,
            transport_name="expert data plane",
        )
        self.server = await asyncio.start_server(
            self._connection,
            self.host,
            self.port,
            ssl=self.tls.ssl_context() if self.tls is not None else None,
        )
        return self.endpoint

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted_ns = time.perf_counter_ns()
        try:
            if self.tls is not None:
                tls_object = writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                self.tls.validate_peer_der(peer_der)
            payload = await read_length_frame(reader)
            packet = decode_packet(payload)
            if packet.kind == "request":
                request, activation, down_accumulators, decode_ns = decode_request(
                    payload,
                    include_down_accumulators=True,
                )
                response, output = await self.runtime.execute(
                    request,
                    activation,
                    bytes_received=len(payload),
                    decode_ns=decode_ns,
                    accepted_ns=accepted_ns,
                    down_accumulators=down_accumulators,
                )
                encoded, encode_ns = encode_response(response, output)
                response.execution_metadata.serialisation_ns += encode_ns
                encoded, _ = encode_response(response, output)
                writer.write(frame_with_length(encoded))
                await writer.drain()
            elif packet.kind == "control":
                await self._control(packet.semantic, writer)
            else:
                raise ValueError("expert worker accepts request or control frames only")
        except asyncio.CancelledError as error:
            await self._write_error(writer, f"CancelledError: {error}")
        except Exception as error:
            self.runtime.telemetry.rejected_requests += 1
            await self._write_error(writer, f"{type(error).__name__}: {error}")
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    @staticmethod
    async def _write_error(writer: asyncio.StreamWriter, detail: str) -> None:
        failure = encode_packet(
            ExpertPacket(kind="control", semantic={"ok": False, "error": detail}, blobs=())
        )
        with suppress(ConnectionError):
            writer.write(frame_with_length(failure))
            await writer.drain()

    async def _control(self, payload: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        command = str(payload.get("command"))
        if command in {"manifest", "status", "heartbeat"}:
            result: dict[str, Any] = {"ok": True, "status": self.runtime.status()}
        elif command == "cache_drop":
            self.runtime.validate_peer_authentication(payload.get("authentication"))
            self.runtime.store.drop_cache()
            result = {"ok": True}
        elif command == "cancel_request":
            self.runtime.validate_peer_authentication(payload.get("authentication"))
            result = {
                "ok": True,
                "was_active": self.runtime.cancel_request(str(payload["request_id"])),
            }
        elif command == "cancel_session":
            self.runtime.validate_peer_authentication(payload.get("authentication"))
            locally_cancelled = self.runtime.cancel_session(str(payload["session_id"]))
            propagated = await self.runtime.propagate_cancel_session(str(payload["session_id"]))
            result = {
                "ok": True,
                "cancelled": locally_cancelled,
                "delegated_workers_contacted": propagated["delegated_workers_contacted"],
                "delegated_failures": propagated["failures"],
            }
        elif command == "install_route":
            lease = SignedExpertRouteLease.model_validate(payload["route_lease"])
            self.runtime.install_route(lease)
            result = {
                "ok": True,
                "topology_id": lease.topology_id,
                "route_generation": lease.route_generation,
            }
        elif command == "remove_route":
            result = {
                "ok": True,
                "removed": self.runtime.remove_route(
                    topology_id=str(payload["topology_id"]),
                    route_generation=int(payload["route_generation"]),
                ),
            }
        else:
            result = {"ok": False, "error": f"unsupported control command {command!r}"}
        encoded = encode_packet(ExpertPacket(kind="control", semantic=result, blobs=()))
        writer.write(frame_with_length(encoded))
        await writer.drain()


__all__ = ["ExpertServiceTelemetry", "ExpertWorkerRuntime", "ExpertWorkerServer"]
