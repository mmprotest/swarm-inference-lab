"""Stage-internal MoE execution contract and canonical backend implementations."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from swarm_inference.execution.microshard import (
    MicroshardRange,
    physical_microshard_ownership,
)
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.expert import (
    SUPPORTED_EXPERT_PROTOCOL_VERSIONS,
    DeterminismMode,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ExpertPeerHandshake,
    ExpertProtocolVersion,
    ExpertResponseMode,
    ReductionMode,
    SignedExpertRouteLease,
    TransportCodec,
    expert_route_lease_hash,
    sign_expert_peer_handshake,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature


@dataclass(frozen=True, slots=True)
class MoeBackendCapabilities:
    backend: str
    whole_expert: bool
    native_microshard: bool
    local: bool
    exact: bool = True
    supported_codecs: tuple[str, ...] = (TransportCodec.RAW_FP32.value,)
    supported_reduction_modes: tuple[str, ...] = (ReductionMode.FIXED_ORDER_FP32.value,)


@dataclass(frozen=True, slots=True)
class MoeExecutionEvent:
    event: str
    backend: str
    session_id: str
    request_id: str
    token_position: int
    layer_id: int
    expert_id: int | None = None
    worker_ids: tuple[str, ...] = ()
    request_bytes: int = 0
    response_bytes: int = 0
    elapsed_ns: int = 0
    result_hash: str = ""
    fallback_reason: str | None = None
    total_messages: int = 0
    critical_path_messages: int = 0
    serial_waits: int = 0
    parallel_waits: int = 0
    fanout_depth: int = 0
    reduction_depth: int = 0
    critical_path_sync_rounds: int = 0
    scheduler_dispatch_ns: int = 0
    reduction_ns: int = 0
    communication_ns: int = 0
    root_dispatches: int = 0
    coordinator_waits: int = 0
    coordinator_sync_rounds: int = 0
    worker_sync_rounds: int = 0
    fanout_nodes: int = 0
    topology_construction_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "backend": self.backend,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "token_position": self.token_position,
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "worker_ids": list(self.worker_ids),
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "elapsed_ns": self.elapsed_ns,
            "result_hash": self.result_hash,
            "fallback_reason": self.fallback_reason,
            "total_messages": self.total_messages,
            "critical_path_messages": self.critical_path_messages,
            "serial_waits": self.serial_waits,
            "parallel_waits": self.parallel_waits,
            "fanout_depth": self.fanout_depth,
            "reduction_depth": self.reduction_depth,
            "critical_path_sync_rounds": self.critical_path_sync_rounds,
            "scheduler_dispatch_ns": self.scheduler_dispatch_ns,
            "reduction_ns": self.reduction_ns,
            "communication_ns": self.communication_ns,
            "root_dispatches": self.root_dispatches,
            "coordinator_waits": self.coordinator_waits,
            "coordinator_sync_rounds": self.coordinator_sync_rounds,
            "worker_sync_rounds": self.worker_sync_rounds,
            "fanout_nodes": self.fanout_nodes,
            "topology_construction_ns": self.topology_construction_ns,
        }


@dataclass(frozen=True, slots=True)
class MoeExecutionResult:
    output: torch.Tensor
    events: tuple[MoeExecutionEvent, ...] = ()
    metrics: dict[str, int | float | str] = field(default_factory=dict)


@runtime_checkable
class MoeExecutionBackend(Protocol):
    def capabilities(self) -> MoeBackendCapabilities: ...

    def open_session(self, session_id: str) -> None: ...

    def execute_layer(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        deadline_ns: int,
    ) -> MoeExecutionResult: ...

    def close_session(self, session_id: str) -> None: ...

    def cancel_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...


class ExpertClient(Protocol):
    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        down_accumulators: np.ndarray | None = None,
    ) -> tuple[Any, np.ndarray, dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class WholeExpertTarget:
    worker_id: str
    client: ExpertClient
    expert_hash: str = ""


@dataclass(frozen=True, slots=True)
class MicroshardTarget:
    ownership: MicroshardRange
    client: ExpertClient


@dataclass(frozen=True, slots=True)
class _FanoutNode:
    node_id: str
    target: MicroshardTarget | None = None
    children: tuple[_FanoutNode, ...] = ()


@dataclass(frozen=True, slots=True)
class _FanoutTopology:
    root_children: tuple[_FanoutNode, ...]
    depth: int
    node_count: int


@dataclass(slots=True)
class _FanoutCollector:
    expected_partials: int
    condition: Condition = field(default_factory=Condition)
    partials: list[tuple[str, np.ndarray]] = field(default_factory=list)
    request_bytes: int = 0
    response_bytes: int = 0
    maximum_request_ns: int = 0
    scheduler_dispatch_ns: int = 0
    error: BaseException | None = None

    def cancelled(self) -> bool:
        with self.condition:
            return self.error is not None

    def record_dispatch(self, elapsed_ns: int) -> None:
        with self.condition:
            self.scheduler_dispatch_ns += elapsed_ns

    def record_partial(self, result: tuple[str, np.ndarray, int, int, int]) -> None:
        owner_key, partial, sent, received, request_elapsed_ns = result
        with self.condition:
            if self.error is not None:
                return
            self.partials.append((owner_key, partial))
            self.request_bytes += sent
            self.response_bytes += received
            self.maximum_request_ns = max(self.maximum_request_ns, request_elapsed_ns)
            self.condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self.condition:
            if self.error is None:
                self.error = exc
            self.condition.notify_all()

    def wait(self, deadline_ns: int) -> list[tuple[str, np.ndarray]]:
        with self.condition:
            while len(self.partials) < self.expected_partials and self.error is None:
                remaining_ns = deadline_ns - time.time_ns()
                if remaining_ns <= 0:
                    self.error = TimeoutError("microshard fan-out deadline elapsed")
                    break
                self.condition.wait(timeout=remaining_ns / 1_000_000_000)
            if self.error is not None:
                raise self.error
            return list(self.partials)


class _ExpertRouteAuthentication:
    """Bind direct expert calls to the coordinator-authorized product route."""

    def __init__(
        self,
        *,
        topology_id: str,
        route_generation: int,
        model_revision: str,
        model_fingerprint: str,
        quantization_fingerprint: str,
    ) -> None:
        self.topology_id = topology_id
        self.route_generation = route_generation
        self.model_revision = model_revision
        self.model_fingerprint = model_fingerprint
        self.quantization_fingerprint = quantization_fingerprint
        self.lease: SignedExpertRouteLease | None = None
        self.identity: WorkerIdentity | None = None
        self.worker_id = ""

    def configure(
        self,
        lease: SignedExpertRouteLease,
        *,
        identity: WorkerIdentity,
        worker_id: str,
    ) -> None:
        if (
            lease.topology_id != self.topology_id
            or lease.route_generation < self.route_generation
            or lease.model_revision != self.model_revision
            or lease.model_fingerprint != self.model_fingerprint
            or lease.quantization_fingerprint != self.quantization_fingerprint
        ):
            raise ValueError("expert lease does not match the stage backend identity")
        participant = next(
            (item for item in lease.participants if item.worker_id == worker_id), None
        )
        if participant is None or "contiguous-stage" not in participant.roles:
            raise ValueError("stage worker is not authorized by the expert lease")
        if (
            participant.worker_public_key != identity.public_key_b64
            or participant.worker_public_key_fingerprint != identity.public_key_fingerprint
        ):
            raise ValueError("stage identity does not match the expert lease")
        self.lease = lease
        self.identity = identity
        self.worker_id = worker_id
        self.route_generation = lease.route_generation

    def request_authentication(self, peer_worker_id: str) -> dict[str, Any] | None:
        lease = self.lease
        identity = self.identity
        if lease is None or identity is None:
            return None
        peer = next((item for item in lease.participants if item.worker_id == peer_worker_id), None)
        if peer is None or not set(peer.roles).intersection(
            {"whole-expert", "expert-microshard", "reducer"}
        ):
            raise ValueError("expert peer is not authorized by the installed route")
        handshake = ExpertPeerHandshake(
            protocol_versions=list(SUPPORTED_EXPERT_PROTOCOL_VERSIONS),
            selected_version=ExpertProtocolVersion.V1,
            worker_id=self.worker_id,
            public_key_fingerprint=identity.public_key_fingerprint,
            topology_id=lease.topology_id,
            route_generation=lease.route_generation,
            peer_worker_id=peer_worker_id,
            model_revision=lease.model_revision,
            quantization_fingerprint=lease.quantization_fingerprint,
            nonce=uuid4().hex,
            timestamp_unix_ns=time.time_ns(),
            route_lease_hash=expert_route_lease_hash(lease),
        )
        return sign_expert_peer_handshake(handshake, identity).model_dump(mode="json")

    def verify_response(
        self,
        *,
        peer_worker_id: str,
        request: ExpertExecutionRequest,
        response: Any,
        result: np.ndarray,
    ) -> None:
        lease = self.lease
        if lease is None:
            return
        participant = next(
            (item for item in lease.participants if item.worker_id == peer_worker_id), None
        )
        if participant is None or str(getattr(response, "worker_id", "")) != peer_worker_id:
            raise ValueError("expert response worker identity mismatch")
        integrity = getattr(response, "integrity", None)
        if integrity is None:
            raise ValueError("authenticated expert response has no integrity metadata")
        result_hash = "sha256:" + sha256_bytes(np.ascontiguousarray(result).tobytes())
        if str(getattr(integrity, "result_hash", "")) != result_hash:
            raise ValueError("authenticated expert result hash mismatch")
        payload = canonical_json_bytes(
            {
                "worker_id": peer_worker_id,
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
            str(getattr(integrity, "worker_signature", "")),
        )


def _result_hash(value: torch.Tensor) -> str:
    source = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return "sha256:" + hashlib.sha256(source.tobytes()).hexdigest()


class _SessionBackend:
    def __init__(self) -> None:
        self._sessions: set[str] = set()
        self._cancelled: set[str] = set()
        self._closed = False
        self._events: list[MoeExecutionEvent] = []

    def open_session(self, session_id: str) -> None:
        if self._closed:
            raise RuntimeError("MoE backend is closed")
        if not session_id:
            raise ValueError("MoE session ID cannot be empty")
        if session_id in self._sessions:
            raise ValueError("MoE session is already open")
        self._sessions.add(session_id)
        self._cancelled.discard(session_id)

    def close_session(self, session_id: str) -> None:
        self._require_session(session_id)
        self._sessions.remove(session_id)
        self._cancelled.discard(session_id)

    def cancel_session(self, session_id: str) -> None:
        self._require_session(session_id)
        self._cancelled.add(session_id)
        self._sessions.remove(session_id)

    def close(self) -> None:
        self._sessions.clear()
        self._cancelled.clear()
        self._closed = True

    def _require_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            if session_id in self._cancelled:
                raise RuntimeError("MoE session was cancelled")
            raise ValueError("MoE session is not open")

    def _start_call(self, session_id: str, deadline_ns: int) -> None:
        self._require_session(session_id)
        if time.time_ns() >= deadline_ns:
            raise TimeoutError("MoE execution deadline elapsed")

    def status(self) -> dict[str, Any]:
        return {
            "active_sessions": len(self._sessions),
            "events": [event.to_dict() for event in self._events[-128:]],
        }


def _geometry(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    if hidden_states.ndim not in {2, 3}:
        raise ValueError("MoE hidden states must be [rows, hidden] or [batch, sequence, hidden]")
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    if selected_experts.ndim != 2 or routing_weights.shape != selected_experts.shape:
        raise ValueError("selected experts and routing weights must share [rows, top_k] shape")
    if selected_experts.shape[0] != flat.shape[0]:
        raise ValueError("routing row count does not match hidden states")
    if router_logits.shape[0] != flat.shape[0]:
        raise ValueError("router logits row count does not match hidden states")
    return flat, int(flat.shape[0]), int(flat.shape[1])


def _selected_rows(
    selected_experts: torch.Tensor, expert_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    token_indices, ranks = torch.where(selected_experts == expert_id)
    return token_indices, ranks


class LocalMoeBackend(_SessionBackend):
    """Execute selected experts already resident inside the owning stage."""

    def __init__(self, experts: dict[tuple[int, int], nn.Module]) -> None:
        super().__init__()
        self.experts = dict(experts)

    def capabilities(self) -> MoeBackendCapabilities:
        return MoeBackendCapabilities(
            backend="local", whole_expert=True, native_microshard=False, local=True
        )

    def execute_expert_rows(
        self,
        *,
        layer_id: int,
        expert_id: int,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[str, ...], dict[str, Any]]:
        module = self.experts.get((layer_id, expert_id))
        if module is None:
            raise KeyError(f"local stage does not own layer {layer_id} expert {expert_id}")
        return module(activation), (), {}

    @torch.inference_mode()
    def execute_layer(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        deadline_ns: int,
    ) -> MoeExecutionResult:
        self._start_call(session_id, deadline_ns)
        flat, _, hidden = _geometry(hidden_states, router_logits, selected_experts, routing_weights)
        final = torch.zeros_like(flat)
        events: list[MoeExecutionEvent] = []
        expert_count = int(router_logits.shape[-1])
        for expert_id in range(expert_count):
            token_indices, ranks = _selected_rows(selected_experts, expert_id)
            if token_indices.numel() == 0:
                continue
            started = time.perf_counter_ns()
            output, workers, _ = self.execute_expert_rows(
                layer_id=layer_id,
                expert_id=expert_id,
                activation=flat[None, token_indices].reshape(-1, hidden),
            )
            weighted = output * routing_weights[token_indices, ranks, None]
            final.index_add_(0, token_indices, weighted.to(flat.dtype))
            events.append(
                MoeExecutionEvent(
                    event="local_expert_result_consumed",
                    backend="local",
                    session_id=session_id,
                    request_id=request_id,
                    token_position=token_position,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    worker_ids=workers,
                    elapsed_ns=time.perf_counter_ns() - started,
                    result_hash=_result_hash(output),
                )
            )
        self._events.extend(events)
        return MoeExecutionResult(
            output=final.reshape_as(hidden_states),
            events=tuple(events),
            metrics={"local_expert_calls": len(events)},
        )


class WholeExpertRemoteBackend(_SessionBackend):
    """Dispatch selected activations to workers owning complete experts."""

    def __init__(
        self,
        *,
        targets: dict[tuple[int, int], WholeExpertTarget],
        model_id: str,
        model_revision: str,
        model_fingerprint: str,
        quantization_fingerprint: str,
        topology_id: str,
        route_generation: int,
    ) -> None:
        super().__init__()
        self.targets = dict(targets)
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_fingerprint = model_fingerprint
        self.quantization_fingerprint = quantization_fingerprint
        self.topology_id = topology_id
        self.route_generation = route_generation
        self._route_auth = _ExpertRouteAuthentication(
            topology_id=topology_id,
            route_generation=route_generation,
            model_revision=model_revision,
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
        )

    def configure_route(
        self,
        lease: SignedExpertRouteLease,
        *,
        identity: WorkerIdentity,
        worker_id: str,
    ) -> None:
        self._route_auth.configure(lease, identity=identity, worker_id=worker_id)
        self.route_generation = lease.route_generation

    def capabilities(self) -> MoeBackendCapabilities:
        return MoeBackendCapabilities(
            backend="whole-expert-remote",
            whole_expert=True,
            native_microshard=False,
            local=False,
        )

    def cancel_session(self, session_id: str) -> None:
        clients = {item.worker_id: item.client for item in self.targets.values()}
        for worker_id, client in clients.items():
            control = getattr(client, "control", None)
            if control is not None:
                control(
                    "cancel_session",
                    session_id=session_id,
                    authentication=self._route_auth.request_authentication(worker_id),
                )
        super().cancel_session(session_id)

    def execute_expert_rows(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        expert_id: int,
        activation: torch.Tensor,
        deadline_ns: int,
    ) -> tuple[torch.Tensor, MoeExecutionEvent]:
        target = self.targets.get((layer_id, expert_id))
        if target is None:
            raise KeyError(f"no remote owner for layer {layer_id} expert {expert_id}")
        source = activation.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        subrequest_id = f"{request_id}:token-{token_position}:layer-{layer_id}:expert-{expert_id}"
        request = ExpertExecutionRequest(
            request_id=subrequest_id,
            session_id=session_id,
            token_position=token_position,
            sequence_id=token_position,
            route_generation=self.route_generation,
            topology_id=self.topology_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_fingerprint=self.model_fingerprint,
            quantization_fingerprint=self.quantization_fingerprint,
            layer_id=layer_id,
            batch_rows=int(source.shape[0]),
            latent_dimension=int(source.shape[1]),
            expert_ids=[expert_id],
            expert_hashes=({expert_id: target.expert_hash} if target.expert_hash else {}),
            routing_weights=[1.0],
            top_k=1,
            response_mode=ExpertResponseMode.PER_WORKER_FAST,
            activations={},
            deadline_ns=deadline_ns,
            execution_mode=ExpertExecutionMode.WHOLE_EXPERT,
            determinism_mode=DeterminismMode.EXACT,
            compression=TransportCodec.RAW_FP32,
            reduction_mode=ReductionMode.FIXED_ORDER_FP32,
            metadata={"exact_contribution_representation": "unweighted_expert_output"},
            authentication=self._route_auth.request_authentication(target.worker_id),
        )
        started = time.perf_counter_ns()
        response, result, transport = target.client.execute(request, source)
        self._route_auth.verify_response(
            peer_worker_id=target.worker_id,
            request=request,
            response=response,
            result=result,
        )
        if getattr(response, "status", "ok") != "ok":
            raise RuntimeError(str(getattr(response, "error", "remote expert failed")))
        integrity = getattr(response, "integrity", None)
        if integrity is None:
            raise ValueError("remote expert response is missing integrity metadata")
        remote_fingerprint = str(getattr(integrity, "model_fingerprint", ""))
        if self.model_fingerprint and remote_fingerprint != self.model_fingerprint:
            raise ValueError("remote expert model fingerprint mismatch")
        hashes = getattr(integrity, "expert_hashes", {})
        if target.expert_hash and hashes.get(expert_id) != target.expert_hash:
            raise ValueError("remote expert content hash mismatch")
        output = torch.from_numpy(np.ascontiguousarray(result, dtype=np.float32)).to(
            device=activation.device, dtype=activation.dtype
        )
        if output.shape != activation.shape:
            raise ValueError("remote whole-expert result shape mismatch")
        event = MoeExecutionEvent(
            event="remote_whole_expert_result_consumed",
            backend="whole-expert-remote",
            session_id=session_id,
            request_id=subrequest_id,
            token_position=token_position,
            layer_id=layer_id,
            expert_id=expert_id,
            worker_ids=(target.worker_id,),
            request_bytes=int(transport.get("request_bytes", 0)),
            response_bytes=int(transport.get("response_bytes", 0)),
            elapsed_ns=time.perf_counter_ns() - started,
            result_hash=_result_hash(output),
            total_messages=2,
            critical_path_messages=2,
            serial_waits=1,
            parallel_waits=1,
            fanout_depth=1,
            critical_path_sync_rounds=1,
            root_dispatches=1,
            coordinator_waits=0,
            coordinator_sync_rounds=0,
            worker_sync_rounds=1,
            fanout_nodes=1,
            communication_ns=int(
                transport.get("request_elapsed_ns", time.perf_counter_ns() - started)
            ),
        )
        return output, event

    @torch.inference_mode()
    def execute_layer(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        deadline_ns: int,
    ) -> MoeExecutionResult:
        self._start_call(session_id, deadline_ns)
        flat, _, hidden = _geometry(hidden_states, router_logits, selected_experts, routing_weights)
        final = torch.zeros_like(flat)
        events: list[MoeExecutionEvent] = []
        for expert_id in range(int(router_logits.shape[-1])):
            token_indices, ranks = _selected_rows(selected_experts, expert_id)
            if token_indices.numel() == 0:
                continue
            output, event = self.execute_expert_rows(
                session_id=session_id,
                request_id=request_id,
                token_position=token_position,
                layer_id=layer_id,
                expert_id=expert_id,
                activation=flat[None, token_indices].reshape(-1, hidden),
                deadline_ns=deadline_ns,
            )
            final.index_add_(
                0,
                token_indices,
                (output * routing_weights[token_indices, ranks, None]).to(flat.dtype),
            )
            events.append(event)
        self._events.extend(events)
        return MoeExecutionResult(
            output=final.reshape_as(hidden_states),
            events=tuple(events),
            metrics={
                "remote_expert_calls": len(events),
                "bytes_transferred": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
                "expert_critical_path_ns": sum(item.elapsed_ns for item in events),
                "serial_waits_per_token": float(sum(item.serial_waits for item in events)),
                "messages_per_token": float(sum(item.total_messages for item in events)),
                "payload_bytes_per_token": float(
                    sum(item.request_bytes + item.response_bytes for item in events)
                ),
                "critical_path_messages": sum(
                    item.critical_path_messages for item in events
                ),
                "critical_path_sync_rounds": sum(
                    item.critical_path_sync_rounds for item in events
                ),
                "root_dispatches": sum(item.root_dispatches for item in events),
                "coordinator_waits": sum(item.coordinator_waits for item in events),
                "coordinator_sync_rounds": sum(
                    item.coordinator_sync_rounds for item in events
                ),
                "worker_sync_rounds": sum(item.worker_sync_rounds for item in events),
                "fanout_depth": max((item.fanout_depth for item in events), default=0),
                "fanout_nodes": sum(item.fanout_nodes for item in events),
                "coordinator_activation_bytes": 0,
                "worker_to_worker_bytes": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
            },
        )


class MicroshardRemoteBackend(_SessionBackend):
    """Dispatch matched slices concurrently and reduce them deterministically."""

    def __init__(
        self,
        *,
        targets: dict[tuple[int, int], list[MicroshardTarget]],
        model_id: str,
        model_revision: str,
        model_fingerprint: str,
        quantization_fingerprint: str,
        topology_id: str,
        route_generation: int,
        maximum_parallel_requests: int = 32,
        fanout_branching_factor: int = 8,
        reduction_branching_factor: int = 8,
    ) -> None:
        super().__init__()
        if maximum_parallel_requests <= 0:
            raise ValueError("maximum_parallel_requests must be positive")
        if reduction_branching_factor < 2 or reduction_branching_factor > 32:
            raise ValueError("reduction_branching_factor must be between 2 and 32")
        if fanout_branching_factor < 2 or fanout_branching_factor > 32:
            raise ValueError("fanout_branching_factor must be between 2 and 32")
        self.targets = {key: list(value) for key, value in targets.items()}
        for value in self.targets.values():
            physical_microshard_ownership([item.ownership for item in value])
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_fingerprint = model_fingerprint
        self.quantization_fingerprint = quantization_fingerprint
        self.topology_id = topology_id
        self.route_generation = route_generation
        self.maximum_parallel_requests = maximum_parallel_requests
        self.fanout_branching_factor = fanout_branching_factor
        self.reduction_branching_factor = reduction_branching_factor
        self._executor = ThreadPoolExecutor(
            max_workers=maximum_parallel_requests,
            thread_name_prefix="swarm-microshard",
        )
        topology_started = time.perf_counter_ns()
        self._fanout_topologies = {
            key: self._build_fanout_topology(self._ordered_targets(value))
            for key, value in self.targets.items()
        }
        self.topology_construction_ns = time.perf_counter_ns() - topology_started
        self._route_auth = _ExpertRouteAuthentication(
            topology_id=topology_id,
            route_generation=route_generation,
            model_revision=model_revision,
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
        )

    def close(self) -> None:
        super().close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def configure_route(
        self,
        lease: SignedExpertRouteLease,
        *,
        identity: WorkerIdentity,
        worker_id: str,
    ) -> None:
        self._route_auth.configure(lease, identity=identity, worker_id=worker_id)
        self.route_generation = lease.route_generation

    def capabilities(self) -> MoeBackendCapabilities:
        return MoeBackendCapabilities(
            backend="microshard-remote",
            whole_expert=False,
            native_microshard=True,
            local=False,
        )

    def cancel_session(self, session_id: str) -> None:
        clients = {
            target.ownership.worker_id: target.client
            for targets in self.targets.values()
            for target in targets
        }
        for worker_id, client in clients.items():
            control = getattr(client, "control", None)
            if control is not None:
                control(
                    "cancel_session",
                    session_id=session_id,
                    authentication=self._route_auth.request_authentication(worker_id),
                )
        super().cancel_session(session_id)

    @staticmethod
    def _ordered_targets(targets: list[MicroshardTarget]) -> list[MicroshardTarget]:
        return sorted(
            targets,
            key=lambda item: (
                item.ownership.hidden_start,
                item.ownership.hidden_end,
                item.ownership.worker_id,
            ),
        )

    def _build_fanout_topology(
        self, targets: list[MicroshardTarget]
    ) -> _FanoutTopology:
        leaf_index = {id(target): index for index, target in enumerate(targets)}

        def build(items: list[MicroshardTarget], prefix: str) -> tuple[_FanoutNode, ...]:
            if len(items) <= self.fanout_branching_factor:
                return tuple(
                    _FanoutNode(
                        node_id=f"{prefix}/worker-{leaf_index[id(target)]:06d}",
                        target=target,
                    )
                    for target in items
                )
            group_size = (len(items) + self.fanout_branching_factor - 1) // (
                self.fanout_branching_factor
            )
            groups = [
                items[index : index + group_size]
                for index in range(0, len(items), group_size)
            ]
            return tuple(
                _FanoutNode(
                    node_id=f"{prefix}/group-{index:04d}",
                    children=build(group, f"{prefix}/group-{index:04d}"),
                )
                for index, group in enumerate(groups)
            )

        root_children = build(targets, "root")

        def depth(node: _FanoutNode) -> int:
            return 1 if node.target is not None else 1 + max(depth(item) for item in node.children)

        def count(node: _FanoutNode) -> int:
            return 1 + sum(count(item) for item in node.children)

        return _FanoutTopology(
            root_children=root_children,
            depth=max((depth(item) for item in root_children), default=0),
            node_count=sum(count(item) for item in root_children),
        )

    @staticmethod
    def _sum_partial_group(group: tuple[np.ndarray, ...]) -> np.ndarray:
        result = np.zeros_like(group[0], dtype=np.float32)
        for partial in group:
            result += np.asarray(partial, dtype=np.float32)
        return result

    def _reduce_partials(
        self, partials: list[tuple[str, np.ndarray]]
    ) -> tuple[np.ndarray, int]:
        ordered = sorted(partials, key=lambda item: item[0])
        if not ordered:
            raise ValueError("at least one microshard partial is required")
        shape = ordered[0][1].shape
        if any(partial.shape != shape for _, partial in ordered):
            raise ValueError("remote microshard partial shapes do not match")
        level = [np.ascontiguousarray(partial, dtype=np.float32) for _, partial in ordered]
        rounds = 0
        while len(level) > 1:
            groups = [
                tuple(level[index : index + self.reduction_branching_factor])
                for index in range(0, len(level), self.reduction_branching_factor)
            ]
            futures = [self._executor.submit(self._sum_partial_group, group) for group in groups]
            # Result lookup is in group order, so completion timing cannot alter
            # floating-point addition order.
            level = [future.result() for future in futures]
            rounds += 1
        return level[0], rounds

    def _dispatch_target(
        self,
        *,
        target: MicroshardTarget,
        fanout_request_id: str,
        session_id: str,
        token_position: int,
        layer_id: int,
        expert_id: int,
        source: np.ndarray,
        deadline_ns: int,
    ) -> tuple[str, np.ndarray, int, int, int]:
        owner = target.ownership
        subrequest_id = (
            f"{fanout_request_id}:slice-{owner.hidden_start}-{owner.hidden_end}"
        )
        request = ExpertExecutionRequest(
            request_id=subrequest_id,
            session_id=session_id,
            token_position=token_position,
            sequence_id=token_position,
            route_generation=self.route_generation,
            topology_id=self.topology_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            model_fingerprint=self.model_fingerprint,
            quantization_fingerprint=self.quantization_fingerprint,
            layer_id=layer_id,
            batch_rows=int(source.shape[0]),
            latent_dimension=int(source.shape[1]),
            expert_ids=[expert_id],
            expert_hashes={expert_id: owner.content_hash},
            routing_weights=[1.0],
            top_k=1,
            response_mode=ExpertResponseMode.PER_WORKER_FAST,
            activations={},
            deadline_ns=deadline_ns,
            execution_mode=ExpertExecutionMode.MICROSHARD,
            determinism_mode=DeterminismMode.EXACT,
            compression=TransportCodec.RAW_FP32,
            hidden_start=owner.hidden_start,
            hidden_end=owner.hidden_end,
            down_accumulators=None,
            microshard_final=False,
            reduction_mode=ReductionMode.FIXED_ORDER_FP32,
            metadata={"exact_contribution_representation": "unweighted_expert_output"},
            authentication=self._route_auth.request_authentication(owner.worker_id),
        )
        started = time.perf_counter_ns()
        response, result, transport = target.client.execute(request, source)
        elapsed_ns = time.perf_counter_ns() - started
        self._route_auth.verify_response(
            peer_worker_id=owner.worker_id,
            request=request,
            response=response,
            result=result,
        )
        if getattr(response, "status", "ok") != "ok":
            raise RuntimeError(str(getattr(response, "error", "remote microshard failed")))
        partial = np.ascontiguousarray(result, dtype=np.float32)
        expected_shape = (source.shape[0], source.shape[1])
        if partial.shape != expected_shape:
            raise ValueError("remote microshard partial shape mismatch")
        integrity = getattr(response, "integrity", None)
        if integrity is None:
            raise ValueError("remote microshard response is missing integrity metadata")
        remote_fingerprint = str(getattr(integrity, "model_fingerprint", ""))
        if self.model_fingerprint and remote_fingerprint != self.model_fingerprint:
            raise ValueError("remote microshard model fingerprint mismatch")
        hashes = getattr(integrity, "expert_hashes", {})
        if owner.content_hash and hashes.get(expert_id) != owner.content_hash:
            raise ValueError("remote microshard content hash mismatch")
        owner_key = (
            f"{owner.hidden_start:020d}:{owner.hidden_end:020d}:{owner.worker_id}"
        )
        return (
            owner_key,
            partial,
            int(transport.get("request_bytes", 0)),
            int(transport.get("response_bytes", 0)),
            int(transport.get("request_elapsed_ns", elapsed_ns)),
        )

    def _dispatch_node(
        self,
        *,
        node: _FanoutNode,
        collector: _FanoutCollector,
        fanout_request_id: str,
        session_id: str,
        token_position: int,
        layer_id: int,
        expert_id: int,
        source: np.ndarray,
        deadline_ns: int,
    ) -> None:
        if collector.cancelled():
            return
        try:
            if node.target is not None:
                collector.record_partial(
                    self._dispatch_target(
                        target=node.target,
                        fanout_request_id=fanout_request_id,
                        session_id=session_id,
                        token_position=token_position,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        source=source,
                        deadline_ns=deadline_ns,
                    )
                )
                return
            dispatch_started = time.perf_counter_ns()
            for child in node.children:
                if collector.cancelled():
                    break
                self._executor.submit(
                    self._dispatch_node,
                    node=child,
                    collector=collector,
                    fanout_request_id=fanout_request_id,
                    session_id=session_id,
                    token_position=token_position,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    source=source,
                    deadline_ns=deadline_ns,
                )
            collector.record_dispatch(time.perf_counter_ns() - dispatch_started)
        except BaseException as exc:
            collector.fail(exc)

    def execute_expert_rows(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        expert_id: int,
        activation: torch.Tensor,
        deadline_ns: int,
    ) -> tuple[torch.Tensor, MoeExecutionEvent]:
        targets = self.targets.get((layer_id, expert_id))
        if not targets:
            raise KeyError(f"no microshard owners for layer {layer_id} expert {expert_id}")
        source = activation.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        request_bytes = response_bytes = 0
        started = time.perf_counter_ns()
        fanout_request_id = (
            f"{request_id}:token-{token_position}:layer-{layer_id}:expert-{expert_id}:"
            "microshard-fanout"
        )
        ordered = self._ordered_targets(targets)
        topology = self._fanout_topologies[(layer_id, expert_id)]
        collector = _FanoutCollector(expected_partials=len(ordered))
        dispatch_started = time.perf_counter_ns()
        for node in topology.root_children:
            self._executor.submit(
                self._dispatch_node,
                node=node,
                collector=collector,
                fanout_request_id=fanout_request_id,
                session_id=session_id,
                token_position=token_position,
                layer_id=layer_id,
                expert_id=expert_id,
                source=source,
                deadline_ns=deadline_ns,
            )
        coordinator_dispatch_ns = time.perf_counter_ns() - dispatch_started
        partials = collector.wait(deadline_ns)
        request_bytes = collector.request_bytes
        response_bytes = collector.response_bytes
        scheduler_dispatch_ns = coordinator_dispatch_ns + collector.scheduler_dispatch_ns
        reduction_started = time.perf_counter_ns()
        accumulator, reduction_depth = self._reduce_partials(partials)
        reduction_ns = time.perf_counter_ns() - reduction_started
        output = torch.from_numpy(accumulator).to(
            device=activation.device, dtype=activation.dtype
        )
        event = MoeExecutionEvent(
            event="remote_microshard_result_consumed",
            backend="microshard-remote",
            session_id=session_id,
            request_id=fanout_request_id,
            token_position=token_position,
            layer_id=layer_id,
            expert_id=expert_id,
            worker_ids=tuple(item.ownership.worker_id for item in ordered),
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            elapsed_ns=time.perf_counter_ns() - started,
            result_hash=_result_hash(output),
            total_messages=2 * len(ordered),
            critical_path_messages=2 * topology.depth,
            serial_waits=topology.depth + reduction_depth,
            parallel_waits=len(ordered),
            fanout_depth=topology.depth,
            reduction_depth=reduction_depth,
            critical_path_sync_rounds=topology.depth + reduction_depth,
            scheduler_dispatch_ns=scheduler_dispatch_ns,
            reduction_ns=reduction_ns,
            communication_ns=collector.maximum_request_ns,
            root_dispatches=len(topology.root_children),
            coordinator_waits=0,
            coordinator_sync_rounds=0,
            worker_sync_rounds=topology.depth + reduction_depth,
            fanout_nodes=topology.node_count,
            topology_construction_ns=self.topology_construction_ns,
        )
        return output, event

    @torch.inference_mode()
    def execute_layer(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        deadline_ns: int,
    ) -> MoeExecutionResult:
        self._start_call(session_id, deadline_ns)
        flat, _, hidden = _geometry(hidden_states, router_logits, selected_experts, routing_weights)
        final = torch.zeros_like(flat)
        events: list[MoeExecutionEvent] = []
        for expert_id in range(int(router_logits.shape[-1])):
            token_indices, ranks = _selected_rows(selected_experts, expert_id)
            if token_indices.numel() == 0:
                continue
            output, event = self.execute_expert_rows(
                session_id=session_id,
                request_id=request_id,
                token_position=token_position,
                layer_id=layer_id,
                expert_id=expert_id,
                activation=flat[None, token_indices].reshape(-1, hidden),
                deadline_ns=deadline_ns,
            )
            final.index_add_(
                0,
                token_indices,
                (output * routing_weights[token_indices, ranks, None]).to(flat.dtype),
            )
            events.append(event)
        self._events.extend(events)
        return MoeExecutionResult(
            output=final.reshape_as(hidden_states),
            events=tuple(events),
            metrics={
                "remote_microshard_calls": len(events),
                "bytes_transferred": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
                "expert_critical_path_ns": sum(item.elapsed_ns for item in events),
                "reduction_mode": ReductionMode.TREE_FP32.value,
                "worker_partial_reduction_mode": ReductionMode.FIXED_ORDER_FP32.value,
                "logical_microshard_workers": sum(len(item.worker_ids) for item in events),
                "total_messages": sum(item.total_messages for item in events),
                "critical_path_messages": sum(
                    item.critical_path_messages for item in events
                ),
                "serial_waits": sum(item.serial_waits for item in events),
                "parallel_waits": sum(item.parallel_waits for item in events),
                "fanout_depth": max((item.fanout_depth for item in events), default=0),
                "reduction_depth": sum(item.reduction_depth for item in events),
                "critical_path_sync_rounds": sum(
                    item.critical_path_sync_rounds for item in events
                ),
                "scheduler_dispatch_ns": sum(
                    item.scheduler_dispatch_ns for item in events
                ),
                "reduction_ns": sum(item.reduction_ns for item in events),
                "communication_ns": sum(item.communication_ns for item in events),
                "coordinator_activation_bytes": 0,
                "worker_to_worker_bytes": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
                "maximum_parallel_requests": self.maximum_parallel_requests,
                "fanout_branching_factor": self.fanout_branching_factor,
                "reduction_branching_factor": self.reduction_branching_factor,
                "root_dispatches": sum(item.root_dispatches for item in events),
                "coordinator_waits": sum(item.coordinator_waits for item in events),
                "coordinator_sync_rounds": sum(
                    item.coordinator_sync_rounds for item in events
                ),
                "worker_sync_rounds": sum(item.worker_sync_rounds for item in events),
                "fanout_nodes": sum(item.fanout_nodes for item in events),
                "topology_construction_ns": self.topology_construction_ns,
                "serial_waits_per_token": float(sum(item.serial_waits for item in events)),
                "messages_per_token": float(sum(item.total_messages for item in events)),
                "payload_bytes_per_token": float(
                    sum(item.request_bytes + item.response_bytes for item in events)
                ),
            },
        )


class HybridMoeBackend(_SessionBackend):
    """Per-expert placement across native, Colibri, and direct remote backends."""

    def __init__(
        self,
        *,
        local: LocalMoeBackend | None = None,
        colibri: Any | None = None,
        whole_remote: WholeExpertRemoteBackend | None = None,
        microshard_remote: MicroshardRemoteBackend | None = None,
        placement: dict[tuple[int, int], str],
        allow_local_fallback: bool = False,
        fallback_placements: set[tuple[int, int]] | None = None,
        require_remote: bool = False,
    ) -> None:
        super().__init__()
        self.local = local
        self.colibri = colibri
        self.whole_remote = whole_remote
        self.microshard_remote = microshard_remote
        self.placement = dict(placement)
        self.require_remote = require_remote
        approved_fallbacks = (
            (set(self.placement) if allow_local_fallback else set())
            if fallback_placements is None
            else set(fallback_placements)
        )
        unknown_fallbacks = approved_fallbacks - set(self.placement)
        if unknown_fallbacks:
            raise ValueError(
                f"fallback permission names unknown expert placements: {sorted(unknown_fallbacks)}"
            )
        # Forced-remote validation is fail-closed even if a caller also supplied
        # a stale fallback flag.
        self.fallback_placements = set() if require_remote else approved_fallbacks
        self.allow_local_fallback = bool(self.fallback_placements)

    def capabilities(self) -> MoeBackendCapabilities:
        return MoeBackendCapabilities(
            backend="hybrid",
            whole_expert=True,
            native_microshard=True,
            local=self.local is not None or self.colibri is not None,
        )

    def configure_route(
        self,
        lease: SignedExpertRouteLease,
        *,
        identity: WorkerIdentity,
        worker_id: str,
    ) -> None:
        for backend in (self.whole_remote, self.microshard_remote):
            if backend is not None:
                backend.configure_route(lease, identity=identity, worker_id=worker_id)

    def status(self) -> dict[str, Any]:
        events = list(self._events)
        return {
            **super().status(),
            **(
                self.colibri.status()
                if self.colibri is not None and hasattr(self.colibri, "status")
                else {}
            ),
            "remote_whole_expert_calls": sum(
                item.event == "remote_whole_expert_result_consumed" for item in events
            ),
            "remote_microshard_calls": sum(
                item.event == "remote_microshard_result_consumed" for item in events
            ),
            "fallbacks": sum(item.event == "expert_local_fallback" for item in events),
            "bytes_transferred": sum(item.request_bytes + item.response_bytes for item in events),
            "expert_critical_path_ns": sum(item.elapsed_ns for item in events),
            "total_messages": sum(item.total_messages for item in events),
            "critical_path_messages": sum(item.critical_path_messages for item in events),
            "serial_waits": sum(item.serial_waits for item in events),
            "parallel_waits": sum(item.parallel_waits for item in events),
            "critical_path_sync_rounds": sum(
                item.critical_path_sync_rounds for item in events
            ),
            "root_dispatches": sum(item.root_dispatches for item in events),
            "coordinator_waits": sum(item.coordinator_waits for item in events),
            "coordinator_sync_rounds": sum(
                item.coordinator_sync_rounds for item in events
            ),
            "worker_sync_rounds": sum(item.worker_sync_rounds for item in events),
            "fanout_depth": max((item.fanout_depth for item in events), default=0),
            "reduction_depth": sum(item.reduction_depth for item in events),
            "fanout_nodes": sum(item.fanout_nodes for item in events),
            "topology_construction_ns": max(
                (item.topology_construction_ns for item in events), default=0
            ),
            "scheduler_dispatch_ns": sum(item.scheduler_dispatch_ns for item in events),
            "reduction_ns": sum(item.reduction_ns for item in events),
            "communication_ns": sum(item.communication_ns for item in events),
            "coordinator_activation_bytes": 0,
            "worker_to_worker_bytes": sum(
                item.request_bytes + item.response_bytes for item in events
            ),
            "reduction_mode": (
                ReductionMode.TREE_FP32.value
                if any(item.event == "remote_microshard_result_consumed" for item in events)
                else "none"
            ),
        }

    def open_session(self, session_id: str) -> None:
        super().open_session(session_id)
        for backend in (self.local, self.colibri, self.whole_remote, self.microshard_remote):
            if backend is not None:
                backend.open_session(session_id)

    def close_session(self, session_id: str) -> None:
        for backend in (self.local, self.colibri, self.whole_remote, self.microshard_remote):
            if backend is not None:
                backend.close_session(session_id)
        super().close_session(session_id)

    def cancel_session(self, session_id: str) -> None:
        for backend in (self.local, self.colibri, self.whole_remote, self.microshard_remote):
            if backend is not None:
                backend.cancel_session(session_id)
        super().cancel_session(session_id)

    def close(self) -> None:
        for backend in (self.local, self.colibri, self.whole_remote, self.microshard_remote):
            if backend is not None:
                backend.close()
        super().close()

    @torch.inference_mode()
    def execute_layer(
        self,
        *,
        session_id: str,
        request_id: str,
        token_position: int,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        deadline_ns: int,
    ) -> MoeExecutionResult:
        self._start_call(session_id, deadline_ns)
        flat, _, hidden = _geometry(hidden_states, router_logits, selected_experts, routing_weights)
        final = torch.zeros_like(flat)
        events: list[MoeExecutionEvent] = []
        remote_calls = 0
        remote_whole_calls = 0
        remote_microshard_calls = 0
        colibri_calls = 0
        for expert_id in range(int(router_logits.shape[-1])):
            token_indices, ranks = _selected_rows(selected_experts, expert_id)
            if token_indices.numel() == 0:
                continue
            source = flat[None, token_indices].reshape(-1, hidden)
            strategy = self.placement.get((layer_id, expert_id), "local")
            try:
                if strategy == "local":
                    if self.require_remote:
                        raise RuntimeError("forced-remote mode rejected local expert placement")
                    if self.local is None:
                        raise KeyError("local expert backend is unavailable")
                    output, workers, _ = self.local.execute_expert_rows(
                        layer_id=layer_id, expert_id=expert_id, activation=source
                    )
                    event = MoeExecutionEvent(
                        event="local_expert_result_consumed",
                        backend="local",
                        session_id=session_id,
                        request_id=request_id,
                        token_position=token_position,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        worker_ids=workers,
                        result_hash=_result_hash(output),
                    )
                elif strategy == "colibri":
                    if self.require_remote:
                        raise RuntimeError("forced-remote mode rejected colocated Colibri")
                    if self.colibri is None:
                        raise KeyError("Colibri expert backend is unavailable")
                    started = time.perf_counter_ns()
                    output, workers, _ = self.colibri.execute_expert_rows(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        activation=source,
                    )
                    colibri_calls += 1
                    event = MoeExecutionEvent(
                        event="colibri_expert_result_consumed",
                        backend="colibri",
                        session_id=session_id,
                        request_id=request_id,
                        token_position=token_position,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        worker_ids=workers,
                        elapsed_ns=time.perf_counter_ns() - started,
                        result_hash=_result_hash(output),
                    )
                elif strategy == "whole-remote":
                    if self.whole_remote is None:
                        raise KeyError("whole-expert remote backend is unavailable")
                    output, event = self.whole_remote.execute_expert_rows(
                        session_id=session_id,
                        request_id=request_id,
                        token_position=token_position,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        activation=source,
                        deadline_ns=deadline_ns,
                    )
                    remote_calls += 1
                    remote_whole_calls += 1
                elif strategy == "microshard-remote":
                    if self.microshard_remote is None:
                        raise KeyError("microshard remote backend is unavailable")
                    output, event = self.microshard_remote.execute_expert_rows(
                        session_id=session_id,
                        request_id=request_id,
                        token_position=token_position,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        activation=source,
                        deadline_ns=deadline_ns,
                    )
                    remote_calls += 1
                    remote_microshard_calls += 1
                else:
                    raise ValueError(f"unknown expert placement strategy {strategy!r}")
            except Exception as exc:
                if (
                    (layer_id, expert_id) not in self.fallback_placements
                    or self.require_remote
                    or strategy == "local"
                    or self.local is None
                ):
                    raise
                output, workers, _ = self.local.execute_expert_rows(
                    layer_id=layer_id, expert_id=expert_id, activation=source
                )
                event = MoeExecutionEvent(
                    event="expert_local_fallback",
                    backend="local",
                    session_id=session_id,
                    request_id=request_id,
                    token_position=token_position,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    worker_ids=workers,
                    result_hash=_result_hash(output),
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
            final.index_add_(
                0,
                token_indices,
                (output * routing_weights[token_indices, ranks, None]).to(flat.dtype),
            )
            events.append(event)
        if self.require_remote and remote_calls == 0:
            raise RuntimeError("forced-remote mode produced no remote expert contribution")
        self._events.extend(events)
        return MoeExecutionResult(
            output=final.reshape_as(hidden_states),
            events=tuple(events),
            metrics={
                "remote_expert_calls": remote_calls,
                "remote_whole_expert_calls": remote_whole_calls,
                "remote_microshard_calls": remote_microshard_calls,
                "colibri_expert_calls": colibri_calls,
                "fallbacks": sum(item.event == "expert_local_fallback" for item in events),
                "bytes_transferred": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
                "expert_critical_path_ns": sum(item.elapsed_ns for item in events),
                "logical_microshard_workers": sum(
                    len(item.worker_ids)
                    for item in events
                    if item.event == "remote_microshard_result_consumed"
                ),
                "total_messages": sum(item.total_messages for item in events),
                "serial_waits": sum(item.serial_waits for item in events),
                "serial_waits_per_token": float(sum(item.serial_waits for item in events)),
                "messages_per_token": float(sum(item.total_messages for item in events)),
                "payload_bytes_per_token": float(
                    sum(item.request_bytes + item.response_bytes for item in events)
                ),
                "critical_path_messages": sum(
                    item.critical_path_messages for item in events
                ),
                "parallel_waits": sum(item.parallel_waits for item in events),
                "critical_path_sync_rounds": sum(
                    item.critical_path_sync_rounds for item in events
                ),
                "fanout_depth": max((item.fanout_depth for item in events), default=0),
                "reduction_depth": sum(item.reduction_depth for item in events),
                "root_dispatches": sum(item.root_dispatches for item in events),
                "coordinator_waits": sum(item.coordinator_waits for item in events),
                "coordinator_sync_rounds": sum(
                    item.coordinator_sync_rounds for item in events
                ),
                "worker_sync_rounds": sum(item.worker_sync_rounds for item in events),
                "fanout_nodes": sum(item.fanout_nodes for item in events),
                "topology_construction_ns": max(
                    (item.topology_construction_ns for item in events), default=0
                ),
                "scheduler_dispatch_ns": sum(
                    item.scheduler_dispatch_ns for item in events
                ),
                "reduction_ns": sum(item.reduction_ns for item in events),
                "communication_ns": sum(item.communication_ns for item in events),
                "coordinator_activation_bytes": 0,
                "worker_to_worker_bytes": sum(
                    item.request_bytes + item.response_bytes for item in events
                ),
            },
        )


__all__ = [
    "HybridMoeBackend",
    "LocalMoeBackend",
    "MicroshardRemoteBackend",
    "MicroshardTarget",
    "MoeBackendCapabilities",
    "MoeExecutionBackend",
    "MoeExecutionEvent",
    "MoeExecutionResult",
    "WholeExpertRemoteBackend",
    "WholeExpertTarget",
]
