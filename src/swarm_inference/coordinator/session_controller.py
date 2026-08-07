"""Product session lifecycle and token streaming over a deployed stage ring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

import torch

from swarm_inference.coordinator.deployment import DeploymentManager
from swarm_inference.coordinator.durable_state import DurableCoordinatorState
from swarm_inference.coordinator.event_stream import BoundedRequestEventStream
from swarm_inference.exceptions import BackpressureError, IntegrityError, TransportError
from swarm_inference.protocol.messages import (
    Ack,
    StreamEventType,
    SubmitRequest,
)
from swarm_inference.protocol.product import (
    CancelProductResponse,
    PlanWorkerAssignment,
    ProductRequestPhase,
    ProductRequestRecoveryState,
    ProductSessionStatus,
    ProductStagePlan,
    ProductTokenPublication,
)
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    GetStageStatusRequest,
    OpenStageSessionRequest,
    StageActionResponse,
    StageStatusResponse,
    TokenizeStageRequest,
    TokenizeStageResponse,
)
from swarm_inference.runtime.telemetry import ProductTelemetry
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.security.tls import TlsClientConfig
from swarm_inference.transport.stage_ring_connection import StageRingConnectionPool
from swarm_inference.transport.stage_tensor import pack_tensor, unpack_tensor


class SessionControlTransport(Protocol):
    async def open_stage_session(
        self, endpoint: str, request: OpenStageSessionRequest
    ) -> StageActionResponse: ...

    async def close_stage_session(
        self, endpoint: str, request: CloseStageSessionRequest
    ) -> StageActionResponse: ...

    async def cancel_stage_session(
        self, endpoint: str, request: CancelStageSessionRequest
    ) -> StageActionResponse: ...

    async def tokenize_stage(
        self, endpoint: str, request: TokenizeStageRequest
    ) -> TokenizeStageResponse: ...

    async def get_stage_status(
        self, endpoint: str, request: GetStageStatusRequest
    ) -> StageStatusResponse: ...


@dataclass(slots=True)
class _ProductRequestState:
    submission: SubmitRequest
    stream: BoundedRequestEventStream
    session_id: str
    plan: ProductStagePlan | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    pending_publications: dict[int, ProductTokenPublication] = field(default_factory=dict)
    replay_publications: dict[int, ProductTokenPublication] = field(default_factory=dict)
    opened_stage_ids: set[int] = field(default_factory=set)
    publication_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancellation_requested: bool = False
    cancellation_reason: str = "client cancelled"
    disconnected: bool = False
    replaying: bool = False
    request_generation: int = 1
    recovery_count: int = 0
    status: ProductRequestPhase = ProductRequestPhase.PENDING
    durable: ProductRequestRecoveryState | None = None
    released_kv_bytes: int = 0
    token_accepted_times_s: list[float] = field(default_factory=list)
    error: str | None = None
    started_s: float = field(default_factory=time.perf_counter)
    first_token_s: float | None = None
    task: asyncio.Task[None] | None = None


class ProductSessionController:
    """Open session state only, drive token inputs, and publish ordered events."""

    def __init__(
        self,
        *,
        deployments: DeploymentManager,
        transport: SessionControlTransport,
        event_queue_capacity: int,
        request_timeout_s: float,
        data_queue_capacity: int = 256,
        state: DurableCoordinatorState,
        telemetry: ProductTelemetry,
        cleanup_timeout_s: float,
        recovery_timeout_s: float,
        maximum_recovery_attempts: int,
        data_tls: TlsClientConfig | None = None,
    ) -> None:
        self.deployments = deployments
        self.transport = transport
        self.event_queue_capacity = event_queue_capacity
        self.request_timeout_s = request_timeout_s
        self.state = state
        self.telemetry = telemetry
        self.cleanup_timeout_s = cleanup_timeout_s
        self.recovery_timeout_s = recovery_timeout_s
        self.maximum_recovery_attempts = maximum_recovery_attempts
        self.data_pool = StageRingConnectionPool(
            queue_capacity=data_queue_capacity,
            read_timeout_s=request_timeout_s,
            write_timeout_s=min(30.0, request_timeout_s),
            tls=data_tls,
        )
        self._active: dict[str, _ProductRequestState] = {}
        self._terminal = self.state.load_requests()
        self._shutting_down = False

    @property
    def active_count(self) -> int:
        return len(self._active)

    def start(self, submission: SubmitRequest) -> BoundedRequestEventStream:
        if self._shutting_down:
            raise RuntimeError("coordinator is shutting down")
        if submission.request_id in self._active or submission.request_id in self._terminal:
            raise ValueError(f"duplicate or replayed request ID {submission.request_id}")
        stream = BoundedRequestEventStream(
            request_id=submission.request_id,
            capacity=self.event_queue_capacity,
        )
        state = _ProductRequestState(
            submission=submission,
            stream=stream,
            session_id=f"session-{uuid4().hex}",
            request_generation=submission.request_generation,
        )
        self._active[submission.request_id] = state
        state.task = asyncio.create_task(
            self._run(state),
            name=f"product-request:{submission.request_id}",
        )
        return stream

    async def _run(self, state: _ProductRequestState) -> None:
        submission = state.submission
        try:
            state.status = ProductRequestPhase.PENDING
            state.stream.publish(
                StreamEventType.REQUEST_ACCEPTED,
                status_detail="request admitted to the product session controller",
                model_revision=submission.model_revision,
            )
            plan = self.deployments.ready_plan(
                model_id=submission.model_id,
                model_revision=submission.model_revision,
            )
            state.plan = plan
            state.stream.publish(
                StreamEventType.TOPOLOGY_SELECTED,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                status_detail=plan.report.reason_for_selection,
            )
            state.prompt_token_ids = await self._prompt_tokens(state)
            requested_tokens = len(state.prompt_token_ids) + submission.max_new_tokens
            if requested_tokens > plan.max_sequence_tokens:
                raise ValueError(
                    f"request needs {requested_tokens} sequence tokens but topology was planned "
                    f"for at most {plan.max_sequence_tokens}"
                )
            now_unix_ns = time.time_ns()
            state.status = ProductRequestPhase.RUNNING
            state.durable = ProductRequestRecoveryState(
                request_id=submission.request_id,
                request_generation=state.request_generation,
                session_id=state.session_id,
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
                tokenizer_revision=plan.model.tokenizer_revision,
                topology_id=plan.topology_id,
                route_generation=plan.generation,
                prompt_token_ids=list(state.prompt_token_ids),
                accepted_generated_token_ids=[],
                next_token_position=0,
                active_workers=[item.worker_id for item in plan.assignments],
                stage_assignments=list(plan.assignments),
                status=ProductRequestPhase.RUNNING,
                started_unix_ns=now_unix_ns,
                updated_unix_ns=now_unix_ns,
            )
            self.state.save_request(state.durable)
            await self._open_sessions(state)
            self.telemetry.emit(
                "session_opened",
                request_id=submission.request_id,
                request_generation=state.request_generation,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                route_generation=plan.generation,
                worker_ids=[item.worker_id for item in plan.assignments],
            )
            state.stream.publish(
                StreamEventType.SESSION_OPENED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                status_detail=f"opened on {plan.stage_count} persistent stages",
            )
            state.stream.publish(
                StreamEventType.PREFILL_STARTED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                token_position=0,
                status_detail=f"prefill started with {len(state.prompt_token_ids)} tokens",
            )
            last_token: int | None = None
            for output_position in range(submission.max_new_tokens):
                if state.cancellation_requested:
                    raise asyncio.CancelledError
                while True:
                    try:
                        await self._ensure_route_healthy(state)
                        response_token = await self._execute_token_step(
                            state,
                            output_position=output_position,
                            last_token=last_token,
                            replay_only=False,
                        )
                        publication = await self._wait_for_publication(
                            state,
                            output_position,
                            replay_only=False,
                        )
                        if publication.token_id != response_token:
                            raise IntegrityError(
                                f"stage-zero publication token {publication.token_id} differs "
                                f"from direct ring response {response_token} at position "
                                f"{output_position}"
                            )
                        self._accept_token(state, publication)
                        last_token = response_token
                        break
                    except (TimeoutError, TransportError) as exc:
                        await self._recover(state, exc)
                        last_token = state.output_token_ids[-1] if state.output_token_ids else None
            state.released_kv_bytes += await self._close_sessions(state, cancel=False)
            state.status = ProductRequestPhase.COMPLETED
            self._update_durable(state, status=ProductRequestPhase.COMPLETED)
            self.telemetry.emit(
                "session_closed",
                request_id=submission.request_id,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                route_generation=state.plan.generation if state.plan else plan.generation,
                released_kv_bytes=state.released_kv_bytes,
            )
            state.stream.publish(
                StreamEventType.SESSION_CLOSED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                status_detail="session KV state released; stages remain resident",
            )
            completed_s = time.perf_counter()
            elapsed = completed_s - state.started_s
            state.stream.finish(
                StreamEventType.REQUEST_COMPLETED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                status_detail="distributed request completed",
                final_token_ids=list(state.output_token_ids),
                timing_metrics={
                    "time_to_first_token_s": (
                        state.first_token_s - state.started_s
                        if state.first_token_s is not None
                        else elapsed
                    ),
                    "end_to_end_s": elapsed,
                    "generated_tokens_per_second": (
                        len(state.output_token_ids) / elapsed if elapsed > 0 else 0.0
                    ),
                },
            )
        except asyncio.CancelledError:
            state.cancellation_requested = True
            state.released_kv_bytes += await self._bounded_close_sessions(state, cancel=True)
            state.status = ProductRequestPhase.CANCELLED
            self._update_durable(
                state,
                status=ProductRequestPhase.CANCELLED,
                last_error=state.cancellation_reason,
            )
            self.telemetry.emit(
                "session_cancelled",
                request_id=submission.request_id,
                request_generation=state.request_generation,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else None,
                route_generation=state.plan.generation if state.plan else None,
                released_kv_bytes=state.released_kv_bytes,
                reason=state.cancellation_reason,
            )
            state.stream.fail(
                state.cancellation_reason,
                cancelled=True,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else None,
                model_revision=state.plan.model.model_revision if state.plan else None,
            )
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            state.released_kv_bytes += await self._bounded_close_sessions(state, cancel=True)
            state.status = ProductRequestPhase.FAILED
            self._update_durable(
                state,
                status=ProductRequestPhase.FAILED,
                last_error=state.error,
            )
            state.stream.fail(
                state.error,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else None,
                model_revision=state.plan.model.model_revision if state.plan else None,
            )
        finally:
            state.publication_event.set()
            if state.durable is not None:
                self._terminal[submission.request_id] = state.durable.model_copy(deep=True)
            self._active.pop(submission.request_id, None)

    async def _prompt_tokens(self, state: _ProductRequestState) -> list[int]:
        if state.submission.prompt_token_ids:
            return list(state.submission.prompt_token_ids)
        plan = state.plan
        assert plan is not None
        stage_zero = plan.assignments[0]
        response = await self.transport.tokenize_stage(
            stage_zero.control_endpoint,
            TokenizeStageRequest(
                worker_id=stage_zero.worker_id,
                request_id=f"{state.submission.request_id}:tokenize",
                model_id=plan.model.model_id,
                model_revision=plan.model.model_revision,
                tokenizer_revision=plan.model.tokenizer_revision,
                topology_id=plan.topology_id,
                route_generation=plan.generation,
                stage_id=0,
                device=stage_zero.device,
                dtype=plan.model.dtype,
                text=state.submission.prompt or "",
            ),
        )
        if not response.token_ids:
            raise IntegrityError("stage-zero tokenizer returned an empty prompt")
        return [int(value) for value in response.token_ids]

    def _session_request(
        self,
        state: _ProductRequestState,
        stage_id: int,
    ) -> OpenStageSessionRequest:
        plan = state.plan
        assert plan is not None
        item = plan.assignments[stage_id]
        return OpenStageSessionRequest(
            worker_id=item.worker_id,
            request_id=f"{state.submission.request_id}:open:{stage_id}",
            model_id=plan.model.model_id,
            model_revision=plan.model.model_revision,
            tokenizer_revision=plan.model.tokenizer_revision,
            topology_id=plan.topology_id,
            route_generation=plan.generation,
            stage_id=stage_id,
            device=item.device,
            dtype=plan.model.dtype,
            session_id=state.session_id,
            request_generation=state.request_generation,
        )

    async def _open_sessions(self, state: _ProductRequestState) -> None:
        plan = state.plan
        assert plan is not None
        requests = [self._session_request(state, item.stage_id) for item in plan.assignments]
        results = await asyncio.gather(
            *(
                self.transport.open_stage_session(item.control_endpoint, request)
                for item, request in zip(plan.assignments, requests, strict=True)
            ),
            return_exceptions=True,
        )
        errors: list[str] = []
        for item, result in zip(plan.assignments, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"stage {item.stage_id}: {result}")
            elif not result.accepted:
                errors.append(f"stage {item.stage_id}: {result.detail}")
            else:
                state.opened_stage_ids.add(item.stage_id)
        if errors:
            raise RuntimeError("session open failed: " + "; ".join(errors))

    async def _close_sessions(self, state: _ProductRequestState, *, cancel: bool) -> int:
        plan = state.plan
        if plan is None or not state.opened_stage_ids:
            return 0
        stage_ids = sorted(state.opened_stage_ids)
        operations = []
        for stage_id in stage_ids:
            item = plan.assignments[stage_id]
            base = self._session_request(state, stage_id).model_copy(
                update={"request_id": f"{state.submission.request_id}:close:{stage_id}"}
            )
            if cancel:
                operations.append(
                    self.transport.cancel_stage_session(
                        item.control_endpoint,
                        CancelStageSessionRequest(**base.model_dump()),
                    )
                )
            else:
                operations.append(
                    self.transport.close_stage_session(
                        item.control_endpoint,
                        CloseStageSessionRequest(**base.model_dump()),
                    )
                )
        results = await asyncio.gather(*operations, return_exceptions=True)
        errors: list[str] = []
        released_kv_bytes = 0
        for stage_id, result in zip(stage_ids, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"stage {stage_id}: {type(result).__name__}: {result}")
            elif not result.accepted:
                errors.append(f"stage {stage_id}: {result.detail}")
            else:
                state.opened_stage_ids.discard(stage_id)
                released_kv_bytes += result.released_kv_bytes
        if cancel:
            # Failed workers cannot acknowledge cleanup; their reservations are
            # stale and must not keep the coordinator request alive.
            state.opened_stage_ids.clear()
        if errors and not cancel:
            raise RuntimeError("session close failed: " + "; ".join(errors))
        return released_kv_bytes

    async def _bounded_close_sessions(
        self,
        state: _ProductRequestState,
        *,
        cancel: bool,
    ) -> int:
        try:
            return await asyncio.wait_for(
                self._close_sessions(state, cancel=cancel),
                timeout=self.cleanup_timeout_s,
            )
        except TimeoutError:
            state.opened_stage_ids.clear()
            return 0

    async def _execute_token_step(
        self,
        state: _ProductRequestState,
        *,
        output_position: int,
        last_token: int | None,
        replay_only: bool,
    ) -> int:
        plan = state.plan
        assert plan is not None
        stage_zero = plan.assignments[0]
        prefill = output_position == 0
        if prefill:
            values = state.prompt_token_ids
        else:
            assert last_token is not None
            values = [last_token]
        cache_position = 0 if prefill else len(state.prompt_token_ids) + output_position - 1
        packed = pack_tensor(
            torch.tensor([values], dtype=torch.int64),
            requested_mode="none",
        )
        message = StageMessage(
            operation=Operation.PREFILL if prefill else Operation.DECODE,
            model_revision=plan.model.model_revision,
            tokenizer_revision=plan.model.tokenizer_revision,
            topology_id=plan.topology_id,
            stage_id=0,
            layer_start=stage_zero.assignment.layer_start,
            layer_end=stage_zero.assignment.layer_end,
            session_id=state.session_id,
            request_id=state.submission.request_id,
            sequence_number=output_position,
            token_position=output_position,
            source_stage=-1,
            destination_stage=0,
            tensor_shape=packed.shape,
            tensor_dtype=packed.dtype,
            compression_mode=packed.compression_mode,
            payload=packed.payload,
            attributes={
                "model_id": plan.model.model_id,
                "route_generation": plan.generation,
                "request_generation": state.request_generation,
                "replay_only": replay_only,
                "source_worker_id": "coordinator",
                "destination_worker_id": stage_zero.worker_id,
                "cache_position_start": cache_position,
                "deadline_ns": time.time_ns() + int(self.request_timeout_s * 1_000_000_000),
                "expert_trace": [],
                "expert_metrics": {},
                "tensor": packed.attributes(),
            },
        )
        response = await self.data_pool.send(stage_zero.data_endpoint, message)
        if response.operation == Operation.ERROR:
            detail = str(response.attributes.get("error", "stage ring failed"))
            if any(item.data_endpoint in detail for item in plan.assignments):
                raise TransportError(detail)
            raise TransportError(
                f"stage-zero worker {stage_zero.worker_id} at {stage_zero.data_endpoint}: {detail}"
            )
        if (
            response.operation != Operation.TOKEN_RESULT
            or response.status != "OK"
            or response.request_id != state.submission.request_id
            or response.session_id != state.session_id
            or response.topology_id != plan.topology_id
            or response.model_revision != plan.model.model_revision
            or response.token_position != output_position
            or response.destination_stage != -1
            or int(response.attributes.get("route_generation", -1)) != plan.generation
            or int(response.attributes.get("request_generation", -1)) != state.request_generation
            or bool(response.attributes.get("replay_only", False)) != replay_only
        ):
            raise IntegrityError("stage-zero token result identity is invalid")
        tensor_metadata = response.attributes.get("tensor")
        if not isinstance(tensor_metadata, dict):
            raise IntegrityError("stage-zero token result lacks tensor metadata")
        token_tensor, _ = unpack_tensor(response.payload, dict(tensor_metadata))
        if token_tensor.numel() != 1 or token_tensor.dtype != torch.int64:
            raise IntegrityError("stage-zero result is not one int64 token")
        return int(token_tensor.item())

    async def _wait_for_publication(
        self,
        state: _ProductRequestState,
        position: int,
        *,
        replay_only: bool,
    ) -> ProductTokenPublication:
        deadline = asyncio.get_running_loop().time() + self.request_timeout_s
        publications = state.replay_publications if replay_only else state.pending_publications
        while position not in publications:
            if state.error is not None:
                raise RuntimeError(state.error)
            if state.cancellation_requested:
                raise asyncio.CancelledError
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"token publication {position} timed out")
            state.publication_event.clear()
            if position in publications:
                break
            await asyncio.wait_for(state.publication_event.wait(), timeout=remaining)
        return publications.pop(position)

    async def publish_token(self, publication: ProductTokenPublication) -> Ack:
        state = self._active.get(publication.request_id)
        if state is None:
            return Ack(accepted=False, detail="unknown or closed product request")
        plan = state.plan
        if plan is None:
            return Ack(accepted=False, detail="request has no selected topology")
        stage_zero = plan.assignments[0]
        if not publication.signature:
            return Ack(accepted=False, detail="token publication signature is missing")
        try:
            capability = self.deployments.registry.capability(publication.worker_id)
            verify_signature(
                capability.public_key,
                canonical_json_bytes(publication.model_dump(mode="json", exclude={"signature"})),
                publication.signature,
            )
        except Exception as exc:
            return Ack(
                accepted=False,
                detail=f"token publication authentication failed: {type(exc).__name__}: {exc}",
            )
        if state.cancellation_requested:
            return Ack(accepted=False, detail="request cancellation is in progress")
        if state.status == ProductRequestPhase.RECOVERING and not state.replaying:
            return Ack(accepted=False, detail="old route output is disabled during recovery")
        if (
            publication.worker_id != stage_zero.worker_id
            or publication.session_id != state.session_id
            or publication.topology_id != plan.topology_id
            or publication.route_generation != plan.generation
            or publication.model_revision != plan.model.model_revision
            or publication.request_generation != state.request_generation
        ):
            return Ack(
                accepted=False,
                detail="stale token publication identity does not match the active generation",
            )
        if publication.replay_only != state.replaying:
            return Ack(accepted=False, detail="token publication replay marker is invalid")
        try:
            self._validate_expert_trace(publication, plan)
        except (IntegrityError, ValueError) as exc:
            return Ack(accepted=False, detail=f"expert contribution trace is invalid: {exc}")
        position = publication.token_position
        if not publication.replay_only and position < len(state.output_token_ids):
            if state.output_token_ids[position] == publication.token_id:
                return Ack(accepted=True, detail="duplicate token publication ignored")
            state.error = "conflicting duplicate token publication"
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        if not publication.replay_only and position != len(state.output_token_ids):
            state.error = (
                f"out-of-order token publication {position}; expected {len(state.output_token_ids)}"
            )
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        target = (
            state.replay_publications if publication.replay_only else state.pending_publications
        )
        existing = target.get(position)
        if existing is not None:
            if existing.token_id == publication.token_id:
                return Ack(accepted=True, detail="duplicate token publication ignored")
            state.error = "conflicting duplicate token publication"
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        target[position] = publication.model_copy(deep=True)
        state.publication_event.set()
        return Ack(accepted=True, detail="token publication staged for coordinator verification")

    @staticmethod
    def _validate_expert_trace(
        publication: ProductTokenPublication,
        plan: ProductStagePlan,
    ) -> None:
        placements = {
            (item.layer_id, item.expert_id): item
            for stage in plan.expert_plans
            for item in stage.placements
        }
        remote_events = 0
        remote_whole_events = 0
        remote_microshard_events = 0
        fallback_events = 0
        traced_bytes = 0
        for event in publication.expert_trace:
            event_name = str(event.get("event", ""))
            if event.get("session_id") != publication.session_id:
                raise IntegrityError("expert trace session identity mismatch")
            if int(event.get("token_position", -1)) != publication.token_position:
                raise IntegrityError("expert trace token identity mismatch")
            request_id = str(event.get("request_id", ""))
            if request_id != publication.request_id and not request_id.startswith(
                f"{publication.request_id}:"
            ):
                raise IntegrityError("expert trace request identity mismatch")
            layer_id = int(event.get("layer_id", -1))
            expert_id = int(event.get("expert_id", -1))
            placement = placements.get((layer_id, expert_id))
            if event_name == "local_expert_result_consumed":
                if placement is None or placement.strategy != "local":
                    raise IntegrityError("local expert contribution is not present in the plan")
                if any(stage.require_remote_experts for stage in plan.expert_plans):
                    raise IntegrityError("forced-remote execution consumed a local expert result")
                if not str(event.get("result_hash", "")).startswith("sha256:"):
                    raise IntegrityError("local expert contribution has no result hash")
                continue
            if event_name == "expert_local_fallback":
                if placement is None or not placement.local_fallback_permitted:
                    raise IntegrityError("unplanned expert fallback was reported")
                if any(stage.require_remote_experts for stage in plan.expert_plans):
                    raise IntegrityError("forced-remote execution reported a fallback")
                if not event.get("fallback_reason"):
                    raise IntegrityError("expert fallback has no visible reason")
                if not str(event.get("result_hash", "")).startswith("sha256:"):
                    raise IntegrityError("expert fallback has no result hash")
                fallback_events += 1
                continue
            if event_name not in {
                "remote_whole_expert_result_consumed",
                "remote_microshard_result_consumed",
            }:
                raise IntegrityError(f"unknown expert contribution event {event_name!r}")
            remote_events += 1
            if event_name == "remote_whole_expert_result_consumed":
                remote_whole_events += 1
            else:
                remote_microshard_events += 1
            expected_strategy = (
                "whole-remote"
                if event_name == "remote_whole_expert_result_consumed"
                else "microshard-remote"
            )
            if placement is None or placement.strategy != expected_strategy:
                raise IntegrityError("remote contribution does not match the installed plan")
            workers = {str(item) for item in event.get("worker_ids", [])}
            if not workers or workers != set(placement.worker_ids):
                raise IntegrityError(
                    "remote contribution worker identities do not exactly match the plan"
                )
            if int(event.get("request_bytes", 0)) <= 0 or int(event.get("response_bytes", 0)) <= 0:
                raise IntegrityError("remote contribution has no data-plane byte proof")
            traced_bytes += int(event["request_bytes"]) + int(event["response_bytes"])
            if not str(event.get("result_hash", "")).startswith("sha256:"):
                raise IntegrityError("remote contribution has no result hash")
        if any(stage.require_remote_experts for stage in plan.expert_plans) and remote_events == 0:
            raise IntegrityError("forced-remote execution produced no remote contribution")
        metrics = publication.expert_metrics
        expected_metrics = {
            "remote_expert_calls": remote_events,
            "remote_whole_expert_calls": remote_whole_events,
            "remote_microshard_calls": remote_microshard_events,
            "fallbacks": fallback_events,
            "bytes_transferred": traced_bytes,
        }
        for name, expected in expected_metrics.items():
            if name in metrics and int(metrics[name]) != expected:
                raise IntegrityError(
                    f"expert metric {name}={metrics[name]!r} does not match trace value {expected}"
                )
        if remote_events and int(metrics.get("remote_expert_calls", -1)) != remote_events:
            raise IntegrityError("remote expert call metric is missing from contribution proof")
        if remote_events and int(metrics.get("bytes_transferred", -1)) != traced_bytes:
            raise IntegrityError("remote expert byte metric is missing from contribution proof")

    def _update_durable(
        self,
        state: _ProductRequestState,
        *,
        status: ProductRequestPhase | None = None,
        last_error: str | None = None,
    ) -> None:
        durable = state.durable
        plan = state.plan
        if durable is None or plan is None:
            return
        updated = durable.model_copy(
            update={
                "request_generation": state.request_generation,
                "session_id": state.session_id,
                "topology_id": plan.topology_id,
                "route_generation": plan.generation,
                "accepted_generated_token_ids": list(state.output_token_ids),
                "next_token_position": len(state.output_token_ids),
                "active_workers": [item.worker_id for item in plan.assignments],
                "stage_assignments": list(plan.assignments),
                "recovery_count": state.recovery_count,
                "last_healthy_checkpoint": len(state.output_token_ids),
                "status": status or state.status,
                "last_error": last_error,
                "updated_unix_ns": time.time_ns(),
            },
            deep=True,
        )
        self.state.save_request(updated)
        state.durable = updated

    def _accept_token(
        self,
        state: _ProductRequestState,
        publication: ProductTokenPublication,
    ) -> None:
        plan = state.plan
        durable = state.durable
        assert plan is not None and durable is not None
        position = len(state.output_token_ids)
        if publication.token_position != position:
            raise IntegrityError(
                f"token acceptance position {publication.token_position} is not {position}"
            )
        prospective_tokens = [*state.output_token_ids, publication.token_id]
        prospective = durable.model_copy(
            update={
                "accepted_generated_token_ids": prospective_tokens,
                "next_token_position": position + 1,
                "last_healthy_checkpoint": position + 1,
                "status": ProductRequestPhase.RUNNING,
                "updated_unix_ns": time.time_ns(),
            },
            deep=True,
        )
        try:
            state.stream.publish(
                StreamEventType.TOKEN_GENERATED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                token_position=position,
                token_id=publication.token_id,
                decoded_text_fragment=publication.decoded_text_fragment,
                status_detail="greedy token accepted after route and publication verification",
                expert_trace=publication.expert_trace,
                expert_metrics=publication.expert_metrics,
            )
        except BackpressureError:
            state.cancellation_requested = True
            state.cancellation_reason = "bounded client event queue exhausted"
            raise
        self.state.append_replay_token(
            request_id=state.submission.request_id,
            request_generation=state.request_generation,
            route_generation=plan.generation,
            token_position=position,
            token_id=publication.token_id,
        )
        self.state.save_request(prospective)
        state.output_token_ids.append(publication.token_id)
        state.durable = prospective
        accepted_s = time.perf_counter()
        state.token_accepted_times_s.append(accepted_s)
        if state.first_token_s is None:
            state.first_token_s = accepted_s
        self.telemetry.emit(
            "token_accepted",
            request_id=state.submission.request_id,
            request_generation=state.request_generation,
            session_id=state.session_id,
            topology_id=plan.topology_id,
            route_generation=plan.generation,
            token_position=position,
            token_id=publication.token_id,
            remote_expert_contributions=sum(
                item.get("event")
                in {
                    "remote_whole_expert_result_consumed",
                    "remote_microshard_result_consumed",
                }
                for item in publication.expert_trace
            ),
            expert_bytes=int(publication.expert_metrics.get("bytes_transferred", 0)),
        )

    async def _detect_route_failures(
        self,
        state: _ProductRequestState,
    ) -> tuple[set[str], bool]:
        plan = state.plan
        assert plan is not None
        expired = set(self.deployments.registry.expire())

        async def probe(item: PlanWorkerAssignment) -> tuple[str | None, bool]:
            healthy, _ = self.deployments.registry.registration_health(item.worker_id)
            if item.worker_id in expired or not healthy:
                return item.worker_id, False
            try:
                response = await asyncio.wait_for(
                    self.transport.get_stage_status(
                        item.control_endpoint,
                        GetStageStatusRequest(
                            worker_id=item.worker_id,
                            request_id=(
                                f"{state.submission.request_id}:health:{plan.generation}:"
                                f"{item.stage_id}:{uuid4().hex}"
                            ),
                            topology_id=plan.topology_id,
                            deadline_unix_ns=(
                                time.time_ns()
                                + int(self.deployments.control_timeout_s * 1_000_000_000)
                            ),
                        ),
                    ),
                    timeout=self.deployments.control_timeout_s,
                )
            except Exception:
                return item.worker_id, False
            route = response.installed_route
            if route is None or route.route_generation != plan.generation:
                return None, True
            return None, False

        results = await asyncio.gather(*(probe(item) for item in plan.assignments))
        failed = {worker_id for worker_id, _ in results if worker_id is not None}
        return failed, any(mismatch for _, mismatch in results)

    async def _ensure_route_healthy(self, state: _ProductRequestState) -> None:
        failed, route_mismatch = await self._detect_route_failures(state)
        if failed or route_mismatch:
            raise TransportError(
                "active stage route failed health checks: "
                + (", ".join(sorted(failed)) if failed else "route-generation mismatch")
            )

    async def _recover(self, state: _ProductRequestState, failure: BaseException) -> None:
        if state.cancellation_requested:
            raise asyncio.CancelledError
        if state.recovery_count >= self.maximum_recovery_attempts:
            raise TransportError(
                f"maximum recovery attempts reached after {type(failure).__name__}: {failure}"
            ) from failure
        old_plan = state.plan
        assert old_plan is not None
        state.recovery_count += 1
        state.status = ProductRequestPhase.RECOVERING
        self._update_durable(
            state,
            status=ProductRequestPhase.RECOVERING,
            last_error=f"{type(failure).__name__}: {failure}",
        )
        self.telemetry.emit(
            "recovery_started",
            request_id=state.submission.request_id,
            request_generation=state.request_generation,
            topology_id=old_plan.topology_id,
            old_route_generation=old_plan.generation,
            recovery_count=state.recovery_count,
            accepted_token_count=len(state.output_token_ids),
            failure=f"{type(failure).__name__}: {failure}",
        )
        state.stream.publish(
            StreamEventType.RECOVERY_STARTED,
            session_id=state.session_id,
            topology_id=old_plan.topology_id,
            model_revision=old_plan.model.model_revision,
            status_detail=(
                f"restart-and-replay recovery {state.recovery_count} started after "
                f"{type(failure).__name__}"
            ),
        )
        try:
            failed_worker_ids, _ = await self._detect_route_failures(state)
            failure_detail = str(failure)
            failed_worker_ids.update(
                item.worker_id
                for item in old_plan.assignments
                if item.data_endpoint in failure_detail
            )
            if "token publication" in failure_detail.lower():
                failed_worker_ids.add(old_plan.assignments[0].worker_id)
            for worker_id in sorted(failed_worker_ids):
                self.deployments.registry.mark_unhealthy(worker_id)
                self.telemetry.emit(
                    "worker_unhealthy",
                    worker_id=worker_id,
                    topology_id=old_plan.topology_id,
                    route_generation=old_plan.generation,
                    reason=f"{type(failure).__name__}: {failure}",
                )
            state.released_kv_bytes += await self._bounded_close_sessions(state, cancel=True)
            new_plan = await asyncio.wait_for(
                self.deployments.recover(failed_worker_ids=failed_worker_ids),
                timeout=self.recovery_timeout_s,
            )
            state.plan = new_plan
            state.request_generation += 1
            state.session_id = f"session-{uuid4().hex}"
            state.pending_publications.clear()
            state.replay_publications.clear()
            state.publication_event.clear()
            state.error = None
            await self._open_sessions(state)
            self.telemetry.emit(
                "session_opened",
                request_id=state.submission.request_id,
                request_generation=state.request_generation,
                session_id=state.session_id,
                topology_id=new_plan.topology_id,
                route_generation=new_plan.generation,
                recovery=True,
            )
            self._update_durable(state, status=ProductRequestPhase.RECOVERING)
            state.replaying = True
            replay_last_token: int | None = None
            for position, expected_token in enumerate(state.output_token_ids):
                if state.cancellation_requested:
                    raise asyncio.CancelledError
                replayed_token = await self._execute_token_step(
                    state,
                    output_position=position,
                    last_token=replay_last_token,
                    replay_only=True,
                )
                publication = await self._wait_for_publication(
                    state,
                    position,
                    replay_only=True,
                )
                if replayed_token != expected_token or publication.token_id != expected_token:
                    raise IntegrityError(
                        f"replay divergence at token {position}: expected {expected_token}, "
                        f"ring={replayed_token}, publication={publication.token_id}"
                    )
                replay_last_token = expected_token
                self.telemetry.emit(
                    "replay_token_verified",
                    request_id=state.submission.request_id,
                    request_generation=state.request_generation,
                    topology_id=new_plan.topology_id,
                    route_generation=new_plan.generation,
                    token_position=position,
                    token_id=expected_token,
                )
            state.replaying = False
            state.status = ProductRequestPhase.RUNNING
            self._update_durable(state, status=ProductRequestPhase.RUNNING)
            self.telemetry.emit(
                "recovery_completed",
                request_id=state.submission.request_id,
                request_generation=state.request_generation,
                topology_id=new_plan.topology_id,
                route_generation=new_plan.generation,
                recovery_count=state.recovery_count,
                verified_token_count=len(state.output_token_ids),
            )
            state.stream.publish(
                StreamEventType.RECOVERY_COMPLETED,
                session_id=state.session_id,
                topology_id=new_plan.topology_id,
                model_revision=new_plan.model.model_revision,
                status_detail=(
                    f"recovery {state.recovery_count} verified "
                    f"{len(state.output_token_ids)} accepted tokens"
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state.replaying = False
            self.telemetry.emit(
                "recovery_failed",
                request_id=state.submission.request_id,
                request_generation=state.request_generation,
                topology_id=state.plan.topology_id if state.plan else old_plan.topology_id,
                route_generation=(
                    state.plan.generation if state.plan is not None else old_plan.generation
                ),
                recovery_count=state.recovery_count,
                error=f"{type(exc).__name__}: {exc}",
            )
            state.stream.publish(
                StreamEventType.RECOVERY_FAILED,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else old_plan.topology_id,
                model_revision=(
                    state.plan.model.model_revision
                    if state.plan is not None
                    else old_plan.model.model_revision
                ),
                status_detail=f"recovery failed safely: {type(exc).__name__}: {exc}",
            )
            raise

    async def cancel(
        self,
        request_id: str,
        *,
        reason: str = "client cancelled",
    ) -> CancelProductResponse:
        state = self._active.get(request_id)
        if state is None:
            terminal = self._terminal.get(request_id)
            if terminal is None:
                return CancelProductResponse(
                    request_id=request_id,
                    accepted=False,
                    idempotent=True,
                    status=ProductRequestPhase.FAILED,
                    detail="unknown request",
                )
            cancelled = terminal.status == ProductRequestPhase.CANCELLED
            return CancelProductResponse(
                request_id=request_id,
                accepted=cancelled,
                idempotent=True,
                status=terminal.status,
                detail=(
                    "request is already cancelled"
                    if cancelled
                    else f"request is already {terminal.status.value}"
                ),
            )
        if state.cancellation_requested:
            return CancelProductResponse(
                request_id=request_id,
                accepted=True,
                idempotent=True,
                status=ProductRequestPhase.CANCELLED,
                released_kv_bytes=state.released_kv_bytes,
                detail="request cancellation is already in progress",
            )
        state.cancellation_requested = True
        state.cancellation_reason = reason
        state.publication_event.set()
        if state.task is not None:
            state.task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(state.task),
                    timeout=self.cleanup_timeout_s,
                )
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                pass
        return CancelProductResponse(
            request_id=request_id,
            accepted=True,
            status=ProductRequestPhase.CANCELLED,
            released_kv_bytes=state.released_kv_bytes,
            detail=reason,
        )

    async def disconnect(self, request_id: str) -> None:
        state = self._active.get(request_id)
        if state is not None:
            state.disconnected = True
        await self.cancel(request_id, reason="client disconnected; session cancelled")

    async def statuses(self, *, include_terminal: bool) -> list[ProductSessionStatus]:
        values: list[ProductSessionStatus] = []
        for state in list(self._active.values()):
            durable = state.durable
            plan = state.plan
            if durable is None or plan is None:
                continue
            kv_bytes = 0
            queue_depth = 0
            try:
                worker_statuses = await asyncio.gather(
                    *(
                        self.transport.get_stage_status(
                            item.control_endpoint,
                            GetStageStatusRequest(
                                worker_id=item.worker_id,
                                request_id=(
                                    f"status:{state.submission.request_id}:{item.stage_id}:"
                                    f"{uuid4().hex}"
                                ),
                                topology_id=plan.topology_id,
                            ),
                        )
                        for item in plan.assignments
                    ),
                    return_exceptions=True,
                )
                for response in worker_statuses:
                    if isinstance(response, BaseException):
                        continue
                    queue_depth += response.execution_queue_depth + response.token_queue_depth
                    kv_bytes += sum(
                        session.kv_cache_bytes
                        for session in response.sessions
                        if session.session_id == state.session_id
                    )
            except Exception:
                pass
            inter_token = None
            if len(state.token_accepted_times_s) > 1:
                intervals = [
                    right - left
                    for left, right in zip(
                        state.token_accepted_times_s,
                        state.token_accepted_times_s[1:],
                        strict=True,
                    )
                ]
                inter_token = sum(intervals) / len(intervals)
            values.append(
                ProductSessionStatus(
                    request_id=state.submission.request_id,
                    request_generation=state.request_generation,
                    session_id=state.session_id,
                    model_id=plan.model.model_id,
                    model_revision=plan.model.model_revision,
                    tokenizer_revision=plan.model.tokenizer_revision,
                    topology_id=plan.topology_id,
                    route_generation=plan.generation,
                    status=state.status,
                    token_position=len(state.output_token_ids),
                    accepted_token_ids=list(state.output_token_ids),
                    active_workers=[item.worker_id for item in plan.assignments],
                    kv_cache_bytes=kv_bytes,
                    queue_depth=queue_depth,
                    recovery_count=state.recovery_count,
                    last_healthy_checkpoint=len(state.output_token_ids),
                    time_to_first_token_s=(
                        state.first_token_s - state.started_s
                        if state.first_token_s is not None
                        else None
                    ),
                    inter_token_latency_s=inter_token,
                    last_error=state.error,
                )
            )
        if include_terminal:
            active_ids = {item.request_id for item in values}
            for durable in self._terminal.values():
                if durable.request_id in active_ids:
                    continue
                values.append(
                    ProductSessionStatus(
                        request_id=durable.request_id,
                        request_generation=durable.request_generation,
                        session_id=durable.session_id,
                        model_id=durable.model_id,
                        model_revision=durable.model_revision,
                        tokenizer_revision=durable.tokenizer_revision,
                        topology_id=durable.topology_id,
                        route_generation=durable.route_generation,
                        status=durable.status,
                        token_position=durable.next_token_position,
                        accepted_token_ids=list(durable.accepted_generated_token_ids),
                        active_workers=list(durable.active_workers),
                        recovery_count=durable.recovery_count,
                        last_healthy_checkpoint=durable.last_healthy_checkpoint,
                        last_error=durable.last_error,
                    )
                )
        return sorted(values, key=lambda item: item.request_id)

    @property
    def generated_token_count(self) -> int:
        terminal_tokens = sum(
            len(item.accepted_generated_token_ids)
            for request_id, item in self._terminal.items()
            if request_id not in self._active
        )
        active_tokens = sum(len(item.output_token_ids) for item in self._active.values())
        return terminal_tokens + active_tokens

    @property
    def recovery_count(self) -> int:
        terminal = sum(
            item.recovery_count
            for request_id, item in self._terminal.items()
            if request_id not in self._active
        )
        active = sum(item.recovery_count for item in self._active.values())
        return terminal + active

    @property
    def queue_depth(self) -> int:
        return sum(item.stream.qsize for item in self._active.values())

    async def close(self) -> None:
        self._shutting_down = True
        await asyncio.gather(
            *(
                self.cancel(request_id, reason="coordinator shutting down")
                for request_id in list(self._active)
            ),
            return_exceptions=True,
        )
        await self.data_pool.close()


__all__ = ["ProductSessionController", "SessionControlTransport"]
