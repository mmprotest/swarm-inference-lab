"""Product session lifecycle and token streaming over a deployed stage ring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

import torch

from swarm_inference.coordinator.deployment import DeploymentManager
from swarm_inference.coordinator.event_stream import BoundedRequestEventStream
from swarm_inference.exceptions import BackpressureError, IntegrityError, TransportError
from swarm_inference.protocol.messages import (
    Ack,
    StreamEventType,
    SubmitRequest,
)
from swarm_inference.protocol.product import ProductStagePlan, ProductTokenPublication
from swarm_inference.protocol.stage_ring import Operation, StageMessage
from swarm_inference.protocol.stage_worker import (
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    OpenStageSessionRequest,
    StageActionResponse,
    TokenizeStageRequest,
    TokenizeStageResponse,
)
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


@dataclass(slots=True)
class _ProductRequestState:
    submission: SubmitRequest
    stream: BoundedRequestEventStream
    session_id: str
    plan: ProductStagePlan | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    opened_stage_ids: set[int] = field(default_factory=set)
    publication_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancellation_requested: bool = False
    disconnected: bool = False
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
    ) -> None:
        self.deployments = deployments
        self.transport = transport
        self.event_queue_capacity = event_queue_capacity
        self.request_timeout_s = request_timeout_s
        self.data_pool = StageRingConnectionPool(
            queue_capacity=data_queue_capacity,
            read_timeout_s=request_timeout_s,
            write_timeout_s=min(30.0, request_timeout_s),
        )
        self._active: dict[str, _ProductRequestState] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def start(self, submission: SubmitRequest) -> BoundedRequestEventStream:
        if submission.request_id in self._active:
            raise ValueError(f"duplicate active request ID {submission.request_id}")
        stream = BoundedRequestEventStream(
            request_id=submission.request_id,
            capacity=self.event_queue_capacity,
        )
        state = _ProductRequestState(
            submission=submission,
            stream=stream,
            session_id=f"session-{uuid4().hex}",
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
            await self._open_sessions(state)
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
                response_token = await self._execute_token_step(
                    state,
                    output_position=output_position,
                    last_token=last_token,
                )
                published_token = await self._wait_for_publication(state, output_position)
                if published_token != response_token:
                    raise IntegrityError(
                        f"stage-zero publication token {published_token} differs from direct "
                        f"ring response {response_token} at position {output_position}"
                    )
                last_token = response_token
            await self._close_sessions(state, cancel=False)
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
            await self._close_sessions(state, cancel=True)
            state.stream.fail(
                "client disconnected; session cancelled",
                cancelled=True,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else None,
                model_revision=state.plan.model.model_revision if state.plan else None,
            )
        except Exception as exc:
            state.error = f"{type(exc).__name__}: {exc}"
            await self._close_sessions(state, cancel=True)
            state.stream.fail(
                state.error,
                session_id=state.session_id,
                topology_id=state.plan.topology_id if state.plan else None,
                model_revision=state.plan.model.model_revision if state.plan else None,
            )
        finally:
            state.publication_event.set()
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

    async def _close_sessions(self, state: _ProductRequestState, *, cancel: bool) -> None:
        plan = state.plan
        if plan is None or not state.opened_stage_ids:
            return
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
        for stage_id, result in zip(stage_ids, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"stage {stage_id}: {type(result).__name__}: {result}")
            elif not result.accepted:
                errors.append(f"stage {stage_id}: {result.detail}")
            else:
                state.opened_stage_ids.discard(stage_id)
        if errors and not cancel:
            raise RuntimeError("session close failed: " + "; ".join(errors))

    async def _execute_token_step(
        self,
        state: _ProductRequestState,
        *,
        output_position: int,
        last_token: int | None,
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
                "source_worker_id": "coordinator",
                "destination_worker_id": stage_zero.worker_id,
                "cache_position_start": cache_position,
                "tensor": packed.attributes(),
            },
        )
        response = await self.data_pool.send(stage_zero.data_endpoint, message)
        if response.operation == Operation.ERROR:
            raise TransportError(str(response.attributes.get("error", "stage ring failed")))
        if (
            response.operation != Operation.TOKEN_RESULT
            or response.status != "OK"
            or response.request_id != state.submission.request_id
            or response.session_id != state.session_id
            or response.topology_id != plan.topology_id
            or response.model_revision != plan.model.model_revision
            or response.token_position != output_position
            or response.destination_stage != -1
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
    ) -> int:
        deadline = asyncio.get_running_loop().time() + self.request_timeout_s
        while len(state.output_token_ids) <= position:
            if state.error is not None:
                raise RuntimeError(state.error)
            if state.cancellation_requested:
                raise asyncio.CancelledError
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"token publication {position} timed out")
            state.publication_event.clear()
            if len(state.output_token_ids) > position:
                break
            await asyncio.wait_for(state.publication_event.wait(), timeout=remaining)
        return state.output_token_ids[position]

    async def publish_token(self, publication: ProductTokenPublication) -> Ack:
        state = self._active.get(publication.request_id)
        if state is None:
            return Ack(accepted=False, detail="unknown or closed product request")
        plan = state.plan
        if plan is None:
            return Ack(accepted=False, detail="request has no selected topology")
        stage_zero = plan.assignments[0]
        if (
            publication.worker_id != stage_zero.worker_id
            or publication.session_id != state.session_id
            or publication.topology_id != plan.topology_id
            or publication.route_generation != plan.generation
            or publication.model_revision != plan.model.model_revision
        ):
            state.error = "token publication identity does not match the active session"
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        position = publication.token_position
        if position < len(state.output_token_ids):
            if state.output_token_ids[position] == publication.token_id:
                return Ack(accepted=True, detail="duplicate token publication ignored")
            state.error = "conflicting duplicate token publication"
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        if position != len(state.output_token_ids):
            state.error = (
                f"out-of-order token publication {position}; expected {len(state.output_token_ids)}"
            )
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        try:
            state.stream.publish(
                StreamEventType.TOKEN_GENERATED,
                session_id=state.session_id,
                topology_id=plan.topology_id,
                model_revision=plan.model.model_revision,
                token_position=position,
                token_id=publication.token_id,
                decoded_text_fragment=publication.decoded_text_fragment,
                status_detail="token published by stage zero",
            )
        except BackpressureError as exc:
            state.error = str(exc)
            state.cancellation_requested = True
            state.publication_event.set()
            return Ack(accepted=False, detail=state.error)
        state.output_token_ids.append(publication.token_id)
        if state.first_token_s is None:
            state.first_token_s = time.perf_counter()
        state.publication_event.set()
        return Ack(accepted=True, detail="token publication accepted")

    async def disconnect(self, request_id: str) -> None:
        state = self._active.get(request_id)
        if state is None:
            return
        state.disconnected = True
        state.cancellation_requested = True
        state.publication_event.set()
        if state.task is not None:
            state.task.cancel()
            await asyncio.gather(state.task, return_exceptions=True)

    async def close(self) -> None:
        await asyncio.gather(
            *(self.disconnect(request_id) for request_id in list(self._active)),
            return_exceptions=True,
        )
        await self.data_pool.close()


__all__ = ["ProductSessionController", "SessionControlTransport"]
