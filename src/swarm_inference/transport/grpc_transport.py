"""gRPC AsyncIO transport using protobuf envelopes and chunked activation streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeVar

import grpc

from swarm_inference.config.models import StrictModel, SyntheticModelConfig
from swarm_inference.exceptions import TransportError
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    Ack,
    ActivationRequest,
    ActivationResult,
    CancelRequest,
    HealthResponse,
    StageAssignmentMessage,
    WireChunk,
    parse_message,
    serialize_message,
)
from swarm_inference.worker.agent import WorkerAgent

ResponseT = TypeVar("ResponseT", bound=StrictModel)


def _wire_chunks(
    serialized: bytes,
    *,
    message_id: str,
    maximum_message_bytes: int,
) -> list[WireChunk]:
    payload_size = max(256, (maximum_message_bytes - 2048) * 3 // 4)
    parts = [
        serialized[offset : offset + payload_size]
        for offset in range(0, len(serialized), payload_size)
    ] or [b""]
    return [
        WireChunk(
            message_id=message_id,
            chunk_index=index,
            chunk_count=len(parts),
            total_length=len(serialized),
            payload=part,
            checksum=sha256_bytes(part),
        )
        for index, part in enumerate(parts)
    ]


def _reassemble_wire_chunks(chunks: list[WireChunk]) -> bytes:
    if not chunks:
        raise TransportError("empty activation stream")
    ordered = sorted(chunks, key=lambda item: item.chunk_index)
    first = ordered[0]
    if [item.chunk_index for item in ordered] != list(range(first.chunk_count)):
        raise TransportError("activation stream chunk sequence is incomplete")
    if any(
        item.message_id != first.message_id
        or item.chunk_count != first.chunk_count
        or item.total_length != first.total_length
        or sha256_bytes(item.payload) != item.checksum
        for item in ordered
    ):
        raise TransportError("activation stream chunk metadata or checksum mismatch")
    payload = b"".join(item.payload for item in ordered)
    if len(payload) != first.total_length:
        raise TransportError("activation stream length mismatch")
    return payload


class GrpcTransport:
    """Reusable client transport; a future QUIC client can implement the same protocol."""

    def __init__(
        self,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        timeout_s: float = 120.0,
    ) -> None:
        if maximum_message_bytes <= 1024:
            raise ValueError("maximum_message_bytes must exceed 1024")
        self.maximum_message_bytes = maximum_message_bytes
        self.timeout_s = timeout_s
        self._channels: dict[str, grpc.aio.Channel] = {}

    def _channel(self, endpoint: str) -> grpc.aio.Channel:
        channel = self._channels.get(endpoint)
        if channel is None:
            options = [
                ("grpc.max_send_message_length", self.maximum_message_bytes),
                ("grpc.max_receive_message_length", self.maximum_message_bytes),
            ]
            channel = grpc.aio.insecure_channel(endpoint, options=options)
            self._channels[endpoint] = channel
        return channel

    async def _unary(
        self,
        endpoint: str,
        path: str,
        request: StrictModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        call = self._channel(endpoint).unary_unary(
            path,
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            response = await call(serialize_message(request), timeout=self.timeout_s)
            return parse_message(response, response_type)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"gRPC {path} to {endpoint} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def assign(self, endpoint: str, assignment: StageAssignmentMessage) -> Ack:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/Assign",
            assignment,
            Ack,
        )

    async def execute(
        self,
        endpoint: str,
        request: ActivationRequest,
    ) -> ActivationResult:
        serialized = serialize_message(request)
        # The response may be much larger than the input (for example, a final
        # vocabulary-logit tensor), so input size alone cannot safely select a
        # unary RPC. The streaming method uses one chunk for small messages and
        # automatically chunks either direction when necessary.
        request_chunks = _wire_chunks(
            serialized,
            message_id=request.metadata.tensor_id,
            maximum_message_bytes=self.maximum_message_bytes,
        )

        async def chunks() -> AsyncIterator[bytes]:
            for chunk in request_chunks:
                yield serialize_message(chunk)

        call = self._channel(endpoint).stream_stream(
            "/swarm.v1.Worker/ExecuteStream",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            responses = call(chunks(), timeout=self.timeout_s)
            response_chunks = [parse_message(response, WireChunk) async for response in responses]
            return parse_message(_reassemble_wire_chunks(response_chunks), ActivationResult)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"chunked gRPC execute to {endpoint} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def cancel(self, endpoint: str, request: CancelRequest) -> Ack:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/Cancel",
            request,
            Ack,
        )

    async def health(self, endpoint: str) -> HealthResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/Health",
            Ack(accepted=True, detail="probe"),
            HealthResponse,
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(channel.close() for channel in self._channels.values()),
            return_exceptions=True,
        )
        self._channels.clear()


class WorkerRpcServer:
    """Expose a WorkerAgent through gRPC generic protobuf handlers."""

    def __init__(
        self,
        *,
        agent: WorkerAgent,
        synthetic_config: SyntheticModelConfig | None = None,
        model_shard_root: str | None = None,
        maximum_message_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.agent = agent
        self.synthetic_config = synthetic_config
        self.model_shard_root = model_shard_root
        self.maximum_message_bytes = maximum_message_bytes
        self.server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ]
        )
        handlers = {
            "Assign": grpc.unary_unary_rpc_method_handler(
                self._assign,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Execute": grpc.unary_unary_rpc_method_handler(
                self._execute,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "ExecuteStream": grpc.stream_stream_rpc_method_handler(
                self._execute_stream,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Cancel": grpc.unary_unary_rpc_method_handler(
                self._cancel,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                self._health,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
        }
        self.server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler("swarm.v1.Worker", handlers),)
        )
        self.bound_port: int | None = None

    async def start(self, endpoint: str) -> int:
        await self.agent.start()
        self.bound_port = self.server.add_insecure_port(endpoint)
        if self.bound_port == 0:
            raise TransportError(f"could not bind worker gRPC endpoint {endpoint}")
        await self.server.start()
        return self.bound_port

    async def stop(self, grace_s: float = 2.0) -> None:
        await self.server.stop(grace_s)
        await self.agent.stop()

    async def wait_for_termination(self) -> None:
        await self.server.wait_for_termination()

    async def _assign(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            assignment = parse_message(data, StageAssignmentMessage)
            if assignment.worker_id != self.agent.capability.worker_id:
                raise TransportError("assignment worker ID does not match this worker")
            synthetic_config = assignment.synthetic_model or self.synthetic_config
            if synthetic_config is not None:
                self.agent.assign_synthetic(
                    config=synthetic_config,
                    stage=assignment.stage,
                )
            elif (
                assignment.architecture_config is not None and assignment.model_manifest is not None
            ):
                shard_path = assignment.shard_path
                if self.model_shard_root is not None:
                    from pathlib import Path

                    shard_path = str(
                        Path(self.model_shard_root).expanduser().resolve() / shard_path
                    )
                self.agent.assign_qwen3(
                    config=assignment.architecture_config,
                    manifest=assignment.model_manifest,
                    stage=assignment.stage,
                    shard_path=shard_path,
                    shard_hash=assignment.shard_hash,
                    dtype=assignment.dtype,
                )
            else:
                raise TransportError(
                    "assignment supplies neither synthetic config nor Qwen3 config/manifest"
                )
            return serialize_message(Ack(accepted=True, detail="stage loaded"))
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _execute(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            request = parse_message(data, ActivationRequest)
            result = await self.agent.execute(request)
            return serialize_message(result)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            raise

    async def _execute_stream(
        self,
        iterator: AsyncIterator[bytes],
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[bytes]:
        try:
            chunks: list[WireChunk] = []
            async for raw in iterator:
                chunks.append(parse_message(raw, WireChunk))
            payload = _reassemble_wire_chunks(chunks)
            request = parse_message(payload, ActivationRequest)
            result = await self.agent.execute(request)
            for chunk in _wire_chunks(
                serialize_message(result),
                message_id=result.metadata.tensor_id,
                maximum_message_bytes=self.maximum_message_bytes,
            ):
                yield serialize_message(chunk)
        except Exception as exc:
            await context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
            raise

    async def _cancel(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        request = parse_message(data, CancelRequest)
        self.agent.cancel(request.request_id)
        return serialize_message(Ack(accepted=True, detail="request state deleted"))

    async def _health(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        parse_message(data, Ack)
        return serialize_message(self.agent.health())
