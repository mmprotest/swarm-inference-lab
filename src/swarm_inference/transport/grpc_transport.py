"""gRPC AsyncIO transport using protobuf envelopes and chunked activation streams."""

from __future__ import annotations

import asyncio
import hmac
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import grpc

from swarm_inference.config.models import StrictModel, SyntheticModelConfig
from swarm_inference.exceptions import MemoryLimitExceededError, TransportError
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.engine_worker import (
    EngineActionResponse,
    EngineInferenceChunk,
    EngineInferenceResponse,
    PrepareEngineRequest,
    SubmitEngineRequest,
    UnloadEngineRequest,
)
from swarm_inference.protocol.messages import (
    Ack,
    ActivationRequest,
    ActivationResult,
    CacheControlRequest,
    CacheControlResponse,
    CancelRequest,
    DataPlaneAck,
    DataPlaneEnvelope,
    HealthResponse,
    RouteInstallRequest,
    StageAssignmentMessage,
    WireChunk,
    parse_message,
    serialize_message,
)
from swarm_inference.protocol.product import (
    WorkerModelProbeRequest,
    WorkerModelProbeResponse,
)
from swarm_inference.protocol.stage_worker import (
    ArtifactTransferLease,
    ArtifactTransferResponse,
    CancelStageSessionRequest,
    CloseStageSessionRequest,
    CompleteArtifactRequest,
    DrainWorkerRequest,
    GetStageCapabilitiesRequest,
    GetStageCapabilitiesResponse,
    GetStageStatusRequest,
    InstallStageRouteRequest,
    LoadStageRequest,
    OpenStageSessionRequest,
    PrepareArtifactRequest,
    RemoveStageRouteRequest,
    StageActionResponse,
    StageStatusResponse,
    TokenizeStageRequest,
    TokenizeStageResponse,
    UnloadStageRequest,
    VerifyArtifactRequest,
    VerifyStageRouteRequest,
    WriteArtifactChunkRequest,
    verify_artifact_transfer_lease,
)
from swarm_inference.security.identity import public_key_fingerprint
from swarm_inference.worker.agent import WorkerAgent

if TYPE_CHECKING:
    from swarm_inference.cluster.artifacts import ArtifactManager
    from swarm_inference.worker.engine_runtime import PersistentEngineRuntime
    from swarm_inference.worker.stage_runtime import PersistentStageRuntime

ResponseT = TypeVar("ResponseT", bound=StrictModel)


@dataclass(slots=True)
class GrpcTransportMetrics:
    channels_created: int = 0
    streams_created: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    payload_bytes_sent: int = 0
    payload_bytes_received: int = 0
    control_bytes: int = 0
    serialisation_time_ms: float = 0.0
    deserialisation_time_ms: float = 0.0
    call_time_ms: float = 0.0

    def snapshot(self) -> dict[str, int | float]:
        return asdict(self)


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
        self._closed = False
        self.metrics = GrpcTransportMetrics()

    def _channel(self, endpoint: str) -> grpc.aio.Channel:
        if self._closed:
            raise TransportError("gRPC transport is closed")
        channel = self._channels.get(endpoint)
        if channel is None:
            options = [
                ("grpc.max_send_message_length", self.maximum_message_bytes),
                ("grpc.max_receive_message_length", self.maximum_message_bytes),
            ]
            channel = grpc.aio.insecure_channel(endpoint, options=options)
            self._channels[endpoint] = channel
            self.metrics.channels_created += 1
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
            serialization_started = time.perf_counter_ns()
            serialized = serialize_message(request)
            self.metrics.serialisation_time_ms += (
                time.perf_counter_ns() - serialization_started
            ) / 1_000_000
            self.metrics.messages_sent += 1
            self.metrics.payload_bytes_sent += len(serialized)
            call_started = time.perf_counter_ns()
            response = await call(serialized, timeout=self.timeout_s)
            self.metrics.call_time_ms += (time.perf_counter_ns() - call_started) / 1_000_000
            deserialization_started = time.perf_counter_ns()
            parsed = parse_message(response, response_type)
            self.metrics.deserialisation_time_ms += (
                time.perf_counter_ns() - deserialization_started
            ) / 1_000_000
            self.metrics.messages_received += 1
            self.metrics.payload_bytes_received += len(response)
            return parsed
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

    async def install_route(self, endpoint: str, request: RouteInstallRequest) -> Ack:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/InstallRoute",
            request,
            Ack,
        )

    async def dispatch(
        self,
        endpoint: str,
        request: DataPlaneEnvelope,
    ) -> DataPlaneAck:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/DispatchRoute",
            request,
            DataPlaneAck,
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
        self.metrics.streams_created += 1
        try:
            responses = call(chunks(), timeout=self.timeout_s)
            response_chunks = [parse_message(response, WireChunk) async for response in responses]
            self.metrics.messages_sent += len(request_chunks)
            self.metrics.messages_received += len(response_chunks)
            self.metrics.payload_bytes_sent += len(serialized)
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

    async def cache_control(
        self,
        endpoint: str,
        request: CacheControlRequest,
    ) -> CacheControlResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/CacheControl",
            request,
            CacheControlResponse,
        )

    async def health(self, endpoint: str) -> HealthResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/Health",
            Ack(accepted=True, detail="probe"),
            HealthResponse,
        )

    async def get_stage_capabilities(
        self,
        endpoint: str,
        request: GetStageCapabilitiesRequest,
    ) -> GetStageCapabilitiesResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/GetStageCapabilities",
            request,
            GetStageCapabilitiesResponse,
        )

    async def inspect_stage_model(
        self,
        endpoint: str,
        request: WorkerModelProbeRequest,
    ) -> WorkerModelProbeResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/InspectStageModel",
            request,
            WorkerModelProbeResponse,
        )

    async def prepare_artifact(
        self,
        endpoint: str,
        request: PrepareArtifactRequest,
    ) -> ArtifactTransferResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/PrepareArtifact",
            request,
            ArtifactTransferResponse,
        )

    async def write_artifact_chunk(
        self,
        endpoint: str,
        request: WriteArtifactChunkRequest,
    ) -> ArtifactTransferResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/WriteArtifactChunk",
            request,
            ArtifactTransferResponse,
        )

    async def complete_artifact(
        self,
        endpoint: str,
        request: CompleteArtifactRequest,
    ) -> ArtifactTransferResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/CompleteArtifact",
            request,
            ArtifactTransferResponse,
        )

    async def verify_artifact(
        self,
        endpoint: str,
        request: VerifyArtifactRequest,
    ) -> ArtifactTransferResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/VerifyArtifact",
            request,
            ArtifactTransferResponse,
        )

    async def prepare_engine(
        self, endpoint: str, request: PrepareEngineRequest
    ) -> EngineActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/PrepareEngine",
            request,
            EngineActionResponse,
        )

    async def submit_engine(
        self, endpoint: str, request: SubmitEngineRequest
    ) -> EngineInferenceResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/SubmitEngine",
            request,
            EngineInferenceResponse,
        )

    async def submit_engine_stream(
        self, endpoint: str, request: SubmitEngineRequest
    ) -> AsyncIterator[EngineInferenceChunk]:
        call = self._channel(endpoint).unary_stream(
            "/swarm.v1.Worker/SubmitEngineStream",
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        serialization_started = time.perf_counter_ns()
        serialized = serialize_message(request)
        self.metrics.serialisation_time_ms += (
            time.perf_counter_ns() - serialization_started
        ) / 1_000_000
        self.metrics.messages_sent += 1
        self.metrics.payload_bytes_sent += len(serialized)
        self.metrics.streams_created += 1
        call_started = time.perf_counter_ns()
        try:
            responses = call(serialized, timeout=self.timeout_s)
            async for raw in responses:
                self.metrics.messages_received += 1
                self.metrics.payload_bytes_received += len(raw)
                deserialization_started = time.perf_counter_ns()
                chunk = parse_message(raw, EngineInferenceChunk)
                self.metrics.deserialisation_time_ms += (
                    time.perf_counter_ns() - deserialization_started
                ) / 1_000_000
                yield chunk
            self.metrics.call_time_ms += (
                time.perf_counter_ns() - call_started
            ) / 1_000_000
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                "streaming gRPC engine submission to "
                f"{endpoint} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def unload_engine(
        self, endpoint: str, request: UnloadEngineRequest
    ) -> EngineActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/UnloadEngine",
            request,
            EngineActionResponse,
        )

    async def load_stage(self, endpoint: str, request: LoadStageRequest) -> StageActionResponse:
        return await self._unary(
            endpoint, "/swarm.v1.Worker/LoadStage", request, StageActionResponse
        )

    async def unload_stage(self, endpoint: str, request: UnloadStageRequest) -> StageActionResponse:
        return await self._unary(
            endpoint, "/swarm.v1.Worker/UnloadStage", request, StageActionResponse
        )

    async def install_stage_route(
        self, endpoint: str, request: InstallStageRouteRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/InstallStageRoute",
            request,
            StageActionResponse,
        )

    async def remove_stage_route(
        self, endpoint: str, request: RemoveStageRouteRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/RemoveStageRoute",
            request,
            StageActionResponse,
        )

    async def verify_stage_route(
        self, endpoint: str, request: VerifyStageRouteRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/VerifyStageRoute",
            request,
            StageActionResponse,
        )

    async def open_stage_session(
        self, endpoint: str, request: OpenStageSessionRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/OpenStageSession",
            request,
            StageActionResponse,
        )

    async def close_stage_session(
        self, endpoint: str, request: CloseStageSessionRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/CloseStageSession",
            request,
            StageActionResponse,
        )

    async def cancel_stage_session(
        self, endpoint: str, request: CancelStageSessionRequest
    ) -> StageActionResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/CancelStageSession",
            request,
            StageActionResponse,
        )

    async def get_stage_status(
        self, endpoint: str, request: GetStageStatusRequest
    ) -> StageStatusResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/GetStageStatus",
            request,
            StageStatusResponse,
        )

    async def tokenize_stage(
        self, endpoint: str, request: TokenizeStageRequest
    ) -> TokenizeStageResponse:
        return await self._unary(
            endpoint,
            "/swarm.v1.Worker/TokenizeStage",
            request,
            TokenizeStageResponse,
        )

    async def drain_worker(self, endpoint: str, request: DrainWorkerRequest) -> StageActionResponse:
        return await self._unary(
            endpoint, "/swarm.v1.Worker/DrainWorker", request, StageActionResponse
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
        stage_runtime: PersistentStageRuntime | None = None,
        engine_runtime: PersistentEngineRuntime | None = None,
        artifact_manager: ArtifactManager | None = None,
        trusted_coordinator_fingerprint: str | None = None,
        maximum_active_artifact_transfers: int = 128,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        if shutdown_timeout_s <= 0:
            raise ValueError("worker gRPC shutdown timeout must be positive")
        if maximum_active_artifact_transfers <= 0:
            raise ValueError("active artifact transfer bound must be positive")
        self.agent = agent
        self.synthetic_config = synthetic_config
        self.model_shard_root = model_shard_root
        self.maximum_message_bytes = maximum_message_bytes
        self.stage_runtime = stage_runtime
        self.engine_runtime = engine_runtime
        self.artifact_manager = artifact_manager
        self.trusted_coordinator_fingerprint = trusted_coordinator_fingerprint
        self._artifact_coordinator_public_key: str | None = None
        self._artifact_authorizations: OrderedDict[str, ArtifactTransferLease] = OrderedDict()
        self._maximum_active_artifact_transfers = maximum_active_artifact_transfers
        self.shutdown_timeout_s = shutdown_timeout_s
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
            "InstallRoute": grpc.unary_unary_rpc_method_handler(
                self._install_route,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "DispatchRoute": grpc.unary_unary_rpc_method_handler(
                self._dispatch_route,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "PeerStream": grpc.stream_stream_rpc_method_handler(
                self._peer_stream,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Cancel": grpc.unary_unary_rpc_method_handler(
                self._cancel,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "CacheControl": grpc.unary_unary_rpc_method_handler(
                self._cache_control,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Health": grpc.unary_unary_rpc_method_handler(
                self._health,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "GetStageCapabilities": grpc.unary_unary_rpc_method_handler(
                self._get_stage_capabilities,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "InspectStageModel": grpc.unary_unary_rpc_method_handler(
                self._inspect_stage_model,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "PrepareArtifact": grpc.unary_unary_rpc_method_handler(
                self._prepare_artifact,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "WriteArtifactChunk": grpc.unary_unary_rpc_method_handler(
                self._write_artifact_chunk,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "CompleteArtifact": grpc.unary_unary_rpc_method_handler(
                self._complete_artifact,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "VerifyArtifact": grpc.unary_unary_rpc_method_handler(
                self._verify_artifact,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "PrepareEngine": grpc.unary_unary_rpc_method_handler(
                self._prepare_engine,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "SubmitEngine": grpc.unary_unary_rpc_method_handler(
                self._submit_engine,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "SubmitEngineStream": grpc.unary_stream_rpc_method_handler(
                self._submit_engine_stream,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "UnloadEngine": grpc.unary_unary_rpc_method_handler(
                self._unload_engine,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "LoadStage": grpc.unary_unary_rpc_method_handler(
                self._load_stage,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "UnloadStage": grpc.unary_unary_rpc_method_handler(
                self._unload_stage,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "InstallStageRoute": grpc.unary_unary_rpc_method_handler(
                self._install_stage_route,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "RemoveStageRoute": grpc.unary_unary_rpc_method_handler(
                self._remove_stage_route,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "VerifyStageRoute": grpc.unary_unary_rpc_method_handler(
                self._verify_stage_route,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "OpenStageSession": grpc.unary_unary_rpc_method_handler(
                self._open_stage_session,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "CloseStageSession": grpc.unary_unary_rpc_method_handler(
                self._close_stage_session,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "CancelStageSession": grpc.unary_unary_rpc_method_handler(
                self._cancel_stage_session,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "GetStageStatus": grpc.unary_unary_rpc_method_handler(
                self._get_stage_status,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "TokenizeStage": grpc.unary_unary_rpc_method_handler(
                self._tokenize_stage,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "DrainWorker": grpc.unary_unary_rpc_method_handler(
                self._drain_worker,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
        }
        self.server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler("swarm.v1.Worker", handlers),)
        )
        self.bound_port: int | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._stopping = False
        self._closed = False

    async def start(self, endpoint: str) -> int:
        async with self._lifecycle_lock:
            if self._started:
                raise RuntimeError("worker gRPC server is already started")
            if self._closed:
                raise RuntimeError("worker gRPC server cannot restart after shutdown")
            self._stopping = False
            try:
                await self.agent.start()
                if self.stage_runtime is not None:
                    await self.stage_runtime.start()
                self.bound_port = self.server.add_insecure_port(endpoint)
                if self.bound_port == 0:
                    raise TransportError(f"could not bind worker gRPC endpoint {endpoint}")
                await self.server.start()
            except BaseException:
                if self.stage_runtime is not None:
                    await self.stage_runtime.close()
                await self.agent.stop()
                self.bound_port = None
                self._closed = True
                raise
            self._started = True
            return self.bound_port

    async def stop(self, grace_s: float = 2.0) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                return
            self._stopping = True
            try:
                if self._started:
                    await asyncio.wait_for(
                        self.server.stop(grace_s),
                        timeout=max(self.shutdown_timeout_s, grace_s + 1.0),
                    )
            finally:
                try:
                    await self.agent.stop()
                finally:
                    try:
                        if self.stage_runtime is not None:
                            await self.stage_runtime.close()
                    finally:
                        if self.engine_runtime is not None:
                            await self.engine_runtime.close()
            self.bound_port = None
            self._started = False
            self._closed = True

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
            self.agent.configure_data_plane(assignment)
            return serialize_message(Ack(accepted=True, detail="stage loaded"))
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _install_route(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, RouteInstallRequest)
            if request.worker_id != self.agent.capability.worker_id:
                raise TransportError("route installation addressed to another worker")
            self.agent.install_route(request.route)
            return serialize_message(Ack(accepted=True, detail="route installed"))
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _dispatch_route(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            envelope = parse_message(data, DataPlaneEnvelope)
            ack = await self.agent.handle_data_plane(envelope)
            return serialize_message(ack)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            raise

    async def _peer_stream(
        self,
        iterator: AsyncIterator[bytes],
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[bytes]:
        responses: asyncio.Queue[bytes] = asyncio.Queue(maxsize=max(1, self.agent.inbound_capacity))
        inbound_slots = asyncio.Semaphore(self.agent.inbound_capacity)
        tasks: set[asyncio.Task[None]] = set()
        input_finished = asyncio.Event()

        async def process(raw: bytes) -> None:
            try:
                deserialization_started = time.perf_counter_ns()
                envelope = parse_message(raw, DataPlaneEnvelope)
                deserialization_ms = (time.perf_counter_ns() - deserialization_started) / 1_000_000
                transfer_ms = max(
                    0.0,
                    (time.time_ns() - envelope.timestamp_unix_ns) / 1_000_000,
                )
                self.agent.peer_pool.record_received(
                    payload_bytes=envelope.payload_length,
                    control_bytes=max(0, len(raw) - envelope.payload_length),
                    deserialisation_ms=deserialization_ms,
                    transfer_ms=transfer_ms,
                )
                ack = await self.agent.handle_data_plane(envelope)
            except Exception as exc:
                ack = DataPlaneAck(
                    message_id=(envelope.message_id if "envelope" in locals() else "unparseable"),
                    status="invalid_stage_transition",
                    detail=f"{type(exc).__name__}: {exc}",
                    accepted_timestamp_unix_ns=time.time_ns(),
                )
            await responses.put(serialize_message(ack))

        async def read_requests() -> None:
            def release_slot(completed: asyncio.Task[None]) -> None:
                tasks.discard(completed)
                inbound_slots.release()

            try:
                async for raw in iterator:
                    await inbound_slots.acquire()
                    task = asyncio.create_task(process(raw))
                    tasks.add(task)
                    task.add_done_callback(release_slot)
            finally:
                input_finished.set()

        reader = asyncio.create_task(read_requests())
        try:
            while not input_finished.is_set() or tasks or not responses.empty():
                try:
                    response = await asyncio.wait_for(responses.get(), timeout=0.1)
                except TimeoutError:
                    continue
                try:
                    yield response
                finally:
                    responses.task_done()
        except asyncio.CancelledError:
            if self._stopping:
                return
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
            raise
        finally:
            reader.cancel()
            await asyncio.gather(reader, *tasks, return_exceptions=True)

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

    async def _cache_control(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        request = parse_message(data, CacheControlRequest)
        if request.worker_id != self.agent.capability.worker_id:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "cache control addressed to another worker",
            )
        response = await self.agent.cache_control(request)
        return serialize_message(response)

    async def _health(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        parse_message(data, Ack)
        health = self.agent.health()
        if self.stage_runtime is not None:
            status = await self.stage_runtime.status(
                GetStageStatusRequest(
                    worker_id=self.agent.capability.worker_id,
                    request_id="health",
                )
            )
            proof = dict(health.proof)
            proof["stage_runtime"] = status.model_dump(mode="json")
            loaded_stages = set(health.loaded_stages)
            if status.loaded_stage is not None:
                loaded_stages.add(status.loaded_stage.assignment.stage_id)
            health = health.model_copy(
                update={
                    "loaded_stages": sorted(loaded_stages),
                    "proof": proof,
                }
            )
        return serialize_message(health)

    def configure_artifact_trust(
        self,
        *,
        coordinator_public_key: str,
        coordinator_fingerprint: str,
    ) -> None:
        """Pin the authenticated coordinator after worker registration."""

        actual = public_key_fingerprint(coordinator_public_key)
        if not hmac.compare_digest(actual, coordinator_fingerprint):
            raise TransportError("coordinator artifact public-key fingerprint mismatch")
        expected = self.trusted_coordinator_fingerprint
        if expected is not None and not hmac.compare_digest(expected, coordinator_fingerprint):
            raise TransportError("coordinator artifact identity differs from the membership pin")
        existing = self._artifact_coordinator_public_key
        if existing is not None and existing != coordinator_public_key:
            raise TransportError("coordinator artifact identity cannot change while running")
        self._artifact_coordinator_public_key = coordinator_public_key

    def _require_artifact_manager(self) -> ArtifactManager:
        if self.artifact_manager is None:
            raise RuntimeError("artifact management is disabled on this worker")
        return self.artifact_manager

    def _authorize_artifact(
        self,
        lease: ArtifactTransferLease,
        *,
        worker_id: str,
        artifact_id: str,
    ) -> None:
        if worker_id != self.agent.capability.worker_id:
            raise PermissionError("artifact operation is addressed to another worker")
        public_key = self._artifact_coordinator_public_key
        fingerprint = self.trusted_coordinator_fingerprint
        if public_key is None or fingerprint is None:
            raise PermissionError("artifact coordinator trust is not configured")
        verify_artifact_transfer_lease(
            lease,
            trusted_coordinator_public_key=public_key,
            trusted_coordinator_fingerprint=fingerprint,
            destination_worker_id=worker_id,
            artifact_id=artifact_id,
        )

    def _require_stage_runtime(self) -> PersistentStageRuntime:
        if self.stage_runtime is None:
            raise RuntimeError("persistent stage runtime is disabled on this worker")
        return self.stage_runtime

    async def _get_stage_capabilities(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, GetStageCapabilitiesRequest)
            response = await self._require_stage_runtime().get_capabilities(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _inspect_stage_model(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, WorkerModelProbeRequest)
            response = await self._require_stage_runtime().inspect_model(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _prepare_artifact(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, PrepareArtifactRequest)
            self._authorize_artifact(
                request.lease,
                worker_id=request.worker_id,
                artifact_id=request.manifest.artifact_id,
            )
            status = self._require_artifact_manager().prepare_incoming(
                manifest=request.manifest,
                source=request.lease.source_node_id,
                chunks_total=request.chunks_total,
            )
            if status.state != "complete":
                existing = self._artifact_authorizations.get(status.transfer_id)
                if existing is None:
                    if (
                        len(self._artifact_authorizations)
                        >= self._maximum_active_artifact_transfers
                    ):
                        raise RuntimeError("active artifact transfer authorization bound reached")
                    self._artifact_authorizations[status.transfer_id] = request.lease
                elif existing.nonce != request.lease.nonce:
                    raise PermissionError("artifact transfer authorization changed during resume")
            return serialize_message(
                ArtifactTransferResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    artifact_id=request.manifest.artifact_id,
                    accepted=True,
                    transfer_id=status.transfer_id,
                    bytes_completed=status.bytes_completed,
                    chunks_completed=status.chunks_completed,
                    complete=status.state == "complete",
                    verified=status.state == "complete",
                    detail=("artifact already verified" if status.state == "complete" else "ready"),
                )
            )
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _write_artifact_chunk(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, WriteArtifactChunkRequest)
            self._authorize_artifact(
                request.lease,
                worker_id=request.worker_id,
                artifact_id=request.chunk.artifact_id,
            )
            active = self._artifact_authorizations.get(request.transfer_id)
            if active is None or active.nonce != request.lease.nonce:
                raise PermissionError("artifact transfer is not authorized")
            manager = self._require_artifact_manager()
            manager.write_chunk(
                transfer_id=request.transfer_id,
                chunk=request.chunk,
                payload=request.payload,
            )
            status = next(
                item for item in manager.transfers() if item.transfer_id == request.transfer_id
            )
            return serialize_message(
                ArtifactTransferResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    artifact_id=request.chunk.artifact_id,
                    accepted=True,
                    transfer_id=request.transfer_id,
                    bytes_completed=status.bytes_completed,
                    chunks_completed=status.chunks_completed,
                    detail="chunk verified",
                )
            )
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
            raise

    async def _complete_artifact(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, CompleteArtifactRequest)
            self._authorize_artifact(
                request.lease,
                worker_id=request.worker_id,
                artifact_id=request.manifest.artifact_id,
            )
            active = self._artifact_authorizations.get(request.transfer_id)
            if active is None or active.nonce != request.lease.nonce:
                raise PermissionError("artifact transfer is not authorized")
            status = self._require_artifact_manager().complete_incoming(
                transfer_id=request.transfer_id,
                manifest=request.manifest,
            )
            self._artifact_authorizations.pop(request.transfer_id, None)
            return serialize_message(
                ArtifactTransferResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    artifact_id=request.manifest.artifact_id,
                    accepted=True,
                    transfer_id=request.transfer_id,
                    bytes_completed=status.bytes_completed,
                    chunks_completed=status.chunks_completed,
                    complete=True,
                    verified=True,
                    detail="artifact fully verified and published",
                )
            )
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.DATA_LOSS, str(exc))
            raise

    async def _verify_artifact(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, VerifyArtifactRequest)
            self._authorize_artifact(
                request.lease,
                worker_id=request.worker_id,
                artifact_id=request.artifact_id,
            )
            directory = self._require_artifact_manager().resolve(request.artifact_id)
            return serialize_message(
                ArtifactTransferResponse(
                    worker_id=request.worker_id,
                    request_id=request.request_id,
                    artifact_id=request.artifact_id,
                    accepted=True,
                    complete=True,
                    verified=True,
                    detail=f"verified cached artifact {directory.name[:12]}",
                )
            )
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _load_stage(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, LoadStageRequest)
            response = await self._require_stage_runtime().load_stage(request)
            return serialize_message(response)
        except (MemoryError, MemoryLimitExceededError) as exc:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _unload_stage(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, UnloadStageRequest)
            response = await self._require_stage_runtime().unload_stage(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _install_stage_route(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, InstallStageRouteRequest)
            response = await self._require_stage_runtime().install_route(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _remove_stage_route(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, RemoveStageRouteRequest)
            response = await self._require_stage_runtime().remove_route(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _verify_stage_route(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, VerifyStageRouteRequest)
            response = await self._require_stage_runtime().verify_route(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _open_stage_session(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, OpenStageSessionRequest)
            response = await self._require_stage_runtime().open_session(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            raise

    async def _close_stage_session(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, CloseStageSessionRequest)
            response = await self._require_stage_runtime().close_session(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _cancel_stage_session(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, CancelStageSessionRequest)
            response = await self._require_stage_runtime().cancel_session(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _get_stage_status(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, GetStageStatusRequest)
            response = await self._require_stage_runtime().status(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    async def _tokenize_stage(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, TokenizeStageRequest)
            response = await self._require_stage_runtime().tokenize(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _drain_worker(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, DrainWorkerRequest)
            response = await self._require_stage_runtime().drain(request)
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise

    def configure_engine_trust(
        self,
        *,
        coordinator_public_key: str,
        expected_fingerprint: str | None,
    ) -> None:
        if self.engine_runtime is None:
            return
        self.engine_runtime.configure_trust(
            coordinator_public_key=coordinator_public_key,
            expected_fingerprint=expected_fingerprint,
        )

    def _require_engine_runtime(self) -> PersistentEngineRuntime:
        if self.engine_runtime is None:
            raise RuntimeError("worker engine runtime is disabled")
        return self.engine_runtime

    async def _prepare_engine(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, PrepareEngineRequest)
            return serialize_message(await self._require_engine_runtime().prepare(request))
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _submit_engine(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, SubmitEngineRequest)
            return serialize_message(await self._require_engine_runtime().submit(request))
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _submit_engine_stream(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> AsyncIterator[bytes]:
        try:
            request = parse_message(data, SubmitEngineRequest)
            async for event in self._require_engine_runtime().stream(request):
                yield serialize_message(
                    EngineInferenceChunk(
                        worker_id=request.worker_id,
                        request_id=request.request_id,
                        deployment_id=request.deployment_id,
                        event=event,
                    )
                )
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise

    async def _unload_engine(
        self,
        data: bytes,
        context: grpc.aio.ServicerContext[Any, Any],
    ) -> bytes:
        try:
            request = parse_message(data, UnloadEngineRequest)
            return serialize_message(await self._require_engine_runtime().unload(request))
        except PermissionError as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise
        except Exception as exc:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise
