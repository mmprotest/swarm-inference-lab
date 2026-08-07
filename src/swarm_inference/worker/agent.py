"""Worker lifecycle independent of transport implementation."""

from __future__ import annotations

import asyncio
import os
import time

from swarm_inference.config.models import (
    DataPlaneMode,
    ModelManifest,
    OperationKind,
    QueueConfig,
    StageDefinition,
    SyntheticModelConfig,
    WorkerCapability,
)
from swarm_inference.exceptions import BackpressureError, RouteMessageError, TransportError
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    Ack,
    ActivationRequest,
    ActivationResult,
    CacheControlRequest,
    CacheControlResponse,
    DataPlaneAck,
    DataPlaneEnvelope,
    FinalResultMessage,
    HealthResponse,
    HopTelemetry,
    RoutePlan,
    StageAssignmentMessage,
)
from swarm_inference.protocol.routes import (
    decode_route_key,
    sign_data_envelope,
    sign_final_result,
    verify_data_envelope,
    verify_route_plan,
)
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor
from swarm_inference.runtime.telemetry import lifecycle_observer
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.security.tls import TlsClientConfig
from swarm_inference.transport.peer import FinalResultClient, PeerConnectionPool
from swarm_inference.worker.execution import ExecutionEngine
from swarm_inference.worker.metrics import WorkerMetrics
from swarm_inference.worker.shard_manager import ShardManager


class WorkerAgent:
    def __init__(
        self,
        *,
        capability: WorkerCapability,
        identity: WorkerIdentity,
        queue_config: QueueConfig,
        total_memory_limit_bytes: int | None = None,
        outbound_queue_capacity: int = 1024,
        inbound_queue_capacity: int = 1024,
        max_inflight_operations: int = 256,
        reconnect_attempts: int = 5,
        reconnect_initial_backoff_ms: float = 25.0,
        reconnect_max_backoff_ms: float = 1000.0,
        peer_tls: TlsClientConfig | None = None,
        coordinator_tls: TlsClientConfig | None = None,
    ) -> None:
        if capability.memory_limit_bytes is None:
            raise ValueError("worker capability must declare an enforced memory limit")
        self.capability = capability
        self.identity = identity
        self.shards = ShardManager(
            memory_limit_bytes=capability.memory_limit_bytes,
            total_memory_limit_bytes=total_memory_limit_bytes,
        )
        self.metrics = WorkerMetrics(worker_id=capability.worker_id)
        self.execution = ExecutionEngine(
            worker_id=capability.worker_id,
            identity=identity,
            shards=self.shards,
            queue_config=queue_config,
            metrics=self.metrics,
        )
        self.peer_pool = PeerConnectionPool(
            queue_capacity=outbound_queue_capacity,
            reconnect_attempts=reconnect_attempts,
            reconnect_initial_backoff_ms=reconnect_initial_backoff_ms,
            reconnect_max_backoff_ms=reconnect_max_backoff_ms,
            tls=peer_tls,
        )
        self._coordinator_tls = coordinator_tls
        self._inbound_capacity = inbound_queue_capacity
        self._inbound_active = 0
        self._inbound_lock = asyncio.Lock()
        self._inflight_semaphore = asyncio.Semaphore(max_inflight_operations)
        self._data_plane_mode = DataPlaneMode.COORDINATOR_RELAY
        self._route_key: bytes | None = None
        self._coordinator_data_endpoint: str | None = None
        self._final_client: FinalResultClient | None = None
        self._routes: dict[str, RoutePlan] = {}
        self._assigned_shard_hashes: dict[int, str] = {}
        self._cancelled_requests: set[str] = set()
        self._message_results: dict[str, DataPlaneAck] = {}
        self._message_futures: dict[str, asyncio.Future[DataPlaneAck]] = {}
        self._message_lock = asyncio.Lock()
        self._stage_replay_inputs: dict[tuple[str, int], list[ActivationRequest]] = {}

    async def start(self) -> None:
        await self.execution.start()

    @property
    def inbound_capacity(self) -> int:
        return self._inbound_capacity

    async def stop(self) -> None:
        await self.peer_pool.close()
        if self._final_client is not None:
            await self._final_client.close()
            self._final_client = None
        await self.execution.stop()
        for stage_id in list(self.shards.modules):
            self.shards.unload(stage_id)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
                torch.cuda.empty_cache()
        except (ImportError, OSError, RuntimeError):
            pass

    def configure_data_plane(self, assignment: StageAssignmentMessage) -> None:
        self._data_plane_mode = assignment.data_plane_mode
        self._assigned_shard_hashes[assignment.stage.stage_id] = assignment.shard_hash
        if assignment.data_plane_mode == DataPlaneMode.DIRECT:
            if assignment.route_signing_key is None:
                raise ValueError("direct assignment is missing route_signing_key")
            if assignment.coordinator_data_endpoint is None:
                raise ValueError("direct assignment is missing coordinator_data_endpoint")
            key = decode_route_key(assignment.route_signing_key)
            if self._route_key is not None and self._route_key != key:
                raise ValueError("worker received a different route signing key")
            self._route_key = key
            self._coordinator_data_endpoint = assignment.coordinator_data_endpoint
            if self._final_client is None:
                self._final_client = FinalResultClient(
                    assignment.coordinator_data_endpoint,
                    tls=self._coordinator_tls,
                )

    def install_route(self, route: RoutePlan) -> None:
        if self._data_plane_mode != DataPlaneMode.DIRECT:
            raise RouteMessageError(
                "invalid_stage_transition",
                "route installation is only valid in direct mode",
            )
        if self._route_key is None:
            raise RouteMessageError("unknown_request", "worker has no route signing key")
        verify_route_plan(route, self._route_key)
        if route.route_lease_expiry_unix_ns <= time.time_ns():
            raise RouteMessageError(
                "stale_route",
                f"route {route.route_id} lease has expired",
            )
        owned = [hop for hop in route.assignments if hop.worker_id == self.capability.worker_id]
        if len(owned) != 1:
            raise RouteMessageError(
                "invalid_stage_transition",
                f"route {route.route_id} does not assign exactly one stage to "
                f"{self.capability.worker_id}",
            )
        hop = owned[0]
        actual_hash = self._assigned_shard_hashes.get(hop.stage_id)
        if actual_hash != hop.expected_shard_hash:
            raise RouteMessageError(
                "invalid_stage_transition",
                f"stage {hop.stage_id} shard hash mismatch: "
                f"expected={hop.expected_shard_hash} loaded={actual_hash}",
            )
        existing = self._routes.get(route.route_id)
        if existing is not None and route.route_generation < existing.route_generation:
            raise RouteMessageError(
                "stale_route",
                f"route generation {route.route_generation} is older than "
                f"{existing.route_generation}",
            )
        self._routes[route.route_id] = route.model_copy(deep=True)
        self._cancelled_requests.discard(route.request_id)

    def assign_synthetic(
        self,
        *,
        config: SyntheticModelConfig,
        stage: object,
        corrupt: bool = False,
    ) -> None:
        from swarm_inference.config.models import StageDefinition

        if not isinstance(stage, StageDefinition):
            raise TypeError("stage must be StageDefinition")
        self.shards.load_synthetic(config=config, stage=stage, corrupt=corrupt)
        self.capability.current_shard_assignments = sorted(self.shards.modules)

    def assign_qwen3(
        self,
        *,
        config: dict[str, object],
        manifest: ModelManifest,
        stage: StageDefinition,
        shard_path: str,
        shard_hash: str,
        dtype: str | None,
    ) -> None:
        module = self.shards.load_qwen3(
            config=config,
            manifest=manifest,
            stage=stage,
            shard_path=shard_path,
            expected_hash=shard_hash,
            backend=self.capability.backend,
            dtype_name=dtype,
        )
        recorder = lifecycle_observer()
        warmup_started = time.monotonic_ns()
        perform_warmup = os.environ.get("SWARM_STAGE_LOCAL_WARMUP", "0") == "1"
        if recorder is not None:
            recorder.emit(
                "local_warmup_started",
                monotonic_ns=warmup_started,
                details={
                    "warmup_level": "stage-local",
                    "warmup_performed": perform_warmup,
                },
            )
        try:
            warmup_metrics: dict[str, object] = {}
            if perform_warmup:
                warmup = getattr(module, "warmup", None)
                if not callable(warmup):
                    raise RuntimeError("real stage module does not expose stage-local warmup")
                warmup_metrics = dict(
                    warmup(
                        sequence_length=int(os.environ.get("SWARM_WARMUP_SEQUENCE_LENGTH", "128"))
                    )
                )
            warmup_completed = time.monotonic_ns()
            if recorder is not None:
                recorder.emit(
                    "local_warmup_completed",
                    monotonic_ns=warmup_completed,
                    duration_ns=warmup_completed - warmup_started,
                    details={
                        "warmup_level": "stage-local",
                        "warmup_performed": perform_warmup,
                        **warmup_metrics,
                    },
                )
        except Exception as exc:
            warmup_completed = time.monotonic_ns()
            if recorder is not None:
                recorder.emit(
                    "local_warmup_completed",
                    monotonic_ns=warmup_completed,
                    duration_ns=warmup_completed - warmup_started,
                    error=f"{type(exc).__name__}: {exc}",
                    details={
                        "warmup_level": "stage-local",
                        "warmup_performed": perform_warmup,
                    },
                )
            raise
        self.capability.current_shard_assignments = sorted(self.shards.modules)

    async def execute(self, request: ActivationRequest) -> ActivationResult:
        return await self.execution.submit(request)

    def cancel(self, request_id: str) -> None:
        self._cancelled_requests.add(request_id)
        for module in self.shards.modules.values():
            module.cancel(request_id)
        for route_id in [
            route_id for route_id, route in self._routes.items() if route.request_id == request_id
        ]:
            self._routes.pop(route_id, None)
        for key in [key for key in self._stage_replay_inputs if key[0] == request_id]:
            self._stage_replay_inputs.pop(key, None)

    async def cache_control(
        self,
        request: CacheControlRequest,
    ) -> CacheControlResponse:
        if request.stage_id not in self.shards.modules:
            return CacheControlResponse(
                accepted=False,
                worker_id=self.capability.worker_id,
                request_id=request.request_id,
                stage_id=request.stage_id,
                action=request.action,
                detail="stage is not loaded by this worker",
            )
        module = self.shards.module(request.stage_id)
        inspect_cache = getattr(module, "inspect_cache", None)
        reset_cache = getattr(module, "reset_cache", None)
        if not callable(inspect_cache) or not callable(reset_cache):
            return CacheControlResponse(
                accepted=False,
                worker_id=self.capability.worker_id,
                request_id=request.request_id,
                stage_id=request.stage_id,
                action=request.action,
                detail="loaded stage does not expose real-model cache control",
            )
        module_revision = str(getattr(module, "model_revision", ""))
        if module_revision != request.model_revision:
            return CacheControlResponse(
                accepted=False,
                worker_id=self.capability.worker_id,
                request_id=request.request_id,
                stage_id=request.stage_id,
                action=request.action,
                detail=(
                    f"model revision mismatch: loaded={module_revision} "
                    f"requested={request.model_revision}"
                ),
            )
        before = list(inspect_cache(request.request_id))
        replay_count = 0
        replay_bytes = 0
        replay_started = time.perf_counter()
        if request.action in {"clear", "clear-and-replay"}:
            reset_cache(request.request_id, for_replay=True)
        if request.action in {"replay", "clear-and-replay"}:
            entries = self._stage_replay_inputs.get(
                (request.request_id, request.stage_id),
                [],
            )
            for entry in entries:
                decoded = decode_tensor(entry.tensor_payload)
                module.execute(
                    decoded.array,
                    request_id=request.request_id,
                    operation=OperationKind.REPLAY,
                    token_position=entry.metadata.token_position,
                    sequence_length=entry.metadata.sequence_length,
                    cache_generation=entry.metadata.cache_generation,
                    route_generation=entry.metadata.route_generation,
                )
                replay_count += 1
                replay_bytes += len(entry.tensor_payload)
        duration = (
            time.perf_counter() - replay_started
            if request.action in {"replay", "clear-and-replay"}
            else 0.0
        )
        after = list(inspect_cache(request.request_id))
        return CacheControlResponse(
            accepted=True,
            worker_id=self.capability.worker_id,
            request_id=request.request_id,
            stage_id=request.stage_id,
            action=request.action,
            replay_input_count=replay_count,
            replay_bytes=replay_bytes,
            replay_duration_s=duration,
            cache_before=before,
            cache_after=after,
            detail="cache operation completed",
        )

    async def _claim_message(
        self,
        message_id: str,
    ) -> tuple[bool, asyncio.Future[DataPlaneAck]]:
        async with self._message_lock:
            existing_result = self._message_results.get(message_id)
            loop = asyncio.get_running_loop()
            if existing_result is not None:
                future: asyncio.Future[DataPlaneAck] = loop.create_future()
                future.set_result(existing_result)
                return False, future
            existing_future = self._message_futures.get(message_id)
            if existing_future is not None:
                return False, existing_future
            future = loop.create_future()
            self._message_futures[message_id] = future
            return True, future

    async def _complete_message(self, message_id: str, ack: DataPlaneAck) -> None:
        async with self._message_lock:
            future = self._message_futures.pop(message_id, None)
            self._message_results[message_id] = ack
            if len(self._message_results) > 100_000:
                oldest = next(iter(self._message_results))
                self._message_results.pop(oldest, None)
            if future is not None and not future.done():
                future.set_result(ack)

    def _validate_envelope(self, envelope: DataPlaneEnvelope) -> tuple[RoutePlan, int]:
        if self._data_plane_mode != DataPlaneMode.DIRECT:
            raise RouteMessageError(
                "invalid_stage_transition",
                "direct message received while worker is not in direct mode",
            )
        if self._route_key is None:
            raise RouteMessageError("unknown_request", "worker has no route key")
        route = self._routes.get(envelope.route_id)
        if route is None:
            raise RouteMessageError(
                "unknown_request",
                f"unknown route {envelope.route_id}",
            )
        verify_data_envelope(envelope, self._route_key)
        if route.route_lease_expiry_unix_ns <= time.time_ns():
            raise RouteMessageError(
                "stale_route",
                f"route {route.route_id} lease expired",
            )
        if envelope.route_generation != route.route_generation:
            raise RouteMessageError(
                "stale_route",
                f"route generation mismatch: installed={route.route_generation} "
                f"message={envelope.route_generation}",
            )
        if envelope.request_id != route.request_id:
            raise RouteMessageError(
                "unknown_request",
                f"route request {route.request_id} does not match {envelope.request_id}",
            )
        if envelope.request_id in self._cancelled_requests:
            raise RouteMessageError(
                "unknown_request",
                f"request {envelope.request_id} is cancelled",
            )
        if envelope.payload_length != len(envelope.tensor_payload):
            raise RouteMessageError(
                "invalid_checksum",
                f"payload length mismatch: declared={envelope.payload_length} "
                f"actual={len(envelope.tensor_payload)}",
            )
        actual_checksum = sha256_bytes(envelope.tensor_payload)
        if envelope.payload_checksum != actual_checksum:
            raise RouteMessageError(
                "invalid_checksum",
                f"activation checksum mismatch: expected={envelope.payload_checksum} "
                f"actual={actual_checksum}",
            )
        try:
            stage_index = next(
                index
                for index, hop in enumerate(route.assignments)
                if hop.worker_id == self.capability.worker_id
            )
        except StopIteration as exc:
            raise RouteMessageError(
                "invalid_stage_transition",
                f"worker {self.capability.worker_id} is not in route",
            ) from exc
        hop = route.assignments[stage_index]
        expected_source = (
            "coordinator" if stage_index == 0 else route.assignments[stage_index - 1].worker_id
        )
        if (
            envelope.stage_id != hop.stage_id
            or envelope.destination_worker != hop.worker_id
            or envelope.source_worker != expected_source
        ):
            raise RouteMessageError(
                "invalid_stage_transition",
                "message skips, repeats, or misaddresses a route stage: "
                f"expected source={expected_source} stage={hop.stage_id} "
                f"destination={hop.worker_id}; got source={envelope.source_worker} "
                f"stage={envelope.stage_id} destination={envelope.destination_worker}",
            )
        metadata = envelope.tensor_metadata
        if (
            metadata.request_id != envelope.request_id
            or metadata.stage_id != hop.stage_id
            or metadata.token_position != envelope.token_position
            or metadata.operation != envelope.operation
            or metadata.route_generation != envelope.route_generation
            or metadata.model_id != route.model_id
            or metadata.model_revision != route.model_revision
        ):
            raise RouteMessageError(
                "invalid_stage_transition",
                "activation metadata does not match the signed route envelope",
            )
        decoded = decode_tensor(envelope.tensor_payload, copy=False)
        if (
            decoded.request_id != envelope.request_id
            or decoded.stage_id != hop.stage_id
            or decoded.token_position != envelope.token_position
        ):
            raise RouteMessageError(
                "invalid_stage_transition",
                "tensor metadata does not match the expected route stage",
            )
        return route, stage_index

    async def handle_data_plane(self, envelope: DataPlaneEnvelope) -> DataPlaneAck:
        recorder = lifecycle_observer()
        if recorder is not None and not envelope.replay_only:
            recorder.emit_once(
                "first-request-received",
                "first_request_received",
                details={
                    "request_id": envelope.request_id,
                    "operation": envelope.operation.value,
                    "token_position": envelope.token_position,
                },
            )
        owner, prior = await self._claim_message(envelope.message_id)
        if not owner:
            prior_ack = await prior
            return prior_ack.model_copy(
                update={
                    "status": "duplicate",
                    "detail": f"duplicate of accepted message {envelope.message_id}",
                    "accepted_timestamp_unix_ns": time.time_ns(),
                }
            )
        ack: DataPlaneAck
        try:
            async with self._inbound_lock:
                if self._inbound_active >= self._inbound_capacity:
                    raise BackpressureError(
                        f"inbound queue capacity {self._inbound_capacity} reached"
                    )
                self._inbound_active += 1
            try:
                async with self._inflight_semaphore:
                    ack = await self._handle_data_plane_owned(envelope)
            finally:
                async with self._inbound_lock:
                    self._inbound_active -= 1
        except BackpressureError as exc:
            ack = DataPlaneAck(
                message_id=envelope.message_id,
                status="backpressured",
                detail=str(exc),
                accepted_timestamp_unix_ns=time.time_ns(),
            )
        except RouteMessageError as exc:
            status = exc.status
            if status not in {
                "stale_route",
                "invalid_checksum",
                "invalid_stage_transition",
                "unknown_request",
            }:
                status = "invalid_stage_transition"
            ack = DataPlaneAck(
                message_id=envelope.message_id,
                status=status,
                detail=exc.detail,
                accepted_timestamp_unix_ns=time.time_ns(),
            )
        except (TransportError, OSError) as exc:
            ack = DataPlaneAck(
                message_id=envelope.message_id,
                status="destination_unavailable",
                detail=str(exc),
                accepted_timestamp_unix_ns=time.time_ns(),
            )
        except Exception as exc:
            ack = DataPlaneAck(
                message_id=envelope.message_id,
                status="invalid_stage_transition",
                detail=f"{type(exc).__name__}: {exc}",
                accepted_timestamp_unix_ns=time.time_ns(),
            )
        await self._complete_message(envelope.message_id, ack)
        return ack

    async def _handle_data_plane_owned(
        self,
        envelope: DataPlaneEnvelope,
    ) -> DataPlaneAck:
        validation_started = time.perf_counter_ns()
        route, stage_index = self._validate_envelope(envelope)
        validation_ms = (time.perf_counter_ns() - validation_started) / 1_000_000
        if envelope.hop_telemetry:
            previous = envelope.hop_telemetry[-1]
            measured_transfer_ms = max(
                0.0,
                (time.time_ns() - envelope.timestamp_unix_ns) / 1_000_000,
            )
            envelope.hop_telemetry[-1] = previous.model_copy(
                update={
                    "transfer_ms": measured_transfer_ms,
                    "hop_end_to_end_ms": (previous.hop_end_to_end_ms + measured_transfer_ms),
                }
            )
        request = ActivationRequest(
            metadata=envelope.tensor_metadata,
            tensor_payload=envelope.tensor_payload,
        )
        if not envelope.replay_only:
            self._stage_replay_inputs.setdefault(
                (envelope.request_id, envelope.stage_id),
                [],
            ).append(request.model_copy(deep=True))
        result = await self.execute(request)
        next_hop = (
            route.assignments[stage_index + 1] if stage_index + 1 < len(route.assignments) else None
        )
        serialisation_started = time.perf_counter_ns()
        if next_hop is not None:
            decoded = decode_tensor(result.tensor_payload)
            forwarded_payload = encode_tensor(
                ActivationTensor(
                    tensor_id=(
                        f"{envelope.request_id}:{envelope.token_position}:{next_hop.stage_id}"
                    ),
                    request_id=envelope.request_id,
                    stage_id=next_hop.stage_id,
                    token_position=envelope.token_position,
                    sequence_length=decoded.sequence_length,
                    array=decoded.array,
                    logical_dtype=decoded.logical_dtype,
                )
            )
        else:
            forwarded_payload = result.tensor_payload
        serialisation_ms = (time.perf_counter_ns() - serialisation_started) / 1_000_000
        destination = next_hop.worker_id if next_hop is not None else route.final_result_destination
        telemetry = HopTelemetry(
            stage_id=envelope.stage_id,
            worker_id=self.capability.worker_id,
            source_worker=self.capability.worker_id,
            destination_worker=destination,
            execution_ms=result.execution_ms,
            queue_ms=result.queue_ms,
            serialisation_ms=serialisation_ms,
            deserialisation_ms=0.0,
            integrity_validation_ms=validation_ms,
            cache_update_ms=0.0,
            stream_queue_ms=0.0,
            transfer_ms=0.0,
            hop_end_to_end_ms=(
                result.execution_ms + result.queue_ms + serialisation_ms + validation_ms
            ),
            payload_bytes=len(forwarded_payload),
        )
        all_telemetry = [*envelope.hop_telemetry, telemetry]
        if next_hop is not None:
            if self._route_key is None:
                raise RouteMessageError("unknown_request", "route key disappeared")
            metadata = envelope.tensor_metadata.model_copy(
                update={
                    "tensor_id": (
                        f"{envelope.request_id}:{envelope.token_position}:{next_hop.stage_id}"
                    ),
                    "stage_id": next_hop.stage_id,
                }
            )
            forwarded = DataPlaneEnvelope(
                message_id=(
                    f"{route.route_id}:{route.route_generation}:"
                    f"{envelope.token_position}:{next_hop.stage_id}"
                ),
                route_id=route.route_id,
                route_generation=route.route_generation,
                request_id=route.request_id,
                stage_id=next_hop.stage_id,
                source_worker=self.capability.worker_id,
                destination_worker=next_hop.worker_id,
                token_position=envelope.token_position,
                operation=envelope.operation,
                tensor_metadata=metadata,
                tensor_payload=forwarded_payload,
                payload_length=len(forwarded_payload),
                payload_checksum=sha256_bytes(forwarded_payload),
                sequence_number=envelope.sequence_number + 1,
                timestamp_unix_ns=time.time_ns(),
                hop_telemetry=all_telemetry,
                replay_only=envelope.replay_only,
            )
            forwarded = sign_data_envelope(forwarded, self._route_key)
            downstream_ack = await self.peer_pool.send(
                next_hop.worker_data_endpoint,
                forwarded,
            )
            if not downstream_ack.accepted:
                return downstream_ack
        elif not envelope.replay_only:
            if self._final_client is None or self._route_key is None:
                raise TransportError("final-result client is unavailable")
            final_message = FinalResultMessage(
                message_id=(
                    f"final:{route.route_id}:{route.route_generation}:{envelope.token_position}"
                ),
                route_id=route.route_id,
                route_generation=route.route_generation,
                request_id=route.request_id,
                token_position=envelope.token_position,
                result=result,
                hop_telemetry=all_telemetry,
                payload_checksum=sha256_bytes(result.tensor_payload),
                timestamp_unix_ns=time.time_ns(),
            )
            final_message = sign_final_result(final_message, self._route_key)
            final_ack: Ack = await self._final_client.send(final_message)
            if not final_ack.accepted:
                raise TransportError(f"coordinator rejected final result: {final_ack.detail}")
        return DataPlaneAck(
            message_id=envelope.message_id,
            status="accepted",
            detail="activation accepted and committed by downstream path",
            accepted_timestamp_unix_ns=time.time_ns(),
        )

    def health(self) -> HealthResponse:
        return HealthResponse(
            worker_id=self.capability.worker_id,
            healthy=True,
            queue_depth=self.execution.queue_depth,
            loaded_stages=sorted(self.shards.modules),
            detail="ready",
            proof=self.proof(),
        )

    def proof(self) -> dict[str, object]:
        proof: dict[str, object] = {
            # Health proofs cross deliberately small control-message limits in
            # chunking tests. Defaults are part of the strict capability schema,
            # so omitting them retains meaning while bounding wire overhead as
            # compatibility evidence grows.
            "capability": self.capability.model_dump(
                mode="json",
                exclude_defaults=True,
                exclude_none=True,
            ),
            "shards": self.shards.proof(),
            "metrics": self.metrics.snapshot(),
            "data_plane_mode": self._data_plane_mode.value,
            "installed_routes": len(self._routes),
            "peer_connections": self.peer_pool.snapshot(),
            "inbound_active": self._inbound_active,
            "inbound_capacity": self._inbound_capacity,
        }
        canonical = canonical_json_bytes(proof)
        proof["proof_checksum"] = sha256_bytes(canonical)
        proof["proof_signature"] = self.identity.sign(canonical)
        return proof
