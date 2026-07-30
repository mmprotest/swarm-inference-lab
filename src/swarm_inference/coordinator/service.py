"""Central coordinator core and gRPC control service."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import grpc
import numpy as np

from swarm_inference.config.models import (
    ExperimentConfig,
    HealthStatus,
    ModelManifest,
    OperationKind,
    RequestState,
    RequestStatus,
    SamplingConfig,
    StageDefinition,
    StageReplica,
    StrictModel,
    VerificationState,
    WorkloadClass,
)
from swarm_inference.coordinator.placement import estimate_worker_stage_rate
from swarm_inference.coordinator.registry import WorkerRegistry
from swarm_inference.coordinator.replay_log import ReplayLog
from swarm_inference.exceptions import (
    IntegrityError,
    NoValidRouteError,
    ReplayUnavailableError,
    TransportError,
)
from swarm_inference.model.synthetic import synthetic_activation
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import (
    Ack,
    ActivationMetadata,
    ActivationRequest,
    ActivationResult,
    Heartbeat,
    RegistrationRequest,
    RegistrationResponse,
    StageAssignmentMessage,
    SubmitRequest,
    SubmitResponse,
    parse_message,
    serialize_message,
)
from swarm_inference.protocol.tensor_codec import ActivationTensor, decode_tensor, encode_tensor
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.simulation.model import build_synthetic_stages
from swarm_inference.transport.base import ActivationTransport
from swarm_inference.transport.grpc_transport import GrpcTransport

ResponseT = TypeVar("ResponseT", bound=StrictModel)


@dataclass(slots=True)
class RuntimeRequestMetrics:
    request_id: str
    started_s: float
    first_token_s: float | None = None
    completed_s: float | None = None
    stage_execution_s: float = 0.0
    queue_s: float = 0.0
    transport_s: float = 0.0
    replay_s: float = 0.0
    replay_bytes: int = 0
    retries: int = 0
    route_changes: int = 0
    per_stage: list[dict[str, Any]] = field(default_factory=list)


class CoordinatorCore:
    """Coordinator owns metadata, routes, token commitment, and replay inputs."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        registry: WorkerRegistry | None = None,
        transport: ActivationTransport | None = None,
        model_manifest: ModelManifest | None = None,
        architecture_config: dict[str, Any] | None = None,
        runtime_dtype: str | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if (model_manifest is None) != (architecture_config is None):
            raise ValueError(
                "real-model runtime requires both model_manifest and architecture_config"
            )
        self.config = config
        self.registry = registry or WorkerRegistry()
        self.transport = transport or GrpcTransport()
        self.model_manifest = model_manifest
        self.architecture_config = architecture_config
        self.runtime_dtype = runtime_dtype
        self.tokenizer = tokenizer
        if model_manifest is None:
            self.stages = build_synthetic_stages(config.model)
            self.runtime_model_id = config.model_id
            self.runtime_model_revision = config.model_revision
        else:
            self.stages = self._runtime_stages(model_manifest, runtime_dtype)
            self.runtime_model_id = model_manifest.model_id
            self.runtime_model_revision = model_manifest.model_revision
        self.replay = ReplayLog()
        self.events: list[dict[str, Any]] = []
        self.request_metrics: list[dict[str, Any]] = []
        self._rebalance_lock = asyncio.Lock()
        self._assigned: set[tuple[int, str]] = set()

    @staticmethod
    def _runtime_stages(
        manifest: ModelManifest,
        runtime_dtype: str | None,
    ) -> list[StageDefinition]:
        source_width = {
            "F16": 2,
            "BF16": 2,
            "F32": 4,
        }.get(manifest.weight_dtype.upper())
        target_width = {
            "f16": 2,
            "float16": 2,
            "bf16": 2,
            "bfloat16": 2,
            "f32": 4,
            "float32": 4,
        }.get((runtime_dtype or manifest.weight_dtype).lower())
        if source_width is None or target_width is None:
            raise ValueError(
                f"unsupported source/runtime dtype pair: {manifest.weight_dtype}/{runtime_dtype}"
            )
        return [
            stage.model_copy(
                update={
                    "required_memory_bytes": math.ceil(
                        stage.required_memory_bytes * target_width / source_width
                    )
                }
            )
            for stage in manifest.stages
        ]

    def _worker_can_host(self, worker: Any, stage: Any) -> bool:
        if stage.required_memory_bytes > worker.effective_memory_bytes:
            return False
        if estimate_worker_stage_rate(worker, stage) <= 0:
            return False
        if self.model_manifest is None:
            return True
        if worker.backend not in self.model_manifest.compatible_worker_backends:
            return False
        normalised_dtype = (self.runtime_dtype or self.model_manifest.weight_dtype).lower()
        aliases = {
            "f16": "float16",
            "bf16": "bfloat16",
            "f32": "float32",
        }
        required_dtype = aliases.get(normalised_dtype, normalised_dtype)
        return required_dtype in worker.supported_dtypes

    async def close(self) -> None:
        await self.transport.close()

    async def register(self, request: RegistrationRequest) -> RegistrationResponse:
        signed_payload = canonical_json_bytes(
            {
                "capability": request.capability.model_dump(mode="json"),
                "benchmark_nonce": request.benchmark_nonce,
            }
        )
        verify_signature(
            request.capability.public_key,
            signed_payload,
            request.signature,
        )
        self.registry.register(request.capability, benchmark_verified=True)
        self.events.append(
            {
                "event_type": "worker_registered",
                "worker_id": request.capability.worker_id,
                "endpoint": request.capability.endpoint,
                "timestamp_monotonic_s": time.monotonic(),
            }
        )
        await self.rebalance()
        return RegistrationResponse(accepted=True, heartbeat_interval_s=2.0)

    async def heartbeat(self, request: Heartbeat) -> Ack:
        capability = self.registry.capability(request.worker_id)
        signed_payload = canonical_json_bytes(
            {
                "worker_id": request.worker_id,
                "queue_depth": request.queue_depth,
                "assignments": request.assignments,
                "monotonic_ns": request.monotonic_ns,
                "timestamp": request.timestamp.isoformat(),
            }
        )
        verify_signature(capability.public_key, signed_payload, request.signature)
        self.registry.heartbeat(
            request.worker_id,
            queue_depth=request.queue_depth,
            assignments=request.assignments,
        )
        return Ack(accepted=True, detail="heartbeat recorded")

    async def rebalance(self) -> None:
        async with self._rebalance_lock:
            workers = self.registry.workers()
            if not workers:
                return
            assigned_worker_ids = {worker_id for _, worker_id in self._assigned}
            remaining = [
                worker for worker in workers if worker.worker_id not in assigned_worker_ids
            ]
            replica_counts = {
                stage.stage_id: len(self.registry.replicas(stage.stage_id)) for stage in self.stages
            }
            assignments: list[tuple[Any, Any, float]] = []

            # First establish complete coverage. Workers are selected by their
            # measured stage benchmark, not a declared relative-speed field.
            for stage in sorted(
                self.stages,
                key=lambda item: (replica_counts[item.stage_id], item.stage_id),
            ):
                if replica_counts[stage.stage_id] > 0:
                    continue
                candidates = [
                    worker for worker in remaining if self._worker_can_host(worker, stage)
                ]
                if not candidates:
                    self.events.append(
                        {
                            "event_type": "placement_pending",
                            "detail": f"no compatible worker for uncovered stage {stage.stage_id}",
                            "worker_count": len(workers),
                        }
                    )
                    return
                capability = max(
                    candidates,
                    key=lambda worker: (
                        estimate_worker_stage_rate(worker, stage),
                        worker.worker_id,
                    ),
                )
                rate = estimate_worker_stage_rate(capability, stage)
                assignments.append((capability, stage, rate))
                remaining.remove(capability)
                replica_counts[stage.stage_id] += 1

            # A tied set of bottleneck stages needs a complete replica round
            # before min pipeline capacity rises. Keep an incomplete round idle
            # rather than claiming a positive marginal throughput benefit.
            while len(remaining) >= len(self.stages):
                round_assignments: list[tuple[Any, Any, float]] = []
                round_workers = list(remaining)
                for stage in sorted(
                    self.stages,
                    key=lambda item: (replica_counts[item.stage_id], item.stage_id),
                ):
                    candidates = [
                        worker for worker in round_workers if self._worker_can_host(worker, stage)
                    ]
                    if not candidates:
                        round_assignments = []
                        break
                    capability = max(
                        candidates,
                        key=lambda worker: (
                            estimate_worker_stage_rate(worker, stage),
                            worker.worker_id,
                        ),
                    )
                    rate = estimate_worker_stage_rate(capability, stage)
                    round_assignments.append((capability, stage, rate))
                    round_workers.remove(capability)
                if not round_assignments:
                    break
                assignments.extend(round_assignments)
                for capability, stage, _ in round_assignments:
                    remaining.remove(capability)
                    replica_counts[stage.stage_id] += 1

            for capability, stage, rate in assignments:
                key = (stage.stage_id, capability.worker_id)
                if capability.endpoint is None:
                    self.events.append(
                        {
                            "event_type": "placement_rejected",
                            "worker_id": capability.worker_id,
                            "stage_id": stage.stage_id,
                            "detail": "worker has no advertised endpoint",
                        }
                    )
                    continue
                if self.model_manifest is None:
                    shard_hash = "synthetic-deterministic"
                    assignment = StageAssignmentMessage(
                        worker_id=capability.worker_id,
                        stage=stage,
                        shard_path="synthetic://deterministic",
                        shard_hash=shard_hash,
                        model_id=self.runtime_model_id,
                        model_revision=self.runtime_model_revision,
                        synthetic_model=self.config.model,
                    )
                else:
                    shard_name = f"stage-{stage.stage_id:03d}"
                    try:
                        shard_hash = self.model_manifest.shard_hashes[shard_name]
                    except KeyError as exc:
                        raise IntegrityError(
                            f"model manifest has no hash for required shard {shard_name}"
                        ) from exc
                    assignment = StageAssignmentMessage(
                        worker_id=capability.worker_id,
                        stage=stage,
                        shard_path=shard_name,
                        shard_hash=shard_hash,
                        model_id=self.runtime_model_id,
                        model_revision=self.runtime_model_revision,
                        architecture_config=self.architecture_config,
                        model_manifest=self.model_manifest,
                        dtype=self.runtime_dtype,
                    )
                try:
                    ack = await self.transport.assign(capability.endpoint, assignment)
                except TransportError as exc:
                    self.events.append(
                        {
                            "event_type": "assignment_failed",
                            "worker_id": capability.worker_id,
                            "stage_id": stage.stage_id,
                            "detail": str(exc),
                        }
                    )
                    continue
                if not ack.accepted:
                    continue
                replica = StageReplica(
                    stage_id=stage.stage_id,
                    worker_id=capability.worker_id,
                    shard_hash=shard_hash,
                    load_status="loaded",
                    warm=True,
                    measured_service_rate=rate,
                    health=HealthStatus.HEALTHY,
                    endpoint=capability.endpoint,
                )
                self.registry.add_replica(replica)
                self._assigned.add(key)
                self.events.append(
                    {
                        "event_type": "stage_assigned",
                        "worker_id": replica.worker_id,
                        "stage_id": replica.stage_id,
                        "predicted_service_rate": rate,
                        "marginal_basis": (
                            "required coverage"
                            if replica_counts[stage.stage_id] == 1
                            else "complete balanced replica round"
                        ),
                    }
                )
            for capability in remaining:
                self.events.append(
                    {
                        "event_type": "worker_left_idle",
                        "worker_id": capability.worker_id,
                        "detail": "incomplete replica round has non-positive immediate pipeline gain",
                    }
                )

    def stage_coverage(self) -> dict[int, int]:
        return {
            stage.stage_id: len(
                [
                    replica
                    for replica in self.registry.replicas(stage.stage_id)
                    if replica.health == HealthStatus.HEALTHY
                ]
            )
            for stage in self.stages
        }

    async def wait_for_coverage(
        self,
        *,
        minimum_replicas: int = 1,
        timeout_s: float = 30.0,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            await self.rebalance()
            if all(value >= minimum_replicas for value in self.stage_coverage().values()):
                return
            await asyncio.sleep(0.05)
        raise NoValidRouteError(
            f"stage coverage did not reach {minimum_replicas} replica(s): {self.stage_coverage()}"
        )

    def _choose_route(self) -> dict[int, StageReplica]:
        route: dict[int, StageReplica] = {}
        for stage in self.stages:
            replicas = sorted(
                [
                    replica
                    for replica in self.registry.replicas(stage.stage_id)
                    if replica.health == HealthStatus.HEALTHY and replica.endpoint is not None
                ],
                key=lambda replica: (
                    replica.queue_depth / max(replica.measured_service_rate, 1e-12),
                    -replica.reputation,
                    replica.worker_id,
                ),
            )
            if not replicas:
                raise NoValidRouteError(f"no healthy replica for stage {stage.stage_id}")
            route[stage.stage_id] = replicas[0]
        return route

    async def submit(self, submission: SubmitRequest) -> SubmitResponse:
        started = time.perf_counter()
        metrics = RuntimeRequestMetrics(
            request_id=submission.request_id,
            started_s=started,
        )
        if submission.prompt_token_ids:
            prompt_token_ids = submission.prompt_token_ids
        elif self.model_manifest is not None:
            if self.tokenizer is None:
                raise IntegrityError(
                    "real-model text submission requires a coordinator tokenizer; "
                    "provide prompt_token_ids or configure tokenizer_path"
                )
            encoded_prompt = self.tokenizer(submission.prompt or "", return_tensors=None)
            prompt_token_ids = [int(value) for value in encoded_prompt["input_ids"]]
        else:
            prompt_token_ids = [byte + 1 for byte in (submission.prompt or "").encode("utf-8")]
        if self.model_manifest is not None:
            if submission.model_id not in {"synthetic", self.runtime_model_id}:
                raise IntegrityError(
                    f"request model {submission.model_id!r} does not match {self.runtime_model_id!r}"
                )
            if submission.model_revision not in {
                "synthetic-v1",
                self.runtime_model_revision,
            }:
                raise IntegrityError(
                    "request model revision does not match the assigned immutable revision"
                )
        request_state = RequestState(
            request_id=submission.request_id,
            workload_class=WorkloadClass(submission.workload_class),
            prompt_token_ids=prompt_token_ids,
            sampling=SamplingConfig(
                temperature=0,
                max_new_tokens=submission.max_new_tokens,
            ),
            random_seed=submission.random_seed,
            status=RequestStatus.RUNNING,
        )
        route = self._choose_route()
        request_state.stage_route = [route[index].worker_id for index in range(len(self.stages))]
        outputs: list[int] = []
        try:
            for output_position in range(submission.max_new_tokens):
                operation = OperationKind.PREFILL if output_position == 0 else OperationKind.DECODE
                token_ids = prompt_token_ids if output_position == 0 else [outputs[-1]]
                if self.model_manifest is None:
                    token_position = output_position
                    activation = synthetic_activation(
                        token_ids,
                        hidden_size=self.config.model.hidden_size,
                        dtype=self.config.model.activation_dtype,
                    )
                else:
                    token_position = (
                        0
                        if operation == OperationKind.PREFILL
                        else len(prompt_token_ids) + output_position - 1
                    )
                    activation = np.asarray([token_ids], dtype=np.int64)
                for stage in self.stages:
                    replica = route[stage.stage_id]
                    tensor = ActivationTensor(
                        tensor_id=(f"{submission.request_id}:{output_position}:{stage.stage_id}"),
                        request_id=submission.request_id,
                        stage_id=stage.stage_id,
                        token_position=token_position,
                        sequence_length=len(token_ids),
                        array=activation,
                    )
                    encoded = encode_tensor(tensor)
                    self.replay.append(
                        request_id=submission.request_id,
                        model_revision=self.runtime_model_revision,
                        stage_id=stage.stage_id,
                        cache_generation=0,
                        token_position=token_position,
                        operation=operation,
                        payload=encoded,
                        recorded_monotonic_ns=time.monotonic_ns(),
                    )
                    result, replacement = await self._execute_with_recovery(
                        request=submission,
                        request_state=request_state,
                        metrics=metrics,
                        route=route,
                        replica=replica,
                        stage_id=stage.stage_id,
                        operation=operation,
                        token_position=token_position,
                        sequence_length=len(token_ids),
                        encoded=encoded,
                    )
                    if replacement is not None:
                        route[stage.stage_id] = replacement
                    activation = decode_tensor(result.tensor_payload).array
                if self.model_manifest is None:
                    digest = hashlib.sha256(
                        np.ascontiguousarray(activation).tobytes()
                        + submission.random_seed.to_bytes(8, "little", signed=True)
                        + output_position.to_bytes(8, "little")
                    ).digest()
                    token_id = int.from_bytes(digest[:4], "little") % 151_936
                else:
                    if (
                        activation.ndim != 3
                        or activation.shape[-1] != self.model_manifest.vocabulary_size
                    ):
                        raise IntegrityError(
                            "final real-model stage did not return [batch, sequence, vocabulary] logits"
                        )
                    token_id = int(np.argmax(activation[0, -1, :]))
                outputs.append(token_id)
                request_state.committed_output_tokens.append(token_id)
                request_state.current_token_position += 1
                if metrics.first_token_s is None:
                    metrics.first_token_s = time.perf_counter()
            request_state.status = RequestStatus.COMPLETED
            request_state.verification_state = VerificationState.VERIFIED
            metrics.completed_s = time.perf_counter()
            self.events.append(
                {
                    "event_type": "request_completed",
                    "request_id": submission.request_id,
                    "verified_tokens": len(outputs),
                }
            )
            return SubmitResponse(
                request_id=submission.request_id,
                output_token_ids=outputs,
                status="completed",
                verified=True,
                time_to_first_token_s=(
                    metrics.first_token_s - started if metrics.first_token_s else None
                ),
                end_to_end_s=metrics.completed_s - started,
            )
        except Exception as exc:
            request_state.status = RequestStatus.FAILED
            request_state.verification_state = VerificationState.REJECTED
            metrics.completed_s = time.perf_counter()
            self.events.append(
                {
                    "event_type": "request_failed",
                    "request_id": submission.request_id,
                    "detail": str(exc),
                }
            )
            return SubmitResponse(
                request_id=submission.request_id,
                output_token_ids=outputs,
                status="failed",
                verified=False,
                time_to_first_token_s=(
                    metrics.first_token_s - started if metrics.first_token_s else None
                ),
                end_to_end_s=metrics.completed_s - started,
                detail=str(exc),
            )
        finally:
            self.request_metrics.append(
                {
                    "request_id": metrics.request_id,
                    "time_to_first_token_s": (
                        metrics.first_token_s - metrics.started_s
                        if metrics.first_token_s is not None
                        else None
                    ),
                    "end_to_end_s": (
                        metrics.completed_s - metrics.started_s
                        if metrics.completed_s is not None
                        else None
                    ),
                    "stage_execution_s": metrics.stage_execution_s,
                    "queue_s": metrics.queue_s,
                    "transport_s": metrics.transport_s,
                    "replay_s": metrics.replay_s,
                    "replay_bytes": metrics.replay_bytes,
                    "retry_count": metrics.retries,
                    "route_changes": metrics.route_changes,
                    "per_stage": metrics.per_stage,
                }
            )
            await self._cancel_all(submission.request_id, self.runtime_model_revision)

    async def _call_replica(
        self,
        *,
        replica: StageReplica,
        submission: SubmitRequest,
        stage_id: int,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        encoded: bytes,
        audit: bool = False,
    ) -> tuple[ActivationResult, float]:
        if replica.endpoint is None:
            raise TransportError(f"worker {replica.worker_id} has no endpoint")
        request = ActivationRequest(
            metadata=ActivationMetadata(
                request_id=submission.request_id,
                tensor_id=f"{submission.request_id}:{token_position}:{stage_id}",
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                cache_generation=0,
                model_id=submission.model_id,
                model_revision=submission.model_revision,
                audit=audit,
            ),
            tensor_payload=encoded,
        )
        started = time.perf_counter()
        result = await self.transport.execute(replica.endpoint, request)
        elapsed = time.perf_counter() - started
        if sha256_bytes(result.tensor_payload) != result.checksum:
            raise IntegrityError(f"result checksum mismatch from worker {result.worker_id}")
        capability = self.registry.capability(result.worker_id)
        signed_payload = canonical_json_bytes(
            {
                "worker_id": result.worker_id,
                "request_id": result.metadata.request_id,
                "stage_id": result.metadata.stage_id,
                "token_position": result.metadata.token_position,
                "checksum": result.checksum,
            }
        )
        verify_signature(capability.public_key, signed_payload, result.signature)
        # The tensor decoder verifies the internal activation checksum and IDs.
        decoded = decode_tensor(result.tensor_payload)
        if (
            decoded.request_id != submission.request_id
            or decoded.stage_id != stage_id
            or decoded.token_position != token_position
        ):
            raise IntegrityError("worker result metadata does not match stage request")
        return result, elapsed

    async def _execute_with_recovery(
        self,
        *,
        request: SubmitRequest,
        request_state: RequestState,
        metrics: RuntimeRequestMetrics,
        route: dict[int, StageReplica],
        replica: StageReplica,
        stage_id: int,
        operation: OperationKind,
        token_position: int,
        sequence_length: int,
        encoded: bytes,
    ) -> tuple[ActivationResult, StageReplica | None]:
        try:
            result, elapsed = await self._call_replica(
                replica=replica,
                submission=request,
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                encoded=encoded,
            )
            metrics.stage_execution_s += result.execution_ms / 1000
            metrics.queue_s += result.queue_ms / 1000
            metrics.transport_s += max(
                0.0, elapsed - (result.execution_ms + result.queue_ms) / 1000
            )
            metrics.per_stage.append(
                {
                    "stage_id": stage_id,
                    "worker_id": replica.worker_id,
                    "token_position": token_position,
                    "execution_ms": result.execution_ms,
                    "queue_ms": result.queue_ms,
                    "transport_elapsed_s": elapsed,
                    "activation_bytes_sent": len(encoded),
                    "activation_bytes_received": len(result.tensor_payload),
                }
            )
            return result, None
        except Exception as original:
            alternatives = sorted(
                [
                    candidate
                    for candidate in self.registry.replicas(stage_id)
                    if candidate.worker_id != replica.worker_id
                    and candidate.health == HealthStatus.HEALTHY
                    and candidate.endpoint is not None
                ],
                key=lambda candidate: candidate.worker_id,
            )
            if not alternatives:
                raise TransportError(
                    f"stage {stage_id} worker {replica.worker_id} failed and no backup "
                    f"replica is available: {original}"
                ) from original
            replacement = alternatives[0]
            replay_started = time.perf_counter()
            replay_bytes = 0
            if token_position > 0:
                try:
                    entries = self.replay.entries_for(
                        request_id=request.request_id,
                        model_revision=request.model_revision,
                        stage_id=stage_id,
                        cache_generation=0,
                        through_token_position=token_position - 1,
                    )
                except ReplayUnavailableError:
                    entries = []
                for entry in entries:
                    replay_bytes += len(entry.payload)
                    replay_sequence = decode_tensor(entry.payload).sequence_length
                    await self._call_replica(
                        replica=replacement,
                        submission=request,
                        stage_id=stage_id,
                        operation=entry.operation,
                        token_position=entry.token_position,
                        sequence_length=replay_sequence,
                        encoded=entry.payload,
                    )
            replay_elapsed = time.perf_counter() - replay_started
            metrics.replay_s += replay_elapsed
            metrics.replay_bytes += replay_bytes
            metrics.retries += 1
            metrics.route_changes += 1
            request_state.retry_count += 1
            self.events.append(
                {
                    "event_type": "stage_recovered",
                    "request_id": request.request_id,
                    "stage_id": stage_id,
                    "failed_worker_id": replica.worker_id,
                    "replacement_worker_id": replacement.worker_id,
                    "replay_bytes": replay_bytes,
                    "replay_duration_s": replay_elapsed,
                    "failure": str(original),
                }
            )
            result, elapsed = await self._call_replica(
                replica=replacement,
                submission=request,
                stage_id=stage_id,
                operation=operation,
                token_position=token_position,
                sequence_length=sequence_length,
                encoded=encoded,
            )
            metrics.stage_execution_s += result.execution_ms / 1000
            metrics.queue_s += result.queue_ms / 1000
            metrics.transport_s += max(
                0.0, elapsed - (result.execution_ms + result.queue_ms) / 1000
            )
            metrics.per_stage.append(
                {
                    "stage_id": stage_id,
                    "worker_id": replacement.worker_id,
                    "token_position": token_position,
                    "execution_ms": result.execution_ms,
                    "queue_ms": result.queue_ms,
                    "transport_elapsed_s": elapsed,
                    "activation_bytes_sent": len(encoded),
                    "activation_bytes_received": len(result.tensor_payload),
                    "recovered": True,
                }
            )
            return result, replacement

    async def _cancel_all(self, request_id: str, model_revision: str) -> None:
        from swarm_inference.protocol.messages import CancelRequest

        endpoints = {
            replica.endpoint for replica in self.registry.replicas() if replica.endpoint is not None
        }
        await asyncio.gather(
            *(
                self.transport.cancel(
                    endpoint,
                    CancelRequest(
                        request_id=request_id,
                        model_revision=model_revision,
                    ),
                )
                for endpoint in endpoints
            ),
            return_exceptions=True,
        )
        self.replay.delete_request(request_id)


class CoordinatorRpcServer:
    def __init__(
        self,
        core: CoordinatorCore,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.core = core
        self.server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ]
        )
        handlers = {
            "Register": grpc.unary_unary_rpc_method_handler(
                self._register,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Heartbeat": grpc.unary_unary_rpc_method_handler(
                self._heartbeat,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
            "Submit": grpc.unary_unary_rpc_method_handler(
                self._submit,
                request_deserializer=lambda value: value,
                response_serializer=lambda value: value,
            ),
        }
        self.server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler("swarm.v1.Coordinator", handlers),)
        )
        self.bound_port: int | None = None

    async def start(self, endpoint: str) -> int:
        self.bound_port = self.server.add_insecure_port(endpoint)
        if self.bound_port == 0:
            raise TransportError(f"could not bind coordinator endpoint {endpoint}")
        await self.server.start()
        return self.bound_port

    async def stop(self, grace_s: float = 2.0) -> None:
        await self.server.stop(grace_s)
        await self.core.close()

    async def wait_for_termination(self) -> None:
        await self.server.wait_for_termination()

    async def _register(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            response = await self.core.register(parse_message(data, RegistrationRequest))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise

    async def _heartbeat(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        try:
            response = await self.core.heartbeat(parse_message(data, Heartbeat))
            return serialize_message(response)
        except Exception as exc:
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))
            raise

    async def _submit(self, data: bytes, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        response = await self.core.submit(parse_message(data, SubmitRequest))
        return serialize_message(response)


class CoordinatorClient:
    def __init__(
        self,
        endpoint: str,
        *,
        maximum_message_bytes: int = 4 * 1024 * 1024,
        timeout_s: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", maximum_message_bytes),
                ("grpc.max_receive_message_length", maximum_message_bytes),
            ],
        )

    async def _call(
        self,
        path: str,
        request: StrictModel,
        response_type: type[ResponseT],
    ) -> ResponseT:
        call = self.channel.unary_unary(
            path,
            request_serializer=lambda value: value,
            response_deserializer=lambda value: value,
        )
        try:
            data = await call(serialize_message(request), timeout=self.timeout_s)
            return parse_message(data, response_type)
        except grpc.aio.AioRpcError as exc:
            raise TransportError(
                f"coordinator RPC {path} failed ({exc.code().name}): {exc.details()}"
            ) from exc

    async def register(self, request: RegistrationRequest) -> RegistrationResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Register",
            request,
            RegistrationResponse,
        )

    async def heartbeat(self, request: Heartbeat) -> Ack:
        return await self._call("/swarm.v1.Coordinator/Heartbeat", request, Ack)

    async def submit(self, request: SubmitRequest) -> SubmitResponse:
        return await self._call(
            "/swarm.v1.Coordinator/Submit",
            request,
            SubmitResponse,
        )

    async def close(self) -> None:
        await self.channel.close()
