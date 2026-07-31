"""Bounded asynchronous stage execution queue."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from swarm_inference.config.models import BackpressurePolicy, QueueConfig
from swarm_inference.exceptions import BackpressureError
from swarm_inference.experiments.fanout_lifecycle import lifecycle_recorder
from swarm_inference.model.stage_module import (
    BatchExecutionMetadata,
    BatchStageModule,
    StageExecutionMetadata,
)
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import ActivationRequest, ActivationResult
from swarm_inference.protocol.tensor_codec import decode_tensor, encode_tensor
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.security.signatures import canonical_json_bytes
from swarm_inference.worker.metrics import WorkerMetrics
from swarm_inference.worker.shard_manager import ShardManager


@dataclass(slots=True)
class QueuedExecution:
    request: ActivationRequest
    enqueued_at: float
    future: asyncio.Future[ActivationResult]


class ExecutionEngine:
    """One bounded queue; correctness does not depend on microbatching."""

    def __init__(
        self,
        *,
        worker_id: str,
        identity: WorkerIdentity,
        shards: ShardManager,
        queue_config: QueueConfig,
        metrics: WorkerMetrics,
    ) -> None:
        self.worker_id = worker_id
        self.identity = identity
        self.shards = shards
        self.queue_config = queue_config
        self.metrics = metrics
        self._queue: asyncio.Queue[QueuedExecution] = asyncio.Queue(maxsize=queue_config.capacity)
        self._runner: asyncio.Task[None] | None = None
        self._stopping = False

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(self._run(), name=f"execute:{self.worker_id}")

    async def stop(self) -> None:
        self._stopping = True
        if self._runner is not None:
            self._runner.cancel()
            with suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None

    async def submit(self, request: ActivationRequest) -> ActivationResult:
        if self._runner is None:
            await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ActivationResult] = loop.create_future()
        item = QueuedExecution(
            request=request,
            enqueued_at=time.perf_counter(),
            future=future,
        )
        if (
            self.queue_config.backpressure_policy == BackpressurePolicy.REJECT
            and self._queue.full()
        ):
            raise BackpressureError(
                f"worker {self.worker_id} queue capacity {self._queue.maxsize} reached"
            )
        try:
            await asyncio.wait_for(
                self._queue.put(item),
                timeout=self.queue_config.request_deadline_ms / 1000,
            )
            return await asyncio.wait_for(
                future,
                timeout=self.queue_config.request_deadline_ms / 1000,
            )
        except TimeoutError as exc:
            if not future.done():
                future.cancel()
            raise BackpressureError("stage operation deadline exceeded") from exc

    async def _run(self) -> None:
        while not self._stopping:
            first = await self._queue.get()
            batch = [first]
            maximum = self.queue_config.max_microbatch_size
            wait_s = self.queue_config.max_microbatch_wait_ms / 1000
            if maximum > 1 and wait_s > 0:
                deadline = asyncio.get_running_loop().time() + wait_s
                while len(batch) < maximum:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        candidate = await asyncio.wait_for(self._queue.get(), remaining)
                    except TimeoutError:
                        break
                    if self._compatible(first.request, candidate.request):
                        batch.append(candidate)
                    else:
                        # Preserve boundedness and ordering. This branch is rare
                        # because coordinators route a stage queue homogeneously.
                        self._queue.task_done()
                        if not candidate.future.done():
                            candidate.future.set_exception(
                                BackpressureError("incompatible operation reached microbatch")
                            )
            if len(batch) > 1:
                module = self.shards.module(first.request.metadata.stage_id)
                if isinstance(module, BatchStageModule):
                    try:
                        results = self._execute_batch(batch, module=module)
                    except Exception as exc:
                        self.metrics.failures += len(batch)
                        for item in batch:
                            if not item.future.done():
                                item.future.set_exception(exc)
                            self._queue.task_done()
                    else:
                        for item, result in zip(batch, results, strict=True):
                            if not item.future.done():
                                item.future.set_result(result)
                            self._queue.task_done()
                    await asyncio.sleep(0)
                    continue
            for item in batch:
                try:
                    result = self._execute_one(item)
                except Exception as exc:
                    self.metrics.failures += 1
                    if not item.future.done():
                        item.future.set_exception(exc)
                else:
                    if not item.future.done():
                        item.future.set_result(result)
                finally:
                    self._queue.task_done()
                await asyncio.sleep(0)

    @staticmethod
    def _compatible(left: ActivationRequest, right: ActivationRequest) -> bool:
        first = left.metadata
        second = right.metadata
        return (
            first.model_revision == second.model_revision
            and first.stage_id == second.stage_id
            and first.operation == second.operation
            and first.sequence_length == second.sequence_length
            and first.token_position == second.token_position
            and first.cache_generation == second.cache_generation
            and first.route_generation == second.route_generation
        )

    def _execute_batch(
        self,
        items: list[QueuedExecution],
        *,
        module: BatchStageModule,
    ) -> list[ActivationResult]:
        decoded = [decode_tensor(item.request.tensor_payload) for item in items]
        for item, activation in zip(items, decoded, strict=True):
            if activation.request_id != item.request.metadata.request_id:
                raise BackpressureError("activation request ID does not match metadata")
            if activation.stage_id != item.request.metadata.stage_id:
                raise BackpressureError("activation stage ID does not match metadata")
        first_metadata = items[0].request.metadata
        batch_metadata = BatchExecutionMetadata(
            requests=tuple(
                StageExecutionMetadata(
                    request_id=item.request.metadata.request_id,
                    token_position=item.request.metadata.token_position,
                    sequence_length=item.request.metadata.sequence_length,
                    cache_generation=item.request.metadata.cache_generation,
                    route_generation=item.request.metadata.route_generation,
                )
                for item in items
            )
        )
        activation_batch = np.concatenate(
            [activation.array for activation in decoded],
            axis=0,
        )
        started = time.perf_counter()
        output_batch = module.execute_batch(
            activation_batch,
            metadata=batch_metadata,
            operation=first_metadata.operation,
        )
        elapsed = time.perf_counter() - started
        if int(output_batch.shape[0]) != len(items):
            raise BackpressureError(
                f"batched stage returned batch {int(output_batch.shape[0])}, expected {len(items)}"
            )
        recorder = lifecycle_recorder()
        if recorder is not None:
            recorder.emit(
                "real_batch_forward_completed",
                duration_ns=int(elapsed * 1_000_000_000),
                details={
                    "execution_profile": getattr(
                        module,
                        "execution_profile",
                        "unknown",
                    ),
                    "batch_size": len(items),
                    "request_ids": batch_metadata.request_ids,
                    "operation": first_metadata.operation.value,
                },
            )
        return [
            self._encode_result(
                item,
                activation,
                output_batch[index : index + 1],
                elapsed=elapsed,
                started=started,
            )
            for index, (item, activation) in enumerate(zip(items, decoded, strict=True))
        ]

    def _execute_one(self, item: QueuedExecution) -> ActivationResult:
        request = item.request
        activation = decode_tensor(request.tensor_payload)
        if activation.request_id != request.metadata.request_id:
            raise BackpressureError("activation request ID does not match metadata")
        if activation.stage_id != request.metadata.stage_id:
            raise BackpressureError("activation stage ID does not match metadata")
        module = self.shards.module(request.metadata.stage_id)
        recorder = lifecycle_recorder()
        operation_started_ns = time.monotonic_ns()
        if recorder is not None:
            operation_details = {
                "execution_profile": getattr(
                    module,
                    "execution_profile",
                    "qwen3_correctness",
                ),
                "request_id": request.metadata.request_id,
                "operation": request.metadata.operation.value,
                "token_position": request.metadata.token_position,
                "sequence_length": request.metadata.sequence_length,
                "route_generation": request.metadata.route_generation,
            }
            recorder.emit_once(
                "first-stage-operation-started",
                "first_stage_operation_started",
                monotonic_ns=operation_started_ns,
                details=operation_details,
            )
            recorder.emit_once(
                f"request-stage-started:{request.metadata.request_id}",
                "request_stage_operation_started",
                monotonic_ns=operation_started_ns,
                details=operation_details,
            )
        started = time.perf_counter()
        output_array = module.execute(
            activation.array,
            request_id=request.metadata.request_id,
            operation=request.metadata.operation,
            token_position=request.metadata.token_position,
            sequence_length=request.metadata.sequence_length,
            cache_generation=request.metadata.cache_generation,
            route_generation=request.metadata.route_generation,
        )
        elapsed = time.perf_counter() - started
        operation_completed_ns = time.monotonic_ns()
        if recorder is not None:
            completion_details = {
                "execution_profile": getattr(
                    module,
                    "execution_profile",
                    "qwen3_correctness",
                ),
                "request_id": request.metadata.request_id,
                "operation": request.metadata.operation.value,
                "token_position": request.metadata.token_position,
                "sequence_length": request.metadata.sequence_length,
                "route_generation": request.metadata.route_generation,
            }
            recorder.emit_once(
                "first-stage-operation-completed",
                "first_stage_operation_completed",
                monotonic_ns=operation_completed_ns,
                duration_ns=operation_completed_ns - operation_started_ns,
                details=completion_details,
            )
            recorder.emit_once(
                f"request-stage-completed:{request.metadata.request_id}",
                "request_stage_operation_completed",
                monotonic_ns=operation_completed_ns,
                duration_ns=operation_completed_ns - operation_started_ns,
                details=completion_details,
            )
        return self._encode_result(
            item,
            activation,
            output_array,
            elapsed=elapsed,
            started=started,
        )

    def _encode_result(
        self,
        item: QueuedExecution,
        activation: Any,
        output_array: np.ndarray,
        *,
        elapsed: float,
        started: float,
    ) -> ActivationResult:
        request = item.request
        output = encode_tensor(
            type(activation)(
                tensor_id=f"{activation.tensor_id}:out",
                request_id=activation.request_id,
                stage_id=activation.stage_id,
                token_position=activation.token_position,
                sequence_length=activation.sequence_length,
                array=output_array,
                logical_dtype=("bfloat16" if output_array.dtype == np.dtype(np.uint16) else None),
            )
        )
        checksum = sha256_bytes(output)
        signed = canonical_json_bytes(
            {
                "worker_id": self.worker_id,
                "request_id": request.metadata.request_id,
                "stage_id": request.metadata.stage_id,
                "token_position": request.metadata.token_position,
                "checksum": checksum,
            }
        )
        queue_ms = (started - item.enqueued_at) * 1000
        self.metrics.record_success(
            received_bytes=len(request.tensor_payload),
            sent_bytes=len(output),
            service_s=elapsed,
            queue_depth=self.queue_depth,
        )
        return ActivationResult(
            metadata=request.metadata,
            tensor_payload=output,
            worker_id=self.worker_id,
            execution_ms=elapsed * 1000,
            queue_ms=max(0.0, queue_ms),
            checksum=checksum,
            signature=self.identity.sign(signed),
        )
