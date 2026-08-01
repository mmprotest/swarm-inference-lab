"""Real four-stage CUDA/CPU pipeline over Universal Worker jobs."""

from __future__ import annotations

import base64
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from swarm_inference.worker.abi import (
    BackendAdapter,
    TensorPayload,
    TokenPayload,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
)


class JobClient(Protocol):
    async def submit(self, job: WorkerJob) -> WorkerJobResult: ...


RankEndpoint = BackendAdapter | JobClient
RecoveryFactory = Callable[[int, int], Awaitable[RankEndpoint]]


class BackendStageError(RuntimeError):
    def __init__(self, stage_id: int, result: WorkerJobResult) -> None:
        super().__init__(f"stage {stage_id}: {result.status.value}: {result.detail}")
        self.stage_id = stage_id
        self.result = result


@dataclass(slots=True)
class MixedPipelineResult:
    output_token_ids: list[int]
    metrics: dict[str, Any]
    boundary_metrics: list[dict[str, Any]]
    recovery_events: list[dict[str, Any]]


async def _submit(endpoint: RankEndpoint, job: WorkerJob) -> WorkerJobResult:
    if isinstance(endpoint, BackendAdapter):
        return await endpoint.execute(job)
    return await endpoint.submit(job)


class MixedPipelineCoordinator:
    def __init__(
        self,
        *,
        endpoints: list[RankEndpoint],
        model_id: str,
        model_revision: str,
        partition_hash: str,
        shard_hashes: list[str],
        recovery_factory: RecoveryFactory | None = None,
        deadline_ms: int = 300_000,
    ) -> None:
        if len(endpoints) != 4 or len(shard_hashes) != 4:
            raise ValueError("Experiment 007 mixed route requires exactly four ranks")
        self.endpoints = endpoints
        self.model_id = model_id
        self.model_revision = model_revision
        self.partition_hash = partition_hash
        self.shard_hashes = shard_hashes
        self.recovery_factory = recovery_factory
        self.deadline_ms = deadline_ms
        self.route_generation = 1

    async def _operation(
        self,
        *,
        request_id: str,
        input_token_ids: list[int],
        token_position: int,
        prefill: bool,
    ) -> tuple[int, list[dict[str, Any]]]:
        payload: TokenPayload | TensorPayload = TokenPayload(token_ids=input_token_ids)
        boundary_rows: list[dict[str, Any]] = []
        role = (
            WorkerJobType.PIPELINE_STAGE_PREFILL if prefill else WorkerJobType.PIPELINE_STAGE_DECODE
        )
        for stage_id, endpoint in enumerate(self.endpoints):
            serialization_started = time.perf_counter()
            job = WorkerJob(
                job_id=uuid4().hex,
                request_id=request_id,
                role=role,
                model_id=self.model_id,
                model_revision=self.model_revision,
                partition_manifest_hash=self.partition_hash,
                shard_hash=self.shard_hashes[stage_id],
                input_payload=payload,
                deadline_ms=self.deadline_ms,
                priority=100,
                route_generation=self.route_generation,
                metadata={
                    "token_position": token_position,
                    "cache_generation": 0,
                    "stage_id": stage_id,
                },
            )
            serialisation_ms = (time.perf_counter() - serialization_started) * 1000
            submitted = time.perf_counter()
            result = await _submit(endpoint, job)
            round_trip_ms = (time.perf_counter() - submitted) * 1000
            if result.status != WorkerJobStatus.ACCEPTED:
                raise BackendStageError(stage_id, result)
            if not isinstance(result.output_payload, TensorPayload):
                raise RuntimeError(f"stage {stage_id} returned a non-tensor payload")
            encoded_bytes = len(base64.b64decode(result.output_payload.data_base64))
            boundary_rows.append(
                {
                    "request_id": request_id,
                    "route_generation": self.route_generation,
                    "operation": "prefill" if prefill else "decode",
                    "stage_id": stage_id,
                    "source_backend": ("coordinator" if stage_id == 0 else f"stage-{stage_id - 1}"),
                    "destination_backend": f"stage-{stage_id}",
                    "payload_bytes": encoded_bytes,
                    "serialisation_ms": serialisation_ms,
                    "round_trip_ms": round_trip_ms,
                    "execution_ms": float(result.metrics.get("execution_ms", 0.0)),
                    "cache_bytes": int(result.metrics.get("cache_bytes", 0)),
                    "input_logical_dtype": result.metrics.get("input_logical_dtype"),
                    "output_logical_dtype": result.metrics.get("output_logical_dtype"),
                    "synthetic_fallback": bool(result.metrics.get("synthetic_fallback", True)),
                }
            )
            payload = result.output_payload
        if not isinstance(payload, TensorPayload):
            raise RuntimeError("mixed pipeline did not produce final tensor logits")
        tensor = payload.to_tensor()
        logits = tensor.array
        if logits.dtype == np.uint16:
            raise RuntimeError(
                "final mixed-pipeline stage returned hidden states instead of logits"
            )
        sampled = int(np.argmax(logits[0, -1, :]))
        return sampled, boundary_rows

    async def generate(
        self,
        *,
        request_id: str,
        prompt_token_ids: list[int],
        output_tokens: int,
        reference_token_ids: list[int] | None = None,
    ) -> MixedPipelineResult:
        if not prompt_token_ids or output_tokens <= 0:
            raise ValueError("mixed pipeline requires a prompt and positive output token count")
        started = time.perf_counter()
        output: list[int] = []
        boundary_rows: list[dict[str, Any]] = []
        recovery_events: list[dict[str, Any]] = []
        prefill_seconds = 0.0
        decode_seconds = 0.0
        attempts = 0
        while True:
            try:
                prefill_started = time.perf_counter()
                first, rows = await self._operation(
                    request_id=request_id,
                    input_token_ids=prompt_token_ids,
                    token_position=0,
                    prefill=True,
                )
                prefill_seconds += time.perf_counter() - prefill_started
                output = [first]
                boundary_rows.extend(rows)
                while len(output) < output_tokens:
                    decode_started = time.perf_counter()
                    token, rows = await self._operation(
                        request_id=request_id,
                        input_token_ids=[output[-1]],
                        token_position=len(prompt_token_ids) + len(output) - 1,
                        prefill=False,
                    )
                    decode_seconds += time.perf_counter() - decode_started
                    output.append(token)
                    boundary_rows.extend(rows)
                break
            except BackendStageError as exc:
                if self.recovery_factory is None or attempts >= 1:
                    raise
                old_generation = self.route_generation
                self.route_generation += 1
                self.endpoints[exc.stage_id] = await self.recovery_factory(
                    exc.stage_id, self.route_generation
                )
                recovery_events.append(
                    {
                        "stage_id": exc.stage_id,
                        "failed_status": exc.result.status.value,
                        "old_route_generation": old_generation,
                        "new_route_generation": self.route_generation,
                        "replay_tokens": len(prompt_token_ids) + len(output),
                        "explicit_same_backend_replacement": True,
                    }
                )
                output = []
                boundary_rows = []
                attempts += 1
        elapsed = time.perf_counter() - started
        identity = reference_token_ids is None or output == reference_token_ids[: len(output)]
        first_mismatch = None
        if reference_token_ids is not None and not identity:
            for index, (actual, expected) in enumerate(
                zip(output, reference_token_ids, strict=False)
            ):
                if actual != expected:
                    first_mismatch = {"index": index, "actual": actual, "expected": expected}
                    break
        return MixedPipelineResult(
            output_token_ids=output,
            metrics={
                "classification": "measured_mixed_backend",
                "request_id": request_id,
                "prompt_tokens": len(prompt_token_ids),
                "output_tokens": len(output),
                "end_to_end_seconds": elapsed,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "output_tokens_per_second": len(output) / max(elapsed, 1e-12),
                "time_to_first_token_seconds": prefill_seconds,
                "exact_greedy_token_identity": identity,
                "first_mismatch": first_mismatch,
                "route_generation": self.route_generation,
                "stage_local_kv_cache": True,
                "synthetic_execution": any(
                    bool(row["synthetic_fallback"]) for row in boundary_rows
                ),
            },
            boundary_metrics=boundary_rows,
            recovery_events=recovery_events,
        )

    async def cancel(self, request_id: str) -> None:
        for endpoint in self.endpoints:
            if isinstance(endpoint, BackendAdapter):
                await endpoint.cancel(request_id)
            else:
                await endpoint.cancel(request_id)  # type: ignore[attr-defined]
