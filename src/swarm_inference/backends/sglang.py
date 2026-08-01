"""Production CUDA target adapter for an isolated SGLang service."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from swarm_inference.backends.http import post_json
from swarm_inference.worker.abi import (
    BackendAdapter,
    ResultClassification,
    TokenPayload,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
)

JsonTransport = Callable[[str, str, dict[str, Any], float], dict[str, Any]]


class SGLangAdapter(BackendAdapter):
    backend_id = "sglang"
    supported_jobs = frozenset(
        {
            WorkerJobType.TARGET_PREFILL,
            WorkerJobType.TARGET_DECODE,
            WorkerJobType.BACKGROUND_GENERATE,
        }
    )

    def __init__(
        self,
        *,
        endpoint: str,
        capabilities: WorkerCapabilities,
        model_revision: str,
        transport: JsonTransport = post_json,
        model_load_seconds: float = 0.0,
        warmup_seconds: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self._capabilities = capabilities
        self.model_revision = model_revision
        self.transport = transport
        self._profile = WorkerBenchmarkProfile(
            model_revision=model_revision,
            model_load_seconds=model_load_seconds,
            warmup_seconds=warmup_seconds,
        )
        self._cancelled: set[str] = set()

    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return self._profile

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        rejected = self.admission_result(job)
        if rejected is not None:
            return rejected
        if job.model_revision != self.model_revision:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.UNSUPPORTED,
                detail=(
                    f"loaded revision {self.model_revision} does not match "
                    f"job revision {job.model_revision}"
                ),
            )
        if not isinstance(job.input_payload, TokenPayload):
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.INCOMPATIBLE_DTYPE,
                detail="SGLang target jobs require a token payload",
            )
        generation = job.generation_parameters
        if generation is None:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.UNSUPPORTED,
                detail="generation parameters are required",
            )
        timeout_seconds = max(0.001, job.remaining_deadline_ms / 1000)
        request = {
            "input_ids": job.input_payload.token_ids,
            "sampling_params": {
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "top_k": generation.top_k,
                "max_new_tokens": generation.max_new_tokens,
                "ignore_eos": generation.ignore_eos,
            },
            "stream": False,
            "rid": job.request_id,
        }
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.transport,
                    self.endpoint,
                    "/generate",
                    request,
                    timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.DEADLINE_IMPOSSIBLE,
                detail="SGLang target request exceeded its deadline",
            )
        except Exception as exc:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.BACKEND_FAILURE,
                detail=f"SGLang request failed: {type(exc).__name__}: {exc}",
            )
        elapsed = time.perf_counter() - started
        if job.request_id in self._cancelled:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.CANCELLED,
            )
        output_ids = _sglang_output_ids(response)
        rate = len(output_ids) / elapsed if elapsed else 0.0
        self._profile.decode_tokens_per_second = rate
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.ACCEPTED,
            output_payload=TokenPayload(
                token_ids=output_ids,
                tokenizer_hash=job.input_payload.tokenizer_hash,
            ),
            metrics={
                "elapsed_ms": elapsed * 1000,
                "output_tokens_per_second": rate,
                "meta_info": response.get("meta_info", {}),
                "backend_id": self.backend_id,
            },
            classification=ResultClassification.MEASURED_CUDA,
        )

    async def cancel(self, request_id: str) -> bool:
        self._cancelled.add(request_id)
        with suppress(Exception):
            await asyncio.to_thread(
                self.transport,
                self.endpoint,
                "/abort_request",
                {"rid": request_id},
                5.0,
            )
        # Cancellation remains locally authoritative; an unreachable target is
        # reported by the in-flight job rather than replaced by a fallback.
        return True


def _sglang_output_ids(response: dict[str, Any]) -> list[int]:
    direct = response.get("output_ids")
    if isinstance(direct, list):
        return [int(value) for value in direct]
    meta = response.get("meta_info")
    if isinstance(meta, dict):
        for key in ("output_ids", "completion_token_ids"):
            value = meta.get(key)
            if isinstance(value, list):
                return [int(item) for item in value]
    raise RuntimeError("SGLang response did not expose output token IDs")
