"""Quantised llama.cpp CPU adapter wrapped by the Universal Worker ABI."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
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


class LlamaCppAdapter(BackendAdapter):
    backend_id = "llamacpp"
    supported_jobs = frozenset(
        {
            WorkerJobType.SPECULATIVE_DRAFT,
            WorkerJobType.BACKGROUND_GENERATE,
        }
    )

    def __init__(
        self,
        *,
        endpoint: str,
        capabilities: WorkerCapabilities,
        model_revision: str,
        tokenizer_hash: str,
        gguf_hash: str,
        weight_format: str,
        transport: JsonTransport = post_json,
        model_load_seconds: float = 0.0,
        warmup_seconds: float = 0.0,
    ) -> None:
        self.endpoint = endpoint
        self._capabilities = capabilities
        self.model_revision = model_revision
        self.tokenizer_hash = tokenizer_hash
        self.gguf_hash = gguf_hash
        self.weight_format = weight_format
        self.transport = transport
        self._profile = WorkerBenchmarkProfile(
            model_revision=model_revision,
            shard_hash=gguf_hash,
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
            return self._rejected(job, "loaded GGUF revision does not match the job revision")
        if not isinstance(job.input_payload, TokenPayload):
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.INCOMPATIBLE_DTYPE,
                detail="llama.cpp generation requires a token payload",
            )
        if job.input_payload.tokenizer_hash != self.tokenizer_hash:
            return self._rejected(job, "tokenizer identity is absent or incompatible")
        generation = job.generation_parameters
        if generation is None:
            return self._rejected(job, "generation parameters are required")
        timeout_seconds = max(0.001, job.remaining_deadline_ms / 1000)
        request = {
            "prompt": job.input_payload.token_ids,
            "n_predict": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "top_k": generation.top_k,
            "seed": generation.seed,
            "ignore_eos": generation.ignore_eos,
            "cache_prompt": True,
            "n_probs": 1,
            "return_tokens": True,
            "id_slot": -1,
        }
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.transport,
                    self.endpoint,
                    "/completion",
                    request,
                    timeout_seconds,
                ),
                timeout=timeout_seconds,
            )
            output_ids = await self._extract_output_ids(response, timeout_seconds)
        except TimeoutError:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.DEADLINE_IMPOSSIBLE,
                detail="llama.cpp request exceeded its deadline",
            )
        except Exception as exc:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.BACKEND_FAILURE,
                detail=f"llama.cpp request failed: {type(exc).__name__}: {exc}",
            )
        elapsed = time.perf_counter() - started
        if job.request_id in self._cancelled:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.CANCELLED,
            )
        rate = len(output_ids) / elapsed if elapsed else 0.0
        if job.role == WorkerJobType.SPECULATIVE_DRAFT:
            self._profile.draft_tokens_per_second = rate
        else:
            self._profile.background_tokens_per_second = rate
        timings = response.get("timings", {})
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.ACCEPTED,
            output_payload=TokenPayload(
                token_ids=output_ids,
                tokenizer_hash=self.tokenizer_hash,
            ),
            metrics={
                "elapsed_ms": elapsed * 1000,
                "output_tokens_per_second": rate,
                "timings": timings if isinstance(timings, dict) else {},
                "gguf_hash": self.gguf_hash,
                "weight_format": self.weight_format,
                "backend_id": self.backend_id,
            },
            classification=ResultClassification.MEASURED_X86_CPU,
        )

    def _rejected(self, job: WorkerJob, detail: str) -> WorkerJobResult:
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.UNSUPPORTED,
            detail=detail,
        )

    async def _extract_output_ids(
        self,
        response: dict[str, Any],
        timeout_seconds: float,
    ) -> list[int]:
        for key in ("tokens", "token_ids", "output_ids"):
            direct = response.get(key)
            if isinstance(direct, list) and all(isinstance(item, int) for item in direct):
                return [int(item) for item in direct]
        probabilities = response.get("completion_probabilities")
        if isinstance(probabilities, list):
            ids: list[int] = []
            complete = True
            for item in probabilities:
                if not isinstance(item, dict) or not isinstance(item.get("id"), int):
                    complete = False
                    break
                ids.append(int(item["id"]))
            if complete and ids:
                return ids
        content = response.get("content")
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp response did not expose token IDs or content")
        tokenized = await asyncio.to_thread(
            self.transport,
            self.endpoint,
            "/tokenize",
            {"content": content, "add_special": False},
            timeout_seconds,
        )
        tokens = tokenized.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
            raise RuntimeError("llama.cpp /tokenize did not expose token IDs")
        return [int(item) for item in tokens]

    async def cancel(self, request_id: str) -> bool:
        self._cancelled.add(request_id)
        return True
