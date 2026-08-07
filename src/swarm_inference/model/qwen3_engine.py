"""Monolithic and grouped-stage execution engine for Experiment 004."""

from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swarm_inference.model.qwen3 import Qwen3StageModule
from swarm_inference.model.qwen3_runtime import (
    AttentionBackend,
    Qwen3EngineOptions,
    Qwen3ExecutionProfile,
    auto_attention_candidates,
    nvtx_range,
)
from swarm_inference.model.qwen3_sampling import SamplingParameters
from swarm_inference.model.shard_builder import (
    ResolvedModel,
    build_manifest,
    inspect_qwen3_model,
)
from swarm_inference.model.stage_module import (
    BatchExecutionMetadata,
    StageExecutionMetadata,
)
from swarm_inference.worker.continuous_scheduler import ContinuousBatchScheduler


@dataclass(slots=True)
class FastGenerationMetrics:
    profile: str
    attention_backend: str
    cache_backend: str
    compile_mode: str
    logical_stage_count: int
    physical_worker_process_count: int
    cuda_context_count: int
    batch_size: int
    prompt_tokens_per_request: int
    output_tokens_per_request: int
    useful_prompt_tokens: int
    padding_tokens: int
    host_to_device_bytes: int
    device_to_host_bytes: int
    full_logit_equivalent_bytes: int
    coordinator_bound_bytes: int
    cuda_synchronisations: int
    host_to_device_ms: float
    prefill_ms: float
    prefill_tokens_per_second: float
    decode_ms: float
    decode_output_tokens_per_second: float
    aggregate_verified_output_tokens_per_second: float
    end_to_end_ms: float
    ttft_ms: float
    sampling_ms: float
    serialisation_ms: float
    tokenisation_ms: float
    device_to_host_ms: float
    gpu_kernel_and_transfer_ms: float
    scheduler_ms: float
    queue_wait_ms: float
    inter_token_latency_ms_p50: float
    inter_token_latency_ms_p95: float
    inter_token_latency_ms_p99: float
    peak_vram_bytes: int
    cache_reserved_bytes: int
    cache_used_bytes: int
    cache_allocation_count: int
    cache_fragmentation_fraction: float
    cache_accounting_status: str
    batch_forward_count: int
    full_logits_transferred: bool
    prefill_mode: str = "homogeneous_full_prompt"
    chunked_prefill_supported: bool = False
    kernels_per_decode_token: float | None = None
    kernel_count_status: str = "unavailable_nsight_systems_not_installed"
    compile_diagnostics: dict[str, Any] = field(default_factory=dict)
    cuda_graph_capture_ms: float = 0.0
    cuda_graph_replay_count: int = 0
    cuda_graph_verified: bool = False
    cuda_graph_admission_status: str = "not_requested"
    cuda_graph_projected_bytes: int = 0
    cuda_graph_budget_bytes: int = 0
    cuda_graph_fallback_reason: str | None = None
    scheduler_policy: str = "latency"
    scheduler_metrics: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FastGenerationResult:
    request_ids: tuple[str, ...]
    output_token_ids: list[list[int]]
    selected_logits: list[list[float]]
    metrics: FastGenerationMetrics

    def payload(self) -> dict[str, Any]:
        return {
            "request_ids": list(self.request_ids),
            "output_token_ids": self.output_token_ids,
            "selected_logits": self.selected_logits,
            "metrics": self.metrics.payload(),
        }


@dataclass(frozen=True, slots=True)
class LoadedFastEngine:
    engine: Qwen3FastEngine
    manifest: Any
    model_load_seconds: float
    attention_selection_seconds: float
    source_path: Path


@dataclass(slots=True)
class _CudaGraphVariant:
    token_position: int
    input_buffer: Any
    token_ids: Any
    selected_logits: Any
    graph: Any


@dataclass(slots=True)
class _CudaGraphBundle:
    request_ids: tuple[str, ...]
    prompt_length: int
    output_tokens: int
    variants: list[_CudaGraphVariant]
    capture_ms: float
    pool: Any


@dataclass(frozen=True, slots=True)
class _CudaGraphAdmission:
    admitted: bool
    projected_bytes: int
    budget_bytes: int
    model_parameter_bytes: int
    cache_bytes: int
    retained_logit_bytes: int
    retained_hidden_bytes: int
    runtime_reserve_bytes: int
    reason: str | None


def estimate_cuda_graph_bundle_bytes(
    *,
    model_parameter_bytes: int,
    batch_size: int,
    maximum_sequence_length: int,
    output_tokens: int,
    vocabulary_size: int,
    hidden_size: int,
    key_value_heads: int,
    head_dimension: int,
    layer_count: int,
    dtype_bytes: int,
    runtime_reserve_bytes: int = 512 * 1024 * 1024,
) -> dict[str, int]:
    """Project retained memory for the position-specialised CUDA graph bundle.

    Each captured position retains a vocabulary-sized lm-head output allocation
    and roughly three hidden-sized work buffers per decoder layer.  The estimate
    is intentionally conservative: its purpose is to reject graph variants that
    would force WDDM/CUDA memory paging, not to report allocator-precise usage.
    """

    variant_count = max(0, output_tokens - 1)
    cache_bytes = (
        batch_size
        * maximum_sequence_length
        * key_value_heads
        * head_dimension
        * 2
        * layer_count
        * dtype_bytes
    )
    retained_logit_bytes = variant_count * batch_size * vocabulary_size * dtype_bytes
    retained_hidden_bytes = variant_count * batch_size * hidden_size * dtype_bytes * 3 * layer_count
    projected_bytes = (
        model_parameter_bytes
        + cache_bytes
        + retained_logit_bytes
        + retained_hidden_bytes
        + runtime_reserve_bytes
    )
    return {
        "projected_bytes": projected_bytes,
        "model_parameter_bytes": model_parameter_bytes,
        "cache_bytes": cache_bytes,
        "retained_logit_bytes": retained_logit_bytes,
        "retained_hidden_bytes": retained_hidden_bytes,
        "runtime_reserve_bytes": runtime_reserve_bytes,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


class Qwen3GroupedStageChain:
    """Run multiple logical Qwen3 stages through direct same-device tensors.

    This is the local performance topology for experimenting with logical
    stage boundaries without adding processes, CUDA contexts, host copies, or
    serialisation. Each child retains its own stage-local KV cache.
    """

    def __init__(self, stages: Sequence[Qwen3StageModule]) -> None:
        if not stages:
            raise ValueError("a grouped Qwen3 chain requires at least one logical stage")
        self.stages = tuple(stages)
        first = self.stages[0]
        last = self.stages[-1]
        if any(stage.execution_profile != "qwen3_fast" for stage in self.stages):
            raise ValueError("every grouped Qwen3 stage must use qwen3_fast")
        if any(stage.device != first.device for stage in self.stages):
            raise ValueError("grouped Qwen3 stages must share one CUDA device")
        if any(stage.torch is not first.torch for stage in self.stages):
            raise ValueError("grouped Qwen3 stages must share one PyTorch runtime")
        if any(stage.engine_options != first.engine_options for stage in self.stages):
            raise ValueError("grouped Qwen3 stages must use identical engine options")
        if first.embed_tokens is None:
            raise ValueError("the first grouped stage must own embeddings")
        if last.lm_head is None:
            raise ValueError("the final grouped stage must own the output head")
        if any(stage.embed_tokens is not None for stage in self.stages[1:]):
            raise ValueError("only the first grouped stage may own embeddings")
        if any(stage.lm_head is not None for stage in self.stages[:-1]):
            raise ValueError("only the final grouped stage may own the output head")
        for left, right in zip(self.stages, self.stages[1:], strict=False):
            if left.stage.layer_end != right.stage.layer_start:
                raise ValueError("grouped Qwen3 stage layer ranges must be contiguous")

        self.stage_id = first.stage_id
        self.required_memory_bytes = sum(stage.required_memory_bytes for stage in self.stages)
        self.execution_profile = first.execution_profile
        self.profile = first.execution_profile
        self.engine_options = first.engine_options
        self.config = first.config
        self.torch = first.torch
        self.device = first.device
        self.dtype = first.dtype
        self.embed_tokens = first.embed_tokens
        self.lm_head = last.lm_head
        self.attention_evidence = first.attention_evidence

    @property
    def logical_stage_count(self) -> int:
        return len(self.stages)

    @property
    def attention_backend(self) -> str:
        return self.stages[0].attention_backend

    @attention_backend.setter
    def attention_backend(self, value: str) -> None:
        for stage in self.stages:
            stage.attention_backend = value
            stage.config._attn_implementation = value
            stage.attention_evidence.selected = value

    def topology(self) -> dict[str, int | str]:
        return {
            "tensor_path": "in_process_gpu",
            "logical_stage_count": self.logical_stage_count,
            "physical_worker_process_count": 1,
            "cuda_context_count": 1,
        }

    def prefill_cuda(
        self,
        input_tensor: Any,
        metadata: StageExecutionMetadata,
    ) -> Any:
        output = input_tensor
        for stage in self.stages:
            output = stage.prefill_cuda(output, metadata)
        return output

    def decode_cuda(
        self,
        input_tensor: Any,
        metadata: StageExecutionMetadata,
    ) -> Any:
        output = input_tensor
        for stage in self.stages:
            output = stage.decode_cuda(output, metadata)
        return output

    def prefill_batch_cuda(
        self,
        input_tensors: Any,
        metadata: BatchExecutionMetadata,
    ) -> Any:
        output = input_tensors
        for stage in self.stages:
            output = stage.prefill_batch_cuda(output, metadata)
        return output

    def decode_batch_cuda(
        self,
        input_tensors: Any,
        metadata: BatchExecutionMetadata,
    ) -> Any:
        output = input_tensors
        for stage in self.stages:
            output = stage.decode_batch_cuda(output, metadata)
        return output

    def sample_cuda(
        self,
        logits: Any,
        *,
        request_ids: tuple[str, ...],
        parameters: SamplingParameters | None = None,
        token_history: Any | None = None,
    ) -> Any:
        return self.stages[-1].sample_cuda(
            logits,
            request_ids=request_ids,
            parameters=parameters,
            token_history=token_history,
        )

    def cancel(self, request_id: str) -> None:
        for stage in self.stages:
            stage.cancel(request_id)

    def cancel_batch(self, request_ids: tuple[str, ...]) -> None:
        for stage in self.stages:
            stage.cancel_batch(request_ids)

    def cache_bytes(self) -> int:
        return sum(stage.cache_bytes() for stage in self.stages)

    def state_summary(self) -> dict[str, Any]:
        states = [stage.state_summary() for stage in self.stages]
        batch_forward_counts = [int(state["fast_batch_forward_count"]) for state in states]
        return {
            "execution_profile": self.execution_profile,
            "attention_backend": self.attention_backend,
            "attention_backend_evidence": self.attention_evidence.payload(),
            "cache_backend": self.engine_options.cache_backend.value,
            "cache_dtype": self.engine_options.cache_dtype.value,
            "compile_mode": self.engine_options.compile_mode.value,
            "compile_diagnostics": {
                "requested_mode": self.engine_options.compile_mode.value,
                "logical_stage_count": self.logical_stage_count,
                "stage_diagnostics": [state["compile_diagnostics"] for state in states],
                "prefill_compiled": all(
                    bool(state["compile_diagnostics"]["prefill_compiled"]) for state in states
                ),
                "decode_compiled": all(
                    bool(state["compile_diagnostics"]["decode_compiled"]) for state in states
                ),
                "fallback_used": any(
                    bool(state["compile_diagnostics"]["fallback_used"]) for state in states
                ),
            },
            "logical_stage_count": self.logical_stage_count,
            "physical_worker_process_count": 1,
            "cuda_context_count": 1,
            "cache_count": sum(int(state["cache_count"]) for state in states),
            "cache_bytes": sum(int(state["cache_bytes"]) for state in states),
            "caches": [cache for state in states for cache in list(state["caches"])],
            # A complete grouped model forward has occurred only after every
            # logical stage has executed once, so report the common minimum.
            "fast_batch_forward_count": min(batch_forward_counts),
            "logical_stage_forward_count": sum(batch_forward_counts),
            "stages": states,
        }


class Qwen3FastEngine:
    """One-process engine that keeps the local stage chain on one CUDA device."""

    def __init__(
        self,
        stage: Qwen3StageModule | Qwen3GroupedStageChain,
    ) -> None:
        if stage.execution_profile != "qwen3_fast":
            raise ValueError("Qwen3FastEngine requires a qwen3_fast stage")
        if stage.embed_tokens is None or stage.lm_head is None:
            raise ValueError("monolithic fast engine requires embedding and final model stage")
        self.stage = stage
        self.torch = stage.torch
        self.device = stage.device
        self.profile = stage.execution_profile
        self.logical_stage_count = getattr(stage, "logical_stage_count", 1)
        self._cuda_graph_bundles: dict[tuple[int, int, int], _CudaGraphBundle] = {}
        self._cuda_graph_admissions: dict[tuple[int, int, int], _CudaGraphAdmission] = {}

    def _cuda_graph_stage(self) -> Qwen3StageModule:
        if isinstance(self.stage, Qwen3GroupedStageChain):
            raise RuntimeError(
                "manual CUDA graph capture currently requires one logical "
                "stage; grouped-stage eager and torch.compile paths remain supported"
            )
        return self.stage

    def _cuda_graph_admission(
        self,
        *,
        batch_size: int,
        prompt_length: int,
        output_tokens: int,
    ) -> _CudaGraphAdmission:
        graph_stage = self._cuda_graph_stage()
        key = (batch_size, prompt_length, output_tokens)
        existing = self._cuda_graph_admissions.get(key)
        if existing is not None:
            return existing

        parameter_storages: dict[int, int] = {}
        for parameter in graph_stage._parameters().values():
            storage = parameter.untyped_storage()
            parameter_storages.setdefault(int(storage.data_ptr()), int(storage.nbytes()))
        model_parameter_bytes = sum(parameter_storages.values())
        config = graph_stage.config
        attention_heads = int(config.num_attention_heads)
        key_value_heads = int(getattr(config, "num_key_value_heads", None) or attention_heads)
        head_dimension = int(
            getattr(config, "head_dim", None) or int(config.hidden_size) // attention_heads
        )
        dtype_bytes = int(self.torch.empty((), dtype=graph_stage.dtype).element_size())
        projection = estimate_cuda_graph_bundle_bytes(
            model_parameter_bytes=model_parameter_bytes,
            batch_size=batch_size,
            maximum_sequence_length=max(
                prompt_length + output_tokens,
                int(graph_stage.engine_options.max_sequence_length),
            ),
            output_tokens=output_tokens,
            vocabulary_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            key_value_heads=key_value_heads,
            head_dimension=head_dimension,
            layer_count=len(graph_stage._layer_modules),
            dtype_bytes=dtype_bytes,
        )
        total_memory = int(self.torch.cuda.get_device_properties(self.device).total_memory)
        budget_bytes = int(total_memory * 0.88)
        projected_bytes = int(projection["projected_bytes"])
        admitted = projected_bytes <= budget_bytes
        reason = None
        if not admitted:
            reason = (
                "projected position-specialised CUDA graph bundle memory "
                f"{projected_bytes} exceeds the {budget_bytes}-byte "
                "88% device-memory budget; using eager/dynamic current-length "
                "decode to prevent CUDA/WDDM paging and static-view allocator "
                "growth"
            )
        admission = _CudaGraphAdmission(
            admitted=admitted,
            projected_bytes=projected_bytes,
            budget_bytes=budget_bytes,
            model_parameter_bytes=int(projection["model_parameter_bytes"]),
            cache_bytes=int(projection["cache_bytes"]),
            retained_logit_bytes=int(projection["retained_logit_bytes"]),
            retained_hidden_bytes=int(projection["retained_hidden_bytes"]),
            runtime_reserve_bytes=int(projection["runtime_reserve_bytes"]),
            reason=reason,
        )
        self._cuda_graph_admissions[key] = admission
        return admission

    def _attention_greedy_probe(
        self,
        backend: AttentionBackend,
        *,
        output_tokens: int,
    ) -> tuple[list[int], float]:
        torch = self.torch
        self.stage.config._attn_implementation = backend.value
        self.stage.attention_backend = backend.value
        request_id = f"attention-full-decode-{backend.value}"
        token_buffer = torch.empty((1, output_tokens), dtype=torch.long, device=self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        metadata = StageExecutionMetadata(
            request_id=request_id,
            token_position=0,
            sequence_length=1,
        )
        logits = self.stage.prefill_cuda(
            torch.ones((1, 1), dtype=torch.long, device=self.device),
            metadata,
        )
        token_buffer[:, 0] = torch.argmax(logits[:, -1, :], dim=-1)
        for index in range(1, output_tokens):
            logits = self.stage.decode_cuda(
                token_buffer[:, index - 1 : index],
                StageExecutionMetadata(
                    request_id=request_id,
                    token_position=index,
                    sequence_length=1,
                ),
            )
            token_buffer[:, index] = torch.argmax(logits[:, -1, :], dim=-1)
        end.record()
        host_tokens = token_buffer.cpu()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        self.stage.cancel(request_id)
        return (
            [int(value) for value in host_tokens[0].tolist()],
            output_tokens / (elapsed_ms / 1000) if elapsed_ms else 0.0,
        )

    def autotune_attention_backend(self, *, output_tokens: int = 64) -> str:
        """Select the fastest backend that matches an autoregressive oracle."""

        if (
            self.stage.engine_options.profile != Qwen3ExecutionProfile.FAST
            or self.stage.engine_options.attention_backend != AttentionBackend.AUTO
        ):
            return self.stage.attention_backend
        token_count = min(
            output_tokens,
            self.stage.engine_options.max_sequence_length,
        )
        if token_count < 2:
            raise ValueError("attention full-decode probe requires cache capacity >= 2")
        tuning_stages = (
            self.stage.stages if isinstance(self.stage, Qwen3GroupedStageChain) else (self.stage,)
        )
        saved_compiled = tuple(
            (stage._compiled_prefill, stage._compiled_decode) for stage in tuning_stages
        )
        for stage in tuning_stages:
            stage._compiled_prefill = None
            stage._compiled_decode = None
        evidence = self.stage.attention_evidence
        evidence.full_decode_probe_tokens = token_count
        try:
            reference, eager_rate = self._attention_greedy_probe(
                AttentionBackend.EAGER,
                output_tokens=token_count,
            )
            evidence.full_decode_correct[AttentionBackend.EAGER.value] = True
            evidence.full_decode_tokens_per_second[AttentionBackend.EAGER.value] = eager_rate
            successful = [(AttentionBackend.EAGER, eager_rate)]
            for candidate in auto_attention_candidates(evidence.available):
                if candidate == AttentionBackend.EAGER:
                    continue
                try:
                    tokens, rate = self._attention_greedy_probe(
                        candidate,
                        output_tokens=token_count,
                    )
                    exact = tokens == reference
                    evidence.full_decode_correct[candidate.value] = exact
                    evidence.full_decode_tokens_per_second[candidate.value] = rate
                    if exact:
                        successful.append((candidate, rate))
                    else:
                        mismatch = next(
                            index
                            for index, pair in enumerate(zip(tokens, reference, strict=True))
                            if pair[0] != pair[1]
                        )
                        evidence.diagnostics[candidate.value] = (
                            f"full autoregressive greedy identity failed at output token {mismatch}"
                        )
                except Exception as exc:
                    evidence.full_decode_correct[candidate.value] = False
                    evidence.diagnostics[candidate.value] = (
                        f"full decode probe failed: {type(exc).__name__}: {exc}"
                    )
            selected = max(successful, key=lambda item: item[1])[0]
            self.stage.config._attn_implementation = selected.value
            self.stage.attention_backend = selected.value
            evidence.selected = selected.value
            return selected.value
        finally:
            for stage, (saved_prefill, saved_decode) in zip(
                tuning_stages,
                saved_compiled,
                strict=True,
            ):
                stage._compiled_prefill = saved_prefill
                stage._compiled_decode = saved_decode

    def _capture_cuda_graph_bundle(
        self,
        cuda_inputs: Any,
        *,
        batch_size: int,
        prompt_length: int,
        output_tokens: int,
    ) -> _CudaGraphBundle:
        """Capture exact-prefix-shape decode variants for one batch bucket."""

        torch = self.torch
        graph_stage = self._cuda_graph_stage()
        key = (batch_size, prompt_length, output_tokens)
        request_ids = tuple(
            f"cuda-graph-b{batch_size}-p{prompt_length}-slot-{index}" for index in range(batch_size)
        )
        prefill_metadata = BatchExecutionMetadata(
            requests=tuple(
                StageExecutionMetadata(
                    request_id=request_id,
                    token_position=0,
                    sequence_length=prompt_length,
                )
                for request_id in request_ids
            )
        )
        capture_started = time.perf_counter()
        logits = graph_stage.prefill_batch_cuda(cuda_inputs, prefill_metadata)
        token_ids = torch.argmax(logits[:, -1, :], dim=-1)
        pool = torch.cuda.graph_pool_handle()
        variants: list[_CudaGraphVariant] = []
        for output_index in range(1, output_tokens):
            token_position = prompt_length + output_index - 1
            metadata = BatchExecutionMetadata(
                requests=tuple(
                    StageExecutionMetadata(
                        request_id=request_id,
                        token_position=token_position,
                        sequence_length=1,
                    )
                    for request_id in request_ids
                )
            )
            graph_input = torch.empty((batch_size, 1), dtype=torch.long, device=self.device)
            graph_input.copy_(token_ids.view(batch_size, 1))
            if output_index == 1:
                # Initialises the one-token SDPA/kernel path before capture.
                warm_logits = graph_stage.decode_batch_cuda(graph_input, metadata)
                torch.argmax(warm_logits[:, -1, :], dim=-1)
                cache = graph_stage._get_static_cache(metadata)
                with torch.inference_mode():
                    cache.rollback(token_position)
            torch.cuda.current_stream(self.device).synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                graph_logits = graph_stage.decode_batch_cuda(graph_input, metadata)
                graph_token_ids = torch.argmax(graph_logits[:, -1, :], dim=-1)
                graph_selected_logits = graph_logits[
                    torch.arange(batch_size, device=self.device),
                    -1,
                    graph_token_ids,
                ]
            graph.replay()
            torch.cuda.current_stream(self.device).synchronize()
            variants.append(
                _CudaGraphVariant(
                    token_position=token_position,
                    input_buffer=graph_input,
                    token_ids=graph_token_ids,
                    selected_logits=graph_selected_logits,
                    graph=graph,
                )
            )
            token_ids = graph_token_ids
        cache = next(
            cache
            for cache in graph_stage._static_caches.values()
            if cache.request_ids == request_ids
        )
        with torch.inference_mode():
            cache.rollback(0)
        bundle = _CudaGraphBundle(
            request_ids=request_ids,
            prompt_length=prompt_length,
            output_tokens=output_tokens,
            variants=variants,
            capture_ms=(time.perf_counter() - capture_started) * 1000,
            pool=pool,
        )
        self._cuda_graph_bundles[key] = bundle
        return bundle

    def generate_batch(
        self,
        input_token_ids: Any,
        *,
        request_ids: tuple[str, ...],
        output_tokens: int,
        sampling: SamplingParameters | None = None,
        queue_wait_ms: float = 0.0,
        padding_tokens: int = 0,
        scheduler_policy: str = "latency",
    ) -> FastGenerationResult:
        torch = self.torch
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        if input_token_ids.ndim != 2:
            raise ValueError("input_token_ids must have shape [batch, sequence]")
        batch_size = int(input_token_ids.shape[0])
        prompt_length = int(input_token_ids.shape[1])
        if batch_size != len(request_ids):
            raise ValueError("request_ids must match input batch size")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request IDs must be unique")
        selected_sampling = sampling or SamplingParameters()
        request_started = time.perf_counter()
        scheduler_started = time.perf_counter()
        scheduler_batch_capacity = batch_size * 2 if scheduler_policy == "balanced" else batch_size
        scheduler = ContinuousBatchScheduler(
            policy=scheduler_policy,
            max_batch_size=scheduler_batch_capacity,
            prefill_chunk_size=prompt_length,
            kv_token_capacity=(batch_size * (prompt_length + output_tokens) * 2),
        )
        admission_started_ns = time.monotonic_ns()
        with nvtx_range(
            torch,
            "request_admission",
            enabled=self.stage.engine_options.nvtx_enabled,
        ):
            for request_id in request_ids:
                scheduler.admit(
                    request_id,
                    prompt_tokens=prompt_length,
                    output_tokens=output_tokens,
                    admitted_ns=admission_started_ns,
                )
        prefill_iteration = scheduler.next_iteration()
        if set(prefill_iteration.prefill_request_ids) != set(request_ids):
            raise RuntimeError(
                "continuous scheduler did not admit the complete homogeneous prefill batch"
            )
        graph_bundle = None
        graph_stage: Qwen3StageModule | None = None
        graph_admission: _CudaGraphAdmission | None = None
        dynamic_cache_memory_fallback = False
        execution_request_ids = request_ids
        if self.stage.engine_options.compile_mode.value == "manual_cuda_graph":
            graph_stage = self._cuda_graph_stage()
            if not selected_sampling.greedy:
                raise ValueError("manual CUDA graph mode currently supports greedy sampling only")
            graph_key = (batch_size, prompt_length, output_tokens)
            graph_bundle = self._cuda_graph_bundles.get(graph_key)
            if graph_bundle is None:
                graph_admission = self._cuda_graph_admission(
                    batch_size=batch_size,
                    prompt_length=prompt_length,
                    output_tokens=output_tokens,
                )
                if graph_admission.admitted:
                    # Capture is a declared cold-readiness activity and is not
                    # included in measured prefill/decode events below.
                    temporary_cuda_inputs = input_token_ids.to(
                        device=self.device,
                        dtype=torch.long,
                        non_blocking=False,
                    )
                    graph_bundle = self._capture_cuda_graph_bundle(
                        temporary_cuda_inputs,
                        batch_size=batch_size,
                        prompt_length=prompt_length,
                        output_tokens=output_tokens,
                    )
                else:
                    graph_stage.use_dynamic_cache_memory_fallback(request_ids)
                    dynamic_cache_memory_fallback = True
            else:
                graph_admission = self._cuda_graph_admissions.get(graph_key)
            if graph_bundle is not None:
                execution_request_ids = graph_bundle.request_ids
        metadata_items = tuple(
            StageExecutionMetadata(
                request_id=request_id,
                token_position=0,
                sequence_length=prompt_length,
            )
            for request_id in execution_request_ids
        )
        prefill_metadata = BatchExecutionMetadata(
            requests=metadata_items,
            padded_sequence_length=prompt_length,
            padding_tokens=padding_tokens,
        )
        scheduler_ms = (time.perf_counter() - scheduler_started) * 1000

        host_to_device_bytes = int(input_token_ids.numel() * input_token_ids.element_size())
        h2d_start = torch.cuda.Event(enable_timing=True)
        h2d_end = torch.cuda.Event(enable_timing=True)
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        sampling_events: list[tuple[Any, Any]] = []
        decode_events: list[tuple[Any, Any]] = []
        torch.cuda.reset_peak_memory_stats(self.device)
        h2d_start.record()
        cuda_inputs = input_token_ids.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )
        h2d_end.record()
        output_buffer = torch.empty(
            (batch_size, output_tokens),
            device=self.device,
            dtype=torch.long,
        )
        selected_logit_buffer = torch.empty(
            (batch_size, output_tokens),
            device=self.device,
            dtype=self.stage.dtype,
        )
        prefill_start.record()
        logits = self.stage.prefill_batch_cuda(cuda_inputs, prefill_metadata)
        prefill_end.record()
        sample_start = torch.cuda.Event(enable_timing=True)
        sample_end = torch.cuda.Event(enable_timing=True)
        sample_start.record()
        sampled = self.stage.sample_cuda(
            logits,
            request_ids=request_ids,
            parameters=selected_sampling,
        )
        output_buffer[:, 0].copy_(sampled.token_ids)
        selected_logit_buffer[:, 0].copy_(sampled.selected_logits)
        sample_end.record()
        sampling_events.append((sample_start, sample_end))
        scheduler.mark_prefill(
            prefill_iteration.prefill_request_ids,
            useful_tokens=batch_size * prompt_length - padding_tokens,
            padding_tokens=padding_tokens,
        )
        first_decode_iteration = scheduler.next_iteration()
        if set(first_decode_iteration.decode_request_ids) != set(request_ids):
            raise RuntimeError("continuous scheduler did not select every active decode request")
        scheduler.mark_decode(first_decode_iteration.decode_request_ids)

        graph_capture_ms = graph_bundle.capture_ms if graph_bundle is not None else 0.0
        graph_replay_count = 0
        graph_verified = False
        for output_index in range(1, output_tokens):
            scheduling_iteration = scheduler.next_iteration()
            if set(scheduling_iteration.decode_request_ids) != set(request_ids):
                raise RuntimeError("continuous scheduler decode batch lost an active request")
            decode_metadata = BatchExecutionMetadata(
                requests=tuple(
                    StageExecutionMetadata(
                        request_id=request_id,
                        token_position=prompt_length + output_index - 1,
                        sequence_length=1,
                    )
                    for request_id in execution_request_ids
                )
            )
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record()
            if graph_bundle is not None:
                assert graph_stage is not None
                variant = graph_bundle.variants[output_index - 1]
                variant.input_buffer.copy_(output_buffer[:, output_index - 1 : output_index])
                variant.graph.replay()
                cache = graph_stage._get_static_cache(decode_metadata)
                cache.graph_advance(expected_position=decode_metadata.token_position)
                graph_replay_count += 1
                graph_verified = True
                sampled_token_ids = variant.token_ids
                sampled_selected_logits = variant.selected_logits
                logits = None
            else:
                logits = self.stage.decode_batch_cuda(
                    output_buffer[:, output_index - 1 : output_index],
                    decode_metadata,
                )
            decode_end.record()
            decode_events.append((decode_start, decode_end))
            sample_start = torch.cuda.Event(enable_timing=True)
            sample_end = torch.cuda.Event(enable_timing=True)
            sample_start.record()
            if logits is not None:
                sampled = self.stage.sample_cuda(
                    logits,
                    request_ids=request_ids,
                    parameters=selected_sampling,
                    token_history=output_buffer[:, :output_index],
                )
                sampled_token_ids = sampled.token_ids
                sampled_selected_logits = sampled.selected_logits
            output_buffer[:, output_index].copy_(sampled_token_ids)
            selected_logit_buffer[:, output_index].copy_(sampled_selected_logits)
            sample_end.record()
            sampling_events.append((sample_start, sample_end))
            scheduler.mark_decode(scheduling_iteration.decode_request_ids)

        d2h_start = torch.cuda.Event(enable_timing=True)
        d2h_end = torch.cuda.Event(enable_timing=True)
        d2h_start.record()
        host_outputs = output_buffer.cpu()
        host_selected_logits = selected_logit_buffer.cpu()
        d2h_end.record()
        # One explicit measurement boundary covers every previously recorded
        # event and the final compact result copy.
        d2h_end.synchronize()
        h2d_ms = float(h2d_start.elapsed_time(h2d_end))
        prefill_ms = float(prefill_start.elapsed_time(prefill_end))
        decode_latencies = [float(start.elapsed_time(end)) for start, end in decode_events]
        decode_ms = sum(decode_latencies)
        sampling_latencies = [float(start.elapsed_time(end)) for start, end in sampling_events]
        sampling_ms = sum(sampling_latencies)
        ttft_ms = h2d_ms + prefill_ms + sampling_latencies[0]
        d2h_ms = float(d2h_start.elapsed_time(d2h_end))
        gpu_kernel_and_transfer_ms = h2d_ms + prefill_ms + decode_ms + sampling_ms + d2h_ms
        output_lists = [[int(value) for value in row] for row in host_outputs.tolist()]
        selected_lists = [
            [float(value) for value in row] for row in host_selected_logits.float().tolist()
        ]
        end_to_end_ms = (time.perf_counter() - request_started) * 1000
        output_count = batch_size * output_tokens
        decoded_count = batch_size * max(0, output_tokens - 1)
        selected_logit_bytes = int(
            selected_logit_buffer.numel() * selected_logit_buffer.element_size()
        )
        output_id_bytes = int(output_buffer.numel() * output_buffer.element_size())
        coordinator_bound_bytes = output_id_bytes + selected_logit_bytes
        full_logit_equivalent_bytes = (
            output_count
            * int(self.stage.config.vocab_size)
            * 4  # legacy final logits are converted to FP32 before transport
        )
        state = self.stage.state_summary()
        compile_diagnostics = dict(state["compile_diagnostics"])
        if graph_admission is not None and not graph_admission.admitted:
            compile_diagnostics.update(
                {
                    "fallback_used": True,
                    "fallback_reason": graph_admission.reason,
                    "cuda_graph_admission_status": "rejected_memory_projection",
                    "cuda_graph_projected_bytes": graph_admission.projected_bytes,
                    "cuda_graph_budget_bytes": graph_admission.budget_bytes,
                }
            )
        cache_summaries = list(state["caches"])
        cache_reserved_bytes = sum(
            int(cache.get("reserved_bytes", cache.get("cache_bytes", 0)))
            for cache in cache_summaries
        )
        cache_used_bytes = sum(int(cache.get("cache_bytes", 0)) for cache in cache_summaries)
        cache_allocation_count = sum(
            int(cache.get("allocation_count", 0)) for cache in cache_summaries
        )
        cache_fragmentation_fraction = (
            (cache_reserved_bytes - cache_used_bytes) / cache_reserved_bytes
            if cache_reserved_bytes
            else 0.0
        )
        scheduler_state = scheduler.state()
        metrics = FastGenerationMetrics(
            profile=self.profile,
            attention_backend=self.stage.attention_backend,
            cache_backend=(
                "dynamic_reference_memory_fallback"
                if dynamic_cache_memory_fallback
                else self.stage.engine_options.cache_backend.value
            ),
            compile_mode=self.stage.engine_options.compile_mode.value,
            logical_stage_count=self.logical_stage_count,
            physical_worker_process_count=1,
            cuda_context_count=1,
            batch_size=batch_size,
            prompt_tokens_per_request=prompt_length,
            output_tokens_per_request=output_tokens,
            useful_prompt_tokens=batch_size * prompt_length - padding_tokens,
            padding_tokens=padding_tokens,
            host_to_device_bytes=host_to_device_bytes,
            device_to_host_bytes=coordinator_bound_bytes,
            full_logit_equivalent_bytes=full_logit_equivalent_bytes,
            coordinator_bound_bytes=coordinator_bound_bytes,
            cuda_synchronisations=1,
            host_to_device_ms=h2d_ms,
            prefill_ms=prefill_ms,
            prefill_tokens_per_second=(
                (batch_size * prompt_length - padding_tokens) / (prefill_ms / 1000)
                if prefill_ms
                else 0.0
            ),
            decode_ms=decode_ms,
            decode_output_tokens_per_second=(
                decoded_count / (decode_ms / 1000) if decode_ms else 0.0
            ),
            aggregate_verified_output_tokens_per_second=(
                output_count / (end_to_end_ms / 1000) if end_to_end_ms else 0.0
            ),
            end_to_end_ms=end_to_end_ms,
            ttft_ms=ttft_ms,
            sampling_ms=sampling_ms,
            serialisation_ms=0.0,
            tokenisation_ms=0.0,
            device_to_host_ms=d2h_ms,
            gpu_kernel_and_transfer_ms=gpu_kernel_and_transfer_ms,
            scheduler_ms=(
                scheduler_ms + float(scheduler_state["metrics"]["scheduler_overhead_ms"])
            ),
            queue_wait_ms=queue_wait_ms,
            inter_token_latency_ms_p50=_percentile(decode_latencies, 0.50),
            inter_token_latency_ms_p95=_percentile(decode_latencies, 0.95),
            inter_token_latency_ms_p99=_percentile(decode_latencies, 0.99),
            peak_vram_bytes=int(torch.cuda.max_memory_allocated(self.device)),
            cache_reserved_bytes=cache_reserved_bytes,
            cache_used_bytes=cache_used_bytes,
            cache_allocation_count=cache_allocation_count,
            cache_fragmentation_fraction=cache_fragmentation_fraction,
            cache_accounting_status="measured_before_request_cleanup",
            batch_forward_count=int(state["fast_batch_forward_count"]),
            full_logits_transferred=selected_sampling.return_full_logits,
            compile_diagnostics=compile_diagnostics,
            cuda_graph_capture_ms=graph_capture_ms,
            cuda_graph_replay_count=graph_replay_count,
            cuda_graph_verified=graph_verified,
            cuda_graph_admission_status=(
                "captured"
                if graph_bundle is not None
                else (
                    "rejected_memory_projection"
                    if graph_admission is not None and not graph_admission.admitted
                    else "not_requested"
                )
            ),
            cuda_graph_projected_bytes=(
                graph_admission.projected_bytes if graph_admission is not None else 0
            ),
            cuda_graph_budget_bytes=(
                graph_admission.budget_bytes if graph_admission is not None else 0
            ),
            cuda_graph_fallback_reason=(
                graph_admission.reason if graph_admission is not None else None
            ),
            scheduler_policy=scheduler_policy,
            scheduler_metrics=dict(scheduler_state["metrics"]),
        )
        if graph_bundle is None:
            self.stage.cancel_batch(execution_request_ids)
        else:
            assert graph_stage is not None
            cache = next(
                cache
                for cache in graph_stage._static_caches.values()
                if cache.request_ids == execution_request_ids
            )
            with torch.inference_mode():
                cache.rollback(0)
        return FastGenerationResult(
            request_ids=request_ids,
            output_token_ids=output_lists,
            selected_logits=selected_lists,
            metrics=metrics,
        )

    def warmup(
        self,
        *,
        prompt_length: int,
        output_tokens: int = 4,
        batch_size: int = 1,
        index: int = 0,
    ) -> FastGenerationResult:
        inputs = self.torch.zeros(
            (batch_size, prompt_length),
            dtype=self.torch.long,
        )
        return self.generate_batch(
            inputs,
            request_ids=tuple(f"fast-warmup-{index}-{member}" for member in range(batch_size)),
            output_tokens=output_tokens,
        )


def load_qwen3_fast_engine(
    *,
    model_id: str,
    model_revision: str,
    model_path: Path,
    options: Qwen3EngineOptions,
    device: str = "cuda",
    dtype_name: str = "bfloat16",
) -> LoadedFastEngine:
    import torch
    from transformers import Qwen3Config

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported Qwen3 engine dtype {dtype_name!r}")
    started = time.perf_counter()
    resolved = ResolvedModel(
        model_id=model_id,
        revision=model_revision,
        path=model_path.resolve(),
        downloaded=False,
    )
    description = inspect_qwen3_model(resolved)
    maximum = sum(item.bytes for item in description.tensors) * 2
    manifest = build_manifest(
        description,
        target_stage_bytes=maximum,
        maximum_stage_bytes=maximum,
        stage_count=1,
    )
    config = Qwen3Config.from_pretrained(model_path, local_files_only=True)
    stage = Qwen3StageModule(
        config=config,
        stage=manifest.stages[0],
        device=device,
        dtype=dtype,
        engine_options=options,
    )
    stage.load_weights(model_path, manifest=manifest)
    load_seconds = time.perf_counter() - started
    engine = Qwen3FastEngine(stage)
    attention_selection_seconds = 0.0
    if options.attention_backend == AttentionBackend.AUTO:
        attention_selection_started = time.perf_counter()
        engine.autotune_attention_backend()
        attention_selection_seconds = time.perf_counter() - attention_selection_started
    return LoadedFastEngine(
        engine=engine,
        manifest=manifest,
        model_load_seconds=load_seconds,
        attention_selection_seconds=attention_selection_seconds,
        source_path=model_path.resolve(),
    )


def summarise_generation_repeats(
    results: list[FastGenerationResult],
) -> dict[str, float]:
    if not results:
        raise ValueError("at least one generation result is required")
    rates = [result.metrics.aggregate_verified_output_tokens_per_second for result in results]
    mean = statistics.mean(rates)
    standard_deviation = statistics.pstdev(rates)
    return {
        "median": statistics.median(rates),
        "minimum": min(rates),
        "maximum": max(rates),
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean if mean else 0.0,
    }
