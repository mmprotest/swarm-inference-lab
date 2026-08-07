"""Persistent dense-Qwen3 executor for the canonical native stage runtime."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from swarm_inference.config.models import CacheSpec, StageDefinition, TensorSpec
from swarm_inference.engines.interfaces import ExecutionDevice
from swarm_inference.execution.fast_path import FastPathMeasurement
from swarm_inference.execution.fast_path_registry import (
    FastPathAdmissionError,
    NativeFastPathRegistry,
)
from swarm_inference.execution.interfaces import StageExecutionResult, WeightOwnership
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.qwen3 import Qwen3StageModule
from swarm_inference.model.qwen3_cache import StaticStageKVCache
from swarm_inference.model.qwen3_fast_path import Qwen3CudaFastPath
from swarm_inference.model.qwen3_runtime import (
    Qwen3CacheBackend,
    Qwen3CompileMode,
    Qwen3EngineOptions,
    Qwen3ExecutionProfile,
)
from swarm_inference.model.stage_module import BatchExecutionMetadata, StageExecutionMetadata
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.runtime.performance_profiles import (
    FastPathProfileStore,
    profile_key_from_runtime,
)

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "f16": torch.float16,
    "float32": torch.float32,
    "f32": torch.float32,
}


@dataclass(slots=True)
class _StageCudaGraphSlot:
    """One reusable session-isolated KV/cache pointer set for a resident stage."""

    slot_id: str
    input_buffer: torch.Tensor
    output_buffer: torch.Tensor
    sampled_token_ids: torch.Tensor | None
    graph: Any
    pool: Any
    cache: StaticStageKVCache
    capture_ms: float
    admission_required_bytes: int
    admission_available_bytes: int
    session_id: str | None = None
    replay_count: int = 0


def _stage_cuda_graph_required_bytes(module: Qwen3StageModule) -> int:
    """Conservatively project retained bytes before allocating/capturing a slot."""

    config = module.config
    attention_heads = int(config.num_attention_heads)
    key_value_heads = int(getattr(config, "num_key_value_heads", None) or attention_heads)
    head_dimension = int(
        getattr(config, "head_dim", None) or int(config.hidden_size) // attention_heads
    )
    dtype_bytes = int(torch.empty((), dtype=module.dtype).element_size())
    layer_count = module.stage.layer_end - module.stage.layer_start
    static_cache_bytes = (
        2
        * layer_count
        * key_value_heads
        * head_dimension
        * module.engine_options.max_sequence_length
        * dtype_bytes
    )
    output_width = int(config.vocab_size) if module.lm_head is not None else int(config.hidden_size)
    retained_output_bytes = output_width * dtype_bytes
    # CUDA graph private pools retain intermediate allocations.  Keep a fixed
    # runtime reserve plus a layer/hidden-size projection, then admit against
    # currently free memory rather than total device memory.
    retained_hidden_bytes = max(
        64 * 1024 * 1024,
        3 * layer_count * int(config.hidden_size) * dtype_bytes * 1024,
    )
    runtime_reserve_bytes = 256 * 1024 * 1024
    return (
        static_cache_bytes + retained_output_bytes + retained_hidden_bytes + runtime_reserve_bytes
    )


def _capture_stage_cuda_graph(
    module: Qwen3StageModule,
    *,
    slot_id: str,
    input_template: torch.Tensor,
    pool: Any | None = None,
) -> _StageCudaGraphSlot:
    """Capture one position-independent decode graph backed by an isolated KV slot."""

    if module.device.type != "cuda":
        raise RuntimeError("stage CUDA graph capture requires a CUDA stage")
    if module.engine_options.compile_mode != Qwen3CompileMode.MANUAL_CUDA_GRAPH:
        raise RuntimeError("stage CUDA graph capture requires manual_cuda_graph mode")
    if module.attention_backend == "eager":
        raise RuntimeError(
            "reusable stage CUDA graphs require a fixed-shape attention backend; "
            "eager attention remains an exact static-cache fallback"
        )
    required_bytes = _stage_cuda_graph_required_bytes(module)
    available_bytes, _ = torch.cuda.mem_get_info(module.device)
    budget_bytes = int(available_bytes * 0.88)
    if required_bytes > budget_bytes:
        raise RuntimeError(
            "stage CUDA graph memory admission rejected capture: "
            f"required={required_bytes} available_budget={budget_bytes}"
        )

    metadata_item = StageExecutionMetadata(
        request_id=slot_id,
        token_position=0,
        sequence_length=1,
    )
    batch_metadata = BatchExecutionMetadata(requests=(metadata_item,))
    cache = module.begin_cuda_graph_decode(batch_metadata)
    input_buffer = torch.empty_like(input_template)
    input_buffer.copy_(input_template)

    # Initialise kernels outside capture, then restore the logical KV length.
    warm_output = module.decode_cuda(input_buffer, metadata_item)
    if module.lm_head is not None:
        torch.argmax(warm_output[:, -1, :], dim=-1)
    torch.cuda.current_stream(module.device).synchronize()
    with torch.inference_mode():
        cache.rollback(0)
    module.update_cuda_graph_position(0)

    capture_started = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    selected_pool = pool if pool is not None else torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=selected_pool):
        output_buffer = module.decode_cuda(input_buffer, metadata_item)
        sampled_token_ids = (
            torch.argmax(output_buffer[:, -1, :], dim=-1) if module.lm_head is not None else None
        )
    torch.cuda.current_stream(module.device).synchronize()
    with torch.inference_mode():
        cache.rollback(0)
    return _StageCudaGraphSlot(
        slot_id=slot_id,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        sampled_token_ids=sampled_token_ids,
        graph=graph,
        pool=selected_pool,
        cache=cache,
        capture_ms=(time.perf_counter() - capture_started) * 1000,
        admission_required_bytes=required_bytes,
        admission_available_bytes=budget_bytes,
    )


def _replay_stage_cuda_graph(
    module: Qwen3StageModule,
    slot: _StageCudaGraphSlot,
    *,
    value: torch.Tensor,
    token_position: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    module.update_cuda_graph_position(token_position)
    slot.input_buffer.copy_(value)
    slot.graph.replay()
    slot.cache.graph_advance(expected_position=token_position)
    slot.replay_count += 1
    return slot.output_buffer, slot.sampled_token_ids


def _engine_options(
    request: LoadStageRequest,
    *,
    config: dict[str, Any],
) -> tuple[Qwen3EngineOptions, str, str | None]:
    requested = request.fast_path_mode.strip().lower()
    device = torch.device(request.device)
    if requested in {"", "auto", "auto-exact"}:
        requested = "gpu_native_dynamic_cache" if device.type == "cuda" else "eager"
    if device.type != "cuda" and requested != "eager":
        return (
            Qwen3EngineOptions.from_values(profile=Qwen3ExecutionProfile.CORRECTNESS),
            "eager",
            f"{request.fast_path_mode} requires CUDA on this stage",
        )
    maximum = int(config.get("max_position_embeddings", 4096))
    if requested == "eager":
        return (
            Qwen3EngineOptions.from_values(
                profile=Qwen3ExecutionProfile.CORRECTNESS,
                max_sequence_length=maximum,
            ),
            requested,
            None,
        )
    cache = (
        Qwen3CacheBackend.DYNAMIC_REFERENCE
        if requested == "gpu_native_dynamic_cache"
        else Qwen3CacheBackend.STATIC
    )
    compile_mode = {
        "gpu_native_dynamic_cache": Qwen3CompileMode.EAGER,
        "static_cache": Qwen3CompileMode.EAGER,
        "torch_compile_default": Qwen3CompileMode.DEFAULT,
        "torch_compile_reduce_overhead": Qwen3CompileMode.REDUCE_OVERHEAD,
        "torch_compile_max_autotune": Qwen3CompileMode.MAX_AUTOTUNE,
        "manual_cuda_graph": Qwen3CompileMode.MANUAL_CUDA_GRAPH,
    }.get(requested)
    if compile_mode is None:
        raise ValueError(f"unsupported Qwen3 fast-path mode {request.fast_path_mode!r}")
    return (
        Qwen3EngineOptions.from_values(
            profile=Qwen3ExecutionProfile.FAST,
            cache_backend=cache,
            compile_mode=compile_mode,
            max_sequence_length=maximum,
            final_worker_sampling=True,
            static_cache_fixed_shape=False,
        ),
        requested,
        None,
    )


def _stage_definition(
    request: LoadStageRequest,
    *,
    config: dict[str, Any],
    options: Qwen3EngineOptions,
) -> StageDefinition:
    hidden_size = int(config["hidden_size"])
    return StageDefinition(
        stage_id=request.assignment.stage_id,
        layer_start=request.assignment.layer_start,
        layer_end=request.assignment.layer_end,
        owns_embeddings=request.assignment.owns_embeddings,
        owns_final_norm=request.assignment.owns_final_norm,
        owns_output_head=request.assignment.owns_output_projection,
        required_memory_bytes=max(1, request.assignment.weight_bytes),
        estimated_execution_ms={"default": request.assignment.estimated_compute_ns / 1_000_000},
        input_spec=TensorSpec(
            dtype="int64" if request.assignment.owns_embeddings else request.dtype,
            shape=["batch", "sequence", hidden_size],
        ),
        output_spec=TensorSpec(
            dtype=request.dtype,
            shape=["batch", "sequence", hidden_size],
        ),
        cache_spec=CacheSpec(
            format=options.cache_backend.value,
            bytes_per_token=request.assignment.kv_cache_bytes_per_token,
            reconstructable_by_replay=True,
        ),
    )


def _load_module(
    request: LoadStageRequest,
    *,
    root: Path,
    config: dict[str, Any],
    options: Qwen3EngineOptions,
) -> Qwen3StageModule:
    try:
        dtype = _DTYPES[request.dtype.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3 execution dtype {request.dtype!r}") from exc
    module = Qwen3StageModule(
        config=config,
        stage=_stage_definition(request, config=config, options=options),
        device=request.device,
        dtype=dtype,
        engine_options=options,
    )
    module.load_owned_weights(root, model_revision=request.model_revision)
    return module


def _synchronise(module: Qwen3StageModule) -> None:
    if module.device.type == "cuda":
        torch.cuda.synchronize(module.device)


def _probe_input(
    module: Qwen3StageModule,
    *,
    sequence_length: int,
) -> torch.Tensor:
    if module.embed_tokens is not None:
        vocabulary = int(module.config.vocab_size)
        return (
            torch.arange(
                1,
                sequence_length + 1,
                dtype=torch.long,
                device=module.device,
            )
            .remainder(max(2, vocabulary))
            .unsqueeze(0)
        )
    hidden = int(module.config.hidden_size)
    values = torch.arange(
        sequence_length * hidden,
        dtype=torch.float32,
        device=module.device,
    ).reshape(1, sequence_length, hidden)
    return (values.remainder(97) / 97).to(dtype=module.dtype)


def _signature(
    module: Qwen3StageModule,
    output: torch.Tensor,
) -> tuple[tuple[int, ...], torch.Tensor]:
    last = output[:, -1, :].detach()
    tokens = (
        tuple(int(item) for item in torch.argmax(last, dim=-1).cpu().tolist())
        if module.lm_head is not None
        else ()
    )
    return tokens, last.cpu()


def _probe_module(
    module: Qwen3StageModule,
    *,
    request: LoadStageRequest,
    request_id: str,
    repeat_count: int = 5,
) -> tuple[
    float,
    list[float],
    tuple[tuple[tuple[int, ...], torch.Tensor], ...],
]:
    """Run a real stage-local prefill/decode probe and retain exactness evidence."""

    prompt_length = max(1, min(8, request.fast_path_context_bucket))
    prefill_input = _probe_input(module, sequence_length=prompt_length)
    metadata = StageExecutionMetadata(
        request_id=request_id,
        token_position=0,
        sequence_length=prompt_length,
    )
    _synchronise(module)
    prefill_started = time.perf_counter_ns()
    output = module.prefill_cuda(prefill_input, metadata)
    _synchronise(module)
    prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000
    signatures = [_signature(module, output)]
    decode_ms: list[float] = []
    next_token = (
        torch.argmax(output[:, -1, :], dim=-1).view(1, 1)
        if module.embed_tokens is not None and module.lm_head is not None
        else _probe_input(module, sequence_length=1)
    )
    for offset in range(repeat_count):
        decode_metadata = StageExecutionMetadata(
            request_id=request_id,
            token_position=prompt_length + offset,
            sequence_length=1,
        )
        _synchronise(module)
        started = time.perf_counter_ns()
        output = module.decode_cuda(next_token, decode_metadata)
        _synchronise(module)
        decode_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        signatures.append(_signature(module, output))
        if module.embed_tokens is not None and module.lm_head is not None:
            next_token = torch.argmax(output[:, -1, :], dim=-1).view(1, 1)
    module.reset_cache(request_id)
    return prefill_ms, decode_ms, tuple(signatures)


def _probe_manual_cuda_graph_module(
    module: Qwen3StageModule,
    *,
    request: LoadStageRequest,
    request_id: str,
    repeat_count: int = 5,
) -> tuple[
    float,
    list[float],
    tuple[tuple[tuple[int, ...], torch.Tensor], ...],
]:
    """Measure the same probe through the reusable stage graph replay path."""

    prompt_length = max(1, min(8, request.fast_path_context_bucket))
    slot = _capture_stage_cuda_graph(
        module,
        slot_id=request_id,
        input_template=_probe_input(module, sequence_length=1),
    )
    prefill_input = _probe_input(module, sequence_length=prompt_length)
    metadata = StageExecutionMetadata(
        request_id=request_id,
        token_position=0,
        sequence_length=prompt_length,
    )
    module.update_cuda_graph_position(0)
    _synchronise(module)
    prefill_started = time.perf_counter_ns()
    output = module.prefill_cuda(prefill_input, metadata)
    _synchronise(module)
    prefill_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000
    signatures = [_signature(module, output)]
    decode_ms: list[float] = []
    next_value = (
        torch.argmax(output[:, -1, :], dim=-1).view(1, 1)
        if module.embed_tokens is not None and module.lm_head is not None
        else _probe_input(module, sequence_length=1)
    )
    for offset in range(repeat_count):
        token_position = prompt_length + offset
        _synchronise(module)
        started = time.perf_counter_ns()
        output, _ = _replay_stage_cuda_graph(
            module,
            slot,
            value=next_value,
            token_position=token_position,
        )
        _synchronise(module)
        decode_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        signatures.append(_signature(module, output))
        if module.embed_tokens is not None and module.lm_head is not None:
            next_value = torch.argmax(output[:, -1, :], dim=-1).view(1, 1)
    module.reset_cuda_graph_slot(request_id)
    return prefill_ms, decode_ms, tuple(signatures)


def _exactness(
    reference: tuple[tuple[tuple[int, ...], torch.Tensor], ...],
    candidate: tuple[tuple[tuple[int, ...], torch.Tensor], ...],
    *,
    final_stage: bool,
) -> tuple[bool, dict[str, Any]]:
    if len(reference) != len(candidate):
        return False, {"reason": "probe result length differs"}
    token_exact = all(left[0] == right[0] for left, right in zip(reference, candidate, strict=True))
    if final_stage:
        boundary_exact = all(
            torch.equal(left[1], right[1]) for left, right in zip(reference, candidate, strict=True)
        )
        logits_close = all(
            torch.allclose(left[1].float(), right[1].float(), atol=1e-4, rtol=1e-4)
            for left, right in zip(reference, candidate, strict=True)
        )
        return token_exact, {
            "token_ids_exact": token_exact,
            "diagnostic_logits_exact": boundary_exact,
            "diagnostic_logits_close": logits_close,
        }
    boundary_exact = all(
        torch.equal(left[1], right[1]) for left, right in zip(reference, candidate, strict=True)
    )
    return boundary_exact, {
        "token_ids_exact": token_exact,
        "stage_boundary_exact": boundary_exact,
    }


def _cuda_device() -> ExecutionDevice:
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    properties = torch.cuda.get_device_properties(index)
    driver_reader = getattr(torch._C, "_cuda_getDriverVersion", None)
    driver = str(driver_reader()) if callable(driver_reader) else "unavailable"
    return ExecutionDevice(
        device_id=f"cuda:{index}",
        device_type="cuda",
        name=str(properties.name),
        uuid=str(getattr(properties, "uuid", "")) or None,
        total_memory_bytes=int(total),
        usable_memory_bytes=int(free),
        runtime_version=f"torch={torch.__version__};cuda={torch.version.cuda or 'unknown'}",
        driver_version=driver,
        features=Qwen3CudaFastPath.candidate_modes,
    )


def _descriptor_for_stage(
    request: LoadStageRequest,
    config: dict[str, Any],
) -> ResolvedModelDescriptor:
    architectures = config.get("architectures")
    architecture = (
        str(architectures[0])
        if isinstance(architectures, list) and architectures
        else str(config.get("model_type") or "Qwen3ForCausalLM")
    )
    fingerprint = request.model_content_fingerprint or (
        "sha256:"
        + hashlib.sha256(f"{request.model_id}@{request.model_revision}".encode()).hexdigest()
    )
    return ResolvedModelDescriptor(
        model_id=request.model_id,
        revision=request.model_revision,
        content_fingerprint=fingerprint,
        source_type="local",
        format="safetensors",
        architecture=architecture,
        files=(),
        quantization=request.quantization,
        weight_bytes=request.assignment.weight_bytes,
        layer_count=request.assignment.layer_end - request.assignment.layer_start,
        tokenizer_identity=request.tokenizer_revision,
    )


class Qwen3StageExecutor:
    """Adapt ``Qwen3StageModule`` to the product ``StageExecutor`` contract."""

    def __init__(
        self,
        *,
        module: Qwen3StageModule,
        request: LoadStageRequest,
        selected_fast_path: str,
        fallback_reason: str | None,
    ) -> None:
        self.module = module
        self.request = request
        self.selected_fast_path = selected_fast_path
        self.fast_path_fallback_reason = fallback_reason
        self._sessions: set[str] = set()
        self._closed = False
        self._graph_slots: list[_StageCudaGraphSlot] = []
        self._session_graph_slots: dict[str, _StageCudaGraphSlot] = {}
        self._cuda_graph_pool: Any | None = None
        self._graph_capture_failure_reason: str | None = None
        self._graph_capture_failures = 0
        parameters = module._parameters()
        unique_storages: dict[tuple[str, int], int] = {}
        for parameter in parameters.values():
            storage = parameter.untyped_storage()
            unique_storages[(str(parameter.device), int(storage.data_ptr()))] = int(
                storage.nbytes()
            )
        names = tuple(sorted(parameters))
        ownership_payload = json.dumps(names, separators=(",", ":")).encode("utf-8")
        self._ownership = WeightOwnership(
            stage_id=request.assignment.stage_id,
            layer_start=request.assignment.layer_start,
            layer_end=request.assignment.layer_end,
            parameter_names=names,
            parameter_bytes=sum(unique_storages.values()),
            parameter_count=sum(int(parameter.numel()) for parameter in parameters.values()),
            owns_embeddings=request.assignment.owns_embeddings,
            owns_final_norm=request.assignment.owns_final_norm,
            owns_output_projection=request.assignment.owns_output_projection,
            ownership_hash=hashlib.sha256(ownership_payload).hexdigest(),
        )

    @classmethod
    def from_load_request(
        cls,
        request: LoadStageRequest,
        resolved_model_path: Path | None,
        *,
        fast_path_profile_store: FastPathProfileStore | None = None,
    ) -> Qwen3StageExecutor:
        if resolved_model_path is None:
            raise FileNotFoundError("native Qwen3 loading requires an acquired checkpoint")
        root = resolved_model_path.expanduser().resolve()
        config_path = root / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen3 checkpoint config is missing: {config_path}")
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ValueError("Qwen3 checkpoint config must be a JSON object")
        requested_mode = request.fast_path_mode.strip().lower()
        auto_cuda = (
            requested_mode in {"", "auto", "auto-exact"}
            and torch.device(request.device).type == "cuda"
        )
        if auto_cuda and fast_path_profile_store is not None:
            descriptor = _descriptor_for_stage(request, config_value)
            device = _cuda_device()
            maximum_context = max(
                16,
                min(
                    int(config_value.get("max_position_embeddings", 4096)),
                    request.fast_path_context_bucket,
                ),
            )
            batch_bucket = request.fast_path_batch_bucket
            reference_signatures: tuple[tuple[tuple[int, ...], torch.Tensor], ...] | None = None

            def configured(options: Qwen3EngineOptions) -> Qwen3EngineOptions:
                return replace(
                    options,
                    max_sequence_length=maximum_context,
                    max_batch_size=batch_bucket,
                    cuda_graph_batch_sizes=(batch_bucket,),
                )

            def benchmark_runner(
                candidate: str,
                candidate_options: Qwen3EngineOptions,
            ) -> FastPathMeasurement:
                nonlocal reference_signatures
                if reference_signatures is None:
                    reference_options = Qwen3EngineOptions.from_values(
                        profile=Qwen3ExecutionProfile.CORRECTNESS,
                        max_sequence_length=maximum_context,
                        max_batch_size=batch_bucket,
                    )
                    reference_module = _load_module(
                        request,
                        root=root,
                        config=config_value,
                        options=reference_options,
                    )
                    _, _, reference_signatures = _probe_module(
                        reference_module,
                        request=request,
                        request_id="fast-path-reference",
                    )
                    del reference_module
                    torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device.device_id)
                prepared_at = time.perf_counter()
                module = _load_module(
                    request,
                    root=root,
                    config=config_value,
                    options=configured(candidate_options),
                )
                probe = (
                    _probe_manual_cuda_graph_module
                    if candidate == "manual_cuda_graph"
                    else _probe_module
                )
                prefill_ms, decode_ms, signatures = probe(
                    module,
                    request=request,
                    request_id=f"fast-path-{candidate}",
                )
                prepare_seconds = time.perf_counter() - prepared_at
                memory_bytes = int(torch.cuda.max_memory_allocated(device.device_id))
                exact, diagnostics = _exactness(
                    reference_signatures,
                    signatures,
                    final_stage=request.assignment.owns_output_projection,
                )
                del module
                torch.cuda.empty_cache()
                mean_decode_ms = statistics.fmean(decode_ms)
                coefficient = (
                    statistics.pstdev(decode_ms) / mean_decode_ms
                    if len(decode_ms) > 1 and mean_decode_ms > 0
                    else 0.0
                )
                return FastPathMeasurement(
                    fast_path_id="qwen3_cuda",
                    candidate_mode=candidate,
                    exactness_passed=exact,
                    prefill_tokens_s=(
                        min(8, request.fast_path_context_bucket) / max(prefill_ms / 1000, 1e-9)
                    ),
                    decode_tokens_s=len(decode_ms) / max(sum(decode_ms) / 1000, 1e-9),
                    ttft_ms=prefill_ms,
                    memory_bytes=memory_bytes,
                    prepare_seconds=prepare_seconds,
                    supported_batch_sizes=(batch_bucket,),
                    batch_bucket=batch_bucket,
                    context_bucket=request.fast_path_context_bucket,
                    repeat_count=len(decode_ms),
                    coefficient_of_variation=coefficient,
                    failure_reason=(None if exact else "exactness oracle failed"),
                    diagnostics=diagnostics,
                )

            def prepare_runner(
                _candidate: str,
                candidate_options: Qwen3EngineOptions,
            ) -> Qwen3StageModule:
                return _load_module(
                    request,
                    root=root,
                    config=config_value,
                    options=configured(candidate_options),
                )

            profile_key = profile_key_from_runtime(
                model_content_fingerprint=descriptor.content_fingerprint,
                adapter_id="qwen3_dense",
                adapter_version="3",
                engine_version="native-stage-streaming-v3",
                fast_path_id="qwen3_cuda",
                device_uuid=device.uuid,
                driver_version=device.driver_version,
                runtime_version=device.runtime_version,
                dtype=request.dtype,
                quantization=request.quantization,
                stage_ownership=request.assignment.to_dict(),
                batch_bucket=batch_bucket,
                context_bucket=request.fast_path_context_bucket,
            )
            if (
                request.fast_path_profile_fingerprint is not None
                and request.fast_path_profile_fingerprint != profile_key.fingerprint
            ):
                raise ValueError(
                    "requested fast-path profile fingerprint does not match this stage runtime"
                )
            registry = NativeFastPathRegistry((Qwen3CudaFastPath(),))
            try:
                prepared = registry.admit(
                    fast_path_id=request.fast_path_id or "qwen3_cuda",
                    model=descriptor,
                    device=device,
                    profile_key=profile_key,
                    profile_store=fast_path_profile_store,
                    objective=request.fast_path_objective,
                    benchmark_kwargs={
                        "runner": benchmark_runner,
                        "batch_bucket": batch_bucket,
                        "context_bucket": request.fast_path_context_bucket,
                    },
                    prepare_kwargs={"runner": prepare_runner},
                )
                module = prepared.implementation
                if not isinstance(module, Qwen3StageModule):
                    raise TypeError("Qwen3 fast-path preparation returned an invalid module")
                selected = prepared.candidate_mode
                fallback = None
                request = request.model_copy(
                    update={"fast_path_profile_fingerprint": prepared.profile_fingerprint}
                )
            except FastPathAdmissionError as exc:
                options, selected, _ = _engine_options(
                    # No optimized candidate passed the exactness oracle.  The
                    # only admissible fallback is the adapter reference path;
                    # an unverified GPU-native mode would turn a negative
                    # measurement into a silent optimization selection.
                    request.model_copy(update={"fast_path_mode": "eager"}),
                    config=config_value,
                )
                module = _load_module(
                    request,
                    root=root,
                    config=config_value,
                    options=options,
                )
                fallback = f"exact fast-path admission failed: {exc}"
        else:
            options, selected, fallback = _engine_options(request, config=config_value)
            module = _load_module(
                request,
                root=root,
                config=config_value,
                options=options,
            )
        return cls(
            module=module,
            request=request,
            selected_fast_path=selected,
            fallback_reason=fallback,
        )

    @property
    def ownership(self) -> WeightOwnership:
        return self._ownership

    def open_session(self, session_id: str) -> None:
        if self._closed:
            raise RuntimeError("stage executor is closed")
        if not session_id:
            raise ValueError("stage session ID cannot be empty")
        if session_id in self._sessions:
            raise ValueError("stage session is already open")
        self._sessions.add(session_id)
        if self.selected_fast_path != "manual_cuda_graph":
            return
        available = next(
            (slot for slot in self._graph_slots if slot.session_id is None),
            None,
        )
        if available is None and self._graph_capture_failure_reason is None:
            slot_id = (
                f"cuda-graph-stage-{self.request.assignment.stage_id}-slot-{len(self._graph_slots)}"
            )
            try:
                available = _capture_stage_cuda_graph(
                    self.module,
                    slot_id=slot_id,
                    input_template=_probe_input(self.module, sequence_length=1),
                    pool=self._cuda_graph_pool,
                )
                self._cuda_graph_pool = available.pool
                self._graph_slots.append(available)
            except Exception as exc:
                self._graph_capture_failures += 1
                self._graph_capture_failure_reason = f"{type(exc).__name__}: {exc}"
                # A failed capture may have allocated a slot cache before the
                # CUDA runtime rejected the graph. It has no captured pointer
                # contract and can therefore be released normally.
                with suppress(Exception):
                    self.module.reset_cache(slot_id)
                if not self._graph_slots:
                    self.module.end_cuda_graph_decode()
        if available is not None:
            available.session_id = session_id
            self._session_graph_slots[session_id] = available

    def _require_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"unknown Qwen3 stage session {session_id!r}")

    def _execute(
        self,
        *,
        session_id: str,
        value: torch.Tensor,
        cache_position_start: int,
        prefill: bool,
    ) -> StageExecutionResult:
        self._require_session(session_id)
        graph_slot = self._session_graph_slots.get(session_id)
        execution_id = graph_slot.slot_id if graph_slot is not None else session_id
        query_length = int(value.shape[1])
        metadata = StageExecutionMetadata(
            request_id=execution_id,
            token_position=cache_position_start,
            sequence_length=query_length,
        )
        cuda_value = value.to(
            device=self.module.device,
            dtype=torch.long if self.module.embed_tokens is not None else self.module.dtype,
        )
        started = time.perf_counter_ns()
        graph_sampled: torch.Tensor | None = None
        if not prefill and graph_slot is not None:
            output, graph_sampled = _replay_stage_cuda_graph(
                self.module,
                graph_slot,
                value=cuda_value,
                token_position=cache_position_start,
            )
        else:
            # Once graph buffers exist, all one-token eager calls must update
            # their shared position view even when this particular session is
            # using the per-stage static-cache fallback.
            if query_length == 1 and self.module._graph_position is not None:
                self.module.update_cuda_graph_position(cache_position_start)
            output = (
                self.module.prefill_cuda(cuda_value, metadata)
                if prefill
                else self.module.decode_cuda(cuda_value, metadata)
            )
        sampled: torch.Tensor | None = graph_sampled
        logits: torch.Tensor | None = None
        if self.module.lm_head is not None:
            logits = output
            if sampled is not None:
                pass
            elif self.module.engine_options.profile == Qwen3ExecutionProfile.FAST:
                sampled = self.module.sample_cuda(
                    output,
                    request_ids=(execution_id,),
                ).token_ids
            else:
                sampled = torch.argmax(output[:, -1, :], dim=-1)
            sampled = sampled.detach().to(device="cpu", dtype=torch.int64)
        if self.module.device.type == "cuda":
            torch.cuda.synchronize(self.module.device)
        compute_ns = time.perf_counter_ns() - started
        # Keep the boundary on its execution device. Intermediate stages copy
        # it exactly once while packing the direct peer message; the final
        # stage returns only compact sampled IDs and avoids a vocabulary-sized
        # CPU transfer on every token.
        boundary = output.detach()
        actual_fast_path = (
            "manual_cuda_graph"
            if graph_slot is not None and not prefill
            else (
                "static_cache"
                if self.selected_fast_path == "manual_cuda_graph" and graph_slot is None
                else self.selected_fast_path
            )
        )
        graph_fallback = (
            self._graph_capture_failure_reason
            if self.selected_fast_path == "manual_cuda_graph" and graph_slot is None
            else None
        )
        return StageExecutionResult(
            hidden_states=output,
            stage_boundary_hidden_states=boundary,
            router_logits=(),
            final_hidden_states=None,
            logits=logits,
            sampled_token_ids=sampled,
            all_sampled_token_ids=sampled,
            cache_sequence_length=cache_position_start + query_length,
            compute_ns=compute_ns,
            expert_metrics={
                "fast_path": actual_fast_path,
                "fast_path_requested": self.request.fast_path_mode,
                "fast_path_fallback": (graph_fallback or self.fast_path_fallback_reason or "none"),
                "cuda_graph_replays": graph_slot.replay_count if graph_slot else 0,
                "cuda_graph_capture_ms": graph_slot.capture_ms if graph_slot else 0.0,
            },
        )

    def execute_prefill(
        self,
        *,
        session_id: str,
        token_ids: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._execute(
            session_id=session_id,
            value=token_ids,
            cache_position_start=cache_position_start,
            prefill=True,
        )

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> StageExecutionResult:
        return self._execute(
            session_id=session_id,
            value=hidden_states,
            cache_position_start=cache_position_start,
            prefill=False,
        )

    def kv_cache_bytes(self, session_id: str) -> int:
        self._require_session(session_id)
        graph_slot = self._session_graph_slots.get(session_id)
        cache_id = graph_slot.slot_id if graph_slot is not None else session_id
        summaries = self.module.inspect_cache(cache_id)
        return sum(
            int(item.get("reserved_bytes", item.get("cache_bytes", 0))) for item in summaries
        )

    def close_session(self, session_id: str) -> int:
        self._require_session(session_id)
        graph_slot = self._session_graph_slots.pop(session_id, None)
        if graph_slot is not None:
            released = self.module.reset_cuda_graph_slot(graph_slot.slot_id)
            graph_slot.session_id = None
        else:
            released = self.module.reset_cache(session_id)
        self._sessions.remove(session_id)
        return released

    def cancel_session(self, session_id: str) -> int:
        return self.close_session(session_id)

    def fast_path_status(self) -> dict[str, Any]:
        return {
            "requested": self.request.fast_path_mode,
            "selected": self.selected_fast_path,
            "profile_fingerprint": self.request.fast_path_profile_fingerprint,
            "fallback_reason": self.fast_path_fallback_reason,
            "cuda_graph": {
                "slot_count": len(self._graph_slots),
                "active_slot_count": len(self._session_graph_slots),
                "capture_failure_count": self._graph_capture_failures,
                "capture_failure_reason": self._graph_capture_failure_reason,
                "capture_ms": sum(slot.capture_ms for slot in self._graph_slots),
                "replay_count": sum(slot.replay_count for slot in self._graph_slots),
                "admission_required_bytes": sum(
                    slot.admission_required_bytes for slot in self._graph_slots
                ),
                "admission_available_bytes": (
                    min(
                        (slot.admission_available_bytes for slot in self._graph_slots),
                        default=0,
                    )
                ),
            },
            "module": self.module.state_summary(),
        }

    def close(self) -> None:
        for session_id in list(self._sessions):
            self.cancel_session(session_id)
        slot_ids = [slot.slot_id for slot in self._graph_slots]
        self._session_graph_slots.clear()
        self._graph_slots.clear()
        self._cuda_graph_pool = None
        self.module.end_cuda_graph_decode()
        for slot_id in slot_ids:
            self.module.reset_cache(slot_id)
        self._closed = True


__all__ = ["Qwen3StageExecutor"]
