"""Dense Qwen3 adapter and partial stage module.

The worker module constructs only the decoder layers and endpoint components
owned by its stage. It never calls ``Qwen3ForCausalLM.from_pretrained``.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.config.models import ModelManifest, StageDefinition
from swarm_inference.exceptions import (
    IntegrityError,
    UnsupportedArchitectureError,
    UnsupportedCacheFormatError,
)
from swarm_inference.experiments.fanout_lifecycle import lifecycle_recorder
from swarm_inference.model.adapter import (
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
)
from swarm_inference.model.qwen3_cache import StaticStageKVCache
from swarm_inference.model.qwen3_runtime import (
    AttentionBackend,
    AttentionBackendEvidence,
    CompileDiagnostics,
    Qwen3CacheBackend,
    Qwen3CompileMode,
    Qwen3EngineOptions,
    Qwen3ExecutionProfile,
    attention_backend_availability,
    auto_attention_candidates,
    nvtx_range,
    resolve_cache_torch_dtype,
    validate_attention_backend,
)
from swarm_inference.model.qwen3_sampling import (
    SamplingParameters,
    SamplingResult,
    SamplingState,
    sample_final_logits,
)
from swarm_inference.model.stage_module import (
    BatchExecutionMetadata,
    StageExecutionMetadata,
)
from swarm_inference.protocol.checksums import sha256_file

_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(slots=True)
class StageLocalKVCache:
    """A global-index-preserving cache owned by one contiguous stage."""

    cache: Any
    request_id: str
    model_revision: str
    stage_id: int
    layer_start: int
    layer_end: int
    route_generation: int
    cache_generation: int
    sequence_length: int = 0
    replay_generation: int = 0
    allocation_count: int = 0
    append_count: int = 0

    def global_to_local(self, global_layer_index: int) -> int:
        if not self.layer_start <= global_layer_index < self.layer_end:
            raise IndexError(
                f"global layer {global_layer_index} is outside stage "
                f"[{self.layer_start}, {self.layer_end})"
            )
        return global_layer_index - self.layer_start

    def local_to_global(self, local_layer_index: int) -> int:
        layer_count = self.layer_end - self.layer_start
        if not 0 <= local_layer_index < layer_count:
            raise IndexError(f"local layer {local_layer_index} is outside [0, {layer_count})")
        return self.layer_start + local_layer_index

    def advance(self, *, token_position: int, query_length: int) -> None:
        if token_position != self.sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} cache position mismatch: "
                f"expected={self.sequence_length} actual={token_position}"
            )
        self.sequence_length += query_length
        # DynamicCache replaces each owned key/value tensor on every model
        # invocation. Count those logical tensor allocations so the experiment
        # can compare its allocation behaviour with the fixed static cache.
        self.allocation_count += 2 * (self.layer_end - self.layer_start)
        self.append_count += 1

    def _owned_layers(self) -> list[tuple[int, Any]]:
        layers = getattr(self.cache, "layers", None)
        if layers is None:
            return []
        if len(layers) < self.layer_end:
            raise UnsupportedCacheFormatError(
                f"Transformers cache exposes {len(layers)} layers but stage "
                f"{self.stage_id} owns global layer {self.layer_end - 1}"
            )
        return [
            (global_index, layers[global_index])
            for global_index in range(self.layer_start, self.layer_end)
        ]

    def memory_bytes(self) -> int:
        total = 0
        for _, layer in self._owned_layers():
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if tensor is not None and hasattr(tensor, "numel"):
                    total += int(tensor.numel() * tensor.element_size())
        if not getattr(self.cache, "layers", None):
            for name in ("key_cache", "value_cache"):
                tensors = getattr(self.cache, name, [])
                for global_index in range(self.layer_start, self.layer_end):
                    if global_index < len(tensors):
                        tensor = tensors[global_index]
                        if tensor is not None:
                            total += int(tensor.numel() * tensor.element_size())
        return total

    def summary(self) -> dict[str, Any]:
        layers: list[dict[str, Any]] = []
        for global_index, layer in self._owned_layers():
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            initialised = bool(getattr(layer, "is_initialized", False))
            if callable(getattr(layer, "is_initialized", None)):
                initialised = bool(layer.is_initialized())
            layers.append(
                {
                    "global_layer_index": global_index,
                    "local_layer_index": self.global_to_local(global_index),
                    "initialised": initialised,
                    "key_shape": list(key.shape) if key is not None else None,
                    "value_shape": list(value.shape) if value is not None else None,
                    "dtype": str(key.dtype).removeprefix("torch.") if key is not None else None,
                }
            )
        return {
            "backend": "dynamic_reference",
            "request_id": self.request_id,
            "model_revision": self.model_revision,
            "stage_id": self.stage_id,
            "route_generation": self.route_generation,
            "cache_generation": self.cache_generation,
            "replay_generation": self.replay_generation,
            "sequence_length": self.sequence_length,
            "owned_layer_count": self.layer_end - self.layer_start,
            "initialised_layer_count": sum(1 for layer in layers if layer["initialised"]),
            "cache_bytes": self.memory_bytes(),
            "reserved_bytes": self.memory_bytes(),
            "allocation_count": self.allocation_count,
            "append_count": self.append_count,
            "fragmentation_bytes": 0,
            "fragmentation_fraction": 0.0,
            "allocation_count_definition": "logical key/value tensor replacements",
            "layers": layers,
        }


def _normalise_dtype(value: Any) -> str:
    text = str(value).upper().replace("SAFETENSORS.", "")
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _weight_map(model_path: Path) -> dict[str, str]:
    indexes = sorted(model_path.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) > 1:
            raise IntegrityError(
                f"multiple safetensors index files found: {[path.name for path in indexes]}"
            )
        payload = json.loads(indexes[0].read_text(encoding="utf-8"))
        index_mapping = payload.get("weight_map")
        if not isinstance(index_mapping, dict) or not index_mapping:
            raise IntegrityError(f"invalid weight_map in {indexes[0]}")
        return {str(name): str(file) for name, file in index_mapping.items()}
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise IntegrityError(f"no safetensors weights found under {model_path}")
    from safetensors import safe_open

    source_mapping: dict[str, str] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors handle is not iterable
                if name in source_mapping:
                    raise IntegrityError(f"tensor {name} occurs in multiple source files")
                source_mapping[name] = file.name
    return source_mapping


class Qwen3Adapter:
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"qwen3"})
    supported_architectures: ClassVar[frozenset[str]] = frozenset({"Qwen3ForCausalLM"})

    def supports(self, config: Any) -> bool:
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        if not isinstance(config, dict):
            return False
        model_type = str(config.get("model_type", "")).lower()
        architectures = set(config.get("architectures") or [])
        return model_type in self.supported_model_types and (
            not architectures or bool(architectures & self.supported_architectures)
        )

    def map_tensor_to_component(self, tensor_name: str) -> ComponentRef:
        if tensor_name.startswith("model.embed_tokens."):
            return ComponentRef(ComponentKind.EMBEDDING)
        match = _LAYER_PATTERN.match(tensor_name)
        if match:
            return ComponentRef(
                ComponentKind.DECODER_LAYER,
                layer_index=int(match.group(1)),
            )
        if tensor_name.startswith("model.norm."):
            return ComponentRef(ComponentKind.FINAL_NORM)
        if tensor_name.startswith("lm_head."):
            return ComponentRef(ComponentKind.OUTPUT_HEAD)
        raise UnsupportedArchitectureError(
            f"Qwen3 tensor cannot be mapped to a required component: {tensor_name}"
        )

    def describe(
        self,
        model_path: Path,
        *,
        model_id: str,
        model_revision: str,
    ) -> ModelDescription:
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise IntegrityError(f"model config is missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not self.supports(config):
            raise UnsupportedArchitectureError(
                f"unsupported architecture: model_type={config.get('model_type')!r}, "
                f"architectures={config.get('architectures')!r}; supported=dense Qwen3"
            )
        mapping = _weight_map(model_path)
        by_file: dict[str, list[str]] = {}
        for name, source in mapping.items():
            by_file.setdefault(source, []).append(name)
        from safetensors import safe_open

        tensors: list[TensorInfo] = []
        for source, names in sorted(by_file.items()):
            source_path = model_path / source
            if not source_path.is_file():
                raise IntegrityError(f"safetensors index references missing file {source_path}")
            with safe_open(source_path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                for name in sorted(names):
                    if name not in keys:
                        raise IntegrityError(
                            f"safetensors index maps {name} to {source}, but key is absent"
                        )
                    tensor_slice = handle.get_slice(name)
                    shape = tuple(int(value) for value in tensor_slice.get_shape())
                    dtype = _normalise_dtype(tensor_slice.get_dtype())
                    try:
                        width = _DTYPE_BYTES[dtype]
                    except KeyError as exc:
                        raise UnsupportedArchitectureError(
                            f"unsupported safetensors dtype {dtype} for tensor {name}"
                        ) from exc
                    tensors.append(
                        TensorInfo(
                            name=name,
                            source_file=source,
                            dtype=dtype,
                            shape=shape,
                            bytes=math.prod(shape) * width,
                            component=self.map_tensor_to_component(name),
                        )
                    )
        return ModelDescription(
            model_id=model_id,
            model_revision=model_revision,
            model_path=model_path,
            config=config,
            tensors=sorted(tensors, key=lambda item: item.name),
            source_file_hashes={
                source: sha256_file(model_path / source) for source in sorted(set(mapping.values()))
            },
            config_file_hashes={
                path.name: sha256_file(path)
                for path in sorted(model_path.iterdir())
                if path.is_file()
                and path.name
                in {
                    "config.json",
                    "generation_config.json",
                }
            },
            tokenizer_file_hashes={
                path.name: sha256_file(path)
                for path in sorted(model_path.iterdir())
                if path.is_file()
                and (
                    path.name.startswith("tokenizer")
                    or path.name.startswith("vocab")
                    or path.name.startswith("merges")
                    or path.name.startswith("special_tokens")
                    or path.suffix == ".model"
                )
            },
        )

    def create_stage_module(
        self,
        config: Any,
        stage: StageDefinition,
        device: Any,
        dtype: Any,
    ) -> Qwen3StageModule:
        return Qwen3StageModule(
            config=config,
            stage=stage,
            device=device,
            dtype=dtype,
        )

    def load_stage_weights(
        self,
        module: Qwen3StageModule,
        shard_path: Path,
        *,
        manifest: ModelManifest,
    ) -> list[str]:
        return module.load_weights(shard_path, manifest=manifest)


class Qwen3StageModule:
    """A torch module containing one contiguous Qwen3 layer interval."""

    def __init__(
        self,
        *,
        config: Any,
        stage: StageDefinition,
        device: Any,
        dtype: Any,
        engine_options: Qwen3EngineOptions | None = None,
    ) -> None:
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RMSNorm,
            Qwen3RotaryEmbedding,
        )

        if isinstance(config, dict):
            config = Qwen3Config.from_dict(config)
        if getattr(config, "model_type", None) != "qwen3":
            raise UnsupportedArchitectureError(
                f"Qwen3StageModule received model_type={getattr(config, 'model_type', None)!r}"
            )
        self.engine_options = engine_options or Qwen3EngineOptions.from_values(
            max_sequence_length=min(
                4096,
                int(getattr(config, "max_position_embeddings", 4096)),
            )
        )
        self.execution_profile = self.engine_options.profile.value
        self.attention_evidence = AttentionBackendEvidence(
            requested=self.engine_options.attention_backend.value
        )
        availability = attention_backend_availability(torch)
        # FlashInfer is a valid user-facing selection but this Transformers
        # decoder does not expose a FlashInfer attention adapter.
        availability[AttentionBackend.FLASHINFER.value] = False
        self.attention_evidence.available = dict(availability)
        validate_attention_backend(
            self.engine_options.attention_backend,
            availability=availability,
        )
        if self.engine_options.profile == Qwen3ExecutionProfile.CORRECTNESS:
            selected_attention = AttentionBackend.EAGER
        elif self.engine_options.attention_backend == AttentionBackend.AUTO:
            candidates = auto_attention_candidates(availability)
            if not candidates:
                raise RuntimeError("no compatible Qwen3 attention backend is available")
            # The final auto choice is benchmarked after weights load. SDPA is
            # the safe provisional choice on installations without flash-attn.
            selected_attention = candidates[0]
        else:
            selected_attention = self.engine_options.attention_backend
        config._attn_implementation = selected_attention.value
        self.attention_backend = selected_attention.value
        self.attention_evidence.selected = selected_attention.value
        if getattr(config, "use_sliding_window", False) or (
            "sliding_attention" in getattr(config, "layer_types", [])
        ):
            raise UnsupportedCacheFormatError(
                "sliding-window Qwen3 stage execution is not implemented"
            )
        self.torch = torch
        self.config = config
        self.stage = stage
        self.stage_id = stage.stage_id
        self.required_memory_bytes = stage.required_memory_bytes
        self.device = torch.device(device)
        self.dtype = dtype
        if (
            self.device.type == "cuda"
            and self.engine_options.profile == Qwen3ExecutionProfile.CORRECTNESS
        ):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        elif self.device.type == "cuda":
            torch.use_deterministic_algorithms(False)
            # Kernel scheduling remains non-deterministic in qwen3_fast, but
            # reduced-precision accumulation can change a near-tied greedy
            # argmax as batch shape changes (observed at output token 42 for
            # Qwen3-0.6B, batch 4).  FP32 accumulation for BF16/FP16 GEMMs is
            # therefore an output-identity constraint, not a performance
            # profiling setting.
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.embed_tokens = (
            torch.nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                getattr(config, "pad_token_id", None),
                device=self.device,
                dtype=dtype,
            )
            if stage.owns_embeddings
            else None
        )
        self.layers = torch.nn.ModuleDict(
            {
                str(layer_index): Qwen3DecoderLayer(config, layer_index).to(
                    device=self.device, dtype=dtype
                )
                for layer_index in range(stage.layer_start, stage.layer_end)
            }
        )
        self._layer_modules = tuple(
            self.layers[str(layer_index)]
            for layer_index in range(stage.layer_start, stage.layer_end)
        )
        self._layer_cache_argument_styles: tuple[str, ...] = tuple(
            self._cache_argument_style(layer) for layer in self._layer_modules
        )
        self._layer_calls = tuple(
            self._bind_layer_call(layer, style)
            for layer, style in zip(
                self._layer_modules,
                self._layer_cache_argument_styles,
                strict=True,
            )
        )
        self.norm = (
            Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
                device=self.device, dtype=dtype
            )
            if stage.owns_final_norm
            else None
        )
        self.lm_head = (
            torch.nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                device=self.device,
                dtype=dtype,
            )
            if stage.owns_output_head
            else None
        )
        if (
            self.embed_tokens is not None
            and self.lm_head is not None
            and bool(getattr(config, "tie_word_embeddings", False))
        ):
            # A one-stage layout must retain the official model's tied parameter
            # identity instead of allocating a second vocabulary matrix.
            self.lm_head.weight = self.embed_tokens.weight
        # Match the official full-model construction order exactly: Transformers
        # creates RoPE frequencies on CPU and subsequently moves the model.  Direct
        # CUDA construction changes two float32 inv_freq values for Qwen3-0.6B by
        # one ULP, which phase-amplifies on 512-token prompts and breaks boundary
        # validation even though short prompts and greedy tokens still agree.
        self.rotary_emb = Qwen3RotaryEmbedding(config=config).to(self.device)
        maximum_positions = int(getattr(config, "max_position_embeddings", 4096))
        if self.engine_options.max_sequence_length > maximum_positions:
            raise ValueError(
                "configured fast-cache sequence length exceeds model maximum: "
                f"{self.engine_options.max_sequence_length} > {maximum_positions}"
            )
        self._position_buffer = torch.arange(
            self.engine_options.max_sequence_length,
            device=self.device,
            dtype=torch.long,
        )
        self._causal_mask_cache: dict[tuple[int, int], Any] = {}
        self.model_revision = "unloaded"
        self._caches: dict[tuple[str, str, int, int, int], StageLocalKVCache] = {}
        self._static_caches: dict[
            tuple[tuple[str, ...], str, int, int, int],
            StaticStageKVCache,
        ] = {}
        self._dynamic_cache_fallback_groups: set[tuple[str, ...]] = set()
        self._cache_replay_generations: dict[str, int] = {}
        self._cache_history: list[dict[str, Any]] = []
        self._boundary_records: dict[str, dict[str, Any]] = {}
        self._last_layer_hidden: Any | None = None
        self._transfer_metrics: dict[str, float | int] = {
            "operation_count": 0,
            "host_to_device_copy_ms": 0.0,
            "device_to_host_copy_ms": 0.0,
            "boundary_diagnostic_copy_ms": 0.0,
            "cuda_execution_ms": 0.0,
            "host_to_device_bytes": 0,
            "device_to_host_bytes": 0,
            "transport_tensor_bytes": 0,
        }
        self._transfer_history: list[dict[str, Any]] = []
        self._finite_output_checks = 0
        self._fast_forward_count = 0
        self._fast_batch_forward_count = 0
        self._full_logit_return_count = 0
        self._sampled_token_return_count = 0
        self._compile_diagnostics = CompileDiagnostics(
            requested_mode=self.engine_options.compile_mode.value
        )
        self._compiled_prefill: Callable[..., Any] | None = None
        self._compiled_decode: Callable[..., Any] | None = None
        self._attention_tuning_seconds = 0.0
        self._sampling_state = SamplingState(torch, self.device)
        self._graph_position: Any | None = None
        self._graph_position_ids: Any | None = None
        self._graph_attention_mask: Any | None = None
        self._graph_batch_size = 0
        self.loaded_source_tensors: list[str] = []
        for layer in self.layers.values():
            layer.eval()
            layer.requires_grad_(False)
        for endpoint in (self.embed_tokens, self.norm, self.lm_head, self.rotary_emb):
            if endpoint is not None:
                endpoint.eval()
                endpoint.requires_grad_(False)

    @staticmethod
    def _cache_argument_style(layer: Any) -> str:
        """Resolve Transformers cache spelling exactly once during stage load."""

        signature = inspect.signature(layer.forward)
        if "past_key_values" in signature.parameters:
            return "past_key_values"
        if "past_key_value" in signature.parameters:
            return "past_key_value"
        raise UnsupportedCacheFormatError(
            "installed Transformers Qwen3 decoder has no supported cache argument"
        )

    @staticmethod
    def _bind_layer_call(layer: Any, cache_argument_style: str) -> Callable[..., Any]:
        if cache_argument_style == "past_key_values":

            def call(
                hidden_states: Any,
                attention_mask: Any,
                position_ids: Any,
                use_cache: bool,
                position_embeddings: Any,
                cache_position: Any,
                cache: Any,
            ) -> Any:
                output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    past_key_values=cache,
                )
                return output[0] if isinstance(output, tuple) else output

            return call
        if cache_argument_style == "past_key_value":

            def legacy_call(
                hidden_states: Any,
                attention_mask: Any,
                position_ids: Any,
                use_cache: bool,
                position_embeddings: Any,
                cache_position: Any,
                cache: Any,
            ) -> Any:
                output = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position,
                    past_key_value=cache,
                )
                return output[0] if isinstance(output, tuple) else output

            return legacy_call
        raise UnsupportedCacheFormatError(
            f"unsupported bound Qwen3 cache argument style {cache_argument_style!r}"
        )

    def _new_cache(self) -> Any:
        from transformers import DynamicCache

        try:
            return DynamicCache(config=self.config)
        except TypeError:
            return DynamicCache()

    def warmup(self, *, sequence_length: int = 128) -> dict[str, object]:
        """Run a disposable representative operation without retaining KV state."""

        if sequence_length <= 0:
            raise ValueError("warmup sequence length must be positive")
        started = time.perf_counter_ns()
        if self.embed_tokens is not None:
            inputs = self.torch.zeros(
                (1, sequence_length),
                device=self.device,
                dtype=self.torch.long,
            )
        else:
            inputs = self.torch.zeros(
                (1, sequence_length, self.config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
        with self.torch.inference_mode():
            output = self.forward(
                inputs,
                request_id=f"stage-local-warmup-{self.stage_id}",
                token_position=0,
                cache_generation=0,
                route_generation=0,
                use_cache=False,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        duration_ns = time.perf_counter_ns() - started
        output_shape = [int(value) for value in output.shape]
        del output
        del inputs
        return {
            "sequence_length": sequence_length,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "output_shape": output_shape,
            "disposable_cache": True,
            "duration_seconds": duration_ns / 1_000_000_000,
        }

    def _attention_probe_input(self) -> Any:
        sequence_length = min(64, self.engine_options.max_sequence_length)
        if self.embed_tokens is not None:
            return self.torch.arange(
                sequence_length,
                device=self.device,
                dtype=self.torch.long,
            ).remainder(int(self.config.vocab_size))[None, :]
        return self.torch.linspace(
            -0.5,
            0.5,
            steps=sequence_length * int(self.config.hidden_size),
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, sequence_length, int(self.config.hidden_size))

    def _probe_attention_forward(self, inputs: Any) -> Any:
        with self.torch.inference_mode():
            return self.forward(
                inputs,
                request_id=f"attention-startup-check-{self.stage_id}",
                token_position=0,
                cache_generation=0,
                route_generation=0,
                use_cache=False,
            )

    def _autotune_attention_backend(self) -> None:
        if self.engine_options.profile != Qwen3ExecutionProfile.FAST:
            return
        started = time.perf_counter()
        requested = self.engine_options.attention_backend
        availability = self.attention_evidence.available
        if requested == AttentionBackend.AUTO:
            candidates = auto_attention_candidates(availability)
        else:
            candidates = (requested,)
        inputs = self._attention_probe_input()
        original_backend = self.attention_backend
        self.config._attn_implementation = AttentionBackend.EAGER.value
        self.attention_backend = AttentionBackend.EAGER.value
        reference = self._probe_attention_forward(inputs)
        reference_selection = self.torch.argmax(reference[:, -1, :], dim=-1)
        successful: list[tuple[AttentionBackend, float]] = []
        for candidate in candidates:
            if not availability.get(candidate.value, False):
                self.attention_evidence.startup_correct[candidate.value] = False
                self.attention_evidence.diagnostics[candidate.value] = "backend unavailable"
                continue
            self.config._attn_implementation = candidate.value
            self.attention_backend = candidate.value
            try:
                candidate_output = self._probe_attention_forward(inputs)
                candidate_selection = self.torch.argmax(
                    candidate_output[:, -1, :],
                    dim=-1,
                )
                selection_matches = bool(self.torch.equal(candidate_selection, reference_selection))
                finite = bool(self.torch.isfinite(candidate_output).all().item())
                correct = selection_matches and finite
                self.attention_evidence.startup_correct[candidate.value] = correct
                if not correct:
                    self.attention_evidence.diagnostics[candidate.value] = (
                        "startup greedy token or finite-output check failed"
                    )
                    continue
                timings: list[float] = []
                if self.device.type == "cuda":
                    for _ in range(5):
                        start_event = self.torch.cuda.Event(  # type: ignore[no-untyped-call]
                            enable_timing=True
                        )
                        end_event = self.torch.cuda.Event(  # type: ignore[no-untyped-call]
                            enable_timing=True
                        )
                        start_event.record()
                        self._probe_attention_forward(inputs)
                        end_event.record()
                        end_event.synchronize()
                        timings.append(float(start_event.elapsed_time(end_event)))
                else:
                    for _ in range(3):
                        timing_started = time.perf_counter()
                        self._probe_attention_forward(inputs)
                        timings.append((time.perf_counter() - timing_started) * 1000)
                median_ms = statistics.median(timings)
                self.attention_evidence.median_cuda_ms[candidate.value] = median_ms
                successful.append((candidate, median_ms))
            except Exception as exc:
                self.attention_evidence.startup_correct[candidate.value] = False
                self.attention_evidence.diagnostics[candidate.value] = (
                    f"{type(exc).__name__}: {exc}"
                )
        if not successful:
            self.config._attn_implementation = original_backend
            self.attention_backend = original_backend
            raise RuntimeError(
                "no requested Qwen3 attention backend passed the startup correctness check: "
                f"{self.attention_evidence.diagnostics}"
            )
        selected = min(successful, key=lambda item: item[1])[0]
        self.config._attn_implementation = selected.value
        self.attention_backend = selected.value
        self.attention_evidence.selected = selected.value
        self._attention_tuning_seconds = time.perf_counter() - started
        del reference
        del inputs

    def _causal_mask(
        self,
        hidden_states: Any,
        *,
        token_position: int,
    ) -> Any:
        torch = self.torch
        batch, query_length, _ = hidden_states.shape
        key_length = token_position + query_length
        key = (token_position, query_length)
        cached = self._causal_mask_cache.get(key)
        if cached is None:
            query_positions = self._position_buffer[token_position : token_position + query_length][
                :, None
            ]
            key_positions = self._position_buffer[:key_length][None, :]
            allowed = key_positions <= query_positions
            minimum = torch.finfo(hidden_states.dtype).min
            mask = torch.empty(
                (query_length, key_length),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            mask.fill_(minimum)
            mask.masked_fill_(allowed, 0)
            cached = mask[None, None, :, :]
            self._causal_mask_cache[key] = cached
        return cached.expand(batch, 1, query_length, key_length)

    def forward(
        self,
        inputs: Any,
        *,
        request_id: str,
        token_position: int,
        cache_generation: int = 0,
        route_generation: int = 0,
        use_cache: bool = True,
    ) -> Any:
        torch = self.torch
        with nvtx_range(
            torch,
            "embedding",
            enabled=self.engine_options.nvtx_enabled,
        ):
            if self.embed_tokens is not None:
                if inputs.dtype not in {torch.int32, torch.int64}:
                    raise ValueError("embedding-owning stage requires integer token IDs")
                hidden_states = self.embed_tokens(inputs.to(self.device))
            else:
                hidden_states = inputs.to(device=self.device, dtype=self.dtype)
        cache_key = (
            request_id,
            self.model_revision,
            self.stage_id,
            route_generation,
            cache_generation,
        )
        cache_record = self._caches.get(cache_key)
        if use_cache and cache_record is None:
            if token_position != 0:
                raise UnsupportedCacheFormatError(
                    f"missing cache for request={request_id} at token_position={token_position}"
                )
            cache_record = StageLocalKVCache(
                cache=self._new_cache(),
                request_id=request_id,
                model_revision=self.model_revision,
                stage_id=self.stage_id,
                layer_start=self.stage.layer_start,
                layer_end=self.stage.layer_end,
                route_generation=route_generation,
                cache_generation=cache_generation,
                replay_generation=self._cache_replay_generations.get(request_id, 0),
            )
            self._caches[cache_key] = cache_record
        cache = cache_record.cache if cache_record is not None else None
        query_length = int(hidden_states.shape[1])
        if cache_record is not None and token_position != cache_record.sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} cache length mismatch for request={request_id}: "
                f"expected={cache_record.sequence_length} actual_position={token_position}"
            )
        hidden_states = self._forward_layers(
            hidden_states,
            cache=cache,
            token_position=token_position,
            use_cache=use_cache,
            fixed_cache_shape=False,
        )
        self._last_layer_hidden = hidden_states.detach()
        if cache_record is not None:
            cache_record.advance(
                token_position=token_position,
                query_length=query_length,
            )
        if self.norm is not None:
            with nvtx_range(
                torch,
                "final_norm",
                enabled=self.engine_options.nvtx_enabled,
            ):
                hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            with nvtx_range(
                torch,
                "lm_head",
                enabled=self.engine_options.nvtx_enabled,
            ):
                return self.lm_head(hidden_states)
        return hidden_states

    def _position_views(
        self,
        *,
        token_position: int,
        query_length: int,
        batch_size: int,
    ) -> tuple[Any, Any]:
        if (
            self._graph_position is not None
            and query_length == 1
            and batch_size == self._graph_batch_size
        ):
            return self._graph_position_ids, self._graph_position
        end = token_position + query_length
        if end > int(self._position_buffer.shape[0]):
            raise UnsupportedCacheFormatError(
                f"position {end} exceeds configured engine capacity "
                f"{int(self._position_buffer.shape[0])}"
            )
        cache_position = self._position_buffer[token_position:end]
        position_ids = cache_position.unsqueeze(0)
        if batch_size > 1:
            position_ids = position_ids.expand(batch_size, query_length)
        return position_ids, cache_position

    def _forward_layers(
        self,
        hidden_states: Any,
        *,
        cache: Any,
        token_position: int,
        use_cache: bool,
        fixed_cache_shape: bool,
    ) -> Any:
        torch = self.torch
        batch_size = int(hidden_states.shape[0])
        query_length = int(hidden_states.shape[1])
        with nvtx_range(
            torch,
            "position_setup",
            enabled=self.engine_options.nvtx_enabled,
        ):
            position_ids, cache_position = self._position_views(
                token_position=token_position,
                query_length=query_length,
                batch_size=batch_size,
            )
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        with nvtx_range(
            torch,
            "attention_mask_setup",
            enabled=self.engine_options.nvtx_enabled,
        ):
            attention_mask = self._attention_mask_for_backend(
                hidden_states,
                token_position=token_position,
                fixed_cache_shape=fixed_cache_shape,
            )
        compiled = self._compiled_decode if query_length == 1 else self._compiled_prefill
        if compiled is not None:
            compile_started = time.perf_counter()
            static_sequence_length = (
                cache.sequence_length if isinstance(cache, StaticStageKVCache) else None
            )
            try:
                hidden_states = compiled(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_cache,
                    position_embeddings,
                    cache_position,
                    cache,
                )
            except Exception as exc:
                if isinstance(cache, StaticStageKVCache) and static_sequence_length is not None:
                    # A backend failure may occur after some layer cache writes.
                    # The position itself has not been committed yet; clear the
                    # attempted slot before running the eager fallback.
                    cache.rollback(static_sequence_length)
                    cache.prepare_append(
                        token_position=static_sequence_length,
                        query_length=int(hidden_states.shape[1]),
                    )
                self._compile_diagnostics.fallback_used = True
                self._compile_diagnostics.fallback_reason = f"{type(exc).__name__}: {exc}"
                self._compile_diagnostics.graph_break_count += 1
                if query_length == 1:
                    self._compiled_decode = None
                else:
                    self._compiled_prefill = None
                return self._layer_stack(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_cache,
                    position_embeddings,
                    cache_position,
                    cache,
                )
            if query_length == 1:
                first_execution = not self._compile_diagnostics.decode_compiled
                self._compile_diagnostics.decode_compiled = True
            else:
                first_execution = not self._compile_diagnostics.prefill_compiled
                self._compile_diagnostics.prefill_compiled = True
            if first_execution:
                if self.device.type == "cuda":
                    # Compilation is a declared cold-readiness boundary.
                    self.torch.cuda.current_stream(self.device).synchronize()
                self._compile_diagnostics.compile_seconds += time.perf_counter() - compile_started
            self._compile_diagnostics.verified_execution = True
            return hidden_states
        return self._layer_stack(
            hidden_states,
            attention_mask,
            position_ids,
            use_cache,
            position_embeddings,
            cache_position,
            cache,
        )

    def _layer_stack(
        self,
        hidden_states: Any,
        attention_mask: Any,
        position_ids: Any,
        use_cache: bool,
        position_embeddings: Any,
        cache_position: Any,
        cache: Any,
    ) -> Any:
        torch = self.torch
        for layer_call in self._layer_calls:
            with nvtx_range(
                torch,
                "decoder_layer",
                enabled=self.engine_options.nvtx_enabled,
            ):
                hidden_states = layer_call(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_cache,
                    position_embeddings,
                    cache_position,
                    cache,
                )
        return hidden_states

    def _configure_compile(self) -> None:
        if self.engine_options.profile != Qwen3ExecutionProfile.FAST:
            return
        mode = self.engine_options.compile_mode
        if mode in {
            Qwen3CompileMode.EAGER,
            Qwen3CompileMode.MANUAL_CUDA_GRAPH,
        }:
            return
        compile_mode = {
            Qwen3CompileMode.DEFAULT: None,
            Qwen3CompileMode.REDUCE_OVERHEAD: "reduce-overhead",
            Qwen3CompileMode.MAX_AUTOTUNE: "max-autotune",
        }[mode]
        try:
            kwargs: dict[str, Any] = {
                "fullgraph": True,
                "dynamic": False,
            }
            if compile_mode is not None:
                kwargs["mode"] = compile_mode
            self._compiled_prefill = self.torch.compile(self._layer_stack, **kwargs)
            self._compiled_decode = self.torch.compile(self._layer_stack, **kwargs)
        except Exception as exc:
            self._compiled_prefill = None
            self._compiled_decode = None
            self._compile_diagnostics.fallback_used = True
            self._compile_diagnostics.fallback_reason = f"{type(exc).__name__}: {exc}"

    def _attention_mask_for_backend(
        self,
        hidden_states: Any,
        *,
        token_position: int,
        fixed_cache_shape: bool,
    ) -> Any:
        if self.attention_backend == AttentionBackend.EAGER.value:
            return self._causal_mask(
                hidden_states,
                token_position=token_position,
            )
        if fixed_cache_shape:
            if (
                self._graph_attention_mask is not None
                and int(hidden_states.shape[0]) == self._graph_batch_size
                and int(hidden_states.shape[1]) == 1
            ):
                return self._graph_attention_mask
            query_length = int(hidden_states.shape[1])
            key_length = self.engine_options.max_sequence_length
            query_positions = self._position_buffer[token_position : token_position + query_length][
                :, None
            ]
            key_positions = self._position_buffer[:key_length][None, :]
            allowed = key_positions <= query_positions
            return allowed[None, None, :, :].expand(
                int(hidden_states.shape[0]),
                1,
                query_length,
                key_length,
            )
        # SDPA and FlashAttention can infer causal prefill. A one-token decode
        # over an occupied-prefix cache is non-causal over that prefix, which
        # is also correct.
        return None

    def _static_cache_key(
        self,
        metadata: BatchExecutionMetadata,
    ) -> tuple[tuple[str, ...], str, int, int, int]:
        return (
            metadata.request_ids,
            self.model_revision,
            self.stage_id,
            metadata.route_generation,
            metadata.cache_generation,
        )

    def _new_static_cache(
        self,
        metadata: BatchExecutionMetadata,
    ) -> StaticStageKVCache:
        if metadata.batch_size > self.engine_options.max_batch_size:
            raise UnsupportedCacheFormatError(
                f"batch size {metadata.batch_size} exceeds configured maximum "
                f"{self.engine_options.max_batch_size}"
            )
        cache_dtype = resolve_cache_torch_dtype(
            self.torch,
            self.engine_options.cache_dtype,
            model_dtype=self.dtype,
            device=self.device,
        )
        if cache_dtype != self.dtype:
            raise UnsupportedCacheFormatError(
                f"cache dtype {cache_dtype} must match attention dtype {self.dtype} "
                "until a compatible mixed-dtype attention kernel is selected"
            )
        return StaticStageKVCache(
            torch_module=self.torch,
            request_ids=metadata.request_ids,
            model_revision=self.model_revision,
            stage_id=self.stage_id,
            layer_start=self.stage.layer_start,
            layer_end=self.stage.layer_end,
            route_generation=metadata.route_generation,
            cache_generation=metadata.cache_generation,
            max_sequence_length=self.engine_options.max_sequence_length,
            key_value_head_count=int(
                getattr(self.config, "num_key_value_heads", None) or self.config.num_attention_heads
            ),
            head_dimension=int(
                getattr(self.config, "head_dim", None)
                or self.config.hidden_size // self.config.num_attention_heads
            ),
            dtype=cache_dtype,
            device=self.device,
            fixed_shape=self.engine_options.static_cache_fixed_shape,
        )

    def _get_static_cache(
        self,
        metadata: BatchExecutionMetadata,
    ) -> StaticStageKVCache:
        key = self._static_cache_key(metadata)
        cache = self._static_caches.get(key)
        if cache is None:
            if metadata.token_position != 0:
                raise UnsupportedCacheFormatError(
                    f"missing static cache for requests={metadata.request_ids} "
                    f"at token_position={metadata.token_position}"
                )
            cache = self._new_static_cache(metadata)
            self._static_caches[key] = cache
        return cache

    def begin_cuda_graph_decode(
        self,
        metadata: BatchExecutionMetadata,
    ) -> StaticStageKVCache:
        """Install stable position/mask buffers used by an actual CUDA graph."""

        if self.engine_options.compile_mode != Qwen3CompileMode.MANUAL_CUDA_GRAPH:
            raise RuntimeError("CUDA graph setup requires manual_cuda_graph mode")
        if self.engine_options.cache_backend != Qwen3CacheBackend.STATIC:
            raise RuntimeError("CUDA graph decode requires the static cache")
        if metadata.sequence_length != 1:
            raise ValueError("CUDA graph decode captures exactly one token")
        cache = self._get_static_cache(metadata)
        cache.fixed_shape = True
        self._graph_batch_size = metadata.batch_size
        self._graph_position = self.torch.empty((1,), dtype=self.torch.long, device=self.device)
        self._graph_position.fill_(metadata.token_position)
        self._graph_position_ids = self._graph_position.view(1, 1).expand(metadata.batch_size, 1)
        self._graph_attention_mask = self.torch.empty(
            (
                metadata.batch_size,
                1,
                1,
                self.engine_options.max_sequence_length,
            ),
            dtype=self.torch.bool,
            device=self.device,
        )
        self.update_cuda_graph_position(metadata.token_position)
        return cache

    def update_cuda_graph_position(self, token_position: int) -> None:
        if self._graph_position is None or self._graph_attention_mask is None:
            raise RuntimeError("CUDA graph decode buffers are not initialised")
        if not 0 <= token_position < self.engine_options.max_sequence_length:
            raise ValueError("CUDA graph token position exceeds cache capacity")
        self._graph_position.fill_(token_position)
        self._graph_attention_mask.zero_()
        self._graph_attention_mask[..., : token_position + 1].fill_(True)

    def end_cuda_graph_decode(self) -> None:
        self._graph_position = None
        self._graph_position_ids = None
        self._graph_attention_mask = None
        self._graph_batch_size = 0

    def _get_dynamic_batch_cache(
        self,
        metadata: BatchExecutionMetadata,
    ) -> StageLocalKVCache:
        group_id = "\x1f".join(metadata.request_ids)
        key = (
            group_id,
            self.model_revision,
            self.stage_id,
            metadata.route_generation,
            metadata.cache_generation,
        )
        record = self._caches.get(key)
        if record is None:
            if metadata.token_position != 0:
                raise UnsupportedCacheFormatError(
                    f"missing dynamic batch cache for requests={metadata.request_ids}"
                )
            record = StageLocalKVCache(
                cache=self._new_cache(),
                request_id=group_id,
                model_revision=self.model_revision,
                stage_id=self.stage_id,
                layer_start=self.stage.layer_start,
                layer_end=self.stage.layer_end,
                route_generation=metadata.route_generation,
                cache_generation=metadata.cache_generation,
            )
            self._caches[key] = record
        return record

    def _fast_forward_cuda(
        self,
        input_tensor: Any,
        metadata: BatchExecutionMetadata,
        *,
        batched_call: bool,
    ) -> Any:
        torch = self.torch
        # The CUDA-native tensor boundary is used by both profiles.  The fast
        # profile selects the static cache, compact final-worker sampling and
        # optional compilation/graphs; the correctness profile deliberately
        # retains eager attention, deterministic CUDA settings and DynamicCache.
        # Allowing the oracle through this boundary is important for a fair
        # batch-shape comparison without changing the legacy NumPy execute()
        # compatibility interface.
        if input_tensor.device != self.device:
            raise ValueError(
                f"CUDA-native stage input must already be on {self.device}; "
                f"received {input_tensor.device}"
            )
        if int(input_tensor.shape[0]) != metadata.batch_size:
            raise ValueError(
                f"input batch {int(input_tensor.shape[0])} does not match "
                f"metadata batch {metadata.batch_size}"
            )
        if int(input_tensor.shape[1]) != metadata.sequence_length:
            raise ValueError(
                f"input sequence {int(input_tensor.shape[1])} does not match "
                f"metadata sequence {metadata.sequence_length}"
            )
        with nvtx_range(
            torch,
            "embedding",
            enabled=self.engine_options.nvtx_enabled,
        ):
            if self.embed_tokens is not None:
                if input_tensor.dtype not in {torch.int32, torch.int64}:
                    raise ValueError("embedding-owning fast stage requires integer token IDs")
                hidden_states = self.embed_tokens(input_tensor)
            else:
                if input_tensor.dtype != self.dtype:
                    raise ValueError(
                        f"fast hidden-state dtype must be {self.dtype}; "
                        f"received {input_tensor.dtype}"
                    )
                hidden_states = input_tensor
        static_cache: StaticStageKVCache | None = None
        dynamic_record: StageLocalKVCache | None = None
        use_dynamic_memory_fallback = metadata.request_ids in self._dynamic_cache_fallback_groups
        if (
            self.engine_options.cache_backend == Qwen3CacheBackend.STATIC
            and not use_dynamic_memory_fallback
        ):
            static_cache = self._get_static_cache(metadata)
            with nvtx_range(
                torch,
                "cache_update",
                enabled=self.engine_options.nvtx_enabled,
            ):
                static_cache.prepare_append(
                    token_position=metadata.token_position,
                    query_length=metadata.sequence_length,
                )
            cache: Any = static_cache
            fixed_cache_shape = static_cache.fixed_shape
        else:
            dynamic_record = self._get_dynamic_batch_cache(metadata)
            if dynamic_record.sequence_length != metadata.token_position:
                raise UnsupportedCacheFormatError(
                    f"dynamic batch cache expected position "
                    f"{dynamic_record.sequence_length}, got {metadata.token_position}"
                )
            cache = dynamic_record.cache
            fixed_cache_shape = False
        hidden_states = self._forward_layers(
            hidden_states,
            cache=cache,
            token_position=metadata.token_position,
            use_cache=True,
            fixed_cache_shape=fixed_cache_shape,
        )
        if static_cache is not None:
            with nvtx_range(
                torch,
                "cache_update",
                enabled=self.engine_options.nvtx_enabled,
            ):
                static_cache.commit_append()
        elif dynamic_record is not None:
            dynamic_record.advance(
                token_position=metadata.token_position,
                query_length=metadata.sequence_length,
            )
        if self.engine_options.boundary_diagnostics or any(
            item.diagnostic for item in metadata.requests
        ):
            self._last_layer_hidden = hidden_states.detach()
            if metadata.token_position == 0:
                for request_id in metadata.request_ids:
                    self._record_boundary(request_id, self._last_layer_hidden)
        if self.norm is not None:
            with nvtx_range(
                torch,
                "final_norm",
                enabled=self.engine_options.nvtx_enabled,
            ):
                hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            # Only the newest position can select the next token. This avoids
            # projecting every prefill position over the full vocabulary.
            hidden_states = hidden_states[:, -1:, :]
            with nvtx_range(
                torch,
                "lm_head",
                enabled=self.engine_options.nvtx_enabled,
            ):
                hidden_states = self.lm_head(hidden_states)
        self._fast_forward_count += 1
        if batched_call:
            self._fast_batch_forward_count += 1
        return hidden_states

    def prefill_cuda(
        self,
        input_tensor: Any,
        metadata: StageExecutionMetadata,
    ) -> Any:
        batch = BatchExecutionMetadata(requests=(metadata,))
        with self.torch.inference_mode():
            return self._fast_forward_cuda(input_tensor, batch, batched_call=False)

    def decode_cuda(
        self,
        input_tensor: Any,
        metadata: StageExecutionMetadata,
    ) -> Any:
        batch = BatchExecutionMetadata(requests=(metadata,))
        with self.torch.inference_mode():
            return self._fast_forward_cuda(input_tensor, batch, batched_call=False)

    def prefill_batch_cuda(
        self,
        input_tensors: Any,
        metadata: BatchExecutionMetadata,
    ) -> Any:
        with self.torch.inference_mode():
            return self._fast_forward_cuda(input_tensors, metadata, batched_call=True)

    def decode_batch_cuda(
        self,
        input_tensors: Any,
        metadata: BatchExecutionMetadata,
    ) -> Any:
        with self.torch.inference_mode():
            return self._fast_forward_cuda(input_tensors, metadata, batched_call=True)

    def sample_cuda(
        self,
        logits: Any,
        *,
        request_ids: tuple[str, ...],
        parameters: SamplingParameters | None = None,
        token_history: Any | None = None,
    ) -> SamplingResult:
        if self.lm_head is None:
            raise RuntimeError("only the final Qwen3 stage can sample")
        if self.engine_options.profile != Qwen3ExecutionProfile.FAST:
            raise RuntimeError("final-worker sampling requires profile qwen3_fast")
        selected_parameters = parameters or SamplingParameters(
            return_full_logits=self.engine_options.diagnostic_full_logits
        )
        if (
            selected_parameters.return_full_logits
            and not self.engine_options.diagnostic_full_logits
        ):
            raise RuntimeError(
                "full-logit return requires diagnostic_full_logits=true in qwen3_fast"
            )
        with nvtx_range(
            self.torch,
            "sampling",
            enabled=self.engine_options.nvtx_enabled,
        ):
            result = sample_final_logits(
                self.torch,
                logits,
                parameters=selected_parameters,
                request_ids=request_ids,
                state=self._sampling_state,
                token_history=token_history,
            )
        self._sampled_token_return_count += int(result.token_ids.numel())
        if result.full_logits is not None:
            self._full_logit_return_count += 1
        return result

    def execute_batch(
        self,
        activations: Any,
        *,
        metadata: BatchExecutionMetadata,
        operation: Any,
    ) -> Any:
        """Run one real batch after an explicit remote/same-host CPU boundary."""

        import numpy as np

        if self.engine_options.profile != Qwen3ExecutionProfile.FAST:
            raise RuntimeError("execute_batch is available only in qwen3_fast")
        contiguous = np.ascontiguousarray(activations)
        if self.embed_tokens is not None:
            tensor = self.torch.from_numpy(contiguous.astype(np.int64, copy=False)).to(self.device)
        elif contiguous.dtype == np.uint16 and self.dtype == self.torch.bfloat16:
            tensor = self.torch.from_numpy(contiguous).view(self.torch.bfloat16).to(self.device)
        else:
            tensor = self.torch.from_numpy(contiguous).to(
                device=self.device,
                dtype=self.dtype,
            )
        operation_name = str(getattr(operation, "value", operation))
        if operation_name == "prefill":
            output = self.prefill_batch_cuda(tensor, metadata)
        elif operation_name == "decode":
            output = self.decode_batch_cuda(tensor, metadata)
        else:
            raise ValueError(f"unsupported batched operation {operation_name!r}")
        if self.lm_head is not None and not self.engine_options.diagnostic_full_logits:
            raise RuntimeError(
                "qwen3_fast final stages return compact sampling results; "
                "enable diagnostic_full_logits only for explicit debugging"
            )
        host = output.detach().cpu()
        if host.dtype == self.torch.bfloat16:
            if self.lm_head is None:
                return host.contiguous().view(self.torch.uint16).numpy()
            self._full_logit_return_count += 1
            return host.float().numpy()
        return host.numpy()

    def execute(
        self,
        activation: Any,
        *,
        request_id: str,
        operation: Any,
        token_position: int,
        sequence_length: int,
        cache_generation: int,
        route_generation: int = 0,
    ) -> Any:
        import numpy as np

        if int(activation.shape[1]) != sequence_length:
            raise ValueError(
                f"activation sequence length {activation.shape[1]} does not match "
                f"metadata {sequence_length}"
            )
        operation_name = str(getattr(operation, "value", operation))
        host_to_device_started = time.perf_counter()
        if self.embed_tokens is not None:
            tensor = self.torch.from_numpy(
                np.ascontiguousarray(activation).astype(np.int64, copy=False)
            ).to(self.device)
        else:
            contiguous = np.ascontiguousarray(activation)
            if contiguous.dtype == np.uint16 and self.dtype == self.torch.bfloat16:
                tensor = self.torch.from_numpy(contiguous).view(self.torch.bfloat16).to(self.device)
            else:
                tensor = self.torch.from_numpy(contiguous).to(
                    device=self.device,
                    dtype=self.dtype,
                )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        host_to_device_ms = (
            (time.perf_counter() - host_to_device_started) * 1000
            if self.device.type == "cuda"
            else 0.0
        )
        execution_started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.forward(
                tensor,
                request_id=request_id,
                token_position=token_position,
                cache_generation=cache_generation,
                route_generation=route_generation,
                use_cache=True,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        execution_ms = (
            (time.perf_counter() - execution_started) * 1000 if self.device.type == "cuda" else 0.0
        )
        boundary_copy_ms = 0.0
        if operation_name == "prefill" and self.engine_options.boundary_diagnostics:
            boundary_copy_started = time.perf_counter()
            self._record_boundary(request_id, self._last_layer_hidden)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
                boundary_copy_ms = (time.perf_counter() - boundary_copy_started) * 1000
        if self.lm_head is not None:
            output = output[:, -1:, :]
        device_to_host_bytes = int(output.numel() * output.element_size())
        device_to_host_started = time.perf_counter()
        output = output.detach().cpu()
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        device_to_host_ms = (
            (time.perf_counter() - device_to_host_started) * 1000
            if self.device.type == "cuda"
            else 0.0
        )
        self._finite_output_checks += 1
        if not bool(self.torch.isfinite(output).all().item()):
            raise IntegrityError(
                f"stage {self.stage_id} produced NaN or infinity for request {request_id}"
            )
        if output.dtype == self.torch.bfloat16:
            if self.lm_head is None:
                result = output.contiguous().view(self.torch.uint16).numpy()
            else:
                result = output.float().numpy()
        else:
            result = output.numpy()
        transfer: dict[str, Any] = {
            "execution_profile": self.execution_profile,
            "attention_backend": self.attention_backend,
            "request_id": request_id,
            "stage_id": self.stage_id,
            "operation": operation_name,
            "token_position": token_position,
            "sequence_length": sequence_length,
            "host_to_device_copy_ms": host_to_device_ms,
            "device_to_host_copy_ms": device_to_host_ms,
            "boundary_diagnostic_copy_ms": boundary_copy_ms,
            "cuda_execution_ms": execution_ms,
            "host_to_device_bytes": int(np.ascontiguousarray(activation).nbytes),
            "device_to_host_bytes": device_to_host_bytes,
            "transport_tensor_bytes": int(result.nbytes),
        }
        self._transfer_history.append(transfer)
        self._transfer_metrics["operation_count"] = (
            int(self._transfer_metrics["operation_count"]) + 1
        )
        for key in (
            "host_to_device_copy_ms",
            "device_to_host_copy_ms",
            "boundary_diagnostic_copy_ms",
            "cuda_execution_ms",
        ):
            self._transfer_metrics[key] = float(self._transfer_metrics[key]) + float(transfer[key])
        for key in (
            "host_to_device_bytes",
            "device_to_host_bytes",
            "transport_tensor_bytes",
        ):
            self._transfer_metrics[key] = int(self._transfer_metrics[key]) + int(transfer[key])
        return result

    def _record_boundary(self, request_id: str, value: Any | None) -> None:
        reference_root = os.environ.get("SWARM_REFERENCE_BOUNDARY_ROOT")
        if not reference_root or value is None:
            return
        import numpy as np

        safe_request_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)
        reference_file = (
            Path(reference_root)
            / safe_request_id
            / f"reference-layer-{self.stage.layer_end:04d}.npy"
        )
        record: dict[str, Any] = {
            "request_id": request_id,
            "stage_id": self.stage_id,
            "layer_end": self.stage.layer_end,
            "reference_file": str(reference_file),
        }
        if not reference_file.is_file():
            record.update(
                {
                    "within_tolerance": False,
                    "error": "reference boundary file is missing",
                }
            )
            self._boundary_records[request_id] = record
            return
        reference = np.load(reference_file, allow_pickle=False)
        actual = value.detach().float().cpu().numpy()
        if actual.shape != reference.shape:
            record.update(
                {
                    "within_tolerance": False,
                    "shape_identity": False,
                    "distributed_shape": list(actual.shape),
                    "reference_shape": list(reference.shape),
                }
            )
            self._boundary_records[request_id] = record
            return
        actual64 = actual.astype(np.float64)
        reference64 = reference.astype(np.float64)
        difference = actual64 - reference64
        absolute = np.abs(difference)
        denominator = np.maximum(np.abs(reference64), 1e-12)
        actual_flat = actual64.reshape(-1)
        reference_flat = reference64.reshape(-1)
        norm_product = float(np.linalg.norm(actual_flat) * np.linalg.norm(reference_flat))
        cosine = (
            float(np.dot(actual_flat, reference_flat) / norm_product)
            if norm_product > 0
            else float(actual_flat.size == 0 or np.array_equal(actual_flat, reference_flat))
        )
        atol = float(os.environ.get("SWARM_BOUNDARY_ATOL", "0.02"))
        rtol = float(os.environ.get("SWARM_BOUNDARY_RTOL", "0.02"))
        minimum_cosine = float(os.environ.get("SWARM_BOUNDARY_MINIMUM_COSINE", "0.999"))
        nan_count = int(np.isnan(actual64).sum())
        inf_count = int(np.isinf(actual64).sum())
        allclose = bool(np.allclose(actual64, reference64, atol=atol, rtol=rtol))
        record.update(
            {
                "within_tolerance": (
                    allclose and cosine >= minimum_cosine and nan_count == 0 and inf_count == 0
                ),
                "shape_identity": True,
                "dtype_identity": str(actual.dtype) == str(reference.dtype),
                "distributed_dtype": str(actual.dtype),
                "reference_dtype": str(reference.dtype),
                "maximum_absolute_error": float(absolute.max(initial=0.0)),
                "mean_absolute_error": float(absolute.mean()) if absolute.size else 0.0,
                "maximum_relative_error": float((absolute / denominator).max(initial=0.0)),
                "cosine_similarity": cosine,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "atol": atol,
                "rtol": rtol,
                "minimum_cosine_similarity": minimum_cosine,
            }
        )
        self._boundary_records[request_id] = record

    def cancel(self, request_id: str) -> None:
        for dynamic_key in [
            candidate for candidate in self._caches if request_id in candidate[0].split("\x1f")
        ]:
            record = self._caches.pop(dynamic_key)
            summary = record.summary()
            summary.update({"event": "deleted", "stale_after_operation": False})
            self._cache_history.append(summary)
        for static_key in [
            candidate
            for candidate, cache in self._static_caches.items()
            if request_id in cache.request_ids
        ]:
            cache = self._static_caches.pop(static_key)
            summary = cache.summary()
            released = cache.delete()
            summary.update(
                {
                    "event": "deleted",
                    "released_bytes": released,
                    "stale_after_operation": False,
                }
            )
            self._cache_history.append(summary)
        self._sampling_state.delete(request_id)

    def cancel_batch(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self.cancel(request_id)
        self._dynamic_cache_fallback_groups.discard(request_ids)

    def use_dynamic_cache_memory_fallback(
        self,
        request_ids: tuple[str, ...],
    ) -> None:
        """Use contiguous current-length KV storage for one rejected graph batch."""

        if not request_ids:
            raise ValueError("dynamic cache fallback requires request IDs")
        self._dynamic_cache_fallback_groups.add(request_ids)

    def reset_cache(self, request_id: str, *, for_replay: bool = False) -> int:
        removed = sum(
            record.memory_bytes()
            for key, record in self._caches.items()
            if request_id in key[0].split("\x1f")
        )
        removed += sum(
            cache.reserved_bytes
            for cache in self._static_caches.values()
            if request_id in cache.request_ids
        )
        for dynamic_key in [
            candidate for candidate in self._caches if request_id in candidate[0].split("\x1f")
        ]:
            record = self._caches.pop(dynamic_key)
            summary = record.summary()
            summary.update(
                {
                    "event": "reset-for-replay",
                    "stale_after_operation": False,
                }
            )
            self._cache_history.append(summary)
        for static_key in [
            candidate
            for candidate, cache in self._static_caches.items()
            if request_id in cache.request_ids
        ]:
            cache = self._static_caches.pop(static_key)
            summary = cache.summary()
            cache.delete()
            summary.update(
                {
                    "event": "reset-for-replay",
                    "stale_after_operation": False,
                }
            )
            self._cache_history.append(summary)
        if for_replay:
            self._cache_replay_generations[request_id] = (
                self._cache_replay_generations.get(request_id, 0) + 1
            )
        return removed

    def inspect_cache(self, request_id: str | None = None) -> list[dict[str, Any]]:
        dynamic = [
            record.summary()
            for key, record in sorted(self._caches.items())
            if request_id is None or request_id in key[0].split("\x1f")
        ]
        static = [
            cache.summary()
            for _, cache in sorted(self._static_caches.items())
            if request_id is None or request_id in cache.request_ids
        ]
        return [*dynamic, *static]

    def cache_bytes(self) -> int:
        return sum(record.memory_bytes() for record in self._caches.values()) + sum(
            cache.reserved_bytes for cache in self._static_caches.values()
        )

    def state_summary(self) -> dict[str, Any]:
        return {
            "execution_profile": self.execution_profile,
            "attention_backend": self.attention_backend,
            "attention_backend_evidence": self.attention_evidence.payload(),
            "cache_backend": self.engine_options.cache_backend.value,
            "dynamic_cache_memory_fallback_group_count": len(self._dynamic_cache_fallback_groups),
            "cache_dtype": self.engine_options.cache_dtype.value,
            "compile_mode": self.engine_options.compile_mode.value,
            "compile_diagnostics": self._compile_diagnostics.payload(),
            "matmul_accumulation": "fp32",
            "allow_bf16_reduced_precision_reduction": bool(
                self.torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "allow_fp16_reduced_precision_reduction": bool(
                self.torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "stage_id": self.stage_id,
            "layer_start": self.stage.layer_start,
            "layer_end": self.stage.layer_end,
            "owns_embeddings": self.stage.owns_embeddings,
            "owns_final_norm": self.stage.owns_final_norm,
            "owns_output_head": self.stage.owns_output_head,
            "loaded_source_tensors": self.loaded_source_tensors,
            "cache_count": len(self._caches) + len(self._static_caches),
            "cache_bytes": self.cache_bytes(),
            "caches": self.inspect_cache(),
            "cache_history": self._cache_history[-100:],
            "causal_mask_cache_entry_count": len(self._causal_mask_cache),
            "causal_mask_cache_bytes": sum(
                int(mask.numel() * mask.element_size()) for mask in self._causal_mask_cache.values()
            ),
            "boundary_records": list(self._boundary_records.values()),
            "transfer_metrics": dict(self._transfer_metrics),
            "transfer_history": self._transfer_history[-1000:],
            "finite_output_checks": self._finite_output_checks,
            "all_checked_outputs_finite": True,
            "fast_forward_count": self._fast_forward_count,
            "fast_batch_forward_count": self._fast_batch_forward_count,
            "sampled_token_return_count": self._sampled_token_return_count,
            "full_logit_return_count": self._full_logit_return_count,
            "attention_tuning_seconds": self._attention_tuning_seconds,
        }

    def _target_parameter_name(
        self,
        source_name: str,
        *,
        manifest: ModelManifest,
    ) -> str | None:
        if source_name.startswith("model.embed_tokens."):
            suffix = source_name.removeprefix("model.embed_tokens.")
            if self.embed_tokens is not None:
                return f"embed_tokens.{suffix}"
            if (
                self.lm_head is not None
                and source_name in manifest.shared_tensors
                and self.stage_id in manifest.shared_tensors[source_name]
            ):
                return f"lm_head.{suffix}"
        match = _LAYER_PATTERN.match(source_name)
        if match:
            layer_index = int(match.group(1))
            if self.stage.layer_start <= layer_index < self.stage.layer_end:
                return source_name.removeprefix("model.")
        if source_name.startswith("model.norm.") and self.norm is not None:
            return source_name.removeprefix("model.")
        if source_name.startswith("lm_head.") and self.lm_head is not None:
            return source_name
        return None

    def _target_parameter_names(
        self,
        source_name: str,
        *,
        manifest: ModelManifest,
    ) -> list[str]:
        primary = self._target_parameter_name(source_name, manifest=manifest)
        targets = [primary] if primary is not None else []
        if (
            source_name.startswith("model.embed_tokens.")
            and self.embed_tokens is not None
            and self.lm_head is not None
            and self.stage.owns_embeddings
            and self.stage.owns_output_head
        ):
            suffix = source_name.removeprefix("model.embed_tokens.")
            lm_target = f"lm_head.{suffix}"
            if lm_target not in targets:
                targets.append(lm_target)
        return targets

    def _parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        if self.embed_tokens is not None:
            parameters.update(
                {
                    f"embed_tokens.{name}": value
                    for name, value in self.embed_tokens.named_parameters()
                }
            )
        parameters.update(
            {f"layers.{name}": value for name, value in self.layers.named_parameters()}
        )
        if self.norm is not None:
            parameters.update(
                {f"norm.{name}": value for name, value in self.norm.named_parameters()}
            )
        if self.lm_head is not None:
            parameters.update(
                {f"lm_head.{name}": value for name, value in self.lm_head.named_parameters()}
            )
        return parameters

    def load_weights(
        self,
        shard_path: Path,
        *,
        manifest: ModelManifest,
    ) -> list[str]:
        from safetensors import safe_open

        root = shard_path.expanduser().resolve()
        files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
        if not files:
            raise IntegrityError(f"no stage safetensors found under {root}")
        if self.stage_id >= len(manifest.stages):
            raise IntegrityError(
                f"stage {self.stage_id} is outside manifest stage count {len(manifest.stages)}"
            )
        declared_stage = manifest.stages[self.stage_id]
        if (
            declared_stage.layer_start != self.stage.layer_start
            or declared_stage.layer_end != self.stage.layer_end
        ):
            raise IntegrityError(f"stage {self.stage_id} layer range does not match the manifest")
        parameters = self._parameters()
        loaded_parameters: set[str] = set()
        loaded_sources: list[str] = []
        expected_source_dtype = {
            "F16": self.torch.float16,
            "BF16": self.torch.bfloat16,
            "F32": self.torch.float32,
        }.get(manifest.weight_dtype.upper())
        if expected_source_dtype is None:
            raise IntegrityError(f"unsupported manifest source dtype {manifest.weight_dtype}")
        recorder = lifecycle_recorder()
        read_started = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "shard_read_started",
                monotonic_ns=read_started,
                bytes_count=declared_stage.required_memory_bytes,
                details={"file_count": len(files)},
            )
        cpu_values: dict[str, Any] = {}
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                for source_name in handle.keys():  # noqa: SIM118
                    if source_name not in declared_stage.tensor_names:
                        raise IntegrityError(
                            f"stage {self.stage_id} shard contains tensor not declared "
                            f"by stage manifest: {source_name}"
                        )
                    if source_name in cpu_values:
                        raise IntegrityError(
                            f"stage {self.stage_id} shard repeats tensor {source_name}"
                        )
                    cpu_values[source_name] = handle.get_tensor(source_name)
        read_completed = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "shard_read_completed",
                monotonic_ns=read_completed,
                duration_ns=read_completed - read_started,
                bytes_count=declared_stage.required_memory_bytes,
                details={"file_count": len(files), "tensor_count": len(cpu_values)},
            )
        transfer_started = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "host_to_device_transfer_started",
                monotonic_ns=transfer_started,
                bytes_count=declared_stage.required_memory_bytes,
                details={"weight_materialisation_mode": "host-to-device-copy"},
            )
        with self.torch.no_grad():
            for source_name, value in cpu_values.items():
                target_names = self._target_parameter_names(
                    source_name,
                    manifest=manifest,
                )
                if not target_names:
                    raise IntegrityError(
                        f"stage {self.stage_id} shard contains unowned tensor {source_name}"
                    )
                if value.dtype != expected_source_dtype:
                    raise IntegrityError(
                        f"dtype mismatch for {source_name}: "
                        f"shard={value.dtype} manifest={expected_source_dtype}"
                    )
                for target_name in target_names:
                    try:
                        parameter = parameters[target_name]
                    except KeyError as exc:
                        raise IntegrityError(
                            f"no stage parameter {target_name} for source {source_name}"
                        ) from exc
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise IntegrityError(
                            f"shape mismatch for {source_name}: shard={tuple(value.shape)} "
                            f"module={tuple(parameter.shape)}"
                        )
                    parameter.copy_(value.to(device=self.device, dtype=self.dtype))
                    loaded_parameters.add(target_name)
                loaded_sources.append(source_name)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        transfer_completed = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "host_to_device_transfer_completed",
                monotonic_ns=transfer_completed,
                duration_ns=transfer_completed - transfer_started,
                bytes_count=declared_stage.required_memory_bytes,
                details={"weight_materialisation_mode": "host-to-device-copy"},
            )
        cpu_values.clear()
        missing = sorted(set(parameters) - loaded_parameters)
        if missing:
            raise IntegrityError(
                f"stage {self.stage_id} is missing parameters after shard load: {missing}"
            )
        if set(loaded_sources) != set(declared_stage.tensor_names):
            raise IntegrityError(
                f"stage {self.stage_id} loaded source tensor set does not match its stage manifest"
            )
        self.model_revision = manifest.model_revision
        self.loaded_source_tensors = sorted(loaded_sources)
        self._autotune_attention_backend()
        self._configure_compile()
        return self.loaded_source_tensors
