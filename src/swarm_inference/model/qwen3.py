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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.config.models import ModelManifest, StageDefinition
from swarm_inference.exceptions import (
    IntegrityError,
    UnsupportedArchitectureError,
    UnsupportedCacheFormatError,
)
from swarm_inference.model.adapter import (
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
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
        # Stage execution and the separate full-reference phase deliberately use the
        # same explicit attention implementation.  A config loaded directly from
        # config.json otherwise leaves this private Transformers selector as None.
        config._attn_implementation = "eager"
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
        if self.device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
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
        # Match the official full-model construction order exactly: Transformers
        # creates RoPE frequencies on CPU and subsequently moves the model.  Direct
        # CUDA construction changes two float32 inv_freq values for Qwen3-0.6B by
        # one ULP, which phase-amplifies on 512-token prompts and breaks boundary
        # validation even though short prompts and greedy tokens still agree.
        self.rotary_emb = Qwen3RotaryEmbedding(config=config).to(self.device)
        self.model_revision = "unloaded"
        self._caches: dict[tuple[str, str, int, int, int], StageLocalKVCache] = {}
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
        self.loaded_source_tensors: list[str] = []
        for layer in self.layers.values():
            layer.eval()
            layer.requires_grad_(False)
        for endpoint in (self.embed_tokens, self.norm, self.lm_head, self.rotary_emb):
            if endpoint is not None:
                endpoint.eval()
                endpoint.requires_grad_(False)

    def _new_cache(self) -> Any:
        from transformers import DynamicCache

        try:
            return DynamicCache(config=self.config)
        except TypeError:
            return DynamicCache()

    def _causal_mask(
        self,
        hidden_states: Any,
        *,
        token_position: int,
    ) -> Any:
        torch = self.torch
        batch, query_length, _ = hidden_states.shape
        key_length = token_position + query_length
        query_positions = torch.arange(
            token_position,
            token_position + query_length,
            device=hidden_states.device,
        )[:, None]
        key_positions = torch.arange(key_length, device=hidden_states.device)[None, :]
        allowed = key_positions <= query_positions
        minimum = torch.finfo(hidden_states.dtype).min
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.full(
                (),
                minimum,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ),
        )
        return mask[None, None, :, :].expand(batch, 1, query_length, key_length)

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
        query_length = hidden_states.shape[1]
        if cache_record is not None and token_position != cache_record.sequence_length:
            raise UnsupportedCacheFormatError(
                f"stage {self.stage_id} cache length mismatch for request={request_id}: "
                f"expected={cache_record.sequence_length} actual_position={token_position}"
            )
        position_ids = torch.arange(
            token_position,
            token_position + query_length,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        cache_position = position_ids[0]
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask = self._causal_mask(
            hidden_states,
            token_position=token_position,
        )
        for layer_index in range(self.stage.layer_start, self.stage.layer_end):
            layer = self.layers[str(layer_index)]
            signature = inspect.signature(layer.forward)
            kwargs: dict[str, Any] = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": use_cache,
                "position_embeddings": position_embeddings,
                "cache_position": cache_position,
            }
            if "past_key_values" in signature.parameters:
                kwargs["past_key_values"] = cache
            elif "past_key_value" in signature.parameters:
                kwargs["past_key_value"] = cache
            else:
                raise UnsupportedCacheFormatError(
                    "installed Transformers Qwen3 decoder has no supported cache argument"
                )
            output = layer(hidden_states, **kwargs)
            hidden_states = output[0] if isinstance(output, tuple) else output
        self._last_layer_hidden = hidden_states.detach()
        if cache_record is not None:
            cache_record.advance(
                token_position=token_position,
                query_length=int(query_length),
            )
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states

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
        if operation_name == "prefill":
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
        if output.dtype == self.torch.bfloat16:
            if self.lm_head is None:
                result = output.contiguous().view(self.torch.uint16).numpy()
            else:
                result = output.float().numpy()
        else:
            result = output.numpy()
        transfer: dict[str, Any] = {
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
        for key in [key for key in self._caches if key[0] == request_id]:
            record = self._caches.pop(key)
            summary = record.summary()
            summary.update({"event": "deleted", "stale_after_operation": False})
            self._cache_history.append(summary)

    def reset_cache(self, request_id: str, *, for_replay: bool = False) -> int:
        removed = sum(
            record.memory_bytes() for key, record in self._caches.items() if key[0] == request_id
        )
        for key in [key for key in self._caches if key[0] == request_id]:
            record = self._caches.pop(key)
            summary = record.summary()
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
        return [
            record.summary()
            for key, record in sorted(self._caches.items())
            if request_id is None or key[0] == request_id
        ]

    def cache_bytes(self) -> int:
        return sum(record.memory_bytes() for record in self._caches.values())

    def state_summary(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "layer_start": self.stage.layer_start,
            "layer_end": self.stage.layer_end,
            "owns_embeddings": self.stage.owns_embeddings,
            "owns_final_norm": self.stage.owns_final_norm,
            "owns_output_head": self.stage.owns_output_head,
            "loaded_source_tensors": self.loaded_source_tensors,
            "cache_count": len(self._caches),
            "cache_bytes": self.cache_bytes(),
            "caches": self.inspect_cache(),
            "cache_history": self._cache_history[-100:],
            "boundary_records": list(self._boundary_records.values()),
            "transfer_metrics": dict(self._transfer_metrics),
            "transfer_history": self._transfer_history[-1000:],
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
        with self.torch.no_grad():
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    for source_name in handle.keys():  # noqa: SIM118
                        if source_name not in declared_stage.tensor_names:
                            raise IntegrityError(
                                f"stage {self.stage_id} shard contains tensor not declared "
                                f"by stage manifest: {source_name}"
                            )
                        target_name = self._target_parameter_name(
                            source_name,
                            manifest=manifest,
                        )
                        if target_name is None:
                            raise IntegrityError(
                                f"stage {self.stage_id} shard contains unowned tensor {source_name}"
                            )
                        try:
                            parameter = parameters[target_name]
                        except KeyError as exc:
                            raise IntegrityError(
                                f"no stage parameter {target_name} for source {source_name}"
                            ) from exc
                        value = handle.get_tensor(source_name)
                        if value.dtype != expected_source_dtype:
                            raise IntegrityError(
                                f"dtype mismatch for {source_name}: "
                                f"shard={value.dtype} manifest={expected_source_dtype}"
                            )
                        if tuple(value.shape) != tuple(parameter.shape):
                            raise IntegrityError(
                                f"shape mismatch for {source_name}: shard={tuple(value.shape)} "
                                f"module={tuple(parameter.shape)}"
                            )
                        parameter.copy_(value.to(device=self.device, dtype=self.dtype))
                        loaded_parameters.add(target_name)
                        loaded_sources.append(source_name)
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
        return self.loaded_source_tensors
