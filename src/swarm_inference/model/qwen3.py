"""Dense Qwen3 adapter and partial stage module.

The worker module constructs only the decoder layers and endpoint components
owned by its stage. It never calls ``Qwen3ForCausalLM.from_pretrained``.
"""

from __future__ import annotations

import inspect
import json
import math
import re
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
        self.rotary_emb = Qwen3RotaryEmbedding(config=config, device=self.device).to(self.device)
        self._caches: dict[tuple[str, int], Any] = {}
        self.loaded_source_tensors: list[str] = []
        for layer in self.layers.values():
            layer.eval()

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
        use_cache: bool = True,
    ) -> Any:
        torch = self.torch
        if self.embed_tokens is not None:
            if inputs.dtype not in {torch.int32, torch.int64}:
                raise ValueError("embedding-owning stage requires integer token IDs")
            hidden_states = self.embed_tokens(inputs.to(self.device))
        else:
            hidden_states = inputs.to(device=self.device, dtype=self.dtype)
        cache_key = (request_id, cache_generation)
        cache = self._caches.get(cache_key)
        if use_cache and cache is None:
            if token_position != 0:
                raise UnsupportedCacheFormatError(
                    f"missing cache for request={request_id} at token_position={token_position}"
                )
            cache = self._new_cache()
            self._caches[cache_key] = cache
        query_length = hidden_states.shape[1]
        position_ids = torch.arange(
            token_position,
            token_position + query_length,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
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
    ) -> Any:
        import numpy as np

        if int(activation.shape[1]) != sequence_length:
            raise ValueError(
                f"activation sequence length {activation.shape[1]} does not match "
                f"metadata {sequence_length}"
            )
        if self.embed_tokens is not None:
            tensor = self.torch.from_numpy(
                np.ascontiguousarray(activation).astype(np.int64, copy=False)
            ).to(self.device)
        else:
            tensor = self.torch.from_numpy(np.ascontiguousarray(activation)).to(
                device=self.device,
                dtype=self.dtype,
            )
        with self.torch.inference_mode():
            output = self.forward(
                tensor,
                request_id=request_id,
                token_position=token_position,
                cache_generation=cache_generation,
                use_cache=True,
            )
        output = output.detach().cpu()
        if output.dtype == self.torch.bfloat16:
            output = output.float()
        return output.numpy()

    def cancel(self, request_id: str) -> None:
        for key in [key for key in self._caches if key[0] == request_id]:
            self._caches.pop(key, None)

    def cache_bytes(self) -> int:
        total = 0
        for cache in self._caches.values():
            for layer in getattr(cache, "layers", []):
                for name in ("keys", "values", "key_cache", "value_cache"):
                    tensor = getattr(layer, name, None)
                    if tensor is not None and hasattr(tensor, "numel"):
                        total += tensor.numel() * tensor.element_size()
            for name in ("key_cache", "value_cache"):
                tensors = getattr(cache, name, [])
                total += sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in tensors
                    if tensor is not None
                )
        return total

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
        parameters = self._parameters()
        loaded_parameters: set[str] = set()
        loaded_sources: list[str] = []
        with self.torch.no_grad():
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    for source_name in handle.keys():  # noqa: SIM118
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
        self.loaded_source_tensors = sorted(loaded_sources)
        return self.loaded_source_tensors
