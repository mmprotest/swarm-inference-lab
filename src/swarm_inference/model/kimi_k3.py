"""Native Kimi K3 CUDA adapter for canonical persistent stage workers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.exceptions import IntegrityError, UnsupportedArchitectureError
from swarm_inference.model.adapter import (
    AdapterSupportReport,
    AdapterSupportStatus,
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import StageAssignment

_LAYER_PATTERN = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
_FINAL_ATTENTION_NAMES = frozenset(
    {
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
    }
)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _SafetensorCatalog:
    """Read only the Safetensors headers needed for ownership validation."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        index_path = self.root / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("Kimi checkpoint Safetensors index is invalid") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise IntegrityError("Kimi checkpoint Safetensors index has no weight map")
        self.weight_map = {str(name): str(source) for name, source in weight_map.items()}
        self._headers: dict[str, dict[str, Any]] = {}

    def _header(self, source: str) -> dict[str, Any]:
        cached = self._headers.get(source)
        if cached is not None:
            return cached
        path = self.root / source
        try:
            with path.open("rb") as stream:
                header_length_bytes = stream.read(8)
                if len(header_length_bytes) != 8:
                    raise IntegrityError(f"Safetensors header is truncated: {source}")
                header_length = int.from_bytes(header_length_bytes, "little")
                if not 2 <= header_length <= 1024**3:
                    raise IntegrityError(f"Safetensors header length is invalid: {source}")
                payload = stream.read(header_length)
            header = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"Safetensors header is invalid: {source}") from exc
        if not isinstance(header, dict):
            raise IntegrityError(f"Safetensors header is not an object: {source}")
        self._headers[source] = header
        return header

    def tensor_info(self, name: str) -> tuple[str, str, tuple[int, ...], int]:
        try:
            source = self.weight_map[name]
        except KeyError as exc:
            raise IntegrityError(f"Kimi checkpoint is missing tensor {name}") from exc
        metadata = self._header(source).get(name)
        if not isinstance(metadata, dict):
            raise IntegrityError(f"tensor {name} is absent from indexed shard {source}")
        dtype = str(metadata.get("dtype", "")).upper()
        shape_value = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            dtype not in _DTYPE_BYTES
            or not isinstance(shape_value, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape_value)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in offsets)
        ):
            raise IntegrityError(f"tensor metadata is invalid: {name}")
        shape = tuple(int(value) for value in shape_value)
        byte_count = int(offsets[1]) - int(offsets[0])
        expected = math.prod(shape) * _DTYPE_BYTES[dtype]
        if byte_count != expected:
            raise IntegrityError(f"tensor byte count is invalid: {name}")
        return source, dtype, shape, byte_count


class KimiK3CudaAdapter:
    """Register Kimi's exact CUDA runtime with the canonical native loader."""

    adapter_id = "kimi_k3_cuda"
    adapter_version = "1"
    supported_model_types: ClassVar[frozenset[str]] = frozenset({"kimi_k3"})
    supported_architectures: ClassVar[frozenset[str]] = frozenset(
        {"KimiK3ForConditionalGeneration"}
    )

    def load_tokenizer(
        self,
        model_path: Path,
        identity_path: Path | None,
        *,
        worker_id: str,
    ) -> Any:
        if identity_path is None:
            raise IntegrityError("Kimi stage zero has no pinned tokenizer identity")
        from swarm_inference.model.kimi_tokenizer import load_pinned_kimi_tokenizer

        return load_pinned_kimi_tokenizer(
            model_path,
            identity_path,
            expected_worker_id=worker_id,
        )

    @staticmethod
    def encode_prompt(
        tokenizer: Any,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        from swarm_inference.model.kimi_tokenizer import (
            apply_kimi_prompt_special_tokens,
        )

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        return apply_kimi_prompt_special_tokens(
            tokenizer,
            token_ids,
            add_special_tokens=add_special_tokens,
        )

    def supports(self, config: Any) -> bool:
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        if not isinstance(config, dict):
            return False
        model_type = str(config.get("model_type", "")).lower()
        architectures = set(config.get("architectures") or ())
        text = config.get("text_config")
        return (
            model_type in self.supported_model_types
            and (not architectures or bool(architectures & self.supported_architectures))
            and isinstance(text, dict)
            and str(text.get("model_type", "")).lower() == "kimi_linear"
            and int(text.get("num_hidden_layers", 0)) == 93
        )

    def probe_model(self, model: ResolvedModelDescriptor) -> AdapterSupportReport:
        if model.format != "safetensors":
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_FORMAT,
                "native Kimi K3 CUDA execution requires Safetensors",
            )
        architecture = str(model.architecture or model.architecture_raw or "").lower()
        configuration_matches = self.supports(model.configuration)
        if not configuration_matches and not (
            "kimi" in architecture and "k3" in architecture
        ):
            return AdapterSupportReport(
                self.adapter_id,
                AdapterSupportStatus.UNSUPPORTED_ARCHITECTURE,
                f"architecture {model.architecture!r} is not Kimi K3",
            )
        return AdapterSupportReport(
            self.adapter_id,
            AdapterSupportStatus.SUPPORTED,
            "Kimi K3 Safetensors checkpoint is supported by the native CUDA adapter",
        )

    def map_tensor_to_component(self, tensor_name: str) -> ComponentRef:
        if tensor_name.startswith("language_model.model.embed_tokens."):
            return ComponentRef(ComponentKind.EMBEDDING)
        match = _LAYER_PATTERN.match(tensor_name)
        if match:
            return ComponentRef(ComponentKind.DECODER_LAYER, int(match.group(1)))
        if tensor_name in _FINAL_ATTENTION_NAMES or tensor_name.startswith(
            "language_model.model.norm."
        ):
            return ComponentRef(ComponentKind.FINAL_NORM)
        if tensor_name.startswith("language_model.lm_head."):
            return ComponentRef(ComponentKind.OUTPUT_HEAD)
        raise UnsupportedArchitectureError(
            f"Kimi K3 tensor is outside the certified text graph: {tensor_name}"
        )

    def _owned_tensor_names(
        self,
        catalog: _SafetensorCatalog,
        assignment: StageAssignment,
    ) -> tuple[str, ...]:
        prefixes = tuple(
            f"language_model.model.layers.{layer}."
            for layer in range(assignment.layer_start, assignment.layer_end)
        )
        names = [
            name
            for name in catalog.weight_map
            if name.startswith(prefixes)
            or (
                assignment.owns_embeddings
                and name.startswith("language_model.model.embed_tokens.")
            )
            or (
                assignment.owns_final_norm
                and (
                    name in _FINAL_ATTENTION_NAMES
                    or name.startswith("language_model.model.norm.")
                )
            )
            or (
                assignment.owns_output_projection
                and name.startswith("language_model.lm_head.")
            )
        ]
        return tuple(sorted(names))

    def validate_stage_assignment(
        self,
        model_path: Path,
        *,
        assignment: StageAssignment,
        stage_count: int,
        model_revision: str,
        tokenizer_revision: str,
        remote_experts: set[tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        del model_revision, tokenizer_revision
        if remote_experts:
            raise ValueError("the certified Kimi stage keeps every assigned layer expert local")
        config_path = model_path / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("Kimi checkpoint config is invalid") from exc
        if not self.supports(config):
            raise UnsupportedArchitectureError("checkpoint is not the certified 93-layer Kimi K3")
        if not 0 <= assignment.stage_id < stage_count:
            raise IntegrityError("Kimi stage ID is outside its topology")
        if not 0 <= assignment.layer_start < assignment.layer_end <= 93:
            raise IntegrityError("Kimi stage layer interval is invalid")
        catalog = _SafetensorCatalog(model_path)
        names = self._owned_tensor_names(catalog, assignment)
        if not names:
            raise IntegrityError("Kimi stage owns no checkpoint tensors")
        covered_layers = {
            component.layer_index
            for name in names
            if (component := self.map_tensor_to_component(name)).kind
            == ComponentKind.DECODER_LAYER
        }
        expected_layers = set(range(assignment.layer_start, assignment.layer_end))
        if covered_layers != expected_layers:
            raise IntegrityError("Kimi stage does not cover every assigned layer")
        endpoint_kinds = {self.map_tensor_to_component(name).kind for name in names}
        for kind, required in (
            (ComponentKind.EMBEDDING, assignment.owns_embeddings),
            (ComponentKind.FINAL_NORM, assignment.owns_final_norm),
            (ComponentKind.OUTPUT_HEAD, assignment.owns_output_projection),
        ):
            if required and kind not in endpoint_kinds:
                raise IntegrityError(f"Kimi stage omits required endpoint {kind.value}")
        source_bytes = sum(catalog.tensor_info(name)[3] for name in names)
        if source_bytes != assignment.weight_bytes:
            raise IntegrityError(
                f"Kimi stage source bytes {source_bytes} differ from assignment "
                f"{assignment.weight_bytes}"
            )
        return {
            "tensor_count": len(names),
            "source_bytes": source_bytes,
            "layer_ids": sorted(covered_layers),
        }

    def create_stage_executor(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise TypeError("Kimi native stage execution requires keyword control arguments")
        request = kwargs.get("request")
        resolved_model_path = kwargs.get("resolved_model_path")
        if request is None or resolved_model_path is None:
            raise TypeError("Kimi native stage execution requires a load request and model path")
        library_value = request.native_runtime_library
        expected_hash = request.native_runtime_library_sha256
        if library_value is None or expected_hash is None:
            raise IntegrityError("Kimi native stage load has no exact CUDA library identity")
        library = Path(library_value).expanduser().resolve()
        if not library.is_file():
            raise FileNotFoundError(f"Kimi native CUDA library is missing: {library}")
        actual_hash = _sha256_file(library)
        if actual_hash != expected_hash:
            raise IntegrityError(
                "Kimi native CUDA library SHA-256 differs from the load request"
            )
        if request.fast_path_id != "colibri-kimi-k3-cuda":
            raise IntegrityError("Kimi stage did not request the certified CUDA fast path")
        device_text = str(request.device).lower()
        if not device_text.startswith("native-cuda:"):
            raise IntegrityError("Kimi native adapter requires a native-cuda device")
        device = int(device_text.split(":", 1)[1])
        # H014-026b promotes the already-qualified executor through the product
        # adapter seam.  General layer ownership is the next research cycle.
        from swarm_inference.execution.kimi_k3_stage import KimiK3StageExecutor

        return KimiK3StageExecutor(
            request=request,
            checkpoint=Path(resolved_model_path),
            cuda_library=library,
            device=device,
        )

    def describe(
        self,
        model_path: Path,
        *,
        model_id: str,
        model_revision: str,
    ) -> ModelDescription:
        config_path = model_path / "config.json"
        index_path = model_path / "model.safetensors.index.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not self.supports(config):
            raise UnsupportedArchitectureError("checkpoint is not Kimi K3")
        catalog = _SafetensorCatalog(model_path)
        tensors: list[TensorInfo] = []
        for name in sorted(catalog.weight_map):
            try:
                component = self.map_tensor_to_component(name)
            except UnsupportedArchitectureError:
                continue
            source, dtype, shape, byte_count = catalog.tensor_info(name)
            tensors.append(
                TensorInfo(
                    name=name,
                    source_file=source,
                    dtype=dtype,
                    shape=shape,
                    bytes=byte_count,
                    component=component,
                )
            )
        return ModelDescription(
            model_id=model_id,
            model_revision=model_revision,
            model_path=model_path.resolve(),
            config=config,
            tensors=tensors,
            source_file_hashes={},
            config_file_hashes={
                "config.json": _sha256_file(config_path),
                "model.safetensors.index.json": _sha256_file(index_path),
            },
        )

    def inspect(self, model: ResolvedModelDescriptor) -> ModelDescription:
        if not model.local_paths:
            raise FileNotFoundError("native Kimi inspection requires acquired local files")
        roots = {str(Path(path).resolve().parent) for path in model.local_paths}
        if len(roots) != 1:
            raise IntegrityError("native Kimi files do not share one checkpoint directory")
        return self.describe(
            Path(next(iter(roots))),
            model_id=model.model_id,
            model_revision=model.revision,
        )

    def build_stage_artifact(self, *args: Any, **kwargs: Any) -> Any:
        from swarm_inference.cluster.artifacts import build_native_stage_artifact

        return build_native_stage_artifact(self, *args, **kwargs)

    def create_stage_module(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise UnsupportedArchitectureError("Kimi K3 uses only its native CUDA stage executor")

    def load_stage_weights(self, *args: Any, **kwargs: Any) -> list[str]:
        del args, kwargs
        raise UnsupportedArchitectureError("Kimi weights load through the native CUDA executor")

    def reference_executor(self, model: ResolvedModelDescriptor, **kwargs: Any) -> Any:
        del model, kwargs
        raise UnsupportedArchitectureError(
            "Kimi K3 reference execution is provided by the certified serial oracle"
        )

    def fast_paths(self) -> tuple[Any, ...]:
        return ()


__all__ = ["KimiK3CudaAdapter"]
