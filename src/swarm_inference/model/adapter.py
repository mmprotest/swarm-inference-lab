"""Native architecture adapters and their production registry."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from swarm_inference.config.models import ModelManifest, StageDefinition
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.partition import LayerCost, ModelPartitionMetadata, StageAssignment


class ComponentKind(StrEnum):
    EMBEDDING = "embedding"
    DECODER_LAYER = "decoder-layer"
    FINAL_NORM = "final-norm"
    OUTPUT_HEAD = "output-head"


@dataclass(frozen=True, slots=True)
class ComponentRef:
    kind: ComponentKind
    layer_index: int | None = None


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    source_file: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    component: ComponentRef


@dataclass(slots=True)
class ModelDescription:
    model_id: str
    model_revision: str
    model_path: Path
    config: dict[str, Any]
    tensors: list[TensorInfo]
    source_file_hashes: dict[str, str]
    config_file_hashes: dict[str, str] = field(default_factory=dict)
    tokenizer_file_hashes: dict[str, str] = field(default_factory=dict)


NativeModelDescription = ModelDescription


class AdapterSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_ARCHITECTURE = "UNSUPPORTED_ARCHITECTURE"
    INCOMPLETE_METADATA = "INCOMPLETE_METADATA"


@dataclass(frozen=True, slots=True)
class AdapterSupportReport:
    adapter_id: str
    status: AdapterSupportStatus
    reason: str

    @property
    def supported(self) -> bool:
        return self.status == AdapterSupportStatus.SUPPORTED


class ModelAdapter(Protocol):
    def supports(self, config: Any) -> bool: ...

    def describe(
        self,
        model_path: Path,
        *,
        model_id: str,
        model_revision: str,
    ) -> ModelDescription: ...

    def map_tensor_to_component(self, tensor_name: str) -> ComponentRef: ...

    def create_stage_module(
        self,
        config: Any,
        stage: StageDefinition,
        device: Any,
        dtype: Any,
    ) -> Any: ...

    def load_stage_weights(
        self,
        module: Any,
        shard_path: Path,
        *,
        manifest: ModelManifest,
    ) -> list[str]: ...


@runtime_checkable
class NativeModelAdapter(Protocol):
    """Complete native adapter boundary used by the canonical stage engine."""

    adapter_id: str
    adapter_version: str

    def supports(self, config: Any) -> bool: ...

    def map_tensor_to_component(self, tensor_name: str) -> ComponentRef: ...

    def probe_model(self, model: ResolvedModelDescriptor) -> AdapterSupportReport: ...

    def inspect(self, model: ResolvedModelDescriptor) -> NativeModelDescription: ...

    def build_stage_artifact(self, *args: Any, **kwargs: Any) -> Any: ...

    def create_stage_executor(self, *args: Any, **kwargs: Any) -> Any: ...

    def reference_executor(self, model: ResolvedModelDescriptor, **kwargs: Any) -> Any: ...

    def fast_paths(self) -> tuple[Any, ...]: ...


class NativeModelAdapterRegistry:
    """Data-driven architecture dispatch; generic callers never branch on families."""

    def __init__(self, adapters: tuple[NativeModelAdapter, ...] = ()) -> None:
        self._adapters: dict[str, NativeModelAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: NativeModelAdapter, *, replace: bool = False) -> None:
        adapter_id = adapter.adapter_id.strip()
        if not adapter_id:
            raise ValueError("native adapter ID cannot be empty")
        if adapter_id in self._adapters and not replace:
            raise ValueError(f"native adapter {adapter_id!r} is already registered")
        self._adapters[adapter_id] = adapter

    def get(self, adapter_id: str) -> NativeModelAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"native adapter {adapter_id!r} is not registered") from exc

    def adapters(self) -> tuple[NativeModelAdapter, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))

    def probe_all(self, model: ResolvedModelDescriptor) -> tuple[AdapterSupportReport, ...]:
        return tuple(adapter.probe_model(model) for adapter in self.adapters())

    def resolve(self, model: ResolvedModelDescriptor) -> NativeModelAdapter:
        matches = [adapter for adapter in self.adapters() if adapter.probe_model(model).supported]
        if not matches:
            reasons = "; ".join(
                f"{item.adapter_id}={item.status.value}: {item.reason}"
                for item in self.probe_all(model)
            )
            raise LookupError(f"no native model adapter supports the descriptor; {reasons}")
        if len(matches) > 1:
            names = ", ".join(sorted(item.adapter_id for item in matches))
            raise LookupError(f"native model adapter probes are ambiguous: {names}")
        return matches[0]

    def resolve_config(self, config: Any) -> NativeModelAdapter:
        matches = [
            adapter
            for adapter in self.adapters()
            if callable(getattr(adapter, "supports", None)) and adapter.supports(config)
        ]
        if len(matches) != 1:
            raise LookupError(
                "model configuration must match exactly one native adapter; matched="
                + ",".join(sorted(item.adapter_id for item in matches))
            )
        return matches[0]


def default_native_adapter_registry() -> NativeModelAdapterRegistry:
    """Load built-in and third-party adapters through the packaging boundary.

    Installed distributions expose adapters as entry points.  Reading the source
    project's entry-point table is a development-only fallback for an editable
    checkout whose distribution metadata predates the current source tree.
    """

    targets: dict[str, str] = {}
    for entry_point in importlib.metadata.entry_points(group="swarm_inference.native_adapters"):
        targets[entry_point.name] = entry_point.value

    for parent in Path(__file__).resolve().parents:
        project_file = parent / "pyproject.toml"
        if not project_file.is_file():
            continue
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))
        configured = (
            project.get("project", {})
            .get("entry-points", {})
            .get("swarm_inference.native_adapters", {})
        )
        for name, target in configured.items():
            targets.setdefault(str(name), str(target))
        break

    adapters: list[NativeModelAdapter] = []
    for name, target in sorted(targets.items()):
        module_name, separator, attribute_name = target.partition(":")
        if not separator or not module_name or not attribute_name:
            raise RuntimeError(f"native adapter entry point {name!r} has invalid target {target!r}")
        factory: Any = importlib.import_module(module_name)
        for attribute in attribute_name.split("."):
            factory = getattr(factory, attribute)
        adapter = factory()
        if not isinstance(adapter, NativeModelAdapter):
            raise TypeError(f"native adapter entry point {name!r} does not satisfy the protocol")
        adapters.append(adapter)
    return NativeModelAdapterRegistry(tuple(adapters))


def partition_metadata_from_description(
    description: ModelDescription,
    *,
    tokenizer_revision: str,
) -> ModelPartitionMetadata:
    """Build planner facts for a dense decoder from adapter-owned tensor mappings."""

    config = description.config
    layer_count = int(config["num_hidden_layers"])
    per_layer = [0] * layer_count
    embedding = 0
    final = 0
    for tensor in description.tensors:
        if tensor.component.kind == ComponentKind.DECODER_LAYER:
            if tensor.component.layer_index is None:
                raise ValueError(f"decoder tensor {tensor.name} has no layer index")
            per_layer[tensor.component.layer_index] += tensor.bytes
        elif tensor.component.kind == ComponentKind.EMBEDDING:
            embedding += tensor.bytes
        elif tensor.component.kind in {ComponentKind.FINAL_NORM, ComponentKind.OUTPUT_HEAD}:
            final += tensor.bytes
    if bool(config.get("tie_word_embeddings")) and not any(
        item.component.kind == ComponentKind.OUTPUT_HEAD for item in description.tensors
    ):
        final += embedding
    if not per_layer or any(value <= 0 for value in per_layer):
        raise ValueError("every configured decoder layer must own checkpoint tensors")
    dtypes = {item.dtype.upper() for item in description.tensors}
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
    dtype_bytes = max((widths.get(item, 0) for item in dtypes), default=0)
    if dtype_bytes <= 0:
        raise ValueError(f"unsupported native tensor dtypes: {sorted(dtypes)}")
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or hidden // heads)
    kv_bytes = 2 * kv_heads * head_dim * dtype_bytes
    activation_bytes = hidden * dtype_bytes
    costs = tuple(
        LayerCost(
            layer_id=index,
            execution_ns=max(1, int(weight / 4_000_000_000 * 1e9)),
            weight_bytes=weight,
            kv_bytes_per_token=kv_bytes,
            peak_temporary_bytes=max(activation_bytes * 16, weight // 32),
            activation_bytes=activation_bytes,
            measured=False,
        )
        for index, weight in enumerate(per_layer)
    )
    identity = json.dumps(
        {
            "model_revision": description.model_revision,
            "tokenizer_revision": tokenizer_revision,
            "source_hashes": description.source_file_hashes,
            "costs": [
                {
                    "layer": item.layer_id,
                    "weights": item.weight_bytes,
                    "kv": item.kv_bytes_per_token,
                }
                for item in costs
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return ModelPartitionMetadata(
        layer_costs=costs,
        embedding_weight_bytes=embedding,
        final_weight_bytes=final,
        dtype_bytes=dtype_bytes,
        hidden_size=hidden,
        model_revision=description.model_revision,
        tokenizer_revision=tokenizer_revision,
        metadata_hash=digest,
        model_fingerprint="sha256:" + digest,
        quantization_fingerprint="sha256:"
        + hashlib.sha256(
            json.dumps(config.get("quantization_config"), sort_keys=True, default=str).encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def validate_dense_stage_assignment(
    metadata: ModelPartitionMetadata,
    *,
    assignment: StageAssignment,
    stage_count: int,
) -> None:
    if stage_count > len(metadata.layer_costs):
        raise ValueError("stage topology cannot exceed decoder layer count")
    if assignment.layer_end > len(metadata.layer_costs):
        raise ValueError("stage assignment exceeds decoder layer count")
    selected = metadata.layer_costs[assignment.layer_start : assignment.layer_end]
    expected_weight_bytes = sum(item.weight_bytes for item in selected)
    if assignment.stage_id == 0:
        expected_weight_bytes += metadata.embedding_weight_bytes
    if assignment.stage_id == stage_count - 1:
        expected_weight_bytes += metadata.final_weight_bytes
    if assignment.weight_bytes != expected_weight_bytes:
        raise ValueError("stage assignment weight bytes differ from adapter inspection")
    expected_kv = sum(item.kv_bytes_per_token for item in selected)
    if assignment.kv_cache_bytes_per_token != expected_kv:
        raise ValueError("stage assignment KV bytes differ from adapter inspection")


__all__ = [
    "AdapterSupportReport",
    "AdapterSupportStatus",
    "ComponentKind",
    "ComponentRef",
    "ModelAdapter",
    "ModelDescription",
    "NativeModelAdapter",
    "NativeModelAdapterRegistry",
    "NativeModelDescription",
    "TensorInfo",
    "default_native_adapter_registry",
    "partition_metadata_from_description",
    "validate_dense_stage_assignment",
]
