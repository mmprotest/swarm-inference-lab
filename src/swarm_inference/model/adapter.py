"""Architecture adapter interface for inspection, sharding, and stage construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from swarm_inference.config.models import ModelManifest, StageDefinition


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
