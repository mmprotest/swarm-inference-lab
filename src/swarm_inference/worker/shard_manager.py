"""Stage-only shard loading with logical memory and hash enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.config.models import (
    Backend,
    ModelManifest,
    StageDefinition,
    SyntheticModelConfig,
)
from swarm_inference.exceptions import IntegrityError, MemoryLimitExceededError
from swarm_inference.model.manifest import hash_shard_directory
from swarm_inference.model.qwen3 import Qwen3Adapter
from swarm_inference.model.stage_module import StageModule
from swarm_inference.model.synthetic import SyntheticStageModule
from swarm_inference.protocol.checksums import sha256_file


class ShardManager:
    def __init__(self, *, memory_limit_bytes: int) -> None:
        if memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        self.memory_limit_bytes = memory_limit_bytes
        self.modules: dict[int, StageModule] = {}
        self.loaded_tensor_names: dict[int, list[str]] = {}
        self.loaded_bytes: dict[int, int] = {}

    @property
    def total_loaded_bytes(self) -> int:
        return sum(self.loaded_bytes.values())

    def _reserve(self, stage: StageDefinition) -> None:
        projected = self.total_loaded_bytes + stage.required_memory_bytes
        if projected > self.memory_limit_bytes:
            raise MemoryLimitExceededError(
                f"loading stage {stage.stage_id} would use {projected} bytes; "
                f"worker logical limit is {self.memory_limit_bytes}"
            )

    def load_synthetic(
        self,
        *,
        config: SyntheticModelConfig,
        stage: StageDefinition,
        corrupt: bool = False,
    ) -> StageModule:
        if stage.stage_id in self.modules:
            raise IntegrityError(f"stage {stage.stage_id} is already loaded")
        self._reserve(stage)
        module = SyntheticStageModule(config=config, stage=stage, corrupt=corrupt)
        self.modules[stage.stage_id] = module
        self.loaded_tensor_names[stage.stage_id] = list(stage.tensor_names)
        self.loaded_bytes[stage.stage_id] = stage.required_memory_bytes
        return module

    def verify_shard(self, path: str | Path, expected_hash: str) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise IntegrityError(f"shard does not exist: {resolved}")
        actual = sha256_file(resolved)
        if actual != expected_hash:
            raise IntegrityError(
                f"shard hash mismatch for {resolved}: expected={expected_hash} actual={actual}"
            )

    def load_qwen3(
        self,
        *,
        config: dict[str, Any],
        manifest: ModelManifest,
        stage: StageDefinition,
        shard_path: str | Path,
        expected_hash: str,
        backend: Backend,
        dtype_name: str | None = None,
    ) -> StageModule:
        if stage.stage_id in self.modules:
            raise IntegrityError(f"stage {stage.stage_id} is already loaded")
        self._reserve(stage)
        root = Path(shard_path).expanduser().resolve()
        if not root.is_dir():
            raise IntegrityError(f"Qwen3 stage shard directory does not exist: {root}")
        actual_hash = hash_shard_directory(root)
        if actual_hash != expected_hash:
            raise IntegrityError(
                f"stage directory hash mismatch for {root}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        import torch

        device = {
            Backend.TORCH_CPU: torch.device("cpu"),
            Backend.TORCH_CUDA: torch.device("cuda"),
            Backend.TORCH_MPS: torch.device("mps"),
        }.get(backend)
        if device is None:
            raise IntegrityError(f"backend {backend.value} cannot execute Qwen3")
        normalised = (dtype_name or manifest.weight_dtype).lower()
        dtypes = {
            "f16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "f32": torch.float32,
            "float32": torch.float32,
        }
        try:
            dtype = dtypes[normalised]
        except KeyError as exc:
            raise IntegrityError(f"unsupported Qwen3 execution dtype {normalised}") from exc
        adapter = Qwen3Adapter()
        module = adapter.create_stage_module(config, stage, device, dtype)
        loaded = adapter.load_stage_weights(module, root, manifest=manifest)
        self.modules[stage.stage_id] = module
        self.loaded_tensor_names[stage.stage_id] = loaded
        self.loaded_bytes[stage.stage_id] = stage.required_memory_bytes
        return module

    def module(self, stage_id: int) -> StageModule:
        try:
            return self.modules[stage_id]
        except KeyError as exc:
            raise IntegrityError(f"stage {stage_id} is not loaded by this worker") from exc

    def unload(self, stage_id: int) -> None:
        module = self.modules.pop(stage_id, None)
        if module is not None:
            for request_key in list(module.state_summary().get("state_checksums", {})):
                module.cancel(str(request_key).split(":", 1)[0])
        self.loaded_tensor_names.pop(stage_id, None)
        self.loaded_bytes.pop(stage_id, None)

    def proof(self) -> dict[str, Any]:
        return {
            "memory_limit_bytes": self.memory_limit_bytes,
            "total_loaded_bytes": self.total_loaded_bytes,
            "stages": {
                str(stage_id): {
                    "logical_loaded_bytes": self.loaded_bytes[stage_id],
                    "tensor_names": self.loaded_tensor_names[stage_id],
                    "module_state": self.modules[stage_id].state_summary(),
                }
                for stage_id in sorted(self.modules)
            },
        }
