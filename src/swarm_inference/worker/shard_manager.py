"""Stage-only shard loading with logical memory and hash enforcement."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

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
from swarm_inference.protocol.checksums import sha256_bytes, sha256_file
from swarm_inference.runtime.telemetry import lifecycle_observer
from swarm_inference.security.signatures import canonical_json_bytes

_LOAD_RECORD_ENRICHMENT_FIELDS = {
    "load_record_checksum_verified",
    "peak_cuda_memory_bytes",
    "peak_cuda_reserved_bytes",
    "process_memory_after_execution",
    "proof_verified",
    "source_health_proof_checksum",
    "source_health_proof_signature",
    "stage_transfer_metrics",
    "worker_id",
}


def attach_load_record_checksum(record: dict[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "load_record_checksum" and key not in _LOAD_RECORD_ENRICHMENT_FIELDS
    }
    return {
        **record,
        "load_record_checksum": sha256_bytes(canonical_json_bytes(unsigned)),
    }


def verify_load_record_checksum(record: dict[str, Any]) -> bool:
    expected = str(record.get("load_record_checksum", ""))
    if not expected:
        return False
    unsigned = {
        key: value
        for key, value in record.items()
        if key != "load_record_checksum" and key not in _LOAD_RECORD_ENRICHMENT_FIELDS
    }
    return sha256_bytes(canonical_json_bytes(unsigned)) == expected


class ShardManager:
    def __init__(
        self,
        *,
        memory_limit_bytes: int,
        total_memory_limit_bytes: int | None = None,
    ) -> None:
        if memory_limit_bytes <= 0:
            raise ValueError("memory_limit_bytes must be positive")
        if total_memory_limit_bytes is not None and total_memory_limit_bytes <= 0:
            raise ValueError("total_memory_limit_bytes must be positive")
        self.memory_limit_bytes = memory_limit_bytes
        self.total_memory_limit_bytes = total_memory_limit_bytes
        self.modules: dict[int, StageModule] = {}
        self.loaded_tensor_names: dict[int, list[str]] = {}
        self.loaded_bytes: dict[int, int] = {}
        self.loaded_total_bytes: dict[int, int] = {}
        self.load_records: dict[int, dict[str, Any]] = {}

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
        stage_total_bytes = stage.required_total_memory_bytes or stage.required_memory_bytes
        projected_total = sum(self.loaded_total_bytes.values()) + stage_total_bytes
        if (
            self.total_memory_limit_bytes is not None
            and projected_total > self.total_memory_limit_bytes
        ):
            raise MemoryLimitExceededError(
                f"loading stage {stage.stage_id} would require an estimated "
                f"{projected_total} total bytes; worker logical total-memory limit "
                f"is {self.total_memory_limit_bytes}"
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
        self.loaded_total_bytes[stage.stage_id] = (
            stage.required_total_memory_bytes or stage.required_memory_bytes
        )
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
        recorder = lifecycle_observer()
        verification_started = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "shard_verification_started",
                monotonic_ns=verification_started,
                bytes_count=stage.required_memory_bytes,
                details={"shard_path": str(root)},
            )
        actual_hash = hash_shard_directory(root)
        verification_completed = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "shard_verification_completed",
                monotonic_ns=verification_completed,
                duration_ns=verification_completed - verification_started,
                bytes_count=stage.required_memory_bytes,
                details={
                    "shard_path": str(root),
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "hash_valid": actual_hash == expected_hash,
                },
            )
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
        before = self._process_memory_snapshot(torch)
        if backend == Backend.TORCH_CUDA:
            torch.cuda.reset_peak_memory_stats(device)
        construction_started = time.monotonic_ns()
        if recorder is not None:
            recorder.emit(
                "stage_module_construction_started",
                monotonic_ns=construction_started,
                bytes_count=stage.required_memory_bytes,
            )
        try:
            module = adapter.create_stage_module(config, stage, device, dtype)
            if backend == Backend.TORCH_CUDA:
                torch.cuda.synchronize(device)
            construction_completed = time.monotonic_ns()
            if recorder is not None:
                recorder.emit(
                    "stage_module_construction_completed",
                    monotonic_ns=construction_completed,
                    duration_ns=construction_completed - construction_started,
                    memory_metrics=self._process_memory_snapshot(torch),
                )
            weight_load_started = time.monotonic_ns()
            if recorder is not None:
                recorder.emit(
                    "weight_load_started",
                    monotonic_ns=weight_load_started,
                    bytes_count=stage.required_memory_bytes,
                    details={"weight_materialisation_mode": "host-to-device-copy"},
                )
            loaded = adapter.load_stage_weights(module, root, manifest=manifest)
            if backend == Backend.TORCH_CUDA:
                torch.cuda.synchronize(device)
            weight_load_completed = time.monotonic_ns()
            if recorder is not None:
                recorder.emit(
                    "weight_load_completed",
                    monotonic_ns=weight_load_completed,
                    duration_ns=weight_load_completed - weight_load_started,
                    bytes_count=stage.required_memory_bytes,
                    memory_metrics=self._process_memory_snapshot(torch),
                    details={"weight_materialisation_mode": "host-to-device-copy"},
                )
        except torch.cuda.OutOfMemoryError as exc:
            failure = self._process_memory_snapshot(torch)
            raise MemoryLimitExceededError(
                f"CUDA out of memory loading worker stage {stage.stage_id}; "
                f"shard_bytes={stage.required_memory_bytes}; "
                f"allocated={failure['cuda_allocated_bytes']}; "
                f"reserved={failure['cuda_reserved_bytes']}; "
                f"free_vram={failure['cuda_free_bytes']}; "
                f"peak={failure['cuda_peak_allocated_bytes']}"
            ) from exc
        after = self._process_memory_snapshot(torch)
        self.modules[stage.stage_id] = module
        self.loaded_tensor_names[stage.stage_id] = loaded
        self.loaded_bytes[stage.stage_id] = stage.required_memory_bytes
        self.loaded_total_bytes[stage.stage_id] = (
            stage.required_total_memory_bytes or stage.required_memory_bytes
        )
        all_source_names = set(manifest.tensor_to_stages)
        if not all_source_names:
            all_source_names = {
                name for manifest_stage in manifest.stages for name in manifest_stage.tensor_names
            }
        required_total = stage.required_total_memory_bytes or stage.required_memory_bytes
        self.load_records[stage.stage_id] = attach_load_record_checksum(
            {
                "process_id": os.getpid(),
                "stage_id": stage.stage_id,
                "shard_path": str(root),
                "shard_hash": actual_hash,
                "tensor_names_loaded": loaded,
                "tensor_count": len(loaded),
                "total_loaded_weight_bytes": stage.required_memory_bytes,
                "stage_required_total_memory_bytes": required_total,
                "decoder_layer_range": [stage.layer_start, stage.layer_end],
                "decoder_layer_count": stage.layer_end - stage.layer_start,
                "embeddings_loaded": stage.owns_embeddings,
                "final_norm_loaded": stage.owns_final_norm,
                "lm_head_loaded": stage.owns_output_head,
                "cuda_memory_before_load": before,
                "cuda_memory_after_load": after,
                "host_rss_before_load": before["host_rss_bytes"],
                "host_rss_after_load": after["host_rss_bytes"],
                "full_model_bytes": manifest.total_weight_bytes,
                "logical_worker_weight_limit_bytes": self.memory_limit_bytes,
                "logical_worker_total_memory_limit_bytes": self.total_memory_limit_bytes,
                "full_model_exceeds_logical_limit": (
                    manifest.total_weight_bytes > self.memory_limit_bytes
                ),
                "full_model_exceeds_logical_total_limit": (
                    self.total_memory_limit_bytes is not None
                    and manifest.total_weight_bytes > self.total_memory_limit_bytes
                ),
                "stage_fits_logical_total_memory_limit": (
                    self.total_memory_limit_bytes is not None
                    and required_total <= self.total_memory_limit_bytes
                ),
                "loaded_complete_source_tensor_set": set(loaded) == all_source_names,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        print(
            json.dumps(
                {
                    "event": "stage_shard_loaded",
                    "process_id": os.getpid(),
                    "stage_id": stage.stage_id,
                    "decoder_layer_range": [stage.layer_start, stage.layer_end],
                    "shard_hash": actual_hash,
                    "tensor_count": len(loaded),
                    "loaded_weight_bytes": stage.required_memory_bytes,
                    "cuda_allocated_bytes": after["cuda_allocated_bytes"],
                    "host_rss_bytes": after["host_rss_bytes"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return module

    @staticmethod
    def _process_memory_snapshot(torch_module: Any | None = None) -> dict[str, int]:
        process = psutil.Process()
        result = {
            "host_rss_bytes": int(process.memory_info().rss),
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "cuda_peak_allocated_bytes": 0,
            "cuda_peak_reserved_bytes": 0,
            "cuda_free_bytes": 0,
            "cuda_total_bytes": 0,
        }
        torch = torch_module
        if torch is None:
            try:
                import torch as imported_torch

                torch = imported_torch
            except (ImportError, OSError):
                return result
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info(0)
            result.update(
                {
                    "cuda_allocated_bytes": int(torch.cuda.memory_allocated(0)),
                    "cuda_reserved_bytes": int(torch.cuda.memory_reserved(0)),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                    "cuda_free_bytes": int(free),
                    "cuda_total_bytes": int(total),
                }
            )
        return result

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
        self.loaded_total_bytes.pop(stage_id, None)
        self.load_records.pop(stage_id, None)

    def proof(self) -> dict[str, Any]:
        current_memory = self._process_memory_snapshot()
        return {
            "memory_limit_bytes": self.memory_limit_bytes,
            "total_memory_limit_bytes": self.total_memory_limit_bytes,
            "total_loaded_bytes": self.total_loaded_bytes,
            "total_loaded_estimated_bytes": sum(self.loaded_total_bytes.values()),
            "process_id": os.getpid(),
            "current_process_memory": current_memory,
            "stages": {
                str(stage_id): {
                    "logical_loaded_bytes": self.loaded_bytes[stage_id],
                    "tensor_names": self.loaded_tensor_names[stage_id],
                    "module_state": self.modules[stage_id].state_summary(),
                    "load_record": self.load_records.get(stage_id, {}),
                }
                for stage_id in sorted(self.modules)
            },
        }
