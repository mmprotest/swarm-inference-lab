"""Persistent real K3 expert-fragment executor for mandatory small GPUs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _array_fingerprint
from swarm_inference.execution.kimi_k3_graph_runtime import (
    KimiCudaGraphRunner,
    _LayerResources,
    _pointer_offset,
)

from .constants import LATENT_SIZE, SUB_LAYER_TARGET, TOP_K


class ExpertPartitionExecutor:
    """Own complete experts forming one strict fragment of an ordinary layer."""

    native_primitive = "coli_cuda_resident_k3_mxfp4_expert_partition"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        *,
        worker_index: int,
        worker_count: int,
        layer: int = SUB_LAYER_TARGET,
        device: int = 0,
    ) -> None:
        if worker_count != 4 or not 0 <= worker_index < worker_count:
            raise ValueError("E025 promoted expert partition requires four workers")
        started = time.perf_counter_ns()
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.layer = layer
        self.runner = KimiCudaGraphRunner(checkpoint, cuda_library, device)
        self.runtime = self.runner.runtime
        self.runtime.set_telemetry("minimal")
        self.resources = _LayerResources(self.runtime)
        self.owned_experts = tuple(
            expert
            for expert in range(self.runner.config.experts)
            if expert % worker_count == worker_index
        )
        self.handles: dict[int, tuple[Any, Any, Any]] = {}
        for loaded, expert in enumerate(self.owned_experts, start=1):
            self.handles[expert] = self.runner._upload_expert(
                self.resources,
                layer,
                expert,
            )
            if loaded % 64 == 0 or loaded == len(self.owned_experts):
                print(
                    f"[e025:expert-load] worker={worker_index} "
                    f"expert={loaded}/{len(self.owned_experts)}",
                    flush=True,
                )
        self.input = self.resources.allocate(LATENT_SIZE)
        self.output = self.resources.allocate(TOP_K * LATENT_SIZE)
        self.zero_output = np.zeros((TOP_K, LATENT_SIZE), dtype=np.float32)
        self.runtime.upload_activation(self.output, self.zero_output)
        self.runtime.synchronize()
        self.invocation_count = 0
        self.native_expert_calls = 0
        self.last_execution: dict[str, Any] = {}
        self.ready = {
            "worker_index": worker_index,
            "worker_count": worker_count,
            "layer": layer,
            "owned_expert_count": len(self.owned_experts),
            "owned_expert_min": min(self.owned_experts),
            "owned_expert_max": max(self.owned_experts),
            "ownership": f"expert_id modulo {worker_count} equals {worker_index}",
            "resident_expert_tensor_bytes": self.resources.resident_tensor_bytes,
            "persistent_buffer_bytes": (TOP_K + 1) * LATENT_SIZE * 4,
            "tracked_fragment_bytes": self.resources.resident_tensor_bytes
            + (TOP_K + 1) * LATENT_SIZE * 4,
            "memory_after_load": self.runtime.mem_info(),
            "weight_fingerprint": "sha256:" + self.resources.weight_digest.hexdigest(),
            "cuda_library_sha256": self.runner.runtime.sha256,
            "startup_wall_ms": (time.perf_counter_ns() - started) / 1e6,
            "native_primitive": self.native_primitive,
            "whole_layer_fallback": False,
        }

    def execute(
        self,
        selected_expert_ids: np.ndarray,
        latent_activation: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        selected = np.ascontiguousarray(selected_expert_ids, dtype=np.int32).reshape(-1)
        latent = np.ascontiguousarray(latent_activation, dtype=np.float32).reshape(-1)
        if selected.shape != (TOP_K,) or len(set(selected.tolist())) != TOP_K:
            raise ValueError("E025 expert request requires 16 unique ordered experts")
        if latent.shape != (LATENT_SIZE,) or not np.isfinite(latent).all():
            raise ValueError("E025 expert request has an invalid latent activation")
        owned_slots = [
            slot
            for slot, expert in enumerate(selected.tolist())
            if int(expert) in self.handles
        ]
        started = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, latent)
        self.runtime.upload_activation(self.output, self.zero_output)
        self.runtime.profile_begin()
        for slot in owned_slots:
            expert = int(selected[slot])
            self.runtime.execute_resident(
                self.handles[expert],
                _pointer_offset(self.output, slot * LATENT_SIZE),
                self.input,
                1,
            )
        cuda_ms = self.runtime.profile_end()
        output = self.runtime.download_activation(
            self.output,
            (TOP_K, LATENT_SIZE),
        )
        self.invocation_count += 1
        self.native_expert_calls += len(owned_slots)
        record = {
            "worker_index": self.worker_index,
            "layer": self.layer,
            "invocation_count": self.invocation_count,
            "owned_selected_count": len(owned_slots),
            "owned_slots": owned_slots,
            "owned_selected_expert_ids": [int(selected[slot]) for slot in owned_slots],
            "selected_expert_ids": selected.tolist(),
            "native_expert_calls": len(owned_slots),
            "native_expert_calls_total": self.native_expert_calls,
            "cuda_ms": cuda_ms,
            "wall_ms": (time.perf_counter_ns() - started) / 1e6,
            "input_fingerprint": _array_fingerprint(latent),
            "output_fingerprint": _array_fingerprint(output),
            "weight_loads_during_execute": 0,
            "persistent_buffer_allocations_during_execute": 0,
            "whole_layer_fallback": False,
            "native_primitive": self.native_primitive,
        }
        self.last_execution = record
        return output, record

    def health(self) -> dict[str, Any]:
        self.runtime.synchronize()
        return {
            "cuda_error_state_ok": self.runtime.error_state_ok(),
            "memory": self.runtime.mem_info(),
            "invocation_count": self.invocation_count,
            "native_expert_calls": self.native_expert_calls,
            "whole_layer_fallback": False,
        }

    def close(self) -> None:
        self.resources.close()
        self.runner.close()


__all__ = ["ExpertPartitionExecutor"]
