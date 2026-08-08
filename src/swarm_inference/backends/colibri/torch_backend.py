"""Persistent Colibri routed-expert component for native PyTorch stages.

The outer stage supplies exact router decisions. This component owns only
adapter-described expert tensors and keeps them in a bounded device LRU backed
by the immutable stage artifact. No hidden state traverses the coordinator.
"""

from __future__ import annotations

import json
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.backends.colibri.architecture import ExpertDescriptor
from swarm_inference.execution.moe import MoeBackendCapabilities
from swarm_inference.model.architecture import architecture_from_config
from swarm_inference.model.descriptor import ResolvedModelDescriptor

_HEADER_LENGTH = struct.Struct("<Q")


@dataclass(slots=True)
class _ResidentExpert:
    tensors: dict[str, torch.Tensor]
    roles: dict[str, str]
    byte_size: int


def _tensor_inventory(root: Path) -> tuple[tuple[str, tuple[int, ...], str, int], ...]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError("Colibri stage artifact has no safetensors index")
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = raw_index.get("weight_map") if isinstance(raw_index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Colibri stage artifact index has no tensor map")
    by_file: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(str(filename), set()).add(str(name))
    result: list[tuple[str, tuple[int, ...], str, int]] = []
    for filename, selected in sorted(by_file.items()):
        path = (root / filename).resolve()
        if path.parent != root and root not in path.parents:
            raise ValueError("Colibri tensor path escapes its immutable artifact")
        with path.open("rb") as handle:
            encoded = handle.read(_HEADER_LENGTH.size)
            if len(encoded) != _HEADER_LENGTH.size:
                raise ValueError(f"truncated safetensors header in {filename}")
            length = _HEADER_LENGTH.unpack(encoded)[0]
            if length <= 1 or length > min(path.stat().st_size - 8, 256 * 1024 * 1024):
                raise ValueError(f"unsafe safetensors header length in {filename}")
            header = json.loads(handle.read(length))
        for name in sorted(selected):
            metadata = header.get(name) if isinstance(header, dict) else None
            if not isinstance(metadata, dict):
                raise ValueError(f"safetensors index names missing tensor {name}")
            shape = tuple(int(item) for item in metadata.get("shape", ()))
            dtype = str(metadata.get("dtype", ""))
            offsets = metadata.get("data_offsets")
            if (
                not shape
                or any(item <= 0 for item in shape)
                or not dtype
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(item, int) for item in offsets)
                or offsets[0] < 0
                or offsets[1] <= offsets[0]
            ):
                raise ValueError(f"invalid Colibri tensor metadata for {name}")
            byte_size = offsets[1] - offsets[0]
            result.append((name, shape, dtype, byte_size))
    return tuple(result)


def describe_colibri_experts(
    model_path: Path,
    *,
    model_id: str,
    model_revision: str,
) -> tuple[str, tuple[ExpertDescriptor, ...]]:
    """Build exact expert descriptors from an immutable stage artifact."""

    root = model_path.expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Colibri stage configuration must be an object")
    adapter = default_colibri_adapter_registry().resolve_config(config)
    inventory = _tensor_inventory(root)
    architecture = architecture_from_config(config)
    descriptor = ResolvedModelDescriptor(
        model_id=model_id,
        revision=model_revision,
        content_fingerprint=f"artifact:{model_revision}",
        source_type="local",
        format="safetensors",
        architecture=architecture.canonical,
        architecture_raw=architecture.raw,
        architecture_source=architecture.source,
        files=(),
        weight_bytes=sum(item[3] for item in inventory),
        layer_count=int(config.get("num_hidden_layers") or 0),
        hidden_size=int(config.get("hidden_size") or 0) or None,
        configuration=config,
    )
    profile = adapter.inspect_model(descriptor)
    mappings = adapter.map_tensor_names(inventory, config=config)
    experts = tuple(
        item for item in adapter.describe_experts(mappings, profile) if item.expert_type == "routed"
    )
    if not experts:
        raise ValueError("Colibri adapter described no routed experts in the stage artifact")
    return adapter.adapter_id, experts


class ColibriMoeBackend:
    """Colocated expert execution with bounded VRAM/RAM residency."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_id: str,
        model_revision: str,
        selected_experts: set[tuple[int, int]],
        device: str | torch.device,
        cache_budget_bytes: int | None = None,
    ) -> None:
        self.root = model_path.expanduser().resolve()
        self.device = torch.device(device)
        self.adapter_id, described = describe_colibri_experts(
            self.root,
            model_id=model_id,
            model_revision=model_revision,
        )
        self.descriptors = {
            (item.layer_index, item.expert_index): item
            for item in described
            if (item.layer_index, item.expert_index) in selected_experts
        }
        if set(self.descriptors) != set(selected_experts):
            missing = sorted(set(selected_experts) - set(self.descriptors))
            raise ValueError(f"Colibri artifact omits delegated experts: {missing[:3]}")
        raw_index = json.loads(
            (self.root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.weight_map = {
            str(name): str(path) for name, path in raw_index["weight_map"].items()
        }
        if cache_budget_bytes is None:
            if self.device.type == "cuda" and torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                cache_budget_bytes = max(256 * 1024**2, int(free_bytes * 0.65))
            else:
                cache_budget_bytes = 8 * 1024**3
        if cache_budget_bytes <= 0:
            raise ValueError("Colibri cache budget must be positive")
        self.cache_budget_bytes = int(cache_budget_bytes)
        self.cache: OrderedDict[tuple[int, int], _ResidentExpert] = OrderedDict()
        self.cache_bytes = 0
        self.peak_cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.bytes_read = 0
        self.compute_ns = 0
        self.calls = 0
        self._sessions: set[str] = set()
        self._closed = False

    def capabilities(self) -> MoeBackendCapabilities:
        return MoeBackendCapabilities(
            backend=f"colibri:{self.adapter_id}",
            whole_expert=True,
            native_microshard=False,
            local=True,
            exact=True,
        )

    def open_session(self, session_id: str) -> None:
        if self._closed:
            raise RuntimeError("Colibri expert backend is closed")
        if not session_id or session_id in self._sessions:
            raise ValueError("Colibri session identity is empty or already active")
        self._sessions.add(session_id)

    def close_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise ValueError("Colibri session is not active")
        self._sessions.remove(session_id)

    def cancel_session(self, session_id: str) -> None:
        self.close_session(session_id)

    def close(self) -> None:
        self._sessions.clear()
        self.cache.clear()
        self.cache_bytes = 0
        self._closed = True

    @staticmethod
    def _linear(source: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 2:
            raise ValueError("Colibri projection weight must be rank two")
        if weight.shape[1] == source.shape[-1]:
            return torch.nn.functional.linear(source, weight)
        if weight.shape[0] == source.shape[-1]:
            return source @ weight
        raise ValueError("Colibri projection does not consume the activation width")

    def _load(self, key: tuple[int, int]) -> _ResidentExpert:
        descriptor = self.descriptors[key]
        tensors: dict[str, torch.Tensor] = {}
        roles: dict[str, str] = {}
        byte_size = 0
        for group in descriptor.tensor_groups:
            for name, role in zip(group.tensor_names, group.tensor_roles, strict=True):
                if role.endswith(("_scale", "_shape")):
                    raise ValueError(
                        "the PyTorch Colibri component requires floating checkpoint projections"
                    )
                filename = self.weight_map.get(name)
                if filename is None:
                    raise KeyError(f"Colibri artifact has no tensor {name}")
                with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
                    tensor = handle.get_tensor(name).contiguous()
                slices = descriptor.routing_metadata.get("tensor_slices", {})
                selection = slices.get(name) if isinstance(slices, dict) else None
                if isinstance(selection, dict):
                    tensor = tensor.select(
                        int(selection["axis"]), int(selection["index"])
                    ).contiguous()
                if not tensor.is_floating_point():
                    raise ValueError("Colibri PyTorch component cannot reinterpret packed weights")
                byte_size += tensor.numel() * tensor.element_size()
                tensors[name] = tensor.to(device=self.device, non_blocking=False)
                roles[name] = role
        if byte_size > self.cache_budget_bytes:
            raise MemoryError("one Colibri expert exceeds the configured residency budget")
        while self.cache and self.cache_bytes + byte_size > self.cache_budget_bytes:
            _, evicted = self.cache.popitem(last=False)
            self.cache_bytes -= evicted.byte_size
        resident = _ResidentExpert(tensors=tensors, roles=roles, byte_size=byte_size)
        self.cache[key] = resident
        self.cache_bytes += byte_size
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)
        self.bytes_read += byte_size
        return resident

    def _get(self, key: tuple[int, int]) -> _ResidentExpert:
        resident = self.cache.pop(key, None)
        if resident is None:
            self.cache_misses += 1
            return self._load(key)
        self.cache_hits += 1
        self.cache[key] = resident
        return resident

    @staticmethod
    def _projection(resident: _ResidentExpert, marker: str) -> torch.Tensor | None:
        return next(
            (
                tensor
                for name, tensor in resident.tensors.items()
                if marker in resident.roles[name].casefold()
                and "scale" not in resident.roles[name].casefold()
                and "shape" not in resident.roles[name].casefold()
            ),
            None,
        )

    @torch.inference_mode()
    def execute_expert_rows(
        self,
        *,
        layer_id: int,
        expert_id: int,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[str, ...], dict[str, Any]]:
        started = time.perf_counter_ns()
        resident = self._get((layer_id, expert_id))
        fused = self._projection(resident, "gate+up")
        gate = self._projection(resident, "gate")
        up = self._projection(resident, "up")
        down = self._projection(resident, "down")
        if down is None:
            raise ValueError("Colibri expert has no down projection")
        compute_dtype = down.dtype
        source = activation.to(device=self.device, dtype=compute_dtype)
        if fused is not None:
            projected = self._linear(source, fused)
            if projected.shape[-1] % 2:
                raise ValueError("Colibri fused gate/up width is not even")
            gate_values, up_values = projected.chunk(2, dim=-1)
        else:
            if gate is None or up is None:
                raise ValueError("Colibri expert lacks gate/up projections")
            gate_values = self._linear(source, gate)
            up_values = self._linear(source, up)
        output = self._linear(torch.nn.functional.silu(gate_values) * up_values, down)
        output = output.to(device=activation.device, dtype=activation.dtype)
        elapsed = time.perf_counter_ns() - started
        self.calls += 1
        self.compute_ns += elapsed
        return output, (), {
            "colibri_compute_ns": elapsed,
            "colibri_cache_hits": self.cache_hits,
            "colibri_cache_misses": self.cache_misses,
            "colibri_expert_movement_bytes": self.bytes_read,
        }

    def status(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "colibri_adapter_id": self.adapter_id,
            "colibri_owned_experts": [list(item) for item in sorted(self.descriptors)],
            "colibri_cache_resident_bytes": self.cache_bytes,
            "colibri_cache_budget_bytes": self.cache_budget_bytes,
            "colibri_peak_cache_bytes": self.peak_cache_bytes,
            "colibri_cache_hits": self.cache_hits,
            "colibri_cache_misses": self.cache_misses,
            "colibri_cache_hit_rate": self.cache_hits / total if total else 0.0,
            "colibri_expert_movement_bytes": self.bytes_read,
            "colibri_expert_calls": self.calls,
            "colibri_compute_ns": self.compute_ns,
            "coordinator_activation_bytes": 0,
        }


__all__ = ["ColibriMoeBackend", "describe_colibri_experts"]
