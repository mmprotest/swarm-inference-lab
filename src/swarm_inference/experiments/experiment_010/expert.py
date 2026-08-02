"""Executable whole-expert and matched tensor-microshard kernels."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from swarm_inference.experiments.experiment_010.schemas import (
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ReductionMode,
)


def silu(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    sigmoid = np.empty_like(source)
    positive = source >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exp_negative = np.exp(source[~positive])
    sigmoid[~positive] = exp_negative / (1.0 + exp_negative)
    return source * sigmoid


@dataclass(frozen=True, slots=True)
class ExpertWeights:
    up: np.ndarray
    gate: np.ndarray
    down: np.ndarray
    content_hash: str
    native_format: str = "float32"
    scale_group_size: int | None = None
    hidden_offset: int = 0
    logical_intermediate_dimension: int | None = None

    def __post_init__(self) -> None:
        up = np.asarray(self.up)
        gate = np.asarray(self.gate)
        down = np.asarray(self.down)
        if up.ndim != 2 or gate.shape != up.shape:
            raise ValueError("up and gate projections must share [intermediate, latent] shape")
        if down.shape != (up.shape[1], up.shape[0]):
            raise ValueError("down projection must have [latent, intermediate] shape")
        logical = self.logical_intermediate_dimension or int(up.shape[0])
        if self.hidden_offset < 0 or self.hidden_offset + up.shape[0] > logical:
            raise ValueError("expert slice lies outside its logical intermediate dimension")
        if not self.content_hash:
            raise ValueError("expert content hash is required")

    @property
    def latent_dimension(self) -> int:
        return int(self.up.shape[1])

    @property
    def intermediate_dimension(self) -> int:
        return int(self.up.shape[0])

    @property
    def logical_width(self) -> int:
        return self.logical_intermediate_dimension or self.intermediate_dimension

    @property
    def byte_size(self) -> int:
        return int(self.up.nbytes + self.gate.nbytes + self.down.nbytes)


class ExpertLoader(Protocol):
    def __call__(self, layer_id: int, expert_id: int) -> ExpertWeights: ...


def expert_content_hash(up: np.ndarray, gate: np.ndarray, down: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, tensor in (("up", up), ("gate", gate), ("down", down)):
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(1, "big"))
        digest.update(encoded_name)
        contiguous = np.ascontiguousarray(tensor)
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def deterministic_expert(
    *,
    latent_dimension: int,
    intermediate_dimension: int,
    seed: int,
) -> ExpertWeights:
    generator = np.random.default_rng(seed)
    scale = 0.025
    up = generator.normal(scale=scale, size=(intermediate_dimension, latent_dimension)).astype(
        np.float32
    )
    gate = generator.normal(scale=scale, size=(intermediate_dimension, latent_dimension)).astype(
        np.float32
    )
    down = generator.normal(scale=scale, size=(latent_dimension, intermediate_dimension)).astype(
        np.float32
    )
    return ExpertWeights(
        up=up,
        gate=gate,
        down=down,
        content_hash=expert_content_hash(up, gate, down),
    )


def slice_expert_weights(
    weights: ExpertWeights,
    *,
    hidden_start: int,
    hidden_end: int,
) -> ExpertWeights:
    """Create a physical matched slice without retaining the parent arrays."""

    if hidden_start < 0 or hidden_end <= hidden_start or hidden_end > weights.logical_width:
        raise ValueError("expert slice range is invalid")
    if weights.hidden_offset != 0 or weights.intermediate_dimension != weights.logical_width:
        raise ValueError("slice creation requires an unsliced source expert")
    if weights.scale_group_size is not None and (
        hidden_start % weights.scale_group_size or hidden_end % weights.scale_group_size
    ):
        raise ValueError("expert slice splits a native quantisation group")
    up = np.ascontiguousarray(weights.up[hidden_start:hidden_end]).copy()
    gate = np.ascontiguousarray(weights.gate[hidden_start:hidden_end]).copy()
    down = np.ascontiguousarray(weights.down[:, hidden_start:hidden_end]).copy()
    return ExpertWeights(
        up=up,
        gate=gate,
        down=down,
        content_hash=expert_content_hash(up, gate, down),
        native_format=weights.native_format,
        scale_group_size=weights.scale_group_size,
        hidden_offset=hidden_start,
        logical_intermediate_dimension=weights.logical_width,
    )


def execute_expert(
    activation: np.ndarray,
    weights: ExpertWeights,
    *,
    hidden_start: int | None = None,
    hidden_end: int | None = None,
) -> np.ndarray:
    source = np.ascontiguousarray(activation, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != weights.latent_dimension:
        raise ValueError("expert activation must be [batch_rows, latent_dimension]")
    if hidden_start is None:
        if weights.hidden_offset != 0 or weights.intermediate_dimension != weights.logical_width:
            raise ValueError("whole-expert execution cannot use sliced expert weights")
        global_start, global_end = 0, weights.logical_width
    else:
        global_start = hidden_start
        global_end = weights.logical_width if hidden_end is None else hidden_end
    if global_start < 0 or global_end <= global_start or global_end > weights.logical_width:
        raise ValueError("expert intermediate slice is out of bounds")
    start = global_start - weights.hidden_offset
    end = global_end - weights.hidden_offset
    if start < 0 or end > weights.intermediate_dimension:
        raise ValueError("requested microshard is not resident in this worker")
    if weights.scale_group_size is not None:
        if global_start % weights.scale_group_size:
            raise ValueError("microshard start splits a native quantisation group")
        if global_end != weights.logical_width and global_end % weights.scale_group_size:
            raise ValueError("microshard end splits a native quantisation group")
    up = source @ np.asarray(weights.up[start:end], dtype=np.float32).T
    gate = source @ np.asarray(weights.gate[start:end], dtype=np.float32).T
    activated = silu(gate) * up
    return np.ascontiguousarray(
        activated @ np.asarray(weights.down[:, start:end], dtype=np.float32).T,
        dtype=np.float32,
    )


def reduce_partials(
    partials: list[tuple[str, np.ndarray]],
    *,
    mode: ReductionMode | str,
) -> np.ndarray:
    if not partials:
        raise ValueError("at least one partial result is required")
    selected = ReductionMode(mode)
    ordered = sorted(partials, key=lambda item: item[0])
    if selected == ReductionMode.FIXED_ORDER_FP32:
        result = np.zeros_like(ordered[0][1], dtype=np.float32)
        for _, partial in ordered:
            result += np.asarray(partial, dtype=np.float32)
        return result
    if selected == ReductionMode.TREE_FP32:
        level = [np.asarray(item[1], dtype=np.float32) for item in ordered]
        while len(level) > 1:
            following = []
            for index in range(0, len(level), 2):
                following.append(
                    level[index] + level[index + 1] if index + 1 < len(level) else level[index]
                )
            level = following
        return level[0]
    return np.sum(np.stack([item[1] for item in partials]), axis=0)


class ExpertStore:
    """Budgeted per-process expert ownership and independent LRU cache."""

    def __init__(
        self,
        *,
        owned: set[tuple[int, int]],
        loader: ExpertLoader,
        residency_budget_bytes: int,
        cache_budget_bytes: int,
    ) -> None:
        if residency_budget_bytes <= 0 or cache_budget_bytes < 0:
            raise ValueError("expert budgets must be positive/non-negative")
        self.owned = set(owned)
        self.loader = loader
        self.residency_budget_bytes = residency_budget_bytes
        self.cache_budget_bytes = min(cache_budget_bytes, residency_budget_bytes)
        self.cache: OrderedDict[tuple[int, int], ExpertWeights] = OrderedDict()
        self.cache_bytes = 0
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        self.peak_resident_bytes = 0

    def get(self, layer_id: int, expert_id: int) -> ExpertWeights:
        key = (layer_id, expert_id)
        if key not in self.owned:
            raise KeyError(f"worker does not own layer {layer_id} expert {expert_id}")
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.hits += 1
            self.cache[key] = cached
            return cached
        self.misses += 1
        loaded = self.loader(layer_id, expert_id)
        if loaded.byte_size > self.residency_budget_bytes:
            raise MemoryError("one expert exceeds the worker residency budget")
        self.bytes_read += loaded.byte_size
        if self.cache_budget_bytes == 0:
            return loaded
        while self.cache and self.cache_bytes + loaded.byte_size > self.cache_budget_bytes:
            _, evicted = self.cache.popitem(last=False)
            self.cache_bytes -= evicted.byte_size
        if loaded.byte_size <= self.cache_budget_bytes:
            self.cache[key] = loaded
            self.cache_bytes += loaded.byte_size
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.cache_bytes)
        return loaded

    def drop_cache(self) -> None:
        self.cache.clear()
        self.cache_bytes = 0

    def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter_ns()
        before_hits, before_misses, before_read = self.hits, self.misses, self.bytes_read
        output = np.zeros((request.batch_rows, request.latent_dimension), dtype=np.float32)
        executed = []
        for expert_id, routing_weight in zip(
            request.expert_ids, request.routing_weights, strict=True
        ):
            weights = self.get(request.layer_id, expert_id)
            if weights.latent_dimension != request.latent_dimension:
                raise ValueError("worker expert geometry does not match request")
            partial = execute_expert(
                activation,
                weights,
                hidden_start=(
                    request.hidden_start
                    if request.execution_mode == ExpertExecutionMode.MICROSHARD
                    else None
                ),
                hidden_end=(
                    request.hidden_end
                    if request.execution_mode == ExpertExecutionMode.MICROSHARD
                    else None
                ),
            )
            output += np.float32(routing_weight) * partial
            executed.append(expert_id)
        return output, {
            "experts_executed": executed,
            "bytes_read": self.bytes_read - before_read,
            "cache_hits": self.hits - before_hits,
            "cache_misses": self.misses - before_misses,
            "compute_ns": time.perf_counter_ns() - started,
            "resident_tensor_bytes": self.cache_bytes,
            "expert_resident_bytes": self.cache_bytes,
        }


def safetensors_expert_loader(
    model_path: Path,
    *,
    tensor_file_by_name: dict[str, str] | None = None,
) -> ExpertLoader:
    """Return a lazy loader for unmodified Hugging Face expert tensors."""

    root = model_path.expanduser().resolve()
    default_file: str | None = None
    if tensor_file_by_name is None:
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            tensor_file_by_name = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        else:
            files = sorted(root.glob("*.safetensors"))
            if len(files) != 1:
                raise FileNotFoundError("model needs a safetensors index or one tensor file")
            tensor_file_by_name = {}
            default_file = files[0].name

    def load(layer_id: int, expert_id: int) -> ExpertWeights:
        import torch
        from safetensors import safe_open

        prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
        names = {
            "up": f"{prefix}.up_proj.weight",
            "gate": f"{prefix}.gate_proj.weight",
            "down": f"{prefix}.down_proj.weight",
        }
        arrays: dict[str, np.ndarray] = {}
        for projection, name in names.items():
            filename = tensor_file_by_name.get(name, default_file or "")
            if not filename:
                raise KeyError(f"model index has no tensor {name}")
            with safe_open(root / filename, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(name).to(dtype=torch.float32)
            arrays[projection] = tensor.numpy().copy()
        return ExpertWeights(
            up=arrays["up"],
            gate=arrays["gate"],
            down=arrays["down"],
            content_hash=expert_content_hash(arrays["up"], arrays["gate"], arrays["down"]),
            native_format="bfloat16",
        )

    return load


def safetensors_expert_ownership_entry(
    model_path: Path,
    *,
    layer_id: int,
    expert_id: int,
) -> dict[str, Any]:
    """Inventory the original expert tensors and their execution residency.

    The content hash is over the source dtype bytes.  ``logical_bytes`` is the
    larger FP32 execution residency used by the current NumPy worker and is
    therefore the value enforced by its logical budget.
    """

    import torch
    from safetensors import safe_open

    root = model_path.expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        default_file: str | None = None
    else:
        files = sorted(root.glob("*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError("model needs a safetensors index or one tensor file")
        weight_map = {}
        default_file = files[0].name
    prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
    names = [
        f"{prefix}.up_proj.weight",
        f"{prefix}.gate_proj.weight",
        f"{prefix}.down_proj.weight",
    ]
    digest = hashlib.sha256()
    native_bytes = 0
    execution_bytes = 0
    source_files = []
    dtypes = set()
    shapes = {}
    for name in names:
        filename = weight_map.get(name, default_file or "")
        if not filename:
            raise KeyError(f"model index has no tensor {name}")
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(2, "big"))
        digest.update(encoded_name)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        native_bytes += tensor.numel() * tensor.element_size()
        execution_bytes += tensor.numel() * np.dtype(np.float32).itemsize
        source_files.append(filename)
        dtypes.add(str(tensor.dtype))
        shapes[name.rsplit(".", 2)[-2]] = list(tensor.shape)
    return {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "path": str(root),
        "content_hash": "sha256:" + digest.hexdigest(),
        "logical_bytes": execution_bytes,
        "native_bytes": native_bytes,
        "native_dtypes": sorted(dtypes),
        "source_files": sorted(set(source_files)),
        "tensor_shapes": shapes,
        "persistent_execution_format": "fp32_numpy",
        "source_reencoded": True,
    }


def npz_expert_loader(files: dict[tuple[int, int], Path]) -> ExpertLoader:
    """Load only explicitly owned fixture files in the worker process."""

    resolved = {key: value.expanduser().resolve() for key, value in files.items()}

    def load(layer_id: int, expert_id: int) -> ExpertWeights:
        path = resolved.get((layer_id, expert_id))
        if path is None:
            raise KeyError(f"no owned fixture for layer {layer_id} expert {expert_id}")
        with np.load(path, allow_pickle=False) as archive:
            up = np.asarray(archive["up"], dtype=np.float32)
            gate = np.asarray(archive["gate"], dtype=np.float32)
            down = np.asarray(archive["down"], dtype=np.float32)
            hidden_offset = int(archive["hidden_start"]) if "hidden_start" in archive else 0
            logical_width = (
                int(archive["logical_intermediate_dimension"])
                if "logical_intermediate_dimension" in archive
                else int(up.shape[0])
            )
        return ExpertWeights(
            up=up,
            gate=gate,
            down=down,
            content_hash=expert_content_hash(up, gate, down),
            hidden_offset=hidden_offset,
            logical_intermediate_dimension=logical_width,
        )

    return load
