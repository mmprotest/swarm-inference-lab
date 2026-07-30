"""Safetensors-index inspection and contiguous Qwen3 shard construction."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.config.models import (
    AttentionConfig,
    Backend,
    CacheSpec,
    ModelManifest,
    StageDefinition,
    TensorSpec,
)
from swarm_inference.exceptions import IntegrityError, MemoryLimitExceededError
from swarm_inference.model.adapter import ComponentKind, ModelDescription, TensorInfo
from swarm_inference.model.manifest import hash_shard_directory, save_manifest
from swarm_inference.model.qwen3 import Qwen3Adapter


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    model_id: str
    revision: str
    path: Path
    downloaded: bool


def resolve_model(
    model: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
) -> ResolvedModel:
    local = Path(model).expanduser()
    if local.exists():
        resolved = local.resolve()
        if not resolved.is_dir():
            raise IntegrityError(f"model path is not a directory: {resolved}")
        return ResolvedModel(
            model_id=model,
            revision=revision or "local-unversioned",
            path=resolved,
            downloaded=False,
        )
    if not allow_download:
        raise IntegrityError(f"model {model!r} is not local and downloads are disabled")
    from huggingface_hub import model_info, snapshot_download

    info = model_info(model, revision=revision)
    exact_revision = info.sha
    if exact_revision is None:
        raise IntegrityError(f"model registry did not resolve an immutable revision for {model!r}")
    path = snapshot_download(
        repo_id=model,
        revision=exact_revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges*",
        ],
    )
    return ResolvedModel(
        model_id=model,
        revision=exact_revision,
        path=Path(path).resolve(),
        downloaded=True,
    )


def inspect_qwen3_model(resolved: ResolvedModel) -> ModelDescription:
    return Qwen3Adapter().describe(
        resolved.path,
        model_id=resolved.model_id,
        model_revision=resolved.revision,
    )


def model_inspection_payload(description: ModelDescription) -> dict[str, Any]:
    """Build the reader-facing, header-only Qwen3 inspection evidence."""

    config = description.config
    layer_count = int(config["num_hidden_layers"])
    per_layer = [0] * layer_count
    for tensor in description.tensors:
        if tensor.component.kind == ComponentKind.DECODER_LAYER:
            assert tensor.component.layer_index is not None
            per_layer[tensor.component.layer_index] += tensor.bytes
    dtypes = sorted({tensor.dtype for tensor in description.tensors})
    if len(dtypes) != 1:
        raise IntegrityError(f"mixed source weight dtypes are unsupported: {dtypes}")
    dtype_bytes = _DTYPE_WIDTHS.get(dtypes[0])
    if dtype_bytes is None:
        raise IntegrityError(f"unsupported source weight dtype {dtypes[0]}")
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or heads)
    head_dim = int(config.get("head_dim") or hidden // heads)
    embedding = _component_bytes(description.tensors, ComponentKind.EMBEDDING)
    explicit_head = _component_bytes(description.tensors, ComponentKind.OUTPUT_HEAD)
    tied = bool(config.get("tie_word_embeddings", False))
    output_head = explicit_head or (embedding if tied else 0)
    total = sum(tensor.bytes for tensor in description.tensors)
    activation_per_token = hidden * dtype_bytes
    cache_per_token_layer = 2 * kv_heads * head_dim * dtype_bytes
    return {
        "model_id": description.model_id,
        "requested_model_id": description.model_id,
        "resolved_revision": description.model_revision,
        "local_snapshot_path": str(description.model_path),
        "architecture": (config.get("architectures") or [None])[0],
        "model_type": config.get("model_type"),
        "decoder_layer_count": layer_count,
        "hidden_size": hidden,
        "intermediate_size": int(config["intermediate_size"]),
        "attention_head_count": heads,
        "key_value_head_count": kv_heads,
        "head_dimension": head_dim,
        "vocabulary_size": int(config["vocab_size"]),
        "maximum_position_embeddings": int(config["max_position_embeddings"]),
        "rope": {
            "theta": config.get("rope_theta"),
            "scaling": config.get("rope_scaling"),
        },
        "weight_dtype": dtypes[0],
        "tied_embeddings": tied,
        "tensor_count": len(description.tensors),
        "parameter_count": sum(math.prod(tensor.shape) for tensor in description.tensors),
        "total_source_weight_bytes": total,
        "embedding_bytes": embedding,
        "output_head_bytes": output_head,
        "explicit_output_head_bytes": explicit_head,
        "final_normalisation_bytes": _component_bytes(
            description.tensors, ComponentKind.FINAL_NORM
        ),
        "per_layer_bytes": per_layer,
        "largest_layer": {
            "layer_index": max(range(layer_count), key=per_layer.__getitem__),
            "bytes": max(per_layer),
        },
        "estimated_activation_bytes": {
            "per_token_at_each_stage_boundary": activation_per_token,
            "sequence_1": activation_per_token,
            "sequence_128": activation_per_token * 128,
            "sequence_512": activation_per_token * 512,
        },
        "estimated_kv_cache_bytes_per_token_per_layer": cache_per_token_layer,
        "transformers_version_requirement": config.get("transformers_version"),
        "config_hashes": description.config_file_hashes,
        "tokenizer_hashes": description.tokenizer_file_hashes,
        "safetensors_hashes": description.source_file_hashes,
        "full_model_instantiated": False,
        "gpu_weights_loaded": False,
    }


_DTYPE_WIDTHS = {
    "F16": 2,
    "BF16": 2,
    "F32": 4,
}


def _component_bytes(
    tensors: list[TensorInfo],
    kind: ComponentKind,
) -> int:
    return sum(item.bytes for item in tensors if item.component.kind == kind)


def _partition_ranges(
    *,
    layer_bytes: list[int],
    embedding_bytes: int,
    final_bytes: int,
    target_stage_bytes: int,
    maximum_stage_bytes: int,
) -> list[tuple[int, int]]:
    layer_count = len(layer_bytes)
    if not layer_count:
        raise IntegrityError("model has no decoder layers")
    if target_stage_bytes <= 0 or maximum_stage_bytes <= 0:
        raise ValueError("stage byte targets must be positive")
    if target_stage_bytes > maximum_stage_bytes:
        target_stage_bytes = maximum_stage_bytes
    if layer_count == 1:
        required = embedding_bytes + layer_bytes[0] + final_bytes
        if required > maximum_stage_bytes:
            raise MemoryLimitExceededError(
                f"single-layer model stage requires {required} bytes, cap is {maximum_stage_bytes}"
            )
        return [(0, 1)]
    if final_bytes + layer_bytes[-1] > maximum_stage_bytes:
        raise MemoryLimitExceededError(
            "output head, final norm, and final decoder layer require "
            f"{final_bytes + layer_bytes[-1]} bytes, cap is {maximum_stage_bytes}"
        )
    final_start = layer_count - 1
    final_total = final_bytes + layer_bytes[-1]
    while final_start > 0:
        candidate = final_total + layer_bytes[final_start - 1]
        combined_embedding = embedding_bytes if final_start - 1 == 0 else 0
        if (
            candidate <= target_stage_bytes
            and candidate + combined_embedding <= maximum_stage_bytes
        ):
            final_start -= 1
            final_total = candidate
        else:
            break
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while cursor < final_start:
        overhead = embedding_bytes if cursor == 0 else 0
        end = cursor
        total = overhead
        while end < final_start:
            candidate = total + layer_bytes[end]
            if candidate > maximum_stage_bytes:
                break
            if end > cursor and candidate > target_stage_bytes:
                break
            total = candidate
            end += 1
        if end == cursor:
            raise MemoryLimitExceededError(
                f"embedding/layer {cursor} requires {overhead + layer_bytes[cursor]} "
                f"bytes, cap is {maximum_stage_bytes}"
            )
        ranges.append((cursor, end))
        cursor = end
    ranges.append((final_start, layer_count))
    return ranges


def _partition_exact_stage_count(
    *,
    layer_bytes: list[int],
    embedding_bytes: int,
    final_bytes: int,
    stage_count: int,
    maximum_stage_bytes: int,
) -> list[tuple[int, int]]:
    """Return exactly ``stage_count`` balanced contiguous ranges.

    The minimum possible maximum stage weight is found first.  Among all
    partitions at that cap, the dynamic program minimises squared distance
    from the ideal byte count.  Endpoint ownership is part of each candidate's
    actual byte cost, so tied-output duplication is never hidden.
    """

    layer_count = len(layer_bytes)
    if not 1 <= stage_count <= layer_count:
        raise ValueError(f"stage_count must be between 1 and the {layer_count} decoder layers")
    if maximum_stage_bytes <= 0:
        raise ValueError("maximum_stage_bytes must be positive")
    prefix = [0]
    for value in layer_bytes:
        prefix.append(prefix[-1] + value)

    def stage_cost(stage_id: int, start: int, end: int) -> int:
        overhead = 0
        if stage_id == 0:
            overhead += embedding_bytes
        if stage_id == stage_count - 1:
            overhead += final_bytes
        return prefix[end] - prefix[start] + overhead

    def feasible(cap: int) -> bool:
        positions = {0}
        for stage_id in range(stage_count):
            next_positions: set[int] = set()
            remaining = stage_count - stage_id - 1
            for start in positions:
                maximum_end = layer_count - remaining
                for end in range(start + 1, maximum_end + 1):
                    if stage_cost(stage_id, start, end) <= cap:
                        next_positions.add(end)
                    else:
                        break
            positions = next_positions
            if not positions:
                return False
        return layer_count in positions

    lower = max(
        max(layer_bytes),
        embedding_bytes + layer_bytes[0],
        final_bytes + layer_bytes[-1],
    )
    upper = sum(layer_bytes) + embedding_bytes + final_bytes
    if not feasible(maximum_stage_bytes):
        raise MemoryLimitExceededError(
            f"no contiguous {stage_count}-stage partition fits the "
            f"{maximum_stage_bytes}-byte stage cap"
        )
    upper = min(upper, maximum_stage_bytes)
    while lower < upper:
        middle = (lower + upper) // 2
        if feasible(middle):
            upper = middle
        else:
            lower = middle + 1
    optimal_cap = lower
    ideal = (sum(layer_bytes) + embedding_bytes + final_bytes) / stage_count
    states: dict[int, tuple[float, list[tuple[int, int]]]] = {
        0: (0.0, []),
    }
    for stage_id in range(stage_count):
        next_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        remaining = stage_count - stage_id - 1
        for start, (score, ranges) in states.items():
            maximum_end = layer_count - remaining
            for end in range(start + 1, maximum_end + 1):
                cost = stage_cost(stage_id, start, end)
                if cost > optimal_cap:
                    break
                candidate = (score + (cost - ideal) ** 2, [*ranges, (start, end)])
                previous = next_states.get(end)
                if previous is None or candidate[0] < previous[0]:
                    next_states[end] = candidate
        states = next_states
    try:
        return states[layer_count][1]
    except KeyError as exc:  # pragma: no cover - guarded by feasible()
        raise IntegrityError("exact-stage partition search produced no complete result") from exc


def build_manifest(
    description: ModelDescription,
    *,
    target_stage_bytes: int,
    maximum_stage_bytes: int,
    stage_count: int | None = None,
) -> ModelManifest:
    config = description.config
    layer_count = int(config["num_hidden_layers"])
    per_layer = [0] * layer_count
    for tensor in description.tensors:
        if tensor.component.kind == ComponentKind.DECODER_LAYER:
            if tensor.component.layer_index is None:
                raise IntegrityError(f"layer tensor has no layer index: {tensor.name}")
            if not 0 <= tensor.component.layer_index < layer_count:
                raise IntegrityError(
                    f"tensor {tensor.name} maps outside configured layer count {layer_count}"
                )
            per_layer[tensor.component.layer_index] += tensor.bytes
    if any(value == 0 for value in per_layer):
        missing_layers = [index for index, value in enumerate(per_layer) if value == 0]
        raise IntegrityError(f"decoder layers have no mapped tensors: {missing_layers}")
    embedding = _component_bytes(description.tensors, ComponentKind.EMBEDDING)
    norm = _component_bytes(description.tensors, ComponentKind.FINAL_NORM)
    explicit_head = _component_bytes(description.tensors, ComponentKind.OUTPUT_HEAD)
    tied = bool(config.get("tie_word_embeddings", False))
    head = explicit_head or (embedding if tied else 0)
    if head == 0:
        raise IntegrityError("Qwen3 model has neither lm_head weights nor tied embeddings")
    if stage_count is None:
        ranges = _partition_ranges(
            layer_bytes=per_layer,
            embedding_bytes=embedding,
            final_bytes=norm + head,
            target_stage_bytes=target_stage_bytes,
            maximum_stage_bytes=maximum_stage_bytes,
        )
    else:
        ranges = _partition_exact_stage_count(
            layer_bytes=per_layer,
            embedding_bytes=embedding,
            final_bytes=norm + head,
            stage_count=stage_count,
            maximum_stage_bytes=maximum_stage_bytes,
        )
    dtype = description.tensors[0].dtype
    hidden = int(config["hidden_size"])
    dtype_bytes = {
        "F16": 2,
        "BF16": 2,
        "F32": 4,
    }.get(dtype)
    if dtype_bytes is None:
        raise IntegrityError(f"unsupported Qwen3 manifest weight dtype {dtype}")
    stages: list[StageDefinition] = []
    shared_tensors: dict[str, list[int]] = {}
    for stage_id, (start, end) in enumerate(ranges):
        first = stage_id == 0
        last = stage_id == len(ranges) - 1
        names = [
            tensor.name
            for tensor in description.tensors
            if (
                (first and tensor.component.kind == ComponentKind.EMBEDDING)
                or (
                    tensor.component.kind == ComponentKind.DECODER_LAYER
                    and tensor.component.layer_index is not None
                    and start <= tensor.component.layer_index < end
                )
                or (last and tensor.component.kind == ComponentKind.FINAL_NORM)
                or (last and tensor.component.kind == ComponentKind.OUTPUT_HEAD)
            )
        ]
        if last and tied and explicit_head == 0:
            tied_names = [
                tensor.name
                for tensor in description.tensors
                if tensor.component.kind == ComponentKind.EMBEDDING
            ]
            names.extend(tied_names)
            for name in tied_names:
                shared_tensors[name] = sorted({0, stage_id})
        required = sum(tensor.bytes for tensor in description.tensors if tensor.name in set(names))
        if required > maximum_stage_bytes:
            raise MemoryLimitExceededError(
                f"stage {stage_id} requires {required} bytes, cap is {maximum_stage_bytes}"
            )
        stages.append(
            StageDefinition(
                stage_id=stage_id,
                layer_start=start,
                layer_end=end,
                owns_embeddings=first,
                owns_final_norm=last,
                owns_output_head=last,
                required_memory_bytes=required,
                estimated_execution_ms={},
                input_spec=TensorSpec(
                    dtype="int64" if first else dtype.lower(),
                    shape=["batch", "sequence"]
                    if first
                    else [
                        "batch",
                        "sequence",
                        hidden,
                    ],
                ),
                output_spec=TensorSpec(
                    dtype=dtype.lower(),
                    shape=(
                        ["batch", "sequence", int(config["vocab_size"])]
                        if last
                        else ["batch", "sequence", hidden]
                    ),
                ),
                cache_spec=CacheSpec(
                    bytes_per_token=(
                        2
                        * int(config.get("num_key_value_heads") or config["num_attention_heads"])
                        * int(
                            config.get("head_dim") or hidden // int(config["num_attention_heads"])
                        )
                        * dtype_bytes
                        * (end - start)
                    )
                ),
                tensor_names=sorted(set(names)),
                tensor_count=len(set(names)),
                required_total_memory_bytes=(
                    required
                    + 128 * 1024 * 1024
                    + (
                        2
                        * int(config.get("num_key_value_heads") or config["num_attention_heads"])
                        * int(
                            config.get("head_dim") or hidden // int(config["num_attention_heads"])
                        )
                        * dtype_bytes
                        * (end - start)
                        * 2048
                    )
                ),
            )
        )
    all_stage_names = [name for stage in stages for name in stage.tensor_names]
    counts = Counter(all_stage_names)
    source_names = {tensor.name for tensor in description.tensors}
    if set(counts) != source_names:
        missing_tensors = sorted(source_names - set(counts))
        extra = sorted(set(counts) - source_names)
        raise IntegrityError(
            f"stage tensor union mismatch; missing={missing_tensors}, extra={extra}"
        )
    illegal_duplicates = sorted(
        name for name, count in counts.items() if count > 1 and name not in shared_tensors
    )
    if illegal_duplicates:
        raise IntegrityError(
            f"tensors assigned to multiple stages without shared declaration: {illegal_duplicates}"
        )
    architecture = (config.get("architectures") or ["Qwen3ForCausalLM"])[0]
    total_source_bytes = sum(tensor.bytes for tensor in description.tensors)
    tensor_to_stages = {
        name: [stage.stage_id for stage in stages if name in stage.tensor_names]
        for name in sorted(source_names)
    }
    total_sharded_bytes = sum(stage.required_memory_bytes for stage in stages)
    return ModelManifest(
        model_id=description.model_id,
        model_revision=description.model_revision,
        architecture=str(architecture),
        tokenizer_id=description.model_id,
        layer_count=layer_count,
        hidden_size=hidden,
        attention=AttentionConfig(
            head_count=int(config["num_attention_heads"]),
            key_value_head_count=int(
                config.get("num_key_value_heads") or config["num_attention_heads"]
            ),
            head_dimension=int(
                config.get("head_dim") or hidden // int(config["num_attention_heads"])
            ),
            rope_theta=float(
                config.get("rope_theta")
                or (config.get("rope_scaling") or {}).get("rope_theta")
                or (config.get("rope_parameters") or {}).get("rope_theta")
                or 1_000_000
            ),
            sliding_window=(
                int(config["sliding_window"])
                if config.get("use_sliding_window") and config.get("sliding_window")
                else None
            ),
        ),
        vocabulary_size=int(config["vocab_size"]),
        weight_dtype=dtype,
        quantisation_format=(
            str(config.get("quantization_config", {}).get("quant_method"))
            if config.get("quantization_config")
            else None
        ),
        total_weight_bytes=total_source_bytes,
        embedding_bytes=embedding,
        output_head_bytes=head,
        per_layer_weight_bytes=per_layer,
        estimated_cache_bytes_per_token_per_layer=(
            2
            * int(config.get("num_key_value_heads") or config["num_attention_heads"])
            * int(config.get("head_dim") or hidden // int(config["num_attention_heads"]))
            * dtype_bytes
        ),
        activation_bytes_per_stage_boundary=hidden * dtype_bytes,
        stages=stages,
        shard_hashes={},
        compatible_worker_backends=[
            Backend.TORCH_CPU,
            Backend.TORCH_CUDA,
            Backend.TORCH_MPS,
        ],
        shared_tensors=shared_tensors,
        source_files=description.source_file_hashes,
        config_files=description.config_file_hashes,
        tokenizer_files=description.tokenizer_file_hashes,
        total_sharded_weight_bytes=total_sharded_bytes,
        duplicated_tensor_bytes=total_sharded_bytes - total_source_bytes,
        duplicated_tensors=shared_tensors,
        tensor_to_stages=tensor_to_stages,
        final_normalisation_bytes=norm,
        embedding_owner=0,
        final_normalisation_owner=len(stages) - 1,
        lm_head_owner=len(stages) - 1,
        tied_weight_treatment=(
            "copied-to-embedding-and-lm-head-owner"
            if tied and explicit_head == 0
            else (
                "source-checkpoint-stores-separate-embedding-and-lm-head-tensors"
                if tied
                else "explicit-lm-head"
            )
        ),
        transformers_version_requirement=(
            str(config["transformers_version"]) if config.get("transformers_version") else None
        ),
        supported_dtypes=[dtype.lower()],
    )


def shard_model(
    description: ModelDescription,
    *,
    output: str | Path,
    target_stage_bytes: int,
    maximum_stage_bytes: int,
    maximum_output_file_bytes: int = 512 * 1024 * 1024,
    stage_count: int | None = None,
) -> ModelManifest:
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise IntegrityError(f"refusing to overwrite non-empty shard output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        description,
        target_stage_bytes=target_stage_bytes,
        maximum_stage_bytes=maximum_stage_bytes,
        stage_count=stage_count,
    )
    tensor_by_name = {tensor.name: tensor for tensor in description.tensors}
    from safetensors import safe_open
    from safetensors.torch import save_file

    for stage in manifest.stages:
        stage_dir = root / f"stage-{stage.stage_id:03d}"
        stage_dir.mkdir()
        part_index = 0
        pending: dict[str, Any] = {}
        pending_bytes = 0
        single_output_file = stage.required_memory_bytes <= maximum_output_file_bytes

        def flush(
            stage_dir: Path = stage_dir,
            single_output_file: bool = single_output_file,
        ) -> None:
            nonlocal part_index, pending, pending_bytes
            if not pending:
                return
            destination = (
                stage_dir / "weights.safetensors"
                if single_output_file
                else stage_dir / f"weights-{part_index:05d}.safetensors"
            )
            save_file(pending, destination)
            part_index += 1
            pending = {}
            pending_bytes = 0

        grouped: dict[str, list[str]] = {}
        for name in stage.tensor_names:
            grouped.setdefault(tensor_by_name[name].source_file, []).append(name)
        for source_file, names in sorted(grouped.items()):
            with safe_open(
                description.model_path / source_file,
                framework="pt",
                device="cpu",
            ) as handle:
                for name in sorted(names):
                    tensor_info = tensor_by_name[name]
                    if pending and pending_bytes + tensor_info.bytes > maximum_output_file_bytes:
                        flush()
                    pending[name] = handle.get_tensor(name)
                    pending_bytes += tensor_info.bytes
                    if pending_bytes >= maximum_output_file_bytes:
                        flush()
        flush()
        (stage_dir / "stage.json").write_text(
            json.dumps(
                {
                    **stage.model_dump(mode="json"),
                    "logical_weight_bytes": stage.required_memory_bytes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _verify_generated_union(root, manifest, description)
    shard_hashes = {
        f"stage-{stage.stage_id:03d}": hash_shard_directory(root / f"stage-{stage.stage_id:03d}")
        for stage in manifest.stages
    }
    manifest.shard_hashes = shard_hashes
    manifest.stages = [
        stage.model_copy(update={"shard_hash": shard_hashes[f"stage-{stage.stage_id:03d}"]})
        for stage in manifest.stages
    ]
    config_dir = root / "config"
    tokenizer_dir = root / "tokenizer"
    config_dir.mkdir()
    tokenizer_dir.mkdir()
    for name in manifest.config_files:
        shutil.copy2(description.model_path / name, config_dir / name)
    for name in manifest.tokenizer_files:
        shutil.copy2(description.model_path / name, tokenizer_dir / name)
    save_manifest(manifest, root / "manifest.json")
    (root / "source.json").write_text(
        json.dumps(
            {
                "model_id": description.model_id,
                "model_revision": description.model_revision,
                "source_path": str(description.model_path),
                "source_file_hashes": description.source_file_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "hashes.json").write_text(
        json.dumps(
            {
                "model_id": description.model_id,
                "resolved_revision": description.model_revision,
                "source_weight_files": description.source_file_hashes,
                "config_files": description.config_file_hashes,
                "tokenizer_files": description.tokenizer_file_hashes,
                "stage_shards": shard_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validation = {
        "status": "PASS",
        "every_source_tensor_assigned": True,
        "unsupported_tensors_ignored": [],
        "decoder_layers_owned_exactly_once": True,
        "stage_count": len(manifest.stages),
        "stage_hashes_valid": True,
        "union_reconstructs_required_state": True,
        "duplicated_tensors": manifest.duplicated_tensors,
        "duplicated_tensor_bytes": manifest.duplicated_tensor_bytes,
        "no_stage_contains_full_model": all(
            stage.required_memory_bytes < manifest.total_weight_bytes for stage in manifest.stages
        ),
        "full_source_weight_bytes": manifest.total_weight_bytes,
        "total_sharded_weight_bytes": manifest.total_sharded_weight_bytes,
    }
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _verify_generated_union(
    root: Path,
    manifest: ModelManifest,
    description: ModelDescription,
) -> None:
    from safetensors import safe_open

    counts: Counter[str] = Counter()
    for stage in manifest.stages:
        stage_dir = root / f"stage-{stage.stage_id:03d}"
        generated_names: set[str] = set()
        for file in sorted(stage_dir.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for name in handle.keys():  # noqa: SIM118
                    if name in generated_names:
                        raise IntegrityError(
                            f"generated stage {stage.stage_id} duplicates tensor {name}"
                        )
                    generated_names.add(name)
                    source = next(tensor for tensor in description.tensors if tensor.name == name)
                    value = handle.get_slice(name)
                    if tuple(value.get_shape()) != source.shape:
                        raise IntegrityError(f"generated tensor shape mismatch for {name}")
        if generated_names != set(stage.tensor_names):
            raise IntegrityError(f"generated stage {stage.stage_id} tensor set mismatch")
        counts.update(generated_names)
    expected = Counter(tensor.name for tensor in description.tensors)
    for name in manifest.shared_tensors:
        expected[name] = len(manifest.shared_tensors[name])
    if counts != expected:
        raise IntegrityError(
            f"generated shard union does not reconstruct expected state dictionary: "
            f"generated={counts - expected}, missing={expected - counts}"
        )
