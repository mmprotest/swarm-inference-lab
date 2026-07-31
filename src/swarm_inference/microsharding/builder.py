"""Safetensors-native dense Qwen3 microshard construction and validation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from swarm_inference.exceptions import IntegrityError
from swarm_inference.microsharding.schemas import (
    ModelPartitionPlan,
    TensorShard,
    build_dense_partition_plan,
    validate_tensor_shard_union,
)
from swarm_inference.model.adapter import ComponentKind, ModelDescription, TensorInfo
from swarm_inference.model.shard_builder import inspect_qwen3_model, resolve_model
from swarm_inference.protocol.checksums import sha256_file

_DTYPE_WIDTHS = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}


@dataclass(frozen=True, slots=True)
class SliceAssignment:
    tensor: TensorInfo
    stage_id: int
    rank: int
    world_size: int
    axis: int | None
    start: int
    end: int
    replicated: bool
    partition_mode: str
    logical_rank_id: str


@dataclass(frozen=True, slots=True)
class MicroshardBuildResult:
    output: Path
    plan: ModelPartitionPlan
    validation: dict[str, Any]
    manifest: dict[str, Any]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_hash(value: Any) -> str:
    contiguous = value.detach().cpu().contiguous()
    import torch

    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _read_tensor(description: ModelDescription, tensor: TensorInfo) -> Any:
    with safe_open(
        description.model_path / tensor.source_file,
        framework="pt",
        device="cpu",
    ) as handle:
        return handle.get_tensor(tensor.name)


def _read_slice(description: ModelDescription, assignment: SliceAssignment) -> Any:
    with safe_open(
        description.model_path / assignment.tensor.source_file,
        framework="pt",
        device="cpu",
    ) as handle:
        source_slice = handle.get_slice(assignment.tensor.name)
        if assignment.axis is None:
            return source_slice[:]
        selection: list[slice] = [slice(None)] * len(assignment.tensor.shape)
        selection[assignment.axis] = slice(assignment.start, assignment.end)
        return source_slice[tuple(selection)]


def _stage_for_layer(plan: ModelPartitionPlan, layer_id: int) -> int:
    for stage in plan.pipeline_stages:
        if any(layer.layer_id == layer_id for layer in stage.layer_plans):
            return stage.stage_id
    raise IntegrityError(f"layer {layer_id} is absent from the partition plan")


def _rank_id(stage_id: int, rank: int) -> str:
    return f"stage-{stage_id:03d}-rank-{rank:03d}"


def _replicated_assignments(
    tensor: TensorInfo,
    *,
    stage_id: int,
    degree: int,
    mode: str,
) -> list[SliceAssignment]:
    return [
        SliceAssignment(
            tensor=tensor,
            stage_id=stage_id,
            rank=rank,
            world_size=degree,
            axis=None,
            start=0,
            end=tensor.shape[0],
            replicated=True,
            partition_mode=mode,
            logical_rank_id=_rank_id(stage_id, rank),
        )
        for rank in range(degree)
    ]


def _ranged_assignments(
    tensor: TensorInfo,
    *,
    stage_id: int,
    degree: int,
    axis: int,
    ranges: dict[int, tuple[int, int]],
    mode: str,
) -> list[SliceAssignment]:
    return [
        SliceAssignment(
            tensor=tensor,
            stage_id=stage_id,
            rank=rank,
            world_size=degree,
            axis=axis,
            start=ranges[rank][0],
            end=ranges[rank][1],
            replicated=False,
            partition_mode=mode,
            logical_rank_id=_rank_id(stage_id, rank),
        )
        for rank in range(degree)
    ]


def _head_ranges(
    ownership: dict[int, list[int]], head_dimension: int
) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for rank, heads in ownership.items():
        if not heads:
            raise IntegrityError(f"rank {rank} received no heads")
        if heads != list(range(heads[0], heads[-1] + 1)):
            raise IntegrityError(
                "the primary one-dimensional implementation requires contiguous heads"
            )
        result[rank] = (heads[0] * head_dimension, (heads[-1] + 1) * head_dimension)
    return result


def _assignment_key(item: SliceAssignment) -> tuple[int, int, str]:
    return item.stage_id, item.rank, item.tensor.name


def plan_tensor_assignments(
    description: ModelDescription,
    plan: ModelPartitionPlan,
) -> list[SliceAssignment]:
    """Map every source tensor to precise rank-local ownership."""

    degree = int(plan.metadata["tensor_parallel_degree"])
    assignments: list[SliceAssignment] = []
    tied = bool(description.config.get("tie_word_embeddings", False))
    explicit_output_head = any(
        item.component.kind == ComponentKind.OUTPUT_HEAD for item in description.tensors
    )
    first_stage = plan.pipeline_stages[0].stage_id
    final_stage = plan.pipeline_stages[-1].stage_id

    for tensor in description.tensors:
        name = tensor.name
        if tensor.component.kind == ComponentKind.EMBEDDING:
            stages_and_modes: list[tuple[int, str]] = [(first_stage, "vocabulary_embedding")]
            if tied and not explicit_output_head and final_stage != first_stage:
                stages_and_modes.append((final_stage, "vocabulary_lm_head"))
            if tied and not explicit_output_head and final_stage == first_stage:
                stages_and_modes = [(first_stage, "vocabulary_tied_embedding_lm_head")]
            for stage_id, mode in stages_and_modes:
                if plan.vocabulary_parallel:
                    from swarm_inference.microsharding.schemas import balanced_ranges

                    ranges = {
                        rank: interval
                        for rank, interval in enumerate(balanced_ranges(tensor.shape[0], degree))
                    }
                    assignments.extend(
                        _ranged_assignments(
                            tensor,
                            stage_id=stage_id,
                            degree=degree,
                            axis=0,
                            ranges=ranges,
                            mode=mode,
                        )
                    )
                else:
                    assignments.extend(
                        _replicated_assignments(
                            tensor,
                            stage_id=stage_id,
                            degree=degree,
                            mode=f"replicated_{mode}",
                        )
                    )
            continue
        if tensor.component.kind == ComponentKind.FINAL_NORM:
            assignments.extend(
                _replicated_assignments(
                    tensor,
                    stage_id=final_stage,
                    degree=degree,
                    mode="replicated_final_norm",
                )
            )
            continue
        if tensor.component.kind == ComponentKind.OUTPUT_HEAD:
            if plan.vocabulary_parallel:
                from swarm_inference.microsharding.schemas import balanced_ranges

                ranges = {
                    rank: interval
                    for rank, interval in enumerate(balanced_ranges(tensor.shape[0], degree))
                }
                assignments.extend(
                    _ranged_assignments(
                        tensor,
                        stage_id=final_stage,
                        degree=degree,
                        axis=0,
                        ranges=ranges,
                        mode="vocabulary_lm_head",
                    )
                )
            else:
                assignments.extend(
                    _replicated_assignments(
                        tensor,
                        stage_id=final_stage,
                        degree=degree,
                        mode="replicated_lm_head",
                    )
                )
            continue
        if tensor.component.kind != ComponentKind.DECODER_LAYER:
            raise IntegrityError(f"unhandled source component for {name}")
        layer_id = tensor.component.layer_index
        if layer_id is None:
            raise IntegrityError(f"decoder tensor {name} lacks a layer index")
        stage_id = _stage_for_layer(plan, layer_id)
        layer = next(
            item
            for stage in plan.pipeline_stages
            for item in stage.layer_plans
            if item.layer_id == layer_id
        )
        suffix = name.split(f"model.layers.{layer_id}.", maxsplit=1)[1]
        if suffix in {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.o_proj.bias",
            "mlp.down_proj.bias",
        }:
            assignments.extend(
                _replicated_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    mode="replicated_layer_parameter",
                )
            )
            continue
        attention = layer.attention
        assert layer.mlp is not None
        if suffix.startswith("self_attn.q_proj."):
            ranges = _head_ranges(attention.query_head_ownership, attention.head_dimension)
            assignments.extend(
                _ranged_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    axis=0,
                    ranges=ranges,
                    mode="column_parallel_query",
                )
            )
        elif suffix.startswith(("self_attn.k_proj.", "self_attn.v_proj.")):
            ranges = _head_ranges(attention.kv_head_ownership, attention.head_dimension)
            assignments.extend(
                _ranged_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    axis=0,
                    ranges=ranges,
                    mode=(
                        "kv_head_replication"
                        if attention.kv_replication_groups
                        else "column_parallel_kv"
                    ),
                )
            )
        elif suffix == "self_attn.o_proj.weight":
            ranges = _head_ranges(attention.query_head_ownership, attention.head_dimension)
            assignments.extend(
                _ranged_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    axis=1,
                    ranges=ranges,
                    mode="row_parallel_attention_output",
                )
            )
        elif suffix.startswith(("mlp.gate_proj.", "mlp.up_proj.")):
            assignments.extend(
                _ranged_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    axis=0,
                    ranges=layer.mlp.intermediate_ranges,
                    mode="column_parallel_mlp",
                )
            )
        elif suffix == "mlp.down_proj.weight":
            assignments.extend(
                _ranged_assignments(
                    tensor,
                    stage_id=stage_id,
                    degree=degree,
                    axis=1,
                    ranges=layer.mlp.intermediate_ranges,
                    mode="row_parallel_mlp_output",
                )
            )
        else:
            raise IntegrityError(f"unmapped Qwen3 layer tensor: {name}")

    deduplicated: dict[tuple[int, int, str], SliceAssignment] = {}
    for assignment in assignments:
        key = _assignment_key(assignment)
        previous = deduplicated.get(key)
        if previous is not None and previous != assignment:
            raise IntegrityError(f"conflicting assignments for {key}")
        deduplicated[key] = assignment
    mapped = {item.tensor.name for item in deduplicated.values()}
    expected = {item.name for item in description.tensors}
    if mapped != expected:
        raise IntegrityError(
            f"source tensor mapping mismatch: missing={sorted(expected - mapped)}, "
            f"unexpected={sorted(mapped - expected)}"
        )
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.stage_id, item.rank, item.tensor.name),
    )


def _rank_directory(output: Path, *, stage_count: int, stage_id: int, rank: int) -> Path:
    if stage_count == 1:
        return output / "ranks" / f"rank-{rank:03d}"
    return output / "stages" / f"stage-{stage_id:03d}" / "ranks" / f"rank-{rank:03d}"


def _source_hashes(
    description: ModelDescription,
    tensors: Iterable[TensorInfo],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for tensor in tensors:
        hashes[tensor.name] = _tensor_hash(_read_tensor(description, tensor))
    return hashes


def build_microshards_from_description(
    description: ModelDescription,
    *,
    pipeline_stage_count: int,
    tensor_parallel_degree: int,
    output: Path,
    vocabulary_parallel: bool = True,
) -> MicroshardBuildResult:
    """Construct rank-local safetensors without instantiating a decoder layer."""

    target = output.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise IntegrityError(f"microshard output must be new or empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    config = description.config
    query_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads") or query_heads)
    head_dimension = int(config.get("head_dim") or int(config["hidden_size"]) // query_heads)
    plan = build_dense_partition_plan(
        model_id=description.model_id,
        model_revision=description.model_revision,
        layer_count=int(config["num_hidden_layers"]),
        hidden_size=int(config["hidden_size"]),
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dimension=head_dimension,
        intermediate_size=int(config["intermediate_size"]),
        pipeline_stage_count=pipeline_stage_count,
        tensor_parallel_degree=tensor_parallel_degree,
        vocabulary_parallel=vocabulary_parallel,
        dtype=str(config.get("torch_dtype", "bfloat16")),
    )
    assignments = plan_tensor_assignments(description, plan)
    plan.metadata["explicit_lm_head"] = any(
        item.component.kind == ComponentKind.OUTPUT_HEAD for item in description.tensors
    )
    tensors_by_name = {item.name: item for item in description.tensors}
    source_hashes = _source_hashes(description, tensors_by_name.values())
    grouped: dict[tuple[int, int], list[SliceAssignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[(assignment.stage_id, assignment.rank)].append(assignment)
    tensor_shards: list[TensorShard] = []
    rank_summaries: list[dict[str, Any]] = []
    stage_count = len(plan.pipeline_stages)
    for (stage_id, rank), rank_assignments in sorted(grouped.items()):
        rank_directory = _rank_directory(
            target,
            stage_count=stage_count,
            stage_id=stage_id,
            rank=rank,
        )
        rank_directory.mkdir(parents=True, exist_ok=True)
        local_values: dict[str, Any] = {}
        local_entries: list[TensorShard] = []
        for assignment in rank_assignments:
            # ``get_slice`` may return storage backed by the source mmap.  A rank
            # holds hundreds of slices until its output file is committed; clone
            # each slice so closed safetensors handles cannot leave stale mappings.
            value = _read_slice(description, assignment).clone().contiguous()
            local_values[assignment.tensor.name] = value
            width = _DTYPE_WIDTHS.get(assignment.tensor.dtype)
            if width is None:
                raise IntegrityError(f"unsupported dtype {assignment.tensor.dtype}")
            local_shape = tuple(int(item) for item in value.shape)
            entry = TensorShard(
                tensor_name=assignment.tensor.name,
                global_shape=assignment.tensor.shape,
                local_shape=local_shape,
                shard_axis=assignment.axis,
                shard_start=assignment.start,
                shard_end=assignment.end,
                rank=rank,
                world_size=assignment.world_size,
                dtype=assignment.tensor.dtype,
                source_file=assignment.tensor.source_file,
                source_tensor_hash=source_hashes[assignment.tensor.name],
                local_tensor_hash=_tensor_hash(value),
                replicated=assignment.replicated,
                partition_mode=assignment.partition_mode,
                logical_bytes=int(value.numel() * value.element_size()),
                stage_id=stage_id,
                logical_rank_id=assignment.logical_rank_id,
            )
            local_entries.append(entry)
            tensor_shards.append(entry)
        weights_path = rank_directory / "weights.safetensors"
        save_file(local_values, weights_path)
        _json_write(
            rank_directory / "shard_manifest.json",
            {
                "schema_version": plan.schema_version,
                "model_id": plan.model_id,
                "model_revision": plan.model_revision,
                "stage_id": stage_id,
                "rank": rank,
                "logical_rank_id": _rank_id(stage_id, rank),
                "weight_file": weights_path.name,
                "weight_file_hash": sha256_file(weights_path),
                "logical_weight_bytes": sum(item.logical_bytes for item in local_entries),
                "sharded_weight_bytes": sum(
                    item.logical_bytes for item in local_entries if not item.replicated
                ),
                "replicated_weight_bytes": sum(
                    item.logical_bytes for item in local_entries if item.replicated
                ),
                "tensors": [item.model_dump(mode="json") for item in local_entries],
            },
        )
        rank_summaries.append(
            {
                "stage_id": stage_id,
                "rank": rank,
                "logical_rank_id": _rank_id(stage_id, rank),
                "logical_weight_bytes": sum(item.logical_bytes for item in local_entries),
                "replicated_weight_bytes": sum(
                    item.logical_bytes for item in local_entries if item.replicated
                ),
                "tensor_count": len(local_entries),
                "path": str(rank_directory.relative_to(target)),
            }
        )
        del local_values
    plan.tensor_shards = tensor_shards
    manifest = {
        "schema_version": plan.schema_version,
        "model_id": plan.model_id,
        "model_revision": plan.model_revision,
        "source_path": str(description.model_path),
        "source_tensor_names": sorted(tensors_by_name),
        "source_tensor_count": len(tensors_by_name),
        "source_weight_bytes": sum(item.bytes for item in description.tensors),
        "source_file_hashes": description.source_file_hashes,
        "pipeline_stage_count": stage_count,
        "tensor_parallel_degree": tensor_parallel_degree,
        "logical_pipeline_rank_workers": plan.logical_pipeline_rank_workers,
        "logical_layer_shards": plan.logical_layer_shards,
        "transformer_layer_count": plan.layer_count,
        "more_partitions_than_layers": plan.logical_pipeline_rank_workers > plan.layer_count,
        "vocabulary_parallel": vocabulary_parallel,
        "physical_process_count": 1,
        "cuda_context_count": 1,
        "rank_summaries": rank_summaries,
        "full_decoder_layer_instantiated": False,
    }
    _json_write(target / "manifest.json", manifest)
    _json_write(target / "parallel_plan.json", plan.model_dump(mode="json"))
    _json_write(
        target / "collective_plan.json",
        {
            "groups": [item.model_dump(mode="json") for item in plan.collective_groups],
            "operations": [
                operation.model_dump(mode="json")
                for stage in plan.pipeline_stages
                for layer in stage.layer_plans
                for operation in layer.collective_operations
            ],
        },
    )
    config_directory = target / "config"
    config_directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        source = description.model_path / name
        if source.is_file():
            shutil.copy2(source, config_directory / name)
    validation = validate_microshards(target, source_model=description.model_path)
    hashes = {
        str(path.relative_to(target)).replace("\\", "/"): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name not in {"hashes.json", "validation.json"}
    }
    _json_write(target / "hashes.json", hashes)
    _json_write(target / "validation.json", validation)
    return MicroshardBuildResult(
        output=target,
        plan=plan,
        validation=validation,
        manifest=manifest,
    )


def build_microshards(
    *,
    model: str,
    revision: str | None,
    pipeline_stage_count: int,
    tensor_parallel_degree: int,
    output: Path,
    vocabulary_parallel: bool = True,
    cache_dir: Path | None = None,
    allow_download: bool = True,
) -> MicroshardBuildResult:
    resolved = resolve_model(
        model,
        revision=revision,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    return build_microshards_from_description(
        inspect_qwen3_model(resolved),
        pipeline_stage_count=pipeline_stage_count,
        tensor_parallel_degree=tensor_parallel_degree,
        output=output,
        vocabulary_parallel=vocabulary_parallel,
    )


def _load_rank_manifests(root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(root.rglob("shard_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        manifests.append(payload)
    if not manifests:
        raise IntegrityError(f"no rank shard manifests found under {root}")
    return manifests


def validate_microshards(
    path: Path,
    *,
    source_model: Path | None = None,
) -> dict[str, Any]:
    """Validate values, hashes, shape declarations, and complete source coverage."""

    root = path.expanduser().resolve()
    manifest_path = root / "manifest.json"
    plan_path = root / "parallel_plan.json"
    if not manifest_path.is_file() or not plan_path.is_file():
        raise IntegrityError("microshard manifest.json or parallel_plan.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = ModelPartitionPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    rank_manifests = _load_rank_manifests(root)
    failures: list[str] = []
    entries: list[TensorShard] = []
    actual_rank_ids: set[str] = set()
    for rank_manifest in rank_manifests:
        manifest_file = Path(rank_manifest["_path"])
        weights_path = manifest_file.parent / str(rank_manifest["weight_file"])
        if not weights_path.is_file():
            failures.append(f"missing rank weights: {weights_path}")
            continue
        if sha256_file(weights_path) != rank_manifest["weight_file_hash"]:
            failures.append(f"rank weight-file hash mismatch: {weights_path}")
        actual_rank_ids.add(str(rank_manifest["logical_rank_id"]))
        declared = {
            item.tensor_name: item
            for item in (TensorShard.model_validate(raw) for raw in rank_manifest["tensors"])
        }
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            if actual_names != set(declared):
                failures.append(
                    f"{weights_path}: tensor names differ; "
                    f"missing={sorted(set(declared) - actual_names)}, "
                    f"unexpected={sorted(actual_names - set(declared))}"
                )
            for name in sorted(actual_names & set(declared)):
                value = handle.get_tensor(name)
                entry = declared[name]
                if tuple(value.shape) != entry.local_shape:
                    failures.append(f"{entry.logical_rank_id}:{name}: local shape mismatch")
                if _tensor_hash(value) != entry.local_tensor_hash:
                    failures.append(f"{entry.logical_rank_id}:{name}: local tensor hash mismatch")
                entries.append(entry)
    union = validate_tensor_shard_union(entries)
    failures.extend(union["failures"])
    expected_sources = set(manifest["source_tensor_names"])
    mapped_sources = {entry.tensor_name for entry in entries}
    if expected_sources != mapped_sources:
        failures.append(
            f"source mapping mismatch: missing={sorted(expected_sources - mapped_sources)}, "
            f"unexpected={sorted(mapped_sources - expected_sources)}"
        )
    source_hash_validation = "NOT_REQUESTED"
    if source_model is not None:
        source_hash_validation = "PASS"
        source_path = source_model.expanduser().resolve()
        first_entry_by_name = {entry.tensor_name: entry for entry in entries}
        for entry in first_entry_by_name.values():
            with safe_open(
                source_path / entry.source_file,
                framework="pt",
                device="cpu",
            ) as handle:
                source_value = handle.get_tensor(entry.tensor_name)
            if _tensor_hash(source_value) != entry.source_tensor_hash:
                source_hash_validation = "FAIL"
                failures.append(f"source tensor hash mismatch: {entry.tensor_name}")
                break
    expected_rank_count = len(plan.pipeline_stages) * int(plan.metadata["tensor_parallel_degree"])
    if len(actual_rank_ids) != expected_rank_count:
        failures.append(
            f"logical rank count is {len(actual_rank_ids)}, expected {expected_rank_count}"
        )
    complete_matrix_violations = [
        entry.tensor_name
        for entry in entries
        if entry.world_size > 1
        and not entry.replicated
        and entry.local_shape == entry.global_shape
        and entry.partition_mode != "kv_head_replication"
        and any(
            marker in entry.tensor_name
            for marker in (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
                "embed_tokens",
            )
        )
    ]
    if complete_matrix_violations:
        failures.append(
            "complete large matrices found on TP ranks: "
            + ", ".join(sorted(set(complete_matrix_violations)))
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "pipeline_stage_count": len(plan.pipeline_stages),
        "tensor_parallel_degree": int(plan.metadata["tensor_parallel_degree"]),
        "transformer_layer_count": plan.layer_count,
        "logical_pipeline_rank_workers": plan.logical_pipeline_rank_workers,
        "logical_layer_shards": plan.logical_layer_shards,
        "more_partitions_than_layers": plan.logical_pipeline_rank_workers > plan.layer_count,
        "source_tensor_coverage_status": "PASS" if expected_sources == mapped_sources else "FAIL",
        "source_hash_validation": source_hash_validation,
        "no_complete_large_matrix_status": ("PASS" if not complete_matrix_violations else "FAIL"),
        "complete_matrix_violations": sorted(set(complete_matrix_violations)),
        "union_validation": union,
        "failures": failures,
    }


def inspect_partition(path: Path) -> dict[str, Any]:
    root = path.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation_path = root / "validation.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else validate_microshards(root)
    )
    ranks = manifest.get("rank_summaries", [])
    return {
        "path": str(root),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "pipeline_stage_count": manifest["pipeline_stage_count"],
        "tensor_parallel_degree": manifest["tensor_parallel_degree"],
        "transformer_layer_count": manifest["transformer_layer_count"],
        "logical_pipeline_rank_workers": manifest["logical_pipeline_rank_workers"],
        "logical_layer_shards": manifest["logical_layer_shards"],
        "more_partitions_than_layers": manifest["more_partitions_than_layers"],
        "maximum_logical_rank_weight_bytes": max(
            (int(item["logical_weight_bytes"]) for item in ranks), default=0
        ),
        "maximum_replicated_rank_weight_bytes": max(
            (int(item["replicated_weight_bytes"]) for item in ranks), default=0
        ),
        "validation_status": validation["status"],
        "no_complete_large_matrix_status": validation["no_complete_large_matrix_status"],
    }
