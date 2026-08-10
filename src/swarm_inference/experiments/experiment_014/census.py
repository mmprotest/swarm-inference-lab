"""Fail-closed, header-only census of the official Kimi K3 checkpoint.

The checkpoint contains almost half a million tensors and roughly 1.56 TB of
payload data.  A complete inventory does not need to map those payloads: the
Safetensors headers contain exact shapes, dtypes, and byte ranges.  This module
reads every header twice.  The first pass validates and summarizes the model;
the second streams the complete JSON inventory without retaining hundreds of
megabytes of Python dictionaries in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO

from swarm_inference.model.safetensors import (
    SafetensorsTensorInfo,
    inspect_safetensors,
    normalize_safetensors_dtype,
)

SCHEMA_VERSION = "experiment-014-k3-checkpoint-census-v1"
OFFICIAL_MODEL_ID = "moonshotai/Kimi-K3"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LAYER = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")
_ROUTED_EXPERT = re.compile(r"^block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight_(packed|scale)$")


class CheckpointCensusError(ValueError):
    """The checkpoint cannot satisfy the complete-census gate."""


@dataclass(frozen=True, slots=True)
class TensorClassification:
    component: str
    role: str
    required_for_text_generation: bool
    layer: int | None = None
    attention_type: str | None = None
    routed_expert: int | None = None
    shared_expert: str | None = None
    projection: str | None = None
    quantization_format: str | None = None
    quantization_block_size: int | None = None
    related_tensor: str | None = None


_COMMON_LAYER_ROLES = {
    "input_layernorm.weight": "attention_input_norm",
    "post_attention_layernorm.weight": "moe_input_norm",
    "self_attention_res_norm.weight": "attention_residual_score_norm",
    "self_attention_res_proj.weight": "attention_residual_score_projection",
    "mlp_res_norm.weight": "mlp_residual_score_norm",
    "mlp_res_proj.weight": "mlp_residual_score_projection",
}

_KDA_ROLES = {
    "self_attn.q_proj.weight": "kda_query_projection",
    "self_attn.k_proj.weight": "kda_key_projection",
    "self_attn.v_proj.weight": "kda_value_projection",
    "self_attn.q_conv1d.weight": "kda_query_short_convolution",
    "self_attn.k_conv1d.weight": "kda_key_short_convolution",
    "self_attn.v_conv1d.weight": "kda_value_short_convolution",
    "self_attn.b_proj.weight": "kda_beta_projection",
    "self_attn.o_norm.weight": "kda_per_head_output_norm",
    "self_attn.o_proj.weight": "kda_output_projection",
    "self_attn.A_log": "kda_log_decay",
    "self_attn.f_a_proj.weight": "kda_decay_low_rank_a",
    "self_attn.f_b_proj.weight": "kda_decay_low_rank_b",
    "self_attn.dt_bias": "kda_decay_bias",
    "self_attn.g_proj.weight": "kda_output_gate_projection",
}

_MLA_ROLES = {
    "self_attn.q_a_proj.weight": "mla_query_low_rank_a",
    "self_attn.q_a_layernorm.weight": "mla_query_low_rank_norm",
    "self_attn.q_b_proj.weight": "mla_query_low_rank_b",
    "self_attn.kv_a_proj_with_mqa.weight": "mla_kv_compression_projection",
    "self_attn.kv_a_layernorm.weight": "mla_kv_compression_norm",
    "self_attn.kv_b_proj.weight": "mla_kv_expansion_projection",
    "self_attn.o_proj.weight": "mla_output_projection",
    "self_attn.g_proj.weight": "mla_output_gate_projection",
}

_DENSE_MLP_ROLES = {
    "mlp.gate_proj.weight": ("dense_mlp_gate_projection", "gate"),
    "mlp.up_proj.weight": ("dense_mlp_up_projection", "up"),
    "mlp.down_proj.weight": ("dense_mlp_down_projection", "down"),
}

_MOE_ROLES = {
    "block_sparse_moe.gate.weight": ("moe_router", "router_weight", None),
    "block_sparse_moe.gate.e_score_correction_bias": (
        "moe_router",
        "router_score_correction_bias",
        None,
    ),
    "block_sparse_moe.routed_expert_down_proj.weight": (
        "latent_moe",
        "routed_latent_down_projection",
        "down",
    ),
    "block_sparse_moe.routed_expert_norm.weight": (
        "latent_moe",
        "routed_latent_output_norm",
        None,
    ),
    "block_sparse_moe.routed_expert_up_proj.weight": (
        "latent_moe",
        "routed_latent_up_projection",
        "up",
    ),
    "block_sparse_moe.shared_experts.gate_proj.weight": (
        "shared_experts",
        "shared_expert_gate_projection",
        "gate",
    ),
    "block_sparse_moe.shared_experts.up_proj.weight": (
        "shared_experts",
        "shared_expert_up_projection",
        "up",
    ),
    "block_sparse_moe.shared_experts.down_proj.weight": (
        "shared_experts",
        "shared_expert_down_projection",
        "down",
    ),
}


def _effective_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    if not isinstance(nested, Mapping):
        raise CheckpointCensusError("Kimi K3 config is missing text_config")
    return nested


def _layer_attention_types(config: Mapping[str, Any]) -> dict[int, str]:
    text = _effective_config(config)
    layer_count = int(text["num_hidden_layers"])
    linear = text.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise CheckpointCensusError("Kimi K3 config is missing linear_attn_config")
    kda_one_based = {int(value) for value in linear.get("kda_layers", [])}
    full_one_based = {int(value) for value in linear.get("full_attn_layers", [])}
    expected = set(range(1, layer_count + 1))
    if kda_one_based & full_one_based or kda_one_based | full_one_based != expected:
        raise CheckpointCensusError(
            "KDA and full-attention layer lists must be disjoint and cover every layer"
        )
    return {
        layer: "kda" if layer + 1 in kda_one_based else "gated_mla" for layer in range(layer_count)
    }


def classify_tensor(name: str, config: Mapping[str, Any]) -> TensorClassification:
    """Classify one actual checkpoint tensor; unknown text tensors fail closed."""

    if name == "language_model.model.embed_tokens.weight":
        return TensorClassification("embedding", "token_embedding", True)
    if name == "language_model.model.norm.weight":
        return TensorClassification("final_norm", "final_rms_norm", True)
    if name == "language_model.model.output_attn_res_norm.weight":
        return TensorClassification(
            "final_attention_residual", "final_attention_residual_score_norm", True
        )
    if name == "language_model.model.output_attn_res_proj.weight":
        return TensorClassification(
            "final_attention_residual", "final_attention_residual_score_projection", True
        )
    if name == "language_model.lm_head.weight":
        return TensorClassification("lm_head", "output_projection", True, projection="output")
    if name.startswith("vision_tower."):
        return TensorClassification("vision_tower", "vision_parameter", False)
    if name.startswith("mm_projector."):
        return TensorClassification("vision_projector", "multimodal_projection", False)

    match = _LAYER.match(name)
    if match is None:
        raise CheckpointCensusError(f"unclassified checkpoint tensor {name!r}")
    layer = int(match.group(1))
    suffix = match.group(2)
    attention_types = _layer_attention_types(config)
    if layer not in attention_types:
        raise CheckpointCensusError(f"tensor {name!r} references invalid layer {layer}")
    attention_type = attention_types[layer]

    expert = _ROUTED_EXPERT.match(suffix)
    if expert is not None:
        expert_id = int(expert.group(1))
        projection_code = expert.group(2)
        storage = expert.group(3)
        projection = {"w1": "gate", "w2": "down", "w3": "up"}[projection_code]
        base = name.rsplit("_", 1)[0]
        if storage == "packed":
            return TensorClassification(
                component="routed_expert",
                role=f"routed_expert_{projection}_packed_weight",
                required_for_text_generation=True,
                layer=layer,
                attention_type=attention_type,
                routed_expert=expert_id,
                projection=projection,
                quantization_format="mxfp4-e2m1",
                quantization_block_size=32,
                related_tensor=f"{base}_scale",
            )
        return TensorClassification(
            component="quantization_scale",
            role=f"routed_expert_{projection}_ue8m0_scale",
            required_for_text_generation=True,
            layer=layer,
            attention_type=attention_type,
            routed_expert=expert_id,
            projection=projection,
            quantization_format="ue8m0",
            quantization_block_size=32,
            related_tensor=f"{base}_packed",
        )

    common = _COMMON_LAYER_ROLES.get(suffix)
    if common is not None:
        return TensorClassification(
            "attention_residual" if "attention_" in suffix else "mlp_residual",
            common,
            True,
            layer=layer,
            attention_type=attention_type,
        )
    dense = _DENSE_MLP_ROLES.get(suffix)
    if dense is not None:
        return TensorClassification(
            "dense_mlp",
            dense[0],
            True,
            layer=layer,
            attention_type=attention_type,
            projection=dense[1],
        )
    moe = _MOE_ROLES.get(suffix)
    if moe is not None:
        component, role, projection = moe
        return TensorClassification(
            component,
            role,
            True,
            layer=layer,
            attention_type=attention_type,
            shared_expert="fused_shared_experts" if component == "shared_experts" else None,
            projection=projection,
        )
    attention_roles = _KDA_ROLES if attention_type == "kda" else _MLA_ROLES
    role = attention_roles.get(suffix)
    if role is not None:
        return TensorClassification(
            f"{attention_type}_attention",
            role,
            True,
            layer=layer,
            attention_type=attention_type,
        )
    raise CheckpointCensusError(f"unclassified required text tensor {name!r}")


def _logical_shape(
    tensor: SafetensorsTensorInfo,
    classification: TensorClassification,
    config: Mapping[str, Any],
) -> tuple[int, ...]:
    if classification.quantization_format != "mxfp4-e2m1":
        return tensor.shape
    text = _effective_config(config)
    latent = int(text["routed_expert_hidden_size"])
    intermediate = int(text["moe_intermediate_size"])
    if classification.projection == "down":
        expected = (latent, intermediate // 2)
        logical = (latent, intermediate)
    else:
        expected = (intermediate, latent // 2)
        logical = (intermediate, latent)
    if tensor.shape != expected:
        raise CheckpointCensusError(
            f"MXFP4 tensor {tensor.name!r} has shape {tensor.shape}, expected {expected}"
        )
    return logical


def _logical_bytes(
    tensor: SafetensorsTensorInfo,
    classification: TensorClassification,
    logical_shape: tuple[int, ...],
) -> int:
    if classification.quantization_format == "mxfp4-e2m1":
        # The census defines logical bytes as the BF16-equivalent dequantized
        # tensor footprint. Physical bytes retain the exact native MXFP4 size.
        return math.prod(logical_shape) * 2
    return tensor.byte_size


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCensusError(f"invalid JSON metadata: {path}") from exc
    if not isinstance(value, dict):
        raise CheckpointCensusError(f"JSON metadata must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_integrity_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_payloads_resumable(
    *,
    checkpoint: Path,
    shard_names: tuple[str, ...],
    revision: str,
    config_sha256: str,
    index_sha256: str,
    receipt_path: Path,
    workers: int,
) -> dict[str, str]:
    if workers < 1:
        raise CheckpointCensusError("hash worker count must be positive")
    identity = {
        "revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
    }
    cached: dict[str, Any] = {}
    if receipt_path.is_file():
        existing = _load_json(receipt_path)
        if existing.get("checkpoint_identity") != identity:
            raise CheckpointCensusError(
                "existing payload-integrity receipt belongs to a different checkpoint"
            )
        raw_cached = existing.get("shards")
        if isinstance(raw_cached, dict):
            cached = dict(raw_cached)

    verified: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, Path, str, int, int]] = []
    for shard_name in shard_names:
        path = checkpoint / shard_name
        _metadata_revision, expected = _download_metadata(checkpoint, shard_name)
        if expected is None or _SHA256.fullmatch(expected) is None:
            raise CheckpointCensusError(
                f"weight shard {shard_name} has no upstream LFS SHA-256 identity"
            )
        stat = path.stat()
        candidate = cached.get(shard_name)
        if (
            isinstance(candidate, dict)
            and candidate.get("expected_sha256") == expected
            and candidate.get("verified_local_sha256") == expected
            and candidate.get("file_bytes") == stat.st_size
            and candidate.get("file_mtime_ns") == stat.st_mtime_ns
        ):
            verified[shard_name] = dict(candidate)
            continue
        pending.append((shard_name, path, expected, stat.st_size, stat.st_mtime_ns))

    receipt: dict[str, Any] = {
        "schema_version": "experiment-014-k3-payload-integrity-v1",
        "checkpoint_identity": identity,
        "shard_count": len(shard_names),
        "verified_shard_count": len(verified),
        "complete": not pending,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "shards": verified,
    }
    _write_integrity_receipt(receipt_path, receipt)
    if pending:
        print(
            json.dumps(
                {
                    "event": "payload_hash_start",
                    "cached_shards": len(verified),
                    "pending_shards": len(pending),
                    "workers": workers,
                    "receipt": str(receipt_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def verify_one(item: tuple[str, Path, str, int, int]) -> tuple[str, dict[str, Any]]:
        shard_name, path, expected, size, mtime_ns = item
        started = datetime.now(UTC)
        actual = _sha256(path)
        completed = datetime.now(UTC)
        if actual != expected:
            raise CheckpointCensusError(f"local payload hash mismatch for {shard_name}")
        return shard_name, {
            "expected_sha256": expected,
            "verified_local_sha256": actual,
            "file_bytes": size,
            "file_mtime_ns": mtime_ns,
            "started_at_utc": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
        }

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="k3-sha256") as executor:
        futures = {executor.submit(verify_one, item): item[0] for item in pending}
        for future in as_completed(futures):
            shard_name, result = future.result()
            verified[shard_name] = result
            receipt.update(
                {
                    "verified_shard_count": len(verified),
                    "complete": len(verified) == len(shard_names),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "shards": dict(sorted(verified.items())),
                }
            )
            _write_integrity_receipt(receipt_path, receipt)
            print(
                json.dumps(
                    {
                        "event": "payload_hash_verified",
                        "shard": shard_name,
                        "duration_seconds": result["duration_seconds"],
                        "verified_shards": len(verified),
                        "total_shards": len(shard_names),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        shard_name: str(verified[shard_name]["verified_local_sha256"]) for shard_name in shard_names
    }


def _download_metadata(checkpoint: Path, filename: str) -> tuple[str | None, str | None]:
    path = checkpoint / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    if not path.is_file():
        return None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise CheckpointCensusError(f"malformed Hugging Face download metadata: {path}")
    revision, etag = lines[0].strip().lower(), lines[1].strip().lower()
    if _REVISION.fullmatch(revision) is None or _CONTENT_ID.fullmatch(etag) is None:
        raise CheckpointCensusError(f"invalid revision/hash in download metadata: {path}")
    return revision, etag


def _revision(checkpoint: Path, shard_names: tuple[str, ...]) -> str:
    observed: set[str] = set()
    for filename in ("config.json", "model.safetensors.index.json", *shard_names):
        revision, _etag = _download_metadata(checkpoint, filename)
        if revision is not None:
            observed.add(revision)
    tree_directory = checkpoint / ".cache" / "huggingface" / "trees"
    if tree_directory.is_dir():
        observed.update(
            path.stem.lower()
            for path in tree_directory.glob("*.json")
            if _REVISION.fullmatch(path.stem.lower()) is not None
        )
    if len(observed) != 1:
        raise CheckpointCensusError(
            f"checkpoint revision is missing or inconsistent: {sorted(observed)}"
        )
    return next(iter(observed))


def _architecture_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    text = _effective_config(config)
    attention = _layer_attention_types(config)
    quant = text.get("quantization_config")
    return {
        "root_model_type": config.get("model_type"),
        "root_architecture": (config.get("architectures") or [None])[0],
        "text_model_type": text.get("model_type"),
        "text_architecture": (text.get("architectures") or [None])[0],
        "transformer_layers": int(text["num_hidden_layers"]),
        "kda_layers": [layer for layer, value in attention.items() if value == "kda"],
        "gated_mla_layers": [layer for layer, value in attention.items() if value == "gated_mla"],
        "hidden_size": int(text["hidden_size"]),
        "routed_expert_hidden_size": int(text["routed_expert_hidden_size"]),
        "moe_intermediate_size": int(text["moe_intermediate_size"]),
        "routed_experts": int(text["num_experts"]),
        "selected_routed_experts_per_token": int(text["num_experts_per_token"]),
        "shared_experts": int(text["num_shared_experts"]),
        "vocabulary_size": int(text["vocab_size"]),
        "max_position_embeddings": int(text["max_position_embeddings"]),
        "quantization_format": quant.get("format") if isinstance(quant, Mapping) else None,
        "weight_representation": "native MXFP4 E2M1 + UE8M0 group-32 routed experts",
        "activation_target": "MXFP8",
    }


@dataclass(slots=True)
class _ScanSummary:
    tensor_count: int = 0
    physical_bytes: int = 0
    logical_bytes: int = 0
    required_tensor_count: int = 0
    required_physical_bytes: int = 0
    required_logical_bytes: int = 0
    unclassified_required_tensors: int = 0


def _iter_tensors(
    checkpoint: Path,
    shard_names: tuple[str, ...],
) -> Iterator[tuple[str, SafetensorsTensorInfo]]:
    for shard_name in shard_names:
        for tensor in inspect_safetensors(checkpoint / shard_name):
            yield shard_name, tensor


def _validate_semantic_coverage(
    *,
    config: Mapping[str, Any],
    layer_role_counts: Mapping[int, Counter[str]],
    expert_masks: Mapping[tuple[int, int], int],
) -> None:
    text = _effective_config(config)
    layer_count = int(text["num_hidden_layers"])
    expert_count = int(text["num_experts"])
    dense_layers = int(text["first_k_dense_replace"])
    expected_common = set(_COMMON_LAYER_ROLES.values())
    expected_kda = set(_KDA_ROLES.values())
    expected_mla = set(_MLA_ROLES.values())
    expected_dense = {value[0] for value in _DENSE_MLP_ROLES.values()}
    expected_moe = {value[1] for value in _MOE_ROLES.values()}
    attention = _layer_attention_types(config)
    for layer in range(layer_count):
        observed = layer_role_counts.get(layer, Counter())
        required = expected_common | (expected_kda if attention[layer] == "kda" else expected_mla)
        required |= expected_dense if layer < dense_layers else expected_moe
        missing = sorted(role for role in required if observed[role] != 1)
        if missing:
            raise CheckpointCensusError(
                f"layer {layer} is missing or duplicates semantic roles: {missing}"
            )
        if layer < dense_layers:
            if any(key[0] == layer for key in expert_masks):
                raise CheckpointCensusError(
                    f"dense layer {layer} unexpectedly contains routed experts"
                )
            continue
        for expert in range(expert_count):
            if expert_masks.get((layer, expert)) != 0b111111:
                raise CheckpointCensusError(
                    f"layer {layer} routed expert {expert} does not have all packed/scale projections"
                )


def _record_payload(
    *,
    shard_name: str,
    tensor: SafetensorsTensorInfo,
    classification: TensorClassification,
    logical_shape: tuple[int, ...],
    logical_bytes: int,
) -> dict[str, Any]:
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "logical_shape": list(logical_shape),
        "dtype": normalize_safetensors_dtype(tensor.dtype),
        "physical_bytes": tensor.byte_size,
        "logical_bytes": logical_bytes,
        "logical_elements": math.prod(logical_shape),
        "safetensors_file": shard_name,
        "byte_range": [tensor.data_offset, tensor.data_offset + tensor.byte_size],
        **asdict(classification),
    }


def _write_json_item(handle: TextIO, value: Any, *, first: bool) -> bool:
    if not first:
        handle.write(",\n")
    json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return False


def build_checkpoint_census(
    checkpoint: Path,
    output_path: Path,
    *,
    verify_payload_hashes: bool = False,
    integrity_receipt_path: Path | None = None,
    hash_workers: int = 1,
) -> dict[str, Any]:
    """Build the exact census and return its compact validation receipt."""

    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise CheckpointCensusError(f"checkpoint directory does not exist: {root}")
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = _load_json(config_path)
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointCensusError("checkpoint index has no weight_map")
    shard_names = tuple(sorted({str(value) for value in weight_map.values()}))
    if any(Path(name).name != name for name in shard_names):
        raise CheckpointCensusError("checkpoint index contains a non-local shard path")
    revision = _revision(root, shard_names)
    architecture = _architecture_payload(config)
    config_sha256 = _sha256(config_path)
    index_sha256 = _sha256(index_path)

    summary = _ScanSummary()
    seen: set[str] = set()
    components: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    layers: dict[int, Counter[str]] = defaultdict(Counter)
    expert_masks: dict[tuple[int, int], int] = {}
    shard_tensor_counts: Counter[str] = Counter()
    shard_tensor_bytes: Counter[str] = Counter()
    shard_ranges: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    projection_bits = {
        ("gate", "mxfp4-e2m1"): 0,
        ("gate", "ue8m0"): 1,
        ("down", "mxfp4-e2m1"): 2,
        ("down", "ue8m0"): 3,
        ("up", "mxfp4-e2m1"): 4,
        ("up", "ue8m0"): 5,
    }
    for shard_name, tensor in _iter_tensors(root, shard_names):
        if tensor.name in seen:
            raise CheckpointCensusError(f"duplicate tensor in Safetensors headers: {tensor.name}")
        seen.add(tensor.name)
        indexed_shard = weight_map.get(tensor.name)
        if indexed_shard != shard_name:
            raise CheckpointCensusError(
                f"index/header shard mismatch for {tensor.name!r}: {indexed_shard!r} != {shard_name!r}"
            )
        classification = classify_tensor(tensor.name, config)
        logical_shape = _logical_shape(tensor, classification, config)
        logical_bytes = _logical_bytes(tensor, classification, logical_shape)
        summary.tensor_count += 1
        summary.physical_bytes += tensor.byte_size
        summary.logical_bytes += logical_bytes
        if classification.required_for_text_generation:
            summary.required_tensor_count += 1
            summary.required_physical_bytes += tensor.byte_size
            summary.required_logical_bytes += logical_bytes
        components[classification.component] += 1
        roles[classification.role] += 1
        dtypes[normalize_safetensors_dtype(tensor.dtype)] += 1
        if classification.layer is not None:
            layers[classification.layer][classification.role] += 1
        if classification.routed_expert is not None:
            key = (classification.layer or 0, classification.routed_expert)
            bit = projection_bits[
                (classification.projection or "", classification.quantization_format or "")
            ]
            expert_masks[key] = expert_masks.get(key, 0) | (1 << bit)
        shard_tensor_counts[shard_name] += 1
        shard_tensor_bytes[shard_name] += tensor.byte_size
        shard_ranges[shard_name].append(
            (tensor.data_offset, tensor.data_offset + tensor.byte_size, tensor.name)
        )

    indexed_names = {str(name) for name in weight_map}
    if seen != indexed_names:
        missing = sorted(indexed_names - seen)[:10]
        extra = sorted(seen - indexed_names)[:10]
        raise CheckpointCensusError(
            f"index/header tensor mismatch; missing={missing}, extra={extra}"
        )
    for shard_name, ranges in shard_ranges.items():
        ordered = sorted(ranges)
        for previous, current in pairwise(ordered):
            if previous[1] > current[0]:
                raise CheckpointCensusError(
                    f"overlapping Safetensors ranges in {shard_name}: {previous[2]} / {current[2]}"
                )
    _validate_semantic_coverage(
        config=config,
        layer_role_counts=layers,
        expert_masks=expert_masks,
    )
    if summary.tensor_count != len(weight_map):
        raise CheckpointCensusError("complete tensor count does not match the checkpoint index")

    verified_hashes: dict[str, str] = {}
    if verify_payload_hashes:
        receipt = (
            integrity_receipt_path.expanduser().resolve()
            if integrity_receipt_path is not None
            else output_path.expanduser().resolve().with_name("k3-checkpoint-integrity.json")
        )
        verified_hashes = _verify_payloads_resumable(
            checkpoint=root,
            shard_names=shard_names,
            revision=revision,
            config_sha256=config_sha256,
            index_sha256=index_sha256,
            receipt_path=receipt,
            workers=hash_workers,
        )

    shard_files: list[dict[str, Any]] = []
    revisions = set()
    for shard_name in shard_names:
        path = root / shard_name
        metadata_revision, etag = _download_metadata(root, shard_name)
        if metadata_revision is not None:
            revisions.add(metadata_revision)
        if etag is None or _SHA256.fullmatch(etag) is None:
            raise CheckpointCensusError(
                f"weight shard {shard_name} has no upstream LFS SHA-256 identity"
            )
        actual_hash = verified_hashes.get(shard_name)
        if actual_hash is not None and etag is not None and actual_hash != etag:
            raise CheckpointCensusError(f"local payload hash mismatch for {shard_name}")
        shard_files.append(
            {
                "name": shard_name,
                "physical_file_bytes": path.stat().st_size,
                "tensor_payload_bytes": shard_tensor_bytes[shard_name],
                "tensor_count": shard_tensor_counts[shard_name],
                "upstream_lfs_sha256": etag,
                "verified_local_sha256": actual_hash,
            }
        )
    if revisions and revisions != {revision}:
        raise CheckpointCensusError("shard metadata revisions are inconsistent")

    declared_total = int(index.get("metadata", {}).get("total_size", 0))
    if declared_total != summary.physical_bytes:
        raise CheckpointCensusError(
            f"index declares {declared_total} tensor bytes but headers contain {summary.physical_bytes}"
        )
    physical_file_bytes = sum(item["physical_file_bytes"] for item in shard_files)
    identity_seed = {
        "model_id": OFFICIAL_MODEL_ID,
        "revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shards": [
            [item["name"], item["physical_file_bytes"], item["upstream_lfs_sha256"]]
            for item in shard_files
        ],
    }
    checkpoint_fingerprint = hashlib.sha256(
        json.dumps(identity_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    compact_summary = {
        **asdict(summary),
        "component_counts": dict(sorted(components.items())),
        "role_counts": dict(sorted(roles.items())),
        "dtype_counts": dict(sorted(dtypes.items())),
        "unclassified_required_tensors": 0,
        "all_required_tensors_classified": True,
        "index_declared_tensor_bytes": declared_total,
        "checkpoint_physical_file_bytes": physical_file_bytes,
    }
    checkpoint_payload = {
        "model_id": OFFICIAL_MODEL_ID,
        "revision": revision,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "source": "local immutable Hugging Face snapshot",
        "local_path": str(root),
        "payload_hash_verification": (
            "all local shard SHA-256 values verified"
            if verify_payload_hashes
            else "upstream LFS SHA-256 identities recorded; local payload hashing not requested"
        ),
    }

    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{\n")
        for key, value in (
            ("schema_version", SCHEMA_VERSION),
            ("generated_at_utc", datetime.now(UTC).isoformat()),
            ("checkpoint", checkpoint_payload),
            ("architecture", architecture),
            (
                "byte_definitions",
                {
                    "physical_bytes": "exact bytes occupied by the tensor payload in Safetensors",
                    "logical_bytes": (
                        "BF16-equivalent dequantized bytes for MXFP4 packed weights; exact "
                        "physical bytes for every other tensor"
                    ),
                    "byte_range": "absolute half-open [start,end) offsets in the source shard",
                },
            ),
            ("summary", compact_summary),
            ("shards", shard_files),
        ):
            handle.write(json.dumps(key) + ":")
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(",\n")
        handle.write('"tensors":[\n')
        first = True
        for shard_name, tensor in _iter_tensors(root, shard_names):
            classification = classify_tensor(tensor.name, config)
            logical_shape = _logical_shape(tensor, classification, config)
            first = _write_json_item(
                handle,
                _record_payload(
                    shard_name=shard_name,
                    tensor=tensor,
                    classification=classification,
                    logical_shape=logical_shape,
                    logical_bytes=_logical_bytes(tensor, classification, logical_shape),
                ),
                first=first,
            )
        handle.write("\n]}\n")
    temporary.replace(destination)
    census_sha256 = _sha256(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "output_path": str(destination),
        "output_bytes": destination.stat().st_size,
        "output_sha256": census_sha256,
        "checkpoint": checkpoint_payload,
        "architecture": architecture,
        "summary": compact_summary,
    }


__all__ = [
    "CheckpointCensusError",
    "TensorClassification",
    "build_checkpoint_census",
    "classify_tensor",
]
