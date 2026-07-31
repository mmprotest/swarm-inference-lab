"""Official-checkpoint-derived Kimi K3 memory and topology projections."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarm_inference.microsharding.projection import (
    NETWORK_PROFILES,
    estimate_collective,
)

K3_MODEL_ID = "moonshotai/Kimi-K3"
K3_VERIFIED_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
K3_REQUIRED_WORDING = (
    "This Kimi K3 result is a checkpoint-derived and trace-derived projection. "
    "It is not a physical Kimi K3 inference result."
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class K3Metadata:
    model_id: str
    revision: str
    config_path: str
    index_path: str
    config_sha256: str
    index_sha256: str
    total_checkpoint_bytes: int
    checkpoint_file_count: int
    checkpoint_tensor_count: int
    transformer_layer_count: int
    kda_layer_ids: tuple[int, ...]
    gated_mla_layer_ids: tuple[int, ...]
    hidden_size: int
    routed_expert_hidden_size: int
    moe_intermediate_size: int
    routed_expert_count: int
    active_experts_per_token: int
    shared_expert_count: int
    vocabulary_size: int
    attention_head_count: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    value_head_dim: int
    estimated_bytes_per_routed_expert: int
    estimated_shared_expert_bytes_per_layer: int
    estimated_largest_attention_tensor_bytes: int
    estimated_kv_bytes_per_token_per_mla_layer: int
    estimated_recurrent_state_bytes_per_kda_layer: int

    def payload(self) -> dict[str, Any]:
        return {
            "classification": "k3_checkpoint_projection",
            "model_id": self.model_id,
            "revision": self.revision,
            "official_config_path": self.config_path,
            "official_index_path": self.index_path,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "total_checkpoint_bytes": self.total_checkpoint_bytes,
            "checkpoint_file_count": self.checkpoint_file_count,
            "checkpoint_tensor_count": self.checkpoint_tensor_count,
            "transformer_layer_count": self.transformer_layer_count,
            "attention_type_by_layer": {
                str(layer): ("kda" if layer + 1 in self.kda_layer_ids else "gated_mla")
                for layer in range(self.transformer_layer_count)
            },
            "kda_layer_count": len(self.kda_layer_ids),
            "gated_mla_layer_count": len(self.gated_mla_layer_ids),
            "routed_expert_count": self.routed_expert_count,
            "active_experts_per_token": self.active_experts_per_token,
            "shared_expert_count": self.shared_expert_count,
            "hidden_size": self.hidden_size,
            "routed_expert_hidden_size": self.routed_expert_hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "vocabulary_size": self.vocabulary_size,
            "attention_head_count": self.attention_head_count,
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "value_head_dim": self.value_head_dim,
            "estimated_bytes_per_expert": self.estimated_bytes_per_routed_expert,
            "largest_expert_tensor": {
                "description": (
                    "estimated MXFP4 gate, up, or down tensor including per-group scales"
                ),
                "bytes": self.estimated_bytes_per_routed_expert // 3,
            },
            "estimated_shared_expert_bytes_per_layer": (
                self.estimated_shared_expert_bytes_per_layer
            ),
            "largest_attention_tensor": {
                "description": "estimated BF16 Gated-MLA output projection",
                "bytes": self.estimated_largest_attention_tensor_bytes,
            },
            "estimated_kv_bytes_per_token_per_mla_layer": (
                self.estimated_kv_bytes_per_token_per_mla_layer
            ),
            "estimated_recurrent_state_bytes_per_kda_layer": (
                self.estimated_recurrent_state_bytes_per_kda_layer
            ),
            "quantization_note": (
                "Routed experts use official MXFP4 packed weights with one byte-scale per "
                "32 values; ignored attention/shared tensors are estimated at BF16."
            ),
            "required_wording": K3_REQUIRED_WORDING,
            "physical_inference_performed": False,
        }


def resolve_k3_metadata(
    *,
    model_id: str = K3_MODEL_ID,
    revision: str = K3_VERIFIED_REVISION,
    cache_dir: Path | None = None,
) -> K3Metadata:
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(model_id, revision=revision, files_metadata=True)
    exact_revision = info.sha
    if exact_revision is None:
        raise ValueError("official K3 repository did not resolve an immutable revision")
    config_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=exact_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    index_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=exact_revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    root = json.loads(config_path.read_text(encoding="utf-8"))
    config = root["text_config"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    linear = config["linear_attn_config"]
    kda = tuple(int(item) for item in linear["kda_layers"])
    full = tuple(int(item) for item in linear["full_attn_layers"])
    hidden = int(config["hidden_size"])
    latent = int(config["routed_expert_hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["num_experts"])
    # MXFP4 stores two values per byte plus a one-byte scale for each group of 32.
    logical_parameters_per_expert = 3 * latent * intermediate
    packed_bytes_per_expert = math.ceil(logical_parameters_per_expert / 2)
    scale_bytes_per_expert = math.ceil(logical_parameters_per_expert / 32)
    bytes_per_expert = packed_bytes_per_expert + scale_bytes_per_expert
    shared_intermediate = int(config["num_shared_experts"]) * intermediate
    shared_bytes = 3 * hidden * shared_intermediate * 2
    attention_output_width = int(config["num_attention_heads"]) * int(config["v_head_dim"])
    largest_attention = hidden * attention_output_width * 2
    kv_per_token = (int(config["kv_lora_rank"]) + int(config["qk_rope_head_dim"])) * 2
    recurrent = int(linear["num_heads"]) * int(linear["head_dim"]) ** 2 * 2
    safetensors = [
        sibling for sibling in (info.siblings or []) if sibling.rfilename.endswith(".safetensors")
    ]
    total = int(index.get("metadata", {}).get("total_size") or 0)
    if total <= 0:
        total = sum(int(item.size or 0) for item in safetensors)
    return K3Metadata(
        model_id=model_id,
        revision=exact_revision,
        config_path=str(config_path),
        index_path=str(index_path),
        config_sha256=_file_hash(config_path),
        index_sha256=_file_hash(index_path),
        total_checkpoint_bytes=total,
        checkpoint_file_count=len(safetensors),
        checkpoint_tensor_count=len(index["weight_map"]),
        transformer_layer_count=int(config["num_hidden_layers"]),
        kda_layer_ids=kda,
        gated_mla_layer_ids=full,
        hidden_size=hidden,
        routed_expert_hidden_size=latent,
        moe_intermediate_size=intermediate,
        routed_expert_count=experts,
        active_experts_per_token=int(config["num_experts_per_token"]),
        shared_expert_count=int(config["num_shared_experts"]),
        vocabulary_size=int(config["vocab_size"]),
        attention_head_count=int(config["num_attention_heads"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        q_lora_rank=int(config["q_lora_rank"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        value_head_dim=int(config["v_head_dim"]),
        estimated_bytes_per_routed_expert=bytes_per_expert,
        estimated_shared_expert_bytes_per_layer=shared_bytes,
        estimated_largest_attention_tensor_bytes=largest_attention,
        estimated_kv_bytes_per_token_per_mla_layer=kv_per_token,
        estimated_recurrent_state_bytes_per_kda_layer=recurrent,
    )


@dataclass(frozen=True, slots=True)
class K3Candidate:
    name: str
    pipeline_stages: int
    attention_tp: int
    expert_parallel: int
    ranks_per_cell: int
    intra_cell_network: str
    inter_cell_network: str
    global_tensor_group: bool = False


K3_CANDIDATES = (
    K3Candidate("whole_layer_pipeline", 93, 1, 1, 1, "nvlink_class", "home_lan_10gbe"),
    K3Candidate("attention_tp4_ep16", 93, 4, 16, 16, "nvlink_class", "home_lan_10gbe"),
    K3Candidate("attention_tp8_ep32", 93, 8, 32, 32, "nvlink_class", "home_lan_10gbe"),
    K3Candidate("attention_tp16_ep32", 93, 16, 32, 32, "nvlink_class", "home_lan_10gbe"),
    K3Candidate("regional_cell_hierarchy", 12, 8, 32, 32, "home_lan_10gbe", "regional"),
    K3Candidate(
        "global_tensor_parallel_negative_control",
        1,
        32,
        32,
        32,
        "global_residential",
        "global_residential",
        True,
    ),
)


def _project_candidate(metadata: K3Metadata, candidate: K3Candidate) -> dict[str, Any]:
    layers_per_stage = math.ceil(metadata.transformer_layer_count / candidate.pipeline_stages)
    routed_per_rank = math.ceil(metadata.routed_expert_count / candidate.expert_parallel)
    expert_bytes_per_layer_rank = routed_per_rank * metadata.estimated_bytes_per_routed_expert
    attention_bytes_per_layer_rank = math.ceil(
        metadata.estimated_largest_attention_tensor_bytes / candidate.attention_tp
    )
    shared_bytes_per_layer_rank = math.ceil(
        metadata.estimated_shared_expert_bytes_per_layer / candidate.attention_tp
    )
    router_bytes = metadata.routed_expert_count * metadata.hidden_size * 2
    layer_rank_bytes = (
        expert_bytes_per_layer_rank
        + attention_bytes_per_layer_rank
        + shared_bytes_per_layer_rank
        + router_bytes
    )
    max_rank_bytes = layer_rank_bytes * layers_per_stage
    intra = NETWORK_PROFILES[candidate.intra_cell_network]
    hidden_payload = metadata.hidden_size * 2
    dispatch_payload = metadata.hidden_size * 2 * metadata.active_experts_per_token
    attention_collective = estimate_collective(
        operation="all_reduce_sum",
        algorithm="recursive_doubling",
        rank_count=candidate.attention_tp,
        payload_bytes=hidden_payload,
        network=intra,
    )
    dispatch = estimate_collective(
        operation="all_to_all",
        algorithm="ring",
        rank_count=candidate.expert_parallel,
        payload_bytes=dispatch_payload,
        network=intra,
    )
    # Assumption: a high-end GPU reads its selected routed-expert shard at an
    # effective 600 GB/s after kernel/quantisation overhead.  This is explicit
    # projection input, not a K3 physical measurement.
    active_experts_per_rank = max(
        1, math.ceil(metadata.active_experts_per_token / candidate.expert_parallel)
    )
    active_weight_bytes = active_experts_per_rank * metadata.estimated_bytes_per_routed_expert
    assumed_compute_ms = max(active_weight_bytes / 600_000_000_000 * 1_000, 0.05)
    layer_latency = (
        assumed_compute_ms
        + 2 * attention_collective.completion_time_ms
        + 2 * dispatch.completion_time_ms
    )
    inter = NETWORK_PROFILES[candidate.inter_cell_network]
    pipeline_hop_ms = inter.one_way_latency_ms + (
        0.0
        if inter.bandwidth_mbps is None
        else hidden_payload * 8 / (inter.bandwidth_mbps * 1_000_000) * 1_000
    )
    single_token_ms = (
        metadata.transformer_layer_count * layer_latency
        + max(candidate.pipeline_stages - 1, 0) * pipeline_hop_ms
    )
    single_tps = 1_000 / max(single_token_ms, 1e-9)
    aggregate_tps = single_tps * min(64, candidate.pipeline_stages)
    fanout = min(metadata.active_experts_per_token, candidate.expert_parallel)
    replicated_bytes = (
        shared_bytes_per_layer_rank * layers_per_stage + router_bytes * layers_per_stage
    )
    expected_collectives = (2 if candidate.attention_tp > 1 else 0) + (
        2 if candidate.expert_parallel > 1 else 0
    )
    logical_payload_bytes = (2 * hidden_payload if candidate.attention_tp > 1 else 0) + (
        2 * dispatch_payload if candidate.expert_parallel > 1 else 0
    )
    logical_aggregate_bytes = (
        2 * attention_collective.aggregate_bytes if candidate.attention_tp > 1 else 0
    ) + (2 * dispatch.aggregate_bytes if candidate.expert_parallel > 1 else 0)
    row = {
        "classification": "k3_checkpoint_projection",
        "plan": candidate.name,
        "pipeline_stage_count": candidate.pipeline_stages,
        "ranks_per_cell": candidate.ranks_per_cell,
        "total_logical_ranks": candidate.pipeline_stages * candidate.ranks_per_cell,
        "attention_tensor_parallel_degree": candidate.attention_tp,
        "expert_parallel_degree": candidate.expert_parallel,
        "maximum_weight_bytes_per_rank": max_rank_bytes,
        "replicated_bytes_per_rank": replicated_bytes,
        "estimated_bytes_per_attention_shard": attention_bytes_per_layer_rank,
        "estimated_bytes_per_expert": metadata.estimated_bytes_per_routed_expert,
        "expected_collectives_per_token_per_layer": expected_collectives,
        "expert_dispatch_fanout": fanout,
        "logical_payload_bytes_per_token_per_layer": logical_payload_bytes,
        "logical_aggregate_bytes_per_token_per_layer": logical_aggregate_bytes,
        "logical_bytes_per_token_per_layer": logical_aggregate_bytes,
        "projected_single_stream_tokens_per_second": single_tps,
        "projected_aggregate_tokens_per_second_concurrency_64": aggregate_tps,
        "network_requirements": {
            "intra_cell": candidate.intra_cell_network,
            "inter_cell": candidate.inter_cell_network,
            "global_synchronous_tensor_group": candidate.global_tensor_group,
        },
        "minimum_availability_requirements": (
            f"all {candidate.ranks_per_cell} synchronous ranks per active cell"
        ),
        "node_32_gib_can_participate": max_rank_bytes <= 32 * 1024**3,
        "node_8_gib_can_participate": max_rank_bytes <= 8 * 1024**3,
        "node_1_gib_can_participate": max_rank_bytes <= 1 * 1024**3,
        "raspberry_pi_positive_marginal_benefit": False,
        "target_20_tps_reached": single_tps >= 20,
        "memory_feasible": max_rank_bytes <= 32 * 1024**3,
        "latency_beneficial": single_tps > 0 and not candidate.global_tensor_group,
        "throughput_beneficial": aggregate_tps > single_tps,
        "assumed_compute_ms_per_layer_rank": assumed_compute_ms,
        "required_wording": K3_REQUIRED_WORDING,
    }
    if candidate.global_tensor_group:
        row["recommendation"] = (
            "negative control only; global synchronous tensor collectives are not recommended"
        )
        row["latency_beneficial"] = False
    else:
        row["recommendation"] = "hierarchical low-latency cell candidate"
    return row


def project_k3(metadata: K3Metadata) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plans = [_project_candidate(metadata, candidate) for candidate in K3_CANDIDATES]
    projections: list[dict[str, Any]] = []
    for plan in plans:
        for concurrency in (1, 4, 16, 64):
            single = float(plan["projected_single_stream_tokens_per_second"])
            pipeline = int(plan["pipeline_stage_count"])
            occupancy = min(1.0, concurrency / max(pipeline, 1))
            aggregate = single * min(concurrency, pipeline)
            projections.append(
                {
                    "classification": "k3_checkpoint_projection",
                    "plan": plan["plan"],
                    "concurrency": concurrency,
                    "projected_single_stream_tokens_per_second": single,
                    "projected_aggregate_tokens_per_second": aggregate,
                    "pipeline_occupancy": occupancy,
                    "physical_measurement": False,
                    "required_wording": K3_REQUIRED_WORDING,
                }
            )
    return plans, projections
