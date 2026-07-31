"""Budgeted one-layer Qwen3-30B-A3B expert-parallel measurement."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from swarm_inference.microsharding.dense import numerical_error_metrics
from swarm_inference.microsharding.moe import (
    ExpertParallelMoE,
    MoEFixtureState,
    TinyMoEConfig,
)


@dataclass(frozen=True, slots=True)
class RealMoEDownloadPlan:
    model_id: str
    revision: str
    selected_layer: int
    required_files: tuple[str, ...]
    required_file_count: int
    required_download_bytes: int
    selected_layer_tensor_bytes: int
    unrelated_bytes_forced_by_file_co_location: int
    maximum_download_bytes: int
    within_budget: bool
    config_path: str
    index_path: str
    selected_tensor_names: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "selected_layer": self.selected_layer,
            "required_files": list(self.required_files),
            "required_file_count": self.required_file_count,
            "required_download_bytes": self.required_download_bytes,
            "selected_layer_tensor_bytes": self.selected_layer_tensor_bytes,
            "unrelated_bytes_forced_by_file_co_location": (
                self.unrelated_bytes_forced_by_file_co_location
            ),
            "maximum_download_bytes": self.maximum_download_bytes,
            "within_budget": self.within_budget,
            "config_path": self.config_path,
            "index_path": self.index_path,
            "selected_tensor_count": len(self.selected_tensor_names),
            "selected_tensor_names": list(self.selected_tensor_names),
        }


def _qwen_moe_tensor_bytes(name: str, config: dict[str, Any]) -> int:
    width = 2 if str(config.get("torch_dtype", "bfloat16")) in {"bfloat16", "float16"} else 4
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // int(config["num_attention_heads"]))
    query_width = int(config["num_attention_heads"]) * head_dim
    kv_width = int(config["num_key_value_heads"]) * head_dim
    expert_intermediate = int(config["moe_intermediate_size"])
    num_experts = int(config["num_experts"])
    if ".mlp.experts." in name:
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return expert_intermediate * hidden * width
        if name.endswith("down_proj.weight"):
            return hidden * expert_intermediate * width
    if name.endswith(".mlp.gate.weight"):
        return num_experts * hidden * width
    if name.endswith("self_attn.q_proj.weight"):
        return query_width * hidden * width
    if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
        return kv_width * hidden * width
    if name.endswith("self_attn.o_proj.weight"):
        return hidden * query_width * width
    if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
        return head_dim * width
    if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
        return hidden * width
    shared_size = config.get("shared_expert_intermediate_size")
    if ".shared_expert." in name and shared_size is not None:
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return int(shared_size) * hidden * width
        if name.endswith("down_proj.weight"):
            return hidden * int(shared_size) * width
    raise ValueError(f"cannot derive real MoE tensor bytes for {name}")


def inspect_real_moe_download(
    *,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    revision: str | None = None,
    selected_layer: int = 24,
    maximum_download_gib: float = 25.0,
    cache_dir: Path | None = None,
) -> RealMoEDownloadPlan:
    """Resolve metadata and calculate the exact file-co-location cost first."""

    from huggingface_hub import hf_hub_download, model_info

    info = model_info(model_id, revision=revision, files_metadata=True)
    exact_revision = info.sha
    if exact_revision is None:
        raise ValueError(f"no immutable revision resolved for {model_id}")
    cache_value = str(cache_dir) if cache_dir is not None else None
    config_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            revision=exact_revision,
            cache_dir=cache_value,
        )
    )
    index_path = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            revision=exact_revision,
            cache_dir=cache_value,
        )
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    prefix = f"model.layers.{selected_layer}."
    names = tuple(sorted(name for name in weight_map if name.startswith(prefix)))
    if not names:
        raise ValueError(f"selected MoE layer {selected_layer} has no tensors")
    required_files = tuple(sorted({str(weight_map[name]) for name in names}))
    sizes = {
        sibling.rfilename: int(sibling.size)
        for sibling in (info.siblings or [])
        if sibling.size is not None
    }
    missing_sizes = [name for name in required_files if name not in sizes]
    if missing_sizes:
        raise ValueError(f"official metadata omitted file sizes: {missing_sizes}")
    required_bytes = sum(sizes[name] for name in required_files)
    selected_bytes = sum(_qwen_moe_tensor_bytes(name, config) for name in names)
    maximum_bytes = int(maximum_download_gib * 1024**3)
    return RealMoEDownloadPlan(
        model_id=model_id,
        revision=exact_revision,
        selected_layer=selected_layer,
        required_files=required_files,
        required_file_count=len(required_files),
        required_download_bytes=required_bytes,
        selected_layer_tensor_bytes=selected_bytes,
        unrelated_bytes_forced_by_file_co_location=required_bytes - selected_bytes,
        maximum_download_bytes=maximum_bytes,
        within_budget=required_bytes <= maximum_bytes,
        config_path=str(config_path),
        index_path=str(index_path),
        selected_tensor_names=names,
    )


def download_real_moe_layer_files(
    plan: RealMoEDownloadPlan,
    *,
    cache_dir: Path | None = None,
) -> list[Path]:
    if not plan.within_budget:
        raise ValueError(
            f"required download {plan.required_download_bytes} exceeds budget "
            f"{plan.maximum_download_bytes}"
        )
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(
                repo_id=plan.model_id,
                filename=name,
                revision=plan.revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        )
        for name in plan.required_files
    ]


def _load_layer_state(plan: RealMoEDownloadPlan, files: list[Path]) -> dict[str, torch.Tensor]:
    selected = set(plan.selected_tensor_names)
    state: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118
                if name in selected:
                    state[name] = handle.get_tensor(name)
    if set(state) != selected:
        raise ValueError(
            f"downloaded files do not cover selected layer: missing={sorted(selected - set(state))}"
        )
    return state


def _causal_mask(
    hidden: torch.Tensor,
) -> torch.Tensor:
    sequence = hidden.shape[1]
    allowed = torch.arange(sequence, device=hidden.device).unsqueeze(0) <= torch.arange(
        sequence, device=hidden.device
    ).unsqueeze(1)
    mask = torch.full(
        (sequence, sequence),
        torch.finfo(hidden.dtype).min,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    mask.masked_fill_(allowed, 0)
    return mask[None, None, :, :]


@torch.inference_mode()
def run_real_moe_layer_measurement(
    plan: RealMoEDownloadPlan,
    files: list[Path],
    *,
    expert_parallel_degrees: list[int],
    expert_tensor_degrees: list[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sequence_length: int = 8,
    atol: float = 0.02,
    minimum_cosine_similarity: float = 0.999,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run a full reference, then rank-local experts with a common attention path."""

    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    config_payload = json.loads(Path(plan.config_path).read_text(encoding="utf-8"))
    config = Qwen3MoeConfig.from_dict(config_payload)
    config._attn_implementation = "eager"
    source = _load_layer_state(plan, files)
    prefix = f"model.layers.{plan.selected_layer}."
    local_state = {name.removeprefix(prefix): value for name, value in source.items()}
    target_device = torch.device(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6006)
    hidden = torch.randn(
        (1, sequence_length, int(config.hidden_size)), generator=generator, dtype=torch.float32
    ).to(device=target_device, dtype=dtype)
    position_ids = torch.arange(sequence_length, device=target_device).unsqueeze(0)
    rotary = Qwen3MoeRotaryEmbedding(config=config).to(target_device)
    cos, sin = rotary(hidden, position_ids)
    mask = _causal_mask(hidden)
    reference_layer = Qwen3MoeDecoderLayer(config, plan.selected_layer).to(
        device=target_device, dtype=dtype
    )
    incompatibility = reference_layer.load_state_dict(local_state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ValueError(f"strict real MoE reference load failed: {incompatibility}")
    reference_layer.eval()
    normalised_attention = reference_layer.input_layernorm(hidden)
    attention_output, _ = reference_layer.self_attn(
        hidden_states=normalised_attention,
        position_embeddings=(cos, sin),
        attention_mask=mask,
        position_ids=position_ids,
    )
    post_attention = hidden + attention_output
    expert_input = reference_layer.post_attention_layernorm(post_attention)
    reference_expert, router_logits = reference_layer.mlp(expert_input)
    reference_output = post_attention + reference_expert
    probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    reference_weights, reference_indices = torch.topk(
        probabilities, int(config.num_experts_per_tok), dim=-1
    )
    if bool(config.norm_topk_prob):
        reference_weights = reference_weights / reference_weights.sum(dim=-1, keepdim=True)
    reference_cpu = {
        "post_attention": post_attention.cpu(),
        "expert_input": expert_input.cpu(),
        "expert_output": reference_expert.cpu(),
        "output": reference_output.cpu(),
        "router_logits": router_logits.cpu(),
        "routing_weights": reference_weights.cpu(),
        "selected_experts": reference_indices.cpu(),
    }
    del reference_layer, rotary, normalised_attention, attention_output
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()

    experts = {
        expert_id: (
            source[f"{prefix}mlp.experts.{expert_id}.gate_proj.weight"],
            source[f"{prefix}mlp.experts.{expert_id}.up_proj.weight"],
            source[f"{prefix}mlp.experts.{expert_id}.down_proj.weight"],
        )
        for expert_id in range(int(config.num_experts))
    }
    state = MoEFixtureState(
        router=source[f"{prefix}mlp.gate.weight"],
        experts=experts,
        input_norm=torch.ones(int(config.hidden_size), dtype=torch.float32),
        shared_expert=None,
    )
    fixture_config = TinyMoEConfig(
        hidden_size=int(config.hidden_size),
        expert_intermediate_size=int(config.moe_intermediate_size),
        num_experts=int(config.num_experts),
        top_k=int(config.num_experts_per_tok),
        norm_topk_prob=bool(config.norm_topk_prob),
        shared_expert_intermediate_size=None,
        rms_norm_eps=float(config.rms_norm_eps),
    )
    results: list[dict[str, Any]] = []
    routing_trace: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    for token_index, (indices, weights) in enumerate(
        zip(reference_indices.tolist(), reference_weights.tolist(), strict=True)
    ):
        routing_trace.append(
            {
                "classification": "real_moe_layer_measurement",
                "token_index": token_index,
                "selected_experts": indices,
                "routing_weights": weights,
            }
        )
    for ep_degree in expert_parallel_degrees:
        if ep_degree > fixture_config.num_experts:
            continue
        for etp_degree in expert_tensor_degrees:
            if fixture_config.expert_intermediate_size % etp_degree:
                continue
            micro = ExpertParallelMoE(
                fixture_config,
                state,
                expert_parallel_degree=ep_degree,
                expert_tensor_parallel_degree=etp_degree,
                ownership_strategy="contiguous",
                device=target_device,
                dtype=dtype,
                apply_input_norm=False,
                add_residual=False,
            )
            measured = micro(reference_cpu["expert_input"].to(target_device))
            micro_output = reference_cpu["post_attention"].to(target_device) + measured.output
            expert_errors = numerical_error_metrics(
                reference_cpu["expert_output"], measured.output.cpu()
            )
            output_errors = numerical_error_metrics(reference_cpu["output"], micro_output.cpu())
            router_errors = numerical_error_metrics(
                reference_cpu["router_logits"], measured.router_logits.cpu()
            )
            routing_match = torch.equal(
                reference_cpu["selected_experts"], measured.selected_experts.cpu()
            )
            routing_weight_errors = numerical_error_metrics(
                reference_cpu["routing_weights"], measured.routing_weights.cpu()
            )
            memory = micro.memory_report()
            passed = (
                routing_match
                and output_errors["maximum_absolute_error"] <= atol
                and output_errors["cosine_similarity"] >= minimum_cosine_similarity
                and output_errors["nan_count"] == 0
                and output_errors["inf_count"] == 0
                and memory["status"] == "PASS"
            )
            results.append(
                {
                    "classification": "real_moe_layer_measurement",
                    "model_id": plan.model_id,
                    "revision": plan.revision,
                    "layer_id": plan.selected_layer,
                    "expert_parallel_degree": ep_degree,
                    "expert_tensor_parallel_degree": etp_degree,
                    "status": "PASS" if passed else "FAIL",
                    "router_indices_exact": routing_match,
                    "router_maximum_absolute_error": router_errors["maximum_absolute_error"],
                    "routing_weight_maximum_absolute_error": routing_weight_errors[
                        "maximum_absolute_error"
                    ],
                    "expert_maximum_absolute_error": expert_errors["maximum_absolute_error"],
                    "final_maximum_absolute_error": output_errors["maximum_absolute_error"],
                    "final_mean_absolute_error": output_errors["mean_absolute_error"],
                    "final_cosine_similarity": output_errors["cosine_similarity"],
                    "selected_rank_fanout": measured.metrics["selected_rank_fanout"],
                    "dispatch_bytes": measured.metrics["dispatch_bytes"],
                    "return_bytes": measured.metrics["return_bytes"],
                    "tokens_dispatched_per_rank": measured.metrics["tokens_dispatched_per_rank"],
                    "active_ranks_per_token": measured.metrics["active_ranks_per_token"],
                    "routing_time_ms": measured.metrics["routing_time_ms"],
                    "packing_time_ms": measured.metrics["packing_time_ms"],
                    "all_to_all_time_ms": measured.metrics["all_to_all_time_ms"],
                    "expert_combination_time_ms": measured.metrics["expert_combination_time_ms"],
                    "maximum_local_expert_compute_time_ms": max(
                        measured.metrics["local_expert_compute_time_ms"].values(),
                        default=0.0,
                    ),
                    "expert_imbalance": measured.metrics["expert_imbalance"],
                    "idle_rank_fraction": measured.metrics["idle_rank_fraction"],
                    "same_gpu_wall_clock_ms": measured.metrics["same_gpu_wall_clock_ms"],
                    "maximum_expert_bytes_per_rank": memory["maximum_expert_bytes_per_rank"],
                    "attention_execution_mode": "replicated_common_component",
                    "physical_network_measured": False,
                }
            )
            partition_rows.extend(
                {
                    "expert_parallel_degree": ep_degree,
                    "expert_tensor_parallel_degree": etp_degree,
                    **row,
                }
                for row in memory["ranks"]
            )
            del micro, measured, micro_output
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
    return (
        results,
        {
            "model_id": plan.model_id,
            "revision": plan.revision,
            "layer_id": plan.selected_layer,
            "mode": "expert_parallel",
            "router_mode": "replicated",
            "router_replicated": True,
            "routed_expert_count": int(config.num_experts),
            "top_k": int(config.num_experts_per_tok),
            "hidden_size": int(config.hidden_size),
            "expert_intermediate_size": int(config.moe_intermediate_size),
            "weight_dtype": str(dtype).removeprefix("torch."),
            "weight_dtype_bytes": torch.empty((), dtype=dtype).element_size(),
            "source_tensor_bytes": plan.selected_layer_tensor_bytes,
            "attention_execution_mode": "replicated_common_component",
            "full_checkpoint_loaded": False,
            "end_to_end_model_inference_claimed": False,
            "partition_rows": partition_rows,
        },
        routing_trace,
    )
