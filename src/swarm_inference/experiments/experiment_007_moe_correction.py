"""Matched-path CPU expert benchmark for the Experiment 007 correction run."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from swarm_inference.experiments.cpu_moe import (
    CpuExpertCache,
    ExpertWeights,
    StoredExpert,
    _stored_expert_call,
)
from swarm_inference.microsharding.real_moe import RealMoEDownloadPlan, _load_layer_state

CANONICAL_EXECUTOR_ID = "experiment-007-canonical-moe-placement-v1"
T = TypeVar("T")
ExpertBackend = Literal["cuda", "cpu"]
CorrectionExpertFormat = Literal["bfloat16", "int8", "four_bit"]
CorrectionPlacementPolicy = Literal[
    "coldest_experts_on_cpu",
    "hottest_experts_on_cpu",
    "random_experts_on_cpu",
    "load_balanced_experts_on_cpu",
    "frequency_band_experts_on_cpu",
]


@dataclass(frozen=True, slots=True)
class MoEExecutionPlan:
    layer_id: int
    model_revision: str
    router_backend: str
    shared_expert_backend: str
    expert_backend_by_id: dict[int, ExpertBackend]
    expert_format_by_id: dict[int, str]
    batch_size: int
    token_count: int
    top_k: int
    dtype: str
    execution_profile: str
    executor_id: str = CANONICAL_EXECUTOR_ID

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoutingCorpus:
    post_attention: torch.Tensor
    router_logits: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    manifest: dict[str, Any]

    @property
    def token_count(self) -> int:
        return int(self.post_attention.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.selected_experts.shape[1])


@dataclass(slots=True)
class MoEModelState:
    plan: RealMoEDownloadPlan
    config: Any
    post_attention_norm_weight: torch.Tensor
    router_weight: torch.Tensor
    experts: dict[int, ExpertWeights]

    @property
    def num_experts(self) -> int:
        return int(self.config.num_experts)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def top_k(self) -> int:
        return int(self.config.num_experts_per_tok)

    @property
    def rms_norm_eps(self) -> float:
        return float(self.config.rms_norm_eps)


@dataclass(slots=True)
class PreparedPlacement:
    gpu_experts: dict[int, ExpertWeights]
    cpu_cache: CpuExpertCache | None
    gpu_weight_bytes: int
    gpu_memory_allocated_bytes: int
    cpu_weight_bytes: int
    quantisation_ms: float
    maximum_quantised_weight_error: float

    def release(self) -> None:
        self.gpu_experts.clear()
        if self.cpu_cache is not None:
            self.cpu_cache.entries.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@dataclass(slots=True)
class ExecutionObservation:
    output: torch.Tensor | None
    expert_output: torch.Tensor | None
    selected_experts: torch.Tensor | None
    routing_weights: torch.Tensor | None
    timings: dict[str, float]
    cpu_expert_calls: int
    gpu_expert_calls: int
    bytes_gpu_to_cpu: int
    bytes_cpu_to_gpu: int
    no_nan: bool
    no_inf: bool


@dataclass(slots=True)
class RepeatStatistics:
    repeats: int
    total_measured_repeats: int
    measurement_epochs: int
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    standard_deviation_ms: float
    coefficient_of_variation: float
    timing_medians: dict[str, float]
    raw_total_layer_ms: list[float] = field(default_factory=list)
    unstable_epoch_coefficients_of_variation: list[float] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_routing_corpus_hash(
    tensors: dict[str, torch.Tensor],
    *,
    model_revision: str,
    layer_id: int,
    seed: int,
) -> str:
    """Hash corpus meaning rather than nondeterministically ordered safetensors metadata."""

    digest = hashlib.sha256()
    identity = {
        "format": "experiment-007-routing-corpus-v1",
        "layer_id": layer_id,
        "model_revision": model_revision,
        "seed": seed,
    }
    digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        descriptor = {
            "dtype": str(tensor.dtype),
            "name": name,
            "shape": list(tensor.shape),
        }
        digest.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        raw = tensor.view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 4 * 1024 * 1024):
            digest.update(raw[start : start + 4 * 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def load_moe_model_state(plan: RealMoEDownloadPlan, files: list[Path]) -> MoEModelState:
    from transformers import Qwen3MoeConfig

    payload = json.loads(Path(plan.config_path).read_text(encoding="utf-8"))
    config = Qwen3MoeConfig.from_dict(payload)
    config._attn_implementation = "eager"
    source = _load_layer_state(plan, files)
    prefix = f"model.layers.{plan.selected_layer}."
    experts = {
        expert_id: (
            source[f"{prefix}mlp.experts.{expert_id}.gate_proj.weight"].cpu().contiguous(),
            source[f"{prefix}mlp.experts.{expert_id}.up_proj.weight"].cpu().contiguous(),
            source[f"{prefix}mlp.experts.{expert_id}.down_proj.weight"].cpu().contiguous(),
        )
        for expert_id in range(int(config.num_experts))
    }
    return MoEModelState(
        plan=plan,
        config=config,
        post_attention_norm_weight=source[f"{prefix}post_attention_layernorm.weight"]
        .cpu()
        .contiguous(),
        router_weight=source[f"{prefix}mlp.gate.weight"].cpu().contiguous(),
        experts=experts,
    )


def _load_colocated_layer_state(files: list[Path], layer_id: int) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_id}."
    state: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118
                if name.startswith(prefix):
                    state[name.removeprefix(prefix)] = handle.get_tensor(name)
    if not state:
        raise RuntimeError(f"validated MoE shard contains no tensors for layer {layer_id}")
    return state


@torch.inference_mode()
def _run_probe_layer(
    model: MoEModelState,
    files: list[Path],
    *,
    layer_id: int,
    hidden_chunks: list[torch.Tensor],
    attention_only: bool,
) -> list[torch.Tensor]:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16
    local_state = _load_colocated_layer_state(files, layer_id)
    layer = Qwen3MoeDecoderLayer(model.config, layer_id).to(device=device, dtype=dtype)
    incompatible = layer.load_state_dict(local_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict real MoE probe layer {layer_id} load failed: {incompatible}")
    layer.eval()
    rotary = Qwen3MoeRotaryEmbedding(config=model.config).to(device)
    outputs: list[torch.Tensor] = []
    for hidden_cpu in hidden_chunks:
        hidden = hidden_cpu.to(device=device, dtype=dtype)
        sequence_length = int(hidden.shape[1])
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        cos, sin = rotary(hidden, position_ids)
        minimum = torch.finfo(dtype).min
        causal = torch.full((sequence_length, sequence_length), minimum, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)[None, None, :, :]
        normalised = layer.input_layernorm(hidden)
        attention_output, _ = layer.self_attn(
            hidden_states=normalised,
            position_embeddings=(cos, sin),
            attention_mask=causal,
            position_ids=position_ids,
        )
        post_attention = hidden + attention_output
        if attention_only:
            output = post_attention
        else:
            expert_output, _ = layer.mlp(layer.post_attention_layernorm(post_attention))
            output = post_attention + expert_output
        outputs.append(output.cpu().contiguous())
        if not attention_only:
            del expert_output
        del hidden, normalised, attention_output, post_attention, output, cos, sin, causal
    del layer, rotary, local_state
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def _probe_prompts() -> list[tuple[str, str]]:
    return [
        ("general_text", "Explain why measurement denominators matter in systems research."),
        ("code", "Write a bounded queue with cancellation and deterministic accounting."),
        ("reasoning", "If two workers finish at different times, derive their shared-window rate."),
        (
            "long_form",
            "Describe heterogeneous inference placement and its failure modes in detail.",
        ),
    ]


@torch.inference_mode()
def build_routing_corpus(
    model: MoEModelState,
    files: list[Path],
    *,
    token_count: int,
    seed: int,
    output_path: Path,
) -> RoutingCorpus:
    """Build a frozen corpus from real co-located layers and the selected layer's router.

    The validated shard contains complete layers 22 through 24. Prompt-hash-conditioned probes
    traverse the two complete predecessor layers before becoming inputs to layer 24, so the saved
    tensors are real layer inputs and router outputs. They are not claimed to be full-model prompt
    activations because layers 0 through 21 are not present in the validated artifact.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("the matched MoE correction benchmark requires CUDA")
    prompts = _probe_prompts()
    sequence_length = 256
    chunks = math.ceil(token_count / sequence_length)
    hidden_chunks: list[torch.Tensor] = []
    prompt_hashes: list[dict[str, Any]] = []
    for chunk_index in range(chunks):
        workload_class, prompt = prompts[chunk_index % len(prompts)]
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompt_hashes.append(
            {
                "workload_class": workload_class,
                "prompt_sha256": prompt_hash,
                "chunk_index": chunk_index,
            }
        )
        derived_seed = seed + chunk_index + int(prompt_hash[:8], 16)
        generator = torch.Generator(device="cpu").manual_seed(derived_seed)
        hidden_chunks.append(
            torch.randn(
                (1, sequence_length, model.hidden_size),
                generator=generator,
                dtype=torch.float32,
            ).to(dtype=torch.bfloat16)
        )
    predecessor_layers = [model.plan.selected_layer - 2, model.plan.selected_layer - 1]
    for predecessor_layer in predecessor_layers:
        hidden_chunks = _run_probe_layer(
            model,
            files,
            layer_id=predecessor_layer,
            hidden_chunks=hidden_chunks,
            attention_only=False,
        )
    selected_layer_inputs = torch.cat(
        [chunk.reshape(-1, model.hidden_size) for chunk in hidden_chunks], dim=0
    )[:token_count].contiguous()
    post_attention_chunks = _run_probe_layer(
        model,
        files,
        layer_id=model.plan.selected_layer,
        hidden_chunks=hidden_chunks,
        attention_only=True,
    )
    post_attention = torch.cat(
        [chunk.reshape(-1, model.hidden_size) for chunk in post_attention_chunks], dim=0
    )[:token_count].contiguous()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    normed = _rms_norm(
        post_attention.to(device=device),
        model.post_attention_norm_weight.to(device=device, dtype=dtype),
        model.rms_norm_eps,
    )
    router_logits_gpu = F.linear(normed, model.router_weight.to(device=device, dtype=dtype))
    routing_weights_gpu, selected_gpu = _router_topk(
        router_logits_gpu,
        top_k=model.top_k,
        normalise_topk=bool(model.config.norm_topk_prob),
    )
    torch.cuda.synchronize()
    router_logits = router_logits_gpu.cpu().contiguous()
    selected = selected_gpu.cpu().contiguous()
    routing_weights = routing_weights_gpu.cpu().contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_tensors = {
        "post_attention": post_attention,
        "selected_layer_input": selected_layer_inputs,
        "router_logits": router_logits,
        "selected_experts": selected,
        "routing_weights": routing_weights,
    }
    save_file(
        corpus_tensors,
        str(output_path),
        metadata={
            "model_revision": model.plan.revision,
            "layer_id": str(model.plan.selected_layer),
            "seed": str(seed),
            "source": "real_predecessor_layer_probe_hidden_states",
        },
    )
    corpus_hash = _canonical_routing_corpus_hash(
        corpus_tensors,
        model_revision=model.plan.revision,
        layer_id=model.plan.selected_layer,
        seed=seed,
    )
    artifact_file_hash = sha256_file(output_path)
    counts = Counter(int(item) for item in selected.flatten().tolist())
    manifest = {
        "status": "PASS",
        "classification": "measured_cuda",
        "model_id": model.plan.model_id,
        "model_revision": model.plan.revision,
        "layer_id": model.plan.selected_layer,
        "hidden_state_dtype": "bfloat16",
        "hidden_state_shape": list(post_attention.shape),
        "selected_layer_input_shape": list(selected_layer_inputs.shape),
        "router_logits_shape": list(router_logits.shape),
        "selected_expert_shape": list(selected.shape),
        "routing_weight_shape": list(routing_weights.shape),
        "token_count": token_count,
        "routed_expert_call_count": int(selected.numel()),
        "top_k": model.top_k,
        "random_seed": seed,
        "corpus_path": str(output_path.resolve()),
        "corpus_sha256": corpus_hash,
        "corpus_hash": corpus_hash,
        "corpus_hash_definition": "canonical_sorted_tensor_content_v1",
        "artifact_file_sha256": artifact_file_hash,
        "prompt_hashes": prompt_hashes,
        "workload_classes": sorted({item[0] for item in prompts}),
        "source_hidden_state_method": (
            "deterministic prompt-hash-conditioned probes passed through complete real layers "
            f"{predecessor_layers}, then the selected layer input norm, attention, residual, "
            "post-attention norm, and router"
        ),
        "real_predecessor_layers": predecessor_layers,
        "selected_layer_inputs_are_real_layer_outputs": True,
        "full_model_prompt_activation_capture": False,
        "full_model_capture_reason": (
            "validated shard contains complete layers 22 through 24, not layers 0 through 21"
        ),
        "natural_router_outputs_unmodified": True,
        "expert_frequency_histogram": {
            str(key): counts.get(key, 0) for key in range(model.num_experts)
        },
    }
    del hidden_chunks, post_attention_chunks, normed, router_logits_gpu
    del routing_weights_gpu, selected_gpu
    gc.collect()
    torch.cuda.empty_cache()
    return RoutingCorpus(
        post_attention=post_attention,
        router_logits=router_logits,
        selected_experts=selected,
        routing_weights=routing_weights,
        manifest=manifest,
    )


def select_cpu_experts_corrected(
    policy: CorrectionPlacementPolicy,
    *,
    count: int,
    num_experts: int,
    routing_counts: dict[int, int],
    seed: int,
) -> list[int]:
    if not 1 <= count < num_experts:
        raise ValueError("CPU expert count must be non-zero and below the complete expert set")
    expert_ids = list(range(num_experts))
    if policy == "coldest_experts_on_cpu":
        ordered = sorted(expert_ids, key=lambda item: (routing_counts.get(item, 0), item))
    elif policy == "hottest_experts_on_cpu":
        ordered = sorted(expert_ids, key=lambda item: (-routing_counts.get(item, 0), item))
    elif policy == "random_experts_on_cpu":
        ordered = expert_ids.copy()
        random.Random(seed + count).shuffle(ordered)
    elif policy == "load_balanced_experts_on_cpu":
        mean = sum(routing_counts.values()) / max(num_experts, 1)
        ordered = sorted(
            expert_ids, key=lambda item: (abs(routing_counts.get(item, 0) - mean), item)
        )
    elif policy == "frequency_band_experts_on_cpu":
        by_frequency = sorted(expert_ids, key=lambda item: (routing_counts.get(item, 0), item))
        if count == 1:
            ordered = [by_frequency[len(by_frequency) // 2]]
        else:
            positions = [
                round(index * (len(by_frequency) - 1) / (count - 1)) for index in range(count)
            ]
            selected = [by_frequency[position] for position in positions]
            ordered = selected + [item for item in by_frequency if item not in selected]
    else:
        raise ValueError(f"unsupported corrected placement policy {policy!r}")
    return sorted(ordered[:count])


def placement_dispatch_metrics(
    selected_experts: torch.Tensor, cpu_expert_ids: list[int]
) -> dict[str, Any]:
    owned = set(cpu_expert_ids)
    counts = Counter(int(item) for item in selected_experts.flatten().tolist())
    dispatch_count = sum(counts.get(expert_id, 0) for expert_id in owned)
    unique_selected = sum(1 for expert_id in owned if counts.get(expert_id, 0) > 0)
    return {
        "expected_cpu_dispatch_count": dispatch_count,
        "expected_cpu_dispatch_fraction": dispatch_count / max(selected_experts.numel(), 1),
        "unique_cpu_experts_selected": unique_selected,
    }


def make_execution_plan(
    model: MoEModelState,
    corpus: RoutingCorpus,
    *,
    cpu_expert_ids: list[int],
    weight_format: CorrectionExpertFormat,
    execution_profile: str,
) -> MoEExecutionPlan:
    owned = set(cpu_expert_ids)
    return MoEExecutionPlan(
        layer_id=model.plan.selected_layer,
        model_revision=model.plan.revision,
        router_backend="cuda",
        shared_expert_backend="not_present_in_qwen3_30b_a3b",
        expert_backend_by_id={
            expert_id: "cpu" if expert_id in owned else "cuda"
            for expert_id in range(model.num_experts)
        },
        expert_format_by_id={
            expert_id: weight_format if expert_id in owned else "bfloat16"
            for expert_id in range(model.num_experts)
        },
        batch_size=1,
        token_count=corpus.token_count,
        top_k=corpus.top_k,
        dtype="bfloat16",
        execution_profile=execution_profile,
    )


def validate_matched_plans(baseline: MoEExecutionPlan, hybrid: MoEExecutionPlan) -> None:
    if baseline.executor_id != hybrid.executor_id:
        raise ValueError("different executor IDs invalidate the matched benchmark")
    comparable_fields = (
        "layer_id",
        "model_revision",
        "router_backend",
        "shared_expert_backend",
        "batch_size",
        "token_count",
        "top_k",
        "dtype",
    )
    for name in comparable_fields:
        if getattr(baseline, name) != getattr(hybrid, name):
            raise ValueError(f"matched MoE plan field differs: {name}")
    if any(value != "cuda" for value in baseline.expert_backend_by_id.values()):
        raise ValueError("matched baseline must place every expert on CUDA")
    for expert_id, backend in hybrid.expert_backend_by_id.items():
        baseline_format = baseline.expert_format_by_id[expert_id]
        hybrid_format = hybrid.expert_format_by_id[expert_id]
        if backend == "cuda" and baseline_format != hybrid_format:
            raise ValueError("non-offloaded expert formats differ")
        if backend == "cpu" and hybrid_format not in {"bfloat16", "int8", "four_bit"}:
            raise ValueError("offloaded expert format differs without a separate diagnostic")
        if backend == "cpu" and hybrid_format == "bfloat16" and baseline_format != hybrid_format:
            raise ValueError("matched BF16 offloaded expert format differs")


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    source_dtype = hidden.dtype
    values = hidden.float()
    variance = values.pow(2).mean(-1, keepdim=True)
    normalised = values * torch.rsqrt(variance + eps)
    return (weight.float() * normalised).to(source_dtype)


def _router_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    normalise_topk: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the one canonical routing implementation used by corpus and executor."""

    probabilities = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, top_k, dim=-1)
    if normalise_topk:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(torch.bfloat16), selected


def _cuda_section(operation: Callable[[], T]) -> tuple[T, float]:
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


class CanonicalMoEExecutor:
    """One executor used by all-GPU and every hybrid placement."""

    def __init__(self, model: MoEModelState, corpus: RoutingCorpus) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("canonical MoE executor requires CUDA")
        self.model = model
        self.corpus = corpus
        self.device = torch.device("cuda")
        self.post_attention = corpus.post_attention.to(self.device, dtype=torch.bfloat16)
        self.norm_weight = model.post_attention_norm_weight.to(self.device, dtype=torch.bfloat16)
        self.router_weight = model.router_weight.to(self.device, dtype=torch.bfloat16)
        self.expected_selected = corpus.selected_experts.to(self.device)
        self.expected_weights = corpus.routing_weights.to(self.device, dtype=torch.bfloat16)
        self.assignment_indices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for expert_id in range(model.num_experts):
            matches = torch.nonzero(self.expected_selected == expert_id, as_tuple=False)
            self.assignment_indices[expert_id] = (
                matches[:, 0].to(self.device),
                matches[:, 1].to(self.device),
            )

    def subset(self, token_indices: list[int]) -> CanonicalMoEExecutor:
        selected = torch.tensor(token_indices, dtype=torch.long)
        post_attention = self.corpus.post_attention.index_select(0, selected).contiguous()
        post_attention_gpu = post_attention.to(self.device, dtype=torch.bfloat16)
        normalised = _rms_norm(
            post_attention_gpu,
            self.norm_weight,
            self.model.rms_norm_eps,
        )
        router_logits_gpu = F.linear(normalised, self.router_weight)
        routing_weights_gpu, selected_experts_gpu = _router_topk(
            router_logits_gpu,
            top_k=self.model.top_k,
            normalise_topk=bool(self.model.config.norm_topk_prob),
        )
        torch.cuda.synchronize()
        subset_corpus = RoutingCorpus(
            post_attention=post_attention,
            router_logits=router_logits_gpu.cpu().contiguous(),
            selected_experts=selected_experts_gpu.cpu().contiguous(),
            routing_weights=routing_weights_gpu.cpu().contiguous(),
            manifest={
                **self.corpus.manifest,
                "mode": "controlled_coverage_diagnostic",
                "subset_router_recomputed_for_subset_shape": True,
            },
        )
        del normalised, post_attention_gpu, router_logits_gpu
        del routing_weights_gpu, selected_experts_gpu
        return CanonicalMoEExecutor(self.model, subset_corpus)

    def prepare(self, plan: MoEExecutionPlan) -> PreparedPlacement:
        if plan.executor_id != CANONICAL_EXECUTOR_ID:
            raise ValueError("execution plan does not target the canonical matched executor")
        cpu_ids = [
            expert_id
            for expert_id, backend in plan.expert_backend_by_id.items()
            if backend == "cpu"
        ]
        gpu_experts: dict[int, ExpertWeights] = {}
        for expert_id, source in self.model.experts.items():
            if expert_id not in cpu_ids:
                gpu_experts[expert_id] = cast(
                    ExpertWeights,
                    tuple(item.to(self.device, dtype=torch.bfloat16) for item in source),
                )
        cpu_cache: CpuExpertCache | None = None
        if cpu_ids:
            first_format = plan.expert_format_by_id[cpu_ids[0]]
            cache_format = {
                "bfloat16": "BF16",
                "int8": "INT8",
                "four_bit": "Q4",
            }[first_format]
            cpu_cache = CpuExpertCache(
                self.model.experts,
                capacity=len(cpu_ids),
                weight_format=cast(Any, cache_format),
            )
            cpu_cache.prefetch(cpu_ids)
        torch.cuda.synchronize()
        gpu_bytes = sum(
            int(item.numel() * item.element_size())
            for weights in gpu_experts.values()
            for item in weights
        )
        entries: list[StoredExpert] = (
            list(cpu_cache.entries.values()) if cpu_cache is not None else []
        )
        return PreparedPlacement(
            gpu_experts=gpu_experts,
            cpu_cache=cpu_cache,
            gpu_weight_bytes=gpu_bytes,
            gpu_memory_allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
            cpu_weight_bytes=sum(item.storage_bytes for item in entries),
            quantisation_ms=sum(item.quantisation_ms for item in entries),
            maximum_quantised_weight_error=max(
                (item.maximum_weight_error for item in entries), default=0.0
            ),
        )

    @torch.inference_mode()
    def execute(
        self,
        plan: MoEExecutionPlan,
        prepared: PreparedPlacement,
        *,
        capture_output: bool,
    ) -> ExecutionObservation:
        total_started = time.perf_counter()
        normalised, router_time = _cuda_section(
            lambda: _rms_norm(self.post_attention, self.norm_weight, self.model.rms_norm_eps)
        )
        router_logits, router_linear_time = _cuda_section(
            lambda: F.linear(normalised, self.router_weight)
        )

        def route() -> tuple[torch.Tensor, torch.Tensor]:
            return _router_topk(
                router_logits,
                top_k=self.model.top_k,
                normalise_topk=bool(self.model.config.norm_topk_prob),
            )

        routed, routing_topk_time = _cuda_section(route)
        routing_weights, selected = routed
        router_ids_equal = torch.equal(selected, self.expected_selected)
        if not router_ids_equal:
            raise RuntimeError("canonical executor router IDs differ from the frozen corpus")

        packed: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        def pack() -> None:
            for expert_id, (token_indices, slots) in self.assignment_indices.items():
                if token_indices.numel():
                    packed[expert_id] = (
                        token_indices,
                        slots,
                        normalised.index_select(0, token_indices),
                    )

        _, dispatch_pack_time = _cuda_section(pack)
        gpu_outputs: dict[int, torch.Tensor] = {}
        cpu_inputs: dict[int, torch.Tensor] = {}
        cpu_outputs: dict[int, torch.Tensor] = {}
        cpu_ids = {
            expert_id
            for expert_id, backend in plan.expert_backend_by_id.items()
            if backend == "cpu"
        }
        gpu_ids = set(packed) - cpu_ids

        def gpu_compute() -> None:
            for expert_id in sorted(gpu_ids):
                local_input = packed[expert_id][2]
                gate, up, down = prepared.gpu_experts[expert_id]
                gpu_outputs[expert_id] = F.linear(
                    F.silu(F.linear(local_input, gate)) * F.linear(local_input, up), down
                )

        _, gpu_expert_compute_time = _cuda_section(gpu_compute)
        bytes_gpu_to_cpu = 0
        transfer_started = time.perf_counter()
        for expert_id in sorted(cpu_ids & set(packed)):
            local_cpu = packed[expert_id][2].to(device="cpu", dtype=torch.bfloat16)
            cpu_inputs[expert_id] = local_cpu
            bytes_gpu_to_cpu += int(local_cpu.numel() * local_cpu.element_size())
        torch.cuda.synchronize()
        gpu_to_cpu_transfer_time = (time.perf_counter() - transfer_started) * 1000

        cpu_queue_time = 0.0
        cpu_compute_started = time.perf_counter()
        dequantisation_ms = 0.0
        if prepared.cpu_cache is None and cpu_inputs:
            raise RuntimeError("CPU expert inputs exist without a prepared CPU expert cache")
        for expert_id, local_cpu in cpu_inputs.items():
            assert prepared.cpu_cache is not None
            output, dequant_ms = _stored_expert_call(local_cpu, prepared.cpu_cache.get(expert_id))
            cpu_outputs[expert_id] = output
            dequantisation_ms += dequant_ms
        cpu_expert_compute_time = (time.perf_counter() - cpu_compute_started) * 1000

        bytes_cpu_to_gpu = 0
        transfer_started = time.perf_counter()
        for expert_id, output in cpu_outputs.items():
            gpu_outputs[expert_id] = output.to(device=self.device, dtype=torch.bfloat16)
            bytes_cpu_to_gpu += int(output.numel() * output.element_size())
        torch.cuda.synchronize()
        cpu_to_gpu_transfer_time = (time.perf_counter() - transfer_started) * 1000

        combined = torch.zeros_like(normalised)

        def combine() -> torch.Tensor:
            for expert_id in sorted(packed):
                token_indices, slots, _local = packed[expert_id]
                weighted = gpu_outputs[expert_id] * routing_weights[token_indices, slots, None]
                combined.index_add_(0, token_indices, weighted)
            return self.post_attention + combined

        output, combine_time = _cuda_section(combine)
        synchronisation_started = time.perf_counter()
        torch.cuda.synchronize()
        synchronisation_time = (time.perf_counter() - synchronisation_started) * 1000
        total_layer_time = (time.perf_counter() - total_started) * 1000
        cpu_calls = sum(int(self.assignment_indices[item][0].numel()) for item in cpu_ids)
        total_calls = int(self.expected_selected.numel())
        captured_output = output.cpu() if capture_output else None
        captured_expert = combined.cpu() if capture_output else None
        captured_selected = selected.cpu() if capture_output else None
        captured_weights = routing_weights.cpu() if capture_output else None
        if capture_output:
            assert captured_output is not None
            no_nan = not bool(torch.isnan(captured_output.float()).any())
            no_inf = not bool(torch.isinf(captured_output.float()).any())
        else:
            no_nan = True
            no_inf = True
        return ExecutionObservation(
            output=captured_output,
            expert_output=captured_expert,
            selected_experts=captured_selected,
            routing_weights=captured_weights,
            timings={
                "input_normalisation_time_ms": router_time,
                "router_time_ms": router_linear_time,
                "routing_topk_time_ms": routing_topk_time,
                "dispatch_pack_time_ms": dispatch_pack_time,
                "gpu_to_cpu_transfer_time_ms": gpu_to_cpu_transfer_time,
                "cpu_queue_time_ms": cpu_queue_time,
                "cpu_expert_compute_time_ms": cpu_expert_compute_time,
                "cpu_dequantisation_time_ms": dequantisation_ms,
                "cpu_to_gpu_transfer_time_ms": cpu_to_gpu_transfer_time,
                "gpu_expert_compute_time_ms": gpu_expert_compute_time,
                "shared_expert_time_ms": 0.0,
                "combine_time_ms": combine_time,
                "synchronisation_time_ms": synchronisation_time,
                "total_layer_time_ms": total_layer_time,
            },
            cpu_expert_calls=cpu_calls,
            gpu_expert_calls=total_calls - cpu_calls,
            bytes_gpu_to_cpu=bytes_gpu_to_cpu,
            bytes_cpu_to_gpu=bytes_cpu_to_gpu,
            no_nan=no_nan,
            no_inf=no_inf,
        )


def _coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def _stable_cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_elements: int = 1_048_576,
) -> float:
    """Compute a bounded cosine with float64 reductions over bounded chunks.

    A single float32 reduction over the 20M-element routing corpus can accumulate enough
    roundoff to report a cosine above one. Chunked float64 dot products keep the diagnostic
    meaningful without adding hundreds of megabytes of temporary double tensors.
    """

    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("cosine inputs must have identical shapes")
    dot_product = 0.0
    left_squared = 0.0
    right_squared = 0.0
    for start in range(0, left_flat.numel(), chunk_elements):
        end = min(start + chunk_elements, left_flat.numel())
        left_chunk = left_flat[start:end].to(dtype=torch.float64)
        right_chunk = right_flat[start:end].to(dtype=torch.float64)
        dot_product += float(torch.dot(left_chunk, right_chunk).item())
        left_squared += float(torch.dot(left_chunk, left_chunk).item())
        right_squared += float(torch.dot(right_chunk, right_chunk).item())
    denominator = math.sqrt(left_squared * right_squared)
    if denominator == 0.0:
        return 1.0 if left_squared == right_squared else 0.0
    return min(1.0, max(-1.0, dot_product / denominator))


def measure_repeated(
    executor: CanonicalMoEExecutor,
    plan: MoEExecutionPlan,
    prepared: PreparedPlacement,
    *,
    warmup_iterations: int,
    minimum_repeats: int,
    maximum_repeats: int,
    maximum_epochs: int,
    maximum_cv: float,
) -> tuple[RepeatStatistics, ExecutionObservation]:
    observations: list[ExecutionObservation] = []
    unstable_epoch_cvs: list[float] = []
    total_measured_repeats = 0
    measurement_epochs = 0
    for epoch in range(maximum_epochs):
        measurement_epochs = epoch + 1
        for _ in range(warmup_iterations):
            executor.execute(plan, prepared, capture_output=False)
        observations = []
        while len(observations) < maximum_repeats:
            observations.append(executor.execute(plan, prepared, capture_output=False))
            totals = [item.timings["total_layer_time_ms"] for item in observations]
            if (
                len(observations) >= minimum_repeats
                and _coefficient_of_variation(totals) <= maximum_cv
            ):
                break
        total_measured_repeats += len(observations)
        totals = [item.timings["total_layer_time_ms"] for item in observations]
        epoch_cv = _coefficient_of_variation(totals)
        if epoch_cv <= maximum_cv:
            break
        unstable_epoch_cvs.append(epoch_cv)
    totals = [item.timings["total_layer_time_ms"] for item in observations]
    timing_names = observations[0].timings.keys()
    stats = RepeatStatistics(
        repeats=len(observations),
        total_measured_repeats=total_measured_repeats,
        measurement_epochs=measurement_epochs,
        median_ms=statistics.median(totals),
        minimum_ms=min(totals),
        maximum_ms=max(totals),
        standard_deviation_ms=statistics.pstdev(totals),
        coefficient_of_variation=_coefficient_of_variation(totals),
        timing_medians={
            name: statistics.median([item.timings[name] for item in observations])
            for name in timing_names
        },
        raw_total_layer_ms=totals,
        unstable_epoch_coefficients_of_variation=unstable_epoch_cvs,
    )
    validation = executor.execute(plan, prepared, capture_output=True)
    return stats, validation


def correctness_metrics(
    baseline: ExecutionObservation,
    hybrid: ExecutionObservation,
    *,
    weight_format: CorrectionExpertFormat,
    atol: float,
    rtol: float,
    minimum_cosine_similarity: float,
) -> dict[str, Any]:
    if baseline.output is None or hybrid.output is None:
        raise ValueError("correctness comparison requires captured outputs")
    if baseline.expert_output is None or hybrid.expert_output is None:
        raise ValueError("correctness comparison requires captured expert outputs")
    difference = (hybrid.output.float() - baseline.output.float()).abs()
    expert_difference = (hybrid.expert_output.float() - baseline.expert_output.float()).abs()
    denominator = baseline.output.float().abs().clamp_min(1e-8)
    cosine = _stable_cosine_similarity(hybrid.output, baseline.output)
    ids_equal = bool(
        baseline.selected_experts is not None
        and hybrid.selected_experts is not None
        and torch.equal(baseline.selected_experts, hybrid.selected_experts)
    )
    if baseline.routing_weights is None or hybrid.routing_weights is None:
        weights_error = math.inf
    else:
        weights_error = float(
            (baseline.routing_weights.float() - hybrid.routing_weights.float()).abs().max().item()
        )
    within = bool(
        torch.allclose(hybrid.output.float(), baseline.output.float(), atol=atol, rtol=rtol)
    )
    passed = (
        ids_equal
        and weights_error <= max(atol, rtol)
        and within
        and cosine >= minimum_cosine_similarity
        and baseline.no_nan
        and baseline.no_inf
        and hybrid.no_nan
        and hybrid.no_inf
    )
    return {
        "output_correctness_passed": passed,
        "correctness_label": (
            (
                "numerically_equivalent_within_tolerance"
                if weight_format == "bfloat16"
                else "quantised_output_within_diagnostic_tolerance"
            )
            if passed
            else "outside_tolerance"
        ),
        "bitwise_exact": bool(torch.equal(hybrid.output, baseline.output)),
        "router_expert_ids_identical": ids_equal,
        "routing_weights_maximum_absolute_error": weights_error,
        "expert_output_maximum_absolute_error": float(expert_difference.max().item()),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "maximum_relative_error": float((difference / denominator).max().item()),
        "cosine_similarity": cosine,
        "cosine_accumulation_dtype": "float64_chunked",
        "nan_count": int(torch.isnan(hybrid.output.float()).sum().item()),
        "inf_count": int(torch.isinf(hybrid.output.float()).sum().item()),
        "atol": atol,
        "rtol": rtol,
        "minimum_cosine_similarity": minimum_cosine_similarity,
        "quantised": weight_format != "bfloat16",
        "downstream_greedy_token_identity": "not_available_layer_not_reintegrated",
    }


def controlled_coverage_indices(
    selected_experts: torch.Tensor,
    cpu_expert_ids: list[int],
    *,
    minimum_calls: int = 5,
) -> list[int]:
    chosen: set[int] = set()
    for expert_id in cpu_expert_ids:
        matches = torch.nonzero(selected_experts == expert_id, as_tuple=False)
        for token_index in matches[:minimum_calls, 0].tolist():
            chosen.add(int(token_index))
    return sorted(chosen)


def valid_positive_expert_result(
    row: dict[str, Any],
    *,
    minimum_dispatch_fraction: float,
    minimum_retained_fraction: float,
) -> bool:
    return bool(
        row.get("benchmark_mode") == "natural_routing"
        and row.get("weight_format") == "bfloat16"
        and int(row.get("cpu_expert_calls", 0)) > 0
        and float(row.get("cpu_dispatch_fraction", 0.0)) >= minimum_dispatch_fraction
        and int(row.get("gpu_memory_saved_bytes", 0)) > 0
        and bool(row.get("output_correctness_passed"))
        and bool(row.get("matched_baseline_used"))
        and float(row.get("throughput_retained_fraction", 0.0)) >= minimum_retained_fraction
        and float(row.get("coefficient_of_variation", math.inf)) <= 0.10
    )


def benchmark_matched_moe(
    model: MoEModelState,
    corpus: RoutingCorpus,
    *,
    policies: tuple[str, ...],
    expert_counts: tuple[int, ...],
    formats: tuple[str, ...],
    seed: int,
    warmup_iterations: int,
    repeats: int,
    maximum_repeats: int,
    maximum_variability_epochs: int,
    maximum_cv: float,
    minimum_dispatch_fraction: float,
    minimum_retained_fraction: float,
    atol: float,
    rtol: float,
    minimum_cosine_similarity: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    executor = CanonicalMoEExecutor(model, corpus)
    routing_counts = Counter(int(item) for item in corpus.selected_experts.flatten().tolist())
    result_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    controlled_rows: list[dict[str, Any]] = []
    all_gpu_plan = make_execution_plan(
        model,
        corpus,
        cpu_expert_ids=[],
        weight_format="bfloat16",
        execution_profile="natural_routing_all_gpu",
    )
    full_expert_bytes = sum(
        int(item.numel() * item.element_size())
        for weights in model.experts.values()
        for item in weights
    )
    for count in expert_counts:
        for policy_name in policies:
            policy = cast(CorrectionPlacementPolicy, policy_name)
            cpu_ids = select_cpu_experts_corrected(
                policy,
                count=count,
                num_experts=model.num_experts,
                routing_counts=dict(routing_counts),
                seed=seed,
            )
            dispatch = placement_dispatch_metrics(corpus.selected_experts, cpu_ids)
            inactive = int(dispatch["expected_cpu_dispatch_count"]) == 0
            baseline_prepared = executor.prepare(all_gpu_plan)
            try:
                baseline_stats, baseline_observation = measure_repeated(
                    executor,
                    all_gpu_plan,
                    baseline_prepared,
                    warmup_iterations=warmup_iterations,
                    minimum_repeats=repeats,
                    maximum_repeats=maximum_repeats,
                    maximum_epochs=maximum_variability_epochs,
                    maximum_cv=maximum_cv,
                )
                baseline_gpu_bytes = baseline_prepared.gpu_weight_bytes
                baseline_gpu_allocated = baseline_prepared.gpu_memory_allocated_bytes
            finally:
                baseline_prepared.release()
            baseline_tps = corpus.token_count / max(baseline_stats.median_ms / 1000, 1e-12)
            baseline_row = {
                "classification": "measured_cuda",
                "benchmark_mode": "natural_routing",
                "arm": "all_gpu",
                "placement_policy": policy_name,
                "cpu_expert_count": count,
                "cpu_expert_ids": [],
                "weight_format": "bfloat16",
                "executor_id": CANONICAL_EXECUTOR_ID,
                "matched_baseline_used": True,
                "token_count": corpus.token_count,
                "routed_expert_calls": int(corpus.selected_experts.numel()),
                "cpu_expert_calls": 0,
                "gpu_expert_calls": int(corpus.selected_experts.numel()),
                "cpu_dispatch_fraction": 0.0,
                "gpu_memory_saved_bytes": 0,
                "gpu_expert_weight_bytes": baseline_gpu_bytes,
                "gpu_memory_allocated_bytes": baseline_gpu_allocated,
                "cpu_memory_used_bytes": 0,
                "median_total_layer_ms": baseline_stats.median_ms,
                "minimum_total_layer_ms": baseline_stats.minimum_ms,
                "maximum_total_layer_ms": baseline_stats.maximum_ms,
                "standard_deviation_ms": baseline_stats.standard_deviation_ms,
                "coefficient_of_variation": baseline_stats.coefficient_of_variation,
                "measured_repeats": baseline_stats.repeats,
                "total_measured_repeats": baseline_stats.total_measured_repeats,
                "measurement_epochs": baseline_stats.measurement_epochs,
                "unstable_epoch_coefficients_of_variation": (
                    baseline_stats.unstable_epoch_coefficients_of_variation
                ),
                "layer_throughput_tokens_per_second": baseline_tps,
                "throughput_retained_fraction": 1.0,
                "latency_multiplier": 1.0,
                "output_correctness_passed": True,
                "placement_activity": "all_gpu_control",
                "positive_performance_eligible": False,
                "positive_performance_pass": False,
                **dispatch,
            }
            result_rows.append(baseline_row)
            for name, value in baseline_stats.timing_medians.items():
                timing_rows.append(
                    {
                        "classification": "measured_cuda",
                        "arm": "all_gpu",
                        "placement_policy": policy_name,
                        "cpu_expert_count": count,
                        "weight_format": "bfloat16",
                        "timing_component": name,
                        "median_ms": value,
                        "executor_id": CANONICAL_EXECUTOR_ID,
                    }
                )
            for format_name in formats:
                weight_format = cast(CorrectionExpertFormat, format_name)
                hybrid_plan = make_execution_plan(
                    model,
                    corpus,
                    cpu_expert_ids=cpu_ids,
                    weight_format=weight_format,
                    execution_profile=f"natural_routing_hybrid_{weight_format}",
                )
                validate_matched_plans(all_gpu_plan, hybrid_plan)
                hybrid_prepared = executor.prepare(hybrid_plan)
                try:
                    hybrid_stats, hybrid_observation = measure_repeated(
                        executor,
                        hybrid_plan,
                        hybrid_prepared,
                        warmup_iterations=warmup_iterations,
                        minimum_repeats=repeats,
                        maximum_repeats=maximum_repeats,
                        maximum_epochs=maximum_variability_epochs,
                        maximum_cv=maximum_cv,
                    )
                    correctness = correctness_metrics(
                        baseline_observation,
                        hybrid_observation,
                        weight_format=weight_format,
                        atol=atol if weight_format == "bfloat16" else max(atol, 0.75),
                        rtol=rtol if weight_format == "bfloat16" else max(rtol, 0.25),
                        minimum_cosine_similarity=(
                            minimum_cosine_similarity if weight_format == "bfloat16" else 0.99
                        ),
                    )
                    expected_saved = sum(
                        int(item.numel() * item.element_size())
                        for expert_id in cpu_ids
                        for item in model.experts[expert_id]
                    )
                    measured_saved = (
                        baseline_gpu_allocated - hybrid_prepared.gpu_memory_allocated_bytes
                    )
                    retained = baseline_stats.median_ms / max(hybrid_stats.median_ms, 1e-12)
                    activity = (
                        "inactive_cpu_placement"
                        if inactive
                        else (
                            "active_below_minimum_dispatch"
                            if float(dispatch["expected_cpu_dispatch_fraction"])
                            < minimum_dispatch_fraction
                            else "active_cpu_placement"
                        )
                    )
                    hybrid_tps = corpus.token_count / max(hybrid_stats.median_ms / 1000, 1e-12)
                    row = {
                        "classification": "measured_mixed_backend",
                        "benchmark_mode": "natural_routing",
                        "arm": "hybrid_gpu_cpu",
                        "placement_policy": policy_name,
                        "cpu_expert_count": count,
                        "cpu_expert_ids": cpu_ids,
                        "weight_format": weight_format,
                        "executor_id": CANONICAL_EXECUTOR_ID,
                        "matched_baseline_executor_id": CANONICAL_EXECUTOR_ID,
                        "matched_baseline_used": True,
                        "cpu_results_consumed_before_total_timer_end": True,
                        "total_timer_includes_final_synchronisation": True,
                        "token_count": corpus.token_count,
                        "routed_expert_calls": int(corpus.selected_experts.numel()),
                        "cpu_expert_calls": hybrid_observation.cpu_expert_calls,
                        "gpu_expert_calls": hybrid_observation.gpu_expert_calls,
                        "cpu_dispatch_fraction": hybrid_observation.cpu_expert_calls
                        / max(corpus.selected_experts.numel(), 1),
                        "bytes_gpu_to_cpu": hybrid_observation.bytes_gpu_to_cpu,
                        "bytes_cpu_to_gpu": hybrid_observation.bytes_cpu_to_gpu,
                        "expected_gpu_memory_saved_bytes": expected_saved,
                        "gpu_memory_saved_bytes": measured_saved,
                        "gpu_memory_saved_matches_expected": measured_saved == expected_saved,
                        "baseline_gpu_expert_weight_bytes": baseline_gpu_bytes,
                        "gpu_expert_weight_bytes": hybrid_prepared.gpu_weight_bytes,
                        "gpu_memory_allocated_bytes": (hybrid_prepared.gpu_memory_allocated_bytes),
                        "cpu_memory_used_bytes": hybrid_prepared.cpu_weight_bytes,
                        "quantisation_time_ms": hybrid_prepared.quantisation_ms,
                        "maximum_quantised_weight_error": (
                            hybrid_prepared.maximum_quantised_weight_error
                        ),
                        "matched_baseline_median_layer_ms": baseline_stats.median_ms,
                        "median_total_layer_ms": hybrid_stats.median_ms,
                        "minimum_total_layer_ms": hybrid_stats.minimum_ms,
                        "maximum_total_layer_ms": hybrid_stats.maximum_ms,
                        "standard_deviation_ms": hybrid_stats.standard_deviation_ms,
                        "coefficient_of_variation": hybrid_stats.coefficient_of_variation,
                        "measured_repeats": hybrid_stats.repeats,
                        "total_measured_repeats": hybrid_stats.total_measured_repeats,
                        "measurement_epochs": hybrid_stats.measurement_epochs,
                        "unstable_epoch_coefficients_of_variation": (
                            hybrid_stats.unstable_epoch_coefficients_of_variation
                        ),
                        "matched_baseline_throughput_tokens_per_second": baseline_tps,
                        "layer_throughput_tokens_per_second": hybrid_tps,
                        "throughput_retained_fraction": retained,
                        "latency_multiplier": hybrid_stats.median_ms
                        / max(baseline_stats.median_ms, 1e-12),
                        "placement_activity": activity,
                        "positive_performance_eligible": (
                            weight_format == "bfloat16"
                            and not inactive
                            and float(dispatch["expected_cpu_dispatch_fraction"])
                            >= minimum_dispatch_fraction
                        ),
                        **dispatch,
                        **correctness,
                    }
                    row["memory_offload_pass"] = bool(
                        measured_saved > 0
                        and measured_saved == expected_saved
                        and correctness["output_correctness_passed"]
                    )
                    row["positive_performance_pass"] = valid_positive_expert_result(
                        row,
                        minimum_dispatch_fraction=minimum_dispatch_fraction,
                        minimum_retained_fraction=minimum_retained_fraction,
                    )
                    result_rows.append(row)
                    correctness_rows.append(
                        {
                            "classification": "measured_mixed_backend",
                            **{
                                key: row[key]
                                for key in (
                                    "placement_policy",
                                    "cpu_expert_count",
                                    "weight_format",
                                    "cpu_expert_calls",
                                    "cpu_dispatch_fraction",
                                    "output_correctness_passed",
                                    "correctness_label",
                                    "bitwise_exact",
                                    "router_expert_ids_identical",
                                    "routing_weights_maximum_absolute_error",
                                    "expert_output_maximum_absolute_error",
                                    "maximum_absolute_error",
                                    "mean_absolute_error",
                                    "maximum_relative_error",
                                    "cosine_similarity",
                                    "cosine_accumulation_dtype",
                                    "nan_count",
                                    "inf_count",
                                    "downstream_greedy_token_identity",
                                )
                            },
                        }
                    )
                    for name, value in hybrid_stats.timing_medians.items():
                        timing_rows.append(
                            {
                                "classification": "measured_mixed_backend",
                                "arm": "hybrid_gpu_cpu",
                                "placement_policy": policy_name,
                                "cpu_expert_count": count,
                                "weight_format": weight_format,
                                "timing_component": name,
                                "median_ms": value,
                                "executor_id": CANONICAL_EXECUTOR_ID,
                            }
                        )
                    if weight_format == "bfloat16":
                        # Select more than the required five calls because the subset is routed
                        # again at its own GEMM shape. BF16 boundary ties can otherwise remove a
                        # call selected from the full-corpus trace without representing forced
                        # routing.
                        indices = controlled_coverage_indices(
                            corpus.selected_experts,
                            cpu_ids,
                            minimum_calls=20,
                        )
                        if indices:
                            diagnostic_executor = executor.subset(indices)
                            diagnostic_baseline_plan = make_execution_plan(
                                model,
                                diagnostic_executor.corpus,
                                cpu_expert_ids=[],
                                weight_format="bfloat16",
                                execution_profile="controlled_coverage_all_gpu",
                            )
                            diagnostic_plan = make_execution_plan(
                                model,
                                diagnostic_executor.corpus,
                                cpu_expert_ids=cpu_ids,
                                weight_format="bfloat16",
                                execution_profile="controlled_coverage_diagnostic",
                            )
                            validate_matched_plans(diagnostic_baseline_plan, diagnostic_plan)
                            diagnostic_baseline_prepared = diagnostic_executor.prepare(
                                diagnostic_baseline_plan
                            )
                            try:
                                diagnostic_baseline = diagnostic_executor.execute(
                                    diagnostic_baseline_plan,
                                    diagnostic_baseline_prepared,
                                    capture_output=True,
                                )
                            finally:
                                diagnostic_baseline_prepared.release()
                            diagnostic_prepared = diagnostic_executor.prepare(diagnostic_plan)
                            try:
                                diagnostic = diagnostic_executor.execute(
                                    diagnostic_plan,
                                    diagnostic_prepared,
                                    capture_output=True,
                                )
                                diagnostic_cache_metrics = (
                                    diagnostic_prepared.cpu_cache.metrics()
                                    if diagnostic_prepared.cpu_cache is not None
                                    else {}
                                )
                            finally:
                                diagnostic_prepared.release()
                            controlled_counts = placement_dispatch_metrics(
                                diagnostic_executor.corpus.selected_experts, cpu_ids
                            )
                            diagnostic_frequency = Counter(
                                int(item)
                                for item in diagnostic_executor.corpus.selected_experts.flatten().tolist()
                            )
                            if diagnostic_baseline.output is None or diagnostic.output is None:
                                raise RuntimeError(
                                    "controlled-coverage correctness outputs were not captured"
                                )
                            diagnostic_error = (
                                diagnostic.output.float() - diagnostic_baseline.output.float()
                            ).abs()
                            diagnostic_correct = bool(
                                torch.allclose(
                                    diagnostic.output.float(),
                                    diagnostic_baseline.output.float(),
                                    atol=atol,
                                    rtol=rtol,
                                )
                            )
                            controlled_rows.append(
                                {
                                    "classification": "measured_mixed_backend",
                                    "benchmark_mode": "controlled_coverage_diagnostic",
                                    "primary_performance_result": False,
                                    "natural_router_outputs_unmodified": True,
                                    "forced_routing": False,
                                    "subset_router_recomputed_for_subset_shape": True,
                                    "matched_controlled_baseline_used": True,
                                    "placement_policy": policy_name,
                                    "cpu_expert_count": count,
                                    "cpu_expert_ids": cpu_ids,
                                    "token_count": len(indices),
                                    "cpu_expert_calls": diagnostic.cpu_expert_calls,
                                    "minimum_calls_per_owned_expert": min(
                                        (diagnostic_frequency.get(item, 0) for item in cpu_ids),
                                        default=0,
                                    ),
                                    "minimum_required_calls_per_owned_expert": 5,
                                    "minimum_coverage_passed": all(
                                        diagnostic_frequency.get(item, 0) >= 5 for item in cpu_ids
                                    ),
                                    "all_owned_experts_covered": int(
                                        controlled_counts["unique_cpu_experts_selected"]
                                    )
                                    == len(cpu_ids),
                                    "output_correctness_passed": diagnostic_correct,
                                    "maximum_absolute_error": float(diagnostic_error.max().item()),
                                    "mean_absolute_error": float(diagnostic_error.mean().item()),
                                    "cpu_results_consumed_before_total_timer_end": True,
                                    **diagnostic_cache_metrics,
                                    **controlled_counts,
                                }
                            )
                            del diagnostic_executor
                finally:
                    hybrid_prepared.release()
    if full_expert_bytes <= 0:
        raise RuntimeError("real expert weights were not loaded")
    return result_rows, timing_rows, correctness_rows, controlled_rows
