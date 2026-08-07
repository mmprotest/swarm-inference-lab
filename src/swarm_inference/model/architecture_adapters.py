"""Architecture-family adapters driven by checkpoint metadata and tensor layout.

The built-ins cover implemented layout families, not repository names.  New
architectures can be added through the ``swarm_inference.architecture_adapters``
entry-point group without changing model resolution or engine planning.
"""

from __future__ import annotations

import importlib.metadata
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from swarm_inference.model.architecture import (
    ArchitectureSupport,
    DenseOrMoe,
    ExpertDescriptor,
    ModelArchitectureProfile,
    ShardReduction,
    TensorGroupDescriptor,
    TensorInterpretation,
    TensorRole,
    TensorShardSemantics,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.quantization import quantization_from_config

_KEY_RE = re.compile(r"[^a-z0-9]")
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)(?:experts?|local_experts)\.(\d+)(?:\.|$)")
_SHARED_EXPERT_RE = re.compile(r"(?:^|\.)(?:shared_experts?)\.(\d+)(?:\.|$)")
_PROJECTION_RE = re.compile(
    r"(?:^|\.)(gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3|fc1|fc2)(?:\.|$)"
)


def _key(value: object) -> str:
    return _KEY_RE.sub("", str(value).casefold())


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value,)
    return ()


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _number(config: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    for alias in aliases:
        value = config.get(alias)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _quantization(config: dict[str, Any]) -> str | None:
    quantization = quantization_from_config(config)
    if quantization is not None:
        return quantization
    dtype = _effective_config(config).get("dtype") or _effective_config(config).get("torch_dtype")
    return str(dtype).casefold() if dtype else None


def _attention_metadata(config: dict[str, Any]) -> dict[str, Any]:
    effective = _effective_config(config)
    keys = (
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_head_dim",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "sliding_window",
        "full_attention_interval",
        "attn_res_block_size",
        "rotary_dim",
        "partial_rotary_factor",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "layer_types",
        "linear_attn_config",
    )
    return {key: effective[key] for key in keys if key in effective and effective[key] is not None}


def _routing_metadata(config: dict[str, Any]) -> dict[str, Any]:
    effective = _effective_config(config)
    keys = (
        "scoring_func",
        "topk_method",
        "norm_topk_prob",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "router_aux_loss_coef",
        "seq_aux",
        "use_routing_bias",
    )
    return {key: effective[key] for key in keys if key in effective and effective[key] is not None}


def _modalities(config: dict[str, Any]) -> tuple[str, ...]:
    values = ["text"]
    if isinstance(config.get("vision_config"), dict):
        values.append("vision")
    if isinstance(config.get("audio_config"), dict):
        values.append("audio")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    architecture_id: str
    adapter_id: str
    dense_or_moe: DenseOrMoe
    root_model_types: tuple[str, ...] = ()
    root_architectures: tuple[str, ...] = ()
    text_model_types: tuple[str, ...] = ()
    text_architectures: tuple[str, ...] = ()
    gguf_architectures: tuple[str, ...] = ()
    attention_type: str = "transformer-attention"
    tensor_layout: str = "transformers-decoder"
    router_type: str | None = None
    expert_layout: str = "none"
    capabilities: tuple[str, ...] = ()
    validation_notes: tuple[str, ...] = ()
    tied_output_tensor_name: str = "lm_head.weight"


@runtime_checkable
class ModelArchitectureAdapter(Protocol):
    """Execution-neutral architecture adapter contract."""

    adapter_id: str
    adapter_version: ClassVar[str]
    architecture_id: str
    gguf_architectures: tuple[str, ...]

    def match_score(self, config: dict[str, Any]) -> int: ...

    def probe(self, resolved_model: ResolvedModelDescriptor) -> ArchitectureSupport: ...

    def inspect(self, resolved_model: ResolvedModelDescriptor) -> ModelArchitectureProfile: ...

    def tensor_layout(self) -> str: ...

    def layer_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]: ...

    def attention_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]: ...

    def expert_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]: ...

    def routing_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]: ...

    def interpret_tensor(
        self,
        name: str,
        *,
        shape: tuple[int, ...] = (),
        dtype: str | None = None,
        byte_size: int = 0,
    ) -> TensorInterpretation: ...

    def shard_semantics(self, tensor: TensorInterpretation) -> tuple[TensorShardSemantics, ...]: ...

    def describe_experts(
        self,
        tensors: tuple[TensorInterpretation, ...],
        profile: ModelArchitectureProfile,
    ) -> tuple[ExpertDescriptor, ...]: ...

    def tied_weight_alias(
        self, *, tensor_names: tuple[str, ...], config: dict[str, Any]
    ) -> tuple[str, str] | None: ...

    def validate_execution(
        self, *, output_token_ids: tuple[int, ...], requested_tokens: int
    ) -> None: ...


class MetadataArchitectureAdapter:
    """One exact metadata signature with adapter-owned tensor semantics."""

    adapter_version: ClassVar[str] = "1"

    def __init__(self, spec: ArchitectureSpec) -> None:
        self.spec = spec
        self.adapter_id = spec.adapter_id
        self.architecture_id = spec.architecture_id
        self.gguf_architectures = spec.gguf_architectures

    def _claim_keys(self, values: tuple[str, ...]) -> frozenset[str]:
        return frozenset(_key(value) for value in values)

    def match_score(self, config: dict[str, Any]) -> int:
        nested = config.get("text_config")
        nested = nested if isinstance(nested, dict) else {}
        candidates = (
            (100, config.get("model_type"), self.spec.root_model_types),
            (90, _strings(config.get("architectures")), self.spec.root_architectures),
            (80, nested.get("model_type"), self.spec.text_model_types),
            (70, _strings(nested.get("architectures")), self.spec.text_architectures),
        )
        best = 0
        for score, supplied, claims in candidates:
            claim_keys = self._claim_keys(claims)
            supplied_values = supplied if isinstance(supplied, tuple) else _strings(supplied)
            if any(_key(value) in claim_keys for value in supplied_values):
                best = max(best, score)
        return best

    def _dimensions(self, model: ResolvedModelDescriptor) -> tuple[int, int]:
        effective = _effective_config(model.configuration)
        layers = _number(effective, ("num_hidden_layers", "n_layer", "num_layers"))
        hidden = _number(effective, ("hidden_size", "n_embd", "d_model"))
        layers = layers if layers is not None else model.layer_count
        hidden = hidden if hidden is not None else model.hidden_size
        if layers is None or layers <= 0 or hidden is None or hidden <= 0:
            raise ValueError(
                "architecture metadata is missing positive layer and hidden dimensions"
            )
        return layers, hidden

    def _expert_topology(self, config: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
        if self.spec.dense_or_moe == "dense":
            return None, None, None
        effective = _effective_config(config)
        experts = _number(
            effective,
            ("num_experts", "n_routed_experts", "num_local_experts", "experts_per_layer"),
        )
        top_k = _number(
            effective,
            ("num_experts_per_tok", "num_experts_per_token", "experts_per_token", "top_k"),
        )
        shared = _number(
            effective,
            ("n_shared_experts", "num_shared_experts", "shared_expert_count"),
        )
        if experts is None or experts <= 0 or top_k is None or top_k <= 0:
            raise ValueError("MoE architecture metadata is missing expert count or active top-k")
        if top_k > experts:
            raise ValueError("MoE active top-k exceeds routed expert count")
        return experts, top_k, max(0, shared or 0)

    def probe(self, resolved_model: ResolvedModelDescriptor) -> ArchitectureSupport:
        score = self.match_score(resolved_model.configuration)
        reasons: list[str] = []
        supported = score > 0 or resolved_model.architecture == self.architecture_id
        confidence = "exact_metadata" if supported else "insufficient"
        if supported:
            try:
                self._dimensions(resolved_model)
                self._expert_topology(resolved_model.configuration)
                reasons.append("exact architecture metadata and required dimensions validate")
            except ValueError as exc:
                supported = False
                confidence = "insufficient"
                reasons.append(str(exc))
        else:
            reasons.append("checkpoint metadata does not match this architecture signature")
        return ArchitectureSupport(
            adapter_id=self.adapter_id,
            architecture_id=self.architecture_id,
            supported=supported,
            confidence=confidence,
            reasons=tuple(reasons),
        )

    def inspect(self, resolved_model: ResolvedModelDescriptor) -> ModelArchitectureProfile:
        support = self.probe(resolved_model)
        if not support.supported:
            raise ValueError("; ".join(support.reasons))
        config = resolved_model.configuration
        effective = _effective_config(config)
        layers, hidden = self._dimensions(resolved_model)
        experts, top_k, shared = self._expert_topology(config)
        expert_intermediate = (
            _number(
                effective,
                (
                    "moe_intermediate_size",
                    "expert_intermediate_size",
                    "intermediate_size",
                    "ffn_hidden_size",
                ),
            )
            if experts is not None
            else None
        )
        modalities = _modalities(config)
        capabilities = {
            "causal-lm",
            "architecture-adapter",
            "layer-staging",
            *self.spec.capabilities,
        }
        if experts is not None:
            capabilities.update(("routed-experts", "expert-placement", "expert-microsharding"))
            if shared:
                capabilities.add("shared-experts")
        if len(modalities) > 1:
            capabilities.add("multimodal")
        total_parameters = resolved_model.parameter_count
        active_parameters = _number(
            effective, ("num_active_parameters", "active_parameter_count", "active_parameters")
        )
        source = resolved_model.architecture_source
        raw_values = tuple(
            dict.fromkeys(
                (
                    *_strings(config.get("architectures")),
                    *_strings(config.get("model_type")),
                    *_strings(effective.get("architectures")),
                    *_strings(effective.get("model_type")),
                )
            )
        )
        return ModelArchitectureProfile(
            architecture_id=self.architecture_id,
            adapter_id=self.adapter_id,
            dense_or_moe=self.spec.dense_or_moe,
            layer_count=layers,
            hidden_size=hidden,
            attention_type=self.spec.attention_type,
            attention_metadata=_attention_metadata(config),
            expert_count=experts,
            experts_per_token=top_k,
            shared_expert_count=shared,
            expert_intermediate_size=expert_intermediate,
            router_type=self.spec.router_type,
            routing_metadata=_routing_metadata(config),
            tensor_layout=self.spec.tensor_layout,
            checkpoint_format=resolved_model.format,
            quantization=resolved_model.quantization or _quantization(config),
            multimodal=len(modalities) > 1,
            modalities=modalities,
            capabilities=frozenset(capabilities),
            total_parameters=total_parameters,
            active_parameters=active_parameters,
            vocab_size=_number(effective, ("vocab_size", "n_vocab")),
            configuration_source=source,
            raw_architectures=raw_values,
            validation_notes=self.spec.validation_notes,
        )

    def tensor_layout(self) -> str:
        return self.spec.tensor_layout

    def layer_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]:
        return {
            "layer_count": profile.layer_count,
            "contiguous_stage_boundaries": True,
            "stage_local_kv": True,
            "tensor_layout": self.spec.tensor_layout,
        }

    def attention_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]:
        return {"type": profile.attention_type, **profile.attention_metadata}

    def expert_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]:
        return {
            "layout": self.spec.expert_layout,
            "routed_experts": profile.expert_count,
            "experts_per_token": profile.experts_per_token,
            "shared_experts": profile.shared_expert_count,
        }

    def routing_layout(self, profile: ModelArchitectureProfile) -> dict[str, Any]:
        return {
            "router_type": profile.router_type,
            "experts_per_token": profile.experts_per_token,
            **profile.routing_metadata,
        }

    def _role(self, name: str, expert_index: int | None) -> TensorRole:
        lowered = name.casefold()
        if "vision" in lowered or "audio" in lowered or "multi_modal" in lowered:
            return TensorRole.MULTIMODAL
        if "embed" in lowered or lowered.endswith("tok_embeddings.weight"):
            return TensorRole.EMBEDDING
        if "lm_head" in lowered or lowered.endswith(("output.weight", "output_layer.weight")):
            return TensorRole.OUTPUT_HEAD
        if "shared_expert" in lowered or "shared_experts" in lowered:
            return TensorRole.SHARED_EXPERT
        if "always_on_expert" in lowered:
            return TensorRole.ALWAYS_ON_EXPERT
        if expert_index is not None or ".experts." in lowered or ".local_experts." in lowered:
            return TensorRole.ROUTED_EXPERT
        if any(value in lowered for value in ("router", "e_score_correction", "mlp.gate.weight")):
            return TensorRole.ROUTER
        if any(value in lowered for value in ("self_attn", "attention", "attn.")):
            return TensorRole.ATTENTION_NORM if "norm" in lowered else TensorRole.ATTENTION
        if "norm" in lowered:
            return (
                TensorRole.FINAL_NORM
                if _LAYER_RE.search(name) is None
                else TensorRole.ATTENTION_NORM
            )
        if any(value in lowered for value in ("mlp", "feed_forward", "ffn")):
            return TensorRole.DENSE_MLP
        return TensorRole.UNKNOWN

    def shard_semantics(self, tensor: TensorInterpretation) -> tuple[TensorShardSemantics, ...]:
        projection = (tensor.projection or "").casefold()
        role = tensor.role.value
        if projection in {"gate_proj", "up_proj", "w1", "w3", "fc1", "gate_up_proj"}:
            return (
                TensorShardSemantics(
                    tensor_role=role,
                    shard_axis=0,
                    output_axis=0,
                    reduction=ShardReduction.CONCATENATE,
                    alignment=32,
                    notes=("adapter-described intermediate/output feature partition",),
                ),
            )
        if projection in {"down_proj", "w2", "fc2"}:
            return (
                TensorShardSemantics(
                    tensor_role=role,
                    shard_axis=1,
                    reduction=ShardReduction.SUM,
                    alignment=32,
                    notes=("partial outputs are reduced deterministically",),
                ),
            )
        if (
            tensor.role == TensorRole.ROUTED_EXPERT
            and self.spec.expert_layout == "fused-expert-axis"
        ):
            return (
                TensorShardSemantics(
                    tensor_role=role,
                    shard_axis=0,
                    reduction=ShardReduction.GATHER,
                    alignment=1,
                    notes=("axis zero indexes complete experts",),
                ),
            )
        return ()

    def interpret_tensor(
        self,
        name: str,
        *,
        shape: tuple[int, ...] = (),
        dtype: str | None = None,
        byte_size: int = 0,
    ) -> TensorInterpretation:
        layer_match = _LAYER_RE.search(name)
        expert_match = _EXPERT_RE.search(name)
        shared_expert_match = _SHARED_EXPERT_RE.search(name)
        projection_match = _PROJECTION_RE.search(name)
        layer = int(layer_match.group(1)) if layer_match else None
        expert = int(expert_match.group(1)) if expert_match else None
        role = self._role(name, expert)
        expert_type: Literal["routed", "shared", "always_on", "latent", "grouped"] | None = None
        if role == TensorRole.ROUTED_EXPERT:
            expert_type = "routed" if expert is not None else "grouped"
        elif role == TensorRole.SHARED_EXPERT:
            expert_type = "shared"
            expert = int(shared_expert_match.group(1)) if shared_expert_match else 0
        elif role == TensorRole.ALWAYS_ON_EXPERT:
            expert_type = "always_on"
            expert = 0
        projection = projection_match.group(1) if projection_match else None
        if projection is not None and any(
            marker in name.casefold()
            for marker in ("weight_scale", "scale_inv", "weight_scale_inv")
        ):
            projection = f"{projection}_scale"
        provisional = TensorInterpretation(
            tensor_name=name,
            role=role,
            layer_index=layer,
            expert_index=expert,
            expert_type=expert_type,
            projection=projection,
            tensor_group=(
                f"layer-{layer}:expert-{expert}"
                if layer is not None and expert is not None
                else f"layer-{layer}:{role.value}"
                if layer is not None
                else role.value
            ),
            shape=shape,
            dtype=dtype,
            byte_size=max(0, byte_size),
        )
        return provisional.model_copy(update={"shard_semantics": self.shard_semantics(provisional)})

    def describe_experts(
        self,
        tensors: tuple[TensorInterpretation, ...],
        profile: ModelArchitectureProfile,
    ) -> tuple[ExpertDescriptor, ...]:
        grouped: dict[
            tuple[
                int,
                int,
                Literal["routed", "shared", "always_on", "latent", "grouped"],
            ],
            list[TensorInterpretation],
        ] = defaultdict(list)
        fused: dict[int, list[TensorInterpretation]] = defaultdict(list)
        for tensor in tensors:
            if tensor.layer_index is None or tensor.expert_type is None:
                continue
            if tensor.expert_index is not None:
                grouped[(tensor.layer_index, tensor.expert_index, tensor.expert_type)].append(
                    tensor
                )
            elif tensor.expert_type == "grouped":
                fused[tensor.layer_index].append(tensor)
        experts: list[ExpertDescriptor] = []
        for (layer, expert, expert_type), items in sorted(grouped.items()):
            ordered = sorted(items, key=lambda item: item.tensor_name)
            parameters = sum(math.prod(item.shape) for item in ordered if item.shape)
            group = TensorGroupDescriptor(
                group_id=f"layer-{layer}:expert-{expert}",
                tensor_names=tuple(item.tensor_name for item in ordered),
                tensor_roles=tuple(item.projection or item.role.value for item in ordered),
                tensor_shapes=tuple(item.shape for item in ordered),
                parameter_count=parameters or None,
                memory_bytes=sum(item.byte_size for item in ordered),
                shard_semantics=tuple(
                    semantic for item in ordered for semantic in item.shard_semantics
                ),
            )
            experts.append(
                ExpertDescriptor(
                    layer_index=layer,
                    expert_index=expert,
                    expert_type=expert_type,
                    tensor_groups=(group,),
                    parameter_count=parameters or None,
                    memory_bytes=group.memory_bytes,
                    input_shape=(profile.hidden_size,),
                    output_shape=(profile.hidden_size,),
                    routing_metadata={
                        "router_type": profile.router_type,
                        "experts_per_token": profile.experts_per_token,
                    },
                )
            )
        for layer, items in sorted(fused.items()):
            if profile.expert_count is None:
                continue
            ordered = sorted(items, key=lambda item: item.tensor_name)
            if not ordered or any(
                not item.shape or item.shape[0] != profile.expert_count for item in ordered
            ):
                continue
            for expert in range(profile.expert_count):
                parameters = sum(math.prod(item.shape[1:]) for item in ordered)
                memory = sum(item.byte_size // profile.expert_count for item in ordered)
                group = TensorGroupDescriptor(
                    group_id=f"layer-{layer}:expert-{expert}",
                    tensor_names=tuple(item.tensor_name for item in ordered),
                    tensor_roles=tuple(item.projection or item.role.value for item in ordered),
                    tensor_shapes=tuple(item.shape[1:] for item in ordered),
                    parameter_count=parameters,
                    memory_bytes=memory,
                    shard_semantics=tuple(
                        semantic for item in ordered for semantic in item.shard_semantics
                    ),
                )
                experts.append(
                    ExpertDescriptor(
                        layer_index=layer,
                        expert_index=expert,
                        expert_type="routed",
                        tensor_groups=(group,),
                        parameter_count=parameters,
                        memory_bytes=memory,
                        input_shape=(profile.hidden_size,),
                        output_shape=(profile.hidden_size,),
                        routing_metadata={
                            "router_type": profile.router_type,
                            "experts_per_token": profile.experts_per_token,
                            "tensor_slices": {
                                item.tensor_name: {"axis": 0, "index": expert} for item in ordered
                            },
                        },
                    )
                )
        return tuple(
            sorted(
                experts, key=lambda item: (item.layer_index, item.expert_type, item.expert_index)
            )
        )

    def tied_weight_alias(
        self, *, tensor_names: tuple[str, ...], config: dict[str, Any]
    ) -> tuple[str, str] | None:
        effective = _effective_config(config)
        tied = bool(config.get("tie_word_embeddings", effective.get("tie_word_embeddings", False)))
        if not tied:
            return None
        embeddings = [
            name
            for name in tensor_names
            if self.interpret_tensor(name).role == TensorRole.EMBEDDING
        ]
        if len(embeddings) != 1:
            raise ValueError(
                "tied checkpoint must expose exactly one adapter-recognized token embedding"
            )
        return embeddings[0], self.spec.tied_output_tensor_name

    def validate_execution(
        self, *, output_token_ids: tuple[int, ...], requested_tokens: int
    ) -> None:
        if requested_tokens < 0:
            raise ValueError("requested token count cannot be negative")
        if len(output_token_ids) > requested_tokens:
            raise ValueError("architecture execution produced more tokens than requested")
        if any(token < 0 for token in output_token_ids):
            raise ValueError("architecture execution produced a negative token ID")


class ModelArchitectureAdapterRegistry:
    """Deterministic registry whose dispatch uses metadata evidence only."""

    def __init__(self, adapters: tuple[ModelArchitectureAdapter, ...] = ()) -> None:
        self._adapters: dict[str, ModelArchitectureAdapter] = {}
        self._architectures: dict[str, ModelArchitectureAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ModelArchitectureAdapter) -> None:
        if not isinstance(adapter, ModelArchitectureAdapter):
            raise TypeError("architecture adapter does not satisfy the protocol")
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"architecture adapter {adapter.adapter_id!r} is already registered")
        if adapter.architecture_id in self._architectures:
            raise ValueError(f"architecture {adapter.architecture_id!r} has multiple adapters")
        self._adapters[adapter.adapter_id] = adapter
        self._architectures[adapter.architecture_id] = adapter

    def adapters(self) -> tuple[ModelArchitectureAdapter, ...]:
        return tuple(self._adapters[key] for key in sorted(self._adapters))

    def get(self, adapter_id: str) -> ModelArchitectureAdapter:
        return self._adapters[adapter_id]

    def get_by_architecture(self, architecture_id: str | None) -> ModelArchitectureAdapter | None:
        return self._architectures.get(str(architecture_id)) if architecture_id else None

    def has_architecture(self, architecture_id: str) -> bool:
        return architecture_id in self._architectures

    def architecture_for_identifier(self, value: str) -> str | None:
        supplied = _key(value)
        matches: set[str] = set()
        for adapter in self.adapters():
            spec = getattr(adapter, "spec", None)
            if not isinstance(spec, ArchitectureSpec):
                if supplied in {_key(item) for item in adapter.gguf_architectures}:
                    matches.add(adapter.architecture_id)
                continue
            claims = (
                *spec.root_model_types,
                *spec.root_architectures,
                *spec.text_model_types,
                *spec.text_architectures,
                *spec.gguf_architectures,
                spec.architecture_id,
            )
            if supplied in {_key(item) for item in claims}:
                matches.add(adapter.architecture_id)
        return next(iter(matches)) if len(matches) == 1 else None

    def resolve_config(self, config: dict[str, Any]) -> ModelArchitectureAdapter | None:
        scored = [(adapter.match_score(config), adapter) for adapter in self.adapters()]
        best = max((score for score, _ in scored), default=0)
        matches = [adapter for score, adapter in scored if score == best and score > 0]
        return matches[0] if len(matches) == 1 else None

    def resolve_model(self, model: ResolvedModelDescriptor) -> ModelArchitectureAdapter | None:
        adapter = self.resolve_config(model.configuration)
        if adapter is not None:
            return adapter
        return self.get_by_architecture(model.architecture)

    def inspect(self, model: ResolvedModelDescriptor) -> ModelArchitectureProfile:
        adapter = self.resolve_model(model)
        if adapter is None:
            raw = model.architecture_raw or model.architecture or "missing"
            raise LookupError(f"no architecture adapter validates metadata {raw!r}")
        return adapter.inspect(model)

    def gguf_architecture_mapping(self) -> dict[str, tuple[str, ...]]:
        return {
            adapter.architecture_id: adapter.gguf_architectures
            for adapter in self.adapters()
            if adapter.gguf_architectures
        }


def _specs() -> tuple[ArchitectureSpec, ...]:
    return (
        ArchitectureSpec(
            "qwen3_dense",
            "qwen3-dense",
            "dense",
            root_model_types=("qwen3",),
            root_architectures=("Qwen3ForCausalLM",),
            gguf_architectures=("qwen3",),
            attention_type="rope-gqa",
            capabilities=("qwen-rope",),
        ),
        ArchitectureSpec(
            "qwen3_moe",
            "qwen3-moe",
            "moe",
            root_model_types=("qwen3_moe",),
            root_architectures=("Qwen3MoeForCausalLM",),
            gguf_architectures=("qwen3moe",),
            attention_type="rope-gqa",
            tensor_layout="transformers-indexed-experts",
            router_type="softmax-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("qwen-rope", "sparse-moe"),
        ),
        ArchitectureSpec(
            "qwen3_5_dense",
            "qwen3-5-dense",
            "dense",
            root_model_types=("qwen3_5",),
            root_architectures=("Qwen3_5ForConditionalGeneration",),
            text_model_types=("qwen3_5_text", "qwen3_5"),
            text_architectures=("Qwen3_5ForCausalLM",),
            gguf_architectures=("qwen35",),
            attention_type="hybrid-gated-delta-and-full-attention",
            tensor_layout="qwen3-5-hybrid-decoder",
            capabilities=("linear-attention", "periodic-full-attention", "multimodal-wrapper"),
        ),
        ArchitectureSpec(
            "qwen3_5_moe",
            "qwen3-5-moe",
            "moe",
            root_model_types=("qwen3_5_moe",),
            root_architectures=("Qwen3_5MoeForConditionalGeneration",),
            text_model_types=("qwen3_5_moe_text", "qwen3_5_moe"),
            text_architectures=("Qwen3_5MoeForCausalLM",),
            gguf_architectures=("qwen35moe",),
            attention_type="hybrid-gated-delta-and-full-attention",
            tensor_layout="qwen3-5-hybrid-indexed-experts",
            router_type="softmax-top-k",
            expert_layout="indexed-separate-or-fused-projections",
            capabilities=("linear-attention", "periodic-full-attention", "sparse-moe"),
            validation_notes=(
                "Qwen 3.5 and later checkpoints that retain qwen3_5_moe metadata share this adapter",
            ),
        ),
        ArchitectureSpec(
            "kimi_k2_moe",
            "kimi-k2-moe",
            "moe",
            root_model_types=("kimi_k2", "kimi_k25"),
            root_architectures=("KimiK25ForConditionalGeneration",),
            text_model_types=("kimi_k2", "deepseek_v3"),
            text_architectures=("DeepseekV3ForCausalLM",),
            gguf_architectures=("kimi2", "deepseek2"),
            attention_type="multi-head-latent-attention",
            tensor_layout="deepseek-compatible-indexed-experts",
            router_type="sigmoid-noaux-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("mla", "shared-experts", "sparse-moe"),
        ),
        ArchitectureSpec(
            "kimi_k3_moe",
            "kimi-k3-moe",
            "moe",
            root_model_types=("kimi_k3",),
            root_architectures=("KimiK3ForConditionalGeneration",),
            text_model_types=("kimi_k3_text", "kimi_linear"),
            text_architectures=("KimiLinearForCausalLM", "KimiK3ForCausalLM"),
            attention_type="hybrid-kda-latent-attention",
            tensor_layout="kimi-k3-native-mxfp4-experts",
            router_type="stable-latent-sigmoid-top-k",
            expert_layout="indexed-w1-w2-w3-packed",
            capabilities=("kda", "mla", "mxfp4", "shared-experts", "sparse-moe"),
        ),
        ArchitectureSpec(
            "glm_moe",
            "glm-moe",
            "moe",
            root_model_types=("glm_moe", "glm_moe_dsa", "glm4_moe"),
            root_architectures=("GlmMoeForCausalLM", "GlmMoeDsaForCausalLM"),
            gguf_architectures=("glm4moe", "glmmoe"),
            attention_type="multi-head-latent-attention-with-dsa",
            tensor_layout="glm-indexed-experts",
            router_type="sigmoid-noaux-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("mla", "dsa", "shared-experts", "sparse-moe"),
        ),
        ArchitectureSpec(
            "deepseek_v3_moe",
            "deepseek-v3-moe",
            "moe",
            root_model_types=("deepseek_v3", "deepseek_v32"),
            root_architectures=("DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM"),
            gguf_architectures=("deepseek2", "deepseek3"),
            attention_type="multi-head-latent-attention",
            tensor_layout="deepseek-indexed-routed-and-shared-experts",
            router_type="sigmoid-noaux-grouped-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("mla", "shared-experts", "sparse-moe", "mtp"),
            validation_notes=(
                "DeepSeek V3.x checkpoints retaining V3 routed-expert and MLA semantics share this adapter",
            ),
        ),
        ArchitectureSpec(
            "minimax_moe",
            "minimax-moe",
            "moe",
            root_model_types=("minimax", "minimax_m2", "minimax_m3_vl"),
            root_architectures=(
                "MiniMaxForCausalLM",
                "MiniMaxM2ForCausalLM",
                "MiniMaxM3SparseForConditionalGeneration",
            ),
            text_model_types=("minimax_m3",),
            gguf_architectures=("minimax", "minimax-m2"),
            attention_type="hybrid-linear-and-full-attention",
            tensor_layout="minimax-indexed-or-fused-experts",
            router_type="sigmoid-routing-bias-top-k",
            expert_layout="indexed-separate-or-fused-projections",
            capabilities=("linear-attention", "shared-experts", "sparse-moe"),
        ),
        ArchitectureSpec(
            "llama_dense",
            "llama-dense",
            "dense",
            root_model_types=("llama",),
            root_architectures=("LlamaForCausalLM",),
            gguf_architectures=("llama",),
            attention_type="rope-gqa",
            tensor_layout="llama-decoder",
            capabilities=("gqa",),
        ),
        ArchitectureSpec(
            "llama4_moe",
            "llama4-moe",
            "moe",
            root_model_types=("llama4",),
            root_architectures=("Llama4ForConditionalGeneration", "Llama4ForCausalLM"),
            text_model_types=("llama4_text",),
            gguf_architectures=("llama4",),
            attention_type="irope-gqa",
            tensor_layout="llama4-interleaved-moe",
            router_type="softmax-top-k",
            expert_layout="fused-expert-axis",
            capabilities=("irope", "early-fusion", "sparse-moe"),
        ),
        ArchitectureSpec(
            "mistral_dense",
            "mistral-dense",
            "dense",
            root_model_types=("mistral",),
            root_architectures=("MistralForCausalLM",),
            gguf_architectures=("mistral",),
            attention_type="sliding-window-rope-gqa",
            tensor_layout="mistral-decoder",
            capabilities=("gqa", "sliding-window-attention"),
        ),
        ArchitectureSpec(
            "mixtral_moe",
            "mixtral-moe",
            "moe",
            root_model_types=("mixtral",),
            root_architectures=("MixtralForCausalLM",),
            gguf_architectures=("llama", "mixtral"),
            attention_type="sliding-window-rope-gqa",
            tensor_layout="mixtral-indexed-experts",
            router_type="softmax-top-k",
            expert_layout="indexed-w1-w2-w3",
            capabilities=("gqa", "sliding-window-attention", "sparse-moe"),
        ),
        ArchitectureSpec(
            "mistral4_moe",
            "mistral4-moe",
            "moe",
            root_model_types=("mistral3",),
            root_architectures=("Mistral3ForConditionalGeneration",),
            text_model_types=("mistral4",),
            gguf_architectures=("mistral4",),
            attention_type="multi-head-latent-attention",
            tensor_layout="mistral4-indexed-experts",
            router_type="sigmoid-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("mla", "shared-experts", "sparse-moe"),
        ),
        ArchitectureSpec(
            "gemma_dense",
            "gemma-dense",
            "dense",
            root_model_types=("gemma", "gemma2", "gemma3", "gemma4"),
            root_architectures=(
                "GemmaForCausalLM",
                "Gemma2ForCausalLM",
                "Gemma3ForConditionalGeneration",
                "Gemma4ForConditionalGeneration",
                "Gemma4UnifiedForConditionalGeneration",
            ),
            text_model_types=("gemma3_text", "gemma4_text"),
            gguf_architectures=("gemma", "gemma2", "gemma3", "gemma4"),
            attention_type="alternating-sliding-and-global-attention",
            tensor_layout="gemma-decoder",
            capabilities=("sliding-window-attention", "logit-softcapping"),
        ),
        # Compatibility is retained through the same adapter mechanism as any
        # third-party architecture; it is intentionally not a product default.
        ArchitectureSpec(
            "olmoe_moe",
            "olmoe-compat",
            "moe",
            root_model_types=("olmoe",),
            root_architectures=("OlmoeForCausalLM",),
            gguf_architectures=("olmoe",),
            attention_type="rope-gqa",
            tensor_layout="transformers-indexed-experts",
            router_type="softmax-top-k",
            expert_layout="indexed-separate-projections",
            capabilities=("sparse-moe",),
            validation_notes=("compatibility adapter; no product-default status",),
        ),
    )


@lru_cache(maxsize=1)
def default_architecture_adapter_registry() -> ModelArchitectureAdapterRegistry:
    adapters: list[ModelArchitectureAdapter] = [
        MetadataArchitectureAdapter(spec) for spec in _specs()
    ]
    for entry_point in importlib.metadata.entry_points(
        group="swarm_inference.architecture_adapters"
    ):
        loaded = entry_point.load()
        value = loaded() if callable(loaded) else loaded
        if isinstance(value, tuple):
            adapters.extend(value)
        else:
            adapters.append(value)
    return ModelArchitectureAdapterRegistry(tuple(adapters))


__all__ = [
    "MetadataArchitectureAdapter",
    "ModelArchitectureAdapter",
    "ModelArchitectureAdapterRegistry",
    "default_architecture_adapter_registry",
]
