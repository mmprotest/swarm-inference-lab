"""Concrete architecture adapters backed by code in pinned Colibri v1.4.0."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import re
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

from swarm_inference.backends.colibri.architecture import (
    ColibriArchitectureAdapter,
    ColibriArchitectureAdapterRegistry,
    ColibriCompatibilityStatus,
    ColibriExecutionProfile,
    ColibriModelProfile,
    ColibriReplayInvocation,
    ColibriRoutingDescriptor,
    ColibriRuntimeCapabilities,
    ColibriSupportResult,
    ColibriTensorMapping,
    ExpertDescriptor,
    architecture_claim_key,
    architecture_metadata_values,
)
from swarm_inference.engines.interfaces import ClusterCapabilities
from swarm_inference.model.architecture import (
    ShardReduction,
    TensorGroupDescriptor,
    TensorShardSemantics,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.model.quantization import (
    normalize_quantization_name,
    quantization_from_config,
)

_LAYER_RE = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts?\.(\d+)(?:\.|$)")
_SHARED_EXPERT_RE = re.compile(r"(?:^|\.)shared_experts?\.(\d+)(?:\.|$)")


def _normalise_quantization(value: str | None) -> str | None:
    return normalize_quantization_name(value)


def _declared_quantization(model: ResolvedModelDescriptor) -> str | None:
    explicit = _normalise_quantization(model.quantization)
    configured = quantization_from_config(model.configuration)
    if explicit is not None and not explicit.startswith("compressed-tensors"):
        return explicit
    return configured or explicit


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _number(config: dict[str, Any], aliases: tuple[str, ...]) -> int | None:
    for key in aliases:
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _tensor_product(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 0


def _dense_role(name: str) -> str:
    lowered = name.casefold()
    if "shared_expert" in lowered:
        return "shared_expert"
    if "router" in lowered or lowered.endswith(("mlp.gate.weight", "moe.gate.weight")):
        return "router"
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered or lowered.endswith("output.weight"):
        return "output_head"
    if "norm" in lowered:
        return "normalization"
    if any(item in lowered for item in ("self_attn", "self_attention", "attention")):
        return "attention"
    return "dense_parameter"


class _BaseColibriAdapter:
    adapter_id: ClassVar[str]
    adapter_version: ClassVar[str] = "1"
    claimed_architectures: ClassVar[tuple[str, ...]]
    engine_basename: ClassVar[str]
    gateway_architecture: ClassVar[str | None]
    supports_text_calibration: ClassVar[bool] = False
    required_config: ClassVar[dict[str, tuple[str, ...]]]
    accepted_quantizations: ClassVar[frozenset[str]]
    native_quantization: ClassVar[str]
    checkpoint_layout: ClassVar[str]
    routing_kind: ClassVar[str]
    routing_normalization: ClassVar[str]
    routing_score_correction: ClassVar[str | None] = None
    routed_weight_semantics: ClassVar[str]
    tokenizer_mode: ClassVar[str] = "gateway-text"
    launch_mode: ClassVar[str] = "persistent-gateway"
    static_limitations: ClassVar[tuple[str, ...]] = ()
    exact_replay: ClassVar[bool] = False
    tensor_microshards: ClassVar[bool] = False
    direct_peer_model_data: ClassVar[bool] = False
    topology: ClassVar[str] = "colibri-streamed-sparse-model"
    complete_model: ClassVar[bool] = True
    component_capabilities: ClassVar[tuple[str, ...]] = ()
    tuning_settings: ClassVar[tuple[str, ...]] = (
        "OMP_NUM_THREADS",
        "COLI_NUMA",
        "PIPE",
        "DIRECT",
        "LOADERS",
        "RAM_GB",
        "PILOT_REAL",
        "PREFETCH",
        "CHUNK",
    )

    def validate_contract(self) -> None:
        if not self.adapter_id.strip() or not self.engine_basename.strip():
            raise ValueError("Colibri adapters require IDs and engine basenames")
        if not self.claimed_architectures or not self.required_config:
            raise ValueError(
                f"Colibri adapter {self.adapter_id!r} must claim metadata and config fields"
            )
        if any(not architecture_claim_key(value) for value in self.claimed_architectures):
            raise ValueError(f"Colibri adapter {self.adapter_id!r} has an invalid claim")
        if not self.accepted_quantizations or not self.native_quantization:
            raise ValueError(f"Colibri adapter {self.adapter_id!r} lacks a quantization contract")

    def matches_config(self, config: dict[str, Any]) -> bool:
        claims = {architecture_claim_key(value) for value in self.claimed_architectures}
        return any(
            architecture_claim_key(value) in claims
            for value in architecture_metadata_values(config)
        )

    def _config_values(self, config: dict[str, Any]) -> dict[str, int]:
        effective = _effective_config(config)
        values: dict[str, int] = {}
        missing: list[str] = []
        for logical, aliases in self.required_config.items():
            value = _number(effective, aliases)
            if value is None:
                missing.append("/".join(aliases))
            else:
                values[logical] = value
        if missing:
            raise ValueError(
                f"{self.adapter_id} config is missing required numeric fields: "
                + ", ".join(missing)
            )
        if values["layers"] <= 0 or values["hidden"] <= 0 or values["experts"] <= 0:
            raise ValueError(f"{self.adapter_id} model dimensions must be positive")
        if values["top_k"] <= 0 or values["top_k"] > values["experts"]:
            raise ValueError(f"{self.adapter_id} experts-per-token is outside the expert range")
        if values["intermediate"] <= 0:
            raise ValueError(f"{self.adapter_id} expert intermediate size must be positive")
        return values

    def inspect_model(self, model: ResolvedModelDescriptor) -> ColibriModelProfile:
        limitations = list(self.static_limitations)
        config = model.configuration
        if config:
            if not self.matches_config(config):
                raise ValueError(
                    f"architecture metadata does not identify the {self.adapter_id} adapter"
                )
            values = self._config_values(config)
            shared = self._shared_experts(_effective_config(config))
        else:
            values = {
                "layers": model.layer_count or 0,
                "hidden": model.hidden_size or 0,
                "experts": 0,
                "top_k": 0,
                "intermediate": 0,
            }
            shared = 0
            limitations.append(
                "model config was not retained by the caller; deployment must validate topology"
            )
        return ColibriModelProfile(
            adapter_id=self.adapter_id,
            architecture=model.architecture or self.claimed_architectures[0],
            architecture_raw=model.architecture_raw,
            checkpoint_format=model.format,
            checkpoint_layout=self.checkpoint_layout,
            quantization=_declared_quantization(model) or self.native_quantization,
            layer_count=values["layers"] or model.layer_count,
            hidden_size=values["hidden"] or model.hidden_size,
            expert_count=values["experts"] or None,
            experts_per_token=values["top_k"] or None,
            shared_expert_count=max(0, shared),
            expert_intermediate_size=values["intermediate"] or None,
            routing_kind=self.routing_kind,
            architecture_metadata=self._architecture_metadata(_effective_config(config)),
            required_config_fields=tuple(self.required_config),
            tokenizer_mode=self.tokenizer_mode,
            launch_mode=self.launch_mode,
            limitations=tuple(dict.fromkeys(limitations)),
        )

    def _architecture_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {}

    def _shared_experts(self, config: dict[str, Any]) -> int:
        value = _number(config, ("num_shared_experts", "n_shared_experts"))
        return max(0, value or 0)

    def supports(
        self,
        model: ResolvedModelDescriptor,
        runtime: ColibriRuntimeCapabilities,
    ) -> ColibriSupportResult:
        limitations: list[str] = []
        reasons: list[str] = []
        runtime_supported = (
            runtime.installed
            and bool(runtime.runtime_version)
            and bool(runtime.binary_hashes)
            and self.adapter_id in runtime.adapters
        )
        if not runtime.installed:
            reasons.append("Colibri runtime is not installed for this worker profile")
        elif not runtime.runtime_version or not runtime.binary_hashes:
            reasons.append("Colibri runtime lacks immutable revision or binary-hash evidence")
        elif self.adapter_id not in runtime.adapters:
            reasons.append(
                f"installed Colibri runtime does not advertise adapter {self.adapter_id!r}"
            )

        format_supported = model.format.casefold() == "safetensors" and (
            not runtime.formats
            or model.format.casefold() in {value.casefold() for value in runtime.formats}
        )
        if not format_supported:
            reasons.append(f"adapter {self.adapter_id} consumes Safetensors, not {model.format!r}")

        quantization = _declared_quantization(model)
        quantization_supported = quantization is None or quantization in self.accepted_quantizations
        if quantization_supported and quantization and runtime.quantizations:
            runtime_quantizations = {
                value
                for item in runtime.quantizations
                if (value := _normalise_quantization(item)) is not None
            }
            quantization_supported = quantization in runtime_quantizations
        if not quantization_supported:
            reasons.append(
                f"quantization {quantization!r} is not executable by adapter {self.adapter_id}"
            )
        elif quantization is None:
            limitations.append(
                f"quantization is determined from the adapter-validated {self.checkpoint_layout} layout"
            )
        if not runtime.quantizations:
            limitations.append(
                "worker manifest does not enumerate quantizations; deployment revalidates tensor layout"
            )

        # Architecture identity and artifact-format compatibility are separate
        # facts.  A GGUF descriptor may intentionally carry only the bounded
        # metadata needed to identify its architecture; trying to interpret it
        # as the adapter's Safetensors layout would turn a format rejection into
        # a false architecture rejection.
        architecture_supported = True
        if format_supported:
            try:
                profile = self.inspect_model(model)
                limitations.extend(profile.limitations)
            except ValueError as exc:
                architecture_supported = False
                reasons.append(str(exc))

        tokenizer_supported = True
        if self.tokenizer_mode == "local-token-ids" and model.tokenizer_identity is None:
            limitations.append(
                "tokenizer identity is unavailable at probe time; immutable acquisition must provide it"
            )
        if not runtime.device_types:
            runtime_supported = False
            reasons.append("Colibri runtime advertises no executable device")

        supported = all(
            (
                runtime_supported,
                format_supported,
                quantization_supported,
                architecture_supported,
                tokenizer_supported,
            )
        )
        if supported:
            reasons.append(
                f"pinned Colibri runtime and adapter {self.adapter_id} validate the model profile"
            )
            classification = (
                ColibriCompatibilityStatus.SUPPORTED_WITH_LIMITATIONS
                if limitations
                else ColibriCompatibilityStatus.SUPPORTED
            )
        elif not format_supported:
            classification = ColibriCompatibilityStatus.INCOMPATIBLE_FORMAT
        else:
            classification = ColibriCompatibilityStatus.UNSUPPORTED_BY_PINNED_COLIBRI
        return ColibriSupportResult(
            supported=supported,
            classification=classification,
            architecture=model.architecture,
            adapter_id=self.adapter_id,
            reasons=tuple(dict.fromkeys(reasons)),
            limitations=tuple(dict.fromkeys(limitations)),
            runtime_version=runtime.runtime_version,
            model_format_supported=format_supported,
            architecture_supported=architecture_supported,
            quantization_supported=quantization_supported,
            tokenizer_supported=tokenizer_supported,
            cluster_supported=True,
            runtime_supported=runtime_supported,
            required_features=(self.checkpoint_layout, self.routing_kind),
        )

    def _required_memory_bytes(
        self,
        model: ResolvedModelDescriptor,
        profile: ColibriModelProfile,
    ) -> int:
        return int(model.weight_bytes)

    def build_execution_profile(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> ColibriExecutionProfile:
        del cluster
        profile = self.inspect_model(model)
        required = min(int(model.weight_bytes), self._required_memory_bytes(model, profile))
        movement = max(0, int(model.weight_bytes) - required)
        persistent = self.launch_mode == "persistent-gateway"
        return ColibriExecutionProfile(
            adapter_id=self.adapter_id,
            engine_basename=self.engine_basename,
            complete_model=self.complete_model,
            component_capabilities=self.component_capabilities,
            gateway_architecture=self.gateway_architecture,
            topology=self.topology,
            routing_mode=self.routing_kind,
            required_memory_bytes=required,
            expected_expert_movement_bytes=movement,
            persistent_worker=persistent,
            persistent_expert_residency=persistent,
            supports_streaming=persistent,
            supports_exact_replay=self.exact_replay,
            whole_expert_execution=True,
            tensor_microshards=self.tensor_microshards,
            direct_peer_model_data=self.direct_peer_model_data,
            coordinator_activation_relay=False,
            cache_policy="routing-aware bounded expert LRU",
            required_features=(self.checkpoint_layout, self.routing_kind),
            limitations=profile.limitations,
        )

    def validate_model_identity(
        self,
        model: ResolvedModelDescriptor,
        *,
        tensor_names: tuple[str, ...] = (),
    ) -> None:
        self.inspect_model(model)
        if tensor_names:
            self._validate_tensor_layout(tensor_names)

    def _validate_tensor_layout(self, tensor_names: tuple[str, ...]) -> None:
        raise NotImplementedError

    def map_tensor_names(
        self,
        tensor_names: tuple[tuple[str, tuple[int, ...], str, int], ...],
        *,
        config: dict[str, Any],
    ) -> tuple[ColibriTensorMapping, ...]:
        return tuple(
            self._map_tensor(
                name=name, shape=shape, dtype=dtype, byte_size=byte_size, config=config
            )
            for name, shape, dtype, byte_size in tensor_names
        )

    def _map_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        byte_size: int,
        config: dict[str, Any],
    ) -> ColibriTensorMapping:
        layer = _LAYER_RE.search(name)
        expert = _EXPERT_RE.search(name)
        shared = _SHARED_EXPERT_RE.search(name)
        role = _dense_role(name)
        return ColibriTensorMapping(
            tensor_name=name,
            tensor_role=role,
            layer_index=int(layer.group(1)) if layer else -1,
            expert_index=(
                int(expert.group(1))
                if expert
                else int(shared.group(1))
                if shared
                else 0
                if role == "shared_expert"
                else None
            ),
            logical_shape=shape,
            packed_shape=shape,
            quantization_format=dtype.casefold(),
            packing="safetensors-native",
            scale_format="none",
            reencoding_allowed=dtype.casefold() in {"f32", "f16", "bf16"},
            byte_size=byte_size,
        )

    def describe_experts(
        self,
        tensors: tuple[ColibriTensorMapping, ...],
        profile: ColibriModelProfile,
    ) -> tuple[ExpertDescriptor, ...]:
        grouped: dict[tuple[int, int, str], list[ColibriTensorMapping]] = defaultdict(list)
        fused: dict[int, list[ColibriTensorMapping]] = defaultdict(list)
        for tensor in tensors:
            if tensor.layer_index >= 0 and tensor.expert_index is not None:
                expert_type = "shared" if tensor.tensor_role == "shared_expert" else "routed"
                grouped[(tensor.layer_index, tensor.expert_index, expert_type)].append(tensor)
            elif tensor.layer_index >= 0 and tensor.tensor_role.startswith("routed_expert_"):
                fused[tensor.layer_index].append(tensor)
        experts: list[ExpertDescriptor] = []
        for (layer, expert, expert_type), items in sorted(grouped.items()):
            ordered = sorted(items, key=lambda item: item.tensor_name)
            shard_semantics = tuple(
                TensorShardSemantics(
                    tensor_role=item.tensor_role,
                    shard_axis=item.shard_axis,
                    reduction=(
                        ShardReduction.SUM
                        if item.projection in {"down", "down_proj", "w2"}
                        else ShardReduction.CONCATENATE
                    ),
                    alignment=item.scale_group_size or 1,
                    notes=(self._shard_semantics(),),
                )
                for item in ordered
                if item.shard_axis is not None
            )
            group = TensorGroupDescriptor(
                group_id=f"layer-{layer}:expert-{expert}",
                tensor_names=tuple(item.tensor_name for item in ordered),
                tensor_roles=tuple(item.tensor_role for item in ordered),
                tensor_shapes=tuple(item.logical_shape for item in ordered),
                parameter_count=sum(
                    _tensor_product(item.logical_shape)
                    for item in ordered
                    if not item.tensor_role.endswith(("_scale", "_shape"))
                ),
                memory_bytes=sum(item.byte_size for item in ordered),
                shard_semantics=shard_semantics,
            )
            experts.append(
                ExpertDescriptor(
                    layer_index=layer,
                    expert_index=expert,
                    expert_type=expert_type,
                    tensor_groups=(group,),
                    parameter_count=group.parameter_count,
                    memory_bytes=group.memory_bytes,
                    input_shape=(profile.hidden_size,) if profile.hidden_size else (),
                    output_shape=(profile.hidden_size,) if profile.hidden_size else (),
                    routing_metadata={
                        "router": self.routing_kind,
                        "expert_intermediate_size": profile.expert_intermediate_size,
                        "quantization_formats": sorted(
                            {item.quantization_format for item in ordered}
                        ),
                        "weight_group_size": next(
                            (
                                item.scale_group_size
                                for item in ordered
                                if item.scale_group_size is not None
                            ),
                            None,
                        ),
                    },
                )
            )
        if profile.expert_count is not None:
            for layer, items in sorted(fused.items()):
                ordered = sorted(items, key=lambda item: item.tensor_name)
                if not ordered or any(
                    not item.logical_shape or item.logical_shape[0] != profile.expert_count
                    for item in ordered
                ):
                    continue
                for expert in range(profile.expert_count):
                    parameter_count = sum(
                        _tensor_product(item.logical_shape[1:]) for item in ordered
                    )
                    memory_bytes = sum(item.byte_size // profile.expert_count for item in ordered)
                    group = TensorGroupDescriptor(
                        group_id=f"layer-{layer}:expert-{expert}",
                        tensor_names=tuple(item.tensor_name for item in ordered),
                        tensor_roles=tuple(item.tensor_role for item in ordered),
                        tensor_shapes=tuple(item.logical_shape[1:] for item in ordered),
                        parameter_count=parameter_count,
                        memory_bytes=memory_bytes,
                    )
                    experts.append(
                        ExpertDescriptor(
                            layer_index=layer,
                            expert_index=expert,
                            expert_type="routed",
                            tensor_groups=(group,),
                            parameter_count=parameter_count,
                            memory_bytes=memory_bytes,
                            input_shape=(profile.hidden_size,) if profile.hidden_size else (),
                            output_shape=(profile.hidden_size,) if profile.hidden_size else (),
                            routing_metadata={
                                "router": self.routing_kind,
                                "tensor_slices": {
                                    item.tensor_name: {"axis": 0, "index": expert}
                                    for item in ordered
                                },
                            },
                        )
                    )
        return tuple(
            sorted(
                experts,
                key=lambda item: (item.layer_index, item.expert_type, item.expert_index),
            )
        )

    def _shard_semantics(self) -> str:
        return "projection-output ranges with adapter-defined quantization alignment"

    def describe_routing(self, profile: ColibriModelProfile) -> ColibriRoutingDescriptor:
        if profile.expert_count is None or profile.experts_per_token is None:
            raise ValueError("complete expert topology is required to describe routing")
        return ColibriRoutingDescriptor(
            routing_kind=self.routing_kind,
            expert_count=profile.expert_count,
            experts_per_token=profile.experts_per_token,
            shared_expert_count=profile.shared_expert_count,
            normalization=self.routing_normalization,
            score_correction=self.routing_score_correction,
            routed_weight_semantics=self.routed_weight_semantics,
        )

    def validate_execution_result(
        self,
        *,
        output_token_ids: tuple[int, ...],
        requested_tokens: int,
    ) -> None:
        if requested_tokens <= 0:
            raise ValueError("requested token count must be positive")
        if not output_token_ids:
            raise ValueError(f"{self.adapter_id} returned no output token IDs")
        if len(output_token_ids) > requested_tokens:
            raise ValueError(f"{self.adapter_id} returned more tokens than requested")

    def validate_generation_request(
        self,
        *,
        stream: bool,
        temperature: float,
        top_p: float,
        has_token_ids: bool,
    ) -> None:
        del stream, temperature, top_p, has_token_ids

    def calibration_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        prompt: str,
        continuation_tokens: int,
    ) -> ColibriReplayInvocation:
        del engine, model_path, cap, prompt, continuation_tokens
        raise NotImplementedError(
            f"text-to-token calibration is not implemented by adapter {self.adapter_id}"
        )

    def replay_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        quant_bits: int,
        reference: Path,
        prompt_ids: tuple[int, ...],
        completion_tokens: int,
        teacher_forced: bool,
    ) -> ColibriReplayInvocation:
        del engine, model_path, cap, quant_bits, reference, prompt_ids, completion_tokens
        del teacher_forced
        raise NotImplementedError(f"replay is not implemented by adapter {self.adapter_id}")


class Glm52ColibriAdapter(_BaseColibriAdapter):
    adapter_id = "glm-5.2"
    claimed_architectures = (
        "glm_moe",
        "glm_moe_dsa",
        "GlmMoeDsaForCausalLM",
        "GlmMoeDsaModel",
    )
    engine_basename = "colibri"
    gateway_architecture = "glm"
    supports_text_calibration = True
    exact_replay = True
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    accepted_quantizations = frozenset(
        {
            "int2",
            "int2-row",
            "int3",
            "int3-g64",
            "int4",
            "int4-g64",
            "int4-grouped",
            "int8",
            "int8-row",
            "fp8",
            "fp8-e4m3",
        }
    )
    native_quantization = "int4-g64"
    checkpoint_layout = "glm52-colibri-projection-container-v1"
    routing_kind = "sigmoid-noaux-top-k"
    routing_normalization = "norm_topk_prob configuration"
    routing_score_correction = "e_score_correction_bias"
    routed_weight_semantics = "raw sigmoid scores scaled by routed_scaling_factor"

    def _config_values(self, config: dict[str, Any]) -> dict[str, int]:
        values = super()._config_values(config)
        n_group = _number(config, ("n_group",))
        if n_group != 1:
            raise ValueError("pinned GLM-5.2 Colibri requires n_group=1")
        return values

    def _validate_tensor_layout(self, tensor_names: tuple[str, ...]) -> None:
        names = set(tensor_names)
        suffixes = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        if not all(
            any(".mlp.experts." in name and name.endswith(suffix) for name in names)
            for suffix in suffixes
        ):
            raise ValueError("GLM-5.2 Colibri requires per-expert gate/up/down projection tensors")
        if not any(name.endswith("mlp.gate.weight") for name in names):
            raise ValueError("GLM-5.2 Colibri checkpoint is missing the MoE router tensor")

    def _required_memory_bytes(
        self, model: ResolvedModelDescriptor, profile: ColibriModelProfile
    ) -> int:
        if not all(
            (
                profile.layer_count,
                profile.hidden_size,
                profile.expert_count,
                profile.expert_intermediate_size,
            )
        ):
            return int(model.weight_bytes)
        config = _effective_config(model.configuration)
        first_dense = max(0, _number(config, ("first_k_dense_replace",)) or 0)
        sparse_layers = max(0, int(profile.layer_count or 0) - first_dense)
        bits = 4
        quant = _declared_quantization(model) or ""
        match = re.search(r"(?:int|fp)(\d+)", quant)
        if match:
            bits = min(32, max(2, int(match.group(1))))
        per_expert = math.ceil(
            3
            * int(profile.hidden_size or 0)
            * int(profile.expert_intermediate_size or 0)
            * bits
            / 8
        )
        bank = min(
            int(model.weight_bytes),
            sparse_layers * int(profile.expert_count or 0) * per_expert,
        )
        dense = max(0, int(model.weight_bytes) - bank)
        cache = bank * min(16, int(profile.expert_count or 1)) // int(profile.expert_count or 1)
        return dense + cache + 2 * 1024**3

    def _map_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        byte_size: int,
        config: dict[str, Any],
    ) -> ColibriTensorMapping:
        base = super()._map_tensor(
            name=name, shape=shape, dtype=dtype, byte_size=byte_size, config=config
        )
        expert = _EXPERT_RE.search(name)
        if expert is None:
            return base
        projection = next(
            (value for value in ("gate_proj", "up_proj", "down_proj") if f".{value}." in name),
            None,
        )
        if projection is None:
            return base
        role = {
            "gate_proj": "routed_expert_gate_projection",
            "up_proj": "routed_expert_up_projection",
            "down_proj": "routed_expert_down_projection",
        }[projection]
        quant = (
            _normalise_quantization(_declared_quantization_from_config(config)) or dtype.casefold()
        )
        return base.model_copy(
            update={
                "tensor_role": role,
                "projection": projection.removesuffix("_proj"),
                "shard_axis": 0,
                "quantization_format": quant,
                "packing": "colibri-quantized-projection-container",
                "scale_format": "adapter-validated-sidecar",
                "reencoding_allowed": False,
            }
        )

    def calibration_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        prompt: str,
        continuation_tokens: int,
    ) -> ColibriReplayInvocation:
        return ColibriReplayInvocation(
            command=(str(engine), str(cap)),
            environment={
                "SNAP": str(model_path),
                "PROMPT": prompt,
                "NGEN": str(continuation_tokens),
                "TOKENS": "1",
                "PROF": "1",
            },
            exact_replay=False,
        )

    def replay_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        quant_bits: int,
        reference: Path,
        prompt_ids: tuple[int, ...],
        completion_tokens: int,
        teacher_forced: bool,
    ) -> ColibriReplayInvocation:
        del model_path, quant_bits, prompt_ids, completion_tokens
        if not teacher_forced:
            raise ValueError("GLM replay invocation requires teacher-forced tokens")
        return ColibriReplayInvocation(
            command=(str(engine), str(cap)),
            environment={"REF": str(reference), "REF_FORCE": "1", "REPLAY": "1"},
            exact_replay=True,
        )


def _declared_quantization_from_config(config: dict[str, Any]) -> str | None:
    return quantization_from_config(config)


class _ComposableSparseMoeColibriAdapter(_BaseColibriAdapter):
    """Swarm-owned Colibri expert component for standard routed MoE layouts."""

    engine_basename = "swarm_moe"
    gateway_architecture = None
    accepted_quantizations = frozenset(
        {
            "bf16",
            "bfloat16",
            "f16",
            "float16",
            "fp8",
            "fp8-e4m3",
            "fp8-float8-e4m3fn",
            "int4-g32",
        }
    )
    native_quantization = "checkpoint-native"
    checkpoint_layout = "swarm-colibri-adapter-described-experts-v1"
    routing_normalization = "adapter configuration"
    routed_weight_semantics = "adapter-provided selected weights scale exact expert outputs"
    topology = "hybrid-native-stage-colibri-experts"
    complete_model = False
    component_capabilities = (
        "moe-routing",
        "expert-execution",
        "expert-storage-tiering",
        "expert-placement",
        "expert-microsharding",
    )
    tensor_microshards = True
    direct_peer_model_data = True
    static_limitations = (
        "this pinned Swarm extension executes routed-expert components; attention, KV, "
        "tokenization, and sampling require a compatible hybrid plan",
        "packed symmetric INT4-G32 uses the Swarm Colibri reference ABI; other packed "
        "integer layouts require a quantization-specific kernel",
    )

    def _validate_tensor_layout(self, tensor_names: tuple[str, ...]) -> None:
        names = set(tensor_names)
        separate = all(
            any(
                (".experts." in name or ".local_experts." in name) and f".{projection}." in name
                for name in names
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        w123 = all(
            any((".experts." in name) and f".{projection}." in name for name in names)
            for projection in ("w1", "w2", "w3")
        )
        fused = any(
            ".experts." in name and ("gate_up_proj" in name or "gate_up_proj" in name.casefold())
            for name in names
        ) and any(".experts." in name and "down_proj" in name for name in names)
        if not (separate or w123 or fused):
            raise ValueError(
                f"{self.adapter_id} has no adapter-recognized routed expert tensor groups"
            )
        if any(name.endswith(".weight_packed") for name in names):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = next(
                    (
                        name.removesuffix(".weight_packed")
                        for name in names
                        if f".{projection}." in name and name.endswith(".weight_packed")
                    ),
                    None,
                )
                if prefix is None or not all(
                    f"{prefix}.{suffix}" in names for suffix in ("weight_scale", "weight_shape")
                ):
                    raise ValueError(
                        f"{self.adapter_id} packed {projection} lacks scale/shape metadata"
                    )

    def _required_memory_bytes(
        self, model: ResolvedModelDescriptor, profile: ColibriModelProfile
    ) -> int:
        if not all(
            (
                profile.layer_count,
                profile.hidden_size,
                profile.expert_count,
                profile.expert_intermediate_size,
            )
        ):
            return int(model.weight_bytes)
        config = _effective_config(model.configuration)
        sparse_layers = max(
            1,
            int(profile.layer_count or 1)
            - max(0, _number(config, ("first_k_dense_replace",)) or 0),
        )
        experts = int(profile.expert_count or 1)
        hot_experts = min(experts, max(1, int(profile.experts_per_token or 1) * 2))
        average_expert = max(1, int(model.weight_bytes) // max(sparse_layers * experts, 1))
        # This adapter is a routed-expert component.  Dense weights are owned by
        # the outer component and must not be double-counted during hybrid
        # admission.  Keep two route sets plus an explicit runtime/workspace
        # reserve resident; all other experts remain in the RAM/NVMe tiers.
        return min(int(model.weight_bytes), hot_experts * average_expert + 1024**3)

    def _map_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        byte_size: int,
        config: dict[str, Any],
    ) -> ColibriTensorMapping:
        base = super()._map_tensor(
            name=name, shape=shape, dtype=dtype, byte_size=byte_size, config=config
        )
        lowered = name.casefold()
        expert = _EXPERT_RE.search(name)
        if expert is None and ".experts." not in lowered:
            return base
        projection = next(
            (
                value
                for value in (
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "gate_up_proj",
                    "w1",
                    "w2",
                    "w3",
                )
                if f".{value}" in lowered
            ),
            None,
        )
        if projection is None:
            return base
        semantic = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
            "gate_up_proj": "gate+up",
            "w1": "gate",
            "w2": "down",
            "w3": "up",
        }[projection]
        quantization = _normalise_quantization(_declared_quantization_from_config(config))
        storage = (
            "shape"
            if lowered.endswith(".weight_shape")
            else "scale"
            if any(
                marker in lowered for marker in ("weight_scale", "scale_inv", "weight_scale_inv")
            )
            else "packed"
            if lowered.endswith(".weight_packed")
            else "weight"
        )
        group_match = re.search(r"-g(\d+)$", quantization or "")
        group_size = int(group_match.group(1)) if group_match else None
        effective_quantization = quantization or dtype.casefold()
        logical_shape = (
            (shape[0], shape[1] * 8)
            if storage == "packed" and quantization == "int4-g32" and len(shape) == 2
            else shape
        )
        return base.model_copy(
            update={
                "tensor_role": (
                    f"routed_expert_{semantic}_projection"
                    if storage in {"weight", "packed"}
                    else f"routed_expert_{semantic}_{storage}"
                ),
                "projection": semantic,
                "shard_axis": (
                    None if storage in {"scale", "shape"} else 1 if semantic == "down" else 0
                ),
                "logical_shape": logical_shape,
                "quantization_format": (
                    effective_quantization
                    if storage in {"weight", "packed"}
                    else "int32-metadata"
                    if storage == "shape"
                    else dtype.casefold()
                ),
                "packing": (
                    "compressed-tensors-signed-offset-eight-nibbles-i32"
                    if storage == "packed" and quantization == "int4-g32"
                    else "adapter-described-checkpoint-native"
                ),
                "scale_format": "bf16-per-group"
                if quantization == "int4-g32"
                else "checkpoint-native",
                "scale_group_size": group_size,
                "reencoding_allowed": False,
            }
        )


class Qwen3MoeColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "qwen3-moe"
    claimed_architectures = ("qwen3_moe", "Qwen3MoeForCausalLM")
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    routing_kind = "softmax-top-k"


class Qwen35MoeColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "qwen3-5-moe"
    claimed_architectures = (
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "Qwen3_5MoeForConditionalGeneration",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_experts", "n_routed_experts", "num_local_experts"),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size", "intermediate_size"),
    }
    routing_kind = "softmax-top-k"


class KimiK2ColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "kimi-k2-moe"
    claimed_architectures = (
        "kimi_k2_moe",
        "kimi_k2",
        "kimi_k25",
        "KimiK25ForConditionalGeneration",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    routing_kind = "sigmoid-noaux-top-k"
    routing_score_correction = "e_score_correction_bias when present"


class DeepSeekV3ColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "deepseek-v3-moe"
    claimed_architectures = (
        "deepseek_v3_moe",
        "deepseek_v3",
        "deepseek_v32",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    routing_kind = "sigmoid-noaux-grouped-top-k"
    routing_score_correction = "e_score_correction_bias"


class DeepSeekV4ColibriAdapter(_ComposableSparseMoeColibriAdapter):
    """Component contract for V4 routed experts.

    V4 cannot reuse the V3 outer model or routing contract.  The official
    checkpoints mix packed FP4 experts with FP8 dense tensors and start with
    token-id hash-routed layers.  The adapter therefore makes those
    requirements visible during planning; the floating PyTorch component
    rejects the packed tensors and an external pin-bound Colibri runtime must
    advertise this adapter before such a plan is executable.
    """

    adapter_id = "deepseek-v4-moe"
    claimed_architectures = (
        "deepseek_v4_moe",
        "deepseek_v4",
        "DeepseekV4ForCausalLM",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    checkpoint_layout = "deepseek-v4-mixed-fp4-fp8-v1"
    routing_kind = "static-hash-then-sqrtsoftplus-noaux-top-k"
    routing_score_correction = "e_score_correction_bias after hash-routed layers"
    static_limitations = _ComposableSparseMoeColibriAdapter.static_limitations + (
        "official V4 packed FP4 expert tensors require a pin-bound V4 Colibri kernel; "
        "the embedded floating PyTorch component fails closed",
    )

    def _architecture_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            key: config[key]
            for key in (
                "expert_dtype",
                "num_hash_layers",
                "scoring_func",
                "swiglu_limit",
                "hc_mult",
                "hc_eps",
                "index_topk",
                "sliding_window",
            )
            if key in config
        }


class MiniMaxMoeColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "minimax-moe"
    claimed_architectures = (
        "minimax_moe",
        "minimax",
        "minimax_m2",
        "minimax_m3_vl",
        "MiniMaxForCausalLM",
        "MiniMaxM2ForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_local_experts", "n_routed_experts"),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("intermediate_size", "moe_intermediate_size"),
    }
    routing_kind = "sigmoid-routing-bias-top-k"
    routing_score_correction = "routing bias"


class MixtralColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "mixtral-moe"
    claimed_architectures = ("mixtral_moe", "mixtral", "MixtralForCausalLM")
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_local_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("intermediate_size",),
    }
    routing_kind = "softmax-top-k"


class Llama4MoeColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "llama4-moe"
    claimed_architectures = (
        "llama4_moe",
        "llama4",
        "llama4_text",
        "Llama4ForConditionalGeneration",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_local_experts", "num_experts"),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("intermediate_size", "moe_intermediate_size"),
    }
    routing_kind = "softmax-top-k"


class Mistral4MoeColibriAdapter(_ComposableSparseMoeColibriAdapter):
    adapter_id = "mistral4-moe"
    claimed_architectures = (
        "mistral4_moe",
        "mistral4",
        "Mistral3ForConditionalGeneration",
    )
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size",),
    }
    routing_kind = "sigmoid-top-k"


class KimiK3ColibriAdapter(_BaseColibriAdapter):
    adapter_id = "kimi-k3"
    claimed_architectures = (
        "kimi_k3_moe",
        "kimi_k3",
        "kimi_k3_text",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
    )
    engine_basename = "kimi_k3"
    gateway_architecture = "kimi"
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("num_experts",),
        "top_k": ("num_experts_per_token",),
        "intermediate": ("moe_intermediate_size",),
    }
    accepted_quantizations = frozenset({"mxfp4", "mxfp4-pack-quantized"})
    native_quantization = "mxfp4"
    checkpoint_layout = "kimi-k3-native-mxfp4-v1"
    routing_kind = "stable-latent-moe-sigmoid-top-k"
    routing_normalization = "selected raw sigmoid scores renormalized"
    routing_score_correction = "e_score_correction_bias"
    routed_weight_semantics = "top-k latent experts plus shared full-width experts"
    static_limitations = ("pinned Kimi K3 path is text-only and does not load the vision tower",)

    def _config_values(self, config: dict[str, Any]) -> dict[str, int]:
        values = super()._config_values(config)
        effective = _effective_config(config)
        for field in (
            "routed_expert_hidden_size",
            "first_k_dense_replace",
            "attn_res_block_size",
        ):
            if _number(effective, (field,)) is None:
                raise ValueError(f"Kimi K3 config is missing required numeric field {field}")
        linear = effective.get("linear_attn_config")
        if not isinstance(linear, dict) or not isinstance(linear.get("kda_layers"), list):
            raise ValueError("Kimi K3 config requires linear_attn_config.kda_layers")
        latent = int(effective["routed_expert_hidden_size"])
        if latent < 32 or latent % 32 or values["intermediate"] % 32:
            raise ValueError("Kimi K3 latent and expert dimensions must be multiples of 32")
        return values

    def _architecture_metadata(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "routed_expert_hidden_size": int(config["routed_expert_hidden_size"]),
            "attention_residual_block_size": int(config["attn_res_block_size"]),
            "linear_attention_layers": tuple(config["linear_attn_config"]["kda_layers"]),
        }

    def _validate_tensor_layout(self, tensor_names: tuple[str, ...]) -> None:
        names = set(tensor_names)
        for projection in ("w1", "w2", "w3"):
            if not any(
                ".block_sparse_moe.experts." in name
                and name.endswith(f".{projection}.weight_packed")
                for name in names
            ):
                raise ValueError(
                    f"Kimi K3 checkpoint is missing native {projection}.weight_packed experts"
                )
            if not any(
                ".block_sparse_moe.experts." in name
                and name.endswith(f".{projection}.weight_scale")
                for name in names
            ):
                raise ValueError(
                    f"Kimi K3 checkpoint is missing UE8M0 {projection}.weight_scale tensors"
                )

    def _required_memory_bytes(
        self, model: ResolvedModelDescriptor, profile: ColibriModelProfile
    ) -> int:
        effective = _effective_config(model.configuration)
        latent = _number(effective, ("routed_expert_hidden_size",))
        if not all(
            (
                profile.layer_count,
                profile.expert_count,
                profile.expert_intermediate_size,
                latent,
            )
        ):
            return int(model.weight_bytes)
        per_expert = math.ceil(
            3 * int(latent or 0) * int(profile.expert_intermediate_size or 0) * 0.5 * 1.07
        )
        bank = min(
            int(model.weight_bytes),
            int(profile.layer_count or 0) * int(profile.expert_count or 0) * per_expert,
        )
        dense = max(0, int(model.weight_bytes) - bank)
        cache = int(profile.layer_count or 0) * per_expert
        return dense + cache + 2 * 1024**3

    def _map_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        byte_size: int,
        config: dict[str, Any],
    ) -> ColibriTensorMapping:
        base = super()._map_tensor(
            name=name, shape=shape, dtype=dtype, byte_size=byte_size, config=config
        )
        expert = _EXPERT_RE.search(name)
        match = re.search(r"\.(w[123])\.weight_(packed|scale)$", name)
        if expert is None or match is None:
            return base
        projection_name, storage = match.groups()
        projection = {"w1": "gate", "w2": "down", "w3": "up"}[projection_name]
        role = (
            "routed_expert_scale"
            if storage == "scale"
            else f"routed_expert_{projection}_projection"
        )
        effective = _effective_config(config)
        latent = _number(effective, ("routed_expert_hidden_size",)) or 0
        intermediate = _number(effective, ("moe_intermediate_size",)) or 0
        logical = shape
        if latent and intermediate and storage == "packed":
            logical = (latent, intermediate) if projection == "down" else (intermediate, latent)
        return base.model_copy(
            update={
                "tensor_role": role,
                "projection": projection,
                "shard_axis": 0,
                "logical_shape": logical,
                "quantization_format": "ue8m0" if storage == "scale" else "mxfp4",
                "packing": (
                    "one_scale_per_32_values" if storage == "scale" else "e2m1_two_nibbles"
                ),
                "scale_format": "ue8m0",
                "scale_group_size": 32,
                "quantization_aware_trained": True,
                "reencoding_allowed": False,
            }
        )

    def replay_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        quant_bits: int,
        reference: Path,
        prompt_ids: tuple[int, ...],
        completion_tokens: int,
        teacher_forced: bool,
    ) -> ColibriReplayInvocation:
        del cap, quant_bits, reference, teacher_forced
        return ColibriReplayInvocation(
            command=(
                str(engine),
                str(model_path),
                "--ids",
                " ".join(str(value) for value in prompt_ids),
                "--ngen",
                str(completion_tokens),
            ),
            exact_replay=False,
        )


class InklingColibriAdapter(_BaseColibriAdapter):
    adapter_id = "inkling"
    claimed_architectures = (
        "inkling",
        "inkling_text",
        "InklingForCausalLM",
        "InklingForConditionalGeneration",
    )
    engine_basename = "inkling"
    gateway_architecture = "inkling"
    required_config: ClassVar[dict[str, tuple[str, ...]]] = {
        "layers": ("num_hidden_layers",),
        "hidden": ("hidden_size",),
        "experts": ("n_routed_experts",),
        "top_k": ("num_experts_per_tok",),
        "intermediate": ("moe_intermediate_size", "intermediate_size"),
    }
    accepted_quantizations = frozenset(
        {"int4", "int4-g64", "int4-grouped", "bf16", "bfloat16", "mixed-int4-bf16"}
    )
    native_quantization = "mixed-int4-bf16"
    checkpoint_layout = "inkling-fused-expert-axis-v1"
    routing_kind = "sigmoid-loss-free-bias-top-k"
    routing_normalization = "joint routed/shared route scaling"
    routing_score_correction = "loss-free expert bias"
    routed_weight_semantics = "top-k routed plus configured shared experts"
    static_limitations = (
        "pinned Inkling path is text-only; vision, audio, and MTP weights are not loaded",
    )

    def _validate_tensor_layout(self, tensor_names: tuple[str, ...]) -> None:
        names = set(tensor_names)
        if not any(name.endswith("mlp.experts.gate_up_proj") for name in names):
            raise ValueError("Inkling checkpoint is missing fused expert gate_up_proj banks")
        if not any(name.endswith("mlp.experts.down_proj") for name in names):
            raise ValueError("Inkling checkpoint is missing fused expert down_proj banks")

    def _required_memory_bytes(
        self, model: ResolvedModelDescriptor, profile: ColibriModelProfile
    ) -> int:
        del profile
        quant = _declared_quantization(model) or ""
        dense_int4 = "dense-int4" in quant or bool(
            model.artifact_metadata.get("inkling_dense_int4")
        )
        minimum = 25 * 1024**3 if dense_int4 else 100 * 1024**3
        return min(int(model.weight_bytes), minimum)

    def _map_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        dtype: str,
        byte_size: int,
        config: dict[str, Any],
    ) -> ColibriTensorMapping:
        base = super()._map_tensor(
            name=name, shape=shape, dtype=dtype, byte_size=byte_size, config=config
        )
        if name.endswith("mlp.experts.gate_up_proj"):
            return base.model_copy(
                update={
                    "tensor_role": "routed_expert_fused_gate_up_bank",
                    "projection": "gate+up",
                    "shard_axis": 0,
                    "quantization_format": "int4-grouped",
                    "packing": "expert-axis-fused-int4",
                    "scale_format": "float32-per-group",
                    "scale_group_size": 64,
                    "reencoding_allowed": False,
                }
            )
        if name.endswith("mlp.experts.down_proj"):
            return base.model_copy(
                update={
                    "tensor_role": "routed_expert_down_bank",
                    "projection": "down",
                    "shard_axis": 0,
                    "quantization_format": "int4-grouped",
                    "packing": "expert-axis-fused-int4",
                    "scale_format": "float32-per-group",
                    "scale_group_size": 64,
                    "reencoding_allowed": False,
                }
            )
        return base

    def describe_experts(
        self,
        tensors: tuple[ColibriTensorMapping, ...],
        profile: ColibriModelProfile,
    ) -> tuple[ExpertDescriptor, ...]:
        banks: dict[int, list[ColibriTensorMapping]] = defaultdict(list)
        for tensor in tensors:
            if tensor.layer_index >= 0 and tensor.tensor_role in {
                "routed_expert_fused_gate_up_bank",
                "routed_expert_down_bank",
            }:
                banks[tensor.layer_index].append(tensor)
        if profile.expert_count is None:
            return ()
        experts: list[ExpertDescriptor] = []
        for layer, items in sorted(banks.items()):
            ordered = sorted(items, key=lambda item: item.tensor_name)
            for expert in range(profile.expert_count):
                group = TensorGroupDescriptor(
                    group_id=f"layer-{layer}:expert-bank-slice-{expert}",
                    tensor_names=tuple(item.tensor_name for item in ordered),
                    tensor_roles=tuple(item.tensor_role for item in ordered),
                    tensor_shapes=tuple(item.logical_shape for item in ordered),
                    parameter_count=(
                        sum(_tensor_product(item.logical_shape) for item in ordered)
                        // profile.expert_count
                    ),
                    memory_bytes=sum(item.byte_size for item in ordered) // profile.expert_count,
                    shard_semantics=(
                        TensorShardSemantics(
                            tensor_role="fused_expert_bank",
                            shard_axis=0,
                            reduction=ShardReduction.GATHER,
                            notes=("axis zero is sliced at the adapter-described expert index",),
                        ),
                    ),
                )
                experts.append(
                    ExpertDescriptor(
                        layer_index=layer,
                        expert_index=expert,
                        expert_type="grouped",
                        tensor_groups=(group,),
                        parameter_count=group.parameter_count,
                        memory_bytes=group.memory_bytes,
                        input_shape=(profile.hidden_size,) if profile.hidden_size else (),
                        output_shape=(profile.hidden_size,) if profile.hidden_size else (),
                        routing_metadata={
                            "router": self.routing_kind,
                            "tensor_slice_axis": 0,
                            "tensor_slice_index": expert,
                            "expert_intermediate_size": profile.expert_intermediate_size,
                            "quantization_formats": sorted(
                                {item.quantization_format for item in ordered}
                            ),
                        },
                    )
                )
        return tuple(experts)

    def replay_invocation(
        self,
        *,
        engine: Path,
        model_path: Path,
        cap: int,
        quant_bits: int,
        reference: Path,
        prompt_ids: tuple[int, ...],
        completion_tokens: int,
        teacher_forced: bool,
    ) -> ColibriReplayInvocation:
        del model_path, prompt_ids, completion_tokens, teacher_forced
        return ColibriReplayInvocation(
            command=(str(engine), str(cap), str(quant_bits), str(reference)),
            exact_replay=False,
        )


def default_colibri_adapter_registry() -> ColibriArchitectureAdapterRegistry:
    """Load built-in and third-party adapters through the wheel-safe entry-point boundary."""

    targets: dict[str, str] = {}
    for entry_point in importlib.metadata.entry_points(group="swarm_inference.colibri_adapters"):
        targets[entry_point.name] = entry_point.value
    for parent in Path(__file__).resolve().parents:
        project_file = parent / "pyproject.toml"
        if not project_file.is_file():
            continue
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))
        configured = (
            project.get("project", {})
            .get("entry-points", {})
            .get("swarm_inference.colibri_adapters", {})
        )
        for name, target in configured.items():
            targets.setdefault(str(name), str(target))
        break
    # Installed wheels always carry entry-point metadata.  This explicit
    # built-in fallback keeps source-tree and zip-import tests deterministic.
    if not targets:
        targets = {
            "deepseek-v3-moe": "swarm_inference.backends.colibri.adapters:DeepSeekV3ColibriAdapter",
            "deepseek-v4-moe": "swarm_inference.backends.colibri.adapters:DeepSeekV4ColibriAdapter",
            "glm-5.2": "swarm_inference.backends.colibri.adapters:Glm52ColibriAdapter",
            "inkling": "swarm_inference.backends.colibri.adapters:InklingColibriAdapter",
            "kimi-k2-moe": "swarm_inference.backends.colibri.adapters:KimiK2ColibriAdapter",
            "kimi-k3": "swarm_inference.backends.colibri.adapters:KimiK3ColibriAdapter",
            "llama4-moe": "swarm_inference.backends.colibri.adapters:Llama4MoeColibriAdapter",
            "minimax-moe": "swarm_inference.backends.colibri.adapters:MiniMaxMoeColibriAdapter",
            "mistral4-moe": "swarm_inference.backends.colibri.adapters:Mistral4MoeColibriAdapter",
            "mixtral-moe": "swarm_inference.backends.colibri.adapters:MixtralColibriAdapter",
            "qwen3-5-moe": "swarm_inference.backends.colibri.adapters:Qwen35MoeColibriAdapter",
            "qwen3-moe": "swarm_inference.backends.colibri.adapters:Qwen3MoeColibriAdapter",
        }
    adapters: list[ColibriArchitectureAdapter] = []
    for name, target in sorted(targets.items()):
        module_name, separator, attribute_name = target.partition(":")
        if not separator or not module_name or not attribute_name:
            raise RuntimeError(f"Colibri adapter entry point {name!r} is invalid")
        factory: Any = importlib.import_module(module_name)
        for attribute in attribute_name.split("."):
            factory = getattr(factory, attribute)
        adapters.append(factory())
    return ColibriArchitectureAdapterRegistry(tuple(adapters))


__all__ = [
    "DeepSeekV3ColibriAdapter",
    "DeepSeekV4ColibriAdapter",
    "Glm52ColibriAdapter",
    "InklingColibriAdapter",
    "KimiK2ColibriAdapter",
    "KimiK3ColibriAdapter",
    "Llama4MoeColibriAdapter",
    "MiniMaxMoeColibriAdapter",
    "Mistral4MoeColibriAdapter",
    "MixtralColibriAdapter",
    "Qwen3MoeColibriAdapter",
    "Qwen35MoeColibriAdapter",
    "default_colibri_adapter_registry",
]
