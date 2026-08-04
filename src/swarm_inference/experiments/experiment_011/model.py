"""Real OLMoE contiguous-stage loading and stateful execution."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeDecoderLayer,
    OlmoeModel,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
)

from swarm_inference.experiments.experiment_011.partition import StageAssignment


@dataclass(slots=True)
class StageSessionState:
    cache: DynamicCache
    sequence_length: int = 0
    closed: bool = False


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    hidden_states: torch.Tensor
    stage_boundary_hidden_states: torch.Tensor
    router_logits: tuple[torch.Tensor, ...]
    final_hidden_states: torch.Tensor | None
    logits: torch.Tensor | None
    sampled_token_ids: torch.Tensor | None
    all_sampled_token_ids: torch.Tensor | None
    cache_sequence_length: int
    compute_ns: int


@dataclass(frozen=True, slots=True)
class WeightOwnership:
    stage_id: int
    layer_start: int
    layer_end: int
    parameter_names: tuple[str, ...]
    parameter_bytes: int
    parameter_count: int
    owns_embeddings: bool
    owns_final_norm: bool
    owns_output_projection: bool
    ownership_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "parameter_names": list(self.parameter_names),
            "parameter_bytes": self.parameter_bytes,
            "parameter_count": self.parameter_count,
            "owns_embeddings": self.owns_embeddings,
            "owns_final_norm": self.owns_final_norm,
            "owns_output_projection": self.owns_output_projection,
            "ownership_hash": self.ownership_hash,
        }


class SafeTensorRepository:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        index_path = self.model_path / "model.safetensors.index.json"
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = raw["weight_map"]

    def load_module(self, module: nn.Module, *, global_prefix: str) -> tuple[list[str], int]:
        local_keys = list(module.state_dict().keys())
        global_keys = [f"{global_prefix}{key}" for key in local_keys]
        missing = [key for key in global_keys if key not in self.weight_map]
        if missing:
            raise KeyError(f"model shard index is missing stage parameters: {missing[:3]}")
        grouped: dict[str, list[tuple[str, str]]] = {}
        for local_key, global_key in zip(local_keys, global_keys, strict=True):
            grouped.setdefault(self.weight_map[global_key], []).append((local_key, global_key))
        state: dict[str, torch.Tensor] = {}
        for shard, pairs in grouped.items():
            with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
                for local_key, global_key in pairs:
                    state[local_key] = handle.get_tensor(global_key)
        module.load_state_dict(state, strict=True, assign=True)
        byte_count = sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        return global_keys, byte_count


class ContiguousOlmoeStage(nn.Module):
    """Only the modules and KV entries owned by one contiguous stage."""

    def __init__(
        self,
        *,
        model_path: Path,
        assignment: StageAssignment,
        stage_count: int,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_path = model_path.resolve()
        self.assignment = assignment
        self.stage_count = stage_count
        self.device_name = device
        self.device = torch.device(device)
        self.dtype = dtype
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        config._attn_implementation = "eager"
        self.global_config = config
        self.local_config = copy.deepcopy(config)
        self.local_config.num_hidden_layers = len(assignment.layer_ids)
        repository = SafeTensorRepository(self.model_path)
        owned_names: list[str] = []
        owned_bytes = 0

        self.embed_tokens: nn.Embedding | None = None
        if assignment.owns_embeddings:
            with torch.device("meta"):
                embedding = nn.Embedding(
                    config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
                )
            names, size = repository.load_module(embedding, global_prefix="model.embed_tokens.")
            self.embed_tokens = embedding.to(device=self.device, dtype=dtype)
            owned_names.extend(names)
            owned_bytes += size

        layers = []
        for local_layer_id, global_layer_id in enumerate(assignment.layer_ids):
            with torch.device("meta"):
                layer = OlmoeDecoderLayer(self.local_config, local_layer_id)
            names, size = repository.load_module(
                layer, global_prefix=f"model.layers.{global_layer_id}."
            )
            layer = layer.to(device=self.device, dtype=dtype)
            layer.eval()
            layers.append(layer)
            owned_names.extend(names)
            owned_bytes += size
            gc.collect()
        self.layers = nn.ModuleList(layers)
        self.rotary_emb = OlmoeRotaryEmbedding(self.local_config, device=self.device)

        self.norm: OlmoeRMSNorm | None = None
        self.lm_head: nn.Linear | None = None
        if assignment.owns_final_norm:
            with torch.device("meta"):
                norm = OlmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            names, size = repository.load_module(norm, global_prefix="model.norm.")
            self.norm = norm.to(device=self.device, dtype=dtype)
            owned_names.extend(names)
            owned_bytes += size
        if assignment.owns_output_projection:
            with torch.device("meta"):
                head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            names, size = repository.load_module(head, global_prefix="lm_head.")
            self.lm_head = head.to(device=self.device, dtype=dtype)
            owned_names.extend(names)
            owned_bytes += size

        self.eval()
        ownership_payload = json.dumps(sorted(owned_names), separators=(",", ":")).encode("utf-8")
        self.ownership = WeightOwnership(
            stage_id=assignment.stage_id,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            parameter_names=tuple(sorted(owned_names)),
            parameter_bytes=owned_bytes,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
            owns_embeddings=assignment.owns_embeddings,
            owns_final_norm=assignment.owns_final_norm,
            owns_output_projection=assignment.owns_output_projection,
            ownership_hash=hashlib.sha256(ownership_payload).hexdigest(),
        )
        self.sessions: dict[str, StageSessionState] = {}

    def open_session(self, session_id: str) -> None:
        if session_id in self.sessions and not self.sessions[session_id].closed:
            raise ValueError("stage session is already open")
        self.sessions[session_id] = StageSessionState(cache=DynamicCache(config=self.local_config))

    def close_session(self, session_id: str) -> int:
        state = self._session(session_id)
        bytes_before = self.kv_cache_bytes(session_id)
        state.closed = True
        del self.sessions[session_id]
        return bytes_before

    def cancel_session(self, session_id: str) -> int:
        return self.close_session(session_id)

    def crop_session(self, session_id: str, sequence_length: int) -> None:
        state = self._session(session_id)
        if sequence_length < 0 or sequence_length > state.sequence_length:
            raise ValueError("invalid KV-cache rollback position")
        state.cache.crop(sequence_length)
        state.sequence_length = sequence_length

    def _session(self, session_id: str) -> StageSessionState:
        state = self.sessions.get(session_id)
        if state is None or state.closed:
            raise ValueError("stage session is not open")
        return state

    def kv_cache_bytes(self, session_id: str) -> int:
        state = self._session(session_id)
        total = 0
        for layer in state.cache.layers:
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if isinstance(tensor, torch.Tensor):
                    total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _causal_mask(
        hidden_states: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[1]
        target_length = past_seen_tokens + sequence_length + 1
        return OlmoeModel._prepare_4d_causal_attention_mask_with_cache_position(
            None,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            cache_position=cache_position,
            batch_size=hidden_states.shape[0],
        )

    @torch.inference_mode()
    def execute(
        self,
        *,
        session_id: str,
        cache_position_start: int,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        capture_router_logits: bool = False,
    ) -> StageExecutionResult:
        state = self._session(session_id)
        if cache_position_start != state.sequence_length:
            raise ValueError(
                f"stage cache position {cache_position_start} does not match owned state "
                f"length {state.sequence_length}"
            )
        if (input_ids is None) == (hidden_states is None):
            raise ValueError("provide exactly one of token IDs or hidden states")
        if input_ids is not None:
            if self.embed_tokens is None:
                raise ValueError("only stage zero can embed token IDs")
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        sequence_length = int(hidden_states.shape[1])
        cache_position = torch.arange(
            cache_position_start,
            cache_position_start + sequence_length,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = self._causal_mask(hidden_states, cache_position, state.sequence_length)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        routers: list[torch.Tensor] = []
        for layer in self.layers:
            outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=state.cache,
                output_attentions=False,
                output_router_logits=capture_router_logits,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = outputs[0]
            if capture_router_logits:
                routers.append(outputs[-1].detach())
        stage_boundary_hidden_states = hidden_states
        final_hidden = None
        logits = None
        token_ids = None
        all_token_ids = None
        if self.norm is not None:
            final_hidden = self.norm(hidden_states)
            hidden_states = final_hidden
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
            all_token_ids = torch.argmax(logits, dim=-1)
            token_ids = all_token_ids[:, -1]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        compute_ns = time.perf_counter_ns() - started
        state.sequence_length += sequence_length
        return StageExecutionResult(
            hidden_states=hidden_states,
            stage_boundary_hidden_states=stage_boundary_hidden_states,
            router_logits=tuple(routers),
            final_hidden_states=final_hidden,
            logits=logits,
            sampled_token_ids=token_ids,
            all_sampled_token_ids=all_token_ids,
            cache_sequence_length=state.sequence_length,
            compute_ns=compute_ns,
        )
