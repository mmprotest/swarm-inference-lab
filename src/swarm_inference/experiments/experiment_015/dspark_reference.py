"""CPU BF16 reference for the public Kimi K3 DSpark checkpoint.

This is a correctness/reference path. It is deliberately not capacity evidence.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_015.contracts import Experiment015Error


def _torch() -> Any:
    import torch

    return torch


def _rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    variance = value.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (value.to(torch.float32) * torch.rsqrt(variance + epsilon)).to(value.dtype) * weight


def _yarn_inverse_frequencies(config: dict[str, Any], device: Any) -> tuple[Any, float]:
    torch = _torch()
    rope = config["rope_parameters"]
    dimension = int(config["qk_rope_head_dim"])
    base = float(rope["rope_theta"])
    factor = float(rope["factor"])
    original = int(rope["original_max_position_embeddings"])

    def correction(rotations: float) -> float:
        return dimension * math.log(original / (rotations * 2 * math.pi)) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction(float(rope.get("beta_fast", 32)))), 0)
    high = min(math.ceil(correction(float(rope.get("beta_slow", 1)))), dimension - 1)
    frequencies = base ** (
        torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension
    )
    extrapolated = 1.0 / frequencies
    interpolated = 1.0 / (factor * frequencies)
    ramp = torch.clamp(
        (torch.arange(dimension // 2, dtype=torch.float32, device=device) - low)
        / max(high - low, 0.001),
        0,
        1,
    )
    inverse = interpolated * ramp + extrapolated * (1.0 - ramp)
    mscale = float(rope.get("mscale", 1.0))
    mscale_all = float(rope.get("mscale_all_dim", 1.0))
    attention_factor = (0.1 * mscale * math.log(factor) + 1.0) / (
        0.1 * mscale_all * math.log(factor) + 1.0
    )
    return inverse, attention_factor


def _apply_rope(values: Any, positions: Any, inverse: Any, scale: float) -> Any:
    torch = _torch()
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse.unsqueeze(0)
    cosine = torch.cos(angles) * scale
    sine = torch.sin(angles) * scale
    pairs = values.to(torch.float32).reshape(*values.shape[:-1], -1, 2)
    even, odd = pairs.unbind(dim=-1)
    while cosine.ndim < even.ndim:
        cosine = cosine.unsqueeze(-2)
        sine = sine.unsqueeze(-2)
    rotated = torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), -1)
    return rotated.flatten(-2).to(values.dtype)


@dataclass(slots=True)
class _LayerWeights:
    input_norm: Any
    post_norm: Any
    q_a: Any
    q_a_norm: Any
    q_b: Any
    kv_a: Any
    kv_a_norm: Any
    kv_b: Any
    output: Any
    gate: Any
    up: Any
    down: Any


class K3DSparkReference:
    """Faithful single-request BF16 DSpark forward for local correctness tests."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        torch = _torch()
        from safetensors import safe_open

        self.checkpoint = checkpoint.expanduser().resolve()
        self.config = json.loads((self.checkpoint / "config.json").read_text())
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self._file = safe_open(
            str(self.checkpoint / "model.safetensors"), framework="pt", device=str(device)
        )
        self.layers: list[_LayerWeights] = []
        for layer in range(int(self.config["num_hidden_layers"])):
            prefix = f"layers.{layer}"
            self.layers.append(
                _LayerWeights(
                    input_norm=self._weight(f"{prefix}.input_layernorm.weight"),
                    post_norm=self._weight(f"{prefix}.post_attention_layernorm.weight"),
                    q_a=self._weight(f"{prefix}.self_attn.q_a_proj.weight"),
                    q_a_norm=self._weight(f"{prefix}.self_attn.q_a_layernorm.weight"),
                    q_b=self._weight(f"{prefix}.self_attn.q_b_proj.weight"),
                    kv_a=self._weight(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    kv_a_norm=self._weight(f"{prefix}.self_attn.kv_a_layernorm.weight"),
                    kv_b=self._weight(f"{prefix}.self_attn.kv_b_proj.weight"),
                    output=self._weight(f"{prefix}.self_attn.o_proj.weight"),
                    gate=self._weight(f"{prefix}.mlp.gate_proj.weight"),
                    up=self._weight(f"{prefix}.mlp.up_proj.weight"),
                    down=self._weight(f"{prefix}.mlp.down_proj.weight"),
                )
            )
        self.context_projection = self._weight("context_proj.weight")
        self.context_norm = self._weight("context_norm.weight")
        self.final_norm = self._weight("final_norm.weight")
        self.markov_w1 = self._weight("markov_head.markov_w1.weight")
        self.markov_w2 = self._weight("markov_head.markov_w2.weight")
        self.inverse_frequency, self.rope_scale = _yarn_inverse_frequencies(
            self.config, self.device
        )
        rope = self.config["rope_parameters"]
        yarn_mscale = (
            0.1
            * float(rope.get("mscale_all_dim", 1.0))
            * math.log(float(rope["factor"]))
            + 1.0
        )
        self.attention_scale = (
            int(self.config["qk_nope_head_dim"])
            + int(self.config["qk_rope_head_dim"])
        ) ** -0.5 * yarn_mscale**2
        self.context_cache: list[tuple[Any, Any]] = []

    def _weight(self, name: str) -> Any:
        return self._file.get_tensor(name).to(self.device)

    def combine_target_hidden(self, target_hidden: np.ndarray | Any) -> Any:
        torch = _torch()
        values = torch.as_tensor(target_hidden, dtype=self.dtype, device=self.device)
        if values.shape[-1] != int(self.config["target_hidden_size"]) * int(
            self.config["num_target_layers"]
        ):
            raise Experiment015Error("DSpark target hidden concatenation has wrong width")
        combined = torch.nn.functional.linear(values, self.context_projection)
        return _rms_norm(combined, self.context_norm, float(self.config["rms_norm_eps"]))

    def precompute_context(self, states: Any, positions: Any) -> None:
        torch = _torch()
        positions = torch.as_tensor(positions, dtype=torch.int64, device=self.device)
        self.context_cache = []
        for weights in self.layers:
            compressed = torch.nn.functional.linear(states, weights.kv_a)
            latent, rope = compressed.split(
                [int(self.config["kv_lora_rank"]), int(self.config["qk_rope_head_dim"])],
                dim=-1,
            )
            latent = _rms_norm(
                latent, weights.kv_a_norm, float(self.config["rms_norm_eps"])
            )
            rope = _apply_rope(rope, positions, self.inverse_frequency, self.rope_scale)
            self.context_cache.append((latent, rope))

    def _attention(self, hidden: Any, positions: Any, weights: _LayerWeights, index: int) -> Any:
        torch = _torch()
        q_low = torch.nn.functional.linear(hidden, weights.q_a)
        q_low = _rms_norm(q_low, weights.q_a_norm, float(self.config["rms_norm_eps"]))
        heads = int(self.config["num_attention_heads"])
        nope = int(self.config["qk_nope_head_dim"])
        rope_width = int(self.config["qk_rope_head_dim"])
        value_width = int(self.config["v_head_dim"])
        latent_width = int(self.config["kv_lora_rank"])
        query = torch.nn.functional.linear(q_low, weights.q_b).view(
            hidden.shape[0], heads, nope + rope_width
        )
        query_nope, query_rope = query.split([nope, rope_width], dim=-1)
        query_rope = _apply_rope(
            query_rope, positions, self.inverse_frequency, self.rope_scale
        )
        context_latent, context_rope = self.context_cache[index]
        query_compressed = torch.nn.functional.linear(hidden, weights.kv_a)
        query_latent, query_key_rope = query_compressed.split(
            [int(self.config["kv_lora_rank"]), rope_width], dim=-1
        )
        query_latent = _rms_norm(
            query_latent, weights.kv_a_norm, float(self.config["rms_norm_eps"])
        )
        query_key_rope = _apply_rope(
            query_key_rope, positions, self.inverse_frequency, self.rope_scale
        )
        latent = torch.cat((context_latent, query_latent), dim=0)
        key_rope = torch.cat((context_rope, query_key_rope), dim=0)
        expanded = torch.nn.functional.linear(latent, weights.kv_b).view(
            latent.shape[0], heads, nope + value_width
        )
        key_nope, values = expanded.split([nope, value_width], dim=-1)
        scores = torch.einsum("qhd,khd->qhk", query_nope, key_nope)
        scores += torch.einsum("qhd,kd->qhk", query_rope, key_rope)
        scores *= self.attention_scale
        probabilities = torch.softmax(scores.to(torch.float32), dim=-1).to(hidden.dtype)
        attended = torch.einsum("qhk,khd->qhd", probabilities, values)
        del latent_width
        return torch.nn.functional.linear(attended.flatten(1), weights.output)

    def forward_block(
        self,
        target_context_states: np.ndarray | Any,
        target_context_positions: list[int],
        query_embeddings: np.ndarray | Any,
        query_positions: list[int],
        target_lm_head: np.ndarray | Any,
    ) -> tuple[Any, dict[str, Any]]:
        torch = _torch()
        if not query_positions:
            raise ValueError("DSpark query block cannot be empty")
        started = time.perf_counter()
        context = self.combine_target_hidden(target_context_states)
        self.precompute_context(context, target_context_positions)
        hidden = torch.as_tensor(query_embeddings, dtype=self.dtype, device=self.device)
        if hidden.shape != (len(query_positions), int(self.config["hidden_size"])):
            raise ValueError("DSpark query embeddings do not match query positions")
        positions = torch.as_tensor(query_positions, dtype=torch.int64, device=self.device)
        residual = hidden
        for index, weights in enumerate(self.layers):
            normalized = _rms_norm(
                hidden, weights.input_norm, float(self.config["rms_norm_eps"])
            )
            hidden = residual + self._attention(normalized, positions, weights, index)
            residual = hidden
            normalized = _rms_norm(
                hidden, weights.post_norm, float(self.config["rms_norm_eps"])
            )
            gate = torch.nn.functional.linear(normalized, weights.gate)
            up = torch.nn.functional.linear(normalized, weights.up)
            hidden = residual + torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, weights.down
            )
            residual = hidden
        hidden = _rms_norm(hidden, self.final_norm, float(self.config["rms_norm_eps"]))
        head = torch.as_tensor(target_lm_head, dtype=self.dtype, device=self.device)
        logits = torch.nn.functional.linear(hidden, head)
        return logits, {
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "query_rows": len(query_positions),
            "context_rows": int(context.shape[0]),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "classification": "MEASURED reference correctness; not capacity evidence",
        }

    def apply_markov_bias(self, base_logits: Any, anchor_token_id: int) -> Any:
        torch = _torch()
        previous = int(anchor_token_id)
        rows = []
        for row in base_logits:
            transition = torch.nn.functional.linear(
                self.markov_w1[previous].unsqueeze(0), self.markov_w2
            ).squeeze(0)
            corrected = row + transition
            rows.append(corrected)
            previous = int(corrected.argmax())
        return torch.stack(rows)
