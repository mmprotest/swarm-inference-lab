"""Kimi K3-shaped native MXFP4 fixture and executable operator path."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.expert import reduce_partials
from swarm_inference.experiments.experiment_010.schemas import ReductionMode

KIMI_LATENT_DIMENSION = 3584
KIMI_EXPERT_INTERMEDIATE_DIMENSION = 3072
KIMI_ROUTED_EXPERTS = 16
KIMI_LOGICAL_MOE_LAYERS = 92
MXFP4_GROUP_SIZE = 32
MXFP4_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class MXFP4Tensor:
    """Official compressed-tensors mxfp4-pack-quantized row layout."""

    packed: np.ndarray
    scales: np.ndarray
    output_dimension: int
    input_dimension: int

    def __post_init__(self) -> None:
        packed = np.asarray(self.packed)
        scales = np.asarray(self.scales)
        if self.input_dimension % MXFP4_GROUP_SIZE:
            raise ValueError("MXFP4 input dimension must align to 32-value scale groups")
        if packed.dtype != np.uint8 or packed.shape != (
            self.output_dimension,
            self.input_dimension // 2,
        ):
            raise ValueError("MXFP4 packed tensor must be uint8 [O, I/2]")
        if scales.dtype != np.uint8 or scales.shape != (
            self.output_dimension,
            self.input_dimension // MXFP4_GROUP_SIZE,
        ):
            raise ValueError("MXFP4 scales must be UE8M0 uint8 [O, I/32]")
        if np.any((scales == 0) | (scales == 255)):
            raise ValueError("fixture uses only finite non-denormal UE8M0 scales")

    @property
    def byte_size(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mxfp4-e2m1-low-even-ue8m0-g32")
        digest.update(np.ascontiguousarray(self.packed).tobytes())
        digest.update(np.ascontiguousarray(self.scales).tobytes())
        return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class KimiExpert:
    gate: MXFP4Tensor
    up: MXFP4Tensor
    down: MXFP4Tensor
    expert_id: int

    def __post_init__(self) -> None:
        if self.gate.output_dimension != self.up.output_dimension:
            raise ValueError("Kimi gate and up output widths differ")
        if self.gate.input_dimension != self.up.input_dimension:
            raise ValueError("Kimi gate and up latent widths differ")
        if self.down.input_dimension != self.gate.output_dimension:
            raise ValueError("Kimi down input must match expert intermediate width")
        if self.down.output_dimension != self.gate.input_dimension:
            raise ValueError("Kimi down output must match latent width")

    @property
    def latent_dimension(self) -> int:
        return self.gate.input_dimension

    @property
    def intermediate_dimension(self) -> int:
        return self.gate.output_dimension

    @property
    def byte_size(self) -> int:
        return self.gate.byte_size + self.up.byte_size + self.down.byte_size


def deterministic_mxfp4_tensor(
    *,
    output_dimension: int,
    input_dimension: int,
    seed: int,
    sparse: bool = False,
) -> MXFP4Tensor:
    if input_dimension % MXFP4_GROUP_SIZE:
        raise ValueError("generated MXFP4 width must align to 32")
    generator = np.random.default_rng(seed)
    shape = (output_dimension, input_dimension // 2)
    if sparse:
        packed = np.zeros(shape, dtype=np.uint8)
        # Retain deterministic non-zero work in one group so the path is not a
        # zero-result shortcut while keeping CI fixtures bounded.
        columns = min(16, shape[1])
        packed[:, :columns] = generator.integers(
            0, 256, (output_dimension, columns), dtype=np.uint8
        )
    else:
        packed = generator.integers(0, 256, shape, dtype=np.uint8)
    # Values around exponent 119 keep K3-sized dot products numerically tame.
    scales = generator.integers(
        116,
        123,
        (output_dimension, input_dimension // MXFP4_GROUP_SIZE),
        dtype=np.uint8,
    )
    return MXFP4Tensor(
        packed=packed,
        scales=scales,
        output_dimension=output_dimension,
        input_dimension=input_dimension,
    )


def deterministic_kimi_expert(
    *,
    expert_id: int,
    latent_dimension: int = KIMI_LATENT_DIMENSION,
    intermediate_dimension: int = KIMI_EXPERT_INTERMEDIATE_DIMENSION,
    seed: int = 1010,
    sparse: bool = False,
) -> KimiExpert:
    if intermediate_dimension % MXFP4_GROUP_SIZE:
        raise ValueError("Kimi intermediate dimension must align to MXFP4 groups")
    return KimiExpert(
        gate=deterministic_mxfp4_tensor(
            output_dimension=intermediate_dimension,
            input_dimension=latent_dimension,
            seed=seed + expert_id * 3,
            sparse=sparse,
        ),
        up=deterministic_mxfp4_tensor(
            output_dimension=intermediate_dimension,
            input_dimension=latent_dimension,
            seed=seed + expert_id * 3 + 1,
            sparse=sparse,
        ),
        down=deterministic_mxfp4_tensor(
            output_dimension=latent_dimension,
            input_dimension=intermediate_dimension,
            seed=seed + expert_id * 3 + 2,
            sparse=sparse,
        ),
        expert_id=expert_id,
    )


def _ue8m0(scales: np.ndarray) -> np.ndarray:
    return np.exp2(np.asarray(scales, dtype=np.int16) - 127).astype(np.float32)


def _decode_group(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    values[:, 0::2] = MXFP4_VALUES[low]
    values[:, 1::2] = MXFP4_VALUES[high]
    return values * _ue8m0(scales)[:, None]


def mxfp4_matmul(
    activation: np.ndarray,
    tensor: MXFP4Tensor,
    *,
    input_start: int = 0,
    input_end: int | None = None,
) -> np.ndarray:
    """Tile-decode native bytes without a persistent dequantized tensor."""

    source = np.ascontiguousarray(activation, dtype=np.float32)
    end = tensor.input_dimension if input_end is None else input_end
    if input_start % MXFP4_GROUP_SIZE or end % MXFP4_GROUP_SIZE:
        raise ValueError("MXFP4 matmul slice must preserve 32-value scale groups")
    if input_start < 0 or end <= input_start or end > tensor.input_dimension:
        raise ValueError("MXFP4 matmul slice is invalid")
    if source.ndim != 2 or source.shape[1] != end - input_start:
        raise ValueError("MXFP4 activation does not match selected input slice")
    output = np.zeros((source.shape[0], tensor.output_dimension), dtype=np.float32)
    for start in range(input_start, end, MXFP4_GROUP_SIZE):
        local = start - input_start
        group_index = start // MXFP4_GROUP_SIZE
        packed_start = start // 2
        packed_end = packed_start + MXFP4_GROUP_SIZE // 2
        packed_group = tensor.packed[:, packed_start:packed_end]
        # Deterministic generated full-size fixtures may use structurally zero
        # groups. Skipping those groups preserves the native packed layout and
        # exact mathematics while keeping the 92-layer CI/operator replay
        # bounded; no dequantized representation is retained.
        if not np.any(packed_group):
            continue
        weights = _decode_group(packed_group, tensor.scales[:, group_index])
        output += source[:, local : local + MXFP4_GROUP_SIZE] @ weights.T
    return output


def situ_glu(
    gate: np.ndarray,
    up: np.ndarray,
    *,
    beta_one: float = 4.0,
    beta_two: float = 25.0,
) -> np.ndarray:
    """Kimi K3 SiTU-GLU, not the SiLU used by OLMoE."""

    gate_values = np.asarray(gate, dtype=np.float32)
    up_values = np.asarray(up, dtype=np.float32)
    sigmoid = np.empty_like(gate_values)
    positive = gate_values >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-gate_values[positive]))
    exp_negative = np.exp(gate_values[~positive])
    sigmoid[~positive] = exp_negative / (1.0 + exp_negative)
    return (
        beta_one
        * np.tanh(gate_values / beta_one)
        * sigmoid
        * beta_two
        * np.tanh(up_values / beta_two)
    )


def execute_kimi_expert(
    activation: np.ndarray,
    expert: KimiExpert,
    *,
    routing_weight: float,
    hidden_start: int = 0,
    hidden_end: int | None = None,
) -> np.ndarray:
    end = expert.intermediate_dimension if hidden_end is None else hidden_end
    if hidden_start % MXFP4_GROUP_SIZE or end % MXFP4_GROUP_SIZE:
        raise ValueError("Kimi microshard boundaries must align to 32")
    if hidden_start < 0 or end <= hidden_start or end > expert.intermediate_dimension:
        raise ValueError("Kimi microshard hidden range is invalid")
    # Gate/up row slicing does not split their input scale groups. Down input
    # slicing is group-aligned and consumes only matching hidden columns.
    gate_tensor = MXFP4Tensor(
        packed=expert.gate.packed[hidden_start:end],
        scales=expert.gate.scales[hidden_start:end],
        output_dimension=end - hidden_start,
        input_dimension=expert.latent_dimension,
    )
    up_tensor = MXFP4Tensor(
        packed=expert.up.packed[hidden_start:end],
        scales=expert.up.scales[hidden_start:end],
        output_dimension=end - hidden_start,
        input_dimension=expert.latent_dimension,
    )
    gate = mxfp4_matmul(activation, gate_tensor)
    up = mxfp4_matmul(activation, up_tensor)
    hidden = situ_glu(gate, up)
    down = mxfp4_matmul(
        hidden,
        expert.down,
        input_start=hidden_start,
        input_end=end,
    )
    return np.float32(routing_weight) * down


def execute_kimi_topk(
    activation: np.ndarray,
    experts: list[KimiExpert],
    routing_weights: list[float],
    *,
    shard_ranges: list[tuple[int, int]] | None = None,
    reduction_mode: ReductionMode | str = ReductionMode.FIXED_ORDER_FP32,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(experts) != len(routing_weights):
        raise ValueError("Kimi experts and routing weights differ in length")
    if len(experts) != KIMI_ROUTED_EXPERTS:
        raise ValueError("Kimi K3 operator replay requires top-16 routed experts")
    width = experts[0].intermediate_dimension
    ranges = shard_ranges or [(0, width)]
    if ranges[0][0] != 0 or ranges[-1][1] != width:
        raise ValueError("Kimi shard ranges do not cover the intermediate width")
    for previous, following in pairwise(ranges):
        if previous[1] != following[0]:
            raise ValueError("Kimi shard ranges have a gap or overlap")
    started = time.perf_counter_ns()
    partials = []
    for shard_index, (start, end) in enumerate(ranges):
        partial = np.zeros((activation.shape[0], experts[0].latent_dimension), dtype=np.float32)
        for expert, weight in zip(experts, routing_weights, strict=True):
            partial += execute_kimi_expert(
                activation,
                expert,
                routing_weight=weight,
                hidden_start=start,
                hidden_end=end,
            )
        partials.append((f"shard-{shard_index:04d}", partial))
    output = reduce_partials(partials, mode=reduction_mode)
    return output, {
        "elapsed_ns": time.perf_counter_ns() - started,
        "expert_count": len(experts),
        "shard_count": len(ranges),
        "shard_ranges": [list(item) for item in ranges],
        "native_weight_bytes": sum(item.byte_size for item in experts),
        "activation_bytes": int(np.asarray(activation).nbytes),
        "result_bytes": int(output.nbytes),
        "native_format": "mxfp4_e2m1_ue8m0_g32",
        "persistent_dequantized_bytes": 0,
        "category": "SYNTHETIC_FIXTURE",
    }


def kimi_fixture_inventory(experts: list[KimiExpert]) -> dict[str, Any]:
    return {
        "category": "SYNTHETIC_FIXTURE",
        "description": "deterministically generated valid MXFP4 tensors; not Kimi K3 weights",
        "latent_dimension": experts[0].latent_dimension if experts else KIMI_LATENT_DIMENSION,
        "expert_intermediate_dimension": (
            experts[0].intermediate_dimension if experts else KIMI_EXPERT_INTERMEDIATE_DIMENSION
        ),
        "projection_count": 3,
        "top_k": KIMI_ROUTED_EXPERTS,
        "logical_moe_layers": KIMI_LOGICAL_MOE_LAYERS,
        "packing": "e2m1_two_nibbles_low_even",
        "scale_format": "ue8m0",
        "scale_group_size": MXFP4_GROUP_SIZE,
        "experts": [
            {
                "expert_id": expert.expert_id,
                "native_bytes": expert.byte_size,
                "gate_hash": expert.gate.content_hash,
                "up_hash": expert.up.content_hash,
                "down_hash": expert.down.content_hash,
            }
            for expert in experts
        ],
        "native_bytes": sum(expert.byte_size for expert in experts),
        "reencoded": False,
        "persistent_dequantized_bytes": 0,
    }


def run_full_kimi_k3_fixture(*, seed: int = 1010) -> dict[str, Any]:
    """Execute the exact official geometry and a 92-layer routed replay.

    These are deterministically generated sparse-but-valid packed tensors, not
    checkpoint weights.  Zero-group skipping is reported explicitly.
    """

    experts = [
        deterministic_kimi_expert(expert_id=index, seed=seed, sparse=True)
        for index in range(KIMI_ROUTED_EXPERTS)
    ]
    generator = np.random.default_rng(seed)
    activation = generator.normal(0, 0.1, (1, KIMI_LATENT_DIMENSION)).astype(np.float32)
    routing_weights = [1 / KIMI_ROUTED_EXPERTS] * KIMI_ROUTED_EXPERTS
    whole, whole_metrics = execute_kimi_topk(activation, experts, routing_weights)
    equal_ranges = [(index, index + 768) for index in range(0, 3072, 768)]
    asymmetric_ranges = [(0, 384), (384, 1024), (1024, 3072)]
    equal, equal_metrics = execute_kimi_topk(
        activation, experts, routing_weights, shard_ranges=equal_ranges
    )
    asymmetric, asymmetric_metrics = execute_kimi_topk(
        activation, experts, routing_weights, shard_ranges=asymmetric_ranges
    )
    replay_started = time.perf_counter_ns()
    replay_state = activation
    for _layer_id in range(KIMI_LOGICAL_MOE_LAYERS):
        replay_state, _ = execute_kimi_topk(replay_state, experts, routing_weights)
    replay_ns = time.perf_counter_ns() - replay_started

    def comparison(candidate: np.ndarray) -> dict[str, Any]:
        difference = candidate.astype(np.float64) - whole.astype(np.float64)
        denominator = max(float(np.linalg.norm(whole.astype(np.float64).ravel())), 1e-30)
        return {
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "relative_l2_error": float(np.linalg.norm(difference.ravel()) / denominator),
            "exact": bool(np.array_equal(candidate, whole)),
        }

    common = {
        "category": "SYNTHETIC_FIXTURE",
        "latent_dimension": KIMI_LATENT_DIMENSION,
        "expert_intermediate_dimension": KIMI_EXPERT_INTERMEDIATE_DIMENSION,
        "top_k": KIMI_ROUTED_EXPERTS,
        "native_format": "mxfp4_e2m1_ue8m0_g32",
        "generated_weights": True,
        "checkpoint_weights": False,
        "zero_group_skipping": True,
    }
    inventory = kimi_fixture_inventory(experts)
    inventory["exact_official_geometry"] = True
    inventory["zero_group_skipping"] = True
    rows = [
        {
            **common,
            "component": "whole_expert_top16_exact_geometry",
            "elapsed_ms": whole_metrics["elapsed_ns"] / 1e6,
            **whole_metrics,
        },
        {
            **common,
            "component": "equal_microshards_top16_exact_geometry",
            "elapsed_ms": equal_metrics["elapsed_ns"] / 1e6,
            **equal_metrics,
            **comparison(equal),
        },
        {
            **common,
            "component": "asymmetric_microshards_top16_exact_geometry",
            "elapsed_ms": asymmetric_metrics["elapsed_ns"] / 1e6,
            **asymmetric_metrics,
            **comparison(asymmetric),
        },
        {
            **common,
            "component": "routed_92_layer_replay_exact_geometry",
            "elapsed_ms": replay_ns / 1e6,
            "elapsed_ns": replay_ns,
            "logical_layers": KIMI_LOGICAL_MOE_LAYERS,
            "result_bytes": int(replay_state.nbytes),
            "result_sha256": hashlib.sha256(replay_state.tobytes()).hexdigest(),
        },
    ]
    return {
        "inventory": inventory,
        "rows": rows,
        "whole_equal_relative_l2_error": comparison(equal)["relative_l2_error"],
        "whole_asymmetric_relative_l2_error": comparison(asymmetric)["relative_l2_error"],
        "exact_geometry": True,
        "top16": True,
        "logical_layers_executed": KIMI_LOGICAL_MOE_LAYERS,
        "category": "SYNTHETIC_FIXTURE",
    }
