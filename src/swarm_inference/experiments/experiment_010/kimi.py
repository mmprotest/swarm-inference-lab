"""Kimi K3-shaped native MXFP4 fixture and executable operator path."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
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


class NativeMXFP4Runtime:
    """Narrow ctypes binding to Colibri's compiled ``matmul_mxfp4`` kernel."""

    ABI = "colibri-native-mxfp4-fixture-v1"

    def __init__(self, library_path: Path) -> None:
        path = Path(library_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self._library = ctypes.CDLL(str(path))
        self._library.coli_kimi_mxfp4_runtime_abi.argtypes = []
        self._library.coli_kimi_mxfp4_runtime_abi.restype = ctypes.c_char_p
        abi = self._library.coli_kimi_mxfp4_runtime_abi().decode("ascii")
        if abi != self.ABI:
            raise RuntimeError(f"unexpected Colibri MXFP4 runtime ABI {abi!r}")
        pointer = ctypes.c_void_p
        self._library.coli_kimi_mxfp4_matmul.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_kimi_mxfp4_matmul.restype = ctypes.c_int
        self._library.coli_kimi_mxfp4_matmul_input_slice.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.coli_kimi_mxfp4_matmul_input_slice.restype = ctypes.c_int
        self._library.coli_kimi_situ_glu.argtypes = [
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._library.coli_kimi_situ_glu.restype = ctypes.c_int
        self._library.coli_kimi_scale_add.argtypes = [
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_float,
        ]
        self._library.coli_kimi_scale_add.restype = ctypes.c_int

    @staticmethod
    def _pointer(array: np.ndarray) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(array.ctypes.data))

    def matmul(self, activation: np.ndarray, tensor: MXFP4Tensor) -> np.ndarray:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        packed = np.ascontiguousarray(tensor.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(tensor.scales, dtype=np.uint8)
        if source.ndim != 2 or source.shape[1] != tensor.input_dimension:
            raise ValueError("native MXFP4 activation shape mismatch")
        output = np.empty(
            (source.shape[0], tensor.output_dimension), dtype=np.float32
        )
        status = self._library.coli_kimi_mxfp4_matmul(
            self._pointer(output),
            self._pointer(source),
            self._pointer(packed),
            self._pointer(scales),
            source.shape[0],
            tensor.input_dimension,
            tensor.output_dimension,
        )
        if status:
            raise RuntimeError(f"Colibri native MXFP4 matmul failed: {status}")
        return output

    def matmul_input_slice(
        self,
        activation: np.ndarray,
        tensor: MXFP4Tensor,
        input_start: int,
        input_end: int,
    ) -> np.ndarray:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        packed = np.ascontiguousarray(tensor.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(tensor.scales, dtype=np.uint8)
        if source.ndim != 2 or source.shape[1] != input_end - input_start:
            raise ValueError("native MXFP4 sliced activation shape mismatch")
        output = np.empty(
            (source.shape[0], tensor.output_dimension), dtype=np.float32
        )
        status = self._library.coli_kimi_mxfp4_matmul_input_slice(
            self._pointer(output),
            self._pointer(source),
            self._pointer(packed),
            self._pointer(scales),
            source.shape[0],
            tensor.input_dimension,
            tensor.output_dimension,
            input_start,
            input_end,
        )
        if status:
            raise RuntimeError(f"Colibri native MXFP4 sliced matmul failed: {status}")
        return output

    def situ_glu(self, gate: np.ndarray, up: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(gate, dtype=np.float32)
        up_values = np.ascontiguousarray(up, dtype=np.float32)
        if result.shape != up_values.shape:
            raise ValueError("native SiTU gate/up shape mismatch")
        status = self._library.coli_kimi_situ_glu(
            self._pointer(result),
            self._pointer(up_values),
            result.size,
            ctypes.c_float(4.0),
            ctypes.c_float(25.0),
        )
        if status:
            raise RuntimeError(f"Colibri native SiTU failed: {status}")
        return result

    def scale_add(
        self, destination: np.ndarray, source: np.ndarray, scale: float
    ) -> None:
        if (
            destination.dtype != np.float32
            or source.dtype != np.float32
            or not destination.flags.c_contiguous
            or not source.flags.c_contiguous
            or destination.shape != source.shape
        ):
            raise ValueError("native scale-add requires matching contiguous float32 arrays")
        status = self._library.coli_kimi_scale_add(
            self._pointer(destination),
            self._pointer(source),
            destination.size,
            ctypes.c_float(scale),
        )
        if status:
            raise RuntimeError(f"Colibri native scale-add failed: {status}")


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
        # Every 32-value group starts with a +0.5 E2M1 value.  The remaining
        # 31 values retain their seeded dense distribution.  This makes the
        # no-all-zero-group contract structural instead of probabilistic.
        packed[:, 0:: MXFP4_GROUP_SIZE // 2] &= np.uint8(0xF0)
        packed[:, 0:: MXFP4_GROUP_SIZE // 2] |= np.uint8(0x01)
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


def mxfp4_zero_group_count(tensor: MXFP4Tensor) -> int:
    """Count groups whose E2M1 nibbles are all either +0 or -0."""

    groups = tensor.packed.reshape(
        tensor.output_dimension,
        tensor.input_dimension // MXFP4_GROUP_SIZE,
        MXFP4_GROUP_SIZE // 2,
    )
    has_nonzero_low = np.any((groups & np.uint8(0x07)) != 0, axis=2)
    has_nonzero_high = np.any((groups & np.uint8(0x70)) != 0, axis=2)
    return int(np.count_nonzero(~(has_nonzero_low | has_nonzero_high)))


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
    native_runtime: NativeMXFP4Runtime | None = None,
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
    if native_runtime is None:
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
    gate = native_runtime.matmul(activation, gate_tensor)
    up = native_runtime.matmul(activation, up_tensor)
    hidden = native_runtime.situ_glu(gate, up)
    if hidden_start == 0 and end == expert.intermediate_dimension:
        down = native_runtime.matmul(hidden, expert.down)
    else:
        down = native_runtime.matmul_input_slice(
            hidden,
            expert.down,
            hidden_start,
            end,
        )
    weighted = np.zeros_like(down)
    native_runtime.scale_add(weighted, down, routing_weight)
    return weighted


def execute_kimi_topk(
    activation: np.ndarray,
    experts: list[KimiExpert],
    routing_weights: list[float],
    *,
    shard_ranges: list[tuple[int, int]] | None = None,
    reduction_mode: ReductionMode | str = ReductionMode.FIXED_ORDER_FP32,
    native_runtime: NativeMXFP4Runtime | None = None,
    coalesced_transport: bool = False,
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
    transport_ns = 0
    transport_bytes = 0
    compute_ns = 0
    partials = []
    coalesced_activation: np.ndarray | None = None
    if coalesced_transport:
        transport_started = time.perf_counter_ns()
        coalesced_activation = np.array(activation, dtype=np.float32, copy=True, order="C")
        transport_ns += time.perf_counter_ns() - transport_started
        transport_bytes += int(coalesced_activation.nbytes)
    for shard_index, (start, end) in enumerate(ranges):
        if coalesced_activation is None:
            transport_started = time.perf_counter_ns()
            shard_activation = np.array(activation, dtype=np.float32, copy=True, order="C")
            transport_ns += time.perf_counter_ns() - transport_started
            transport_bytes += int(shard_activation.nbytes)
        else:
            shard_activation = coalesced_activation
        partial = np.zeros((activation.shape[0], experts[0].latent_dimension), dtype=np.float32)
        compute_started = time.perf_counter_ns()
        for expert, weight in zip(experts, routing_weights, strict=True):
            contribution = execute_kimi_expert(
                shard_activation,
                expert,
                routing_weight=weight,
                hidden_start=start,
                hidden_end=end,
                native_runtime=native_runtime,
            )
            if native_runtime is None:
                partial += contribution
            else:
                native_runtime.scale_add(partial, contribution, 1.0)
        compute_ns += time.perf_counter_ns() - compute_started
        transport_started = time.perf_counter_ns()
        returned_partial = np.array(partial, dtype=np.float32, copy=True, order="C")
        transport_ns += time.perf_counter_ns() - transport_started
        transport_bytes += int(returned_partial.nbytes)
        partials.append((f"shard-{shard_index:04d}", returned_partial))
    reduction_started = time.perf_counter_ns()
    if native_runtime is None:
        output = reduce_partials(partials, mode=reduction_mode)
    else:
        output = np.zeros_like(partials[0][1])
        for _shard_id, partial in partials:
            native_runtime.scale_add(output, partial, 1.0)
    reduction_ns = time.perf_counter_ns() - reduction_started
    latent = experts[0].latent_dimension
    intermediate = experts[0].intermediate_dimension
    expert_count = len(experts)
    groups_processed = 3 * latent * intermediate // MXFP4_GROUP_SIZE * expert_count
    multiply_accumulate_count = 3 * latent * intermediate * expert_count
    return output, {
        "elapsed_ns": time.perf_counter_ns() - started,
        "compute_ns": compute_ns,
        "transport_ns": transport_ns,
        "reduction_ns": reduction_ns,
        "expert_count": len(experts),
        "shard_count": len(ranges),
        "shard_ranges": [list(item) for item in ranges],
        "coalesced_transport": coalesced_transport,
        "transport_message_count": 1 if coalesced_transport else len(ranges),
        "transport_bytes": transport_bytes,
        "transport_kind": "measured_process_memory_copy_fixture",
        "native_weight_bytes": sum(item.byte_size for item in experts),
        "native_weight_bytes_read": sum(item.byte_size for item in experts),
        "activation_bytes": int(np.asarray(activation).nbytes),
        "result_bytes": int(output.nbytes),
        "multiply_accumulate_count": multiply_accumulate_count,
        "real_operations_performed": 2 * multiply_accumulate_count,
        "groups_processed": groups_processed,
        "groups_with_arithmetic": groups_processed,
        "zero_quantization_groups": 0,
        "native_format": "mxfp4_e2m1_ue8m0_g32",
        "arithmetic_backend": (
            NativeMXFP4Runtime.ABI if native_runtime is not None else "numpy_diagnostic"
        ),
        "native_runtime_sha256": (
            native_runtime.sha256 if native_runtime is not None else None
        ),
        "persistent_dequantized_bytes": 0,
        "category": "SYNTHETIC_FIXTURE",
    }


def kimi_fixture_inventory(experts: list[KimiExpert]) -> dict[str, Any]:
    tensor_zero_groups = [
        {
            "expert_id": expert.expert_id,
            "gate": mxfp4_zero_group_count(expert.gate),
            "up": mxfp4_zero_group_count(expert.up),
            "down": mxfp4_zero_group_count(expert.down),
        }
        for expert in experts
    ]
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
        "dense_fixture": all(
            not (row[projection])
            for row in tensor_zero_groups
            for projection in ("gate", "up", "down")
        ),
        "zero_quantization_group_count": sum(
            int(row[projection])
            for row in tensor_zero_groups
            for projection in ("gate", "up", "down")
        ),
        "zero_groups_by_expert": tensor_zero_groups,
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


def run_full_kimi_k3_fixture(
    *, native_library: Path, seed: int = 1010
) -> dict[str, Any]:
    """Execute dense official geometry through Colibri's compiled MXFP4 kernel.

    The tensors are deterministic synthetic E2M1/UE8M0 bytes, never checkpoint
    weights.  Every quantization group contains arithmetic and the official
    path fails closed when the compiled Colibri adapter is unavailable.
    """

    native_runtime = NativeMXFP4Runtime(native_library)
    experts = [
        deterministic_kimi_expert(expert_id=index, seed=seed, sparse=False)
        for index in range(KIMI_ROUTED_EXPERTS)
    ]
    generator = np.random.default_rng(seed)
    activation = generator.normal(0, 0.1, (1, KIMI_LATENT_DIMENSION)).astype(np.float32)
    routing_weights = [1 / KIMI_ROUTED_EXPERTS] * KIMI_ROUTED_EXPERTS

    transport_started = time.perf_counter_ns()
    single_activation = np.array(activation, dtype=np.float32, copy=True, order="C")
    single_transport_ns = time.perf_counter_ns() - transport_started
    single_compute_started = time.perf_counter_ns()
    single = execute_kimi_expert(
        single_activation,
        experts[0],
        routing_weight=1.0,
        native_runtime=native_runtime,
    )
    single_compute_ns = time.perf_counter_ns() - single_compute_started
    transport_started = time.perf_counter_ns()
    single = np.array(single, dtype=np.float32, copy=True, order="C")
    single_transport_ns += time.perf_counter_ns() - transport_started

    whole, whole_metrics = execute_kimi_topk(
        activation,
        experts,
        routing_weights,
        native_runtime=native_runtime,
    )
    equal_ranges = [(index, index + 768) for index in range(0, 3072, 768)]
    asymmetric_ranges = [(0, 384), (384, 1024), (1024, 3072)]
    equal, equal_metrics = execute_kimi_topk(
        activation,
        experts,
        routing_weights,
        shard_ranges=equal_ranges,
        native_runtime=native_runtime,
    )
    asymmetric, asymmetric_metrics = execute_kimi_topk(
        activation,
        experts,
        routing_weights,
        shard_ranges=asymmetric_ranges,
        native_runtime=native_runtime,
    )
    coalesced, coalesced_metrics = execute_kimi_topk(
        activation,
        experts,
        routing_weights,
        shard_ranges=equal_ranges,
        native_runtime=native_runtime,
        coalesced_transport=True,
    )

    replay_totals = {
        "compute_ns": 0,
        "transport_ns": 0,
        "reduction_ns": 0,
        "transport_bytes": 0,
        "transport_message_count": 0,
        "native_weight_bytes_read": 0,
        "multiply_accumulate_count": 0,
        "real_operations_performed": 0,
        "groups_processed": 0,
        "groups_with_arithmetic": 0,
    }
    replay_started = time.perf_counter_ns()
    replay_state = activation
    for _layer_id in range(KIMI_LOGICAL_MOE_LAYERS):
        replay_state, layer_metrics = execute_kimi_topk(
            replay_state,
            experts,
            routing_weights,
            native_runtime=native_runtime,
        )
        for name in replay_totals:
            replay_totals[name] += int(layer_metrics[name])
    replay_ns = time.perf_counter_ns() - replay_started
    if not np.isfinite(replay_state).all():
        raise RuntimeError("dense Kimi 92-layer replay produced a non-finite result")

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
        "dense_fixture": True,
        "zero_group_skipping": False,
        "native_arithmetic": True,
        "native_runtime_abi": NativeMXFP4Runtime.ABI,
        "native_runtime_path": str(native_runtime.path),
        "native_runtime_sha256": native_runtime.sha256,
    }
    inventory = kimi_fixture_inventory(experts)
    inventory["exact_official_geometry"] = True
    inventory["zero_group_skipping"] = False
    inventory["native_arithmetic"] = True
    inventory["native_runtime_abi"] = NativeMXFP4Runtime.ABI
    inventory["native_runtime_path"] = str(native_runtime.path)
    inventory["native_runtime_sha256"] = native_runtime.sha256
    if inventory["zero_quantization_group_count"]:
        raise RuntimeError("dense Kimi fixture contains an all-zero quantization group")
    one_expert_macs = 3 * KIMI_LATENT_DIMENSION * KIMI_EXPERT_INTERMEDIATE_DIMENSION
    single_metrics = {
        "elapsed_ns": single_compute_ns + single_transport_ns,
        "compute_ns": single_compute_ns,
        "transport_ns": single_transport_ns,
        "reduction_ns": 0,
        "transport_bytes": int(single_activation.nbytes + single.nbytes),
        "transport_message_count": 1,
        "native_weight_bytes": experts[0].byte_size,
        "native_weight_bytes_read": experts[0].byte_size,
        "multiply_accumulate_count": one_expert_macs,
        "real_operations_performed": 2 * one_expert_macs,
        "groups_processed": one_expert_macs // MXFP4_GROUP_SIZE,
        "groups_with_arithmetic": one_expert_macs // MXFP4_GROUP_SIZE,
        "zero_quantization_groups": 0,
        "result_bytes": int(single.nbytes),
        "result_sha256": hashlib.sha256(single.tobytes()).hexdigest(),
        "arithmetic_backend": NativeMXFP4Runtime.ABI,
        "persistent_dequantized_bytes": 0,
    }
    rows = [
        {
            **common,
            "component": "whole_expert_native_dense",
            "elapsed_ms": single_metrics["elapsed_ns"] / 1e6,
            **single_metrics,
        },
        {
            **common,
            "component": "top16_layer_native_dense",
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
            "component": "coalesced_microshards_top16_exact_geometry",
            "elapsed_ms": coalesced_metrics["elapsed_ns"] / 1e6,
            **coalesced_metrics,
            **comparison(coalesced),
        },
        {
            **common,
            "component": "routed_92_layer_replay_exact_geometry",
            "elapsed_ms": replay_ns / 1e6,
            "elapsed_ns": replay_ns,
            "logical_layers": KIMI_LOGICAL_MOE_LAYERS,
            "result_bytes": int(replay_state.nbytes),
            "result_sha256": hashlib.sha256(replay_state.tobytes()).hexdigest(),
            **replay_totals,
            "zero_quantization_groups": 0,
            "arithmetic_backend": NativeMXFP4Runtime.ABI,
            "persistent_dequantized_bytes": 0,
        },
    ]
    return {
        "inventory": inventory,
        "rows": rows,
        "whole_equal_relative_l2_error": comparison(equal)["relative_l2_error"],
        "whole_asymmetric_relative_l2_error": comparison(asymmetric)["relative_l2_error"],
        "whole_coalesced_relative_l2_error": comparison(coalesced)["relative_l2_error"],
        "exact_geometry": True,
        "dense_fixture": True,
        "zero_quantization_group_count": 0,
        "groups_with_arithmetic": replay_totals["groups_with_arithmetic"],
        "native_arithmetic": True,
        "native_runtime_sha256": native_runtime.sha256,
        "top16": True,
        "logical_layers_executed": KIMI_LOGICAL_MOE_LAYERS,
        "category": "SYNTHETIC_FIXTURE",
    }


def write_full_kimi_k3_fixture(
    *, native_library: Path, output_directory: Path, seed: int = 1010
) -> dict[str, Any]:
    result = run_full_kimi_k3_fixture(native_library=native_library, seed=seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kimi_fixture_inventory.json").write_text(
        json.dumps(result["inventory"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status = {key: value for key, value in result.items() if key not in {"inventory", "rows"}}
    (output / "dense_kimi_fixture_results.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in result["rows"] for key in row})
    with (output / "kimi_operator_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-dense-native", action="store_true")
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1010)
    arguments = parser.parse_args()
    if not arguments.full_dense_native:
        parser.error("select --full-dense-native")
    write_full_kimi_k3_fixture(
        native_library=arguments.native_library,
        output_directory=arguments.output,
        seed=arguments.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
