from __future__ import annotations

import numpy as np

from swarm_inference.backends.colibri.architecture import ColibriRoutingDescriptor
from swarm_inference.backends.colibri.sparse_moe import (
    execute_routed_layer,
    select_routed_experts,
    standard_swiglu_kernel,
)
from swarm_inference.execution.expert import (
    RoutedComputationStore,
    RoutedComputationWeights,
    routed_computation_content_hash,
    silu,
)
from swarm_inference.model.architecture import ExpertDescriptor, TensorGroupDescriptor


def _weights(
    expert_id: int,
    *,
    expert_type: str = "routed",
    seed: int = 1,
    fused: bool = False,
) -> RoutedComputationWeights:
    generator = np.random.default_rng(seed)
    hidden, intermediate = 3, 4
    if fused:
        tensors = {
            f"expert.{expert_id}.gate_up": generator.normal(size=(2 * intermediate, hidden)).astype(
                np.float32
            ),
            f"expert.{expert_id}.down": generator.normal(size=(hidden, intermediate)).astype(
                np.float32
            ),
        }
        roles = {
            f"expert.{expert_id}.gate_up": "routed_expert_gate+up_projection",
            f"expert.{expert_id}.down": "routed_expert_down_projection",
        }
    else:
        tensors = {
            f"expert.{expert_id}.gate": generator.normal(size=(intermediate, hidden)).astype(
                np.float32
            ),
            f"expert.{expert_id}.up": generator.normal(size=(intermediate, hidden)).astype(
                np.float32
            ),
            f"expert.{expert_id}.down": generator.normal(size=(hidden, intermediate)).astype(
                np.float32
            ),
        }
        roles = {
            f"expert.{expert_id}.gate": f"{expert_type}_expert_gate_projection",
            f"expert.{expert_id}.up": f"{expert_type}_expert_up_projection",
            f"expert.{expert_id}.down": f"{expert_type}_expert_down_projection",
        }
    group = TensorGroupDescriptor(
        group_id=f"layer-0:{expert_type}-{expert_id}",
        tensor_names=tuple(tensors),
        tensor_roles=tuple(roles.values()),
        tensor_shapes=tuple(value.shape for value in tensors.values()),
        parameter_count=sum(value.size for value in tensors.values()),
        memory_bytes=sum(value.nbytes for value in tensors.values()),
    )
    descriptor = ExpertDescriptor(
        layer_index=0,
        expert_index=expert_id,
        expert_type=expert_type,  # type: ignore[arg-type]
        tensor_groups=(group,),
        parameter_count=group.parameter_count,
        memory_bytes=group.memory_bytes,
        input_shape=(hidden,),
        output_shape=(hidden,),
        routing_metadata={"activation": "silu"},
    )
    return RoutedComputationWeights(
        descriptor=descriptor,
        tensors=tensors,
        tensor_roles=roles,
        content_hash=routed_computation_content_hash(tensors, roles),
    )


def _manual(activation: np.ndarray, weights: RoutedComputationWeights) -> np.ndarray:
    by_role = {weights.tensor_roles[name]: value for name, value in weights.tensors.items()}
    gate = next(value for role, value in by_role.items() if "gate" in role)
    up = next(value for role, value in by_role.items() if "up" in role)
    down = next(value for role, value in by_role.items() if "down" in role)
    return (silu(activation @ gate.T) * (activation @ up.T)) @ down.T


def _pack_symmetric_int4(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.int8)
    if source.ndim != 2 or source.shape[1] % 8 or np.any(source < -8) or np.any(source > 7):
        raise ValueError("INT4 test values must be [rows, columns divisible by 8] in [-8, 7]")
    shifted = (
        (source.astype(np.int32) + 8)
        .astype(np.uint32)
        .reshape(source.shape[0], source.shape[1] // 8, 8)
    )
    words = np.sum(
        shifted << (np.arange(8, dtype=np.uint32) * np.uint32(4))[None, None, :],
        axis=-1,
        dtype=np.uint32,
    )
    return words.view(np.int32)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def test_standard_adapter_described_swiglu_matches_reference() -> None:
    activation = np.array([[0.25, -0.5, 1.5], [1.0, 0.0, -1.0]], dtype=np.float32)
    weights = _weights(0, seed=11)

    actual = standard_swiglu_kernel(
        activation,
        weights,
        routing_metadata={"activation": "silu"},
    )

    np.testing.assert_allclose(actual, _manual(activation, weights), rtol=1e-6, atol=1e-6)


def test_fused_gate_up_layout_matches_separate_reference() -> None:
    activation = np.array([[0.5, -0.25, 1.0]], dtype=np.float32)
    fused = _weights(0, seed=17, fused=True)
    matrix = fused.tensors["expert.0.gate_up"]
    gate, up = np.split(matrix, 2, axis=0)
    down = fused.tensors["expert.0.down"]
    expected = (silu(activation @ gate.T) * (activation @ up.T)) @ down.T

    actual = standard_swiglu_kernel(
        activation,
        fused,
        routing_metadata={"activation": "silu", "fused_projection_order": "gate_up"},
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_packed_symmetric_int4_g32_matches_dequantized_reference() -> None:
    generator = np.random.default_rng(31)
    hidden = intermediate = 32
    activation = generator.normal(size=(2, hidden)).astype(np.float32)
    quantized = {
        name: generator.integers(-8, 8, size=(rows, columns), dtype=np.int8)
        for name, rows, columns in (
            ("gate", intermediate, hidden),
            ("up", intermediate, hidden),
            ("down", hidden, intermediate),
        )
    }
    scale_values = {
        name: generator.uniform(0.01, 0.2, size=(values.shape[0], 1)).astype(np.float32)
        for name, values in quantized.items()
    }
    scale_bits = {name: _bf16_bits(value) for name, value in scale_values.items()}
    rounded_scales = {
        name: (value.astype(np.uint32) << np.uint32(16)).view(np.float32)
        for name, value in scale_bits.items()
    }
    tensors: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}
    formats: dict[str, str] = {}
    for name in ("gate", "up", "down"):
        for suffix, tensor, role, tensor_format in (
            (
                "packed",
                _pack_symmetric_int4(quantized[name]),
                f"routed_expert_{name}_projection",
                "int4-g32",
            ),
            (
                "scale",
                scale_bits[name],
                f"routed_expert_{name}_scale",
                "bf16",
            ),
            (
                "shape",
                np.array(quantized[name].shape, dtype=np.int32),
                f"routed_expert_{name}_shape",
                "int32-metadata",
            ),
        ):
            key = f"expert.0.{name}.{suffix}"
            tensors[key] = tensor
            roles[key] = role
            formats[key] = tensor_format
    group = TensorGroupDescriptor(
        group_id="layer-0:routed-0",
        tensor_names=tuple(tensors),
        tensor_roles=tuple(roles.values()),
        tensor_shapes=tuple(value.shape for value in tensors.values()),
        parameter_count=3 * hidden * intermediate,
        memory_bytes=sum(value.nbytes for value in tensors.values()),
    )
    descriptor = ExpertDescriptor(
        layer_index=0,
        expert_index=0,
        expert_type="routed",
        tensor_groups=(group,),
        parameter_count=group.parameter_count,
        memory_bytes=group.memory_bytes,
        input_shape=(hidden,),
        output_shape=(hidden,),
        routing_metadata={"activation": "silu", "weight_group_size": 32},
    )
    weights = RoutedComputationWeights(
        descriptor=descriptor,
        tensors=tensors,
        tensor_roles=roles,
        tensor_formats=formats,
        content_hash=routed_computation_content_hash(tensors, roles),
    )
    dequantized = {
        name: quantized[name].astype(np.float32) * rounded_scales[name] for name in quantized
    }
    expected = (
        silu(activation @ dequantized["gate"].T) * (activation @ dequantized["up"].T)
    ) @ dequantized["down"].T

    actual = standard_swiglu_kernel(
        activation,
        weights,
        routing_metadata=descriptor.routing_metadata,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_router_correction_changes_selection_without_changing_raw_weights() -> None:
    routing = ColibriRoutingDescriptor(
        routing_kind="sigmoid-noaux-top-k",
        expert_count=4,
        experts_per_token=2,
        normalization="selected raw sigmoid scores renormalized",
        routed_weight_semantics="weighted sum",
    )
    selection = select_routed_experts(
        np.zeros((1, 4), dtype=np.float32),
        routing,
        correction_bias=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )

    assert selection.expert_ids.tolist() == [[2, 3]]
    np.testing.assert_array_equal(selection.weights, np.array([[0.5, 0.5]], dtype=np.float32))


def test_grouped_routing_is_deterministic_and_respects_group_mask() -> None:
    routing = ColibriRoutingDescriptor(
        routing_kind="sigmoid-noaux-grouped-top-k",
        expert_count=8,
        experts_per_token=2,
        normalization="selected scores renormalized",
        routed_weight_semantics="weighted sum",
    )
    logits = np.array([[0, 0, 1, 2, 8, 7, 3, 4]], dtype=np.float32)
    selection = select_routed_experts(
        logits,
        routing,
        routing_metadata={"n_group": 4, "topk_group": 1},
    )

    assert selection.expert_ids.tolist() == [[4, 5]]
    np.testing.assert_allclose(selection.weights.sum(axis=1), np.ones(1), atol=1e-7)


def test_generic_routed_layer_keeps_shared_and_routed_expert_zero_distinct() -> None:
    routed_zero = _weights(0, seed=21)
    routed_one = _weights(1, seed=22)
    shared_zero = _weights(0, expert_type="shared", seed=23)
    values = {
        (0, 0, "routed"): routed_zero,
        (0, 1, "routed"): routed_one,
        (0, 0, "shared"): shared_zero,
    }
    loads: list[tuple[int, int, str]] = []

    def loader(layer: int, expert: int, expert_type: str) -> RoutedComputationWeights:
        loads.append((layer, expert, expert_type))
        return values[(layer, expert, expert_type)]

    store = RoutedComputationStore(
        descriptors=tuple(value.descriptor for value in values.values()),
        loader=loader,
        residency_budget_bytes=100_000,
        cache_budget_bytes=100_000,
    )
    routing = ColibriRoutingDescriptor(
        routing_kind="softmax-top-k",
        expert_count=2,
        experts_per_token=1,
        shared_expert_count=1,
        normalization="selected probability renormalized",
        routed_weight_semantics="weighted routed plus unweighted shared",
    )
    activation = np.array([[0.25, -0.5, 1.0]], dtype=np.float32)
    logits = np.array([[4.0, -4.0]], dtype=np.float32)

    first, first_telemetry = execute_routed_layer(
        activation,
        logits,
        layer_id=0,
        routing=routing,
        store=store,
        shared_expert_ids=(0,),
    )
    second, second_telemetry = execute_routed_layer(
        activation,
        logits,
        layer_id=0,
        routing=routing,
        store=store,
        shared_expert_ids=(0,),
    )
    expected = _manual(activation, routed_zero) + _manual(activation, shared_zero)

    np.testing.assert_allclose(first, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(second, first)
    assert loads == [(0, 0, "routed"), (0, 0, "shared")]
    assert first_telemetry["cache_misses"] == 2
    assert first_telemetry["expert_movement_bytes"] == (
        routed_zero.byte_size + shared_zero.byte_size
    )
    assert second_telemetry["cache_hits"] == 2
    assert second_telemetry["expert_cache_hit_rate"] == 1.0
