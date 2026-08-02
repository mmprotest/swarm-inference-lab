from __future__ import annotations

import time

import numpy as np
import pytest

from swarm_inference.experiments.experiment_010.codecs import decode_array, encode_array
from swarm_inference.experiments.experiment_010.expert import (
    ExpertWeights,
    deterministic_expert,
    execute_expert,
    reduce_partials,
    slice_expert_weights,
)
from swarm_inference.experiments.experiment_010.schemas import (
    NetworkShapeProfile,
    ReductionMode,
    TransportCodec,
)
from swarm_inference.experiments.experiment_010.transport import (
    NetworkShaper,
    ShapedTransportError,
)


def _geometry() -> tuple[np.ndarray, ExpertWeights]:
    activation = np.random.default_rng(1010).normal(0, 0.1, (3, 64)).astype(np.float32)
    weights = deterministic_expert(latent_dimension=64, intermediate_dimension=64, seed=1010)
    return activation, weights


def test_microshard_slice_alignment() -> None:
    _, weights = _geometry()
    with pytest.raises(ValueError, match="invalid"):
        slice_expert_weights(weights, hidden_start=32, hidden_end=65)


def test_microshard_quantization_alignment() -> None:
    _, weights = _geometry()
    quantized = ExpertWeights(
        up=weights.up,
        gate=weights.gate,
        down=weights.down,
        content_hash=weights.content_hash,
        native_format="native-test",
        scale_group_size=32,
    )
    with pytest.raises(ValueError, match="quantisation group"):
        slice_expert_weights(quantized, hidden_start=16, hidden_end=64)
    aligned = slice_expert_weights(quantized, hidden_start=32, hidden_end=64)
    assert aligned.hidden_offset == 32


def _reconstruct(ranges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    activation, weights = _geometry()
    whole = execute_expert(activation, weights)
    partials = []
    for index, (start, end) in enumerate(ranges):
        sliced = slice_expert_weights(weights, hidden_start=start, hidden_end=end)
        partials.append(
            (
                f"worker-{index}",
                execute_expert(
                    activation,
                    sliced,
                    hidden_start=start,
                    hidden_end=end,
                ),
            )
        )
    return whole, reduce_partials(partials, mode=ReductionMode.FIXED_ORDER_FP32)


def test_microshard_reconstruction() -> None:
    whole, reconstructed = _reconstruct([(0, 32), (32, 64)])
    np.testing.assert_allclose(reconstructed, whole, rtol=2e-6, atol=2e-8)


def test_microshard_whole_expert_equivalence() -> None:
    whole, reconstructed = _reconstruct([(0, 16), (16, 48), (48, 64)])
    relative = np.linalg.norm((reconstructed - whole).ravel()) / np.linalg.norm(whole.ravel())
    assert relative < 2e-6


def test_microshard_fixed_reduction() -> None:
    partials = [
        ("b", np.asarray([[3.0, 4.0]], dtype=np.float32)),
        ("a", np.asarray([[1.0, 2.0]], dtype=np.float32)),
    ]
    forward = reduce_partials(partials, mode=ReductionMode.FIXED_ORDER_FP32)
    reverse = reduce_partials(list(reversed(partials)), mode=ReductionMode.FIXED_ORDER_FP32)
    np.testing.assert_array_equal(forward, reverse)


def test_microshard_equal_partition() -> None:
    _, reconstructed = _reconstruct([(0, 32), (32, 64)])
    assert reconstructed.shape == (3, 64)


def test_microshard_asymmetric_partition() -> None:
    whole, reconstructed = _reconstruct([(0, 16), (16, 64)])
    np.testing.assert_allclose(reconstructed, whole, rtol=2e-6, atol=2e-8)


def test_network_shaper_bandwidth() -> None:
    shaper = NetworkShaper(
        NetworkShapeProfile(name="bandwidth", bandwidth_bps=8e6, one_way_latency_ms=0)
    )
    result = shaper.enforce(1000, direction="request")
    assert result["imposed_delay_ns"] >= 1_000_000
    assert shaper.snapshot()["payload_bytes"] == 1000


def test_network_shaper_latency() -> None:
    shaper = NetworkShaper(NetworkShapeProfile(name="latency", one_way_latency_ms=2.5))
    started = time.perf_counter_ns()
    result = shaper.enforce(1, direction="request")
    assert result["imposed_delay_ns"] == 2_500_000
    assert time.perf_counter_ns() - started >= 2_000_000


def test_network_shaper_jitter() -> None:
    profile = NetworkShapeProfile(name="jitter", one_way_latency_ms=5, jitter_ms=2, seed=1010)
    first = NetworkShaper(profile).enforce(1, direction="request")
    second = NetworkShaper(profile).enforce(1, direction="request")
    assert first["imposed_delay_ns"] == second["imposed_delay_ns"]
    assert first["imposed_delay_ns"] != 5_000_000


def test_network_shaper_outage() -> None:
    shaper = NetworkShaper(
        NetworkShapeProfile(
            name="outage",
            one_way_latency_ms=0,
            outage_intervals_ms=[(0, 10_000)],
        )
    )
    with pytest.raises(ShapedTransportError, match="outage"):
        shaper.enforce(1, direction="request")


def _codec_roundtrip(codec: TransportCodec) -> tuple[np.ndarray, np.ndarray, int, int]:
    source = np.random.default_rng(1010).normal(0, 0.2, (4, 32)).astype(np.float32)
    encoded = encode_array(source, name="activation", codec=codec)
    decoded = decode_array(encoded.metadata, encoded.payload)
    return source, decoded.array, encoded.metadata.raw_bytes, encoded.metadata.encoded_bytes


def test_transport_codec_fp32() -> None:
    source, decoded, raw_bytes, encoded_bytes = _codec_roundtrip(TransportCodec.RAW_FP32)
    np.testing.assert_array_equal(decoded, source)
    assert encoded_bytes == raw_bytes


def test_transport_codec_fp16() -> None:
    source, decoded, raw_bytes, encoded_bytes = _codec_roundtrip(TransportCodec.RAW_FP16)
    np.testing.assert_allclose(decoded, source, rtol=1e-3, atol=1e-3)
    assert encoded_bytes == raw_bytes // 2


def test_transport_codec_int8() -> None:
    source, decoded, raw_bytes, encoded_bytes = _codec_roundtrip(TransportCodec.INT8_PER_VECTOR)
    relative = np.linalg.norm((decoded - source).ravel()) / np.linalg.norm(source.ravel())
    assert relative < 0.02
    assert encoded_bytes == raw_bytes // 4


def test_transport_codec_lossless() -> None:
    source = np.zeros((16, 64), dtype=np.float32)
    encoded = encode_array(source, name="activation", codec=TransportCodec.LOSSLESS_GENERAL)
    decoded = decode_array(encoded.metadata, encoded.payload).array
    np.testing.assert_array_equal(decoded, source)
    assert encoded.metadata.encoded_bytes < encoded.metadata.raw_bytes
