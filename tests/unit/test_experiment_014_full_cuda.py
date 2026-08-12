from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _CudaRuntime,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    _parse_oracle_routes,
    _quantize_bf16_grouped_int4,
)


def test_route_trace_uses_layer_field_and_per_layer_occurrence_position(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "routes.txt"
    trace.write_text(
        "\n".join(
            (
                "0 0 1 7:0.6 9:0.4",
                "1 0 2 4:0.7 5:0.3",
                "92 0 1 8:0.55 3:0.45",
                "93 0 2 6:0.8 1:0.2",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    routes = _parse_oracle_routes(trace)

    assert routes == {
        1: {0: [7, 9], 1: [8, 3]},
        2: {0: [4, 5], 1: [6, 1]},
    }


def test_grouped_int4_matches_rounded_reciprocal_quantizer() -> None:
    values = np.zeros((1, 64), dtype=np.float32)
    values[0, 0] = np.float32(-3.28125)
    values[0, 1] = np.float32(-1.171875)
    bf16 = (values.view(np.uint32) >> np.uint32(16)).astype(np.uint16)

    quantized = _quantize_bf16_grouped_int4(bf16)

    assert quantized.scales[0, 0] == np.float32(0.46875)
    # The second value is a rounding discriminator: reciprocal-multiply gives
    # -3 (encoded nibble 5), while direct division gives -2 (nibble 6).
    assert quantized.packed[0, 0] == np.uint8(0x51)
    assert quantized.packed.shape == (1, 32)


def test_kimi_expert_batch_preflight_rejects_above_certified_limit() -> None:
    runtime = object.__new__(_CudaRuntime)
    runtime.expert_max_certified_batch = 2

    runtime._validate_expert_batch(1)
    runtime._validate_expert_batch(2)
    with pytest.raises(
        KimiCudaError,
        match=r"requested=4, certified_max=2",
    ):
        runtime._validate_expert_batch(4)
