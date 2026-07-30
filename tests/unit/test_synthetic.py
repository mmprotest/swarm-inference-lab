from __future__ import annotations

import numpy as np
import pytest

from swarm_inference.config.models import OperationKind, SyntheticModelConfig
from swarm_inference.exceptions import ConfigurationError
from swarm_inference.model.synthetic import SyntheticStageModule, synthetic_activation
from swarm_inference.simulation.model import build_synthetic_stages


def test_synthetic_output_is_deterministic_and_layer_dependent() -> None:
    config = SyntheticModelConfig(layer_count=4, stage_count=2, hidden_size=8)
    stage = build_synthetic_stages(config)[0]
    source = synthetic_activation([1, 2], hidden_size=8, dtype="float32")
    first = SyntheticStageModule(config=config, stage=stage)
    second = SyntheticStageModule(config=config, stage=stage)
    left = first.execute(
        source,
        request_id="r",
        operation=OperationKind.PREFILL,
        token_position=0,
        sequence_length=2,
        cache_generation=0,
    )
    right = second.execute(
        source,
        request_id="r",
        operation=OperationKind.PREFILL,
        token_position=0,
        sequence_length=2,
        cache_generation=0,
    )
    assert np.array_equal(left, right)
    assert not np.array_equal(left, source)


def test_synthetic_decode_requires_contiguous_cache() -> None:
    config = SyntheticModelConfig(layer_count=2, stage_count=1, hidden_size=4)
    module = SyntheticStageModule(config=config, stage=build_synthetic_stages(config)[0])
    source = synthetic_activation([1], hidden_size=4, dtype="float32")
    with pytest.raises(ConfigurationError, match="missing cache"):
        module.execute(
            source,
            request_id="r",
            operation=OperationKind.DECODE,
            token_position=1,
            sequence_length=1,
            cache_generation=0,
        )


def test_canary_detects_corruption() -> None:
    config = SyntheticModelConfig(layer_count=2, stage_count=1, hidden_size=8)
    stage = build_synthetic_stages(config)[0]
    good = SyntheticStageModule(config=config, stage=stage).canary()
    bad = SyntheticStageModule(config=config, stage=stage, corrupt=True).canary()
    assert good != bad
