from __future__ import annotations

import inspect
import time

import numpy as np
import pytest

from swarm_inference.config.models import (
    OperationKind,
    SyntheticComputeConfig,
    SyntheticModelConfig,
)
from swarm_inference.exceptions import ConfigurationError
from swarm_inference.experiments.calibration import calibrate_synthetic_compute
from swarm_inference.model.synthetic import (
    SyntheticStageModule,
    deterministic_cpu_kernel,
    synthetic_activation,
)
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


def test_calibrated_cpu_output_changes_by_stage_and_token() -> None:
    config = SyntheticModelConfig(
        layer_count=2,
        stage_count=2,
        hidden_size=32,
        cpu_work_units=2,
        cpu_kernel_buffer_bytes=16 * 1024,
    )
    stages = build_synthetic_stages(config)
    source = synthetic_activation([1], hidden_size=32, dtype="float32")
    token_module = SyntheticStageModule(config=config, stage=stages[0])
    stage_zero = token_module.execute(
        source,
        request_id="request",
        operation=OperationKind.PREFILL,
        token_position=0,
        sequence_length=1,
        cache_generation=0,
    )
    stage_one = SyntheticStageModule(config=config, stage=stages[1]).execute(
        source,
        request_id="request",
        operation=OperationKind.PREFILL,
        token_position=0,
        sequence_length=1,
        cache_generation=0,
    )
    later_token = token_module.execute(
        source,
        request_id="request",
        operation=OperationKind.DECODE,
        token_position=1,
        sequence_length=1,
        cache_generation=0,
    )
    assert not np.array_equal(stage_zero, stage_one)
    assert not np.array_equal(stage_zero, later_token)


def test_primary_cpu_kernel_does_real_work_without_sleep(monkeypatch) -> None:
    monkeypatch.setattr(
        time,
        "sleep",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sleep must not be used")),
    )
    before = time.process_time_ns()
    results = [
        deterministic_cpu_kernel(
            b"input",
            work_units=1000,
            buffer_bytes=16 * 1024,
            seed_material=b"context",
        )
        for _ in range(16)
    ]
    cpu_time_ns = time.process_time_ns() - before
    assert len(set(results)) == 1
    assert cpu_time_ns > 0
    assert "time.sleep(" not in inspect.getsource(deterministic_cpu_kernel)


def test_calibration_reaches_range_and_returns_frozen_work_units() -> None:
    calibration = calibrate_synthetic_compute(
        SyntheticComputeConfig(
            mode="calibrated_cpu",
            target_stage_ms=8,
            acceptable_min_ms=5,
            acceptable_max_ms=12,
            activation_bytes=16 * 1024,
            calibration_warmup_iterations=8,
            calibration_measurement_iterations=24,
        ),
        cpu_id=None,
        timeout_s=60,
    )
    assert calibration.acceptable
    assert 5 <= calibration.median_stage_ms <= 12
    assert calibration.work_units > 0
    assert calibration.measurement_iterations == 24


def test_worker_thread_environment_is_single_threaded(repository_root) -> None:
    source = (repository_root / "src" / "swarm_inference" / "worker" / "process_main.py").read_text(
        encoding="utf-8"
    )
    for variable in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        assert variable in source
    assert 'os.environ.setdefault(_thread_variable, "1")' in source
