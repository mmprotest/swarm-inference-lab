from __future__ import annotations

import os
from pathlib import Path

import pytest

from swarm_inference.experiments.experiment_002 import run_experiment_002


@pytest.mark.gpu
@pytest.mark.model_download
@pytest.mark.slow
def test_four_process_real_qwen3_cuda_experiment(repository_root: Path) -> None:
    """Opt-in CI reproduction; the required local run also emits the same proof."""

    if os.environ.get("SWARM_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("set SWARM_RUN_REAL_MODEL_TESTS=1 for the full RTX 5090 proof")
    run = run_experiment_002(
        config_path=(
            repository_root / "configs" / "experiments" / "experiment_002_qwen3_real_loopback.yaml"
        ),
        output_root=repository_root / "artifacts" / "test-runs",
    )
    assert run.passed
    assert run.summary["stage_isolation_status"] == "PASS"
    assert run.summary["direct_data_plane_status"] == "PASS"
    assert run.summary["cache_replay_status"] == "PASS"
