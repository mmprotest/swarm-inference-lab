from __future__ import annotations

import json

import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.loopback_matrix import run_loopback_matrix
from swarm_inference.experiments.runner import validate_run


@pytest.mark.asyncio
async def test_loopback_matrix_writes_aggregate_report(repository_root, tmp_path) -> None:
    config = load_experiment_config(
        repository_root / "configs" / "experiments" / "first_loopback_scaling.yaml"
    )
    config.output_root = str(tmp_path / "runs")
    config.node_counts = [2, 4]
    config.concurrent_request_counts = [1, 16]
    config.warmup_s = 0
    config.steady_state_s = 0.05
    config.model.layer_count = 2
    config.model.stage_count = 2
    config.model.bytes_per_layer = 1024 * 1024
    config.model.hidden_size = 16
    config.workload.prompt_tokens = 1
    config.workload.output_tokens = 1

    run = await run_loopback_matrix(config, repeats=1, duration_s=0.05)

    assert not run.passed
    assert validate_run(run.run_dir) == []
    summary = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_integrity_status"] == "PASS"
    assert summary["scaling_hypothesis_status"] == "FAIL"
    assert summary["overall_status"] == "FAIL"
    assert len(summary["matrix_results"]) == 4
    assert len(summary["child_runs"]) == 4
