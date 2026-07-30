from __future__ import annotations

import json

import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.loopback import run_loopback_experiment
from swarm_inference.experiments.runner import validate_run


@pytest.mark.asyncio
async def test_native_processes_register_execute_and_write_artifacts(
    repository_root,
    tmp_path,
) -> None:
    config = load_experiment_config(
        repository_root / "configs" / "experiments" / "scaling_loopback.yaml"
    )
    config.output_root = str(tmp_path / "runs")
    config.model.stage_count = 2
    config.model.layer_count = 2
    config.model.bytes_per_layer = 1024 * 1024
    config.model.hidden_size = 32
    config.workload.concurrent_requests = 2
    config.workload.prompt_tokens = 2
    config.workload.output_tokens = 2
    run = await run_loopback_experiment(config, worker_count=2)

    assert validate_run(run.run_dir) == []
    summary = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "single-host-loopback"
    workers = json.loads((run.run_dir / "worker_manifest.json").read_text(encoding="utf-8"))[
        "workers"
    ]
    assert len(workers) == 2
    assert all(worker["profile_source"] == "mixed" for worker in workers)
