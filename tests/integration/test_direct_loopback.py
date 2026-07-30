from __future__ import annotations

import json

import pytest

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.loopback import run_loopback_experiment
from swarm_inference.experiments.runner import validate_run


@pytest.mark.asyncio
async def test_direct_loopback_bypasses_coordinator_and_reuses_streams(
    repository_root,
    tmp_path,
) -> None:
    config = load_experiment_config(
        repository_root / "configs" / "experiments" / "experiment_001_replica_scaling.yaml"
    )
    config.matrix = None
    config.node_counts = []
    config.concurrent_request_counts = []
    config.output_root = str(tmp_path / "runs")
    config.warmup_s = 0
    config.steady_state_s = 0.2
    config.workload.concurrent_requests = 8
    config.workload.prompt_tokens = 1
    config.workload.output_tokens = 4
    config.model.cpu_work_units = 1
    config.synthetic_compute.mode = "legacy"
    config.synthetic_compute.work_units = 1
    config.profiling.enabled = True
    config.profiling.sample_interval_ms = 20

    run = await run_loopback_experiment(
        config,
        worker_count=4,
        sustained=True,
        duration_s=0.2,
    )

    assert validate_run(run.run_dir) == []
    result = run.summary["primary_result"]
    assert result["completion_fraction"] == 1
    assert result["coordinator_activation_bytes"] == 0
    assert result["worker_to_worker_activation_bytes"] > 0
    assert result["data_messages_sent"] > 0
    assert result["peer_streams_created"] <= (
        result["active_peer_pairs"] + result["peer_stream_reconnects"]
    )
    assert result["data_messages_sent"] > result["peer_streams_created"]
    assert all(operations > 0 for operations in result["operations_by_replica"].values())
    assert (run.run_dir / "profile.json").is_file()
    profile = json.loads((run.run_dir / "profile.json").read_text(encoding="utf-8"))
    assert len(profile["top_five_wall_time_sources"]) == 5
    assert profile["peer_stream_utilisation_messages_per_stream"] > 1
