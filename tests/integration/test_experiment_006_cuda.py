from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.model_download
@pytest.mark.slow
def test_experiment_006_real_qwen_full_matrix_evidence() -> None:
    """Opt-in audit of the complete RTX 5090 run produced by the experiment CLI."""

    configured = os.environ.get("SWARM_EXPERIMENT_006_RUN")
    if not configured:
        pytest.skip("set SWARM_EXPERIMENT_006_RUN to audit the complete RTX 5090 run")
    run = Path(configured).expanduser().resolve()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["dense_partition_status"] == "PASS"
    assert summary["dense_layer_correctness_status"] == "PASS"
    assert summary["dense_token_identity_status"] == "PASS"
    assert summary["kv_partition_status"] == "PASS"
    assert summary["collective_projection_status"] == "PASS"
    assert summary["deterministic_moe_status"] == "PASS"
    assert summary["more_partitions_than_layers_status"] == "PASS"
    assert summary["pipeline_stage_counts_tested"] == [1, 4]
    assert summary["tensor_parallel_degrees_tested"] == [1, 2, 4, 8]
    assert summary["maximum_logical_layer_shards"] == 224
    assert summary["maximum_logical_pipeline_rank_workers"] == 32

    with (run / "dense_correctness.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    combinations = {
        (int(row["pipeline_stage_count"]), int(row["tensor_parallel_degree"])) for row in rows
    }
    assert combinations == {
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 8),
    }
    assert len(rows) == 64
    assert all(row["exact_token_identity"] == "PASS" for row in rows)
    assert all(int(row["generated_token_count"]) >= 32 for row in rows)
