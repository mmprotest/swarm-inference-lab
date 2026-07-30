from __future__ import annotations

from copy import deepcopy

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.status import evaluate_matrix_statuses


def _config(repository_root):
    return load_experiment_config(
        repository_root / "configs" / "experiments" / "experiment_001_replica_scaling.yaml"
    )


def _good_rows() -> list[dict[str, object]]:
    throughputs = {2: 100.0, 4: 170.0, 8: 270.0}
    rows: list[dict[str, object]] = []
    for workers in [2, 4, 8]:
        for concurrency in [1, 16, 64]:
            for repeat in [1, 2, 3]:
                rows.append(
                    {
                        "point_id": f"n{workers}-c{concurrency}-r{repeat}",
                        "node_count": workers,
                        "concurrent_request_count": concurrency,
                        "repeat": repeat,
                        "measured_duration_s": 30.1,
                        "completion_fraction": 1.0,
                        "committed_token_correctness": 1.0,
                        "data_plane_mode": "direct",
                        "coordinator_activation_bytes": 0,
                        "peer_streams_created": max(1, workers // 2),
                        "active_peer_pairs": max(1, workers // 2),
                        "peer_stream_reconnects": 0,
                        "meaningful_replica_fraction": 1.0,
                        "replica_imbalance_ratio": 1.05,
                        "prediction_error_fraction": 0.10,
                        "aggregate_verified_output_tokens_s": (
                            throughputs[workers]
                            if concurrency == 64
                            else max(20.0, throughputs[workers] / 2)
                        ),
                    }
                )
    return rows


def _evaluate(config, rows):
    return evaluate_matrix_statuses(
        config=config,
        point_rows=rows,
        worker_counts=[2, 4, 8],
        concurrency_counts=[1, 16, 64],
        repeats=3,
        measured_duration_s=30,
        child_validation_errors=[],
    )


def test_good_fixture_passes_every_status(repository_root) -> None:
    statuses, _, evidence = _evaluate(_config(repository_root), _good_rows())
    assert set(statuses.values()) == {"PASS"}
    assert evidence["scaling_ratios"]["2_to_4"] == 1.7
    assert evidence["scaling_ratios"]["4_to_8"] > 1.5


def test_complete_flat_scaling_is_integrity_pass_hypothesis_fail(
    repository_root,
) -> None:
    rows = _good_rows()
    for row in rows:
        if row["concurrent_request_count"] == 64:
            row["aggregate_verified_output_tokens_s"] = 100.0
    statuses, _, _ = _evaluate(_config(repository_root), rows)
    assert statuses["experiment_integrity_status"] == "PASS"
    assert statuses["scaling_hypothesis_status"] == "FAIL"
    assert statuses["overall_status"] == "FAIL"


def test_missing_point_fails_integrity(repository_root) -> None:
    statuses, _, _ = _evaluate(_config(repository_root), _good_rows()[:-1])
    assert statuses["experiment_integrity_status"] == "FAIL"
    assert statuses["overall_status"] == "FAIL"


def test_direct_mode_with_relayed_bytes_fails_direct_status(
    repository_root,
) -> None:
    rows = _good_rows()
    rows[0]["coordinator_activation_bytes"] = 1
    statuses, _, _ = _evaluate(_config(repository_root), rows)
    assert statuses["direct_data_plane_status"] == "FAIL"
    assert statuses["overall_status"] == "FAIL"


def test_starved_replicas_fail_utilisation(repository_root) -> None:
    rows = _good_rows()
    for row in rows:
        if row["concurrent_request_count"] == 64:
            row["meaningful_replica_fraction"] = 0.5
    statuses, _, _ = _evaluate(_config(repository_root), rows)
    assert statuses["replica_utilisation_status"] == "FAIL"
    assert statuses["overall_status"] == "FAIL"


def test_excess_prediction_error_fails_capacity_status(
    repository_root,
) -> None:
    rows = deepcopy(_good_rows())
    for row in rows:
        row["prediction_error_fraction"] = 0.30
    statuses, _, _ = _evaluate(_config(repository_root), rows)
    assert statuses["capacity_prediction_status"] == "FAIL"
    assert statuses["overall_status"] == "FAIL"
