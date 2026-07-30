from __future__ import annotations

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.experiments.runner import run_experiment, validate_run
from swarm_inference.simulation.simulator import Simulator


def test_smoke_simulation_records_all_required_metric_classes(repository_root) -> None:
    config = load_experiment_config(repository_root / "configs/experiments/smoke.yaml")
    result = Simulator(config).run()
    assert result.summary["execution_mode"] == "simulation"
    assert result.summary["aggregate_verified_output_tokens_s"] > 0
    assert result.summary["completed_verified_requests"] == 4
    assert result.requests
    assert result.workers
    assert result.stage_metrics
    assert result.network_metrics
    assert all("decode_tokens_s" in request for request in result.requests)


def test_complete_artifact_tree_and_report(tmp_path, repository_root) -> None:
    config = load_experiment_config(repository_root / "configs/experiments/smoke.yaml")
    config.output_root = str(tmp_path)
    run = run_experiment(config)
    assert run.summary["status"] == "FAIL"
    assert "correctness:greedy_token_identity" in run.summary["failed_acceptance_criteria"]
    assert validate_run(run.run_dir) == []
    html = run.report_path.read_text(encoding="utf-8")
    assert "simulation" in html
    assert "Aggregate verified throughput is" in html


def test_network_payload_size_affects_completion(repository_root) -> None:
    config = load_experiment_config(repository_root / "configs/experiments/smoke.yaml")
    from swarm_inference.simulation.network import NetworkEmulator

    network = NetworkEmulator(config.network, seed=1)
    small = network.transmit(source="a", destination="b", now_s=0, payload_bytes=100)
    large = network.transmit(source="c", destination="d", now_s=0, payload_bytes=1_000_000)
    assert large.serialization_s > small.serialization_s


def test_churn_evidence_requires_failure_while_request_is_active(repository_root) -> None:
    config = load_experiment_config(repository_root / "configs/experiments/smoke.yaml")
    config.faults.churn_rate_per_hour = 1
    config.steady_state_s = 20_000
    config.nodes[0].compute_rate_layers_s = 0.01
    result = Simulator(config).run()
    assert result.summary["failures_during_active_requests"] > 0
    assert any(
        event.event_type == "worker_failed" and int(event.details["active_request_count"]) > 0
        for event in result.events
    )
