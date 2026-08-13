from swarm_inference.experiments.experiment_020.topology import evaluate_topology

GATES = {
    "intra_pod": {
        "maximum_rtt_ms": 1.0,
        "minimum_bandwidth_gbps_at_maximum_rtt": 10.0,
    },
    "inter_pod": {
        "maximum_rtt_ms": 10.0,
        "minimum_bandwidth_gbps_at_maximum_rtt": 2.0,
    },
}


def _path(kind: str, rtt: float, bandwidth: float):
    return {
        "source": "a",
        "destination": "b",
        "kind": kind,
        "samples": [
            {"rtt_ms": rtt, "upload_gbps": bandwidth, "download_gbps": bandwidth}
            for _ in range(10)
        ],
    }


def test_simulated_topology_is_accepted_only_after_measured_paths_pass() -> None:
    result = evaluate_topology(
        [_path("intra_pod", 0.5, 20.0), _path("inter_pod", 5.0, 5.0)], GATES
    )
    assert result["decision"] == "TOPOLOGY_ACCEPTED"
    assert result["marketplace_metadata_trusted_without_probe"] is False


def test_one_slow_path_rejects_topology() -> None:
    result = evaluate_topology(
        [_path("intra_pod", 0.5, 20.0), _path("inter_pod", 15.0, 5.0)], GATES
    )
    assert result["decision"] == "TOPOLOGY_REJECTED"
