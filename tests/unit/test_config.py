from __future__ import annotations

import pytest
from pydantic import ValidationError

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.config.models import (
    Backend,
    ExecutionMode,
    ExperimentConfig,
    NetworkProfile,
    NodeProfile,
    SchedulerMode,
)
from swarm_inference.exceptions import ConfigurationError


def test_all_shipped_experiment_configs_validate(repository_root) -> None:
    paths = sorted((repository_root / "configs" / "experiments").glob("*.yaml"))
    assert paths
    for path in paths:
        config = load_experiment_config(path)
        assert config.seed
        assert config.execution_mode in ExecutionMode


def test_unknown_config_field_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
name: invalid
execution_mode: simulation
seed: 1
scheduler: static
network:
  name: local
  base_latency_ms: 1
  upload_bandwidth_bytes_s: 1000
  download_bandwidth_bytes_s: 1000
nodes:
  - name: node
    memory_bytes: 1000000000
    compute_rate_layers_s: 1
    supported_backends: [synthetic]
    network_profile: local
    reliability: 1
    max_concurrent_stage_operations: 1
    typo: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="typo"):
        load_experiment_config(path)


def test_network_probability_validation() -> None:
    with pytest.raises(ValidationError):
        NetworkProfile(
            name="bad",
            base_latency_ms=1,
            upload_bandwidth_bytes_s=1,
            download_bandwidth_bytes_s=1,
            packet_loss=1.1,
        )


def test_experiment_requires_nodes() -> None:
    with pytest.raises(ValidationError, match="at least one node"):
        ExperimentConfig(
            name="empty",
            execution_mode=ExecutionMode.SIMULATION,
            seed=1,
            scheduler=SchedulerMode.STATIC,
            network=NetworkProfile(
                name="local",
                base_latency_ms=1,
                upload_bandwidth_bytes_s=1,
                download_bandwidth_bytes_s=1,
            ),
            nodes=[],
        )


def test_node_profile_memory_and_backend() -> None:
    profile = NodeProfile(
        name="cpu",
        memory_bytes=1024,
        compute_rate_layers_s=10,
        supported_backends=[Backend.TORCH_CPU],
        network_profile="home-lan",
        reliability=0.9,
    )
    assert profile.memory_bytes == 1024
    assert profile.measured is False
