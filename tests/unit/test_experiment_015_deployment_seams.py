from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from swarm_inference.experiments.experiment_014 import deployment_admission
from swarm_inference.experiments.experiment_014.deployment_admission import (
    AdmissionRejected,
    write_local_node_observation,
)
from swarm_inference.experiments.experiment_014.remote_acquisition import (
    AcquisitionError,
    _resolve_distribution_placement,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _gpu_observation(runtime_sha256: str) -> dict[str, object]:
    return {
        "nvidia_smi_status": "MEASURED",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "compute_capability": "8.6",
        "vram_total_bytes": 24 * 1024**3,
        "cuda_initialization_attempted": True,
        "cuda_initialized": True,
        "binary_accepts_sm86": True,
        "binary_has_forward_ptx": True,
        "native_max_certified_expert_batch": 16,
        "cuda_error_state_ok": True,
        "runtime_sha256": runtime_sha256,
    }


def test_assigned_stage_receipt_populates_fleet_runtime_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_sha = "a" * 64
    monkeypatch.setattr(
        deployment_admission,
        "inspect_local_gpu",
        lambda runtime_path, device=0: _gpu_observation(runtime_sha),
    )
    canary = tmp_path / "assigned.json"
    _write_json(
        canary,
        {
            "status": "PASS",
            "worker_id": "k3-worker-089",
            "runtime": {"sha256": runtime_sha},
            "acceptance_gates": {"assigned_stage_pass": True},
            "assigned_stage": {
                "acceptance_gates": {
                    "post_guard_safe_fixture": True,
                    "prepare_exactly_seven_calls": True,
                }
            },
        },
    )
    output = tmp_path / "observation.json"
    receipt = write_local_node_observation(
        output,
        runtime_path=tmp_path / "runtime.so",
        network_rtt_ms=1.0,
        network_bandwidth_gbps=10.0,
        assignment_sha256="b" * 64,
        checkpoint_revision="revision",
        package_version="0.1.0rc11",
        disk_path=tmp_path,
        assigned_stage_canary_path=canary,
    )
    assert receipt["runtime"]["safe_fixture_pass"] is True
    assert receipt["runtime"]["assigned_stage_canary_pass"] is True
    assert receipt["runtime"]["prepare_seven_calls_pass"] is True
    assert receipt["runtime"]["assigned_stage_canary_sha256"] == hashlib.sha256(
        canary.read_bytes()
    ).hexdigest()


def test_assigned_stage_receipt_rejects_wrong_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        deployment_admission,
        "inspect_local_gpu",
        lambda runtime_path, device=0: _gpu_observation("a" * 64),
    )
    canary = tmp_path / "assigned.json"
    _write_json(
        canary,
        {
            "status": "PASS",
            "worker_id": "k3-worker-089",
            "runtime": {"sha256": "b" * 64},
            "acceptance_gates": {"assigned_stage_pass": True},
            "assigned_stage": {"acceptance_gates": {}},
        },
    )
    with pytest.raises(AdmissionRejected, match="identity is not passing"):
        write_local_node_observation(
            tmp_path / "observation.json",
            runtime_path=tmp_path / "runtime.so",
            network_rtt_ms=1.0,
            network_bandwidth_gbps=10.0,
            assignment_sha256="c" * 64,
            checkpoint_revision="revision",
            package_version="0.1.0rc11",
            disk_path=tmp_path,
            assigned_stage_canary_path=canary,
        )


def test_distribution_placement_is_sibling_and_hash_locked(tmp_path: Path) -> None:
    placement = tmp_path / "placement.json"
    placement.write_text("{}\n", encoding="utf-8")
    distribution_path = tmp_path / "distribution.json"
    distribution = {
        "placement_manifest": placement.name,
        "placement_manifest_sha256": hashlib.sha256(placement.read_bytes()).hexdigest(),
    }
    assert _resolve_distribution_placement(distribution_path, distribution) == placement

    for unsafe in (str(placement.resolve()), "../placement.json", "missing.json"):
        candidate = {**distribution, "placement_manifest": unsafe}
        with pytest.raises(AcquisitionError):
            _resolve_distribution_placement(distribution_path, candidate)

