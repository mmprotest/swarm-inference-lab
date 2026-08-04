from __future__ import annotations

import builtins
import json

import pytest

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.exceptions import BackendIncompatibleError
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.worker.capabilities import _gpu_details, measure_capabilities


def test_synthetic_backend_does_not_import_torch(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("synthetic capability detection imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model, total, available, dtypes = _gpu_details(Backend.SYNTHETIC)
    assert model is None
    assert total == available == 0
    assert "float32" in dtypes


def test_torch_worker_fails_precisely_when_pytorch_is_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("test missing torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(BackendIncompatibleError, match="PyTorch could not be imported"):
        _gpu_details(Backend.TORCH_CPU)


def test_old_capability_payload_defaults_new_stage_fields() -> None:
    old_payload = {
        "worker_id": "legacy",
        "public_key": "key",
        "hostname": "host",
        "operating_system": "test",
        "architecture": "test",
        "backend": "synthetic",
        "cpu_model": "test",
        "logical_cpu_count": 1,
        "total_ram_bytes": 1024,
        "available_ram_bytes": 512,
        "upload_bandwidth_bytes_s": 0,
        "download_bandwidth_bytes_s": 0,
        "coordinator_latency_ms": 0,
    }
    capability = WorkerCapability.model_validate(old_payload)
    assert capability.control_endpoint is None
    assert capability.data_plane_endpoint is None
    assert capability.currently_loaded_model_revisions == []
    assert capability.currently_loaded_topology_ids == []
    assert capability.active_session_count == 0
    assert not capability.stage_runtime_enabled


def test_new_stage_capability_fields_round_trip() -> None:
    capability = WorkerCapability.model_validate(
        {
            "worker_id": "stage-worker",
            "public_key": "key",
            "hostname": "host",
            "operating_system": "test",
            "architecture": "test",
            "backend": "torch-cpu",
            "cpu_model": "test",
            "logical_cpu_count": 1,
            "total_ram_bytes": 1024,
            "available_ram_bytes": 512,
            "upload_bandwidth_bytes_s": 0,
            "download_bandwidth_bytes_s": 0,
            "coordinator_latency_ms": 0,
            "control_endpoint": "worker.test:50052",
            "data_plane_endpoint": "worker.test:50053",
            "device_identifier": "cpu",
            "stage_ring_protocol_version": STAGE_RING_PROTOCOL_VERSION,
            "supported_model_adapters": ["olmoe"],
            "supported_stage_execution_backends": ["canonical-contiguous-olmoe"],
            "supported_activation_dtypes": ["float32"],
            "configured_memory_limit_bytes": 512,
            "currently_loaded_model_revisions": ["revision"],
            "currently_loaded_topology_ids": ["topology"],
            "active_session_count": 2,
            "stage_runtime_enabled": True,
        }
    )
    restored = WorkerCapability.model_validate_json(json.dumps(capability.model_dump(mode="json")))
    assert restored == capability


def test_capability_measurement_does_not_invent_unmeasured_network_rates() -> None:
    capability = measure_capabilities(
        backend=Backend.SYNTHETIC,
        identity=WorkerIdentity.generate(),
        endpoint="worker.test:50052",
        control_endpoint="worker.test:50052",
        data_plane_endpoint=None,
    )
    assert capability.upload_bandwidth_bytes_s == 0
    assert capability.download_bandwidth_bytes_s == 0
