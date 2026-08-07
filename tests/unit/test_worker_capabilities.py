from __future__ import annotations

import builtins
import json

import pytest

from swarm_inference.config.models import (
    Backend,
    OperationKind,
    StageBenchmark,
    WorkerCapability,
)
from swarm_inference.exceptions import BackendIncompatibleError
from swarm_inference.protocol.stage_ring import STAGE_RING_PROTOCOL_VERSION
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.worker import capabilities as capabilities_module
from swarm_inference.worker.capabilities import (
    _gpu_details,
    measure_capabilities,
    measure_selected_device_benchmark,
)


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
            "supported_model_adapters": ["qwen3_dense"],
            "supported_stage_execution_backends": ["canonical-native-stage"],
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


def test_selected_device_benchmark_records_real_cpu_evidence() -> None:
    benchmark = measure_selected_device_benchmark(
        worker_class="test-cpu",
        device_identifier="cpu",
        dtype_name="float32",
        samples=3,
        warmup_iterations=0,
        small_hidden_size=16,
        representative_hidden_size=64,
        memory_copy_bytes=4096,
    )

    assert benchmark.measurement_source == "selected-device-torch"
    assert benchmark.correctness_passed
    assert benchmark.device == "cpu"
    assert benchmark.dtype == "float32"
    assert benchmark.samples == len(benchmark.sample_ms) == 3
    assert benchmark.median_ms is not None
    assert benchmark.memory_copy_bandwidth_bytes_s is not None
    assert benchmark.memory_copy_bandwidth_bytes_s > 0
    assert benchmark.dimensions["memory_copy_bytes"] == 4096


def test_capability_advertises_only_dtype_with_completed_benchmark(monkeypatch) -> None:
    benchmark = StageBenchmark(
        worker_class="test",
        operation=OperationKind.DECODE,
        sequence_length=1,
        batch_size=1,
        mean_ms=1,
        median_ms=1,
        p95_ms=1,
        samples=3,
        sample_ms=[1, 1, 1],
        device="cpu",
        dtype="float16",
        benchmark_version="selected-device-decode-v1",
        measured_at_unix_ns=1,
        memory_copy_bandwidth_bytes_s=1000,
        measurement_source="selected-device-torch",
    )
    monkeypatch.setattr(
        capabilities_module,
        "_gpu_details",
        lambda *_args, **_kwargs: (None, 0, 0, ["float32", "float16"]),
    )
    monkeypatch.setattr(
        capabilities_module,
        "measure_selected_device_benchmark",
        lambda **_kwargs: benchmark,
    )

    capability = measure_capabilities(
        backend=Backend.TORCH_CPU,
        identity=WorkerIdentity.generate(),
        device_identifier="cpu",
        benchmark_dtype="float16",
        stage_runtime_enabled=True,
    )

    assert capability.supported_dtypes == ["float16"]
    assert capability.supported_activation_dtypes == ["float16"]
    assert capability.stage_benchmarks == [benchmark]


def test_capability_rejects_dtype_when_correctness_probe_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        capabilities_module,
        "_gpu_details",
        lambda *_args, **_kwargs: (None, 0, 0, ["float32"]),
    )

    with pytest.raises(BackendIncompatibleError, match="failed the correctness probe"):
        measure_capabilities(
            backend=Backend.TORCH_CPU,
            identity=WorkerIdentity.generate(),
            device_identifier="cpu",
            benchmark_dtype="float16",
        )


def test_capability_rejects_dtype_when_selected_device_benchmark_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        capabilities_module,
        "_gpu_details",
        lambda *_args, **_kwargs: (None, 0, 0, ["float32"]),
    )

    def fail_benchmark(**_kwargs):
        raise BackendIncompatibleError("device benchmark failed")

    monkeypatch.setattr(
        capabilities_module,
        "measure_selected_device_benchmark",
        fail_benchmark,
    )

    with pytest.raises(BackendIncompatibleError, match="device benchmark failed"):
        measure_capabilities(
            backend=Backend.TORCH_CPU,
            identity=WorkerIdentity.generate(),
            device_identifier="cpu",
            benchmark_dtype="float32",
        )
