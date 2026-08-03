from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from swarm_inference.backends.colibri.cuda import (
    ColibriCudaError,
    run_colibri_cuda_kernel_proof,
    run_colibri_real_olmoe_cuda_expert,
)
from swarm_inference.backends.colibri.schemas import ColibriCapabilityReport
from swarm_inference.experiments.experiment_010.bundle import (
    REQUIRED_FILES,
    Experiment010Bundle,
)
from swarm_inference.experiments.experiment_010.kimi import (
    KIMI_ROUTED_EXPERTS,
    MXFP4_GROUP_SIZE,
    NativeMXFP4Runtime,
    deterministic_kimi_expert,
    execute_kimi_topk,
    kimi_fixture_inventory,
    mxfp4_zero_group_count,
)
from swarm_inference.experiments.experiment_010.schemas import EvidenceCategory
from swarm_inference.simulation.expert_model import (
    calibrate_expert_simulator,
    project_virtual_topologies,
)


def _capability(*, supports_cuda: bool, proof: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "bridge_version": "1.0",
        "platform": "Windows",
        "architecture": "AMD64",
        "model_families": ["olmoe"],
        "execution_backends": ["cpu", "cuda"] if supports_cuda else ["cpu"],
        "quantization_formats": ["int8"],
        "supports_cpu": True,
        "supports_cuda": supports_cuda,
        "supports_vulkan": False,
        "supports_metal": False,
        "supports_multi_gpu": False,
        "supports_expert_residency": True,
        "supports_route_trace": True,
        "supports_usage_history": True,
        "supports_expert_prefetch": True,
        "supports_dynamic_reconfiguration": False,
        "supports_native_mxfp4": False,
        "supports_tensor_microshards": False,
        "supports_full_expert_placement": True,
        "supports_exact_replay": True,
        "supports_prefill_decode_separation": True,
        "storage_tiers": [],
        "gpu_devices": [],
        "cpu": {},
        "memory": {},
        "storage": {},
        "cuda_kernel_proof": proof,
    }


def test_colibri_cuda_capability_truthful() -> None:
    proof = {
        "dll_loaded": True,
        "device_detected": True,
        "kernel_executed": True,
        "correctness_passed": True,
    }
    report = ColibriCapabilityReport.model_validate(_capability(supports_cuda=True, proof=proof))
    assert report.supports_cuda is True
    assert report.cuda_kernel_proof == proof


def test_colibri_cuda_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "failed-proof.json"
    with pytest.raises(ColibriCudaError, match="missing"):
        run_colibri_cuda_kernel_proof(tmp_path / "missing.dll", output_path=output)
    proof = json.loads(output.read_text(encoding="utf-8"))
    assert proof["dll_loaded"] is False
    assert proof["kernel_executed"] is False
    with pytest.raises(ValidationError, match="successful bound kernel proof"):
        ColibriCapabilityReport.model_validate(_capability(supports_cuda=True, proof=proof))


@pytest.fixture(scope="module")
def live_cuda_proof(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[2]
    dll = repository_root / "build" / "colibri" / "bin" / "coli_cuda.dll"
    if not dll.is_file():
        pytest.skip("the pinned Colibri CUDA runtime has not been built")
    return run_colibri_cuda_kernel_proof(
        dll,
        output_path=tmp_path_factory.mktemp("colibri-cuda") / "proof.json",
        latent_dimension=32,
        intermediate_dimension=32,
        batch_rows=2,
    )


@pytest.mark.gpu
def test_colibri_cuda_kernel_execution_proof(live_cuda_proof: dict[str, Any]) -> None:
    assert live_cuda_proof["dll_loaded"]
    assert live_cuda_proof["device_detected"]
    assert live_cuda_proof["kernel_executed"]
    assert live_cuda_proof["resident_tensor_bytes"] > 0


@pytest.mark.gpu
def test_colibri_cuda_cpu_equivalence(live_cuda_proof: dict[str, Any]) -> None:
    assert live_cuda_proof["correctness_passed"] is True
    assert live_cuda_proof["maximum_absolute_error"] <= 2e-4
    assert live_cuda_proof["relative_l2_error"] <= 2e-4


@pytest.mark.gpu
def test_real_olmoe_cuda_expert(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    dll_candidates = (
        repository_root / "build" / "colibri" / "source" / "c" / "coli_cuda.dll",
        repository_root / "build" / "colibri" / "bin" / "coli_cuda.dll",
    )
    dll = next((path for path in dll_candidates if path.is_file()), None)
    model = (
        repository_root
        / "artifacts"
        / "models"
        / "colibri"
        / "olmoe-1b-7b-0125-instruct-merged"
    )
    if dll is None or not model.is_dir():
        pytest.skip("the exact Level A model and Colibri CUDA runtime are required")
    proof = run_colibri_real_olmoe_cuda_expert(
        dll,
        model,
        layer_id=0,
        expert_id=5,
        output_path=tmp_path / "real-expert.json",
    )
    assert proof["native_tensor_bytes_used"] is True
    assert proof["kernel_executed"] is True
    assert proof["nonzero_vram_residency"] is True
    assert proof["no_silent_cpu_fallback"] is True
    assert proof["correctness_passed"] is True


@pytest.mark.gpu
def test_real_olmoe_cuda_generation() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    evidence = (
        repository_root
        / "artifacts"
        / "runs"
        / "experiment-010-correction-work"
        / "phase-9"
        / "real_model_cuda_results.json"
    )
    if not evidence.is_file():
        pytest.skip("the real CUDA token-path workload has not been reproduced")
    document = json.loads(evidence.read_text(encoding="utf-8"))
    row = document["result"]
    assert document["complete"] is True
    assert row["exact_token_identity"] is True
    assert row["router_trace_identity"] is True
    assert row["remote_results_consumed"] > 0
    assert row["cuda_execution_count"] > 0
    assert row["gpu_resident_bytes"] > 0
    assert row["cuda_fallback_count"] == 0


@pytest.fixture(scope="module")
def kimi_system() -> tuple[list[Any], np.ndarray, list[float]]:
    experts = [
        deterministic_kimi_expert(
            expert_id=index,
            latent_dimension=32,
            intermediate_dimension=64,
            seed=1010,
        )
        for index in range(KIMI_ROUTED_EXPERTS)
    ]
    activation = np.random.default_rng(1010).normal(0, 0.1, (2, 32)).astype(np.float32)
    routing = [1 / KIMI_ROUTED_EXPERTS] * KIMI_ROUTED_EXPERTS
    return experts, activation, routing


def test_kimi_mxfp4_fixture_layout(kimi_system: tuple[list[Any], np.ndarray, list[float]]) -> None:
    expert = kimi_system[0][0]
    assert MXFP4_GROUP_SIZE == 32
    assert expert.gate.packed.shape == (64, 16)
    assert expert.gate.scales.shape == (64, 1)
    assert expert.gate.packed.dtype == np.uint8
    assert expert.gate.scales.dtype == np.uint8


def test_kimi_top16_routing(kimi_system: tuple[list[Any], np.ndarray, list[float]]) -> None:
    experts, activation, routing = kimi_system
    output, metrics = execute_kimi_topk(activation, experts, routing)
    assert output.shape == kimi_system[1].shape
    assert metrics["expert_count"] == 16
    assert metrics["native_format"] == "mxfp4_e2m1_ue8m0_g32"


def test_kimi_whole_expert_fixture(
    kimi_system: tuple[list[Any], np.ndarray, list[float]],
) -> None:
    experts, activation, routing = kimi_system
    output, metrics = execute_kimi_topk(activation, experts, routing)
    assert np.isfinite(output).all()
    assert metrics["shard_count"] == 1
    assert metrics["persistent_dequantized_bytes"] == 0


def test_kimi_microshard_fixture(
    kimi_system: tuple[list[Any], np.ndarray, list[float]],
) -> None:
    experts, activation, routing = kimi_system
    whole, _ = execute_kimi_topk(activation, experts, routing)
    sharded, metrics = execute_kimi_topk(
        activation, experts, routing, shard_ranges=[(0, 32), (32, 64)]
    )
    np.testing.assert_allclose(sharded, whole, rtol=2e-6, atol=1e-10)
    assert metrics["shard_ranges"] == [[0, 32], [32, 64]]


def test_dense_kimi_fixture_has_no_zero_groups(
    kimi_system: tuple[list[Any], np.ndarray, list[float]],
) -> None:
    experts, _, _ = kimi_system
    assert all(
        mxfp4_zero_group_count(tensor) == 0
        for expert in experts
        for tensor in (expert.gate, expert.up, expert.down)
    )
    assert kimi_fixture_inventory(experts)["dense_fixture"] is True


def test_dense_kimi_fixture_executes_all_groups(
    kimi_system: tuple[list[Any], np.ndarray, list[float]],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    library = repository_root / "build" / "colibri" / "bin" / "coli_kimi_mxfp4.dll"
    if not library.is_file():
        pytest.skip("the compiled Colibri MXFP4 fixture adapter is required")
    experts, activation, routing = kimi_system
    _, metrics = execute_kimi_topk(
        activation,
        experts,
        routing,
        native_runtime=NativeMXFP4Runtime(library),
    )
    expected_groups = (
        3
        * experts[0].latent_dimension
        * experts[0].intermediate_dimension
        // MXFP4_GROUP_SIZE
        * KIMI_ROUTED_EXPERTS
    )
    assert metrics["groups_processed"] == expected_groups
    assert metrics["groups_with_arithmetic"] == expected_groups
    assert metrics["zero_quantization_groups"] == 0
    assert metrics["arithmetic_backend"] == NativeMXFP4Runtime.ABI


def _calibrated_model() -> tuple[Any, dict[str, Any]]:
    rows = []
    for index in range(8):
        compute = 100_000 * (index + 1)
        total = 50_000 + 1.5 * compute
        rows.append(
            {
                "configuration_id": f"k{index}",
                "workload_id": "operator",
                "worker_compute_ns": compute,
                "measured_total_ns": total,
                "measured_throughput": 1e9 / total,
                "measured_p95_latency_ms": total / 1e6,
                "verified_tokens": 1,
            }
        )
    model, _ = calibrate_expert_simulator(rows)
    return model, rows[0]


def test_kimi_projection_category_labels(
    kimi_system: tuple[list[Any], np.ndarray, list[float]],
) -> None:
    model, base = _calibrated_model()
    projections = project_virtual_topologies(model, base)
    inventory = kimi_fixture_inventory(kimi_system[0])
    assert inventory["category"] == EvidenceCategory.SYNTHETIC_FIXTURE.value
    assert all(
        row["category"] == EvidenceCategory.SIMULATED_CALIBRATED.value for row in projections
    )
    assert "not Kimi K3 weights" in inventory["description"]


def test_measured_emulated_projected_separation() -> None:
    assert {item.value for item in EvidenceCategory} == {
        "MEASURED_PHYSICAL",
        "MEASURED_SINGLE_HOST",
        "MEASURED_NETWORK_EMULATION",
        "SYNTHETIC_FIXTURE",
        "SIMULATED_CALIBRATED",
        "SIMULATED_UNCALIBRATED",
        "PROJECTED",
    }


def test_partial_result_resume(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    first = Experiment010Bundle(root, resume=False)
    first.complete_configuration("direct-tcp")
    resumed = Experiment010Bundle(root, resume=True)
    assert resumed.is_configuration_complete("direct-tcp")
    resumed.complete_configuration("shared-memory")
    assert set(resumed.checkpoint["completed_configurations"]) == {
        "direct-tcp",
        "shared-memory",
    }


def test_evidence_raw_data_present(tmp_path: Path) -> None:
    bundle = Experiment010Bundle(tmp_path / "evidence", resume=False)
    for name in REQUIRED_FILES:
        bundle.write_text(name, "{}\n")
    audit = bundle.audit()
    assert audit["complete"] is True
    assert audit["missing"] == []
    assert audit["empty"] == []
