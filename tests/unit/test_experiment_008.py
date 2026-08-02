from __future__ import annotations

import struct
from pathlib import Path

import pytest

from swarm_inference.config.experiment_008 import load_experiment_008_config
from swarm_inference.experiments.experiment_008.acquisition import ResolvedModel
from swarm_inference.experiments.experiment_008.bundle import EvidenceBundle
from swarm_inference.experiments.experiment_008.cost_model import (
    CandidateEstimate,
    CriticalTask,
    UtilityObjective,
    critical_path,
    planner_regret_fraction,
    prediction_quality,
    select_positive_utility,
)
from swarm_inference.experiments.experiment_008.experts import (
    BoundedExpertPredictor,
    ExpertActivation,
    ExpertLRUCache,
    ExpertMeasurement,
    activation_statistics,
    classify_residency,
    evaluate_prediction,
)
from swarm_inference.experiments.experiment_008.fixture import validate_tiny_moe_fixture
from swarm_inference.experiments.experiment_008.gguf import (
    build_expert_microshards,
    build_preflight,
    build_tensor_tiles,
    inspect_gguf,
    tensor_role,
)
from swarm_inference.experiments.experiment_008.hardware import (
    latency_summary,
    percentile,
    profile_fingerprint,
)
from swarm_inference.experiments.experiment_008.planning import (
    BackendCapabilities,
    baseline_search_space,
    build_phase_plan,
    select_best_stock_by_workload,
)
from swarm_inference.experiments.experiment_008.runner import (
    _BUFFER,
    Experiment008Options,
    _matched_cpu_moe_utility,
    _model_execution_precheck,
)
from swarm_inference.experiments.experiment_008.schemas import (
    EvidenceClass,
    ExecutionStatus,
    Experiment008Verdict,
    ExpertMicroshard,
    GateResult,
    GateStatus,
    TensorTile,
    overall_verdict,
)


def _tile(role: str, *, start: int = 0, end: int = 16) -> TensorTile:
    return TensorTile(
        model_id="fixture-moe",
        model_revision="immutable",
        layer_id=2,
        tensor_name=f"blk.2.ffn_{role}_exps.weight",
        tensor_role=f"routed_expert_{role}_projection",
        expert_id=7,
        logical_shape=[32, 16],
        logical_slice={"projection_range": {"hidden_start": start, "hidden_end": end}},
        physical_layout="row_major",
        dtype="float32",
        quantization="none",
        quantization_metadata={},
        accumulator_dtype="float32",
        byte_size=2048,
        content_hash="sha256:" + "0" * 64,
        allowed_backends=["torch-fixture"],
        current_residency="CPU",
        planned_execution_device="GPU_AFTER_PREFETCH",
    )


def test_experiment_options_reject_multi_character_configuration() -> None:
    options = Experiment008Options(config_path=Path("config.yaml"), full=True)
    options.configuration = "AB"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="one of A through G"):
        options.validate()


def test_model_execution_precheck_reuses_matching_real_generation(tmp_path: Path) -> None:
    bundle = EvidenceBundle(tmp_path / "bundle", resume=False)
    bundle.write_json("model_preflight.json", {"model_file_sha256": "abc"})
    bundle.write_json(
        "baseline_search.json",
        {
            "results": [
                {
                    "candidate_id": "stock-real",
                    "workload": "decode",
                    "status": "COMPLETED",
                    "classification": "MEASURED",
                    "exit_code": 0,
                }
            ]
        },
    )
    config = load_experiment_008_config(
        Path("configs/experiments/experiment_008_adaptive_moe.yaml")
    )
    resolved = ResolvedModel(
        candidate="preferred",
        model_id="fixture",
        artifact_repository="fixture/repo",
        filename="fixture.gguf",
        quantization="Q4_K_M",
        architecture="fixturemoe",
        requested_revision="commit",
        resolved_revision="sha256:abc",
        path=str(tmp_path / "fixture.gguf"),
        source="fixture",
        file_size=1,
        file_sha256="abc",
    )
    result = _model_execution_precheck(
        bundle=bundle,
        config=config,
        executable=tmp_path / "unused.exe",
        resolved=resolved,
        candidate_name="preferred",
        capabilities=BackendCapabilities(
            conventional_layer_offload=True,
            tensor_buffer_override=True,
            cpu_moe=True,
            asynchronous_backend_scheduler=True,
            operation_level_overlap_trace=False,
            expert_routing_trace=False,
            per_expert_dynamic_residency=False,
            expert_prefetch=False,
            separate_process_phase_plans=True,
            in_request_phase_switch=False,
            deterministic_greedy_tokens=True,
            final_logits=False,
        ),
    )
    assert result["status"] == "COMPLETED"
    assert result["reused"] is True
    assert result["source_candidate_id"] == "stock-real"


def _gate(gate_id: int, status: GateStatus) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        name=f"gate-{gate_id}",
        status=status,
        evidence_class=EvidenceClass.MEASURED,
        reasons=["test"],
    )


def test_experiment_008_config_uses_required_models(repository_root: Path) -> None:
    config = load_experiment_008_config(
        repository_root / "configs" / "experiments" / "experiment_008_adaptive_moe.yaml"
    )
    assert config.models.preferred.model_id == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert config.models.preferred.quantization == "Q4_K_M"
    assert config.models.fallback.model_id == "Qwen/Qwen3.6-35B-A3B"
    assert config.models.fallback.quantization == "Q8_0"
    assert config.workloads.decode_prompt_count >= 20
    assert config.workloads.long_prompt_count >= 5


def test_expert_microshard_requires_matching_projection_ranges() -> None:
    shard = ExpertMicroshard(
        layer_id=2,
        expert_id=7,
        hidden_start=0,
        hidden_end=16,
        up=_tile("up"),
        gate=_tile("gate"),
        down=_tile("down"),
    )
    assert shard.down.logical_slice["projection_range"]["hidden_end"] == 16
    with pytest.raises(ValueError, match="same projection range"):
        ExpertMicroshard(
            layer_id=2,
            expert_id=7,
            hidden_start=0,
            hidden_end=16,
            up=_tile("up"),
            gate=_tile("gate", end=8),
            down=_tile("down"),
        )


def test_quick_or_unmeasured_run_cannot_issue_official_pass() -> None:
    passing = [_gate(index, GateStatus.PASS) for index in range(1, 7)]
    assert (
        overall_verdict(
            passing,
            real_model_generation_succeeded=False,
            official_full_run=False,
        )
        == Experiment008Verdict.FAIL
    )
    assert (
        overall_verdict(
            passing,
            real_model_generation_succeeded=True,
            official_full_run=False,
        )
        == Experiment008Verdict.PARTIAL
    )


def test_verdict_requires_foundational_gates_and_separates_performance() -> None:
    passing = [_gate(index, GateStatus.PASS) for index in range(1, 7)]
    assert (
        overall_verdict(passing, real_model_generation_succeeded=True, official_full_run=True)
        == Experiment008Verdict.PASS_STRONG
    )
    passing[2] = _gate(3, GateStatus.FAIL)
    assert (
        overall_verdict(passing, real_model_generation_succeeded=True, official_full_run=True)
        == Experiment008Verdict.PASS_CAPACITY_AND_ARCHITECTURE
    )
    passing[1] = _gate(2, GateStatus.FAIL)
    assert (
        overall_verdict(passing, real_model_generation_succeeded=True, official_full_run=True)
        == Experiment008Verdict.PARTIAL
    )


def test_noncompleted_observation_requires_explicit_reason() -> None:
    from swarm_inference.experiments.experiment_008.schemas import BenchmarkObservation

    with pytest.raises(ValueError, match="unavailable_reason"):
        BenchmarkObservation(
            configuration="D",
            workload="decode",
            plan_id="d-decode",
            status=ExecutionStatus.UNSUPPORTED,
            metrics={"decode_tokens_per_second": None},
        )


def test_observation_preserves_structured_metric_diagnostics() -> None:
    from swarm_inference.experiments.experiment_008.schemas import BenchmarkObservation

    observation = BenchmarkObservation(
        configuration="A",
        workload="decode",
        plan_id="stock",
        status=ExecutionStatus.COMPLETED,
        evidence_class=EvidenceClass.MEASURED,
        metrics={"errors": [], "per_request": [{"tokens": 16}]},
    )
    assert observation.metrics["errors"] == []


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _write_minimal_moe_gguf(path: Path) -> None:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "fixturemoe"),
        ("general.alignment", 4, 32),
        ("fixturemoe.block_count", 4, 1),
        ("fixturemoe.expert_count", 4, 2),
        ("fixturemoe.expert_used_count", 4, 1),
        ("fixturemoe.expert_shared_count", 4, 1),
    ]
    tensors = [
        ("blk.0.ffn_up_exps.weight", (4, 8, 2), 0, 0),
        ("blk.0.ffn_gate_exps.weight", (4, 8, 2), 0, 256),
        ("blk.0.ffn_down_exps.weight", (8, 4, 2), 0, 512),
    ]
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value_type, value in metadata:
        payload += _gguf_string(key) + struct.pack("<I", value_type)
        payload += _gguf_string(str(value)) if value_type == 8 else struct.pack("<I", value)
    for name, shape, value_type, offset in tensors:
        payload += _gguf_string(name) + struct.pack("<I", len(shape))
        payload += b"".join(struct.pack("<Q", item) for item in shape)
        payload += struct.pack("<IQ", value_type, offset)
    payload += b"\0" * ((32 - len(payload) % 32) % 32)
    payload += bytes(768)
    path.write_bytes(payload)


def test_gguf_inventory_and_expert_tiles_use_logical_projection_slices(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _write_minimal_moe_gguf(path)
    inventory = inspect_gguf(path)
    assert inventory.metadata["fixturemoe.expert_count"] == 2
    assert inventory.tensor_bytes == 768
    tiles = build_tensor_tiles(
        inventory, model_id="fixture", model_revision="rev", hash_contents=True
    )
    assert {tile.tensor_role for tile in tiles} == {
        "routed_expert_up_projection",
        "routed_expert_gate_projection",
        "routed_expert_down_projection",
    }
    assert {tile.current_residency for tile in tiles} == {"UNPLANNED"}
    shards = build_expert_microshards(
        inventory, model_id="fixture", model_revision="rev", hash_contents=True
    )
    assert len(shards) == 2
    assert shards[1].up.byte_size == 128
    assert shards[1].up.logical_slice["expert_start"] == 1
    assert shards[1].down.logical_slice["projection_range"] == {
        "hidden_start": 0,
        "hidden_end": 8,
    }


def test_shared_expert_tensor_is_not_misclassified_as_router(tmp_path: Path) -> None:
    assert tensor_role("blk.0.ffn_gate_inp_shexp.weight") == "shared_expert"
    path = tmp_path / "fixture.gguf"
    _write_minimal_moe_gguf(path)
    inventory = inspect_gguf(path)
    preflight = build_preflight(
        inventory,
        model_id="fixture",
        model_revision="rev",
        configured_architecture="fixturemoe",
        configured_quantization="none",
        system_ram_available_bytes=10_000,
        physical_vram_bytes=100,
        backend="fixture",
        backend_limitations=[],
    )
    assert preflight.shared_expert_count == 1


def test_cost_model_uses_critical_path_for_overlap() -> None:
    completion, path = critical_path(
        [
            CriticalTask("gpu", 8.0),
            CriticalTask("cpu", 6.0),
            CriticalTask("reduce", 2.0, ("gpu", "cpu")),
        ]
    )
    assert completion == pytest.approx(10.0)
    assert path == ["gpu", "reduce"]


def test_prediction_quality_uses_only_informative_pairs_within_a_workload() -> None:
    quality = prediction_quality(
        [
            {
                "plan_id": "decode-a",
                "comparison_group": "decode",
                "predicted_ms": 10.0,
                "measured_ms": 12.0,
            },
            {
                "plan_id": "decode-b",
                "comparison_group": "decode",
                "predicted_ms": 20.0,
                "measured_ms": 25.0,
            },
            {
                "plan_id": "prefill-a",
                "comparison_group": "prefill_32k",
                "predicted_ms": 10.0,
                "measured_ms": 1000.0,
            },
        ]
    )
    assert quality["informative_pair_count"] == 1
    assert quality["ranking_agreement_fraction"] == 1.0

    tied = prediction_quality(
        [
            {
                "plan_id": "a",
                "comparison_group": "decode",
                "predicted_ms": 1.0,
                "measured_ms": 1.0,
            },
            {
                "plan_id": "b",
                "comparison_group": "decode",
                "predicted_ms": 1.0,
                "measured_ms": 2.0,
            },
        ]
    )
    assert tied["informative_pair_count"] == 0
    assert tied["ranking_agreement_fraction"] is None


def _estimate(
    plan_id: str,
    *,
    decode: float = 10.0,
    ttft: float = 100.0,
    mixed: float = 10.0,
    p95: float = 100.0,
    vram: int = 100,
    unsupported: list[str] | None = None,
) -> CandidateEstimate:
    return CandidateEstimate(
        plan_id=plan_id,
        objective=UtilityObjective.MAXIMUM_DECODE_THROUGHPUT,
        decode_tokens_per_second=decode,
        time_to_first_token_ms=ttft,
        mixed_verified_tokens_per_second=mixed,
        interactive_p95_ms=p95,
        peak_vram_bytes=vram,
        peak_ram_bytes=100,
        pcie_bytes=10,
        cpu_utilisation_percent=10,
        gpu_utilisation_percent=80,
        unsupported_techniques=unsupported or [],
    )


def test_positive_utility_planner_can_reject_harmful_or_unsupported_techniques() -> None:
    candidates = [
        _estimate("stock"),
        _estimate("faster", decode=12.0),
        _estimate("latency-harm", decode=20.0, p95=120.0),
        _estimate("unsupported", decode=30.0, unsupported=["dynamic_expert_cache"]),
    ]
    selected = select_positive_utility(
        candidates,
        baseline_plan_id="stock",
        objective=UtilityObjective.MAXIMUM_DECODE_THROUGHPUT,
    )
    assert selected.selected_plan_id == "faster"
    rejected = {row["plan_id"]: row["rejection_reason"] for row in selected.ranking}
    assert rejected["latency-harm"] == "interactive p95 constraint exceeded"
    assert rejected["unsupported"] == "candidate requests unsupported techniques"
    assert planner_regret_fraction({"stock": 0.0, "faster": 0.2}, "faster") == 0.0


def test_lower_latency_utility_is_reported_as_fractional_reduction() -> None:
    selected = select_positive_utility(
        [_estimate("stock", ttft=100.0), _estimate("faster", ttft=75.0)],
        baseline_plan_id="stock",
        objective=UtilityObjective.MINIMUM_TIME_TO_FIRST_TOKEN,
    )
    assert selected.selected_plan_id == "faster"
    assert selected.selected_utility == pytest.approx(0.25)


def test_activation_classes_use_measured_latency_and_gpu_budget() -> None:
    events = [
        ExpertActivation(token_index=0, layer_id=0, phase="prefill", expert_ids=[0, 1]),
        ExpertActivation(token_index=1, layer_id=0, phase="decode", expert_ids=[0, 2]),
        ExpertActivation(token_index=2, layer_id=0, phase="decode", expert_ids=[0, 1]),
    ]
    measurements = [
        ExpertMeasurement(
            layer_id=0,
            expert_id=expert,
            weight_bytes=100,
            cpu_latency_ms=4.0 if expert != 2 else 1.0,
            gpu_latency_ms=1.0,
            transfer_latency_ms=2.0,
        )
        for expert in range(3)
    ]
    stats, coactivation = activation_statistics(events, measurements)
    assert next(row for row in stats if row.expert_id == 0).activation_probability == 1.0
    assert next(row for row in stats if row.expert_id == 0).consecutive_token_reuse == 2
    assert coactivation
    decisions = classify_residency(stats, gpu_budget_bytes=100)
    by_expert = {row.expert_id: row for row in decisions}
    assert by_expert[0].residency_class == "hot"
    assert by_expert[2].policy == "cpu_execute"


def test_bounded_predictor_and_prefetch_accounting_prioritise_visible_latency() -> None:
    predictor = BoundedExpertPredictor(window=4, maximum_predictions=2)
    predictor.observe(
        ExpertActivation(token_index=0, layer_id=0, phase="decode", expert_ids=[3, 5])
    )
    assert predictor.predict(token_index=1, layer_id=0, mode="previous_token_reuse") == [3, 5]
    result = evaluate_prediction(
        token_index=1,
        layer_id=0,
        predictor="previous_token_reuse",
        predicted=[3, 5],
        actual=[3, 7],
        bytes_by_expert={3: 100, 5: 100, 7: 100},
        transfer_ms_by_expert={3: 3.0, 5: 3.0, 7: 3.0},
        overlap_available_ms=2.0,
        measured_interference_ms=0.5,
    )
    assert result.true_positives == result.false_positives == result.false_negatives == 1
    assert result.useful_bytes == result.wasted_bytes == 100
    assert result.visible_transfer_latency_removed_ms == pytest.approx(1.5)


def test_expert_cache_is_byte_bounded_and_does_not_evict_pinned_hot_entries() -> None:
    cache = ExpertLRUCache(200, pinned=[(0, 0)])
    cache.access((0, 0), 100)
    cache.access((0, 1), 100)
    result = cache.access((0, 2), 100)
    assert result.evicted == ((0, 1),)
    assert (0, 0) in cache.entries and (0, 2) in cache.entries
    assert cache.used_bytes == 200


def test_profile_fingerprint_is_canonical_and_latency_percentiles_are_interpolated() -> None:
    assert profile_fingerprint({"b": 2, "a": 1}) == profile_fingerprint({"a": 1, "b": 2})
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)
    summary = latency_summary([3.0, 1.0, 2.0])
    assert summary["median_ms"] == 2.0
    assert summary["sample_count"] == 3


def test_baseline_search_is_bounded_and_selected_per_workload(repository_root: Path) -> None:
    config = load_experiment_008_config(
        repository_root / "configs" / "experiments" / "experiment_008_adaptive_moe.yaml"
    )
    candidates = baseline_search_space(config.baseline_search, seed=8008)
    assert len(candidates) == config.baseline_search.maximum_candidates
    assert all(candidate.microbatch_size <= candidate.batch_size for candidate in candidates)
    assert {candidate.cpu_moe_layers for candidate in candidates} == set(
        config.baseline_search.cpu_moe_layers
    )
    assert any(candidate.cpu_moe_layers == 24 for candidate in candidates)
    selected = select_best_stock_by_workload(
        [
            {
                "candidate_id": "decode-best",
                "status": "COMPLETED",
                "classification": "MEASURED",
                "workload": "decode",
                "decode_tokens_per_second": 10.0,
            },
            {
                "candidate_id": "prefill-best",
                "status": "COMPLETED",
                "classification": "MEASURED",
                "workload": "prefill_32k",
                "time_to_first_token_ms": 50.0,
            },
        ]
    )
    assert selected["decode"]["candidate_id"] == "decode-best"  # type: ignore[index]
    assert selected["prefill_32k"]["candidate_id"] == "prefill-best"  # type: ignore[index]


def test_full_plan_enables_only_positive_supported_techniques() -> None:
    capabilities = BackendCapabilities(
        conventional_layer_offload=True,
        tensor_buffer_override=True,
        cpu_moe=True,
        asynchronous_backend_scheduler=True,
        operation_level_overlap_trace=True,
        expert_routing_trace=False,
        per_expert_dynamic_residency=False,
        expert_prefetch=False,
        separate_process_phase_plans=True,
        in_request_phase_switch=False,
        deterministic_greedy_tokens=True,
        final_logits=False,
    )
    plan = build_phase_plan(
        configuration="G",
        phase="decode",
        capabilities=capabilities,
        tiles=[_tile("up")],
        stock_arguments=["--n-gpu-layers", "auto"],
        cpu_moe_layers=24,
        measured_utility_by_technique={
            "tensor_granular_placement": 0.1,
            "asymmetric_cpu_gpu_partition": 0.2,
            "asynchronous_cpu_gpu_overlap": -0.1,
            "separate_prefill_decode_plans": 0.05,
        },
    )
    decisions = {item.technique: item for item in plan.techniques}
    assert decisions["tensor_granular_placement"].enabled
    assert decisions["asymmetric_cpu_gpu_partition"].enabled
    assert not decisions["asynchronous_cpu_gpu_overlap"].enabled
    assert (
        decisions["activation_aware_expert_cache"].execution_status == ExecutionStatus.UNSUPPORTED
    )
    assert "--n-cpu-moe" in plan.backend_arguments
    assert "--override-tensor" in plan.backend_arguments
    override_index = plan.backend_arguments.index("--override-tensor")
    assert "ffn_gate_inp" in plan.backend_arguments[override_index + 1]


def test_stock_and_tensor_aware_plans_do_not_relabel_backend_managed_tensors() -> None:
    capabilities = BackendCapabilities(
        conventional_layer_offload=True,
        tensor_buffer_override=True,
        cpu_moe=True,
        asynchronous_backend_scheduler=False,
        operation_level_overlap_trace=False,
        expert_routing_trace=False,
        per_expert_dynamic_residency=False,
        expert_prefetch=False,
        separate_process_phase_plans=True,
        in_request_phase_switch=False,
        deterministic_greedy_tokens=True,
        final_logits=False,
    )
    expert = _tile("up")
    router = expert.model_copy(
        update={
            "tensor_name": "blk.2.ffn_gate_inp.weight",
            "tensor_role": "router",
            "expert_id": None,
        }
    )
    normalisation = expert.model_copy(
        update={
            "tensor_name": "blk.2.attn_norm.weight",
            "tensor_role": "normalisation",
            "expert_id": None,
        }
    )
    stock = build_phase_plan(
        configuration="A",
        phase="decode",
        capabilities=capabilities,
        tiles=[expert, router, normalisation],
        stock_arguments=["--n-gpu-layers", "auto"],
        cpu_moe_layers=24,
    )
    tensor_aware = build_phase_plan(
        configuration="B",
        phase="decode",
        capabilities=capabilities,
        tiles=[expert, router, normalisation],
        stock_arguments=["--n-gpu-layers", "auto"],
        cpu_moe_layers=24,
    )
    assert {item.residency for item in stock.placements} == {"BACKEND_MANAGED"}
    by_role = {item.tensor_role: item for item in tensor_aware.placements}
    assert by_role["router"].residency == "GPU"
    assert by_role["normalisation"].residency == "GPU"
    assert by_role["routed_expert_up_projection"].residency == "CPU"
    assert "--override-tensor" in tensor_aware.backend_arguments


def test_cpu_moe_utility_requires_an_otherwise_matched_zero_share_measurement() -> None:
    selected = {
        "status": "COMPLETED",
        "workload": "decode",
        "gpu_layers": "auto",
        "cpu_threads": 20,
        "batch_size": 2048,
        "microbatch_size": 512,
        "memory_map": True,
        "flash_attention": True,
        "cpu_moe_layers": 24,
        "decode_tokens_per_second": 12.0,
    }
    zero = {
        **selected,
        "cpu_moe_layers": 0,
        "decode_tokens_per_second": 10.0,
    }
    assert _matched_cpu_moe_utility([zero, selected], selected, workload="decode") == pytest.approx(
        0.2
    )
    unmatched = {**zero, "batch_size": 512}
    assert _matched_cpu_moe_utility([unmatched, selected], selected, workload="decode") is None


def test_model_buffer_parser_counts_cuda_host_as_a_model_buffer() -> None:
    match = _BUFFER.search("CUDA_Host model buffer size = 16270.91 MiB")
    assert match is not None
    assert match.group("device") == "CUDA_Host"


def test_tiny_moe_fixture_covers_required_tensor_execution_equivalences() -> None:
    result = validate_tiny_moe_fixture()
    required = {
        "tensor_tile_reconstruction",
        "expert_microshard_equivalence",
        "cache_hit_and_miss_equivalence",
        "prefetch_enabled_disabled_equivalence",
        "separate_prefill_decode_plan_equivalence",
    }
    assert result["passed"] is True
    assert required <= result["checks"].keys()
    assert all(result["checks"][name]["allclose"] for name in required)
