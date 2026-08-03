"""Experiment 009 Colibri dependency, adapter, telemetry, and ABI tests."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import psutil
import pytest
from pydantic import ValidationError

from swarm_inference.backends.colibri.constants import (
    BRIDGE_EVENT_TYPES,
    COLIBRI_COMMIT,
    COLIBRI_LICENSE,
    COLIBRI_RELEASE,
)
from swarm_inference.backends.colibri.dependency import (
    binary_fingerprint,
    load_build_manifest,
    patch_manifest,
    verify_colibri_checkout,
)
from swarm_inference.backends.colibri.model import ColibriModelInspector
from swarm_inference.backends.colibri.placement import (
    PromptPartition,
    calibration_hot_pin_bitmap,
    validate_prompt_partitions,
)
from swarm_inference.backends.colibri.plan import ColibriPlanTranslator
from swarm_inference.backends.colibri.probe import ColibriCapabilityProbe
from swarm_inference.backends.colibri.process import ColibriProcess
from swarm_inference.backends.colibri.replay import (
    ColibriBenchmarkRunner,
    ColibriFixedReplayTuner,
    ColibriReplayRunner,
    ReplayTokenSequence,
    TuningCandidate,
)
from swarm_inference.backends.colibri.schemas import (
    BridgeEvent,
    ColibriGenerationResult,
    ColibriMode,
    NativeQuantizationMetadata,
    RouteSelection,
    TelemetryLevel,
    TuningSample,
)
from swarm_inference.backends.colibri.telemetry import (
    ColibriRouteTraceReader,
    ColibriTelemetryReader,
    ColibriUsageHistoryReader,
)
from swarm_inference.experiments.experiment_009.bundle import REQUIRED_FILES, EvidenceBundle
from swarm_inference.experiments.experiment_009.runner import (
    Experiment009Options,
    run_experiment_009,
)
from swarm_inference.experiments.experiment_009.schemas import Experiment009Verdict
from swarm_inference.microsharding.expert_abi import (
    ExpertMicroshardDescriptor,
    ExpertProjectionSlice,
    descriptor_content_hash,
    executable_microshard_equivalence,
    validate_expert_microshard_set,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    """Write a minimal spec-valid safetensors file while preserving input order."""

    header: dict[str, dict[str, Any]] = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _make_model(
    root: Path,
    *,
    family: str = "glm-5.2",
    engine_directory: Path | None = None,
) -> tuple[Path, Path]:
    model = root / "model"
    engines = engine_directory or root / "bin"
    model.mkdir(parents=True)
    engines.mkdir(parents=True, exist_ok=True)
    if family == "glm-5.2":
        config: dict[str, Any] = {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 1,
        }
        engine = "colibri.exe"
    elif family == "olmoe":
        config = {
            "model_type": "olmoe",
            "num_hidden_layers": 1,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
        }
        engine = "olmoe.exe"
    elif family == "kimi-k3":
        config = {
            "model_type": "kimi_k3",
            "text_config": {
                "model_type": "kimi_k3",
                "hidden_size": 128,
                "num_hidden_layers": 1,
                "num_experts": 8,
                "num_experts_per_token": 2,
                "num_shared_experts": 2,
                "routed_expert_hidden_size": 64,
                "moe_intermediate_size": 32,
                "quantization_config": {"format": "mxfp4-pack-quantized"},
            },
        }
        engine = "kimi_k3.exe"
    else:  # pragma: no cover - helpers call only explicit supported families
        raise ValueError(family)
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (engines / engine).write_bytes(b"fixture-engine")
    return model, engines


def _bridge_generation(request_id: str, *, latency_ms: float = 10.0) -> ColibriGenerationResult:
    return ColibriGenerationResult(
        request_id=request_id,
        text="AB",
        input_token_ids=[11, 12],
        output_token_ids=[21, 22],
        token_identity_observed=True,
        stop_reason="length",
        prompt_tokens=2,
        completion_tokens=2,
        elapsed_ms=latency_ms,
        time_to_first_token_ms=3.0,
        decode_tokens_per_second=20.0,
    )


def _native_plan() -> dict[str, Any]:
    return {
        "version": 2,
        "policy": {"name": "auto", "quality_preserving": True},
        "model": {
            "model_id": "fixture",
            "dense_bytes": 100,
            "expert_bytes": 100,
            "shared_expert_bytes": 10,
            "expert_count": 2,
        },
        "tiers": {
            "disk": {"cold_expert_bytes": 20, "model_bytes": 200, "available_bytes": 1000},
            "ram": {
                "warm_expert_bytes": 80,
                "budget_bytes": 300,
                "runtime_bytes": 20,
                "expert_cache_bytes": 80,
                "cache_slots_per_layer": 1,
            },
            "vram": {"hot_expert_bytes": 0, "budget_bytes": 0, "expert_capacity": 0},
        },
        "cpu": {"physical_cores": 4, "sockets": 1, "thread_policy": "physical-cores"},
        "tune": {
            "OMP_NUM_THREADS": {"value": 4},
            "PIPE": {"value": 1},
            "DIRECT": {"value": 0},
        },
        "expected_bottleneck": "storage",
    }


def _mxfp4() -> NativeQuantizationMetadata:
    return NativeQuantizationMetadata(
        format_name="mxfp4",
        packing="e2m1_two_nibbles",
        scale_format="ue8m0",
        scale_group_size=32,
        quantization_aware_trained=True,
        reencoding_allowed=False,
        backend_requirements=["colibri-kimi-k3-native-mxfp4"],
        logical_shape=[64, 8],
        packed_shape=[64, 4],
        byte_size=256,
    )


def _projection(
    projection: str,
    start: int,
    end: int,
) -> ExpertProjectionSlice:
    is_down = projection == "down"
    shape = [8, 64] if is_down else [64, 8]
    return ExpertProjectionSlice(
        tensor_id=f"tensor-{projection}",
        tensor_name=f"{projection}.weight",
        projection=projection,  # type: ignore[arg-type]
        logical_axis=1 if is_down else 0,
        slice_start=start,
        slice_end=end,
        logical_shape=shape,
        storage_file="fixture.safetensors",
        storage_offset=start,
        storage_length=max(1, end - start),
        storage_file_size=1024,
        content_hash=f"sha256:{projection}",
    )


def _descriptor(start: int, end: int) -> ExpertMicroshardDescriptor:
    return ExpertMicroshardDescriptor(
        model_id="fixture",
        layer_id=0,
        expert_id=1,
        shard_id=f"{start}-{end}",
        hidden_start=start,
        hidden_end=end,
        up_projection=_projection("up", start, end),
        gate_projection=_projection("gate", start, end),
        down_projection=_projection("down", start, end),
        native_quantization=_mxfp4(),
        required_accumulator="fp32_sum",
        supported_backends=[],
        execution_status="unsupported",
    )


def test_colibri_commit_pin() -> None:
    dependency = json.loads(
        (REPOSITORY_ROOT / "integrations" / "colibri" / "dependency.json").read_text(
            encoding="utf-8"
        )
    )
    checkout = verify_colibri_checkout(REPOSITORY_ROOT / "third_party" / "colibri")
    patches = patch_manifest(REPOSITORY_ROOT / "integrations" / "colibri")
    assert dependency["commit"] == COLIBRI_COMMIT == checkout["commit"]
    assert dependency["release"] == COLIBRI_RELEASE
    assert dependency["license"] == COLIBRI_LICENSE
    assert patches["upstream_commit"] == COLIBRI_COMMIT
    assert [item["name"] for item in patches["patches"]] == [
        "0001-swarm-bridge.patch",
        "0002-olmoe-routing-telemetry.patch",
        "0003-aggregate-runtime-telemetry.patch",
        "0004-olmoe-machine-readable-telemetry.patch",
        "0005-olmoe-shared-expert-runtime.patch",
        "0006-olmoe-external-expert-dispatch.patch",
        "0007-olmoe-native-microshards.patch",
        "0008-olmoe-memory-residency-telemetry.patch",
    ]
    bridge_patch = (
        REPOSITORY_ROOT
        / "integrations"
        / "colibri"
        / "patches"
        / "0004-olmoe-machine-readable-telemetry.patch"
    ).read_text(encoding="utf-8")
    assert "COLI_USAGE_PATH" in bridge_patch
    assert "COLI_HOT_PIN_PATH" in bridge_patch


def test_colibri_memory_residency_patch_contract() -> None:
    patch = (
        REPOSITORY_ROOT
        / "integrations"
        / "colibri"
        / "patches"
        / "0008-olmoe-memory-residency-telemetry.patch"
    ).read_text(encoding="utf-8")
    for required in (
        "QueryWorkingSetEx",
        "GetProcessMemoryInfo",
        "logical_cache_hits",
        "resident_cache_hits",
        "nonresident_cache_hits",
        "coordinator_owned_routed_expert_bytes",
        "capacity-isolated coordinator",
        "test_olmoe_memory_residency",
        "COLI_SWARM_EXPERT_CUDA_TARGET",
        "CPU fallback is forbidden",
        "cuda_resident_tensor_bytes",
        "coli_cuda_expert_mlp",
    ):
        assert required in patch


def test_colibri_license_present() -> None:
    license_path = REPOSITORY_ROOT / "third_party" / "colibri" / "LICENSE"
    text = license_path.read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text
    assert "LICENSE.colibri" in (
        REPOSITORY_ROOT / "integrations" / "colibri" / "build.ps1"
    ).read_text(encoding="utf-8")


def test_colibri_build_fingerprint(tmp_path: Path) -> None:
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    forward = binary_fingerprint([first, second])
    assert forward == binary_fingerprint([second, first])
    second.write_bytes(b"changed")
    assert forward != binary_fingerprint([first, second])
    manifest = tmp_path / "colibri_build.json"
    manifest.write_text(json.dumps({"commit": COLIBRI_COMMIT}), encoding="utf-8")
    assert load_build_manifest(manifest)["commit"] == COLIBRI_COMMIT
    built = REPOSITORY_ROOT / "build" / "colibri" / "colibri_build.json"
    if built.is_file():
        payload = load_build_manifest(built)
        assert payload["source_tree_sha256"]
        assert payload["compiler"]
        required_binaries = {
            "colibri.exe",
            "olmoe.exe",
            "swarm_bridge.py",
        }
        if "0005-olmoe-shared-expert-runtime.patch" in payload.get("patches", []):
            required_binaries.add("olmoe_expert_worker.exe")
        assert {item["name"] for item in payload["binaries"]} >= required_binaries


def test_colibri_capability_handshake(tmp_path: Path) -> None:
    for binary in ("colibri.exe", "olmoe.exe", "inkling.exe", "kimi_k3.exe"):
        (tmp_path / binary).write_bytes(b"fixture")
    (tmp_path / "swarm_bridge.py").write_text("# fixture\n", encoding="utf-8")
    report = ColibriCapabilityProbe(tmp_path).probe()
    assert report.backend == "colibri"
    assert report.colibri_commit == COLIBRI_COMMIT
    assert set(report.model_families) == {"glm-5.2", "olmoe", "inkling", "kimi-k3"}
    assert report.supports_cpu
    assert report.supports_native_mxfp4
    assert report.supports_tensor_microshards is False
    assert report.execution_backends == ["cpu"]


def test_colibri_process_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, engines = _make_model(tmp_path)
    (engines / "openai_server.py").write_text("# fixture\n", encoding="utf-8")
    (engines / "swarm_bridge.py").write_text("# fixture\n", encoding="utf-8")

    class FakeProcess:
        pid = 424242

        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout: float) -> int:
            assert timeout > 0
            self.returncode = 0
            return 0

    fake = FakeProcess()
    monkeypatch.setattr("swarm_inference.backends.colibri.process._is_listening", lambda *_: False)
    monkeypatch.setattr(
        "swarm_inference.backends.colibri.process.get_json",
        lambda *_: {"status": "ok"},
    )
    monkeypatch.setattr(
        "swarm_inference.backends.colibri.process.subprocess.Popen",
        lambda *args, **kwargs: fake,
    )
    process = ColibriProcess(
        engine_directory=engines,
        model_path=model,
        model_id="fixture",
        model_revision="r1",
        ram_safety_reserve_bytes=0,
    )
    monkeypatch.setenv("TOPP", "0.123")
    monkeypatch.setenv("AUTOPIN", "99")
    controlled_environment = process._environment()
    assert "TOPP" not in controlled_environment
    assert "AUTOPIN" not in controlled_environment
    process.start(timeout_seconds=1)
    assert process.running
    assert process.health() == {"status": "ok"}
    assert process.pid_file.is_file()
    assert process._environment()["COLI_SWARM_BRIDGE"] == "1"
    assert process._environment()["PROF"] == "1"
    process.shutdown()
    assert not process.running
    assert not process.pid_file.exists()


def test_colibri_stream_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model, engines = _make_model(tmp_path)
    (engines / "openai_server.py").write_text("# fixture\n", encoding="utf-8")
    (engines / "swarm_bridge.py").write_text("# fixture\n", encoding="utf-8")
    process = ColibriProcess(
        engine_directory=engines,
        model_path=model,
        model_id="fixture",
        model_revision="r1",
        ram_safety_reserve_bytes=0,
    )
    process.process = SimpleNamespace(poll=lambda: None)
    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {
            "choices": [{"text": "B", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            "colibri": {"input_token_ids": [11, 12], "token_ids": [21, 22]},
        },
    ]

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def __iter__(self):  # type: ignore[no-untyped-def]
            for event in events:
                yield f"data: {json.dumps(event)}\n".encode()
            yield b"data: [DONE]\n"

    monkeypatch.setattr(
        "swarm_inference.backends.colibri.process.urlopen", lambda *_args, **_kwargs: FakeResponse()
    )
    chunks: list[str] = []
    result = process.stream_generate(
        prompt="fixture", max_tokens=2, request_id="stream-1", on_text=chunks.append
    )
    assert result.text == "AB"
    assert chunks == ["A", "B"]
    assert result.input_token_ids == [11, 12]
    assert result.output_token_ids == [21, 22]
    assert result.stop_reason == "length"
    assert not process.cancel("stream-1")


def test_colibri_bridge_event_schema(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.ndjson"
    events = [
        BridgeEvent(
            event_type=event_type,
            timestamp_ns=index,
            engine_pid=7,
            request_id="request-1" if "request" in event_type else None,
            model_id="fixture",
            model_revision="r1",
            engine_family="glm-5.2",
            sequence_number=index,
            payload={},
        )
        for index, event_type in enumerate(sorted(BRIDGE_EVENT_TYPES))
    ]
    path.write_text("".join(event.model_dump_json() + "\n" for event in events), encoding="utf-8")
    parsed = ColibriTelemetryReader(path).read()
    assert {event.event_type for event in parsed} == BRIDGE_EVENT_TYPES
    with pytest.raises(ValidationError):
        BridgeEvent(
            event_type="invented",
            timestamp_ns=0,
            engine_pid=1,
            model_id="fixture",
            model_revision="r1",
            engine_family="glm-5.2",
            sequence_number=0,
        )


def test_colibri_model_inventory(tmp_path: Path) -> None:
    model, engines = _make_model(tmp_path, family="olmoe")
    _write_safetensors(
        model / "model.safetensors",
        [
            ("model.embed_tokens.weight", "F32", [2, 2], b"\0" * 16),
            ("model.layers.0.mlp.experts.7.merged_weight", "U8", [3, 4], b"\1" * 12),
            ("model.layers.0.mlp.experts.2.merged_weight", "U8", [3, 4], b"\2" * 12),
        ],
    )
    inventory, tensors, experts, formats = ColibriModelInspector(engines).inspect(model)
    assert inventory.model_family == "olmoe"
    assert inventory.tensor_count == len(tensors) == 3
    assert inventory.expert_count == len(experts) == 2
    assert inventory.expert_geometry["shared_experts"] == 1
    assert {item.format_name for item in formats} >= {"f32", "int8_rowwise"}


def test_colibri_tensor_inventory(tmp_path: Path) -> None:
    model, engines = _make_model(tmp_path, family="olmoe")
    file_path = model / "model.safetensors"
    _write_safetensors(
        file_path,
        [("model.layers.0.mlp.experts.3.merged_weight", "U8", [3, 4], b"x" * 12)],
    )
    _, tensors, _, _ = ColibriModelInspector(engines).inspect(model, content_hash_mode="full")
    tensor = tensors[0]
    assert tensor.layer_id == 0 and tensor.expert_id == 3
    assert tensor.tensor_role == "routed_expert_merged_weight"
    assert tensor.storage_offset >= 8
    assert tensor.storage_offset + tensor.storage_length <= file_path.stat().st_size
    assert tensor.content_hash.startswith("sha256:")


def test_colibri_expert_inventory(tmp_path: Path) -> None:
    model, engines = _make_model(tmp_path, family="olmoe")
    _write_safetensors(
        model / "model.safetensors",
        [
            ("model.layers.0.mlp.experts.7.merged_weight", "U8", [2, 2], b"7" * 4),
            ("model.layers.0.mlp.experts.2.merged_weight", "U8", [2, 2], b"2" * 4),
        ],
    )
    _, _, experts, _ = ColibriModelInspector(engines).inspect(model)
    assert [expert.expert_id for expert in experts] == [7, 2]
    assert [expert.physical_storage_order for expert in experts] == [0, 1]
    assert all(expert.total_bytes == 4 for expert in experts)
    assert all(expert.native_format == "int8_rowwise" for expert in experts)


def test_colibri_native_mxfp4_metadata(tmp_path: Path) -> None:
    model, engines = _make_model(tmp_path, family="kimi-k3")
    tensors: list[tuple[str, str, list[int], bytes]] = []
    for matrix, packed_shape, scale_shape in (
        ("w1", [32, 32], [32, 2]),
        ("w2", [64, 16], [64, 1]),
        ("w3", [32, 32], [32, 2]),
    ):
        prefix = f"model.layers.0.block_sparse_moe.experts.5.{matrix}"
        tensors.append(
            (f"{prefix}.weight_packed", "U8", packed_shape, b"p" * int(np.prod(packed_shape)))
        )
        tensors.append(
            (f"{prefix}.weight_scale", "U8", scale_shape, b"s" * int(np.prod(scale_shape)))
        )
    _write_safetensors(model / "model.safetensors", tensors)
    inventory, imported, experts, _ = ColibriModelInspector(engines).inspect(model)
    packed = [item for item in imported if item.tensor_name.endswith("weight_packed")]
    scales = [item for item in imported if item.tensor_name.endswith("weight_scale")]
    assert inventory.expert_geometry["shared_experts"] == 2
    assert inventory.expert_geometry["routed_expert_hidden_size"] == 64
    assert inventory.expert_geometry["expert_intermediate_size"] == 32
    assert {item.tensor_role for item in packed} == {
        "routed_expert_gate_projection",
        "routed_expert_up_projection",
        "routed_expert_down_projection",
    }
    assert all(item.quantization.format_name == "mxfp4" for item in packed)
    assert all(item.quantization.packing == "e2m1_two_nibbles" for item in packed)
    assert all(item.quantization.scale_group_size == 32 for item in packed)
    assert all(item.quantization.reencoding_allowed is False for item in packed)
    assert all(item.quantization.format_name == "ue8m0" for item in scales)
    assert experts[0].native_format == "mxfp4"
    logical = {item.tensor_role: item.logical_shape for item in packed}
    assert logical["routed_expert_gate_projection"] == [32, 64]
    assert logical["routed_expert_up_projection"] == [32, 64]
    assert logical["routed_expert_down_projection"] == [64, 32]


def test_colibri_route_trace_reader(tmp_path: Path) -> None:
    path = tmp_path / "route.trace"
    path.write_text("0 0 0 3:0.75 1:0.25\n1 0 1 2:1.0\n", encoding="utf-8")
    rows = ColibriRouteTraceReader().read(
        path,
        phase_by_call={0: "prefill", 1: "decode"},
        request_by_call={0: "a", 1: "a"},
        tier_by_expert={(0, 3): "ram", (0, 1): "nvme", (1, 2): "vram"},
    )
    summary = ColibriRouteTraceReader.summarize(rows)
    assert len(rows) == 3
    assert rows[0].routing_weight == 0.75
    assert rows[0].execution_tier == "ram"
    assert summary["selection_count"] == 3
    assert summary["phase_selection_counts"] == {"decode": 1, "prefill": 2}


def test_colibri_usage_history_reader(tmp_path: Path) -> None:
    path = tmp_path / ".coli_usage"
    path.write_text("-1 2 4\n0 1 7\n1 3 5\n", encoding="utf-8")
    result = ColibriUsageHistoryReader().read(path, expected_layers=2, expected_experts=4)
    assert result["format"] == "sparse_text"
    assert result["total_activations"] == 12
    assert result["records"][0] == {"layer_id": 0, "expert_id": 1, "activation_count": 7}


def test_colibri_plan_translation() -> None:
    translator = ColibriPlanTranslator()
    plan = translator.translate(_native_plan(), hardware_fingerprint="hardware-sha256")
    assert plan.backend == "colibri"
    assert plan.ram_budget_bytes == 300
    assert plan.routed_expert_tiers["nvme"]["bytes"] == 20
    assert plan.routed_expert_tiers["ram"]["cache_slots_per_layer"] == 1
    assert plan.routed_expert_tiers["ram"]["cache_bytes"] == 80
    oversized = _native_plan()
    oversized["tiers"]["ram"]["cache_slots_per_layer"] = 99
    reconciled = translator.translate(oversized, hardware_fingerprint="hardware-sha256")
    assert reconciled.routed_expert_tiers["ram"]["cache_slots_per_layer"] == 2
    assert reconciled.routed_expert_tiers["ram"]["planner_cache_capacity_slots_per_layer"] == 99
    assert reconciled.routed_expert_tiers["ram"]["capacity_reconciled_to_inventory"] is True
    assert plan.semantics_preserved
    adjusted = translator.bounded_adjustment(
        plan,
        {"OMP_NUM_THREADS": 2, "PIPE": 0},
        supported_settings={"OMP_NUM_THREADS", "PIPE"},
    )
    assert adjusted.source == "swarm_bounded_adjustment"
    assert ColibriPlanTranslator.environment(adjusted)["OMP_NUM_THREADS"] == "2"
    with pytest.raises(ValueError, match="quality-affecting"):
        translator.bounded_adjustment(plan, {"TOPP": 0.5}, supported_settings={"TOPP"})


def test_colibri_fixed_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = ReplayTokenSequence(
        model_id="fixture",
        model_revision="r1",
        tokenizer_hash="tokenizer",
        prompt_ids=[1, 2],
        continuation_ids=[3, 4],
    )
    assert replay.full_ids == [1, 2, 3, 4]
    assert len(replay.sequence_hash) == 64
    model, engines = _make_model(tmp_path)
    monkeypatch.setenv("TOPP", "0.123")
    monkeypatch.setenv("PILOT", "3")
    runner = ColibriReplayRunner(
        engine_directory=engines,
        model_path=model,
        model_id="fixture",
        model_revision="r1",
        environment={"COLI_SWARM_BRIDGE": "1"},
        ram_safety_reserve_bytes=0,
    )
    controlled_environment = runner._environment(None)
    assert "TOPP" not in controlled_environment
    assert "PILOT" not in controlled_environment
    assert controlled_environment["COLI_SWARM_BRIDGE"] == "1"
    with pytest.raises(ValidationError, match="hash"):
        ReplayTokenSequence(
            model_id="fixture",
            model_revision="r1",
            tokenizer_hash="tokenizer",
            prompt_ids=[1],
            continuation_ids=[2],
            sequence_hash="incorrect",
        )


def test_colibri_adapter_token_identity() -> None:
    class DirectProcess:
        @staticmethod
        def generate(**kwargs: Any) -> ColibriGenerationResult:
            return _bridge_generation(kwargs["request_id"])

    rows = ColibriBenchmarkRunner().compare(
        process=DirectProcess(),  # type: ignore[arg-type]
        prompt="fixture",
        max_tokens=2,
        repeats=3,
        adapter_call=_bridge_generation,
    )
    assert len(rows) == 6
    assert all(row["token_identity_match"] for row in rows)
    assert all(row["token_identity_observed"] for row in rows)


def test_colibri_adapter_overhead_fixture() -> None:
    class DirectProcess:
        @staticmethod
        def generate(**kwargs: Any) -> ColibriGenerationResult:
            return _bridge_generation(kwargs["request_id"], latency_ms=10.0)

    rows = ColibriBenchmarkRunner().compare(
        process=DirectProcess(),  # type: ignore[arg-type]
        prompt="fixture",
        max_tokens=2,
        repeats=3,
        adapter_call=lambda request_id: _bridge_generation(request_id, latency_ms=10.2),
    )
    direct = [row["latency_ms"] for row in rows if row["configuration"] == "direct"]
    adapted = [row["latency_ms"] for row in rows if row["configuration"] == "adapter"]
    assert np.median(adapted) / np.median(direct) - 1 < 0.03


def test_colibri_tuning_rejects_noise() -> None:
    replay = ReplayTokenSequence(
        model_id="fixture",
        model_revision="r1",
        tokenizer_hash="tokenizer",
        prompt_ids=[1],
        continuation_ids=[2, 3],
    )

    def measure(candidate: TuningCandidate, repeat: int, order: str) -> TuningSample:
        return TuningSample(
            candidate_id=candidate.candidate_id,
            repeat=repeat,
            order=order,  # type: ignore[arg-type]
            decode_tokens_per_second=102.0 if candidate.candidate_id == "small" else 100.0,
            p95_latency_ms=10.0,
            input_token_ids=replay.prompt_ids,
            output_token_ids=replay.continuation_ids,
            settings_applied=candidate.settings,
        )

    result = ColibriFixedReplayTuner(repeats=3).tune(
        replay=replay,
        candidates=[
            TuningCandidate(candidate_id="baseline"),
            TuningCandidate(candidate_id="small"),
        ],
        measure=measure,  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert result.selected_candidate_id == "baseline"
    assert result.reverse_confirmed
    assert result.reverse_confirmation is not None
    assert result.rejection_reason == "gain_below_minimum_meaningful_threshold"


def test_colibri_tuning_reverse_confirmation() -> None:
    replay = ReplayTokenSequence(
        model_id="fixture",
        model_revision="r1",
        tokenizer_hash="tokenizer",
        prompt_ids=[1],
        continuation_ids=[2, 3],
    )

    def measure(candidate: TuningCandidate, repeat: int, order: str) -> TuningSample:
        speed = (
            100.0
            if candidate.candidate_id == "baseline"
            else (106.0 if order == "forward" else 105.0)
        )
        return TuningSample(
            candidate_id=candidate.candidate_id,
            repeat=repeat,
            order=order,  # type: ignore[arg-type]
            decode_tokens_per_second=speed,
            p95_latency_ms=10.0,
            input_token_ids=replay.prompt_ids,
            output_token_ids=replay.continuation_ids,
            settings_applied=candidate.settings,
        )

    result = ColibriFixedReplayTuner(repeats=3).tune(
        replay=replay,
        candidates=[TuningCandidate(candidate_id="baseline"), TuningCandidate(candidate_id="fast")],
        measure=measure,  # type: ignore[arg-type]
    )
    assert result.accepted and result.reverse_confirmed
    assert result.selected_candidate_id == "fast"
    assert result.confirmed_gain >= 0.03
    assert result.reverse_confirmation is not None


def test_colibri_heldout_policy_split() -> None:
    groups = (
        "general_chat",
        "coding",
        "mathematics_reasoning",
        "multilingual_long_form",
    )
    rows = [
        PromptPartition(prompt_id=f"cal-{group}", workload_group=group, partition="calibration")
        for group in groups
    ] + [
        PromptPartition(prompt_id=f"held-{group}", workload_group=group, partition="heldout")
        for group in groups
    ]
    result = validate_prompt_partitions(rows)
    assert result["calibration_prompts"] == result["heldout_prompts"] == 4
    with pytest.raises(ValueError, match="disjoint"):
        validate_prompt_partitions([*rows, rows[0]])
    routes = [
        RouteSelection(call_index=0, row_index=0, layer_id=0, expert_id=2),
        RouteSelection(call_index=1, row_index=0, layer_id=0, expert_id=2),
        RouteSelection(call_index=2, row_index=0, layer_id=0, expert_id=1),
        RouteSelection(call_index=3, row_index=0, layer_id=1, expert_id=3),
    ]
    bitmap, metadata = calibration_hot_pin_bitmap(
        routes,
        layer_count=2,
        experts_per_layer=4,
        hot_slots_per_layer=1,
    )
    assert bitmap == bytes([0, 0, 1, 0, 0, 0, 0, 1])
    assert metadata["pinned_expert_count"] == 2
    assert metadata["prompt_text_or_labels_used"] is False


def test_microshard_projection_alignment() -> None:
    descriptor = _descriptor(0, 32)
    assert descriptor.up_projection.logical_axis == 0
    assert descriptor.down_projection.logical_axis == 1
    with pytest.raises(ValidationError, match="same hidden range"):
        ExpertMicroshardDescriptor(
            **{
                **descriptor.model_dump(mode="python", exclude={"content_hash"}),
                "gate_projection": _projection("gate", 32, 64),
            }
        )


def test_microshard_quantization_boundaries() -> None:
    with pytest.raises(ValidationError, match="quantization group"):
        _descriptor(16, 64)
    assert _descriptor(32, 64).hidden_start == 32


def test_microshard_reconstruction() -> None:
    first, second = _descriptor(0, 32), _descriptor(32, 64)
    result = validate_expert_microshard_set([second, first])
    assert result["valid"] and result["covered_hidden_units"] == 64
    assert first.content_hash == descriptor_content_hash(first)
    rng = np.random.default_rng(9)
    equivalence = executable_microshard_equivalence(
        inputs=rng.normal(size=(3, 8)).astype(np.float32),
        up=rng.normal(size=(64, 8)).astype(np.float32),
        gate=rng.normal(size=(64, 8)).astype(np.float32),
        down=rng.normal(size=(8, 64)).astype(np.float32),
        ranges=[(0, 32), (32, 64)],
    )
    assert equivalence["allclose"]
    assert equivalence["maximum_absolute_error"] < 1e-4


def test_unsupported_capability_is_not_advertised(tmp_path: Path) -> None:
    (tmp_path / "colibri.exe").write_bytes(b"fixture")
    (tmp_path / ".build-config").write_text("CUDA=0|CUDA_DLL=0|VK=0|METAL=0", encoding="utf-8")
    report = ColibriCapabilityProbe(tmp_path).probe()
    assert report.supports_cuda is False
    assert report.supports_vulkan is False
    assert report.supports_metal is False
    assert report.supports_multi_gpu is False
    assert report.supports_native_mxfp4 is False
    assert report.supports_tensor_microshards is False
    assert "cuda" not in report.execution_backends


def test_orphaned_process_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    pid_file = tmp_path / "gateway.pid.json"
    pid_file.write_text(json.dumps({"pid": 999999, "model_path": str(model)}), encoding="utf-8")
    monkeypatch.setattr(
        "swarm_inference.backends.colibri.process.psutil.Process",
        lambda pid: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)),
    )
    assert ColibriProcess.cleanup_orphaned_process(pid_file, expected_model=model) is False
    assert not pid_file.exists()


def test_completed_experiment_009_bundle_resumes_without_rerun(tmp_path: Path) -> None:
    root = tmp_path / "experiment_009"
    bundle = EvidenceBundle(root, resume=False)
    for relative in REQUIRED_FILES:
        bundle.write_text(relative, "completed\n")
    bundle.write_json(
        "manifest.json",
        {
            "run_mode": "QUICK",
            "selected_configuration": None,
        },
    )
    bundle.write_json(
        "verdict.json",
        {
            "verdict": Experiment009Verdict.PASS_INTEGRATION.value,
            "run_mode": "QUICK",
            "completed": True,
            "terminal_error": None,
        },
    )
    bundle.complete_stage("evidence_tables")

    outcome = run_experiment_009(
        Experiment009Options(
            config_path=REPOSITORY_ROOT / "configs" / "experiments" / "experiment_009_colibri.yaml",
            output_directory=tmp_path,
            quick=True,
            resume=True,
        )
    )

    assert outcome.completed
    assert outcome.verdict is Experiment009Verdict.PASS_INTEGRATION
    assert outcome.bundle_path == root.resolve()


def test_colibri_real_bridge_fixture_if_built() -> None:
    """Run the actual patched C engine when the local reproducible build exists."""

    engines = REPOSITORY_ROOT / "build" / "colibri" / "bin"
    model = REPOSITORY_ROOT / "build" / "fixtures" / "glm_tiny"
    required = (
        engines / "colibri.exe",
        engines / "openai_server.py",
        engines / "swarm_bridge.py",
        model / "config.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("real pinned Colibri fixture build is not present")
    telemetry = model / ".swarm_colibri_test" / "telemetry.ndjson"
    process = ColibriProcess(
        engine_directory=engines,
        model_path=model,
        model_id="glm-tiny",
        model_revision="fixture-v1",
        mode=ColibriMode.BRIDGE,
        telemetry_level=TelemetryLevel.SUMMARY,
        telemetry_path=telemetry,
        cap=1,
        max_tokens=8,
        environment={"AUTOPIN": "0", "CAP_RAISE": "0", "OMP_NUM_THREADS": "1"},
        ram_safety_reserve_bytes=0,
    )
    try:
        process.start(timeout_seconds=120)
        direct = process.generate(prompt="?", max_tokens=4, request_id="real-direct")
        streamed = process.stream_generate(prompt="?", max_tokens=4, request_id="real-stream")
        assert direct.token_identity_observed and streamed.token_identity_observed
        assert direct.input_token_ids == streamed.input_token_ids == [63]
        assert direct.output_token_ids == streamed.output_token_ids
        assert len(direct.output_token_ids or []) == 4
        assert direct.stop_reason == streamed.stop_reason
    finally:
        process.shutdown()
    events = ColibriTelemetryReader(telemetry).read()
    summary = ColibriTelemetryReader.summarize(events)
    assert summary["event_type_counts"]["request_completed"] >= 2
    assert summary["storage_read_bytes"] > 0
