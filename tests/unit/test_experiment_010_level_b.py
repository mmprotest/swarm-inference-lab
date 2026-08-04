from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm_inference.config.experiment_008 import load_experiment_008_config
from swarm_inference.experiments.experiment_008.acquisition import resolve_model_candidate
from swarm_inference.experiments.experiment_008.runner import Experiment008Options
from swarm_inference.experiments.experiment_010 import correction_bundle
from swarm_inference.experiments.experiment_010.level_b import (
    OFFICIAL_FILENAME,
    OFFICIAL_MODEL_ID,
    OFFICIAL_REPOSITORY,
    PINNED_REVISION,
    Gate17ValidationError,
    validate_config_pins,
    validate_existing_correction_evidence,
    validate_level_b_bundle,
)
from swarm_inference.experiments.experiment_010.runner import Experiment010Options
from swarm_inference.experiments.experiment_010.schemas import Experiment010Mode


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _csv(path, rows)


def _fake_level_b(tmp_path: Path) -> tuple[Path, Path, str]:
    level_root = tmp_path / "level-b-current"
    bundle = level_root / "experiment_008"
    bundle.mkdir(parents=True)
    model = tmp_path / OFFICIAL_FILENAME
    model.write_bytes(b"official-qwen-gate-17-fixture")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    arguments = [
        "--n-gpu-layers",
        "auto",
        "--threads",
        "20",
        "--threads-batch",
        "20",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--flash-attn",
        "on",
    ]
    rows: list[dict[str, object]] = []
    for workload in ("decode", "prefill_8k", "prefill_32k", "mixed"):
        prompt_min = (
            32_000 if workload == "prefill_32k" else 8_000 if workload == "prefill_8k" else 300
        )
        rows.append(
            {
                "configuration": "A",
                "workload": workload,
                "status": "COMPLETED",
                "evidence_class": "MEASURED",
                "evidence_category": "REAL_MODEL_MEASURED",
                "historical_row": False,
                "model_file_sha256": digest,
                "model_filename": OFFICIAL_FILENAME,
                "model_revision": PINNED_REVISION,
                "prompt_token_count_min": prompt_min,
                "prompt_token_count_max": prompt_min,
                "output_token_count": 64,
                "decode_tokens_per_second": 20.0,
                "decode_tokens_per_second_p95": 21.0,
                "time_to_first_token_ms": 100.0,
                "time_to_first_token_p95_ms": 110.0,
                "prefill_tokens_per_second": 1000.0,
                "interactive_tokens_per_second": 18.0,
                "background_tokens_per_second": 17.0,
                "combined_generated_tokens_per_second": 34.0,
                "mixed_verified_tokens_per_second": 34.0,
                "interactive_p95_latency_ms": 60.0,
                "measurement_window_seconds": 120.1 if workload == "mixed" else 10.0,
                "peak_vram_bytes": 24 * 1024**3,
                "peak_system_ram_bytes": 80 * 1024**3,
                "backend_arguments": arguments,
                "backend_command": ["llama-server", "--model", str(model), *arguments],
            }
        )
    _csv(bundle / "benchmark_results.csv", rows)
    resolved = {
        "candidate": "preferred",
        "model_id": OFFICIAL_MODEL_ID,
        "artifact_repository": OFFICIAL_REPOSITORY,
        "filename": OFFICIAL_FILENAME,
        "quantization": "Q4_K_M",
        "architecture": "qwen3next",
        "requested_revision": PINNED_REVISION,
        "resolved_revision": PINNED_REVISION,
        "path": str(model),
        "source": "user-supplied-local-file",
        "file_size": model.stat().st_size,
        "file_sha256": digest,
        "expected_file_size": model.stat().st_size,
        "expected_file_sha256": digest,
        "acquisition_checks": {"local_file_valid": True},
    }
    _json(
        bundle / "model_resolution_attempts.json",
        [{"candidate": "preferred", "status": "COMPLETED", "resolved": resolved}],
    )
    _json(
        bundle / "model_preflight.json",
        {
            "eligible": True,
            "artifact_repository": OFFICIAL_REPOSITORY,
            "resolved_artifact_identity": PINNED_REVISION,
            "model_file_name": OFFICIAL_FILENAME,
            "model_file_size_bytes": model.stat().st_size,
            "model_file_sha256": digest,
            "total_tensor_bytes": 40 * 1024**3,
            "physical_vram_bytes": 32 * 1024**3,
            "system_ram_available_bytes": 72 * 1024**3,
        },
    )
    _json(
        bundle / "model_execution_precheck.json",
        {
            "status": "COMPLETED",
            "model_file_sha256": digest,
            "output_token_ids": [1, 2],
            "generated_text": "Verified real generation.",
        },
    )
    _json(bundle / "tensor_inventory.json", {"tensor_bytes": 40 * 1024**3})
    _json(bundle / "tensor_tiles.json", {"classification": "MEASURED", "tiles": []})
    _json(
        bundle / "backend_acquisition.json",
        {
            "source": "cached-official-release",
            "release_tag": "b9637",
            "cuda_version": "13.3",
            "path": str(tmp_path / "llama-server.exe"),
            "sha256": "b" * 64,
        },
    )
    _json(
        bundle / "backend_probe.json",
        {"gpu_layer_offload_capability": True, "cuda_available": True},
    )
    _json(bundle / "hardware_profile.json", {"classification": "MEASURED"})
    _json(bundle / "environment.json", {"gpu": "NVIDIA GeForce RTX 5090"})
    _csv(bundle / "resource_timeseries.csv", [{"classification": "MEASURED"}])
    _json(bundle / "correctness_results.json", {"classification": "MEASURED"})
    _json(bundle / "manifest.json", {"selected_configuration": "A", "gate_17_only": True})
    (bundle / "report.md").write_text("# Level B\n", encoding="utf-8")
    _json(bundle / "verdict.json", {"real_model_generation_succeeded": True})
    _json(
        bundle / "residency_accounting.json",
        {
            "backend_reported_cpu_model_bytes": 16 * 1024**3,
            "backend_reported_gpu_model_bytes": 24 * 1024**3,
            "system_ram_contributes": True,
        },
    )
    _json(
        bundle / "correctness_tokens.json",
        {
            f"A:{workload}": [
                {
                    "success": True,
                    "output_token_ids": [1, 2, 3],
                    "content": f"real {workload} text",
                }
            ]
            for workload in ("decode", "prefill_8k", "prefill_32k", "mixed")
        },
    )
    (bundle / "logs").mkdir()
    return level_root, bundle, digest


def _options(**updates: object) -> Experiment010Options:
    values: dict[str, object] = {
        "mode": Experiment010Mode.FULL,
        "repeats": 3,
        "correction_pass": True,
        "level_b_only": True,
    }
    values.update(updates)
    return Experiment010Options(**values)  # type: ignore[arg-type]


def test_level_b_only_requires_correction_pass() -> None:
    with pytest.raises(ValueError, match="correction_pass"):
        _options(correction_pass=False).validate()


def test_level_b_only_requires_full_mode() -> None:
    with pytest.raises(ValueError, match="full mode"):
        _options(mode=Experiment010Mode.QUICK, repeats=1).validate()


def test_level_b_only_rejects_skip_level_b() -> None:
    with pytest.raises(ValueError, match="skip_level_b"):
        _options(skip_level_b=True).validate()


def test_level_b_only_uses_existing_correction_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    checks = {name: True for name in correction_bundle._validate_phases.__annotations__}
    del checks
    monkeypatch.setattr(
        correction_bundle,
        "_validate_phases",
        lambda *_args, **_kwargs: {
            "checks": {"level_b_current_workload": False, "repository_integrity": True},
            "paths": {},
        },
    )
    result = validate_existing_correction_evidence(tmp_path, work)
    assert result["work_directory"] == str(work.resolve())
    with pytest.raises(FileNotFoundError, match="correction work"):
        validate_existing_correction_evidence(tmp_path, tmp_path / "missing")


def test_level_b_only_without_model_path_allows_download(repository_root: Path) -> None:
    options = Experiment008Options(
        config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        full=True,
        configuration="A",
        gate_17_only=True,
    )
    options.validate()
    assert options.model_path is None


def test_level_b_only_with_model_path_uses_local_file(
    tmp_path: Path, repository_root: Path
) -> None:
    model = tmp_path / OFFICIAL_FILENAME
    model.write_bytes(b"local")
    config = load_experiment_008_config(
        repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml"
    )
    resolved = resolve_model_candidate(
        config.models.preferred,
        candidate_name="preferred",
        model_path=model,
        cache_dir=tmp_path / "cache",
        skip_download=False,
        require_exact_filename=True,
    )
    assert resolved.path == str(model.resolve())
    assert resolved.source == "user-supplied-local-file"


def test_level_b_official_repository_pinned(repository_root: Path) -> None:
    pins = validate_config_pins(
        repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml"
    )
    assert pins["preferred"]["artifact_repository"] == OFFICIAL_REPOSITORY
    assert pins["preferred"]["revision"] == PINNED_REVISION


def test_level_b_official_filename_pinned(repository_root: Path) -> None:
    pins = validate_config_pins(
        repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml"
    )
    assert pins["preferred"]["filename"] == OFFICIAL_FILENAME


def test_level_b_fallback_model_forbidden(tmp_path: Path, repository_root: Path) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    attempts = json.loads((bundle / "model_resolution_attempts.json").read_text())
    attempts[0]["candidate"] = "fallback"
    _json(bundle / "model_resolution_attempts.json", attempts)
    with pytest.raises(Gate17ValidationError, match="official Qwen3-Next"):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_download_resumable(
    tmp_path: Path, repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"downloaded-official-model"
    digest = hashlib.sha256(data).hexdigest()
    calls: dict[str, object] = {}
    missing = object()

    class Api:
        def __init__(self, **kwargs: object) -> None:
            calls["api_init"] = kwargs

        def model_info(self, *_args: object, **kwargs: object) -> object:
            calls["model_info"] = kwargs
            return SimpleNamespace(
                sha=PINNED_REVISION,
                siblings=[
                    SimpleNamespace(
                        rfilename=OFFICIAL_FILENAME,
                        size=len(data),
                        lfs={"sha256": digest},
                    )
                ],
            )

    def download(**kwargs: object) -> str:
        calls["download"] = kwargs
        target = tmp_path / OFFICIAL_FILENAME
        target.write_bytes(data)
        return str(target)

    hub = types.ModuleType("huggingface_hub")
    hub.__version__ = "test"
    hub.HfApi = Api
    hub.hf_hub_download = download
    hub.try_to_load_from_cache = lambda *_args, **_kwargs: missing
    file_download = types.ModuleType("huggingface_hub.file_download")
    file_download._CACHED_NO_EXIST = missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", file_download)
    config = load_experiment_008_config(
        repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml"
    )
    resolved = resolve_model_candidate(
        config.models.preferred,
        candidate_name="preferred",
        model_path=None,
        cache_dir=tmp_path / "cache",
        skip_download=False,
        require_public_repository=True,
    )
    assert resolved.file_sha256 == digest
    assert calls["download"]["resume_download"] is True  # type: ignore[index]
    assert calls["download"]["revision"] == PINNED_REVISION  # type: ignore[index]


@pytest.mark.parametrize("missing_workload", ["decode", "prefill_8k", "prefill_32k", "mixed"])
def test_level_b_gate_requires_each_workload(
    missing_workload: str, tmp_path: Path, repository_root: Path
) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    rows = [
        row
        for row in _read_rows(bundle / "benchmark_results.csv")
        if row["workload"] != missing_workload
    ]
    _write_rows(bundle / "benchmark_results.csv", rows)
    with pytest.raises(Gate17ValidationError, match=missing_workload):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_gate_requires_decode(tmp_path: Path, repository_root: Path) -> None:
    test_level_b_gate_requires_each_workload("decode", tmp_path, repository_root)


def test_level_b_gate_requires_prefill_8k(tmp_path: Path, repository_root: Path) -> None:
    test_level_b_gate_requires_each_workload("prefill_8k", tmp_path, repository_root)


def test_level_b_gate_requires_prefill_32k(tmp_path: Path, repository_root: Path) -> None:
    test_level_b_gate_requires_each_workload("prefill_32k", tmp_path, repository_root)


def test_level_b_gate_requires_mixed(tmp_path: Path, repository_root: Path) -> None:
    test_level_b_gate_requires_each_workload("mixed", tmp_path, repository_root)


def test_level_b_gate_rejects_synthetic_rows(tmp_path: Path, repository_root: Path) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    rows = _read_rows(bundle / "benchmark_results.csv")
    rows[0]["evidence_category"] = "SYNTHETIC_FIXTURE"
    _write_rows(bundle / "benchmark_results.csv", rows)
    with pytest.raises(Gate17ValidationError, match="decode"):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_gate_rejects_historical_rows(tmp_path: Path, repository_root: Path) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    rows = _read_rows(bundle / "benchmark_results.csv")
    rows[1]["historical_row"] = "true"
    _write_rows(bundle / "benchmark_results.csv", rows)
    with pytest.raises(Gate17ValidationError, match="prefill_8k"):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_gate_requires_over_vram(tmp_path: Path, repository_root: Path) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    preflight = json.loads((bundle / "model_preflight.json").read_text())
    preflight["total_tensor_bytes"] = preflight["physical_vram_bytes"]
    _json(bundle / "model_preflight.json", preflight)
    with pytest.raises(Gate17ValidationError, match="do not exceed"):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_gate_requires_same_model_hash(tmp_path: Path, repository_root: Path) -> None:
    root, bundle, _ = _fake_level_b(tmp_path)
    rows = _read_rows(bundle / "benchmark_results.csv")
    rows[-1]["model_file_sha256"] = "f" * 64
    _write_rows(bundle / "benchmark_results.csv", rows)
    with pytest.raises(Gate17ValidationError, match="same model file SHA-256"):
        validate_level_b_bundle(
            root,
            config_path=repository_root / "configs/experiments/experiment_008_adaptive_moe.yaml",
        )


def test_level_b_only_preserves_existing_gate_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    protected = root / "correctness_results.json"
    protected.write_text("unchanged", encoding="utf-8")
    (root / "verdict.json").write_text("old", encoding="utf-8")
    hashes = correction_bundle._immutable_artifact_hashes(root)
    (root / "verdict.json").write_text("new", encoding="utf-8")
    correction_bundle._assert_immutable_artifacts(root, hashes)
    protected.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="protected"):
        correction_bundle._assert_immutable_artifacts(root, hashes)


def _passing_validation() -> dict[str, object]:
    checks = {
        "repository_integrity": True,
        "level_a_merged_colibri_model": True,
        "level_a_worker_expert_banks": True,
        "level_b_current_workload": True,
        "native_colibri_cuda_real_model_path": True,
        "whole_expert_token_path": True,
        "native_microshard_token_path": True,
        "capacity_isolated_generation": True,
        "mandatory_real_model_workloads": True,
        "real_path_failure_matrix": True,
        "real_path_corruption_matrix": True,
        "measured_real_path_planner": True,
        "simulator_behavioral_parity": True,
        "simulator_heldout_validation": True,
        "dense_kimi_fixture": True,
    }
    return {"checks": checks, "reasons": {}}


def test_level_b_completion_sets_full_complete() -> None:
    completeness = correction_bundle._full_completeness(
        Experiment010Mode.FULL, _passing_validation()
    )
    assert completeness["status"] == "FULL_COMPLETE"
    assert completeness["full_complete"] is True


def test_level_b_completion_sets_pass_closure() -> None:
    validation = _passing_validation()
    completeness = correction_bundle._full_completeness(Experiment010Mode.FULL, validation)
    gates = correction_bundle._gate_rows(validation, artifact_complete=True)
    verdict = (
        "PASS_CLOSURE"
        if completeness["full_complete"] and all(row["status"] == "PASS" for row in gates)
        else "PARTIAL"
    )
    assert verdict == "PASS_CLOSURE"
