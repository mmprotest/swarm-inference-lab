from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm_inference.acceptance.major_models import (
    MajorModelTarget,
    ValidationStatus,
    _component_plan,
    _model_summary_row,
    load_suite,
    resolve_artifact_preflight,
)


def test_component_evidence_reads_canonical_placement() -> None:
    selected = SimpleNamespace(
        components=(
            SimpleNamespace(
                component_type=SimpleNamespace(value="routed-experts"),
                engine_id="colibri",
                placement=SimpleNamespace(
                    device="cuda:0",
                    worker_ids=("local/worker",),
                ),
            ),
        ),
        worker_roles={"local/worker": "routed_expert_component"},
    )
    summary = SimpleNamespace(
        canonical_decision=SimpleNamespace(selected=selected),
        plan=selected,
        engine_id="native-stage",
    )

    components, workers = _component_plan(summary)  # type: ignore[arg-type]

    assert components == [
        {
            "component_type": "routed-experts",
            "engine_id": "colibri",
            "device": "cuda:0",
            "worker_ids": ["local/worker"],
        }
    ]
    assert workers == ["local/worker"]


def test_canonical_matrix_covers_every_required_architecture() -> None:
    suite = load_suite(Path("configs/validation/major_open_weight_models.yaml"))

    assert len(suite.targets) == 16
    assert {target.architecture_id for target in suite.targets} >= {
        "qwen3_dense",
        "qwen3_moe",
        "qwen3_5_dense",
        "qwen3_5_moe",
        "kimi_k2_moe",
        "kimi_k3_moe",
        "glm_moe",
        "deepseek_v3_moe",
        "deepseek_v4_moe",
        "minimax_moe",
        "llama_dense",
        "mistral_dense",
        "mixtral_moe",
        "gemma_dense",
    }
    assert all(target.mandatory for target in suite.targets)
    assert all(
        "colibri" in target.comparison_engines
        for target in suite.targets
        if target.require_colibri
    )


def test_preflight_pins_revision_and_accounts_for_cached_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    suite = load_suite(Path("configs/validation/major_open_weight_models.yaml"))
    target = MajorModelTarget(
        id="fixture",
        family="Fixture",
        publisher="official",
        model_id="official/model",
        architecture_id="fixture_dense",
        dense_or_moe="dense",
        expected_format="safetensors",
    )
    revision = "a" * 40
    files = (
        SimpleNamespace(rfilename="config.json", size=128),
        SimpleNamespace(rfilename="tokenizer.json", size=256),
        SimpleNamespace(rfilename="model.safetensors", size=4096),
    )
    api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: SimpleNamespace(
            sha=revision,
            siblings=files,
            last_modified="immutable-fixture",
        )
    )
    cache = tmp_path / "cache"
    snapshot = cache / "models--official--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"x" * 128)
    monkeypatch.setattr("swarm_inference.acceptance.major_models._cache_root", lambda: cache)

    preflight, identity = resolve_artifact_preflight(
        target,
        suite,
        api=api,  # type: ignore[arg-type]
        output_root=tmp_path,
    )

    assert preflight.revision == revision
    assert preflight.required_artifact_bytes == 4480
    assert preflight.cached_artifact_bytes == 128
    assert preflight.remaining_download_bytes == 4352
    assert preflight.cache_complete is False
    assert preflight.artifact_size_exact is True
    assert identity["official_namespace_verified"] is True


def test_offline_preflight_uses_one_immutable_cached_safetensors_index(
    tmp_path: Path, monkeypatch
) -> None:
    suite = load_suite(Path("configs/validation/major_open_weight_models.yaml"))
    target = MajorModelTarget(
        id="fixture",
        family="Fixture",
        publisher="official",
        model_id="official/model",
        architecture_id="fixture_moe",
        dense_or_moe="moe",
        expected_format="safetensors",
        streaming_model=True,
    )
    revision = "c" * 40
    cache = tmp_path / "cache"
    repository = cache / "models--official--model"
    snapshot = repository / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text(revision, encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"x" * 128)
    index = {
        "metadata": {"total_size": 4096.0},
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        },
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConnectionError("offline")
        )
    )
    monkeypatch.setattr("swarm_inference.acceptance.major_models._cache_root", lambda: cache)

    preflight, identity = resolve_artifact_preflight(
        target,
        suite,
        api=api,  # type: ignore[arg-type]
        output_root=tmp_path,
    )

    assert preflight.revision == revision
    assert preflight.required_artifact_bytes > 4096
    assert preflight.cached_artifact_bytes >= 128
    assert preflight.cache_complete is False
    assert preflight.artifact_size_exact is False
    assert identity["revision_resolution"] == "cached-immutable-safetensors-index"
    assert identity["current_revision_verified_online"] is False


def test_offline_fallback_does_not_mask_publisher_contract_errors(
    tmp_path: Path, monkeypatch
) -> None:
    suite = load_suite(Path("configs/validation/major_open_weight_models.yaml"))
    target = MajorModelTarget(
        id="fixture",
        family="Fixture",
        publisher="official",
        model_id="official/model",
        architecture_id="fixture_dense",
        dense_or_moe="dense",
        expected_format="safetensors",
    )
    cache = tmp_path / "cache"
    monkeypatch.setattr("swarm_inference.acceptance.major_models._cache_root", lambda: cache)
    api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("publisher metadata malformed")
        )
    )

    with pytest.raises(ValueError, match="publisher metadata malformed"):
        resolve_artifact_preflight(
            target,
            suite,
            api=api,  # type: ignore[arg-type]
            output_root=tmp_path,
        )


def test_non_execution_status_can_never_be_reported_as_real_run() -> None:
    target = MajorModelTarget(
        id="fixture",
        family="Fixture",
        publisher="official",
        model_id="official/model",
        architecture_id="fixture_dense",
        dense_or_moe="dense",
        expected_format="safetensors",
    )

    row = _model_summary_row(
        target,
        ValidationStatus.NOT_RUN,
        "b" * 40,
        (),
        None,
        "preflight only",
    )

    assert row["status"] == "NOT_RUN"
    assert row["real_run"] is False
    assert row["correctness"] == "NOT_RUN"
