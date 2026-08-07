from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from swarm_inference.cluster.artifacts import calculate_manifest_content_hash
from swarm_inference.cluster.models import ArtifactFile, ArtifactManifest
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    ExecutionDevice,
    ExecutionEngineCapability,
    WorkerExecutionCapability,
)
from swarm_inference.model.descriptor import ModelFileDescriptor
from swarm_inference.model.gguf import GGUFInventory
from swarm_inference.model.resolver import ModelSourceResolver, ResolutionResources
from swarm_inference.model.safetensors import (
    SafetensorsHeaderError,
    inspect_safetensors_index_payload,
)
from swarm_inference.model.source import parse_model_source
from swarm_inference.model.variants import discover_gguf_variants, select_variant


def test_model_source_accepts_every_public_reference_shape() -> None:
    assert parse_model_source("org/model").model_id == "org/model"
    assert parse_model_source("org/model@commit").requested_revision == "commit"
    combined = parse_model_source("org/model:UD-Q4_K_M@commit")
    assert combined.variant == "UD-Q4_K_M"
    assert combined.requested_revision == "commit"
    url = parse_model_source("https://huggingface.co/org/model/tree/revision")
    assert (url.model_id, url.requested_revision) == ("org/model", "revision")
    windows = parse_model_source(r"C:\models\model.gguf")
    assert windows.source_type == "local"
    assert windows.local_path is not None


def test_multipart_gguf_is_atomic_and_uses_a_user_facing_variant_id() -> None:
    files = (
        ModelFileDescriptor(relative_path="model-UD-Q4_K_M-00001-of-00002.gguf", size_bytes=10),
        ModelFileDescriptor(relative_path="model-UD-Q4_K_M-00002-of-00002.gguf", size_bytes=11),
        # This incomplete split must never become runnable.
        ModelFileDescriptor(relative_path="model-Q8_0-00001-of-00002.gguf", size_bytes=20),
    )

    variants = discover_gguf_variants(files)

    assert len(variants) == 1
    assert variants[0].variant_id == "UD-Q4_K_M"
    assert variants[0].quantization == "UD-Q4_K_M"
    assert [item.multipart_index for item in variants[0].files] == [1, 2]


def test_gguf_variant_discovery_excludes_auxiliary_sidecars() -> None:
    files = (
        ModelFileDescriptor(relative_path="Qwen3.6-UD-Q4_K_M.gguf", size_bytes=100),
        ModelFileDescriptor(relative_path="mmproj-BF16.gguf", size_bytes=20),
        ModelFileDescriptor(relative_path="Qwen3.6-projector-F16.gguf", size_bytes=20),
        ModelFileDescriptor(relative_path="imatrix.Q8_0.gguf", size_bytes=10),
        ModelFileDescriptor(relative_path="tokenizer.gguf", size_bytes=5),
    )

    variants = discover_gguf_variants(files)

    assert [item.variant_id for item in variants] == ["UD-Q4_K_M"]
    assert variants[0].files[0].relative_path == "Qwen3.6-UD-Q4_K_M.gguf"


def test_explicit_gguf_variant_remains_inspectable_when_current_swarm_cannot_fit_it() -> None:
    variants = discover_gguf_variants(
        (ModelFileDescriptor(relative_path="model-Q4_K_M.gguf", size_bytes=100),)
    )

    selection = select_variant(
        variants,
        aggregate_usable_memory_bytes=50,
        requested_variant="Q4_K_M",
    )

    assert selection.selected.variant_id == "Q4_K_M"
    assert selection.candidates[0].feasible is False
    assert "exceeds aggregate usable memory" in selection.candidates[0].reason


def _fake_gguf_metadata_loader(
    *, repo_id: str, filename: str, revision: str, file_size: int
) -> GGUFInventory:
    architecture = "qwen35moe" if repo_id == "org/qwen36-gguf" else "qwen3moe"
    assert revision in {"a" * 40, "c" * 40}
    return GGUFInventory(
        path=Path(filename),
        version=3,
        metadata={
            "general.architecture": architecture,
            f"{architecture}.block_count": 40 if architecture == "qwen35moe" else 4,
            f"{architecture}.embedding_length": 2048 if architecture == "qwen35moe" else 32,
        },
        tensors=(),
        data_offset=min(32, file_size),
        file_size=file_size,
    )


class _FakeHubApi:
    def model_info(self, model_id: str, *, revision: str | None, files_metadata: bool):
        assert model_id == "org/multi-gguf"
        assert revision is None
        assert files_metadata
        siblings = [
            SimpleNamespace(
                rfilename="model-UD-Q4_K_M-00001-of-00002.gguf",
                size=100,
                lfs=None,
                blob_id="part-a",
            ),
            SimpleNamespace(
                rfilename="model-UD-Q4_K_M-00002-of-00002.gguf",
                size=100,
                lfs=None,
                blob_id="part-b",
            ),
            SimpleNamespace(
                rfilename="model-Q8_0.gguf",
                size=400,
                lfs=None,
                blob_id="q8",
            ),
            SimpleNamespace(rfilename="README.md", size=50, lfs=None, blob_id="readme"),
        ]
        return SimpleNamespace(
            sha="a" * 40,
            siblings=siblings,
            config={"architectures": ["Qwen3MoeForCausalLM"]},
            gguf={"architecture": "qwen3moe"},
        )


_MIXED_NATIVE_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "num_hidden_layers": 4,
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
}


class _MixedNativeHubApi:
    def model_info(self, model_id: str, *, revision: str | None, files_metadata: bool):
        assert model_id == "org/mixed-native"
        assert revision is None
        assert files_metadata
        siblings = [
            SimpleNamespace(
                rfilename="model.safetensors",
                size=200,
                lfs={"size": 200, "sha256": "1" * 64},
                blob_id="safe",
            ),
            SimpleNamespace(
                rfilename="pytorch_model.bin",
                size=210,
                lfs={"size": 210, "sha256": "2" * 64},
                blob_id="bin",
            ),
            SimpleNamespace(
                rfilename="config.json",
                size=len(json.dumps(_MIXED_NATIVE_CONFIG).encode("utf-8")),
                lfs=None,
                blob_id="config",
            ),
            SimpleNamespace(rfilename="tokenizer.json", size=30, lfs=None, blob_id="tokenizer"),
        ]
        return SimpleNamespace(
            sha="b" * 40,
            siblings=siblings,
            config={"architectures": ["Qwen3ForCausalLM"]},
        )


_SHARDED_INDEX = {
    "metadata": {"total_size": 17},
    "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    },
}


class _ShardedNativeHubApi:
    def model_info(self, model_id: str, *, revision: str | None, files_metadata: bool):
        assert model_id == "org/sharded-native"
        assert revision is None
        assert files_metadata
        config_bytes = json.dumps(_MIXED_NATIVE_CONFIG).encode("utf-8")
        index_bytes = json.dumps(_SHARDED_INDEX).encode("utf-8")
        return SimpleNamespace(
            sha="d" * 40,
            siblings=[
                SimpleNamespace(
                    rfilename="model-00001-of-00002.safetensors",
                    size=10,
                    lfs={"size": 10, "sha256": "1" * 64},
                    blob_id="shard-1",
                ),
                SimpleNamespace(
                    rfilename="model-00002-of-00002.safetensors",
                    size=11,
                    lfs={"size": 11, "sha256": "2" * 64},
                    blob_id="shard-2",
                ),
                SimpleNamespace(
                    rfilename="training-state.safetensors",
                    size=99,
                    lfs={"size": 99, "sha256": "3" * 64},
                    blob_id="orphan",
                ),
                SimpleNamespace(
                    rfilename="model.safetensors.index.json",
                    size=len(index_bytes),
                    lfs=None,
                    blob_id="index",
                ),
                SimpleNamespace(
                    rfilename="config.json",
                    size=len(config_bytes),
                    lfs=None,
                    blob_id="config",
                ),
            ],
            config={"architectures": ["Qwen3ForCausalLM"]},
            safetensors={"total": 100},
        )


class _NestedQwen36HubApi:
    def model_info(self, model_id: str, *, revision: str | None, files_metadata: bool):
        assert model_id == "org/qwen36-gguf"
        assert revision is None
        assert files_metadata
        return SimpleNamespace(
            sha="c" * 40,
            siblings=[
                SimpleNamespace(
                    rfilename="Qwen3.6-UD-Q4_K_M.gguf",
                    size=220,
                    lfs=None,
                    blob_id="qwen36-gguf",
                )
            ],
            config={
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 40,
                    "hidden_size": 2048,
                    "torch_dtype": "bfloat16",
                },
            },
            gguf={"architecture": "qwen35moe"},
        )


def test_remote_qwen36_uses_exact_gguf_architecture_and_nested_text_facts() -> None:
    descriptor = ModelSourceResolver(
        api=_NestedQwen36HubApi(),
        gguf_metadata_loader=_fake_gguf_metadata_loader,
    ).resolve("org/qwen36-gguf:UD-Q4_K_M")

    assert descriptor.architecture == "qwen3_5_moe"
    assert descriptor.architecture_raw == "qwen35moe"
    assert descriptor.architecture_source == "gguf.general.architecture"
    assert descriptor.layer_count == 40
    assert descriptor.hidden_size == 2048
    assert descriptor.activation_dtype_bytes == 2


def test_native_resolution_prefers_safetensors_without_acquiring_duplicate_weights(
    tmp_path: Path,
) -> None:
    def metadata_loader(*, repo_id: str, filename: str, revision: str, cache_dir: str):
        assert (repo_id, filename, revision) == ("org/mixed-native", "config.json", "b" * 40)
        assert cache_dir == str(tmp_path.resolve())
        target = tmp_path / "metadata" / filename
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(_MIXED_NATIVE_CONFIG), encoding="utf-8")
        return target

    resolver = ModelSourceResolver(
        api=_MixedNativeHubApi(),
        cache_directory=tmp_path,
        metadata_loader=metadata_loader,
    )

    descriptor = resolver.resolve("org/mixed-native")

    names = {item.relative_path for item in descriptor.files}
    assert descriptor.format == "safetensors"
    assert descriptor.weight_bytes == 200
    assert "model.safetensors" in names
    assert "pytorch_model.bin" not in names
    assert {"config.json", "tokenizer.json"}.issubset(names)
    assert descriptor.tokenizer_identity is not None
    assert descriptor.layer_count == 4
    assert descriptor.hidden_size == 8
    assert descriptor.architecture_profile is not None
    assert descriptor.artifact_metadata["config"]["sha256"]


def test_sharded_safetensors_resolution_uses_the_immutable_index_map(tmp_path: Path) -> None:
    def metadata_loader(*, repo_id: str, filename: str, revision: str, cache_dir: str):
        assert repo_id == "org/sharded-native"
        assert revision == "d" * 40
        assert cache_dir == str(tmp_path.resolve())
        payload = _SHARDED_INDEX if filename.endswith(".index.json") else _MIXED_NATIVE_CONFIG
        target = tmp_path / "metadata" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    descriptor = ModelSourceResolver(
        api=_ShardedNativeHubApi(),
        cache_directory=tmp_path,
        metadata_loader=metadata_loader,
    ).resolve("org/sharded-native")

    names = {item.relative_path for item in descriptor.files}
    assert descriptor.weight_bytes == 21
    assert "training-state.safetensors" not in names
    assert {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "config.json",
    } == names
    identity = descriptor.artifact_metadata["safetensors_index"]
    assert identity["tensor_count"] == 3
    assert identity["shard_count"] == 2
    assert identity["tensors_per_shard"] == {
        "model-00001-of-00002.safetensors": 1,
        "model-00002-of-00002.safetensors": 2,
    }
    assert len(identity["mapping_sha256"]) == 64


@pytest.mark.parametrize(
    "target",
    (
        "missing.safetensors",
        "../outside.safetensors",
        "/absolute.safetensors",
        r"C:\outside.safetensors",
        "model.bin",
    ),
)
def test_safetensors_index_targets_fail_closed(target: str) -> None:
    with pytest.raises(SafetensorsHeaderError):
        inspect_safetensors_index_payload(
            {"weight_map": {"model.layers.0.weight": target}},
            source="model.safetensors.index.json",
            available_files=("model.safetensors",),
        )


def test_safetensors_index_accepts_integral_total_size_emitted_as_json_float() -> None:
    inventory = inspect_safetensors_index_payload(
        {
            "metadata": {"total_size": 71_903_645_408.0},
            "weight_map": {"model.layers.0.weight": "model.safetensors"},
        },
        source="model.safetensors.index.json",
        available_files=("model.safetensors",),
    )

    assert inventory.declared_total_size == 71_903_645_408


@pytest.mark.parametrize("value", (-1, 1.5, float("inf"), "100", True))
def test_safetensors_index_rejects_invalid_total_size(value: object) -> None:
    with pytest.raises(SafetensorsHeaderError, match="total_size"):
        inspect_safetensors_index_payload(
            {
                "metadata": {"total_size": value},
                "weight_map": {"model.layers.0.weight": "model.safetensors"},
            },
            source="model.safetensors.index.json",
            available_files=("model.safetensors",),
        )


def test_hub_resolution_selects_and_downloads_only_one_complete_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolver = ModelSourceResolver(
        cache_directory=tmp_path / "cache",
        api=_FakeHubApi(),
        gguf_metadata_loader=_fake_gguf_metadata_loader,
    )
    resolution = resolver.inspect(
        "org/multi-gguf",
        resources=ResolutionResources(
            aggregate_usable_memory_bytes=300,
            local_fast_memory_bytes=200,
        ),
    )
    descriptor = resolution.descriptor
    assert descriptor.revision == "a" * 40
    assert descriptor.variant == "UD-Q4_K_M"
    assert descriptor.quantization == "UD-Q4_K_M"
    assert descriptor.architecture == "qwen3_moe"
    assert descriptor.architecture_raw == "qwen3moe"
    assert descriptor.architecture_source == "gguf.general.architecture"
    assert descriptor.weight_bytes == 200
    assert len(descriptor.files) == 2
    assert not any(item.relative_path == "README.md" for item in descriptor.files)

    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, revision: str, cache_dir: str):
        assert revision == "a" * 40
        calls.append((repo_id, filename))
        target = tmp_path / "snapshot" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 100)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    paths = resolver.acquire(descriptor)

    assert len(paths) == 2
    assert calls == [
        ("org/multi-gguf", "model-UD-Q4_K_M-00001-of-00002.gguf"),
        ("org/multi-gguf", "model-UD-Q4_K_M-00002-of-00002.gguf"),
    ]


def test_cluster_memory_deduplicates_devices_exposed_by_multiple_engines() -> None:
    device = ExecutionDevice(
        device_id="cuda:0",
        device_type="cuda",
        uuid="gpu-1",
        name="GPU",
        usable_memory_bytes=24,
    )
    cluster = ClusterCapabilities(
        workers=(
            WorkerExecutionCapability(
                worker_id="node-a/gpu-0",
                node_id="node-a",
                engines=(
                    ExecutionEngineCapability(
                        engine_id="native-stage", enabled=True, devices=(device,)
                    ),
                    ExecutionEngineCapability(
                        engine_id="llamacpp-rpc", enabled=True, devices=(device,)
                    ),
                ),
            ),
        )
    )
    assert cluster.aggregate_usable_memory_bytes == 24


def test_artifact_manifest_v2_represents_non_stage_artifacts() -> None:
    file = ArtifactFile(relative_path="model.gguf", size_bytes=3, sha256="1" * 64)
    provisional = ArtifactManifest(
        artifact_id="0" * 64,
        model_id="org/model",
        model_revision="a" * 40,
        model_fingerprint="sha256:" + "2" * 64,
        engine_id="llamacpp-rpc",
        model_format="gguf",
        artifact_kind="gguf",
        files=[file],
        content_hash="0" * 64,
        total_size_bytes=3,
        total_bytes=3,
    )
    content_hash = calculate_manifest_content_hash(provisional)
    manifest = provisional.model_copy(
        update={"artifact_id": content_hash, "content_hash": content_hash}
    )

    assert manifest.adapter_id is None
    assert manifest.stage_assignment_id is None
    assert manifest.artifact_kind == "gguf"
