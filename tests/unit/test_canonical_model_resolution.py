from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub

from swarm_inference.cluster.artifacts import calculate_manifest_content_hash
from swarm_inference.cluster.models import ArtifactFile, ArtifactManifest
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    ExecutionDevice,
    ExecutionEngineCapability,
    WorkerExecutionCapability,
)
from swarm_inference.model.descriptor import ModelFileDescriptor
from swarm_inference.model.resolver import ModelSourceResolver, ResolutionResources
from swarm_inference.model.source import parse_model_source
from swarm_inference.model.variants import discover_gguf_variants


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
            SimpleNamespace(rfilename="config.json", size=20, lfs=None, blob_id="config"),
            SimpleNamespace(rfilename="tokenizer.json", size=30, lfs=None, blob_id="tokenizer"),
        ]
        return SimpleNamespace(
            sha="b" * 40,
            siblings=siblings,
            config={"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 4},
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
    descriptor = ModelSourceResolver(api=_NestedQwen36HubApi()).resolve("org/qwen36-gguf:UD-Q4_K_M")

    assert descriptor.architecture == "qwen3_moe"
    assert descriptor.architecture_raw == "qwen35moe"
    assert descriptor.architecture_source == "gguf.general.architecture"
    assert descriptor.layer_count == 40
    assert descriptor.hidden_size == 2048
    assert descriptor.activation_dtype_bytes == 2


def test_native_resolution_prefers_safetensors_without_acquiring_duplicate_weights(
    tmp_path: Path,
) -> None:
    resolver = ModelSourceResolver(api=_MixedNativeHubApi(), cache_directory=tmp_path)

    descriptor = resolver.resolve("org/mixed-native")

    names = {item.relative_path for item in descriptor.files}
    assert descriptor.format == "safetensors"
    assert descriptor.weight_bytes == 200
    assert "model.safetensors" in names
    assert "pytorch_model.bin" not in names
    assert {"config.json", "tokenizer.json"}.issubset(names)
    assert descriptor.tokenizer_identity is not None


def test_hub_resolution_selects_and_downloads_only_one_complete_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    resolver = ModelSourceResolver(cache_directory=tmp_path / "cache", api=_FakeHubApi())
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
