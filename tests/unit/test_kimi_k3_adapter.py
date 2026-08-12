from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.adapter import ComponentKind, default_native_adapter_registry
from swarm_inference.model.kimi_k3 import KimiK3CudaAdapter
from swarm_inference.model.kimi_tokenizer import apply_kimi_prompt_special_tokens
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime


def _assignment(*, device: str = "native-cuda:0") -> StageAssignment:
    return StageAssignment(
        stage_id=92,
        layer_start=92,
        layer_end=93,
        layer_ids=(92,),
        weight_bytes=1,
        estimated_compute_ns=1,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=1,
        peak_temporary_bytes=1,
        activation_bytes=1,
        device=device,
        owns_embeddings=False,
        owns_final_norm=True,
        owns_output_projection=True,
    )


def _request(**updates):
    values = {
        "worker_id": "worker-092",
        "request_id": "load",
        "model_id": "moonshotai/Kimi-K3",
        "model_revision": "revision",
        "tokenizer_revision": "sha256:" + "1" * 64,
        "topology_id": "topology",
        "stage_count": 93,
        "assignment": _assignment(),
        "adapter_id": "kimi_k3_cuda",
        "fast_path_id": "colibri-kimi-k3-cuda",
        "device": "native-cuda:0",
        "dtype": "float32",
        "model_path": "checkpoint",
    }
    values.update(updates)
    return LoadStageRequest(**values)


def test_kimi_prompt_special_tokens_follow_native_exactly_once_bos_rule() -> None:
    class _Tokenizer:
        bos_token_id = 163584

    tokenizer = _Tokenizer()
    assert apply_kimi_prompt_special_tokens(
        tokenizer, [18699], add_special_tokens=True
    ) == [163584, 18699]
    assert apply_kimi_prompt_special_tokens(
        tokenizer, [163584, 18699], add_special_tokens=True
    ) == [163584, 18699]
    assert apply_kimi_prompt_special_tokens(
        tokenizer, [18699], add_special_tokens=False
    ) == [18699]
    with pytest.raises(IntegrityError, match="no valid BOS"):
        apply_kimi_prompt_special_tokens(object(), [18699], add_special_tokens=True)


def test_kimi_adapter_is_discovered_and_maps_every_endpoint_class() -> None:
    adapter = default_native_adapter_registry().get("kimi_k3_cuda")
    assert isinstance(adapter, KimiK3CudaAdapter)
    assert adapter.supports(
        {
            "model_type": "kimi_k3",
            "architectures": ["KimiK3ForConditionalGeneration"],
            "text_config": {
                "model_type": "kimi_linear",
                "num_hidden_layers": 93,
            },
        }
    )
    assert (
        adapter.map_tensor_to_component("language_model.model.embed_tokens.weight").kind
        == ComponentKind.EMBEDDING
    )
    layer = adapter.map_tensor_to_component(
        "language_model.model.layers.92.self_attn.q_a_proj.weight"
    )
    assert layer.kind == ComponentKind.DECODER_LAYER
    assert layer.layer_index == 92
    assert (
        adapter.map_tensor_to_component(
            "language_model.model.output_attn_res_proj.weight"
        ).kind
        == ComponentKind.FINAL_NORM
    )
    assert (
        adapter.map_tensor_to_component("language_model.lm_head.weight").kind
        == ComponentKind.OUTPUT_HEAD
    )


def test_native_library_identity_is_paired_and_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="supplied together"):
        _request(native_runtime_library=str(tmp_path / "runtime.dll"))

    library = tmp_path / "runtime.dll"
    library.write_bytes(b"not-the-certified-library")
    request = _request(
        native_runtime_library=str(library),
        native_runtime_library_sha256="0" * 64,
    )
    with pytest.raises(IntegrityError, match="SHA-256 differs"):
        KimiK3CudaAdapter().create_stage_executor(
            request=request,
            resolved_model_path=tmp_path,
        )


def test_verification_major_load_configuration_round_trips() -> None:
    request = _request(
        fast_path_mode="verification-major",
        fast_path_batch_bucket=17,
        fast_path_context_bucket=8192,
    )

    restored = LoadStageRequest.model_validate_json(request.model_dump_json())

    assert restored.fast_path_mode == "verification-major"
    assert restored.fast_path_batch_bucket == 17
    assert restored.fast_path_context_bucket == 8192


def test_worker_pinned_identity_attests_unannotated_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "weights.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    model_content_fingerprint = "content-fingerprint"
    identity_path = tmp_path / "trusted-model-identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "schema_version": "swarm-model-identity-v1",
                "model_id": "test/model",
                "model_revision": "exact-revision",
                "tokenizer_revision": "sha256:" + "2" * 64,
                "adapter_id": "qwen3_dense",
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "safetensors_index_sha256": hashlib.sha256(
                    index_path.read_bytes()
                ).hexdigest(),
                "model_content_fingerprint": model_content_fingerprint,
                "assignment_sha256": "3" * 64,
            }
        ),
        encoding="utf-8",
    )
    runtime = PersistentStageRuntime(
        worker_id="worker",
        device="cpu",
        dtype="float32",
        memory_limit_bytes=1024,
        maximum_sessions=1,
        configured_model_path=tmp_path,
        configured_model_identity_path=identity_path,
    )
    adapter = runtime._verify_model_identity_values(
        model_id="test/model",
        requested_model_revision="exact-revision",
        requested_tokenizer_revision="sha256:" + "2" * 64,
        model_path=tmp_path.resolve(),
        requested_adapter_id="qwen3_dense",
        requested_model_content_fingerprint=model_content_fingerprint,
    )
    assert adapter.adapter_id == "qwen3_dense"

    index_path.write_text(index_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(IntegrityError, match="does not match local checkpoint"):
        runtime._verify_model_identity_values(
            model_id="test/model",
            requested_model_revision="exact-revision",
            requested_tokenizer_revision="sha256:" + "2" * 64,
            model_path=tmp_path.resolve(),
            requested_adapter_id="qwen3_dense",
            requested_model_content_fingerprint=model_content_fingerprint,
        )
