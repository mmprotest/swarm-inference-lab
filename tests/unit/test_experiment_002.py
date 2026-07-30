from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.config.real_model import load_real_experiment_config
from swarm_inference.exceptions import MemoryLimitExceededError
from swarm_inference.experiments.real_status import evaluate_experiment_002_status
from swarm_inference.experiments.runner import validate_run, write_artifact_manifest
from swarm_inference.model.adapter import (
    ComponentKind,
    ComponentRef,
    ModelDescription,
    TensorInfo,
)
from swarm_inference.model.manifest import hash_shard_directory
from swarm_inference.model.qwen3 import Qwen3Adapter, StageLocalKVCache
from swarm_inference.model.shard_builder import build_manifest
from swarm_inference.protocol.tensor_codec import (
    ActivationTensor,
    decode_tensor,
    encode_tensor,
)
from swarm_inference.worker.shard_manager import (
    ShardManager,
    attach_load_record_checksum,
    verify_load_record_checksum,
)


def _description(tmp_path: Path) -> ModelDescription:
    tensors = [
        TensorInfo(
            name="model.embed_tokens.weight",
            source_file="model.safetensors",
            dtype="BF16",
            shape=(50, 10),
            bytes=1000,
            component=ComponentRef(ComponentKind.EMBEDDING),
        ),
        TensorInfo(
            name="model.norm.weight",
            source_file="model.safetensors",
            dtype="BF16",
            shape=(10,),
            bytes=20,
            component=ComponentRef(ComponentKind.FINAL_NORM),
        ),
    ]
    for index in range(8):
        tensors.append(
            TensorInfo(
                name=f"model.layers.{index}.self_attn.q_proj.weight",
                source_file="model.safetensors",
                dtype="BF16",
                shape=(5, 10),
                bytes=100,
                component=ComponentRef(
                    ComponentKind.DECODER_LAYER,
                    layer_index=index,
                ),
            )
        )
    return ModelDescription(
        model_id="tiny-qwen3",
        model_revision="a" * 40,
        model_path=tmp_path,
        config={
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "num_hidden_layers": 8,
            "hidden_size": 10,
            "intermediate_size": 20,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 5,
            "vocab_size": 50,
            "max_position_embeddings": 128,
            "tie_word_embeddings": True,
        },
        tensors=tensors,
        source_file_hashes={"model.safetensors": "b" * 64},
    )


def test_qwen3_tensor_name_mapping_is_fail_closed() -> None:
    adapter = Qwen3Adapter()
    assert adapter.map_tensor_to_component("model.embed_tokens.weight").kind == "embedding"
    layer = adapter.map_tensor_to_component("model.layers.17.mlp.down_proj.weight")
    assert layer.kind == "decoder-layer"
    assert layer.layer_index == 17
    assert adapter.map_tensor_to_component("model.norm.weight").kind == "final-norm"
    assert adapter.map_tensor_to_component("lm_head.weight").kind == "output-head"
    with pytest.raises(Exception, match="cannot be mapped"):
        adapter.map_tensor_to_component("unsupported.weight")


def test_exact_four_stage_manifest_and_tied_weight_accounting(tmp_path: Path) -> None:
    manifest = build_manifest(
        _description(tmp_path),
        target_stage_bytes=1000,
        maximum_stage_bytes=5000,
        stage_count=4,
    )
    assert len(manifest.stages) == 4
    assert [(stage.layer_start, stage.layer_end) for stage in manifest.stages] == [
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 8),
    ]
    assert manifest.shared_tensors == {"model.embed_tokens.weight": [0, 3]}
    assert manifest.duplicated_tensor_bytes == 1000
    assert manifest.total_sharded_weight_bytes == manifest.total_weight_bytes + 1000
    assert manifest.stages[0].owns_embeddings
    assert manifest.stages[-1].owns_output_head
    assert all(stage.tensor_count == len(stage.tensor_names) for stage in manifest.stages)


def test_stage_local_cache_global_offset_mapping() -> None:
    record = StageLocalKVCache(
        cache=object(),
        request_id="r",
        model_revision="a" * 40,
        stage_id=2,
        layer_start=7,
        layer_end=14,
        route_generation=3,
        cache_generation=1,
    )
    assert record.global_to_local(7) == 0
    assert record.global_to_local(13) == 6
    assert record.local_to_global(0) == 7
    assert record.local_to_global(6) == 13
    with pytest.raises(IndexError):
        record.global_to_local(6)
    with pytest.raises(IndexError):
        record.local_to_global(7)
    record.advance(token_position=0, query_length=5)
    record.advance(token_position=5, query_length=1)
    assert record.sequence_length == 6


def test_bfloat16_tensor_transport_preserves_raw_bits_and_metadata() -> None:
    bits = np.array([[[0x3F80, 0x4000]]], dtype=np.uint16)
    encoded = encode_tensor(
        ActivationTensor(
            tensor_id="bf16",
            request_id="r",
            stage_id=1,
            token_position=3,
            sequence_length=1,
            array=bits,
            logical_dtype="bfloat16",
        )
    )
    decoded = decode_tensor(encoded)
    assert decoded.logical_dtype == "bfloat16"
    assert decoded.array.dtype == np.uint16
    assert np.array_equal(decoded.array, bits)


def _passing_bundle() -> dict[str, object]:
    stages = [
        {
            "stage_id": index,
            "layer_start": index * 2,
            "layer_end": (index + 1) * 2,
            "required_memory_bytes": 100,
        }
        for index in range(4)
    ]
    prompts = [
        {
            "name": (f"concurrent-{index}" if index in {7, 8} else f"prompt-{index}"),
            "phase": "smoke" if index == 0 else "suite",
            "status": "completed",
            "token_identity": True,
            "passed": True,
        }
        for index in range(9)
    ]
    prompts.append(
        {
            "name": "cache-replay",
            "phase": "replay",
            "status": "completed",
            "token_identity": True,
            "passed": True,
        }
    )
    return {
        "environment": {
            "python_version": "3.11.9",
            "cuda_available": True,
            "gpu": {
                "model": "NVIDIA GeForce RTX 5090",
                "bf16_supported": True,
            },
            "pytorch_version": "2.13.0+cu130",
            "transformers_version": "4.57.6",
            "safetensors_version": "0.8.0",
            "required_packages_preserved": True,
        },
        "manifest": {
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "a" * 40,
            "architecture": "Qwen3ForCausalLM",
            "layer_count": 8,
            "total_weight_bytes": 1000,
            "total_sharded_weight_bytes": 1100,
            "stages": stages,
        },
        "shard_validation": {
            "status": "PASS",
            "every_source_tensor_assigned": True,
            "decoder_layers_owned_exactly_once": True,
            "stage_hashes_valid": True,
            "union_reconstructs_required_state": True,
        },
        "reference": {
            "phase": "independent-full-model-reference",
            "process_id": 9000,
            "model_id": "Qwen/Qwen3-0.6B",
            "model_revision": "a" * 40,
            "device": "cuda",
            "dtype": "bfloat16",
            "full_model_loaded": True,
            "memory_counted_as_swarm": False,
            "greedy": True,
            "temperature": 0,
            "thinking_enabled": False,
            "results": [{} for _ in prompts],
        },
        "distributed": {
            "execution_mode": "single-host-loopback-real-model",
            "worker_backend": "torch-cuda",
            "one_stage_per_worker": True,
            "complete_stage_coverage": True,
            "model_larger_than_each_worker_limit": True,
            "model_larger_than_each_worker_total_limit": True,
            "worker_load_proofs": [
                {
                    "proof_verified": True,
                    "load_record_checksum": "b" * 64,
                    "load_record_checksum_verified": True,
                    "source_health_proof_checksum": "c" * 64,
                    "source_health_proof_signature": "signed",
                    "loaded_complete_source_tensor_set": False,
                    "decoder_layer_count": 2,
                    "stage_fits_logical_total_memory_limit": True,
                    "full_model_exceeds_logical_total_limit": True,
                    "stage_transfer_metrics": {
                        "operation_count": 1,
                        "host_to_device_copy_ms": 1.0,
                        "device_to_host_copy_ms": 1.0,
                        "host_to_device_bytes": 1,
                        "device_to_host_bytes": 1,
                    },
                    "process_id": 1000 + index,
                }
                for index in range(4)
            ],
            "coordinator_proof": {
                "coordinator_model_weight_bytes": 0,
                "coordinator_full_model_loaded": False,
                "process_id": 2000,
            },
            "transport_metrics": {
                "data_plane": "direct",
                "coordinator_activation_bytes": 0,
                "worker_to_worker_activation_bytes": 1,
                "peer_streams_created": 3,
                "activation_messages_sent": 3,
                "activation_messages_received": 3,
                "persistent_streams": True,
                "coordinator_relay_fallback": False,
            },
            "cache_metrics": {
                "histories": [
                    {
                        "owned_layer_count": 2,
                        "initialised_layer_count": 2,
                    }
                ],
                "owned_layer_count": 8,
                "active_cache_count_after_completion": 0,
                "stale_request_cache_remains": False,
            },
            "boundary_diagnostics": [
                {
                    "within_tolerance": True,
                    "nan_count": 0,
                    "inf_count": 0,
                }
                for _ in range(4)
            ],
            "cache_replay": {
                "cache_replay_preserved_output": True,
                "replay_input_count": 4,
                "replay_bytes": 100,
                "tokens_committed_before_failure": 4,
            },
            "log_validation": {
                "status": "PASS",
                "ignored_fatal_exception_count": 0,
            },
            "cleanup": {
                "all_workers_stopped": True,
                "all_workers_exited_zero": True,
                "stale_worker_processes": [],
                "worker_identity_files_removed": True,
                "worker_shutdown_files_removed": True,
            },
            "prompt_results": prompts,
        },
        "quality_gates": {
            "required": True,
            "overall_status": "PASS",
        },
    }


def test_real_status_is_fail_closed_for_every_mandatory_evidence() -> None:
    bundle = _passing_bundle()
    assert evaluate_experiment_002_status(bundle)["overall_status"] == "PASS"

    cases = [
        ("worker_load_proofs", []),
        (
            "transport_metrics",
            {
                "data_plane": "coordinator-relay",
                "coordinator_activation_bytes": 1,
            },
        ),
        ("boundary_diagnostics", [{"within_tolerance": False}]),
        ("cache_replay", {}),
    ]
    for key, replacement in cases:
        modified = _passing_bundle()
        modified["distributed"][key] = replacement  # type: ignore[index]
        assert evaluate_experiment_002_status(modified)["overall_status"] == "FAIL"

    mismatch = _passing_bundle()
    mismatch["distributed"]["prompt_results"][0]["token_identity"] = False  # type: ignore[index]
    assert evaluate_experiment_002_status(mismatch)["overall_status"] == "FAIL"

    failed_quality = _passing_bundle()
    failed_quality["quality_gates"]["overall_status"] = "FAIL"  # type: ignore[index]
    assert evaluate_experiment_002_status(failed_quality)["overall_status"] == "FAIL"

    failed_cleanup = _passing_bundle()
    failed_cleanup["distributed"]["cleanup"]["worker_identity_files_removed"] = False  # type: ignore[index]
    assert evaluate_experiment_002_status(failed_cleanup)["overall_status"] == "FAIL"

    shared_reference_process = _passing_bundle()
    shared_reference_process["reference"]["process_id"] = 1000  # type: ignore[index]
    assert evaluate_experiment_002_status(shared_reference_process)["overall_status"] == "FAIL"


def test_real_experiment_config_is_strict(repository_root: Path) -> None:
    config = load_real_experiment_config(
        repository_root / "configs" / "experiments" / "experiment_002_qwen3_real_loopback.yaml"
    )
    assert config.model.stage_count == 4
    assert config.workers.count == 4
    assert config.data_plane == "direct"


def test_worker_load_record_checksum_detects_tampering() -> None:
    record = attach_load_record_checksum(
        {
            "worker_id": "worker-0",
            "stage_id": 0,
            "tensor_names_loaded": ["model.embed_tokens.weight"],
            "total_loaded_weight_bytes": 100,
        }
    )
    assert verify_load_record_checksum(record)
    record["total_loaded_weight_bytes"] = 101
    assert not verify_load_record_checksum(record)


def test_worker_enforces_logical_total_memory_limit(tmp_path: Path) -> None:
    stage = (
        build_manifest(
            _description(tmp_path),
            target_stage_bytes=1000,
            maximum_stage_bytes=5000,
            stage_count=4,
        )
        .stages[0]
        .model_copy(update={"required_total_memory_bytes": 6000})
    )
    manager = ShardManager(
        memory_limit_bytes=5000,
        total_memory_limit_bytes=5500,
    )
    with pytest.raises(MemoryLimitExceededError, match="total-memory limit"):
        manager._reserve(stage)


def test_validate_run_dispatches_to_real_model_evidence_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    model_root = tmp_path / "model"
    run.mkdir()
    bundle = _passing_bundle()
    manifest = bundle["manifest"]
    distributed = bundle["distributed"]
    proofs = distributed["worker_load_proofs"]  # type: ignore[index]
    stage_hashes: dict[str, str] = {}
    for index, proof in enumerate(proofs):  # type: ignore[union-attr]
        stage_dir = model_root / f"stage-{index:03d}"
        stage_dir.mkdir(parents=True)
        weights = stage_dir / "weights.safetensors"
        weights.write_bytes(f"stage-{index}".encode())
        shard_hash = hash_shard_directory(stage_dir)
        stage_hashes[f"stage-{index:03d}"] = shard_hash
        proof["stage_id"] = index
        proof["shard_path"] = str(stage_dir)
        proof["shard_hash"] = shard_hash
        proofs[index] = attach_load_record_checksum(proof)
        manifest["stages"][index]["shard_hash"] = shard_hash  # type: ignore[index]

    shard_validation = bundle["shard_validation"]
    shard_hash_payload = {
        "model_id": manifest["model_id"],  # type: ignore[index]
        "resolved_revision": manifest["model_revision"],  # type: ignore[index]
        "stage_shards": stage_hashes,
    }
    model_root.mkdir(exist_ok=True)
    (model_root / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (model_root / "hashes.json").write_text(
        json.dumps(shard_hash_payload),
        encoding="utf-8",
    )
    (model_root / "validation.json").write_text(
        json.dumps(shard_validation),
        encoding="utf-8",
    )

    statuses = evaluate_experiment_002_status(bundle)
    summary = {
        **statuses,
        "execution_mode": "single-host-loopback-real-model",
    }
    json_payloads = {
        "environment.json": bundle["environment"],
        "git.json": {},
        "model_inspection.json": {},
        "model_manifest.json": manifest,
        "shard_hashes.json": shard_hash_payload,
        "reference.json": bundle["reference"],
        "distributed.json": distributed,
        "correctness.json": {},
        "worker_load_proofs.json": proofs,
        "coordinator_proof.json": distributed["coordinator_proof"],  # type: ignore[index]
        "transport_metrics.json": distributed["transport_metrics"],  # type: ignore[index]
        "cache_metrics.json": distributed["cache_metrics"],  # type: ignore[index]
        "cache_replay.json": distributed["cache_replay"],  # type: ignore[index]
        "log_validation.json": distributed["log_validation"],  # type: ignore[index]
        "quality_gates.json": bundle["quality_gates"],
        "summary.json": summary,
        "tensors/boundary-diagnostics.json": distributed[  # type: ignore[index]
            "boundary_diagnostics"
        ],
    }
    for relative, payload in json_payloads.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for relative in ("config.requested.yaml", "config.resolved.yaml"):
        (run / relative).write_text("name: experiment-002\n", encoding="utf-8")
    (run / "prompt_results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in distributed["prompt_results"]) + "\n",  # type: ignore[index]
        encoding="utf-8",
    )
    (run / "events.jsonl").write_text("{}\n", encoding="utf-8")
    conclusion = (
        "PASS: A real Qwen3 model was split across four process-isolated workers, "
        "real hidden states crossed worker boundaries, stage-local KV caches were "
        "used, and distributed greedy output matched the independent full-model "
        "reference exactly."
    )
    (run / "report.html").write_text(conclusion, encoding="utf-8")
    for name in (
        "coordinator.log",
        "reference.log",
        "worker-000.log",
        "worker-001.log",
        "worker-002.log",
        "worker-003.log",
    ):
        path = run / "logs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for name in (
        "stage_weight_bytes.png",
        "worker_memory.png",
        "prefill_latency.png",
        "decode_latency.png",
        "boundary_errors.png",
        "activation_bytes.png",
        "kv_cache_bytes.png",
    ):
        path = run / "charts" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"png")
    write_artifact_manifest(run)

    assert validate_run(run) == []
    (model_root / "stage-000" / "weights.safetensors").write_bytes(b"tampered")
    assert any("shard checksum mismatch" in error for error in validate_run(run))
