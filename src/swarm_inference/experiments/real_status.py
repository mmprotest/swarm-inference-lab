"""Fail-closed acceptance status for Experiment 002."""

from __future__ import annotations

import re
from typing import Any


def _status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def evaluate_experiment_002_status(bundle: dict[str, Any]) -> dict[str, str]:
    environment = bundle.get("environment", {})
    manifest = bundle.get("manifest", {})
    shard_validation = bundle.get("shard_validation", {})
    reference = bundle.get("reference", {})
    distributed = bundle.get("distributed", {})
    coordinator = distributed.get("coordinator_proof", {})
    transport = distributed.get("transport_metrics", {})
    cache = distributed.get("cache_metrics", {})
    boundaries = distributed.get("boundary_diagnostics", [])
    replay = distributed.get("cache_replay", {})
    prompts = distributed.get("prompt_results", [])
    worker_proofs = distributed.get("worker_load_proofs", [])
    quality_gates = bundle.get("quality_gates", {})
    log_validation = distributed.get("log_validation", {})
    cleanup = distributed.get("cleanup", {})

    environment_ok = bool(
        environment.get("python_version", "").startswith("3.11.")
        and environment.get("cuda_available") is True
        and environment.get("gpu", {}).get("model") == "NVIDIA GeForce RTX 5090"
        and environment.get("gpu", {}).get("bf16_supported") is True
        and environment.get("pytorch_version")
        and environment.get("transformers_version")
        and environment.get("safetensors_version")
        and environment.get("required_packages_preserved") is True
    )
    revision = str(manifest.get("model_revision", ""))
    model_revision_ok = bool(
        manifest.get("model_id")
        and re.fullmatch(r"[0-9a-f]{40,64}", revision)
        and manifest.get("architecture") == "Qwen3ForCausalLM"
        and reference.get("model_id") == manifest.get("model_id")
        and reference.get("model_revision") == revision
    )
    stages = manifest.get("stages", [])
    sharding_ok = bool(
        len(stages) == 4
        and shard_validation.get("status") == "PASS"
        and shard_validation.get("every_source_tensor_assigned") is True
        and shard_validation.get("decoder_layers_owned_exactly_once") is True
        and shard_validation.get("stage_hashes_valid") is True
        and shard_validation.get("union_reconstructs_required_state") is True
        and manifest.get("total_sharded_weight_bytes", 0) >= manifest.get("total_weight_bytes", 1)
    )
    stage_isolation_ok = bool(
        len(worker_proofs) == 4
        and distributed.get("one_stage_per_worker") is True
        and distributed.get("complete_stage_coverage") is True
        and distributed.get("model_larger_than_each_worker_limit") is True
        and distributed.get("model_larger_than_each_worker_total_limit") is True
        and all(item.get("proof_verified") is True for item in worker_proofs)
        and all(
            item.get("load_record_checksum_verified") is True
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("load_record_checksum", "")),
            )
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("source_health_proof_checksum", "")),
            )
            and bool(item.get("source_health_proof_signature"))
            for item in worker_proofs
        )
        and all(
            item.get("loaded_complete_source_tensor_set") is False
            and item.get("decoder_layer_count", manifest.get("layer_count", 0))
            < manifest.get("layer_count", 0)
            and item.get("stage_fits_logical_total_memory_limit") is True
            and item.get("full_model_exceeds_logical_total_limit") is True
            for item in worker_proofs
        )
        and coordinator.get("coordinator_model_weight_bytes") == 0
        and coordinator.get("coordinator_full_model_loaded") is False
    )
    reference_process_id = reference.get("process_id")
    worker_process_ids = {
        item.get("process_id") for item in worker_proofs if isinstance(item.get("process_id"), int)
    }
    reference_independent_ok = bool(
        reference.get("phase") == "independent-full-model-reference"
        and reference.get("full_model_loaded") is True
        and reference.get("memory_counted_as_swarm") is False
        and reference.get("greedy") is True
        and reference.get("temperature") == 0
        and reference.get("thinking_enabled") is False
        and reference.get("device") == "cuda"
        and reference.get("dtype") == "bfloat16"
        and isinstance(reference_process_id, int)
        and reference_process_id > 0
        and reference_process_id != coordinator.get("process_id")
        and reference_process_id not in worker_process_ids
        and len(reference.get("results", [])) >= len(prompts)
    )
    copy_evidence_ok = bool(
        len(worker_proofs) == 4
        and all(
            item.get("stage_transfer_metrics", {}).get("operation_count", 0) > 0
            and item.get("stage_transfer_metrics", {}).get(
                "host_to_device_copy_ms",
                0,
            )
            > 0
            and item.get("stage_transfer_metrics", {}).get(
                "device_to_host_copy_ms",
                0,
            )
            > 0
            and item.get("stage_transfer_metrics", {}).get(
                "host_to_device_bytes",
                0,
            )
            > 0
            and item.get("stage_transfer_metrics", {}).get(
                "device_to_host_bytes",
                0,
            )
            > 0
            for item in worker_proofs
        )
    )
    real_execution_ok = bool(
        distributed.get("execution_mode") == "single-host-loopback-real-model"
        and distributed.get("worker_backend") == "torch-cuda"
        and prompts
        and all(item.get("status") == "completed" for item in prompts)
        and reference_independent_ok
        and copy_evidence_ok
    )
    direct_ok = bool(
        transport.get("data_plane") == "direct"
        and transport.get("coordinator_activation_bytes") == 0
        and transport.get("worker_to_worker_activation_bytes", 0) > 0
        and transport.get("peer_streams_created", 0) >= 3
        and transport.get("activation_messages_sent", 0) > 0
        and transport.get("activation_messages_received", 0) > 0
        and transport.get("persistent_streams") is True
        and transport.get("coordinator_relay_fallback") is False
    )
    cache_histories = cache.get("histories", [])
    kv_ok = bool(
        cache_histories
        and cache.get("owned_layer_count") == manifest.get("layer_count")
        and cache.get("active_cache_count_after_completion") == 0
        and cache.get("stale_request_cache_remains") is False
        and all(item.get("owned_layer_count", 0) > 0 for item in cache_histories)
        and all(
            item.get("initialised_layer_count") == item.get("owned_layer_count")
            for item in cache_histories
        )
    )
    boundary_ok = bool(
        boundaries
        and len(boundaries) >= 4
        and all(item.get("within_tolerance") is True for item in boundaries)
        and all(item.get("nan_count") == 0 for item in boundaries)
        and all(item.get("inf_count") == 0 for item in boundaries)
    )
    token_identity_ok = bool(
        prompts
        and all(item.get("token_identity") is True for item in prompts)
        and all(item.get("passed") is True for item in prompts)
    )
    replay_ok = bool(
        replay.get("cache_replay_preserved_output") is True
        and replay.get("replay_input_count", 0) > 0
        and replay.get("replay_bytes", 0) > 0
        and replay.get("tokens_committed_before_failure", 0) > 0
    )
    suite_prompts = [item for item in prompts if item.get("phase") in {"smoke", "suite"}]
    concurrent = [item for item in suite_prompts if item.get("name", "").startswith("concurrent-")]
    prompt_suite_ok = bool(
        len(suite_prompts) >= 9
        and len(concurrent) >= 2
        and all(item.get("passed") is True for item in suite_prompts)
    )
    quality_ok = bool(
        quality_gates.get("required") is not True or quality_gates.get("overall_status") == "PASS"
    )
    logs_and_cleanup_ok = bool(
        log_validation.get("status") == "PASS"
        and log_validation.get("ignored_fatal_exception_count") == 0
        and cleanup.get("all_workers_stopped") is True
        and cleanup.get("all_workers_exited_zero") is True
        and not cleanup.get("stale_worker_processes")
        and cleanup.get("worker_identity_files_removed") is True
        and cleanup.get("worker_shutdown_files_removed") is True
    )
    statuses = {
        "environment_status": _status(environment_ok),
        "model_revision_status": _status(model_revision_ok),
        "sharding_status": _status(sharding_ok),
        "stage_isolation_status": _status(stage_isolation_ok),
        "real_model_execution_status": _status(real_execution_ok),
        "direct_data_plane_status": _status(direct_ok),
        "kv_cache_status": _status(kv_ok),
        "boundary_correctness_status": _status(boundary_ok),
        "token_identity_status": _status(token_identity_ok),
        "cache_replay_status": _status(replay_ok),
        "prompt_suite_status": _status(prompt_suite_ok),
    }
    integrity_ok = (
        all(value == "PASS" for value in statuses.values()) and quality_ok and logs_and_cleanup_ok
    )
    result = {
        "experiment_integrity_status": _status(integrity_ok),
        **statuses,
    }
    result["overall_status"] = _status(all(value == "PASS" for value in result.values()))
    return result
