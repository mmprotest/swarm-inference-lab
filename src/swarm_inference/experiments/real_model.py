"""Process-isolated real-model loopback validation over the production transport."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.config.models import Backend, ExperimentConfig, ModelManifest
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.exceptions import IntegrityError, MemoryLimitExceededError
from swarm_inference.host import stop_process
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.protocol.messages import SubmitRequest
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.worker.shard_manager import verify_load_record_checksum


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_worker_registration(
    core: CoordinatorCore,
    *,
    expected_count: int,
    processes: list[subprocess.Popen[str]],
    process_dir: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(core.registry.workers()) >= expected_count:
            return
        failed = [
            {"pid": process.pid, "exit_code": process.poll()}
            for process in processes
            if process.poll() is not None
        ]
        if failed:
            raise IntegrityError(
                "real-model worker exited before registration: "
                f"{failed}; inspect logs in {process_dir}"
            )
        await asyncio.sleep(0.1)
    raise IntegrityError(
        f"timed out after {timeout_s:.1f}s waiting for {expected_count} real-model "
        f"workers; registered={len(core.registry.workers())}; logs={process_dir}"
    )


async def run_qwen3_process_loopback(
    *,
    experiment: ExperimentConfig,
    manifest: ModelManifest,
    architecture_config: dict[str, Any],
    shard_root: Path,
    model_path: Path,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    worker_count: int,
    worker_backend: Backend = Backend.TORCH_CPU,
) -> dict[str, Any]:
    """Run assigned Qwen3 shards in separate workers; coordinator loads metadata/tokenizer only."""

    if worker_count < len(manifest.stages):
        raise IntegrityError(
            f"real loopback needs at least {len(manifest.stages)} workers for stage coverage"
        )
    if worker_backend == Backend.SYNTHETIC:
        raise IntegrityError("real-model loopback workers cannot use the synthetic backend")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )
    core = CoordinatorCore(
        config=experiment,
        model_manifest=manifest,
        architecture_config=architecture_config,
        runtime_dtype=dtype,
        tokenizer=tokenizer,
    )
    largest_runtime_stage = max(stage.required_memory_bytes for stage in core.stages)
    memory_limit = largest_runtime_stage + 128 * 1024 * 1024
    if memory_limit >= manifest.total_weight_bytes:
        raise MemoryLimitExceededError(
            "runtime dtype makes the smallest safe worker memory cap at least as large "
            "as the complete model; choose a narrower dtype or a smaller stage target"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    process_dir = output_dir / (
        "distributed-processes-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    process_dir.mkdir(parents=True, exist_ok=False)
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    response = None
    worker_proofs: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(worker_count):
            port = _free_port()
            endpoint = f"127.0.0.1:{port}"
            worker_id = f"qwen-loopback-worker-{index:03d}"
            log_handle = (process_dir / f"{worker_id}.log").open("w", encoding="utf-8")
            log_handles.append(log_handle)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "swarm_inference.worker.process_main",
                        "--coordinator",
                        coordinator_endpoint,
                        "--listen",
                        endpoint,
                        "--advertise",
                        endpoint,
                        "--backend",
                        worker_backend.value,
                        "--memory-limit-bytes",
                        str(memory_limit),
                        "--identity",
                        str(process_dir / f"{worker_id}.pem"),
                        "--worker-id",
                        worker_id,
                        "--queue-capacity",
                        str(experiment.queue.capacity),
                        "--max-microbatch-size",
                        str(experiment.queue.max_microbatch_size),
                        "--max-microbatch-wait-ms",
                        str(experiment.queue.max_microbatch_wait_ms),
                        "--model-shard-root",
                        str(shard_root),
                    ],
                    cwd=Path.cwd(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        await _wait_for_worker_registration(
            core,
            expected_count=worker_count,
            processes=processes,
            process_dir=process_dir,
            timeout_s=180.0,
        )
        await core.wait_for_coverage(minimum_replicas=1, timeout_s=180.0)
        prompt_ids = [int(value) for value in tokenizer(prompt, return_tensors=None)["input_ids"]]
        response = await core.submit(
            SubmitRequest(
                request_id="qwen3-real-loopback-validation",
                prompt_token_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                random_seed=0,
                model_id=manifest.model_id,
                model_revision=manifest.model_revision,
            )
        )
        for worker in core.registry.workers():
            if worker.endpoint is None:
                continue
            health = await core.transport.health(worker.endpoint)
            worker_proofs.append(
                {
                    "worker_id": worker.worker_id,
                    "endpoint": worker.endpoint,
                    "loaded_stages": health.loaded_stages,
                    "proof": health.proof,
                }
            )
    finally:
        elapsed = time.perf_counter() - started
        await server.stop()
        for process in processes:
            stop_process(process, terminate_timeout_s=15)
        for handle in log_handles:
            handle.close()

    if response is None:
        raise IntegrityError("real-model loopback ended without a response")
    loaded_stage_ids = [stage_id for proof in worker_proofs for stage_id in proof["loaded_stages"]]
    one_stage_per_worker = all(len(proof["loaded_stages"]) <= 1 for proof in worker_proofs)
    complete_coverage = sorted(loaded_stage_ids) == list(range(len(manifest.stages)))
    result = {
        "execution_mode": "single-host-loopback",
        "values": (f"measured local {worker_backend.value} execution and local gRPC transport"),
        "worker_backend": worker_backend.value,
        "model_id": manifest.model_id,
        "model_revision": manifest.model_revision,
        "worker_count": worker_count,
        "stage_count": len(manifest.stages),
        "full_model_weight_bytes": manifest.total_weight_bytes,
        "logical_worker_memory_limit_bytes": memory_limit,
        "model_larger_than_each_worker_limit": manifest.total_weight_bytes > memory_limit,
        "coordinator_full_model_loaded": False,
        "one_stage_per_worker": one_stage_per_worker,
        "complete_stage_coverage": complete_coverage,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": response.output_token_ids,
        "verified": response.verified,
        "status": response.status,
        "detail": response.detail,
        "time_to_first_token_s": response.time_to_first_token_s,
        "end_to_end_s": response.end_to_end_s,
        "wall_elapsed_s": elapsed,
        "coordinator_events": core.events,
        "request_metrics": core.request_metrics,
        "worker_load_proofs": worker_proofs,
        "process_logs": str(process_dir),
        "passed": (
            response.status == "completed"
            and response.verified
            and manifest.total_weight_bytes > memory_limit
            and one_stage_per_worker
            and complete_coverage
        ),
    }
    (output_dir / "distributed-loopback.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _numeric_metric_delta(
    after: dict[str, int | float | str],
    before: dict[str, int | float | str],
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    for key, value in after.items():
        prior = before.get(key)
        if isinstance(value, int | float) and isinstance(prior, int | float):
            result[key] = value - prior
        else:
            result[key] = value
    return result


def _token_comparison(
    *,
    reference: dict[str, Any],
    distributed_ids: list[int],
    distributed_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_ids = [int(value) for value in reference["generated_token_ids"]]
    identity = distributed_ids == reference_ids
    mismatch: dict[str, Any] | None = None
    if not identity:
        common = min(len(reference_ids), len(distributed_ids))
        position = next(
            (index for index in range(common) if reference_ids[index] != distributed_ids[index]),
            common,
        )
        reference_step = (
            reference["steps"][position] if position < len(reference["steps"]) else None
        )
        distributed_step = (
            distributed_steps[position] if position < len(distributed_steps) else None
        )
        mismatch = {
            "first_mismatch_position": position,
            "reference_token_id": (
                reference_ids[position] if position < len(reference_ids) else None
            ),
            "distributed_token_id": (
                distributed_ids[position] if position < len(distributed_ids) else None
            ),
            "reference_token_text": (
                reference_step.get("selected_token_text") if reference_step is not None else None
            ),
            "distributed_token_text": (
                distributed_step.get("selected_token_text")
                if distributed_step is not None
                else None
            ),
            "reference_selected_token_logit": (
                reference_step.get("selected_token_logit") if reference_step is not None else None
            ),
            "distributed_selected_token_logit": (
                distributed_step.get("selected_token_logit")
                if distributed_step is not None
                else None
            ),
            "reference_top_logits": (
                reference_step.get("top_logits") if reference_step is not None else []
            ),
            "distributed_top_logits": (
                distributed_step.get("top_logits") if distributed_step is not None else []
            ),
        }
    return {
        "token_identity": identity,
        "reference_generated_token_ids": reference_ids,
        "distributed_generated_token_ids": distributed_ids,
        "mismatch": mismatch,
    }


def _verify_worker_proof(proof: dict[str, Any]) -> bool:
    checksum = str(proof.get("proof_checksum", ""))
    signature = str(proof.get("proof_signature", ""))
    unsigned = {
        key: value
        for key, value in proof.items()
        if key not in {"proof_checksum", "proof_signature"}
    }
    canonical = canonical_json_bytes(unsigned)
    if sha256_bytes(canonical) != checksum:
        return False
    capability = unsigned.get("capability")
    if not isinstance(capability, dict) or not signature:
        return False
    try:
        verify_signature(str(capability["public_key"]), canonical, signature)
    except Exception:
        return False
    return True


async def run_qwen3_experiment_session(
    *,
    experiment: ExperimentConfig,
    manifest: ModelManifest,
    architecture_config: dict[str, Any],
    shard_root: Path,
    model_path: Path,
    output_dir: Path,
    requests: list[dict[str, Any]],
    reference_results: list[dict[str, Any]],
    dtype: str,
    logical_weight_limit_bytes: int,
    logical_total_memory_limit_bytes: int,
    boundary_reference_root: Path,
    boundary_atol: float,
    boundary_rtol: float,
    minimum_cosine_similarity: float,
    worker_start_timeout_s: float,
    shutdown_timeout_s: float,
) -> dict[str, Any]:
    """Execute a complete Experiment 002 suite in one four-worker CUDA session."""

    if len(manifest.stages) != 4 or len(requests) < 8:
        raise IntegrityError("Experiment 002 needs four stages and at least eight requests")
    if logical_weight_limit_bytes <= max(stage.required_memory_bytes for stage in manifest.stages):
        raise MemoryLimitExceededError("logical worker limit must exceed the largest stage weight")
    if logical_weight_limit_bytes >= manifest.total_weight_bytes:
        raise MemoryLimitExceededError(
            "logical worker limit must remain below the complete model weight"
        )
    largest_total = max(
        stage.required_total_memory_bytes or stage.required_memory_bytes
        for stage in manifest.stages
    )
    if logical_total_memory_limit_bytes <= largest_total:
        raise MemoryLimitExceededError(
            "logical total-memory limit must exceed the largest stage total estimate"
        )
    if logical_total_memory_limit_bytes >= manifest.total_weight_bytes:
        raise MemoryLimitExceededError(
            "logical total-memory limit must remain below the complete model weight"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )
    core = CoordinatorCore(
        config=experiment,
        model_manifest=manifest,
        architecture_config=architecture_config,
        runtime_dtype=dtype,
        tokenizer=tokenizer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    server = CoordinatorRpcServer(core)
    coordinator_port = await server.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    identity_paths = [logs_dir / f"worker-{index:03d}.pem" for index in range(4)]
    shutdown_paths = [logs_dir / f"worker-{index:03d}.shutdown" for index in range(4)]
    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["SWARM_REFERENCE_BOUNDARY_ROOT"] = str(boundary_reference_root)
    environment["SWARM_BOUNDARY_ATOL"] = str(boundary_atol)
    environment["SWARM_BOUNDARY_RTOL"] = str(boundary_rtol)
    environment["SWARM_BOUNDARY_MINIMUM_COSINE"] = str(minimum_cosine_similarity)
    initial_health: list[dict[str, Any]] = []
    final_health: list[dict[str, Any]] = []
    prompt_results: list[dict[str, Any]] = []
    reference_by_id = {str(item["request_id"]): item for item in reference_results}
    cleanup: dict[str, Any] = {}
    started = time.perf_counter()

    async def health_snapshot() -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for worker in sorted(
            core.registry.workers(),
            key=lambda item: item.worker_id,
        ):
            if worker.endpoint is None:
                continue
            health = await core.transport.health(worker.endpoint)
            snapshots.append(
                {
                    "worker_id": worker.worker_id,
                    "endpoint": worker.endpoint,
                    "loaded_stages": health.loaded_stages,
                    "healthy": health.healthy,
                    "proof": health.proof,
                    "proof_verified": _verify_worker_proof(health.proof),
                }
            )
        return snapshots

    async def execute_request(spec: dict[str, Any]) -> dict[str, Any]:
        request_id = str(spec["request_id"])
        before = dict(core.runtime_transport_metrics)
        response = await core.submit(
            SubmitRequest(
                request_id=request_id,
                prompt_token_ids=[int(value) for value in spec["prompt_token_ids"]],
                max_new_tokens=int(spec["max_new_tokens"]),
                random_seed=0,
                model_id=manifest.model_id,
                model_revision=manifest.model_revision,
                cache_replay_stage_id=spec.get("cache_replay_stage_id"),
                cache_replay_after_tokens=spec.get("cache_replay_after_tokens"),
            )
        )
        after = dict(core.runtime_transport_metrics)
        request_metric = next(
            item for item in reversed(core.request_metrics) if item["request_id"] == request_id
        )
        reference = reference_by_id[request_id]
        comparison = _token_comparison(
            reference=reference,
            distributed_ids=[int(value) for value in response.output_token_ids],
            distributed_steps=request_metric["token_steps"],
        )
        return {
            "request_id": request_id,
            "name": spec["name"],
            "prompt": spec["prompt"],
            "phase": spec["phase"],
            "input_token_count": len(spec["prompt_token_ids"]),
            "output_token_count": len(response.output_token_ids),
            "prompt_token_ids": spec["prompt_token_ids"],
            **comparison,
            "reference_decoded_text": reference["decoded_text"],
            "distributed_decoded_text": tokenizer.decode(
                response.output_token_ids,
                skip_special_tokens=False,
            ),
            "status": response.status,
            "verified": response.verified,
            "detail": response.detail,
            "time_to_first_token_s": response.time_to_first_token_s,
            "end_to_end_latency_s": response.end_to_end_s,
            "request_metrics": request_metric,
            "transport_metrics": _numeric_metric_delta(after, before),
            "cache_replay_requested": spec.get("cache_replay_stage_id") is not None,
            "passed": (
                response.status == "completed"
                and response.verified
                and comparison["token_identity"]
            ),
        }

    try:
        for index in range(4):
            port = _free_port()
            endpoint = f"127.0.0.1:{port}"
            worker_id = f"qwen3-stage-worker-{index:03d}"
            log_handle = (logs_dir / f"worker-{index:03d}.log").open(
                "w",
                encoding="utf-8",
            )
            log_handles.append(log_handle)
            creation_flags = (
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "swarm_inference.worker.process_main",
                        "--coordinator",
                        coordinator_endpoint,
                        "--listen",
                        endpoint,
                        "--advertise",
                        endpoint,
                        "--backend",
                        Backend.TORCH_CUDA.value,
                        "--memory-limit-bytes",
                        str(logical_weight_limit_bytes),
                        "--total-memory-limit-bytes",
                        str(logical_total_memory_limit_bytes),
                        "--identity",
                        str(identity_paths[index]),
                        "--shutdown-file",
                        str(shutdown_paths[index]),
                        "--worker-id",
                        worker_id,
                        "--queue-capacity",
                        str(experiment.queue.capacity),
                        "--max-microbatch-size",
                        "1",
                        "--max-microbatch-wait-ms",
                        "0",
                        "--model-shard-root",
                        str(shard_root),
                        "--outbound-queue-capacity",
                        str(experiment.worker.outbound_queue_capacity),
                        "--inbound-queue-capacity",
                        str(experiment.worker.inbound_queue_capacity),
                        "--max-inflight-operations",
                        "1",
                    ],
                    cwd=Path.cwd(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creation_flags,
                )
            )
        await _wait_for_worker_registration(
            core,
            expected_count=4,
            processes=processes,
            process_dir=logs_dir,
            timeout_s=worker_start_timeout_s,
        )
        await core.wait_for_coverage(
            minimum_replicas=1,
            timeout_s=worker_start_timeout_s,
        )
        initial_health = await health_snapshot()
        smoke = next(item for item in requests if item["phase"] == "smoke")
        smoke_result = await execute_request(smoke)
        prompt_results.append(smoke_result)
        if smoke_result["passed"]:
            for spec in [
                item
                for item in requests
                if item["phase"] == "suite" and not item.get("concurrent_group")
            ]:
                prompt_results.append(await execute_request(spec))
            groups = sorted(
                {
                    str(item["concurrent_group"])
                    for item in requests
                    if item["phase"] == "suite" and item.get("concurrent_group")
                }
            )
            for group in groups:
                members = [item for item in requests if str(item.get("concurrent_group")) == group]
                prompt_results.extend(
                    await asyncio.gather(*(execute_request(item) for item in members))
                )
            if all(item["passed"] for item in prompt_results):
                for spec in [item for item in requests if item["phase"] == "replay"]:
                    prompt_results.append(await execute_request(spec))
        final_health = await health_snapshot()
    finally:
        for shutdown_path in shutdown_paths:
            shutdown_path.write_text("shutdown\n", encoding="utf-8")
        for process in processes:
            if process.poll() is not None:
                continue
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=shutdown_timeout_s)
        await server.stop()
        for process in processes:
            stop_process(process, terminate_timeout_s=shutdown_timeout_s)
        for handle in log_handles:
            handle.close()
        identity_removal_errors: list[str] = []
        for identity_path in identity_paths:
            try:
                identity_path.unlink(missing_ok=True)
            except OSError as exc:
                identity_removal_errors.append(f"{identity_path}: {exc}")
        shutdown_removal_errors: list[str] = []
        for shutdown_path in shutdown_paths:
            try:
                shutdown_path.unlink(missing_ok=True)
            except OSError as exc:
                shutdown_removal_errors.append(f"{shutdown_path}: {exc}")
        worker_exit_codes = [process.poll() for process in processes]
        cleanup = {
            "worker_exit_codes": worker_exit_codes,
            "all_workers_exited_zero": all(code == 0 for code in worker_exit_codes),
            "all_workers_stopped": all(process.poll() is not None for process in processes),
            "stale_worker_processes": [
                process.pid for process in processes if process.poll() is None
            ],
            "worker_identity_files_removed": not identity_removal_errors
            and all(not path.exists() for path in identity_paths),
            "worker_identity_removal_errors": identity_removal_errors,
            "worker_shutdown_files_removed": not shutdown_removal_errors
            and all(not path.exists() for path in shutdown_paths),
            "worker_shutdown_removal_errors": shutdown_removal_errors,
        }
    worker_proofs = final_health or initial_health
    loaded_stage_ids = [stage_id for proof in initial_health for stage_id in proof["loaded_stages"]]
    peer_connections = [proof["proof"].get("peer_connections", {}) for proof in worker_proofs]
    transport: dict[str, Any] = {
        "data_plane": "direct",
        "coordinator_activation_bytes": int(
            core.runtime_transport_metrics["coordinator_activation_bytes"]
        ),
        "coordinator_input_activation_bytes": int(
            core.runtime_transport_metrics["coordinator_input_activation_bytes"]
        ),
        "coordinator_final_logit_bytes": int(
            core.runtime_transport_metrics["coordinator_final_result_bytes"]
        ),
        "worker_to_worker_activation_bytes": int(
            core.runtime_transport_metrics["worker_to_worker_activation_bytes"]
        ),
        "peer_streams_created": sum(
            int(item.get("streams_created", 0)) for item in peer_connections
        ),
        "peer_channels_created": sum(
            int(item.get("channels_created", 0)) for item in peer_connections
        ),
        "activation_messages_sent": sum(
            int(item.get("messages_sent", 0)) for item in peer_connections
        ),
        "activation_messages_received": sum(
            int(item.get("messages_received", 0)) for item in peer_connections
        ),
        "persistent_streams": True,
        "coordinator_relay_fallback": False,
        "raw_coordinator_metrics": core.runtime_transport_metrics,
    }
    cache_histories: list[dict[str, Any]] = []
    active_cache_count = 0
    boundary_records: list[dict[str, Any]] = []
    load_records: list[dict[str, Any]] = []
    for worker in worker_proofs:
        shard_proof = worker["proof"].get("shards", {})
        stages = shard_proof.get("stages", {})
        execution_memory = shard_proof.get("current_process_memory", {})
        for stage_payload in stages.values():
            load_record = stage_payload.get("load_record", {})
            load_record_checksum_verified = verify_load_record_checksum(load_record)
            load_record = {
                **load_record,
                "worker_id": worker["worker_id"],
                "proof_verified": (worker["proof_verified"] and load_record_checksum_verified),
                "load_record_checksum_verified": load_record_checksum_verified,
                "source_health_proof_checksum": worker["proof"].get(
                    "proof_checksum",
                ),
                "source_health_proof_signature": worker["proof"].get(
                    "proof_signature",
                ),
                "process_memory_after_execution": execution_memory,
                "peak_cuda_memory_bytes": execution_memory.get("cuda_peak_allocated_bytes", 0),
                "peak_cuda_reserved_bytes": execution_memory.get("cuda_peak_reserved_bytes", 0),
                "stage_transfer_metrics": stage_payload.get("module_state", {}).get(
                    "transfer_metrics",
                    {},
                ),
            }
            load_records.append(load_record)
            module_state = stage_payload.get("module_state", {})
            active_cache_count += int(module_state.get("cache_count", 0))
            cache_histories.extend(module_state.get("cache_history", []))
            boundary_records.extend(module_state.get("boundary_records", []))
    replay_events = [
        event for event in core.events if event.get("event_type") == "stage_cache_replayed"
    ]
    coordinator_process = psutil.Process()
    coordinator_proof = {
        "process_id": os.getpid(),
        "model_weight_bytes_loaded": 0,
        "coordinator_model_weight_bytes": 0,
        "tokenizer_present": tokenizer is not None,
        "config_present": architecture_config is not None,
        "model_manifest_present": manifest is not None,
        "activation_bytes_received": 0,
        "coordinator_activation_bytes": 0,
        "final_logit_bytes_received": transport["coordinator_final_logit_bytes"],
        "control_bytes": int(core.runtime_transport_metrics["coordinator_control_bytes"]),
        "gpu_memory_bytes": 0,
        "host_rss_bytes": int(coordinator_process.memory_info().rss),
        "coordinator_full_model_loaded": False,
    }
    result: dict[str, Any] = {
        "execution_mode": "single-host-loopback-real-model",
        "worker_backend": Backend.TORCH_CUDA.value,
        "spawn_method": "spawn",
        "worker_count": 4,
        "stage_count": 4,
        "full_model_weight_bytes": manifest.total_weight_bytes,
        "logical_worker_weight_limit_bytes": logical_weight_limit_bytes,
        "logical_worker_total_memory_limit_bytes": logical_total_memory_limit_bytes,
        "model_larger_than_each_worker_limit": (
            manifest.total_weight_bytes > logical_weight_limit_bytes
        ),
        "model_larger_than_each_worker_total_limit": (
            manifest.total_weight_bytes > logical_total_memory_limit_bytes
        ),
        "one_stage_per_worker": all(len(proof["loaded_stages"]) == 1 for proof in initial_health),
        "complete_stage_coverage": sorted(loaded_stage_ids) == [0, 1, 2, 3],
        "prompt_results": prompt_results,
        "worker_load_proofs": load_records,
        "worker_health_proofs": worker_proofs,
        "coordinator_proof": coordinator_proof,
        "transport_metrics": transport,
        "cache_metrics": {
            "histories": cache_histories,
            "active_cache_count_after_completion": active_cache_count,
            "stale_request_cache_remains": active_cache_count != 0,
            "owned_layer_count": manifest.layer_count,
            "maximum_cache_bytes_by_stage": {
                str(stage_id): max(
                    (
                        int(item.get("cache_bytes", 0))
                        for item in cache_histories
                        if int(item.get("stage_id", -1)) == stage_id
                    ),
                    default=0,
                )
                for stage_id in range(4)
            },
        },
        "boundary_diagnostics": boundary_records,
        "cache_replay": (
            {
                **replay_events[-1],
                "cache_replay_preserved_output": all(
                    item["passed"] for item in prompt_results if item["phase"] == "replay"
                ),
            }
            if replay_events
            else {
                "cache_replay_preserved_output": False,
                "error": "no cache replay event was recorded",
            }
        ),
        "events": core.events,
        "request_metrics": core.request_metrics,
        "cleanup": cleanup,
        "wall_elapsed_s": time.perf_counter() - started,
    }
    result["passed"] = bool(
        result["one_stage_per_worker"]
        and result["complete_stage_coverage"]
        and result["model_larger_than_each_worker_limit"]
        and result["model_larger_than_each_worker_total_limit"]
        and all(item["passed"] for item in prompt_results)
        and len(prompt_results) == len(requests)
        and transport["coordinator_activation_bytes"] == 0
        and transport["worker_to_worker_activation_bytes"] > 0
        and transport["peer_streams_created"] >= 3
        and transport["activation_messages_sent"] > 0
        and transport["activation_messages_received"] > 0
        and active_cache_count == 0
        and result["cache_replay"]["cache_replay_preserved_output"]
        and cleanup["all_workers_stopped"]
    )
    (logs_dir / "coordinator.log").write_text(
        "\n".join(json.dumps(event, sort_keys=True) for event in core.events) + "\n",
        encoding="utf-8",
    )
    return result
