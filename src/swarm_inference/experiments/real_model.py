"""Process-isolated real-model loopback validation over the production transport."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.config.models import Backend, ExperimentConfig, ModelManifest
from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer
from swarm_inference.exceptions import IntegrityError, MemoryLimitExceededError
from swarm_inference.host import stop_process
from swarm_inference.protocol.messages import SubmitRequest


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
