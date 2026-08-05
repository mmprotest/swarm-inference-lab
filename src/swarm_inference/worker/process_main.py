"""Minimal argparse entrypoint used to launch independently killable workers."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

_PYTHON_ENTRY_NS = time.monotonic_ns()


def _write_managed_process_lifecycle_record(
    lifecycle_path: str | Path,
    payload: dict[str, Any],
) -> Path:
    """Write one process-owned record without sharing an append target.

    Multiple independently launched workers can finish at the same instant.
    A unique sidecar makes each write single-owner and lets the parent aggregate
    the records after every child has exited.
    """

    base = Path(lifecycle_path).expanduser().resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    record_path = base.with_name(f"{base.name}.worker-{os.getpid()}-{uuid4().hex}.json")
    record_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return record_path


def _early_lifecycle_event(event_name: str, timestamp_ns: int | None = None) -> None:
    """Emit before importing the worker stack so import time is directly measured."""

    path = os.environ.get("SWARM_LIFECYCLE_FILE")
    if not path:
        return
    timestamp = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
    origin = int(os.environ.get("SWARM_EXPERIMENT_ORIGIN_NS", timestamp))
    payload = {
        "experiment_id": os.environ.get("SWARM_EXPERIMENT_ID", "unknown"),
        "worker_id": os.environ.get("SWARM_WORKER_ID", "unknown"),
        "stage_id": int(os.environ.get("SWARM_STAGE_ID", "-1")),
        "process_id": os.getpid(),
        "monotonic_timestamp_ns": timestamp,
        "experiment_elapsed_ns": timestamp - origin,
        "wall_clock_utc": datetime.now(UTC).isoformat(),
        "event_name": event_name,
    }
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


_early_lifecycle_event("python_entry_started", _PYTHON_ENTRY_NS)
_early_lifecycle_event("python_imports_started")

import argparse  # noqa: E402
import asyncio  # noqa: E402
from contextlib import suppress  # noqa: E402
from pathlib import Path  # noqa: E402

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")
# CUDA kernels used by the independent reference and the stage workers must
# make the same deterministic algorithm choices across separate processes.
# This variable has to be present before the CUDA runtime is initialised.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from swarm_inference.config.models import (  # noqa: E402
    Backend,
    BackpressurePolicy,
    QueueConfig,
)
from swarm_inference.runtime.telemetry import (  # noqa: E402
    configure_lifecycle_observer,
    lifecycle_observer,
    lifecycle_observer_from_environment,
)
from swarm_inference.worker.service import run_worker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="swarm-inference-lab worker process")
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--listen", required=True)
    parser.add_argument("--advertise", required=True)
    parser.add_argument("--backend", choices=[item.value for item in Backend], required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--total-memory-limit-bytes", type=int)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--trusted-coordinator-fingerprint")
    parser.add_argument("--shutdown-file")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--queue-capacity", type=int, default=256)
    parser.add_argument("--max-microbatch-size", type=int, default=1)
    parser.add_argument("--max-microbatch-wait-ms", type=float, default=0.0)
    parser.add_argument("--model-shard-root")
    parser.add_argument("--cpu-affinity", type=int)
    parser.add_argument("--outbound-queue-capacity", type=int, default=1024)
    parser.add_argument("--inbound-queue-capacity", type=int, default=1024)
    parser.add_argument("--max-inflight-operations", type=int, default=256)
    parser.add_argument("--reconnect-attempts", type=int, default=5)
    parser.add_argument("--reconnect-initial-backoff-ms", type=float, default=25.0)
    parser.add_argument("--reconnect-max-backoff-ms", type=float, default=1000.0)
    parser.add_argument("--stage-local-warmup", action="store_true")
    parser.add_argument("--warmup-sequence-length", type=int, default=128)
    parser.add_argument("--stage-runtime", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-listen")
    parser.add_argument("--data-advertise")
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-cache-dir")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--max-stage-sessions", type=int, default=256)
    arguments = parser.parse_args()
    if arguments.stage_runtime and (
        not arguments.data_listen or not arguments.data_advertise or not arguments.device
    ):
        parser.error("--stage-runtime requires --data-listen, --data-advertise, and --device")
    if arguments.stage_runtime and not arguments.trusted_coordinator_fingerprint:
        parser.error("--stage-runtime requires --trusted-coordinator-fingerprint")
    if not arguments.stage_runtime and any(
        (
            arguments.data_listen,
            arguments.data_advertise,
            arguments.device,
            arguments.model_cache_dir,
            arguments.allow_model_download,
        )
    ):
        parser.error("stage data, device, and download options require --stage-runtime")
    configure_lifecycle_observer(lifecycle_observer_from_environment())
    recorder = lifecycle_observer()
    if arguments.stage_local_warmup:
        os.environ["SWARM_STAGE_LOCAL_WARMUP"] = "1"
    else:
        os.environ["SWARM_STAGE_LOCAL_WARMUP"] = "0"
    os.environ["SWARM_WARMUP_SEQUENCE_LENGTH"] = str(arguments.warmup_sequence_length)
    if arguments.cpu_affinity is not None:
        try:
            import psutil

            psutil.Process().cpu_affinity([arguments.cpu_affinity])
        except (AttributeError, OSError, ValueError, psutil.Error):
            pass
    torch_module: Any = None
    if Backend(arguments.backend) != Backend.SYNTHETIC:
        import torch as imported_torch

        torch_module = imported_torch
        torch_module.set_num_threads(1)
        if Backend(arguments.backend) == Backend.TORCH_CUDA:
            torch_module.use_deterministic_algorithms(True)
            torch_module.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch_module.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # PyTorch permits setting the inter-op pool only before it starts. The
        # environment variables above still constrain native libraries.
        with suppress(RuntimeError):
            torch_module.set_num_interop_threads(1)
    _early_lifecycle_event("python_imports_completed")
    if recorder is not None and Backend(arguments.backend) == Backend.TORCH_CUDA:
        cuda_started = time.monotonic_ns()
        recorder.emit("cuda_initialisation_started", monotonic_ns=cuda_started)
        try:
            assert torch_module is not None
            cuda_device = torch_module.device(arguments.device or "cuda")
            torch_module.cuda.init()
            marker = torch_module.zeros(1, dtype=torch_module.float32, device=cuda_device)
            marker.add_(1)
            torch_module.cuda.synchronize(cuda_device)
            memory = {
                "torch_cuda_allocated_bytes": int(torch_module.cuda.memory_allocated(cuda_device)),
                "torch_cuda_reserved_bytes": int(torch_module.cuda.memory_reserved(cuda_device)),
            }
            completed = time.monotonic_ns()
            recorder.emit(
                "cuda_initialisation_completed",
                monotonic_ns=completed,
                duration_ns=completed - cuda_started,
                memory_metrics=memory,
                details={"warmup_level": "cuda-context", "minimal_cuda_operation": True},
            )
            del marker
        except Exception as exc:
            completed = time.monotonic_ns()
            recorder.emit(
                "cuda_initialisation_completed",
                monotonic_ns=completed,
                duration_ns=completed - cuda_started,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
    print(
        json.dumps(
            {
                "event": "worker_process_started",
                "worker_id": arguments.worker_id,
                "process_id": os.getpid(),
                "backend": arguments.backend,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    async def run_until_shutdown() -> None:
        stop_event = asyncio.Event() if arguments.shutdown_file else None
        shutdown_watcher: asyncio.Task[None] | None = None
        if stop_event is not None:
            shutdown_path = Path(arguments.shutdown_file)

            async def watch_shutdown_file() -> None:
                while not shutdown_path.exists():
                    await asyncio.sleep(0.1)
                stop_event.set()

            shutdown_watcher = asyncio.create_task(
                watch_shutdown_file(),
                name=f"shutdown-file:{arguments.worker_id}",
            )
        try:
            await run_worker(
                coordinator_endpoint=arguments.coordinator,
                listen_endpoint=arguments.listen,
                advertised_endpoint=arguments.advertise,
                backend=Backend(arguments.backend),
                memory_limit_bytes=arguments.memory_limit_bytes,
                identity_path=arguments.identity,
                total_memory_limit_bytes=arguments.total_memory_limit_bytes,
                worker_id=arguments.worker_id,
                model_shard_root=arguments.model_shard_root,
                queue_config=QueueConfig(
                    capacity=arguments.queue_capacity,
                    max_microbatch_size=arguments.max_microbatch_size,
                    max_microbatch_wait_ms=arguments.max_microbatch_wait_ms,
                    backpressure_policy=BackpressurePolicy.REJECT,
                ),
                stop_event=stop_event,
                outbound_queue_capacity=arguments.outbound_queue_capacity,
                inbound_queue_capacity=arguments.inbound_queue_capacity,
                max_inflight_operations=arguments.max_inflight_operations,
                reconnect_attempts=arguments.reconnect_attempts,
                reconnect_initial_backoff_ms=arguments.reconnect_initial_backoff_ms,
                reconnect_max_backoff_ms=arguments.reconnect_max_backoff_ms,
                stage_runtime_enabled=arguments.stage_runtime,
                data_listen_endpoint=arguments.data_listen,
                data_advertised_endpoint=arguments.data_advertise,
                device=arguments.device,
                dtype=arguments.dtype,
                model_cache_dir=arguments.model_cache_dir,
                allow_model_download=arguments.allow_model_download,
                max_stage_sessions=arguments.max_stage_sessions,
                trusted_coordinator_fingerprint=arguments.trusted_coordinator_fingerprint,
            )
        finally:
            if shutdown_watcher is not None:
                shutdown_watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await shutdown_watcher

    clean_shutdown = False
    try:
        asyncio.run(run_until_shutdown())
        clean_shutdown = True
    finally:
        if recorder is not None:
            recorder.emit("worker_shutdown_completed")
        print(
            json.dumps(
                {
                    "event": "worker_process_stopped",
                    "worker_id": arguments.worker_id,
                    "process_id": os.getpid(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        lifecycle_path = os.environ.get("SWARM_PROCESS_LIFECYCLE_LOG")
        if lifecycle_path:
            lifecycle_payload = {
                "schema_version": 1,
                "cluster_id": f"worker-entry:{os.getpid()}",
                "processes": [
                    {
                        "role": arguments.worker_id,
                        "pid": os.getpid(),
                        "exit_code": 0 if clean_shutdown else 1,
                        "expected_exit_reason": None,
                        "graceful_shutdown_count": int(clean_shutdown),
                        "unexpected_terminate_count": 0,
                        "unexpected_kill_count": 0,
                        "expected_terminate_count": 0,
                        "expected_kill_count": 0,
                    }
                ],
                "graceful_shutdown_count": int(clean_shutdown),
                "unexpected_terminate_count": 0,
                "unexpected_kill_count": 0,
                "expected_terminate_count": 0,
                "expected_kill_count": 0,
                "leaked_process_count": 0,
            }
            _write_managed_process_lifecycle_record(lifecycle_path, lifecycle_payload)


if __name__ == "__main__":
    main()
