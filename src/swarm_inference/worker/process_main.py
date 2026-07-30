"""Minimal argparse entrypoint used to launch independently killable workers."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path

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
    arguments = parser.parse_args()
    if arguments.cpu_affinity is not None:
        try:
            import psutil

            psutil.Process().cpu_affinity([arguments.cpu_affinity])
        except (AttributeError, OSError, ValueError, psutil.Error):
            pass
    if Backend(arguments.backend) != Backend.SYNTHETIC:
        import torch

        torch.set_num_threads(1)
        if Backend(arguments.backend) == Backend.TORCH_CUDA:
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # PyTorch permits setting the inter-op pool only before it starts. The
        # environment variables above still constrain native libraries.
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
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
            )
        finally:
            if shutdown_watcher is not None:
                shutdown_watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await shutdown_watcher

    try:
        asyncio.run(run_until_shutdown())
    finally:
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


if __name__ == "__main__":
    main()
