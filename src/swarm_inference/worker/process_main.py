"""Minimal argparse entrypoint used to launch independently killable workers."""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import suppress

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

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
    parser.add_argument("--identity", required=True)
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
        # PyTorch permits setting the inter-op pool only before it starts. The
        # environment variables above still constrain native libraries.
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
    asyncio.run(
        run_worker(
            coordinator_endpoint=arguments.coordinator,
            listen_endpoint=arguments.listen,
            advertised_endpoint=arguments.advertise,
            backend=Backend(arguments.backend),
            memory_limit_bytes=arguments.memory_limit_bytes,
            identity_path=arguments.identity,
            worker_id=arguments.worker_id,
            model_shard_root=arguments.model_shard_root,
            queue_config=QueueConfig(
                capacity=arguments.queue_capacity,
                max_microbatch_size=arguments.max_microbatch_size,
                max_microbatch_wait_ms=arguments.max_microbatch_wait_ms,
                backpressure_policy=BackpressurePolicy.REJECT,
            ),
            outbound_queue_capacity=arguments.outbound_queue_capacity,
            inbound_queue_capacity=arguments.inbound_queue_capacity,
            max_inflight_operations=arguments.max_inflight_operations,
            reconnect_attempts=arguments.reconnect_attempts,
            reconnect_initial_backoff_ms=arguments.reconnect_initial_backoff_ms,
            reconnect_max_backoff_ms=arguments.reconnect_max_backoff_ms,
        )
    )


if __name__ == "__main__":
    main()
