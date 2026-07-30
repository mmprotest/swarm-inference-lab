"""Minimal argparse entrypoint used to launch independently killable workers."""

from __future__ import annotations

import argparse
import asyncio

from swarm_inference.config.models import Backend, BackpressurePolicy, QueueConfig
from swarm_inference.worker.service import run_worker


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
    arguments = parser.parse_args()
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
        )
    )


if __name__ == "__main__":
    main()
