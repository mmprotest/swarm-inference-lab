"""LEGACY_FROZEN process entry point retained for Experiment 010 reproduction."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
from pathlib import Path

from swarm_inference.experiments.experiment_010.worker import (
    ExpertUniversalAdapter,
    ExpertWorkerRuntime,
    ExpertWorkerServer,
)
from swarm_inference.worker.abi import WorkerIdentity, WorkerProtocolVersion
from swarm_inference.worker.universal import UniversalWorkerServer


async def _run(configuration_path: Path, ready_path: Path) -> None:
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    runtime = ExpertWorkerRuntime(configuration)
    server = ExpertWorkerServer(
        runtime,
        host=str(configuration.get("host", "127.0.0.1")),
        port=int(configuration.get("port", 0)),
    )
    host, port = await server.start()
    data_endpoint = f"{host}:{port}"
    adapter = ExpertUniversalAdapter(runtime, data_endpoint=data_endpoint)
    identity = WorkerIdentity(
        worker_id=runtime.worker_id,
        node_id=socket.gethostname(),
        public_key=("local-ephemeral-sha256:" + hashlib.sha256(runtime.secret).hexdigest()),
        backend_id=adapter.backend_id,
        protocol_version=WorkerProtocolVersion(
            major=1,
            minor=1,
            capabilities={
                "jobs",
                "cancel",
                "heartbeat",
                "clean-shutdown",
                "moe-expert",
                "direct-expert-data-plane",
            },
        ),
    )
    universal = UniversalWorkerServer(
        adapter=adapter,
        identity=identity,
        host=str(configuration.get("host", "127.0.0.1")),
        port=0,
    )
    control_host, control_port = await universal.start()
    control_endpoint = f"{control_host}:{control_port}"
    runtime.control_endpoint = control_endpoint
    temporary = ready_path.with_suffix(f".{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "endpoint": data_endpoint,
                "control_endpoint": control_endpoint,
                "identity": identity.model_dump(mode="json"),
                "capabilities": adapter.capabilities().model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, ready_path)
    try:
        await runtime._shutdown.wait()
    finally:
        await server.close()
        await universal.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(_run(arguments.config, arguments.ready))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
