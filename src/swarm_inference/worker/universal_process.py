"""Standalone Universal Worker process used by heterogeneous experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
from pathlib import Path

from swarm_inference.backends.torch_rank import TorchRankAdapter
from swarm_inference.worker.abi import (
    WorkerCapabilities,
    WorkerIdentity,
    WorkerProtocolVersion,
)
from swarm_inference.worker.universal import UniversalWorkerServer


def _memory_bytes() -> tuple[int, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except (ImportError, OSError):
        return 8 * 1024**3, 4 * 1024**3


def _cpu_features() -> list[str]:
    features: list[str] = []
    try:
        import torch

        capability = torch.backends.cpu.get_cpu_capability()
        features.append(str(capability).lower())
        if bool(torch.backends.mkldnn.enabled):
            features.append("mkldnn")
    except (AttributeError, ImportError):
        pass
    return sorted(set(features))


def discover_capabilities(device: str) -> WorkerCapabilities:
    total_memory, available_memory = _memory_bytes()
    logical = os.cpu_count() or 1
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
    except (ImportError, OSError):
        physical = logical
    accelerator_type: str | None = None
    accelerator_model: str | None = None
    accelerator_memory = 0
    maximum_weight = int(available_memory * 0.75)
    features = ["stage_local_kv", "canonical_safetensors", "exact_greedy"]
    if device == "cuda":
        import torch

        accelerator_type = "cuda"
        accelerator_model = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        accelerator_memory = int(total)
        maximum_weight = int(free * 0.75)
        features.extend(["gpu_resident_execution", "cuda_events"])
    return WorkerCapabilities(
        architecture=platform.machine(),
        operating_system=platform.platform(),
        cpu_model=platform.processor() or "unknown",
        physical_cpu_cores=physical,
        logical_cpu_cores=logical,
        cpu_features=_cpu_features(),
        accelerator_type=accelerator_type,
        accelerator_model=accelerator_model,
        accelerator_memory_bytes=accelerator_memory,
        system_memory_bytes=total_memory,
        supported_weight_formats=["safetensors", "microshard-safetensors"],
        supported_activation_dtypes=["bfloat16", "float16", "float32", "int8"],
        supported_cache_dtypes=["bfloat16", "float16"],
        supported_collectives=[],
        maximum_weight_bytes=maximum_weight,
        maximum_cache_bytes=max(1, int(available_memory * 0.20)),
        maximum_batch_size=1,
        maximum_context_length=4096,
        measured_network_upload_bps=0.0,
        measured_network_download_bps=0.0,
        coordinator_latency_ms=0.0,
        backend_features=features,
    )


async def _run(arguments: argparse.Namespace) -> None:
    capabilities = discover_capabilities(arguments.device)
    adapter = TorchRankAdapter.from_microshard(
        partition_root=arguments.partition,
        stage_id=arguments.stage_id,
        device=arguments.device,
        capabilities=capabilities,
        partition_hash=arguments.partition_hash,
        dtype=arguments.dtype,
        warmup_sequence_length=arguments.warmup_sequence_length,
    )
    identity = WorkerIdentity(
        worker_id=arguments.worker_id,
        node_id=socket.gethostname(),
        public_key="local-experiment-ephemeral",
        backend_id=adapter.backend_id,
        protocol_version=WorkerProtocolVersion(
            major=1,
            minor=0,
            capabilities={
                "jobs",
                "cancel",
                "heartbeat",
                "shard-hash",
                "clean-shutdown",
            },
        ),
    )
    server = UniversalWorkerServer(
        adapter=adapter,
        identity=identity,
        host=arguments.host,
        port=arguments.port,
    )
    host, port = await server.start()
    ready = {
        "status": "ready",
        "host": host,
        "port": port,
        "worker_id": arguments.worker_id,
        "backend_id": adapter.backend_id,
        "stage_id": arguments.stage_id,
        "pid": os.getpid(),
        "identity": identity.model_dump(mode="json"),
        "capabilities": capabilities.model_dump(mode="json"),
        "benchmark": adapter.benchmark_profile().model_dump(mode="json"),
        "shard_hash": adapter.shard_hash,
    }
    arguments.ready_file.parent.mkdir(parents=True, exist_ok=True)
    arguments.ready_file.write_text(
        json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(ready, sort_keys=True), flush=True)
    await server.serve_until_shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--partition-hash", required=True)
    parser.add_argument("--stage-id", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--warmup-sequence-length", type=int, default=8)
    arguments = parser.parse_args()
    asyncio.run(_run(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
