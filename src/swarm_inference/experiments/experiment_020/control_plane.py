"""Faithful lightweight E021 controller and worker-daemon dry runs."""

from __future__ import annotations

import asyncio
import json
import shutil
import statistics
import time
from dataclasses import dataclass
from typing import Any

import psutil

from .transport import (
    Frame,
    MessageType,
    generate_run_tls_material,
    new_run_credential,
    read_frame,
    security_receipt,
    tls_contexts,
    write_frame,
)


@dataclass(frozen=True, slots=True)
class LogicalWorker:
    worker_id: str
    pod_id: str
    gpu_index: int
    layers: tuple[int, ...]


class SwarmHostAgent:
    """Lifecycle owner for explicit per-GPU workers; never a compute resource."""

    def __init__(self, pod_id: str, workers: list[LogicalWorker]) -> None:
        if not workers or any(worker.pod_id != pod_id for worker in workers):
            raise ValueError("host agent workers must belong to one pod")
        if len({worker.gpu_index for worker in workers}) != len(workers):
            raise ValueError("one explicit worker is required per GPU")
        self.pod_id = pod_id
        self.workers = tuple(workers)
        self.running: set[str] = set()

    @property
    def is_compute_resource(self) -> bool:
        return False

    def start(self) -> None:
        self.running = {worker.worker_id for worker in self.workers}

    def health(self) -> dict[str, bool]:
        return {worker.worker_id: worker.worker_id in self.running for worker in self.workers}

    def stop(self) -> None:
        self.running.clear()


def logical_topology(worker_count: int, workers_per_pod: int = 8) -> list[LogicalWorker]:
    if worker_count <= 0 or workers_per_pod <= 0:
        raise ValueError("positive topology required")
    workers: list[LogicalWorker] = []
    pod_count = (worker_count + workers_per_pod - 1) // workers_per_pod
    for index in range(worker_count):
        pod = index // workers_per_pod
        start = (93 * pod) // pod_count
        stop = (93 * (pod + 1)) // pod_count
        workers.append(
            LogicalWorker(
                worker_id=f"pod-{pod:03d}.worker-{index % workers_per_pod:02d}",
                pod_id=f"pod-{pod:03d}",
                gpu_index=index % workers_per_pod,
                layers=tuple(range(start, stop)),
            )
        )
    return workers


async def _run(worker_count: int, workers_per_pod: int) -> dict[str, Any]:
    topology = logical_topology(worker_count, workers_per_pod)
    material = generate_run_tls_material()
    credential = new_run_credential()
    server_context, client_context = tls_contexts(material)
    registered: dict[str, asyncio.StreamWriter] = {}
    readers: dict[str, asyncio.StreamReader] = {}
    registration_complete = asyncio.Event()
    queue_latencies: list[float] = []
    message_count = 0
    failures: list[str] = []
    lock = asyncio.Lock()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal message_count
        try:
            frame = await read_frame(reader, credential)
            message_count += 1
            if frame.message_type is not MessageType.REGISTER:
                raise RuntimeError("first worker frame must register")
            async with lock:
                registered[frame.worker_id] = writer
                readers[frame.worker_id] = reader
                if len(registered) == worker_count:
                    registration_complete.set()
            await write_frame(
                writer,
                Frame(MessageType.REGISTERED, frame.request_id, 0, frame.worker_id, "registered"),
                credential,
            )
            message_count += 1
            await asyncio.wait_for(registration_complete.wait(), timeout=60.0)
            enqueued = time.perf_counter_ns()
            worker = next(item for item in topology if item.worker_id == frame.worker_id)
            payload = json.dumps(
                {
                    "operator": "expert_stripe_local_top16_accumulation",
                    "layers": worker.layers,
                    "routes": list(range(16)),
                    "network_outputs": 1,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            await write_frame(
                writer,
                Frame(
                    MessageType.EXECUTE_SHARD,
                    "e021-dry-run-request",
                    0,
                    frame.worker_id,
                    "state-0",
                    payload,
                ),
                credential,
            )
            message_count += 1
            result = await read_frame(reader, credential)
            message_count += 1
            queue_latencies.append((time.perf_counter_ns() - enqueued) / 1e6)
            if result.message_type is not MessageType.SHARD_RESULT:
                raise RuntimeError("worker did not return shard result")
            await write_frame(
                writer,
                Frame(MessageType.SHUTDOWN, result.request_id, 0, frame.worker_id, "done"),
                credential,
            )
            message_count += 1
        except Exception as exc:  # receipt records the exact daemon failure
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(
        handle,
        "127.0.0.1",
        0,
        ssl=server_context,
        backlog=max(128, worker_count),
        limit=MAX_STREAM_LIMIT,
    )
    port = int(server.sockets[0].getsockname()[1])

    async def worker_daemon(worker: LogicalWorker) -> None:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            port,
            ssl=client_context,
            server_hostname="localhost",
            limit=MAX_STREAM_LIMIT,
        )
        await write_frame(
            writer,
            Frame(MessageType.REGISTER, "register", 0, worker.worker_id, "initial"),
            credential,
        )
        registered_frame = await read_frame(reader, credential)
        if registered_frame.message_type is not MessageType.REGISTERED:
            raise RuntimeError("registration rejected")
        execute = await read_frame(reader, credential)
        if execute.message_type is not MessageType.EXECUTE_SHARD:
            raise RuntimeError("unexpected controller message")
        body = json.loads(execute.payload)
        if body["network_outputs"] != 1 or len(body["routes"]) != 16:
            raise RuntimeError("central per-expert RPC architecture detected")
        await write_frame(
            writer,
            Frame(
                MessageType.SHARD_RESULT,
                execute.request_id,
                execute.chunk_id,
                worker.worker_id,
                execute.state_id,
                b"partial-latent-vector",
            ),
            credential,
        )
        shutdown = await read_frame(reader, credential)
        if shutdown.message_type is not MessageType.SHUTDOWN:
            raise RuntimeError("clean shutdown not received")
        writer.close()
        await writer.wait_closed()

    process = psutil.Process()
    cpu_before = process.cpu_times()
    memory_before = process.memory_info().rss
    started = time.perf_counter_ns()
    async with server:
        await asyncio.wait_for(
            asyncio.gather(*(worker_daemon(worker) for worker in topology)),
            timeout=max(120.0, worker_count * 0.25),
        )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    cpu_after = process.cpu_times()
    memory_after = process.memory_info().rss
    receipt = {
        "schema_version": "experiment-020-control-plane-dry-run-v1",
        "status": "PASS" if not failures and len(registered) == worker_count else "FAIL",
        "worker_count": worker_count,
        "pod_count": len({worker.pod_id for worker in topology}),
        "workers_per_pod": workers_per_pod,
        "explicit_worker_ids": len({worker.worker_id for worker in topology}),
        "host_agent_is_compute_resource": False,
        "persistent_tls_connections": len(registered),
        "messages": message_count,
        "messages_per_worker": message_count / worker_count,
        "per_expert_rpc": False,
        "top16_routes_per_worker_message": True,
        "elapsed_seconds": elapsed,
        "controller_cpu_seconds": (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        ),
        "controller_ram_delta_bytes": memory_after - memory_before,
        "controller_rss_bytes": memory_after,
        "queue_latency_p50_ms": statistics.median(queue_latencies),
        "queue_latency_p95_ms": sorted(queue_latencies)[
            min(len(queue_latencies) - 1, int(len(queue_latencies) * 0.95))
        ],
        "failures": failures,
        "security": security_receipt(material, credential),
        "credential_value_persisted": False,
    }
    # The certificate hash is retained; the per-run private key and credential
    # are ephemeral runtime material and are removed after every rehearsal.
    shutil.rmtree(material.directory, ignore_errors=True)
    return receipt


MAX_STREAM_LIMIT = 17 * 1024 * 1024


def run_control_plane_dry_run(
    worker_count: int, workers_per_pod: int = 8
) -> dict[str, Any]:
    return asyncio.run(_run(worker_count, workers_per_pod))


__all__ = [
    "LogicalWorker",
    "SwarmHostAgent",
    "logical_topology",
    "run_control_plane_dry_run",
]
