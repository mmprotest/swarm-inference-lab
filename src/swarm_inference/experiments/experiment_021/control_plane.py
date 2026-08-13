"""Independent-machine E021 controller scale harness.

Connections are established in bounded batches to avoid turning connection
setup itself into an accidental SYN/TLS storm.  Every connection remains open,
and dispatch begins only after all requested workers are registered, so the
peak-open-connection measurement is still the requested fleet size.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import statistics
import time
from typing import Any

import psutil

from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    generate_run_tls_material,
    new_run_credential,
    read_frame,
    security_receipt,
    tls_contexts,
    write_frame,
)

MAX_STREAM_LIMIT = 17 * 1024 * 1024


async def _run(
    worker_count: int,
    *,
    connection_batch: int,
) -> dict[str, Any]:
    if worker_count <= 0 or connection_batch <= 0:
        raise ValueError("worker_count and connection_batch must be positive")
    material = generate_run_tls_material()
    credential = new_run_credential()
    server_context, client_context = tls_contexts(material)
    registered: dict[str, asyncio.StreamWriter] = {}
    registration_condition = asyncio.Condition()
    dispatch = asyncio.Event()
    queue_latencies: list[float] = []
    scheduler_latencies: list[float] = []
    failures: list[str] = []
    message_count = 0
    peak_open_connections = 0

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal message_count, peak_open_connections
        registered_at = 0
        try:
            frame = await read_frame(reader, credential)
            message_count += 1
            if frame.message_type is not MessageType.REGISTER:
                raise RuntimeError("first worker frame must be REGISTER")
            if not frame.worker_id.startswith("machine-") or not frame.worker_id.endswith(
                ".worker"
            ):
                raise RuntimeError("worker ID is not an independent-machine identity")
            async with registration_condition:
                registered[frame.worker_id] = writer
                peak_open_connections = max(peak_open_connections, len(registered))
                registered_at = time.perf_counter_ns()
                registration_condition.notify_all()
            await write_frame(
                writer,
                Frame(
                    MessageType.REGISTERED,
                    frame.request_id,
                    0,
                    frame.worker_id,
                    "registered",
                ),
                credential,
            )
            message_count += 1
            await dispatch.wait()
            enqueued = time.perf_counter_ns()
            scheduler_latencies.append((enqueued - registered_at) / 1e6)
            payload = json.dumps(
                {
                    "operator": "expert_stripe_local_top16_accumulation",
                    "routes": list(range(16)),
                    "network_outputs": 1,
                    "whole_layer_fallback": False,
                    "whole_expert_fallback": False,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            await write_frame(
                writer,
                Frame(
                    MessageType.EXECUTE_SHARD,
                    "e021-control-plane-scale",
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
                raise RuntimeError("worker did not return SHARD_RESULT")
            await write_frame(
                writer,
                Frame(
                    MessageType.SHUTDOWN,
                    result.request_id,
                    0,
                    frame.worker_id,
                    "done",
                ),
                credential,
            )
            message_count += 1
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    server = await asyncio.start_server(
        handle,
        "127.0.0.1",
        0,
        ssl=server_context,
        backlog=max(128, connection_batch * 2),
        limit=MAX_STREAM_LIMIT,
    )
    port = int(server.sockets[0].getsockname()[1])

    async def worker(index: int) -> None:
        worker_id = f"machine-{index:04d}.worker"
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                "127.0.0.1",
                port,
                ssl=client_context,
                server_hostname="localhost",
                limit=MAX_STREAM_LIMIT,
            ),
            timeout=30.0,
        )
        try:
            await write_frame(
                writer,
                Frame(MessageType.REGISTER, "register", 0, worker_id, "initial"),
                credential,
            )
            response = await read_frame(reader, credential)
            if response.message_type is not MessageType.REGISTERED:
                raise RuntimeError("registration rejected")
            execute = await read_frame(reader, credential)
            if execute.message_type is not MessageType.EXECUTE_SHARD:
                raise RuntimeError("unexpected controller message")
            body = json.loads(execute.payload)
            if body["network_outputs"] != 1 or len(body["routes"]) != 16:
                raise RuntimeError("per-expert RPC architecture detected")
            if body["whole_layer_fallback"] or body["whole_expert_fallback"]:
                raise RuntimeError("forbidden fallback requested")
            await write_frame(
                writer,
                Frame(
                    MessageType.SHARD_RESULT,
                    execute.request_id,
                    execute.chunk_id,
                    worker_id,
                    execute.state_id,
                    b"partial-latent-vector",
                ),
                credential,
            )
            shutdown = await read_frame(reader, credential)
            if shutdown.message_type is not MessageType.SHUTDOWN:
                raise RuntimeError("clean shutdown not received")
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    process = psutil.Process()
    cpu_before = process.cpu_times()
    memory_before = process.memory_info().rss
    started = time.perf_counter_ns()
    tasks: list[asyncio.Task[None]] = []
    try:
        async with server:
            for start in range(0, worker_count, connection_batch):
                stop = min(worker_count, start + connection_batch)
                tasks.extend(asyncio.create_task(worker(index)) for index in range(start, stop))
                async with registration_condition:
                    await asyncio.wait_for(
                        registration_condition.wait_for(
                            lambda target=stop: len(registered) >= target or bool(failures)
                        ),
                        timeout=60.0,
                    )
                if failures:
                    raise RuntimeError(failures[0])
                # Yield so the server can drain accept/TLS completion queues
                # before the next bounded connection batch.
                await asyncio.sleep(0)
            dispatch.set()
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=max(120.0, worker_count * 0.10),
            )
    finally:
        dispatch.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        shutil.rmtree(material.directory, ignore_errors=True)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    cpu_after = process.cpu_times()
    memory_after = process.memory_info().rss
    status = (
        not failures
        and len(registered) == worker_count
        and peak_open_connections == worker_count
        and message_count == worker_count * 5
    )
    return {
        "schema_version": "experiment-021-control-plane-scale-v1",
        "status": "PASS" if status else "FAIL",
        "worker_count": worker_count,
        "machine_count": worker_count,
        "compute_workers_per_machine": 1,
        "connection_batch": connection_batch,
        "connection_setup_is_bounded": True,
        "dispatch_begins_after_all_workers_registered": True,
        "peak_open_connections": peak_open_connections,
        "explicit_worker_ids": len(registered),
        "messages": message_count,
        "messages_per_worker": message_count / worker_count,
        "per_expert_rpc": False,
        "top16_routes_per_worker_message": True,
        "same_host_collective": False,
        "host_or_controller_is_compute_resource": False,
        "native_shard_compute_invoked": False,
        "elapsed_seconds": elapsed,
        "controller_cpu_seconds": (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        ),
        "controller_ram_delta_bytes": memory_after - memory_before,
        "controller_rss_bytes": memory_after,
        "scheduler_latency_p50_ms": statistics.median(scheduler_latencies),
        "scheduler_latency_p95_ms": sorted(scheduler_latencies)[
            min(len(scheduler_latencies) - 1, int(len(scheduler_latencies) * 0.95))
        ],
        "queue_latency_p50_ms": statistics.median(queue_latencies),
        "queue_latency_p95_ms": sorted(queue_latencies)[
            min(len(queue_latencies) - 1, int(len(queue_latencies) * 0.95))
        ],
        "failures": failures,
        "security": security_receipt(material, credential),
        "credential_value_persisted": False,
    }


def run_control_plane_scale(
    worker_count: int,
    *,
    connection_batch: int = 100,
) -> dict[str, Any]:
    return asyncio.run(_run(worker_count, connection_batch=connection_batch))


__all__ = ["run_control_plane_scale"]
