"""Post-rental network measurement records and preregistered E021 gates."""

from __future__ import annotations

import asyncio
import shutil
import ssl
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .transport import (
    Frame,
    MessageType,
    generate_run_tls_material,
    new_run_credential,
    read_frame,
    tls_contexts,
    write_frame,
)


def summarize_probe(samples: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not samples:
        raise ValueError("network probe requires samples")
    rtts = [float(row["rtt_ms"]) for row in samples]
    uploads = [float(row["upload_gbps"]) for row in samples]
    downloads = [float(row["download_gbps"]) for row in samples]
    return {
        "rtt_ms_p50": statistics.median(rtts),
        "rtt_ms_p95": sorted(rtts)[int(0.95 * (len(rtts) - 1))],
        "jitter_ms": statistics.pstdev(rtts),
        "upload_gbps_p10": sorted(uploads)[int(0.10 * (len(uploads) - 1))],
        "download_gbps_p10": sorted(downloads)[int(0.10 * (len(downloads) - 1))],
    }


def evaluate_topology(
    paths: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    results = []
    for path in paths:
        kind = str(path["kind"])
        gate = gates[kind]
        metrics = summarize_probe(path["samples"])
        bandwidth = min(metrics["upload_gbps_p10"], metrics["download_gbps_p10"])
        passed = (
            metrics["rtt_ms_p95"] <= float(gate["maximum_rtt_ms"])
            and bandwidth >= float(gate["minimum_bandwidth_gbps_at_maximum_rtt"])
            and metrics["jitter_ms"]
            <= float(gate.get("maximum_jitter_ms", float("inf")))
        )
        results.append(
            {
                "source": path["source"],
                "destination": path["destination"],
                "kind": kind,
                **metrics,
                "bidirectional_bandwidth_gbps_p10": bandwidth,
                "pass": passed,
            }
        )
    accepted = bool(results) and all(row["pass"] for row in results)
    return {
        "schema_version": "experiment-021-topology-decision-v1",
        "decision": "TOPOLOGY_ACCEPTED" if accepted else "TOPOLOGY_REJECTED",
        "marketplace_metadata_trusted_without_probe": False,
        "paths": results,
    }


async def measure_path(
    host: str,
    port: int,
    client_context: ssl.SSLContext,
    credential: bytes,
    *,
    samples: int = 10,
    bandwidth_payload_bytes: int = 4 * 1024 * 1024,
) -> list[dict[str, float]]:
    """Measure RTT, both directions, and jitter on one persistent TLS path."""

    reader, writer = await asyncio.open_connection(
        host, port, ssl=client_context, server_hostname="localhost", limit=17 * 1024 * 1024
    )
    payload = bytes((index * 17) & 0xFF for index in range(bandwidth_payload_bytes))
    results = []
    try:
        for index in range(samples):
            started = time.perf_counter_ns()
            await write_frame(
                writer,
                Frame(MessageType.HEALTH, f"ping-{index}", 0, "probe", "ping"),
                credential,
            )
            ping = await read_frame(reader, credential)
            rtt_ms = (time.perf_counter_ns() - started) / 1e6
            if ping.message_type is not MessageType.SHARD_RESULT:
                raise RuntimeError("network ping probe failed")

            started = time.perf_counter_ns()
            await write_frame(
                writer,
                Frame(
                    MessageType.EXECUTE_SHARD,
                    f"upload-{index}",
                    0,
                    "probe",
                    "upload",
                    payload,
                ),
                credential,
            )
            upload = await read_frame(reader, credential)
            upload_seconds = (time.perf_counter_ns() - started) / 1e9
            if upload.message_type is not MessageType.SHARD_RESULT:
                raise RuntimeError("network upload probe failed")

            started = time.perf_counter_ns()
            await write_frame(
                writer,
                Frame(
                    MessageType.HEALTH,
                    f"download-{index}",
                    0,
                    "probe",
                    "download",
                    str(bandwidth_payload_bytes).encode("ascii"),
                ),
                credential,
            )
            download = await read_frame(reader, credential)
            download_seconds = (time.perf_counter_ns() - started) / 1e9
            if download.payload != payload:
                raise RuntimeError("network download probe checksum/payload failed")
            results.append(
                {
                    "rtt_ms": rtt_ms,
                    "upload_gbps": bandwidth_payload_bytes * 8 / upload_seconds / 1e9,
                    "download_gbps": bandwidth_payload_bytes * 8 / download_seconds / 1e9,
                }
            )
    finally:
        writer.close()
        await writer.wait_closed()
    return results


async def _loopback_probe() -> dict[str, Any]:
    material = generate_run_tls_material()
    credential = new_run_credential()
    server_context, client_context = tls_contexts(material)
    probe_payload = bytes((index * 17) & 0xFF for index in range(4 * 1024 * 1024))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                frame = await read_frame(reader, credential)
                if frame.state_id == "ping":
                    payload = b"pong"
                elif frame.state_id == "upload":
                    payload = b"accepted"
                elif frame.state_id == "download":
                    requested = int(frame.payload.decode("ascii"))
                    payload = probe_payload[:requested]
                else:
                    raise RuntimeError("unsupported topology probe")
                await write_frame(
                    writer,
                    Frame(
                        MessageType.SHARD_RESULT,
                        frame.request_id,
                        frame.chunk_id,
                        frame.worker_id,
                        frame.state_id,
                        payload,
                    ),
                    credential,
                )
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(
        handle, "127.0.0.1", 0, ssl=server_context, limit=17 * 1024 * 1024
    )
    port = int(server.sockets[0].getsockname()[1])
    try:
        async with server:
            samples = await measure_path(
                "127.0.0.1", port, client_context, credential, samples=10
            )
    finally:
        shutil.rmtree(material.directory, ignore_errors=True)
    return {
        "schema_version": "experiment-020-network-probe-smoke-v1",
        "status": "PASS",
        "persistent_tls_connection": True,
        "authenticated_binary_protocol": True,
        "credential_value_persisted": False,
        "samples": samples,
        "summary": summarize_probe(samples),
    }


def run_loopback_probe_smoke() -> dict[str, Any]:
    return asyncio.run(_loopback_probe())


__all__ = [
    "evaluate_topology",
    "measure_path",
    "run_loopback_probe_smoke",
    "summarize_probe",
]
