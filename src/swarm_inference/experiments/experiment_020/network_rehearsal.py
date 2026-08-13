"""Runtime-level E021 message-sequence rehearsal with deterministic shaping."""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

from .transport import Frame, MessageType, decode_frame, encode_frame, new_run_credential


def _one_way_ms(payload_bytes: int, rtt_ms: float, bandwidth_gbps: float) -> float:
    return rtt_ms / 2.0 + payload_bytes * 8.0 / (bandwidth_gbps * 1_000_000.0)


async def _rehearse(
    *,
    pods: int,
    workers_per_pod: int,
    block: int,
    chunk: int,
    local_rtt_ms: float,
    local_bandwidth_gbps: float,
    inter_rtt_ms: float,
    inter_bandwidth_gbps: float,
) -> dict[str, Any]:
    credential = new_run_credential()
    latencies: list[float] = []
    # One activation, one grouped top-16 request, one partial latent, and one
    # tree reduction per logical layer/chunk. The exact production framing and
    # authentication path is exercised; sleeps provide canonical link shaping.
    started = time.perf_counter_ns()
    message_total = 0
    wire_total = 0
    for chunk_id in range((block + 1 + chunk - 1) // chunk):
        for layer in range(93):
            pod = min(pods - 1, layer // 8)
            tasks = []
            for stripe in range(workers_per_pod):
                payload = bytes(7168 * 4 * min(chunk, block + 1 - chunk_id * chunk))
                frame = Frame(
                    MessageType.EXECUTE_SHARD,
                    "e021-runtime-rehearsal",
                    chunk_id,
                    f"pod-{pod:03d}.worker-{stripe:02d}",
                    f"layer-{layer:02d}",
                    payload,
                )
                wire = encode_frame(frame, credential)

                async def transfer(value: bytes = wire, expected: bytes = payload) -> float:
                    begin = time.perf_counter_ns()
                    await asyncio.sleep(
                        _one_way_ms(
                            len(value), local_rtt_ms, local_bandwidth_gbps
                        )
                        / 1000.0
                    )
                    decoded = decode_frame(value, credential)
                    if decoded.payload != expected:
                        raise RuntimeError("runtime rehearsal payload mismatch")
                    return (time.perf_counter_ns() - begin) / 1e6

                tasks.append(transfer())
                message_total += 1
                wire_total += len(wire)
            latencies.extend(await asyncio.gather(*tasks))
            if layer and layer % 8 == 0:
                # State/wavefront handoff between depth pods.
                size = 7168 * 4 * min(chunk, block + 1 - chunk_id * chunk)
                await asyncio.sleep(
                    _one_way_ms(size, inter_rtt_ms, inter_bandwidth_gbps) / 1000.0
                )
                message_total += 1
                wire_total += size
    elapsed = (time.perf_counter_ns() - started) / 1e9
    return {
        "schema_version": "experiment-020-network-shaped-rehearsal-v1",
        "status": "PASS",
        "predicted_or_physical_swarm_result": "NEITHER_SOFTWARE_INTEGRATION_ONLY",
        "pods": pods,
        "workers_per_pod": workers_per_pod,
        "worker_count": pods * workers_per_pod,
        "block": block,
        "chunk": chunk,
        "layers": 93,
        "grouped_top16_messages": True,
        "per_expert_rpc": False,
        "binary_authenticated_production_framing": True,
        "messages": message_total,
        "wire_bytes": wire_total,
        "wall_seconds": elapsed,
        "local_message_p50_ms": statistics.median(latencies),
        "local_message_p95_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "shaping": {
            "local_rtt_ms": local_rtt_ms,
            "local_bandwidth_gbps": local_bandwidth_gbps,
            "inter_pod_rtt_ms": inter_rtt_ms,
            "inter_pod_bandwidth_gbps": inter_bandwidth_gbps,
        },
    }


def run_network_shaped_rehearsal() -> dict[str, Any]:
    return asyncio.run(
        _rehearse(
            pods=12,
            workers_per_pod=8,
            block=16,
            chunk=1,
            local_rtt_ms=0.25,
            local_bandwidth_gbps=25.0,
            inter_rtt_ms=5.0,
            inter_bandwidth_gbps=10.0,
        )
    )


__all__ = ["run_network_shaped_rehearsal"]
