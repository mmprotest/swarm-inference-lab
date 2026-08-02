"""Measured expert data planes and byte-acting deterministic network shaping."""

from __future__ import annotations

import contextlib
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from swarm_inference.config.models import (
    JitterDistribution,
    NetworkProfile,
    OutageWindow,
)
from swarm_inference.experiments.experiment_010.schemas import (
    DataPlane,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    NetworkShapeProfile,
)
from swarm_inference.experiments.experiment_010.wire import (
    MAX_FRAME_BYTES,
    ExpertPacket,
    decode_packet,
    decode_response,
    encode_packet,
    encode_request,
    frame_with_length,
)
from swarm_inference.simulation.network import NetworkEmulator

_LENGTH = struct.Struct(">Q")


NETWORK_PROFILES: dict[str, NetworkShapeProfile] = {
    "loopback_unshaped": NetworkShapeProfile(
        name="loopback_unshaped", bandwidth_bps=None, one_way_latency_ms=0.0
    ),
    "fabric_100g": NetworkShapeProfile(
        name="fabric_100g", bandwidth_bps=100e9, one_way_latency_ms=0.05
    ),
    "lan_10g": NetworkShapeProfile(name="lan_10g", bandwidth_bps=10e9, one_way_latency_ms=0.10),
    "lan_2_5g": NetworkShapeProfile(name="lan_2_5g", bandwidth_bps=2.5e9, one_way_latency_ms=0.30),
    "lan_1g": NetworkShapeProfile(name="lan_1g", bandwidth_bps=1e9, one_way_latency_ms=0.50),
    "wifi": NetworkShapeProfile(
        name="wifi", bandwidth_bps=300e6, one_way_latency_ms=5.0, jitter_ms=2.0
    ),
    "regional_wan": NetworkShapeProfile(
        name="regional_wan", bandwidth_bps=100e6, one_way_latency_ms=20.0, jitter_ms=5.0
    ),
    "global_wan": NetworkShapeProfile(
        name="global_wan", bandwidth_bps=50e6, one_way_latency_ms=100.0, jitter_ms=20.0
    ),
}


class ShapedTransportError(ConnectionError):
    """A real transfer was rejected by the configured emulation profile."""


@dataclass(slots=True)
class ShaperMetrics:
    profile: str
    transfers: int = 0
    payload_bytes: int = 0
    imposed_delay_ns: int = 0
    queue_wait_ns: int = 0
    losses: int = 0
    duplications: int = 0
    reorderings: int = 0
    outages: int = 0
    started_ns: int = 0


class NetworkShaper:
    """Throttle actual message bytes before transport completion.

    The shaper owns a deterministic random generator, a bounded concurrent-flow
    semaphore, and a serialized bandwidth clock. It therefore changes the wall
    clock of the real socket/shared-memory operation rather than annotating a
    completed measurement with an estimated delay.
    """

    def __init__(self, profile: NetworkShapeProfile) -> None:
        self.profile = profile
        self._flow_slots = threading.BoundedSemaphore(profile.concurrent_flow_limit)
        self._origin_ns = time.perf_counter_ns()
        self.metrics = ShaperMetrics(profile=profile.name, started_ns=self._origin_ns)
        bandwidth_bytes_s = (
            profile.bandwidth_bps / 8 if profile.bandwidth_bps is not None else float(2**63)
        )
        self.simulation_profile = NetworkProfile(
            name=profile.name,
            base_latency_ms=profile.one_way_latency_ms,
            jitter_ms=profile.jitter_ms,
            jitter_distribution=(
                JitterDistribution.UNIFORM if profile.jitter_ms else JitterDistribution.NONE
            ),
            upload_bandwidth_bytes_s=bandwidth_bytes_s,
            download_bandwidth_bytes_s=bandwidth_bytes_s,
            packet_loss=profile.message_loss_probability,
            duplication_probability=profile.duplication_probability,
            reordering_probability=profile.reordering_probability,
            outage_windows=[
                OutageWindow(start_s=start / 1000, end_s=end / 1000)
                for start, end in profile.outage_intervals_ms
            ],
            max_in_flight_bytes=max(profile.queue_depth * (1 << 20), 1),
            measured=False,
        )
        self.emulator = NetworkEmulator(self.simulation_profile, seed=profile.seed)

    @contextlib.contextmanager
    def flow(self, timeout_s: float) -> Any:
        queued_ns = time.perf_counter_ns()
        if not self._flow_slots.acquire(timeout=timeout_s):
            raise TimeoutError("network shaper concurrent-flow queue timed out")
        self.metrics.queue_wait_ns += time.perf_counter_ns() - queued_ns
        try:
            yield
        finally:
            self._flow_slots.release()

    def enforce(self, byte_count: int, *, direction: str) -> dict[str, Any]:
        if byte_count < 0:
            raise ValueError("network shaper byte count cannot be negative")
        now_s = (time.perf_counter_ns() - self._origin_ns) / 1e9
        transmission = self.emulator.transmit(
            source=f"{direction}:source",
            destination=f"{direction}:destination",
            now_s=now_s,
            payload_bytes=byte_count,
        )
        if transmission.outage:
            self.metrics.outages += 1
        if transmission.lost:
            self.metrics.losses += 1
            raise ShapedTransportError(f"{self.profile.name} rejected {direction} message")
        delay_ns = int(max(0.0, transmission.completed_at_s - now_s) * 1e9)
        duplicated = transmission.duplicated
        reordered = transmission.reordered
        if duplicated:
            self.metrics.duplications += 1
            # A duplicate occupies the bandwidth resource even though the RPC
            # layer will discard the second logical delivery.
            delay_ns += int(transmission.serialization_s * 1e9)
        if reordered:
            self.metrics.reorderings += 1
            delay_ns += int(max(0.001, transmission.latency_s) * 1e9)
        if delay_ns:
            time.sleep(delay_ns / 1e9)
        self.metrics.transfers += 1
        self.metrics.payload_bytes += byte_count
        self.metrics.imposed_delay_ns += delay_ns
        return {
            "direction": direction,
            "bytes": byte_count,
            "imposed_delay_ns": delay_ns,
            "duplicated": duplicated,
            "reordered": reordered,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            **asdict(self.metrics),
            "requested": self.profile.model_dump(mode="json"),
            "elapsed_ns": time.perf_counter_ns() - self._origin_ns,
        }


@dataclass(slots=True)
class ExpertTransportMetrics:
    data_plane: str
    network_profile: str
    messages_sent: int = 0
    messages_received: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    payload_bytes: int = 0
    serialisation_ns: int = 0
    copy_ns: int = 0
    socket_ns: int = 0
    kernel_transition_ns: int = 0
    shared_memory_bytes: int = 0
    queue_ns: int = 0
    shaping_ns: int = 0
    total_request_ns: int = 0

    def snapshot(self) -> dict[str, int | str]:
        return asdict(self)


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    host, separator, raw_port = endpoint.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid expert endpoint {endpoint!r}")
    return host, int(raw_port)


def _recv_exact(connection: socket.socket, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("expert socket closed before the frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket) -> bytes:
    length = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))[0]
    if length > MAX_FRAME_BYTES:
        raise ValueError("expert response frame is too large")
    return _recv_exact(connection, length)


class ExpertTransportClient:
    """Coordinator-side client for direct/relayed TCP and shared memory."""

    def __init__(
        self,
        endpoint: str,
        *,
        data_plane: DataPlane | str = DataPlane.DIRECT_TCP,
        network_profile: NetworkShapeProfile | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.endpoint = endpoint
        self.data_plane = DataPlane(data_plane)
        self.timeout_s = timeout_s
        profile = network_profile or NETWORK_PROFILES["loopback_unshaped"]
        self.shaper = NetworkShaper(profile)
        self.metrics = ExpertTransportMetrics(
            data_plane=self.data_plane.value,
            network_profile=profile.name,
        )

    def _round_trip(self, payload: bytes, *, shape_payload: bool = True) -> bytes:
        host, port = _parse_endpoint(self.endpoint)
        framed = frame_with_length(payload)
        started = time.perf_counter_ns()
        with self.shaper.flow(self.timeout_s):
            if shape_payload:
                shaping_started = time.perf_counter_ns()
                self.shaper.enforce(len(framed), direction="request")
                self.metrics.shaping_ns += time.perf_counter_ns() - shaping_started
            socket_started = time.perf_counter_ns()
            with socket.create_connection((host, port), timeout=self.timeout_s) as connection:
                connection.settimeout(self.timeout_s)
                connection.sendall(framed)
                response = _recv_frame(connection)
            self.metrics.socket_ns += time.perf_counter_ns() - socket_started
            if shape_payload:
                shaping_started = time.perf_counter_ns()
                self.shaper.enforce(len(response) + _LENGTH.size, direction="response")
                self.metrics.shaping_ns += time.perf_counter_ns() - shaping_started
        self.metrics.messages_sent += 1
        self.metrics.messages_received += 1
        self.metrics.request_bytes += len(framed)
        self.metrics.response_bytes += len(response) + _LENGTH.size
        self.metrics.total_request_ns += time.perf_counter_ns() - started
        return response

    def execute(
        self, request: ExpertExecutionRequest, activation: np.ndarray
    ) -> tuple[ExpertExecutionResponse, np.ndarray, dict[str, Any]]:
        started = time.perf_counter_ns()
        before = self.metrics.snapshot()
        shaper_before = self.shaper.snapshot()
        encoded, encode_ns = encode_request(request, activation)
        self.metrics.serialisation_ns += encode_ns
        self.metrics.payload_bytes += int(np.asarray(activation).nbytes)
        if self.data_plane == DataPlane.SHARED_MEMORY:
            payload = self._execute_shared_memory(encoded)
        else:
            payload = self._round_trip(encoded)
        packet = decode_packet(payload)
        if packet.kind == "control":
            raise RuntimeError(str(packet.semantic.get("error", "worker request failed")))
        response, result, decode_ns = decode_response(payload)
        self.metrics.serialisation_ns += decode_ns
        self.metrics.queue_ns += response.execution_metadata.queue_ns
        elapsed = time.perf_counter_ns() - started
        after = self.metrics.snapshot()
        shaper_after = self.shaper.snapshot()
        metric_delta = {
            key: (
                value - before[key]
                if isinstance(value, int) and isinstance(before.get(key), int)
                else value
            )
            for key, value in after.items()
        }
        shaper_delta = {
            key: (
                value - shaper_before[key]
                if isinstance(value, int) and isinstance(shaper_before.get(key), int)
                else value
            )
            for key, value in shaper_after.items()
        }
        shaper_delta["elapsed_ns"] = elapsed
        return (
            response,
            result,
            {
                **metric_delta,
                "request_elapsed_ns": elapsed,
                "shaper": shaper_delta,
            },
        )

    def _execute_shared_memory(self, encoded: bytes) -> bytes:
        transition_started = time.perf_counter_ns()
        copy_started = time.perf_counter_ns()
        source = shared_memory.SharedMemory(create=True, size=len(encoded))
        # On Windows a named mapping ceases to exist as soon as the final
        # handle closes.  The coordinator therefore owns both mappings for the
        # complete exchange; a worker-created result mapping could disappear
        # between its close and our open.
        result_capacity = max(1 << 20, len(encoded) * 2 + (64 << 10))
        destination = shared_memory.SharedMemory(create=True, size=result_capacity)
        self.metrics.kernel_transition_ns += time.perf_counter_ns() - transition_started
        try:
            source.buf[: len(encoded)] = encoded
            self.metrics.copy_ns += time.perf_counter_ns() - copy_started
            control = encode_packet(
                ExpertPacket(
                    kind="control",
                    semantic={
                        "command": "shared_memory",
                        "name": source.name,
                        "size": len(encoded),
                        "result_name": destination.name,
                        "result_capacity": result_capacity,
                    },
                    blobs=(),
                )
            )
            response_payload = self._round_trip(control, shape_payload=False)
            packet = decode_packet(response_payload)
            if packet.kind != "control" or not packet.semantic.get("ok"):
                raise RuntimeError(str(packet.semantic.get("error", "shared memory failed")))
            if str(packet.semantic["name"]) != destination.name:
                raise RuntimeError("worker returned the wrong shared-memory result buffer")
            size = int(packet.semantic["size"])
            if size > result_capacity:
                raise RuntimeError("worker response exceeded shared-memory result capacity")
            copy_started = time.perf_counter_ns()
            result = bytes(destination.buf[:size])
            self.metrics.copy_ns += time.perf_counter_ns() - copy_started
            self.metrics.shared_memory_bytes += len(encoded) + size
            return result
        finally:
            source.close()
            with contextlib.suppress(FileNotFoundError):
                source.unlink()
            destination.close()
            with contextlib.suppress(FileNotFoundError):
                destination.unlink()

    def control(self, command: str, **payload: Any) -> dict[str, Any]:
        encoded = encode_packet(
            ExpertPacket(kind="control", semantic={"command": command, **payload}, blobs=())
        )
        response = decode_packet(self._round_trip(encoded, shape_payload=False))
        if response.kind != "control":
            raise RuntimeError("worker returned a non-control response")
        if not response.semantic.get("ok"):
            raise RuntimeError(str(response.semantic.get("error", "worker control failed")))
        return response.semantic


def measured_network_profile(
    requested: NetworkShapeProfile,
    shaper_snapshot: dict[str, Any],
) -> dict[str, Any]:
    elapsed_ns = int(shaper_snapshot["elapsed_ns"])
    payload_bytes = int(shaper_snapshot["payload_bytes"])
    transfers = int(shaper_snapshot["transfers"])
    return {
        "profile": requested.name,
        "requested_bandwidth_bps": requested.bandwidth_bps,
        "requested_one_way_latency_ms": requested.one_way_latency_ms,
        "achieved_effective_bandwidth_bps": (
            payload_bytes * 8 / (elapsed_ns / 1e9) if elapsed_ns > 0 else None
        ),
        "mean_imposed_delay_ms": (
            int(shaper_snapshot["imposed_delay_ns"]) / transfers / 1e6 if transfers else None
        ),
        "actual_payload_bytes": payload_bytes,
        "actual_transfers": transfers,
        "losses": int(shaper_snapshot["losses"]),
        "outages": int(shaper_snapshot["outages"]),
        "category": "MEASURED_NETWORK_EMULATION",
    }
