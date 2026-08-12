"""Separate coarse and internal Experiment 015 network domains."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    """One shaped one-way service domain."""

    name: str
    rtt_ms: float
    bandwidth_gbps: float
    measured_loopback_base_ms: float = 0.6102
    framing_bytes: int = 235

    def __post_init__(self) -> None:
        if self.rtt_ms < 0 or self.bandwidth_gbps <= 0:
            raise ValueError("network RTT must be non-negative and bandwidth positive")
        if self.measured_loopback_base_ms < 0 or self.framing_bytes < 0:
            raise ValueError("network base/framing must be non-negative")

    def service_ms(self, payload_bytes: int) -> float:
        if payload_bytes < 0:
            raise ValueError("payload bytes must be non-negative")
        wire_bytes = payload_bytes + self.framing_bytes
        transfer_ms = wire_bytes * 8.0 / (self.bandwidth_gbps * 1_000_000.0)
        return self.measured_loopback_base_ms + self.rtt_ms / 2.0 + transfer_ms


@dataclass(frozen=True, slots=True)
class MicrocellGeometry:
    """Count slow and fast boundaries for adjacent-layer microcells."""

    layers: int
    depth: int

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.depth <= 0:
            raise ValueError("layer count and cell depth must be positive")

    @property
    def cell_count(self) -> int:
        return math.ceil(self.layers / self.depth)

    @property
    def total_boundaries(self) -> int:
        return max(0, self.layers - 1)

    @property
    def coarse_boundaries(self) -> int:
        return max(0, self.cell_count - 1)

    @property
    def internal_boundaries(self) -> int:
        return self.total_boundaries - self.coarse_boundaries

    def dependency_latency_ms(
        self,
        *,
        compute_ms: float,
        payload_bytes: int,
        internal: NetworkProfile,
        coarse: NetworkProfile,
    ) -> float:
        if compute_ms < 0:
            raise ValueError("compute latency must be non-negative")
        return (
            compute_ms
            + self.internal_boundaries * internal.service_ms(payload_bytes)
            + self.coarse_boundaries * coarse.service_ms(payload_bytes)
        )


def activation_payload_bytes(rows: int, *, hidden: int = 7168, attnres_slots: int = 8) -> int:
    """Actual Kimi FP32 boundary: hidden plus eight AttnRes rows."""
    if rows <= 0 or hidden <= 0 or attnres_slots < 0:
        raise ValueError("activation geometry must be positive")
    return rows * (attnres_slots + 1) * hidden * 4


__all__ = ["MicrocellGeometry", "NetworkProfile", "activation_payload_bytes"]
