"""Measured network-topology domains used by canonical engine planning."""

from __future__ import annotations

from enum import StrEnum

from pydantic import ConfigDict, Field, PositiveInt

from swarm_inference.config.models import StrictModel


class TopologyDomain(StrEnum):
    LOCAL_FAST = "local-fast"
    REGIONAL = "regional"
    WAN = "wan"
    UNKNOWN = "unknown"


class NetworkCostConfidence(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNMEASURED = "unmeasured"


class TopologyThresholds(StrictModel):
    """Property-based thresholds; labels never imply physical geography."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    local_max_rtt_ms: float = Field(default=5.0, gt=0)
    local_min_bandwidth_bytes_s: float = Field(default=125_000_000.0, gt=0)
    local_max_jitter_ms: float = Field(default=2.0, ge=0)
    local_min_stability: float = Field(default=0.98, ge=0, le=1)
    regional_max_rtt_ms: float = Field(default=40.0, gt=0)
    regional_min_bandwidth_bytes_s: float = Field(default=12_500_000.0, gt=0)


class NetworkLinkProfile(StrictModel):
    """One directed link measurement with honest missing-value semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rtt_ms: float | None = Field(default=None, ge=0)
    bandwidth_bytes_s: float | None = Field(default=None, gt=0)
    jitter_ms: float | None = Field(default=None, ge=0)
    stability: float | None = Field(default=None, ge=0, le=1)
    sample_count: PositiveInt | None = None
    authenticated: bool = False
    provenance: str = "unmeasured"

    def classify(self, thresholds: TopologyThresholds | None = None) -> TopologyDomain:
        selected = thresholds or TopologyThresholds()
        # A single clearly slow measurement is enough to keep fine-grained RPC
        # off the critical path.  Missing data is never interpreted as a fast
        # zero-latency/zero-byte link.
        if (self.rtt_ms is not None and self.rtt_ms > selected.regional_max_rtt_ms) or (
            self.bandwidth_bytes_s is not None
            and self.bandwidth_bytes_s < selected.regional_min_bandwidth_bytes_s
        ):
            return TopologyDomain.WAN
        if self.rtt_ms is None or self.bandwidth_bytes_s is None:
            return TopologyDomain.UNKNOWN
        if (
            self.rtt_ms <= selected.local_max_rtt_ms
            and self.bandwidth_bytes_s >= selected.local_min_bandwidth_bytes_s
            and self.jitter_ms is not None
            and self.jitter_ms <= selected.local_max_jitter_ms
            and self.stability is not None
            and self.stability >= selected.local_min_stability
        ):
            return TopologyDomain.LOCAL_FAST
        if (
            self.rtt_ms <= selected.regional_max_rtt_ms
            and self.bandwidth_bytes_s >= selected.regional_min_bandwidth_bytes_s
        ):
            return TopologyDomain.REGIONAL
        return TopologyDomain.WAN

    @property
    def confidence(self) -> NetworkCostConfidence:
        if self.rtt_ms is None or self.bandwidth_bytes_s is None:
            return NetworkCostConfidence.UNMEASURED
        if self.sample_count is not None and self.authenticated:
            return NetworkCostConfidence.MEASURED
        return NetworkCostConfidence.ESTIMATED


class NetworkPathSummary(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregate_rtt_ms: float | None = Field(default=None, ge=0)
    aggregate_jitter_ms: float | None = Field(default=None, ge=0)
    bottleneck_bandwidth_bytes_s: float | None = Field(default=None, gt=0)
    wan_boundaries: int | None = Field(default=None, ge=0)
    domains: tuple[TopologyDomain, ...]
    confidence: NetworkCostConfidence
    provenance: str


def summarize_network_path(
    links: tuple[NetworkLinkProfile, ...],
) -> NetworkPathSummary:
    if not links:
        return NetworkPathSummary(
            aggregate_rtt_ms=0.0,
            aggregate_jitter_ms=0.0,
            bottleneck_bandwidth_bytes_s=None,
            wan_boundaries=0,
            domains=(),
            confidence=NetworkCostConfidence.MEASURED,
            provenance="no-network-boundary",
        )
    domains = tuple(item.classify() for item in links)
    measurements_complete = all(
        item.rtt_ms is not None and item.bandwidth_bytes_s is not None for item in links
    )
    jitter_complete = all(item.jitter_ms is not None for item in links)
    confidence = (
        NetworkCostConfidence.UNMEASURED
        if not measurements_complete
        else NetworkCostConfidence.MEASURED
        if all(item.confidence == NetworkCostConfidence.MEASURED for item in links)
        else NetworkCostConfidence.ESTIMATED
    )
    return NetworkPathSummary(
        aggregate_rtt_ms=(
            sum(float(item.rtt_ms) for item in links if item.rtt_ms is not None)
            if all(item.rtt_ms is not None for item in links)
            else None
        ),
        aggregate_jitter_ms=(
            sum(float(item.jitter_ms) for item in links if item.jitter_ms is not None)
            if jitter_complete
            else None
        ),
        bottleneck_bandwidth_bytes_s=(
            min(
                float(item.bandwidth_bytes_s)
                for item in links
                if item.bandwidth_bytes_s is not None
            )
            if all(item.bandwidth_bytes_s is not None for item in links)
            else None
        ),
        wan_boundaries=(
            sum(domain == TopologyDomain.WAN for domain in domains)
            if TopologyDomain.UNKNOWN not in domains
            else None
        ),
        domains=domains,
        confidence=confidence,
        provenance="; ".join(item.provenance for item in links),
    )


__all__ = [
    "NetworkCostConfidence",
    "NetworkLinkProfile",
    "NetworkPathSummary",
    "TopologyDomain",
    "TopologyThresholds",
    "summarize_network_path",
]
