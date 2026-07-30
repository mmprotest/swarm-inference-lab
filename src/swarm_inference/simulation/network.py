"""Deterministic application-level directed-link emulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from swarm_inference.config.models import JitterDistribution, NetworkProfile


@dataclass(frozen=True, slots=True)
class Transmission:
    source: str
    destination: str
    payload_bytes: int
    requested_at_s: float
    started_at_s: float
    completed_at_s: float
    queueing_s: float
    latency_s: float
    serialization_s: float
    jitter_s: float
    lost: bool
    duplicated: bool
    reordered: bool
    outage: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DeterministicLink:
    def __init__(
        self,
        *,
        source: str,
        destination: str,
        profile: NetworkProfile,
        rng: np.random.Generator,
    ) -> None:
        self.source = source
        self.destination = destination
        self.profile = profile
        self.rng = rng
        self.next_serialisation_available_s = 0.0

    def _jitter_s(self) -> float:
        maximum = self.profile.jitter_ms / 1000.0
        if maximum == 0 or self.profile.jitter_distribution == JitterDistribution.NONE:
            return 0.0
        if self.profile.jitter_distribution == JitterDistribution.UNIFORM:
            return float(self.rng.uniform(-maximum, maximum))
        return float(self.rng.normal(0.0, maximum / 3.0))

    def transmit(self, *, now_s: float, payload_bytes: int) -> Transmission:
        if payload_bytes < 0:
            raise ValueError("payload_bytes cannot be negative")
        permanent = (
            self.profile.permanent_failure_at_s is not None
            and now_s >= self.profile.permanent_failure_at_s
        )
        outage = permanent or any(
            window.start_s <= now_s < window.end_s for window in self.profile.outage_windows
        )
        bandwidth = min(
            self.profile.upload_bandwidth_bytes_s,
            self.profile.download_bandwidth_bytes_s,
        )
        started = max(now_s, self.next_serialisation_available_s)
        queueing = started - now_s
        serialization = payload_bytes / bandwidth
        self.next_serialisation_available_s = started + serialization
        jitter = self._jitter_s()
        latency = max(0.0, self.profile.base_latency_ms / 1000.0 + jitter)
        complete = started + serialization + latency
        lost = outage or bool(self.rng.random() < self.profile.packet_loss)
        duplicated = not lost and bool(self.rng.random() < self.profile.duplication_probability)
        reordered = not lost and bool(self.rng.random() < self.profile.reordering_probability)
        return Transmission(
            source=self.source,
            destination=self.destination,
            payload_bytes=payload_bytes,
            requested_at_s=now_s,
            started_at_s=started,
            completed_at_s=complete,
            queueing_s=queueing,
            latency_s=latency,
            serialization_s=serialization,
            jitter_s=jitter,
            lost=lost,
            duplicated=duplicated,
            reordered=reordered,
            outage=outage,
        )


class NetworkEmulator:
    """Create independent deterministic state for every directed link."""

    def __init__(self, profile: NetworkProfile, *, seed: int) -> None:
        self.profile = profile
        self._seed_sequence = np.random.SeedSequence(seed)
        self._links: dict[tuple[str, str], DeterministicLink] = {}
        self.transmissions: list[Transmission] = []

    def link(self, source: str, destination: str) -> DeterministicLink:
        key = (source, destination)
        if key not in self._links:
            child = self._seed_sequence.spawn(1)[0]
            self._links[key] = DeterministicLink(
                source=source,
                destination=destination,
                profile=self.profile,
                rng=np.random.default_rng(child),
            )
        return self._links[key]

    def transmit(
        self,
        *,
        source: str,
        destination: str,
        now_s: float,
        payload_bytes: int,
    ) -> Transmission:
        transmission = self.link(source, destination).transmit(
            now_s=now_s,
            payload_bytes=payload_bytes,
        )
        self.transmissions.append(transmission)
        return transmission
