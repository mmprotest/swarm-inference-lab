"""Ad hoc node contribution economics derived only from measured startup and throughput."""

from __future__ import annotations


def productive_fraction(lease_seconds: float, startup_seconds: float) -> float:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if startup_seconds < 0:
        raise ValueError("startup_seconds cannot be negative")
    return max(0.0, lease_seconds - startup_seconds) / lease_seconds


def productive_tokens(
    lease_seconds: float,
    startup_seconds: float,
    verified_tokens_per_second: float,
) -> float:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if startup_seconds < 0:
        raise ValueError("startup_seconds cannot be negative")
    if verified_tokens_per_second < 0:
        raise ValueError("verified_tokens_per_second cannot be negative")
    return max(0.0, lease_seconds - startup_seconds) * verified_tokens_per_second


def minimum_lease_duration(startup_seconds: float, target_productive_fraction: float) -> float:
    if startup_seconds < 0:
        raise ValueError("startup_seconds cannot be negative")
    if not 0 <= target_productive_fraction < 1:
        raise ValueError("target productive fraction must be in [0, 1)")
    return startup_seconds / (1 - target_productive_fraction)
