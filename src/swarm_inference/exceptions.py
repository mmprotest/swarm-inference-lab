"""Explicit failure types used across execution modes."""

from __future__ import annotations


class SwarmError(RuntimeError):
    """Base class for user-facing runtime errors."""


class ConfigurationError(SwarmError):
    """A configuration is invalid or internally inconsistent."""


class UnsupportedArchitectureError(SwarmError):
    """The requested model architecture has no compatible adapter."""


class UnsupportedCacheFormatError(SwarmError):
    """The requested attention cache format is not implemented."""


class MemoryLimitExceededError(SwarmError):
    """A model stage cannot fit within a worker's enforced logical memory limit."""


class BackendIncompatibleError(SwarmError):
    """A worker backend cannot execute a model stage."""


class InsufficientStageCoverageError(SwarmError):
    """Healthy replicas do not cover every stage."""


class NoValidRouteError(SwarmError):
    """No ordered route through all model stages is available."""


class IntegrityError(SwarmError):
    """A checksum, signature, shard hash, or audit check failed."""


class BackpressureError(SwarmError):
    """A bounded queue rejected work under its configured policy."""


class TransportError(SwarmError):
    """A transport operation failed."""


class WorkerUnavailableError(TransportError):
    """A selected worker is unavailable."""


class ReplayUnavailableError(SwarmError):
    """Required stage inputs are missing from the replay log."""


class RouteMessageError(IntegrityError):
    """A direct data-plane message violates its installed route."""

    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class EnvironmentIncompatibleError(SwarmError):
    """The requested execution backend cannot run in the current environment."""
