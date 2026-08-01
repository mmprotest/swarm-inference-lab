"""Worker implementations and the backend-neutral Universal Worker ABI.

Legacy worker exports are loaded lazily so a minimal rank environment does not
import the full experiment/reporting stack merely by importing ``worker.abi``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from swarm_inference.worker.abi import (
    BackendAdapter,
    BackendArtifactMapping,
    WorkerCapabilities,
    WorkerIdentity,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
    WorkerProtocolVersion,
)

if TYPE_CHECKING:
    from swarm_inference.worker.agent import WorkerAgent

__all__ = [
    "BackendAdapter",
    "BackendArtifactMapping",
    "WorkerAgent",
    "WorkerCapabilities",
    "WorkerIdentity",
    "WorkerJob",
    "WorkerJobResult",
    "WorkerJobStatus",
    "WorkerJobType",
    "WorkerProtocolVersion",
    "measure_capabilities",
]


def __getattr__(name: str) -> Any:
    if name == "WorkerAgent":
        from swarm_inference.worker.agent import WorkerAgent

        return WorkerAgent
    if name == "measure_capabilities":
        from swarm_inference.worker.capabilities import measure_capabilities

        return measure_capabilities
    raise AttributeError(name)
