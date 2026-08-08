"""Reusable stateful stage-execution contracts and implementations.

The package exports are deliberately lazy.  Importing a NumPy-only execution
primitive (for example, the canonical expert store) must not initialise the
PyTorch runtime or a CUDA context as a side effect of package discovery.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swarm_inference.execution.interfaces import (
        StageExecutionResult,
        StageExecutor,
        WeightOwnership,
    )
    from swarm_inference.execution.moe import (
        HybridMoeBackend,
        LocalMoeBackend,
        MicroshardRemoteBackend,
        MoeExecutionBackend,
        MoeExecutionResult,
        WholeExpertRemoteBackend,
    )

_EXPORT_MODULES = {
    "HybridMoeBackend": "swarm_inference.execution.moe",
    "LocalMoeBackend": "swarm_inference.execution.moe",
    "MicroshardRemoteBackend": "swarm_inference.execution.moe",
    "MoeExecutionBackend": "swarm_inference.execution.moe",
    "MoeExecutionResult": "swarm_inference.execution.moe",
    "StageExecutionResult": "swarm_inference.execution.interfaces",
    "StageExecutor": "swarm_inference.execution.interfaces",
    "WeightOwnership": "swarm_inference.execution.interfaces",
    "WholeExpertRemoteBackend": "swarm_inference.execution.moe",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value

__all__ = [
    "HybridMoeBackend",
    "LocalMoeBackend",
    "MicroshardRemoteBackend",
    "MoeExecutionBackend",
    "MoeExecutionResult",
    "StageExecutionResult",
    "StageExecutor",
    "WeightOwnership",
    "WholeExpertRemoteBackend",
]
