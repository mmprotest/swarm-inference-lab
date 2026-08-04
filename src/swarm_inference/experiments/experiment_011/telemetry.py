"""Deprecated compatibility exports for canonical runtime telemetry.

New code must import :mod:`swarm_inference.runtime.telemetry` directly.
"""

from swarm_inference.runtime.telemetry import (
    TraceContext,
    TraceWriter,
    merge_traces,
    read_trace,
    reconstruct_critical_path,
)

__all__ = [
    "TraceContext",
    "TraceWriter",
    "merge_traces",
    "read_trace",
    "reconstruct_critical_path",
]
