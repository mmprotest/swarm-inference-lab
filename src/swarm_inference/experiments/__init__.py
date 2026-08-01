"""Reproducible experiment execution and artifact reporting.

The public runner exports are lazy so model/rank workers can reuse lightweight
experiment lifecycle helpers without importing charting dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from swarm_inference.experiments.runner import ExperimentRun

__all__ = ["ExperimentRun", "run_experiment"]


def __getattr__(name: str) -> Any:
    if name in {"ExperimentRun", "run_experiment"}:
        from swarm_inference.experiments.runner import ExperimentRun, run_experiment

        return {"ExperimentRun": ExperimentRun, "run_experiment": run_experiment}[name]
    raise AttributeError(name)
