"""Research CLI extension exports.

This module is an extension provider: experiments depend on canonical product
modules, while product modules discover this provider through an entry point.
"""

from __future__ import annotations

from typing import Any


def get_export(name: str) -> Any:
    if name in {"run_experiment", "validate_run"}:
        from swarm_inference.experiments import runner

        return getattr(runner, name)
    if name in {"run_loopback_experiment", "run_physical_experiment"}:
        from swarm_inference.experiments import loopback

        return getattr(loopback, name)
    if name == "run_loopback_matrix":
        from swarm_inference.experiments.loopback_matrix import run_loopback_matrix

        return run_loopback_matrix
    if name in {"MicroshardingOptions", "run_microsharding_experiment"}:
        from swarm_inference.experiments import microsharding

        return getattr(microsharding, name)
    if name == "run_engine_performance_experiment":
        from swarm_inference.experiments.engine_performance import (
            run_engine_performance_experiment,
        )

        return run_engine_performance_experiment
    if name in {"HeterogeneousOptions", "run_heterogeneous_node_experiment"}:
        from swarm_inference.experiments import heterogeneous_node_utility

        return getattr(heterogeneous_node_utility, name)
    if name in {"Experiment007CorrectionOptions", "run_experiment_007_corrections"}:
        from swarm_inference.experiments import experiment_007_corrections

        return getattr(experiment_007_corrections, name)
    if name in {"Experiment008Options", "run_experiment_008"}:
        from swarm_inference.experiments.experiment_008 import runner

        return getattr(runner, name)
    if name == "run_worker_fanout_experiment":
        from swarm_inference.experiments.worker_fanout import run_worker_fanout_experiment

        return run_worker_fanout_experiment
    if name == "run_qwen3_process_loopback":
        from swarm_inference.experiments.real_model import run_qwen3_process_loopback

        return run_qwen3_process_loopback
    if name == "run_experiment_002":
        from swarm_inference.experiments.experiment_002 import run_experiment_002

        return run_experiment_002
    if name == "render_html_report":
        from swarm_inference.experiments.reporting import render_html_report

        return render_html_report
    raise KeyError(name)


__all__ = ["get_export"]
