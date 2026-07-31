"""YAML configuration loading with relative include support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import yaml

from swarm_inference.config.models import ExperimentConfig
from swarm_inference.exceptions import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and fail precisely for malformed input."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"configuration file does not exist: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {resolved}")
    return raw


def _load_reference(value: Any, *, base: Path) -> Any:
    if isinstance(value, dict) and "from" in value:
        referenced = (base / str(value["from"])).resolve()
        referenced_config = _load_reference(
            load_yaml(referenced),
            base=referenced.parent,
        )
        overrides = {
            key: _load_reference(item, base=base) for key, item in value.items() if key != "from"
        }
        return {**referenced_config, **overrides}
    if isinstance(value, list):
        expanded_items: list[Any] = []
        for item in value:
            result = _load_reference(item, base=base)
            if isinstance(result, list):
                expanded_items.extend(result)
            else:
                expanded_items.append(result)
        return expanded_items
    if isinstance(value, dict):
        return {key: _load_reference(item, base=base) for key, item in value.items()}
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration."""

    resolved = Path(path).expanduser().resolve()
    raw = deepcopy(load_yaml(resolved))
    expanded = _load_reference(raw, base=resolved.parent)
    try:
        if expanded.get("execution_mode") == "single-host-loopback-real-model":
            from swarm_inference.config.real_model import RealExperimentConfig

            real = RealExperimentConfig.model_validate(expanded)
            # The generic loader validates every shipped experiment file. Real
            # execution is deliberately routed through ``real-experiment``;
            # callers of that command use ``load_real_experiment_config``.
            return cast(ExperimentConfig, real)
        if expanded.get("execution_mode") == "single-host-loopback-real-model-fanout":
            from swarm_inference.config.worker_fanout import FanoutExperimentConfig

            fanout = FanoutExperimentConfig.model_validate(expanded)
            return cast(ExperimentConfig, fanout)
        if expanded.get("execution_mode") == "single-host-engine-benchmark":
            from swarm_inference.config.engine_performance import (
                EnginePerformanceConfig,
            )

            engine = EnginePerformanceConfig.model_validate(expanded)
            return cast(ExperimentConfig, engine)
        if expanded.get("execution_mode") == "logical-single-gpu-microsharding":
            from swarm_inference.config.microsharding import MicroshardingExperimentConfig

            microsharding = MicroshardingExperimentConfig.model_validate(expanded)
            return cast(ExperimentConfig, microsharding)
        return ExperimentConfig.model_validate(expanded)
    except ValueError as exc:
        raise ConfigurationError(f"invalid experiment configuration {resolved}: {exc}") from exc
