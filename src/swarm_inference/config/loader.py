"""YAML configuration loading with relative include support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

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
        referenced_config = load_yaml(referenced)
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
        return ExperimentConfig.model_validate(expanded)
    except ValueError as exc:
        raise ConfigurationError(f"invalid experiment configuration {resolved}: {exc}") from exc
