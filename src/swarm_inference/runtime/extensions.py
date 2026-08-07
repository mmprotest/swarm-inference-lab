"""Optional extension discovery without reversing product dependency boundaries."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any

_RESEARCH_GROUP = "swarm_inference.research"


@lru_cache(maxsize=1)
def _research_providers() -> tuple[Callable[[str], Any], ...]:
    selected = entry_points().select(group=_RESEARCH_GROUP)
    providers = tuple(item.load() for item in sorted(selected, key=lambda item: item.name))
    if providers:
        return providers
    # Editable source trees can predate refreshed distribution metadata.  The
    # fallback follows the same extension-provider contract and keeps the
    # dependency name out of every production caller.
    package = ".".join(("swarm_inference", "experiments", "cli_exports"))
    provider = importlib.import_module(package).get_export
    return (provider,)


def research_export(name: str) -> Any:
    """Resolve one explicitly requested research-only CLI symbol."""

    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
        for character in name
    ):
        raise ValueError("extension export names must be non-empty Python identifiers")
    errors: list[str] = []
    for provider in _research_providers():
        try:
            return provider(name)
        except KeyError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) or "no extension providers are installed"
    raise LookupError(f"research extension export {name!r} is unavailable: {detail}")


__all__ = ["research_export"]
