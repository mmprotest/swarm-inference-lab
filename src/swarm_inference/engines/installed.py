"""Discovery of installer-owned execution-engine runtime manifests.

General engines are never discovered from ``PATH``.  A normal native install owns
the binaries and their immutable manifests below the application root; explicit
paths and environment variables exist only as development/diagnostic overrides.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from swarm_inference.native_install import native_install_record

_MANIFEST_RELATIVE_PATHS = {
    "llamacpp-rpc": Path("runtime") / "engines" / "llamacpp" / "runtime-manifest.json",
    "colibri": Path("runtime") / "engines" / "colibri" / "runtime-manifest.json",
}
_MANIFEST_ENVIRONMENT = {
    "llamacpp-rpc": "SWARM_LLAMACPP_RUNTIME_MANIFEST",
    "colibri": "SWARM_COLIBRI_RUNTIME_MANIFEST",
}


@dataclass(frozen=True, slots=True)
class InstalledEngineManifests:
    """Resolved manifest paths for the engine runtimes owned by this install."""

    llamacpp: Path | None = None
    colibri: Path | None = None


def _candidate_install_root(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    installed = native_install_record()
    return installed[0] if installed is not None else None


def resolve_installed_engine_manifest(
    engine_id: str,
    *,
    explicit: Path | None = None,
    install_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve one manifest without consulting arbitrary executable search paths."""

    try:
        relative = _MANIFEST_RELATIVE_PATHS[engine_id]
        environment_name = _MANIFEST_ENVIRONMENT[engine_id]
    except KeyError as exc:
        raise ValueError(f"unknown installed engine {engine_id!r}") from exc
    if explicit is not None:
        return explicit.expanduser().resolve()
    values = os.environ if environment is None else environment
    if raw := values.get(environment_name):
        return Path(raw).expanduser().resolve()
    root = _candidate_install_root(install_root)
    if root is None:
        return None
    candidate = (root / relative).resolve()
    return candidate if candidate.is_file() else None


def discover_installed_engine_manifests(
    *,
    llamacpp: Path | None = None,
    colibri: Path | None = None,
    install_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> InstalledEngineManifests:
    """Return all configured manifests using one deterministic precedence order."""

    return InstalledEngineManifests(
        llamacpp=resolve_installed_engine_manifest(
            "llamacpp-rpc",
            explicit=llamacpp,
            install_root=install_root,
            environment=environment,
        ),
        colibri=resolve_installed_engine_manifest(
            "colibri",
            explicit=colibri,
            install_root=install_root,
            environment=environment,
        ),
    )


__all__ = [
    "InstalledEngineManifests",
    "discover_installed_engine_manifests",
    "resolve_installed_engine_manifest",
]
