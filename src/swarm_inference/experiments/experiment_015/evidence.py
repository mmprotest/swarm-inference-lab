"""Inspectable evidence utilities for Experiment 015."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.contracts import Experiment015Error


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 identity of one regular file."""
    source = path.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise Experiment015Error(f"evidence source is absent, non-regular, or linked: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    """Create a stable size/hash receipt for one evidence source."""
    source = path.expanduser().resolve()
    display = source
    if relative_to is not None:
        with suppress(ValueError):
            display = source.relative_to(relative_to.expanduser().resolve())
    return {
        "path": display.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject ambiguous/non-object evidence."""
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Experiment015Error(f"invalid JSON evidence: {source}") from exc
    if not isinstance(value, dict):
        raise Experiment015Error(f"expected a JSON object: {source}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic JSON without exposing a partial artifact."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)


def atomic_text(path: Path, value: str) -> None:
    """Write UTF-8 text without exposing a partial artifact."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, destination)
