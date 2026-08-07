"""Canonical local paths for immutable model sources."""

from pathlib import Path


def materialized_snapshot_path(
    cache_root: Path,
    model_id: str,
    model_revision: str,
) -> Path:
    """Return the direct-file snapshot path for one immutable model revision."""

    return cache_root / "materialized" / model_id.replace("/", "--") / "snapshots" / model_revision
