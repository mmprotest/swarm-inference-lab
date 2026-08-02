"""Verification and fingerprint helpers for the pinned Colibri dependency."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from swarm_inference.backends.colibri.constants import (
    COLIBRI_COMMIT,
    COLIBRI_LICENSE,
    COLIBRI_RELEASE,
    COLIBRI_REPOSITORY,
    COLIBRI_REPOSITORY_URL,
)
from swarm_inference.protocol.checksums import sha256_file


def _git(path: Path, *arguments: str) -> str:
    safe = path.resolve().as_posix()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={safe}", "-C", str(path), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git exited {result.returncode}")
    return result.stdout.strip()


def verify_colibri_checkout(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Colibri checkout does not exist: {root}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit != COLIBRI_COMMIT:
        raise ValueError(f"Colibri commit mismatch: expected {COLIBRI_COMMIT}, found {commit}")
    license_path = root / "LICENSE"
    if not license_path.is_file():
        raise FileNotFoundError("Colibri Apache-2.0 license is missing")
    license_text = license_path.read_text(encoding="utf-8", errors="strict")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Colibri LICENSE is not the expected Apache-2.0 license text")
    return {
        "repository": COLIBRI_REPOSITORY,
        "repository_url": COLIBRI_REPOSITORY_URL,
        "release": COLIBRI_RELEASE,
        "commit": commit,
        "license": COLIBRI_LICENSE,
        "license_sha256": sha256_file(license_path),
        "checkout": str(root),
    }


def patch_manifest(integration_root: str | Path) -> dict[str, Any]:
    root = Path(integration_root).expanduser().resolve()
    names = [
        line.strip()
        for line in (root / "patches" / "series").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    patches = []
    for name in names:
        path = root / "patches" / name
        if not path.is_file():
            raise FileNotFoundError(f"missing Colibri patch {path}")
        patches.append({"name": name, "sha256": sha256_file(path)})
    return {
        "schema_version": "experiment-009-colibri-patches-v1",
        "upstream_commit": COLIBRI_COMMIT,
        "patches": patches,
    }


def binary_fingerprint(paths: list[str | Path]) -> str:
    digest = hashlib.sha256()
    for path_value in sorted(Path(item).resolve() for item in paths):
        name = path_value.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path_value)))
    return digest.hexdigest()


def load_build_manifest(path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("commit") != COLIBRI_COMMIT:
        raise ValueError("Colibri build manifest does not match the pinned commit")
    return payload
