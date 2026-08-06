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

COLIBRI_SOURCE_REMEDIATION = "git submodule update --init --recursive third_party/colibri"


class ColibriSourceDependencyError(RuntimeError):
    """The source-only pinned checkout contract is absent or inconsistent."""


def _source_dependency_error(detail: str) -> ColibriSourceDependencyError:
    return ColibriSourceDependencyError(f"{detail}. Remediation: {COLIBRI_SOURCE_REMEDIATION}")


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
    if not root.is_dir() or not (root / ".git").exists():
        raise _source_dependency_error(f"pinned Colibri source checkout is missing at {root}")
    try:
        commit = _git(root, "rev-parse", "HEAD")
    except RuntimeError as exc:
        raise _source_dependency_error(f"cannot inspect pinned Colibri checkout: {exc}") from exc
    if commit != COLIBRI_COMMIT:
        raise _source_dependency_error(
            f"Colibri commit mismatch: expected {COLIBRI_COMMIT}, found {commit}"
        )
    license_path = root / "LICENSE"
    if not license_path.is_file():
        raise _source_dependency_error("Colibri Apache-2.0 license is missing")
    license_text = license_path.read_text(encoding="utf-8", errors="strict")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise _source_dependency_error(
            "Colibri LICENSE is not the expected Apache-2.0 license text"
        )
    return {
        "repository": COLIBRI_REPOSITORY,
        "repository_url": COLIBRI_REPOSITORY_URL,
        "release": COLIBRI_RELEASE,
        "commit": commit,
        "license": COLIBRI_LICENSE,
        "license_sha256": sha256_file(license_path),
        "checkout": str(root),
    }


def verify_colibri_source_contract(repository_root: str | Path) -> dict[str, Any]:
    """Verify the source-only submodule pin and licence against dependency.json."""

    root = Path(repository_root).expanduser().resolve()
    dependency_path = root / "integrations" / "colibri" / "dependency.json"
    if not dependency_path.is_file():
        raise _source_dependency_error(
            f"Colibri dependency manifest is missing at {dependency_path}"
        )
    try:
        dependency = json.loads(dependency_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _source_dependency_error(f"Colibri dependency manifest is invalid: {exc}") from exc
    expected = {
        "commit": COLIBRI_COMMIT,
        "release": COLIBRI_RELEASE,
        "license": COLIBRI_LICENSE,
        "repository": COLIBRI_REPOSITORY,
        "repository_url": COLIBRI_REPOSITORY_URL,
    }
    mismatches = [
        f"{key} expected {value!r}, found {dependency.get(key)!r}"
        for key, value in expected.items()
        if dependency.get(key) != value
    ]
    if mismatches:
        raise _source_dependency_error(
            "Colibri dependency manifest mismatch: " + "; ".join(mismatches)
        )
    checkout = verify_colibri_checkout(root / "third_party" / "colibri")
    declared_license = root / str(dependency.get("license_path", ""))
    if declared_license.resolve() != (root / "third_party" / "colibri" / "LICENSE").resolve():
        raise _source_dependency_error("Colibri dependency manifest has an unexpected licence path")
    if not declared_license.is_file():
        raise _source_dependency_error(f"declared Colibri licence is missing at {declared_license}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "dependency_manifest": str(dependency_path),
        **checkout,
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
