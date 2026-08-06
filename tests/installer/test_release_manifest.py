from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from release_common import (
    ReleaseError,
    manifest_file_entries,
    sha256_file,
    validate_manifest,
    verify_manifest_files,
)
from sign_windows_artifact import _redact_sensitive

HASH = "sha256:" + "1" * 64


def _file(filename: str) -> dict[str, Any]:
    return {"filename": filename, "sha256": HASH, "size_bytes": 1}


def _signed(filename: str) -> dict[str, Any]:
    return {
        **_file(filename),
        "signature_status": "unsigned-prerelease",
        "signature_verification": "not-signed",
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_scope": "release",
        "product": "swarm-inference-lab",
        "version": "0.1.0rc1",
        "git_tag": "v0.1.0-rc.1",
        "git_commit": "a" * 40,
        "channel": "prerelease",
        "built_at_utc": "2026-08-06T00:00:00Z",
        "minimum_windows": "10.0.22621",
        "architecture": "x86_64",
        "python": {"version": "3.11.15"},
        "uv": {"version": "0.12.0", **_file("uv.exe")},
        "wheel": _file("swarm_inference_lab-0.1.0rc1-py3-none-any.whl"),
        "runtime_profiles": {
            "windows-x64-cpu": _file("windows-x64-cpu.requirements.lock"),
            "windows-x64-cuda": _file("windows-x64-cuda.requirements.lock"),
        },
        "bootstrapper": _signed("SwarmBootstrap.exe"),
        "installer": _signed("SwarmInferenceSetup-x64.exe"),
        "payload": [
            _file("LICENSE"),
            _file("swarm.ico"),
            _file("wizard-small.bmp"),
            _file("wizard-large.bmp"),
        ],
    }


def test_unsigned_prerelease_is_explicit_and_strict() -> None:
    manifest = validate_manifest(_manifest())
    assert manifest["channel"] == "prerelease"
    assert manifest["installer"]["signature_status"] == "unsigned-prerelease"
    malformed = copy.deepcopy(manifest)
    malformed["unexpected"] = True
    with pytest.raises(ReleaseError, match="keys mismatch"):
        validate_manifest(malformed)


def test_signing_diagnostics_redact_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WINDOWS_SIGNING_PASSWORD", "do-not-print-this-password")
    monkeypatch.setenv("GITHUB_TOKEN", "do-not-print-this-token")
    diagnostic = _redact_sensitive(
        "password=do-not-print-this-password token=do-not-print-this-token"
    )
    assert "do-not-print" not in diagnostic
    assert diagnostic.count("<redacted>") == 2


def test_stable_release_rejects_unsigned_executables() -> None:
    manifest = _manifest()
    manifest.update(version="0.1.0", git_tag="v0.1.0", channel="stable")
    with pytest.raises(ReleaseError, match="stable releases require signed"):
        validate_manifest(manifest)


def test_duplicate_filename_and_non_sha256_identity_are_rejected() -> None:
    duplicate = _manifest()
    duplicate["payload"][0]["filename"] = duplicate["wheel"]["filename"]
    with pytest.raises(ReleaseError, match="duplicate filenames"):
        validate_manifest(duplicate)
    malformed = _manifest()
    malformed["wheel"]["sha256"] = "sha512:" + "0" * 128
    with pytest.raises(ReleaseError, match="canonical SHA-256"):
        validate_manifest(malformed)


def test_manifest_verifies_every_hash_and_rejects_untracked_payload(tmp_path: Path) -> None:
    manifest = _manifest()
    for index, entry in enumerate(manifest_file_entries(manifest)):
        path = tmp_path / entry["filename"]
        path.write_bytes(f"payload-{index}".encode())
        entry["sha256"] = sha256_file(path)
        entry["size_bytes"] = path.stat().st_size
    verify_manifest_files(manifest, tmp_path, allow_manifest=False)
    (tmp_path / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(ReleaseError, match="unexpected files"):
        verify_manifest_files(manifest, tmp_path, allow_manifest=False)
    (tmp_path / "unexpected.bin").unlink()
    (tmp_path / manifest["wheel"]["filename"]).write_bytes(b"tampered")
    with pytest.raises(ReleaseError, match="SHA-256 mismatch"):
        verify_manifest_files(manifest, tmp_path, allow_manifest=False)
