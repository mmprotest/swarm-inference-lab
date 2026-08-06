from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest

from swarm_inference import native_install, native_update


def _release(*, tag: str, prerelease: bool) -> dict[str, object]:
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": False,
        "html_url": f"https://github.com/{native_update.REPOSITORY}/releases/tag/{tag}",
        "assets": [],
    }


def _manifest(*, version: str, tag: str, signature_status: str) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "manifest_scope": "release",
            "product": "swarm-inference-lab",
            "version": version,
            "git_tag": tag,
            "git_commit": "a" * 40,
            "channel": "prerelease" if "rc" in version else "stable",
            "installer": {
                "filename": native_update.INSTALLER_FILENAME,
                "sha256": "sha256:" + "b" * 64,
                "signature_status": signature_status,
            },
        }
    ).encode()


def test_native_update_tag_mapping_is_exact() -> None:
    assert native_update._version_from_tag("v0.1.0-rc.1") == "0.1.0rc1"
    assert native_update._tag_from_version("0.1.0rc1") == "v0.1.0-rc.1"
    assert native_update._version_from_tag("v1.2.3") == "1.2.3"
    with pytest.raises(ValueError, match="unsupported"):
        native_update._version_from_tag("latest")


def test_native_update_rejects_foreign_redirects() -> None:
    handler = native_update._RestrictedRedirect()
    request = urllib.request.Request("https://github.com/example")
    with pytest.raises(RuntimeError, match="untrusted host"):
        handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            {},
            "https://downloads.example.invalid/setup.exe",
        )


def test_native_update_rejects_unsigned_stable_manifest() -> None:
    release = _release(tag="v0.1.0", prerelease=False)
    raw = _manifest(version="0.1.0", tag="v0.1.0", signature_status="unsigned-prerelease")
    with pytest.raises(RuntimeError, match="stable native releases must have a signed installer"):
        native_update._validated_manifest(raw, release)


def test_native_update_accepts_explicit_unsigned_prerelease_manifest() -> None:
    release = _release(tag="v0.1.0-rc.1", prerelease=True)
    raw = _manifest(
        version="0.1.0rc1",
        tag="v0.1.0-rc.1",
        signature_status="unsigned-prerelease",
    )
    assert native_update._validated_manifest(raw, release)["git_commit"] == "a" * 40


def test_native_install_detection_requires_matching_runtime_and_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "Programs" / "SwarmInference"
    executable = root / "runtime" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    record_path = root / "app" / "install-record.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "installation_mode": "native-windows",
                "application_path": str(root.resolve()),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    detected = native_install.native_install_record()
    assert detected is not None
    assert detected[0] == root.resolve()


def test_native_update_stages_verified_setup_and_manifest_together(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup = b"fixture setup"
    setup_digest = hashlib.sha256(setup).hexdigest()
    release = _release(tag="v0.1.0-rc.1", prerelease=True)
    release["assets"] = [
        {
            "name": native_update.MANIFEST_FILENAME,
            "browser_download_url": (
                "https://github.com/mmprotest/swarm-inference-lab/releases/download/"
                "v0.1.0-rc.1/release-manifest.json"
            ),
        },
        {
            "name": native_update.INSTALLER_FILENAME,
            "browser_download_url": (
                "https://github.com/mmprotest/swarm-inference-lab/releases/download/"
                "v0.1.0-rc.1/SwarmInferenceSetup-x64.exe"
            ),
        },
    ]
    manifest = json.loads(
        _manifest(
            version="0.1.0rc1",
            tag="v0.1.0-rc.1",
            signature_status="unsigned-prerelease",
        )
    )
    manifest["installer"]["sha256"] = f"sha256:{setup_digest}"
    raw_manifest = json.dumps(manifest, sort_keys=True).encode()
    install_root = tmp_path / "Programs" / "SwarmInference"
    monkeypatch.setattr(native_update, "native_install_record", lambda: (install_root, {}))
    monkeypatch.setattr(native_update, "_select_release", lambda **_: release)
    monkeypatch.setattr(native_update, "_authenticode_valid", lambda _: False)

    def download(url: str, destination: Path, **_: object) -> str:
        data = raw_manifest if destination.name == native_update.MANIFEST_FILENAME else setup
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(native_update, "_download", download)
    result = native_update.prepare_native_update(channel="prerelease", launch=False)
    controlled_setup = Path(result.installer_path)
    assert controlled_setup.name == native_update.INSTALLER_FILENAME
    assert controlled_setup.read_bytes() == setup
    assert (controlled_setup.parent / native_update.MANIFEST_FILENAME).read_bytes() == raw_manifest
