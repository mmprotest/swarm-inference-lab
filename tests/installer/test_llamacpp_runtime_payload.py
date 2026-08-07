from __future__ import annotations

from pathlib import Path
from typing import Any

import build_windows_installer
import pytest
from generate_release_manifest import _engine_runtime_manifest
from release_common import ReleaseError, sha256_file


def _archive(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "filename": path.name,
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/b9637/{path.name}",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _profile(archives: list[dict[str, Any]], *, cuda: bool) -> dict[str, Any]:
    digest = "sha256:" + "1" * 64
    return {
        "platform": "windows-x64",
        "archives": archives,
        "server_binary": "llama-server.exe",
        "server_sha256": digest,
        "rpc_server_binary": "rpc-server.exe",
        "rpc_server_sha256": digest,
        "build_flags": {"GGML_CUDA": cuda, "GGML_RPC": True},
        "device_support": ["CPU", "CUDA"] if cuda else ["CPU"],
    }


def test_pinned_llamacpp_archives_are_staged_and_manifested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    cache = root / "build" / "toolchain-downloads"
    cpu = _archive(cache / "llama-cpu.zip", b"cpu")
    cuda = _archive(cache / "llama-cuda.zip", b"cuda")
    cudart = _archive(cache / "cudart.zip", b"runtime")
    metadata = {
        "repository": "ggml-org/llama.cpp",
        "release_tag": "b9637",
        "runtime_revision": "a" * 40,
        "profiles": {
            "windows-x64-cpu": _profile([cpu], cuda=False),
            "windows-x64-cuda": _profile([cuda, cudart], cuda=True),
        },
    }
    payload = tmp_path / "payload"
    payload.mkdir()
    monkeypatch.setattr(build_windows_installer, "ROOT", root)

    build_windows_installer._stage_llamacpp_archives(
        {"llamacpp": metadata},
        payload,
    )

    assert {item.name for item in payload.iterdir()} == {
        "llama-cpu.zip",
        "llama-cuda.zip",
        "cudart.zip",
    }
    runtime = _engine_runtime_manifest(payload, metadata)
    assert runtime["runtime_revision"] == "a" * 40
    assert [item["filename"] for item in runtime["profiles"]["windows-x64-cuda"]["archives"]] == [
        "llama-cuda.zip",
        "cudart.zip",
    ]


def test_tampered_cached_llamacpp_archive_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    cache = root / "build" / "toolchain-downloads"
    pinned = _archive(cache / "llama.zip", b"pinned")
    (cache / "llama.zip").write_bytes(b"tampered")
    metadata = {
        "profiles": {
            "windows-x64-cpu": {"archives": [pinned]},
            "windows-x64-cuda": {"archives": [pinned]},
        }
    }
    payload = tmp_path / "payload"
    payload.mkdir()
    monkeypatch.setattr(build_windows_installer, "ROOT", root)

    with pytest.raises(ReleaseError, match="pinned hash mismatch"):
        build_windows_installer._stage_llamacpp_archives(
            {"llamacpp": metadata},
            payload,
        )
