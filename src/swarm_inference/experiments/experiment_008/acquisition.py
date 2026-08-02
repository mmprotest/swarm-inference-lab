"""Reproducible model and native llama.cpp binary acquisition for Experiment 008."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.config.experiment_008 import Experiment008ModelCandidate
from swarm_inference.experiments.experiment_008.backend import file_sha256


@dataclass(slots=True)
class ResolvedModel:
    candidate: str
    model_id: str
    artifact_repository: str
    filename: str
    quantization: str
    architecture: str
    requested_revision: str | None
    resolved_revision: str
    path: str
    source: str
    file_size: int
    file_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_local_model(model_path: Path, filename: str) -> Path:
    resolved = model_path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"model path does not exist: {resolved}")
    exact = resolved / filename
    if exact.is_file():
        return exact
    ggufs = sorted(resolved.glob("*.gguf"))
    if len(ggufs) == 1:
        return ggufs[0]
    if not ggufs:
        raise FileNotFoundError(f"no GGUF file found in {resolved}")
    raise ValueError(
        f"multiple GGUF files found in {resolved}; pass the exact model file with -ModelPath"
    )


def resolve_model_candidate(
    candidate: Experiment008ModelCandidate,
    *,
    candidate_name: str,
    model_path: Path | None,
    cache_dir: Path,
    skip_download: bool,
) -> ResolvedModel:
    """Resolve one configured candidate to an immutable revision and content hash."""

    if model_path is not None:
        path = _find_local_model(model_path, candidate.filename)
        digest = file_sha256(path)
        return ResolvedModel(
            candidate=candidate_name,
            model_id=candidate.model_id,
            artifact_repository=candidate.artifact_repository,
            filename=path.name,
            quantization=candidate.quantization,
            architecture=candidate.architecture,
            requested_revision=candidate.revision,
            resolved_revision=f"sha256:{digest}",
            path=str(path),
            source="user-supplied-local-file",
            file_size=path.stat().st_size,
            file_sha256=digest,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to resolve the configured model") from exc

    requested = candidate.revision or "main"
    if skip_download:
        cached = try_to_load_from_cache(
            candidate.artifact_repository,
            candidate.filename,
            revision=requested,
            cache_dir=cache_dir,
        )
        if not isinstance(cached, str) or cached == _CACHED_NO_EXIST:
            raise FileNotFoundError(
                f"{candidate.artifact_repository}/{candidate.filename}@{requested} is not cached"
            )
        path = Path(cached).resolve()
        digest = file_sha256(path)
        return ResolvedModel(
            candidate=candidate_name,
            model_id=candidate.model_id,
            artifact_repository=candidate.artifact_repository,
            filename=candidate.filename,
            quantization=candidate.quantization,
            architecture=candidate.architecture,
            requested_revision=candidate.revision,
            resolved_revision=f"cached-{requested};sha256:{digest}",
            path=str(path),
            source="huggingface-cache-offline",
            file_size=path.stat().st_size,
            file_sha256=digest,
        )

    info = HfApi().model_info(
        candidate.artifact_repository, revision=requested, files_metadata=True
    )
    resolved_revision = str(info.sha)
    sibling = next((item for item in info.siblings if item.rfilename == candidate.filename), None)
    if sibling is None:
        raise FileNotFoundError(
            f"configured file {candidate.filename} is absent from {candidate.artifact_repository}@{resolved_revision}"
        )
    path = Path(
        hf_hub_download(
            repo_id=candidate.artifact_repository,
            filename=candidate.filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
        )
    ).resolve()
    digest = file_sha256(path)
    return ResolvedModel(
        candidate=candidate_name,
        model_id=candidate.model_id,
        artifact_repository=candidate.artifact_repository,
        filename=candidate.filename,
        quantization=candidate.quantization,
        architecture=candidate.architecture,
        requested_revision=candidate.revision,
        resolved_revision=resolved_revision,
        path=str(path),
        source="huggingface-hub",
        file_size=path.stat().st_size,
        file_sha256=digest,
    )


@dataclass(slots=True)
class DownloadedAsset:
    name: str
    url: str
    path: str
    byte_size: int
    sha256: str


def _download(url: str, destination: Path) -> DownloadedAsset:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "swarm-inference-lab/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=8 << 20)
    except (OSError, urllib.error.URLError):
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)
    return DownloadedAsset(
        name=destination.name,
        url=url,
        path=str(destination),
        byte_size=destination.stat().st_size,
        sha256=file_sha256(destination),
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"unsafe path in release archive: {member.filename}") from exc
        handle.extractall(destination)


def _release_assets(tag: str) -> dict[str, str]:
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
    request = urllib.request.Request(url, headers={"User-Agent": "swarm-inference-lab/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    assets = payload.get("assets") if isinstance(payload, dict) else None
    if not isinstance(assets, list):
        raise RuntimeError(f"GitHub release {tag} did not expose assets")
    return {
        str(item["name"]): str(item["browser_download_url"])
        for item in assets
        if isinstance(item, dict) and "name" in item and "browser_download_url" in item
    }


def resolve_llama_server(
    *,
    supplied_path: Path | None,
    allow_download: bool,
    release_tag: str,
    cuda_version: str,
    destination_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve an exact native server or download official pinned Windows assets."""

    if supplied_path is not None:
        executable = supplied_path.expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"supplied llama-server path is not a file: {executable}")
        return executable, {
            "source": "user-supplied",
            "path": str(executable),
            "sha256": file_sha256(executable),
        }

    located = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if located:
        executable = Path(located).resolve()
        return executable, {
            "source": "PATH",
            "path": str(executable),
            "sha256": file_sha256(executable),
        }
    release_root = destination_root / release_tag / f"cuda-{cuda_version}"
    existing = next(iter(release_root.rglob("llama-server.exe")), None)
    if existing is not None:
        return existing.resolve(), {
            "source": "cached-official-release",
            "release_tag": release_tag,
            "cuda_version": cuda_version,
            "path": str(existing.resolve()),
            "sha256": file_sha256(existing),
        }
    if not allow_download:
        raise FileNotFoundError(
            "no native llama-server executable was supplied, installed, or cached"
        )
    assets = _release_assets(release_tag)
    main_name = f"llama-{release_tag}-bin-win-cuda-{cuda_version}-x64.zip"
    runtime_name = f"cudart-llama-bin-win-cuda-{cuda_version}-x64.zip"
    missing = [name for name in (main_name, runtime_name) if name not in assets]
    if missing:
        candidates = sorted(name for name in assets if "win-cuda" in name or "cudart" in name)
        raise RuntimeError(
            f"release {release_tag} lacks expected assets {missing}; available CUDA assets: {candidates}"
        )
    downloads = release_root / "downloads"
    acquired: list[DownloadedAsset] = []
    for name in (main_name, runtime_name):
        archive = downloads / name
        if archive.is_file():
            acquired.append(
                DownloadedAsset(
                    name=name,
                    url=assets[name],
                    path=str(archive),
                    byte_size=archive.stat().st_size,
                    sha256=file_sha256(archive),
                )
            )
        else:
            acquired.append(_download(assets[name], archive))
        _safe_extract(archive, release_root)
    executable = next(iter(release_root.rglob("llama-server.exe")), None)
    if executable is None:
        raise RuntimeError("official llama.cpp release did not contain llama-server.exe")
    manifest = {
        "source": "official-github-release",
        "repository": "ggml-org/llama.cpp",
        "release_tag": release_tag,
        "cuda_version": cuda_version,
        "path": str(executable.resolve()),
        "sha256": file_sha256(executable),
        "assets": [asdict(item) for item in acquired],
    }
    (release_root / "acquisition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return executable.resolve(), manifest
