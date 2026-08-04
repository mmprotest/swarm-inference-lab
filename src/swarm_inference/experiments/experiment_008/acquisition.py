"""Reproducible model and native llama.cpp binary acquisition for Experiment 008."""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from swarm_inference.config.experiment_008 import Experiment008ModelCandidate
from swarm_inference.experiments.experiment_008.backend import file_sha256

LEVEL_B_MINIMUM_FREE_BYTES = 65 * 1024**3


class ModelAcquisitionPreflightError(RuntimeError):
    """Model acquisition failure carrying the checks completed before download."""

    def __init__(self, message: str, *, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


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
    cache_path: str | None = None
    expected_file_size: int | None = None
    expected_file_sha256: str | None = None
    acquisition_checks: dict[str, Any] = field(default_factory=dict)

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


def _cache_volume(cache_dir: Path) -> dict[str, Any]:
    """Return a fail-closed local-volume classification for the cache path."""

    resolved = cache_dir.expanduser().resolve()
    if os.name == "nt":
        import ctypes

        root = resolved.anchor
        drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(root))  # type: ignore[attr-defined]
        names = {
            0: "UNKNOWN",
            1: "NO_ROOT_DIRECTORY",
            2: "REMOVABLE",
            3: "FIXED",
            4: "REMOTE",
            5: "CDROM",
            6: "RAMDISK",
        }
        return {
            "path": str(resolved),
            "volume_root": root,
            "drive_type": names.get(drive_type, f"UNKNOWN_{drive_type}"),
            "local": drive_type in {3, 6},
        }

    network_filesystems = {
        "9p",
        "afs",
        "cifs",
        "fuse.sshfs",
        "nfs",
        "nfs4",
        "smbfs",
    }
    try:
        import psutil

        matches = [
            item
            for item in psutil.disk_partitions(all=True)
            if resolved == Path(item.mountpoint) or Path(item.mountpoint) in resolved.parents
        ]
        selected = max(matches, key=lambda item: len(item.mountpoint), default=None)
    except (ImportError, OSError):
        selected = None
    filesystem = str(selected.fstype).lower() if selected is not None else "unknown"
    return {
        "path": str(resolved),
        "volume_root": selected.mountpoint if selected is not None else resolved.anchor,
        "drive_type": filesystem,
        "local": filesystem not in network_filesystems,
    }


def _sibling_metadata(sibling: Any) -> tuple[int | None, str | None]:
    size = getattr(sibling, "size", None)
    expected_size = int(size) if isinstance(size, int) and size >= 0 else None
    lfs = getattr(sibling, "lfs", None)
    expected_sha: str | None = None
    if isinstance(lfs, dict):
        candidate = lfs.get("sha256") or lfs.get("oid")
        if isinstance(candidate, str):
            expected_sha = candidate.removeprefix("sha256:").lower()
    elif lfs is not None:
        candidate = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        if isinstance(candidate, str):
            expected_sha = candidate.removeprefix("sha256:").lower()
    if expected_sha is not None and (
        len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        expected_sha = None
    return expected_size, expected_sha


def _verify_downloaded_model(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Hugging Face did not materialize the configured file: {path}")
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f"downloaded model size mismatch: expected {expected_size} bytes, got {actual_size} bytes"
        )
    digest = file_sha256(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256:
        raise RuntimeError(
            f"downloaded model SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return actual_size, digest


def resolve_model_candidate(
    candidate: Experiment008ModelCandidate,
    *,
    candidate_name: str,
    model_path: Path | None,
    cache_dir: Path,
    skip_download: bool,
    require_exact_filename: bool = False,
    minimum_free_bytes: int = 0,
    require_local_cache_volume: bool = False,
    require_public_repository: bool = False,
) -> ResolvedModel:
    """Resolve one configured candidate to an immutable revision and content hash."""

    if model_path is not None:
        path = _find_local_model(model_path, candidate.filename)
        if require_exact_filename and path.name != candidate.filename:
            raise ValueError(
                f"required model filename is {candidate.filename}, but the supplied file is {path.name}"
            )
        digest = file_sha256(path)
        resolved_revision = f"sha256:{digest}"
        expected_size: int | None = None
        expected_sha256: str | None = None
        local_checks: dict[str, Any] = {
            "schema_version": "experiment-008-model-acquisition-preflight-v1",
            "download_requested": False,
            "exact_filename": path.name == candidate.filename,
            "local_file_verified": True,
            "materialized_path": str(path),
            "model_file_size_bytes": path.stat().st_size,
            "model_file_sha256": digest,
        }
        if require_public_repository:
            try:
                import huggingface_hub
                from huggingface_hub import HfApi
            except ImportError as exc:
                local_checks["huggingface_hub_installed"] = False
                raise ModelAcquisitionPreflightError(
                    "huggingface-hub is required to verify the official local model",
                    receipt=local_checks,
                ) from exc
            local_checks["huggingface_hub_installed"] = True
            local_checks["huggingface_hub_version"] = huggingface_hub.__version__
            try:
                info = HfApi(token=False).model_info(
                    candidate.artifact_repository,
                    revision=candidate.revision,
                    files_metadata=True,
                    token=False,
                )
            except Exception as exc:
                local_checks.update(
                    {
                        "huggingface_connectivity": False,
                        "public_repository_accessible_without_token": False,
                        "resolution_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise ModelAcquisitionPreflightError(
                    f"official Hugging Face repository could not be resolved at {candidate.revision}: {exc}",
                    receipt=local_checks,
                ) from exc
            sibling = next(
                (item for item in info.siblings if item.rfilename == candidate.filename), None
            )
            if sibling is None:
                local_checks["configured_filename_exists"] = False
                raise ModelAcquisitionPreflightError(
                    f"configured file {candidate.filename} is absent from "
                    f"{candidate.artifact_repository}@{info.sha}",
                    receipt=local_checks,
                )
            expected_size, expected_sha256 = _sibling_metadata(sibling)
            try:
                actual_size, digest = _verify_downloaded_model(
                    path,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                )
            except Exception as exc:
                local_checks["local_file_valid"] = False
                raise ModelAcquisitionPreflightError(
                    f"supplied local model does not match the official pinned artifact: {exc}",
                    receipt=local_checks,
                ) from exc
            resolved_revision = str(info.sha)
            local_checks.update(
                {
                    "huggingface_connectivity": True,
                    "public_repository_accessible_without_token": True,
                    "configured_filename_exists": True,
                    "resolved_revision": resolved_revision,
                    "repository_file_size_bytes": expected_size,
                    "repository_file_sha256": expected_sha256,
                    "local_file_valid": True,
                    "model_file_size_bytes": actual_size,
                    "model_file_sha256": digest,
                }
            )
        return ResolvedModel(
            candidate=candidate_name,
            model_id=candidate.model_id,
            artifact_repository=candidate.artifact_repository,
            filename=path.name,
            quantization=candidate.quantization,
            architecture=candidate.architecture,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            path=str(path),
            source="user-supplied-local-file",
            file_size=path.stat().st_size,
            file_sha256=digest,
            expected_file_size=expected_size,
            expected_file_sha256=expected_sha256,
            acquisition_checks=local_checks,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": "experiment-008-model-acquisition-preflight-v1",
        "artifact_repository": candidate.artifact_repository,
        "filename": candidate.filename,
        "requested_revision": candidate.revision,
        "cache_path": str(cache_dir.resolve()),
        "minimum_free_bytes": int(minimum_free_bytes),
        "skip_download": skip_download,
        "public_unauthenticated_resolution_requested": require_public_repository,
    }
    try:
        import huggingface_hub
        from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
    except ImportError as exc:
        receipt["huggingface_hub_installed"] = False
        raise ModelAcquisitionPreflightError(
            "huggingface-hub is required to resolve the configured model", receipt=receipt
        ) from exc

    receipt["huggingface_hub_installed"] = True
    receipt["huggingface_hub_version"] = huggingface_hub.__version__
    volume = _cache_volume(cache_dir)
    receipt["cache_volume"] = volume
    if require_local_cache_volume and not volume["local"]:
        raise ModelAcquisitionPreflightError(
            f"model cache must be on a local volume: {cache_dir.resolve()}", receipt=receipt
        )

    requested = candidate.revision or "main"
    try:
        api = HfApi(token=False if require_public_repository else None)
        info = api.model_info(
            candidate.artifact_repository,
            revision=requested,
            files_metadata=True,
            token=False if require_public_repository else None,
        )
    except Exception as exc:
        receipt.update(
            {
                "huggingface_connectivity": False,
                "public_repository_accessible_without_token": False,
                "resolution_error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise ModelAcquisitionPreflightError(
            f"official Hugging Face repository could not be resolved at {requested}: {exc}",
            receipt=receipt,
        ) from exc
    receipt["huggingface_connectivity"] = True
    receipt["public_repository_accessible_without_token"] = require_public_repository
    resolved_revision = str(info.sha)
    receipt["resolved_revision"] = resolved_revision
    sibling = next((item for item in info.siblings if item.rfilename == candidate.filename), None)
    if sibling is None:
        receipt["configured_filename_exists"] = False
        raise ModelAcquisitionPreflightError(
            f"configured file {candidate.filename} is absent from {candidate.artifact_repository}@{resolved_revision}",
            receipt=receipt,
        )
    receipt["configured_filename_exists"] = True
    expected_size, expected_sha256 = _sibling_metadata(sibling)
    receipt["repository_file_size_bytes"] = expected_size
    receipt["repository_file_sha256"] = expected_sha256

    cached = try_to_load_from_cache(
        candidate.artifact_repository,
        candidate.filename,
        revision=resolved_revision,
        cache_dir=cache_dir,
    )
    cached_path = (
        Path(cached).resolve() if isinstance(cached, str) and cached != _CACHED_NO_EXIST else None
    )
    receipt["verified_cached_file_present_before_download"] = cached_path is not None
    partial_files = sorted(
        str(path.resolve()) for path in cache_dir.rglob("*.incomplete") if path.is_file()
    )
    receipt["incomplete_files_before_download"] = partial_files
    receipt["resumable_cache_ready"] = True

    if cached_path is not None:
        try:
            actual_size, digest = _verify_downloaded_model(
                cached_path,
                expected_size=expected_size,
                expected_sha256=expected_sha256,
            )
        except Exception as exc:
            receipt["cached_file_valid"] = False
            receipt["cached_file_error"] = f"{type(exc).__name__}: {exc}"
            raise ModelAcquisitionPreflightError(
                f"cached official model is invalid and was not silently replaced: {exc}",
                receipt=receipt,
            ) from exc
        receipt.update(
            {
                "cached_file_valid": True,
                "cache_reused": True,
                "download_started": False,
                "available_free_bytes": shutil.disk_usage(cache_dir).free,
                "model_file_size_bytes": actual_size,
                "model_file_sha256": digest,
            }
        )
        return ResolvedModel(
            candidate=candidate_name,
            model_id=candidate.model_id,
            artifact_repository=candidate.artifact_repository,
            filename=candidate.filename,
            quantization=candidate.quantization,
            architecture=candidate.architecture,
            requested_revision=candidate.revision,
            resolved_revision=resolved_revision,
            path=str(cached_path),
            source="huggingface-hub",
            file_size=actual_size,
            file_sha256=digest,
            cache_path=str(cache_dir.resolve()),
            expected_file_size=expected_size,
            expected_file_sha256=expected_sha256,
            acquisition_checks=receipt,
        )

    available_free_bytes = shutil.disk_usage(cache_dir).free
    receipt["available_free_bytes"] = available_free_bytes
    if skip_download:
        raise ModelAcquisitionPreflightError(
            f"{candidate.artifact_repository}/{candidate.filename}@{resolved_revision} is not cached",
            receipt=receipt,
        )
    if minimum_free_bytes > 0 and available_free_bytes < minimum_free_bytes:
        raise ModelAcquisitionPreflightError(
            "insufficient free storage for model download: "
            f"required free bytes={minimum_free_bytes}, available free bytes={available_free_bytes}, "
            f"cache path={cache_dir.resolve()}",
            receipt=receipt,
        )

    receipt["download_started"] = True
    try:
        downloaded = hf_hub_download(
            repo_id=candidate.artifact_repository,
            filename=candidate.filename,
            revision=resolved_revision,
            cache_dir=cache_dir,
            token=False if require_public_repository else None,
            force_download=False,
            resume_download=True,
        )
        path = Path(downloaded).resolve()
        actual_size, digest = _verify_downloaded_model(
            path,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    except Exception as exc:
        receipt["download_error"] = f"{type(exc).__name__}: {exc}"
        raise ModelAcquisitionPreflightError(
            f"official model download or verification failed: {exc}", receipt=receipt
        ) from exc
    receipt.update(
        {
            "cache_reused": False,
            "download_completed": True,
            "model_file_size_bytes": actual_size,
            "model_file_sha256": digest,
            "materialized_path": str(path),
        }
    )
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
        file_size=actual_size,
        file_sha256=digest,
        cache_path=str(cache_dir.resolve()),
        expected_file_size=expected_size,
        expected_file_sha256=expected_sha256,
        acquisition_checks=receipt,
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
            "configured_release_tag": release_tag,
            "configured_cuda_version": cuda_version,
            "path": str(executable),
            "sha256": file_sha256(executable),
        }

    located = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if located:
        executable = Path(located).resolve()
        return executable, {
            "source": "PATH",
            "configured_release_tag": release_tag,
            "configured_cuda_version": cuda_version,
            "path": str(executable),
            "sha256": file_sha256(executable),
        }
    release_root = destination_root / release_tag / f"cuda-{cuda_version}"
    existing = next(iter(release_root.rglob("llama-server.exe")), None)
    if existing is not None:
        return existing.resolve(), {
            "source": "cached-official-release",
            "repository": "ggml-org/llama.cpp",
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
