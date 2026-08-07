"""Verified, release-aware launcher for native Windows installer upgrades."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal
from uuid import uuid4

from swarm_inference.native_install import native_install_record

REPOSITORY = "mmprotest/swarm-inference-lab"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
INSTALLER_FILENAME = "SwarmInferenceSetup-x64.exe"
MANIFEST_FILENAME = "release-manifest.json"
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "api.github.com",
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
_SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")
_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$")
_MAX_METADATA_BYTES = 8 * 1024 * 1024
# The setup embeds both pinned CPU and CUDA llama.cpp runtime archives so a
# normal installation never depends on PATH or a post-install engine download.
_MAX_INSTALLER_BYTES = 1024 * 1024 * 1024


@dataclass(frozen=True)
class NativeUpdateResult:
    tag: str
    version: str
    release_url: str
    installer_path: str
    installer_sha256: str
    signature_status: str
    launched: bool
    process_id: int | None


class _RestrictedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        request: urllib.request.Request,
        file_pointer: BinaryIO,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
            raise RuntimeError(
                f"release download redirected to an untrusted host: {parsed.hostname}"
            )
        return super().redirect_request(request, file_pointer, code, message, headers, new_url)


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_RestrictedRedirect())


def _request(url: str) -> urllib.request.Request:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise RuntimeError("release request URL is not an approved HTTPS host")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "swarm-inference-native-updater/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _read_bounded(response: BinaryIO, maximum: int) -> bytes:
    data = response.read(maximum + 1)
    if len(data) > maximum:
        raise RuntimeError("GitHub response exceeded the bounded size limit")
    return data


def _get_json(url: str, *, timeout_seconds: float) -> Any:
    try:
        with _opener().open(_request(url), timeout=timeout_seconds) as response:
            return json.loads(_read_bounded(response, _MAX_METADATA_BYTES))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"bounded GitHub release request failed: {exc}") from exc


def _version_from_tag(tag: str) -> str:
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ValueError(f"unsupported release tag: {tag}")
    base = ".".join(match.groups(default="")[:3])
    return f"{base}rc{match.group(4)}" if match.group(4) else base


def _tag_from_version(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?", version)
    if match is None:
        raise ValueError(f"unsupported release version: {version}")
    base = f"v{match.group(1)}.{match.group(2)}.{match.group(3)}"
    return f"{base}-rc.{match.group(4)}" if match.group(4) else base


def _select_release(
    *, channel: Literal["stable", "prerelease"], version: str | None, timeout_seconds: float
) -> dict[str, Any]:
    if version:
        tag = version if version.startswith("v") else _tag_from_version(version)
        release = _get_json(
            f"{API_ROOT}/releases/tags/{urllib.parse.quote(tag)}", timeout_seconds=timeout_seconds
        )
    elif channel == "stable":
        release = _get_json(f"{API_ROOT}/releases/latest", timeout_seconds=timeout_seconds)
    else:
        releases = _get_json(f"{API_ROOT}/releases?per_page=20", timeout_seconds=timeout_seconds)
        if not isinstance(releases, list):
            raise RuntimeError("GitHub releases response was not a list")
        release = next(
            (
                item
                for item in releases
                if isinstance(item, dict) and item.get("prerelease") and not item.get("draft")
            ),
            None,
        )
        if release is None:
            raise RuntimeError("no published prerelease is available")
    if not isinstance(release, dict) or release.get("draft"):
        raise RuntimeError("selected GitHub release is invalid or still a draft")
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or _TAG.fullmatch(release_tag) is None:
        raise RuntimeError("selected GitHub release has an unsupported tag")
    if version is None and bool(release.get("prerelease")) != (channel == "prerelease"):
        raise RuntimeError("selected GitHub release does not match the requested channel")
    html_url = release.get("html_url")
    if not isinstance(html_url, str) or not html_url.startswith(
        f"https://github.com/{REPOSITORY}/releases/"
    ):
        raise RuntimeError("selected release does not belong to the expected repository")
    return release


def _asset_url(release: dict[str, Any], filename: str) -> str:
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("selected release has no asset inventory")
    matches = [item for item in assets if isinstance(item, dict) and item.get("name") == filename]
    if len(matches) != 1:
        raise RuntimeError(f"selected release must contain exactly one {filename} asset")
    url = matches[0].get("browser_download_url")
    if not isinstance(url, str):
        raise RuntimeError(f"selected release has no download URL for {filename}")
    parsed = urllib.parse.urlsplit(url)
    expected_prefix = f"/{REPOSITORY}/releases/download/"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or not parsed.path.startswith(expected_prefix)
    ):
        raise RuntimeError(f"selected release has an untrusted download URL for {filename}")
    return url


def _download(url: str, destination: Path, *, maximum: int, timeout_seconds: float) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            _opener().open(_request(url), timeout=timeout_seconds) as response,
            temporary.open("xb") as output,
        ):
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > maximum:
                    raise RuntimeError(f"download of {destination.name} exceeded its size limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"bounded download of {destination.name} failed: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _validated_manifest(raw: bytes, release: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("release manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("release manifest must be an object")
    required = {
        "schema_version",
        "manifest_scope",
        "product",
        "version",
        "git_tag",
        "git_commit",
        "channel",
        "installer",
    }
    if not required.issubset(manifest):
        raise RuntimeError("release manifest is missing identity fields")
    tag = str(release["tag_name"])
    if (
        manifest["schema_version"] != 1
        or manifest["manifest_scope"] != "release"
        or manifest["product"] != "swarm-inference-lab"
        or manifest["git_tag"] != tag
        or manifest["version"] != _version_from_tag(tag)
        or manifest["channel"] != ("prerelease" if release.get("prerelease") else "stable")
        or not re.fullmatch(r"[0-9a-f]{40}", str(manifest["git_commit"]))
    ):
        raise RuntimeError("release manifest identity does not match GitHub release metadata")
    installer = manifest["installer"]
    if not isinstance(installer, dict) or installer.get("filename") != INSTALLER_FILENAME:
        raise RuntimeError("release manifest has an invalid installer identity")
    if _SHA256.fullmatch(str(installer.get("sha256", ""))) is None:
        raise RuntimeError("release manifest installer SHA-256 is invalid")
    if release.get("prerelease"):
        if installer.get("signature_status") not in {"signed", "unsigned-prerelease"}:
            raise RuntimeError("prerelease signature status is not explicit")
    elif installer.get("signature_status") != "signed":
        raise RuntimeError("stable native releases must have a signed installer")
    return manifest


def _authenticode_valid(path: Path) -> bool:
    if os.name != "nt":
        return False

    class Guid(ctypes.Structure):
        _fields_ = [
            ("data1", ctypes.c_ulong),
            ("data2", ctypes.c_ushort),
            ("data3", ctypes.c_ushort),
            ("data4", ctypes.c_ubyte * 8),
        ]

    class WinTrustFileInfo(ctypes.Structure):
        _fields_ = [
            ("cbStruct", ctypes.c_ulong),
            ("pcwszFilePath", ctypes.c_wchar_p),
            ("hFile", ctypes.c_void_p),
            ("pgKnownSubject", ctypes.c_void_p),
        ]

    class WinTrustData(ctypes.Structure):
        _fields_ = [
            ("cbStruct", ctypes.c_ulong),
            ("pPolicyCallbackData", ctypes.c_void_p),
            ("pSIPClientData", ctypes.c_void_p),
            ("dwUIChoice", ctypes.c_ulong),
            ("fdwRevocationChecks", ctypes.c_ulong),
            ("dwUnionChoice", ctypes.c_ulong),
            ("pFile", ctypes.POINTER(WinTrustFileInfo)),
            ("dwStateAction", ctypes.c_ulong),
            ("hWVTStateData", ctypes.c_void_p),
            ("pwszURLReference", ctypes.c_wchar_p),
            ("dwProvFlags", ctypes.c_ulong),
            ("dwUIContext", ctypes.c_ulong),
            ("pSignatureSettings", ctypes.c_void_p),
        ]

    action_guid = Guid(
        0x00AAC56B,
        0xCD44,
        0x11D0,
        (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
    )
    file_info = WinTrustFileInfo(ctypes.sizeof(WinTrustFileInfo), str(path), None, None)
    data = WinTrustData(
        ctypes.sizeof(WinTrustData),
        None,
        None,
        2,
        0,
        1,
        ctypes.pointer(file_info),
        0,
        None,
        None,
        0x00001000,
        0,
        None,
    )
    win_verify_trust = ctypes.windll.wintrust.WinVerifyTrust
    result = win_verify_trust(None, ctypes.byref(action_guid), ctypes.byref(data))
    return int(result) == 0


def prepare_native_update(
    *,
    channel: Literal["stable", "prerelease"] = "stable",
    version: str | None = None,
    timeout_seconds: float = 30.0,
    launch: bool = True,
) -> NativeUpdateResult:
    """Download, verify, and optionally launch the authoritative native installer."""

    if not 5 <= timeout_seconds <= 120:
        raise ValueError("update request timeout must be in [5, 120] seconds")
    installation = native_install_record()
    if installation is None:
        raise RuntimeError(
            "native Windows installation not detected; use the documented developer/offline recovery path"
        )
    install_root, _ = installation
    release = _select_release(channel=channel, version=version, timeout_seconds=timeout_seconds)
    tag = str(release["tag_name"])
    staging = install_root / "update-staging" / tag
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = staging / MANIFEST_FILENAME
    manifest_digest = _download(
        _asset_url(release, MANIFEST_FILENAME),
        manifest_path,
        maximum=_MAX_METADATA_BYTES,
        timeout_seconds=timeout_seconds,
    )
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != manifest_digest:
        raise RuntimeError("controlled manifest staging copy changed after download")
    manifest = _validated_manifest(raw_manifest, release)
    installer_path = staging / INSTALLER_FILENAME
    installer_digest = _download(
        _asset_url(release, INSTALLER_FILENAME),
        installer_path,
        maximum=_MAX_INSTALLER_BYTES,
        timeout_seconds=timeout_seconds,
    )
    expected = str(manifest["installer"]["sha256"])[len("sha256:") :]
    if not hmac.compare_digest(installer_digest, expected):
        installer_path.unlink(missing_ok=True)
        raise RuntimeError("downloaded setup SHA-256 does not match the release manifest")
    controlled_directory = staging / "verified"
    controlled_directory.mkdir(parents=True, exist_ok=True)
    controlled_manifest = controlled_directory / MANIFEST_FILENAME
    shutil.copyfile(manifest_path, controlled_manifest)
    if hashlib.sha256(controlled_manifest.read_bytes()).hexdigest() != manifest_digest:
        controlled_manifest.unlink(missing_ok=True)
        raise RuntimeError("controlled release manifest copy failed SHA-256 re-verification")
    controlled_copy = controlled_directory / INSTALLER_FILENAME
    shutil.copyfile(installer_path, controlled_copy)
    if hashlib.sha256(controlled_copy.read_bytes()).hexdigest() != expected:
        controlled_copy.unlink(missing_ok=True)
        raise RuntimeError("controlled setup copy failed SHA-256 re-verification")
    signature_status = str(manifest["installer"]["signature_status"])
    authenticode_valid = _authenticode_valid(controlled_copy)
    if signature_status == "signed" and not authenticode_valid:
        raise RuntimeError("setup claims to be signed but failed Authenticode verification")
    if not release.get("prerelease") and not authenticode_valid:
        raise RuntimeError("stable setup failed Authenticode verification")
    process: subprocess.Popen[bytes] | None = None
    if launch:
        log_path = install_root / "logs" / f"update-{tag}.log"
        process = subprocess.Popen(
            [
                str(controlled_copy),
                "/CURRENTUSER",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                f"/LOG={log_path}",
            ],
            shell=False,
            close_fds=True,
        )
    return NativeUpdateResult(
        tag=tag,
        version=str(manifest["version"]),
        release_url=str(release["html_url"]),
        installer_path=str(controlled_copy),
        installer_sha256=f"sha256:{expected}",
        signature_status=signature_status,
        launched=launch,
        process_id=process.pid if process else None,
    )


__all__ = ["NativeUpdateResult", "prepare_native_update"]
