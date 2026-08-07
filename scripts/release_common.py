"""Shared strict release metadata, hashing, and version helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "swarm-inference-lab"
MINIMUM_WINDOWS = "10.0.22621"
ARCHITECTURE = "x86_64"
ZERO_SHA256 = "sha256:" + ("0" * 64)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?:rc(?P<rc>\d+))?$")
_TAG = re.compile(r"^v(?P<release>\d+\.\d+\.\d+)(?:-rc\.(?P<rc>\d+))?$")


class ReleaseError(RuntimeError):
    """A fail-closed release input or artifact validation error."""


def read_pyproject_version(path: Path = ROOT / "pyproject.toml") -> str:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    try:
        value = document["project"]["version"]
    except (KeyError, TypeError) as exc:
        raise ReleaseError(f"project.version is missing from {path}") from exc
    if not isinstance(value, str) or _VERSION.fullmatch(value) is None:
        raise ReleaseError(f"unsupported package version {value!r}")
    return value


def pep440_to_git_tag(version: str) -> str:
    match = _VERSION.fullmatch(version)
    if match is None:
        raise ReleaseError(f"unsupported package version {version!r}")
    release = match.group("release")
    rc = match.group("rc")
    return f"v{release}-rc.{rc}" if rc is not None else f"v{release}"


def git_tag_to_pep440(tag: str) -> str:
    match = _TAG.fullmatch(tag)
    if match is None:
        raise ReleaseError(f"unsupported Git release tag {tag!r}")
    release = match.group("release")
    rc = match.group("rc")
    return f"{release}rc{rc}" if rc is not None else release


def release_channel(version: str) -> str:
    if _VERSION.fullmatch(version) is None:
        raise ReleaseError(f"unsupported package version {version!r}")
    return "prerelease" if "rc" in version else "stable"


def run_captured(
    argv: Sequence[str],
    *,
    cwd: Path = ROOT,
    timeout_seconds: float = 120.0,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    if not 0 < timeout_seconds <= 3600:
        raise ValueError("process timeout must be in (0, 3600] seconds")
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReleaseError(f"command timed out after {timeout_seconds:.0f}s: {argv[0]}") from exc
    if completed.returncode != 0:
        diagnostics = (completed.stderr or completed.stdout).strip()[-4000:]
        raise ReleaseError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n{diagnostics}"
        )
    return completed


def git_output(arguments: Sequence[str], *, cwd: Path = ROOT) -> str:
    return run_captured(["git", *arguments], cwd=cwd, timeout_seconds=30).stdout.strip()


def git_commit(*, cwd: Path = ROOT) -> str:
    commit = git_output(["rev-parse", "HEAD"], cwd=cwd).lower()
    if _COMMIT.fullmatch(commit) is None:
        raise ReleaseError(f"Git returned an invalid commit identity: {commit!r}")
    return commit


def git_is_dirty(*, cwd: Path = ROOT) -> bool:
    return bool(git_output(["status", "--porcelain=v1", "--untracked-files=all"], cwd=cwd))


def verify_release_identity(
    *,
    version: str,
    tag: str,
    commit: str,
    cwd: Path = ROOT,
    require_clean: bool = True,
    require_tag: bool = True,
) -> dict[str, Any]:
    expected_tag = pep440_to_git_tag(version)
    if tag != expected_tag or git_tag_to_pep440(tag) != version:
        raise ReleaseError(f"package version {version!r} maps to {expected_tag!r}, not {tag!r}")
    actual_commit = git_commit(cwd=cwd)
    if commit.lower() != actual_commit:
        raise ReleaseError(
            f"requested commit {commit!r} does not match checked-out commit {actual_commit!r}"
        )
    dirty = git_is_dirty(cwd=cwd)
    if require_clean and dirty:
        raise ReleaseError("release builds require a clean source tree")
    tag_commit: str | None = None
    if require_tag:
        try:
            tag_commit = git_output(["rev-list", "-n", "1", tag], cwd=cwd).lower()
        except ReleaseError as exc:
            raise ReleaseError(f"release tag {tag!r} does not exist locally") from exc
        if tag_commit != actual_commit:
            raise ReleaseError(f"release tag {tag!r} points to {tag_commit}, not {actual_commit}")
    return {
        "schema_version": 1,
        "product": PRODUCT,
        "version": version,
        "git_tag": tag,
        "git_commit": actual_commit,
        "tag_commit": tag_commit,
        "source_tree_clean": not dirty,
        "verified_at_utc": utc_now(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return f"sha512:{digest.hexdigest()}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"JSON document must be an object: {path}")
    return value


def load_toolchain(path: Path = ROOT / "installer/windows/toolchain.json") -> dict[str, Any]:
    document = read_json_object(path)
    _require_keys(
        document,
        {"schema_version", "python", "uv", "dotnet", "inno_setup", "llamacpp"},
        "toolchain",
    )
    if document["schema_version"] != 1:
        raise ReleaseError("unsupported toolchain schema version")
    _validate_llamacpp_toolchain(document["llamacpp"])
    return document


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ReleaseError(f"{label} keys mismatch; missing={missing}, unexpected={unexpected}")


def _require_sha256(value: object, label: str, *, allow_zero: bool = False) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReleaseError(f"{label} is not a canonical SHA-256 identity")
    if not allow_zero and value == ZERO_SHA256:
        raise ReleaseError(f"{label} cannot use the embedded-manifest placeholder hash")
    return value


def _safe_basename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise ReleaseError(f"{label} must be one safe basename")
    return value


def _validate_llamacpp_profile(
    value: object,
    label: str,
    *,
    source_archives: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be an object")
    _require_keys(
        value,
        {
            "platform",
            "archives",
            "server_binary",
            "server_sha256",
            "rpc_server_binary",
            "rpc_server_sha256",
            "build_flags",
            "device_support",
        },
        label,
    )
    if value["platform"] != "windows-x64":
        raise ReleaseError(f"{label}.platform must be windows-x64")
    _safe_basename(value["server_binary"], f"{label}.server_binary")
    _safe_basename(value["rpc_server_binary"], f"{label}.rpc_server_binary")
    _require_sha256(value["server_sha256"], f"{label}.server_sha256")
    _require_sha256(value["rpc_server_sha256"], f"{label}.rpc_server_sha256")
    archives = value["archives"]
    if not isinstance(archives, list) or not archives:
        raise ReleaseError(f"{label}.archives must be a non-empty array")
    for index, archive in enumerate(archives):
        archive_label = f"{label}.archives[{index}]"
        if not isinstance(archive, dict):
            raise ReleaseError(f"{archive_label} must be an object")
        if source_archives:
            _require_keys(
                archive,
                {"filename", "url", "sha256", "size_bytes"},
                archive_label,
            )
            url = archive["url"]
            if not isinstance(url, str) or not url.startswith(
                "https://github.com/ggml-org/llama.cpp/releases/download/"
            ):
                raise ReleaseError(f"{archive_label}.url is not an official HTTPS release asset")
            plain = {key: archive[key] for key in ("filename", "sha256", "size_bytes")}
        else:
            plain = archive
        _validate_plain_file(plain, archive_label, require_size=True)
    flags = value["build_flags"]
    if (
        not isinstance(flags, dict)
        or not flags
        or any(
            not isinstance(key, str) or not isinstance(item, bool) for key, item in flags.items()
        )
        or flags.get("GGML_RPC") is not True
    ):
        raise ReleaseError(f"{label}.build_flags must prove GGML_RPC and contain booleans")
    devices = value["device_support"]
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, str) or not item for item in devices)
        or len({item.casefold() for item in devices}) != len(devices)
        or "CPU" not in devices
    ):
        raise ReleaseError(f"{label}.device_support is invalid")
    return value


def _validate_llamacpp_identity(value: object, label: str, *, source_archives: bool) -> None:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be an object")
    _require_keys(
        value,
        {"repository", "release_tag", "runtime_revision", "profiles"},
        label,
    )
    if value["repository"] != "ggml-org/llama.cpp":
        raise ReleaseError(f"{label}.repository is invalid")
    if not isinstance(value["release_tag"], str) or not value["release_tag"]:
        raise ReleaseError(f"{label}.release_tag is invalid")
    if (
        not isinstance(value["runtime_revision"], str)
        or _COMMIT.fullmatch(value["runtime_revision"]) is None
    ):
        raise ReleaseError(f"{label}.runtime_revision must be an immutable commit")
    profiles = value["profiles"]
    if not isinstance(profiles, dict):
        raise ReleaseError(f"{label}.profiles must be an object")
    _require_keys(
        profiles,
        {"windows-x64-cpu", "windows-x64-cuda"},
        f"{label}.profiles",
    )
    cpu = _validate_llamacpp_profile(
        profiles["windows-x64-cpu"],
        f"{label}.profiles.windows-x64-cpu",
        source_archives=source_archives,
    )
    cuda = _validate_llamacpp_profile(
        profiles["windows-x64-cuda"],
        f"{label}.profiles.windows-x64-cuda",
        source_archives=source_archives,
    )
    if cpu["build_flags"].get("GGML_CUDA") is not False or "CUDA" in cpu["device_support"]:
        raise ReleaseError(f"{label} CPU profile may not claim CUDA")
    if cuda["build_flags"].get("GGML_CUDA") is not True or "CUDA" not in cuda["device_support"]:
        raise ReleaseError(f"{label} CUDA profile must prove CUDA support")


def _validate_llamacpp_toolchain(value: object) -> None:
    _validate_llamacpp_identity(value, "llamacpp toolchain", source_archives=True)


def _validate_plain_file(
    value: object,
    label: str,
    *,
    require_size: bool,
    allow_zero_hash: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be an object")
    keys = {"filename", "sha256", "size_bytes"} if require_size else {"filename", "sha256"}
    if not require_size and "size_bytes" in value:
        keys.add("size_bytes")
    _require_keys(value, keys, label)
    filename = value["filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ReleaseError(f"{label}.filename must be one safe basename")
    _require_sha256(value["sha256"], f"{label}.sha256", allow_zero=allow_zero_hash)
    if "size_bytes" in value and (
        not isinstance(value["size_bytes"], int) or value["size_bytes"] < 0
    ):
        raise ReleaseError(f"{label}.size_bytes must be a non-negative integer")
    return value


def _validate_signed_file(
    value: object,
    label: str,
    *,
    allow_zero_hash: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReleaseError(f"{label} must be an object")
    required = {"filename", "sha256", "size_bytes", "signature_status", "signature_verification"}
    optional = {"publisher_subject"}
    actual = set(value)
    if not required.issubset(actual) or actual - required - optional:
        raise ReleaseError(
            f"{label} keys mismatch; missing={sorted(required - actual)}, "
            f"unexpected={sorted(actual - required - optional)}"
        )
    _validate_plain_file(
        {key: value[key] for key in ("filename", "sha256", "size_bytes")},
        label,
        require_size=True,
        allow_zero_hash=allow_zero_hash,
    )
    status = value["signature_status"]
    verification = value["signature_verification"]
    if status == "signed":
        if verification != "valid" or not isinstance(value.get("publisher_subject"), str):
            raise ReleaseError(f"{label} signed status requires a verified publisher subject")
    elif status == "unsigned-prerelease":
        if verification != "not-signed" or "publisher_subject" in value:
            raise ReleaseError(f"{label} unsigned status must be explicit and publisher-free")
    else:
        raise ReleaseError(f"{label}.signature_status is invalid")
    return value


def validate_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "manifest_scope",
        "product",
        "version",
        "git_tag",
        "git_commit",
        "channel",
        "built_at_utc",
        "minimum_windows",
        "architecture",
        "python",
        "uv",
        "wheel",
        "runtime_profiles",
        "engine_runtimes",
        "bootstrapper",
        "installer",
        "payload",
    }
    _require_keys(document, expected, "release manifest")
    if document["schema_version"] != 1 or document["product"] != PRODUCT:
        raise ReleaseError("release manifest product or schema identity is invalid")
    scope = document["manifest_scope"]
    if scope not in {"embedded-payload", "release"}:
        raise ReleaseError("release manifest scope is invalid")
    version = document["version"]
    tag = document["git_tag"]
    if not isinstance(version, str) or not isinstance(tag, str):
        raise ReleaseError("release manifest version and tag must be strings")
    if pep440_to_git_tag(version) != tag or git_tag_to_pep440(tag) != version:
        raise ReleaseError("release manifest version and tag do not map to each other")
    if (
        not isinstance(document["git_commit"], str)
        or _COMMIT.fullmatch(document["git_commit"]) is None
    ):
        raise ReleaseError("release manifest Git commit must be a 40-character lowercase SHA")
    channel = release_channel(version)
    if document["channel"] != channel:
        raise ReleaseError(f"version {version} requires release channel {channel}")
    if document["minimum_windows"] != MINIMUM_WINDOWS or document["architecture"] != ARCHITECTURE:
        raise ReleaseError("release manifest platform identity is invalid")
    built_at = document["built_at_utc"]
    if not isinstance(built_at, str) or not built_at.endswith("Z"):
        raise ReleaseError("release manifest timestamp must be UTC")
    try:
        datetime.fromisoformat(built_at.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ReleaseError("release manifest timestamp is invalid") from exc
    python = document["python"]
    if not isinstance(python, dict):
        raise ReleaseError("release manifest Python metadata must be an object")
    _require_keys(python, {"version"}, "python")
    if (
        not isinstance(python["version"], str)
        or re.fullmatch(r"3\.11\.\d+", python["version"]) is None
    ):
        raise ReleaseError("release manifest must pin an exact Python 3.11 version")
    uv = document["uv"]
    if not isinstance(uv, dict):
        raise ReleaseError("release manifest uv metadata must be an object")
    _require_keys(uv, {"version", "filename", "sha256", "size_bytes"}, "uv")
    if uv["filename"] != "uv.exe" or not isinstance(uv["version"], str):
        raise ReleaseError("release manifest uv identity is invalid")
    _validate_plain_file(
        {key: uv[key] for key in ("filename", "sha256", "size_bytes")},
        "uv",
        require_size=True,
    )
    _validate_plain_file(document["wheel"], "wheel", require_size=True)
    profiles = document["runtime_profiles"]
    if not isinstance(profiles, dict):
        raise ReleaseError("runtime_profiles must be an object")
    _require_keys(profiles, {"windows-x64-cpu", "windows-x64-cuda"}, "runtime_profiles")
    _validate_plain_file(profiles["windows-x64-cpu"], "CPU profile", require_size=True)
    _validate_plain_file(profiles["windows-x64-cuda"], "CUDA profile", require_size=True)
    engine_runtimes = document["engine_runtimes"]
    if not isinstance(engine_runtimes, dict):
        raise ReleaseError("engine_runtimes must be an object")
    _require_keys(engine_runtimes, {"llamacpp"}, "engine_runtimes")
    _validate_llamacpp_identity(
        engine_runtimes["llamacpp"],
        "engine_runtimes.llamacpp",
        source_archives=False,
    )
    _validate_signed_file(document["bootstrapper"], "bootstrapper", allow_zero_hash=False)
    _validate_signed_file(
        document["installer"], "installer", allow_zero_hash=scope == "embedded-payload"
    )
    payload = document["payload"]
    if not isinstance(payload, list):
        raise ReleaseError("payload must be an array")
    for index, item in enumerate(payload):
        _validate_plain_file(item, f"payload[{index}]", require_size=True)
    if channel == "stable":
        for label in ("bootstrapper", "installer"):
            signed = document[label]
            if not isinstance(signed, dict) or signed.get("signature_status") != "signed":
                raise ReleaseError("stable releases require signed bootstrapper and installer")
    elif any(
        isinstance(document[label], dict)
        and document[label].get("signature_status") == "unsigned-prerelease"
        and document[label].get("signature_verification") != "not-signed"
        for label in ("bootstrapper", "installer")
    ):
        raise ReleaseError("unsigned prerelease status must be explicit")
    filenames = [entry["filename"] for entry in manifest_file_entries(document)]
    folded = [name.casefold() for name in filenames]
    if len(folded) != len(set(folded)):
        raise ReleaseError("release manifest contains duplicate filenames")
    return dict(document)


def manifest_file_entries(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    profiles = document["runtime_profiles"]
    engine_runtimes = document["engine_runtimes"]
    payload = document["payload"]
    if (
        not isinstance(profiles, Mapping)
        or not isinstance(engine_runtimes, Mapping)
        or not isinstance(payload, list)
    ):
        raise ReleaseError("release manifest file collections are malformed")
    entries: list[dict[str, Any]] = []
    for key in ("uv", "wheel", "bootstrapper", "installer"):
        value = document[key]
        if not isinstance(value, dict):
            raise ReleaseError(f"release manifest {key} entry is malformed")
        entries.append(value)
    for key in ("windows-x64-cpu", "windows-x64-cuda"):
        value = profiles[key]
        if not isinstance(value, dict):
            raise ReleaseError(f"release manifest {key} entry is malformed")
        entries.append(value)
    llamacpp = engine_runtimes.get("llamacpp")
    if not isinstance(llamacpp, Mapping) or not isinstance(llamacpp.get("profiles"), Mapping):
        raise ReleaseError("release manifest llama.cpp engine runtimes are malformed")
    engine_profiles = llamacpp["profiles"]
    for key in ("windows-x64-cpu", "windows-x64-cuda"):
        profile = engine_profiles.get(key)
        if not isinstance(profile, Mapping) or not isinstance(profile.get("archives"), list):
            raise ReleaseError(f"release manifest llama.cpp {key} profile is malformed")
        for value in profile["archives"]:
            if not isinstance(value, dict):
                raise ReleaseError(f"release manifest llama.cpp {key} archive is malformed")
            entries.append(value)
    for value in payload:
        if not isinstance(value, dict):
            raise ReleaseError("release manifest payload entry is malformed")
        entries.append(value)
    return entries


def verify_manifest_files(
    document: Mapping[str, Any],
    directory: Path,
    *,
    allow_manifest: bool = True,
) -> None:
    validated = validate_manifest(document)
    scope = validated["manifest_scope"]
    expected: set[str] = set()
    for entry in manifest_file_entries(validated):
        filename = entry["filename"]
        if scope == "embedded-payload" and filename == "SwarmInferenceSetup-x64.exe":
            continue
        expected.add(filename)
        path = directory / filename
        if not path.is_file():
            raise ReleaseError(f"release payload file is missing: {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ReleaseError(
                f"SHA-256 mismatch for {filename}: expected {entry['sha256']}, got {actual_hash}"
            )
        if "size_bytes" in entry and path.stat().st_size != entry["size_bytes"]:
            raise ReleaseError(f"size mismatch for {filename}")
    allowed_metadata = {"release-manifest.json"} if allow_manifest else set()
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    unexpected = sorted(actual - expected - allowed_metadata)
    if unexpected:
        raise ReleaseError(f"release payload contains unexpected files: {unexpected}")


def file_entry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseError(f"release input file is missing: {path}")
    return {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_sha256sums(path: Path, files: Iterable[Path]) -> None:
    selected = sorted(files, key=lambda item: item.name.casefold())
    names = [item.name.casefold() for item in selected]
    if len(names) != len(set(names)):
        raise ReleaseError("cannot generate checksums for duplicate filenames")
    lines = [f"{sha256_file(item).removeprefix('sha256:')} *{item.name}" for item in selected]
    atomic_write_text(path, "\n".join(lines) + "\n")


def parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw:
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([^/\\]+)", raw)
        if match is None:
            raise ReleaseError(f"invalid SHA256SUMS line {line_number}")
        filename = match.group(2)
        if filename.casefold() in {item.casefold() for item in result}:
            raise ReleaseError(f"duplicate SHA256SUMS filename: {filename}")
        result[filename] = f"sha256:{match.group(1)}"
    if not result:
        raise ReleaseError("SHA256SUMS is empty")
    return result
