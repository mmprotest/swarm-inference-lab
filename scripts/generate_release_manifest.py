"""Generate strict release manifests, checksums, and a runtime SBOM."""

from __future__ import annotations

import argparse
import json
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

from release_common import (
    ARCHITECTURE,
    MINIMUM_WINDOWS,
    PRODUCT,
    ZERO_SHA256,
    ReleaseError,
    atomic_write_json,
    file_entry,
    git_commit,
    load_toolchain,
    pep440_to_git_tag,
    read_pyproject_version,
    release_channel,
    utc_now,
    validate_manifest,
    write_sha256sums,
)

PROFILE_FILENAMES = {
    "windows-x64-cpu": "windows-x64-cpu.requirements.lock",
    "windows-x64-cuda": "windows-x64-cuda.requirements.lock",
}
PAYLOAD_FILENAMES = ("LICENSE", "swarm.ico", "wizard-small.bmp", "wizard-large.bmp")


def _engine_runtime_manifest(
    payload_directory: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    profiles = metadata.get("profiles")
    if not isinstance(profiles, dict):
        raise ReleaseError("llama.cpp toolchain profiles are malformed")
    manifest_profiles: dict[str, Any] = {}
    for profile_id in ("windows-x64-cpu", "windows-x64-cuda"):
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict) or not isinstance(profile.get("archives"), list):
            raise ReleaseError(f"llama.cpp toolchain profile {profile_id} is malformed")
        archives: list[dict[str, Any]] = []
        for expected in profile["archives"]:
            if not isinstance(expected, dict):
                raise ReleaseError(f"llama.cpp {profile_id} archive metadata is malformed")
            entry = file_entry(payload_directory / str(expected["filename"]))
            if (
                entry["sha256"] != expected["sha256"]
                or entry["size_bytes"] != expected["size_bytes"]
            ):
                raise ReleaseError(
                    f"llama.cpp source archive {entry['filename']} does not match its immutable pin"
                )
            archives.append(entry)
        manifest_profiles[profile_id] = {
            key: profile[key]
            for key in (
                "platform",
                "server_binary",
                "server_sha256",
                "rpc_server_binary",
                "rpc_server_sha256",
                "build_flags",
                "device_support",
            )
        }
        manifest_profiles[profile_id]["archives"] = archives
    return {
        "repository": metadata["repository"],
        "release_tag": metadata["release_tag"],
        "runtime_revision": metadata["runtime_revision"],
        "profiles": manifest_profiles,
    }


def _wheel_version(path: Path) -> str:
    if not zipfile.is_zipfile(path):
        raise ReleaseError(f"application wheel is not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ReleaseError("application wheel must contain exactly one METADATA document")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    matches = [
        line.removeprefix("Version: ")
        for line in metadata.splitlines()
        if line.startswith("Version: ")
    ]
    if len(matches) != 1:
        raise ReleaseError("application wheel metadata does not contain one version")
    return matches[0]


def _signed_entry(
    path: Path | None,
    *,
    filename: str,
    status: str,
    publisher_subject: str | None,
) -> dict[str, Any]:
    if path is None:
        entry: dict[str, Any] = {
            "filename": filename,
            "sha256": ZERO_SHA256,
            "size_bytes": 0,
        }
    else:
        entry = file_entry(path)
        if entry["filename"] != filename:
            raise ReleaseError(f"expected {filename}, got {entry['filename']}")
    if status not in {"signed", "unsigned-prerelease"}:
        raise ReleaseError(f"invalid signature status {status!r}")
    entry["signature_status"] = status
    entry["signature_verification"] = "valid" if status == "signed" else "not-signed"
    if status == "signed":
        if not publisher_subject:
            raise ReleaseError("signed artifacts require a verified publisher subject")
        entry["publisher_subject"] = publisher_subject
    elif publisher_subject:
        raise ReleaseError("an unsigned artifact cannot claim a publisher subject")
    return entry


def build_manifest(
    *,
    payload_directory: Path,
    installer: Path | None,
    version: str,
    tag: str,
    commit: str,
    built_at_utc: str,
    bootstrapper_signature_status: str,
    installer_signature_status: str,
    publisher_subject: str | None,
) -> dict[str, Any]:
    if pep440_to_git_tag(version) != tag:
        raise ReleaseError(f"package version {version} does not map to tag {tag}")
    toolchain = load_toolchain()
    python = toolchain["python"]
    uv_metadata = toolchain["uv"]
    llamacpp_metadata = toolchain["llamacpp"]
    if (
        not isinstance(python, dict)
        or not isinstance(uv_metadata, dict)
        or not isinstance(llamacpp_metadata, dict)
    ):
        raise ReleaseError("toolchain Python, uv, or llama.cpp metadata is malformed")
    wheel_name = f"swarm_inference_lab-{version}-py3-none-any.whl"
    wheel_path = payload_directory / wheel_name
    if _wheel_version(wheel_path) != version:
        raise ReleaseError("wheel metadata version does not match the release version")
    uv_path = payload_directory / "uv.exe"
    if file_entry(uv_path)["sha256"] != uv_metadata["executable_sha256"]:
        raise ReleaseError("payload uv.exe does not match the repository-controlled pin")
    bootstrapper_path = payload_directory / "SwarmBootstrap.exe"
    profile_entries = {
        name: file_entry(payload_directory / filename)
        for name, filename in PROFILE_FILENAMES.items()
    }
    payload_entries = [file_entry(payload_directory / filename) for filename in PAYLOAD_FILENAMES]
    scope = "release" if installer is not None else "embedded-payload"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_scope": scope,
        "product": PRODUCT,
        "version": version,
        "git_tag": tag,
        "git_commit": commit.lower(),
        "channel": release_channel(version),
        "built_at_utc": built_at_utc,
        "minimum_windows": MINIMUM_WINDOWS,
        "architecture": ARCHITECTURE,
        "python": {"version": python["version"]},
        "uv": {
            "version": uv_metadata["version"],
            **file_entry(uv_path),
        },
        "wheel": file_entry(wheel_path),
        "runtime_profiles": profile_entries,
        "engine_runtimes": {
            "llamacpp": _engine_runtime_manifest(payload_directory, llamacpp_metadata),
        },
        "bootstrapper": _signed_entry(
            bootstrapper_path,
            filename="SwarmBootstrap.exe",
            status=bootstrapper_signature_status,
            publisher_subject=publisher_subject,
        ),
        "installer": _signed_entry(
            installer,
            filename="SwarmInferenceSetup-x64.exe",
            status=installer_signature_status,
            publisher_subject=publisher_subject,
        ),
        "payload": payload_entries,
    }
    return validate_manifest(manifest)


def generate_sbom(
    *,
    version: str,
    tag: str,
    commit: str,
    built_at_utc: str,
    profile_paths: list[Path],
) -> dict[str, Any]:
    requirements: set[tuple[str, str]] = set()
    for path in profile_paths:
        text = path.read_text(encoding="utf-8")
        requirements.update(
            (match.group(1).lower().replace("_", "-"), match.group(2))
            for match in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==([^ ;\\]+)", text)
        )
    components = [
        {
            "type": "library",
            "bom-ref": f"pkg:pypi/{name}@{package_version}",
            "name": name,
            "version": package_version,
            "purl": f"pkg:pypi/{name}@{package_version}",
        }
        for name, package_version in sorted(requirements)
    ]
    llamacpp = load_toolchain()["llamacpp"]
    if not isinstance(llamacpp, dict):
        raise ReleaseError("llama.cpp toolchain metadata is malformed")
    runtime_revision = str(llamacpp["runtime_revision"])
    components.append(
        {
            "type": "application",
            "bom-ref": f"pkg:github/ggml-org/llama.cpp@{runtime_revision}",
            "name": "llama.cpp",
            "version": str(llamacpp["release_tag"]),
            "purl": f"pkg:github/ggml-org/llama.cpp@{runtime_revision}",
            "properties": [
                {"name": "swarm:runtime_revision", "value": runtime_revision},
                {"name": "swarm:release_tag", "value": str(llamacpp["release_tag"])},
            ],
        }
    )
    namespace = uuid.UUID("94b6963c-df15-45ec-b5f6-89b0f68f1e40")
    serial = uuid.uuid5(namespace, f"{PRODUCT}:{version}:{commit}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": built_at_utc,
            "component": {
                "type": "application",
                "bom-ref": f"pkg:pypi/{PRODUCT}@{version}",
                "name": PRODUCT,
                "version": version,
                "purl": f"pkg:pypi/{PRODUCT}@{version}",
                "properties": [
                    {"name": "swarm:git_tag", "value": tag},
                    {"name": "swarm:git_commit", "value": commit},
                ],
            },
        },
        "components": components,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--installer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version")
    parser.add_argument("--tag")
    parser.add_argument("--commit")
    parser.add_argument("--built-at-utc")
    parser.add_argument(
        "--bootstrapper-signature-status",
        choices=("signed", "unsigned-prerelease"),
        default="unsigned-prerelease",
    )
    parser.add_argument(
        "--installer-signature-status",
        choices=("signed", "unsigned-prerelease"),
        default="unsigned-prerelease",
    )
    parser.add_argument("--publisher-subject")
    parser.add_argument("--sbom-output", type=Path)
    parser.add_argument("--checksums-output", type=Path)
    parser.add_argument("--acceptance-zip", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    version = arguments.version or read_pyproject_version()
    tag = arguments.tag or pep440_to_git_tag(version)
    commit = (arguments.commit or git_commit()).lower()
    built_at = arguments.built_at_utc or utc_now()
    payload_directory = arguments.payload_dir.resolve()
    installer = arguments.installer.resolve() if arguments.installer is not None else None
    try:
        manifest = build_manifest(
            payload_directory=payload_directory,
            installer=installer,
            version=version,
            tag=tag,
            commit=commit,
            built_at_utc=built_at,
            bootstrapper_signature_status=arguments.bootstrapper_signature_status,
            installer_signature_status=arguments.installer_signature_status,
            publisher_subject=arguments.publisher_subject,
        )
        atomic_write_json(arguments.output, manifest)
        evidence_files = [arguments.output.resolve()]
        if arguments.sbom_output is not None:
            sbom = generate_sbom(
                version=version,
                tag=tag,
                commit=commit,
                built_at_utc=built_at,
                profile_paths=[payload_directory / name for name in PROFILE_FILENAMES.values()],
            )
            atomic_write_json(arguments.sbom_output, sbom)
            evidence_files.append(arguments.sbom_output.resolve())
        if installer is not None:
            evidence_files.extend([installer, payload_directory / manifest["wheel"]["filename"]])
        if arguments.acceptance_zip is not None:
            if not arguments.acceptance_zip.is_file():
                raise ReleaseError("productization acceptance ZIP is missing")
            evidence_files.append(arguments.acceptance_zip.resolve())
        if arguments.checksums_output is not None:
            write_sha256sums(arguments.checksums_output, evidence_files)
    except (OSError, ReleaseError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(arguments.output.resolve()),
                "scope": manifest["manifest_scope"],
                "version": version,
                "git_tag": tag,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
