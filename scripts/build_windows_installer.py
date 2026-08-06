"""Build the complete pinned native Windows release candidate payload."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

from generate_release_manifest import build_manifest, generate_sbom
from generate_runtime_profiles import generate_profiles
from release_common import (
    ROOT,
    ReleaseError,
    atomic_write_json,
    git_commit,
    git_is_dirty,
    load_toolchain,
    pep440_to_git_tag,
    read_pyproject_version,
    run_captured,
    sha256_file,
    sha512_file,
    utc_now,
    verify_manifest_files,
    verify_release_identity,
    write_sha256sums,
)
from sign_windows_artifact import authenticode_info, sign_file, signing_environment_present

MAX_TOOL_DOWNLOAD_BYTES = 768 * 1024 * 1024
FINAL_FILENAMES = (
    "SwarmInferenceSetup-x64.exe",
    "SwarmBootstrap.exe",
    "uv.exe",
    "windows-x64-cpu.requirements.lock",
    "windows-x64-cuda.requirements.lock",
    "LICENSE",
    "swarm.ico",
    "wizard-small.bmp",
    "wizard-large.bmp",
    "release-manifest.json",
    "SHA256SUMS",
    "swarm-inference-sbom.json",
    "productization-acceptance.zip",
)


def _download(url: str, destination: Path, expected: str) -> None:
    if not url.startswith("https://"):
        raise ReleaseError(f"tool download must use HTTPS: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    algorithm, expected_digest = expected.split(":", 1)
    if algorithm not in {"sha256", "sha512"}:
        raise ReleaseError("tool download has an unsupported digest algorithm")
    digest = hashlib.new(algorithm)
    size = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "swarm-release-builder/1"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("xb") as output,
        ):
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > MAX_TOOL_DOWNLOAD_BYTES:
                    raise ReleaseError("tool download exceeded the bounded size limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if not hmac.compare_digest(digest.hexdigest(), expected_digest):
            raise ReleaseError(f"download hash mismatch for {destination.name}")
        os.replace(temporary, destination)
    except (TimeoutError, urllib.error.URLError) as exc:
        raise ReleaseError(f"bounded tool download failed for {destination.name}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _verify(path: Path, expected: str) -> None:
    algorithm = expected.split(":", 1)[0]
    actual = sha256_file(path) if algorithm == "sha256" else sha512_file(path)
    if actual != expected:
        raise ReleaseError(
            f"pinned hash mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ReleaseError(f"ZIP archive has an unsafe member: {member.filename}")
        bundle.extractall(destination)


def _verified_uv(toolchain: dict[str, Any], explicit: Path | None) -> Path:
    metadata = toolchain["uv"]
    if not isinstance(metadata, dict):
        raise ReleaseError("uv toolchain metadata is malformed")
    if explicit is not None:
        executable = explicit.resolve()
    else:
        root = ROOT / "build" / "toolchain" / f"uv-{metadata['version']}"
        executable = root / "uv.exe"
        if not executable.is_file():
            archive = ROOT / "build" / "toolchain-downloads" / metadata["archive_filename"]
            if not archive.is_file():
                _download(metadata["url"], archive, metadata["archive_sha256"])
            _verify(archive, metadata["archive_sha256"])
            _safe_extract(archive, root)
    _verify(executable, metadata["executable_sha256"])
    version = run_captured([str(executable), "--version"], timeout_seconds=30).stdout.strip()
    if version != f"uv {metadata['version']} (0c2f6f0b1 2026-07-28)" and not version.startswith(
        f"uv {metadata['version']} "
    ):
        raise ReleaseError(f"unexpected uv version output: {version}")
    return executable


def _verified_dotnet(toolchain: dict[str, Any], explicit: Path | None) -> Path:
    metadata = toolchain["dotnet"]
    if not isinstance(metadata, dict):
        raise ReleaseError(".NET toolchain metadata is malformed")
    if explicit is not None:
        executable = explicit.resolve()
    else:
        root = ROOT / "build" / "toolchain" / f"dotnet-{metadata['version']}"
        executable = root / "dotnet.exe"
        if not executable.is_file():
            archive = ROOT / "build" / "toolchain-downloads" / metadata["archive_filename"]
            if not archive.is_file():
                _download(metadata["url"], archive, metadata["archive_sha512"])
            _verify(archive, metadata["archive_sha512"])
            _safe_extract(archive, root)
    _verify(executable, metadata["executable_sha256"])
    version = run_captured([str(executable), "--version"], timeout_seconds=30).stdout.strip()
    if version != metadata["version"]:
        raise ReleaseError(f"unexpected .NET SDK version: {version}")
    return executable


def _verify_pinned_publisher_identity(
    information: dict[str, Any],
    metadata: dict[str, Any],
    *,
    label: str,
) -> None:
    status = str(information.get("status") or "")
    subject = str(information.get("subject") or "")
    thumbprint = str(information.get("thumbprint") or "").replace(" ", "").upper()
    expected_thumbprint = str(metadata["publisher_thumbprint"]).replace(" ", "").upper()
    # Get-AuthenticodeSignature's chain status depends on the runner's root and
    # revocation caches. The executable hash is verified immediately before this
    # check, so pin the actual signer certificate identity and reject statuses
    # that indicate missing or damaged signature bytes.
    accepted_statuses = {"Valid", "UnknownError", "NotTrusted", "CertificateOnly"}
    if status not in accepted_statuses or not subject or thumbprint != expected_thumbprint:
        raise ReleaseError(
            f"{label} publisher verification failed "
            f"(status={status!r}, subject={subject!r}, thumbprint={thumbprint!r})"
        )


def _verified_iscc(toolchain: dict[str, Any], explicit: Path | None) -> Path:
    metadata = toolchain["inno_setup"]
    if not isinstance(metadata, dict):
        raise ReleaseError("Inno Setup toolchain metadata is malformed")
    if explicit is not None:
        compiler = explicit.resolve()
    else:
        root = ROOT / "build" / "toolchain" / f"inno-{metadata['version']}"
        compiler = root / "ISCC.exe"
        if not compiler.is_file():
            installer = ROOT / "build" / "toolchain-downloads" / metadata["filename"]
            if not installer.is_file():
                _download(metadata["url"], installer, metadata["sha256"])
            _verify(installer, metadata["sha256"])
            installer_signature = authenticode_info(installer)
            _verify_pinned_publisher_identity(
                installer_signature,
                metadata,
                label="Inno Setup installer",
            )
            run_captured(
                [
                    str(installer),
                    "/VERYSILENT",
                    "/SUPPRESSMSGBOXES",
                    "/NORESTART",
                    "/CURRENTUSER",
                    f"/DIR={root}",
                ],
                timeout_seconds=180,
            )
    _verify(compiler, metadata["compiler_sha256"])
    compiler_signature = authenticode_info(compiler)
    _verify_pinned_publisher_identity(
        compiler_signature,
        metadata,
        label="Inno Setup compiler",
    )
    return compiler


def _build_wheel(uv: Path, output: Path, version: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "UV_CACHE_DIR": str((ROOT / ".uv-cache").resolve()),
            "UV_NO_SYSTEM_CONFIG": "1",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    run_captured(
        [str(uv), "build", "--wheel", "--out-dir", str(output), str(ROOT)],
        timeout_seconds=300,
        environment=environment,
    )
    expected = output / f"swarm_inference_lab-{version}-py3-none-any.whl"
    if not expected.is_file() or len(list(output.glob("*.whl"))) != 1:
        raise ReleaseError("wheel build did not produce exactly the expected release wheel")
    return expected


def _rewrite_fixture_wheel(
    source: Path,
    destination: Path,
    *,
    source_version: str,
    fixture_version: str,
    doctor_failure: bool,
) -> Path:
    """Create a deterministic version/doctor fixture without editing the checkout."""

    old_dist_info = f"swarm_inference_lab-{source_version}.dist-info/"
    new_dist_info = f"swarm_inference_lab-{fixture_version}.dist-info/"
    record_name = new_dist_info + "RECORD"
    files: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            name = info.filename.replace(old_dist_info, new_dist_info, 1)
            if name.endswith(".dist-info/RECORD"):
                continue
            content = archive.read(info)
            if name.endswith(".dist-info/METADATA"):
                text = content.decode("utf-8")
                marker = f"Version: {source_version}\n"
                if text.count(marker) != 1:
                    raise ReleaseError("fixture wheel METADATA version marker is not unique")
                content = text.replace(marker, f"Version: {fixture_version}\n").encode()
            if doctor_failure and name == "swarm_inference/commands/node.py":
                text = content.decode("utf-8")
                marker = (
                    '    """Probe real device operation, platform support, state, and service '
                    'tooling."""\n\n    try:\n'
                )
                replacement = (
                    '    """Probe real device operation, platform support, state, and service '
                    'tooling."""\n\n    try:\n'
                    '        raise RuntimeError("intentional installer doctor failure fixture")\n'
                )
                if text.count(marker) != 1:
                    raise ReleaseError("doctor fixture injection marker is not unique")
                content = text.replace(marker, replacement).encode()
            files.append((name, content))
    if doctor_failure and not any(name == "swarm_inference/commands/node.py" for name, _ in files):
        raise ReleaseError("fixture wheel lacks the node doctor command")
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, content in sorted(files):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        writer.writerow((name, f"sha256={digest}", len(content)))
    writer.writerow((record_name, "", ""))
    files.append((record_name, record_buffer.getvalue().encode()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as out:
        for name, content in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            out.writestr(info, content)
    return destination


def _dotnet_environment() -> dict[str, str]:
    home = ROOT / ".tmp" / "dotnet-home"
    appdata = ROOT / ".tmp" / "dotnet-appdata"
    packages = ROOT / ".tmp" / "nuget-packages"
    for directory in (home, appdata, packages):
        directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "DOTNET_CLI_HOME": str(home),
            "APPDATA": str(appdata),
            "NUGET_PACKAGES": str(packages),
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
        }
    )
    return environment


def _dotnet_version_properties(version: str) -> tuple[str, str]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?", version)
    if match is None:
        raise ReleaseError(f"unsupported bootstrapper product version: {version}")
    major, minor, patch, release_candidate = match.groups()
    informational = (
        f"{major}.{minor}.{patch}-rc.{release_candidate}"
        if release_candidate is not None
        else f"{major}.{minor}.{patch}"
    )
    file_version = f"{major}.{minor}.{patch}.{release_candidate or '0'}"
    return informational, file_version


def _build_bootstrapper(dotnet: Path, output: Path, version: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    informational_version, file_version = _dotnet_version_properties(version)
    run_captured(
        [
            str(dotnet),
            "publish",
            str(ROOT / "installer/windows/SwarmBootstrap/SwarmBootstrap.csproj"),
            "-c",
            "Release",
            "-r",
            "win-x64",
            "--self-contained",
            "true",
            "-p:PublishSingleFile=true",
            f"-p:Version={informational_version}",
            f"-p:AssemblyVersion={file_version}",
            f"-p:FileVersion={file_version}",
            f"-p:InformationalVersion={version}",
            "-o",
            str(output),
        ],
        timeout_seconds=600,
        environment=_dotnet_environment(),
    )
    executable = output / "SwarmBootstrap.exe"
    extras = [path for path in output.iterdir() if path.is_file() and path.name != executable.name]
    if not executable.is_file() or extras:
        raise ReleaseError(f"single-file bootstrapper output is invalid; extras={extras}")
    return executable


def _copy_payload_inputs(payload: Path, wheel: Path, uv: Path, bootstrapper: Path) -> None:
    sources = {
        wheel: payload / wheel.name,
        uv: payload / "uv.exe",
        bootstrapper: payload / "SwarmBootstrap.exe",
        ROOT / "LICENSE": payload / "LICENSE",
        ROOT / "installer/windows/assets/swarm.ico": payload / "swarm.ico",
        ROOT / "installer/windows/assets/wizard-small.bmp": payload / "wizard-small.bmp",
        ROOT / "installer/windows/assets/wizard-large.bmp": payload / "wizard-large.bmp",
    }
    for source, destination in sources.items():
        if not source.is_file():
            raise ReleaseError(f"required payload input is missing: {source}")
        shutil.copyfile(source, destination)


def _acceptance_zip(path: Path, supplied: Path | None, build_identity: dict[str, Any]) -> None:
    if supplied is not None:
        if not supplied.is_file() or not zipfile.is_zipfile(supplied):
            raise ReleaseError("supplied productization acceptance evidence is not a ZIP")
        shutil.copyfile(supplied, path)
        return
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "build-only-evidence.json",
            json.dumps({"status": "BUILD_ONLY_NOT_RELEASE_ACCEPTANCE", **build_identity}, indent=2)
            + "\n",
        )


def _clean_final_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    known = set(FINAL_FILENAMES)
    known.update(path.name for path in directory.glob("swarm_inference_lab-*.whl"))
    for name in known:
        (directory / name).unlink(missing_ok=True)


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    if os.name != "nt":
        raise ReleaseError("the native installer must be built on Windows x86-64")
    source_version = read_pyproject_version()
    version = arguments.fixture_version or source_version
    tag = arguments.tag or pep440_to_git_tag(version)
    if tag != pep440_to_git_tag(version):
        raise ReleaseError(f"version {version} maps to {pep440_to_git_tag(version)}, not {tag}")
    commit = git_commit()
    dirty = git_is_dirty()
    if arguments.release and dirty:
        raise ReleaseError("release installer builds reject a dirty source tree")
    if arguments.release and (arguments.fixture_version or arguments.fixture_doctor_failure):
        raise ReleaseError("release builds cannot use acceptance fixture mutation")
    if arguments.fixture_doctor_failure and arguments.fixture_version is None:
        raise ReleaseError("a doctor-failure fixture requires --fixture-version")
    if arguments.release:
        verify_release_identity(
            version=version,
            tag=tag,
            commit=commit,
            require_clean=True,
            require_tag=True,
        )
    toolchain = load_toolchain()
    uv = _verified_uv(toolchain, arguments.uv)
    dotnet = _verified_dotnet(toolchain, arguments.dotnet)
    iscc = _verified_iscc(toolchain, arguments.iscc)
    build_root = ROOT / "build" / "windows-installer" / f"{version}-{uuid.uuid4().hex}"
    payload = build_root / "payload"
    payload.mkdir(parents=True)
    source_wheel = _build_wheel(uv, build_root / "wheel", source_version)
    if version != source_version or arguments.fixture_doctor_failure:
        wheel = _rewrite_fixture_wheel(
            source_wheel,
            build_root / "fixture-wheel" / f"swarm_inference_lab-{version}-py3-none-any.whl",
            source_version=source_version,
            fixture_version=version,
            doctor_failure=arguments.fixture_doctor_failure,
        )
    else:
        wheel = source_wheel
    bootstrapper = _build_bootstrapper(dotnet, build_root / "bootstrapper", version)
    signing = signing_environment_present()
    if not signing and "rc" not in version:
        raise ReleaseError("stable release builds require Authenticode signing secrets")
    publisher: str | None = None
    bootstrapper_status = "unsigned-prerelease"
    if signing:
        signature = sign_file(bootstrapper, signtool=arguments.signtool)
        publisher = str(signature["subject"])
        bootstrapper_status = "signed"
    _copy_payload_inputs(payload, wheel, uv, bootstrapper)
    generate_profiles(uv=uv, output_directory=payload)
    built_at = utc_now()
    embedded = build_manifest(
        payload_directory=payload,
        installer=None,
        version=version,
        tag=tag,
        commit=commit,
        built_at_utc=built_at,
        bootstrapper_signature_status=bootstrapper_status,
        installer_signature_status="signed" if signing else "unsigned-prerelease",
        publisher_subject=publisher,
    )
    atomic_write_json(payload / "release-manifest.json", embedded)
    verify_manifest_files(embedded, payload)
    setup_output = build_root / "setup"
    setup_output.mkdir()
    numeric_version = version.replace("rc", ".") if "rc" in version else version + ".0"
    compiler_arguments = [
        str(iscc),
        f"/DPayloadDir={payload}",
        f"/DOutputDir={setup_output}",
        f"/DProductVersion={version}",
        f"/DDisplayVersion={numeric_version}",
        f"/DWheelFilename={wheel.name}",
        f"/DBootstrapperSha256={sha256_file(bootstrapper).removeprefix('sha256:')}",
        str(ROOT / "installer/windows/swarm-inference.iss"),
    ]
    run_captured(compiler_arguments, timeout_seconds=600)
    setup = setup_output / "SwarmInferenceSetup-x64.exe"
    if not setup.is_file():
        raise ReleaseError("Inno Setup did not produce SwarmInferenceSetup-x64.exe")
    installer_status = "unsigned-prerelease"
    if signing:
        setup_signature = sign_file(setup, signtool=arguments.signtool)
        if setup_signature.get("subject") != publisher:
            raise ReleaseError("bootstrapper and setup were signed by different publishers")
        installer_status = "signed"
    final = (arguments.output_dir or ROOT / "release/generated").resolve()
    _clean_final_directory(final)
    final_setup = final / setup.name
    final_wheel = final / wheel.name
    shutil.copyfile(setup, final_setup)
    shutil.copyfile(wheel, final_wheel)
    for filename in (
        "SwarmBootstrap.exe",
        "uv.exe",
        "windows-x64-cpu.requirements.lock",
        "windows-x64-cuda.requirements.lock",
        "LICENSE",
        "swarm.ico",
        "wizard-small.bmp",
        "wizard-large.bmp",
    ):
        shutil.copyfile(payload / filename, final / filename)
    acceptance = final / "productization-acceptance.zip"
    identity = {
        "version": version,
        "git_tag": tag,
        "git_commit": commit,
        "source_tree_clean": not dirty,
        "built_at_utc": built_at,
    }
    _acceptance_zip(acceptance, arguments.acceptance_zip, identity)
    final_manifest = build_manifest(
        payload_directory=payload,
        installer=final_setup,
        version=version,
        tag=tag,
        commit=commit,
        built_at_utc=built_at,
        bootstrapper_signature_status=bootstrapper_status,
        installer_signature_status=installer_status,
        publisher_subject=publisher,
    )
    manifest_path = final / "release-manifest.json"
    atomic_write_json(manifest_path, final_manifest)
    sbom_path = final / "swarm-inference-sbom.json"
    atomic_write_json(
        sbom_path,
        generate_sbom(
            version=version,
            tag=tag,
            commit=commit,
            built_at_utc=built_at,
            profile_paths=[
                payload / "windows-x64-cpu.requirements.lock",
                payload / "windows-x64-cuda.requirements.lock",
            ],
        ),
    )
    checksums = final / "SHA256SUMS"
    checksum_files = [
        path
        for path in final.iterdir()
        if path.is_file() and path.name not in {checksums.name, ".gitkeep"}
    ]
    write_sha256sums(checksums, checksum_files)
    receipt = {
        "schema_version": 1,
        **identity,
        "signature_status": installer_status,
        "publisher_subject": publisher,
        "installer": {"path": str(final_setup), "sha256": sha256_file(final_setup)},
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "payload_directory": str(payload),
        "release_directory": str(final),
        "toolchain": toolchain,
    }
    atomic_write_json(build_root / "build-receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag")
    parser.add_argument(
        "--release", action="store_true", help="Require a clean tagged source build."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--acceptance-zip", type=Path)
    parser.add_argument("--uv", type=Path)
    parser.add_argument("--dotnet", type=Path)
    parser.add_argument("--iscc", type=Path)
    parser.add_argument("--signtool", type=Path)
    parser.add_argument(
        "--fixture-version",
        help="Non-release fixture package version used by deterministic upgrade acceptance.",
    )
    parser.add_argument(
        "--fixture-doctor-failure",
        action="store_true",
        help="Inject an intentional installed-doctor failure into a non-release fixture wheel.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        receipt = build(build_parser().parse_args(argv))
    except (OSError, ReleaseError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "PASS", **receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
