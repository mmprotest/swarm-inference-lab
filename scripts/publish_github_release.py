"""Create, verify, and optionally publish a draft-first GitHub Release."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from release_common import (
    PRODUCT,
    ROOT,
    ReleaseError,
    atomic_write_json,
    git_commit,
    parse_sha256sums,
    read_json_object,
    read_pyproject_version,
    run_captured,
    validate_manifest,
    verify_release_identity,
)
from verify_release_payload import verify_payload

REPOSITORY = "mmprotest/swarm-inference-lab"
REQUIRED_RELEASE_ASSETS = {
    "SwarmInferenceSetup-x64.exe",
    "release-manifest.json",
    "SHA256SUMS",
    "swarm-inference-sbom.json",
    "productization-acceptance.zip",
}


def _gh() -> str:
    executable = shutil.which("gh.exe") or shutil.which("gh")
    if executable is None:
        raise ReleaseError("the authenticated official gh CLI is unavailable")
    return executable


def _release_metadata(gh: str, tag: str, *, allow_missing: bool) -> dict[str, Any] | None:
    process = subprocess.run(
        [
            gh,
            "release",
            "view",
            tag,
            "--repo",
            REPOSITORY,
            "--json",
            "tagName,name,isDraft,isPrerelease,url,assets,targetCommitish",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode != 0:
        if allow_missing and "release not found" in (process.stderr + process.stdout).lower():
            return None
        raise ReleaseError(f"gh release view failed: {(process.stderr or process.stdout)[-2000:]}")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise ReleaseError("gh release view returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseError("gh release view returned no release object")
    return value


def _asset_paths(release_directory: Path) -> list[Path]:
    sums_path = release_directory / "SHA256SUMS"
    checksums = parse_sha256sums(sums_path)
    names = set(checksums) | {sums_path.name}
    missing_required = REQUIRED_RELEASE_ASSETS - names
    wheels = sorted(name for name in names if name.endswith("-py3-none-any.whl"))
    if missing_required or len(wheels) != 1:
        raise ReleaseError(
            f"release assets are incomplete; missing={sorted(missing_required)}, wheels={wheels}"
        )
    paths = [release_directory / name for name in sorted(names, key=str.casefold)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ReleaseError(f"release asset files are missing: {missing}")
    return paths


def _notes(manifest: dict[str, Any]) -> str:
    signature = manifest["installer"]["signature_status"]
    signature_text = "Authenticode signed" if signature == "signed" else "UNSIGNED PRERELEASE"
    return f"""# Swarm Inference {manifest["version"]}

Windows installation:

1. Download `SwarmInferenceSetup-x64.exe` from this release.
2. Double-click it and complete the per-user installer.
3. Open a new terminal.
4. Run `swarm --version` and `swarm node doctor`.

Installer status: **{signature_text}**.

The application installer, Python runtime, uv executable, application wheel, and exact CPU/CUDA
dependency profiles are identified by `release-manifest.json` and `SHA256SUMS`. CUDA software is
packaged, but physical RTX 5090 validation is a separate acceptance gate and is not claimed by
the GitHub-hosted CPU build.
"""


def _title(manifest: dict[str, Any]) -> str:
    if manifest["installer"]["signature_status"] == "unsigned-prerelease":
        return f"{manifest['git_tag']} — unsigned prerelease"
    return str(manifest["git_tag"])


def _verify_download(
    *,
    gh: str,
    tag: str,
    expected_assets: set[str],
    run_install_test: bool,
    phase: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"swarm-release-{phase}-") as raw:
        directory = Path(raw)
        run_captured(
            [gh, "release", "download", tag, "--repo", REPOSITORY, "--dir", str(directory)],
            timeout_seconds=600,
        )
        downloaded = {path.name for path in directory.iterdir() if path.is_file()}
        if downloaded != expected_assets:
            raise ReleaseError(
                f"{phase} release download asset mismatch; "
                f"missing={sorted(expected_assets - downloaded)}, "
                f"unexpected={sorted(downloaded - expected_assets)}"
            )
        verification = verify_payload(
            manifest_path=directory / "release-manifest.json",
            payload_directories=[directory],
            checksums_path=directory / "SHA256SUMS",
        )
        if run_install_test:
            powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
            if powershell is None:
                raise ReleaseError("PowerShell is unavailable for downloaded-setup acceptance")
            run_captured(
                [
                    powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(ROOT / "scripts/test_windows_installer.ps1"),
                    "-SetupPath",
                    str(directory / "SwarmInferenceSetup-x64.exe"),
                    "-Label",
                    phase,
                ],
                timeout_seconds=2400,
            )
        return {"phase": phase, "assets": sorted(downloaded), **verification}


def publish(arguments: argparse.Namespace) -> dict[str, Any]:
    release_directory = arguments.release_dir.resolve()
    manifest = validate_manifest(read_json_object(release_directory / "release-manifest.json"))
    version = read_pyproject_version()
    tag = arguments.tag
    commit = git_commit()
    verify_release_identity(
        version=version,
        tag=tag,
        commit=commit,
        require_clean=True,
        require_tag=True,
    )
    if manifest["version"] != version or manifest["git_tag"] != tag:
        raise ReleaseError("release manifest does not identify the tagged package version")
    if manifest["git_commit"] != commit:
        raise ReleaseError("release manifest commit does not identify the tagged checkout")
    if manifest["product"] != PRODUCT:
        raise ReleaseError("release manifest product identity is invalid")
    assets = _asset_paths(release_directory)
    verify_payload(
        manifest_path=release_directory / "release-manifest.json",
        payload_directories=[release_directory],
        checksums_path=release_directory / "SHA256SUMS",
    )
    gh = _gh()
    existing = _release_metadata(gh, tag, allow_missing=True)
    if existing is not None and not arguments.resume_draft:
        raise ReleaseError(f"a GitHub Release already exists for immutable tag {tag}")
    if existing is not None and not existing.get("isDraft"):
        raise ReleaseError(f"release {tag} is already published and will not be overwritten")

    with tempfile.TemporaryDirectory(prefix="swarm-release-notes-") as raw:
        notes = Path(raw) / "release-notes.md"
        notes.write_text(_notes(manifest), encoding="utf-8", newline="\n")
        if existing is None:
            command = [
                gh,
                "release",
                "create",
                tag,
                "--repo",
                REPOSITORY,
                "--verify-tag",
                "--title",
                _title(manifest),
                "--notes-file",
                str(notes),
                "--draft",
            ]
            if manifest["channel"] == "prerelease":
                command.append("--prerelease")
            command.extend(str(path) for path in assets)
            run_captured(command, timeout_seconds=900)

    draft = _release_metadata(gh, tag, allow_missing=False)
    assert draft is not None
    if not draft.get("isDraft") or draft.get("tagName") != tag:
        raise ReleaseError("GitHub release was not created as the expected draft")
    expected_assets = {path.name for path in assets}
    remote_assets = {
        item.get("name")
        for item in draft.get("assets", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if remote_assets != expected_assets:
        raise ReleaseError(
            "draft release asset inventory does not match the verified local payload"
        )
    draft_verification = _verify_download(
        gh=gh,
        tag=tag,
        expected_assets=expected_assets,
        run_install_test=not arguments.skip_install_test,
        phase="draft",
    )
    if not arguments.publish:
        return {
            "status": "PASS",
            "published": False,
            "draft": True,
            "url": draft.get("url"),
            "assets": sorted(expected_assets),
            "draft_verification": draft_verification,
        }

    edit = [gh, "release", "edit", tag, "--repo", REPOSITORY, "--draft=false"]
    if manifest["channel"] == "prerelease":
        edit.append("--prerelease")
    else:
        edit.append("--latest")
    run_captured(edit, timeout_seconds=120)
    published = _release_metadata(gh, tag, allow_missing=False)
    assert published is not None
    if published.get("isDraft") or bool(published.get("isPrerelease")) != (
        manifest["channel"] == "prerelease"
    ):
        raise ReleaseError("published GitHub Release channel metadata is incorrect")
    post_verification = _verify_download(
        gh=gh,
        tag=tag,
        expected_assets=expected_assets,
        run_install_test=not arguments.skip_install_test,
        phase="published",
    )
    return {
        "status": "PASS",
        "published": True,
        "draft": False,
        "url": published.get("url"),
        "assets": sorted(expected_assets),
        "draft_verification": draft_verification,
        "post_publication_verification": post_verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/generated")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--resume-draft", action="store_true")
    parser.add_argument("--skip-install-test", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = publish(arguments)
        if arguments.output is not None:
            atomic_write_json(arguments.output, result)
    except (OSError, ReleaseError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "FAIL", "detail": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
