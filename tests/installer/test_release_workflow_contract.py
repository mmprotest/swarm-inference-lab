from __future__ import annotations

import json
import re
from pathlib import Path


def test_release_workflow_is_tag_gated_and_has_required_permissions(
    repository_root: Path,
) -> None:
    workflow = (repository_root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert 'tags:\n      - "v*"' in workflow
    assert "contents: write" in workflow
    assert "id-token: write" in workflow
    assert "attestations: write" in workflow
    assert "--release --tag $env:RELEASE_TAG" in workflow
    assert "verify_release_version.py --tag $env:RELEASE_TAG" in workflow


def test_release_workflow_is_draft_first_and_uploads_required_assets(
    repository_root: Path,
) -> None:
    workflow = (repository_root / ".github/workflows/release.yml").read_text(encoding="utf-8")
    publisher = (repository_root / "scripts/publish_github_release.py").read_text(encoding="utf-8")
    assert "publish_github_release.py" in workflow
    assert '"--draft"' in publisher
    assert '"release", "download"' in publisher
    assert '"release", "edit"' in publisher
    assert "post_publication_verification" in publisher
    for asset in (
        "SwarmInferenceSetup-x64.exe",
        "release-manifest.json",
        "SHA256SUMS",
        "swarm-inference-sbom.json",
        "productization-acceptance.zip",
    ):
        assert asset in publisher


def test_pull_request_workflow_never_publishes_and_runs_lifecycle_acceptance(
    repository_root: Path,
) -> None:
    workflow = (repository_root / ".github/workflows/installer.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "publish_github_release.py" not in workflow
    for script in (
        "test_windows_installer.ps1",
        "test_windows_upgrade.ps1",
        "test_windows_uninstall.ps1",
    ):
        assert script in workflow
    assert "--fixture-doctor-failure" in workflow


def test_workflow_actions_are_immutable_commit_pins(repository_root: Path) -> None:
    for name in ("installer.yml", "release.yml"):
        workflow = (repository_root / ".github/workflows" / name).read_text(encoding="utf-8")
        action_refs = re.findall(r"(?m)^\s+uses:\s+([^\s#]+)", workflow)
        assert action_refs
        assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)
        assert not re.search(r"(?m)^\s+uses:\s+[^\s]+@v\d+", workflow)


def test_windows_workflows_pin_available_build_python_separately_from_runtime(
    repository_root: Path,
) -> None:
    for name in ("installer.yml", "release.yml"):
        workflow = (repository_root / ".github/workflows" / name).read_text(encoding="utf-8")
        assert 'python-version: "3.11.9"' in workflow
        assert 'python-version: "3.11.15"' not in workflow
    toolchain = json.loads(
        (repository_root / "installer/windows/toolchain.json").read_text(encoding="utf-8")
    )
    assert toolchain["python"]["version"] == "3.11.15"


def test_windows_workflows_use_portable_hosted_cpu_and_fail_fast(
    repository_root: Path,
) -> None:
    for name in ("installer.yml", "release.yml"):
        workflow = (repository_root / ".github/workflows" / name).read_text(encoding="utf-8")
        assert "ATEN_CPU_CAPABILITY: default" in workflow
        assert "$PSNativeCommandUseErrorActionPreference = $true" in workflow
    productization = (repository_root / ".github/workflows/productization.yml").read_text(
        encoding="utf-8"
    )
    assert "Select portable CPU kernels on hosted Windows" in productization


def test_normal_readme_does_not_use_powershell_script_installation(repository_root: Path) -> None:
    readme = (repository_root / "README.md").read_text(encoding="utf-8")
    normal = readme.split("Developer, CI and offline recovery installation", maxsplit=1)[0]
    assert "SwarmInferenceSetup-x64.exe" in normal
    assert "install.ps1" not in normal
