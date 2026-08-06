from __future__ import annotations

from pathlib import Path


def test_inno_setup_is_per_user_upgrade_stable_and_invokes_native_engine(
    repository_root: Path,
) -> None:
    source = (repository_root / "installer/windows/swarm-inference.iss").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "946cb20d-3399-4c3c-ad55-41c851c02e56" in lowered
    assert "privilegesrequired=lowest" in lowered
    assert "defaultdirname={localappdata}\\programs\\swarminference" in lowered
    assert "architecturesallowed=x64compatible" in lowered
    assert "setuparchitecture=x64" in lowered
    assert "swarmbootstrap.exe" in lowered
    assert "preparetoinstall" in lowered
    assert "exitcode <> 0" in lowered
    assert "{param:backend|auto}" in lowered
    assert "{param:purgestate|}" in lowered
    assert "{param:allowdowngrade|0}" in lowered
    assert "--setup-path" in lowered
    assert "install.ps1" not in lowered
    assert "powershell" not in lowered


def test_bootstrapper_is_self_contained_single_file_and_has_required_operations(
    repository_root: Path,
) -> None:
    project = (
        repository_root / "installer/windows/SwarmBootstrap/SwarmBootstrap.csproj"
    ).read_text(encoding="utf-8")
    program = (repository_root / "installer/windows/SwarmBootstrap/Program.cs").read_text(
        encoding="utf-8"
    )
    assert "<RuntimeIdentifier>win-x64</RuntimeIdentifier>" in project
    assert "<SelfContained>true</SelfContained>" in project
    assert "<PublishSingleFile>true</PublishSingleFile>" in project
    for operation in ("install", "repair", "upgrade", "uninstall", "doctor", "detect-backend"):
        assert f'"{operation}"' in program


def test_bootstrapper_binds_public_manifest_to_exact_setup(repository_root: Path) -> None:
    source = (
        repository_root / "installer/windows/SwarmBootstrap/ReleaseManifestResolver.cs"
    ).read_text(encoding="utf-8")
    verifier = (repository_root / "installer/windows/SwarmBootstrap/HashVerifier.cs").read_text(
        encoding="utf-8"
    )
    assert "release-assets.githubusercontent.com" in source
    assert "objects.githubusercontent.com" in source
    assert "MaximumManifestBytes" in source
    assert "AllowAutoRedirect = false" in source
    assert "setupPath" in source
    assert "release.Installer.SizeBytes" in verifier
    assert "release.Installer.Sha256" in verifier


def test_install_and_state_roots_are_separate(repository_root: Path) -> None:
    layout = (repository_root / "installer/windows/SwarmBootstrap/InstallLayout.cs").read_text(
        encoding="utf-8"
    )
    assert 'Path.Combine(Path.GetFullPath(localAppData), "SwarmInference")' in layout
    assert "SwarmInference" in layout
    assert "StateRoot" in layout
    record = (repository_root / "installer/windows/SwarmBootstrap/InstallRecord.cs").read_text(
        encoding="utf-8"
    )
    assert "native-windows" in record
    assert "InstallationMode" in record


def test_candidate_payload_cache_is_published_before_managed_python_creation(
    repository_root: Path,
) -> None:
    source = (repository_root / "installer/windows/SwarmBootstrap/RuntimeInstaller.cs").read_text(
        encoding="utf-8"
    )
    assert source.count("transaction.PublishPayloadCache();") == 1
    assert source.index("transaction.PublishPayloadCache();") < source.index(
        "InstallCandidateAsync("
    )
    assert "HashVerifier.Verify(controlledUv, manifest.Uv);" in source
