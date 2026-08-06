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
