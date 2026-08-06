from __future__ import annotations

from pathlib import Path

import pytest
import release_common
from release_common import (
    ReleaseError,
    git_tag_to_pep440,
    pep440_to_git_tag,
    read_pyproject_version,
    verify_release_identity,
)


def test_current_version_is_single_sourced_from_pyproject(repository_root: Path) -> None:
    assert read_pyproject_version(repository_root / "pyproject.toml") == "0.1.0rc5"
    source = (repository_root / "src/swarm_inference/__init__.py").read_text(encoding="utf-8")
    assert 'version("swarm-inference-lab")' in source
    assert '__version__ = "0.1.0' not in source
    assert '__version__ = "0.0.0+source"' in source


@pytest.mark.parametrize(
    ("version", "tag"),
    [("0.1.0rc1", "v0.1.0-rc.1"), ("2.4.6rc12", "v2.4.6-rc.12"), ("1.2.3", "v1.2.3")],
)
def test_pep440_and_git_tag_mapping(version: str, tag: str) -> None:
    assert pep440_to_git_tag(version) == tag
    assert git_tag_to_pep440(tag) == version


def test_release_identity_rejects_mismatch_dirty_tree_and_moved_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "1" * 40
    monkeypatch.setattr(release_common, "git_commit", lambda **_kwargs: commit)
    monkeypatch.setattr(release_common, "git_is_dirty", lambda **_kwargs: False)
    monkeypatch.setattr(release_common, "git_output", lambda *_args, **_kwargs: commit)
    with pytest.raises(ReleaseError, match="maps to"):
        verify_release_identity(version="0.1.0rc1", tag="v0.1.0-rc.2", commit=commit)

    monkeypatch.setattr(release_common, "git_is_dirty", lambda **_kwargs: True)
    with pytest.raises(ReleaseError, match="clean source tree"):
        verify_release_identity(version="0.1.0rc1", tag="v0.1.0-rc.1", commit=commit)

    monkeypatch.setattr(release_common, "git_is_dirty", lambda **_kwargs: False)
    monkeypatch.setattr(release_common, "git_output", lambda *_args, **_kwargs: "2" * 40)
    with pytest.raises(ReleaseError, match="points to"):
        verify_release_identity(version="0.1.0rc1", tag="v0.1.0-rc.1", commit=commit)


@pytest.mark.parametrize("value", ["0.1.0-rc.1", "v0.1.0rc1", "latest", "0.1"])
def test_unsupported_version_spellings_are_rejected(value: str) -> None:
    with pytest.raises(ReleaseError):
        pep440_to_git_tag(value)
