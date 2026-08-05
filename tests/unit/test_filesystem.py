from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

import swarm_inference.filesystem as filesystem


def test_atomic_replace_retries_transient_sharing_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    actual_replace = os.replace
    calls = 0

    def transient_then_replace(first: Path, second: Path) -> None:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError(errno.EACCES, "sharing violation", str(second))
        actual_replace(first, second)

    monkeypatch.setattr(filesystem.os, "replace", transient_then_replace)
    monkeypatch.setattr(filesystem.time, "sleep", lambda _seconds: None)
    filesystem.replace_atomically(source, destination)
    assert calls == 3
    assert destination.read_text(encoding="utf-8") == "new"
    assert not source.exists()


def test_atomic_replace_does_not_retry_unrelated_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    source.write_text("new", encoding="utf-8")
    calls = 0

    def fail(_source: Path, _destination: Path) -> None:
        nonlocal calls
        calls += 1
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(filesystem.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        filesystem.replace_atomically(source, tmp_path / "destination.json")
    assert calls == 1
