from __future__ import annotations

import builtins

import pytest

from swarm_inference.config.models import Backend
from swarm_inference.exceptions import BackendIncompatibleError
from swarm_inference.worker.capabilities import _gpu_details


def test_synthetic_backend_does_not_import_torch(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("synthetic capability detection imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model, total, available, dtypes = _gpu_details(Backend.SYNTHETIC)
    assert model is None
    assert total == available == 0
    assert "float32" in dtypes


def test_torch_worker_fails_precisely_when_pytorch_is_missing(monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("test missing torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(BackendIncompatibleError, match="PyTorch could not be imported"):
        _gpu_details(Backend.TORCH_CPU)
