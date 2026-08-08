from __future__ import annotations

import pytest

from swarm_inference.engines import local_capabilities


def test_native_adapter_failure_does_not_block_other_engine_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_registry() -> None:
        raise PermissionError("native numerical runtime is unavailable")

    monkeypatch.setattr(local_capabilities, "default_native_adapter_registry", broken_registry)

    capability = local_capabilities._native_capability()

    assert not capability.enabled
    assert capability.adapters == ()
    assert "native adapter discovery failed" in capability.detail
