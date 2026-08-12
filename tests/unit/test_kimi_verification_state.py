from __future__ import annotations

from types import SimpleNamespace

import pytest

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor


class _CopyRuntime:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.copies: list[tuple[object, object, int]] = []
        self.synchronizations = 0

    def execute_copy(self, destination: object, source: object, count: int) -> None:
        if self.fail:
            raise RuntimeError("copy failed")
        self.copies.append((destination, source, count))

    def synchronize(self) -> None:
        self.synchronizations += 1


def _executor(*, fail: bool = False) -> PersistentKimiStageExecutor:
    executor = object.__new__(PersistentKimiStageExecutor)
    executor.layer = 91
    executor.config = SimpleNamespace(
        kda_layers=frozenset({89}),
        kv_lora=512,
        query_rope=64,
    )
    executor.runtime = _CopyRuntime(fail=fail)
    executor._sessions = {
        "source": SimpleNamespace(
            attention_state={"latent_cache": "source-latent", "rope_cache": "source-rope"},
            maximum_context=32,
            cache_length=11,
        )
    }

    def open_session(
        session_id: str,
        *,
        maximum_context_override: int | None = None,
    ) -> None:
        executor._sessions[session_id] = SimpleNamespace(
            attention_state={
                "latent_cache": "destination-latent",
                "rope_cache": "destination-rope",
            },
            maximum_context=maximum_context_override,
            cache_length=0,
        )

    executor.open_session = open_session  # type: ignore[method-assign]
    return executor


def test_clone_session_state_copies_only_active_mla_prefix() -> None:
    executor = _executor()

    result = executor.clone_session_state("source", "destination", maximum_context_override=64)

    assert executor.runtime.copies == [
        ("destination-latent", "source-latent", 11 * 512),
        ("destination-rope", "source-rope", 11 * 64),
    ]
    assert executor.runtime.synchronizations == 1
    assert executor._sessions["destination"].cache_length == 11
    assert result["copied_device_to_device_bytes"] == 11 * (512 + 64) * 4


def test_clone_session_state_rejects_truncation_before_allocation() -> None:
    executor = _executor()

    with pytest.raises(ValueError, match="cannot truncate"):
        executor.clone_session_state("source", "destination", maximum_context_override=10)

    assert "destination" not in executor._sessions


def test_clone_session_state_cleans_up_destination_after_copy_failure() -> None:
    executor = _executor(fail=True)
    cleaned: list[str] = []

    def close_session(session_id: str) -> int:
        cleaned.append(session_id)
        del executor._sessions[session_id]
        return 0

    executor.close_session = close_session  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="copy failed"):
        executor.clone_session_state("source", "destination")

    assert cleaned == ["destination"]
    assert "destination" not in executor._sessions
