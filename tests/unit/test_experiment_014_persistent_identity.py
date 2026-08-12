from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import swarm_inference.execution.kimi_k3_stage as kimi_stage
from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _configure_kimi_cpu_transport_threads,
    _prepare_retained_device_bytes,
    _production_batch_capacity,
)
from swarm_inference.experiments.experiment_014.cuda import KimiCudaError
from swarm_inference.experiments.experiment_014.persistent_stages import (
    _resolve_worker_identity_manifest,
)


def _write_identity(path: Path, *, worker_id: str, assignment_sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "assignment_sha256": assignment_sha256,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_scoped_identity_directory_resolves_exact_worker(tmp_path: Path) -> None:
    identity = _write_identity(
        tmp_path / "k3-worker-003-model-identity.json",
        worker_id="k3-worker-003",
        assignment_sha256="a" * 64,
    )

    resolved, evidence = _resolve_worker_identity_manifest(tmp_path, layer=3)

    assert resolved == identity.resolve()
    assert evidence["worker_id"] == "k3-worker-003"
    assert evidence["assignment_sha256"] == "a" * 64
    assert len(evidence["manifest_sha256"]) == 64


def test_scoped_identity_direct_file_remains_supported(tmp_path: Path) -> None:
    identity = _write_identity(
        tmp_path / "identity.json",
        worker_id="k3-worker-001",
        assignment_sha256="b" * 64,
    )

    resolved, _ = _resolve_worker_identity_manifest(identity, layer=1)

    assert resolved == identity.resolve()


@pytest.mark.parametrize(
    ("worker_id", "assignment_sha256", "message"),
    [
        ("k3-worker-002", "c" * 64, "wrong worker"),
        ("k3-worker-001", "not-a-digest", "no valid assignment"),
    ],
)
def test_scoped_identity_rejects_wrong_worker_or_assignment(
    tmp_path: Path,
    worker_id: str,
    assignment_sha256: str,
    message: str,
) -> None:
    identity = _write_identity(
        tmp_path / "identity.json",
        worker_id=worker_id,
        assignment_sha256=assignment_sha256,
    )

    with pytest.raises(ValueError, match=message):
        _resolve_worker_identity_manifest(identity, layer=1)


def test_scoped_identity_rejects_missing_worker_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing scoped identity"):
        _resolve_worker_identity_manifest(tmp_path, layer=1)


def test_production_batch_capacity_caps_native_batch_16() -> None:
    assert _production_batch_capacity((1, 2, 4, 8, 16)) == 8
    assert _production_batch_capacity((1, 2, 4)) == 4


def test_batch_9_rejects_before_runtime_access() -> None:
    executor = object.__new__(PersistentKimiStageExecutor)
    executor._owns_embeddings = False
    executor._owns_final_endpoint = False
    executor._weights = {"mlp_type": "moe"}
    executor._batch_capacity = 8

    with pytest.raises(
        KimiCudaError,
        match="requested=9, certified_max=8",
    ):
        executor.execute_decode_batch(
            session_ids=tuple(f"stream-{row}" for row in range(9)),
            hidden_states=torch.zeros((9, 9, 7168), dtype=torch.float32),
            cache_position_start=0,
        )


def test_prepare_retained_bytes_reports_cuda_allocator_plateau() -> None:
    assert _prepare_retained_device_bytes(10 * 1024**2, 8 * 1024**2) == 2 * 1024**2
    assert _prepare_retained_device_bytes(8 * 1024**2, 10 * 1024**2) == 0


def test_kimi_cpu_transport_thread_contract_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(kimi_stage, "_KIMI_CPU_TRANSPORT_CONFIGURED", False)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)

    expected = {"intraop_threads": 1, "interop_threads": 1}
    assert _configure_kimi_cpu_transport_threads() == expected
    assert _configure_kimi_cpu_transport_threads() == expected
    assert calls == [("intra", 1), ("interop", 1)]


def test_kimi_cpu_transport_rejects_preinitialized_interop_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kimi_stage, "_KIMI_CPU_TRANSPORT_CONFIGURED", False)
    monkeypatch.setattr(torch, "set_num_threads", lambda _value: None)
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda _value: (_ for _ in ()).throw(RuntimeError("already started")),
    )
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 20)

    with pytest.raises(KimiCudaError, match="initialized before"):
        _configure_kimi_cpu_transport_threads()
