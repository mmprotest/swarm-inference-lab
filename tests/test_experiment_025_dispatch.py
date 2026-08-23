from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from swarm_inference.experiments.experiment_020.transport import Frame, MessageType
from swarm_inference.experiments.experiment_025 import canary_runtime as canary_module
from swarm_inference.experiments.experiment_025 import worker as worker_module
from swarm_inference.experiments.experiment_025.wire import (
    Action,
    pack_payload,
    unpack_payload,
)


class _FakeNativeStage:
    def __init__(self) -> None:
        self.request = SimpleNamespace(
            assignment=SimpleNamespace(
                owns_embeddings=False,
                owns_final_norm=False,
                owns_output_projection=False,
            )
        )
        self.execution_records: list[dict[str, object]] = []
        self.called = False

    def execute_decode(
        self,
        *,
        session_id: str,
        hidden_states: torch.Tensor,
        cache_position_start: int,
    ) -> SimpleNamespace:
        self.called = True
        assert session_id == "physical-session"
        assert cache_position_start == 2
        output = hidden_states + 1
        self.execution_records.append(
            {
                "position": cache_position_start,
                "layer_output": output.numpy()[0, 0],
                "boundary_output": output.numpy(),
                "selected_expert_ids": [7, 11],
                "weight_loads_during_execute": 0,
                "external_expert_dispatch": None,
            }
        )
        return SimpleNamespace(
            stage_boundary_hidden_states=output,
            final_hidden_states=None,
            logits=None,
            sampled_token_ids=None,
            cache_sequence_length=3,
            compute_ns=1234,
        )


def test_execute_shard_frame_reaches_current_native_stage_dispatch(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        worker_module,
        "PersistentKimiStageExecutor",
        _FakeNativeStage,
    )
    runtime = object.__new__(worker_module.WorkerRuntime)
    runtime.worker_id = "e025-stage-010"
    runtime.role = "BACKBONE_STAGE"
    runtime.executor = _FakeNativeStage()
    runtime.collective = None
    runtime.lock = threading.Lock()
    runtime.telemetry_path = tmp_path / "telemetry.jsonl"
    boundary = np.zeros((1, 9, 7168), dtype=np.float32)
    request = Frame(
        MessageType.EXECUTE_SHARD,
        "dispatch-test",
        2,
        runtime.worker_id,
        "physical-session",
        pack_payload(
            Action.EXECUTE_STAGE,
            {"session_id": "physical-session", "position": 2, "layer": 10},
            {"boundary": boundary},
        ),
    )
    response = runtime.process(request)
    assert response.message_type is MessageType.SHARD_RESULT
    action, metadata, arrays = unpack_payload(response.payload)
    assert action is Action.EXECUTE_STAGE
    assert runtime.executor.called is True
    assert metadata["native_dispatch"] is True
    assert metadata["native_primitive"] == "PersistentKimiStageExecutor"
    assert metadata["cached_output"] is False
    assert metadata["synthetic_tensor"] is False
    assert metadata["controller_compute_fallback"] is False
    assert metadata["execution"]["weight_loads_during_execute"] == 0
    assert metadata["execution"]["boundary_output_shape"] == [1, 9, 7168]
    np.testing.assert_array_equal(arrays["boundary"], boundary + 1)
    assert "REQUEST_COMPLETE" in runtime.telemetry_path.read_text(encoding="utf-8")


def test_physical_stage_receipt_retains_external_expert_dispatch(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    boundary = np.zeros((1, 1, 4), dtype=np.float32)
    execution = {
        "selected_expert_ids": [7, 11],
        "weight_loads_during_execute": 0,
        "external_expert_dispatch": {
            "workers_invoked": 4,
            "every_selected_expert_executed_once": True,
            "native_expert_calls": 16,
            "whole_layer_fallback": False,
            "workers": [],
        },
    }

    class _FakeCanaryConnection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _FakeCanaryConnection:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(
            self,
            action: Action,
            _metadata: dict[str, object],
            _arrays: dict[str, np.ndarray] | None = None,
        ) -> tuple[MessageType, dict[str, object], dict[str, np.ndarray], dict[str, int]]:
            network = {"request_wire_bytes": 1, "response_wire_bytes": 1}
            if action is Action.REGISTER:
                return MessageType.SHARD_RESULT, {
                    "ready": {
                        "worker_id": "parent",
                        "gpu": {"torch_cuda_available": True},
                    }
                }, {}, network
            if action is Action.OPEN_SESSION:
                return MessageType.SHARD_RESULT, {"opened": True}, {}, network
            if action is Action.EXECUTE_STAGE:
                return MessageType.SHARD_RESULT, {
                    "native_dispatch": True,
                    "execution": execution,
                }, {"boundary": boundary}, network
            if action is Action.HEALTH:
                return MessageType.SHARD_RESULT, {
                    "runtime": {"cuda_error_state_ok": True}
                }, {}, network
            if action is Action.CLOSE_SESSION:
                return MessageType.SHARD_RESULT, {"released_kv_bytes": 1}, {}, network
            raise AssertionError(action)

    worker = SimpleNamespace(
        worker_id="parent",
        layer=89,
        host="127.0.0.1",
        port=1,
        endpoint=lambda: {"worker_id": "parent", "layer": 89},
    )
    trace = tmp_path / "trace.bin"
    routes = tmp_path / "routes.txt"
    credential = tmp_path / "credential.bin"
    certificate = tmp_path / "certificate.pem"
    for path in (trace, routes, credential, certificate):
        path.write_bytes(b"fixture")
    monkeypatch.setattr(  # type: ignore[attr-defined]
        canary_module, "CanaryConnection", _FakeCanaryConnection
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        canary_module, "_stage_fixtures", lambda *_args, **_kwargs: ([boundary], [boundary])
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        canary_module, "_parse_oracle_routes", lambda _path: {89: {0: [7, 11]}}
    )
    receipt = canary_module.run_physical_stage_fixture(
        worker=worker,
        checkpoint=tmp_path,
        oracle_trace=trace,
        oracle_routes=routes,
        credential_path=credential,
        certificate=certificate,
        output_path=tmp_path / "receipt.json",
        cycle_id="schema-regression",
    )
    assert receipt["status"] == "PASS"
    assert receipt["rows"][0]["execution"]["external_expert_dispatch"] == execution[
        "external_expert_dispatch"
    ]
