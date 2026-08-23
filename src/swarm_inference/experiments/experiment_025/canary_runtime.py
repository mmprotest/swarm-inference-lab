"""Physical one-stage and four-worker sub-layer correctness canaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _numerical_metrics,
)
from swarm_inference.execution.kimi_k3_graph_runtime import _parse_oracle_routes
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_020.transport import Frame, MessageType

from .constants import GIB, MIB, SUB_LAYER_TARGET, SUB_LAYER_WORKERS
from .io import atomic_write_json, read_json, sha256_file, utc_now
from .provisioning import LiveWorker
from .wire import Action, AuthenticatedConnection, pack_payload, unpack_payload


class CanaryConnection:
    def __init__(
        self,
        worker: LiveWorker,
        credential: bytes,
        certificate: Path,
    ) -> None:
        self.worker = worker
        self.connection = AuthenticatedConnection(
            worker.host,
            worker.port,
            credential,
            certificate,
            timeout_seconds=180.0,
        )
        self.sequence = 0

    def request(
        self,
        action: Action,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
        *,
        allow_error: bool = False,
    ) -> tuple[MessageType, dict[str, Any], dict[str, np.ndarray], dict[str, int]]:
        self.sequence += 1
        before = (self.connection.sent_bytes, self.connection.received_bytes)
        frame = Frame(
            MessageType.EXECUTE_SHARD,
            f"e025-canary-{self.worker.worker_id}-{self.sequence:04d}",
            self.sequence,
            self.worker.worker_id,
            str(metadata.get("session_id", "e025-canary-control")),
            pack_payload(action, metadata, arrays),
        )
        response = self.connection.request(frame)
        response_action, response_metadata, response_arrays = unpack_payload(
            response.payload
        )
        if response.message_type is MessageType.ERROR:
            if allow_error:
                return response.message_type, response_metadata, response_arrays, {
                    "request_wire_bytes": self.connection.sent_bytes - before[0],
                    "response_wire_bytes": self.connection.received_bytes - before[1],
                }
            raise RuntimeError(f"physical canary worker failed: {response_metadata}")
        if response.message_type is not MessageType.SHARD_RESULT:
            raise RuntimeError("physical canary returned an invalid message type")
        if response_action is not action:
            raise RuntimeError("physical canary returned the wrong action")
        return response.message_type, response_metadata, response_arrays, {
            "request_wire_bytes": self.connection.sent_bytes - before[0],
            "response_wire_bytes": self.connection.received_bytes - before[1],
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> CanaryConnection:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def run_physical_stage_fixture(
    *,
    worker: LiveWorker,
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    credential_path: Path,
    certificate: Path,
    output_path: Path,
    cycle_id: str,
) -> dict[str, Any]:
    inputs, expected = _stage_fixtures(
        checkpoint.expanduser().resolve(),
        oracle_trace.expanduser().resolve(),
        layer=worker.layer,
    )
    expected_routes = _parse_oracle_routes(oracle_routes).get(worker.layer, {})
    session_id = f"{cycle_id}-physical"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter_ns()
    with CanaryConnection(
        worker,
        credential_path.read_bytes(),
        certificate,
    ) as client:
        _, registered, _, registration_network = client.request(Action.REGISTER, {})
        _, opened, _, open_network = client.request(
            Action.OPEN_SESSION,
            {"session_id": session_id, "maximum_context": 3},
        )
        if opened.get("opened") is not True:
            raise RuntimeError("physical stage canary session did not open")
        for position, (stage_input, expected_output) in enumerate(
            zip(inputs, expected, strict=True)
        ):
            arrays = (
                {"token_ids": np.ascontiguousarray(stage_input, dtype=np.int64)}
                if worker.layer == 0
                else {"boundary": np.ascontiguousarray(stage_input, dtype=np.float32)}
            )
            message_type, metadata, output_arrays, network = client.request(
                Action.EXECUTE_STAGE,
                {
                    "session_id": session_id,
                    "position": position,
                    "layer": worker.layer,
                },
                arrays,
            )
            actual = np.ascontiguousarray(output_arrays["boundary"], dtype=np.float32)
            metrics = _numerical_metrics(expected_output, actual)
            execution = metadata["execution"]
            observed_routes = [int(value) for value in execution["selected_expert_ids"]]
            rows.append(
                {
                    "position": position,
                    "message_type": message_type.name,
                    "input_fingerprint": _array_fingerprint(stage_input),
                    "expected_fingerprint": _array_fingerprint(expected_output),
                    "actual_fingerprint": _array_fingerprint(actual),
                    "metrics": metrics,
                    "within_gate": float(metrics["relative_l2_error"]) <= 1e-4
                    and float(metrics["maximum_absolute_error"]) <= 1e-3,
                    "expected_route": expected_routes.get(position, []),
                    "observed_route": observed_routes,
                    "route_exact": observed_routes == expected_routes.get(position, []),
                    "native_dispatch": metadata.get("native_dispatch") is True,
                    "execution": execution,
                    "weight_loads_during_execute": execution.get(
                        "weight_loads_during_execute"
                    ),
                    "network": network,
                }
            )
        _, health, _, health_network = client.request(Action.HEALTH, {})
        _, closed, _, close_network = client.request(
            Action.CLOSE_SESSION,
            {"session_id": session_id},
        )
    gates = {
        "registered_identity_exact": registered["ready"]["worker_id"]
        == worker.worker_id,
        "real_cuda_visible": bool(registered["ready"]["gpu"]["torch_cuda_available"]),
        "all_three_outputs_numerically_correct": all(row["within_gate"] for row in rows),
        "all_routes_exact": all(row["route_exact"] for row in rows),
        "all_production_execute_shard_dispatches_native": all(
            row["message_type"] == "SHARD_RESULT" and row["native_dispatch"]
            for row in rows
        ),
        "no_execute_time_weight_loads": all(
            int(row["weight_loads_during_execute"] or 0) == 0 for row in rows
        ),
        "gpu_health_valid": bool(health["runtime"]["cuda_error_state_ok"]),
        "session_closed": int(closed["released_kv_bytes"]) > 0,
    }
    payload = {
        "schema_version": "experiment-025-physical-stage-canary-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class": "PHYSICAL_SINGLE_MACHINE_CONSUMER_GPU_CANARY",
        "cycle_id": cycle_id,
        "worker": worker.endpoint(),
        "ready": registered["ready"],
        "oracle_trace": str(oracle_trace.resolve()),
        "oracle_trace_sha256": sha256_file(oracle_trace.resolve()),
        "oracle_routes": str(oracle_routes.resolve()),
        "oracle_routes_sha256": sha256_file(oracle_routes.resolve()),
        "rows": rows,
        "health": health,
        "transport": {
            "registration": registration_network,
            "open": open_network,
            "health": health_network,
            "close": close_network,
        },
        "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
        "gates": gates,
    }
    atomic_write_json(output_path, payload)
    return payload


def run_physical_sub_layer_canary(
    *,
    parent: LiveWorker,
    fragments: list[LiveWorker],
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    physical_placement: Path,
    credential_path: Path,
    certificate: Path,
    output_path: Path,
) -> dict[str, Any]:
    if len(fragments) != SUB_LAYER_WORKERS:
        raise ValueError("E025 physical sub-layer canary requires four fragments")
    if len({worker.machine_id for worker in fragments}) != SUB_LAYER_WORKERS:
        raise ValueError("E025 physical sub-layer workers are not on distinct machines")
    positive_path = output_path.with_name(output_path.stem + "-positive.json")
    positive = run_physical_stage_fixture(
        worker=parent,
        checkpoint=checkpoint,
        oracle_trace=oracle_trace,
        oracle_routes=oracle_routes,
        credential_path=credential_path,
        certificate=certificate,
        output_path=positive_path,
        cycle_id="E025-SUB-LAYER-POSITIVE",
    )
    disabled_worker = fragments[0]
    negative_session = "e025-negative-control"
    inputs, _ = _stage_fixtures(
        checkpoint.expanduser().resolve(),
        oracle_trace.expanduser().resolve(),
        layer=SUB_LAYER_TARGET,
    )
    negative: dict[str, Any] = {}
    with (
        CanaryConnection(
            disabled_worker,
            credential_path.read_bytes(),
            certificate,
        ) as fragment_client,
        CanaryConnection(
            parent,
            credential_path.read_bytes(),
            certificate,
        ) as parent_client,
    ):
        _, disabled, _, _ = fragment_client.request(Action.DISABLE_EXECUTION, {})
        _, opened, _, _ = parent_client.request(
            Action.OPEN_SESSION,
            {"session_id": negative_session, "maximum_context": 3},
        )
        message_type, failure, _, _ = parent_client.request(
            Action.EXECUTE_STAGE,
            {
                "session_id": negative_session,
                "position": 0,
                "layer": SUB_LAYER_TARGET,
            },
            {"boundary": np.ascontiguousarray(inputs[0], dtype=np.float32)},
            allow_error=True,
        )
        _, negative_closed, _, _ = parent_client.request(
            Action.CLOSE_SESSION,
            {"session_id": negative_session},
        )
        _, enabled, _, _ = fragment_client.request(Action.ENABLE_EXECUTION, {})
        negative = {
            "disabled_worker_id": disabled_worker.worker_id,
            "disabled_machine_id": disabled_worker.machine_id,
            "disable_acknowledged": disabled.get("execution_enabled") is False,
            "frozen_placement_execute_message_type": message_type.name,
            "frozen_placement_execution_failed": message_type is MessageType.ERROR,
            "failure": failure,
            "explicit_replan_performed": False,
            "enable_acknowledged": enabled.get("execution_enabled") is True,
            "negative_session_opened": opened.get("opened") is True,
            "negative_session_closed": int(negative_closed["released_kv_bytes"]) > 0,
        }
    placement = read_json(physical_placement)
    complete_peak = int(placement["sub_layer_proof"]["complete_layer_runtime_peak_bytes"])
    memory_rows: list[dict[str, Any]] = []
    for worker in fragments:
        actual_vram = int(worker.ready["gpu"]["vram_mib"]) * MIB
        usable = int(actual_vram * 0.9)
        tracked_fragment = int(worker.ready["executor"]["tracked_fragment_bytes"])
        load_memory = worker.ready["executor"]["memory_after_load"]
        measured_runtime_used = int(load_memory["total_bytes"]) - int(
            load_memory["free_bytes"]
        )
        reconciled_fragment_peak = (
            measured_runtime_used + 256 * MIB + 64 * MIB + 128 * MIB
        )
        memory_rows.append(
            {
                "worker_id": worker.worker_id,
                "machine_id": worker.machine_id,
                "gpu_name": worker.ready["gpu"]["gpu_name"],
                "gpu_uuid": worker.ready["gpu"]["gpu_uuid"],
                "physical_vram_bytes": actual_vram,
                "physical_vram_gib": actual_vram / GIB,
                "safety_fraction": 0.10,
                "usable_vram_bytes": usable,
                "complete_layer_peak_bytes": complete_peak,
                "complete_layer_peak_gib": complete_peak / GIB,
                "assigned_fragment_tracked_bytes": tracked_fragment,
                "assigned_fragment_tracked_gib": tracked_fragment / GIB,
                "measured_runtime_used_after_load_bytes": measured_runtime_used,
                "runtime_reserve_bytes": 448 * MIB,
                "reconciled_fragment_peak_bytes": reconciled_fragment_peak,
                "reconciled_fragment_peak_gib": reconciled_fragment_peak / GIB,
                "complete_layer_fits": complete_peak <= usable,
                "fragment_fits": reconciled_fragment_peak <= usable,
                "assigned_expert_count": worker.ready["assignment"][
                    "owned_expert_count"
                ],
                "assigned_expert_ids": worker.ready["assignment"]["owned_expert_ids"],
                "native_primitive": worker.ready["executor"]["native_primitive"],
                "prepare_execution": worker.ready["prepare"]["execution"],
            }
        )
    positive_dispatch = [
        row["execution"]["external_expert_dispatch"] for row in positive["rows"]
    ]
    gates = {
        "positive_numerical_canary": positive["status"] == "PASS",
        "four_physical_fragment_workers": len(memory_rows) == 4,
        "four_distinct_physical_machines": len(
            {row["machine_id"] for row in memory_rows}
        )
        == 4,
        "complete_layer_cannot_fit_each": all(
            row["complete_layer_fits"] is False for row in memory_rows
        ),
        "each_fragment_fits": all(row["fragment_fits"] is True for row in memory_rows),
        "all_workers_invoked_every_positive_step": all(
            int(row["workers_invoked"]) == 4 for row in positive_dispatch
        ),
        "all_selected_experts_exactly_once": all(
            row["every_selected_expert_executed_once"] is True
            for row in positive_dispatch
        ),
        "real_native_fragment_calls": all(
            int(row["native_expert_calls"]) == 16 for row in positive_dispatch
        ),
        "every_fragment_executes_real_selected_experts": all(
            sum(
                int(worker_row["native_expert_calls"])
                for dispatch in positive_dispatch
                for worker_row in dispatch["workers"]
                if worker_row["worker_id"] == fragment.worker_id
            )
            > 0
            for fragment in fragments
        ),
        "no_monolithic_fallback": all(
            row["whole_layer_fallback"] is False for row in positive_dispatch
        ),
        "negative_control_fails_without_worker": negative[
            "frozen_placement_execution_failed"
        ],
        "negative_control_required_no_replan": negative["explicit_replan_performed"]
        is False,
        "disabled_worker_restored": negative["enable_acknowledged"],
    }
    payload = {
        "schema_version": "experiment-025-physical-sub-layer-canary-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class": "PHYSICAL_MULTI_MACHINE_CONSUMER_GPU_SUB_LAYER_CANARY",
        "layer": SUB_LAYER_TARGET,
        "parent": parent.endpoint(),
        "fragments": [worker.endpoint() for worker in fragments],
        "memory_proof": memory_rows,
        "positive": positive,
        "negative_control": negative,
        "gates": gates,
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = [
    "CanaryConnection",
    "run_physical_stage_fixture",
    "run_physical_sub_layer_canary",
]
