"""Two-token physical autoregressive correctness for corrected E024."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_graph_runtime import KimiCudaGraphRunner
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_022.manifest_correctness import (
    _parse_oracle_routes,
)

from .correctness import ModelInvalidError
from .freeze import (
    CHECKPOINT,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    PHYSICAL_RELATIVE_L2_MAX,
)
from .physical_d import DManifestK3Runner
from .placement import CommodityPlacement, validate_commodity_architecture
from .service_calibration import (
    CUDA_LIBRARY_RELATIVE_PATH,
    GROUPED_LIBRARY_RELATIVE_PATH,
    ORACLE_ROOT_RELATIVE_PATH,
    SHARD_LIBRARY_RELATIVE_PATH,
    _isolated_request,
)

LAYERS = 93
HIDDEN = 7168
VOCABULARY = 163_840
MAXIMUM_CONTEXT = 8


def _relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    numerator = float(np.linalg.norm((actual64 - reference64).ravel()))
    denominator = float(np.linalg.norm(reference64.ravel()))
    return numerator / denominator if denominator else numerator


def _fingerprint(value: np.ndarray) -> str:
    source = np.ascontiguousarray(value, dtype=np.float32)
    return "sha256:" + hashlib.sha256(source.tobytes()).hexdigest()


class _CapturingReferenceRunner(KimiCudaGraphRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.captured_hidden_by_layer: dict[int, np.ndarray] = {}
        self.captured_attnres_by_layer: dict[int, np.ndarray] = {}
        self.captured_final_hidden: np.ndarray | None = None
        self.captured_logits: np.ndarray | None = None

    def begin_capture_step(self) -> None:
        self.captured_hidden_by_layer.clear()
        self.captured_attnres_by_layer.clear()
        self.captured_final_hidden = None
        self.captured_logits = None

    def execute_layer(self, layer: int, *args: Any, **kwargs: Any) -> Any:
        result = super().execute_layer(layer, *args, **kwargs)
        self.captured_hidden_by_layer[layer] = np.ascontiguousarray(
            result[0], dtype=np.float32
        )
        block_residuals = args[1]
        self.captured_attnres_by_layer[layer] = np.ascontiguousarray(
            block_residuals[:, : result[1]], dtype=np.float32
        )
        return result

    def execute_final_head(self, *args: Any, **kwargs: Any) -> Any:
        result = super().execute_final_head(*args, **kwargs)
        self.captured_final_hidden = np.ascontiguousarray(result[0], dtype=np.float32)
        self.captured_logits = np.ascontiguousarray(result[1], dtype=np.float32)
        return result


class _CapturingPersistentDRunner(DManifestK3Runner):
    """D runner with a resident production-native dense layer-0 executor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        request = _isolated_request(
            self.checkpoint,
            self.cuda_library,
            layer=0,
            maximum_context=MAXIMUM_CONTEXT,
        )
        self.dense_executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=self.checkpoint,
            cuda_library=self.cuda_library,
            device=0,
        )
        self.dense_executor.prepare_for_ready()
        self.dense_session_id = "e024-two-token-dense-layer0"
        self.dense_executor.open_session(
            self.dense_session_id,
            maximum_context_override=MAXIMUM_CONTEXT,
        )
        self.dense_layer0_receipts: list[dict[str, Any]] = []
        self.captured_hidden_by_layer: dict[int, np.ndarray] = {}
        self.captured_attnres_by_layer: dict[int, np.ndarray] = {}
        self.captured_final_hidden: np.ndarray | None = None
        self.captured_logits: np.ndarray | None = None

    def begin_capture_step(self) -> None:
        self.captured_hidden_by_layer.clear()
        self.captured_attnres_by_layer.clear()
        self.captured_final_hidden = None
        self.captured_logits = None

    def _dense_state_arrays(self) -> dict[str, np.ndarray]:
        session = self.dense_executor._require_session(self.dense_session_id)
        config = self.dense_executor.config
        if 0 in config.kda_layers:
            shapes = {
                "state": (
                    config.kda_heads,
                    config.kda_head_dimension,
                    config.kda_head_dimension,
                ),
                "window_q": (config.kda_projection, config.convolution_width),
                "window_k": (config.kda_projection, config.convolution_width),
                "window_v": (config.kda_projection, config.convolution_width),
            }
        else:
            shapes = {
                "latent_cache": (MAXIMUM_CONTEXT, config.kv_lora),
                "rope_cache": (MAXIMUM_CONTEXT, config.query_rope),
            }
        return {
            name: self.dense_executor.runtime.download_activation(
                session.attention_state[name], shape
            )
            for name, shape in shapes.items()
        }

    def _execute_dense_layer_zero(
        self,
        hidden_rows: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
        positions: list[int],
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        if hidden_rows.shape != (1, HIDDEN) or block_count != 0 or len(positions) != 1:
            raise ValueError("two-token dense layer-0 correctness requires one row")
        boundary = np.zeros((1, 9, HIDDEN), dtype=np.float32)
        boundary[0, 0] = hidden_rows[0]
        result = self.dense_executor.execute_decode(
            session_id=self.dense_session_id,
            hidden_states=torch.from_numpy(boundary),
            cache_position_start=int(positions[0]),
        )
        output_boundary = np.ascontiguousarray(
            result.stage_boundary_hidden_states.detach().cpu().numpy(),
            dtype=np.float32,
        )
        output = output_boundary[:, 0]
        block_residuals[:, 0] = output_boundary[:, 1]
        record = self.dense_executor.execution_records[-1]
        if (
            int(record["weight_loads_during_execute"]) != 0
            or int(record["materializations_during_execute"]) != 0
        ):
            raise RuntimeError("dense layer 0 performed a timed checkpoint/materialization")
        state_arrays = self._dense_state_arrays()
        self.states[0] = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in state_arrays.items()
        }
        state_evidence = self.dense_executor.session_state_evidence(
            self.dense_session_id
        )
        owner = str(self.assignments[0]["pieces"][0]["node_id"])
        self.logical_workers_instantiated.add(owner)
        receipt = {
            "step": len(self.dense_layer0_receipts) + 1,
            "layer": 0,
            "candidate_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
            "candidate_type": "WHOLE_LAYER",
            "degree": 1,
            "worker_id": owner,
            "backend_identity": "nvidia_cuda_persistent_kimi_stage",
            "production_native": True,
            "timed_checkpoint_reads": int(record["weight_loads_during_execute"]),
            "timed_materializations": int(
                record["materializations_during_execute"]
            ),
            "resident_weights": True,
        }
        self.dense_layer0_receipts.append(receipt)
        self.dispatch_receipts.append(
            {
                "layer": 0,
                "worker_id": owner,
                "assignment_id": LAYER_ZERO_WHOLE_CANDIDATE_ID,
                "task_type": "WHOLE_LAYER",
                "native_primitive": "persistent_kimi_dense_whole_layer",
                "worker_native_wall_ms": float(record["wall_ms"]),
                "worker_protocol_wall_ms": 0.0,
                "output_sha256": _fingerprint(output),
            }
        )
        return output, 1, {
            "layer": 0,
            "attention_type": str(state_evidence["attention_type"]),
            "mlp_type": "dense",
            "positions": positions,
            "input_fingerprint": _fingerprint(hidden_rows),
            "output_fingerprint": _fingerprint(output),
            "state_input_fingerprint": None,
            "state_output": state_evidence,
            "routes": [],
            "timing": {"load_ms": 0.0, "execute_ms": float(record["wall_ms"])},
            "memory": {
                "resident_weight_bytes": self.dense_executor.resident_device_bytes
            },
            "backend_identity": "nvidia_cuda_persistent_kimi_stage",
            "manifest_assignment": self.assignments[0],
            "authenticated_execute_shard": True,
            "native_primitive": "persistent_kimi_dense_whole_layer",
            "worker_protocol_wall_ms": 0.0,
            "attnres_state_fingerprint": _fingerprint(block_residuals[:, :1]),
        }

    def execute_layer(self, layer: int, *args: Any, **kwargs: Any) -> Any:
        if layer == 0:
            result = self._execute_dense_layer_zero(
                args[0],
                args[1],
                args[2],
                args[3],
            )
        else:
            result = super().execute_layer(layer, *args, **kwargs)
        self.captured_hidden_by_layer[layer] = np.ascontiguousarray(
            result[0], dtype=np.float32
        )
        block_residuals = args[1]
        self.captured_attnres_by_layer[layer] = np.ascontiguousarray(
            block_residuals[:, : result[1]], dtype=np.float32
        )
        return result

    def execute_final_head(self, *args: Any, **kwargs: Any) -> Any:
        result = super().execute_final_head(*args, **kwargs)
        self.captured_final_hidden = np.ascontiguousarray(result[0], dtype=np.float32)
        self.captured_logits = np.ascontiguousarray(result[1], dtype=np.float32)
        return result

    def close(self) -> None:
        try:
            self.dense_executor.close()
        finally:
            super().close()


@dataclass(frozen=True, slots=True)
class _StepSnapshot:
    token_input: int
    token_output: int
    result: dict[str, Any]
    hidden_by_layer: dict[int, np.ndarray]
    attnres_by_layer: dict[int, np.ndarray]
    states: dict[int, dict[str, np.ndarray]]
    final_hidden: np.ndarray
    logits: np.ndarray


def _snapshot(runner: Any, result: dict[str, Any], token_input: int) -> _StepSnapshot:
    if runner.captured_final_hidden is None or runner.captured_logits is None:
        raise RuntimeError("full correctness runner omitted endpoint arrays")
    return _StepSnapshot(
        token_input=token_input,
        token_output=int(result["sampled_token_id"]),
        result=result,
        hidden_by_layer={
            layer: value.copy() for layer, value in runner.captured_hidden_by_layer.items()
        },
        attnres_by_layer={
            layer: value.copy() for layer, value in runner.captured_attnres_by_layer.items()
        },
        states={
            layer: {name: value.copy() for name, value in arrays.items()}
            for layer, arrays in runner.states.items()
        },
        final_hidden=runner.captured_final_hidden.copy(),
        logits=runner.captured_logits.copy(),
    )


def _execute_two_steps(
    runner: Any,
    *,
    trace: np.ndarray,
    routes: dict[int, dict[int, list[int]]],
    logits: np.ndarray,
    label: str,
) -> tuple[_StepSnapshot, _StepSnapshot]:
    token = 163584
    snapshots: list[_StepSnapshot] = []
    for position in (0, 1):
        runner.begin_capture_step()
        result = runner.execute_pass(
            [token],
            [position],
            layer_limit=LAYERS,
            maximum_context=MAXIMUM_CONTEXT,
            oracle_trace=trace,
            oracle_layer_count=LAYERS,
            oracle_routes=routes,
            oracle_logits=logits,
            progress_label=f"e024-{label}-step-{position + 1}",
        )
        snapshots.append(_snapshot(runner, result, token))
        token = int(result["sampled_token_id"])
    return snapshots[0], snapshots[1]


def _compare_step(
    reference: _StepSnapshot,
    actual: _StepSnapshot,
    *,
    step: int,
) -> dict[str, Any]:
    hidden_errors = {
        layer: _relative_l2(reference.hidden_by_layer[layer], actual.hidden_by_layer[layer])
        for layer in range(LAYERS)
    }
    attnres_errors = {
        layer: _relative_l2(
            reference.attnres_by_layer[layer], actual.attnres_by_layer[layer]
        )
        for layer in range(LAYERS)
    }
    state_errors: dict[int, float] = {}
    state_types: dict[int, str] = {}
    for layer in range(LAYERS):
        reference_state = reference.states[layer]
        actual_state = actual.states[layer]
        if set(reference_state) != set(actual_state):
            raise ModelInvalidError(f"state component mismatch at layer {layer}")
        state_errors[layer] = max(
            _relative_l2(reference_state[name], actual_state[name])
            for name in reference_state
        )
        state_types[layer] = str(reference.result["layers"][layer]["attention_type"])
    route_ids_exact = True
    route_weights_exact = True
    for layer in range(1, LAYERS):
        expected_routes = reference.result["layers"][layer]["routes"]
        actual_routes = actual.result["layers"][layer]["routes"]
        route_ids_exact = route_ids_exact and [
            row["selected_expert_ids"] for row in expected_routes
        ] == [row["selected_expert_ids"] for row in actual_routes]
        route_weights_exact = route_weights_exact and [
            row["selected_weights"] for row in expected_routes
        ] == [row["selected_weights"] for row in actual_routes]
    final_hidden_error = _relative_l2(reference.final_hidden, actual.final_hidden)
    logit_error = _relative_l2(reference.logits, actual.logits)
    kda_errors = [
        value
        for layer, value in state_errors.items()
        if state_types[layer] == "KDA"
    ]
    mla_errors = [
        value
        for layer, value in state_errors.items()
        if state_types[layer] != "KDA"
    ]
    finite = all(
        np.isfinite(value).all()
        for value in (
            *actual.hidden_by_layer.values(),
            *actual.attnres_by_layer.values(),
            actual.final_hidden,
            actual.logits,
        )
    ) and all(
        np.isfinite(value).all()
        for arrays in actual.states.values()
        for value in arrays.values()
    )
    passed = (
        len(actual.hidden_by_layer) == LAYERS
        and len(actual.states) == LAYERS
        and route_ids_exact
        and route_weights_exact
        and max(hidden_errors.values()) <= PHYSICAL_RELATIVE_L2_MAX
        and max(attnres_errors.values()) <= PHYSICAL_RELATIVE_L2_MAX
        and max(state_errors.values()) <= PHYSICAL_RELATIVE_L2_MAX
        and final_hidden_error <= PHYSICAL_RELATIVE_L2_MAX
        and logit_error <= PHYSICAL_RELATIVE_L2_MAX
        and reference.token_output == actual.token_output
        and finite
        and bool(actual.result["routing_equality"])
    )
    return {
        "step": step,
        "status": "PASS" if passed else "FAIL",
        "input_token_id": actual.token_input,
        "output_token_id": actual.token_output,
        "reference_output_token_id": reference.token_output,
        "greedy_token_equality": actual.token_output == reference.token_output,
        "complete_93_layer_traversal": len(actual.hidden_by_layer) == LAYERS,
        "route_ids_exact": route_ids_exact,
        "route_weights_exact": route_weights_exact,
        "finite_states_hidden_and_logits": finite,
        "hidden_relative_l2_maximum": max(hidden_errors.values()),
        "final_hidden_relative_l2": final_hidden_error,
        "logit_relative_l2": logit_error,
        "kda_state_relative_l2_maximum": max(kda_errors, default=0.0),
        "mla_state_relative_l2_maximum": max(mla_errors, default=0.0),
        "attnres_relative_l2_maximum": max(attnres_errors.values()),
        "oracle_maximum_layer_relative_l2": float(
            actual.result["maximum_layer_relative_l2_error"]
        ),
        "hidden_fingerprints": {
            str(layer): _fingerprint(value)
            for layer, value in actual.hidden_by_layer.items()
        },
    }


def run_two_token_correctness(
    repo_root: Path,
    placement: CommodityPlacement,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    validate_commodity_architecture(placement)
    model_metadata = json.loads(
        (repo_root / "artifacts/experiment-022/model-metadata.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = placement.as_manifest(model_metadata)
    if (
        manifest["checkpoint_reconciliation"]["gap_bytes"] != 0
        or manifest["checkpoint_reconciliation"]["overlap_bytes"] != 0
    ):
        raise ModelInvalidError("correctness manifest lacks complete checkpoint ownership")
    cuda_library = (repo_root / CUDA_LIBRARY_RELATIVE_PATH).resolve()
    shard_library = (repo_root / SHARD_LIBRARY_RELATIVE_PATH).resolve()
    grouped_library = (repo_root / GROUPED_LIBRARY_RELATIVE_PATH).resolve()
    oracle_root = (repo_root / ORACLE_ROOT_RELATIVE_PATH).resolve()
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    oracle_logits = np.memmap(
        oracle_root / "prefill-logits.f32",
        mode="r",
        dtype="<f4",
        shape=(2, VOCABULARY),
    )
    routes = _parse_oracle_routes(oracle_root / "routes.txt")

    reference_runner = _CapturingReferenceRunner(CHECKPOINT, cuda_library)
    try:
        reference_steps = _execute_two_steps(
            reference_runner,
            trace=trace,
            routes=routes,
            logits=oracle_logits,
            label="reference",
        )
    finally:
        reference_runner.close()

    actual_runner = _CapturingPersistentDRunner(
        CHECKPOINT,
        cuda_library,
        grouped_library,
        manifest,
        shard_library=shard_library,
    )
    try:
        actual_steps = _execute_two_steps(
            actual_runner,
            trace=trace,
            routes=routes,
            logits=oracle_logits,
            label="commodity-d",
        )
        comparisons = [
            _compare_step(reference, actual, step=index + 1)
            for index, (reference, actual) in enumerate(
                zip(reference_steps, actual_steps, strict=True)
            )
        ]
        full_mixed_receipts = list(actual_runner.full_mixed_worker_receipts)
        dense_receipts = list(actual_runner.dense_layer0_receipts)
        d_order_valid = (
            len(full_mixed_receipts) == 2 * 92
            and all(
                row.get("d_transformation_applied") is True
                and row.get("d_canonical_reduction_order") == list(range(8))
                and [
                    int(value["worker_index"])
                    for value in row.get("d_local_fusion_audit", [])
                ]
                == list(range(8))
                for row in full_mixed_receipts
            )
        )
        no_hot_reads = (
            len(dense_receipts) == 2
            and all(int(row["timed_checkpoint_reads"]) == 0 for row in dense_receipts)
            and all(
                int(row.get("checkpoint_reads_in_timed_region", -1)) == 0
                for row in full_mixed_receipts
            )
        )
        architecture_valid = (
            placement.whole_layer_layer_ids == (0,)
            and placement.p8_layer_ids == tuple(range(1, 93))
            and all(
                row["candidate_id"] == LAYER_ZERO_WHOLE_CANDIDATE_ID
                and row["production_native"] is True
                for row in dense_receipts
            )
        )
        status = (
            "PASS"
            if all(row["status"] == "PASS" for row in comparisons)
            and architecture_valid
            and d_order_valid
            and no_hot_reads
            else "FAIL"
        )
        payload = {
            "schema_version": "experiment-024-two-token-correctness-v2",
            "status": status,
            "evidence_class": "PHYSICAL sequential logical workers on one RTX 5090",
            "scenario": placement.scenario.value,
            "placement_kind": placement.placement_kind,
            "available_node_budget": placement.available_node_budget,
            "placement_sha256": placement.placement_sha256,
            "complete_checkpoint_ownership": True,
            "layer_zero_candidate_id": placement.layer_zero_candidate_id,
            "layer_zero_execution_kind": "WHOLE_LAYER",
            "layer_zero_degree": 1,
            "whole_layer_transformer_layer_ids": [0],
            "whole_layer_transformer_layer_count": 1,
            "p8_transformer_layer_ids": list(range(1, 93)),
            "p8_transformer_layer_count": 92,
            "production_native_execution": architecture_valid,
            "no_timed_checkpoint_reads": no_hot_reads,
            "canonical_d_reduction_order": d_order_valid,
            "relative_l2_gate": PHYSICAL_RELATIVE_L2_MAX,
            "step_1_token_t1": actual_steps[0].token_output,
            "step_2_input_token_t1": actual_steps[1].token_input,
            "step_2_token_t2": actual_steps[1].token_output,
            "step_2_consumed_step_1_token": (
                actual_steps[1].token_input == actual_steps[0].token_output
            ),
            "steps": comparisons,
            "dense_layer0_receipts": dense_receipts,
            "full_mixed_receipt_count": len(full_mixed_receipts),
            "full_mixed_layers_per_step": 92,
        }
    finally:
        with suppress(BaseException):
            actual_runner.close()
    if payload["status"] != "PASS":
        raise ModelInvalidError("two-token full autoregressive correctness failed")
    return payload


__all__ = ["run_two_token_correctness"]
