"""Production-native physical validation of the D local-fusion reduction."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_022.manifest_correctness import (
    ManifestK3Runner,
)
from swarm_inference.experiments.experiment_022.resident_replay import (
    _ResidentGroupedGraph,
    _residual_rows,
    _trace_rows,
)

from .correctness import ModelInvalidError
from .freeze import HIDDEN, PHYSICAL_RELATIVE_L2_MAX

LAYERS = 93
ROUTED_EXPERTS = 896


def _relative_l2(reference: np.ndarray, actual: np.ndarray) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    actual64 = np.asarray(actual, dtype=np.float64)
    denominator = float(np.linalg.norm(reference64.ravel()))
    numerator = float(np.linalg.norm((actual64 - reference64).ravel()))
    return numerator / denominator if denominator else numerator


class DFusedResidentMixedLayerGraph(_ResidentGroupedGraph):
    """Fuse routed-first/shared-second worker partials before one reduction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._d_routed_partials: tuple[np.ndarray, ...] | None = None
        self._d_shared_partials: tuple[np.ndarray, ...] | None = None
        self.d_fusion_audit: list[dict[str, Any]] = []

    def _reduce_partials(
        self,
        partials: list[np.ndarray] | tuple[np.ndarray, ...],
        *,
        layer: int | str,
        operator: str,
    ) -> np.ndarray:
        if operator == "latent_up_projection_stripe":
            self._d_routed_partials = tuple(
                np.ascontiguousarray(value, dtype=np.float32) for value in partials
            )
            return self._d_routed_partials[0]
        if operator == "shared_expert_stripe":
            self._d_shared_partials = tuple(
                np.ascontiguousarray(value, dtype=np.float32) for value in partials
            )
            return self._d_shared_partials[0]
        if operator != "routed_shared":
            return super()._reduce_partials(
                partials,
                layer=layer,
                operator=operator,
            )
        if self._d_routed_partials is None or self._d_shared_partials is None:
            raise RuntimeError("D fusion reached reduction without worker partials")
        if len(self._d_routed_partials) != self.degree or len(
            self._d_shared_partials
        ) != self.degree:
            raise RuntimeError("D fusion did not retain exactly eight worker pairs")
        fused: list[np.ndarray] = []
        for worker_index, (routed, shared) in enumerate(
            zip(self._d_routed_partials, self._d_shared_partials, strict=True)
        ):
            operation = (
                f"d_worker_{worker_index:02d}_routed_first_shared_second_fusion"
            )
            value = self._native_boundary_merge(
                [routed, shared],
                layer=int(layer),
                operator=operation,
            )
            fused.append(value)
            self.d_fusion_audit.append(
                {
                    "layer": int(layer),
                    "worker_index": worker_index,
                    "operation": operation,
                    "input_order": ["routed", "shared"],
                    "finite": bool(np.isfinite(value).all()),
                }
            )
        result = super()._reduce_partials(
            fused,
            layer=layer,
            operator="d_canonical_worker_0_to_7",
        )
        self._d_routed_partials = None
        self._d_shared_partials = None
        return result


class DManifestK3Runner(ManifestK3Runner):
    """Manifest runner whose FULL_MIXED layers use physical D semantics."""

    def _prepare_full_mixed_layer(
        self,
        *,
        layer: int,
        values: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
    ) -> dict[str, Any]:
        if self.shard_library is None:
            raise ValueError("FULL_MIXED_STRIPE requires the native shard library")
        if layer in self._prepared_full_mixed:
            raise RuntimeError("full-mixed layer was prepared more than once")
        assignment = self.assignments[layer]
        started = time.perf_counter_ns()
        graph = DFusedResidentMixedLayerGraph(
            self.checkpoint,
            self.cuda_library,
            self.shard_library,
            self.grouped_library,
            degree=int(assignment["degree"]),
            capture_state_arrays=True,
            shutdown_runtime_on_close=False,
        )
        try:
            graph.prepare_full_expert_banks(layer)
            graph.execute_layer_rows(
                layer,
                values,
                self._attnres_list(block_residuals, block_count),
            )
            checkpoint_reads = len(graph.loader.audit)
            graph.runtime.begin_replay()
            state_reset_ms = graph.runtime.reset_persistent_states()
            graph.loader.begin_replay()
            graph.quantizer.begin_replay()
            graph.worker_operations.clear()
            graph.reduction_records.clear()
            graph.d_fusion_audit.clear()
            self._prepared_full_mixed[layer] = graph
            return {
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
                "checkpoint_reads": checkpoint_reads,
                "weight_uploads": graph.runtime.weight_uploads,
                "buffer_allocations": graph.runtime.allocations,
                "state_reset_ms": state_reset_ms,
                "resident_expert_banks": len(graph._full_expert_banks),
                "experts_per_bank": (
                    min(len(bank.handles) for bank in graph._full_expert_banks.values())
                    if graph._full_expert_banks
                    else 0
                ),
                "d_semantics": "ROUTED_FIRST_SHARED_SECOND_LOCAL_FUSION_THEN_WORKER_0_TO_7_REDUCTION",
            }
        except BaseException:
            graph.close()
            raise

    def _execute_full_mixed_layer(self, **kwargs: Any) -> tuple[np.ndarray, int, dict[str, Any]]:
        layer = int(kwargs["layer"])
        value = super()._execute_full_mixed_layer(**kwargs)
        graph = self._prepared_full_mixed[layer]
        if not isinstance(graph, DFusedResidentMixedLayerGraph):
            raise RuntimeError("D manifest execution lost its fused graph")
        receipt = self.full_mixed_worker_receipts[-1]
        receipt["d_local_fusion_audit"] = list(graph.d_fusion_audit)
        receipt["d_canonical_reduction_order"] = list(range(8))
        receipt["d_transformation_applied"] = True
        return value


def _execute_graph_once(
    graph: _ResidentGroupedGraph,
    *,
    layer: int,
    hidden: np.ndarray,
    residuals: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any], int]:
    graph.prepare_full_expert_banks(layer)
    graph.execute_layer_rows(layer, hidden, residuals)
    reads_before = len(graph.loader.audit)
    graph.runtime.begin_replay()
    graph.runtime.reset_persistent_states()
    graph.loader.begin_replay()
    graph.quantizer.begin_replay()
    graph.worker_operations.clear()
    graph.reduction_records.clear()
    if isinstance(graph, DFusedResidentMixedLayerGraph):
        graph.d_fusion_audit.clear()
    output, next_residuals, record = graph.execute_layer_rows(
        layer,
        hidden,
        residuals,
    )
    timed_reads = len(graph.loader.audit) - reads_before
    return output, next_residuals, record, timed_reads


def validate_physical_d(
    *,
    checkpoint: Path,
    cuda_library: Path,
    shard_library: Path,
    grouped_library: Path,
    oracle_root: Path,
) -> dict[str, Any]:
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    cases: list[dict[str, Any]] = []
    for layer in (89, 91):
        for rows in (1, 2, 4):
            current = _ResidentGroupedGraph(
                checkpoint,
                cuda_library,
                shard_library,
                grouped_library,
                degree=8,
                capture_state_arrays=True,
            )
            fused = DFusedResidentMixedLayerGraph(
                checkpoint,
                cuda_library,
                shard_library,
                grouped_library,
                degree=8,
                capture_state_arrays=True,
            )
            try:
                hidden = _trace_rows(trace, layer - 1, rows)
                token_ids = (163584, 18699, 11)
                embeddings = np.ascontiguousarray(
                    np.stack(
                        [
                            current.embedding(token_ids[row % len(token_ids)])[0][0]
                            for row in range(rows)
                        ]
                    ),
                    dtype=np.float32,
                )
                residuals = _residual_rows(trace, layer, embeddings)
                current_output, current_residuals, current_record, current_reads = (
                    _execute_graph_once(
                        current,
                        layer=layer,
                        hidden=hidden,
                        residuals=residuals,
                    )
                )
                fused_output, fused_residuals, fused_record, fused_reads = (
                    _execute_graph_once(
                        fused,
                        layer=layer,
                        hidden=hidden,
                        residuals=residuals,
                    )
                )
                current_routes = current_record["routes"]
                fused_routes = fused_record["routes"]
                state_current = current_record["attention"]["_state_arrays"]
                state_fused = fused_record["attention"]["_state_arrays"]
                state_errors = {
                    name: _relative_l2(state_current[name], state_fused[name])
                    for name in state_current
                }
                residual_error = max(
                    (
                        _relative_l2(reference, actual)
                        for reference, actual in zip(
                            current_residuals,
                            fused_residuals,
                            strict=True,
                        )
                    ),
                    default=0.0,
                )
                fusion_order = [
                    int(row["worker_index"]) for row in fused.d_fusion_audit
                ]
                output_error = _relative_l2(current_output, fused_output)
                passed = (
                    output_error <= PHYSICAL_RELATIVE_L2_MAX
                    and max(state_errors.values(), default=0.0)
                    <= PHYSICAL_RELATIVE_L2_MAX
                    and residual_error <= PHYSICAL_RELATIVE_L2_MAX
                    and current_routes == fused_routes
                    and current_reads == 0
                    and fused_reads == 0
                    and fusion_order == list(range(8))
                    and np.isfinite(fused_output).all()
                )
                cases.append(
                    {
                        "layer": layer,
                        "rows": rows,
                        "status": "PASS" if passed else "FAIL",
                        "output_relative_l2": output_error,
                        "state_relative_l2_maximum": max(
                            state_errors.values(), default=0.0
                        ),
                        "attnres_relative_l2_maximum": residual_error,
                        "route_ids_exact": current_routes["selected_expert_ids"]
                        == fused_routes["selected_expert_ids"],
                        "route_weights_exact": current_routes["selected_weights"]
                        == fused_routes["selected_weights"],
                        "current_timed_checkpoint_reads": current_reads,
                        "d_timed_checkpoint_reads": fused_reads,
                        "d_local_fusion_worker_order": fusion_order,
                        "d_canonical_reduction_order": list(range(8)),
                        "production_native": True,
                        "finite_output": bool(np.isfinite(fused_output).all()),
                    }
                )
            finally:
                fused.close()
                current.close()
    status = "PASS" if all(row["status"] == "PASS" for row in cases) else "FAIL"
    payload = {
        "schema_version": "experiment-024-physical-d-correctness-v1",
        "status": status,
        "evidence_class": "PHYSICAL sequential logical workers on one RTX 5090",
        "candidate_type": "FULL_MIXED_STRIPE",
        "degree": 8,
        "layers": [89, 91],
        "rows": [1, 2, 4],
        "relative_l2_gate": PHYSICAL_RELATIVE_L2_MAX,
        "canonical_reduction_order": list(range(8)),
        "cases": cases,
    }
    if status != "PASS" or any(
        not math.isfinite(float(row["output_relative_l2"])) for row in cases
    ):
        raise ModelInvalidError("physical D correctness failed")
    return payload


__all__ = [
    "DFusedResidentMixedLayerGraph",
    "DManifestK3Runner",
    "validate_physical_d",
]
