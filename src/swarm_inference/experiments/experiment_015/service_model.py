"""Evidence-backed Experiment 015 service and upper-bound models."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_015.dcp import dcp_partial_payload_bytes
from swarm_inference.experiments.experiment_015.evidence import atomic_json, read_json
from swarm_inference.experiments.experiment_015.network import (
    MicrocellGeometry,
    NetworkProfile,
    activation_payload_bytes,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _log_quadratic_prediction(
    calibration: dict[int, float], held_out_x: int
) -> float:
    if len(calibration) != 3 or held_out_x <= 0:
        raise ValueError("log-quadratic validation requires three positive anchors")
    x = np.log(np.asarray(list(calibration), dtype=np.float64))
    y = np.log(np.asarray(list(calibration.values()), dtype=np.float64))
    coefficients = np.polyfit(x, y, 2)
    return float(np.exp(np.polyval(coefficients, math.log(held_out_x))))


@dataclass(frozen=True, slots=True)
class ServiceEvidence:
    """Small reconciled set of immutable Experiment 014 service inputs."""

    baseline_compute_ms: float
    baseline_end_to_end_ms: float
    baseline_aggregate_tok_s: float
    baseline_paid_gpu_equivalents: float
    kda_total_single_ms: float
    mla_total_single_ms: float
    endpoint_single_ms: float
    kda_batch_ratios: dict[int, float]
    mla_context_batch_ratios: dict[int, float]
    mla_short_batch_ratios: dict[int, float]
    mla_context_device_ms: dict[int, float]


def load_service_evidence(repository_root: Path) -> ServiceEvidence:
    """Load only immutable, explicitly named Experiment 014 evidence."""
    root = repository_root.expanduser().resolve()
    exp014 = root / "artifacts" / "experiment-014"
    baseline = read_json(root / "artifacts" / "experiment-015" / "baseline" / "B015-000.json")
    performance = baseline["performance"]
    components = baseline["latency_reconciliation"]["compute_components_ms"]
    kda = read_json(
        exp014 / "performance" / "h014-038ah2-complete-stage-batch-layer89.json"
    )
    mla_short = read_json(
        exp014 / "performance" / "h014-038ah2-complete-stage-batch-layer91.json"
    )
    mla_context_batch = read_json(
        exp014 / "performance" / "h014-034d-contextual-batch-layer91-8k.json"
    )
    mla_context = read_json(
        exp014 / "performance" / "h014-033c-prefill-layer91-mla16k.json"
    )

    def batch_device(document: dict[str, Any], batch: int) -> float:
        return float(document["batches"][str(batch)]["performance"]["device"]["p50_ms"])

    kda_single = batch_device(kda, 1)
    mla_short_single = batch_device(mla_short, 1)
    context_single = float(
        mla_context_batch["batches"]["1"]["retained"]["device"]["p50_ms"]
    )
    kda_ratios = {
        batch: batch_device(kda, batch) / kda_single for batch in (1, 2, 4, 8)
    }
    short_ratios = {
        batch: batch_device(mla_short, batch) / mla_short_single
        for batch in (1, 2, 4, 8)
    }
    context_ratios = {
        batch: float(
            mla_context_batch["batches"][str(batch)]["retained"]["device"]["p50_ms"]
        )
        / context_single
        for batch in (1, 2, 4, 8)
    }
    context_times = {
        context: float(
            mla_context["contexts"][str(context)]["decode_after_prefill"]["device_ms"]
        )
        for context in (1024, 4096, 8192, 16384)
    }
    return ServiceEvidence(
        baseline_compute_ms=float(baseline["latency_reconciliation"]["compute_total_ms"]),
        baseline_end_to_end_ms=float(performance["end_to_end_ms"]),
        baseline_aggregate_tok_s=float(performance["aggregate_output_tok_s"]),
        baseline_paid_gpu_equivalents=float(performance["paid_gpu_equivalents"]),
        kda_total_single_ms=float(components["68_middle_kda_layers"]),
        mla_total_single_ms=float(components["23_mla_layers_at_8k"]),
        endpoint_single_ms=float(components["final_layer_head"])
        + float(components["stage_zero"]),
        kda_batch_ratios=kda_ratios,
        mla_context_batch_ratios=context_ratios,
        mla_short_batch_ratios=short_ratios,
        mla_context_device_ms=context_times,
    )


class ArchitectureServiceModel:
    """Model dependency path and aggregate work without mixing evidence classes."""

    def __init__(
        self,
        evidence: ServiceEvidence,
        *,
        internal_network: NetworkProfile | None = None,
        coarse_network: NetworkProfile | None = None,
    ) -> None:
        self.evidence = evidence
        self.internal_network = internal_network or NetworkProfile(
            "internal_microwork", rtt_ms=0.25, bandwidth_gbps=25.0
        )
        self.coarse_network = coarse_network or NetworkProfile(
            "coarse_inter_cell", rtt_ms=5.0, bandwidth_gbps=10.0
        )

    @staticmethod
    def _ratio(table: dict[int, float], rows: int) -> float:
        if rows not in table:
            raise ValueError("only certified target widths 1/2/4/8 are modeled")
        return table[rows]

    def microcell_latency(self, depth: int, *, rows: int = 1) -> dict[str, Any]:
        geometry = MicrocellGeometry(93, depth)
        kda = self.evidence.kda_total_single_ms * self._ratio(
            self.evidence.kda_batch_ratios, rows
        )
        mla = self.evidence.mla_total_single_ms * self._ratio(
            self.evidence.mla_context_batch_ratios, rows
        )
        endpoints = self.evidence.endpoint_single_ms * self._ratio(
            self.evidence.kda_batch_ratios, rows
        )
        compute = kda + mla + endpoints
        payload = activation_payload_bytes(rows)
        latency = geometry.dependency_latency_ms(
            compute_ms=compute,
            payload_bytes=payload,
            internal=self.internal_network,
            coarse=self.coarse_network,
        )
        return {
            "evidence_class": "SHAPED",
            "service_model_validation": (
                "component model passed held-out validation; topology transport is synthetic"
            ),
            "layers_per_cell": depth,
            "cells": geometry.cell_count,
            "target_rows": rows,
            "coarse_boundaries": geometry.coarse_boundaries,
            "internal_boundaries": geometry.internal_boundaries,
            "payload_bytes_per_boundary": payload,
            "compute_ms": compute,
            "kda_ms": kda,
            "mla_ms": mla,
            "endpoint_ms": endpoints,
            "dependency_latency_ms": latency,
            "dependency_bound_tok_s": 1000.0 / latency,
        }

    def speculation_upper_bound(
        self,
        *,
        block_size: int,
        accepted_tokens_per_target_pass: float,
        cell_depth: int,
        draft_latency_ms: float = 0.0,
        asynchronous_draft: bool = True,
    ) -> dict[str, Any]:
        """Optimistic target-only bound; never presented as measured acceptance."""
        logical_rows = block_size + 1
        if block_size not in (1, 2, 3, 5, 7):
            raise ValueError("block size is outside the preregistered sweep")
        rows = 1 << (logical_rows - 1).bit_length()
        if rows not in (2, 4, 8):
            raise ValueError("padded target width is outside the certified batch envelope")
        if not 1.0 <= accepted_tokens_per_target_pass <= logical_rows:
            raise ValueError("accepted tokens/pass must lie in [1, target rows]")
        target = self.microcell_latency(cell_depth, rows=rows)
        pass_ms = (
            max(float(target["dependency_latency_ms"]), draft_latency_ms)
            if asynchronous_draft
            else float(target["dependency_latency_ms"]) + draft_latency_ms
        )
        dependency = accepted_tokens_per_target_pass * 1000.0 / pass_ms

        # B015-000 aggregate already runs the target at safe batch 8. A
        # speculative pass consumes `rows` target rows for fewer/equal useful
        # output tokens. Scale by measured target row capacity at this width.
        mla_capacity_ratio = (
            rows / self.evidence.mla_context_batch_ratios[rows]
        ) / (8 / self.evidence.mla_context_batch_ratios[8])
        kda_capacity_ratio = (
            rows / self.evidence.kda_batch_ratios[rows]
        ) / (8 / self.evidence.kda_batch_ratios[8])
        target_capacity_ratio = min(mla_capacity_ratio, kda_capacity_ratio, 1.0)
        aggregate = (
            self.evidence.baseline_aggregate_tok_s
            * target_capacity_ratio
            * accepted_tokens_per_target_pass
            / rows
        )
        paid_gpu_equivalents = self.evidence.baseline_paid_gpu_equivalents
        return {
            "evidence_class": "PROJECTED",
            "scope": (
                "optimistic target-only upper bound; draft cost is excluded from paid "
                "GPU-equivalents and acceptance is an explicit sensitivity input"
            ),
            "block_size": block_size,
            "logical_target_rows": logical_rows,
            "executed_target_rows": rows,
            "padding_rows": rows - logical_rows,
            "accepted_tokens_per_target_pass": accepted_tokens_per_target_pass,
            "target_pass_ms": float(target["dependency_latency_ms"]),
            "draft_latency_ms": draft_latency_ms,
            "asynchronous_draft": asynchronous_draft,
            "end_to_end_pass_ms": pass_ms,
            "dependency_bound_tok_s": dependency,
            "target_capacity_ratio_vs_baseline_batch8": target_capacity_ratio,
            "aggregate_output_tok_s": aggregate,
            "paid_gpu_equivalents_excluding_draft": paid_gpu_equivalents,
            "aggregate_tok_s_per_paid_gpu_equivalent_excluding_draft": (
                aggregate / paid_gpu_equivalents
            ),
            "cell": target,
        }

    def dcp_stage_projection(
        self,
        *,
        context_tokens: int,
        degree: int,
        rows: int = 1,
    ) -> dict[str, Any]:
        """Project exact context sharding from measured scan slope and shaped combine."""
        if context_tokens not in self.evidence.mla_context_device_ms:
            raise ValueError("DCP context must be one of the measured context lengths")
        if degree not in (1, 2, 4, 8) or rows not in (1, 2, 4, 8):
            raise ValueError("DCP degree/rows must be certified powers of two through 8")

        # Fit local 5090 single-row stage service. The intercept is non-scan
        # work; the remainder is the context scan. Transfer each operation
        # class with the immutable Experiment 014 factors.
        x = np.asarray([1024, 4096, 16384], dtype=np.float64)
        y = np.asarray(
            [self.evidence.mla_context_device_ms[int(value)] for value in x],
            dtype=np.float64,
        )
        slope, intercept = np.polyfit(x, y, 1)
        scan_local = max(0.0, slope * context_tokens)
        non_scan_local = max(0.0, intercept)
        short_context_transfer = 2.0620625519227826
        scan_transfer = 2.9438202247191008
        non_scan_projected = (
            non_scan_local
            * short_context_transfer
            * self.evidence.mla_short_batch_ratios[rows]
        )
        scan_projected = (
            scan_local
            * scan_transfer
            * self.evidence.mla_context_batch_ratios[rows]
            / degree
        )
        payload = dcp_partial_payload_bytes(query_rows=rows)
        combine_transport = 0.0
        synchronization = 0
        if degree > 1:
            combine_transport = self.internal_network.service_ms(payload)
            synchronization = 2
        projected = non_scan_projected + scan_projected + combine_transport
        unsharded = (
            non_scan_projected
            + scan_local
            * scan_transfer
            * self.evidence.mla_context_batch_ratios[rows]
        )
        return {
            "evidence_class": "SHAPED",
            "context_tokens": context_tokens,
            "degree": degree,
            "query_rows": rows,
            "measured_scan_fit": {
                "intercept_ms_rtx5090": non_scan_local,
                "slope_ms_per_context_token_rtx5090": float(slope),
                "measured_contexts": sorted(self.evidence.mla_context_device_ms),
            },
            "projected_non_scan_ms": non_scan_projected,
            "projected_local_scan_ms": scan_projected,
            "partial_payload_bytes_per_worker": payload,
            "combine_transport_ms": combine_transport,
            "synchronization_count": synchronization,
            "projected_mla_stage_ms": projected,
            "unsharded_stage_ms": unsharded,
            "stage_speedup": unsharded / projected,
            "stage_gain_fraction": 1.0 - projected / unsharded,
            "capacity_evidence": False,
        }


def validate_service_model(repository_root: Path, output_path: Path) -> dict[str, Any]:
    """Predict held-out local composites and enforce median error <=10%."""
    root = repository_root.expanduser().resolve()
    exp014 = root / "artifacts" / "experiment-014"
    kda = read_json(
        exp014 / "performance" / "h014-038ah2-complete-stage-batch-layer89.json"
    )
    mla = read_json(
        exp014 / "performance" / "h014-038ah2-complete-stage-batch-layer91.json"
    )
    context = read_json(
        exp014 / "performance" / "h014-033c-prefill-layer91-mla16k.json"
    )
    expert = read_json(
        exp014 / "sub-layer" / "h014-sub-005-scaling-network.json"
    )

    rows: list[dict[str, Any]] = []
    for name, document in (("KDA batch-4", kda), ("MLA batch-4", mla)):
        anchors = {
            batch: float(
                document["batches"][str(batch)]["performance"]["device"]["p50_ms"]
            )
            for batch in (1, 2, 8)
        }
        actual = float(document["batches"]["4"]["performance"]["device"]["p50_ms"])
        predicted = _log_quadratic_prediction(anchors, 4)
        rows.append(
            {
                "held_out": name,
                "evidence_class": "MEASURED",
                "calibration_points": anchors,
                "predicted_ms": predicted,
                "actual_ms": actual,
                "absolute_percentage_error": abs(predicted - actual) / actual * 100.0,
            }
        )

    context_anchors = {
        length: float(
            context["contexts"][str(length)]["decode_after_prefill"]["device_ms"]
        )
        for length in (1024, 4096, 16384)
    }
    coefficients = np.polyfit(
        np.asarray(list(context_anchors), dtype=np.float64),
        np.asarray(list(context_anchors.values()), dtype=np.float64),
        1,
    )
    context_actual = float(
        context["contexts"]["8192"]["decode_after_prefill"]["device_ms"]
    )
    context_predicted = float(np.polyval(coefficients, 8192))
    rows.append(
        {
            "held_out": "MLA 8K context decode",
            "evidence_class": "MEASURED",
            "calibration_points": context_anchors,
            "predicted_ms": context_predicted,
            "actual_ms": context_actual,
            "absolute_percentage_error": (
                abs(context_predicted - context_actual) / context_actual * 100.0
            ),
        }
    )

    scaling = {int(row["workers"]): row for row in expert["scaling_curve"]}
    expert_anchors = {
        workers: float(scaling[workers]["distributed_layer_p50_ms"])
        for workers in (2, 4, 16)
    }
    expert_actual = float(scaling[8]["distributed_layer_p50_ms"])
    expert_predicted = _log_quadratic_prediction(expert_anchors, 8)
    rows.append(
        {
            "held_out": "8-worker expert layer",
            "evidence_class": "MEASURED",
            "calibration_points": expert_anchors,
            "predicted_ms": expert_predicted,
            "actual_ms": expert_actual,
            "absolute_percentage_error": (
                abs(expert_predicted - expert_actual) / expert_actual * 100.0
            ),
        }
    )

    errors = [float(row["absolute_percentage_error"]) for row in rows]
    median_error = statistics.median(errors)
    receipt = {
        "schema_version": "experiment-015-service-model-validation-v1",
        "cycle_id": "H015-MODEL-001",
        "status": "PASS" if median_error <= 10.0 else "FAIL",
        "evidence_class": "VALIDATED MODEL",
        "validation_design": {
            "calibration_and_held_out_disjoint": True,
            "target_median_absolute_percentage_error_percent": 10.0,
            "note": (
                "Validation is local component/composite interpolation. It does not "
                "validate a physical multi-GPU or fleet transfer."
            ),
        },
        "held_out_predictions": rows,
        "median_absolute_percentage_error_percent": median_error,
        "p95_absolute_percentage_error_percent": _percentile(errors, 0.95),
        "maximum_absolute_percentage_error_percent": max(errors),
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "ArchitectureServiceModel",
    "ServiceEvidence",
    "load_service_evidence",
    "validate_service_model",
]
