"""Targeted tensor-parallel admission surfaces from real Kimi CUDA timings."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.evidence import (
    atomic_json,
    file_identity,
    read_json,
)


def _collective_ms(
    payload_bytes: int,
    degree: int,
    *,
    rtt_ms: float,
    bandwidth_gbps: float,
    software_base_ms: float,
) -> float:
    if degree == 1:
        return 0.0
    rounds = math.ceil(math.log2(degree))
    ring_volume_factor = 2.0 * (degree - 1) / degree
    return (
        software_base_ms
        + rounds * rtt_ms / 2.0
        + payload_bytes
        * ring_volume_factor
        * 8.0
        / (bandwidth_gbps * 1_000_000.0)
    )


def analyze_targeted_tp(
    repository_root: Path,
    output_path: Path,
    *,
    rtt_ms: float = 0.25,
    bandwidth_gbps: float = 25.0,
    software_base_ms: float = 0.02,
) -> dict[str, Any]:
    """Calculate complete-operation TP bounds without pretending they ran distributed."""
    root = repository_root.expanduser().resolve()
    cuda_root = root / "artifacts" / "experiment-014" / "cuda"
    kda_path = cuda_root / "h014-038-regression-kda-stage.json"
    mla_path = cuda_root / "h014-038-regression-mla-stage.json"
    head_path = cuda_root / "h014-038-regression-lm-head.json"
    kda = read_json(kda_path)
    mla = read_json(mla_path)
    head = read_json(head_path)
    if any(document.get("status") != "PASS" for document in (kda, mla, head)):
        raise ValueError("targeted TP source timing did not pass its real CUDA gate")

    operations = [
        {
            "operation": "KDA q projection",
            "measured_tp1_ms": float(
                kda["benchmark"]["individual_projection_device_ms_per_call"]["q"]
            ),
            "collective": "all-gather projected rows",
            "collective_payload_bytes": 12_288 * 4,
        },
        {
            "operation": "KDA output projection",
            "measured_tp1_ms": float(
                kda["benchmark"]["individual_projection_device_ms_per_call"]["output"]
            ),
            "collective": "all-reduce hidden row",
            "collective_payload_bytes": 7_168 * 4,
        },
        {
            "operation": "MLA five-projection group",
            "measured_tp1_ms": float(
                mla["benchmark"]["phase_device_ms_per_call"][
                    "five_projection_and_query_norm_kernels"
                ]
            ),
            "collective": "lower-bound all-reduce hidden row",
            "collective_payload_bytes": 7_168 * 4,
        },
        {
            "operation": "LM head",
            "measured_tp1_ms": float(head["benchmark"]["queued_service_ms_per_call"]),
            "collective": "all-gather FP32 vocabulary logits",
            "collective_payload_bytes": 163_840 * 4,
        },
    ]

    rows: list[dict[str, Any]] = []
    for operation in operations:
        for degree in (1, 2, 4):
            local_compute = float(operation["measured_tp1_ms"]) / degree
            collective = _collective_ms(
                int(operation["collective_payload_bytes"]),
                degree,
                rtt_ms=rtt_ms,
                bandwidth_gbps=bandwidth_gbps,
                software_base_ms=software_base_ms,
            )
            total = local_compute + collective
            rows.append(
                {
                    "evidence_class": "SHAPED" if degree > 1 else "MEASURED",
                    **operation,
                    "degree": degree,
                    "ideal_local_compute_ms": local_compute,
                    "shaped_collective_ms": collective,
                    "operation_latency_ms": total,
                    "operation_speedup": float(operation["measured_tp1_ms"]) / total,
                    "full_stage_measured": degree == 1,
                    "admitted": degree == 1,
                }
            )

    best_by_operation: list[dict[str, Any]] = []
    for operation in operations:
        candidates = [row for row in rows if row["operation"] == operation["operation"]]
        best_by_operation.append(min(candidates, key=lambda row: row["operation_latency_ms"]))

    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-targeted-tp-v1",
        "cycle_id": "H015-007A",
        "status": "PASS",
        "evidence_class": "SHAPED",
        "network": {
            "domain": "internal_microwork",
            "rtt_ms": rtt_ms,
            "bandwidth_gbps": bandwidth_gbps,
            "software_base_ms": software_base_ms,
        },
        "method": (
            "real Kimi TP1 CUDA operation time divided by degree plus a shaped "
            "collective; no physical or same-GPU sharded kernel is claimed"
        ),
        "rows": rows,
        "best_degree_by_operation": best_by_operation,
        "useful_operation_sensitivity": [
            item["operation"]
            for item in best_by_operation
            if int(item["degree"]) > 1 and float(item["operation_speedup"]) > 1.0
        ],
        "retained_operations": [],
        "decision": "STOP_NO_COMPLETE_STAGE_TP_IMPLEMENTATION",
        "sources": {
            "kda": file_identity(kda_path, relative_to=root),
            "mla": file_identity(mla_path, relative_to=root),
            "lm_head": file_identity(head_path, relative_to=root),
        },
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = ["analyze_targeted_tp"]
