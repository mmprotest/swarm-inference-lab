"""Immutable Experiment 014 capture for B015-000."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.contracts import (
    EconomicsConfig,
    Experiment015Error,
)
from swarm_inference.experiments.experiment_015.evidence import (
    atomic_json,
    file_identity,
    read_json,
)

BASELINE_ID = "B015-000"
EXPECTED_WORKERS = 93
EXPECTED_SAFE_BATCH = 8
EXPECTED_FIRST_REJECTED_BATCH = 9
EXPECTED_AGGREGATE_TOK_S = 97.15028433444262
EXPECTED_DEPENDENCY_TOK_S = 0.9502949393520926
EXPECTED_END_TO_END_MS = 1052.304877769628
EXPECTED_SUB_LAYER_WORKERS = 4
EXPECTED_SUB_LAYER_RELATIVE_THROUGHPUT = 0.7880896676852558
EXPECTED_SUB_LAYER_WORKER_BYTES = 3_931_060_224
EXPECTED_SMALLEST_WORKER_BYTES = 982_890_496


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise Experiment015Error(
            f"Experiment 014 immutable baseline changed: {name}={actual}, "
            f"expected {expected}"
        )


def capture_baseline(
    repository_root: Path,
    output_path: Path,
    *,
    economics: EconomicsConfig | None = None,
) -> dict[str, Any]:
    """Capture and reconcile immutable Experiment 014 evidence without editing it."""
    root = repository_root.expanduser().resolve()
    source_root = root / "artifacts" / "experiment-014"
    sources = {
        "machine_summary": source_root / "machine-summary.json",
        "acceptance_gates": source_root / "acceptance-gates.json",
        "evidence_integrity": source_root / "evidence-integrity.json",
        "capacity": source_root
        / "performance"
        / "h014-038ak-final-capacity-topology-economics.json",
        "coarse_network": source_root
        / "coarse"
        / "h014-038ah-network-analysis.json",
        "expert_microwork": source_root
        / "sub-layer"
        / "h014-sub-011-promoted-four-worker-real-expert.json",
        "expert_scaling": source_root
        / "sub-layer"
        / "h014-sub-005-scaling-network.json",
        "kda_batch": source_root
        / "performance"
        / "h014-038ah2-complete-stage-batch-layer89.json",
        "mla_batch": source_root
        / "performance"
        / "h014-038ah2-complete-stage-batch-layer91.json",
        "mla_context": source_root
        / "performance"
        / "h014-033c-prefill-layer91-mla16k.json",
        "routing": source_root
        / "sub-layer"
        / "h014-sub-010-routing-imbalance.json",
        "routing_calls": source_root
        / "sub-layer"
        / "h014-sub-010-routing-calls.csv",
        "cuda_binary": source_root
        / "cuda"
        / "native"
        / "coli_cuda-sm86-h014-038-final.dll",
    }
    for name, path in sources.items():
        if not path.is_file():
            raise Experiment015Error(f"missing immutable baseline source {name}: {path}")

    machine = read_json(sources["machine_summary"])
    gates = read_json(sources["acceptance_gates"])
    capacity = read_json(sources["capacity"])
    metrics = machine.get("metrics", {})
    if not isinstance(metrics, dict):
        raise Experiment015Error("Experiment 014 machine summary lacks metrics")
    if machine.get("status") != "PASS" or gates.get("status") != "PASS":
        raise Experiment015Error("Experiment 014 immutable baseline is not passing")
    if int(metrics.get("worker_count", -1)) != EXPECTED_WORKERS:
        raise Experiment015Error("Experiment 014 worker count changed")
    if int(metrics.get("safe_certified_batch", -1)) != EXPECTED_SAFE_BATCH:
        raise Experiment015Error("Experiment 014 safe batch changed")
    if int(metrics.get("first_rejected_batch", -1)) != EXPECTED_FIRST_REJECTED_BATCH:
        raise Experiment015Error("Experiment 014 first rejected batch changed")
    if int(metrics.get("best_sub_layer_workers", -1)) != EXPECTED_SUB_LAYER_WORKERS:
        raise Experiment015Error("Experiment 014 expert worker selection changed")

    aggregate = float(metrics["projected_aggregate_output_tokens_per_second"])
    dependency = float(metrics["projected_per_user_decode_tokens_per_second"])
    end_to_end = float(metrics["projected_end_to_end_ms"])
    _assert_close("aggregate_tok_s", aggregate, EXPECTED_AGGREGATE_TOK_S)
    _assert_close("dependency_tok_s", dependency, EXPECTED_DEPENDENCY_TOK_S)
    _assert_close("end_to_end_ms", end_to_end, EXPECTED_END_TO_END_MS)
    _assert_close(
        "sub_layer_relative_throughput",
        float(metrics["sub_layer_relative_throughput_percent"]) / 100.0,
        EXPECTED_SUB_LAYER_RELATIVE_THROUGHPUT,
    )
    if int(metrics["best_exact_sub_layer_worker_bytes"]) != EXPECTED_SUB_LAYER_WORKER_BYTES:
        raise Experiment015Error("Experiment 014 four-worker resident bytes changed")
    if int(metrics["smallest_tested_worker_bytes"]) != EXPECTED_SMALLEST_WORKER_BYTES:
        raise Experiment015Error("Experiment 014 smallest worker bytes changed")

    latency = capacity.get("decode_latency", {})
    compute = latency.get("compute_components_ms", {})
    edge_count = int(latency.get("edge_count", -1))
    edge_ms = float(latency.get("coarse_edge_service_ms", float("nan")))
    compute_ms = sum(float(value) for value in compute.values())
    reconstructed_ms = compute_ms + edge_count * edge_ms
    _assert_close("reconstructed_end_to_end_ms", reconstructed_ms, end_to_end)
    _assert_close("dependency_from_latency", 1000.0 / reconstructed_ms, dependency)

    config = economics or EconomicsConfig()
    paid_gpu_equivalents = float(EXPECTED_WORKERS)
    tok_s_per_paid_gpu = aggregate / paid_gpu_equivalents
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-immutable-baseline-v1",
        "baseline_id": BASELINE_ID,
        "status": "PASS",
        "immutability": {
            "source_experiment": "014",
            "source_artifacts_modified": False,
            "capture_only": True,
            "physical_3090_canary_run": False,
            "gpu_fleet_rented": False,
        },
        "architecture": {
            "topology": "WHOLE-LAYER",
            "physical_stages": EXPECTED_WORKERS,
            "primary_transformer_layers_per_stage": 1,
            "safe_batch": EXPECTED_SAFE_BATCH,
            "first_rejected_batch": EXPECTED_FIRST_REJECTED_BATCH,
            "persistent_runtime": True,
            "routing_semantics": "immutable Experiment 014 exact routing",
        },
        "performance": {
            "evidence_class": "PROJECTED",
            "aggregate_output_tok_s": aggregate,
            "dependency_bound_tok_s": dependency,
            "end_to_end_ms": end_to_end,
            "paid_gpu_equivalents": paid_gpu_equivalents,
            "aggregate_tok_s_per_paid_gpu_equivalent": tok_s_per_paid_gpu,
            "cost_per_million_output_tokens_usd": (
                config.cost_per_million_output_tokens(aggregate, paid_gpu_equivalents)
            ),
        },
        "latency_reconciliation": {
            "compute_components_ms": compute,
            "compute_total_ms": compute_ms,
            "coarse_edge_count": edge_count,
            "coarse_edge_service_ms": edge_ms,
            "coarse_edge_total_ms": edge_count * edge_ms,
            "reconstructed_end_to_end_ms": reconstructed_ms,
            "absolute_error_ms": abs(reconstructed_ms - end_to_end),
        },
        "sub_layer": {
            "evidence_class": "MEASURED",
            "best_expert_workers": EXPECTED_SUB_LAYER_WORKERS,
            "worker_resident_bytes": EXPECTED_SUB_LAYER_WORKER_BYTES,
            "complete_layer_relative_throughput": (
                EXPECTED_SUB_LAYER_RELATIVE_THROUGHPUT
            ),
            "smallest_tested_worker_bytes": EXPECTED_SMALLEST_WORKER_BYTES,
            "maximum_tested_rtt_ms": float(metrics["sub_layer_maximum_tested_rtt_ms"]),
            "minimum_tested_bandwidth_gbps": float(
                metrics["sub_layer_minimum_tested_bandwidth_gbps"]
            ),
            "conclusion": "FUNCTIONAL BUT NOT CURRENTLY ECONOMIC",
        },
        "economics": {
            "gpu_hourly_price_usd": config.gpu_hourly_price_usd,
            "output_price_per_million_usd": config.output_price_per_million_usd,
            "target_gpu_margin_fraction": config.target_gpu_margin_fraction,
            "break_even_tok_s_per_paid_gpu": config.break_even_tok_s_per_paid_gpu,
            "margin_tok_s_per_paid_gpu": config.margin_tok_s_per_paid_gpu,
        },
        "sources": {
            name: file_identity(path, relative_to=root) for name, path in sources.items()
        },
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = ["BASELINE_ID", "capture_baseline"]
