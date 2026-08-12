"""Pipeline-occupancy and asynchronous-speculation admission analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.evidence import atomic_json, read_json


def analyze_pipeline_occupancy(repository_root: Path, output_path: Path) -> dict[str, Any]:
    """Report target-stage occupancy bounds without inventing a draft GPU timing."""
    root = repository_root.expanduser().resolve()
    baseline = read_json(root / "artifacts" / "experiment-015" / "baseline" / "B015-000.json")
    performance = baseline["performance"]
    reconciliation = baseline["latency_reconciliation"]
    stages = int(baseline["architecture"]["physical_stages"])
    compute_ms = float(reconciliation["compute_total_ms"])
    latency_ms = float(performance["end_to_end_ms"])
    useful_stage_occupancy = compute_ms / (stages * latency_ms)

    rows = [
        {
            "mode": "ordinary autoregressive, one dependency-bound stream",
            "evidence_class": "PROJECTED",
            "useful_target_stage_occupancy_fraction": useful_stage_occupancy,
            "draft_target_overlap_fraction": 0.0,
            "accepted_output_tok_s": float(performance["dependency_bound_tok_s"]),
        },
        {
            "mode": "synchronous DSpark draft-then-verify",
            "evidence_class": "PROJECTED",
            "useful_target_stage_occupancy_fraction": "NOT_ESTABLISHED",
            "draft_target_overlap_fraction": 0.0,
            "accepted_output_tok_s": "NOT_ESTABLISHED",
        },
        {
            "mode": "asynchronous pipeline-aware DSpark",
            "evidence_class": "PROJECTED",
            "useful_target_stage_occupancy_fraction": "NOT_ESTABLISHED",
            "draft_target_overlap_fraction": "NOT_ESTABLISHED",
            "accepted_output_tok_s": "NOT_ESTABLISHED",
        },
    ]
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-pipeline-occupancy-v1",
        "cycle_id": "H015-002A",
        "status": "PASS",
        "rows": rows,
        "baseline_stage_time_definition": (
            "sum of retained Kimi compute divided by 93 paid stages and full dependency latency"
        ),
        "baseline_useful_target_stage_occupancy_percent": useful_stage_occupancy * 100.0,
        "early_cancellation": {
            "status": "NOT_BENCHMARKED",
            "layers_avoided": "NOT_ESTABLISHED",
            "cuda_work_avoided": "NOT_ESTABLISHED",
            "messages_avoided": "NOT_ESTABLISHED",
            "corrupted_state_count": 0,
            "note": "zero refers to transaction unit tests, not a distributed cancellation run",
        },
        "decision": "STOP_UNTIL_STANDARD_DSPARK_ACCEPTANCE_AND_GPU_DRAFT_LATENCY_PASS",
        "limitation": (
            "the pinned public DSpark runtime does not currently support pipeline parallelism, "
            "and Swarm lacks representative target acceptance plus a GPU draft timing"
        ),
    }
    atomic_json(output_path, receipt)
    return receipt


__all__ = ["analyze_pipeline_occupancy"]
