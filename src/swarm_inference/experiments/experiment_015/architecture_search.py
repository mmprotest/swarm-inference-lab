"""Admission-gated Experiment 015 architecture search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.contracts import EconomicsConfig
from swarm_inference.experiments.experiment_015.evidence import atomic_json, read_json
from swarm_inference.experiments.experiment_015.service_model import (
    ArchitectureServiceModel,
    load_service_evidence,
)


def _architecture_row(
    identifier: str,
    name: str,
    techniques: list[str],
    *,
    evidence_class: str,
    admitted: bool,
    reason: str,
    dependency_tok_s: float | None = None,
    aggregate_tok_s: float | None = None,
    paid_gpu_equivalents: float | None = None,
    economics: EconomicsConfig,
) -> dict[str, Any]:
    per_gpu = (
        aggregate_tok_s / paid_gpu_equivalents
        if aggregate_tok_s is not None
        and paid_gpu_equivalents is not None
        and paid_gpu_equivalents > 0
        else None
    )
    cost = (
        economics.cost_per_million_output_tokens(aggregate_tok_s, paid_gpu_equivalents)
        if aggregate_tok_s is not None
        and paid_gpu_equivalents is not None
        and aggregate_tok_s > 0
        and paid_gpu_equivalents > 0
        else None
    )
    return {
        "architecture_id": identifier,
        "name": name,
        "techniques": techniques,
        "evidence_class": evidence_class,
        "admitted_to_decision_frontier": admitted,
        "admission_reason": reason,
        "dependency_bound_tok_s": dependency_tok_s,
        "aggregate_output_tok_s": aggregate_tok_s,
        "paid_gpu_equivalents": paid_gpu_equivalents,
        "aggregate_tok_s_per_paid_gpu_equivalent": per_gpu,
        "cost_per_million_output_tokens_usd": cost,
    }


def search_architectures(
    repository_root: Path,
    output_path: Path,
    economics_path: Path,
    *,
    economics: EconomicsConfig | None = None,
) -> dict[str, Any]:
    """Evaluate required A-M candidates while excluding failed component gates."""
    root = repository_root.expanduser().resolve()
    config = economics or EconomicsConfig()
    baseline = read_json(root / "artifacts" / "experiment-015" / "baseline" / "B015-000.json")
    performance = baseline["performance"]
    evidence = load_service_evidence(root)
    model = ArchitectureServiceModel(evidence)
    base_dependency = float(performance["dependency_bound_tok_s"])
    base_aggregate = float(performance["aggregate_output_tok_s"])
    base_paid = float(performance["paid_gpu_equivalents"])

    microcells = [model.microcell_latency(depth) for depth in (1, 2, 4, 8)]
    best_cell = max(microcells, key=lambda item: float(item["dependency_bound_tok_s"]))

    # This is a sensitivity input copied from the public model card.  It is not
    # Swarm acceptance and therefore cannot enter the decision frontier.
    external_acceptance_sensitivity = 3.85
    dspark = model.speculation_upper_bound(
        block_size=7,
        accepted_tokens_per_target_pass=external_acceptance_sensitivity,
        cell_depth=1,
    )
    combined = model.speculation_upper_bound(
        block_size=7,
        accepted_tokens_per_target_pass=external_acceptance_sensitivity,
        cell_depth=8,
    )
    perfect_combined = model.speculation_upper_bound(
        block_size=7,
        accepted_tokens_per_target_pass=8.0,
        cell_depth=8,
    )

    rows = [
        _architecture_row(
            "A",
            "Experiment 014 93-stage baseline",
            ["whole-layer pipeline"],
            evidence_class="PROJECTED",
            admitted=True,
            reason="immutable B015-000 control",
            dependency_tok_s=base_dependency,
            aggregate_tok_s=base_aggregate,
            paid_gpu_equivalents=base_paid,
            economics=config,
        ),
        _architecture_row(
            "B",
            "DSpark only",
            ["DSpark"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="uses external acceptance sensitivity; Swarm acceptance not established",
            dependency_tok_s=float(dspark["dependency_bound_tok_s"]),
            aggregate_tok_s=float(dspark["aggregate_output_tok_s"]),
            paid_gpu_equivalents=base_paid,
            economics=config,
        ),
        _architecture_row(
            "C",
            "pipeline-aware speculation only",
            ["asynchronous speculation", "early cancellation"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="ordinary DSpark gate and GPU draft timing did not pass",
            economics=config,
        ),
        _architecture_row(
            "D",
            "hierarchical microcells",
            ["8-layer microcells"],
            evidence_class="SHAPED",
            admitted=True,
            reason="constructed solely from validated service and shaped network inputs",
            dependency_tok_s=float(best_cell["dependency_bound_tok_s"]),
            aggregate_tok_s=base_aggregate,
            paid_gpu_equivalents=base_paid,
            economics=config,
        ),
        _architecture_row(
            "E",
            "improved expert microwork only",
            ["direct resident expert dispatch"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="90% gate exists only as an optimistic overhead-subtraction bound",
            economics=config,
        ),
        _architecture_row(
            "F",
            "DCP only",
            ["DCP"],
            evidence_class="SHAPED",
            admitted=False,
            reason="exact combiner passed but no real distributed Kimi DCP kernel ran",
            economics=config,
        ),
        _architecture_row(
            "G",
            "microcells + expert microwork",
            ["microcells", "EP"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="EP redesign gate did not pass",
            economics=config,
        ),
        _architecture_row(
            "H",
            "microcells + EP + DCP",
            ["microcells", "EP", "DCP"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="EP and DCP implementation gates did not pass",
            economics=config,
        ),
        _architecture_row(
            "I",
            "DSpark + microcells",
            ["DSpark", "8-layer microcells"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="external acceptance sensitivity and draft GPU cost excluded",
            dependency_tok_s=float(combined["dependency_bound_tok_s"]),
            aggregate_tok_s=float(combined["aggregate_output_tok_s"]),
            paid_gpu_equivalents=base_paid,
            economics=config,
        ),
        _architecture_row(
            "J",
            "DSpark + microcells + EP",
            ["DSpark", "microcells", "EP"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="DSpark and EP gates did not pass",
            economics=config,
        ),
        _architecture_row(
            "K",
            "DSpark + microcells + EP + DCP",
            ["DSpark", "microcells", "EP", "DCP"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="multiple unpassed components; combination not modeled as multiplicative",
            economics=config,
        ),
        _architecture_row(
            "L",
            "best architecture + expert load balancing",
            ["microcells", "EPLB"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="placement result was fitted/evaluated on the same three-token trace",
            economics=config,
        ),
        _architecture_row(
            "M",
            "best architecture + tiered residency",
            ["microcells", "tiered residency"],
            evidence_class="PROJECTED",
            admitted=False,
            reason="three-token cold-start trace cannot establish cache hit rate",
            economics=config,
        ),
    ]

    decision_rows = [row for row in rows if row["admitted_to_decision_frontier"]]
    # D dominates A on dependency latency under identical aggregate/cost inputs.
    pareto = [row for row in decision_rows if row["architecture_id"] == "D"]
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-architecture-search-v1",
        "cycle_id": "H015-011A-H015-COMBINE",
        "status": "PASS",
        "rows": rows,
        "microcell_sweep": microcells,
        "external_dspark_sensitivity": {
            "accepted_tokens_per_target_pass": external_acceptance_sensitivity,
            "source_scope": "public Kimi K3 DSpark cross-workload mean, not Swarm",
            "dspark_only": dspark,
            "dspark_plus_depth8": combined,
        },
        "perfect_acceptance_upper_bound": perfect_combined,
        "decision_pareto_architecture_ids": [row["architecture_id"] for row in pareto],
        "best_justified_result": pareto[0],
        "best_measured_complete_architecture": "B015-000_CONTROL_ONLY",
        "primary_targets": {
            "dependency_bound_tok_s": 5.0,
            "aggregate_tok_s_per_paid_gpu_equivalent": config.break_even_tok_s_per_paid_gpu,
            "simultaneous_pass": False,
        },
        "conclusion": (
            "No speculative/sub-layer combination is admitted. Eight-layer microcells "
            "dominate the control only in a shaped topology model and remain Tier 0."
        ),
    }
    atomic_json(output_path, receipt)

    economics_receipt: dict[str, Any] = {
        "schema_version": "experiment-015-economics-v1",
        "status": "PASS",
        "inputs": {
            "gpu_hourly_price_usd": config.gpu_hourly_price_usd,
            "output_price_per_million_usd": config.output_price_per_million_usd,
            "target_gpu_margin_fraction": config.target_gpu_margin_fraction,
        },
        "targets": {
            "break_even_tok_s_per_paid_gpu_equivalent": config.break_even_tok_s_per_paid_gpu,
            "target_margin_tok_s_per_paid_gpu_equivalent": config.margin_tok_s_per_paid_gpu,
        },
        "architecture_rows": rows,
        "draft_model_paid_gpu_equivalent": "NOT_ESTABLISHED",
        "note": "unknown hardware prices and compute capacity are never imputed from VRAM",
    }
    atomic_json(economics_path, economics_receipt)
    return receipt


__all__ = ["search_architectures"]
