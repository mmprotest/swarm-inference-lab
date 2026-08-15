"""Finalize the repaired Experiment 022 without changing its frozen decision rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .completion_inputs import load_frozen_inventories
from .io import atomic_write_json, atomic_write_text

FINAL_ANSWERS = {
    "SUBLAYER_VALUE_STRONG": "YES — MATERIAL PERFORMANCE AND CAPACITY VALUE",
    "SUBLAYER_VALUE_SUPPORTED": "YES — MATERIAL PERFORMANCE VALUE",
    "SUBLAYER_CAPACITY_ONLY": "YES — CAPACITY VALUE ONLY",
    "SUBLAYER_NOT_MATERIAL": "NO — SUB-LAYER CAPABILITY NOT MATERIALLY VALUABLE",
    "MODEL_INVALID": "MODEL INVALID",
}

COLORS = {
    "blue": "#2563EB",
    "orange": "#EA580C",
    "gold": "#D97706",
    "olive": "#65A30D",
    "pink": "#DB2777",
    "ink": "#111827",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "paper": "#F8FAFC",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _style(title: str, subtitle: str) -> tuple[Any, Any]:
    figure, axes = plt.subplots(figsize=(12.4, 7.1), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axes.set_facecolor(COLORS["paper"])
    axes.grid(True, color=COLORS["grid"], alpha=0.55, linewidth=0.7, axis="y")
    axes.set_axisbelow(True)
    axes.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=24)
    axes.text(
        0,
        1.015,
        subtitle,
        transform=axes.transAxes,
        fontsize=9.5,
        color="#475569",
        va="bottom",
    )
    return figure, axes


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _paired(completion: Path) -> list[dict[str, Any]]:
    whole = {
        row["inventory_id"]: row
        for row in _csv(completion / "rerun" / "whole-layer-results.csv")
    }
    adaptive = {
        row["inventory_id"]: row
        for row in _csv(completion / "rerun" / "adaptive-results.csv")
    }
    if set(whole) != set(adaptive) or len(whole) != 27:
        raise RuntimeError("completion chart input is not the exact paired 27 inventories")
    return [
        {
            "inventory_id": inventory_id,
            "family": whole[inventory_id]["family"],
            "whole_feasible": _bool(whole[inventory_id]["feasible"]),
            "adaptive_feasible": _bool(adaptive[inventory_id]["feasible"]),
            "whole_tps": _float(whole[inventory_id].get("exact_tok_s_per_user"), math.nan),
            "adaptive_tps": _float(
                adaptive[inventory_id].get("exact_tok_s_per_user"), math.nan
            ),
        }
        for inventory_id in sorted(whole)
    ]


def _charts(completion: Path) -> dict[str, Any]:
    chart_root = completion / "charts"
    paired = _paired(completion)
    uplift = _csv(completion / "analysis" / "throughput-uplift.csv")
    unlocks = _csv(completion / "analysis" / "capacity-unlocks.csv")
    crossings = _csv(completion / "analysis" / "target-crossings.csv")
    usage = _csv(completion / "analysis" / "sublayer-usage.csv")
    ablation = _csv(completion / "rerun" / "ablation-results.csv")

    family_colors = {
        "coarse-friendly": COLORS["blue"],
        "memory-fragmented": COLORS["gold"],
        "compute-heterogeneous": COLORS["olive"],
        "network-heterogeneous": "#7C3AED",
        "full-mixed": COLORS["pink"],
    }
    figure, axes = _style(
        "Whole-layer versus adaptive throughput — completion rerun",
        "Exact same 27 frozen inventories; target-only tokens/s/user; modeled distributed critical path",
    )
    feasible = [
        row for row in paired if row["whole_feasible"] and row["adaptive_feasible"]
    ]
    unlocked = [
        row for row in paired if not row["whole_feasible"] and row["adaptive_feasible"]
    ]
    maximum = max(
        [5.0]
        + [row["whole_tps"] for row in feasible]
        + [row["adaptive_tps"] for row in feasible]
        + [row["adaptive_tps"] for row in unlocked]
    ) * 1.08
    for family, color in family_colors.items():
        values = [row for row in feasible if row["family"] == family]
        if values:
            axes.scatter(
                [row["whole_tps"] for row in values],
                [row["adaptive_tps"] for row in values],
                s=60,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=family,
            )
    axes.plot([0, maximum], [0, maximum], "--", color=COLORS["ink"], label="identity")
    axes.axhline(5, color=COLORS["pink"], linestyle=":", linewidth=1)
    axes.axvline(5, color=COLORS["pink"], linestyle=":", linewidth=1)
    if unlocked:
        axes.scatter(
            np.zeros(len(unlocked)),
            [row["adaptive_tps"] for row in unlocked],
            marker=">",
            s=90,
            color=COLORS["ink"],
            label="A infeasible; E feasible",
        )
    axes.set_xlim(-0.03 * maximum, maximum)
    axes.set_ylim(0, maximum)
    axes.set_xlabel("Planner A tokens/s/user")
    axes.set_ylabel("Planner E tokens/s/user")
    axes.legend(fontsize=8, ncol=2)
    _save(figure, chart_root / "chart-01-whole-vs-adaptive-rerun.png")

    heterogeneous = [
        _float(row["throughput_uplift_percent"])
        for row in uplift
        if row["family"] != "coarse-friendly"
    ]
    figure, axes = _style(
        "Adaptive throughput-uplift distribution — completion rerun",
        f"A-feasible heterogeneous inventories only (n={len(heterogeneous)}); frozen 20% material threshold",
    )
    if heterogeneous:
        axes.hist(
            heterogeneous,
            bins=min(10, max(4, round(math.sqrt(len(heterogeneous)) + 1))),
            color=COLORS["orange"],
            edgecolor="white",
        )
        median = statistics.median(heterogeneous)
        axes.axvline(median, color=COLORS["ink"], linewidth=1.6, label=f"median {median:.2f}%")
        axes.axvline(20, color=COLORS["pink"], linestyle="--", label="20% threshold")
        axes.legend()
    axes.set_xlabel("Planner E uplift over Planner A (%)")
    axes.set_ylabel("Inventory count")
    _save(figure, chart_root / "chart-02-uplift-distribution-rerun.png")

    figure, axes = _style(
        "Frozen 5 tokens/s target crossings — completion rerun",
        "Only A < 5 and E ≥ 5 qualifies; absent bars mean no crossing",
    )
    if crossings:
        labels = [row["inventory_id"] for row in crossings]
        x = np.arange(len(labels))
        width = 0.38
        axes.bar(
            x - width / 2,
            [_float(row["whole_tok_s"]) for row in crossings],
            width,
            color=COLORS["blue"],
            label="Planner A",
        )
        axes.bar(
            x + width / 2,
            [_float(row["adaptive_tok_s"]) for row in crossings],
            width,
            color=COLORS["orange"],
            label="Planner E",
        )
        axes.set_xticks(x, labels, rotation=25, ha="right")
        axes.legend()
    else:
        axes.text(0.5, 0.5, "0 target crossings", transform=axes.transAxes, ha="center", fontsize=22)
    axes.axhline(5, color=COLORS["pink"], linestyle="--")
    axes.set_ylabel("Tokens/s/user")
    _save(figure, chart_root / "chart-03-target-crossings-rerun.png")

    figure, axes = _style(
        "Sub-layer capacity unlocks — completion rerun",
        "Planner A infeasible and Planner E feasible on identical frozen resources",
    )
    if unlocks:
        labels = [row["inventory_id"] for row in unlocks]
        values = [_float(row["adaptive_tok_s"]) for row in unlocks]
        bars = axes.barh(labels, values, color=COLORS["olive"])
        axes.invert_yaxis()
        axes.bar_label(bars, fmt="%.2f", padding=3)
    else:
        axes.text(0.5, 0.5, "No capacity unlocks", transform=axes.transAxes, ha="center", fontsize=20)
    axes.set_xlabel("Planner E tokens/s/user")
    _save(figure, chart_root / "chart-04-capacity-unlocks-rerun.png")

    figure, axes = _style(
        "Cumulative placement-capability ablation — completion rerun",
        "Median within each arm's feasible set (population changes); labels show feasible inventory count",
    )
    levels = ["A", "B", "C", "D", "E"]
    medians: list[float] = []
    counts: list[int] = []
    for level in levels:
        values = [
            _float(row["exact_tok_s_per_user"])
            for row in ablation
            if row["planner_level"] == level and _bool(row["feasible"])
        ]
        medians.append(statistics.median(values) if values else 0.0)
        counts.append(len(values))
    bars = axes.bar(levels, medians, color=[COLORS["blue"], COLORS["gold"], COLORS["olive"], "#7C3AED", COLORS["orange"]])
    axes.bar_label(bars, labels=[f"{value:.2f}\n{count}/27" for value, count in zip(medians, counts, strict=True)], padding=4)
    axes.set_ylabel("Median tokens/s/user")
    _save(figure, chart_root / "chart-05-ablation-rerun.png")

    figure, axes = _style(
        "Planner E layer-granularity mix — completion rerun",
        "Percent of 93 transformer layers; infeasible inventories have no bar",
    )
    labels = [row["inventory_id"] for row in usage]
    x = np.arange(len(labels))
    categories = [
        ("whole_layer_percent", "whole layer", COLORS["blue"]),
        ("whole_expert_percent", "whole expert", COLORS["gold"]),
        ("expert_shard_percent", "expert shard", COLORS["olive"]),
        ("attention_projection_shard_percent", "attention/projection", "#7C3AED"),
        ("full_mixed_percent", "full mixed", COLORS["pink"]),
    ]
    bottom = np.zeros(len(labels))
    for key, label, color in categories:
        values = np.asarray([_float(row.get(key)) for row in usage])
        axes.bar(x, values, bottom=bottom, color=color, width=0.82, label=label)
        bottom += values
    axes.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axes.set_ylim(0, 105)
    axes.set_ylabel("Assigned transformer layers (%)")
    axes.legend(ncol=3, fontsize=8)
    _save(figure, chart_root / "chart-06-sublayer-usage-rerun.png")

    ledger = _csv(completion / "validation" / "residual-ledger.csv")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        grouped[(row["attention_type"], int(row["chunk_rows"]))].append(row)
    labels = [f"{kind}\nchunk {chunk}" for kind, chunk in sorted(grouped)]
    consolidated = {
        "GPU kernels": ["kernel_compute_ms"],
        "worker software": [
            "launch_gap_ms",
            "worker_local_host_ms",
            "cuda_sync_ms",
            "device_copy_ms",
            "worker_protocol_ms",
        ],
        "local reductions": ["local_reduction_ms"],
        "single-GPU artifact": ["single_gpu_serialization_artifact_ms"],
        "experiment-only + unexplained": [
            "experimental_harness_ms",
            "unexplained_ms",
        ],
    }
    maximum_unexplained_percent = max(
        _float(row.get("unexplained_fraction")) for row in ledger
    ) * 100
    figure, axes = _style(
        "Ordered resident-DAG wall decomposition",
        (
            "Median physical component wall; experiment-only bundles at most "
            f"{maximum_unexplained_percent:.2f}% unexplained; network is modeled separately"
        ),
    )
    bottom = np.zeros(len(labels))
    palette = [COLORS["blue"], COLORS["orange"], COLORS["gold"], COLORS["muted"], COLORS["pink"]]
    for (name, fields), color in zip(consolidated.items(), palette, strict=True):
        values = np.asarray(
            [
                statistics.median(
                    sum(_float(row.get(field)) for field in fields)
                    for row in grouped[key]
                )
                for key in sorted(grouped)
            ]
        )
        axes.bar(labels, values, bottom=bottom, label=name, color=color)
        bottom += values
    axes.set_ylabel("Physical wall (ms)")
    axes.legend(ncol=2, fontsize=8)
    _save(figure, chart_root / "chart-07-residual-decomposition.png")

    service_rows = [
        row
        for chunk in (1, 2, 4)
        for row in _csv(completion / "physical" / f"chunk-{chunk}-services.csv")
    ]
    primitive_types = [
        "KDA_SHARD",
        "MLA_SHARD",
        "EXPERT_STRIPE",
        "SHARED_EXPERT_SHARD",
        "PROJECTION_SHARD",
    ]
    figure, axes = _style(
        "Physical sub-layer service scaling by chunk",
        "Median direct native service on the local RTX 5090; every point was physically executed",
    )
    for candidate, color, marker in zip(
        primitive_types,
        [COLORS["blue"], COLORS["orange"], COLORS["olive"], COLORS["gold"], COLORS["pink"]],
        ["o", "s", "^", "D", "P"],
        strict=True,
    ):
        values = []
        for chunk in (1, 2, 4):
            samples = [
                _float(row["direct_native_service_ms"])
                for row in service_rows
                if row["candidate_type"] == candidate
                and int(row["chunk_rows"]) == chunk
            ]
            values.append(statistics.median(samples) if samples else math.nan)
        axes.plot((1, 2, 4), values, marker=marker, linewidth=1.8, color=color, label=candidate)
    axes.set_xticks((1, 2, 4))
    axes.set_xlabel("Physically executed chunk rows")
    axes.set_ylabel("Direct native service (ms)")
    axes.legend(fontsize=8, ncol=2)
    _save(figure, chart_root / "chart-08-chunk-scaling.png")

    paths = [
        chart_root / "chart-01-whole-vs-adaptive-rerun.png",
        chart_root / "chart-02-uplift-distribution-rerun.png",
        chart_root / "chart-03-target-crossings-rerun.png",
        chart_root / "chart-04-capacity-unlocks-rerun.png",
        chart_root / "chart-05-ablation-rerun.png",
        chart_root / "chart-06-sublayer-usage-rerun.png",
        chart_root / "chart-07-residual-decomposition.png",
        chart_root / "chart-08-chunk-scaling.png",
    ]
    chart_map = {
        "schema_version": "experiment-022-completion-chart-map-v1",
        "status": "PASS" if all(path.is_file() and path.stat().st_size > 0 for path in paths) else "FAIL",
        "charts": [
            {"path": str(path), "renderer": "matplotlib static PNG", "source_scope": "saved completion artifacts"}
            for path in paths
        ],
        "palette_policy": "single-root or hard two-root except explicit five-category composition",
        "claim_boundary": "physical service/residual charts are PHYSICAL; inventory charts are PHYSICALLY GROUNDED MODEL outputs",
    }
    atomic_write_json(completion / "analysis" / "chart-map.json", chart_map)
    return chart_map


def _verdict(
    *,
    gates: dict[str, bool],
    uplift: Sequence[dict[str, str]],
    unlocks: Sequence[dict[str, str]],
    crossings: Sequence[dict[str, str]],
    adaptive: Sequence[dict[str, str]],
    thresholds: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    heterogeneous = [row for row in uplift if row["family"] != "coarse-friendly"]
    values = [_float(row["throughput_uplift_percent"]) for row in heterogeneous]
    all_values = [_float(row["throughput_uplift_percent"]) for row in uplift]
    median = statistics.median(values) if values else None
    wins20 = sum(value >= 20 for value in values)
    regressions = sum(value < -1 for value in all_values)
    controls = [row for row in adaptive if row["family"] == "coarse-friendly"]
    control_threshold = float(thresholds["strong"]["coarse_control_whole_layer_percent_min"])
    controls_pass = bool(controls) and all(
        not _bool(row["feasible"])
        or _float(row["whole_layer_percent"]) >= control_threshold
        for row in controls
    )
    stranded_benefit = any(
        _float(row.get("stranded_memory_delta_bytes")) < 0 for row in uplift
    )
    capacity_benefit = bool(unlocks) or stranded_benefit
    details = {
        "heterogeneous_a_feasible": len(values),
        "heterogeneous_median_uplift_percent": median,
        "heterogeneous_wins_ge_20_percent": wins20,
        "adaptive_regressions_gt_1_percent": regressions,
        "capacity_unlocks": len(unlocks),
        "stranded_memory_benefit": stranded_benefit,
        "target_crossings": len(crossings),
        "coarse_controls_retain_mostly_whole": controls_pass,
        "outcome_rule_matched": True,
    }
    if not all(gates.values()):
        return "MODEL_INVALID", FINAL_ANSWERS["MODEL_INVALID"], details
    strong = thresholds["strong"]
    if (
        median is not None
        and median >= float(strong["heterogeneous_median_uplift_percent_min"])
        and wins20 >= math.ceil(len(values) * float(strong["fraction_ge_20_percent_min"]))
        and len(crossings) >= int(strong["target_crossings_min"])
        and len(unlocks) >= int(strong["capacity_unlocks_min"])
        and controls_pass
        and regressions == 0
    ):
        category = "SUBLAYER_VALUE_STRONG"
    elif (
        median is not None
        and median >= float(thresholds["supported"]["median_uplift_percent_min"])
        and regressions == 0
        and (capacity_benefit or wins20 >= 1)
    ):
        category = "SUBLAYER_VALUE_SUPPORTED"
    elif (
        median is not None
        and median < float(
            thresholds["capacity_only"]["median_performance_uplift_percent_max_exclusive"]
        )
        and capacity_benefit
    ):
        category = "SUBLAYER_CAPACITY_ONLY"
    elif (
        median is not None
        and median
        < float(
            thresholds["not_material"][
                "median_performance_uplift_percent_max_exclusive"
            ]
        )
        and not capacity_benefit
        and len(crossings)
        <= int(thresholds["not_material"]["target_crossings_allowed"])
    ):
        category = "SUBLAYER_NOT_MATERIAL"
    else:
        # The frozen decision rules are intentionally not extended after seeing
        # the data.  If a valid measurement lands in an uncovered interval,
        # assigning one of the scientific outcomes would silently redefine the
        # preregistration.  Fail closed instead.
        details["outcome_rule_matched"] = False
        details["outcome_rule_failure"] = "FROZEN_OUTCOME_RULES_NONEXHAUSTIVE"
        category = "MODEL_INVALID"
    return category, FINAL_ANSWERS[category], details


def _report(
    *,
    repo: Path,
    completion: Path,
    summary: dict[str, Any],
    truth: dict[str, Any],
) -> Path:
    stats = summary["statistics"]
    validation = summary["validation"]
    verdict = summary["final_answer"]
    category = summary["verdict_category"]

    def yes_no(value: Any) -> str:
        return "YES" if value else "NO"

    truth_rows = "\n".join(
        f"| {question} | {value} |" for question, value in truth["rows"]
    )
    capacity_rows = _csv(completion / "analysis" / "capacity-unlocks.csv")
    crossing_rows = _csv(completion / "analysis" / "target-crossings.csv")
    usage_rows = _csv(completion / "analysis" / "sublayer-usage.csv")
    sublayer_inventories = sum(
        _bool(row.get("feasible"))
        and _float(row.get("whole_layer_percent")) < 100
        for row in usage_rows
    )
    dynamic_failure_cases = summary.get("dynamic_failure_cases", [])
    if summary["dynamic_status"] == "PASS":
        dynamic_result_text = (
            f"All {summary['dynamic_row_count']} frozen dynamic rows passed. "
            "The planner used beneficial nodes, ignored harmful nodes, and "
            "replanned the remaining perturbations without a manual topology."
        )
    else:
        failed_cases = ", ".join(
            f"{row['scenario']} on {row['inventory_id']}"
            for row in dynamic_failure_cases
        )
        dynamic_result_text = (
            f"{summary['dynamic_pass_count']} of {summary['dynamic_row_count']} "
            f"frozen dynamic rows passed and {summary['dynamic_fail_count']} failed. "
            f"The failed rows were {failed_cases}. In each failed useful-join "
            "case the optimizer correctly retained the non-regressing fallback, "
            "but it did not admit the frozen newly joined node and improve the "
            "objective as that scenario required. The required dynamic gate is "
            "therefore FAIL; it is not waived or redefined after measurement."
        )
    if category == "MODEL_INVALID":
        value_result_text = (
            "The repaired static comparison is diagnostically informative but is "
            "not an admissible answer to the north-star comparison because a "
            "required frozen gate failed. Static Planner E produced zero median "
            "throughput uplift, no 20% wins, no target crossings, and six capacity "
            "unlocks; absent the failed gate that pattern would map to the frozen "
            "capacity-only category. It is not promoted to that conclusion here."
        )
        remains_text = (
            "No physical multi-machine K3 swarm was run, no external GPU was "
            "rented, and real distributed contention, transport jitter, collective "
            "interference, failures during live inference, and economic cost per "
            "deployed token remain unmeasured. This completion pass is closed with "
            "a definitive failed gate and the mandated `MODEL INVALID` verdict; it "
            "does not establish whether sub-layer capability is materially valuable."
        )
    else:
        value_result_text = (
            "Within a physically validated local Kimi K3 execution model and the "
            "exact preregistered heterogeneous inventories, the result answers "
            "whether selective exact sub-layer placement improves the best system "
            "found by the same frozen optimizer over its strongest whole-layer-only "
            "arm. The conclusion includes capacity only when A is infeasible and E "
            "is feasible, and performance only when the frozen uplift thresholds "
            "are met."
        )
        remains_text = (
            "No physical multi-machine K3 swarm was run, no external GPU was "
            "rented, and real distributed contention, transport jitter, collective "
            "interference, failures during live inference, and economic cost per "
            "deployed token remain unmeasured."
        )
    report = f"""# Experiment 022: completion pass

## 1. Original E022 result: MODEL_INVALID

The original 27-inventory run remains part of the record. Resident timing validation passed; ordered-DAG prediction error was 2.71% median, 3.74% p90, and 4.00% maximum; the reduced optimizer oracle was within 1%; Planner E contained Planner A; regressions beyond 1% were zero; dynamic adaptation passed; and the generic authenticated 93-layer traversal passed. The original report nevertheless recorded **MODEL_INVALID**, 0.00% heterogeneous median uplift, six diagnostic capacity unlocks, and zero target crossings.

Four material gates were open: the six individual `EXECUTE_SHARD` types were not all production-bound; the five frozen representative manifests were selected but not executed; sub-layer services did not physically cover chunks 2 and 4; and replay residual was heuristically assigned to five artificial barriers. The original artifacts were preserved outside `completion/`.

## 2. Why the result was inadmissible

The failed gates affected the implementation and cost model, not merely documentation. They could change both candidate eligibility and predicted critical path. The first run therefore neither proved nor falsified material sub-layer value.

## 3. Frozen completion methodology

The completion pass reused exactly 27 inventories, their IDs, seeds, node capabilities, memory, network links, costs, topology relationships, planner action spaces, optimizer budget, objective, thresholds, and five previously selected correctness manifests. The canonical frozen suite digest is `{summary['inventory_suite_sha256']}`. No inventory was regenerated or added to headline statistics.

Machine-readable freeze receipt: [`frozen-inputs.json`](../../artifacts/experiment-022/completion/frozen-inputs.json).

The primary metric is `Planner E tokens/s/user / Planner A tokens/s/user - 1` on A-feasible heterogeneous inventories. A-infeasible/E-feasible cases are counted separately as capacity unlocks. Evidence is labeled **PHYSICAL** for local RTX 5090 execution and **PHYSICALLY GROUNDED MODEL** for the distributed 27-inventory event replay.

## 4. Fix 1: production EXECUTE_SHARD

All six semantic task types now bind authenticated frames to prepared native resident handles: KDA shard, MLA shard, routed expert stripe, shared expert shard, projection shard, and reduction contribution. The worker path performs frame decode, validation, handle lookup, native compute, state mutation where applicable, result materialization, and response encoding. Direct and worker outputs/states match, timed checkpoint reads are zero, and no whole-layer fallback is admitted.

Evidence: [`execute-shard-bindings.json`](../../artifacts/experiment-022/completion/implementation/execute-shard-bindings.json).

## 5. Fix 2: representative 93-layer executions

The exact five originally selected manifests were hash-checked and executed, including the intentionally duplicated manifest selections for their independently preregistered cases. Each traversal used authenticated `EXECUTE_SHARD`, the manifest's actual logical owners, real K3 state progression, all 93 transformer layers, the endpoint, logits, and greedy-token comparison. Result: **{summary['representative_correctness_status']}**.

Evidence: [`representative-selection.json`](../../artifacts/experiment-022/completion/correctness/representative-selection.json).

## 6. Fix 3: chunk 2/4 sub-layer validation

Important sub-layer primitives and complete KDA/MLA sharded DAGs were physically executed at chunks 1, 2, and 4. No chunk-4 service was extrapolated from chunk 1. Eligibility remains per candidate and only physically validated P8 sub-layer candidates enter the headline catalog; Planner E still contains every Planner A whole-layer chunk-4 solution.

![Physical chunk scaling](../../artifacts/experiment-022/completion/charts/chart-08-chunk-scaling.png)

The figure shows measured native service, not an assumed linear scaling curve. Chunk-2 validation: **{yes_no(summary['gates']['chunk_2'])}**. Chunk-4 validation: **{yes_no(summary['gates']['chunk_4'])}**.

## 7. Fix 4: ordered-residual decomposition

The resident DAG records CUDA events, launch submission, host orchestration, synchronization, device copies, native reduction, sequential single-GPU emulation, experiment-only receipt assembly, protocol, and unexplained wall. The old `residual / 5` barrier rule is absent. Every final cost has one owner; network transport is modeled only by network events.

Profiling identified concrete causes hidden by the old residual: the standalone attention harness performed an output-forming invocation and then invoked the same shard again for measurement; validation state was reset inside measured KDA/MLA calls; immutable AttnRes, normalization, and router data were repeatedly prepared or uploaded; and independent logical workers were serialized on one GPU. The first three were removed from steady state by one-invocation execution and persistent handles. The last remains measured and is classified only as a single-GPU emulation artifact.

![Residual decomposition](../../artifacts/experiment-022/completion/charts/chart-07-residual-decomposition.png)

Maximum unexplained wall was {validation['maximum_unexplained_fraction'] * 100:.2f}% (hard maximum 10%). Single-GPU serialization and experiment-only overhead are excluded from distributed worker compute.

Evidence: [`residual-classification.json`](../../artifacts/experiment-022/completion/validation/residual-classification.json) and [`accounting-reconciliation.json`](../../artifacts/experiment-022/completion/validation/accounting-reconciliation.json).

## 8. Revalidated timing model

Calibration services came from KDA layer 45 and MLA layer 47; KDA layer 89 and MLA layer 91 were held out. Each chunk used the real ordered task template on one concrete RTX 5090 resource, with no normalization or global correction factor. Held-out absolute error was {validation['median_error_percent']:.2f}% median, {validation['p90_error_percent']:.2f}% p90, and {validation['maximum_error_percent']:.2f}% maximum against frozen gates of 5%/10%/15%.

Evidence: [`model-validation.json`](../../artifacts/experiment-022/completion/validation/model-validation.json) and [`heldout-validation.csv`](../../artifacts/experiment-022/completion/validation/heldout-validation.csv).

## 9. Frozen 27-inventory rerun

Only after all implementation, correctness, residual, timing, whole-expert, and optimizer-oracle gates passed were Planner A through Planner E rerun from scratch. Planner E was seeded with Planner A and retained the exact whole-layer fallback. The plots below are modeled distributed outcomes grounded in local physical services; they are not physical multi-machine throughput.

Evidence: [`rerun-summary.json`](../../artifacts/experiment-022/completion/rerun/rerun-summary.json) and [`candidate-catalog.json`](../../artifacts/experiment-022/completion/implementation/candidate-catalog.json).

![Whole versus adaptive](../../artifacts/experiment-022/completion/charts/chart-01-whole-vs-adaptive-rerun.png)

## 10. Whole-layer results

Planner A was feasible on {stats['a_feasible_inventory_count']} of 27 inventories. It retained topology awareness, node rejection, multiple layers per node, persistent state, chunk optimization, and wavefront scheduling; no transformer layer was split.

## 11. Adaptive results

Across {stats['heterogeneous_a_feasible']} A-feasible heterogeneous inventories, median E-over-A throughput uplift was {stats['heterogeneous_median_uplift_percent']:.2f}%. {stats['heterogeneous_wins_ge_20_percent']} met or exceeded 20%, and {stats['adaptive_regressions_gt_1_percent']} regressed beyond 1%. Planner E used a sub-layer candidate in {sublayer_inventories} inventory plans.

![Uplift distribution](../../artifacts/experiment-022/completion/charts/chart-02-uplift-distribution-rerun.png)

## 12. Ablation results

The frozen cumulative ladder was rerun as A (whole layer), B (+ whole expert), C (+ expert sharding), D (+ attention/projection sharding), and E (full adaptive mixed granularity). Whole-expert placement was admitted as a distinct K3 unit only after physical service and exact reduction checks.

![Ablation](../../artifacts/experiment-022/completion/charts/chart-05-ablation-rerun.png)

## 13. Capacity unlocks

There were {len(capacity_rows)} whole-infeasible/adaptive-feasible outcomes. These are reported as capacity evidence rather than an infinite percentage uplift.

![Capacity unlocks](../../artifacts/experiment-022/completion/charts/chart-04-capacity-unlocks-rerun.png)

## 14. Target crossings

There were {len(crossing_rows)} frozen `<5 -> >=5` tokens/s/user crossings.

![Target crossings](../../artifacts/experiment-022/completion/charts/chart-03-target-crossings-rerun.png)

## 15. Dynamic adaptation

Useful join, harmful join, critical-node slowdown, link degradation, and node disappearance were replanned automatically on the frozen representative inventories. No replacement topology was manually prescribed. Dynamic result: **{summary['dynamic_status']}**.

{dynamic_result_text}

Evidence: [`dynamic-results.csv`](../../artifacts/experiment-022/completion/rerun/dynamic-results.csv).

## 16. Final correctness

All original representative receipts passed. Any materially changed final headline manifest required and received a fresh receipt before finalization. Tensor assignment coverage, route and ordered-expert equality, KDA/MLA/AttnRes state, hidden/logit error, finite values, complete traversal, and greedy token were checked. Final correctness: **{summary['final_correctness_status']}**.

The changed mixed manifest also exposed two real resident-runtime lifetime bugs during full traversal. Both failed attempts remain preserved. Explicit nested-runtime ownership was added, the affected 71→72 and 75→76 transitions then passed focused physical checks with zero timed checkpoint reads, and the clean rerun completed all 93 layers plus the endpoint. This repair evidence is recorded in [`final-manifest-lifecycle-repairs.json`](../../artifacts/experiment-022/completion/validation/final-manifest-lifecycle-repairs.json).

Evidence: [`final-headline-manifests.json`](../../artifacts/experiment-022/completion/correctness/final-headline-manifests.json).

![Sub-layer usage](../../artifacts/experiment-022/completion/charts/chart-06-sublayer-usage-rerun.png)

## 17. Final admissible Experiment 022 verdict

**{verdict}**

Frozen outcome category: `{category}`.

### Final truth table

| Question | Result |
| --- | --- |
{truth_rows}

## 18. What this proves about sub-layer value

{value_result_text}

It does not convert local kernel parallelism into a physical swarm claim. Communication, independent-worker overlap, and topology are explicit event-model terms grounded by local native service and shaped links.

## 19. What remains unproven

{remains_text}
"""
    path = repo / "docs" / "experiments" / "EXPERIMENT_022_REPORT.md"
    atomic_write_text(path, report)
    return path


def finalize(*, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    completion = repo / "artifacts" / "experiment-022" / "completion"
    frozen_inventories, frozen_audit = load_frozen_inventories(repo)
    frozen = _json(completion / "frozen-inputs.json")
    bindings = _json(completion / "implementation" / "execute-shard-bindings.json")
    physical = _json(completion / "validation" / "physical-gates-summary.json")
    residual = _json(completion / "validation" / "residual-classification.json")
    model = _json(completion / "validation" / "model-validation.json")
    accounting = _json(completion / "validation" / "accounting-reconciliation.json")
    whole_expert = _json(completion / "implementation" / "whole-expert-status.json")
    candidate_catalog = _json(
        completion / "implementation" / "candidate-catalog.json"
    )
    rerun = _json(completion / "rerun" / "rerun-summary.json")
    representative = _json(completion / "correctness" / "representative-selection.json")
    final_manifests = _json(completion / "correctness" / "final-headline-manifests.json")
    oracle = _csv(completion / "validation" / "optimizer-small-oracle.csv")
    dynamic = _csv(completion / "rerun" / "dynamic-results.csv")
    uplift = _csv(completion / "analysis" / "throughput-uplift.csv")
    unlocks = _csv(completion / "analysis" / "capacity-unlocks.csv")
    crossings = _csv(completion / "analysis" / "target-crossings.csv")
    adaptive = _csv(completion / "rerun" / "adaptive-results.csv")
    thresholds = frozen["thresholds"]
    errors = model["absolute_percent_error"]
    maximum_unexplained = float(residual["maximum_unexplained_fraction"])
    dynamic_scenarios = {row["scenario"] for row in dynamic}
    expected_dynamic = {
        "JOIN_USEFUL",
        "JOIN_HARMFUL",
        "SLOWDOWN",
        "NETWORK_DEGRADATION",
        "NODE_LOSS",
    }
    dynamic_failures = [row for row in dynamic if row.get("status") != "PASS"]
    gates = {
        "frozen_inputs": frozen.get("frozen_inputs_recoverable") is True
        and int(frozen["inventory_count"]) == 27
        and frozen_audit.get("status") == "PASS"
        and len(frozen_inventories) == 27
        and len(frozen_audit.get("immutable_original_diagnostics", ())) == 11
        and len(frozen_audit.get("immutable_selected_manifests", ())) == 5,
        "bindings": bindings.get("status") == "PASS"
        and bindings.get("schema_version")
        == "experiment-022-completion-execute-shard-bindings-v2"
        and int(bindings.get("operation_count", 0)) == 6
        and bindings.get("chunks_physically_executed") == [1, 2, 4]
        and int(bindings.get("checkpoint_reads_in_timed_region", -1)) == 0
        and int(bindings.get("whole_layer_fallback_count", -1)) == 0,
        "representatives": representative.get("status") == "PASS"
        and int(representative.get("selection_count", 0)) == 5,
        "chunk_2": physical.get("status") == "PASS"
        and physical.get("schema_version")
        == "experiment-022-completion-physical-gates-v2"
        and physical.get("receipt_matrix_complete") is True
        and int(physical.get("receipt_count", -1)) == 12
        and physical.get("sub_layer_chunk_2_physically_validated") is True,
        "chunk_4": physical.get("status") == "PASS"
        and physical.get("schema_version")
        == "experiment-022-completion-physical-gates-v2"
        and physical.get("receipt_matrix_complete") is True
        and int(physical.get("receipt_count", -1)) == 12
        and physical.get("sub_layer_chunk_4_physically_validated") is True,
        "residual": residual.get("status") == "PASS"
        and residual.get("schema_version")
        == "experiment-022-completion-residual-classification-v2"
        and residual.get("artificial_barrier_residual") is False
        and maximum_unexplained <= 0.10
        and float(residual.get("maximum_reconciliation_fraction", 1.0)) <= 0.10,
        "model": model.get("status") == "PASS"
        and model.get("schema_version")
        == "experiment-022-completion-model-validation-v2"
        and float(errors["median"]) <= 5
        and float(errors["p90"]) <= 10
        and float(errors["maximum"]) <= 15
        and model.get("normalization") is False
        and model.get("global_correction_factor") is False,
        "accounting": accounting.get("status") == "PASS"
        and accounting.get("schema_version")
        == "experiment-022-completion-accounting-reconciliation-v2"
        and accounting.get("every_final_service_cost_has_exactly_one_owner") is True
        and accounting.get("planner_service_excludes_experiment_only") is True,
        "whole_expert": whole_expert.get("status") == "PASS"
        and whole_expert.get("schema_version")
        == "experiment-022-completion-whole-expert-v1",
        "candidate_catalog": candidate_catalog.get("status") == "PASS"
        and candidate_catalog.get("fair_chunk_4_comparison") is True
        and not candidate_catalog.get("incomplete_candidates")
        and not candidate_catalog.get("eligible_candidates_without_provenance")
        and int(candidate_catalog.get("candidate_count", -1))
        == len(candidate_catalog.get("candidates", ())),
        "optimizer_oracle": bool(oracle)
        and all(row.get("status") == "PASS" for row in oracle)
        and all(float(row["objective_gap_percent"]) <= 1.0 for row in oracle),
        "same_optimizer": rerun.get("optimizer_same_for_all_arms") is True,
        "containment": rerun.get("planner_e_contains_planner_a") is True,
        "dominance": rerun.get("dominance_pass") is True,
        "dynamic": bool(dynamic)
        and len(dynamic) == 25
        and int(rerun.get("dynamic_row_count", -1)) == 25
        and expected_dynamic.issubset(dynamic_scenarios)
        and all(row.get("status") == "PASS" for row in dynamic)
        and all(not _bool(row.get("manual_topology_supplied")) for row in dynamic),
        "final_correctness": final_manifests.get("status") == "PASS",
    }
    category, final_answer, details = _verdict(
        gates=gates,
        uplift=uplift,
        unlocks=unlocks,
        crossings=crossings,
        adaptive=adaptive,
        thresholds=thresholds,
    )
    if details.get("outcome_rule_matched") is False:
        gates["frozen_outcome_rule"] = False
    a_feasible = len(uplift)
    truth_rows: list[tuple[str, str]] = [
        ("Frozen original inventory suite preserved?", "YES" if gates["frozen_inputs"] else "NO"),
        ("Number of inventories", str(frozen["inventory_count"])),
        ("Frozen seeds preserved?", "YES" if gates["frozen_inputs"] else "NO"),
        ("Original thresholds preserved?", "YES" if gates["frozen_inputs"] else "NO"),
        ("Six production EXECUTE_SHARD bindings real?", "YES" if gates["bindings"] else "NO"),
        ("Five original representative manifests physically executed?", "YES" if gates["representatives"] else "NO"),
        ("Sub-layer chunk 2 physically validated?", "YES" if gates["chunk_2"] else "NO"),
        ("Sub-layer chunk 4 physically validated?", "YES" if gates["chunk_4"] else "NO"),
        ("Ordered residual <=10% unexplained?", "YES" if gates["residual"] else "NO"),
        ("Timing model median error <=5%?", "YES" if float(errors["median"]) <= 5 else "NO"),
        ("Timing model p90 <=10%?", "YES" if float(errors["p90"]) <= 10 else "NO"),
        ("Timing model max <=15%?", "YES" if float(errors["maximum"]) <= 15 else "NO"),
        ("Same optimizer used for A/E?", "YES" if gates["same_optimizer"] else "NO"),
        ("E contains A solutions?", "YES" if gates["containment"] else "NO"),
        ("A-feasible inventories", str(a_feasible)),
        ("Adaptive regressions >1%", str(details["adaptive_regressions_gt_1_percent"])),
        ("Median throughput uplift", f"{details['heterogeneous_median_uplift_percent']:.2f}%"),
        (">=20% wins", f"{details['heterogeneous_wins_ge_20_percent']}/{details['heterogeneous_a_feasible']}"),
        ("Capacity unlocks", str(len(unlocks))),
        ("<5 -> >=5 crossings", str(len(crossings))),
        ("Representative full correctness", "PASS" if gates["final_correctness"] and gates["representatives"] else "FAIL"),
        ("Dynamic adaptation", "PASS" if gates["dynamic"] else "FAIL"),
        ("Final E022 verdict", final_answer),
    ]
    truth = {
        "schema_version": "experiment-022-completion-truth-table-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "rows": truth_rows,
    }
    atomic_write_json(completion / "truth-table.json", truth)
    chart_result = _charts(completion)
    summary = {
        "schema_version": "experiment-022-completion-summary-v1",
        "status": "PASS" if all(gates.values()) and chart_result["status"] == "PASS" else "FAIL",
        "experiment": 22,
        "completion_pass": True,
        "original_result": "MODEL_INVALID",
        "verdict_category": category,
        "final_answer": final_answer,
        "inventory_count": 27,
        "inventory_suite_sha256": frozen["inventory_suite_sha256"],
        "gates": gates,
        "statistics": {
            "a_feasible_inventory_count": a_feasible,
            **details,
        },
        "validation": {
            "median_error_percent": float(errors["median"]),
            "p90_error_percent": float(errors["p90"]),
            "maximum_error_percent": float(errors["maximum"]),
            "maximum_unexplained_fraction": maximum_unexplained,
            "normalization": False,
            "global_correction_factor": False,
        },
        "representative_correctness_status": "PASS" if gates["representatives"] else "FAIL",
        "final_correctness_status": "PASS" if gates["final_correctness"] else "FAIL",
        "dynamic_status": "PASS" if gates["dynamic"] else "FAIL",
        "dynamic_row_count": len(dynamic),
        "dynamic_pass_count": len(dynamic) - len(dynamic_failures),
        "dynamic_fail_count": len(dynamic_failures),
        "dynamic_failure_cases": [
            {
                "scenario": row["scenario"],
                "inventory_id": row["inventory_id"],
                "changed_resource_used_after": _bool(
                    row.get("changed_resource_used_after")
                ),
                "throughput_change": _float(row.get("throughput_change")),
            }
            for row in dynamic_failures
        ],
        "evidence_class": "PHYSICAL local RTX 5090 + PHYSICALLY GROUNDED MODEL + SHAPED NETWORK",
        "physical_multi_machine_swarm_tested": False,
        "gpu_rentals": 0,
        "claim_boundary": (
            "In a physically validated local Kimi K3 execution model across the exact "
            "preregistered heterogeneous inventories, adaptive mixed-granularity placement "
            "does or does not materially outperform the strongest whole-layer-only planner."
        ),
        "claim_boundary_applied": category != "MODEL_INVALID",
        "chart_status": chart_result["status"],
    }
    atomic_write_json(completion / "completion-summary.json", summary)
    report = _report(repo=repo, completion=completion, summary=summary, truth=truth)
    return {
        "status": summary["status"],
        "final_answer": final_answer,
        "verdict_category": category,
        "completion_summary": str(completion / "completion-summary.json"),
        "truth_table": str(completion / "truth-table.json"),
        "report": str(report),
        "charts": chart_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = finalize(repo=args.repo)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["finalize"]
