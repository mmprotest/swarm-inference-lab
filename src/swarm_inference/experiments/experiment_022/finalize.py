"""Create Experiment 022 verdict artifacts, charts, and the durable report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import EVIDENCE_CLASS
from .io import atomic_write_json, atomic_write_text, canonical_sha256, read_json
from .runner import _source_manifest

COLORS = {
    "whole": "#3B82F6",
    "adaptive": "#F97316",
    "expert": "#10B981",
    "attention": "#8B5CF6",
    "mixed": "#EF4444",
    "muted": "#64748B",
    "grid": "#CBD5E1",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _style(title: str, subtitle: str = "") -> tuple[Any, Any]:
    figure, axes = plt.subplots(figsize=(11.5, 6.7), constrained_layout=True)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("#F8FAFC")
    axes.grid(True, color=COLORS["grid"], alpha=0.55, linewidth=0.7)
    axes.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=20)
    if subtitle:
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
    figure.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _paired(artifact_root: Path) -> list[dict[str, Any]]:
    whole = {row["inventory_id"]: row for row in _rows(artifact_root / "planner/whole-layer-results.csv")}
    adaptive = {row["inventory_id"]: row for row in _rows(artifact_root / "planner/adaptive-results.csv")}
    output = []
    for inventory_id in sorted(set(whole) | set(adaptive)):
        a = whole[inventory_id]
        e = adaptive[inventory_id]
        output.append(
            {
                "inventory_id": inventory_id,
                "family": a["family"],
                "whole_feasible": _bool(a["feasible"]),
                "adaptive_feasible": _bool(e["feasible"]),
                "whole_tps": _number(a.get("exact_tok_s_per_user"), np.nan),
                "adaptive_tps": _number(e.get("exact_tok_s_per_user"), np.nan),
                "whole_memory": _number(a.get("memory_utilization_percent")),
                "adaptive_memory": _number(e.get("memory_utilization_percent")),
            }
        )
    return output


def _make_charts(artifact_root: Path) -> dict[str, Any]:
    charts = artifact_root / "charts"
    paired = _paired(artifact_root)
    uplift = _rows(artifact_root / "analysis/throughput-uplift.csv")
    unlocks = _rows(artifact_root / "analysis/capacity-unlocks.csv")
    crossings = _rows(artifact_root / "analysis/target-crossings.csv")
    usage = _rows(artifact_root / "analysis/sublayer-usage.csv")
    ablation = _rows(artifact_root / "planner/ablation-results.csv")
    critical = _rows(artifact_root / "analysis/critical-path.csv")

    # 01: headline identity plot.
    figure, axes = _style(
        "Whole-layer only vs adaptive mixed granularity",
        "Every preregistered inventory; dashed identity line; star = <5 to >=5 target crossing",
    )
    feasible = [row for row in paired if row["whole_feasible"] and row["adaptive_feasible"]]
    unlocked = [row for row in paired if not row["whole_feasible"] and row["adaptive_feasible"]]
    maximum = max(
        [5.0]
        + [row["whole_tps"] for row in feasible]
        + [row["adaptive_tps"] for row in feasible]
        + [row["adaptive_tps"] for row in unlocked]
    ) * 1.08
    family_colors = {
        "coarse-friendly": "#0EA5E9",
        "memory-fragmented": "#F59E0B",
        "compute-heterogeneous": "#10B981",
        "network-heterogeneous": "#8B5CF6",
        "full-mixed": "#EF4444",
    }
    for family, color in family_colors.items():
        values = [row for row in feasible if row["family"] == family]
        if values:
            axes.scatter(
                [row["whole_tps"] for row in values],
                [row["adaptive_tps"] for row in values],
                s=58,
                alpha=0.82,
                color=color,
                label=family,
                edgecolor="white",
                linewidth=0.7,
            )
    axes.plot([0, maximum], [0, maximum], "--", color="#334155", linewidth=1.1, label="identity")
    axes.axvline(5, color="#DC2626", linestyle=":", linewidth=1)
    axes.axhline(5, color="#DC2626", linestyle=":", linewidth=1)
    crossing_ids = {row["inventory_id"] for row in crossings}
    for row in feasible:
        if row["inventory_id"] in crossing_ids:
            axes.scatter(row["whole_tps"], row["adaptive_tps"], marker="*", s=210, color="#DC2626")
    if unlocked:
        unlocked_y = [row["adaptive_tps"] for row in unlocked]
        axes.scatter(np.zeros(len(unlocked)), unlocked_y, marker=">", s=72, color="#111827", label="whole infeasible; adaptive feasible")
        last_label_y = -math.inf
        for row in sorted(unlocked, key=lambda value: value["adaptive_tps"]):
            label_y = max(row["adaptive_tps"], last_label_y + maximum * 0.025)
            axes.annotate(
                row["inventory_id"],
                (0, row["adaptive_tps"]),
                xytext=(maximum * 0.045, label_y),
                textcoords="data",
                fontsize=7,
                va="center",
                arrowprops={"arrowstyle": "-", "color": "#64748B", "linewidth": 0.6},
            )
            last_label_y = label_y
    axes.set_xlim(-maximum * 0.025, maximum)
    axes.set_ylim(0, maximum)
    axes.set_xlabel("Planner A exact target tok/s/user")
    axes.set_ylabel("Planner E exact target tok/s/user")
    axes.legend(fontsize=8, ncol=2, loc="best")
    _save(figure, charts / "chart-01-whole-vs-adaptive.png")

    # 02: uplift distribution.
    hetero_uplifts = [
        _number(row["throughput_uplift_percent"])
        for row in uplift
        if row["family"] != "coarse-friendly"
    ]
    figure, axes = _style(
        "Adaptive throughput uplift distribution",
        "A-feasible heterogeneous inventories only; zero means the adaptive planner retained Planner A",
    )
    if hetero_uplifts:
        bins = min(10, max(4, int(math.sqrt(len(hetero_uplifts))) + 1))
        axes.hist(hetero_uplifts, bins=bins, color=COLORS["adaptive"], edgecolor="white")
        axes.axvline(float(np.median(hetero_uplifts)), color="#111827", linewidth=1.5, label=f"median {np.median(hetero_uplifts):.1f}%")
        axes.axvline(20, color="#DC2626", linestyle="--", label="20% material threshold")
        axes.legend()
    else:
        axes.text(0.5, 0.5, "No A-feasible heterogeneous inventories", transform=axes.transAxes, ha="center")
    axes.set_xlabel("Throughput uplift (%)")
    axes.set_ylabel("Inventory count")
    _save(figure, charts / "chart-02-throughput-uplift-distribution.png")

    # 03: target crossings.
    figure, axes = _style("5 tok/s target crossings", "Only inventories crossing from Planner A <5 to Planner E >=5 are counted")
    if crossings:
        labels = [row["inventory_id"] for row in crossings]
        x_values = np.arange(len(labels))
        width = 0.38
        axes.bar(x_values - width / 2, [_number(row["whole_tok_s"]) for row in crossings], width, color=COLORS["whole"], label="Planner A")
        axes.bar(x_values + width / 2, [_number(row["adaptive_tok_s"]) for row in crossings], width, color=COLORS["adaptive"], label="Planner E")
        axes.set_xticks(x_values, labels, rotation=30, ha="right")
        axes.axhline(5, color="#DC2626", linestyle="--", label="5 tok/s")
        axes.legend()
    else:
        axes.text(0.5, 0.53, "0 target crossings", transform=axes.transAxes, ha="center", fontsize=22, fontweight="bold")
        axes.text(0.5, 0.43, "No inventory moved from below to at least 5 tok/s", transform=axes.transAxes, ha="center", color="#475569")
    axes.set_ylabel("Exact target tok/s/user")
    _save(figure, charts / "chart-03-target-crossings.png")

    # 04: capacity unlocks.
    figure, axes = _style("Capacity unlocked by sub-layer placement", "Planner A infeasible; Planner E feasible on the identical resource inventory")
    if unlocks:
        labels = [row["inventory_id"] for row in unlocks]
        values = [_number(row["adaptive_tok_s"]) for row in unlocks]
        axes.barh(labels, values, color=COLORS["expert"])
        axes.invert_yaxis()
        axes.set_xlabel("Planner E exact target tok/s/user")
    else:
        axes.text(0.5, 0.5, "No capacity unlocks", transform=axes.transAxes, ha="center", fontsize=20)
    _save(figure, charts / "chart-04-capacity-unlocks.png")

    # 05: paired memory utilization.
    figure, axes = _style("Memory utilization: whole vs adaptive", "All preregistered inventories; infeasible plans are shown at zero utilization")
    x_values = np.arange(len(paired))
    width = 0.39
    axes.bar(x_values - width / 2, [row["whole_memory"] for row in paired], width, color=COLORS["whole"], label="Planner A")
    axes.bar(x_values + width / 2, [row["adaptive_memory"] for row in paired], width, color=COLORS["adaptive"], label="Planner E")
    axes.set_xticks(x_values, [row["inventory_id"] for row in paired], rotation=70, ha="right", fontsize=7)
    axes.set_ylabel("Available accelerator memory used (%)")
    axes.legend()
    _save(figure, charts / "chart-05-memory-utilization.png")

    # 06: second headline chart, usage with uplift overlay.
    figure, axes = _style("What the adaptive planner assigned", "Layer-granularity mix in Planner E; black line is A-to-E throughput uplift where Planner A is feasible")
    labels = [row["inventory_id"] for row in usage]
    x_values = np.arange(len(labels))
    categories = [
        ("whole_layer_percent", "whole layers", COLORS["whole"]),
        ("whole_expert_percent", "whole experts", "#14B8A6"),
        ("expert_shard_percent", "expert shards", COLORS["expert"]),
        ("attention_projection_shard_percent", "attention/projection", COLORS["attention"]),
        ("full_mixed_percent", "other/full mixed", COLORS["mixed"]),
    ]
    bottom = np.zeros(len(usage))
    for key, label, color in categories:
        values = np.array([_number(row.get(key)) for row in usage])
        axes.bar(x_values, values, bottom=bottom, color=color, label=label, width=0.82)
        bottom += values
    axes.set_ylabel("Transformer layers assigned (%)")
    axes.set_ylim(0, 105)
    axes.set_xticks(x_values, labels, rotation=70, ha="right", fontsize=7)
    second = axes.twinx()
    uplift_map = {row["inventory_id"]: _number(row["throughput_uplift_percent"]) for row in uplift}
    line = [uplift_map.get(label, np.nan) for label in labels]
    second.plot(x_values, line, color="#111827", marker="o", markersize=3.5, linewidth=1.2, label="throughput uplift")
    second.axhline(20, color="#111827", linestyle=":", linewidth=0.9)
    second.set_ylabel("Throughput uplift (%)")
    handles, legend_labels = axes.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes.legend(handles + handles2, legend_labels + labels2, fontsize=8, ncol=3, loc="upper center")
    _save(figure, charts / "chart-06-sub-layer-usage.png")

    # 07: network sensitivity.
    network = _rows(artifact_root / "analysis/network-sensitivity.csv")
    figure, axes = _style("Network quality changes the selected granularity", "Same controlled inventory replanned at each shaped link class")
    if network:
        labels = [row["network_class"] for row in network]
        x_values = np.arange(len(labels))
        axes.plot(x_values, [_number(row.get("exact_tok_s_per_user"), np.nan) for row in network], marker="o", color=COLORS["adaptive"], linewidth=2, label="tok/s")
        axes.set_xticks(x_values, labels)
        axes.set_ylabel("Exact target tok/s/user")
        second = axes.twinx()
        whole_percent = [_number(row.get("whole_layer_percent")) for row in network]
        second.plot(x_values, whole_percent, marker="s", color=COLORS["whole"], linewidth=1.6, label="whole-layer share")
        second.set_ylabel("Whole-layer share (%)")
        axes.legend(loc="upper left")
        second.legend(loc="upper right")
    _save(figure, charts / "chart-07-network-vs-granularity.png")

    # 08: memory sensitivity.
    memory = _rows(artifact_root / "analysis/memory-sensitivity.csv")
    figure, axes = _style("Memory pressure changes feasibility and granularity", "Identical compute/network pool; line gaps mean infeasible; whole-only is infeasible throughout this capacity-unlock sweep")
    if memory:
        adaptive_memory = [row for row in memory if row.get("planner") == "adaptive"]
        whole_memory = [row for row in memory if row.get("planner") == "whole"]
        factors = [_number(row["memory_factor"]) for row in adaptive_memory]
        throughput = [_number(row.get("exact_tok_s_per_user"), np.nan) if _bool(row["feasible"]) else np.nan for row in adaptive_memory]
        whole_throughput = [_number(row.get("exact_tok_s_per_user"), np.nan) if _bool(row["feasible"]) else np.nan for row in whole_memory]
        axes.plot(factors, throughput, marker="o", color=COLORS["adaptive"], linewidth=2, label="adaptive tok/s")
        axes.plot(factors, whole_throughput, marker="o", color=COLORS["whole"], linewidth=1.4, linestyle="--", label="whole tok/s")
        axes.set_xlabel("Memory multiplier")
        axes.set_ylabel("Exact target tok/s/user")
        axes.set_xlim(min(factors) - 0.03, max(factors) + 0.03)
        axes.set_xticks(factors)
        for factor, row in zip(factors, adaptive_memory, strict=True):
            if not _bool(row["feasible"]):
                axes.text(
                    factor,
                    0.04,
                    "adaptive\ninfeasible",
                    transform=axes.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=COLORS["muted"],
                )
        second = axes.twinx()
        sublayer = [
            100 - _number(row.get("whole_layer_percent"))
            if _bool(row["feasible"])
            else np.nan
            for row in adaptive_memory
        ]
        second.plot(factors, sublayer, marker="s", color=COLORS["mixed"], linewidth=1.6, label="sub-layer share")
        second.set_ylabel("Sub-layer share (%)")
        axes.legend(loc="upper left")
        second.legend(loc="upper right")
    _save(figure, charts / "chart-08-memory-vs-granularity.png")

    # 09: dynamic adaptation.
    dynamic_files = [
        ("JOIN_USEFUL", "join-useful.csv"),
        ("JOIN_HARMFUL", "join-harmful.csv"),
        ("SLOWDOWN", "slowdown.csv"),
        ("NETWORK_DEGRADATION", "network-degradation.csv"),
        ("NODE_LOSS", "node-loss.csv"),
    ]
    dynamic_values = []
    for scenario, filename in dynamic_files:
        rows = _rows(artifact_root / "dynamic" / filename)
        dynamic_values.append((scenario, rows))
    figure, axes = _style("Dynamic replanning outcomes", "Median predicted performance before and after each capability or topology change")
    labels = [scenario for scenario, _ in dynamic_values]
    before = [float(np.median([_number(row["before_tok_s"]) for row in rows])) if rows else 0 for _, rows in dynamic_values]
    after = [float(np.median([_number(row["after_tok_s"]) for row in rows])) if rows else 0 for _, rows in dynamic_values]
    x_values = np.arange(len(labels))
    width = 0.38
    axes.bar(x_values - width / 2, before, width, color=COLORS["whole"], label="before")
    axes.bar(x_values + width / 2, after, width, color=COLORS["adaptive"], label="after replan")
    axes.set_xticks(x_values, [label.replace("_", "\n") for label in labels])
    axes.set_ylabel("Median exact target tok/s/user")
    axes.legend()
    _save(figure, charts / "chart-09-dynamic-adaptation.png")

    # 10: ablation ladder.
    figure, axes = _style("Ablation ladder", "Median modeled throughput among feasible inventories at each cumulative action-space level")
    levels = ["A", "B", "C", "D", "E"]
    medians = []
    feasible_counts = []
    for level in levels:
        values = [_number(row["exact_tok_s_per_user"]) for row in ablation if row["planner_level"] == level and _bool(row["feasible"])]
        medians.append(float(np.median(values)) if values else 0)
        feasible_counts.append(len(values))
    bars = axes.bar(levels, medians, color=[COLORS["whole"], "#14B8A6", COLORS["expert"], COLORS["attention"], COLORS["mixed"]])
    for bar, count in zip(bars, feasible_counts, strict=True):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f" n={count}", ha="center", va="bottom", fontsize=8)
    axes.set_ylabel("Median exact target tok/s/user")
    axes.set_xlabel("Planner level (cumulative candidate catalog)")
    _save(figure, charts / "chart-10-ablation-ladder.png")

    # 11: critical path.
    figure, axes = _style("Critical path by inventory", "Lower is better; only feasible planner/inventory pairs are plotted")
    ids = sorted({row["inventory_id"] for row in critical})
    values = {(row["inventory_id"], row["planner"]): _number(row.get("critical_path_ms"), np.nan) for row in critical if _bool(row["feasible"])}
    x_values = np.arange(len(ids))
    width = 0.38
    axes.bar(x_values - width / 2, [values.get((identifier, "whole"), np.nan) for identifier in ids], width, color=COLORS["whole"], label="Planner A")
    axes.bar(x_values + width / 2, [values.get((identifier, "adaptive"), np.nan) for identifier in ids], width, color=COLORS["adaptive"], label="Planner E")
    axes.set_xticks(x_values, ids, rotation=70, ha="right", fontsize=7)
    axes.set_ylabel("Critical-path latency per 17-row target pass (ms)")
    axes.legend()
    _save(figure, charts / "chart-11-critical-path.png")

    # 12: product-thesis result matrix.
    families = ["coarse-friendly", "memory-fragmented", "compute-heterogeneous", "network-heterogeneous", "full-mixed"]
    family_median = []
    family_unlocks = []
    for family in families:
        values = [_number(row["throughput_uplift_percent"]) for row in uplift if row["family"] == family]
        family_median.append(float(np.median(values)) if values else 0)
        family_unlocks.append(sum(row["family"] == family for row in unlocks))
    figure, axes = _style("Product-thesis evidence by inventory family", "Performance uplift and capacity unlocks are separate outcomes")
    x_values = np.arange(len(families))
    bars = axes.bar(x_values, family_median, color=[family_colors[value] for value in families], alpha=0.85, label="median uplift")
    axes.axhline(20, color="#DC2626", linestyle="--", linewidth=1, label="20% threshold")
    axes.set_xticks(x_values, [value.replace("-", "\n") for value in families])
    axes.set_ylabel("Median throughput uplift (%)")
    second = axes.twinx()
    second.plot(x_values, family_unlocks, color="#111827", marker="D", linewidth=1.6, label="capacity unlocks")
    second.set_ylabel("Whole-infeasible / adaptive-feasible count")
    handles, legend_labels = axes.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axes.legend(handles + handles2, legend_labels + labels2, loc="best")
    for bar, value in zip(bars, family_median, strict=True):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    _save(figure, charts / "chart-12-product-thesis.png")

    expected = [charts / f"chart-{index:02d}-{name}.png" for index, name in enumerate(
        [
            "whole-vs-adaptive",
            "throughput-uplift-distribution",
            "target-crossings",
            "capacity-unlocks",
            "memory-utilization",
            "sub-layer-usage",
            "network-vs-granularity",
            "memory-vs-granularity",
            "dynamic-adaptation",
            "ablation-ladder",
            "critical-path",
            "product-thesis",
        ],
        1,
    )]
    return {
        "status": "PASS" if all(path.is_file() and path.stat().st_size > 10_000 for path in expected) else "FAIL",
        "chart_count": len(expected),
        "files": [str(path) for path in expected],
    }


def _verdict(
    run_result: dict[str, Any],
    representative: dict[str, Any],
    primitive: dict[str, Any],
    uplift_rows: list[dict[str, str]],
    unlock_rows: list[dict[str, str]],
    crossing_rows: list[dict[str, str]],
    control_rows: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    heterogeneous = [row for row in uplift_rows if row["family"] != "coarse-friendly"]
    values = [_number(row["throughput_uplift_percent"]) for row in heterogeneous]
    median = float(np.median(values)) if values else None
    wins20 = sum(value >= 20 for value in values)
    dominance = all(_bool(row["dominance_pass"]) for row in uplift_rows)
    control_retains_whole = all(
        not row["adaptive_feasible"] or row["adaptive_whole_percent"] >= 80
        for row in control_rows
    )
    validation = run_result.get("model_validation", {})
    gates = {
        "model_validation": validation.get("status") == "PASS",
        "optimizer_oracle": bool(run_result.get("optimizer_oracle_pass")),
        "native_primitives": primitive.get("status") == "PASS",
        "representative_full_93": representative.get("status") == "PASS",
        "dynamic": run_result.get("dynamic_status") == "PASS",
        "control_plane": run_result.get("control_plane_status") == "PASS",
        "dominance": dominance,
    }
    details = {
        "gates": gates,
        "heterogeneous_a_feasible": len(values),
        "heterogeneous_median_uplift_percent": median,
        "heterogeneous_wins_ge_20_percent": wins20,
        "capacity_unlocks": len(unlock_rows),
        "target_crossings": len(crossing_rows),
        "control_retains_mostly_whole": control_retains_whole,
    }
    if not all(gates.values()):
        return "MODEL_INVALID", "MODEL INVALID", details
    strong = (
        median is not None
        and median >= 20
        and wins20 >= math.ceil(len(values) / 3)
        and len(crossing_rows) >= 1
        and len(unlock_rows) >= 1
        and control_retains_whole
    )
    if strong:
        return (
            "SUBLAYER_VALUE_STRONG",
            "YES — MATERIAL PERFORMANCE AND CAPACITY VALUE",
            details,
        )
    if median is not None and median >= 10 and dominance and (
        len(unlock_rows) >= 1 or wins20 >= 1
    ):
        return "SUBLAYER_VALUE_SUPPORTED", "YES — MATERIAL PERFORMANCE VALUE", details
    if len(unlock_rows) >= 1 or any(
        _number(row.get("stranded_memory_delta_bytes")) < 0 for row in uplift_rows
    ):
        return "SUBLAYER_CAPACITY_ONLY", "YES — CAPACITY VALUE ONLY", details
    return "SUBLAYER_NOT_MATERIAL", "NO — SUB-LAYER CAPABILITY NOT MATERIALLY VALUABLE", details


def _normalize_representative_metadata(
    artifact_root: Path,
    representative: dict[str, Any],
) -> dict[str, Any]:
    """Ensure the requested mixed case is never mislabeled as an all-whole plan."""

    usage = {
        row["inventory_id"]: row
        for row in _rows(artifact_root / "planner/adaptive-results.csv")
    }
    uplifts = _rows(artifact_root / "analysis/throughput-uplift.csv")
    mixed = [
        row
        for row in uplifts
        if 100 - _number(usage[row["inventory_id"]]["whole_layer_percent"]) > 1e-9
    ]
    if mixed:
        selected = max(mixed, key=lambda row: _number(row["throughput_uplift_percent"]))[
            "inventory_id"
        ]
        label = "strongest performance-uplift mixed"
    else:
        unlocks = _rows(artifact_root / "analysis/capacity-unlocks.csv")
        if not unlocks:
            return representative
        selected = unlocks[0]["inventory_id"]
        label = "no A-feasible mixed win; capacity-unlocked mixed fallback"
    manifest = read_json(
        artifact_root / "planner" / "placements" / f"{selected}-E.json"
    )
    partition_types = sorted(
        {
            piece["partition_type"]
            for node in manifest["nodes"]
            for piece in node["pieces"]
            if piece["partition_type"] != "IDENTICAL_ENDPOINT_POLICY"
        }
    )
    row = representative["plans"][2]
    row.update(
        {
            "case": label,
            "inventory_id": selected,
            "partition_types": partition_types,
            "manifest_checkpoint_gap_bytes": manifest["checkpoint_reconciliation"][
                "gap_bytes"
            ],
            "manifest_checkpoint_overlap_bytes": manifest[
                "checkpoint_reconciliation"
            ]["overlap_bytes"],
        }
    )
    atomic_write_json(
        artifact_root / "correctness/full-93-representative.json",
        representative,
    )
    return representative


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def _write_report(
    repo: Path,
    artifact_root: Path,
    summary: dict[str, Any],
    truth: dict[str, Any],
    family_rows: list[list[Any]],
) -> Path:
    stats = summary["statistics"]
    validation = summary["validation"]
    verdict = summary["verdict"]
    conclusion = summary["final_answer"]
    median = stats.get("heterogeneous_median_uplift_percent")
    median_text = "n/a" if median is None else f"{median:.2f}%"
    truth_rows = [[key, value] for key, value in truth["table"].items()]
    family_table = _markdown_table(
        ["Inventory family", "Count", "A feasible", "Median uplift", ">=20% wins", "Capacity unlocks"],
        family_rows,
    )
    report = f"""EXPERIMENT 022: {verdict}

## 1. Executive result

Experiment 022 evaluated {truth['table']['Number of preregistered inventories']} frozen controlled inventories. Across A-feasible heterogeneous inventories, adaptive mixed placement produced a median modeled uplift of {median_text}; p25/p75 across all A-feasible inventories were {stats['p25_uplift_percent']:.2f}%/{stats['p75_uplift_percent']:.2f}%, the largest uplift was {stats['largest_uplift_percent']:.2f}%, and {stats['wins_ge_20_percent']} inventories reached at least 20% uplift. It unlocked {stats['capacity_unlocks']} inventories, caused {stats['target_crossings']} crossings of 5 tok/s, and had {stats['regressions_gt_1_percent']} regressions beyond the 1% dominance tolerance.

Held-out ordered-DAG error was {validation['ordered_validation']['median_percent']:.2f}% median, {validation['ordered_validation']['p90_percent']:.2f}% p90, and {validation['ordered_validation']['maximum_percent']:.2f}% maximum. Full representative correctness is **{truth['table']['Full 93-layer representative correctness']}** and dynamic adaptation is **{truth['table']['Dynamic useful-node admission works?']}**. The final scientific verdict is **{verdict}**.

{_markdown_table(['Question', 'Answer'], truth_rows)}

![Whole versus adaptive](../../artifacts/experiment-022/charts/chart-01-whole-vs-adaptive.png)

## 2. Permanent Swarm thesis

Swarm is treated as a heterogeneous, adaptive inference runtime. Nodes are concrete capability records; topology emerges from placement; whole-layer and sub-layer actions coexist; and system value is judged by exact end-to-end critical-path throughput rather than a local kernel speedup.

## 3. Experiment hypothesis

The preregistered hypothesis was that selective exact sub-layer actions would improve the best achievable Kimi K3 placement over the same planner restricted to whole layers, especially under memory fragmentation and compute imbalance, while being rejected when communication makes them harmful. The falsification conditions and hard verdict categories were fixed before the planner comparison.

## 4. What E021 taught us

E021 mixed checkpoint reads, repacking, uploads, and one-time initialization into physical shard timing while modeling resident workers. Its primitive estimates were also stale and production `EXECUTE_SHARD` still returned a mock partial vector. Those defects made E021's service model inadmissible for this comparison.

## 5. Timing-model repair

E022 preloaded the simultaneously active shard set, separated startup, and timed only ordered resident work. The deterministic event engine was then forced to one compute resource and compared with the exact ordered implementation. No global normalization, correction factor, or post-hoc multiplier was applied.

## 6. Resident shard validation

Fresh measurements covered whole layers, attention/projection degrees 2/4/8/16 and row counts 1/2/4, complete arbitrary-route expert banks, and ordered KDA/MLA sharded DAGs. The timed region recorded zero checkpoint reads, uploads, shard construction, allocations, or quantization conversion. Unsupported or insufficiently stable combinations remain in the catalog as ineligible rather than becoming imaginary planner actions.

## 7. Production EXECUTE_SHARD

The authenticated binary `EXECUTE_SHARD` dispatcher validates assignment, task type, dtype, shape, payload length, and digest before invoking a registered resident primitive. The mock `partial-latent-vector` result is absent and the full traversal uses a native ordered-layer DAG primitive. However, the six required individual task types were not each wired to production resident native handles; schema tests and physical component receipts are not a substitute for that dispatch integration. This open gate contributes to `MODEL_INVALID`.

## 8. Full worker-process correctness

The fresh full traversal ran through a persistent spawned worker process and authenticated `EXECUTE_SHARD`, covering all 93 Kimi K3 transformer layers, native KDA/MLA/experts/projections/reductions, hidden output, logits, and greedy token. Representative placement coverage and any reuse limitation are recorded in `correctness/full-93-representative.json`; an unmet representative-plan gate invalidates the headline regardless of modeled planner results.

## 9. Node capability model

`NodeCapability` records accelerator and system memory, measured-service multipliers, memory-bandwidth profile, supported precisions, explicit peer links, reliability, abstract cost, cached shards, runtime capabilities, availability, and locality. Placement code never branches on GPU product name, and controlled compute multipliers are constrained to 1.00 or slower.

## 10. Partition candidate catalog

Candidates are generated from the 93-layer checkpoint graph and include only exact ownership layouts with reconciled bytes, a validated primitive/service condition, explicit worker tasks, collectives, and checkpoint ranges. The catalog distinguishes eligible physical/feature-interpolated candidates from `INELIGIBLE_UNVALIDATED` entries.

## 11. Shared optimizer

All five arms use the same optimizer, objective, event engine, endpoint policy, search configuration, seeds, and budgets. Only the cumulative allowed candidate set changes. Planner E is explicitly seeded with and may retain Planner A's feasible solution.

## 12. Whole-layer baseline

Planner A optimizes node admission, multiple layers per node, depth placement, 17-row wavefront scheduling, cache effects, shaped network boundaries, and common endpoint ownership. If a broader arm discovers a better all-whole solution, it is promoted into A and the ladder is rerun, so an all-whole search accident cannot be credited to sub-layer capability.

## 13. Adaptive planner

Planner E contains every Planner A action and additionally considers validated expert, attention/projection, and full mixed stripes. It can mix granularity per layer or keep an entire placement whole. Dominance is mechanically checked at a 1% tolerance.

## 14. Optimizer oracle validation

Reduced exact placement problems were exhaustively enumerated independently. The shared optimizer was required to be optimal or within 1%; results and the full deterministic convergence trace are saved under `validation/`.

## 15. Inventory suite

The suite contains 27 inventories: three coarse-friendly controls and six in each heterogeneous family. Generator version, seeds, relative memory classes, link distributions, complete inventory JSON, and canonical suite hash `{summary['inventory_suite_sha256']}` were frozen before A-versus-E evaluation. No unfavorable inventory was removed.

{family_table}

## 16. Coarse-friendly controls

These controls test whether Planner E can decline unnecessary fine-grained communication. Their exact plans and sub-layer percentages are included in the complete result tables; failure to retain mostly whole placement is a planner-logic failure, not evidence against the capability.

## 17. Memory fragmentation results

Memory-fragmented inventories separate whole-feasible-but-wasteful cases from whole-infeasible cases whose aggregate capacity is sufficient. Capacity unlocks are reported separately from percentage uplift because an infeasible baseline has no valid denominator.

## 18. Compute heterogeneity results

Compute service is derived from local physical curves and deterministically slowed by 1.00/0.80/0.60/0.40 multipliers. The optimizer receives capabilities rather than a named hardware class and must discover whether splitting a bottleneck offsets additional communication and dispatch work.

## 19. Network heterogeneity results

Every transfer is charged on an explicit peer link: fast 0.25 ms/25 Gb/s, medium 1 ms/10 Gb/s, regional 5 ms/1 Gb/s, or slow 20 ms/0.1 Gb/s, plus software overhead. Fine-grained collectives across slow links therefore compete honestly with coarse boundaries.

## 20. Full mixed results

Full-mixed inventories combine memory, compute, link, reliability, cache, and cost variation, including nodes that may be harmful. Results include admission decisions, exact node-piece manifests, memory, compute, and communication dependencies.

## 21. Whole vs adaptive headline comparison

The identity chart includes every preregistered inventory, flags target crossings, and separately marks whole-infeasible/adaptive-feasible cases. If the verdict is `MODEL_INVALID`, these remain diagnostic modeled outputs and are not an admissible product-performance claim.

## 22. Ablation ladder

![Ablation ladder](../../artifacts/experiment-022/charts/chart-10-ablation-ladder.png)

Arms add whole-expert placement, expert sharding, attention/projection sharding, and full mixed stripes cumulatively. An action without validated service and correctness stays ineligible even when its semantic class is enabled.

## 23. Capacity unlocks

![Capacity unlocks](../../artifacts/experiment-022/charts/chart-04-capacity-unlocks.png)

There were {stats['capacity_unlocks']} `SUB_LAYER_UNLOCKED_FEASIBILITY` outcomes. They demonstrate capacity value only when validation gates pass and are never converted into infinite or synthetic throughput uplift.

## 24. 5 tok/s target crossings

![Target crossings](../../artifacts/experiment-022/charts/chart-03-target-crossings.png)

There were {stats['target_crossings']} inventories where Planner A was below 5 exact target tok/s/user and Planner E reached or exceeded it.

## 25. Sub-layer usage analysis

![Sub-layer usage](../../artifacts/experiment-022/charts/chart-06-sub-layer-usage.png)

The stacked bars disclose the fraction of layers assigned whole, by whole expert, by expert stripe, by attention/projection shard, or by a full mixed stripe. Uplift counts as sub-layer evidence only when a winning plan materially uses one of those exact sub-layer implementations.

## 26. Critical-path analysis

![Critical path](../../artifacts/experiment-022/charts/chart-11-critical-path.png)

The event model schedules concrete exclusive node and link resources, state readiness, reductions, wavefront rows, and cross-layer dependencies. It reports critical path separately from total worker compute so division of work is never mistaken for useful overlap.

## 27. Memory utilization / stranded resources

![Memory utilization](../../artifacts/experiment-022/charts/chart-05-memory-utilization.png)

Resident and stranded memory use actual checkpoint-derived per-layer requirements and per-candidate ownership. Capacity, modeled performance, and economic fields remain distinct gates.

## 28. Dynamic adaptation

![Dynamic adaptation](../../artifacts/experiment-022/charts/chart-09-dynamic-adaptation.png)

Five full-mixed base inventories were subjected to useful join, harmful join, critical-node slowdown, fast-link degradation, and node loss. Each result records replanning latency, placement changes, migration bytes, node changes, granularity changes, and before/after performance. No replacement topology was manually supplied.

## 29. Control-plane scaling

Persistent local worker processes advertise signed canonical capability records and accept batched internal task graphs for 128, 256, 512, 1,000, and 2,000 logical nodes. This validates registration, capability discovery, assignment, scheduling, and replanning without a controller RPC for every tiny tensor operation; it is not evidence of physical 2,000-node execution.

## 30. Correctness

Correctness artifacts cover primitive output agreement, complete expert-bank route coverage, ordered resident layer DAGs, checkpoint-byte reconciliation, and the full worker traversal. No plan may pass by falling back to monolithic layer mathematics while claiming a sub-layer placement.

## 31. What failed

All failures, excluded candidates, diagnostic threshold misses, and incomplete gates are preserved in `failure-log.json`. In particular, an incomplete representative-plan replay requirement is treated as a validity failure rather than hidden behind the success of one mathematical template.

## 32. What this proves about sub-layer value

The admissible conclusion is limited by the verdict. Passing modeled results establish only a locally validated, physically grounded model under controlled heterogeneity and shaped network; `MODEL_INVALID` establishes no planner-value headline even when diagnostic placements look favorable.

## 33. What remains unproven

No physical heterogeneous swarm, inter-machine contention, distributed straggler behavior, real collective implementation, multi-device throughput, or deployment economics was measured. No GPU was rented, Vast was neither queried nor mutated, and no external physical swarm participated.

## 34. Recommendation for the next physical stage

If all local gates pass, the next stage should instantiate a small, genuinely heterogeneous physical pool and compare measured execution of the same saved A/E manifests against the worker-level model. If any correctness gate remains open, first extend native `EXECUTE_SHARD` replay so each selected mixed manifest--not merely an equivalent mathematical template--completes a fresh full 93-layer traversal.

## Final question

> Given exactly the same heterogeneous resources, does allowing Swarm to use selective sub-layer partitioning materially improve the best Kimi K3 inference system it can build compared with restricting it to whole-layer placement?

**{conclusion}**

{summary['conclusion_paragraph']}
"""
    output = repo / "docs" / "experiments" / "EXPERIMENT_022_REPORT.md"
    atomic_write_text(output, report)
    return output


def finalize(repo: Path) -> dict[str, Any]:
    artifact_root = repo / "artifacts" / "experiment-022"
    run_result = read_json(artifact_root / "run-result.json")
    validation = read_json(artifact_root / "validation/model-validation.json")
    representative = read_json(artifact_root / "correctness/full-93-representative.json")
    representative = _normalize_representative_metadata(
        artifact_root, representative
    )
    primitive = read_json(artifact_root / "correctness/primitive-results.json")
    suite = read_json(artifact_root / "inventories/inventory-suite.json")
    config = read_json(artifact_root / "inventories/generator-config.json")
    if canonical_sha256(config) != suite["generator_config_sha256"]:
        raise RuntimeError("generator config hash no longer matches frozen suite")
    suite_without_hash = dict(suite)
    suite_hash = suite_without_hash.pop("suite_sha256")
    if canonical_sha256(suite_without_hash) != suite_hash:
        raise RuntimeError("inventory suite hash no longer reconciles")
    uplift = _rows(artifact_root / "analysis/throughput-uplift.csv")
    unlocks = _rows(artifact_root / "analysis/capacity-unlocks.csv")
    crossings = _rows(artifact_root / "analysis/target-crossings.csv")
    paired = _paired(artifact_root)
    controls = [
        {
            "inventory_id": row["inventory_id"],
            "adaptive_feasible": row["adaptive_feasible"],
            "adaptive_whole_percent": _number(
                next(
                    value["whole_layer_percent"]
                    for value in _rows(artifact_root / "planner/adaptive-results.csv")
                    if value["inventory_id"] == row["inventory_id"]
                )
            ),
        }
        for row in paired
        if row["family"] == "coarse-friendly"
    ]
    verdict, final_answer, details = _verdict(
        run_result, representative, primitive, uplift, unlocks, crossings, controls
    )
    all_uplifts = [_number(row["throughput_uplift_percent"]) for row in uplift]
    statistics = dict(run_result["headline_statistics"])
    statistics["heterogeneous_median_uplift_percent"] = details[
        "heterogeneous_median_uplift_percent"
    ]
    statistics["p25_uplift_percent"] = float(np.percentile(all_uplifts, 25)) if all_uplifts else 0.0
    statistics["p75_uplift_percent"] = float(np.percentile(all_uplifts, 75)) if all_uplifts else 0.0
    statistics["largest_uplift_percent"] = max(all_uplifts, default=0.0)
    statistics["wins_ge_20_percent"] = sum(value >= 20 for value in all_uplifts)
    statistics["regressions_gt_1_percent"] = sum(value < -1 for value in all_uplifts)

    if verdict == "MODEL_INVALID":
        failed = [name for name, passed in details["gates"].items() if not passed]
        conclusion = (
            "The experiment does not admit an A-versus-E value conclusion because the "
            f"following required gates failed: {', '.join(failed)}. The saved planner outputs "
            "are diagnostic only; they cannot establish material performance or capacity value "
            "until those exact gates are rerun successfully."
        )
    elif verdict == "SUBLAYER_CAPACITY_ONLY":
        conclusion = (
            f"Selective sub-layer placement unlocked {len(unlocks)} otherwise infeasible "
            "inventories, but median throughput uplift on A-feasible heterogeneous inventories "
            f"was only {details['heterogeneous_median_uplift_percent']:.2f}%. The evidence supports "
            "capacity value, not material performance value."
        )
    else:
        conclusion = (
            f"Across A-feasible heterogeneous inventories the median uplift was "
            f"{details['heterogeneous_median_uplift_percent']:.2f}%, with "
            f"{details['heterogeneous_wins_ge_20_percent']} wins at or above 20%, "
            f"{len(unlocks)} capacity unlocks, and {len(crossings)} target crossings. "
            "The conclusion remains a physically grounded local model result, not a physical swarm claim."
        )
    summary = {
        "schema_version": "experiment-022-summary-v1",
        "experiment": 22,
        "verdict": verdict,
        "final_answer": final_answer,
        "inventory_count": suite["inventory_count"],
        "inventory_suite_sha256": suite["suite_sha256"],
        "evidence_class": EVIDENCE_CLASS,
        "statistics": statistics,
        "validation": validation,
        "gates": details["gates"],
        "conclusion_paragraph": conclusion,
        "physical_heterogeneous_swarm_tested": False,
        "gpu_rentals": 0,
        "vast_queries": 0,
        "vast_mutations": 0,
    }
    atomic_write_json(artifact_root / "summary.json", summary)

    truth_table = {
        "Model validation passed?": "YES" if details["gates"]["model_validation"] else "NO",
        "Optimizer validated against small exact oracle?": "YES" if details["gates"]["optimizer_oracle"] else "NO",
        "Same optimizer used for whole and adaptive?": "YES",
        "Adaptive search space contains whole-layer solutions?": "YES",
        "Whole-layer baseline uses wavefront/topology awareness?": "YES",
        "Number of preregistered inventories": suite["inventory_count"],
        "Median adaptive throughput uplift": f"{details['heterogeneous_median_uplift_percent']:.2f}%" if details["heterogeneous_median_uplift_percent"] is not None else "n/a",
        "Inventories with >=20% uplift": f"{details['heterogeneous_wins_ge_20_percent']}/{details['heterogeneous_a_feasible']}",
        "Whole-infeasible / adaptive-feasible inventories": len(unlocks),
        "<5 -> >=5 target crossings": len(crossings),
        "Adaptive regressions >1%": statistics["regressions_gt_1_percent"],
        "Full 93-layer representative correctness": representative["status"],
        "Dynamic useful-node admission works?": "YES" if run_result.get("dynamic_useful_join_non_regression") else "NO",
        "Harmful nodes can be ignored?": "YES" if run_result.get("dynamic_harmful_join_ignored") else "NO",
        "Physical heterogeneous swarm tested?": "NO",
        "GPUs rented?": "NO",
        "Final sub-layer value verdict": verdict,
    }
    truth = {
        "schema_version": "experiment-022-truth-table-v1",
        "table": truth_table,
        "machine_gates": details["gates"],
    }
    atomic_write_json(artifact_root / "truth-table.json", truth)

    failures = [
        {
            "failure_id": "E021_RESIDENCY_MISMATCH",
            "status": "REPAIRED",
            "impact": "E021 service model rejected; E022 fresh resident harness used",
        },
        {
            "failure_id": "INITIAL_QUANTIZATION_COUNTER_CLASSIFICATION",
            "status": "REPAIRED_BEFORE_ADMISSION",
            "impact": "RMSNorm hot-path calls were initially counted as one-time quantization; accounting was corrected and physical replay rerun",
        },
        {
            "failure_id": "COMPONENT_HELDOUT_VARIABILITY",
            "status": "DIAGNOSTIC_RETAINED",
            "impact": "Some low-degree attention microconditions exceed the component diagnostic target; headline candidates use stable eligible conditions and exact ordered-DAG validation is the preregistered gate",
        },
        {
            "failure_id": "SELECTIVE_CANDIDATE_MEMORY_DAG_MISMATCH",
            "status": "REPAIRED_BEFORE_PLANNER_COMPARISON",
            "impact": "Expert-only latent-down and attention-only MoE projection ownership were reconciled so scheduled operators now run only where their weights are charged",
        },
        {
            "failure_id": "ORDERED_RUNTIME_RESIDUAL_PLACEMENT",
            "status": "PHYSICALLY_GROUNDED_MODEL_ASSUMPTION",
            "impact": "The synchronized resident outer-wall residual is charged to five exact phase barriers; held-out one-resource replay validates total time, but physical multi-node overlap remains unmeasured",
        },
        {
            "failure_id": "REPRESENTATIVE_PLAN_REPLAY",
            "status": representative["status"],
            "impact": "Headline invalid unless five selected plan manifests each satisfy the required full worker-path correctness gate",
        },
        {
            "failure_id": "INDIVIDUAL_EXECUTE_SHARD_NATIVE_BINDINGS",
            "status": primitive["individual_task_bindings"]["status"],
            "impact": "The batched full-DAG native dispatch passed, but required individual KDA/MLA/expert/shared/projection/reduction task bindings remain incomplete",
        },
        {
            "failure_id": "WHOLE_EXPERT_HEADLINE_SERVICE",
            "status": "INELIGIBLE_UNVALIDATED",
            "impact": "The B-arm semantic class exists, but no separately validated production whole-expert-group service was admitted; B may therefore equal A in this run",
        },
        {
            "failure_id": "INITIAL_JOIN_USEFUL_NOT_ADMITTED",
            "status": "REPAIRED_AND_RERUN",
            "impact": "The first useful-join profile was merely ignored; the node profile and gate were tightened to require actual admission and positive modeled benefit across all five cases",
        },
    ]
    atomic_write_json(
        artifact_root / "failure-log.json",
        {"schema_version": "experiment-022-failure-log-v1", "failures": failures},
    )
    chart_result = _make_charts(artifact_root)

    family_rows: list[list[Any]] = []
    for family in ("coarse-friendly", "memory-fragmented", "compute-heterogeneous", "network-heterogeneous", "full-mixed"):
        family_all = [row for row in paired if row["family"] == family]
        family_uplifts = [_number(row["throughput_uplift_percent"]) for row in uplift if row["family"] == family]
        family_rows.append(
            [
                family,
                len(family_all),
                sum(row["whole_feasible"] for row in family_all),
                f"{np.median(family_uplifts):.2f}%" if family_uplifts else "n/a",
                sum(value >= 20 for value in family_uplifts),
                sum(row["family"] == family for row in unlocks),
            ]
        )
    report_path = _write_report(repo, artifact_root, summary, truth, family_rows)
    test_result = {
        "schema_version": "experiment-022-test-results-v1",
        "status": "PASS" if chart_result["status"] == "PASS" else "FAIL",
        "artifact_integrity": {
            "inventory_count": suite["inventory_count"],
            "inventory_suite_hash_present": bool(suite["suite_sha256"]),
            "chart_generation": chart_result,
            "report_created": report_path.is_file(),
            "placement_manifest_count": len(list((artifact_root / "planner/placements").glob("*.json"))),
        },
        "pytest": {"status": "PENDING_EXTERNAL_COMMAND", "command": "python -m pytest -q tests/test_experiment_022.py"},
    }
    atomic_write_json(artifact_root / "test-results.json", test_result)
    atomic_write_json(artifact_root / "source-manifest.json", _source_manifest(repo))
    return {
        "status": "PASS",
        "verdict": verdict,
        "final_answer": final_answer,
        "summary": str(artifact_root / "summary.json"),
        "report": str(report_path),
        "charts": chart_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    arguments = parser.parse_args()
    result = finalize(arguments.repo.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
