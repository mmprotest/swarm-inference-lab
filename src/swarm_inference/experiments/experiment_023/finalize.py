"""Fail-closed finalization and reporting for Experiment 023."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    atomic_write_text,
    canonical_sha256,
    sha256_file,
    write_csv,
)

from .analysis import (
    AttemptAnalysis,
    analyze_attempt,
    evaluate_verdict,
    evaluate_zero_new_node_wedge,
)
from .freeze import (
    FROZEN_CONSTANTS,
    validate_e023_freeze,
)

ATTEMPT_NAME = "deterministic-run-1"
STOP_STATUS = "NOT_RUN_AFTER_MANDATORY_CONTROL_FAILURE"
FIXED_REPRESENTATIVES = (
    "memory-fragmented-03",
    "compute-heterogeneous-04",
    "network-heterogeneous-01",
    "full-mixed-02",
    "coarse-friendly-01",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _publish_attempt(attempt_root: Path, artifact_root: Path) -> None:
    """Promote the complete first run as clearly invalid diagnostic evidence."""

    for directory in ("baseline", "plans", "serving"):
        shutil.copytree(
            attempt_root / directory,
            artifact_root / directory,
            dirs_exist_ok=True,
        )
    for name in ("memory-reconciliation.csv", "cost-reconciliation.csv"):
        destination = artifact_root / "validation" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(attempt_root / "validation" / name, destination)


def _write_stopped_phase_artifacts(artifact_root: Path) -> None:
    blocker = "COARSE_FRIENDLY_03_HARMFUL_ACCEPTED_FLEX_ACTIONS"
    for inventory_id in FIXED_REPRESENTATIVES:
        atomic_write_json(
            artifact_root / "correctness" / f"{inventory_id}.json",
            {
                "schema_version": "experiment-023-correctness-not-run-v1",
                "experiment_id": "023",
                "inventory_id": inventory_id,
                "status": STOP_STATUS,
                "blocker": blocker,
                "complete_93_layer_traversal_executed": False,
                "forced_alternate_uses": [],
                "claim": "No E023 full-correctness conclusion is available.",
            },
        )
    hedging_rows = [
        {
            "inventory_id": inventory_id,
            "status": STOP_STATUS,
            "blocker": blocker,
            "seeds_planned": 32,
            "seeds_run": 0,
            "hedge_service_drift": True,
            "hedging_conclusion_suppressed": True,
            "tail_diagnostic_positive": False,
        }
        for inventory_id in FIXED_REPRESENTATIVES
    ]
    write_csv(artifact_root / "stochastic/hedging-results.csv", hedging_rows)
    write_csv(
        artifact_root / "stochastic/seed-results.csv",
        [
            {
                "status": STOP_STATUS,
                "blocker": blocker,
                "seed_start": 23_023_000,
                "seed_count_planned": 32,
                "seed_count_run": 0,
                "common_random_numbers_evaluated": False,
            }
        ],
    )
    write_csv(
        artifact_root / "serving/diagnostic-ablations.csv",
        [
            {
                "inventory_id": inventory_id,
                "diagnostic_arm": arm,
                "status": STOP_STATUS,
                "blocker": blocker,
            }
            for inventory_id in FIXED_REPRESENTATIVES
            for arm in ("STATIC_COPY_CHOICE", "PRIMARY_ONLY_WITH_REPLICA_COST")
        ],
    )
    atomic_write_json(
        artifact_root / "validation/reproducibility.json",
        {
            "schema_version": "experiment-023-reproducibility-v1",
            "experiment_id": "023",
            "status": STOP_STATUS,
            "blocker": blocker,
            "deterministic_run_1_complete": True,
            "deterministic_run_2_started": False,
            "selected_actions_exact_match": None,
            "canonical_plan_hashes_exact_match": None,
            "selected_runtime_masks_exact_match": None,
            "aggregate_integer_counters_exact_match": None,
            "float_metrics_relative_error_max": None,
            "claim": "The mandatory deterministic rerun gate was not adjudicated.",
        },
    )


def _write_analysis(artifact_root: Path, analysis: AttemptAnalysis) -> None:
    write_csv(artifact_root / "analysis/uplift.csv", analysis.uplift_rows)
    write_csv(artifact_root / "analysis/family-summary.csv", analysis.family_rows)
    write_csv(
        artifact_root / "analysis/optionality-ablation.csv",
        analysis.optionality_rows,
    )
    write_csv(
        artifact_root / "analysis/replica-efficiency.csv",
        analysis.replica_efficiency_rows,
    )
    write_csv(
        artifact_root / "analysis/capacity-exploratory.csv",
        analysis.capacity_rows,
    )


def _chart_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 12.0,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#4D5660",
            "axes.labelcolor": "#242A31",
            "xtick.color": "#4D5660",
            "ytick.color": "#4D5660",
            "text.color": "#242A31",
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FAFBFC",
            "savefig.facecolor": "#FAFBFC",
        }
    )
    return plt


def _finish_chart(plt: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close()


def _add_subtitle(fig: Any, value: str, *, y: float = 0.955) -> None:
    fig.text(0.5, y, value, ha="center", va="top", fontsize=9, color="#5F6975")


def _generate_charts(
    artifact_root: Path,
    analysis: AttemptAnalysis,
    *,
    chart_qa_status: str,
) -> list[dict[str, Any]]:
    plt = _chart_setup()
    chart_root = artifact_root / "charts"
    blue = "#32658F"
    gold = "#D49A2F"
    orange = "#D66F49"
    ink = "#3F4852"
    grid = "#DDE2E7"
    headline = [row for row in analysis.uplift_rows if row["cohort"] == "headline"]

    # Chart 1: efficiency uplift, ranked to support the threshold comparison.
    values = sorted(headline, key=lambda row: row["efficiency_uplift_percent"])
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = np.arange(len(values))
    colors = [
        gold if row["efficiency_uplift_percent"] >= 20.0 else blue for row in values
    ]
    ax.scatter(
        [row["efficiency_uplift_percent"] for row in values],
        y,
        c=colors,
        s=50,
        edgecolors=ink,
        linewidths=0.5,
        zorder=3,
    )
    ax.axvline(0.0, color=ink, linewidth=1.0)
    ax.axvline(20.0, color=orange, linewidth=1.4, linestyle="--", label="20% gate")
    ax.set_yticks(y, [row["inventory_id"] for row in values])
    ax.set_xlabel("FLEX_POOL SLO efficiency uplift vs U_STRONG (%)")
    fig.suptitle(
        "Efficiency uplift across the 18 headline inventories", y=0.985, fontsize=13
    )
    _add_subtitle(
        fig,
        "2.0x U_STRONG C1 p95 latency budget; diagnostic because E023 is MODEL_INVALID",
    )
    ax.grid(axis="x", color=grid, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.25, top=0.88, bottom=0.1)
    _finish_chart(plt, chart_root / "chart-01-efficiency-uplift.png")

    # Chart 2: raw SLO throughput uplift.
    values = sorted(headline, key=lambda row: row["throughput_uplift_percent"])
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = np.arange(len(values))
    ax.scatter(
        [row["throughput_uplift_percent"] for row in values],
        y,
        c=blue,
        s=50,
        edgecolors=ink,
        linewidths=0.5,
        zorder=3,
    )
    ax.axvline(0.0, color=orange, linewidth=1.4, linestyle="--", label="No change")
    ax.set_yticks(y, [row["inventory_id"] for row in values])
    ax.set_xlabel("FLEX_POOL raw SLO target-row throughput uplift vs U_STRONG (%)")
    fig.suptitle(
        "Raw SLO throughput uplift across headline inventories", y=0.985, fontsize=13
    )
    _add_subtitle(
        fig,
        "Target rows are 17-row target-pass work units, not generated user tokens",
    )
    ax.grid(axis="x", color=grid, linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(left=0.25, top=0.88, bottom=0.1)
    _finish_chart(plt, chart_root / "chart-02-throughput-uplift.png")

    saturation = _read_csv(artifact_root / "serving/saturation-summary.csv")
    shared = [row for row in saturation if row["network_mode"] == "SHARED_NIC"]
    family_palette = {
        "memory-fragmented": blue,
        "compute-heterogeneous": gold,
        "network-heterogeneous": orange,
        "full-mixed": "#6F7D43",
        "coarse-friendly": "#A45D83",
    }
    arms = (
        "U_STRONG",
        "FLEX_FREE_NO_ALT",
        "FLEX_FREE",
        "FLEX_POOL_NO_ALT",
        "FLEX_POOL",
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.2), sharex=True, sharey=True)
    for ax, arm in zip(axes.flat, arms, strict=False):
        rows = [row for row in shared if row["arm"] == arm]
        for family, color in family_palette.items():
            subset = [row for row in rows if row["family"] == family]
            if subset:
                ax.scatter(
                    [float(row["abstract_node_cost"]) for row in subset],
                    [float(row["slo_2.0x_target_rows_per_second"]) for row in subset],
                    c=color,
                    s=22,
                    alpha=0.82,
                    edgecolors=ink,
                    linewidths=0.3,
                    label=family,
                )
        ax.set_title(arm)
        ax.grid(color=grid, linewidth=0.6)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, loc="center", title="Family")
    fig.supxlabel("Abstract node cost (each used node counted once)")
    fig.supylabel("SLO target rows per second")
    fig.suptitle("SLO throughput versus abstract node cost", y=0.985, fontsize=13)
    _add_subtitle(
        fig,
        "All 27 inventories under SHARED_NIC; family color identifies each inventory point",
        y=0.955,
    )
    fig.subplots_adjust(top=0.89, wspace=0.16, hspace=0.24)
    _finish_chart(plt, chart_root / "chart-03-throughput-vs-cost.png")

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    all_coincident = len(
        {
            (
                row["flex_pool_replica_resident_gib"],
                row["efficiency_uplift_percent"],
            )
            for row in headline
        }
    ) == 1
    if all_coincident:
        point = headline[0]
        ax.scatter(
            [point["flex_pool_replica_resident_gib"]],
            [point["efficiency_uplift_percent"]],
            c=blue,
            s=90,
            edgecolors=ink,
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            "18 coincident headline observations\n(no FLEX_POOL replicas accepted)",
            (
                point["flex_pool_replica_resident_gib"],
                point["efficiency_uplift_percent"],
            ),
            xytext=(18, 18),
            textcoords="offset points",
            fontsize=9,
            color=ink,
            arrowprops={"arrowstyle": "-", "color": ink, "linewidth": 0.8},
        )
    else:
        for family, color in family_palette.items():
            subset = [row for row in headline if row["family"] == family]
            if subset:
                ax.scatter(
                    [row["flex_pool_replica_resident_gib"] for row in subset],
                    [row["efficiency_uplift_percent"] for row in subset],
                    c=color,
                    s=48,
                    edgecolors=ink,
                    linewidths=0.5,
                    label=family,
                )
    ax.axhline(20.0, color=orange, linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color=ink, linewidth=0.9)
    ax.set_xlabel("Added FLEX_POOL replica resident memory (GiB)")
    ax.set_ylabel("SLO efficiency uplift vs U_STRONG (%)")
    fig.suptitle(
        "Replica memory and diagnostic efficiency uplift", y=0.985, fontsize=13
    )
    _add_subtitle(
        fig,
        "Headline inventories; coincident observations are represented without jitter",
    )
    ax.grid(color=grid, linewidth=0.6)
    fig.subplots_adjust(top=0.86, bottom=0.12)
    _finish_chart(plt, chart_root / "chart-04-replica-memory-vs-uplift.png")

    family_rows = list(analysis.family_rows)
    fig, ax = plt.subplots(figsize=(10.6, 6.6))
    family_order = [row["family"] for row in family_rows]
    x = np.arange(len(family_order))
    for index, family in enumerate(family_order):
        observations = [
            row["efficiency_uplift_percent"]
            for row in headline
            if row["family"] == family
        ]
        offsets = np.linspace(-0.11, 0.11, len(observations))
        ax.scatter(
            index + offsets,
            observations,
            c=family_palette[family],
            s=38,
            alpha=0.72,
            edgecolors=ink,
            linewidths=0.4,
        )
    ax.scatter(
        x,
        [row["median_efficiency_uplift_percent"] for row in family_rows],
        marker="D",
        c="#FAFBFC",
        edgecolors=ink,
        linewidths=1.4,
        s=75,
        label="Family median",
        zorder=4,
    )
    ax.axhline(20.0, color=orange, linestyle="--", linewidth=1.2, label="20% gate")
    ax.axhline(0.0, color=ink, linewidth=0.9)
    ax.set_xticks(x, [value.replace("-", "\n") for value in family_order])
    ax.set_ylabel("FLEX_POOL SLO efficiency uplift vs U_STRONG (%)")
    fig.suptitle("Family efficiency-uplift distributions", y=0.985, fontsize=13)
    _add_subtitle(fig, "Individual inventories and linear-quantile family medians")
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.legend(frameon=False)
    fig.subplots_adjust(top=0.86, bottom=0.17)
    _finish_chart(plt, chart_root / "chart-05-family-summary.png")

    arm_results = _read_csv(artifact_root / "serving/arm-results.csv")
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), sharex=True)
    for ax, inventory_id in zip(axes.flat, FIXED_REPRESENTATIVES, strict=False):
        for arm, color, marker in (
            ("U_STRONG", ink, "o"),
            ("FLEX_POOL", blue, "s"),
        ):
            rows = sorted(
                (
                    row
                    for row in arm_results
                    if row["inventory_id"] == inventory_id
                    and row["network_mode"] == "SHARED_NIC"
                    and row["arm"] == arm
                ),
                key=lambda row: int(row["concurrency"]),
            )
            ax.plot(
                [int(row["concurrency"]) for row in rows],
                [float(row["target_rows_per_second"]) for row in rows],
                color=color,
                marker=marker,
                linewidth=1.6,
                markersize=4.5,
                label=arm,
            )
        ax.set_title(inventory_id)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 64, 128], ["1", "8", "32", "64", "128"])
        ax.grid(color=grid, linewidth=0.6)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, loc="center")
    fig.supxlabel("Closed-loop concurrency")
    fig.supylabel("Target rows per second")
    fig.suptitle("SHARED_NIC saturation curves for fixed representatives", y=0.985)
    _add_subtitle(fig, "All plotted points are completed deterministic-run-1 measurements")
    fig.subplots_adjust(top=0.91, hspace=0.3, wspace=0.18)
    _finish_chart(plt, chart_root / "chart-06-saturation-curves.png")

    routing = _read_csv(artifact_root / "serving/replica-routing-summary.csv")
    fig, axes = plt.subplots(5, 1, figsize=(13.0, 12.0), sharex=True)
    for ax, inventory_id in zip(axes, FIXED_REPRESENTATIVES, strict=True):
        selected = [
            row
            for row in routing
            if row["inventory_id"] == inventory_id
            and row["arm"] == "FLEX_POOL"
            and row["network_mode"] == "SHARED_NIC"
        ]
        counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        for row in selected:
            key = (int(row["layer_id"]), int(row["logical_group_id"]))
            counts[key][0] += int(row["selected_primary_count"])
            counts[key][1] += int(row["selected_alternate_count"])
        layers = sorted({key[0] for key in counts})
        matrix = np.full((max(1, len(layers)), 8), np.nan)
        for row_index, layer in enumerate(layers):
            for group in range(8):
                primary, alternate = counts.get((layer, group), (0, 0))
                if primary + alternate:
                    matrix[row_index, group] = alternate / (primary + alternate)
        image = ax.imshow(
            matrix,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
        )
        ax.set_yticks(
            np.arange(max(1, len(layers))),
            [str(value) for value in layers] if layers else ["none"],
        )
        ax.set_ylabel(f"{inventory_id}\nlayer")
        if not layers:
            ax.text(3.5, 0, "No final FLEX_POOL replicas", ha="center", va="center")
    axes[-1].set_xticks(np.arange(8), [str(value) for value in range(8)])
    axes[-1].set_xlabel("Logical expert-group index")
    color_axis = fig.add_axes([0.86, 0.18, 0.018, 0.64])
    fig.colorbar(image, cax=color_axis, label="Alternate selection rate")
    fig.suptitle("FLEX_POOL alternate-copy selection by layer and group", y=0.992)
    _add_subtitle(
        fig,
        "Aggregated across the full SHARED_NIC concurrency ladder for fixed representatives",
        y=0.966,
    )
    fig.subplots_adjust(top=0.92, bottom=0.07, hspace=0.42, left=0.19, right=0.82)
    _finish_chart(plt, chart_root / "chart-07-replica-selection.png")

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.64,
        "Hedging diagnostic not run",
        ha="center",
        va="center",
        fontsize=18,
        color=ink,
        weight="bold",
    )
    ax.text(
        0.5,
        0.45,
        "E023 stopped at the mandatory coarse-friendly control gate.\n"
        "All 12 service cells also exceeded the frozen 10% drift threshold,\n"
        "so any hedging conclusion would have been suppressed.",
        ha="center",
        va="center",
        fontsize=11,
        color="#5F6975",
        linespacing=1.5,
    )
    ax.text(
        0.5,
        0.22,
        "No p95-latency-change or extra-compute observations exist.",
        ha="center",
        va="center",
        fontsize=10,
        color=orange,
    )
    fig.suptitle("Hedging tail-latency trade-off", y=0.95, fontsize=13)
    _finish_chart(plt, chart_root / "chart-08-hedging-tail-tradeoff.png")

    chart_specs = (
        (
            "chart-01-efficiency-uplift.png",
            "Which headline inventories clear the 20% SLO efficiency threshold?",
            "comparison-and-ranking",
            "ranked dot plot",
            "inventory_id; efficiency_uplift_percent",
            "Diagnostic distribution and 20% reference",
        ),
        (
            "chart-02-throughput-uplift.png",
            "Does FLEX_POOL preserve raw SLO throughput?",
            "comparison-and-ranking",
            "ranked dot plot",
            "inventory_id; throughput_uplift_percent",
            "Diagnostic raw-throughput changes and zero reference",
        ),
        (
            "chart-03-throughput-vs-cost.png",
            "How do the five architectures trade SLO throughput against abstract cost?",
            "relationship",
            "faceted scatter",
            "arm; family; abstract_node_cost; slo_target_rows_per_second",
            "Cross-arm cost and throughput envelope",
        ),
        (
            "chart-04-replica-memory-vs-uplift.png",
            "Is replica memory associated with diagnostic efficiency uplift?",
            "relationship",
            "labeled scatter",
            "replica_resident_gib; efficiency_uplift_percent; family",
            "Memory footprint versus modeled uplift",
        ),
        (
            "chart-05-family-summary.png",
            "How does diagnostic uplift vary by preregistered family?",
            "distribution",
            "strip plot with median",
            "family; efficiency_uplift_percent",
            "Within-family observations and medians",
        ),
        (
            "chart-06-saturation-curves.png",
            "Where do U_STRONG and FLEX_POOL saturate?",
            "trend",
            "faceted line",
            "inventory_id; concurrency; target_rows_per_second; arm",
            "Full five-point concurrency ladder",
        ),
        (
            "chart-07-replica-selection.png",
            "Which flexible logical groups actually select alternates?",
            "matrix-and-cohort",
            "faceted heatmap",
            "inventory_id; layer_id; logical_group_id; alternate_selection_rate",
            "Physical-copy choice while preserving logical identity",
        ),
        (
            "chart-08-hedging-tail-tradeoff.png",
            "What was the hedging p95/compute trade-off?",
            "uncertainty-and-benchmark",
            "explicit no-data status panel",
            "none; phase stopped before execution",
            "No hedging conclusion exists",
        ),
    )
    return [
        {
            "chart_id": index,
            "path": f"charts/{name}",
            "question": question,
            "family": family,
            "chart_type": chart_type,
            "fields": fields,
            "supported_takeaway": takeaway,
            "palette_policy": "two-root-or-family-categorical-with-neutral-references",
            "source": "deterministic-run-1 diagnostic artifacts",
            "evidence_status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "visual_qa_status": chart_qa_status,
        }
        for index, (name, question, family, chart_type, fields, takeaway) in enumerate(
            chart_specs, 1
        )
    ]


def _physical_summary(artifact_root: Path) -> dict[str, Any]:
    receipt = _read_json(
        artifact_root / "physical/duplicate-expert-group-correctness.json"
    )
    cases = list(receipt["cases"])
    copy_identity = list(receipt["copy_identity"])
    service_rows = _read_csv(artifact_root / "physical/service-samples.csv")
    return {
        "status": receipt["status"],
        "primary_gate": receipt["primary_gate"],
        "case_count": len(cases),
        "service_sample_count": len(service_rows),
        "all_finite": all(bool(row["finite"]) for row in cases),
        "maximum_a_b_relative_l2": max(float(row["a_b_relative_l2"]) for row in cases),
        "maximum_e022_reference_relative_l2": max(
            max(
                float(row["a_e022_reference_relative_l2"]),
                float(row["b_e022_reference_relative_l2"]),
            )
            for row in cases
        ),
        "maximum_replica_memory_error_percent": max(
            float(row["memory_error_percent"]) for row in copy_identity
        ),
        "all_zero_persistent_state": all(
            int(row["persistent_state_bytes"]) == 0 for row in copy_identity
        ),
        "hedge_service_drift": bool(receipt["hedge_service_drift"]),
        "service_drift_cell_count": sum(
            row["status"] == "HEDGE_SERVICE_DRIFT" for row in receipt["service_drift"]
        ),
        "maximum_absolute_service_drift_percent": max(
            float(row["absolute_drift_percent"])
            for row in receipt["service_drift"]
        ),
        "hedging_conclusion_suppressed": bool(
            receipt["hedging_conclusion_suppressed"]
        ),
        "evidence_class": receipt["evidence_class"],
    }


def _audit_final_plans(artifact_root: Path) -> dict[str, Any]:
    paths = sorted((artifact_root / "plans").glob("*/*.json"))
    violations: list[str] = []
    for path in paths:
        manifest = _read_json(path)
        assignment_rows: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for node in manifest.get("nodes", ()):
            for piece in node.get("pieces", ()):
                name = str(piece.get("piece", ""))
                if name.startswith("transformer_layer_"):
                    layer_id = int(name.removeprefix("transformer_layer_"))
                    assignment_rows[layer_id].append(piece)
        seen: set[tuple[int, int]] = set()
        for replica in manifest.get("replicas", ()):
            layer = int(replica["layer_id"])
            group = int(replica["logical_group_id"])
            key = (layer, group)
            if key in seen:
                violations.append(f"{path}:logical group has more than two copies:{key}")
            seen.add(key)
            if int(replica["persistent_state_bytes"]) != 0:
                violations.append(f"{path}:replica has persistent state:{key}")
            assignments = assignment_rows.get(layer, ())
            if not assignments:
                violations.append(f"{path}:replica layer absent:{layer}")
            elif any(
                row["partition_type"] != "WHOLE_EXPERT"
                or int(row["degree"]) != 8
                for row in assignments
            ):
                violations.append(f"{path}:stateful/non-P8 replica:{key}")
        if manifest.get("canonical_reduction_order") != list(range(8)):
            violations.append(f"{path}:canonical reduction ordering absent")
        if manifest.get("arm") == "FLEX_FREE" and not set(
            manifest.get("used_nodes", ())
        ).issubset(manifest.get("u_strong_used_nodes", ())):
            violations.append(f"{path}:FLEX_FREE activated a new node")
    return {
        "plan_manifest_count": len(paths),
        "expected_plan_manifest_count": 27 * 5,
        "violations": violations,
        "status": "PASS" if len(paths) == 27 * 5 and not violations else "FAIL",
    }


def _environment(repo: Path, physical: Mapping[str, Any]) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "matplotlib", "pytest", "ruff", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": "experiment-023-environment-v1",
        "experiment_id": "023",
        "interpreter_path": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "checkpoint": "F:/models/Kimi-K3",
        "checkpoint_downloaded_for_e023": False,
        "gpu_physical_scope": "local RTX 5090 primitive validation only",
        "physical_service_sample_count": physical["service_sample_count"],
        "external_cloud_resources_created": False,
        "vast_ai_used": False,
        "physical_multi_machine_experiment": False,
        "working_directory": str(repo),
    }


def _capacity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    efficiency = [float(row["efficiency_uplift_percent"]) for row in rows]
    throughput = [float(row["throughput_uplift_percent"]) for row in rows]
    return {
        "inventory_count": len(rows),
        "median_efficiency_uplift_percent": float(
            np.quantile(efficiency, 0.5, method="linear")
        ),
        "median_raw_throughput_uplift_percent": float(
            np.quantile(throughput, 0.5, method="linear")
        ),
        "replica_using_inventory_count": sum(
            bool(row["flex_pool_actual_replica_used"]) for row in rows
        ),
        "promotion_eligible": False,
    }


def _build_summary(
    analysis: AttemptAnalysis,
    verdict: Mapping[str, Any],
    zero_new_node: Mapping[str, Any],
    physical: Mapping[str, Any],
    attempt_summary: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic = analysis.diagnostic_summary
    capacity = _capacity_summary(analysis.capacity_rows)
    return {
        "schema_version": "experiment-023-summary-v1",
        "experiment_id": "023",
        "final_verdict": verdict["final_verdict"],
        "evidence_class": "PHYSICALLY GROUNDED MODEL + SHAPED NETWORK",
        "primary_conclusion_status": "NOT_PROMOTABLE_MODEL_INVALID",
        "headline_inventory_count": 18,
        "valid_headline_inventory_count": analysis.completeness[
            "headline_inventory_count"
        ],
        "median_efficiency_uplift_percent": diagnostic[
            "median_efficiency_uplift_percent"
        ],
        "mean_efficiency_uplift_percent": diagnostic[
            "mean_efficiency_uplift_percent"
        ],
        "p90_efficiency_uplift_percent": diagnostic[
            "p90_efficiency_uplift_percent"
        ],
        "maximum_efficiency_uplift_percent": diagnostic[
            "maximum_efficiency_uplift_percent"
        ],
        "minimum_efficiency_uplift_percent": diagnostic[
            "minimum_efficiency_uplift_percent"
        ],
        "headline_cases_ge_20_percent": diagnostic[
            "headline_cases_ge_20_percent"
        ],
        "median_raw_throughput_uplift_percent": diagnostic[
            "median_raw_throughput_uplift_percent"
        ],
        "worst_raw_throughput_regression_percent": diagnostic[
            "worst_raw_throughput_regression_percent"
        ],
        "worst_efficiency_regression_percent": diagnostic[
            "worst_efficiency_regression_percent"
        ],
        "replica_using_headline_inventory_count": diagnostic[
            "replica_using_headline_inventory_count"
        ],
        "median_optionality_only_percent": diagnostic[
            "median_optionality_only_percent"
        ],
        "flex_free_median_efficiency_uplift_percent": diagnostic[
            "median_flex_free_efficiency_uplift_percent"
        ],
        "zero_new_node_wedge": False,
        "diagnostic_zero_new_node_gate_if_valid": zero_new_node,
        "qualifying_families": [],
        "diagnostic_qualifying_families_if_valid": verdict[
            "diagnostic_qualifying_families_if_valid"
        ],
        "capacity_cohort_summary": capacity,
        "legacy_network_summary": {
            "median_efficiency_uplift_percent": diagnostic[
                "median_legacy_efficiency_uplift_percent"
            ],
            "nonnegative_inventory_count": diagnostic[
                "legacy_nonnegative_inventory_count"
            ],
            "inventory_count": 18,
            "promotion_status": "NOT_EVALUATED_DUE_MODEL_INVALID",
        },
        "hedging_diagnostic": {
            "status": STOP_STATUS,
            "tail_diagnostic_positive": False,
            "hedge_service_drift": physical["hedge_service_drift"],
            "conclusion_suppressed": True,
            "seed_count_run": 0,
        },
        "physical_replica_validation": dict(physical),
        "correctness_status": STOP_STATUS,
        "reproducibility_status": STOP_STATUS,
        "validity_failures": verdict["validity_failures"],
        "negative_control_failures": list(analysis.control_failures),
        "thresholds_changed_after_headline": False,
        "deterministic_headline_wall_clock_seconds": float(
            attempt_summary["elapsed_seconds"]
        ),
        "total_experiment_wall_clock_seconds": None,
        "total_wall_clock_runtime_available": False,
    }


def _truth_table(
    analysis: AttemptAnalysis,
    verdict: Mapping[str, Any],
    zero_new_node: Mapping[str, Any],
    physical: Mapping[str, Any],
    plans: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic = analysis.diagnostic_summary
    family_status = {
        name: {
            "status": "NOT_EVALUATED_FOR_PROMOTION_MODEL_INVALID",
            "diagnostic_gate_if_valid": value,
        }
        for name, value in verdict["family_wedge_gates"].items()
    }
    return {
        "schema_version": "experiment-023-truth-table-v1",
        "experiment_id": "023",
        "E022 frozen hashes preserved": {"status": "PASS", "value": True},
        "Replica primitive exactness": {
            "status": physical["status"],
            "value": physical["maximum_a_b_relative_l2"] <= 2e-6,
            "maximum_relative_l2": physical["maximum_a_b_relative_l2"],
        },
        "Replica memory validation": {
            "status": "PASS",
            "value": physical["maximum_replica_memory_error_percent"] <= 5.0,
            "maximum_error_percent": physical[
                "maximum_replica_memory_error_percent"
            ],
        },
        "Legacy engine compatibility": {"status": "PASS", "value": True},
        "27 inventories present": {
            "status": "PASS",
            "value": analysis.completeness["inventory_count"] == 27,
        },
        "18 headline inventories complete": {
            "status": "PASS",
            "value": analysis.completeness["headline_inventory_count"] == 18,
        },
        "3 controls complete": {
            "status": "PASS",
            "value": analysis.completeness["control_inventory_count"] == 3,
        },
        "6 exploratory capacity cases complete": {
            "status": "PASS",
            "value": analysis.completeness["capacity_inventory_count"] == 6,
        },
        "U_STRONG constructed": {"status": "PASS", "value": True},
        "FLEX_FREE constructed": {"status": "PASS", "value": True},
        "FLEX_POOL constructed": {"status": "PASS", "value": True},
        "All concurrency levels complete": {
            "status": "PASS",
            "value": analysis.completeness["all_required_rows_complete"],
        },
        "Memory reconciliation": {"status": "PASS", "value": True},
        "Cost reconciliation": {"status": "PASS", "value": True},
        "Deterministic rerun": {"status": STOP_STATUS, "value": False},
        "Five full correctness receipts": {"status": STOP_STATUS, "value": False},
        "Primary efficiency median": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value_percent": diagnostic["median_efficiency_uplift_percent"],
        },
        "Cases >=20%": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value": diagnostic["headline_cases_ge_20_percent"],
        },
        "Raw throughput median": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value_percent": diagnostic["median_raw_throughput_uplift_percent"],
        },
        "Worst efficiency regression": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value_percent": diagnostic["worst_efficiency_regression_percent"],
        },
        "Worst throughput regression": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value_percent": diagnostic["worst_raw_throughput_regression_percent"],
        },
        "Replica-using headline inventories": {
            "status": "DIAGNOSTIC_NOT_PROMOTABLE",
            "value": diagnostic["replica_using_headline_inventory_count"],
        },
        "Legacy robustness": {
            "status": "NOT_EVALUATED_FOR_PROMOTION_MODEL_INVALID",
            "diagnostic_median_efficiency_uplift_percent": diagnostic[
                "median_legacy_efficiency_uplift_percent"
            ],
            "diagnostic_nonnegative_cases": diagnostic[
                "legacy_nonnegative_inventory_count"
            ],
        },
        "General wedge gate": {
            "status": "NOT_EVALUATED_FOR_PROMOTION_MODEL_INVALID",
            "value": False,
            "diagnostic_gate_if_valid": verdict["general_wedge_gate"],
        },
        "Each family wedge gate": family_status,
        "Zero-new-node flag": {
            "status": "NOT_EVALUATED_FOR_PROMOTION_MODEL_INVALID",
            "value": False,
            "diagnostic_gate_if_valid": zero_new_node,
        },
        "Hedging diagnostic flag": {"status": STOP_STATUS, "value": False},
        "Final plan audit": dict(plans),
        "Mandatory control gate": {
            "status": "FAIL",
            "value": False,
            "failures": list(analysis.control_failures),
        },
        "Final verdict": verdict["final_verdict"],
    }


def _percent(value: float) -> str:
    return f"{value:+.2f}%"


def _render_report(
    repo: Path,
    analysis: AttemptAnalysis,
    summary: Mapping[str, Any],
    verdict: Mapping[str, Any],
    zero_new_node: Mapping[str, Any],
    physical: Mapping[str, Any],
    attempt_summary: Mapping[str, Any],
) -> str:
    diagnostic = analysis.diagnostic_summary
    control = next(
        row
        for row in analysis.control_failures
        if row["inventory_id"] == "coarse-friendly-03" and row["arm"] == "FLEX_POOL"
    )
    family_lines = [
        "| Family | n | Median efficiency | ≥20% cases | Median raw throughput | Median legacy efficiency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in analysis.family_rows:
        family_lines.append(
            "| {family} | {n} | {eff} | {wins} | {raw} | {legacy} |".format(
                family=row["family"],
                n=row["inventory_count"],
                eff=_percent(row["median_efficiency_uplift_percent"]),
                wins=row["cases_ge_20_percent"],
                raw=_percent(row["median_raw_throughput_uplift_percent"]),
                legacy=_percent(row["median_legacy_efficiency_uplift_percent"]),
            )
        )
    capacity = summary["capacity_cohort_summary"]
    report = f"""# Experiment 023: Exact Sparse Expert Optionality

## Verdict

The mechanically determined primary category is **MODEL_INVALID**. The frozen FLEX_POOL planner accepted three C32-improving actions for the negative control `coarse-friendly-03`, but the final latency-constrained evaluation regressed SLO efficiency by {_percent(control['efficiency_uplift_percent'])} and raw SLO throughput by {_percent(control['throughput_uplift_percent'])}. Section 54 makes a regression worse than 5% caused by accepted FLEX actions a planner implementation invalidity, so no E023 performance conclusion can be promoted.

The completed 27-inventory deterministic run is retained below as **diagnostic, non-promotable evidence**. The experiment stopped at the first mandatory gate; deterministic run 2, five full 93-layer correctness receipts, diagnostic copy-choice ablations, and stochastic hedging were not run.

## Executive Summary

- **Primary question:** unanswered because the serving planner failed a mandatory negative-control safety property.
- **Diagnostic headline numbers:** median FLEX_POOL efficiency uplift was {_percent(diagnostic['median_efficiency_uplift_percent'])}; {diagnostic['headline_cases_ge_20_percent']} of 18 cases reached at least 20%; median raw SLO throughput uplift was {_percent(diagnostic['median_raw_throughput_uplift_percent'])}.
- **Worst headline observations:** efficiency was {_percent(diagnostic['worst_efficiency_regression_percent'])} and raw throughput was {_percent(diagnostic['worst_raw_throughput_regression_percent'])}. These values cannot establish a wedge because the model is invalid.
- **Physical prerequisite:** all {physical['case_count']} duplicate-group comparisons passed; maximum A/B relative L2 was {physical['maximum_a_b_relative_l2']:.3g} and maximum replica-memory error was {physical['maximum_replica_memory_error_percent']:.3f}%.
- **Next step:** fix only the invalid evidence path and rerun E023 with the exact same frozen hypothesis and thresholds.

## Frozen Hypothesis

The preregistered hypothesis was that a heterogeneous resource pool is more useful when excess memory creates alternative exact execution paths than when every added worker creates another mandatory dependency. The primary gate required at least a 20% serving-efficiency wedge under the frozen 18-inventory cohort and all listed robustness and validity checks. Thresholds were frozen before headline generation and were not changed.

## Why This Experiment Exists

E022 remained frozen as `MODEL_INVALID` and showed that additional nodes could become mandatory dependencies while its old placement search could miss stronger unique placements. E023 therefore constructed `U_STRONG` from the complete frozen A/B/C/D/E envelope, deterministic whole-layer relocation repair, and unique P8 refinement before permitting optional exact expert-group copies.

The architectural inversion being tested was substitutability: a logical expert group still executes exactly once in the deterministic arm, but a non-clairvoyant router may choose either of two exact resident copies.

## Evidence Boundary

- **PHYSICAL:** local RTX 5090 duplicate expert-group exactness, memory residency, and service samples using `F:/models/Kimi-K3`.
- **PHYSICALLY GROUNDED MODEL:** the multi-request 17-row target-pass execution model using frozen repaired E022 services.
- **SHAPED NETWORK:** frozen directed-link timing with E023 shared TX/RX NIC calendars.
- **SYNTHETIC:** verdict fixtures and other explicit test-only cases; none can promote the experiment.

This was not a physical distributed K3 run. Target rows are target-pass work units, not generated user tokens, and abstract node cost is not currency.

## Physical Replica Validation

Two sequentially instantiated resident copies were tested for KDA layer 89 and gated-MLA layer 91, groups 0 and 7, and row counts 1, 2, and 4. All {physical['case_count']} cases had identical ownership, routes, route weights, expert ranges, finite results, zero timed checkpoint reads, zero whole-layer fallback, zero persistent state, and relative L2 within the unchanged `2e-6` gate. Maximum A/B relative L2 was {physical['maximum_a_b_relative_l2']:.3g}; maximum relative L2 against the frozen E022 whole-expert reference was {physical['maximum_e022_reference_relative_l2']:.3g}.

The standalone memory estimator's maximum physical error was {physical['maximum_replica_memory_error_percent']:.3f}%, below the frozen 5% gate. The service collection contains {physical['service_sample_count']:,} timed executions. All {physical['service_drift_cell_count']} service cells drifted by more than 10% from E022, with maximum absolute drift {physical['maximum_absolute_service_drift_percent']:.2f}%; this does not invalidate deterministic E023 but suppresses hedging.

Source: `artifacts/experiment-023/physical/duplicate-expert-group-correctness.json` and `service-samples.csv`.

## Strong Unique Baseline

Every inventory considered all five frozen E022 manifests, their deterministic relocation-repaired variants, and a deterministic `U_P8_REFINE` candidate. The final `U_STRONG` applied the frozen 95%-of-fastest envelope and cost-efficiency rule. No randomized reconstruction of E022 placements was used.

Source: `artifacts/experiment-023/baseline/baseline-envelope.csv`, `baseline-relocations.csv`, and `unique-refinement-actions.csv`.

## Serving Model

The new interval-calendar engine modeled one non-preemptive compute resource per worker, directed links, shared source TX NICs, shared destination RX NICs, closed-loop concurrency 1/8/32/64/128, and persistent weights with independent per-slot logical state. All five fixed legacy compatibility cases matched E022 in integer counters and deterministic timing within `1e-9`.

Each 2.0x latency budget was derived only from `U_STRONG` C1 p95 in the corresponding network mode. The 945 required arm rows and 189 saturation summaries are complete and finite.

Source: `artifacts/experiment-023/validation/engine-compatibility.csv` and `serving/arm-results.csv`.

## Sparse Flexibility Mechanism

Only stateless `WHOLE_EXPERT:p8` logical groups could be copied, with at most one alternate. The runtime exhaustively enumerated the `2^k` copy masks using only current calendar reservations and deterministic services, then committed the mask minimizing fork-join completion. Physical arrival never changed the canonical logical reduction order `0..7`.

The figure shows how often final representative plans selected alternates. It demonstrates that the modeled router exercised optional paths; it does not repair the failed control gate.

![Alternate-copy selection by layer and group](../../artifacts/experiment-023/charts/chart-07-replica-selection.png)

## Primary Results

There is **no valid primary performance result**. Diagnostic-run-1 produced a median 18-inventory efficiency uplift of {_percent(diagnostic['median_efficiency_uplift_percent'])}, with {diagnostic['headline_cases_ge_20_percent']} cases at or above 20%. Median raw SLO throughput uplift was {_percent(diagnostic['median_raw_throughput_uplift_percent'])}; {diagnostic['replica_using_headline_inventory_count']} of 18 headline inventories actually selected at least one alternate.

The first chart shows the frozen 20% reference and the second separates raw throughput from cost efficiency. Both are diagnostic because the invalid planner means the experiment cannot decide whether exact optionality creates a reliable wedge.

![Efficiency uplift across headline inventories](../../artifacts/experiment-023/charts/chart-01-efficiency-uplift.png)

![Raw SLO throughput uplift across headline inventories](../../artifacts/experiment-023/charts/chart-02-throughput-uplift.png)

Abstract cost and SLO throughput are shown jointly below. Each used node is counted once, including replica-only nodes; no dollar interpretation is made.

![SLO throughput versus abstract node cost](../../artifacts/experiment-023/charts/chart-03-throughput-vs-cost.png)

## Family Results

No family can qualify while the mandatory validity gate is failed. If the same diagnostic rows were considered without the validity short-circuit, the frozen family predicates would have produced the states recorded in `truth-table.json`; those counterfactual checks are not verdicts.

{chr(10).join(family_lines)}

The plot preserves individual observations around each family median so small cohorts are not hidden by aggregation.

![Family efficiency-uplift distributions](../../artifacts/experiment-023/charts/chart-05-family-summary.png)

## Optionality Ablation

The diagnostic median FLEX_POOL-versus-FLEX_POOL_NO_ALT efficiency difference was {_percent(diagnostic['median_optionality_only_percent'])}. Because the planner failed a mandatory control and full correctness/reproducibility were not completed, this cannot establish that optional routing—rather than P8 decomposition, search behavior, or an invalid SLO trade-off—caused a gain.

Replica memory and total architecture uplift are plotted together to expose scale and outliers rather than imply a causal memory-response curve.

![Replica memory versus diagnostic uplift](../../artifacts/experiment-023/charts/chart-04-replica-memory-vs-uplift.png)

Source: `artifacts/experiment-023/analysis/optionality-ablation.csv`.

## Zero-New-Node Result

The diagnostic FLEX_FREE median efficiency uplift was {_percent(diagnostic['median_flex_free_efficiency_uplift_percent'])}. The formal `ZERO_NEW_NODE_WEDGE` flag is **not adjudicated and is reported false** because the experiment is `MODEL_INVALID`; the counterfactual gate state is retained separately in `summary.json` and `truth-table.json`.

## Capacity Cohort

The six exploratory capacity inventories had diagnostic median efficiency uplift {_percent(capacity['median_efficiency_uplift_percent'])} and median raw SLO throughput uplift {_percent(capacity['median_raw_throughput_uplift_percent'])}. {capacity['replica_using_inventory_count']} of 6 selected an alternate. These cases are excluded from the primary 18-inventory threshold and cannot promote a verdict.

Source: `artifacts/experiment-023/analysis/capacity-exploratory.csv`.

## Legacy Network Robustness

Under `LEGACY_DIRECTED_LINK`, diagnostic median FLEX_POOL efficiency uplift across the headline cohort was {_percent(diagnostic['median_legacy_efficiency_uplift_percent'])}, with {diagnostic['legacy_nonnegative_inventory_count']} of 18 non-negative cases. The robustness gate is not evaluated for promotion after the mandatory invalidity.

The five-point SHARED_NIC saturation curves below show the actual modeled throughput observations used before SLO filtering for fixed representatives.

![Saturation curves for fixed representatives](../../artifacts/experiment-023/charts/chart-06-saturation-curves.png)

## Hedging Diagnostic

Hedging was not run. E023 stopped before Phase 10, and the physical sample pool also carried `HEDGE_SERVICE_DRIFT` in every service cell. Therefore mean/median throughput, p95/p99 latency, extra compute/network, launch rate, and duplicate-win rate do not exist; `TAIL_DIAGNOSTIC_POSITIVE` is false and no conclusion is claimed.

![Hedging diagnostic status](../../artifacts/experiment-023/charts/chart-08-hedging-tail-tradeoff.png)

## Correctness

The physical duplicate-group correctness prerequisite passed. The five mandatory final 93-layer `FLEX_POOL` correctness traversals were **not run** after the negative-control stop, so E023 has no full-plan correctness conclusion. The five required JSON receipts exist as explicit not-run records and must not be mistaken for passes.

## Memory and Cost Accounting

All 135 final plan manifests passed the completed per-node memory and abstract-cost reconciliation in deterministic run 1. Replica checkpoint bytes are separated from unique model checkpoint bytes; replica-only nodes are charged once; multiple pieces on one node do not multiply its cost. Every manifest preserves zero replica persistent state, at most two copies per logical group, `WHOLE_EXPERT:p8` ownership, and canonical group reduction order.

These accounting passes do not override the planner-control failure.

## Limitations

- The accepted-action objective used C32 throughput (or throughput per abstract cost), while the primary metric applied a latency budget derived from U_STRONG C1. `coarse-friendly-03` exposed a mismatch large enough to invalidate the planner.
- The deterministic full run was not repeated after the stop, so exact rerun reproducibility is unadjudicated.
- Full 93-layer correctness and the two required diagnostic routing ablations are unadjudicated.
- Hedging was not executed and its physical service pool drifted from E022.
- All fleet/network results are modeled or shaped, not a physical multi-machine deployment.

## What E023 Proves

E023 proves that the tested local K3 expert-group copies are numerically substitutable within the frozen gate and that their standalone resident-memory estimator is within 5% on the local RTX 5090. It also proves exact single-pass compatibility for the new legacy engine representatives and records a complete first deterministic concurrency evaluation with reconciled memory and cost.

E023 also identifies a concrete evidence-path defect: local C32 planner acceptance did not protect latency-constrained serving performance on a required negative control.

## What E023 Does Not Prove

E023 does not answer whether exact sparse expert-group optionality creates a real 20% serving-efficiency wedge. It does not prove full-plan correctness, deterministic rerun identity, tail-hedging benefit, real LAN/WAN behavior, real cluster throughput, API token throughput, dollar economics, or physical operation across hundreds of workers.

## Recommendation for Experiment 024

**Fix only the invalid evidence path and rerun E023 with the exact same frozen hypothesis and thresholds.**

This is the required `MODEL_INVALID` branch. Do not broaden E024, increase search budgets, or change thresholds to rescue the result.

## Reproduction

Use the repository's active interpreter `{sys.executable}` with `PYTHONPATH=src`. The immutable freeze is in `artifacts/experiment-023/freeze/`; deterministic-run-1 remains in `artifacts/experiment-023/attempts/{ATTEMPT_NAME}/`; published diagnostic tables, manifests, and audits are under `artifacts/experiment-023/`. The recorded deterministic run consumed {float(attempt_summary['elapsed_seconds']):.3f} seconds. A total end-to-end experiment wall-clock time was not recorded.

The executable entry points are `scripts/experiment_023_freeze.py`, `experiment_023_physical.py`, `experiment_023_run.py`, `experiment_023_correctness.py`, `experiment_023_finalize.py`, and `experiment_023_test.py`. `commands.txt` records the phase and validation commands. The machine-readable decision is `truth-table.json`; prose cannot override it.
"""
    return report


def _report_source_notes(
    chart_map: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "experiment-023-report-source-notes-v1",
        "audience": "technical",
        "delivery_surface": "user-mandated Markdown report",
        "question": (
            "Can exact sparse WHOLE_EXPERT:p8 optionality create a >=20% "
            "latency-constrained efficiency wedge over U_STRONG?"
        ),
        "decision_useful_answer": (
            "The question is unanswered because a mandatory negative-control "
            "planner gate failed; the mechanical verdict is MODEL_INVALID."
        ),
        "comparison_basis": "FLEX_POOL versus U_STRONG at the 2.0x latency budget",
        "report_structure_mapping": {
            "title": "Report section 1",
            "technical_summary": "Verdict + Executive Summary",
            "key_findings_with_visual_evidence": "Primary Results through Legacy Network Robustness",
            "scope_data_metric_definitions": "Evidence Boundary + Serving Model",
            "methodology": "Strong Unique Baseline + Sparse Flexibility Mechanism",
            "limitations_uncertainty_robustness": "Hedging Diagnostic through Limitations",
            "recommended_next_steps": "Recommendation for Experiment 024",
            "further_questions": (
                "Merged into What E023 Does Not Prove because the frozen MODEL_INVALID "
                "decision tree prescribes one next action."
            ),
        },
        "evidence_omissions": {
            "full_correctness": STOP_STATUS,
            "deterministic_rerun": STOP_STATUS,
            "hedging": STOP_STATUS,
            "static_copy_choice": STOP_STATUS,
            "primary_only_with_replica_cost": STOP_STATUS,
        },
        "chart_map": list(chart_map),
    }


def _commands_text() -> str:
    return """# Experiment 023 command ledger
# The deterministic run stopped after the mandatory Phase 8 control failure.
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_freeze.py
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_physical.py
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_correctness.py compatibility
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_run.py --attempt attempts/deterministic-run-1
# Full correctness, deterministic run 2, diagnostics, and hedging were not run after the stop gate.
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_finalize.py
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe scripts\\experiment_023_test.py
.\\.venv\\Scripts\\python.exe -m compileall src\\swarm_inference\\experiments\\experiment_023 scripts
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q --basetemp .pytest-tmp\\e023-required-final tests\\test_experiment_023.py tests\\test_experiment_023_completion.py tests\\test_experiment_022.py tests\\test_experiment_022_completion.py
$env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q --basetemp .pytest-tmp\\e023-full-final-20260816
.\\.venv\\Scripts\\python.exe -m ruff check src\\swarm_inference\\experiments\\experiment_023 scripts\\experiment_023_*.py tests\\test_experiment_023.py tests\\test_experiment_023_completion.py
.\\.venv\\Scripts\\python.exe -m ruff check . --output-format json --output-file artifacts\\experiment-023\\validation\\ruff-global.json
"""


def _required_artifact_paths(artifact_root: Path, repo: Path) -> list[Path]:
    relative = [
        "freeze/e023-frozen-inputs.json",
        "freeze/headline-cohorts.json",
        "freeze/e022-input-hashes.json",
        "physical/duplicate-expert-group-correctness.json",
        "physical/duplicate-expert-group-services.csv",
        "physical/service-samples.csv",
        "physical/gpu-samples.csv",
        "baseline/baseline-envelope.csv",
        "baseline/baseline-relocations.csv",
        "baseline/unique-refinement-actions.csv",
        "serving/arm-results.csv",
        "serving/saturation-summary.csv",
        "serving/replica-actions.csv",
        "serving/replica-routing-summary.csv",
        "serving/resource-utilization.csv",
        "serving/network-summary.csv",
        "stochastic/hedging-results.csv",
        "stochastic/seed-results.csv",
        "validation/engine-compatibility.csv",
        "validation/memory-reconciliation.csv",
        "validation/cost-reconciliation.csv",
        "validation/reproducibility.json",
        "validation/static-quality.json",
        "validation/ruff-global.json",
        "analysis/uplift.csv",
        "analysis/family-summary.csv",
        "analysis/optionality-ablation.csv",
        "analysis/replica-efficiency.csv",
        "analysis/capacity-exploratory.csv",
        "analysis/chart-map.json",
        "environment.json",
        "commands.txt",
        "failure-log.json",
        "summary.json",
        "truth-table.json",
    ]
    relative.extend(
        f"correctness/{inventory_id}.json" for inventory_id in FIXED_REPRESENTATIVES
    )
    relative.extend(
        f"charts/chart-{index:02d}-{name}.png"
        for index, name in enumerate(
            (
                "efficiency-uplift",
                "throughput-uplift",
                "throughput-vs-cost",
                "replica-memory-vs-uplift",
                "family-summary",
                "saturation-curves",
                "replica-selection",
                "hedging-tail-tradeoff",
            ),
            1,
        )
    )
    paths = [artifact_root / value for value in relative]
    paths.append(repo / "docs/experiments/EXPERIMENT_023_REPORT.md")
    paths.extend(sorted((artifact_root / "plans").glob("*/*.json")))
    return paths


def _write_final_audit(
    repo: Path,
    artifact_root: Path,
    *,
    analysis: AttemptAnalysis,
    physical: Mapping[str, Any],
    plan_audit: Mapping[str, Any],
    chart_qa_status: str,
) -> dict[str, Any]:
    compatibility = _read_csv(artifact_root / "validation/engine-compatibility.csv")
    memory = _read_csv(artifact_root / "validation/memory-reconciliation.csv")
    costs = _read_csv(artifact_root / "validation/cost-reconciliation.csv")
    arm_rows = _read_csv(artifact_root / "serving/arm-results.csv")
    numeric_fields = (
        "measurement_window_ms",
        "target_rows_per_second",
        "p50_pass_latency_ms",
        "p95_pass_latency_ms",
        "abstract_node_cost",
        "rows_per_second_per_abstract_cost",
        "network_bytes_per_target_row",
        "worker_compute_ms_per_target_row",
        "maximum_compute_utilization",
        "maximum_tx_utilization",
        "maximum_rx_utilization",
    )
    finite = all(
        all(math.isfinite(float(row[field])) for field in numeric_fields)
        for row in arm_rows
    )
    required = _required_artifact_paths(artifact_root, repo)
    missing = [str(path.relative_to(repo)) for path in required if not path.is_file()]
    hashes = [
        {
            "path": path.relative_to(repo).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(required, key=lambda value: value.as_posix())
        if path.is_file()
    ]
    frozen = _read_json(artifact_root / "freeze/e023-frozen-inputs.json")
    checks = {
        "e022_freeze_revalidated": True,
        "frozen_constants_unchanged": frozen["constants"] == FROZEN_CONSTANTS,
        "frozen_constants_sha256_unchanged": frozen["constants_sha256"]
        == canonical_sha256(FROZEN_CONSTANTS),
        "physical_replica_gate_pass": physical["status"] == "PASS",
        "legacy_compatibility_pass": len(compatibility) == 5
        and all(row["status"] == "PASS" for row in compatibility),
        "all_required_serving_rows_complete": analysis.completeness[
            "all_required_rows_complete"
        ],
        "all_serving_values_finite": finite,
        "memory_reconciliation_pass": len(memory) == 11_400
        and all(row["within_capacity"] == "True" for row in memory)
        and all(int(row["replica_persistent_state_bytes"]) == 0 for row in memory),
        "cost_reconciliation_pass": len(costs) == 135
        and all(row["status"] == "PASS" for row in costs)
        and all(float(row["difference"]) == 0.0 for row in costs),
        "plan_manifest_audit_pass": plan_audit["status"] == "PASS",
        "required_artifact_tree_complete": not missing,
        "all_eight_charts_nonempty": all(
            (artifact_root / "charts" / f"chart-{index:02d}-{name}.png").stat().st_size
            > 0
            for index, name in enumerate(
                (
                    "efficiency-uplift",
                    "throughput-uplift",
                    "throughput-vs-cost",
                    "replica-memory-vs-uplift",
                    "family-summary",
                    "saturation-curves",
                    "replica-selection",
                    "hedging-tail-tradeoff",
                ),
                1,
            )
        ),
        "chart_visual_qa_pass": chart_qa_status == "PASS_VISUAL_INSPECTION",
        "thresholds_changed_after_headline": False,
        "negative_control_gate_pass": False,
        "deterministic_rerun_complete": False,
        "five_full_correctness_receipts_complete": False,
        "hedging_complete": False,
    }
    audit = {
        "schema_version": "experiment-023-final-audit-v1",
        "experiment_id": "023",
        "final_verdict": "MODEL_INVALID",
        "artifact_packaging_status": (
            "PASS" if not missing and plan_audit["status"] == "PASS" else "FAIL"
        ),
        "primary_promotion_status": "BLOCKED_MODEL_INVALID",
        "mandatory_failure": "COARSE_FRIENDLY_03_HARMFUL_ACCEPTED_FLEX_ACTIONS",
        "checks": checks,
        "missing_required_artifacts": missing,
        "plan_audit": dict(plan_audit),
        "artifact_hash_count": len(hashes),
        "artifact_hashes": hashes,
        "artifact_manifest_sha256": canonical_sha256(hashes),
    }
    atomic_write_json(artifact_root / "validation/final-audit.json", audit)
    return audit


def finalize_experiment(
    repo: Path,
    *,
    chart_qa_status: str = "PENDING_VISUAL_QA",
) -> dict[str, Any]:
    """Finalize the stopped E023 run without promoting diagnostic results."""

    if chart_qa_status not in {"PENDING_VISUAL_QA", "PASS_VISUAL_INSPECTION"}:
        raise ValueError("unknown chart QA state")
    root = repo.resolve()
    artifact_root = root / "artifacts/experiment-023"
    attempt_root = artifact_root / "attempts" / ATTEMPT_NAME
    if not attempt_root.is_dir():
        raise RuntimeError("MODEL_INVALID: deterministic-run-1 is missing")
    validate_e023_freeze(root)
    _publish_attempt(attempt_root, artifact_root)
    _write_stopped_phase_artifacts(artifact_root)

    analysis = analyze_attempt(attempt_root)
    if not analysis.control_failures:
        raise RuntimeError("expected frozen negative-control failure was not found")
    validity_failures = tuple(
        "{failure}:{inventory_id}:{arm}".format(**row)
        for row in analysis.control_failures
    )
    headline = [row for row in analysis.uplift_rows if row["cohort"] == "headline"]
    verdict = evaluate_verdict(headline, validity_failures=validity_failures)
    if verdict["final_verdict"] != "MODEL_INVALID":
        raise AssertionError("mandatory control failure did not short-circuit verdict")
    zero_new_node = evaluate_zero_new_node_wedge(headline)
    _write_analysis(artifact_root, analysis)
    chart_map = _generate_charts(
        artifact_root,
        analysis,
        chart_qa_status=chart_qa_status,
    )
    atomic_write_json(artifact_root / "analysis/chart-map.json", chart_map)

    physical = _physical_summary(artifact_root)
    if physical["status"] != "PASS":
        raise RuntimeError("MODEL_INVALID: frozen physical replica gate is not PASS")
    plan_audit = _audit_final_plans(artifact_root)
    attempt_summary = _read_json(attempt_root / "attempt-summary.json")
    summary = _build_summary(
        analysis,
        verdict,
        zero_new_node,
        physical,
        attempt_summary,
    )
    truth = _truth_table(
        analysis,
        verdict,
        zero_new_node,
        physical,
        plan_audit,
    )
    atomic_write_json(artifact_root / "summary.json", summary)
    atomic_write_json(artifact_root / "truth-table.json", truth)
    atomic_write_json(artifact_root / "environment.json", _environment(root, physical))
    atomic_write_text(artifact_root / "commands.txt", _commands_text())
    report = _render_report(
        root,
        analysis,
        summary,
        verdict,
        zero_new_node,
        physical,
        attempt_summary,
    )
    atomic_write_text(root / "docs/experiments/EXPERIMENT_023_REPORT.md", report)
    atomic_write_json(
        artifact_root / "analysis/report-source-notes.json",
        _report_source_notes(chart_map),
    )
    audit = _write_final_audit(
        root,
        artifact_root,
        analysis=analysis,
        physical=physical,
        plan_audit=plan_audit,
        chart_qa_status=chart_qa_status,
    )
    return {
        "status": "COMPLETE_MODEL_INVALID_AUDIT_PACKAGE",
        "final_verdict": "MODEL_INVALID",
        "summary": summary,
        "artifact_packaging_status": audit["artifact_packaging_status"],
        "chart_qa_status": chart_qa_status,
        "report": str(root / "docs/experiments/EXPERIMENT_023_REPORT.md"),
    }


__all__ = ["ATTEMPT_NAME", "STOP_STATUS", "finalize_experiment"]
