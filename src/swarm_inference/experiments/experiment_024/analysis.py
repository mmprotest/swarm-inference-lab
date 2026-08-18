"""Experiment 024 chart construction and visual-QA metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHART_FILENAMES = (
    "chart-01-communication-bytes.png",
    "chart-02-stage-a-gap-closure.png",
    "chart-03-performance-cost-frontier.png",
    "chart-04-cost-per-million.png",
    "chart-05-output-throughput.png",
    "chart-06-contributor-payout-frontier.png",
    "chart-07-whole-layer-incapable-compute.png",
    "chart-08-current-vs-d.png",
    "chart-09-saturation-curves.png",
    "chart-10-active-nodes-vs-cost.png",
)


def _invalid_panel(path: Path, title: str, x_label: str, y_label: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 6), dpi=160)
    fig.patch.set_facecolor("#f7f8fa")
    axis.set_facecolor("#ffffff")
    axis.set_title(title, fontsize=15, weight="bold", pad=16)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#d9dde5", linewidth=0.8, alpha=0.8)
    axis.text(
        0.5,
        0.58,
        "MODEL_INVALID",
        ha="center",
        va="center",
        fontsize=24,
        weight="bold",
        color="#a61b1b",
    )
    axis.text(
        0.5,
        0.43,
        "No production-native P8 candidate exists for transformer layer 0.\n"
        "The fail-closed Phase 0 gate prevents performance inference.",
        ha="center",
        va="center",
        fontsize=11,
        color="#30343b",
    )
    axis.text(
        0.5,
        0.10,
        "No measured or modeled Stage A/Stage B point is plotted.",
        ha="center",
        va="center",
        fontsize=9,
        color="#5e6570",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_invalid_charts(charts_dir: Path) -> dict[str, object]:
    """Render explicitly non-quantitative charts for a Phase 0 invalid run."""

    charts_dir.mkdir(parents=True, exist_ok=True)
    definitions = (
        ("Communication bytes", "Stage A arm", "Bytes per row"),
        ("Stage A gap closure", "Stressed cell", "Gap closure (%)"),
        (
            "Performance-cost frontier",
            "Cost per M / $15 Kimi benchmark",
            "Performance retention",
        ),
        ("Cost per million output tokens", "Scenario", "USD per M output tokens"),
        ("Output throughput", "Scenario", "Aggregate output tokens/s"),
        (
            "Contributor payout frontier",
            "Target serving cost (USD/M)",
            "Maximum payout per active node-hour",
        ),
        (
            "Whole-layer-incapable compute",
            "Scenario",
            "Transformer compute share",
        ),
        ("CURRENT versus D", "Scenario", "Relative effect"),
        ("Saturation curves", "Closed-loop concurrency", "Output tokens/s"),
        ("Active nodes versus cost", "Active nodes", "USD per M output tokens"),
    )
    entries: list[dict[str, str]] = []
    for filename, (title, x_label, y_label) in zip(
        CHART_FILENAMES, definitions, strict=True
    ):
        path = charts_dir / filename
        _invalid_panel(path, title, x_label, y_label)
        entries.append(
            {
                "file": f"charts/{filename}",
                "status": "MODEL_INVALID_NO_PERFORMANCE_DATA",
                "visual_qa": "PENDING",
            }
        )
    chart_map = {
        "schema_version": "experiment-024-chart-map-v1",
        "status": "MODEL_INVALID",
        "charts": entries,
    }
    return chart_map


def update_visual_qa(chart_map_path: Path, *, status: str) -> None:
    value = json.loads(chart_map_path.read_text(encoding="utf-8"))
    for chart in value["charts"]:
        chart["visual_qa"] = status
    chart_map_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_authoritative_charts(
    charts_dir: Path,
    *,
    stage_a_rows: tuple[dict[str, Any], ...],
    gap_rows: tuple[dict[str, Any], ...],
    decode_rows: tuple[dict[str, Any], ...],
    frontier_rows: tuple[dict[str, Any], ...],
    canonical_rows: tuple[dict[str, Any], ...],
    causal_rows: tuple[dict[str, Any], ...],
    payout_rows: tuple[dict[str, Any], ...],
) -> dict[str, object]:
    """Render the ten frozen E024 decision charts from authoritative rows."""

    charts_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "COMMODITY_GOOD": "#167d4a",
        "COMMODITY_REGIONAL": "#2864b7",
        "COMMODITY_WAN": "#b3541e",
    }

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    arms = ["A_CURRENT", "B_RETAIN_HIDDEN", "C_SLICE_LATENT", "D_FUSE_OUTPUT"]
    values = [1_004_416, 803_712, 715_904, 515_200]
    axis.bar(arms, values, color=["#6d7785", "#4f83a5", "#347f86", "#167d4a"])
    axis.axhline(502_712, color="#a61b1b", linestyle="--", label="lower bound")
    axis.set_ylabel("Bytes per row")
    axis.set_title("Frozen MoE communication payload")
    axis.tick_params(axis="x", rotation=18)
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[0])

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    for scenario, color in colors.items():
        values = [
            float(row["latency_gap_closure_percent"])
            for row in gap_rows
            if row["scenario"] == scenario
        ]
        axis.plot(range(len(values)), values, ".", color=color, label=scenario)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Frozen Stage A cell")
    axis.set_ylabel("A→D latency improvement (%)")
    axis.set_title("Stage A gap closure across layers, rows, and concurrency")
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[1])

    fig, axis = plt.subplots(figsize=(9, 6), dpi=160)
    markers = {"SWARM_CURRENT_OPT": "o", "SWARM_D_OPT": "^"}
    for row in frontier_rows:
        axis.scatter(
            float(row["api_cost_ratio_at_0_15"]),
            float(row["performance_retention"]),
            color=colors[str(row["scenario"])],
            marker=markers[str(row["architecture"])],
            alpha=0.8,
            s=45,
        )
    axis.axvline(1.0, color="#a61b1b", linestyle="--", label="Kimi cost parity")
    axis.set_xlabel("Cost per M / $15 Kimi benchmark")
    axis.set_ylabel("Performance retention")
    axis.set_title("E024 performance-cost frontier: 1 whole + 92 P8")
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[2])

    scenarios = [row["scenario"].removeprefix("COMMODITY_") for row in canonical_rows]
    fig, axis = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    axis.bar(
        scenarios,
        [float(row["cost_per_M_at_0_15"]) for row in canonical_rows],
        color=[colors[str(row["scenario"])] for row in canonical_rows],
    )
    axis.axhline(15.0, color="#a61b1b", linestyle="--", label="Kimi $15/M")
    axis.set_ylabel("USD per million output tokens")
    axis.set_title("Canonical serving cost at $0.15/active-node-hour")
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[3])

    fig, axis = plt.subplots(figsize=(8.5, 5.5), dpi=160)
    axis.bar(
        scenarios,
        [float(row["aggregate_output_tokens_per_second"]) for row in canonical_rows],
        color=[colors[str(row["scenario"])] for row in canonical_rows],
    )
    axis.set_ylabel("Aggregate output tokens/s")
    axis.set_title("Canonical closed-loop autoregressive throughput")
    _save(fig, charts_dir / CHART_FILENAMES[4])

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    for scenario, color in colors.items():
        rows = [row for row in payout_rows if row["scenario"] == scenario]
        axis.plot(
            [float(row["target_cost_per_M"]) for row in rows],
            [float(row["max_uniform_payout_per_active_node_hour"]) for row in rows],
            "o-",
            color=color,
            label=scenario,
        )
    axis.set_xlabel("Target output cost (USD/M)")
    axis.set_ylabel("Maximum payout / active node-hour")
    axis.set_title("Contributor payout frontier")
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[5])

    fig, axis = plt.subplots(figsize=(9, 5.8), dpi=160)
    x = list(range(len(canonical_rows)))
    width = 0.34
    axis.bar(
        [value - width / 2 for value in x],
        [
            float(row["overall_whole_layer_incapable_compute_share"])
            for row in canonical_rows
        ],
        width,
        label="overall transformer compute",
        color="#6d7785",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [
            float(row["p8_required_whole_layer_incapable_compute_share"])
            for row in canonical_rows
        ],
        width,
        label="P8-required layers 1-92",
        color="#167d4a",
    )
    axis.set_xticks(x, scenarios)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Whole-layer-incapable compute share")
    axis.set_title("Fine-grained work runs on workers that cannot host its whole layer")
    axis.text(
        0.5,
        -0.17,
        "Layer 0 is intentionally whole because it fits the commodity class.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[6])

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    if causal_rows:
        causal_scenarios = [
            row["scenario"].removeprefix("COMMODITY_") for row in causal_rows
        ]
        x = list(range(len(causal_rows)))
        axis.bar(
            [value - 0.18 for value in x],
            [
                float(row["execution_only_throughput_improvement_percent"])
                for row in causal_rows
            ],
            0.36,
            label="throughput improvement",
        )
        axis.bar(
            [value + 0.18 for value in x],
            [
                float(row["execution_only_cost_reduction_percent"])
                for row in causal_rows
            ],
            0.36,
            label="cost reduction",
        )
        axis.set_xticks(x, causal_scenarios)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_ylabel("Percent")
    axis.set_title("CURRENT versus D on identical D placement")
    axis.legend()
    _save(fig, charts_dir / CHART_FILENAMES[7])

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    canonical_keys = {
        (
            row["scenario"],
            row["architecture"],
            int(row["available_node_budget"]),
        )
        for row in canonical_rows
    }
    for key in sorted(canonical_keys):
        rows = sorted(
            (
                row
                for row in decode_rows
                if (
                    row["scenario"],
                    row["architecture"],
                    int(row["available_node_budget"]),
                )
                == key
            ),
            key=lambda row: int(row["concurrency"]),
        )
        axis.plot(
            [int(row["concurrency"]) for row in rows],
            [float(row["aggregate_output_tokens_per_second"]) for row in rows],
            "o-",
            label=f"{key[0].removeprefix('COMMODITY_')} {key[1]}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Closed-loop decode concurrency")
    axis.set_ylabel("Aggregate output tokens/s")
    axis.set_title("Canonical saturation curves")
    axis.legend(fontsize=7)
    _save(fig, charts_dir / CHART_FILENAMES[8])

    fig, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    for row in frontier_rows:
        axis.scatter(
            int(row["active_node_count"]),
            float(row["cost_per_M_at_0_15"]),
            color=colors[str(row["scenario"])],
            marker=markers[str(row["architecture"])],
        )
    axis.axhline(15.0, color="#a61b1b", linestyle="--")
    axis.set_xlabel("Active commodity nodes")
    axis.set_ylabel("USD per million output tokens")
    axis.set_title("Active nodes versus serving cost")
    _save(fig, charts_dir / CHART_FILENAMES[9])

    return {
        "schema_version": "experiment-024-chart-map-v2",
        "status": "PASS",
        "charts": [
            {
                "file": f"charts/{filename}",
                "status": "PASS",
                "visual_qa": "PENDING",
            }
            for filename in CHART_FILENAMES
        ],
        "source_row_counts": {
            "stage_a": len(stage_a_rows),
            "gap": len(gap_rows),
            "decode": len(decode_rows),
            "frontier": len(frontier_rows),
            "canonical": len(canonical_rows),
        },
    }


__all__ = [
    "CHART_FILENAMES",
    "render_authoritative_charts",
    "render_invalid_charts",
    "update_visual_qa",
]
