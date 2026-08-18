"""Experiment 024 chart construction and visual-QA metadata."""

from __future__ import annotations

import json
from pathlib import Path

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


__all__ = ["CHART_FILENAMES", "render_invalid_charts", "update_visual_qa"]
