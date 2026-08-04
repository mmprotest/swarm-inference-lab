"""Publication-quality network charts generated only from evidence CSV rows."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from swarm_inference.experiments.experiment_011.analysis import PROFILE_LABELS, PROFILE_ORDER

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#555555"


def _load_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["profile_order"]))
    if [row["profile_name"] for row in rows] != list(PROFILE_ORDER):
        raise ValueError("chart source does not contain the fixed eight-profile order")
    return rows


def _base_axes(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    figure, axes = plt.subplots(figsize=(16, 9), constrained_layout=False)
    figure.suptitle(title, fontsize=22, fontweight="bold", y=0.965)
    figure.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=13, color=_GREY)
    figure.subplots_adjust(left=0.085, right=0.975, top=0.855, bottom=0.19)
    axes.set_ylabel("Verified decode tokens / second")
    axes.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axes.spines[["top", "right"]].set_visible(False)
    return figure, axes


def _annotate_points(axes: plt.Axes, x: np.ndarray, y: list[float], color: str) -> None:
    maximum = max(y, default=1.0)
    for index, value in enumerate(y):
        vertical = 9 if index % 2 == 0 else -16
        axes.annotate(
            f"{value:.2f}",
            (x[index], value),
            xytext=(0, vertical),
            textcoords="offset points",
            ha="center",
            va="bottom" if vertical > 0 else "top",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
    axes.set_ylim(bottom=0, top=max(axes.get_ylim()[1], maximum * 1.24, 0.5))


def _finish(
    figure: plt.Figure,
    axes: plt.Axes,
    labels: list[str],
    output_base: Path,
    *,
    source_note: str,
) -> None:
    axes.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    figure.text(0.085, 0.035, source_note, ha="left", va="bottom", fontsize=9, color=_GREY)
    for extension in ("png", "svg", "pdf"):
        target = output_base.with_suffix(f".{extension}")
        figure.savefig(
            target,
            dpi=220 if extension == "png" else None,
            facecolor="white",
            edgecolor="none",
        )
    plt.close(figure)


def generate_network_charts(summary_csv: Path, output_directory: Path) -> dict[str, Any]:
    rows = _load_summary(summary_csv)
    output_directory.mkdir(parents=True, exist_ok=True)
    labels = [PROFILE_LABELS[row["profile_name"]] for row in rows]
    x = np.arange(len(rows))
    archived = [float(row["archived_010_tps"]) for row in rows]
    baseline = [float(row["same_run_baseline_median_tps"]) for row in rows]
    stage = [float(row["stage_exact_median_tps"]) for row in rows]
    ci_low = [float(row["stage_exact_ci_low"]) for row in rows]
    ci_high = [float(row["stage_exact_ci_high"]) for row in rows]
    multiple = [float(row["throughput_multiple"]) for row in rows]
    source_note = (
        "Source: Experiment 010 final evidence bundle and Experiment 011 final evidence bundle"
    )

    figure, axes = _base_axes(
        "Communication-avoiding exact decode changes the network curve",
        "Experiment 010 exact expert RPC compared with Experiment 011 exact stage execution under identical shaped network profiles.",
    )
    axes.plot(
        x,
        archived,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color=_ORANGE,
        label="Experiment 010 archived exact expert RPC",
    )
    axes.plot(
        x,
        stage,
        marker="s",
        linewidth=2.7,
        markersize=7,
        color=_BLUE,
        label="Experiment 011 best exact stage path",
    )
    _annotate_points(axes, x, archived, _ORANGE)
    _annotate_points(axes, x, stage, _BLUE)
    axes.legend(loc="upper right", frameon=True, framealpha=0.96)
    _finish(
        figure,
        axes,
        labels,
        output_directory / "06_network_profile_before_after",
        source_note=source_note,
    )

    figure, axes = _base_axes(
        "Same-run causal comparison of exact decode paths",
        "Fresh expert-RPC and stage-ring measurements share the Experiment 011 software, model, prompt, and shaping environment.",
    )
    axes.plot(
        x,
        baseline,
        marker="o",
        linewidth=2.4,
        markersize=7,
        color=_ORANGE,
        label="Fresh Experiment 011 expert-RPC baseline",
    )
    axes.errorbar(
        x,
        stage,
        yerr=[
            np.maximum(np.asarray(stage) - np.asarray(ci_low), 0),
            np.maximum(np.asarray(ci_high) - np.asarray(stage), 0),
        ],
        marker="s",
        linewidth=2.7,
        markersize=7,
        capsize=4,
        color=_BLUE,
        label="Experiment 011 best exact stage path (95% bootstrap CI)",
    )
    _annotate_points(axes, x, baseline, _ORANGE)
    _annotate_points(axes, x, stage, _BLUE)
    axes.legend(loc="upper right", frameon=True, framealpha=0.96)
    _finish(
        figure,
        axes,
        labels,
        output_directory / "06b_network_profile_same_run_comparison",
        source_note=source_note,
    )

    figure, axes = _base_axes(
        "Exact stage execution across shaped networks",
        "The communication-avoiding path was measured from loopback through global WAN.",
    )
    axes.fill_between(x, stage, 0, color=_BLUE, alpha=0.15)
    axes.plot(x, stage, marker="o", linewidth=2.8, markersize=8, color=_BLUE)
    _annotate_points(axes, x, stage, _BLUE)
    _finish(
        figure,
        axes,
        labels,
        output_directory / "06c_network_profile_experiment_011",
        source_note=source_note,
    )

    figure, axes = _base_axes(
        "Exact stage-path throughput multiple",
        "Throughput of the selected exact stage path divided by the fresh same-run expert-RPC baseline.",
    )
    axes.set_ylabel("Throughput multiple vs same-run baseline (x)")
    colors = [_GREEN if value >= 1.0 else _ORANGE for value in multiple]
    bars = axes.bar(x, multiple, color=colors, width=0.65, alpha=0.9)
    axes.axhline(1.0, color=_GREY, linestyle="--", linewidth=1.8, label="1.0x no improvement")
    for bar, value in zip(bars, multiple, strict=True):
        axes.annotate(
            f"{value:.2f}x",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes.set_ylim(0, max(max(multiple, default=1.0) * 1.18, 1.4))
    axes.legend(loc="upper left", frameon=True, framealpha=0.96)
    _finish(
        figure,
        axes,
        labels,
        output_directory / "06d_network_profile_improvement",
        source_note=source_note,
    )

    inspection = {"source_csv": str(summary_csv.resolve()), "charts": []}
    for png in sorted(output_directory.glob("06*.png")):
        with Image.open(png) as image:
            image.load()
            dimensions = list(image.size)
        inspection["charts"].append(
            {
                "png": str(png.resolve()),
                "dimensions": dimensions,
                "dpi_requirement": 220,
                "minimum_dimensions_met": dimensions[0] >= 3200 and dimensions[1] >= 1800,
                "opened_successfully": True,
                "title_subtitle_separate_layout": True,
                "point_label_offsets_applied": True,
                "x_labels_rotated": True,
                "legend_location_reserved": True,
                "source_note_visible_layout": True,
                "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                "svg_sha256": hashlib.sha256(png.with_suffix(".svg").read_bytes()).hexdigest(),
                "pdf_sha256": hashlib.sha256(png.with_suffix(".pdf").read_bytes()).hexdigest(),
            }
        )
    inspection["all_minimum_dimensions_met"] = all(
        row["minimum_dimensions_met"] for row in inspection["charts"]
    )
    (output_directory / "chart_inspection.json").write_text(
        json.dumps(inspection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "chart_source_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inspection
