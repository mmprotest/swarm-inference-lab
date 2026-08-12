"""Generate the seven publication charts required by Experiment 016."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "navy": "#17324D",
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C94C4C",
    "gray": "#8C98A4",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _finish(
    figure: plt.Figure,
    path: Path,
    *,
    source: str,
    legend: bool = False,
) -> None:
    figure.text(0.01, 0.012, f"Source: {source}", fontsize=7.5, color="#53606B")
    if legend:
        for axis in figure.axes:
            handles, _labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(loc="best")
                break
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def generate(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    artifact_root = root / "artifacts" / "experiment-016"
    chart_root = artifact_root / "charts"
    chart_root.mkdir(parents=True, exist_ok=True)
    summary = _load(artifact_root / "summary.json")
    _style()
    index: list[dict[str, str]] = []

    arms = summary["arms"]
    labels = [
        "015 baseline",
        "Device\nresident",
        "Verification\nmajor",
        "+ exact\nDCP8",
        "+ overlap\n(retained)",
    ]
    throughput = [float(row["verifier_tokens_per_second"]) for row in arms]
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    bars = axis.bar(
        labels,
        throughput,
        color=[COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["teal"], COLORS["navy"]],
        width=0.67,
    )
    axis.set_ylabel("Accepted target tokens / second")
    axis.set_title("Verifier throughput by arm", loc="left", fontweight="bold")
    axis.set_ylim(0, max(throughput) * 1.22)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    path = chart_root / "chart-01-verifier-throughput-by-arm.png"
    _finish(
        figure,
        path,
        source="Measured Kimi stages bridged through the fixed Experiment 015 8-layer topology.",
    )
    index.append({"chart": "Verifier throughput by arm", "path": str(path.relative_to(root))})

    decomposition = summary["latency_decomposition"]
    components = [
        "KDA layers",
        "MLA layers",
        "endpoint",
        "topology communication",
        "DCP communication",
    ]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["gray"], COLORS["orange"]]
    figure, axis = plt.subplots(figsize=(8.8, 5.6))
    bottoms = np.zeros(2)
    for component, color in zip(components, colors, strict=True):
        values = [
            float(decomposition["baseline"][component]),
            float(decomposition["final"][component]),
        ]
        axis.bar(
            ["Experiment 015", "Experiment 016 final"],
            values,
            bottom=bottoms,
            label=component,
            color=color,
            width=0.58,
        )
        bottoms += np.asarray(values)
    axis.set_ylabel("Target pass latency (ms; 8 accepted tokens)")
    axis.set_title("Verifier latency decomposition", loc="left", fontweight="bold")
    axis.set_ylim(0, float(max(bottoms)) * 1.10)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    axis.text(0, bottoms[0] + 35, f"{bottoms[0]:.0f} ms", ha="center", fontweight="bold")
    axis.text(1, bottoms[1] + 35, f"{bottoms[1]:.0f} ms", ha="center", fontweight="bold")
    path = chart_root / "chart-02-verifier-latency-decomposition.png"
    _finish(
        figure,
        path,
        source="Experiment 015 decomposition; Experiment 016 measured stage substitutions and shaped communication.",
    )
    index.append({"chart": "Verifier latency decomposition", "path": str(path.relative_to(root))})

    blocks = summary["block_scaling"]
    x = [int(row["candidate_block_size"]) for row in blocks]
    total = [float(row["total_wall_p50_ms"]) for row in blocks]
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    axis.plot(
        x,
        total,
        marker="o",
        linewidth=2.4,
        color=COLORS["blue"],
        label="Measured final grouped MLA layer",
    )
    linear = [total[0] * value for value in x]
    axis.plot(
        x, linear, linestyle="--", color=COLORS["gray"], linewidth=1.5, label="Linear from block 1"
    )
    axis.set_xlabel("Speculative candidate block size")
    axis.set_ylabel("Total verification wall p50 (ms)")
    axis.set_xticks(x)
    axis.set_title("Verification cost grows sublinearly", loc="left", fontweight="bold")
    path = chart_root / "chart-03-block-size-vs-total-latency.png"
    _finish(
        figure,
        path,
        source="Real Kimi layer 91 at 8K context; 20 retained runs per point.",
        legend=True,
    )
    index.append({"chart": "Block size vs total latency", "path": str(path.relative_to(root))})

    per_candidate = [float(row["latency_per_candidate_ms"]) for row in blocks]
    per_accepted = [float(row["latency_per_accepted_token_ms"]) for row in blocks]
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    axis.plot(
        x, per_candidate, marker="o", linewidth=2.4, color=COLORS["blue"], label="Per candidate"
    )
    axis.plot(
        x,
        per_accepted,
        marker="s",
        linewidth=2.2,
        color=COLORS["teal"],
        label="Per accepted token (incl. bonus)",
    )
    axis.set_xlabel("Speculative candidate block size")
    axis.set_ylabel("Verification wall p50 (ms)")
    axis.set_xticks(x)
    axis.set_title("Amortization lowers per-token verification cost", loc="left", fontweight="bold")
    path = chart_root / "chart-04-block-size-vs-latency-per-candidate.png"
    _finish(figure, path, source="Real Kimi layer 91 at 8K context.", legend=True)
    index.append(
        {"chart": "Block size vs latency per candidate", "path": str(path.relative_to(root))}
    )

    assignments = [int(row["total_assignments"]) for row in blocks]
    unique = [float(row["mean_unique_experts"]) for row in blocks]
    reuse = [float(row["mean_assignments_per_touched_expert"]) for row in blocks]
    figure, axis = plt.subplots(figsize=(9.1, 5.5))
    bars = axis.bar(x, assignments, color=COLORS["blue"], alpha=0.82, label="Total assignments")
    axis.set_xlabel("Speculative candidate block size")
    axis.set_ylabel("Token-expert assignments")
    axis.set_xticks(x)
    twin = axis.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(x, unique, marker="o", color=COLORS["orange"], linewidth=2.3, label="Unique experts")
    twin.plot(
        x,
        reuse,
        marker="s",
        color=COLORS["teal"],
        linewidth=2.3,
        label="Assignments / touched expert",
    )
    twin.set_ylabel("Experts or mean reuse")
    axis.set_title(
        "Expert union saturates while reuse keeps growing", loc="left", fontweight="bold"
    )
    handles = [bars, *twin.get_lines()]
    axis.legend(handles, [item.get_label() for item in handles], loc="upper left", fontsize=8)
    path = chart_root / "chart-05-expert-reuse.png"
    _finish(figure, path, source="Real Kimi layer-91 routes; cyclic three-boundary fixture.")
    index.append({"chart": "Expert reuse", "path": str(path.relative_to(root))})

    oracle = [float(row["whole_system_oracle_tok_s_per_user"]) for row in arms]
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    bars = axis.bar(
        labels,
        oracle,
        color=[COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["teal"], COLORS["navy"]],
        width=0.67,
    )
    axis.axhline(5.0, color=COLORS["red"], linestyle="--", linewidth=2.0, label="5 tok/s target")
    axis.set_ylabel("Oracle tok/s/user (zero draft cost, 100% acceptance)")
    axis.set_title("The optimized oracle remains below 5 tok/s/user", loc="left", fontweight="bold")
    axis.set_ylim(0, 5.8)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    path = chart_root / "chart-06-oracle-throughput.png"
    _finish(
        figure,
        path,
        source="Fixed 8-layer topology projection from measured stage service.",
        legend=True,
    )
    index.append({"chart": "Oracle tok/s/user", "path": str(path.relative_to(root))})

    dcp = summary["dcp"]["rows"]
    contexts = sorted({int(row["context_tokens"]) for row in dcp})
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    degree_colors = {1: COLORS["gray"], 2: COLORS["orange"], 4: COLORS["blue"], 8: COLORS["teal"]}
    for degree in (1, 2, 4, 8):
        rows = sorted(
            (row for row in dcp if int(row["degree"]) == degree),
            key=lambda row: int(row["context_tokens"]),
        )
        axis.plot(
            [int(row["context_tokens"]) / 1024 for row in rows],
            [float(row["shaped_wall_p50_ms"]) for row in rows],
            marker="o",
            linewidth=2.2,
            color=degree_colors[degree],
            label=f"DCP{degree}",
        )
    axis.set_xlabel("Context length (K tokens)")
    axis.set_ylabel("Whole layer wall p50 + shaped transport (ms)")
    axis.set_xticks(
        [value / 1024 for value in contexts], [f"{value // 1024}K" for value in contexts]
    )
    axis.set_title("DCP crossover is small and context-dependent", loc="left", fontweight="bold")
    path = chart_root / "chart-07-dcp-scaling.png"
    _finish(
        figure,
        path,
        source="Exact local GPU compute; 0.25 ms RTT / 25 Gb/s internal transport shaping.",
        legend=True,
    )
    index.append({"chart": "DCP scaling by context", "path": str(path.relative_to(root))})

    (artifact_root / "chart-index.json").write_text(
        json.dumps({"schema_version": "experiment-016-chart-index-v1", "charts": index}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    charts = generate(arguments.root)
    print(f"[h016-figures] generated={len(charts)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
