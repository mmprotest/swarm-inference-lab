"""Render and validate the seven required Experiment 017 charts."""

# Chart copy deliberately uses typographic mathematical punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INK = "#17202A"
MUTED = "#667085"
GRID = "#D0D5DD"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D1495B"
PURPLE = "#7353BA"
GRAY = "#98A2B3"
GREEN = "#3A8D5D"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], field: str) -> float:
    value = row[field]
    return float(value) if value else float("nan")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _oracle_progress(summary: dict[str, Any], path: Path) -> None:
    values = [2.1838203221, 2.6691, summary["primary_result"]["oracle_tok_s_per_user"], 0]
    labels = [
        "Experiment 015\noracle",
        "Experiment 016\nretained",
        "E017 best exact*\nblock 16",
        "Qualified\napproximate",
    ]
    colors = [GRAY, BLUE, TEAL, "white"]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.bar(labels, values, color=colors, edgecolor=[GRAY, BLUE, TEAL, GRAY], linewidth=1.5)
    bars[-1].set_hatch("///")
    for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
        if index == 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.16,
                "N/A\nnone qualified",
                ha="center",
                va="bottom",
                color=MUTED,
                fontweight="bold",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.1,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                color=INK,
                fontweight="bold",
            )
    ax.axhline(5.0, color=RED, linewidth=2.0, label="Target: 5.0000")
    ax.axhline(5.3382, color=PURPLE, linewidth=1.7, linestyle="--", label="PASS_STRONG: 5.3382")
    ax.set_ylim(0, 5.85)
    ax.set_ylabel("Zero-draft oracle (output tok/s/user)")
    fig.suptitle(
        "Experiment 017 did not cross the 5 tok/s/user target",
        x=0.075,
        y=0.985,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0,
        1.015,
        "No E017 latency arm survived its whole-layer gate; *block 16 was already present in E016.",
        transform=ax.transAxes,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", ncol=2)
    fig.subplots_adjust(top=0.86)
    _save(fig, path)


def _kda_budget(summary: dict[str, Any], path: Path) -> None:
    tracker = summary["target_tracker"]
    baseline = float(tracker["baseline_exp016_kda_ms"])
    budget = float(tracker["initial_required_kda_ms_if_non_kda_frozen"])
    required = float(tracker["initial_required_additional_kda_speedup"])
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.2, 5.9), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    bars = left.barh(
        ["Allowed KDA at 5 tok/s", "E016 KDA allocation"],
        [budget, baseline],
        color=[TEAL, RED],
        height=0.56,
    )
    for bar, value in zip(bars, [budget, baseline], strict=True):
        left.text(
            value + 25,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.2f} ms",
            va="center",
            fontweight="bold",
        )
    left.set_xlim(0, 1770)
    left.set_xlabel("Block-7 bridge contribution (ms)")
    left.set_title("Frozen non-KDA budget", loc="left")
    left.grid(axis="x")
    left.spines[["top", "right", "left"]].set_visible(False)
    right.axis("off")
    right.text(
        0.5,
        0.67,
        f"{required:.2f}×",
        ha="center",
        va="center",
        fontsize=38,
        color=RED,
        fontweight="bold",
    )
    right.text(
        0.5, 0.48, "required KDA reduction", ha="center", color=INK, fontsize=13, fontweight="bold"
    )
    right.text(
        0.5,
        0.30,
        "1600.00 ms total budget\n− 1437.84 ms non-KDA\n= 162.16 ms KDA",
        ha="center",
        va="center",
        color=MUTED,
        linespacing=1.45,
    )
    fig.suptitle(
        "The original block-7 KDA budget was an order-of-magnitude gap",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    _save(fig, path)


def _decomposition(summary: dict[str, Any], path: Path) -> None:
    system = summary["new_bottleneck"]["block7_components_ms"]
    phases = summary["new_bottleneck"]["phase_components_ms"]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.5), sharex=True)
    palettes = [BLUE, TEAL, ORANGE, PURPLE, GRAY, GREEN, RED, "#8D6E63"]
    for ax, data, title in (
        (axes[0], system, "Experiment 016 validated-bridge allocation"),
        (axes[1], phases, "Measured/reconciled execution phases"),
    ):
        left = 0.0
        total = sum(float(value) for value in data.values())
        for index, (label, value_raw) in enumerate(data.items()):
            value = float(value_raw)
            ax.barh(
                [0],
                [value],
                left=[left],
                color=palettes[index % len(palettes)],
                height=0.48,
                label=label,
            )
            if value / total >= 0.055:
                ax.text(
                    left + value / 2,
                    0,
                    f"{label}\n{value:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8.5,
                    fontweight="bold",
                )
            left += value
        ax.set_yticks([])
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=8)
    axes[1].set_xlabel("Block-7 target pass (ms)")
    axes[1].set_xlim(0, 3150)
    fig.suptitle(
        "“KDA” was a layer-family bucket—not recurrent-state time",
        x=0.06,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.subplots_adjust(hspace=0.82)
    _save(fig, path)


def _state_traffic(rows: list[dict[str, str]], path: Path) -> None:
    blocks = np.array([int(row["block_size"]) for row in rows])
    read = np.array([_float(row, "logical_state_read_bytes") for row in rows]) / (1024**2)
    write = np.array([_float(row, "logical_state_write_bytes") for row in rows]) / (1024**2)
    factors = np.array([_float(row, "compact_factor_bytes") for row in rows]) / (1024**2)
    snapshots = np.array([_float(row, "hypothetical_full_snapshot_bytes") for row in rows]) / (
        1024**2
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.plot(blocks, read, marker="o", linewidth=2.2, color=BLUE, label="Serial logical reads")
    ax.plot(blocks, write, marker="o", linewidth=2.2, color=RED, label="Serial logical writes")
    ax.plot(
        blocks,
        snapshots,
        marker="s",
        linewidth=2.0,
        color=PURPLE,
        label="Hypothetical full snapshots",
    )
    ax.plot(blocks, factors, marker="D", linewidth=2.0, color=TEAL, label="Compact token factors")
    ax.set_xticks(blocks)
    ax.set_xlabel("Candidate block size")
    ax.set_ylabel("Logical/modelled traffic or capacity (MiB)")
    ax.set_title(
        "Compact factors save snapshot capacity, not retained-path traffic", loc="left", fontsize=16
    )
    ax.text(
        0.02,
        0.93,
        "Accepted-state copy: 0 bytes · 0 launches · 0 ms",
        transform=ax.transAxes,
        color=GREEN,
        fontweight="bold",
    )
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.13))
    _save(fig, path)


def _expert_service(rows: list[dict[str, str]], path: Path) -> None:
    measured = [row for row in rows if row["available"] == "True"]
    labels = ["Fused gate/up\n(canonical E016)", "Unfused\nexact control"]
    expert = [_float(row, "expert_phase_device_ms") for row in measured]
    full = [_float(row, "full_kda_layer_device_ms") for row in measured]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.6, 6.1))
    expert_bars = ax.bar(x - width / 2, expert, width, color=ORANGE, label="Routed expert phase")
    full_bars = ax.bar(x + width / 2, full, width, color=BLUE, label="Full layer 89")
    for bars in (expert_bars, full_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                f"{bar.get_height():.3f}",
                ha="center",
                fontweight="bold",
                fontsize=9,
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(full) * 1.22)
    ax.set_ylabel("CUDA-event p50 (ms)")
    ax.set_title("Native MXFP4 fusion helps—but was already the baseline", loc="left", fontsize=16)
    ax.text(
        0.02,
        0.91,
        "Real block-7 routing: 36 experts · mean M=3.56 · max M=8",
        transform=ax.transAxes,
        color=MUTED,
    )
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    _save(fig, path)


def _speed_quality(summary: dict[str, Any], rows: list[dict[str, str]], path: Path) -> None:
    approximate = next(row for row in rows if row["mode"] == "bf16-state-fp32-update")
    error = _float(approximate, "relative_l2_error")
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    ax.scatter([0], [2.6691], s=145, color=BLUE, label="Exact block-7 control", zorder=3)
    ax.scatter(
        [0],
        [summary["primary_result"]["oracle_tok_s_per_user"]],
        s=145,
        color=TEAL,
        marker="D",
        label="Exact best block (existing E016 curve)",
        zorder=3,
    )
    ax.scatter(
        [error],
        [0.22],
        s=165,
        facecolors="none",
        edgecolors=RED,
        marker="X",
        linewidths=2.2,
        label="BF16-state reference (no GPU throughput)",
        zorder=3,
    )
    ax.axvline(0.003, color=PURPLE, linestyle="--", linewidth=1.8, label="Full-graph rel-L2 gate")
    ax.axhline(5.0, color=RED, linewidth=1.8, label="5 tok/s target")
    ax.annotate(
        "Failed layer screen\nrel L2 = 0.003523\nthroughput not measured",
        xy=(error, 0.22),
        xytext=(0.00385, 1.05),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        color=INK,
    )
    ax.set_xlim(-0.00012, 0.0056)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel("Relative L2 error (lower is better)")
    ax.set_ylabel("Qualified zero-draft oracle (tok/s/user)")
    ax.set_title("No speed–quality point reached the target", loc="left", fontsize=16)
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.7)
    _save(fig, path)


def _block_oracle(rows: list[dict[str, str]], path: Path) -> None:
    controls = sorted(
        (row for row in rows if row["configuration"] == "E016 exact control"),
        key=lambda row: int(row["block_size"]),
    )
    fused = sorted(
        (row for row in rows if row["configuration"] == "Arm A fused KDA window"),
        key=lambda row: int(row["block_size"]),
    )
    blocks = [int(row["block_size"]) for row in controls]
    exact_y = [_float(row, "oracle_tok_s_per_user") for row in controls]
    fused_y = [_float(row, "oracle_tok_s_per_user") for row in fused]
    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    ax.plot(blocks, exact_y, marker="o", linewidth=2.4, color=BLUE, label="E016 exact control")
    ax.plot(
        blocks,
        fused_y,
        marker="s",
        linewidth=2.0,
        color=ORANGE,
        linestyle="--",
        label="Arm A fused window (rejected)",
    )
    ax.axhline(5.0, color=RED, linewidth=2.0, label="5 tok/s target")
    best_index = int(np.argmax(exact_y))
    ax.annotate(
        f"Best: {exact_y[best_index]:.4f}\n(existing E016 curve)",
        xy=(blocks[best_index], exact_y[best_index]),
        xytext=(11.1, 3.7),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontweight="bold",
    )
    ax.set_xticks(blocks)
    ax.set_ylim(0, 5.55)
    ax.set_xlabel("Candidate block size")
    ax.set_ylabel("Zero-draft oracle (tok/s/user)")
    ax.set_title("Larger blocks amortize work, but still miss 5 tok/s", loc="left", fontsize=16)
    ax.grid()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    _save(fig, path)


def _validate(paths: list[Path]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            values = np.asarray(rgb)
            nonwhite_fraction = float(np.mean(np.any(values < 248, axis=2)))
            width, height = rgb.size
        if width < 1200 or height < 700 or nonwhite_fraction < 0.02:
            raise RuntimeError(
                f"chart validation failed for {path}: {width}x{height}, ink={nonwhite_fraction:.4f}"
            )
        checks.append(
            {
                "file": path.name,
                "width_px": width,
                "height_px": height,
                "nonwhite_fraction": nonwhite_fraction,
                "status": "PASS",
            }
        )
    return {"schema_version": "experiment-017-chart-validation-v1", "charts": checks}


def render(root: Path) -> dict[str, Any]:
    root = root.resolve()
    artifacts = root / "artifacts" / "experiment-017"
    results = artifacts / "results"
    charts = artifacts / "charts"
    summary = _read_json(artifacts / "summary.json")
    _style()
    names = [
        "chart-01-oracle-progress.png",
        "chart-02-kda-budget-to-5.png",
        "chart-03-kda-layer-decomposition.png",
        "chart-04-state-traffic.png",
        "chart-05-expert-service.png",
        "chart-06-speed-quality-pareto.png",
        "chart-07-block-size-oracle.png",
    ]
    paths = [charts / name for name in names]
    _oracle_progress(summary, paths[0])
    _kda_budget(summary, paths[1])
    _decomposition(summary, paths[2])
    _state_traffic(_read_csv(results / "state-traffic.csv"), paths[3])
    _expert_service(_read_csv(results / "expert-backends.csv"), paths[4])
    _speed_quality(summary, _read_csv(results / "precision-matrix.csv"), paths[5])
    _block_oracle(_read_csv(results / "oracle-by-block.csv"), paths[6])
    validation = _validate(paths)
    (charts / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    print(json.dumps(render(arguments.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
