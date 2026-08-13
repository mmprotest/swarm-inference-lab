"""Render the required Experiment 018 scientific figures."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#172033"
BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#15803d"
AMBER = "#d97706"
RED = "#dc2626"
GRAY = "#94a3b8"
LIGHT = "#e2e8f0"
CLAIM = "Independent-resource event model • physical K3 service • shaped network"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _figure(title: str, subtitle: str = CLAIM) -> tuple[Any, Any]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": LIGHT,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.suptitle(title, x=0.07, y=0.965, ha="left", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.07, 0.925, subtitle, ha="left", fontsize=9, color="#475569")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.14)
    return fig, ax


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _chart_01(root: Path, summary: Mapping[str, Any]) -> None:
    baseline = _read_csv(root / "baseline/oracle-curve.csv")
    sweep = _read_csv(root / "wavefront/sweep.csv")
    blocks = [4, 7, 12, 16]
    serial = {
        int(row["verification_block"]): float(row["oracle_tok_s_per_user"])
        for row in baseline
    }
    wave = {
        block: max(
            float(row["oracle_tok_s_per_user"])
            for row in sweep
            if row["result_class"] == "C_wavefront_attnres_cache"
            and int(row["block_candidates"]) == block
        )
        for block in blocks
    }
    fig, ax = _figure("Chart 01 — Exact oracle progress")
    x = np.arange(len(blocks))
    width = 0.36
    ax.bar(x - width / 2, [serial[b] for b in blocks], width, color=GRAY, label="Serial E016/E017")
    ax.bar(x + width / 2, [wave[b] for b in blocks], width, color=BLUE, label="Wavefront + cache")
    ax.axhline(5.0, color=RED, linewidth=2, linestyle="--", label="5 tok/s/user target")
    ax.set_xticks(x, [f"block {block}" for block in blocks])
    ax.set_ylabel("Exact output tokens / second / user")
    ax.set_ylim(0, max(6.2, max(wave.values()) * 1.15))
    ax.legend(frameon=False, ncols=3, loc="upper left")
    for index, value in enumerate(wave.values()):
        ax.text(index + width / 2, value + 0.08, f"{value:.2f}", ha="center", color=INK)
    _save(fig, root / "charts/chart-01-oracle-progress.png")


def _chart_02(root: Path, summary: Mapping[str, Any]) -> None:
    best = summary["best_exact"]
    path = root / "wavefront/event-traces" / (
        f"block-{int(best['block_candidates']):02d}-chunk-{int(best['chunk_size']):02d}-cache.json"
    )
    trace = json.loads(path.read_text(encoding="utf-8"))
    records = [
        row
        for row in trace["event_run"]["records"]
        if row["kind"] == "compute" and "cell" in row["metadata"]
    ]
    fig, ax = _figure(
        "Chart 02 — Winning token x depth wavefront",
        f"Block {best['block_candidates']} • chunk {best['chunk_size']} • colored by chunk • {CLAIM}",
    )
    maximum_chunk = max(int(row["metadata"]["chunk"]) for row in records)
    colors = plt.get_cmap("turbo", maximum_chunk + 1)
    for row in records:
        cell = int(row["metadata"]["cell"])
        chunk = int(row["metadata"]["chunk"])
        ax.barh(
            cell,
            float(row["duration_ms"]),
            left=float(row["start_ms"]),
            height=0.66,
            color=colors(chunk),
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_yticks(range(12), [f"cell {value}" for value in range(12)])
    ax.invert_yaxis()
    ax.set_xlabel("Modeled critical time (ms)")
    ax.set_ylabel("Fixed 8-layer microcell")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    scalar = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0, vmax=max(1, maximum_chunk)), cmap=colors
    )
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label("Chunk ID")
    _save(fig, root / "charts/chart-02-wavefront-gantt.png")


def _chart_03(root: Path, summary: Mapping[str, Any]) -> None:
    data = json.loads((root / "wavefront/critical-path.json").read_text(encoding="utf-8"))
    rows = data["cumulative_by_cell"]
    x = [int(row["microcell_id"]) for row in rows]
    fig, ax = _figure("Chart 03 — Serial cumulative latency vs wavefront critical path")
    ax.plot(x, [row["serial_cumulative_historical_target_ms"] for row in rows], marker="o", color=GRAY, linewidth=2.5, label="Immutable serial cumulative target")
    ax.plot(x, [row["wavefront_latest_compute_finish_ms"] for row in rows], marker="o", color=BLUE, linewidth=2.5, label="Wavefront latest finish")
    ax.set_xlabel("Microcell depth index")
    ax.set_ylabel("Cumulative / latest finish (ms)")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    _save(fig, root / "charts/chart-03-critical-path.png")


def _chart_04(root: Path, summary: Mapping[str, Any]) -> None:
    best = summary["best_exact"]
    rows = [
        row
        for row in _read_csv(root / "physical/microcell-service.csv")
        if int(row["chunk_rows"]) == int(best["chunk_size"])
    ]
    cells = [int(row["microcell_id"]) for row in rows]
    service = [float(row["service_p50_ms"]) for row in rows]
    utilization = {
        int(key): float(value)
        for key, value in best["steady_stage_utilization"].items()
    }
    fig, ax = _figure("Chart 04 — Per-microcell service and utilization")
    colors = [RED if cell == int(best["slowest_stage"]) else BLUE for cell in cells]
    ax.bar(cells, service, color=colors, alpha=0.9, label="Service p50")
    ax.set_xlabel("Microcell")
    ax.set_ylabel("Measured/anchored service p50 (ms)")
    ax.set_xticks(cells)
    right = ax.twinx()
    right.plot(cells, [100 * utilization[cell] for cell in cells], color=AMBER, marker="o", linewidth=2, label="Steady utilization")
    right.axhline(70, color=GREEN, linestyle="--", linewidth=1.5, label="70% steady target")
    right.set_ylabel("Modeled post-fill utilization (%)", color=AMBER)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncols=3, loc="upper left")
    _save(fig, root / "charts/chart-04-stage-utilization.png")


def _chart_05(root: Path, summary: Mapping[str, Any]) -> None:
    network = summary["best_exact"]["network"]
    values = [
        float(network["current_attnres_bytes"]),
        float(network["cached_attnres_total_bytes"]),
        float(network["cache_reference_bytes"]),
    ]
    labels = ["Current repeated", "Seed + references", "Steady references"]
    fig, ax = _figure("Chart 05 — AttnRes network bytes", "First seed is included; steady-state references are shown separately")
    bars = ax.bar(labels, np.asarray(values) / (1024**2), color=[GRAY, BLUE, GREEN])
    ax.set_ylabel("AttnRes transport (MiB / verification block)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value / (1024**2):.2f}", ha="center", va="bottom")
    _save(fig, root / "charts/chart-05-attnres-bytes.png")


def _chart_06(root: Path, summary: Mapping[str, Any]) -> None:
    rows = [row for row in _read_csv(root / "microshards/shard-size.csv") if int(row["batch_rows"]) == 1]
    degrees = [int(row["split_degree"]) for row in rows]
    sizes = [float(row["runtime_mib_per_worker_max"]) for row in rows]
    fig, ax = _figure("Chart 06 — Native expert microshard size", "Actual uploaded native MXFP4 representation; no estimated FP4 substitution")
    ax.plot(degrees, sizes, color=BLUE, marker="o", linewidth=2.5)
    ax.axhline(1.0, color=GREEN, linestyle="--", linewidth=2, label="1 MiB target")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(degrees, [str(value) for value in degrees])
    ax.set_xlabel("Split degree")
    ax.set_ylabel("Maximum runtime MiB / logical worker")
    ax.legend(frameon=False)
    _save(fig, root / "charts/chart-06-microshard-size.png")


def _chart_07(root: Path, summary: Mapping[str, Any]) -> None:
    rows = [row for row in _read_csv(root / "microshards/network-sensitivity.csv") if int(row["batch_rows"]) == 1]
    profiles = ["local_microcell", "fast_regional", "regional_wan_like", "adverse_public_wan"]
    degrees = [8, 16, 32]
    fig, ax = _figure("Chart 07 — Microshard latency / network sensitivity", "Shaped links applied to measured real shard payload and service")
    x = np.arange(len(profiles))
    width = 0.23
    for offset, degree in enumerate(degrees):
        values = [
            float(next(row["critical_path_ms"] for row in rows if row["profile"] == profile and int(row["split_degree"]) == degree))
            for profile in profiles
        ]
        ax.bar(x + (offset - 1) * width, values, width, label=f"{degree}-way")
    ax.set_xticks(x, [value.replace("_", "\n") for value in profiles])
    ax.set_yscale("log")
    ax.set_ylabel("Critical microshard fanout + compute + reduction (ms, log)")
    ax.legend(frameon=False, ncols=3)
    _save(fig, root / "charts/chart-07-network-sensitivity.png")


def _chart_08(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _read_csv(root / "control-plane/scaling.csv")
    counts = [int(row["logical_task_count"]) for row in rows]
    coordinator = [float(row["coordinator_wall_p50_ms"]) for row in rows]
    local = [float(row["worker_local_critical_p50_ms"]) for row in rows]
    fig, ax = _figure("Chart 08 — Hierarchical control-plane scaling", "Physical CPU benchmark; only compact worker summaries enter the serial coordinator path")
    ax.plot(counts, coordinator, color=BLUE, marker="o", linewidth=2.5, label="Serial coordinator p50")
    ax.plot(counts, local, color=AMBER, marker="o", linewidth=2.5, label="Worker-local critical p50")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Logical task count")
    ax.set_ylabel("Physical CPU planning time (ms, log)")
    ax.legend(frameon=False)
    _save(fig, root / "charts/chart-08-control-plane.png")


def _chart_09(root: Path, summary: Mapping[str, Any]) -> None:
    audit = json.loads((root / "repeated-work-audit.json").read_text(encoding="utf-8"))
    rows = [row for row in audit if row.get("estimated_upper_bound_fraction") is not None]
    labels = [str(row["operation"]).replace("repeated boundary transport of ", "").replace("RMS normalization and depth-query dot for ", "") for row in rows]
    upper = [100 * float(row["estimated_upper_bound_fraction"]) for row in rows]
    measured = [100 * float(row.get("measured_gain_fraction") or 0.0) for row in rows]
    fig, ax = _figure("Chart 09 — Repeated-work bounds vs retained gain", "A bound is not counted as a gain unless a wall-time result survives")
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, upper, width, color=GRAY, label="Estimated upper bound")
    ax.bar(x + width / 2, measured, width, color=GREEN, label="Measured retained gain")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction of scoped work (%)")
    ax.legend(frameon=False)
    _save(fig, root / "charts/chart-09-repeated-work.png")


def _chart_10(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _read_csv(root / "economics/results.csv")
    labels = [row["architecture"].split()[0] for row in rows]
    cost = [float(row["projected_usd_per_1m_output_tokens"]) for row in rows]
    throughput = [float(row["tok_s_per_user"]) for row in rows]
    fig, ax = _figure("Chart 10 — System economics", "Inherited $0.15/GPU-hour and 93-equivalent snapshot; not current market pricing")
    x = np.arange(len(rows))
    bars = ax.bar(x, cost, color=[GRAY, GRAY, GRAY, BLUE])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Projected USD / 1M output tokens")
    right = ax.twinx()
    right.plot(x, throughput, color=GREEN, marker="o", linewidth=2.5)
    right.axhline(5.0, color=RED, linestyle="--", linewidth=1.5)
    right.set_ylabel("tok/s/user", color=GREEN)
    for bar, value in zip(bars, cost, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${value:.2f}", ha="center", va="bottom")
    _save(fig, root / "charts/chart-10-economics.png")


def build_all_charts(root: Path, summary: Mapping[str, Any]) -> None:
    builders: tuple[Callable[[Path, Mapping[str, Any]], None], ...] = (
        _chart_01,
        _chart_02,
        _chart_03,
        _chart_04,
        _chart_05,
        _chart_06,
        _chart_07,
        _chart_08,
        _chart_09,
        _chart_10,
    )
    for builder in builders:
        builder(root, summary)


__all__ = ["build_all_charts"]
