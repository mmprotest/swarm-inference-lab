"""Source-backed E021 chart suite with explicit MODEL_INVALID treatment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .io import atomic_write_json, sha256_file

CAPS = (8, 4, 2, 1)
REGIMES = ("A", "B", "C", "D")
COLORS = {
    "A": "#2A9D8F",
    "B": "#457B9D",
    "C": "#E9C46A",
    "D": "#E76F51",
    "compute": "#496A81",
    "network": "#D95D39",
    "pass": "#2A9D8F",
    "fail": "#C1121F",
    "not_measured": "#8D99AE",
}


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _invalid_banner(figure: Any) -> None:
    figure.subplots_adjust(top=0.82, bottom=0.14)
    figure.text(
        0.5,
        0.985,
        "MODEL INVALID — diagnostic projections only; no admissible exact throughput",
        ha="center",
        va="top",
        color="#FFFFFF",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#9B2226", "edgecolor": "none"},
    )


def _source(figure: Any, text: str) -> None:
    figure.text(0.01, 0.012, text, fontsize=7.5, color="#5B6573")


def _style(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _best_sweep(rows: list[dict[str, str]], cap: int, regime: str) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if int(float(row["memory_cap_gib"])) == cap and row["regime"] == regime
    ]
    return max(candidates, key=lambda row: float(row["exact_tok_s_per_user"]))


def chart_memory_curve(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "memory-network-curve.csv")
    figure, axis = plt.subplots(figsize=(10, 5.7))
    for regime in REGIMES:
        values = [
            float(
                next(
                    row["diagnostic_tok_s_per_user"]
                    for row in rows
                    if int(row["worker_memory_cap_gib"]) == cap
                    and row["regime"] == regime
                )
            )
            for cap in CAPS
        ]
        axis.plot(
            range(len(CAPS)),
            values,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=2,
            linewidth=2,
            color=COLORS[regime],
            label=f"Regime {regime}",
        )
    axis.axhline(5.0, color="#111827", linestyle="--", linewidth=1.5, label="5 tok/s target")
    axis.set_xticks(range(len(CAPS)), [f"{cap} GiB" for cap in CAPS])
    axis.set_xlabel("Total peak memory cap per independent machine")
    axis.set_ylabel("Diagnostic modeled tok/s/user (inadmissible)")
    axis.set_title("Memory fragmentation by independent-machine network regime", pad=28)
    axis.legend(ncol=3, frameon=False, loc="upper right")
    _style(axis)
    _invalid_banner(figure)
    _source(figure, "Source: simulation/memory-network-curve.csv · Hollow markers denote invalid-model outputs")
    _save(figure, output)


def chart_network_envelope(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "network-envelope.csv")
    rtts = sorted({float(row["rtt_ms"]) for row in rows})
    bandwidths = sorted({float(row["bandwidth_gbps"]) for row in rows})
    values = np.asarray(
        [
            [
                float(
                    next(
                        row["exact_tok_s_per_user"]
                        for row in rows
                        if float(row["rtt_ms"]) == rtt
                        and float(row["bandwidth_gbps"]) == bandwidth
                    )
                )
                for rtt in rtts
            ]
            for bandwidth in bandwidths
        ],
        dtype=np.float64,
    )
    x = np.log10(rtts)
    y = np.log10(bandwidths)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    cmap = LinearSegmentedColormap.from_list(
        "e021", ["#6D1A36", "#E76F51", "#F4A261", "#E9C46A", "#2A9D8F"]
    )
    image = axis.contourf(x, y, values, levels=18, cmap=cmap)
    if float(np.min(values)) <= 5.0 <= float(np.max(values)):
        contour = axis.contour(x, y, values, levels=[5.0], colors="white", linewidths=2.5)
        axis.clabel(contour, fmt={5.0: "diagnostic 5 tok/s"}, fontsize=9)
    axis.scatter(
        np.repeat(x, len(y)),
        np.tile(y, len(x)),
        s=10,
        color="white",
        alpha=0.35,
    )
    axis.set_xticks(x, [f"{value:g}" for value in rtts])
    axis.set_yticks(y, [f"{value:g}" for value in bandwidths])
    axis.set_xlabel("RTT between independent machines (ms, log-spaced)")
    axis.set_ylabel("Link bandwidth (Gb/s, log-spaced)")
    axis.set_title("8 GiB candidate: diagnostic RTT by bandwidth envelope", pad=28)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Diagnostic modeled tok/s/user (inadmissible)")
    _invalid_banner(figure)
    _source(figure, "Source: simulation/network-envelope.csv · White contour is not an admissible 5 tok/s region")
    _save(figure, output)


def chart_worker_count(root: Path, output: Path) -> None:
    rows = _csv(root / "placement" / "worker-memory-tiers.csv")
    by_cap = {int(row["worker_cap_gib"]): row for row in rows}
    counts = [int(by_cap[cap]["worker_count"]) for cap in CAPS]
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    bars = axis.bar(range(len(CAPS)), counts, color="#496A81", width=0.62)
    axis.bar_label(bars, labels=[f"{value:,}" for value in counts], padding=4)
    axis.set_xticks(range(len(CAPS)), [f"{cap} GiB" for cap in CAPS])
    axis.set_xlabel("Peak memory cap per independent machine")
    axis.set_ylabel("Independent machines / compute workers")
    axis.set_title("Exact full-checkpoint placement expands as memory fragments")
    _style(axis)
    _source(figure, "Source: placement/worker-memory-tiers.csv · One worker = one independent machine")
    _save(figure, output)


def chart_critical_path(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "sweep.csv")
    selected = [_best_sweep(rows, 8, regime) for regime in REGIMES]
    compute = [float(row["critical_path_compute_ms"]) for row in selected]
    network = [float(row["network_critical_path_ms"]) for row in selected]
    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(REGIMES))
    axis.bar(x, compute, color=COLORS["compute"], label="Worker compute")
    axis.bar(x, network, bottom=compute, color=COLORS["network"], label="Network")
    axis.set_xticks(x, [f"Regime {value}" for value in REGIMES])
    axis.set_ylabel("Diagnostic critical-path milliseconds")
    axis.set_title("8 GiB candidate critical path: network rapidly dominates", pad=28)
    axis.legend(frameon=False)
    _style(axis)
    _invalid_banner(figure)
    _source(figure, "Source: simulation/sweep.csv · Best block/chunk selected independently within each regime")
    _save(figure, output)


def chart_network_compute(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "sweep.csv")
    selected = [_best_sweep(rows, 8, regime) for regime in REGIMES]
    compute = [100 * float(row["compute_share_of_critical_path"]) for row in selected]
    network = [100 * float(row["network_share_of_critical_path"]) for row in selected]
    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(REGIMES))
    axis.bar(x, compute, color=COLORS["compute"], label="Compute share")
    axis.bar(x, network, bottom=compute, color=COLORS["network"], label="Network share")
    axis.set_xticks(x, [f"Regime {value}" for value in REGIMES])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Share of diagnostic critical path (%)")
    axis.set_title("Network versus compute on the diagnostic critical path", pad=28)
    axis.legend(frameon=False, ncol=2)
    _style(axis)
    _invalid_banner(figure)
    _source(figure, "Source: simulation/sweep.csv")
    _save(figure, output)


def chart_whole_layer_control(root: Path, output: Path) -> None:
    sweep = _csv(root / "simulation" / "sweep.csv")
    swarm = _best_sweep(sweep, 8, "B")
    controls = _csv(root / "simulation" / "whole-layer-control.csv")
    control = max(controls, key=lambda row: float(row["exact_tok_s_per_user"]))
    labels = ["8 GiB sub-layer\n376 machines", "20 GiB whole-layer\n93 machines"]
    values = [float(swarm["exact_tok_s_per_user"]), float(control["exact_tok_s_per_user"])]
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    bars = axis.bar(range(2), values, color=["#8D99AE", "#2A9D8F"], width=0.58)
    axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4)
    axis.set_xticks(range(2), labels)
    axis.set_ylabel("Diagnostic modeled tok/s/user")
    axis.set_title("Fragmentation tax control at Regime B", pad=28)
    axis.text(
        0,
        values[0] * 0.5,
        "MODEL\nINVALID",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    _style(axis)
    _invalid_banner(figure)
    _source(figure, "Sources: simulation/sweep.csv; simulation/whole-layer-control.csv · Controls are not headline Swarm evidence")
    _save(figure, output)


def chart_utilization(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "worker-utilization.csv")
    figure, axis = plt.subplots(figsize=(10, 5.6))
    width = 0.19
    x = np.arange(len(CAPS))
    for offset, regime in enumerate(REGIMES):
        values = [
            100
            * float(
                next(
                    row["average_worker_utilization"]
                    for row in rows
                    if int(row["worker_memory_cap_gib"]) == cap
                    and row["regime"] == regime
                )
            )
            for cap in CAPS
        ]
        axis.bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=f"Regime {regime}",
            color=COLORS[regime],
        )
    axis.set_xticks(x, [f"{cap} GiB" for cap in CAPS])
    axis.set_xlabel("Peak memory cap per independent machine")
    axis.set_ylabel("Average diagnostic worker utilization (%)")
    axis.set_title("One-request utilization remains low across fragmentation tiers", pad=28)
    axis.legend(frameon=False, ncol=4)
    _style(axis)
    _invalid_banner(figure)
    _source(figure, "Source: simulation/worker-utilization.csv")
    _save(figure, output)


def chart_market(root: Path, output: Path) -> None:
    data = json.loads(
        (root / "vast" / "fragmented-fleet-feasibility.json").read_text(encoding="utf-8")
    )
    by_cap = {int(row["worker_memory_cap_gib"]): row for row in data["tiers"]}
    required = [int(by_cap[cap]["required_independent_machines"]) for cap in CAPS]
    available = [int(by_cap[cap]["unique_kernel_compatible_machines"]) for cap in CAPS]
    figure, axis = plt.subplots(figsize=(9.8, 5.6))
    x = np.arange(len(CAPS))
    width = 0.35
    left = axis.bar(x - width / 2, required, width, color="#496A81", label="Required")
    right = axis.bar(x + width / 2, available, width, color="#E76F51", label="Observed compatible")
    axis.bar_label(left, labels=[f"{value:,}" for value in required], padding=3)
    axis.bar_label(right, labels=[f"{value:,}" for value in available], padding=3)
    axis.set_xticks(x, [f"{cap} GiB" for cap in CAPS])
    axis.set_xlabel("Worker memory tier")
    axis.set_ylabel("Unique independent single-GPU machines")
    axis.set_title("Read-only fragmented Vast inventory is insufficient")
    axis.legend(frameon=False)
    _style(axis)
    _source(figure, f"Source: vast/fragmented-fleet-feasibility.json · Snapshot {data['snapshot_at']} · Offer metadata is not RTT evidence")
    _save(figure, output)


def chart_concurrency(root: Path, output: Path) -> None:
    rows = _csv(root / "simulation" / "concurrency.csv")
    requests = [int(row["concurrent_requests"]) for row in rows]
    per_user = [float(row["per_user_tok_s_p50_diagnostic"]) for row in rows]
    aggregate = [float(row["aggregate_tok_s_diagnostic"]) for row in rows]
    figure, left = plt.subplots(figsize=(9.8, 5.6))
    right = left.twinx()
    left.plot(requests, per_user, color="#457B9D", marker="o", markerfacecolor="white", markeredgewidth=2, linewidth=2, label="Per user")
    right.plot(requests, aggregate, color="#E76F51", marker="s", markerfacecolor="white", markeredgewidth=2, linewidth=2, label="Aggregate")
    left.axhline(5.0, color="#111827", linestyle="--", linewidth=1.2)
    left.set_xscale("log", base=2)
    left.set_xticks(requests, [str(value) for value in requests])
    left.set_xlabel("Concurrent real requests")
    left.set_ylabel("Diagnostic per-user tok/s", color="#457B9D")
    right.set_ylabel("Diagnostic aggregate tok/s", color="#E76F51")
    left.set_title("Concurrency occupies capacity but erodes per-user rate", pad=28)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.grid(axis="y", color="#D9DEE5", linewidth=0.7)
    lines = left.get_lines()[:1] + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    _invalid_banner(figure)
    _source(figure, "Source: simulation/concurrency.csv · 8 GiB / Regime B diagnostic model")
    _save(figure, output)


def chart_evidence_stack(root: Path, output: Path) -> None:
    gates = [
        ("Checkpoint\nplacement", "PASS"),
        ("Whole-layer\ninfeasibility", "PASS"),
        ("Ordered physical\nmodel replay", "FAIL"),
        ("Event\naccounting", "PASS"),
        ("Worker-process\n93 layers", "FAIL"),
        ("Controller\n2,000", "PASS"),
        ("Market fleet\navailability", "FAIL"),
        ("Physical\nSwarm", "NOT MEASURED"),
    ]
    colors = [
        COLORS["pass"] if status == "PASS" else COLORS["fail"] if status == "FAIL" else COLORS["not_measured"]
        for _, status in gates
    ]
    figure, axis = plt.subplots(figsize=(11, 5.7))
    bars = axis.bar(range(len(gates)), [1] * len(gates), color=colors, width=0.72)
    axis.set_xticks(range(len(gates)), [name for name, _ in gates], fontsize=8.5)
    axis.set_ylim(0, 1.2)
    axis.set_yticks([])
    axis.set_title("E021 evidence stack stops at model validation")
    for bar, (_, status) in zip(bars, gates, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            0.52,
            status.replace(" ", "\n"),
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            fontweight="bold",
        )
    axis.spines[:].set_visible(False)
    axis.text(
        2,
        1.09,
        "Decisive stop: MODEL_INVALID",
        ha="center",
        color=COLORS["fail"],
        fontweight="bold",
    )
    _source(figure, "Sources: E021 placement, validation, correctness, control-plane, Vast, and summary receipts")
    _save(figure, output)


def generate_charts(artifact_root: Path) -> dict[str, Any]:
    chart_root = artifact_root / "charts"
    definitions = [
        ("chart-01-memory-fragmentation-curve.png", chart_memory_curve),
        ("chart-02-network-envelope.png", chart_network_envelope),
        ("chart-03-worker-count.png", chart_worker_count),
        ("chart-04-critical-path.png", chart_critical_path),
        ("chart-05-network-vs-compute.png", chart_network_compute),
        ("chart-06-whole-layer-control.png", chart_whole_layer_control),
        ("chart-07-worker-utilization.png", chart_utilization),
        ("chart-08-market-fragmented-inventory.png", chart_market),
        ("chart-09-concurrency.png", chart_concurrency),
        ("chart-10-evidence-stack.png", chart_evidence_stack),
    ]
    rows = []
    for filename, function in definitions:
        path = chart_root / filename
        function(artifact_root, path)
        rows.append(
            {
                "file": f"charts/{filename}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "exists": True,
            }
        )
    receipt = {
        "schema_version": "experiment-021-chart-index-v1",
        "status": "PASS" if len(rows) == 10 and all(row["bytes"] > 0 for row in rows) else "FAIL",
        "chart_count": len(rows),
        "model_invalid_throughput_policy": (
            "diagnostic values are labeled inadmissible and exact/admissible throughput is suppressed"
        ),
        "charts": rows,
    }
    atomic_write_json(artifact_root / "chart-index.json", receipt)
    return receipt


__all__ = ["generate_charts"]
