"""Render the twelve required Experiment 019 scientific figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
PURPLE = "#7c3aed"
GRAY = "#94a3b8"
LIGHT = "#e2e8f0"
SUBTITLE = "Explicit bounded microworkers • physical RTX 5090 shard service • no microcells"


def _evidence_subtitle(summary: Mapping[str, Any]) -> str:
    if summary.get("classification") == "MODEL_INVALID":
        return (
            "PROVISIONAL DIAGNOSTIC OUTPUT • Gate 7 serial reconstruction failed • "
            "not an admissible throughput claim"
        )
    return SUBTITLE


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _figure(title: str, subtitle: str = SUBTITLE) -> tuple[Any, Any]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": LIGHT,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    figure, axes = plt.subplots(figsize=(12, 6.75), dpi=160)
    figure.patch.set_facecolor("white")
    axes.set_facecolor("white")
    figure.suptitle(
        title,
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(0.07, 0.925, subtitle, ha="left", fontsize=9, color="#475569")
    axes.grid(axis="y", color=LIGHT, linewidth=0.8, alpha=0.8)
    axes.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.15)
    return figure, axes


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _tier_best(root: Path) -> list[dict[str, str]]:
    rows = [row for row in _csv(root / "simulation/worker-sweep.csv") if row["valid"] == "True"]
    result = []
    for tier in (20, 8, 4, 2, 1):
        selected = [row for row in rows if float(row["memory_cap_gib"]) == tier]
        if selected:
            result.append(max(selected, key=lambda row: float(row["exact_tok_s_per_user"])))
    return result


def _chart_01(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _tier_best(root)
    labels = [f"{int(float(row['memory_cap_gib']))} GiB" for row in rows]
    values = [float(row["exact_tok_s_per_user"]) for row in rows]
    figure, axes = _figure(
        "Chart 01 — Bounded-worker scheduler output by tier",
        _evidence_subtitle(summary),
    )
    bars = axes.bar(labels, values, color=[BLUE, CYAN, GREEN, AMBER, PURPLE][: len(rows)])
    axes.axhline(5.0, color=RED, linestyle="--", linewidth=2, label="5 tok/s/user target")
    axes.set_ylabel("Provisional scheduler tokens / second / user")
    axes.legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, values, strict=True):
        axes.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    _save(figure, root / "charts/chart-01-swarm-throughput.png")


def _chart_02(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _tier_best(root)
    caps = [float(row["memory_cap_gib"]) for row in rows]
    peaks = [float(row["max_worker_peak_gib"]) for row in rows]
    x = np.arange(len(rows))
    figure, axes = _figure("Chart 02 — Worker peak memory stays inside hard caps")
    axes.bar(x - 0.18, caps, 0.36, color=GRAY, label="Hard cap")
    axes.bar(x + 0.18, peaks, 0.36, color=BLUE, label="Maximum accounted peak")
    axes.set_xticks(x, [f"Tier {label}" for label in ("A", "B", "C", "D", "E")[: len(rows)]])
    axes.set_ylabel("GiB / physical worker")
    axes.legend(frameon=False)
    _save(figure, root / "charts/chart-02-worker-memory.png")


def _chart_03(root: Path, summary: Mapping[str, Any]) -> None:
    best = summary["best_exact"]
    degree = int(best["stripe_degree"])
    depth = int(best["depth_span"])
    pods = int(best["pod_count"])
    shown_pods = min(4, pods)
    manifest = json.loads(
        (root / "placement/worker-manifest-20g.json").read_text(encoding="utf-8")
    )
    worker_peaks = {
        row["worker_id"]: float(row["peak_total_bytes"]) / 1024**3
        for row in manifest["workers"]
    }
    figure, axes = _figure(
        "Chart 03 — Tier-A candidate: actual physical worker placement",
        f"{best['worker_count']} concrete workers • P={degree} • depth span={depth} • no pod compute service",
    )
    axes.set_xlim(-0.8, shown_pods * 2.4)
    axes.set_ylim(-1.75, degree + 1.2)
    axes.axis("off")
    for pod in range(shown_pods):
        x = pod * 2.4
        start = pod * depth
        stop = min(93, start + depth)
        axes.text(x + 0.75, degree + 0.7, f"POD {pod}\nlayers {start}–{stop - 1}", ha="center", fontweight="bold", color=INK)
        for stripe in range(degree):
            y = degree - stripe - 0.2
            rectangle = plt.Rectangle((x, y), 1.5, 0.62, facecolor="#eff6ff", edgecolor=BLUE, linewidth=1)
            axes.add_patch(rectangle)
            worker_id = f"pod-{pod:03d}.worker-{stripe:02d}"
            axes.text(
                x + 0.75,
                y + 0.31,
                f"{worker_id}\nstripe {stripe} • {worker_peaks[worker_id]:.3f} GiB",
                ha="center",
                va="center",
                fontsize=6.4,
            )
        axes.text(
            x + 0.75,
            -0.52,
            f"explicit P={degree} local collectives\n0.25 ms RTT / 25 Gb/s",
            ha="center",
            va="center",
            fontsize=6.5,
            color=BLUE,
        )
        if pod < shown_pods - 1:
            axes.annotate("", xy=(x + 2.25, degree / 2), xytext=(x + 1.55, degree / 2), arrowprops={"arrowstyle": "->", "color": RED, "lw": 2})
            axes.text(x + 1.9, degree / 2 + 0.25, "5 ms / 10 Gb/s\nexplicit handoff", ha="center", fontsize=7, color=RED)
    if pods > shown_pods:
        axes.text(shown_pods * 2.4 - 0.25, degree / 2, f"… {pods - shown_pods} more pods", rotation=90, va="center", color="#475569")
    axes.text(-0.72, -1.25, "wavefront snapshot →", color=INK, fontsize=7, fontweight="bold")
    for pod in range(shown_pods):
        x = pod * 2.4
        chunk = shown_pods - 1 - pod
        axes.text(
            x + 0.75,
            -1.25,
            f"chunk c{chunk} active",
            ha="center",
            va="center",
            fontsize=6.8,
            color=(BLUE, CYAN, AMBER, PURPLE)[chunk % 4],
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": (BLUE, CYAN, AMBER, PURPLE)[chunk % 4]},
        )
    _save(figure, root / "charts/chart-03-worker-placement.png")


def _chart_04(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _csv(root / "physical/expert-stripe-bank.csv")
    selected = [row for row in rows if int(row["rows"]) == 1]
    degrees = [int(row["stripe_degree"]) for row in selected]
    canonical = [float(row["canonical_whole_ms"]) for row in selected]
    stripe = [float(row["stripe_canonical_local_ms"]) for row in selected]
    naive = [float(row["naive_per_expert_local_ms"]) for row in selected]
    figure, axes = _figure("Chart 04 — Coalesced expert stripes vs per-expert RPC")
    axes.plot(degrees, canonical, marker="o", color=GRAY, linewidth=2, label="Whole-expert canonical")
    axes.plot(degrees, naive, marker="o", color=RED, linewidth=2, label="Naive 16 × P network outputs")
    axes.plot(degrees, stripe, marker="o", color=BLUE, linewidth=2.5, label="Expert stripe bank: one output / worker")
    axes.set_xscale("log", base=2)
    axes.set_xticks(degrees, [str(value) for value in degrees])
    axes.set_xlabel("Expert stripe degree P")
    axes.set_ylabel("Routed phase latency (ms, canonical local)")
    axes.legend(frameon=False)
    _save(figure, root / "charts/chart-04-expert-stripe.png")


def _chart_05(root: Path, summary: Mapping[str, Any]) -> None:
    phases = [
        ("AttnRes + norm", BLUE),
        ("P attention workers", CYAN),
        ("attention all-reduce", RED),
        ("router + latent-down", AMBER),
        ("latent all-gather", RED),
        ("P expert banks", PURPLE),
        ("latent all-reduce", RED),
        ("up + shared colocated", GREEN),
        ("hidden all-reduce", RED),
    ]
    figure, axes = _figure("Chart 05 — One sharded K3 layer is a worker DAG")
    axes.axis("off")
    for index, (label, color) in enumerate(phases):
        x = index % 5
        y = 1 - index // 5
        left = x * 2.25
        bottom = y * 2.1
        box = plt.Rectangle((left, bottom), 1.75, 0.72, facecolor="white", edgecolor=color, linewidth=2)
        axes.add_patch(box)
        axes.text(left + 0.875, bottom + 0.36, label, ha="center", va="center", fontsize=8)
        if index < len(phases) - 1:
            next_x = (index + 1) % 5
            next_y = 1 - (index + 1) // 5
            if next_y == y:
                axes.annotate("", xy=(left + 2.18, bottom + 0.36), xytext=(left + 1.78, bottom + 0.36), arrowprops={"arrowstyle": "->", "color": INK})
            else:
                axes.annotate("", xy=(0.85, 0.82), xytext=(left + 0.85, bottom - 0.08), arrowprops={"arrowstyle": "->", "color": INK, "connectionstyle": "arc3,rad=0.25"})
    axes.set_xlim(-0.4, 11.0)
    axes.set_ylim(-0.3, 3.2)
    _save(figure, root / "charts/chart-05-sharded-layer-dag.png")


def _chart_06(root: Path, summary: Mapping[str, Any]) -> None:
    path = Path(summary["best_exact"]["event_trace_path"])
    if not path.is_absolute():
        path = root.parent.parent / path
    trace = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in trace["records"] if row["resource_type"] == "microworker"]
    workers = sorted({row["worker_id"] for row in records})[:24]
    worker_index = {worker: index for index, worker in enumerate(workers)}
    figure, axes = _figure("Chart 06 — Exact wavefront over concrete workers", "First 24 active workers from winning trace; color denotes chunk")
    colormap = plt.get_cmap("turbo", max(2, max(int(row["chunk_id"]) for row in records) + 1))
    for row in records:
        worker = row["worker_id"]
        if worker not in worker_index:
            continue
        axes.barh(worker_index[worker], row["finish_time"] - row["start_time"], left=row["start_time"], height=0.7, color=colormap(int(row["chunk_id"])), edgecolor="none")
    axes.set_yticks(range(len(workers)), workers, fontsize=6.5)
    axes.invert_yaxis()
    axes.set_xlabel("Modeled critical time (ms)")
    axes.set_ylabel("Concrete worker_id")
    axes.grid(axis="x", color=LIGHT)
    axes.grid(axis="y", visible=False)
    _save(figure, root / "charts/chart-06-wavefront-workers.png")


def _chart_07(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _csv(root / "simulation/monolith-tax.csv")
    labels = [row["component"] for row in rows]
    starts = [float(row["start_ms"]) for row in rows]
    deltas = [float(row["delta_ms"]) for row in rows]
    colors = [GREEN if value < 0 else RED for value in deltas]
    figure, axes = _figure(
        "Chart 07 — Provisional E018-to-E019 latency decomposition",
        _evidence_subtitle(summary),
    )
    axes.bar(np.arange(len(rows)), deltas, bottom=starts, color=colors)
    axes.axhline(float(rows[0]["e018_target_pass_ms"]), color=GRAY, linestyle="--", label="E018 coarse wavefront")
    axes.axhline(
        float(rows[0]["e019_target_pass_ms"]),
        color=PURPLE,
        linestyle=":",
        linewidth=2,
        label="Provisional E019 scheduler output",
    )
    axes.set_xticks(np.arange(len(rows)), labels, rotation=28, ha="right")
    axes.set_ylabel("Target-pass latency (ms)")
    axes.legend(frameon=False)
    _save(figure, root / "charts/chart-07-monolith-tax.png")


def _chart_08(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _csv(root / "network/network-sensitivity.csv")
    profiles = list(dict.fromkeys(row["local_profile"] for row in rows))
    values = [float(next(row["exact_tok_s_per_user"] for row in rows if row["local_profile"] == profile and row["activation_strategy"] == "hybrid")) for profile in profiles]
    figure, axes = _figure(
        "Chart 08 — Fine-grained execution requires locality",
        _evidence_subtitle(summary),
    )
    axes.plot(range(len(profiles)), values, marker="o", linewidth=2.5, color=BLUE)
    axes.axhline(5.0, color=RED, linestyle="--", linewidth=2)
    axes.set_xticks(range(len(profiles)), [value.replace("_", "\n") for value in profiles])
    axes.set_ylabel("Provisional scheduler tok/s/user")
    _save(figure, root / "charts/chart-08-network-sensitivity.png")


def _chart_09(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _tier_best(root)
    caps = [float(row["memory_cap_gib"]) for row in rows]
    throughput = [float(row["exact_tok_s_per_user"]) for row in rows]
    figure, axes = _figure(
        "Chart 09 — Provisional worker-cap frontier",
        _evidence_subtitle(summary),
    )
    axes.plot(caps, throughput, marker="o", linewidth=3, color=BLUE)
    axes.axhline(5.0, color=RED, linestyle="--", linewidth=2, label="5 tok/s/user target")
    axes.set_xscale("log", base=2)
    axes.invert_xaxis()
    axes.set_xticks(caps, [f"{value:g} GiB" for value in caps])
    axes.set_xlabel("Hard peak resident cap")
    axes.set_ylabel("Best provisional scheduler tok/s/user")
    axes.legend(frameon=False)
    _save(figure, root / "charts/chart-09-worker-cap-vs-throughput.png")


def _chart_10(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _tier_best(root)
    workers = [int(row["worker_count"]) for row in rows]
    throughput = [float(row["exact_tok_s_per_user"]) for row in rows]
    figure, axes = _figure(
        "Chart 10 — Scheduler output does not hide worker count",
        _evidence_subtitle(summary),
    )
    axes.scatter(workers, throughput, s=90, c=[BLUE, CYAN, GREEN, AMBER, PURPLE][: len(rows)])
    for row, x, y in zip(rows, workers, throughput, strict=True):
        axes.annotate(f"{int(float(row['memory_cap_gib']))} GiB", (x, y), xytext=(6, 6), textcoords="offset points")
    axes.axhline(5.0, color=RED, linestyle="--")
    axes.set_xlabel("Resident physical workers")
    axes.set_ylabel("Provisional scheduler tok/s/user")
    _save(figure, root / "charts/chart-10-worker-count-vs-throughput.png")


def _chart_11(root: Path, summary: Mapping[str, Any]) -> None:
    rows = _csv(root / "economics/results.csv")
    selected = [row for row in rows if row["candidate_scope"] == "tier_best"]
    labels = [f"{int(float(row['memory_cap_gib']))} GiB" for row in selected]
    resident = [float(row["resident_gib_per_tok_s"]) for row in selected]
    active = [float(row["active_worker_seconds_per_output_token"]) for row in selected]
    figure, axes = _figure(
        "Chart 11 — Capacity reservation and active compute are different",
        _evidence_subtitle(summary),
    )
    x = np.arange(len(selected))
    axes.bar(x - 0.18, resident, 0.36, color=BLUE, label="Resident GiB / tok/s")
    axes.set_xticks(x, labels)
    axes.set_ylabel("Resident GiB / tok/s")
    right = axes.twinx()
    right.bar(x + 0.18, active, 0.36, color=AMBER, label="Worker-seconds / output token")
    right.set_ylabel("Active worker-seconds / output token", color=AMBER)
    lines, line_labels = axes.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    axes.legend(lines + lines2, line_labels + labels2, frameon=False)
    _save(figure, root / "charts/chart-11-economics.png")


def _chart_12(root: Path, summary: Mapping[str, Any]) -> None:
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    gates = validation["hard_gates"]
    labels = [row["gate"].replace("Gate ", "G") for row in gates]
    values = [1 if row["status"] == "PASS" else -1 for row in gates]
    colors = [GREEN if value > 0 else RED for value in values]
    figure, axes = _figure(
        "Chart 12 — Evidence stack and hard-gate disposition",
        _evidence_subtitle(summary),
    )
    axes.barh(range(len(gates)), values, color=colors)
    axes.set_yticks(range(len(gates)), labels)
    axes.axvline(0, color=INK, linewidth=1)
    axes.set_xlim(-1.05, 1.05)
    axes.set_xticks((-1, 1), ("FAIL", "PASS"))
    axes.invert_yaxis()
    for index, row in enumerate(gates):
        axes.text(
            0.03 if values[index] > 0 else -0.03,
            index,
            row["name"],
            va="center",
            ha="left" if values[index] > 0 else "right",
            color="white",
            fontsize=8,
        )
    _save(figure, root / "charts/chart-12-evidence-stack.png")


def build_all_charts(root: Path, summary: Mapping[str, Any]) -> list[Path]:
    builders = (
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
        _chart_11,
        _chart_12,
    )
    for builder in builders:
        builder(root, summary)
    return sorted((root / "charts").glob("chart-*.png"))


__all__ = ["build_all_charts"]
