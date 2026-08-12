"""Static, source-backed figures for Experiment 015."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm_inference.experiments.experiment_015.evidence import read_json

BLUE = "#2563EB"
ORANGE = "#EA580C"
TEAL = "#0F766E"
GRAY = "#64748B"
LIGHT = "#CBD5E1"
RED = "#B91C1C"


def _finish(
    figure: plt.Figure,
    axis: plt.Axes,
    path: Path,
    *,
    source: str,
    legend: bool = False,
) -> None:
    if legend:
        axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.01, 0.01, source, fontsize=7, color=GRAY)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    figure.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(figure)


def _bar_labels(axis: plt.Axes, bars: Any, *, digits: int = 2) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _blank_figure(title: str, message: str, path: Path, *, source: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(0.5, 0.55, "NOT ESTABLISHED", ha="center", va="center", fontsize=20, color=RED)
    axis.text(0.5, 0.39, message, ha="center", va="center", fontsize=10, color=GRAY, wrap=True)
    axis.set_axis_off()
    _finish(figure, axis, path, source=source)


def generate_figures(repository_root: Path, output_directory: Path) -> list[Path]:
    """Generate the twenty preregistered figures with visible evidence labels."""
    root = repository_root.expanduser().resolve()
    out = output_directory.expanduser().resolve()
    baseline = read_json(root / "artifacts" / "experiment-015" / "baseline" / "B015-000.json")
    architecture = read_json(
        root / "artifacts" / "experiment-015" / "architecture-pareto" / "results.json"
    )
    microwork = read_json(root / "artifacts" / "experiment-015" / "microwork" / "results.json")
    dcp = read_json(root / "artifacts" / "experiment-015" / "dcp" / "results.json")
    tp = read_json(root / "artifacts" / "experiment-015" / "tp" / "results.json")
    routing = read_json(root / "artifacts" / "experiment-015" / "expert-routing" / "results.json")
    residency = read_json(root / "artifacts" / "experiment-015" / "expert-residency" / "results.json")
    pipeline = read_json(
        root / "artifacts" / "experiment-015" / "pipeline-occupancy" / "results.json"
    )
    paths: list[Path] = []

    def save(number: int, slug: str) -> Path:
        path = out / f"{number:02d}-{slug}.png"
        paths.append(path)
        return path

    base_dep = float(baseline["performance"]["dependency_bound_tok_s"])
    best_cell = architecture["best_justified_result"]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(
        ["B015-000\nPROJECTED", "Depth-8 cells\nSHAPED"],
        [base_dep, float(best_cell["dependency_bound_tok_s"])],
        color=[GRAY, BLUE],
        zorder=2,
    )
    _bar_labels(axis, bars)
    axis.axhline(5, color=RED, linestyle="--", label="5 tok/s target")
    axis.set_ylabel("Dependency-bound output tok/s")
    axis.set_title("Dependency speed: baseline vs best justified result", loc="left", fontweight="bold")
    _finish(figure, axis, save(1, "baseline-vs-best-dependency"), source="Source: B015-000 + SHAPED microcell topology model. No Experiment 015 end-to-end measurement.", legend=True)

    public_acceptance = {
        "Coding": 4.38,
        "Math/reasoning": (5.64 + 3.82 + 2.72) / 3.0,
        "General chat": 3.14,
        "Creative": 2.79,
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(
        list(public_acceptance),
        [1.0 / value for value in public_acceptance.values()],
        color=ORANGE,
        zorder=2,
    )
    _bar_labels(axis, bars, digits=3)
    axis.set_ylabel("Target traversals / accepted token")
    axis.set_title("Traversal sensitivity from public DSpark acceptance", loc="left", fontweight="bold")
    _finish(figure, axis, save(2, "target-traversals-per-accepted-token"), source="EXTERNAL REFERENCE SENSITIVITY: Inferact Kimi-K3-DSpark model card; not Swarm evidence.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    block_sizes = [1, 2, 3, 5, 7]
    axis.plot(block_sizes, [size + 1 for size in block_sizes], linestyle="--", color=LIGHT, marker="o", label="mathematical maximum")
    axis.text(4.0, 2.0, "Swarm accepted length was not measured", color=RED, ha="center", fontsize=11)
    axis.set_xticks(block_sizes)
    axis.set_xlabel("Draft block size")
    axis.set_ylabel("Accepted output tokens / target pass")
    axis.set_ylim(0, 8.5)
    axis.set_title("DSpark block size vs accepted length", loc="left", fontweight="bold")
    _finish(figure, axis, save(3, "dspark-block-size-vs-accepted-length"), source="The dashed line is a bound, not an observation. Representative Swarm target verification did not complete.", legend=True)

    target_pass_ms = float(architecture["external_dspark_sensitivity"]["dspark_plus_depth8"]["target_pass_ms"])
    speeds = [value * 1000.0 / target_pass_ms / base_dep for value in public_acceptance.values()]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(list(public_acceptance), speeds, color=ORANGE, zorder=2)
    _bar_labels(axis, bars)
    axis.axhline(1.0, color=GRAY, linewidth=1)
    axis.axhline(2.0, color=RED, linestyle="--", label="first speculation gate")
    axis.set_ylabel("Speedup vs 0.95 tok/s")
    axis.set_title("Speculative speedup by workload: target-only sensitivity", loc="left", fontweight="bold")
    _finish(figure, axis, save(4, "speculative-speedup-by-workload"), source="PROJECTED sensitivity using external acceptance; excludes draft GPU latency/cost.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    occupancy = float(pipeline["baseline_useful_target_stage_occupancy_percent"])
    bars = axis.bar(["Autoregressive\nprojected", "Synchronous DSpark\nnot established", "Async DSpark\nnot established"], [occupancy, 0.0, 0.0], color=[BLUE, LIGHT, LIGHT], zorder=2)
    _bar_labels(axis, bars)
    axis.text(1, 0.08, "N/A", ha="center", color=RED)
    axis.text(2, 0.08, "N/A", ha="center", color=RED)
    axis.set_ylabel("Useful target stage-time (%)")
    axis.set_title("Pipeline utilization with/without async speculation", loc="left", fontweight="bold")
    _finish(figure, axis, save(5, "pipeline-utilization"), source="Baseline is a service-model ratio. DSpark occupancy was not executed and zero-height bars mean N/A, not zero utilization.")

    cells = architecture["microcell_sweep"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["layers_per_cell"] for row in cells], [row["dependency_latency_ms"] for row in cells], marker="o", color=BLUE)
    axis.set_xticks([1, 2, 4, 8])
    axis.set_xlabel("Layers per microcell")
    axis.set_ylabel("Dependency latency (ms)")
    axis.set_title("Macrocell depth vs dependency latency", loc="left", fontweight="bold")
    _finish(figure, axis, save(6, "macrocell-depth-vs-latency"), source="SHAPED; held-out-validated component model with internal 0.25 ms/25 Gbps and coarse 5 ms/10 Gbps.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["coarse_boundaries"] for row in cells], [row["dependency_bound_tok_s"] for row in cells], marker="o", color=TEAL)
    for row in cells:
        axis.annotate(f"d={row['layers_per_cell']}", (row["coarse_boundaries"], row["dependency_bound_tok_s"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Coarse inter-cell boundaries / token")
    axis.set_ylabel("Dependency-bound tok/s")
    axis.set_title("Coarse boundaries vs stream speed", loc="left", fontweight="bold")
    _finish(figure, axis, save(7, "coarse-boundaries-vs-stream-speed"), source="SHAPED; compute remains sequential through all 93 layers.")

    ep_rows = microwork["measured_scaling"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([str(row["workers"]) for row in ep_rows], [100.0 * float(row["relative_complete_layer_throughput"]) for row in ep_rows], color=BLUE, zorder=2)
    _bar_labels(axis, bars, digits=1)
    axis.axhline(90, color=RED, linestyle="--", label="strong research target")
    axis.set_xlabel("Expert workers")
    axis.set_ylabel("Complete-layer throughput vs one GPU (%)")
    axis.set_title("Expert worker count vs complete-layer throughput", loc="left", fontweight="bold")
    _finish(figure, axis, save(8, "expert-workers-vs-layer-throughput"), source="MEASURED real Kimi/CUDA; four-worker point uses the promoted Experiment 014 repeat.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([str(row["workers"]) for row in ep_rows], [float(row["worker_gib"]) for row in ep_rows], color=TEAL, zorder=2)
    _bar_labels(axis, bars)
    axis.set_xlabel("Expert workers")
    axis.set_ylabel("Tracked GiB / logical worker")
    axis.set_title("Expert worker count vs resident memory", loc="left", fontweight="bold")
    _finish(figure, axis, save(9, "expert-workers-vs-memory"), source="MEASURED tracked device bytes; runtime/headroom can require more physical VRAM.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([str(row["workers"]) for row in ep_rows], [float(row["mean_transport_bytes_per_token"]) / 1000 for row in ep_rows], color=ORANGE, zorder=2)
    _bar_labels(axis, bars, digits=0)
    axis.set_xlabel("Expert workers")
    axis.set_ylabel("Mean transport kB / layer-token")
    axis.set_title("Expert worker count vs network demand", loc="left", fontweight="bold")
    _finish(figure, axis, save(10, "expert-workers-vs-network"), source="MEASURED actual Kimi payloads from retained microwork runs.")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for context in (1024, 4096, 8192, 16384):
        subset = [row for row in dcp["performance_rows"] if int(row["context_tokens"]) == context]
        axis.plot([row["degree"] for row in subset], [row["projected_mla_stage_ms"] for row in subset], marker="o", label=f"{context // 1024}K")
    axis.set_xticks([1, 2, 4, 8])
    axis.set_xlabel("DCP workers")
    axis.set_ylabel("MLA stage service (ms)")
    axis.set_title("DCP workers vs MLA service by context", loc="left", fontweight="bold")
    _finish(figure, axis, save(11, "dcp-workers-vs-mla-service"), source="SHAPED from real Kimi context timings + exact partial payload; no distributed Kimi DCP kernel.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for operation in ("KDA q projection", "KDA output projection", "MLA five-projection group", "LM head"):
        subset = [row for row in tp["rows"] if row["operation"] == operation]
        axis.plot([row["degree"] for row in subset], [row["operation_latency_ms"] for row in subset], marker="o", label=operation)
    axis.set_xticks([1, 2, 4])
    axis.set_xlabel("TP degree")
    axis.set_ylabel("Operation latency (ms)")
    axis.set_title("TP degree vs operation latency", loc="left", fontweight="bold")
    _finish(figure, axis, save(12, "tp-degree-vs-latency"), source="TP1 MEASURED real CUDA; TP2/4 SHAPED ideal compute plus collective. Logits/head dominates scale.", legend=True)

    static = routing["static_modulo_placement"]
    balanced = routing["trace_fitted_balanced_placement"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    width = 0.36
    axis.bar(x - width / 2, [static["mean_critical_worker_selections"], static["p95_critical_worker_selections"], static["maximum_critical_worker_selections"]], width, label="static", color=GRAY)
    axis.bar(x + width / 2, [balanced["mean_critical_worker_selections"], balanced["p95_critical_worker_selections"], balanced["maximum_critical_worker_selections"]], width, label="trace-fitted", color=BLUE)
    axis.set_xticks(x, ["mean", "p95", "max"])
    axis.set_ylabel("Selections on critical worker / route row")
    axis.set_title("Expert load distribution", loc="left", fontweight="bold")
    _finish(figure, axis, save(13, "expert-load-distribution"), source="MEASURED three-token, 92-layer trace. Fitted plan has no held-out evaluation.", legend=True)

    _blank_figure("Hot-expert replication effect", "No representative route corpus passed the replication admission gate.", save(14, "hot-expert-replication-effect"), source="Three-token trace was insufficient; no VRAM/latency replication result is reported.")

    predictor = routing["previous_token_same_layer_predictor"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(["Precision", "Recall", "Maximum useful\ntransport fraction"], [100 * float(predictor["precision"]), 100 * float(predictor["recall"]), 100 * float(predictor["useful_activation_bytes"]) / float(predictor["predicted_activation_bytes"])], color=[BLUE, TEAL, ORANGE], zorder=2)
    _bar_labels(axis, bars, digits=1)
    axis.set_ylabel("Percent")
    axis.set_title("Route prediction usefulness", loc="left", fontweight="bold")
    _finish(figure, axis, save(15, "route-prediction-usefulness"), source="MEASURED previous-token predictor on three real Kimi tokens; not representative.")

    residence_rows = residency["rows"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([100 * row["gpu_residency_fraction"] for row in residence_rows], [100 * row["cold_start_hit_rate"] for row in residence_rows], marker="o", color=ORANGE)
    axis.text(62, 10, "Throughput NOT ESTABLISHED", color=RED, fontsize=10)
    axis.set_xlabel("GPU expert residency (%)")
    axis.set_ylabel("Cold-start LRU hit rate (%)")
    axis.set_title("GPU residency fraction vs observed cache hit", loc="left", fontweight="bold")
    _finish(figure, axis, save(16, "residency-vs-throughput"), source="PROJECTED cold-start cache replay over only three tokens; not a throughput benchmark.")

    plotted = [row for row in architecture["rows"] if row["architecture_id"] in {"A", "B", "D", "I"}]
    labels = [row["architecture_id"] for row in plotted]
    colors = [BLUE if row["admitted_to_decision_frontier"] else ORANGE for row in plotted]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, [float(row["aggregate_tok_s_per_paid_gpu_equivalent"]) for row in plotted], color=colors, zorder=2)
    _bar_labels(axis, bars)
    axis.axhline(2.78, color=RED, linestyle="--", label="break-even target")
    axis.set_xlabel("Architecture (orange = unadmitted sensitivity)")
    axis.set_ylabel("Aggregate tok/s / paid-GPU-equivalent")
    axis.set_title("Throughput per paid GPU by architecture", loc="left", fontweight="bold")
    _finish(figure, axis, save(17, "throughput-per-paid-gpu"), source="A/D are decision-model rows. B/I use external acceptance and exclude draft paid compute.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, [float(row["dependency_bound_tok_s"]) for row in plotted], color=colors, zorder=2)
    _bar_labels(axis, bars)
    axis.axhline(5.0, color=RED, linestyle="--", label="interactive target")
    axis.set_xlabel("Architecture (orange = unadmitted sensitivity)")
    axis.set_ylabel("Dependency-bound tok/s")
    axis.set_title("Dependency-bound speed by architecture", loc="left", fontweight="bold")
    _finish(figure, axis, save(18, "dependency-by-architecture"), source="A PROJECTED, D SHAPED, B/I external-acceptance sensitivities.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(labels, [float(row["cost_per_million_output_tokens_usd"]) for row in plotted], color=colors, zorder=2)
    _bar_labels(axis, bars)
    axis.axhline(15.0, color=RED, linestyle="--", label="$15/M selling price")
    axis.set_xlabel("Architecture (orange = unadmitted sensitivity)")
    axis.set_ylabel("GPU cost / million output tokens (USD)")
    axis.set_title("Cost per million output tokens", loc="left", fontweight="bold")
    _finish(figure, axis, save(19, "cost-per-million"), source="$0.15 per paid-GPU-hour. Draft cost excluded from B/I, making them optimistic.", legend=True)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    for row in plotted:
        x_value = float(row["dependency_bound_tok_s"])
        y_value = float(row["aggregate_tok_s_per_paid_gpu_equivalent"])
        admitted = bool(row["admitted_to_decision_frontier"])
        axis.scatter(x_value, y_value, s=90, color=BLUE if admitted else "white", edgecolor=BLUE if admitted else ORANGE, linewidth=2, zorder=3)
        axis.annotate(row["architecture_id"], (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontweight="bold")
    axis.axvline(5.0, color=RED, linestyle="--", linewidth=1)
    axis.axhline(2.78, color=RED, linestyle="--", linewidth=1)
    axis.set_xlim(0, 5.4)
    axis.set_ylim(0, 3.05)
    axis.set_xlabel("Dependency-bound tok/s")
    axis.set_ylabel("Aggregate tok/s / paid-GPU-equivalent")
    axis.set_title("Experiment 015 architecture Pareto view", loc="left", fontweight="bold")
    _finish(figure, axis, save(20, "pareto-frontier"), source="Filled = admitted decision evidence; hollow = external/unadmitted sensitivity. Red lines are simultaneous targets.")

    return paths


__all__ = ["generate_figures"]
