"""Evidence charts and answer-first HTML report for Experiment 007."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REQUIRED_CHARTS = (
    "sglang_gpu_baseline.png",
    "mixed_backend_latency.png",
    "backend_boundary_breakdown.png",
    "cpu_draft_acceptance.png",
    "cpu_draft_speedup.png",
    "speculative_break_even.png",
    "cpu_expert_memory_savings.png",
    "cpu_expert_latency.png",
    "expert_placement_tradeoff.png",
    "background_throughput.png",
    "interactive_latency_impact.png",
    "role_utility.png",
    "planner_prediction_vs_actual.png",
    "planner_regret.png",
    "contribution_frontier.png",
    "availability_break_even.png",
    "network_role_viability.png",
)


def _placeholder(path: Path, title: str, detail: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, detail, ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    *,
    horizontal: bool = False,
) -> None:
    if not labels or not values:
        _placeholder(path, title, "No measured observations were available for this arm.")
        return
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
    if horizontal:
        axis.barh(labels, values, color="#31688e")
        axis.set_xlabel(ylabel)
    else:
        axis.bar(labels, values, color="#31688e")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
    axis.axhline(0, color="#333333", linewidth=0.8) if not horizontal else axis.axvline(
        0, color="#333333", linewidth=0.8
    )
    axis.set_title(title)
    axis.grid(axis="y" if not horizontal else "x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_heterogeneous_charts(
    run_directory: Path,
    *,
    sglang: list[dict[str, Any]],
    mixed: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    speculative: list[dict[str, Any]],
    break_even: list[dict[str, Any]],
    experts: list[dict[str, Any]],
    background: list[dict[str, Any]],
    planner_measurements: list[dict[str, Any]],
    regret: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    availability: list[dict[str, Any]],
    network: list[dict[str, Any]],
) -> None:
    charts = run_directory / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    _bar(
        charts / "sglang_gpu_baseline.png",
        "Stock SGLang verified throughput",
        [f"{row.get('workload')} c{row.get('concurrency')}" for row in sglang],
        [float(row.get("aggregate_verified_throughput", 0)) for row in sglang],
        "verified output tokens/s",
    )
    _bar(
        charts / "mixed_backend_latency.png",
        "Mixed CUDA/CPU latency",
        [str(row.get("route", "mixed")) for row in mixed],
        [float(row.get("end_to_end_ms", 0)) for row in mixed],
        "milliseconds",
    )
    _bar(
        charts / "backend_boundary_breakdown.png",
        "Backend-boundary latency",
        [f"s{row.get('stage_id')}-{row.get('operation')}" for row in boundaries[:64]],
        [float(row.get("round_trip_ms", 0)) for row in boundaries[:64]],
        "milliseconds",
    )
    _bar(
        charts / "cpu_draft_acceptance.png",
        "CPU draft acceptance",
        [f"{row.get('weight_format')} L{row.get('draft_length')}" for row in speculative],
        [100 * float(row.get("acceptance_rate", 0)) for row in speculative],
        "accepted candidates (%)",
    )
    _bar(
        charts / "cpu_draft_speedup.png",
        "Lossless speculative throughput change",
        [f"{row.get('weight_format')} L{row.get('draft_length')}" for row in speculative],
        [100 * float(row.get("speedup_fraction", 0)) for row in speculative],
        "change (%)",
    )
    representative = [row for row in break_even if float(row.get("one_way_latency_ms", -1)) == 10]
    _bar(
        charts / "speculative_break_even.png",
        "Speculative break-even at 10 ms one-way latency",
        [str(row.get("draft_speed_tokens_per_second")) for row in representative],
        [100 * float(row.get("speedup_fraction", 0)) for row in representative],
        "change (%)",
    )
    _bar(
        charts / "cpu_expert_memory_savings.png",
        "CPU expert GPU-memory relief",
        [f"{row.get('placement_policy')} {row.get('cpu_expert_count')}" for row in experts],
        [float(row.get("gpu_memory_saved_bytes", 0)) / 2**30 for row in experts],
        "GiB",
    )
    _bar(
        charts / "cpu_expert_latency.png",
        "Hybrid MoE layer latency",
        [f"{row.get('placement_policy')} {row.get('cpu_expert_count')}" for row in experts],
        [float(row.get("hybrid_layer_latency_ms", 0)) for row in experts],
        "milliseconds",
    )
    _bar(
        charts / "expert_placement_tradeoff.png",
        "Expert placement throughput retained",
        [f"{row.get('placement_policy')} {row.get('weight_format')}" for row in experts],
        [100 * float(row.get("throughput_retained_fraction", 0)) for row in experts],
        "baseline throughput retained (%)",
    )
    _bar(
        charts / "background_throughput.png",
        "Combined verified background capacity",
        [f"g{row.get('gpu_concurrency')}/c{row.get('cpu_concurrency')}" for row in background],
        [float(row.get("total_combined_verified_tokens_per_second", 0)) for row in background],
        "verified output tokens/s",
    )
    _bar(
        charts / "interactive_latency_impact.png",
        "Interactive p95 latency impact",
        [f"g{row.get('gpu_concurrency')}/c{row.get('cpu_concurrency')}" for row in background],
        [100 * float(row.get("interactive_p95_increase_fraction", 0)) for row in background],
        "change (%)",
    )
    _bar(
        charts / "role_utility.png",
        "Measured role utility",
        [str(row.get("role")) for row in planner_measurements],
        [float(row.get("measured_utility", 0)) for row in planner_measurements],
        "normalised utility",
    )
    _bar(
        charts / "planner_prediction_vs_actual.png",
        "Planner prediction error",
        [str(row.get("role")) for row in planner_measurements],
        [
            float(row.get("measured_utility", 0)) - float(row.get("predicted_utility", 0))
            for row in planner_measurements
        ],
        "actual minus predicted utility",
    )
    _bar(
        charts / "planner_regret.png",
        "Planner regret",
        [str(row.get("objective")) for row in regret],
        [100 * float(row.get("planner_regret_fraction", 0)) for row in regret],
        "regret (%)",
    )
    useful_count = [
        sum(
            1
            for role in (
                "critical_path",
                "speculative_draft",
                "moe_expert",
                "background_inference",
                "integrity_audit",
                "shard_cache",
            )
            if row.get(role) == "useful"
        )
        for row in frontier
    ]
    _bar(
        charts / "contribution_frontier.png",
        "Contribution frontier",
        [str(row.get("device_profile")) for row in frontier],
        [float(value) for value in useful_count],
        "measured useful roles",
    )
    _bar(
        charts / "availability_break_even.png",
        "Productive fraction by lease",
        [f"{row.get('role')} {row.get('lease_duration_seconds')}s" for row in availability],
        [100 * float(row.get("productive_fraction", 0)) for row in availability],
        "productive lease (%)",
    )
    _bar(
        charts / "network_role_viability.png",
        "Network-emulated role utility",
        [f"{row.get('role')} {row.get('network_profile')}" for row in network],
        [float(row.get("projected_verified_tokens_per_second", 0)) for row in network],
        "verified output tokens/s",
    )


def render_heterogeneous_report(
    run_directory: Path,
    *,
    summary: dict[str, Any],
    findings: dict[str, Any],
    frontier: list[dict[str, Any]],
) -> Path:
    status_keys = [key for key in summary if key.endswith("_status")]
    status_rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td class='{str(summary[key]).lower()}'>"
        f"{html.escape(str(summary[key]))}</td></tr>"
        for key in status_keys
    )
    findings_rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td><code>"
        f"{html.escape(json.dumps(value, sort_keys=True, default=str))}</code></td></tr>"
        for key, value in findings.items()
    )
    frontier_columns = [
        "device_profile",
        "classification",
        "critical_path",
        "speculative_draft",
        "moe_expert",
        "background_inference",
        "integrity_audit",
        "shard_cache",
        "idle",
    ]
    frontier_header = "".join(f"<th>{html.escape(item)}</th>" for item in frontier_columns)
    frontier_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in frontier_columns
        )
        + "</tr>"
        for row in frontier
    )
    figures = "\n".join(
        f"<figure><img src='charts/{name}' alt='{name}'><figcaption>{name}</figcaption></figure>"
        for name in REQUIRED_CHARTS
    )
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 007 — heterogeneous node utility</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}th,td{{border:1px solid #bcccdc;padding:.45rem;text-align:left}}
th{{background:#f0f4f8}}.pass{{color:#137333;font-weight:700}}.fail{{color:#b31412;font-weight:700}}
.partial_pass,.not_useful,.blocked,.skipped{{color:#8a4b08;font-weight:700}}.warning{{padding:1rem;background:#fff3cd;border:1px solid #f0c36d}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:1rem}}figure{{margin:0}}img{{width:100%;border:1px solid #bcccdc}}
code{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style></head><body>
<h1>Experiment 007: Heterogeneous Node Utility</h1>
<p class="warning"><strong>Scope:</strong> heterogeneous-single-host-real-model. Measured CUDA, measured x86 CPU, and measured mixed-backend rows are separated from ARM64 compatibility, emulated-network, and projected-device rows. QEMU timings are excluded from contribution claims. Raspberry Pi performance remains unproven.</p>
<h2>Answer</h2><p>{html.escape(str(summary.get("conclusion", "No conclusion recorded.")))}</p>
<h2>Status</h2><table>{status_rows}</table>
<h2>Measured findings</h2><table>{findings_rows}</table>
<h2>Contribution frontier</h2><table><thead><tr>{frontier_header}</tr></thead><tbody>{frontier_rows}</tbody></table>
<h2>Charts</h2><div class="charts">{figures}</div>
<h2>Evidence and reproducibility</h2><p>The adjacent JSON, JSONL, and CSV files contain raw measurements, canonical artifact mappings, worker protocol and capability evidence, planner predictions, network-event replays, and availability economics. Backend build and launch commands are preserved under <code>artifacts/backend-environments</code> and <code>logs/</code>.</p>
</body></html>"""
    path = run_directory / "report.html"
    path.write_text(content, encoding="utf-8")
    return path
