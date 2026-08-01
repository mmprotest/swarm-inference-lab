"""Charts and self-contained report for Experiment 007 benchmark corrections."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_chart(
    path: Path,
    *,
    title: str,
    x: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(10.0, min(18.0, len(x) * 0.48))
    figure, axis = plt.subplots(figsize=(width, 5.8))
    if not x or not series:
        axis.text(0.5, 0.5, "No valid observations", ha="center", va="center")
        axis.set_axis_off()
    else:
        positions = list(range(len(x)))
        if len(series) == 1:
            axis.bar(
                positions,
                series[0][1],
                label=series[0][0],
                color="#2563eb",
                edgecolor="#172033",
                linewidth=0.5,
            )
        else:
            colours = ("#2563eb", "#d97706")
            styles = ("-", "--")
            for index, (label, values) in enumerate(series):
                axis.plot(
                    positions,
                    values,
                    marker="o" if index == 0 else "s",
                    linestyle=styles[index % len(styles)],
                    color=colours[index % len(colours)],
                    label=label,
                )
            axis.legend()
        axis.set_xticks(positions, x, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if any(value < 0 for _label, values in series for value in values):
            axis.axhline(0, color="#172033", linewidth=0.8)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_correction_charts(
    run_directory: Path,
    *,
    moe_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    planner_rows: list[dict[str, Any]],
    regret_rows: list[dict[str, Any]],
) -> None:
    chart_root = run_directory / "charts"
    bf16 = [
        row
        for row in moe_rows
        if row.get("arm") == "hybrid_gpu_cpu"
        and row.get("weight_format") == "bfloat16"
        and row.get("benchmark_mode") == "natural_routing"
    ]
    labels = [
        f"{str(row['placement_policy']).removesuffix('_experts_on_cpu').removesuffix('_on_cpu')}\n{row['cpu_expert_count']}"
        for row in bf16
    ]
    _save_chart(
        chart_root / "cpu_expert_calls_by_placement.png",
        title="Real CPU expert calls by natural-routing placement",
        x=labels,
        series=[("CPU expert calls", [float(row["cpu_expert_calls"]) for row in bf16])],
        ylabel="Routed calls",
    )
    _save_chart(
        chart_root / "cpu_dispatch_fraction.png",
        title="CPU dispatch fraction",
        x=labels,
        series=[("Dispatch fraction", [float(row["cpu_dispatch_fraction"]) for row in bf16])],
        ylabel="Fraction",
    )
    _save_chart(
        chart_root / "matched_expert_latency.png",
        title="Matched all-GPU and hybrid layer latency",
        x=labels,
        series=[
            (
                "Matched all-GPU",
                [float(row["matched_baseline_median_layer_ms"]) for row in bf16],
            ),
            ("Hybrid BF16", [float(row["median_total_layer_ms"]) for row in bf16]),
        ],
        ylabel="Median milliseconds",
    )
    _save_chart(
        chart_root / "matched_expert_throughput_retained.png",
        title="Matched BF16 throughput retained",
        x=labels,
        series=[
            (
                "Retained throughput",
                [float(row["throughput_retained_fraction"]) for row in bf16],
            )
        ],
        ylabel="Fraction",
    )
    count_groups = sorted({int(row["cpu_expert_count"]) for row in bf16})
    memory_values = (
        [
            max(
                float(row["gpu_memory_saved_bytes"])
                for row in bf16
                if int(row["cpu_expert_count"]) == count
            )
            for count in count_groups
        ]
        if bf16
        else []
    )
    _save_chart(
        chart_root / "gpu_memory_saved_by_expert_count.png",
        title="GPU expert-weight memory saved",
        x=[str(item) for item in count_groups],
        series=[("Bytes saved", memory_values)],
        ylabel="Bytes",
    )
    timing_components = sorted({str(row["timing_component"]) for row in timing_rows})
    timing_values = [
        sum(float(row["median_ms"]) for row in timing_rows if row["timing_component"] == name)
        / max(sum(row["timing_component"] == name for row in timing_rows), 1)
        for name in timing_components
    ]
    _save_chart(
        chart_root / "expert_timing_breakdown.png",
        title="Canonical MoE timing components",
        x=timing_components,
        series=[("Mean median time", timing_values)],
        ylabel="Milliseconds",
    )

    closed = [row for row in background_rows if row.get("traffic_mode") == "closed_loop"]
    background_labels = [f"g{row['gpu_concurrency']}/c{row['cpu_concurrency']}" for row in closed]
    _save_chart(
        chart_root / "fixed_window_gpu_throughput.png",
        title="Fixed-window GPU throughput",
        x=background_labels,
        series=[
            (
                "GPU only",
                [float(row["baseline_gpu_verified_tps_median"]) for row in closed],
            ),
            (
                "GPU with CPU",
                [float(row["gpu_verified_tps_median"]) for row in closed],
            ),
        ],
        ylabel="Verified tokens/s",
    )
    _save_chart(
        chart_root / "fixed_window_cpu_throughput.png",
        title="Fixed-window CPU background throughput",
        x=background_labels,
        series=[("CPU", [float(row["cpu_verified_tps_median"]) for row in closed])],
        ylabel="Verified tokens/s",
    )
    _save_chart(
        chart_root / "fixed_window_combined_throughput.png",
        title="Fixed-window combined verified throughput",
        x=background_labels,
        series=[
            (
                "Combined",
                [float(row["combined_verified_tps_median"]) for row in closed],
            )
        ],
        ylabel="Verified tokens/s",
    )
    _save_chart(
        chart_root / "gpu_p95_interference.png",
        title="GPU p95 latency interference",
        x=background_labels,
        series=[("p95 change", [float(row["gpu_p95_latency_change_fraction"]) for row in closed])],
        ylabel="Fraction",
    )
    _save_chart(
        chart_root / "gpu_throughput_interference.png",
        title="GPU throughput interference",
        x=background_labels,
        series=[
            (
                "Throughput change",
                [float(row["gpu_throughput_change_fraction"]) for row in closed],
            )
        ],
        ylabel="Fraction",
    )
    _save_chart(
        chart_root / "combined_gain.png",
        title="Combined fixed-window gain",
        x=background_labels,
        series=[("Combined gain", [float(row["combined_gain_fraction"]) for row in closed])],
        ylabel="Fraction",
    )
    planner_labels = [str(row["point_id"]) for row in planner_rows]
    _save_chart(
        chart_root / "planner_prediction_vs_actual.png",
        title="Held-out planner prediction versus actual",
        x=planner_labels,
        series=[
            ("Predicted", [float(row["predicted_utility"]) for row in planner_rows]),
            ("Measured", [float(row["measured_utility"]) for row in planner_rows]),
        ],
        ylabel="Normalised utility",
    )
    _save_chart(
        chart_root / "planner_held_out_regret.png",
        title="Held-out planner regret",
        x=[str(row.get("planner_selected_role", "planner")) for row in regret_rows],
        series=[
            ("Regret fraction", [float(row["planner_regret_fraction"]) for row in regret_rows])
        ],
        ylabel="Fraction",
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    headings = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '')))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{headings}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _display_status(value: Any) -> str:
    return str(value).replace("_", " ")


def _percent(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{100 * float(value):.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):,.{digits}f}"


def render_correction_report(
    run_directory: Path,
    *,
    summary: dict[str, Any],
    superseded: dict[str, Any],
    moe_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    held_out_rows: list[dict[str, Any]],
) -> Path:
    best_expert = max(
        (
            row
            for row in moe_rows
            if row.get("arm") == "hybrid_gpu_cpu"
            and row.get("weight_format") == "bfloat16"
            and bool(row.get("positive_performance_eligible"))
        ),
        key=lambda row: float(row.get("throughput_retained_fraction", -math.inf)),
        default=None,
    )
    best_background = max(
        (row for row in background_rows if row.get("traffic_mode") == "closed_loop"),
        key=lambda row: float(row.get("combined_gain_fraction", -math.inf)),
        default=None,
    )
    corpus_manifest_path = run_directory / "moe_routing_corpus_manifest.json"
    corpus_manifest = (
        json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        if corpus_manifest_path.is_file()
        else {}
    )
    planner_headline = summary.get("corrected_headlines", {}).get("planner") or {}
    expert_validity = (
        "PASS"
        if summary.get("cpu_expert_matched_baseline_status") == "PASS"
        and summary.get("cpu_expert_active_dispatch_status") == "PASS"
        else "FAIL"
    )
    background_validity = (
        "PASS"
        if summary.get("background_fixed_window_status") == "PASS"
        and summary.get("background_token_accounting_status") == "PASS"
        else "FAIL"
    )
    status_block = "\n".join(
        [
            f"CPU expert benchmark validity: {expert_validity}",
            f"CPU expert memory offload: {summary['cpu_expert_memory_offload_status']}",
            "CPU expert positive performance: "
            + _display_status(summary["cpu_expert_positive_performance_status"]),
            f"Background benchmark validity: {background_validity}",
            "Background positive contribution: "
            + _display_status(summary["background_positive_contribution_status"]),
            f"Planner held-out evaluation: {summary['planner_held_out_evaluation_status']}",
            "Corrected Experiment 007: "
            + _display_status(summary["corrected_experiment_007_status"]),
        ]
    )
    comparison_rows = [
        {
            "measurement": "CPU expert throughput retained",
            "original reported result": superseded["superseded_unmatched_cpu_expert_result"][
                "throughput_retained_fraction"
            ],
            "corrected result": best_expert.get("throughput_retained_fraction")
            if best_expert
            else "no eligible placement",
            "reason for difference": "canonical matched executor replaces unmatched HF/custom paths",
        },
        {
            "measurement": "Background combined gain",
            "original reported result": superseded["superseded_fixed_job_background_result"][
                "combined_gain_fraction"
            ],
            "corrected result": best_background.get("combined_gain_fraction")
            if best_background
            else "no valid point",
            "reason for difference": "shared fixed window replaces fixed-job paired makespan",
        },
    ]
    expert_table_rows = sorted(
        [
            row
            for row in moe_rows
            if row.get("arm") == "hybrid_gpu_cpu" and row.get("weight_format") == "bfloat16"
        ],
        key=lambda row: (
            str(row["placement_policy"]),
            int(row["cpu_expert_count"]),
        ),
    )
    expert_summary = (
        (
            "The best eligible matched BF16 placement generated "
            f"{int(best_expert['cpu_expert_calls']):,} CPU expert calls "
            f"({_percent(best_expert['cpu_dispatch_fraction'])} of routed calls), saved "
            f"{int(best_expert['gpu_memory_saved_bytes']):,} GPU bytes, and retained "
            f"{_percent(best_expert['throughput_retained_fraction'])} of its paired all-GPU "
            f"throughput. Its latency multiplier was "
            f"{_number(best_expert['latency_multiplier'], 3)}x."
        )
        if best_expert
        else "No natural-routing BF16 placement met the active-dispatch eligibility gate."
    )
    background_summary = (
        (
            "The highest-gain closed-loop point added "
            f"{_number(best_background['cpu_verified_tps_median'])} verified CPU tokens/s. "
            "Combined fixed-window throughput changed by "
            f"{_percent(best_background['combined_gain_fraction'])}; GPU throughput changed by "
            f"{_percent(best_background['gpu_throughput_change_fraction'])} and GPU p95 latency "
            f"changed by {_percent(best_background['gpu_p95_latency_change_fraction'])}."
        )
        if best_background
        else "No valid closed-loop fixed-window comparison was produced."
    )
    chart_explanations = {
        "cpu_expert_calls_by_placement": (
            "Inactive and low-dispatch placements remain visible; a zero-call bar cannot support "
            "a performance claim."
        ),
        "cpu_dispatch_fraction": (
            "Dispatch share is the eligibility denominator. Primary comparisons require at least "
            "1% of routed expert calls to execute on CPU."
        ),
        "matched_expert_latency": (
            "Each hybrid point is paired with the same canonical all-GPU executor, frozen corpus, "
            "dtype, kernels, and timing boundary."
        ),
        "matched_expert_throughput_retained": (
            "Retained throughput compares matched all-GPU and hybrid execution; 70% is the "
            "positive-performance gate."
        ),
        "gpu_memory_saved_by_expert_count": (
            "Observed CUDA-allocation savings must equal the removed BF16 expert-weight bytes."
        ),
        "expert_timing_breakdown": (
            "The total timer ends only after CPU returns are combined on GPU and synchronized."
        ),
        "fixed_window_gpu_throughput": (
            "GPU-only and paired arms use the same fixture corpus and measurement duration."
        ),
        "fixed_window_cpu_throughput": (
            "CPU tokens count only at streaming completion timestamps inside the shared window."
        ),
        "fixed_window_combined_throughput": (
            "Combined rate divides GPU plus CPU token completions by the fixed window, never by "
            "request makespan."
        ),
        "gpu_p95_interference": (
            "The background role must keep median-repeat GPU p95 degradation at or below 5%."
        ),
        "gpu_throughput_interference": (
            "The background role must keep median-repeat GPU throughput degradation above -5%."
        ),
        "combined_gain": (
            "A useful point requires 10% combined gain while both GPU guardrails pass."
        ),
        "planner_prediction_vs_actual": (
            "Predictions are fit only on calibration points; actuals are held out until selection."
        ),
        "planner_held_out_regret": (
            "Regret compares the selected held-out point with the best subsequently observed "
            "held-out point, including idle."
        ),
    }
    chart_blocks = "".join(
        (
            f"<section><h3>{html.escape(path.stem.replace('_', ' ').title())}</h3>"
            f"<p>{html.escape(chart_explanations.get(path.stem, 'Corrected evidence.'))}</p>"
            f'<img src="charts/{html.escape(path.name)}" '
            f'alt="{html.escape(path.stem)}"></section>'
        )
        for path in sorted((run_directory / "charts").glob("*.png"))
    )
    css = """
    :root{color-scheme:light dark}body{font-family:Segoe UI,Arial,sans-serif;max-width:1300px;margin:24px auto;padding:0 18px;color:#172033;background:#fff;line-height:1.5}
    pre.status{font-size:16px;background:#eef5ff;border-left:5px solid #2563eb;padding:18px;white-space:pre-wrap}
    .table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px}th,td{border:1px solid #cbd5e1;padding:7px;text-align:left}th{background:#e2e8f0}
    img{max-width:100%;border:1px solid #cbd5e1;margin:8px 0 22px}code{background:#f1f5f9;padding:2px 4px}
    .warning{background:#fff7ed;border-left:5px solid #ea580c;padding:14px}section{margin:24px 0}a{color:#1d4ed8}
    @media(prefers-color-scheme:dark){body{color:#e5e7eb;background:#111827}pre.status{background:#172554}th{background:#1f2937}th,td,img{border-color:#475569}.warning{background:#431407}code{background:#1f2937}a{color:#93c5fd}}
    """
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light dark"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment 007 benchmark corrections</title><style>{css}</style></head><body>
<pre class="status">{html.escape(status_block)}</pre>
<h1>Experiment 007 benchmark corrections</h1>
<h2>Technical summary</h2>
<ul><li>{html.escape(expert_summary)}</li><li>{html.escape(background_summary)}</li><li>The held-out planner selected <strong>{html.escape(str(planner_headline.get("planner_selected_role", "unavailable")))}</strong> with {_percent(planner_headline.get("planner_regret_fraction"))} regret.</li></ul>
<p class="warning"><strong>Superseded evidence:</strong> The original MoE speed and background combined-throughput headlines are historical only. They are not reused by the corrected planner.</p>
<h2>The corrected denominators change the comparison</h2>
{_table(comparison_rows, ["measurement", "original reported result", "corrected result", "reason for difference"])}
<h2>Scope and metric definitions</h2>
<p>The expert grain is one frozen routed token position at Qwen3-30B-A3B layer {html.escape(str(corpus_manifest.get("layer_id", "unknown")))}. The corpus contains {int(corpus_manifest.get("token_count", 0)):,} positions and {int(corpus_manifest.get("routed_expert_call_count", 0)):,} routed calls. <code>throughput_retained_fraction</code> compares the same canonical executor with only expert placement changed. The background grain is a streaming token completion whose monotonic timestamp lies inside one shared serving window; <code>combined_verified_tps = (GPU tokens + CPU tokens) / window seconds</code>.</p>
<h2>Matched execution makes CPU expert cost observable</h2>
<p>{html.escape(expert_summary)}</p>
{_table(expert_table_rows, ["placement_policy", "cpu_expert_count", "cpu_expert_calls", "cpu_dispatch_fraction", "gpu_memory_saved_bytes", "matched_baseline_median_layer_ms", "median_total_layer_ms", "throughput_retained_fraction", "latency_multiplier", "positive_performance_pass"])}
<h2>Fixed-window accounting isolates added CPU capacity</h2>
<p>{html.escape(background_summary)}</p>
{_table(background_rows, ["traffic_mode", "gpu_concurrency", "cpu_concurrency", "measurement_window_seconds", "baseline_gpu_verified_tps_median", "gpu_verified_tps_median", "cpu_verified_tps_median", "combined_verified_tps_median", "combined_gain_fraction", "gpu_p95_latency_change_fraction", "gpu_throughput_change_fraction", "positive_contribution_pass"])}
<h2>The planner is evaluated out of sample</h2>
<p>Counts 2 and 8 are held out for MoE; GPU concurrency 4 is held out for background capacity. The planner sees features and calibration fits, but not held-out measured utility, before selecting.</p>
{_table(held_out_rows, ["point_id", "role", "predicted_utility", "measured_utility", "prediction_error"])}
<h2>Visual evidence</h2>{chart_blocks}
<h2>Method and reproducibility</h2>
<p>The frozen routing corpus, execution plans, per-component timings, token events, repeat-level serving rows, correctness audits, calibration split, and held-out results are preserved as inspectable files. The corpus identity is a canonical hash over sorted tensor names, dtypes, shapes, bytes, and model identity; the raw safetensors file hash is retained separately because metadata-key order is not stable. MoE timing uses ten warm-ups and at least five repeats. If a 30-repeat epoch exceeds 10% coefficient of variation, the executor re-warms and uses the first subsequent predeclared epoch satisfying the same 10% gate; every failed epoch CV and the total repeat count remain visible in the matched-results table. Start with <a href="moe_routing_corpus_manifest.json">the routing manifest</a>, <a href="moe_matched_results.csv">matched MoE results</a>, <a href="background_token_events.jsonl">token events</a>, and <a href="planner_held_out_results.csv">held-out planner results</a>.</p>
<h2>Limitations and robustness</h2>
<p>The validated 30B shard contains complete real layers 22-24, so layer-24 inputs traverse two real predecessor layers, but this is not a full-model prompt-activation capture. The natural router is never overridden, and that provenance is recorded in the corpus manifest. The run measures one RTX 5090 and one x86 CPU on one host. CPU package power and per-container host CPU attribution may be unavailable; those fields are labelled rather than estimated. Smoke runs validate machinery only and cannot support performance conclusions.</p>
<h2>Recommended next step</h2><p>Use only configurations whose corrected status gates pass. If neither role is useful, retain idle as the safe placement; do not tune thresholds or reuse the superseded measurements.</p>
<h2>Further question</h2><p>A future run with captured layer-24 activations from a complete Qwen3-30B-A3B generation path would test whether the placement ranking is stable under truly end-to-end prompt distributions.</p>
<h2>Machine-readable status</h2><pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
</body></html>"""
    report_path = run_directory / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
