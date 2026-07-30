"""CSV, chart, and HTML evidence rendering for Experiment 003."""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

TABLE_FILES = (
    "worker_count_results.csv",
    "worker_lifecycle.csv",
    "pipeline_readiness.csv",
    "cold_inference.csv",
    "warm_inference.csv",
    "resource_usage.csv",
    "worker_memory.csv",
    "gpu_process_memory.csv",
    "activation_traffic.csv",
    "stream_metrics.csv",
    "correctness.csv",
    "acquisition_results.csv",
    "node_economics.csv",
)

CHART_FILES = (
    "maximum_worker_search.png",
    "pipeline_ready_time_by_workers.png",
    "worker_ready_distribution.png",
    "cold_ttft_by_workers.png",
    "warm_tps_by_workers.png",
    "aggregate_tps_by_workers.png",
    "gpu_memory_by_workers.png",
    "host_memory_by_workers.png",
    "process_overhead_by_workers.png",
    "activation_bytes_by_workers.png",
    "stage_load_time_distribution.png",
    "lifecycle_breakdown.png",
    "shard_acquisition_time.png",
    "productive_fraction_by_lease.png",
    "productive_tokens_by_lease.png",
)

DEFAULT_COLUMNS: dict[str, list[str]] = {
    name: ["experiment_id", "worker_count", "phase", "repeat"] for name in TABLE_FILES
}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = list(DEFAULT_COLUMNS.get(path.name, ["status"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def write_all_tables(run_dir: Path, tables: Mapping[str, list[dict[str, Any]]]) -> None:
    for name in TABLE_FILES:
        write_csv(run_dir / name, tables.get(name, []))


def _plot_lines(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    x: str,
    y: str,
    title: str,
    ylabel: str,
    group: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    usable = [row for row in rows if row.get(x) is not None and row.get(y) is not None]
    if group is None:
        points = sorted(
            ((float(row[x]), float(row[y])) for row in usable),
            key=lambda item: item[0],
        )
        if points:
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
            )
    else:
        groups = sorted({str(row.get(group, "")) for row in usable})
        for label in groups:
            points = sorted(
                (
                    (float(row[x]), float(row[y]))
                    for row in usable
                    if str(row.get(group, "")) == label
                ),
                key=lambda item: item[0],
            )
            if points:
                axis.plot(
                    [item[0] for item in points],
                    [item[1] for item in points],
                    marker="o",
                    label=label,
                )
        if groups:
            axis.legend()
    if not usable:
        axis.text(0.5, 0.5, "No measured evidence", ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " "))
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_charts(run_dir: Path, tables: Mapping[str, list[dict[str, Any]]]) -> None:
    charts = run_dir / "charts"
    count_rows = tables.get("worker_count_results.csv", [])
    lifecycle = tables.get("worker_lifecycle.csv", [])
    cold = tables.get("cold_inference.csv", [])
    warm = tables.get("warm_inference.csv", [])
    resources = tables.get("resource_usage.csv", [])
    activation = tables.get("activation_traffic.csv", [])
    acquisition = tables.get("acquisition_results.csv", [])
    economics = tables.get("node_economics.csv", [])
    specifications = {
        "maximum_worker_search.png": (
            count_rows,
            "worker_count",
            "runnable_numeric",
            "Maximum runnable worker search",
            "Runnable (1/0)",
            None,
        ),
        "pipeline_ready_time_by_workers.png": (
            tables.get("pipeline_readiness.csv", []),
            "worker_count",
            "pipeline_ready_seconds",
            "Pipeline readiness by worker count",
            "Seconds",
            "phase",
        ),
        "worker_ready_distribution.png": (
            lifecycle,
            "worker_count",
            "worker_ready_seconds",
            "Worker ready-time distribution",
            "Seconds",
            "phase",
        ),
        "cold_ttft_by_workers.png": (
            cold,
            "worker_count",
            "ttft_seconds",
            "Cold TTFT by worker count",
            "Seconds",
            "variant",
        ),
        "warm_tps_by_workers.png": (
            [row for row in warm if int(row.get("concurrency", 0)) == 1],
            "worker_count",
            "output_tokens_per_second",
            "Warm verified output rate",
            "Tokens/s",
            None,
        ),
        "aggregate_tps_by_workers.png": (
            [row for row in warm if int(row.get("concurrency", 0)) == 4],
            "worker_count",
            "aggregate_verified_tokens_per_second",
            "Concurrency-4 aggregate verified throughput",
            "Tokens/s",
            None,
        ),
        "gpu_memory_by_workers.png": (
            resources,
            "worker_count",
            "gpu_memory_used_bytes",
            "Aggregate physical GPU memory",
            "Bytes",
            "phase",
        ),
        "host_memory_by_workers.png": (
            resources,
            "worker_count",
            "aggregate_experiment_rss_bytes",
            "Aggregate host RSS",
            "Bytes",
            "phase",
        ),
        "process_overhead_by_workers.png": (
            resources,
            "worker_count",
            "python_process_count",
            "Python process overhead",
            "Processes",
            "phase",
        ),
        "activation_bytes_by_workers.png": (
            activation,
            "worker_count",
            "worker_to_worker_activation_bytes",
            "Direct activation traffic",
            "Bytes",
            "phase",
        ),
        "stage_load_time_distribution.png": (
            lifecycle,
            "worker_count",
            "weight_load_seconds",
            "Stage weight-load time",
            "Seconds",
            "phase",
        ),
        "lifecycle_breakdown.png": (
            lifecycle,
            "worker_count",
            "cached_time_to_contribution_seconds",
            "Cached-cold lifecycle time",
            "Seconds",
            "phase",
        ),
        "shard_acquisition_time.png": (
            acquisition,
            "shard_bytes",
            "total_acquisition_duration_seconds",
            "Emulated shard-acquisition time",
            "Seconds",
            "profile",
        ),
        "productive_fraction_by_lease.png": (
            economics,
            "lease_seconds",
            "productive_fraction",
            "Productive fraction by lease",
            "Fraction",
            "node_state",
        ),
        "productive_tokens_by_lease.png": (
            economics,
            "lease_seconds",
            "productive_tokens",
            "Productive verified tokens by lease",
            "Tokens",
            "node_state",
        ),
    }
    for name, (rows, x, y, title, ylabel, group) in specifications.items():
        _plot_lines(
            charts / name,
            rows=list(rows),
            x=x,
            y=y,
            title=title,
            ylabel=ylabel,
            group=group,
        )


def render_report(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    count_rows: list[dict[str, Any]],
    acquisition_rows: list[dict[str, Any]],
    economics_rows: list[dict[str, Any]],
    rejoin: Mapping[str, Any],
) -> Path:
    def esc(value: Any) -> str:
        return html.escape(str(value))

    def table(rows: list[dict[str, Any]], fields: list[str]) -> str:
        headings = "".join(f"<th>{esc(field)}</th>" for field in fields)
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(row.get(field, ''))}</td>" for field in fields) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>"

    status_rows = [
        {"status": key, "value": value}
        for key, value in summary.items()
        if key.endswith("_status") or key == "overall_status"
    ]
    count_fields = [
        "worker_count",
        "attempted",
        "runnable",
        "stable",
        "median_pipeline_ready_seconds",
        "median_worker_ready_seconds",
        "maximum_worker_ready_seconds",
        "median_cuda_initialisation_seconds",
        "median_shard_read_seconds",
        "median_weight_load_seconds",
        "median_local_warmup_seconds",
        "median_cold_ttft_no_stage_warmup_seconds",
        "median_cold_ttft_with_stage_warmup_seconds",
        "median_warm_ttft_seconds",
        "median_warm_end_to_end_seconds",
        "median_warm_output_tokens_per_second",
        "median_concurrency_4_verified_tps",
        "peak_gpu_memory_bytes",
        "peak_host_memory_bytes",
        "failure_reason",
    ]
    acquisition_fields = [
        "worker_count",
        "stage_role",
        "stage_id",
        "profile",
        "shard_bytes",
        "total_acquisition_duration_seconds",
        "time_to_contribution_seconds",
        "verification_duration_seconds",
        "verification_seconds_worker",
        "shard_read_seconds",
        "weight_load_seconds",
        "warmup_seconds",
        "measurement_class",
    ]
    economics_fields = [
        "worker_count",
        "node_state",
        "profile",
        "lease_seconds",
        "startup_seconds",
        "productive_fraction",
        "productive_tokens",
        "minimum_lease_50_seconds",
        "minimum_lease_75_seconds",
        "minimum_lease_90_seconds",
        "minimum_lease_95_seconds",
    ]
    profiling_path = run_dir / "profiling.json"
    profiling = (
        json.loads(profiling_path.read_text(encoding="utf-8"))
        if profiling_path.is_file()
        else {"profiles": []}
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Experiment 003 — Worker Fan-Out, Cold Start, and Ad Hoc Node Economics</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#1d2733}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:1rem 0}}
th,td{{border:1px solid #ccd4dc;padding:.4rem;text-align:left;vertical-align:top}}
th{{background:#edf3f7}} code,pre{{background:#f4f6f8;padding:.2rem}} img{{width:48%;min-width:480px}}
.PASS{{color:#08783e;font-weight:700}} .FAIL{{color:#a32020;font-weight:700}}
</style></head><body>
<h1>Experiment 003: Worker Fan-Out, Cold Start, and Ad Hoc Node Economics</h1>
<p><strong>Execution mode:</strong> single-host-loopback-real-model-fanout.
Every process shared one NVIDIA RTX 5090. Increasing worker count did not add physical
GPU compute.</p>
<h2>Primary results</h2>
<ul>
<li>Maximum semantic worker count: {esc(summary.get("maximum_semantic_worker_count"))}</li>
<li>Maximum runnable worker count: {esc(summary.get("maximum_runnable_worker_count"))}</li>
<li>Maximum stable worker count: {esc(summary.get("maximum_stable_worker_count"))}</li>
<li>Single-request latency optimum: {esc(summary.get("single_request_latency_optimal_worker_count"))}</li>
<li>Concurrency-4 throughput optimum: {esc(summary.get("concurrency_4_throughput_optimal_worker_count"))}</li>
</ul>
<h2>Experiment statuses</h2>{table(status_rows, ["status", "value"])}
<h2>Worker-count evidence</h2>{table(count_rows, count_fields)}
<h2>Unprovisioned shard acquisition</h2>
<p>All throttled profiles are labelled <strong>emulated-shard-acquisition</strong>;
they are application-level local transfers, not physical network tests.</p>
{table(acquisition_rows, acquisition_fields)}
<h2>Contribution economics</h2>{table(economics_rows, economics_fields)}
<h2>Rejoin and cache replay</h2><pre>{esc(json.dumps(dict(rejoin), indent=2, sort_keys=True))}</pre>
<h2>Representative-count profiling</h2>
<pre>{esc(json.dumps(profiling, indent=2, sort_keys=True))}</pre>
<h2>Charts</h2>
{"".join(f'<img src="charts/{name}" alt="{name}">' for name in CHART_FILES)}
<h2>File-cache control</h2>
<p>{esc(summary.get("file_cache_control"))}</p>
<h2>Limitations</h2>
<p>This experiment does not prove multi-machine, LAN, WAN, Raspberry Pi, Kimi K3,
worldwide swarm, or additional-compute scaling. It measures stage granularity,
CUDA-process overhead, process and memory limits, startup economics, and sequential
pipeline overhead on one Windows host and one RTX 5090. Ad hoc-node viability is
interpreted only through the measured lease-duration economics above.</p>
<h2>Conclusion</h2>
<p><strong>Maximum runnable worker count: {esc(summary.get("maximum_runnable_worker_count"))}<br>
Maximum stable worker count: {esc(summary.get("maximum_stable_worker_count"))}<br>
Single-request performance optimum: {esc(summary.get("single_request_latency_optimal_worker_count"))}<br>
Concurrency-4 throughput optimum: {esc(summary.get("concurrency_4_throughput_optimal_worker_count"))}</strong></p>
<p>A cached-cold worker required
{esc(summary.get("cached_cold_median_time_to_contribution_seconds"))} seconds to become useful.
A hot-standby worker required
{esc(summary.get("hot_standby_median_time_to_contribution_seconds"))} seconds to become useful.
Unprovisioned time to contribution by emulated transfer profile:
<code>{esc(summary.get("unprovisioned_time_to_contribution_by_profile"))}</code>.</p>
<p>The maximum worker fan-out is limited by
<strong>{esc(summary.get("measured_limiting_factor"))}</strong>.</p>
</body></html>"""
    report = run_dir / "report.html"
    report.write_text(document, encoding="utf-8")
    return report
