"""Evidence charts and answer-first HTML reporting for Experiment 006."""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_CHARTS = (
    "per_rank_weight_bytes.png",
    "largest_tensor_shard.png",
    "kv_cache_bytes_by_rank.png",
    "dense_token_correctness.png",
    "boundary_error_by_tp_degree.png",
    "collective_bytes_per_layer.png",
    "collective_latency_break_even.png",
    "decode_tps_by_tp_degree.png",
    "prefill_tps_by_tp_degree.png",
    "hybrid_pp_tp_projection.png",
    "straggler_sensitivity.png",
    "communication_compression_tradeoff.png",
    "expert_bytes_per_rank.png",
    "expert_fanout.png",
    "expert_imbalance.png",
    "expert_cache_hit_rate.png",
    "k3_memory_per_rank.png",
    "k3_projected_tps.png",
)


def _finish(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _empty(path: Path, title: str, message: str = "No observations") -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    _finish(fig, path)


def _bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
) -> None:
    if not labels:
        _empty(path, title)
        return
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    _finish(fig, path)


def _line(
    path: Path,
    title: str,
    groups: dict[str, list[tuple[float, float]]],
    xlabel: str,
    ylabel: str,
    *,
    log_y: bool = False,
) -> None:
    if not groups:
        _empty(path, title)
        return
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for label, points in sorted(groups.items()):
        ordered = sorted(points)
        axis.plot(
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            marker="o",
            linewidth=1.4,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    if len(groups) <= 16:
        axis.legend(fontsize=8)
    _finish(fig, path)


def _groups(
    rows: list[dict[str, Any]],
    *,
    label: Callable[[dict[str, Any]], str],
    x: Callable[[dict[str, Any]], float],
    y: Callable[[dict[str, Any]], float],
) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        result.setdefault(label(row), []).append((x(row), y(row)))
    return result


def generate_microsharding_charts(
    chart_dir: Path,
    *,
    memory: list[dict[str, Any]],
    correctness: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    kv: list[dict[str, Any]],
    collective_metrics: list[dict[str, Any]],
    projections: list[dict[str, Any]],
    break_even: list[dict[str, Any]],
    hybrid: list[dict[str, Any]],
    heterogeneous: list[dict[str, Any]],
    compression: list[dict[str, Any]],
    moe: list[dict[str, Any]],
    expert_projection: list[dict[str, Any]],
    expert_cache: list[dict[str, Any]],
    k3_plans: list[dict[str, Any]],
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    memory_rank_rows = [row for row in memory if row.get("row_type") == "configuration"]
    _bar(
        chart_dir / "per_rank_weight_bytes.png",
        "Maximum logical weight bytes per rank",
        [
            f"PP{row['pipeline_stage_count']} TP{row['tensor_parallel_degree']}"
            for row in memory_rank_rows
        ],
        [float(row["maximum_rank_weight_bytes"]) for row in memory_rank_rows],
        "bytes",
    )
    _line(
        chart_dir / "largest_tensor_shard.png",
        "Dominant matrix sharding",
        _groups(
            memory_rank_rows,
            label=lambda row: f"PP{row['pipeline_stage_count']}",
            x=lambda row: float(row["tensor_parallel_degree"]),
            y=lambda row: float(row["largest_matrix_shard_bytes"]),
        ),
        "tensor-parallel degree",
        "largest local matrix bytes",
    )
    _bar(
        chart_dir / "kv_cache_bytes_by_rank.png",
        "KV cache bytes per logical rank",
        [f"TP{row['tensor_parallel_degree']}:R{row['tp_rank']}" for row in kv],
        [float(row["bytes"]) for row in kv],
        "bytes",
    )
    correctness_groups: dict[str, list[dict[str, Any]]] = {}
    for row in correctness:
        key = f"PP{row['pipeline_stage_count']} TP{row['tensor_parallel_degree']}"
        correctness_groups.setdefault(key, []).append(row)
    _bar(
        chart_dir / "dense_token_correctness.png",
        "Exact greedy-token identity",
        list(correctness_groups),
        [
            sum(item["exact_token_identity"] == "PASS" for item in rows) / max(len(rows), 1)
            for rows in correctness_groups.values()
        ],
        "passing prompt fraction",
    )
    _line(
        chart_dir / "boundary_error_by_tp_degree.png",
        "Layer-boundary maximum absolute error",
        _groups(
            boundaries,
            label=lambda row: f"L{row['layer_id']} {row['boundary']}",
            x=lambda row: float(row["tensor_parallel_degree"]),
            y=lambda row: float(row["maximum_absolute_error"]),
        ),
        "tensor-parallel degree",
        "maximum absolute error",
        log_y=True,
    )
    collective_grouped: dict[int, float] = {}
    for row in collective_metrics:
        raw_layer = row.get("layer_id")
        layer = int(raw_layer) if raw_layer is not None else -1
        if layer >= 0:
            collective_grouped[layer] = collective_grouped.get(layer, 0.0) + float(
                row.get("aggregate_bytes", row.get("logical_aggregate_bytes", 0))
            )
    _bar(
        chart_dir / "collective_bytes_per_layer.png",
        "Logical collective bytes by layer",
        [str(item) for item in sorted(collective_grouped)],
        [collective_grouped[item] for item in sorted(collective_grouped)],
        "logical aggregate bytes",
    )
    _line(
        chart_dir / "collective_latency_break_even.png",
        "Tensor-parallel break-even latency",
        _groups(
            break_even,
            label=lambda row: f"{row['workload']} TP{row['tensor_parallel_degree']}",
            x=lambda row: float(row.get("batch_size", row.get("sequence_length", 1))),
            y=lambda row: float(row["maximum_one_way_latency_ms"]),
        ),
        "batch size or sequence length",
        "maximum one-way latency (ms)",
    )
    decode = [
        row
        for row in projections
        if row["workload"] == "decode"
        and row["network_profile"] in {"nvlink_class", "home_lan_10gbe"}
    ]
    _line(
        chart_dir / "decode_tps_by_tp_degree.png",
        "Projected decode throughput",
        _groups(
            decode,
            label=lambda row: f"{row['network_profile']} B{row['batch_size']}",
            x=lambda row: float(row["tensor_parallel_degree"]),
            y=lambda row: float(row["projected_tokens_per_second"]),
        ),
        "tensor-parallel degree",
        "projected tokens/s",
    )
    prefill = [
        row
        for row in projections
        if row["workload"] == "prefill" and row["network_profile"] == "nvlink_class"
    ]
    _line(
        chart_dir / "prefill_tps_by_tp_degree.png",
        "Projected prefill throughput",
        _groups(
            prefill,
            label=lambda row: f"S{row['sequence_length']} B{row['batch_size']}",
            x=lambda row: float(row["tensor_parallel_degree"]),
            y=lambda row: float(row["projected_tokens_per_second"]),
        ),
        "tensor-parallel degree",
        "projected prompt tokens/s",
    )
    _line(
        chart_dir / "hybrid_pp_tp_projection.png",
        "Hybrid pipeline + tensor projection",
        _groups(
            [row for row in hybrid if row["concurrent_requests"] in {1, 64}],
            label=lambda row: f"PP{row['pipeline_stage_count']} C{row['concurrent_requests']}",
            x=lambda row: float(row["tensor_parallel_degree"]),
            y=lambda row: float(row["aggregate_tokens_per_second"]),
        ),
        "tensor-parallel degree",
        "projected aggregate tokens/s",
    )
    _bar(
        chart_dir / "straggler_sensitivity.png",
        "Heterogeneous-rank group slowdown",
        [str(row["rank_profile"]) for row in heterogeneous],
        [float(row["group_slowdown"]) for row in heterogeneous],
        "slowdown factor",
    )
    _bar(
        chart_dir / "communication_compression_tradeoff.png",
        "Communication compression trade-off",
        [str(row["format"]) for row in compression],
        [float(row["message_bytes"]) for row in compression],
        "message bytes",
    )
    moe_memory = [row for row in moe if "maximum_expert_bytes_per_rank" in row]
    _bar(
        chart_dir / "expert_bytes_per_rank.png",
        "Deterministic MoE expert bytes per rank",
        [
            f"E{row['expert_count']} EP{row['expert_parallel_degree']} ETP{row['expert_tensor_parallel_degree']}"
            for row in moe_memory
        ],
        [float(row["maximum_expert_bytes_per_rank"]) for row in moe_memory],
        "bytes",
    )
    _bar(
        chart_dir / "expert_fanout.png",
        "Expert dispatch fan-out",
        [f"EP{row['expert_parallel_degree']} B{row['batch_size']}" for row in expert_projection],
        [float(row["fanout"]) for row in expert_projection],
        "active ranks",
    )
    _bar(
        chart_dir / "expert_imbalance.png",
        "Expert load imbalance",
        [f"EP{row['expert_parallel_degree']} {row['placement']}" for row in expert_projection],
        [float(row["expert_imbalance"]) for row in expert_projection],
        "max / mean assignments",
    )
    _bar(
        chart_dir / "expert_cache_hit_rate.png",
        "On-demand expert-cache hit rate",
        [str(row["policy"]) for row in expert_cache],
        [float(row["expert_cache_hit_rate"]) for row in expert_cache],
        "hit rate",
    )
    _bar(
        chart_dir / "k3_memory_per_rank.png",
        "Kimi K3 projected memory per rank",
        [str(row["plan"]) for row in k3_plans],
        [float(row["maximum_weight_bytes_per_rank"]) for row in k3_plans],
        "bytes",
    )
    _bar(
        chart_dir / "k3_projected_tps.png",
        "Kimi K3 projected single-stream throughput",
        [str(row["plan"]) for row in k3_plans],
        [float(row["projected_single_stream_tokens_per_second"]) for row in k3_plans],
        "projected tokens/s",
    )
    missing = [name for name in REQUIRED_CHARTS if not (chart_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"chart generation omitted required files: {missing}")


def render_microsharding_report(
    path: Path,
    *,
    summary: dict[str, Any],
    conclusion: str,
    run_metadata: dict[str, Any],
    artifact_names: list[str],
) -> None:
    status_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
        if key.endswith("_status")
    )
    metadata_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in run_metadata.items()
    )
    chart_html = "".join(
        f'<figure><img src="charts/{name}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in REQUIRED_CHARTS
    )
    artifact_html = "".join(f"<li><code>{html.escape(name)}</code></li>" for name in artifact_names)
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Experiment 006 — Intra-Layer Microsharding</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}"
        "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccd1d1;padding:.45rem;text-align:left}"
        "th{background:#f4f6f7}figure{margin:1.5rem 0}img{max-width:100%;height:auto;border:1px solid #ddd}"
        "code{word-break:break-word}.pass{color:#196f3d}.warning{color:#9c640c}</style></head><body>"
        "<h1>Experiment 006: Intra-Layer Microsharding</h1>"
        f"<p><strong>Overall status:</strong> {html.escape(str(summary.get('overall_status')))}</p>"
        f"<p>{html.escape(conclusion)}</p>"
        "<h2>Claim boundary</h2><p>This run measures logical ranks in one process, one CUDA context, and one RTX 5090. Independent-rank, low-latency-cell, WAN, expert-placement, and Kimi K3 results are projections; they are not physical distributed speedups or memory pooling.</p>"
        f"<h2>Acceptance statuses</h2><table>{status_rows}</table>"
        f"<h2>Resolved run metadata</h2><table>{metadata_rows}</table>"
        f"<h2>Charts</h2>{chart_html}"
        f"<h2>Evidence files</h2><ul>{artifact_html}</ul>"
        "</body></html>\n",
        encoding="utf-8",
    )
