"""Raw-data-driven plots and answer-first Experiment 010 report."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

plt.switch_backend("Agg")


PLOT_FILES = (
    "01_strategy_throughput.png",
    "02_ttft_by_strategy.png",
    "03_p95_by_strategy.png",
    "04_data_plane_comparison.png",
    "05_coalescing_traffic.png",
    "06_bytes_per_output_token.png",
    "07_whole_expert_break_even.png",
    "08_microshard_break_even.png",
    "09_best_shard_count.png",
    "10_cpu_gpu_utilisation.png",
    "11_worker_queue.png",
    "12_expert_residency.png",
    "13_expert_ownership.png",
    "14_cache_hit_rate.png",
    "15_prefill_union_deduplication.png",
    "16_throughput_vs_concurrency.png",
    "17_failure_recovery_latency.png",
    "18_detection_vs_verification_overhead.png",
    "19_simulator_throughput.png",
    "20_simulator_p95.png",
    "21_planner_regret.png",
    "22_worker_marginal_utility.png",
    "23_kimi_operator_breakdown.png",
    "24_kimi_projected_throughput.png",
    "25_capacity_bytes_per_worker.png",
    "26_peak_rss_per_worker.png",
    "27_codec_break_even.png",
    "28_verdict_dashboard.png",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    try:
        if value in (None, "", "null", "None"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _save(path: Path, title: str, draw: Callable[[Axes], bool]) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    measured = draw(axis)
    axis.set_title(title)
    if not measured:
        axis.text(
            0.5,
            0.5,
            "No applicable raw observations in this run",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar_draw(rows: list[dict[str, str]], x: str, y: str, ylabel: str) -> Callable[[Axes], bool]:
    def draw(axis: Axes) -> bool:
        values = [(row.get(x, ""), _number(row.get(y))) for row in rows]
        values = [(label, value) for label, value in values if label and value is not None]
        if not values:
            return False
        axis.bar([item[0] for item in values], [item[1] for item in values])
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=30)
        return True

    return draw


def _scatter_draw(
    rows: list[dict[str, str]], x: str, y: str, xlabel: str, ylabel: str
) -> Callable[[Axes], bool]:
    def draw(axis: Axes) -> bool:
        values = [(_number(row.get(x)), _number(row.get(y))) for row in rows]
        values = [(left, right) for left, right in values if left is not None and right is not None]
        if not values:
            return False
        axis.scatter([item[0] for item in values], [item[1] for item in values])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        return True

    return draw


def _heatmap_draw(
    rows: list[dict[str, str]],
    *,
    value: str,
    title_label: str,
) -> Callable[[Axes], bool]:
    def draw(axis: Axes) -> bool:
        bandwidths = sorted(
            {number for row in rows if (number := _number(row.get("bandwidth_bps"))) is not None}
        )
        latencies = sorted(
            {
                number
                for row in rows
                if (number := _number(row.get("one_way_latency_ms"))) is not None
            }
        )
        if not bandwidths or not latencies:
            return False
        matrix = np.full((len(latencies), len(bandwidths)), np.nan)
        for row in rows:
            bandwidth = _number(row.get("bandwidth_bps"))
            latency = _number(row.get("one_way_latency_ms"))
            observed = _number(row.get(value))
            if bandwidth in bandwidths and latency in latencies and observed is not None:
                matrix[latencies.index(latency), bandwidths.index(bandwidth)] = observed
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_xticks(range(len(bandwidths)), [f"{item / 1e9:g}" for item in bandwidths])
        axis.set_yticks(range(len(latencies)), [f"{item:g}" for item in latencies])
        axis.set_xlabel("Bandwidth (Gbps)")
        axis.set_ylabel("One-way latency (ms)")
        axis.figure.colorbar(image, ax=axis, label=title_label)
        return True

    return draw


def generate_required_plots(bundle: Path) -> list[str]:
    plots = bundle / "plots"
    plots.mkdir(exist_ok=True)
    whole = _csv_rows(bundle / "whole_expert_results.csv")
    micro = _csv_rows(bundle / "microshard_results.csv")
    planes = _csv_rows(bundle / "data_plane_results.csv")
    coalescing = _csv_rows(bundle / "coalescing_results.csv")
    break_even = _csv_rows(bundle / "break_even_surface.csv")
    resources = _csv_rows(bundle / "resource_timeseries.csv")
    batching = _csv_rows(bundle / "batching_results.csv")
    failures = _csv_rows(bundle / "failure_results.csv")
    verification = _csv_rows(bundle / "verification_results.csv")
    validation = _csv_rows(bundle / "simulator_validation.csv")
    planner = _csv_rows(bundle / "planner_results.csv")
    utility = _csv_rows(bundle / "worker_marginal_utility.csv")
    kimi = _csv_rows(bundle / "kimi_operator_results.csv")
    kimi_projection = _csv_rows(bundle / "kimi_projections.csv")
    codecs = _csv_rows(bundle / "codec_results.csv")
    combined = [*whole, *micro]

    specifications: list[tuple[str, str, Callable[[Axes], bool]]] = [
        (
            PLOT_FILES[0],
            "Monolithic, whole expert, and microshard throughput",
            _bar_draw(combined, "configuration", "throughput", "Verified operations/s"),
        ),
        (
            PLOT_FILES[1],
            "TTFT by execution strategy",
            _bar_draw(combined, "configuration", "ttft_ms", "TTFT (ms)"),
        ),
        (
            PLOT_FILES[2],
            "p95 latency by execution strategy",
            _bar_draw(combined, "configuration", "p95_latency_ms", "p95 latency (ms)"),
        ),
        (
            PLOT_FILES[3],
            "Direct TCP, relayed TCP, and shared memory",
            _bar_draw(planes, "data_plane", "median_total_ms", "Median request time (ms)"),
        ),
        (
            PLOT_FILES[4],
            "Naive versus coalesced traffic",
            _bar_draw(coalescing, "protocol", "messages_per_layer", "Messages/layer"),
        ),
        (
            PLOT_FILES[5],
            "Bytes transferred per output token",
            _bar_draw(planes, "data_plane", "bytes_per_output_token", "Bytes/token"),
        ),
        (
            PLOT_FILES[6],
            "Whole-expert break-even",
            _heatmap_draw(
                break_even, value="remote_whole_expert_beneficial", title_label="Beneficial (0/1)"
            ),
        ),
        (
            PLOT_FILES[7],
            "Microshard break-even",
            _heatmap_draw(
                break_even, value="microsharding_beneficial", title_label="Beneficial (0/1)"
            ),
        ),
        (
            PLOT_FILES[8],
            "Best shard count by worker/network point",
            _scatter_draw(
                break_even,
                "worker_compute_speed_ratio",
                "best_shard_count",
                "Worker speed ratio",
                "Best shards",
            ),
        ),
        (
            PLOT_FILES[9],
            "CPU and GPU utilisation timeline",
            _scatter_draw(
                resources,
                "elapsed_ms",
                "gpu_utilization_percent",
                "Elapsed (ms)",
                "GPU utilisation (%)",
            ),
        ),
        (
            PLOT_FILES[10],
            "Worker queue timeline",
            _scatter_draw(
                resources, "elapsed_ms", "worker_queue_depth", "Elapsed (ms)", "Queue depth"
            ),
        ),
        (
            PLOT_FILES[11],
            "Expert residency by worker",
            _bar_draw(utility, "worker_id", "resident_tensor_bytes", "Resident bytes"),
        ),
        (
            PLOT_FILES[12],
            "Expert ownership map",
            _bar_draw(utility, "worker_id", "owned_expert_count", "Owned experts"),
        ),
        (
            PLOT_FILES[13],
            "Cache hit rate by policy",
            _bar_draw(batching, "policy", "cache_hit_rate", "Cache hit rate"),
        ),
        (
            PLOT_FILES[14],
            "Prefill expert-union deduplication",
            _bar_draw(batching, "policy", "deduplication_ratio", "Deduplication ratio"),
        ),
        (
            PLOT_FILES[15],
            "Aggregate throughput versus concurrency",
            _scatter_draw(whole, "concurrency", "throughput", "Concurrency", "Verified ops/s"),
        ),
        (
            PLOT_FILES[16],
            "Failure recovery latency",
            _bar_draw(failures, "failure_type", "recovery_latency_ms", "Recovery latency (ms)"),
        ),
        (
            PLOT_FILES[17],
            "Detection rate versus verification overhead",
            _scatter_draw(
                verification,
                "verification_overhead_fraction",
                "detection_rate",
                "Verification overhead",
                "Detection rate",
            ),
        ),
        (
            PLOT_FILES[18],
            "Simulator predicted versus measured throughput",
            _scatter_draw(
                validation, "measured_throughput", "predicted_throughput", "Measured", "Predicted"
            ),
        ),
        (
            PLOT_FILES[19],
            "Simulator predicted versus measured p95",
            _scatter_draw(
                validation,
                "measured_p95_latency_ms",
                "predicted_p95_latency_ms",
                "Measured p95 (ms)",
                "Predicted p95 (ms)",
            ),
        ),
        (
            PLOT_FILES[20],
            "Planner regret",
            _bar_draw(planner, "phase", "regret_fraction", "Regret fraction"),
        ),
        (
            PLOT_FILES[21],
            "Worker marginal utility",
            _bar_draw(utility, "worker_id", "mean_utility", "Marginal utility"),
        ),
        (
            PLOT_FILES[22],
            "Kimi K3-shaped operator breakdown",
            _bar_draw(kimi, "component", "elapsed_ms", "Elapsed (ms)"),
        ),
        (
            PLOT_FILES[23],
            "Kimi projected throughput with uncertainty",
            _scatter_draw(
                kimi_projection,
                "node_count",
                "predicted_throughput",
                "Virtual nodes",
                "Projected ops/s",
            ),
        ),
        (
            PLOT_FILES[24],
            "Capacity bytes per worker",
            _bar_draw(utility, "worker_id", "expert_bytes", "Owned expert bytes"),
        ),
        (
            PLOT_FILES[25],
            "Peak RSS per worker",
            _bar_draw(utility, "worker_id", "peak_rss_bytes", "Peak RSS bytes"),
        ),
        (
            PLOT_FILES[26],
            "Transport codec break-even",
            _scatter_draw(
                codecs,
                "raw_bytes",
                "total_time_ns",
                "Raw payload bytes",
                "Encode + transfer + decode (ns)",
            ),
        ),
    ]
    for filename, title, draw in specifications:
        _save(plots / filename, title, draw)

    verdict = {}
    verdict_path = bundle / "verdict.json"
    if verdict_path.is_file():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8-sig"))

    def verdict_draw(axis: Axes) -> bool:
        gates = verdict.get("gates", [])
        if not gates:
            return False
        labels = [f"G{item['gate_id']}" for item in gates]
        values = [1 if item["status"] == "PASS" else 0 for item in gates]
        colors = ["#2ca02c" if value else "#d62728" for value in values]
        axis.bar(labels, values, color=colors)
        axis.set_ylim(0, 1.15)
        axis.set_ylabel("Gate pass")
        return True

    _save(plots / PLOT_FILES[27], "Cumulative Experiment 010 verdict", verdict_draw)
    return [f"plots/{name}" for name in PLOT_FILES]


REPORT_SECTIONS = (
    "Original project goal",
    "Experiment 010 question",
    "Previous evidence",
    "Hardware and environment",
    "Models and fixtures",
    "Colibri CUDA result",
    "Virtual-node architecture",
    "Expert RPC implementation",
    "Microshard implementation",
    "Direct data-plane comparison",
    "Coalesced protocol result",
    "Capacity-isolation result",
    "Prefill result",
    "Decode result",
    "Concurrent-service result",
    "Routing-aware batching result",
    "Network break-even result",
    "Codec result",
    "Failure recovery result",
    "Incorrect-worker detection result",
    "Planner result",
    "Simulator calibration result",
    "Kimi K3-shaped result",
    "Remaining single-machine limitations",
    "Exact questions that require physical machines",
    "Implications for the original project thesis",
    "Go or no-go recommendation",
    "Overall verdict",
)


def build_report(
    *,
    verdict: dict[str, Any],
    environment: dict[str, Any],
    cuda: dict[str, Any],
    summaries: dict[str, Any],
) -> str:
    overall = verdict.get("verdict", "PARTIAL")
    mode = verdict.get("mode", "unknown")
    answer = verdict.get("answer_first", "No valid official conclusion was produced.")
    sections: dict[str, list[str]] = {
        "Original project goal": [
            "Run arbitrary open LLMs across heterogeneous consumer hardware while preserving semantics, rejecting harmful resources, and enabling models that exceed any one node's capacity."
        ],
        "Experiment 010 question": [answer],
        "Previous evidence": [
            "Experiments 001-009 established process isolation, direct activation transport, logical microshards, heterogeneous role selection, over-VRAM execution, and exact Colibri routing/token replay. Experiment 009's confirmed bundle remains the baseline.",
            str(summaries.get("component_reuse", "No component map was recorded.")),
        ],
        "Hardware and environment": [json.dumps(environment, sort_keys=True, default=str)],
        "Models and fixtures": [json.dumps(summaries.get("models", {}), sort_keys=True)],
        "Colibri CUDA result": [
            f"DLL loaded={cuda.get('dll_loaded')}; RTX detected={cuda.get('device_detected')}; kernel executed={cuda.get('kernel_executed')}; correctness={cuda.get('correctness_passed')}; resident tensors={cuda.get('resident_tensor_count')}; resident bytes={cuda.get('resident_tensor_bytes')}; tensor shapes={cuda.get('tensor_shapes')}; H2D bytes={cuda.get('host_to_device_bytes')}; D2H bytes={cuda.get('device_to_host_bytes')}. This is a fused Colibri CUDA expert-kernel proof, not Level A generation through a Colibri RPC hook."
        ],
        "Virtual-node architecture": [str(summaries.get("workers", "Not measured."))],
        "Expert RPC implementation": [str(summaries.get("whole_expert", "Not measured."))],
        "Microshard implementation": [str(summaries.get("microshards", "Not measured."))],
        "Direct data-plane comparison": [str(summaries.get("data_planes", "Not measured."))],
        "Coalesced protocol result": [str(summaries.get("coalescing", "Not measured."))],
        "Capacity-isolation result": [str(summaries.get("capacity", "Not measured."))],
        "Prefill result": [str(summaries.get("prefill", "Not measured."))],
        "Decode result": [str(summaries.get("decode", "Not measured."))],
        "Concurrent-service result": [str(summaries.get("concurrency", "Not measured."))],
        "Routing-aware batching result": [str(summaries.get("batching", "Not measured."))],
        "Network break-even result": [str(summaries.get("break_even", "Not measured."))],
        "Codec result": [str(summaries.get("codecs", "Not measured."))],
        "Failure recovery result": [str(summaries.get("failures", "Not measured."))],
        "Incorrect-worker detection result": [str(summaries.get("verification", "Not measured."))],
        "Planner result": [str(summaries.get("planner", "Not measured."))],
        "Simulator calibration result": [str(summaries.get("simulator", "Not measured."))],
        "Kimi K3-shaped result": [
            str(summaries.get("kimi", "Not measured.")),
            "Fixture measurements are SYNTHETIC_FIXTURE operator evidence, never full-model Kimi K3 inference.",
        ],
        "Remaining single-machine limitations": [
            str(
                summaries.get(
                    "limitations",
                    "Independent NICs, memory buses, failure domains, clocks, and thermal envelopes are absent on one workstation.",
                )
            )
        ],
        "Exact questions that require physical machines": [
            str(
                summaries.get(
                    "physical_questions",
                    "Whether independent hosts preserve the measured ranking under real NIC/DMA stacks; whether aggregate memory bandwidth and storage scale; whether clock drift, packet paths, machine loss, thermal limits, and heterogeneous accelerators change correctness or tail latency.",
                )
            )
        ],
        "Implications for the original project thesis": [
            "Only passed measured gates strengthen the thesis. Emulated networks and calibrated projections narrow the next experiment; neither proves physical distributed inference."
        ],
        "Go or no-go recommendation": [
            str(verdict.get("recommendation", "NO-GO until failed gates close."))
        ],
        "Overall verdict": [
            (
                f"Mode `{mode}` produced the official `{overall}` verdict."
                if mode == "full"
                else f"Mode `{mode}` produced `{overall}`; only `full` mode can issue an official verdict."
            ),
            "Failed gates: " + ", ".join(str(item) for item in verdict.get("failed_gates", [])),
            "### Required questions, explicit answers",
            "",
            *[
                f"- {question}: {response}"
                for question, response in summaries.get("required_answers", {}).items()
            ],
        ],
    }
    lines = ["# Experiment 010 report", "", "## Answer-first summary", "", answer]
    for index, name in enumerate(REPORT_SECTIONS, start=1):
        lines.extend(["", f"## {index}. {name}", "", *sections[name]])
    return "\n".join(lines) + "\n"
