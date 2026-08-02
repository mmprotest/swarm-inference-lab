"""Raw-data plots and answer-first Markdown reporting for Experiment 008."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _save_plot(
    path: Path,
    *,
    title: str,
    source: str,
    draw: Callable[[Any, Any], bool],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
    available = draw(figure, axis)
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
    if not available:
        axis.clear()
        axis.set_axis_off()
        axis.text(
            0.5,
            0.55,
            "Measurement unavailable",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color="#5d6470",
        )
        axis.text(
            0.5,
            0.42,
            "No completed measured rows were present; no zero was imputed.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6b7280",
        )
        axis.set_title(title, loc="left", fontsize=15, fontweight="bold")
    figure.text(0.01, 0.005, f"Source: {source}", fontsize=8, color="#6b7280")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_required_plots(bundle_root: Path) -> list[str]:
    """Generate the ten required figures strictly from persisted raw artifacts."""

    ablation = _read_csv(bundle_root / "ablation_results.csv")
    expert = _read_csv(bundle_root / "expert_activation_matrix.csv")
    predictions = _read_csv(bundle_root / "cost_model_predictions.csv")
    configurations = [row.get("configuration", "?") for row in ablation]
    paths: list[str] = []

    def path(name: str) -> Path:
        result = bundle_root / "plots" / name
        paths.append(str(result.relative_to(bundle_root)))
        return result

    def throughput(_figure: Any, axis: Any) -> bool:
        decode = [_number(row.get("decode_tokens_per_second")) for row in ablation]
        mixed = [_number(row.get("mixed_verified_tokens_per_second")) for row in ablation]
        if not any(value is not None for value in [*decode, *mixed]):
            return False
        x = list(range(len(configurations)))
        axis.bar(
            [value - 0.2 for value in x],
            [float("nan") if value is None else value for value in decode],
            0.4,
            label="Decode",
        )
        axis.bar(
            [value + 0.2 for value in x],
            [float("nan") if value is None else value for value in mixed],
            0.4,
            label="Mixed verified",
        )
        axis.set_xticks(x, configurations)
        axis.set_ylabel("tokens / second")
        axis.legend(frameon=False)
        return True

    _save_plot(
        path("01_throughput_by_configuration.png"),
        title="Throughput by cumulative configuration",
        source="ablation_results.csv",
        draw=throughput,
    )

    def ttft(_figure: Any, axis: Any) -> bool:
        eight = [_number(row.get("ttft_8k")) for row in ablation]
        thirty_two = [_number(row.get("ttft_32k")) for row in ablation]
        if not any(value is not None for value in [*eight, *thirty_two]):
            return False
        x = list(range(len(configurations)))
        axis.plot(
            x, [float("nan") if value is None else value for value in eight], "o-", label="8K"
        )
        axis.plot(
            x, [float("nan") if value is None else value for value in thirty_two], "o-", label="32K"
        )
        axis.set_xticks(x, configurations)
        axis.set_ylabel("time to first token (ms)")
        axis.legend(frameon=False)
        return True

    _save_plot(
        path("02_ttft_by_configuration.png"),
        title="Long-prefill time to first token",
        source="ablation_results.csv",
        draw=ttft,
    )

    def memory(_figure: Any, axis: Any) -> bool:
        vram = [_number(row.get("peak_vram")) for row in ablation]
        ram = [_number(row.get("peak_ram")) for row in ablation]
        if not any(value is not None for value in [*vram, *ram]):
            return False
        x = list(range(len(configurations)))
        gib = 1024**3
        axis.bar(
            [value - 0.2 for value in x],
            [float("nan") if value is None else value / gib for value in vram],
            0.4,
            label="VRAM",
        )
        axis.bar(
            [value + 0.2 for value in x],
            [float("nan") if value is None else value / gib for value in ram],
            0.4,
            label="System RAM",
        )
        axis.set_xticks(x, configurations)
        axis.set_ylabel("peak used memory (GiB)")
        axis.legend(frameon=False)
        return True

    _save_plot(
        path("03_peak_vram_ram.png"),
        title="Peak VRAM and system RAM",
        source="ablation_results.csv",
        draw=memory,
    )

    def pcie(_figure: Any, axis: Any) -> bool:
        values = [_number(row.get("pcie_bytes_per_token")) for row in ablation]
        if not any(value is not None for value in values):
            return False
        axis.bar(
            configurations,
            [float("nan") if value is None else value / 1024**2 for value in values],
            color="#4c78a8",
        )
        axis.set_ylabel("sampled PCIe MiB / output token")
        return True

    _save_plot(
        path("04_pcie_bytes_per_token.png"),
        title="PCIe traffic on decode workload",
        source="ablation_results.csv; resource_timeseries.csv",
        draw=pcie,
    )

    def overlap(figure: Any, axis: Any) -> bool:
        profile = _read_json(bundle_root / "hardware_profile.json", {})
        measurement = profile.get("overlap_measurement", {})
        cpu = measurement.get("cpu_interval_monotonic_ns")
        gpu = measurement.get("gpu_and_transfer_interval_monotonic_ns")
        if not (
            isinstance(cpu, list)
            and len(cpu) == 2
            and isinstance(gpu, list)
            and len(gpu) == 2
            and all(isinstance(value, (int, float)) for value in [*cpu, *gpu])
        ):
            return False
        origin = min(float(cpu[0]), float(gpu[0]))

        def interval(values: list[Any]) -> tuple[float, float]:
            start = (float(values[0]) - origin) / 1_000_000
            duration = (float(values[1]) - float(values[0])) / 1_000_000
            return start, duration

        cpu_start, cpu_duration = interval(cpu)
        gpu_start, gpu_duration = interval(gpu)
        axis.broken_barh([(cpu_start, cpu_duration)], (20, 8), facecolors="#f28e2b")
        axis.broken_barh([(gpu_start, gpu_duration)], (8, 8), facecolors="#4c78a8")
        axis.set_yticks([24, 12], ["CPU matmul", "CUDA matmul + H2D"])
        axis.set_xlabel("elapsed time from first launch (ms)")
        overlap_percent = _number(measurement.get("overlap_percent"))
        if overlap_percent is not None:
            axis.text(
                0.99,
                0.96,
                f"interval overlap: {overlap_percent:.1f}%",
                transform=axis.transAxes,
                ha="right",
                va="top",
            )
        figure.text(
            0.99,
            0.005,
            "Hardware-profiler calibration; target-model overlap is reported separately and may be unsupported.",
            fontsize=8,
            color="#6b7280",
            ha="right",
        )
        return True

    _save_plot(
        path("05_cpu_gpu_overlap.png"),
        title="CPU/GPU execution overlap timeline",
        source="hardware_profile.json; profiler_trace/pytorch_overlap.json",
        draw=overlap,
    )

    def activation(_figure: Any, axis: Any) -> bool:
        rows = [row for row in expert if _number(row.get("activation_probability")) is not None]
        if not rows:
            return False
        rows.sort(key=lambda row: _number(row.get("activation_probability")) or 0, reverse=True)
        rows = rows[:40]
        labels = [f"L{row.get('layer_id')} E{row.get('expert_id')}" for row in rows]
        values = [_number(row.get("activation_probability")) or 0 for row in rows]
        axis.bar(range(len(rows)), values, color="#f28e2b")
        axis.set_xticks(range(len(rows)), labels, rotation=90, fontsize=7)
        axis.set_ylabel("activation probability")
        return True

    _save_plot(
        path("06_expert_activation_frequency.png"),
        title="Most frequently activated experts",
        source="expert_activation_matrix.csv",
        draw=activation,
    )

    def cache(_figure: Any, axis: Any) -> bool:
        values = [_number(row.get("expert_cache_hit_rate")) for row in ablation]
        if not any(value is not None for value in values):
            return False
        axis.bar(
            configurations,
            [float("nan") if value is None else value for value in values],
            color="#e15759",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("cache hit rate")
        return True

    _save_plot(
        path("07_expert_cache_hit_rate.png"),
        title="Expert cache hit rate",
        source="ablation_results.csv; expert_trace_summary.json",
        draw=cache,
    )

    def prefetch(_figure: Any, axis: Any) -> bool:
        summary = _read_json(bundle_root / "expert_trace_summary.json", {})
        useful = _number(summary.get("useful_prefetch_bytes"))
        wasted = _number(summary.get("wasted_prefetch_bytes"))
        if useful is None and wasted is None:
            return False
        axis.bar(
            ["Useful", "Wasted"],
            [(useful or 0) / 1024**2, (wasted or 0) / 1024**2],
            color=["#59a14f", "#e15759"],
        )
        axis.set_ylabel("prefetched MiB")
        return True

    _save_plot(
        path("08_prefetch_useful_wasted.png"),
        title="Useful versus wasted expert prefetch bytes",
        source="expert_trace_summary.json",
        draw=prefetch,
    )

    def prediction(_figure: Any, axis: Any) -> bool:
        rows = [
            row
            for row in predictions
            if _number(row.get("predicted_ms")) is not None
            and _number(row.get("measured_ms")) is not None
        ]
        if not rows:
            return False
        predicted = [_number(row.get("predicted_ms")) or 0 for row in rows]
        measured = [_number(row.get("measured_ms")) or 0 for row in rows]
        axis.scatter(predicted, measured, color="#4c78a8")
        limit = max([*predicted, *measured])
        axis.plot([0, limit], [0, limit], "--", color="#6b7280", label="perfect prediction")
        axis.set_xlabel("predicted completion (ms)")
        axis.set_ylabel("measured completion (ms)")
        axis.legend(frameon=False)
        return True

    _save_plot(
        path("09_predicted_vs_measured.png"),
        title="Cost-model prediction versus measurement",
        source="cost_model_predictions.csv",
        draw=prediction,
    )

    def waterfall(_figure: Any, axis: Any) -> bool:
        changes = [_number(row.get("decode_change_vs_previous")) for row in ablation]
        if not any(value is not None for value in changes[1:]):
            return False
        running = 0.0
        starts: list[float] = []
        heights: list[float] = []
        colors: list[str] = []
        for value in changes:
            if value is None:
                starts.append(float("nan"))
                heights.append(float("nan"))
                colors.append("#9ca3af")
                continue
            delta = value * 100
            starts.append(running if delta >= 0 else running + delta)
            heights.append(abs(delta))
            colors.append("#59a14f" if delta >= 0 else "#e15759")
            running += delta
        axis.bar(configurations, heights, bottom=starts, color=colors)
        axis.axhline(0, color="#374151", linewidth=0.8)
        axis.set_ylabel("cumulative decode change (percentage points)")
        return True

    _save_plot(
        path("10_cumulative_waterfall.png"),
        title="Cumulative decode improvement waterfall",
        source="ablation_results.csv",
        draw=waterfall,
    )
    return paths


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gib(value: Any) -> str:
    number = _number(value)
    return "not measured" if number is None else f"{number / 1024**3:.3f} GiB"


def build_report(bundle_root: Path) -> str:
    """Build the required answer-first report solely from saved bundle evidence."""

    verdict = _read_json(bundle_root / "verdict.json", {})
    preflight = _read_json(bundle_root / "model_preflight.json", {})
    environment = _read_json(bundle_root / "environment.json", {})
    manifest = _read_json(bundle_root / "manifest.json", {})
    baseline = _read_json(bundle_root / "baseline_search.json", {})
    adaptive = _read_json(bundle_root / "adaptive_plan.json", {})
    correctness = _read_json(bundle_root / "correctness_results.json", {})
    residency = _read_json(bundle_root / "residency_accounting.json", {})
    expert = _read_json(bundle_root / "expert_trace_summary.json", {})
    ablation = _read_csv(bundle_root / "ablation_results.csv")
    benchmarks = _read_csv(bundle_root / "benchmark_results.csv")
    by_config = {row.get("configuration", ""): row for row in ablation}
    stock = by_config.get("A", {})
    final = by_config.get("G", {})
    completed_mixed = {
        row.get("configuration", ""): row
        for row in benchmarks
        if row.get("status") == "COMPLETED" and row.get("workload") == "mixed"
    }
    stock_mixed = completed_mixed.get("A", {})
    final_mixed = completed_mixed.get("G", {})
    gates = verdict.get("gates", []) if isinstance(verdict, dict) else []

    passed = [str(row.get("name")) for row in gates if row.get("status") == "PASS"]
    failed = [str(row.get("name")) for row in gates if row.get("status") == "FAIL"]
    not_evaluated = [str(row.get("name")) for row in gates if row.get("status") == "NOT_EVALUATED"]
    overall = verdict.get("overall_verdict", "FAIL")
    summary = verdict.get("answer_first_summary") or (
        "No official conclusion was available because the run did not produce complete measured evidence."
    )
    role_gpu = residency.get("planned_gpu_tensor_roles", [])
    role_cpu = residency.get("planned_cpu_tensor_roles", [])
    role_managed = residency.get("backend_managed_tensor_roles", [])
    capacity_witness = residency.get("capacity_accounting_witness") or {}
    split = residency.get("split_tensors", [])
    cached = expert.get("gpu_cached_experts")
    techniques = adaptive.get("technique_decisions", [])
    rejected = [row for row in techniques if isinstance(row, dict) and not row.get("enabled")]
    comparisons = correctness.get("comparisons", [])

    def comparison_counts(configuration: str, workload: str | None = None) -> tuple[int, int]:
        selected = [
            row
            for row in comparisons
            if isinstance(row, dict)
            and row.get("configuration") == configuration
            and (workload is None or row.get("workload") == workload)
        ]
        return sum(row.get("exact_token_identity") is True for row in selected), len(selected)

    b_exact, b_total = comparison_counts("B")
    g_exact, g_total = comparison_counts("G")
    g_mixed_exact, g_mixed_total = comparison_counts("G", "mixed")
    g_nonmixed_exact = g_exact - g_mixed_exact
    g_nonmixed_total = g_total - g_mixed_total
    overlap = _number(final.get("cpu_gpu_overlap_percent"))
    overlap_text = "not measured" if overlap is None else f"{_fmt(overlap)}%"
    visible_transfer_ms = expert.get("visible_transfer_latency_removed_ms")
    visible_transfer_text = (
        "not measured" if visible_transfer_ms is None else f"{_fmt(visible_transfer_ms)} ms"
    )
    visible_transfer_answer = (
        "not measured"
        if visible_transfer_ms is None
        else f"{_fmt(visible_transfer_ms)} ms removed from the critical path"
    )

    table_header = (
        "| Config | Decode tok/s | 8K TTFT ms | 32K TTFT ms | Mixed verified tok/s | "
        "Interactive p95 ms | Peak VRAM | Peak RAM | Token identity |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    table_rows = []
    for row in ablation:
        table_rows.append(
            "| {configuration} | {decode} | {ttft8} | {ttft32} | {mixed} | {p95} | {vram} | {ram} | {identity} |".format(
                configuration=row.get("configuration", "?"),
                decode=_fmt(_number(row.get("decode_tokens_per_second"))),
                ttft8=_fmt(_number(row.get("ttft_8k"))),
                ttft32=_fmt(_number(row.get("ttft_32k"))),
                mixed=_fmt(_number(row.get("mixed_verified_tokens_per_second"))),
                p95=_fmt(_number(row.get("interactive_p95"))),
                vram=_gib(row.get("peak_vram")),
                ram=_gib(row.get("peak_ram")),
                identity=_fmt(_number(row.get("token_identity_rate"))),
            )
        )

    backend_limitations = preflight.get("backend_limitations") or manifest.get(
        "backend_limitations", []
    )
    lines = [
        "# Experiment 008: Single-Host Adaptive MoE Saturation",
        "",
        "## Answer-first summary",
        "",
        str(summary),
        "",
        f"**Overall verdict: `{overall}`.** Passed gates: {', '.join(passed) or 'none'}. "
        f"Failed gates: {', '.join(failed) or 'none'}. Not evaluated: {', '.join(not_evaluated) or 'none'}.",
        "",
        "## 1. Experiment question",
        "",
        "Can one Windows workstation automatically combine RTX GPU compute and memory with CPU compute and system RAM to execute a sparse MoE larger than physical VRAM, and beat a fairly tuned stock offload plan on at least one meaningful workload?",
        "",
        "## 2. Selected model and why",
        "",
        f"Model: `{preflight.get('model_id', 'not resolved')}` from artifact revision `{preflight.get('artifact_repository_revision', 'not resolved')}`; "
        f"resolved local identity: `{preflight.get('model_revision', 'not resolved')}`. "
        f"Architecture: `{preflight.get('model_architecture', 'not measured')}`; quantization: `{preflight.get('quantization_format', 'not measured')}`. "
        f"Tensor storage: {_gib(preflight.get('total_tensor_bytes'))}; expert storage: {_gib(preflight.get('total_expert_bytes'))}. "
        f"Genuinely exceeds 32 GiB: {_fmt(preflight.get('genuinely_exceeds_32gb'))}; exceeds physical VRAM: {_fmt(preflight.get('genuinely_exceeds_physical_vram'))}.",
        "",
        "The preferred candidate is attempted first. The fallback is permitted only after an explicit recorded rejection; no silent model switch is allowed.",
        "",
        "## 3. Hardware and software environment",
        "",
        f"Host fingerprint: `{environment.get('hardware_fingerprint', 'not captured')}`. Python: `{environment.get('python', 'not captured')}`; "
        f"PyTorch: `{environment.get('pytorch', 'not captured')}`; CUDA runtime: `{environment.get('cuda_runtime', 'not captured')}`; "
        f"GPU: `{environment.get('gpu', 'not captured')}`; CPU: `{environment.get('cpu', 'not captured')}`. "
        f"Physical VRAM: {_gib(preflight.get('physical_vram_bytes'))}; available RAM at preflight: {_gib(preflight.get('system_ram_available_bytes'))}.",
        "",
        "## 4. Architecture implemented",
        "",
        "The integrated implementation provides streaming GGUF tensor inspection, logical tensor tiles, valid matched expert projection slices, measured CPU/CUDA/PCIe/storage profiles, a critical-path cost model, bounded baseline search, capability-aware placement, expert trace statistics, byte-bounded cache policy, bounded expert predictors, phase-specific plans, resumable execution, deterministic comparison, telemetry, raw-data plots, and gate evaluation.",
        "",
        "## 5. Research concepts represented",
        "",
        "The A-G plan schema represents tensor-granular placement, asymmetric CPU/GPU partitioning, asynchronous overlap, activation-aware caching, predictive prefetch, separate prefill/decode plans, and positive-utility selection. A technique is marked enabled only when the backend exposes its required hooks and, under G, its measured incremental utility is positive.",
        "",
        "## 6. Baselines",
        "",
        f"Bounded stock candidates recorded: {len(baseline.get('candidates', [])) if isinstance(baseline, dict) else 0}. "
        "The selected baseline is stored separately for decode, 8K prefill, 32K prefill, and mixed service. Every attempted command, status, and exit code is retained in `baseline_search.json` and `logs/`.",
        "",
        "## 7. Cumulative ablation results",
        "",
        table_header,
        *table_rows,
        "",
        "![Throughput](plots/01_throughput_by_configuration.png)",
        "",
        "![Waterfall](plots/10_cumulative_waterfall.png)",
        "",
        "## 8. Capacity result",
        "",
        f"Weights exceeded GPU memory: {_fmt(preflight.get('genuinely_exceeds_physical_vram'))}. "
        f"Backend-reported CPU-mapped model buffer: {_gib(residency.get('backend_reported_cpu_model_bytes'))}; "
        f"CUDA model buffer: {_gib(residency.get('backend_reported_gpu_model_bytes'))}. "
        f"The additive no-mmap capacity witness used {_gib(capacity_witness.get('host_model_bytes'))} host and {_gib(capacity_witness.get('gpu_model_bytes'))} GPU model buffers with {_fmt(capacity_witness.get('buffer_sum_error_fraction'), digits=6)} relative accounting error. "
        f"Explicit GPU roles: {', '.join(map(str, role_gpu)) or 'none'}; explicit CPU roles: {', '.join(map(str, role_cpu)) or 'none'}; "
        f"backend-managed roles: {', '.join(map(str, role_managed)) or 'none'}. "
        f"Residency reconciliation: {_fmt(residency.get('reconciled'))}.",
        "",
        "## 9. Correctness result",
        "",
        f"Deterministic executions: {_fmt(correctness.get('deterministic_execution_count'))}; token identity rate: {_fmt(correctness.get('token_identity_rate'))}; "
        f"generated-text identity rate: {_fmt(correctness.get('text_identity_rate'))}; "
        f"fixture equivalence checks passed: {_fmt(correctness.get('fixture_checks_passed'))}. "
        f"Final-logit comparison: {correctness.get('final_logits_limitation', 'not measured')}. "
        f"Measured candidate breakdown: B matched {b_exact}/{b_total}; G matched {g_exact}/{g_total}. "
        f"G matched all {g_nonmixed_exact}/{g_nonmixed_total} decode and prefill cases, but only "
        f"{g_mixed_exact}/{g_mixed_total} mixed-service streams.",
        "",
        "## 10. Decode result",
        "",
        f"Stock decode: {_fmt(_number(stock.get('decode_tokens_per_second')))} tok/s. Adaptive decode: {_fmt(_number(final.get('decode_tokens_per_second')))} tok/s. "
        f"Adaptive change versus stock: {_fmt(_number(final.get('decode_change_vs_stock')))}. Median/p95 token latency is retained per request in `benchmark_results.csv` and logs.",
        "",
        "## 11. Prefill result",
        "",
        f"Stock 8K/32K TTFT: {_fmt(_number(stock.get('ttft_8k')))} / {_fmt(_number(stock.get('ttft_32k')))} ms. "
        f"Adaptive 8K/32K TTFT: {_fmt(_number(final.get('ttft_8k')))} / {_fmt(_number(final.get('ttft_32k')))} ms. "
        f"Prefill and decode plans differ: {_fmt(adaptive.get('prefill_decode_plans_differ'))}.",
        "",
        "![TTFT](plots/02_ttft_by_configuration.png)",
        "",
        "## 12. Mixed-service result",
        "",
        f"Stock/adaptive raw generated throughput: "
        f"{_fmt(_number(stock_mixed.get('combined_generated_tokens_per_second')))} / "
        f"{_fmt(_number(final_mixed.get('combined_generated_tokens_per_second')))} tok/s. "
        f"Stock/adaptive mixed verified throughput: {_fmt(_number(stock.get('mixed_verified_tokens_per_second')))} / {_fmt(_number(final.get('mixed_verified_tokens_per_second')))} tok/s. "
        f"Stock/adaptive interactive p95: {_fmt(_number(stock.get('interactive_p95')))} / {_fmt(_number(final.get('interactive_p95')))} ms. "
        f"Adaptive verification status was `{final_mixed.get('verification_status', 'not measured')}`; "
        f"the measured zero verified throughput reflects {g_mixed_exact}/{g_mixed_total} exact mixed outputs, not an imputed missing value.",
        "",
        "## 13. Planner quality",
        "",
        f"Measured regret: {_fmt(verdict.get('planner_quality', {}).get('regret_fraction'))}; pairwise predicted/measured ranking agreement: "
        f"{_fmt(verdict.get('planner_quality', {}).get('ranking_agreement_fraction'))}. Every plan explanation is stored in `candidate_plans.json`, `prefill_plan.json`, and `decode_plan.json`.",
        "",
        "![Prediction quality](plots/09_predicted_vs_measured.png)",
        "",
        "## 14. Resource utilisation",
        "",
        f"Peak adaptive VRAM/RAM: {_gib(final.get('peak_vram'))} / {_gib(final.get('peak_ram'))}. "
        f"Sampled adaptive decode PCIe bytes per token: {_fmt(_number(final.get('pcie_bytes_per_token')))}. "
        f"Proven target-runtime CPU/GPU overlap: {overlap_text}. "
        "Temperatures, power, per-core CPU load, GPU compute/memory utilisation, PCIe rates, and disk counters are in `resource_timeseries.csv`.",
        "",
        "![Memory](plots/03_peak_vram_ram.png)",
        "",
        "![PCIe](plots/04_pcie_bytes_per_token.png)",
        "",
        "## 15. Positive CPU utility",
        "",
        f"System RAM contributes capacity: {_fmt(residency.get('system_ram_contributes'))}. CPU compute has positive measured performance utility: {_fmt(residency.get('positive_cpu_performance_utility'))}. "
        "CPU utilisation alone was not counted as utility.",
        "",
        "## 16. Techniques rejected by the planner",
        "",
        *(
            [f"- `{row.get('technique')}` -- {row.get('reason')}" for row in rejected]
            or ["No rejected-technique decisions were available."]
        ),
        "",
        "## 17. Limitations",
        "",
        *(
            [f"- {item}" for item in backend_limitations]
            or ["- No backend limitation was recorded."]
        ),
        "",
        f"Exact split tensors: {json.dumps(split, sort_keys=True)}; {residency.get('split_explanation', 'no split explanation')}. "
        f"Exact GPU-cached experts: {json.dumps(cached, sort_keys=True)}. "
        f"Useful/wasted prefetch bytes: {_fmt(expert.get('useful_prefetch_bytes'))} / {_fmt(expert.get('wasted_prefetch_bytes'))}. "
        f"Visible transfer latency removed: {visible_transfer_text}. "
        f"Configuration B also peaked at {_gib(by_config.get('B', {}).get('peak_ram'))} system RAM and reached a median 32K TTFT of "
        f"{_fmt(_number(by_config.get('B', {}).get('ttft_32k')))} ms; this is measured evidence of severe pressure in that plan, "
        "not proof of a single dominant bottleneck for G.",
        "",
        "## 18. Implications for physical multi-machine execution",
        "",
        "The tensor-tile identifiers and logical slices are reusable for future physical placement, but local PCIe measurements are not network measurements. Any network extrapolation must remain PROJECTED until physical hosts execute and exchange the same logical tensors.",
        "",
        "## 19. Implications for Kimi K3",
        "",
        "This experiment tests the local scheduling and evidence architecture needed before a Kimi K3-scale swarm. It does not demonstrate Kimi K3, remote tensor transport, or distributed expert execution. The distributed thesis is strengthened only to the extent indicated by the measured capacity/performance gates above; missing or negative performance evidence weakens the claim that finer distribution is automatically faster.",
        "",
        "## 20. Overall verdict",
        "",
        f"`{overall}` -- {summary}",
        "",
        "### Direct answers",
        "",
        f"- Did the model genuinely exceed GPU memory? {_fmt(preflight.get('genuinely_exceeds_physical_vram'))}.",
        f"- What remained in RAM? Adaptive-plan load measured {_gib(residency.get('backend_reported_cpu_model_bytes'))} mapped (potentially aliasing CUDA copies); the additive no-mmap capacity witness held {_gib(capacity_witness.get('host_model_bytes'))} in host model buffers. Explicitly forced CPU roles: {', '.join(map(str, role_cpu)) or 'none'}; backend-managed tensor identities are not claimed as exact.",
        f"- What remained in VRAM? Adaptive-plan load measured {_gib(residency.get('backend_reported_gpu_model_bytes'))}; the additive no-mmap capacity witness held {_gib(capacity_witness.get('gpu_model_bytes'))}. Explicitly forced GPU roles: {', '.join(map(str, role_gpu)) or 'none'}; backend-managed tensor identities are not claimed as exact.",
        f"- Which tensors were split? {json.dumps(split, sort_keys=True)}; {residency.get('split_explanation', 'not established')}.",
        f"- Which experts were cached? {json.dumps(cached, sort_keys=True)}.",
        f"- How much PCIe traffic occurred? {_fmt(_number(final.get('pcie_bytes_per_token')))} sampled bytes per output token for adaptive decode.",
        f"- How much transfer time was hidden? {visible_transfer_answer}.",
        f"- Did expert prediction help? {expert.get('prediction_conclusion', 'not measured')}.",
        f"- Were prefill and decode plans different? {_fmt(adaptive.get('prefill_decode_plans_differ'))}.",
        f"- Did the CPU improve a meaningful metric? {_fmt(residency.get('positive_cpu_performance_utility'))}; capacity utility: {_fmt(residency.get('system_ram_contributes'))}.",
        f"- Did the planner beat stock offloading? {verdict.get('planner_beat_stock', 'not measured')}.",
        f"- Did the planner beat equal microsharding? {verdict.get('planner_beat_equal_microsharding', 'not measured')} (the executable comparator is an equal 24/48 expert-layer CPU/GPU offload, not within-tensor projection microsharding).",
        f"- Largest/least technique contributions: {verdict.get('largest_technique_contribution', 'not measured')} / {verdict.get('least_technique_contribution', 'not measured')}.",
        f"- Dominant bottleneck: {verdict.get('dominant_bottleneck', 'not established')}.",
        f"- Effect on the distributed swarm thesis: {verdict.get('swarm_thesis_implication', 'not established')}.",
        "",
        "### Evidence boundaries",
        "",
        f"Run mode: `{manifest.get('run_mode', 'unknown')}`. Only rows tagged `MEASURED` are used for official gates. Fixture checks are `EMULATED`; cost-only extrapolations are `PROJECTED`. Missing values are null in machine-readable artifacts, never zero-filled.",
    ]
    return "\n".join(lines) + "\n"


def build_bundle_readme(bundle_root: Path) -> str:
    manifest = _read_json(bundle_root / "manifest.json", {})
    verdict = _read_json(bundle_root / "verdict.json", {})
    return (
        "# Experiment 008 evidence bundle\n\n"
        f"Run ID: `{manifest.get('run_id', 'unknown')}`  \n"
        f"Mode: `{manifest.get('run_mode', 'unknown')}`  \n"
        f"Verdict: `{verdict.get('overall_verdict', 'not yet evaluated')}`\n\n"
        "`report.md` is the answer-first narrative. JSON and CSV files are the source evidence; "
        "plots are generated from those saved raw files. `checkpoint.json` records resumable stages. "
        "Only a completed `--full` run can pass official acceptance gates.\n"
    )
