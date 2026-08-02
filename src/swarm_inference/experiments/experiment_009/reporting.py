"""Evidence-backed plots and narrative reporting for Experiment 009."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    try:
        if value in (None, "", "None", "null"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _save(
    target: Path,
    title: str,
    draw: Callable[[Axes], bool],
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    measured = draw(axis)
    axis.set_title(title)
    if not measured:
        axis.text(
            0.5,
            0.5,
            "Not measured in this run",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _bar(
    rows: list[dict[str, str]],
    *,
    key: str,
    value: str,
    ylabel: str,
) -> Callable[[Axes], bool]:
    def draw(axis: Axes) -> bool:
        pairs: list[tuple[str, float]] = []
        for row in rows:
            name = row.get(key, "")
            number = _number(row.get(value))
            if name and number is not None:
                pairs.append((name, number))
        if not pairs:
            return False
        axis.bar([name for name, _ in pairs], [number for _, number in pairs])
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        return True

    return draw


def generate_required_plots(bundle: Path) -> list[str]:
    plots = bundle / "plots"
    plots.mkdir(exist_ok=True)
    overhead = _rows(bundle / "adapter_overhead_results.csv")
    activation = _rows(bundle / "expert_activation.csv")
    residency = _rows(bundle / "tier_residency.csv")
    storage = _rows(bundle / "storage_events.csv")
    tuning = _rows(bundle / "tuning_results.csv")
    policies = _rows(bundle / "heldout_policy_results.csv")

    def throughput(axis: Axes) -> bool:
        groups: dict[str, list[float]] = {}
        for row in overhead:
            value = _number(row.get("decode_tokens_per_second"))
            if value is not None:
                groups.setdefault(row.get("configuration", "unknown"), []).append(value)
        if not groups:
            return False
        labels = sorted(groups)
        axis.bar(labels, [median(groups[label]) for label in labels])
        axis.set_ylabel("Median decode tokens/s")
        return True

    _save(
        plots / "01_direct_vs_adapter_throughput.png", "Direct Colibri vs swarm adapter", throughput
    )

    def overhead_plot(axis: Axes) -> bool:
        by_workload: dict[str, dict[str, list[float]]] = {}
        for row in overhead:
            value = _number(row.get("latency_ms"))
            if value is None:
                continue
            by_workload.setdefault(row.get("workload", "fixture"), {}).setdefault(
                row.get("configuration", "unknown"), []
            ).append(value)
        labels, values = [], []
        for workload, configs in sorted(by_workload.items()):
            if configs.get("direct") and configs.get("adapter"):
                labels.append(workload)
                values.append(median(configs["adapter"]) / median(configs["direct"]) * 100 - 100)
        if not labels:
            return False
        axis.bar(labels, values)
        axis.axhline(3, color="red", linestyle="--", linewidth=1)
        axis.set_ylabel("Median latency overhead (%)")
        axis.tick_params(axis="x", rotation=25)
        return True

    _save(
        plots / "02_adapter_overhead_by_workload.png", "Adapter overhead by workload", overhead_plot
    )

    def activation_plot(axis: Axes) -> bool:
        ranked = sorted(
            (
                (
                    f"L{row.get('layer_id')} E{row.get('expert_id')}",
                    _number(row.get("activation_count")),
                )
                for row in activation
            ),
            key=lambda item: item[1] or 0,
            reverse=True,
        )[:30]
        measured_ranked: list[tuple[str, float]] = [
            (name, value) for name, value in ranked if value is not None
        ]
        if not measured_ranked:
            return False
        axis.bar(range(len(measured_ranked)), [value for _, value in measured_ranked])
        axis.set_xticks(
            range(len(measured_ranked)),
            [name for name, _ in measured_ranked],
            rotation=80,
            fontsize=7,
        )
        axis.set_ylabel("Observed activations")
        return True

    _save(
        plots / "03_expert_activation_frequency.png", "Expert activation frequency", activation_plot
    )
    _save(
        plots / "04_expert_cache_hit_rate.png",
        "Expert cache hit rate",
        _bar(policies, key="policy", value="expert_hit_rate", ylabel="Held-out hit rate"),
    )
    _save(
        plots / "05_expert_tier_residency.png",
        "Expert tier residency",
        _bar(residency, key="tier", value="allocated_expert_bytes", ylabel="Expert bytes"),
    )
    _save(
        plots / "06_bytes_read_per_token.png",
        "Bytes read per token",
        _bar(policies, key="policy", value="bytes_read_per_token", ylabel="Bytes/token"),
    )

    def storage_compute(axis: Axes) -> bool:
        totals: dict[str, float] = {}
        for row in storage:
            duration = _number(row.get("duration_ms"))
            if duration is not None:
                totals[row.get("category", "storage")] = (
                    totals.get(row.get("category", "storage"), 0) + duration
                )
        if not totals:
            return False
        axis.bar(list(totals), list(totals.values()))
        axis.set_ylabel("Observed duration (ms)")
        return True

    _save(plots / "07_storage_vs_compute_time.png", "Storage versus compute time", storage_compute)
    _save(
        plots / "08_candidate_tuning_performance.png",
        "Fixed-replay candidate performance",
        _bar(
            tuning,
            key="candidate_id",
            value="median_decode_tokens_per_second",
            ylabel="Median decode tokens/s",
        ),
    )

    def predicted_measured(axis: Axes) -> bool:
        pairs: list[tuple[float, float]] = []
        for row in policies:
            left = _number(row.get("predicted_hit_rate"))
            right = _number(row.get("expert_hit_rate"))
            if left is not None and right is not None:
                pairs.append((left, right))
        if not pairs:
            return False
        axis.scatter([left for left, _ in pairs], [right for _, right in pairs])
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set_xlabel("Predicted placement value")
        axis.set_ylabel("Measured held-out hit rate")
        return True

    _save(
        plots / "09_predicted_vs_measured_placement.png",
        "Predicted versus measured placement value",
        predicted_measured,
    )
    _save(
        plots / "10_heldout_policy_comparison.png",
        "Held-out policy comparison",
        _bar(
            policies,
            key="policy",
            value="decode_tokens_per_second",
            ylabel="Measured decode tokens/s",
        ),
    )

    def working_set(axis: Axes) -> bool:
        # Keep one entry per unique physical expert.  Several experts can first
        # appear on the same router call, so deduplicating call indices alone
        # understates the working set.
        first_by_expert: dict[tuple[int, int], int] = {}
        for row in activation:
            value = _number(row.get("first_call_index"))
            layer = _number(row.get("layer_id"))
            expert = _number(row.get("expert_id"))
            if value is None or layer is None or expert is None:
                continue
            key = (int(layer), int(expert))
            first_by_expert[key] = min(int(value), first_by_expert.get(key, int(value)))
        first_calls = sorted(first_by_expert.values())
        if not first_calls:
            return False
        axis.step(first_calls, list(range(1, len(first_calls) + 1)), where="post")
        axis.set_xlabel("Route call index")
        axis.set_ylabel("Cumulative unique layer-experts")
        return True

    _save(
        plots / "11_expert_working_set_growth.png",
        "Expert working-set growth over tokens",
        working_set,
    )

    def phase_routes(axis: Axes) -> bool:
        totals: dict[str, float] = {}
        for row in activation:
            value = _number(row.get("activation_count"))
            if value is not None:
                totals[row.get("phase", "unknown")] = (
                    totals.get(row.get("phase", "unknown"), 0) + value
                )
        if not totals:
            return False
        axis.bar(list(totals), list(totals.values()))
        axis.set_ylabel("Expert selections")
        return True

    _save(
        plots / "12_prefill_vs_decode_routing.png",
        "Prefill versus decode routing characteristics",
        phase_routes,
    )
    return sorted(path.name for path in plots.glob("*.png"))


def build_bundle_readme(mode: str) -> str:
    return f"""# Experiment 009 evidence bundle

This `{mode}` bundle contains raw, machine-readable evidence for the pinned Colibri runtime.
Values missing from the runtime are represented as unavailable, never as zero. Fixture evidence
exercises real Colibri code but cannot support the official performance verdict.

Start with `verdict.json` and `report.md`; use the JSON/CSV/NDJSON files for audit and the bundled
`reproduce.ps1` command for reproduction.
"""


def build_report(context: dict[str, Any]) -> str:
    answers = context["answers"]
    lines = [
        "# Experiment 009: Colibri-Backed Adaptive Expert Runtime",
        "",
        f"**Verdict:** `{context['verdict']}`  ",
        f"**Run mode:** `{context['run_mode']}`  ",
        f"**Evidence classification:** `{context['evidence_class']}`",
        "",
        "## Answer",
        "",
        context["summary"],
        "",
        "## Acceptance gates",
        "",
        "| Gate | Status | Evidence | Reason |",
        "|---:|---|---|---|",
    ]
    for gate in context["gates"]:
        lines.append(
            f"| {gate['gate_id']} | {gate['status']} | {gate.get('evidence_class') or 'n/a'} | "
            f"{'<br>'.join(gate.get('reasons') or ['none'])} |"
        )
    lines.extend(["", "## Required questions", ""])
    questions = (
        "Was Colibri integrated as a real backend?",
        "Which Colibri commit was used?",
        "Were any upstream patches required?",
        "Did the adapter preserve exact tokens?",
        "What overhead did the adapter introduce?",
        "Which model families were detected?",
        "Which model was used for the official benchmark?",
        "Were real expert routes observed?",
        "How many experts were active?",
        "What lived in VRAM, RAM and storage?",
        "How many bytes were read per token?",
        "Did routing history improve held-out performance?",
        "Did prefetching hide transfer time or steal bandwidth?",
        "Did the swarm tuner select a different plan from Colibri?",
        "Did that plan produce a reverse-confirmed gain?",
        "Which settings were rejected?",
        "Was native quantization preserved?",
        "Is the microshard ABI valid?",
        "Which microshard operations remain unsupported?",
        "What exact backend control is now available that Experiment 008 lacked?",
        "Does this integration strengthen the path toward distributed Kimi K3 inference?",
        "What is the next smallest physical multi-node experiment enabled by this work?",
    )
    for index, question in enumerate(questions, 1):
        lines.extend([f"{index}. **{question}** {answers.get(str(index), 'Not measured.')} "])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in context.get("limitations", [])],
            "",
            "Distributed Kimi K3 inference was not implemented or claimed.",
            "",
        ]
    )
    return "\n".join(lines)
