"""Offline PNG chart generation for standard run artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _empty(path: Path, title: str, message: str = "No observations") -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    _save(fig, path)


def generate_charts(
    *,
    chart_dir: Path,
    scaling_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    stages: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    network: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    if scaling_rows:
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in scaling_rows:
            groups.setdefault(int(row["concurrent_requests"]), []).append(row)
        for concurrency, rows in sorted(groups.items()):
            ordered = sorted(rows, key=lambda row: int(row["node_count"]))
            axis.plot(
                [int(row["node_count"]) for row in ordered],
                [float(row["throughput"]) for row in ordered],
                marker="o",
                label=f"{concurrency} requests",
            )
        axis.set_xlabel("Workers")
        axis.set_ylabel("Aggregate verified output tokens/s")
        axis.legend()
        axis.grid(alpha=0.25)
        axis.set_title("Aggregate verified throughput")
        _save(fig, chart_dir / "aggregate_throughput.png")
    else:
        plt.close(fig)
        _empty(chart_dir / "aggregate_throughput.png", "Aggregate verified throughput")

    completed = [row for row in requests if row.get("verification_state") == "verified"]
    if completed:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            [str(row["request_id"]) for row in completed],
            [float(row["decode_tokens_s"]) for row in completed],
        )
        axis.set_xlabel("Request")
        axis.set_ylabel("Decode tokens/s")
        axis.set_title("Per-request throughput")
        axis.tick_params(axis="x", rotation=90)
        _save(fig, chart_dir / "per_request_throughput.png")
    else:
        _empty(chart_dir / "per_request_throughput.png", "Per-request throughput")

    if scaling_rows:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        ordered = sorted(
            scaling_rows,
            key=lambda row: (int(row["concurrent_requests"]), int(row["node_count"])),
        )
        axis.plot(
            range(len(ordered)),
            [float(row["homogeneous_scaling_efficiency"]) for row in ordered],
            marker="o",
            label="homogeneous",
        )
        axis.plot(
            range(len(ordered)),
            [float(row["capacity_normalised_efficiency"]) for row in ordered],
            marker="s",
            label="capacity-normalised",
        )
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_ylabel("Efficiency")
        axis.set_xlabel("Scaling observation")
        axis.set_title("Scaling efficiency")
        axis.legend()
        axis.grid(alpha=0.25)
        _save(fig, chart_dir / "scaling_efficiency.png")
    else:
        _empty(chart_dir / "scaling_efficiency.png", "Scaling efficiency")

    if stages:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            [f"{row.get('run_key', '')}:S{row['stage_id']}" for row in stages],
            [float(row["utilisation"]) for row in stages],
        )
        axis.axhline(0.7, color="red", linewidth=1, linestyle="--", label="70% criterion")
        axis.set_ylabel("Utilisation")
        axis.set_ylim(0, 1)
        axis.set_title("Stage utilisation")
        axis.tick_params(axis="x", rotation=90)
        axis.legend()
        _save(fig, chart_dir / "stage_utilisation.png")
    else:
        _empty(chart_dir / "stage_utilisation.png", "Stage utilisation")

    if workers:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            [str(row["worker_id"]) for row in workers],
            [int(row["queue_depth"]) for row in workers],
        )
        axis.set_ylabel("Final queue depth")
        axis.set_title("Bounded worker queues")
        axis.tick_params(axis="x", rotation=90)
        _save(fig, chart_dir / "queue_depth.png")
    else:
        _empty(chart_dir / "queue_depth.png", "Bounded worker queues")

    if network:
        sent_by_link: dict[str, int] = {}
        for row in network:
            key = f"{row['source']}→{row['destination']}"
            sent_by_link[key] = sent_by_link.get(key, 0) + int(row["payload_bytes"])
        fig, axis = plt.subplots(figsize=(8, 4.5))
        largest = sorted(sent_by_link.items(), key=lambda item: item[1], reverse=True)[:20]
        axis.bar([item[0] for item in largest], [item[1] for item in largest])
        axis.set_ylabel("Bytes")
        axis.set_title("Network traffic by directed link (top 20)")
        axis.tick_params(axis="x", rotation=90)
        _save(fig, chart_dir / "network_bytes.png")
    else:
        _empty(chart_dir / "network_bytes.png", "Network traffic")

    if failures:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        types: dict[str, int] = {}
        for row in failures:
            event_type = str(row.get("event_type", "unknown"))
            types[event_type] = types.get(event_type, 0) + 1
        axis.bar(list(types), list(types.values()))
        axis.set_ylabel("Events")
        axis.set_title("Failure and recovery behaviour")
        axis.tick_params(axis="x", rotation=45)
        _save(fig, chart_dir / "failure_recovery.png")
    else:
        _empty(
            chart_dir / "failure_recovery.png",
            "Failure and recovery behaviour",
            "No failures or recoveries observed",
        )
