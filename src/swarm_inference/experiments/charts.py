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


def generate_experiment_001_charts(
    *,
    chart_dir: Path,
    scaling_rows: list[dict[str, Any]],
    aggregate_points: list[dict[str, Any]],
    replica_rows: list[dict[str, Any]],
    transport_rows: list[dict[str, Any]],
    capacity_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    """Generate the additional figures required by Experiment 001."""

    chart_dir.mkdir(parents=True, exist_ok=True)
    if scaling_rows:
        primary_concurrency = max(int(row["concurrent_requests"]) for row in scaling_rows)
        primary = sorted(
            [row for row in scaling_rows if int(row["concurrent_requests"]) == primary_concurrency],
            key=lambda row: int(row["node_count"]),
        )
        ratios = [
            (
                float(primary[index]["throughput"])
                / max(float(primary[index - 1]["throughput"]), 1e-12)
            )
            for index in range(1, len(primary))
        ]
        labels = [
            f"{primary[index - 1]['node_count']}→{primary[index]['node_count']}"
            for index in range(1, len(primary))
        ]
        fig, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(labels, ratios)
        axis.axhline(1.5, color="red", linestyle="--", label="1.50 threshold")
        axis.set_ylabel("Median throughput ratio")
        axis.set_title(f"Scaling ratios at concurrency {primary_concurrency}")
        axis.legend()
        _save(fig, chart_dir / "scaling_ratios.png")
    else:
        _empty(chart_dir / "scaling_ratios.png", "Scaling ratios")

    if replica_rows:
        primary_points = {
            str(row["point_id"]) for row in replica_rows if "-c64-" in str(row["point_id"])
        }
        visible = [
            row for row in replica_rows if str(row["point_id"]) in primary_points
        ] or replica_rows
        fig, axis = plt.subplots(figsize=(10, 5))
        labels = [f"{row['point_id']}:S{row['stage_id']}:{row['worker_id']}" for row in visible]
        axis.bar(labels, [float(row["operation_share"]) for row in visible])
        axis.axhline(0.05, color="red", linestyle="--", label="Meaningful use")
        axis.set_ylabel("Share of stage operations")
        axis.set_title("Replica distribution")
        axis.tick_params(axis="x", rotation=90)
        axis.legend()
        _save(fig, chart_dir / "replica_distribution.png")
    else:
        _empty(chart_dir / "replica_distribution.png", "Replica distribution")

    if transport_rows:
        ordered = sorted(
            transport_rows,
            key=lambda row: (
                int(row.get("concurrent_request_count") or 0),
                int(row.get("node_count") or 0),
                int(row.get("repeat") or 0),
            ),
        )
        labels = [str(row["point_id"]) for row in ordered]
        components = [
            ("stage_execution_time_ms", "Stage execution"),
            ("serialisation_time_ms", "Serialisation"),
            ("deserialisation_time_ms", "Deserialisation"),
            ("stream_queue_time_ms", "Stream queue"),
            ("hop_transfer_time_ms", "Hop transfer"),
            ("admission_time_ms", "Admission"),
        ]
        fig, axis = plt.subplots(figsize=(10, 5))
        bottom = [0.0] * len(ordered)
        for field, label in components:
            values = [float(row.get(field) or 0.0) for row in ordered]
            axis.bar(labels, values, bottom=bottom, label=label)
            bottom = [previous + value for previous, value in zip(bottom, values, strict=True)]
        axis.set_ylabel("Cumulative measured milliseconds")
        axis.set_title("Latency and processing breakdown")
        axis.tick_params(axis="x", rotation=90)
        axis.legend(ncol=2)
        _save(fig, chart_dir / "latency_breakdown.png")

        fig, axis = plt.subplots(figsize=(10, 5))
        coordinator = [float(row.get("coordinator_activation_bytes") or 0) for row in ordered]
        peer = [float(row.get("worker_to_worker_activation_bytes") or 0) for row in ordered]
        positions = list(range(len(ordered)))
        axis.bar(
            [position - 0.2 for position in positions],
            coordinator,
            width=0.4,
            label="Coordinator-relayed activations",
        )
        axis.bar(
            [position + 0.2 for position in positions],
            peer,
            width=0.4,
            label="Worker-to-worker activations",
        )
        axis.set_xticks(positions, labels, rotation=90)
        axis.set_ylabel("Bytes")
        axis.set_title("Coordinator versus peer activation bytes")
        axis.legend()
        _save(fig, chart_dir / "coordinator_vs_peer_bytes.png")

        fig, axis = plt.subplots(figsize=(10, 5))
        reuse = [
            float(row.get("data_messages_sent") or 0)
            / max(float(row.get("peer_streams_created") or 0), 1.0)
            for row in ordered
        ]
        axis.bar(labels, reuse)
        axis.set_ylabel("Messages per created peer stream")
        axis.set_title("Persistent stream reuse")
        axis.tick_params(axis="x", rotation=90)
        _save(fig, chart_dir / "stream_reuse.png")
    else:
        _empty(chart_dir / "latency_breakdown.png", "Latency breakdown")
        _empty(
            chart_dir / "coordinator_vs_peer_bytes.png",
            "Coordinator versus peer bytes",
        )
        _empty(chart_dir / "stream_reuse.png", "Persistent stream reuse")

    if capacity_rows:
        ordered_capacity = sorted(
            capacity_rows,
            key=lambda row: (
                int(row.get("concurrent_request_count") or 0),
                int(row.get("node_count") or 0),
                int(row.get("repeat") or 0),
            ),
        )
        fig, axis = plt.subplots(figsize=(10, 5))
        labels = [str(row["point_id"]) for row in ordered_capacity]
        errors = [
            float(row.get("prediction_error_fraction") or 0.0) * 100 for row in ordered_capacity
        ]
        axis.bar(labels, errors)
        axis.axhline(25.0, color="red", linestyle="--", label="25% threshold")
        axis.set_ylabel("Absolute prediction error (%)")
        axis.set_title("Capacity prediction error")
        axis.tick_params(axis="x", rotation=90)
        axis.legend()
        _save(fig, chart_dir / "capacity_prediction_error.png")
    else:
        _empty(
            chart_dir / "capacity_prediction_error.png",
            "Capacity prediction error",
        )

    # The standard generator already writes stage_utilisation.png. Ensure the
    # Experiment 001 contract remains complete even when called independently.
    if not (chart_dir / "stage_utilisation.png").is_file():
        if stage_rows:
            fig, axis = plt.subplots(figsize=(8, 4.5))
            axis.bar(
                [f"{row.get('point_id', '')}:S{row['stage_id']}" for row in stage_rows],
                [float(row.get("utilisation", 0.0)) for row in stage_rows],
            )
            axis.set_ylabel("Utilisation")
            axis.set_title("Stage utilisation")
            axis.tick_params(axis="x", rotation=90)
            _save(fig, chart_dir / "stage_utilisation.png")
        else:
            _empty(chart_dir / "stage_utilisation.png", "Stage utilisation")
