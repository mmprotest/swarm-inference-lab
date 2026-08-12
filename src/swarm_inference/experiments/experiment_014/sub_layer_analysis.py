"""Scaling and physical-network replay for real Kimi expert microwork."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.cuda import _sha256_file

SCHEMA_VERSION = "experiment-014-k3-sub-layer-scaling-network-v1"
WORKER_COUNTS = (2, 4, 8, 16)
RTT_PROFILES_MS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
BANDWIDTH_PROFILES_GBPS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0)
NETWORK_SWEEP_BANDWIDTH_GBPS = 100.0
BANDWIDTH_SWEEP_RTT_MS = 0.25
USEFUL_CAPACITY_RETENTION = 0.90


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _transmission_ms(payload_bytes: int, bandwidth_gbps: float) -> float:
    return payload_bytes * 8.0 / (bandwidth_gbps * 1_000_000.0)


def _projection(
    row: dict[str, Any], *, rtt_ms: float, bandwidth_gbps: float
) -> dict[str, Any]:
    transmission_ms = _transmission_ms(row["critical_path_payload_bytes"], bandwidth_gbps)
    projected_ms = (
        row["parent_nonexpert_p50_ms"]
        + row["critical_worker_device_p50_ms"]
        + row["loopback_coordination_base_p50_ms"]
        + rtt_ms
        + transmission_ms
    )
    relative = row["reference_layer_p50_ms"] / projected_ms
    return {
        "rtt_ms": rtt_ms,
        "bandwidth_gbps": bandwidth_gbps,
        "transmission_ms": transmission_ms,
        "projected_distributed_layer_p50_ms": projected_ms,
        "whole_layer_relative_throughput": relative,
        "break_even_vs_resident": relative >= 1.0,
        "useful_90_percent_capacity": relative >= USEFUL_CAPACITY_RETENTION,
    }


def _write_csv(path: Path, matrix: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)
    os.replace(temporary, path)


def _render_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    workers = [row["workers"] for row in rows]
    navy = "#17324d"
    blue = "#2878b5"
    orange = "#e07a2f"
    green = "#2f8f6b"
    red = "#b5423c"
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.suptitle("Experiment 014 — Real Kimi Expert Microwork Scaling", fontsize=16)

    axes[0, 0].plot(
        workers,
        [row["reference_layer_p50_ms"] for row in rows],
        marker="o",
        color=navy,
        label="Resident reference",
    )
    axes[0, 0].plot(
        workers,
        [row["distributed_layer_p50_ms"] for row in rows],
        marker="o",
        color=orange,
        label="Same-GPU distributed",
    )
    axes[0, 0].set_title("Complete-layer wall p50")
    axes[0, 0].set_ylabel("Milliseconds")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        workers,
        [row["worker_gib"] for row in rows],
        marker="o",
        color=green,
    )
    axes[0, 1].set_title("Model state per microworker")
    axes[0, 1].set_ylabel("GiB")
    for row in rows:
        axes[0, 1].annotate(
            f"{row['worker_fraction_percent']:.1f}%",
            (row["workers"], row["worker_gib"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    axes[1, 0].plot(
        workers,
        [row["critical_worker_device_p50_ms"] for row in rows],
        marker="o",
        color=blue,
        label="Critical expert compute",
    )
    axes[1, 0].plot(
        workers,
        [row["loopback_coordination_base_p50_ms"] for row in rows],
        marker="o",
        color=red,
        label="Exposed coordination",
    )
    axes[1, 0].plot(
        workers,
        [row["expert_roundtrip_p50_ms"] for row in rows],
        marker="o",
        color=orange,
        label="Observed expert roundtrip",
    )
    axes[1, 0].set_title("Expert-path decomposition")
    axes[1, 0].set_ylabel("Milliseconds")
    axes[1, 0].legend(frameon=False)

    for row in rows:
        values = [
            point["whole_layer_relative_throughput"] * 100.0
            for point in row["rtt_sweep_100gbps"]
        ]
        axes[1, 1].plot(
            list(RTT_PROFILES_MS), values, marker="o", label=f"{row['workers']} workers"
        )
    axes[1, 1].axhline(90.0, color=red, linestyle="--", linewidth=1, label="90% useful gate")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title("Projected capacity vs RTT at 100 Gbps")
    axes[1, 1].set_xlabel("RTT (ms, log scale)")
    axes[1, 1].set_ylabel("Resident-layer throughput (%)")
    axes[1, 1].legend(frameon=False, ncol=2, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(workers if axis is not axes[1, 1] else list(RTT_PROFILES_MS))
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def analyze_sub_layer_scaling(
    receipts: dict[int, Path],
    output_path: Path,
    csv_path: Path,
    chart_path: Path,
    *,
    cycle_id: str = "H014-SUB-005",
) -> dict[str, Any]:
    if tuple(sorted(receipts)) != WORKER_COUNTS:
        raise ValueError(f"sub-layer scaling requires receipts for {WORKER_COUNTS}")
    raw: dict[int, dict[str, Any]] = {}
    for workers, path in receipts.items():
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS" or int(
            payload["configuration"]["worker_count"]
        ) != workers:
            raise ValueError(f"invalid {workers}-worker receipt")
        raw[workers] = payload

    rows: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for workers in WORKER_COUNTS:
        source = raw[workers]
        reference_ms = float(
            source["performance"]["single_gpu_complete_layer_wall"]["p50_ms"]
        )
        reference_expert_ms = float(
            source["single_gpu_reference"]["routed_expert_device"]["p50_ms"]
        )
        critical_ms = float(
            source["performance"]["worker_compute_critical_device"]["p50_ms"]
        )
        coordination_ms = float(
            source["performance"]["exposed_loopback_and_coordination"]["p50_ms"]
        )
        worker_bytes = int(source["memory"]["smallest_worker_tracked_bytes"])
        row: dict[str, Any] = {
            "workers": workers,
            "reference_layer_p50_ms": reference_ms,
            "reference_expert_p50_ms": reference_expert_ms,
            "parent_nonexpert_p50_ms": reference_ms - reference_expert_ms,
            "distributed_layer_p50_ms": float(
                source["performance"]["distributed_complete_layer_wall"]["p50_ms"]
            ),
            "distributed_layer_p95_ms": float(
                source["performance"]["distributed_complete_layer_wall"]["p95_ms"]
            ),
            "distributed_layer_p99_ms": float(
                source["performance"]["distributed_complete_layer_wall"]["p99_ms"]
            ),
            "distributed_layers_per_second": float(
                source["performance"]["aggregate_layers_per_second"]
            ),
            "same_gpu_relative_throughput": float(
                source["performance"]["whole_layer_relative_throughput"]
            ),
            "critical_worker_device_p50_ms": critical_ms,
            "critical_worker_device_p95_ms": float(
                source["performance"]["worker_compute_critical_device"]["p95_ms"]
            ),
            "critical_worker_device_p99_ms": float(
                source["performance"]["worker_compute_critical_device"]["p99_ms"]
            ),
            "loopback_coordination_base_p50_ms": coordination_ms,
            "expert_roundtrip_p50_ms": float(
                source["performance"]["loopback_expert_roundtrip"]["p50_ms"]
            ),
            "expert_roundtrip_p95_ms": float(
                source["performance"]["loopback_expert_roundtrip"]["p95_ms"]
            ),
            "expert_roundtrip_p99_ms": float(
                source["performance"]["loopback_expert_roundtrip"]["p99_ms"]
            ),
            "sub_layer_efficiency": float(
                source["performance"]["sub_layer_efficiency"]
            ),
            "worker_bytes": worker_bytes,
            "worker_gib": worker_bytes / 1024**3,
            "worker_fraction_percent": float(
                source["memory"]["workers"][0]["complete_layer_fraction_percent"]
            ),
            "mean_workers_contacted": float(
                source["communication"]["mean_workers_contacted"]
            ),
            "mean_total_transport_bytes": float(
                source["communication"]["mean_total_transport_bytes"]
            ),
            "critical_path_payload_bytes": int(
                source["communication"]["maximum_critical_path_payload_bytes"]
            ),
            "synchronization_points": int(
                source["communication"]["synchronization_points"]
            ),
            "maximum_selected_on_one_worker": int(
                source["routing_imbalance"]["hottest_worker_selected_count"]
            ),
        }
        row["rtt_sweep_100gbps"] = [
            _projection(
                row, rtt_ms=rtt_ms, bandwidth_gbps=NETWORK_SWEEP_BANDWIDTH_GBPS
            )
            for rtt_ms in RTT_PROFILES_MS
        ]
        row["bandwidth_sweep_at_0_25ms"] = [
            _projection(
                row,
                rtt_ms=BANDWIDTH_SWEEP_RTT_MS,
                bandwidth_gbps=bandwidth,
            )
            for bandwidth in BANDWIDTH_PROFILES_GBPS
        ]
        base_ms = (
            row["parent_nonexpert_p50_ms"]
            + critical_ms
            + coordination_ms
        )
        tx_100 = _transmission_ms(
            row["critical_path_payload_bytes"], NETWORK_SWEEP_BANDWIDTH_GBPS
        )
        row["maximum_break_even_rtt_ms_at_100gbps"] = max(
            0.0, reference_ms - base_ms - tx_100
        )
        row["maximum_useful_rtt_ms_at_100gbps"] = max(
            0.0, reference_ms / USEFUL_CAPACITY_RETENTION - base_ms - tx_100
        )
        useful_rtts = [
            point["rtt_ms"]
            for point in row["rtt_sweep_100gbps"]
            if point["useful_90_percent_capacity"]
        ]
        row["maximum_useful_tested_rtt_ms_at_100gbps"] = (
            max(useful_rtts) if useful_rtts else "LOOPBACK_ONLY"
        )
        bandwidth_allowance_ms = (
            reference_ms / USEFUL_CAPACITY_RETENTION
            - base_ms
            - BANDWIDTH_SWEEP_RTT_MS
        )
        row["minimum_useful_bandwidth_gbps_at_0_25ms"] = (
            row["critical_path_payload_bytes"]
            * 8.0
            / (bandwidth_allowance_ms * 1_000_000.0)
            if bandwidth_allowance_ms > 0
            else None
        )
        useful_bandwidths = [
            point["bandwidth_gbps"]
            for point in row["bandwidth_sweep_at_0_25ms"]
            if point["useful_90_percent_capacity"]
        ]
        row["minimum_useful_tested_bandwidth_gbps_at_0_25ms"] = (
            min(useful_bandwidths) if useful_bandwidths else None
        )
        for rtt_ms in RTT_PROFILES_MS:
            for bandwidth in BANDWIDTH_PROFILES_GBPS:
                projected = _projection(
                    row, rtt_ms=rtt_ms, bandwidth_gbps=bandwidth
                )
                matrix.append({"workers": workers, **projected})
        rows.append(row)

    useful_candidates = [
        row
        for row in rows
        if row["maximum_useful_tested_rtt_ms_at_100gbps"] != "LOOPBACK_ONLY"
    ]
    best = max(
        useful_candidates,
        key=lambda row: (
            row["maximum_useful_rtt_ms_at_100gbps"],
            -row["worker_bytes"],
        ),
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS",
        "hypothesis": {
            "prediction": (
                "Real measured microwork payloads will define a non-empty low-latency "
                "network region retaining at least 90% of resident complete-layer capacity."
            ),
            "useful_capacity_retention": USEFUL_CAPACITY_RETENTION,
        },
        "method": {
            "class": "COUNTERFACTUAL_PHYSICAL_LINK_REPLAY",
            "physical_multi_gpu_measurement": False,
            "formula": (
                "parent nonexpert p50 + measured critical worker device p50 + "
                "measured loopback coordination base + RTT + critical payload / bandwidth"
            ),
            "conservative_assumption": (
                "Measured loopback coordination is retained and physical link cost is added."
            ),
            "rtt_profiles_ms": list(RTT_PROFILES_MS),
            "bandwidth_profiles_gbps": list(BANDWIDTH_PROFILES_GBPS),
        },
        "sources": {
            str(workers): {
                "path": str(receipts[workers].resolve()),
                "sha256": _sha256_file(receipts[workers].resolve()),
            }
            for workers in WORKER_COUNTS
        },
        "scaling_curve": rows,
        "network_matrix_rows": len(matrix),
        "best_projected_topology": {
            "workers": best["workers"],
            "worker_bytes": best["worker_bytes"],
            "worker_fraction_percent": best["worker_fraction_percent"],
            "maximum_useful_rtt_ms_at_100gbps": best[
                "maximum_useful_rtt_ms_at_100gbps"
            ],
            "minimum_useful_bandwidth_gbps_at_0_25ms": best[
                "minimum_useful_bandwidth_gbps_at_0_25ms"
            ],
        },
        "hypothesis_supported": bool(useful_candidates),
        "inspection": {
            "actual_bottleneck": (
                "As worker count rises, selected-expert compute shrinks but fanout, "
                "framing, serialization, and collection grow; useful operation is "
                "therefore confined to a sub-millisecond low-latency domain."
            ),
            "maximum_viable_definition": "at least 90% resident complete-layer throughput",
            "strict_break_even_also_reported": True,
        },
        "decision": (
            "RETAIN_ONLY_INSIDE_FAST_EXPERT_DOMAINS"
            if useful_candidates
            else "SUB_LAYER_FUNCTIONAL_BUT_NOT_ECONOMIC"
        ),
        "artifacts": {
            "network_matrix_csv": str(csv_path.resolve()),
            "scaling_chart": str(chart_path.resolve()),
        },
    }
    _write_csv(csv_path, matrix)
    _render_chart(chart_path, rows)
    _atomic_json(output_path, result)
    return result
