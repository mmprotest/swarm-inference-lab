"""Coarse Kimi stage-edge network sensitivity from measured TCP payloads."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMA_VERSION = "experiment-014-k3-coarse-network-analysis-v2"
RTT_MS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
BANDWIDTH_GBPS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0)
USEFUL_CAPACITY_PERCENT = 90.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _transfer_ms(wire_bytes: float, bandwidth_gbps: float) -> float:
    return wire_bytes * 8.0 / (bandwidth_gbps * 1_000_000.0)


def _select_coupled_admission(
    rows: list[dict[str, Any]], *, preferred_rtt_ms: float
) -> dict[str, Any]:
    """Select the least-bandwidth tested row that is useful at one exact RTT."""
    viable = [
        row
        for row in rows
        if float(row["rtt_ms"]) == preferred_rtt_ms
        and bool(row["useful_90_percent"])
    ]
    if not viable:
        raise ValueError(
            f"no tested >=90%-capacity coarse edge exists at {preferred_rtt_ms:g} ms"
        )
    return min(viable, key=lambda row: float(row["bandwidth_gbps"]))


def _draw_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    stage0_ms: float,
    stage1_ms: float,
    base_ms: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.empty((len(RTT_MS), len(BANDWIDTH_GBPS)), dtype=np.float64)
    lookup = {
        (float(row["rtt_ms"]), float(row["bandwidth_gbps"])): float(
            row["capacity_retention_percent"]
        )
        for row in rows
    }
    for y, rtt in enumerate(RTT_MS):
        for x, bandwidth in enumerate(BANDWIDTH_GBPS):
            matrix[y, x] = lookup[(rtt, bandwidth)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    heat = axes[0].imshow(
        matrix,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=100,
        cmap="viridis",
    )
    axes[0].set_xticks(range(len(BANDWIDTH_GBPS)), [str(value) for value in BANDWIDTH_GBPS])
    axes[0].set_yticks(range(len(RTT_MS)), [str(value) for value in RTT_MS])
    axes[0].set_xlabel("Bandwidth (Gbps)")
    axes[0].set_ylabel("RTT (ms)")
    axes[0].set_title("Coarse-edge pipeline capacity retention")
    fig.colorbar(heat, ax=axes[0], label="Retention (%)")

    for bandwidth in (0.5, 1.0, 2.5, 5.0, 10.0, 100.0):
        values = [lookup[(rtt, bandwidth)] for rtt in RTT_MS]
        axes[1].plot(RTT_MS, values, marker="o", label=f"{bandwidth:g} Gbps")
    axes[1].axhline(USEFUL_CAPACITY_PERCENT, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 103)
    axes[1].set_xlabel("RTT (ms, log scale)")
    axes[1].set_ylabel("Capacity retention (%)")
    axes[1].set_title(
        f"Measured base {base_ms:.3f} ms; stages {stage0_ms:.3f}/{stage1_ms:.3f} ms"
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Experiment 014 — real Kimi coarse stage-edge sensitivity")
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, path)


def analyze_coarse_network(
    coarse_receipt_path: Path,
    fine_receipt_path: Path,
    output_path: Path,
    csv_path: Path,
    chart_path: Path,
    edge_class_path: Path,
    *,
    cycle_id: str = "H014-032d",
) -> dict[str, Any]:
    """Replay a measured one-way coarse frame across RTT/bandwidth profiles."""
    coarse = json.loads(coarse_receipt_path.read_text(encoding="utf-8"))
    fine = json.loads(fine_receipt_path.read_text(encoding="utf-8"))
    if coarse.get("status") != "PASS" or not coarse.get("hypothesis_supported"):
        raise ValueError("coarse network analysis requires the passing H014-032c receipt")
    if fine.get("status") != "PASS":
        raise ValueError("coarse/fine comparison requires passing sub-layer network analysis")

    timing = coarse["performance"]["timing"]
    stage0_ms = float(timing["stage0_device"]["p50_ms"])
    stage1_ms = float(timing["stage1_device"]["p50_ms"])
    bottleneck_ms = max(stage0_ms, stage1_ms)
    base_ms = float(timing["exposed_transport_and_serialization"]["p50_ms"])
    wire_bytes = float(coarse["performance"]["wire"]["mean_interstage_request_wire_bytes"])
    useful_edge_service_ms = bottleneck_ms / (USEFUL_CAPACITY_PERCENT / 100.0)

    rows: list[dict[str, Any]] = []
    for rtt in RTT_MS:
        for bandwidth in BANDWIDTH_GBPS:
            transfer_ms = _transfer_ms(wire_bytes, bandwidth)
            edge_service_ms = base_ms + rtt / 2.0 + transfer_ms
            pipeline_service_ms = max(bottleneck_ms, edge_service_ms)
            retention = 100.0 * bottleneck_ms / pipeline_service_ms
            rows.append(
                {
                    "edge_class": "kimi_coarse_stage_fp32_v1",
                    "rtt_ms": rtt,
                    "bandwidth_gbps": bandwidth,
                    "measured_loopback_and_serialization_base_ms": base_ms,
                    "one_way_propagation_ms": rtt / 2.0,
                    "wire_transfer_ms": transfer_ms,
                    "edge_service_ms": edge_service_ms,
                    "stage_bottleneck_ms": bottleneck_ms,
                    "pipeline_service_ms": pipeline_service_ms,
                    "capacity_retention_percent": retention,
                    "network_hidden_by_pipeline_ms": min(edge_service_ms, bottleneck_ms),
                    "network_exposed_to_pipeline_ms": max(0.0, edge_service_ms - bottleneck_ms),
                    "two_stage_sequential_latency_ms": stage0_ms + stage1_ms + edge_service_ms,
                    "useful_90_percent": retention >= USEFUL_CAPACITY_PERCENT,
                }
            )

    exact_max_rtt: dict[str, float] = {}
    tested_max_rtt: dict[str, float | None] = {}
    for bandwidth in BANDWIDTH_GBPS:
        transfer = _transfer_ms(wire_bytes, bandwidth)
        exact = max(0.0, 2.0 * (useful_edge_service_ms - base_ms - transfer))
        exact_max_rtt[str(bandwidth)] = exact
        viable = [
            float(row["rtt_ms"])
            for row in rows
            if float(row["bandwidth_gbps"]) == bandwidth and row["useful_90_percent"]
        ]
        tested_max_rtt[str(bandwidth)] = max(viable) if viable else None

    exact_min_bandwidth: dict[str, float | None] = {}
    tested_min_bandwidth: dict[str, float | None] = {}
    for rtt in RTT_MS:
        available_ms = useful_edge_service_ms - base_ms - rtt / 2.0
        exact = wire_bytes * 8.0 / (available_ms * 1_000_000.0) if available_ms > 0 else None
        exact_min_bandwidth[str(rtt)] = exact
        viable = [
            float(row["bandwidth_gbps"])
            for row in rows
            if float(row["rtt_ms"]) == rtt and row["useful_90_percent"]
        ]
        tested_min_bandwidth[str(rtt)] = min(viable) if viable else None

    recommended_rtt = 5.0
    recommended = _select_coupled_admission(rows, preferred_rtt_ms=recommended_rtt)
    recommended_bandwidth = float(recommended["bandwidth_gbps"])
    recommended_useful = bool(recommended["useful_90_percent"])
    fine_recommended = next(
        row for row in fine["scaling_curve"] if int(row["workers"]) == 4
    )
    fine_rtt = float(fine_recommended["maximum_useful_tested_rtt_ms_at_100gbps"])
    fine_bandwidth = float(
        fine_recommended["minimum_useful_tested_bandwidth_gbps_at_0_25ms"]
    )
    fine_exact_rtt = float(fine_recommended["maximum_useful_rtt_ms_at_100gbps"])
    fine_exact_bandwidth = float(
        fine_recommended["minimum_useful_bandwidth_gbps_at_0_25ms"]
    )
    useful_exists = any(bool(row["useful_90_percent"]) for row in rows)
    looser_than_fine = (
        exact_max_rtt["100.0"] > fine_exact_rtt
        and exact_min_bandwidth["0.25"] is not None
        and float(exact_min_bandwidth["0.25"]) < fine_exact_bandwidth
    )

    _write_csv(csv_path, rows)
    _draw_chart(
        chart_path,
        rows,
        stage0_ms=stage0_ms,
        stage1_ms=stage1_ms,
        base_ms=base_ms,
    )
    edge_class = {
        "schema_version": "swarm-network-edge-class-v1",
        "edge_class": "kimi_coarse_stage_fp32_v1",
        "decomposition": "coarse_stage_boundary",
        "representation": "float32[1,9,7168]",
        "activation_payload_bytes": int(
            coarse["performance"]["wire"]["activation_payload_bytes"]
        ),
        "measured_wire_bytes": wire_bytes,
        "minimum_capacity_retention_percent": USEFUL_CAPACITY_PERCENT,
        "admission": {
            "maximum_rtt_ms": recommended_rtt,
            "minimum_bandwidth_gbps": recommended_bandwidth,
            "coupled_operating_point": True,
            "modeled_capacity_retention_percent_at_boundary": recommended[
                "capacity_retention_percent"
            ],
        },
        "exact_100gbps_maximum_rtt_ms": exact_max_rtt["100.0"],
        "physical_evidence": "loopback only; RTT/bandwidth rows are counterfactual replay",
        "source_receipt": coarse_receipt_path.as_posix(),
        "source_receipt_sha256": _sha256(coarse_receipt_path),
    }
    _atomic_text(edge_class_path, json.dumps(edge_class, indent=2, sort_keys=True) + "\n")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": (
            "PASS"
            if useful_exists and looser_than_fine and recommended_useful
            else "FAIL"
        ),
        "hypothesis": (
            "The measured FP32 coarse frame has a practical >=90%-capacity region "
            "with materially looser latency requirements than fine expert microwork."
        ),
        "method": {
            "kind": "counterfactual physical-link replay from measured loopback payload/service",
            "formula": (
                "edge_ms = measured_loopback_serialization_base + RTT/2 + "
                "wire_bytes*8/(bandwidth_gbps*1e6); pipeline_ms=max(stage_bottleneck,edge_ms)"
            ),
            "not_a_physical_lan_claim": True,
            "capacity_floor_percent": USEFUL_CAPACITY_PERCENT,
        },
        "measured_inputs": {
            "stage0_device_p50_ms": stage0_ms,
            "stage1_device_p50_ms": stage1_ms,
            "stage_bottleneck_p50_ms": bottleneck_ms,
            "loopback_and_serialization_base_p50_ms": base_ms,
            "activation_payload_bytes": coarse["performance"]["wire"][
                "activation_payload_bytes"
            ],
            "production_direction_wire_bytes": wire_bytes,
        },
        "profiles": {
            "rtt_ms": list(RTT_MS),
            "bandwidth_gbps": list(BANDWIDTH_GBPS),
            "matrix_rows": len(rows),
        },
        "boundaries": {
            "exact_maximum_rtt_ms_by_bandwidth_gbps": exact_max_rtt,
            "tested_maximum_rtt_ms_by_bandwidth_gbps": tested_max_rtt,
            "exact_minimum_bandwidth_gbps_by_rtt_ms": exact_min_bandwidth,
            "tested_minimum_bandwidth_gbps_by_rtt_ms": tested_min_bandwidth,
        },
        "recommendation": {
            "edge_class": "kimi_coarse_stage_fp32_v1",
            "maximum_tested_rtt_ms": recommended_rtt,
            "minimum_tested_bandwidth_gbps": recommended_bandwidth,
            "coupled_operating_point": True,
            "capacity_retention_percent": recommended["capacity_retention_percent"],
            "two_stage_sequential_latency_ms": recommended[
                "two_stage_sequential_latency_ms"
            ],
            "use_only_inside_fast_domains": False,
        },
        "acceptance_gates": {
            "at_least_one_useful_tested_operating_point": useful_exists,
            "recommended_coupled_operating_point_retains_at_least_90_percent": (
                recommended_useful
            ),
            "coarse_network_envelope_materially_looser_than_fine": looser_than_fine,
        },
        "fine_comparison": {
            "fine_recommended_maximum_tested_rtt_ms": fine_rtt,
            "fine_recommended_minimum_tested_bandwidth_gbps": fine_bandwidth,
            "fine_exact_maximum_rtt_ms_at_100gbps": fine_exact_rtt,
            "fine_exact_minimum_bandwidth_gbps_at_0_25ms": fine_exact_bandwidth,
            "coarse_exact_maximum_rtt_ms_at_100gbps": exact_max_rtt["100.0"],
            "coarse_exact_minimum_bandwidth_gbps_at_0_25ms": exact_min_bandwidth[
                "0.25"
            ],
            "coarse_recommended_maximum_tested_rtt_ms": recommended_rtt,
            "coarse_recommended_minimum_tested_bandwidth_gbps": recommended_bandwidth,
            "coarse_rtt_envelope_multiple": recommended_rtt / fine_rtt,
            "coarse_is_materially_looser": looser_than_fine,
        },
        "inspection": {
            "actual_bottleneck": (
                "RTT above the coarse pipeline stage time; bandwidth becomes limiting "
                "below a few Gbps at multi-millisecond RTT"
            ),
            "format_decision": (
                f"FP32 passes the coupled {recommended_rtt:g} ms / "
                f"{recommended_bandwidth:g} Gbps capacity admission point. Boundary "
                "precision still requires a numerical test because lower bandwidth "
                "rentals and per-stream latency may benefit."
            ),
        },
        "decision": {
            "coarse_fp32_edge": "RETAIN",
            "next_hypothesis": (
                "Test FP16 and BF16 serialization of the exact stage-0 boundary through "
                "real stage-1 CUDA; retain a lower precision only if numerical gates pass."
            ),
        },
        "artifacts": {
            "matrix_csv": csv_path.as_posix(),
            "chart": chart_path.as_posix(),
            "edge_class": edge_class_path.as_posix(),
            "coarse_receipt": coarse_receipt_path.as_posix(),
            "coarse_receipt_sha256": _sha256(coarse_receipt_path),
            "fine_receipt": fine_receipt_path.as_posix(),
            "fine_receipt_sha256": _sha256(fine_receipt_path),
        },
    }
    _atomic_text(output_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
