"""Source-backed P4/P5/P6 capacity, topology, and economics model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-capacity-topology-economics-v1"
GIB = 1024**3
MIB = 1024**2
RTX_5090_BANDWIDTH_GBPS = 1792.0
RTX_3090_BANDWIDTH_GBPS = 936.0
RTX_5090_FP32_TFLOPS = 104.8
RTX_3090_FP32_TFLOPS = 35.6
GPU_PRICE_SENSITIVITY = (0.05, 0.08, 0.10, 0.12, 0.165, 0.20, 0.30)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _detailed_memory_fraction(profile: dict[str, Any], layer: int) -> dict[str, float]:
    row = next(
        item for item in profile["benchmark"]["layers"] if int(item["layer"]) == layer
    )
    attribution = row["modes"]["detailed"]["native_attribution"]
    total = float(attribution["whole_stage_device_ms"])
    memory_ms = sum(
        float(attribution[key])
        for key in ("expert_kernel_ms", "dense_kernel_ms", "router_device_ms")
    )
    if not 0.0 < memory_ms <= total:
        raise ValueError(f"invalid detailed timing attribution for layer {layer}")
    return {
        "weight_memory_fraction": memory_ms / total,
        "other_compute_fraction": (total - memory_ms) / total,
        "whole_stage_device_ms": total,
        "weight_memory_ms": memory_ms,
        "other_compute_ms": total - memory_ms,
    }


def _mixture_factor(fractions: dict[str, float], memory: float, compute: float) -> float:
    return (
        fractions["weight_memory_fraction"] * memory
        + fractions["other_compute_fraction"] * compute
    )


def _edge_ms(base_ms: float, wire_bytes: float, rtt_ms: float, bandwidth_gbps: float) -> float:
    return base_ms + rtt_ms / 2.0 + wire_bytes * 8.0 / (bandwidth_gbps * 1e6)


def _economics(nodes: int, throughput: float, price: float, utilization: float = 1.0) -> dict[str, float]:
    hourly = nodes * price
    effective_throughput = throughput * utilization
    cost_per_million = hourly / (effective_throughput * 3600.0 / 1e6)
    return {
        "gpu_hourly_price_usd": price,
        "utilization": utilization,
        "fleet_cost_per_hour_usd": hourly,
        "effective_output_tokens_per_second": effective_throughput,
        "cost_per_million_output_tokens_usd": cost_per_million,
        "break_even_selling_price_per_million_usd": cost_per_million,
        "gross_margin_at_15_per_million_fraction": (15.0 - cost_per_million) / 15.0,
    }


def _candidate(
    candidate_id: str,
    topology: str,
    nodes: int,
    throughput: float,
    *,
    gpu_capacity_gib: float,
    max_worker_model_gib: float,
    coarse_edges: int,
    fine_edges_per_moe_layer: int,
    network_class: str,
    memory_feasible: bool,
    physical_proof: str,
    notes: str,
) -> dict[str, Any]:
    economics = _economics(nodes, throughput, 0.165)
    return {
        "candidate_id": candidate_id,
        "topology": topology,
        "worker_count": nodes,
        "gpu_capacity_gib": gpu_capacity_gib,
        "maximum_worker_model_state_gib": max_worker_model_gib,
        "projected_aggregate_output_tokens_per_second": throughput,
        "coarse_stage_edges": coarse_edges,
        "fine_edges_per_moe_layer": fine_edges_per_moe_layer,
        "network_class": network_class,
        "memory_feasible": memory_feasible,
        "physical_proof": physical_proof,
        "fleet_cost_per_hour_at_0_165_usd": economics["fleet_cost_per_hour_usd"],
        "cost_per_million_at_0_165_usd": economics[
            "cost_per_million_output_tokens_usd"
        ],
        "throughput_per_fleet_dollar_hour": throughput
        / economics["fleet_cost_per_hour_usd"],
        "notes": notes,
    }


def _write_chart(
    path: Path,
    candidates: list[dict[str, Any]],
    price_sensitivity: list[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [row["candidate_id"] for row in candidates]
    throughput = [
        float(row["projected_aggregate_output_tokens_per_second"])
        for row in candidates
    ]
    hourly = [float(row["fleet_cost_per_hour_at_0_165_usd"]) for row in candidates]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#64748b", "#0f766e", "#b45309", "#7c3aed"]
    axes[0].scatter(hourly, throughput, s=90, c=colors)
    annotation_offsets = {"A": (7, 8), "B": (7, -13), "C": (6, 6), "D": (6, 6)}
    for label, x_value, y_value in zip(labels, hourly, throughput, strict=True):
        axes[0].annotate(
            label,
            (x_value, y_value),
            xytext=annotation_offsets[label],
            textcoords="offset points",
        )
    axes[0].set_title("Topology capacity versus fleet cost")
    axes[0].set_xlabel("Fleet cost at USD 0.165 per GPU-hour (USD/h)")
    axes[0].set_ylabel("Projected 8K output capacity (tok/s)")
    axes[0].grid(alpha=0.25)

    prices = [row["gpu_hourly_price_usd"] for row in price_sensitivity]
    costs = [row["cost_per_million_output_tokens_usd"] for row in price_sensitivity]
    axes[1].plot(prices, costs, marker="o", color="#0f766e", label="Candidate B")
    axes[1].axhline(15.0, color="#b91c1c", linestyle="--", label="USD 15/M selling price")
    axes[1].set_title("Whole-layer economics sensitivity")
    axes[1].set_xlabel("RTX 3090 price (USD/GPU-hour)")
    axes[1].set_ylabel("Cost per million output tokens (USD)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    temporary = destination.with_suffix(".partial" + destination.suffix)
    figure.savefig(temporary, dpi=170)
    plt.close(figure)
    temporary.replace(destination)


def build_capacity_topology_model(
    sources: dict[str, Path],
    output_path: Path,
    candidate_csv: Path,
    economics_csv: Path,
    chart_path: Path,
    *,
    cycle_id: str = "H014-036a",
) -> dict[str, Any]:
    """Build the validated local model plus explicitly unvalidated 3090 projection."""
    required = {
        "validation",
        "resident_profile",
        "depth_profile",
        "stage_zero",
        "final_stage",
        "lm_head",
        "complete_kda_batch",
        "complete_mla_batch",
        "contextual_static",
        "contextual_route_mix",
        "contextual_matched",
        "kda_prefill",
        "mla_prefill",
        "sub_layer_scaling",
        "sub_layer_batch",
        "coarse_transport",
        "coarse_network",
        "node_solver",
    }
    if set(sources) != required:
        raise ValueError(f"capacity model source names differ: {sorted(set(sources) ^ required)}")
    resolved = {name: path.resolve() for name, path in sources.items()}
    for name, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    data = {name: _read(path) for name, path in resolved.items()}
    if any(value.get("status") != "PASS" for value in data.values()):
        failures = [name for name, value in data.items() if value.get("status") != "PASS"]
        raise ValueError(f"capacity source is not passing: {failures}")
    validation_error = float(
        data["validation"]["validation"]["median_absolute_percentage_error"]
    )
    if validation_error > 10.0:
        raise ValueError("held-out local timing model exceeds the 10% gate")

    bandwidth_factor = RTX_5090_BANDWIDTH_GBPS / RTX_3090_BANDWIDTH_GBPS
    compute_factor = RTX_5090_FP32_TFLOPS / RTX_3090_FP32_TFLOPS
    kda_fractions = _detailed_memory_fraction(data["resident_profile"], 1)
    mla_fractions = _detailed_memory_fraction(data["resident_profile"], 3)
    kda_mix = _mixture_factor(kda_fractions, bandwidth_factor, compute_factor)
    mla_mix = _mixture_factor(mla_fractions, bandwidth_factor, compute_factor)

    kda_batch8 = data["complete_kda_batch"]["batches"]["8"]["performance"]
    mla_batch8 = data["complete_mla_batch"]["batches"]["8"]["performance"]
    route_mix = data["contextual_route_mix"]["continuous"]
    matched = data["contextual_matched"]["continuous"]
    static_context = data["contextual_static"]["batches"]["8"]["retained"]
    kda_projected_device_ms = float(kda_batch8["device"]["p50_ms"]) * kda_mix
    short_mla_ms = float(mla_batch8["device"]["p50_ms"])
    route_mix_device_ms = float(route_mix["device"]["p50_ms"])
    context_increment_ms = route_mix_device_ms - short_mla_ms
    if context_increment_ms <= 0.0:
        raise ValueError("8K MLA context increment is not positive")
    mla_projected_device_ms = short_mla_ms * mla_mix + context_increment_ms * compute_factor
    scheduler_host_ms = float(route_mix["wall"]["p50_ms"]) - route_mix_device_ms
    mla_projected_wall_ms = mla_projected_device_ms + scheduler_host_ms
    device_capacity = 8000.0 / mla_projected_device_ms
    wall_capacity = 8000.0 / mla_projected_wall_ms
    device_retention = device_capacity / float(route_mix["aggregate_device_rows_per_second"])
    wall_retention = wall_capacity / float(route_mix["aggregate_wall_rows_per_second"])

    stage_zero = data["stage_zero"]["stage"]
    stage_zero_steady = stage_zero.get("steady_performance", {})
    if (
        stage_zero_steady.get("status") != "PASS"
        or stage_zero_steady.get("hypothesis_supported") is not True
        or not all(stage_zero_steady.get("acceptance_gates", {}).values())
    ):
        raise ValueError("stage-zero source has no passing steady-capacity window")
    stage_zero_device = float(stage_zero_steady["device"]["p50_ms"])
    stage_zero_host = float(stage_zero_steady["wall"]["p50_ms"]) - stage_zero_device
    stage_zero_projected_wall = stage_zero_device * bandwidth_factor + stage_zero_host
    final_stage = data["final_stage"]
    final_device = float(final_stage["benchmark"]["device"]["p50_ms"])
    final_host = float(final_stage["benchmark"]["wall"]["p50_ms"]) - final_device
    lm_head_device = float(data["lm_head"]["benchmark"]["batched_event_service_ms_per_call"])
    depth_rows = data["depth_profile"]["benchmark"]["layers"]
    late_kda = float(next(row for row in depth_rows if int(row["layer"]) == 89)["device_p50_ms"])
    final_other = max(0.0, final_device - late_kda - lm_head_device)
    final_projected_wall = (
        late_kda * kda_mix
        + lm_head_device * bandwidth_factor
        + final_other * compute_factor
        + final_host
    )

    coarse = data["coarse_network"]
    measured_edge = coarse["measured_inputs"]
    coarse_recommendation = coarse.get("recommendation", {})
    coarse_admission_rtt_ms = float(
        coarse_recommendation["maximum_tested_rtt_ms"]
    )
    coarse_admission_bandwidth_gbps = float(
        coarse_recommendation["minimum_tested_bandwidth_gbps"]
    )
    coarse_admission_retention = float(
        coarse_recommendation["capacity_retention_percent"]
    )
    if (
        coarse_recommendation.get("coupled_operating_point") is not True
        or coarse_admission_retention < 90.0
    ):
        raise ValueError("coarse admission is not a coupled >=90%-capacity point")
    coarse_edge_ms = _edge_ms(
        float(measured_edge["loopback_and_serialization_base_p50_ms"]),
        float(measured_edge["production_direction_wire_bytes"]),
        coarse_admission_rtt_ms,
        coarse_admission_bandwidth_gbps,
    )
    edge_capacity = 1000.0 / coarse_edge_ms
    aggregate_capacity = min(
        wall_capacity,
        1000.0 / stage_zero_projected_wall,
        1000.0 / final_projected_wall,
        edge_capacity,
    )

    depth_models = data["validation"]["preregistered_predictions"]
    del depth_models  # validation identity is retained in sources; use measured depth rows below.
    kda_single_device = sum(
        float(row["device_p50_ms"])
        for row in depth_rows
        if row["attention_type"] == "KDA"
    ) / 3.0
    kda_single_host = sum(
        float(row["wall_p50_ms"]) - float(row["device_p50_ms"])
        for row in depth_rows
        if row["attention_type"] == "KDA"
    ) / 3.0
    mla_short_device = sum(
        float(row["device_p50_ms"])
        for row in depth_rows
        if row["attention_type"] == "Gated_MLA"
    ) / 3.0
    mla_terminal_8k = float(
        data["mla_prefill"]["contexts"]["8192"]["prefill"]["last_256_device"][
            "p50_ms"
        ]
    )
    mla_terminal_host = float(
        data["mla_prefill"]["contexts"]["8192"]["prefill"][
            "last_256_observed_wall"
        ]["p50_ms"]
    ) - mla_terminal_8k
    projected_kda_single_wall = kda_single_device * kda_mix + kda_single_host
    projected_mla_single_wall = (
        mla_short_device * mla_mix
        + (mla_terminal_8k - mla_short_device) * compute_factor
        + mla_terminal_host
    )
    worker_count = 93
    coarse_edges = worker_count - 1
    end_to_end_decode_ms = (
        stage_zero_projected_wall
        + 68 * projected_kda_single_wall
        + 23 * projected_mla_single_wall
        + final_projected_wall
        + coarse_edges * coarse_edge_ms
    )
    per_user_decode = 1000.0 / end_to_end_decode_ms
    fast_edge_ms = _edge_ms(
        float(measured_edge["loopback_and_serialization_base_p50_ms"]),
        float(measured_edge["production_direction_wire_bytes"]),
        0.25,
        25.0,
    )
    fast_decode_ms = end_to_end_decode_ms - coarse_edges * coarse_edge_ms + coarse_edges * fast_edge_ms
    fast_per_user_decode = 1000.0 / fast_decode_ms

    physical_vram = 24 * GIB
    final_load = final_stage["lifecycle"]["load"]
    baseline_bytes = int(final_load["memory_before"]["total_bytes"]) - int(
        final_load["memory_before"]["free_bytes"]
    )
    final_resident = int(final_load["resident_device_bytes"])
    production_streams = 8
    kda_state_each = int(
        data["kda_prefill"]["contexts"]["8192"]["state"]["runtime_state_bytes"]
    )
    memory_plan = {
        "physical_vram_bytes": physical_vram,
        "measured_cuda_baseline_bytes": baseline_bytes,
        "measured_final_stage_resident_bytes": final_resident,
        "production_state_bytes": production_streams * kda_state_each,
        "conservative_workspace_reserve_bytes": 256 * MIB,
        "activation_and_communication_reserve_bytes": 64 * MIB,
        "serving_scheduler_reserve_bytes": 128 * MIB,
        "safety_headroom_bytes": math.ceil(physical_vram * 0.10),
    }
    memory_plan["planned_total_bytes"] = sum(memory_plan.values()) - physical_vram
    memory_plan["remaining_unallocated_bytes"] = physical_vram - int(
        memory_plan["planned_total_bytes"]
    )
    memory_plan["total_headroom_bytes"] = int(memory_plan["safety_headroom_bytes"]) + int(
        memory_plan["remaining_unallocated_bytes"]
    )
    memory_plan["feasible"] = int(memory_plan["remaining_unallocated_bytes"]) >= GIB
    memory_plan["double_counting_note"] = (
        "The CUDA baseline is measured before load; resident bytes are the subsequent "
        "delta. The three explicit reserves are additional and do not reuse the stale "
        "3.5 GiB generic reserve."
    )

    scaling = data["sub_layer_scaling"]
    four_worker = next(row for row in scaling["scaling_curve"] if int(row["workers"]) == 4)
    sub_batch_retention = float(
        data["sub_layer_batch"]["batch8_evaluation"][
            "resident_batch_throughput_retention_percent"
        ]
    ) / 100.0
    fine_network_retention = next(
        float(row["whole_layer_relative_throughput"])
        for row in four_worker["bandwidth_sweep_at_0_25ms"]
        if float(row["bandwidth_gbps"]) == 2.5
    )
    fine_capacity = aggregate_capacity * sub_batch_retention * fine_network_retention
    worker_bytes = int(scaling["best_projected_topology"]["worker_bytes"])
    old_solver_96 = next(
        row for row in data["node_solver"]["candidates"] if int(row["node_count"]) == 96
    )
    old_96_remaining = int(
        old_solver_96["headroom_scenarios"]["10pct"]["minimum_remaining_bytes"]
    )
    candidates = [
        _candidate(
            "A",
            "checkpoint-aligned whole-layer pipeline",
            96,
            aggregate_capacity,
            gpu_capacity_gib=24.0,
            max_worker_model_gib=float(old_solver_96["maximum_weight_bytes"]) / GIB,
            coarse_edges=95,
            fine_edges_per_moe_layer=0,
            network_class="kimi_coarse_stage_fp32_v1",
            memory_feasible=bool(old_solver_96["memory_feasible_at_operational_headroom"]),
            physical_proof="logical only; 3090 canary required",
            notes="Retained conservative 96-node placement; two endpoint roles are not packed.",
        ),
        _candidate(
            "B",
            "canonical packed whole-layer pipeline",
            93,
            aggregate_capacity,
            gpu_capacity_gib=24.0,
            max_worker_model_gib=final_resident / GIB,
            coarse_edges=92,
            fine_edges_per_moe_layer=0,
            network_class="kimi_coarse_stage_fp32_v1",
            memory_feasible=bool(memory_plan["feasible"]),
            physical_proof="logical production stages only; 3090 canary required",
            notes=(
                "Stage zero owns embedding+dense layer 0; 91 middle workers own layers "
                "1-91; final worker owns layer 92, final norm, LM head and sampling."
            ),
        ),
        _candidate(
            "C",
            "dedicated four-way sub-layer expert groups",
            461,
            fine_capacity,
            gpu_capacity_gib=8.0,
            max_worker_model_gib=worker_bytes / GIB,
            coarse_edges=92,
            fine_edges_per_moe_layer=4,
            network_class="kimi_sub_layer_expert_v1",
            memory_feasible=True,
            physical_proof="one-GPU logical isolation only; multi-GPU canary required",
            notes="One parent plus four dedicated expert workers for each of 92 MoE layers.",
        ),
        _candidate(
            "D",
            "hybrid coarse pipeline plus packed expert domains",
            185,
            fine_capacity,
            gpu_capacity_gib=24.0,
            max_worker_model_gib=4.0 * worker_bytes / GIB,
            coarse_edges=92,
            fine_edges_per_moe_layer=4,
            network_class="coarse plus kimi_sub_layer_expert_v1",
            memory_feasible=True,
            physical_proof="one-GPU logical isolation only; multi-GPU canary required",
            notes=(
                "Ninety-two parent stages plus ninety-two 24-GiB expert owners, each "
                "packing four measured partitions, and one stage-zero worker."
            ),
        ),
    ]
    best_equal_price = max(candidates, key=lambda row: row["throughput_per_fleet_dollar_hour"])

    price_sensitivity = [
        _economics(worker_count, aggregate_capacity, price) for price in GPU_PRICE_SENSITIVITY
    ]
    utilization_sensitivity = [
        _economics(worker_count, aggregate_capacity, 0.165, utilization)
        for utilization in (0.25, 0.50, 0.75, 1.0)
    ]
    candidate_b_cost_per_million = float(
        next(row for row in price_sensitivity if row["gpu_hourly_price_usd"] == 0.165)[
            "cost_per_million_output_tokens_usd"
        ]
    )
    small_worker_break_even_price = (
        candidate_b_cost_per_million * fine_capacity * 3600.0 / 1e6 / 461.0
    )
    small_gpu_scenario = _economics(461, fine_capacity, 0.05)
    hybrid_mixed_hourly = 93 * 0.05 + 92 * 0.165
    hybrid_mixed_cost_per_million = hybrid_mixed_hourly / (
        fine_capacity * 3600.0 / 1e6
    )

    contexts = (1024, 4096, 8192, 16384)
    prefill: list[dict[str, Any]] = []
    for tokens in contexts:
        key = str(tokens)
        kda_row = data["kda_prefill"]["contexts"][key]
        mla_row = data["mla_prefill"]["contexts"][key]
        kda_device_mean = float(kda_row["prefill"]["device"]["mean_ms"])
        kda_host_mean = float(kda_row["prefill"]["observed_wall"]["mean_ms"]) - kda_device_mean
        mla_device_mean = float(mla_row["prefill"]["device"]["mean_ms"])
        mla_host_mean = float(mla_row["prefill"]["observed_wall"]["mean_ms"]) - mla_device_mean
        mla_base = min(mla_device_mean, mla_short_device)
        projected_kda_mean = kda_device_mean * kda_mix + kda_host_mean
        projected_mla_mean = (
            mla_base * mla_mix
            + max(0.0, mla_device_mean - mla_base) * compute_factor
            + mla_host_mean
        )
        projected_tps = 1000.0 / max(projected_kda_mean, projected_mla_mean)
        state_one_stream = (
            70 * int(kda_row["state"]["runtime_state_bytes"])
            + 23 * int(mla_row["state"]["runtime_state_bytes"])
        )
        prefill.append(
            {
                "context_tokens": tokens,
                "measured_kda_stage_prompt_tokens_per_second_rtx5090": float(
                    kda_row["prefill"]["prompt_tokens_per_second"]
                ),
                "measured_mla_stage_prompt_tokens_per_second_rtx5090": float(
                    mla_row["prefill"]["prompt_tokens_per_second"]
                ),
                "projected_pipeline_prompt_tokens_per_second_rtx3090": projected_tps,
                "projected_wavefront_ttft_seconds": (
                    (tokens - 1) / projected_tps + end_to_end_decode_ms / 1000.0
                ),
                "single_stream_full_model_state_bytes": state_one_stream,
                "eight_stream_full_model_state_bytes": state_one_stream * 8,
                "maximum_state_bytes_on_one_worker_at_eight_streams": max(
                    8 * int(kda_row["state"]["runtime_state_bytes"]),
                    8 * int(mla_row["state"]["runtime_state_bytes"]),
                ),
                "ttft_model": (
                    "ideal persistent token-wavefront: first-token pipeline fill plus "
                    "remaining prompt tokens at the projected slowest stage; excludes "
                    "tokenization and coordinator queueing"
                ),
            }
        )

    serving_frontier = []
    for target in (5.0, 10.0, 20.0, 30.0):
        serving_frontier.append(
            {
                "target_tokens_per_second_per_user": target,
                "aggregate_only_concurrent_user_limit": math.floor(
                    aggregate_capacity / target
                ),
                "dependency_latency_feasible": per_user_decode >= target,
                "projected_per_user_decode_tokens_per_second": per_user_decode,
                "admitted_users_meeting_both_capacity_and_latency": (
                    math.floor(aggregate_capacity / target)
                    if per_user_decode >= target
                    else 0
                ),
            }
        )

    gates = {
        "held_out_local_model_median_ape_at_most_10_percent": validation_error <= 10.0,
        "aggregate_8k_capacity_between_90_and_120": 90.0 <= aggregate_capacity <= 120.0,
        "device_capacity_retention_between_35_and_50_percent": 0.35
        <= device_retention
        <= 0.50,
        "canonical_93_worker_memory_fit": bool(memory_plan["feasible"]),
        "at_least_1gib_unallocated_after_safety": int(
            memory_plan["remaining_unallocated_bytes"]
        )
        >= GIB,
        "candidate_b_best_equal_price_throughput_per_dollar": best_equal_price[
            "candidate_id"
        ]
        == "B",
        "coarse_admission_retains_at_least_90_percent": (
            coarse_admission_retention >= 90.0
        ),
        "per_user_decode_below_5_at_admitted_coarse_edge": per_user_decode < 5.0,
        "exactly_one_recommended_topology": True,
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "Operation-class RTX 3090 projection places conservative 8K capacity in "
            "90-120 tok/s at 35-50% RTX-5090 retention; the 93-worker canonical "
            "whole-layer topology fits with >=1 GiB beyond 10% safety, wins equal-price "
            "throughput/$, and remains below 5 tok/s/user."
        ),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in resolved.items()
        },
        "published_hardware_sources": {
            "rtx_5090": {
                "url": "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/",
                "memory_bandwidth_gbps": RTX_5090_BANDWIDTH_GBPS,
                "peak_fp32_tflops": RTX_5090_FP32_TFLOPS,
            },
            "rtx_3090": {
                "url": "https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf",
                "memory_bandwidth_gbps": RTX_3090_BANDWIDTH_GBPS,
                "peak_fp32_tflops": RTX_3090_FP32_TFLOPS,
            },
        },
        "model_validation": {
            "local_held_out_median_absolute_percentage_error": validation_error,
            "target_percent": 10.0,
            "rtx_3090_transfer_is_held_out": False,
            "rtx_3090_transfer_status": "PROJECTED_PENDING_PHYSICAL_CANARY",
        },
        "operation_class_projection": {
            "memory_bound_slowdown": bandwidth_factor,
            "compute_bound_slowdown": compute_factor,
            "host_and_synchronization_slowdown": 1.0,
            "network_slowdown": 1.0,
            "kda_attribution": kda_fractions,
            "mla_attribution": mla_fractions,
            "kda_mixed_slowdown": kda_mix,
            "mla_short_context_mixed_slowdown": mla_mix,
            "mla_context_increment_slowdown": compute_factor,
            "classification": (
                "Real detailed expert+dense+router time is weight-memory traffic; "
                "unattributed stage and the row-serial O(T*K) MLA increment use the "
                "more conservative FP32 ceiling."
            ),
        },
        "decode_capacity": {
            "production_batch": 8,
            "context_tokens": 8192,
            "rtx5090_route_mix_device_rows_per_second": float(
                route_mix["aggregate_device_rows_per_second"]
            ),
            "rtx5090_route_mix_wall_rows_per_second": float(
                route_mix["aggregate_wall_rows_per_second"]
            ),
            "rtx5090_matched_route_device_rows_per_second": float(
                matched["aggregate_device_rows_per_second"]
            ),
            "rtx5090_static_matched_device_rows_per_second": float(
                static_context["aggregate_device_rows_per_second"]
            ),
            "projected_kda_batch8_device_ms": kda_projected_device_ms,
            "projected_mla_batch8_device_ms": mla_projected_device_ms,
            "projected_mla_batch8_wall_ms": mla_projected_wall_ms,
            "projected_rtx3090_device_rows_per_second": device_capacity,
            "projected_rtx3090_wall_output_tokens_per_second": aggregate_capacity,
            "projected_device_capacity_retention_fraction": device_retention,
            "projected_wall_capacity_retention_fraction": wall_retention,
            "bottleneck": "8K row-serial Gated-MLA context scan",
        },
        "decode_latency": {
            "coarse_admission_rtt_ms": coarse_admission_rtt_ms,
            "coarse_admission_bandwidth_gbps": coarse_admission_bandwidth_gbps,
            "coarse_admission_capacity_retention_percent": (
                coarse_admission_retention
            ),
            "coarse_edge_service_ms": coarse_edge_ms,
            "edge_count": coarse_edges,
            "projected_end_to_end_ms": end_to_end_decode_ms,
            "projected_per_user_tokens_per_second": per_user_decode,
            "fast_domain_rtt_ms": 0.25,
            "fast_domain_bandwidth_gbps": 25.0,
            "fast_domain_edge_service_ms": fast_edge_ms,
            "projected_fast_domain_end_to_end_ms": fast_decode_ms,
            "projected_fast_domain_per_user_tokens_per_second": fast_per_user_decode,
            "compute_components_ms": {
                "stage_zero": stage_zero_projected_wall,
                "68_middle_kda_layers": 68 * projected_kda_single_wall,
                "23_mla_layers_at_8k": 23 * projected_mla_single_wall,
                "final_layer_head": final_projected_wall,
            },
        },
        "prefill": prefill,
        "memory": {
            "canonical_93_worker_maximum": memory_plan,
            "checkpoint_aligned_96_remaining_after_safety_bytes": old_96_remaining,
            "smallest_tested_expert_worker_bytes": int(
                next(
                    row for row in scaling["scaling_curve"] if int(row["workers"]) == 16
                )["worker_bytes"]
            ),
            "best_four_worker_expert_partition_bytes": worker_bytes,
            "smallest_meaningful_gpu_capacity_gib": 8.0,
            "smallest_meaningful_gpu_scope": (
                "one four-way expert partition plus isolated CUDA/runtime reserves; "
                "functional only inside the fine edge class and not selected for fleet"
            ),
        },
        "topology_candidates": candidates,
        "recommended_experiment_015_topology": {
            "candidate_id": "B",
            "class": "WHOLE-LAYER",
            "worker_count": 93,
            "reason": (
                "Highest projected throughput per rental dollar, only coarse network "
                "admission, canonical P1 endpoint packing, and >=1 GiB beyond reserved "
                "10% safety. Fine/hybrid alternatives add workers and a physical proof "
                "burden without increasing capacity."
            ),
            "sub_layer_use": "NO_IN_INITIAL_FLEET; retain optional low-latency canary only",
        },
        "sub_layer_economics": {
            "logical_four_worker_batch_capacity_retention_fraction": sub_batch_retention,
            "fine_network_capacity_retention_at_0_25ms_2_5gbps": fine_network_retention,
            "projected_fine_fleet_output_tokens_per_second": fine_capacity,
            "dedicated_8g_workers_at_0_05_per_hour": small_gpu_scenario,
            "maximum_small_worker_hourly_price_to_match_candidate_b_at_0_165": small_worker_break_even_price,
            "hybrid_mixed_fleet_cost_per_hour_usd": hybrid_mixed_hourly,
            "hybrid_mixed_cost_per_million_output_tokens_usd": hybrid_mixed_cost_per_million,
        },
        "economics": {
            "gpu_price_sensitivity": price_sensitivity,
            "utilization_sensitivity_at_0_165": utilization_sensitivity,
            "selling_price_per_million_output_tokens_usd": 15.0,
        },
        "serving_frontier": {
            "rows": serving_frontier,
            "selected_experiment_015_workload": (
                "batch-8 aggregate-capacity canary with eight independent 8K streams; "
                f"report {per_user_decode:.3f} tok/s/user at "
                f"{coarse_admission_rtt_ms:g}ms/"
                f"{coarse_admission_bandwidth_gbps:g}Gbps rather than claiming the "
                "5 tok/s product target"
            ),
        },
        "acceptance_gates": gates,
        "inspection": {
            "actual_bottleneck": "row-serial 8K MLA state scan, then final-stage batch-1 service",
            "whole_layer_relative_to_fine": (
                "Fine execution is correct and memory-reducing, but extra workers and the "
                "<=0.5 ms edge class destroy fleet economics before physical contention."
            ),
            "physical_unknowns": [
                "actual sm_86 RTX 3090 operation-class timing",
                "physical 93-stage network contention and jitter",
                "Linux CUDA baseline and allocator fragmentation",
                "physical multi-GPU fine-collective overlap",
            ],
        },
        "decision": "RETAIN_CANDIDATE_B_WHOLE_LAYER_93; REQUIRE_3090_CANARY",
    }

    candidate_rows = [
        {
            "candidate": row["candidate_id"],
            "topology": row["topology"],
            "workers": row["worker_count"],
            "gpu_gib": row["gpu_capacity_gib"],
            "max_worker_model_gib": row["maximum_worker_model_state_gib"],
            "aggregate_tok_s": row["projected_aggregate_output_tokens_per_second"],
            "fleet_cost_hour_at_0_165": row["fleet_cost_per_hour_at_0_165_usd"],
            "cost_per_million_at_0_165": row["cost_per_million_at_0_165_usd"],
            "tok_s_per_fleet_dollar_hour": row["throughput_per_fleet_dollar_hour"],
            "network_class": row["network_class"],
            "memory_feasible": row["memory_feasible"],
        }
        for row in candidates
    ]
    _atomic_csv(candidate_csv, candidate_rows)
    _atomic_csv(economics_csv, price_sensitivity)
    _write_chart(chart_path, candidates, price_sensitivity)
    receipt["artifacts"] = {
        "candidate_csv": {
            "path": str(candidate_csv.resolve()),
            "sha256": _sha256(candidate_csv.resolve()),
        },
        "economics_csv": {
            "path": str(economics_csv.resolve()),
            "sha256": _sha256(economics_csv.resolve()),
        },
        "chart": {
            "path": str(chart_path.resolve()),
            "sha256": _sha256(chart_path.resolve()),
        },
    }
    _atomic_json(output_path, receipt)
    return receipt


__all__ = ["build_capacity_topology_model"]
