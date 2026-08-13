"""Execute Experiment 018 coarse/fine models from physical evidence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.network import NetworkProfile
from swarm_inference.experiments.experiment_018.analysis import HISTORICAL_CURVE
from swarm_inference.experiments.experiment_018.fine_wavefront import FineWavefrontModel
from swarm_inference.experiments.experiment_018.wavefront import (
    MicrocellServiceProfile,
    WavefrontModel,
    partition_rows,
)

BLOCKS = (4, 7, 12, 16)
CHUNKS = (1, 2, 4, 8)
NETWORK_PROFILES = {
    "local_microcell": (0.25, 25.0),
    "fast_regional": (1.0, 10.0),
    "regional_wan_like": (5.0, 1.0),
    "adverse_public_wan": (20.0, 0.1),
}


def historical_by_block() -> dict[int, dict[str, float | int]]:
    return {
        block: {
            "accepted_rows": accepted,
            "target_pass_ms": target_ms,
            "oracle_tok_s_per_user": oracle,
        }
        for block, accepted, target_ms, oracle in HISTORICAL_CURVE
    }


def _zero_network(name: str) -> NetworkProfile:
    return NetworkProfile(
        name,
        rtt_ms=0.0,
        bandwidth_gbps=1e12,
        measured_loopback_base_ms=0.0,
        framing_bytes=0,
    )


def run_coarse_sweep(
    profiles: tuple[MicrocellServiceProfile, ...],
    trace_directory: Path,
    *,
    control_setup_ms: float,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, bool], Any]]:
    trace_directory.mkdir(parents=True, exist_ok=True)
    canonical = WavefrontModel(profiles, control_setup_ms=control_setup_ms)
    compute_only = WavefrontModel(
        profiles,
        internal_network=_zero_network("zero-internal"),
        coarse_network=_zero_network("zero-coarse"),
        control_setup_ms=0.0,
    )
    baseline = historical_by_block()
    rows: list[dict[str, Any]] = []
    retained: dict[tuple[int, int, bool], Any] = {}
    for block in BLOCKS:
        for chunk in CHUNKS:
            if chunk > block:
                continue
            for cache in (False, True):
                result = canonical.run(
                    block_candidates=block,
                    maximum_chunk_rows=chunk,
                    cache_enabled=cache,
                )
                zero = compute_only.run(
                    block_candidates=block,
                    maximum_chunk_rows=chunk,
                    cache_enabled=cache,
                )
                retained[(block, chunk, cache)] = result
                chunk_rows = partition_rows(block + 1, chunk)
                stage_means = [
                    sum(float(profile.compute_ms_by_rows[value]) for profile in profiles)
                    / len(profiles)
                    for value in chunk_rows
                ]
                ideal_compute = sum(stage_means) + (len(profiles) - 1) * max(
                    stage_means
                )
                fill_drain_loss = (len(profiles) - 1) * max(stage_means)
                stage_imbalance_loss = max(0.0, zero.total_ms - ideal_compute)
                communication_loss = max(
                    0.0, result.total_ms - zero.total_ms - control_setup_ms
                )
                serial = baseline[block]
                row = {
                    "result_class": (
                        "C_wavefront_attnres_cache" if cache else "B_wavefront_only"
                    ),
                    **result.summary(),
                    "serial_baseline_ms": serial["target_pass_ms"],
                    "serial_baseline_oracle_tok_s_per_user": serial[
                        "oracle_tok_s_per_user"
                    ],
                    "speedup_vs_corresponding_serial": float(
                        serial["target_pass_ms"]
                    )
                    / result.total_ms,
                    "serial_waits_per_accepted_token": (
                        result.serial_waits / result.accepted_rows
                    ),
                    "messages_per_accepted_token": (
                        result.messages / result.accepted_rows
                    ),
                    "worker_idle_ms_by_microcell": {
                        cell: result.total_ms * (1.0 - utilization)
                        for cell, utilization in result.stage_utilization.items()
                    },
                    "compute_only_wavefront_ms": zero.total_ms,
                    "ideal_balanced_compute_pipeline_ms": ideal_compute,
                    "loss_fill_drain_ms": fill_drain_loss,
                    "loss_stage_imbalance_ms": stage_imbalance_loss,
                    "loss_communication_ms": communication_loss,
                    "loss_explicit_state_barrier_ms": 0.0,
                    "loss_control_plane_ms": control_setup_ms,
                    "pipeline_efficiency_definition": (
                        "balanced pipeline latency using the mean measured stage "
                        "compute service for each actual chunk, excluding handoff, "
                        "divided by observed event-DAG makespan including communication"
                    ),
                    "evidence_class": "VALIDATED INDEPENDENT-RESOURCE MODEL + SHAPED NETWORK",
                }
                rows.append(row)
                trace_path = trace_directory / (
                    f"block-{block:02d}-chunk-{chunk:02d}-"
                    f"{'cache' if cache else 'current'}.json"
                )
                trace_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "experiment-018-event-trace-v1",
                            "claim_boundary": (
                                "deterministic independent-resource execution using "
                                "physical real-K3 service and shaped communication"
                            ),
                            "configuration": {
                                "block_candidates": block,
                                "accepted_rows": block + 1,
                                "maximum_chunk_rows": chunk,
                                "actual_chunk_rows": list(chunk_rows),
                                "attnres_cache": cache,
                            },
                            "result": row,
                            "event_run": result.event_run.to_json(),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
    return rows, retained


def coarse_network_sensitivity(
    profiles: tuple[MicrocellServiceProfile, ...],
    *,
    block: int,
    chunk: int,
    cache_enabled: bool,
    control_setup_ms: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rtt in (1, 5, 10, 20, 50, 100):
        for bandwidth in (0.1, 1.0, 10.0):
            model = WavefrontModel(
                profiles,
                coarse_network=NetworkProfile(
                    "coarse-sensitivity",
                    rtt_ms=float(rtt),
                    bandwidth_gbps=bandwidth,
                ),
                control_setup_ms=control_setup_ms,
            )
            result = model.run(
                block_candidates=block,
                maximum_chunk_rows=chunk,
                cache_enabled=cache_enabled,
            )
            rows.append(
                {
                    "coarse_rtt_ms": rtt,
                    "coarse_bandwidth_gbps": bandwidth,
                    "block_candidates": block,
                    "chunk_rows": chunk,
                    "cache_enabled": cache_enabled,
                    "target_pass_ms": result.total_ms,
                    "oracle_tok_s_per_user": result.oracle_tok_s_per_user,
                    "critical_path_communication_ms": (
                        result.critical_path_communication_ms
                    ),
                    "evidence_class": "SHAPED NETWORK SENSITIVITY",
                }
            )
    return rows


def shard_service_from_physical(
    microshard_receipt: dict[str, Any], split_degree: int
) -> dict[int, float]:
    result: dict[int, float] = {}
    for rows in (1, 2, 4, 8):
        record = next(
            row
            for row in microshard_receipt["results"]
            if int(row["split_degree"]) == split_degree
            and int(row["batch_rows"]) == rows
        )
        result[rows] = max(
            float(item["p50_ms"])
            for item in record["measurements"]["per_shard_wall"]
        )
    return result


def run_fine_sweep(
    profiles: tuple[MicrocellServiceProfile, ...],
    expert_components: dict[int, dict[int, float]],
    routes: dict[int, tuple[tuple[int, ...], ...]],
    route_weights: dict[int, tuple[tuple[float, ...], ...]],
    microshard_receipt: dict[str, Any],
    *,
    coarse_winner: dict[str, Any],
    control_setup_ms: float,
) -> tuple[list[dict[str, Any]], Any]:
    block = int(coarse_winner["block_candidates"])
    cache = bool(coarse_winner["cache_enabled"])
    configurations = {(block, chunk, 32) for chunk in CHUNKS if chunk <= block}
    winning_chunk = int(coarse_winner["chunk_size"])
    configurations.update((block, winning_chunk, degree) for degree in (8, 16))
    rows: list[dict[str, Any]] = []
    best_run = None
    for candidate_block, chunk, degree in sorted(configurations):
        coarse = WavefrontModel(profiles, control_setup_ms=control_setup_ms)
        model = FineWavefrontModel(
            profiles,
            expert_component_ms_by_cell_rows=expert_components,
            shard_service_ms_by_rows=shard_service_from_physical(
                microshard_receipt, degree
            ),
            routes_by_layer_position=routes,
            route_weights_by_layer_position=route_weights,
            split_degree=degree,
            coarse_model=coarse,
        )
        result = model.run(
            block_candidates=candidate_block,
            maximum_chunk_rows=chunk,
            cache_enabled=cache,
        )
        summary = {
            "result_class": "D_wavefront_fine_microshards",
            **result.summary(),
            "critical_path_waits_per_accepted_token": (
                result.critical_path_waits / result.accepted_rows
            ),
            "messages_per_accepted_token": (
                result.message_count / result.accepted_rows
            ),
            "worker_idle_ms_by_microcell": {
                cell: result.total_ms * (1.0 - utilization)
                for cell, utilization in result.stage_utilization.items()
            },
            "evidence_class": (
                "VALIDATED INDEPENDENT-RESOURCE MODEL from PHYSICAL native-MXFP4 "
                "shards + SHAPED NETWORK"
            ),
        }
        rows.append(summary)
        if best_run is None or result.oracle_tok_s_per_user > best_run.oracle_tok_s_per_user:
            best_run = result
    if best_run is None:
        raise RuntimeError("fine wavefront sweep emitted no configuration")
    return rows, best_run


def microshard_network_sensitivity(
    microshard_receipt: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sensitivity: list[dict[str, Any]] = []
    reductions: list[dict[str, Any]] = []
    for degree in (8, 16, 32):
        for rows in (1, 2, 4, 8):
            whole_record = next(
                item
                for item in microshard_receipt["results"]
                if int(item["split_degree"]) == 1
                and int(item["batch_rows"]) == rows
            )
            whole_wall = float(
                whole_record["measurements"]["sequential_one_gpu_wall"]["p50_ms"]
            )
            record = next(
                item
                for item in microshard_receipt["results"]
                if int(item["split_degree"]) == degree
                and int(item["batch_rows"]) == rows
            )
            compute = float(
                record["measurements"]["independent_resource_compute_ceiling_ms"]
            )
            input_bytes = int(record["input_payload_bytes_per_worker"])
            output_bytes = int(record["output_contribution_bytes_per_worker"])
            leaves = 16 * degree
            depth = math.ceil(math.log2(leaves))
            messages = leaves + leaves - 1
            total_payload = leaves * input_bytes + (leaves - 1) * output_bytes
            for profile_name, (rtt, bandwidth) in NETWORK_PROFILES.items():
                network = NetworkProfile(
                    profile_name,
                    rtt_ms=rtt,
                    bandwidth_gbps=bandwidth,
                )
                fanout = network.service_ms(input_bytes)
                reduction = depth * network.service_ms(output_bytes)
                critical = fanout + compute + reduction
                sensible = critical < whole_wall
                sensitivity.append(
                    {
                        "profile": profile_name,
                        "rtt_ms": rtt,
                        "bandwidth_gbps": bandwidth,
                        "split_degree": degree,
                        "batch_rows": rows,
                        "physical_shard_compute_p50_ms": compute,
                        "fanout_critical_ms": fanout,
                        "reduction_critical_ms": reduction,
                        "critical_path_ms": critical,
                        "whole_expert_physical_wall_p50_ms": whole_wall,
                        "latency_margin_vs_whole_expert_ms": whole_wall - critical,
                        "latency_sensible": sensible,
                        "messages": messages,
                        "total_payload_bytes": total_payload,
                        "useful_geographic_radius": (
                            profile_name if sensible else "none-at-this-shape"
                        ),
                        "evidence_class": "SHAPED NETWORK from PHYSICAL payload/service",
                    }
                )
            if rows == 1:
                reductions.append(
                    {
                        "split_degree": degree,
                        "selected_experts": 16,
                        "leaf_contributions": leaves,
                        "serial_reduction_depth": leaves - 1,
                        "hierarchical_reduction_depth": depth,
                        "messages": messages,
                        "fanout": leaves,
                        "payload_bytes": total_payload,
                        "physical_stable_fp32_reduction_p50_ms": float(
                            record["measurements"]["stable_fp32_reduction_wall"][
                                "p50_ms"
                            ]
                        ),
                        "route_weight_application": "before stable tree reduction",
                        "deterministic_order": "expert ID, then shard ID",
                    }
                )
    return sensitivity, reductions


def result_without_event(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.pop("event_run", None)
    return payload


__all__ = [
    "BLOCKS",
    "CHUNKS",
    "coarse_network_sensitivity",
    "historical_by_block",
    "microshard_network_sensitivity",
    "result_without_event",
    "run_coarse_sweep",
    "run_fine_sweep",
    "shard_service_from_physical",
]
