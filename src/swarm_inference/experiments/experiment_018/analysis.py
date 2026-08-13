"""Evidence reconciliation and measured service construction for Experiment 018."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_015.network import NetworkProfile
from swarm_inference.experiments.experiment_018.wavefront import (
    MicrocellServiceProfile,
    current_boundary_payload,
)

HISTORICAL_CURVE = (
    (1, 2, 1337.7682243433117, 1.4950272876916082),
    (2, 3, 1732.3314842740583, 1.7317701763396423),
    (4, 5, 2221.4136161386446, 2.250819011675643),
    (7, 8, 2997.238528880436, 2.6691235692169797),
    (12, 13, 4291.965465036907, 3.028915331658712),
    (16, 17, 5484.922520799224, 3.099405677205971),
)
INTERNAL_NETWORK = NetworkProfile("internal", rtt_ms=0.25, bandwidth_gbps=25.0)
COARSE_NETWORK = NetworkProfile("coarse", rtt_ms=5.0, bandwidth_gbps=10.0)
INTERNAL_BOUNDARIES = 81
COARSE_BOUNDARIES = 11
DENSE_LAYER0_ROW_P50_MS = 14.1824
DENSE_LAYER0_RESIDENT_BYTES = 3_040_870_400
LM_HEAD_ROW_P50_MS = 1.6209
LM_HEAD_RESIDENT_BYTES = 1_175_060_480


def topology_transport_ms(rows: int) -> float:
    payload = current_boundary_payload(rows)
    return (
        INTERNAL_BOUNDARIES * INTERNAL_NETWORK.service_ms(payload)
        + COARSE_BOUNDARIES * COARSE_NETWORK.service_ms(payload)
    )


def historical_compute_by_rows() -> dict[int, float]:
    return {
        accepted: target_ms - topology_transport_ms(accepted)
        for _block, accepted, target_ms, _oracle in HISTORICAL_CURVE
    }


def _arm(layer: dict[str, Any], candidates: int) -> dict[str, Any]:
    return next(
        row
        for row in layer["performance_rows"]
        if row["arm"] == "C_verification_major_expert_batching"
        and int(row["candidate_count"]) == candidates
    )


def reconcile_baseline(
    fresh_path: Path,
    historical_physical_path: Path,
) -> dict[str, Any]:
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    old = json.loads(historical_physical_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for block, accepted, historical_ms, historical_oracle in HISTORICAL_CURVE:
        old_kda = float(_arm(old["layers"]["89"], block)["wall"]["p50_ms"])
        old_mla = float(_arm(old["layers"]["91"], block)["wall"]["p50_ms"])
        new_kda = float(_arm(fresh["layers"]["89"], block)["wall"]["p50_ms"])
        new_mla = float(_arm(fresh["layers"]["91"], block)["wall"]["p50_ms"])
        old_weighted = 69 * old_kda + 24 * old_mla
        new_weighted = 69 * new_kda + 24 * new_mla
        compute_scale = new_weighted / old_weighted
        network_ms = topology_transport_ms(accepted)
        reproduced_ms = network_ms + (historical_ms - network_ms) * compute_scale
        deviation = (reproduced_ms / historical_ms - 1.0) * 100.0
        rows.append(
            {
                "verification_block": block,
                "accepted_rows": accepted,
                "historical_target_pass_ms": historical_ms,
                "reproduced_target_pass_ms": reproduced_ms,
                "deviation_percent": deviation,
                "historical_oracle_tok_s_per_user": historical_oracle,
                "reproduced_oracle_tok_s_per_user": accepted * 1000.0
                / reproduced_ms,
                "representative_compute_scale": compute_scale,
                "routes_and_state_exact": all(
                    bool(
                        fresh["layers"][str(layer)]["correctness"]["expert_major"][
                            "pass"
                        ]
                    )
                    for layer in (89, 91)
                ),
            }
        )
    maximum = max(abs(float(row["deviation_percent"])) for row in rows)
    return {
        "status": "PASS" if maximum <= 3.0 else "FAIL",
        "gate_percent": 3.0,
        "maximum_absolute_deviation_percent": maximum,
        "historical_curve_retained_as_denominator": True,
        "rows": rows,
        "sources": {
            "fresh": fresh["sources"],
            "historical": old["sources"],
        },
    }


def _phase_fractions(service: dict[str, Any]) -> dict[str, float]:
    phase = service["phase_decomposition"]
    values = {
        name: float(result["wall"]["p50_ms"]) for name, result in phase.items()
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("physical layer phase profile has no positive time")
    return {name: value / total for name, value in values.items()}


def _phase_p50(phase: dict[str, Any], name: str) -> float:
    value = phase.get(name)
    return float(value["wall"]["p50_ms"]) if value else 0.0


def _sample_components(
    layer: dict[str, Any], rows: int
) -> tuple[float, float, float, dict[str, float]]:
    service = layer["service"][str(rows)]
    wall = float(service["wall"]["p50_ms"])
    cuda = min(wall, float(service["cuda"]["p50_ms"]))
    fractions = _phase_fractions(service)
    phases = {name: wall * fraction for name, fraction in fractions.items()}
    return wall, cuda, wall - cuda, phases


def build_measured_profiles(
    service_receipt_path: Path,
) -> tuple[
    tuple[MicrocellServiceProfile, ...],
    list[dict[str, Any]],
    dict[int, dict[int, float]],
    dict[str, Any],
]:
    receipt = json.loads(service_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("physical service receipt has not passed")
    plan = receipt["microcell_plan"]
    raw_by_cell: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    layer_rows: list[dict[str, Any]] = []
    for layer_id, layer in sorted(receipt["layers"].items(), key=lambda item: int(item[0])):
        for rows in (1, 2, 4, 8):
            service = layer["service"][str(rows)]
            phase = service["phase_decomposition"]
            layer_rows.append(
                {
                    "layer": int(layer_id),
                    "attention_type": layer["attention_type"],
                    "attnres_snapshot_layer": layer["attnres_snapshot_layer"],
                    "chunk_rows": rows,
                    "wall_p50_ms": service["wall"]["p50_ms"],
                    "wall_p90_ms": service["wall"]["p90_ms"],
                    "wall_p99_ms": service["wall"]["p99_ms"],
                    "cuda_p50_ms": service["cuda"]["p50_ms"],
                    "host_overhead_p50_ms": service["host_overhead"]["p50_ms"],
                    "resident_device_bytes": layer["load"]["resident_device_bytes"],
                    "load_wall_ms_excluded": layer["load"]["wall_ms"],
                    "prepare_wall_ms_excluded": layer["load"]["prepare_wall_ms"],
                    "weight_fingerprint": layer["load"]["weight_fingerprint"],
                    "routing_p50_ms": _phase_p50(phase, "router"),
                    "expert_service_p50_ms": (
                        _phase_p50(phase, "expert_dispatch")
                        + _phase_p50(phase, "routed_expert_compute")
                        + _phase_p50(phase, "expert_collection")
                        + _phase_p50(phase, "shared_expert")
                    ),
                    "attention_and_state_update_p50_ms": _phase_p50(
                        phase, "attention_and_pre_moe"
                    ),
                    "latent_moe_p50_ms": (
                        _phase_p50(phase, "latent_down")
                        + _phase_p50(phase, "scatter_reduction")
                        + _phase_p50(phase, "latent_up")
                    ),
                    "attnres_and_residual_p50_ms": _phase_p50(
                        phase, "residual"
                    ),
                    "output_preparation_p50_ms": _phase_p50(
                        phase, "boundary_d2h"
                    ),
                    "phase_isolation_note": (
                        "KDA/MLA state update is co-measured with attention; AttnRes is "
                        "co-measured with residual because the inherited physical runtime "
                        "does not expose narrower synchronization-free phase boundaries"
                    ),
                    "phase_decomposition": phase,
                }
            )

    for row in plan:
        cell = int(row["microcell_id"])
        kda = receipt["layers"][str(row["representative_kda_layer"])]
        mla = receipt["layers"][str(row["representative_mla_layer"])]
        kda_moe_count = int(row["kda_layer_count"]) - int(row["dense_layer_count"])
        mla_count = int(row["mla_layer_count"])
        for rows in (1, 2, 4, 8):
            k_wall, k_cuda, k_host, k_phases = _sample_components(kda, rows)
            m_wall, m_cuda, m_host, m_phases = _sample_components(mla, rows)
            dense_wall = int(row["dense_layer_count"]) * DENSE_LAYER0_ROW_P50_MS * rows
            endpoint_wall = int(row["endpoint_layer_count"]) * LM_HEAD_ROW_P50_MS * rows
            wall = kda_moe_count * k_wall + mla_count * m_wall + dense_wall + endpoint_wall
            cuda = kda_moe_count * k_cuda + mla_count * m_cuda
            host = kda_moe_count * k_host + mla_count * m_host + dense_wall + endpoint_wall
            phases: dict[str, float] = defaultdict(float)
            for name, value in k_phases.items():
                phases[name] += kda_moe_count * value
            for name, value in m_phases.items():
                phases[name] += mla_count * value
            if dense_wall:
                phases["dense_layer_0"] += dense_wall
            if endpoint_wall:
                phases["endpoint_lm_head"] += endpoint_wall
            raw_by_cell[cell][rows] = {
                "wall": wall,
                "cuda": cuda,
                "host": host,
                "phases": dict(phases),
                "expert": phases.get("routed_expert_compute", 0.0),
            }

    raw_totals = {
        rows: sum(raw_by_cell[cell][rows]["wall"] for cell in range(12))
        for rows in (1, 2, 4, 8)
    }
    historical = historical_compute_by_rows()
    target_compute = {
        1: historical[2] * raw_totals[1] / raw_totals[2],
        2: historical[2],
        4: historical[3] + (historical[5] - historical[3]) / 2.0,
        8: historical[8],
    }
    normalization = {
        rows: target_compute[rows] / raw_totals[rows] for rows in (1, 2, 4, 8)
    }
    profiles: list[MicrocellServiceProfile] = []
    expert_components: dict[int, dict[int, float]] = defaultdict(dict)
    methodology_rows: list[dict[str, Any]] = []
    for row in plan:
        cell = int(row["microcell_id"])
        compute: dict[int, float] = {}
        cuda: dict[int, float] = {}
        host: dict[int, float] = {}
        phases: dict[int, dict[str, float]] = {}
        for rows in (1, 2, 4, 8):
            scale = normalization[rows]
            raw = raw_by_cell[cell][rows]
            compute[rows] = float(raw["wall"]) * scale
            cuda[rows] = float(raw["cuda"]) * scale
            host[rows] = float(raw["host"]) * scale
            phases[rows] = {
                name: value * scale for name, value in raw["phases"].items()
            }
            expert_components[cell][rows] = float(raw["expert"]) * scale
            methodology_rows.append(
                {
                    "microcell_id": cell,
                    "chunk_rows": rows,
                    "raw_measured_composition_ms": raw["wall"],
                    "historical_compute_target_ms": target_compute[rows],
                    "global_normalization_factor": scale,
                    "normalized_cell_service_ms": compute[rows],
                    "expert_component_ms": expert_components[cell][rows],
                }
            )
        phase_totals: dict[str, float] = defaultdict(float)
        for values in phases.values():
            for name, value in values.items():
                phase_totals[name] += value
        bottleneck = max(phase_totals, key=phase_totals.get)
        kda_resident = int(
            receipt["layers"][str(row["representative_kda_layer"])]["load"]
            ["resident_device_bytes"]
        )
        mla_resident = int(
            receipt["layers"][str(row["representative_mla_layer"])]["load"]
            ["resident_device_bytes"]
        )
        resident = (
            (int(row["kda_layer_count"]) - int(row["dense_layer_count"]))
            * kda_resident
            + int(row["mla_layer_count"]) * mla_resident
            + int(row["dense_layer_count"]) * DENSE_LAYER0_RESIDENT_BYTES
            + int(row["endpoint_layer_count"]) * LM_HEAD_RESIDENT_BYTES
        )
        profiles.append(
            MicrocellServiceProfile(
                microcell_id=cell,
                layer_start=int(row["layer_start"]),
                layer_end=int(row["layer_end"]),
                compute_ms_by_rows=compute,
                cuda_ms_by_rows=cuda,
                host_overhead_ms_by_rows=host,
                phase_ms_by_rows=phases,
                resident_bytes=resident,
                bottleneck_operator=bottleneck,
            )
        )
    methodology = {
        "method": (
            "fresh real-layer p50s set cell imbalance and chunk scaling; one global "
            "factor per chunk size anchors total compute to the immutable E016/E017 "
            "serial denominator after subtracting the inherited shaped topology"
        ),
        "historical_denominator_not_replaced": True,
        "raw_total_ms_by_rows": raw_totals,
        "target_compute_ms_by_rows": target_compute,
        "normalization_factor_by_rows": normalization,
        "edge_inputs": {
            "dense_layer_0_row_p50_ms": DENSE_LAYER0_ROW_P50_MS,
            "dense_layer_0_source": (
                "E014 h014-038aj2 persistent real-K3 stage-zero physical p50"
            ),
            "lm_head_row_p50_ms": LM_HEAD_ROW_P50_MS,
            "lm_head_source": "E014 h014-025al resident real-K3 LM-head physical p50",
        },
        "rows": methodology_rows,
    }
    return tuple(profiles), layer_rows, dict(expert_components), methodology


def stage_balance_rows(
    profiles: tuple[MicrocellServiceProfile, ...],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rows in (1, 2, 4, 8):
        values = [float(profile.compute_ms_by_rows[rows]) for profile in profiles]
        maximum = max(values)
        mean = statistics.fmean(values)
        results.append(
            {
                "chunk_rows": rows,
                "max_stage_time_ms": maximum,
                "mean_stage_time_ms": mean,
                "median_stage_time_ms": statistics.median(values),
                "coefficient_of_variation": statistics.pstdev(values) / mean,
                "max_over_mean": maximum / mean,
                "slowest_microcell": values.index(maximum),
                "steady_state_ceiling_rows_per_second": rows * 1000.0 / maximum,
                "service_ms_by_microcell": values,
            }
        )
    return results


def parse_oracle_routes(path: Path) -> dict[int, tuple[tuple[int, ...], ...]]:
    by_layer: dict[int, dict[int, tuple[int, ...]]] = defaultdict(dict)
    for expected_record, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        fields = raw.split()
        if len(fields) != 19:
            raise ValueError(f"invalid K3 route row: {raw[:80]!r}")
        if int(fields[0]) != expected_record:
            raise ValueError("oracle route records are not in execution order")
        layer = int(fields[2])
        position = len(by_layer[layer])
        experts = tuple(int(value.split(":", 1)[0]) for value in fields[3:])
        if len(experts) != 16 or len(set(experts)) != 16:
            raise ValueError("oracle route row is not exact top-16")
        by_layer[layer][position] = experts
    expected_layers = set(range(1, 93))
    if set(by_layer) != expected_layers:
        raise ValueError("oracle route trace does not cover K3 routed layers 1..92")
    result: dict[int, tuple[tuple[int, ...], ...]] = {}
    for layer, positions in by_layer.items():
        ordered_positions = sorted(positions)
        if ordered_positions != list(range(len(positions))):
            raise ValueError(f"layer {layer} route positions are not contiguous")
        result[layer] = tuple(positions[position] for position in ordered_positions)
    return result


def parse_oracle_route_weights(
    path: Path,
) -> dict[int, tuple[tuple[float, ...], ...]]:
    by_layer: dict[int, dict[int, tuple[float, ...]]] = defaultdict(dict)
    for expected_record, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines()
    ):
        fields = raw.split()
        if len(fields) != 19:
            raise ValueError(f"invalid K3 route row: {raw[:80]!r}")
        if int(fields[0]) != expected_record:
            raise ValueError("oracle route-weight records are not in execution order")
        layer = int(fields[2])
        position = len(by_layer[layer])
        weights = tuple(float(value.split(":", 1)[1]) for value in fields[3:])
        if len(weights) != 16 or any(not math.isfinite(value) for value in weights):
            raise ValueError("oracle route weights are not finite exact top-16 values")
        by_layer[layer][position] = weights
    if set(by_layer) != set(range(1, 93)):
        raise ValueError("oracle route weights do not cover K3 routed layers 1..92")
    return {
        layer: tuple(positions[position] for position in sorted(positions))
        for layer, positions in by_layer.items()
    }


def useful_parallelism_from_trace(run: EventRunLike) -> dict[str, float]:
    records = {record.task_id: record for record in run.records}
    total_compute = sum(
        record.duration_ms
        for record in run.records
        if record.kind in {"compute", "expert"}
    )
    critical_compute = sum(
        records[task_id].duration_ms
        for task_id in run.critical_path
        if records[task_id].kind in {"compute", "expert"}
    )
    return {
        "useful_parallelism": total_compute / max(critical_compute, 1e-12),
        "critical_path_fraction": run.makespan_ms / max(run.serial_sum_ms, 1e-12),
    }


class EventRunLike:
    records: tuple[Any, ...]
    critical_path: tuple[str, ...]
    makespan_ms: float
    serial_sum_ms: float


__all__ = [
    "COARSE_NETWORK",
    "HISTORICAL_CURVE",
    "INTERNAL_NETWORK",
    "build_measured_profiles",
    "historical_compute_by_rows",
    "parse_oracle_routes",
    "reconcile_baseline",
    "stage_balance_rows",
    "topology_transport_ms",
    "useful_parallelism_from_trace",
]
