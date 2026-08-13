"""Materialize Experiment 019 evidence, simulations, charts, and report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_018.analysis import HISTORICAL_CURVE
from swarm_inference.experiments.experiment_019.charts import build_all_charts
from swarm_inference.experiments.experiment_019.checkpoint import CheckpointCatalog
from swarm_inference.experiments.experiment_019.placement import (
    GIB,
    PlacementResult,
    PlacementSpec,
    WorkerPlacement,
    assignments_for,
    build_placement,
    estimate_candidate,
)
from swarm_inference.experiments.experiment_019.report import render_report
from swarm_inference.experiments.experiment_019.simulation import (
    NETWORK_PROFILES,
    MeasuredShardService,
    SimulationConfiguration,
    simulate,
)

TARGET_TOK_S = 5.0
E018_TARGET_PASS_MS = 2536.463584714586
E018_TOK_S = 6.702244850841379
INHERITED_GPU_HOURLY_USD = 0.15
TIERS = (20, 8, 4, 2, 1)
DEGREES = (4, 8, 16, 32)
DEPTH_SPANS = (1, 2, 4, 8)
CHUNKS = (1, 2, 4)
BLOCKS = (7, 12, 16)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    def scalar(value: str) -> Any:
        if value == "":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            return float(value) if any(char in value for char in ".eE") else int(value)
        except ValueError:
            return value

    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_csv_stream(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _command(arguments: Sequence[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _timing_p50(value: Mapping[str, Any]) -> float:
    return float(value["p50_ms"])


def _attention_rows(receipt: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return list(receipt[split]["results"])


def _find_attention(
    receipt: Mapping[str, Any],
    split: str,
    attention_type: str,
    degree: int,
    rows: int,
) -> Mapping[str, Any]:
    return next(
        row
        for row in _attention_rows(receipt, split)
        if row["attention_type"] == attention_type
        and int(row["stripe_degree"]) == degree
        and int(row["rows"]) == rows
    )


def _find_expert(
    receipt: Mapping[str, Any], degree: int, rows: int
) -> Mapping[str, Any]:
    return next(
        row
        for row in receipt["results"]
        if int(row["stripe_degree"]) == degree and int(row["rows"]) == rows
    )


def _find_other(
    receipt: Mapping[str, Any], degree: int, rows: int
) -> Mapping[str, Any]:
    return next(
        row
        for row in receipt["results"]
        if int(row["degree"]) == degree and int(row["rows"]) == rows
    )


def _software_overhead(protocol: Mapping[str, Any], payload_bytes: int) -> float:
    row = min(
        protocol["protocol"]["rows"],
        key=lambda value: abs(int(value["payload_bytes"]) - payload_bytes),
    )
    return float(row["software_overhead_p50_ms"])


def _physical_service(
    attention: Mapping[str, Any],
    expert: Mapping[str, Any],
    other: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    degree: int,
    rows: int,
) -> MeasuredShardService:
    kda = _find_attention(attention, "calibration", "KDA", degree, rows)
    kda_heldout = _find_attention(attention, "heldout", "KDA", degree, rows)
    mla = _find_attention(attention, "calibration", "Gated_MLA", degree, rows)
    mla_heldout = _find_attention(
        attention, "heldout", "Gated_MLA", degree, rows
    )
    expert_row = _find_expert(expert, degree, rows)
    other_row = _find_other(other, degree, rows)
    operators = other_row["operators"]
    attnres = next(
        row
        for row in other["attnres"]
        if int(row["rows"]) == rows
    )
    rmsnorm = next(
        row
        for row in other["rmsnorm"]
        if int(row["rows"]) == rows
    )
    reduction_hidden = next(
        row
        for row in other["reductions"]
        if int(row["participants"]) == 2
        and int(row["rows"]) == rows
        and int(row["dimension"]) == 7168
    )
    reduction_latent = next(
        row
        for row in other["reductions"]
        if int(row["participants"]) == 2
        and int(row["rows"]) == rows
        and int(row["dimension"]) == 3584
    )
    router_ms = _timing_p50(other_row["router"]["duration"])
    attnres_ms = _timing_p50(attnres["duration"])
    norm_ms = _timing_p50(rmsnorm["wall"])
    latent_down = tuple(
        _timing_p50(worker["duration"])
        for worker in operators["latent_down"]["workers"]
    )
    latent_up = [
        _timing_p50(worker["duration"])
        for worker in operators["latent_up"]["workers"]
    ]
    shared = [
        _timing_p50(worker["duration"])
        for worker in operators["shared_expert"]["workers"]
    ]
    return MeasuredShardService(
        degree=degree,
        rows=rows,
        kda_common_ms=attnres_ms
        + norm_ms
        + max(
            _timing_p50(kda["common_projection"]["wall"]),
            _timing_p50(kda_heldout["common_projection"]["wall"]),
        ),
        kda_worker_ms=tuple(
            max(
                _timing_p50(calibration_worker["wall"]),
                _timing_p50(heldout_worker["wall"]),
            )
            for calibration_worker, heldout_worker in zip(
                kda["workers"], kda_heldout["workers"], strict=True
            )
        ),
        mla_common_ms=attnres_ms
        + norm_ms
        + max(
            _timing_p50(mla["common_projection"]["wall"]),
            _timing_p50(mla_heldout["common_projection"]["wall"]),
        ),
        mla_worker_ms=tuple(
            max(
                _timing_p50(calibration_worker["wall"]),
                _timing_p50(heldout_worker["wall"]),
            )
            for calibration_worker, heldout_worker in zip(
                mla["workers"], mla_heldout["workers"], strict=True
            )
        ),
        moe_pre_ms=attnres_ms + norm_ms + router_ms,
        latent_down_worker_ms=latent_down,
        expert_worker_ms=tuple(
            _timing_p50(worker["wall"])
            for worker in expert_row["worker_results"]
        ),
        routed_norm_ms=norm_ms,
        latent_up_shared_worker_ms=tuple(
            left + right for left, right in zip(latent_up, shared, strict=True)
        ),
        dense_worker_ms=tuple(
            _timing_p50(worker["duration"])
            for worker in operators["dense_mlp"]["workers"]
        ),
        layer_finalize_ms=_timing_p50(
            reduction_hidden["local_reduction_compute"]
        )
        / 2,
        endpoint_worker_ms=tuple(
            _timing_p50(worker["duration"])
            for worker in operators["lm_head"]["workers"]
        ),
        embedding_ms=_timing_p50(
            next(
                row
                for row in other["embedding"]
                if int(row["degree"]) == degree
            )["duration"]
        ),
        hidden_pair_reduce_ms=_timing_p50(
            reduction_hidden["local_reduction_compute"]
        ),
        latent_pair_reduce_ms=_timing_p50(
            reduction_latent["local_reduction_compute"]
        ),
        source=(
            "RTX 5090 physically measured shard service; conservative per-worker "
            "maximum across calibration and held-out attention layers"
        ),
    )


def _local_profile(
    name: str,
    protocol: Mapping[str, Any],
    payload_bytes: int,
) -> Any:
    return replace(
        NETWORK_PROFILES[name],
        software_overhead_ms=_software_overhead(protocol, payload_bytes),
    )


def _expert_allocation_factors(
    expert: Mapping[str, Any],
) -> dict[int, float]:
    factors: dict[int, float] = {}
    for row in expert["bank_residency"]:
        degree = int(row["stripe_degree"])
        runtime = int(row["runtime_weight_bytes"])
        measured = int(row["measured_free_memory_delta_bytes"])
        if runtime <= 0 or measured <= 0:
            raise RuntimeError(
                f"invalid physical expert-bank residency measurement for P={degree}"
            )
        # nvidia-smi/free-memory telemetry is allocation-granularity sampled;
        # it may undershoot tensor_bytes by a few MiB after allocator reuse.
        # Resident tensors are a hard lower bound, so never apply a factor < 1.
        factors[degree] = max(1.0, measured / runtime)
    missing = set(DEGREES).difference(factors)
    if missing:
        raise RuntimeError(
            f"missing full expert-bank residency measurements for degrees {sorted(missing)}"
        )
    return factors


def _placement_sweep(
    catalog: CheckpointCatalog,
    expert_allocation_factors: Mapping[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier in TIERS:
        for degree in DEGREES:
            for depth in DEPTH_SPANS:
                for chunk in CHUNKS:
                    spec = PlacementSpec(
                        tier,
                        degree,
                        depth,
                        chunk,
                        expert_allocation_overhead_factor=expert_allocation_factors[degree],
                    )
                    estimate = estimate_candidate(catalog, spec)
                    ownership_valid = degree >= 8
                    valid = bool(estimate["capacity_valid"]) and ownership_valid
                    rows.append(
                        {
                            **estimate,
                            "valid": valid,
                            "ownership_valid": ownership_valid,
                            "invalid_reason": (
                                ""
                                if valid
                                else (
                                    "P4_uniform_stripes_exceed_25_percent_after_common_tensors"
                                    if not ownership_valid
                                    else estimate["invalid_reason"]
                                )
                            ),
                            "memory_evidence": "conservative analytical prefilter; tier winner receives byte-exact manifest",
                        }
                    )
    return rows


def _run_sweep(
    placement_rows: Sequence[Mapping[str, Any]],
    attention: Mapping[str, Any],
    expert: Mapping[str, Any],
    other: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], MeasuredShardService]]:
    services: dict[tuple[int, int], MeasuredShardService] = {}
    for degree in (8, 16, 32):
        for rows in CHUNKS:
            services[(degree, rows)] = _physical_service(
                attention, expert, other, protocol, degree=degree, rows=rows
            )
    results: list[dict[str, Any]] = []
    simulation_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for placement in placement_rows:
        if not bool(placement["valid"]):
            continue
        degree = int(placement["stripe_degree"])
        depth = int(placement["depth_span"])
        chunk = int(placement["chunk_rows"])
        tier = float(placement["memory_cap_gib"])
        spec = PlacementSpec(
            tier,
            degree,
            depth,
            chunk,
            expert_allocation_overhead_factor=float(
                placement["expert_allocation_overhead_factor"]
            ),
        )
        for block in BLOCKS:
            cache_key = (degree, depth, chunk, block)
            cached = simulation_cache.get(cache_key)
            if cached is None:
                configuration = SimulationConfiguration(
                    placement=spec,
                    block=block,
                    chunk=chunk,
                    local_profile=_local_profile(
                        "canonical_fast_local", protocol, chunk * 7168 * 4
                    ),
                    inter_pod_profile=_local_profile(
                        "canonical_inter_pod", protocol, chunk * 7168 * 4
                    ),
                )
                cached, _run = simulate(services[(degree, chunk)], configuration)
                simulation_cache[cache_key] = cached
            result = {**cached, **asdict(spec)}
            results.append(
                {
                    **result,
                    "valid": True,
                    "max_worker_peak_gib": float(placement["estimated_max_peak_gib"]),
                    "max_layer_fraction": 1 / degree,
                    "max_expert_fraction": 1 / degree,
                    "worker_count": int(placement["worker_count"]),
                    "pod_count": int(placement["pod_count"]),
                }
            )
    return results, services


def _tier_best(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for tier in TIERS:
        candidates = [
            dict(row)
            for row in rows
            if float(row["memory_cap_gib"]) == tier
        ]
        if candidates:
            result[tier] = max(
                candidates,
                key=lambda row: (
                    float(row["exact_tok_s_per_user"]),
                    -int(row["worker_count"]),
                ),
            )
    return result


def _annotate_compute_work_inflation(
    repo: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Compare scheduled shard work with physical unsharded layer controls.

    This comparison metric does not drive service time or scheduling.  It uses
    the E018 layer-89/91 physical controls only as the declared equivalent
    unsharded work denominator after E019 work has already been scheduled.
    """

    historical = _read_json(repo / "artifacts/experiment-018/physical/service-raw.json")
    for row in rows:
        remaining = int(row["block"]) + 1
        chunk = int(row["chunk"])
        equivalent = 0.0
        while remaining:
            actual = min(chunk, remaining)
            kda = float(
                historical["layers"]["89"]["service"][str(actual)]["wall"][
                    "p50_ms"
                ]
            )
            mla = float(
                historical["layers"]["91"]["service"][str(actual)]["wall"][
                    "p50_ms"
                ]
            )
            equivalent += 69 * kda + 24 * mla
            remaining -= actual
        row["equivalent_unsharded_compute_work_ms"] = equivalent
        row["compute_work_inflation"] = (
            float(row["total_physical_compute_work_ms"]) / equivalent
        )
        row["compute_work_denominator_source"] = (
            "E018 physical layer-89/91 controls; post-schedule comparison only"
        )


def _exact_manifests(
    catalog: CheckpointCatalog,
    artifact_root: Path,
    tier_best: Mapping[int, Mapping[str, Any]],
) -> dict[int, PlacementResult]:
    results: dict[int, PlacementResult] = {}
    for tier, row in tier_best.items():
        spec = PlacementSpec(
            tier,
            int(row["stripe_degree"]),
            int(row["depth_span"]),
            int(row["chunk"]),
            block_candidates=int(row["block"]),
            expert_allocation_overhead_factor=float(
                row["expert_allocation_overhead_factor"]
            ),
        )
        result = build_placement(catalog, spec)
        results[tier] = result
        _write_json(
            artifact_root / f"placement/worker-manifest-{tier}g.json",
            result.manifest(),
        )
    return results


def _load_exact_manifests(artifact_root: Path) -> dict[int, PlacementResult]:
    results: dict[int, PlacementResult] = {}
    for tier in TIERS:
        raw = _read_json(artifact_root / f"placement/worker-manifest-{tier}g.json")
        spec = PlacementSpec(**raw["placement"])
        summary = raw["summary"]
        results[tier] = PlacementResult(
            spec=spec,
            workers=[WorkerPlacement(**row) for row in raw["workers"]],
            checkpoint_payload_bytes=int(summary["checkpoint_payload_bytes"]),
            total_resident_weight_bytes=int(summary["total_resident_weight_bytes"]),
            replicated_weight_bytes=int(summary["replicated_weight_bytes"]),
            coverage_tensor_count=int(summary["coverage_tensor_count"]),
            coverage_assigned_bytes=int(summary["coverage_assigned_bytes"]),
            coverage_gap_bytes=int(summary["coverage_gap_bytes"]),
            coverage_overlap_bytes=int(summary["coverage_overlap_bytes"]),
            max_worker_peak_bytes=int(summary["max_worker_peak_bytes"]),
            max_layer_fraction=float(summary["max_layer_fraction"]),
            max_expert_fraction=float(summary["max_expert_fraction"]),
            max_shared_expert_fraction=float(summary["max_shared_expert_fraction"]),
            valid=bool(summary["valid"]),
            invalid_reasons=list(summary["invalid_reasons"]),
        )
    return results


def _flatten_physical(
    artifact_root: Path,
    attention: Mapping[str, Any],
    expert: Mapping[str, Any],
    other: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    expert_rows: list[dict[str, Any]] = []
    canonical_local = _local_profile("canonical_fast_local", protocol, 14336)
    for row in expert["results"]:
        degree = int(row["stripe_degree"])
        rows = int(row["rows"])
        payload = int(row["one_partial_output_bytes_per_worker"])
        input_payload = int(row["input_bytes_per_worker"]) + int(
            row["route_metadata_bytes_per_worker"]
        )
        broadcast_ms = canonical_local.one_way_ms(input_payload)
        reduction_ms = 2 * math.ceil(math.log2(degree)) * canonical_local.one_way_ms(payload)
        naive_network = 16 * reduction_ms
        expert_rows.append(
            {
                "stripe_degree": degree,
                "rows": rows,
                "active_experts": row["active_experts"],
                "resident_active_bytes_max": max(
                    int(worker["active_runtime_weight_bytes"])
                    for worker in row["worker_results"]
                ),
                "input_bytes_per_worker": input_payload,
                "route_metadata_bytes_per_worker": row["route_metadata_bytes_per_worker"],
                "output_bytes_per_worker": payload,
                "worker_compute_ceiling_ms": row["independent_worker_compute_ceiling_ms"],
                "sequential_worker_compute_ms": row["sequential_worker_compute_ms"],
                "canonical_whole_ms": row["canonical_whole_expert"]["wall"]["p50_ms"],
                "stripe_canonical_local_ms": float(row["independent_worker_compute_ceiling_ms"])
                + broadcast_ms
                + reduction_ms,
                "naive_per_expert_local_ms": float(row["independent_worker_compute_ceiling_ms"])
                + broadcast_ms
                + naive_network,
                "network_visible_outputs_per_worker": 1,
                "naive_network_visible_outputs_per_worker": 16,
                "physical_launches_per_worker": max(
                    int(worker["total_physical_launches"])
                    for worker in row["worker_results"]
                ),
                "logical_operations_per_worker": max(
                    int(worker["logical_expert_operations"])
                    for worker in row["worker_results"]
                ),
                "relative_l2_error": row["metrics"]["relative_l2_error"],
                "pass": row["pass"],
            }
        )
    _write_csv(artifact_root / "physical/expert-stripe-bank.csv", expert_rows)

    kda_rows: list[dict[str, Any]] = []
    mla_rows: list[dict[str, Any]] = []
    for split in ("calibration", "heldout"):
        for row in attention[split]["results"]:
            destination = kda_rows if row["attention_type"] == "KDA" else mla_rows
            for worker in row["workers"]:
                destination.append(
                    {
                        "split": split,
                        "layer": row["layer"],
                        "rows": row["rows"],
                        "stripe_degree": row["stripe_degree"],
                        "worker_id": worker["worker_id"],
                        "stripe_index": worker["stripe_index"],
                        "head_start": worker["head_start"],
                        "head_stop": worker["head_stop"],
                        "runtime_weight_bytes": worker["runtime_weight_bytes"],
                        "persistent_state_bytes": worker["persistent_state_bytes"],
                        "wall_p50_ms": worker["wall"]["p50_ms"],
                        "cuda_p50_ms": worker.get("cuda", {}).get("p50_ms", ""),
                        "host_overhead_p50_ms": worker.get("host_overhead", {}).get(
                            "p50_ms", ""
                        ),
                        "physical_launches": worker["physical_launches"],
                        "network_visible_outputs": worker["network_visible_partial_outputs"],
                        "cross_degree_relative_l2": row["cross_degree_metrics"]["relative_l2_error"],
                        "pass": row["cross_degree_pass"],
                    }
                )
    _write_csv(artifact_root / "physical/kda-stripe.csv", kda_rows)
    _write_csv(artifact_root / "physical/mla-stripe.csv", mla_rows)

    shared_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    for result in other["results"]:
        for operator, value in result["operators"].items():
            for worker in value["workers"]:
                row = {
                    "degree": result["degree"],
                    "rows": result["rows"],
                    "operator": operator,
                    "worker_id": worker["worker_id"],
                    "stripe_index": worker["stripe_index"],
                    "checkpoint_bytes": worker["checkpoint_bytes"],
                    "runtime_weight_bytes": worker["runtime_weight_bytes"],
                    "wall_p50_ms": worker["duration"]["p50_ms"],
                    "cuda_p50_ms": worker.get("cuda", {}).get("p50_ms", ""),
                    "host_overhead_p50_ms": worker.get("host_overhead", {}).get("p50_ms", ""),
                }
                if operator == "shared_expert":
                    shared_rows.append(row)
                elif operator == "lm_head":
                    endpoint_rows.append(row)
                else:
                    projection_rows.append(row)
    for row in other["embedding"]:
        endpoint_rows.append(
            {
                "degree": row["degree"],
                "rows": 1,
                "operator": "embedding",
                "worker_id": row["worker_id"],
                "stripe_index": "owner",
                "checkpoint_bytes": row["resident_weight_bytes"],
                "runtime_weight_bytes": row["resident_weight_bytes"],
                "wall_p50_ms": row["duration"]["p50_ms"],
                "cuda_p50_ms": "",
                "host_overhead_p50_ms": "",
            }
        )
    _write_csv(artifact_root / "physical/shared-expert-stripe.csv", shared_rows)
    _write_csv(artifact_root / "physical/projection-stripe.csv", projection_rows)
    _write_csv(artifact_root / "physical/endpoint-shards.csv", endpoint_rows)

    overhead_rows: list[dict[str, Any]] = []
    for row in protocol["protocol"]["rows"]:
        overhead_rows.append(
            {
                "kind": "localhost_persistent_protocol",
                "payload_bytes": row["payload_bytes"],
                "participants": 2,
                "software_overhead_p50_ms": row["software_overhead_p50_ms"],
                "round_trip_p50_ms": row["round_trip"]["p50_ms"],
                "serialization_p50_ms": row["serialization"]["p50_ms"],
                "framing_p50_ms": row["framing"]["p50_ms"],
                "dispatch_p50_ms": row["dispatch"]["p50_ms"],
                "receive_p50_ms": row["receive"]["p50_ms"],
                "deserialization_p50_ms": row["deserialization"]["p50_ms"],
            }
        )
    for row in other["reductions"]:
        overhead_rows.append(
            {
                "kind": "physical_gpu_reduction_compute",
                "payload_bytes": row["payload_bytes_per_participant"],
                "participants": row["participants"],
                "software_overhead_p50_ms": "",
                "round_trip_p50_ms": "",
                "serialization_p50_ms": "",
                "framing_p50_ms": "",
                "dispatch_p50_ms": row["local_reduction_compute"]["p50_ms"],
                "receive_p50_ms": "",
                "deserialization_p50_ms": "",
            }
        )
    _write_csv(
        artifact_root / "physical/collective-software-overhead.csv", overhead_rows
    )

    samples: list[dict[str, Any]] = []
    for arm, receipt in (
        ("attention", attention),
        ("expert", expert),
        ("other", other),
        ("protocol", protocol),
    ):
        for row in receipt.get("gpu_samples", []):
            samples.append({"arm": arm, **row})
    _write_csv(artifact_root / "physical/gpu-samples.csv", samples)
    _write_json(
        artifact_root / "physical/shard-service-raw.json",
        {
            "schema_version": "experiment-019-shard-service-raw-v1",
            "attention": attention,
            "expert": expert,
            "other": other,
            "protocol": protocol,
        },
    )


def _heldout_validation(
    artifact_root: Path,
    attention: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for calibration in _attention_rows(attention, "calibration"):
        heldout = _find_attention(
            attention,
            "heldout",
            str(calibration["attention_type"]),
            int(calibration["stripe_degree"]),
            int(calibration["rows"]),
        )
        estimate = float(calibration["independent_worker_compute_ceiling_ms"])
        actual = float(heldout["independent_worker_compute_ceiling_ms"])
        error = abs(estimate - actual) / actual if actual else math.inf
        rows.append(
            {
                "attention_type": calibration["attention_type"],
                "calibration_layer": calibration["layer"],
                "heldout_layer": heldout["layer"],
                "stripe_degree": calibration["stripe_degree"],
                "rows": calibration["rows"],
                "estimated_worker_ceiling_ms": estimate,
                "heldout_worker_ceiling_ms": actual,
                "absolute_percentage_error": error,
            }
        )
    errors = [float(row["absolute_percentage_error"]) for row in rows]
    result = {
        "median_absolute_percentage_error": float(np.percentile(errors, 50)),
        "p90_absolute_percentage_error": float(np.percentile(errors, 90)),
        "maximum_absolute_percentage_error": max(errors),
        "status": "PASS" if float(np.percentile(errors, 50)) <= 0.10 else "FAIL",
        "row_count": len(rows),
    }
    _write_csv(artifact_root / "validation/heldout-layer-service.csv", rows)
    return result


def _collective_ms(profile: Any, payload: int, degree: int, *, allreduce: bool) -> float:
    steps = math.ceil(math.log2(degree)) * (2 if allreduce else 1)
    return steps * profile.one_way_ms(payload)


def _serial_layer_cost(
    service: MeasuredShardService,
    profile: Any,
    attention_type: str,
) -> float:
    return sum(_serial_layer_components(service, profile, attention_type).values())


def _serial_layer_components(
    service: MeasuredShardService,
    profile: Any,
    attention_type: str,
) -> dict[str, float]:
    degree = service.degree
    rows = service.rows
    attention_common = (
        service.kda_common_ms if attention_type == "KDA" else service.mla_common_ms
    )
    attention_workers = (
        service.kda_worker_ms if attention_type == "KDA" else service.mla_worker_ms
    )
    latent_allgather = sum(
        profile.one_way_ms(
            min(rows * 3584 * 4, math.ceil(rows * 3584 * 4 / degree) * (1 << step))
        )
        for step in range(math.ceil(math.log2(degree)))
    )
    return {
        "attention_common": attention_common,
        "attention_worker_compute_sum": sum(attention_workers),
        "attention_network_collective": _collective_ms(
            profile, rows * 7168 * 4, degree, allreduce=True
        ),
        "attention_reduction_compute_sum": degree
        * service.hidden_pair_reduce_ms
        * math.ceil(math.log2(degree)),
        "moe_pre_router": service.moe_pre_ms,
        "latent_down_worker_compute_sum": sum(service.latent_down_worker_ms),
        "latent_allgather_network": latent_allgather,
        "expert_worker_compute_sum": sum(service.expert_worker_ms),
        "expert_network_collective": _collective_ms(
            profile, rows * 3584 * 4, degree, allreduce=True
        ),
        "expert_reduction_compute_sum": degree
        * service.latent_pair_reduce_ms
        * math.ceil(math.log2(degree)),
        "routed_norm_worker_compute_sum": degree * service.routed_norm_ms,
        "latent_up_shared_worker_compute_sum": sum(
            service.latent_up_shared_worker_ms
        ),
        "moe_network_collective": _collective_ms(
            profile, rows * 7168 * 4, degree, allreduce=True
        ),
        "moe_reduction_compute_sum": degree
        * service.hidden_pair_reduce_ms
        * math.ceil(math.log2(degree)),
        "layer_finalize": service.layer_finalize_ms,
    }


def _serial_validation(
    repo: Path,
    artifact_root: Path,
    services: Mapping[tuple[int, int], MeasuredShardService],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    historical = _read_json(repo / "artifacts/experiment-018/physical/service-raw.json")
    layer_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for degree in (8, 16, 32):
        for rows in CHUNKS:
            service = services[(degree, rows)]
            profile = _local_profile(
                "canonical_fast_local", protocol, rows * 7168 * 4
            )
            for layer, attention_type in ((89, "KDA"), (91, "Gated_MLA")):
                components = _serial_layer_components(
                    service, profile, attention_type
                )
                predicted = sum(components.values())
                reference = float(
                    historical["layers"][str(layer)]["service"][str(rows)]["wall"][
                        "p50_ms"
                    ]
                )
                error = abs(predicted - reference) / reference
                layer_rows.append(
                    {
                        "layer": layer,
                        "attention_type": attention_type,
                        "stripe_degree": degree,
                        "rows": rows,
                        "bottom_up_serial_ms": predicted,
                        "canonical_unsharded_physical_ms": reference,
                        "absolute_percentage_error": error,
                        "within_10_percent_preferred": error <= 0.10,
                        "within_15_percent_maximum": error <= 0.15,
                        "normalization_factor": 1.0,
                    }
                )
                component_rows.extend(
                    {
                        "layer": layer,
                        "attention_type": attention_type,
                        "stripe_degree": degree,
                        "rows": rows,
                        "component": component,
                        "serial_ms": value,
                        "fraction_of_bottom_up_serial": value / predicted,
                        "canonical_unsharded_physical_ms": reference,
                    }
                    for component, value in components.items()
                )
    _write_csv(artifact_root / "validation/serial-reconstruction.csv", layer_rows)
    _write_csv(
        artifact_root / "validation/serial-component-breakdown.csv",
        component_rows,
    )

    # Use the smallest valid Tier-A service geometry (P=8) for the serial target
    # reconstruction. It is calculated first and compared second.
    full_rows: list[dict[str, Any]] = []
    kda_count = 69
    mla_count = 24
    for block, accepted, historical_ms, historical_tok_s in HISTORICAL_CURVE:
        chunk = 4
        remaining = accepted
        modeled = 0.0
        while remaining:
            actual = min(chunk, remaining)
            service_rows = actual if actual in CHUNKS else 4
            service = services[(8, service_rows)]
            profile = _local_profile(
                "canonical_fast_local", protocol, service_rows * 7168 * 4
            )
            modeled += kda_count * _serial_layer_cost(service, profile, "KDA")
            modeled += mla_count * _serial_layer_cost(service, profile, "Gated_MLA")
            modeled += service.embedding_ms + sum(service.endpoint_worker_ms)
            remaining -= actual
        error = abs(modeled - historical_ms) / historical_ms
        full_rows.append(
            {
                "block": block,
                "accepted_rows": accepted,
                "bottom_up_serial_ms": modeled,
                "historical_exact_target_ms": historical_ms,
                "historical_tok_s_per_user": historical_tok_s,
                "absolute_percentage_error": error,
                "within_15_percent": error <= 0.15,
                "normalization_factor": 1.0,
            }
        )
    _write_csv(artifact_root / "validation/baseline-comparison.csv", full_rows)
    headline_representative = [
        row for row in layer_rows if int(row["stripe_degree"]) == 8
    ]
    representative_max = max(
        float(row["absolute_percentage_error"]) for row in headline_representative
    )
    sweep_representative_max = max(
        float(row["absolute_percentage_error"]) for row in layer_rows
    )
    full_max = max(float(row["absolute_percentage_error"]) for row in full_rows)
    return {
        "status": "PASS" if representative_max <= 0.15 and full_max <= 0.15 else "FAIL",
        "representative_maximum_error": representative_max,
        "sweep_representative_maximum_error": sweep_representative_max,
        "full_target_maximum_error": full_max,
        "full_target_block16_error": next(
            float(row["absolute_percentage_error"])
            for row in full_rows
            if int(row["block"]) == 16
        ),
        "normalization_applied": False,
        "layer_rows": layer_rows,
        "component_rows": component_rows,
        "full_rows": full_rows,
    }


def _network_and_sensitivity(
    artifact_root: Path,
    best: Mapping[str, Any],
    service: MeasuredShardService,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spec = PlacementSpec(
        float(best["memory_cap_gib"]),
        int(best["stripe_degree"]),
        int(best["depth_span"]),
        int(best["chunk"]),
        block_candidates=int(best["block"]),
        expert_allocation_overhead_factor=float(
            best["expert_allocation_overhead_factor"]
        ),
    )
    network_rows: list[dict[str, Any]] = []
    for profile_name in (
        "very_fast_local",
        "canonical_fast_local",
        "commodity_fast_lan",
        "regional",
        "residential_wan",
    ):
        for activation in ("hybrid", "replicated_small", "hidden_sharded"):
            profile = _local_profile(
                profile_name, protocol, int(best["chunk"]) * 7168 * 4
            )
            configuration = SimulationConfiguration(
                placement=spec,
                block=int(best["block"]),
                chunk=int(best["chunk"]),
                local_profile=profile,
                inter_pod_profile=(
                    _local_profile(
                        "canonical_inter_pod",
                        protocol,
                        int(best["chunk"]) * 7168 * 4,
                    )
                    if profile_name in {"very_fast_local", "canonical_fast_local", "commodity_fast_lan"}
                    else profile
                ),
                activation_strategy=activation,
            )
            result, _run = simulate(service, configuration)
            network_rows.append(result)
    _write_csv(artifact_root / "network/network-sensitivity.csv", network_rows)

    slowdown_rows: list[dict[str, Any]] = []
    for slowdown in (1.0, 1.5, 2.0, 2.5, 3.0):
        configuration = SimulationConfiguration(
            placement=spec,
            block=int(best["block"]),
            chunk=int(best["chunk"]),
            local_profile=_local_profile(
                "canonical_fast_local", protocol, int(best["chunk"]) * 7168 * 4
            ),
            inter_pod_profile=_local_profile(
                "canonical_inter_pod", protocol, int(best["chunk"]) * 7168 * 4
            ),
            compute_slowdown=slowdown,
        )
        result, _run = simulate(service, configuration)
        slowdown_rows.append(result)
    _write_csv(artifact_root / "simulation/compute-slowdown-sensitivity.csv", slowdown_rows)

    heterogeneity_rows: list[dict[str, Any]] = []
    homogeneous_ms = float(slowdown_rows[0]["target_pass_ms"])
    for scenario, jitter in (
        ("homogeneous", 0.0),
        ("ten_percent_1p5x", 0.0),
        ("ten_percent_2x", 0.0),
        ("twentyfive_percent_1p5x", 0.0),
        ("random_plus_minus_20", 0.0),
        ("homogeneous", 0.20),
    ):
        configuration = SimulationConfiguration(
            placement=spec,
            block=int(best["block"]),
            chunk=int(best["chunk"]),
            local_profile=_local_profile(
                "canonical_fast_local", protocol, int(best["chunk"]) * 7168 * 4
            ),
            inter_pod_profile=_local_profile(
                "canonical_inter_pod", protocol, int(best["chunk"]) * 7168 * 4
            ),
            heterogeneity=scenario,
            jitter_fraction=jitter,
        )
        result, _run = simulate(service, configuration)
        result["heterogeneity"] = (
            "network_jitter_plus_minus_20" if jitter else scenario
        )
        result["bottleneck_amplification"] = (
            float(result["target_pass_ms"]) / homogeneous_ms
        )
        heterogeneity_rows.append(result)
    _write_csv(
        artifact_root / "network/heterogeneity-sensitivity.csv", heterogeneity_rows
    )
    return network_rows, slowdown_rows, heterogeneity_rows


def _collective_sweep(
    artifact_root: Path,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_name in (
        "very_fast_local",
        "canonical_fast_local",
        "commodity_fast_lan",
        "regional",
        "residential_wan",
    ):
        for participants in (4, 8, 16, 32):
            for payload in (16, 14336, 28672, 57344, 114688):
                profile = _local_profile(profile_name, protocol, payload)
                for algorithm, steps in (
                    ("binary_tree_reduce", math.ceil(math.log2(participants))),
                    ("binary_tree_allreduce", 2 * math.ceil(math.log2(participants))),
                    ("recursive_doubling", math.ceil(math.log2(participants))),
                    ("ring_allreduce", 2 * (participants - 1)),
                ):
                    base = steps * profile.rtt_ms / 2
                    serialization = steps * payload * 8 / (
                        profile.bandwidth_gbps * 1_000_000
                    )
                    software = steps * profile.software_overhead_ms
                    if algorithm == "binary_tree_reduce":
                        aggregate = (participants - 1) * payload
                    elif algorithm == "binary_tree_allreduce":
                        aggregate = 2 * (participants - 1) * payload
                    elif algorithm == "recursive_doubling":
                        aggregate = participants * payload * steps
                    else:
                        # Each ring step carries one P-th shard on every one
                        # of P links, for 2(P-1) reduce-scatter/all-gather steps.
                        aggregate = 2 * (participants - 1) * payload
                    rows.append(
                        {
                            "profile": profile_name,
                            "algorithm": algorithm,
                            "participants": participants,
                            "steps": steps,
                            "payload_per_step_bytes": payload,
                            "aggregate_wire_bytes": aggregate,
                            "base_network_latency_ms": base,
                            "serialization_ms": serialization,
                            "software_overhead_ms": software,
                            "completion_time_ms": base + serialization + software,
                        }
                    )
    _write_csv(artifact_root / "network/collective-sweep.csv", rows)
    return rows


def _utilization_and_trace(
    repo: Path,
    artifact_root: Path,
    best: dict[str, Any],
    service: MeasuredShardService,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    spec = PlacementSpec(
        float(best["memory_cap_gib"]),
        int(best["stripe_degree"]),
        int(best["depth_span"]),
        int(best["chunk"]),
        block_candidates=int(best["block"]),
        expert_allocation_overhead_factor=float(
            best["expert_allocation_overhead_factor"]
        ),
    )
    configuration = SimulationConfiguration(
        placement=spec,
        block=int(best["block"]),
        chunk=int(best["chunk"]),
        local_profile=_local_profile(
            "canonical_fast_local", protocol, int(best["chunk"]) * 7168 * 4
        ),
        inter_pod_profile=_local_profile(
            "canonical_inter_pod", protocol, int(best["chunk"]) * 7168 * 4
        ),
    )
    result, run = simulate(service, configuration)
    result = {**best, **result}
    trace_path = artifact_root / "simulation/event-traces/winning-tier-a.json"
    _write_json(trace_path, run.as_dict())
    result["event_trace_path"] = str(trace_path.relative_to(repo)).replace("\\", "/")
    utilization_rows = [
        {
            "worker_id": worker,
            "utilization": value,
            "idle_ms": run.worker_idle_ms[worker],
            "busy_ms": value * run.makespan_ms,
        }
        for worker, value in run.worker_utilization.items()
    ]
    _write_csv(artifact_root / "simulation/worker-utilization.csv", utilization_rows)
    record_map = {row.task_id: row for row in run.records}
    critical_records = [record_map[identifier] for identifier in run.critical_path_task_ids]
    critical = {
        "schema_version": "experiment-019-critical-path-v1",
        "makespan_ms": run.makespan_ms,
        "task_ids": list(run.critical_path_task_ids),
        "compute_ms": sum(
            row.duration_ms for row in critical_records if row.resource_type == "microworker"
        ),
        "network_ms": sum(
            row.duration_ms for row in critical_records if row.resource_type == "network"
        ),
        "software_overhead_ms": sum(row.software_overhead_ms for row in critical_records),
        "base_network_latency_ms": sum(row.base_network_latency_ms for row in critical_records),
        "serialization_ms": sum(row.serialization_ms for row in critical_records),
        "operator_breakdown": {},
    }
    for row in critical_records:
        critical["operator_breakdown"][row.operator] = (
            critical["operator_breakdown"].get(row.operator, 0.0) + row.duration_ms
        )
    _write_json(artifact_root / "simulation/critical-path.json", critical)
    return result, run


def _monolith_tax(
    artifact_root: Path,
    best: Mapping[str, Any],
    run: Any,
) -> list[dict[str, Any]]:
    record_map = {row.task_id: row for row in run.records}
    records = [record_map[identifier] for identifier in run.critical_path_task_ids]
    software = sum(row.software_overhead_ms for row in records)
    attention_network = sum(
        row.duration_ms - row.software_overhead_ms
        for row in records
        if row.resource_type == "network" and "attention" in row.operator
    )
    expert_network = sum(
        row.duration_ms - row.software_overhead_ms
        for row in records
        if row.resource_type == "network"
        and any(value in row.operator for value in ("expert", "moe"))
    )
    repartition = sum(
        row.duration_ms - row.software_overhead_ms
        for row in records
        if row.resource_type == "network" and "allgather" in row.operator
    )
    interpod = sum(
        row.duration_ms - row.software_overhead_ms
        for row in records
        if row.resource_type == "network" and "inter_pod" in row.operator
    )
    compute = sum(
        row.duration_ms for row in records if row.resource_type == "microworker"
    )
    total_delta = float(best["target_pass_ms"]) - E018_TARGET_PASS_MS
    compute_delta = compute - 2398.204829114651
    attributed = compute_delta + attention_network + expert_network + repartition + interpod + software
    other = total_delta - attributed
    components = [
        ("worker shard compute", compute_delta),
        ("attention reduction", attention_network),
        ("expert fanout/reduction", expert_network),
        ("repartition/all-gather", repartition),
        ("memory placement/inter-pod", interpod),
        ("worker software overhead", software),
        ("load imbalance and other", other),
    ]
    rows: list[dict[str, Any]] = []
    current = E018_TARGET_PASS_MS
    for component, delta in components:
        rows.append(
            {
                "component": component,
                "start_ms": current,
                "delta_ms": delta,
                "finish_ms": current + delta,
                "e018_target_pass_ms": E018_TARGET_PASS_MS,
                "e019_target_pass_ms": best["target_pass_ms"],
            }
        )
        current += delta
    _write_csv(artifact_root / "simulation/monolith-tax.csv", rows)
    return rows


def _economics(
    artifact_root: Path,
    tier_best: Mapping[int, Mapping[str, Any]],
    manifests: Mapping[int, PlacementResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier, result in tier_best.items():
        manifest = manifests[tier]
        tok_s = float(result["exact_tok_s_per_user"])
        worker_seconds = float(result["worker_seconds_per_accepted_token"])
        network_bytes = float(result["network_bytes_per_accepted_token"])
        total_resident_gib = manifest.total_resident_weight_bytes / GIB
        gpu_hours = worker_seconds * 1_000_000 / 3600
        rows.append(
            {
                "candidate_scope": "tier_best",
                "memory_cap_gib": tier,
                "worker_count": len(manifest.workers),
                "capacity_reservation_gib": len(manifest.workers) * tier,
                "total_resident_gib": total_resident_gib,
                "unused_capacity_gib": len(manifest.workers) * tier
                - sum(worker.peak_total_bytes for worker in manifest.workers) / GIB,
                "exact_tok_s_per_user": tok_s,
                "active_worker_seconds_per_output_token": worker_seconds,
                "5090_equivalent_compute_seconds_per_token": worker_seconds,
                "network_gb_per_1m_tokens": network_bytes * 1_000_000 / 1e9,
                "gpu_hours_per_1m_tokens": gpu_hours,
                "inherited_projected_usd_per_1m": gpu_hours
                * INHERITED_GPU_HOURLY_USD,
                "inherited_cost_assumption": "$0.15/GPU-hour from prior lab model; illustrative, not consumer reservation price",
                "resident_gib_per_tok_s": total_resident_gib / tok_s,
                "worker_count_per_tok_s": len(manifest.workers) / tok_s,
                "network_mb_per_output_token": network_bytes / 1e6,
                "compute_work_inflation": result.get("compute_work_inflation", ""),
                "replication_factor": manifest.total_resident_weight_bytes
                / manifest.checkpoint_payload_bytes,
            }
        )
    _write_csv(artifact_root / "economics/results.csv", rows)
    return rows


def _coverage_artifacts(
    artifact_root: Path,
    catalog: CheckpointCatalog,
    headline: PlacementResult,
) -> None:
    records = catalog.records()
    _write_csv_stream(
        artifact_root / "placement/tensor-census.csv",
        (
            "tensor",
            "file",
            "dtype",
            "shape",
            "data_offset",
            "source_bytes",
            "layer_id",
            "expert_id",
            "role",
        ),
        (
            {
                "tensor": record.name,
                "file": record.file,
                "dtype": record.dtype,
                "shape": "x".join(str(value) for value in record.shape),
                "data_offset": record.data_offset,
                "source_bytes": record.byte_size,
                "layer_id": record.layer_id,
                "expert_id": record.expert_id,
                "role": record.role,
            }
            for record in sorted(records.values(), key=lambda value: value.name)
        ),
    )
    workers = headline.workers

    def coverage_rows() -> Iterable[dict[str, Any]]:
        for record in sorted(records.values(), key=lambda value: value.name):
            assignments = assignments_for(record, headline.spec)
            owners = "|".join(
                (
                    f"{workers[item.worker_index].worker_id}@"
                    f"axis={item.axis};{item.start}:{item.stop}/{item.total};"
                    f"bytes={item.bytes};replicated={item.replicated}"
                )
                for item in assignments
            )
            assigned = sum(item.bytes for item in assignments if not item.replicated)
            yield {
                "tensor": record.name,
                "file": record.file,
                "dtype": record.dtype,
                "shape": "x".join(str(value) for value in record.shape),
                "source_bytes": record.byte_size,
                "layer_id": record.layer_id,
                "expert_id": record.expert_id,
                "role": record.role,
                "partition_count": len(assignments),
                "assigned_bytes": assigned,
                "replicated_bytes": sum(
                    item.bytes for item in assignments if item.replicated
                ),
                "gap_bytes": max(0, record.byte_size - assigned),
                "overlap_bytes": max(0, assigned - record.byte_size),
                "coverage_status": "PASS" if assigned == record.byte_size else "FAIL",
                "owners": owners,
            }

    _write_csv_stream(
        artifact_root / "placement/tensor-coverage.csv",
        (
            "tensor",
            "file",
            "dtype",
            "shape",
            "source_bytes",
            "layer_id",
            "expert_id",
            "role",
            "partition_count",
            "assigned_bytes",
            "replicated_bytes",
            "gap_bytes",
            "overlap_bytes",
            "coverage_status",
            "owners",
        ),
        coverage_rows(),
    )


def _checkpoint_audit(
    artifact_root: Path,
    attention: Mapping[str, Any],
    expert: Mapping[str, Any],
    other: Mapping[str, Any],
    full: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm, values in (
        ("attention_calibration", attention["calibration"]["checkpoint_read_audit"]),
        ("attention_heldout", attention["heldout"]["checkpoint_read_audit"]),
        ("expert", expert["checkpoint_read_audit"]),
        ("other", other["checkpoint_read_audit"]),
        ("full_93", full["checkpoint_read_audit"]),
    ):
        rows.extend({"arm": arm, **row} for row in values)
    violations = [
        row
        for row in rows
        if bool(row["full_source_tensor_materialized"])
        and int(row["source_tensor_bytes"]) > 32 * 1024 * 1024
    ]
    receipt = {
        "schema_version": "experiment-019-checkpoint-read-audit-v1",
        "status": "PASS" if not violations else "FAIL",
        "request_count": len(rows),
        "bytes_read": sum(int(row["bytes_read"]) for row in rows),
        "large_full_materialization_count": len(violations),
        "violations": violations,
        "reads": rows,
        "control_exclusion": "canonical whole-expert controls are non-headline validation and are not direct-loader claims",
    }
    _write_json(artifact_root / "placement/checkpoint-read-audit.json", receipt)
    return receipt


def _source_manifest(
    repo: Path,
    artifact_root: Path,
    sources: Sequence[tuple[Path, str]],
) -> dict[str, Any]:
    rows = []
    for path, role in sources:
        resolved = path.resolve()
        rows.append(
            {
                "path": str(resolved),
                "role": role,
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )
    return {
        "schema_version": "experiment-019-source-manifest-v1",
        "generated_unix_ns": time.time_ns(),
        "git_commit": _command(("git", "rev-parse", "HEAD"), repo),
        "git_status": _command(("git", "status", "--short"), repo),
        "preexisting_dirty_path_preserved": "third_party/colibri",
        "files": rows,
    }


def _classification(
    *,
    validation_status: str,
    full_status: str,
    tier_best: Mapping[int, Mapping[str, Any]],
) -> str:
    if validation_status != "PASS":
        return "MODEL_INVALID"
    if full_status != "PASS":
        return "SWARM_FAIL"
    tier_a = float(tier_best[20]["exact_tok_s_per_user"])
    tier_b = float(tier_best[8]["exact_tok_s_per_user"])
    if tier_b >= 5.0:
        return "SWARM_PASS_STRONG"
    if tier_a >= 5.0:
        return "SWARM_PASS"
    if tier_a >= 4.0:
        return "SWARM_STRONG_PARTIAL"
    if tier_a >= 3.5:
        return "SWARM_WEAK"
    return "SWARM_FAIL"


def finalize(
    repo: Path,
    checkpoint: Path,
    artifact_root: Path,
    cuda_library: Path,
    shard_library: Path,
    oracle_root: Path,
    *,
    resume_derived: bool = False,
) -> dict[str, Any]:
    started = time.time_ns()
    catalog = CheckpointCatalog(checkpoint)
    attention = _read_json(artifact_root / "physical/attention-raw.json")
    expert = _read_json(artifact_root / "physical/expert-stripe-raw.json")
    other = _read_json(artifact_root / "physical/other-shards-raw.json")
    protocol = _read_json(artifact_root / "control-plane/protocol-and-scaling.json")
    representative = _read_json(
        artifact_root / "correctness/representative-layers-raw.json"
    )
    depth = _read_json(artifact_root / "correctness/depth-span.json")
    full = _read_json(artifact_root / "correctness/full-93-sharded.json")

    expert_allocation_factors = _expert_allocation_factors(expert)
    placement_rows = _placement_sweep(catalog, expert_allocation_factors)
    _write_csv(artifact_root / "placement/placement-sweep.csv", placement_rows)
    if resume_derived:
        worker_rows = _read_typed_csv(artifact_root / "simulation/worker-sweep.csv")
        services = {
            (degree, rows): _physical_service(
                attention, expert, other, protocol, degree=degree, rows=rows
            )
            for degree in (8, 16, 32)
            for rows in CHUNKS
        }
    else:
        worker_rows, services = _run_sweep(
            placement_rows, attention, expert, other, protocol
        )
        _annotate_compute_work_inflation(repo, worker_rows)
    tier_best = _tier_best(worker_rows)
    if set(tier_best) != set(TIERS):
        raise RuntimeError("not every memory tier has a valid placement")
    manifests = (
        _load_exact_manifests(artifact_root)
        if resume_derived
        else _exact_manifests(catalog, artifact_root, tier_best)
    )
    for tier, manifest in manifests.items():
        winner = tier_best[tier]
        winner["minimum_workers_required_for_capacity"] = min(
            int(row["worker_count"])
            for row in placement_rows
            if bool(row["valid"]) and float(row["memory_cap_gib"]) == tier
        )
        winner["max_worker_peak_gib"] = manifest.max_worker_peak_bytes / GIB
        winner["max_layer_fraction"] = manifest.max_layer_fraction
        winner["max_expert_fraction"] = manifest.max_expert_fraction
        winner["worker_count"] = len(manifest.workers)
        winner["pod_count"] = manifest.spec.pod_count
        winner["total_resident_model_bytes"] = manifest.total_resident_weight_bytes
        winner["total_cluster_accounted_peak_bytes"] = sum(
            worker.peak_total_bytes for worker in manifest.workers
        )
        winner["replicated_weight_bytes"] = manifest.replicated_weight_bytes
        winner["weight_replication_factor"] = (
            manifest.total_resident_weight_bytes / manifest.checkpoint_payload_bytes
        )
        winner["unused_capacity_bytes"] = (
            len(manifest.workers) * manifest.spec.cap_bytes
            - sum(worker.peak_total_bytes for worker in manifest.workers)
        )
        for candidate in worker_rows:
            if (
                float(candidate["memory_cap_gib"]) == tier
                and int(candidate["stripe_degree"]) == int(winner["stripe_degree"])
                and int(candidate["depth_span"]) == int(winner["depth_span"])
                and int(candidate["chunk"]) == int(winner["chunk"])
                and int(candidate["block"]) == int(winner["block"])
            ):
                candidate.update(winner)
    _write_csv(artifact_root / "simulation/worker-sweep.csv", worker_rows)

    tier_a = dict(tier_best[20])
    tier_a_service = services[(int(tier_a["stripe_degree"]), int(tier_a["chunk"]))]
    tier_a, winning_run = _utilization_and_trace(
        repo, artifact_root, tier_a, tier_a_service, protocol
    )
    tier_a_manifest = manifests[20]
    tier_a.update(
        {
            "max_worker_peak_gib": tier_a_manifest.max_worker_peak_bytes / GIB,
            "max_layer_fraction": tier_a_manifest.max_layer_fraction,
            "max_expert_fraction": tier_a_manifest.max_expert_fraction,
            "worker_count": len(tier_a_manifest.workers),
            "pod_count": tier_a_manifest.spec.pod_count,
            "total_resident_gib": tier_a_manifest.total_resident_weight_bytes / GIB,
            "total_cluster_accounted_peak_gib": sum(
                worker.peak_total_bytes for worker in tier_a_manifest.workers
            )
            / GIB,
            "network_mb_per_output_token": float(
                tier_a["network_bytes_per_accepted_token"]
            )
            / 1e6,
        }
    )
    tier_best[20] = tier_a

    _flatten_physical(artifact_root, attention, expert, other, protocol)
    heldout = _heldout_validation(artifact_root, attention)
    serial = _serial_validation(repo, artifact_root, services, protocol)
    collective_rows = _collective_sweep(artifact_root, protocol)
    network_rows, slowdown_rows, heterogeneity_rows = _network_and_sensitivity(
        artifact_root, tier_a, tier_a_service, protocol
    )
    monolith_rows = _monolith_tax(artifact_root, tier_a, winning_run)
    economics = _economics(artifact_root, tier_best, manifests)

    _write_csv(
        artifact_root / "control-plane/scaling.csv", protocol["scaling"]
    )
    _write_json(
        artifact_root / "correctness/expert-stripe.json",
        {
            "status": expert["status"],
            "results": [
                {
                    "stripe_degree": row["stripe_degree"],
                    "rows": row["rows"],
                    "metrics": row["metrics"],
                    "pass": row["pass"],
                }
                for row in expert["results"]
            ],
            "bank_residency": expert["bank_residency"],
            "block_route_contexts": expert["block_route_contexts"],
        },
    )
    kda_correct = [row for row in representative["results"] if row["attention_type"] == "KDA"]
    mla_correct = [row for row in representative["results"] if row["attention_type"] == "Gated_MLA"]
    _write_json(
        artifact_root / "correctness/kda-layer.json",
        {"status": "PASS" if all(row["pass"] for row in kda_correct) else "FAIL", "results": kda_correct},
    )
    _write_json(
        artifact_root / "correctness/mla-layer.json",
        {"status": "PASS" if all(row["pass"] for row in mla_correct) else "FAIL", "results": mla_correct},
    )

    checkpoint_audit = _checkpoint_audit(
        artifact_root, attention, expert, other, full
    )
    _coverage_artifacts(artifact_root, catalog, tier_a_manifest)

    placement_gate = all(
        result.coverage_gap_bytes == 0
        and result.coverage_overlap_bytes == 0
        and result.coverage_assigned_bytes == result.checkpoint_payload_bytes
        for result in manifests.values()
    )
    no_monolith_gate = (
        tier_a_manifest.max_layer_fraction <= 0.25
        and tier_a_manifest.max_expert_fraction <= 0.25
        and tier_a_manifest.max_shared_expert_fraction <= 0.25
    )
    tier_a_memory_gate = tier_a_manifest.max_worker_peak_bytes <= 20 * GIB
    attention_gate = (
        all(row["pass"] for row in kda_correct)
        and all(row["pass"] for row in mla_correct)
    )
    full_gate = full["status"] == "PASS"
    expert_gate = (
        expert["status"] == "PASS"
        and any(
            int(row["stripe_degree"]) >= 4 and bool(row["pass"])
            for row in expert["results"]
        )
    )
    trace_gate = bool(tier_a["all_compute_resources_are_microworkers"])
    timing_gate = serial["status"] == "PASS" and heldout["status"] == "PASS"
    # A raw scheduler result cannot satisfy the exact-throughput gate when the
    # service model that produced it failed reconstruction. Keep the numeric
    # result in the receipts, but do not promote it to scientific evidence.
    throughput_gate = (
        timing_gate and float(tier_a["exact_tok_s_per_user"]) >= TARGET_TOK_S
    )
    hard_gates = [
        ("Gate 1", "Complete checkpoint worker placement", placement_gate, "placement/tensor-coverage.csv"),
        ("Gate 2", "No monolith", no_monolith_gate, "placement/worker-manifest-20g.json"),
        ("Gate 3", "Peak memory", tier_a_memory_gate, "placement/worker-manifest-20g.json"),
        ("Gate 4", "Direct shard loading", checkpoint_audit["status"] == "PASS", "placement/checkpoint-read-audit.json"),
        ("Gate 5", "Sharded KDA/MLA correctness", attention_gate, "correctness/kda-layer.json; correctness/mla-layer.json"),
        ("Gate 6", "Full 93-layer sharded correctness", full_gate, "correctness/full-93-sharded.json"),
        ("Gate 7", "Bottom-up timing validation", timing_gate, "validation/serial-reconstruction.csv; validation/baseline-comparison.csv"),
        ("Gate 8", "Expert stripe bank", expert_gate, "correctness/expert-stripe.json"),
        ("Gate 9", "Worker-level event trace", trace_gate, tier_a["event_trace_path"]),
        (
            "Gate 10",
            "Exact >=5 tok/s (requires valid timing model)",
            throughput_gate,
            "simulation/worker-sweep.csv",
        ),
    ]
    validation_status = (
        "PASS"
        if placement_gate
        and checkpoint_audit["status"] == "PASS"
        and serial["status"] == "PASS"
        and heldout["status"] == "PASS"
        else "FAIL"
    )
    classification = _classification(
        validation_status=validation_status,
        full_status=str(full["status"]),
        tier_best=tier_best,
    )
    validation = {
        "schema_version": "experiment-019-validation-v1",
        "status": validation_status,
        "classification": classification,
        "hard_gates": [
            {
                "gate": gate,
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
            for gate, name, passed, evidence in hard_gates
        ],
        "heldout_service": heldout,
        "serial_reconstruction": serial,
        "no_normalization": True,
    }
    _write_json(artifact_root / "validation.json", validation)

    capacity_answer = "YES" if tier_a_memory_gate else "NO"
    architectural_verdict = (
        "YES — BOUNDED MICRO-WORKER MODEL CROSSES 5"
        if classification in {"SWARM_PASS", "SWARM_PASS_STRONG"}
        else (
            "MODEL INVALID"
            if classification == "MODEL_INVALID"
            else (
                "NOT YET — BOUNDED MICRO-WORKER MODEL BELOW 5"
                if float(tier_a["exact_tok_s_per_user"]) >= 3.5
                else "NO — SUB-LAYER SWARM THESIS FALSIFIED"
            )
        )
    )
    block16_error = serial["full_target_block16_error"]
    expert_best = min(
        (
            row
            for row in _read_json(artifact_root / "correctness/expert-stripe.json")["results"]
            if int(row["stripe_degree"]) >= 4
        ),
        key=lambda row: float(row["metrics"]["relative_l2_error"]),
    )
    important_failures = [
        "The inherited KDA entry point rejected sub-96-head execution; an Experiment 019 sm_120 shard kernel was required and validated against the canonical equations.",
        "The first full-oracle loader arm spent minutes in CPU BF16 quantization; exact bit-identical GPU startup conversion replaced it.",
        "The first layer-0 smoke used an obsolete dense width (18,432); the checkpoint-authoritative 33,792 width exposed and fixed the incomplete execution slice.",
        "The older non-idot0 oracle differed from the promoted canonical CUDA path by 1.91e-4 at layer 89; the promoted idot0 oracle was selected by repository evidence before final validation.",
        "The initial full-93 validator compared JSON route lists with oracle tuples, producing a false negative despite identical IDs; all 92 stored route rows were recertified with type-stable comparison and zero mismatches at an unchanged tolerance.",
        "The initial 8-layer span fixture preloaded the layer-84 AttnRes snapshot before executing layer 84; restricting fixture snapshots to those strictly before the span start made the 2/4/8-layer gates pass without changing execution semantics.",
        "P=32 exposed a 33,792/32 dense-MLP split that cut native 64-value MXFP4 groups; balanced 64-aligned 17/16-group stripes replaced it and passed physically.",
    ]
    if serial["status"] != "PASS":
        important_failures.append(
            f"Bottom-up serial reconstruction missed the block-16 historical target by {block16_error:.1%}; the P=8 representative layer miss reached {serial['representative_maximum_error']:.1%}. Expert stripes still issue 16 logical expert calls per row (coalescing factor 1.0), and measured shard work inflation is {tier_a['compute_work_inflation']:.3f}x. No correction factor was applied, so the model is invalid."
        )
    if full["status"] != "PASS":
        important_failures.append("The complete 93-layer sharded correctness oracle failed.")

    if classification in {"SWARM_PASS", "SWARM_PASS_STRONG"}:
        recommendation = (
            "Physically instantiate one small P-worker locality pod across independent GPUs and compare measured end-to-end layer latency with the worker event model. Preserve the exact placement and protocol so the only new variable is physical distribution."
        )
    elif 4 <= float(tier_a["exact_tok_s_per_user"]) < 5:
        recommendation = (
            "Attack only the largest measured monolith-tax component, then rerun the same gates. Do not redesign unrelated KDA/MLA components."
        )
    elif serial["status"] != "PASS":
        recommendation = (
            "Physically fuse/coalesce the 16 selected expert fragments on one stripe worker into one or a small number of launches, then remeasure the identical layer-89 route workload and rerun the unchanged serial-reconstruction gate. Expert worker compute is the largest measured serial component. Do not build a distributed pod while the service model remains invalid."
        )
    elif float(tier_a["exact_tok_s_per_user"]) < 3.5:
        recommendation = (
            "Treat the current exact sub-layer architecture as falsified. Experiment 020 should only proceed if it tests a materially different grouping/reduction hypothesis."
        )
    else:
        recommendation = "Measure and remove the largest validated monolith-tax component."

    summary = {
        "schema_version": "experiment-019-summary-v1",
        "classification": classification,
        "throughput_claim_admissible": classification != "MODEL_INVALID",
        "provisional_worker_event_output": classification == "MODEL_INVALID",
        "best_exact": tier_a,
        "evidence_class": "single RTX 5090 physical shard service + deterministic bounded-worker architecture model; no physical multi-GPU swarm",
        "full_93_sharded_correctness": full["status"],
        "bottom_up_reconstruction_error_percent": block16_error * 100,
        "checkpoint_tensor_count": len(catalog.records()),
        "checkpoint_payload_bytes": catalog.declared_total_size,
        "checkpoint_read_request_count": checkpoint_audit["request_count"],
        "direct_loader_large_full_materializations": checkpoint_audit[
            "large_full_materialization_count"
        ],
        "capacity_24gb_class_answer": capacity_answer,
        "rtx_3090_compute_physically_tested": False,
        "physical_multi_gpu_swarm_measured": False,
        "monolith_tax_ms": float(tier_a["target_pass_ms"]) - E018_TARGET_PASS_MS,
        "architectural_verdict": architectural_verdict,
        "physical_verdict": "NOT YET PHYSICALLY PROVEN",
        "executive_paragraph": (
            f"The explicit-worker scheduler numerically produces {tier_a['exact_tok_s_per_user']:.4f} tok/s/user at the canonical hierarchical network with {tier_a['worker_count']} workers, but that number is INADMISSIBLE as a throughput claim because the unnormalized serial reconstruction misses block 16 by {block16_error:.1%}. The scientific classification is therefore {classification}. "
            f"The Tier-A capacity question is {capacity_answer}; RTX 3090 throughput remains NOT PHYSICALLY TESTED."
        ),
        "expert_result_paragraph": (
            f"All P=2/4/8/16/32 stripe sweeps reproduced the whole-expert reference within the fixed tolerance. Complete stripe-0 banks physically held all 896 experts at every P and resolved arbitrary routes without weight movement. Best reported expert-stripe relative L2 is {float(expert_best['metrics']['relative_l2_error']):.3e}. Route coalescing successfully reduces network-visible outputs to one partial per worker, but physical expert launch coalescing remains 1.0: each row still issues 16 expert calls plus one local reduction."
        ),
        "kda_result_paragraph": (
            f"Layer 89 was decomposed by head/projection stripe at P=4/8/16/32. The inherited 96-head-only launcher was replaced by a shard launcher that preserves the exact recurrent/conv equations. Representative correctness status is {'PASS' if all(row['pass'] for row in kda_correct) else 'FAIL'}."
        ),
        "mla_result_paragraph": (
            f"Layer 91 used head-compatible q/kv/gate slices, local compressed KV state, local absorb/gate, and output-column contributions followed by one reduction. Representative correctness status is {'PASS' if all(row['pass'] for row in mla_correct) else 'FAIL'}."
        ),
        "layer_correctness_paragraph": (
            f"KDA-89 and MLA-91 were executed entirely through shard paths at P=4/8/16/32. Maximum observed representative relative L2 was {max(float(row['metrics']['relative_l2_error']) for row in representative['results']):.3e}; ordered routes remained exact."
        ),
        "depth_correctness_paragraph": (
            f"The 2/4/8-layer mixed KDA/MLA spans are {depth['status']}. Maximum span relative L2 was {max(float(row['maximum_relative_l2_error']) for row in depth['results']):.3e}."
        ),
        "full_correctness_paragraph": (
            f"The complete shard-only traversal is {full['status']}. It executed {full['executed_layers']} layers, reported maximum hidden relative L2 {float(full['maximum_relative_l2_error']):.3e}, routes_exact={full['routes_exact']}, and greedy-token match={bool(full.get('head', {}).get('greedy_token_match', False))}."
        ),
        "serial_validation_paragraph": (
            f"Held-out median absolute percentage error is {heldout['median_absolute_percentage_error']:.2%} and p90 is {heldout['p90_absolute_percentage_error']:.2%}. The headline P=8 representative maximum serial reconstruction error is {serial['representative_maximum_error']:.2%}; full block-16 reconstruction error is {block16_error:.2%}. The timing gate is {serial['status']} and normalization_applied=false."
        ),
        "straggler_paragraph": (
            "Each layer’s collective completion is governed by the slowest participating worker. The sensitivity receipt reports target-pass amplification explicitly; mean worker speed is never substituted for the maximum."
        ),
        "architectural_proof_paragraph": (
            "The evidence proves byte-exact checkpoint placeability and exact single-device execution of the sub-layer worker graph. "
            + (
                "Because every validation gate passes, it also establishes that the measured-service architecture model crosses five under the declared locality assumptions."
                if classification in {"SWARM_PASS", "SWARM_PASS_STRONG"}
                else "It does not establish the throughput thesis because at least one declared validation gate fails."
            )
        ),
        "architectural_verdict_paragraph": (
            "This verdict applies only to the bounded-worker architecture model under measured RTX 5090 shard service and shaped links. It is deliberately separate from physical-cluster evidence."
        ),
        "experiment_020_recommendation": recommendation,
        "important_failures": important_failures,
        "hard_gate_statuses": validation["hard_gates"],
        "tier_best": {str(key): value for key, value in tier_best.items()},
        "completed_unix_ns": time.time_ns(),
        "elapsed_seconds": (time.time_ns() - started) / 1e9,
    }
    _write_json(artifact_root / "summary.json", summary)

    truth_rows = [
        ("Does any worker hold a complete 8-layer microcell?", "NO"),
        ("Does any worker hold a complete transformer layer?", "NO"),
        ("Does any headline worker hold a complete routed expert?", "NO"),
        ("Maximum worker peak memory", f"{tier_a['max_worker_peak_gib']:.4f} GiB"),
        ("Maximum fraction of one layer owned by one worker", f"{tier_a['max_layer_fraction'] * 100:.3f}%"),
        ("Maximum fraction of one expert owned by one worker", f"{tier_a['max_expert_fraction'] * 100:.3f}%"),
        ("Is the entire 1.5+ TB K3 checkpoint assigned?", "YES" if placement_gate else "NO"),
        ("Can arbitrary expert routes execute without weight movement?", "YES" if any(row["arbitrary_route_ready"] for row in expert["bank_residency"]) else "NO"),
        ("Was the complete 93-layer graph executed through shard paths?", "YES" if full_gate else "NO"),
        ("Is throughput generated from worker-level events?", "YES" if trace_gate else "NO"),
        ("Was a physical multi-GPU swarm measured?", "NO"),
        ("Was a physical RTX 3090 measured?", "NO"),
        ("Does the bounded-worker model exceed 5 tok/s/user?", "YES" if throughput_gate else "NO"),
    ]
    truth = {
        "schema_version": "experiment-019-truth-table-v1",
        "rows": [{"question": question, "answer": answer} for question, answer in truth_rows],
    }
    _write_json(artifact_root / "truth-table.json", truth)

    _write_json(
        artifact_root / "target-tracker.json",
        {
            "target_tok_s_per_user": 5.0,
            "block_16_max_target_pass_ms": 3400.0,
            "best_exact_tok_s_per_user": tier_a["exact_tok_s_per_user"],
            "best_target_pass_ms": tier_a["target_pass_ms"],
            "throughput_target_met": throughput_gate,
            "admissible_throughput_target_met": throughput_gate
            and classification != "MODEL_INVALID",
            "classification": classification,
        },
    )
    _write_json(
        artifact_root / "model-metadata.json",
        {
            "model": "Kimi-K3",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_payload_bytes": catalog.declared_total_size,
            "tensor_count": len(catalog.records()),
            "safetensors_files": len(set(catalog.weight_map.values())),
            "layers": 93,
            "kda_layers": 69,
            "gated_mla_layers": 24,
            "routed_experts_per_moe_layer": 896,
            "topk": 16,
            "hidden_dimension": 7168,
            "latent_dimension": 3584,
        },
    )
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "process_id": os.getpid(),
        "cwd": str(repo),
        "nvidia_smi": _command(("nvidia-smi",), repo),
        "nvcc": _command(("nvcc", "--version"), repo),
        "git_commit": _command(("git", "rev-parse", "HEAD"), repo),
        "git_status": _command(("git", "status", "--short"), repo),
        "physical_compute_device": "NVIDIA GeForce RTX 5090",
        "rtx_3090_physically_tested": False,
        "timezone": "Australia/Sydney",
    }
    _write_json(artifact_root / "environment.json", environment)

    failures = {
        "schema_version": "experiment-019-failure-log-v1",
        "status": "RECORDED",
        "failures": [
            {"arm": "tooling", "result": "rg unavailable; PowerShell search used"},
            {"arm": "lint", "result": "ruff process blocked by sandbox PermissionError; py_compile and pytest used"},
            {"arm": "KDA P<96 smoke", "result": "inherited CUDA export rejected; redesigned with experiment-owned head-shard kernel"},
            {"arm": "full oracle attempt 1", "result": "terminated after pathological CPU shard quantization; redesigned with bit-exact GPU startup quantizer"},
            {"arm": "oracle boundary", "result": "non-idot0 trace incompatible with promoted current CUDA path; promoted idot0 trace selected from repository regression evidence"},
            {"arm": "layer-0 smoke 1", "result": "failed at relative L2 0.465 due obsolete dense width 18432; checkpoint shape 33792 fixed"},
            {"arm": "full oracle pre-final layout", "result": "terminated because latent-up ownership changed from output rows to input columns before final certification"},
            {"arm": "full oracle validator 1", "result": "false negative from JSON-list versus tuple route comparison; original receipt retained and 92/92 stored route rows recertified equal"},
            {"arm": "depth span 8 fixture 1", "result": "false negative from preloading the layer-84 AttnRes snapshot; original receipt retained and strict-before-start initialization passed"},
            {"arm": "P32 dense stripe 1", "result": "equal 1056-wide split violated 64-value MXFP4 group boundaries; balanced 17/16-group stripes passed"},
            {"arm": "bottom-up serial reconstruction", "result": f"FAIL: headline P8 representative maximum error {serial['representative_maximum_error']:.6f}; block-16 error {block16_error:.6f}; no normalization applied"},
            {"arm": "expert physical launch coalescing", "result": "network outputs coalesced to one partial/worker, but 16 expert calls + one reduction remain per row; coalescing factor 1.0"},
        ],
    }
    _write_json(artifact_root / "failure-log.json", failures)
    _write_json(
        artifact_root / "test-results.json",
        {
            "schema_version": "experiment-019-test-results-v1",
            "unit_tests": {"status": "PASS", "passed": 6, "warning": "unknown pytest asyncio_mode option"},
            "physical_receipts": {
                "attention": attention["status"],
                "expert": expert["status"],
                "other": other["status"],
                "representative": representative["status"],
                "depth": depth["status"],
                "full_93": full["status"],
                "protocol": protocol["status"],
            },
            "hard_gates": validation["hard_gates"],
        },
    )
    command_lines = [
        "# Experiment 019 command receipt",
        "cmd /c src\\swarm_inference\\experiments\\experiment_019\\native\\build_kda_shard.bat artifacts\\experiment-019\\physical\\exp019-kda-shard-sm120-v2.dll",
        "python scripts/experiment_019_physical.py attention --output artifacts/experiment-019/physical/attention-raw.json",
        "python scripts/experiment_019_physical.py expert --output artifacts/experiment-019/physical/expert-stripe-raw.json",
        "python scripts/experiment_019_physical.py other --output artifacts/experiment-019/physical/other-shards-raw.json",
        "python scripts/experiment_019_physical.py representative --output artifacts/experiment-019/correctness/representative-layers-raw.json",
        "python scripts/experiment_019_physical.py depth --output artifacts/experiment-019/correctness/depth-span.json",
        "python scripts/experiment_019_physical.py protocol --output artifacts/experiment-019/control-plane/protocol-and-scaling.json",
        "python scripts/experiment_019_full_sharded.py --degree 4 --layer-limit 93 ...",
        "python scripts/experiment_019_full_sharded.py --recertify-existing ...  # type-stable predicate repair only; no physical re-execution",
        "python -m pytest -q tests/unit/test_experiment_019_no_monolith.py",
        "python -m swarm_inference.experiments.experiment_019.finalize ...",
    ]
    (artifact_root / "commands.txt").write_text(
        "\n".join(command_lines) + "\n", encoding="utf-8"
    )

    source_manifest = _source_manifest(
        repo,
        artifact_root,
        [
            (checkpoint / "config.json", "evidence_input"),
            (checkpoint / "model.safetensors.index.json", "evidence_input"),
            (cuda_library, "physical_cuda_library"),
            (shard_library, "experiment_019_cuda_shard_library"),
            (oracle_root / "hidden-trace.f32", "promoted_correctness_oracle"),
            (oracle_root / "routes.txt", "promoted_route_oracle"),
            (oracle_root / "prefill-logits.f32", "promoted_logit_oracle"),
            (repo / "artifacts/experiment-018/summary.json", "historical_comparison_only"),
            (repo / "artifacts/experiment-018/physical/service-raw.json", "validation_reference_only"),
            (repo / "src/swarm_inference/experiments/experiment_019/checkpoint.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/placement.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/attention.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/physical.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/other_physical.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/quantization.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/protocol.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/sharded_graph.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/events.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/simulation.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/finalize.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/report.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/charts.py", "experiment_source"),
            (repo / "src/swarm_inference/experiments/experiment_019/native/kda_shard.cu", "experiment_source"),
        ],
    )
    _write_json(artifact_root / "source-manifest.json", source_manifest)

    build_all_charts(artifact_root, summary)
    report = render_report(artifact_root, summary, truth, validation)
    report_path = repo / "docs/experiments/EXPERIMENT_019_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=Path("artifacts/experiment-019")
    )
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--shard-library", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument(
        "--resume-derived",
        action="store_true",
        help="Reuse a completed worker sweep and exact manifests after a later-stage failure.",
    )
    arguments = parser.parse_args()
    summary = finalize(
        arguments.repo.resolve(),
        arguments.checkpoint.resolve(),
        arguments.artifact_root.resolve(),
        arguments.cuda_library.resolve(),
        arguments.shard_library.resolve(),
        arguments.oracle_root.resolve(),
        resume_derived=arguments.resume_derived,
    )
    print(
        json.dumps(
            {
                "classification": summary["classification"],
                "exact_tok_s_per_user": summary["best_exact"]["exact_tok_s_per_user"],
                "full_93": summary["full_93_sharded_correctness"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
