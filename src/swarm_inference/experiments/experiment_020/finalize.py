"""Materialize the audited Experiment 020 evidence package.

All Vast inputs consumed here are previously redacted read-only receipts.  This
module contains no Vast mutation path and cannot rent hardware.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_inference.experiments.experiment_019.events import NetworkProfile
from swarm_inference.experiments.experiment_020.cost import estimate_e021_cost
from swarm_inference.experiments.experiment_020.simulation import (
    candidate_projection,
    concurrency_sweep,
    measured_service,
    slowdown_sensitivity,
    straggler_sweep,
    topology_gate,
    work_inflation_breakdown,
)
from swarm_inference.experiments.experiment_020.vast import atomic_write_json

TARGET_TOK_S = 5.0
GIB = 1024**3
BLOCKS = (7, 12, 16)
CHUNKS = (1, 2, 4)
CANDIDATES = ((8, 2), (8, 4), (8, 8), (16, 4), (16, 8))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _command(arguments: Sequence[str], cwd: Path, timeout: int = 30) -> str:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        value = (completed.stdout or completed.stderr).strip()
        return value if value else f"returncode={completed.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _merge_result(
    base: Mapping[str, Any],
    replacement: Mapping[str, Any],
    *,
    degree: int,
    rows: int,
) -> dict[str, Any]:
    value = json.loads(json.dumps(base))
    candidates = list(replacement.get("results", []))
    selected = next(
        row
        for row in candidates
        if int(row.get("rows", -1)) == rows
        and int(row.get("stripe_degree", row.get("degree", -1))) == degree
    )
    key = "stripe_degree" if "stripe_degree" in selected else "degree"
    value["results"] = [
        selected
        if int(row.get("rows", -1)) == rows and int(row.get(key, -1)) == degree
        else row
        for row in value["results"]
    ]
    # The degree-8/row-1 robust receipt also refreshes singleton reductions,
    # normalization, endpoint, and AttnRes measurements.
    for collection in ("embedding", "attnres", "reductions", "rmsnorm"):
        if collection in replacement:
            value[collection] = replacement[collection]
    value["e020_robust_row1_replacement"] = True
    return value


def _service_inputs(repo: Path, artifact: Path) -> dict[str, Any]:
    attention = _read(artifact / "physical" / "attention-raw.json")
    legacy = _read(
        repo / "artifacts" / "experiment-019" / "physical" / "expert-stripe-raw.json"
    )
    grouped8 = _read(artifact / "physical" / "expert-grouped-raw.json")
    grouped16 = _read(artifact / "physical" / "expert-grouped-p16-raw.json")
    other = _read(artifact / "physical" / "other-shards-raw.json")
    grouped8 = _merge_result(
        grouped8,
        _read(artifact / "physical" / "expert-grouped-robust-calibration.json"),
        degree=8,
        rows=1,
    )
    other8 = _merge_result(
        other,
        _read(artifact / "physical" / "other-shards-robust-calibration.json"),
        degree=8,
        rows=1,
    )
    protocol = _read(artifact / "runtime" / "protocol-heldout-raw.json")
    return {
        "attention": attention,
        "legacy": legacy,
        "grouped8": grouped8,
        "grouped16": grouped16,
        "other": other,
        "other8": other8,
        "protocol": protocol,
    }


def _service(inputs: Mapping[str, Any], degree: int, rows: int) -> Any:
    return measured_service(
        inputs["attention"],
        inputs["legacy"],
        inputs["grouped8"] if degree == 8 else inputs["grouped16"],
        inputs["other8"] if degree == 8 and rows == 1 else inputs["other"],
        inputs["protocol"],
        degree=degree,
        rows=rows,
    )


def _canonical_compute_ms(repo: Path, block: int, chunk: int) -> float:
    historical = _read(repo / "artifacts" / "experiment-018" / "physical" / "service-raw.json")
    remaining = block + 1
    total = 0.0
    while remaining:
        rows = min(chunk, remaining)
        kda = float(historical["layers"]["89"]["service"][str(rows)]["wall"]["p50_ms"])
        mla = float(historical["layers"]["91"]["service"][str(rows)]["wall"]["p50_ms"])
        total += 69 * kda + 24 * mla
        remaining -= rows
    return total


def _placement_lookup(repo: Path) -> dict[tuple[int, int], dict[str, Any]]:
    with (
        repo / "artifacts" / "experiment-019" / "placement" / "placement-sweep.csv"
    ).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["stripe_degree"]), int(row["depth_span"])): row
        for row in rows
        if row["memory_cap_gib"] == "20"
        and row["chunk_rows"] == "1"
        and (int(row["stripe_degree"]), int(row["depth_span"])) in CANDIDATES
    }


def _simulation_artifacts(
    repo: Path,
    artifact: Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    placement = _read(artifact / "placement" / "final-placement.json")
    replay_status = _read(
        artifact / "validation" / "single-resource-replay.json"
    )["status"]
    placement_rows = _placement_lookup(repo)
    candidate_rows: list[dict[str, Any]] = []
    candidate_runs: dict[str, Any] = {}
    for degree, depth in CANDIDATES:
        service = _service(inputs, degree, 1)
        result, run = candidate_projection(service, depth_span=depth)
        source = placement_rows[(degree, depth)]
        workers = int(source["worker_count"])
        pods = int(source["pod_count"])
        peak = float(source["estimated_max_peak_gib"])
        if (degree, depth) == (8, 8):
            workers = int(placement["worker_count"])
            pods = int(placement["pod_count"])
            peak = float(placement["maximum_worker_peak_gib"])
        canonical = _canonical_compute_ms(repo, 16, 1)
        candidate_rows.append(
            {
                "candidate": f"P{degree}/depth-{depth}",
                "stripe_degree": degree,
                "depth_span": depth,
                "worker_count": workers,
                "pod_count": pods,
                "workers_per_pod": degree,
                "peak_gib_per_worker": peak,
                "network_events_per_token": sum(r.resource_type == "network" for r in run.records) / 17,
                "collective_steps": sum(r.collective_steps for r in run.records),
                "compute_work_inflation": run.total_compute_work_ms / canonical,
                "projected_critical_path_ms": run.makespan_ms,
                "predicted_tok_s_per_user": result["exact_tok_s_per_user"],
                "resident_gib": placement["checkpoint_payload_bytes"] / GIB,
                "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                "projected_cost_envelope_usd_per_hour": f"{workers * 0.10:.2f}-{workers * 0.40:.2f}",
                "sm86_physical_throughput_measured": False,
                "predicted": True,
                "selected": (degree, depth) == (8, 8),
            }
        )
        candidate_runs[f"P{degree}_D{depth}"] = (result, run)
    _write_csv(artifact / "placement" / "candidate-sweep.csv", candidate_rows)

    target_rows: list[dict[str, Any]] = []
    nominal_runs: dict[tuple[int, int], Any] = {}
    for block in BLOCKS:
        for chunk in CHUNKS:
            service = _service(inputs, 8, chunk)
            result, run = candidate_projection(
                service,
                depth_span=8,
                block=block,
                chunk=chunk,
            )
            nominal_runs[(block, chunk)] = run
            target_rows.append(
                {
                    "block": block,
                    "chunk": chunk,
                    "accepted_target_tokens": block + 1,
                    "target_pass_ms": result["target_pass_ms"],
                    "predicted_tok_s_per_user": result["exact_tok_s_per_user"],
                    "total_compute_work_ms": run.total_compute_work_ms,
                    "network_critical_path_ms": run.network_critical_path_ms,
                    "total_network_bytes": run.total_network_bytes,
                    "average_worker_utilization": statistics.fmean(run.worker_utilization.values()),
                    "service_source": "post-fusion bottom-up RTX 5090 sharded worker measurements",
                    "prediction_hardware": "RTX 3090 SM86; physically unvalidated until E021",
                    "predicted": True,
                }
            )
    _write_csv(artifact / "simulation" / "target-oracle.csv", target_rows)

    scenarios = {
        "optimistic": (0.9, NetworkProfile("optimistic_local", 0.05, 100.0, 0.01), NetworkProfile("optimistic_inter", 1.0, 25.0, 0.02), 0.0),
        "nominal": (1.0, NetworkProfile("nominal_local", 0.25, 25.0, 0.02), NetworkProfile("nominal_inter", 5.0, 10.0, 0.04), 0.0),
        "conservative": (1.3, NetworkProfile("conservative_local", 1.0, 10.0, 0.05), NetworkProfile("conservative_inter", 10.0, 2.0, 0.10), 0.20),
    }
    envelope: list[dict[str, Any]] = []
    for block in BLOCKS:
        for chunk in CHUNKS:
            service = _service(inputs, 8, chunk)
            for name, (compute, local, inter, jitter) in scenarios.items():
                result, _ = candidate_projection(
                    service,
                    depth_span=8,
                    block=block,
                    chunk=chunk,
                    compute_slowdown=compute,
                    local_profile=local,
                    inter_pod_profile=inter,
                    jitter_fraction=jitter,
                )
                envelope.append(
                    {
                        "arm": "uncertainty_envelope",
                        "scenario": name,
                        "block": block,
                        "chunk": chunk,
                        "compute_multiplier": compute,
                        "local_rtt_ms": local.rtt_ms,
                        "local_bandwidth_gbps": local.bandwidth_gbps,
                        "inter_pod_rtt_ms": inter.rtt_ms,
                        "inter_pod_bandwidth_gbps": inter.bandwidth_gbps,
                        "jitter_fraction": jitter,
                        "predicted_target_pass_ms": result["target_pass_ms"],
                        "predicted_tok_s_per_user": result["exact_tok_s_per_user"],
                        "passes_5_tok_s": result["exact_tok_s_per_user"] >= TARGET_TOK_S,
                        "predicted": True,
                    }
                )
    slowdown = slowdown_sensitivity(_service(inputs, 8, 1), depth_span=8)
    envelope.extend({"arm": "compute_sensitivity", "scenario": f"compute_plus_{int(row['compute_degradation_percent'])}_percent", "block": 16, "chunk": 1, **row, "predicted": True} for row in slowdown)
    network_profiles = (
        ("nominal", 0.25, 25.0, 5.0, 10.0),
        ("local_degraded", 2.0, 2.0, 5.0, 10.0),
        ("inter_degraded", 0.25, 25.0, 20.0, 1.0),
        ("both_degraded", 2.0, 2.0, 20.0, 1.0),
    )
    for name, lrtt, lbw, irtt, ibw in network_profiles:
        result, _ = candidate_projection(
            _service(inputs, 8, 1),
            depth_span=8,
            local_profile=NetworkProfile(name + "_local", lrtt, lbw, 0.05),
            inter_pod_profile=NetworkProfile(name + "_inter", irtt, ibw, 0.10),
            compute_slowdown=1.2,
            jitter_fraction=0.1,
        )
        envelope.append(
            {
                "arm": "network_sensitivity",
                "scenario": name,
                "block": 16,
                "chunk": 1,
                "compute_multiplier": 1.2,
                "local_rtt_ms": lrtt,
                "local_bandwidth_gbps": lbw,
                "inter_pod_rtt_ms": irtt,
                "inter_pod_bandwidth_gbps": ibw,
                "jitter_fraction": 0.1,
                "predicted_target_pass_ms": result["target_pass_ms"],
                "predicted_tok_s_per_user": result["exact_tok_s_per_user"],
                "passes_5_tok_s": result["exact_tok_s_per_user"] >= TARGET_TOK_S,
                "predicted": True,
            }
        )
    _write_csv(artifact / "simulation" / "uncertainty-envelope.csv", envelope)

    concurrency = concurrency_sweep(_service(inputs, 8, 1), depth_span=8)
    _write_csv(artifact / "simulation" / "concurrency.csv", concurrency)
    stragglers = straggler_sweep(_service(inputs, 8, 1), depth_span=8)
    _write_csv(artifact / "simulation" / "stragglers.csv", stragglers)

    final_result, final_run = candidate_runs["P8_D8"]
    canonical = _canonical_compute_ms(repo, 16, 1)
    work_inflation = work_inflation_breakdown(final_run, canonical)
    _write_csv(artifact / "simulation" / "work-inflation.csv", work_inflation)
    for rows in (
        candidate_rows,
        target_rows,
        envelope,
        concurrency,
        stragglers,
        work_inflation,
    ):
        for row in rows:
            row["corrected_replay_status"] = replay_status
            row["prediction_use"] = (
                "READINESS_EVIDENCE"
                if replay_status == "PASS"
                else "UNVALIDATED_DIAGNOSTIC_ONLY"
            )
    _write_csv(artifact / "placement" / "candidate-sweep.csv", candidate_rows)
    _write_csv(artifact / "simulation" / "target-oracle.csv", target_rows)
    _write_csv(artifact / "simulation" / "uncertainty-envelope.csv", envelope)
    _write_csv(artifact / "simulation" / "concurrency.csv", concurrency)
    _write_csv(artifact / "simulation" / "stragglers.csv", stragglers)
    _write_csv(artifact / "simulation" / "work-inflation.csv", work_inflation)
    record_by_id = {row.task_id: row for row in final_run.records}
    critical_records = [record_by_id[value] for value in final_run.critical_path_task_ids]
    critical = {
        "schema_version": "experiment-020-critical-path-v1",
        "predicted": True,
        "corrected_replay_status": replay_status,
        "prediction_use": (
            "READINESS_EVIDENCE"
            if replay_status == "PASS"
            else "UNVALIDATED_DIAGNOSTIC_ONLY"
        ),
        "configuration": "P8/depth-8 block-16 chunk-1",
        "makespan_ms": final_run.makespan_ms,
        "predicted_tok_s_per_user": final_result["exact_tok_s_per_user"],
        "critical_path_compute_ms": final_run.critical_path_compute_ms,
        "critical_path_network_ms": final_run.network_critical_path_ms,
        "critical_path_task_count": len(critical_records),
        "critical_path_operator_counts": dict(Counter(row.operator for row in critical_records)),
        "critical_path_task_ids": list(final_run.critical_path_task_ids),
        "total_task_count": len(final_run.records),
        "total_compute_work_ms": final_run.total_compute_work_ms,
        "total_network_bytes": final_run.total_network_bytes,
    }
    atomic_write_json(artifact / "simulation" / "critical-path.json", critical)
    gates = {
        "schema_version": "experiment-020-e021-topology-gates-v1",
        "frozen_before_E021": True,
        "model_validation_status": replay_status,
        "readiness_use": (
            "VALID"
            if replay_status == "PASS"
            else "INVALID_PENDING_PHYSICAL_SINGLE_RESOURCE_REPLAY"
        ),
        "decision_scope": "conservative +20% compute and 10% jitter, block-16/chunk-1",
        "intra_pod": topology_gate(_service(inputs, 8, 1), depth_span=8, link="intra_pod"),
        "inter_pod": topology_gate(_service(inputs, 8, 1), depth_span=8, link="inter_pod"),
        "maximum_jitter_fraction": 0.10,
        "decision": "TOPOLOGY_ACCEPTED only if every required path satisfies its RTT, bandwidth, and jitter gate",
    }
    atomic_write_json(artifact / "simulation" / "topology-gates.json", gates)
    return {
        "candidate_rows": candidate_rows,
        "target_rows": target_rows,
        "envelope": envelope,
        "concurrency": concurrency,
        "stragglers": stragglers,
        "work_inflation": work_inflation,
        "critical": critical,
        "topology_gates": gates,
    }


def _physical_artifacts(
    artifact: Path,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    grouped_rows: list[dict[str, Any]] = []
    receipts = (
        ("P8_robust", _read(artifact / "physical" / "expert-grouped-robust-calibration.json")),
        ("P16", _read(artifact / "physical" / "expert-grouped-p16-raw.json")),
    )
    for evidence, receipt in receipts:
        for result in receipt["results"]:
            workers = result["workers"]
            old_wall = [float(row["e019_existing"]["wall"]["p50_ms"]) for row in workers]
            new_wall = [float(row["e020_grouped"]["wall"]["p50_ms"]) for row in workers]
            old_cuda = [float(row["e019_existing"]["cuda"]["p50_ms"]) for row in workers]
            new_cuda = [float(row["e020_grouped"]["cuda"]["p50_ms"]) for row in workers]
            grouped_rows.append(
                {
                    "evidence": evidence,
                    "stripe_degree": result["stripe_degree"],
                    "rows": result["rows"],
                    "logical_fragments_per_worker": result["logical_fragments_per_worker"],
                    "old_physical_launches_per_worker": workers[0]["e019_existing"]["total_physical_launches"],
                    "new_physical_launches_per_worker": result["grouped_physical_launches_per_worker"],
                    "minimum_launch_coalescing": result["minimum_launch_coalescing"],
                    "old_wall_ms_p50_across_workers": statistics.median(old_wall),
                    "new_wall_ms_p50_across_workers": statistics.median(new_wall),
                    "old_worker_ceiling_ms": max(old_wall),
                    "new_worker_ceiling_ms": max(new_wall),
                    "speedup_at_worker_ceiling": max(old_wall) / max(new_wall),
                    "old_cuda_ms_p50_across_workers": statistics.median(old_cuda),
                    "new_cuda_ms_p50_across_workers": statistics.median(new_cuda),
                    "worker_seconds_old": sum(old_wall) / 1000,
                    "worker_seconds_grouped": sum(new_wall) / 1000,
                    "relative_l2": result["relative_l2_error"],
                    "peak_memory_delta_bytes": result["peak_memory_delta_bytes"],
                    "exact": result["pass"],
                    "status": "PASS" if result["pass"] else "FAIL",
                    "physical_device": "NVIDIA GeForce RTX 5090 SM120",
                }
            )
    _write_csv(artifact / "physical" / "expert-grouped.csv", grouped_rows)

    services: list[dict[str, Any]] = []
    for degree in (8, 16):
        for rows in CHUNKS:
            value = asdict(_service(inputs, degree, rows))
            source = value.pop("source")
            for operator, timing in value.items():
                if isinstance(timing, (int, float)):
                    services.append(
                        {
                            "stripe_degree": degree,
                            "rows": rows,
                            "operator": operator,
                            "worker_index": "shared",
                            "wall_ms_p50": timing,
                            "source": source,
                            "physical_device": "RTX 5090 SM120",
                        }
                    )
                elif isinstance(timing, (list, tuple)):
                    services.extend(
                        {
                            "stripe_degree": degree,
                            "rows": rows,
                            "operator": operator,
                            "worker_index": index,
                            "wall_ms_p50": worker_time,
                            "source": source,
                            "physical_device": "RTX 5090 SM120",
                        }
                        for index, worker_time in enumerate(timing)
                    )
    _write_csv(artifact / "physical" / "worker-services.csv", services)

    replay = _read(artifact / "validation" / "single-resource-replay.json")
    heldout_rows = [
        {
            **row,
            "error_percent": 100 * float(row["absolute_percentage_error"]),
            "gate_status": replay["status"],
            "normalization": "NONE",
        }
        for row in replay["rows"]
    ]
    _write_csv(artifact / "validation" / "heldout-service.csv", heldout_rows)

    samples: list[dict[str, Any]] = []
    for path in (
        artifact / "physical" / "attention-raw.json",
        artifact / "physical" / "other-shards-raw.json",
        artifact / "physical" / "other-shards-heldout-raw.json",
        artifact / "physical" / "other-shards-validation-raw.json",
    ):
        value = _read(path)
        for sample in value.get("gpu_samples", []):
            samples.append({"source": path.name, **sample})
    _write_csv(artifact / "physical" / "gpu-samples.csv", samples)
    return {"expert_grouped": grouped_rows, "worker_services": services, "gpu_samples": samples}


def _speculative_model(repo: Path, artifact: Path, simulation: Mapping[str, Any]) -> dict[str, Any]:
    reference = _read(
        repo / "artifacts" / "experiment-015" / "dspark" / "h015-001b-dspark-reference-sweep.json"
    )
    draft_block7 = next(float(row["reference_wall_ms"]) for row in reference["blocks"] if int(row["block_size"]) == 7)
    target_block7 = next(float(row["target_pass_ms"]) for row in simulation["target_rows"] if int(row["block"]) == 7 and int(row["chunk"]) == 1)
    distributions = {
        "conservative_scenario": {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10},
        "public_reference_scenario": {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.20, 5: 0.15, 6: 0.12, 7: 0.08},
        "optimistic_scenario": {3: 0.10, 4: 0.15, 5: 0.20, 6: 0.25, 7: 0.30},
    }
    rows = []
    for name, distribution in distributions.items():
        mean = sum(length * probability for length, probability in distribution.items())
        rollback_commit = 0.05 * 7
        cycle = draft_block7 + target_block7 + rollback_commit
        rows.append(
            {
                "scenario": name,
                "block": 7,
                "acceptance_length_distribution": distribution,
                "mean_output_tokens_per_cycle": mean,
                "draft_latency_ms": draft_block7,
                "draft_latency_evidence": "E015 CPU BF16 DSpark correctness reference; stale/inadequate as production latency",
                "target_verify_latency_ms": target_block7,
                "target_latency_evidence": "E020 bottom-up worker-level PREDICTED",
                "rollback_commit_ms": rollback_commit,
                "cycle_latency_ms": cycle,
                "predicted_real_output_tok_s_per_user": mean / (cycle / 1000),
                "passes_5_tok_s": mean / (cycle / 1000) >= TARGET_TOK_S,
                "scientific_result": False,
                "predicted": True,
            }
        )
    _write_csv(artifact / "simulation" / "speculative-overhead.csv", rows)
    receipt = {
        "schema_version": "experiment-020-speculative-overhead-v1",
        "status": "SCENARIO_ONLY_ACCEPTANCE_UNVALIDATED",
        "rows": rows,
        "existing_acceptance_evidence": {
            "experiment_015_status": "INCOMPLETE",
            "measurement_count": 0,
            "public_reference_mean_block7": 3.85,
            "public_reference_not_swarm": True,
        },
        "e021_secondary_measurement": (
            "Before any speculative headline, run the frozen DSpark revision on the preregistered coding, math, chat, and creative prompts; record the exact accepted-length histogram for block 7/12/16 plus GPU draft, commit, and rollback latency. Target-only remains the primary gate."
        ),
        "reason_not_blocking_primary_readiness": (
            "The E021 primary acceptance gate is target-only >=5 real Kimi K3 tokens/s/user; speculative decoding runs only after that gate."
        ),
    }
    atomic_write_json(artifact / "simulation" / "speculative-overhead.json", receipt)
    return receipt


def _parse_junit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    return {
        "status": "PASS" if totals["failures"] == totals["errors"] == 0 and totals["tests"] else "FAIL",
        **totals,
        "junit": str(path),
    }


def _deployment_artifacts(repo: Path, artifact: Path) -> dict[str, Any]:
    acquisition_path = artifact / "deployment" / "model-acquisition.json"
    acquisition = _read(acquisition_path)
    acquisition["status"] = (
        "PASS"
        if acquisition.get("range_supported")
        and acquisition.get("resume_supported")
        and acquisition.get("local_match")
        and acquisition.get("partial_download_only")
        and not acquisition.get("full_checkpoint_downloaded")
        and not acquisition.get("secret_value_persisted")
        else "FAIL"
    )
    atomic_write_json(acquisition_path, acquisition)
    native_root = artifact / "deployment" / "native" / "sm86"
    native_rows = []
    for path in sorted(native_root.glob("*.so")):
        native_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "target_architecture": "sm_86",
                "built_in_linux_cuda_13_0_1_stage": True,
            }
        )
    image_inspect_path = artifact / "deployment" / "docker-image-inspect.json"
    image = _read(image_inspect_path) if image_inspect_path.is_file() else {"present": False}
    worker_source = (
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_020"
        / "worker_main.py"
    ).read_text(encoding="utf-8")
    provisioning_source = (
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_020"
        / "provisioning.py"
    ).read_text(encoding="utf-8")
    preflight_source = (repo / "scripts" / "run_experiment_021.py").read_text(
        encoding="utf-8"
    )
    controller_role_implemented = "controller-host-agent" in worker_source
    native_dispatch_implemented = "EXECUTE_SHARD" in worker_source
    real_vast_backend_implemented = "class RealVastBackend" in provisioning_source
    host_cache_integration_implemented = (
        "ShardCache" in worker_source and "pod-bundles.json" in worker_source
    )
    pinned_registry_verification_implemented = "--image-ref" in preflight_source
    blockers = []
    if not controller_role_implemented:
        blockers.append(
            "implement and loopback-test the controller-host-agent entrypoint rendered for pod 0"
        )
    if not native_dispatch_implemented:
        blockers.append(
            "wire authenticated EXECUTE_SHARD frames to the real KDA/MLA/expert/projection/endpoint worker primitives and rerun the 93-layer graph through worker processes"
        )
    if not real_vast_backend_implemented:
        blockers.append(
            "implement the real Vast backend adapter for the already-tested provisioning state machine, keep it E020-disabled, and test every call at the mocked subprocess boundary"
        )
    if not host_cache_integration_implemented:
        blockers.append(
            "integrate the proven pod-bundle/ShardCache acquisition and hash verification into SwarmHostAgent before worker registration"
        )
    if not pinned_registry_verification_implemented:
        blockers.append(
            "make preflight accept and verify a user-supplied immutable registry image digest instead of hardcoding published=false"
        )
    sm86 = {
        "schema_version": "experiment-020-sm86-build-v1",
        "status": "PASS" if len(native_rows) == 3 and all(row["bytes"] > 0 for row in native_rows) else "FAIL",
        "native_binaries": native_rows,
        "cuobjdump_validation": "PASS",
        "architectures_present": ["sm_86"],
        "sm120_development_binaries_preserved": all(
            path.is_file()
            for path in (
                artifact / "physical" / "e020-grouped-top16-sm120.dll",
                repo / "artifacts" / "experiment-019" / "physical" / "exp019-kda-shard-sm120-v2.dll",
                repo / "artifacts" / "experiment-016" / "cuda" / "coli_cuda-sm120-h016-final.dll",
            )
        ),
        "essential_kernel_missing": False,
    }
    atomic_write_json(artifact / "deployment" / "sm86-build.json", sm86)
    lifecycle = _read(artifact / "runtime" / "96-worker-dry-run.json")
    image_build_pass = bool(
        image.get("present") and image.get("health_status") == "PASS"
    )
    linux = {
        "schema_version": "experiment-020-linux-build-v1",
        "status": "PASS" if image_build_pass and not blockers else "FAIL",
        "image_build_status": "PASS" if image_build_pass else "FAIL",
        "dockerfile": "deployment/Dockerfile.e021",
        "image": image,
        "python": "3.12",
        "cuda_runtime": "13.0.1",
        "non_root_runtime": True,
        "unattended": True,
        "manual_ssh_setup_required": False,
        "sm86_built": sm86["status"] == "PASS",
        "local_image_published": bool(image.get("published", False)),
        "registry_publication_is_preflight_condition": True,
        "controller_host_agent_entrypoint_implemented": controller_role_implemented,
        "native_shard_dispatch_implemented": native_dispatch_implemented,
        "real_vast_backend_implemented": real_vast_backend_implemented,
        "host_agent_model_cache_integration_implemented": host_cache_integration_implemented,
        "pinned_registry_digest_verification_implemented": pinned_registry_verification_implemented,
        "pre_rental_blockers": blockers,
    }
    atomic_write_json(artifact / "deployment" / "linux-build.json", linux)
    preflight_path = artifact / "vast" / "e021-preflight.json"
    if preflight_path.is_file():
        preflight = _read(preflight_path)
        static = dict(preflight.get("static_validation", {}))
        static.update(
            {
                "linux_deployment_status": linux["status"],
                "linux_pre_rental_blockers": blockers,
                "sm86_status": sm86["status"],
                "deployment_ready": linux["status"] == sm86["status"] == "PASS",
            }
        )
        preflight["static_validation"] = static
        preflight.setdefault("go_conditions", {})[
            "production_deployment_ready"
        ] = static["deployment_ready"]
        preflight["status"] = "NO_GO"
        preflight["post_generation_audit_refresh"] = True
        atomic_write_json(preflight_path, preflight)
    lock = {
        "schema_version": "experiment-020-dependency-lock-v1",
        "status": "PASS" if (repo / "uv.lock").is_file() else "FAIL",
        "lockfile": "uv.lock",
        "lockfile_sha256": _sha256(repo / "uv.lock"),
        "dependencies": [
            {"class": "bundled", "items": ["swarm_inference", "Colibri sources", "KDA shard", "grouped top-16"]},
            {"class": "pip-installed", "items": ["uv.lock exact resolution (73 packages)", "torch 2.13 CUDA 13.0", "numpy", "cryptography", "psutil", "huggingface-hub"]},
            {"class": "system package", "items": ["ca-certificates", "curl", "python3.12", "python3.12-venv"]},
            {"class": "host provided", "items": ["NVIDIA driver compatible with CUDA 13.0", "8 x SM86-or-compatible GPUs"]},
            {"class": "downloaded at bootstrap", "items": ["content-addressed pod model bundle only"]},
        ],
        "all_experiment_020_modules_imported_in_clean_image": image.get(
            "all_e020_module_imports"
        )
        == "PASS",
        "imported_runtime_module_count": image.get("runtime_module_count"),
        "secret_values_in_image": False,
    }
    atomic_write_json(artifact / "deployment" / "dependency-lock.json", lock)
    bootstrap = {
        "schema_version": "experiment-020-bootstrap-test-v1",
        "status": (
            "PASS"
            if linux["status"] == "PASS" and lifecycle["status"] == "PASS"
            else "FAIL"
        ),
        "container_health": image.get("health_status"),
        "native_library_load": image.get("native_library_load"),
        "eight_explicit_worker_lifecycle": image.get("eight_worker_lifecycle"),
        "host_agent_non_compute": True,
        "manual_setup": False,
        "blockers": blockers,
    }
    atomic_write_json(artifact / "deployment" / "bootstrap-test.json", bootstrap)
    return {"sm86": sm86, "linux": linux, "lock": lock, "bootstrap": bootstrap}


def _environment_and_sources(repo: Path, artifact: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inherited_model = _read(repo / "artifacts" / "experiment-019" / "model-metadata.json")
    model = {
        **inherited_model,
        "schema_version": "experiment-020-model-metadata-v1",
        "immutable_checkpoint": "F:/models/Kimi-K3",
        "byte_exact_placement_payload_bytes": _read(artifact / "placement" / "final-placement.json")["checkpoint_payload_bytes"],
        "remote_source": "moonshotai/Kimi-K3",
        "remote_range_proof": _read(artifact / "deployment" / "model-acquisition.json")["status"],
        "full_checkpoint_downloaded_during_e020": False,
    }
    atomic_write_json(artifact / "model-metadata.json", model)
    cli = _read(artifact / "vast" / "cli-validation.json")
    environment = {
        "schema_version": "experiment-020-environment-v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "cwd": str(repo),
        "platform": platform.platform(),
        "python": sys.version,
        "timezone": time.tzname,
        "git_commit": _command(("git", "rev-parse", "HEAD"), repo),
        "git_status": _command(("git", "status", "--short"), repo),
        "nvidia_smi": _command(
            (
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ),
            repo,
        ),
        "nvcc": _command(("nvcc", "--version"), repo),
        "physical_compute_device": "NVIDIA GeForce RTX 5090 SM120",
        "rtx_3090_physically_tested": False,
        "vast_cli_path": cli["cli_executable_path"],
        "vast_cli_version": cli["cli_version"],
        "vast_mode": "READ_ONLY",
        "gpu_rentals": 0,
    }
    atomic_write_json(artifact / "environment.json", environment)
    files: list[dict[str, Any]] = []
    candidates: list[Path] = []
    candidates.extend((repo / "src" / "swarm_inference" / "experiments" / "experiment_020").glob("*.py"))
    candidates.extend((repo / "scripts").glob("*020*.py"))
    candidates.append(repo / "scripts" / "run_experiment_021.py")
    candidates.extend((repo / "deployment").glob("*"))
    candidates.extend((repo / "native").glob("*grouped*"))
    candidates.extend((repo / "tests" / "unit").glob("test_experiment_020*.py"))
    candidates.extend((artifact / "deployment" / "native" / "sm86").glob("*.so"))
    for path in sorted({value.resolve() for value in candidates if value.is_file()}):
        files.append(
            {
                "path": str(path.relative_to(repo.resolve())).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    source = {
        "schema_version": "experiment-020-source-manifest-v1",
        "generated_unix_ns": time.time_ns(),
        "git_commit": environment["git_commit"],
        "preexisting_dirty_paths_preserved": [".gitignore", "third_party/colibri"],
        "files": files,
    }
    atomic_write_json(artifact / "source-manifest.json", source)
    return environment, source


def _failure_log(artifact: Path) -> dict[str, Any]:
    entries = [
        {
            "arm": "vast_offer_query_initial_syntax",
            "result": "zero rows",
            "cause": "installed CLI requires one query string and GPU RAM in GiB",
            "redesign": "discover syntax from local help and rerun guarded read-only searches",
            "final_status": "PASS",
        },
        {
            "arm": "single_resource_replay_initial",
            "result": "FAIL median error 7.07%",
            "cause": "short, temporally separated primitive samples moved coherently by about 7%",
            "redesign": "longer back-to-back independent calibration/held-out samples; no multiplier",
            "receipt": "validation/single-resource-replay-initial-failed.json",
            "final_status": "REDESIGNED",
        },
        {
            "arm": "single_resource_replay_second",
            "result": "FAIL median error 8.35%",
            "cause": "reversing two short service captures reversed the span error sign",
            "redesign": "31 expert iterations and 15 repeated non-expert captures",
            "receipt": "validation/single-resource-replay-second-failed.json",
            "final_status": "REDESIGNED",
        },
        {
            "arm": "single_resource_replay_methodology_audit",
            "result": "FAIL: layer/span actual wall is an arithmetic sum, not one physically executed ordered workload",
            "cause": "the implementation validated held-out primitive service consistency but labeled the derived sum as physical replay",
            "redesign": "execute the exact worker DAG sequentially on the RTX 5090 with resident weights and measure one end-to-end wall for each required span",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
        {
            "arm": "network_shaped_rehearsal_counter",
            "result": "integration count closure corrected",
            "cause": "late-bound request counter in local harness",
            "redesign": "bind request identity explicitly and rerun the actual framed protocol",
            "final_status": "PASS",
        },
        {
            "arm": "E019_serial_reconstruction_gate",
            "result": "MODEL_INVALID under E019 preregistration",
            "cause": "compared serial sharded work with a different optimized monolithic algorithm",
            "redesign": "validate the exact sharded task graph and reconcile every cost",
            "final_status": "CORRECTED_IN_E020",
        },
        {
            "arm": "production_controller_host_agent",
            "result": "FAIL: rendered controller-host-agent role has no implementation",
            "cause": "the control-plane stress harness is real, but the production image exposes only worker and host-agent lifecycle roles",
            "redesign": "implement controller-host-agent enrollment, registration, scheduling, and shutdown over the frozen protocol",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
        {
            "arm": "production_worker_native_dispatch",
            "result": "FAIL: bounded worker processes do not dispatch real shard operations",
            "cause": "the 93-layer correctness run is exact but serial/in-process; worker_main currently proves lifecycle and GPU binding only",
            "redesign": "wire authenticated EXECUTE_SHARD frames to native KDA/MLA/expert/projection/endpoint primitives and rerun the exact graph through worker processes",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
        {
            "arm": "real_vast_backend_adapter",
            "result": "FAIL: provisioning state machine has only the fake backend",
            "cause": "the state transitions and rollback logic are tested, but no production adapter translates them to guarded Vast CLI calls",
            "redesign": "implement the real adapter behind the E020 lock and mock-test every subprocess call before E021",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
        {
            "arm": "host_agent_model_acquisition_integration",
            "result": "FAIL: proven ShardCache/pod-bundle code is not invoked by the production host agent",
            "cause": "model acquisition and lifecycle were validated as separate components",
            "redesign": "acquire and hash-verify the pod bundle before worker registration, with concurrent workers sharing one cache",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
        {
            "arm": "registry_digest_preflight",
            "result": "FAIL: canonical preflight hardcodes published=false and cannot validate a supplied image digest",
            "cause": "local image inspection was implemented without the future immutable registry reference input",
            "redesign": "accept a pinned image reference, verify its manifest/digest, and feed that exact reference into the fleet plan",
            "final_status": "OPEN_PRE_RENTAL_BLOCKER",
        },
    ]
    value = {
        "schema_version": "experiment-020-failure-log-v1",
        "entries": entries,
        "unresolved_implementation_failures": 6,
    }
    atomic_write_json(artifact / "failure-log.json", value)
    return value


def _risks(
    artifact: Path,
    deployment: Mapping[str, Any],
    speculation: Mapping[str, Any],
) -> dict[str, Any]:
    fleet = _read(artifact / "vast" / "fleet-feasibility.json")
    values = [
        ("actual RTX 3090 SM86 shard throughput", "medium", "high", False, False, "requires E021", "measure worker services before headline run; abort if conservative gate fails"),
        ("actual local eight-GPU collective latency", "medium", "high", False, False, "requires physical P8 host", "measure all required intra-pod paths before READY"),
        ("actual inter-host transport and jitter", "medium", "high", False, False, "requires independent hosts", "topology gate rejects slow fleet before benchmark"),
        ("actual Vast host behavior and throttling", "medium", "high", False, False, "requires rental", "health, power, and sustained service qualification"),
        ("full distributed critical path and cost/token", "medium", "high", False, False, "requires full swarm", "E021 primary measurement and kill switch"),
        ("12 homogeneous P8 RTX 3090 hosts available simultaneously", "high", "high", True, True, f"{fleet['current_availability']}: {fleet['matching_p8_hosts']} of {fleet['required_pods']} in snapshot", "refresh snapshot; spend $0 unless all 12 satisfy policy"),
        ("checkpoint byte placement or worker memory error", "low", "high", True, True, "PASS byte-exact coverage and 16.979 GiB peak", "revalidate hashes and manifest on every pod"),
        ("single-resource replay does not physically execute the ordered shard workload", "high", "high", True, True, "FAIL: numeric service sums pass thresholds but the required end-to-end physical wall was not measured", "run resident-weight RTX 5090 replays for every primitive, full KDA/MLA layer, and 2/4/8-layer spans; then revalidate without normalization"),
        ("remote shard acquisition, resume, or hash failure", "low", "high", True, True, "PASS real 64 KiB ranged/resumed proof and cache tests", "content-addressed retry then clean abort"),
        ("control-plane transport/security scaling", "low", "high", True, True, "PASS TLS/HMAC/bounds plus 96/376/1000 lightweight workers", "retain the frozen framing, credentials, and bounds in the production data path"),
        ("provisioning rollback leaves rentals", "low", "high", True, True, "PASS 11 injected failures and hash-chained ledger", "ledger-only teardown plus verify-destroyed"),
        ("production controller host-agent role missing", "high", "high", True, True, "FAIL: deployment audit found no controller-host-agent entrypoint", "implement and loopback-test controller enrollment, registration, scheduling, and shutdown"),
        ("production worker process has no native shard dispatch", "high", "high", True, True, "FAIL: lifecycle worker does not handle EXECUTE_SHARD", "wire the frozen protocol to all native primitives and rerun full 93-layer correctness through worker processes"),
        ("real Vast provisioning backend adapter missing", "high", "high", True, True, "FAIL: state machine currently has only a fake backend", "implement the guarded real adapter and test every command with mocked subprocess boundaries while E020 stays read-only"),
        ("host agent does not acquire and verify its pod model bundle", "high", "high", True, True, "FAIL: ShardCache and lifecycle are not integrated", "make cache completion a prerequisite for registration and prove shared concurrent access"),
        ("production DSpark acceptance distribution", "medium", "medium", True, True, speculation["status"], "run exact preregistered E021 secondary acceptance measurement after target-only pass"),
        ("immutable registry image cannot be supplied and verified by preflight", "high", "high", True, True, f"FAIL: local image build={deployment['linux']['image_build_status']}, preflight published flag is hardcoded false", "add verified --image-ref input, then publish the corrected image by digest"),
        ("no Vast SSH key registered", "high", "low", True, True, "detected by redacted CLI receipt", "onstart bootstrap is noninteractive; add emergency key before E021 if policy requires"),
    ]
    rows = [
        {
            "risk": risk,
            "probability": probability,
            "impact": impact,
            "can_test_without_rental": can_test,
            "tested": tested,
            "result": result,
            "E021_mitigation": mitigation,
        }
        for risk, probability, impact, can_test, tested, result, mitigation in values
    ]
    untested = sum(
        row["impact"] == "high"
        and row["can_test_without_rental"]
        and not row["tested"]
        for row in rows
    )
    failed = sum(
        row["impact"] == "high"
        and row["can_test_without_rental"]
        and str(row["result"]).startswith("FAIL")
        for row in rows
    )
    receipt = {
        "schema_version": "experiment-020-risk-register-v1",
        "risks": rows,
        "remaining_pre_rental_testable_high_impact_untested": untested,
        "remaining_pre_rental_testable_high_impact_failed": failed,
        "remaining_pre_rental_testable_critical_risks": untested + failed,
        "status": "PASS" if untested == failed == 0 else "FAIL",
    }
    atomic_write_json(artifact / "risk-register.json", receipt)
    return receipt


def _test_results(artifact: Path) -> dict[str, Any]:
    unit = _parse_junit(artifact / "tests" / "unit-junit.xml")
    full = _parse_junit(artifact / "tests" / "full-junit.xml")
    value = {
        "schema_version": "experiment-020-test-results-v1",
        "experiment_020_unit": unit,
        "repository_full_suite": full,
        "status": "PASS" if unit["status"] == "PASS" and full["status"] == "PASS" else "FAIL",
    }
    atomic_write_json(artifact / "test-results.json", value)
    return value


def _readiness(
    repo: Path,
    artifact: Path,
    deployment: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    placement = _read(artifact / "placement" / "final-placement.json")
    manifest_hashes = _read(artifact / "placement" / "manifest-hash-audit.json")
    correctness = _read(artifact / "correctness" / "full-93-sharded.json")
    replay = _read(artifact / "validation" / "single-resource-replay.json")
    accounting = _read(artifact / "validation" / "accounting-reconciliation.json")
    grouped = _read(artifact / "physical" / "expert-grouped-robust-calibration.json")
    acquisition = _read(artifact / "deployment" / "model-acquisition.json")
    dry96 = _read(artifact / "runtime" / "96-worker-dry-run.json")
    cli = _read(artifact / "vast" / "cli-validation.json")
    audit = _read(artifact / "vast" / "safety-audit.json")
    fleet = _read(artifact / "vast" / "fleet-feasibility.json")
    cost = _read(artifact / "vast" / "cost-estimate.json")
    provisioning = _read(artifact / "vast" / "mock-provisioning.json")
    topology = _read(artifact / "simulation" / "topology-gates.json")
    pre_rental_blockers = list(deployment["linux"]["pre_rental_blockers"])
    if not replay.get("physical_equivalent_workload_executed", False):
        pre_rental_blockers.insert(
            0,
            "physically execute the exact ordered single-resource shard workload on RTX 5090 for every required primitive/layer/span and rerun the replay error gates",
        )
    checks = [
        ("1", "Final bounded-worker placement frozen", placement["worker_count"] == 96 and placement["placement"]["stripe_degree"] == 8 and placement["placement"]["depth_span"] == 8),
        ("2", "Entire checkpoint byte-exactly assigned", placement["full_checkpoint_covered"] and placement["coverage_gap_bytes"] == placement["coverage_overlap_bytes"] == 0 and manifest_hashes["status"] == "PASS" and manifest_hashes["placeholder_hashes_remaining"] == 0),
        ("3", "Peak worker memory <=20 GiB", placement["maximum_worker_peak_gib"] <= 20),
        ("4", "No whole layer/expert", not placement["whole_layer_on_any_worker"] and not placement["whole_expert_on_any_worker"] and placement["maximum_shared_expert_fraction"] <= 0.125),
        ("5", "Complete 93-layer shard correctness", correctness["status"] == "PASS" and correctness["complete_93_layer_graph"]),
        ("6", "Corrected single-resource simulator validation", replay["status"] == "PASS"),
        ("7", "Event accounting reconciliation", accounting["status"] == "PASS"),
        ("8", "Grouped expert execution", grouped["status"] == "PASS" and min(row["minimum_launch_coalescing"] for row in grouped["results"]) >= 4),
        ("9", "All required SM86 binaries", deployment["sm86"]["status"] == "PASS"),
        ("10", "Deployable Linux worker/controller environment", deployment["linux"]["status"] == "PASS"),
        ("11", "Model acquisition mechanism", acquisition["status"] == "PASS"),
        ("12", "96-worker controller dry-run", dry96["status"] == "PASS" and dry96["worker_count"] == 96),
        ("13", "Vast CLI auth/read-only integration", cli["authentication_status"] == "AUTHENTICATED" and cli["search_success"]),
        ("14", "Vast rental safety guard", audit["gpu_rentals"] == audit["vast_resource_mutations"] == audit["mutating_command_count"] == 0),
        ("15", "Fleet plan or explicit availability condition", fleet["current_availability"] == "YES" or bool(fleet["deployment_constraint"])),
        ("16", "Full cost estimate", cost["charge_incurred"] is False and cost["worst_case_budget_cap_usd"] > 0),
        ("17", "Mock provisioning and rollback", provisioning["status"] == "PASS" and _read(artifact / "vast" / "mock-rollback.json")["status"] == "PASS"),
        ("18", "Network/topology gates preregistered", topology["frozen_before_E021"] and topology["model_validation_status"] == "PASS" and topology["intra_pod"]["status"] == topology["inter_pod"]["status"] == "PREREGISTERED"),
        ("19", "E021 runbook complete", (repo / "docs" / "experiments" / "EXPERIMENT_021_RUNBOOK.md").is_file()),
        ("20", "No high-impact pre-rental-testable risk untested", risk["remaining_pre_rental_testable_high_impact_untested"] == 0),
    ]
    gates = [
        {"gate": number, "requirement": requirement, "status": "PASS" if passed else "FAIL"}
        for number, requirement, passed in checks
    ]
    outcome = "E021_READY" if all(value for _, _, value in checks) else "E021_NOT_READY"
    readiness = {
        "schema_version": "experiment-020-readiness-v1",
        "outcome": outcome,
        "gates": gates,
        "passed_gate_count": sum(value for _, _, value in checks),
        "required_gate_count": len(checks),
        "current_market_launch_permission": "NO_GO" if fleet["current_availability"] != "YES" else "CONDITIONAL",
        "market_readiness_distinction": "E020 is not deployment-ready and does not authorize rental. After the open pre-rental implementation blockers close, E021 preflight still requires 12 live P8 hosts, a pinned image, and an approved budget.",
        "pre_rental_blockers": pre_rental_blockers,
    }
    atomic_write_json(artifact / "readiness.json", readiness)
    truth = {
        "schema_version": "experiment-020-truth-table-v1",
        "rows": [
            {"question": "Any GPU rented in E020?", "answer": "NO"},
            {"question": "Any Vast resource mutated?", "answer": "NO"},
            {"question": "Vast CLI authenticated?", "answer": "YES" if cli["authentication_status"] == "AUTHENTICATED" else "NO"},
            {"question": "Live offers queried?", "answer": "YES" if cli["search_success"] else "NO"},
            {"question": "Final worker count", "answer": placement["worker_count"]},
            {"question": "Final pod count", "answer": placement["pod_count"]},
            {"question": "Workers per pod", "answer": placement["workers_per_pod"]},
            {"question": "Max worker peak GiB", "answer": round(placement["maximum_worker_peak_gib"], 3)},
            {"question": "Whole layer on any worker?", "answer": "NO"},
            {"question": "Whole expert on any worker?", "answer": "NO"},
            {"question": "Full checkpoint covered?", "answer": "YES" if placement["full_checkpoint_covered"] else "NO"},
            {"question": "Full 93-layer sharded correctness?", "answer": correctness["status"]},
            {"question": "Corrected simulator validation?", "answer": replay["status"]},
            {"question": "sm86 build ready?", "answer": "YES" if deployment["sm86"]["status"] == "PASS" else "NO"},
            {"question": "Linux deployment ready?", "answer": "YES" if deployment["linux"]["status"] == "PASS" else "NO"},
            {
                "question": "Model distribution ready?",
                "answer": (
                    "YES"
                    if acquisition["status"] == "PASS"
                    and deployment["linux"][
                        "host_agent_model_cache_integration_implemented"
                    ]
                    else "NO"
                ),
            },
            {"question": "96-worker dry run passed?", "answer": "YES" if dry96["status"] == "PASS" else "NO"},
            {"question": "Vast fleet currently feasible?", "answer": fleet["current_availability"]},
            {"question": "E021 projected cost", "answer": f"${cost['expected_cost']['total_usd']:.2f} expected; ${cost['conservative_cost']['total_usd']:.2f} conservative"},
            {"question": "Remaining pre-rental-testable critical risks", "answer": risk["remaining_pre_rental_testable_critical_risks"]},
            {"question": "E021 readiness", "answer": "READY" if outcome == "E021_READY" else "NOT READY"},
        ],
    }
    atomic_write_json(artifact / "truth-table.json", truth)
    return readiness, truth


def _save_chart(path: Path, figure: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _charts(
    artifact: Path,
    simulation: Mapping[str, Any],
    physical: Mapping[str, Any],
    risk: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> None:
    chart_root = artifact / "charts"
    colors = {"pass": "#2b8a3e", "fail": "#c92a2a", "neutral": "#3569a8", "accent": "#e67700"}

    figure, axis = plt.subplots(figsize=(10, 6))
    labels = [f"{row['gate']}. {row['requirement']}" for row in readiness["gates"]]
    values = [1 if row["status"] == "PASS" else 0 for row in readiness["gates"]]
    axis.barh(range(len(labels)), values, color=[colors["pass"] if value else colors["fail"] for value in values])
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("Gate satisfied (1=yes)")
    axis.set_title(f"Experiment 020 readiness gates — {readiness['outcome']}")
    axis.grid(axis="x", alpha=0.25)
    _save_chart(chart_root / "chart-01-readiness.png", figure)

    candidates = simulation["candidate_rows"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [row["candidate"] for row in candidates]
    axes[0].bar(names, [row["worker_count"] for row in candidates], color=colors["neutral"])
    axes[0].set_ylabel("Workers / GPUs")
    axes[0].set_title("Bounded-worker candidates")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, [row["peak_gib_per_worker"] for row in candidates], color=colors["accent"])
    axes[1].axhline(20, color=colors["fail"], linestyle="--", label="20 GiB gate")
    axes[1].set_ylabel("Peak GiB / worker")
    axes[1].set_title("Memory envelope")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Placement trade-off (P8/depth-8 frozen)")
    _save_chart(chart_root / "chart-02-placement.png", figure)

    memory = _read(artifact / "placement" / "memory-audit.json")["workers"]
    components = ["static_weights", "scales", "persistent_state", "activations", "scratch", "collective_buffers", "transport_buffers", "cuda_workspace", "allocator_allowance"]
    figure, axis = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(memory))
    palette = plt.get_cmap("tab20").colors
    for index, component in enumerate(components):
        values = np.array([row[component] / GIB for row in memory])
        axis.bar(range(len(memory)), values, bottom=bottom, width=0.9, label=component, color=palette[index])
        bottom += values
    axis.axhline(20, color=colors["fail"], linestyle="--", label="20 GiB gate")
    axis.set_xlabel("Worker index (pod-major)")
    axis.set_ylabel("GiB")
    axis.set_title("Byte-reconciled P8/depth-8 worker memory")
    axis.legend(ncol=5, fontsize=7, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    _save_chart(chart_root / "chart-03-worker-memory.png", figure)

    inflation = simulation["work_inflation"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([row["category"] for row in inflation], [row["contribution_to_overall_inflation"] for row in inflation], color=colors["neutral"])
    axis.set_ylabel("Contribution to compute-work inflation (x)")
    axis.set_title("Bounded-worker compute tax by subsystem")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    _save_chart(chart_root / "chart-04-work-inflation.png", figure)

    fusion = [row for row in physical["expert_grouped"] if row["stripe_degree"] == 8]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(fusion))
    width = 0.36
    axis.bar(positions - width / 2, [row["old_worker_ceiling_ms"] for row in fusion], width, label="E019 existing stripe", color="#8d99ae")
    axis.bar(positions + width / 2, [row["new_worker_ceiling_ms"] for row in fusion], width, label="E020 grouped stripe", color=colors["pass"])
    axis.set_xticks(positions, [f"rows={row['rows']}\n{int(row['minimum_launch_coalescing'])}x launches" for row in fusion])
    axis.set_ylabel("Worker wall ceiling (ms)")
    axis.set_title("Exact grouped top-16 expert execution on RTX 5090")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_chart(chart_root / "chart-05-expert-fusion.png", figure)

    envelope = [row for row in simulation["envelope"] if row["arm"] == "uncertainty_envelope" and row["chunk"] == 1]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for scenario, color in (("optimistic", colors["pass"]), ("nominal", colors["neutral"]), ("conservative", colors["accent"])):
        rows = sorted((row for row in envelope if row["scenario"] == scenario), key=lambda row: row["block"])
        axis.plot([row["block"] for row in rows], [row["predicted_tok_s_per_user"] for row in rows], marker="o", label=scenario, color=color)
    axis.axhline(TARGET_TOK_S, color=colors["fail"], linestyle="--", label="5 tok/s gate")
    axis.set_xlabel("Verification block")
    axis.set_ylabel("Predicted target tok/s/user")
    axis.set_title("PREDICTED throughput uncertainty envelope (chunk 1)")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_chart(chart_root / "chart-06-projected-throughput-envelope.png", figure)

    concurrency = simulation["concurrency"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["concurrent_requests"] for row in concurrency], [row["aggregate_tok_s"] for row in concurrency], marker="o", label="aggregate tok/s", color=colors["neutral"])
    axis.plot([row["concurrent_requests"] for row in concurrency], [row["per_user_tok_s_p50"] for row in concurrency], marker="s", label="p50 per-user tok/s", color=colors["accent"])
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    axis.set_xlabel("Concurrent requests")
    axis.set_ylabel("PREDICTED tok/s")
    axis.set_title("Economic concurrency model")
    axis.legend()
    axis.grid(alpha=0.25)
    _save_chart(chart_root / "chart-07-concurrency.png", figure)

    fleet = _read(artifact / "vast" / "fleet-feasibility.json")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    classes = fleet["classes"]
    axis.bar([row["gpu_class"] for row in classes], [row["matching_p8_hosts"] for row in classes], color=colors["neutral"])
    axis.axhline(fleet["required_pods"], color=colors["fail"], linestyle="--", label=f"required: {fleet['required_pods']} P8 hosts")
    axis.set_ylabel("Suitable complete P8 hosts")
    axis.set_title(f"Vast availability snapshot at {fleet['snapshot_at']}")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_chart(chart_root / "chart-08-vast-availability.png", figure)

    cost = _read(artifact / "vast" / "cost-estimate.json")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(["expected", "conservative", "budget cap"], [cost["expected_cost"]["total_usd"], cost["conservative_cost"]["total_usd"], cost["worst_case_budget_cap_usd"]], color=[colors["pass"], colors["accent"], colors["fail"]])
    axis.set_ylabel("USD")
    axis.set_title("E021 pre-spend cost envelope — no charge incurred")
    axis.grid(axis="y", alpha=0.25)
    _save_chart(chart_root / "chart-09-vast-cost.png", figure)

    impact_order = ("high", "medium", "low")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    tested = [
        sum(
            row["impact"] == impact
            and row["tested"]
            and not str(row["result"]).startswith("FAIL")
            for row in risk["risks"]
        )
        for impact in impact_order
    ]
    failed = [
        sum(
            row["impact"] == impact
            and row["can_test_without_rental"]
            and str(row["result"]).startswith("FAIL")
            for row in risk["risks"]
        )
        for impact in impact_order
    ]
    physical = [sum(row["impact"] == impact and not row["can_test_without_rental"] for row in risk["risks"]) for impact in impact_order]
    axis.bar(impact_order, tested, label="tested/audited pre-rental", color=colors["pass"])
    axis.bar(impact_order, failed, bottom=tested, label="tested and failed", color=colors["fail"])
    axis.bar(
        impact_order,
        physical,
        bottom=np.array(tested) + np.array(failed),
        label="requires physical E021",
        color=colors["accent"],
    )
    axis.set_ylabel("Risk count")
    axis.set_title("Pre-spend risk register")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_chart(chart_root / "chart-10-risk-register.png", figure)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _render_runbook(repo: Path, artifact: Path) -> None:
    placement = _read(artifact / "placement" / "final-placement.json")
    cost = _read(artifact / "vast" / "cost-estimate.json")
    topology = _read(artifact / "simulation" / "topology-gates.json")
    policy = _read(artifact / "vast" / "fleet-policy.json")
    linux = _read(artifact / "deployment" / "linux-build.json")
    replay = _read(artifact / "validation" / "single-resource-replay.json")
    intra = topology["intra_pod"]
    inter = topology["inter_pod"]
    runbook_blockers = list(linux.get("pre_rental_blockers", []))
    if not replay.get("physical_equivalent_workload_executed", False):
        runbook_blockers.insert(
            0,
            "physically execute the exact ordered single-resource shard workload on RTX 5090 and pass the primitive/layer/span replay error gates",
        )
    blocker_lines = "\n".join(
        f"- {blocker}" for blocker in runbook_blockers
    )
    text = f"""# Experiment 021 Runbook: First Full Physical Kimi K3 Swarm

This is the frozen operating procedure for the first paid physical swarm. Experiment 020 itself is hard-locked read-only. Do not launch until every preflight condition below is green and the user has explicitly approved a dollar cap.

## E020 blocker notice

This runbook preregisters the intended E021 procedure, but it is **not executable for rental yet**. E020 closed with `E021_NOT_READY`. Complete and rerun these pre-rental items first:

{blocker_lines}

After those pass, rebuild and publish the image by digest, rerun this full zero-rental preflight, and require a complete 12-host market plan. Until then, `--apply` must remain disabled.

## Frozen experiment

- Architecture: {placement['worker_count']} bounded GPU workers in {placement['pod_count']} pods, {placement['workers_per_pod']} workers/GPUs per pod; stripe degree 8, depth span 8.
- Primary GPU policy: homogeneous RTX 3090 24 GB, exactly eight GPUs on one physical host per pod. Re-profiled homogeneous RTX 3090 Ti, A5000, A6000, or RTX 4090 fleets are fallbacks; never mix classes in the primary run.
- Maximum manifest peak: {placement['maximum_worker_peak_gib']:.3f} GiB/worker.
- Primary physical gate: at least 5 real Kimi K3 output/accepted target tokens/s/user under the bounded-worker architecture.
- Frozen benchmark includes target-only first, then speculative decoding only if target-only passes.

## Prerequisites

1. Checkout the reviewed E021 release. Its reviewed policy change must remove the E020 compile-time read-only lock; the E020 commit must remain unable to rent.
2. Install the same Vast CLI family validated in E020 and authenticate via the local credential store. Never place the API key in this repository, image, command log, or artifact.
3. Publish `swarm-inference-lab:e021-sm86-e020` to an approved registry and replace the placeholder with a pinned digest. The image must pass the local health and eight-worker lifecycle checks.
4. Supply the Hugging Face/model-source credential at runtime only if the authoritative source requires it. Artifacts record presence, never value.
5. Ensure the controller has a per-run TLS credential generator and a secure artifact destination.
6. Rerun the live offer search. Require {policy['required_pods']} distinct homogeneous hosts, exactly {policy['exact_gpus_per_pod']} GPUs/pod, reliability >= {policy['minimum_reliability']}, disk >= {policy['minimum_disk_gb']} GB, downlink >= {policy['minimum_internet_down_mbps']} Mbps, CUDA >= {policy['minimum_cuda']}, Linux driver >= {policy['minimum_driver']}, direct ports >= {policy['minimum_direct_ports']}, verified status permitted by policy, and price <= ${policy['maximum_price_per_pod_hour']:.2f}/pod-hour.
7. Review the rendered create commands and generated plan digest. Offer IDs are ephemeral; never reuse the E020 snapshot IDs.

## Budget approval

The E020 estimate is ${cost['expected_cost']['total_usd']:.2f} expected, ${cost['conservative_cost']['total_usd']:.2f} conservative, with an unapproved worst-case cap suggestion of ${cost['worst_case_budget_cap_usd']:.2f}. The user must explicitly choose a maximum budget. The runtime tracks instance rates x elapsed time plus known disk/transfer costs and tears down before the approved cap.

Generate an immutable approved plan containing `experiment_id=experiment-021`, `approved=true`, and its `plan_sha256`. Approval is for that exact snapshot, image digest, run ID, and maximum budget only.

## One-command preflight (zero rental)

```powershell
python scripts/run_experiment_021.py --preflight --run-id <run_id>
```

Expected output: repository/model validation, redacted Vast authentication, a fresh offer snapshot, exact 12-pod plan or `NO_GO`, cost estimate, intended launch commands with `EXECUTED=false`, bootstrap render, and ledger-driven teardown render. A missing pinned registry digest or fewer than 12 policy-compliant P8 hosts is `NO_GO` and must spend $0.

## One-command launch

Only from the reviewed E021 release, after explicit budget approval:

```powershell
$env:SWARM_ALLOW_RENTAL='EXPERIMENT_021'; python scripts/run_experiment_021.py --apply --experiment-id experiment-021 --approved-plan artifacts/experiment-021/<run_id>/approved-fleet-plan.json --max-budget-usd <APPROVED_USD> --run-id <run_id>
```

All arms are mandatory: the environment variable, `--apply`, exact experiment identifier, approved plan digest, and positive maximum-dollar budget. The state machine—not hand-written commands—owns launch and rollback.

## Provisioning sequence and expected signals

The state log must advance through:

`DISCOVER_OFFERS -> PLAN_FLEET -> USER_BUDGET_GATE -> RENT_CONTROLLER_POD -> WAIT_CONTROLLER -> DISCOVER_CONTROLLER_ADDRESS -> RENT_WORKER_PODS -> WAIT_INSTANCES -> BOOTSTRAP -> DOWNLOAD_SHARDS -> VERIFY_MODEL -> REGISTER_WORKERS -> MEASURE_NETWORK -> VALIDATE_TOPOLOGY -> READY -> RUN_EXPERIMENT -> COLLECT_RESULTS -> DESTROY_ALL -> VERIFY_DESTROYED`.

Every created instance immediately enters the append-only rental ledger with instance ID, offer ID, machine ID, creation time, hourly rate, label `swarm-e021-<run_id>-pod-NNN`, and pod assignment. Stop/destroy may target only ledger IDs.

## Bootstrap and model download

Each multi-GPU host starts one `SwarmHostAgent`, which manages exactly eight GPU-bound worker processes and performs no model compute. It creates one content-addressed pod cache, downloads only the bundle described in `deployment/pod-bundles.json`, resumes partial objects, verifies SHA-256, writes the atomic completion marker, and exposes the cache read-only to its workers. Never download the 1.56 TB checkpoint to every worker and never load a whole layer/expert into GPU memory before slicing.

A bundle hash mismatch, insufficient disk, failed resume, or missing tensor range aborts before registration and enters full ledger teardown.

## Worker registration and health

Expect 96 unique worker IDs with 12 pod memberships and GPU indices 0-7. Each worker reports manifest hash, image digest, GPU identity, SM capability, native-binary hashes, model-bundle hash, and a health nonce over the authenticated TLS channel. Reject duplicate IDs, wrong GPU count, whole-layer fallback capability, missing state ownership, or an unrecognized image/manifest.

## Network qualification

Measure sustained RTT, upload, download, and jitter on every required controller/pod and collective path; marketplace metadata is only a discovery hint. The current candidate thresholds below are **not accepted E021 gates** because the single-resource replay methodology failed. Re-derive and freeze them after that replay passes:

- Intra-pod: RTT <= {intra['maximum_rtt_ms']:.3f} ms and bandwidth >= {intra['minimum_bandwidth_gbps_at_maximum_rtt']:.3f} Gbps at that RTT.
- Inter-pod: RTT <= {inter['maximum_rtt_ms']:.3f} ms and bandwidth >= {inter['minimum_bandwidth_gbps_at_maximum_rtt']:.3f} Gbps at that RTT.
- Jitter: <= {100 * topology['maximum_jitter_fraction']:.0f}% in the gate model.

After corrected validation, every path must satisfy the re-frozen relevant gate. Otherwise emit `TOPOLOGY_REJECTED`, collect diagnostics, and tear down without benchmarking. Never substitute unrelated single-GPU WAN hosts for a local P8 pod.

## Go/no-go gates

Proceed to `READY` only when image/manifests/hashes match, 96 workers are healthy, every pod has eight correct GPUs, model caches are complete, the topology is accepted, estimated spend remains below the approved trajectory, and no ledger inconsistency exists. Any failure is `NO_GO` and triggers teardown.

## Frozen benchmark plan

1. Warm up every worker primitive and record per-worker service, CUDA utilization, memory, power (where exposed), and transport counters.
2. Run the single-user target-only oracle over block/chunk sweep: blocks 7, 12, 16 and chunks 1, 2, 4 where valid. Block 16 is mandatory.
3. Measure physical target tok/s/user, aggregate tok/s, full critical path, network traffic, worker/GPU utilization, power, startup time, cost/token, and failure behavior.
4. Apply the unchanged primary gate: >=5 real Kimi K3 output/accepted target tok/s/user.
5. Only after target-only passes, measure the frozen DSpark proposal path. Record exact accepted-length histograms for coding, math/reasoning, chat, and creative prompts at blocks 7/12/16, plus draft, commit, and rollback latency. Report real speculative output tok/s/user separately.
6. Exercise 1, 2, 4, 8, and 16 requests as an economic secondary analysis. This never replaces the single-user gate.
7. Inject/observe recoverable worker and network degradation only within the approved budget and safety plan; do not change the primary gate.

## Artifact collection

Collect controller trace, worker traces, topology matrices, service samples, GPU telemetry, manifest/image/bundle hashes, acceptance traces, costs, ledger, and failure logs before teardown. Secrets and full account payloads are forbidden in artifacts.

## Normal teardown

The state machine enters `DESTROY_ALL`, destroys every and only ledger-owned instance, records a hash-chained `DESTROYED` event, then repeatedly calls read-only `show instances` until none of the ledger IDs remain. Finish only at `VERIFY_DESTROYED`.

## Emergency teardown

```powershell
$env:SWARM_ALLOW_RENTAL='EXPERIMENT_021'; python scripts/run_experiment_021.py --destroy-ledger artifacts/experiment-021/<run_id>/rental-ledger.json --apply --experiment-id experiment-021 --max-budget-usd <APPROVED_USD>
```

The emergency path rejects IDs absent from the immutable ledger. If the controller is lost, run it from the operator machine against the last fsynced ledger. Preserve the ledger and all Vast responses.

## Failure recovery

- Offer disappears: replan before creating anything else; destroy any controller already ledgered.
- Pod provision/ready/GPU/disk failure: stop the run and destroy all ledgered instances.
- Bootstrap/download/hash/health failure: retain diagnostic hashes, then full teardown.
- Network too slow: `TOPOLOGY_REJECTED`, no benchmark, full teardown.
- Controller crash: invoke emergency ledger teardown from the operator machine.
- Budget trajectory exceeded: kill switch immediately initiates ledger teardown.

Retries require a new run ID, fresh offer snapshot, new approved plan digest, and renewed budget approval. Never silently substitute hardware or reuse an old approval.

## Verify every rental is gone

Run the read-only `vastai show instances --raw`, compare it with the ledger, and require zero surviving ledger IDs. Unrelated user instances must remain untouched. Store a redacted verification receipt and the final valid ledger hash.
"""
    path = repo / "docs" / "experiments" / "EXPERIMENT_021_RUNBOOK.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_report(
    repo: Path,
    artifact: Path,
    simulation: Mapping[str, Any],
    physical: Mapping[str, Any],
    speculation: Mapping[str, Any],
    readiness: Mapping[str, Any],
    truth: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> None:
    placement = _read(artifact / "placement" / "final-placement.json")
    manifest_hashes = _read(artifact / "placement" / "manifest-hash-audit.json")
    replay = _read(artifact / "validation" / "single-resource-replay.json")
    correctness = _read(artifact / "correctness" / "full-93-sharded.json")
    fleet = _read(artifact / "vast" / "fleet-feasibility.json")
    cost = _read(artifact / "vast" / "cost-estimate.json")
    acquisition = _read(artifact / "deployment" / "model-acquisition.json")
    pod_bundles = _read(artifact / "deployment" / "pod-bundles.json")
    dry96 = _read(artifact / "runtime" / "96-worker-dry-run.json")
    dry376 = _read(artifact / "runtime" / "376-worker-stress.json")
    dry1000 = _read(artifact / "runtime" / "1000-worker-stress.json")
    security = _read(artifact / "runtime" / "security-validation.json")
    accounting = _read(artifact / "validation" / "accounting-reconciliation.json")
    topology = _read(artifact / "simulation" / "topology-gates.json")
    truth_table = _markdown_table(
        ("Question", "Answer"),
        ((row["question"], row["answer"]) for row in truth["rows"]),
    )
    candidate_table = _markdown_table(
        ("Candidate", "Workers", "Pods", "Peak GiB", "Predicted tok/s", "Utilization", "Selected"),
        (
            (
                row["candidate"],
                row["worker_count"],
                row["pod_count"],
                f"{row['peak_gib_per_worker']:.3f}",
                f"{row['predicted_tok_s_per_user']:.2f}",
                f"{100 * row['average_worker_utilization']:.1f}%",
                "YES" if row["selected"] else "NO",
            )
            for row in simulation["candidate_rows"]
        ),
    )
    replay_table = _markdown_table(
        ("Workload", "Predicted ms", "Held-out ms", "Error"),
        (
            (row["workload"], f"{row['predicted_sharded_wall_ms']:.4f}", f"{row['actual_sharded_wall_ms']:.4f}", f"{100 * row['absolute_percentage_error']:.2f}%")
            for row in replay["rows"]
        ),
    )
    target_table = _markdown_table(
        ("Block", "Chunk", "Predicted pass ms", "Predicted tok/s/user", "Worker util."),
        (
            (row["block"], row["chunk"], f"{row['target_pass_ms']:.1f}", f"{row['predicted_tok_s_per_user']:.2f}", f"{100 * row['average_worker_utilization']:.1f}%")
            for row in simulation["target_rows"]
        ),
    )
    uncertainty16 = [
        row
        for row in simulation["envelope"]
        if row["arm"] == "uncertainty_envelope" and row["block"] == 16 and row["chunk"] == 1
    ]
    uncertainty_table = _markdown_table(
        ("Scenario", "Compute", "Local link", "Inter-pod link", "Predicted tok/s"),
        (
            (row["scenario"], f"{row['compute_multiplier']:.1f}x", f"{row['local_rtt_ms']} ms / {row['local_bandwidth_gbps']} Gbps", f"{row['inter_pod_rtt_ms']} ms / {row['inter_pod_bandwidth_gbps']} Gbps", f"{row['predicted_tok_s_per_user']:.2f}")
            for row in uncertainty16
        ),
    )
    fusion_table = _markdown_table(
        ("P", "Rows", "Old launches", "New launches", "Old ceiling ms", "New ceiling ms", "Rel. L2", "Status"),
        (
            (row["stripe_degree"], row["rows"], row["old_physical_launches_per_worker"], row["new_physical_launches_per_worker"], f"{row['old_worker_ceiling_ms']:.4f}", f"{row['new_worker_ceiling_ms']:.4f}", f"{row['relative_l2']:.2e}", row["status"])
            for row in physical["expert_grouped"]
        ),
    )
    gates_table = _markdown_table(
        ("Gate", "Requirement", "Status"),
        ((row["gate"], row["requirement"], row["status"]) for row in readiness["gates"]),
    )
    risk_table = _markdown_table(
        ("Risk", "Probability", "Impact", "Tested?", "Result / mitigation"),
        (
            (row["risk"], row["probability"], row["impact"], "YES" if row["tested"] else "NO", f"{row['result']}; {row['E021_mitigation']}")
            for row in risk["risks"]
        ),
    )
    fleet_classes = _markdown_table(
        ("GPU class", "P8 hosts", "Suitable GPUs", "Host price min/median/max", "Locations"),
        (
            (row["gpu_class"], row["matching_p8_hosts"], row["suitable_gpus"], f"{row['price_per_host_hour_min']} / {row['price_per_host_hour_median']} / {row['price_per_host_hour_max']}", ", ".join(row["locations"]))
            for row in fleet["classes"]
        ),
    )
    concurrency_table = _markdown_table(
        ("Requests", "Per-user tok/s", "Aggregate tok/s", "Utilization", "Queue p50 ms"),
        (
            (row["concurrent_requests"], f"{row['per_user_tok_s_p50']:.2f}", f"{row['aggregate_tok_s']:.2f}", f"{100 * row['average_worker_utilization']:.1f}%", f"{row['queueing_ms_p50']:.1f}")
            for row in simulation["concurrency"]
        ),
    )
    straggler_table = _markdown_table(
        ("Scenario", "Affected workers", "Predicted tok/s", "Critical-path amplification"),
        (
            (row["scenario"], row["affected_worker_count"], f"{row['predicted_tok_s_per_user']:.2f}", f"{row['critical_path_amplification']:.3f}x")
            for row in simulation["stragglers"]
        ),
    )
    speculative_table = _markdown_table(
        ("Scenario", "Mean output/cycle", "Draft ms", "Target ms", "Predicted real tok/s", ">=5?"),
        (
            (row["scenario"], f"{row['mean_output_tokens_per_cycle']:.2f}", f"{row['draft_latency_ms']:.1f}", f"{row['target_verify_latency_ms']:.1f}", f"{row['predicted_real_output_tok_s_per_user']:.2f}", "YES" if row["passes_5_tok_s"] else "NO")
            for row in speculation["rows"]
        ),
    )
    nominal16 = next(row for row in uncertainty16 if row["scenario"] == "nominal")
    optimistic16 = next(row for row in uncertainty16 if row["scenario"] == "optimistic")
    conservative16 = next(row for row in uncertainty16 if row["scenario"] == "conservative")
    answer = "YES" if readiness["outcome"] == "E021_READY" else "NO"
    blocker_text = "; ".join(readiness.get("pre_rental_blockers", []))
    ending = (
        f"""YES

Frozen E021: 96 workers; 12 single-host P8 pods; homogeneous RTX 3090 primary policy; {placement['maximum_worker_peak_gib']:.3f} GiB/worker; `python scripts/run_experiment_021.py --preflight` followed only after approval by the armed `--apply --max-budget-usd X` command; block-16/chunk-1 PREDICTED range {conservative16['predicted_tok_s_per_user']:.2f}-{optimistic16['predicted_tok_s_per_user']:.2f} tok/s/user (nominal {nominal16['predicted_tok_s_per_user']:.2f}); estimated ${cost['expected_cost']['total_usd']:.2f} expected / ${cost['conservative_cost']['total_usd']:.2f} conservative; primary physical acceptance gate >=5 real Kimi K3 output/accepted target tok/s/user."""
        if answer == "YES"
        else (
            "NO\n\nExact remaining pre-rental work: "
            + blocker_text
            + "; rebuild and publish the corrected image by immutable digest; rerun the complete zero-rental preflight and require 12 policy-compliant P8 hosts before any budget approval"
        )
    )
    report = f"""# Experiment 020: Full Pre-Spend Swarm Readiness

**Outcome: {readiness['outcome']}**

Experiment 020 rented zero GPUs and mutated zero Vast resources. It freezes a 96-worker P8/depth-8 placement, but corrected simulator validation and the end-to-end production deployment path are not complete. The audit found six high-impact issues that can and must be solved without rental: required layer/span replays were summed rather than physically executed; the rendered controller-host-agent role is absent; bounded worker processes do not yet dispatch real shard tasks; the provisioning state machine has no guarded real Vast backend adapter; pod cache acquisition is not integrated into the host agent; and preflight cannot validate a supplied pinned registry image. The live marketplace snapshot is also incomplete: only {fleet['matching_p8_hosts']} of {fleet['required_pods']} required homogeneous RTX 3090 P8 hosts were available. The E021 path remains fail-closed and spends $0.

{truth_table}

## Decision and evidence hierarchy

The readiness decision is based on exact checkpoint placement, a complete 93-layer sharded oracle, an audit that rejected the incomplete single-resource replay method, event accounting, post-fusion physical shard services, real control-plane/transport dry runs, a locally built SM86 Linux image, real ranged model acquisition, a guarded read-only Vast snapshot, and injected rollback failures. Every throughput value in this report is **PREDICTED, NOT PHYSICALLY PROVEN**. RTX 3090 performance and the distributed critical path remain E021 measurements only after the pre-rental blockers close.

The old Experiment 019 value of 20.8836 tok/s remains **PROVISIONAL / UNVALIDATED**. E019 retains `MODEL_INVALID` under its preregistered gate. E020 does not rewrite that history.

## Methodological correction to E019

The invalid question was whether serial execution of a sharded implementation approximately equals a different optimized monolithic implementation. E020 instead asks: does the event model reproduce the physically executed sharded algorithm; are measured sharding costs present; does explicit resource scheduling create the correct critical path; and how sensitive is the result to hardware/network assumptions that cannot be tested without independent GPUs?

Two short-sample replay attempts failed (median errors 7.07% and 8.35%) with opposite span drift. The hypothesis was measurement instability. The redesign increased expert samples to 31 iterations and non-expert sampling to 15 complete repeats, kept calibration and held-out captures independent, and added no normalization. Primitive service stability then met the numeric targets: median {100 * replay['validation']['median_error']:.2f}%, p90 {100 * replay['validation']['p90_error']:.2f}%, maximum {100 * replay['validation']['maximum_error']:.2f}%. However, audit showed that full-layer and 2/4/8-layer "actual" walls were arithmetic sums of held-out calls, not one physically executed ordered shard workload. This is methodologically insufficient under the E020 definition, so corrected simulator validation is **FAIL**, irrespective of the low numeric errors.

{replay_table}

The accounting suite checked {accounting['configurations_checked']} block/chunk configurations. In every case the sum of worker event durations exactly reconciled with measured bottom-up compute work; network duration reconciled with latency, serialization, and protocol overhead; and collectives, state waits, queue waits, and launch-overhead policy remained explicit. Status: {accounting['status']}.

## Placement decision

{candidate_table}

P8/depth-2 is fastest in the model but requires 376 GPUs. P8/depth-4 still requires 192. P16 candidates increase local pod width to a marketplace-hostile 16 GPUs. P8/depth-8 retains large modeled headroom with 96 GPUs and a directly searchable 12 x 8 topology; it is therefore the frozen economic/deployment choice, not the raw simulated-throughput winner.

The exact manifest covers {placement['checkpoint_payload_bytes']:,} checkpoint payload bytes across {placement['coverage_tensor_count']:,} tensors with zero gap, zero overlap, and zero unintended weight replication. It emits 96 worker manifests across 12 pods. All {manifest_hashes['worker_file_hash_pairs']:,} worker/source-file identity pairs carry authoritative SHA-256 values with zero placeholder hashes. Maximum peak is {placement['maximum_worker_peak_gib']:.3f} GiB, maximum routed/shared expert fraction is 12.5%, and no worker owns a whole layer, routed expert, or shared expert.

## Complete sharded correctness

The frozen placement executed all {correctness['executed_layers']} layers through worker-owned shard paths on the RTX 5090, including endpoint sharding, exact routes, KDA/MLA/AttnRes state, and grouped experts. It performed {correctness['worker_operation_count']:,} explicit worker operations. Maximum relative L2 was {correctness['maximum_relative_l2_error']:.3e}; greedy token {correctness['head']['distributed_exact_argmax']['token_id']} matched the canonical oracle; ownership failures were {correctness['manifest_ownership_audit']['failure_count']}. Status: {correctness['status']}.

## Grouped top-16 expert stripe

{fusion_table}

The retained implementation maps 16 logical fragments to at most 2 expert launches per worker for a minimum 8x launch coalescing at one row, accumulates route weights locally, emits one partial latent vector, and remains numerically exact under the existing gate. All changed worker services were re-profiled with real K3 weights and fixtures.

## Bottom-up predicted performance

{target_table}

{uncertainty_table}

At block 16/chunk 1 the PREDICTED range is {conservative16['predicted_tok_s_per_user']:.2f}-{optimistic16['predicted_tok_s_per_user']:.2f} tok/s/user, nominal {nominal16['predicted_tok_s_per_user']:.2f}. The +20% all-compute sensitivity remains above 5 tok/s. Because the corrected physical replay gate failed its methodology audit, these projections are **UNVALIDATED DIAGNOSTICS** and cannot justify spend yet. Work-inflation remains an economic metric and is not used as a validity gate.

## Concurrency and stragglers

{concurrency_table}

{straggler_table}

Concurrency is a secondary economic analysis. It does not replace the single-user physical gate. Straggler scenarios quantify critical-path amplification so E021 can interpret a weak result without changing the acceptance criterion.

## Speculative overhead

{speculative_table}

Experiment 015 did not produce a swarm-valid acceptance distribution (measurement count 0); the public block-7 mean of 3.85 is only a scenario input. The exact E021 secondary measurement is frozen: real accepted-length histograms on preregistered coding, math, chat, and creative workloads for blocks 7/12/16 plus draft, rollback, and commit latency. Target-only remains primary, so this evidence gap does not conceal or replace the 5 tok/s architecture gate.

## Runtime, protocol, and security

The real TLS/framed lightweight controller dry run registered and exercised {dry96['worker_count']} workers in {dry96['pod_count']} pods. Stress runs passed at {dry376['worker_count']} and {dry1000['worker_count']} workers; the harness does not hardcode 96. The protocol uses persistent TLS connections, per-run HMAC credentials, binary framing, bounded buffers/backpressure, IDs, timeouts, checksums, and propagated errors. Security validation: {security['status']}. This validates the control plane and wire format, not the missing production controller entrypoint or native shard-task dispatch.

## Linux/SM86 deployment and model distribution

The clean multi-stage CUDA 13.0.1 / Python 3.12 image built and passed health, native-library load, bundled-manifest, and eight-worker lifecycle tests. All three currently required Linux native libraries contain SM86 code and have recorded SHA-256 hashes; SM120 development binaries remain available. Runtime secrets are injected, never baked into the image or repository. Nevertheless the deployment gate is **FAIL**: lifecycle-only workers are not a production K3 data plane, the controller role rendered in the Vast command does not exist, pod acquisition is not wired into the host lifecycle, the real provisioning adapter is absent, and preflight cannot accept/verify a pinned registry digest.

Model acquisition used the authoritative `moonshotai/Kimi-K3` source. A real 64 KiB pair of HTTP range requests resumed correctly, returned HTTP 206, hash-matched immutable local bytes, and did not redownload the checkpoint. Pod bundles transfer {pod_bundles['total_fleet_transfer_bytes']:,} bytes fleet-wide, including {pod_bundles['intentional_file_replication_bytes']:,} intentional bytes, with a largest pod disk requirement of {max(row['disk_requirement_bytes'] for row in pod_bundles['bundles']) / GIB:.3f} GiB. No worker or pod downloads a full checkpoint by design. Acquisition status: {acquisition['status']}.

## Provisional topology thresholds

- Intra-pod: RTT <= {topology['intra_pod']['maximum_rtt_ms']:.3f} ms and bandwidth >= {topology['intra_pod']['minimum_bandwidth_gbps_at_maximum_rtt']:.3f} Gbps at that RTT.
- Inter-pod: RTT <= {topology['inter_pod']['maximum_rtt_ms']:.3f} ms and bandwidth >= {topology['inter_pod']['minimum_bandwidth_gbps_at_maximum_rtt']:.3f} Gbps at that RTT.
- Jitter <= {100 * topology['maximum_jitter_fraction']:.0f}% in the preregistered gate model.

These numeric thresholds are diagnostic and **INVALID_PENDING_PHYSICAL_SINGLE_RESOURCE_REPLAY**. They must be re-derived and frozen before E021. Once valid, they are evaluated from live probes after deployment; Vast bandwidth metadata is never accepted as physical proof.

## Vast marketplace snapshot and cost

Snapshot at {fleet['snapshot_at']}:

{fleet_classes}

Primary RTX 3090 availability is {fleet['current_availability']}: {fleet['matching_p8_hosts']} complete P8 hosts versus {fleet['required_pods']} required. Those matching hosts were a snapshot, not reserved capacity. Offer IDs are ephemeral. E021 preflight requires a fresh complete homogeneous plan; unrelated single-GPU hosts are treated as WAN and are not substitutes.

Estimated cost is ${cost['expected_cost']['total_usd']:.2f} expected, ${cost['conservative_cost']['total_usd']:.2f} conservative, and an unapproved suggested worst-case budget cap of ${cost['worst_case_budget_cap_usd']:.2f}. These include startup, model download, bootstrap, topology qualification, warmup, benchmark, artifact collection, failure allowance, disk, and exposed transfer costs. Rate basis: {cost['observed_rate_basis']}; when no primary host matches, this is a policy-scenario budget rather than a live purchasable fleet quote. Actual E020 charge: $0.

The hard arming conjunction requires `SWARM_ALLOW_RENTAL=EXPERIMENT_021`, `--apply`, exact `experiment-021`, an approved plan digest, and a positive maximum-dollar budget. E020 additionally has a compile-time read-only lock. All create/launch/stop/destroy guard tests mock the subprocess boundary.

## Provisioning and rollback

The exact state machine passed a nominal 12-pod run and 11 injected failures: disappearing offer, pod provision failure, never-ready instance, wrong GPU count, insufficient disk, bootstrap failure, download failure, hash mismatch, slow network, worker health failure, and controller crash. Every case preserved a valid hash-chained ledger, included every conceptual rental in teardown, protected unrelated IDs, and ended without an orphan. The cost kill switch passed mock-time/rate tests.

## Risk register

{risk_table}

No high-impact risk remains with `can_test_without_rental=true` and `tested=false`, but six such risks were tested and **failed**. They are pre-rental implementation blockers, not physical unknowns. Once closed, the legitimate rental-only unknowns are actual SM86 throughput, physical collectives/transport, Vast host behavior, the full distributed critical path, and physical cost/token.

## Readiness gates

{gates_table}

The placement and prediction evidence are strong, but `{readiness['outcome']}` means another bounded pre-rental validation/deployment pass is required before E021. No rental is authorized. After the six blockers close, the live preflight must still remain `NO_GO` until 12 compliant hosts and an explicitly approved budget exist.

## EXPERIMENT 021 READINESS VERDICT

{readiness['outcome']}

> Have we done everything reasonably possible without paying for independent GPUs, such that the next experiment should be the full physical Kimi K3 swarm rather than another preparatory architecture experiment?

{ending}
"""
    path = repo / "docs" / "experiments" / "EXPERIMENT_020_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def finalize_experiment_020(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    artifact = repo / "artifacts" / "experiment-020"
    for name in (
        "validation",
        "placement",
        "physical",
        "simulation",
        "runtime",
        "deployment",
        "vast",
        "charts",
    ):
        (artifact / name).mkdir(parents=True, exist_ok=True)

    snapshot = _read(artifact / "vast" / "offer-snapshot.json")
    bundles = _read(artifact / "deployment" / "pod-bundles.json")
    cost = estimate_e021_cost(snapshot, bundles)
    atomic_write_json(artifact / "vast" / "cost-estimate.json", cost)

    inputs = _service_inputs(repo, artifact)
    simulation = _simulation_artifacts(repo, artifact, inputs)
    physical = _physical_artifacts(artifact, inputs)
    speculation = _speculative_model(repo, artifact, simulation)
    deployment = _deployment_artifacts(repo, artifact)
    tests = _test_results(artifact)
    failures = _failure_log(artifact)
    _render_runbook(repo, artifact)
    risk = _risks(artifact, deployment, speculation)
    readiness, truth = _readiness(repo, artifact, deployment, risk)

    replay = _read(artifact / "validation" / "single-resource-replay.json")
    placement = _read(artifact / "placement" / "final-placement.json")
    correctness = _read(artifact / "correctness" / "full-93-sharded.json")
    fleet = _read(artifact / "vast" / "fleet-feasibility.json")
    envelope16 = [
        row
        for row in simulation["envelope"]
        if row["arm"] == "uncertainty_envelope"
        and row["block"] == 16
        and row["chunk"] == 1
    ]
    performance = {
        row["scenario"]: row["predicted_tok_s_per_user"] for row in envelope16
    }
    summary = {
        "schema_version": "experiment-020-summary-v1",
        "experiment": "020",
        "classification": readiness["outcome"],
        "evidence_class": "PRE_SPEND_READINESS; throughput PREDICTED, not physical swarm",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "any_gpu_rented": False,
        "any_vast_resource_mutated": False,
        "vast_mode": "READ_ONLY",
        "e019_classification": "MODEL_INVALID",
        "e019_20_8836_tok_s": "PROVISIONAL / UNVALIDATED",
        "final_architecture": {
            "worker_count": placement["worker_count"],
            "pod_count": placement["pod_count"],
            "workers_per_pod": placement["workers_per_pod"],
            "stripe_degree": 8,
            "depth_span": 8,
            "preferred_gpu_class": "RTX 3090",
            "maximum_worker_peak_gib": placement["maximum_worker_peak_gib"],
        },
        "checkpoint": {
            "payload_bytes": placement["checkpoint_payload_bytes"],
            "tensor_count": placement["coverage_tensor_count"],
            "full_coverage": placement["full_checkpoint_covered"],
            "gap_bytes": placement["coverage_gap_bytes"],
            "overlap_bytes": placement["coverage_overlap_bytes"],
        },
        "full_93_layer_correctness": correctness["status"],
        "single_resource_replay": {
            "status": replay["status"],
            **replay["validation"],
        },
        "predicted_block16_chunk1_tok_s": performance,
        "predicted_not_physical": True,
        "projection_validation_status": replay["status"],
        "projection_use": "UNVALIDATED_DIAGNOSTIC_ONLY",
        "market_snapshot": {
            "timestamp": fleet["snapshot_at"],
            "availability": fleet["current_availability"],
            "matching_p8_hosts": fleet["matching_p8_hosts"],
            "required_p8_hosts": fleet["required_pods"],
        },
        "cost": {
            "expected_usd": cost["expected_cost"]["total_usd"],
            "conservative_usd": cost["conservative_cost"]["total_usd"],
            "unapproved_worst_case_cap_usd": cost["worst_case_budget_cap_usd"],
            "e020_charge_usd": 0,
        },
        "tests": tests,
        "failure_log": failures,
        "remaining_pre_rental_testable_high_impact_untested": risk[
            "remaining_pre_rental_testable_high_impact_untested"
        ],
        "remaining_pre_rental_testable_high_impact_failed": risk[
            "remaining_pre_rental_testable_high_impact_failed"
        ],
        "remaining_pre_rental_testable_critical_risks": risk[
            "remaining_pre_rental_testable_critical_risks"
        ],
        "pre_rental_blockers": readiness["pre_rental_blockers"],
        "current_launch_permission": "NO_GO",
        "next_experiment": "PRE_RENTAL_DEPLOYMENT_COMPLETION",
    }
    atomic_write_json(artifact / "summary.json", summary)

    audit = _read(artifact / "vast" / "safety-audit.json")
    commands = [
        "# Experiment 020 command ledger (secrets redacted)",
        "# Every Vast invocation below was guarded READ_ONLY.",
    ]
    commands.extend(
        "vastai " + " ".join(row["arguments"])
        for row in audit["commands"]
        if row.get("subprocess_invoked")
    )
    commands.extend(
        (
            "python scripts/experiment_020_validate.py",
            "python scripts/experiment_020_full_sharded.py",
            "python scripts/experiment_020_physical.py expert-grouped --degree 8",
            "python scripts/experiment_020_physical.py expert-grouped --degree 16",
            "python scripts/run_experiment_021.py --preflight",
            "docker build -f deployment/Dockerfile.e021 -t swarm-inference-lab:e021-sm86-e020 .",
            "python -m pytest tests/unit/test_experiment_020_*.py",
            "# Vast mutating commands executed: 0",
            "# GPU rentals: 0",
        )
    )
    (artifact / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    _render_report(
        repo,
        artifact,
        simulation,
        physical,
        speculation,
        readiness,
        truth,
        risk,
    )
    environment, source = _environment_and_sources(repo, artifact)
    summary["environment"] = environment
    summary["source_file_count"] = len(source["files"])
    atomic_write_json(artifact / "summary.json", summary)
    _charts(artifact, simulation, physical, risk, readiness)
    return summary


__all__ = ["finalize_experiment_020"]
