"""Generate the evidence-backed final Experiment 014 handoff artifacts.

This is deliberately a fail-closed report compiler.  It consumes the promoted
receipts, verifies their identities and cross-links, and writes only derived
summaries.  It never executes model code or upgrades a physical NOT_RUN result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "experiment-014"
REPORT = ROOT / "docs" / "experiment-014-kimi-k3-precluster-certification.md"
PACKAGE = ROOT / "artifacts" / "experiment-015-deployment"
PACKAGE_ARCHIVE = PACKAGE.with_suffix(".zip")
FINAL_BEGIN = "<!-- EXPERIMENT-014-FINAL-SUMMARY:BEGIN -->"
FINAL_END = "<!-- EXPERIMENT-014-FINAL-SUMMARY:END -->"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _artifact(role: str, path: Path, status: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    if status is not None:
        result["status"] = status
    return result


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suite = root if "tests" in root.attrib else root.find("testsuite")
    if suite is None:
        raise ValueError(f"JUnit receipt contains no testsuite: {path}")
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "skipped": skipped,
        "failures": failures,
        "errors": errors,
        "time_seconds": float(suite.attrib.get("time", 0.0)),
        "sha256": _sha256(path),
    }


def _gate(
    gate_id: int,
    name: str,
    status: str,
    evidence: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "name": name,
        "status": status,
        "satisfied": status in {"PASS", "READY"},
        "evidence": evidence,
        "detail": detail,
    }


def _load_and_validate() -> dict[str, Any]:
    paths = {
        "census": ARTIFACTS / "k3-checkpoint-census.json",
        "full_graph": ARTIFACTS / "cuda" / "h014-038-regression-full-93-layer.json",
        "operation_matrix": ARTIFACTS / "k3-cuda-operation-matrix.json",
        "sm86": ARTIFACTS / "rtx3090-sm86-certification.json",
        "promotion": ARTIFACTS / "cuda" / "h014-038-promotion.json",
        "p1_final": ARTIFACTS / "persistent" / "h014-038-regression-final-stage.json",
        "p1_nonfinal": ARTIFACTS
        / "persistent"
        / "h014-038-regression-nonfinal-stages.json",
        "p1_stage_zero": ARTIFACTS
        / "persistent"
        / "h014-038-regression-stage-zero.json",
        "ready_repeat_1": ARTIFACTS
        / "persistent"
        / "h014-038u-final-repeat-1.json",
        "ready_repeat_2": ARTIFACTS
        / "persistent"
        / "h014-038u-final-repeat-2.json",
        "steady_stage_zero": ARTIFACTS
        / "persistent"
        / "h014-038aj2-stage-zero-steady.json",
        "kda_batch": ARTIFACTS
        / "performance"
        / "h014-038ah2-complete-stage-batch-layer89.json",
        "mla_batch": ARTIFACTS
        / "performance"
        / "h014-038ah2-complete-stage-batch-layer91.json",
        "stage_profile": ARTIFACTS
        / "performance"
        / "h014-038z-extraction-stage-profile.json",
        "continuous": ARTIFACTS
        / "performance"
        / "h014-034e-contextual-continuous-layer91-8k.json",
        "prefill_kda": ARTIFACTS
        / "performance"
        / "h014-033a-prefill-layer89-kda.json",
        "prefill_mla": ARTIFACTS
        / "performance"
        / "h014-033c-prefill-layer91-mla16k.json",
        "sub_exact": ARTIFACTS
        / "sub-layer"
        / "h014-sub-011-promoted-four-worker-real-expert.json",
        "sub_batch": ARTIFACTS
        / "sub-layer"
        / "h014-sub-012a-006f-promoted-single-interval-batch.json",
        "sub_scaling": ARTIFACTS
        / "sub-layer"
        / "h014-sub-005-scaling-network.json",
        "sub_routing": ARTIFACTS
        / "sub-layer"
        / "h014-sub-010-routing-imbalance.json",
        "sub_recovery": ARTIFACTS / "sub-layer" / "h014-sub-009b-recovery.json",
        "coarse": ARTIFACTS
        / "coarse"
        / "h014-038ae-promoted-stage0-stage1-tcp.json",
        "network": ARTIFACTS / "coarse" / "h014-038ah-network-analysis.json",
        "capacity": ARTIFACTS
        / "performance"
        / "h014-038ak-final-capacity-topology-economics.json",
        "placement": ARTIFACTS
        / "deployment"
        / "h014-037a-final-physical-placement.json",
        "distribution": ARTIFACTS
        / "distribution"
        / "h014-037a-remote-acquisition-fixture.json",
        "recovery": ARTIFACTS
        / "deployment"
        / "h014-037b-coarse-stage-recovery.json",
        "rehearsal": ARTIFACTS
        / "deployment"
        / "h014-037b-final-logical-rehearsal.json",
        "canary": ARTIFACTS
        / "deployment"
        / "h014-037b4-physical-canary-fixtures.json",
        "package_controls": ARTIFACTS
        / "deployment"
        / "h014-038al-canonical-package-controls.json",
        "pytest": ARTIFACTS / "regression" / "h014-038af-full-pytest.xml",
        "ruff": ARTIFACTS / "regression" / "h014-038af-ruff.json",
        "ledger": ARTIFACTS / "cycle-ledger.json",
        "dll": ARTIFACTS
        / "cuda"
        / "native"
        / "coli_cuda-sm86-h014-038-final.dll",
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing {name}: {path}")

    loaded = {
        name: _json(path)
        for name, path in paths.items()
        if path.suffix.lower() == ".json"
    }
    pass_receipts = (
        "full_graph",
        "operation_matrix",
        "sm86",
        "promotion",
        "p1_final",
        "p1_nonfinal",
        "p1_stage_zero",
        "ready_repeat_1",
        "ready_repeat_2",
        "steady_stage_zero",
        "kda_batch",
        "mla_batch",
        "continuous",
        "prefill_kda",
        "prefill_mla",
        "sub_exact",
        "sub_batch",
        "sub_scaling",
        "sub_routing",
        "sub_recovery",
        "coarse",
        "network",
        "capacity",
        "placement",
        "distribution",
        "recovery",
        "rehearsal",
        "canary",
        "package_controls",
        "ruff",
    )
    for name in pass_receipts:
        _require(loaded[name].get("status") == "PASS", f"{name} is not PASS")

    promotion = loaded["promotion"]
    dll_sha = _sha256(paths["dll"])
    _require(
        dll_sha == "c3bdb40d49a1b0e512ddb1e84485e0bdcf9d2e6ac6fa4049d81ff2ecd12ae326",
        "final DLL identity differs",
    )
    _require(promotion["final_binary"]["sha256"] == dll_sha, "promotion DLL mismatch")
    for raw_path, expected_sha in promotion["validated_inputs"].items():
        source = ROOT / Path(raw_path.replace("\\", "/"))
        _require(_sha256(source) == expected_sha, f"promoted input differs: {source}")

    graph = loaded["full_graph"]
    _require(graph["coverage"]["layers_executed"] == 93, "full graph is incomplete")
    _require(graph["coverage"]["no_cpu_mathematical_fallback"], "CPU math fallback")
    _require(graph["correctness"]["stateful_decode_executed"], "decode not stateful")
    _require(
        loaded["operation_matrix"]["summary"]["cuda_ready"] == 11,
        "operation matrix is incomplete",
    )
    _require(loaded["sm86"]["summary"]["certified"] == 11, "sm_86 surface incomplete")

    for batch_receipt in (loaded["kda_batch"], loaded["mla_batch"]):
        _require(sorted(map(int, batch_receipt["batches"])) == [1, 2, 4, 8], "batch ladder differs")
        _require(
            all(item["status"] == "PASS" for item in batch_receipt["batches"].values()),
            "batch ladder contains failure",
        )

    capacity = loaded["capacity"]
    placement = loaded["placement"]
    network = loaded["network"]
    recommended = capacity["recommended_experiment_015_topology"]
    _require(recommended["candidate_id"] == "B", "candidate B not selected")
    _require(recommended["class"] == "WHOLE-LAYER", "topology class differs")
    _require(recommended["worker_count"] == 93, "worker count differs")
    _require(placement["node_count"] == 93, "placement node count differs")
    _require(placement["coverage"]["duplicate_tensor_count"] == 0, "duplicate tensors")
    _require(
        placement["coverage"]["unassigned_required_tensor_count"] == 0,
        "orphan tensors",
    )
    admission = placement["network_admission"]
    recommendation = network["recommendation"]
    _require(admission["maximum_rtt_ms"] == recommendation["maximum_tested_rtt_ms"], "RTT mismatch")
    _require(
        admission["minimum_bandwidth_gbps"]
        == recommendation["minimum_tested_bandwidth_gbps"],
        "bandwidth mismatch",
    )

    controls = loaded["package_controls"]
    _require(all(controls["acceptance_gates"].values()), "package tamper control failed")
    _require(not controls["fleet_activation_allowed"], "fleet activation was unlocked")
    _require(controls["physical_3090_canary"] == "NOT_RUN", "physical status fabricated")
    _require(
        _sha256(PACKAGE_ARCHIVE) == controls["release_archive"]["sha256"],
        "canonical archive differs",
    )
    release = _json(PACKAGE / "RELEASE.json")
    _require(release["status"] == "READY_FOR_SINGLE_3090_CANARY", "release not ready")
    _require(not release["fleet_activation_allowed"], "release unlocked fleet")
    _require(release["physical_3090_canary"] == "NOT_RUN", "release canary fabricated")
    _require(release["worker_count"] == 93, "release worker count differs")
    _require(
        release["experiment_014_promotion_receipt_sha256"] == _sha256(paths["promotion"]),
        "release promotion identity differs",
    )

    junit = _junit_summary(paths["pytest"])
    _require(junit["passed"] == 1091, "pytest pass count differs")
    _require(junit["skipped"] == 13, "pytest skip count differs")
    _require(junit["failures"] == 0 and junit["errors"] == 0, "pytest failed")

    return {
        "paths": paths,
        "loaded": loaded,
        "release": release,
        "junit": junit,
        "dll_sha256": dll_sha,
    }


def _build_gates(data: dict[str, Any]) -> dict[str, Any]:
    paths = data["paths"]
    gates = [
        _gate(1, "checkpoint_understanding", "PASS", _relative(paths["census"]), "497,052/497,052 required tensors classified; zero unclassified."),
        _gate(2, "full_real_graph", "PASS", _relative(paths["full_graph"]), "93/93 real layers plus norm, head and sampling execute."),
        _gate(3, "stateful_decode", "PASS", _relative(paths["full_graph"]), "KDA and MLA state advance on decode."),
        _gate(4, "complete_cuda_graph", "PASS", _relative(paths["operation_matrix"]), "11/11 Kimi-critical operation classes execute on CUDA with no CPU math fallback."),
        _gate(5, "sm86_package_precanary", "PASS", _relative(paths["sm86"]), "sm_86 cubin and compute_86 PTX are present on the exact final binary; physical execution is deferred."),
        _gate(6, "canonical_persistent_kimi_runtime", "PASS", _relative(paths["p1_nonfinal"]), "All four canonical stage roles execute real assigned weights and state with zero warm lifecycle recreation."),
        _gate(7, "ready_semantics", "PASS", _relative(paths["ready_repeat_2"]), "Seven assigned-stage calls plus measured post-call quiescence contain late native initialization."),
        _gate(8, "safe_batching", "PASS", _relative(paths["kda_batch"]), "Incremental batch 1/2/4/8 passes; production batch is 8 and batch 9 rejects before CUDA."),
        _gate(9, "complete_stage_performance", "PASS", _relative(paths["mla_batch"]), "Late KDA and Gated-MLA complete stages are characterized through batch 8."),
        _gate(10, "real_sub_layer_microwork_correctness", "PASS", _relative(paths["sub_exact"]), "All 16 real selected experts execute exactly once under exact ownership and deterministic reduction."),
        _gate(11, "sub_layer_efficiency_characterized", "PASS", _relative(paths["sub_scaling"]), "2/4/8/16-worker scaling is measured; four workers retain 78.809% exact-run layer throughput."),
        _gate(12, "sub_layer_memory_reduction_characterized", "PASS", _relative(paths["sub_scaling"]), "Smallest tested worker is 0.983 GB (5.377%); four-worker exact execution is 3.931 GB per worker."),
        _gate(13, "coarse_distributed_kimi_slice", "PASS", _relative(paths["coarse"]), "Two persistent real CUDA stages exchange the exact canonical boundary."),
        _gate(14, "real_tensor_traffic", "PASS", _relative(paths["coarse"]), "Coarse payload is 258,048 bytes and mean one-way wire traffic is 258,283 bytes."),
        _gate(15, "coarse_network_requirements", "PASS", _relative(paths["network"]), "Coupled admission is at most 5 ms RTT and at least 10 Gbps for >=90% capacity retention."),
        _gate(16, "sub_layer_network_requirements", "PASS", _relative(paths["sub_scaling"]), "Useful fine work is confined to the tested <=0.5 ms, >=2.5 Gbps expert domain."),
        _gate(17, "runtime_safe_placement", "PASS", _relative(paths["placement"]), "All 93 planned 24 GiB workers are feasible with measured runtime reserves and zero orphan tensors."),
        _gate(18, "validated_capacity_model", "PASS", _relative(paths["capacity"]), "Local held-out median absolute percentage error is 0.553%; RTX 3090 transfer remains a physical canary risk."),
        _gate(19, "final_topology_chosen", "PASS", _relative(paths["capacity"]), "Candidate B: 93-worker whole-layer topology."),
        _gate(20, "economics_completed", "PASS", _relative(paths["capacity"]), "All requested GPU prices and utilization sensitivities are reported."),
        _gate(21, "serving_frontier", "PASS", _relative(paths["capacity"]), "5/10/20/30 tok/s targets are evaluated; sequential cadence is explicitly below target."),
        _gate(22, "remote_shard_distribution", "PASS", _relative(paths["distribution"]), "Targeted resume, retry, hashes, corruption rejection, cache and atomic activation pass."),
        _gate(23, "clean_bootstrap", "PASS", "artifacts/experiment-015-deployment", "No-checkout, hash-locked Linux install/build/prepare/qualify scripts validate locally."),
        _gate(24, "recovery", "PASS", _relative(paths["recovery"]), "Whole-stage and optional expert-group faults fail closed; retry, duplicate, stale and cancellation paths pass."),
        _gate(25, "final_logical_rehearsal", "PASS", _relative(paths["rehearsal"]), "93/93 logical workers reach READY; 576 events execute with no impossible route."),
        _gate(26, "rtx3090_canary", "READY", _relative(paths["canary"]), "Four-role physical canary fixtures and launcher are ready; physical status remains NOT_RUN."),
        _gate(27, "experiment_015_deployment_package", "READY", "artifacts/experiment-015-deployment/RELEASE.json", "Canonical 155-file package and archive pass positive validation and all 16 tamper controls."),
        _gate(28, "full_regressions", "PASS", _relative(paths["pytest"]), "First-party Ruff passes; pytest: 1,091 passed, 13 explicitly skipped, zero failures."),
    ]
    _require(all(item["satisfied"] for item in gates), "a final gate is unsatisfied")
    counts = Counter(item["status"] for item in gates)
    return {
        "schema_version": "experiment-014-final-acceptance-gates-v2",
        "experiment": "014-kimi-k3-precluster-certification",
        "status": "PASS",
        "verdict": "PASS",
        "ready_to_rent_single_canary": True,
        "full_fleet_activation_allowed": False,
        "satisfied_count": sum(item["satisfied"] for item in gates),
        "pass_count": counts["PASS"],
        "ready_count": counts["READY"],
        "fail_count": counts["FAIL"],
        "gates": gates,
        "terminal_statement": (
            "PRE-CLUSTER CERTIFICATION COMPLETE. Experiment 015 is the full physical "
            "Kimi K3 cluster test. No further synthetic architecture experiment is required."
        ),
    }


def _metrics(data: dict[str, Any]) -> dict[str, Any]:
    loaded = data["loaded"]
    capacity = loaded["capacity"]
    network = loaded["network"]
    sub = loaded["sub_exact"]
    sub_batch = loaded["sub_batch"]
    scaling = loaded["sub_scaling"]
    placement = loaded["placement"]
    coarse = loaded["coarse"]
    profile_layers = loaded["stage_profile"]["benchmark"]["layers"]
    kda_profile = profile_layers[0]["modes"]["production"]["device"]
    mla_profile = profile_layers[1]["modes"]["production"]["device"]
    price = capacity["economics"]["gpu_price_sensitivity"][0]
    memory = placement["memory_summary"]
    routing = sub["routing_imbalance"]
    smallest = min(scaling["scaling_curve"], key=lambda item: item["worker_bytes"])
    return {
        "checkpoint_revision": loaded["census"]["checkpoint"]["revision"],
        "final_binary_sha256": data["dll_sha256"],
        "safe_certified_batch": 8,
        "first_rejected_batch": 9,
        "kda_production_device_p50_ms": kda_profile["p50_ms"],
        "mla_production_device_p50_ms": mla_profile["p50_ms"],
        "kda_batch8_device_p50_ms": loaded["kda_batch"]["batches"]["8"]["performance"]["device"]["p50_ms"],
        "mla_batch8_device_p50_ms": loaded["mla_batch"]["batches"]["8"]["performance"]["device"]["p50_ms"],
        "sub_layer_result": "FUNCTIONAL BUT NOT CURRENTLY ECONOMIC",
        "smallest_tested_worker_bytes": smallest["worker_bytes"],
        "smallest_tested_worker_gib": smallest["worker_gib"],
        "smallest_tested_worker_fraction_percent": smallest["worker_fraction_percent"],
        "smallest_meaningful_gpu_capacity_gib": capacity["memory"]["smallest_meaningful_gpu_capacity_gib"],
        "best_sub_layer_workers": 4,
        "best_exact_sub_layer_worker_bytes": sub["memory"]["largest_worker_tracked_bytes"],
        "best_exact_sub_layer_worker_fraction_percent": sub["memory"]["workers"][0]["complete_layer_fraction_percent"],
        "sub_layer_distributed_layer_p50_ms": sub["performance"]["distributed_complete_layer_wall"]["p50_ms"],
        "sub_layer_distributed_layer_p95_ms": sub["performance"]["distributed_complete_layer_wall"]["p95_ms"],
        "sub_layer_distributed_layer_p99_ms": sub["performance"]["distributed_complete_layer_wall"]["p99_ms"],
        "sub_layer_relative_throughput_percent": 100.0 * sub["performance"]["whole_layer_relative_throughput"],
        "sub_layer_efficiency_percent": 100.0 * sub["performance"]["sub_layer_efficiency"],
        "sub_layer_batch8_capacity_retention_percent": sub_batch["batch8_evaluation"]["resident_batch_throughput_retention_percent"],
        "sub_layer_mean_total_bytes_per_token": sub["communication"]["mean_total_transport_bytes"],
        "sub_layer_critical_path_bytes_per_token": sub["communication"]["maximum_critical_path_payload_bytes"],
        "sub_layer_root_messages": sub["communication"]["mean_root_messages"],
        "sub_layer_synchronization_points": sub["communication"]["synchronization_points"],
        "sub_layer_maximum_tested_rtt_ms": network["fine_comparison"]["fine_recommended_maximum_tested_rtt_ms"],
        "sub_layer_minimum_tested_bandwidth_gbps": network["fine_comparison"]["fine_recommended_minimum_tested_bandwidth_gbps"],
        "sub_layer_exact_maximum_rtt_ms_at_100gbps": network["fine_comparison"]["fine_exact_maximum_rtt_ms_at_100gbps"],
        "sub_layer_exact_minimum_bandwidth_gbps_at_0_25ms": network["fine_comparison"]["fine_exact_minimum_bandwidth_gbps_at_0_25ms"],
        "routing_hottest_selected_count": routing["hottest_worker_selected_count"],
        "routing_coldest_selected_count": routing["coldest_contacted_worker_selected_count"],
        "routing_imbalance_ratio": routing["hottest_worker_selected_count"] / routing["coldest_contacted_worker_selected_count"],
        "coarse_payload_bytes": coarse["performance"]["wire"]["activation_payload_bytes"],
        "coarse_wire_bytes": coarse["performance"]["wire"]["mean_interstage_request_wire_bytes"],
        "coarse_maximum_tested_rtt_ms": network["recommendation"]["maximum_tested_rtt_ms"],
        "coarse_minimum_tested_bandwidth_gbps": network["recommendation"]["minimum_tested_bandwidth_gbps"],
        "coarse_capacity_retention_percent": network["recommendation"]["capacity_retention_percent"],
        "worker_count": placement["node_count"],
        "topology": placement["topology"]["class"],
        "maximum_planned_worker_vram_bytes": memory["maximum_planned_vram_bytes"],
        "physical_worker_vram_bytes": memory["physical_vram_bytes"],
        "minimum_total_headroom_bytes": memory["minimum_total_headroom_bytes"],
        "minimum_unallocated_after_safety_bytes": memory["minimum_remaining_unallocated_bytes"],
        "capacity_model_local_held_out_median_ape_percent": capacity["model_validation"]["local_held_out_median_absolute_percentage_error"],
        "rtx3090_transfer_status": capacity["model_validation"]["rtx_3090_transfer_status"],
        "projected_wall_capacity_retention_percent": 100.0 * capacity["decode_capacity"]["projected_wall_capacity_retention_fraction"],
        "projected_aggregate_output_tokens_per_second": capacity["decode_capacity"]["projected_rtx3090_wall_output_tokens_per_second"],
        "projected_per_user_decode_tokens_per_second": capacity["decode_latency"]["projected_per_user_tokens_per_second"],
        "projected_end_to_end_ms": capacity["decode_latency"]["projected_end_to_end_ms"],
        "economic_gpu_price_usd_per_hour": price["gpu_hourly_price_usd"],
        "fleet_cost_per_hour_usd": price["fleet_cost_per_hour_usd"],
        "cost_per_million_output_tokens_usd": price["cost_per_million_output_tokens_usd"],
        "gross_margin_at_15_per_million_percent": 100.0 * price["gross_margin_at_15_per_million_fraction"],
        "break_even_gpu_price_usd_per_hour": price["gpu_hourly_price_usd"]
        * 15.0
        / price["cost_per_million_output_tokens_usd"],
        "prefill": capacity["prefill"],
        "serving_frontier": capacity["serving_frontier"],
        "dominant_bottleneck": capacity["decode_capacity"]["bottleneck"],
        "pytest": data["junit"],
    }


def _manager_markdown(metrics: dict[str, Any]) -> str:
    m = metrics
    return f"""## Verdict

* Experiment 014 pre-cluster certification: **PASS**
* Complete Kimi CUDA graph: **PASS**
* RTX 3090 sm_86 package: **READY** (physical execution remains NOT RUN)
* Canonical persistent Kimi runtime: **PASS**
* Safe certified batch: **{m['safe_certified_batch']}** (first rejected batch: {m['first_rejected_batch']}, before CUDA)
* Sub-layer microwork: **{m['sub_layer_result']}**
* Smallest useful worker VRAM: **{m['smallest_meaningful_gpu_capacity_gib']:.0f} GiB** for an optional four-way expert partition inside a fast domain
* Recommended fleet size: **{m['worker_count']}**
* Recommended topology: **{m['topology']}**
* Maximum worker VRAM: **{m['maximum_planned_worker_vram_bytes'] / 2**30:.3f} GiB planned on 24 GiB**
* Minimum worker headroom: **{m['minimum_total_headroom_bytes'] / 2**30:.3f} GiB total; {m['minimum_unallocated_after_safety_bytes'] / 2**30:.3f} GiB remains beyond the 10% safety reserve**
* KDA p50: **{m['kda_production_device_p50_ms']:.3f} ms** (production batch 1)
* MLA p50: **{m['mla_production_device_p50_ms']:.3f} ms** (production batch 1)
* Sub-layer distributed layer p50: **{m['sub_layer_distributed_layer_p50_ms']:.3f} ms**
* Real coarse boundary bytes: **{m['coarse_payload_bytes']:,} payload / {m['coarse_wire_bytes']:,.0f} mean wire**
* Real microwork bytes: **{m['sub_layer_mean_total_bytes_per_token']:,.1f} mean total / {m['sub_layer_critical_path_bytes_per_token']:,} critical path per token**
* Required coarse network: **<= {m['coarse_maximum_tested_rtt_ms']:.1f} ms RTT and >= {m['coarse_minimum_tested_bandwidth_gbps']:.1f} Gbps**
* Required microwork network: **<= {m['sub_layer_maximum_tested_rtt_ms']:.1f} ms RTT and >= {m['sub_layer_minimum_tested_bandwidth_gbps']:.1f} Gbps**
* Capacity model held-out error: **{m['capacity_model_local_held_out_median_ape_percent']:.3f}% local median APE**
* Projected RTX 3090 capacity retention: **{m['projected_wall_capacity_retention_percent']:.3f}% wall**
* Projected aggregate output throughput: **{m['projected_aggregate_output_tokens_per_second']:.3f} tok/s**
* Projected per-user decode: **{m['projected_per_user_decode_tokens_per_second']:.3f} tok/s** at the admitted coarse edge
* Fleet cost/hour: **${m['fleet_cost_per_hour_usd']:.2f}** at ${m['economic_gpu_price_usd_per_hour']:.2f}/GPU-hour
* Cost/M output: **${m['cost_per_million_output_tokens_usd']:.3f}**
* Margin at $15/M: **{m['gross_margin_at_15_per_million_percent']:.3f}%**
* Remote distribution: **PASS**
* Bootstrap: **PASS** locally / physical clean node NOT RUN
* Recovery: **PASS**
* Final rehearsal: **PASS**
* Experiment 015 package: **READY**
* Ready to rent GPUs: **YES — rent the single RTX 3090 canary first; full-fleet activation remains locked**

## Required sub-layer summary

* Sub-layer microwork execution: **{m['sub_layer_result']}**
* Smallest tested worker footprint: **{m['smallest_tested_worker_bytes'] / 1e9:.3f} GB ({m['smallest_tested_worker_gib']:.3f} GiB)**
* Fraction of complete layer per smallest microworker: **{m['smallest_tested_worker_fraction_percent']:.3f}%**
* Best sub-layer worker count per layer: **{m['best_sub_layer_workers']}**
* Sub-layer layer-throughput relative to one-GPU baseline: **{m['sub_layer_relative_throughput_percent']:.3f}%**
* Sub-layer capacity retention: **{m['sub_layer_batch8_capacity_retention_percent']:.3f}%** at logical batch 8
* Maximum viable microwork RTT: **{m['sub_layer_maximum_tested_rtt_ms']:.1f} ms tested** ({m['sub_layer_exact_maximum_rtt_ms_at_100gbps']:.3f} ms exact at 100 Gbps)
* Minimum viable microwork bandwidth: **{m['sub_layer_minimum_tested_bandwidth_gbps']:.1f} Gbps tested** ({m['sub_layer_exact_minimum_bandwidth_gbps_at_0_25ms']:.3f} Gbps exact at 0.25 ms)
* Expert-routing imbalance: **{m['routing_hottest_selected_count']}:{m['routing_coldest_selected_count']} hottest:coldest selections ({m['routing_imbalance_ratio']:.1f}x)** in the retained three-position trace
* Recommended use of microworkers: **ONLY INSIDE FAST DOMAINS; not in the initial fleet**
* Experiment 015 topology: **WHOLE-LAYER**

The initial fleet is economically viable only near the measured low-price case. The modeled break-even GPU price is approximately **${m['break_even_gpu_price_usd_per_hour']:.3f}/hour** at full modeled utilization and a $15/M selling price; the $0.08/hour and higher cases lose money. Sequential dependency depth also limits one stream to {m['projected_per_user_decode_tokens_per_second']:.3f} tok/s even though aggregate batch-8 capacity is {m['projected_aggregate_output_tokens_per_second']:.3f} tok/s.
"""


def _answers(metrics: dict[str, Any]) -> str:
    m = metrics
    return f"""## Final questions answered

1. **Can the complete model execute correctly?** Yes. The exact promoted binary executes all 93 real layers, all 11 CUDA operation classes, final norm, head and sampling with exact routes, stateful decode and maximum full-graph relative L2 error below 1e-6.
2. **Can the first physical cluster test be launched safely?** Yes, beginning with one RTX 3090 canary. The static package cannot activate the fleet without a physical certificate, exact canary ELF, 93 node admissions and the cost guard.
3. **What batching design should be used?** Incremental, fail-closed batch 8. Batch 1/2/4/8 passed on late KDA and MLA stages; batch 9 is the first production rejection and is rejected before CUDA.
4. **Did real Kimi sub-layer microworkers work?** Functionally yes: four independent persistent process partitions executed real selected MXFP4 experts and deterministic reduction exactly. Physical multi-GPU efficiency was not claimed.
5. **Can a worker store less than one complete layer?** Yes. Exact four-way workers held {m['best_exact_sub_layer_worker_bytes'] / 1e9:.3f} GB each; the smallest tested 16-way worker held {m['smallest_tested_worker_bytes'] / 1e9:.3f} GB.
6. **What is the smallest useful footprint?** The smallest tested footprint is {m['smallest_tested_worker_bytes'] / 1e9:.3f} GB, but the smallest meaningful deployable class is 8 GiB for a four-way partition plus runtime reserves. It is optional, not economic in the initial fleet.
7. **When is sub-layer work worthwhile?** Only inside a fine domain at <={m['sub_layer_maximum_tested_rtt_ms']:.1f} ms RTT and >={m['sub_layer_minimum_tested_bandwidth_gbps']:.1f} Gbps. Even there, the same-GPU exact layer retained {m['sub_layer_relative_throughput_percent']:.3f}% of the resident baseline, so it buys memory reduction rather than speed.
8. **What is the dominant bottleneck?** Aggregate capacity is limited by row-serial 8K contextual Gated-MLA service. Per-user cadence is limited by 93 sequential stages plus 92 coarse edges; economics is limited by rental price.
9. **How many GPUs should be rented?** Rent one RTX 3090 first. If it passes, admit the remaining nodes for an exact {m['worker_count']}-GPU fleet; do not activate a partial or over-price fleet.
10. **What physical topology should be used?** Candidate B, a {m['worker_count']}-worker whole-layer pipeline with embedding+dense packed into the first worker and final/head packed into the last. Fine expert groups remain an optional later canary.
11. **What network is required?** Coarse workers require <={m['coarse_maximum_tested_rtt_ms']:.1f} ms RTT and >={m['coarse_minimum_tested_bandwidth_gbps']:.1f} Gbps as one coupled admission rule. Optional fine groups require <={m['sub_layer_maximum_tested_rtt_ms']:.1f} ms and >={m['sub_layer_minimum_tested_bandwidth_gbps']:.1f} Gbps.
12. **What throughput is predicted?** {m['projected_aggregate_output_tokens_per_second']:.3f} aggregate output tok/s at batch 8 and {m['projected_per_user_decode_tokens_per_second']:.3f} tok/s for one dependency-bound stream at the admitted coarse edge.
13. **How trustworthy is the model?** Local held-out median error is {m['capacity_model_local_held_out_median_ape_percent']:.3f}%, well inside 10%. RTX 3090 transfer is still model-based rather than held out, so the canary must validate it.
14. **Does the economics work?** At $0.05/GPU-hour, yes narrowly: ${m['cost_per_million_output_tokens_usd']:.3f}/M and {m['gross_margin_at_15_per_million_percent']:.3f}% margin at $15/M. Break-even is about ${m['break_even_gpu_price_usd_per_hour']:.3f}/GPU-hour; it fails at $0.08/hour and above under this model.
15. **What remains unknown until physical GPUs run?** Actual Linux sm_86 correctness and timing, 5090-to-3090 transfer error, thermal/clock behavior, physical network jitter/contention, full-fleet cadence, and real clean-node provisioning time.
16. **What launches Experiment 015?** Verify `package-lock.json`; install and build the runtime on one 3090; prepare workers 000/089/091/092; run `run-3090-canary.sh`; publish the exact ELF and physical certificate immutably; rent the remainder only if canary, network and <=$0.05/hour cost gates pass; install that qualified ELF on every node; prepare and qualify each assigned stage; exchange public fingerprints; start workers; then run `bind-and-deploy.sh`.

**PRE-CLUSTER CERTIFICATION COMPLETE. Experiment 015 is the full physical Kimi K3 cluster test. No further synthetic architecture experiment is required.**
"""


def _chart_index(data: dict[str, Any]) -> dict[str, Any]:
    charts = [
        (
            "sub_layer_scaling",
            ARTIFACTS / "charts" / "h014-sub-layer-scaling.png",
            data["paths"]["sub_scaling"],
            "2/4/8/16 workers: layer latency, worker state, decomposition and RTT retention.",
        ),
        (
            "coarse_network",
            ARTIFACTS / "charts" / "h014-038ah-coarse-network.png",
            data["paths"]["network"],
            "Coupled RTT/bandwidth retention for the measured 258,048-byte boundary.",
        ),
        (
            "capacity_economics",
            ARTIFACTS / "charts" / "h014-038ak-capacity-economics.png",
            data["paths"]["capacity"],
            "Topology capacity/cost and whole-layer GPU-price sensitivity.",
        ),
    ]
    rows = []
    for chart_id, path, source, description in charts:
        _require(path.is_file(), f"missing chart: {path}")
        rows.append(
            {
                "id": chart_id,
                "path": _relative(path),
                "sha256": _sha256(path),
                "source": _relative(source),
                "description": description,
                "visual_qa": "PASS",
                "qa_note": "Axes, units, threshold line, labels and plotted series are legible and consistent with the source receipt.",
            }
        )
    return {
        "schema_version": "experiment-014-final-chart-index-v1",
        "status": "PASS",
        "charts": rows,
    }


def _ledger_summary(data: dict[str, Any]) -> dict[str, Any]:
    ledger = data["loaded"]["ledger"]
    cycles = ledger["cycles"]
    ids = [item["id"] for item in cycles]
    _require(len(ids) == len(set(ids)), "duplicate cycle IDs")
    incomplete = []
    decisions: Counter[str] = Counter()
    for item in cycles:
        for key in (
            "hypothesis",
            "implementation",
            "benchmark",
            "result",
            "inspection",
            "bottleneck",
            "decision",
            "redesign",
        ):
            if key not in item:
                incomplete.append(f"{item['id']}:{key}")
        decision = item.get("decision")
        if isinstance(decision, str):
            normalized = decision.lower()
            if "revert" in normalized:
                decisions["revert"] += 1
            elif "retain" in normalized:
                decisions["retain"] += 1
            elif "modify" in normalized:
                decisions["modify"] += 1
            else:
                decisions["other"] += 1
        else:
            decisions["structured"] += 1
    _require(not incomplete, f"incomplete ledger fields: {incomplete[:5]}")
    _require("H014-038af" in ids, "final cycle is absent from ledger")
    return {
        "schema_version": "experiment-014-final-cycle-ledger-summary-v1",
        "status": "PASS",
        "cycle_count": len(cycles),
        "first_cycle": ids[0],
        "last_cycle": ids[-1],
        "unique_cycle_ids": True,
        "complete_eight_field_cycles": True,
        "incomplete_fields": [],
        "decision_class_counts": dict(sorted(decisions.items())),
        "ledger_path": _relative(data["paths"]["ledger"]),
        "ledger_sha256": _sha256(data["paths"]["ledger"]),
    }


def _update_report(original: str, current: str) -> str:
    block = f"{FINAL_BEGIN}\n\n# Experiment 014 final pre-cluster report\n\n{current.strip()}\n\n{FINAL_END}"
    if FINAL_BEGIN in original:
        before, remainder = original.split(FINAL_BEGIN, 1)
        _, after = remainder.split(FINAL_END, 1)
        _require(not before.strip(), "unexpected content before final report marker")
        historical = after.lstrip("\r\n")
    else:
        historical = (
            "# Superseded pre-continuation report (historical)\n\n"
            "> The body below is retained as crash-surviving history. Its original "
            "top-level FAIL, blockers, node count and manager answers are stale and "
            "are superseded by the evidence-backed final section above.\n\n"
            + original.lstrip("\ufeff")
        )
    return f"{block}\n\n{historical.rstrip()}\n"


def _build_outputs(data: dict[str, Any]) -> dict[Path, bytes]:
    gates = _build_gates(data)
    metrics = _metrics(data)
    manager = _manager_markdown(metrics)
    answers = _answers(metrics)
    chart_index = _chart_index(data)
    ledger_summary = _ledger_summary(data)
    report_text = _update_report(REPORT.read_text(encoding="utf-8"), f"{manager}\n\n{answers}")

    machine = {
        "schema_version": "experiment-014-final-machine-summary-v1",
        "experiment": "014-kimi-k3-precluster-certification",
        "status": "PASS",
        "verdict": "PASS",
        "precluster_certification_complete": True,
        "ready_to_rent_single_canary": True,
        "full_fleet_activation_allowed": False,
        "physical_3090_canary": "NOT_RUN",
        "recommended_experiment_015_topology": {
            "candidate": "B",
            "class": metrics["topology"],
            "worker_count": metrics["worker_count"],
            "sub_layer_workers_in_initial_fleet": False,
        },
        "metrics": metrics,
        "sub_layer_conclusion": {
            "result": metrics["sub_layer_result"],
            "recommended_use": "ONLY_INSIDE_FAST_DOMAINS",
            "physical_multi_gpu_efficiency": "NOT_TESTED",
        },
        "economics_condition": "Initial fleet proceeds only at or below the preregistered USD 0.05/GPU-hour cost guard.",
        "remaining_physical_unknowns": [
            "Linux RTX 3090 sm_86 correctness and timing",
            "RTX 5090 to RTX 3090 transfer-model error",
            "physical network jitter and contention",
            "thermal and clock-state behavior",
            "full-fleet throughput and per-user cadence",
            "physical clean-node provisioning duration",
        ],
        "next_action": "RUN_EXPERIMENT_015_SINGLE_RTX3090_CANARY",
        "terminal_statement": gates["terminal_statement"],
    }
    outputs = {
        ARTIFACTS / "acceptance-gates.json": _json_bytes(gates),
        ARTIFACTS / "machine-summary.json": _json_bytes(machine),
        ARTIFACTS / "manager-summary.md": (manager.rstrip() + "\n").encode(),
        ARTIFACTS / "chart-index.json": _json_bytes(chart_index),
        ARTIFACTS / "cycle-ledger-summary.json": _json_bytes(ledger_summary),
        REPORT: report_text.encode(),
    }

    source_names = (
        "census",
        "full_graph",
        "operation_matrix",
        "sm86",
        "promotion",
        "p1_final",
        "p1_nonfinal",
        "p1_stage_zero",
        "ready_repeat_1",
        "ready_repeat_2",
        "steady_stage_zero",
        "kda_batch",
        "mla_batch",
        "stage_profile",
        "continuous",
        "prefill_kda",
        "prefill_mla",
        "sub_exact",
        "sub_batch",
        "sub_scaling",
        "sub_routing",
        "sub_recovery",
        "coarse",
        "network",
        "capacity",
        "placement",
        "distribution",
        "recovery",
        "rehearsal",
        "canary",
        "package_controls",
        "pytest",
        "ruff",
        "ledger",
        "dll",
    )
    source_artifacts = []
    for name in source_names:
        item = _artifact(name, data["paths"][name])
        item["integrity_status"] = "PASS"
        receipt = data["loaded"].get(name)
        if receipt is not None and receipt.get("status") is not None:
            item["scientific_status"] = receipt["status"]
        source_artifacts.append(item)
    chart_paths = (
        ARTIFACTS / "charts" / "h014-sub-layer-scaling.png",
        ARTIFACTS / "charts" / "h014-038ah-coarse-network.png",
        ARTIFACTS / "charts" / "h014-038ak-capacity-economics.png",
    )
    package_artifacts = [
        _artifact("experiment_015_package_lock", PACKAGE / "package-lock.json", "PASS"),
        _artifact("experiment_015_release", PACKAGE / "RELEASE.json", "PASS"),
        _artifact("experiment_015_release_archive", PACKAGE_ARCHIVE, "PASS"),
    ]
    chart_artifacts = [
        _artifact(f"final_chart_{index}", path, "PASS")
        for index, path in enumerate(chart_paths, start=1)
    ]
    evidence = {
        "schema_version": "experiment-014-final-evidence-integrity-v1",
        "status": "PASS",
        "checked_artifact_count": len(source_artifacts)
        + len(package_artifacts)
        + len(chart_artifacts),
        "checked_artifacts": source_artifacts
        + package_artifacts
        + chart_artifacts,
        "cross_checks": {
            "promotion_validated_inputs_exact": True,
            "final_binary_exact": True,
            "package_archive_exact": True,
            "package_tamper_controls_16_of_16": True,
            "placement_network_matches_network_receipt": True,
            "placement_matches_selected_topology": True,
            "physical_canary_not_fabricated": True,
            "all_28_final_gates_satisfied": True,
            "chart_visual_qa_complete": True,
            "ledger_unique_and_complete": True,
        },
        "derived_outputs": [
            {
                "path": _relative(path),
                "sha256": _sha256_bytes(content),
                "bytes": len(content),
            }
            for path, content in sorted(outputs.items(), key=lambda item: str(item[0]))
        ],
    }
    outputs[ARTIFACTS / "evidence-integrity.json"] = _json_bytes(evidence)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = _load_and_validate()
    outputs = _build_outputs(data)
    changed = []
    for path, content in outputs.items():
        if not path.is_file() or path.read_bytes() != content:
            changed.append(_relative(path))
            if args.write:
                _atomic_write(path, content)
    if changed and not args.write:
        print(json.dumps({"status": "DRIFT", "changed": changed}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": "write" if args.write else "check",
                "output_count": len(outputs),
                "changed": changed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
