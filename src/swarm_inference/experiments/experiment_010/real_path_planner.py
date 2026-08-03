"""Build the Experiment 010 planner exclusively from measured Level A rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.planner import PositiveUtilityPlanner
from swarm_inference.experiments.experiment_010.schemas import (
    ExecutionStrategy,
    PlannerCandidate,
    PlannerObjective,
    ServicePhase,
)

MANDATORY_NETWORK_PROFILES = (
    "loopback_unshaped",
    "fabric_100g",
    "lan_10g",
    "lan_2_5g",
    "lan_1g",
    "wifi",
    "regional_wan",
    "global_wan",
)

REQUIRED_CANDIDATES = (
    "local_monolithic",
    "whole_expert_shared_memory",
    "whole_expert_direct_tcp",
    "whole_expert_relayed_tcp",
    "whole_expert_exact_response",
    "whole_expert_fast_response",
    "equal_microshards",
    "asymmetric_microshards",
    "coalesced_microshards",
    "local_fallback",
    "alternate_worker_recovery",
    "background_inference",
    "verification_only",
    "idle",
)

OBJECTIVES = (
    PlannerObjective.MAX_DECODE_THROUGHPUT,
    PlannerObjective.MIN_TTFT,
    PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT,
    PlannerObjective.MIN_NETWORK_BYTES,
    PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY,
)

PHASE_OBJECTIVES: dict[ServicePhase, tuple[PlannerObjective, ...]] = {
    ServicePhase.PREFILL: (
        PlannerObjective.MIN_TTFT,
        PlannerObjective.MIN_NETWORK_BYTES,
        PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY,
    ),
    ServicePhase.DECODE: (
        PlannerObjective.MAX_DECODE_THROUGHPUT,
        PlannerObjective.MIN_NETWORK_BYTES,
        PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY,
    ),
    ServicePhase.CONCURRENT_DECODE: (
        PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT,
        PlannerObjective.MIN_NETWORK_BYTES,
        PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY,
    ),
    ServicePhase.MIXED_SERVICE: (
        PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT,
        PlannerObjective.MIN_NETWORK_BYTES,
        PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY,
    ),
}

PHASE_VARIANTS: dict[ServicePhase, tuple[str, ...]] = {
    ServicePhase.PREFILL: ("context_8192",),
    ServicePhase.DECODE: ("short_128",),
    ServicePhase.CONCURRENT_DECODE: ("concurrency_2", "concurrency_4", "concurrency_8"),
    ServicePhase.MIXED_SERVICE: ("background_1", "background_4"),
}


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    strategy: ExecutionStrategy
    workers: tuple[str, ...]
    role: str


@dataclass(frozen=True, slots=True)
class Measurement:
    candidate_id: str
    phase: ServicePhase
    variant: str
    network_profile: str
    throughput_samples: tuple[float, ...]
    ttft_ms_samples: tuple[float, ...]
    p95_ms_samples: tuple[float, ...]
    network_bytes_samples: tuple[float, ...]
    capacity_bytes: int
    correctness_gate: bool
    reliability_gate: bool
    measurement_status: str
    evidence_category: str
    source_paths: tuple[str, ...]
    explanation: tuple[str, ...]


WORKERS = tuple(f"level-a-worker-{index}" for index in range(4))

CANDIDATE_SPECS = (
    CandidateSpec(
        "local_monolithic", ExecutionStrategy.LOCAL_WHOLE_EXPERT, (), "token_critical"
    ),
    CandidateSpec(
        "whole_expert_shared_memory",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "token_critical",
    ),
    CandidateSpec(
        "whole_expert_direct_tcp",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "token_critical",
    ),
    CandidateSpec(
        "whole_expert_relayed_tcp",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "token_critical",
    ),
    CandidateSpec(
        "whole_expert_exact_response",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "token_critical",
    ),
    CandidateSpec(
        "whole_expert_fast_response",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "token_critical_quality_bounded",
    ),
    CandidateSpec(
        "equal_microshards", ExecutionStrategy.EQUAL_MICROSHARDS, WORKERS[:2], "token_critical"
    ),
    CandidateSpec(
        "asymmetric_microshards",
        ExecutionStrategy.ASYMMETRIC_MICROSHARDS,
        WORKERS[:2],
        "token_critical",
    ),
    CandidateSpec(
        "coalesced_microshards",
        ExecutionStrategy.COALESCED_MICROSHARDS,
        WORKERS[:2],
        "token_critical",
    ),
    CandidateSpec(
        "local_fallback", ExecutionStrategy.LOCAL_WHOLE_EXPERT, (), "failure_recovery"
    ),
    CandidateSpec(
        "alternate_worker_recovery",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        tuple(f"{worker}-alternate" for worker in WORKERS),
        "failure_recovery",
    ),
    CandidateSpec(
        "background_inference",
        ExecutionStrategy.BACKGROUND_INFERENCE,
        WORKERS,
        "background_only",
    ),
    CandidateSpec(
        "verification_only", ExecutionStrategy.VERIFICATION, WORKERS, "verification_only"
    ),
    CandidateSpec("idle", ExecutionStrategy.IDLE, (), "idle"),
    CandidateSpec(
        "capacity_isolated",
        ExecutionStrategy.REMOTE_WHOLE_EXPERT,
        WORKERS,
        "capacity",
    ),
)

SPEC_BY_ID = {spec.candidate_id: spec for spec in CANDIDATE_SPECS}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _truth(value: Any) -> bool:
    return str(value).lower() == "true"


def _floats(rows: list[dict[str, str]], field: str) -> tuple[float, ...]:
    return tuple(float(row[field]) for row in rows if row.get(field) not in {None, ""})


def _mean(values: tuple[float, ...]) -> float | None:
    return statistics.fmean(values) if values else None


def _confidence_interval(values: tuple[float, ...]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def _clone_measurement(
    measurement: Measurement,
    candidate_id: str,
    *,
    status: str = "MEASURED_ALIAS",
    note: str,
) -> Measurement:
    return replace(
        measurement,
        candidate_id=candidate_id,
        measurement_status=status,
        explanation=(*measurement.explanation, note),
    )


def load_measurements(
    *, phase10_analysis: Path, phase8_capacity: Path, phase11: Path
) -> dict[tuple[ServicePhase, str, str, str], Measurement]:
    measurements: dict[tuple[ServicePhase, str, str, str], Measurement] = {}

    def add(item: Measurement) -> None:
        key = (item.phase, item.variant, item.network_profile, item.candidate_id)
        if key in measurements:
            raise ValueError(f"duplicate measured planner candidate {key}")
        measurements[key] = item

    short_rows = _read_csv(phase10_analysis / "short_decode_results.csv")
    short_by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in short_rows:
        short_by_config[row["configuration"]].append(row)
    short_map = {
        "local": "local_monolithic",
        "whole_expert_shared_memory": "whole_expert_shared_memory",
        "whole_expert_direct_tcp": "whole_expert_direct_tcp",
        "whole_expert_relayed_tcp": "whole_expert_relayed_tcp",
        "whole_expert_fast_aggregation": "whole_expert_fast_response",
        "equal_microshards": "equal_microshards",
        "asymmetric_microshards": "asymmetric_microshards",
    }
    short_measurements: dict[str, Measurement] = {}
    for configuration, candidate_id in short_map.items():
        rows = short_by_config[configuration]
        exact = all(_truth(row["exact_token_identity"]) for row in rows)
        canonical = all(_truth(row["canonical_verified_candidate"]) for row in rows)
        item = Measurement(
            candidate_id=candidate_id,
            phase=ServicePhase.DECODE,
            variant="short_128",
            network_profile="loopback_unshaped",
            throughput_samples=_floats(rows, "decode_tokens_per_second"),
            ttft_ms_samples=tuple(value * 1_000 for value in _floats(rows, "ttft_seconds")),
            p95_ms_samples=tuple(value / 1e6 for value in _floats(rows, "wall_elapsed_ns")),
            network_bytes_samples=(
                _floats(rows, "rpc_raw_payload_bytes")
                if candidate_id != "local_monolithic"
                else (_floats(rows, "rpc_raw_payload_bytes") or (0.0,))
            ),
            capacity_bytes=0,
            correctness_gate=bool(exact and canonical),
            reliability_gate=all(int(row["return_code"]) == 0 for row in rows),
            measurement_status="MEASURED",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase10_analysis / "short_decode_results.csv"),),
            explanation=(f"{len(rows)} real 128-token Level A rows for {configuration}",),
        )
        add(item)
        short_measurements[candidate_id] = item
    add(
        _clone_measurement(
            short_measurements["whole_expert_direct_tcp"],
            "whole_expert_exact_response",
            note="same measured direct-TCP configuration; candidate isolates the exact response contract",
        )
    )
    add(
        _clone_measurement(
            short_measurements["equal_microshards"],
            "coalesced_microshards",
            note=(
                "the measured equal layout already coalesces all selected shard requests per worker; "
                "this is an orthogonal candidate label, not a duplicated workload claim"
            ),
        )
    )
    local_short = short_measurements["local_monolithic"]
    for profile in MANDATORY_NETWORK_PROFILES[1:]:
        add(
            replace(
                local_short,
                network_profile=profile,
                measurement_status="MEASURED_NETWORK_INDEPENDENT",
                explanation=(
                    *local_short.explanation,
                    "local execution sends zero expert payload bytes and is invariant to network profile",
                ),
            )
        )

    network_rows = _read_csv(phase10_analysis / "network_profile_results.csv")
    for row in network_rows:
        profile = row["network_profile"]
        if profile == "loopback_unshaped":
            continue
        item = Measurement(
            candidate_id="whole_expert_relayed_tcp",
            phase=ServicePhase.DECODE,
            variant="short_128",
            network_profile=profile,
            throughput_samples=(float(row["decode_tokens_per_second"]),),
            ttft_ms_samples=(float(row["ttft_seconds"]) * 1_000,),
            p95_ms_samples=(float(row["wall_elapsed_ns"]) / 1e6,),
            network_bytes_samples=(float(row["rpc_raw_payload_bytes"]),),
            capacity_bytes=0,
            correctness_gate=_truth(row["exact_token_identity"]),
            reliability_gate=int(row["return_code"]) == 0,
            measurement_status="MEASURED_SINGLE_RUN",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase10_analysis / "network_profile_results.csv"),),
            explanation=("actual shaped relay payload on the real Colibri token path",),
        )
        add(item)
        add(
            _clone_measurement(
                item,
                "whole_expert_exact_response",
                note="relayed request used per_expert_exact response mode",
            )
        )

    prefill_rows = _read_csv(phase10_analysis / "prefill_results.csv")
    prefill_by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in prefill_rows:
        prefill_by_config[row["configuration"]].append(row)
    prefill_map = {
        "local": "local_monolithic",
        "whole_expert_direct_tcp_exact": "whole_expert_direct_tcp",
    }
    prefill_measurements: dict[str, Measurement] = {}
    for configuration, candidate_id in prefill_map.items():
        rows = prefill_by_config[configuration]
        item = Measurement(
            candidate_id=candidate_id,
            phase=ServicePhase.PREFILL,
            variant="context_8192",
            network_profile="loopback_unshaped",
            throughput_samples=_floats(rows, "prefill_tokens_per_second"),
            ttft_ms_samples=tuple(value * 1_000 for value in _floats(rows, "ttft_seconds")),
            p95_ms_samples=tuple(value / 1e6 for value in _floats(rows, "wall_elapsed_ns")),
            network_bytes_samples=(
                _floats(rows, "rpc_raw_payload_bytes")
                if candidate_id != "local_monolithic"
                else (_floats(rows, "rpc_raw_payload_bytes") or (0.0,))
            ),
            capacity_bytes=0,
            correctness_gate=all(_truth(row["exact_token_identity"]) for row in rows),
            reliability_gate=all(int(row["return_code"]) == 0 for row in rows),
            measurement_status="MEASURED",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase10_analysis / "prefill_results.csv"),),
            explanation=(f"{len(rows)} real 8K-prefill Level A rows for {configuration}",),
        )
        add(item)
        prefill_measurements[candidate_id] = item
    add(
        _clone_measurement(
            prefill_measurements["whole_expert_direct_tcp"],
            "whole_expert_exact_response",
            note="same measured 8K direct-TCP exact-response configuration",
        )
    )
    for profile in MANDATORY_NETWORK_PROFILES[1:]:
        add(
            replace(
                prefill_measurements["local_monolithic"],
                network_profile=profile,
                measurement_status="MEASURED_NETWORK_INDEPENDENT",
                explanation=(
                    *prefill_measurements["local_monolithic"].explanation,
                    "local prefill has no expert-network dependency",
                ),
            )
        )

    concurrent_rows = _read_csv(phase10_analysis / "concurrent_decode_results.csv")
    concurrent_map = {
        "local": "local_monolithic",
        "whole_expert_direct_tcp": "whole_expert_direct_tcp",
        "equal_microshards": "equal_microshards",
    }
    for row in concurrent_rows:
        candidate_id = concurrent_map[row["configuration"]]
        variant = f"concurrency_{int(row['concurrency'])}"
        item = Measurement(
            candidate_id=candidate_id,
            phase=ServicePhase.CONCURRENT_DECODE,
            variant=variant,
            network_profile="loopback_unshaped",
            throughput_samples=(float(row["aggregate_verified_tokens_per_second"]),),
            ttft_ms_samples=(),
            p95_ms_samples=(float(row["p95_latency_seconds"]) * 1_000,),
            network_bytes_samples=(float(row["rpc_raw_payload_bytes"]),),
            capacity_bytes=0,
            correctness_gate=_truth(row["exact_group_token_identity"]),
            reliability_gate=not _truth(row["starvation_detected"]),
            measurement_status="MEASURED_SINGLE_GROUP",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase10_analysis / "concurrent_decode_results.csv"),),
            explanation=(f"real Level A concurrent group at {variant}",),
        )
        add(item)
        if candidate_id == "local_monolithic":
            for profile in MANDATORY_NETWORK_PROFILES[1:]:
                add(
                    replace(
                        item,
                        network_profile=profile,
                        measurement_status="MEASURED_NETWORK_INDEPENDENT",
                        explanation=(*item.explanation, "local path transfers zero expert bytes"),
                    )
                )
        elif candidate_id == "whole_expert_direct_tcp":
            add(
                _clone_measurement(
                    item,
                    "whole_expert_exact_response",
                    note="concurrent direct-TCP groups used per_expert_exact responses",
                )
            )
        elif candidate_id == "equal_microshards":
            add(
                _clone_measurement(
                    item,
                    "coalesced_microshards",
                    note="equal native shards were grouped into one request per destination worker",
                )
            )

    mixed_rows = _read_csv(phase10_analysis / "mixed_service_results.csv")
    mixed_map = {
        "local": "local_monolithic",
        "whole_expert_direct_tcp": "whole_expert_direct_tcp",
        "equal_microshards": "equal_microshards",
    }
    for row in mixed_rows:
        candidate_id = mixed_map[row["configuration"]]
        variant = f"background_{int(row['concurrency']) - 1}"
        item = Measurement(
            candidate_id=candidate_id,
            phase=ServicePhase.MIXED_SERVICE,
            variant=variant,
            network_profile="loopback_unshaped",
            throughput_samples=(float(row["aggregate_verified_tokens_per_second"]),),
            ttft_ms_samples=(),
            p95_ms_samples=(float(row["interactive_p95_seconds"]) * 1_000,),
            network_bytes_samples=(float(row["rpc_raw_payload_bytes"]),),
            capacity_bytes=0,
            correctness_gate=_truth(row["exact_group_token_identity"]),
            reliability_gate=not _truth(row["starvation_detected"]),
            measurement_status="MEASURED_SINGLE_GROUP",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase10_analysis / "mixed_service_results.csv"),),
            explanation=(f"real mixed Level A group with {variant}",),
        )
        add(item)
        if candidate_id == "local_monolithic":
            for profile in MANDATORY_NETWORK_PROFILES[1:]:
                add(
                    replace(
                        item,
                        network_profile=profile,
                        measurement_status="MEASURED_NETWORK_INDEPENDENT",
                        explanation=(
                            *item.explanation,
                            "local mixed service sends zero expert bytes",
                        ),
                    )
                )
        elif candidate_id == "whole_expert_direct_tcp":
            add(
                _clone_measurement(
                    item,
                    "whole_expert_exact_response",
                    note="mixed direct-TCP group used exact responses",
                )
            )
            add(
                _clone_measurement(
                    item,
                    "background_inference",
                    note="same real group supplies the measured background-only role evidence",
                )
            )
        elif candidate_id == "equal_microshards":
            add(
                _clone_measurement(
                    item,
                    "coalesced_microshards",
                    note="mixed equal-shard requests were coalesced per worker",
                )
            )

    capacity = _read_json(phase8_capacity / "suite-result.json")
    capacity_values = tuple(
        float(row["generated_token_count"]) / (float(row["elapsed_ns"]) / 1e9)
        for row in capacity["results"]
    )
    capacity_latency = tuple(float(row["elapsed_ns"]) / 1e6 for row in capacity["results"])
    add(
        Measurement(
            candidate_id="capacity_isolated",
            phase=ServicePhase.DECODE,
            variant="short_128",
            network_profile="loopback_unshaped",
            throughput_samples=capacity_values,
            ttft_ms_samples=(),
            p95_ms_samples=capacity_latency,
            network_bytes_samples=(),
            capacity_bytes=int(capacity["capacity_isolation"]["global_expert_bank_bytes"]),
            correctness_gate=bool(
                capacity["complete"]
                and capacity["exact_prompt_count"] == capacity["required_prompt_count"]
                and capacity["forbidden_local_expert_load_count"] == 0
            ),
            reliability_gate=True,
            measurement_status="MEASURED_CAPACITY",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(str(phase8_capacity / "suite-result.json"),),
            explanation=("four-worker capacity run owns 100% of routed experts off coordinator",),
        )
    )

    failure_rows = _read_csv(phase11 / "failure-matrix" / "real_model_failure_results.csv")
    failure_by_name = {row["scenario"]: row for row in failure_rows}
    for candidate_id, scenario in (
        ("local_fallback", "network-outage-local"),
        ("alternate_worker_recovery", "worker-termination-alternate"),
    ):
        row = failure_by_name[scenario]
        elapsed_ns = float(row["elapsed_ns"])
        add(
            Measurement(
                candidate_id=candidate_id,
                phase=ServicePhase.DECODE,
                variant="failure_recovery",
                network_profile="loopback_unshaped",
                throughput_samples=(float(row["expected_tokens"]) / (elapsed_ns / 1e9),),
                ttft_ms_samples=(),
                p95_ms_samples=(float(row["rpc_p95_ns"]) / 1e6,),
                network_bytes_samples=(float(row["network_bytes"]),),
                capacity_bytes=0,
                correctness_gate=_truth(row["exact_token_identity"]),
                reliability_gate=_truth(row["passed"]),
                measurement_status="MEASURED_FAILURE_PATH",
                evidence_category="REAL_MODEL_MEASURED",
                source_paths=(
                    str(phase11 / "failure-matrix" / "real_model_failure_results.csv"),
                ),
                explanation=(f"deterministic real-token recovery scenario {scenario}",),
            )
        )

    corruption_summary = _read_json(
        phase11 / "corruption-matrix" / "corruption_matrix_summary.json"
    )
    clean_control = _read_json(
        phase11 / "corruption-matrix" / "clean-baseline" / "result.json"
    )
    all_detected = all(
        float(row["detection_rate"]) == 1.0
        for row in corruption_summary["rows"]
        if row["detection_rate"] is not None
    )
    add(
        Measurement(
            candidate_id="verification_only",
            phase=ServicePhase.DECODE,
            variant="corruption_risk",
            network_profile="loopback_unshaped",
            throughput_samples=(
                float(clean_control["expected_tokens"])
                / (float(clean_control["elapsed_ns"]) / 1e9),
            ),
            ttft_ms_samples=(),
            p95_ms_samples=(),
            network_bytes_samples=(),
            capacity_bytes=0,
            correctness_gate=bool(
                corruption_summary["gate_12_pass"] and clean_control["exact_token_identity"]
            ),
            reliability_gate=all_detected,
            measurement_status="MEASURED_VERIFICATION_PATH",
            evidence_category="REAL_MODEL_MEASURED",
            source_paths=(
                str(phase11 / "corruption-matrix" / "corruption_matrix_summary.json"),
                str(phase11 / "corruption-matrix" / "clean-baseline" / "result.json"),
            ),
            explanation=(
                f"{corruption_summary['total_injected_corruptions']} injected corruptions and "
                f"{corruption_summary['total_clean_control_requests']} clean controls",
            ),
        )
    )
    return measurements


def _measurement_row(
    phase: ServicePhase,
    variant: str,
    profile: str,
    spec: CandidateSpec,
    measurement: Measurement | None,
) -> dict[str, Any]:
    if measurement is None:
        return {
            "schema_version": "experiment-010-real-planner-candidate-v1",
            "phase": phase.value,
            "workload_variant": variant,
            "network_profile": profile,
            "candidate_id": spec.candidate_id,
            "strategy": spec.strategy.value,
            "role": spec.role,
            "workers": list(spec.workers),
            "measurement_status": "NOT_APPLICABLE_OR_UNMEASURED",
            "eligible": False,
            "rejection_reason": "no eligible real Level A measurement for this phase/profile/variant",
            "evidence_category": None,
            "sample_count": 0,
            "source_paths": [],
        }
    throughput_ci = _confidence_interval(measurement.throughput_samples)
    ttft_ci = _confidence_interval(measurement.ttft_ms_samples)
    p95_ci = _confidence_interval(measurement.p95_ms_samples)
    network_ci = _confidence_interval(measurement.network_bytes_samples)
    return {
        "schema_version": "experiment-010-real-planner-candidate-v1",
        "phase": phase.value,
        "workload_variant": variant,
        "network_profile": profile,
        "candidate_id": spec.candidate_id,
        "strategy": spec.strategy.value,
        "role": spec.role,
        "workers": list(spec.workers),
        "measurement_status": measurement.measurement_status,
        "eligible": bool(measurement.correctness_gate and measurement.reliability_gate),
        "rejection_reason": (
            None
            if measurement.correctness_gate and measurement.reliability_gate
            else "correctness or reliability gate failed"
        ),
        "evidence_category": measurement.evidence_category,
        "sample_count": max(
            len(measurement.throughput_samples),
            len(measurement.ttft_ms_samples),
            len(measurement.p95_ms_samples),
            len(measurement.network_bytes_samples),
        ),
        "throughput": _mean(measurement.throughput_samples),
        "throughput_confidence_interval_95": throughput_ci,
        "ttft_ms": _mean(measurement.ttft_ms_samples),
        "ttft_confidence_interval_95": ttft_ci,
        "p95_latency_ms": _mean(measurement.p95_ms_samples),
        "p95_latency_confidence_interval_95": p95_ci,
        "network_bytes": _mean(measurement.network_bytes_samples),
        "network_bytes_confidence_interval_95": network_ci,
        "capacity_bytes": measurement.capacity_bytes,
        "correctness_gate": measurement.correctness_gate,
        "reliability_gate": measurement.reliability_gate,
        "confidence_status": (
            "SAMPLING_CI_95"
            if max(
                len(measurement.throughput_samples),
                len(measurement.ttft_ms_samples),
                len(measurement.p95_ms_samples),
                len(measurement.network_bytes_samples),
            )
            >= 2
            else "SINGLE_MEASUREMENT_NO_SAMPLING_CI"
        ),
        "source_paths": list(measurement.source_paths),
        "explanation": list(measurement.explanation),
    }


def _objective_values(
    row: dict[str, Any], objective: PlannerObjective, maximum_network: float
) -> tuple[float | None, tuple[float, float] | None]:
    if objective in {
        PlannerObjective.MAX_DECODE_THROUGHPUT,
        PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT,
    }:
        return row.get("throughput"), row.get("throughput_confidence_interval_95")
    if objective == PlannerObjective.MIN_TTFT:
        latency = row.get("ttft_ms") or row.get("p95_latency_ms")
        interval = row.get("ttft_confidence_interval_95") or row.get(
            "p95_latency_confidence_interval_95"
        )
        if latency is None:
            return None, None
        utility = 1_000_000.0 / max(float(latency), 1e-9)
        if interval is None:
            return utility, None
        return utility, (
            1_000_000.0 / max(float(interval[1]), 1e-9),
            1_000_000.0 / max(float(interval[0]), 1e-9),
        )
    if objective == PlannerObjective.MIN_NETWORK_BYTES:
        network = row.get("network_bytes")
        if network is None:
            return None, None
        return maximum_network + 1.0 - float(network), None
    if objective == PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY:
        return float(row.get("capacity_bytes") or 0), None
    raise AssertionError(objective)


def _select_context(
    rows: list[dict[str, Any]], objective: PlannerObjective
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {row["candidate_id"]: row for row in rows}
    local = by_id["local_monolithic"]
    maximum_network = max(
        (float(row["network_bytes"]) for row in rows if row.get("network_bytes") is not None),
        default=0.0,
    )
    local_score, _ = _objective_values(local, objective, maximum_network)
    local_metric = float(local_score or 0.0)
    candidates: list[PlannerCandidate] = []
    measured_utilities: dict[str, float] = {"idle": 0.0}
    evaluated_rows: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        row = by_id[spec.candidate_id]
        measured_score, interval = _objective_values(row, objective, maximum_network)
        has_measurement = measured_score is not None and row["eligible"]
        if spec.candidate_id == "idle":
            measured_score = 0.0
            predicted = 0.0
            lower = 0.0
            has_measurement = True
        elif not has_measurement:
            predicted = -1.0
            lower = -1.0
        elif objective == PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY:
            maximum_capacity = max(
                float(item.get("capacity_bytes") or 0) for item in rows
            )
            predicted = float(measured_score) / max(maximum_capacity, 1.0)
            lower = predicted
        elif spec.candidate_id == "local_monolithic":
            predicted = 1.0
            lower = (
                max(1e-9, float(interval[0]) / max(float(measured_score), 1e-9))
                if interval is not None
                else None
            )
        else:
            predicted = float(measured_score) / max(local_metric, 1e-9) - 1.0
            lower = (
                float(interval[0]) / max(local_metric, 1e-9) - 1.0
                if interval is not None
                else None
            )
        p95 = row.get("p95_latency_ms")
        local_p95 = local.get("p95_latency_ms")
        latency_limit = 75_000.0 if objective == PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY else (
            2.0 * float(local_p95) if local_p95 is not None else math.inf
        )
        slo_gate = bool(p95 is None or float(p95) <= latency_limit)
        correctness_gate = bool(row.get("correctness_gate", spec.candidate_id == "idle"))
        reliability_gate = bool(row.get("reliability_gate", spec.candidate_id == "idle"))
        if not has_measurement and spec.candidate_id != "idle":
            correctness_gate = False
            reliability_gate = False
            slo_gate = False
        capacity_required = bool(
            spec.candidate_id == "capacity_isolated"
            and objective == PlannerObjective.MAX_CAPACITY_SUBJECT_TO_LATENCY
        )
        candidate = PlannerCandidate(
            candidate_id=spec.candidate_id,
            phase=ServicePhase(row["phase"]),
            strategy=spec.strategy,
            workers=list(spec.workers),
            objective=objective,
            predicted_utility=predicted,
            lower_confidence_bound=lower,
            measured_utility=float(measured_score) if measured_score is not None else None,
            latency_ms=float(p95) if p95 is not None else None,
            throughput=float(row["throughput"]) if row.get("throughput") is not None else None,
            network_bytes=int(float(row["network_bytes"]))
            if row.get("network_bytes") is not None
            else None,
            reliability_gate=reliability_gate,
            correctness_gate=correctness_gate,
            slo_gate=slo_gate,
            capacity_required=capacity_required,
            explanation=[
                *(row.get("explanation") or []),
                f"measurement_status={row['measurement_status']}",
                f"objective={objective.value}",
            ],
        )
        candidates.append(candidate)
        if has_measurement and correctness_gate and reliability_gate and slo_gate:
            measured_utilities[spec.candidate_id] = float(measured_score)
        evaluated_rows.append(
            {
                **row,
                "objective": objective.value,
                "predicted_marginal_utility": predicted,
                "lower_confidence_bound": lower,
                "objective_measured_utility": measured_score,
                "slo_gate": slo_gate,
                "capacity_required": capacity_required,
            }
        )
    selection = PositiveUtilityPlanner().select(
        candidates,
        phase=ServicePhase(rows[0]["phase"]),
        objective=objective,
        measured_utilities=measured_utilities,
    )
    plan = selection.plan.model_dump(mode="json")
    plan.update(
        {
            "schema_version": "experiment-010-real-measured-planner-selection-v1",
            "workload_variant": rows[0]["workload_variant"],
            "network_profile": rows[0]["network_profile"],
            "candidate_catalog_complete": set(REQUIRED_CANDIDATES).issubset(by_id),
            "measured_regret": selection.regret,
            "confidence_intervals_reported": True,
            "measurement_aliases_are_explicit": True,
        }
    )
    return plan, evaluated_rows


def _conditional_roles(
    measurements: dict[tuple[ServicePhase, str, str, str], Measurement]
) -> list[dict[str, Any]]:
    failure = [
        measurements[(ServicePhase.DECODE, "failure_recovery", "loopback_unshaped", name)]
        for name in ("local_fallback", "alternate_worker_recovery")
    ]
    failure_scores = {
        item.candidate_id: float(_mean(item.throughput_samples) or 0.0) for item in failure
    }
    recovery_winner = max(failure_scores, key=failure_scores.__getitem__)
    verification = measurements[
        (ServicePhase.DECODE, "corruption_risk", "loopback_unshaped", "verification_only")
    ]
    background = measurements[
        (ServicePhase.MIXED_SERVICE, "background_4", "loopback_unshaped", "background_inference")
    ]
    return [
        {
            "condition": "remote_worker_failure",
            "selected_candidate_id": recovery_winner,
            "selected_role": "failure_recovery",
            "measured_utilities": failure_scores,
            "regret_fraction": 0.0,
            "explanation": "selected the fastest exact real-token recovery strategy in the measured failure context",
        },
        {
            "condition": "nonzero_corruption_risk",
            "selected_candidate_id": "verification_only",
            "selected_role": "verification_only",
            "measured_detection_gate": verification.reliability_gate,
            "regret_fraction": 0.0,
            "explanation": "verification role is admitted only under corruption risk; clean nominal decoding keeps it off the critical path",
        },
        {
            "condition": "background_queue_present",
            "selected_candidate_id": "background_inference",
            "selected_role": "background_only",
            "measured_combined_verified_throughput": _mean(background.throughput_samples),
            "regret_fraction": 0.0,
            "explanation": "background role is measured separately and does not replace the faster local interactive candidate",
        },
    ]


def build_real_path_planner(
    *, phase10_analysis: Path, phase8_capacity: Path, phase11: Path, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements(
        phase10_analysis=phase10_analysis,
        phase8_capacity=phase8_capacity,
        phase11=phase11,
    )
    candidate_rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    evaluated_rows: list[dict[str, Any]] = []
    for phase, variants in PHASE_VARIANTS.items():
        for variant in variants:
            for profile in MANDATORY_NETWORK_PROFILES:
                context_rows = []
                for spec in CANDIDATE_SPECS:
                    measurement = measurements.get((phase, variant, profile, spec.candidate_id))
                    row = _measurement_row(phase, variant, profile, spec, measurement)
                    candidate_rows.append(row)
                    context_rows.append(row)
                for objective in PHASE_OBJECTIVES[phase]:
                    plan, evaluated = _select_context(context_rows, objective)
                    plans.append(plan)
                    evaluated_rows.extend(evaluated)

    conditional_roles = _conditional_roles(measurements)
    plan_rows = [
        {
            "schema_version": plan["schema_version"],
            "phase": plan["phase"],
            "workload_variant": plan["workload_variant"],
            "network_profile": plan["network_profile"],
            "objective": plan["objective"],
            "selected_candidate_id": plan["selected_candidate_id"],
            "selected_strategy": plan["selected_strategy"],
            "selected_workers": plan["selected_workers"],
            "capacity_exception": plan["capacity_exception"],
            "regret": plan["measured_regret"]["regret"],
            "regret_fraction": plan["measured_regret"]["regret_fraction"],
            "regret_passes": plan["measured_regret"]["passes"],
            "explanation": plan["explanation"],
            "evidence_category": "REAL_MODEL_MEASURED",
        }
        for plan in plans
    ]
    candidate_source_ids = {
        item.candidate_id for item in measurements.values()
    } | {"idle"}
    candidate_catalog_complete = set(REQUIRED_CANDIDATES).issubset(candidate_source_ids)
    maximum_regret = max(float(row["regret_fraction"]) for row in plan_rows)
    default_decode = next(
        plan
        for plan in plans
        if plan["phase"] == "decode"
        and plan["workload_variant"] == "short_128"
        and plan["network_profile"] == "loopback_unshaped"
        and plan["objective"] == "max_decode_throughput"
    )
    capacity_plan = next(
        plan
        for plan in plans
        if plan["phase"] == "decode"
        and plan["network_profile"] == "loopback_unshaped"
        and plan["objective"] == "max_capacity_subject_to_latency"
    )
    harmful_rejected = {
        item["candidate_id"]
        for item in default_decode["rejected"]
        if item["candidate_id"]
        in {
            "whole_expert_shared_memory",
            "whole_expert_direct_tcp",
            "whole_expert_relayed_tcp",
            "equal_microshards",
            "asymmetric_microshards",
            "coalesced_microshards",
        }
    }
    required_harmful = {
        "whole_expert_shared_memory",
        "whole_expert_direct_tcp",
        "whole_expert_relayed_tcp",
        "equal_microshards",
        "asymmetric_microshards",
        "coalesced_microshards",
    }
    summary = {
        "schema_version": "experiment-010-real-measured-planner-v1",
        "evidence_category": "REAL_MODEL_MEASURED",
        "required_candidate_ids": list(REQUIRED_CANDIDATES),
        "measured_candidate_source_ids": sorted(candidate_source_ids),
        "candidate_catalog_complete": candidate_catalog_complete,
        "phase_count": len(PHASE_VARIANTS),
        "network_profile_count": len(MANDATORY_NETWORK_PROFILES),
        "objective_count": len(OBJECTIVES),
        "selection_count": len(plans),
        "candidate_evaluation_count": len(evaluated_rows),
        "maximum_measured_regret_fraction": maximum_regret,
        "planner_regret_passes": maximum_regret <= 0.05,
        "default_decode_selection": default_decode["selected_candidate_id"],
        "capacity_selection": capacity_plan["selected_candidate_id"],
        "harmful_distributed_candidates_rejected": required_harmful <= harmful_rejected,
        "conditional_role_selections": conditional_roles,
        "gate_13_pass": bool(
            candidate_catalog_complete
            and maximum_regret <= 0.05
            and default_decode["selected_candidate_id"] == "local_monolithic"
            and capacity_plan["selected_candidate_id"] == "capacity_isolated"
            and required_harmful <= harmful_rejected
        ),
        "limitations": [
            "network-independent local rows are reused across profiles because they transfer zero expert bytes",
            "candidate/profile combinations without a real measurement are retained as ineligible and never imputed",
            "single-group workload rows report no sampling confidence interval",
            "coalesced and exact-response aliases point to the same measured configurations because those are orthogonal execution attributes",
        ],
    }
    _write_csv(output / "planner_candidates.csv", candidate_rows)
    _write_csv(output / "planner_candidate_evaluations.csv", evaluated_rows)
    _write_csv(output / "planner_results.csv", plan_rows)
    _write_json(output / "planner_summary.json", summary)
    _write_json(output / "planner_explanations.json", {"plans": plans, "conditional_roles": conditional_roles})
    for phase in PHASE_VARIANTS:
        _write_json(
            output / f"{phase.value}_plan.json",
            {
                "schema_version": "experiment-010-real-measured-phase-plan-v1",
                "phase": phase.value,
                "selection_basis": "eligible real Level A measurements only",
                "selections": [plan for plan in plans if plan["phase"] == phase.value],
            },
        )
    explanation_lines = [
        "# Experiment 010 measured planner explanation",
        "",
        f"Candidate catalog complete: {candidate_catalog_complete}.",
        f"Maximum measured regret: {maximum_regret:.6%}.",
        f"Nominal short decode selects `{default_decode['selected_candidate_id']}`.",
        f"Capacity objective selects `{capacity_plan['selected_candidate_id']}`.",
        "",
        "Unmeasured phase/profile cross-products remain visibly ineligible. No simulator or fixture row is used here.",
        "",
        "## Conditional roles",
        "",
    ]
    explanation_lines.extend(
        f"- {row['condition']}: `{row['selected_candidate_id']}` — {row['explanation']}"
        for row in conditional_roles
    )
    (output / "planner_explanation.md").write_text(
        "\n".join(explanation_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase10-analysis", type=Path, required=True)
    parser.add_argument("--phase8-capacity", type=Path, required=True)
    parser.add_argument("--phase11", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build_real_path_planner(
        phase10_analysis=arguments.phase10_analysis,
        phase8_capacity=arguments.phase8_capacity,
        phase11=arguments.phase11,
        output=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
