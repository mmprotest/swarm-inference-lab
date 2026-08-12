"""Incremental long-context complete-stage batching for H014-034a.

This harness intentionally changes no model arithmetic.  It tests the production
row-cooperative stage path at a populated Gated-MLA cache, retains every batch
size before arming the next one, and performs the post-size health protocol that
became mandatory after H014-027v.
"""

from __future__ import annotations

import json
import time
import traceback
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import (
    _batch_boundaries,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.persistent_stages import (
    _stage_fixtures,
)
from swarm_inference.experiments.experiment_014.prefill_stage import (
    _lifecycle_execution_delta,
    _pop_record,
    _reference,
    _safe_fixture,
    _timing,
)
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
    _request,
)

SCHEMA_VERSION = "experiment-014-k3-contextual-batch-v1"
CERTIFIED_BATCHES = (1, 2, 4, 8)
MINIMUM_BATCH8_CAPACITY_GAIN = 1.25
MLA_STATE_BYTES_PER_TOKEN = 2_304


def _phase_for_row(row: int, batch: int) -> int:
    """Give every non-singleton stream an identical peer when possible."""
    if batch == 1:
        return 0
    return (row // 2) % 3


def _inputs(boundaries: list[torch.Tensor], phases: tuple[int, ...], position: int) -> torch.Tensor:
    return torch.cat(
        [boundaries[(position + phase) % len(boundaries)] for phase in phases],
        dim=0,
    )


def _selected_routes(record: dict[str, Any]) -> list[list[int]]:
    return [[int(value) for value in row] for row in record["selected_expert_ids"]]


def _paired_exact(
    outputs: np.ndarray, routes: list[list[int]], phases: tuple[int, ...]
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for phase in sorted(set(phases)):
        rows = [index for index, value in enumerate(phases) if value == phase]
        anchor = rows[0]
        for row in rows[1:]:
            comparisons.append(
                {
                    "phase": phase,
                    "anchor_row": anchor,
                    "compared_row": row,
                    "output_bit_exact": bool(np.array_equal(outputs[anchor], outputs[row])),
                    "route_exact": routes[anchor] == routes[row],
                }
            )
    return {
        "comparisons": comparisons,
        "pass": all(row["output_bit_exact"] and row["route_exact"] for row in comparisons),
        "vacuous_for_batch_one": not comparisons,
    }


def _state_pairs_exact(states: list[dict[str, Any]], phases: tuple[int, ...]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for phase in sorted(set(phases)):
        rows = [index for index, value in enumerate(phases) if value == phase]
        anchor = rows[0]
        for row in rows[1:]:
            comparisons.append(
                {
                    "phase": phase,
                    "anchor_row": anchor,
                    "compared_row": row,
                    "active_prefix_fingerprint_exact": states[anchor]["active_prefix_fingerprint"]
                    == states[row]["active_prefix_fingerprint"],
                    "full_allocation_fingerprint_exact": states[anchor]["fingerprint"]
                    == states[row]["fingerprint"],
                    "length_exact": states[anchor]["cache_sequence_length"]
                    == states[row]["cache_sequence_length"],
                }
            )
    return {
        "comparisons": comparisons,
        "pass": all(
            row["active_prefix_fingerprint_exact"]
            and row["full_allocation_fingerprint_exact"]
            and row["length_exact"]
            for row in comparisons
        ),
        "vacuous_for_batch_one": not comparisons,
    }


def _prime_workspace(
    executor: PersistentKimiStageExecutor,
    boundaries: list[torch.Tensor],
    expected_fingerprints: list[str],
    expected_routes: list[list[int]],
    *,
    batch: int,
    cycle_id: str,
) -> dict[str, Any]:
    before = executor.runtime.mem_info()
    repeats: list[dict[str, Any]] = []
    for repeat in range(2):
        session_ids = tuple(
            f"{cycle_id.lower()}-b{batch}-prime-{repeat}-{row}" for row in range(batch)
        )
        for session_id in session_ids:
            executor.open_session(session_id, maximum_context_override=3)
        rows: list[dict[str, Any]] = []
        try:
            for position in range(3):
                record = executor.execute_decode_batch(
                    session_ids=session_ids,
                    hidden_states=torch.cat([boundaries[position]] * batch, dim=0),
                    cache_position_starts=(position,) * batch,
                )
                outputs = np.asarray(record["boundary_output"])
                routes = _selected_routes(record)
                rows.append(
                    {
                        "position": position,
                        "outputs_exact": all(
                            _array_fingerprint(outputs[row]) == expected_fingerprints[position]
                            for row in range(batch)
                        ),
                        "routes_exact": all(
                            routes[row] == expected_routes[position] for row in range(batch)
                        ),
                        "all_selected_experts_executed_once": bool(
                            record["routing"]["all_selected_experts_executed_once"]
                        ),
                    }
                )
                _pop_record(executor, record)
            states = [executor.session_state_evidence(session_id) for session_id in session_ids]
        finally:
            for session_id in session_ids:
                with suppress(KeyError):
                    executor.close_session(session_id)
        executor.runtime.synchronize()
        after = executor.runtime.mem_info()
        repeats.append(
            {
                "repeat": repeat + 1,
                "rows": rows,
                "state_full_fingerprints_equal": len(
                    {str(state["fingerprint"]) for state in states}
                )
                == 1,
                "state_active_prefix_fingerprints_equal": len(
                    {str(state["active_prefix_fingerprint"]) for state in states}
                )
                == 1,
                "state_lengths_exact": all(
                    int(state["cache_sequence_length"]) == 3 for state in states
                ),
                "cuda_error_state_ok": executor.runtime.error_state_ok(),
                "memory_after_close": after,
            }
        )
    first_free = int(repeats[0]["memory_after_close"]["free_bytes"])
    second_free = int(repeats[1]["memory_after_close"]["free_bytes"])
    exact = all(
        all(
            row["outputs_exact"]
            and row["routes_exact"]
            and row["all_selected_experts_executed_once"]
            for row in repeat["rows"]
        )
        and repeat["state_full_fingerprints_equal"]
        and repeat["state_active_prefix_fingerprints_equal"]
        and repeat["state_lengths_exact"]
        and repeat["cuda_error_state_ok"]
        for repeat in repeats
    )
    return {
        "before": before,
        "repeats": repeats,
        "first_use_high_water_bytes": max(0, int(before["free_bytes"]) - first_free),
        "second_prime_growth_bytes": max(0, first_free - second_free),
        "stable_high_water": second_free >= first_free,
        "exact": exact,
        "pass": exact and second_free >= first_free,
    }


def _run_size(
    executor: PersistentKimiStageExecutor,
    boundaries: list[torch.Tensor],
    expected_fingerprints: list[str],
    expected_routes: list[list[int]],
    *,
    batch: int,
    context: int,
    warmup: int,
    iterations: int,
    cycle_id: str,
    device: int,
    expected_context_output_fingerprint: str,
    expected_context_state_fingerprint: str,
) -> dict[str, Any]:
    phases = tuple(_phase_for_row(row, batch) for row in range(batch))
    maximum_context = context + 1 + warmup + iterations
    session_ids = tuple(f"{cycle_id.lower()}-b{batch}-stream-{row}" for row in range(batch))
    workspace_prime = _prime_workspace(
        executor,
        boundaries,
        expected_fingerprints,
        expected_routes,
        batch=batch,
        cycle_id=cycle_id,
    )
    if not workspace_prime["pass"]:
        return {
            "batch": batch,
            "status": "FAIL",
            "workspace_prime": workspace_prime,
            "failure": "batch workspace failed exact or stable repeated priming",
        }
    memory_before_open = executor.runtime.mem_info()
    for session_id in session_ids:
        executor.open_session(session_id, maximum_context_override=maximum_context)
    memory_after_open = executor.runtime.mem_info()
    lifecycle_before = executor.lifecycle_snapshot()
    prefill_device_ms: list[float] = []
    prefill_wall_ms: list[float] = []
    all_experts_once = True
    sampled_pair_checks: list[dict[str, Any]] = []
    checkpoints = {0, 1, 2, 1_023, 4_095, context - 1}
    prefill_started = time.perf_counter_ns()
    states: list[dict[str, Any]] = []
    try:
        for position in range(context):
            call_started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=session_ids,
                hidden_states=_inputs(boundaries, phases, position),
                cache_position_starts=(position,) * batch,
            )
            prefill_wall_ms.append((time.perf_counter_ns() - call_started) / 1e6)
            prefill_device_ms.append(float(record["device_ms"]))
            all_experts_once = all_experts_once and bool(
                record["routing"]["all_selected_experts_executed_once"]
            )
            if position in checkpoints:
                outputs = np.asarray(record["boundary_output"])
                pair_check = _paired_exact(outputs, _selected_routes(record), phases)
                pair_check["position"] = position
                sampled_pair_checks.append(pair_check)
            _pop_record(executor, record)
        prefill_total_wall_ms = (time.perf_counter_ns() - prefill_started) / 1e6

        validation = executor.execute_decode_batch(
            session_ids=session_ids,
            hidden_states=_inputs(boundaries, phases, context),
            cache_position_starts=(context,) * batch,
        )
        validation_outputs = np.asarray(validation["boundary_output"])
        validation_routes = _selected_routes(validation)
        validation_pairs = _paired_exact(validation_outputs, validation_routes, phases)
        phase_zero_rows = [row for row, phase in enumerate(phases) if phase == 0]
        phase_zero_validation = [
            {
                "row": row,
                "output_fingerprint": _array_fingerprint(validation_outputs[row]),
                "expected_fingerprint": expected_context_output_fingerprint,
                "output_exact": _array_fingerprint(validation_outputs[row])
                == expected_context_output_fingerprint,
            }
            for row in phase_zero_rows
        ]
        all_experts_once = all_experts_once and bool(
            validation["routing"]["all_selected_experts_executed_once"]
        )
        _pop_record(executor, validation)
        validation_states = [
            executor.session_state_evidence(session_id) for session_id in session_ids
        ]
        phase_zero_state = [
            {
                "row": row,
                "active_prefix_fingerprint": validation_states[row]["active_prefix_fingerprint"],
                "full_allocation_fingerprint": validation_states[row]["fingerprint"],
                "expected_fingerprint": expected_context_state_fingerprint,
                "active_prefix_fingerprint_exact": validation_states[row][
                    "active_prefix_fingerprint"
                ]
                == expected_context_state_fingerprint,
            }
            for row in phase_zero_rows
        ]

        for index in range(warmup):
            position = context + 1 + index
            record = executor.execute_decode_batch(
                session_ids=session_ids,
                hidden_states=_inputs(boundaries, phases, position),
                cache_position_starts=(position,) * batch,
            )
            all_experts_once = all_experts_once and bool(
                record["routing"]["all_selected_experts_executed_once"]
            )
            _pop_record(executor, record)

        device_ms: list[float] = []
        record_wall_ms: list[float] = []
        observed_wall_ms: list[float] = []
        retained_pairs: list[dict[str, Any]] = []
        expert_hits: Counter[int] = Counter()
        unique_experts: list[int] = []
        native_calls = 0
        for index in range(iterations):
            position = context + 1 + warmup + index
            call_started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=session_ids,
                hidden_states=_inputs(boundaries, phases, position),
                cache_position_starts=(position,) * batch,
            )
            observed_wall_ms.append((time.perf_counter_ns() - call_started) / 1e6)
            device_ms.append(float(record["device_ms"]))
            record_wall_ms.append(float(record["wall_ms"]))
            routes = _selected_routes(record)
            for route in routes:
                expert_hits.update(route)
                all_experts_once = all_experts_once and len(route) == 16 and len(set(route)) == 16
            routing = record["routing"]
            all_experts_once = all_experts_once and bool(
                routing["all_selected_experts_executed_once"]
            )
            unique_experts.append(int(routing["unique_experts"]))
            native_calls += int(routing["native_routed_expert_calls"])
            pair_check = _paired_exact(np.asarray(record["boundary_output"]), routes, phases)
            pair_check["position"] = position
            retained_pairs.append(pair_check)
            _pop_record(executor, record)

        states = [executor.session_state_evidence(session_id) for session_id in session_ids]
        lifecycle_after = executor.lifecycle_snapshot()
    finally:
        for session_id in session_ids:
            with suppress(KeyError):
                executor.close_session(session_id)

    memory_after_close = executor.runtime.mem_info()
    lifecycle_delta = _lifecycle_execution_delta(lifecycle_before, lifecycle_after)
    state_pairs = _state_pairs_exact(states, phases)
    expected_length = context + 1 + warmup + iterations
    expected_state_bytes = maximum_context * MLA_STATE_BYTES_PER_TOKEN

    safe = _safe_fixture(
        executor,
        boundaries,
        expected_fingerprints,
        expected_routes,
        session_id=f"{cycle_id.lower()}-b{batch}-safe",
    )
    executor.runtime.synchronize()
    cuda_error_state_ok = executor.runtime.error_state_ok()
    memory_after_safe = executor.runtime.mem_info()
    nvidia_health = _health_snapshot(device)
    device = _timing(device_ms)
    wall = _timing(observed_wall_ms)
    total_selections = batch * iterations * 16
    gates = {
        "workspace_prime_exact": bool(workspace_prime["exact"]),
        "workspace_high_water_stable": bool(workspace_prime["stable_high_water"]),
        "sampled_prefill_pairs_exact": all(bool(row["pass"]) for row in sampled_pair_checks),
        "validation_pairs_exact": bool(validation_pairs["pass"]),
        "retained_pairs_exact": all(bool(row["pass"]) for row in retained_pairs),
        "phase_zero_output_matches_h014_033c": all(
            bool(row["output_exact"]) for row in phase_zero_validation
        ),
        "phase_zero_state_matches_h014_033c": all(
            bool(row["active_prefix_fingerprint_exact"]) for row in phase_zero_state
        ),
        "state_pairs_exact": bool(state_pairs["pass"]),
        "all_selected_experts_once": all_experts_once,
        "state_lengths_exact": all(
            int(state["cache_sequence_length"]) == expected_length for state in states
        ),
        "state_bytes_exact": all(int(state["bytes"]) == expected_state_bytes for state in states),
        "state_finite_and_isolated": all(
            bool(state["finite"]) and bool(state["zero_suffix"]) for state in states
        ),
        "execution_lifecycle_zero": all(value == 0 for value in lifecycle_delta.values()),
        "session_memory_recovered": int(memory_after_close["free_bytes"])
        >= int(memory_before_open["free_bytes"]),
        "safe_fixture_exact": bool(safe["pass"]),
        "cuda_error_state_ok": cuda_error_state_ok,
        "nvidia_health_measured": nvidia_health["status"] == "MEASURED",
    }
    return {
        "batch": batch,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "context_tokens_before_validation": context,
        "phase_by_row": list(phases),
        "maximum_context": maximum_context,
        "workspace_prime": workspace_prime,
        "prefill": {
            "rows": batch * context,
            "rounds": context,
            "total_observed_wall_ms": prefill_total_wall_ms,
            "aggregate_rows_per_second": batch * context / (prefill_total_wall_ms / 1000.0),
            "device": _timing(prefill_device_ms),
            "observed_wall": _timing(prefill_wall_ms),
            "last_256_device": _timing(prefill_device_ms[-256:]),
            "sampled_pair_checks": sampled_pair_checks,
        },
        "validation_at_8k_decode": {
            "position": context,
            "device_ms": float(validation["device_ms"]),
            "wall_ms": float(validation["wall_ms"]),
            "phase_zero_reference": phase_zero_validation,
            "phase_zero_state_reference": phase_zero_state,
            "pair_exactness": validation_pairs,
        },
        "retained": {
            "warmup_rounds": warmup,
            "rounds": iterations,
            "rows": batch * iterations,
            "device": device,
            "record_wall": _timing(record_wall_ms),
            "observed_wall": wall,
            "aggregate_device_rows_per_second": batch * 1000.0 / device["p50_ms"],
            "aggregate_wall_rows_per_second": batch * 1000.0 / wall["p50_ms"],
            "per_row_device_service_ms": device["p50_ms"] / batch,
            "pair_exactness": {
                "all_rounds_exact": all(bool(row["pass"]) for row in retained_pairs),
                "checks": retained_pairs,
            },
        },
        "routing": {
            "total_selections": total_selections,
            "unique_experts_across_retained": len(expert_hits),
            "mean_unique_experts_per_batch": float(np.mean(unique_experts)),
            "repeated_selections_per_batch": batch * 16 - float(np.mean(unique_experts)),
            "native_routed_expert_calls": native_calls,
            "effective_weight_reuse_rows_per_native_call": total_selections / native_calls,
            "fixture_replays_three_real_boundaries": True,
        },
        "state": {
            "bytes_each": expected_state_bytes,
            "bytes_total": expected_state_bytes * batch,
            "expected_length": expected_length,
            "evidence": states,
            "pair_exactness": state_pairs,
        },
        "memory": {
            "before_open": memory_before_open,
            "after_open": memory_after_open,
            "after_close": memory_after_close,
            "after_safe_fixture": memory_after_safe,
            "session_allocation_bytes": max(
                0,
                int(memory_before_open["free_bytes"]) - int(memory_after_open["free_bytes"]),
            ),
        },
        "lifecycle_execution_delta": lifecycle_delta,
        "post_size_checks": {
            "known_safe_fixture": safe,
            "cuda_synchronize": "PASS",
            "cuda_error_state_ok": cuda_error_state_ok,
            "free_vram_bytes": int(memory_after_safe["free_bytes"]),
            "nvidia_smi": nvidia_health,
        },
        "acceptance_gates": gates,
    }


def benchmark_contextual_batch(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    reference_receipt: Path,
    long_context_receipt: Path,
    output_path: Path,
    *,
    layer: int = 91,
    context: int = 8_192,
    batches: tuple[int, ...] = CERTIFIED_BATCHES,
    device: int = 0,
    warmup: int = 5,
    iterations: int = 20,
    cycle_id: str = "H014-034a",
) -> dict[str, Any]:
    """Certify complete-stage batching at a populated real MLA cache."""
    if layer != 91 or context != 8_192 or batches != CERTIFIED_BATCHES:
        raise ValueError("H014-034a requires layer 91, context 8192 and batches 1/2/4/8")
    if warmup < 3 or iterations < 20:
        raise ValueError("H014-034a requires >=3 warmup and >=20 retained rounds")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "reference_receipt": reference_receipt.resolve(),
        "long_context_receipt": long_context_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    reference_data = json.loads(paths["reference_receipt"].read_text(encoding="utf-8"))
    expected_fingerprints, expected_routes = _reference(reference_data, layer)
    long_context = json.loads(paths["long_context_receipt"].read_text(encoding="utf-8"))
    context_reference = long_context["contexts"][str(context)]
    cuda_sha = _sha256_file(paths["cuda_library"])
    provenance = {
        "long_context_status_pass": long_context.get("status") == "PASS",
        "long_context_layer_exact": int(long_context["configuration"]["layer"]) == layer,
        "long_context_context_pass": context_reference.get("status") == "PASS",
        "candidate_sha_exact": str(long_context["sources"]["cuda_library"]["sha256"]) == cuda_sha,
        "reference_status_pass": reference_data.get("status") == "PASS",
    }
    provenance["pass"] = all(provenance.values())
    if not provenance["pass"]:
        raise ValueError("H014-034a provenance is not the passing H014-033c candidate")
    expected_context_output = str(context_reference["decode_after_prefill"]["output_fingerprint"])
    expected_context_state = str(context_reference["state"]["after_decode"]["fingerprint"])

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "The guarded batch-1 MLA absorb used once per independent row safely "
                "supports real 8K complete-stage batches 1/2/4/8, preserves exact "
                "output/route/state, and batch 8 retains >=1.25x aggregate device "
                "capacity versus batch 1 at the same context."
            ),
            "minimum_batch8_capacity_gain": MINIMUM_BATCH8_CAPACITY_GAIN,
        },
        "configuration": {
            "layer": layer,
            "context": context,
            "batches": list(batches),
            "warmup_rounds": warmup,
            "retained_rounds": iterations,
            "device": device,
            "advance_policy": "arm next size only after prior exact and health PASS",
        },
        "implementation": {
            "cuda_change": False,
            "production_path": (
                "session-owned cache append and guarded batch-1 MLA absorb per row; "
                "row-cooperative dense/router/expert execution"
            ),
            "old_batch_and_ragged_abis_modified": False,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "provenance": provenance,
        "expected_8k_decode": {
            "output_fingerprint": expected_context_output,
            "state_fingerprint": expected_context_state,
        },
        "device_identity": _device_identity(device),
        "gpu_health_before": _health_snapshot(device),
        "batches": {},
        "progress": [{"phase": "preregistered"}],
    }
    _atomic_json(output_path, receipt)
    executor: PersistentKimiStageExecutor | None = None
    try:
        if receipt["gpu_health_before"]["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi unavailable before contextual batching")
        fixtures, _ = _stage_fixtures(paths["checkpoint"], paths["oracle_trace"], layer=layer)
        boundaries = [
            _batch_boundaries(fixtures, batch=1, position=position) for position in range(3)
        ]
        request = _request(
            paths["checkpoint"],
            paths["cuda_library"],
            layer=layer,
            device=device,
            cycle_id=cycle_id,
            maximum_context=context + 1 + warmup + iterations,
        )
        load_started = time.perf_counter_ns()
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
        )
        receipt["load"] = {
            "wall_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "resident_device_bytes": executor.resident_device_bytes,
            "weight_fingerprint": executor.weight_fingerprint,
            "lifecycle": executor.lifecycle_snapshot(),
            "device_shared_memory_limits": executor.runtime.shared_memory_limits,
        }
        receipt["progress"].append({"phase": "stage_ready"})
        _atomic_json(output_path, receipt)

        for batch in batches:
            receipt["progress"].append(
                {"phase": f"batch_{batch}_armed", "prior_sizes_passed": True}
            )
            _atomic_json(output_path, receipt)
            result = _run_size(
                executor,
                boundaries,
                expected_fingerprints,
                expected_routes,
                batch=batch,
                context=context,
                warmup=warmup,
                iterations=iterations,
                cycle_id=cycle_id,
                device=device,
                expected_context_output_fingerprint=expected_context_output,
                expected_context_state_fingerprint=expected_context_state,
            )
            receipt["batches"][str(batch)] = result
            receipt["progress"].append(
                {"phase": f"batch_{batch}_retained", "status": result["status"]}
            )
            _atomic_json(output_path, receipt)
            if result["status"] != "PASS":
                raise RuntimeError(f"contextual batch {batch} failed closed")

        batch_one = receipt["batches"]["1"]["retained"]
        batch_eight = receipt["batches"]["8"]["retained"]
        capacity_gain = (
            8.0 * float(batch_one["device"]["p50_ms"]) / float(batch_eight["device"]["p50_ms"])
        )
        best_batch = max(
            batches,
            key=lambda value: float(
                receipt["batches"][str(value)]["retained"]["aggregate_device_rows_per_second"]
            ),
        )
        receipt["summary"] = {
            "execution_pass": True,
            "batch8_capacity_gain_vs_same_context_batch1": capacity_gain,
            "minimum_capacity_gain": MINIMUM_BATCH8_CAPACITY_GAIN,
            "capacity_gate_pass": capacity_gain >= MINIMUM_BATCH8_CAPACITY_GAIN,
            "best_measured_batch": best_batch,
            "best_aggregate_device_rows_per_second": receipt["batches"][str(best_batch)][
                "retained"
            ]["aggregate_device_rows_per_second"],
        }
        receipt["hypothesis_supported"] = bool(receipt["summary"]["capacity_gate_pass"])
        receipt["inspection"] = {
            "actual_bottleneck": (
                "row-serial context-dependent MLA score/value cache scans if batch-8 "
                "capacity gain contracts relative to the short-context H014-030f result"
            ),
            "fixture_limitation": (
                "The workload cyclically replays three immutable real stage boundaries; "
                "it is exact real Kimi computation but not a natural-prompt route distribution."
            ),
        }
        receipt["decision"] = {
            "contextual_batching": "RETAIN",
            "production_batch_at_8k": best_batch,
            "candidate_binary": "QUARANTINED_PENDING_FINAL_P0_P1_REGRESSION",
            "next_hypothesis": (
                "Measure the continuous FIFO with eight 8K-populated streams when batch 8 "
                "is the best contextual aggregate-capacity point; otherwise measure the "
                "best retained size."
            ),
        }
        receipt["status"] = "PASS"
        receipt["progress"].append({"phase": "complete", "status": receipt["status"]})
        _atomic_json(output_path, receipt)
        return receipt
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            if executor is not None:
                executor.runtime.synchronize()
                receipt["cuda_error_state_after_failure"] = executor.runtime.error_state_ok()
                receipt["free_vram_after_failure"] = executor.runtime.mem_info()["free_bytes"]
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
    finally:
        if executor is not None:
            executor.close()
