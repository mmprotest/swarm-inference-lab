"""Eight-stream continuous decode at a populated real Gated-MLA cache."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _process_snapshot,
    _timing,
    _warm_lifecycle_delta,
)
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.contextual_batch import (
    _phase_for_row,
    _prime_workspace,
    _state_pairs_exact,
)
from swarm_inference.experiments.experiment_014.continuous_batch import (
    PersistentKimiContinuousBatchScheduler,
    _cancel_and_reuse,
    _fixture_tensor,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.persistent_stages import (
    MODEL_CONTENT_FINGERPRINT,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    _CaptureConnectionPool,
    _source_assignment,
    _stage_fixtures,
)
from swarm_inference.experiments.experiment_014.prefill_stage import (
    _pop_record,
    _reference,
    _safe_fixture,
)
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
)
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

SCHEMA_VERSION = "experiment-014-k3-contextual-continuous-v1"
STATIC_CAPACITY_RETENTION_GATE = 0.95
RESPONSE_P99_GATE_MS = 35.0
FORMATION_P99_GATE_MS = 0.5


def _pair_dispatch(dispatch: dict[str, Any], phases: tuple[int, ...]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    rows = dispatch["rows"]
    for phase in sorted(set(phases)):
        members = [row for row, value in enumerate(phases) if value == phase]
        anchor = members[0]
        for row in members[1:]:
            comparisons.append(
                {
                    "phase": phase,
                    "anchor_row": anchor,
                    "compared_row": row,
                    "output_bit_exact": bool(
                        np.array_equal(rows[anchor]["output"], rows[row]["output"])
                    ),
                    "route_exact": rows[anchor]["selected_expert_ids"]
                    == rows[row]["selected_expert_ids"],
                }
            )
    return {
        "comparisons": comparisons,
        "pass": all(row["output_bit_exact"] and row["route_exact"] for row in comparisons),
    }


def _submit_round(
    scheduler: PersistentKimiContinuousBatchScheduler,
    fixtures: list[np.ndarray],
    *,
    batch: int,
    position: int,
    phases: tuple[int, ...],
) -> dict[str, Any]:
    for row in range(batch):
        scheduler.submit(
            f"stream-{row}",
            _fixture_tensor(fixtures, phases[row], position),
            position=position,
        )
    return scheduler.dispatch_once()


async def _benchmark_contextual_continuous(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    reference_receipt: Path,
    long_context_receipt: Path,
    static_contextual_receipt: Path,
    prior_continuous_receipt: Path | None,
    output_path: Path,
    *,
    layer: int,
    context: int,
    batch: int,
    warmup: int,
    iterations: int,
    device: int,
    cycle_id: str,
) -> dict[str, Any]:
    if (layer, context, batch) != (91, 8_192, 8):
        raise ValueError(
            "contextual continuous certification requires layer 91, context 8192 and batch 8"
        )
    if warmup < 3 or iterations < 50:
        raise ValueError(
            "contextual continuous certification requires >=3 warmup and >=50 retained rounds"
        )
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "identity_manifest": identity_manifest.resolve(),
        "graph_certification": graph_certification.resolve(),
        "reference_receipt": reference_receipt.resolve(),
        "long_context_receipt": long_context_receipt.resolve(),
        "static_contextual_receipt": static_contextual_receipt.resolve(),
    }
    if prior_continuous_receipt is not None:
        paths["prior_continuous_receipt"] = prior_continuous_receipt.resolve()
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    cuda_sha = _sha256_file(paths["cuda_library"])
    graph = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    graph_fixture = graph.get("fixture", {})
    reference = json.loads(paths["reference_receipt"].read_text(encoding="utf-8"))
    expected_fingerprints, expected_routes = _reference(reference, layer)
    long_context = json.loads(paths["long_context_receipt"].read_text(encoding="utf-8"))
    context_reference = long_context["contexts"][str(context)]
    static = json.loads(paths["static_contextual_receipt"].read_text(encoding="utf-8"))
    static_batch = static["batches"][str(batch)]
    static_rows_per_second = float(static_batch["retained"]["aggregate_device_rows_per_second"])
    minimum_rows_per_second = static_rows_per_second * STATIC_CAPACITY_RETENTION_GATE
    prior_continuous = (
        json.loads(paths["prior_continuous_receipt"].read_text(encoding="utf-8"))
        if "prior_continuous_receipt" in paths
        else None
    )
    prior_rows_per_second = (
        float(prior_continuous["continuous"]["aggregate_device_rows_per_second"])
        if prior_continuous is not None
        else None
    )
    minimum_improved_rows_per_second = (
        prior_rows_per_second * 1.03 if prior_rows_per_second is not None else None
    )
    provenance = {
        "graph_status_pass": graph.get("status") == "PASS",
        "graph_trace_exact": graph_fixture.get("oracle_trace_sha256")
        == _sha256_file(paths["oracle_trace"]),
        "graph_routes_exact": graph_fixture.get("oracle_routes_sha256")
        == _sha256_file(paths["oracle_routes"]),
        "long_context_pass": long_context.get("status") == "PASS"
        and context_reference.get("status") == "PASS",
        "static_contextual_pass": static.get("status") == "PASS"
        and static_batch.get("status") == "PASS",
        "candidate_sha_exact": str(static["sources"]["cuda_library"]["sha256"])
        == cuda_sha
        == str(long_context["sources"]["cuda_library"]["sha256"]),
        "prior_continuous_pass": prior_continuous is None
        or (
            prior_continuous.get("status") == "PASS"
            and prior_continuous.get("execution_pass") is True
            and str(prior_continuous["sources"]["cuda_library"]["sha256"]) == cuda_sha
        ),
    }
    provenance["pass"] = all(provenance.values())
    if not provenance["pass"]:
        raise ValueError("H014-034e source provenance is not passing and exact")

    expected_context_output = str(context_reference["decode_after_prefill"]["output_fingerprint"])
    expected_context_state = str(context_reference["state"]["after_decode"]["fingerprint"])
    maximum_context = context + warmup + iterations
    phases = tuple(_phase_for_row(row, batch) for row in range(batch))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "Comparator-matched persistent 8K layer-91 streams improve device "
                "capacity >=3% over the prior continuous route mix, retain >=95% of "
                "static batch 8, keep response p99 <35 ms and formation p99 <0.5 ms, "
                "and preserve exact state/output, fairness and cancellation/reuse."
                if prior_continuous is not None
                else (
                    "Eight persistent 8K layer-91 streams retain >=95% of static batch-8 "
                    "device capacity, response p99 <35 ms, formation p99 <0.5 ms, exact "
                    "paired state/output, equal completions and correct cancellation/reuse."
                )
            ),
            "minimum_device_rows_per_second": minimum_rows_per_second,
            "minimum_rows_per_second_vs_prior_continuous": (minimum_improved_rows_per_second),
            "response_p99_gate_ms": RESPONSE_P99_GATE_MS,
            "formation_p99_gate_ms": FORMATION_P99_GATE_MS,
        },
        "configuration": {
            "layer": layer,
            "context": context,
            "active_streams": batch,
            "maximum_batch": batch,
            "warmup_rounds": warmup,
            "retained_rounds": iterations,
            "maximum_context": maximum_context,
            "phase_by_row": list(phases),
        },
        "implementation": {
            "scheduler_change": False,
            "cuda_change": False,
            "preload": "same persistent FIFO and production complete-stage batch path",
            "executor_record_retention_during_preload": 0,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "provenance": provenance,
        "device_identity": _device_identity(device),
        "gpu_health_before": _health_snapshot(device),
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before contextual continuous run")

    assignment = _source_assignment(
        paths["checkpoint"], layer=layer, device=f"native-cuda:{device}"
    )
    runtime = PersistentStageRuntime(
        worker_id=f"{cycle_id.lower()}-worker-{layer:03d}",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=24,
        configured_model_path=paths["checkpoint"],
        configured_model_identity_path=paths["identity_manifest"],
        connection_pool=_CaptureConnectionPool(),  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=f"{cycle_id.lower()}-layer-{layer}",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=maximum_context,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(paths["cuda_library"]),
        native_runtime_library_sha256=cuda_sha,
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(paths["checkpoint"]),
    )
    scheduler: PersistentKimiContinuousBatchScheduler | None = None
    runtime_closed = False
    try:
        load_started = time.perf_counter_ns()
        response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        receipt["load"] = {
            "accepted": response.accepted,
            "wall_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "resident_device_bytes": executor.resident_device_bytes,
            "weight_fingerprint": executor.weight_fingerprint,
            "lifecycle": executor.lifecycle_snapshot(),
        }
        fixtures, _ = _stage_fixtures(paths["checkpoint"], paths["oracle_trace"], layer=layer)
        boundaries = [_fixture_tensor(fixtures, 0, position) for position in range(3)]
        workspace = _prime_workspace(
            executor,
            boundaries,
            expected_fingerprints,
            expected_routes,
            batch=batch,
            cycle_id=cycle_id,
        )
        receipt["batch8_workspace_prime"] = workspace
        retain(
            "loaded_and_batch8_primed",
            accepted=response.accepted,
            workspace_status="PASS" if workspace["pass"] else "FAIL",
        )
        if not response.accepted or not workspace["pass"]:
            raise RuntimeError("stage load or batch-8 workspace prime failed")

        memory_before_streams = executor.runtime.mem_info()
        scheduler = PersistentKimiContinuousBatchScheduler(
            executor,
            maximum_batch=batch,
            maximum_active_streams=batch,
            session_prefix=f"{cycle_id.lower()}-perf",
        )
        for row in range(batch):
            scheduler.open_stream(f"stream-{row}", maximum_context=maximum_context)

        preload_device_ms: list[float] = []
        preload_wall_ms: list[float] = []
        preload_pair_checks: list[dict[str, Any]] = []
        all_experts_once = True
        checkpoints = {0, 1, 2, 1_023, 4_095, context - 1}
        preload_started = time.perf_counter_ns()
        for position in range(context):
            dispatch = _submit_round(
                scheduler,
                fixtures,
                batch=batch,
                position=position,
                phases=phases,
            )
            preload_device_ms.append(float(dispatch["device_ms"]))
            preload_wall_ms.append(float(dispatch["wall_ms"]))
            all_experts_once = all_experts_once and bool(
                dispatch["routing"]["all_selected_experts_executed_once"]
            )
            if position in checkpoints:
                pair = _pair_dispatch(dispatch, phases)
                pair["position"] = position
                preload_pair_checks.append(pair)
            _pop_record(executor, executor.execution_records[-1])
            if position + 1 in (1_024, 4_096):
                retain(
                    f"preload_{position + 1}",
                    rounds=position + 1,
                    pair_checks_pass=all(bool(row["pass"]) for row in preload_pair_checks),
                )
        preload_total_wall_ms = (time.perf_counter_ns() - preload_started) / 1e6
        preload_states = [
            executor.session_state_evidence(scheduler.session_id(f"stream-{row}"))
            for row in range(batch)
        ]
        preload_state_pairs = _state_pairs_exact(preload_states, phases)
        executor.runtime.synchronize()
        preload_health = _health_snapshot(device)
        receipt["preload"] = {
            "rounds": context,
            "rows": context * batch,
            "total_observed_wall_ms": preload_total_wall_ms,
            "aggregate_rows_per_second": context * batch / (preload_total_wall_ms / 1000.0),
            "device": _timing(preload_device_ms),
            "wall": _timing(preload_wall_ms),
            "last_256_device": _timing(preload_device_ms[-256:]),
            "pair_checks": preload_pair_checks,
            "state": preload_states,
            "state_pairs": preload_state_pairs,
            "all_selected_experts_executed_once": all_experts_once,
            "cuda_error_state_ok": executor.runtime.error_state_ok(),
            "free_vram_bytes": executor.runtime.mem_info()["free_bytes"],
            "nvidia_smi": preload_health,
        }
        preload_pass = (
            all(bool(row["pass"]) for row in preload_pair_checks)
            and preload_state_pairs["pass"]
            and all_experts_once
            and all(
                int(state["cache_sequence_length"]) == context
                and bool(state["finite"])
                and bool(state["zero_suffix"])
                for state in preload_states
            )
            and receipt["preload"]["cuda_error_state_ok"]
            and preload_health["status"] == "MEASURED"
        )
        retain("preload_8192", status="PASS" if preload_pass else "FAIL")
        if not preload_pass:
            raise RuntimeError("8K continuous preload failed closed")

        warm_validation: dict[str, Any] | None = None
        for index in range(warmup):
            position = context + index
            dispatch = _submit_round(
                scheduler,
                fixtures,
                batch=batch,
                position=position,
                phases=phases,
            )
            all_experts_once = all_experts_once and bool(
                dispatch["routing"]["all_selected_experts_executed_once"]
            )
            if index == 0:
                phase_zero_rows = [row for row, phase in enumerate(phases) if phase == 0]
                states = [
                    executor.session_state_evidence(scheduler.session_id(f"stream-{row}"))
                    for row in range(batch)
                ]
                warm_validation = {
                    "position": position,
                    "pair_exactness": _pair_dispatch(dispatch, phases),
                    "phase_zero_outputs": [
                        {
                            "row": row,
                            "fingerprint": _array_fingerprint(dispatch["rows"][row]["output"]),
                            "exact": _array_fingerprint(dispatch["rows"][row]["output"])
                            == expected_context_output,
                        }
                        for row in phase_zero_rows
                    ],
                    "phase_zero_states": [
                        {
                            "row": row,
                            "active_prefix_fingerprint": states[row]["active_prefix_fingerprint"],
                            "exact": states[row]["active_prefix_fingerprint"]
                            == expected_context_state,
                        }
                        for row in phase_zero_rows
                    ],
                }
            _pop_record(executor, executor.execution_records[-1])
        if warm_validation is None:
            raise RuntimeError("contextual continuous warm validation was not captured")
        warm_validation_pass = (
            warm_validation["pair_exactness"]["pass"]
            and all(bool(row["exact"]) for row in warm_validation["phase_zero_outputs"])
            and all(bool(row["exact"]) for row in warm_validation["phase_zero_states"])
        )
        receipt["warm_validation"] = warm_validation
        retain(
            "warm_validation",
            status="PASS" if warm_validation_pass else "FAIL",
        )
        if not warm_validation_pass:
            raise RuntimeError("8K warm reference differs from H014-033c")

        before_retained = _process_snapshot(runtime, executor)
        device_ms: list[float] = []
        wall_ms: list[float] = []
        formation_ms: list[float] = []
        last_enqueue_ms: list[float] = []
        queue_ms: list[float] = []
        response_ms: list[float] = []
        completion_ns: dict[str, list[int]] = {f"stream-{row}": [] for row in range(batch)}
        completion_counts = {f"stream-{row}": 0 for row in range(batch)}
        routing_rows: list[dict[str, Any]] = []
        retained_pairs: list[dict[str, Any]] = []
        for index in range(iterations):
            position = context + warmup + index
            dispatch = _submit_round(
                scheduler,
                fixtures,
                batch=batch,
                position=position,
                phases=phases,
            )
            device_ms.append(float(dispatch["device_ms"]))
            wall_ms.append(float(dispatch["wall_ms"]))
            formation_ms.append(float(dispatch["formation_delay_ms"]))
            last_enqueue_ms.append(float(dispatch["last_enqueue_to_start_ms"]))
            routing_rows.append(dispatch["routing"])
            pair = _pair_dispatch(dispatch, phases)
            pair["position"] = position
            retained_pairs.append(pair)
            for row in dispatch["rows"]:
                stream_id = str(row["stream_id"])
                queue_ms.append(float(row["queue_delay_ms"]))
                response_ms.append(float(row["response_ms"]))
                completion_ns[stream_id].append(int(row["completed_ns"]))
                completion_counts[stream_id] += 1
            all_experts_once = all_experts_once and bool(
                dispatch["routing"]["all_selected_experts_executed_once"]
            )
            _pop_record(executor, executor.execution_records[-1])
        after_retained = _process_snapshot(runtime, executor)
        lifecycle_delta = _warm_lifecycle_delta(before_retained, after_retained)
        retained_states = [
            executor.session_state_evidence(scheduler.session_id(f"stream-{row}"))
            for row in range(batch)
        ]
        retained_state_pairs = _state_pairs_exact(retained_states, phases)
        cadence_by_stream: dict[str, dict[str, Any]] = {}
        for stream_id, timestamps in completion_ns.items():
            cadence_by_stream[stream_id] = _timing(
                [
                    (timestamps[index] - timestamps[index - 1]) / 1e6
                    for index in range(1, len(timestamps))
                ]
            )
        device_timing = _timing(device_ms)
        wall_timing = _timing(wall_ms)
        aggregate_device = batch * 1000.0 / device_timing["p50_ms"]
        aggregate_wall = batch * 1000.0 / wall_timing["p50_ms"]
        total_selections = sum(int(row["total_selections"]) for row in routing_rows)
        native_calls = sum(int(row["native_routed_expert_calls"]) for row in routing_rows)
        continuous = {
            "active_streams": batch,
            "retained_rounds": iterations,
            "retained_rows": batch * iterations,
            "device": device_timing,
            "wall": wall_timing,
            "aggregate_device_rows_per_second": aggregate_device,
            "aggregate_wall_rows_per_second": aggregate_wall,
            "per_row_device_service_ms": device_timing["p50_ms"] / batch,
            "static_device_capacity_retention": aggregate_device / static_rows_per_second,
            "capacity_gain_vs_contextual_batch1": static["summary"][
                "batch8_capacity_gain_vs_same_context_batch1"
            ],
            "capacity_gain_vs_prior_continuous": (
                aggregate_device / prior_rows_per_second
                if prior_rows_per_second is not None
                else None
            ),
            "queue_delay": _timing(queue_ms),
            "response": _timing(response_ms),
            "batch_formation_delay": _timing(formation_ms),
            "last_enqueue_to_dispatch": _timing(last_enqueue_ms),
            "per_stream_cadence": cadence_by_stream,
            "fairness": {
                "completion_counts": completion_counts,
                "minimum_completions": min(completion_counts.values()),
                "maximum_completions": max(completion_counts.values()),
                "max_min_completion_ratio": max(completion_counts.values())
                / min(completion_counts.values()),
                "equal_completion_counts": len(set(completion_counts.values())) == 1,
            },
            "routing": {
                "total_selections": total_selections,
                "native_routed_expert_calls": native_calls,
                "effective_weight_reuse_rows_per_native_call": total_selections / native_calls,
                "mean_unique_experts": float(
                    np.mean([int(row["unique_experts"]) for row in routing_rows])
                ),
                "maximum_rows_for_one_expert": max(
                    int(row["maximum_rows_for_one_expert"]) for row in routing_rows
                ),
            },
            "state": {
                "bytes_each": executor.kv_cache_bytes(scheduler.session_id("stream-0")),
                "bytes_total": sum(int(row["bytes"]) for row in retained_states),
                "evidence": retained_states,
                "pair_exactness": retained_state_pairs,
            },
            "pair_exactness": {
                "all_rounds_exact": all(bool(row["pass"]) for row in retained_pairs),
                "checks": retained_pairs,
            },
            "lifecycle_delta": lifecycle_delta,
            "lifecycle_deltas_zero": all(value == 0 for value in lifecycle_delta.values()),
            "scheduler": scheduler.snapshot(),
        }
        receipt["continuous"] = continuous
        retain(
            "continuous_retained",
            device_p50_ms=device_timing["p50_ms"],
            rows_per_second=aggregate_device,
            static_capacity_retention=continuous["static_device_capacity_retention"],
        )

        cancellation = _cancel_and_reuse(executor, scheduler, fixtures, layer=layer, batch=batch)
        receipt["cancellation_and_slot_reuse"] = cancellation
        retain(
            "cancellation_and_slot_reuse",
            status="PASS" if cancellation["pass"] else "FAIL",
        )

        scheduler.close()
        scheduler = None
        memory_after_stream_close = executor.runtime.mem_info()
        safe = _safe_fixture(
            executor,
            boundaries,
            expected_fingerprints,
            expected_routes,
            session_id=f"{cycle_id.lower()}-safe",
        )
        executor.runtime.synchronize()
        health_loaded = _health_snapshot(device)
        receipt["post_run_checks"] = {
            "known_safe_fixture": safe,
            "cuda_synchronize": "PASS",
            "cuda_error_state_ok": executor.runtime.error_state_ok(),
            "memory_before_streams": memory_before_streams,
            "memory_after_stream_close": memory_after_stream_close,
            "session_memory_recovered": int(memory_after_stream_close["free_bytes"])
            >= int(memory_before_streams["free_bytes"]),
            "free_vram_bytes": executor.runtime.mem_info()["free_bytes"],
            "nvidia_smi": health_loaded,
        }
        queue_timing = continuous["queue_delay"]
        response_timing = continuous["response"]
        formation_timing = continuous["batch_formation_delay"]
        execution_pass = (
            all_experts_once
            and all(bool(row["pass"]) for row in retained_pairs)
            and retained_state_pairs["pass"]
            and all(
                int(state["cache_sequence_length"]) == maximum_context
                and bool(state["finite"])
                and bool(state["zero_suffix"])
                for state in retained_states
            )
            and continuous["fairness"]["equal_completion_counts"]
            and continuous["lifecycle_deltas_zero"]
            and cancellation["pass"]
            and safe["pass"]
            and receipt["post_run_checks"]["cuda_error_state_ok"]
            and receipt["post_run_checks"]["session_memory_recovered"]
            and health_loaded["status"] == "MEASURED"
        )
        gate_evaluation = {
            "device_capacity_at_least_95_percent_static": aggregate_device
            >= minimum_rows_per_second,
            "response_p99_below_35_ms": response_timing["p99_ms"] < RESPONSE_P99_GATE_MS,
            "formation_p99_below_0_5_ms": formation_timing["p99_ms"] < FORMATION_P99_GATE_MS,
            "equal_completion_counts": continuous["fairness"]["equal_completion_counts"],
            "device_capacity_at_least_3_percent_above_prior": (
                minimum_improved_rows_per_second is None
                or aggregate_device >= minimum_improved_rows_per_second
            ),
            "observed_queue_p99_ms": queue_timing["p99_ms"],
            "observed_response_p99_ms": response_timing["p99_ms"],
            "observed_formation_p99_ms": formation_timing["p99_ms"],
            "observed_static_capacity_retention": continuous["static_device_capacity_retention"],
            "observed_capacity_gain_vs_prior": continuous["capacity_gain_vs_prior_continuous"],
        }
        hypothesis_supported = execution_pass and all(
            bool(gate_evaluation[name])
            for name in (
                "device_capacity_at_least_95_percent_static",
                "response_p99_below_35_ms",
                "formation_p99_below_0_5_ms",
                "equal_completion_counts",
                "device_capacity_at_least_3_percent_above_prior",
            )
        )
        receipt["execution_pass"] = execution_pass
        receipt["hypothesis_gate_evaluation"] = gate_evaluation
        receipt["hypothesis_supported"] = hypothesis_supported
        receipt["inspection"] = {
            "actual_bottleneck": (
                "complete-stage context-dependent CUDA service when FIFO formation and "
                "queue tails remain small relative to device p99"
            ),
            "fixture_limitation": (
                "Three immutable real stage boundaries are replayed cyclically; physical "
                "arrival jitter and natural-prompt route diversity are not represented."
            ),
        }
        receipt["decision"] = {
            "scheduler": "RETAIN" if execution_pass else "REJECT",
            "production_batch_at_8k": batch if execution_pass else None,
            "next_hypothesis": (
                "Use measured 8K continuous capacity and cadence in the validated P4 "
                "coarse/fine topology model."
            ),
        }
        receipt["status"] = "PASS" if execution_pass else "FAIL"
        await runtime.close()
        runtime_closed = True
        receipt["gpu_health_after_runtime_close"] = _health_snapshot(device)
        retain("complete", status=receipt["status"])
        return receipt
    finally:
        if scheduler is not None:
            with suppress(Exception):
                scheduler.close()
        if not runtime_closed:
            await runtime.close()


def benchmark_contextual_continuous(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    graph_certification: Path,
    reference_receipt: Path,
    long_context_receipt: Path,
    static_contextual_receipt: Path,
    prior_continuous_receipt: Path | None,
    output_path: Path,
    *,
    layer: int = 91,
    context: int = 8_192,
    batch: int = 8,
    warmup: int = 5,
    iterations: int = 50,
    device: int = 0,
    cycle_id: str = "H014-034e",
) -> dict[str, Any]:
    """Run and atomically retain contextual continuous decoding."""
    try:
        return asyncio.run(
            _benchmark_contextual_continuous(
                checkpoint,
                cuda_library,
                oracle_trace,
                oracle_routes,
                identity_manifest,
                graph_certification,
                reference_receipt,
                long_context_receipt,
                static_contextual_receipt,
                prior_continuous_receipt,
                output_path,
                layer=layer,
                context=context,
                batch=batch,
                warmup=warmup,
                iterations=iterations,
                device=device,
                cycle_id=cycle_id,
            )
        )
    except Exception as exc:
        if output_path.exists():
            receipt = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            receipt = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": cycle_id,
                "status": "RUNNING",
                "progress": [],
            }
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
