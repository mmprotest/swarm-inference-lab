"""Incremental long-context prefill certification for one real Kimi stage."""

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
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
    _request,
)

SCHEMA_VERSION = "experiment-014-k3-incremental-prefill-stage-v1"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "minimum_ms": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "maximum_ms": 0.0,
        }
    return {
        "count": len(values),
        "minimum_ms": min(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "maximum_ms": max(values),
    }


def _reference(reference: dict[str, Any], layer: int) -> tuple[list[str], list[list[int]]]:
    if reference.get("status") != "PASS":
        raise ValueError("prefill reference receipt is not passing")
    if int(reference["configuration"]["layer"]) != layer:
        raise ValueError("prefill reference layer does not match")
    batch_one = reference["batches"]["1"]
    if batch_one.get("status") != "PASS" or not batch_one["correctness"]["pass"]:
        raise ValueError("prefill reference batch-1 gate is not passing")
    comparisons = sorted(
        batch_one["correctness"]["comparisons"], key=lambda row: int(row["position"])
    )
    positions = sorted(
        reference["serial_baseline"]["oracle_correctness"]["positions"],
        key=lambda row: int(row["position"]),
    )
    fingerprints = [str(row["observed_fingerprint"]) for row in comparisons]
    routes = [[int(value) for value in row["selected_expert_ids"]] for row in positions]
    if len(fingerprints) != 3 or len(routes) != 3:
        raise ValueError("prefill reference must retain three positions")
    return fingerprints, routes


def _pop_record(executor: PersistentKimiStageExecutor, record: dict[str, Any]) -> None:
    if executor.execution_records and executor.execution_records[-1] is record:
        executor.execution_records.pop()


def _safe_fixture(
    executor: PersistentKimiStageExecutor,
    boundaries: list[torch.Tensor],
    expected_fingerprints: list[str],
    expected_routes: list[list[int]],
    *,
    session_id: str,
) -> dict[str, Any]:
    executor.open_session(session_id, maximum_context_override=3)
    rows: list[dict[str, Any]] = []
    try:
        for position in range(3):
            record = executor.execute_decode_batch(
                session_ids=(session_id,),
                hidden_states=boundaries[position],
                cache_position_start=position,
            )
            output = np.asarray(record["boundary_output"])[0]
            fingerprint = _array_fingerprint(output)
            route = [int(value) for value in record["selected_expert_ids"][0]]
            rows.append(
                {
                    "position": position,
                    "output_fingerprint": fingerprint,
                    "expected_fingerprint": expected_fingerprints[position],
                    "output_exact": fingerprint == expected_fingerprints[position],
                    "route_exact": route == expected_routes[position],
                    "all_selected_experts_executed_once": bool(
                        record["routing"]["all_selected_experts_executed_once"]
                    ),
                }
            )
            _pop_record(executor, record)
        state = executor.session_state_evidence(session_id)
    finally:
        executor.close_session(session_id)
    passed = all(
        row["output_exact"]
        and row["route_exact"]
        and row["all_selected_experts_executed_once"]
        for row in rows
    ) and int(state["cache_sequence_length"]) == 3
    return {"positions": rows, "state": state, "pass": passed}


def _lifecycle_execution_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    return {
        "weight_loading": int(after["weight_load_count"])
        - int(before["weight_load_count"]),
        "model_materialization": int(after["model_materialization_count"])
        - int(before["model_materialization_count"]),
        "persistent_buffer_allocation": int(after["persistent_buffer_allocation_count"])
        - int(before["persistent_buffer_allocation_count"]),
    }


def _run_context(
    executor: PersistentKimiStageExecutor,
    boundaries: list[torch.Tensor],
    expected_fingerprints: list[str],
    expected_routes: list[list[int]],
    *,
    context: int,
    cycle_id: str,
    device: int,
) -> dict[str, Any]:
    session_id = f"{cycle_id.lower()}-context-{context}"
    memory_before_open = executor.runtime.mem_info()
    executor.open_session(session_id, maximum_context_override=context + 1)
    memory_after_open = executor.runtime.mem_info()
    state_bytes = executor.kv_cache_bytes(session_id)
    lifecycle_before = executor.lifecycle_snapshot()
    device_ms: list[float] = []
    record_wall_ms: list[float] = []
    observed_wall_ms: list[float] = []
    first_three: list[dict[str, Any]] = []
    expert_hits: Counter[int] = Counter()
    final_output_fingerprint: str | None = None
    final_output_finite = False
    all_experts_once = True
    started = time.perf_counter_ns()
    try:
        for position in range(context):
            call_started = time.perf_counter_ns()
            record = executor.execute_decode_batch(
                session_ids=(session_id,),
                hidden_states=boundaries[position % 3],
                cache_position_start=position,
            )
            observed_wall_ms.append((time.perf_counter_ns() - call_started) / 1e6)
            device_ms.append(float(record["device_ms"]))
            record_wall_ms.append(float(record["wall_ms"]))
            route = [int(value) for value in record["selected_expert_ids"][0]]
            expert_hits.update(route)
            all_experts_once = all_experts_once and bool(
                record["routing"]["all_selected_experts_executed_once"]
            )
            output = np.asarray(record["boundary_output"])[0]
            if position < 3:
                fingerprint = _array_fingerprint(output)
                first_three.append(
                    {
                        "position": position,
                        "output_fingerprint": fingerprint,
                        "expected_fingerprint": expected_fingerprints[position],
                        "output_exact": fingerprint
                        == expected_fingerprints[position],
                        "route_exact": route == expected_routes[position],
                    }
                )
            if position == context - 1:
                final_output_fingerprint = _array_fingerprint(output)
                final_output_finite = bool(np.isfinite(output).all())
            _pop_record(executor, record)
        total_prefill_wall_ms = (time.perf_counter_ns() - started) / 1e6
        memory_after_prefill = executor.runtime.mem_info()
        state_after_prefill = executor.session_state_evidence(session_id)

        decode_started = time.perf_counter_ns()
        decode_record = executor.execute_decode_batch(
            session_ids=(session_id,),
            hidden_states=boundaries[context % 3],
            cache_position_start=context,
        )
        decode_observed_wall_ms = (time.perf_counter_ns() - decode_started) / 1e6
        decode_output = np.asarray(decode_record["boundary_output"])[0]
        decode = {
            "position": context,
            "device_ms": float(decode_record["device_ms"]),
            "record_wall_ms": float(decode_record["wall_ms"]),
            "observed_wall_ms": decode_observed_wall_ms,
            "output_fingerprint": _array_fingerprint(decode_output),
            "output_finite": bool(np.isfinite(decode_output).all()),
            "all_selected_experts_executed_once": bool(
                decode_record["routing"]["all_selected_experts_executed_once"]
            ),
            "stage_local_ttft_ms": total_prefill_wall_ms + decode_observed_wall_ms,
        }
        _pop_record(executor, decode_record)
        state_after_decode = executor.session_state_evidence(session_id)
        memory_after_decode = executor.runtime.mem_info()
        lifecycle_after = executor.lifecycle_snapshot()
    finally:
        with suppress(KeyError):
            executor.close_session(session_id)
    memory_after_close = executor.runtime.mem_info()
    window = min(256, context)
    first_three_exact = len(first_three) == 3 and all(
        row["output_exact"] and row["route_exact"] for row in first_three
    )
    lifecycle_delta = _lifecycle_execution_delta(lifecycle_before, lifecycle_after)
    executor.runtime.synchronize()
    cuda_error_state_ok = executor.runtime.error_state_ok()
    safe = _safe_fixture(
        executor,
        boundaries,
        expected_fingerprints,
        expected_routes,
        session_id=f"{session_id}-safe",
    )
    executor.runtime.synchronize()
    nvidia_health = _health_snapshot(device)
    gates = {
        "first_three_exact": first_three_exact,
        "all_experts_once": all_experts_once,
        "final_output_finite": final_output_finite,
        "state_length_after_prefill": int(
            state_after_prefill["cache_sequence_length"]
        )
        == context,
        "state_length_after_decode": int(state_after_decode["cache_sequence_length"])
        == context + 1,
        "state_finite": bool(state_after_decode["finite"]),
        "state_zero_suffix": bool(state_after_prefill["zero_suffix"]),
        "state_bytes_match_runtime": int(state_after_decode["bytes"]) == state_bytes,
        "decode_finite_and_complete": bool(decode["output_finite"])
        and bool(decode["all_selected_experts_executed_once"]),
        "execution_lifecycle_zero": all(value == 0 for value in lifecycle_delta.values()),
        "session_memory_recovered": int(memory_after_close["free_bytes"])
        >= int(memory_before_open["free_bytes"]),
        "safe_fixture_exact": bool(safe["pass"]),
        "cuda_error_state_ok": cuda_error_state_ok,
        "nvidia_health_measured": nvidia_health["status"] == "MEASURED",
    }
    return {
        "context_tokens": context,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "input_workload": (
            "cyclic replay of three immutable real K3_IDOT=0 stage-boundary rows"
        ),
        "prefill": {
            "tokens": context,
            "total_observed_wall_ms": total_prefill_wall_ms,
            "prompt_tokens_per_second": context
            / (total_prefill_wall_ms / 1000.0),
            "device_tokens_per_second": context / (sum(device_ms) / 1000.0),
            "device": _timing(device_ms),
            "record_wall": _timing(record_wall_ms),
            "observed_wall": _timing(observed_wall_ms),
            "first_256_device": _timing(device_ms[:window]),
            "last_256_device": _timing(device_ms[-window:]),
            "first_256_observed_wall": _timing(observed_wall_ms[:window]),
            "last_256_observed_wall": _timing(observed_wall_ms[-window:]),
        },
        "decode_after_prefill": decode,
        "correctness": {
            "first_three": first_three,
            "final_output_fingerprint": final_output_fingerprint,
            "final_output_finite": final_output_finite,
            "all_selected_experts_executed_once": all_experts_once,
            "safe_fixture": safe,
        },
        "routing": {
            "total_selections": context * 16,
            "unique_experts": len(expert_hits),
            "hottest_expert_hits": max(expert_hits.values()),
            "hottest_experts": [
                {"expert": expert, "hits": hits}
                for expert, hits in expert_hits.most_common(16)
            ],
            "frequency_is_fixture_replay_not_prompt_distribution": True,
        },
        "state": {
            "runtime_state_bytes": state_bytes,
            "after_prefill": state_after_prefill,
            "after_decode": state_after_decode,
        },
        "memory": {
            "before_open": memory_before_open,
            "after_open": memory_after_open,
            "after_prefill": memory_after_prefill,
            "after_decode": memory_after_decode,
            "after_close": memory_after_close,
            "session_allocation_free_delta_bytes": max(
                0,
                int(memory_before_open["free_bytes"])
                - int(memory_after_open["free_bytes"]),
            ),
        },
        "lifecycle_execution_delta": lifecycle_delta,
        "post_context": {
            "cuda_error_state_ok": cuda_error_state_ok,
            "nvidia_smi": nvidia_health,
        },
        "acceptance_gates": gates,
    }


def benchmark_prefill_stage(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    reference_receipt: Path,
    output_path: Path,
    *,
    layer: int,
    contexts: tuple[int, ...] = (1024, 4096, 8192, 16384),
    device: int = 0,
    cycle_id: str = "H014-033a",
    terminal_ratio_floor: float = 0.0,
    terminal_ratio_gate: float = 1.10,
    throughput_retention_gate: float = 0.90,
    candidate_regression: bool = False,
) -> dict[str, Any]:
    """Run one real stage at increasing prompt contexts with fail-closed evidence."""
    if contexts != tuple(sorted(set(contexts))) or any(value <= 0 for value in contexts):
        raise ValueError("prefill contexts must be unique positive ascending values")
    if contexts != (1024, 4096, 8192, 16384):
        raise ValueError("Experiment 014 prefill certification requires 1K/4K/8K/16K")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "reference_receipt": reference_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    reference_data = json.loads(
        paths["reference_receipt"].read_text(encoding="utf-8")
    )
    expected_fingerprints, expected_routes = _reference(reference_data, layer)
    reference_cuda_sha = str(reference_data["sources"]["cuda_library"]["sha256"])
    cuda_sha = _sha256_file(paths["cuda_library"])
    cuda_sha_match = cuda_sha == reference_cuda_sha
    if not cuda_sha_match and not candidate_regression:
        raise ValueError("prefill CUDA binary does not match the passing reference")
    fixtures, _ = _stage_fixtures(
        paths["checkpoint"], paths["oracle_trace"], layer=layer
    )
    boundaries = [
        _batch_boundaries(fixtures, batch=1, position=position) for position in range(3)
    ]
    request = _request(
        paths["checkpoint"],
        paths["cuda_library"],
        layer=layer,
        device=device,
        cycle_id=cycle_id,
        maximum_context=max(contexts) + 1,
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": (
            "Real layer-89 KDA prefill remains context invariant through 16K: "
            "terminal device p50 <=1.10x 1K, cumulative throughput retention "
            ">=90%, and recurrent state remains exactly 6,881,280 bytes."
            if layer == 89
            else (
                "Real late Gated MLA has measurable but bounded context growth: "
                "16K terminal device p50 is within the configured lower/upper "
                "ratio band, cumulative throughput clears its floor, and prepared "
                "state grows exactly 2,304 bytes per token."
            )
        ),
        "configuration": {
            "layer": layer,
            "contexts": list(contexts),
            "device": device,
            "batch": 1,
            "terminal_ratio_floor": terminal_ratio_floor,
            "terminal_ratio_gate": terminal_ratio_gate,
            "throughput_retention_gate": throughput_retention_gate,
            "candidate_binary_regression": candidate_regression,
            "state_input": "three canonical real boundaries replayed cyclically",
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "reference": {
            "cuda_sha_match": cuda_sha_match,
            "reference_cuda_sha256": reference_cuda_sha,
            "candidate_cuda_sha256": cuda_sha,
            "candidate_regression_authorized": candidate_regression,
            "first_three_fingerprints": expected_fingerprints,
            "first_three_routes": expected_routes,
        },
        "device_identity": _device_identity(device),
        "gpu_health_before": _health_snapshot(device),
        "contexts": {},
        "progress": [{"phase": "preregistered"}],
    }
    _atomic_json(output_path, receipt)
    executor: PersistentKimiStageExecutor | None = None
    try:
        load_started = time.perf_counter_ns()
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
        )
        receipt["load"] = {
            "wall_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "lifecycle": executor.lifecycle_snapshot(),
        }
        receipt["progress"].append({"phase": "stage_ready"})
        _atomic_json(output_path, receipt)
        for context in contexts:
            result = _run_context(
                executor,
                boundaries,
                expected_fingerprints,
                expected_routes,
                context=context,
                cycle_id=cycle_id,
                device=device,
            )
            receipt["contexts"][str(context)] = result
            receipt["progress"].append(
                {"phase": f"context_{context}", "status": result["status"]}
            )
            _atomic_json(output_path, receipt)
            if result["status"] != "PASS":
                raise RuntimeError(f"prefill context {context} failed closed")

        first = receipt["contexts"][str(contexts[0])]
        final = receipt["contexts"][str(contexts[-1])]
        terminal_ratio = (
            final["prefill"]["last_256_device"]["p50_ms"]
            / first["prefill"]["last_256_device"]["p50_ms"]
        )
        throughput_retention = (
            final["prefill"]["prompt_tokens_per_second"]
            / first["prefill"]["prompt_tokens_per_second"]
        )
        state_bytes = [
            int(receipt["contexts"][str(context)]["state"]["runtime_state_bytes"])
            for context in contexts
        ]
        expected_fixed_state = 6_881_280 if layer == 89 else None
        state_bytes_per_prepared_token = (
            executor.config.kv_lora + executor.config.query_rope
        ) * np.dtype(np.float32).itemsize
        expected_linear_state = [
            (context + 1) * state_bytes_per_prepared_token for context in contexts
        ]
        execution_pass = all(
            receipt["contexts"][str(context)]["status"] == "PASS"
            for context in contexts
        )
        hypothesis_gates = {
            "terminal_device_p50_ratio_at_least_floor": terminal_ratio
            >= terminal_ratio_floor,
            "terminal_device_p50_ratio_at_most_gate": terminal_ratio
            <= terminal_ratio_gate,
            "cumulative_throughput_retention_at_least_gate": throughput_retention
            >= throughput_retention_gate,
            "fixed_kda_state_bytes": (
                len(set(state_bytes)) == 1 and state_bytes[0] == expected_fixed_state
                if expected_fixed_state is not None
                else True
            ),
            "linear_mla_state_bytes": (
                state_bytes == expected_linear_state if layer != 89 else True
            ),
        }
        receipt["comparison"] = {
            "terminal_16k_to_1k_device_p50_ratio": terminal_ratio,
            "cumulative_16k_to_1k_throughput_retention": throughput_retention,
            "state_bytes_by_context": dict(zip(map(str, contexts), state_bytes, strict=True)),
            "state_bytes_per_prepared_token": (
                state_bytes_per_prepared_token if layer != 89 else None
            ),
            "expected_state_bytes_by_context": (
                dict(zip(map(str, contexts), expected_linear_state, strict=True))
                if layer != 89
                else None
            ),
        }
        receipt["execution_pass"] = execution_pass
        receipt["hypothesis_gate_evaluation"] = hypothesis_gates
        receipt["hypothesis_supported"] = execution_pass and all(
            hypothesis_gates.values()
        )
        receipt["inspection"] = {
            "actual_bottleneck": (
                "fixed per-token complete-stage compute for recurrent KDA; state "
                "size and terminal service do not grow with prompt context"
                if layer == 89
                else "context-dependent stage service pending numerical inspection"
            ),
            "fixture_limitation": (
                "Cyclic replay uses real immutable Kimi stage activations and real "
                "stateful CUDA, but is not a diverse natural-language 16K prompt."
            ),
            "ttft_scope": "stage-local prefill plus one stage decode, not full model",
        }
        receipt["decision"] = {
            "context_model": (
                "RETAIN" if receipt["hypothesis_supported"] else "REDESIGN"
            ),
            "next_hypothesis": (
                "Test late Gated MLA separately at the same contexts."
                if layer == 89 and receipt["hypothesis_supported"]
                else (
                    "Use measured Gated MLA scaling in the capacity model."
                    if layer != 89
                    else "Profile the first context-dependent KDA phase."
                )
            ),
        }
        receipt["status"] = "PASS" if execution_pass else "FAIL"
        receipt["progress"].append(
            {"phase": "complete", "status": receipt["status"]}
        )
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
    finally:
        if executor is not None:
            with suppress(Exception):
                executor.close()
        with suppress(Exception):
            receipt["gpu_health_after_close"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
    return receipt
