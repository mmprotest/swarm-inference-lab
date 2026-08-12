"""Exact device-resident Kimi MLA DCP benchmark on one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.dcp import dcp_partial_payload_bytes
from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.execution.verification import VerificationBlock
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request
from swarm_inference.experiments.experiment_015.network import NetworkProfile
from swarm_inference.experiments.experiment_016.benchmark import _inputs

SCHEMA_VERSION = "experiment-016-kimi-mla-device-dcp-v1"
CONTEXTS = (2048, 8192, 32768)
DEGREES = (1, 2, 4, 8)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prefill_to(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    if start % 16 or end % 16 or end < start:
        raise ValueError("DCP prefill boundaries must be nondecreasing multiples of 16")
    wall_started = time.perf_counter_ns()
    device_ms = 0.0
    for position in range(start, end, 16):
        record = executor.execute_verification_block(
            block=VerificationBlock(
                session_id=session_id,
                cache_position_start=position,
                candidate_count=16,
                include_bonus_token=False,
            ),
            hidden_states=_inputs(fixtures, position=position, rows=16),
            expert_strategy="expert_major",
            dcp_degree=1,
        )
        if record["device_ms"] is None:
            raise RuntimeError("DCP prefill omitted device timing")
        device_ms += float(record["device_ms"])
    return {
        "start": start,
        "end": end,
        "rows": end - start,
        "calls": (end - start) // 16,
        "wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
        "device_ms": device_ms,
    }


def _execute(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    context: int,
    candidates: int,
    degree: int,
    profile: bool = False,
) -> dict[str, Any]:
    return executor.execute_verification_block(
        block=VerificationBlock(session_id, context, candidates),
        hidden_states=_inputs(fixtures, position=context, rows=candidates + 1),
        expert_strategy="expert_major",
        profile_phases=profile,
        dcp_degree=degree,
    )


def _measure(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    context: int,
    candidates: int,
    degree: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    rows = candidates + 1
    position = context
    wall_ms: list[float] = []
    device_ms: list[float] = []
    retained: list[dict[str, Any]] = []
    for index in range(warmup + iterations):
        started = time.perf_counter_ns()
        record = _execute(
            executor,
            fixtures,
            session_id,
            context=position,
            candidates=candidates,
            degree=degree,
        )
        wall = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            wall_ms.append(wall)
            device_ms.append(float(record["device_ms"]))
            retained.append(record)
        position += rows
    profiles = []
    for _ in range(profile_iterations):
        profiles.append(
            _execute(
                executor,
                fixtures,
                session_id,
                context=position,
                candidates=candidates,
                degree=degree,
                profile=True,
            )
        )
        position += rows
    phase_names = sorted({name for record in profiles for name in record["phase_device_ms"]})
    phases = {
        name: {
            "device": _timing([float(record["phase_device_ms"][name]) for record in profiles]),
            "wall": _timing([float(record["phase_wall_ms"][name]) for record in profiles]),
        }
        for name in phase_names
    }
    wall = _timing(wall_ms)
    device = _timing(device_ms)
    payload = dcp_partial_payload_bytes(query_rows=rows)
    transport = NetworkProfile("h016-low-latency-domain-shaped", rtt_ms=0.25, bandwidth_gbps=25.0)
    shaped_transport_ms = 0.0 if degree == 1 else transport.service_ms(payload)
    return {
        "context_tokens": context,
        "degree": degree,
        "candidate_count": candidates,
        "verification_rows": rows,
        "wall": wall,
        "device": device,
        "latency_per_accepted_token_wall_ms": wall["p50_ms"] / rows,
        "verifier_tokens_per_second": rows * 1000.0 / wall["p50_ms"],
        "phase_decomposition": phases,
        "local_gpu_synchronization_count": retained[-1]["synchronization_count"],
        "partial_payload_bytes_per_worker": payload,
        "shaped_internal_transport_ms": shaped_transport_ms,
        "shaped_wall_p50_ms": wall["p50_ms"] + shaped_transport_ms,
        "maximum_device_memory_growth_bytes_after_warmup": max(
            int(record["device_memory_growth_bytes"]) for record in retained
        ),
        "repeated_execution_allocations_zero": all(
            int(record["persistent_buffer_allocations_during_execute"]) == 0 for record in retained
        ),
        "evidence_class": "MEASURED_LOCAL_GPU_COMPUTE_PLUS_SHAPED_TRANSPORT",
    }


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    output_path: Path,
    *,
    contexts: tuple[int, ...] = CONTEXTS,
    degrees: tuple[int, ...] = DEGREES,
    candidates: int = 7,
    layer: int = 91,
    device: int = 0,
    warmup: int = 2,
    iterations: int = 12,
    profile_iterations: int = 2,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "scope": (
            "complete exact Kimi MLA context-shard compute/reduction on one RTX 5090; "
            "inter-device transport is shaped, not physically measured"
        ),
        "configuration": {
            "contexts": list(contexts),
            "degrees": list(degrees),
            "candidate_count": candidates,
            "verification_rows": candidates + 1,
            "layer": layer,
            "device": device,
            "warmup": warmup,
            "iterations": iterations,
            "profile_iterations": profile_iterations,
        },
        "network_shaping": {
            "domain": "low-latency microcell",
            "rtt_ms": 0.25,
            "jitter_ms": 0.0,
            "bandwidth_gbps": 25.0,
            "packet_loss_percent": 0.0,
        },
        "prefill_segments": [],
        "rows": [],
        "correctness": [],
    }
    _atomic_json(output_path, receipt)
    executor: PersistentKimiStageExecutor | None = None
    try:
        if not contexts or tuple(sorted(contexts)) != contexts:
            raise ValueError("DCP contexts must be non-empty and increasing")
        if any(degree not in DEGREES for degree in degrees) or 1 not in degrees:
            raise ValueError("DCP degrees must be selected from 1,2,4,8 and include 1")
        fixtures, _expected = _stage_fixtures(checkpoint, oracle_trace, layer=layer)
        rows = candidates + 1
        tail = (warmup + iterations + profile_iterations + 4) * rows
        maximum_context = max(contexts) + tail
        base_request = _request(
            checkpoint,
            cuda_library,
            layer=layer,
            device=device,
            cycle_id="H016-DCP-GPU",
            maximum_context=maximum_context,
        )
        request = base_request.model_copy(
            update={
                "fast_path_mode": "verification-major",
                "fast_path_batch_bucket": 17,
            }
        )
        load_started = time.perf_counter_ns()
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=checkpoint,
            cuda_library=cuda_library,
            device=device,
        )
        receipt["load_wall_ms"] = (time.perf_counter_ns() - load_started) / 1e6
        executor.prepare_for_ready()
        receipt["load"] = {
            "resident_device_bytes": executor.resident_device_bytes,
            "tracked_device_bytes": executor.tracked_device_bytes,
            "lifecycle": executor.lifecycle_snapshot(),
        }
        source_id = "h016-dcp-source"
        executor.open_session(source_id, maximum_context_override=maximum_context)
        current = 0
        every_correct = True
        for context in contexts:
            segment = _prefill_to(
                executor,
                fixtures,
                source_id,
                start=current,
                end=context,
            )
            receipt["prefill_segments"].append(segment)
            current = context
            clone_capacity = context + tail
            correctness_ids = {
                degree: f"h016-dcp-{context}-d{degree}-correctness" for degree in degrees
            }
            performance_ids = {
                degree: f"h016-dcp-{context}-d{degree}-performance" for degree in degrees
            }
            for session_id in (*correctness_ids.values(), *performance_ids.values()):
                executor.clone_session_state(
                    source_id,
                    session_id,
                    maximum_context_override=clone_capacity,
                )
            try:
                reference = _execute(
                    executor,
                    fixtures,
                    correctness_ids[1],
                    context=context,
                    candidates=candidates,
                    degree=1,
                )
                reference_state = executor.session_state_evidence(correctness_ids[1])
                for degree in degrees:
                    if degree == 1:
                        actual = reference
                    else:
                        actual = _execute(
                            executor,
                            fixtures,
                            correctness_ids[degree],
                            context=context,
                            candidates=candidates,
                            degree=degree,
                        )
                    metrics = _numerical_metrics(
                        actual["boundary_output"], reference["boundary_output"]
                    )
                    state = executor.session_state_evidence(correctness_ids[degree])
                    route_exact = actual["selected_expert_ids"] == reference["selected_expert_ids"]
                    state_exact = (
                        state["active_prefix_fingerprint"]
                        == reference_state["active_prefix_fingerprint"]
                    )
                    passed = (
                        float(metrics["relative_l2_error"]) <= 2e-5 and route_exact and state_exact
                    )
                    every_correct = every_correct and passed
                    receipt["correctness"].append(
                        {
                            "context_tokens": context,
                            "degree": degree,
                            "metrics": metrics,
                            "route_exact": route_exact,
                            "state_active_prefix_fingerprint_exact": state_exact,
                            "pass": passed,
                        }
                    )
                context_rows = []
                for degree in degrees:
                    row = _measure(
                        executor,
                        fixtures,
                        performance_ids[degree],
                        context=context,
                        candidates=candidates,
                        degree=degree,
                        warmup=warmup,
                        iterations=iterations,
                        profile_iterations=profile_iterations,
                    )
                    context_rows.append(row)
                    receipt["rows"].append(row)
                baseline_wall = float(context_rows[0]["wall"]["p50_ms"])
                for row in context_rows:
                    row["local_gpu_speedup_vs_dcp1"] = baseline_wall / float(row["wall"]["p50_ms"])
                    row["shaped_speedup_vs_dcp1"] = baseline_wall / float(row["shaped_wall_p50_ms"])
                _atomic_json(output_path, receipt)
                print(
                    f"[h016-dcp-gpu] context={context} "
                    + " ".join(
                        f"d{row['degree']}={row['wall']['p50_ms']:.3f}ms" for row in context_rows
                    ),
                    flush=True,
                )
            finally:
                for session_id in (*correctness_ids.values(), *performance_ids.values()):
                    executor.close_session(session_id)
        receipt["correctness_pass"] = every_correct
        receipt["complete_kimi_dcp_compute_gate"] = "PASS" if every_correct else "FAIL"
        receipt["physical_multi_device_communication_gate"] = "NOT_RUN_SINGLE_GPU"
        receipt["status"] = "PASS" if every_correct else "FAIL"
        receipt["finished_unix_ns"] = time.time_ns()
        _atomic_json(output_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(output_path, receipt)
        raise
    finally:
        if executor is not None:
            executor.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="2,8,32")
    parser.add_argument("--degrees", default="1,2,4,8")
    parser.add_argument("--candidates", type=int, default=7)
    parser.add_argument("--layer", type=int, default=91)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--profile-iterations", type=int, default=2)
    arguments = parser.parse_args()
    contexts = tuple(int(value) * 1024 for value in arguments.contexts.split(",") if value)
    degrees = tuple(int(value) for value in arguments.degrees.split(",") if value)
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.oracle_trace,
        arguments.output,
        contexts=contexts,
        degrees=degrees,
        candidates=arguments.candidates,
        layer=arguments.layer,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        profile_iterations=arguments.profile_iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
