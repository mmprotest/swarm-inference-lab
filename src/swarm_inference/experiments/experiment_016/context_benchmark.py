"""Exact same-session block-7 verification at an 8K Kimi MLA context."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.execution.verification import VerificationBlock
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request
from swarm_inference.experiments.experiment_016.benchmark import _inputs

SCHEMA_VERSION = "experiment-016-kimi-mla-8k-verification-v1"
BLOCK_SIZES = (1, 2, 4, 7, 12, 16)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prefill(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    context: int,
) -> dict[str, Any]:
    if context % 16:
        raise ValueError("8K context prefill must divide into 16-row exact blocks")
    wall_started = time.perf_counter_ns()
    device_ms: list[float] = []
    for position in range(0, context, 16):
        record = executor.execute_verification_block(
            block=VerificationBlock(
                session_id=session_id,
                cache_position_start=position,
                candidate_count=16,
                include_bonus_token=False,
            ),
            hidden_states=_inputs(fixtures, position=position, rows=16),
            expert_strategy="expert_major",
        )
        if record["device_ms"] is None:
            raise RuntimeError("8K prefill omitted device timing")
        device_ms.append(float(record["device_ms"]))
    return {
        "rows": context,
        "calls": context // 16,
        "wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
        "device_ms": sum(device_ms),
        "last_32_calls": _timing(device_ms[-32:]),
        "state": executor.session_state_evidence(session_id),
    }


def _serial_block(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    position: int,
    *,
    rows: int = 8,
) -> tuple[np.ndarray, list[list[int]], float, float]:
    outputs: list[np.ndarray] = []
    routes: list[list[int]] = []
    device_ms = 0.0
    started = time.perf_counter_ns()
    for row in range(rows):
        result = executor.execute_decode(
            session_id=session_id,
            hidden_states=_inputs(fixtures, position=position + row, rows=1),
            cache_position_start=position + row,
        )
        outputs.append(result.stage_boundary_hidden_states.detach().cpu().numpy()[0].copy())
        record = executor.execution_records[-1]
        routes.append(list(record["selected_expert_ids"]))
        device_ms += float(record["device_ms"])
    return (
        np.stack(outputs),
        routes,
        (time.perf_counter_ns() - started) / 1e6,
        device_ms,
    )


def _measure_serial(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    position: int,
    candidates: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    rows = candidates + 1
    wall_ms: list[float] = []
    device_ms: list[float] = []
    for index in range(warmup + iterations):
        _outputs, _routes, wall, device = _serial_block(
            executor, fixtures, session_id, position, rows=rows
        )
        if index >= warmup:
            wall_ms.append(wall)
            device_ms.append(device)
        position += rows
    wall = _timing(wall_ms)
    return {
        "arm": "A_serial_target_rows",
        "candidate_count": candidates,
        "verification_rows": rows,
        "wall": wall,
        "device": _timing(device_ms),
        "latency_per_accepted_token_wall_ms": wall["p50_ms"] / rows,
        "verifier_tokens_per_second": rows * 1000.0 / wall["p50_ms"],
        "host_device_transfer_count_per_block": 2 * rows,
    }


def _retained_route_summary(records: list[dict[str, Any]], *, rows: int) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    for record in records:
        for route in record["selected_expert_ids"]:
            counts.update(int(value) for value in route)
    block_stats = [record["routing"] for record in records]
    assignment_counts = sorted(counts.values())
    return {
        "scope": "real Kimi layer-91 routing at context positions >=8192",
        "retained_blocks": len(records),
        "retained_positions": rows * len(records),
        "total_assignments": rows * 16 * len(records),
        "unique_experts_across_retained": len(counts),
        "mean_unique_experts_per_block": statistics.fmean(
            float(item["unique_experts"]) for item in block_stats
        ),
        "mean_unique_experts_per_assignment": statistics.fmean(
            float(item["unique_experts_per_assignment"]) for item in block_stats
        ),
        "mean_assignments_per_touched_expert_within_block": statistics.fmean(
            float(item["mean_assignments_per_touched_expert"]) for item in block_stats
        ),
        "mean_adjacent_position_overlap": statistics.fmean(
            float(item["mean_adjacent_position_overlap"]) for item in block_stats
        ),
        "assignments_per_expert_across_retained_p50": float(np.percentile(assignment_counts, 50)),
        "assignments_per_expert_across_retained_p95": float(np.percentile(assignment_counts, 95)),
        "assignments_per_expert_across_retained_p99": float(np.percentile(assignment_counts, 99)),
        "top_experts": [
            {"expert_id": expert, "assignments": count}
            for expert, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
    }


def _measure_block(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    session_id: str,
    *,
    position: int,
    candidates: int,
    strategy: str,
    warmup: int,
    iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    rows = candidates + 1
    wall_ms: list[float] = []
    device_ms: list[float] = []
    records: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for index in range(warmup + iterations):
        started = time.perf_counter_ns()
        record = executor.execute_verification_block(
            block=VerificationBlock(session_id, position, candidates),
            hidden_states=_inputs(fixtures, position=position, rows=rows),
            expert_strategy=strategy,
        )
        observed_wall = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            wall_ms.append(observed_wall)
            if record["device_ms"] is None:
                raise RuntimeError("8K retained block omitted device timing")
            device_ms.append(float(record["device_ms"]))
            records.append(record)
        position += rows
    for _ in range(profile_iterations):
        profiles.append(
            executor.execute_verification_block(
                block=VerificationBlock(session_id, position, candidates),
                hidden_states=_inputs(fixtures, position=position, rows=rows),
                expert_strategy=strategy,
                profile_phases=True,
            )
        )
        position += rows
    names = sorted({name for record in profiles for name in record["phase_device_ms"]})
    phases = {
        name: {
            "device": _timing([float(record["phase_device_ms"][name]) for record in profiles]),
            "wall": _timing([float(record["phase_wall_ms"][name]) for record in profiles]),
        }
        for name in names
    }
    wall = _timing(wall_ms)
    return {
        "arm": (
            "B_device_resident_token_major"
            if strategy == "token_major"
            else "C_verification_major_expert_batching"
        ),
        "candidate_count": candidates,
        "verification_rows": rows,
        "wall": wall,
        "device": _timing(device_ms),
        "latency_per_accepted_token_wall_ms": wall["p50_ms"] / rows,
        "verifier_tokens_per_second": rows * 1000.0 / wall["p50_ms"],
        "routing": _retained_route_summary(records, rows=rows),
        "phase_decomposition": phases,
        "transfers": records[-1]["transfers"],
        "dispatch": records[-1]["dispatch"],
        "synchronization_count": records[-1]["synchronization_count"],
        "all_repeated_execution_allocations_zero": all(
            int(record["persistent_buffer_allocations_during_execute"]) == 0 for record in records
        ),
        "maximum_device_memory_growth_bytes": max(
            int(record["device_memory_growth_bytes"]) for record in records
        ),
    }


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    output_path: Path,
    *,
    context: int = 8192,
    layer: int = 91,
    device: int = 0,
    warmup: int = 3,
    iterations: int = 20,
    profile_iterations: int = 3,
    block_sizes: tuple[int, ...] = BLOCK_SIZES,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "configuration": {
            "context": context,
            "layer": layer,
            "block_sizes": list(block_sizes),
            "verification_rows": [value + 1 for value in block_sizes],
            "warmup": warmup,
            "iterations": iterations,
            "profile_iterations": profile_iterations,
            "device": device,
        },
        "progress": [],
    }
    _atomic_json(output_path, receipt)
    executor: PersistentKimiStageExecutor | None = None
    try:
        fixtures, _expected = _stage_fixtures(checkpoint, oracle_trace, layer=layer)
        maximum_rows = max(block_sizes) + 1
        base_request = _request(
            checkpoint,
            cuda_library,
            layer=layer,
            device=device,
            cycle_id="H016-8K",
            maximum_context=(
                context + (warmup + iterations + profile_iterations + 2) * maximum_rows
            ),
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
        source_id = "h016-8k-source"
        executor.open_session(
            source_id,
            maximum_context_override=context
            + (warmup + iterations + profile_iterations + 2) * maximum_rows,
        )
        receipt["prefill"] = _prefill(executor, fixtures, source_id, context=context)
        receipt["progress"].append({"phase": "8k_prefill_complete"})
        _atomic_json(output_path, receipt)

        receipt["results"] = {}
        every_correct = True
        for candidates in block_sizes:
            rows = candidates + 1
            clone_capacity = context + (warmup + iterations + profile_iterations + 2) * rows
            clone_ids = {
                name: f"h016-8k-c{candidates}-{name.replace('_', '-')}"
                for name in (
                    "serial_correctness",
                    "token_correctness",
                    "expert_correctness",
                    "serial_performance",
                    "token_performance",
                    "expert_performance",
                )
            }
            clones = {
                name: executor.clone_session_state(
                    source_id,
                    session_id,
                    maximum_context_override=clone_capacity,
                )
                for name, session_id in clone_ids.items()
            }
            try:
                serial_output, serial_routes, _wall, _device = _serial_block(
                    executor,
                    fixtures,
                    clone_ids["serial_correctness"],
                    context,
                    rows=rows,
                )
                correctness: dict[str, Any] = {}
                for name, strategy in (
                    ("token_correctness", "token_major"),
                    ("expert_correctness", "expert_major"),
                ):
                    record = executor.execute_verification_block(
                        block=VerificationBlock(clone_ids[name], context, candidates),
                        hidden_states=_inputs(fixtures, position=context, rows=rows),
                        expert_strategy=strategy,
                    )
                    metrics = _numerical_metrics(record["boundary_output"], serial_output)
                    correctness[strategy] = {
                        "metrics": metrics,
                        "route_exact": (record["selected_expert_ids"] == serial_routes),
                        "state_active_prefix_fingerprint_exact": (
                            executor.session_state_evidence(clone_ids[name])[
                                "active_prefix_fingerprint"
                            ]
                            == executor.session_state_evidence(clone_ids["serial_correctness"])[
                                "active_prefix_fingerprint"
                            ]
                        ),
                        "pass": (
                            float(metrics["relative_l2_error"]) <= 2e-6
                            and record["selected_expert_ids"] == serial_routes
                        ),
                    }

                arm_a = _measure_serial(
                    executor,
                    fixtures,
                    clone_ids["serial_performance"],
                    position=context,
                    candidates=candidates,
                    warmup=warmup,
                    iterations=iterations,
                )
                arm_b = _measure_block(
                    executor,
                    fixtures,
                    clone_ids["token_performance"],
                    position=context,
                    candidates=candidates,
                    strategy="token_major",
                    warmup=warmup,
                    iterations=iterations,
                    profile_iterations=profile_iterations,
                )
                arm_c = _measure_block(
                    executor,
                    fixtures,
                    clone_ids["expert_performance"],
                    position=context,
                    candidates=candidates,
                    strategy="expert_major",
                    warmup=warmup,
                    iterations=iterations,
                    profile_iterations=profile_iterations,
                )
                a_p50 = float(arm_a["wall"]["p50_ms"])
                b_p50 = float(arm_b["wall"]["p50_ms"])
                c_p50 = float(arm_c["wall"]["p50_ms"])
                arm_b["speedup_vs_serial"] = a_p50 / b_p50
                arm_c["speedup_vs_serial"] = a_p50 / c_p50
                arm_c["speedup_vs_device_resident_token_major"] = b_p50 / c_p50
                block_correct = all(bool(item["pass"]) for item in correctness.values())
                every_correct = every_correct and block_correct
                receipt["results"][str(candidates)] = {
                    "candidate_count": candidates,
                    "verification_rows": rows,
                    "state_clones": clones,
                    "correctness": correctness,
                    "arms": [arm_a, arm_b, arm_c],
                    "status": "PASS" if block_correct else "FAIL",
                }
                _atomic_json(output_path, receipt)
                print(
                    f"[h016-8k] block={candidates} serial={a_p50:.4f} ms "
                    f"grouped={c_p50:.4f} ms speedup={a_p50 / c_p50:.3f}x",
                    flush=True,
                )
            finally:
                for session_id in clone_ids.values():
                    executor.close_session(session_id)
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
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--layer", type=int, default=91)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--block-sizes", default="1,2,4,7,12,16")
    arguments = parser.parse_args()
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.oracle_trace,
        arguments.output,
        context=arguments.context,
        layer=arguments.layer,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        profile_iterations=arguments.profile_iterations,
        block_sizes=tuple(int(value) for value in arguments.block_sizes.split(",") if value),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
