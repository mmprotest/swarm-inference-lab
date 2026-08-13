"""Physical real-K3 service and streamed-chunk benchmark for Experiment 018."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _numerical_metrics,
)
from swarm_inference.execution.kimi_k3_graph_runtime import _CheckpointReader
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.execution.verification import VerificationBlock
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request
from swarm_inference.experiments.experiment_016.benchmark import (
    _atomic_json,
    _command,
    _gpu_summary,
    _GpuSampler,
    _inputs,
    _source,
    _write_gpu_samples,
)

SCHEMA_VERSION = "experiment-018-physical-microcell-v1"
CHUNK_ROWS = (1, 2, 4, 8)
OVERALL_BLOCKS = (4, 7, 12, 16)
CORRECTNESS_RELATIVE_L2_GATE = 2e-6


def _timing_with_p90(values: Sequence[float]) -> dict[str, float]:
    result = _timing([float(value) for value in values])
    result["p90_ms"] = float(np.percentile(values, 90))
    return result


def representative_plan(checkpoint: Path) -> list[dict[str, Any]]:
    """Choose real KDA/MLA samples inside every fixed eight-layer cell."""

    reader = _CheckpointReader(checkpoint)
    config = reader.config
    plan: list[dict[str, Any]] = []
    for cell in range(12):
        start = cell * 8
        end = min(config.layers, start + 8)
        eligible = [layer for layer in range(start, end) if layer not in (0, 92)]
        kda = [layer for layer in eligible if layer in config.kda_layers]
        mla = [layer for layer in eligible if layer not in config.kda_layers]
        snapshots = [
            layer for layer in kda if layer and layer % config.residual_block == 0
        ]
        if not kda or not mla:
            raise RuntimeError(f"cell {cell} lacks a benchmarkable KDA/MLA sample")
        selected_kda = snapshots[0] if snapshots else kda[0]
        if cell == 11 and 89 in kda:
            selected_kda = 89
        plan.append(
            {
                "microcell_id": cell,
                "layer_start": start,
                "layer_end": end,
                "kda_layer_count": sum(
                    layer in config.kda_layers for layer in range(start, end)
                ),
                "mla_layer_count": sum(
                    layer not in config.kda_layers for layer in range(start, end)
                ),
                "dense_layer_count": int(start == 0),
                "endpoint_layer_count": int(end == config.layers),
                "representative_kda_layer": selected_kda,
                "representative_mla_layer": mla[0],
            }
        )
    return plan


def _phase_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for record in records for name in record["phase_device_ms"]})
    return {
        name: {
            "device": _timing(
                [float(record["phase_device_ms"][name]) for record in records]
            ),
            "wall": _timing_with_p90(
                [float(record["phase_wall_ms"][name]) for record in records]
            ),
        }
        for name in names
    }


def _measure_rows(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    layer: int,
    rows: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    session_id = f"h018-l{layer}-r{rows}-service"
    calls = warmup + iterations + profile_iterations
    executor.open_session(session_id, maximum_context_override=calls * rows)
    wall_values: list[float] = []
    device_values: list[float] = []
    host_values: list[float] = []
    retained: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    position = 0
    try:
        for index in range(warmup + iterations):
            block = VerificationBlock(
                session_id,
                position,
                rows,
                include_bonus_token=False,
            )
            started = time.perf_counter_ns()
            record = executor.execute_verification_block(
                block=block,
                hidden_states=_inputs(fixtures, position=position, rows=rows),
                expert_strategy="expert_major",
            )
            wall_ms = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                device_ms = float(record["device_ms"])
                wall_values.append(wall_ms)
                device_values.append(device_ms)
                host_values.append(max(0.0, wall_ms - device_ms))
                retained.append(record)
            position += rows
        for _ in range(profile_iterations):
            block = VerificationBlock(
                session_id,
                position,
                rows,
                include_bonus_token=False,
            )
            profiles.append(
                executor.execute_verification_block(
                    block=block,
                    hidden_states=_inputs(fixtures, position=position, rows=rows),
                    expert_strategy="expert_major",
                    profile_phases=True,
                )
            )
            position += rows
        state = executor.session_state_evidence(session_id)
    finally:
        with suppress(KeyError):
            executor.close_session(session_id)
    last = retained[-1]
    return {
        "evidence_class": "PHYSICAL",
        "layer": layer,
        "chunk_rows": rows,
        "warmup": warmup,
        "iterations": iterations,
        "profile_iterations": profile_iterations,
        "wall": _timing_with_p90(wall_values),
        "cuda": _timing_with_p90(device_values),
        "host_overhead": _timing_with_p90(host_values),
        "phase_decomposition": _phase_summary(profiles),
        "routing": {
            "mean_unique_experts": statistics.fmean(
                float(record["routing"]["unique_experts"]) for record in retained
            ),
            "mean_native_routed_calls": statistics.fmean(
                float(record["routing"]["native_routed_expert_calls"])
                for record in retained
            ),
            "last": last["routing"],
        },
        "transfers": last["transfers"],
        "synchronization_count": last["synchronization_count"],
        "state": state,
        "no_hot_path_weight_loads": all(
            int(record["weight_loads_during_execute"]) == 0 for record in retained
        ),
        "no_hot_path_materializations": all(
            int(record["materializations_during_execute"]) == 0 for record in retained
        ),
        "no_hot_path_allocations": all(
            int(record["persistent_buffer_allocations_during_execute"]) == 0
            for record in retained
        ),
    }


def _execute_exact_rows(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    session_id: str,
    row_chunks: Sequence[int],
) -> tuple[np.ndarray, list[list[int]], list[list[float]], dict[str, Any]]:
    total_rows = sum(row_chunks)
    executor.open_session(session_id, maximum_context_override=total_rows)
    outputs: list[np.ndarray] = []
    routes: list[list[int]] = []
    weights: list[list[float]] = []
    position = 0
    try:
        for rows in row_chunks:
            block = VerificationBlock(
                session_id,
                position,
                rows,
                include_bonus_token=False,
            )
            record = executor.execute_verification_block(
                block=block,
                hidden_states=_inputs(fixtures, position=position, rows=rows),
                expert_strategy="expert_major",
            )
            outputs.append(np.ascontiguousarray(record["boundary_output"]))
            routes.extend(record["selected_expert_ids"])
            weights.extend(record["selected_weights"])
            position += rows
        state = executor.session_state_evidence(session_id)
    finally:
        with suppress(KeyError):
            executor.close_session(session_id)
    return np.concatenate(outputs, axis=0), routes, weights, state


def _equivalence_sweep(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    layer: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block_candidates in OVERALL_BLOCKS:
        total_rows = block_candidates + 1
        reference_output, reference_routes, reference_weights, reference_state = (
            _execute_exact_rows(
                executor,
                fixtures,
                session_id=f"h018-l{layer}-b{block_candidates}-monolithic",
                row_chunks=(total_rows,),
            )
        )
        for chunk_rows in CHUNK_ROWS:
            if chunk_rows > block_candidates:
                continue
            chunks: list[int] = []
            remaining = total_rows
            while remaining:
                value = min(chunk_rows, remaining)
                chunks.append(value)
                remaining -= value
            output, routes, weights, state = _execute_exact_rows(
                executor,
                fixtures,
                session_id=(
                    f"h018-l{layer}-b{block_candidates}-stream-c{chunk_rows}"
                ),
                row_chunks=chunks,
            )
            metrics = _numerical_metrics(reference_output, output)
            state_fields = (
                "attention_type",
                "bytes",
                "cache_sequence_length",
                "active_prefix_fingerprint",
                "finite",
                "zero_suffix",
            )
            state_exact = all(state.get(key) == reference_state.get(key) for key in state_fields)
            route_exact = routes == reference_routes
            weights_exact = weights == reference_weights
            bit_exact = bool(np.array_equal(reference_output, output))
            passed = (
                float(metrics["relative_l2_error"]) <= CORRECTNESS_RELATIVE_L2_GATE
                and route_exact
                and weights_exact
                and state_exact
            )
            results.append(
                {
                    "evidence_class": "PHYSICAL",
                    "layer": layer,
                    "attention_type": (
                        "KDA" if layer in executor.config.kda_layers else "Gated_MLA"
                    ),
                    "attnres_snapshot_layer": layer % executor.config.residual_block == 0,
                    "candidate_block_size": block_candidates,
                    "accepted_rows": total_rows,
                    "maximum_chunk_rows": chunk_rows,
                    "actual_chunk_rows": chunks,
                    "metrics": metrics,
                    "output_bit_exact": bit_exact,
                    "routes_exact": route_exact,
                    "route_weights_exact": weights_exact,
                    "state_exact": state_exact,
                    "reference_output_fingerprint": _array_fingerprint(reference_output),
                    "streamed_output_fingerprint": _array_fingerprint(output),
                    "reference_state_fingerprint": reference_state.get(
                        "active_prefix_fingerprint"
                    ),
                    "streamed_state_fingerprint": state.get(
                        "active_prefix_fingerprint"
                    ),
                    "pass": passed,
                }
            )
    return results


def _benchmark_layer(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    *,
    layer: int,
    device: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
    run_equivalence: bool,
    progress: Callable[[str, dict[str, Any]], None],
    gpu_sampler: _GpuSampler,
) -> dict[str, Any]:
    fixtures, _expected = _stage_fixtures(checkpoint, oracle_trace, layer=layer)
    base_request = _request(
        checkpoint,
        cuda_library,
        layer=layer,
        device=device,
        cycle_id="H018",
        maximum_context=max(512, (warmup + iterations + profile_iterations) * 8),
    )
    request = base_request.model_copy(
        update={"fast_path_mode": "verification-major", "fast_path_batch_bucket": 17}
    )
    started = time.perf_counter_ns()
    executor = PersistentKimiStageExecutor(
        request=request,
        checkpoint=checkpoint,
        cuda_library=cuda_library,
        device=device,
    )
    try:
        load_wall_ms = (time.perf_counter_ns() - started) / 1e6
        prepare_started = time.perf_counter_ns()
        executor.prepare_for_ready()
        prepare_wall_ms = (time.perf_counter_ns() - prepare_started) / 1e6
        progress(
            f"layer_{layer}_resident",
            {
                "load_wall_ms": load_wall_ms,
                "prepare_wall_ms": prepare_wall_ms,
                "resident_device_bytes": executor.resident_device_bytes,
            },
        )
        gpu_sampler.start()
        try:
            equivalence = (
                _equivalence_sweep(executor, fixtures, layer=layer)
                if run_equivalence
                else []
            )
            if equivalence:
                progress(
                    f"layer_{layer}_chunk_equivalence",
                    {
                        "cases": len(equivalence),
                        "passed": sum(bool(row["pass"]) for row in equivalence),
                        "all_pass": all(bool(row["pass"]) for row in equivalence),
                    },
                )
            service: dict[str, Any] = {}
            for rows in CHUNK_ROWS:
                service[str(rows)] = _measure_rows(
                    executor,
                    fixtures,
                    layer=layer,
                    rows=rows,
                    warmup=warmup,
                    iterations=iterations,
                    profile_iterations=profile_iterations,
                )
                progress(
                    f"layer_{layer}_rows_{rows}",
                    {
                        "wall_p50_ms": service[str(rows)]["wall"]["p50_ms"],
                        "wall_p90_ms": service[str(rows)]["wall"]["p90_ms"],
                        "wall_p99_ms": service[str(rows)]["wall"]["p99_ms"],
                    },
                )
        finally:
            gpu_sampler.stop()
        return {
            "layer": layer,
            "attention_type": (
                "KDA" if layer in executor.config.kda_layers else "Gated_MLA"
            ),
            "attnres_snapshot_layer": layer % executor.config.residual_block == 0,
            "load": {
                "wall_ms": load_wall_ms,
                "prepare_wall_ms": prepare_wall_ms,
                "resident_device_bytes": executor.resident_device_bytes,
                "tracked_device_bytes": executor.tracked_device_bytes,
                "weight_fingerprint": executor.weight_fingerprint,
                "model_weight_loading_excluded_from_service": True,
            },
            "service": service,
            "chunk_equivalence": equivalence,
            "lifecycle": executor.lifecycle_snapshot(),
        }
    finally:
        executor.close()


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    output_path: Path,
    gpu_samples_path: Path,
    *,
    device: int = 0,
    warmup: int = 3,
    iterations: int = 20,
    profile_iterations: int = 5,
    layers: Sequence[int] | None = None,
    equivalence_layers: frozenset[int] = frozenset({84, 89, 91}),
    resume: bool = True,
) -> dict[str, Any]:
    for path in (checkpoint, cuda_library, oracle_trace):
        if not path.exists():
            raise FileNotFoundError(path)
    plan = representative_plan(checkpoint)
    selected = tuple(
        sorted(
            set(layers)
            if layers is not None
            else {
                int(row[key])
                for row in plan
                for key in ("representative_kda_layer", "representative_mla_layer")
            }
        )
    )
    receipt: dict[str, Any]
    if resume and output_path.exists():
        receipt = json.loads(output_path.read_text(encoding="utf-8"))
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cannot resume an incompatible physical receipt")
        receipt["status"] = "RUNNING"
        receipt.pop("failure", None)
        old_source = receipt["sources"]["benchmark_source"]
        new_source = _source(Path(__file__))
        receipt.setdefault("resume_history", []).append(
            {
                "unix_ns": time.time_ns(),
                "completed_layers_retained": sorted(receipt["layers"], key=int),
                "previous_benchmark_source": old_source,
                "current_benchmark_source": new_source,
                "reason": (
                    "move GPU sampling after PREPARE so its monitoring subprocess "
                    "cannot violate the existing thread-quiescence readiness gate; "
                    "timed service implementation unchanged"
                ),
            }
        )
        receipt["sources"]["benchmark_source"] = new_source
    else:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "RUNNING",
            "started_unix_ns": time.time_ns(),
            "configuration": {
                "layers": list(selected),
                "chunk_rows": list(CHUNK_ROWS),
                "overall_candidate_blocks": list(OVERALL_BLOCKS),
                "equivalence_layers": sorted(equivalence_layers),
                "warmup": warmup,
                "iterations": iterations,
                "profile_iterations": profile_iterations,
                "device": device,
                "relative_l2_gate": CORRECTNESS_RELATIVE_L2_GATE,
                "claim_boundary": (
                    "PHYSICAL one-layer-at-a-time RTX 5090 service; no cross-cell "
                    "physical overlap"
                ),
            },
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "host_ram_total_bytes": psutil.virtual_memory().total,
                "git_commit": _command(["git", "rev-parse", "HEAD"]),
                "git_status": _command(["git", "status", "--short"]),
                "nvidia_smi": _command(["nvidia-smi"]),
            },
            "sources": {
                "checkpoint_config": _source(checkpoint / "config.json"),
                "checkpoint_index": _source(
                    checkpoint / "model.safetensors.index.json"
                ),
                "cuda_library": _source(cuda_library),
                "oracle_trace": _source(oracle_trace),
                "benchmark_source": _source(Path(__file__)),
            },
            "microcell_plan": plan,
            "progress": [],
            "layers": {},
        }

    def progress(phase: str, evidence: dict[str, Any]) -> None:
        receipt["progress"].append(
            {"phase": phase, "unix_ns": time.time_ns(), "evidence": evidence}
        )
        _atomic_json(output_path, receipt)
        print(f"[h018] {phase}: {evidence}", flush=True)

    _atomic_json(output_path, receipt)
    sampler = _GpuSampler(device=device)
    if resume and gpu_samples_path.exists():
        with gpu_samples_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sampler.samples.append(
                    {
                        "sample_unix_ns": int(row["sample_unix_ns"]),
                        "timestamp": row["timestamp"],
                        "gpu_utilization_percent": float(
                            row["gpu_utilization_percent"]
                        ),
                        "memory_controller_utilization_percent": float(
                            row["memory_controller_utilization_percent"]
                        ),
                        "vram_used_mib": float(row["vram_used_mib"]),
                        "vram_total_mib": float(row["vram_total_mib"]),
                        "power_w": float(row["power_w"]),
                        "sm_clock_mhz": float(row["sm_clock_mhz"]),
                        "memory_clock_mhz": float(row["memory_clock_mhz"]),
                    }
                )
    try:
        for layer in selected:
            if str(layer) in receipt["layers"]:
                progress(f"layer_{layer}_resume_skip", {"status": "already complete"})
                continue
            receipt["layers"][str(layer)] = _benchmark_layer(
                checkpoint,
                cuda_library,
                oracle_trace,
                layer=layer,
                device=device,
                warmup=warmup,
                iterations=iterations,
                profile_iterations=profile_iterations,
                run_equivalence=layer in equivalence_layers,
                progress=progress,
                gpu_sampler=sampler,
            )
            _atomic_json(output_path, receipt)
        sampler.stop()
        equivalence = [
            row
            for layer in receipt["layers"].values()
            for row in layer["chunk_equivalence"]
        ]
        receipt["chunk_equivalence_pass"] = bool(equivalence) and all(
            bool(row["pass"]) for row in equivalence
        )
        receipt["all_service_layers_complete"] = all(
            str(layer) in receipt["layers"] for layer in selected
        )
        receipt["gpu_sampling"] = _gpu_summary(sampler.samples)
        receipt["status"] = (
            "PASS"
            if receipt["chunk_equivalence_pass"]
            and receipt["all_service_layers_complete"]
            else "FAIL"
        )
        receipt["finished_unix_ns"] = time.time_ns()
        _write_gpu_samples(gpu_samples_path, sampler.samples)
        _atomic_json(output_path, receipt)
        return receipt
    except BaseException as exc:
        sampler.stop()
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["gpu_sampling"] = _gpu_summary(sampler.samples)
        receipt["finished_unix_ns"] = time.time_ns()
        _write_gpu_samples(gpu_samples_path, sampler.samples)
        _atomic_json(output_path, receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-samples", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--layers")
    parser.add_argument("--equivalence-layers", default="84,89,91")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    layers = (
        tuple(int(value) for value in arguments.layers.split(",") if value)
        if arguments.layers
        else None
    )
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.oracle_trace,
        arguments.output,
        arguments.gpu_samples,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        profile_iterations=arguments.profile_iterations,
        layers=layers,
        equivalence_layers=frozenset(
            int(value) for value in arguments.equivalence_layers.split(",") if value
        ),
        resume=not arguments.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
