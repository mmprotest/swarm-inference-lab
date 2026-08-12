"""Physical Kimi K3 verification-major benchmark for Experiment 016."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _timing,
)
from swarm_inference.execution.verification import VerificationBlock
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request

SCHEMA_VERSION = "experiment-016-verification-major-physical-v1"
BLOCK_SIZES = (1, 2, 4, 7, 12, 16)
CORRECTNESS_RELATIVE_L2_GATE = 2e-6


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _command(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _inputs(
    fixtures: list[np.ndarray],
    *,
    position: int,
    rows: int,
) -> torch.Tensor:
    values = np.concatenate(
        [fixtures[(position + row) % len(fixtures)] for row in range(rows)], axis=0
    )
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def _summary(values: list[float]) -> dict[str, float]:
    return _timing([float(value) for value in values])


class _GpuSampler:
    """Low-rate nvidia-smi sampler kept separate from CUDA event timings."""

    def __init__(self, *, device: int, interval_s: float = 0.5) -> None:
        self.device = device
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _GpuSampler:
        self.start()
        return self

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU sampler is already running")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="h016-nvidia-smi")
        self._thread.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None

    def _run(self) -> None:
        query = (
            "timestamp,utilization.gpu,utilization.memory,memory.used,"
            "memory.total,power.draw,clocks.sm,clocks.mem"
        )
        while not self._stop.is_set():
            started = time.time_ns()
            raw = _command(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device),
                ]
            )
            parts = [item.strip() for item in raw.split(",")]
            if len(parts) == 8 and not raw.startswith("UNAVAILABLE"):
                with suppress(ValueError):
                    self.samples.append(
                        {
                            "sample_unix_ns": started,
                            "timestamp": parts[0],
                            "gpu_utilization_percent": float(parts[1]),
                            "memory_controller_utilization_percent": float(parts[2]),
                            "vram_used_mib": float(parts[3]),
                            "vram_total_mib": float(parts[4]),
                            "power_w": float(parts[5]),
                            "sm_clock_mhz": float(parts[6]),
                            "memory_clock_mhz": float(parts[7]),
                        }
                    )
            self._stop.wait(self.interval_s)


def _gpu_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"status": "UNAVAILABLE", "sample_count": 0}
    result: dict[str, Any] = {"status": "MEASURED", "sample_count": len(samples)}
    for source, target in (
        ("gpu_utilization_percent", "gpu_utilization_percent"),
        (
            "memory_controller_utilization_percent",
            "memory_controller_utilization_percent",
        ),
        ("vram_used_mib", "vram_used_mib"),
        ("power_w", "power_w"),
    ):
        values = [float(row[source]) for row in samples]
        result[target] = {
            "minimum": min(values),
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }
    return result


def _write_gpu_samples(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(samples[0]) if samples else ["sample_unix_ns"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    os.replace(temporary, path)


def _serial_correctness(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    rows: int,
    layer: int,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    session_id = f"h016-l{layer}-serial-correctness"
    executor.open_session(session_id, maximum_context_override=rows)
    outputs: list[np.ndarray] = []
    routes: list[list[int]] = []
    try:
        for position in range(rows):
            result = executor.execute_decode(
                session_id=session_id,
                hidden_states=_inputs(fixtures, position=position, rows=1),
                cache_position_start=position,
            )
            outputs.append(result.stage_boundary_hidden_states.detach().cpu().numpy()[0].copy())
            routes.append(list(executor.execution_records[-1]["selected_expert_ids"]))
        state = executor.session_state_evidence(session_id)
    finally:
        executor.close_session(session_id)
    return np.stack(outputs), routes, state


def _block_correctness(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    reference_output: np.ndarray,
    reference_routes: list[list[int]],
    reference_state: dict[str, Any],
    *,
    candidates: int,
    layer: int,
    strategy: str,
) -> dict[str, Any]:
    block = VerificationBlock(
        session_id=f"h016-l{layer}-{strategy}-correctness",
        cache_position_start=0,
        candidate_count=candidates,
    )
    executor.open_session(block.session_id, maximum_context_override=block.row_count)
    try:
        record = executor.execute_verification_block(
            block=block,
            hidden_states=_inputs(fixtures, position=0, rows=block.row_count),
            expert_strategy=strategy,
        )
        output = np.ascontiguousarray(record["boundary_output"])
        state = executor.session_state_evidence(block.session_id)
    finally:
        with suppress(KeyError):
            executor.close_session(block.session_id)
    metrics = _numerical_metrics(output, reference_output)
    route_exact = record["selected_expert_ids"] == reference_routes
    state_geometry_exact = (
        state["cache_sequence_length"] == reference_state["cache_sequence_length"]
        and state["bytes"] == reference_state["bytes"]
        and state["attention_type"] == reference_state["attention_type"]
    )
    return {
        "strategy": strategy,
        "rows": block.row_count,
        "metrics": metrics,
        "route_exact": route_exact,
        "state_geometry_exact": state_geometry_exact,
        "state_active_prefix_fingerprint_exact": (
            state["active_prefix_fingerprint"] == reference_state["active_prefix_fingerprint"]
        ),
        "output_fingerprint": _array_fingerprint(output),
        "serial_output_fingerprint": _array_fingerprint(reference_output),
        "pass": (
            float(metrics["relative_l2_error"]) <= CORRECTNESS_RELATIVE_L2_GATE
            and route_exact
            and state_geometry_exact
        ),
    }


def _measure_serial(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    candidates: int,
    layer: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    rows = candidates + 1
    session_id = f"h016-l{layer}-c{candidates}-serial-performance"
    executor.open_session(
        session_id,
        maximum_context_override=(warmup + iterations) * rows,
    )
    wall_ms: list[float] = []
    device_ms: list[float] = []
    try:
        position = 0
        for block_index in range(warmup + iterations):
            started = time.perf_counter_ns()
            block_device_ms = 0.0
            for _row in range(rows):
                executor.execute_decode(
                    session_id=session_id,
                    hidden_states=_inputs(fixtures, position=position, rows=1),
                    cache_position_start=position,
                )
                block_device_ms += float(executor.execution_records[-1]["device_ms"])
                position += 1
            if block_index >= warmup:
                wall_ms.append((time.perf_counter_ns() - started) / 1e6)
                device_ms.append(block_device_ms)
        state = executor.session_state_evidence(session_id)
    finally:
        executor.close_session(session_id)
    return {
        "arm": "A_serial_target_rows",
        "evidence_class": "MEASURED",
        "candidate_count": candidates,
        "verification_rows": rows,
        "wall": _summary(wall_ms),
        "device": _summary(device_ms),
        "latency_per_candidate_wall_ms": _summary(wall_ms)["p50_ms"] / candidates,
        "latency_per_accepted_token_wall_ms": _summary(wall_ms)["p50_ms"] / rows,
        "verifier_tokens_per_second": rows * 1000.0 / _summary(wall_ms)["p50_ms"],
        "state": state,
        "host_device_transfer_count_per_block": 2 * rows,
        "boundary_bytes_each_direction_per_block": rows * 9 * 7168 * 4,
        "routed_expert_native_calls_per_block": rows * 16,
    }


def _measure_block(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    *,
    candidates: int,
    layer: int,
    strategy: str,
    warmup: int,
    iterations: int,
    profile_iterations: int,
) -> dict[str, Any]:
    rows = candidates + 1
    session_id = f"h016-l{layer}-c{candidates}-{strategy}-performance"
    total_calls = warmup + iterations + profile_iterations
    executor.open_session(
        session_id,
        maximum_context_override=total_calls * rows,
    )
    wall_ms: list[float] = []
    device_ms: list[float] = []
    retained: list[dict[str, Any]] = []
    phase_records: list[dict[str, Any]] = []
    try:
        position = 0
        for block_index in range(warmup + iterations):
            block = VerificationBlock(session_id, position, candidates)
            started = time.perf_counter_ns()
            record = executor.execute_verification_block(
                block=block,
                hidden_states=_inputs(fixtures, position=position, rows=rows),
                expert_strategy=strategy,
            )
            observed_wall_ms = (time.perf_counter_ns() - started) / 1e6
            if block_index >= warmup:
                wall_ms.append(observed_wall_ms)
                if record["device_ms"] is None:
                    raise RuntimeError("unprofiled verification block omitted device time")
                device_ms.append(float(record["device_ms"]))
                retained.append(record)
            position += rows
        for _ in range(profile_iterations):
            block = VerificationBlock(session_id, position, candidates)
            phase_records.append(
                executor.execute_verification_block(
                    block=block,
                    hidden_states=_inputs(fixtures, position=position, rows=rows),
                    expert_strategy=strategy,
                    profile_phases=True,
                )
            )
            position += rows
        state = executor.session_state_evidence(session_id)
    finally:
        executor.close_session(session_id)
    phase_names = sorted({name for record in phase_records for name in record["phase_device_ms"]})
    phase_summary = {
        name: {
            "device": _summary(
                [float(record["phase_device_ms"][name]) for record in phase_records]
            ),
            "wall": _summary([float(record["phase_wall_ms"][name]) for record in phase_records]),
        }
        for name in phase_names
    }
    wall = _summary(wall_ms)
    device = _summary(device_ms)
    last = retained[-1]
    routing_rows = [record["routing"] for record in retained]
    selection_counts: Counter[int] = Counter()
    for record in retained:
        for route in record["selected_expert_ids"]:
            selection_counts.update(int(value) for value in route)
    counts = sorted(selection_counts.values())
    retained_routing = {
        "fixture_scope": (
            "stateful execution over a cyclic replay of three real Kimi stage "
            "boundaries; exact routes, but not a representative natural-text corpus"
        ),
        "retained_blocks": len(retained),
        "retained_positions": len(retained) * rows,
        "total_assignments": len(retained) * rows * 16,
        "unique_experts_across_retained": len(selection_counts),
        "mean_unique_experts_per_block": statistics.fmean(
            float(item["unique_experts"]) for item in routing_rows
        ),
        "mean_unique_experts_per_assignment": statistics.fmean(
            float(item["unique_experts_per_assignment"]) for item in routing_rows
        ),
        "mean_assignments_per_touched_expert_within_block": statistics.fmean(
            float(item["mean_assignments_per_touched_expert"]) for item in routing_rows
        ),
        "mean_adjacent_position_overlap": statistics.fmean(
            float(item["mean_adjacent_position_overlap"]) for item in routing_rows
        ),
        "maximum_assignments_for_one_expert_within_block": max(
            int(item["maximum_assignments_for_one_expert"]) for item in routing_rows
        ),
        "assignments_per_expert_across_retained_p50": float(np.percentile(counts, 50)),
        "assignments_per_expert_across_retained_p95": float(np.percentile(counts, 95)),
        "assignments_per_expert_across_retained_p99": float(np.percentile(counts, 99)),
        "top_experts_across_retained": [
            {"expert_id": expert, "assignments": count}
            for expert, count in sorted(
                selection_counts.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
        "last_block": last["routing"],
    }
    return {
        "arm": (
            "B_device_resident_token_major"
            if strategy == "token_major"
            else "C_verification_major_expert_batching"
        ),
        "evidence_class": "MEASURED",
        "candidate_count": candidates,
        "verification_rows": rows,
        "wall": wall,
        "device": device,
        "latency_per_candidate_wall_ms": wall["p50_ms"] / candidates,
        "latency_per_accepted_token_wall_ms": wall["p50_ms"] / rows,
        "verifier_tokens_per_second": rows * 1000.0 / wall["p50_ms"],
        "routing": retained_routing,
        "dispatch": last["dispatch"],
        "transfers": last["transfers"],
        "synchronization_count": last["synchronization_count"],
        "phase_decomposition": phase_summary,
        "profile_iterations": profile_iterations,
        "state": state,
        "maximum_device_memory_growth_bytes": max(
            int(record["device_memory_growth_bytes"]) for record in retained
        ),
        "all_repeated_execution_allocations_zero": all(
            int(record["persistent_buffer_allocations_during_execute"]) == 0 for record in retained
        ),
        "all_weight_loads_zero": all(
            int(record["weight_loads_during_execute"]) == 0 for record in retained
        ),
        "all_materializations_zero": all(
            int(record["materializations_during_execute"]) == 0 for record in retained
        ),
    }


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
    fast_path_mode: str,
    fused_gate_up: bool,
    progress: Callable[[str, dict[str, Any]], None],
    gpu_sampler: _GpuSampler,
) -> dict[str, Any]:
    fixtures, _expected = _stage_fixtures(checkpoint, oracle_trace, layer=layer)
    base_request = _request(
        checkpoint,
        cuda_library,
        layer=layer,
        device=device,
        cycle_id="H016",
        maximum_context=(warmup + iterations + profile_iterations) * 17,
    )
    request = base_request.model_copy(
        update={
            "fast_path_mode": fast_path_mode,
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
    try:
        executor.runtime.set_fused_gate_up(fused_gate_up)
        load_wall_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor.prepare_for_ready()
        lifecycle_ready = executor.lifecycle_snapshot()
        progress(
            f"layer_{layer}_ready",
            {
                "load_wall_ms": load_wall_ms,
                "resident_device_bytes": executor.resident_device_bytes,
            },
        )
        gpu_sampler.start()
        correctness_rows = max(BLOCK_SIZES) + 1
        serial_output, serial_routes, serial_state = _serial_correctness(
            executor,
            fixtures,
            rows=correctness_rows,
            layer=layer,
        )
        correctness = {
            strategy: _block_correctness(
                executor,
                fixtures,
                serial_output,
                serial_routes,
                serial_state,
                candidates=max(BLOCK_SIZES),
                layer=layer,
                strategy=strategy,
            )
            for strategy in ("token_major", "expert_major")
        }
        progress(f"layer_{layer}_correctness", correctness)
        rows: list[dict[str, Any]] = []
        for candidates in BLOCK_SIZES:
            serial = _measure_serial(
                executor,
                fixtures,
                candidates=candidates,
                layer=layer,
                warmup=warmup,
                iterations=iterations,
            )
            token_major = _measure_block(
                executor,
                fixtures,
                candidates=candidates,
                layer=layer,
                strategy="token_major",
                warmup=warmup,
                iterations=iterations,
                profile_iterations=profile_iterations,
            )
            expert_major = _measure_block(
                executor,
                fixtures,
                candidates=candidates,
                layer=layer,
                strategy="expert_major",
                warmup=warmup,
                iterations=iterations,
                profile_iterations=profile_iterations,
            )
            serial_p50 = float(serial["wall"]["p50_ms"])
            token_p50 = float(token_major["wall"]["p50_ms"])
            expert_p50 = float(expert_major["wall"]["p50_ms"])
            token_major["speedup_vs_serial"] = serial_p50 / token_p50
            expert_major["speedup_vs_serial"] = serial_p50 / expert_p50
            expert_major["speedup_vs_device_resident_token_major"] = token_p50 / expert_p50
            rows.extend((serial, token_major, expert_major))
            progress(
                f"layer_{layer}_block_{candidates}",
                {
                    "serial_wall_p50_ms": serial_p50,
                    "token_major_wall_p50_ms": token_p50,
                    "expert_major_wall_p50_ms": expert_p50,
                    "expert_major_speedup_vs_serial": serial_p50 / expert_p50,
                },
            )
        lifecycle_after = executor.lifecycle_snapshot()
        return {
            "layer": layer,
            "attention_type": ("KDA" if layer in executor.config.kda_layers else "Gated_MLA"),
            "load": {
                "wall_ms": load_wall_ms,
                "resident_device_bytes": executor.resident_device_bytes,
                "tracked_device_bytes": executor.tracked_device_bytes,
                "weight_fingerprint": executor.weight_fingerprint,
                "lifecycle_ready": lifecycle_ready,
            },
            "correctness": correctness,
            "performance_rows": rows,
            "lifecycle_after": lifecycle_after,
        }
    finally:
        gpu_sampler.stop()
        executor.close()


def benchmark(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    output_path: Path,
    gpu_samples_path: Path,
    *,
    layers: tuple[int, ...] = (89, 91),
    device: int = 0,
    warmup: int = 3,
    iterations: int = 20,
    profile_iterations: int = 3,
    fast_path_mode: str = "verification-major",
    fused_gate_up: bool = True,
) -> dict[str, Any]:
    paths = (checkpoint.resolve(), cuda_library.resolve(), oracle_trace.resolve())
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "started_unix_ns": time.time_ns(),
        "configuration": {
            "layers": list(layers),
            "block_sizes": list(BLOCK_SIZES),
            "verification_rows": [value + 1 for value in BLOCK_SIZES],
            "warmup": warmup,
            "iterations": iterations,
            "profile_iterations": profile_iterations,
            "device": device,
            "exactness_relative_l2_gate": CORRECTNESS_RELATIVE_L2_GATE,
            "fast_path_mode": fast_path_mode,
            "fused_gate_up": fused_gate_up,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "process_id": os.getpid(),
            "host_ram_total_bytes": psutil.virtual_memory().total,
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "git_status": _command(["git", "status", "--short"]),
            "nvidia_smi": _command(["nvidia-smi"]),
        },
        "sources": {
            "checkpoint_config": _source(checkpoint / "config.json"),
            "checkpoint_index": _source(checkpoint / "model.safetensors.index.json"),
            "cuda_library": _source(cuda_library),
            "oracle_trace": _source(oracle_trace),
        },
        "progress": [],
        "layers": {},
    }

    def progress(phase: str, evidence: dict[str, Any]) -> None:
        receipt["progress"].append(
            {"phase": phase, "unix_ns": time.time_ns(), "evidence": evidence}
        )
        _atomic_json(output_path, receipt)
        print(f"[h016] {phase}: {evidence}", flush=True)

    _atomic_json(output_path, receipt)
    sampler = _GpuSampler(device=device)
    try:
        for layer in layers:
            receipt["layers"][str(layer)] = _benchmark_layer(
                checkpoint,
                cuda_library,
                oracle_trace,
                layer=layer,
                device=device,
                warmup=warmup,
                iterations=iterations,
                profile_iterations=profile_iterations,
                fast_path_mode=fast_path_mode,
                fused_gate_up=fused_gate_up,
                progress=progress,
                gpu_sampler=sampler,
            )
            _atomic_json(output_path, receipt)
        correctness_pass = all(
            all(bool(result["pass"]) for result in layer_result["correctness"].values())
            for layer_result in receipt["layers"].values()
        )
        receipt["gpu_sampling"] = _gpu_summary(sampler.samples)
        receipt["correctness_pass"] = correctness_pass
        receipt["status"] = "PASS" if correctness_pass else "FAIL"
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
    parser.add_argument("--layers", default="89,91")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument(
        "--fast-path-mode",
        choices=("verification-major", "verification-major-kda-window"),
        default="verification-major",
    )
    parser.add_argument(
        "--fused-gate-up",
        choices=("true", "false"),
        default="true",
    )
    arguments = parser.parse_args()
    layers = tuple(int(value) for value in arguments.layers.split(",") if value)
    benchmark(
        arguments.checkpoint,
        arguments.cuda_library,
        arguments.oracle_trace,
        arguments.output,
        arguments.gpu_samples,
        layers=layers,
        device=arguments.device,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        profile_iterations=arguments.profile_iterations,
        fast_path_mode=arguments.fast_path_mode,
        fused_gate_up=arguments.fused_gate_up == "true",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
