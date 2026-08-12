"""Real resident Kimi K3 GPU performance experiments for Experiment 014."""

from __future__ import annotations

import asyncio
import ctypes
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _process_snapshot,
    _timing,
    _warm_lifecycle_delta,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import (
    _parse_oracle_routes,
    _pointer_offset,
)
from swarm_inference.experiments.experiment_014.persistent_stages import (
    MODEL_CONTENT_FINGERPRINT,
    MODEL_REVISION,
    TOKENIZER_REVISION,
    _CaptureConnectionPool,
    _resolve_worker_identity_manifest,
    _source_assignment,
    _stage_fixtures,
)
from swarm_inference.model.kimi_k3 import _SafetensorCatalog
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

SCHEMA_VERSION = "experiment-014-k3-resident-stage-profile-v1"


class _GpuSampler:
    """Low-rate nvidia-smi sampler kept outside individual call timings."""

    def __init__(self, device: int) -> None:
        self.device = device
        self.process: subprocess.Popen[str] | None = None
        self.error: str | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "dmon",
            "-i",
            str(self.device),
            "-s",
            "u",
            "-d",
            "1",
        ]
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags,
            )
        except OSError as exc:
            self.error = str(exc)

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"status": "UNAVAILABLE", "error": self.error}
        self.process.terminate()
        try:
            stdout, stderr = self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            stdout, stderr = self.process.communicate(timeout=5)
        samples: list[tuple[float, float]] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 3:
                continue
            try:
                samples.append((float(fields[1]), float(fields[2])))
            except ValueError:
                continue
        if not samples:
            return {
                "status": "UNAVAILABLE",
                "error": stderr.strip() or "nvidia-smi dmon returned no numeric samples",
            }
        sm = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        return {
            "status": "MEASURED",
            "sampling_interval_seconds": 1,
            "sample_count": len(samples),
            "sm_utilization_percent": {
                "mean": statistics.fmean(sm),
                "maximum": max(sm),
            },
            "memory_utilization_percent": {
                "mean": statistics.fmean(memory),
                "maximum": max(memory),
            },
            "note": "device-wide one-second nvidia-smi samples; not kernel compute occupancy",
        }


def _expert_source_bytes(catalog: _SafetensorCatalog, layer: int) -> dict[str, int]:
    expert_prefix = f"language_model.model.layers.{layer}.block_sparse_moe.experts."
    shared_prefix = (
        f"language_model.model.layers.{layer}.block_sparse_moe.shared_experts."
    )
    routed_total = sum(
        catalog.tensor_info(name)[3]
        for name in catalog.weight_map
        if name.startswith(expert_prefix)
    )
    shared_total = sum(
        catalog.tensor_info(name)[3]
        for name in catalog.weight_map
        if name.startswith(shared_prefix)
    )
    if routed_total <= 0 or routed_total % 896:
        raise ValueError(f"layer {layer} routed-expert source bytes are invalid")
    return {
        "routed_expert_bytes_each": routed_total // 896,
        "routed_expert_bytes_all": routed_total,
        "shared_expert_bytes": shared_total,
    }


def _native_attribution(
    executor: PersistentKimiStageExecutor,
    *,
    routed_expert_device_ms: list[float],
    shared_expert_device_ms: list[float],
    iterations: int,
    expert_source_bytes: dict[str, int],
) -> dict[str, Any]:
    expert = executor.runtime.stats()
    dense = executor.runtime.dense_stats()
    router = executor.runtime.router_stats()
    routed_kernel_ms = sum(routed_expert_device_ms)
    shared_kernel_ms = sum(shared_expert_device_ms)
    expert_kernel_ms = routed_kernel_ms + shared_kernel_ms
    dense_kernel_ms = float(dense["kernel_ms_per_call"]) * int(dense["calls"])
    router_device_ms = int(router["calls"]) * (
        float(router["logits_ms_per_call"])
        + float(router["selection_ms_per_call"])
    )
    router_d2h_ms = int(router["calls"]) * float(router["d2h_ms_per_call"])
    attributed_device_ms = expert_kernel_ms + dense_kernel_ms + router_device_ms
    expert_bytes = iterations * (
        16 * expert_source_bytes["routed_expert_bytes_each"]
        + expert_source_bytes["shared_expert_bytes"]
    )
    return {
        "expert": expert,
        "dense": dense,
        "router": router,
        "whole_stage_device_ms": None,
        "whole_stage_device_reference_mode": "production",
        "routed_expert_kernel_ms": routed_kernel_ms,
        "shared_expert_kernel_ms": shared_kernel_ms,
        "expert_kernel_ms": expert_kernel_ms,
        "dense_kernel_ms": dense_kernel_ms,
        "router_device_ms": router_device_ms,
        "router_d2h_ms": router_d2h_ms,
        "attributed_device_ms": attributed_device_ms,
        "unattributed_device_ms": None,
        "expert_kernel_share_percent": None,
        "dense_kernel_share_percent": None,
        "router_device_share_percent": None,
        "expert_effective_source_weight_bandwidth_gbps": (
            expert_bytes / (expert_kernel_ms * 1e6) if expert_kernel_ms else 0.0
        ),
        "phase_timed_routed_expert_groups": len(routed_expert_device_ms),
        "phase_timed_shared_experts": len(shared_expert_device_ms),
        "expected_router_calls": iterations,
    }


def _run_mode(
    executor: PersistentKimiStageExecutor,
    stage_runtime: PersistentStageRuntime,
    inputs: list[np.ndarray],
    expected_boundaries: list[np.ndarray],
    expected_routes: dict[int, list[int]],
    *,
    layer: int,
    mode: str,
    warmup: int,
    iterations: int,
    device: int,
    expert_source_bytes: dict[str, int],
    cycle_id: str,
) -> dict[str, Any]:
    total = warmup + iterations
    executor.set_research_telemetry_mode(mode)
    session_id = f"{cycle_id.lower()}-layer-{layer}-{mode}"
    memory_before = executor.runtime.mem_info()
    executor.open_session(session_id)
    memory_after_open = executor.runtime.mem_info()
    sampler = _GpuSampler(device)
    sampler.start()
    wall_ms: list[float] = []
    device_ms: list[float] = []
    process_cpu_ms: list[float] = []
    host_and_sync_ms: list[float] = []
    routed_expert_device_ms: list[float] = []
    shared_expert_device_ms: list[float] = []
    selected_experts: Counter[int] = Counter()
    correctness: list[dict[str, Any]] = []
    before_warm: dict[str, Any] | None = None
    final_boundary: np.ndarray | None = None
    try:
        for position in range(total):
            if position == warmup:
                executor.runtime.reset_stats()
                executor.runtime.reset_dense_stats()
                executor.runtime.reset_router_stats()
                before_warm = _process_snapshot(stage_runtime, executor)
            fixture_index = position if position < 3 else position % 3
            boundary = torch.from_numpy(inputs[fixture_index].copy())
            cpu_started = time.process_time_ns()
            wall_started = time.perf_counter_ns()
            result = executor.execute_decode(
                session_id=session_id,
                hidden_states=boundary,
                cache_position_start=position,
            )
            call_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
            call_cpu_ms = (time.process_time_ns() - cpu_started) / 1e6
            record = executor.execution_records[-1]
            output = (
                result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
            )
            final_boundary = output
            if position < 3:
                metrics = _numerical_metrics(output, expected_boundaries[position])
                observed_routes = list(record["selected_expert_ids"])
                correctness.append(
                    {
                        "position": position,
                        "input_fingerprint": _array_fingerprint(inputs[position]),
                        "output_fingerprint": _array_fingerprint(output),
                        "expected_output_fingerprint": _array_fingerprint(
                            expected_boundaries[position]
                        ),
                        "metrics": metrics,
                        "selected_expert_ids": observed_routes,
                        "expected_expert_ids": expected_routes[position],
                        "routing_equality": observed_routes == expected_routes[position],
                    }
                )
            if position >= warmup:
                wall_ms.append(call_wall_ms)
                process_cpu_ms.append(call_cpu_ms)
                if record["device_ms"] is not None:
                    device_value = float(record["device_ms"])
                    device_ms.append(device_value)
                    host_and_sync_ms.append(max(0.0, call_wall_ms - device_value))
                if record["routed_expert_device_ms"] is not None:
                    routed_expert_device_ms.append(
                        float(record["routed_expert_device_ms"])
                    )
                if record["shared_expert_device_ms"] is not None:
                    shared_expert_device_ms.append(
                        float(record["shared_expert_device_ms"])
                    )
                selected_experts.update(int(value) for value in record["selected_expert_ids"])
        if before_warm is None or final_boundary is None:
            raise RuntimeError("resident stage benchmark did not reach its retained window")
        after_warm = _process_snapshot(stage_runtime, executor)
        native = _native_attribution(
            executor,
            routed_expert_device_ms=routed_expert_device_ms,
            shared_expert_device_ms=shared_expert_device_ms,
            iterations=iterations,
            expert_source_bytes=expert_source_bytes,
        )
        lifecycle_delta = _warm_lifecycle_delta(before_warm, after_warm)
        state_bytes = executor.kv_cache_bytes(session_id)
    finally:
        gpu_utilization = sampler.stop()
        executor.close_session(session_id)
    memory_after_close = executor.runtime.mem_info()
    maximum_error = max(
        float(row["metrics"]["relative_l2_error"]) for row in correctness
    )
    routes_exact = all(bool(row["routing_equality"]) for row in correctness)
    lifecycle_zero = all(value == 0 for value in lifecycle_delta.values())
    detailed_counter_gate = mode != "detailed" or (
        len(routed_expert_device_ms) == iterations
        and len(shared_expert_device_ms) == iterations
        and int(native["router"]["calls"]) == iterations
    )
    return {
        "mode": mode,
        "warmup_iterations": warmup,
        "retained_iterations": iterations,
        "wall": _timing(wall_ms),
        "device": _timing(device_ms) if device_ms else None,
        "process_cpu": _timing(process_cpu_ms),
        "host_and_synchronization_exposure": (
            _timing(host_and_sync_ms) if host_and_sync_ms else None
        ),
        "native_attribution": native,
        "correctness": {
            "positions": correctness,
            "maximum_relative_l2_error": maximum_error,
            "routing_equality": routes_exact,
            "pass": maximum_error <= 3e-5 and routes_exact,
        },
        "final_boundary_fingerprint": _array_fingerprint(final_boundary),
        "routing_distribution": {
            "selection_count": sum(selected_experts.values()),
            "unique_experts": len(selected_experts),
            "maximum_selections_for_one_expert": max(selected_experts.values()),
            "most_selected": [
                {"expert_id": expert, "count": count}
                for expert, count in selected_experts.most_common(10)
            ],
        },
        "memory": {
            "resident_stage_bytes": executor.resident_device_bytes,
            "session_allocation_free_delta_bytes": max(
                0, memory_before["free_bytes"] - memory_after_open["free_bytes"]
            ),
            "state_bytes": state_bytes,
            "free_bytes_before_session": memory_before["free_bytes"],
            "free_bytes_after_session_open": memory_after_open["free_bytes"],
            "free_bytes_after_session_close": memory_after_close["free_bytes"],
        },
        "transfers_per_call": {
            "h2d_bytes": 114752,
            "d2h_bytes": 28672,
            "d2d_bytes": 0,
            "wire_boundary_bytes_not_copied_wholesale_to_device": 258048,
        },
        "kernel_count": {
            "estimated_total_per_call": 91,
            "basis": "P0 retained kernel counts plus the exact canonical call graph",
            "phase_timed_routed_expert_groups": len(routed_expert_device_ms),
            "phase_timed_shared_experts": len(shared_expert_device_ms),
            "native_dense_calls": int(native["dense"]["calls"]),
            "native_router_calls": int(native["router"]["calls"]),
        },
        "gpu_utilization": gpu_utilization,
        "compute_utilization": None,
        "compute_utilization_note": "kernel occupancy requires Nsight and is not inferred from device-wide SM utilization",
        "lifecycle_delta": lifecycle_delta,
        "lifecycle_deltas_zero": lifecycle_zero,
        "detailed_counter_gate": detailed_counter_gate,
        "status": (
            "PASS"
            if maximum_error <= 3e-5
            and routes_exact
            and lifecycle_zero
            and detailed_counter_gate
            else "FAIL"
        ),
    }


async def _profile_layer(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    *,
    layer: int,
    device: int,
    warmup: int,
    iterations: int,
    mode_order: tuple[str, ...],
    cycle_id: str,
) -> dict[str, Any]:
    resolved_identity, identity_evidence = _resolve_worker_identity_manifest(
        identity_manifest,
        layer=layer,
    )
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id=identity_evidence["worker_id"],
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=4,
        configured_model_path=checkpoint,
        configured_model_identity_path=resolved_identity,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-layer-{layer}-load",
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
        fast_path_context_bucket=warmup + iterations,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    load_started = time.perf_counter_ns()
    try:
        load_response = await runtime.load_stage(request)
        load_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        source_bytes = _expert_source_bytes(_SafetensorCatalog(checkpoint), layer)
        modes = {
            mode: _run_mode(
                executor,
                runtime,
                inputs,
                expected_boundaries,
                expected_routes,
                layer=layer,
                mode=mode,
                warmup=warmup,
                iterations=iterations,
                device=device,
                expert_source_bytes=source_bytes,
                cycle_id=cycle_id,
            )
            for mode in mode_order
        }
        fingerprints = {
            str(row["final_boundary_fingerprint"]) for row in modes.values()
        }
        telemetry_overhead: dict[str, float] | None = None
        expert_share: float | None = None
        if {"minimal", "production", "detailed"}.issubset(modes):
            minimal_p50 = float(modes["minimal"]["wall"]["p50_ms"])
            production_p50 = float(modes["production"]["wall"]["p50_ms"])
            detailed_p50 = float(modes["detailed"]["wall"]["p50_ms"])
            telemetry_overhead = {
                "production_relative_to_minimal_percent": 100.0
                * (production_p50 / minimal_p50 - 1.0),
                "detailed_relative_to_minimal_percent": 100.0
                * (detailed_p50 / minimal_p50 - 1.0),
            }
            detailed_attribution = modes["detailed"]["native_attribution"]
            production_device_total = (
                float(modes["production"]["device"]["mean_ms"]) * iterations
            )
            expert_kernel_ms = float(detailed_attribution["expert_kernel_ms"])
            dense_kernel_ms = float(detailed_attribution["dense_kernel_ms"])
            router_device_ms = float(detailed_attribution["router_device_ms"])
            expert_share = 100.0 * expert_kernel_ms / production_device_total
            detailed_attribution["whole_stage_device_ms"] = production_device_total
            detailed_attribution["expert_kernel_share_percent"] = expert_share
            detailed_attribution["dense_kernel_share_percent"] = (
                100.0 * dense_kernel_ms / production_device_total
            )
            detailed_attribution["router_device_share_percent"] = (
                100.0 * router_device_ms / production_device_total
            )
            detailed_attribution["unattributed_device_ms"] = max(
                0.0,
                production_device_total
                - expert_kernel_ms
                - dense_kernel_ms
                - router_device_ms,
            )
        status = (
            "PASS"
            if load_response.accepted
            and all(row["status"] == "PASS" for row in modes.values())
            and len(fingerprints) == 1
            else "FAIL"
        )
        return {
            "layer": layer,
            "attention_type": "KDA" if layer % 4 != 3 else "Gated_MLA",
            "assignment": {
                "identity_manifest": str(resolved_identity),
                **identity_evidence,
                "source_weight_bytes": assignment.weight_bytes,
                "resident_device_bytes": executor.resident_device_bytes,
                "weight_fingerprint": executor.weight_fingerprint,
                **source_bytes,
            },
            "load": {"accepted": load_response.accepted, "elapsed_ms": load_ms},
            "mode_order": list(mode_order),
            "modes": modes,
            "cross_mode_output_equality": len(fingerprints) == 1,
            "telemetry_overhead": telemetry_overhead,
            "expert_kernel_share_percent": expert_share,
            "status": status,
        }
    finally:
        await runtime.close()


async def _benchmark_resident_stage_profile(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError("resident stage profiling requires >=3 warmups and >=20 retained calls")
    layers = [
        await _profile_layer(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            layer=1,
            device=device,
            warmup=warmup,
            iterations=iterations,
            mode_order=("minimal", "production", "detailed"),
            cycle_id="H014-027a",
        ),
        await _profile_layer(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            layer=3,
            device=device,
            warmup=warmup,
            iterations=iterations,
            mode_order=("production", "minimal", "detailed"),
            cycle_id="H014-027a",
        ),
    ]
    expert_dominance = all(
        float(row["expert_kernel_share_percent"]) > 70.0 for row in layers
    )
    production_overhead_gate = all(
        abs(
            float(
                row["telemetry_overhead"][
                    "production_relative_to_minimal_percent"
                ]
            )
        )
        <= 5.0
        for row in layers
    )
    hypothesis_supported = expert_dominance and production_overhead_gate
    if expert_dominance:
        bottleneck = "RESIDENT_MXFP4_EXPERT_WEIGHT_SERVICE"
        redesign = (
            "H014-027b tests whether expert grouping/dispatch or weight-memory service "
            "is the dominant submechanism before changing kernels."
        )
    else:
        detailed = [row["modes"]["production"] for row in layers]
        host_share = [
            100.0
            * float(row["host_and_synchronization_exposure"]["mean_ms"])
            / float(row["wall"]["mean_ms"])
            for row in detailed
        ]
        bottleneck = (
            "HOST_OR_SYNCHRONIZATION_EXPOSURE"
            if max(host_share) > 30.0
            else "MIXED_UNATTRIBUTED_STAGE_SERVICE"
        )
        redesign = (
            "H014-027b isolates the measured non-expert remainder before any expert "
            "kernel redesign."
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": "H014-027a",
        "hypothesis": (
            "Resident routed plus shared expert kernels exceed 70% of canonical batch-1 "
            "device service in both KDA and Gated-MLA MoE layers, while production "
            "telemetry changes wall p50 by at most 5% relative to minimal telemetry."
        ),
        "implementation": (
            "Measurement-only native telemetry selection and opt-in cumulative router "
            "statistics; no arithmetic, layout, scheduling, weight, or state change."
        ),
        "backend": {
            "identity": "nvidia_cuda_persistent_kimi_stage",
            "device": _device_identity(device),
            "cuda_library": str(cuda_library.resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "target_binary": "sm_86+compute_86 PTX",
            "cpu_mathematical_fallbacks": 0,
        },
        "benchmark": {
            "batch": 1,
            "warmup_iterations_per_mode": warmup,
            "retained_iterations_per_mode": iterations,
            "layers": layers,
        },
        "result": {
            "expert_dominance_prediction_supported": expert_dominance,
            "production_telemetry_overhead_gate_supported": production_overhead_gate,
            "hypothesis_supported": hypothesis_supported,
        },
        "inspection": {
            "actual_bottleneck": bottleneck,
            "instrumentation_materially_determined_result": False,
            "detailed_telemetry_retained_for_attribution_only": True,
            "production_or_minimal_modes_required_for_service_rate": True,
        },
        "decision": "RETAIN_MEASUREMENT_CONTROLS",
        "redesign": redesign,
        "status": "PASS" if all(row["status"] == "PASS" for row in layers) else "FAIL",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["output_path"] = str(output_path.resolve())
    receipt["output_sha256"] = _sha256_file(output_path)
    return receipt


def benchmark_resident_stage_profile(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run H014-027a against two real canonical persistent Kimi stages."""
    return asyncio.run(
        _benchmark_resident_stage_profile(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    )


def _depth_profile_row(row: dict[str, Any], *, depth: str) -> dict[str, Any]:
    production = row["modes"]["production"]
    return {
        "layer": int(row["layer"]),
        "depth": depth,
        "attention_type": str(row["attention_type"]),
        "assignment": row["assignment"],
        "load": row["load"],
        "production": production,
        "device_p50_ms": float(production["device"]["p50_ms"]),
        "device_p95_ms": float(production["device"]["p95_ms"]),
        "device_p99_ms": float(production["device"]["p99_ms"]),
        "wall_p50_ms": float(production["wall"]["p50_ms"]),
        "correctness_pass": bool(production["correctness"]["pass"]),
        "lifecycle_deltas_zero": bool(production["lifecycle_deltas_zero"]),
        "status": str(row["status"]),
    }


def _depth_spread(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p50 = [float(row["device_p50_ms"]) for row in rows]
    p95 = [float(row["device_p95_ms"]) for row in rows]
    p50_spread = 100.0 * (max(p50) / min(p50) - 1.0)
    p95_spread = 100.0 * (max(p95) / min(p95) - 1.0)
    return {
        "layers": [int(row["layer"]) for row in rows],
        "device_p50_ms": p50,
        "device_p95_ms": p95,
        "device_p50_max_to_min_spread_percent": p50_spread,
        "device_p95_max_to_min_spread_percent": p95_spread,
        "p50_gate_percent": 10.0,
        "p95_gate_percent": 15.0,
        "hypothesis_supported": p50_spread <= 10.0 and p95_spread <= 15.0,
    }


async def _benchmark_depth_profile(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    early_profile_path: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError("depth profiling requires >=3 warmups and >=20 retained calls")
    early_profile = json.loads(early_profile_path.read_text(encoding="utf-8"))
    if (
        early_profile.get("cycle_id") != "H014-027a"
        or early_profile.get("status") != "PASS"
    ):
        raise ValueError("H014-027m requires the retained H014-027a profile")
    early_by_layer = {
        int(row["layer"]): row for row in early_profile["benchmark"]["layers"]
    }
    if not {1, 3}.issubset(early_by_layer):
        raise ValueError("H014-027a profile is missing early KDA/MLA evidence")

    rows = [
        _depth_profile_row(early_by_layer[1], depth="early"),
        _depth_profile_row(early_by_layer[3], depth="early"),
    ]
    for layer, depth in ((45, "middle"), (47, "middle"), (89, "late"), (91, "late")):
        profiled = await _profile_layer(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            layer=layer,
            device=device,
            warmup=warmup,
            iterations=iterations,
            mode_order=("production",),
            cycle_id="H014-027m",
        )
        rows.append(_depth_profile_row(profiled, depth=depth))

    kda_rows = sorted(
        (row for row in rows if row["attention_type"] == "KDA"),
        key=lambda row: int(row["layer"]),
    )
    mla_rows = sorted(
        (row for row in rows if row["attention_type"] == "Gated_MLA"),
        key=lambda row: int(row["layer"]),
    )
    spreads = {
        "KDA": _depth_spread(kda_rows),
        "Gated_MLA": _depth_spread(mla_rows),
    }
    hypothesis_supported = all(
        bool(row["hypothesis_supported"]) for row in spreads.values()
    )
    execution_pass = all(
        row["status"] == "PASS"
        and bool(row["correctness_pass"])
        and bool(row["lifecycle_deltas_zero"])
        for row in rows
    )
    slowest = max(rows, key=lambda row: float(row["device_p50_ms"]))
    receipt = {
        "schema_version": "experiment-014-k3-depth-profile-v1",
        "cycle_id": "H014-027m",
        "hypothesis": (
            "Within KDA and Gated MLA separately, early/middle/late warm batch-1 "
            "production device p50 spans at most 10% and p95 at most 15%."
        ),
        "implementation": (
            "Measurement-only production-telemetry profiling of unresolved real middle "
            "and late stages; retained H014-027a supplies immutable early-stage evidence."
        ),
        "backend": {
            "identity": "nvidia_cuda_persistent_kimi_stage",
            "device": _device_identity(device),
            "cuda_library": str(cuda_library.resolve()),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "target_binary": "sm_86+compute_86 PTX",
            "cpu_mathematical_fallbacks": 0,
        },
        "source": {
            "early_profile": str(early_profile_path.resolve()),
            "early_profile_sha256": _sha256_file(early_profile_path),
            "early_layers_reused": [1, 3],
            "fresh_layers": [45, 47, 89, 91],
        },
        "benchmark": {
            "batch": 1,
            "warmup_iterations": warmup,
            "retained_iterations": iterations,
            "layers": sorted(rows, key=lambda row: int(row["layer"])),
            "spreads": spreads,
            "slowest_device_p50": {
                "layer": int(slowest["layer"]),
                "attention_type": str(slowest["attention_type"]),
                "milliseconds": float(slowest["device_p50_ms"]),
            },
        },
        "result": {
            "execution_pass": execution_pass,
            "hypothesis_supported": hypothesis_supported,
        },
        "inspection": {
            "actual_bottleneck": (
                "ARCHITECTURE_CLASS_NOT_LAYER_DEPTH"
                if hypothesis_supported
                else "LAYER_DEPTH_OR_UNMODELED_STAGE_VARIANCE"
            ),
            "one_time_load_and_prepare_excluded_from_service_time": True,
            "production_telemetry_only_for_fresh_layers": True,
        },
        "decision": (
            "RETAIN_ARCHITECTURE_CLASS_REPRESENTATIVES"
            if hypothesis_supported
            else "MODIFY_LAYER_SERVICE_MODEL"
        ),
        "redesign": (
            "Use measured slowest and median complete layers in batch/concurrency work."
            if hypothesis_supported
            else "Inspect the outlying depth and add it explicitly to the service model."
        ),
        "status": "PASS" if execution_pass else "FAIL",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["output_path"] = str(output_path.resolve())
    receipt["output_sha256"] = _sha256_file(output_path)
    return receipt


def benchmark_depth_profile(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    early_profile_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run H014-027m over early, middle, and late real Kimi stages."""
    return asyncio.run(
        _benchmark_depth_profile(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            early_profile_path.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    )


def _run_concurrency_configuration(
    executor: PersistentKimiStageExecutor,
    stage_runtime: PersistentStageRuntime,
    inputs: list[np.ndarray],
    expected_boundaries: list[np.ndarray],
    expected_routes: dict[int, list[int]],
    *,
    streams: int,
    label: str,
    warmup: int,
    iterations: int,
    device: int,
    cycle_id: str,
) -> dict[str, Any]:
    session_ids = [
        f"{cycle_id.lower()}-{label}-stream-{index:02d}"
        for index in range(streams)
    ]
    records_before = len(executor.execution_records)
    before_open_snapshot = _process_snapshot(stage_runtime, executor)
    memory_before = executor.runtime.mem_info()
    for session_id in session_ids:
        executor.open_session(session_id)
    after_open_snapshot = _process_snapshot(stage_runtime, executor)
    memory_after_open = executor.runtime.mem_info()
    correctness_rows: list[dict[str, Any]] = []
    fingerprints_by_position: dict[int, list[str]] = {0: [], 1: [], 2: []}
    last_completion_ns: dict[str, int] = {}
    sampler = _GpuSampler(device)
    gpu_utilization: dict[str, Any] = {"status": "NOT_STARTED"}
    try:
        for position in range(warmup):
            fixture_index = position if position < 3 else position % 3
            for session_id in session_ids:
                result = executor.execute_decode(
                    session_id=session_id,
                    hidden_states=torch.from_numpy(inputs[fixture_index].copy()),
                    cache_position_start=position,
                )
                last_completion_ns[session_id] = time.perf_counter_ns()
                if position < 3:
                    record = executor.execution_records[-1]
                    output = (
                        result.stage_boundary_hidden_states.detach()
                        .cpu()
                        .numpy()
                        .copy()
                    )
                    metrics = _numerical_metrics(
                        output, expected_boundaries[position]
                    )
                    observed_routes = list(record["selected_expert_ids"])
                    fingerprint = _array_fingerprint(output)
                    fingerprints_by_position[position].append(fingerprint)
                    correctness_rows.append(
                        {
                            "stream_id": session_id,
                            "position": position,
                            "input_fingerprint": _array_fingerprint(
                                inputs[position]
                            ),
                            "output_fingerprint": fingerprint,
                            "expected_output_fingerprint": _array_fingerprint(
                                expected_boundaries[position]
                            ),
                            "relative_l2_error": float(
                                metrics["relative_l2_error"]
                            ),
                            "maximum_absolute_error": float(
                                metrics["maximum_absolute_error"]
                            ),
                            "cosine_similarity": float(
                                metrics["cosine_similarity"]
                            ),
                            "routing_equality": (
                                observed_routes == expected_routes[position]
                            ),
                            "selected_expert_ids": observed_routes,
                        }
                    )

        before = _process_snapshot(stage_runtime, executor)
        call_wall_ms: list[float] = []
        device_ms: list[float] = []
        inter_completion_ms: list[float] = []
        queue_exposure_ms: list[float] = []
        selected_experts: Counter[int] = Counter()
        selected_groups: Counter[tuple[int, ...]] = Counter()
        sampler.start()
        after_sampler_start = _process_snapshot(stage_runtime, executor)
        block_started = time.perf_counter_ns()
        block_finished = block_started
        try:
            for position in range(warmup, warmup + iterations):
                fixture_index = position % 3
                for session_id in session_ids:
                    call_started = time.perf_counter_ns()
                    executor.execute_decode(
                        session_id=session_id,
                        hidden_states=torch.from_numpy(inputs[fixture_index].copy()),
                        cache_position_start=position,
                    )
                    completed = time.perf_counter_ns()
                    wall_value = (completed - call_started) / 1e6
                    call_wall_ms.append(wall_value)
                    interval = (completed - last_completion_ns[session_id]) / 1e6
                    inter_completion_ms.append(interval)
                    queue_exposure_ms.append(max(0.0, interval - wall_value))
                    last_completion_ns[session_id] = completed
                    record = executor.execution_records[-1]
                    if record["device_ms"] is not None:
                        device_ms.append(float(record["device_ms"]))
                    group = tuple(int(value) for value in record["selected_expert_ids"])
                    selected_groups[group] += 1
                    selected_experts.update(group)
            block_finished = time.perf_counter_ns()
            after_compute = _process_snapshot(stage_runtime, executor)
        finally:
            block_wall_ms = (block_finished - block_started) / 1e6
            gpu_utilization = sampler.stop()
        after = _process_snapshot(stage_runtime, executor)
        lifecycle_delta = _warm_lifecycle_delta(before, after)
        state_bytes = sum(executor.kv_cache_bytes(value) for value in session_ids)
    finally:
        for session_id in session_ids:
            executor.close_session(session_id)
    after_close_snapshot = _process_snapshot(stage_runtime, executor)
    memory_after_close = executor.runtime.mem_info()
    del executor.execution_records[records_before:]
    after_record_cleanup_snapshot = _process_snapshot(stage_runtime, executor)

    maximum_error = max(
        float(row["relative_l2_error"]) for row in correctness_rows
    )
    routes_exact = all(bool(row["routing_equality"]) for row in correctness_rows)
    cross_stream_equal = all(
        len(set(values)) == 1 for values in fingerprints_by_position.values()
    )
    lifecycle_zero = all(value == 0 for value in lifecycle_delta.values())
    memory_recovered = memory_after_close["free_bytes"] >= memory_before["free_bytes"]
    total_calls = streams * iterations
    aggregate_rate = total_calls / (block_wall_ms / 1000.0)
    created_at_sampler_start = sorted(
        set(after_sampler_start["os_thread_ids"]) - set(before["os_thread_ids"])
    )
    created_during_compute = sorted(
        set(after_compute["os_thread_ids"])
        - set(after_sampler_start["os_thread_ids"])
    )
    created_at_sampler_stop = sorted(
        set(after["os_thread_ids"]) - set(after_compute["os_thread_ids"])
    )
    compute_threads_persisted = sorted(
        set(created_during_compute).intersection(after["os_thread_ids"])
    )
    request_phase_snapshots = (
        before_open_snapshot,
        after_open_snapshot,
        before,
        after,
        after_close_snapshot,
        after_record_cleanup_snapshot,
    )
    request_phase_names = (
        "before_open",
        "after_open",
        "after_warmup",
        "after_retained",
        "after_close",
        "after_record_cleanup",
    )
    request_thread_counts = {
        name: len(snapshot["os_thread_ids"])
        for name, snapshot in zip(
            request_phase_names, request_phase_snapshots, strict=True
        )
    }
    request_thread_creations = {
        f"{request_phase_names[index]}_to_{request_phase_names[index + 1]}": sorted(
            set(request_phase_snapshots[index + 1]["os_thread_ids"])
            - set(request_phase_snapshots[index]["os_thread_ids"])
        )
        for index in range(len(request_phase_snapshots) - 1)
    }
    close_created_threads = request_thread_creations[
        "after_retained_to_after_close"
    ]
    execution_pass = (
        maximum_error <= 3e-5
        and routes_exact
        and cross_stream_equal
        and lifecycle_zero
        and memory_recovered
        and len(device_ms) == total_calls
    )
    return {
        "label": label,
        "streams": streams,
        "warmup_calls_per_stream": warmup,
        "retained_calls_per_stream": iterations,
        "retained_calls_total": total_calls,
        "block_wall_ms": block_wall_ms,
        "aggregate_stage_operations_per_second": aggregate_rate,
        "per_stream_stage_operations_per_second": aggregate_rate / streams,
        "call_wall": _timing(call_wall_ms),
        "device": _timing(device_ms),
        "per_stream_inter_completion": _timing(inter_completion_ms),
        "queue_exposure": _timing(queue_exposure_ms),
        "gpu_utilization": gpu_utilization,
        "lifecycle_phase_trace": {
            "parent_os_thread_counts": {
                "before_sampler_start": len(before["os_thread_ids"]),
                "after_sampler_start": len(after_sampler_start["os_thread_ids"]),
                "after_compute_before_sampler_stop": len(
                    after_compute["os_thread_ids"]
                ),
                "after_sampler_stop": len(after["os_thread_ids"]),
            },
            "os_thread_ids_created_at_sampler_start": created_at_sampler_start,
            "os_thread_ids_created_during_compute": created_during_compute,
            "os_thread_ids_created_at_sampler_stop": created_at_sampler_stop,
            "compute_created_thread_ids_persisting_after_sampler_stop": (
                compute_threads_persisted
            ),
            "child_process_ids_created_at_sampler_start": sorted(
                set(after_sampler_start["child_process_ids"])
                - set(before["child_process_ids"])
            ),
            "child_process_ids_remaining_after_sampler_stop": sorted(
                set(after["child_process_ids"]) - set(before["child_process_ids"])
            ),
        },
        "request_lifecycle_phase_trace": {
            "parent_os_thread_counts": request_thread_counts,
            "os_thread_ids_created_by_phase": request_thread_creations,
            "close_created_thread_ids_persisting_after_record_cleanup": sorted(
                set(close_created_threads).intersection(
                    after_record_cleanup_snapshot["os_thread_ids"]
                )
            ),
        },
        "memory": {
            "resident_stage_bytes": executor.resident_device_bytes,
            "session_allocation_bytes": max(
                0, memory_before["free_bytes"] - memory_after_open["free_bytes"]
            ),
            "request_state_bytes": state_bytes,
            "free_bytes_before_open": memory_before["free_bytes"],
            "free_bytes_after_open": memory_after_open["free_bytes"],
            "free_bytes_after_close": memory_after_close["free_bytes"],
            "temporary_memory_recovered": memory_recovered,
        },
        "wire_and_copy_accounting": {
            "h2d_bytes_per_call": 114752,
            "d2h_bytes_per_call": 28672,
            "d2d_bytes_per_call": 0,
            "stage_boundary_bytes": 258048,
            "retained_h2d_bytes": total_calls * 114752,
            "retained_d2h_bytes": total_calls * 28672,
        },
        "routing": {
            "selection_count": sum(selected_experts.values()),
            "unique_experts": len(selected_experts),
            "unique_top16_groups": len(selected_groups),
            "most_reused_top16_group_count": max(selected_groups.values()),
            "most_selected_experts": [
                {"expert_id": expert, "count": count}
                for expert, count in selected_experts.most_common(10)
            ],
        },
        "correctness": {
            "positions_per_stream": 3,
            "input_fingerprints": [
                _array_fingerprint(inputs[position]) for position in range(3)
            ],
            "expected_output_fingerprints": [
                _array_fingerprint(expected_boundaries[position])
                for position in range(3)
            ],
            "maximum_relative_l2_error": maximum_error,
            "routing_equality": routes_exact,
            "cross_stream_output_equality": cross_stream_equal,
            "pass": maximum_error <= 3e-5 and routes_exact and cross_stream_equal,
        },
        "lifecycle_delta": lifecycle_delta,
        "lifecycle_deltas_zero": lifecycle_zero,
        "status": "PASS" if execution_pass else "FAIL",
    }


async def _benchmark_concurrency_baseline(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    depth_profile_path: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
    cycle_id: str,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError(
            "concurrency baseline requires >=3 warmups and >=20 retained calls"
        )
    if cycle_id not in {
        "H014-027n",
        "H014-027o",
        "H014-027p",
        "H014-027q",
        "H014-027s",
    }:
        raise ValueError("unsupported concurrency research cycle")
    depth_profile = json.loads(depth_profile_path.read_text(encoding="utf-8"))
    slowest = depth_profile.get("benchmark", {}).get("slowest_device_p50", {})
    if (
        depth_profile.get("cycle_id") != "H014-027m"
        or depth_profile.get("status") != "PASS"
        or int(slowest.get("layer", -1)) != 89
    ):
        raise ValueError("H014-027n requires retained H014-027m layer-89 evidence")
    layer = 89
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id=f"{cycle_id.lower()}-concurrency-worker-089",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=16,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-layer-89-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id=f"{cycle_id.lower()}-layer-89",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=warmup + iterations,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    load_started = time.perf_counter_ns()
    try:
        load_response = await runtime.load_stage(request)
        load_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        executor.set_research_telemetry_mode("production")
        configuration_spec = [
            (1, "streams-1-pre"),
            (2, "streams-2"),
            (4, "streams-4"),
            (8, "streams-8"),
            (16, "streams-16"),
        ]
        if cycle_id in {"H014-027o", "H014-027p", "H014-027q", "H014-027s"}:
            configuration_spec.append((8, "streams-8-repeat"))
        configuration_spec.append((1, "streams-1-post"))
        configurations = [
            _run_concurrency_configuration(
                executor,
                runtime,
                inputs,
                expected_boundaries,
                expected_routes,
                streams=streams,
                label=label,
                warmup=warmup,
                iterations=iterations,
                device=device,
                cycle_id=cycle_id,
            )
            for streams, label in configuration_spec
        ]
        by_label = {str(row["label"]): row for row in configurations}
        pre_rate = float(
            by_label["streams-1-pre"]["aggregate_stage_operations_per_second"]
        )
        post_rate = float(
            by_label["streams-1-post"]["aggregate_stage_operations_per_second"]
        )
        baseline_rate = statistics.fmean((pre_rate, post_rate))
        stream16 = by_label["streams-16"]
        stream16_aggregate = float(
            stream16["aggregate_stage_operations_per_second"]
        )
        stream16_per_stream = float(
            stream16["per_stream_stage_operations_per_second"]
        )
        aggregate_change = 100.0 * (stream16_aggregate / baseline_rate - 1.0)
        per_stream_retention = 100.0 * (stream16_per_stream / baseline_rate)
        drift = 100.0 * abs(post_rate / pre_rate - 1.0)
        serving_lifecycle_pass = bool(load_response.accepted) and all(
            row["status"] == "PASS" for row in configurations
        )
        baseline_hypothesis_supported = (
            aggregate_change <= 10.0
            and stream16_per_stream <= baseline_rate / 8.0
            and drift <= 5.0
        )
        semantic_execution_pass = bool(load_response.accepted) and all(
            bool(row["correctness"]["pass"])
            and bool(row["memory"]["temporary_memory_recovered"])
            and all(
                value == 0
                for key, value in row["lifecycle_delta"].items()
                if key != "thread_creation"
            )
            for row in configurations
        )
        trace_hypothesis_supported: bool | None = None
        request_hypothesis_supported: bool | None = None
        prepared_hypothesis_supported: bool | None = None
        reset_activity_hypothesis_supported: bool | None = None
        if cycle_id == "H014-027o":
            first_trace = by_label["streams-8"]["lifecycle_phase_trace"]
            repeat = by_label["streams-8-repeat"]
            compute_created = list(
                first_trace["os_thread_ids_created_during_compute"]
            )
            trace_hypothesis_supported = (
                len(compute_created) == 2
                and not first_trace["os_thread_ids_created_at_sampler_start"]
                and not first_trace["os_thread_ids_created_at_sampler_stop"]
                and sorted(compute_created)
                == sorted(
                    first_trace[
                        "compute_created_thread_ids_persisting_after_sampler_stop"
                    ]
                )
                and int(repeat["lifecycle_delta"]["thread_creation"]) == 0
            )
        if cycle_id == "H014-027p":
            request_trace = by_label["streams-8"][
                "request_lifecycle_phase_trace"
            ]
            request_creations = request_trace["os_thread_ids_created_by_phase"]
            close_created = list(
                request_creations["after_retained_to_after_close"]
            )
            request_hypothesis_supported = (
                len(close_created) == 2
                and not request_creations["before_open_to_after_open"]
                and not request_creations["after_open_to_after_warmup"]
                and not request_creations["after_warmup_to_after_retained"]
                and not request_creations["after_close_to_after_record_cleanup"]
                and sorted(close_created)
                == sorted(
                    request_trace[
                        "close_created_thread_ids_persisting_after_record_cleanup"
                    ]
                )
                and int(
                    by_label["streams-8-repeat"]["lifecycle_delta"][
                        "thread_creation"
                    ]
                )
                == 0
            )
        prepare_evidence = executor.lifecycle_snapshot()["prepare_warmup"]
        if not isinstance(prepare_evidence, dict):
            raise RuntimeError("concurrency benchmark omitted PREPARE evidence")
        if cycle_id == "H014-027q":
            reset_evidence = prepare_evidence.get("request_reset_fixture")
            no_request_phase_threads = all(
                not created
                for row in configurations
                for created in row["request_lifecycle_phase_trace"][
                    "os_thread_ids_created_by_phase"
                ].values()
            )
            prepared_hypothesis_supported = (
                isinstance(reset_evidence, dict)
                and int(reset_evidence["session_count"]) == 8
                and int(reset_evidence["model_operations"]) == 0
                and int(reset_evidence["active_sessions_after"]) == 0
                and bool(reset_evidence["temporary_memory_recovered"])
                and float(prepare_evidence["total_measured_wall_ms"]) <= 250.0
                and int(prepare_evidence["maximum_temporary_device_bytes"])
                <= 64 * 1024**2
                and no_request_phase_threads
                and serving_lifecycle_pass
            )
        if cycle_id == "H014-027s":
            first_reset_trace = by_label["streams-8"][
                "request_lifecycle_phase_trace"
            ]
            first_reset_creations = first_reset_trace[
                "os_thread_ids_created_by_phase"
            ]
            close_threads = list(
                first_reset_creations["after_retained_to_after_close"]
            )
            no_other_request_threads = all(
                not created
                for row in configurations
                for phase, created in row["request_lifecycle_phase_trace"][
                    "os_thread_ids_created_by_phase"
                ].items()
                if not (
                    row["label"] == "streams-8"
                    and phase == "after_retained_to_after_close"
                )
            )
            reset_activity_hypothesis_supported = (
                len(close_threads) == 2
                and sorted(close_threads)
                == sorted(
                    first_reset_trace[
                        "close_created_thread_ids_persisting_after_record_cleanup"
                    ]
                )
                and no_other_request_threads
                and serving_lifecycle_pass
                and int(prepare_evidence["stage_fixture"]["iterations"]) == 7
                and float(prepare_evidence["total_measured_wall_ms"]) <= 250.0
                and int(prepare_evidence["maximum_temporary_device_bytes"])
                <= 8 * 1024**2
            )
        if cycle_id == "H014-027n":
            hypothesis_supported = baseline_hypothesis_supported
        elif cycle_id == "H014-027o":
            hypothesis_supported = bool(trace_hypothesis_supported)
        elif cycle_id == "H014-027p":
            hypothesis_supported = bool(request_hypothesis_supported)
        elif cycle_id == "H014-027q":
            hypothesis_supported = bool(prepared_hypothesis_supported)
        else:
            hypothesis_supported = bool(reset_activity_hypothesis_supported)
        if cycle_id == "H014-027n":
            artifact_pass = serving_lifecycle_pass
        elif cycle_id in {"H014-027q", "H014-027s"}:
            artifact_pass = semantic_execution_pass and hypothesis_supported
        else:
            artifact_pass = semantic_execution_pass
        if cycle_id == "H014-027o":
            cycle_hypothesis = (
                "The two H014-027n OS threads are created during retained CUDA "
                "execution, persist after sampler stop, and do not recur in a later "
                "8-stream repeat."
            )
            cycle_implementation = (
                "Evidence-only parent process/thread phase snapshots around the "
                "unchanged sampler and retained execution, plus an 8-stream repeat."
            )
            actual_bottleneck = (
                "ONE_TIME_LAZY_EXECUTION_THREADS_BEFORE_BATCHING"
                if hypothesis_supported
                else "THREAD_CREATION_SOURCE_NOT_AS_PREDICTED"
            )
            decision = (
                "MOVE_LAZY_THREAD_CREATION_TO_PREPARE"
                if hypothesis_supported
                else "INSPECT_THREAD_PHASE_TRACE"
            )
            redesign = (
                "H014-027p moves the measured one-time thread trigger into PREPARE."
                if hypothesis_supported
                else "Form the next lifecycle hypothesis from the phase that changed."
            )
        elif cycle_id == "H014-027p":
            cycle_hypothesis = (
                "The two one-time parent threads are created while closing the first "
                "8-session request set, persist through cleanup, and are not created "
                "during open, warmup, retained execution, or sampling."
            )
            cycle_implementation = (
                "Evidence-only request lifecycle snapshots before/after session open, "
                "warmup, retained execution, close, and research-record cleanup; the "
                "post-compute snapshot is outside the service timer."
            )
            actual_bottleneck = (
                "ONE_TIME_THREADS_DURING_REQUEST_CLOSE"
                if hypothesis_supported
                else "REQUEST_LIFECYCLE_THREAD_PHASE_NOT_AS_PREDICTED"
            )
            decision = (
                "MOVE_REQUEST_CLOSE_THREAD_TRIGGER_TO_PREPARE"
                if hypothesis_supported
                else "INSPECT_REQUEST_LIFECYCLE_TRACE"
            )
            redesign = (
                "H014-027q primes the measured request-lifecycle trigger before READY."
                if hypothesis_supported
                else "Use the exact changing request phase for the next hypothesis."
            )
        elif cycle_id == "H014-027q":
            cycle_hypothesis = (
                "An eight-session open/close PREPARE fixture absorbs the one-time reset "
                "threads so every later request phase creates zero parent threads within "
                "250 ms PREPARE wall and 64 MiB transient VRAM."
            )
            cycle_implementation = (
                "Production PREPARE adds one isolated eight-session open/close fixture "
                "after stage warmup; it performs no model operation and frees all state."
            )
            actual_bottleneck = (
                "REQUEST_RESET_THREAD_TRANSITION_RESOLVED_BEFORE_READY"
                if hypothesis_supported
                else "PREPARE_REQUEST_RESET_PRIMING_INSUFFICIENT"
            )
            decision = (
                "RETAIN_PREPARED_REQUEST_RESET_AND_IMPLEMENT_BATCH2"
                if hypothesis_supported
                else "INSPECT_PREPARE_RESET_TRACE"
            )
            redesign = (
                "H014-027r tests the minimum real batch-2 canonical CUDA stage path."
                if hypothesis_supported
                else "Redesign reset priming from the first failing phase."
            )
        elif cycle_id == "H014-027s":
            cycle_hypothesis = (
                "The retained seven-call PREPARE preserves zero per-generation lifecycle "
                "churn; exactly two persistent threads may arise only on the first "
                "sustained eight-session RESET and never on its repeat."
            )
            cycle_implementation = (
                "Exact H014-027l PREPARE revert with unchanged request-phase tracing; "
                "request-specific reset activity remains explicit and separate."
            )
            actual_bottleneck = (
                "SYNCHRONOUS_SERIAL_SCHEDULER_WITH_BOUNDED_RESET_ACTIVITY"
                if hypothesis_supported
                else "UNBOUNDED_OR_MISLOCATED_REQUEST_LIFECYCLE_ACTIVITY"
            )
            decision = (
                "RETAIN_H014_027L_PREPARE_AND_IMPLEMENT_BATCH2"
                if hypothesis_supported
                else "INSPECT_REVERTED_REQUEST_LIFECYCLE"
            )
            redesign = (
                "H014-027t tests the minimum real batch-2 canonical CUDA stage path."
                if hypothesis_supported
                else "Resolve the first phase violating the reset-activity contract."
            )
        else:
            cycle_hypothesis = (
                "Synchronous round-robin 1-to-16 request concurrency raises aggregate "
                "layer-89 service by at most 10%, cuts 16-stream per-stream cadence to "
                "at most one eighth, and preserves <=5% pre/post control drift."
            )
            cycle_implementation = (
                "Measurement-only registered READY worker with independent request "
                "state; no batching, overlap, kernel, weight, or scheduler change."
            )
            actual_bottleneck = (
                "SYNCHRONOUS_SERIAL_STAGE_SCHEDULER"
                if baseline_hypothesis_supported
                else "UNEXPECTED_CONCURRENCY_OR_MEASUREMENT_DRIFT"
            )
            decision = (
                "IMPLEMENT_MINIMUM_REAL_BATCH_PATH"
                if baseline_hypothesis_supported and serving_lifecycle_pass
                else "INSPECT_CONCURRENCY_TRACE"
            )
            redesign = (
                "H014-027o locates the lazy thread transition before batching."
                if baseline_hypothesis_supported
                else "Resolve the measured concurrency effect before batching."
            )
        receipt = {
            "schema_version": "experiment-014-k3-concurrency-baseline-v1",
            "cycle_id": cycle_id,
            "hypothesis": cycle_hypothesis,
            "implementation": cycle_implementation,
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "device": _device_identity(device),
                "cuda_library": str(cuda_library.resolve()),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "target_binary": "sm_86+compute_86 PTX",
                "cpu_mathematical_fallbacks": 0,
            },
            "source": {
                "depth_profile": str(depth_profile_path.resolve()),
                "depth_profile_sha256": _sha256_file(depth_profile_path),
                "selected_slowest_layer": layer,
            },
            "load": {
                "accepted": load_response.accepted,
                "elapsed_ms": load_ms,
                "prepare_warmup": prepare_evidence,
            },
            "benchmark": {
                "batch": 1,
                "scheduler": "synchronous_round_robin",
                "metric_scope": (
                    "single-stage operations per second; not full-model output tokens/s"
                ),
                "configurations": configurations,
                "single_stream_baseline_operations_per_second": baseline_rate,
                "single_stream_pre_post_drift_percent": drift,
                "stream16_aggregate_change_percent": aggregate_change,
                "stream16_per_stream_retention_percent": per_stream_retention,
            },
            "result": {
                "semantic_execution_pass": semantic_execution_pass,
                "serving_lifecycle_pass": serving_lifecycle_pass,
                "baseline_scheduler_prediction_supported": (
                    baseline_hypothesis_supported
                ),
                "one_time_compute_thread_prediction_supported": (
                    trace_hypothesis_supported
                ),
                "request_close_thread_prediction_supported": (
                    request_hypothesis_supported
                ),
                "prepared_request_lifecycle_prediction_supported": (
                    prepared_hypothesis_supported
                ),
                "bounded_reset_activity_prediction_supported": (
                    reset_activity_hypothesis_supported
                ),
                "hypothesis_supported": hypothesis_supported,
            },
            "inspection": {
                "actual_bottleneck": actual_bottleneck,
                "aggregate_stage_rate_is_not_cluster_throughput": True,
                "full_model_throughput_projection": None,
            },
            "decision": decision,
            "redesign": redesign,
            "status": "PASS" if artifact_pass else "FAIL",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["output_path"] = str(output_path.resolve())
        receipt["output_sha256"] = _sha256_file(output_path)
        return receipt
    finally:
        await runtime.close()


def benchmark_concurrency_baseline(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    depth_profile_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
    cycle_id: str = "H014-027n",
) -> dict[str, Any]:
    """Run H014-027n synchronous multi-stream baseline on the slowest layer."""
    return asyncio.run(
        _benchmark_concurrency_baseline(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            depth_profile_path.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            warmup=warmup,
            iterations=iterations,
            cycle_id=cycle_id,
        )
    )


def _validate_reset_trace(receipt: dict[str, Any]) -> dict[str, Any]:
    configurations = receipt["benchmark"]["configurations"]
    by_label = {str(row["label"]): row for row in configurations}
    first = by_label["streams-8"]
    repeat = by_label["streams-8-repeat"]
    first_trace = first["request_lifecycle_phase_trace"]
    first_creations = first_trace["os_thread_ids_created_by_phase"]
    close_ids = set(first_creations["after_retained_to_after_close"])
    cleanup_ids = set(first_creations["after_close_to_after_record_cleanup"])
    reset_union = sorted(close_ids.union(cleanup_ids))
    no_execute_side_threads = not any(
        first_creations[phase]
        for phase in (
            "before_open_to_after_open",
            "after_open_to_after_warmup",
            "after_warmup_to_after_retained",
        )
    )
    no_threads_outside_first_reset = all(
        not created
        for row in configurations
        for phase, created in row["request_lifecycle_phase_trace"][
            "os_thread_ids_created_by_phase"
        ].items()
        if not (
            row["label"] == "streams-8"
            and phase
            in {
                "after_retained_to_after_close",
                "after_close_to_after_record_cleanup",
            }
        )
    )
    counts = first_trace["parent_os_thread_counts"]
    threads_persisted = (
        int(counts["after_record_cleanup"])
        == int(counts["before_open"]) + len(reset_union)
    )
    repeat_creations = repeat["request_lifecycle_phase_trace"][
        "os_thread_ids_created_by_phase"
    ]
    repeat_zero = not any(repeat_creations.values())
    retained_lifecycle_zero = all(
        bool(row["lifecycle_deltas_zero"]) for row in configurations
    )
    semantic_pass = all(
        bool(row["correctness"]["pass"])
        and bool(row["memory"]["temporary_memory_recovered"])
        for row in configurations
    )
    contract_pass = (
        len(reset_union) == 2
        and no_execute_side_threads
        and no_threads_outside_first_reset
        and threads_persisted
        and repeat_zero
        and retained_lifecycle_zero
        and semantic_pass
    )
    return {
        "cycle_id": receipt["cycle_id"],
        "first_reset_thread_ids": reset_union,
        "first_reset_close_thread_ids": sorted(close_ids),
        "first_reset_cleanup_thread_ids": sorted(cleanup_ids),
        "first_reset_thread_count": len(reset_union),
        "no_threads_during_open_warmup_or_retained_execute": (
            no_execute_side_threads
        ),
        "no_threads_outside_first_reset": no_threads_outside_first_reset,
        "first_reset_threads_persisted": threads_persisted,
        "repeated_reset_created_zero_threads": repeat_zero,
        "all_retained_lifecycle_deltas_zero": retained_lifecycle_zero,
        "all_correctness_and_memory_gates_pass": semantic_pass,
        "status": "PASS" if contract_pass else "FAIL",
    }


def validate_reset_contract(
    trace_a_path: Path,
    trace_b_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run H014-027t over two immutable real request lifecycle traces."""
    trace_paths = [
        trace_a_path.expanduser().resolve(),
        trace_b_path.expanduser().resolve(),
    ]
    receipts = [
        json.loads(path.read_text(encoding="utf-8")) for path in trace_paths
    ]
    if {str(receipt.get("cycle_id")) for receipt in receipts} != {
        "H014-027p",
        "H014-027s",
    }:
        raise ValueError("H014-027t requires H014-027p and H014-027s traces")
    validations = [_validate_reset_trace(receipt) for receipt in receipts]
    hypothesis_supported = all(row["status"] == "PASS" for row in validations)
    receipt = {
        "schema_version": "experiment-014-k3-reset-contract-v1",
        "cycle_id": "H014-027t",
        "hypothesis": (
            "Across H014-027p and H014-027s, exactly two persistent parent threads "
            "appear only in the first sustained eight-session close-plus-cleanup RESET "
            "boundary; open, warmup, retained EXECUTE, and repeat RESET create zero."
        ),
        "implementation": (
            "Evidence-only mechanical validation of two immutable real-weight request "
            "lifecycle receipts; no CUDA execution or runtime change."
        ),
        "sources": [
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "cycle_id": receipt["cycle_id"],
            }
            for path, receipt in zip(trace_paths, receipts, strict=True)
        ],
        "benchmark": {"trace_validations": validations},
        "result": {"hypothesis_supported": hypothesis_supported},
        "inspection": {
            "actual_bottleneck": (
                "SYNCHRONOUS_SERIAL_SCHEDULER"
                if hypothesis_supported
                else "UNBOUNDED_REQUEST_LIFECYCLE_ACTIVITY"
            ),
            "reset_activity_classification": (
                "BOUNDED_REQUEST_SPECIFIC_TEARDOWN"
                if hypothesis_supported
                else "CERTIFICATION_BLOCKER"
            ),
        },
        "decision": (
            "RETAIN_RESET_CONTRACT_AND_IMPLEMENT_BATCH2"
            if hypothesis_supported
            else "INSPECT_FAILED_RESET_TRACE"
        ),
        "redesign": (
            "H014-027u tests the minimum real batch-2 canonical CUDA stage path."
            if hypothesis_supported
            else "Resolve the first failed mechanical reset invariant."
        ),
        "status": "PASS" if hypothesis_supported else "FAIL",
    }
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt["output_path"] = str(resolved_output)
    receipt["output_sha256"] = _sha256_file(resolved_output)
    return receipt


def _run_batch2_primitive_mode(
    runtime: Any,
    execute: Any,
    input_pointer: ctypes.c_void_p,
    output_pointer: ctypes.c_void_p,
    *,
    input_dimension: int,
    output_dimension: int,
    batched: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        if batched:
            execute(output_pointer, input_pointer, 2)
        else:
            for row in range(2):
                execute(
                    _pointer_offset(output_pointer, row * output_dimension),
                    _pointer_offset(input_pointer, row * input_dimension),
                    1,
                )
    runtime.synchronize()
    wall_ms: list[float] = []
    device_ms: list[float] = []
    for _ in range(iterations):
        wall_started = time.perf_counter_ns()
        runtime.profile_begin()
        if batched:
            execute(output_pointer, input_pointer, 2)
        else:
            for row in range(2):
                execute(
                    _pointer_offset(output_pointer, row * output_dimension),
                    _pointer_offset(input_pointer, row * input_dimension),
                    1,
                )
        device_ms.append(runtime.profile_end())
        wall_ms.append((time.perf_counter_ns() - wall_started) / 1e6)
    output = runtime.download_activation(output_pointer, (2, output_dimension))
    return {
        "mode": "batch2" if batched else "serial_batch1_pair",
        "warmup_pairs": warmup,
        "retained_pairs": iterations,
        "native_calls_per_pair": 1 if batched else 2,
        "rows_per_pair": 2,
        "wall": _timing(wall_ms),
        "device": _timing(device_ms),
        "output": output,
        "output_fingerprint": _array_fingerprint(output),
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
    }


def _benchmark_batch2_primitive(
    executor: PersistentKimiStageExecutor,
    *,
    name: str,
    execute: Any,
    activation: np.ndarray,
    output_dimension: int,
    weight_bytes: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    runtime = executor.runtime
    input_rows = np.ascontiguousarray(
        np.stack((activation, activation)), dtype=np.float32
    )
    input_dimension = int(input_rows.shape[1])
    memory_before = runtime.mem_info()
    input_pointer = runtime.allocate(input_rows.nbytes)
    serial_pointer = runtime.allocate(2 * output_dimension * 4)
    batch_pointer = runtime.allocate(2 * output_dimension * 4)
    try:
        runtime.upload_activation(input_pointer, input_rows)
        memory_after_allocation = runtime.mem_info()
        serial_pre = _run_batch2_primitive_mode(
            runtime,
            execute,
            input_pointer,
            serial_pointer,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            batched=False,
            warmup=warmup,
            iterations=iterations,
        )
        batch2 = _run_batch2_primitive_mode(
            runtime,
            execute,
            input_pointer,
            batch_pointer,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            batched=True,
            warmup=warmup,
            iterations=iterations,
        )
        serial_post = _run_batch2_primitive_mode(
            runtime,
            execute,
            input_pointer,
            serial_pointer,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            batched=False,
            warmup=warmup,
            iterations=iterations,
        )
    finally:
        runtime.free(batch_pointer)
        runtime.free(serial_pointer)
        runtime.free(input_pointer)
    memory_after_free = runtime.mem_info()
    serial_reference = np.ascontiguousarray(serial_post.pop("output"))
    serial_pre_output = np.ascontiguousarray(serial_pre.pop("output"))
    batch_output = np.ascontiguousarray(batch2.pop("output"))
    pre_post_metrics = _numerical_metrics(serial_reference, serial_pre_output)
    batch_metrics = _numerical_metrics(serial_reference, batch_output)
    serial_device_p50 = statistics.fmean(
        (
            float(serial_pre["device"]["p50_ms"]),
            float(serial_post["device"]["p50_ms"]),
        )
    )
    serial_wall_p50 = statistics.fmean(
        (
            float(serial_pre["wall"]["p50_ms"]),
            float(serial_post["wall"]["p50_ms"]),
        )
    )
    batch_device_p50 = float(batch2["device"]["p50_ms"])
    batch_wall_p50 = float(batch2["wall"]["p50_ms"])
    device_speedup = serial_device_p50 / batch_device_p50
    wall_speedup = serial_wall_p50 / batch_wall_p50
    correctness_pass = (
        float(batch_metrics["relative_l2_error"]) <= 1e-6
        and float(batch_metrics["cosine_similarity"]) >= 0.999999
        and float(pre_post_metrics["relative_l2_error"]) <= 1e-7
    )
    return {
        "primitive": name,
        "dimensions": {
            "batch": 2,
            "input": input_dimension,
            "output": output_dimension,
        },
        "input_fingerprint": _array_fingerprint(input_rows),
        "weight_source_bytes": weight_bytes,
        "modes": {
            "serial_pre": serial_pre,
            "batch2": batch2,
            "serial_post": serial_post,
        },
        "correctness": {
            "serial_pre_vs_post": pre_post_metrics,
            "batch2_vs_serial_post": batch_metrics,
            "serial_output_fingerprint": _array_fingerprint(serial_reference),
            "batch2_output_fingerprint": _array_fingerprint(batch_output),
            "pass": correctness_pass,
        },
        "performance": {
            "serial_pair_device_p50_ms": serial_device_p50,
            "batch2_pair_device_p50_ms": batch_device_p50,
            "device_pair_throughput_speedup": device_speedup,
            "serial_pair_wall_p50_ms": serial_wall_p50,
            "batch2_pair_wall_p50_ms": batch_wall_p50,
            "wall_pair_throughput_speedup": wall_speedup,
            "serial_rows_per_second": 2000.0 / serial_device_p50,
            "batch2_rows_per_second": 2000.0 / batch_device_p50,
            "serial_effective_weight_bandwidth_gbps": (
                2 * weight_bytes / (serial_device_p50 * 1e6)
            ),
            "batch2_effective_weight_bandwidth_gbps": (
                2 * weight_bytes / (batch_device_p50 * 1e6)
            ),
        },
        "memory": {
            "temporary_device_bytes": max(
                0,
                int(memory_before["free_bytes"])
                - int(memory_after_allocation["free_bytes"]),
            ),
            "free_bytes_before": memory_before["free_bytes"],
            "free_bytes_after_free": memory_after_free["free_bytes"],
            "temporary_memory_recovered": (
                memory_after_free["free_bytes"] >= memory_before["free_bytes"]
            ),
        },
        "timed_transfer_gate": all(
            int(mode[key]) == 0
            for mode in (serial_pre, batch2, serial_post)
            for key in ("timed_h2d_bytes", "timed_d2h_bytes")
        ),
        "no_weight_or_layout_conversion": True,
        "hypothesis_supported": device_speedup >= 1.5 and correctness_pass,
        "status": (
            "PASS"
            if correctness_pass
            and memory_after_free["free_bytes"] >= memory_before["free_bytes"]
            else "FAIL"
        ),
    }


async def _benchmark_batch2_primitives(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    depth_profile_path: Path,
    reset_contract_path: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError("batch2 primitive benchmark requires >=3 warmups and >=20 pairs")
    depth = json.loads(depth_profile_path.read_text(encoding="utf-8"))
    reset = json.loads(reset_contract_path.read_text(encoding="utf-8"))
    if (
        depth.get("cycle_id") != "H014-027m"
        or depth.get("status") != "PASS"
        or int(depth["benchmark"]["slowest_device_p50"]["layer"]) != 89
        or reset.get("cycle_id") != "H014-027t"
        or reset.get("status") != "PASS"
    ):
        raise ValueError("H014-027u requires retained depth and RESET evidence")
    layer = 89
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    runtime = PersistentStageRuntime(
        worker_id="h014-027u-batch2-worker-089",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=_CaptureConnectionPool(),  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id="h014-027u-layer-89-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id="h014-027u-layer-89",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    try:
        load_response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        session_id = "h014-027u-activation-fixture"
        executor.open_session(session_id)
        try:
            result = executor.execute_decode(
                session_id=session_id,
                hidden_states=torch.from_numpy(inputs[0].copy()),
                cache_position_start=0,
            )
            record = executor.execution_records[-1]
            session = executor._require_session(session_id)
            mlp_input = executor.runtime.download_activation(
                session.mlp_input, (executor.config.hidden,)
            )
            latent_input = executor.runtime.download_activation(
                session.latent_input, (executor.config.latent,)
            )
            observed_boundary = (
                result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
            )
            route_exact = (
                list(record["selected_expert_ids"]) == expected_routes[0]
            )
            selected_expert = int(record["selected_expert_ids"][0])
        finally:
            executor.close_session(session_id)
        oracle_metrics = _numerical_metrics(observed_boundary, expected_boundaries[0])
        catalog = _SafetensorCatalog(checkpoint)
        moe_prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
        latent_down_name = f"{moe_prefix}.routed_expert_down_proj.weight"
        source_bytes = _expert_source_bytes(catalog, layer)
        primitives = [
            _benchmark_batch2_primitive(
                executor,
                name="latent_down_dense",
                execute=lambda output, source, batch: executor.runtime.execute_dense(
                    executor._weights["latent_down"], output, source, batch
                ),
                activation=mlp_input,
                output_dimension=executor.config.latent,
                weight_bytes=int(catalog.tensor_info(latent_down_name)[3]),
                warmup=warmup,
                iterations=iterations,
            ),
            _benchmark_batch2_primitive(
                executor,
                name="selected_mxfp4_routed_expert",
                execute=lambda output, source, batch: executor.runtime.execute_resident(
                    executor._experts[selected_expert], output, source, batch
                ),
                activation=latent_input,
                output_dimension=executor.config.latent,
                weight_bytes=int(source_bytes["routed_expert_bytes_each"]),
                warmup=warmup,
                iterations=iterations,
            ),
            _benchmark_batch2_primitive(
                executor,
                name="shared_expert",
                execute=lambda output, source, batch: executor.runtime.execute_resident(
                    executor._weights["shared_mlp"], output, source, batch
                ),
                activation=mlp_input,
                output_dimension=executor.config.hidden,
                weight_bytes=int(source_bytes["shared_expert_bytes"]),
                warmup=warmup,
                iterations=iterations,
            ),
        ]
        oracle_pass = (
            float(oracle_metrics["relative_l2_error"]) <= 3e-5 and route_exact
        )
        execution_pass = (
            bool(load_response.accepted)
            and oracle_pass
            and all(row["status"] == "PASS" for row in primitives)
        )
        hypothesis_supported = execution_pass and all(
            bool(row["hypothesis_supported"]) for row in primitives
        )
        receipt = {
            "schema_version": "experiment-014-k3-batch2-primitives-v1",
            "cycle_id": "H014-027u",
            "hypothesis": (
                "Existing resident batch=2 dense, routed-expert, and shared-expert "
                "primitives provide at least 1.5x pair throughput with <=1e-6 relative "
                "error and no timed transfers or conversion."
            ),
            "implementation": (
                "Benchmark-only contiguous two-row adapter over unchanged registered "
                "layer-89 resident handles; no kernel, stage, state, or scheduler change."
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "device": _device_identity(device),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "target_binary": "sm_86+compute_86 PTX",
                "cpu_mathematical_fallbacks": 0,
            },
            "sources": {
                "depth_profile": {
                    "path": str(depth_profile_path),
                    "sha256": _sha256_file(depth_profile_path),
                },
                "reset_contract": {
                    "path": str(reset_contract_path),
                    "sha256": _sha256_file(reset_contract_path),
                },
            },
            "fixture": {
                "layer": layer,
                "attention_type": "KDA",
                "stage_weight_fingerprint": executor.weight_fingerprint,
                "input_boundary_fingerprint": _array_fingerprint(inputs[0]),
                "oracle_output_fingerprint": _array_fingerprint(
                    expected_boundaries[0]
                ),
                "observed_output_fingerprint": _array_fingerprint(observed_boundary),
                "oracle_metrics": oracle_metrics,
                "selected_expert_id": selected_expert,
                "selected_expert_ids": list(record["selected_expert_ids"]),
                "routing_equality": route_exact,
            },
            "benchmark": {
                "warmup_pairs_per_mode": warmup,
                "retained_pairs_per_mode": iterations,
                "primitives": primitives,
            },
            "result": {
                "execution_pass": execution_pass,
                "hypothesis_supported": hypothesis_supported,
            },
            "inspection": {
                "actual_bottleneck": (
                    "CANONICAL_BATCH_ADAPTER_MISSING"
                    if hypothesis_supported
                    else "GENERIC_BATCH_PRIMITIVE_OR_CORRECTNESS_LIMIT"
                ),
                "timed_weight_uploads": 0,
                "timed_layout_conversions": 0,
            },
            "decision": (
                "IMPLEMENT_MINIMUM_CANONICAL_BATCH2_ADAPTER"
                if hypothesis_supported
                else "PROFILE_FAILING_BATCH2_PRIMITIVE"
            ),
            "redesign": (
                "H014-027v integrates batch-2 weight-reading phases into one canonical stage."
                if hypothesis_supported
                else "Form the next hypothesis from the failing primitive timing/output."
            ),
            "status": "PASS" if execution_pass else "FAIL",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["output_path"] = str(output_path.resolve())
        receipt["output_sha256"] = _sha256_file(output_path)
        return receipt
    finally:
        await runtime.close()


def benchmark_batch2_primitives(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    depth_profile_path: Path,
    reset_contract_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run H014-027u over existing resident CUDA batch-2 primitives."""
    return asyncio.run(
        _benchmark_batch2_primitives(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            depth_profile_path.expanduser().resolve(),
            reset_contract_path.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    )


def _run_batch_scaling_mode(
    runtime: Any,
    execute: Any,
    activation: np.ndarray,
    *,
    output_dimension: int,
    batch: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    input_rows = np.ascontiguousarray(
        np.repeat(activation.reshape(1, -1), batch, axis=0), dtype=np.float32
    )
    memory_before = runtime.mem_info()
    input_pointer = runtime.allocate(input_rows.nbytes)
    output_pointer = runtime.allocate(batch * output_dimension * 4)
    try:
        runtime.upload_activation(input_pointer, input_rows)
        memory_after_allocation = runtime.mem_info()
        for _ in range(warmup):
            execute(output_pointer, input_pointer, batch)
        runtime.synchronize()
        memory_after_warmup = runtime.mem_info()
        wall_ms: list[float] = []
        device_ms: list[float] = []
        for _ in range(iterations):
            wall_started = time.perf_counter_ns()
            runtime.profile_begin()
            execute(output_pointer, input_pointer, batch)
            device_ms.append(runtime.profile_end())
            wall_ms.append((time.perf_counter_ns() - wall_started) / 1e6)
        output = runtime.download_activation(
            output_pointer, (batch, output_dimension)
        )
    finally:
        runtime.free(output_pointer)
        runtime.free(input_pointer)
    memory_after_free = runtime.mem_info()
    return {
        "batch": batch,
        "warmup_calls": warmup,
        "retained_calls": iterations,
        "native_calls_per_retained_call": 1,
        "wall": _timing(wall_ms),
        "device": _timing(device_ms),
        "output": output,
        "output_fingerprint": _array_fingerprint(output),
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "memory": {
            "explicit_input_output_bytes": (
                input_rows.nbytes + batch * output_dimension * 4
            ),
            "peak_allocated_delta_bytes": max(
                0,
                int(memory_before["free_bytes"])
                - min(
                    int(memory_after_allocation["free_bytes"]),
                    int(memory_after_warmup["free_bytes"]),
                ),
            ),
            "persistent_workspace_growth_bytes": max(
                0,
                int(memory_before["free_bytes"])
                - int(memory_after_free["free_bytes"]),
            ),
            "free_bytes_before": memory_before["free_bytes"],
            "free_bytes_after_free": memory_after_free["free_bytes"],
        },
    }


def _benchmark_batch_scaling_primitive(
    executor: PersistentKimiStageExecutor,
    *,
    name: str,
    execute: Any,
    activation: np.ndarray,
    output_dimension: int,
    weight_bytes: int,
    kernel_structure: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    modes = [
        _run_batch_scaling_mode(
            executor.runtime,
            execute,
            activation,
            output_dimension=output_dimension,
            batch=batch,
            warmup=warmup,
            iterations=iterations,
        )
        for batch in (1, 2, 4, 8, 16)
    ]
    batch1_post = _run_batch_scaling_mode(
        executor.runtime,
        execute,
        activation,
        output_dimension=output_dimension,
        batch=1,
        warmup=warmup,
        iterations=iterations,
    )
    reference = np.ascontiguousarray(batch1_post.pop("output"))[0]
    batch1_pre_p50 = float(modes[0]["device"]["p50_ms"])
    batch1_post_p50 = float(batch1_post["device"]["p50_ms"])
    batch1_control_p50 = statistics.fmean((batch1_pre_p50, batch1_post_p50))
    correctness_pass = True
    timed_transfer_gate = True
    for mode in modes:
        output = np.ascontiguousarray(mode.pop("output"))
        expected = np.ascontiguousarray(
            np.repeat(reference.reshape(1, -1), int(mode["batch"]), axis=0)
        )
        metrics = _numerical_metrics(output, expected)
        exact = bool(np.array_equal(output, expected))
        mode["correctness"] = {
            "metrics": metrics,
            "bit_exact": exact,
            "pass": (
                float(metrics["relative_l2_error"]) <= 1e-6
                and float(metrics["cosine_similarity"]) >= 0.999999
            ),
        }
        device_p50 = float(mode["device"]["p50_ms"])
        batch = int(mode["batch"])
        mode["performance"] = {
            "rows_per_second": 1000.0 * batch / device_p50,
            "per_row_device_p50_ms": device_p50 / batch,
            "row_throughput_speedup_vs_batch1": (
                batch * batch1_control_p50 / device_p50
            ),
            "duplicated_weight_traffic_gbps": (
                batch * weight_bytes / (device_p50 * 1e6)
            ),
        }
        correctness_pass = correctness_pass and bool(mode["correctness"]["pass"])
        timed_transfer_gate = timed_transfer_gate and all(
            int(mode[key]) == 0 for key in ("timed_h2d_bytes", "timed_d2h_bytes")
        )
    post_output_fingerprint = _array_fingerprint(reference)
    post_metrics = _numerical_metrics(reference, reference)
    control_drift_percent = (
        abs(batch1_pre_p50 - batch1_post_p50) / batch1_control_p50 * 100.0
    )
    hypothesis_supported = correctness_pass and all(
        float(mode["performance"]["row_throughput_speedup_vs_batch1"]) < 1.25
        for mode in modes[1:]
    )
    return {
        "primitive": name,
        "dimensions": {
            "input": int(activation.size),
            "output": output_dimension,
            "batches": [1, 2, 4, 8, 16],
        },
        "input_fingerprint": _array_fingerprint(activation),
        "weight_source_bytes": weight_bytes,
        "kernel_structure": kernel_structure,
        "modes": modes,
        "batch1_post_control": batch1_post,
        "batch1_reference_fingerprint": post_output_fingerprint,
        "batch1_reference_self_check": post_metrics,
        "batch1_control_device_p50_ms": batch1_control_p50,
        "batch1_control_drift_percent": control_drift_percent,
        "correctness_pass": correctness_pass,
        "timed_transfer_gate": timed_transfer_gate,
        "hypothesis_supported": hypothesis_supported,
        "status": (
            "PASS"
            if correctness_pass and timed_transfer_gate and control_drift_percent <= 5.0
            else "FAIL"
        ),
    }


async def _benchmark_batch_scaling(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    batch2_profile_path: Path,
    output_path: Path,
    *,
    device: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    if warmup < 3 or iterations < 20:
        raise ValueError("batch scaling requires >=3 warmups and >=20 calls")
    batch2_profile = json.loads(batch2_profile_path.read_text(encoding="utf-8"))
    if (
        batch2_profile.get("cycle_id") != "H014-027u"
        or batch2_profile.get("status") != "PASS"
        or batch2_profile.get("result", {}).get("hypothesis_supported") is not False
    ):
        raise ValueError("H014-027v requires the retained falsified H014-027u result")
    native_source = (
        Path(__file__).resolve().parents[4]
        / "third_party"
        / "colibri"
        / "c"
        / "backend_cuda.cu"
    )
    layer = 89
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    stage_runtime = PersistentStageRuntime(
        worker_id="h014-027v-batch-scaling-worker-089",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=_CaptureConnectionPool(),  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=stage_runtime.worker_id,
        request_id="h014-027v-layer-89-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id="h014-027v-layer-89",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    try:
        load_response = await stage_runtime.load_stage(request)
        executor = stage_runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        session_id = "h014-027v-activation-fixture"
        executor.open_session(session_id)
        try:
            result = executor.execute_decode(
                session_id=session_id,
                hidden_states=torch.from_numpy(inputs[0].copy()),
                cache_position_start=0,
            )
            record = executor.execution_records[-1]
            session = executor._require_session(session_id)
            mlp_input = executor.runtime.download_activation(
                session.mlp_input, (executor.config.hidden,)
            )
            latent_input = executor.runtime.download_activation(
                session.latent_input, (executor.config.latent,)
            )
            observed_boundary = (
                result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
            )
            route_exact = list(record["selected_expert_ids"]) == expected_routes[0]
            selected_expert = int(record["selected_expert_ids"][0])
        finally:
            executor.close_session(session_id)
        oracle_metrics = _numerical_metrics(observed_boundary, expected_boundaries[0])
        catalog = _SafetensorCatalog(checkpoint)
        moe_prefix = f"language_model.model.layers.{layer}.block_sparse_moe"
        latent_down_name = f"{moe_prefix}.routed_expert_down_proj.weight"
        source_bytes = _expert_source_bytes(catalog, layer)
        primitives = [
            _benchmark_batch_scaling_primitive(
                executor,
                name="latent_down_dense",
                execute=lambda output, source, batch: executor.runtime.execute_dense(
                    executor._weights["latent_down"], output, source, batch
                ),
                activation=mlp_input,
                output_dimension=executor.config.latent,
                weight_bytes=int(catalog.tensor_info(latent_down_name)[3]),
                kernel_structure=(
                    "one quant_matmul launch; grid=(output,row); independent "
                    "weight reads per row"
                ),
                warmup=warmup,
                iterations=iterations,
            ),
            _benchmark_batch_scaling_primitive(
                executor,
                name="selected_mxfp4_routed_expert",
                execute=lambda output, source, batch: executor.runtime.execute_resident(
                    executor._experts[selected_expert], output, source, batch
                ),
                activation=latent_input,
                output_dimension=executor.config.latent,
                weight_bytes=int(source_bytes["routed_expert_bytes_each"]),
                kernel_structure=(
                    "default fused gate/up + SiTU + down launches; projection "
                    "grids=(output,row); repeated selected expert is best-case reuse"
                ),
                warmup=warmup,
                iterations=iterations,
            ),
            _benchmark_batch_scaling_primitive(
                executor,
                name="shared_expert",
                execute=lambda output, source, batch: executor.runtime.execute_resident(
                    executor._weights["shared_mlp"], output, source, batch
                ),
                activation=mlp_input,
                output_dimension=executor.config.hidden,
                weight_bytes=int(source_bytes["shared_expert_bytes"]),
                kernel_structure=(
                    "default fused gate/up + SiTU + down launches; projection "
                    "grids=(output,row); independent weight reads per row"
                ),
                warmup=warmup,
                iterations=iterations,
            ),
        ]
        oracle_pass = (
            float(oracle_metrics["relative_l2_error"]) <= 3e-5 and route_exact
        )
        execution_pass = (
            bool(load_response.accepted)
            and oracle_pass
            and all(row["status"] == "PASS" for row in primitives)
        )
        hypothesis_supported = execution_pass and all(
            bool(row["hypothesis_supported"]) for row in primitives
        )
        maximum_speedup = max(
            float(mode["performance"]["row_throughput_speedup_vs_batch1"])
            for row in primitives
            for mode in row["modes"][1:]
        )
        receipt = {
            "schema_version": "experiment-014-k3-batch-scaling-v1",
            "cycle_id": "H014-027v",
            "hypothesis": (
                "The unchanged row-parallel primitives remain below 1.25x row-"
                "throughput speedup at batch 2/4/8/16 for every tested class, with "
                "correct outputs and plateauing per-row latency."
            ),
            "implementation": (
                "Benchmark-only batch sweep over unchanged registered layer-89 "
                "resident handles and duplicated oracle-derived activations."
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "device": _device_identity(device),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "native_source_sha256": _sha256_file(native_source),
                "target_binary": "sm_86+compute_86 PTX",
                "cpu_mathematical_fallbacks": 0,
            },
            "sources": {
                "batch2_profile": {
                    "path": str(batch2_profile_path),
                    "sha256": _sha256_file(batch2_profile_path),
                },
                "native_source": str(native_source),
            },
            "fixture": {
                "layer": layer,
                "attention_type": "KDA",
                "stage_weight_fingerprint": executor.weight_fingerprint,
                "input_boundary_fingerprint": _array_fingerprint(inputs[0]),
                "observed_output_fingerprint": _array_fingerprint(observed_boundary),
                "oracle_metrics": oracle_metrics,
                "selected_expert_id": selected_expert,
                "selected_expert_ids": list(record["selected_expert_ids"]),
                "routing_equality": route_exact,
                "activation_policy": (
                    "identical real activation rows; selected routed expert is a "
                    "best-case repeated-expert reuse bound"
                ),
            },
            "benchmark": {
                "batches": [1, 2, 4, 8, 16],
                "warmup_calls_per_size": warmup,
                "retained_calls_per_size": iterations,
                "primitives": primitives,
            },
            "result": {
                "execution_pass": execution_pass,
                "hypothesis_supported": hypothesis_supported,
                "maximum_row_throughput_speedup": maximum_speedup,
            },
            "inspection": {
                "actual_bottleneck": (
                    "ROW_PARALLEL_WEIGHT_RELOAD"
                    if hypothesis_supported
                    else "BATCH_SCALING_REQUIRES_PRIMITIVE_SPECIFIC_INSPECTION"
                ),
                "timed_weight_uploads": 0,
                "timed_layout_conversions": 0,
            },
            "decision": (
                "PROTOTYPE_MINIMUM_ROW_COOPERATIVE_KERNEL"
                if hypothesis_supported
                else "RETAIN_ONLY_SCALING_PRIMITIVES"
            ),
            "redesign": (
                "H014-027w tests one row-cooperative weight-sharing kernel against "
                "the immutable scaling baseline."
                if hypothesis_supported
                else "Integrate only the primitives whose scaling is supported."
            ),
            "status": "PASS" if execution_pass else "FAIL",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["output_path"] = str(output_path.resolve())
        receipt["output_sha256"] = _sha256_file(output_path)
        return receipt
    finally:
        await stage_runtime.close()


def benchmark_batch_scaling(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    batch2_profile_path: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    """Run H014-027v over unchanged real-Kimi CUDA batch primitives."""
    return asyncio.run(
        _benchmark_batch_scaling(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            batch2_profile_path.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
    )


def _run_expert_group_block(
    executor: PersistentKimiStageExecutor,
    stage_runtime: PersistentStageRuntime,
    *,
    session_id: str,
    groups: list[list[int]],
    label: str,
    warmup: int,
    iterations: int,
    expert_bytes_each: int,
    device: int,
) -> dict[str, Any]:
    session = executor._require_session(session_id)
    runtime = executor.runtime
    sampler = _GpuSampler(device)
    sampler.start()
    wall_ms: list[float] = []
    device_ms: list[float] = []
    before: dict[str, Any] | None = None
    validation_fingerprint: str | None = None
    total = warmup + iterations
    try:
        for index in range(total):
            if index == warmup:
                before = _process_snapshot(stage_runtime, executor)
            group = groups[index % len(groups)]
            wall_started = time.perf_counter_ns()
            runtime.profile_begin()
            for slot, expert in enumerate(group):
                runtime.execute_resident(
                    executor._experts[expert],
                    _pointer_offset(session.expert_rows, slot * executor.config.latent),
                    session.latent_input,
                    1,
                )
            elapsed_device_ms = runtime.profile_end()
            elapsed_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
            if index >= warmup:
                device_ms.append(elapsed_device_ms)
                wall_ms.append(elapsed_wall_ms)
                if validation_fingerprint is None and group == groups[0]:
                    outputs = runtime.download_activation(
                        session.expert_rows,
                        (executor.config.topk, executor.config.latent),
                    )
                    validation_fingerprint = _array_fingerprint(outputs)
        if before is None or validation_fingerprint is None:
            raise RuntimeError(f"expert working-set block {label} retained no evidence")
        after = _process_snapshot(stage_runtime, executor)
    finally:
        gpu_utilization = sampler.stop()
    lifecycle_delta = _warm_lifecycle_delta(before, after)
    device = _timing(device_ms)
    group_source_bytes = 16 * expert_bytes_each
    return {
        "label": label,
        "unique_experts_in_schedule": len({expert for group in groups for expert in group}),
        "group_count_in_schedule": len(groups),
        "warmup_groups": warmup,
        "retained_groups": iterations,
        "wall": _timing(wall_ms),
        "device": device,
        "group_source_weight_bytes": group_source_bytes,
        "effective_source_weight_bandwidth_gbps": (
            group_source_bytes / (float(device["mean_ms"]) * 1e6)
        ),
        "timed_transfers": {"h2d_bytes": 0, "d2h_bytes": 0, "d2d_bytes": 0},
        "kernel_launches_per_group": 48,
        "kernel_launch_basis": "16 x (fused gate/up + SiTU + down)",
        "validation_output_fingerprint": validation_fingerprint,
        "gpu_utilization": gpu_utilization,
        "lifecycle_delta": lifecycle_delta,
        "lifecycle_deltas_zero": all(value == 0 for value in lifecycle_delta.values()),
    }


async def _benchmark_expert_working_set(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    steady_profile_path: Path,
    output_path: Path,
    *,
    device: int,
) -> dict[str, Any]:
    steady_profile = json.loads(steady_profile_path.read_text(encoding="utf-8"))
    if steady_profile.get("status") != "PASS" or steady_profile.get("cycle_id") != "H014-027a":
        raise ValueError("H014-027b requires the retained H014-027a profile")
    steady_layer = next(
        row for row in steady_profile["benchmark"]["layers"] if row["layer"] == 1
    )
    steady_device_p50 = float(steady_layer["modes"]["production"]["device"]["p50_ms"])
    layer = 1
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id="k3-working-set-worker-001",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id="h014-027b-layer-1-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id="h014-027b-layer-1",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    load_started = time.perf_counter_ns()
    try:
        load_response = await runtime.load_stage(request)
        load_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        executor.set_research_telemetry_mode("minimal")
        session_id = "h014-027b-state"
        executor.open_session(session_id)
        cold_calls: list[dict[str, Any]] = []
        try:
            for position in range(3):
                wall_started = time.perf_counter_ns()
                result = executor.execute_decode(
                    session_id=session_id,
                    hidden_states=torch.from_numpy(inputs[position].copy()),
                    cache_position_start=position,
                )
                external_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
                record = executor.execution_records[-1]
                output = result.stage_boundary_hidden_states.detach().cpu().numpy()
                metrics = _numerical_metrics(output, expected_boundaries[position])
                observed_routes = list(record["selected_expert_ids"])
                cold_calls.append(
                    {
                        "position": position,
                        "external_wall_ms": external_wall_ms,
                        "device_ms": float(record["device_ms"]),
                        "relative_l2_error": metrics["relative_l2_error"],
                        "maximum_absolute_error": metrics["maximum_absolute_error"],
                        "cosine_similarity": metrics["cosine_similarity"],
                        "routing_equality": observed_routes == expected_routes[position],
                        "selected_expert_ids": observed_routes,
                    }
                )
            fixed_ids = list(cold_calls[-1]["selected_expert_ids"])
            remaining = [expert for expert in range(896) if expert not in fixed_ids]
            permutation = fixed_ids + remaining
            rotating_groups = [
                permutation[offset : offset + 16]
                for offset in range(0, len(permutation), 16)
            ]
            expert_bytes = _expert_source_bytes(_SafetensorCatalog(checkpoint), layer)
            fixed_pre = _run_expert_group_block(
                executor,
                runtime,
                session_id=session_id,
                groups=[fixed_ids],
                label="fixed_top16_pre",
                warmup=20,
                iterations=100,
                expert_bytes_each=expert_bytes["routed_expert_bytes_each"],
                device=device,
            )
            rotating = _run_expert_group_block(
                executor,
                runtime,
                session_id=session_id,
                groups=rotating_groups,
                label="rotating_all_896",
                warmup=56,
                iterations=112,
                expert_bytes_each=expert_bytes["routed_expert_bytes_each"],
                device=device,
            )
            fixed_post = _run_expert_group_block(
                executor,
                runtime,
                session_id=session_id,
                groups=[fixed_ids],
                label="fixed_top16_post",
                warmup=20,
                iterations=100,
                expert_bytes_each=expert_bytes["routed_expert_bytes_each"],
                device=device,
            )
        finally:
            executor.close_session(session_id)
        fixed_p50 = statistics.fmean(
            [
                float(fixed_pre["device"]["p50_ms"]),
                float(fixed_post["device"]["p50_ms"]),
            ]
        )
        rotating_p50 = float(rotating["device"]["p50_ms"])
        working_set_ratio = rotating_p50 / fixed_p50
        fingerprints_equal = len(
            {
                fixed_pre["validation_output_fingerprint"],
                rotating["validation_output_fingerprint"],
                fixed_post["validation_output_fingerprint"],
            }
        ) == 1
        cold_device = [float(row["device_ms"]) for row in cold_calls]
        cold_to_steady_ratio = statistics.median(cold_device) / steady_device_p50
        hypothesis_supported = working_set_ratio >= 2.0
        lifecycle_pass = all(
            row["lifecycle_deltas_zero"]
            for row in (fixed_pre, rotating, fixed_post)
        )
        correctness_pass = all(
            float(row["relative_l2_error"]) <= 3e-5
            and bool(row["routing_equality"])
            for row in cold_calls
        )
        receipt = {
            "schema_version": "experiment-014-k3-expert-working-set-v1",
            "cycle_id": "H014-027b",
            "hypothesis": (
                "Cycling all 896 resident experts raises 16-expert-group device p50 "
                "by at least 2x versus a repeatedly reused real top-16 group."
            ),
            "implementation": (
                "Benchmark-only direct dispatch through the unchanged resident expert "
                "primitive using one oracle-seeded latent activation."
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "device": _device_identity(device),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "target_binary": "sm_86+compute_86 PTX",
                "cpu_mathematical_fallbacks": 0,
            },
            "checkpoint": {
                "layer": layer,
                "source_weight_bytes": assignment.weight_bytes,
                "resident_device_bytes": executor.resident_device_bytes,
                "weight_fingerprint": executor.weight_fingerprint,
                **expert_bytes,
            },
            "load": {"accepted": load_response.accepted, "elapsed_ms": load_ms},
            "cold_canonical_calls": cold_calls,
            "steady_reference": {
                "path": str(steady_profile_path.resolve()),
                "sha256": _sha256_file(steady_profile_path),
                "production_device_p50_ms": steady_device_p50,
                "cold_median_to_steady_p50_ratio": cold_to_steady_ratio,
            },
            "benchmark": {
                "fixed_top16_pre": fixed_pre,
                "rotating_all_896": rotating,
                "fixed_top16_post": fixed_post,
                "working_set_p50_ratio": working_set_ratio,
                "bracketing_output_fingerprints_equal": fingerprints_equal,
            },
            "result": {
                "hypothesis_supported": hypothesis_supported,
                "correctness_pass": correctness_pass,
                "lifecycle_pass": lifecycle_pass,
            },
            "inspection": {
                "actual_bottleneck": (
                    "EXPERT_WORKING_SET_REUSE"
                    if hypothesis_supported
                    else "NOT_EXPLAINED_BY_EXPERT_WORKING_SET_REUSE"
                ),
                "timed_h2d_bytes": 0,
                "timed_d2h_bytes": 0,
                "instrumentation_mode": "one CUDA event pair per 16-expert group",
            },
            "decision": "RETAIN_BENCHMARK_ONLY",
            "redesign": (
                "H014-027c tests expert scheduling/reuse in real batch and concurrency."
                if hypothesis_supported
                else "H014-027c profiles the cold canonical call sequence for clocks, paging, and driver scheduling."
            ),
            "status": (
                "PASS"
                if load_response.accepted
                and correctness_pass
                and lifecycle_pass
                and fingerprints_equal
                else "FAIL"
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["output_path"] = str(output_path.resolve())
        receipt["output_sha256"] = _sha256_file(output_path)
        return receipt
    finally:
        await runtime.close()


def benchmark_expert_working_set(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    steady_profile: Path,
    output_path: Path,
    *,
    device: int = 0,
) -> dict[str, Any]:
    """Run the H014-027b resident expert working-set experiment."""
    return asyncio.run(
        _benchmark_expert_working_set(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            steady_profile.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
        )
    )


async def _benchmark_device_warmup(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    steady_profile_path: Path,
    cold_profile_path: Path,
    output_path: Path,
    *,
    device: int,
    minimum_warmup_device_ms: float,
) -> dict[str, Any]:
    steady = json.loads(steady_profile_path.read_text(encoding="utf-8"))
    cold = json.loads(cold_profile_path.read_text(encoding="utf-8"))
    if steady.get("cycle_id") != "H014-027a" or steady.get("status") != "PASS":
        raise ValueError("device warmup requires the retained H014-027a profile")
    if cold.get("cycle_id") != "H014-027b" or cold.get("status") != "PASS":
        raise ValueError("device warmup requires the retained H014-027b profile")
    steady_layer = next(
        row for row in steady["benchmark"]["layers"] if row["layer"] == 1
    )
    steady_device_p50 = float(steady_layer["modes"]["production"]["device"]["p50_ms"])
    baseline_cold = [float(row["device_ms"]) for row in cold["cold_canonical_calls"]]
    baseline_cold_median = statistics.median(baseline_cold)
    layer = 1
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id="k3-device-warmup-worker-001",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id="h014-027c-layer-1-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        topology_id="h014-027c-layer-1",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_context_bucket=3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    load_started = time.perf_counter_ns()
    try:
        load_response = await runtime.load_stage(request)
        load_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        executor.set_research_telemetry_mode("minimal")
        session_id = "h014-027c-state"
        executor.open_session(session_id)
        try:
            session = executor._require_session(session_id)
            zeros = np.zeros(executor.config.hidden, dtype=np.float32)
            executor.runtime.upload_activation(session.input_row, zeros)
            executor.runtime.upload_activation(session.prefix_row, zeros)
            before_warmup = _process_snapshot(runtime, executor)
            warmup_wall_started = time.perf_counter_ns()
            warmup_device_ms = 0.0
            warmup_launches = 0
            batch_launches = 4096
            warmup_batches: list[float] = []
            while warmup_device_ms < minimum_warmup_device_ms:
                executor.runtime.profile_begin()
                for _ in range(batch_launches):
                    executor.runtime.execute_add(
                        session.prefix_row,
                        session.input_row,
                        executor.config.hidden,
                    )
                batch_device_ms = executor.runtime.profile_end()
                warmup_batches.append(batch_device_ms)
                warmup_device_ms += batch_device_ms
                warmup_launches += batch_launches
                if warmup_launches >= 1_000_000:
                    raise RuntimeError("generic CUDA warmup did not reach its device-time gate")
            warmup_wall_ms = (time.perf_counter_ns() - warmup_wall_started) / 1e6
            after_warmup = _process_snapshot(runtime, executor)
            canonical_calls: list[dict[str, Any]] = []
            for position in range(3):
                wall_started = time.perf_counter_ns()
                result = executor.execute_decode(
                    session_id=session_id,
                    hidden_states=torch.from_numpy(inputs[position].copy()),
                    cache_position_start=position,
                )
                external_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
                record = executor.execution_records[-1]
                output = result.stage_boundary_hidden_states.detach().cpu().numpy()
                metrics = _numerical_metrics(output, expected_boundaries[position])
                observed_routes = list(record["selected_expert_ids"])
                canonical_calls.append(
                    {
                        "position": position,
                        "external_wall_ms": external_wall_ms,
                        "device_ms": float(record["device_ms"]),
                        "relative_l2_error": metrics["relative_l2_error"],
                        "maximum_absolute_error": metrics["maximum_absolute_error"],
                        "cosine_similarity": metrics["cosine_similarity"],
                        "routing_equality": observed_routes == expected_routes[position],
                        "selected_expert_ids": observed_routes,
                    }
                )
        finally:
            executor.close_session(session_id)
        lifecycle_delta = _warm_lifecycle_delta(before_warmup, after_warmup)
        warmed_device = [float(row["device_ms"]) for row in canonical_calls]
        warmed_median = statistics.median(warmed_device)
        cold_improvement = baseline_cold_median / warmed_median
        warmed_to_steady = warmed_median / steady_device_p50
        hypothesis_supported = cold_improvement >= 5.0 and warmed_to_steady <= 2.0
        correctness_pass = all(
            float(row["relative_l2_error"]) <= 3e-5
            and bool(row["routing_equality"])
            for row in canonical_calls
        )
        lifecycle_pass = all(value == 0 for value in lifecycle_delta.values())
        receipt = {
            "schema_version": "experiment-014-k3-device-warmup-v1",
            "cycle_id": "H014-027c",
            "hypothesis": (
                "At least 100 ms of unrelated resident CUDA add work reduces the first "
                "three canonical call median by at least 5x and to at most 2x steady p50."
            ),
            "implementation": (
                "Benchmark-only PREPARE warmup through the generic resident add primitive; "
                "no expert weights or Kimi arithmetic are touched."
            ),
            "backend": {
                "identity": "nvidia_cuda_persistent_kimi_stage",
                "device": _device_identity(device),
                "cuda_library_sha256": _sha256_file(cuda_library),
                "target_binary": "sm_86+compute_86 PTX",
                "cpu_mathematical_fallbacks": 0,
            },
            "load": {"accepted": load_response.accepted, "elapsed_ms": load_ms},
            "warmup": {
                "primitive": "generic resident float32 add",
                "expert_weight_bytes_read": 0,
                "one_time_h2d_bytes": 2 * executor.config.hidden * 4,
                "minimum_device_ms": minimum_warmup_device_ms,
                "measured_device_ms": warmup_device_ms,
                "measured_wall_ms": warmup_wall_ms,
                "kernel_launches": warmup_launches,
                "batch_device_ms": warmup_batches,
                "lifecycle_delta": lifecycle_delta,
            },
            "benchmark": {
                "baseline_cold_device_ms": baseline_cold,
                "baseline_cold_median_ms": baseline_cold_median,
                "warmed_canonical_calls": canonical_calls,
                "warmed_median_device_ms": warmed_median,
                "steady_production_device_p50_ms": steady_device_p50,
                "cold_to_warmed_improvement": cold_improvement,
                "warmed_to_steady_ratio": warmed_to_steady,
            },
            "sources": {
                "steady_profile": {
                    "path": str(steady_profile_path.resolve()),
                    "sha256": _sha256_file(steady_profile_path),
                },
                "cold_profile": {
                    "path": str(cold_profile_path.resolve()),
                    "sha256": _sha256_file(cold_profile_path),
                },
            },
            "result": {
                "hypothesis_supported": hypothesis_supported,
                "correctness_pass": correctness_pass,
                "lifecycle_pass": lifecycle_pass,
            },
            "inspection": {
                "actual_bottleneck": (
                    "CUDA_DEVICE_NOT_COMPUTE_READY_AT_READY_BOUNDARY"
                    if hypothesis_supported
                    else "KERNEL_OR_DRIVER_SPECIFIC_COLD_TRANSIENT"
                ),
                "warmup_is_one_time_prepare_work": True,
                "warmup_is_excluded_from_service_rate": True,
            },
            "decision": (
                "RETAIN_PREPARE_WARMUP"
                if hypothesis_supported
                else "RETAIN_BENCHMARK_ONLY"
            ),
            "redesign": (
                "Require a measured device-compute warmup before READY and continue P2 steady profiling."
                if hypothesis_supported
                else "H014-027d captures an Nsight trace of the cold canonical path."
            ),
            "status": (
                "PASS"
                if load_response.accepted and correctness_pass and lifecycle_pass
                else "FAIL"
            ),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt["output_path"] = str(output_path.resolve())
        receipt["output_sha256"] = _sha256_file(output_path)
        return receipt
    finally:
        await runtime.close()


def benchmark_device_warmup(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    steady_profile: Path,
    cold_profile: Path,
    output_path: Path,
    *,
    device: int = 0,
    minimum_warmup_device_ms: float = 100.0,
) -> dict[str, Any]:
    """Run H014-027c's unrelated-compute readiness experiment."""
    return asyncio.run(
        _benchmark_device_warmup(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            steady_profile.expanduser().resolve(),
            cold_profile.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            minimum_warmup_device_ms=minimum_warmup_device_ms,
        )
    )


async def _integrated_readiness_layer(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes_path: Path,
    identity_manifest: Path,
    *,
    layer: int,
    steady_device_p50_ms: float,
    device: int,
    cycle_id: str,
) -> dict[str, Any]:
    assignment = _source_assignment(
        checkpoint, layer=layer, device=f"native-cuda:{device}"
    )
    inputs, expected_boundaries = _stage_fixtures(
        checkpoint, oracle_trace, layer=layer
    )
    expected_routes = _parse_oracle_routes(oracle_routes_path)[layer]
    connection_pool = _CaptureConnectionPool()
    runtime = PersistentStageRuntime(
        worker_id=f"k3-ready-worker-{layer:03d}",
        device=f"native-cuda:{device}",
        dtype="float32",
        memory_limit_bytes=31 * 1024**3,
        maximum_sessions=2,
        configured_model_path=checkpoint,
        configured_model_identity_path=identity_manifest,
        connection_pool=connection_pool,  # type: ignore[arg-type]
    )
    request = LoadStageRequest(
        worker_id=runtime.worker_id,
        request_id=f"{cycle_id.lower()}-layer-{layer}-load",
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
        fast_path_context_bucket=3,
        model_content_fingerprint=MODEL_CONTENT_FINGERPRINT,
        native_runtime_library=str(cuda_library),
        native_runtime_library_sha256=_sha256_file(cuda_library),
        device=f"native-cuda:{device}",
        dtype="float32",
        model_path=str(checkpoint),
    )
    load_started = time.perf_counter_ns()
    try:
        load_response = await runtime.load_stage(request)
        load_ms = (time.perf_counter_ns() - load_started) / 1e6
        executor = runtime.loaded_executor
        if not isinstance(executor, PersistentKimiStageExecutor):
            raise TypeError("registered adapter returned a non-Kimi stage executor")
        prepare = executor.lifecycle_snapshot()["prepare_warmup"]
        if not isinstance(prepare, dict):
            raise RuntimeError("registered executor omitted PREPARE warmup evidence")
        session_id = f"{cycle_id.lower()}-layer-{layer}-state"
        executor.open_session(session_id)
        before = _process_snapshot(runtime, executor)
        calls: list[dict[str, Any]] = []
        try:
            for position in range(3):
                wall_started = time.perf_counter_ns()
                result = executor.execute_decode(
                    session_id=session_id,
                    hidden_states=torch.from_numpy(inputs[position].copy()),
                    cache_position_start=position,
                )
                external_wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
                record = executor.execution_records[-1]
                output = result.stage_boundary_hidden_states.detach().cpu().numpy()
                metrics = _numerical_metrics(output, expected_boundaries[position])
                observed_routes = list(record["selected_expert_ids"])
                calls.append(
                    {
                        "position": position,
                        "external_wall_ms": external_wall_ms,
                        "device_ms": float(record["device_ms"]),
                        "input_fingerprint": _array_fingerprint(inputs[position]),
                        "output_fingerprint": _array_fingerprint(output),
                        "expected_output_fingerprint": _array_fingerprint(
                            expected_boundaries[position]
                        ),
                        "relative_l2_error": metrics["relative_l2_error"],
                        "maximum_absolute_error": metrics["maximum_absolute_error"],
                        "cosine_similarity": metrics["cosine_similarity"],
                        "routing_equality": observed_routes == expected_routes[position],
                        "selected_expert_ids": observed_routes,
                    }
                )
            state_bytes = executor.kv_cache_bytes(session_id)
            after = _process_snapshot(runtime, executor)
        finally:
            executor.close_session(session_id)
        lifecycle_delta = _warm_lifecycle_delta(before, after)
        device_values = [float(row["device_ms"]) for row in calls]
        median_device_ms = statistics.median(device_values)
        readiness_ratio = median_device_ms / steady_device_p50_ms
        correctness_pass = all(
            float(row["relative_l2_error"]) <= 3e-5
            and bool(row["routing_equality"])
            for row in calls
        )
        stage_fixture = prepare.get("stage_fixture")
        warmup_gate = (
            int(prepare["count"]) == 1
            and isinstance(stage_fixture, dict)
            and int(stage_fixture["iterations"]) == 7
            and float(prepare["total_measured_wall_ms"]) <= 250.0
            and int(prepare["maximum_temporary_device_bytes"]) <= 8 * 1024**2
            and bool(prepare["temporary_memory_recovered"])
            and bool(stage_fixture["temporary_memory_recovered"])
            and int(stage_fixture["active_sessions_after"]) == 0
            and bool(stage_fixture["serving_execute_count_restored"])
            and bool(stage_fixture["research_records_removed"])
            and bool(stage_fixture["output_finite"])
        )
        lifecycle_pass = all(value == 0 for value in lifecycle_delta.values())
        return {
            "layer": layer,
            "attention_type": "KDA" if layer % 4 != 3 else "Gated_MLA",
            "load": {"accepted": load_response.accepted, "elapsed_ms": load_ms},
            "assignment": {
                "source_weight_bytes": assignment.weight_bytes,
                "resident_device_bytes": executor.resident_device_bytes,
                "weight_fingerprint": executor.weight_fingerprint,
            },
            "prepare_warmup": prepare,
            "canonical_calls": calls,
            "first_three_device": _timing(device_values),
            "median_to_steady_p50_ratio": readiness_ratio,
            "steady_device_p50_ms": steady_device_p50_ms,
            "state_bytes": state_bytes,
            "lifecycle_delta": lifecycle_delta,
            "correctness_pass": correctness_pass,
            "warmup_gate": warmup_gate,
            "lifecycle_pass": lifecycle_pass,
            "status": (
                "PASS"
                if load_response.accepted
                and correctness_pass
                and warmup_gate
                and lifecycle_pass
                and readiness_ratio <= 2.0
                else "FAIL"
            ),
        }
    finally:
        await runtime.close()


async def _benchmark_integrated_readiness(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    steady_profile_path: Path,
    output_path: Path,
    *,
    device: int,
    cycle_id: str,
) -> dict[str, Any]:
    steady = json.loads(steady_profile_path.read_text(encoding="utf-8"))
    if steady.get("cycle_id") != "H014-027a" or steady.get("status") != "PASS":
        raise ValueError("integrated readiness requires the retained H014-027a profile")
    steady_by_layer = {
        int(row["layer"]): float(row["modes"]["production"]["device"]["p50_ms"])
        for row in steady["benchmark"]["layers"]
    }
    layers = [
        await _integrated_readiness_layer(
            checkpoint,
            cuda_library,
            oracle_trace,
            oracle_routes,
            identity_manifest,
            layer=layer,
            steady_device_p50_ms=steady_by_layer[layer],
            device=device,
            cycle_id=cycle_id,
        )
        for layer in (1, 3)
    ]
    supported = all(row["status"] == "PASS" for row in layers)
    receipt = {
        "schema_version": "experiment-014-k3-integrated-readiness-v1",
        "cycle_id": cycle_id,
        "hypothesis": (
            "Seven isolated assigned-stage PREPARE calls make each following KDA/MLA "
            "call at most 2x retained steady p50 within 250 ms and 8 MiB transient VRAM."
        ),
        "implementation": (
            "Production PREPARE uses one temporary 7-position assigned-stage session; "
            "all temporary state is freed before READY."
        ),
        "backend": {
            "identity": "nvidia_cuda_persistent_kimi_stage",
            "device": _device_identity(device),
            "cuda_library_sha256": _sha256_file(cuda_library),
            "target_binary": "sm_86+compute_86 PTX",
            "cpu_mathematical_fallbacks": 0,
        },
        "source": {
            "steady_profile": str(steady_profile_path.resolve()),
            "steady_profile_sha256": _sha256_file(steady_profile_path),
        },
        "benchmark": {"layers": layers},
        "result": {"hypothesis_supported": supported},
        "inspection": {
            "actual_bottleneck": (
                "RESOLVED_READY_BOUNDARY_DEVICE_WARMUP"
                if supported
                else "READY_BOUNDARY_DEVICE_WARMUP_STILL_INSUFFICIENT"
            ),
            "warmup_count_per_load": 1,
            "warmup_in_warm_execute_path": False,
        },
        "decision": "RETAIN_PRODUCTION_PREPARE_WARMUP" if supported else "MODIFY",
        "redesign": (
            "Continue P2 steady representative-layer, batch, and concurrency characterization."
            if supported
            else "Inspect the failing role before continuing P2."
        ),
        "status": "PASS" if supported else "FAIL",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["output_path"] = str(output_path.resolve())
    receipt["output_sha256"] = _sha256_file(output_path)
    return receipt


def benchmark_integrated_readiness(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    identity_manifest: Path,
    steady_profile: Path,
    output_path: Path,
    *,
    device: int = 0,
    cycle_id: str = "H014-027d",
) -> dict[str, Any]:
    """Run the H014-027d production PREPARE/READY validation."""
    return asyncio.run(
        _benchmark_integrated_readiness(
            checkpoint.expanduser().resolve(),
            cuda_library.expanduser().resolve(),
            oracle_trace.expanduser().resolve(),
            oracle_routes.expanduser().resolve(),
            identity_manifest.expanduser().resolve(),
            steady_profile.expanduser().resolve(),
            output_path.expanduser().resolve(),
            device=device,
            cycle_id=cycle_id,
        )
    )


__all__ = [
    "benchmark_batch2_primitives",
    "benchmark_batch_scaling",
    "benchmark_concurrency_baseline",
    "benchmark_depth_profile",
    "benchmark_device_warmup",
    "benchmark_expert_working_set",
    "benchmark_integrated_readiness",
    "benchmark_resident_stage_profile",
    "validate_reset_contract",
]
