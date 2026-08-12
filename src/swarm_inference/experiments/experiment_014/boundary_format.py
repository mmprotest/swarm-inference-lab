"""Numerical certification of lower-precision real Kimi coarse boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import _parse_oracle_routes
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request

SCHEMA_VERSION = "experiment-014-k3-coarse-boundary-format-v1"
STRICT_OUTPUT_RELATIVE_L2_GATE = 3e-5
FRAME_BYTES = 4


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _bf16_encode(values: np.ndarray) -> bytes:
    source = np.ascontiguousarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2").tobytes()


def _bf16_decode(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    bf16 = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
    return np.ascontiguousarray((bf16 << np.uint32(16)).view(np.float32).reshape(shape))


def _encode_fp32(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def _decode_fp32(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(payload, dtype="<f4").reshape(shape).copy()


def _encode_fp16(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f2").tobytes()


def _decode_fp16(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(payload, dtype="<f2").reshape(shape).astype(np.float32)


_FORMATS: dict[
    str,
    tuple[Callable[[np.ndarray], bytes], Callable[[bytes, tuple[int, ...]], np.ndarray]],
] = {
    "fp32": (_encode_fp32, _decode_fp32),
    "fp16": (_encode_fp16, _decode_fp16),
    "bf16": (_bf16_encode, _bf16_decode),
}


def _wire_frame(format_name: str, shape: tuple[int, ...], payload: bytes) -> bytes:
    return pickle.dumps(
        {"format": format_name, "shape": shape, "payload": payload}, protocol=5
    )


def _format_timing(
    boundaries: list[np.ndarray], *, format_name: str, iterations: int
) -> dict[str, Any]:
    encode, decode = _FORMATS[format_name]
    encode_ms: list[float] = []
    serialize_ms: list[float] = []
    decode_ms: list[float] = []
    wire_sizes: list[int] = []
    payload_sizes: list[int] = []
    for index in range(iterations):
        boundary = boundaries[index % len(boundaries)]
        started = time.perf_counter_ns()
        payload = encode(boundary)
        encode_ms.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        frame = _wire_frame(format_name, tuple(boundary.shape), payload)
        serialize_ms.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        decoded = decode(payload, tuple(boundary.shape))
        decode_ms.append((time.perf_counter_ns() - started) / 1e6)
        if decoded.shape != boundary.shape or not np.isfinite(decoded).all():
            raise RuntimeError(f"{format_name} boundary codec returned invalid values")
        payload_sizes.append(len(payload))
        wire_sizes.append(len(frame) + FRAME_BYTES)
    return {
        "iterations": iterations,
        "payload_bytes": int(np.median(payload_sizes)),
        "wire_bytes": int(np.median(wire_sizes)),
        "encode": _timing(encode_ms),
        "serialization": _timing(serialize_ms),
        "decode": _timing(decode_ms),
        "codec_service": _timing(
            [encode_ms[index] + serialize_ms[index] + decode_ms[index] for index in range(iterations)]
        ),
    }


def benchmark_coarse_boundary_formats(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    graph_certification: Path,
    coarse_receipt: Path,
    output_path: Path,
    *,
    device: int = 0,
    timing_iterations: int = 100,
    cycle_id: str = "H014-032e",
) -> dict[str, Any]:
    """Execute FP32/FP16/BF16 copies of actual stage-0 boundaries on CUDA."""
    if timing_iterations < 30:
        raise ValueError("boundary-format timing requires at least 30 iterations")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "graph_certification": graph_certification.resolve(),
        "coarse_receipt": coarse_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    coarse = json.loads(paths["coarse_receipt"].read_text(encoding="utf-8"))
    fixture = graph.get("fixture", {})
    provenance = {
        "graph_status": graph.get("status"),
        "trace_matches": fixture.get("oracle_trace_sha256")
        == _sha256_file(paths["oracle_trace"]),
        "routes_matches": fixture.get("oracle_routes_sha256")
        == _sha256_file(paths["oracle_routes"]),
        "coarse_status": coarse.get("status"),
        "coarse_hypothesis_supported": coarse.get("hypothesis_supported"),
    }
    provenance["pass"] = (
        provenance["graph_status"] == "PASS"
        and provenance["trace_matches"]
        and provenance["routes_matches"]
        and provenance["coarse_status"] == "PASS"
        and provenance["coarse_hypothesis_supported"] is True
    )
    if not provenance["pass"]:
        raise ValueError("boundary formats are not joined to passing graph/coarse evidence")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "FP16 halves the actual real stage-0 boundary while preserving exact "
                "layer-1 routes and graph-relative output error <=3e-5; BF16 is "
                "characterized under the same gate."
            ),
            "relative_l2_gate": STRICT_OUTPUT_RELATIVE_L2_GATE,
        },
        "configuration": {
            "source_layer": 0,
            "destination_layer": 1,
            "formats": list(_FORMATS),
            "positions": 3,
            "timing_iterations": timing_iterations,
            "device": device,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "provenance": provenance,
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    retain("preregistered")
    receipt["gpu_health_before"] = _health_snapshot(device)
    receipt["device_identity"] = _device_identity(device)
    retain("gpu_health_before", status=receipt["gpu_health_before"]["status"])
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before boundary-format benchmark")

    stage1: PersistentKimiStageExecutor | None = None
    stage0: PersistentKimiStageExecutor | None = None
    stage0_session = f"{cycle_id.lower()}-stage0"
    format_sessions = {name: f"{cycle_id.lower()}-stage1-{name}" for name in _FORMATS}
    try:
        load_started = time.perf_counter_ns()
        stage1 = PersistentKimiStageExecutor(
            request=_request(
                paths["checkpoint"],
                paths["cuda_library"],
                layer=1,
                device=device,
                cycle_id=cycle_id,
                maximum_context=3,
            ),
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
        )
        stage1.prepare_for_ready()
        stage0 = PersistentKimiStageExecutor(
            request=_request(
                paths["checkpoint"],
                paths["cuda_library"],
                layer=0,
                device=device,
                cycle_id=cycle_id,
                maximum_context=3,
            ),
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
        )
        stage0.prepare_for_ready()
        receipt["load"] = {
            "wall_ms": (time.perf_counter_ns() - load_started) / 1e6,
            "stage0_resident_device_bytes": stage0.resident_device_bytes,
            "stage1_resident_device_bytes": stage1.resident_device_bytes,
            "sum_resident_device_bytes": stage0.resident_device_bytes
            + stage1.resident_device_bytes,
            "stage0_weight_fingerprint": stage0.weight_fingerprint,
            "stage1_weight_fingerprint": stage1.weight_fingerprint,
            "stage0_prepare": stage0.prepare_warmup,
            "stage1_prepare": stage1.prepare_warmup,
        }
        retain("both_stages_loaded", sum_resident_device_bytes=receipt["load"]["sum_resident_device_bytes"])

        stage0.open_session(stage0_session, maximum_context_override=3)
        for session in format_sessions.values():
            stage1.open_session(session, maximum_context_override=3)
        before_stage0 = stage0.lifecycle_snapshot()
        before_stage1 = stage1.lifecycle_snapshot()
        expected_routes = _parse_oracle_routes(paths["oracle_routes"])[1]
        _, expected_boundaries = _stage_fixtures(
            paths["checkpoint"], paths["oracle_trace"], layer=1
        )
        coarse_comparisons = coarse["correctness"]["comparisons"]
        token_ids = (163584, 18699, 11)
        actual_boundaries: list[np.ndarray] = []
        rows_by_format: dict[str, list[dict[str, Any]]] = {
            name: [] for name in _FORMATS
        }
        outputs_by_format: dict[str, list[np.ndarray]] = {
            name: [] for name in _FORMATS
        }
        fp32_outputs: list[np.ndarray] = []
        fp32_routes: list[list[int]] = []

        for position, token_id in enumerate(token_ids):
            stage0_result = stage0.execute_prefill(
                session_id=stage0_session,
                token_ids=torch.tensor([[token_id]], dtype=torch.int64),
                cache_position_start=position,
            )
            actual = (
                stage0_result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
            )
            actual_boundaries.append(actual)
            if _array_fingerprint(actual) != coarse_comparisons[position][
                "stage0_boundary_fingerprint"
            ]:
                raise RuntimeError("reproduced stage-0 boundary differs from H014-032c")

            decoded_by_format: dict[str, np.ndarray] = {}
            for format_name, (encode, decode) in _FORMATS.items():
                payload = encode(actual)
                decoded_by_format[format_name] = decode(payload, tuple(actual.shape))
                input_metrics = _numerical_metrics(decoded_by_format[format_name], actual)
                result = stage1.execute_decode(
                    session_id=format_sessions[format_name],
                    hidden_states=torch.from_numpy(decoded_by_format[format_name]),
                    cache_position_start=position,
                )
                output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                outputs_by_format[format_name].append(output)
                record = stage1.execution_records[-1]
                if format_name == "fp32":
                    fp32_outputs.append(output)
                    fp32_routes.append(list(record["selected_expert_ids"]))
                rows_by_format[format_name].append(
                    {
                        "position": position,
                        "payload_bytes": len(payload),
                        "wire_bytes": len(
                            _wire_frame(format_name, tuple(actual.shape), payload)
                        )
                        + FRAME_BYTES,
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "decoded_input_metrics": input_metrics,
                        "graph_output_metrics": _numerical_metrics(
                            output, expected_boundaries[position]
                        ),
                        "output_fingerprint": _array_fingerprint(output),
                        "selected_expert_ids": list(record["selected_expert_ids"]),
                        "selected_weights": list(record["selected_weights"]),
                        "routes_match_graph_oracle": list(record["selected_expert_ids"])
                        == expected_routes[position],
                        "stage1_device_ms": float(record["device_ms"]),
                    }
                )

        after_stage0 = stage0.lifecycle_snapshot()
        after_stage1 = stage1.lifecycle_snapshot()
        states = {
            name: stage1.session_state_evidence(session)
            for name, session in format_sessions.items()
        }
        stage0_state = stage0.session_state_evidence(stage0_session)
        lifecycle_delta = {
            "stage0_weight_loading": after_stage0["weight_load_count"]
            - before_stage0["weight_load_count"],
            "stage0_model_materialization": after_stage0["model_materialization_count"]
            - before_stage0["model_materialization_count"],
            "stage0_persistent_buffer_allocation": after_stage0[
                "persistent_buffer_allocation_count"
            ]
            - before_stage0["persistent_buffer_allocation_count"],
            "stage1_weight_loading": after_stage1["weight_load_count"]
            - before_stage1["weight_load_count"],
            "stage1_model_materialization": after_stage1["model_materialization_count"]
            - before_stage1["model_materialization_count"],
            "stage1_persistent_buffer_allocation": after_stage1[
                "persistent_buffer_allocation_count"
            ]
            - before_stage1["persistent_buffer_allocation_count"],
        }

        format_results: dict[str, Any] = {}
        for format_name, rows in rows_by_format.items():
            for position, row in enumerate(rows):
                row["output_vs_fp32_metrics"] = _numerical_metrics(
                    outputs_by_format[format_name][position],
                    fp32_outputs[position],
                )
            maximum_graph_error = max(
                float(row["graph_output_metrics"]["relative_l2_error"]) for row in rows
            )
            routes_exact = all(bool(row["routes_match_graph_oracle"]) for row in rows)
            state = states[format_name]
            passed = (
                maximum_graph_error <= STRICT_OUTPUT_RELATIVE_L2_GATE
                and routes_exact
                and state["finite"]
                and state["cache_sequence_length"] == 3
            )
            format_results[format_name] = {
                "positions": rows,
                "maximum_graph_relative_l2_error": maximum_graph_error,
                "exact_routes": routes_exact,
                "state": state,
                "payload_bytes": rows[0]["payload_bytes"],
                "wire_bytes": rows[0]["wire_bytes"],
                "pass": passed,
            }

        # Re-execution above retains graph metrics.  FP32 is the numerical control;
        # compare format fingerprints where exact and report graph-relative deltas.
        for format_name in ("fp16", "bf16"):
            format_results[format_name]["output_bit_exact_to_fp32"] = [
                format_results[format_name]["positions"][position]["output_fingerprint"]
                == _array_fingerprint(fp32_outputs[position])
                for position in range(3)
            ]
            format_results[format_name]["routes_exact_to_fp32"] = [
                format_results[format_name]["positions"][position][
                    "selected_expert_ids"
                ]
                == fp32_routes[position]
                for position in range(3)
            ]
        format_results["fp32"]["output_bit_exact_to_fp32"] = [True, True, True]
        format_results["fp32"]["routes_exact_to_fp32"] = [True, True, True]

        for format_name in _FORMATS:
            format_results[format_name]["host_timing"] = _format_timing(
                actual_boundaries,
                format_name=format_name,
                iterations=timing_iterations,
            )

        fp16_supported = bool(format_results["fp16"]["pass"])
        bf16_supported = bool(format_results["bf16"]["pass"])
        receipt["formats"] = format_results
        receipt["state_isolation"] = {
            "stage0": stage0_state,
            "stage1_fingerprints_unique": len(
                {row["fingerprint"] for row in states.values()}
            )
            == len(states),
            "all_finite": all(bool(row["finite"]) for row in states.values()),
            "all_positions_three": all(
                int(row["cache_sequence_length"]) == 3 for row in states.values()
            ),
        }
        receipt["lifecycle_delta"] = lifecycle_delta
        receipt["lifecycle_deltas_zero"] = all(
            int(value) == 0 for value in lifecycle_delta.values()
        )
        receipt["hypothesis_supported"] = fp16_supported
        receipt["inspection"] = {
            "actual_bottleneck": (
                "numerical precision" if not fp16_supported else "stage compute and physical RTT"
            ),
            "fp16_payload_reduction_percent": 100.0
            * (1.0 - format_results["fp16"]["payload_bytes"] / format_results["fp32"]["payload_bytes"]),
            "bf16_supported": bf16_supported,
        }
        receipt["decision"] = {
            "production_boundary_format": "FP16" if fp16_supported else "FP32",
            "fp16": "RETAIN" if fp16_supported else "REJECT_NUMERICAL",
            "bf16": "RETAIN" if bf16_supported else "REJECT_NUMERICAL",
            "next_hypothesis": (
                "Integrate the selected lower-precision codec into the two-process TCP "
                "slice and measure real wire/service." if fp16_supported else
                "Retain FP32 coarse boundaries; do not weaken numerical gates."
            ),
        }
        retain(
            "format_execution_complete",
            fp16_supported=fp16_supported,
            bf16_supported=bf16_supported,
        )

        for session in format_sessions.values():
            stage1.close_session(session)
        stage0.close_session(stage0_session)
        stage0.runtime.synchronize()
        stage1.runtime.synchronize()
        receipt["post_run_checks"] = {
            "stage0_cuda_error_state_ok": stage0.runtime.error_state_ok(),
            "stage1_cuda_error_state_ok": stage1.runtime.error_state_ok(),
            "stage0_memory": stage0.runtime.mem_info(),
            "stage1_memory": stage1.runtime.mem_info(),
            "nvidia_smi": _health_snapshot(device),
        }
        execution_pass = (
            format_results["fp32"]["pass"]
            and receipt["lifecycle_deltas_zero"]
            and receipt["state_isolation"]["all_finite"]
            and receipt["state_isolation"]["all_positions_three"]
            and receipt["post_run_checks"]["stage0_cuda_error_state_ok"]
            and receipt["post_run_checks"]["stage1_cuda_error_state_ok"]
            and receipt["post_run_checks"]["nvidia_smi"]["status"] == "MEASURED"
        )
        receipt["execution_pass"] = execution_pass
        receipt["status"] = "PASS" if execution_pass else "FAIL"
        retain("complete", status=receipt["status"])
        return receipt
    except BaseException as exc:
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
    finally:
        if stage0 is not None:
            with suppress(Exception):
                stage0.close()
        if stage1 is not None:
            with suppress(Exception):
                stage1.close()
