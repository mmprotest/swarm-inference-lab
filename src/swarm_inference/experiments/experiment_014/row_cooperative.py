"""Incremental real-Kimi row-cooperative batch certification for H014-028."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_014.batch_safety import (
    MEMORY_STABILITY_TOLERANCE_BYTES,
    _health_snapshot,
    _run_mode,
    _serializable_modes,
    _upload_expert,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _CudaRuntime,
    _device_identity,
    _load_real_expert,
    _numerical_metrics,
    _sha256_file,
)

SCHEMA_VERSION = "experiment-014-k3-row-cooperative-batch2-v1"
MINIMUM_PAIR_THROUGHPUT_SPEEDUP = 1.50
MAXIMUM_BATCH1_REGRESSION_RATIO = 1.05


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _inspect_binary(binary: Path, cuobjdump: Path) -> dict[str, Any]:
    commands = {
        "elf": [str(cuobjdump), "--list-elf", str(binary)],
        "ptx": [str(cuobjdump), "--list-ptx", str(binary)],
        "sass": [str(cuobjdump), "--dump-sass", str(binary)],
        "ptx_text": [str(cuobjdump), "--dump-ptx", str(binary)],
    }
    outputs: dict[str, str] = {}
    returncodes: dict[str, int] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=120
        )
        outputs[name] = completed.stdout + completed.stderr
        returncodes[name] = completed.returncode
    symbols = ("mxfp4_matmul_pair_rows", "mxfp4_matmul_rows")
    return {
        "cuobjdump": str(cuobjdump.resolve()),
        "returncodes": returncodes,
        "sm86_cubin": "sm_86.cubin" in outputs["elf"],
        "compute86_ptx": "sm_86.ptx" in outputs["ptx"],
        "row_cooperative_symbols": {
            symbol: {
                "sm86_sass": symbol in outputs["sass"],
                "compute86_ptx": symbol in outputs["ptx_text"],
            }
            for symbol in symbols
        },
        "pass": (
            all(code == 0 for code in returncodes.values())
            and "sm_86.cubin" in outputs["elf"]
            and "sm_86.ptx" in outputs["ptx"]
            and all(
                symbol in outputs["sass"] and symbol in outputs["ptx_text"]
                for symbol in symbols
            )
        ),
    }


def _memory_stable(modes: dict[str, tuple[dict[str, Any], np.ndarray]]) -> bool:
    return all(
        int(record["memory"]["persistent_growth_bytes"])
        <= MEMORY_STABILITY_TOLERANCE_BYTES
        for record, _output in modes.values()
    )


def _compare(
    candidate: dict[str, tuple[dict[str, Any], np.ndarray]],
    deployed: dict[str, tuple[dict[str, Any], np.ndarray]],
) -> dict[str, Any]:
    candidate_serial = np.concatenate(
        (candidate["row0"][1], candidate["row1"][1]), axis=0
    )
    candidate_b1_sum_ms = sum(
        float(candidate[name][0]["device"]["p50_ms"])
        for name in ("row0", "row1")
    )
    deployed_b1_sum_ms = sum(
        float(deployed[name][0]["device"]["p50_ms"])
        for name in ("row0", "row1")
    )
    candidate_b2_ms = float(candidate["batch2"][0]["device"]["p50_ms"])
    deployed_b2_ms = float(deployed["batch2"][0]["device"]["p50_ms"])
    candidate_speedup = candidate_b1_sum_ms / candidate_b2_ms
    deployed_speedup = deployed_b1_sum_ms / deployed_b2_ms
    candidate_b1_mean = candidate_b1_sum_ms / 2.0
    deployed_b1_mean = deployed_b1_sum_ms / 2.0
    exact_by_mode = {
        name: bool(np.array_equal(candidate[name][1], deployed[name][1]))
        for name in candidate
    }
    serial_metrics = _numerical_metrics(candidate["batch2"][1], candidate_serial)
    return {
        "candidate_batch2_vs_candidate_serial": {
            **serial_metrics,
            "bit_exact": bool(
                np.array_equal(candidate["batch2"][1], candidate_serial)
            ),
        },
        "candidate_vs_deployed": {
            "bit_exact_by_mode": exact_by_mode,
            "all_bit_exact": all(exact_by_mode.values()),
        },
        "performance": {
            "candidate_serial_pair_device_p50_ms": candidate_b1_sum_ms,
            "candidate_batch2_device_p50_ms": candidate_b2_ms,
            "candidate_pair_throughput_speedup": candidate_speedup,
            "deployed_serial_pair_device_p50_ms": deployed_b1_sum_ms,
            "deployed_batch2_device_p50_ms": deployed_b2_ms,
            "deployed_pair_throughput_speedup": deployed_speedup,
            "candidate_over_deployed_batch2_ratio": candidate_b2_ms / deployed_b2_ms,
            "candidate_batch1_device_p50_mean_ms": candidate_b1_mean,
            "deployed_batch1_device_p50_mean_ms": deployed_b1_mean,
            "candidate_over_deployed_batch1_ratio": (
                candidate_b1_mean / deployed_b1_mean
            ),
            "minimum_preregistered_pair_throughput_speedup": (
                MINIMUM_PAIR_THROUGHPUT_SPEEDUP
            ),
            "maximum_preregistered_batch1_regression_ratio": (
                MAXIMUM_BATCH1_REGRESSION_RATIO
            ),
        },
        "gates": {
            "batch2_bit_exact_to_candidate_serial": bool(
                np.array_equal(candidate["batch2"][1], candidate_serial)
            ),
            "candidate_bit_exact_to_deployed": all(exact_by_mode.values()),
            "pair_throughput_at_least_1_50x": (
                candidate_speedup >= MINIMUM_PAIR_THROUGHPUT_SPEEDUP
            ),
            "batch1_regression_at_most_5_percent": (
                candidate_b1_mean / deployed_b1_mean
                <= MAXIMUM_BATCH1_REGRESSION_RATIO
            ),
        },
    }


def benchmark_row_cooperative_batch2(
    checkpoint: Path,
    candidate_library: Path,
    deployed_library: Path,
    cuda_source: Path,
    cuda_header: Path,
    cuobjdump: Path,
    output_path: Path,
    *,
    device: int = 0,
    layer: int = 89,
    expert_id: int = 803,
    warmup: int = 50,
    iterations: int = 300,
) -> dict[str, Any]:
    """Certify only batch 1 then batch 2; never launch an uncertified size."""
    if warmup < 20 or iterations < 100:
        raise ValueError("H014-028 requires >=20 warmups and >=100 retained calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "candidate_library": candidate_library.resolve(),
        "deployed_library": deployed_library.resolve(),
        "cuda_source": cuda_source.resolve(),
        "cuda_header": cuda_header.resolve(),
        "cuobjdump": cuobjdump.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": "H014-028a",
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "Sharing each real MXFP4 expert weight traversal across exactly two "
                "token rows yields >=1.50x pair throughput versus two batch-1 calls, "
                "with bit-exact output and <=5% batch-1 regression."
            ),
            "minimum_pair_throughput_speedup": MINIMUM_PAIR_THROUGHPUT_SPEEDUP,
            "maximum_batch1_regression_ratio": MAXIMUM_BATCH1_REGRESSION_RATIO,
        },
        "implementation": {
            "change": (
                "Two-row templated gate/up pair and down-projection kernels reuse "
                "one MXFP4 nibble/scale load across both rows."
            ),
            "batch1_path_changed": False,
            "native_certified_ceiling": 2,
            "larger_batches_executed": False,
            "deployment_binary_mutated": False,
        },
        "configuration": {
            "device": device,
            "layer": layer,
            "expert_id": expert_id,
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "tested_batches_in_order": [1, 2],
            "cpu_mathematical_fallbacks": 0,
            "timed_transfer_bytes": 0,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    candidate_runtime: _CudaRuntime | None = None
    deployed_runtime: _CudaRuntime | None = None
    try:
        receipt["artifact_inspection"] = _inspect_binary(
            paths["candidate_library"], paths["cuobjdump"]
        )
        retain(
            "candidate_sm86_and_compute86_inspection",
            passed=receipt["artifact_inspection"]["pass"],
        )
        if not receipt["artifact_inspection"]["pass"]:
            raise RuntimeError("candidate binary inspection failed")

        health_before = _health_snapshot(device)
        receipt["gpu_health_before"] = health_before
        receipt["device_identity"] = _device_identity(device)
        retain("gpu_health_before", status=health_before["status"])
        if health_before["status"] != "MEASURED":
            raise RuntimeError("nvidia-smi unavailable before candidate execution")

        real_expert = _load_real_expert(checkpoint, layer, expert_id)
        expert_source_bytes = sum(
            tensor.packed.nbytes + tensor.scales.nbytes
            for tensor in (real_expert.gate, real_expert.up, real_expert.down)
        )
        rng = np.random.default_rng(140280803)
        rows = np.ascontiguousarray(
            rng.normal(0.0, 0.25, size=(2, 3584)), dtype=np.float32
        )
        receipt["fixture"] = {
            "real_checkpoint_weights": True,
            "expert_source_shards": real_expert.source_shards,
            "expert_source_bytes": expert_source_bytes,
            "activation_shape": list(rows.shape),
            "activation_fingerprint": _array_fingerprint(rows),
        }

        candidate_runtime = _CudaRuntime(paths["candidate_library"], device)
        candidate_runtime.set_telemetry("minimal")
        if candidate_runtime.expert_max_certified_batch != 2:
            raise RuntimeError("candidate did not retain max_certified_batch=2")
        candidate_handles = _upload_expert(candidate_runtime, real_expert)
        candidate_modes: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for name, source in (("row0", rows[0:1]), ("row1", rows[1:2])):
            candidate_modes[name] = _run_mode(
                candidate_runtime,
                candidate_handles,
                source,
                warmup=warmup,
                iterations=iterations,
            )
            receipt["candidate"] = {
                "binary_sha256": candidate_runtime.sha256,
                "max_certified_batch": candidate_runtime.expert_max_certified_batch,
                "modes": _serializable_modes(candidate_modes),
            }
            retain(f"candidate_real_expert_batch1_{name}")

        candidate_modes["batch2"] = _run_mode(
            candidate_runtime,
            candidate_handles,
            rows,
            warmup=warmup,
            iterations=iterations,
        )
        receipt["candidate"]["modes"] = _serializable_modes(candidate_modes)
        candidate_runtime.synchronize()
        health_after_candidate = _health_snapshot(device)
        receipt["gpu_health_after_candidate_batch2"] = health_after_candidate
        retain(
            "candidate_real_expert_batch2_and_health",
            nvidia_smi=health_after_candidate["status"],
        )
        candidate_runtime.close()
        candidate_runtime = None

        deployed_runtime = _CudaRuntime(paths["deployed_library"], device)
        deployed_runtime.set_telemetry("minimal")
        deployed_handles = _upload_expert(deployed_runtime, real_expert)
        deployed_modes: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
        for name, source in (
            ("row0", rows[0:1]),
            ("row1", rows[1:2]),
            ("batch2", rows),
        ):
            deployed_modes[name] = _run_mode(
                deployed_runtime,
                deployed_handles,
                source,
                warmup=warmup,
                iterations=iterations,
            )
        receipt["deployed_control"] = {
            "binary_sha256": deployed_runtime.sha256,
            "max_certified_batch": deployed_runtime.expert_max_certified_batch,
            "modes": _serializable_modes(deployed_modes),
        }
        deployed_runtime.close()
        deployed_runtime = None
        retain("deployed_binary_batch1_batch2_control")

        comparison = _compare(candidate_modes, deployed_modes)
        receipt["comparison"] = comparison
        health_final = _health_snapshot(device)
        receipt["gpu_health_final"] = health_final
        uuids = {
            snapshot.get("uuid")
            for snapshot in (
                health_before,
                health_after_candidate,
                health_final,
            )
        }
        health_pass = (
            all(
                snapshot["status"] == "MEASURED"
                for snapshot in (
                    health_before,
                    health_after_candidate,
                    health_final,
                )
            )
            and len(uuids) == 1
        )
        gates = {
            **comparison["gates"],
            "candidate_memory_stable": _memory_stable(candidate_modes),
            "deployed_memory_stable": _memory_stable(deployed_modes),
            "gpu_health": health_pass,
            "native_ceiling_still_two": (
                receipt["candidate"]["max_certified_batch"] == 2
            ),
            "artifact_contract": receipt["artifact_inspection"]["pass"],
        }
        receipt["gates"] = gates
        benchmark_valid = all(
            gates[name]
            for name in (
                "batch2_bit_exact_to_candidate_serial",
                "candidate_bit_exact_to_deployed",
                "candidate_memory_stable",
                "deployed_memory_stable",
                "gpu_health",
                "native_ceiling_still_two",
                "artifact_contract",
            )
        )
        hypothesis_supported = benchmark_valid and all(comparison["gates"].values())
        receipt["status"] = "PASS" if benchmark_valid else "FAIL"
        receipt["hypothesis_result"] = (
            "SUPPORTED" if hypothesis_supported else "FALSIFIED"
        )
        receipt["hypothesis_supported"] = hypothesis_supported
        speedup = comparison["performance"][
            "candidate_pair_throughput_speedup"
        ]
        receipt["inspection"] = (
            f"The exact row-cooperative batch-2 path delivered {speedup:.6f}x "
            "pair throughput; numerical, batch-1, memory, artifact, and GPU-health "
            "gates were inspected separately."
        )
        receipt["bottleneck"] = (
            "Cross-row weight traversal was materially amortized."
            if hypothesis_supported
            else (
                "Explicit source-level weight reuse did not retain enough complete "
                "expert service, indicating occupancy, cache reuse, or extra per-row "
                "accumulation/reduction dominates the expected DRAM saving."
            )
        )
        receipt["decision"] = (
            "RETAIN_AND_BUILD_SEPARATE_BATCH4_CANDIDATE"
            if hypothesis_supported
            else "DO_NOT_PROMOTE_OR_EXPOSE_BATCH4"
        )
        receipt["redesign"] = (
            "Instantiate the same design for exactly batch 4 and certify it alone."
            if hypothesis_supported
            else (
                "Inspect kernel resource use and CUDA activity before selecting a "
                "different cross-row reuse design; deployed max remains 2."
            )
        )
        receipt["effective_weight_reuse"] = {
            "architectural_rows_per_weight_traversal": 2,
            "deployed_rows_per_weight_traversal": 1,
            "measured_dram_bytes": None,
            "qualification": (
                "source-level traversal reuse; hardware DRAM traffic requires a "
                "profiler counter and is not inferred from source alone"
            ),
        }
        retain("final_h014_028a_inspection", hypothesis=receipt["hypothesis_result"])
    except Exception as exc:
        receipt["status"] = "FAIL"
        receipt["hypothesis_result"] = "INVALID_BENCHMARK"
        receipt["hypothesis_supported"] = False
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        receipt["decision"] = "KEEP_CANDIDATE_QUARANTINED"
        receipt["redesign"] = "Inspect this retained failure before further GPU work."
        with suppress(Exception):
            receipt["gpu_health_on_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
    finally:
        for runtime in (deployed_runtime, candidate_runtime):
            if runtime is not None:
                with suppress(Exception):
                    runtime.close()

    result = dict(receipt)
    result["output_path"] = str(output_path.resolve())
    result["output_sha256"] = _sha256_file(output_path)
    return result
