"""Run local RTX 5090 resident evidence arms for Experiment 022."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_019.attention import benchmark_attention_stripes
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.physical import (
    ROUTED_EXPERTS,
    _upload_stripe_experts,
    real_route_workload,
    striped_latent_down,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
    _execute_grouped,
)
from swarm_inference.experiments.experiment_022.resident_replay import replay_resident_layer


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def attention(arguments: argparse.Namespace) -> dict[str, object]:
    values = {}
    for split, kda, mla in (("calibration", 45, 47), ("heldout", 89, 91)):
        print(f"[e022 resident attention] {split} KDA={kda} MLA={mla}", flush=True)
        values[split] = benchmark_attention_stripes(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.shard_library,
            kda_layer=kda,
            mla_layer=mla,
            degrees=(2, 4, 8, 16),
            rows_sweep=(1, 2, 4),
            warmup=arguments.warmup,
            iterations=arguments.iterations,
        )
    status = all(value["status"] == "PASS" for value in values.values())
    return {
        "schema_version": "experiment-022-resident-attention-v1",
        "status": "PASS" if status else "FAIL",
        "evidence_class": "PHYSICAL RTX 5090 all attention partition shards resident",
        "calibration": values["calibration"],
        "heldout": values["heldout"],
        "timed_checkpoint_reads": 0,
        "timed_weight_uploads": 0,
        "timed_shard_creation": 0,
        "startup_separate": True,
    }


def ordered(arguments: argparse.Namespace) -> dict[str, object]:
    rows = []
    for layer in (45, 47, 89, 91):
        print(f"[e022 ordered resident] layer={layer} degree=8", flush=True)
        value = replay_resident_layer(
            arguments.checkpoint,
            arguments.cuda_library,
            arguments.shard_library,
            arguments.grouped_library,
            arguments.oracle_root,
            layer=layer,
            degree=8,
            warmup=max(1, arguments.warmup),
            iterations=arguments.iterations,
        )
        value["split"] = "calibration" if layer in (45, 47) else "heldout"
        rows.append(value)
        _write(arguments.output, {
            "schema_version": "experiment-022-resident-ordered-suite-v1",
            "status": "RUNNING",
            "results": rows,
        })
    return {
        "schema_version": "experiment-022-resident-ordered-suite-v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "results": rows,
    }


def expert_bank(arguments: argparse.Namespace) -> dict[str, object]:
    catalog = CheckpointCatalog(arguments.checkpoint)
    loader = DirectShardLoader(catalog)
    runtime = _CudaRuntime(arguments.cuda_library, 0)
    runtime.set_telemetry("minimal")
    runtime.set_fused_gate_up(True)
    quantizer = GpuShardQuantizer(arguments.shard_library, 0)
    grouped = GroupedTop16Runtime(arguments.grouped_library)
    results: list[dict[str, object]] = []
    try:
        hidden, routes, weights, workload = real_route_workload(
            arguments.checkpoint,
            arguments.oracle_root / "hidden-trace.f32",
            arguments.oracle_root / "routes.txt",
            layer=89,
            rows=4,
            loader=loader,
        )
        latent, projection = striped_latent_down(
            runtime,
            loader,
            hidden,
            layer=89,
            degree=16,
            warmup=arguments.warmup,
            iterations=arguments.iterations,
            quantizer=quantizer,
        )
        reference: dict[int, np.ndarray] = {}
        for degree in (2, 4, 8, 16):
            print(f"[e022 resident expert bank] degree={degree}", flush=True)
            before = runtime.mem_info()
            started = time.perf_counter_ns()
            residents = []
            for stripe in range(degree):
                residents.append(
                    _upload_stripe_experts(
                        runtime,
                        loader,
                        layer=89,
                        experts=list(range(ROUTED_EXPERTS)),
                        degree=degree,
                        stripe=stripe,
                        worker_id=f"e022.full-bank.p{degree}.worker-{stripe:02d}",
                    )
                )
                print(
                    f"[e022 resident expert bank] degree={degree} loaded={stripe + 1}/{degree}",
                    flush=True,
                )
            startup_ms = (time.perf_counter_ns() - started) / 1e6
            after = runtime.mem_info()
            try:
                for row_count in (1, 2, 4):
                    audit_before = len(loader.audit)
                    partials = []
                    workers = []
                    for stripe, resident in enumerate(residents):
                        measured, partial = _execute_grouped(
                            runtime,
                            grouped,
                            resident,
                            latent[:row_count],
                            routes[:row_count],
                            weights[:row_count],
                            warmup=arguments.warmup,
                            iterations=arguments.iterations,
                        )
                        partials.append(partial)
                        workers.append(
                            {
                                "stripe": stripe,
                                "runtime_weight_bytes": resident.runtime_bytes,
                                "wall": measured["wall"],
                                "cuda": measured["cuda"],
                                "native_physical_launches": measured["total_physical_launches"],
                            }
                        )
                    output = np.sum(np.stack(partials), axis=0, dtype=np.float64).astype(np.float32)
                    if row_count not in reference:
                        reference[row_count] = output
                    metrics = _numerical_metrics(reference[row_count], output)
                    results.append(
                        {
                            "degree": degree,
                            "rows": row_count,
                            "layer": 89,
                            "resident_expert_count_per_worker": ROUTED_EXPERTS,
                            "all_partition_workers_simultaneously_resident": True,
                            "arbitrary_route_ready": all(len(value.handles) == ROUTED_EXPERTS for value in residents),
                            "runtime_weight_bytes": sum(value.runtime_bytes for value in residents),
                            "measured_free_memory_delta_bytes": before["free_bytes"] - after["free_bytes"],
                            "startup_ms": startup_ms,
                            "startup_excluded_from_service": True,
                            "timed_checkpoint_reads": len(loader.audit) - audit_before,
                            "workers": workers,
                            "independent_worker_compute_ceiling_ms": max(float(value["wall"]["p50_ms"]) for value in workers),
                            "cross_degree_metrics": metrics,
                            "pass": float(metrics["relative_l2_error"]) <= 2e-6,
                        }
                    )
            finally:
                for resident in reversed(residents):
                    resident.close()
            _write(arguments.output, {
                "schema_version": "experiment-022-resident-expert-bank-v1",
                "status": "RUNNING",
                "workload": workload,
                "latent_projection": projection,
                "results": results,
            })
        return {
            "schema_version": "experiment-022-resident-expert-bank-v1",
            "status": "PASS" if all(bool(row["pass"]) for row in results) else "FAIL",
            "evidence_class": "PHYSICAL RTX 5090 complete arbitrary-route expert banks resident",
            "workload": workload,
            "latent_projection": projection,
            "results": results,
            "checkpoint_read_audit": loader.audit,
            "startup_quantization_audit": quantizer.audit,
        }
    finally:
        grouped.close()
        runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("attention", "ordered", "expert-bank"))
    parser.add_argument("--checkpoint", type=Path, default=Path("F:/models/Kimi-K3"))
    parser.add_argument("--cuda-library", type=Path, default=Path("artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll"))
    parser.add_argument("--shard-library", type=Path, default=Path("artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll"))
    parser.add_argument("--grouped-library", type=Path, default=Path("artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll"))
    parser.add_argument("--oracle-root", type=Path, default=Path("artifacts/experiment-014/oracle-full-93-idot0"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.checkpoint = arguments.checkpoint.resolve()
    arguments.cuda_library = arguments.cuda_library.resolve()
    arguments.shard_library = arguments.shard_library.resolve()
    arguments.grouped_library = arguments.grouped_library.resolve()
    arguments.oracle_root = arguments.oracle_root.resolve()
    receipt = globals()[arguments.arm.replace("-", "_")](arguments)
    _write(arguments.output, receipt)
    print(json.dumps({"arm": arguments.arm, "status": receipt["status"]}), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
