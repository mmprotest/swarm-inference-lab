"""Static and binary evidence for RTX 3090 (Ampere ``sm_86``) compatibility."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-rtx3090-compatibility-v1"
CUDA_OPERATION_SCHEMA_VERSION = "experiment-014-k3-cuda-operation-matrix-v1"

_FULL_EVIDENCE_SCHEMAS = {
    "experiment-014-k3-cuda-real-embedding-v1": "embedding",
    "experiment-014-k3-cuda-real-dense-v1": "dense_projection",
    "experiment-014-k3-cuda-real-kda-stage-v1": "KDA",
    "experiment-014-k3-cuda-real-mla-stage-v1": "Gated_MLA",
    "experiment-014-k3-cuda-real-attnres-v1": "attention_residual",
    "experiment-014-k3-cuda-router-v1": "router",
    "experiment-014-k3-cuda-real-expert-v1": "MXFP4_routed_expert",
    "experiment-014-k3-cuda-real-shared-expert-v1": "shared_experts",
    "experiment-014-k3-cuda-real-moe-reduction-v1": "MoE_reduction",
    "experiment-014-k3-cuda-real-final-norm-v1": "final_norm",
    "experiment-014-k3-cuda-real-lm-head-v1": "LM_head",
}

_FULL_OPERATION_SYMBOLS = {
    "embedding": ("kimi_embedding_bf16",),
    "dense_projection": ("quant_matmul",),
    "KDA": ("quant_matmul", "kimi_kda_core"),
    "Gated_MLA": (
        "quant_matmul",
        "pipe_rmsnorm_rows",
        "attention_absorb_kernel",
        "kimi_mla_sigmoid_gate",
    ),
    "attention_residual": ("kimi_attnres_mix", "pipe_rmsnorm_rows"),
    "router": ("pipe_router_logits", "pipe_router_select"),
    "MXFP4_routed_expert": ("mxfp4_matmul_pair", "kimi_situ_mul"),
    "shared_experts": ("grouped_hidden_g4_dual", "grouped_down_g4"),
    "MoE_reduction": ("weighted_sum_rows",),
    "final_norm": ("pipe_rmsnorm_rows",),
    "LM_head": ("quant_matmul",),
}

_FULL_OPERATION_PRIMITIVES = {
    "embedding": "resident BF16 embedding gather",
    "dense_projection": "resident grouped-int4 quant_matmul",
    "KDA": "generic resident projections plus persistent Kimi KDA state core",
    "Gated_MLA": "generic resident projections/norm/absorb attention plus cache and sigmoid adapters",
    "attention_residual": "resident FP32 Kimi AttnRes mix plus generic RMSNorm",
    "router": "resident FP32 router GEMV plus deterministic parallel top-16",
    "MXFP4_routed_expert": "native fmt=7 resident MXFP4 plus fused SiTU expert",
    "shared_experts": "resident grouped-int4 SiTU shared-expert MLP",
    "MoE_reduction": "resident fixed-order FP32 weighted_sum_rows",
    "final_norm": "resident FP32 RMSNorm",
    "LM_head": "resident per-row-int8 vocabulary projection",
}


class CompatibilityError(ValueError):
    """Compatibility evidence is missing or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cuda_operation_matrix(
    kimi_source: Path,
    cuda_source: Path,
    makefile: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Audit primitive reuse before changing the Kimi execution path.

    This deliberately does not infer CUDA readiness from the presence of a
    similarly named primitive.  A class remains a blocker until real Kimi
    correctness and backend-identity evidence is attached in a later cycle.
    """

    kimi_path = kimi_source.expanduser().resolve()
    cuda_path = cuda_source.expanduser().resolve()
    make_path = makefile.expanduser().resolve()
    for path in (kimi_path, cuda_path, make_path):
        if not path.is_file():
            raise CompatibilityError(f"missing CUDA operation source: {path}")
    kimi = kimi_path.read_text(encoding="utf-8", errors="replace")
    cuda = cuda_path.read_text(encoding="utf-8", errors="replace")
    make = make_path.read_text(encoding="utf-8", errors="replace")
    candidates = (
        (
            "embedding",
            "st_read_slice_f32 token-row load",
            None,
            "add a device gather/upload path and bind embedding weights",
            True,
            "custom Kimi CUDA",
        ),
        (
            "dense_projection",
            "w_matmul (f32/int8/grouped-int4)",
            "coli_cuda_matmul / coli_cuda_pipe_gemm",
            "store a ColiCudaTensor handle in W and dispatch without CPU fallback",
            False,
            "generic existing CUDA",
        ),
        (
            "KDA",
            "kda_forward recurrent state and causal short convolution",
            "pipe GEMM and RMSNorm cover projections/norm only",
            "retain generic GEMMs; add GPU short-convolution and recurrent state update",
            True,
            "adapted CUDA",
        ),
        (
            "Gated_MLA",
            "mla_forward causal latent-cache attention plus output gate",
            "coli_cuda_attention_absorb_batch plus pipe GEMM/RMSNorm",
            "adapt cache/state layout and add the sigmoid output-gate multiply",
            True,
            "adapted CUDA",
        ),
        (
            "attention_residual",
            "res_mix RMS-normalized softmax residual mixing",
            "coli_cuda_pipe_rmsnorm and coli_cuda_pipe_add are partial building blocks",
            "add fixed-order score softmax and weighted residual mix",
            True,
            "adapted CUDA",
        ),
        (
            "router",
            "896-way sigmoid+bias top-16 selection in moe_forward",
            "coli_cuda_pipe_router",
            "upload real router weights/bias and bind exact Kimi routing parameters",
            False,
            "generic existing CUDA",
        ),
        (
            "MXFP4_routed_expert",
            "matmul_mxfp4 E2M1/UE8M0 group-32 plus SiTU-GLU",
            None,
            "add native MXFP4 tensor format and Kimi expert activation to the resident expert path",
            True,
            "custom Kimi CUDA",
        ),
        (
            "shared_experts",
            "three w_matmul projections plus SiTU-GLU",
            "coli_cuda_matmul / coli_cuda_pipe_gemm",
            "reuse GEMMs and add a SiTU-GLU device epilogue",
            True,
            "adapted CUDA",
        ),
        (
            "MoE_reduction",
            "fixed-order FP32 weighted expert accumulation",
            "weighted_sum_rows and sum_slots kernels inside the resident expert path",
            "expose/bind the existing fixed-order reduction independently of the GLM expert path",
            False,
            "adapted CUDA",
        ),
        (
            "final_norm",
            "rmsnorm_",
            "coli_cuda_pipe_rmsnorm",
            "upload the real final-norm weight and dispatch the existing primitive",
            False,
            "generic existing CUDA",
        ),
        (
            "LM_head",
            "w_matmul vocabulary projection",
            "coli_cuda_matmul / coli_cuda_pipe_gemm",
            "bind the real LM-head tensor and retain output on GPU until sampling transfer",
            False,
            "generic existing CUDA",
        ),
    )
    operations = [
        {
            "kimi_operation": operation,
            "existing_implementation": implementation,
            "existing_cuda_primitive": primitive,
            "wiring_needed": wiring,
            "new_kernel_needed": new_kernel,
            "implementation_kind": kind,
            "sm86_status": "BLOCKER",
            "readiness": "NOT_CUDA_READY",
            "evidence_requirement": (
                "real Kimi weights/state, recorded CUDA backend identity, numerical comparison, "
                "and an inspected sm_86 binary"
            ),
        }
        for operation, implementation, primitive, wiring, new_kernel, kind in candidates
    ]
    wiring_only = sum(not row["new_kernel_needed"] for row in operations)
    declared_primitives = {
        symbol: symbol in cuda
        for symbol in (
            "coli_cuda_matmul",
            "coli_cuda_pipe_gemm",
            "coli_cuda_pipe_rmsnorm",
            "coli_cuda_pipe_router",
            "coli_cuda_attention_absorb_batch",
            "weighted_sum_rows",
            "sum_slots",
        )
    }
    payload = {
        "schema_version": CUDA_OPERATION_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL",
        "hypothesis": {
            "id": "H014-025a",
            "prediction": (
                "at least 6 of 11 Kimi-critical operation classes require wiring only and no "
                "new mathematical CUDA kernel"
            ),
            "result": "FALSIFIED" if wiring_only < 6 else "SUPPORTED",
        },
        "source_evidence": {
            "kimi_source": str(kimi_path),
            "kimi_source_sha256": _sha256(kimi_path),
            "cuda_source": str(cuda_path),
            "cuda_source_sha256": _sha256(cuda_path),
            "makefile": str(make_path),
            "makefile_sha256": _sha256(make_path),
            "kimi_has_cuda_dispatch": "COLI_CUDA" in kimi,
            "kimi_make_target_links_cuda_object": any(
                "kimi_k3$(EXE):" in line and "CUDA_OBJ" in line for line in make.splitlines()
            ),
            "declared_primitives": declared_primitives,
        },
        "operations": operations,
        "summary": {
            "critical_operation_classes": len(operations),
            "cuda_ready": 0,
            "wiring_only_candidates": wiring_only,
            "new_or_adapted_kernel_candidates": len(operations) - wiring_only,
            "hypothesis_minimum_wiring_only": 6,
            "blockers": len(operations),
            "actual_bottleneck": (
                "the generic backend covers dense GEMM, router, norm, attention core, and internal "
                "reduction pieces, but lacks Kimi MXFP4/SiTU, KDA state math, embedding gather, "
                "Gated-MLA gating, and AttnRes mixing"
            ),
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": CUDA_OPERATION_SCHEMA_VERSION,
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "summary": payload["summary"],
    }


def certify_sm86_routed_expert(
    operation_matrix_path: Path,
    benchmark_path: Path,
    binary_path: Path,
    cuobjdump_path: Path,
    certification_path: Path,
    router_benchmark_path: Path | None = None,
    dense_benchmark_paths: list[Path] | None = None,
    final_norm_benchmark_path: Path | None = None,
) -> dict[str, Any]:
    """Promote only operations backed by real fixtures in the exact sm_86 binary."""

    matrix_source = operation_matrix_path.expanduser().resolve()
    benchmark_source = benchmark_path.expanduser().resolve()
    binary_source = binary_path.expanduser().resolve()
    cuobjdump = cuobjdump_path.expanduser().resolve()
    router_source = (
        router_benchmark_path.expanduser().resolve()
        if router_benchmark_path is not None
        else None
    )
    dense_sources = [path.expanduser().resolve() for path in dense_benchmark_paths or []]
    final_norm_source = (
        final_norm_benchmark_path.expanduser().resolve()
        if final_norm_benchmark_path is not None
        else None
    )
    inputs = [matrix_source, benchmark_source, binary_source, cuobjdump]
    if router_source is not None:
        inputs.append(router_source)
    inputs.extend(dense_sources)
    if final_norm_source is not None:
        inputs.append(final_norm_source)
    for path in inputs:
        if not path.is_file():
            raise CompatibilityError(f"missing sm_86 certification input: {path}")
    matrix = json.loads(matrix_source.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_source.read_text(encoding="utf-8"))
    if benchmark.get("status") != "PASS":
        raise CompatibilityError("real routed-expert benchmark did not pass")
    backend = benchmark.get("backend", {})
    correctness = benchmark.get("correctness", {})
    negotiation = backend.get("capability_negotiation", {})
    expected_hash = str(backend.get("cuda_library_sha256", ""))
    actual_hash = _sha256(binary_source)
    if expected_hash != actual_hash:
        raise CompatibilityError("benchmark binary hash does not match certification binary")
    if (
        int(backend.get("binary_min_compute_capability", -1)) != 86
        or backend.get("binary_has_forward_ptx") is not True
        or negotiation.get("sm_75") is not False
        or negotiation.get("sm_86") is not True
        or float(correctness.get("relative_l2_error", float("inf"))) > 3e-4
        or float(correctness.get("cosine_similarity", 0.0)) < 0.999999
    ):
        raise CompatibilityError("routed-expert benchmark lacks required sm_86 invariants")

    def inspect(*arguments: str) -> str:
        completed = subprocess.run(
            [str(cuobjdump), *arguments, str(binary_source)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            raise CompatibilityError(
                f"cuobjdump {' '.join(arguments)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout

    elf_listing = inspect("--list-elf")
    ptx = inspect("--dump-ptx")
    required_ptx_entries = ["quant_matmul", "mxfp4_matmul_pair", "kimi_situ_mul"]
    if router_source is not None:
        required_ptx_entries.extend(("pipe_router_logits", "pipe_router_select"))
    if final_norm_source is not None:
        required_ptx_entries.append("pipe_rmsnorm_rows")
    artifact_contract = {
        "sm86_cubin": "sm_86.cubin" in elf_listing,
        "compute86_ptx": ".target sm_86" in ptx,
        "required_ptx_entries": {
            name: name in ptx for name in required_ptx_entries
        },
    }
    if not (
        artifact_contract["sm86_cubin"]
        and artifact_contract["compute86_ptx"]
        and all(artifact_contract["required_ptx_entries"].values())
    ):
        raise CompatibilityError("sm_86 binary lacks required cubin/PTX coverage")

    target = None
    for row in matrix.get("operations", []):
        if row.get("kimi_operation") == "MXFP4_routed_expert":
            target = row
            break
    if target is None:
        raise CompatibilityError("operation matrix has no MXFP4 routed-expert row")
    target.update(
        {
            "existing_cuda_primitive": "native fmt=7 resident MXFP4 plus fused SiTU expert",
            "wiring_needed": "bind certified device-resident expert ABI into Kimi stage lifecycle",
            "new_kernel_needed": False,
            "sm86_status": "CERTIFIED",
            "readiness": "CUDA_READY",
            "certification_evidence": {
                "benchmark": str(benchmark_source),
                "benchmark_sha256": _sha256(benchmark_source),
                "binary": str(binary_source),
                "binary_sha256": actual_hash,
                "relative_l2_error": correctness["relative_l2_error"],
                "cosine_similarity": correctness["cosine_similarity"],
                "resident_synchronized_p50_ms_on_rtx5090": benchmark["benchmark"][
                    "resident"
                ]["synchronized_generation"]["p50_ms"],
                "artifact_contract": artifact_contract,
                "physical_sm86_execution": False,
                "physical_validation": "Experiment 015 single-node canary",
            },
        }
    )
    if router_source is not None:
        router_benchmark = json.loads(router_source.read_text(encoding="utf-8"))
        router_backend = router_benchmark.get("backend", {})
        router_correctness = router_benchmark.get("correctness", {})
        router_metrics = router_correctness.get("selected_weight_metrics", {})
        router_negotiation = router_backend.get("capability_negotiation", {})
        if (
            router_benchmark.get("status") != "PASS"
            or str(router_backend.get("cuda_library_sha256", "")) != actual_hash
            or int(router_backend.get("binary_min_compute_capability", -1)) != 86
            or router_backend.get("binary_has_forward_ptx") is not True
            or router_negotiation.get("sm_75") is not False
            or router_negotiation.get("sm_86") is not True
            or router_correctness.get("routing_equality") is not True
            or int(router_correctness.get("cuda_keff", -1)) != 16
            or float(router_metrics.get("relative_l2_error", float("inf"))) > 1e-6
        ):
            raise CompatibilityError("router benchmark lacks required sm_86 invariants")
        router_target = next(
            (
                row
                for row in matrix.get("operations", [])
                if row.get("kimi_operation") == "router"
            ),
            None,
        )
        if router_target is None:
            raise CompatibilityError("operation matrix has no router row")
        router_target.update(
            {
                "existing_cuda_primitive": (
                    "resident FP32 router GEMV plus deterministic parallel top-16"
                ),
                "wiring_needed": (
                    "bind certified device-resident router ABI into Kimi stage lifecycle"
                ),
                "new_kernel_needed": False,
                "implementation_kind": "adapted CUDA",
                "sm86_status": "CERTIFIED",
                "readiness": "CUDA_READY",
                "certification_evidence": {
                    "benchmark": str(router_source),
                    "benchmark_sha256": _sha256(router_source),
                    "binary": str(binary_source),
                    "binary_sha256": actual_hash,
                    "routing_equality": True,
                    "selected_weight_relative_l2_error": router_metrics[
                        "relative_l2_error"
                    ],
                    "warm_p50_ms_on_rtx5090": router_benchmark["benchmark"]["wall"][
                        "p50_ms"
                    ],
                    "artifact_contract": {
                        "sm86_cubin": artifact_contract["sm86_cubin"],
                        "compute86_ptx": artifact_contract["compute86_ptx"],
                        "required_ptx_entries": {
                            name: artifact_contract["required_ptx_entries"][name]
                            for name in ("pipe_router_logits", "pipe_router_select")
                        },
                    },
                    "physical_sm86_execution": False,
                    "physical_validation": "Experiment 015 single-node canary",
                },
            }
        )
    if dense_sources:
        dense_evidence = []
        for dense_source in dense_sources:
            dense_benchmark = json.loads(dense_source.read_text(encoding="utf-8"))
            dense_backend = dense_benchmark.get("backend", {})
            dense_correctness = dense_benchmark.get("correctness", {}).get(
                "cuda_vs_production_quantized_oracle", {}
            )
            dense_negotiation = dense_backend.get("capability_negotiation", {})
            if (
                dense_benchmark.get("status") != "PASS"
                or str(dense_backend.get("cuda_library_sha256", "")) != actual_hash
                or int(dense_backend.get("binary_min_compute_capability", -1)) != 86
                or dense_backend.get("binary_has_forward_ptx") is not True
                or dense_negotiation.get("sm_75") is not False
                or dense_negotiation.get("sm_86") is not True
                or float(dense_correctness.get("relative_l2_error", float("inf")))
                > 1e-5
                or float(dense_correctness.get("cosine_similarity", 0.0)) < 0.999999
            ):
                raise CompatibilityError(
                    f"dense benchmark lacks required sm_86 invariants: {dense_source}"
                )
            dense_evidence.append(
                {
                    "benchmark": str(dense_source),
                    "benchmark_sha256": _sha256(dense_source),
                    "dimensions": dense_benchmark["fixture"]["dimensions"],
                    "relative_l2_error": dense_correctness["relative_l2_error"],
                    "cosine_similarity": dense_correctness["cosine_similarity"],
                    "warm_p50_ms_on_rtx5090": dense_benchmark["benchmark"]["wall"][
                        "p50_ms"
                    ],
                    "kernel_ms_on_rtx5090": dense_benchmark["benchmark"][
                        "detailed_profile"
                    ]["kernel_ms_per_call"],
                }
            )
        dense_target = next(
            (
                row
                for row in matrix.get("operations", [])
                if row.get("kimi_operation") == "dense_projection"
            ),
            None,
        )
        if dense_target is None:
            raise CompatibilityError("operation matrix has no dense-projection row")
        dense_target.update(
            {
                "existing_cuda_primitive": "resident grouped-int4 quant_matmul",
                "wiring_needed": (
                    "bind certified device-resident GEMV into Kimi stage lifecycle"
                ),
                "new_kernel_needed": False,
                "implementation_kind": "generic existing CUDA",
                "sm86_status": "CERTIFIED",
                "readiness": "CUDA_READY",
                "certification_evidence": {
                    "benchmarks": dense_evidence,
                    "binary": str(binary_source),
                    "binary_sha256": actual_hash,
                    "artifact_contract": {
                        "sm86_cubin": artifact_contract["sm86_cubin"],
                        "compute86_ptx": artifact_contract["compute86_ptx"],
                        "required_ptx_entries": {
                            "quant_matmul": artifact_contract["required_ptx_entries"][
                                "quant_matmul"
                            ]
                        },
                    },
                    "physical_sm86_execution": False,
                    "physical_validation": "Experiment 015 single-node canary",
                },
            }
        )
    if final_norm_source is not None:
        norm_benchmark = json.loads(final_norm_source.read_text(encoding="utf-8"))
        norm_backend = norm_benchmark.get("backend", {})
        norm_correctness = norm_benchmark.get("correctness", {})
        norm_negotiation = norm_backend.get("capability_negotiation", {})
        if (
            norm_benchmark.get("status") != "PASS"
            or str(norm_backend.get("cuda_library_sha256", "")) != actual_hash
            or int(norm_backend.get("binary_min_compute_capability", -1)) != 86
            or norm_backend.get("binary_has_forward_ptx") is not True
            or norm_negotiation.get("sm_75") is not False
            or norm_negotiation.get("sm_86") is not True
            or float(norm_correctness.get("relative_l2_error", float("inf"))) > 2e-6
            or float(norm_correctness.get("cosine_similarity", 0.0)) < 0.999999
        ):
            raise CompatibilityError("final-norm benchmark lacks required sm_86 invariants")
        norm_target = next(
            (
                row
                for row in matrix.get("operations", [])
                if row.get("kimi_operation") == "final_norm"
            ),
            None,
        )
        if norm_target is None:
            raise CompatibilityError("operation matrix has no final-norm row")
        norm_target.update(
            {
                "existing_cuda_primitive": "resident FP32 RMSNorm",
                "wiring_needed": (
                    "bind certified resident RMSNorm into final Kimi stage lifecycle"
                ),
                "new_kernel_needed": False,
                "implementation_kind": "generic existing CUDA",
                "sm86_status": "CERTIFIED",
                "readiness": "CUDA_READY",
                "certification_evidence": {
                    "benchmark": str(final_norm_source),
                    "benchmark_sha256": _sha256(final_norm_source),
                    "binary": str(binary_source),
                    "binary_sha256": actual_hash,
                    "relative_l2_error": norm_correctness["relative_l2_error"],
                    "cosine_similarity": norm_correctness["cosine_similarity"],
                    "warm_p50_ms_on_rtx5090": norm_benchmark["benchmark"]["wall"][
                        "p50_ms"
                    ],
                    "batched_device_service_ms_on_rtx5090": norm_benchmark[
                        "benchmark"
                    ]["batched_event_service_ms_per_call"],
                    "artifact_contract": {
                        "sm86_cubin": artifact_contract["sm86_cubin"],
                        "compute86_ptx": artifact_contract["compute86_ptx"],
                        "required_ptx_entries": {
                            "pipe_rmsnorm_rows": artifact_contract[
                                "required_ptx_entries"
                            ]["pipe_rmsnorm_rows"]
                        },
                    },
                    "physical_sm86_execution": False,
                    "physical_validation": "Experiment 015 single-node canary",
                },
            }
        )
    ready = sum(row.get("readiness") == "CUDA_READY" for row in matrix["operations"])
    matrix["status"] = "PASS" if ready == len(matrix["operations"]) else "FAIL"
    matrix["summary"]["cuda_ready"] = ready
    matrix["summary"]["blockers"] = len(matrix["operations"]) - ready
    temporary_matrix = matrix_source.with_suffix(matrix_source.suffix + ".partial")
    temporary_matrix.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_matrix.replace(matrix_source)

    certification_operations = []
    for row in matrix["operations"]:
        certified = (
            row.get("readiness") == "CUDA_READY"
            and row.get("sm86_status") in {"CERTIFIED", "FALLBACK_CERTIFIED"}
        )
        certification_operations.append(
            {
                "kimi_operation": row["kimi_operation"],
                "status": "CERTIFIED" if certified else "BLOCKER",
                "implementation_kind": row["implementation_kind"],
                "evidence": row.get("certification_evidence") if certified else None,
                "blocker": None if certified else row["wiring_needed"],
            }
        )
    certification = {
        "schema_version": "experiment-014-rtx3090-sm86-certification-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if ready == len(matrix["operations"]) else "FAIL",
        "target": {
            "gpu_class": "RTX 3090",
            "compute_capability": "sm_86",
            "binary": str(binary_source),
            "binary_sha256": actual_hash,
            "binary_bytes": binary_source.stat().st_size,
        },
        "summary": {
            "critical_operation_classes": len(certification_operations),
            "certified": ready,
            "fallback_certified": 0,
            "blockers": len(certification_operations) - ready,
            "physical_sm86_execution_deferred_to_canary": True,
        },
        "operations": certification_operations,
    }
    destination = certification_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(certification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": certification["schema_version"],
        "status": certification["status"],
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "summary": certification["summary"],
    }


def certify_sm86_complete_kimi(
    operation_matrix_path: Path,
    benchmark_paths: list[Path],
    binary_path: Path,
    cuobjdump_path: Path,
    qualification_path: Path,
    certification_path: Path,
    *,
    cycle_id: str = "H014-025ae",
) -> dict[str, Any]:
    """Fail closed unless all 11 Kimi classes pass on one inspected sm_86 binary."""

    matrix_source = operation_matrix_path.expanduser().resolve()
    binary_source = binary_path.expanduser().resolve()
    cuobjdump = cuobjdump_path.expanduser().resolve()
    evidence_sources = [path.expanduser().resolve() for path in benchmark_paths]
    for path in (matrix_source, binary_source, cuobjdump, *evidence_sources):
        if not path.is_file():
            raise CompatibilityError(f"missing complete sm_86 certification input: {path}")
    actual_hash = _sha256(binary_source)
    matrix = json.loads(matrix_source.read_text(encoding="utf-8"))
    matrix_rows = {
        str(row.get("kimi_operation")): row for row in matrix.get("operations", [])
    }
    required_operations = set(_FULL_OPERATION_SYMBOLS)
    if set(matrix_rows) != required_operations:
        raise CompatibilityError("operation matrix does not contain exactly the 11 Kimi classes")

    evidence: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        operation: [] for operation in required_operations
    }
    for source in evidence_sources:
        document = json.loads(source.read_text(encoding="utf-8"))
        schema = str(document.get("schema_version", ""))
        operation = _FULL_EVIDENCE_SCHEMAS.get(schema)
        if operation is None:
            raise CompatibilityError(f"unrecognized CUDA evidence schema {schema!r}: {source}")
        backend = document.get("backend", {})
        negotiation = backend.get("capability_negotiation", {})
        if (
            document.get("status") != "PASS"
            or str(backend.get("cuda_library_sha256", "")) != actual_hash
            or not str(backend.get("identity", "")).startswith("nvidia_cuda")
            or backend.get("cpu_fallback_allowed") is not False
            or int(backend.get("binary_min_compute_capability", -1)) != 86
            or backend.get("binary_has_forward_ptx") is not True
            or negotiation.get("sm_75") is not False
            or negotiation.get("sm_86") is not True
        ):
            raise CompatibilityError(
                f"{operation} evidence is not a passing no-fallback fixture on {actual_hash}: {source}"
            )
        evidence[operation].append((source, document))
    missing = [operation for operation, rows in evidence.items() if not rows]
    if missing:
        raise CompatibilityError(f"missing operation-class evidence: {missing}")
    if len(evidence["dense_projection"]) < 2:
        raise CompatibilityError("dense projection certification requires small and large fixtures")
    duplicates = {
        operation: len(rows)
        for operation, rows in evidence.items()
        if operation != "dense_projection" and len(rows) != 1
    }
    if duplicates:
        raise CompatibilityError(f"ambiguous operation evidence counts: {duplicates}")

    def require(condition: bool, operation: str, message: str) -> None:
        if not condition:
            raise CompatibilityError(f"{operation} correctness invariant failed: {message}")

    numerical: dict[str, Any] = {}
    expert = evidence["MXFP4_routed_expert"][0][1]["correctness"]
    require(float(expert["relative_l2_error"]) <= 3e-4, "MXFP4_routed_expert", "relative L2")
    require(float(expert["cosine_similarity"]) >= 0.999999, "MXFP4_routed_expert", "cosine")
    numerical["MXFP4_routed_expert"] = {
        "relative_l2_error": expert["relative_l2_error"],
        "cosine_similarity": expert["cosine_similarity"],
    }

    router = evidence["router"][0][1]["correctness"]
    require(router.get("routing_equality") is True, "router", "selected IDs")
    require(int(router.get("cuda_keff", -1)) == 16, "router", "effective top-k")
    require(
        float(router["selected_weight_metrics"]["relative_l2_error"]) <= 1e-6,
        "router",
        "selected weights",
    )
    numerical["router"] = {
        "routing_equality": True,
        "selected_expert_ids": router["cuda_expert_ids"],
        "selected_weight_relative_l2_error": router["selected_weight_metrics"][
            "relative_l2_error"
        ],
    }

    dense_summaries = []
    for _, document in evidence["dense_projection"]:
        correctness = document["correctness"]["cuda_vs_production_quantized_oracle"]
        require(
            float(correctness["relative_l2_error"]) <= 1e-5,
            "dense_projection",
            "relative L2",
        )
        require(
            float(correctness["cosine_similarity"]) >= 0.999999,
            "dense_projection",
            "cosine",
        )
        dense_summaries.append(
            {
                "dimensions": document["fixture"]["dimensions"],
                "relative_l2_error": correctness["relative_l2_error"],
                "cosine_similarity": correctness["cosine_similarity"],
            }
        )
    numerical["dense_projection"] = dense_summaries

    embedding = evidence["embedding"][0][1]["correctness"]
    require(embedding.get("bit_exact") is True, "embedding", "BF16 gather bytes")
    numerical["embedding"] = {
        "bit_exact": True,
        "relative_l2_error": embedding["relative_l2_error"],
    }

    norm = evidence["final_norm"][0][1]["correctness"]
    require(float(norm["relative_l2_error"]) <= 2e-6, "final_norm", "relative L2")
    require(float(norm["cosine_similarity"]) >= 0.999999, "final_norm", "cosine")
    numerical["final_norm"] = {
        "relative_l2_error": norm["relative_l2_error"],
        "cosine_similarity": norm["cosine_similarity"],
    }

    head = evidence["LM_head"][0][1]["correctness"]
    head_serial = head["cuda_vs_retained_serial_oracle"]
    require(float(head_serial["relative_l2_error"]) <= 1e-5, "LM_head", "serial relative L2")
    require(
        int(head["cuda_argmax_token_id"])
        == int(head["serial_argmax_token_id"])
        == int(head["expected_generated_token_id"]),
        "LM_head",
        "argmax token",
    )
    numerical["LM_head"] = {
        "cuda_vs_serial_relative_l2_error": head_serial["relative_l2_error"],
        "argmax_token_id": head["cuda_argmax_token_id"],
    }

    shared = evidence["shared_experts"][0][1]["correctness"][
        "cuda_vs_production_quantized_oracle"
    ]
    require(float(shared["relative_l2_error"]) <= 3e-5, "shared_experts", "relative L2")
    require(float(shared["cosine_similarity"]) >= 0.999999, "shared_experts", "cosine")
    numerical["shared_experts"] = {
        "relative_l2_error": shared["relative_l2_error"],
        "cosine_similarity": shared["cosine_similarity"],
    }

    reduction = evidence["MoE_reduction"][0][1]["correctness"]
    require(reduction.get("routing_equality") is True, "MoE_reduction", "router identity")
    require(reduction.get("repeat_bit_exact") is True, "MoE_reduction", "repeatability")
    require(
        float(reduction["reduction_metrics"]["relative_l2_error"]) <= 2e-6,
        "MoE_reduction",
        "relative L2",
    )
    numerical["MoE_reduction"] = {
        "relative_l2_error": reduction["reduction_metrics"]["relative_l2_error"],
        "routing_equality": True,
        "repeat_bit_exact": True,
    }

    attnres = evidence["attention_residual"][0][1]["correctness"]
    require(attnres.get("repeat_bit_exact") is True, "attention_residual", "repeatability")
    require(
        float(attnres["cuda_final_vs_retained_serial"]["relative_l2_error"]) <= 2e-6,
        "attention_residual",
        "serial relative L2",
    )
    numerical["attention_residual"] = {
        "cuda_vs_serial_relative_l2_error": attnres[
            "cuda_final_vs_retained_serial"
        ]["relative_l2_error"],
        "repeat_bit_exact": True,
    }

    kda = evidence["KDA"][0][1]["correctness"]
    require(float(kda["maximum_output_relative_l2_error"]) <= 3e-5, "KDA", "output")
    require(float(kda["state_metrics"]["relative_l2_error"]) <= 3e-5, "KDA", "state")
    require(kda.get("repeat_bit_exact") is True, "KDA", "repeatability")
    require(kda.get("post_benchmark_state_finite") is True, "KDA", "finite state")
    numerical["KDA"] = {
        "maximum_output_relative_l2_error": kda["maximum_output_relative_l2_error"],
        "state_relative_l2_error": kda["state_metrics"]["relative_l2_error"],
        "repeat_bit_exact": True,
    }

    mla = evidence["Gated_MLA"][0][1]["correctness"]
    require(float(mla["maximum_output_relative_l2_error"]) <= 3e-5, "Gated_MLA", "output")
    require(
        float(mla["latent_cache_metrics"]["relative_l2_error"]) <= 3e-5,
        "Gated_MLA",
        "latent cache",
    )
    require(
        float(mla["rope_cache_metrics"]["relative_l2_error"]) <= 3e-5,
        "Gated_MLA",
        "NoPE cache",
    )
    require(
        mla.get("rope_cache_copy_from_cuda_projection_bit_exact") is True,
        "Gated_MLA",
        "NoPE cache copy",
    )
    require(mla.get("repeat_output_bit_exact") is True, "Gated_MLA", "repeatability")
    numerical["Gated_MLA"] = {
        "maximum_output_relative_l2_error": mla["maximum_output_relative_l2_error"],
        "latent_cache_relative_l2_error": mla["latent_cache_metrics"][
            "relative_l2_error"
        ],
        "rope_cache_relative_l2_error": mla["rope_cache_metrics"][
            "relative_l2_error"
        ],
        "rope_cache_copy_bit_exact": True,
        "repeat_bit_exact": True,
    }

    def inspect(*arguments: str) -> str:
        completed = subprocess.run(
            [str(cuobjdump), *arguments, str(binary_source)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode:
            raise CompatibilityError(
                f"cuobjdump {' '.join(arguments)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout

    elf_listing = inspect("--list-elf")
    ptx = inspect("--dump-ptx")
    sass = inspect("--dump-sass")
    all_symbols = sorted(
        {symbol for symbols in _FULL_OPERATION_SYMBOLS.values() for symbol in symbols}
    )
    symbol_contract = {
        symbol: {"compute86_ptx": symbol in ptx, "sm86_cubin_sass": symbol in sass}
        for symbol in all_symbols
    }
    artifact_contract = {
        "sm86_cubin": "sm_86.cubin" in elf_listing,
        "compute86_ptx": ".target sm_86" in ptx,
        "blackwell_cubin_absent": "sm_100.cubin" not in elf_listing
        and "sm_120.cubin" not in elf_listing,
        "required_symbols": symbol_contract,
    }
    if not (
        artifact_contract["sm86_cubin"]
        and artifact_contract["compute86_ptx"]
        and artifact_contract["blackwell_cubin_absent"]
        and all(
            coverage["compute86_ptx"] and coverage["sm86_cubin_sass"]
            for coverage in symbol_contract.values()
        )
    ):
        raise CompatibilityError("sm_86 binary lacks complete PTX/cubin symbol coverage")

    operation_evidence: dict[str, Any] = {}
    for operation in sorted(required_operations):
        operation_sources = evidence[operation]
        operation_contract = {
            symbol: symbol_contract[symbol]
            for symbol in _FULL_OPERATION_SYMBOLS[operation]
        }
        operation_evidence[operation] = {
            "benchmarks": [
                {
                    "path": str(source),
                    "sha256": _sha256(source),
                    "schema_version": document["schema_version"],
                    "backend_identity": document["backend"]["identity"],
                }
                for source, document in operation_sources
            ],
            "binary": str(binary_source),
            "binary_sha256": actual_hash,
            "numerical": numerical[operation],
            "artifact_contract": operation_contract,
            "physical_sm86_execution": False,
            "physical_validation": "Experiment 015 single-RTX-3090 canary",
        }
        matrix_rows[operation].update(
            {
                "existing_cuda_primitive": _FULL_OPERATION_PRIMITIVES[operation],
                "wiring_needed": None,
                "new_kernel_needed": False,
                "sm86_status": "CERTIFIED",
                "readiness": "CUDA_READY",
                "certification_evidence": operation_evidence[operation],
            }
        )

    generated = datetime.now(UTC).isoformat()
    qualification = {
        "schema_version": "experiment-014-k3-cuda-same-binary-qualification-v1",
        "generated_at_utc": generated,
        "cycle_id": cycle_id,
        "status": "PASS",
        "hypothesis": (
            "All 11 Kimi-critical operation classes retain real-weight correctness and "
            "no-fallback CUDA identity on one exact sm_86+compute_86 deployment binary."
        ),
        "binary": {
            "path": str(binary_source),
            "sha256": actual_hash,
            "bytes": binary_source.stat().st_size,
            "artifact_contract": artifact_contract,
            "architecture_specific_intrinsics_audit": (
                "All required entry points compiled to sm_86 SASS and compute_86 PTX; "
                "the package contains no Blackwell-only cubin."
            ),
        },
        "operations": operation_evidence,
        "summary": {
            "critical_operation_classes": 11,
            "cuda_ready": 11,
            "same_binary_hashes": 1,
            "benchmark_artifacts": len(evidence_sources),
            "blockers": 0,
            "physical_sm86_execution_deferred_to_canary": True,
        },
    }
    qualification_destination = qualification_path.expanduser().resolve()
    qualification_destination.parent.mkdir(parents=True, exist_ok=True)
    qualification_temporary = qualification_destination.with_suffix(
        qualification_destination.suffix + ".partial"
    )
    qualification_temporary.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    qualification_temporary.replace(qualification_destination)

    matrix["generated_at_utc"] = generated
    matrix["status"] = "PASS"
    matrix["promotion_cycle"] = cycle_id
    matrix["same_binary_qualification"] = {
        "path": str(qualification_destination),
        "sha256": _sha256(qualification_destination),
        "binary_sha256": actual_hash,
    }
    matrix["summary"]["cuda_ready"] = 11
    matrix["summary"]["blockers"] = 0
    matrix["summary"]["actual_bottleneck"] = (
        "The CUDA operation surface is complete; the next dependency is canonical "
        "persistent full-graph stage integration, not another component kernel."
    )
    cuda_source_value = matrix.get("source_evidence", {}).get("cuda_source")
    if isinstance(cuda_source_value, str) and Path(cuda_source_value).is_file():
        matrix["source_evidence"]["cuda_source_sha256"] = _sha256(Path(cuda_source_value))
    matrix_temporary = matrix_source.with_suffix(matrix_source.suffix + ".partial")
    matrix_temporary.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix_temporary.replace(matrix_source)

    certification_operations = [
        {
            "kimi_operation": operation,
            "status": "CERTIFIED",
            "implementation_kind": matrix_rows[operation]["implementation_kind"],
            "evidence": operation_evidence[operation],
            "blocker": None,
        }
        for operation in sorted(required_operations)
    ]
    certification = {
        "schema_version": "experiment-014-rtx3090-sm86-certification-v2",
        "generated_at_utc": generated,
        "status": "PASS",
        "target": {
            "gpu_class": "RTX 3090",
            "compute_capability": "sm_86",
            "binary": str(binary_source),
            "binary_sha256": actual_hash,
            "binary_bytes": binary_source.stat().st_size,
            "physical_sm86_execution": False,
            "physical_validation": "Experiment 015 single-RTX-3090 canary",
        },
        "artifact_contract": artifact_contract,
        "summary": {
            "critical_operation_classes": 11,
            "certified": 11,
            "fallback_certified": 0,
            "blockers": 0,
            "physical_sm86_execution_deferred_to_canary": True,
        },
        "operations": certification_operations,
    }
    certification_destination = certification_path.expanduser().resolve()
    certification_destination.parent.mkdir(parents=True, exist_ok=True)
    certification_temporary = certification_destination.with_suffix(
        certification_destination.suffix + ".partial"
    )
    certification_temporary.write_text(
        json.dumps(certification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    certification_temporary.replace(certification_destination)
    return {
        "schema_version": certification["schema_version"],
        "status": "PASS",
        "qualification_path": str(qualification_destination),
        "qualification_sha256": _sha256(qualification_destination),
        "operation_matrix_path": str(matrix_source),
        "operation_matrix_sha256": _sha256(matrix_source),
        "certification_path": str(certification_destination),
        "certification_sha256": _sha256(certification_destination),
        "binary_sha256": actual_hash,
        "summary": certification["summary"],
    }


def build_compatibility_matrix(
    kimi_source: Path,
    cuda_source: Path,
    makefile: Path,
    output_path: Path,
    *,
    sm86_binary: Path | None = None,
) -> dict[str, Any]:
    kimi_path = kimi_source.expanduser().resolve()
    cuda_path = cuda_source.expanduser().resolve()
    make_path = makefile.expanduser().resolve()
    for path in (kimi_path, cuda_path, make_path):
        if not path.is_file():
            raise CompatibilityError(f"missing compatibility source: {path}")
    kimi = kimi_path.read_text(encoding="utf-8", errors="replace")
    cuda = cuda_path.read_text(encoding="utf-8", errors="replace")
    make = make_path.read_text(encoding="utf-8", errors="replace")
    kimi_has_cuda_dispatch = "COLI_CUDA" in kimi
    kimi_has_vulkan_dispatch = "COLI_VULKAN" in kimi
    make_kimi_uses_cuda_object = any(
        "kimi_k3$(EXE):" in line and "CUDA_OBJ" in line for line in make.splitlines()
    )
    make_has_sm86 = "arch=compute_86,code=sm_86" in make
    generic_cuda_has_ampere_paths = "__CUDA_ARCH__ >= 750" in cuda and "wmma::mma_sync" in cuda
    binary = sm86_binary.expanduser().resolve() if sm86_binary is not None else None
    binary_evidence = {
        "path": str(binary) if binary is not None else None,
        "exists": bool(binary and binary.is_file()),
        "sha256": _sha256(binary) if binary and binary.is_file() else None,
        "architecture_metadata": "NOT_INSPECTED" if binary and binary.is_file() else "MISSING",
    }
    operations = []
    for name, source_symbol, feature in (
        ("token_embedding", "st_read_slice_f32", "host checkpoint read + GPU gather"),
        ("dense_projection", "w_matmul", "BF16/int8/int4 matmul"),
        ("kda_attention", "kda_forward", "recurrent matrix update + short convolution"),
        ("gated_mla_attention", "mla_forward", "causal latent-cache attention"),
        ("attnres", "res_mix", "RMS-normalized residual mixing"),
        ("moe_router", "moe_forward", "896-way sigmoid+bias top-16"),
        ("native_mxfp4_routed_expert", "matmul_mxfp4", "E2M1/UE8M0 group-32 decode/matmul"),
        ("shared_experts", "shared_experts", "SiTU-GLU projections"),
        ("fixed_order_moe_reduction", "moe_forward", "FP32 weighted reduction"),
        ("final_norm", "rmsnorm_", "RMS normalization"),
        ("lm_head", "lm_head", "163840-way output projection"),
    ):
        operations.append(
            {
                "operation": name,
                "source": str(kimi_path),
                "source_symbol": source_symbol,
                "backend": "CPU reference; optional Vulkan; no Kimi CUDA dispatch",
                "required_cuda_feature": feature,
                "target_architectures": ["sm_86"],
                "compile_flags": ["-gencode arch=compute_86,code=sm_86"],
                "ptx_cubin_architecture": binary_evidence["architecture_metadata"],
                "fallback_behaviour": "mathematically implemented CPU reference path",
                "blackwell_only_assumption": False,
                "architecture_specific_intrinsics": "none in Kimi source; CUDA backend WMMA >= sm_75",
                "expected_sm86_support": "generic CUDA primitives should compile, but this operation is not wired to them",
                "compatibility_status": "BLOCKER",
                "blocker": "full Kimi K3 engine has no CUDA dispatch, so an sm_86 binary cannot execute this operation on the RTX 3090",
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL",
        "target": {
            "gpu": "NVIDIA GeForce RTX 3090",
            "compute_capability": "8.6",
            "cuda_architecture": "sm_86",
            "purpose": "compatibility certification, not speed measurement",
        },
        "source_evidence": {
            "kimi_source": str(kimi_path),
            "kimi_source_sha256": _sha256(kimi_path),
            "cuda_source": str(cuda_path),
            "cuda_source_sha256": _sha256(cuda_path),
            "makefile": str(make_path),
            "makefile_sha256": _sha256(make_path),
            "kimi_has_cuda_dispatch": kimi_has_cuda_dispatch,
            "kimi_has_vulkan_dispatch": kimi_has_vulkan_dispatch,
            "kimi_make_target_links_cuda_object": make_kimi_uses_cuda_object,
            "portable_build_declares_sm86": make_has_sm86,
            "generic_cuda_has_ampere_wmma_paths": generic_cuda_has_ampere_paths,
        },
        "generic_sm86_binary": binary_evidence,
        "operations": operations,
        "summary": {
            "critical_operations": len(operations),
            "certified_for_sm86": 0,
            "fallback_available_but_not_product_gpu_path": len(operations),
            "blockers": len(operations),
            "gate_9_sm86_package": False,
            "specific_blocker": "Colibri's full Kimi K3 target links only CPU/Vulkan code; generic coli_cuda is not called by kimi_k3.c",
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "summary": payload["summary"],
    }


__all__ = [
    "CompatibilityError",
    "build_compatibility_matrix",
    "build_cuda_operation_matrix",
    "certify_sm86_complete_kimi",
    "certify_sm86_routed_expert",
]
