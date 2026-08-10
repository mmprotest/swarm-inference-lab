"""Static and binary evidence for RTX 3090 (Ampere ``sm_86``) compatibility."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-rtx3090-compatibility-v1"


class CompatibilityError(ValueError):
    """Compatibility evidence is missing or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


__all__ = ["CompatibilityError", "build_compatibility_matrix"]
