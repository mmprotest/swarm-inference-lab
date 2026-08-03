"""Create deterministic Colibri source and binary build fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    excluded = {".git", "__pycache__"}
    generated_names = {
        ".build-config",
        "colibri",
        "colibri.exe",
        "glm",
        "glm.exe",
        "inkling",
        "inkling.exe",
        "kimi_k3",
        "kimi_k3.exe",
        "olmoe",
        "olmoe.exe",
        "olmoe_expert_worker",
        "olmoe_expert_worker.exe",
        "coli_kimi_mxfp4.dll",
        "libcoli_kimi_mxfp4.so",
        "coli_cuda.dll",
        "coli_cuda.exp",
        "coli_cuda.lib",
    }
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not excluded.intersection(path.parts)
        and path.name not in generated_names
        and path.suffix not in {".o", ".obj", ".pyc"}
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr).splitlines()
    return output[0].strip() if output else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--patches", nargs="*", default=[])
    parser.add_argument("--patch-directory", type=Path)
    parser.add_argument("--wire-adapter-directory", type=Path)
    parser.add_argument("--patch-manifest-output", type=Path)
    args = parser.parse_args()
    binaries = []
    for path in sorted(item for item in args.bin.iterdir() if item.is_file()):
        binaries.append({"name": path.name, "byte_size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "schema_version": "experiment-010-correction-colibri-build-v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "repository": "JustVugg/colibri",
        "release": "v1.4.0",
        "commit": args.commit,
        "license": "Apache-2.0",
        "source_tree_sha256": tree_hash(args.source),
        "patches": args.patches,
        "shared_expert_runtime": {
            "header": "c/olmoe_expert_runtime.h",
            "source": "c/olmoe_expert_runtime.c",
            "worker": "c/olmoe_expert_worker.c",
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/olmoe_expert_runtime.h",
                    "c/olmoe_expert_runtime.c",
                    "c/olmoe_expert_worker.c",
                )
            ),
        },
        "external_expert_dispatch": {
            "header": "c/olmoe_external_dispatch.h",
            "source": "c/olmoe_external_dispatch.c",
            "socket_source": "c/olmoe_expert_socket.c",
            "wire_header": "c/swarm_expert_wire.h",
            "wire_source": "c/swarm_expert_wire.c",
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/olmoe_external_dispatch.h",
                    "c/olmoe_external_dispatch.c",
                    "c/olmoe_expert_socket.c",
                    "c/swarm_expert_wire.h",
                    "c/swarm_expert_wire.c",
                )
            ),
        },
        "native_microshards": {
            "quantized_bytes_preserved": True,
            "exact_reduction": "ordered_unscaled_down_accumulator_then_original_row_scale",
            "wire_chain_state": "SWARMEX1/SWARMT01 down_accumulators",
            "runtime_source": "c/olmoe_expert_runtime.c",
            "dispatch_source": "c/olmoe_external_dispatch.c",
            "worker_source": "c/olmoe_expert_worker.c",
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/olmoe_expert_runtime.c",
                    "c/olmoe_external_dispatch.c",
                    "c/olmoe_expert_worker.c",
                    "c/swarm_expert_wire.c",
                )
            ),
        },
        "memory_residency_telemetry": {
            "header": "c/olmoe_memory_residency.h",
            "source": "c/olmoe_memory_residency.c",
            "windows_apis": [
                "GetProcessMemoryInfo",
                "GetPerformanceInfo",
                "GlobalMemoryStatusEx",
                "QueryWorkingSetEx",
            ],
            "resident_cache_classification": "sampled_QueryWorkingSetEx",
            "unavailable_windows_metrics": [
                "pagefile_read_bytes",
                "hard_soft_fault_attribution",
                "process_memory_compression_bytes",
            ],
            "capacity_isolation_guard": "skip_local_runtime_when_all_routed_experts_are_remote",
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/olmoe_memory_residency.h",
                    "c/olmoe_memory_residency.c",
                    "c/tests/test_olmoe_memory_residency.c",
                )
            ),
        },
        "native_data_planes": {
            "canonical_protocol": "SWARMEX1",
            "supported": ["direct_tcp", "relayed_tcp", "shared_memory"],
            "shared_memory_header": "c/olmoe_expert_shm.h",
            "shared_memory_source": "c/olmoe_expert_shm.c",
            "shared_memory_control": "SWARMEX1 control handshake over TCP",
            "worker_queue_measurement": "mutex_wait_ns_excluding_compute",
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/olmoe_expert_shm.h",
                    "c/olmoe_expert_shm.c",
                    "c/tests/test_olmoe_expert_shm.c",
                )
            ),
        },
        "native_olmoe_cuda": {
            "backend_header": "c/backend_cuda.h",
            "backend_source": "c/backend_cuda.cu",
            "runtime_loader": "c/backend_loader.c",
            "shared_runtime": "c/olmoe_expert_runtime.c",
            "native_format": "merged_int8_with_original_f32_row_scales",
            "selection_environment": "COLI_SWARM_EXPERT_CUDA_TARGET",
            "failure_contract": "required_target_fails_without_cpu_fallback",
            "telemetry": [
                "cuda_resident_tensor_bytes",
                "cuda_weight_upload_ns",
                "cuda_execution_count",
                "cuda_h2d_ns",
                "cuda_kernel_ns",
                "cuda_d2h_ns",
                "cuda_fallback_count",
            ],
            "present": all(
                (args.source / relative).is_file()
                for relative in (
                    "c/backend_cuda.h",
                    "c/backend_cuda.cu",
                    "c/backend_loader.c",
                    "c/olmoe_expert_runtime.c",
                )
            ),
        },
        "native_kimi_mxfp4_fixture": {
            "adapter_source": "integrations/colibri/adapter/kimi_mxfp4_runtime.c",
            "shared_kernel": "c/quant.h::matmul_mxfp4",
            "abi": "colibri-native-mxfp4-fixture-v1",
            "persistent_dequantization": False,
            "zero_group_skip": False,
            "binary_names": ["coli_kimi_mxfp4.dll", "libcoli_kimi_mxfp4.so"],
            "present": any(
                (args.bin / name).is_file()
                for name in ("coli_kimi_mxfp4.dll", "libcoli_kimi_mxfp4.so")
            ),
        },
        "platform": platform.platform(),
        "compiler": command_version([os.environ.get("CC", "gcc"), "--version"]),
        "binaries": binaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.patch_manifest_output is not None:
        if args.patches and args.patch_directory is None:
            parser.error("--patch-directory is required when patched files are recorded")
        patch_rows = [
            {
                "name": name,
                "sha256": sha256_file(args.patch_directory / name),
            }
            for name in args.patches
        ]
        wire_rows = []
        if args.wire_adapter_directory is not None:
            wire_rows = [
                {"name": name, "sha256": sha256_file(args.wire_adapter_directory / name)}
                for name in (
                    "swarm_expert_wire.h",
                    "swarm_expert_wire.c",
                    "kimi_mxfp4_runtime.c",
                )
            ]
        patch_payload = {
            "schema_version": "experiment-010-correction-colibri-patches-v1",
            "upstream_commit": args.commit,
            "bridge_enabled": bool(args.patches),
            "patches": patch_rows,
            "wire_adapter": wire_rows,
        }
        args.patch_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.patch_manifest_output.write_text(
            json.dumps(patch_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
