"""Executable capability negotiation for a concrete Colibri build."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.backends.colibri.constants import (
    COLIBRI_BRIDGE_VERSION,
    COLIBRI_COMMIT,
    COLIBRI_RELEASE,
)
from swarm_inference.backends.colibri.schemas import ColibriCapabilityReport


def _binary(directory: Path, basename: str) -> Path | None:
    for name in (basename, f"{basename}.exe"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _nvidia_devices() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    devices = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) < 6:
            continue
        try:
            devices.append(
                {
                    "index": int(row[0].strip()),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "total_memory_bytes": int(float(row[3])) * 1024 * 1024,
                    "free_memory_bytes": int(float(row[4])) * 1024 * 1024,
                    "compute_capability": row[5].strip(),
                }
            )
        except ValueError:
            continue
    return devices


def _storage_inventory(model_path: Path | None) -> dict[str, Any]:
    devices: list[dict[str, Any]] = []
    seen: set[str] = set()
    for partition in psutil.disk_partitions(all=False):
        mountpoint = partition.mountpoint
        identity = f"{partition.device}|{mountpoint}"
        if identity in seen:
            continue
        seen.add(identity)
        try:
            usage = psutil.disk_usage(mountpoint)
        except OSError:
            continue
        devices.append(
            {
                "device": partition.device,
                "mountpoint": mountpoint,
                "filesystem": partition.fstype,
                "total_bytes": usage.total,
                "available_bytes": usage.free,
                "model_storage": bool(
                    model_path is not None
                    and os.path.splitdrive(str(model_path))[0].lower()
                    == os.path.splitdrive(mountpoint)[0].lower()
                ),
            }
        )
    return {"devices": devices}


class ColibriCapabilityProbe:
    """Report only features the supplied binaries can actually execute."""

    def __init__(
        self,
        engine_directory: str | Path,
        *,
        source_directory: str | Path | None = None,
        build_manifest: str | Path | None = None,
        model_path: str | Path | None = None,
        cuda_proof: str | Path | None = None,
    ) -> None:
        self.engine_directory = Path(engine_directory).expanduser().resolve()
        self.source_directory = (
            None if source_directory is None else Path(source_directory).expanduser().resolve()
        )
        self.build_manifest = (
            None if build_manifest is None else Path(build_manifest).expanduser().resolve()
        )
        self.model_path = None if model_path is None else Path(model_path).expanduser().resolve()
        self.cuda_proof = None if cuda_proof is None else Path(cuda_proof).expanduser().resolve()

    def _validated_cuda_proof(self, cuda_dll: Path) -> dict[str, Any] | None:
        if self.cuda_proof is None or not self.cuda_proof.is_file() or not cuda_dll.is_file():
            return None
        try:
            proof = json.loads(self.cuda_proof.read_text(encoding="utf-8-sig"))
            digest = hashlib.sha256(cuda_dll.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError):
            return None
        required = (
            proof.get("dll_loaded"),
            proof.get("device_detected"),
            proof.get("kernel_executed"),
            proof.get("correctness_passed"),
            proof.get("cuda_dll_sha256") == digest,
        )
        return proof if all(required) else None

    def _build_flags(self) -> dict[str, str]:
        candidates = []
        if self.source_directory is not None:
            candidates.append(self.source_directory / "c" / ".build-config")
            candidates.append(self.source_directory / ".build-config")
        candidates.append(self.engine_directory / ".build-config")
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            result = {}
            for field in text.split("|"):
                if "=" in field:
                    key, value = field.split("=", 1)
                    result[key.strip()] = value.strip()
            return result
        return {}

    def probe(self) -> ColibriCapabilityReport:
        if self.build_manifest is not None:
            manifest = json.loads(self.build_manifest.read_text(encoding="utf-8-sig"))
            if manifest.get("commit") != COLIBRI_COMMIT:
                raise ValueError("Colibri build manifest is not pinned to Experiment 009 commit")
        built_adapters = tuple(
            adapter
            for adapter in default_colibri_adapter_registry().adapters()
            if _binary(self.engine_directory, adapter.engine_basename) is not None
        )
        built_families = [adapter.adapter_id for adapter in built_adapters]
        flags = self._build_flags()
        cuda_dll = self.engine_directory / "coli_cuda.dll"
        cuda_build_present = flags.get("CUDA") == "1" or (
            flags.get("CUDA_DLL") == "1" and cuda_dll.is_file()
        )
        cuda_proof = self._validated_cuda_proof(cuda_dll) if cuda_build_present else None
        supports_cuda = cuda_proof is not None
        supports_vulkan = flags.get("VK") == "1"
        supports_metal = flags.get("METAL") == "1" and platform.system() == "Darwin"
        execution = ["cpu"] if built_families else []
        if supports_cuda:
            execution.append("cuda")
        if supports_vulkan:
            execution.append("vulkan")
        if supports_metal:
            execution.append("metal")
        gpu_devices = _nvidia_devices()
        virtual_memory = psutil.virtual_memory()
        physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        logical = psutil.cpu_count(logical=True) or physical
        bridge_present = (self.engine_directory / "swarm_bridge.py").is_file()
        formats = {"bf16", "f16", "f32", "int8_rowwise"}
        if "glm-5.2" in built_families:
            formats.update({"int4", "int3", "fp8_e4m3"})
        if "kimi-k3" in built_families:
            formats.add("mxfp4")
        storage = _storage_inventory(self.model_path)
        tiers = [
            {"name": "ram", "executable": bool(built_families)},
            {"name": "nvme", "executable": bool(built_families)},
            {
                "name": "vram",
                "executable": supports_cuda or supports_vulkan or supports_metal,
            },
        ]
        return ColibriCapabilityReport(
            colibri_version=COLIBRI_RELEASE,
            colibri_commit=COLIBRI_COMMIT,
            bridge_version=COLIBRI_BRIDGE_VERSION if bridge_present else "",
            platform=platform.platform(),
            architecture=platform.machine(),
            model_families=built_families,
            execution_backends=execution,
            quantization_formats=sorted(formats),
            supports_cpu="cpu" in execution,
            supports_cuda=supports_cuda,
            supports_vulkan=supports_vulkan,
            supports_metal=supports_metal,
            supports_multi_gpu=supports_cuda and len(gpu_devices) > 1,
            supports_expert_residency=bool(built_families),
            supports_route_trace=bool(built_families),
            supports_usage_history=bool(built_families),
            supports_expert_prefetch=bool(built_families),
            supports_dynamic_reconfiguration=False,
            supports_native_mxfp4=any(
                adapter.native_quantization == "mxfp4" for adapter in built_adapters
            ),
            supports_tensor_microshards=any(
                adapter.tensor_microshards for adapter in built_adapters
            ),
            supports_full_expert_placement=bool(built_families),
            supports_exact_replay=any(adapter.exact_replay for adapter in built_adapters),
            supports_prefill_decode_separation=bool(built_families),
            storage_tiers=tiers,
            gpu_devices=gpu_devices,
            cpu={
                "model": platform.processor(),
                "physical_cores": physical,
                "logical_cores": logical,
            },
            memory={
                "total_bytes": virtual_memory.total,
                "available_bytes": virtual_memory.available,
            },
            storage=storage,
            cuda_kernel_proof=cuda_proof,
        )

    @staticmethod
    def supported_tuning_settings(
        report: ColibriCapabilityReport, *, model_family: str | None = None
    ) -> set[str]:
        registry = default_colibri_adapter_registry()
        settings = set(
            registry.get(model_family).tuning_settings
            if model_family is not None
            else next(iter(registry.adapters())).tuning_settings
        )
        if report.supports_cuda:
            settings.update({"COLI_CUDA_PIPE", "COLI_CUDA_ASYNC", "CUDA_EXPERT_GB", "PIN_GB"})
        return settings
