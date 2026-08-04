"""Exhaustive local discovery and identity capture for Experiment 011."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
import transformers


@dataclass(frozen=True, slots=True)
class DiscoveredAssets:
    repository_root: str
    experiment_010_final_bundle: str
    experiment_010_zip: str
    experiment_010_network_results: str
    experiment_010_workload_reference: str
    experiment_010_transport_manifest: str
    native_engine: str
    native_expert_worker: str
    native_model_path: str
    source_model_path: str
    worker_bank_paths: tuple[str, ...]
    model_fingerprint: str
    searched_locations: tuple[dict[str, Any], ...]
    missing_artifacts: tuple[str, ...]
    compatible_draft_models: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["worker_bank_paths"] = list(self.worker_bank_paths)
        value["searched_locations"] = list(self.searched_locations)
        value["missing_artifacts"] = list(self.missing_artifacts)
        value["compatible_draft_models"] = list(self.compatible_draft_models)
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_search(rows: list[dict[str, Any]], *, category: str, path: Path, detail: str) -> None:
    rows.append(
        {
            "category": category,
            "path": str(path.resolve()) if path.exists() else str(path.absolute()),
            "exists": path.exists(),
            "detail": detail,
        }
    )


def discover_assets(
    repository_root: Path, *, model_override: Path | None = None
) -> DiscoveredAssets:
    root = repository_root.resolve()
    searched: list[dict[str, Any]] = []
    final_bundle = (
        root / "artifacts" / "runs" / "experiment-010-correction-final" / "experiment_010"
    )
    final_zip = root / "artifacts" / "runs" / "experiment-010-correction-final.zip"
    network_results = (
        root
        / "artifacts"
        / "runs"
        / "experiment-010-correction-work"
        / "phase-10"
        / "final-binary"
        / "network-profiles"
        / "network_profile_results.csv"
    )
    reference = (
        root
        / "artifacts"
        / "runs"
        / "experiment-010-correction-work"
        / "phase-6"
        / "local-correctness-references"
        / "code-01"
        / "reference.json"
    )
    transport_manifest = final_bundle / "transport_profiles.json"
    engine = root / "build" / "colibri" / "bin" / "olmoe.exe"
    worker = root / "build" / "colibri" / "bin" / "olmoe_expert_worker.exe"
    native_model = root / "artifacts" / "models" / "colibri" / "olmoe-1b-7b-0125-instruct-merged"
    source_model = model_override or (
        root / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
    )
    banks_root = (
        root / "artifacts" / "runs" / "experiment-010-correction-work" / "phase-6" / "banks"
    )
    banks = tuple(banks_root / f"level-a-worker-{index}" for index in range(4))
    for category, path, detail in (
        ("experiment_010_manifest", final_bundle / "manifest.json", "final evidence manifest"),
        ("experiment_010_run_metadata", final_bundle / "verdict.json", "final verdict"),
        ("experiment_010_archive", final_zip, "final ZIP bundle"),
        ("repository_configuration", root / "pyproject.toml", "project configuration"),
        ("repository_artifact", network_results, "exact archived network workload rows"),
        ("repository_artifact", reference, "prompt and token reference"),
        ("experiment_010_evidence_reference", transport_manifest, "exact shaping profiles"),
        ("native_binary", engine, "Experiment 010 monolithic/coordinator executable"),
        ("native_binary", worker, "Experiment 010 whole-expert worker executable"),
        ("model_cache", native_model, "Experiment 010 merged native model"),
        ("model_cache", source_model, "canonical safetensors and tokenizer"),
    ):
        _record_search(searched, category=category, path=path, detail=detail)
    for bank in banks:
        _record_search(
            searched, category="experiment_010_artifact", path=bank, detail="worker bank"
        )
    hf_locations = [
        Path.home() / ".cache" / "huggingface" / "hub",
        root / ".cache" / "huggingface" / "hub",
        root / ".cache" / "experiment_008" / "huggingface" / "hub",
    ]
    for variable in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            hf_locations.append(Path(value))
    for location in hf_locations:
        _record_search(
            searched,
            category="hugging_face_cache",
            path=location,
            detail="searched for canonical and compatible draft weights",
        )
    launch_scripts = sorted(root.glob("**/*.ps1"))
    searched.append(
        {
            "category": "previous_launch_scripts",
            "path": str(root),
            "exists": True,
            "detail": f"searched {len(launch_scripts)} PowerShell scripts for model references",
            "matching_scripts": [
                str(path.resolve())
                for path in launch_scripts
                if "experiment-010" in path.read_text(encoding="utf-8", errors="ignore").lower()
                or "olmoe" in path.read_text(encoding="utf-8", errors="ignore").lower()
            ],
        }
    )
    env_names = sorted(
        name
        for name in os.environ
        if any(marker in name.upper() for marker in ("MODEL", "HF_", "HUGGING", "TRANSFORMERS"))
    )
    searched.append(
        {
            "category": "environment_variables",
            "path": "process environment",
            "exists": True,
            "detail": "names inspected; secret values deliberately not persisted",
            "variable_names": env_names,
        }
    )
    compatible_drafts: list[str] = []
    # Compatibility is proven conservatively from exact tokenizer artifact
    # hashes.  A cache entry without an identical tokenizer is not selected.
    target_tokenizer = source_model / "tokenizer.json"
    target_hash = sha256_file(target_tokenizer) if target_tokenizer.is_file() else None
    if target_hash:
        for location in hf_locations:
            if not location.is_dir():
                continue
            for candidate in location.glob("models--*--*/snapshots/*/tokenizer.json"):
                try:
                    if sha256_file(candidate) == target_hash:
                        compatible_drafts.append(str(candidate.parent.resolve()))
                except OSError:
                    continue
    required = {
        "Experiment 010 final evidence manifest": final_bundle / "manifest.json",
        "Experiment 010 archived network rows": network_results,
        "Experiment 010 prompt reference": reference,
        "Experiment 010 transport manifest": transport_manifest,
        "native engine": engine,
        "native whole-expert worker": worker,
        "native merged model": native_model / "config.json",
        "canonical safetensors index": source_model / "model.safetensors.index.json",
        "canonical tokenizer": source_model / "tokenizer.json",
        **{f"worker bank {index}": bank / "manifest.json" for index, bank in enumerate(banks)},
    }
    missing = tuple(name for name, path in required.items() if not path.is_file())
    model_fingerprint = ""
    if network_results.is_file():
        import csv

        with network_results.open("r", encoding="utf-8", newline="") as handle:
            first = next(csv.DictReader(handle))
        model_fingerprint = str(first["model_fingerprint"])
    return DiscoveredAssets(
        repository_root=str(root),
        experiment_010_final_bundle=str(final_bundle),
        experiment_010_zip=str(final_zip),
        experiment_010_network_results=str(network_results),
        experiment_010_workload_reference=str(reference),
        experiment_010_transport_manifest=str(transport_manifest),
        native_engine=str(engine),
        native_expert_worker=str(worker),
        native_model_path=str(native_model),
        source_model_path=str(source_model.resolve()),
        worker_bank_paths=tuple(str(bank) for bank in banks),
        model_fingerprint=model_fingerprint,
        searched_locations=tuple(searched),
        missing_artifacts=missing,
        compatible_draft_models=tuple(sorted(set(compatible_drafts))),
    )


def git_identity(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    diff = run("diff", "--binary")
    staged = run("diff", "--cached", "--binary")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "staged_diff_sha256": hashlib.sha256(staged.encode("utf-8")).hexdigest(),
        "dirty": bool(status),
    }


def environment_identity() -> dict[str, Any]:
    virtual = psutil.virtual_memory()
    gpu = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        if len(fields) >= 4:
            gpu = {
                "name": fields[0],
                "uuid": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": int(fields[3]),
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu = {"query_error": "nvidia-smi identity query failed"}
    return {
        "captured_at_ns": time_ns(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "host_memory_bytes": virtual.total,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers_version": transformers.__version__,
        "gpu": gpu,
    }


def time_ns() -> int:
    import time

    return time.time_ns()
