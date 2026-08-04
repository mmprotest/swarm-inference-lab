"""Strict current-run validation for Experiment 010 Gate 17.

This module consumes the native Experiment 008 Configuration A bundle.  It
does not acquire a model, execute a workload, or manufacture a measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swarm_inference.config.experiment_008 import load_experiment_008_config

OFFICIAL_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
OFFICIAL_REPOSITORY = "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF"
OFFICIAL_FILENAME = "Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
OFFICIAL_QUANTIZATION = "Q4_K_M"
OFFICIAL_ARCHITECTURE = "qwen3next"
PINNED_REVISION = "4c8630cf7af926a9c5095cb4bbbbc65d36e20f77"
LLAMA_CPP_RELEASE = "b9637"
LLAMA_CPP_CUDA_VERSION = "13.3"
REQUIRED_WORKLOADS = ("decode", "prefill_8k", "prefill_32k", "mixed")
REQUIRED_LEVEL_B_FILES = (
    "benchmark_results.csv",
    "model_resolution_attempts.json",
    "model_preflight.json",
    "model_execution_precheck.json",
    "tensor_inventory.json",
    "tensor_tiles.json",
    "backend_acquisition.json",
    "backend_probe.json",
    "hardware_profile.json",
    "environment.json",
    "resource_timeseries.csv",
    "correctness_results.json",
    "manifest.json",
    "report.md",
    "verdict.json",
)


class Gate17ValidationError(RuntimeError):
    """The current Level B evidence does not meet Gate 17."""

    def __init__(self, errors: list[str], *, result: dict[str, Any]) -> None:
        super().__init__("Gate 17 validation failed: " + "; ".join(errors))
        self.errors = errors
        self.result = result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value


def _json_cell(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _true(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _historical_row(row: dict[str, Any]) -> bool:
    if any(
        _true(row.get(key))
        for key in (
            "historical",
            "historical_row",
            "historical_evidence",
            "previous_evidence_reused_as_current_measurement",
        )
    ):
        return True
    return any(
        "HISTORICAL" in str(row.get(key, "")).upper()
        for key in ("evidence_category", "evidence_class", "measurement_source")
    )


def _measured_row(row: dict[str, Any]) -> bool:
    category = str(row.get("evidence_category") or "").upper()
    evidence_class = str(row.get("evidence_class") or "").upper()
    if "SYNTHETIC" in category or "SYNTHETIC" in evidence_class:
        return False
    return category in {"REAL_MODEL_MEASURED", "MEASURED"} or evidence_class == "MEASURED"


def locate_experiment_008_bundle(level_b_root: Path) -> Path:
    root = level_b_root.expanduser().resolve()
    candidates = (root / "experiment_008", root)
    bundle = next(
        (candidate for candidate in candidates if (candidate / "benchmark_results.csv").is_file()),
        None,
    )
    if bundle is None:
        raise FileNotFoundError(
            f"no current Experiment 008 bundle exists beneath Level B root {root}"
        )
    return bundle


def validate_config_pins(config_path: Path) -> dict[str, Any]:
    config = load_experiment_008_config(config_path)
    preferred = config.models.preferred
    expected = {
        "model_id": OFFICIAL_MODEL_ID,
        "artifact_repository": OFFICIAL_REPOSITORY,
        "filename": OFFICIAL_FILENAME,
        "quantization": OFFICIAL_QUANTIZATION,
        "architecture": OFFICIAL_ARCHITECTURE,
        "revision": PINNED_REVISION,
    }
    actual = {key: getattr(preferred, key) for key in expected}
    if actual != expected:
        raise ValueError(f"Experiment 008 official Level B pin mismatch: {actual}")
    if config.backend.release_tag != LLAMA_CPP_RELEASE:
        raise ValueError("Experiment 008 llama.cpp release is not pinned to b9637")
    if config.backend.windows_cuda_version != LLAMA_CPP_CUDA_VERSION:
        raise ValueError("Experiment 008 Windows CUDA package is not pinned to 13.3")
    return {
        "preferred": actual,
        "backend_release_tag": config.backend.release_tag,
        "backend_cuda_version": config.backend.windows_cuda_version,
    }


def validate_existing_correction_evidence(
    repository_root: Path,
    work_directory: Path,
    *,
    final_bundle: Path | None = None,
) -> dict[str, Any]:
    """Require every reusable correction gate before the Level B download starts."""

    repository = repository_root.expanduser().resolve()
    work = work_directory.expanduser().resolve()
    if not work.is_dir():
        raise FileNotFoundError(f"existing correction work directory is missing: {work}")

    from swarm_inference.experiments.experiment_010.correction_bundle import _validate_phases

    validation = _validate_phases(repository, work, None)
    excluded = {"level_b_current_workload"}
    failed = sorted(
        name
        for name, passed in validation["checks"].items()
        if name not in excluded and passed is not True
    )
    if failed:
        raise RuntimeError(
            "existing correction evidence is incomplete outside Gate 17: " + ", ".join(failed)
        )

    final_path = final_bundle.expanduser().resolve() if final_bundle is not None else None
    if final_path is not None:
        if final_path.name != "experiment_010":
            final_path = final_path / "experiment_010"
        verdict_path = final_path / "verdict.json"
        if not verdict_path.is_file():
            raise FileNotFoundError(f"existing correction bundle is missing: {final_path}")
        verdict = _read_json(verdict_path)
        failed_existing = [
            int(row.get("gate_id", 0))
            for row in verdict.get("gates", [])
            if int(row.get("gate_id", 0)) <= 16 and row.get("status") != "PASS"
        ]
        if failed_existing:
            raise RuntimeError(
                f"existing correction bundle has failed reusable gates: {failed_existing}"
            )

    sources = []
    for name, path in validation["paths"].items():
        if path.is_file():
            sources.append(
                {
                    "phase": name,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "status": "REUSABLE_CORRECTION_EVIDENCE_COMPLETE",
        "work_directory": str(work),
        "reused_gate_count": 16,
        "level_a_or_other_workloads_rerun": False,
        "validated_sources": sources,
    }


def _attempt_resolution(attempts: Any) -> dict[str, Any]:
    if not isinstance(attempts, list):
        return {}
    return next(
        (
            row
            for row in reversed(attempts)
            if isinstance(row, dict)
            and row.get("candidate") == "preferred"
            and row.get("status") == "COMPLETED"
        ),
        {},
    )


def _required_arguments(arguments: Any) -> bool:
    if not isinstance(arguments, list):
        return False
    expected = {
        "--n-gpu-layers": "auto",
        "--threads": "20",
        "--threads-batch": "20",
        "--batch-size": "2048",
        "--ubatch-size": "512",
        "--flash-attn": "on",
    }
    for option, value in expected.items():
        if option not in arguments:
            return False
        index = arguments.index(option)
        if index + 1 >= len(arguments) or str(arguments[index + 1]) != value:
            return False
    return (
        "--no-mmap" not in arguments
        and "--override-tensor" not in arguments
        and "--cpu-moe" not in arguments
        and "--n-cpu-moe" not in arguments
    )


def validate_level_b_bundle(
    level_b_root: Path,
    *,
    config_path: Path,
    write_result: bool = True,
) -> dict[str, Any]:
    """Validate and optionally seal the current Experiment 008 Gate 17 bundle."""

    validate_config_pins(config_path)
    root = level_b_root.expanduser().resolve()
    bundle = locate_experiment_008_bundle(root)
    result_path = root / "gate-17-validation.json"
    errors: list[str] = []

    missing = [name for name in REQUIRED_LEVEL_B_FILES if not (bundle / name).is_file()]
    if not (bundle / "logs").is_dir():
        missing.append("logs/")
    if missing:
        errors.append("missing raw Level B evidence: " + ", ".join(sorted(missing)))

    rows = _read_csv(bundle / "benchmark_results.csv") if not missing else []
    current_rows = [row for row in rows if row.get("configuration") == "A"]
    synthetic_candidates = [
        row
        for row in current_rows
        if row.get("status") == "COMPLETED"
        and "SYNTHETIC"
        in str(row.get("evidence_category") or row.get("evidence_class") or "").upper()
    ]
    historical_candidates = [
        row for row in current_rows if row.get("status") == "COMPLETED" and _historical_row(row)
    ]
    eligible = [
        row
        for row in current_rows
        if row.get("status") == "COMPLETED" and _measured_row(row) and not _historical_row(row)
    ]
    by_workload = {
        workload: [row for row in eligible if row.get("workload") == workload]
        for workload in REQUIRED_WORKLOADS
    }
    completed = sorted(workload for workload, values in by_workload.items() if values)
    missing_workloads = sorted(set(REQUIRED_WORKLOADS) - set(completed))
    if missing_workloads:
        errors.append("missing completed real workloads: " + ", ".join(missing_workloads))
    selected_rows = [
        by_workload[workload][0] for workload in REQUIRED_WORKLOADS if by_workload[workload]
    ]

    preflight = (
        _read_json(bundle / "model_preflight.json")
        if (bundle / "model_preflight.json").is_file()
        else {}
    )
    execution_precheck = (
        _read_json(bundle / "model_execution_precheck.json")
        if (bundle / "model_execution_precheck.json").is_file()
        else {}
    )
    attempts = (
        _read_json(bundle / "model_resolution_attempts.json")
        if (bundle / "model_resolution_attempts.json").is_file()
        else []
    )
    attempt = _attempt_resolution(attempts)
    resolved = attempt.get("resolved", {}) if isinstance(attempt, dict) else {}
    inventory = (
        _read_json(bundle / "tensor_inventory.json")
        if (bundle / "tensor_inventory.json").is_file()
        else {}
    )
    backend_acquisition = (
        _read_json(bundle / "backend_acquisition.json")
        if (bundle / "backend_acquisition.json").is_file()
        else {}
    )
    backend_probe = (
        _read_json(bundle / "backend_probe.json")
        if (bundle / "backend_probe.json").is_file()
        else {}
    )
    residency = (
        _read_json(bundle / "residency_accounting.json")
        if (bundle / "residency_accounting.json").is_file()
        else {}
    )
    tokens = (
        _read_json(bundle / "correctness_tokens.json")
        if (bundle / "correctness_tokens.json").is_file()
        else {}
    )

    repository = resolved.get("artifact_repository") or preflight.get("artifact_repository")
    revision = resolved.get("resolved_revision") or preflight.get("resolved_artifact_identity")
    filename = resolved.get("filename") or preflight.get("model_file_name")
    model_sha256 = resolved.get("file_sha256") or preflight.get("model_file_sha256")
    model_file_bytes = int(resolved.get("file_size") or preflight.get("model_file_size_bytes") or 0)
    tensor_bytes = int(preflight.get("total_tensor_bytes") or inventory.get("tensor_bytes") or 0)
    physical_vram_bytes = int(preflight.get("physical_vram_bytes") or 0)

    if resolved.get("model_id") != OFFICIAL_MODEL_ID:
        errors.append("resolved model ID is not the configured official Qwen3-Next model")
    if repository != OFFICIAL_REPOSITORY:
        errors.append("resolved model repository is not the official Qwen GGUF repository")
    if resolved.get("requested_revision") != PINNED_REVISION or revision != PINNED_REVISION:
        errors.append("resolved model revision does not match the pinned official revision")
    if filename != OFFICIAL_FILENAME:
        errors.append("resolved model filename does not exactly match the configured Q4_K_M file")
    if resolved.get("quantization") != OFFICIAL_QUANTIZATION:
        errors.append("resolved model quantization is not Q4_K_M")
    if resolved.get("architecture") != OFFICIAL_ARCHITECTURE:
        errors.append("resolved model architecture is not qwen3next")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        errors.append("complete model file SHA-256 is absent")
    expected_sha256 = resolved.get("expected_file_sha256")
    if expected_sha256 and expected_sha256 != model_sha256:
        errors.append("model file SHA-256 does not match official repository metadata")
    if model_file_bytes <= 0:
        errors.append("model file size is absent")
    if (
        resolved.get("expected_file_size")
        and int(resolved["expected_file_size"]) != model_file_bytes
    ):
        errors.append("model file size does not match official repository metadata")
    model_path = Path(str(resolved.get("path", "")))
    if not model_path.is_file() or model_path.stat().st_size != model_file_bytes:
        errors.append("resolved model file is absent or truncated")
    if not preflight.get("eligible"):
        errors.append("Experiment 008 model preflight was not eligible")
    if execution_precheck.get("status") != "COMPLETED":
        errors.append("real deterministic generation precheck did not pass")
    if (
        not execution_precheck.get("output_token_ids")
        or not str(execution_precheck.get("generated_text") or "").strip()
    ):
        errors.append("real deterministic generation precheck lacks text or token IDs")
    if execution_precheck.get("model_file_sha256") != model_sha256:
        errors.append("generation precheck used a different model file SHA-256")
    if tensor_bytes <= physical_vram_bytes or physical_vram_bytes <= 0:
        errors.append("logical model tensor bytes do not exceed physical GPU VRAM")

    row_hashes = {row.get("model_file_sha256") for row in selected_rows}
    if row_hashes != {model_sha256}:
        errors.append("required workloads did not all use the same model file SHA-256")
    if any(row.get("model_filename") != OFFICIAL_FILENAME for row in selected_rows):
        errors.append("one or more required workloads used a different model filename")
    if any(row.get("model_revision") != PINNED_REVISION for row in selected_rows):
        errors.append("one or more required workloads used a different model revision")
    if any(not _required_arguments(_json_cell(row, "backend_arguments")) for row in selected_rows):
        errors.append("one or more workloads did not use the established Configuration A arguments")
    for row in selected_rows:
        workload = str(row.get("workload"))
        if (_float(row, "peak_vram_bytes") or 0) <= 0:
            errors.append(f"{workload} has no measured GPU residency")
        if (_float(row, "peak_system_ram_bytes") or 0) <= 0:
            errors.append(f"{workload} has no measured host RAM")
        if (_float(row, "output_token_count") or 0) <= 0:
            errors.append(f"{workload} has no generated tokens")
        if (_float(row, "time_to_first_token_ms") or 0) <= 0 and workload != "mixed":
            errors.append(f"{workload} has no measured TTFT")
    if (
        by_workload["prefill_8k"]
        and (_float(by_workload["prefill_8k"][0], "prompt_token_count_min") or 0) < 7_500
    ):
        errors.append("prefill_8k did not use an approximately 8,000-token prompt")
    if (
        by_workload["prefill_32k"]
        and (_float(by_workload["prefill_32k"][0], "prompt_token_count_min") or 0) < 30_000
    ):
        errors.append("prefill_32k did not use an approximately 32,000-token prompt")
    if (
        by_workload["mixed"]
        and (_float(by_workload["mixed"][0], "measurement_window_seconds") or 0) < 120
    ):
        errors.append("mixed workload did not complete the configured 120-second measurement")

    for workload in REQUIRED_WORKLOADS:
        generations = tokens.get(f"A:{workload}") if isinstance(tokens, dict) else None
        if not isinstance(generations, list) or not generations:
            errors.append(f"{workload} has no raw generation receipt")
            continue
        if not all(item.get("success") and item.get("output_token_ids") for item in generations):
            errors.append(f"{workload} has failed or tokenless raw generation receipts")
        if not any(str(item.get("content") or "").strip() for item in generations):
            errors.append(f"{workload} has no real generated text")

    if not backend_probe.get("gpu_layer_offload_capability"):
        errors.append("llama.cpp backend lacks GPU-layer-offload capability")
    if not backend_probe.get("cuda_available"):
        errors.append("CUDA was unavailable to the llama.cpp Gate 17 backend")
    configured_release = backend_acquisition.get("release_tag") or backend_acquisition.get(
        "configured_release_tag"
    )
    if configured_release != LLAMA_CPP_RELEASE:
        errors.append("llama.cpp backend receipt is not pinned to b9637")
    if not backend_acquisition.get("sha256"):
        errors.append("llama.cpp executable SHA-256 is absent")
    if not backend_acquisition.get("path"):
        errors.append("llama.cpp executable path is absent")
    host_model_bytes = float(residency.get("backend_reported_cpu_model_bytes") or 0)
    if not residency.get("system_ram_contributes") and host_model_bytes <= 0:
        errors.append("backend evidence does not show a nonzero host-memory model contribution")
    if float(residency.get("backend_reported_gpu_model_bytes") or 0) <= 0:
        errors.append("backend evidence does not show a nonzero GPU model contribution")

    acquisition_checks = resolved.get("acquisition_checks") or {}
    if resolved.get("source") == "huggingface-hub":
        if not acquisition_checks.get("huggingface_connectivity"):
            errors.append("Hugging Face connectivity was not proven")
        if not acquisition_checks.get("public_repository_accessible_without_token"):
            errors.append("public repository access without a token was not proven")
        if not acquisition_checks.get("resumable_cache_ready"):
            errors.append("resumable Hugging Face cache was not proven")
        if not (acquisition_checks.get("cache_volume") or {}).get("local"):
            errors.append("Hugging Face cache was not on a local volume")
    elif resolved.get("source") != "user-supplied-local-file":
        errors.append(
            "model acquisition source is neither huggingface-hub nor the supplied local file"
        )

    workload_results = []
    for row in selected_rows:
        workload_results.append(
            {
                "workload": row.get("workload"),
                "status": row.get("status"),
                "evidence_category": row.get("evidence_category"),
                "model_file_sha256": row.get("model_file_sha256"),
                "prompt_token_count_min": row.get("prompt_token_count_min"),
                "prompt_token_count_max": row.get("prompt_token_count_max"),
                "output_token_count": row.get("output_token_count"),
                "decode_tokens_per_second": row.get("decode_tokens_per_second"),
                "decode_tokens_per_second_p95": row.get("decode_tokens_per_second_p95"),
                "time_to_first_token_ms": row.get("time_to_first_token_ms"),
                "time_to_first_token_p95_ms": row.get("time_to_first_token_p95_ms"),
                "prefill_tokens_per_second": row.get("prefill_tokens_per_second"),
                "interactive_tokens_per_second": row.get("interactive_tokens_per_second"),
                "background_tokens_per_second": row.get("background_tokens_per_second"),
                "combined_generated_tokens_per_second": row.get(
                    "combined_generated_tokens_per_second"
                ),
                "mixed_verified_tokens_per_second": row.get("mixed_verified_tokens_per_second"),
                "interactive_p95_latency_ms": row.get("interactive_p95_latency_ms"),
                "measurement_window_seconds": row.get("measurement_window_seconds"),
                "peak_vram_bytes": row.get("peak_vram_bytes"),
                "peak_system_ram_bytes": row.get("peak_system_ram_bytes"),
                "backend_arguments": _json_cell(row, "backend_arguments"),
                "backend_command": _json_cell(row, "backend_command"),
            }
        )

    result = {
        "gate_id": 17,
        "gate_name": "current Level B over-VRAM workload",
        "passed": not errors,
        "model_repository": repository or "",
        "model_revision": revision or "",
        "model_filename": filename or "",
        "model_sha256": model_sha256 or "",
        "model_file_bytes": model_file_bytes,
        "tensor_bytes": tensor_bytes,
        "physical_vram_bytes": physical_vram_bytes,
        "available_system_ram_bytes": int(preflight.get("system_ram_available_bytes") or 0),
        "completed_workloads": completed,
        "required_workloads": list(REQUIRED_WORKLOADS),
        "benchmark_rows": len(current_rows),
        "synthetic_rows_used": 0,
        "historical_rows_used": 0,
        "synthetic_candidate_rows_rejected": len(synthetic_candidates),
        "historical_candidate_rows_rejected": len(historical_candidates),
        "benchmark_results_sha256": (
            _sha256(bundle / "benchmark_results.csv")
            if (bundle / "benchmark_results.csv").is_file()
            else None
        ),
        "model_source": resolved.get("source"),
        "model_cache_path": resolved.get("cache_path"),
        "backend_source": backend_acquisition.get("source"),
        "backend_release_tag": configured_release,
        "backend_executable": backend_acquisition.get("path"),
        "backend_sha256": backend_acquisition.get("sha256"),
        "backend_gpu_model_bytes": residency.get("backend_reported_gpu_model_bytes"),
        "backend_host_model_bytes": residency.get("backend_reported_cpu_model_bytes"),
        "workload_results": workload_results,
        "errors": errors,
    }
    if write_result:
        _write_json(result_path, result)
    if errors:
        raise Gate17ValidationError(errors, result=result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Experiment 010 Level B Gate 17")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--repository-root", type=Path, required=True)
    preflight.add_argument("--work-directory", type=Path, required=True)
    preflight.add_argument("--final-bundle", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--level-b-root", type=Path, required=True)
    validate.add_argument("--config", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(arguments)
    try:
        if namespace.command == "preflight":
            payload = validate_existing_correction_evidence(
                namespace.repository_root,
                namespace.work_directory,
                final_bundle=namespace.final_bundle,
            )
        else:
            payload = validate_level_b_bundle(
                namespace.level_b_root,
                config_path=namespace.config,
                write_result=True,
            )
    except (FileNotFoundError, Gate17ValidationError, RuntimeError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
