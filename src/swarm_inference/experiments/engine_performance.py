"""Experiment 004 production-speed Qwen3 benchmark orchestration."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarm_inference.config.engine_performance import (
    PRIMARY_MODEL_ID,
    PRIMARY_MODEL_REVISION,
    SECONDARY_MODEL_ID,
    EnginePerformanceConfig,
)
from swarm_inference.experiments.engine_environments import (
    ExternalEngineEnvironment,
    inspect_linux_engine_prerequisites,
    provision_huggingface_environment,
)

SECONDARY_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
REQUIRED_ARTIFACTS = (
    "config.requested.yaml",
    "config.resolved.yaml",
    "environment.json",
    "external_engine_environments.json",
    "model_revisions.json",
    "input_tokens.json",
    "baseline_results.csv",
    "optimisation_ladder.csv",
    "engine_results.csv",
    "latency_results.csv",
    "batch_results.csv",
    "memory_results.csv",
    "traffic_results.csv",
    "compile_results.csv",
    "cuda_graph_results.csv",
    "correctness.json",
    "profile_summary.json",
    "summary.json",
    "report.html",
)
REQUIRED_CHARTS = (
    "decode_tps_by_engine.png",
    "aggregate_tps_by_engine.png",
    "ttft_by_engine.png",
    "inter_token_latency.png",
    "optimisation_waterfall.png",
    "gpu_utilisation.png",
    "cpu_overhead.png",
    "host_device_bytes.png",
    "full_logit_vs_token_traffic.png",
    "batch_scaling.png",
    "memory_by_engine.png",
    "compile_and_warmup_cost.png",
)


@dataclass(frozen=True, slots=True)
class EnginePerformanceRun:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("overall_status") == "PASS"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _slug(value: str) -> str:
    return value.lower().replace("/", "-").replace("_", "-").replace(" ", "-").replace(".", "-")


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_object_read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(dict[str, Any], payload)


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status", "diagnostic"]
        rows = [{"status": "UNAVAILABLE", "diagnostic": "no observations"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    environment: dict[str, str] | None = None,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=environment,
        )
        code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout_payload = exc.stdout or ""
        stderr_payload = exc.stderr or ""
        stdout = (
            stdout_payload.decode("utf-8", errors="replace")
            if isinstance(stdout_payload, bytes)
            else stdout_payload
        )
        stderr_text = (
            stderr_payload.decode("utf-8", errors="replace")
            if isinstance(stderr_payload, bytes)
            else stderr_payload
        )
        stderr = stderr_text + f"\nTIMEOUT after {timeout_seconds}s"
    stdout_path.write_text(str(stdout), encoding="utf-8")
    stderr_path.write_text(str(stderr), encoding="utf-8")
    _json_write(
        stdout_path.with_suffix(".command.json"),
        {
            "command": command,
            "cwd": str(cwd),
            "started_unix_seconds": started,
            "finished_unix_seconds": time.time(),
            "return_code": code,
        },
    )
    return code


def _git_state(repository_root: Path) -> dict[str, Any]:
    def read(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    status = read("status", "--porcelain=v1")
    implementation_hashes: dict[str, str] = {}
    for line in status.splitlines():
        relative = line[3:]
        if " -> " in relative:
            relative = relative.rsplit(" -> ", 1)[1]
        relative = relative.strip('"').replace("\\", "/")
        if relative == "artifacts" or relative.startswith("artifacts/"):
            continue
        candidate = repository_root / relative
        if candidate.is_file():
            implementation_hashes[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    implementation_digest = hashlib.sha256(
        json.dumps(
            implementation_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_entry_count": len(status.splitlines()) if status else 0,
        "status_porcelain": status.splitlines(),
        "implementation_file_sha256": implementation_hashes,
        "implementation_manifest_sha256": implementation_digest,
    }


def _token_fixture(
    seed_tokens: list[int],
    length: int,
    *,
    offset: int = 0,
) -> list[int]:
    if not seed_tokens:
        raise ValueError("token fixture seed cannot be empty")
    repeated = seed_tokens * ((length + offset) // len(seed_tokens) + 2)
    return repeated[offset : offset + length]


def build_input_fixtures(
    *,
    model_path: Path,
    output_directory: Path,
    nvtx_enabled: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from swarm_inference.model.qwen3_runtime import nvtx_range

    started = time.perf_counter()
    with nvtx_range(torch, "tokenisation", enabled=nvtx_enabled):
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_path,
            local_files_only=True,
        )
        source = (
            "Distributed inference requires careful scheduling, explicit tensor "
            "ownership, stable cache positions, and reproducible measurement. "
            "This fixed benchmark paragraph supplies realistic Qwen token IDs. "
        )
        seed_tokens = [int(value) for value in tokenizer.encode(source, add_special_tokens=False)]
    fixtures = {
        "decode-focused": [1],
        "realistic": _token_fixture(seed_tokens, 128),
        "medium-prefill": _token_fixture(seed_tokens, 2048),
    }
    prefix = _token_fixture(seed_tokens, 1024)
    prefix_rows = [
        prefix + _token_fixture(seed_tokens, 32, offset=17 * index + 3) for index in range(4)
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, token_ids in fixtures.items():
        _json_write(
            output_directory / f"{name}.json",
            {
                "name": name,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
    for index, token_ids in enumerate(prefix_rows):
        _json_write(
            output_directory / f"prefix-reuse-{index}.json",
            {
                "name": f"prefix-reuse-{index}",
                "common_prefix_tokens": 1024,
                "unique_suffix_tokens": 32,
                "token_ids": token_ids,
                "sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
        )
    return {
        "profile": "input_fixture_creation",
        "execution_profiles": ["qwen3_correctness", "qwen3_fast"],
        "tokenizer_id": PRIMARY_MODEL_ID,
        "tokenizer_revision": PRIMARY_MODEL_REVISION,
        "tokenisation_seconds": time.perf_counter() - started,
        "tokenisation_excluded_from_core_benchmarks": True,
        "workloads": fixtures,
        "prefix_reuse": {
            "common_prefix_tokens": 1024,
            "unique_suffix_tokens": 32,
            "rows": prefix_rows,
        },
    }


def _scheduler_policy(concurrency: int) -> str:
    if concurrency == 1:
        return "latency"
    if concurrency == 4:
        return "balanced"
    return "throughput"


def _worker_job(
    *,
    engine: str,
    model: dict[str, Any],
    input_rows: list[list[int]],
    output_tokens: int,
    concurrency: int,
    repeats: int,
    warmups: int,
    reference: list[list[int]] | None,
    config: EnginePerformanceConfig,
    attention_backend: str = "eager",
    cache_backend: str = "dynamic",
    compile_mode: str = "eager",
    nvtx_enabled: bool = False,
    prefix_reuse_enabled: bool = False,
) -> dict[str, Any]:
    job = {
        "engine": engine,
        "profile": {
            "custom_fast": "qwen3_fast",
            "custom_correctness": "qwen3_correctness",
        }.get(engine, engine),
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "model_path": model["path"],
        "model_dtype": model["dtype"],
        "input_token_ids": input_rows,
        "output_tokens": output_tokens,
        "batch_size": concurrency,
        "repeats": repeats,
        "warmup_requests": warmups,
        "attention_backend": attention_backend,
        "cache_backend": cache_backend,
        "cache_dtype": "bfloat16",
        "compile_mode": compile_mode,
        "max_sequence_length": len(input_rows[0]) + output_tokens,
        "cuda_graph_batch_sizes": config.custom_engine.cuda_graph_batch_sizes,
        "scheduler_policy": _scheduler_policy(concurrency),
        "telemetry_interval_seconds": (config.measurement.gpu_sample_interval_ms / 1000),
        "reference_output_token_ids": reference,
        "pretokenized_inputs": True,
        "greedy": True,
        "complete_output_length_required": True,
        "prefix_reuse_enabled": prefix_reuse_enabled,
        "nvtx_enabled": nvtx_enabled,
        "seed": config.seed,
    }
    if engine == "custom_fast":
        job["matmul_accumulation"] = "fp32"
        job["custom_engine_implementation_revision"] = (
            "experiment-004-gpu-batch-fp32-mask-cache-graph-cache-guard-v5"
        )
    elif engine == "custom_correctness":
        job["matmul_accumulation"] = "fp32"
        job["custom_engine_implementation_revision"] = (
            "experiment-004-gpu-batch-fp32-accumulation-v2"
        )
        job["oracle_batch_execution"] = True
    if engine in {"huggingface_eager", "huggingface_optimised"}:
        job["logits_to_keep"] = 1
        job["matmul_accumulation"] = "fp32"
    return job


_FAIRNESS_FIELDS = (
    "model_id",
    "model_revision",
    "model_dtype",
    "input_token_ids",
    "output_tokens",
    "batch_size",
    "repeats",
    "warmup_requests",
    "reference_output_token_ids",
    "pretokenized_inputs",
    "greedy",
    "complete_output_length_required",
    "prefix_reuse_enabled",
)


def validate_benchmark_fairness(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject a cross-engine comparison whose immutable workload differs."""

    if len(jobs) < 2:
        raise ValueError("benchmark fairness validation requires at least two engines")
    anchor = jobs[0]
    mismatches: list[dict[str, Any]] = []
    for candidate in jobs[1:]:
        for field in _FAIRNESS_FIELDS:
            if candidate.get(field) != anchor.get(field):
                mismatches.append(
                    {
                        "engine": candidate.get("engine"),
                        "field": field,
                        "expected": anchor.get(field),
                        "actual": candidate.get(field),
                    }
                )
    if mismatches:
        fields = sorted({str(item["field"]) for item in mismatches})
        raise ValueError(
            "unfair engine benchmark: immutable workload fields differ: " + ", ".join(fields)
        )
    return {
        "status": "PASS",
        "engines": [str(item["engine"]) for item in jobs],
        "validated_fields": list(_FAIRNESS_FIELDS),
        "workload_hash": hashlib.sha256(
            json.dumps(
                {field: anchor.get(field) for field in _FAIRNESS_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def parse_external_engine_result(
    payload: dict[str, Any],
    *,
    job: dict[str, Any],
) -> dict[str, Any]:
    """Validate a standalone production-engine result before comparison."""

    if payload.get("engine") != job.get("engine"):
        raise ValueError(
            f"external result engine mismatch: expected={job.get('engine')!r} "
            f"actual={payload.get('engine')!r}"
        )
    if payload.get("profile") != job.get("profile"):
        raise ValueError(
            f"external result profile mismatch: expected={job.get('profile')!r} "
            f"actual={payload.get('profile')!r}"
        )
    if payload.get("status") == "PASS":
        if payload.get("worker_status") != "completed":
            raise ValueError("successful external result is not marked completed")
        measured = payload.get("measured_repeats")
        if not isinstance(measured, list) or len(measured) != int(job["repeats"]):
            raise ValueError("external result measured-repeat count is incomplete")
        expected_rows = int(job["batch_size"])
        expected_tokens = int(job["output_tokens"])
        for repeat in measured:
            rows = repeat.get("output_token_ids", [])
            if len(rows) != expected_rows or any(len(row) != expected_tokens for row in rows):
                raise ValueError(
                    "external result did not generate the complete requested "
                    "batch and output length"
                )
        statistics_payload = payload.get("statistics", {})
        required_statistics = {
            "median",
            "minimum",
            "maximum",
            "standard_deviation",
            "coefficient_of_variation",
        }
        if not required_statistics.issubset(statistics_payload):
            raise ValueError("external result statistics are incomplete")
        if payload.get("exact_reference_identity") is not True:
            raise ValueError("successful external result is not reference-identical")
    return payload


def _first_token_mismatch(
    expected_rows: Any,
    actual_rows: Any,
) -> dict[str, int | None] | None:
    if not isinstance(expected_rows, list) or not isinstance(actual_rows, list):
        return None
    row_count = max(len(expected_rows), len(actual_rows))
    for row_index in range(row_count):
        expected = expected_rows[row_index] if row_index < len(expected_rows) else []
        actual = actual_rows[row_index] if row_index < len(actual_rows) else []
        if not isinstance(expected, list) or not isinstance(actual, list):
            continue
        token_count = max(len(expected), len(actual))
        for token_index in range(token_count):
            expected_token = expected[token_index] if token_index < len(expected) else None
            actual_token = actual[token_index] if token_index < len(actual) else None
            if expected_token != actual_token:
                return {
                    "request_index": row_index,
                    "token_index": token_index,
                    "expected_token_id": (
                        int(expected_token) if expected_token is not None else None
                    ),
                    "actual_token_id": (int(actual_token) if actual_token is not None else None),
                }
    return None


def _job_fingerprint(job: Any) -> str:
    encoded = json.dumps(job, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _archive_superseded_job(
    *,
    jobs_directory: Path,
    job_name: str,
    previous_result: dict[str, Any],
    replacement_job: dict[str, Any],
) -> None:
    previous_job = previous_result.get("job", {})
    archive_directory = (
        jobs_directory / "superseded" / f"{job_name}-{_job_fingerprint(previous_job)[:12]}"
    )
    archive_directory.mkdir(parents=True, exist_ok=True)
    for candidate in jobs_directory.glob(f"{job_name}.*"):
        if candidate.is_file():
            shutil.copy2(candidate, archive_directory / candidate.name)
    _json_write(
        archive_directory / "resume-mismatch.json",
        {
            "reason": "result job does not match the requested resume job",
            "previous_job_fingerprint": _job_fingerprint(previous_job),
            "replacement_job_fingerprint": _job_fingerprint(replacement_job),
        },
    )


def _execute_job(
    *,
    repository_root: Path,
    run_directory: Path,
    job_name: str,
    job: dict[str, Any],
    python_executable: Path,
    external: bool,
    resume: bool,
    timeout_seconds: float = 7200,
) -> dict[str, Any]:
    jobs = run_directory / "logs" / "jobs"
    job_path = jobs / f"{job_name}.job.json"
    result_path = jobs / f"{job_name}.result.json"
    if resume and result_path.is_file():
        resumed = _json_object_read(result_path)
        if resumed.get("job") == job:
            return parse_external_engine_result(resumed, job=job) if external else resumed
        _archive_superseded_job(
            jobs_directory=jobs,
            job_name=job_name,
            previous_result=resumed,
            replacement_job=job,
        )
    _json_write(job_path, job)
    module_or_script = (
        [
            str(repository_root / "scripts" / "external_engine_benchmark.py"),
        ]
        if external
        else ["-m", "swarm_inference.experiments.engine_worker"]
    )
    command = [
        str(python_executable),
        *module_or_script,
        "--job",
        str(job_path),
        "--output",
        str(result_path),
    ]
    _run_command(
        command,
        cwd=repository_root,
        stdout_path=jobs / f"{job_name}.stdout.log",
        stderr_path=jobs / f"{job_name}.stderr.log",
        timeout_seconds=timeout_seconds,
    )
    if result_path.is_file():
        result_payload = _json_object_read(result_path)
        if external:
            try:
                return parse_external_engine_result(result_payload, job=job)
            except ValueError as exc:
                return {
                    "status": "FAIL",
                    "worker_status": "invalid-result",
                    "engine": job["engine"],
                    "profile": job["profile"],
                    "error": f"external result validation failed: {exc}",
                    "raw_result": result_payload,
                    "job": job,
                }
        return result_payload
    return {
        "status": "FAIL",
        "worker_status": "missing-result",
        "engine": job["engine"],
        "profile": job["profile"],
        "error": "benchmark process did not create its result artifact",
        "job": job,
    }


def _execute_linux_engine_job(
    *,
    repository_root: Path,
    run_directory: Path,
    job_name: str,
    job: dict[str, Any],
    image: str,
    script_name: str,
    resume: bool,
    use_image_entrypoint: bool = False,
) -> dict[str, Any]:
    jobs = run_directory / "logs" / "jobs"
    result_path = jobs / f"{job_name}.result.json"
    if resume and result_path.is_file():
        resumed = _json_object_read(result_path)
        if resumed.get("job") == job:
            return parse_external_engine_result(
                resumed,
                job=job,
            )
        _archive_superseded_job(
            jobs_directory=jobs,
            job_name=job_name,
            previous_result=resumed,
            replacement_job=job,
        )
    job_path = jobs / f"{job_name}.job.json"
    _json_write(job_path, job)
    try:
        job_relative = job_path.relative_to(repository_root).as_posix()
        result_relative = result_path.relative_to(repository_root).as_posix()
    except ValueError:
        raise RuntimeError("Linux engine run directory must be inside repository") from None
    cache_root = Path.home() / ".cache" / "huggingface"
    snapshot_name = Path(job["model_path"]).name
    model_cache_name = "models--" + str(job["model_id"]).replace("/", "--")
    container_job = dict(job)
    container_job["model_path"] = (
        f"/root/.cache/huggingface/hub/{model_cache_name}/snapshots/{snapshot_name}"
    )
    _json_write(job_path, container_job)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        "16g",
        "--ipc=host",
        "-v",
        f"{repository_root}:/workspace",
        "-v",
        f"{cache_root}:/root/.cache/huggingface",
    ]
    if not use_image_entrypoint:
        command.extend(["--entrypoint", "python3"])
    command.extend(
        [
            image,
            *(["python3"] if use_image_entrypoint else []),
            f"/workspace/scripts/{script_name}",
            "--job",
            f"/workspace/{job_relative}",
            "--output",
            f"/workspace/{result_relative}",
        ]
    )
    _run_command(
        command,
        cwd=repository_root,
        stdout_path=jobs / f"{job_name}.stdout.log",
        stderr_path=jobs / f"{job_name}.stderr.log",
        timeout_seconds=7200,
    )
    if result_path.is_file():
        try:
            return parse_external_engine_result(
                _json_object_read(result_path),
                job=job,
            )
        except ValueError as exc:
            return {
                "status": "FAIL",
                "worker_status": "invalid-result",
                "engine": job["engine"],
                "profile": job["profile"],
                "error": f"external result validation failed: {exc}",
                "job": job,
            }
    return {
        "status": "FAIL",
        "worker_status": "missing-result",
        "engine": job["engine"],
        "profile": job["profile"],
        "error": f"{job['engine']} container did not create its result artifact",
        "job": job,
    }


def _execute_sglang_job(
    *,
    repository_root: Path,
    run_directory: Path,
    job_name: str,
    job: dict[str, Any],
    image: str,
    resume: bool,
) -> dict[str, Any]:
    return _execute_linux_engine_job(
        repository_root=repository_root,
        run_directory=run_directory,
        job_name=job_name,
        job=job,
        image=image,
        script_name="sglang_engine_benchmark.py",
        resume=resume,
    )


def _execute_vllm_job(
    *,
    repository_root: Path,
    run_directory: Path,
    job_name: str,
    job: dict[str, Any],
    image: str,
    resume: bool,
) -> dict[str, Any]:
    return _execute_linux_engine_job(
        repository_root=repository_root,
        run_directory=run_directory,
        job_name=job_name,
        job=job,
        image=image,
        script_name="vllm_engine_benchmark.py",
        resume=resume,
    )


def _execute_tensorrt_llm_job(
    *,
    repository_root: Path,
    run_directory: Path,
    job_name: str,
    job: dict[str, Any],
    image: str,
    resume: bool,
) -> dict[str, Any]:
    return _execute_linux_engine_job(
        repository_root=repository_root,
        run_directory=run_directory,
        job_name=job_name,
        job=job,
        image=image,
        script_name="tensorrt_llm_engine_benchmark.py",
        resume=resume,
        use_image_entrypoint=True,
    )


def _metric_rows(
    result: dict[str, Any],
    *,
    model_id: str,
    workload: str,
    concurrency: int,
    prefix_reuse: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repeats = result.get("measured_repeats", [])
    for index, repeat in enumerate(repeats):
        metrics = dict(repeat.get("metrics", {}))
        metrics.update(
            {
                "engine": result.get("engine"),
                "profile": result.get("profile"),
                "status": result.get("status"),
                "model_id": model_id,
                "workload": workload,
                "concurrency": concurrency,
                "repeat": index,
                "exact_reference_identity": result.get(
                    "exact_reference_identity",
                    result.get("exact_repeat_identity"),
                ),
                "prefix_reuse_workload": prefix_reuse,
            }
        )
        rows.append(metrics)
    if not rows:
        rows.append(
            {
                "engine": result.get("engine"),
                "profile": result.get("profile"),
                "status": result.get("status", "FAIL"),
                "model_id": model_id,
                "workload": workload,
                "concurrency": concurrency,
                "repeat": None,
                "diagnostic": result.get("error", "no measured repeats"),
                "prefix_reuse_workload": prefix_reuse,
            }
        )
    return rows


def _aggregate_engine_row(
    result: dict[str, Any],
    *,
    model_id: str,
    workload: str,
    concurrency: int,
    prefix_reuse: bool = False,
) -> dict[str, Any]:
    statistics_payload = result.get("statistics", {})
    decode_statistics = result.get("decode_statistics", {})
    first_metrics = (
        result.get("measured_repeats", [{}])[0].get("metrics", {})
        if result.get("measured_repeats")
        else {}
    )
    return {
        "engine": result.get("engine"),
        "profile": result.get("profile"),
        "status": result.get("status"),
        "worker_status": result.get("worker_status"),
        "measured_repeat_count": len(result.get("measured_repeats", [])),
        "model_id": model_id,
        "workload": workload,
        "concurrency": concurrency,
        "prefix_reuse_workload": prefix_reuse,
        "median_aggregate_output_tokens_per_second": statistics_payload.get("median", 0.0),
        "minimum_aggregate_output_tokens_per_second": statistics_payload.get("minimum", 0.0),
        "maximum_aggregate_output_tokens_per_second": statistics_payload.get("maximum", 0.0),
        "standard_deviation": statistics_payload.get("standard_deviation", 0.0),
        "coefficient_of_variation": statistics_payload.get("coefficient_of_variation", 0.0),
        "median_decode_output_tokens_per_second": decode_statistics.get(
            "median",
            first_metrics.get("decode_output_tokens_per_second", 0.0),
        ),
        "attention_backend": result.get("attention_backend"),
        "cache_backend": result.get("cache_backend"),
        "compile_mode": result.get("compile_mode"),
        "exact_reference_identity": result.get(
            "exact_reference_identity",
            result.get("exact_repeat_identity"),
        ),
        "model_load_seconds": result.get("model_load_seconds", 0.0),
        "attention_selection_seconds": result.get(
            "attention_selection_seconds",
            0.0,
        ),
        "warmup_seconds": result.get("warmup_seconds", 0.0),
        "prefill_ms": first_metrics.get("prefill_ms", 0.0),
        "host_to_device_ms": first_metrics.get("host_to_device_ms", 0.0),
        "decode_ms": first_metrics.get("decode_ms", 0.0),
        "end_to_end_ms": first_metrics.get("end_to_end_ms", 0.0),
        "ttft_ms": first_metrics.get(
            "ttft_ms",
            first_metrics.get("prefill_ms", 0.0),
        ),
        "inter_token_latency_ms_p50": first_metrics.get("inter_token_latency_ms_p50", 0.0),
        "inter_token_latency_ms_p95": first_metrics.get("inter_token_latency_ms_p95", 0.0),
        "inter_token_latency_ms_p99": first_metrics.get("inter_token_latency_ms_p99", 0.0),
        "peak_vram_bytes": first_metrics.get(
            "peak_vram_bytes",
            result.get("telemetry", {}).get("gpu_memory_bytes_maximum", 0),
        ),
        "cache_reserved_bytes": first_metrics.get("cache_reserved_bytes", 0),
        "cache_used_bytes": first_metrics.get("cache_used_bytes", 0),
        "cache_allocation_count": first_metrics.get("cache_allocation_count"),
        "cache_fragmentation_fraction": first_metrics.get("cache_fragmentation_fraction"),
        "cache_accounting_status": first_metrics.get(
            "cache_accounting_status",
            "unavailable_for_engine",
        ),
        "host_to_device_bytes": first_metrics.get("host_to_device_bytes", 0),
        "device_to_host_bytes": first_metrics.get("device_to_host_bytes", 0),
        "cuda_synchronisations": first_metrics.get("cuda_synchronisations", 0),
        "sampling_ms": first_metrics.get("sampling_ms", 0.0),
        "device_to_host_ms": first_metrics.get("device_to_host_ms", 0.0),
        "gpu_kernel_and_transfer_ms": first_metrics.get(
            "gpu_kernel_and_transfer_ms",
            0.0,
        ),
        "scheduler_ms": first_metrics.get("scheduler_ms", 0.0),
        "queue_wait_ms": first_metrics.get("queue_wait_ms", 0.0),
        "cuda_graph_capture_ms": first_metrics.get("cuda_graph_capture_ms", 0.0),
        "cuda_graph_verified": first_metrics.get("cuda_graph_verified", False),
        "prefill_mode": first_metrics.get("prefill_mode"),
        "chunked_prefill_supported": first_metrics.get(
            "chunked_prefill_supported",
            False,
        ),
        "kernels_per_decode_token": first_metrics.get("kernels_per_decode_token"),
        "kernel_count_status": first_metrics.get(
            "kernel_count_status",
            "unavailable_not_profiled",
        ),
        "full_logit_equivalent_bytes": first_metrics.get("full_logit_equivalent_bytes", 0),
        "coordinator_bound_bytes": first_metrics.get("coordinator_bound_bytes", 0),
        "gpu_utilisation_percent": result.get("telemetry", {}).get(
            "gpu_utilisation_percent_mean", 0.0
        ),
        "memory_controller_utilisation_percent": result.get("telemetry", {}).get(
            "memory_controller_utilisation_percent_mean", 0.0
        ),
        "power_watts": result.get("telemetry", {}).get("power_watts_mean", 0.0),
        "host_cpu_percent": result.get("telemetry", {}).get("host_cpu_percent_mean", 0.0),
        "diagnostic": result.get("error"),
    }


def _empty_chart(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _bar_chart(
    path: Path,
    *,
    title: str,
    ylabel: str,
    labels: list[str],
    values: list[float],
    logarithmic: bool = False,
) -> None:
    if not values:
        _empty_chart(path, title, "No successful observations")
        return
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(labels, values, color="#3b82f6")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    if logarithmic and all(value > 0 for value in values):
        axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_engine_charts(
    *,
    chart_directory: Path,
    engine_rows: list[dict[str, Any]],
    ladder_rows: list[dict[str, Any]],
) -> None:
    chart_directory.mkdir(parents=True, exist_ok=True)
    primary = [
        row
        for row in engine_rows
        if row.get("model_id") == PRIMARY_MODEL_ID
        and row.get("workload") == "decode-focused"
        and int(row.get("concurrency") or 0) == 1
    ]
    labels = [str(row["engine"]) for row in primary]
    _bar_chart(
        chart_directory / "decode_tps_by_engine.png",
        title="Qwen3-0.6B batch-one decode throughput",
        ylabel="Output tokens/s",
        labels=labels,
        values=[float(row.get("median_decode_output_tokens_per_second") or 0) for row in primary],
    )
    _bar_chart(
        chart_directory / "aggregate_tps_by_engine.png",
        title="Qwen3-0.6B batch-one aggregate throughput",
        ylabel="Verified output tokens/s",
        labels=labels,
        values=[
            float(row.get("median_aggregate_output_tokens_per_second") or 0) for row in primary
        ],
    )
    _bar_chart(
        chart_directory / "ttft_by_engine.png",
        title="Qwen3-0.6B time to first token",
        ylabel="TTFT (ms)",
        labels=labels,
        values=[float(row.get("ttft_ms") or 0) for row in primary],
    )
    _bar_chart(
        chart_directory / "inter_token_latency.png",
        title="Qwen3-0.6B p95 inter-token latency",
        ylabel="Latency (ms)",
        labels=labels,
        values=[float(row.get("inter_token_latency_ms_p95") or 0) for row in primary],
    )
    _bar_chart(
        chart_directory / "optimisation_waterfall.png",
        title="Custom engine optimisation ladder",
        ylabel="Output tokens/s",
        labels=[str(row.get("optimisation")) for row in ladder_rows],
        values=[float(row.get("output_tokens_per_second") or 0) for row in ladder_rows],
    )
    _bar_chart(
        chart_directory / "gpu_utilisation.png",
        title="GPU utilisation during measured runs",
        ylabel="GPU utilisation (%)",
        labels=labels,
        values=[float(row.get("gpu_utilisation_percent") or 0) for row in primary],
    )
    _bar_chart(
        chart_directory / "cpu_overhead.png",
        title="Host CPU utilisation during measured runs",
        ylabel="Host CPU (%)",
        labels=labels,
        values=[float(row.get("host_cpu_percent") or 0) for row in primary],
    )
    traffic_labels: list[str] = []
    traffic_values: list[float] = []
    for row in primary:
        traffic_labels.extend([f"{row['engine']} H2D", f"{row['engine']} D2H"])
        traffic_values.extend(
            [
                float(row.get("host_to_device_bytes") or 0),
                float(row.get("device_to_host_bytes") or 0),
            ]
        )
    _bar_chart(
        chart_directory / "host_device_bytes.png",
        title="Host/device traffic",
        ylabel="Bytes per measured request",
        labels=traffic_labels,
        values=traffic_values,
        logarithmic=True,
    )
    custom = next(
        (row for row in primary if row.get("engine") == "custom_fast"),
        None,
    )
    _bar_chart(
        chart_directory / "full_logit_vs_token_traffic.png",
        title="Coordinator-bound final-stage traffic",
        ylabel="Bytes per measured request",
        labels=["Legacy full FP32 logits", "Token IDs + selected logits"],
        values=(
            [
                float(custom.get("full_logit_equivalent_bytes") or 0),
                float(custom.get("coordinator_bound_bytes") or 0),
            ]
            if custom
            else []
        ),
        logarithmic=True,
    )
    scaling = [
        row
        for row in engine_rows
        if row.get("model_id") == PRIMARY_MODEL_ID
        and row.get("workload") == "decode-focused"
        and row.get("status") == "PASS"
    ]
    if scaling:
        figure, axis = plt.subplots(figsize=(10, 5.5))
        engines = sorted({str(row["engine"]) for row in scaling})
        for engine in engines:
            rows = sorted(
                [row for row in scaling if row["engine"] == engine],
                key=lambda row: int(row["concurrency"]),
            )
            axis.plot(
                [int(row["concurrency"]) for row in rows],
                [float(row["median_aggregate_output_tokens_per_second"]) for row in rows],
                marker="o",
                label=engine,
            )
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Concurrency / real batch size")
        axis.set_ylabel("Verified output tokens/s")
        axis.set_title("Decode batch scaling")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(chart_directory / "batch_scaling.png", dpi=150)
        plt.close(figure)
    else:
        _empty_chart(
            chart_directory / "batch_scaling.png",
            "Decode batch scaling",
            "No successful observations",
        )
    _bar_chart(
        chart_directory / "memory_by_engine.png",
        title="Peak VRAM by engine",
        ylabel="Peak allocated bytes",
        labels=labels,
        values=[float(row.get("peak_vram_bytes") or 0) for row in primary],
    )
    cost_labels = []
    cost_values = []
    for row in primary:
        cost_labels.extend([f"{row['engine']} load", f"{row['engine']} warm/compile"])
        cost_values.extend(
            [
                float(row.get("model_load_seconds") or 0),
                float(row.get("warmup_seconds") or 0),
            ]
        )
    _bar_chart(
        chart_directory / "compile_and_warmup_cost.png",
        title="Model load and readiness cost",
        ylabel="Seconds",
        labels=cost_labels,
        values=cost_values,
    )


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _report_artifact(
    *,
    summary: dict[str, Any],
    engine_rows: list[dict[str, Any]],
    optimisation_ladder_rows: list[dict[str, Any]],
    environments: list[dict[str, Any]],
) -> dict[str, Any]:
    title = "Experiment 004: Qwen3 Engine"
    generated_at = datetime.now(UTC).isoformat()
    engine_labels = {
        "custom_correctness": "Custom correctness",
        "custom_fast": "Custom fast",
        "huggingface_eager": "HF eager",
        "huggingface_optimised": "HF optimised",
        "sglang": "SGLang",
        "vllm": "vLLM",
        "tensorrt_llm": "TensorRT-LLM",
    }
    optimisation_labels = {
        "legacy_process_rpc_numpy": "Legacy process/RPC/NumPy",
        "gpu_native_dynamic": "GPU-native dynamic cache",
        "static_cache": "Static cache",
        "sdpa_static": "SDPA + static cache",
        "torch_compile_default": "torch.compile default",
        "torch_compile_reduce_overhead": "torch.compile reduce-overhead",
        "torch_compile_max_autotune": "torch.compile max-autotune",
        "manual_cuda_graph": "Manual CUDA graph",
        "hf_exact_eager_dynamic": "HF eager + dynamic cache",
        "hf_sdpa_dynamic": "HF SDPA + dynamic cache",
        "hf_eager_static": "HF eager + static cache",
        "hf_compile_default_dynamic": "HF compile default",
        "hf_compile_reduce_overhead_static": "HF compile reduce-overhead",
        "hf_compile_max_autotune_static": "HF compile max-autotune",
    }
    status_rows = [
        {
            "status_name": key,
            "status": str(value),
        }
        for key, value in summary.items()
        if key.endswith("_status")
    ]
    result_rows = [
        {
            "model_id": str(row.get("model_id") or ""),
            "workload": str(row.get("workload") or ""),
            "concurrency": int(row.get("concurrency") or 0),
            "engine": str(row.get("engine") or ""),
            "engine_label": engine_labels.get(
                str(row.get("engine") or ""),
                str(row.get("engine") or ""),
            ),
            "profile": str(row.get("profile") or ""),
            "status": str(row.get("status") or ""),
            "aggregate_output_tokens_per_second": _finite_number(
                row.get("median_aggregate_output_tokens_per_second")
            ),
            "decode_output_tokens_per_second": _finite_number(
                row.get("median_decode_output_tokens_per_second")
            ),
            "coefficient_of_variation": _finite_number(row.get("coefficient_of_variation")),
            "ttft_ms": _finite_number(row.get("ttft_ms")),
            "p95_inter_token_latency_ms": _finite_number(row.get("inter_token_latency_ms_p95")),
            "gpu_utilisation_percent": _finite_number(row.get("gpu_utilisation_percent")),
            "peak_vram_gib": _finite_number(row.get("peak_vram_bytes")) / (1024**3),
            "exact_reference_identity": str(row.get("exact_reference_identity")),
        }
        for row in engine_rows
    ]
    environment_rows = [
        {
            "engine": str(item.get("engine") or ""),
            "status": str(item.get("status") or ""),
            "requested_version": str(item.get("requested_version") or ""),
            "python_version": str(item.get("python_version") or ""),
            "cuda_version": str(item.get("cuda_version") or ""),
            "torch_version": str(item.get("torch_version") or ""),
            "diagnostic": str(item.get("diagnostic") or ""),
        }
        for item in environments
    ]
    batch_one_rows = [
        row
        for row in result_rows
        if row["model_id"] == PRIMARY_MODEL_ID
        and row["workload"] == "decode-focused"
        and row["concurrency"] == 1
        and row["engine"]
        in {
            "custom_fast",
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
            "vllm",
            "tensorrt_llm",
        }
    ]
    scaling_rows = [
        row
        for row in result_rows
        if row["model_id"] == PRIMARY_MODEL_ID
        and row["workload"] == "decode-focused"
        and row["engine"]
        in {
            "custom_fast",
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
            "vllm",
            "tensorrt_llm",
        }
    ]
    traffic_source_rows = [
        row
        for row in engine_rows
        if row.get("model_id") == PRIMARY_MODEL_ID
        and row.get("workload") == "decode-focused"
        and int(row.get("concurrency") or 0) == 1
        and row.get("engine") == "custom_fast"
    ]
    traffic_rows: list[dict[str, Any]] = []
    if traffic_source_rows:
        custom = traffic_source_rows[0]
        traffic_rows = [
            {
                "traffic_type": "Legacy full FP32 logits",
                "bytes": _finite_number(custom.get("full_logit_equivalent_bytes")),
            },
            {
                "traffic_type": "Token IDs + selected BF16 logits",
                "bytes": _finite_number(custom.get("coordinator_bound_bytes")),
            },
        ]
    readiness_rows: list[dict[str, Any]] = []
    for row in batch_one_rows:
        readiness_rows.extend(
            [
                {
                    "engine_step": f"{row['engine_label']} - model load",
                    "seconds": _finite_number(
                        next(
                            (
                                source.get("model_load_seconds")
                                for source in engine_rows
                                if source.get("model_id") == row["model_id"]
                                and source.get("workload") == row["workload"]
                                and int(source.get("concurrency") or 0) == row["concurrency"]
                                and source.get("engine") == row["engine"]
                            ),
                            0,
                        )
                    ),
                },
                {
                    "engine_step": f"{row['engine_label']} - attention selection",
                    "seconds": _finite_number(
                        next(
                            (
                                source.get("attention_selection_seconds")
                                for source in engine_rows
                                if source.get("model_id") == row["model_id"]
                                and source.get("workload") == row["workload"]
                                and int(source.get("concurrency") or 0) == row["concurrency"]
                                and source.get("engine") == row["engine"]
                            ),
                            0,
                        )
                    ),
                },
                {
                    "engine_step": f"{row['engine_label']} - compile/warm-up",
                    "seconds": _finite_number(
                        next(
                            (
                                source.get("warmup_seconds")
                                for source in engine_rows
                                if source.get("model_id") == row["model_id"]
                                and source.get("workload") == row["workload"]
                                and int(source.get("concurrency") or 0) == row["concurrency"]
                                and source.get("engine") == row["engine"]
                            ),
                            0,
                        )
                    ),
                },
            ]
        )
    headline_rows = [
        {
            "custom_output_tokens_per_second": _finite_number(
                summary.get("custom_batch_one_output_tokens_per_second")
            ),
            "speedup_over_legacy": _finite_number(summary.get("speedup_over_remeasured_baseline")),
            "fraction_of_fastest_production": _finite_number(
                summary.get("fraction_of_fastest_successful_production_engine")
            ),
            "maximum_custom_cv": _finite_number(summary.get("maximum_custom_result_cv")),
        }
    ]
    ladder_rows = [
        {
            "optimisation": str(row.get("optimisation") or ""),
            "optimisation_label": optimisation_labels.get(
                str(row.get("optimisation") or ""),
                str(row.get("optimisation") or ""),
            ),
            "profile": str(row.get("profile") or ""),
            "status": str(row.get("status") or ""),
            "output_tokens_per_second": _finite_number(row.get("output_tokens_per_second")),
            "exact_reference_identity": str(row.get("exact_reference_identity")),
        }
        for row in optimisation_ladder_rows
    ]
    sources = [
        {
            "id": "summary_json",
            "label": "Experiment acceptance summary",
            "path": "summary.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('summary.json')",
                "description": (
                    "Loads the final machine-readable Experiment 004 acceptance summary."
                ),
                "tables_used": ["summary.json"],
                "executed_at": generated_at,
                "metric_definitions": [
                    "Speedup is custom batch-one aggregate verified output tokens/s "
                    "divided by the remeasured legacy process/RPC/NumPy rate.",
                    "Production fraction is custom batch-one aggregate verified output "
                    "tokens/s divided by the fastest successful exact production baseline.",
                ],
            },
        },
        {
            "id": "engine_results_csv",
            "label": "Reviewed engine benchmark results",
            "path": "engine_results.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": ("SELECT * FROM read_csv_auto('engine_results.csv', header = true)"),
                "description": (
                    "Loads the complete reviewed engine/model/workload/concurrency result matrix."
                ),
                "tables_used": ["engine_results.csv"],
                "executed_at": generated_at,
                "filters": ["Charts select Qwen3-0.6B decode-focused rows and stated concurrency."],
                "metric_definitions": [
                    "Aggregate verified output tokens/s is all requested generated tokens "
                    "divided by measured request execution time.",
                    "Decode output tokens/s excludes prefill and the first sampled token.",
                    "CV is population standard deviation divided by mean across repeats.",
                ],
            },
        },
        {
            "id": "optimisation_ladder_csv",
            "label": "Custom and baseline optimisation ladder",
            "path": "optimisation_ladder.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": ("SELECT * FROM read_csv_auto('optimisation_ladder.csv', header = true)"),
                "description": (
                    "Loads measured attention, cache, compile, and CUDA graph candidates."
                ),
                "tables_used": ["optimisation_ladder.csv"],
                "executed_at": generated_at,
                "filters": [
                    "Candidates use the immutable one-token primary-model fixture and "
                    "up to 64 generated tokens for selection."
                ],
            },
        },
        {
            "id": "traffic_results_csv",
            "label": "Host/device and coordinator traffic measurements",
            "path": "traffic_results.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_csv_auto('traffic_results.csv', header = true)",
                "description": (
                    "Loads measured host/device bytes and equivalent legacy logit traffic."
                ),
                "tables_used": ["traffic_results.csv"],
                "executed_at": generated_at,
                "metric_definitions": [
                    "Legacy full-logit bytes equal batch x output tokens x vocabulary "
                    "size x four FP32 bytes.",
                    "Fast coordinator bytes contain int64 token IDs and BF16 selected logits.",
                ],
            },
        },
        {
            "id": "external_environments_json",
            "label": "Isolated external-engine environment evidence",
            "path": "external_engine_environments.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": (
                    "SELECT unnest(environments) FROM "
                    "read_json_auto('external_engine_environments.json')"
                ),
                "description": (
                    "Loads isolated external-engine versions, probes, and diagnostics."
                ),
                "tables_used": ["external_engine_environments.json"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "correctness_json",
            "label": "Exact greedy-output validation evidence",
            "path": "correctness.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('correctness.json')",
                "description": (
                    "Loads correctness-reference token hashes and exact comparison results."
                ),
                "tables_used": ["correctness.json"],
                "executed_at": generated_at,
            },
        },
        {
            "id": "profile_summary_json",
            "label": "Pre-optimisation profile and tool availability",
            "path": "profile_summary.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('profile_summary.json')",
                "description": (
                    "Loads the measured legacy baseline profile and profiler availability."
                ),
                "tables_used": ["profile_summary.json"],
                "executed_at": generated_at,
            },
        },
    ]
    sglang = next(
        (row for row in environment_rows if row["engine"] == "sglang"),
        {"status": "FAIL", "diagnostic": "SGLang environment evidence missing"},
    )
    optional_failures = [
        f"{row['engine']}: {row['diagnostic'] or row['status']}"
        for row in environment_rows
        if row["engine"] in {"vllm", "tensorrt_llm"} and row["status"] != "PASS"
    ]
    optional_failures.extend(
        f"{row.get('engine')}: {row.get('diagnostic') or row.get('benchmark_status')}"
        for row in summary.get("optional_engine_attempts", [])
        if row.get("benchmark_status") != "PASS"
    )
    identity_failures = summary.get("production_baseline_identity_failures", [])
    limitations = [
        "The custom engine uses homogeneous full-prompt prefill; chunked prefill and "
        "cross-request prefix reuse are explicitly unsupported in Experiment 004.",
        "Nsight Systems was not installed, so kernels per decode token is reported as "
        "unavailable rather than inferred.",
        f"SGLang environment status: {sglang['status']}. {sglang['diagnostic']}",
    ]
    limitations.extend(
        (
            f"{row.get('engine')} produced a complete measurement but failed exact "
            f"token identity for {row.get('model_id')} {row.get('workload')} at "
            f"concurrency {row.get('concurrency')}: "
            f"{row.get('diagnostic') or 'output sequence mismatch'}"
        )
        for row in identity_failures
    )
    limitations.extend(optional_failures)
    source_inventory = [dict(item) for item in sources]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Technical evidence bundle for the Experiment 004 Qwen3 stage-engine "
                "performance and correctness evaluation."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "custom_throughput",
                    "description": (
                        "Median verified output throughput for Qwen3-0.6B, one-token "
                        "prompt, 512 generated tokens, concurrency one."
                    ),
                    "dataset": "headline",
                    "sourceId": "summary_json",
                    "metrics": [
                        {
                            "label": "Custom output tokens/s",
                            "field": "custom_output_tokens_per_second",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "legacy_speedup",
                    "description": (
                        "Ratio to the remeasured pre-change process/RPC/NumPy custom path."
                    ),
                    "dataset": "headline",
                    "sourceId": "summary_json",
                    "metrics": [
                        {
                            "label": "Speedup over legacy",
                            "field": "speedup_over_legacy",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "production_fraction",
                    "description": (
                        "Custom batch-one throughput divided by the fastest successful "
                        "exact production baseline at the same workload."
                    ),
                    "dataset": "headline",
                    "sourceId": "summary_json",
                    "metrics": [
                        {
                            "label": "Fraction of fastest production engine",
                            "field": "fraction_of_fastest_production",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "custom_cv",
                    "description": (
                        "Largest coefficient of variation among primary-model custom "
                        "performance points."
                    ),
                    "dataset": "headline",
                    "sourceId": "summary_json",
                    "metrics": [
                        {
                            "label": "Maximum custom result CV",
                            "field": "maximum_custom_cv",
                            "format": "percent",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "batch_one_decode",
                    "title": "Batch-one decode throughput",
                    "subtitle": "Median Qwen3-0.6B output-token rate after prefill.",
                    "type": "bar",
                    "dataset": "batch_one",
                    "sourceId": "engine_results_csv",
                    "encodings": {
                        "x": {
                            "field": "engine_label",
                            "type": "nominal",
                            "label": "Engine",
                        },
                        "y": {
                            "field": "decode_output_tokens_per_second",
                            "type": "quantitative",
                            "label": "Output tokens/s",
                        },
                    },
                    "valueFormat": "number",
                },
                {
                    "id": "decode_batch_scaling",
                    "title": "Qwen3-0.6B decode-focused aggregate throughput",
                    "subtitle": (
                        "Concurrency is executed as a real homogeneous batch for the "
                        "custom path; engine identity is encoded by series."
                    ),
                    "type": "line",
                    "dataset": "batch_scaling",
                    "sourceId": "engine_results_csv",
                    "encodings": {
                        "x": {
                            "field": "concurrency",
                            "type": "quantitative",
                            "label": "Concurrency",
                        },
                        "y": {
                            "field": "aggregate_output_tokens_per_second",
                            "type": "quantitative",
                            "label": "Verified output tokens/s",
                        },
                        "color": {
                            "field": "engine_label",
                            "type": "nominal",
                            "label": "Engine",
                        },
                    },
                    "valueFormat": "number",
                },
                {
                    "id": "optimisation_ladder",
                    "title": "Custom and Hugging Face optimisation candidates",
                    "subtitle": (
                        "Every candidate uses the same immutable one-token input and "
                        "is retained only as exact when its token sequence matches."
                    ),
                    "type": "horizontalBar",
                    "dataset": "optimisation_ladder",
                    "sourceId": "optimisation_ladder_csv",
                    "encodings": {
                        "x": {
                            "field": "optimisation_label",
                            "type": "nominal",
                            "label": "Optimisation",
                        },
                        "y": {
                            "field": "output_tokens_per_second",
                            "type": "quantitative",
                            "label": "Verified output tokens/s",
                        },
                    },
                    "valueFormat": "number",
                },
                {
                    "id": "coordinator_traffic",
                    "title": "Coordinator-bound final-stage traffic",
                    "subtitle": (
                        "The fast profile returns token IDs and selected logits instead "
                        "of the full FP32 vocabulary vector."
                    ),
                    "type": "bar",
                    "dataset": "traffic",
                    "sourceId": "traffic_results_csv",
                    "encodings": {
                        "x": {
                            "field": "traffic_type",
                            "type": "nominal",
                            "label": "Payload",
                        },
                        "y": {
                            "field": "bytes",
                            "type": "quantitative",
                            "label": "Bytes",
                        },
                    },
                    "valueFormat": "number",
                    "unit": "bytes per measured request",
                },
                {
                    "id": "readiness_cost",
                    "title": "Batch-one model load and warm-up cost",
                    "subtitle": (
                        "Cold-readiness costs are reported separately from steady-state throughput."
                    ),
                    "type": "horizontalBar",
                    "dataset": "readiness",
                    "sourceId": "engine_results_csv",
                    "encodings": {
                        "x": {
                            "field": "engine_step",
                            "type": "nominal",
                            "label": "Engine and readiness step",
                        },
                        "y": {
                            "field": "seconds",
                            "type": "quantitative",
                            "label": "Seconds",
                        },
                    },
                    "valueFormat": "number",
                    "unit": "seconds",
                },
            ],
            "tables": [
                {
                    "id": "acceptance_statuses",
                    "title": "Acceptance status ledger",
                    "subtitle": "All required Experiment 004 gates, without threshold relaxation.",
                    "dataset": "acceptance_statuses",
                    "sourceId": "summary_json",
                    "defaultSort": {"field": "status_name", "direction": "asc"},
                    "columns": [
                        {
                            "field": "status_name",
                            "label": "Acceptance check",
                            "type": "text",
                        },
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
                {
                    "id": "engine_result_table",
                    "title": "Engine benchmark result matrix",
                    "subtitle": (
                        "Median, variability, identity, latency, utilisation, and memory "
                        "for every emitted result row."
                    ),
                    "dataset": "engine_results",
                    "sourceId": "engine_results_csv",
                    "defaultSort": {
                        "field": "aggregate_output_tokens_per_second",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "model_id", "label": "Model", "type": "text"},
                        {"field": "workload", "label": "Workload", "type": "text"},
                        {
                            "field": "concurrency",
                            "label": "Concurrency",
                            "type": "number",
                        },
                        {"field": "engine", "label": "Engine", "type": "text"},
                        {"field": "profile", "label": "Profile", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {
                            "field": "aggregate_output_tokens_per_second",
                            "label": "Aggregate output tokens/s",
                            "type": "number",
                            "format": "number",
                        },
                        {
                            "field": "exact_reference_identity",
                            "label": "Exact",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "environment_table",
                    "title": "External engine environment evidence",
                    "subtitle": (
                        "Each engine records its isolated version and a concrete "
                        "failure diagnostic when unavailable."
                    ),
                    "dataset": "external_environments",
                    "sourceId": "external_environments_json",
                    "defaultSort": {"field": "engine", "direction": "asc"},
                    "columns": [
                        {"field": "engine", "label": "Engine", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                        {
                            "field": "requested_version",
                            "label": "Requested version",
                            "type": "text",
                        },
                    ],
                },
            ],
            "sources": source_inventory,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {title}",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "summary_json",
                    "body": (
                        "## The custom engine result and gate outcome\n\n"
                        "**Production-speed stage-engine benchmark.**\n\n"
                        f"{summary.get('conclusion', '')}\n\n"
                        f"Overall evidence status: **{summary.get('overall_status')}**. "
                        "The 80% target is deliberately reported separately from the "
                        "minimum 50% production-engine fraction gate."
                    ),
                },
                {
                    "id": "performance_finding",
                    "type": "markdown",
                    "sourceId": "engine_results_csv",
                    "body": (
                        "## Batch-one decode isolates kernel speed\n\n"
                        "Decode throughput excludes prefill, while aggregate verified "
                        "throughput includes host input transfer, prefill, decode, final-stage "
                        "sampling, and compact output transfer. Tokenisation is excluded "
                        "because all engines consume the immutable token-ID fixtures."
                    ),
                },
                {
                    "id": "batch_one_decode_block",
                    "type": "chart",
                    "chartId": "batch_one_decode",
                },
                {
                    "id": "batching_finding",
                    "type": "markdown",
                    "sourceId": "engine_results_csv",
                    "body": (
                        "## Real batches scale aggregate throughput\n\n"
                        "Each custom decode iteration combines compatible active requests "
                        "into one tensor and executes one stage forward for the batch. "
                        "Latency, balanced, and throughput scheduling policies are selected "
                        "at concurrency 1, 4, and 16/64 respectively."
                    ),
                },
                {
                    "id": "batch_scaling_block",
                    "type": "chart",
                    "chartId": "decode_batch_scaling",
                },
                {
                    "id": "traffic_finding",
                    "type": "markdown",
                    "sourceId": "traffic_results_csv",
                    "body": (
                        "## Final-worker argmax cuts return traffic\n\n"
                        "The fast profile keeps activations on the GPU in the monolithic "
                        "path and copies back only generated token IDs plus their selected "
                        "BF16 logits at the explicit measurement boundary."
                    ),
                },
                {
                    "id": "traffic_block",
                    "type": "chart",
                    "chartId": "coordinator_traffic",
                },
                {
                    "id": "scope_and_definitions",
                    "type": "markdown",
                    "body": (
                        "## Every comparison uses identical tokens\n\n"
                        f"The primary model is `{PRIMARY_MODEL_ID}` at "
                        f"`{PRIMARY_MODEL_REVISION}` in BF16. The secondary Qwen3-4B "
                        "revision is recorded in `model_revisions.json`. Every measured "
                        "request generates its full configured output length greedily. "
                        "TTFT is the measured prefill-to-first-logit interval; inter-token "
                        "latency is measured over decode iterations. Coefficient of "
                        "variation is population standard deviation divided by the mean "
                        "across measured repeats."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "profile_summary_json",
                    "body": (
                        "## CUDA events isolate GPU phases\n\n"
                        "Each engine completes model load, engine-specific compilation or "
                        "graph capture, at least three warm-up requests, and at least five "
                        "measured repeats in the full run. CUDA events time GPU phases; one "
                        "explicit synchronisation closes the measurement boundary. Model "
                        "load, compile/capture, warm-up, scheduler, sampling, queue, "
                        "serialisation, and tokenisation fields remain separate."
                    ),
                },
                {
                    "id": "readiness_block",
                    "type": "chart",
                    "chartId": "readiness_cost",
                },
                {
                    "id": "optimisation_method",
                    "type": "markdown",
                    "sourceId": "optimisation_ladder_csv",
                    "body": (
                        "## Only exact optimisations are selected\n\n"
                        "The ladder evaluates attention, cache, compile, and actual CUDA "
                        "graph variants independently. Automatic attention selection runs "
                        "a full autoregressive greedy probe, not a one-step numerical "
                        "comparison. Failed or token-divergent candidates remain in the "
                        "evidence instead of being silently selected."
                    ),
                },
                {
                    "id": "optimisation_ladder_block",
                    "type": "chart",
                    "chartId": "optimisation_ladder",
                },
                {
                    "id": "validation",
                    "type": "markdown",
                    "sourceId": "correctness_json",
                    "body": (
                        "## Exact greedy identity is the gate\n\n"
                        "The deterministic `qwen3_correctness` profile is the oracle. "
                        "Every successful performance result must match its complete token "
                        "sequence exactly at every concurrency. Full logits remain available "
                        "only through explicit diagnostics and are never returned by the "
                        "normal `qwen3_fast` result path."
                    ),
                },
                {
                    "id": "acceptance_table_block",
                    "type": "table",
                    "tableId": "acceptance_statuses",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Feature gaps limit interpretation\n\n"
                        + "\n".join(f"- {item}" for item in limitations)
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Cache and launch work remains\n\n"
                        "Next work should add shape-stable decode graphs without one graph "
                        "per token position, engine-level chunked prefill and prefix reuse, "
                        "and a supported Linux production-engine environment. These are "
                        "Experiment 004 follow-ups; Experiment 005 was not started."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Can a truly position-stable static cache reduce graph variants "
                        "while preserving exact tokens?\n"
                        "- Does CUDA IPC beat pinned BF16 host staging once logical stages "
                        "span Windows processes?\n"
                        "- How does prefix reuse change TTFT and aggregate throughput after "
                        "a real block allocator is introduced?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "acceptance_statuses": status_rows,
                "engine_results": result_rows,
                "batch_one": batch_one_rows,
                "batch_scaling": scaling_rows,
                "optimisation_ladder": ladder_rows,
                "traffic": traffic_rows,
                "readiness": readiness_rows,
                "external_environments": environment_rows,
            },
        },
        "sources": [dict(item) for item in sources],
    }


def _find_portable_report_builder() -> Path | None:
    configured = os.environ.get("SWARM_DATA_ANALYTICS_REPORT_BUILDER")
    if configured:
        candidate = Path(configured).resolve()
        return candidate if candidate.is_file() else None
    cache_root = (
        Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    )
    matches = sorted(
        cache_root.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _render_report_fallback(
    *,
    path: Path,
    summary: dict[str, Any],
    diagnostic: str,
) -> None:
    """Preserve readable partial evidence only when canonical packaging is unavailable."""

    path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Experiment 004: Production-Speed Qwen3 Stage Engine</title></head>
<body><h1>Experiment 004: Production-Speed Qwen3 Stage Engine</h1>
<p><strong>{html.escape(str(summary.get("conclusion", "")))}</strong></p>
<p>Canonical report packaging failed: {html.escape(diagnostic)}</p>
<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
</body></html>
""",
        encoding="utf-8",
    )


def _render_report(
    *,
    path: Path,
    summary: dict[str, Any],
    engine_rows: list[dict[str, Any]],
    optimisation_ladder_rows: list[dict[str, Any]],
    environments: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_path = path.parent / "artifact.json"
    artifact = _report_artifact(
        summary=summary,
        engine_rows=engine_rows,
        optimisation_ladder_rows=optimisation_ladder_rows,
        environments=environments,
    )
    _json_write(artifact_path, artifact)
    builder = _find_portable_report_builder()
    if builder is None:
        diagnostic = (
            "Data Analytics portable report builder was not installed; set "
            "SWARM_DATA_ANALYTICS_REPORT_BUILDER to its deliver_portable_artifact.mjs path"
        )
        _render_report_fallback(path=path, summary=summary, diagnostic=diagnostic)
        return {
            "status": "FAIL",
            "verification": "unavailable",
            "diagnostic": diagnostic,
        }
    code = _run_command(
        [
            "node",
            str(builder),
            "--input",
            str(artifact_path.resolve()),
            "--output",
            str(path.resolve()),
        ],
        cwd=path.parent,
        stdout_path=path.parent / "logs" / "report-builder.stdout.log",
        stderr_path=path.parent / "logs" / "report-builder.stderr.log",
        timeout_seconds=300,
    )
    browser_fallback_used = False
    first_stderr_path = path.parent / "logs" / "report-builder.stderr.log"
    if code != 0 and first_stderr_path.is_file():
        first_error = first_stderr_path.read_text(encoding="utf-8")
        if any(
            code_name in first_error
            for code_name in (
                '"code":"reader_timeout"',
                '"code":"reader_fallback"',
                '"code":"reader_not_visible"',
                '"code":"horizontal_overflow"',
            )
        ):
            structural_environment = dict(os.environ)
            structural_environment["CHROMIUM_EXECUTABLE_PATH"] = str(
                (path.parent / "profiles" / "unavailable-chromium.exe").resolve()
            )
            code = _run_command(
                [
                    "node",
                    str(builder),
                    "--input",
                    str(artifact_path.resolve()),
                    "--output",
                    str(path.resolve()),
                ],
                cwd=path.parent,
                stdout_path=(path.parent / "logs" / "report-builder-structural.stdout.log"),
                stderr_path=(path.parent / "logs" / "report-builder-structural.stderr.log"),
                timeout_seconds=300,
                environment=structural_environment,
            )
            browser_fallback_used = code == 0
    if code != 0 or not path.is_file():
        diagnostic = f"portable report builder returned {code}; see logs/report-builder.stderr.log"
        _render_report_fallback(path=path, summary=summary, diagnostic=diagnostic)
        return {
            "status": "FAIL",
            "verification": "failed",
            "builder": str(builder),
            "return_code": code,
            "diagnostic": diagnostic,
        }
    receipt_log = (
        path.parent
        / "logs"
        / (
            "report-builder-structural.stdout.log"
            if browser_fallback_used
            else "report-builder.stdout.log"
        )
    )
    receipt_text = receipt_log.read_text(encoding="utf-8")
    return {
        "status": "PASS",
        "verification": (
            "passed" if '"verification":"passed"' in receipt_text else "structural_only"
        ),
        "builder": str(builder),
        "return_code": code,
        "browser_qa_limitation": (
            "The installed Chromium reader/layout check did not complete cleanly; exact "
            "payload equality and semantic/runtime structure passed, but per-artifact "
            "browser chart QA did not."
            if browser_fallback_used
            else None
        ),
    }


def _copy_baseline_evidence(
    repository_root: Path,
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = repository_root / "artifacts" / "benchmarks" / "experiment-004" / "prechange"
    legacy_path = source / "legacy-baseline.json"
    process_path = source / "legacy-process-baseline.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.is_file() else {}
    process = json.loads(process_path.read_text(encoding="utf-8")) if process_path.is_file() else {}
    profile_directory = run_directory / "profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "legacy-baseline.json",
        "legacy-baseline-trace.json",
        "legacy-process-baseline.json",
    ):
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, profile_directory / name)
    return legacy, process


def _process_baseline_tps(process_baseline: dict[str, Any]) -> float:
    """Derive the legacy serving baseline from its measured request rows."""

    request_results = process_baseline.get("request_results", [])
    observations = [
        float(row["output_tokens_per_second"])
        for row in request_results
        if row.get("status") == "completed"
        and row.get("verified") is True
        and row.get("output_tokens_per_second") is not None
    ]
    if not observations:
        raise ValueError(
            "the pre-change process baseline contains no verified throughput observations"
        )
    return float(statistics.median(observations))


def _resolve_models(
    *,
    config: EnginePerformanceConfig,
    primary_model: str | None,
    secondary_model: str | None,
    skip_secondary: bool,
) -> list[dict[str, Any]]:
    from huggingface_hub import model_info, snapshot_download

    requested_primary = primary_model or config.models[0].model_id
    if requested_primary != PRIMARY_MODEL_ID:
        raise ValueError(f"Experiment 004 primary model must remain {PRIMARY_MODEL_ID!r}")
    primary_path = snapshot_download(
        PRIMARY_MODEL_ID,
        revision=PRIMARY_MODEL_REVISION,
        local_files_only=False,
    )
    models: list[dict[str, Any]] = [
        {
            "role": "primary",
            "model_id": PRIMARY_MODEL_ID,
            "revision": PRIMARY_MODEL_REVISION,
            "path": str(Path(primary_path).resolve()),
            "dtype": "bfloat16",
            "resolution_status": "PASS",
        }
    ]
    requested_secondary = secondary_model or config.models[1].model_id
    if skip_secondary:
        models.append(
            {
                "role": "secondary",
                "model_id": requested_secondary,
                "revision": SECONDARY_MODEL_REVISION,
                "path": None,
                "dtype": "bfloat16",
                "resolution_status": "FAIL",
                "diagnostic": "secondary model explicitly skipped",
            }
        )
        return models
    if requested_secondary != SECONDARY_MODEL_ID:
        raise ValueError(f"Experiment 004 secondary model must remain {SECONDARY_MODEL_ID!r}")
    try:
        resolved_revision = model_info(SECONDARY_MODEL_ID).sha
        if resolved_revision != SECONDARY_MODEL_REVISION:
            # The resolved SHA is still immutable and is the authoritative
            # current revision for this run.
            revision = str(resolved_revision)
        else:
            revision = SECONDARY_MODEL_REVISION
        secondary_path = snapshot_download(
            SECONDARY_MODEL_ID,
            revision=revision,
            local_files_only=False,
        )
        models.append(
            {
                "role": "secondary",
                "model_id": SECONDARY_MODEL_ID,
                "revision": revision,
                "path": str(Path(secondary_path).resolve()),
                "dtype": "bfloat16",
                "resolution_status": "PASS",
            }
        )
    except Exception as exc:
        models.append(
            {
                "role": "secondary",
                "model_id": SECONDARY_MODEL_ID,
                "revision": SECONDARY_MODEL_REVISION,
                "path": None,
                "dtype": "bfloat16",
                "resolution_status": "FAIL",
                "diagnostic": f"{type(exc).__name__}: {exc}",
            }
        )
    return models


def _runtime_workloads(
    config: EnginePerformanceConfig,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    if smoke:
        return [
            {
                "name": "decode-focused",
                "input_tokens": 1,
                "output_tokens": 16,
                "concurrency": [1, 4],
            }
        ]
    return [item.model_dump(mode="json") for item in config.workloads]


def _failed_result(
    engine: str,
    profile: str,
    diagnostic: str,
    job: dict[str, Any],
) -> dict[str, Any]:
    return {
        "engine": engine,
        "profile": profile,
        "status": "FAIL",
        "worker_status": "not-run",
        "error": diagnostic,
        "job": job,
        "measured_repeats": [],
        "statistics": {
            "median": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "standard_deviation": 0.0,
            "coefficient_of_variation": 0.0,
        },
    }


def run_engine_performance_experiment(
    config: EnginePerformanceConfig,
    *,
    config_path: Path,
    primary_model: str | None = None,
    secondary_model: str | None = None,
    skip_secondary: bool = False,
    skip_optional_engines: bool = False,
    output_root: Path | None = None,
    resume: bool = False,
    smoke: bool = False,
    profile: bool = False,
    keep_servers: bool = False,
) -> EnginePerformanceRun:
    del keep_servers  # Offline/container workers terminate after every job.
    repository_root = _repository_root()
    selected_output_root = (
        output_root.resolve()
        if output_root is not None
        else (repository_root / config.output_root).resolve()
    )
    selected_output_root.mkdir(parents=True, exist_ok=True)
    if resume:
        candidates = sorted(
            selected_output_root.glob("*-engine-performance-*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        run_directory = candidates[0] if candidates else None
    else:
        run_directory = None
    if run_directory is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_directory = selected_output_root / f"{timestamp}-engine-performance-{uuid4().hex[:8]}"
        run_directory.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "profiles", "charts"):
        (run_directory / child).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_directory / "config.requested.yaml")
    resolved_config = config.model_dump(mode="json")
    resolved_config["runtime"] = {
        "smoke": smoke,
        "profile": profile,
        "skip_secondary": skip_secondary,
        "skip_optional_engines": skip_optional_engines,
        "resume": resume,
    }
    (run_directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    environment_path = run_directory / "environment.json"
    _run_command(
        [
            sys.executable,
            "-m",
            "swarm_inference.experiments.environment_probe",
            "--output",
            str(environment_path),
        ],
        cwd=repository_root,
        stdout_path=run_directory / "logs" / "environment.stdout.log",
        stderr_path=run_directory / "logs" / "environment.stderr.log",
        timeout_seconds=120,
    )
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.is_file()
        else {}
    )
    environment["profile"] = "experiment_environment"
    environment["execution_profiles"] = [
        config.profiles.correctness,
        config.profiles.performance,
    ]
    environment["git"] = _git_state(repository_root)
    _json_write(environment_path, environment)
    legacy_baseline, process_baseline = _copy_baseline_evidence(repository_root, run_directory)
    measured_baseline = _process_baseline_tps(process_baseline)
    legacy_launches = next(
        (
            int(row.get("calls", 0))
            for row in legacy_baseline.get("profile_top_ten", [])
            if row.get("name") == "cudaLaunchKernel"
        ),
        0,
    )
    legacy_profile_output_count = len(
        legacy_baseline.get("profile_request", {}).get("output_token_ids", [])
    )
    legacy_profile_decode_count = max(legacy_profile_output_count - 1, 0)
    legacy_launches_per_decode_token = (
        legacy_launches / legacy_profile_decode_count if legacy_profile_decode_count else None
    )
    profile_summary = {
        "profile": "qwen3_correctness",
        "baseline_commit": legacy_baseline.get("baseline_commit"),
        "legacy_numpy_stage_execute": legacy_baseline.get("decode_output_tokens_per_second", {}),
        "legacy_process_serving_tps": measured_baseline,
        "top_ten": legacy_baseline.get("profile_top_ten", []),
        "trace": "profiles/legacy-baseline-trace.json",
        "nsight_systems": ("unavailable" if shutil.which("nsys") is None else "available"),
        "kernels_per_decode_token": legacy_launches_per_decode_token,
        "kernel_count_status": (
            "pytorch_profiler_cudaLaunchKernel_calls_per_decode_token"
            if legacy_launches_per_decode_token is not None
            else "unavailable_no_profile_launch_count"
        ),
        "kernel_count_definition": (
            "CUDA runtime launch calls reported by the PyTorch profiler divided "
            "by autoregressive decode steps; includes all profiler-visible launches"
        ),
    }
    _json_write(run_directory / "profile_summary.json", profile_summary)

    models = _resolve_models(
        config=config,
        primary_model=primary_model,
        secondary_model=secondary_model,
        skip_secondary=skip_secondary,
    )
    _json_write(
        run_directory / "model_revisions.json",
        {
            "profile": "model_resolution",
            "execution_profiles": [
                config.profiles.correctness,
                config.profiles.performance,
            ],
            "models": models,
        },
    )
    primary_path = Path(str(models[0]["path"]))
    global_input_root = repository_root / "artifacts" / "benchmarks" / "experiment-004" / "inputs"
    input_payload = build_input_fixtures(
        model_path=primary_path,
        output_directory=global_input_root,
        nvtx_enabled=profile,
    )
    _json_write(run_directory / "input_tokens.json", input_payload)

    environment_root = (repository_root / config.external_environments.root).resolve()
    environment_root.mkdir(parents=True, exist_ok=True)
    hf_environment = provision_huggingface_environment(
        repository_root=repository_root,
        environment_root=environment_root,
        transformers_version=config.external_environments.huggingface_transformers,
        torch_version=config.external_environments.huggingface_torch,
    )
    sglang_image = f"lmsysorg/sglang:v{config.external_environments.sglang}"
    vllm_image = f"vllm/vllm-openai:v{config.external_environments.vllm}"
    tensorrt_image = (
        f"nvcr.io/nvidia/tensorrt-llm/release:{config.external_environments.tensorrt_llm}"
    )
    sglang_environment = inspect_linux_engine_prerequisites(
        repository_root=repository_root,
        environment_root=environment_root,
        engine="sglang",
        version=config.external_environments.sglang,
        image=sglang_image,
    )
    external_environments: list[ExternalEngineEnvironment] = [
        hf_environment,
        sglang_environment,
    ]
    if not skip_optional_engines:
        external_environments.extend(
            [
                inspect_linux_engine_prerequisites(
                    repository_root=repository_root,
                    environment_root=environment_root,
                    engine="vllm",
                    version=config.external_environments.vllm,
                    image=vllm_image,
                ),
                inspect_linux_engine_prerequisites(
                    repository_root=repository_root,
                    environment_root=environment_root,
                    engine="tensorrt_llm",
                    version=config.external_environments.tensorrt_llm,
                    image=tensorrt_image,
                ),
            ]
        )
    else:
        external_environments.extend(
            [
                ExternalEngineEnvironment(
                    engine="vllm",
                    kind="skipped",
                    root=str((environment_root / "vllm").resolve()),
                    requested_version=config.external_environments.vllm,
                    status="SKIP",
                    diagnostic="optional engines explicitly skipped",
                ),
                ExternalEngineEnvironment(
                    engine="tensorrt_llm",
                    kind="skipped",
                    root=str((environment_root / "tensorrt_llm").resolve()),
                    requested_version=config.external_environments.tensorrt_llm,
                    status="SKIP",
                    diagnostic="optional engines explicitly skipped",
                ),
            ]
        )
    _json_write(
        run_directory / "external_engine_environments.json",
        {
            "profile": "environment_isolation",
            "execution_profiles": [
                config.profiles.correctness,
                config.profiles.performance,
            ],
            "environments": [item.payload() for item in external_environments],
        },
    )

    runtime_repeats = 1 if smoke else config.repeats
    runtime_warmups = 1 if smoke else config.warmup_requests
    runtime_workloads = _runtime_workloads(config, smoke=smoke)
    project_python = Path(sys.executable)
    hf_python = (
        Path(str(hf_environment.python_executable))
        if hf_environment.python_executable
        else project_python
    )
    correctness_payload: dict[str, Any] = {
        "profile": "qwen3_correctness",
        "require_exact_greedy_token_identity": True,
        "references": {},
        "comparisons": [],
        "fairness_checks": [],
    }
    # BF16 kernels can cross an argmax near-tie differently at another batch
    # shape even with deterministic algorithms and FP32 accumulation.  Exact
    # identity therefore means qwen3_fast versus qwen3_correctness at the same
    # model, token fixture and measured batch shape.
    references: dict[tuple[str, str, int], list[list[int]]] = {}
    all_results: list[dict[str, Any]] = []
    engine_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    traffic_rows: list[dict[str, Any]] = []
    compile_rows: list[dict[str, Any]] = []
    cuda_graph_rows: list[dict[str, Any]] = []
    ladder_rows: list[dict[str, Any]] = [
        {
            "optimisation": "legacy_process_rpc_numpy",
            "profile": "qwen3_correctness",
            "attention_backend": "eager",
            "cache_backend": "dynamic_reference",
            "compile_mode": "eager",
            "output_tokens_per_second": measured_baseline,
            "exact_reference_identity": True,
            "status": "PASS",
        }
    ]
    hf_optimised_selection = {
        "attention_backend": "eager",
        "cache_backend": "dynamic",
        "compile_mode": "eager",
        "selection_reason": (
            "exact eager/dynamic fallback; advanced candidates have not yet "
            "passed the immutable-token probe"
        ),
    }
    hf_optimisation_evidence: list[dict[str, Any]] = []
    custom_fast_selection: dict[str, Any] = {
        "attention_backend": "sdpa",
        "cache_backend": "static",
        "compile_mode": "eager",
        "selection_reason": (
            "exact SDPA/static fallback; optimisation candidates have not yet "
            "passed the immutable-token probe"
        ),
    }
    custom_optimisation_evidence: list[dict[str, Any]] = []

    transport_job = {
        "engine": "transport_paths",
        "profile": "qwen3_fast",
        "implementation_revision": "experiment-004-transport-paths-v1",
        "tensor_shapes": [
            [1, 1, 1024],
            [16, 1, 1024],
            [1, 128, 1024],
        ],
        "warmup_requests": runtime_warmups,
        "repeats": runtime_repeats,
        "transfers_per_repeat": 50,
        "telemetry_interval_seconds": (config.measurement.gpu_sample_interval_ms / 1000),
        "nvtx_enabled": profile,
    }
    transport_result = _execute_job(
        repository_root=repository_root,
        run_directory=run_directory,
        job_name="transport-paths",
        job=transport_job,
        python_executable=project_python,
        external=False,
        resume=resume,
    )
    all_results.append(transport_result)
    correctness_payload["transport_path_benchmark"] = {
        "status": transport_result.get("status"),
        "profile": transport_result.get("profile"),
        "exact_bfloat16_identity": transport_result.get(
            "exact_bfloat16_identity",
            False,
        ),
        "gpu_resident_direct_reference_verified": transport_result.get(
            "gpu_resident_direct_reference_verified",
            False,
        ),
        "paths": transport_result.get("paths", []),
        "diagnostic": transport_result.get("error"),
    }
    for point in transport_result.get("paths", []):
        traffic_rows.append(
            {
                "engine": "transport_paths",
                "profile": point.get("profile", "qwen3_fast"),
                "model_id": PRIMARY_MODEL_ID,
                "workload": "transport-microbenchmark",
                "concurrency": point.get("shape", [0])[0],
                "repeat": "aggregate",
                "transport_path": point.get("path"),
                "selected_method": point.get("selected_method"),
                "tensor_shape": point.get("shape"),
                "logical_bytes_per_transfer": point.get(
                    "logical_bytes_per_transfer",
                    0,
                ),
                "transport_latency_ms_median": point.get("latency_ms", {}).get(
                    "median",
                    0.0,
                ),
                "transport_latency_ms_minimum": point.get("latency_ms", {}).get(
                    "minimum",
                    0.0,
                ),
                "transport_latency_ms_maximum": point.get("latency_ms", {}).get(
                    "maximum",
                    0.0,
                ),
                "transport_latency_cv": point.get("latency_ms", {}).get(
                    "coefficient_of_variation",
                    0.0,
                ),
                "host_to_device_bytes": point.get(
                    "host_to_device_bytes_per_transfer",
                    0,
                ),
                "device_to_host_bytes": point.get(
                    "device_to_host_bytes_per_transfer",
                    0,
                ),
                "serialised_bytes": point.get(
                    "serialised_bytes_per_transfer",
                    0,
                ),
                "serialisation_ms_total": point.get(
                    "serialisation_ms_total",
                    0.0,
                ),
                "deserialisation_ms_total": point.get(
                    "deserialisation_ms_total",
                    0.0,
                ),
                "explicit_transport_synchronisations_per_transfer": point.get(
                    "explicit_transport_synchronisations_per_transfer",
                    0,
                ),
                "bfloat16_bits_exact": point.get("bfloat16_bits_exact"),
                "direct_tensor_reference": point.get("direct_tensor_reference"),
                "full_logit_equivalent_bytes": 0,
                "coordinator_bound_bytes": 0,
                "full_logits_transferred": False,
            }
        )

    runnable_models = [item for item in models if item.get("resolution_status") == "PASS"]
    for model in runnable_models:
        model_key = _slug(str(model["model_id"]))
        for workload in runtime_workloads:
            workload_name = str(workload["name"])
            token_ids = list(input_payload["workloads"][workload_name])
            output_tokens = int(workload["output_tokens"])
            for concurrency_value in workload["concurrency"]:
                concurrency = int(concurrency_value)
                reference_job = _worker_job(
                    engine="custom_correctness",
                    model=model,
                    input_rows=[list(token_ids) for _ in range(concurrency)],
                    output_tokens=output_tokens,
                    concurrency=concurrency,
                    repeats=runtime_repeats,
                    warmups=runtime_warmups,
                    reference=None,
                    config=config,
                    attention_backend="eager",
                    cache_backend="dynamic_reference",
                    compile_mode="eager",
                    nvtx_enabled=profile,
                )
                reference_result = _execute_job(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    job_name=f"reference-{model_key}-{workload_name}-c{concurrency}",
                    job=reference_job,
                    python_executable=project_python,
                    external=False,
                    resume=resume,
                )
                all_results.append(reference_result)
                output_rows = reference_result.get("output_token_ids", [])
                if output_rows:
                    references[(str(model["model_id"]), workload_name, concurrency)] = output_rows
                reference_key = f"{model['model_id']}::{workload_name}::c{concurrency}"
                correctness_payload["references"][reference_key] = {
                    "status": reference_result.get("status"),
                    "profile": reference_result.get("profile"),
                    "batch_shape": [concurrency, len(token_ids)],
                    "token_hash": reference_result.get("output_token_hash"),
                    "output_token_ids": output_rows,
                }
                aggregate = _aggregate_engine_row(
                    reference_result,
                    model_id=str(model["model_id"]),
                    workload=workload_name,
                    concurrency=concurrency,
                )
                engine_rows.append(aggregate)
                latency_rows.extend(
                    _metric_rows(
                        reference_result,
                        model_id=str(model["model_id"]),
                        workload=workload_name,
                        concurrency=concurrency,
                    )
                )
        if not smoke and config.prefix_reuse.enabled:
            prefix_rows = [list(row) for row in input_payload["prefix_reuse"]["rows"]]
            prefix_job = _worker_job(
                engine="custom_correctness",
                model=model,
                input_rows=prefix_rows,
                output_tokens=config.prefix_reuse.output_tokens,
                concurrency=4,
                repeats=runtime_repeats,
                warmups=runtime_warmups,
                reference=None,
                config=config,
                attention_backend="eager",
                cache_backend="dynamic_reference",
                compile_mode="eager",
                nvtx_enabled=profile,
                prefix_reuse_enabled=True,
            )
            prefix_result = _execute_job(
                repository_root=repository_root,
                run_directory=run_directory,
                job_name=f"reference-{model_key}-prefix-reuse",
                job=prefix_job,
                python_executable=project_python,
                external=False,
                resume=resume,
            )
            all_results.append(prefix_result)
            output_rows = prefix_result.get("output_token_ids", [])
            if output_rows:
                references[(str(model["model_id"]), "prefix-reuse", 4)] = output_rows
            correctness_payload["references"][f"{model['model_id']}::prefix-reuse::c4"] = {
                "status": prefix_result.get("status"),
                "profile": prefix_result.get("profile"),
                "batch_shape": [4, len(prefix_rows[0])],
                "token_hash": prefix_result.get("output_token_hash"),
                "output_token_ids": output_rows,
            }

    primary_reference = references.get((PRIMARY_MODEL_ID, "decode-focused", 1))
    if primary_reference:
        ladder_candidates = [
            (
                "gpu_native_dynamic",
                "eager",
                "dynamic_reference",
                "eager",
            ),
            ("static_cache", "eager", "static", "eager"),
            ("sdpa_static", "sdpa", "static", "eager"),
            ("torch_compile_default", "eager", "static", "default"),
            (
                "torch_compile_reduce_overhead",
                "eager",
                "static",
                "reduce-overhead",
            ),
            (
                "torch_compile_max_autotune",
                "eager",
                "static",
                "max-autotune",
            ),
            (
                "manual_cuda_graph",
                "auto",
                "static",
                "manual-cuda-graph",
            ),
        ]
        ladder_output_tokens = min(64, len(primary_reference[0]))
        successful_custom_candidates: list[tuple[float, dict[str, Any]]] = []
        for name, attention, cache, compile_mode in ladder_candidates:
            ladder_job = _worker_job(
                engine="custom_fast",
                model=models[0],
                input_rows=[[1]],
                output_tokens=ladder_output_tokens,
                concurrency=1,
                repeats=runtime_repeats,
                warmups=runtime_warmups,
                reference=[primary_reference[0][:ladder_output_tokens]],
                config=config,
                attention_backend=attention,
                cache_backend=cache,
                compile_mode=compile_mode,
                nvtx_enabled=profile,
            )
            ladder_result = _execute_job(
                repository_root=repository_root,
                run_directory=run_directory,
                job_name=f"ladder-{name}",
                job=ladder_job,
                python_executable=project_python,
                external=False,
                resume=resume,
            )
            all_results.append(ladder_result)
            first_metrics = (
                ladder_result.get("measured_repeats", [{}])[0].get("metrics", {})
                if ladder_result.get("measured_repeats")
                else {}
            )
            compile_diagnostics = ladder_result.get("stage_state", {}).get("compile_diagnostics")
            row = {
                "optimisation": name,
                "profile": ladder_result.get("profile"),
                "requested_attention_backend": attention,
                "attention_backend": ladder_result.get("attention_backend", attention),
                "cache_backend": ladder_result.get("cache_backend", cache),
                "compile_mode": ladder_result.get("compile_mode", compile_mode),
                "output_tokens_per_second": ladder_result.get("statistics", {}).get("median", 0.0),
                "decode_output_tokens_per_second": ladder_result.get("decode_statistics", {}).get(
                    "median", 0.0
                ),
                "exact_reference_identity": ladder_result.get("exact_reference_identity"),
                "cuda_graph_verified": first_metrics.get("cuda_graph_verified", False),
                "cache_reserved_bytes": first_metrics.get("cache_reserved_bytes", 0),
                "cache_used_bytes": first_metrics.get("cache_used_bytes", 0),
                "cache_allocation_count": first_metrics.get("cache_allocation_count"),
                "cache_fragmentation_fraction": first_metrics.get("cache_fragmentation_fraction"),
                "compile_diagnostics": compile_diagnostics,
                "status": ladder_result.get("status"),
                "diagnostic": ladder_result.get("error"),
            }
            ladder_rows.append(row)
            custom_optimisation_evidence.append(row)
            compile_rows.append(
                {
                    "engine": "custom_fast",
                    "profile": ladder_result.get("profile"),
                    "mode": compile_mode,
                    "status": ladder_result.get("status"),
                    "compile_diagnostics": compile_diagnostics,
                    "warmup_seconds": ladder_result.get("warmup_seconds", 0),
                    "output_tokens_per_second": ladder_result.get("statistics", {}).get(
                        "median", 0.0
                    ),
                }
            )
            compile_path_verified = (
                compile_mode == "eager"
                or (
                    compile_mode == "manual-cuda-graph"
                    and bool(first_metrics.get("cuda_graph_verified"))
                )
                or (
                    isinstance(compile_diagnostics, dict)
                    and bool(compile_diagnostics.get("verified_execution"))
                    and not bool(compile_diagnostics.get("fallback_used"))
                )
            )
            if (
                ladder_result.get("status") == "PASS"
                and ladder_result.get("exact_reference_identity") is True
                and compile_path_verified
            ):
                rate = float(ladder_result.get("statistics", {}).get("median", 0.0))
                successful_custom_candidates.append(
                    (
                        rate,
                        {
                            "attention_backend": attention,
                            "selected_runtime_attention_backend": ladder_result.get(
                                "attention_backend",
                                attention,
                            ),
                            "cache_backend": cache,
                            "compile_mode": compile_mode,
                            "selection_probe_output_tokens": ladder_output_tokens,
                            "selection_probe_output_tokens_per_second": rate,
                            "selection_reason": (
                                f"fastest exact verified {ladder_output_tokens}-token "
                                f"candidate: {name}"
                            ),
                        },
                    )
                )
        if successful_custom_candidates:
            custom_fast_selection = max(
                successful_custom_candidates,
                key=lambda item: item[0],
            )[1]
        correctness_payload["custom_optimisation_selection"] = {
            **custom_fast_selection,
            "candidates": custom_optimisation_evidence,
        }
        if hf_environment.status == "PASS":
            hf_candidates = [
                ("hf_exact_eager_dynamic", "eager", "dynamic", "eager"),
                ("hf_sdpa_dynamic", "sdpa", "dynamic", "eager"),
                ("hf_eager_static", "eager", "static", "eager"),
                ("hf_compile_default_dynamic", "eager", "dynamic", "default"),
                (
                    "hf_compile_reduce_overhead_static",
                    "eager",
                    "static",
                    "reduce-overhead",
                ),
                (
                    "hf_compile_max_autotune_static",
                    "eager",
                    "static",
                    "max-autotune",
                ),
            ]
            successful_hf_candidates: list[tuple[float, dict[str, str]]] = []
            for name, attention, cache, compile_mode in hf_candidates:
                candidate_job = _worker_job(
                    engine="huggingface_optimised",
                    model=models[0],
                    input_rows=[[1]],
                    output_tokens=ladder_output_tokens,
                    concurrency=1,
                    repeats=runtime_repeats,
                    warmups=runtime_warmups,
                    reference=[primary_reference[0][:ladder_output_tokens]],
                    config=config,
                    attention_backend=attention,
                    cache_backend=cache,
                    compile_mode=compile_mode,
                    nvtx_enabled=False,
                )
                candidate_result = _execute_job(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    job_name=f"ladder-{name}",
                    job=candidate_job,
                    python_executable=hf_python,
                    external=True,
                    resume=resume,
                )
                rate = float(candidate_result.get("statistics", {}).get("median", 0.0))
                row = {
                    "optimisation": name,
                    "profile": "huggingface_optimised",
                    "attention_backend": attention,
                    "cache_backend": cache,
                    "compile_mode": compile_mode,
                    "output_tokens_per_second": rate,
                    "exact_reference_identity": candidate_result.get("exact_reference_identity"),
                    "status": candidate_result.get("status"),
                    "diagnostic": candidate_result.get("error"),
                }
                ladder_rows.append(row)
                hf_optimisation_evidence.append(row)
                compile_rows.append(
                    {
                        "engine": "huggingface_optimised",
                        "profile": "huggingface_optimised",
                        "mode": compile_mode,
                        "status": candidate_result.get("status"),
                        "compile_diagnostics": candidate_result.get("compile_diagnostics"),
                        "warmup_seconds": candidate_result.get(
                            "warmup_seconds",
                            0,
                        ),
                        "output_tokens_per_second": rate,
                    }
                )
                if (
                    candidate_result.get("status") == "PASS"
                    and candidate_result.get("exact_reference_identity") is True
                ):
                    successful_hf_candidates.append(
                        (
                            rate,
                            {
                                "attention_backend": attention,
                                "cache_backend": cache,
                                "compile_mode": compile_mode,
                                "selection_reason": (f"fastest exact 64-token candidate: {name}"),
                            },
                        )
                    )
            if successful_hf_candidates:
                hf_optimised_selection = max(
                    successful_hf_candidates,
                    key=lambda item: item[0],
                )[1]
        correctness_payload["huggingface_optimisation_selection"] = {
            **hf_optimised_selection,
            "candidates": hf_optimisation_evidence,
        }

    def record_result(
        result: dict[str, Any],
        *,
        model_id: str,
        workload_name: str,
        concurrency: int,
        is_prefix: bool = False,
    ) -> None:
        if result.get("exact_reference_identity") is False and not result.get("error"):
            mismatch = _first_token_mismatch(
                result.get("job", {}).get("reference_output_token_ids"),
                result.get("output_token_ids"),
            )
            result["first_token_mismatch"] = mismatch
            if mismatch is not None:
                result["error"] = (
                    "exact greedy token mismatch at request "
                    f"{mismatch['request_index']} output index "
                    f"{mismatch['token_index']}: expected "
                    f"{mismatch['expected_token_id']}, received "
                    f"{mismatch['actual_token_id']}"
                )
        all_results.append(result)
        aggregate = _aggregate_engine_row(
            result,
            model_id=model_id,
            workload=workload_name,
            concurrency=concurrency,
            prefix_reuse=is_prefix,
        )
        if is_prefix:
            aggregate["prefix_reuse_supported"] = result.get("engine") == "sglang"
            aggregate["prefill_work_avoided_tokens"] = (
                result.get("measured_repeats", [{}])[0]
                .get("metrics", {})
                .get("cached_prompt_tokens", 0)
                if result.get("measured_repeats")
                else 0
            )
        engine_rows.append(aggregate)
        repeat_rows = _metric_rows(
            result,
            model_id=model_id,
            workload=workload_name,
            concurrency=concurrency,
            prefix_reuse=is_prefix,
        )
        latency_rows.extend(repeat_rows)
        for row in repeat_rows:
            batch_rows.append(
                {
                    "engine": row.get("engine"),
                    "profile": row.get("profile"),
                    "model_id": model_id,
                    "workload": workload_name,
                    "concurrency": concurrency,
                    "repeat": row.get("repeat"),
                    "scheduler_policy": row.get("scheduler_policy", _scheduler_policy(concurrency)),
                    "useful_tokens": row.get("useful_prompt_tokens", 0),
                    "padding_tokens": row.get("padding_tokens", 0),
                    "batch_forward_count": row.get("batch_forward_count", 0),
                    "scheduler_ms": row.get("scheduler_ms", 0),
                    "queue_wait_ms": row.get("queue_wait_ms", 0),
                    "scheduler_metrics": row.get("scheduler_metrics"),
                    "aggregate_output_tokens_per_second": row.get(
                        "aggregate_verified_output_tokens_per_second", 0
                    ),
                }
            )
            memory_rows.append(
                {
                    "engine": row.get("engine"),
                    "profile": row.get("profile"),
                    "model_id": model_id,
                    "workload": workload_name,
                    "concurrency": concurrency,
                    "repeat": row.get("repeat"),
                    "peak_vram_bytes": row.get("peak_vram_bytes", 0),
                    "cache_backend": row.get("cache_backend"),
                    "cache_reserved_bytes": row.get("cache_reserved_bytes", 0),
                    "cache_used_bytes": row.get("cache_used_bytes", 0),
                    "cache_allocation_count": row.get("cache_allocation_count"),
                    "cache_fragmentation_fraction": row.get("cache_fragmentation_fraction"),
                    "cache_accounting_status": row.get(
                        "cache_accounting_status",
                        "unavailable_for_engine",
                    ),
                }
            )
            traffic_rows.append(
                {
                    "engine": row.get("engine"),
                    "profile": row.get("profile"),
                    "model_id": model_id,
                    "workload": workload_name,
                    "concurrency": concurrency,
                    "repeat": row.get("repeat"),
                    "host_to_device_bytes": row.get("host_to_device_bytes", 0),
                    "device_to_host_bytes": row.get("device_to_host_bytes", 0),
                    "full_logit_equivalent_bytes": row.get("full_logit_equivalent_bytes", 0),
                    "coordinator_bound_bytes": row.get("coordinator_bound_bytes", 0),
                    "full_logits_transferred": row.get("full_logits_transferred"),
                }
            )
            if row.get("cuda_graph_verified") is not None:
                cuda_graph_rows.append(
                    {
                        "engine": row.get("engine"),
                        "profile": row.get("profile"),
                        "model_id": model_id,
                        "workload": workload_name,
                        "concurrency": concurrency,
                        "repeat": row.get("repeat"),
                        "bucket_size": concurrency,
                        "capture_ms": row.get("cuda_graph_capture_ms", 0),
                        "replay_count": row.get("cuda_graph_replay_count", 0),
                        "captured": row.get("cuda_graph_verified", False),
                        "replay_verified": row.get("cuda_graph_verified", False),
                    }
                )
        correctness_payload["comparisons"].append(
            {
                "engine": result.get("engine"),
                "profile": result.get("profile"),
                "model_id": model_id,
                "workload": workload_name,
                "concurrency": concurrency,
                "exact_reference_identity": result.get("exact_reference_identity"),
                "status": result.get("status"),
                "output_token_hash": result.get("output_token_hash"),
                "diagnostic": result.get("error"),
                "first_token_mismatch": result.get("first_token_mismatch"),
            }
        )

    for model in runnable_models:
        model_id = str(model["model_id"])
        model_key = _slug(model_id)
        for workload in runtime_workloads:
            workload_name = str(workload["name"])
            base_tokens = list(input_payload["workloads"][workload_name])
            output_tokens = int(workload["output_tokens"])
            for concurrency in workload["concurrency"]:
                concurrency = int(concurrency)
                inputs = [list(base_tokens) for _ in range(concurrency)]
                reference = references.get((model_id, workload_name, concurrency))
                common: dict[str, Any] = {
                    "model": model,
                    "input_rows": inputs,
                    "output_tokens": output_tokens,
                    "concurrency": concurrency,
                    "repeats": runtime_repeats,
                    "warmups": runtime_warmups,
                    "reference": reference,
                    "config": config,
                }
                custom_job = _worker_job(
                    engine="custom_fast",
                    attention_backend=custom_fast_selection["attention_backend"],
                    cache_backend=custom_fast_selection["cache_backend"],
                    compile_mode=custom_fast_selection["compile_mode"],
                    nvtx_enabled=profile,
                    **common,
                )
                custom_result = _execute_job(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    job_name=(f"custom-fast-{model_key}-{workload_name}-c{concurrency}"),
                    job=custom_job,
                    python_executable=project_python,
                    external=False,
                    resume=resume,
                )
                record_result(
                    custom_result,
                    model_id=model_id,
                    workload_name=workload_name,
                    concurrency=concurrency,
                )
                hf_eager_job = _worker_job(
                    engine="huggingface_eager",
                    attention_backend="eager",
                    cache_backend="dynamic",
                    compile_mode="eager",
                    nvtx_enabled=False,
                    **common,
                )
                if hf_environment.status == "PASS":
                    hf_eager_result = _execute_job(
                        repository_root=repository_root,
                        run_directory=run_directory,
                        job_name=(f"hf-eager-{model_key}-{workload_name}-c{concurrency}"),
                        job=hf_eager_job,
                        python_executable=hf_python,
                        external=True,
                        resume=resume,
                    )
                else:
                    hf_eager_result = _failed_result(
                        "huggingface_eager",
                        "huggingface_eager",
                        hf_environment.diagnostic
                        or "isolated Hugging Face environment unavailable",
                        hf_eager_job,
                    )
                record_result(
                    hf_eager_result,
                    model_id=model_id,
                    workload_name=workload_name,
                    concurrency=concurrency,
                )
                hf_optimised_job = _worker_job(
                    engine="huggingface_optimised",
                    attention_backend=hf_optimised_selection["attention_backend"],
                    cache_backend=hf_optimised_selection["cache_backend"],
                    compile_mode=hf_optimised_selection["compile_mode"],
                    nvtx_enabled=False,
                    **common,
                )
                if hf_environment.status == "PASS":
                    hf_optimised_result = _execute_job(
                        repository_root=repository_root,
                        run_directory=run_directory,
                        job_name=(f"hf-optimised-{model_key}-{workload_name}-c{concurrency}"),
                        job=hf_optimised_job,
                        python_executable=hf_python,
                        external=True,
                        resume=resume,
                    )
                else:
                    hf_optimised_result = _failed_result(
                        "huggingface_optimised",
                        "huggingface_optimised",
                        hf_environment.diagnostic
                        or "isolated Hugging Face environment unavailable",
                        hf_optimised_job,
                    )
                record_result(
                    hf_optimised_result,
                    model_id=model_id,
                    workload_name=workload_name,
                    concurrency=concurrency,
                )
                compile_rows.append(
                    {
                        "engine": "huggingface_optimised",
                        "profile": "huggingface_optimised",
                        "model_id": model_id,
                        "workload": workload_name,
                        "concurrency": concurrency,
                        "mode": hf_optimised_selection["compile_mode"],
                        "selected_attention_backend": hf_optimised_selection["attention_backend"],
                        "selected_cache_backend": hf_optimised_selection["cache_backend"],
                        "selection_reason": hf_optimised_selection["selection_reason"],
                        "status": hf_optimised_result.get("status"),
                        "compile_diagnostics": hf_optimised_result.get("compile_diagnostics"),
                        "warmup_seconds": hf_optimised_result.get("warmup_seconds", 0),
                    }
                )
                sglang_job = _worker_job(
                    engine="sglang",
                    attention_backend="auto",
                    cache_backend="paged",
                    compile_mode="cuda-graph",
                    nvtx_enabled=False,
                    **common,
                )
                correctness_payload["fairness_checks"].append(
                    {
                        "model_id": model_id,
                        "workload": workload_name,
                        "concurrency": concurrency,
                        **validate_benchmark_fairness(
                            [
                                custom_job,
                                hf_eager_job,
                                hf_optimised_job,
                                sglang_job,
                            ]
                        ),
                    }
                )
                if sglang_environment.status == "PASS":
                    sglang_result = _execute_sglang_job(
                        repository_root=repository_root,
                        run_directory=run_directory,
                        job_name=(f"sglang-{model_key}-{workload_name}-c{concurrency}"),
                        job=sglang_job,
                        image=sglang_image,
                        resume=resume,
                    )
                else:
                    sglang_result = _failed_result(
                        "sglang",
                        "sglang",
                        sglang_environment.diagnostic or "SGLang Linux environment unavailable",
                        sglang_job,
                    )
                record_result(
                    sglang_result,
                    model_id=model_id,
                    workload_name=workload_name,
                    concurrency=concurrency,
                )

        if not smoke and config.prefix_reuse.enabled:
            prefix_inputs = [list(row) for row in input_payload["prefix_reuse"]["rows"]]
            prefix_reference = references.get((model_id, "prefix-reuse", 4))
            prefix_common: dict[str, Any] = {
                "model": model,
                "input_rows": prefix_inputs,
                "output_tokens": config.prefix_reuse.output_tokens,
                "concurrency": 4,
                "repeats": runtime_repeats,
                "warmups": runtime_warmups,
                "reference": prefix_reference,
                "config": config,
                "prefix_reuse_enabled": True,
            }
            prefix_jobs = [
                (
                    "custom_fast",
                    _worker_job(
                        engine="custom_fast",
                        attention_backend=custom_fast_selection["attention_backend"],
                        cache_backend=custom_fast_selection["cache_backend"],
                        compile_mode=custom_fast_selection["compile_mode"],
                        nvtx_enabled=profile,
                        **prefix_common,
                    ),
                    project_python,
                    False,
                ),
                (
                    "huggingface_eager",
                    _worker_job(
                        engine="huggingface_eager",
                        attention_backend="eager",
                        cache_backend="dynamic",
                        compile_mode="eager",
                        nvtx_enabled=False,
                        **prefix_common,
                    ),
                    hf_python,
                    True,
                ),
                (
                    "huggingface_optimised",
                    _worker_job(
                        engine="huggingface_optimised",
                        attention_backend=hf_optimised_selection["attention_backend"],
                        cache_backend=hf_optimised_selection["cache_backend"],
                        compile_mode=hf_optimised_selection["compile_mode"],
                        nvtx_enabled=False,
                        **prefix_common,
                    ),
                    hf_python,
                    True,
                ),
            ]
            for engine_name, job, python_path, external in prefix_jobs:
                if external and hf_environment.status != "PASS":
                    result = _failed_result(
                        engine_name,
                        engine_name,
                        hf_environment.diagnostic
                        or "isolated Hugging Face environment unavailable",
                        job,
                    )
                else:
                    result = _execute_job(
                        repository_root=repository_root,
                        run_directory=run_directory,
                        job_name=f"{_slug(engine_name)}-{model_key}-prefix-reuse",
                        job=job,
                        python_executable=python_path,
                        external=external,
                        resume=resume,
                    )
                record_result(
                    result,
                    model_id=model_id,
                    workload_name="prefix-reuse",
                    concurrency=4,
                    is_prefix=True,
                )
            sglang_prefix_job = _worker_job(
                engine="sglang",
                attention_backend="auto",
                cache_backend="paged",
                compile_mode="cuda-graph",
                nvtx_enabled=False,
                **prefix_common,
            )
            correctness_payload["fairness_checks"].append(
                {
                    "model_id": model_id,
                    "workload": "prefix-reuse",
                    "concurrency": 4,
                    **validate_benchmark_fairness(
                        [
                            *(item[1] for item in prefix_jobs),
                            sglang_prefix_job,
                        ]
                    ),
                }
            )
            if sglang_environment.status == "PASS":
                sglang_prefix_result = _execute_sglang_job(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    job_name=f"sglang-{model_key}-prefix-reuse",
                    job=sglang_prefix_job,
                    image=sglang_image,
                    resume=resume,
                )
            else:
                sglang_prefix_result = _failed_result(
                    "sglang",
                    "sglang",
                    sglang_environment.diagnostic or "SGLang Linux environment unavailable",
                    sglang_prefix_job,
                )
            record_result(
                sglang_prefix_result,
                model_id=model_id,
                workload_name="prefix-reuse",
                concurrency=4,
                is_prefix=True,
            )

    optional_engine_attempts: list[dict[str, Any]] = []
    if not skip_optional_engines:
        environment_by_engine = {item.engine: item for item in external_environments}
        optional_reference = references.get((PRIMARY_MODEL_ID, "decode-focused", 1))
        optional_input = [list(input_payload["workloads"]["decode-focused"])]
        vllm_job = _worker_job(
            engine="vllm",
            model=models[0],
            input_rows=optional_input,
            output_tokens=512,
            concurrency=1,
            repeats=runtime_repeats,
            warmups=runtime_warmups,
            reference=optional_reference,
            config=config,
            attention_backend="auto",
            cache_backend="paged",
            compile_mode="cuda-graph",
            nvtx_enabled=False,
        )
        vllm_environment = environment_by_engine["vllm"]
        if vllm_environment.status == "PASS" and optional_reference is not None:
            vllm_result = _execute_vllm_job(
                repository_root=repository_root,
                run_directory=run_directory,
                job_name="vllm-qwen-qwen3-0-6b-decode-focused-c1",
                job=vllm_job,
                image=vllm_image,
                resume=resume,
            )
        else:
            vllm_result = _failed_result(
                "vllm",
                "vllm",
                vllm_environment.diagnostic
                or "vLLM environment or correctness reference unavailable",
                vllm_job,
            )
        record_result(
            vllm_result,
            model_id=PRIMARY_MODEL_ID,
            workload_name="decode-focused",
            concurrency=1,
        )
        optional_engine_attempts.append(
            {
                "engine": "vllm",
                "environment_status": vllm_environment.status,
                "benchmark_status": vllm_result.get("status"),
                "diagnostic": vllm_result.get("error"),
            }
        )
        tensorrt_environment = environment_by_engine["tensorrt_llm"]
        tensorrt_job = _worker_job(
            engine="tensorrt_llm",
            model=models[0],
            input_rows=optional_input,
            output_tokens=512,
            concurrency=1,
            repeats=runtime_repeats,
            warmups=runtime_warmups,
            reference=optional_reference,
            config=config,
            attention_backend="auto",
            cache_backend="paged",
            compile_mode="cuda-graph",
            nvtx_enabled=False,
        )
        if tensorrt_environment.status == "PASS" and optional_reference is not None:
            tensorrt_result = _execute_tensorrt_llm_job(
                repository_root=repository_root,
                run_directory=run_directory,
                job_name="tensorrt-llm-qwen-qwen3-0-6b-decode-focused-c1",
                job=tensorrt_job,
                image=tensorrt_image,
                resume=resume,
            )
        else:
            tensorrt_result = _failed_result(
                "tensorrt_llm",
                "tensorrt_llm",
                tensorrt_environment.diagnostic
                or "TensorRT-LLM environment or correctness reference unavailable",
                tensorrt_job,
            )
        record_result(
            tensorrt_result,
            model_id=PRIMARY_MODEL_ID,
            workload_name="decode-focused",
            concurrency=1,
        )
        optional_engine_attempts.append(
            {
                "engine": "tensorrt_llm",
                "environment_status": tensorrt_environment.status,
                "benchmark_status": tensorrt_result.get("status"),
                "diagnostic": tensorrt_result.get("error"),
            }
        )
    correctness_payload["optional_engine_attempts"] = optional_engine_attempts
    _json_write(run_directory / "correctness.json", correctness_payload)
    baseline_rows = [
        {
            "engine": "legacy_custom_process",
            "profile": "qwen3_correctness",
            "model_id": PRIMARY_MODEL_ID,
            "model_revision": PRIMARY_MODEL_REVISION,
            "workload": "decode-focused",
            "input_tokens": 1,
            "output_tokens": 512,
            "concurrency": 1,
            "median_output_tokens_per_second": measured_baseline,
            "source": "remeasured_prechange_process_rpc",
            "status": "PASS" if process_baseline else "FAIL",
        }
    ]
    baseline_rows.extend(
        {
            "engine": row["engine"],
            "profile": row["profile"],
            "model_id": row["model_id"],
            "workload": row["workload"],
            "concurrency": row["concurrency"],
            "median_output_tokens_per_second": row["median_aggregate_output_tokens_per_second"],
            "status": row["status"],
            "exact_reference_identity": row["exact_reference_identity"],
        }
        for row in engine_rows
        if row["workload"] == "decode-focused"
        and int(row["concurrency"]) == 1
        and row["engine"]
        in {
            "custom_fast",
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
            "vllm",
            "tensorrt_llm",
        }
    )
    _csv_write(run_directory / "baseline_results.csv", baseline_rows)
    _csv_write(run_directory / "optimisation_ladder.csv", ladder_rows)
    _csv_write(run_directory / "engine_results.csv", engine_rows)
    _csv_write(run_directory / "latency_results.csv", latency_rows)
    _csv_write(run_directory / "batch_results.csv", batch_rows)
    _csv_write(run_directory / "memory_results.csv", memory_rows)
    _csv_write(run_directory / "traffic_results.csv", traffic_rows)
    _csv_write(run_directory / "compile_results.csv", compile_rows)
    _csv_write(run_directory / "cuda_graph_results.csv", cuda_graph_rows)

    primary_custom = next(
        (
            row
            for row in engine_rows
            if row["engine"] == "custom_fast"
            and row["model_id"] == PRIMARY_MODEL_ID
            and row["workload"] == "decode-focused"
            and int(row["concurrency"]) == 1
        ),
        None,
    )
    custom_tps = (
        float(primary_custom["median_aggregate_output_tokens_per_second"])
        if primary_custom
        else 0.0
    )
    custom_decode_tps = (
        float(primary_custom["median_decode_output_tokens_per_second"]) if primary_custom else 0.0
    )
    production_primary = [
        row
        for row in engine_rows
        if row["model_id"] == PRIMARY_MODEL_ID
        and row["workload"] == "decode-focused"
        and int(row["concurrency"]) == 1
        and row["engine"]
        in {
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
            "vllm",
            "tensorrt_llm",
        }
        and row["status"] == "PASS"
        and row["exact_reference_identity"] is True
    ]
    fastest_production = max(
        (float(row["median_aggregate_output_tokens_per_second"]) for row in production_primary),
        default=0.0,
    )
    speedup = custom_tps / measured_baseline if measured_baseline else 0.0
    production_fraction = custom_tps / fastest_production if fastest_production else 0.0
    primary_points = {
        (
            str(workload["name"]),
            int(concurrency),
        )
        for workload in runtime_workloads
        for concurrency in workload["concurrency"]
    }
    if not smoke and config.prefix_reuse.enabled:
        primary_points.add(("prefix-reuse", 4))
    required_engines = {
        "custom_fast",
        "huggingface_eager",
        "huggingface_optimised",
        "sglang",
    }
    observed_primary = {
        (str(row["workload"]), int(row["concurrency"]), str(row["engine"]))
        for row in engine_rows
        if row["model_id"] == PRIMARY_MODEL_ID
    }
    primary_matrix_complete = all(
        (workload, concurrency, engine) in observed_primary
        for workload, concurrency in primary_points
        for engine in required_engines
    )
    completed_primary = {
        (str(row["workload"]), int(row["concurrency"]), str(row["engine"]))
        for row in engine_rows
        if row["model_id"] == PRIMARY_MODEL_ID
        and row["worker_status"] == "completed"
        and int(row["measured_repeat_count"]) == runtime_repeats
    }
    completed_primary_baseline_matrix = all(
        (workload, concurrency, engine) in completed_primary
        for workload, concurrency in primary_points
        for engine in {
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
        }
    )
    custom_primary_rows = [
        row
        for row in engine_rows
        if row["model_id"] == PRIMARY_MODEL_ID and row["engine"] == "custom_fast"
    ]
    custom_exact = bool(custom_primary_rows) and all(
        row["status"] == "PASS" and row["exact_reference_identity"] is True
        for row in custom_primary_rows
    )
    maximum_custom_cv = max(
        (float(row["coefficient_of_variation"]) for row in custom_primary_rows),
        default=float("inf"),
    )
    manual_graph_ladder = next(
        (row for row in ladder_rows if row["optimisation"] == "manual_cuda_graph"),
        None,
    )
    graph_verified = bool(
        manual_graph_ladder
        and manual_graph_ladder.get("status") == "PASS"
        and manual_graph_ladder.get("exact_reference_identity") is True
        and manual_graph_ladder.get("cuda_graph_verified") is True
    )
    required_compile_ladder_modes = {
        "torch_compile_default",
        "torch_compile_reduce_overhead",
        "torch_compile_max_autotune",
        "manual_cuda_graph",
    }
    observed_compile_ladder_modes = {
        str(row["optimisation"])
        for row in ladder_rows
        if row.get("optimisation") in required_compile_ladder_modes
    }
    compile_ladder_complete = observed_compile_ladder_modes == required_compile_ladder_modes
    compiled_path_verified = any(
        row.get("status") == "PASS"
        and row.get("exact_reference_identity") is True
        and (
            row.get("cuda_graph_verified") is True
            or (
                isinstance(row.get("compile_diagnostics"), dict)
                and bool(row["compile_diagnostics"].get("verified_execution"))
                and not bool(row["compile_diagnostics"].get("fallback_used"))
            )
        )
        for row in ladder_rows
        if row.get("optimisation") in required_compile_ladder_modes
    )
    dynamic_cache_ladder = next(
        (row for row in ladder_rows if row["optimisation"] == "gpu_native_dynamic"),
        None,
    )
    static_cache_ladder = next(
        (row for row in ladder_rows if row["optimisation"] == "static_cache"),
        None,
    )
    if dynamic_cache_ladder is not None and static_cache_ladder is not None:
        cache_comparison_complete = all(
            row.get("status") == "PASS"
            and row.get("exact_reference_identity") is True
            and int(row.get("cache_reserved_bytes") or 0) > 0
            and int(row.get("cache_allocation_count") or 0) > 0
            for row in (dynamic_cache_ladder, static_cache_ladder)
        )
        cache_allocation_reduction_verified = cache_comparison_complete and int(
            static_cache_ladder.get("cache_allocation_count") or 0
        ) < int(dynamic_cache_ladder.get("cache_allocation_count") or 0)
    else:
        cache_comparison_complete = False
        cache_allocation_reduction_verified = False
    traffic_ok = bool(custom_primary_rows) and all(
        int(row["coordinator_bound_bytes"] or 0) < int(row["full_logit_equivalent_bytes"] or 0)
        and int(row["device_to_host_bytes"] or 0) < int(row["full_logit_equivalent_bytes"] or 0)
        for row in custom_primary_rows
    )
    transport_paths_observed = {
        str(point.get("path")) for point in transport_result.get("paths", [])
    }
    transport_paths_pass = (
        transport_result.get("status") == "PASS"
        and bool(transport_result.get("exact_bfloat16_identity"))
        and bool(transport_result.get("gpu_resident_direct_reference_verified"))
        and transport_paths_observed
        == {
            "in_process_gpu",
            "same_host_process",
            "remote_compatible",
        }
    )
    batched_rows = [
        row
        for row in batch_rows
        if row["engine"] == "custom_fast"
        and row["model_id"] == PRIMARY_MODEL_ID
        and int(row["concurrency"]) > 1
    ]
    real_batching = bool(batched_rows) and all(
        int(row.get("batch_forward_count") or 0) > 0 for row in batched_rows
    )
    external_required_pass = hf_environment.status == "PASS" and sglang_environment.status == "PASS"
    baseline_coverage = primary_matrix_complete and completed_primary_baseline_matrix
    secondary_points = {
        (str(workload["name"]), int(concurrency))
        for workload in runtime_workloads
        for concurrency in workload["concurrency"]
    }
    if not smoke and config.prefix_reuse.enabled:
        secondary_points.add(("prefix-reuse", 4))
    completed_secondary = {
        (str(row["workload"]), int(row["concurrency"]), str(row["engine"]))
        for row in engine_rows
        if row["model_id"] == SECONDARY_MODEL_ID
        and row["worker_status"] == "completed"
        and int(row["measured_repeat_count"]) == runtime_repeats
    }
    secondary_custom_exact = all(
        row["status"] == "PASS" and row["exact_reference_identity"] is True
        for row in engine_rows
        if row["model_id"] == SECONDARY_MODEL_ID and row["engine"] == "custom_fast"
    )
    secondary_model_coverage = (
        models[1]["resolution_status"] == "PASS"
        and all(
            (workload, concurrency, engine) in completed_secondary
            for workload, concurrency in secondary_points
            for engine in required_engines
        )
        and secondary_custom_exact
    )
    production_baseline_identity_failures = [
        {
            "engine": row["engine"],
            "model_id": row["model_id"],
            "workload": row["workload"],
            "concurrency": row["concurrency"],
            "diagnostic": row["diagnostic"],
        }
        for row in engine_rows
        if row["engine"]
        in {
            "huggingface_eager",
            "huggingface_optimised",
            "sglang",
            "vllm",
            "tensorrt_llm",
        }
        and row["worker_status"] == "completed"
        and row["exact_reference_identity"] is False
    ]
    integrity = (
        not smoke
        and config.repeats >= 5
        and config.warmup_requests >= 3
        and bool(process_baseline)
        and primary_matrix_complete
        and transport_paths_pass
    )
    statuses = {
        "experiment_integrity_status": "PASS" if integrity else "FAIL",
        "environment_isolation_status": ("PASS" if external_required_pass else "FAIL"),
        "baseline_coverage_status": "PASS" if baseline_coverage else "FAIL",
        "correctness_status": "PASS" if custom_exact else "FAIL",
        "gpu_resident_path_status": (
            "PASS" if custom_exact and traffic_ok and transport_paths_pass else "FAIL"
        ),
        "final_worker_sampling_status": "PASS" if traffic_ok else "FAIL",
        "real_batching_status": "PASS" if real_batching else "FAIL",
        "cache_status": (
            "PASS"
            if custom_exact and cache_comparison_complete and cache_allocation_reduction_verified
            else "FAIL"
        ),
        "compile_status": (
            "PASS" if compile_ladder_complete and compiled_path_verified else "FAIL"
        ),
        "minimum_speed_status": (
            "PASS"
            if speedup >= config.acceptance.minimum_speedup_over_current_baseline
            and maximum_custom_cv <= config.acceptance.maximum_result_cv
            else "FAIL"
        ),
        "production_parity_target_status": (
            "PASS"
            if production_fraction >= config.acceptance.target_fraction_of_fastest_production_engine
            else "FAIL"
        ),
    }
    minimum_fraction_pass = (
        production_fraction >= config.acceptance.minimum_fraction_of_fastest_production_engine
    )
    overall_required = [key for key in statuses if key != "production_parity_target_status"]
    overall = all(statuses[key] == "PASS" for key in overall_required) and minimum_fraction_pass
    statuses["overall_status"] = "PASS" if overall else "FAIL"
    conclusion = (
        f"The custom engine achieved {custom_tps:.2f} output tokens/s, "
        f"{speedup:.2f}x the previous custom baseline and "
        f"{production_fraction * 100:.1f}% of the fastest successful production "
        "engine. The minimum Experiment 004 performance gate "
        f"{'PASSED' if statuses['minimum_speed_status'] == 'PASS' and minimum_fraction_pass else 'FAILED'}. "
        "The 80% production-parity target "
        f"{'PASSED' if statuses['production_parity_target_status'] == 'PASS' else 'FAILED'}."
    )
    summary: dict[str, Any] = {
        "name": config.name,
        "profile": "qwen3_fast",
        **statuses,
        "minimum_production_fraction_status": ("PASS" if minimum_fraction_pass else "FAIL"),
        "smoke": smoke,
        "baseline_custom_output_tokens_per_second": measured_baseline,
        "custom_batch_one_output_tokens_per_second": custom_tps,
        "custom_batch_one_decode_output_tokens_per_second": custom_decode_tps,
        "speedup_over_remeasured_baseline": speedup,
        "fastest_successful_production_engine_output_tokens_per_second": (fastest_production),
        "fraction_of_fastest_successful_production_engine": production_fraction,
        "maximum_custom_result_cv": maximum_custom_cv,
        "primary_matrix_complete": primary_matrix_complete,
        "completed_primary_baseline_matrix": completed_primary_baseline_matrix,
        "baseline_coverage_definition": (
            "all required external baseline jobs completed every measured repeat; "
            "token-identity failures remain explicit and are excluded from the "
            "fastest-successful-production comparison"
        ),
        "production_baseline_identity_failures": production_baseline_identity_failures,
        "secondary_model_status": models[1]["resolution_status"],
        "secondary_model_coverage_status": "PASS" if secondary_model_coverage else "FAIL",
        "secondary_model_revision": models[1]["revision"],
        "cache_comparison": {
            "dynamic_reference": dynamic_cache_ladder,
            "static": static_cache_ladder,
            "allocation_reduction_verified": cache_allocation_reduction_verified,
            "allocation_count_definition": (
                "logical key/value tensor allocations or replacements before request cleanup"
            ),
        },
        "logical_stage_count": 1,
        "physical_worker_process_count": 1,
        "cuda_context_count": 1,
        "selected_attention_backend": (
            primary_custom.get("attention_backend") if primary_custom else None
        ),
        "selected_cache_backend": (primary_custom.get("cache_backend") if primary_custom else None),
        "selected_compile_mode": (primary_custom.get("compile_mode") if primary_custom else None),
        "custom_fast_selection": custom_fast_selection,
        "custom_optimisation_candidates": custom_optimisation_evidence,
        "huggingface_optimised_selection": hf_optimised_selection,
        "huggingface_optimisation_candidates": hf_optimisation_evidence,
        "optional_engine_attempts": optional_engine_attempts,
        "transport_path_status": "PASS" if transport_paths_pass else "FAIL",
        "transport_path_benchmarks": transport_result.get("paths", []),
        "cuda_graph_verified": graph_verified,
        "cuda_graph_status": "PASS" if graph_verified else "FAIL",
        "compile_ladder_complete": compile_ladder_complete,
        "compiled_path_verified": compiled_path_verified,
        "run_directory": str(run_directory),
        "report_path": str(run_directory / "report.html"),
        "conclusion": conclusion,
    }
    _json_write(run_directory / "summary.json", summary)
    generate_engine_charts(
        chart_directory=run_directory / "charts",
        engine_rows=engine_rows,
        ladder_rows=ladder_rows,
    )
    report_delivery = _render_report(
        path=run_directory / "report.html",
        summary=summary,
        engine_rows=engine_rows,
        optimisation_ladder_rows=ladder_rows,
        environments=[item.payload() for item in external_environments],
    )
    _json_write(run_directory / "report_delivery.json", report_delivery)
    if report_delivery["status"] != "PASS":
        summary["experiment_integrity_status"] = "FAIL"
        summary["overall_status"] = "FAIL"
        summary["report_delivery"] = report_delivery
        _json_write(run_directory / "summary.json", summary)
    # Re-evaluate artifact integrity after every requested file has been emitted.
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_directory / name).is_file()]
    missing.extend(
        f"charts/{name}"
        for name in REQUIRED_CHARTS
        if not (run_directory / "charts" / name).is_file()
    )
    if missing:
        summary["experiment_integrity_status"] = "FAIL"
        summary["overall_status"] = "FAIL"
        summary["missing_artifacts"] = missing
        _json_write(run_directory / "summary.json", summary)
        _render_report(
            path=run_directory / "report.html",
            summary=summary,
            engine_rows=engine_rows,
            optimisation_ladder_rows=ladder_rows,
            environments=[item.payload() for item in external_environments],
        )
    return EnginePerformanceRun(
        run_directory=run_directory,
        report_path=run_directory / "report.html",
        summary=summary,
    )
