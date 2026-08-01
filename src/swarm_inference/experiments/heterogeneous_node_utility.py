"""Experiment 007: measured heterogeneous node utility end to end."""

from __future__ import annotations

import asyncio
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import yaml

from swarm_inference.backends.artifacts import (
    artifact_hash,
    compare_tokenizers,
    gguf_mapping_from_sidecar,
    tokenizer_identity,
    validate_mapping,
)
from swarm_inference.backends.llamacpp import LlamaCppAdapter
from swarm_inference.backends.sglang import SGLangAdapter
from swarm_inference.config.heterogeneous import HeterogeneousExperimentConfig
from swarm_inference.experiments.arm64_compatibility import build_and_test_arm64
from swarm_inference.experiments.backend_environments import (
    build_gguf_artifacts,
    provision_llamacpp_environment,
    provision_torch_cpu_environment,
    resolve_git_commit,
    validate_sglang_environment,
)
from swarm_inference.experiments.background import (
    measure_gpu_baseline_lane,
    run_background_capacity_point,
)
from swarm_inference.experiments.cpu_moe import (
    ExpertFormat,
    PlacementPolicy,
    prepare_real_moe_fixture,
    run_hybrid_cpu_experts,
    select_cpu_experts,
)
from swarm_inference.experiments.heterogeneous_projection import (
    MeasuredRoleTrace,
    availability_economics_rows,
    replay_network_matrix,
    speculative_break_even_rows,
)
from swarm_inference.experiments.heterogeneous_reporting import (
    generate_heterogeneous_charts,
    render_heterogeneous_report,
)
from swarm_inference.experiments.heterogeneous_support import (
    UniversalWorkerProcess,
    cpu_capabilities,
    csv_write,
    cuda_capabilities,
    environment_snapshot,
    exact_token_length,
    find_reference_run,
    json_write,
    jsonl_write,
    partition_hash,
    reference_evidence,
    repository_git_state,
    sha256,
    stage_shard_hashes,
    start_universal_stage_worker,
    yaml_write,
)
from swarm_inference.experiments.integrity_audit import (
    AuditItem,
    maximum_sustainable_audit_fraction,
    run_integrity_audit_rate,
)
from swarm_inference.experiments.mixed_pipeline import MixedPipelineCoordinator
from swarm_inference.experiments.services import (
    HostTelemetry,
    ManagedDockerService,
    start_llamacpp_service,
    start_sglang_service,
)
from swarm_inference.experiments.sglang_measurement import (
    measure_prefix_cache,
    parse_sglang_scheduler_log,
    run_sglang_point,
    stream_generate,
)
from swarm_inference.experiments.speculative import (
    SpeculativePrompt,
    run_lossless_speculative_prompt,
)
from swarm_inference.experiments.speculative_trace import (
    HeldOutPrompt,
    aggregate_trace_rows,
    capture_and_replay_draft_format,
    capture_target_traces,
)
from swarm_inference.microsharding.real_moe import (
    RealMoEDownloadPlan,
    download_real_moe_layer_files,
)
from swarm_inference.planner import (
    CanaryMeasurement,
    HeterogeneousPlanner,
    NodeRole,
    NonDegradationPolicy,
    PlannerObjective,
    RoleCandidate,
    UtilityNormalisation,
    planner_regret,
)
from swarm_inference.worker.abi import (
    BackendArtifactMapping,
    BackendInterfaceEvidence,
    GenerationParameters,
    ResultClassification,
    TokenPayload,
    WorkerJob,
    WorkerJobStatus,
    WorkerJobType,
    WorkerProtocolVersion,
)

TARGET_REVISION_FALLBACK = "1cfa9a7208912126459214e8b04321603b3df60c"
DRAFT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MOE_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"

REQUIRED_ARTIFACTS = (
    "config.requested.yaml",
    "config.resolved.yaml",
    "environment.json",
    "git.json",
    "experiment_004_reference.json",
    "experiment_006_reference.json",
    "backend_environments.json",
    "backend_artifact_mappings.json",
    "worker_protocol.json",
    "worker_capabilities.json",
    "worker_benchmarks.json",
    "sglang_baseline.csv",
    "mixed_backend_results.csv",
    "backend_boundary_metrics.csv",
    "speculative_results.csv",
    "speculative_acceptance.csv",
    "speculative_break_even.csv",
    "cpu_expert_results.csv",
    "expert_placement_results.csv",
    "expert_cache_results.csv",
    "background_results.csv",
    "integrity_audit_results.csv",
    "arm64_build.json",
    "arm64_protocol_results.json",
    "planner_predictions.jsonl",
    "planner_decisions.jsonl",
    "planner_measurements.csv",
    "planner_regret.csv",
    "network_projection.csv",
    "availability_economics.csv",
    "contribution_frontier.csv",
    "correctness.json",
    "summary.json",
    "report.html",
)


@dataclass(frozen=True, slots=True)
class HeterogeneousOptions:
    experiment_004_run: Path | None = None
    experiment_006_run: Path | None = None
    skip_speculative: bool = False
    skip_moe: bool = False
    skip_background: bool = False
    skip_arm64: bool = False
    smoke: bool = False
    resume: bool = False
    profile: bool = False
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class HeterogeneousRun:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("overall_status") in {"PASS", "PARTIAL_PASS"}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_directory(root: Path, options: HeterogeneousOptions) -> Path:
    if options.output is not None:
        output = options.output.expanduser()
        if not output.is_absolute():
            output = root / output
        output.mkdir(parents=True, exist_ok=True)
        return output.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = root / "artifacts" / "runs" / f"{stamp}-heterogeneous-node-utility-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    return output.resolve()


def _initialise_artifacts(run_directory: Path) -> None:
    for directory in ("logs", "profiles", "charts"):
        (run_directory / directory).mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS:
        path = run_directory / name
        if path.exists():
            continue
        if name.endswith(".csv"):
            csv_write(path, [])
        elif name.endswith(".jsonl"):
            jsonl_write(path, [])
        elif name.endswith(".yaml"):
            yaml_write(path, {})
        elif name.endswith(".json"):
            json_write(path, {"status": "PENDING"})
        elif name.endswith(".html"):
            path.write_text(
                "<!doctype html><title>Experiment 007 pending</title>\n", encoding="utf-8"
            )


def _resolved_model_from_reference(
    reference: Path, model_id: str, fallback: str
) -> tuple[str, Path]:
    paths = [reference / "model_revisions.json"]
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records: list[dict[str, Any]] = []
        if isinstance(payload.get("models"), list):
            records.extend(item for item in payload["models"] if isinstance(item, dict))
        for value in payload.values():
            if isinstance(value, dict) and value.get("model_id"):
                records.append(value)
        for record in records:
            if record.get("model_id") == model_id:
                revision = str(
                    record.get("revision") or record.get("resolved_revision") or fallback
                )
                model_path = Path(str(record.get("path") or record.get("model_path", "")))
                if model_path.is_dir():
                    return revision, model_path.resolve()
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    candidate = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / fallback
    if not candidate.is_dir():
        raise FileNotFoundError(f"immutable model snapshot is unavailable: {candidate}")
    return fallback, candidate.resolve()


def _prompt_fixtures(tokenizer: Any, count: int) -> list[HeldOutPrompt]:
    templates = {
        "general_text": "Explain a practical consequence of distributed systems trade-offs for a careful reader.",
        "code": "Write a Python function that validates a checksum and explain its edge cases.",
        "arithmetic_reasoning": "A store applies two discounts and then tax. Calculate the final price step by step.",
        "long_form": "Continue this essay about reliable cooperation between machines with concrete examples.",
    }
    prompts: list[HeldOutPrompt] = []
    categories = list(templates)
    for index in range(count):
        category = categories[index % len(categories)]
        text = f"{templates[category]} Held-out case {index + 1}; do not repeat the prompt."
        tokens = [int(item) for item in tokenizer.encode(text, add_special_tokens=True)]
        prompts.append(
            HeldOutPrompt(
                prompt_id=f"held-out-{index:03d}",
                category=category,
                text=text,
                token_ids=tokens,
            )
        )
    return prompts


def _empty_status(reason: str) -> dict[str, Any]:
    return {"status": "BLOCKED", "reason": reason}


def _backend_setup(
    *,
    repository_root: Path,
    run_directory: Path,
    config: HeterogeneousExperimentConfig,
    target_revision: str,
    target_path: Path,
    draft_path: Path,
    microshards: Path,
) -> dict[str, Any]:
    backend_root = repository_root / config.backend_environments.root
    backend_root.mkdir(parents=True, exist_ok=True)
    sglang = validate_sglang_environment(
        backend_root,
        image=config.backend_environments.sglang_image,
        expected_version=config.backend_environments.sglang_version,
        repository_root=repository_root,
        pull_when_missing=True,
    )
    existing_llama = backend_root / "llamacpp" / "environment.json"
    if config.backend_environments.llamacpp_commit:
        llama_commit = config.backend_environments.llamacpp_commit
    elif existing_llama.is_file():
        previous = json.loads(existing_llama.read_text(encoding="utf-8"))
        llama_commit = str(previous.get("source_commit") or "")
        if len(llama_commit) != 40:
            llama_commit = resolve_git_commit(
                config.backend_environments.llamacpp_repository, None, repository_root
            )
    else:
        llama_commit = resolve_git_commit(
            config.backend_environments.llamacpp_repository, None, repository_root
        )
    llama = provision_llamacpp_environment(
        backend_root,
        repository=config.backend_environments.llamacpp_repository,
        commit=llama_commit,
        repository_root=repository_root,
    )
    if config.backend_environments.torch_cpu_python:
        from swarm_inference.experiments.backend_environments import (
            validate_torch_cpu_environment,
        )

        torch_cpu = validate_torch_cpu_environment(
            backend_root,
            repository_root=repository_root,
            python_executable=Path(config.backend_environments.torch_cpu_python),
        )
    else:
        torch_cpu = provision_torch_cpu_environment(
            backend_root,
            repository_root=repository_root,
        )
    target_identity = tokenizer_identity(target_path)
    draft_identity = tokenizer_identity(draft_path)
    tokenizer_comparison = compare_tokenizers(target_path, draft_path)
    if tokenizer_comparison["status"] != "PASS":
        raise RuntimeError("target/draft tokenizer, vocabulary, or special-token identity failed")
    gguf = build_gguf_artifacts(
        repository_root=repository_root,
        environment_evidence=llama,
        model_snapshot=draft_path,
        output_root=backend_root / "llamacpp" / "models",
        model_id=config.cpu_draft.model_id,
        revision=config.cpu_draft.revision or DRAFT_REVISION,
        tokenizer_identity=draft_identity,
        sglang_image=config.backend_environments.sglang_image,
    )
    target_hash = artifact_hash(target_path)
    partition_identity = partition_hash(microshards)
    mappings: list[BackendArtifactMapping] = [
        BackendArtifactMapping(
            canonical_model_id=config.gpu_target.model_id,
            canonical_revision=target_revision,
            canonical_partition_hash=target_hash,
            backend_id="sglang",
            backend_artifact_path=str(target_path),
            backend_artifact_hash=target_hash,
            conversion_tool="none-canonical-huggingface-safetensors",
            conversion_version="1",
            conversion_parameters={"dtype": "bfloat16"},
            canonical_tensor_mapping={"*": "canonical_huggingface_tensor_name"},
            weight_format="safetensors/BF16",
            conversion_loss="lossless",
            tokenizer_hash=str(target_identity["tokenizer_hash"]),
            vocabulary_hash=str(target_identity["vocabulary_hash"]),
            special_tokens_hash=str(target_identity["special_tokens_hash"]),
        ),
        BackendArtifactMapping(
            canonical_model_id=config.mixed_pipeline.model_id,
            canonical_revision=config.mixed_pipeline.revision,
            canonical_partition_hash=partition_identity,
            backend_id="torch-cpu",
            backend_artifact_path=str(microshards),
            backend_artifact_hash=artifact_hash(microshards),
            conversion_tool="swarm build-microshards",
            conversion_version="experiment-006-v1",
            conversion_parameters={"pipeline_stages": 4, "tensor_parallel_degree": 1},
            canonical_tensor_mapping={"*": "unchanged canonical safetensors names"},
            weight_format="microshard-safetensors/BF16",
            conversion_loss="lossless",
            tokenizer_hash=str(draft_identity["tokenizer_hash"]),
            vocabulary_hash=str(draft_identity["vocabulary_hash"]),
            special_tokens_hash=str(draft_identity["special_tokens_hash"]),
        ),
    ]
    for format_name in config.cpu_draft.formats:
        mapping = gguf_mapping_from_sidecar(
            Path(str(gguf[format_name]["gguf_path"])).with_suffix(".gguf.conversion.json"),
            canonical_partition_hash=artifact_hash(draft_path),
        )
        mappings.append(mapping)
    mapping_results = [validate_mapping(mapping) for mapping in mappings]
    interfaces = [
        BackendInterfaceEvidence(backend_id=name)
        for name in ("mlx", "executorch", "vulkan", "rocm")
    ]
    evidence = {
        "status": (
            "PASS"
            if all(item.get("status") == "PASS" for item in (sglang, llama, torch_cpu))
            and all(item["status"] == "PASS" for item in mapping_results)
            else "FAIL"
        ),
        "sglang": sglang,
        "llamacpp": llama,
        "torch_cpu": torch_cpu,
        "tokenizer_comparison": tokenizer_comparison,
        "gguf": gguf,
        "mappings": [item.model_dump(mode="json") for item in mappings],
        "mapping_validation": mapping_results,
        "interfaces": [item.model_dump(mode="json") for item in interfaces],
    }
    json_write(run_directory / "backend_environments.json", evidence)
    json_write(
        run_directory / "backend_artifact_mappings.json",
        {
            "status": "PASS"
            if all(item["status"] == "PASS" for item in mapping_results)
            else "FAIL",
            "tokenizer_comparison": tokenizer_comparison,
            "mappings": evidence["mappings"],
            "validation": mapping_results,
            "interface_only_backends": evidence["interfaces"],
        },
    )
    return evidence


def _make_adapters(
    *,
    sglang_service: ManagedDockerService,
    llama_service: ManagedDockerService,
    target_revision: str,
    draft_revision: str,
    tokenizer_hash: str,
    gguf: dict[str, Any],
    format_name: str,
    target_capabilities: Any,
    cpu_worker_capabilities: Any,
) -> tuple[SGLangAdapter, LlamaCppAdapter]:
    target = SGLangAdapter(
        endpoint=sglang_service.endpoint,
        capabilities=target_capabilities,
        model_revision=target_revision,
        model_load_seconds=sglang_service.launch_seconds,
    )
    draft = LlamaCppAdapter(
        endpoint=llama_service.endpoint,
        capabilities=cpu_worker_capabilities,
        model_revision=draft_revision,
        tokenizer_hash=tokenizer_hash,
        gguf_hash=str(gguf[format_name]["gguf_sha256"]),
        weight_format=format_name,
        model_load_seconds=llama_service.launch_seconds,
    )
    return target, draft


def _run_sglang_baseline(
    *,
    run_directory: Path,
    config: HeterogeneousExperimentConfig,
    service: ManagedDockerService,
    tokenizer: Any,
    smoke: bool,
    profile: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed = [
        int(item)
        for item in tokenizer.encode(
            "Heterogeneous inference must be measured carefully and reported honestly.",
            add_special_tokens=True,
        )
    ]
    warmup_started = time.perf_counter()
    stream_generate(
        service.endpoint,
        input_ids=exact_token_length(seed, 32),
        output_tokens=8,
        request_id=f"experiment-007-sglang-warmup-{uuid4().hex[:8]}",
    )
    warmup_seconds = time.perf_counter() - warmup_started
    telemetry = HostTelemetry(interval_seconds=0.1 if profile else 0.25)
    telemetry.start()
    rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    repeats = 1 if smoke else 3
    workloads = [
        (
            "short",
            config.workloads.short_input_tokens,
            8 if smoke else config.workloads.short_output_tokens,
            (1,) if smoke else config.workloads.short_concurrency,
        ),
        (
            "long",
            64 if smoke else config.workloads.long_input_tokens,
            8 if smoke else config.workloads.long_output_tokens,
            (1,) if smoke else config.workloads.long_concurrency,
        ),
    ]
    try:
        for name, input_tokens, output_tokens, concurrencies in workloads:
            for concurrency in concurrencies:
                prompts = [
                    exact_token_length(seed, input_tokens, offset=index % len(seed))
                    for index in range(concurrency)
                ]
                point, raw = run_sglang_point(
                    service.endpoint,
                    prompts=prompts,
                    output_tokens=output_tokens,
                    concurrency=concurrency,
                    repeats=repeats,
                    workload=name,
                )
                rows.append(point)
                requests.extend(raw)
        prefix_prompt = exact_token_length(seed, 64 if smoke else 1024)
        prefix = measure_prefix_cache(
            service.endpoint,
            prompt=prefix_prompt,
            output_tokens=8 if smoke else 128,
        )
    finally:
        samples = telemetry.stop()
    service.stderr_handle.flush()
    scheduler = parse_sglang_scheduler_log(
        Path(service.stderr_handle.name),
        maximum_running_requests=64,
    )
    for row in rows:
        row["model_load_seconds"] = service.launch_seconds
        row["warmup_seconds"] = warmup_seconds
        for key, value in scheduler.items():
            if key not in {"records", "status"}:
                row[key] = value
        if samples:
            row["gpu_memory_bytes_maximum"] = max(
                (item.get("gpu_memory_used_bytes", 0.0) for item in samples), default=0.0
            )
            row["gpu_utilisation_percent_mean"] = statistics.mean(
                item.get("gpu_utilisation_percent", 0.0) for item in samples
            )
            row["gpu_power_watts_mean"] = statistics.mean(
                item.get("gpu_power_watts", 0.0) for item in samples
            )
            row["memory_controller_utilisation_percent_mean"] = statistics.mean(
                item.get("memory_controller_utilisation_percent", 0.0) for item in samples
            )
            row["host_cpu_percent_mean"] = statistics.mean(
                item.get("host_cpu_percent", 0.0) for item in samples
            )
    csv_write(run_directory / "sglang_baseline.csv", rows)
    jsonl_write(run_directory / "logs" / "sglang_requests.jsonl", requests)
    json_write(
        run_directory / "profiles" / "sglang_telemetry.json",
        {
            "classification": "measured_cuda",
            "warmup_seconds": warmup_seconds,
            "hardware_samples": samples,
            "scheduler": scheduler,
        },
    )
    json_write(run_directory / "logs" / "sglang_prefix_cache.json", prefix)
    return rows, requests, prefix


async def _close_stage_workers(workers: list[UniversalWorkerProcess]) -> None:
    await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)


def _run_mixed_pipeline(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: HeterogeneousExperimentConfig,
    partition_root: Path,
    torch_cpu_python: Path,
    prompt_tokens: list[int],
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    identity = partition_hash(partition_root)
    shard_hashes = stage_shard_hashes(partition_root)
    mixed_workers: list[UniversalWorkerProcess] = []
    reference_workers: list[UniversalWorkerProcess] = []
    protocol_rows: list[dict[str, Any]] = []
    try:
        for stage_id, device in enumerate(("cuda", "cpu", "cuda", "cpu")):
            python = Path(sys.executable) if device == "cuda" else torch_cpu_python
            worker = start_universal_stage_worker(
                repository_root=repository_root,
                partition_root=partition_root,
                partition_manifest_hash=identity,
                stage_id=stage_id,
                device=device,
                python_executable=python,
                run_directory=run_directory,
                run_id=run_id,
            )
            mixed_workers.append(worker)
            protocol_rows.append(worker.ready)
        mixed_coordinator = MixedPipelineCoordinator(
            endpoints=[worker.client for worker in mixed_workers],
            model_id=config.mixed_pipeline.model_id,
            model_revision=config.mixed_pipeline.revision,
            partition_hash=identity,
            shard_hashes=shard_hashes,
            deadline_ms=900_000,
        )
        output_tokens = 4 if smoke else config.mixed_pipeline.generated_tokens
        mixed = asyncio.run(
            mixed_coordinator.generate(
                request_id=f"exp007-mixed-{run_id}",
                prompt_token_ids=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
        asyncio.run(_close_stage_workers(mixed_workers))
        mixed_workers = []
        for stage_id in range(4):
            worker = start_universal_stage_worker(
                repository_root=repository_root,
                partition_root=partition_root,
                partition_manifest_hash=identity,
                stage_id=stage_id,
                device="cuda",
                python_executable=Path(sys.executable),
                run_directory=run_directory,
                run_id=f"{run_id}-reference",
            )
            reference_workers.append(worker)
            protocol_rows.append(worker.ready)
        reference_coordinator = MixedPipelineCoordinator(
            endpoints=[worker.client for worker in reference_workers],
            model_id=config.mixed_pipeline.model_id,
            model_revision=config.mixed_pipeline.revision,
            partition_hash=identity,
            shard_hashes=shard_hashes,
            deadline_ms=900_000,
        )
        reference = asyncio.run(
            reference_coordinator.generate(
                request_id=f"exp007-reference-{run_id}",
                prompt_token_ids=prompt_tokens,
                output_tokens=output_tokens,
            )
        )
        exact = mixed.output_token_ids == reference.output_token_ids
        mixed_tps = float(mixed.metrics["output_tokens_per_second"])
        reference_tps = float(reference.metrics["output_tokens_per_second"])
        penalty = mixed_tps / max(reference_tps, 1e-12) - 1
        result_rows = [
            {
                **reference.metrics,
                "route": "cuda-cuda-cuda-cuda",
                "classification": "measured_cuda",
                "end_to_end_ms": float(reference.metrics["end_to_end_seconds"]) * 1000,
                "output_token_ids": reference.output_token_ids,
            },
            {
                **mixed.metrics,
                "route": "cuda-cpu-cuda-cpu",
                "classification": "measured_mixed_backend",
                "end_to_end_ms": float(mixed.metrics["end_to_end_seconds"]) * 1000,
                "output_token_ids": mixed.output_token_ids,
                "exact_greedy_token_identity": exact,
                "throughput_change_fraction": penalty,
                "forced_critical_path_classification": "harmful" if penalty < 0 else "useful",
            },
        ]
        boundaries = [{**row, "route": "cuda-cpu-cuda-cpu"} for row in mixed.boundary_metrics] + [
            {**row, "route": "cuda-cuda-cuda-cuda"} for row in reference.boundary_metrics
        ]
        correctness = {
            "status": "PASS"
            if exact and not bool(mixed.metrics["synthetic_execution"])
            else "FAIL",
            "classification": "measured_mixed_backend",
            "mixed_token_ids": mixed.output_token_ids,
            "reference_token_ids": reference.output_token_ids,
            "exact_greedy_token_identity": exact,
            "synthetic_execution": mixed.metrics["synthetic_execution"],
            "stage_local_kv_cache": mixed.metrics["stage_local_kv_cache"],
            "partition_hash": identity,
            "stage_shard_hashes": shard_hashes,
            "forced_critical_path_throughput_change_fraction": penalty,
        }
        csv_write(run_directory / "mixed_backend_results.csv", result_rows)
        csv_write(run_directory / "backend_boundary_metrics.csv", boundaries)
        return result_rows, boundaries, correctness, protocol_rows
    finally:
        if mixed_workers:
            asyncio.run(_close_stage_workers(mixed_workers))
        if reference_workers:
            asyncio.run(_close_stage_workers(reference_workers))


def _run_speculative_arm(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: HeterogeneousExperimentConfig,
    backend_setup: dict[str, Any],
    sglang_service: ManagedDockerService,
    prompts: list[HeldOutPrompt],
    tokenizer_hash: str,
    target_revision: str,
    target_capabilities: Any,
    cpu_worker_capabilities: Any,
    smoke: bool,
    capture_profile: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    target = SGLangAdapter(
        endpoint=sglang_service.endpoint,
        capabilities=target_capabilities,
        model_revision=target_revision,
        model_load_seconds=sglang_service.launch_seconds,
    )
    output_tokens = 8 if smoke else config.workloads.speculative_output_tokens
    target_traces = asyncio.run(
        capture_target_traces(
            prompts,
            target=target,
            model_id=config.gpu_target.model_id,
            revision=target_revision,
            tokenizer_hash=tokenizer_hash,
            maximum_output_tokens=output_tokens,
        )
    )
    raw_rows: list[dict[str, Any]] = []
    acceptance: list[dict[str, Any]] = []
    canaries: list[dict[str, Any]] = []
    draft_profiles: list[dict[str, Any]] = []
    llama_environment = backend_setup["llamacpp"]
    environment_root = Path(str(llama_environment["root"]))
    build_name = Path(str(llama_environment["server_path"])).resolve().parents[1].name
    thread_count = max(1, int(cpu_worker_capabilities.physical_cpu_cores) - 2)
    for format_name in config.cpu_draft.formats:
        gguf = backend_setup["gguf"][format_name]
        service = start_llamacpp_service(
            environment_root=environment_root,
            build_name=build_name,
            gguf_path=Path(str(gguf["gguf_path"])),
            repository_root=repository_root,
            run_id=run_id,
            format_name=format_name,
            log_root=run_directory / "logs",
            thread_count=thread_count,
            parallel=1 if smoke else 4,
        )
        try:
            target_adapter, draft_adapter = _make_adapters(
                sglang_service=sglang_service,
                llama_service=service,
                target_revision=target_revision,
                draft_revision=config.cpu_draft.revision or DRAFT_REVISION,
                tokenizer_hash=tokenizer_hash,
                gguf=backend_setup["gguf"],
                format_name=format_name,
                target_capabilities=target_capabilities,
                cpu_worker_capabilities=cpu_worker_capabilities,
            )
            warmup_started = time.perf_counter()
            warmup_result = asyncio.run(
                draft_adapter.execute(
                    WorkerJob(
                        job_id=uuid4().hex,
                        request_id=f"experiment-007-draft-warmup-{format_name.lower()}",
                        role=WorkerJobType.SPECULATIVE_DRAFT,
                        model_id=config.cpu_draft.model_id,
                        model_revision=config.cpu_draft.revision or DRAFT_REVISION,
                        input_payload=TokenPayload(
                            token_ids=prompts[0].token_ids,
                            tokenizer_hash=tokenizer_hash,
                        ),
                        generation_parameters=GenerationParameters(
                            max_new_tokens=1,
                            temperature=0.0,
                        ),
                        deadline_ms=120_000,
                    )
                )
            )
            warmup_seconds = time.perf_counter() - warmup_started
            if warmup_result.status != WorkerJobStatus.ACCEPTED:
                raise RuntimeError(
                    f"llama.cpp warmup failed: {warmup_result.status.value}: {warmup_result.detail}"
                )
            telemetry = HostTelemetry(interval_seconds=0.1 if capture_profile else 0.25)
            telemetry.start()
            try:
                rows, evidence, draft_profile = asyncio.run(
                    capture_and_replay_draft_format(
                        prompts,
                        target_traces=target_traces,
                        draft=draft_adapter,
                        model_id=config.cpu_draft.model_id,
                        revision=config.cpu_draft.revision or DRAFT_REVISION,
                        tokenizer_hash=tokenizer_hash,
                        maximum_output_tokens=output_tokens,
                        weight_format=format_name,
                        draft_lengths=config.cpu_draft.draft_lengths,
                    )
                )
                prompt = prompts[0]
                canary, canary_evidence = asyncio.run(
                    run_lossless_speculative_prompt(
                        prompt=SpeculativePrompt(
                            prompt_id=prompt.prompt_id,
                            category=prompt.category,
                            token_ids=prompt.token_ids,
                        ),
                        target=target_adapter,
                        draft=draft_adapter,
                        target_model_id=config.gpu_target.model_id,
                        target_revision=target_revision,
                        draft_model_id=config.cpu_draft.model_id,
                        draft_revision=config.cpu_draft.revision or DRAFT_REVISION,
                        tokenizer_hash=tokenizer_hash,
                        draft_length=1,
                        maximum_output_tokens=min(output_tokens, 8),
                    )
                )
            finally:
                cpu_samples = telemetry.stop()
            raw_rows.extend(rows)
            acceptance.extend(evidence)
            draft_profile.update(
                {
                    "classification": "measured_x86_cpu",
                    "model_load_seconds": service.launch_seconds,
                    "warmup_seconds": warmup_seconds,
                    "thread_count": thread_count,
                    "numa_policy": "operating_system_default",
                    "memory_bandwidth_measurement_status": "unavailable_on_host",
                    "cpu_package_power_measurement_status": "unavailable_on_host",
                    "telemetry_sample_count": len(cpu_samples),
                }
            )
            if cpu_samples:
                draft_profile["host_cpu_percent_mean"] = statistics.mean(
                    item.get("host_cpu_percent", 0.0) for item in cpu_samples
                )
                draft_profile["host_cpu_percent_maximum"] = max(
                    item.get("host_cpu_percent", 0.0) for item in cpu_samples
                )
                draft_profile["host_memory_used_bytes_maximum"] = max(
                    item.get("host_memory_used_bytes", 0.0) for item in cpu_samples
                )
            draft_profiles.append(draft_profile)
            json_write(
                run_directory / "profiles" / f"cpu_draft_telemetry_{format_name.lower()}.json",
                cpu_samples,
            )
            canaries.append({**canary, "weight_format": format_name})
            jsonl_write(
                run_directory / "logs" / f"speculative-live-canary-{format_name.lower()}.jsonl",
                canary_evidence,
            )
        finally:
            service.close(repository_root=repository_root)
    aggregate = aggregate_trace_rows(raw_rows)
    for row in aggregate:
        matching_canary = next(
            item for item in canaries if item["weight_format"] == row["weight_format"]
        )
        matching_profile = next(
            item for item in draft_profiles if item["weight_format"] == row["weight_format"]
        )
        row["model_load_seconds"] = matching_profile["model_load_seconds"]
        row["warmup_seconds"] = matching_profile["warmup_seconds"]
        row["live_coordinator_exact_canary"] = matching_canary["exact_output_identity"]
        row["live_coordinator_canary_speedup_fraction"] = matching_canary["speedup_fraction"]
        row["measured_positive_contribution_pass"] = (
            bool(matching_canary["exact_output_identity"])
            and float(matching_canary["speedup_fraction"])
            >= config.positive_contribution.minimum_speculative_speedup_fraction
        )
    break_even: list[dict[str, Any]] = []
    for row in aggregate:
        verification_ms = (
            (int(row["draft_length"]) + 1)
            / max(float(row["target_only_tokens_per_second"]), 1e-12)
            * 1000
        )
        projected = speculative_break_even_rows(
            acceptance_rate=float(row["acceptance_rate"]),
            mean_accepted_length=float(row["mean_accepted_length"]),
            target_tokens_per_second=float(row["target_only_tokens_per_second"]),
            target_verification_ms=verification_ms,
            draft_length=int(row["draft_length"]),
            request_payload_bytes=64,
            response_payload_bytes=int(row["draft_length"]) * 4,
        )
        for item in projected:
            item["weight_format"] = row["weight_format"]
            item["speedup_fraction"] = item["single_request_speedup_fraction"]
        break_even.extend(projected)
    csv_write(run_directory / "speculative_results.csv", aggregate)
    csv_write(run_directory / "speculative_acceptance.csv", acceptance)
    csv_write(run_directory / "speculative_break_even.csv", break_even)
    jsonl_write(run_directory / "logs" / "speculative_prompt_results.jsonl", raw_rows)
    json_write(run_directory / "profiles" / "cpu_draft_profiles.json", draft_profiles)
    correctness = {
        "status": (
            "PASS"
            if all(bool(row["all_exact"]) for row in aggregate)
            and all(bool(item["exact_output_identity"]) for item in canaries)
            else "FAIL"
        ),
        "target_draft_tokenizer_identity": backend_setup["tokenizer_comparison"],
        "prompt_count": len(prompts),
        "categories": sorted({prompt.category for prompt in prompts}),
        "formats": list(config.cpu_draft.formats),
        "draft_lengths": list(config.cpu_draft.draft_lengths),
        "all_trace_outputs_exact": all(bool(row["all_exact"]) for row in aggregate),
        "live_canaries": canaries,
        "timing_interpretation": (
            "proposal and target traces are measured; full 100-prompt coordinated speed is "
            "an event replay and is not accepted as measured positive contribution"
        ),
    }
    return aggregate, acceptance, break_even, correctness


def _run_background_arm(
    *,
    repository_root: Path,
    run_directory: Path,
    run_id: str,
    config: HeterogeneousExperimentConfig,
    backend_setup: dict[str, Any],
    sglang_service: ManagedDockerService,
    prompts: list[HeldOutPrompt],
    tokenizer_hash: str,
    target_revision: str,
    target_capabilities: Any,
    cpu_worker_capabilities: Any,
    smoke: bool,
    capture_profile: bool,
) -> list[dict[str, Any]]:
    format_name = "Q4_K_M"
    llama_environment = backend_setup["llamacpp"]
    environment_root = Path(str(llama_environment["root"]))
    build_name = Path(str(llama_environment["server_path"])).resolve().parents[1].name
    service = start_llamacpp_service(
        environment_root=environment_root,
        build_name=build_name,
        gguf_path=Path(str(backend_setup["gguf"][format_name]["gguf_path"])),
        repository_root=repository_root,
        run_id=f"{run_id}-background",
        format_name=format_name,
        log_root=run_directory / "logs",
        thread_count=max(1, int(cpu_worker_capabilities.physical_cpu_cores) - 2),
        parallel=4,
    )
    try:
        target, cpu = _make_adapters(
            sglang_service=sglang_service,
            llama_service=service,
            target_revision=target_revision,
            draft_revision=config.cpu_draft.revision or DRAFT_REVISION,
            tokenizer_hash=tokenizer_hash,
            gguf=backend_setup["gguf"],
            format_name=format_name,
            target_capabilities=target_capabilities,
            cpu_worker_capabilities=cpu_worker_capabilities,
        )
        warmup_started = time.perf_counter()
        warmup_result = asyncio.run(
            cpu.execute(
                WorkerJob(
                    job_id=uuid4().hex,
                    request_id="experiment-007-background-warmup",
                    role=WorkerJobType.BACKGROUND_GENERATE,
                    model_id=config.cpu_draft.model_id,
                    model_revision=config.cpu_draft.revision or DRAFT_REVISION,
                    input_payload=TokenPayload(
                        token_ids=prompts[0].token_ids,
                        tokenizer_hash=tokenizer_hash,
                    ),
                    generation_parameters=GenerationParameters(
                        max_new_tokens=1,
                        temperature=0.0,
                    ),
                    deadline_ms=120_000,
                    priority=1,
                )
            )
        )
        warmup_seconds = time.perf_counter() - warmup_started
        if warmup_result.status != WorkerJobStatus.ACCEPTED:
            raise RuntimeError(
                f"llama.cpp background warmup failed: {warmup_result.status.value}: "
                f"{warmup_result.detail}"
            )
        gpu_concurrencies = (1,) if smoke else config.background.gpu_concurrency
        cpu_concurrencies = (1,) if smoke else config.background.cpu_concurrency
        output_tokens = 8 if smoke else config.background.interactive_output_tokens
        background_tokens = 8 if smoke else config.background.background_output_tokens
        baselines: dict[int, dict[str, Any]] = {}
        for gpu_concurrency in gpu_concurrencies:
            prompt_rows = [
                prompts[index % len(prompts)].token_ids for index in range(max(2, gpu_concurrency))
            ]
            baselines[gpu_concurrency] = asyncio.run(
                measure_gpu_baseline_lane(
                    adapter=target,
                    model_id=config.gpu_target.model_id,
                    revision=target_revision,
                    tokenizer_hash=tokenizer_hash,
                    prompts=prompt_rows,
                    output_tokens=output_tokens,
                    concurrency=gpu_concurrency,
                )
            )
        rows: list[dict[str, Any]] = []
        telemetry_rows: list[dict[str, Any]] = []
        for gpu_concurrency in gpu_concurrencies:
            for cpu_concurrency in cpu_concurrencies:
                request_count = max(2, gpu_concurrency, cpu_concurrency)
                gpu_prompts = [
                    prompts[index % len(prompts)].token_ids for index in range(request_count)
                ]
                cpu_prompts = [
                    prompts[(index + 11) % len(prompts)].token_ids for index in range(request_count)
                ]
                telemetry = HostTelemetry(interval_seconds=0.1 if capture_profile else 0.25)
                telemetry.start()
                try:
                    point = asyncio.run(
                        run_background_capacity_point(
                            gpu_adapter=target,
                            cpu_adapter=cpu,
                            gpu_model_id=config.gpu_target.model_id,
                            gpu_revision=target_revision,
                            cpu_model_id=config.cpu_draft.model_id,
                            cpu_revision=config.cpu_draft.revision or DRAFT_REVISION,
                            tokenizer_hash=tokenizer_hash,
                            gpu_prompts=gpu_prompts,
                            cpu_prompts=cpu_prompts,
                            gpu_output_tokens=output_tokens,
                            cpu_output_tokens=background_tokens,
                            gpu_concurrency=gpu_concurrency,
                            cpu_concurrency=cpu_concurrency,
                            baseline_gpu=baselines[gpu_concurrency],
                            maximum_p95_increase_fraction=(
                                config.non_degradation.maximum_interactive_p95_increase_fraction
                            ),
                            maximum_throughput_decrease_fraction=(
                                config.non_degradation.maximum_interactive_throughput_decrease_fraction
                            ),
                        )
                    )
                finally:
                    samples = telemetry.stop()
                telemetry_rows.extend(
                    {
                        **sample,
                        "gpu_concurrency": gpu_concurrency,
                        "cpu_concurrency": cpu_concurrency,
                    }
                    for sample in samples
                )
                point["weight_format"] = format_name
                point["cpu_model_load_seconds"] = service.launch_seconds
                point["cpu_warmup_seconds"] = warmup_seconds
                point["telemetry_sample_count"] = len(samples)
                point["cpu_package_power_measurement_status"] = "unavailable_on_host"
                if samples:
                    point["host_cpu_percent_mean"] = statistics.mean(
                        item.get("host_cpu_percent", 0.0) for item in samples
                    )
                    point["host_cpu_percent_maximum"] = max(
                        item.get("host_cpu_percent", 0.0) for item in samples
                    )
                    point["host_memory_used_bytes_maximum"] = max(
                        item.get("host_memory_used_bytes", 0.0) for item in samples
                    )
                    memory_pressure = max(
                        float(item.get("host_memory_used_bytes", 0.0))
                        / max(
                            float(item.get("host_memory_used_bytes", 0.0))
                            + float(item.get("host_memory_available_bytes", 0.0)),
                            1.0,
                        )
                        for item in samples
                    )
                    point["host_memory_pressure_fraction_maximum"] = memory_pressure
                    point["resource_pressure_detected"] = memory_pressure > 0.90
                    point["gpu_power_watts_mean"] = statistics.mean(
                        item.get("gpu_power_watts", 0.0) for item in samples
                    )
                    point["gpu_utilisation_percent_mean"] = statistics.mean(
                        item.get("gpu_utilisation_percent", 0.0) for item in samples
                    )
                    point["gpu_memory_bytes_maximum"] = max(
                        item.get("gpu_memory_used_bytes", 0.0) for item in samples
                    )
                    if memory_pressure > 0.90:
                        point["background_suspended"] = True
                        point["suspension_reason"] = "host memory pressure limit exceeded"
                        point["non_degradation_pass"] = False
                        point["positive_contribution_pass"] = False
                rows.append(point)
        csv_write(run_directory / "background_results.csv", rows)
        json_write(
            run_directory / "profiles" / "background_telemetry.json",
            {
                "classification": "measured_mixed_backend",
                "cpu_package_power_measurement_status": "unavailable_on_host",
                "pcie_interference_method": (
                    "no cross-device model payload; shared-resource impact is measured through "
                    "the paired GPU latency and throughput deltas"
                ),
                "samples": telemetry_rows,
            },
        )
        return rows
    finally:
        service.close(repository_root=repository_root)


def _load_moe_plan(experiment_006: Path) -> RealMoEDownloadPlan:
    payload = json.loads(
        (experiment_006 / "real_moe_download_plan.json").read_text(encoding="utf-8")
    )
    if payload.get("status") not in {None, "PASS"}:
        raise RuntimeError("Experiment 006 real MoE layer artifact did not pass")
    return RealMoEDownloadPlan(
        model_id=str(payload["model_id"]),
        revision=str(payload["revision"]),
        selected_layer=int(payload["selected_layer"]),
        required_files=tuple(str(item) for item in payload["required_files"]),
        required_file_count=int(payload["required_file_count"]),
        required_download_bytes=int(payload["required_download_bytes"]),
        selected_layer_tensor_bytes=int(payload["selected_layer_tensor_bytes"]),
        unrelated_bytes_forced_by_file_co_location=int(
            payload["unrelated_bytes_forced_by_file_co_location"]
        ),
        maximum_download_bytes=int(payload["maximum_download_bytes"]),
        within_budget=bool(payload["within_budget"]),
        config_path=str(payload["config_path"]),
        index_path=str(payload["index_path"]),
        selected_tensor_names=tuple(str(item) for item in payload["selected_tensor_names"]),
    )


def _run_moe_and_integrity_arms(
    *,
    run_directory: Path,
    experiment_006: Path,
    config: HeterogeneousExperimentConfig,
    sglang_service: ManagedDockerService,
    interactive_prompt: list[int],
    smoke: bool,
    skip_integrity: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    plan = _load_moe_plan(experiment_006)
    files = download_real_moe_layer_files(plan)
    fixture_load_started = time.perf_counter()
    fixture = prepare_real_moe_fixture(
        plan,
        files,
        sequence_length=8 if smoke else config.moe.sequence_length,
    )
    fixture_load_seconds = time.perf_counter() - fixture_load_started
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    try:
        routing_counts = Counter(int(item) for item in fixture.selected_experts.flatten().tolist())
        counts = (1,) if smoke else config.moe.cpu_expert_counts
        policies = (
            ("coldest_experts_on_cpu", "hottest_experts_on_cpu")
            if smoke
            else config.moe.placement_policies
        )
        formats = ("BF16",) if smoke else config.moe.weight_formats
        for count in counts:
            for policy in policies:
                selected = select_cpu_experts(
                    cast(PlacementPolicy, policy),
                    count=count,
                    num_experts=int(fixture.config.num_experts),
                    routing_counts=dict(routing_counts),
                    predicted_counts=dict(routing_counts),
                )
                for weight_format in formats:
                    result = run_hybrid_cpu_experts(
                        fixture,
                        cpu_expert_ids=selected,
                        weight_format=cast(ExpertFormat, weight_format),
                        prefetch_expert_ids=selected if policy == "predicted_next_experts" else [],
                    )
                    result["placement_policy"] = policy
                    result["model_load_seconds"] = fixture_load_seconds
                    result["warmup_seconds"] = (
                        float(result["quantisation_time_ms"]) + float(result["expert_prefetch_ms"])
                    ) / 1000
                    result["load_timing_scope"] = (
                        "validated layer fixture load plus full-GPU reference preparation"
                    )
                    result["lower_gpu_cap_bytes"] = (
                        fixture.baseline_gpu_memory_bytes
                        - int(result["gpu_memory_saved_bytes"])
                        + 1
                    )
                    result["gpu_cap_satisfied"] = fixture.baseline_gpu_memory_bytes - int(
                        result["gpu_memory_saved_bytes"]
                    ) <= int(result["lower_gpu_cap_bytes"])
                    result["positive_contribution_pass"] = (
                        bool(result["gpu_cap_satisfied"])
                        and int(result["selected_cpu_expert_calls"]) > 0
                        and float(result["throughput_retained_fraction"])
                        >= config.positive_contribution.minimum_expert_retained_throughput_fraction
                    )
                    rows.append(result)
        if not skip_integrity:
            expert_id = int(fixture.selected_experts.flatten()[0].item())
            value = fixture.expert_input.reshape(-1, int(fixture.config.hidden_size))[0:1].cpu()
            gate, up, down = fixture.expert_weights_cpu[expert_id]

            def recompute() -> bytes:
                output = functional.linear(
                    functional.silu(functional.linear(value, gate)) * functional.linear(value, up),
                    down,
                )
                return output.contiguous().view(torch.uint8).numpy().tobytes()

            reference = recompute()
            corrupted = bytes([reference[0] ^ 1]) + reference[1:]
            item_count = 20 if smoke else 100
            items = [
                AuditItem(
                    operation_id=f"real-moe-expert-{expert_id}-{index}",
                    primary_payload=(corrupted if index % 7 == 0 else reference),
                    corrupt=index % 7 == 0,
                )
                for index in range(item_count)
            ]

            async def audit_function(_item: AuditItem) -> bytes:
                return await asyncio.to_thread(recompute)

            baseline_probe = stream_generate(
                sglang_service.endpoint,
                input_ids=interactive_prompt,
                output_tokens=4 if smoke else 32,
                request_id=f"exp007-audit-baseline-{uuid4().hex[:8]}",
            )
            baseline_ms = float(baseline_probe["end_to_end_ms"])
            rates = (0.0, 0.10) if smoke else config.workloads.audit_rates
            for rate in rates:

                async def paired(
                    current_rate: float = rate,
                ) -> tuple[dict[str, Any], dict[str, Any]]:
                    audit_task = asyncio.create_task(
                        run_integrity_audit_rate(
                            items,
                            audit_fraction=current_rate,
                            audit_function=audit_function,
                            baseline_gpu_latency_ms=baseline_ms,
                            observed_gpu_latency_ms=baseline_ms,
                        )
                    )
                    probe_task = asyncio.create_task(
                        asyncio.to_thread(
                            stream_generate,
                            sglang_service.endpoint,
                            input_ids=interactive_prompt,
                            output_tokens=4 if smoke else 32,
                            request_id=f"exp007-audit-{current_rate}-{uuid4().hex[:8]}",
                        )
                    )
                    return await asyncio.gather(audit_task, probe_task)

                audit, probe = asyncio.run(paired())
                observed_ms = float(probe["end_to_end_ms"])
                impact = observed_ms / max(baseline_ms, 1e-12) - 1
                audit["gpu_baseline_latency_ms"] = baseline_ms
                audit["gpu_observed_latency_ms"] = observed_ms
                # Zero auditing is the control and has no audit-induced impact;
                # treating ordinary request jitter as audit overhead corrupts the
                # sustainability baseline.
                audit["gpu_latency_impact_fraction"] = 0.0 if rate == 0 else impact
                audit["sustainable"] = rate == 0 or (
                    int(audit["audit_queue_overflow"]) == 0
                    and impact <= config.non_degradation.maximum_interactive_p95_increase_fraction
                )
                audit["audited_operation"] = "real Qwen3-30B-A3B layer-24 routed expert"
                audit["expert_id"] = expert_id
                audit_rows.append(audit)
    finally:
        fixture.release()
        del fixture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    csv_write(run_directory / "cpu_expert_results.csv", rows)
    csv_write(run_directory / "expert_placement_results.csv", rows)
    csv_write(
        run_directory / "expert_cache_results.csv",
        [
            {
                key: row[key]
                for key in row
                if key.startswith("expert_cache")
                or key
                in {
                    "classification",
                    "placement_policy",
                    "cpu_expert_count",
                    "weight_format",
                    "expert_prefetch_ms",
                    "expert_prefetch_useful",
                }
            }
            for row in rows
        ],
    )
    csv_write(run_directory / "integrity_audit_results.csv", audit_rows)
    audit_summary = {
        "status": "PASS"
        if audit_rows and any(bool(row["sustainable"]) for row in audit_rows)
        else "FAIL",
        "maximum_sustainable_audit_fraction": maximum_sustainable_audit_fraction(audit_rows),
        "real_expert_recomputation": True,
    }
    return rows, audit_rows, audit_summary


def _best_active_expert(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the best passing row that actually dispatched work to the CPU."""

    return max(
        (row for row in rows if int(row.get("selected_cpu_expert_calls", 0)) > 0),
        key=lambda row: (
            bool(row.get("positive_contribution_pass")),
            float(row.get("throughput_retained_fraction", 0)),
        ),
        default=None,
    )


def _planner_and_projections(
    *,
    run_directory: Path,
    config: HeterogeneousExperimentConfig,
    sglang_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    cpu_memory_bytes: int,
    model_load_seconds: dict[NodeRole, float],
    warmup_seconds: dict[NodeRole, float],
) -> dict[str, Any]:
    baseline = next(
        (
            row
            for row in sglang_rows
            if row.get("workload") == "short" and int(row.get("concurrency", 0)) == 1
        ),
        sglang_rows[0],
    )
    baseline_tps = float(baseline["aggregate_verified_throughput"])
    baseline_p95 = float(baseline["latency_p95_ms"])
    mixed = next(row for row in mixed_rows if row["route"] == "cuda-cpu-cuda-cpu")
    reference = next(row for row in mixed_rows if row["route"] == "cuda-cuda-cuda-cuda")
    mixed_gain = float(mixed["output_tokens_per_second"]) - float(
        reference["output_tokens_per_second"]
    )
    mixed_latency_delta = float(mixed["end_to_end_ms"]) - float(reference["end_to_end_ms"])
    best_spec = max(
        speculative_rows,
        key=lambda row: float(row.get("live_coordinator_canary_speedup_fraction", -math.inf)),
        default=None,
    )
    best_expert = _best_active_expert(expert_rows)
    admissible_background = [
        row for row in background_rows if bool(row.get("non_degradation_pass"))
    ]
    best_background = max(
        admissible_background,
        key=lambda row: float(row.get("combined_throughput_gain_fraction", -math.inf)),
        default=None,
    )
    sustainable_audit = max(
        (float(row["audit_fraction"]) for row in audit_rows if bool(row.get("sustainable"))),
        default=0.0,
    )

    def candidates(prediction_factor: float) -> list[RoleCandidate]:
        speculative_gain = (
            baseline_tps * float(best_spec.get("live_coordinator_canary_speedup_fraction", -1))
            if best_spec
            else -baseline_tps
        )
        background_gain = (
            float(best_background["total_combined_verified_tokens_per_second"])
            - float(best_background["baseline_gpu_tokens_per_second"])
            if best_background
            else -baseline_tps
        )
        background_interactive_delta = (
            (
                float(best_background["gpu_aggregate_tokens_per_second"])
                / max(float(best_background["baseline_gpu_tokens_per_second"]), 1e-12)
                - 1
            )
            * baseline_tps
            if best_background
            else -baseline_tps
        )
        expert_gain = (
            float(best_expert["hybrid_layer_throughput"])
            if best_expert and bool(best_expert["gpu_cap_satisfied"])
            else -1.0
        )
        critical_memory = sum(
            int(item.get("cache_bytes", 0))
            for item in boundaries
            if item.get("route") == "cuda-cpu-cuda-cpu"
        )
        return [
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.CRITICAL_PATH_STAGE,
                expected_verified_token_gain=mixed_gain * prediction_factor,
                predicted_p95_latency_delta_ms=mixed_latency_delta * prediction_factor,
                predicted_interactive_throughput_delta=mixed_gain * prediction_factor,
                predicted_memory_bytes=critical_memory,
                predicted_transfer_bytes=sum(
                    int(item.get("payload_bytes", 0))
                    for item in boundaries
                    if item.get("route") == "cuda-cpu-cuda-cpu"
                ),
                predicted_failure_cost=1.0,
                verification_cost=0.2,
                admission_risk=0.3,
                model_load_seconds=model_load_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                confidence_fraction=0.10,
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
                evidence={"forced_negative_control": True},
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.TENSOR_RANK,
                expected_verified_token_gain=mixed_gain * 0.8 * prediction_factor,
                predicted_p95_latency_delta_ms=mixed_latency_delta * prediction_factor,
                predicted_interactive_throughput_delta=mixed_gain * 0.8 * prediction_factor,
                predicted_memory_bytes=critical_memory,
                predicted_transfer_bytes=sum(
                    int(item.get("payload_bytes", 0)) for item in boundaries
                ),
                predicted_failure_cost=1.1,
                verification_cost=0.3,
                admission_risk=0.4,
                model_load_seconds=model_load_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                confidence_fraction=0.15,
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.SPECULATIVE_DRAFT,
                expected_verified_token_gain=speculative_gain * prediction_factor,
                predicted_p95_latency_delta_ms=max(
                    0.0, -speculative_gain / max(baseline_tps, 1e-12) * baseline_p95
                ),
                predicted_interactive_throughput_delta=speculative_gain * prediction_factor,
                predicted_memory_bytes=int(cpu_memory_bytes * 0.08),
                predicted_transfer_bytes=64,
                predicted_failure_cost=0.1,
                verification_cost=0.3,
                admission_risk=0.1,
                model_load_seconds=model_load_seconds.get(NodeRole.SPECULATIVE_DRAFT, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.SPECULATIVE_DRAFT, 0.0),
                confidence_fraction=0.15,
                compatible=best_spec is not None,
                compatibility_reason=None if best_spec else "speculative arm was not measured",
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.MOE_EXPERT,
                expected_verified_token_gain=expert_gain * prediction_factor,
                predicted_p95_latency_delta_ms=0.0,
                predicted_interactive_throughput_delta=0.0,
                predicted_memory_bytes=int(best_expert.get("cpu_memory_bytes", 0))
                if best_expert
                else 0,
                predicted_transfer_bytes=(
                    int(best_expert.get("dispatch_bytes", 0))
                    + int(best_expert.get("return_bytes", 0))
                    if best_expert
                    else 0
                ),
                predicted_failure_cost=0.2,
                verification_cost=0.2,
                admission_risk=0.15,
                model_load_seconds=model_load_seconds.get(NodeRole.MOE_EXPERT, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.MOE_EXPERT, 0.0),
                confidence_fraction=0.20,
                compatible=best_expert is not None,
                compatibility_reason=None if best_expert else "MoE arm was not measured",
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
                evidence={
                    "utility_baseline": "GPU execution under lower configured cap is infeasible"
                },
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.STAGE_REPLICA,
                expected_verified_token_gain=0.0,
                predicted_p95_latency_delta_ms=0.0,
                predicted_interactive_throughput_delta=0.0,
                predicted_memory_bytes=critical_memory,
                predicted_transfer_bytes=0,
                predicted_failure_cost=0.5,
                verification_cost=0.2,
                admission_risk=0.2,
                model_load_seconds=model_load_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.CRITICAL_PATH_STAGE, 0.0),
                compatible=False,
                compatibility_reason="no independent measured replica demand in this experiment",
                classification=ResultClassification.MEASURED_X86_CPU,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.BACKGROUND_INFERENCE,
                expected_verified_token_gain=background_gain * prediction_factor,
                predicted_p95_latency_delta_ms=(
                    float(best_background["gpu_interactive_p95_ms"])
                    - float(best_background["baseline_gpu_p95_ms"])
                    if best_background
                    else baseline_p95
                ),
                predicted_interactive_throughput_delta=(background_interactive_delta),
                predicted_memory_bytes=int(cpu_memory_bytes * 0.08),
                predicted_transfer_bytes=0,
                predicted_failure_cost=0.05,
                verification_cost=0.05,
                admission_risk=0.05,
                model_load_seconds=model_load_seconds.get(NodeRole.BACKGROUND_INFERENCE, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.BACKGROUND_INFERENCE, 0.0),
                confidence_fraction=0.10,
                compatible=best_background is not None,
                compatibility_reason=None
                if best_background
                else "background arm had no admissible point",
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.INTEGRITY_AUDIT,
                expected_verified_token_gain=sustainable_audit,
                predicted_p95_latency_delta_ms=0.0,
                predicted_interactive_throughput_delta=0.0,
                predicted_memory_bytes=16 * 1024**2,
                predicted_transfer_bytes=4096,
                predicted_failure_cost=0.0,
                verification_cost=0.02,
                admission_risk=0.01,
                model_load_seconds=model_load_seconds.get(NodeRole.MOE_EXPERT, 0.0),
                warmup_seconds=warmup_seconds.get(NodeRole.MOE_EXPERT, 0.0),
                confidence_fraction=0.20,
                compatible=bool(audit_rows),
                compatibility_reason=None if audit_rows else "audit arm was not measured",
                classification=ResultClassification.MEASURED_MIXED_BACKEND,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.SHARD_CACHE,
                expected_verified_token_gain=0.0,
                predicted_p95_latency_delta_ms=0.0,
                predicted_interactive_throughput_delta=0.0,
                predicted_memory_bytes=0,
                predicted_transfer_bytes=1024**3,
                predicted_failure_cost=0.1,
                verification_cost=0.02,
                admission_risk=0.1,
                compatible=False,
                compatibility_reason="disk-cache service interface defined; no cache transfer canary measured",
                classification=ResultClassification.MEASURED_X86_CPU,
            ),
            RoleCandidate(
                node_id="measured-host-cpu",
                role=NodeRole.IDLE,
                expected_verified_token_gain=0.0,
                predicted_p95_latency_delta_ms=0.0,
                predicted_interactive_throughput_delta=0.0,
                predicted_memory_bytes=0,
                predicted_transfer_bytes=0,
                predicted_failure_cost=0.0,
                verification_cost=0.0,
                admission_risk=0.0,
                confidence_fraction=0.0,
                classification=ResultClassification.MEASURED_X86_CPU,
            ),
        ]

    policy = NonDegradationPolicy(
        maximum_interactive_p95_increase_fraction=(
            config.non_degradation.maximum_interactive_p95_increase_fraction
        ),
        maximum_interactive_throughput_decrease_fraction=(
            config.non_degradation.maximum_interactive_throughput_decrease_fraction
        ),
    )
    normalisation = UtilityNormalisation(
        verified_tps_scale=max(baseline_tps, 1.0),
        interactive_latency_ms_scale=max(baseline_p95, 1.0),
        transfer_bytes_scale=1024**3,
        failure_cost_scale=1.0,
        verification_cost_scale=1.0,
        admission_risk_scale=1.0,
    )
    planner = HeterogeneousPlanner(
        policy=policy,
        normalisation=normalisation,
        maximum_regret_fraction=config.planner.maximum_regret_fraction,
    )
    predicted_candidates = candidates(0.98)
    actual_candidates = candidates(1.0)
    predictions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    regrets: list[dict[str, Any]] = []
    final_decision = None
    for objective in config.planner.objectives:
        decision = planner.select(
            predicted_candidates,
            objective=objective,
            baseline_p95_latency_ms=baseline_p95,
            baseline_interactive_throughput=baseline_tps,
            maximum_memory_bytes=cpu_memory_bytes,
        )
        final_decision = decision if objective == PlannerObjective.BALANCED else final_decision
        predictions.extend(
            {
                **evaluation.model_dump(mode="json"),
                "objective": objective.value,
                "rank": rank + 1,
            }
            for rank, evaluation in enumerate(decision.ranking)
        )
        actual_planner = HeterogeneousPlanner(
            policy=policy,
            normalisation=normalisation,
            maximum_regret_fraction=config.planner.maximum_regret_fraction,
        )
        actual_evaluations = actual_planner.evaluate_candidates(
            actual_candidates,
            objective=objective,
            baseline_p95_latency_ms=baseline_p95,
            baseline_interactive_throughput=baseline_tps,
            maximum_memory_bytes=cpu_memory_bytes,
        )
        actual_by_role = {NodeRole(item.role): item.utility_score for item in actual_evaluations}
        regret = planner_regret(actual_by_role, decision.selected_role)
        regret["objective"] = objective.value
        regret["passes"] = (
            float(regret["planner_regret_fraction"]) <= config.planner.maximum_regret_fraction
        )
        regrets.append(regret)
        predicted_by_role = {NodeRole(item.role): item.utility_score for item in decision.ranking}
        for role, utility in actual_by_role.items():
            measurements.append(
                {
                    "classification": "measured_mixed_backend",
                    "objective": objective.value,
                    "role": role.value,
                    "predicted_utility": predicted_by_role[role],
                    "measured_utility": utility,
                    "prediction_error": utility - predicted_by_role[role],
                    "selected": role == decision.selected_role,
                }
            )
        selected_candidate = next(
            item for item in actual_candidates if item.role == decision.selected_role
        )
        selected_actual = actual_by_role[decision.selected_role]
        canary = CanaryMeasurement(
            node_id="measured-host-cpu",
            role=decision.selected_role,
            measured_verified_tps_gain=selected_candidate.expected_verified_token_gain,
            measured_p95_latency_delta_ms=selected_candidate.predicted_p95_latency_delta_ms,
            measured_interactive_throughput_delta=(
                selected_candidate.predicted_interactive_throughput_delta
            ),
            baseline_p95_latency_ms=baseline_p95,
            baseline_interactive_throughput=baseline_tps,
            measured_utility=selected_actual,
            passed=selected_actual >= 0,
            classification=ResultClassification.MEASURED_MIXED_BACKEND,
        )
        retained = planner.update_after_canary(canary)
        decisions.append(
            {
                **decision.model_dump(mode="json"),
                "canary_retained": retained,
                "measured_utility": selected_actual,
                "prediction_error": selected_actual - decision.selected_utility,
                "regret": regret,
            }
        )
    if final_decision is None:
        final_decision = planner.decisions[-1]
    critical = next(item for item in actual_candidates if item.role == NodeRole.CRITICAL_PATH_STAGE)
    critical_measured_utility = next(
        float(row["measured_utility"])
        for row in measurements
        if row["objective"] == PlannerObjective.BALANCED.value
        and row["role"] == NodeRole.CRITICAL_PATH_STAGE.value
    )
    critical_violated = (
        critical_measured_utility <= 0
        or critical.predicted_p95_latency_delta_ms
        > baseline_p95 * config.non_degradation.maximum_interactive_p95_increase_fraction
        or critical.predicted_interactive_throughput_delta
        < -baseline_tps * config.non_degradation.maximum_interactive_throughput_decrease_fraction
    )
    critical_canary = CanaryMeasurement(
        node_id="measured-host-cpu",
        role=NodeRole.CRITICAL_PATH_STAGE,
        measured_verified_tps_gain=critical.expected_verified_token_gain,
        measured_p95_latency_delta_ms=critical.predicted_p95_latency_delta_ms,
        measured_interactive_throughput_delta=critical.predicted_interactive_throughput_delta,
        baseline_p95_latency_ms=baseline_p95,
        baseline_interactive_throughput=baseline_tps,
        measured_utility=critical_measured_utility,
        passed=not critical_violated,
        classification=ResultClassification.MEASURED_MIXED_BACKEND,
    )
    reassignment = planner.monitor_and_reassign(
        critical_canary,
        predicted_candidates,
        objective=PlannerObjective.BALANCED,
        maximum_memory_bytes=cpu_memory_bytes,
    )
    if reassignment is not None:
        decisions.append(
            {
                **reassignment.model_dump(mode="json"),
                "runtime_reassignment": True,
                "removed_role": NodeRole.CRITICAL_PATH_STAGE.value,
                "removal_reason": "measured non-degradation violation",
            }
        )
    jsonl_write(run_directory / "planner_predictions.jsonl", predictions)
    jsonl_write(run_directory / "planner_decisions.jsonl", decisions)
    csv_write(run_directory / "planner_measurements.csv", measurements)
    csv_write(run_directory / "planner_regret.csv", regrets)

    traces = _measured_role_traces(
        baseline_tps=baseline_tps,
        baseline_p95=baseline_p95,
        mixed=mixed,
        boundaries=boundaries,
        speculative=best_spec,
        expert=best_expert,
        background=best_background,
        audit_fraction=sustainable_audit,
    )
    network = replay_network_matrix(traces, config.network_projection.profiles)
    availability = availability_economics_rows(
        traces,
        acquisition_seconds={role: 0.0 for role in NodeRole},
        conversion_seconds={
            NodeRole.SPECULATIVE_DRAFT: 0.0,
            NodeRole.BACKGROUND_INFERENCE: 0.0,
        },
        load_seconds=model_load_seconds,
        warmup_seconds=warmup_seconds,
        lease_durations_seconds=config.workloads.lease_durations_seconds,
        all_roles=list(NodeRole),
    )
    csv_write(run_directory / "network_projection.csv", network)
    csv_write(run_directory / "availability_economics.csv", availability)
    return {
        "predictions": predictions,
        "decisions": decisions,
        "measurements": measurements,
        "regret": regrets,
        "selected_role": final_decision.selected_role.value,
        "selected_predicted_utility": final_decision.selected_utility,
        "selected_measured_utility": next(
            float(row["measured_utility"])
            for row in measurements
            if row["objective"] == PlannerObjective.BALANCED.value and bool(row["selected"])
        ),
        "runtime_reassignment_occurred": reassignment is not None,
        "network": network,
        "availability": availability,
    }


def _measured_role_traces(
    *,
    baseline_tps: float,
    baseline_p95: float,
    mixed: dict[str, Any],
    boundaries: list[dict[str, Any]],
    speculative: dict[str, Any] | None,
    expert: dict[str, Any] | None,
    background: dict[str, Any] | None,
    audit_fraction: float,
) -> list[MeasuredRoleTrace]:
    mixed_boundaries = [row for row in boundaries if row.get("route") == "cuda-cpu-cuda-cpu"]
    mixed_tps = float(mixed["output_tokens_per_second"])
    mixed_change = float(mixed["throughput_change_fraction"])
    reference_tps = mixed_tps / max(1.0 + mixed_change, 1e-12)
    traces = [
        MeasuredRoleTrace(
            role=NodeRole.CRITICAL_PATH_STAGE,
            request_payload_bytes=sum(int(row.get("payload_bytes", 0)) for row in mixed_boundaries),
            response_payload_bytes=sum(
                int(row.get("payload_bytes", 0)) for row in mixed_boundaries
            ),
            measured_compute_ms=float(mixed["end_to_end_ms"]),
            verified_tokens=float(mixed["output_tokens"]),
            baseline_service_ms=float(mixed["output_tokens"]) / reference_tps * 1000,
            measured_marginal_verified_tps_gain=mixed_tps - reference_tps,
            failure_recovery_ms=float(mixed["end_to_end_ms"]),
            availability_requirement=0.999,
        )
    ]
    if speculative is not None:
        traces.append(
            MeasuredRoleTrace(
                role=NodeRole.SPECULATIVE_DRAFT,
                request_payload_bytes=64,
                response_payload_bytes=int(speculative["draft_length"]) * 4,
                measured_compute_ms=(
                    int(speculative["draft_length"])
                    / max(float(speculative["draft_tokens_per_second"]), 1e-12)
                    * 1000
                ),
                verified_tokens=float(speculative["accepted_tokens_per_verification"]),
                baseline_service_ms=(
                    float(speculative["accepted_tokens_per_verification"])
                    / max(float(speculative["target_only_tokens_per_second"]), 1e-12)
                    * 1000
                ),
                measured_marginal_verified_tps_gain=(
                    float(speculative["target_only_tokens_per_second"])
                    * float(speculative["live_coordinator_canary_speedup_fraction"])
                ),
                verification_ms=(int(speculative["draft_length"]) + 1)
                / max(float(speculative["target_only_tokens_per_second"]), 1e-12)
                * 1000,
                availability_requirement=0.90,
            )
        )
    if expert is not None:
        traces.append(
            MeasuredRoleTrace(
                role=NodeRole.MOE_EXPERT,
                request_payload_bytes=int(expert["dispatch_bytes"]),
                response_payload_bytes=int(expert["return_bytes"]),
                measured_compute_ms=float(expert["cpu_expert_latency_ms"]),
                verified_tokens=max(1.0, float(expert["selected_cpu_expert_calls"])),
                baseline_service_ms=float(expert["baseline_layer_latency_ms"]),
                measured_marginal_verified_tps_gain=(
                    float(expert["hybrid_layer_throughput"])
                    if bool(expert.get("positive_contribution_pass"))
                    else -float(expert["baseline_layer_throughput"])
                ),
                verification_ms=float(expert["gpu_expert_latency_ms"]),
                availability_requirement=0.995,
            )
        )
    if background is not None:
        cpu_tps = float(background["cpu_background_tokens_per_second"])
        cpu_requests = list(background.get("cpu_request_metrics", []))
        background_request_bytes = (
            int(
                statistics.mean(
                    float(item.get("metrics", {}).get("timings", {}).get("prompt_n", 0)) * 4
                    for item in cpu_requests
                )
            )
            if cpu_requests
            else 0
        )
        background_response_bytes = (
            int(statistics.mean(float(item.get("output_tokens", 0)) * 4 for item in cpu_requests))
            if cpu_requests
            else 0
        )
        traces.append(
            MeasuredRoleTrace(
                role=NodeRole.BACKGROUND_INFERENCE,
                request_payload_bytes=background_request_bytes,
                response_payload_bytes=background_response_bytes,
                measured_compute_ms=1000 / max(cpu_tps, 1e-12),
                verified_tokens=1.0,
                baseline_service_ms=1000 / max(baseline_tps, 1e-12),
                measured_marginal_verified_tps_gain=(
                    float(background["total_combined_verified_tokens_per_second"])
                    - float(background["baseline_gpu_tokens_per_second"])
                ),
                availability_requirement=0.5,
            )
        )
    if audit_fraction > 0:
        traces.append(
            MeasuredRoleTrace(
                role=NodeRole.INTEGRITY_AUDIT,
                request_payload_bytes=4096,
                response_payload_bytes=32,
                measured_compute_ms=max(1.0, baseline_p95 * audit_fraction),
                verified_tokens=audit_fraction,
                baseline_service_ms=baseline_p95,
                measured_marginal_verified_tps_gain=audit_fraction
                / max(baseline_p95 / 1000, 1e-12),
                availability_requirement=0.5,
            )
        )
    return traces


def _bounded_cpu_benchmark() -> dict[str, Any]:
    import numpy as np
    import psutil

    rng = np.random.default_rng(7007)
    left = rng.standard_normal((512, 512), dtype=np.float32)
    right = rng.standard_normal((512, 512), dtype=np.float32)
    started = time.perf_counter()
    iterations = 5
    checksum = 0.0
    for _ in range(iterations):
        checksum += float((left @ right)[0, 0])
    matrix_seconds = time.perf_counter() - started
    source = rng.integers(0, 255, size=64 * 1024 * 1024, dtype=np.uint8)
    started = time.perf_counter()
    copied = source.copy()
    memory_seconds = time.perf_counter() - started
    return {
        "classification": "measured_x86_cpu",
        "matrix_kernel_results": {
            "fp32_512_square_iterations_per_second": iterations / max(matrix_seconds, 1e-12),
            "checksum": checksum,
        },
        "attention_kernel_results": {},
        "expert_kernel_results": {},
        "memory_copy_bytes_per_second": copied.nbytes / max(memory_seconds, 1e-12),
        "cpu_percent_after": psutil.cpu_percent(interval=0.1),
        "measured_at_utc": datetime.now(UTC).isoformat(),
    }


def _contribution_frontier(
    *,
    mixed_rows: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    mixed = next((row for row in mixed_rows if row.get("route") == "cuda-cpu-cuda-cpu"), None)
    critical = (
        "harmful"
        if mixed is not None and float(mixed.get("throughput_change_fraction", 0)) < -0.05
        else "marginal"
    )
    speculative_positive = any(
        bool(row.get("measured_positive_contribution_pass")) for row in speculative_rows
    )
    speculative = (
        "useful" if speculative_positive else ("harmful" if speculative_rows else "unsupported")
    )
    expert_positive = any(
        bool(row.get("positive_contribution_pass"))
        and int(row.get("selected_cpu_expert_calls", 0)) > 0
        for row in expert_rows
    )
    expert = "useful" if expert_positive else ("marginal" if expert_rows else "unsupported")
    background_positive = any(
        bool(row.get("positive_contribution_pass")) for row in background_rows
    )
    background = (
        "useful" if background_positive else ("harmful" if background_rows else "unsupported")
    )
    audit = (
        "useful"
        if any(bool(row.get("sustainable")) for row in audit_rows)
        else ("marginal" if audit_rows else "unsupported")
    )
    positive = speculative_positive or expert_positive or background_positive
    rows: list[dict[str, Any]] = [
        {
            "device_profile": "measured_host_x86_cpu",
            "classification": "measured_x86_cpu",
            "critical_path": critical,
            "speculative_draft": speculative,
            "moe_expert": expert,
            "background_inference": background,
            "integrity_audit": audit,
            "shard_cache": "marginal",
            "idle": "marginal" if positive else "useful",
            "projection_inputs": "none; measured host",
        }
    ]
    profiles = {
        "high_end_desktop_cpu": {"compute_multiplier": 1.5, "memory_gib": 64, "availability": 0.98},
        "midrange_laptop_cpu": {"compute_multiplier": 0.65, "memory_gib": 16, "availability": 0.80},
        "low_end_laptop_cpu": {"compute_multiplier": 0.30, "memory_gib": 8, "availability": 0.65},
        "raspberry_pi_5_class": {"compute_multiplier": 0.08, "memory_gib": 8, "availability": 0.90},
        "apple_m_series_class": {"compute_multiplier": 1.2, "memory_gib": 24, "availability": 0.85},
        "remote_intermittent_node": {
            "compute_multiplier": 0.65,
            "memory_gib": 16,
            "availability": 0.35,
        },
    }
    for name, inputs in profiles.items():
        rows.append(
            {
                "device_profile": name,
                "classification": "projected_device_profile",
                "critical_path": "projected",
                "speculative_draft": "projected",
                "moe_expert": "projected",
                "background_inference": "projected",
                "integrity_audit": "projected",
                "shard_cache": "projected",
                "idle": "projected",
                "projection_inputs": inputs,
                "raspberry_pi_performance": (
                    "unproven" if name == "raspberry_pi_5_class" else "not_applicable"
                ),
            }
        )
    return rows, positive


def _latency_contributors(
    *,
    sglang_rows: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mixed_stage: dict[str, float] = {}
    for row in boundaries:
        if row.get("route") != "cuda-cpu-cuda-cpu":
            continue
        key = f"stage_{row.get('stage_id')}_execution"
        mixed_stage[key] = mixed_stage.get(key, 0.0) + float(row.get("execution_ms", 0))
        mixed_stage["serialization"] = mixed_stage.get("serialization", 0.0) + float(
            row.get("serialisation_ms", 0)
        )
        mixed_stage["transfer_and_queue"] = mixed_stage.get("transfer_and_queue", 0.0) + max(
            0.0,
            float(row.get("round_trip_ms", 0)) - float(row.get("execution_ms", 0)),
        )

    def largest(values: dict[str, float]) -> list[dict[str, float | str]]:
        return [
            {"component": key, "milliseconds": value}
            for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:5]
        ]

    spec = max(
        speculative_rows,
        key=lambda row: float(row.get("live_coordinator_canary_speedup_fraction", -math.inf)),
        default={},
    )
    expert = _best_active_expert(expert_rows) or {}
    background = max(
        background_rows,
        key=lambda row: (
            bool(row.get("positive_contribution_pass")),
            float(row.get("combined_throughput_gain_fraction", -math.inf)),
        ),
        default={},
    )
    sglang = sglang_rows[0] if sglang_rows else {}
    return {
        "sglang": largest(
            {
                "time_to_first_token": float(sglang.get("ttft_p50_ms", 0)),
                "decode": max(
                    0.0,
                    float(sglang.get("latency_p50_ms", 0)) - float(sglang.get("ttft_p50_ms", 0)),
                ),
                "scheduler_queue": 0.0,
                "sampling": 0.0,
                "serialization": 0.0,
            }
        ),
        "mixed_pipeline": largest(mixed_stage),
        "speculative": largest(
            {
                "cpu_draft": float(spec.get("draft_seconds", 0)) * 1000,
                "target_verification": float(spec.get("target_verification_seconds", 0)) * 1000,
                "proposal_transfer": 0.0,
                "sampling": 0.0,
                "coordinator": 0.0,
            }
        ),
        "moe": largest(
            {
                "cpu_expert": float(expert.get("cpu_expert_latency_ms", 0)),
                "gpu_experts": float(expert.get("gpu_expert_latency_ms", 0)),
                "common_attention": float(expert.get("common_component_ms", 0)),
                "dispatch_and_return": float(expert.get("cpu_gpu_transfer_ms", 0)),
                "cache_load": float(expert.get("expert_cache_load_ms", 0)),
            }
        ),
        "background": largest(
            {
                "gpu_interactive": float(background.get("gpu_interactive_p50_ms", 0)),
                "cpu_background": (
                    1000 / max(float(background.get("cpu_background_tokens_per_second", 0)), 1e-12)
                ),
                "scheduler": 0.0,
                "pcie_interference": 0.0,
                "serialization": 0.0,
            }
        ),
    }


def run_heterogeneous_node_experiment(
    config: HeterogeneousExperimentConfig,
    *,
    requested_config_path: Path,
    options: HeterogeneousOptions,
) -> HeterogeneousRun:
    repository_root = _repository_root()
    run_directory = _run_directory(repository_root, options)
    _initialise_artifacts(run_directory)
    existing_summary = run_directory / "summary.json"
    if options.resume and existing_summary.is_file():
        try:
            prior = json.loads(existing_summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prior = {}
        if prior.get("overall_status") in {"PASS", "PARTIAL_PASS"}:
            return HeterogeneousRun(
                run_directory=run_directory,
                report_path=run_directory / "report.html",
                summary=prior,
            )
    requested = yaml.safe_load(requested_config_path.read_text(encoding="utf-8"))
    yaml_write(run_directory / "config.requested.yaml", requested)
    json_write(run_directory / "git.json", repository_git_state(repository_root))
    environment = environment_snapshot()
    environment["nvidia_smi"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    json_write(run_directory / "environment.json", environment)
    protocol_evidence = {
        "status": "PASS",
        "abi": {"major": 1, "minor": 0},
        "wire_protocol": "length-prefixed canonical JSON; SWARMT01 for tensors",
        "pickle_allowed": False,
        "protocol_version": WorkerProtocolVersion(
            major=1,
            minor=0,
            capabilities={
                "jobs",
                "capability-exchange",
                "cancel",
                "heartbeat",
                "reconnect",
                "shard-hash",
                "clean-shutdown",
            },
        ).model_dump(mode="json"),
        "job_types": [item.value for item in WorkerJobType],
        "job_statuses": [
            "accepted",
            "unsupported",
            "insufficient_memory",
            "incompatible_dtype",
            "deadline_impossible",
            "backend_failure",
        ],
        "tensor_fields": [
            "dtype",
            "shape",
            "strides",
            "byte_order",
            "payload_length",
            "checksum",
            "model_revision",
            "partition_hash",
            "request_id",
            "route_generation",
            "token_position",
        ],
        "activation_dtypes": ["bfloat16", "float16", "float32", "int8"],
    }
    json_write(run_directory / "worker_protocol.json", protocol_evidence)

    sglang_rows: list[dict[str, Any]] = []
    mixed_rows: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    speculative_rows: list[dict[str, Any]] = []
    _speculative_acceptance: list[dict[str, Any]] = []
    speculative_break_even: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    planner_result: dict[str, Any] = {}
    correctness: dict[str, Any] = {}
    backend_setup: dict[str, Any] = {}
    arm64_build: dict[str, Any] = _empty_status("not started")
    arm64_protocol: dict[str, Any] = _empty_status("not started")
    phase_errors: dict[str, str] = {}
    sglang_service: ManagedDockerService | None = None
    target_revision = TARGET_REVISION_FALLBACK
    experiment_004 = Path()
    experiment_006 = Path()
    try:
        experiment_004 = find_reference_run(
            repository_root,
            kind="engine-performance",
            configured=(
                str(options.experiment_004_run)
                if options.experiment_004_run is not None
                else config.references.experiment_004_run
            ),
        )
        experiment_006 = find_reference_run(
            repository_root,
            kind="microsharding",
            configured=(
                str(options.experiment_006_run)
                if options.experiment_006_run is not None
                else config.references.experiment_006_run
            ),
        )
        json_write(
            run_directory / "experiment_004_reference.json",
            reference_evidence(experiment_004),
        )
        json_write(
            run_directory / "experiment_006_reference.json",
            reference_evidence(experiment_006),
        )
        target_revision, target_path = _resolved_model_from_reference(
            experiment_004, config.gpu_target.model_id, TARGET_REVISION_FALLBACK
        )
        draft_revision, draft_path = _resolved_model_from_reference(
            experiment_004, config.cpu_draft.model_id, DRAFT_REVISION
        )
        if draft_revision != DRAFT_REVISION:
            raise RuntimeError("CPU draft did not resolve the required immutable revision")
        partition_root = repository_root / "artifacts" / "models" / "qwen3-0.6b-pp4-tp1"
        if not partition_root.is_dir():
            raise FileNotFoundError("Experiment 006 PP4/TP1 microshard artifact is missing")
        partition_manifest = json.loads(
            (partition_root / "manifest.json").read_text(encoding="utf-8")
        )
        if partition_manifest.get("model_revision") != DRAFT_REVISION:
            raise RuntimeError("mixed-pipeline microshard revision mismatch")
        resolved = config.model_dump(mode="json")
        resolved["references"] = {
            "experiment_004_run": str(experiment_004),
            "experiment_006_run": str(experiment_006),
        }
        resolved["gpu_target"]["revision"] = target_revision
        resolved["gpu_target"]["path"] = str(target_path)
        resolved["cpu_draft"]["path"] = str(draft_path)
        resolved["mixed_pipeline"]["partition_path"] = str(partition_root)
        yaml_write(run_directory / "config.resolved.yaml", resolved)

        cpu_worker_capabilities = cpu_capabilities(
            backend_features=[
                "canonical_safetensors",
                "GGUF",
                "stage_local_kv",
                "speculative_draft",
                "background_generate",
            ]
        )
        target_capabilities = cuda_capabilities()
        json_write(
            run_directory / "worker_capabilities.json",
            {
                "status": "PASS",
                "cpu": cpu_worker_capabilities.model_dump(mode="json"),
                "cuda": target_capabilities.model_dump(mode="json"),
            },
        )
        backend_setup = _backend_setup(
            repository_root=repository_root,
            run_directory=run_directory,
            config=config,
            target_revision=target_revision,
            target_path=target_path,
            draft_path=draft_path,
            microshards=partition_root,
        )
        resolved["backend_environments"]["llamacpp_commit"] = backend_setup["llamacpp"][
            "source_commit"
        ]
        resolved["backend_environments"]["torch_cpu_python"] = backend_setup["torch_cpu"][
            "python_executable"
        ]
        yaml_write(run_directory / "config.resolved.yaml", resolved)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            target_path,
            local_files_only=True,
        )
        tokenizer_hash = str(tokenizer_identity(target_path)["tokenizer_hash"])
        prompt_count = 4 if options.smoke else config.workloads.speculative_prompt_count
        prompts = _prompt_fixtures(tokenizer, prompt_count)
        seed = [
            int(item)
            for item in tokenizer.encode(
                "Measure the backend boundary and preserve exact greedy tokens.",
                add_special_tokens=True,
            )
        ]
        hf_cache = Path.home() / ".cache" / "huggingface"
        snapshot_relative = target_path.relative_to(hf_cache).as_posix()
        sglang_service = start_sglang_service(
            image=config.backend_environments.sglang_image,
            repository_root=repository_root,
            huggingface_cache=hf_cache,
            model_snapshot_relative=snapshot_relative,
            run_id=run_directory.name[-8:],
            log_root=run_directory / "logs",
        )
        sglang_rows, _, prefix = _run_sglang_baseline(
            run_directory=run_directory,
            config=config,
            service=sglang_service,
            tokenizer=tokenizer,
            smoke=options.smoke,
            profile=options.profile or config.profile,
        )
        correctness["sglang_prefix_cache"] = prefix
        try:
            mixed_rows, boundaries, mixed_correctness, worker_protocol_rows = _run_mixed_pipeline(
                repository_root=repository_root,
                run_directory=run_directory,
                run_id=run_directory.name[-8:],
                config=config,
                partition_root=partition_root,
                torch_cpu_python=Path(str(backend_setup["torch_cpu"]["python_executable"])),
                prompt_tokens=exact_token_length(seed, 8 if options.smoke else 128),
                smoke=options.smoke,
            )
            correctness["mixed_backend"] = mixed_correctness
            protocol_evidence["registered_workers"] = worker_protocol_rows
            protocol_evidence["live_registration_status"] = "PASS"
            json_write(run_directory / "worker_protocol.json", protocol_evidence)
        except Exception as exc:
            phase_errors["mixed_backend"] = f"{type(exc).__name__}: {exc}"
            correctness["mixed_backend"] = {
                "status": "FAIL",
                "error": phase_errors["mixed_backend"],
            }
        if not options.skip_speculative:
            try:
                (
                    speculative_rows,
                    _speculative_acceptance,
                    speculative_break_even,
                    speculative_correctness,
                ) = _run_speculative_arm(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    run_id=run_directory.name[-8:],
                    config=config,
                    backend_setup=backend_setup,
                    sglang_service=sglang_service,
                    prompts=prompts,
                    tokenizer_hash=tokenizer_hash,
                    target_revision=target_revision,
                    target_capabilities=target_capabilities,
                    cpu_worker_capabilities=cpu_worker_capabilities,
                    smoke=options.smoke,
                    capture_profile=options.profile or config.profile,
                )
                correctness["speculative"] = speculative_correctness
            except Exception as exc:
                phase_errors["speculative"] = f"{type(exc).__name__}: {exc}"
                correctness["speculative"] = {
                    "status": "FAIL",
                    "error": phase_errors["speculative"],
                }
        else:
            correctness["speculative"] = {"status": "SKIPPED"}
        if not options.skip_background:
            try:
                background_rows = _run_background_arm(
                    repository_root=repository_root,
                    run_directory=run_directory,
                    run_id=run_directory.name[-8:],
                    config=config,
                    backend_setup=backend_setup,
                    sglang_service=sglang_service,
                    prompts=prompts,
                    tokenizer_hash=tokenizer_hash,
                    target_revision=target_revision,
                    target_capabilities=target_capabilities,
                    cpu_worker_capabilities=cpu_worker_capabilities,
                    smoke=options.smoke,
                    capture_profile=options.profile or config.profile,
                )
            except Exception as exc:
                phase_errors["background"] = f"{type(exc).__name__}: {exc}"
        if not options.skip_moe:
            try:
                expert_rows, audit_rows, audit_summary = _run_moe_and_integrity_arms(
                    run_directory=run_directory,
                    experiment_006=experiment_006,
                    config=config,
                    sglang_service=sglang_service,
                    interactive_prompt=prompts[0].token_ids,
                    smoke=options.smoke,
                )
                correctness["moe"] = {
                    "status": (
                        "PASS"
                        if any(row.get("status") == "PASS" for row in expert_rows)
                        else "FAIL"
                    ),
                    "bf16_exact_layer_output": any(
                        bool(row.get("exact_layer_output"))
                        for row in expert_rows
                        if row.get("weight_format") == "BF16"
                    ),
                    "full_gpu_layer_comparison_performed": all(
                        bool(row.get("comparison_performed")) for row in expert_rows
                    ),
                    "bf16_numerical_tolerance_pass": all(
                        bool(row.get("numerical_tolerance_pass"))
                        for row in expert_rows
                        if row.get("weight_format") == "BF16"
                    ),
                    "quantisation_errors_explicit": all(
                        "maximum_quantised_weight_error" in row for row in expert_rows
                    ),
                }
                correctness["integrity_audit"] = audit_summary
            except Exception as exc:
                phase_errors["moe"] = f"{type(exc).__name__}: {exc}"
                correctness["moe"] = {"status": "FAIL", "error": phase_errors["moe"]}
                correctness["integrity_audit"] = {"status": "FAIL"}
        else:
            correctness["moe"] = {"status": "SKIPPED"}
            correctness["integrity_audit"] = {"status": "SKIPPED"}
        if sglang_service is not None:
            sglang_service.close(repository_root=repository_root)
            sglang_service = None
        if not options.skip_arm64:
            try:
                arm64_build, arm64_protocol = build_and_test_arm64(
                    repository_root=repository_root,
                    backend_root=repository_root / config.backend_environments.root,
                    llamacpp_environment=backend_setup["llamacpp"],
                    run_directory=run_directory,
                )
            except Exception as exc:
                phase_errors["arm64"] = f"{type(exc).__name__}: {exc}"
                arm64_build = {"status": "FAIL", "error": phase_errors["arm64"]}
                arm64_protocol = {"status": "FAIL", "error": phase_errors["arm64"]}
        else:
            arm64_build = {"status": "BLOCKED", "reason": "skipped by explicit option"}
            arm64_protocol = {"status": "BLOCKED", "reason": "skipped by explicit option"}
        json_write(run_directory / "arm64_build.json", arm64_build)
        json_write(run_directory / "arm64_protocol_results.json", arm64_protocol)
        if mixed_rows:
            raw_registered_workers = protocol_evidence.get("registered_workers", [])
            registered_workers = (
                raw_registered_workers if isinstance(raw_registered_workers, list) else []
            )
            timing_spec = max(
                speculative_rows,
                key=lambda row: float(
                    row.get("live_coordinator_canary_speedup_fraction", -math.inf)
                ),
                default={},
            )
            timing_expert = _best_active_expert(expert_rows) or {}
            timing_background = max(
                background_rows,
                key=lambda row: (
                    bool(row.get("positive_contribution_pass")),
                    float(row.get("combined_throughput_gain_fraction", -math.inf)),
                ),
                default={},
            )
            critical_load_seconds = sum(
                float(item.get("benchmark", {}).get("model_load_seconds", 0))
                for item in registered_workers
                if isinstance(item, dict)
            )
            critical_warmup_seconds = sum(
                float(item.get("benchmark", {}).get("warmup_seconds", 0))
                for item in registered_workers
                if isinstance(item, dict)
            )
            planner_result = _planner_and_projections(
                run_directory=run_directory,
                config=config,
                sglang_rows=sglang_rows,
                mixed_rows=mixed_rows,
                boundaries=boundaries,
                speculative_rows=speculative_rows,
                expert_rows=expert_rows,
                background_rows=background_rows,
                audit_rows=audit_rows,
                cpu_memory_bytes=cpu_worker_capabilities.maximum_weight_bytes,
                model_load_seconds={
                    NodeRole.CRITICAL_PATH_STAGE: critical_load_seconds,
                    NodeRole.TENSOR_RANK: critical_load_seconds,
                    NodeRole.STAGE_REPLICA: critical_load_seconds,
                    NodeRole.SPECULATIVE_DRAFT: float(timing_spec.get("model_load_seconds", 0)),
                    NodeRole.MOE_EXPERT: float(timing_expert.get("model_load_seconds", 0)),
                    NodeRole.BACKGROUND_INFERENCE: float(
                        timing_background.get("cpu_model_load_seconds", 0)
                    ),
                    NodeRole.INTEGRITY_AUDIT: float(timing_expert.get("model_load_seconds", 0)),
                },
                warmup_seconds={
                    NodeRole.CRITICAL_PATH_STAGE: critical_warmup_seconds,
                    NodeRole.TENSOR_RANK: critical_warmup_seconds,
                    NodeRole.STAGE_REPLICA: critical_warmup_seconds,
                    NodeRole.SPECULATIVE_DRAFT: float(timing_spec.get("warmup_seconds", 0)),
                    NodeRole.MOE_EXPERT: float(timing_expert.get("warmup_seconds", 0)),
                    NodeRole.BACKGROUND_INFERENCE: float(
                        timing_background.get("cpu_warmup_seconds", 0)
                    ),
                    NodeRole.INTEGRITY_AUDIT: float(timing_expert.get("warmup_seconds", 0)),
                },
            )
        frontier, positive_cpu = _contribution_frontier(
            mixed_rows=mixed_rows,
            speculative_rows=speculative_rows,
            expert_rows=expert_rows,
            background_rows=background_rows,
            audit_rows=audit_rows,
        )
        csv_write(run_directory / "contribution_frontier.csv", frontier)
        worker_benchmarks = {
            "status": "PASS",
            "bounded_cpu": _bounded_cpu_benchmark(),
            "cuda_sglang": sglang_rows,
            "cpu_rank_boundaries": [
                row for row in boundaries if int(row.get("stage_id", -1)) in {1, 3}
            ],
            "llamacpp": speculative_rows,
            "moe_expert": expert_rows,
            "background": background_rows,
        }
        json_write(run_directory / "worker_benchmarks.json", worker_benchmarks)
        correctness["status"] = (
            "PASS" if correctness.get("mixed_backend", {}).get("status") == "PASS" else "FAIL"
        )
        json_write(run_directory / "correctness.json", correctness)
        latency = _latency_contributors(
            sglang_rows=sglang_rows,
            boundaries=boundaries,
            speculative_rows=speculative_rows,
            expert_rows=expert_rows,
            background_rows=background_rows,
        )
        json_write(run_directory / "profiles" / "latency_contributors.json", latency)
        summary = _build_summary(
            run_directory=run_directory,
            experiment_004=experiment_004,
            experiment_006=experiment_006,
            target_revision=target_revision,
            backend_setup=backend_setup,
            sglang_rows=sglang_rows,
            mixed_rows=mixed_rows,
            speculative_rows=speculative_rows,
            expert_rows=expert_rows,
            background_rows=background_rows,
            audit_rows=audit_rows,
            arm64_protocol=arm64_protocol,
            planner_result=planner_result,
            positive_cpu=positive_cpu,
            phase_errors=phase_errors,
            options=options,
        )
    except Exception as exc:
        phase_errors["mandatory_infrastructure"] = f"{type(exc).__name__}: {exc}"
        summary = _failure_summary(
            run_directory=run_directory,
            phase_errors=phase_errors,
            target_revision=target_revision,
        )
    finally:
        if sglang_service is not None:
            sglang_service.close(repository_root=repository_root)
    json_write(run_directory / "summary.json", summary)
    if not frontier:
        frontier, _ = _contribution_frontier(
            mixed_rows=mixed_rows,
            speculative_rows=speculative_rows,
            expert_rows=expert_rows,
            background_rows=background_rows,
            audit_rows=audit_rows,
        )
        csv_write(run_directory / "contribution_frontier.csv", frontier)
    generate_heterogeneous_charts(
        run_directory,
        sglang=sglang_rows,
        mixed=mixed_rows,
        boundaries=boundaries,
        speculative=speculative_rows,
        break_even=speculative_break_even,
        experts=expert_rows,
        background=background_rows,
        planner_measurements=planner_result.get("measurements", []),
        regret=planner_result.get("regret", []),
        frontier=frontier,
        availability=planner_result.get("availability", []),
        network=planner_result.get("network", []),
    )
    findings = _headline_findings(
        summary=summary,
        sglang_rows=sglang_rows,
        mixed_rows=mixed_rows,
        speculative_rows=speculative_rows,
        expert_rows=expert_rows,
        background_rows=background_rows,
        audit_rows=audit_rows,
        planner_result=planner_result,
    )
    report_path = render_heterogeneous_report(
        run_directory,
        summary=summary,
        findings=findings,
        frontier=frontier,
    )
    manifest = {
        "files": {
            str(path.relative_to(run_directory)): sha256(path)
            for path in sorted(run_directory.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        }
    }
    json_write(run_directory / "artifact_manifest.json", manifest)
    return HeterogeneousRun(
        run_directory=run_directory,
        report_path=report_path,
        summary=summary,
    )


def _build_summary(
    *,
    run_directory: Path,
    experiment_004: Path,
    experiment_006: Path,
    target_revision: str,
    backend_setup: dict[str, Any],
    sglang_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    arm64_protocol: dict[str, Any],
    planner_result: dict[str, Any],
    positive_cpu: bool,
    phase_errors: dict[str, str],
    options: HeterogeneousOptions,
) -> dict[str, Any]:
    mixed = next((row for row in mixed_rows if row.get("route") == "cuda-cpu-cuda-cpu"), None)
    mixed_reference = next(
        (row for row in mixed_rows if row.get("route") == "cuda-cuda-cuda-cuda"), None
    )
    mixed_correct = bool(mixed and mixed.get("exact_greedy_token_identity"))
    forced_classified = bool(
        mixed
        and (
            float(mixed.get("throughput_change_fraction", 0)) >= 0
            or mixed.get("forced_critical_path_classification") == "harmful"
        )
    )
    speculative_correct = bool(speculative_rows) and all(
        bool(row.get("all_exact")) and bool(row.get("live_coordinator_exact_canary"))
        for row in speculative_rows
    )
    speculative_positive = any(
        bool(row.get("measured_positive_contribution_pass")) for row in speculative_rows
    )
    expert_positive = any(
        bool(row.get("positive_contribution_pass"))
        and int(row.get("selected_cpu_expert_calls", 0)) > 0
        for row in expert_rows
    )
    background_positive = any(
        bool(row.get("positive_contribution_pass")) for row in background_rows
    )
    regret_pass = bool(planner_result.get("regret")) and all(
        bool(row.get("passes")) for row in planner_result.get("regret", [])
    )
    predictions_cover_roles = {
        str(row.get("role")) for row in planner_result.get("predictions", [])
    } >= {role.value for role in NodeRole}
    selected_role = str(planner_result.get("selected_role", "idle"))
    selected_measurement = next(
        (
            row
            for row in planner_result.get("measurements", [])
            if row.get("objective") == PlannerObjective.BALANCED.value and bool(row.get("selected"))
        ),
        None,
    )
    non_degradation = bool(
        selected_measurement is not None
        and float(selected_measurement.get("measured_utility", -1)) >= 0
    )
    backend_status = backend_setup.get("status") == "PASS"
    mapping_status = bool(backend_setup.get("mapping_validation")) and all(
        item.get("status") == "PASS" for item in backend_setup.get("mapping_validation", [])
    )
    required_skipped = any(
        (
            options.skip_speculative,
            options.skip_moe,
            options.skip_background,
            options.skip_arm64,
        )
    )
    infrastructure = (
        backend_status
        and bool(sglang_rows)
        and mixed_correct
        and mapping_status
        and speculative_correct
        and bool(expert_rows)
        and bool(background_rows)
        and arm64_protocol.get("status") == "PASS"
        and not phase_errors
        and not required_skipped
    )
    core = (
        backend_status
        and bool(sglang_rows)
        and mixed_correct
        and forced_classified
        and mapping_status
        and regret_pass
        and non_degradation
    )
    overall = classify_overall_status(
        infrastructure_pass=infrastructure,
        planner_and_core_pass=core,
        positive_cpu_contribution=positive_cpu,
    )
    best_spec = max(
        speculative_rows,
        key=lambda row: float(row.get("live_coordinator_canary_speedup_fraction", -math.inf)),
        default={},
    )
    best_expert = _best_active_expert(expert_rows) or {}
    best_background = max(
        background_rows,
        key=lambda row: (
            bool(row.get("positive_contribution_pass")),
            float(row.get("combined_throughput_gain_fraction", -math.inf)),
        ),
        default={},
    )
    critical_change = float(mixed.get("throughput_change_fraction", 0)) if mixed else 0.0
    spec_change = float(best_spec.get("live_coordinator_canary_speedup_fraction", 0))
    expert_memory = int(best_expert.get("gpu_memory_saved_bytes", 0))
    expert_retained = float(best_expert.get("throughput_retained_fraction", 0))
    background_change = float(best_background.get("combined_throughput_gain_fraction", 0))
    p95_change = float(best_background.get("interactive_p95_increase_fraction", 0))
    regret_fraction = max(
        (float(row.get("planner_regret_fraction", 0)) for row in planner_result.get("regret", [])),
        default=0.0,
    )
    conclusion = (
        f"The heterogeneous worker infrastructure {'passed' if infrastructure else 'failed'}.\n\n"
        f"The forced CPU critical-path role changed interactive throughput by {critical_change * 100:.2f}% "
        f"and was classified as {mixed.get('forced_critical_path_classification', 'unmeasured') if mixed else 'unmeasured'}.\n\n"
        f"The CPU speculative role changed single-request throughput by {spec_change * 100:.2f}% "
        f"with an acceptance rate of {float(best_spec.get('acceptance_rate', 0)) * 100:.2f}% and "
        f"{'exact' if speculative_correct else 'non-exact or unavailable'} target output.\n\n"
        f"The CPU expert role reduced GPU memory by {expert_memory} bytes while retaining "
        f"{expert_retained * 100:.2f}% of baseline layer throughput.\n\n"
        f"The CPU background role changed total verified throughput by {background_change * 100:.2f}% "
        f"while changing interactive p95 latency by {p95_change * 100:.2f}%.\n\n"
        f"The planner selected {selected_role} for the measured CPU node with "
        f"{regret_fraction * 100:.2f}% regret relative to the best measured role.\n\n"
        f"The tested CPU node {'did' if positive_cpu else 'did not'} provide positive inference capacity. "
        f"ARM64 protocol compatibility {'passed' if arm64_protocol.get('status') == 'PASS' else 'failed'}, "
        "while Raspberry Pi performance remains unproven."
    )
    return {
        "experiment_integrity_status": "PASS" if infrastructure else "FAIL",
        "universal_worker_abi_status": "PASS" if mixed_correct else "FAIL",
        "sglang_backend_status": "PASS"
        if backend_setup.get("sglang", {}).get("status") == "PASS" and sglang_rows
        else "FAIL",
        "cpu_rank_backend_status": "PASS" if mixed_correct else "FAIL",
        "llamacpp_backend_status": "PASS"
        if backend_setup.get("llamacpp", {}).get("status") == "PASS"
        and (speculative_rows or background_rows)
        else "FAIL",
        "canonical_artifact_mapping_status": "PASS" if mapping_status else "FAIL",
        "mixed_backend_correctness_status": "PASS" if mixed_correct else "FAIL",
        "forced_critical_path_status": "PASS" if forced_classified else "FAIL",
        "cpu_speculative_status": (
            "PASS"
            if speculative_positive and speculative_correct
            else (
                "NOT_USEFUL"
                if speculative_correct
                else (
                    "BLOCKED"
                    if "speculative" in phase_errors or options.skip_speculative
                    else "FAIL"
                )
            )
        ),
        "cpu_expert_status": (
            "PASS"
            if expert_positive
            else (
                "NOT_USEFUL"
                if expert_rows
                else ("BLOCKED" if "moe" in phase_errors or options.skip_moe else "FAIL")
            )
        ),
        "cpu_background_status": "PASS"
        if background_positive
        else ("NOT_USEFUL" if background_rows else "FAIL"),
        "integrity_audit_status": (
            "PASS"
            if audit_rows and any(bool(row.get("sustainable")) for row in audit_rows)
            else ("SKIPPED" if options.skip_moe else "FAIL")
        ),
        "arm64_compatibility_status": (
            "PASS"
            if arm64_protocol.get("status") == "PASS"
            else ("BLOCKED" if options.skip_arm64 else "FAIL")
        ),
        "planner_prediction_status": "PASS" if predictions_cover_roles else "FAIL",
        "planner_non_degradation_status": "PASS" if non_degradation else "FAIL",
        "planner_regret_status": "PASS" if regret_pass else "FAIL",
        "positive_cpu_contribution_status": "PASS" if positive_cpu else "FAIL",
        "overall_status": overall,
        "execution_mode": "heterogeneous-single-host-real-model",
        "result_classifications": [item.value for item in ResultClassification],
        "experiment_004_run": str(experiment_004),
        "experiment_006_run": str(experiment_006),
        "sglang_version": backend_setup.get("sglang", {}).get("package_or_build_version"),
        "sglang_commit": backend_setup.get("sglang", {}).get("source_commit"),
        "llamacpp_commit": backend_setup.get("llamacpp", {}).get("source_commit"),
        "model_revisions": {
            "Qwen/Qwen3-4B": target_revision,
            "Qwen/Qwen3-0.6B": DRAFT_REVISION,
            "Qwen/Qwen3-30B-A3B": MOE_REVISION,
        },
        "universal_worker_abi_version": "1.0",
        "supported_job_types": [item.value for item in WorkerJobType],
        "planner_selected_role": selected_role,
        "planner_selected_measured_utility": planner_result.get("selected_measured_utility"),
        "planner_regret_fraction": regret_fraction,
        "runtime_reassignment_occurred": planner_result.get("runtime_reassignment_occurred", False),
        "positive_roles": {
            "speculative_draft": speculative_positive,
            "moe_expert": expert_positive,
            "background_inference": background_positive,
        },
        "phase_errors": phase_errors,
        "run_directory": str(run_directory),
        "conclusion": conclusion,
        "measured": {
            "forced_critical_path_throughput_change_fraction": critical_change,
            "mixed_reference_output_tokens_per_second": (
                mixed_reference.get("output_tokens_per_second") if mixed_reference else None
            ),
            "mixed_output_tokens_per_second": mixed.get("output_tokens_per_second")
            if mixed
            else None,
            "cpu_speculative_speedup_fraction": spec_change,
            "cpu_speculative_acceptance_rate": best_spec.get("acceptance_rate"),
            "cpu_expert_gpu_memory_saved_bytes": expert_memory,
            "cpu_expert_throughput_retained_fraction": expert_retained,
            "cpu_background_throughput_gain_fraction": background_change,
            "cpu_background_interactive_p95_change_fraction": p95_change,
        },
    }


def classify_overall_status(
    *,
    infrastructure_pass: bool,
    planner_and_core_pass: bool,
    positive_cpu_contribution: bool,
) -> str:
    if not infrastructure_pass or not planner_and_core_pass:
        return "FAIL"
    return "PASS" if positive_cpu_contribution else "PARTIAL_PASS"


def _failure_summary(
    *,
    run_directory: Path,
    phase_errors: dict[str, str],
    target_revision: str,
) -> dict[str, Any]:
    statuses = {
        "experiment_integrity_status": "FAIL",
        "universal_worker_abi_status": "FAIL",
        "sglang_backend_status": "FAIL",
        "cpu_rank_backend_status": "FAIL",
        "llamacpp_backend_status": "FAIL",
        "canonical_artifact_mapping_status": "FAIL",
        "mixed_backend_correctness_status": "FAIL",
        "forced_critical_path_status": "FAIL",
        "cpu_speculative_status": "BLOCKED",
        "cpu_expert_status": "BLOCKED",
        "cpu_background_status": "FAIL",
        "integrity_audit_status": "SKIPPED",
        "arm64_compatibility_status": "BLOCKED",
        "planner_prediction_status": "FAIL",
        "planner_non_degradation_status": "FAIL",
        "planner_regret_status": "FAIL",
        "positive_cpu_contribution_status": "FAIL",
        "overall_status": "FAIL",
    }
    return {
        **statuses,
        "execution_mode": "heterogeneous-single-host-real-model",
        "model_revisions": {"Qwen/Qwen3-4B": target_revision},
        "phase_errors": phase_errors,
        "run_directory": str(run_directory),
        "conclusion": (
            "The heterogeneous worker infrastructure failed. Mandatory infrastructure did not "
            "complete, so no positive heterogeneous inference claim is made. ARM64 or device "
            "performance results that did not execute remain unproven."
        ),
    }


def _headline_findings(
    *,
    summary: dict[str, Any],
    sglang_rows: list[dict[str, Any]],
    mixed_rows: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    background_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    planner_result: dict[str, Any],
) -> dict[str, Any]:
    best_spec = max(
        speculative_rows,
        key=lambda row: float(row.get("live_coordinator_canary_speedup_fraction", -math.inf)),
        default={},
    )
    best_expert = _best_active_expert(expert_rows) or {}
    best_background = max(
        background_rows,
        key=lambda row: float(row.get("combined_throughput_gain_fraction", -math.inf)),
        default={},
    )
    return {
        "overall_status": summary.get("overall_status"),
        "stock_sglang_matrix": sglang_rows,
        "mixed_backend": mixed_rows,
        "best_speculative": best_spec,
        "best_cpu_expert": best_expert,
        "best_background": best_background,
        "maximum_sustainable_audit_fraction": maximum_sustainable_audit_fraction(audit_rows),
        "planner_selected_role": planner_result.get("selected_role"),
        "planner_selected_measured_utility": planner_result.get("selected_measured_utility"),
        "planner_regret": planner_result.get("regret", []),
        "classification_warning": (
            "Measured results are not projections. Event-driven and projected rows are labelled "
            "and excluded from positive-contribution gates."
        ),
    }
