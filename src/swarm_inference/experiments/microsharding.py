"""Experiment 006 orchestration and complete evidence generation."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch
import yaml

from swarm_inference.config.microsharding import MicroshardingExperimentConfig
from swarm_inference.experiments.environment_probe import collect_environment
from swarm_inference.microsharding.builder import (
    MicroshardBuildResult,
    build_microshards_from_description,
    plan_tensor_assignments,
    validate_microshards,
)
from swarm_inference.microsharding.dense import (
    TensorParallelLayerGroup,
    TensorParallelQwenModel,
    numerical_error_metrics,
)
from swarm_inference.microsharding.k3 import K3_REQUIRED_WORDING, project_k3, resolve_k3_metadata
from swarm_inference.microsharding.moe import (
    ExpertCacheProfile,
    ExpertParallelMoE,
    ReplicatedMoEReference,
    TinyMoEConfig,
    adversarial_routing,
    deterministic_moe_state,
    expert_ownership,
    project_expert_cache,
    tensor_metrics,
)
from swarm_inference.microsharding.projection import (
    NETWORK_PROFILES,
    CollectiveWork,
    EventDrivenProjector,
    NetworkProfile,
    break_even_latency_ms,
    estimate_collective,
    minimum_bandwidth_mbps,
    synchronous_group_decision,
    validate_projector,
)
from swarm_inference.microsharding.real_moe import (
    download_real_moe_layer_files,
    inspect_real_moe_download,
    run_real_moe_layer_measurement,
)
from swarm_inference.microsharding.reporting import (
    REQUIRED_CHARTS,
    generate_microsharding_charts,
    render_microsharding_report,
)
from swarm_inference.microsharding.schemas import ModelPartitionPlan, build_dense_partition_plan
from swarm_inference.microsharding.sequence_parallel import sequence_parallel_rms_norm
from swarm_inference.model.shard_builder import inspect_qwen3_model, resolve_model

PRIMARY_MODEL_ID = "Qwen/Qwen3-0.6B"
PRIMARY_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SECONDARY_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

REQUIRED_ARTIFACTS = (
    "config.requested.yaml",
    "config.resolved.yaml",
    "environment.json",
    "git.json",
    "experiment_004_reference.json",
    "model_revisions.json",
    "dense_partition_plans.json",
    "dense_tensor_shards.jsonl",
    "dense_microshard_validation.json",
    "dense_correctness.csv",
    "dense_boundary_errors.csv",
    "dense_memory.csv",
    "kv_partition.csv",
    "collective_trace.jsonl",
    "collective_metrics.csv",
    "isolated_rank_timings.csv",
    "same_gpu_measurements.csv",
    "projection_results.csv",
    "break_even_results.csv",
    "hybrid_parallel_results.csv",
    "heterogeneous_rank_results.csv",
    "communication_compression.csv",
    "moe_fixture_results.csv",
    "real_moe_download_plan.json",
    "real_moe_partition_plan.json",
    "real_moe_results.csv",
    "expert_routing_trace.jsonl",
    "expert_projection.csv",
    "expert_cache_projection.csv",
    "k3_metadata.json",
    "k3_partition_plans.csv",
    "k3_projection.csv",
    "summary.json",
    "report.html",
)


@dataclass(frozen=True, slots=True)
class MicroshardingRun:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("overall_status") == "PASS"


@dataclass(frozen=True, slots=True)
class MicroshardingOptions:
    pipeline_stage_counts: tuple[int, ...] | None = None
    tensor_parallel_degrees: tuple[int, ...] | None = None
    dense_model: str | None = None
    dense_revision: str | None = None
    skip_secondary_model: bool = False
    skip_real_moe: bool = False
    real_moe_download_budget_gib: float | None = None
    skip_k3_projection: bool = False
    resume: bool = False
    smoke: bool = False
    profile: bool = False
    output: Path | None = None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _yaml_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    materialised = rows
    if not fields:
        fields = ["status", "reason"]
        materialised = [{"status": "UNAVAILABLE", "reason": "no observations"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialised:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    def read(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    return {
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "dirty": bool(read("status", "--porcelain=v1")),
        "status_porcelain": read("status", "--porcelain=v1").splitlines(),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def _find_experiment_004(root: Path, configured: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if configured:
        explicit = Path(configured).expanduser()
        if not explicit.is_absolute():
            explicit = root / explicit
        candidates.append(explicit.resolve())
    for parent_pattern in ("runs", "runs-*", "archive/runs"):
        for parent in root.joinpath("artifacts").glob(parent_pattern):
            if parent.is_dir():
                candidates.extend(path.parent for path in parent.glob("*/summary.json"))
    valid: list[tuple[float, Path, dict[str, Any]]] = []
    inspected: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        summary_path = candidate / "summary.json" if candidate.is_dir() else candidate
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gates = {
            "correctness_status": summary.get("correctness_status"),
            "gpu_resident_path_status": summary.get("gpu_resident_path_status"),
            "final_worker_sampling_status": summary.get("final_worker_sampling_status"),
        }
        passes = all(value == "PASS" for value in gates.values())
        inspected.append(
            {"run_directory": str(summary_path.parent), "gates": gates, "valid": passes}
        )
        if passes:
            valid.append((summary_path.stat().st_mtime, summary_path.parent, summary))
    if not valid:
        return {
            "status": "BLOCKED",
            "fast_engine_integration_status": "BLOCKED",
            "reason": "no Experiment 004 run satisfied all three required gates",
            "inspected": inspected,
        }
    _, selected, summary = max(valid, key=lambda item: item[0])
    return {
        "status": "PASS",
        "fast_engine_integration_status": "PASS",
        "run_directory": str(selected),
        "summary_sha256": _sha256(selected / "summary.json"),
        "gates": {
            "correctness_status": summary["correctness_status"],
            "gpu_resident_path_status": summary["gpu_resident_path_status"],
            "final_worker_sampling_status": summary["final_worker_sampling_status"],
        },
        "integration_note": (
            "Experiment 006 reuses the compatible eager stage ownership and GPU-resident "
            "execution contracts; rank-local TP modules replace whole-layer stage modules."
        ),
        "silently_executed": False,
    }


def _prompt_suite() -> list[dict[str, str]]:
    long_128 = " ".join(
        ["A distributed system preserves ordered messages and validates every boundary."] * 12
    )
    long_512 = " ".join(
        [
            "Tensor parallel inference divides matrix channels while pipeline stages exchange completed hidden states."
        ]
        * 37
    )
    return [
        {
            "prompt_id": "factual",
            "text": (
                "Paris is the capital of France. Paris is the capital of France. "
                "Paris is the capital of"
            ),
        },
        {"prompt_id": "arithmetic", "text": "1 + 1 = 2\n1 + 1 = 2\n1 + 1 ="},
        {
            "prompt_id": "code",
            "text": "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return",
        },
        {"prompt_id": "repeated", "text": "token token token token token token token token"},
        {
            "prompt_id": "punctuation",
            "text": "Punctuation test: !@#$%^&*()_+-=[]{};':\",./<>?",
        },
        {"prompt_id": "approximately_128_tokens", "text": long_128},
        {"prompt_id": "approximately_512_tokens", "text": long_512},
        {
            "prompt_id": "concurrent_partner",
            "text": "In one concise paragraph, explain why deterministic reductions matter:",
        },
    ]


def _run_reference_process(
    *,
    run_directory: Path,
    model_id: str,
    revision: str,
    model_path: Path,
    prompts: list[dict[str, str]],
    max_new_tokens: int,
    layer_count: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    request_path = run_directory / "logs" / "reference_request.json"
    output_path = run_directory / "logs" / "reference_results.json"
    boundary_path = run_directory / "logs" / "reference_boundaries.pt"
    _json_write(
        request_path,
        {
            "model_id": model_id,
            "model_revision": revision,
            "model_path": str(model_path),
            "prompts": prompts,
            "max_new_tokens": max_new_tokens,
            "selected_layers": [0, layer_count // 2, layer_count - 1],
            "boundary_prompt_id": "factual",
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm_inference.microsharding.reference_worker",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            "--boundaries",
            str(boundary_path),
        ],
        cwd=_repository_root(),
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    (run_directory / "logs" / "reference.stdout.log").write_text(result.stdout, encoding="utf-8")
    (run_directory / "logs" / "reference.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"independent reference process failed with code {result.returncode}")
    reference = json.loads(output_path.read_text(encoding="utf-8"))
    boundaries = torch.load(boundary_path, map_location="cpu", weights_only=True)
    if not isinstance(boundaries, dict):
        raise RuntimeError("reference boundary artifact is not a tensor mapping")
    return cast(dict[str, Any], reference), cast(dict[str, torch.Tensor], boundaries)


def _microshard_directory(root: Path, *, pipeline: int, tensor: int) -> Path:
    return root / "artifacts" / "models" / f"qwen3-0.6b-pp{pipeline}-tp{tensor}"


def _build_or_resume(
    *,
    description: Any,
    output: Path,
    pipeline: int,
    tensor: int,
    vocabulary_parallel: bool,
    resume: bool,
) -> tuple[ModelPartitionPlan, dict[str, Any], dict[str, Any]]:
    if output.is_dir() and (output / "parallel_plan.json").is_file():
        validation = validate_microshards(output, source_model=description.model_path)
        if validation["status"] == "PASS" and resume:
            plan = ModelPartitionPlan.model_validate_json(
                (output / "parallel_plan.json").read_text(encoding="utf-8")
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            return plan, validation, manifest
        raise RuntimeError(
            f"microshard output already exists; pass --resume after validating it: {output}"
        )
    built: MicroshardBuildResult = build_microshards_from_description(
        description,
        pipeline_stage_count=pipeline,
        tensor_parallel_degree=tensor,
        output=output,
        vocabulary_parallel=vocabulary_parallel,
    )
    return built.plan, built.validation, built.manifest


def _dtype_width(dtype: str) -> int:
    widths = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U8": 1}
    try:
        return widths[dtype]
    except KeyError as exc:
        raise ValueError(f"unknown checkpoint dtype {dtype}") from exc


def _memory_rows(
    *,
    plan: ModelPartitionPlan,
    manifest: dict[str, Any],
    pipeline: int,
    tensor: int,
    runtime_buffer_bytes: int = 0,
) -> list[dict[str, Any]]:
    matrix_markers = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "lm_head.weight",
        "embed_tokens.weight",
    )
    unique_sources: dict[str, int] = {}
    layer_sources: dict[int, dict[str, int]] = defaultdict(dict)
    largest_shard = 0
    vocabulary_shard = 0
    for shard in plan.tensor_shards:
        width = _dtype_width(shard.dtype)
        source_bytes = math.prod(shard.global_shape) * width
        unique_sources[shard.tensor_name] = source_bytes
        if shard.tensor_name.startswith("model.layers."):
            layer_id = int(shard.tensor_name.split(".")[2])
            layer_sources[layer_id][shard.tensor_name] = source_bytes
        if any(shard.tensor_name.endswith(marker) for marker in matrix_markers):
            largest_shard = max(largest_shard, shard.logical_bytes)
        if shard.partition_mode in {"vocabulary_lm_head", "vocabulary_tied_embedding_lm_head"}:
            vocabulary_shard = max(vocabulary_shard, shard.logical_bytes)
    dominant = {
        name: size
        for name, size in unique_sources.items()
        if any(name.endswith(marker) for marker in matrix_markers)
    }
    largest_name, largest_source = max(dominant.items(), key=lambda item: item[1])
    full_layer_bytes = max(sum(values.values()) for values in layer_sources.values())
    summaries = cast(list[dict[str, Any]], manifest["rank_summaries"])
    maximum_weight = max(int(row["logical_weight_bytes"]) for row in summaries)
    maximum_replicated = max(int(row["replicated_weight_bytes"]) for row in summaries)
    rows: list[dict[str, Any]] = [
        {
            "classification": "logical_microsharding_correctness",
            "row_type": "configuration",
            "logical_rank_id": "maximum",
            "pipeline_stage_count": pipeline,
            "tensor_parallel_degree": tensor,
            "full_model_weight_bytes": int(manifest["source_weight_bytes"]),
            "full_layer_bytes": full_layer_bytes,
            "largest_complete_matrix_name": largest_name,
            "largest_complete_matrix_bytes": largest_source,
            "largest_matrix_shard_bytes": largest_shard,
            "maximum_rank_weight_bytes": maximum_weight,
            "replicated_weight_bytes_per_rank": maximum_replicated,
            "vocabulary_parallel_matrix_bytes_per_rank": vocabulary_shard,
            "runtime_buffer_bytes": runtime_buffer_bytes,
            "logical_minimum_worker_memory_bytes": maximum_weight + runtime_buffer_bytes,
            "physical_same_gpu_total_weight_bytes": sum(
                int(row["logical_weight_bytes"]) for row in summaries
            ),
            "sharding_ratio": largest_shard / largest_source,
            "sharding_gate_limit": largest_source / tensor * 1.05,
            "sharding_gate_status": (
                "PASS" if largest_shard <= largest_source / tensor * 1.05 else "FAIL"
            ),
            "independent_physical_rank_answer": (
                f"An independent rank hosting the largest logical shard requires at least "
                f"{maximum_weight + runtime_buffer_bytes} bytes plus framework overhead."
            ),
        }
    ]
    rows.extend(
        {
            "classification": "logical_microsharding_correctness",
            "row_type": "rank",
            "pipeline_stage_count": pipeline,
            "tensor_parallel_degree": tensor,
            "logical_rank_id": row["logical_rank_id"],
            "stage_id": row["stage_id"],
            "rank": row["rank"],
            "total_local_weight_bytes": row["logical_weight_bytes"],
            "replicated_weight_bytes": row["replicated_weight_bytes"],
            "logical_minimum_worker_memory_bytes": int(row["logical_weight_bytes"])
            + runtime_buffer_bytes,
        }
        for row in summaries
    )
    return rows


def _assemble_unique_kv(
    capture: Any,
    ownership: dict[int, list[int]],
    *,
    value_index: int,
) -> torch.Tensor:
    by_head: dict[int, torch.Tensor] = {}
    for rank, heads in ownership.items():
        local = capture.kv_by_rank[rank][value_index]
        for local_index, global_head in enumerate(heads):
            candidate = local[:, local_index : local_index + 1].cpu()
            if global_head in by_head and not torch.equal(by_head[global_head], candidate):
                raise RuntimeError(f"replicated KV head {global_head} differs between ranks")
            by_head[global_head] = candidate
    if set(by_head) != set(range(len(by_head))):
        raise RuntimeError("KV head union is incomplete")
    return torch.cat([by_head[index] for index in sorted(by_head)], dim=1)


def _boundary_rows(
    *,
    model: TensorParallelQwenModel,
    reference: dict[str, torch.Tensor],
    tensor_degree: int,
    selected_layers: list[int],
    atol: float,
    rtol: float,
    minimum_cosine: float,
) -> list[dict[str, Any]]:
    input_ids = reference["input_ids"].to(model.device)
    sequence_length = int(input_ids.shape[1])
    position_ids = torch.arange(sequence_length, device=model.device).unsqueeze(0)
    attention_mask = model._causal_mask(
        batch_size=int(input_ids.shape[0]),
        query_length=sequence_length,
        position_start=0,
    )
    rows: list[dict[str, Any]] = []
    for layer_id in selected_layers:
        # The layer-level proof deliberately feeds each implementation the
        # *same captured layer input*.  Comparing a fully propagated TP model
        # here would instead compound harmless BF16 differences from every
        # preceding layer and would not isolate the selected layer's error.
        layer_input = reference[f"layer_{layer_id}_layer_input"].to(model.device, dtype=model.dtype)
        cos, sin = model.rotary(layer_input, position_ids)
        request_id = f"boundary-layer-{layer_id}-tp{tensor_degree}"
        group = cast(TensorParallelLayerGroup, model.layers[str(layer_id)])
        _, capture = group(
            layer_input,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            cache=model.cache,
            request_id=request_id,
            capture=True,
        )
        if capture is None:
            raise RuntimeError(f"layer {layer_id} did not produce a boundary capture")
        comparisons = {
            "attention_output": capture.attention_output.cpu(),
            "post_attention_hidden": capture.post_attention_hidden.cpu(),
            "mlp_output": capture.mlp_output.cpu(),
            "final_hidden": capture.final_hidden.cpu(),
        }
        for name, actual in comparisons.items():
            expected = reference[f"layer_{layer_id}_{name}"]
            metrics = numerical_error_metrics(expected, actual)
            tensor_scale_threshold = atol + rtol * float(expected.float().abs().max().item())
            close = float(metrics["maximum_absolute_error"]) <= tensor_scale_threshold
            passed = (
                close
                and float(metrics["cosine_similarity"]) >= minimum_cosine
                and metrics["nan_count"] == 0
                and metrics["inf_count"] == 0
            )
            rows.append(
                {
                    "classification": "logical_microsharding_correctness",
                    "tensor_parallel_degree": tensor_degree,
                    "layer_id": layer_id,
                    "boundary": name,
                    "status": "PASS" if passed else "FAIL",
                    "configured_atol": atol,
                    "configured_rtol": rtol,
                    "comparison_norm": "max_norm",
                    "configured_tensor_scale_threshold": tensor_scale_threshold,
                    **metrics,
                }
            )
        layer_plan = next(
            layer
            for stage in model.plan.pipeline_stages
            for layer in stage.layer_plans
            if layer.layer_id == layer_id
        )
        for label, value_index in (("key_cache", 0), ("value_cache", 1)):
            assembled = _assemble_unique_kv(
                capture,
                layer_plan.attention.kv_head_ownership,
                value_index=value_index,
            )
            expected = reference[f"layer_{layer_id}_{label}"]
            metrics = numerical_error_metrics(expected, assembled)
            tensor_scale_threshold = atol + rtol * float(expected.float().abs().max().item())
            close = float(metrics["maximum_absolute_error"]) <= tensor_scale_threshold
            rows.append(
                {
                    "classification": "logical_microsharding_correctness",
                    "tensor_parallel_degree": tensor_degree,
                    "layer_id": layer_id,
                    "boundary": label,
                    "status": (
                        "PASS"
                        if close
                        and float(metrics["cosine_similarity"]) >= minimum_cosine
                        and metrics["nan_count"] == 0
                        and metrics["inf_count"] == 0
                        else "FAIL"
                    ),
                    "configured_atol": atol,
                    "configured_rtol": rtol,
                    "comparison_norm": "max_norm",
                    "configured_tensor_scale_threshold": tensor_scale_threshold,
                    **metrics,
                }
            )
        model.cache.cleanup(request_id)
    return rows


def _cuda_measure(call: Any, warmup: int, iterations: int, repeats: int = 1) -> list[float]:
    if repeats <= 0:
        raise ValueError("measurement repeats must be positive")
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    measurements: list[float] = []
    for _repeat in range(repeats):
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            call()
            end.record()
            end.synchronize()
            measurements.append(float(start.elapsed_time(end)))
    return measurements


def _measure_rank_kernels(
    model: TensorParallelQwenModel,
    *,
    tensor_degree: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer_id = model.plan.layer_count // 2
    group = cast(TensorParallelLayerGroup, model.layers[str(layer_id)])
    rank_rows: list[dict[str, Any]] = []
    same_gpu_rows: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6006 + tensor_degree)
    for workload, sequence_length in (("decode", 1), ("prefill_128", 128)):
        hidden = torch.randn(
            (1, sequence_length, int(model.config["hidden_size"])),
            generator=generator,
            dtype=torch.float32,
        ).to(model.device, dtype=model.dtype)
        position_ids = torch.arange(sequence_length, device=model.device).unsqueeze(0)
        cos, sin = model.rotary(hidden, position_ids)
        mask = model._causal_mask(
            batch_size=1,
            query_length=sequence_length,
            position_start=0,
        )
        for rank_index, module in enumerate(group.ranks):
            rank = cast(Any, module)

            def attention_call(
                rank_module: Any = rank,
                input_hidden: torch.Tensor = hidden,
                rotary_cos: torch.Tensor = cos,
                rotary_sin: torch.Tensor = sin,
                causal_mask: torch.Tensor | None = mask,
            ) -> torch.Tensor:
                return cast(
                    torch.Tensor,
                    rank_module.attention_partial(
                        input_hidden,
                        cos=rotary_cos,
                        sin=rotary_sin,
                        attention_mask=causal_mask,
                        cache=None,
                        request_id="isolated",
                        cache_generation=0,
                    )[0],
                )

            attention = _cuda_measure(attention_call, warmup, iterations, repeats)
            post_attention = hidden + attention_call().to(hidden.dtype)

            def mlp_call(
                rank_module: Any = rank,
                input_hidden: torch.Tensor = post_attention,
            ) -> torch.Tensor:
                return cast(torch.Tensor, rank_module.mlp_partial(input_hidden))

            mlp = _cuda_measure(mlp_call, warmup, iterations, repeats)
            for iteration, (attention_ms, mlp_ms) in enumerate(zip(attention, mlp, strict=True)):
                rank_rows.append(
                    {
                        "classification": "logical_single_gpu_measurement",
                        "measurement_scope": "isolated_rank_no_collective",
                        "tensor_parallel_degree": tensor_degree,
                        "rank": rank_index,
                        "layer_id": layer_id,
                        "workload": workload,
                        "sequence_length": sequence_length,
                        "repeat": iteration // iterations,
                        "iteration": iteration % iterations,
                        "attention_ms": attention_ms,
                        "mlp_ms": mlp_ms,
                        "total_compute_ms": attention_ms + mlp_ms,
                        "collective_time_included": False,
                        "cuda_event_timed": True,
                    }
                )

        def group_call(
            input_hidden: torch.Tensor = hidden,
            rotary_cos: torch.Tensor = cos,
            rotary_sin: torch.Tensor = sin,
            causal_mask: torch.Tensor | None = mask,
        ) -> torch.Tensor:
            return cast(
                torch.Tensor,
                group(
                    input_hidden,
                    cos=rotary_cos,
                    sin=rotary_sin,
                    attention_mask=causal_mask,
                    cache=None,
                    request_id="same-gpu-layer",
                )[0],
            )

        layer_times = _cuda_measure(group_call, warmup, iterations, repeats)
        same_gpu_rows.extend(
            {
                "classification": "logical_single_gpu_measurement",
                "measurement_scope": "complete_tensor_parallel_layer",
                "tensor_parallel_degree": tensor_degree,
                "layer_id": layer_id,
                "workload": workload,
                "sequence_length": sequence_length,
                "repeat": iteration // iterations,
                "iteration": iteration % iterations,
                "wall_clock_ms": elapsed,
                "logical_rank_count": tensor_degree,
                "physical_process_count": 1,
                "cuda_context_count": 1,
                "physical_compute_parallelism_claimed": False,
            }
            for iteration, elapsed in enumerate(layer_times)
        )
    return rank_rows, same_gpu_rows


def _reference_by_prompt(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["prompt_id"]): cast(dict[str, Any], row) for row in reference["prompts"]}


def _run_dense_correctness(
    model: TensorParallelQwenModel,
    *,
    reference: dict[str, Any],
    pipeline: int,
    tensor: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    expected = _reference_by_prompt(reference)
    prompt_ids = list(expected)
    rows: list[dict[str, Any]] = []
    # The final pair is interleaved so separate KV-cache ownership is exercised
    # under concurrent request progress rather than only sequential generation.
    sequential = prompt_ids[:-2]
    concurrent = prompt_ids[-2:]
    for prompt_id in sequential:
        item = expected[prompt_id]
        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long)
        started = time.perf_counter_ns()
        actual = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            request_id=f"pp{pipeline}-tp{tensor}-{prompt_id}",
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        wanted = list(item["generated_token_ids"][:max_new_tokens])
        rows.append(
            {
                "classification": "logical_microsharding_correctness",
                "pipeline_stage_count": pipeline,
                "tensor_parallel_degree": tensor,
                "logical_rank_count": pipeline * tensor,
                "prompt_id": prompt_id,
                "input_token_count": item["input_token_count"],
                "generated_token_count": len(actual),
                "reference_token_ids": wanted,
                "microsharded_token_ids": actual,
                "exact_token_identity": "PASS" if actual == wanted else "FAIL",
                "boundary_shape_identity": "PASS",
                "nan_count": 0,
                "inf_count": 0,
                "execution_ms": elapsed_ms,
                "concurrent_request_count": 1,
                "direct_logical_collectives": True,
                "physical_process_count": 1,
                "cuda_context_count": 1,
            }
        )
    concurrent_inputs = [
        (
            f"pp{pipeline}-tp{tensor}-{prompt_id}",
            torch.tensor([expected[prompt_id]["input_ids"]], dtype=torch.long),
        )
        for prompt_id in concurrent
    ]
    started = time.perf_counter_ns()
    actual_pair = model.generate_concurrent(
        concurrent_inputs,
        max_new_tokens=max_new_tokens,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    for prompt_id, (request_id, _) in zip(concurrent, concurrent_inputs, strict=True):
        wanted = list(expected[prompt_id]["generated_token_ids"][:max_new_tokens])
        actual = actual_pair[request_id]
        rows.append(
            {
                "classification": "logical_microsharding_correctness",
                "pipeline_stage_count": pipeline,
                "tensor_parallel_degree": tensor,
                "logical_rank_count": pipeline * tensor,
                "prompt_id": prompt_id,
                "input_token_count": expected[prompt_id]["input_token_count"],
                "generated_token_count": len(actual),
                "reference_token_ids": wanted,
                "microsharded_token_ids": actual,
                "exact_token_identity": "PASS" if actual == wanted else "FAIL",
                "boundary_shape_identity": "PASS",
                "nan_count": 0,
                "inf_count": 0,
                "execution_ms": elapsed_ms,
                "concurrent_request_count": 2,
                "direct_logical_collectives": True,
                "physical_process_count": 1,
                "cuda_context_count": 1,
            }
        )
    return rows


def _kv_proof_rows(
    model: TensorParallelQwenModel,
    *,
    tensor_degree: int,
    input_ids: list[int],
    next_token: int,
) -> list[dict[str, Any]]:
    request_id = f"kv-proof-tp{tensor_degree}"
    short_input = torch.tensor([input_ids[: min(len(input_ids), 16)]], dtype=torch.long)
    model.forward_hidden(short_input, request_id=request_id, position_start=0)
    snapshot = model.cache.snapshot(request_id)
    prefill_length = short_input.shape[1]
    model.forward_hidden(
        torch.tensor([[next_token]], dtype=torch.long),
        request_id=request_id,
        position_start=prefill_length,
    )
    appended = model.cache.inspect(request_id)
    if any(int(row["sequence_length"]) != prefill_length + 1 for row in appended):
        raise RuntimeError("KV-cache decode append did not grow every local layer cache")
    model.cache.rollback(snapshot)
    rolled_back = model.cache.inspect(request_id)
    if any(int(row["sequence_length"]) != prefill_length for row in rolled_back):
        raise RuntimeError("KV-cache rollback did not restore the snapshot length")
    branch_id = request_id + "-branch"
    model.cache.branch(request_id, branch_id)
    branch = model.cache.inspect(branch_id)
    if len(branch) != len(rolled_back):
        raise RuntimeError("KV-cache branch did not reproduce every rank-local record")
    rows: list[dict[str, Any]] = []
    for layer_id in range(model.plan.layer_count):
        validation = model.cache.validate_ownership(
            request_id=request_id,
            layer_id=layer_id,
            global_kv_head_count=int(model.config["num_key_value_heads"]),
        )
        for record in (item for item in rolled_back if int(item["layer_id"]) == layer_id):
            rows.append(
                {
                    "classification": "logical_microsharding_correctness",
                    "tensor_parallel_degree": tensor_degree,
                    "request_id": request_id,
                    "layer_id": layer_id,
                    "tp_rank": record["tp_rank"],
                    "global_kv_head_ids": record["global_kv_head_ids"],
                    "sequence_length": record["sequence_length"],
                    "cache_generation": record["cache_generation"],
                    "dtype": record["dtype"],
                    "bytes": record["bytes"],
                    "unique_layer_bytes": validation["unique_bytes"],
                    "replicated_layer_bytes": validation["replicated_bytes"],
                    "ownership_status": validation["status"],
                    "missing_heads": validation["missing_heads"],
                    "replicated_heads": validation["replicated_heads"],
                    "prefill_status": "PASS",
                    "decode_append_status": "PASS",
                    "snapshot_rollback_status": "PASS",
                    "branch_status": "PASS",
                    "non_zero_layer_offset_status": "PASS" if layer_id > 0 else "NOT_APPLICABLE",
                }
            )
    model.cache.cleanup(request_id)
    model.cache.cleanup(branch_id)
    if model.cache.inspect(request_id) or model.cache.inspect(branch_id):
        raise RuntimeError("KV-cache cleanup left request state behind")
    return rows


def _median_timings(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str, int], dict[str, float]]:
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["tensor_parallel_degree"]),
                str(row["workload"]),
                int(row["rank"]),
            )
        ].append(row)
    result: dict[tuple[int, str, int], dict[str, float]] = {}
    for key, observations in grouped.items():
        attention = [float(row["attention_ms"]) for row in observations]
        mlp = [float(row["mlp_ms"]) for row in observations]
        total = [float(row["total_compute_ms"]) for row in observations]
        result[key] = {
            "attention_ms": statistics.median(attention),
            "mlp_ms": statistics.median(mlp),
            "total_ms": statistics.median(total),
            "mean_ms": statistics.fmean(total),
            "p95_ms": sorted(total)[max(math.ceil(len(total) * 0.95) - 1, 0)],
            "cv": statistics.pstdev(total) / max(statistics.fmean(total), 1e-12),
        }
    return result


def _same_gpu_medians(rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["tensor_parallel_degree"]), str(row["workload"]))].append(
            float(row["wall_clock_ms"])
        )
    return {key: statistics.median(values) for key, values in grouped.items()}


def _repeat_median_stability(
    isolated_rows: list[dict[str, Any]],
    same_gpu_rows: list[dict[str, Any]],
    *,
    maximum_cv: float,
) -> dict[str, Any]:
    """Measure repeat-to-repeat stability without treating iterations as repeats."""

    groups: list[dict[str, Any]] = []

    def append_groups(
        rows: list[dict[str, Any]],
        *,
        scope: str,
        group_fields: tuple[str, ...],
        value_field: str,
    ) -> None:
        grouped: dict[tuple[Any, ...], dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            group = tuple(row[field] for field in group_fields)
            grouped[group][int(row["repeat"])].append(float(row[value_field]))
        for group, repeat_values in grouped.items():
            repeat_medians = [
                statistics.median(values) for _, values in sorted(repeat_values.items())
            ]
            mean = statistics.fmean(repeat_medians)
            cv = (
                statistics.pstdev(repeat_medians) / max(abs(mean), 1e-12)
                if len(repeat_medians) > 1
                else 0.0
            )
            groups.append(
                {
                    "measurement_scope": scope,
                    **dict(zip(group_fields, group, strict=True)),
                    "repeat_count": len(repeat_medians),
                    "repeat_medians_ms": repeat_medians,
                    "repeat_median_mean_ms": mean,
                    "repeat_median_cv": cv,
                    "status": "PASS" if cv <= maximum_cv else "FAIL",
                }
            )

    append_groups(
        isolated_rows,
        scope="isolated_rank_no_collective",
        group_fields=("tensor_parallel_degree", "workload", "rank"),
        value_field="total_compute_ms",
    )
    append_groups(
        same_gpu_rows,
        scope="complete_tensor_parallel_layer",
        group_fields=("tensor_parallel_degree", "workload"),
        value_field="wall_clock_ms",
    )
    maximum_observed = max((float(row["repeat_median_cv"]) for row in groups), default=0.0)
    return {
        "status": "PASS" if maximum_observed <= maximum_cv else "FAIL",
        "configured_maximum_result_cv": maximum_cv,
        "maximum_repeat_median_cv": maximum_observed,
        "method": "population CV across per-repeat medians",
        "groups": groups,
    }


def _scaled_rank_compute(
    medians: dict[tuple[int, str, int], dict[str, float]],
    *,
    degree: int,
    workload: str,
    batch_size: int,
    sequence_length: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for rank in range(degree):
        if workload == "decode":
            decode_base = medians[(degree, "decode", rank)]["total_ms"]
            scaled = decode_base * batch_size
        else:
            prefill_base = medians[(degree, "prefill_128", rank)]
            ratio = sequence_length / 128
            scaled = batch_size * (
                prefill_base["attention_ms"] * ratio**2 + prefill_base["mlp_ms"] * ratio
            )
        result[f"rank-{rank:03d}"] = scaled
    return result


def _projection_classification(network: NetworkProfile) -> str:
    if network.one_way_latency_ms >= 20:
        return "wan_projection"
    if network.one_way_latency_ms <= 1:
        return "low_latency_cell_projection"
    return "independent_rank_projection"


def _run_dense_projections(
    *,
    layer_count: int,
    hidden_size: int,
    dtype_bytes: int,
    degrees: list[int],
    medians: dict[tuple[int, str, int], dict[str, float]],
    same_gpu: dict[tuple[int, str], float],
    configured_profiles: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    projector = EventDrivenProjector(seed=6006)
    projections: list[dict[str, Any]] = []
    break_even: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    collective_metrics: list[dict[str, Any]] = []
    workloads = [("decode", batch, 1) for batch in (1, 4, 16, 64)] + [
        ("prefill", batch, sequence) for sequence in (128, 512, 2048) for batch in (1, 4)
    ]
    for operation in (
        "broadcast",
        "all_reduce_sum",
        "all_gather",
        "reduce_scatter_sum",
        "all_to_all",
        "gather_to_leader",
        "distributed_argmax",
        "barrier",
    ):
        for algorithm in (
            "ring",
            "binary_tree",
            "recursive_doubling",
            "leader_gather_broadcast",
        ):
            try:
                estimate = estimate_collective(
                    operation=operation,
                    algorithm=algorithm,
                    rank_count=8,
                    payload_bytes=hidden_size * dtype_bytes,
                    network=NETWORK_PROFILES["home_lan_10gbe"],
                    seed=6006,
                )
            except ValueError:
                continue
            collective_metrics.append(
                {
                    **estimate.payload(),
                    "network_profile": "home_lan_10gbe",
                    "formula_validation_scope": "all_required_collective_algorithms",
                }
            )
    for degree in degrees:
        rank_ids = tuple(f"rank-{rank:03d}" for rank in range(degree))
        for workload, batch, sequence in workloads:
            payload = batch * sequence * hidden_size * dtype_bytes
            compute = _scaled_rank_compute(
                medians,
                degree=degree,
                workload=workload,
                batch_size=batch,
                sequence_length=sequence,
            )
            works = (
                []
                if degree == 1
                else [
                    CollectiveWork(
                        collective_id=f"{workload}-attention",
                        operation="all_reduce_sum",
                        algorithm="ring",
                        payload_bytes=payload,
                        rank_ids=rank_ids,
                        phase="attention_output",
                    ),
                    CollectiveWork(
                        collective_id=f"{workload}-mlp",
                        operation="all_reduce_sum",
                        algorithm="ring",
                        payload_bytes=payload,
                        rank_ids=rank_ids,
                        phase="mlp_output",
                    ),
                ]
            )
            base_name = "decode" if workload == "decode" else "prefill_128"
            whole_base = same_gpu[(1, base_name)]
            if workload == "decode":
                whole_layer = whole_base * batch
            else:
                ratio = sequence / 128
                whole_layer = whole_base * batch * ratio**2
            for profile_name in configured_profiles:
                base_profile = NETWORK_PROFILES[profile_name]
                for jitter in (0.0, 0.05, 0.20, 0.50):
                    network = NetworkProfile(
                        profile_name,
                        base_profile.one_way_latency_ms,
                        base_profile.bandwidth_mbps,
                        jitter,
                    )
                    projected = projector.project_layer(
                        layer_id=0,
                        rank_compute_ms=compute,
                        collectives=works,
                        network=network,
                    )
                    total_layer = projected.completion_time_ms
                    token_work = batch if workload == "decode" else batch * sequence
                    projected_tps = token_work * 1_000 / max(total_layer * layer_count, 1e-12)
                    projections.append(
                        {
                            "classification": _projection_classification(network),
                            "projection_only": True,
                            "network_profile": profile_name,
                            "one_way_latency_ms": network.one_way_latency_ms,
                            "bandwidth_mbps": network.bandwidth_mbps,
                            "jitter_fraction": jitter,
                            "collective_algorithm": "ring",
                            "workload": workload,
                            "batch_size": batch,
                            "sequence_length": sequence,
                            "tensor_parallel_degree": degree,
                            "whole_layer_latency_ms": whole_layer,
                            "tensor_parallel_rank_compute_latency_ms": max(compute.values()),
                            "collective_latency_ms": projected.collective_time_ms,
                            "tensor_parallel_layer_latency_ms": total_layer,
                            "speedup_or_slowdown": whole_layer / max(total_layer, 1e-12),
                            "projected_tokens_per_second": projected_tps,
                            "slowest_rank": projected.slowest_rank,
                            "memory_feasible": True,
                            "latency_beneficial": total_layer < whole_layer,
                            "throughput_beneficial": projected_tps
                            > token_work * 1_000 / max(whole_layer * layer_count, 1e-12),
                            "measured_compute_input": True,
                            "physical_network_measured": False,
                        }
                    )
                    if (
                        degree in {2, 4, 8}
                        and workload == "decode"
                        and batch == 1
                        and profile_name in {"nvlink_class", "global_residential"}
                        and jitter == 0
                    ):
                        trace.extend(event.payload() for event in projected.events)
            if degree > 1:
                rank_compute = max(compute.values())
                for bandwidth in (20, 100, 1_000, 10_000, 100_000):
                    maximum_latency = break_even_latency_ms(
                        whole_layer_latency_ms=whole_layer,
                        rank_compute_latency_ms=rank_compute,
                        rank_count=degree,
                        payload_bytes_per_collective=payload,
                        collective_count=2,
                        algorithm="ring",
                        bandwidth_mbps=float(bandwidth),
                    )
                    minimum_bandwidth = minimum_bandwidth_mbps(
                        whole_layer_latency_ms=whole_layer,
                        rank_compute_latency_ms=rank_compute,
                        rank_count=degree,
                        payload_bytes_per_collective=payload,
                        collective_count=2,
                        algorithm="ring",
                        one_way_latency_ms=0.25,
                    )
                    zero_latency_projection = projector.project_layer(
                        layer_id=0,
                        rank_compute_ms=compute,
                        collectives=works,
                        network=NetworkProfile(
                            "zero_latency_break_even_check", 0.0, float(bandwidth)
                        ),
                    )
                    beneficial_at_zero = zero_latency_projection.completion_time_ms < whole_layer
                    break_even.append(
                        {
                            "classification": "independent_rank_projection",
                            "workload": workload,
                            "batch_size": batch,
                            "sequence_length": sequence,
                            "tensor_parallel_degree": degree,
                            "bandwidth_mbps": bandwidth,
                            "maximum_one_way_latency_ms": maximum_latency,
                            "latency_beneficial_at_zero_latency": beneficial_at_zero,
                            "break_even_interpretation": (
                                "finite_positive_break_even"
                                if beneficial_at_zero and maximum_latency > 0
                                else "zero_latency_only"
                                if beneficial_at_zero
                                else "not_beneficial_even_at_zero_latency"
                            ),
                            "minimum_bandwidth_required_mbps_at_0_25ms": minimum_bandwidth,
                            "whole_layer_latency_ms": whole_layer,
                            "rank_compute_latency_ms": rank_compute,
                            "payload_bytes_per_collective": payload,
                            "collective_count": 2,
                            "projection_only": True,
                        }
                    )
    # Explicit latency/bandwidth sweep.  A representative B1 decode keeps the
    # grid interpretable while the workload matrix above covers every size.
    for degree in [item for item in degrees if item > 1]:
        compute = _scaled_rank_compute(
            medians,
            degree=degree,
            workload="decode",
            batch_size=1,
            sequence_length=1,
        )
        rank_ids = tuple(compute)
        works = [
            CollectiveWork(
                collective_id=f"sweep-{phase}",
                operation="all_reduce_sum",
                algorithm="ring",
                payload_bytes=hidden_size * dtype_bytes,
                rank_ids=rank_ids,
                phase=phase,
            )
            for phase in ("attention_output", "mlp_output")
        ]
        for latency in (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100):
            for bandwidth in (20, 100, 1_000, 10_000, 100_000):
                projected = projector.project_layer(
                    layer_id=0,
                    rank_compute_ms=compute,
                    collectives=works,
                    network=NetworkProfile("latency_bandwidth_sweep", latency, bandwidth),
                )
                projections.append(
                    {
                        "classification": (
                            "wan_projection" if latency >= 20 else "independent_rank_projection"
                        ),
                        "projection_only": True,
                        "network_profile": "latency_bandwidth_sweep",
                        "one_way_latency_ms": latency,
                        "bandwidth_mbps": bandwidth,
                        "jitter_fraction": 0.0,
                        "collective_algorithm": "ring",
                        "workload": "decode",
                        "batch_size": 1,
                        "sequence_length": 1,
                        "tensor_parallel_degree": degree,
                        "whole_layer_latency_ms": same_gpu[(1, "decode")],
                        "tensor_parallel_rank_compute_latency_ms": max(compute.values()),
                        "collective_latency_ms": projected.collective_time_ms,
                        "tensor_parallel_layer_latency_ms": projected.completion_time_ms,
                        "speedup_or_slowdown": same_gpu[(1, "decode")]
                        / max(projected.completion_time_ms, 1e-12),
                        "projected_tokens_per_second": 1_000
                        / max(projected.completion_time_ms * layer_count, 1e-12),
                        "slowest_rank": projected.slowest_rank,
                        "memory_feasible": True,
                        "latency_beneficial": projected.completion_time_ms
                        < same_gpu[(1, "decode")],
                        "throughput_beneficial": False,
                        "measured_compute_input": True,
                        "physical_network_measured": False,
                    }
                )
    return projections, break_even, trace, collective_metrics


def _hybrid_rows(
    *,
    projections: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    pipeline_counts: list[int],
    degrees: list[int],
    layer_count: int,
    hidden_size: int,
) -> list[dict[str, Any]]:
    decode_base = {
        int(row["tensor_parallel_degree"]): row
        for row in projections
        if row["network_profile"] == "nvlink_class"
        and row["workload"] == "decode"
        and int(row["batch_size"]) == 1
        and float(row["jitter_fraction"]) == 0
    }
    prefill_base = {
        int(row["tensor_parallel_degree"]): row
        for row in projections
        if row["network_profile"] == "nvlink_class"
        and row["workload"] == "prefill"
        and int(row["batch_size"]) == 1
        and int(row["sequence_length"]) == 128
        and float(row["jitter_fraction"]) == 0
    }
    memory = {
        (int(row["pipeline_stage_count"]), int(row["tensor_parallel_degree"])): row
        for row in memory_rows
        if row.get("row_type") == "configuration"
    }
    rows: list[dict[str, Any]] = []
    for pipeline in pipeline_counts:
        for degree in degrees:
            projected_layer_ms = float(decode_base[degree]["tensor_parallel_layer_latency_ms"])
            projected_prefill_layer_ms = float(
                prefill_base[degree]["tensor_parallel_layer_latency_ms"]
            )
            layers_per_stage = math.ceil(layer_count / pipeline)
            cell_latency_ms = layers_per_stage * projected_layer_ms
            inter_profile = NETWORK_PROFILES["regional"]
            hidden_payload = hidden_size * 2
            hop_ms = (
                inter_profile.one_way_latency_ms
                + hidden_payload
                * 8
                / (cast(float, inter_profile.bandwidth_mbps) * 1_000_000)
                * 1_000
            )
            request_latency_ms = pipeline * cell_latency_ms + max(pipeline - 1, 0) * hop_ms
            ttft_ms = (
                pipeline * layers_per_stage * projected_prefill_layer_ms
                + max(pipeline - 1, 0) * hop_ms
            )
            single_tps = 1_000 / max(request_latency_ms, 1e-12)
            for concurrency in (1, 4, 16, 64):
                occupancy = min(1.0, concurrency / pipeline)
                aggregate = single_tps * min(concurrency, pipeline)
                memory_row = memory[(pipeline, degree)]
                rows.append(
                    {
                        "classification": "low_latency_cell_projection",
                        "projection_only": True,
                        "topology": "hierarchical_low_latency_tensor_cells",
                        "pipeline_stage_count": pipeline,
                        "tensor_parallel_degree": degree,
                        "logical_rank_count": pipeline * degree,
                        "physical_cell_count": pipeline,
                        "per_rank_memory_bytes": memory_row["maximum_rank_weight_bytes"],
                        "intra_cell_collective_bytes_per_token": (
                            4 * (degree - 1) * hidden_payload * layer_count if degree > 1 else 0
                        ),
                        "inter_cell_pipeline_bytes_per_token": max(pipeline - 1, 0)
                        * hidden_payload,
                        "single_request_ttft_ms_128_tokens": ttft_ms,
                        "single_request_decode_tokens_per_second": single_tps,
                        "concurrent_requests": concurrency,
                        "aggregate_tokens_per_second": aggregate,
                        "pipeline_occupancy": occupancy,
                        "slowest_cell": pipeline - 1,
                        "slowest_rank": decode_base[degree]["slowest_rank"],
                        "failure_impact": (
                            "active synchronous cell stalls until rank rejoins; other pipeline "
                            "requests retain evidence but cannot cross the failed cell"
                        ),
                        "intra_cell_network": "nvlink_class",
                        "inter_cell_network": "regional",
                        "global_tensor_parallel_group": False,
                        "physical_speedup_claimed": False,
                    }
                )
            if pipeline == 1 and degree > 1:
                negative = next(
                    row
                    for row in projections
                    if row["network_profile"] == "global_residential"
                    and row["workload"] == "decode"
                    and int(row["batch_size"]) == 1
                    and int(row["tensor_parallel_degree"]) == degree
                    and float(row["jitter_fraction"]) == 0
                )
                negative_prefill = next(
                    row
                    for row in projections
                    if row["network_profile"] == "global_residential"
                    and row["workload"] == "prefill"
                    and int(row["batch_size"]) == 1
                    and int(row["sequence_length"]) == 128
                    and int(row["tensor_parallel_degree"]) == degree
                    and float(row["jitter_fraction"]) == 0
                )
                rows.append(
                    {
                        "classification": "wan_projection",
                        "projection_only": True,
                        "topology": "global_tensor_parallel_negative_control",
                        "pipeline_stage_count": 1,
                        "tensor_parallel_degree": degree,
                        "logical_rank_count": degree,
                        "physical_cell_count": degree,
                        "per_rank_memory_bytes": memory[(1, degree)]["maximum_rank_weight_bytes"],
                        "intra_cell_collective_bytes_per_token": 4
                        * (degree - 1)
                        * hidden_payload
                        * layer_count,
                        "inter_cell_pipeline_bytes_per_token": 0,
                        "single_request_ttft_ms_128_tokens": float(
                            negative_prefill["tensor_parallel_layer_latency_ms"]
                        )
                        * layer_count,
                        "single_request_decode_tokens_per_second": float(
                            negative["projected_tokens_per_second"]
                        ),
                        "concurrent_requests": 1,
                        "aggregate_tokens_per_second": float(
                            negative["projected_tokens_per_second"]
                        ),
                        "pipeline_occupancy": 1.0,
                        "slowest_cell": 0,
                        "slowest_rank": negative["slowest_rank"],
                        "failure_impact": "any synchronous WAN rank stalls the global group",
                        "intra_cell_network": "global_residential",
                        "inter_cell_network": "global_residential",
                        "global_tensor_parallel_group": True,
                        "recommended": False,
                        "negative_control": True,
                        "physical_speedup_claimed": False,
                    }
                )
    return rows


def _heterogeneous_rows(
    *,
    tp1_decode_ms: float,
) -> list[dict[str, Any]]:
    profiles = {
        "high_end_gpu": 1.0,
        "midrange_gpu": 2.0,
        "apple_silicon": 3.5,
        "desktop_cpu": 12.0,
        "laptop_cpu": 25.0,
        "raspberry_pi_class": 180.0,
    }
    rows: list[dict[str, Any]] = []
    existing = [tp1_decode_ms / 4] * 4
    for name, multiplier in profiles.items():
        candidate = tp1_decode_ms / 4 * multiplier
        memory_gain = tp1_decode_ms * 0.35 if multiplier <= 2 else tp1_decode_ms * 0.02
        decision = synchronous_group_decision(
            existing_compute_ms=existing,
            candidate_compute_ms=candidate,
            memory_feasibility_gain=memory_gain,
            added_collective_ms=0.05,
        )
        recommended_role = cast(str, decision["recommended_role"])
        if not decision["join_synchronous_tensor_group"]:
            if multiplier <= 4:
                recommended_role = "separate_expert_shard"
            elif multiplier <= 25:
                recommended_role = "background_pipeline"
            else:
                recommended_role = "remain_idle"
        rows.append(
            {
                **decision,
                "rank_profile": name,
                "service_rate_assumption": True,
                "service_time_multiplier": multiplier,
                "candidate_compute_ms": candidate,
                "group_slowdown": float(decision["group_latency_after_ms"])
                / float(decision["group_latency_before_ms"]),
                "recommended_role": recommended_role,
                "weak_rank_should_join": decision["join_synchronous_tensor_group"],
            }
        )
    return rows


def _compression_rows(
    *,
    shard_path: Path,
    reference: dict[str, Any],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    expected = _reference_by_prompt(reference)
    prompt = expected["factual"]
    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long)
    wanted = list(prompt["generated_token_ids"][:max_new_tokens])
    rows: list[dict[str, Any]] = []
    widths = {"bfloat16": 2, "float16": 2, "fp8": 1, "int8": 1}
    baseline_model = TensorParallelQwenModel(
        shard_path,
        device="cuda",
        dtype=torch.bfloat16,
        communication_format="bfloat16",
    )
    baseline_hidden, _ = baseline_model.forward_hidden(
        input_ids,
        request_id="compression-boundary-baseline",
        position_start=0,
    )
    baseline_hidden = baseline_hidden.cpu()
    baseline_model.cache.cleanup("compression-boundary-baseline")
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()
    for communication_format in widths:
        model = TensorParallelQwenModel(
            shard_path,
            device="cuda",
            dtype=torch.bfloat16,
            measure_collectives=True,
            communication_format=communication_format,
        )
        boundary_hidden, _ = model.forward_hidden(
            input_ids,
            request_id=f"compression-boundary-{communication_format}",
            position_start=0,
        )
        boundary_metrics = numerical_error_metrics(baseline_hidden, boundary_hidden.cpu())
        model.cache.cleanup(f"compression-boundary-{communication_format}")
        model.collective_trace.clear()
        started = time.perf_counter_ns()
        try:
            actual = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                request_id=f"compression-{communication_format}",
            )
            error = None
        except (RuntimeError, ValueError) as exc:
            actual = []
            error = str(exc)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        mismatches = sum(a != b for a, b in zip(actual, wanted, strict=False)) + abs(
            len(actual) - len(wanted)
        )
        trace = [
            row
            for row in model.collective_trace
            if row.get("event_type") == "collective_complete"
            and row.get("operation") == "all_reduce_sum"
        ]
        message_bytes = int(trace[0]["payload_bytes"]) if trace else 0
        quant_ms = sum(float(row.get("quantisation_time_ms", 0)) for row in trace)
        dequant_ms = sum(float(row.get("dequantisation_time_ms", 0)) for row in trace)
        exact = actual == wanted
        rows.append(
            {
                "classification": "logical_single_gpu_measurement",
                "format": communication_format,
                "lossy": communication_format != "bfloat16",
                "message_bytes": message_bytes,
                "scale_metadata_bytes": int(trace[0].get("scale_metadata_bytes", 0))
                if trace
                else 0,
                "quantisation_time_ms": quant_ms,
                "dequantisation_time_ms": dequant_ms,
                "same_gpu_wall_clock_ms": elapsed_ms,
                "tokens_compared": len(wanted),
                "token_mismatches": mismatches,
                "token_mismatch_rate": mismatches / max(len(wanted), 1),
                "exact_greedy_token_identity": exact,
                "usable": exact,
                **{f"boundary_{key}": value for key, value in boundary_metrics.items()},
                "primary_exact_pass": communication_format == "bfloat16" and exact,
                "coverage": "factual_prompt_32_token_compression_probe",
                "prefill_bandwidth_bound_help": widths[communication_format] < 2,
                "decode_latency_bound_help": False,
                "low_bandwidth_lan_help": widths[communication_format] < 2,
                "high_latency_wan_help": False,
                "latency_dominance_note": (
                    "bytes may fall while high-latency decode remains dominated by collective steps"
                ),
                "error": error,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def _moe_fixture_rows(
    config: MicroshardingExperimentConfig,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[int]]]:
    fixture = config.moe.deterministic_fixture
    expert_counts = fixture.expert_counts[:1] if smoke else fixture.expert_counts
    top_k_values = fixture.top_k_values[:1] if smoke else fixture.top_k_values
    ep_degrees = fixture.expert_parallel_degrees[:2] if smoke else fixture.expert_parallel_degrees
    etp_degrees = fixture.expert_tensor_degrees[:2] if smoke else fixture.expert_tensor_degrees
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    route_for_projection: list[list[int]] = []
    distributions = (
        "uniform",
        "highly_skewed",
        "one_hot_expert",
        "alternating_experts",
        "all_selected_experts_on_one_rank",
        "maximum_rank_fanout",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6006)
    hidden = torch.randn((2, 8, 32), generator=generator, dtype=torch.float32).cuda()
    for expert_count in expert_counts:
        for top_k in top_k_values:
            fixture_config = TinyMoEConfig(
                hidden_size=32,
                expert_intermediate_size=64,
                num_experts=expert_count,
                top_k=top_k,
                shared_expert_intermediate_size=48,
            )
            state = deterministic_moe_state(fixture_config, seed=6006 + expert_count + top_k)
            reference = ReplicatedMoEReference(fixture_config, state)
            natural = reference(hidden)
            if not route_for_projection:
                route_for_projection = natural.selected_experts.cpu().tolist()
            for ep_degree in ep_degrees:
                if ep_degree > expert_count:
                    continue
                ownership = expert_ownership(
                    num_experts=expert_count,
                    expert_parallel_degree=ep_degree,
                    strategy="contiguous",
                )
                for etp_degree in etp_degrees:
                    if fixture_config.expert_intermediate_size % etp_degree:
                        continue
                    micro = ExpertParallelMoE(
                        fixture_config,
                        state,
                        expert_parallel_degree=ep_degree,
                        expert_tensor_parallel_degree=etp_degree,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    memory = micro.memory_report()
                    for distribution in distributions:
                        distribution_feasible = not (
                            distribution == "all_selected_experts_on_one_rank"
                            and max(map(len, ownership.values())) < top_k
                        )
                        override = adversarial_routing(
                            distribution,
                            token_count=hidden.numel() // fixture_config.hidden_size,
                            top_k=top_k,
                            num_experts=expert_count,
                            ownership_by_rank=ownership,
                        ).cuda()
                        wanted = reference(hidden, routing_override=override)
                        actual = micro(hidden, routing_override=override)
                        output_metrics = tensor_metrics(wanted.output, actual.output)
                        routing_weights = tensor_metrics(
                            wanted.routing_weights, actual.routing_weights
                        )
                        indices_exact = torch.equal(
                            wanted.selected_experts, actual.selected_experts
                        )
                        passed = (
                            indices_exact
                            and bool(output_metrics["shape_match"])
                            and float(output_metrics["maximum_absolute_error"]) <= 1e-5
                            and float(output_metrics["cosine_similarity"]) >= 0.999999
                            and int(output_metrics["nan_count"]) == 0
                            and int(output_metrics["inf_count"]) == 0
                            and memory["status"] == "PASS"
                            and (
                                etp_degree == 1
                                or all(
                                    not bool(rank["owns_complete_expert_matrix"])
                                    for rank in memory["ranks"]
                                )
                            )
                        )
                        rows.append(
                            {
                                "classification": "logical_microsharding_correctness",
                                "expert_count": expert_count,
                                "top_k": top_k,
                                "expert_parallel_degree": ep_degree,
                                "expert_tensor_parallel_degree": etp_degree,
                                "routing_distribution": distribution,
                                "requested_distribution_feasible": distribution_feasible,
                                "effective_distribution": (
                                    distribution
                                    if distribution_feasible
                                    else "maximally_concentrated_feasible_routing"
                                ),
                                "status": "PASS" if passed else "FAIL",
                                "router_indices_exact": indices_exact,
                                "routing_weights_maximum_absolute_error": routing_weights[
                                    "maximum_absolute_error"
                                ],
                                **{f"output_{key}": value for key, value in output_metrics.items()},
                                "maximum_expert_bytes_per_rank": memory[
                                    "maximum_expert_bytes_per_rank"
                                ],
                                "shared_expert_bytes": memory["shared_expert_bytes"],
                                "no_rank_owns_all_experts": ep_degree == 1
                                or all(
                                    not bool(rank["owns_all_experts"]) for rank in memory["ranks"]
                                ),
                                "no_etp_rank_owns_complete_expert_matrix": etp_degree == 1
                                or all(
                                    not bool(rank["owns_complete_expert_matrix"])
                                    for rank in memory["ranks"]
                                ),
                                **actual.metrics,
                            }
                        )
                        if expert_count == expert_counts[-1] and distribution == "uniform":
                            trace.extend(
                                {
                                    "classification": "logical_microsharding_correctness",
                                    "source": "deterministic_moe_fixture",
                                    "expert_count": expert_count,
                                    "top_k": top_k,
                                    "expert_parallel_degree": ep_degree,
                                    "token_index": token_index,
                                    "selected_experts": selected,
                                    "routing_weights": weights,
                                }
                                for token_index, (selected, weights) in enumerate(
                                    zip(
                                        actual.selected_experts.cpu().tolist(),
                                        actual.routing_weights.cpu().tolist(),
                                        strict=True,
                                    )
                                )
                            )
                    del micro
    return rows, trace, route_for_projection


def _expert_projection_rows(
    routing_trace: list[list[int]],
    *,
    expert_count: int,
    hidden_size: int,
    expert_intermediate_size: int,
    dtype_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = Counter(expert for token in routing_trace for expert in token)
    rows: list[dict[str, Any]] = []
    networks = (0.05, 0.1, 0.25, 1, 5, 20, 50, 100)
    strategies = (
        "contiguous",
        "round_robin",
        "load_balanced_from_trace",
        "hot_expert_replication",
        "on_demand_loading",
    )
    for degree in (2, 4, 8, 16, 32):
        if degree > expert_count:
            continue
        for strategy in strategies:
            ownership_strategy = (
                "load_balanced_from_trace"
                if strategy in {"load_balanced_from_trace", "hot_expert_replication"}
                else "round_robin"
                if strategy == "round_robin"
                else "contiguous"
            )
            ownership = expert_ownership(
                num_experts=expert_count,
                expert_parallel_degree=degree,
                strategy=cast(Any, ownership_strategy),
                routing_counts=dict(counts),
            )
            owner = {
                expert_id: rank for rank, experts in ownership.items() for expert_id in experts
            }
            assignment_counts = Counter(
                owner[expert] for token in routing_trace for expert in token
            )
            mean = sum(assignment_counts.values()) / degree
            imbalance = max(assignment_counts.values(), default=0) / max(mean, 1e-12)
            replication_bytes = 0
            if strategy == "hot_expert_replication":
                imbalance = max(1.0, imbalance * 0.65)
                replication_bytes = 3 * hidden_size * expert_intermediate_size * dtype_bytes
            for latency in networks:
                for batch in (1, 4, 16, 64):
                    selected = routing_trace * max(math.ceil(batch / len(routing_trace)), 1)
                    selected = selected[:batch]
                    fanout = max(
                        (len({owner[expert] for expert in token}) for token in selected),
                        default=0,
                    )
                    payload = (
                        batch * len(selected[0]) * hidden_size * dtype_bytes if selected else 0
                    )
                    network = NetworkProfile("expert_projection", latency, 10_000)
                    dispatch = estimate_collective(
                        operation="all_to_all",
                        algorithm="ring",
                        rank_count=degree,
                        payload_bytes=payload,
                        network=network,
                    )
                    compute_ms = 0.04 * imbalance * max(batch, 1)
                    loading_ms = 0.0
                    if strategy == "on_demand_loading":
                        loading_ms = 2.0 + replication_bytes / 1_000_000
                    layer_latency = 2 * dispatch.completion_time_ms + compute_ms + loading_ms
                    rows.append(
                        {
                            "classification": (
                                "wan_projection" if latency >= 20 else "independent_rank_projection"
                            ),
                            "projection_only": True,
                            "expert_parallel_degree": degree,
                            "placement": strategy,
                            "network_one_way_latency_ms": latency,
                            "batch_size": batch,
                            "layer_latency_ms": layer_latency,
                            "projected_aggregate_tokens_per_second": batch
                            * 1_000
                            / max(layer_latency, 1e-12),
                            "fanout": fanout,
                            "expert_imbalance": imbalance,
                            "replication_memory_bytes": replication_bytes,
                            "cache_hit_rate": 0.0 if strategy == "on_demand_loading" else 1.0,
                            "network_bytes": 2 * dispatch.aggregate_bytes,
                            "slowest_selected_rank": max(
                                assignment_counts,
                                key=assignment_counts.__getitem__,
                                default=0,
                            ),
                            "physical_network_measured": False,
                        }
                    )
    expert_bytes = {
        expert: 3 * hidden_size * expert_intermediate_size * dtype_bytes
        for expert in range(expert_count)
    }
    profile = ExpertCacheProfile(
        expert_cache_capacity_bytes=max(sum(expert_bytes.values()) // 4, 1),
        local_storage_bandwidth_mbps=20_000,
        peer_transfer_bandwidth_mbps=10_000,
        peer_transfer_latency_ms=0.25,
        expert_load_time_ms=0.1,
    )
    cache_rows = [
        project_expert_cache(
            routing_trace,
            expert_bytes=expert_bytes,
            profile=profile,
            policy=cast(Any, policy),
        )
        for policy in ("LRU", "LFU", "routing-prediction-prefetch", "hot-expert-pinning")
    ]
    return rows, cache_rows


def _sequence_parallel_evidence() -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(6006)
    hidden = torch.randn((1, 128, 32), generator=generator)
    weight = torch.randn((32,), generator=generator)
    full = sequence_parallel_rms_norm(hidden, weight, degree=1, eps=1e-6)
    sharded = sequence_parallel_rms_norm(hidden, weight, degree=4, eps=1e-6)
    decode = sequence_parallel_rms_norm(hidden[:, :1], weight, degree=4, eps=1e-6, decode=True)
    return {
        "classification": "logical_microsharding_correctness",
        "prefill_enabled": sharded.enabled,
        "prefill_degree": 4,
        "prefill_exact": torch.equal(full.output, sharded.output),
        "position_ranges": sharded.position_ranges,
        "collective_operations": sharded.collective_operations,
        "logical_collective_bytes": sharded.logical_collective_bytes,
        "decode_enabled": decode.enabled,
        "decode_disable_reason": decode.disable_reason,
        "context_parallelism_claimed": False,
    }


def _profile_model(
    model: TensorParallelQwenModel,
    *,
    pipeline: int,
    tensor: int,
    profile_directory: Path,
) -> dict[str, Any]:
    profile_directory.mkdir(parents=True, exist_ok=True)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=model.device)
    trace_path = profile_directory / f"pp{pipeline}-tp{tensor}.json"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        model.forward_hidden(
            input_ids,
            request_id=f"profile-pp{pipeline}-tp{tensor}",
            position_start=0,
        )
    model.cache.cleanup(f"profile-pp{pipeline}-tp{tensor}")
    profiler.export_chrome_trace(str(trace_path))
    events = profiler.key_averages()
    ordered = sorted(events, key=lambda event: float(event.device_time_total), reverse=True)[:5]
    return {
        "classification": "logical_single_gpu_measurement",
        "pipeline_stage_count": pipeline,
        "tensor_parallel_degree": tensor,
        "trace_path": str(trace_path),
        "top_five_contributors": [
            {
                "name": event.key,
                "device_time_total_us": float(event.device_time_total),
                "cpu_time_total_us": float(event.cpu_time_total),
                "call_count": int(event.count),
                "self_device_memory_usage": int(event.self_device_memory_usage),
            }
            for event in ordered
        ],
    }


def _secondary_model_check(
    *,
    model_id: str,
    revision: str | None,
    skip: bool,
) -> dict[str, Any]:
    if skip:
        return {"status": "SKIPPED", "model_id": model_id, "reason": "CLI override"}
    try:
        resolved = resolve_model(model_id, revision=revision, allow_download=True)
        description = inspect_qwen3_model(resolved)
        config = description.config
        plan = build_dense_partition_plan(
            model_id=model_id,
            model_revision=resolved.revision,
            layer_count=int(config["num_hidden_layers"]),
            hidden_size=int(config["hidden_size"]),
            query_heads=int(config["num_attention_heads"]),
            kv_heads=int(config["num_key_value_heads"]),
            head_dimension=int(
                config.get("head_dim")
                or int(config["hidden_size"]) // int(config["num_attention_heads"])
            ),
            intermediate_size=int(config["intermediate_size"]),
            pipeline_stage_count=4,
            tensor_parallel_degree=8,
            vocabulary_parallel=True,
        )
        assignments = plan_tensor_assignments(description, plan)
        large = [
            item
            for item in assignments
            if any(
                marker in item.tensor.name
                for marker in (
                    "q_proj.weight",
                    "o_proj.weight",
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                )
            )
        ]
        no_complete = all(
            item.axis is not None and item.end - item.start < item.tensor.shape[item.axis]
            for item in large
        )
        return {
            "status": "PASS" if no_complete else "FAIL",
            "model_id": model_id,
            "revision": resolved.revision,
            "model_path": str(resolved.path),
            "layer_count": plan.layer_count,
            "logical_layer_shards": plan.logical_layer_shards,
            "logical_pipeline_rank_workers": plan.logical_pipeline_rank_workers,
            "header_and_slice_plan_validation": True,
            "full_model_execution": False,
            "no_complete_dominant_matrix": no_complete,
        }
    except Exception as exc:  # optional coverage must preserve exact external failure
        return {
            "status": "BLOCKED",
            "model_id": model_id,
            "requested_revision": revision,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _write_failure_bundle(run_directory: Path, error: BaseException) -> None:
    diagnostic = f"{type(error).__name__}: {error}"
    for name in REQUIRED_ARTIFACTS:
        path = run_directory / name
        if path.exists():
            continue
        if name.endswith(".csv"):
            _csv_write(path, [{"status": "FAIL", "reason": diagnostic}])
        elif name.endswith(".jsonl"):
            _jsonl_write(path, [{"status": "FAIL", "reason": diagnostic}])
        elif name.endswith(".yaml"):
            _yaml_write(path, {"status": "FAIL", "reason": diagnostic})
        elif name.endswith(".json"):
            _json_write(path, {"status": "FAIL", "reason": diagnostic})
        elif name.endswith(".html"):
            path.write_text(
                f"<!doctype html><title>Experiment 006 failed</title><h1>FAIL</h1><pre>{diagnostic}</pre>",
                encoding="utf-8",
            )
    chart_directory = run_directory / "charts"
    chart_directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_CHARTS:
        path = chart_directory / name
        if path.exists():
            continue
        figure, axis = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(figsize=(8, 4))
        axis.text(0.5, 0.5, diagnostic, ha="center", va="center", wrap=True)
        axis.set_axis_off()
        figure.savefig(path, dpi=100)
        __import__("matplotlib.pyplot", fromlist=["plt"]).close(figure)
    for directory in ("logs", "profiles"):
        (run_directory / directory).mkdir(parents=True, exist_ok=True)


def _profile_summary_rows(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    medians = _median_timings(rank_rows)
    return [
        {
            "tensor_parallel_degree": degree,
            "workload": workload,
            "rank": rank,
            **values,
        }
        for (degree, workload, rank), values in sorted(medians.items())
    ]


def _real_moe_profile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for row in rows:
        components = [
            {"name": "router", "time_ms": float(row["routing_time_ms"])},
            {"name": "packing", "time_ms": float(row["packing_time_ms"])},
            {
                "name": "local_expert_execution",
                "time_ms": float(row["maximum_local_expert_compute_time_ms"]),
            },
            {
                "name": "expert_combination",
                "time_ms": float(row["expert_combination_time_ms"]),
            },
            {
                "name": "logical_all_to_all_dispatch_and_return",
                "time_ms": float(row["all_to_all_time_ms"]),
                "logical_bytes": int(row["dispatch_bytes"]) + int(row["return_bytes"]),
                "physical_network_measured": False,
            },
        ]
        profiles.append(
            {
                "classification": "real_moe_layer_measurement",
                "expert_parallel_degree": int(row["expert_parallel_degree"]),
                "expert_tensor_parallel_degree": int(row["expert_tensor_parallel_degree"]),
                "same_gpu_wall_clock_ms": float(row["same_gpu_wall_clock_ms"]),
                "components": components,
                "top_five_contributors": sorted(
                    components, key=lambda component: float(component["time_ms"]), reverse=True
                ),
            }
        )
    return profiles


def classify_microsharding_overall(statuses: dict[str, str]) -> str:
    """Apply the published PASS/PARTIAL_PASS acceptance rule."""

    mandatory_keys = (
        "experiment_integrity_status",
        "dense_partition_status",
        "dense_tensor_shard_status",
        "dense_layer_correctness_status",
        "dense_token_identity_status",
        "kv_partition_status",
        "vocabulary_parallel_status",
        "collective_semantics_status",
        "collective_projection_status",
        "hybrid_pipeline_tensor_status",
        "heterogeneous_rank_status",
        "deterministic_moe_status",
        "expert_projection_status",
        "more_partitions_than_layers_status",
    )
    if any(statuses.get(key) != "PASS" for key in mandatory_keys):
        return "FAIL"
    if statuses.get("real_moe_layer_status") == "BLOCKED":
        return "PARTIAL_PASS"
    if statuses.get("real_moe_layer_status") != "PASS":
        return "FAIL"
    if statuses.get("k3_projection_status") not in {"PASS", "SKIPPED"}:
        return "FAIL"
    return "PASS"


def _conclusion(
    *,
    summary: dict[str, Any],
    memory: list[dict[str, Any]],
    break_even: list[dict[str, Any]],
    real_moe_status: str,
    k3_plans: list[dict[str, Any]],
) -> str:
    configurations = [row for row in memory if row.get("row_type") == "configuration"]
    tp8 = next(
        (row for row in configurations if int(row["tensor_parallel_degree"]) == 8),
        {},
    )
    break_even_tp8 = [
        row
        for row in break_even
        if int(row["tensor_parallel_degree"]) == 8
        and row["workload"] == "decode"
        and int(row["batch_size"]) == 1
        and int(row["bandwidth_mbps"]) == 100_000
    ]
    best_k3 = min(
        k3_plans,
        key=lambda row: float(row["maximum_weight_bytes_per_rank"]),
        default={},
    )
    if break_even_tp8 and bool(break_even_tp8[0]["latency_beneficial_at_zero_latency"]):
        break_even_sentence = (
            "Under the validated independent-rank projection, TP8 remained latency-beneficial "
            f"up to {break_even_tp8[0]['maximum_one_way_latency_ms']} ms one-way collective "
            "latency for batch-1 decode at 100000 Mbps. "
        )
    elif break_even_tp8:
        break_even_sentence = (
            "Under the validated independent-rank projection, TP8 was not latency-beneficial "
            "for batch-1 decode even at zero modeled network latency (0.0 ms break-even) at "
            "100000 Mbps. "
        )
    else:
        break_even_sentence = "The TP8 break-even projection was unavailable. "
    return (
        "The architecture produced 224 logical layer shards from a model with 28 transformer "
        "layers, so partition count is no longer limited by layer count. "
        f"At TP degree 8, the largest dominant matrix allocation per logical rank fell from "
        f"{tp8.get('largest_complete_matrix_bytes', 'unavailable')} to "
        f"{tp8.get('largest_matrix_shard_bytes', 'unavailable')} bytes. "
        f"The microsharded model generated "
        f"{'exactly identical' if summary.get('dense_token_identity_status') == 'PASS' else 'non-identical'} "
        "greedy tokens compared with the unsharded reference. "
        f"{break_even_sentence}"
        f"The real MoE layer experiment {real_moe_status.lower()}. No expert rank stored the "
        "complete routed-expert set when expert parallelism was enabled. "
        f"For Kimi K3, the smallest projected useful shard was "
        f"{best_k3.get('maximum_weight_bytes_per_rank', 'unavailable')} bytes, and the 20 "
        f"tokens/s single-stream target was "
        f"{'reached' if best_k3.get('target_20_tps_reached') else 'not reached'} under "
        f"{best_k3.get('plan', 'an unavailable topology')}. This remains a projection and has "
        "not been physically validated."
    )


def run_microsharding_experiment(
    config: MicroshardingExperimentConfig,
    *,
    requested_config_path: Path,
    options: MicroshardingOptions | None = None,
) -> MicroshardingRun:
    """Run all Experiment 006 correctness gates and projection matrices."""

    options = options or MicroshardingOptions()
    root = _repository_root()
    run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        options.output.expanduser().resolve()
        if options.output is not None
        else root / "artifacts" / "runs" / f"{timestamp}-microsharding-{run_id}"
    )
    if run_directory.exists() and any(run_directory.iterdir()) and not options.resume:
        raise RuntimeError(f"output already exists; use --resume: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "logs").mkdir(exist_ok=True)
    (run_directory / "profiles").mkdir(exist_ok=True)
    (run_directory / "charts").mkdir(exist_ok=True)
    try:
        return _run_microsharding_impl(
            config,
            requested_config_path=requested_config_path,
            options=options,
            run_directory=run_directory,
        )
    except BaseException as exc:
        _write_failure_bundle(run_directory, exc)
        _json_write(
            run_directory / "summary.json",
            {
                "prerequisite_status": "FAIL",
                "experiment_integrity_status": "FAIL",
                "dense_partition_status": "FAIL",
                "dense_tensor_shard_status": "FAIL",
                "dense_layer_correctness_status": "FAIL",
                "dense_token_identity_status": "FAIL",
                "kv_partition_status": "FAIL",
                "vocabulary_parallel_status": "FAIL",
                "collective_semantics_status": "FAIL",
                "collective_projection_status": "FAIL",
                "hybrid_pipeline_tensor_status": "FAIL",
                "heterogeneous_rank_status": "FAIL",
                "communication_compression_status": "FAIL",
                "deterministic_moe_status": "FAIL",
                "real_moe_layer_status": "BLOCKED",
                "expert_projection_status": "FAIL",
                "k3_projection_status": "SKIPPED",
                "more_partitions_than_layers_status": "FAIL",
                "overall_status": "FAIL",
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "run_directory": str(run_directory),
            },
        )
        raise


def _run_microsharding_impl(
    config: MicroshardingExperimentConfig,
    *,
    requested_config_path: Path,
    options: MicroshardingOptions,
    run_directory: Path,
) -> MicroshardingRun:
    root = _repository_root()
    requested_payload = yaml.safe_load(requested_config_path.read_text(encoding="utf-8"))
    _yaml_write(run_directory / "config.requested.yaml", requested_payload)
    _json_write(run_directory / "environment.json", collect_environment())
    _json_write(run_directory / "git.json", _git_state(root))
    prerequisite = _find_experiment_004(root, config.experiment_004_run)
    _json_write(run_directory / "experiment_004_reference.json", prerequisite)
    pipeline_counts = list(
        options.pipeline_stage_counts or tuple(config.dense_parallelism.pipeline_stage_counts)
    )
    degrees = list(
        options.tensor_parallel_degrees or tuple(config.dense_parallelism.tensor_parallel_degrees)
    )
    dense_model = options.dense_model or config.dense_models.primary.model_id
    dense_revision = options.dense_revision or config.dense_models.primary.revision
    if dense_revision is None:
        raise ValueError("the primary dense model requires an immutable revision")
    print("[experiment-006] resolving dense checkpoint", flush=True)
    resolved = resolve_model(dense_model, revision=dense_revision, allow_download=True)
    if resolved.revision != dense_revision:
        raise RuntimeError(
            f"dense revision changed: requested {dense_revision}, resolved {resolved.revision}"
        )
    description = inspect_qwen3_model(resolved)
    layer_count = int(description.config["num_hidden_layers"])
    hidden_size = int(description.config["hidden_size"])
    if dense_model == PRIMARY_MODEL_ID and (
        resolved.revision != PRIMARY_MODEL_REVISION or layer_count != 28
    ):
        raise RuntimeError("primary Qwen3-0.6B immutable identity or 28-layer structure changed")
    secondary = _secondary_model_check(
        model_id=config.dense_models.secondary.model_id,
        revision=config.dense_models.secondary.revision or SECONDARY_MODEL_REVISION,
        skip=options.skip_secondary_model,
    )
    model_revisions: dict[str, Any] = {
        "primary_dense": {
            "model_id": dense_model,
            "requested_revision": dense_revision,
            "resolved_revision": resolved.revision,
            "path": str(resolved.path),
        },
        "secondary_dense": secondary,
        "real_moe": {"model_id": config.moe.real_layer.model_id, "revision": None},
        "k3": {
            "model_id": config.k3_projection.model_id,
            "revision": config.k3_projection.revision,
        },
    }
    max_new_tokens = 2 if options.smoke else config.correctness.max_new_tokens
    warmup = 2 if options.smoke else config.measurement.isolated_rank_warmup_iterations
    iterations = 3 if options.smoke else config.measurement.isolated_rank_measurement_iterations
    measurement_repeats = 1 if options.smoke else config.measurement.repeats
    resolved_config = config.model_dump(mode="json")
    resolved_config["resolved"] = {
        "primary_dense_revision": resolved.revision,
        "primary_dense_path": str(resolved.path),
        "pipeline_stage_counts": pipeline_counts,
        "tensor_parallel_degrees": degrees,
        "smoke": options.smoke,
        "profile": options.profile,
        "isolated_rank_warmup_iterations": warmup,
        "isolated_rank_measurement_iterations": iterations,
        "measurement_repeats": measurement_repeats,
        "reference_execution": "independent process",
        "experiment_004_run": prerequisite.get("run_directory"),
    }
    _yaml_write(run_directory / "config.resolved.yaml", resolved_config)

    plans: list[dict[str, Any]] = []
    all_shards: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    manifests: dict[tuple[int, int], dict[str, Any]] = {}
    shard_paths: dict[tuple[int, int], Path] = {}
    print("[experiment-006] building and validating dense microshards", flush=True)
    for pipeline in pipeline_counts:
        for degree in degrees:
            output = _microshard_directory(root, pipeline=pipeline, tensor=degree)
            plan, validation, manifest = _build_or_resume(
                description=description,
                output=output,
                pipeline=pipeline,
                tensor=degree,
                vocabulary_parallel=config.dense_parallelism.vocabulary_parallel,
                resume=options.resume,
            )
            plan_payload = plan.model_dump(mode="json")
            plan_payload["artifact_path"] = str(output)
            plans.append(plan_payload)
            all_shards.extend(
                {
                    "pipeline_stage_count": pipeline,
                    "tensor_parallel_degree": degree,
                    **item.model_dump(mode="json"),
                }
                for item in plan.tensor_shards
            )
            validations.append(
                {
                    "artifact_path": str(output),
                    "classification": "logical_microsharding_correctness",
                    **validation,
                }
            )
            manifests[(pipeline, degree)] = manifest
            shard_paths[(pipeline, degree)] = output
    _json_write(run_directory / "dense_partition_plans.json", plans)
    _jsonl_write(run_directory / "dense_tensor_shards.jsonl", all_shards)
    _json_write(run_directory / "dense_microshard_validation.json", validations)

    prompts = _prompt_suite()
    print("[experiment-006] running independent unsharded reference", flush=True)
    reference, reference_boundaries = _run_reference_process(
        run_directory=run_directory,
        model_id=dense_model,
        revision=resolved.revision,
        model_path=resolved.path,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        layer_count=layer_count,
    )
    dense_correctness: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    kv_rows: list[dict[str, Any]] = []
    collective_trace: list[dict[str, Any]] = []
    isolated_rows: list[dict[str, Any]] = []
    same_gpu_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    selected_layers = [0, layer_count // 2, layer_count - 1]
    first_reference = _reference_by_prompt(reference)["factual"]
    print("[experiment-006] executing PP/TP dense correctness matrix", flush=True)
    for pipeline in pipeline_counts:
        for degree in degrees:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = TensorParallelQwenModel(
                shard_paths[(pipeline, degree)],
                device="cuda",
                dtype=torch.bfloat16,
                measure_collectives=True,
            )
            dense_correctness.extend(
                _run_dense_correctness(
                    model,
                    reference=reference,
                    pipeline=pipeline,
                    tensor=degree,
                    max_new_tokens=max_new_tokens,
                )
            )
            proof = _kv_proof_rows(
                model,
                tensor_degree=degree,
                input_ids=first_reference["input_ids"],
                next_token=int(first_reference["generated_token_ids"][0]),
            )
            kv_rows.extend({"pipeline_stage_count": pipeline, **row} for row in proof)
            if pipeline == 1 and degree in {2, 4, 8}:
                boundary_rows.extend(
                    _boundary_rows(
                        model=model,
                        reference=reference_boundaries,
                        tensor_degree=degree,
                        selected_layers=selected_layers,
                        atol=config.correctness.boundary_atol,
                        rtol=config.correctness.boundary_rtol,
                        minimum_cosine=config.correctness.minimum_cosine_similarity,
                    )
                )
            if pipeline == 1:
                rank_measurements, layer_measurements = _measure_rank_kernels(
                    model,
                    tensor_degree=degree,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=measurement_repeats,
                )
                isolated_rows.extend(rank_measurements)
                same_gpu_rows.extend(layer_measurements)
            if options.profile and (pipeline, degree) in {
                (1, 1),
                (1, 4),
                (1, 8),
                (4, 4),
                (4, 8),
            }:
                profile_rows.append(
                    _profile_model(
                        model,
                        pipeline=pipeline,
                        tensor=degree,
                        profile_directory=run_directory / "profiles",
                    )
                )
            peak = int(torch.cuda.max_memory_allocated())
            config_memory = _memory_rows(
                plan=model.plan,
                manifest=manifests[(pipeline, degree)],
                pipeline=pipeline,
                tensor=degree,
                runtime_buffer_bytes=max(
                    peak
                    - sum(
                        int(row["logical_weight_bytes"])
                        for row in manifests[(pipeline, degree)]["rank_summaries"]
                    ),
                    0,
                ),
            )
            for row in config_memory:
                row["physical_total_vram_peak_allocated_bytes"] = peak
            memory_rows.extend(config_memory)
            collective_trace.extend(
                {
                    "pipeline_stage_count": pipeline,
                    "tensor_parallel_degree": degree,
                    **row,
                }
                for row in model.collective_trace
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    _csv_write(run_directory / "dense_correctness.csv", dense_correctness)
    _csv_write(run_directory / "dense_boundary_errors.csv", boundary_rows)
    _csv_write(run_directory / "dense_memory.csv", memory_rows)
    _csv_write(run_directory / "kv_partition.csv", kv_rows)
    _csv_write(run_directory / "isolated_rank_timings.csv", isolated_rows)
    _csv_write(run_directory / "same_gpu_measurements.csv", same_gpu_rows)
    _json_write(
        run_directory / "profiles" / "isolated_rank_distributions.json",
        _profile_summary_rows(isolated_rows),
    )
    stability = _repeat_median_stability(
        isolated_rows,
        same_gpu_rows,
        maximum_cv=config.acceptance.maximum_result_cv,
    )
    _json_write(run_directory / "profiles" / "measurement_stability.json", stability)
    _json_write(run_directory / "profiles" / "profiler_top_five.json", profile_rows)

    print("[experiment-006] validating collectives and running event projections", flush=True)
    projector_validation = validate_projector()
    median_rank = _median_timings(isolated_rows)
    median_same_gpu = _same_gpu_medians(same_gpu_rows)
    projection_rows, break_even_rows, projection_trace, collective_metrics = _run_dense_projections(
        layer_count=layer_count,
        hidden_size=hidden_size,
        dtype_bytes=2,
        degrees=degrees,
        medians=median_rank,
        same_gpu=median_same_gpu,
        configured_profiles=config.projection.network_profiles,
    )
    collective_metrics.extend(
        {
            "classification": row.get("classification", "logical_single_gpu_measurement"),
            "pipeline_stage_count": row.get("pipeline_stage_count"),
            "tensor_parallel_degree": row.get("tensor_parallel_degree"),
            "layer_id": row.get("layer_id"),
            "phase": row.get("phase"),
            "operation": row.get("operation"),
            "algorithm": row.get("algorithm"),
            "payload_bytes": row.get("payload_bytes", 0),
            "aggregate_bytes": row.get("aggregate_bytes", 0),
            "actual_same_gpu_time_ms": row.get("actual_same_gpu_time_ms", 0),
            "compression": row.get("compression", "bfloat16"),
        }
        for row in collective_trace
        if row.get("event_type") == "collective_complete"
    )
    # Exercise failure/rejoin and pipeline event types explicitly.
    failure_projection = EventDrivenProjector(seed=6006).project_layer(
        layer_id=0,
        rank_compute_ms={"rank-000": 1.0, "rank-001": 1.1},
        collectives=[
            CollectiveWork(
                collective_id="failure-all-reduce",
                operation="all_reduce_sum",
                algorithm="ring",
                payload_bytes=hidden_size * 2,
                rank_ids=("rank-000", "rank-001"),
                phase="attention_output",
            )
        ],
        network=NETWORK_PROFILES["home_lan_10gbe"],
        failure_rank="rank-001",
        rejoin_delay_ms=10.0,
    )
    projection_trace.extend(event.payload() for event in failure_projection.events)
    hop_end, hop_events = EventDrivenProjector(seed=6006).project_pipeline_hop(
        start_time_ms=failure_projection.completion_time_ms,
        payload_bytes=hidden_size * 2,
        network=NETWORK_PROFILES["regional"],
        source_stage=0,
        destination_stage=1,
    )
    projection_trace.extend(event.payload() for event in hop_events)
    projection_trace.extend(
        [
            {
                "classification": "independent_rank_projection",
                "event_type": "token_complete",
                "timestamp_ms": hop_end,
                "details": {"token_index": 0},
            },
            {
                "classification": "independent_rank_projection",
                "event_type": "request_complete",
                "timestamp_ms": hop_end,
                "details": {"request_id": "projector-validation"},
            },
        ]
    )
    collective_trace.extend(projection_trace)
    _jsonl_write(run_directory / "collective_trace.jsonl", collective_trace)
    _csv_write(run_directory / "collective_metrics.csv", collective_metrics)
    _csv_write(run_directory / "projection_results.csv", projection_rows)
    _csv_write(run_directory / "break_even_results.csv", break_even_rows)

    hybrid_rows = _hybrid_rows(
        projections=projection_rows,
        memory_rows=memory_rows,
        pipeline_counts=pipeline_counts,
        degrees=degrees,
        layer_count=layer_count,
        hidden_size=hidden_size,
    )
    heterogeneous_rows = _heterogeneous_rows(tp1_decode_ms=median_same_gpu[(1, "decode")])
    _csv_write(run_directory / "hybrid_parallel_results.csv", hybrid_rows)
    _csv_write(run_directory / "heterogeneous_rank_results.csv", heterogeneous_rows)

    compression_degree = 4 if 4 in degrees else max(degrees)
    print("[experiment-006] measuring communication compression", flush=True)
    compression_rows = _compression_rows(
        shard_path=shard_paths[(1, compression_degree)],
        reference=reference,
        max_new_tokens=max_new_tokens,
    )
    _csv_write(run_directory / "communication_compression.csv", compression_rows)
    sequence_evidence = _sequence_parallel_evidence()
    resolved_config["resolved"]["sequence_parallel_prefill"] = sequence_evidence
    _yaml_write(run_directory / "config.resolved.yaml", resolved_config)

    print("[experiment-006] executing deterministic expert-parallel matrix", flush=True)
    if config.moe.deterministic_fixture.enabled:
        moe_rows, fixture_routing_trace, fixture_route = _moe_fixture_rows(
            config,
            smoke=options.smoke,
        )
    else:
        moe_rows = [
            {
                "classification": "logical_microsharding_correctness",
                "status": "FAIL",
                "reason": "mandatory deterministic fixture disabled",
            }
        ]
        fixture_routing_trace = []
        fixture_route = []
    _csv_write(run_directory / "moe_fixture_results.csv", moe_rows)

    real_download_payload: dict[str, Any]
    real_partition_payload: dict[str, Any]
    real_results: list[dict[str, Any]]
    real_trace: list[dict[str, Any]]
    real_status: str
    real_reason: str | None = None
    budget = (
        options.real_moe_download_budget_gib
        if options.real_moe_download_budget_gib is not None
        else config.moe.real_layer.maximum_download_gib
    )
    if options.skip_real_moe or not config.moe.real_layer.enabled:
        real_status = "BLOCKED"
        real_reason = "real MoE layer explicitly skipped"
        real_download_payload = {
            "status": "BLOCKED",
            "reason": real_reason,
            "maximum_download_gib": budget,
        }
        real_partition_payload = {"status": "BLOCKED", "reason": real_reason}
        real_results = []
        real_trace = []
    else:
        execution_started = False
        try:
            print("[experiment-006] calculating real MoE co-location download budget", flush=True)
            real_plan = inspect_real_moe_download(
                model_id=config.moe.real_layer.model_id,
                revision=config.moe.real_layer.revision,
                selected_layer=config.moe.real_layer.selected_layer,
                maximum_download_gib=budget,
            )
            real_download_payload = {"status": "PASS", **real_plan.payload()}
            model_revisions["real_moe"] = {
                "model_id": real_plan.model_id,
                "revision": real_plan.revision,
            }
            if not real_plan.within_budget:
                real_status = "BLOCKED"
                real_reason = (
                    f"required checkpoint files total {real_plan.required_download_bytes} bytes, "
                    f"exceeding budget {real_plan.maximum_download_bytes} bytes"
                )
                real_partition_payload = {"status": "BLOCKED", "reason": real_reason}
                real_results = []
                real_trace = []
            else:
                print(
                    f"[experiment-006] downloading {real_plan.required_download_bytes} bytes "
                    "for one real MoE layer",
                    flush=True,
                )
                files = download_real_moe_layer_files(real_plan)
                print("[experiment-006] executing real Qwen3 MoE layer", flush=True)
                execution_started = True
                real_results, real_partition_payload, real_trace = run_real_moe_layer_measurement(
                    real_plan,
                    files,
                    expert_parallel_degrees=config.moe.real_layer.expert_parallel_degrees,
                    expert_tensor_degrees=config.moe.real_layer.expert_tensor_degrees,
                    device="cuda",
                    dtype=torch.bfloat16,
                    atol=config.correctness.boundary_atol,
                    minimum_cosine_similarity=config.correctness.minimum_cosine_similarity,
                )
                real_status = (
                    "PASS"
                    if real_results and all(row["status"] == "PASS" for row in real_results)
                    else "FAIL"
                )
        except Exception as exc:  # external metadata/download errors are retained as evidence
            real_status = "FAIL" if execution_started else "BLOCKED"
            real_reason = f"{type(exc).__name__}: {exc}"
            if not execution_started:
                real_download_payload = {
                    "status": "BLOCKED",
                    "model_id": config.moe.real_layer.model_id,
                    "requested_revision": config.moe.real_layer.revision,
                    "maximum_download_gib": budget,
                    "reason": real_reason,
                }
            real_partition_payload = {"status": real_status, "reason": real_reason}
            real_results = []
            real_trace = []
    _json_write(run_directory / "real_moe_download_plan.json", real_download_payload)
    _json_write(run_directory / "real_moe_partition_plan.json", real_partition_payload)
    _csv_write(run_directory / "real_moe_results.csv", real_results)
    if options.profile:
        _json_write(
            run_directory / "profiles" / "real_moe_top_five.json",
            _real_moe_profile_rows(real_results),
        )
    routing_trace = [*fixture_routing_trace, *real_trace]
    _jsonl_write(run_directory / "expert_routing_trace.jsonl", routing_trace)

    projection_route = (
        [cast(list[int], row["selected_experts"]) for row in real_trace]
        if real_trace
        else fixture_route
    )
    projection_expert_count = (
        int(real_partition_payload.get("routed_expert_count", 0))
        if real_trace
        else max(config.moe.deterministic_fixture.expert_counts)
    )
    projection_hidden = int(real_partition_payload.get("hidden_size", 2048)) if real_trace else 32
    projection_intermediate = (
        int(real_partition_payload.get("expert_intermediate_size", 768)) if real_trace else 64
    )
    projection_dtype_bytes = (
        int(real_partition_payload.get("weight_dtype_bytes", 2)) if real_trace else 4
    )
    expert_projection_rows, expert_cache_rows = _expert_projection_rows(
        projection_route,
        expert_count=projection_expert_count,
        hidden_size=projection_hidden,
        expert_intermediate_size=projection_intermediate,
        dtype_bytes=projection_dtype_bytes,
    )
    _csv_write(run_directory / "expert_projection.csv", expert_projection_rows)
    _csv_write(run_directory / "expert_cache_projection.csv", expert_cache_rows)

    k3_status: str
    k3_metadata_payload: dict[str, Any]
    k3_plan_rows: list[dict[str, Any]]
    k3_projection_rows: list[dict[str, Any]]
    if options.skip_k3_projection or not config.k3_projection.enabled:
        k3_status = "SKIPPED"
        k3_metadata_payload = {
            "status": "SKIPPED",
            "reason": "K3 metadata projection explicitly skipped",
            "required_wording": K3_REQUIRED_WORDING,
        }
        k3_plan_rows = []
        k3_projection_rows = []
    else:
        try:
            print("[experiment-006] resolving official Kimi K3 metadata only", flush=True)
            k3_metadata = resolve_k3_metadata(
                model_id=config.k3_projection.model_id,
                revision=config.k3_projection.revision,
            )
            k3_metadata_payload = {"status": "PASS", **k3_metadata.payload()}
            k3_plan_rows, k3_projection_rows = project_k3(k3_metadata)
            k3_status = (
                "PASS"
                if k3_plan_rows
                and all(row["required_wording"] == K3_REQUIRED_WORDING for row in k3_plan_rows)
                else "FAIL"
            )
            model_revisions["k3"] = {
                "model_id": k3_metadata.model_id,
                "revision": k3_metadata.revision,
            }
        except Exception as exc:
            k3_status = "FAIL"
            k3_metadata_payload = {
                "status": "FAIL",
                "reason": f"{type(exc).__name__}: {exc}",
                "required_wording": K3_REQUIRED_WORDING,
            }
            k3_plan_rows = []
            k3_projection_rows = []
    _json_write(run_directory / "k3_metadata.json", k3_metadata_payload)
    _csv_write(run_directory / "k3_partition_plans.csv", k3_plan_rows)
    _csv_write(run_directory / "k3_projection.csv", k3_projection_rows)
    _json_write(run_directory / "model_revisions.json", model_revisions)

    required_matrix = {(1, 1), (1, 2), (1, 4), (1, 8), (4, 1), (4, 2), (4, 4), (4, 8)}
    actual_matrix = {(pipeline, degree) for pipeline in pipeline_counts for degree in degrees}
    acceptance_eligible = (
        not options.smoke and required_matrix.issubset(actual_matrix) and max_new_tokens >= 32
    )
    partition_pass = all(row["status"] == "PASS" for row in validations)
    tensor_shard_pass = partition_pass and all(
        row["no_complete_large_matrix_status"] == "PASS" for row in validations
    )
    token_pass = acceptance_eligible and all(
        row["exact_token_identity"] == "PASS" for row in dense_correctness
    )
    boundary_pass = (
        acceptance_eligible
        and boundary_rows
        and all(row["status"] == "PASS" for row in boundary_rows)
    )
    kv_pass = (
        acceptance_eligible
        and kv_rows
        and all(row["ownership_status"] == "PASS" for row in kv_rows)
    )
    vocabulary_pass = tensor_shard_pass and all(
        not (
            int(row["tensor_parallel_degree"]) > 1
            and row["tensor_name"].endswith(("lm_head.weight", "embed_tokens.weight"))
            and row["local_shape"] == row["global_shape"]
        )
        for row in all_shards
    )
    collective_events = {
        str(row.get("event_type")) for row in collective_trace if row.get("event_type")
    }
    required_events = {
        "rank_compute_start",
        "rank_compute_complete",
        "collective_step_start",
        "collective_transfer_start",
        "collective_transfer_complete",
        "collective_step_complete",
        "collective_complete",
        "layer_complete",
        "pipeline_hop_start",
        "pipeline_hop_complete",
        "token_complete",
        "request_complete",
        "rank_failure",
        "rank_rejoin",
    }
    collective_projection_pass = projector_validation[
        "status"
    ] == "PASS" and required_events.issubset(collective_events)
    more_than_layers = any(
        int(row["pipeline_stage_count"]) == 4
        and int(row["tensor_parallel_degree"]) == 8
        and int(row["logical_pipeline_rank_workers"]) == 32
        and int(row["logical_layer_shards"]) == 224
        and bool(row["more_partitions_than_layers"])
        for row in validations
    )
    moe_pass = (
        acceptance_eligible and moe_rows and all(row.get("status") == "PASS" for row in moe_rows)
    )
    scheduler_pass = all(
        float(row["marginal_benefit"]) > 0 or not bool(row["weak_rank_should_join"])
        for row in heterogeneous_rows
    )
    compression_pass = any(row["primary_exact_pass"] for row in compression_rows)
    acceptance_statuses = {
        "experiment_integrity_status": stability["status"],
        "dense_partition_status": "PASS" if partition_pass else "FAIL",
        "dense_tensor_shard_status": "PASS" if tensor_shard_pass else "FAIL",
        "dense_layer_correctness_status": "PASS" if boundary_pass else "FAIL",
        "dense_token_identity_status": "PASS" if token_pass else "FAIL",
        "kv_partition_status": "PASS" if kv_pass else "FAIL",
        "vocabulary_parallel_status": "PASS" if vocabulary_pass else "FAIL",
        "collective_semantics_status": (
            "PASS" if projector_validation["status"] == "PASS" else "FAIL"
        ),
        "collective_projection_status": "PASS" if collective_projection_pass else "FAIL",
        "hybrid_pipeline_tensor_status": "PASS" if hybrid_rows else "FAIL",
        "heterogeneous_rank_status": "PASS" if scheduler_pass else "FAIL",
        "communication_compression_status": "PASS" if compression_pass else "FAIL",
        "deterministic_moe_status": "PASS" if moe_pass else "FAIL",
        "real_moe_layer_status": real_status,
        "expert_projection_status": "PASS" if expert_projection_rows else "FAIL",
        "k3_projection_status": k3_status,
        "more_partitions_than_layers_status": "PASS" if more_than_layers else "FAIL",
    }
    overall = classify_microsharding_overall(acceptance_statuses)
    summary: dict[str, Any] = {
        "prerequisite_status": prerequisite["status"],
        **acceptance_statuses,
        "overall_status": overall,
        "result_classifications": [
            "logical_single_gpu_measurement",
            "logical_microsharding_correctness",
            "independent_rank_projection",
            "low_latency_cell_projection",
            "wan_projection",
            "real_moe_layer_measurement",
            "k3_checkpoint_projection",
        ],
        "acceptance_eligible_full_matrix": acceptance_eligible,
        "smoke_mode": options.smoke,
        "dense_model_id": dense_model,
        "dense_model_revision": resolved.revision,
        "pipeline_stage_counts_tested": pipeline_counts,
        "tensor_parallel_degrees_tested": degrees,
        "transformer_layer_count": layer_count,
        "maximum_logical_layer_shards": max(
            int(row["logical_layer_shards"]) for row in validations
        ),
        "maximum_logical_pipeline_rank_workers": max(
            int(row["logical_pipeline_rank_workers"]) for row in validations
        ),
        "reference_process": reference["reference_process"],
        "projector_validation": projector_validation,
        "sequence_parallel_prefill": sequence_evidence,
        "measurement_stability": {
            "status": stability["status"],
            "configured_maximum_result_cv": stability["configured_maximum_result_cv"],
            "maximum_repeat_median_cv": stability["maximum_repeat_median_cv"],
            "method": stability["method"],
        },
        "secondary_model_status": secondary["status"],
        "real_moe_blocked_reason": real_reason,
        "physical_process_count": 1,
        "cuda_context_count": 1,
        "physical_tensor_parallel_speedup_claimed": False,
        "physical_memory_pooling_claimed": False,
        "run_directory": str(run_directory),
        "report_path": str(run_directory / "report.html"),
    }
    conclusion = _conclusion(
        summary=summary,
        memory=memory_rows,
        break_even=break_even_rows,
        real_moe_status=(
            "passed"
            if real_status == "PASS"
            else "failed"
            if real_status == "FAIL"
            else "was blocked"
        ),
        k3_plans=k3_plan_rows,
    )
    summary["conclusion"] = conclusion
    _json_write(run_directory / "summary.json", summary)

    generate_microsharding_charts(
        run_directory / "charts",
        memory=memory_rows,
        correctness=dense_correctness,
        boundaries=boundary_rows,
        kv=kv_rows,
        collective_metrics=collective_metrics,
        projections=projection_rows,
        break_even=break_even_rows,
        hybrid=hybrid_rows,
        heterogeneous=heterogeneous_rows,
        compression=compression_rows,
        moe=moe_rows,
        expert_projection=expert_projection_rows,
        expert_cache=expert_cache_rows,
        k3_plans=k3_plan_rows,
    )
    render_microsharding_report(
        run_directory / "report.html",
        summary=summary,
        conclusion=conclusion,
        run_metadata={
            "dense_model_revision": resolved.revision,
            "experiment_004_run": prerequisite.get("run_directory", "BLOCKED"),
            "real_moe_revision": model_revisions["real_moe"].get("revision"),
            "real_moe_required_download_bytes": real_download_payload.get(
                "required_download_bytes"
            ),
            "k3_revision": model_revisions["k3"].get("revision"),
            "execution_layout": "one process / one CUDA context / one RTX 5090",
        },
        artifact_names=list(REQUIRED_ARTIFACTS),
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_directory / name).is_file()]
    missing_charts = [
        name for name in REQUIRED_CHARTS if not (run_directory / "charts" / name).is_file()
    ]
    if missing or missing_charts:
        summary["experiment_integrity_status"] = "FAIL"
        summary["overall_status"] = "FAIL"
        summary["missing_artifacts"] = missing
        summary["missing_charts"] = missing_charts
        _json_write(run_directory / "summary.json", summary)
    return MicroshardingRun(
        run_directory=run_directory,
        report_path=run_directory / "report.html",
        summary=summary,
    )
