"""End-to-end correction runner for the two invalid Experiment 007 benchmarks."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import torch
import yaml
from transformers import AutoTokenizer

from swarm_inference.config.experiment_007_corrections import (
    Experiment007CorrectionsConfig,
)
from swarm_inference.experiments.experiment_007_background_correction import (
    TokenEventWriter,
    TrafficMode,
    WorkloadFixture,
    aggregate_fixed_window_results,
    run_serving_window,
    verify_sampled_requests,
)
from swarm_inference.experiments.experiment_007_corrected_planner import (
    corrected_planner_points,
    evaluate_corrected_planner,
)
from swarm_inference.experiments.experiment_007_correction_reporting import (
    generate_correction_charts,
    render_correction_report,
)
from swarm_inference.experiments.experiment_007_moe_correction import (
    CANONICAL_EXECUTOR_ID,
    CorrectionExpertFormat,
    benchmark_matched_moe,
    build_routing_corpus,
    load_moe_model_state,
    make_execution_plan,
)
from swarm_inference.experiments.heterogeneous_support import (
    csv_write,
    environment_snapshot,
    exact_token_length,
    json_write,
    repository_git_state,
    sha256,
    yaml_write,
)
from swarm_inference.experiments.services import (
    ManagedDockerService,
    start_llamacpp_service,
    start_sglang_service,
)
from swarm_inference.microsharding.real_moe import (
    RealMoEDownloadPlan,
    download_real_moe_layer_files,
)

ORIGINAL_RUN_ID = "20260801T013144Z-heterogeneous-node-utility-6a2b51ce"


@dataclass(slots=True)
class Experiment007CorrectionOptions:
    original_run: Path | None = None
    skip_expert_fix: bool = False
    skip_background_fix: bool = False
    smoke: bool = False
    resume: bool = False
    profile: bool = False
    output_root: Path | None = None
    keep_servers: bool = False


@dataclass(frozen=True, slots=True)
class Experiment007CorrectionRun:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("corrected_experiment_007_status") != "FAIL"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _locate_original_run(
    repository_root: Path,
    config: Experiment007CorrectionsConfig,
    option: Path | None,
) -> Path:
    candidates: list[Path] = []
    if option is not None:
        candidates.append(option if option.is_absolute() else repository_root / option)
    else:
        run_id = config.original_run.run_id
        candidates.extend(
            [
                repository_root / "artifacts" / "runs" / run_id,
                repository_root / "artifacts" / "runs-final" / run_id,
            ]
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir() and (resolved / "summary.json").is_file():
            return resolved
    raise FileNotFoundError("the immutable original Experiment 007 run was not found")


def _new_run_directory(
    repository_root: Path,
    config: Experiment007CorrectionsConfig,
    options: Experiment007CorrectionOptions,
) -> Path:
    output = options.output_root or Path(config.output_root)
    if not output.is_absolute():
        output = repository_root / output
    if options.resume:
        if (output / "config.requested.yaml").is_file():
            return output.resolve()
        existing = sorted(output.glob("*-experiment-007-corrections-*"), reverse=True)
        if existing:
            return existing[0].resolve()
        raise FileNotFoundError("--resume requested but no correction run exists")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-experiment-007-corrections-{uuid4().hex[:8]}"
    run_directory = (output / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _required_original_paths(original_run: Path) -> list[Path]:
    return [
        original_run / name
        for name in (
            "cpu_expert_results.csv",
            "expert_placement_results.csv",
            "expert_cache_results.csv",
            "background_results.csv",
            "summary.json",
            "config.requested.yaml",
            "config.resolved.yaml",
        )
    ]


def _preserve_original_evidence(original_run: Path, run_directory: Path) -> dict[str, Any]:
    files = _required_original_paths(original_run)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"original Experiment 007 evidence is incomplete: {missing}")
    summary = json.loads((original_run / "summary.json").read_text(encoding="utf-8"))
    relevant_logs = sorted(
        {
            *original_run.glob("logs/swarm-exp007-sglang-*"),
            *original_run.glob("logs/swarm-exp007-llama-q4-k-m-*-background.*"),
        }
    )
    reference = {
        "status": "PASS",
        "immutable_historical_evidence": True,
        "run_id": original_run.name,
        "run_directory": str(original_run),
        "summary": summary,
        "files": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
        "relevant_logs": {
            path.name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in relevant_logs
            if path.is_file()
        },
    }
    superseded = {
        "superseded_unmatched_cpu_expert_result": {
            "label": "superseded_unmatched_cpu_expert_result",
            "historical_only": True,
            "gpu_memory_saved_bytes": summary["measured"]["cpu_expert_gpu_memory_saved_bytes"],
            "throughput_retained_fraction": summary["measured"][
                "cpu_expert_throughput_retained_fraction"
            ],
            "reported_status": summary["cpu_expert_status"],
            "invalidity": (
                "full-GPU Hugging Face layer and custom hybrid executor used unmatched graphs, "
                "kernels, synchronisation, and timing boundaries"
            ),
        },
        "superseded_fixed_job_background_result": {
            "label": "superseded_fixed_job_background_result",
            "historical_only": True,
            "combined_gain_fraction": summary["measured"][
                "cpu_background_throughput_gain_fraction"
            ],
            "interactive_p95_change_fraction": summary["measured"][
                "cpu_background_interactive_p95_change_fraction"
            ],
            "reported_status": summary["cpu_background_status"],
            "invalidity": (
                "fixed job sets were divided by paired completion makespan rather than a shared "
                "fixed serving window"
            ),
        },
        "superseded_metrics_may_calibrate_planner": False,
    }
    json_write(run_directory / "original_experiment_reference.json", reference)
    json_write(run_directory / "superseded_results.json", superseded)
    return superseded


def _load_moe_plan(experiment_006: Path) -> RealMoEDownloadPlan:
    payload = json.loads(
        (experiment_006 / "real_moe_download_plan.json").read_text(encoding="utf-8")
    )
    if payload.get("status") not in {None, "PASS"}:
        raise RuntimeError("Experiment 006 real MoE evidence did not pass")
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


def _routing_histogram_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    histogram = cast(dict[str, int], manifest["expert_frequency_histogram"])
    calls = int(manifest["routed_expert_call_count"])
    return [
        {
            "classification": "measured_cuda",
            "expert_id": int(expert_id),
            "routed_calls": count,
            "routed_call_fraction": count / max(calls, 1),
            "corpus_hash": manifest["corpus_hash"],
            "benchmark_mode": "natural_routing",
        }
        for expert_id, count in sorted(histogram.items(), key=lambda item: int(item[0]))
    ]


def _run_moe_correction(
    *,
    experiment_006: Path,
    run_directory: Path,
    config: Experiment007CorrectionsConfig,
    smoke: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    plan = _load_moe_plan(experiment_006)
    files = download_real_moe_layer_files(plan)
    model = load_moe_model_state(plan, files)
    runtime_tokens = 512 if smoke else config.cpu_expert.routing_tokens
    corpus = build_routing_corpus(
        model,
        files,
        token_count=runtime_tokens,
        seed=7007,
        output_path=run_directory / "profiles" / "moe_routing_corpus.safetensors",
    )
    json_write(run_directory / "moe_routing_corpus_manifest.json", corpus.manifest)
    csv_write(run_directory / "moe_routing_histogram.csv", _routing_histogram_rows(corpus.manifest))
    policies = (
        config.cpu_expert.placement_policies[:2] if smoke else config.cpu_expert.placement_policies
    )
    counts = (
        config.cpu_expert.cpu_expert_counts[:2] if smoke else config.cpu_expert.cpu_expert_counts
    )
    formats = config.cpu_expert.formats[:1] if smoke else config.cpu_expert.formats
    rows, timing_rows, correctness_rows, controlled_rows = benchmark_matched_moe(
        model,
        corpus,
        policies=policies,
        expert_counts=counts,
        formats=formats,
        seed=7007,
        warmup_iterations=1 if smoke else config.cpu_expert.warmup_iterations,
        repeats=2 if smoke else config.cpu_expert.repeats,
        maximum_repeats=3 if smoke else config.cpu_expert.maximum_repeats,
        maximum_variability_epochs=(1 if smoke else config.cpu_expert.maximum_variability_epochs),
        maximum_cv=1.0 if smoke else config.cpu_expert.maximum_coefficient_of_variation,
        minimum_dispatch_fraction=config.cpu_expert.minimum_cpu_dispatch_fraction,
        minimum_retained_fraction=(
            config.positive_contribution.minimum_expert_retained_throughput_fraction
        ),
        atol=config.cpu_expert.correctness.atol,
        rtol=config.cpu_expert.correctness.rtol,
        minimum_cosine_similarity=(config.cpu_expert.correctness.minimum_cosine_similarity),
    )
    csv_write(run_directory / "moe_matched_results.csv", rows)
    csv_write(run_directory / "moe_timing_breakdown.csv", timing_rows)
    csv_write(run_directory / "moe_correctness.csv", correctness_rows)
    csv_write(run_directory / "moe_controlled_coverage.csv", controlled_rows)
    csv_write(
        run_directory / "moe_memory_results.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "classification",
                    "placement_policy",
                    "cpu_expert_count",
                    "weight_format",
                    "cpu_expert_calls",
                    "cpu_dispatch_fraction",
                    "expected_gpu_memory_saved_bytes",
                    "gpu_memory_saved_bytes",
                    "gpu_memory_saved_matches_expected",
                    "baseline_gpu_expert_weight_bytes",
                    "gpu_expert_weight_bytes",
                    "cpu_memory_used_bytes",
                    "memory_offload_pass",
                )
            }
            for row in rows
            if row.get("arm") == "hybrid_gpu_cpu"
        ],
    )
    plans: list[dict[str, Any]] = []
    for row in rows:
        cpu_ids = [int(item) for item in row.get("cpu_expert_ids", [])]
        plan_format = cast(CorrectionExpertFormat, row["weight_format"])
        plans.append(
            {
                "placement_policy": row["placement_policy"],
                "arm": row["arm"],
                "plan": make_execution_plan(
                    model,
                    corpus,
                    cpu_expert_ids=cpu_ids,
                    weight_format=plan_format,
                    execution_profile=(
                        "natural_routing_all_gpu"
                        if row["arm"] == "all_gpu"
                        else f"natural_routing_hybrid_{plan_format}"
                    ),
                ).to_dict(),
            }
        )
    json_write(run_directory / "moe_execution_plans.json", plans)
    bf16_baselines = [row for row in rows if row.get("arm") == "all_gpu"]
    bf16_hybrids = [
        row
        for row in rows
        if row.get("arm") == "hybrid_gpu_cpu" and row.get("weight_format") == "bfloat16"
    ]
    maximum_cv = 1.0 if smoke else config.cpu_expert.maximum_coefficient_of_variation
    matched_pass = bool(bf16_baselines and bf16_hybrids) and all(
        row.get("executor_id") == CANONICAL_EXECUTOR_ID
        and bool(row.get("matched_baseline_used"))
        and float(row.get("coefficient_of_variation", 1.0)) <= maximum_cv
        for row in bf16_baselines + bf16_hybrids
    )
    active_pass = (
        bool(bf16_hybrids)
        and all(
            int(row["cpu_expert_calls"]) == int(row["expected_cpu_dispatch_count"])
            for row in bf16_hybrids
        )
        and any(int(row["cpu_expert_calls"]) > 0 for row in bf16_hybrids)
    )
    memory_pass = bool(bf16_hybrids) and all(
        bool(row.get("gpu_memory_saved_matches_expected"))
        and int(row.get("gpu_memory_saved_bytes", 0)) > 0
        and bool(row.get("output_correctness_passed"))
        for row in bf16_hybrids
    )
    positive = any(bool(row.get("positive_performance_pass")) for row in bf16_hybrids)
    statuses = {
        "cpu_expert_matched_baseline_status": "PASS" if matched_pass else "FAIL",
        "cpu_expert_active_dispatch_status": "PASS" if active_pass else "FAIL",
        "cpu_expert_memory_offload_status": "PASS" if memory_pass else "FAIL",
        "cpu_expert_positive_performance_status": (
            "PASS" if positive else ("NOT_USEFUL" if matched_pass and active_pass else "FAIL")
        ),
        "model_revision": plan.revision,
        "layer_id": plan.selected_layer,
        "routing_corpus_size": corpus.token_count,
        "routing_corpus_hash": corpus.manifest["corpus_hash"],
    }
    del model, corpus
    gc.collect()
    torch.cuda.empty_cache()
    return rows, timing_rows, correctness_rows, statuses


def _find_backend_mapping(
    backend_evidence: dict[str, Any], backend_id: str, weight_format: str | None = None
) -> dict[str, Any]:
    for mapping in backend_evidence["mappings"]:
        if mapping.get("backend_id") != backend_id:
            continue
        if weight_format is not None and mapping.get("weight_format") != weight_format:
            continue
        return cast(dict[str, Any], mapping)
    raise RuntimeError(f"original backend mapping is missing: {backend_id}/{weight_format}")


def _workload_fixtures(
    tokenizer: Any,
    *,
    input_lengths: tuple[int, ...],
    output_lengths: tuple[int, ...],
    lane: str,
) -> list[WorkloadFixture]:
    prompts = [
        "Explain rate-based accounting for concurrent serving.",
        "Implement a deterministic producer consumer queue.",
        "Calculate the shared-window throughput of two independent workers.",
        "Write a detailed discussion of heterogeneous inference scheduling.",
    ]
    fixtures: list[WorkloadFixture] = []
    for index, (input_length, output_length) in enumerate(
        zip(input_lengths, output_lengths, strict=True)
    ):
        prompt = prompts[(index + (1 if lane == "cpu" else 0)) % len(prompts)]
        seed_tokens = [int(item) for item in tokenizer.encode(prompt, add_special_tokens=True)]
        token_ids = exact_token_length(
            seed_tokens, input_length, offset=index + (11 if lane == "cpu" else 0)
        )
        prompt_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        fixtures.append(
            WorkloadFixture(
                fixture_id=f"{lane}-{input_length}-{output_length}-{index}",
                prompt_token_ids=tuple(token_ids),
                requested_output_tokens=output_length,
                prompt_hash=prompt_hash,
            )
        )
    return fixtures


def _run_background_correction(
    *,
    repository_root: Path,
    original_run: Path,
    run_directory: Path,
    run_id: str,
    config: Experiment007CorrectionsConfig,
    smoke: bool,
    profile: bool,
    keep_servers: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    backend = json.loads((original_run / "backend_environments.json").read_text(encoding="utf-8"))
    mappings = json.loads(
        (original_run / "backend_artifact_mappings.json").read_text(encoding="utf-8")
    )
    target_mapping = _find_backend_mapping(mappings, "sglang")
    cpu_mapping = _find_backend_mapping(mappings, "llamacpp", "GGUF/Q4_K_M")
    target_path = Path(str(target_mapping["backend_artifact_path"]))
    cpu_path = Path(str(cpu_mapping["backend_artifact_path"]))
    if sha256(cpu_path) != str(cpu_mapping["backend_artifact_hash"]):
        raise RuntimeError("pinned Q4 GGUF hash changed before corrected background benchmark")
    huggingface_cache = Path.home() / ".cache" / "huggingface"
    snapshot_relative = str(target_path.relative_to(huggingface_cache)).replace("\\", "/")
    llama_environment = cast(dict[str, Any], backend["llamacpp"])
    environment_root = Path(str(llama_environment["root"]))
    build_name = Path(str(llama_environment["server_path"])).resolve().parents[1].name
    sglang_service: ManagedDockerService | None = None
    llama_service: ManagedDockerService | None = None
    event_writer = TokenEventWriter(run_directory / "background_token_events.jsonl")
    combined_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []
    samples: list[Any] = []
    try:
        sglang_service = start_sglang_service(
            image=str(backend["sglang"]["image"]),
            repository_root=repository_root,
            huggingface_cache=huggingface_cache,
            model_snapshot_relative=snapshot_relative,
            run_id=f"{run_id}-correction",
            log_root=run_directory / "logs",
            maximum_running_requests=64,
        )
        import psutil

        cpu_thread_count = max(1, (psutil.cpu_count(logical=False) or 2) - 2)
        llama_service = start_llamacpp_service(
            environment_root=environment_root,
            build_name=build_name,
            gguf_path=cpu_path,
            repository_root=repository_root,
            run_id=f"{run_id}-correction",
            format_name="Q4_K_M",
            log_root=run_directory / "logs",
            thread_count=cpu_thread_count,
            parallel=4,
        )
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            target_path, local_files_only=True
        )
        input_lengths = (32, 48, 64) if smoke else config.background.input_token_lengths
        output_lengths = (8, 8, 8) if smoke else config.background.output_token_lengths
        gpu_fixtures = _workload_fixtures(
            tokenizer,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            lane="gpu",
        )
        cpu_fixtures = _workload_fixtures(
            tokenizer,
            input_lengths=input_lengths,
            output_lengths=output_lengths,
            lane="cpu",
        )
        warmup_seconds = 1.0 if smoke else config.background.warmup_seconds
        measurement_seconds = 3.0 if smoke else config.background.measurement_seconds
        drain_seconds = 10.0 if smoke else config.background.drain_timeout_seconds
        repeats = 1 if smoke else config.background.repeats
        gpu_concurrencies = (1, 4) if smoke else config.background.gpu_concurrency
        cpu_concurrencies = (1,) if smoke else config.background.cpu_concurrency

        async def one(
            *,
            point_id: str,
            arm: str,
            repeat: int,
            traffic_mode: str,
            gpu_concurrency: int,
            cpu_concurrency: int,
            arrival_rate: float | None,
        ) -> None:
            assert sglang_service is not None and llama_service is not None
            combined, gpu, cpu, point_samples = await run_serving_window(
                run_point_id=point_id,
                arm=cast(Any, arm),
                repeat=repeat,
                traffic_mode=cast(TrafficMode, traffic_mode),
                gpu_endpoint=sglang_service.endpoint,
                cpu_endpoint=llama_service.endpoint,
                gpu_fixtures=gpu_fixtures,
                cpu_fixtures=cpu_fixtures,
                gpu_concurrency=gpu_concurrency,
                cpu_concurrency=cpu_concurrency,
                open_loop_arrival_rate_rps=arrival_rate,
                warmup_seconds=warmup_seconds,
                measurement_seconds=measurement_seconds,
                drain_timeout_seconds=drain_seconds,
                gpu_artifact_hash=str(target_mapping["backend_artifact_hash"]),
                gpu_revision=str(target_mapping["canonical_revision"]),
                cpu_artifact_hash=str(cpu_mapping["backend_artifact_hash"]),
                cpu_revision=str(cpu_mapping["canonical_revision"]),
                event_writer=event_writer,
                cpu_thread_count=cpu_thread_count,
                workload_seed=config.background.workload_seed,
                telemetry_interval_seconds=0.25 if profile else 1.0,
            )
            combined["gpu_model_load_seconds"] = sglang_service.launch_seconds
            combined["cpu_model_load_seconds"] = llama_service.launch_seconds
            combined_rows.append(combined)
            gpu_rows.append(gpu)
            cpu_rows.append(cpu)
            samples.extend(point_samples)
            csv_write(run_directory / "background_window_results.csv", combined_rows)
            csv_write(run_directory / "background_gpu_metrics.csv", gpu_rows)
            csv_write(run_directory / "background_cpu_metrics.csv", cpu_rows)

        for gpu_concurrency in gpu_concurrencies:
            for repeat in range(repeats):
                asyncio.run(
                    one(
                        point_id=f"closed-gpu-only-g{gpu_concurrency}-r{repeat}",
                        arm="gpu_only",
                        repeat=repeat,
                        traffic_mode="closed_loop",
                        gpu_concurrency=gpu_concurrency,
                        cpu_concurrency=0,
                        arrival_rate=None,
                    )
                )
        for cpu_concurrency in cpu_concurrencies:
            for repeat in range(repeats):
                asyncio.run(
                    one(
                        point_id=f"closed-cpu-only-c{cpu_concurrency}-r{repeat}",
                        arm="cpu_only",
                        repeat=repeat,
                        traffic_mode="closed_loop",
                        gpu_concurrency=0,
                        cpu_concurrency=cpu_concurrency,
                        arrival_rate=None,
                    )
                )
        for gpu_concurrency in gpu_concurrencies:
            for cpu_concurrency in cpu_concurrencies:
                for repeat in range(repeats):
                    asyncio.run(
                        one(
                            point_id=(
                                f"closed-paired-g{gpu_concurrency}-c{cpu_concurrency}-r{repeat}"
                            ),
                            arm="gpu_plus_cpu",
                            repeat=repeat,
                            traffic_mode="closed_loop",
                            gpu_concurrency=gpu_concurrency,
                            cpu_concurrency=cpu_concurrency,
                            arrival_rate=None,
                        )
                    )
        if not smoke:
            reference_concurrency = config.background.open_loop_concurrency
            reference_rows = [
                row
                for row in combined_rows
                if row["arm"] == "gpu_only"
                and row["traffic_mode"] == "closed_loop"
                and int(row["gpu_concurrency"]) == reference_concurrency
            ]
            request_capacity = statistics.median(
                float(row["gpu_completed_requests"]) / float(row["measurement_window_seconds"])
                for row in reference_rows
            )
            for load_fraction in config.background.open_loop_load_fractions:
                arrival_rate = max(request_capacity * load_fraction, 0.01)
                load_tag = int(load_fraction * 100)
                for repeat in range(repeats):
                    asyncio.run(
                        one(
                            point_id=f"open-gpu-only-l{load_tag}-r{repeat}",
                            arm="gpu_only",
                            repeat=repeat,
                            traffic_mode="open_loop",
                            gpu_concurrency=reference_concurrency,
                            cpu_concurrency=0,
                            arrival_rate=arrival_rate,
                        )
                    )
                    asyncio.run(
                        one(
                            point_id=f"open-paired-l{load_tag}-r{repeat}",
                            arm="gpu_plus_cpu",
                            repeat=repeat,
                            traffic_mode="open_loop",
                            gpu_concurrency=reference_concurrency,
                            cpu_concurrency=config.background.open_loop_cpu_concurrency,
                            arrival_rate=arrival_rate,
                        )
                    )
        assert sglang_service is not None and llama_service is not None
        correctness_rows = verify_sampled_requests(
            samples,
            gpu_endpoint=sglang_service.endpoint,
            cpu_endpoint=llama_service.endpoint,
            workload_seed=config.background.workload_seed,
        )
        csv_write(run_directory / "background_correctness.csv", correctness_rows)
        comparison_rows = aggregate_fixed_window_results(
            combined_rows,
            minimum_combined_gain_fraction=(
                config.positive_contribution.minimum_combined_gain_fraction
            ),
            maximum_gpu_p95_increase_fraction=(
                config.positive_contribution.maximum_gpu_p95_increase_fraction
            ),
            maximum_gpu_throughput_decrease_fraction=(
                config.positive_contribution.maximum_gpu_throughput_decrease_fraction
            ),
        )
        csv_write(run_directory / "background_combined_metrics.csv", comparison_rows)
        expected_closed_paired = len(gpu_concurrencies) * len(cpu_concurrencies) * repeats
        observed_closed_paired = sum(
            row["arm"] == "gpu_plus_cpu" and row["traffic_mode"] == "closed_loop"
            for row in combined_rows
        )
        fixed_window_pass = observed_closed_paired == expected_closed_paired and all(
            row.get("denominator_kind") == "shared_fixed_measurement_window"
            and abs(float(row["measurement_window_seconds"]) - measurement_seconds) < 1e-6
            for row in combined_rows
        )
        accounting_pass = (
            bool(correctness_rows)
            and all(row["verification_status"] == "PASS" for row in correctness_rows)
            and all(
                abs(
                    float(row["combined_verified_tps"])
                    - (
                        int(row["gpu_verified_output_tokens"])
                        + int(row["cpu_verified_output_tokens"])
                    )
                    / float(row["measurement_window_seconds"])
                )
                < 1e-9
                for row in combined_rows
            )
        )
        positive = any(
            bool(row["positive_contribution_pass"])
            for row in comparison_rows
            if row["traffic_mode"] == "closed_loop"
        )
        statuses = {
            "background_fixed_window_status": "PASS" if fixed_window_pass else "FAIL",
            "background_token_accounting_status": "PASS" if accounting_pass else "FAIL",
            "background_positive_contribution_status": (
                "PASS"
                if positive
                else ("NOT_USEFUL" if fixed_window_pass and accounting_pass else "FAIL")
            ),
            "measurement_window_seconds": measurement_seconds,
            "warmup_seconds": warmup_seconds,
            "drain_timeout_seconds": drain_seconds,
            "repeat_count": repeats,
            "sampled_correctness_count": len(correctness_rows),
        }
        return comparison_rows, correctness_rows, statuses
    finally:
        event_writer.close()
        if not keep_servers:
            if llama_service is not None:
                llama_service.close(repository_root=repository_root)
            if sglang_service is not None:
                sglang_service.close(repository_root=repository_root)


def _empty_required_artifacts(run_directory: Path) -> None:
    csv_names = (
        "moe_routing_histogram.csv",
        "moe_matched_results.csv",
        "moe_timing_breakdown.csv",
        "moe_correctness.csv",
        "moe_memory_results.csv",
        "moe_controlled_coverage.csv",
        "background_window_results.csv",
        "background_gpu_metrics.csv",
        "background_cpu_metrics.csv",
        "background_combined_metrics.csv",
        "background_correctness.csv",
        "planner_calibration.csv",
        "planner_held_out_results.csv",
        "planner_regret.csv",
    )
    for name in csv_names:
        if not (run_directory / name).exists():
            csv_write(run_directory / name, [])
    for name in ("moe_routing_corpus_manifest.json", "moe_execution_plans.json"):
        if not (run_directory / name).exists():
            json_write(run_directory / name, {"status": "NOT_RUN"})
    token_events = run_directory / "background_token_events.jsonl"
    if not token_events.exists():
        token_events.touch()


def _best_valid_expert(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if bool(row.get("positive_performance_eligible"))]
    return max(
        eligible,
        key=lambda row: float(row.get("throughput_retained_fraction", -1.0)),
        default=None,
    )


def _best_background(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row.get("traffic_mode") == "closed_loop"]
    return max(
        eligible,
        key=lambda row: float(row.get("combined_gain_fraction", -1.0)),
        default=None,
    )


def _corrected_status(summary: dict[str, Any]) -> str:
    validity_keys = (
        "cpu_expert_matched_baseline_status",
        "cpu_expert_active_dispatch_status",
        "cpu_expert_memory_offload_status",
        "background_fixed_window_status",
        "background_token_accounting_status",
        "planner_held_out_evaluation_status",
        "planner_regret_status",
    )
    if any(summary.get(key) != "PASS" for key in validity_keys):
        return "FAIL"
    useful = any(
        summary.get(key) == "PASS"
        for key in (
            "cpu_expert_positive_performance_status",
            "background_positive_contribution_status",
        )
    )
    return "PASS" if useful else "PARTIAL_PASS"


def run_experiment_007_corrections(
    config: Experiment007CorrectionsConfig,
    *,
    requested_config_path: Path,
    options: Experiment007CorrectionOptions | None = None,
) -> Experiment007CorrectionRun:
    options = options or Experiment007CorrectionOptions()
    repository_root = _repository_root()
    original_run = _locate_original_run(repository_root, config, options.original_run)
    run_directory = _new_run_directory(repository_root, config, options)
    for directory in ("logs", "profiles", "charts"):
        (run_directory / directory).mkdir(parents=True, exist_ok=True)
    requested_payload = json.loads(json.dumps(config.model_dump(mode="json")))
    if options.resume:
        existing_requested = yaml.safe_load(
            (run_directory / "config.requested.yaml").read_text(encoding="utf-8")
        )
        if existing_requested != requested_payload:
            raise ValueError("--resume configuration differs from the interrupted correction run")
        existing_summary_path = run_directory / "summary.json"
        existing_report_path = run_directory / "report.html"
        if existing_summary_path.is_file() and existing_report_path.is_file():
            existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            if (
                existing_summary.get("corrected_experiment_007_status") in {"PASS", "PARTIAL_PASS"}
                and bool(existing_summary.get("smoke")) == options.smoke
            ):
                return Experiment007CorrectionRun(
                    run_directory,
                    existing_report_path,
                    existing_summary,
                )
        existing_events = run_directory / "background_token_events.jsonl"
        if existing_events.is_file() and existing_events.stat().st_size:
            interrupted_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            existing_events.replace(
                run_directory
                / "profiles"
                / f"background_token_events.interrupted-{interrupted_at}.jsonl"
            )
    yaml_write(run_directory / "config.requested.yaml", requested_payload)
    json_write(run_directory / "environment.json", environment_snapshot())
    json_write(run_directory / "git.json", repository_git_state(repository_root))
    superseded = _preserve_original_evidence(original_run, run_directory)
    original_summary = json.loads((original_run / "summary.json").read_text(encoding="utf-8"))
    experiment_006 = Path(str(original_summary["experiment_006_run"])).resolve()
    moe_source_plan = _load_moe_plan(experiment_006)
    artifact_mappings = json.loads(
        (original_run / "backend_artifact_mappings.json").read_text(encoding="utf-8")
    )
    target_mapping = _find_backend_mapping(artifact_mappings, "sglang")
    background_mapping = _find_backend_mapping(artifact_mappings, "llamacpp", "GGUF/Q4_K_M")
    resolved_payload = json.loads(json.dumps(config.model_dump(mode="json")))
    resolved_payload["requested_config_path"] = str(requested_config_path.resolve())
    resolved_payload["original_run"]["path"] = str(original_run)
    resolved_payload["cpu_expert"].update(
        {
            "experiment_006_run": str(experiment_006),
            "model_id": moe_source_plan.model_id,
            "model_revision": moe_source_plan.revision,
            "layer_id": moe_source_plan.selected_layer,
        }
    )
    resolved_payload["background"]["gpu_target"] = {
        "model_id": target_mapping["canonical_model_id"],
        "model_revision": target_mapping["canonical_revision"],
        "artifact_hash": target_mapping["backend_artifact_hash"],
    }
    resolved_payload["background"]["cpu_worker"] = {
        "model_id": background_mapping["canonical_model_id"],
        "model_revision": background_mapping["canonical_revision"],
        "artifact_hash": background_mapping["backend_artifact_hash"],
        "weight_format": background_mapping["weight_format"],
    }
    resolved_payload["runtime"] = {
        "smoke": options.smoke,
        "profile": options.profile,
        "skip_expert_fix": options.skip_expert_fix,
        "skip_background_fix": options.skip_background_fix,
    }
    yaml_write(run_directory / "config.resolved.yaml", resolved_payload)
    moe_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    held_out_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}
    phase_errors: dict[str, str] = {}
    started = time.perf_counter()
    try:
        if options.skip_expert_fix:
            statuses.update(
                {
                    "cpu_expert_matched_baseline_status": "FAIL",
                    "cpu_expert_active_dispatch_status": "FAIL",
                    "cpu_expert_memory_offload_status": "FAIL",
                    "cpu_expert_positive_performance_status": "FAIL",
                }
            )
        else:
            try:
                moe_rows, timing_rows, _correctness_rows, moe_statuses = _run_moe_correction(
                    experiment_006=experiment_006,
                    run_directory=run_directory,
                    config=config,
                    smoke=options.smoke,
                )
                statuses.update(moe_statuses)
            except Exception as exc:
                phase_errors["cpu_expert"] = f"{type(exc).__name__}: {exc}"
                statuses.update(
                    {
                        "cpu_expert_matched_baseline_status": "FAIL",
                        "cpu_expert_active_dispatch_status": "FAIL",
                        "cpu_expert_memory_offload_status": "FAIL",
                        "cpu_expert_positive_performance_status": "FAIL",
                    }
                )
        if options.skip_background_fix:
            statuses.update(
                {
                    "background_fixed_window_status": "FAIL",
                    "background_token_accounting_status": "FAIL",
                    "background_positive_contribution_status": "FAIL",
                }
            )
        else:
            try:
                background_rows, _background_correctness, background_statuses = (
                    _run_background_correction(
                        repository_root=repository_root,
                        original_run=original_run,
                        run_directory=run_directory,
                        run_id=run_directory.name[-8:],
                        config=config,
                        smoke=options.smoke,
                        profile=options.profile,
                        keep_servers=options.keep_servers,
                    )
                )
                statuses.update(background_statuses)
            except Exception as exc:
                phase_errors["background"] = f"{type(exc).__name__}: {exc}"
                statuses.update(
                    {
                        "background_fixed_window_status": "FAIL",
                        "background_token_accounting_status": "FAIL",
                        "background_positive_contribution_status": "FAIL",
                    }
                )
        try:
            if not moe_rows or not background_rows:
                raise RuntimeError(
                    "both corrected role datasets are required for held-out planning"
                )
            planner_points = corrected_planner_points(
                moe_rows,
                background_rows,
                minimum_expert_retained_fraction=(
                    config.positive_contribution.minimum_expert_retained_throughput_fraction
                ),
            )
            calibration_rows, held_out_rows, regret_rows, planner_model = (
                evaluate_corrected_planner(
                    planner_points,
                    maximum_regret_fraction=config.planner.maximum_regret_fraction,
                )
            )
            csv_write(run_directory / "planner_calibration.csv", calibration_rows)
            csv_write(run_directory / "planner_held_out_results.csv", held_out_rows)
            csv_write(run_directory / "planner_regret.csv", regret_rows)
            json_write(run_directory / "profiles" / "planner_model.json", planner_model)
            planner_pass = bool(regret_rows) and all(bool(row["passes"]) for row in regret_rows)
            statuses["planner_held_out_evaluation_status"] = "PASS"
            statuses["planner_regret_status"] = "PASS" if planner_pass else "FAIL"
        except Exception as exc:
            phase_errors["planner"] = f"{type(exc).__name__}: {exc}"
            statuses["planner_held_out_evaluation_status"] = "FAIL"
            statuses["planner_regret_status"] = "FAIL"
            csv_write(run_directory / "planner_calibration.csv", [])
            csv_write(run_directory / "planner_held_out_results.csv", [])
            csv_write(run_directory / "planner_regret.csv", [])
        summary: dict[str, Any] = {
            **statuses,
            "original_experiment_007_run": str(original_run),
            "original_run_id": original_run.name,
            "experiment_006_run": str(experiment_006),
            "superseded_results_preserved": True,
            "superseded_metrics_used_by_planner": False,
            "phase_errors": phase_errors,
            "run_directory": str(run_directory),
            "runtime_seconds": time.perf_counter() - started,
            "smoke": options.smoke,
        }
        summary["corrected_experiment_007_status"] = _corrected_status(summary)
        summary["experiment_integrity_status"] = (
            "PASS" if summary["corrected_experiment_007_status"] != "FAIL" else "FAIL"
        )
        best_expert = _best_valid_expert(moe_rows)
        best_background = _best_background(background_rows)
        summary["corrected_headlines"] = {
            "cpu_expert": best_expert,
            "background": best_background,
            "planner": regret_rows[0] if regret_rows else None,
        }
        json_write(run_directory / "summary.json", summary)
        _empty_required_artifacts(run_directory)
        generate_correction_charts(
            run_directory,
            moe_rows=moe_rows,
            timing_rows=timing_rows,
            background_rows=background_rows,
            planner_rows=held_out_rows,
            regret_rows=regret_rows,
        )
        report = render_correction_report(
            run_directory,
            summary=summary,
            superseded=superseded,
            moe_rows=moe_rows,
            background_rows=background_rows,
            held_out_rows=held_out_rows,
        )
        return Experiment007CorrectionRun(run_directory, report, summary)
    except BaseException as exc:
        failure = {
            **statuses,
            "corrected_experiment_007_status": "FAIL",
            "experiment_integrity_status": "FAIL",
            "phase_errors": {
                **phase_errors,
                "orchestrator": f"{type(exc).__name__}: {exc}",
            },
            "original_experiment_007_run": str(original_run),
            "run_directory": str(run_directory),
            "runtime_seconds": time.perf_counter() - started,
        }
        json_write(run_directory / "summary.json", failure)
        _empty_required_artifacts(run_directory)
        raise
