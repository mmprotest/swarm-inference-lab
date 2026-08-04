"""Typer command-line interface for all research workflows."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.config.models import Backend, ExecutionMode, QueueConfig
from swarm_inference.doctor import DoctorBackend, inspect_environment, render_doctor_report
from swarm_inference.exceptions import SwarmError
from swarm_inference.experiments.loopback import (
    run_loopback_experiment,
    run_physical_experiment,
)
from swarm_inference.experiments.loopback_matrix import run_loopback_matrix
from swarm_inference.experiments.reporting import render_html_report
from swarm_inference.experiments.runner import run_experiment, validate_run
from swarm_inference.host import resolve_advertised_endpoint
from swarm_inference.logging import configure_logging
from swarm_inference.model.feasibility import analyse_large_model_manifest
from swarm_inference.model.manifest import (
    manifest_summary,
)
from swarm_inference.model.reference import validate_qwen_correctness
from swarm_inference.model.shard_builder import (
    inspect_qwen3_model,
    model_inspection_payload,
    resolve_model,
    shard_model,
)

app = typer.Typer(
    name="swarm",
    help=(
        "Falsifiable heterogeneous consumer-device inference experiments. "
        "Aggregate throughput is never presented as single-request speed."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
experiment_app = typer.Typer(
    name="experiment",
    help="Run simulation, loopback, physical, and named real-model experiments.",
    invoke_without_command=True,
    no_args_is_help=False,
)
app.add_typer(experiment_app, name="experiment")


def _fail(message: str, *, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code)


@app.callback()
def main(
    log_level: Annotated[str, typer.Option(help="Python logging level.")] = "INFO",
    json_logs: Annotated[bool, typer.Option(help="Emit process logs as JSON lines.")] = False,
) -> None:
    configure_logging(level=log_level, json_output=json_logs)


@app.command("doctor")
def doctor_command(
    json_output: Annotated[
        bool, typer.Option("--json", help="Render the full check as JSON.")
    ] = False,
    allow_cpu: Annotated[
        bool,
        typer.Option(help="Permit CPU fallback when auto selects an unavailable CUDA/MPS backend."),
    ] = False,
    backend: Annotated[
        DoctorBackend,
        typer.Option(help="Backend to validate; auto selects CUDA, MPS, or CPU from hardware."),
    ] = DoctorBackend.AUTO,
    bind_host: Annotated[
        str,
        typer.Option(
            help="Address used to test required port availability; defaults to every IPv4 interface."
        ),
    ] = "0.0.0.0",
) -> None:
    """Inspect the native host, PyTorch backends, memory, disk, network, and ports."""

    report = inspect_environment(target_backend=backend, bind_host=bind_host)
    typer.echo(render_doctor_report(report, json_output=json_output))
    if not report.selected_backend_compatible and not (allow_cpu and report.compatible_cpu):
        raise typer.Exit(1)


@app.command("inspect-model")
def inspect_model_command(
    model: Annotated[str, typer.Option(help="Hugging Face model ID or local path.")],
    revision: Annotated[str | None, typer.Option(help="Revision to resolve exactly.")] = None,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Optional Hugging Face cache directory.")
    ] = None,
    allow_download: Annotated[
        bool, typer.Option(help="Allow the explicit model inspection to download files.")
    ] = True,
) -> None:
    """Inspect dense Qwen3 config and safetensors headers without loading the model."""

    try:
        resolved = resolve_model(
            model,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        description = inspect_qwen3_model(resolved)
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"inspect-model failed: {exc}")
    typer.echo(json.dumps(model_inspection_payload(description), indent=2, sort_keys=True))


@app.command("shard-model")
def shard_model_command(
    model: Annotated[str, typer.Option(help="Hugging Face model ID or local path.")],
    output: Annotated[Path, typer.Option(help="New empty stage-shard directory.")],
    revision: Annotated[str | None, typer.Option(help="Revision to resolve exactly.")] = None,
    target_stage_bytes: Annotated[
        int,
        typer.Option(
            "--target-stage-bytes",
            help="Preferred weight bytes per contiguous stage.",
        ),
    ] = 512 * 1024 * 1024,
    max_stage_bytes: Annotated[
        int,
        typer.Option(
            "--max-stage-bytes",
            help="Hard logical weight cap; oversized stages are refused.",
        ),
    ] = 512 * 1024 * 1024,
    stage_count: Annotated[
        int | None,
        typer.Option(
            "--stage-count",
            min=1,
            help="Require exactly this many contiguous tensor-size-balanced stages.",
        ),
    ] = None,
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    allow_download: Annotated[bool, typer.Option()] = True,
) -> None:
    """Build verified stage-only safetensors shards; never instantiate a full model."""

    try:
        resolved = resolve_model(
            model,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
        description = inspect_qwen3_model(resolved)
        manifest = shard_model(
            description,
            output=output,
            target_stage_bytes=target_stage_bytes,
            maximum_stage_bytes=max_stage_bytes,
            stage_count=stage_count,
        )
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"shard-model failed: {exc}")
    typer.echo(json.dumps(manifest_summary(manifest), indent=2, sort_keys=True))


@app.command("inspect-partition")
def inspect_partition_command(
    path: Annotated[
        Path,
        typer.Option("--path", exists=True, file_okay=False, help="Microshard artifact root."),
    ],
) -> None:
    """Inspect a hierarchical tensor/pipeline microshard artifact."""

    from swarm_inference.microsharding.builder import inspect_partition

    try:
        payload = inspect_partition(path)
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"inspect-partition failed: {exc}")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("build-microshards")
def build_microshards_command(
    model: Annotated[str, typer.Option(help="Hugging Face model ID or local path.")],
    output: Annotated[Path, typer.Option(help="New empty microshard directory.")],
    revision: Annotated[str | None, typer.Option(help="Immutable model revision.")] = None,
    pipeline_stages: Annotated[
        int, typer.Option("--pipeline-stages", min=1, help="Pipeline-stage count.")
    ] = 1,
    tensor_parallel_degree: Annotated[
        int,
        typer.Option("--tensor-parallel-degree", min=1, help="Ranks in each parallel cell."),
    ] = 1,
    vocabulary_parallel: Annotated[
        bool,
        typer.Option(
            "--vocabulary-parallel/--no-vocabulary-parallel",
            help="Shard embedding and LM-head vocabulary rows.",
        ),
    ] = True,
    cache_dir: Annotated[Path | None, typer.Option()] = None,
    allow_download: Annotated[bool, typer.Option()] = True,
) -> None:
    """Build rank-local safetensors directly from checkpoint slices."""

    from swarm_inference.microsharding.builder import build_microshards

    try:
        result = build_microshards(
            model=model,
            revision=revision,
            pipeline_stage_count=pipeline_stages,
            tensor_parallel_degree=tensor_parallel_degree,
            output=output,
            vocabulary_parallel=vocabulary_parallel,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"build-microshards failed: {exc}")
    typer.echo(json.dumps(result.validation, indent=2, sort_keys=True))
    if result.validation["status"] != "PASS":
        raise typer.Exit(1)


@app.command("validate-microshards")
def validate_microshards_command(
    path: Annotated[
        Path,
        typer.Option("--path", exists=True, file_okay=False, help="Microshard artifact root."),
    ],
    source_model: Annotated[
        Path | None,
        typer.Option(
            exists=True, file_okay=False, help="Optional source checkpoint for rehashing."
        ),
    ] = None,
) -> None:
    """Validate shard unions, tensor hashes, strict coverage, and memory gates."""

    from swarm_inference.microsharding.builder import validate_microshards

    try:
        payload = validate_microshards(path, source_model=source_model)
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"validate-microshards failed: {exc}")
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise typer.Exit(1)


@app.command("simulate")
def simulate_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    """Run a deterministic event-queue simulation and write a complete report."""

    experiment = load_experiment_config(config)
    if experiment.execution_mode != ExecutionMode.SIMULATION:
        _fail("simulate requires execution_mode: simulation")
    run = run_experiment(experiment)
    typer.echo(f"execution_mode={experiment.execution_mode.value}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"status={run.summary['status']}")
    if not run.passed:
        raise typer.Exit(1)


@experiment_app.callback(invoke_without_command=True)
def experiment_command(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            help="Override local loopback worker count or expected remote physical workers."
        ),
    ] = None,
    listen: Annotated[
        str,
        typer.Option(help="Coordinator bind endpoint for physical modes."),
    ] = "0.0.0.0:50051",
    startup_timeout_s: Annotated[
        float,
        typer.Option(min=1, help="Seconds to wait for physical worker registration."),
    ] = 300.0,
    duration_s: Annotated[
        float | None,
        typer.Option(
            min=0.01,
            help="Override sustained loopback or physical steady-state duration.",
        ),
    ] = None,
    repeats: Annotated[
        int | None,
        typer.Option(min=1, help="Repeats per point for a loopback scaling matrix."),
    ] = None,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile",
            help="Capture lightweight CPU, memory, event-loop, queue, and transport profiles.",
        ),
    ] = False,
    model_manifest: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Optional real-model manifest for a physical experiment.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Resolved model metadata/tokenizer path for a physical real-model run.",
        ),
    ] = None,
    dtype: Annotated[
        str | None,
        typer.Option(help="Physical real-model execution dtype."),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option(help="Prompt repeated by physical real-model workload lanes."),
    ] = "Explain why distributed inference is difficult.",
) -> None:
    """Run a simulation, native loopback, or remote physical experiment."""

    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        _fail("experiment requires --config, or use a named subcommand")
    assert config is not None
    experiment = load_experiment_config(config)
    if profile:
        experiment.profiling.enabled = True
    if experiment.execution_mode not in {
        ExecutionMode.PHYSICAL_LAN,
        ExecutionMode.PHYSICAL_WAN,
    } and (model_manifest is not None or model_path is not None or dtype is not None):
        _fail("--model-manifest, --model-path, and --dtype are physical-mode options")
    if experiment.execution_mode == ExecutionMode.SIMULATION:
        run = run_experiment(experiment)
    elif experiment.execution_mode == ExecutionMode.SINGLE_HOST_LOOPBACK:
        matrix_requested = (
            workers is None
            and len(experiment.node_counts) >= 2
            and bool(experiment.concurrent_request_counts)
        )
        if matrix_requested:
            run = asyncio.run(
                run_loopback_matrix(
                    experiment,
                    repeats=repeats,
                    duration_s=duration_s,
                )
            )
        else:
            count = workers or sum(item.count for item in experiment.nodes)
            run = asyncio.run(
                run_loopback_experiment(
                    experiment,
                    worker_count=count,
                    sustained=duration_s is not None,
                    duration_s=duration_s,
                )
            )
    elif experiment.execution_mode in {
        ExecutionMode.PHYSICAL_LAN,
        ExecutionMode.PHYSICAL_WAN,
    }:
        if (model_manifest is None) != (model_path is None):
            _fail("--model-manifest and --model-path must be supplied together")
        manifest = None
        architecture_config = None
        tokenizer = None
        if model_manifest is not None and model_path is not None:
            from swarm_inference.config.models import ModelManifest

            manifest = ModelManifest.model_validate_json(model_manifest.read_text(encoding="utf-8"))
            architecture_config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8")
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_path,
                local_files_only=True,
            )
        count = workers or sum(item.count for item in experiment.nodes)

        def ready(endpoint: str, run_dir: Path) -> None:
            typer.echo(f"coordinator_listening={endpoint}")
            typer.echo(f"run_directory={run_dir}")
            typer.echo(
                f"start remote workers now; the coordinator is waiting for {count} registrations"
            )

        try:
            run = asyncio.run(
                run_physical_experiment(
                    experiment,
                    expected_worker_count=count,
                    listen_endpoint=listen,
                    startup_timeout_s=startup_timeout_s,
                    duration_s=duration_s,
                    ready_callback=ready,
                    model_manifest=manifest,
                    architecture_config=architecture_config,
                    runtime_dtype=dtype,
                    tokenizer=tokenizer,
                    prompt=prompt,
                )
            )
        except (SwarmError, OSError, TimeoutError, ValueError) as exc:
            _fail(f"physical experiment failed: {exc}")
    else:  # pragma: no cover - exhaustive guard for future execution modes
        _fail(f"unsupported execution mode: {experiment.execution_mode.value}")
    typer.echo(f"execution_mode={experiment.execution_mode.value}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"status={run.summary.get('overall_status', run.summary['status'])}")
    if not run.passed:
        raise typer.Exit(1)


def _positive_csv(value: str | None, option_name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise typer.BadParameter(f"{option_name} values must be positive")
    return parsed


@experiment_app.command("microsharding")
def microsharding_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 006 YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_006_microsharding.yaml"),
    dense_model: Annotated[str | None, typer.Option("--dense-model")] = None,
    dense_revision: Annotated[str | None, typer.Option("--dense-revision")] = None,
    pipeline_stages: Annotated[
        str | None,
        typer.Option("--pipeline-stages", help="Comma-separated pipeline-stage counts."),
    ] = None,
    tensor_parallel_degrees: Annotated[
        str | None,
        typer.Option("--tensor-parallel-degrees", help="Comma-separated TP degrees."),
    ] = None,
    skip_secondary_model: Annotated[bool, typer.Option("--skip-secondary-model")] = False,
    skip_real_moe: Annotated[bool, typer.Option("--skip-real-moe")] = False,
    real_moe_download_budget_gib: Annotated[
        float | None,
        typer.Option("--real-moe-download-budget-gib", min=0.001),
    ] = None,
    skip_k3_projection: Annotated[bool, typer.Option("--skip-k3-projection")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
    profile: Annotated[bool, typer.Option("--profile")] = False,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Run Experiment 006 intra-layer tensor and expert microsharding."""

    from swarm_inference.config.microsharding import load_microsharding_config
    from swarm_inference.experiments.microsharding import (
        MicroshardingOptions,
        run_microsharding_experiment,
    )

    try:
        experiment = load_microsharding_config(config)
        run = run_microsharding_experiment(
            experiment,
            requested_config_path=config,
            options=MicroshardingOptions(
                pipeline_stage_counts=_positive_csv(pipeline_stages, "--pipeline-stages"),
                tensor_parallel_degrees=_positive_csv(
                    tensor_parallel_degrees, "--tensor-parallel-degrees"
                ),
                dense_model=dense_model,
                dense_revision=dense_revision,
                skip_secondary_model=skip_secondary_model,
                skip_real_moe=skip_real_moe,
                real_moe_download_budget_gib=real_moe_download_budget_gib,
                skip_k3_projection=skip_k3_projection,
                resume=resume,
                smoke=smoke,
                profile=profile,
                output=output,
            ),
        )
    except (SwarmError, OSError, RuntimeError, ValueError) as exc:
        _fail(f"microsharding experiment failed: {exc}")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"status={run.summary['overall_status']}")
    if run.summary["overall_status"] in {"FAIL", "BLOCKED"}:
        raise typer.Exit(1)


@experiment_app.command("engine-performance")
def engine_performance_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 004 production-engine benchmark configuration.",
        ),
    ] = Path("configs/experiments/experiment_004_engine_performance.yaml"),
    primary_model: Annotated[str | None, typer.Option("--primary-model")] = None,
    secondary_model: Annotated[str | None, typer.Option("--secondary-model")] = None,
    skip_secondary: Annotated[bool, typer.Option("--skip-secondary")] = False,
    skip_optional_engines: Annotated[bool, typer.Option("--skip-optional-engines")] = False,
    output_root: Annotated[Path | None, typer.Option("--output-root", file_okay=False)] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
    profile: Annotated[bool, typer.Option("--profile")] = False,
    keep_servers: Annotated[bool, typer.Option("--keep-servers")] = False,
) -> None:
    """Run Experiment 004's isolated production-engine benchmark matrix."""

    from swarm_inference.config.engine_performance import (
        load_engine_performance_config,
    )
    from swarm_inference.experiments.engine_performance import (
        run_engine_performance_experiment,
    )

    try:
        experiment = load_engine_performance_config(config)
        run = run_engine_performance_experiment(
            experiment,
            config_path=config,
            primary_model=primary_model,
            secondary_model=secondary_model,
            skip_secondary=skip_secondary,
            skip_optional_engines=skip_optional_engines,
            output_root=output_root,
            resume=resume,
            smoke=smoke,
            profile=profile,
            keep_servers=keep_servers,
        )
    except (SwarmError, OSError, TimeoutError, ValueError) as exc:
        _fail(f"engine-performance experiment failed: {exc}")
    typer.echo("execution_mode=single-host-engine-benchmark")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"overall_status={run.summary['overall_status']}")
    if not run.passed:
        raise typer.Exit(1)


@experiment_app.command("heterogeneous-node-utility")
def heterogeneous_node_utility_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 007 YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_007_heterogeneous_node_utility.yaml"),
    experiment_004_run: Annotated[
        Path | None,
        typer.Option("--experiment-004-run", help="Override the validated Experiment 004 run."),
    ] = None,
    experiment_006_run: Annotated[
        Path | None,
        typer.Option("--experiment-006-run", help="Override the validated Experiment 006 run."),
    ] = None,
    skip_speculative: Annotated[bool, typer.Option("--skip-speculative")] = False,
    skip_moe: Annotated[bool, typer.Option("--skip-moe")] = False,
    skip_background: Annotated[bool, typer.Option("--skip-background")] = False,
    skip_arm64: Annotated[bool, typer.Option("--skip-arm64")] = False,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    profile: Annotated[bool, typer.Option("--profile")] = False,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Run Experiment 007 with real CUDA, x86 CPU, and isolated backends."""

    from swarm_inference.config.heterogeneous import load_heterogeneous_config
    from swarm_inference.experiments.heterogeneous_node_utility import (
        HeterogeneousOptions,
        run_heterogeneous_node_experiment,
    )

    try:
        experiment = load_heterogeneous_config(config)
        run = run_heterogeneous_node_experiment(
            experiment,
            requested_config_path=config,
            options=HeterogeneousOptions(
                experiment_004_run=experiment_004_run,
                experiment_006_run=experiment_006_run,
                skip_speculative=skip_speculative,
                skip_moe=skip_moe,
                skip_background=skip_background,
                skip_arm64=skip_arm64,
                smoke=smoke,
                resume=resume,
                profile=profile,
                output=output,
            ),
        )
    except (SwarmError, OSError, RuntimeError, ValueError) as exc:
        _fail(f"heterogeneous-node-utility experiment failed: {exc}")
    typer.echo(f"execution_mode={experiment.execution_mode}")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"status={run.summary['overall_status']}")
    if not run.passed:
        raise typer.Exit(1)


@experiment_app.command("experiment-007-corrections")
def experiment_007_corrections_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 007 benchmark-correction YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_007_corrections.yaml"),
    original_run: Annotated[
        Path | None,
        typer.Option("--original-run", help="Override the immutable original Experiment 007 run."),
    ] = None,
    skip_expert_fix: Annotated[bool, typer.Option("--skip-expert-fix")] = False,
    skip_background_fix: Annotated[bool, typer.Option("--skip-background-fix")] = False,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    profile: Annotated[bool, typer.Option("--profile")] = False,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    keep_servers: Annotated[bool, typer.Option("--keep-servers")] = False,
) -> None:
    """Correct Experiment 007 MoE and fixed-window background measurements."""

    from swarm_inference.config.experiment_007_corrections import (
        load_experiment_007_corrections_config,
    )
    from swarm_inference.experiments.experiment_007_corrections import (
        Experiment007CorrectionOptions,
        run_experiment_007_corrections,
    )

    try:
        experiment = load_experiment_007_corrections_config(config)
        run = run_experiment_007_corrections(
            experiment,
            requested_config_path=config,
            options=Experiment007CorrectionOptions(
                original_run=original_run,
                skip_expert_fix=skip_expert_fix,
                skip_background_fix=skip_background_fix,
                smoke=smoke,
                resume=resume,
                profile=profile,
                output_root=output_root,
                keep_servers=keep_servers,
            ),
        )
    except (SwarmError, OSError, RuntimeError, ValueError) as exc:
        _fail(f"Experiment 007 corrections failed: {exc}")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"status={run.summary['corrected_experiment_007_status']}")
    if not run.passed:
        raise typer.Exit(1)


@experiment_app.command("adaptive-moe-saturation")
def adaptive_moe_saturation_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 008 YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_008_adaptive_moe.yaml"),
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Exact local over-32-GiB GGUF file or directory."),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option("--output-directory", file_okay=False, help="Explicit run parent directory."),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    quick: Annotated[
        bool,
        typer.Option(
            "--quick", help="Run measured hardware and emulated software validation only."
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Run the real over-VRAM model and official workloads."),
    ] = False,
    skip_download: Annotated[
        bool,
        typer.Option("--skip-download", help="Use only supplied or already cached artifacts."),
    ] = False,
    configuration: Annotated[
        str | None,
        typer.Option(
            "--configuration", help="Run only A, B, C, D, E, F, or G in a resumed bundle."
        ),
    ] = None,
    server_path: Annotated[
        Path | None,
        typer.Option("--server-path", help="Exact native llama-server executable."),
    ] = None,
    gate_17_only: Annotated[
        bool,
        typer.Option(
            "--gate-17-only",
            help="Run only strict official preferred-model Configuration A evidence for Experiment 010 Gate 17.",
        ),
    ] = False,
) -> None:
    """Run Experiment 008 single-host adaptive sparse-MoE saturation."""

    from swarm_inference.experiments.experiment_008.runner import (
        Experiment008Options,
        run_experiment_008,
    )

    selected = configuration.upper() if configuration else None
    if selected is not None and selected not in set("ABCDEFG"):
        _fail("--configuration must be one of A, B, C, D, E, F, or G")
    try:
        outcome = run_experiment_008(
            Experiment008Options(
                config_path=config,
                model_path=model_path,
                output_directory=output_directory,
                resume=resume,
                quick=quick,
                full=full,
                skip_download=skip_download,
                configuration=selected,  # type: ignore[arg-type]
                server_path=server_path,
                gate_17_only=gate_17_only,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _fail(f"adaptive-moe-saturation failed before an evidence bundle could be finalized: {exc}")
    typer.echo("execution_mode=single-host-adaptive-moe-saturation")
    typer.echo(f"bundle={outcome.bundle_path}")
    typer.echo(f"report={outcome.bundle_path / 'report.md'}")
    typer.echo(f"verdict={outcome.verdict.value}")
    if outcome.error:
        typer.echo(f"terminal_error={outcome.error}", err=True)
    if outcome.error or (
        not gate_17_only
        and outcome.verdict.value not in {"PASS_STRONG", "PASS_CAPACITY_AND_ARCHITECTURE"}
    ):
        raise typer.Exit(1)


@experiment_app.command("colibri-adaptive-runtime")
def colibri_adaptive_runtime_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 009 Colibri YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_009_colibri.yaml"),
    colibri_path: Annotated[
        Path | None,
        typer.Option(
            "--colibri-path",
            help="Pinned Colibri checkout or built engine directory.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Converted Colibri practical-model directory."),
    ] = None,
    model_family: Annotated[
        str | None,
        typer.Option("--model-family", help="Explicit Colibri model family; never a fallback."),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(
            "--output-directory",
            file_okay=False,
            help="Explicit evidence-bundle directory, or existing bundle with --resume.",
        ),
    ] = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Run the generated real-engine fixture path."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Run the official practical Colibri MoE matrix."),
    ] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    rebuild_colibri: Annotated[bool, typer.Option("--rebuild-colibri")] = False,
    apply_bridge_patches: Annotated[bool, typer.Option("--apply-bridge-patches")] = False,
    telemetry_level: Annotated[
        str | None,
        typer.Option(
            "--telemetry-level",
            help="Bridge telemetry level: off, summary, detailed, or trace.",
        ),
    ] = None,
    configuration: Annotated[
        str | None,
        typer.Option("--configuration", help="Run only A, B, C, D, or E."),
    ] = None,
    skip_model_download: Annotated[
        bool,
        typer.Option("--skip-model-download", help="Require a pre-provisioned practical model."),
    ] = False,
) -> None:
    """Run Experiment 009's Colibri-backed adaptive expert runtime."""

    from swarm_inference.experiments.experiment_009.runner import (
        Experiment009Options,
        run_experiment_009,
    )

    selected = configuration.upper() if configuration else None
    if selected is not None and selected not in set("ABCDE"):
        _fail("--configuration must be one of A, B, C, D, or E")
    try:
        outcome = run_experiment_009(
            Experiment009Options(
                config_path=config,
                colibri_path=colibri_path,
                model_path=model_path,
                model_family=model_family,
                output_directory=output_directory,
                quick=quick,
                full=full,
                resume=resume,
                rebuild_colibri=rebuild_colibri,
                apply_bridge_patches=apply_bridge_patches,
                telemetry_level=telemetry_level,
                configuration=selected,  # type: ignore[arg-type]
                skip_model_download=skip_model_download,
            )
        )
    except (SwarmError, OSError, RuntimeError, ValueError) as exc:
        _fail(f"colibri-adaptive-runtime failed before finalizing evidence: {exc}")
    typer.echo("execution_mode=colibri-adaptive-expert-runtime")
    typer.echo(f"bundle={outcome.bundle_path}")
    typer.echo(f"report={outcome.bundle_path / 'report.md'}")
    typer.echo(f"verdict={outcome.verdict.value}")
    if outcome.error:
        typer.echo(f"terminal_error={outcome.error}", err=True)
    if outcome.verdict.value not in {"PASS_STRONG", "PASS_INTEGRATION"}:
        raise typer.Exit(1)


@experiment_app.command("worker-fanout")
def worker_fanout_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 003 worker-fanout YAML configuration.",
        ),
    ] = Path("configs/experiments/experiment_003_worker_fanout.yaml"),
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    worker_counts: Annotated[
        str | None,
        typer.Option(
            "--worker-counts",
            help="Comma-separated initial worker counts.",
        ),
    ] = None,
    repeats: Annotated[int | None, typer.Option("--repeats", min=1)] = None,
    max_worker_count: Annotated[
        int | None,
        typer.Option("--max-worker-count", min=1, max=28),
    ] = None,
    skip_acquisition_tests: Annotated[
        bool,
        typer.Option("--skip-acquisition-tests"),
    ] = False,
    skip_rejoin_test: Annotated[
        bool,
        typer.Option("--skip-rejoin-test"),
    ] = False,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    output: Annotated[
        Path,
        typer.Option("--output", help="Run root, or existing run directory with --resume."),
    ] = Path("artifacts/runs"),
    profile: Annotated[bool, typer.Option("--profile")] = False,
    smoke: Annotated[bool, typer.Option("--smoke")] = False,
) -> None:
    """Run real Qwen3 worker fan-out, lifecycle, acquisition, rejoin, and economics."""

    parsed_counts: list[int] | None = None
    if worker_counts is not None:
        try:
            parsed_counts = [
                int(value.strip()) for value in worker_counts.split(",") if value.strip()
            ]
        except ValueError as exc:
            _fail(f"--worker-counts must be comma-separated integers: {exc}")
        if not parsed_counts:
            _fail("--worker-counts cannot be empty")
    from swarm_inference.experiments.worker_fanout import (
        run_worker_fanout_experiment,
    )

    try:
        run = run_worker_fanout_experiment(
            config_path=config,
            model_id=model_id,
            revision=revision,
            worker_counts=parsed_counts,
            repeats=repeats,
            max_worker_count=max_worker_count,
            skip_acquisition_tests=skip_acquisition_tests,
            skip_rejoin_test=skip_rejoin_test,
            resume=resume,
            output=output,
            profile=profile,
            smoke=smoke,
        )
    except (SwarmError, OSError, TimeoutError, ValueError) as exc:
        _fail(f"worker-fanout experiment failed: {exc}")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"overall_status={run.summary['overall_status']}")
    if not run.passed:
        raise typer.Exit(1)


@app.command("coordinator")
def coordinator_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    listen: Annotated[str, typer.Option(help="gRPC bind endpoint.")] = "0.0.0.0:50051",
    model_manifest: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Verified real-model manifest.json; omit for synthetic execution.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Resolved model snapshot containing config and tokenizer metadata only.",
        ),
    ] = None,
    dtype: Annotated[str | None, typer.Option(help="Real-model execution dtype.")] = None,
) -> None:
    """Start the central coordinator without loading full model weights."""

    from swarm_inference.config.models import ModelManifest
    from swarm_inference.coordinator.service import CoordinatorCore, CoordinatorRpcServer

    experiment = load_experiment_config(config)
    manifest = None
    architecture_config = None
    tokenizer = None
    if (model_manifest is None) != (model_path is None):
        _fail("--model-manifest and --model-path must be supplied together")
    if model_manifest is not None and model_path is not None:
        manifest = ModelManifest.model_validate_json(model_manifest.read_text(encoding="utf-8"))
        architecture_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            model_path,
            local_files_only=True,
        )

    async def run() -> None:
        core = CoordinatorCore(
            config=experiment,
            model_manifest=manifest,
            architecture_config=architecture_config,
            runtime_dtype=dtype,
            tokenizer=tokenizer,
        )
        server = CoordinatorRpcServer(core)
        await server.start(listen)
        typer.echo(
            f"coordinator listening on {listen}; execution_mode={experiment.execution_mode.value}; "
            f"model={core.runtime_model_id}@{core.runtime_model_revision}"
        )
        try:
            await server.wait_for_termination()
        finally:
            await server.stop()

    asyncio.run(run())


@app.command("worker")
def worker_command(
    coordinator: Annotated[str, typer.Option(help="Coordinator host:port.")],
    backend: Annotated[Backend, typer.Option(help="Execution backend.")],
    memory_limit_gb: Annotated[float, typer.Option(help="Enforced logical RAM/VRAM limit in GiB.")],
    listen: Annotated[str, typer.Option(help="Worker gRPC bind endpoint.")] = "0.0.0.0:50052",
    advertise: Annotated[
        str | None,
        typer.Option(
            help=(
                "Coordinator-reachable worker endpoint. When omitted, derive the routed "
                "local address and listen port; wildcard addresses are never advertised."
            )
        ),
    ] = None,
    identity: Annotated[Path, typer.Option(help="Persistent Ed25519 private key.")] = Path(
        ".swarm/worker-identity.pem"
    ),
    worker_id: Annotated[str | None, typer.Option()] = None,
    model_shard_root: Annotated[
        Path | None,
        typer.Option(help="Local root containing pre-provisioned real-model stage directories."),
    ] = None,
) -> None:
    """Start a physical or loopback worker using the same transport and agent."""

    from swarm_inference.worker.service import run_worker

    if memory_limit_gb <= 0 or memory_limit_gb > 32:
        _fail("memory-limit-gb must be in (0, 32]")
    try:
        advertised_endpoint = resolve_advertised_endpoint(
            listen_endpoint=listen,
            coordinator_endpoint=coordinator,
            explicit_endpoint=advertise,
        )
    except SwarmError as exc:
        _fail(f"invalid worker endpoint configuration: {exc}")
    asyncio.run(
        run_worker(
            coordinator_endpoint=coordinator,
            listen_endpoint=listen,
            advertised_endpoint=advertised_endpoint,
            backend=backend,
            memory_limit_bytes=int(memory_limit_gb * 1024**3),
            identity_path=identity,
            worker_id=worker_id,
            model_shard_root=model_shard_root,
            queue_config=QueueConfig(),
        )
    )


@app.command("submit")
def submit_command(
    coordinator: Annotated[str, typer.Option(help="Coordinator host:port.")],
    prompt: Annotated[str, typer.Option(help="Prompt text tokenised by the coordinator.")],
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 16,
    temperature: Annotated[float, typer.Option(min=0)] = 0.0,
    seed: Annotated[int, typer.Option()] = 1,
    model_id: Annotated[str, typer.Option(help="Expected model identifier.")] = "synthetic",
    model_revision: Annotated[
        str,
        typer.Option(help="Expected immutable model revision."),
    ] = "synthetic-v1",
) -> None:
    """Submit one request; output is never described as aggregate throughput."""

    if temperature != 0:
        _fail("the initial transport runtime supports greedy temperature=0 only")
    from swarm_inference.coordinator.service import CoordinatorClient
    from swarm_inference.protocol.messages import SubmitRequest

    async def run() -> Any:
        client = CoordinatorClient(coordinator)
        try:
            return await client.submit(
                SubmitRequest(
                    request_id=uuid4().hex,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    random_seed=seed,
                    model_id=model_id,
                    model_revision=model_revision,
                )
            )
        finally:
            await client.close()

    response = asyncio.run(run())
    typer.echo(response.model_dump_json(indent=2))
    if response.status != "completed" or not response.verified:
        raise typer.Exit(1)


@app.command("validate-model")
def validate_model_command(
    shards: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    model_path: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    prompt: Annotated[str, typer.Option()] = "Explain deterministic simulation.",
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 4,
    device: Annotated[str, typer.Option()] = "cpu",
    dtype: Annotated[str, typer.Option()] = "float32",
    atol: Annotated[float, typer.Option(min=0)] = 1e-5,
    rtol: Annotated[float, typer.Option(min=0)] = 1e-4,
    distributed_loopback_workers: Annotated[
        int,
        typer.Option(
            "--distributed-workers",
            "--distributed-loopback-workers",
            min=0,
            help="Also run this many process-isolated stage workers over gRPC; 0 disables.",
        ),
    ] = 0,
    distributed_config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Queue/runtime settings for the optional real-model loopback phase.",
        ),
    ] = Path("configs/experiments/scaling_loopback.yaml"),
    distributed_backend: Annotated[
        Backend,
        typer.Option(
            help="Backend used by process-isolated stage workers in the distributed phase."
        ),
    ] = Backend.TORCH_CPU,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the complete correctness payload."),
    ] = False,
) -> None:
    """Compare split dense Qwen3 against a separate unsplit reference process."""

    distributed_result: dict[str, Any] | None = None
    try:
        if distributed_loopback_workers:
            if distributed_backend == Backend.SYNTHETIC:
                _fail(
                    "validate-model distributed workers require torch-cpu, torch-cuda, or torch-mps"
                )
            from swarm_inference.config.models import ModelManifest
            from swarm_inference.experiments.real_model import run_qwen3_process_loopback

            manifest = ModelManifest.model_validate_json(
                (shards / "manifest.json").read_text(encoding="utf-8")
            )
            architecture_config = json.loads(
                (model_path / "config.json").read_text(encoding="utf-8")
            )
            distributed_result = asyncio.run(
                run_qwen3_process_loopback(
                    experiment=load_experiment_config(distributed_config),
                    manifest=manifest,
                    architecture_config=architecture_config,
                    shard_root=shards.resolve(),
                    model_path=model_path.resolve(),
                    output_dir=output,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    dtype=dtype,
                    worker_count=distributed_loopback_workers,
                    worker_backend=distributed_backend,
                )
            )
        result = validate_qwen_correctness(
            shard_root=shards,
            model_path=model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            dtype_name=dtype,
            atol=atol,
            rtol=rtol,
            output_dir=output,
        )
    except (SwarmError, OSError, TimeoutError, ValueError) as exc:
        _fail(f"validate-model failed: {exc}")
    payload = result.to_dict()
    if distributed_result is not None:
        distributed_identity = (
            distributed_result["output_token_ids"] == payload["reference_token_ids"]
        )
        distributed_result["greedy_token_identity"] = distributed_identity
        distributed_result["passed"] = bool(distributed_result["passed"] and distributed_identity)
        payload["distributed_loopback"] = distributed_result
        payload["passed"] = bool(payload["passed"] and distributed_result["passed"])
        (output / "distributed-loopback.json").write_text(
            json.dumps(distributed_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "correctness.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"status={'PASS' if payload['passed'] else 'FAIL'}")
        typer.echo(f"correctness_artifact={(output / 'correctness.json').resolve()}")
        typer.echo(f"reference_token_ids={payload['reference_token_ids']}")
        typer.echo(f"split_token_ids={payload['distributed_token_ids']}")
        if distributed_result is not None:
            typer.echo("execution_mode=single-host-loopback")
            typer.echo(f"worker_backend={distributed_result['worker_backend']}")
            typer.echo(f"worker_count={distributed_result['worker_count']}")
            typer.echo(f"process_worker_token_ids={distributed_result['output_token_ids']}")
    if not payload["passed"]:
        raise typer.Exit(1)


@app.command("real-experiment")
def real_experiment_command(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Experiment 002 real-model configuration.",
        ),
    ] = Path("configs/experiments/experiment_002_qwen3_real_loopback.yaml"),
    model_id: Annotated[str | None, typer.Option("--model-id")] = None,
    revision: Annotated[str | None, typer.Option()] = None,
    max_new_tokens: Annotated[
        int | None,
        typer.Option(min=4),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(),
    ] = Path("artifacts/runs"),
    skip_download: Annotated[bool, typer.Option()] = False,
    skip_sharding: Annotated[bool, typer.Option()] = False,
    skip_prompt_suite: Annotated[bool, typer.Option()] = False,
    skip_replay_test: Annotated[bool, typer.Option()] = False,
    keep_workers: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the complete process-isolated real Qwen3 Experiment 002."""

    from swarm_inference.experiments.experiment_002 import run_experiment_002

    try:
        run = run_experiment_002(
            config_path=config,
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
            output_root=output_root,
            skip_download=skip_download,
            skip_sharding=skip_sharding,
            skip_prompt_suite=skip_prompt_suite,
            skip_replay_test=skip_replay_test,
            keep_workers=keep_workers,
        )
    except (SwarmError, OSError, TimeoutError, ValueError) as exc:
        _fail(f"real-experiment failed: {exc}")
    typer.echo(f"run_directory={run.run_directory}")
    typer.echo(f"report={run.report_path}")
    typer.echo(f"overall_status={run.summary['overall_status']}")
    if not run.passed:
        raise typer.Exit(1)


@app.command("analyse-large-model")
def analyse_large_model_command(
    model: Annotated[str, typer.Option(help="Model ID or local checkpoint path.")],
    output: Annotated[Path, typer.Option(help="Feasibility JSON output.")],
    revision: Annotated[str | None, typer.Option()] = None,
    node_memory_gb: Annotated[float, typer.Option(min=1, max=32)] = 32,
    safety_fraction: Annotated[float, typer.Option(min=0.1, max=1)] = 0.8,
    allow_download: Annotated[bool, typer.Option()] = False,
) -> None:
    """Analyse a K3-scale config/index without loading or claiming model support."""

    try:
        resolved = resolve_model(
            model,
            revision=revision,
            allow_download=allow_download,
        )
        report = analyse_large_model_manifest(
            resolved,
            node_memory_bytes=int(node_memory_gb * 1024**3),
            safety_fraction=safety_fraction,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (SwarmError, OSError, ValueError) as exc:
        _fail(f"analyse-large-model failed: {exc}")
    typer.echo(f"claim_of_model_support=false\nreport={output.resolve()}")


@app.command("report")
def report_command(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Regenerate the self-contained HTML report from a run directory."""

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    requests = [
        json.loads(line)
        for line in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with (run / "scaling.csv").open(encoding="utf-8", newline="") as handle:
        scaling = list(csv.DictReader(handle))
    report = render_html_report(
        run_dir=run.resolve(),
        summary=summary,
        scaling_rows=scaling,
        request_rows=requests,
    )
    typer.echo(str(report))


@app.command("validate-run")
def validate_run_command(
    run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    """Verify the required artifact set and recorded SHA-256 hashes."""

    errors = validate_run(run)
    if errors:
        _fail("\n".join(errors))
    typer.echo(f"PASS: {run.resolve()}")


if __name__ == "__main__":
    app()
