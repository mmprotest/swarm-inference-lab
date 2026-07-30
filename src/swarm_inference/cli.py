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


@app.command("experiment")
def experiment_command(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
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
