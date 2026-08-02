"""Integrated, resumable Experiment 008 runner.

Only the full path can contribute to official gates.  The quick path exercises
real tensor kernels and all artifact plumbing, but labels fixture evidence as
EMULATED and never promotes it to target-model performance evidence.
"""

from __future__ import annotations

import gc
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from swarm_inference.config.experiment_008 import (
    Experiment008Config,
    Experiment008ModelCandidate,
    load_experiment_008_config,
)
from swarm_inference.experiments.experiment_008.acquisition import (
    ResolvedModel,
    resolve_llama_server,
    resolve_model_candidate,
)
from swarm_inference.experiments.experiment_008.analysis import (
    build_ablation_rows,
    evaluate_gates,
)
from swarm_inference.experiments.experiment_008.backend import (
    GenerationResult,
    LlamaCppClient,
    file_sha256,
    free_local_port,
    launch_llama_server,
    probe_llama_server,
)
from swarm_inference.experiments.experiment_008.benchmark import (
    WorkloadExecution,
    execute_mixed_service,
    execute_prompt_batch,
)
from swarm_inference.experiments.experiment_008.bundle import (
    EvidenceBundle,
    create_bundle_root,
)
from swarm_inference.experiments.experiment_008.cost_model import (
    MeasuredCostModel,
    MeasuredKernelPoint,
    TransferPoint,
    planner_regret_fraction,
    prediction_quality,
)
from swarm_inference.experiments.experiment_008.experts import (
    ExpertActivation,
    activation_statistics,
)
from swarm_inference.experiments.experiment_008.fixture import validate_tiny_moe_fixture
from swarm_inference.experiments.experiment_008.gguf import (
    GGUFInventory,
    build_preflight,
    build_tensor_tiles,
    inspect_gguf,
    tensor_layer,
    tensor_role,
)
from swarm_inference.experiments.experiment_008.hardware import (
    build_hardware_profile,
    collect_hardware_identity,
    profile_fingerprint,
)
from swarm_inference.experiments.experiment_008.planning import (
    BackendCapabilities,
    BaselineCandidate,
    baseline_search_space,
    build_phase_plan,
    select_best_stock_by_workload,
)
from swarm_inference.experiments.experiment_008.reporting import (
    build_bundle_readme,
    build_report,
    generate_required_plots,
)
from swarm_inference.experiments.experiment_008.schemas import (
    BenchmarkObservation,
    EvidenceClass,
    ExecutionStatus,
    Experiment008Verdict,
    GateStatus,
    PhasePlan,
    TensorTile,
    overall_verdict,
)
from swarm_inference.experiments.experiment_008.workloads import (
    WorkloadPrompt,
    build_decode_workload,
    build_long_context_workload,
)

ConfigurationId = Literal["A", "B", "C", "D", "E", "F", "G"]


@dataclass(slots=True)
class Experiment008Options:
    config_path: Path
    model_path: Path | None = None
    output_directory: Path | None = None
    resume: bool = False
    quick: bool = False
    full: bool = False
    skip_download: bool = False
    configuration: ConfigurationId | None = None
    server_path: Path | None = None

    def validate(self) -> None:
        if self.quick == self.full:
            raise ValueError("select exactly one of --quick or --full")
        if self.configuration is not None and self.configuration not in set("ABCDEFG"):
            raise ValueError("configuration must be one of A through G")


@dataclass(slots=True)
class Experiment008Outcome:
    bundle_path: Path
    verdict: Experiment008Verdict
    completed: bool
    error: str | None


def _json_or(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _git_result(root: Path, arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _environment(repository_root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    gpu = identity.get("gpu")
    return {
        "classification": "MEASURED",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "operating_system": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": identity.get("pytorch"),
        "cuda_runtime": identity.get("cuda_runtime"),
        "gpu": gpu.get("name") if isinstance(gpu, dict) else None,
        "gpu_driver": gpu.get("driver") if isinstance(gpu, dict) else None,
        "cpu": identity.get("cpu"),
        "physical_cpu_cores": identity.get("physical_cpu_cores"),
        "logical_cpu_cores": identity.get("logical_cpu_cores"),
        "system_ram_bytes": identity.get("system_ram_bytes"),
        "hardware_fingerprint": identity.get("fingerprint"),
        "git_commit": _git_result(repository_root, ["rev-parse", "HEAD"]),
        "git_status_porcelain": _git_result(repository_root, ["status", "--porcelain"]),
        "command": list(sys.argv),
    }


def _fixture_tiles() -> list[TensorTile]:
    rows: list[TensorTile] = []
    for role, name, size in (
        ("embedding", "token_embd.weight", 4096),
        ("router", "blk.0.ffn_gate_inp.weight", 512),
        ("routed_expert_up_projection", "blk.0.ffn_up_exps.weight", 24_576),
        ("routed_expert_gate_projection", "blk.0.ffn_gate_exps.weight", 24_576),
        ("routed_expert_down_projection", "blk.0.ffn_down_exps.weight", 24_576),
        ("output_head", "output.weight", 4096),
    ):
        rows.append(
            TensorTile(
                model_id="experiment-008-tiny-moe-fixture",
                model_revision="seed-8008",
                layer_id=0 if name.startswith("blk") else -1,
                tensor_name=name,
                tensor_role=role,
                expert_id=None,
                logical_shape=[32, max(size // 128, 1)],
                logical_slice={"kind": "full_tensor"},
                physical_layout="PyTorch contiguous fixture tensor",
                dtype="float32",
                quantization="none",
                quantization_metadata={},
                accumulator_dtype="float32",
                byte_size=size,
                content_hash=f"EMULATED:seed-8008:{name}",
                allowed_backends=["torch-fixture"],
                current_residency="CPU",
                planned_execution_device="CPU",
            )
        )
    return rows


def _empty_observations(
    reason: str, *, evidence: EvidenceClass | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for configuration in "ABCDEFG":
        for workload in ("decode", "prefill_8k", "prefill_32k", "mixed"):
            rows.append(
                BenchmarkObservation(
                    configuration=configuration,  # type: ignore[arg-type]
                    workload=workload,  # type: ignore[arg-type]
                    plan_id=f"{configuration.lower()}-{workload}",
                    status=ExecutionStatus.NOT_RUN,
                    evidence_class=evidence,
                    metrics={
                        "decode_tokens_per_second": None,
                        "time_to_first_token_ms": None,
                        "mixed_verified_tokens_per_second": None,
                        "interactive_p95_latency_ms": None,
                        "peak_vram_bytes": None,
                        "peak_system_ram_bytes": None,
                        "pcie_bytes_per_output_token": None,
                        "cpu_gpu_overlap_percent": None,
                        "expert_cache_hit_rate": None,
                        "useful_prefetch_rate": None,
                    },
                    unavailable_reason=reason,
                ).model_dump(mode="json")
            )
    return rows


def _write_benchmark_csv(bundle: EvidenceBundle, observations: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for observation in observations:
        row = {key: value for key, value in observation.items() if key != "metrics"}
        metrics = observation.get("metrics")
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    bundle.write_csv("benchmark_results.csv", rows)


def _quick_run(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    repository_root: Path,
) -> None:
    tiles = _fixture_tiles()
    bundle.write_json(
        "model_preflight.json",
        {
            "classification": "EMULATED",
            "status": "NOT_RUN",
            "model_id": "experiment-008-tiny-moe-fixture",
            "model_revision": "seed-8008",
            "model_architecture": "tiny-moe-fixture",
            "quantization_format": "none",
            "total_tensor_bytes": sum(tile.byte_size for tile in tiles),
            "total_expert_bytes": sum(
                tile.byte_size for tile in tiles if tile.tensor_role.startswith("routed_expert")
            ),
            "layer_count": 1,
            "routed_expert_count": 6,
            "experts_selected_per_token": 2,
            "shared_expert_count": 0,
            "system_ram_required_bytes": None,
            "system_ram_available_bytes": None,
            "physical_vram_bytes": None,
            "backend_selected": "torch-fixture",
            "backend_limitations": [
                "fixture is not the target model and cannot satisfy any official capacity or performance gate"
            ],
            "genuinely_exceeds_32gb": False,
            "genuinely_exceeds_physical_vram": False,
            "eligible": False,
            "rejection_reasons": ["quick fixture is below the required model capacity"],
        },
    )
    bundle.write_json(
        "tensor_inventory.json",
        {
            "classification": "EMULATED",
            "tensor_count": len(tiles),
            "tensor_bytes": sum(tile.byte_size for tile in tiles),
            "tensors": [
                {"name": tile.tensor_name, "shape": tile.logical_shape, "byte_size": tile.byte_size}
                for tile in tiles
            ],
        },
    )
    bundle.write_json(
        "tensor_tiles.json",
        {
            "classification": "EMULATED",
            "scope": "tiny fixture only",
            "tiles": [tile.model_dump(mode="json") for tile in tiles],
            "expert_microshards": [],
        },
    )
    bundle.complete_stage("fixture-inventory")

    identity = collect_hardware_identity(
        backend="torch-fixture", model="tiny-moe", quantization="none"
    )
    bundle.write_json("environment.json", _environment(repository_root, identity))
    try:
        profile = build_hardware_profile(
            backend="torch-fixture",
            model="tiny-moe",
            quantization="none",
            model_path=None,
            decode_shapes=config.profiling.decode_shapes[:1],
            prefill_shapes=config.profiling.prefill_shapes[:1],
            cpu_thread_counts=config.profiling.cpu_thread_counts[:2],
            payload_bytes=config.profiling.payload_bytes[:3],
            warmups=1,
            iterations=3,
            storage_sample_bytes=config.profiling.storage_sample_bytes,
            trace_path=bundle.root / "profiler_trace" / "quick_torch_overlap.json",
            quick=True,
        )
    except Exception as exc:
        profile = {
            "classification": "MEASURED",
            "status": "FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "identity": identity,
        }
        bundle.record_failure(stage="quick-hardware-profile", error=profile["reason"])
    bundle.write_json("hardware_profile.json", profile)
    bundle.complete_stage("hardware-profile")

    fixture = validate_tiny_moe_fixture(seed=config.workloads.seed)
    selected = fixture.get("selected_expert_ids", [])
    events = [
        ExpertActivation(
            token_index=index,
            layer_id=0,
            phase="prefill" if index == 0 else "decode",
            expert_ids=[int(value) for value in experts],
        )
        for index, experts in enumerate(selected)
    ]
    statistics, coactivation = activation_statistics(events)
    activation_rows = [
        {"classification": "EMULATED", **row.model_dump(mode="json")} for row in statistics
    ]
    bundle.write_csv("expert_activation_matrix.csv", activation_rows)
    bundle.write_csv(
        "expert_coactivation.csv",
        [{"classification": "EMULATED", **row} for row in coactivation],
    )
    bundle.write_json(
        "expert_trace_summary.json",
        {
            "classification": "EMULATED",
            "status": "COMPLETED",
            "scope": "tiny fixture routing only; excluded from official verdict",
            "activation_event_count": len(events),
            "expert_statistics": activation_rows,
            "gpu_cached_experts": None,
            "useful_prefetch_bytes": None,
            "wasted_prefetch_bytes": None,
            "visible_transfer_latency_removed_ms": None,
            "prediction_conclusion": "not measured on the target model",
        },
    )

    capabilities = BackendCapabilities(
        conventional_layer_offload=True,
        tensor_buffer_override=True,
        cpu_moe=True,
        asynchronous_backend_scheduler=False,
        operation_level_overlap_trace=False,
        expert_routing_trace=True,
        per_expert_dynamic_residency=True,
        expert_prefetch=True,
        separate_process_phase_plans=True,
        in_request_phase_switch=True,
        deterministic_greedy_tokens=True,
        final_logits=True,
        limitations=["capabilities describe fixture algorithms, not the target backend"],
    )
    baseline_candidates = baseline_search_space(config.baseline_search, seed=config.workloads.seed)
    bundle.write_json(
        "baseline_search.json",
        {
            "classification": "EMULATED",
            "status": "NOT_RUN",
            "reason": "quick mode does not benchmark target-model offloading",
            "search_space": config.baseline_search.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in baseline_candidates],
            "results": [],
            "selected_by_workload": {},
        },
    )
    plans: list[PhasePlan] = []
    for configuration in "ABCDEFG":
        for phase in ("prefill", "decode", "mixed"):
            plans.append(
                build_phase_plan(
                    configuration=configuration,  # type: ignore[arg-type]
                    phase=phase,  # type: ignore[arg-type]
                    capabilities=capabilities,
                    tiles=tiles,
                    stock_arguments=[],
                    cpu_moe_layers=1,
                    measured_utility_by_technique={},
                )
            )
    bundle.write_json(
        "candidate_plans.json",
        {
            "classification": "EMULATED",
            "plans": [plan.model_dump(mode="json") for plan in plans],
        },
    )
    g_prefill = next(
        plan for plan in plans if plan.configuration == "G" and plan.phase == "prefill"
    )
    g_decode = next(plan for plan in plans if plan.configuration == "G" and plan.phase == "decode")
    bundle.write_json("prefill_plan.json", g_prefill.model_dump(mode="json"))
    bundle.write_json("decode_plan.json", g_decode.model_dump(mode="json"))
    bundle.write_json(
        "adaptive_plan.json",
        {
            "classification": "EMULATED",
            "prefill_plan": g_prefill.model_dump(mode="json"),
            "decode_plan": g_decode.model_dump(mode="json"),
            "prefill_decode_plans_differ": False,
            "technique_decisions": [item.model_dump(mode="json") for item in g_decode.techniques],
        },
    )
    observations = _empty_observations(
        "quick mode validates software with a tiny fixture and does not run target-model workloads"
    )
    _write_benchmark_csv(bundle, observations)
    bundle.write_json("benchmark_results.json", observations)
    bundle.write_csv(
        "cost_model_predictions.csv",
        [
            {
                "classification": "PROJECTED",
                "predicted_evidence_class": "PROJECTED",
                "measured_evidence_class": None,
                "status": "NOT_RUN",
                "plan_id": None,
                "predicted_ms": None,
                "measured_ms": None,
                "reason": "target model was not loaded in quick mode",
            }
        ],
    )
    identity_by_configuration = {configuration: None for configuration in "ABCDEFG"}
    ablations = build_ablation_rows(
        observations, token_identity_by_configuration=identity_by_configuration
    )
    bundle.write_csv("ablation_results.csv", ablations)
    correctness = {
        "classification": "EMULATED",
        "deterministic_execution_count": 1,
        "token_identity_rate": None,
        "fixture_checks_passed": bool(fixture.get("passed")),
        "fixture_results": fixture,
        "checks": {
            "tensor_tile_reconstruction": "covered by unit test",
            "expert_microshard_equivalence": fixture["checks"]["expert_microshard_equivalence"],
            "cpu_gpu_split_equivalence": fixture.get("tensor_runtime", {}).get(
                "cpu_gpu_split_equivalence"
            ),
            "cache_hit_and_miss_equivalence": fixture["checks"]["cache_hit_and_miss_equivalence"],
            "prefetch_enabled_disabled_equivalence": fixture["checks"][
                "prefetch_enabled_disabled_equivalence"
            ],
            "separate_prefill_decode_plan_equivalence": "not a target-model execution",
            "end_to_end_greedy_token_comparison": "not run",
        },
        "final_logits_limitation": "target backend was not loaded",
    }
    bundle.write_json("correctness_results.json", correctness)
    residency = {
        "classification": "EMULATED",
        "status": "NOT_RUN",
        "planned_gpu_tensor_bytes": None,
        "planned_cpu_or_mapped_tensor_bytes": None,
        "planned_gpu_tensor_roles": [],
        "planned_cpu_tensor_roles": [],
        "split_tensors": [],
        "reconciled": False,
        "system_ram_contributes": False,
        "no_complete_gpu_duplicate": False,
        "positive_cpu_performance_utility": False,
        "reason": "quick fixture cannot establish target-model residency",
    }
    bundle.write_json("residency_accounting.json", residency)
    bundle.write_csv(
        "resource_timeseries.csv",
        [
            {
                "classification": "MEASURED",
                "status": "INCOMPLETE",
                "reason": "quick kernel profile is in hardware_profile.json; no target workload ran",
            }
        ],
    )
    bundle.complete_stage("quick-validation")


def _physical_vram_bytes(identity: dict[str, Any]) -> int:
    gpu = identity.get("gpu")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("memory_total_mib"), (int, float)):
        return 0
    return int(float(gpu["memory_total_mib"]) * 1024**2)


def _model_execution_precheck(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    executable: Path,
    resolved: ResolvedModel,
    candidate_name: str,
    capabilities: BackendCapabilities,
) -> dict[str, Any]:
    """Prove backend compatibility before committing to preferred/fallback selection."""

    previous_preflight = _json_or(bundle.root / "model_preflight.json", {})
    previous_baseline = _json_or(bundle.root / "baseline_search.json", {})
    completed_baseline = next(
        (
            row
            for row in previous_baseline.get("results", [])
            if isinstance(row, dict)
            and row.get("status") == "COMPLETED"
            and row.get("classification") == "MEASURED"
        ),
        None,
    )
    if (
        previous_preflight.get("model_file_sha256") == resolved.file_sha256
        and completed_baseline is not None
    ):
        payload = {
            "classification": "MEASURED",
            "status": "COMPLETED",
            "candidate": candidate_name,
            "model_file_sha256": resolved.file_sha256,
            "reused": True,
            "proof_source": "completed real generation in baseline_search.json",
            "source_candidate_id": completed_baseline.get("candidate_id"),
            "source_workload": completed_baseline.get("workload"),
            "exit_code": completed_baseline.get("exit_code"),
        }
        bundle.write_json("model_execution_precheck.json", payload)
        return payload

    candidate = baseline_search_space(config.baseline_search, seed=config.workloads.seed)[0]
    log_root = bundle.root / "logs" / "model-eligibility" / candidate_name
    try:
        server = launch_llama_server(
            executable=executable,
            model_path=Path(resolved.path),
            host=config.backend.host,
            port=free_local_port(config.backend.host, start=config.backend.port_start),
            context_size=512,
            parallel=1,
            plan_arguments=_safe_backend_arguments(candidate.backend_arguments, capabilities),
            logs=log_root,
            startup_timeout_seconds=config.backend.startup_timeout_seconds,
            keep=False,
        )
    except Exception as exc:
        payload = {
            "classification": "MEASURED",
            "status": "FAILED",
            "candidate": candidate_name,
            "model_file_sha256": resolved.file_sha256,
            "reused": False,
            "error": f"{type(exc).__name__}: {exc}",
            "exit_code": getattr(exc, "returncode", None),
        }
        bundle.write_json("model_execution_precheck.json", payload)
        return payload

    client = LlamaCppClient(server.endpoint, timeout_seconds=config.backend.request_timeout_seconds)
    generation: GenerationResult | None = None
    error: str | None = None
    try:
        prompt_ids = client.tokenize(
            "Experiment 008 backend compatibility check. Reply with one short factual sentence.",
            add_special=True,
        )
        generation = client.generate(prompt_ids, output_tokens=2, seed=config.workloads.seed)
        if not generation.success:
            error = generation.error or "greedy generation did not complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        backend_was_running = server.process.poll() is None
        process_exit_code = server.close()
        bundle.write_json(
            str((log_root / "server.exit.json").relative_to(bundle.root)),
            {
                "exit_code": process_exit_code,
                "shutdown_mode": (
                    "controlled_termination_after_model_precheck"
                    if backend_was_running
                    else "backend_exited_before_cleanup"
                ),
                "precheck_exit_code": 0 if error is None else process_exit_code,
            },
        )
    payload = {
        "classification": "MEASURED",
        "status": "COMPLETED" if error is None else "FAILED",
        "candidate": candidate_name,
        "model_file_sha256": resolved.file_sha256,
        "reused": False,
        "output_token_ids": generation.output_token_ids if generation is not None else None,
        "error": error,
        "exit_code": 0 if error is None else process_exit_code,
        "server_process_exit_code": process_exit_code,
    }
    bundle.write_json("model_execution_precheck.json", payload)
    return payload


def _resolve_and_preflight(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    options: Experiment008Options,
    capabilities: BackendCapabilities,
    identity: dict[str, Any],
    cache_dir: Path,
    executable: Path,
) -> tuple[ResolvedModel, GGUFInventory, dict[str, Any]]:
    import psutil

    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[str, Experiment008ModelCandidate]]
    if options.model_path is not None:
        lowered = options.model_path.name.lower()
        selected = config.models.fallback if "3.6-35b" in lowered else config.models.preferred
        selected_name = "fallback" if selected is config.models.fallback else "preferred"
        candidates = [(selected_name, selected)]
    else:
        candidates = [("preferred", config.models.preferred), ("fallback", config.models.fallback)]
    last_error = "no model candidate was attempted"
    for candidate_name, candidate in candidates:
        attempt: dict[str, Any] = {
            "candidate": candidate_name,
            "model_id": candidate.model_id,
            "artifact_repository": candidate.artifact_repository,
            "filename": candidate.filename,
            "started_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            resolved = resolve_model_candidate(
                candidate,
                candidate_name=candidate_name,
                model_path=options.model_path,
                cache_dir=cache_dir,
                skip_download=options.skip_download,
            )
            inventory = inspect_gguf(Path(resolved.path))
            preflight = build_preflight(
                inventory,
                model_id=resolved.model_id,
                model_revision=resolved.resolved_revision,
                configured_architecture=resolved.architecture,
                configured_quantization=resolved.quantization,
                system_ram_available_bytes=int(psutil.virtual_memory().available),
                physical_vram_bytes=_physical_vram_bytes(identity),
                backend=config.backend.backend,
                backend_limitations=capabilities.limitations,
            ).model_dump(mode="json")
            preflight.update(
                {
                    "artifact_repository": resolved.artifact_repository,
                    "artifact_repository_revision": resolved.requested_revision,
                    "resolved_artifact_identity": resolved.resolved_revision,
                    "model_file_name": resolved.filename,
                    "model_file_size_bytes": resolved.file_size,
                    "model_file_sha256": resolved.file_sha256,
                    "model_source": resolved.source,
                    "revision_provenance_note": (
                        "repository revision was resolved by the Hugging Face API"
                        if resolved.source == "huggingface-hub"
                        else "repository revision is the pinned requested revision; local-file identity is independently fixed by SHA-256"
                    ),
                }
            )
            if preflight["eligible"]:
                execution_precheck = _model_execution_precheck(
                    bundle=bundle,
                    config=config,
                    executable=executable,
                    resolved=resolved,
                    candidate_name=candidate_name,
                    capabilities=capabilities,
                )
                preflight["backend_execution_precheck"] = execution_precheck
                if execution_precheck.get("status") != "COMPLETED":
                    preflight["eligible"] = False
                    preflight["rejection_reasons"].append(
                        "backend could not complete real deterministic generation: "
                        + str(execution_precheck.get("error") or "unknown backend failure")
                    )
            attempt.update(
                {
                    "status": "COMPLETED" if preflight["eligible"] else "REJECTED",
                    "resolved": resolved.as_dict(),
                    "preflight": dict(preflight),
                }
            )
            attempts.append(attempt)
            bundle.write_json("model_resolution_attempts.json", attempts)
            if not preflight["eligible"]:
                last_error = "; ".join(preflight["rejection_reasons"])
                if options.model_path is not None:
                    break
                continue
            preflight["selection_process"] = attempts
            return resolved, inventory, preflight
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt.update({"status": "FAILED", "error": last_error})
            attempts.append(attempt)
            bundle.write_json("model_resolution_attempts.json", attempts)
            bundle.record_failure(stage=f"model-resolution-{candidate_name}", error=last_error)
            if options.model_path is not None:
                break
    raise RuntimeError(f"no eligible model candidate could be resolved: {last_error}")


def _inventory_payload(inventory: GGUFInventory) -> dict[str, Any]:
    return {
        "classification": "MEASURED",
        "format": "GGUF",
        "gguf_version": inventory.version,
        "path": str(inventory.path),
        "file_size": inventory.file_size,
        "data_offset": inventory.data_offset,
        "tensor_count": len(inventory.tensors),
        "tensor_bytes": inventory.tensor_bytes,
        "expert_bytes": inventory.expert_bytes,
        "metadata": inventory.metadata,
        "tensors": [
            {
                "name": item.name,
                "layer_id": tensor_layer(item.name),
                "role": tensor_role(item.name),
                "shape": list(item.shape),
                "ggml_type": item.ggml_type,
                "dtype": item.dtype,
                "quantization": item.quantization,
                "offset": inventory.data_offset + item.offset,
                "byte_size": item.byte_size,
            }
            for item in inventory.tensors
        ],
    }


def _trim_prompts(
    prompts: list[WorkloadPrompt], *, output_tokens: int, count: int
) -> list[WorkloadPrompt]:
    return [replace(prompt, requested_output_tokens=output_tokens) for prompt in prompts[:count]]


def _safe_backend_arguments(arguments: list[str], capabilities: BackendCapabilities) -> list[str]:
    """Drop only options proven unavailable by the executable probe."""

    result: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--n-cpu-moe" and not capabilities.cpu_moe:
            index += 2
            continue
        if argument == "--override-tensor" and not capabilities.tensor_buffer_override:
            index += 2
            continue
        result.append(argument)
        index += 1
    return result


def _run_plan_workload(
    *,
    bundle: EvidenceBundle,
    executable: Path,
    model_path: Path,
    config: Experiment008Config,
    configuration: str,
    workload: str,
    plan: PhasePlan,
    prompts: list[WorkloadPrompt],
    capabilities: BackendCapabilities,
    search: bool = False,
    warmup_requests: int = 1,
    measurement_repeats: int = 1,
) -> WorkloadExecution:
    context_size = {
        "decode": 1024,
        "prefill_8k": 8448,
        "prefill_32k": 32768,
        "mixed": 2048,
    }[workload]
    parallel = 2 if workload == "mixed" else 1
    port = free_local_port(config.backend.host, start=config.backend.port_start)
    suffix = "search" if search else "ablation"
    log_root = bundle.root / "logs" / suffix / configuration / workload / plan.plan_id
    arguments = _safe_backend_arguments(plan.backend_arguments, capabilities)
    server = launch_llama_server(
        executable=executable,
        model_path=model_path,
        host=config.backend.host,
        port=port,
        context_size=context_size,
        parallel=parallel,
        plan_arguments=arguments,
        logs=log_root,
        startup_timeout_seconds=config.backend.startup_timeout_seconds,
        keep=config.backend.keep_servers,
    )
    client = LlamaCppClient(server.endpoint, timeout_seconds=config.backend.request_timeout_seconds)
    execution: WorkloadExecution | None = None
    try:
        warmups: list[dict[str, Any]] = []
        for warmup_index in range(warmup_requests):
            prompt = prompts[warmup_index % len(prompts)]
            warmup = client.generate(
                prompt.token_ids,
                output_tokens=min(prompt.requested_output_tokens, 8),
                seed=config.workloads.seed + 100_000 + warmup_index,
            )
            warmups.append(warmup.as_dict())
            if not warmup.success:
                raise RuntimeError(f"warm-up request failed: {warmup.error}")
        bundle.write_json(str((log_root / "warmups.json").relative_to(bundle.root)), warmups)
        repeated: list[WorkloadExecution] = []
        for repeat_index in range(measurement_repeats):
            if workload == "mixed":
                repeated.append(
                    execute_mixed_service(
                        configuration=configuration,
                        plan_id=plan.plan_id,
                        client=client,
                        interactive=prompts[0],
                        background=prompts[1],
                        seed=config.workloads.seed + repeat_index * 10,
                        sample_interval_seconds=config.profiling.resource_sample_interval_seconds,
                    )
                )
            else:
                repeated.append(
                    execute_prompt_batch(
                        configuration=configuration,
                        workload=workload,
                        plan_id=plan.plan_id,
                        client=client,
                        prompts=prompts,
                        seed=config.workloads.seed + repeat_index * 10,
                        sample_interval_seconds=config.profiling.resource_sample_interval_seconds,
                    )
                )
        if len(repeated) == 1:
            execution = repeated[0]
        else:
            numeric_keys = {
                key
                for item in repeated
                for key, value in item.observation.metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            merged_metrics: dict[str, Any] = {
                key: median(
                    float(item.observation.metrics[key])
                    for item in repeated
                    if isinstance(item.observation.metrics.get(key), (int, float))
                    and not isinstance(item.observation.metrics.get(key), bool)
                )
                for key in numeric_keys
            }
            merged_metrics["measurement_repeat_count"] = len(repeated)
            merged_metrics["repeat_metrics_json"] = json.dumps(
                [item.observation.metrics for item in repeated],
                sort_keys=True,
                separators=(",", ":"),
            )
            all_completed = all(
                item.observation.status == ExecutionStatus.COMPLETED for item in repeated
            )
            execution = WorkloadExecution(
                BenchmarkObservation(
                    configuration=configuration,  # type: ignore[arg-type]
                    workload=workload,  # type: ignore[arg-type]
                    plan_id=plan.plan_id,
                    status=(ExecutionStatus.COMPLETED if all_completed else ExecutionStatus.FAILED),
                    evidence_class=(EvidenceClass.MEASURED if all_completed else None),
                    metrics=merged_metrics,
                    unavailable_reason=(
                        None if all_completed else "one or more measurement repeats failed"
                    ),
                ),
                [generation for item in repeated for generation in item.generations],
                [row for item in repeated for row in item.resource_rows],
            )
        execution.observation.metrics["server_launch_seconds"] = server.launch_seconds
        execution.observation.metrics["unsupported_requested_techniques"] = [
            decision.technique
            for decision in plan.techniques
            if decision.execution_status == ExecutionStatus.UNSUPPORTED
        ]
        bundle.write_json(
            str((log_root / "generations.json").relative_to(bundle.root)),
            [item.as_dict() for item in execution.generations],
        )
        return execution
    finally:
        backend_was_running = server.process.poll() is None
        exit_code = server.close()
        stderr_path = log_root / "server.stderr.log"
        stderr_text = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path.is_file()
            else ""
        )
        backend_failure_kind = (
            "OUT_OF_MEMORY"
            if not backend_was_running
            and any(
                marker in stderr_text.lower()
                for marker in ("out of memory", "cuda error 2", "cudaerrormemoryallocation")
            )
            else "BACKEND_CRASH"
            if not backend_was_running
            else None
        )
        shutdown_mode = (
            "controlled_termination_after_workload"
            if backend_was_running
            else "backend_exited_before_cleanup"
        )
        if execution is not None:
            execution.observation.exit_code = (
                0 if execution.observation.status == ExecutionStatus.COMPLETED else exit_code
            )
            execution.observation.metrics["server_process_exit_code"] = exit_code
            execution.observation.metrics["server_shutdown_mode"] = shutdown_mode
            execution.observation.metrics["backend_failure_kind"] = backend_failure_kind
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            pass
        (log_root / "server.exit.json").write_text(
            json.dumps(
                {
                    "exit_code": exit_code,
                    "shutdown_mode": shutdown_mode,
                    "failure_kind": backend_failure_kind,
                    "workload_exit_code": (
                        execution.observation.exit_code if execution is not None else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _tokenize_workloads(
    *,
    bundle: EvidenceBundle,
    executable: Path,
    model_path: Path,
    config: Experiment008Config,
    candidate: BaselineCandidate,
    capabilities: BackendCapabilities,
) -> dict[str, list[WorkloadPrompt]]:
    port = free_local_port(config.backend.host, start=config.backend.port_start)
    log_root = bundle.root / "logs" / "tokenization"
    server = launch_llama_server(
        executable=executable,
        model_path=model_path,
        host=config.backend.host,
        port=port,
        context_size=32768,
        parallel=2,
        plan_arguments=_safe_backend_arguments(candidate.backend_arguments, capabilities),
        logs=log_root,
        startup_timeout_seconds=config.backend.startup_timeout_seconds,
        keep=False,
    )
    client = LlamaCppClient(server.endpoint, timeout_seconds=config.backend.request_timeout_seconds)
    try:

        def tokenizer(text):
            return client.tokenize(text, add_special=True)

        decode = build_decode_workload(
            tokenizer,
            prompt_count=config.workloads.decode_prompt_count,
            input_minimum=config.workloads.decode_input_tokens_min,
            input_maximum=config.workloads.decode_input_tokens_max,
            output_tokens=config.workloads.decode_output_tokens,
            seed=config.workloads.seed,
        )
        long_8k = build_long_context_workload(
            tokenizer,
            target_tokens=8_000,
            prompt_count=config.workloads.long_prompt_count,
            output_tokens=config.workloads.long_output_tokens,
            seed=config.workloads.seed,
        )
        long_32k = build_long_context_workload(
            tokenizer,
            target_tokens=32_000,
            prompt_count=config.workloads.long_prompt_count,
            output_tokens=config.workloads.long_output_tokens,
            seed=config.workloads.seed,
        )
    finally:
        backend_was_running = server.process.poll() is None
        exit_code = server.close()
        bundle.write_json(
            "logs/tokenization/server.exit.json",
            {
                "exit_code": exit_code,
                "shutdown_mode": (
                    "controlled_termination_after_tokenization"
                    if backend_was_running
                    else "backend_exited_before_cleanup"
                ),
                "tokenization_exit_code": 0 if "decode" in locals() else exit_code,
            },
        )
    mixed = [
        replace(
            decode[0],
            workload="mixed",
            requested_output_tokens=config.workloads.mixed_interactive_output_tokens,
        ),
        replace(
            decode[1],
            workload="mixed",
            requested_output_tokens=config.workloads.mixed_background_output_tokens,
        ),
    ]
    workloads = {"decode": decode, "prefill_8k": long_8k, "prefill_32k": long_32k, "mixed": mixed}
    bundle.write_json(
        "workload_manifest.json",
        {
            "classification": "MEASURED",
            "seed": config.workloads.seed,
            "scheduler_feature_contract": [
                "prompt_length",
                "requested_generation_length",
                "batch_size",
                "concurrency",
            ],
            "prompts": [
                prompt.manifest(include_text=True)
                for values in workloads.values()
                for prompt in values
            ],
        },
    )
    return workloads


def _search_prompts(workload: str, prompts: list[WorkloadPrompt]) -> list[WorkloadPrompt]:
    if workload == "decode":
        return _trim_prompts(prompts, output_tokens=64, count=1)
    if workload in {"prefill_8k", "prefill_32k"}:
        return _trim_prompts(prompts, output_tokens=16, count=1)
    return [
        replace(prompts[0], requested_output_tokens=32),
        replace(prompts[1], requested_output_tokens=64),
    ]


def _baseline_plan(candidate: BaselineCandidate, workload: str) -> PhasePlan:
    phase = "decode" if workload == "decode" else "mixed" if workload == "mixed" else "prefill"
    objective = {
        "decode": "maximum_decode_throughput",
        "prefill": "minimum_time_to_first_token",
        "mixed": "maximum_mixed_verified_throughput",
    }[phase]
    return PhasePlan(
        plan_id=f"{candidate.candidate_id}-{workload}",
        configuration="A",
        phase=phase,  # type: ignore[arg-type]
        objective=objective,  # type: ignore[arg-type]
        placements=[],
        techniques=[],
        backend_arguments=candidate.backend_arguments,
        predicted_metrics={},
        constraints={"search_probe": True},
        explanation=["fair stock llama.cpp candidate from the recorded bounded search"],
    )


def _baseline_search(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    executable: Path,
    model_path: Path,
    capabilities: BackendCapabilities,
    workloads: dict[str, list[WorkloadPrompt]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any] | None]]:
    candidates = baseline_search_space(config.baseline_search, seed=config.workloads.seed)
    existing = _json_or(bundle.root / "baseline_search.json", {})
    results: list[dict[str, Any]] = (
        list(existing.get("results", [])) if isinstance(existing, dict) else []
    )
    completed_keys = {
        (row.get("candidate_id"), row.get("workload"))
        for row in results
        if row.get("status") in {"COMPLETED", "UNSUPPORTED"}
    }
    workloads_order = ["decode", "prefill_8k", "prefill_32k", "mixed"]
    equal_target = max(config.baseline_search.cpu_moe_layers) // 2
    equal_candidate = min(
        candidates,
        key=lambda item: (
            abs(item.cpu_moe_layers - equal_target),
            item.cpu_threads != max(config.baseline_search.cpu_threads),
        ),
    )
    assignments: list[tuple[BaselineCandidate, str, str]] = [
        (
            candidate,
            workload,
            (
                "equal_cpu_gpu_expert_layer_split"
                if candidate == equal_candidate
                else "stock_search"
            ),
        )
        for candidate in candidates
        for workload in workloads_order
    ]
    assignment_payload = [
        {
            "candidate_id": candidate.candidate_id,
            "workload": workload,
            "baseline_role": role,
        }
        for candidate, workload, role in assignments
    ]
    for candidate, workload, baseline_role in assignments:
        key = (candidate.candidate_id, workload)
        if key in completed_keys:
            continue
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "workload": workload,
            "classification": "MEASURED",
            "baseline_role": baseline_role,
            **candidate.model_dump(mode="json"),
        }
        try:
            plan = _baseline_plan(candidate, workload)
            execution = _run_plan_workload(
                bundle=bundle,
                executable=executable,
                model_path=model_path,
                config=config,
                configuration="A",
                workload=workload,
                plan=plan,
                prompts=_search_prompts(workload, workloads[workload]),
                capabilities=capabilities,
                search=True,
                warmup_requests=config.baseline_search.warmup_requests,
                measurement_repeats=config.baseline_search.repeats,
            )
            executions = [execution]
            completed = [
                item for item in executions if item.observation.status == ExecutionStatus.COMPLETED
            ]
            if not completed:
                raise RuntimeError("all candidate repeats failed")
            metric_keys = {
                "decode": "decode_tokens_per_second",
                "prefill_8k": "time_to_first_token_ms",
                "prefill_32k": "time_to_first_token_ms",
                "mixed": "combined_generated_tokens_per_second",
            }
            primary_key = metric_keys[workload]
            values = [
                float(item.observation.metrics[primary_key])
                for item in completed
                if isinstance(item.observation.metrics.get(primary_key), (int, float))
            ]
            numeric_metric_keys = {
                key
                for item in completed
                for key, value in item.observation.metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            medians = {
                key: median(
                    float(item.observation.metrics[key])
                    for item in completed
                    if isinstance(item.observation.metrics.get(key), (int, float))
                    and not isinstance(item.observation.metrics.get(key), bool)
                )
                for key in numeric_metric_keys
            }
            row.update(
                {
                    **medians,
                    "status": "COMPLETED" if values else "FAILED",
                    primary_key: median(values) if values else None,
                    "repeat_count": config.baseline_search.repeats,
                    "successful_repeat_count": (config.baseline_search.repeats if completed else 0),
                    "exit_code": 0 if values else None,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "exit_code": getattr(exc, "returncode", None),
                }
            )
            bundle.record_failure(
                stage=f"baseline-{candidate.candidate_id}-{workload}",
                error=str(row["error"]),
                exit_code=row.get("exit_code"),
            )
        results.append(row)
        selected = select_best_stock_by_workload(results)
        bundle.write_json(
            "baseline_search.json",
            {
                "classification": "MEASURED",
                "status": "INCOMPLETE",
                "search_space": config.baseline_search.model_dump(mode="json"),
                "allocation_rule": "every stratified bounded candidate is measured on every workload; the exact 24/48 CPU-expert-layer candidate is tagged as the equal-split comparator",
                "assignments": assignment_payload,
                "candidates": [item.model_dump(mode="json") for item in candidates],
                "results": results,
                "selected_by_workload": selected,
            },
        )
    selected = select_best_stock_by_workload(results)
    bundle.write_json(
        "baseline_search.json",
        {
            "classification": "MEASURED",
            "status": "COMPLETED" if all(selected.values()) else "INCOMPLETE",
            "search_space": config.baseline_search.model_dump(mode="json"),
            "allocation_rule": "every stratified bounded candidate is measured on every workload; the exact 24/48 CPU-expert-layer candidate is tagged as the equal-split comparator",
            "assignments": assignment_payload,
            "candidates": [item.model_dump(mode="json") for item in candidates],
            "results": results,
            "selected_by_workload": selected,
        },
    )
    if not all(selected.values()):
        missing = [key for key, value in selected.items() if value is None]
        raise RuntimeError(f"baseline search produced no completed candidate for {missing}")
    bundle.complete_stage("baseline-search")
    return results, selected


def _candidate_from_selected(row: dict[str, Any]) -> BaselineCandidate:
    fields = {key: row[key] for key in BaselineCandidate.model_fields if key in row}
    return BaselineCandidate.model_validate(fields)


def _phase_for_workload(workload: str) -> str:
    return "decode" if workload == "decode" else "mixed" if workload == "mixed" else "prefill"


def _build_configuration_plan(
    *,
    configuration: str,
    workload: str,
    selected_stock: dict[str, Any],
    capabilities: BackendCapabilities,
    tiles: list[TensorTile],
    measured_utility: dict[str, float | None],
) -> PhasePlan:
    candidate = _candidate_from_selected(selected_stock)
    cpu_moe_layers = candidate.cpu_moe_layers
    plan = build_phase_plan(
        configuration=configuration,  # type: ignore[arg-type]
        phase=_phase_for_workload(workload),  # type: ignore[arg-type]
        capabilities=capabilities,
        tiles=tiles,
        stock_arguments=candidate.backend_arguments,
        cpu_moe_layers=cpu_moe_layers,
        measured_utility_by_technique=measured_utility,
    )
    plan.plan_id = f"{configuration.lower()}-{workload}-{candidate.candidate_id}"
    plan.predicted_metrics = {
        "decode_tokens_per_second": selected_stock.get("decode_tokens_per_second"),
        "time_to_first_token_ms": selected_stock.get("time_to_first_token_ms"),
        "mixed_generated_tokens_per_second": selected_stock.get(
            "combined_generated_tokens_per_second"
        ),
        "peak_vram_bytes": selected_stock.get("peak_vram_bytes"),
        "peak_system_ram_bytes": selected_stock.get("peak_system_ram_bytes"),
        "pcie_bytes_per_output_token": selected_stock.get("pcie_bytes_per_output_token"),
        "cpu_utilisation_percent": selected_stock.get("mean_process_tree_cpu_percent"),
        "gpu_utilisation_percent": selected_stock.get("mean_gpu_compute_utilisation_percent"),
    }
    plan.explanation.append(
        f"stock parameters were selected independently for {workload} by bounded measured search"
    )
    plan.explanation.append(
        "predicted resource and primary metrics use the matching measured stock search probe as historical execution evidence; cost-model extrapolation and final measurement are recorded separately"
    )
    return plan


def _incremental_technique_utilities(
    observations: list[dict[str, Any]],
    *,
    workload_filter: str | None = None,
) -> dict[str, float | None]:
    metric_by_workload = {
        "decode": ("decode_tokens_per_second", False),
        "prefill_8k": ("time_to_first_token_ms", True),
        "prefill_32k": ("time_to_first_token_ms", True),
        "mixed": ("combined_generated_tokens_per_second", False),
    }
    by_config_workload = {
        (row.get("configuration"), row.get("workload")): row
        for row in observations
        if row.get("status") == "COMPLETED"
    }
    mapping = {
        "tensor_granular_placement": ("A", "B"),
        "asynchronous_cpu_gpu_overlap": ("B", "C"),
        "activation_aware_expert_cache": ("C", "D"),
        "predictive_expert_prefetch": ("D", "E"),
        "separate_prefill_decode_plans": ("E", "F"),
    }
    utilities: dict[str, float | None] = {
        "stock_offloading": 0.0,
        # CPU-share utility is evaluated against a matched zero-CPU-MoE
        # baseline below; it must not inherit tensor-placement's A -> B gain.
        "asymmetric_cpu_gpu_partition": None,
    }
    for technique, (before_id, after_id) in mapping.items():
        values: list[float] = []
        for workload, (metric, lower) in metric_by_workload.items():
            if workload_filter is not None and workload != workload_filter:
                continue
            before = (
                by_config_workload.get((before_id, workload), {}).get("metrics", {}).get(metric)
            )
            after = by_config_workload.get((after_id, workload), {}).get("metrics", {}).get(metric)
            if (
                not isinstance(before, (int, float))
                or not isinstance(after, (int, float))
                or before == 0
                or after == 0
            ):
                continue
            values.append(1 - after / before if lower else after / before - 1)
        utilities[technique] = max(values) if values else None
    return utilities


def _matched_cpu_moe_utility(
    baseline_rows: list[dict[str, Any]],
    selected_stock: dict[str, Any],
    *,
    workload: str,
) -> float | None:
    """Return measured CPU-MoE utility against an otherwise identical zero-share arm."""

    selected_cpu_layers = selected_stock.get("cpu_moe_layers")
    if not isinstance(selected_cpu_layers, int):
        return None
    if selected_cpu_layers == 0:
        return 0.0
    comparison_fields = (
        "gpu_layers",
        "cpu_threads",
        "batch_size",
        "microbatch_size",
        "memory_map",
        "flash_attention",
    )
    zero_share = next(
        (
            row
            for row in baseline_rows
            if row.get("status") == "COMPLETED"
            and row.get("workload") == workload
            and row.get("cpu_moe_layers") == 0
            and all(row.get(field) == selected_stock.get(field) for field in comparison_fields)
        ),
        None,
    )
    if zero_share is None:
        return None
    metric, lower_is_better = {
        "decode": ("decode_tokens_per_second", False),
        "prefill_8k": ("time_to_first_token_ms", True),
        "prefill_32k": ("time_to_first_token_ms", True),
        "mixed": ("combined_generated_tokens_per_second", False),
    }[workload]
    selected_value = selected_stock.get(metric)
    zero_value = zero_share.get(metric)
    if (
        not isinstance(selected_value, (int, float))
        or not isinstance(zero_value, (int, float))
        or selected_value <= 0
        or zero_value <= 0
    ):
        return None
    return (
        1 - float(selected_value) / float(zero_value)
        if lower_is_better
        else float(selected_value) / float(zero_value) - 1
    )


def _extract_target_expert_traces(generations: list[GenerationResult]) -> list[ExpertActivation]:
    events: list[ExpertActivation] = []
    for generation in generations:
        trace = generation.raw_final_event.get("experiment_008_expert_trace")
        if not isinstance(trace, list):
            continue
        for row in trace:
            if not isinstance(row, dict):
                continue
            try:
                events.append(ExpertActivation.model_validate(row))
            except ValueError:
                continue
    return events


_BUFFER = re.compile(
    r"(?P<device>CPU(?:_Mapped|_Host)?|CUDA(?:\d+|_Host))\s+model buffer size\s*=\s*(?P<size>[0-9.]+)\s*MiB",
    re.IGNORECASE,
)
_OFFLOAD = re.compile(r"offloaded\s+(?P<gpu>\d+)/(?:\s*)?(?P<total>\d+)\s+layers", re.IGNORECASE)
_FIT_PART = re.compile(
    r"set ngl_per_device.*?\(n_layer, n_part(?:, overflow_type)?\)=\("
    r"(?P<layers>\d+),\s*(?P<parts>\d+)(?:,\s*(?P<overflow>[^)]+))?\)",
    re.IGNORECASE,
)


def _residency_accounting(
    *,
    bundle: EvidenceBundle,
    preflight: dict[str, Any],
    final_decode_plan: PhasePlan,
    tiles: list[TensorTile],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    buffers: list[dict[str, Any]] = []
    offloaded_layers: list[dict[str, Any]] = []
    for log in (
        (bundle.root / "logs" / "ablation" / "G").rglob("server.stderr.log")
        if (bundle.root / "logs" / "ablation" / "G").exists()
        else []
    ):
        text = log.read_text(encoding="utf-8", errors="replace")
        for match in _BUFFER.finditer(text):
            buffers.append(
                {
                    "log": str(log.relative_to(bundle.root)),
                    "device": match.group("device"),
                    "bytes": float(match.group("size")) * 1024**2,
                }
            )
        for match in _OFFLOAD.finditer(text):
            offloaded_layers.append(
                {
                    "log": str(log.relative_to(bundle.root)),
                    "gpu_layers": int(match.group("gpu")),
                    "total_layers": int(match.group("total")),
                }
            )
    # Use the final decode load, not a sum across phase-specific server starts.
    by_log: dict[str, list[dict[str, Any]]] = {}
    for row in buffers:
        by_log.setdefault(str(row["log"]), []).append(row)
    decode_by_log = {log: rows for log, rows in by_log.items() if "decode" in Path(log).parts}
    eligible_logs = decode_by_log or by_log
    selected_log_rows = max(
        eligible_logs.values(),
        key=lambda rows: sum(float(row["bytes"]) for row in rows),
        default=[],
    )

    def is_gpu_buffer(device: Any) -> bool:
        return bool(re.fullmatch(r"CUDA\d+", str(device), re.IGNORECASE))

    def is_host_buffer(device: Any) -> bool:
        return str(device).upper().startswith("CPU") or str(device).upper() == "CUDA_HOST"

    gpu_bytes = sum(
        float(row["bytes"]) for row in selected_log_rows if is_gpu_buffer(row["device"])
    )
    cpu_bytes = sum(
        float(row["bytes"]) for row in selected_log_rows if is_host_buffer(row["device"])
    )
    total = float(preflight.get("total_tensor_bytes", 0))
    reported = gpu_bytes + cpu_bytes
    buffer_sum_error = abs(reported - total) / total if total > 0 and reported > 0 else None

    # mmap keeps file-backed CPU address ranges for tensors that may also be
    # copied to CUDA, so its buffer rows are intentionally non-additive.  Find a
    # measured --no-mmap execution as the physical capacity-accounting witness.
    capacity_candidates: list[dict[str, Any]] = []
    for log in (bundle.root / "logs").rglob("server.stderr.log"):
        command_path = log.with_name("server.command.json")
        command_payload = _json_or(command_path, {})
        command = command_payload.get("command", [])
        if not isinstance(command, list) or "--no-mmap" not in command:
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        rows = [
            {
                "log": str(log.relative_to(bundle.root)),
                "device": match.group("device"),
                "bytes": float(match.group("size")) * 1024**2,
            }
            for match in _BUFFER.finditer(text)
        ]
        candidate_gpu = sum(float(row["bytes"]) for row in rows if is_gpu_buffer(row["device"]))
        candidate_cpu = sum(float(row["bytes"]) for row in rows if is_host_buffer(row["device"]))
        candidate_reported = candidate_gpu + candidate_cpu
        if not rows or candidate_gpu <= 0 or candidate_cpu <= 0 or total <= 0:
            continue
        capacity_candidates.append(
            {
                "log": str(log.relative_to(bundle.root)),
                "command_log": str(command_path.relative_to(bundle.root)),
                "command": command,
                "buffers": rows,
                "gpu_model_bytes": candidate_gpu,
                "host_model_bytes": candidate_cpu,
                "buffer_sum_bytes": candidate_reported,
                "buffer_sum_error_fraction": abs(candidate_reported - total) / total,
                "mmap": False,
            }
        )
    capacity_witness = min(
        capacity_candidates,
        key=lambda row: float(row["buffer_sum_error_fraction"]),
        default=None,
    )
    capacity_gpu_bytes = (
        float(capacity_witness["gpu_model_bytes"]) if capacity_witness is not None else 0.0
    )
    capacity_cpu_bytes = (
        float(capacity_witness["host_model_bytes"]) if capacity_witness is not None else 0.0
    )
    capacity_reconciles = bool(
        capacity_witness is not None
        and float(capacity_witness["buffer_sum_error_fraction"]) <= 0.001
    )
    selected_log = str(selected_log_rows[0]["log"]) if selected_log_rows else None
    selected_log_text = (
        (bundle.root / selected_log).read_text(encoding="utf-8", errors="replace")
        if selected_log
        else ""
    )
    fit_reports = [
        {
            "layers": int(match.group("layers")),
            "parts": int(match.group("parts")),
            "overflow_type": match.group("overflow"),
            "raw": match.group(0),
        }
        for match in _FIT_PART.finditer(selected_log_text)
    ]
    output_offloaded = "offloading output layer to GPU" in selected_log_text
    cpu_moe_layers = 0
    cpu_all_moe = "--cpu-moe" in final_decode_plan.backend_arguments
    tensor_override = "--override-tensor" in final_decode_plan.backend_arguments
    if "--n-cpu-moe" in final_decode_plan.backend_arguments:
        option_index = final_decode_plan.backend_arguments.index("--n-cpu-moe")
        if option_index + 1 < len(final_decode_plan.backend_arguments):
            try:
                cpu_moe_layers = int(final_decode_plan.backend_arguments[option_index + 1])
            except ValueError:
                cpu_moe_layers = 0
    ledger: list[dict[str, Any]] = []
    for tile in tiles:
        moe_weight = (
            tile.tensor_role.startswith("routed_expert") or tile.tensor_role == "shared_expert"
        )
        forced_cpu = moe_weight and (
            cpu_all_moe or (tile.layer_id >= 0 and tile.layer_id < cpu_moe_layers)
        )
        if forced_cpu:
            residency = "CPU"
            reason = "MoE tensor matched the selected CPU expert override"
        elif tensor_override and tile.tensor_role in {"router", "normalisation"}:
            residency = "GPU"
            reason = "tensor matched the explicit CUDA0 router/normalisation override"
        elif tile.tensor_role == "output_head" and output_offloaded:
            residency = "GPU"
            reason = "backend explicitly reported offloading the output layer"
        else:
            residency = "BACKEND_MANAGED"
            reason = (
                "llama.cpp automatic fit may place or fractionally overflow this tensor; "
                "the backend log does not export a per-tensor final ledger"
            )
        ledger.append(
            {
                "tensor_name": tile.tensor_name,
                "tensor_role": tile.tensor_role,
                "layer_id": tile.layer_id,
                "byte_size": tile.byte_size,
                "residency": residency,
                "execution_device": (
                    "GPU"
                    if residency == "GPU"
                    else "CPU"
                    if residency == "CPU"
                    else "BACKEND_SELECTED"
                ),
                "reason": reason,
            }
        )
    ledger_gpu_bytes = sum(row["byte_size"] for row in ledger if row["residency"] == "GPU")
    ledger_cpu_bytes = sum(row["byte_size"] for row in ledger if row["residency"] == "CPU")
    ledger_managed_bytes = sum(
        row["byte_size"] for row in ledger if row["residency"] == "BACKEND_MANAGED"
    )
    tensor_ledger_reconciles = ledger_gpu_bytes + ledger_cpu_bytes + ledger_managed_bytes == int(
        total
    )
    reconciled = (
        tensor_ledger_reconciles
        and capacity_reconciles
        and 0 < capacity_gpu_bytes < total
        and capacity_cpu_bytes > 0
    )

    def metric(config: str, workload: str, key: str) -> float | None:
        for row in observations:
            if row.get("configuration") == config and row.get("workload") == workload:
                value = row.get("metrics", {}).get(key)
                return float(value) if isinstance(value, (int, float)) else None
        return None

    stock_decode = metric("A", "decode", "decode_tokens_per_second")
    adaptive_decode = metric("G", "decode", "decode_tokens_per_second")
    positive_cpu_perf = (
        (cpu_all_moe or cpu_moe_layers > 0)
        and any(
            item.technique == "asymmetric_cpu_gpu_partition" and item.enabled
            for item in final_decode_plan.techniques
        )
        and stock_decode is not None
        and adaptive_decode is not None
        and adaptive_decode > stock_decode
    )
    return {
        "classification": "MEASURED" if selected_log_rows else None,
        "status": "COMPLETED" if selected_log_rows else "INCOMPLETE",
        "total_tensor_bytes": int(total),
        "backend_reported_buffers": selected_log_rows,
        "backend_reported_gpu_model_bytes": gpu_bytes if selected_log_rows else None,
        "backend_reported_cpu_model_bytes": cpu_bytes if selected_log_rows else None,
        "backend_reported_buffer_sum_bytes": reported if selected_log_rows else None,
        "backend_buffer_sum_error_fraction": buffer_sum_error,
        "backend_buffer_additivity": False,
        "backend_buffer_additivity_explanation": "CPU_Mapped may retain file-backed address ranges for tensors copied to CUDA, so CPU_Mapped and CUDA model buffers are not asserted to be disjoint physical residency",
        "capacity_accounting_witness": capacity_witness,
        "capacity_witness_selection": "measured --no-mmap execution with the lowest additive model-buffer error",
        "capacity_witness_additive": capacity_reconciles,
        "reconciled": reconciled,
        "planned_gpu_tensor_bytes": ledger_gpu_bytes,
        "planned_cpu_or_mapped_tensor_bytes": ledger_cpu_bytes,
        "backend_managed_tensor_bytes": ledger_managed_bytes,
        "backend_reported_gpu_model_bytes_is_measured": True,
        "backend_reported_cpu_mapped_bytes_is_measured": True,
        "planned_gpu_tensor_roles": sorted(
            {row["tensor_role"] for row in ledger if row["residency"] == "GPU"}
        ),
        "planned_cpu_tensor_roles": sorted(
            {row["tensor_role"] for row in ledger if row["residency"] == "CPU"}
        ),
        "backend_managed_tensor_roles": sorted(
            {row["tensor_role"] for row in ledger if row["residency"] == "BACKEND_MANAGED"}
        ),
        "tensor_residency_ledger": ledger,
        "tensor_ledger_reconciles_with_inventory": tensor_ledger_reconciles,
        "split_tensors": [],
        "split_explanation": "no exact split tensor names are claimed; automatic fit reported partial overflowing layers but did not export the affected logical tensor slices",
        "backend_fit_partial_layer_reports": fit_reports,
        "offloaded_layer_reports": offloaded_layers,
        "system_ram_contributes": capacity_cpu_bytes > 0
        and total > float(preflight.get("physical_vram_bytes", 0)),
        "no_complete_gpu_duplicate": capacity_gpu_bytes > 0 and capacity_gpu_bytes < total,
        "positive_cpu_performance_utility": positive_cpu_perf,
    }


def _cost_predictions(
    *,
    profile: dict[str, Any],
    preflight: dict[str, Any],
    plans: list[PhasePlan],
    observations: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kernels: list[MeasuredKernelPoint] = []
    for row in profile.get("kernel_measurements", []):
        try:
            kernels.append(
                MeasuredKernelPoint(
                    operation=str(row["operation"]),
                    device=str(row["device"]),
                    shape=[int(value) for value in row["shape"]],
                    median_ms=float(row["median_ms"]),
                    p95_ms=float(row["p95_ms"]),
                    effective_bandwidth_bytes_s=float(row["effective_bandwidth_bytes_s"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    transfers: list[TransferPoint] = []
    for row in profile.get("pcie_measurements", []):
        if row.get("status") != "COMPLETED":
            continue
        try:
            transfers.append(
                TransferPoint(
                    direction=str(row["direction"]),
                    memory_kind=str(row["memory_kind"]),
                    payload_bytes=int(row["payload_bytes"]),
                    median_ms=float(row["median_ms"]),
                    p95_ms=float(row["p95_ms"]),
                    effective_bandwidth_bytes_s=float(row["effective_bandwidth_bytes_s"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not kernels or not transfers:
        return [
            {
                "classification": "PROJECTED",
                "predicted_evidence_class": "PROJECTED",
                "measured_evidence_class": None,
                "status": "INCOMPLETE",
                "plan_id": None,
                "predicted_ms": None,
                "measured_ms": None,
                "reason": "measured kernel or PCIe points were unavailable",
            }
        ]
    model = MeasuredCostModel(kernels, transfers)
    layers = max(int(preflight.get("layer_count", 1)), 1)
    expert_bytes = int(preflight.get("total_expert_bytes", 0))
    experts = max(int(preflight.get("routed_expert_count", 1)), 1)
    active_bytes = max(expert_bytes // experts // layers, 1)
    rows: list[dict[str, Any]] = []
    observation_map = {
        (row.get("configuration"), row.get("workload")): row
        for row in observations
        if row.get("status") == "COMPLETED"
    }

    def cpu_moe_count(arguments: list[str]) -> int:
        if "--cpu-moe" in arguments:
            return layers
        if "--n-cpu-moe" not in arguments:
            return 0
        index = arguments.index("--n-cpu-moe")
        try:
            return max(0, min(layers, int(arguments[index + 1])))
        except (IndexError, ValueError):
            return 0

    def estimate_arguments(
        arguments: list[str],
        *,
        phase: str,
        workload: str,
        enabled: set[str],
    ) -> tuple[float | None, Any, dict[str, Any]]:
        operation = "matrix_vector" if phase != "prefill" else "matrix_matrix"
        cuda_point = next(
            (row for row in kernels if row.device == "cuda" and row.operation == operation),
            None,
        )
        cpu_point = next(
            (row for row in kernels if row.device == "cpu" and row.operation == operation),
            None,
        )
        if cuda_point is None or cpu_point is None:
            return None, None, {"reason": f"missing measured {operation} CPU/CUDA point"}
        token_count = {"decode": 1, "prefill_8k": 8_000, "prefill_32k": 32_000, "mixed": 1}[
            workload
        ]
        cuda_shape = list(cuda_point.shape)
        cpu_shape = list(cpu_point.shape)
        if phase == "prefill":
            cuda_shape[0] = token_count
            cpu_shape[0] = token_count
        cpu_layers = cpu_moe_count(arguments)
        gpu_expert_layers = layers - cpu_layers
        # One GPU core operation plus one expert operation per layer.  This is a
        # measured-shape extrapolation, not a backend execution trace.
        compute_tasks = [(operation, "cuda", cuda_shape)] * layers
        compute_tasks.extend([(operation, "cuda", cuda_shape)] * gpu_expert_layers)
        compute_tasks.extend([(operation, "cpu", cpu_shape)] * cpu_layers)
        hidden_bytes = token_count * 2048 * 4 * cpu_layers
        transfer_tasks: list[tuple[str, str, int]] = []
        if hidden_bytes:
            transfer_tasks.extend(
                [
                    ("device_to_host", "pinned", hidden_bytes),
                    ("host_to_device", "pinned", hidden_bytes),
                ]
            )
        if "predictive_expert_prefetch" in enabled:
            transfer_tasks.append(("host_to_device", "pinned", active_bytes))
        try:
            cost = model.estimate(
                compute_tasks=compute_tasks,
                transfer_tasks=transfer_tasks,
                dequantization_ms=0.0,
                synchronization_ms=0.05 * layers,
                reduction_ms=0.01 * max(cpu_layers, 1),
                cache_miss_ms=0.0,
                contention_ms=0.0,
                asynchronous="asynchronous_cpu_gpu_overlap" in enabled,
            )
        except ValueError as exc:
            return None, None, {"reason": str(exc)}
        return (
            cost.completion_ms,
            cost,
            {
                "input_token_assumption": token_count,
                "cpu_expert_layer_count": cpu_layers,
                "gpu_expert_layer_count": gpu_expert_layers,
                "predicted_pcie_bytes": hidden_bytes * 2
                + (active_bytes if "predictive_expert_prefetch" in enabled else 0),
                "dequantization_assumption": "zero separate latency because selected Q4 kernels fuse dequantization; native llama-bench measurements are retained in hardware_profile.json",
                "contention_assumption": "zero because no target-runtime contention coefficient was measurable",
            },
        )

    for plan in plans:
        workload = (
            "decode"
            if plan.phase == "decode"
            else "mixed"
            if plan.phase == "mixed"
            else "prefill_8k"
            if "prefill_8k" in plan.plan_id
            else "prefill_32k"
        )
        enabled = {item.technique for item in plan.techniques if item.enabled}
        predicted, cost, assumptions = estimate_arguments(
            plan.backend_arguments,
            phase=plan.phase,
            workload=workload,
            enabled=enabled,
        )
        observation = observation_map.get((plan.configuration, workload), {})
        metrics = observation.get("metrics", {})
        if plan.phase == "decode":
            rate = metrics.get("decode_tokens_per_second")
            measured = 1000 / rate if isinstance(rate, (int, float)) and rate > 0 else None
        elif plan.phase == "mixed":
            rate = metrics.get("combined_generated_tokens_per_second")
            measured = 1000 / rate if isinstance(rate, (int, float)) and rate > 0 else None
        else:
            measured = metrics.get("time_to_first_token_ms")
            measured = float(measured) if isinstance(measured, (int, float)) else None
        rows.append(
            {
                "classification": "PROJECTED",
                "predicted_evidence_class": "PROJECTED",
                "measured_evidence_class": "MEASURED" if measured is not None else None,
                "plan_id": plan.plan_id,
                "configuration": plan.configuration,
                "phase": plan.phase,
                "comparison_group": workload,
                "predicted_ms": predicted,
                "measured_ms": measured,
                "prediction_error_fraction": (
                    abs(predicted - measured) / measured
                    if predicted is not None and measured is not None and measured > 0
                    else None
                ),
                "cost_breakdown": cost.model_dump(mode="json") if cost else None,
                "assumptions": assumptions,
                "historical_probe_predictions": plan.predicted_metrics,
            }
        )
    for baseline in baseline_rows:
        if baseline.get("status") != "COMPLETED":
            continue
        workload = str(baseline.get("workload"))
        if workload not in {"decode", "prefill_8k", "prefill_32k", "mixed"}:
            continue
        phase = "decode" if workload == "decode" else "mixed" if workload == "mixed" else "prefill"
        arguments = baseline.get("backend_arguments", [])
        if not isinstance(arguments, list):
            continue
        predicted, cost, assumptions = estimate_arguments(
            [str(item) for item in arguments],
            phase=phase,
            workload=workload,
            enabled=set(),
        )
        if workload == "decode":
            rate = baseline.get("decode_tokens_per_second")
            measured = 1000 / rate if isinstance(rate, (int, float)) and rate > 0 else None
        elif workload == "mixed":
            rate = baseline.get("combined_generated_tokens_per_second")
            measured = 1000 / rate if isinstance(rate, (int, float)) and rate > 0 else None
        else:
            value = baseline.get("time_to_first_token_ms")
            measured = float(value) if isinstance(value, (int, float)) else None
        rows.append(
            {
                "classification": "PROJECTED",
                "predicted_evidence_class": "PROJECTED",
                "measured_evidence_class": "MEASURED" if measured is not None else None,
                "plan_id": f"{baseline.get('candidate_id')}:{workload}",
                "configuration": "BASELINE_SEARCH",
                "phase": phase,
                "comparison_group": workload,
                "predicted_ms": predicted,
                "measured_ms": measured,
                "prediction_error_fraction": (
                    abs(predicted - measured) / measured
                    if predicted is not None and measured is not None and measured > 0
                    else None
                ),
                "cost_breakdown": cost.model_dump(mode="json") if cost else None,
                "assumptions": assumptions,
            }
        )
    return rows


def _profile_native_quantized_backend(*, executable: Path, model_path: Path) -> dict[str, Any]:
    """Measure target GGUF kernels through the matching official llama-bench binary."""

    benchmark = executable.with_name("llama-bench.exe")
    if not benchmark.is_file():
        return {
            "classification": None,
            "status": "UNSUPPORTED",
            "reason": "the selected backend directory does not contain llama-bench.exe",
            "cases": [],
        }
    common = [
        "-m",
        str(model_path),
        "-o",
        "json",
        "-t",
        "20",
        "-mmp",
        "1",
    ]
    cases = {
        "asymmetric_gpu_cpu_q4": [
            "-p",
            "512",
            "-n",
            "64",
            "-r",
            "3",
            "-ngl",
            "99",
            "-ncmoe",
            "24",
            "-b",
            "2048",
            "-ub",
            "512",
            "-fa",
            "on",
        ],
        "cpu_only_q4": [
            "-p",
            "128",
            "-n",
            "16",
            "-r",
            "2",
            "-ngl",
            "0",
            "-ncmoe",
            "48",
            "-b",
            "512",
            "-ub",
            "128",
            "-fa",
            "off",
        ],
    }
    rows: list[dict[str, Any]] = []
    for name, arguments in cases.items():
        command = [str(benchmark), *common, *arguments]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3_600,
                cwd=model_path.parent,
            )
            measurements = json.loads(result.stdout) if result.returncode == 0 else None
            if not isinstance(measurements, list):
                measurements = None
            rows.append(
                {
                    "classification": "MEASURED" if measurements is not None else None,
                    "status": "COMPLETED" if measurements is not None else "FAILED",
                    "case": name,
                    "command": command,
                    "exit_code": result.returncode,
                    "measurements": measurements,
                    "stderr_tail": result.stderr[-8_000:],
                    "reason": None
                    if measurements is not None
                    else "llama-bench did not return a successful JSON measurement array",
                }
            )
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "classification": None,
                    "status": "FAILED",
                    "case": name,
                    "command": command,
                    "exit_code": getattr(exc, "returncode", None),
                    "measurements": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "classification": "MEASURED" if any(row["status"] == "COMPLETED" for row in rows) else None,
        "status": "COMPLETED"
        if all(row["status"] == "COMPLETED" for row in rows)
        else "INCOMPLETE",
        "benchmark_path": str(benchmark),
        "benchmark_sha256": file_sha256(benchmark),
        "scope": "real target GGUF Q4_K/Q6_K kernels measured end-to-end by the matching pinned llama.cpp build",
        "cases": rows,
    }


def _hardware_profile_with_exact_cache(
    *,
    repository_root: Path,
    executable: Path,
    executable_sha256: str,
    config: Experiment008Config,
    resolved: ResolvedModel,
    trace_path: Path,
) -> dict[str, Any]:
    model_identity = f"{resolved.model_id}@{resolved.resolved_revision}"
    identity = collect_hardware_identity(
        backend=config.backend.backend,
        model=model_identity,
        quantization=resolved.quantization,
    )
    runtime_settings = {
        "warmups": config.profiling.warmup_iterations,
        "iterations": config.profiling.measurement_iterations,
        "quick": False,
        "decode_shapes": config.profiling.decode_shapes,
        "prefill_shapes": config.profiling.prefill_shapes,
        "cpu_thread_counts": config.profiling.cpu_thread_counts,
        "payload_bytes": config.profiling.payload_bytes,
        "storage_sample_bytes": config.profiling.storage_sample_bytes,
    }
    cache_inputs = {
        "identity": identity,
        "runtime_settings": runtime_settings,
        "backend_executable_sha256": executable_sha256,
        "model_file_sha256": resolved.file_sha256,
    }
    key = profile_fingerprint(cache_inputs)
    cache_root = repository_root / ".cache" / "experiment_008" / "hardware_profiles"
    cache_path = cache_root / f"{key}.json"
    cache_trace_path = cache_root / f"{key}.trace.json"
    if cache_path.is_file() and cache_trace_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("profile_key") == key
            and cached.get("cache_key_inputs") == cache_inputs
        ):
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cache_trace_path, trace_path)
            cached["profile_reused"] = True
            cached["reused_at_utc"] = datetime.now(UTC).isoformat()
            overlap = cached.get("overlap_measurement")
            if isinstance(overlap, dict):
                overlap["trace_path"] = str(trace_path)
            return cached
    profile = build_hardware_profile(
        backend=config.backend.backend,
        model=model_identity,
        quantization=resolved.quantization,
        model_path=Path(resolved.path),
        decode_shapes=config.profiling.decode_shapes,
        prefill_shapes=config.profiling.prefill_shapes,
        cpu_thread_counts=config.profiling.cpu_thread_counts,
        payload_bytes=config.profiling.payload_bytes,
        warmups=config.profiling.warmup_iterations,
        iterations=config.profiling.measurement_iterations,
        storage_sample_bytes=config.profiling.storage_sample_bytes,
        trace_path=trace_path,
        quick=False,
    )
    profile["native_quantized_backend_measurements"] = _profile_native_quantized_backend(
        executable=executable,
        model_path=Path(resolved.path),
    )
    profile["profile_key"] = key
    profile["cache_key_inputs"] = cache_inputs
    profile["profile_reused"] = False
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    if trace_path.is_file():
        shutil.copyfile(trace_path, cache_trace_path)
    return profile


def _full_run(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    options: Experiment008Options,
    repository_root: Path,
) -> None:
    server_path = options.server_path or (
        Path(config.backend.server_path) if config.backend.server_path else None
    )
    executable, acquisition = resolve_llama_server(
        supplied_path=server_path,
        allow_download=config.backend.server_download and not options.skip_download,
        release_tag=config.backend.release_tag,
        cuda_version=config.backend.windows_cuda_version,
        destination_root=repository_root / "artifacts" / "backend-environments" / "experiment-008",
    )
    probe = probe_llama_server(executable)
    bundle.write_json("backend_acquisition.json", acquisition)
    bundle.write_json("backend_probe.json", probe.as_dict())
    if not probe.capabilities.conventional_layer_offload:
        raise RuntimeError("selected llama.cpp binary lacks conventional GPU layer offloading")
    bundle.complete_stage("backend-probe")

    preliminary_identity = collect_hardware_identity(
        backend=config.backend.backend,
        model=config.models.preferred.model_id,
        quantization=config.models.preferred.quantization,
    )
    cache_dir = repository_root / ".cache" / "experiment_008" / "huggingface"
    resolved, inventory, preflight = _resolve_and_preflight(
        bundle=bundle,
        config=config,
        options=options,
        capabilities=probe.capabilities,
        identity=preliminary_identity,
        cache_dir=cache_dir,
        executable=executable,
    )
    bundle.write_json("model_preflight.json", preflight)
    bundle.write_json("tensor_inventory.json", _inventory_payload(inventory))
    # Full content hashing reads each logical tensor byte exactly once.  The model
    # file hash from acquisition independently identifies the complete artifact.
    tiles = build_tensor_tiles(
        inventory,
        model_id=resolved.model_id,
        model_revision=resolved.resolved_revision,
        hash_contents=True,
    )
    bundle.write_json(
        "tensor_tiles.json",
        {
            "classification": "MEASURED",
            "tiles": [tile.model_dump(mode="json") for tile in tiles],
            "expert_microshards": [],
            "microshard_explanation": "no expert projection microshards were executed by the selected backend; whole logical tensors are represented without arbitrary byte slicing",
        },
    )
    bundle.complete_stage("model-preflight-and-inventory")

    identity = collect_hardware_identity(
        backend=config.backend.backend,
        model=resolved.model_id,
        quantization=resolved.quantization,
    )
    bundle.write_json("environment.json", _environment(repository_root, identity))
    profile = _hardware_profile_with_exact_cache(
        repository_root=repository_root,
        executable=executable,
        executable_sha256=probe.executable_sha256,
        config=config,
        resolved=resolved,
        trace_path=bundle.root / "profiler_trace" / "pytorch_overlap.json",
    )
    bundle.write_json("hardware_profile.json", profile)
    bundle.complete_stage("hardware-profile")

    candidates = baseline_search_space(config.baseline_search, seed=config.workloads.seed)
    workloads = _tokenize_workloads(
        bundle=bundle,
        executable=executable,
        model_path=Path(resolved.path),
        config=config,
        candidate=candidates[0],
        capabilities=probe.capabilities,
    )
    bundle.complete_stage("workload-tokenization")
    baseline_rows, selected = _baseline_search(
        bundle=bundle,
        config=config,
        executable=executable,
        model_path=Path(resolved.path),
        capabilities=probe.capabilities,
        workloads=workloads,
    )

    observations: list[dict[str, Any]] = [
        row
        for row in _json_or(bundle.root / "benchmark_results.json", [])
        if isinstance(row, dict) and row.get("status") != "NOT_RUN"
    ]
    resources: list[dict[str, Any]] = list(_json_or(bundle.root / "resource_timeseries.json", []))
    generation_by_config_workload: dict[tuple[str, str], list[GenerationResult]] = {}
    token_evidence: dict[str, list[dict[str, Any]]] = dict(
        _json_or(bundle.root / "correctness_tokens.json", {})
    )
    persisted_expert_events: list[dict[str, Any]] = list(
        _json_or(bundle.root / "target_expert_events.json", [])
    )
    existing_plan_payload = _json_or(bundle.root / "candidate_plans.json", {"plans": []})
    plans: list[PhasePlan] = [
        PhasePlan.model_validate(item)
        for item in existing_plan_payload.get("plans", [])
        if isinstance(item, dict) and item.get("plan_id")
    ]
    configurations = [options.configuration] if options.configuration else list("ABCDEFG")
    if (
        options.configuration
        and options.configuration != "A"
        and not bundle.is_configuration_complete("A")
    ):
        raise RuntimeError(
            "a selected non-A configuration requires a resumed bundle with completed A evidence"
        )
    for configuration in configurations:
        assert configuration is not None
        if bundle.is_configuration_complete(configuration):
            continue
        for workload in ("decode", "prefill_8k", "prefill_32k", "mixed"):
            if any(
                row.get("configuration") == configuration
                and row.get("workload") == workload
                and row.get("status") in {"COMPLETED", "UNSUPPORTED"}
                for row in observations
            ):
                continue
            stock = selected[workload]
            assert stock is not None
            measured_utility = (
                _incremental_technique_utilities(observations, workload_filter=workload)
                if configuration == "G"
                else {}
            )
            if configuration == "G":
                measured_utility["asymmetric_cpu_gpu_partition"] = _matched_cpu_moe_utility(
                    baseline_rows,
                    stock,
                    workload=workload,
                )
            plan = _build_configuration_plan(
                configuration=configuration,
                workload=workload,
                selected_stock=stock,
                capabilities=probe.capabilities,
                tiles=tiles,
                measured_utility=measured_utility,
            )
            plans = [item for item in plans if item.plan_id != plan.plan_id]
            plans.append(plan)
            unsupported_required = [
                decision.technique
                for decision in plan.techniques
                if decision.execution_status == ExecutionStatus.UNSUPPORTED
            ]
            if configuration != "G" and unsupported_required:
                reason = (
                    "cumulative configuration is unsupported by the probed target backend: "
                    + ", ".join(sorted(unsupported_required))
                )
                observations.append(
                    BenchmarkObservation(
                        configuration=configuration,  # type: ignore[arg-type]
                        workload=workload,  # type: ignore[arg-type]
                        plan_id=plan.plan_id,
                        status=ExecutionStatus.UNSUPPORTED,
                        metrics={
                            "decode_tokens_per_second": None,
                            "time_to_first_token_ms": None,
                            "mixed_verified_tokens_per_second": None,
                            "interactive_p95_latency_ms": None,
                            "peak_vram_bytes": None,
                            "peak_system_ram_bytes": None,
                            "pcie_bytes_per_output_token": None,
                            "cpu_gpu_overlap_percent": None,
                            "expert_cache_hit_rate": None,
                            "useful_prefetch_rate": None,
                        },
                        unavailable_reason=reason,
                    ).model_dump(mode="json")
                )
                bundle.write_json("benchmark_results.json", observations)
                _write_benchmark_csv(bundle, observations)
                bundle.write_json(
                    "candidate_plans.json",
                    {
                        "classification": "MEASURED",
                        "plans": [item.model_dump(mode="json") for item in plans],
                    },
                )
                continue
            try:
                execution = _run_plan_workload(
                    bundle=bundle,
                    executable=executable,
                    model_path=Path(resolved.path),
                    config=config,
                    configuration=configuration,
                    workload=workload,
                    plan=plan,
                    prompts=workloads[workload],
                    capabilities=probe.capabilities,
                )
                observation = execution.observation.model_dump(mode="json")
                observations.append(observation)
                resources.extend(execution.resource_rows)
                generation_by_config_workload[(configuration, workload)] = execution.generations
                token_evidence[f"{configuration}:{workload}"] = [
                    {
                        "success": item.success,
                        "output_token_ids": item.output_token_ids,
                        "content": item.content,
                        "error": item.error,
                    }
                    for item in execution.generations
                ]
                persisted_expert_events.extend(
                    item.model_dump(mode="json")
                    for item in _extract_target_expert_traces(execution.generations)
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = ExecutionStatus.FAILED
                observations.append(
                    BenchmarkObservation(
                        configuration=configuration,  # type: ignore[arg-type]
                        workload=workload,  # type: ignore[arg-type]
                        plan_id=plan.plan_id,
                        status=status,
                        metrics={
                            "decode_tokens_per_second": None,
                            "time_to_first_token_ms": None,
                            "mixed_verified_tokens_per_second": None,
                            "interactive_p95_latency_ms": None,
                            "peak_vram_bytes": None,
                            "peak_system_ram_bytes": None,
                            "pcie_bytes_per_output_token": None,
                            "cpu_gpu_overlap_percent": None,
                            "expert_cache_hit_rate": None,
                            "useful_prefetch_rate": None,
                        },
                        unavailable_reason=error,
                        exit_code=getattr(exc, "returncode", None),
                    ).model_dump(mode="json")
                )
                bundle.record_failure(
                    stage=f"configuration-{configuration}-{workload}",
                    error=error,
                    exit_code=getattr(exc, "returncode", None),
                )
            bundle.write_json("benchmark_results.json", observations)
            _write_benchmark_csv(bundle, observations)
            bundle.write_json("resource_timeseries.json", resources)
            bundle.write_csv("resource_timeseries.csv", resources)
            bundle.write_json("correctness_tokens.json", token_evidence)
            bundle.write_json("target_expert_events.json", persisted_expert_events)
            bundle.write_json(
                "candidate_plans.json",
                {
                    "classification": "MEASURED",
                    "plans": [item.model_dump(mode="json") for item in plans],
                },
            )
        terminal_workloads = {
            str(row.get("workload"))
            for row in observations
            if row.get("configuration") == configuration
            and row.get("status") in {"COMPLETED", "UNSUPPORTED"}
        }
        if terminal_workloads == {"decode", "prefill_8k", "prefill_32k", "mixed"}:
            bundle.complete_configuration(configuration)

    # If a resume skipped configurations, reconstruct their plans from persisted evidence.
    plan_payload = _json_or(bundle.root / "candidate_plans.json", {"plans": []})
    plans = [PhasePlan.model_validate(item) for item in plan_payload.get("plans", [])]
    if not plans:
        raise RuntimeError("no candidate plans were persisted")
    utilities = {
        workload: _incremental_technique_utilities(observations, workload_filter=workload)
        for workload in ("decode", "prefill_8k", "prefill_32k", "mixed")
    }
    for workload, values in utilities.items():
        selected_stock = selected.get(workload)
        values["asymmetric_cpu_gpu_partition"] = (
            _matched_cpu_moe_utility(baseline_rows, selected_stock, workload=workload)
            if selected_stock is not None
            else None
        )
    g_plans = [plan for plan in plans if plan.configuration == "G"]
    if g_plans:
        g_decode = next(plan for plan in g_plans if plan.phase == "decode")
        g_prefill_candidates = [plan for plan in g_plans if plan.phase == "prefill"]
        # Keep the 32K plan as the canonical prefill artifact; all phase plans remain in candidate_plans.
        g_prefill = next(
            (plan for plan in g_prefill_candidates if "prefill_32k" in plan.plan_id),
            g_prefill_candidates[0],
        )
    else:
        latest_configuration = max(plan.configuration for plan in plans)
        g_decode = next(
            plan
            for plan in plans
            if plan.configuration == latest_configuration and plan.phase == "decode"
        )
        g_prefill = next(
            plan
            for plan in plans
            if plan.configuration == latest_configuration and plan.phase == "prefill"
        )
    bundle.write_json("prefill_plan.json", g_prefill.model_dump(mode="json"))
    bundle.write_json("decode_plan.json", g_decode.model_dump(mode="json"))
    bundle.write_json(
        "adaptive_plan.json",
        {
            "classification": "MEASURED",
            "prefill_plan": g_prefill.model_dump(mode="json"),
            "decode_plan": g_decode.model_dump(mode="json"),
            "prefill_decode_plans_differ": g_prefill.backend_arguments
            != g_decode.backend_arguments,
            "technique_decisions": [item.model_dump(mode="json") for item in g_decode.techniques],
            "phase_technique_decisions": {
                plan.plan_id: [item.model_dump(mode="json") for item in plan.techniques]
                for plan in g_plans
            },
            "measured_incremental_utility_by_workload_and_technique": utilities,
            "utility_comparison_note": "unsupported cumulative configurations remain null; G may enable only supported techniques with positive measured utility",
        },
    )

    # Deterministic correctness uses the already-executed full workload tokens.
    reference = {
        workload: token_evidence.get(f"A:{workload}", [])
        for workload in ("decode", "prefill_8k", "prefill_32k", "mixed")
    }
    comparisons: list[dict[str, Any]] = []
    executions = 0
    identity_by_configuration: dict[str, float | None] = {item: None for item in "ABCDEFG"}
    for configuration in "ABCDEFG":
        identities: list[bool] = []
        for workload in ("decode", "prefill_8k", "prefill_32k", "mixed"):
            candidate_generations = token_evidence.get(f"{configuration}:{workload}", [])
            reference_generations = reference.get(workload, [])
            for index, candidate_generation in enumerate(candidate_generations):
                executions += 1
                if configuration == "A":
                    identities.append(bool(candidate_generation.get("success")))
                    continue
                if index >= len(reference_generations):
                    continue
                same = candidate_generation.get("output_token_ids") == reference_generations[
                    index
                ].get("output_token_ids")
                same_text = candidate_generation.get("content") == reference_generations[index].get(
                    "content"
                )
                identities.append(same)
                comparisons.append(
                    {
                        "configuration": configuration,
                        "workload": workload,
                        "prompt_index": index,
                        "reference_token_ids": reference_generations[index].get("output_token_ids"),
                        "candidate_token_ids": candidate_generation.get("output_token_ids"),
                        "exact_token_identity": same,
                        "reference_text": reference_generations[index].get("content"),
                        "candidate_text": candidate_generation.get("content"),
                        "exact_text_identity": same_text,
                    }
                )
        identity_by_configuration[configuration] = (
            sum(identities) / len(identities) if identities else None
        )
    all_identity = [row["exact_token_identity"] for row in comparisons]
    all_text_identity = [row["exact_text_identity"] for row in comparisons]
    for observation in observations:
        if observation.get("workload") != "mixed" or observation.get("status") != "COMPLETED":
            continue
        metrics = observation.get("metrics")
        configuration = str(observation.get("configuration"))
        identity = identity_by_configuration.get(configuration)
        if not isinstance(metrics, dict):
            continue
        generated_rate = metrics.get("combined_generated_tokens_per_second")
        metrics["mixed_verified_tokens_per_second"] = (
            float(generated_rate)
            if identity == 1.0 and isinstance(generated_rate, (int, float))
            else 0.0
            if identity is not None
            else None
        )
        metrics["verification_status"] = (
            "VERIFIED_EXACT_TOKEN_IDENTITY"
            if identity == 1.0
            else "FAILED_TOKEN_IDENTITY"
            if identity is not None
            else "UNAVAILABLE"
        )
    bundle.write_json("benchmark_results.json", observations)
    _write_benchmark_csv(bundle, observations)
    fixture = validate_tiny_moe_fixture(seed=config.workloads.seed)
    fixture_checks = fixture.get("checks", {})
    runtime_fixture = fixture.get("tensor_runtime", {})
    required_check_statuses = {
        "tensor_tile_reconstruction": {
            "classification": "EMULATED",
            "passed": bool(fixture_checks.get("tensor_tile_reconstruction", {}).get("allclose")),
        },
        "expert_microshard_equivalence": {
            "classification": "EMULATED",
            "passed": bool(fixture_checks.get("expert_microshard_equivalence", {}).get("allclose")),
        },
        "cpu_gpu_split_equivalence": {
            "classification": "EMULATED",
            "passed": bool(runtime_fixture.get("cpu_gpu_split_equivalence", {}).get("allclose")),
            "status": runtime_fixture.get("status"),
        },
        "cache_hit_and_cache_miss_equivalence": {
            "classification": "EMULATED",
            "passed": bool(
                fixture_checks.get("cache_hit_and_miss_equivalence", {}).get("allclose")
                and runtime_fixture.get("cache_miss_equivalence", {}).get("allclose")
                and runtime_fixture.get("cache_hit_equivalence", {}).get("allclose")
            ),
        },
        "prefetch_enabled_disabled_equivalence": {
            "classification": "EMULATED",
            "passed": bool(
                fixture_checks.get("prefetch_enabled_disabled_equivalence", {}).get("allclose")
            ),
        },
        "separate_prefill_decode_plan_equivalence": {
            "classification": "EMULATED",
            "passed": bool(
                fixture_checks.get("separate_prefill_decode_plan_equivalence", {}).get("allclose")
            ),
        },
        "end_to_end_greedy_token_comparison": {
            "classification": "MEASURED",
            "passed": bool(all_identity) and all(all_identity),
            "comparison_count": len(all_identity),
        },
    }
    correctness = {
        "classification": "MEASURED" if comparisons else None,
        "reference_configuration": "A",
        "deterministic_execution_count": executions,
        "comparison_count": len(comparisons),
        "token_identity_rate": sum(all_identity) / len(all_identity) if all_identity else None,
        "text_identity_rate": (
            sum(all_text_identity) / len(all_text_identity) if all_text_identity else None
        ),
        "token_identity_by_configuration": identity_by_configuration,
        "fixture_checks_passed": bool(fixture.get("passed")),
        "fixture_results": fixture,
        "required_check_statuses": required_check_statuses,
        "comparisons": comparisons,
        "selected_expert_identity": None,
        "selected_expert_limitation": "selected llama.cpp server did not export routing IDs",
        "final_logits": None,
        "final_logits_limitation": "selected llama.cpp server API did not expose final logits",
        "numerical_equivalence_mode": None,
    }
    bundle.write_json("correctness_results.json", correctness)

    expert_events = [ExpertActivation.model_validate(item) for item in persisted_expert_events]
    if expert_events:
        statistics, coactivation = activation_statistics(expert_events)
        activation_rows = [
            {"classification": "MEASURED", **row.model_dump(mode="json")} for row in statistics
        ]
        bundle.write_csv("expert_activation_matrix.csv", activation_rows)
        bundle.write_csv(
            "expert_coactivation.csv",
            [{"classification": "MEASURED", **row} for row in coactivation],
        )
        expert_summary = {
            "classification": "MEASURED",
            "status": "COMPLETED",
            "activation_event_count": len(expert_events),
            "expert_statistics": activation_rows,
            "gpu_cached_experts": None,
            "useful_prefetch_bytes": None,
            "wasted_prefetch_bytes": None,
            "visible_transfer_latency_removed_ms": None,
            "prediction_conclusion": "backend trace existed but cache/prefetch counters were absent",
        }
    else:
        reason = "selected backend did not expose routed expert IDs; no activation, cache, or prefetch measurement was fabricated"
        bundle.write_csv(
            "expert_activation_matrix.csv",
            [{"classification": None, "status": "UNSUPPORTED", "reason": reason}],
        )
        bundle.write_csv(
            "expert_coactivation.csv",
            [{"classification": None, "status": "UNSUPPORTED", "reason": reason}],
        )
        expert_summary = {
            "classification": None,
            "status": "UNSUPPORTED",
            "reason": reason,
            "activation_event_count": None,
            "gpu_cached_experts": None,
            "useful_prefetch_bytes": None,
            "wasted_prefetch_bytes": None,
            "visible_transfer_latency_removed_ms": None,
            "prediction_conclusion": "not measurable with the selected backend",
        }
    bundle.write_json("expert_trace_summary.json", expert_summary)

    ablations = build_ablation_rows(
        observations,
        token_identity_by_configuration=identity_by_configuration,
    )
    bundle.write_csv("ablation_results.csv", ablations)
    cost_rows = _cost_predictions(
        profile=profile,
        preflight=preflight,
        plans=plans,
        observations=observations,
        baseline_rows=baseline_rows,
    )
    bundle.write_csv("cost_model_predictions.csv", cost_rows)
    quality = prediction_quality(cost_rows)
    measured_decode = {
        str(row["configuration"]): float(row["decode_tokens_per_second"])
        for row in ablations
        if isinstance(row.get("decode_tokens_per_second"), (int, float))
    }
    selected_id = "G" if "G" in measured_decode else max(measured_decode, default="")
    regret = (
        planner_regret_fraction(measured_decode, selected_id)
        if selected_id and measured_decode
        else None
    )
    planner_quality = {
        **quality,
        "regret_fraction": regret,
        "selected_configuration": selected_id or None,
        "all_selected_placements_explained": bool(g_plans)
        and all(bool(item.reason) for plan in g_plans for item in plan.placements),
        "can_reject_harmful_techniques": any(not item.enabled for item in g_decode.techniques),
    }
    residency = _residency_accounting(
        bundle=bundle,
        preflight=preflight,
        final_decode_plan=g_decode,
        tiles=tiles,
        observations=observations,
    )
    bundle.write_json("residency_accounting.json", residency)
    bundle.write_json("planner_quality.json", planner_quality)
    bundle.complete_stage("analysis")


def _ensure_partial_artifacts(bundle: EvidenceBundle, *, reason: str) -> None:
    json_placeholders = {
        "environment.json": {"classification": None, "status": "INCOMPLETE", "reason": reason},
        "hardware_profile.json": {"classification": None, "status": "INCOMPLETE", "reason": reason},
        "model_preflight.json": {"classification": None, "status": "INCOMPLETE", "reason": reason},
        "tensor_inventory.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "tensors": [],
        },
        "tensor_tiles.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "tiles": [],
            "expert_microshards": [],
        },
        "expert_trace_summary.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
        },
        "baseline_search.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "candidates": [],
            "results": [],
            "selected_by_workload": {},
        },
        "candidate_plans.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "plans": [],
        },
        "prefill_plan.json": {"classification": None, "status": "INCOMPLETE", "reason": reason},
        "decode_plan.json": {"classification": None, "status": "INCOMPLETE", "reason": reason},
        "adaptive_plan.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "technique_decisions": [],
            "prefill_decode_plans_differ": None,
        },
        "correctness_results.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "deterministic_execution_count": 0,
            "token_identity_rate": None,
            "fixture_checks_passed": False,
        },
        "residency_accounting.json": {
            "classification": None,
            "status": "INCOMPLETE",
            "reason": reason,
            "reconciled": False,
            "system_ram_contributes": False,
            "no_complete_gpu_duplicate": False,
            "positive_cpu_performance_utility": False,
        },
    }
    for name, payload in json_placeholders.items():
        if not (bundle.root / name).is_file():
            bundle.write_json(name, payload)
    csv_placeholders = {
        "expert_activation_matrix.csv": [
            {"classification": None, "status": "INCOMPLETE", "reason": reason}
        ],
        "expert_coactivation.csv": [
            {"classification": None, "status": "INCOMPLETE", "reason": reason}
        ],
        "cost_model_predictions.csv": [
            {
                "classification": None,
                "status": "INCOMPLETE",
                "reason": reason,
                "predicted_ms": None,
                "measured_ms": None,
            }
        ],
        "benchmark_results.csv": [
            {"classification": None, "status": "INCOMPLETE", "reason": reason}
        ],
        "ablation_results.csv": build_ablation_rows(
            _empty_observations(reason),
            token_identity_by_configuration={item: None for item in "ABCDEFG"},
        ),
        "resource_timeseries.csv": [
            {"classification": None, "status": "INCOMPLETE", "reason": reason}
        ],
    }
    for name, rows in csv_placeholders.items():
        if not (bundle.root / name).is_file():
            bundle.write_csv(name, rows)


def _architecture_audit(
    repository_root: Path,
    bundle: EvidenceBundle,
    correctness: dict[str, Any],
) -> dict[str, Any]:
    # Probe concrete reusable symbols, rather than treating the mere presence of a
    # source file as evidence that an implementation exists.
    from swarm_inference.experiments.experiment_008 import (
        benchmark,
        cost_model,
        experts,
        gguf,
        hardware,
        planning,
        runtime,
    )
    from swarm_inference.experiments.experiment_008 import bundle as bundle_module

    del repository_root  # Imports above resolve through the installed/source package.
    contracts: dict[str, list[Any]] = {
        "hardware_profiling": [hardware.build_hardware_profile, hardware.ResourceSampler],
        "tensor_metadata": [gguf.inspect_gguf, gguf.build_tensor_tiles],
        "cost_estimation": [cost_model.MeasuredCostModel, cost_model.critical_path],
        "tensor_placement": [planning.build_phase_plan],
        "expert_tracing": [experts.activation_statistics],
        "expert_caching": [experts.ExpertLRUCache],
        "tensor_cache_runtime": [runtime.ExpertTensorCache],
        "prefetch_scheduling": [runtime.ExpertTensorCache.prefetch],
        "phase_specific_planning": [planning.build_phase_plan],
        "benchmark_comparison": [benchmark.execute_prompt_batch, benchmark.execute_mixed_service],
        "evidence_bundle_generation": [bundle_module.EvidenceBundle],
    }
    availability = {
        name: bool(symbols) and all(callable(symbol) for symbol in symbols)
        for name, symbols in contracts.items()
    }
    artifact_audit = bundle.audit_required()
    fixture_validation = bool(correctness.get("fixture_checks_passed"))
    complete = all(availability.values()) and artifact_audit["complete"] and fixture_validation
    return {
        "complete": complete,
        "features": availability,
        "artifact_audit": artifact_audit,
        "runtime_fixture_validation_passed": fixture_validation,
        "reasons": [
            f"{name}: {'implemented' if value else 'missing'}"
            for name, value in availability.items()
        ]
        + [
            f"required artifact contract: {'complete' if artifact_audit['complete'] else 'incomplete'}",
            f"runtime equivalence fixture: {'pass' if fixture_validation else 'fail'}",
        ],
        "limitation": "backend-specific target-model hooks are capability-gated; reusable policy components do not imply that every backend can execute them",
    }


def _finalize(
    *,
    bundle: EvidenceBundle,
    config: Experiment008Config,
    options: Experiment008Options,
    repository_root: Path,
    terminal_error: str | None,
) -> Experiment008Verdict:
    reason = terminal_error or "run completed but an artifact was unavailable"
    _ensure_partial_artifacts(bundle, reason=reason)
    preflight = _json_or(bundle.root / "model_preflight.json", {})
    observations = _json_or(bundle.root / "benchmark_results.json", [])
    if not observations:
        observations = _empty_observations(reason)
        bundle.write_json("benchmark_results.json", observations)
        _write_benchmark_csv(bundle, observations)
    correctness = _json_or(bundle.root / "correctness_results.json", {})
    residency = _json_or(bundle.root / "residency_accounting.json", {})
    planner_quality = _json_or(bundle.root / "planner_quality.json", {})
    ablations = build_ablation_rows(
        observations,
        token_identity_by_configuration=correctness.get(
            "token_identity_by_configuration", {item: None for item in "ABCDEFG"}
        ),
    )
    bundle.write_csv("ablation_results.csv", ablations)
    architecture = _architecture_audit(repository_root, bundle, correctness)
    gates = evaluate_gates(
        official_full_run=options.full,
        preflight=preflight if preflight.get("eligible") else None,
        observations=observations,
        ablations=ablations,
        correctness=correctness,
        residency=residency,
        planner_quality=planner_quality,
        architecture_audit=architecture,
        acceptance=config.acceptance,
    )
    benchmark_generation = any(
        row.get("status") == "COMPLETED" and row.get("evidence_class") == "MEASURED"
        for row in observations
    )
    baseline_payload_for_execution = _json_or(bundle.root / "baseline_search.json", {})
    baseline_generation = any(
        row.get("status") == "COMPLETED" and row.get("classification") == "MEASURED"
        for row in baseline_payload_for_execution.get("results", [])
        if isinstance(row, dict)
    )
    real_generation = benchmark_generation or baseline_generation
    verdict = overall_verdict(
        gates,
        real_model_generation_succeeded=real_generation,
        official_full_run=options.full,
    )
    # Only a recognizable external acquisition/capacity limitation promotes a
    # no-generation full attempt to PARTIAL. Internal implementation failures
    # remain FAIL and are never presented as experimental evidence.
    external_markers = (
        "no eligible model candidate",
        "model path does not exist",
        "is not cached",
        "no native llama-server",
        "available system RAM is below",
        "CUDA is unavailable",
    )
    if (
        options.full
        and terminal_error
        and not real_generation
        and any(marker.lower() in terminal_error.lower() for marker in external_markers)
    ):
        verdict = Experiment008Verdict.PARTIAL
    completed_g = next((row for row in ablations if row["configuration"] == "G"), {})
    completed_a = next((row for row in ablations if row["configuration"] == "A"), {})

    def delta(row: dict[str, Any]) -> float | None:
        value = row.get("decode_change_vs_previous")
        return float(value) if isinstance(value, (int, float)) else None

    contribution_names = {
        "B": "tensor-aware static placement plus asymmetric CPU/GPU partitioning",
        "C": "asynchronous CPU/GPU overlap",
        "D": "activation-aware expert cache",
        "E": "predictive expert prefetch",
        "F": "separate prefill and decode plans",
        "G": "positive-utility feature selection",
    }
    contributions = [
        (contribution_names.get(str(row["configuration"]), str(row["configuration"])), delta(row))
        for row in ablations
        if delta(row) is not None and row["configuration"] != "A"
    ]
    largest = max(contributions, key=lambda item: item[1], default=None)
    least = min(contributions, key=lambda item: item[1], default=None)
    if terminal_error:
        summary = f"The full evidence run was incomplete: {terminal_error}. No missing measurement was imputed."
    elif verdict == Experiment008Verdict.PASS_STRONG:
        summary = "The over-VRAM sparse MoE ran correctly and the positive-utility adaptive plan crossed the required performance threshold."
    elif verdict == Experiment008Verdict.PASS_CAPACITY_AND_ARCHITECTURE:
        summary = "The over-VRAM sparse MoE established capacity, correctness, planner, and reusable architecture, but not the adaptive performance threshold."
    elif real_generation:
        summary = (
            "Real model generation succeeded, but one or more foundational acceptance gates failed."
        )
    else:
        summary = "No valid real over-32-GiB model generation evidence was produced; quick evidence is software validation only."
    baseline_payload = _json_or(bundle.root / "baseline_search.json", {})
    equal_results = [
        row
        for row in baseline_payload.get("results", [])
        if row.get("baseline_role") == "equal_cpu_gpu_expert_layer_split"
        and row.get("status") == "COMPLETED"
    ]
    equal_comparison: dict[str, Any] = {}
    for row in equal_results:
        workload = str(row.get("workload"))
        key = {
            "decode": "decode_tokens_per_second",
            "prefill_8k": "ttft_8k",
            "prefill_32k": "ttft_32k",
            "mixed": "mixed_verified_tokens_per_second",
        }.get(workload)
        source_key = (
            "time_to_first_token_ms"
            if workload.startswith("prefill")
            else "decode_tokens_per_second"
            if workload == "decode"
            else "combined_generated_tokens_per_second"
        )
        adaptive_value = completed_g.get(key) if key else None
        equal_value = row.get(source_key)
        equal_comparison[workload] = {
            "adaptive": adaptive_value,
            "equal_split": equal_value,
            "adaptive_better": (
                adaptive_value < equal_value
                if workload.startswith("prefill")
                else adaptive_value > equal_value
            )
            if isinstance(adaptive_value, (int, float)) and isinstance(equal_value, (int, float))
            else None,
        }
    stock_comparison = {
        "decode": {
            "stock": completed_a.get("decode_tokens_per_second"),
            "adaptive": completed_g.get("decode_tokens_per_second"),
        },
        "prefill_32k": {
            "stock": completed_a.get("ttft_32k"),
            "adaptive": completed_g.get("ttft_32k"),
        },
        "mixed": {
            "stock": completed_a.get("mixed_verified_tokens_per_second"),
            "adaptive": completed_g.get("mixed_verified_tokens_per_second"),
        },
    }
    g_decode_observation = next(
        (
            row
            for row in observations
            if row.get("configuration") == "G"
            and row.get("workload") == "decode"
            and row.get("status") == "COMPLETED"
        ),
        {},
    )
    g_metrics = g_decode_observation.get("metrics", {})
    gpu_util = g_metrics.get("mean_gpu_compute_utilisation_percent")
    cpu_util = g_metrics.get("mean_process_tree_cpu_percent")
    if isinstance(gpu_util, (int, float)) and gpu_util >= 80:
        dominant_bottleneck = f"inference from measured decode telemetry: GPU compute saturation ({gpu_util:.1f}% mean)"
    elif (
        isinstance(gpu_util, (int, float))
        and isinstance(cpu_util, (int, float))
        and gpu_util < 60
        and cpu_util >= 100
    ):
        dominant_bottleneck = f"inference from measured decode telemetry: host/offload path (GPU {gpu_util:.1f}% mean, process-tree CPU {cpu_util:.1f}%)"
    else:
        dominant_bottleneck = "not established by the available measured telemetry"
    gate_status_by_id = {gate.gate_id: gate.status for gate in gates}
    if gate_status_by_id.get(3) == GateStatus.PASS:
        swarm_implication = (
            "strengthens the distributed-swarm thesis: measured adaptive scheduling improved a "
            "primary workload while preserving correctness and cross-workload constraints"
        )
    elif gate_status_by_id.get(1) == GateStatus.PASS:
        swarm_implication = (
            "mixed evidence: strengthens the heterogeneous-residency capacity thesis but weakens "
            "the claim that finer-grained scheduling is automatically faster"
        )
    elif real_generation:
        swarm_implication = (
            "weakens the current thesis because real execution did not establish the required "
            "capacity and adaptive-performance evidence"
        )
    else:
        swarm_implication = "inconclusive without real target-model execution"
    verdict_payload = {
        "schema_version": "experiment-008-verdict-v1",
        "overall_verdict": verdict.value,
        "answer_first_summary": summary,
        "official_full_run": options.full,
        "real_model_generation_succeeded": real_generation,
        "real_model_generation_sources": {
            "official_ablation_workloads": benchmark_generation,
            "bounded_baseline_search": baseline_generation,
        },
        "terminal_error": terminal_error,
        "gates": [gate.model_dump(mode="json") for gate in gates],
        "planner_quality": planner_quality,
        "planner_beat_stock": stock_comparison,
        "planner_beat_equal_microsharding": (
            {
                "comparator_scope": "equal 24/48 expert-layer CPU/GPU offload; the target backend did not expose equal within-tensor projection microsharding",
                "workloads": equal_comparison,
            }
            if equal_comparison
            else "not measured"
        ),
        "largest_technique_contribution": largest,
        "least_technique_contribution": least,
        "dominant_bottleneck": dominant_bottleneck,
        "swarm_thesis_implication": swarm_implication,
        "failed_acceptance_gates": [gate.name for gate in gates if gate.status == GateStatus.FAIL],
        "not_evaluated_gates": [
            gate.name for gate in gates if gate.status == GateStatus.NOT_EVALUATED
        ],
    }
    bundle.write_json("verdict.json", verdict_payload)
    generate_required_plots(bundle.root)
    bundle.write_text("report.md", build_report(bundle.root))
    bundle.write_text("README.md", build_bundle_readme(bundle.root))
    source_reproduce = (
        repository_root
        / "experiments"
        / "008_single_host_adaptive_moe_saturation"
        / "reproduce.ps1"
    )
    if source_reproduce.is_file():
        bundle.write_text("reproduce.ps1", source_reproduce.read_text(encoding="utf-8"))
    else:
        bundle.write_text(
            "reproduce.ps1",
            "throw 'Source reproduction script was unavailable when this partial bundle was created.'\n",
        )
    bundle.complete_stage("final-report")
    final_audit = bundle.audit_required()
    bundle.write_json("artifact_audit.json", final_audit)
    return verdict


def run_experiment_008(options: Experiment008Options) -> Experiment008Outcome:
    options.validate()
    repository_root = Path(__file__).resolve().parents[4]
    config = load_experiment_008_config(options.config_path)
    output_base = options.output_directory or (repository_root / config.output_root)
    bundle_root = create_bundle_root(
        output_base,
        explicit=options.output_directory is not None,
    )
    bundle = EvidenceBundle(bundle_root, resume=options.resume)
    run_id = bundle.root.parent.name if bundle.root.name == "experiment_008" else bundle.root.name
    manifest = _json_or(bundle.root / "manifest.json", {})
    if not manifest:
        manifest = {
            "schema_version": "experiment-008-manifest-v1",
            "run_id": run_id,
            "run_mode": "FULL" if options.full else "QUICK",
            "classification_contract": {
                "MEASURED": "directly observed on this machine",
                "EMULATED": "software fixture or failure-injection evidence",
                "PROJECTED": "cost-model extrapolation from measured inputs",
            },
            "started_at_utc": datetime.now(UTC).isoformat(),
            "config_path": str(options.config_path.resolve()),
            "config": config.model_dump(mode="json"),
            "model_path_argument": str(options.model_path.resolve())
            if options.model_path
            else None,
            "skip_download": options.skip_download,
            "resume": options.resume,
            "selected_configuration": options.configuration,
            "official_verdict_eligible": options.full and options.configuration is None,
        }
        bundle.write_json("manifest.json", manifest)
    terminal_error: str | None = None
    interrupted = False
    try:
        if options.quick:
            _quick_run(bundle=bundle, config=config, repository_root=repository_root)
        else:
            _full_run(
                bundle=bundle,
                config=config,
                options=options,
                repository_root=repository_root,
            )
    except KeyboardInterrupt:
        interrupted = True
        terminal_error = "KeyboardInterrupt: run interrupted by user; checkpoint and completed artifacts were preserved"
        bundle.record_failure(stage="runner", error=terminal_error, status="INTERRUPTED")
    except Exception as exc:
        terminal_error = f"{type(exc).__name__}: {exc}"
        bundle.record_failure(
            stage="runner",
            error=terminal_error,
            exit_code=getattr(exc, "returncode", None),
        )
    verdict = _finalize(
        bundle=bundle,
        config=config,
        options=options,
        repository_root=repository_root,
        terminal_error=terminal_error,
    )
    manifest = _json_or(bundle.root / "manifest.json", {})
    manifest.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "terminal_error": terminal_error,
            "interrupted": interrupted,
            "overall_verdict": verdict.value,
            "artifact_audit": bundle.audit_required(),
        }
    )
    bundle.write_json("manifest.json", manifest)
    # Rebuild the two narrative files after the final manifest update.
    bundle.write_text("README.md", build_bundle_readme(bundle.root))
    bundle.write_text("report.md", build_report(bundle.root))
    return Experiment008Outcome(
        bundle_path=bundle.root,
        verdict=verdict,
        completed=terminal_error is None,
        error=terminal_error,
    )
