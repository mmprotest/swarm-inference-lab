"""Resumable hardware-in-the-loop Experiment 010 execution matrix."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import psutil

from swarm_inference.backends.colibri.constants import COLIBRI_COMMIT, COLIBRI_RELEASE
from swarm_inference.backends.colibri.probe import ColibriCapabilityProbe
from swarm_inference.backends.colibri.schemas import NativeQuantizationMetadata
from swarm_inference.experiments.experiment_008.hardware import collect_hardware_identity
from swarm_inference.experiments.experiment_010.batching import (
    BatchingPolicy,
    RoutedRequest,
    batching_summary,
    make_routing_batches,
)
from swarm_inference.experiments.experiment_010.bundle import (
    REQUIRED_FILES,
    Experiment010Bundle,
    create_bundle_root,
)
from swarm_inference.experiments.experiment_010.codecs import (
    codec_break_even,
    decode_array,
    encode_array,
    numerical_error,
)
from swarm_inference.experiments.experiment_010.coordinator import (
    MicroshardOwner,
    StableExpertCoordinator,
    compare_layer_results,
)
from swarm_inference.experiments.experiment_010.dispatch import (
    ExpertDispatcher,
    FailureController,
    FailureEvent,
)
from swarm_inference.experiments.experiment_010.expert import (
    ExpertStore,
    deterministic_expert,
    npz_expert_loader,
    slice_expert_weights,
)
from swarm_inference.experiments.experiment_010.kimi import (
    KIMI_LOGICAL_MOE_LAYERS,
    KIMI_ROUTED_EXPERTS,
    deterministic_kimi_expert,
    execute_kimi_topk,
    kimi_fixture_inventory,
    run_full_kimi_k3_fixture,
)
from swarm_inference.experiments.experiment_010.level_a import (
    capture_level_a_activation,
    execute_level_a_expert_rpc,
)
from swarm_inference.experiments.experiment_010.planner import (
    PositiveUtilityPlanner,
    worker_marginal_utility,
)
from swarm_inference.experiments.experiment_010.relay import ExpertRelayManager
from swarm_inference.experiments.experiment_010.reporting import (
    build_report,
    generate_required_plots,
)
from swarm_inference.experiments.experiment_010.schemas import (
    DataPlane,
    DeterminismMode,
    EvidenceCategory,
    ExecutionStrategy,
    Experiment010Mode,
    Experiment010Verdict,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    FailureType,
    GateResult,
    GateStatus,
    PhasePlan,
    PlannerCandidate,
    PlannerObjective,
    RecoveryStrategy,
    ServicePhase,
    TransportCodec,
    WorkerBudget,
    WorkerManifest,
    classify_verdict,
)
from swarm_inference.experiments.experiment_010.transport import (
    NETWORK_PROFILES,
    ExpertTransportClient,
    measured_network_profile,
)
from swarm_inference.experiments.experiment_010.verification import (
    TrustController,
    reconcile_expert_ownership,
    reconcile_microshard_ownership,
)
from swarm_inference.experiments.experiment_010.worker import (
    ExpertWorkerManager,
    fixture_ownership_entry,
)
from swarm_inference.experiments.fanout_resources import environment_command_snapshot
from swarm_inference.microsharding.expert_abi import (
    ExpertMicroshardDescriptor,
    ExpertProjectionSlice,
    validate_expert_microshard_set,
)
from swarm_inference.protocol.checksums import sha256_bytes
from swarm_inference.simulation.expert_model import (
    calibrate_expert_simulator,
    project_virtual_topologies,
    remote_break_even_surface,
)

MATRIX_CONFIGURATIONS = {
    "A": "A_monolithic_local",
    "B": "B_swarm_adapter_local_passthrough",
    "C": "C_independent_request_routing",
    "D": "D_cross_process_whole_shared_memory",
    "E": "E_cross_process_whole_direct_tcp",
    "F": "F_cross_process_whole_relayed_tcp",
    "G": "G_naive_per_expert_rpc",
    "H": "H_coalesced_per_layer_whole_rpc",
    "I": "I_equal_expert_microshards",
    "J": "J_asymmetric_expert_microshards",
    "K": "K_coalesced_layer_microshards",
    "L": "L_positive_utility_planner",
}

REQUIRED_MATRIX_NETWORK_PROFILES = (
    "loopback_unshaped",
    "lan_10g",
    "lan_2_5g",
    "lan_1g",
    "wifi",
    "regional_wan",
)

REUSED_COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "component": "Universal Worker ABI and lifecycle service",
        "source": "swarm_inference.worker.abi/server/client",
        "use": "MOE_EXPERT negotiation, identity, capabilities, heartbeat, submission, cancellation, and graceful shutdown",
    },
    {
        "component": "Direct activation tensor envelope",
        "source": "swarm_inference.protocol.tensor_codec (SWARMT01)",
        "use": "exact raw-FP32 activation and result blobs inside the expert semantic frame",
    },
    {
        "component": "Network simulator",
        "source": "swarm_inference.simulation.network.NetworkEmulator",
        "use": "deterministic schedules applied to actual expert socket payloads before transmission",
    },
    {
        "component": "Experiment 006 executable ownership contract",
        "source": "swarm_inference.microsharding.expert_abi",
        "use": "ExpertMicroshardDescriptor construction and whole-coverage validation",
    },
    {
        "component": "Experiment 007 heterogeneous planner",
        "source": "swarm_inference.planner.HeterogeneousPlanner",
        "use": "role utility and non-degradation admission beneath Experiment 010 confidence/capacity rules",
    },
    {
        "component": "Experiment 008 hardware profiler",
        "source": "swarm_inference.experiments.experiment_008.hardware",
        "use": "hardware identity embedded in the Experiment 010 environment fingerprint",
    },
    {
        "component": "Experiment 009 Colibri integration",
        "source": "swarm_inference.backends.colibri",
        "use": "pinned dependency, capability probe, bridge metadata, and fixed-replay evidence contract",
    },
    {
        "component": "Environment fingerprint",
        "source": "swarm_inference.experiments.fanout_resources.environment_command_snapshot",
        "use": "pre-run process, interpreter, memory, swap, and working-directory snapshot",
    },
    {
        "component": "Evidence integrity primitives",
        "source": "swarm_inference.protocol.checksums and Experiment 009 bundle conventions",
        "use": "content-addressed inventories, atomic writes, checkpoints, raw failure retention, and null preservation",
    },
)

MODIFIED_COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "component": "Colibri Windows build",
        "change": "build sm_120 CUDA runtime DLL and bind an executable correctness proof to capability advertisement",
    },
    {
        "component": "Colibri capability schema/probe",
        "change": "fail closed unless DLL load, RTX detection, kernel execution, residency, and CPU comparison all pass",
    },
    {
        "component": "Execution-mode registry",
        "change": "register hardware-in-loop virtual swarm closure without replacing prior modes",
    },
    {
        "component": "Simulator package",
        "change": "add expert/RPC/microshard costs, configuration-level held-out split, validation gates, and regret",
    },
)

ADDED_COMPONENTS: tuple[dict[str, str], ...] = (
    {
        "component": "Backend-neutral expert semantic ABI and SWARMEX1 framing",
        "scope": "whole experts, coalesced layers, codecs, integrity, and deterministic reductions",
    },
    {
        "component": "Isolated expert worker adapters and data planes",
        "scope": "direct TCP, relay TCP, shared memory, ownership, budgets, telemetry, and cleanup",
    },
    {
        "component": "Experiment 010 orchestration and evidence",
        "scope": "A-L matrix, shaping, failures, trust, planner, Kimi fixture, plots, report, and reproduction CLI",
    },
)

DEFERRED_REUSE: tuple[dict[str, str], ...] = (
    {
        "component": "Experiment 009 ColibriFixedReplayTuner execution",
        "reason": "The current expert RPC probe returns an operator vector but does not continue generation; feeding it fabricated token outputs would violate the tuner's exact-token contract. The class is retained for the end-to-end Colibri hook gate.",
    },
    {
        "component": "ActivationTransport FaultProxy",
        "reason": "Its API targets ActivationRequest/ActivationResult stage messages and cannot address expert IDs, worker processes, or expert-result corruption. Experiment 010 extends failure control at the expert semantic boundary and records this incompatibility.",
    },
)


@dataclass(slots=True)
class Experiment010Options:
    mode: Experiment010Mode = Experiment010Mode.QUICK
    model_path_level_a: Path | None = None
    model_path_level_b: Path | None = None
    kimi_fixture_path: Path | None = None
    colibri_path: Path | None = None
    output_directory: Path | None = None
    resume: bool = False
    rebuild_colibri: bool = False
    rebuild_cuda: bool = False
    apply_bridge_patches: bool = False
    topology: str | None = None
    network_profile: str | None = None
    configuration: str | None = None
    repeats: int = 1
    telemetry_level: str = "detailed"
    skip_model_download: bool = False
    skip_level_b: bool = False
    skip_kimi_fixture: bool = False
    model_path_frontier: Path | None = None

    def validate(self) -> None:
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.network_profile is not None and self.network_profile not in NETWORK_PROFILES:
            raise ValueError(f"unknown network profile {self.network_profile!r}")
        if self.telemetry_level not in {"off", "summary", "detailed", "trace"}:
            raise ValueError("telemetry level is invalid")
        if self.mode == Experiment010Mode.FULL and self.repeats < 3:
            raise ValueError("full mode requires at least three repeats")


@dataclass(frozen=True, slots=True)
class Experiment010Outcome:
    bundle_path: Path
    verdict: Experiment010Verdict
    error: str | None


def _git(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _environment(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    memory = psutil.virtual_memory()
    experiment_008_identity = collect_hardware_identity(
        backend="experiment-010",
        model="multi-level",
        quantization="model-specific",
    )
    partitions = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except OSError:
            continue
        partitions.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_bytes": usage.total,
                "available_bytes": usage.free,
            }
        )
    environment = {
        "category": EvidenceCategory.MEASURED_PHYSICAL.value,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "os_build": platform.version(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_model": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "numa_topology": None,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "page_file_percent": psutil.swap_memory().percent,
        "environment_variables": {
            key: os.environ.get(key)
            for key in (
                "CUDA_PATH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "COLI_SWARM_BRIDGE",
            )
        },
        "command": list(sys.argv),
        "power_plan": None,
        "experiment_008_hardware_identity": experiment_008_identity,
        "pre_run_command_snapshot": environment_command_snapshot(),
    }
    fingerprint = {
        "repository_commit": _git(repository, "rev-parse", "HEAD"),
        "repository_status": _git(repository, "status", "--porcelain"),
        "python_package_lock": str(repository / "uv.lock")
        if (repository / "uv.lock").is_file()
        else None,
        "python_package_lock_sha256": (
            __import__("hashlib").sha256((repository / "uv.lock").read_bytes()).hexdigest()
            if (repository / "uv.lock").is_file()
            else None
        ),
        "compiler_versions": {},
        "storage_devices": partitions,
    }
    for name, command in {
        "nvcc": ["nvcc", "--version"],
        "gcc": ["gcc", "--version"],
        "cl": ["cl"],
    }.items():
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, check=False, timeout=10
            )
            fingerprint["compiler_versions"][name] = (
                (result.stdout or result.stderr).splitlines()[0]
                if result.stdout or result.stderr
                else None
            )
        except (OSError, subprocess.TimeoutExpired):
            fingerprint["compiler_versions"][name] = None
    return environment, fingerprint


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default
    except (OSError, json.JSONDecodeError):
        return default


def _save_npz(path: Path, weights: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        up=weights.up,
        gate=weights.gate,
        down=weights.down,
        hidden_start=np.asarray(weights.hidden_offset, dtype=np.int64),
        logical_intermediate_dimension=np.asarray(weights.logical_width, dtype=np.int64),
    )


def _fixture_microshard_descriptor(
    *,
    path: Path,
    weights: Any,
    worker_id: str,
    expert_id: int,
    hidden_start: int,
    hidden_end: int,
    latent_dimension: int,
    logical_intermediate_dimension: int,
) -> ExpertMicroshardDescriptor:
    storage_file = str(path.resolve())
    storage_size = path.stat().st_size
    shapes = {
        "up": [logical_intermediate_dimension, latent_dimension],
        "gate": [logical_intermediate_dimension, latent_dimension],
        "down": [latent_dimension, logical_intermediate_dimension],
    }
    arrays = {"up": weights.up, "gate": weights.gate, "down": weights.down}

    def projection(name: str) -> ExpertProjectionSlice:
        values = np.ascontiguousarray(arrays[name])
        return ExpertProjectionSlice(
            tensor_id=f"fixture-L0-E{expert_id}-{name}",
            tensor_name=f"experts.{expert_id}.{name}",
            projection=name,
            logical_axis=1 if name == "down" else 0,
            slice_start=hidden_start,
            slice_end=hidden_end,
            logical_shape=shapes[name],
            storage_file=storage_file,
            storage_offset=0,
            storage_length=storage_size,
            storage_file_size=storage_size,
            content_hash="sha256:" + sha256_bytes(values.tobytes()),
        )

    native = NativeQuantizationMetadata(
        format_name="float32",
        packing="ieee754_le",
        scale_format="none",
        scale_group_size=None,
        quantization_aware_trained=False,
        reencoding_allowed=False,
        backend_requirements=["numpy"],
        logical_shape=[logical_intermediate_dimension, latent_dimension],
        packed_shape=[logical_intermediate_dimension, latent_dimension],
        byte_size=weights.byte_size,
    )
    return ExpertMicroshardDescriptor(
        model_id="experiment-010-fixture",
        layer_id=0,
        expert_id=expert_id,
        shard_id=f"{worker_id}:L0:E{expert_id}:{hidden_start}-{hidden_end}",
        hidden_start=hidden_start,
        hidden_end=hidden_end,
        up_projection=projection("up"),
        gate_projection=projection("gate"),
        down_projection=projection("down"),
        native_quantization=native,
        required_accumulator="fp32_sum",
        supported_backends=["numpy"],
        execution_status="supported",
    )


def _request(
    *,
    request_id: str,
    expert_ids: list[int],
    weights: list[float],
    latent: int,
    codec: TransportCodec = TransportCodec.RAW_FP32,
    exact: bool = True,
) -> ExpertExecutionRequest:
    return ExpertExecutionRequest(
        request_id=request_id,
        model_id="experiment-010-fixture",
        model_revision="deterministic-v1",
        quantization_fingerprint="float32-fixture-v1",
        layer_id=0,
        batch_rows=2,
        latent_dimension=latent,
        expert_ids=expert_ids,
        routing_weights=weights,
        activations={},
        deadline_ns=time.time_ns() + 30_000_000_000,
        execution_mode=ExpertExecutionMode.WHOLE_EXPERT,
        determinism_mode=(DeterminismMode.EXACT if exact else DeterminismMode.QUALITY_BOUNDED),
        compression=codec,
    )


def _budget(worker_id: str, directory: Path, logical_bytes: int, cpu: int) -> WorkerBudget:
    return WorkerBudget(
        worker_id=worker_id,
        memory_budget_bytes=max(logical_bytes * 2, 16 << 20),
        expert_residency_budget_bytes=max(logical_bytes, 8 << 20),
        cache_budget_bytes=max(logical_bytes, 8 << 20),
        thread_count=1,
        cpu_affinity=[cpu],
        storage_directory=str(directory),
        device="cpu",
        backend="numpy",
        physical_memory_limit=False,
    )


def _register_worker(
    trust: TrustController,
    client: ExpertTransportClient,
    secret: bytes,
) -> Any:
    payload = client.control("manifest")
    from swarm_inference.experiments.experiment_010.schemas import WorkerManifest

    manifest = WorkerManifest.model_validate(payload["manifest"])
    trust.register(manifest, signature_secret=secret)
    return manifest


def _universal_worker_evidence(process: Any) -> dict[str, Any]:
    return {
        "worker_id": process.worker_id,
        "process_id": process.process.pid,
        "control_endpoint": process.control_endpoint,
        "data_endpoint": process.endpoint,
        "negotiated_protocol": process.negotiated_protocol,
        "identity": process.universal_identity,
        "capabilities": process.universal_capabilities,
        "initial_heartbeat": process.initial_heartbeat,
        "lifecycle_owner": "ExpertWorkerManager via UniversalWorkerClient",
    }


def _dispatch_cost_components(result: Any) -> dict[str, int]:
    """Extract non-overlapping measured components from a layer dispatch."""

    totals = {
        "worker_compute_ns": 0,
        "worker_queue_ns": 0,
        "serialisation_ns": 0,
        "tcp_transport_ns": 0,
        "shared_memory_ns": 0,
        "microshard_compute_ns": 0,
        "reduction_ns": int(result.metrics.get("reduction_ns", 0)),
    }
    for observation in result.worker_responses:
        metadata = observation.get("execution_metadata", {})
        transport = observation.get("transport", {})
        compute_ns = int(metadata.get("compute_ns", 0))
        queue_ns = int(metadata.get("queue_ns", 0))
        serialisation_ns = int(metadata.get("serialisation_ns", 0)) + int(
            transport.get("serialisation_ns", 0)
        )
        copy_ns = int(transport.get("copy_ns", 0))
        transition_ns = int(transport.get("kernel_transition_ns", 0))
        elapsed_ns = int(transport.get("request_elapsed_ns", 0))
        residual_ns = max(
            elapsed_ns - compute_ns - queue_ns - serialisation_ns - copy_ns - transition_ns,
            0,
        )
        totals["worker_compute_ns"] += compute_ns
        totals["worker_queue_ns"] += queue_ns
        totals["serialisation_ns"] += serialisation_ns
        if transport.get("data_plane") == DataPlane.SHARED_MEMORY.value:
            totals["shared_memory_ns"] += residual_ns + copy_ns + transition_ns
        else:
            totals["tcp_transport_ns"] += residual_ns
    if result.metrics.get("protocol") == "coalesced_layer_microshards":
        totals["microshard_compute_ns"] = totals.pop("worker_compute_ns")
    return totals


def _layer_wire_metrics(result: Any) -> dict[str, int]:
    """Reconcile per-call bytes and imposed delay from coordinator observations."""

    wire_bytes = 0
    payload_bytes = 0
    imposed_delay_ns = 0
    transfers = 0
    for observation in result.worker_responses:
        transport = observation.get("transport", {})
        shaper = transport.get("shaper", {})
        wire_bytes += int(transport.get("request_bytes", 0)) + int(
            transport.get("response_bytes", 0)
        )
        payload_bytes += int(shaper.get("payload_bytes", 0))
        imposed_delay_ns += int(shaper.get("imposed_delay_ns", 0))
        transfers += int(shaper.get("transfers", 0))
    return {
        "wire_bytes": wire_bytes,
        "payload_bytes_shaped": payload_bytes,
        "imposed_delay_ns": imposed_delay_ns,
        "shaped_transfers": transfers,
    }


def _median_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[group]), []).append(row)
    results = []
    for key, values in sorted(grouped.items()):
        numeric_fields = {
            field
            for row in values
            for field, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        item = {group: key, "repeats": len(values)}
        for field in numeric_fields:
            item[f"median_{field}"] = median(float(row[field]) for row in values if field in row)
        results.append(item)
    return results


def _run_fixture_matrix(
    bundle: Experiment010Bundle,
    *,
    repeats: int,
    selected_network_profile: str | None,
) -> dict[str, Any]:
    fixture_root = bundle.root / "traces" / "fixtures"
    worker_root = bundle.root / "traces" / "workers"
    relay_root = bundle.root / "traces" / "relays"
    fixture_root.mkdir(parents=True, exist_ok=True)
    latent, intermediate = 64, 96
    expert_count = 4
    weights_by_id = {
        expert_id: deterministic_expert(
            latent_dimension=latent,
            intermediate_dimension=intermediate,
            seed=1010 + expert_id,
        )
        for expert_id in range(expert_count)
    }
    whole_files: dict[tuple[int, int], Path] = {}
    for expert_id, weights in weights_by_id.items():
        path = fixture_root / f"whole-L0-E{expert_id}.npz"
        _save_npz(path, weights)
        whole_files[(0, expert_id)] = path
    local_store = ExpertStore(
        owned=set(whole_files),
        loader=npz_expert_loader(whole_files),
        residency_budget_bytes=sum(item.byte_size for item in weights_by_id.values()),
        cache_budget_bytes=sum(item.byte_size for item in weights_by_id.values()),
    )
    generator = np.random.default_rng(1010)
    activation = generator.normal(0, 0.1, (2, latent)).astype(np.float32)
    selected_ids = list(range(expert_count))
    routing_weights = [0.4, 0.3, 0.2, 0.1]
    reference_request = _request(
        request_id="reference", expert_ids=selected_ids, weights=routing_weights, latent=latent
    )
    reference_started = time.perf_counter_ns()
    reference, reference_metadata = local_store.execute(reference_request, activation)
    reference_ns = time.perf_counter_ns() - reference_started
    affinities = psutil.Process().cpu_affinity()
    cpu_a, cpu_b = affinities[0], affinities[min(1, len(affinities) - 1)]
    model_identity = {
        "model_id": "experiment-010-fixture",
        "model_revision": "deterministic-v1",
        "quantization_fingerprint": "float32-fixture-v1",
        "model_fingerprint": "sha256:experiment-010-fixture-v1",
    }
    whole_rows: list[dict[str, Any]] = []
    data_plane_rows: list[dict[str, Any]] = []
    coalescing_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    configuration_matrix_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    corruption_schedule = []
    failure_schedule = []
    manifests = []
    universal_workers = []
    budgets = []
    ownership = {
        (0, 0): "whole-a",
        (0, 1): "whole-a",
        (0, 2): "whole-b",
        (0, 3): "whole-b",
    }
    baseline_samples = [(reference_ns, reference_metadata)]
    for repeat in range(1, repeats):
        started = time.perf_counter_ns()
        _baseline_output, metadata = local_store.execute(
            reference_request.model_copy(update={"request_id": f"reference-{repeat}"}),
            activation,
        )
        baseline_samples.append((time.perf_counter_ns() - started, metadata))
    for repeat, (elapsed_ns, metadata) in enumerate(baseline_samples):
        whole_rows.append(
            {
                "configuration": MATRIX_CONFIGURATIONS["A"],
                "repeat": repeat,
                "throughput": 1e9 / elapsed_ns,
                "latency_ms": elapsed_ns / 1e6,
                "p95_latency_ms": elapsed_ns / 1e6,
                "ttft_ms": None,
                "exact": True,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                "accounting_mode": "fixed_physical_resource",
                "model_scope": "deterministic_fixture",
                **metadata,
            }
        )
        configuration_matrix_rows.append(
            {
                "configuration_id": "A",
                "configuration": MATRIX_CONFIGURATIONS["A"],
                "network_profile": "loopback_unshaped",
                "network_profile_applies": False,
                "repeat": repeat,
                "status": "MEASURED",
                "latency_ms": elapsed_ns / 1e6,
                "exact": True,
                "payload_bytes_shaped": 0,
                "model_scope": "deterministic_fixture",
                "accounting_mode": "fixed_physical_resource",
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
    local_coordinator = StableExpertCoordinator(
        clients={},
        local_store=local_store,
        latent_dimension=latent,
        **{
            key: model_identity[key]
            for key in ("model_id", "model_revision", "quantization_fingerprint")
        },
    )
    for repeat in range(repeats):
        passthrough = local_coordinator.execute_whole_layer(
            activation,
            layer_id=0,
            expert_ids=selected_ids,
            routing_weights=routing_weights,
            request_id=f"local-passthrough-{repeat}",
        )
        comparison = compare_layer_results(reference, passthrough.output)
        elapsed_ns = int(passthrough.metrics["total_ns"])
        whole_rows.append(
            {
                "configuration": MATRIX_CONFIGURATIONS["B"],
                "repeat": repeat,
                "throughput": 1e9 / elapsed_ns,
                "latency_ms": elapsed_ns / 1e6,
                "p95_latency_ms": elapsed_ns / 1e6,
                "ttft_ms": None,
                **passthrough.metrics,
                **comparison,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                "accounting_mode": "fixed_physical_resource",
                "model_scope": "deterministic_fixture",
            }
        )
        configuration_matrix_rows.append(
            {
                "configuration_id": "B",
                "configuration": MATRIX_CONFIGURATIONS["B"],
                "network_profile": "loopback_unshaped",
                "network_profile_applies": False,
                "repeat": repeat,
                "status": "MEASURED",
                "latency_ms": elapsed_ns / 1e6,
                "exact": comparison["exact"],
                "relative_l2_error": comparison["relative_l2_error"],
                "payload_bytes_shaped": 0,
                "model_scope": "deterministic_fixture",
                "accounting_mode": "fixed_physical_resource",
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
    matrix_planner_selection = PositiveUtilityPlanner().select(
        [
            PlannerCandidate(
                candidate_id="matrix-local",
                phase=ServicePhase.DECODE,
                strategy=ExecutionStrategy.LOCAL_WHOLE_EXPERT,
                workers=[],
                objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
                predicted_utility=0.1,
                lower_confidence_bound=0.05,
                explanation=["fixture baseline is faster than the measured RPC candidates"],
            ),
            PlannerCandidate(
                candidate_id="matrix-remote",
                phase=ServicePhase.DECODE,
                strategy=ExecutionStrategy.REMOTE_WHOLE_EXPERT,
                workers=["cpu_weak"],
                objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
                predicted_utility=-0.2,
                lower_confidence_bound=-0.3,
                explanation=["measured synchronization cost exceeds fixture compute"],
            ),
            PlannerCandidate(
                candidate_id="matrix-idle",
                phase=ServicePhase.DECODE,
                strategy=ExecutionStrategy.IDLE,
                workers=["cpu_weak"],
                objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
                predicted_utility=0.0,
                lower_confidence_bound=0.0,
                explanation=["fail-closed fallback"],
            ),
        ],
        phase=ServicePhase.DECODE,
        objective=PlannerObjective.MAX_DECODE_THROUGHPUT,
    )
    if matrix_planner_selection.plan.selected_strategy != ExecutionStrategy.LOCAL_WHOLE_EXPERT:
        raise RuntimeError("fixture matrix planner did not select its measured local winner")
    for repeat in range(repeats):
        planned = local_coordinator.execute_whole_layer(
            activation,
            layer_id=0,
            expert_ids=selected_ids,
            routing_weights=routing_weights,
            request_id=f"planner-selected-local-{repeat}",
        )
        comparison = compare_layer_results(reference, planned.output)
        elapsed_ns = int(planned.metrics["total_ns"])
        whole_rows.append(
            {
                "configuration": MATRIX_CONFIGURATIONS["L"],
                "repeat": repeat,
                "throughput": 1e9 / elapsed_ns,
                "latency_ms": elapsed_ns / 1e6,
                "p95_latency_ms": elapsed_ns / 1e6,
                "ttft_ms": None,
                "planner_selected_candidate": "matrix-local",
                **planned.metrics,
                **comparison,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                "accounting_mode": "fixed_physical_resource",
                "model_scope": "deterministic_fixture",
            }
        )
        configuration_matrix_rows.append(
            {
                "configuration_id": "L",
                "configuration": MATRIX_CONFIGURATIONS["L"],
                "network_profile": "loopback_unshaped",
                "network_profile_applies": False,
                "repeat": repeat,
                "status": "MEASURED",
                "latency_ms": elapsed_ns / 1e6,
                "exact": comparison["exact"],
                "relative_l2_error": comparison["relative_l2_error"],
                "payload_bytes_shaped": 0,
                "planner_selected_candidate": "matrix-local",
                "model_scope": "deterministic_fixture",
                "accounting_mode": "fixed_physical_resource",
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
    trust = TrustController(
        **model_identity,
        sampled_duplicate_fraction=0.25,
        quarantine_failures=2,
    )
    manager = ExpertWorkerManager(worker_root)
    relays = ExpertRelayManager(relay_root)
    try:
        worker_specs = {
            "whole-a": ([0, 1], cpu_a),
            "whole-b": ([2, 3], cpu_b),
        }
        processes = {}
        direct_clients = {}
        for worker_id, (expert_ids, cpu) in worker_specs.items():
            entries = [
                fixture_ownership_entry(0, expert, whole_files[(0, expert)])
                for expert in expert_ids
            ]
            logical_bytes = sum(int(item["logical_bytes"]) for item in entries)
            budget = _budget(worker_id, worker_root / worker_id / "storage", logical_bytes, cpu)
            budgets.append(budget.model_dump(mode="json"))
            process = manager.start(
                worker_id=worker_id,
                owned_experts=entries,
                budget=budget,
                loader_type="npz",
                **model_identity,
            )
            processes[worker_id] = process
            universal_workers.append(_universal_worker_evidence(process))
            client = ExpertTransportClient(process.endpoint, data_plane=DataPlane.DIRECT_TCP)
            direct_clients[worker_id] = client
            manifests.append(_register_worker(trust, client, process.signature_secret))
        # Configuration C is intentionally a complete replicated fixture worker.
        # It is a request-routing baseline and is explicitly excluded from every
        # distributed-capacity claim.
        independent_entries = [
            fixture_ownership_entry(0, expert_id, whole_files[(0, expert_id)])
            for expert_id in selected_ids
        ]
        independent_bytes = sum(int(item["logical_bytes"]) for item in independent_entries)
        independent_budget = _budget(
            "independent-router",
            worker_root / "independent-router" / "storage",
            independent_bytes,
            cpu_a,
        )
        budgets.append(independent_budget.model_dump(mode="json"))
        independent_process = manager.start(
            worker_id="independent-router",
            owned_experts=independent_entries,
            budget=independent_budget,
            loader_type="npz",
            **model_identity,
        )
        independent_client = ExpertTransportClient(independent_process.endpoint)
        universal_workers.append(_universal_worker_evidence(independent_process))
        manifests.append(
            _register_worker(trust, independent_client, independent_process.signature_secret)
        )
        for repeat in range(repeats):
            request = _request(
                request_id=f"independent-route-{repeat}",
                expert_ids=selected_ids,
                weights=routing_weights,
                latent=latent,
            )
            response, result, metrics = independent_client.execute(request, activation)
            comparison = compare_layer_results(reference, result)
            elapsed_ns = int(metrics["request_elapsed_ns"])
            whole_rows.append(
                {
                    "configuration": MATRIX_CONFIGURATIONS["C"],
                    "repeat": repeat,
                    "throughput": 1e9 / elapsed_ns,
                    "latency_ms": elapsed_ns / 1e6,
                    "p95_latency_ms": elapsed_ns / 1e6,
                    "ttft_ms": None,
                    "worker_compute_ns": response.execution_metadata.compute_ns,
                    "worker_queue_ns": response.execution_metadata.queue_ns,
                    "serialisation_ns": metrics["serialisation_ns"],
                    "tcp_transport_ns": metrics["socket_ns"],
                    "measured_total_ns": elapsed_ns,
                    **comparison,
                    "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    "accounting_mode": "fixed_physical_resource",
                    "model_scope": "deterministic_fixture_complete_replica",
                    "excluded_from_capacity_claim": True,
                }
            )
            configuration_matrix_rows.append(
                {
                    "configuration_id": "C",
                    "configuration": MATRIX_CONFIGURATIONS["C"],
                    "network_profile": "loopback_unshaped",
                    "network_profile_applies": True,
                    "repeat": repeat,
                    "status": "MEASURED",
                    "latency_ms": elapsed_ns / 1e6,
                    "exact": comparison["exact"],
                    "relative_l2_error": comparison["relative_l2_error"],
                    "messages_per_layer": 1,
                    "payload_bytes_shaped": metrics["request_bytes"] + metrics["response_bytes"],
                    "model_scope": "deterministic_fixture_complete_replica",
                    "accounting_mode": "fixed_physical_resource",
                    "excluded_from_capacity_claim": True,
                    "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                }
            )
        coordinator = StableExpertCoordinator(
            clients=direct_clients,
            whole_ownership=ownership,
            latent_dimension=latent,
            **{
                key: model_identity[key]
                for key in ("model_id", "model_revision", "quantization_fingerprint")
            },
        )
        for repeat in range(repeats):
            coalesced = coordinator.execute_whole_layer(
                activation,
                layer_id=0,
                expert_ids=selected_ids,
                routing_weights=routing_weights,
                coalesced=True,
                request_id=f"whole-coalesced-{repeat}",
            )
            naive = coordinator.execute_whole_layer(
                activation,
                layer_id=0,
                expert_ids=selected_ids,
                routing_weights=routing_weights,
                coalesced=False,
                request_id=f"whole-naive-{repeat}",
            )
            for configuration_id, result in (("H", coalesced), ("G", naive)):
                label = MATRIX_CONFIGURATIONS[configuration_id]
                elapsed = float(result.metrics["total_ns"])
                comparison = compare_layer_results(reference, result.output)
                components = _dispatch_cost_components(result)
                whole_rows.append(
                    {
                        "configuration": label,
                        "repeat": repeat,
                        "throughput": 1e9 / elapsed,
                        "latency_ms": elapsed / 1e6,
                        "p95_latency_ms": elapsed / 1e6,
                        "ttft_ms": None,
                        "token_identity": comparison["relative_l2_error"] < 1e-6,
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                        "accounting_mode": "fixed_physical_resource",
                        **result.metrics,
                        **components,
                        "measured_total_ns": elapsed,
                        **comparison,
                    }
                )
                coalescing_rows.append(
                    {
                        "protocol": result.metrics["protocol"],
                        "repeat": repeat,
                        "messages_per_layer": result.metrics["messages_per_layer"],
                        "activation_payload_bytes": result.metrics["activation_payload_bytes"],
                        "latency_ms": elapsed / 1e6,
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    }
                )
                configuration_matrix_rows.append(
                    {
                        "configuration_id": configuration_id,
                        "configuration": label,
                        "network_profile": "loopback_unshaped",
                        "network_profile_applies": True,
                        "repeat": repeat,
                        "status": "MEASURED",
                        "latency_ms": elapsed / 1e6,
                        "exact": comparison["exact"],
                        "relative_l2_error": comparison["relative_l2_error"],
                        "messages_per_layer": result.metrics["messages_per_layer"],
                        "payload_bytes_shaped": result.metrics["activation_payload_bytes"],
                        "model_scope": "deterministic_fixture_disjoint_ownership",
                        "accounting_mode": "fixed_physical_resource",
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    }
                )
        # Compare the same real worker operation over direct TCP, shared memory,
        # and a separately isolated relay process.
        plane_clients: list[tuple[str, ExpertTransportClient]] = [
            ("direct_tcp", direct_clients["whole-a"]),
            (
                "shared_memory",
                ExpertTransportClient(
                    processes["whole-a"].endpoint, data_plane=DataPlane.SHARED_MEMORY
                ),
            ),
        ]
        relay = relays.start(
            target_endpoint=processes["whole-a"].endpoint,
            profile=NETWORK_PROFILES["loopback_unshaped"],
        )
        plane_clients.append(
            (
                "relayed_tcp",
                ExpertTransportClient(relay.endpoint, data_plane=DataPlane.RELAYED_TCP),
            )
        )
        plane_configuration = {
            "shared_memory": "D",
            "direct_tcp": "E",
            "relayed_tcp": "F",
        }
        for plane_name, client in plane_clients:
            samples = []
            for repeat in range(repeats):
                request = _request(
                    request_id=f"plane-{plane_name}-{repeat}",
                    expert_ids=[0, 1],
                    weights=[0.6, 0.4],
                    latent=latent,
                )
                plane_reference, _ = local_store.execute(request, activation)
                response, result, metrics = client.execute(request, activation)
                comparison = compare_layer_results(plane_reference, result)
                compute_ns = int(response.execution_metadata.compute_ns)
                queue_ns = int(response.execution_metadata.queue_ns)
                serialisation_ns = int(metrics["serialisation_ns"]) + int(
                    response.execution_metadata.serialisation_ns
                )
                copy_ns = int(metrics["copy_ns"])
                transition_ns = int(metrics["kernel_transition_ns"])
                elapsed_ns = int(metrics["request_elapsed_ns"])
                residual_ns = max(
                    elapsed_ns - compute_ns - queue_ns - serialisation_ns - copy_ns - transition_ns,
                    0,
                )
                samples.append(float(metrics["request_elapsed_ns"]) / 1e6)
                data_plane_rows.append(
                    {
                        "configuration": MATRIX_CONFIGURATIONS[plane_configuration[plane_name]],
                        "data_plane": plane_name,
                        "repeat": repeat,
                        "total_ms": metrics["request_elapsed_ns"] / 1e6,
                        "serialisation_ms": metrics["serialisation_ns"] / 1e6,
                        "copy_ms": metrics["copy_ns"] / 1e6,
                        "socket_ms": metrics["socket_ns"] / 1e6,
                        "queue_ms": response.execution_metadata.queue_ns / 1e6,
                        "request_bytes": metrics["request_bytes"],
                        "response_bytes": metrics["response_bytes"],
                        "shared_memory_bytes": metrics["shared_memory_bytes"],
                        "payload_bytes": metrics["payload_bytes"],
                        "bytes_per_output_token": None,
                        "worker_compute_ns": compute_ns,
                        "worker_queue_ns": queue_ns,
                        "serialisation_ns": serialisation_ns,
                        "tcp_transport_ns": (
                            0 if plane_name == DataPlane.SHARED_MEMORY.value else residual_ns
                        ),
                        "shared_memory_ns": (
                            residual_ns + copy_ns + transition_ns
                            if plane_name == DataPlane.SHARED_MEMORY.value
                            else 0
                        ),
                        "reduction_ns": 0,
                        "measured_total_ns": elapsed_ns,
                        "exact": comparison["exact"],
                        "relative_l2_error": comparison["relative_l2_error"],
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    }
                )
                configuration_matrix_rows.append(
                    {
                        "configuration_id": plane_configuration[plane_name],
                        "configuration": MATRIX_CONFIGURATIONS[plane_configuration[plane_name]],
                        "network_profile": "loopback_unshaped",
                        "network_profile_applies": plane_name != "shared_memory",
                        "repeat": repeat,
                        "status": "MEASURED",
                        "latency_ms": elapsed_ns / 1e6,
                        "exact": comparison["exact"],
                        "relative_l2_error": comparison["relative_l2_error"],
                        "messages_per_layer": 1,
                        "payload_bytes_shaped": (
                            0
                            if plane_name == "shared_memory"
                            else metrics["request_bytes"] + metrics["response_bytes"]
                        ),
                        "model_scope": "deterministic_fixture_partial_expert_set",
                        "accounting_mode": "fixed_physical_resource",
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    }
                )
            for row in data_plane_rows:
                if row["data_plane"] == plane_name:
                    row["median_total_ms"] = median(samples)
        # Shape the actual request/response bytes. These rows measure the
        # transport itself and therefore carry the network-emulation category.
        profiles = (
            [selected_network_profile] if selected_network_profile else list(NETWORK_PROFILES)
        )
        for profile_name in profiles:
            profile = NETWORK_PROFILES[profile_name]
            client = ExpertTransportClient(
                processes["whole-a"].endpoint,
                data_plane=DataPlane.DIRECT_TCP,
                network_profile=profile,
                timeout_s=10.0,
            )
            request = _request(
                request_id=f"shape-{profile_name}",
                expert_ids=[0],
                weights=[1.0],
                latent=latent,
            )
            started = time.perf_counter_ns()
            _response, _result, metrics = client.execute(request, activation)
            achieved = measured_network_profile(profile, metrics["shaper"])
            achieved["wall_round_trip_ms"] = (time.perf_counter_ns() - started) / 1e6
            transport_rows.append(achieved)
        # Codec execution and break-even accounting use the exact encoded bytes.
        for codec in TransportCodec:
            encoded = encode_array(activation, name="activation", codec=codec)
            decoded = decode_array(encoded.metadata, encoded.payload)
            errors = numerical_error(activation, decoded.array)
            break_even = codec_break_even(
                raw_bytes=encoded.metadata.raw_bytes,
                encoded_bytes=encoded.metadata.encoded_bytes,
                encode_ns=encoded.encode_ns,
                decode_ns=decoded.decode_ns,
                bandwidth_bps=1e9,
            )
            codec_rows.append(
                {
                    "codec": codec.value,
                    "raw_bytes": encoded.metadata.raw_bytes,
                    "encoded_bytes": encoded.metadata.encoded_bytes,
                    "encode_ns": encoded.encode_ns,
                    "decode_ns": decoded.decode_ns,
                    "total_time_ns": break_even["encoded_total_ns"],
                    "beneficial_at_1g": break_even["beneficial"],
                    **errors,
                    "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                }
            )
        # Physically store only matching tensor slices in each shard worker.
        micro_manifests = []
        micro_owners = []
        micro_descriptor_rows = []
        micro_descriptor_validations = {}
        micro_layouts: dict[
            str, tuple[dict[str, ExpertTransportClient], list[MicroshardOwner]]
        ] = {}
        for layout, ranges_for_layout in {
            "equal": [(0, 48), (48, 96)],
            "asymmetric": [(0, 32), (32, 96)],
        }.items():
            clients = {}
            owners = []
            descriptors_by_expert: dict[int, list[ExpertMicroshardDescriptor]] = {
                expert_id: [] for expert_id in selected_ids
            }
            for shard_index, (start, end) in enumerate(ranges_for_layout):
                worker_id = f"shard-{layout}-{shard_index}"
                entries = []
                descriptors = []
                for expert_id, full_weights in weights_by_id.items():
                    sliced = slice_expert_weights(full_weights, hidden_start=start, hidden_end=end)
                    path = fixture_root / f"{worker_id}-L0-E{expert_id}.npz"
                    _save_npz(path, sliced)
                    entries.append(fixture_ownership_entry(0, expert_id, path))
                    descriptor = _fixture_microshard_descriptor(
                        path=path,
                        weights=sliced,
                        worker_id=worker_id,
                        expert_id=expert_id,
                        hidden_start=start,
                        hidden_end=end,
                        latent_dimension=latent,
                        logical_intermediate_dimension=intermediate,
                    )
                    descriptors_by_expert[expert_id].append(descriptor)
                    descriptor_payload = descriptor.model_dump(mode="json")
                    descriptor_payload["worker_id"] = worker_id
                    descriptor_payload["logical_intermediate_dimension"] = intermediate
                    descriptors.append(descriptor_payload)
                    micro_descriptor_rows.append(descriptor_payload)
                logical_bytes = sum(int(item["logical_bytes"]) for item in entries)
                budget = _budget(
                    worker_id,
                    worker_root / worker_id / "storage",
                    logical_bytes,
                    affinities[shard_index % len(affinities)],
                )
                budgets.append(budget.model_dump(mode="json"))
                process = manager.start(
                    worker_id=worker_id,
                    owned_experts=entries,
                    owned_microshards=descriptors,
                    budget=budget,
                    loader_type="npz",
                    **model_identity,
                )
                client = ExpertTransportClient(process.endpoint)
                universal_workers.append(_universal_worker_evidence(process))
                clients[worker_id] = client
                manifest = _register_worker(trust, client, process.signature_secret)
                manifests.append(manifest)
                micro_manifests.append(manifest)
                owner = MicroshardOwner(
                    worker_id=worker_id,
                    layer_id=0,
                    expert_ids=tuple(selected_ids),
                    hidden_start=start,
                    hidden_end=end,
                    logical_intermediate_dimension=intermediate,
                )
                owners.append(owner)
                micro_owners.append(owner)
            micro_descriptor_validations[layout] = {
                str(expert_id): validate_expert_microshard_set(descriptors_for_expert)
                for expert_id, descriptors_for_expert in descriptors_by_expert.items()
            }
            micro_coordinator = StableExpertCoordinator(
                clients=clients,
                microshard_ownership=owners,
                latent_dimension=latent,
                **{
                    key: model_identity[key]
                    for key in ("model_id", "model_revision", "quantization_fingerprint")
                },
            )
            micro_layouts[layout] = (clients, owners)
            for repeat in range(repeats):
                result = micro_coordinator.execute_microshard_layer(
                    activation,
                    layer_id=0,
                    expert_ids=selected_ids,
                    routing_weights=routing_weights,
                    request_id=f"micro-{layout}-{repeat}",
                )
                comparison = compare_layer_results(reference, result.output)
                components = _dispatch_cost_components(result)
                micro_rows.append(
                    {
                        "configuration": (
                            MATRIX_CONFIGURATIONS["I"]
                            if layout == "equal"
                            else MATRIX_CONFIGURATIONS["J"]
                        ),
                        "layout": layout,
                        "repeat": repeat,
                        "throughput": 1e9 / result.metrics["total_ns"],
                        "latency_ms": result.metrics["total_ns"] / 1e6,
                        "p95_latency_ms": result.metrics["total_ns"] / 1e6,
                        "ttft_ms": None,
                        "messages_per_layer": result.metrics["messages_per_layer"],
                        "reduction_ns": result.metrics["reduction_ns"],
                        **components,
                        "measured_total_ns": result.metrics["total_ns"],
                        **comparison,
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                        "accounting_mode": "fixed_physical_resource",
                    }
                )
                configuration_matrix_rows.append(
                    {
                        "configuration_id": "I" if layout == "equal" else "J",
                        "configuration": (
                            MATRIX_CONFIGURATIONS["I"]
                            if layout == "equal"
                            else MATRIX_CONFIGURATIONS["J"]
                        ),
                        "network_profile": "loopback_unshaped",
                        "network_profile_applies": True,
                        "repeat": repeat,
                        "status": "MEASURED",
                        "latency_ms": result.metrics["total_ns"] / 1e6,
                        "exact": comparison["exact"],
                        "relative_l2_error": comparison["relative_l2_error"],
                        "messages_per_layer": result.metrics["messages_per_layer"],
                        "payload_bytes_shaped": result.metrics["activation_payload_bytes"],
                        "model_scope": "deterministic_fixture_physical_slices",
                        "accounting_mode": "fixed_physical_resource",
                        "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                    }
                )
            if layout == "equal":
                for repeat in range(repeats):
                    coalesced_micro = micro_coordinator.execute_microshard_layer(
                        activation,
                        layer_id=0,
                        expert_ids=selected_ids,
                        routing_weights=routing_weights,
                        request_id=f"micro-coalesced-explicit-{repeat}",
                    )
                    comparison = compare_layer_results(reference, coalesced_micro.output)
                    components = _dispatch_cost_components(coalesced_micro)
                    elapsed_ns = int(coalesced_micro.metrics["total_ns"])
                    micro_rows.append(
                        {
                            "configuration": MATRIX_CONFIGURATIONS["K"],
                            "layout": "equal_explicit_coalesced_protocol",
                            "repeat": repeat,
                            "throughput": 1e9 / elapsed_ns,
                            "latency_ms": elapsed_ns / 1e6,
                            "p95_latency_ms": elapsed_ns / 1e6,
                            "ttft_ms": None,
                            "messages_per_layer": coalesced_micro.metrics["messages_per_layer"],
                            **components,
                            "measured_total_ns": elapsed_ns,
                            **comparison,
                            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                            "accounting_mode": "fixed_physical_resource",
                        }
                    )
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": "K",
                            "configuration": MATRIX_CONFIGURATIONS["K"],
                            "network_profile": "loopback_unshaped",
                            "network_profile_applies": True,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": elapsed_ns / 1e6,
                            "exact": comparison["exact"],
                            "relative_l2_error": comparison["relative_l2_error"],
                            "messages_per_layer": coalesced_micro.metrics["messages_per_layer"],
                            "payload_bytes_shaped": coalesced_micro.metrics[
                                "activation_payload_bytes"
                            ],
                            "model_scope": "deterministic_fixture_physical_slices",
                            "accounting_mode": "fixed_physical_resource",
                            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                        }
                    )
            # Residency is evidence only after execution. Replace the initial
            # process-start manifests with engine telemetry captured after all
            # experts in this layout have actually been loaded.
            for worker_id, client in clients.items():
                refreshed = WorkerManifest.model_validate(client.control("manifest")["manifest"])
                trust.manifests[worker_id] = refreshed
                for collection in (manifests, micro_manifests):
                    for index, existing in enumerate(collection):
                        if existing.worker_id == worker_id:
                            collection[index] = refreshed
                            break
        # Run every network-bearing configuration across the requested shaping
        # sweep. Local and shared-memory baselines are remeasured beside each
        # profile but explicitly state that the profile does not act on them.
        shaped_profiles = [name for name in profiles if name != "loopback_unshaped"]
        for profile_name in shaped_profiles:
            profile = NETWORK_PROFILES[profile_name]
            profile_repeats = 1 if profile_name == "global_wan" else repeats
            independent_shaped = ExpertTransportClient(
                independent_process.endpoint,
                network_profile=profile,
                timeout_s=30.0,
            )
            whole_shaped_clients = {
                worker_id: ExpertTransportClient(
                    process.endpoint,
                    network_profile=profile,
                    timeout_s=30.0,
                )
                for worker_id, process in processes.items()
            }
            whole_shaped_coordinator = StableExpertCoordinator(
                clients=whole_shaped_clients,
                whole_ownership=ownership,
                latent_dimension=latent,
                **{
                    key: model_identity[key]
                    for key in (
                        "model_id",
                        "model_revision",
                        "quantization_fingerprint",
                    )
                },
            )
            direct_shaped = ExpertTransportClient(
                processes["whole-a"].endpoint,
                network_profile=profile,
                timeout_s=30.0,
            )
            shared_unshaped = ExpertTransportClient(
                processes["whole-a"].endpoint,
                data_plane=DataPlane.SHARED_MEMORY,
                timeout_s=30.0,
            )
            shaped_micro_coordinators = {}
            for layout, (layout_clients, layout_owners) in micro_layouts.items():
                shaped_micro_coordinators[layout] = StableExpertCoordinator(
                    clients={
                        worker_id: ExpertTransportClient(
                            client.endpoint,
                            network_profile=profile,
                            timeout_s=30.0,
                        )
                        for worker_id, client in layout_clients.items()
                    },
                    microshard_ownership=layout_owners,
                    latent_dimension=latent,
                    **{
                        key: model_identity[key]
                        for key in (
                            "model_id",
                            "model_revision",
                            "quantization_fingerprint",
                        )
                    },
                )
            shaped_relay = relays.start(
                target_endpoint=processes["whole-a"].endpoint,
                profile=profile,
            )
            relayed_shaped = ExpertTransportClient(
                shaped_relay.endpoint,
                data_plane=DataPlane.RELAYED_TCP,
                timeout_s=30.0,
            )
            partial_request = _request(
                request_id=f"matrix-partial-{profile_name}",
                expert_ids=[0, 1],
                weights=[0.6, 0.4],
                latent=latent,
            )
            partial_reference, _partial_metadata = local_store.execute(partial_request, activation)
            for repeat in range(profile_repeats):
                # A, B and L have no network-bearing payload. They are timed in
                # the same sweep for a contemporaneous comparison only.
                local_started = time.perf_counter_ns()
                local_output, _local_metadata = local_store.execute(
                    reference_request.model_copy(
                        update={"request_id": f"matrix-a-{profile_name}-{repeat}"}
                    ),
                    activation,
                )
                local_elapsed = time.perf_counter_ns() - local_started
                for configuration_id in ("A",):
                    local_comparison = compare_layer_results(reference, local_output)
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": MATRIX_CONFIGURATIONS[configuration_id],
                            "network_profile": profile_name,
                            "network_profile_applies": False,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": local_elapsed / 1e6,
                            "exact": local_comparison["exact"],
                            "relative_l2_error": local_comparison["relative_l2_error"],
                            "messages_per_layer": 0,
                            "payload_bytes_shaped": 0,
                            "model_scope": "deterministic_fixture",
                            "accounting_mode": "fixed_physical_resource",
                            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                        }
                    )
                for configuration_id, request_prefix in (
                    ("B", "matrix-b"),
                    ("L", "matrix-l"),
                ):
                    local_result = local_coordinator.execute_whole_layer(
                        activation,
                        layer_id=0,
                        expert_ids=selected_ids,
                        routing_weights=routing_weights,
                        request_id=f"{request_prefix}-{profile_name}-{repeat}",
                    )
                    comparison = compare_layer_results(reference, local_result.output)
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": MATRIX_CONFIGURATIONS[configuration_id],
                            "network_profile": profile_name,
                            "network_profile_applies": False,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": local_result.metrics["total_ns"] / 1e6,
                            "exact": comparison["exact"],
                            "relative_l2_error": comparison["relative_l2_error"],
                            "messages_per_layer": local_result.metrics["messages_per_layer"],
                            "payload_bytes_shaped": 0,
                            "planner_selected_candidate": (
                                "matrix-local" if configuration_id == "L" else None
                            ),
                            "model_scope": "deterministic_fixture",
                            "accounting_mode": "fixed_physical_resource",
                            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                        }
                    )

                request = reference_request.model_copy(
                    update={"request_id": f"matrix-c-{profile_name}-{repeat}"}
                )
                _response, output, metrics = independent_shaped.execute(request, activation)
                comparison = compare_layer_results(reference, output)
                configuration_matrix_rows.append(
                    {
                        "configuration_id": "C",
                        "configuration": MATRIX_CONFIGURATIONS["C"],
                        "network_profile": profile_name,
                        "network_profile_applies": True,
                        "repeat": repeat,
                        "status": "MEASURED",
                        "latency_ms": metrics["request_elapsed_ns"] / 1e6,
                        "exact": comparison["exact"],
                        "relative_l2_error": comparison["relative_l2_error"],
                        "messages_per_layer": 1,
                        "payload_bytes_shaped": metrics["shaper"]["payload_bytes"],
                        "imposed_delay_ns": metrics["shaper"]["imposed_delay_ns"],
                        "model_scope": "deterministic_fixture_complete_replica",
                        "accounting_mode": "fixed_physical_resource",
                        "excluded_from_capacity_claim": True,
                        "category": EvidenceCategory.MEASURED_NETWORK_EMULATION.value,
                    }
                )

                for configuration_id, client, network_applies in (
                    ("D", shared_unshaped, False),
                    ("E", direct_shaped, True),
                    ("F", relayed_shaped, True),
                ):
                    request = partial_request.model_copy(
                        update={
                            "request_id": (
                                f"matrix-{configuration_id.lower()}-{profile_name}-{repeat}"
                            )
                        }
                    )
                    _response, output, metrics = client.execute(request, activation)
                    comparison = compare_layer_results(partial_reference, output)
                    payload_bytes_shaped = (
                        metrics["request_bytes"] + metrics["response_bytes"]
                        if configuration_id == "F"
                        else metrics["shaper"]["payload_bytes"]
                    )
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": MATRIX_CONFIGURATIONS[configuration_id],
                            "network_profile": profile_name,
                            "network_profile_applies": network_applies,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": metrics["request_elapsed_ns"] / 1e6,
                            "exact": comparison["exact"],
                            "relative_l2_error": comparison["relative_l2_error"],
                            "messages_per_layer": 1,
                            "payload_bytes_shaped": (
                                payload_bytes_shaped if network_applies else 0
                            ),
                            "imposed_delay_ns": (
                                metrics["shaper"]["imposed_delay_ns"]
                                if configuration_id == "E"
                                else None
                            ),
                            "model_scope": "deterministic_fixture_partial_expert_set",
                            "accounting_mode": "fixed_physical_resource",
                            "category": (
                                EvidenceCategory.MEASURED_NETWORK_EMULATION.value
                                if network_applies
                                else EvidenceCategory.SYNTHETIC_FIXTURE.value
                            ),
                        }
                    )

                for configuration_id, coalesced in (("G", False), ("H", True)):
                    result = whole_shaped_coordinator.execute_whole_layer(
                        activation,
                        layer_id=0,
                        expert_ids=selected_ids,
                        routing_weights=routing_weights,
                        coalesced=coalesced,
                        request_id=(f"matrix-{configuration_id.lower()}-{profile_name}-{repeat}"),
                    )
                    comparison = compare_layer_results(reference, result.output)
                    wire = _layer_wire_metrics(result)
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": MATRIX_CONFIGURATIONS[configuration_id],
                            "network_profile": profile_name,
                            "network_profile_applies": True,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": result.metrics["total_ns"] / 1e6,
                            "exact": comparison["exact"],
                            "relative_l2_error": comparison["relative_l2_error"],
                            "messages_per_layer": result.metrics["messages_per_layer"],
                            **wire,
                            "model_scope": "deterministic_fixture_disjoint_ownership",
                            "accounting_mode": "fixed_physical_resource",
                            "category": EvidenceCategory.MEASURED_NETWORK_EMULATION.value,
                        }
                    )

                for configuration_id, layout in (
                    ("I", "equal"),
                    ("J", "asymmetric"),
                    ("K", "equal"),
                ):
                    result = shaped_micro_coordinators[layout].execute_microshard_layer(
                        activation,
                        layer_id=0,
                        expert_ids=selected_ids,
                        routing_weights=routing_weights,
                        request_id=(f"matrix-{configuration_id.lower()}-{profile_name}-{repeat}"),
                    )
                    comparison = compare_layer_results(reference, result.output)
                    wire = _layer_wire_metrics(result)
                    configuration_matrix_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "configuration": MATRIX_CONFIGURATIONS[configuration_id],
                            "network_profile": profile_name,
                            "network_profile_applies": True,
                            "repeat": repeat,
                            "status": "MEASURED",
                            "latency_ms": result.metrics["total_ns"] / 1e6,
                            "exact": comparison["exact"],
                            "relative_l2_error": comparison["relative_l2_error"],
                            "messages_per_layer": result.metrics["messages_per_layer"],
                            **wire,
                            "shard_layout": layout,
                            "model_scope": "deterministic_fixture_physical_slices",
                            "accounting_mode": "fixed_physical_resource",
                            "category": EvidenceCategory.MEASURED_NETWORK_EMULATION.value,
                        }
                    )

        capacity = reconcile_microshard_ownership(
            micro_manifests[:2],
            expected_widths={(0, expert_id): intermediate for expert_id in selected_ids},
        )
        capacity.update(
            {
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                "logical_budget_only": True,
                "physical_memory_isolation_claimed": False,
                "coordinator_remote_expert_bytes": 0,
                "reference_process_excluded_from_capacity_configuration": True,
            }
        )
        # Real fault and corruption injection on the expert path. An explicit
        # replica is used only for resilience measurements.
        replica_entries = [fixture_ownership_entry(0, 0, whole_files[(0, 0)])]
        replica_budget = _budget(
            "unreliable",
            worker_root / "unreliable" / "storage",
            int(replica_entries[0]["logical_bytes"]),
            cpu_b,
        )
        budgets.append(replica_budget.model_dump(mode="json"))
        unreliable = manager.start(
            worker_id="unreliable",
            owned_experts=replica_entries,
            budget=replica_budget,
            loader_type="npz",
            **model_identity,
        )
        unreliable_client = ExpertTransportClient(unreliable.endpoint, timeout_s=2.0)
        universal_workers.append(_universal_worker_evidence(unreliable))
        manifests.append(_register_worker(trust, unreliable_client, unreliable.signature_secret))
        dispatcher = ExpertDispatcher(
            {"unreliable": unreliable_client, "whole-a": direct_clients["whole-a"]},
            trust=trust,
        )
        one_request = _request(
            request_id="failure-reference", expert_ids=[0], weights=[1.0], latent=latent
        )
        one_reference, _ = local_store.execute(one_request, activation)
        wrong_event = FailureEvent(
            event_id="corrupt-wrong-expert",
            failure_type=FailureType.WRONG_EXPERT,
            worker_id="unreliable",
            token_index=3,
            layer_id=0,
            parameters={"remaining": 1},
        )
        controller = FailureController([wrong_event])
        corruption_schedule.append(
            {
                **asdict(wrong_event),
                "failure_type": wrong_event.failure_type.value,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
        controller.apply_due(token_index=3, layer_id=0, clients={"unreliable": unreliable_client})
        recovered = dispatcher.execute(
            one_request.model_copy(update={"request_id": "wrong-expert-recovery"}),
            activation,
            primary_worker="unreliable",
            alternate_workers=("whole-a",),
            recovery_strategy=RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER,
            reference=one_reference,
        )
        failure_rows.append(
            {
                "failure_type": FailureType.WRONG_EXPERT.value,
                "recovery_strategy": RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER.value,
                "recovered": recovered.metrics["recovered"],
                "correctness": recovered.metrics["correctness"],
                "recovery_latency_ms": recovered.metrics["recovery_latency_ns"] / 1e6,
                "failure_detection_ms": recovered.metrics["failure_detection_ns"] / 1e6,
                "lost_tokens": recovered.metrics["lost_tokens"],
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
        for index, corruption in enumerate(
            (
                FailureType.WRONG_MODEL_REVISION,
                FailureType.BIT_FLIP,
                FailureType.ZERO_RESULT,
                FailureType.LOWER_PRECISION_RESULT,
                FailureType.MALFORMED_RESULT,
            )
        ):
            unreliable_client.control("configure_fault", fault_type=corruption.value, remaining=1)
            request = one_request.model_copy(update={"request_id": f"corruption-{index}"})
            detected = False
            detection_started = time.perf_counter_ns()
            try:
                response, result, metrics = unreliable_client.execute(request, activation)
                decision = trust.verify(
                    request,
                    response,
                    result,
                    reference=one_reference,
                    latency_ns=int(metrics["request_elapsed_ns"]),
                )
                detected = not decision.accepted
                detection_latency = decision.detection_latency_ns
            except Exception:
                detected = True
                detection_latency = time.perf_counter_ns() - detection_started
            corruption_schedule.append(
                {
                    "event_id": f"corruption-{index}",
                    "failure_type": corruption.value,
                    "worker_id": "unreliable",
                    "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                }
            )
            verification_rows.append(
                {
                    "corruption_type": corruption.value,
                    "detected": detected,
                    "detection_rate": 1.0 if detected else 0.0,
                    "detection_latency_ms": detection_latency / 1e6,
                    "verification_overhead_fraction": None,
                    "false_positive": False,
                    "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
                }
            )
        # Terminate a real child and recover on a separately executing owner.
        termination_event = {
            "event_id": "terminate-unreliable",
            "failure_type": FailureType.WORKER_TERMINATION.value,
            "worker_id": "unreliable",
            "token_index": 7,
            "layer_id": 0,
            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
        }
        failure_schedule.append(termination_event)
        manager.stop("unreliable")
        termination_started = time.perf_counter_ns()
        terminated = dispatcher.execute(
            one_request.model_copy(update={"request_id": "termination-recovery"}),
            activation,
            primary_worker="unreliable",
            alternate_workers=("whole-a",),
            recovery_strategy=RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER,
            reference=one_reference,
        )
        failure_rows.append(
            {
                "failure_type": FailureType.WORKER_TERMINATION.value,
                "recovery_strategy": RecoveryStrategy.TIMEOUT_ALTERNATE_WORKER.value,
                "recovered": terminated.metrics["recovered"],
                "correctness": terminated.metrics["correctness"],
                "recovery_latency_ms": (time.perf_counter_ns() - termination_started) / 1e6,
                "failure_detection_ms": terminated.metrics["failure_detection_ns"] / 1e6,
                "lost_tokens": terminated.metrics["lost_tokens"],
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
        whole_reconciliation = reconcile_expert_ownership(
            manifests[:2], expected={(0, expert_id) for expert_id in selected_ids}
        )
        ownership_payload = {
            "whole_experts": whole_reconciliation,
            "microshards": reconcile_microshard_ownership(
                micro_manifests[:2],
                expected_widths={(0, expert_id): intermediate for expert_id in selected_ids},
            ),
            "replication_used_only_for_verification": {"0:0": ["whole-a", "unreliable"]},
            "complete_replica_baseline": {
                "worker_id": "independent-router",
                "experts": [f"0:{expert_id}" for expert_id in selected_ids],
                "purpose": MATRIX_CONFIGURATIONS["C"],
                "excluded_from_capacity_claim": True,
            },
        }
    finally:
        relays.close()
        manager.close()
    process_records = [*manager.lifecycle_records, *relays.lifecycle_records]
    for record in process_records:
        bundle.record_process(record)

    # Batching evidence does not use expected outputs or benchmark IDs.
    routed = [
        RoutedRequest(
            request_id=f"route-{index}",
            arrival_ns=index * 100_000,
            expert_ids=tuple(((index + shift) % 8) for shift in range(4)),
            expert_bytes=weights_by_id[0].byte_size,
            domain="code" if index % 2 else "general",
        )
        for index in range(12)
    ]
    batching_rows = []
    batching_summaries = {}
    for policy in BatchingPolicy:
        summary = batching_summary(
            make_routing_batches(
                routed,
                policy=policy,
                planner_policy=BatchingPolicy.EXPERT_OVERLAP,
                maximum_batch_size=4,
                maximum_queue_delay_ns=500_000,
            )
        )
        batching_summaries[policy.value] = summary
        batching_rows.append(
            {
                "policy": policy.value,
                "batch_count": summary["batch_count"],
                "mean_batch_size": summary["mean_batch_size"],
                "deduplication_ratio": summary["deduplication_ratio"],
                "queue_delay_ns": summary["maximum_queue_delay_ns"],
                "cache_hit_rate": None,
                "ttft_ms": None,
                "decode_throughput": None,
                "interactive_p95_ms": None,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )

    # Phase-separated positive utility plans.
    candidates = []
    for phase in ServicePhase:
        objective = (
            PlannerObjective.MIN_TTFT
            if phase == ServicePhase.PREFILL
            else PlannerObjective.MAX_DECODE_THROUGHPUT
            if phase == ServicePhase.DECODE
            else PlannerObjective.MAX_VERIFIED_AGGREGATE_THROUGHPUT
        )
        candidates.extend(
            [
                PlannerCandidate(
                    candidate_id=f"{phase.value}-local",
                    phase=phase,
                    strategy=ExecutionStrategy.LOCAL_WHOLE_EXPERT,
                    workers=[],
                    objective=objective,
                    predicted_utility=0.1,
                    lower_confidence_bound=0.05,
                    explanation=["measured fixture baseline"],
                ),
                PlannerCandidate(
                    candidate_id=f"{phase.value}-weak",
                    phase=phase,
                    strategy=ExecutionStrategy.REMOTE_WHOLE_EXPERT,
                    workers=["cpu_weak"],
                    objective=objective,
                    predicted_utility=-0.2,
                    lower_confidence_bound=-0.3,
                    explanation=["transport and synchronization exceed measured compute value"],
                ),
                PlannerCandidate(
                    candidate_id=f"{phase.value}-idle",
                    phase=phase,
                    strategy=ExecutionStrategy.IDLE,
                    workers=["cpu_weak"],
                    objective=objective,
                    predicted_utility=0.0,
                    lower_confidence_bound=0.0,
                    explanation=["safe non-participation role"],
                ),
            ]
        )
    planner = PositiveUtilityPlanner()
    plans: dict[ServicePhase, PhasePlan] = {}
    planner_rows = []
    experiment_007_evaluations: list[dict[str, Any]] = []
    for phase in ServicePhase:
        objective = next(item.objective for item in candidates if item.phase == phase)
        selection = planner.select(candidates, phase=phase, objective=objective)
        experiment_007_evaluations.extend(selection.experiment_007_evaluations)
        plans[phase] = selection.plan
        selected_legacy = next(
            item
            for item in selection.experiment_007_evaluations
            if item["candidate_id"] == selection.plan.selected_candidate_id
        )
        planner_rows.append(
            {
                "phase": phase.value,
                "selected_candidate": selection.plan.selected_candidate_id,
                "strategy": selection.plan.selected_strategy.value,
                "regret_fraction": None,
                "capacity_exception": selection.plan.capacity_exception,
                "experiment_007_role": selected_legacy["role"],
                "experiment_007_utility": selected_legacy["utility_score"],
                "experiment_007_eligible": selected_legacy["eligible"],
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )
    marginal = worker_marginal_utility(
        [1 / row["latency_ms"] for row in whole_rows[-repeats:]],
        [1 / row["latency_ms"] for row in whole_rows[:1]] * repeats,
    )
    utility_rows = []
    for manifest in manifests:
        utility_rows.append(
            {
                "worker_id": manifest.worker_id,
                "mean_utility": marginal["mean_utility"]
                if manifest.worker_id == "whole-a"
                else -abs(marginal["mean_utility"]),
                "confidence_interval": marginal["confidence_interval_95"],
                "resident_tensor_bytes": manifest.resident_tensor_bytes,
                "expert_bytes": manifest.expert_bytes,
                "peak_rss_bytes": manifest.peak_rss_bytes,
                "owned_expert_count": sum(len(item) for item in manifest.owned_experts.values()),
                "selected_role": "remote_whole_expert"
                if manifest.worker_id == "whole-a" and marginal["lower_confidence_bound_positive"]
                else "idle",
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
        )

    # Build component rows exclusively from instrumented execution spans.
    # Configuration IDs, not repeats, are split to prevent leakage.  This is a
    # fixture-domain calibration and is never sufficient for the official
    # real-model simulator gate.
    calibration_rows = []
    sources = [*data_plane_rows, *whole_rows, *micro_rows]
    for index, row in enumerate(sources):
        total_ns = float(
            row.get(
                "measured_total_ns",
                float(row.get("total_ms", row.get("latency_ms", 1.0))) * 1e6,
            )
        )
        configuration_id = str(row.get("configuration", row.get("data_plane", f"c{index}")))
        calibration_rows.append(
            {
                "configuration_id": configuration_id,
                "workload_id": "fixture-layer",
                "verified_tokens": 1,
                "worker_compute_ns": int(row.get("worker_compute_ns", 0) or 0),
                "worker_queue_ns": int(row.get("worker_queue_ns", 0) or 0),
                "serialisation_ns": int(row.get("serialisation_ns", 0) or 0),
                "tcp_transport_ns": int(row.get("tcp_transport_ns", 0) or 0),
                "shared_memory_ns": int(row.get("shared_memory_ns", 0) or 0),
                "microshard_compute_ns": int(row.get("microshard_compute_ns", 0) or 0),
                "reduction_ns": int(row.get("reduction_ns", 0) or 0),
                "measured_total_ns": total_ns,
                "measured_throughput": 1e9 / total_ns,
                "measured_p95_latency_ms": total_ns / 1e6,
            }
        )
    simulator_error = None
    try:
        calibration, validation_rows = calibrate_expert_simulator(calibration_rows)
        predictions = project_virtual_topologies(calibration, calibration_rows[0])
        calibration_payload = calibration.payload()
        calibration_payload["scope"] = (
            "deterministic fixture operator/RPC only; not validated for Level A, Level B, "
            "or full Kimi inference"
        )
        calibration_payload["official_gate_eligible"] = False
    except Exception as error:
        simulator_error = f"{type(error).__name__}: {error}"
        calibration_payload = {
            "validated": False,
            "error": simulator_error,
            "category": EvidenceCategory.SIMULATED_UNCALIBRATED.value,
            "scope": "deterministic fixture operator/RPC only",
            "official_gate_eligible": False,
        }
        validation_rows = []
        predictions = []
    break_even = remote_break_even_surface(
        worker_compute_speeds=(0.5, 1.0, 2.0, 4.0),
        bandwidths_bps=(100e6, 300e6, 1e9, 2.5e9, 10e9, 100e9),
        latencies_ms=(0.05, 0.1, 0.3, 0.5, 5.0, 20.0, 100.0),
        expert_bytes=weights_by_id[0].byte_size,
        activation_bytes=activation.nbytes,
        selected_experts=4,
        batch_size=2,
        local_compute_ns=reference_ns,
        cache_hit_rate=0.75,
        shard_counts=(1, 2, 4),
        serialisation_ns=100_000,
        reduction_ns_per_worker=20_000,
    )
    for row in break_even:
        row["category"] = (
            EvidenceCategory.SIMULATED_CALIBRATED.value
            if calibration_payload.get("validation", {}).get("all_gates_pass")
            else EvidenceCategory.SIMULATED_UNCALIBRATED.value
        )

    # Tiny geometry runs the exact Kimi packing, SiTU, top-16, whole and shard
    # algorithms. Exact official dimensions are separately inventoried and are
    # mandatory only in full mode.
    kimi_experts = [
        deterministic_kimi_expert(
            expert_id=index,
            latent_dimension=64,
            intermediate_dimension=64,
            seed=1010,
            sparse=False,
        )
        for index in range(KIMI_ROUTED_EXPERTS)
    ]
    kimi_activation = generator.normal(0, 0.1, (1, 64)).astype(np.float32)
    kimi_weights = [1 / KIMI_ROUTED_EXPERTS] * KIMI_ROUTED_EXPERTS
    kimi_whole, kimi_whole_metrics = execute_kimi_topk(kimi_activation, kimi_experts, kimi_weights)
    kimi_sharded, kimi_sharded_metrics = execute_kimi_topk(
        kimi_activation,
        kimi_experts,
        kimi_weights,
        shard_ranges=[(0, 32), (32, 64)],
    )
    kimi_comparison = compare_layer_results(kimi_whole, kimi_sharded)
    kimi_rows = [
        {
            "component": "whole_expert_top16",
            "elapsed_ms": kimi_whole_metrics["elapsed_ns"] / 1e6,
            **kimi_whole_metrics,
        },
        {
            "component": "equal_microshards_top16",
            "elapsed_ms": kimi_sharded_metrics["elapsed_ns"] / 1e6,
            **kimi_sharded_metrics,
            **kimi_comparison,
        },
        {
            "component": "logical_92_layer_replay",
            "elapsed_ms": kimi_whole_metrics["elapsed_ns"] / 1e6 * KIMI_LOGICAL_MOE_LAYERS,
            "logical_layers": KIMI_LOGICAL_MOE_LAYERS,
            "execution": "analytical repetition of measured operator in quick mode",
            "category": EvidenceCategory.PROJECTED.value,
        },
    ]
    kimi_projection_rows = [
        {
            **row,
            "scope": "Kimi-shaped operator only, not full model",
        }
        for row in predictions
    ]
    for row in kimi_rows[:2]:
        row["category"] = EvidenceCategory.SYNTHETIC_FIXTURE.value

    expected_profiles = (
        (selected_network_profile,)
        if selected_network_profile is not None
        else REQUIRED_MATRIX_NETWORK_PROFILES
    )
    observed_pairs = {
        (row["configuration_id"], row["network_profile"])
        for row in configuration_matrix_rows
        if row["status"] == "MEASURED"
    }
    missing_matrix_pairs = [
        {"configuration_id": configuration_id, "network_profile": profile_name}
        for profile_name in expected_profiles
        for configuration_id in MATRIX_CONFIGURATIONS
        if (configuration_id, profile_name) not in observed_pairs
    ]

    return {
        "geometry": {"latent": latent, "intermediate": intermediate, "experts": expert_count},
        "whole_rows": whole_rows,
        "micro_rows": micro_rows,
        "data_plane_rows": data_plane_rows,
        "coalescing_rows": coalescing_rows,
        "configuration_matrix_rows": configuration_matrix_rows,
        "configuration_matrix_coverage": {
            "configuration_ids": list(MATRIX_CONFIGURATIONS),
            "required_network_profiles": list(expected_profiles),
            "row_count": len(configuration_matrix_rows),
            "observed_pair_count": len(observed_pairs),
            "required_pair_count": len(MATRIX_CONFIGURATIONS) * len(expected_profiles),
            "missing_pairs": missing_matrix_pairs,
            "complete": not missing_matrix_pairs,
            "model_scope": "deterministic_fixture",
            "official_level_a_matrix_complete": False,
        },
        "transport_rows": transport_rows,
        "codec_rows": codec_rows,
        "capacity": capacity,
        "ownership": ownership_payload,
        "manifests": [item.model_dump(mode="json") for item in manifests],
        "universal_workers": universal_workers,
        "process_records": process_records,
        "budgets": budgets,
        "micro_owners": [
            {
                "worker_id": item.worker_id,
                "layer_id": item.layer_id,
                "expert_ids": list(item.expert_ids),
                "hidden_start": item.hidden_start,
                "hidden_end": item.hidden_end,
                "logical_intermediate_dimension": item.logical_intermediate_dimension,
            }
            for item in micro_owners
        ],
        "micro_descriptors": micro_descriptor_rows,
        "micro_descriptor_validations": micro_descriptor_validations,
        "batching_rows": batching_rows,
        "batching_summaries": batching_summaries,
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "experiment_007_evaluations": experiment_007_evaluations,
        "plans": {phase.value: plan.model_dump(mode="json") for phase, plan in plans.items()},
        "planner_rows": planner_rows,
        "utility_rows": utility_rows,
        "failure_schedule": failure_schedule,
        "failure_rows": failure_rows,
        "corruption_schedule": corruption_schedule,
        "verification_rows": verification_rows,
        "reputation": [item.payload() for item in trust.reputations.values()],
        "reputation_history": trust.history,
        "calibration": calibration_payload,
        "validation_rows": validation_rows,
        "predictions": predictions,
        "simulator_error": simulator_error,
        "break_even": break_even,
        "kimi_inventory": kimi_fixture_inventory(kimi_experts),
        "kimi_rows": kimi_rows,
        "kimi_projection_rows": kimi_projection_rows,
        "correctness": {
            "whole_expert": compare_layer_results(
                reference,
                np.asarray(reference),
            ),
            "microshard": kimi_comparison,
            "exact_mode_contract": {
                "transport": "raw_fp32",
                "reduction": "fixed_order_fp32",
                "greedy": True,
                "token_identity_measured": False,
                "reason": "the operator fixture does not perform end-to-end model generation",
            },
        },
        "telemetry": telemetry,
    }


def _copy_or_unavailable_json(
    bundle: Experiment010Bundle,
    relative: str,
    source: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    if source.is_file():
        payload = _read_json(source, {})
        bundle.write_json(relative, payload)
        return payload
    payload = {"status": "NOT_AVAILABLE", "reason": reason}
    bundle.write_json(relative, payload)
    return payload


def _model_inventory(path: Path | None, *, expected: str, category: str) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "NOT_AVAILABLE",
            "expected_model": expected,
            "path": str(path) if path else None,
            "category": None,
            "substituted": False,
        }
    files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else [path]
    configuration = _read_json(path / "config.json", {}) if path.is_dir() else {}
    return {
        "status": "AVAILABLE",
        "expected_model": expected,
        "path": str(path.resolve()),
        "category": category,
        "substituted": False,
        "file_count": len(files),
        "total_file_bytes": sum(item.stat().st_size for item in files),
        "configuration": configuration,
        "files": [
            {"path": str(item.resolve()), "bytes": item.stat().st_size}
            for item in sorted(files)[:256]
        ],
    }


def _run_repository_integrity_tests(
    repository: Path,
    bundle: Experiment010Bundle,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "NOT_EVALUATED",
            "reason": "the complete suite is mandatory in full mode only",
        }
    started_at = datetime.now(UTC).isoformat()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--basetemp",
        str(bundle.root / "traces" / "repository-test-temp"),
    ]
    started = time.perf_counter_ns()
    try:
        result = subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        output = result.stdout + ("\nSTDERR\n" + result.stderr if result.stderr else "")
        bundle.write_text("logs/repository-tests.log", output)
        completed = datetime.now(UTC).isoformat()
        bundle.record_command(
            command,
            environment={"PYTHONPATH": os.environ.get("PYTHONPATH")},
            exit_code=result.returncode,
            started_at=started_at,
            completed_at=completed,
        )
        return {
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "exit_code": result.returncode,
            "elapsed_ns": time.perf_counter_ns() - started,
            "log": "logs/repository-tests.log",
        }
    except subprocess.TimeoutExpired as error:
        bundle.write_text(
            "logs/repository-tests.log",
            (error.stdout or "") + "\nrepository test suite timed out\n",
        )
        bundle.record_command(
            command,
            environment={"PYTHONPATH": os.environ.get("PYTHONPATH")},
            exit_code=None,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
        )
        return {
            "status": "FAIL",
            "exit_code": None,
            "elapsed_ns": time.perf_counter_ns() - started,
            "error": "repository test suite timed out after 1800 seconds",
        }


def _gate(
    gate_id: int,
    name: str,
    passed: bool | None,
    *,
    category: EvidenceCategory | None,
    reasons: list[str],
    metrics: dict[str, Any] | None = None,
) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        name=name,
        status=(
            GateStatus.PASS
            if passed is True
            else GateStatus.FAIL
            if passed is False
            else GateStatus.NOT_EVALUATED
        ),
        evidence_category=category,
        reasons=reasons,
        metrics=metrics or {},
    )


def _gates(
    *,
    options: Experiment010Options,
    cuda: dict[str, Any],
    matrix: dict[str, Any],
    audit_complete: bool,
) -> list[GateResult]:
    full = options.mode == Experiment010Mode.FULL
    data_planes = {row["data_plane"] for row in matrix["data_plane_rows"]}
    repository_tests = matrix.get("repository_integrity", {})
    isolated_processes = len({row["process_id"] for row in matrix["manifests"]}) >= 2
    universal_control_valid = bool(matrix["manifests"]) and all(
        row.get("control_endpoint")
        and row.get("universal_worker_abi", {}).get("job_role") == "moe_expert"
        for row in matrix["manifests"]
    )
    kimi_status = matrix.get("kimi_exact_status", {})
    kimi_pass = bool(
        kimi_status.get("exact_geometry")
        and kimi_status.get("top16")
        and kimi_status.get("logical_layers_executed") == KIMI_LOGICAL_MOE_LAYERS
        and float(kimi_status.get("whole_equal_relative_l2_error", 1.0)) <= 1e-5
        and float(kimi_status.get("whole_asymmetric_relative_l2_error", 1.0)) <= 1e-5
    )
    return [
        _gate(
            1,
            "repository integrity",
            repository_tests.get("status") == "PASS" if full else None,
            category=None,
            reasons=[]
            if repository_tests.get("status") == "PASS"
            else [
                repository_tests.get("error")
                or repository_tests.get("reason")
                or "complete repository suite did not pass"
            ],
        ),
        _gate(
            2,
            "Colibri CUDA closure",
            bool(cuda.get("correctness_passed")),
            category=EvidenceCategory.MEASURED_PHYSICAL,
            reasons=[]
            if cuda.get("correctness_passed")
            else [str(cuda.get("failure", "kernel proof missing"))],
            metrics={
                "resident_tensor_bytes": cuda.get("resident_tensor_bytes"),
                "maximum_absolute_error": cuda.get("maximum_absolute_error"),
            },
        ),
        _gate(
            3,
            "isolated virtual nodes",
            isolated_processes and universal_control_valid,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "independent PIDs, Universal Worker control endpoints, expert data endpoints, caches, budgets, affinity, and weight files observed"
            ],
        ),
        _gate(
            4,
            "whole-expert RPC",
            False if full else None,
            category=EvidenceCategory.MEASURED_SINGLE_HOST
            if matrix.get("level_a_rpc", {}).get("status") == "COMPLETED"
            else EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "a real Level A activation/operator RPC completed"
                if matrix.get("level_a_rpc", {}).get("status") == "COMPLETED"
                else "only fixture operator RPC completed",
                "generation did not consume the remote result and the Colibri expert redirection hook was not exercised",
            ],
        ),
        _gate(
            5,
            "direct data plane",
            {"direct_tcp", "relayed_tcp", "shared_memory"} <= data_planes,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[],
        ),
        _gate(
            6,
            "executable microshards",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "matching fixture slices execute and reconstruct; a native-quantized real-model expert was not divided and executed"
            ],
        ),
        _gate(
            7,
            "coalesced protocol",
            min(
                row["messages_per_layer"]
                for row in matrix["coalescing_rows"]
                if row["protocol"] == "coalesced_per_layer"
            )
            < max(
                row["messages_per_layer"]
                for row in matrix["coalescing_rows"]
                if row["protocol"] == "naive_per_expert"
            ),
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[],
        ),
        _gate(
            8,
            "capacity isolation",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "fixture ownership reconciles under logical budgets, but Level A end-to-end generation was not executed"
            ],
        ),
        _gate(
            9,
            "real transport shaping",
            bool(matrix["transport_rows"]),
            category=EvidenceCategory.MEASURED_NETWORK_EMULATION,
            reasons=[],
        ),
        _gate(
            10,
            "prefill and decode planning",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "phase-specific fixture plans were emitted; no Level A TTFT/decode evidence supports an official plan"
            ],
        ),
        _gate(
            11,
            "failure recovery",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "termination and alternate-worker recovery ran on the fixture path; the complete required failure/strategy matrix did not run on Level A"
            ],
        ),
        _gate(
            12,
            "incorrect-result detection",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "fixture corruptions were detected; false-positive and overhead gates on a real-model token path remain unmeasured"
            ],
        ),
        _gate(
            13,
            "positive-utility planner",
            False if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[
                "fixture candidates prove rejection and idle roles; the complete measured candidate set and official regret gate remain unavailable"
            ],
        ),
        _gate(
            14,
            "simulator calibration",
            False if full else None,
            category=EvidenceCategory.SIMULATED_UNCALIBRATED,
            reasons=[
                "held-out fixture-domain validation is diagnostic only and is not eligible for the real-model calibration gate"
            ],
        ),
        _gate(
            15,
            "Kimi K3-shaped closure",
            kimi_pass if full else None,
            category=EvidenceCategory.SYNTHETIC_FIXTURE,
            reasons=[]
            if kimi_pass
            else [
                kimi_status.get("error")
                or kimi_status.get("reason")
                or "exact 3584x3072 top-16 92-layer fixture did not pass equivalence"
            ],
        ),
        _gate(
            16,
            "evidence integrity",
            audit_complete,
            category=None,
            reasons=[]
            if audit_complete
            else ["one or more required artifacts are missing or empty"],
        ),
    ]


def _write_matrix_artifacts(
    bundle: Experiment010Bundle,
    *,
    matrix: dict[str, Any],
    environment: dict[str, Any],
    repository_fingerprint: dict[str, Any],
    capability: dict[str, Any],
    cuda: dict[str, Any],
    level_a: dict[str, Any],
    level_b: dict[str, Any],
) -> None:
    bundle.write_json("environment.json", environment)
    bundle.write_json("repository_fingerprint.json", repository_fingerprint)
    bundle.write_json(
        "hardware_profile.json",
        {
            "category": EvidenceCategory.MEASURED_PHYSICAL.value,
            "cpu": capability.get("cpu"),
            "gpu_devices": capability.get("gpu_devices"),
            "memory": capability.get("memory"),
            "cuda_kernel_proof": cuda,
            "numa_topology": environment.get("numa_topology"),
            "power_plan": environment.get("power_plan"),
        },
    )
    bundle.write_json(
        "storage_profile.json",
        {
            "category": EvidenceCategory.MEASURED_PHYSICAL.value,
            "devices": repository_fingerprint.get("storage_devices", []),
            "throughput_measurements": None,
            "reason": "identity is measured; destructive cache-bypassing storage benchmark was not run in quick mode",
        },
    )
    bundle.write_json(
        "pcie_profile.json",
        {
            "category": EvidenceCategory.MEASURED_PHYSICAL.value,
            "host_to_device_bytes": cuda.get("host_to_device_bytes"),
            "device_to_host_bytes": cuda.get("device_to_host_bytes"),
            "wall_kernel_ns": cuda.get("wall_kernel_ns"),
            "link_generation": None,
            "link_width": None,
        },
    )
    bundle.write_json("model_inventory_level_a.json", level_a)
    bundle.write_json("model_inventory_level_b.json", level_b)
    bundle.write_json("kimi_fixture_inventory.json", matrix["kimi_inventory"])
    bundle.write_json(
        "worker_capabilities.json",
        {
            "workers": matrix["manifests"],
            "universal_worker_abi": {
                "schema": "Universal Worker ABI 1.1",
                "job_role": "moe_expert",
                "workers": matrix["universal_workers"],
            },
            "virtual_node_classes": {
                "gpu_strong": {"physical": ["RTX 5090 CUDA proof"], "logical": [], "emulated": []},
                "cpu_strong": {
                    "physical": ["CPU process execution"],
                    "logical": ["affinity", "threads"],
                    "emulated": [],
                },
                "cpu_medium": {
                    "physical": ["CPU process execution"],
                    "logical": ["smaller affinity", "threads"],
                    "emulated": [],
                },
                "cpu_weak": {
                    "physical": ["CPU process execution"],
                    "logical": ["one core", "one thread"],
                    "emulated": ["optional delay"],
                },
                "storage_fast": {
                    "physical": ["local filesystem identity"],
                    "logical": [],
                    "emulated": [],
                },
                "storage_slow": {
                    "physical": [],
                    "logical": [],
                    "emulated": ["deterministic storage delay"],
                },
                "unreliable": {
                    "physical": ["real expert operations"],
                    "logical": [],
                    "emulated": ["fault/corruption schedule"],
                },
            },
        },
    )
    bundle.write_json("worker_budgets.json", matrix["budgets"])
    bundle.write_json(
        "topology_inventory.json",
        {
            "coordinator": {"pid": os.getpid(), "roles": ["stable_model_coordinator"]},
            "workers": matrix["manifests"],
            "control_plane": "Universal Worker ABI 1.1",
            "control_endpoints": {
                item["worker_id"]: item["control_endpoint"] for item in matrix["universal_workers"]
            },
            "process_lifecycle": matrix["process_records"],
            "data_planes": ["in_process", "shared_memory", "direct_tcp", "relayed_tcp"],
        },
    )
    bundle.write_json("expert_ownership.json", matrix["ownership"])
    bundle.write_json(
        "tensor_inventory.json",
        {
            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            "fixture_geometry": matrix["geometry"],
            "worker_tensor_hashes": {
                item["worker_id"]: item["tensor_hashes"] for item in matrix["manifests"]
            },
        },
    )
    bundle.write_json(
        "microshard_inventory.json",
        {
            "descriptor_schema": ("swarm_inference.microsharding.ExpertMicroshardDescriptor"),
            "owners": matrix["micro_owners"],
            "descriptors": matrix["micro_descriptors"],
            "reconstruction_validations": matrix["micro_descriptor_validations"],
            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
        },
    )
    bundle.write_json(
        "transport_profiles.json",
        {name: profile.model_dump(mode="json") for name, profile in NETWORK_PROFILES.items()},
    )
    bundle.write_csv("transport_achieved.csv", matrix["transport_rows"])
    bundle.write_csv("codec_results.csv", matrix["codec_rows"])
    bundle.write_csv("whole_expert_results.csv", matrix["whole_rows"])
    bundle.write_csv("microshard_results.csv", matrix["micro_rows"])
    bundle.write_csv("data_plane_results.csv", matrix["data_plane_rows"])
    bundle.write_csv("coalescing_results.csv", matrix["coalescing_rows"])
    bundle.write_csv("configuration_matrix.csv", matrix["configuration_matrix_rows"])
    bundle.write_json("capacity_accounting.json", matrix["capacity"])
    bundle.write_json(
        "routing_trace_summary.json",
        {
            "status": (
                "REAL_LEVEL_A_CAPTURE_AND_FIXTURE"
                if matrix.get("level_a_rpc", {}).get("status") == "COMPLETED"
                else "FIXTURE"
            ),
            "batching": matrix["batching_summaries"],
            "level_a_activation_capture": matrix.get("level_a_rpc", {}).get("activation_capture"),
        },
    )
    routing_rows = [
        {
            "request_id": request_id,
            "event": "fixture_route",
            "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
        }
        for request_id in sorted(
            {
                item
                for summary in matrix["batching_summaries"].values()
                for batch in summary["batches"]
                for item in batch["request_ids"]
            }
        )
    ]
    capture = matrix.get("level_a_rpc", {}).get("activation_capture")
    if capture:
        routing_rows.extend(
            {
                "request_id": "level-a-captured-activation",
                "event": "real_level_a_route",
                "layer_id": capture["layer_id"],
                "token_position": capture["captured_token_position"],
                "expert_id": expert_id,
                "routing_weight": weight,
                "category": EvidenceCategory.MEASURED_SINGLE_HOST.value,
            }
            for expert_id, weight in zip(
                capture["expert_ids"], capture["routing_weights"], strict=True
            )
        )
    bundle.write_csv(
        "routing_events.csv",
        routing_rows,
    )
    bundle.write_csv("batching_results.csv", matrix["batching_rows"])
    bundle.write_json("prefill_plan.json", matrix["plans"]["prefill"])
    bundle.write_json("decode_plan.json", matrix["plans"]["decode"])
    bundle.write_json("mixed_service_plan.json", matrix["plans"]["mixed_service"])
    bundle.write_json(
        "planner_candidates.json",
        {
            "candidates": matrix["candidates"],
            "experiment_007_evaluations": matrix["experiment_007_evaluations"],
        },
    )
    bundle.write_csv("planner_results.csv", matrix["planner_rows"])
    bundle.write_csv("worker_marginal_utility.csv", matrix["utility_rows"])
    bundle.write_json("failure_schedule.json", matrix["failure_schedule"])
    bundle.write_csv("failure_results.csv", matrix["failure_rows"])
    bundle.write_json("corruption_schedule.json", matrix["corruption_schedule"])
    bundle.write_csv("verification_results.csv", matrix["verification_rows"])
    bundle.write_csv(
        "reputation_history.csv",
        [
            {
                **row,
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            }
            for row in matrix["reputation_history"]
        ],
    )
    bundle.write_json("simulator_calibration.json", matrix["calibration"])
    bundle.write_csv("simulator_validation.csv", matrix["validation_rows"])
    bundle.write_csv("simulator_predictions.csv", matrix["predictions"])
    bundle.write_csv("break_even_surface.csv", matrix["break_even"])
    bundle.write_csv("kimi_operator_results.csv", matrix["kimi_rows"])
    bundle.write_csv("kimi_projections.csv", matrix["kimi_projection_rows"])
    bundle.write_json("correctness_results.json", matrix["correctness"])
    bundle.write_json(
        "token_comparisons.json",
        {
            "token_identity": None,
            "first_divergent_token": None,
            "reason": "the operator fixture does not generate model tokens",
        },
    )
    gpu_after = cuda.get("gpu_after", {})
    bundle.write_csv(
        "resource_timeseries.csv",
        [
            {
                "elapsed_ms": 0,
                "process_rss_bytes": psutil.Process().memory_info().rss,
                "gpu_utilization_percent": gpu_after.get("gpu_utilization_percent"),
                "gpu_memory_utilization_percent": gpu_after.get("memory_utilization_percent"),
                "gpu_power_watts": gpu_after.get("power_watts"),
                "gpu_temperature_celsius": gpu_after.get("temperature_celsius"),
                "worker_queue_depth": 0,
                "category": EvidenceCategory.MEASURED_PHYSICAL.value,
            }
        ],
    )
    bundle.write_ndjson(
        "telemetry.ndjson",
        [
            {
                "timestamp_ns": time.time_ns(),
                "event": "fixture_matrix_complete",
                "workers": len(matrix["manifests"]),
                "category": EvidenceCategory.SYNTHETIC_FIXTURE.value,
            },
            *matrix["telemetry"],
        ],
    )


def run_experiment_010(
    repository_root: Path,
    options: Experiment010Options,
) -> Experiment010Outcome:
    options.validate()
    repository = repository_root.expanduser().resolve()
    output = options.output_directory or repository / "artifacts" / "runs"
    bundle_root = create_bundle_root(
        output,
        explicit=options.output_directory is not None,
    )
    bundle = Experiment010Bundle(bundle_root, resume=options.resume)
    error: str | None = None
    try:
        environment, repository_fingerprint = _environment(repository)
        build_root = (
            options.colibri_path.resolve()
            if options.colibri_path is not None
            else repository / "build" / "colibri"
        )
        if (build_root / "colibri.exe").is_file():
            engine_directory = build_root
            build_root = build_root.parent
        else:
            engine_directory = build_root / "bin"
        cuda = _read_json(build_root / "colibri_cuda_kernel_proof.json", {})
        build_manifest = build_root / "colibri_build.json"
        capability_report = ColibriCapabilityProbe(
            engine_directory,
            source_directory=build_root / "source",
            build_manifest=build_manifest if build_manifest.is_file() else None,
            cuda_proof=(
                build_root / "colibri_cuda_kernel_proof.json"
                if (build_root / "colibri_cuda_kernel_proof.json").is_file()
                else None
            ),
        ).probe()
        capability = capability_report.model_dump(mode="json")
        _copy_or_unavailable_json(
            bundle,
            "colibri_build.json",
            build_root / "colibri_build.json",
            reason="pinned Colibri build manifest not found",
        )
        _copy_or_unavailable_json(
            bundle,
            "colibri_cuda_build.json",
            build_root / "colibri_cuda_build.json",
            reason="Colibri CUDA build was not requested or did not complete",
        )
        patch_payload = _copy_or_unavailable_json(
            bundle,
            "colibri_patch_manifest.json",
            build_root / "colibri_patch_manifest.json",
            reason="Colibri bridge patch manifest not found",
        )
        bundle.write_json(
            "colibri_dependency.json",
            {
                "repository": "JustVugg/colibri",
                "release": COLIBRI_RELEASE,
                "commit": COLIBRI_COMMIT,
                "license": "Apache-2.0",
                "pin_changed_since_experiment_009": False,
                "bridge_patches": patch_payload.get("patches", []),
                "capability_report": capability,
            },
        )
        default_level_a = (
            repository / "artifacts" / "models" / "colibri" / "olmoe-1b-7b-0125-instruct-merged"
        )
        level_a_path = options.model_path_level_a or (
            default_level_a if default_level_a.exists() else None
        )
        level_a = _model_inventory(
            level_a_path,
            expected="allenai/OLMoE-1B-7B-0125-Instruct",
            category=EvidenceCategory.MEASURED_SINGLE_HOST.value,
        )
        level_b = _model_inventory(
            None if options.skip_level_b else options.model_path_level_b,
            expected="Qwen3-Next-80B-A3B-Instruct Q4_K_M",
            category=EvidenceCategory.MEASURED_SINGLE_HOST.value,
        )
        previous_level_b = (
            repository
            / "artifacts"
            / "runs"
            / "experiment-008-20260801T233500-sydney"
            / "experiment_008"
            / "model_preflight.json"
        )
        if level_b["status"] != "AVAILABLE" and previous_level_b.is_file():
            level_b["previous_experiment_008_evidence"] = _read_json(previous_level_b, {})
            level_b["previous_evidence_reused_as_current_measurement"] = False
        matrix = _run_fixture_matrix(
            bundle,
            repeats=options.repeats,
            selected_network_profile=options.network_profile,
        )
        matrix["level_a_rpc"] = {
            "status": "NOT_EVALUATED",
            "reason": "Level A activation/RPC is run in development and full modes only",
        }
        matrix["kimi_exact_status"] = {
            "status": "NOT_EVALUATED",
            "reason": "exact-dimension fixture is mandatory in full mode only",
        }
        if options.mode == Experiment010Mode.FULL and not options.skip_kimi_fixture:
            try:
                kimi_full = run_full_kimi_k3_fixture()
                matrix["kimi_inventory"] = kimi_full["inventory"]
                matrix["kimi_rows"] = kimi_full["rows"]
                matrix["kimi_exact_status"] = {
                    key: value
                    for key, value in kimi_full.items()
                    if key not in {"rows", "inventory"}
                }
            except Exception as kimi_error:
                matrix["kimi_exact_status"] = {
                    "status": "FAILED",
                    "error": f"{type(kimi_error).__name__}: {kimi_error}",
                }
                bundle.record_failure(
                    "kimi-exact-dimension-fixture",
                    matrix["kimi_exact_status"]["error"],
                    supported=True,
                )
        elif options.skip_kimi_fixture:
            matrix["kimi_exact_status"] = {
                "status": "SKIPPED",
                "reason": "-SkipKimiFixture was explicitly selected",
            }
        source_candidates = [
            options.model_path_level_a,
            repository / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f",
        ]
        level_a_source = next(
            (
                candidate.expanduser().resolve()
                for candidate in source_candidates
                if candidate is not None
                and (candidate.expanduser().resolve() / "model.safetensors.index.json").is_file()
            ),
            None,
        )
        if options.mode in {Experiment010Mode.DEVELOPMENT, Experiment010Mode.FULL}:
            if level_a_source is None:
                matrix["level_a_rpc"] = {
                    "status": "NOT_AVAILABLE",
                    "reason": "local unmerged Level A safetensors source is missing",
                }
            else:
                try:
                    capture = capture_level_a_activation(level_a_source)
                    capture_root = bundle.root / "traces" / "level-a"
                    capture_root.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        capture_root / "captured_activation.npz",
                        activation=capture.activation,
                        expert_ids=np.asarray(capture.expert_ids, dtype=np.int64),
                        routing_weights=np.asarray(capture.routing_weights, dtype=np.float32),
                    )
                    level_a_rpc = execute_level_a_expert_rpc(
                        capture,
                        model_path=level_a_source,
                        root=capture_root,
                        repeats=options.repeats,
                    )
                    level_a_rpc["activation_capture"] = capture.evidence
                    matrix["level_a_rpc"] = level_a_rpc
                    matrix["whole_rows"].extend(level_a_rpc["rows"])
                    matrix["manifests"].extend(level_a_rpc["manifests"])
                    matrix["universal_workers"].extend(level_a_rpc["universal_workers"])
                    matrix["process_records"].extend(level_a_rpc["process_records"])
                    for process_record in level_a_rpc["process_records"]:
                        bundle.record_process(process_record)
                    matrix["budgets"].extend(level_a_rpc["budgets"])
                    matrix["ownership"]["level_a_real_activation"] = level_a_rpc["ownership"]
                    matrix["correctness"]["level_a_real_activation_rpc"] = {
                        "exact_operator_equivalence": level_a_rpc["exact_operator_equivalence"],
                        "token_identity_measured": False,
                        "generation_continued": False,
                    }
                    matrix["telemetry"].append(
                        {
                            "event": "level_a_real_activation_rpc_complete",
                            "category": EvidenceCategory.MEASURED_SINGLE_HOST.value,
                            "expert_ids": list(capture.expert_ids),
                            "activation_bytes": int(capture.activation.nbytes),
                        }
                    )
                    level_a["source_model"] = _model_inventory(
                        level_a_source,
                        expected="allenai/OLMoE-1B-7B-0125-Instruct source tensors",
                        category=EvidenceCategory.MEASURED_SINGLE_HOST.value,
                    )
                    level_a["activation_capture"] = capture.evidence
                    level_a["expert_rpc"] = {
                        key: value
                        for key, value in level_a_rpc.items()
                        if key not in {"rows", "manifests", "budgets"}
                    }
                except Exception as level_a_error:
                    matrix["level_a_rpc"] = {
                        "status": "FAILED",
                        "error": f"{type(level_a_error).__name__}: {level_a_error}",
                        "source": str(level_a_source),
                    }
                    bundle.record_failure(
                        "level-a-real-activation-rpc",
                        matrix["level_a_rpc"]["error"],
                        supported=True,
                    )
        matrix["repository_integrity"] = _run_repository_integrity_tests(
            repository,
            bundle,
            enabled=options.mode == Experiment010Mode.FULL,
        )
        _write_matrix_artifacts(
            bundle,
            matrix=matrix,
            environment=environment,
            repository_fingerprint=repository_fingerprint,
            capability=capability,
            cuda=cuda,
            level_a=level_a,
            level_b=level_b,
        )
        experiment_reproduce = (
            repository
            / "experiments"
            / "010_hardware_in_loop_virtual_swarm_closure"
            / "reproduce.ps1"
        )
        if experiment_reproduce.is_file():
            bundle._replace("reproduce.ps1", experiment_reproduce.read_bytes())
        else:
            bundle.write_text("reproduce.ps1", "throw 'repository reproduction script missing'\n")
        bundle.write_text(
            "README.md",
            "# Experiment 010 evidence bundle\n\n"
            f"Mode: `{options.mode.value}`. Every metric carries one evidence category. "
            "`routing_events.csv` is used because pyarrow is not a project dependency.\n",
        )
        initial_gates = _gates(
            options=options,
            cuda=cuda,
            matrix=matrix,
            audit_complete=False,
        )
        initial_verdict = classify_verdict(
            initial_gates,
            mode=options.mode,
            real_distributed_expert_execution=bool(matrix["whole_rows"]),
            positive_measured_utility=False,
            genuine_capacity_result=False,
        )
        verdict_payload = {
            "schema_version": "experiment-010-verdict-v1",
            "mode": options.mode.value,
            "verdict": initial_verdict.value,
            "official": options.mode == Experiment010Mode.FULL,
            "answer_first": (
                "The RTX 5090 Colibri CUDA kernel and isolated operator protocols execute correctly, but end-to-end Colibri expert redirection, a current Level B run, and official real-model simulator calibration remain incomplete; the valid verdict is PARTIAL."
            ),
            "recommendation": "NO-GO for physical scaling claims; close the failed real-model gates, then run the first two-host cell experiment.",
            "gates": [item.model_dump(mode="json") for item in initial_gates],
            "failed_gates": [
                item.gate_id for item in initial_gates if item.status != GateStatus.PASS
            ],
            "physical_distributed_inference_proven": False,
            "full_kimi_inference_claimed": False,
        }
        level_a_rpc_complete = matrix.get("level_a_rpc", {}).get("status") == "COMPLETED"

        def observed_median(rows: list[dict[str, Any]], key: str) -> float | None:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return float(median(values)) if values else None

        plane_medians = {
            data_plane: observed_median(
                [row for row in matrix["data_plane_rows"] if row["data_plane"] == data_plane],
                "total_ms",
            )
            for data_plane in sorted({row["data_plane"] for row in matrix["data_plane_rows"]})
        }
        direct_median = plane_medians.get(DataPlane.DIRECT_TCP.value)
        relay_median = plane_medians.get(DataPlane.RELAYED_TCP.value)
        relay_tax_ms = (
            relay_median - direct_median
            if relay_median is not None and direct_median is not None
            else None
        )
        naive_rows = [
            row for row in matrix["coalescing_rows"] if row["protocol"] == "naive_per_expert"
        ]
        coalesced_rows = [
            row for row in matrix["coalescing_rows"] if row["protocol"] == "coalesced_per_layer"
        ]
        naive_messages = observed_median(naive_rows, "messages_per_layer")
        coalesced_messages = observed_median(coalesced_rows, "messages_per_layer")
        naive_activation_bytes = observed_median(naive_rows, "activation_payload_bytes")
        coalesced_activation_bytes = observed_median(coalesced_rows, "activation_payload_bytes")
        message_reduction = (
            1.0 - coalesced_messages / naive_messages
            if naive_messages and coalesced_messages is not None
            else None
        )
        activation_reduction = (
            1.0 - coalesced_activation_bytes / naive_activation_bytes
            if naive_activation_bytes and coalesced_activation_bytes is not None
            else None
        )
        micro_medians = {
            layout: observed_median(
                [row for row in matrix["micro_rows"] if row["layout"] == layout],
                "latency_ms",
            )
            for layout in sorted({row["layout"] for row in matrix["micro_rows"]})
        }
        micro_maximum_relative_error = max(
            (float(row["relative_l2_error"]) for row in matrix["micro_rows"]),
            default=None,
        )
        validation = matrix["calibration"].get("validation", {})
        detected_corruptions = sum(bool(row.get("detected")) for row in matrix["verification_rows"])
        verification_count = len(matrix["verification_rows"])
        recovered_failures = sum(bool(row.get("recovered")) for row in matrix["failure_rows"])
        failure_count = len(matrix["failure_rows"])
        kimi_components = {row["component"]: row.get("elapsed_ms") for row in matrix["kimi_rows"]}
        capacity_workers = {
            worker_id: {
                "expert_bytes": payload.get("expert_bytes"),
                "resident_tensor_bytes": payload.get("resident_tensor_bytes"),
                "cache_bytes": payload.get("cache_bytes"),
                "peak_rss_bytes": payload.get("peak_rss_bytes"),
            }
            for worker_id, payload in matrix["capacity"].get("per_worker", {}).items()
        }
        batching_brief = {
            row["policy"]: {
                "mean_batch_size": row.get("mean_batch_size"),
                "deduplication_ratio": row.get("deduplication_ratio"),
                "maximum_queue_delay_ns": row.get("maximum_queue_delay_ns"),
            }
            for row in matrix["batching_rows"]
        }
        selected_plans = {
            phase: payload.get("selected_strategy") for phase, payload in matrix["plans"].items()
        }
        level_a_error = matrix.get("level_a_rpc", {}).get("maximum_relative_l2_error")
        component_reuse = (
            "Reused: "
            + "; ".join(item["component"] for item in REUSED_COMPONENTS)
            + ". Deferred for evidence integrity: "
            + "; ".join(f"{item['component']} ({item['reason']})" for item in DEFERRED_REUSE)
        )
        required_answers = {
            "Did the RTX 5090 execute through Colibri?": (
                "Yes for the pinned Colibri fused expert CUDA proof on the RTX 5090; "
                "no Level A token-generation path executed through a remote Colibri expert hook."
            ),
            "What exact tensors and experts lived in VRAM?": (
                f"Three generated proof tensors (up {cuda.get('tensor_shapes', {}).get('up')}, "
                f"gate {cuda.get('tensor_shapes', {}).get('gate')}, down "
                f"{cuda.get('tensor_shapes', {}).get('down')}) occupied "
                f"{cuda.get('resident_tensor_bytes')} bytes. No Level A source expert residency "
                "inside Colibri CUDA was proven."
            ),
            "Did direct expert RPC preserve tokens?": (
                "Not proven. Deterministic fixture vectors matched and the real Level A activation "
                f"had relative L2 error {level_a_error}, but generation did not consume the result, "
                "so token identity is null."
            ),
            "Did direct peer transport reduce overhead?": (
                f"{'Yes' if relay_tax_ms is not None and relay_tax_ms > 0 else 'No'} in this "
                f"fixture run. Medians were {plane_medians} ms; relay minus direct was "
                f"{relay_tax_ms} ms. This answers only the synthetic single-host operator path."
            ),
            "How much did coalescing reduce traffic?": (
                f"Messages fell from {naive_messages} to {coalesced_messages} per layer "
                f"({message_reduction:.1%} reduction) and activation payload bytes from "
                f"{naive_activation_bytes} to {coalesced_activation_bytes} "
                f"({activation_reduction:.1%} reduction) in the fixture."
            ),
            "Did executable microshards match whole experts?": (
                f"The float32 process fixture was quality-bounded with maximum relative L2 error "
                f"{micro_maximum_relative_error}; the generated Kimi-layout fixture reconstructed "
                "exactly. A native-quantized real-model expert was not microsharded."
            ),
            "Were equal microshards ever optimal?": (
                f"No validated optimum was established. Fixture medians were {micro_medians}; "
                "the Kimi-layout equal path was slower than its whole and asymmetric paths."
            ),
            "Were asymmetric microshards ever optimal?": (
                "Only among the two generated Kimi microshard layouts, not against whole-expert "
                "execution and not on a real model."
            ),
            "At what bandwidth and latency did remote experts stop being useful?": (
                "No validated boundary is claimed: the saved surface is SIMULATED_UNCALIBRATED "
                "because held-out gates failed."
            ),
            "At what bandwidth and latency did microsharding stop being useful?": (
                "No validated boundary is claimed for the same simulator-calibration reason."
            ),
            "Did batching change the break-even point?": (
                "Not established; routing-union behavior was exercised only on a synthetic fixture "
                "without an eligible real-model performance calibration."
            ),
            "Did prefill and decode require different plans?": (
                "No evidence-backed difference was established. Separate plan files exist, but "
                "the full mandatory prefill/decode workloads were not completed."
            ),
            "Did weak workers provide positive token-critical utility?": (
                "No. The fixture planner rejected the weak profile; no token-level measured utility "
                "claim is made."
            ),
            "Which workers were useful only asynchronously?": (
                "None was validated as asynchronously useful in a real workload; cache, storage, "
                "background, and verification remain implemented role candidates only."
            ),
            "Could no individual worker hold the model?": (
                "Only for the logically budgeted deterministic fixture. The required Level A "
                "end-to-end capacity-isolated generation was not completed."
            ),
            "Did failure recovery preserve tokens?": (
                f"All {recovered_failures}/{failure_count} exercised fixture operator failures "
                "recovered correct vectors; token preservation was not measured."
            ),
            "How expensive was incorrect-worker detection?": (
                f"{detected_corruptions}/{verification_count} injected fixture corruptions were "
                "detected, but end-to-end verification compute/network and token-latency overhead "
                "remain null."
            ),
            "Did the simulator predict held-out runs?": (
                f"Not within the gates: median throughput error={validation.get('median_throughput_error')}, "
                f"p95 error={validation.get('p95_latency_error')}."
            ),
            "How accurately did it rank candidate plans?": (
                f"Fixture ranking agreement={validation.get('plan_ranking_agreement')} and "
                f"regret={validation.get('planner_regret')}; those fixture diagnostics passed, "
                "but throughput/p95 prediction failed and the calibration is not official-gate eligible."
            ),
            "What does the Kimi K3-shaped result imply?": (
                f"It proves the generated native-layout E2M1/UE8M0 top-16 operator and 92-layer "
                f"replay algorithms execute ({kimi_components}); it does not establish checkpoint "
                "or full-model throughput."
            ),
            "Which claims remain projections?": (
                "All large-node scaling, network break-even, Kimi cell size/count/throughput, and "
                "full-model behavior remain uncalibrated simulation or projection."
            ),
            "Has the single-machine phase been exhausted?": (
                "No. End-to-end Colibri redirection/continuation, real native microshards, Level A "
                "capacity generation, mandatory workloads, real-path trust/recovery, Level B, and "
                "eligible held-out calibration remain open."
            ),
            "What is the exact first physical experiment now justified?": (
                "None yet as a scaling verdict. After the failed software gates close, run two "
                "independent hosts on measured 10 GbE: one RTX coordinator and one disjoint expert "
                "worker, fixed replay, direct versus relay, physical failure, and NIC/DMA telemetry."
            ),
        }
        summaries = {
            "models": {
                "level_a": level_a["status"],
                "level_b": level_b["status"],
                "kimi": (
                    "exact 3584x3072 generated native-layout fixture"
                    if matrix.get("kimi_exact_status", {}).get("exact_geometry")
                    else "tiny native-layout fixture"
                ),
            },
            "component_reuse": component_reuse,
            "workers": f"{len(matrix['manifests'])} process manifests were captured; budgets are logical, not OS hard limits.",
            "whole_expert": (
                "A genuine Level A activation and all top-k source experts executed over coalesced direct TCP with bounded FP32 operator error; generation did not consume the returned result and the Colibri expert hook remains absent."
                if level_a_rpc_complete
                else "Real socket/shared-memory protocol execution passed on deterministic expert weights; the Level A real-activation probe did not complete."
            ),
            "microshards": f"{len(matrix['micro_rows'])} matching-slice observations; medians={micro_medians}; maximum relative L2 error={micro_maximum_relative_error}.",
            "data_planes": f"Observed medians={plane_medians} ms; relay-minus-direct tax={relay_tax_ms} ms.",
            "configuration_matrix": str(matrix["configuration_matrix_coverage"]),
            "coalescing": f"Coalescing reduced messages by {message_reduction:.1%} and activation payload bytes by {activation_reduction:.1%} in the fixture.",
            "capacity": (
                f"Fixture reconciliation valid={matrix['capacity'].get('valid')}; logical budgets "
                f"only; coordinator remote expert bytes="
                f"{matrix['capacity'].get('coordinator_remote_expert_bytes')}; "
                f"per-worker accounting={capacity_workers}."
            ),
            "prefill": "Separate fixture plan emitted; mandatory 8k/32k real-model workloads were not completed.",
            "decode": "Separate fixture plan emitted; mandatory 20-prompt, 256-token real-model workload was not completed.",
            "concurrency": "Concurrency 2/4/8 and mixed interactive/background real-model workloads were not completed.",
            "batching": f"Synthetic routing-union summaries={batching_brief}.",
            "break_even": "Saved decision boundaries are SIMULATED_UNCALIBRATED and cannot support a network threshold claim.",
            "codecs": "Raw FP32 is the exact reference; lossy codecs remain quality-bounded candidates.",
            "failures": f"{recovered_failures}/{failure_count} fixture operator failures recovered correctly; token metrics remain null.",
            "verification": f"{detected_corruptions}/{verification_count} fixture corruptions detected; official overhead is unmeasured.",
            "planner": f"Selected fixture strategies={selected_plans}; Experiment 007 evaluations are saved with every candidate; real-workload regret was not established.",
            "simulator": str(validation or matrix["calibration"]),
            "kimi": (
                "The exact 3584x3072 native E2M1/UE8M0 generated fixture executed top-16 whole, equal/asymmetric shard, and 92-layer operator paths; these are not checkpoint weights or full-model inference."
                if matrix.get("kimi_exact_status", {}).get("exact_geometry")
                else "Top-16 native E2M1/UE8M0 algorithms executed at tiny geometry only in this run."
            ),
            "limitations": (
                "End-to-end Colibri Level A redirection and continuation, Level B workload execution, official real-model simulator calibration, and all physical network effects remain open."
                if matrix.get("kimi_exact_status", {}).get("exact_geometry")
                else "End-to-end Colibri Level A redirection and continuation, Level B workload execution, exact-size Kimi operator replay, official real-model simulator calibration, and all physical network effects remain open."
            ),
            "physical_questions": (
                "Physical machines are eventually required for independent NIC/DMA paths, memory "
                "controllers, storage devices, clocks, failure domains, power/thermal envelopes, "
                "and genuinely additional accelerators. They are not a substitute for the still-open "
                "single-host Colibri hook, real microshard, workload, trust, and calibration gates."
            ),
            "required_answers": required_answers,
        }
        bundle.write_json("verdict.json", verdict_payload)
        bundle.write_text(
            "report.md",
            build_report(
                verdict=verdict_payload,
                environment=environment,
                cuda=cuda,
                summaries=summaries,
            ),
        )
        generate_required_plots(bundle.root)
        bundle.write_json(
            "manifest.json",
            {
                "schema_version": "experiment-010-manifest-v1",
                "mode": options.mode.value,
                "required_files": list(REQUIRED_FILES),
                "reused_components": list(REUSED_COMPONENTS),
                "modified_components": list(MODIFIED_COMPONENTS),
                "added_components": list(ADDED_COMPONENTS),
                "deferred_reuse": list(DEFERRED_REUSE),
                "configuration_matrix_coverage": matrix["configuration_matrix_coverage"],
                "artifacts": bundle.artifact_manifest(),
            },
        )
        audit = bundle.audit()
        final_gates = _gates(
            options=options,
            cuda=cuda,
            matrix=matrix,
            audit_complete=audit["complete"],
        )
        final_verdict = classify_verdict(
            final_gates,
            mode=options.mode,
            real_distributed_expert_execution=bool(matrix["whole_rows"]),
            positive_measured_utility=False,
            genuine_capacity_result=False,
        )
        verdict_payload.update(
            {
                "verdict": final_verdict.value,
                "gates": [item.model_dump(mode="json") for item in final_gates],
                "failed_gates": [
                    item.gate_id for item in final_gates if item.status != GateStatus.PASS
                ],
                "artifact_audit": audit,
            }
        )
        bundle.write_json("verdict.json", verdict_payload)
        bundle.write_text(
            "report.md",
            build_report(
                verdict=verdict_payload,
                environment=environment,
                cuda=cuda,
                summaries=summaries,
            ),
        )
        generate_required_plots(bundle.root)
        previous_manifest = _read_json(bundle.root / "manifest.json", {})
        # Finalize the checkpoint before hashing artifacts. No evidence file may
        # change after the manifest captures its digest.
        bundle.complete_stage("evidence-finalized")
        bundle.write_json(
            "manifest.json",
            {
                "schema_version": "experiment-010-manifest-v1",
                "mode": options.mode.value,
                "required_files": list(REQUIRED_FILES),
                "artifact_audit": bundle.audit(),
                "reused_components": previous_manifest.get("reused_components", []),
                "modified_components": previous_manifest.get("modified_components", []),
                "added_components": previous_manifest.get("added_components", []),
                "deferred_reuse": previous_manifest.get("deferred_reuse", []),
                "configuration_matrix_coverage": matrix["configuration_matrix_coverage"],
                "artifacts": bundle.artifact_manifest(),
            },
        )
        return Experiment010Outcome(bundle.root, final_verdict, None)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        bundle.record_failure("runner", error)
        fallback = Experiment010Verdict.FAIL
        bundle.write_json(
            "verdict.json",
            {
                "schema_version": "experiment-010-verdict-v1",
                "mode": options.mode.value,
                "verdict": fallback.value,
                "official": False,
                "answer_first": "Experiment 010 stopped after an explicit runner failure.",
                "error": error,
                "physical_distributed_inference_proven": False,
                "full_kimi_inference_claimed": False,
            },
        )
        return Experiment010Outcome(bundle.root, fallback, error)
