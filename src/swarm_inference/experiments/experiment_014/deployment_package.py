"""Bind the final Kimi placement to live workers and exact runtime identity."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from swarm_inference.config.models import Backend, WorkerCapability
from swarm_inference.coordinator.deployment import build_load_stage_request
from swarm_inference.experiments.experiment_014.runtime_qualification import (
    BUILD_ARGUMENTS,
    CERTIFICATE_SCHEMA,
    KIMI_OPERATION_CLASSES,
    KIMI_STAGE_ROLES,
    PHYSICAL_CERTIFICATE_GATES,
    RuntimeQualificationError,
    build_native_source_manifest,
    validate_linux_runtime_certificate,
)
from swarm_inference.model.partition import StageAssignment
from swarm_inference.model.product import ModelResolutionPolicy, ProductModelSpec
from swarm_inference.protocol.product import (
    DirectedLinkSelection,
    PlanWorkerAssignment,
    ProductStagePlan,
    StagePlanReport,
    WorkerEligibilityReport,
    WorkerProductStatus,
    WorkersResponse,
)

SCHEMA_VERSION = "experiment-014-k3-bound-deployment-v1"
FIXTURE_SCHEMA_VERSION = "experiment-014-k3-deployment-identity-fixture-v2"
GIB = 1024**3
PRODUCTION_BATCH = 8
MAX_SEQUENCE_TOKENS = 8192
MODEL_ID = "moonshotai/Kimi-K3"
ADAPTER_ID = "kimi_k3_cuda"
FAST_PATH_ID = "colibri-kimi-k3-cuda"
DEVICE = "native-cuda:0"


class DeploymentBindingError(RuntimeError):
    """The live fleet cannot be bound to the certified placement."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentBindingError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise DeploymentBindingError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DeploymentBindingError(f"{name} must be 64 lowercase hexadecimal digits")
    return value


def _validate_live_worker(
    status: WorkerProductStatus,
    placement_worker: dict[str, Any],
    checkpoint_fingerprint: str,
) -> None:
    capability = status.capability
    worker_id = str(placement_worker["worker_id"])
    errors: list[str] = []
    expected_control = capability.control_endpoint or capability.endpoint
    if capability.worker_id != worker_id:
        errors.append("worker ID mismatch")
    if not status.healthy_registration or status.last_error is not None:
        errors.append("registration is unhealthy")
    if status.active_sessions or status.loaded_stages:
        errors.append("worker is not idle and unloaded")
    if not status.control_endpoint or status.control_endpoint != expected_control:
        errors.append("control endpoint is absent or inconsistent")
    if not status.data_endpoint or status.data_endpoint != capability.data_plane_endpoint:
        errors.append("data endpoint is absent or inconsistent")
    if capability.backend != Backend.TORCH_CUDA:
        errors.append("backend is not torch-cuda")
    if capability.device_identifier != DEVICE:
        errors.append(f"device is not {DEVICE}")
    if not capability.stage_runtime_enabled or capability.stage_ring_protocol_version != 1:
        errors.append("canonical stage runtime protocol is unavailable")
    if "canonical-native-stage" not in capability.supported_stage_execution_backends:
        errors.append("canonical native-stage execution is unavailable")
    if ADAPTER_ID not in capability.supported_model_adapters:
        errors.append("Kimi CUDA adapter is unavailable")
    if "float32" not in {
        *capability.supported_dtypes,
        *capability.supported_activation_dtypes,
    }:
        errors.append("float32 execution was not advertised")
    if capability.model_fingerprint != checkpoint_fingerprint:
        errors.append("configured model identity fingerprint differs")
    if "RTX 3090" not in str(capability.gpu_model or ""):
        errors.append("GPU is not an RTX 3090")
    physical_bytes = int(placement_worker["memory"]["physical_vram_bytes"])
    required_bytes = int(placement_worker["memory"]["planned_total_vram_bytes"])
    if capability.total_vram_bytes < physical_bytes:
        errors.append("physical VRAM is below the placement contract")
    if capability.effective_memory_bytes < required_bytes:
        errors.append("currently effective VRAM is below the stage requirement")
    network = placement_worker["expected_network"]
    minimum_bytes_s = float(network["minimum_bandwidth_gbps"]) * 1_000_000_000 / 8
    if capability.coordinator_latency_ms > float(network["maximum_rtt_ms"]):
        errors.append("coordinator latency exceeds the coarse edge class")
    if min(
        capability.upload_bandwidth_bytes_s,
        capability.download_bandwidth_bytes_s,
    ) < minimum_bytes_s:
        errors.append("advertised bandwidth is below the coarse edge class")
    if errors:
        raise DeploymentBindingError(f"{worker_id}: " + "; ".join(errors))


def _bind_plan(
    placement: dict[str, Any],
    workers_document: dict[str, Any],
    *,
    runtime_identity: dict[str, str],
) -> ProductStagePlan:
    runtime_path = str(runtime_identity.get("worker_local_path", "")).strip()
    if not runtime_path:
        raise DeploymentBindingError("native runtime worker-local path is required")
    runtime_sha = _require_sha256(
        str(runtime_identity.get("sha256", "")),
        name="native runtime SHA-256",
    )
    if placement.get("status") != "PASS" or int(placement.get("node_count", 0)) != 93:
        raise DeploymentBindingError("placement is not the passing 93-worker topology")
    if placement.get("topology", {}).get("class") != "WHOLE-LAYER":
        raise DeploymentBindingError("placement is not the selected whole-layer topology")
    try:
        live = WorkersResponse.model_validate(workers_document)
    except ValidationError as exc:
        raise DeploymentBindingError("live worker document failed schema validation") from exc
    expected_ids = [f"k3-worker-{index:03d}" for index in range(93)]
    by_id: dict[str, WorkerProductStatus] = {}
    for row in live.workers:
        worker_id = row.capability.worker_id
        if worker_id in by_id:
            raise DeploymentBindingError(f"duplicate live worker identity: {worker_id}")
        by_id[worker_id] = row
    if sorted(by_id) != expected_ids:
        missing = sorted(set(expected_ids) - set(by_id))
        unexpected = sorted(set(by_id) - set(expected_ids))
        raise DeploymentBindingError(
            f"live fleet is not exactly 93 assigned workers; missing={missing} unexpected={unexpected}"
        )
    placement_workers = sorted(
        placement["workers"], key=lambda row: int(row["worker_index"])
    )
    if [row["worker_id"] for row in placement_workers] != expected_ids:
        raise DeploymentBindingError("placement worker identities are not canonical")
    checkpoint = placement["checkpoint"]
    checkpoint_fingerprint = _require_sha256(
        str(checkpoint["checkpoint_fingerprint"]),
        name="checkpoint fingerprint",
    )
    assignments: list[PlanWorkerAssignment] = []
    eligibility: list[WorkerEligibilityReport] = []
    headroom: dict[str, int] = {}
    compute: dict[str, float] = {}
    memory: dict[str, int] = {}
    for stage_id, placement_worker in enumerate(placement_workers):
        status = by_id[expected_ids[stage_id]]
        _validate_live_worker(status, placement_worker, checkpoint_fingerprint)
        capability = status.capability
        stage_memory = placement_worker["memory"]
        resident_bytes = int(
            stage_memory["measured_or_representative_resident_device_bytes"]
        )
        state_bytes = int(stage_memory["production_eight_stream_state_bytes"])
        temporary_bytes = sum(
            int(stage_memory[name])
            for name in (
                "workspace_reserve_bytes",
                "activation_and_communication_reserve_bytes",
                "serving_scheduler_reserve_bytes",
            )
        )
        projected_ms = float(
            placement_worker["expected_compute"]["projected_rtx3090_wall_p50_ms"]
        )
        assignment = StageAssignment(
            stage_id=stage_id,
            layer_start=stage_id,
            layer_end=stage_id + 1,
            layer_ids=(stage_id,),
            weight_bytes=resident_bytes,
            estimated_compute_ns=max(1, round(projected_ms * 1_000_000)),
            measured_compute_ns=None,
            kv_cache_bytes_per_token=state_bytes // (PRODUCTION_BATCH * MAX_SEQUENCE_TOKENS),
            peak_temporary_bytes=temporary_bytes,
            activation_bytes=int(placement["network_admission"]["activation_payload_bytes"]),
            device=DEVICE,
            owns_embeddings=stage_id == 0,
            owns_final_norm=stage_id == 92,
            owns_output_projection=stage_id == 92,
        )
        required_memory = int(stage_memory["planned_total_vram_bytes"])
        item = PlanWorkerAssignment(
            stage_id=stage_id,
            worker_id=capability.worker_id,
            control_endpoint=str(status.control_endpoint),
            data_endpoint=str(status.data_endpoint),
            device=DEVICE,
            effective_memory_bytes=capability.effective_memory_bytes,
            required_memory_bytes=required_memory,
            assignment=assignment,
            fast_path_id=FAST_PATH_ID,
            fast_path_mode="resident",
            native_runtime_library=runtime_path,
            native_runtime_library_sha256=runtime_sha,
            worker_role="critical_path_stage",
        )
        assignments.append(item)
        free = capability.effective_memory_bytes - required_memory
        headroom[capability.worker_id] = free
        compute[str(stage_id)] = projected_ms
        memory[capability.worker_id] = required_memory
        eligibility.append(
            WorkerEligibilityReport(
                worker_id=capability.worker_id,
                eligible=True,
                effective_memory_bytes=capability.effective_memory_bytes,
                active_session_count=status.active_sessions,
                exact_model_identity=True,
                artifact_provisionable=True,
                measured_profile=False,
            )
        )
    links = [
        DirectedLinkSelection(
            source_worker_id=assignments[index].worker_id,
            destination_worker_id=assignments[index + 1].worker_id,
            latency_ms=float(placement["network_admission"]["maximum_rtt_ms"]),
            transfer_ms=(
                int(placement["network_admission"]["activation_payload_bytes"])
                * 8
                / (float(placement["network_admission"]["minimum_bandwidth_gbps"]) * 1e9)
                * 1000
            ),
            upload_bytes_per_s=(
                float(placement["network_admission"]["minimum_bandwidth_gbps"])
                * 1e9
                / 8
            ),
            fresh=True,
            measured=False,
            source_endpoint=assignments[index].data_endpoint,
            destination_endpoint=assignments[index + 1].data_endpoint,
            assumption="worker admitted against kimi_coarse_stage_fp32_v1",
        )
        for index in range(92)
    ]
    binding_identity = {
        "placement_ownership_sha256": placement["coverage"]["canonical_ownership_sha256"],
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "runtime_sha256": runtime_sha,
        "runtime_path": runtime_path,
        "workers": [
            {
                "worker_id": item.worker_id,
                "control_endpoint": item.control_endpoint,
                "data_endpoint": item.data_endpoint,
                "assignment_sha256": placement_workers[item.stage_id][
                    "assignment_sha256"
                ],
            }
            for item in assignments
        ],
    }
    binding_sha = _canonical_sha256(binding_identity)
    model = ProductModelSpec(
        model_id=MODEL_ID,
        model_revision=str(checkpoint["revision"]),
        tokenizer_revision=str(checkpoint["revision"]),
        adapter_id=ADAPTER_ID,
        dtype="float32",
        layer_count=93,
        hidden_size=7168,
        metadata_hash=checkpoint_fingerprint,
        model_fingerprint=checkpoint_fingerprint,
        model_format="safetensors",
        quantization="mxfp4",
        resolution_policy=ModelResolutionPolicy.LOCAL_ONLY,
    )
    report = StagePlanReport(
        selected_topology="B-whole-layer-checkpoint-aligned-93",
        worker_assignments=assignments,
        memory_estimates_bytes=memory,
        compute_estimates_ms=compute,
        network_estimates_ms={
            f"{index}->{index + 1}": links[index].latency_ms + links[index].transfer_ms
            for index in range(92)
        },
        reason_for_selection=(
            "Measured Experiment 014 topology B: sub-layer execution is functional but "
            "retains only 81.371% complete-layer throughput at its best logical topology."
        ),
        candidates=[],
        worker_eligibility=eligibility,
        objective_mode="throughput",
        distributed_expected_throughput_tokens_s=97.4565,
        directed_links_selected=links,
        per_stage_headroom_bytes=headroom,
        search_method="experiment-014-measured-candidate-selection",
        beam_width=1,
        measurement_freshness={
            "local_kimi": "measured on exact candidate binary",
            "rtx3090": "projected pending single-3090 canary",
            "network": "admission-measured per physical worker",
        },
        unmeasured_assumptions=[
            "RTX 3090 stage timing remains projected until the physical canary",
            "links are admitted at measured thresholds but not locally reproducible as 93 GPUs",
        ],
        confidence="mixed",
        objective_components={
            "projected_aggregate_tokens_s": 97.4565,
            "maximum_coarse_rtt_ms": 5.0,
            "minimum_coarse_bandwidth_gbps": 5.0,
        },
    )
    return ProductStagePlan(
        plan_id=f"k3-plan-{binding_sha[:24]}",
        topology_id=f"k3-whole-layer-93-{binding_sha[:24]}",
        generation=1,
        created_monotonic_ns=time.monotonic_ns(),
        model=model,
        engine_id="native-stage",
        engine_revision=runtime_sha,
        stage_count=93,
        partition_method="equal",
        max_sequence_tokens=MAX_SEQUENCE_TOKENS,
        assignments=assignments,
        routed_expert_engine="native-stage",
        optional_mechanisms={
            "continuous_batching": True,
            "sub_layer_microwork": False,
        },
        prefill_parameters={
            "maximum_context_tokens": MAX_SEQUENCE_TOKENS,
            "contexts_certified_locally": [1024, 4096, 8192, 16384],
        },
        decode_parameters={
            "production_batch": PRODUCTION_BATCH,
            "maximum_certified_batch": PRODUCTION_BATCH,
            "native_primitive_maximum_certified_batch": 16,
        },
        report=report,
    )


def build_bound_product_plan(
    placement_path: Path,
    workers_status_path: Path,
    output_path: Path,
    *,
    runtime_certificate_path: Path,
    native_source_manifest_path: Path,
) -> dict[str, Any]:
    """Bind the immutable placement to one exact healthy live fleet."""

    placement_source = placement_path.resolve()
    workers_source = workers_status_path.resolve()
    placement = _read(placement_source)
    workers = _read(workers_source)
    try:
        runtime_identity = validate_linux_runtime_certificate(
            runtime_certificate_path,
            native_source_manifest_path,
            placement_source,
        )
    except RuntimeQualificationError as exc:
        raise DeploymentBindingError(str(exc)) from exc
    plan = _bind_plan(
        placement,
        workers,
        runtime_identity=runtime_identity,
    )
    _atomic_json(output_path, plan.model_dump(mode="json"))
    plan_source = output_path.resolve()
    round_trip = ProductStagePlan.model_validate_json(
        plan_source.read_text(encoding="utf-8")
    )
    if round_trip != plan:
        raise DeploymentBindingError("written product plan failed exact typed round trip")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "plan_path": str(plan_source),
        "plan_sha256": _sha256(plan_source),
        "plan_id": plan.plan_id,
        "topology_id": plan.topology_id,
        "worker_count": plan.stage_count,
        "model_revision": plan.model.model_revision,
        "model_fingerprint": plan.model.model_fingerprint,
        "native_runtime_library": runtime_identity["worker_local_path"],
        "native_runtime_library_sha256": runtime_identity["sha256"],
        "runtime_certificate_sha256": runtime_identity["certificate_sha256"],
        "native_source_manifest_sha256": runtime_identity[
            "source_manifest_sha256"
        ],
        "production_batch": plan.decode_parameters["production_batch"],
        "placement_sha256": _sha256(placement_source),
        "workers_status_sha256": _sha256(workers_source),
    }


def _fixture_capability(worker_id: str, index: int, fingerprint: str) -> WorkerCapability:
    control = f"127.0.0.1:{50000 + index}"
    data = f"127.0.0.1:{51000 + index}"
    now = time.time_ns()
    return WorkerCapability(
        worker_id=worker_id,
        node_id=worker_id,
        public_key=f"fixture-public-key-{index}",
        hostname=worker_id,
        operating_system="Linux fixture",
        architecture="x86_64",
        backend=Backend.TORCH_CUDA,
        cpu_model="fixture",
        logical_cpu_count=8,
        physical_cpu_count=4,
        total_ram_bytes=64 * GIB,
        available_ram_bytes=48 * GIB,
        gpu_model="NVIDIA GeForce RTX 3090",
        total_vram_bytes=24 * GIB,
        available_vram_bytes=24 * GIB,
        supported_dtypes=["float32"],
        supported_quantisation_formats=["mxfp4"],
        upload_bandwidth_bytes_s=12.5 * 1e9,
        download_bandwidth_bytes_s=12.5 * 1e9,
        coordinator_latency_ms=0.25,
        memory_limit_bytes=24 * GIB,
        endpoint=control,
        control_endpoint=control,
        data_plane_endpoint=data,
        device_identifier=DEVICE,
        stage_ring_protocol_version=1,
        supported_model_adapters=[ADAPTER_ID],
        supported_stage_execution_backends=["canonical-native-stage"],
        supported_activation_dtypes=["float32"],
        configured_memory_limit_bytes=24 * GIB,
        stage_runtime_enabled=True,
        model_fingerprint=fingerprint,
        agent_version="fixture",
        runtime_version="fixture",
        build_id="fixture",
        package_lock_hash="f" * 64,
        product_protocol_major=1,
        product_protocol_minor=0,
        artifact_format_versions=[1, 2],
        platform_implementation_status="implemented",
        software_validation_status="validated",
        physical_validation_status="validated",
        validation_evidence_ids=[f"fixture-admission-{index}"],
        latest_validation_unix_ns=now,
        validation_detail="H014-037b1 retained identity fixture",
    )


def _fixture_workers(placement: dict[str, Any]) -> dict[str, Any]:
    fingerprint = str(placement["checkpoint"]["checkpoint_fingerprint"])
    rows: list[WorkerProductStatus] = []
    for index in range(93):
        worker_id = f"k3-worker-{index:03d}"
        capability = _fixture_capability(worker_id, index, fingerprint)
        rows.append(
            WorkerProductStatus(
                capability=capability,
                healthy_registration=True,
                heartbeat_age_s=0.01,
                last_heartbeat_unix_ns=time.time_ns(),
                control_endpoint=capability.control_endpoint,
                data_endpoint=capability.data_plane_endpoint,
            )
        )
    return WorkersResponse(workers=rows).model_dump(mode="json")


def _rejected(callable_: Any) -> tuple[bool, str]:
    try:
        callable_()
    except (
        DeploymentBindingError,
        RuntimeQualificationError,
        ValidationError,
        ValueError,
    ) as exc:
        return True, f"{type(exc).__name__}: {exc}"
    return False, "candidate was incorrectly accepted"


def benchmark_deployment_identity_fixture(
    placement_path: Path,
    output_path: Path,
    *,
    cycle_id: str = "H014-037b1",
) -> dict[str, Any]:
    """Exercise the exact 93-stage binder and its fail-closed controls."""

    placement_source = placement_path.resolve()
    placement = _read(placement_source)
    workers = _fixture_workers(placement)
    runtime_path = "/opt/swarm/runtime/libcoli_cuda-sm86.so"
    runtime_sha = hashlib.sha256(b"logical-linux-elf-distinct-from-windows").hexdigest()
    root = Path(__file__).resolve().parents[4]
    source_manifest_path = output_path.resolve().with_name(
        output_path.stem + "-native-source-manifest.json"
    )
    build_native_source_manifest(
        root / "third_party" / "colibri" / "c",
        placement_source,
        source_manifest_path,
    )
    source_manifest_sha = _sha256(source_manifest_path)
    source_manifest = _read(source_manifest_path)
    certificate_path = output_path.resolve().with_name(
        output_path.stem + "-logical-runtime-certificate.json"
    )
    certificate = {
        "schema_version": CERTIFICATE_SCHEMA,
        "status": "PASS",
        "evidence_kind": "logical_fixture",
        "platform": "linux-x86_64",
        "references": {
            "placement_sha256": _sha256(placement_source),
            "source_manifest_sha256": source_manifest_sha,
            "source_bundle_sha256": source_manifest["source_bundle_sha256"],
            "windows_reference_sha256": source_manifest[
                "windows_reference_sha256"
            ],
        },
        "build_arguments": list(BUILD_ARGUMENTS),
        "binary": {
            "worker_local_path": runtime_path,
            "sha256": runtime_sha,
            "bytes": 123456,
            "elf_magic": True,
        },
        "operation_classes": list(KIMI_OPERATION_CLASSES),
        "stage_roles": list(KIMI_STAGE_ROLES),
        "acceptance_gates": {name: True for name in PHYSICAL_CERTIFICATE_GATES},
        "disclosure": "logical certificate accepted only inside H014-037b3 fixture",
    }
    _atomic_json(certificate_path, certificate)
    runtime_identity = validate_linux_runtime_certificate(
        certificate_path,
        source_manifest_path,
        placement_source,
        allow_logical_fixture=True,
    )
    plan = _bind_plan(
        placement,
        workers,
        runtime_identity=runtime_identity,
    )
    fixture_plan_path = output_path.resolve().with_name(
        output_path.stem + "-bound-plan.json"
    )
    _atomic_json(fixture_plan_path, plan.model_dump(mode="json"))
    round_trip = ProductStagePlan.model_validate_json(
        fixture_plan_path.read_text(encoding="utf-8")
    )
    initial = [
        build_load_stage_request(
            plan,
            item,
            request_id=f"fixture:load:{item.stage_id}",
            route_generation=1,
            lease_expiry_unix_ns=time.time_ns() + 60_000_000_000,
            deadline_unix_ns=time.time_ns() + 5_000_000_000,
        )
        for item in plan.assignments
    ]
    recovery_plan = plan.model_copy(update={"generation": 2}, deep=True)
    recovery = [
        build_load_stage_request(
            recovery_plan,
            item,
            request_id=f"fixture:recover-load:2:{item.stage_id}",
            route_generation=2,
            lease_expiry_unix_ns=time.time_ns() + 60_000_000_000,
            deadline_unix_ns=time.time_ns() + 5_000_000_000,
        )
        for item in recovery_plan.assignments
    ]

    def exact_request(request: Any, generation: int) -> bool:
        return bool(
            request.model_id == MODEL_ID
            and request.model_revision == placement["checkpoint"]["revision"]
            and request.tokenizer_revision == placement["checkpoint"]["revision"]
            and request.adapter_id == ADAPTER_ID
            and request.model_content_fingerprint
            == placement["checkpoint"]["checkpoint_fingerprint"]
            and request.native_runtime_library == runtime_path
            and request.native_runtime_library_sha256 == runtime_sha
            and request.fast_path_id == FAST_PATH_ID
            and request.fast_path_mode == "resident"
            and request.fast_path_batch_bucket == PRODUCTION_BATCH
            and request.fast_path_context_bucket == MAX_SEQUENCE_TOKENS
            and request.device == DEVICE
            and request.dtype == "float32"
            and request.route_generation == generation
            and not request.allow_download
        )

    negative_controls: dict[str, dict[str, Any]] = {}

    def record(name: str, callable_: Any) -> bool:
        rejected, detail = _rejected(callable_)
        negative_controls[name] = {"rejected": rejected, "detail": detail}
        return rejected

    missing = copy.deepcopy(workers)
    missing["workers"].pop()
    missing_rejected = record(
        "missing_worker",
        lambda: _bind_plan(
            placement,
            missing,
            runtime_identity=runtime_identity,
        ),
    )
    wrong_device = copy.deepcopy(workers)
    wrong_device["workers"][17]["capability"]["device_identifier"] = "cuda:0"
    wrong_device_rejected = record(
        "wrong_device",
        lambda: _bind_plan(
            placement,
            wrong_device,
            runtime_identity=runtime_identity,
        ),
    )
    wrong_identity = copy.deepcopy(workers)
    wrong_identity["workers"][29]["capability"]["model_fingerprint"] = "0" * 64
    wrong_identity_rejected = record(
        "wrong_model_identity",
        lambda: _bind_plan(
            placement,
            wrong_identity,
            runtime_identity=runtime_identity,
        ),
    )
    unhealthy = copy.deepcopy(workers)
    unhealthy["workers"][41]["healthy_registration"] = False
    unhealthy_rejected = record(
        "unhealthy_registration",
        lambda: _bind_plan(
            placement,
            unhealthy,
            runtime_identity=runtime_identity,
        ),
    )
    endpoint_mismatch = copy.deepcopy(workers)
    endpoint_mismatch["workers"][53]["data_endpoint"] = "127.0.0.1:1"
    endpoint_rejected = record(
        "endpoint_mismatch",
        lambda: _bind_plan(
            placement,
            endpoint_mismatch,
            runtime_identity=runtime_identity,
        ),
    )
    bad_runtime_rejected = record(
        "malformed_runtime_sha",
        lambda: _bind_plan(
            placement,
            workers,
            runtime_identity={
                **runtime_identity,
                "sha256": "not-a-sha",
            },
        ),
    )
    pair_payload = plan.assignments[0].model_dump(mode="python")
    pair_payload["native_runtime_library_sha256"] = None
    pair_rejected = record(
        "missing_runtime_pair_member",
        lambda: PlanWorkerAssignment.model_validate(pair_payload),
    )

    def certificate_control(name: str, mutate: Any) -> bool:
        candidate = copy.deepcopy(certificate)
        mutate(candidate)
        candidate_path = output_path.resolve().with_name(
            output_path.stem + f"-{name}.json"
        )
        _atomic_json(candidate_path, candidate)
        return record(
            name,
            lambda: validate_linux_runtime_certificate(
                candidate_path,
                source_manifest_path,
                placement_source,
                allow_logical_fixture=True,
            ),
        )

    fixture_rejected_by_production = record(
        "logical_fixture_rejected_by_production",
        lambda: validate_linux_runtime_certificate(
            certificate_path,
            source_manifest_path,
            placement_source,
        ),
    )
    absent_certificate_rejected = record(
        "bare_hash_without_certificate",
        lambda: validate_linux_runtime_certificate(
            output_path.resolve().with_name("absent-runtime-certificate.json"),
            source_manifest_path,
            placement_source,
        ),
    )
    wrong_platform_rejected = certificate_control(
        "wrong_platform",
        lambda value: value.update({"platform": "windows-amd64"}),
    )
    wrong_source_rejected = certificate_control(
        "wrong_source",
        lambda value: value["references"].update(
            {"source_bundle_sha256": "0" * 64}
        ),
    )
    wrong_placement_rejected = certificate_control(
        "wrong_placement",
        lambda value: value["references"].update({"placement_sha256": "0" * 64}),
    )
    failed_gate_rejected = certificate_control(
        "failed_physical_gate",
        lambda value: value["acceptance_gates"].update(
            {"nvidia_smi_healthy_after": False}
        ),
    )
    windows_hash_reuse_rejected = certificate_control(
        "windows_hash_reuse",
        lambda value: value["binary"].update(
            {"sha256": source_manifest["windows_reference_sha256"]}
        ),
    )
    gates = {
        "typed_plan_round_trip_exact": round_trip == plan,
        "exact_93_assignments": len(plan.assignments) == 93,
        "all_assignments_model_identity_exact": all(
            item.native_runtime_library == runtime_path
            and item.native_runtime_library_sha256 == runtime_sha
            for item in plan.assignments
        ),
        "all_initial_load_requests_identity_exact": all(
            exact_request(request, 1) for request in initial
        ),
        "all_recovery_load_requests_identity_exact": all(
            exact_request(request, 2) for request in recovery
        ),
        "production_batch_eight_propagated": all(
            request.fast_path_batch_bucket == PRODUCTION_BATCH
            for request in [*initial, *recovery]
        ),
        "missing_worker_rejected": missing_rejected,
        "wrong_device_rejected": wrong_device_rejected,
        "wrong_model_identity_rejected": wrong_identity_rejected,
        "unhealthy_registration_rejected": unhealthy_rejected,
        "endpoint_mismatch_rejected": endpoint_rejected,
        "malformed_runtime_sha_rejected": bad_runtime_rejected,
        "missing_runtime_pair_member_rejected": pair_rejected,
        "linux_sha_distinct_from_windows_reference": (
            runtime_sha != source_manifest["windows_reference_sha256"]
        ),
        "logical_fixture_rejected_by_production": fixture_rejected_by_production,
        "bare_hash_without_certificate_rejected": absent_certificate_rejected,
        "wrong_platform_certificate_rejected": wrong_platform_rejected,
        "wrong_source_certificate_rejected": wrong_source_rejected,
        "wrong_placement_certificate_rejected": wrong_placement_rejected,
        "failed_physical_gate_certificate_rejected": failed_gate_rejected,
        "windows_hash_reuse_rejected": windows_hash_reuse_rejected,
    }
    receipt = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "hypothesis": (
            "A source- and canary-qualified Linux ELF may have a distinct SHA from the "
            "Windows evidence DLL while every assignment and initial/recovery request "
            "carries its exact identity; a bare or mismatched identity must reject."
        ),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "configuration": {
            "logical_workers": 93,
            "model_id": MODEL_ID,
            "model_revision": placement["checkpoint"]["revision"],
            "model_fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
            "runtime_library": runtime_path,
            "runtime_sha256": runtime_sha,
            "windows_reference_sha256": source_manifest[
                "windows_reference_sha256"
            ],
            "runtime_certificate_sha256": _sha256(certificate_path),
            "native_source_manifest_sha256": source_manifest_sha,
            "production_batch": PRODUCTION_BATCH,
            "context_tokens": MAX_SEQUENCE_TOKENS,
            "fixture_only": True,
        },
        "result": {
            "assignment_count": len(plan.assignments),
            "initial_load_request_count": len(initial),
            "recovery_load_request_count": len(recovery),
            "plan_id": plan.plan_id,
            "topology_id": plan.topology_id,
            "fixture_plan_path": str(fixture_plan_path),
            "fixture_plan_sha256": _sha256(fixture_plan_path),
            "placement_path": str(placement_source),
            "placement_sha256": _sha256(placement_source),
            "runtime_certificate_path": str(certificate_path),
            "native_source_manifest_path": str(source_manifest_path),
        },
        "negative_controls": negative_controls,
        "acceptance_gates": gates,
        "inspection": (
            "The logical ELF hash intentionally differs from the Windows reference. The "
            "canonical request builder propagated it through both generations for all "
            "93 stages, while production rejected the fixture-only certificate."
        ),
        "bottleneck": (
            "Physical Linux runtime identity remains pending the single-3090 canary; "
            "the logical product path is now fail-closed."
        ),
        "decision": "RETAIN if every gate passes; otherwise do not package or deploy",
        "redesign": (
            "Build and hash the Linux sm_86 runtime on the canary, then bind the real live "
            "worker-status document with this same command."
        ),
    }
    _atomic_json(output_path, receipt)
    return {
        **receipt,
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path.resolve()),
    }


__all__ = [
    "DeploymentBindingError",
    "benchmark_deployment_identity_fixture",
    "build_bound_product_plan",
]
