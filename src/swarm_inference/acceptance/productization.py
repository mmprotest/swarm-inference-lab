"""Honest, versioned productization acceptance and evidence generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import psutil

ACCEPTANCE_BUNDLE_VERSION = 4
REPEATABILITY_SCHEMA_VERSION = 2
REPEATABILITY_TEST_COMMAND_VERSION = 3
MACHINE_IDENTITY_VERSION = 1
PHYSICAL_EVIDENCE_VERSION = 4
MANAGED_RESOURCE_WARNINGS = (
    "resource_tracker",
    "leaked semaphore",
    "leaked shared_memory",
    "Task was destroyed but it is pending",
    "Unclosed client session",
    "unclosed transport",
    "unclosed event loop",
)

# These tests audit retained Experiment 007 artifacts supplied through opt-in
# environment variables. They are not cluster product tests and running the
# experiment is explicitly outside the productization acceptance contract.
# Keep the exclusions visible in every retained pytest command instead of
# allowing those source-audit tests to appear as implicit skips.
NON_PRODUCT_SOURCE_AUDIT_TESTS = (
    "tests/integration/test_experiment_007_corrections_run.py",
    "tests/integration/test_experiment_007_run.py",
)
NON_GPU_PRODUCT_TEST_ARGUMENTS = (
    "tests/integration",
    "tests/failure",
    "-m",
    "not gpu",
    *(f"--ignore={path}" for path in NON_PRODUCT_SOURCE_AUDIT_TESTS),
)


class AcceptanceStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    NOT_RUN = "NOT_RUN"


class OverallStatus(StrEnum):
    SOFTWARE_ACCEPTANCE_PASS = "SOFTWARE_ACCEPTANCE_PASS"
    REAL_MODEL_ACCEPTANCE_PASS = "REAL_MODEL_ACCEPTANCE_PASS"
    PHYSICAL_ACCEPTANCE_PASS = "PHYSICAL_ACCEPTANCE_PASS"
    INCOMPLETE = "INCOMPLETE"
    FAIL = "FAIL"


GateCategory = Literal["software", "real_model", "physical"]


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    category: GateCategory
    status: AcceptanceStatus
    command: list[str]
    reason: str
    duration_s: float = 0.0
    exit_code: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    resource_warnings: tuple[str, ...] = ()
    graceful_shutdown_count: int = 0
    unexpected_terminate_count: int = 0
    unexpected_kill_count: int = 0
    leaked_process_count: int = 0
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0


@dataclass(frozen=True, slots=True)
class GateSpec:
    name: str
    tests: tuple[str, ...]
    timeout_s: float


SOFTWARE_GATES = (
    GateSpec(
        "pairing_json_contract",
        ("tests/unit/test_pairing_delivery.py::test_pairing_commands_emit_one_json_document",),
        120,
    ),
    GateSpec(
        "invitation_secret_file_protection",
        ("tests/unit/test_pairing_delivery.py::test_protected_invitation_file_contract",),
        120,
    ),
    GateSpec(
        "no_secret_in_machine_output",
        (
            "tests/unit/test_pairing_delivery.py::test_pairing_secret_is_absent_from_machine_channels",
        ),
        120,
    ),
    GateSpec(
        "platform_status_separation",
        ("tests/unit/test_platform_validation_evidence.py",),
        120,
    ),
    GateSpec(
        "firewall_ownership_isolation",
        ("tests/unit/test_firewall_ownership.py",),
        120,
    ),
    GateSpec(
        "unpaired_wheel_installer_success",
        ("tests/unit/test_installer_contract.py",),
        120,
    ),
    GateSpec(
        "confirmation_semantics",
        ("tests/unit/test_command_confirmations.py",),
        120,
    ),
    GateSpec(
        "recursive_source_dependency_validation",
        ("tests/unit/test_colibri_source_contract.py",),
        120,
    ),
    GateSpec(
        "clean_wheel_import_isolation",
        ("tests/integration/test_wheel_install.py",),
        1800,
    ),
    GateSpec(
        "physical_two_machine_configuration_readiness",
        (
            "tests/unit/test_productization_acceptance.py::test_physical_configuration_readiness_contract",
        ),
        120,
    ),
    GateSpec(
        "static_architecture",
        ("tests/unit/test_architecture_boundaries.py",),
        120,
    ),
    GateSpec(
        "configuration_and_documentation",
        (
            "tests/unit/test_product_cli.py",
            "tests/unit/test_documented_product_commands.py",
        ),
        120,
    ),
    GateSpec(
        "cluster_lifecycle_reuse",
        (
            "tests/unit/test_coordinator_runtime.py",
            "tests/unit/test_worker_runtime.py",
        ),
        120,
    ),
    GateSpec(
        "cluster_pairing_and_replay_resistance",
        (
            "tests/unit/test_pairing_protocol.py",
            "tests/integration/test_cluster_pair_and_join.py",
            "tests/integration/test_cluster_revocation.py",
        ),
        180,
    ),
    GateSpec(
        "node_agent_process_restart",
        ("tests/integration/test_cluster_agent_restart.py",),
        180,
    ),
    GateSpec(
        "n_stage_planning_and_ring",
        (
            "tests/unit/test_n_stage_planner.py",
            "tests/integration/test_cluster_n_stage_ring.py",
        ),
        240,
    ),
    GateSpec(
        "directed_network_evidence",
        ("tests/unit/test_network_measurements.py",),
        120,
    ),
    GateSpec(
        "cluster_artifact_integrity",
        (
            "tests/unit/test_artifact_manager.py",
            "tests/unit/test_stage_artifact_builder.py",
            "tests/integration/test_cluster_artifact_transfer.py",
        ),
        240,
    ),
    GateSpec(
        "high_level_cluster_orchestration",
        (
            "tests/unit/test_cluster_cli.py",
            "tests/unit/test_node_cli.py",
            "tests/unit/test_run_cli.py",
            "tests/integration/test_cluster_auto_run.py",
        ),
        240,
    ),
    GateSpec(
        "wheel_installation",
        ("tests/integration/test_wheel_install.py",),
        1800,
    ),
    GateSpec(
        "cross_platform_status_evidence",
        (
            "tests/unit/test_platform_adapters.py",
            "tests/unit/test_service_manager.py",
            "tests/unit/test_ci_platform_matrix.py",
        ),
        180,
    ),
    GateSpec(
        "identity_and_trust_store",
        (
            "tests/unit/test_identity_bootstrap.py",
            "tests/unit/test_product_durable_state.py::test_product_worker_registration_requires_configured_identity_trust",
        ),
        120,
    ),
    GateSpec(
        "transport_failure_semantics",
        (
            "tests/unit/test_stage_ring_service_transport.py",
            "tests/unit/test_persistent_stage_runtime.py::test_old_route_and_request_generation_frames_are_rejected_after_recovery",
        ),
        120,
    ),
    GateSpec("process_harness", ("tests/unit/test_process_harness.py",), 120),
    GateSpec(
        "secure_process_bootstrap",
        (
            "tests/integration/test_product_stage_ring.py::test_secure_identity_bootstrap_registration_route_and_revocation",
        ),
        180,
    ),
    GateSpec(
        "persistent_lifecycle_and_concurrency",
        (
            "tests/integration/test_product_stage_ring.py::test_two_process_product_ring_persists_streams_and_never_relays_activations",
        ),
        240,
    ),
    GateSpec(
        "authenticated_routes_and_peers",
        (
            "tests/integration/test_product_stage_ring.py::test_product_route_lease_and_direct_peer_handshake_are_authenticated",
        ),
        180,
    ),
    GateSpec(
        "deterministic_socket_recovery",
        (
            "tests/integration/test_product_stage_ring.py::test_deterministic_active_socket_closure_recovers_without_duplicate_tokens",
        ),
        180,
    ),
    GateSpec(
        "worker_process_recovery",
        (
            "tests/integration/test_product_stage_ring.py::test_three_worker_restart_and_replay_replaces_failed_stage_without_duplicates",
        ),
        180,
    ),
    GateSpec(
        "replay_divergence",
        (
            "tests/integration/test_product_stage_ring.py::test_restart_and_replay_divergence_fails_before_any_duplicate_token_event",
        ),
        180,
    ),
    GateSpec(
        "cancellation_cleanup",
        (
            "tests/integration/test_product_stage_ring.py::test_product_cancel_during_prefill_and_decode_releases_kv_but_keeps_stages_loaded",
            "tests/integration/test_product_stage_ring.py::test_product_cancel_during_recovery_uses_the_same_bounded_cleanup_path",
        ),
        240,
    ),
    GateSpec(
        "complete_non_gpu_process_suite",
        NON_GPU_PRODUCT_TEST_ARGUMENTS,
        600,
    ),
)

REAL_MODEL_GATES = (
    GateSpec(
        "real_model_baseline",
        (
            "tests/integration/test_universal_real_model_acceptance.py::test_real_model_baseline_evidence",
        ),
        900,
    ),
    GateSpec(
        "real_model_restart_and_replay",
        (
            "tests/integration/test_universal_real_model_acceptance.py::test_real_model_restart_and_replay_evidence",
        ),
        900,
    ),
    GateSpec(
        "real_model_whole_expert",
        (
            "tests/integration/test_universal_real_model_acceptance.py::test_real_model_whole_expert_evidence",
        ),
        900,
    ),
    GateSpec(
        "real_model_native_microshard",
        (
            "tests/integration/test_universal_real_model_acceptance.py::test_real_model_native_microshard_evidence",
        ),
        900,
    ),
)

REAL_MODEL_EVIDENCE_IDS = {
    "real_model_baseline": "real-model-baseline",
    "real_model_restart_and_replay": "real-model-restart-and-replay",
    "real_model_whole_expert": "real-model-whole-expert",
    "real_model_native_microshard": "real-model-native-microshard",
}

PHYSICAL_GATES = {
    "physical_two_machine": "Windows RTX 5090 coordinator/CUDA worker plus Windows CPU node",
    "physical_linux_x86_64_cpu": "Linux x86-64 CPU cluster",
    "physical_macos_arm64_mps": "macOS ARM64 MPS cluster",
    "physical_linux_arm64_cpu": "Linux ARM64 CPU cluster",
}


def aggregate_status(results: list[GateResult]) -> OverallStatus:
    software = [result for result in results if result.category == "software"]
    real = [result for result in results if result.category == "real_model"]
    physical = [result for result in results if result.category == "physical"]
    if any(result.status == AcceptanceStatus.FAIL for result in results):
        return OverallStatus.FAIL
    required_software = {spec.name for spec in SOFTWARE_GATES} | {"process_repeatability"}
    passed_software = {result.name for result in software if result.status == AcceptanceStatus.PASS}
    if not required_software <= passed_software or any(
        result.status != AcceptanceStatus.PASS for result in software
    ):
        return OverallStatus.INCOMPLETE
    required_real = {spec.name for spec in REAL_MODEL_GATES}
    real_pass = required_real <= {
        result.name for result in real if result.status == AcceptanceStatus.PASS
    }
    physical_pass = any(
        result.name == "physical_two_machine" and result.status == AcceptanceStatus.PASS
        for result in physical
    )
    if physical_pass and real_pass:
        return OverallStatus.PHYSICAL_ACCEPTANCE_PASS
    if real_pass:
        return OverallStatus.REAL_MODEL_ACCEPTANCE_PASS
    return OverallStatus.SOFTWARE_ACCEPTANCE_PASS


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256_identity(value: object) -> bool:
    text = str(value)
    digest = text.removeprefix("sha256:")
    return (
        text.startswith("sha256:")
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _bounded_tree_sha256(root: Path, *, maximum_files: int = 10_000) -> str | None:
    if not root.is_dir():
        return None
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if len(files) > maximum_files:
        raise ValueError(f"package hash input exceeds {maximum_files} files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def machine_identity() -> dict[str, Any]:
    node = platform.node().strip().lower()
    boot_marker = ""
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    if boot_path.is_file():
        try:
            boot_marker = boot_path.read_text(encoding="utf-8").strip()
        except OSError:
            boot_marker = ""
    hardware = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
    }
    stable_payload = json.dumps({"node": node, **hardware}, sort_keys=True).encode()
    namespace_payload = f"{node}|{boot_marker or platform.system()}".encode()
    return {
        "format_version": MACHINE_IDENTITY_VERSION,
        "machine_identity": f"sha256:{_sha256_bytes(stable_payload)}",
        "host_identity": f"sha256:{_sha256_bytes(node.encode())}",
        "process_namespace_identity": f"sha256:{_sha256_bytes(namespace_payload)}",
        "hardware": hardware,
        "captured_at": _utc_now(),
    }


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().strip("[]")
    if normalized in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_physical_configuration(configuration: dict[str, Any]) -> list[str]:
    required = {
        "coordinator_host",
        "coordinator_endpoint",
        "worker_a_host",
        "worker_b_host",
        "worker_a_identity_path",
        "worker_b_identity_path",
        "worker_a_machine_identity",
        "worker_b_machine_identity",
        "model_revision",
        "evidence_output_directory",
    }
    errors = [
        f"missing required physical field {name}" for name in sorted(required - set(configuration))
    ]
    if errors:
        return errors
    hosts = [
        str(configuration["coordinator_host"]),
        str(configuration["worker_a_host"]),
        str(configuration["worker_b_host"]),
    ]
    for host in hosts:
        if _is_loopback_host(host):
            errors.append(f"loopback host cannot provide physical evidence: {host}")
    resolved_hosts: list[set[str]] = []
    for host in hosts:
        try:
            resolved_hosts.append(
                {str(item[4][0]).split("%", 1)[0] for item in socket.getaddrinfo(host, None)}
            )
        except socket.gaierror as exc:
            errors.append(f"physical host does not resolve: {host}: {exc}")
            resolved_hosts.append(set())
    for host, addresses in zip(hosts, resolved_hosts, strict=True):
        if any(ipaddress.ip_address(address).is_loopback for address in addresses):
            errors.append(f"physical host resolves to loopback: {host}")
    if hosts[1].strip().lower() == hosts[2].strip().lower():
        errors.append("physical workers resolve to the same declared host")
    if resolved_hosts[1] and resolved_hosts[1] & resolved_hosts[2]:
        errors.append("physical workers resolve to the same network address")
    identities: list[dict[str, Any]] = []
    for key in ("worker_a_machine_identity", "worker_b_machine_identity"):
        value = configuration[key]
        if isinstance(value, str):
            path = Path(value).expanduser().resolve()
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"cannot read {key}: {exc}")
                value = {}
        if not isinstance(value, dict):
            errors.append(f"{key} must be an object or JSON path")
            value = {}
        identities.append(value)
    if all(identities):
        if identities[0].get("machine_identity") == identities[1].get("machine_identity"):
            errors.append("physical workers report the same machine identity")
        if identities[0].get("host_identity") == identities[1].get("host_identity"):
            errors.append("physical workers report the same host identity")
        if identities[0].get("process_namespace_identity") == identities[1].get(
            "process_namespace_identity"
        ):
            errors.append("physical workers report the same process namespace")
    return errors


def _identity_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(Path(value).expanduser().resolve().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _command_text(command: object) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        return " ".join(command)
    return ""


def validate_physical_evidence(
    configuration: dict[str, Any],
    evidence_directory: Path,
) -> tuple[AcceptanceStatus, list[str]]:
    """Validate captured product evidence without treating loopback as physical."""

    summary_path = evidence_directory / "physical-evidence.json"
    if not summary_path.is_file():
        return AcceptanceStatus.NOT_RUN, [f"physical evidence was not captured: {summary_path}"]
    try:
        evidence = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return AcceptanceStatus.FAIL, [f"invalid physical evidence summary: {exc}"]
    if not isinstance(evidence, dict):
        return AcceptanceStatus.FAIL, ["physical evidence summary must be a JSON object"]

    errors: list[str] = []
    if evidence.get("document_type") != "swarm-physical-two-machine-evidence":
        errors.append("physical evidence has an unsupported document type")
    if evidence.get("format_version") != PHYSICAL_EVIDENCE_VERSION:
        errors.append("physical evidence has an unsupported format version")
    if evidence.get("model_revision") != configuration.get("model_revision"):
        errors.append("physical evidence model revision does not match the requested revision")

    configured_identities = {
        "worker_a": _identity_value(configuration.get("worker_a_machine_identity")),
        "worker_b": _identity_value(configuration.get("worker_b_machine_identity")),
    }
    reported_identities = evidence.get("machine_identities")
    if not isinstance(reported_identities, dict):
        errors.append("physical evidence must contain machine_identities")
        reported_identities = {}
    for role in ("worker_a", "worker_b"):
        configured = configured_identities[role].get("machine_identity")
        reported = _identity_value(reported_identities.get(role)).get("machine_identity")
        if not configured or reported != configured:
            errors.append(f"{role} machine identity does not match the physical configuration")

    workers = evidence.get("workers")
    if not isinstance(workers, list) or len(workers) < 2:
        errors.append("physical evidence must contain at least two worker status records")
        workers = []
    worker_ids = {
        str(worker.get("worker_id"))
        for worker in workers
        if isinstance(worker, dict) and worker.get("worker_id")
    }
    worker_machine_ids = {
        str(worker.get("machine_identity"))
        for worker in workers
        if isinstance(worker, dict) and worker.get("machine_identity")
    }
    if len(worker_ids) < 2 or len(worker_machine_ids) < 2:
        errors.append("worker status must prove distinct worker and machine identities")
    reported_worker_identities = evidence.get("worker_identities")
    if not isinstance(reported_worker_identities, dict):
        errors.append("physical evidence must contain worker_identities")
        reported_worker_identities = {}
    fingerprints = {
        str(item.get("fingerprint"))
        for item in reported_worker_identities.values()
        if isinstance(item, dict) and item.get("fingerprint")
    }
    if len(fingerprints) < 2:
        errors.append("physical evidence must prove two distinct worker public fingerprints")
    for worker in workers:
        if not isinstance(worker, dict):
            errors.append("worker status records must be objects")
            continue
        for key in ("control_endpoint", "data_plane_endpoint"):
            endpoint = str(worker.get(key, ""))
            host = endpoint.rsplit(":", 1)[0].strip("[]") if ":" in endpoint else endpoint
            if not host or _is_loopback_host(host):
                errors.append(f"worker {key} is absent or loopback: {endpoint!r}")
        fingerprint = str(worker.get("public_key_fingerprint", ""))
        if fingerprint not in fingerprints:
            errors.append("worker status public fingerprint is absent from worker_identities")

    topology = evidence.get("topology")
    assignments = topology.get("assignments") if isinstance(topology, dict) else None
    assigned_workers = {
        str(item.get("worker_id"))
        for item in assignments or []
        if isinstance(item, dict) and item.get("worker_id")
    }
    if not isinstance(assignments, list) or len(assignments) < 2:
        errors.append("physical topology must contain at least two stage assignments")
    elif len(assigned_workers & worker_ids) < 2:
        errors.append("physical topology does not assign stages to both reported workers")

    normal = evidence.get("normal_run")
    if not isinstance(normal, dict):
        errors.append("physical evidence must contain normal_run")
    else:
        expected = normal.get("expected_token_ids")
        actual = normal.get("token_ids")
        if (
            normal.get("status") != "completed"
            or not isinstance(expected, list)
            or actual != expected
        ):
            errors.append(
                "normal physical inference did not complete with the exact expected tokens"
            )

    recovery = evidence.get("recovery_run")
    if not isinstance(recovery, dict):
        errors.append("physical evidence must contain recovery_run")
    else:
        expected = recovery.get("expected_token_ids")
        events = recovery.get("token_events")
        token_ids = (
            [event.get("token_id") for event in events if isinstance(event, dict)]
            if isinstance(events, list)
            else []
        )
        positions = (
            [event.get("token_position") for event in events if isinstance(event, dict)]
            if isinstance(events, list)
            else []
        )
        if (
            recovery.get("status") != "completed"
            or not isinstance(expected, list)
            or token_ids != expected
        ):
            errors.append("physical recovery did not complete with the exact expected tokens")
        if positions != list(range(len(positions))) or len(positions) != len(set(positions)):
            errors.append("physical recovery token events contain gaps or duplicates")
        recovery_events = recovery.get("recovery_events")
        completed_recoveries = [
            item
            for item in recovery_events or []
            if isinstance(item, dict)
            and item.get("event") in {"recovery_completed", "RECOVERY_COMPLETED"}
        ]
        if len(completed_recoveries) != 1:
            errors.append("physical evidence must contain exactly one completed recovery")
        generations = recovery.get("route_generations")
        if (
            not isinstance(generations, list)
            or len(generations) < 2
            or any(not isinstance(item, int) for item in generations)
            or any(current >= following for current, following in pairwise(generations))
        ):
            errors.append("physical recovery must prove an increasing route generation")

    release = evidence.get("release")
    if not isinstance(release, dict):
        errors.append("physical evidence must identify the GitHub Release installer")
        release = {}
    release_tag = str(release.get("git_tag", ""))
    release_url = str(release.get("url", ""))
    installer_filename = str(release.get("installer_filename", ""))
    release_manifest_hash = release.get("release_manifest_sha256")
    signature_status = release.get("authenticode_status")
    if not release_tag.startswith("v"):
        errors.append("physical release evidence must record the immutable Git tag")
    if not release_url.startswith(
        f"https://github.com/mmprotest/swarm-inference-lab/releases/tag/{release_tag}"
    ):
        errors.append("physical release URL must identify the fixed GitHub repository and tag")
    if installer_filename != "SwarmInferenceSetup-x64.exe":
        errors.append("physical nodes must use SwarmInferenceSetup-x64.exe")
    if not _is_sha256_identity(release_manifest_hash):
        errors.append("physical release evidence must record the release manifest SHA-256")
    if signature_status not in {"signed-valid", "unsigned-prerelease"}:
        errors.append("physical release evidence must record explicit Authenticode status")
    if "-rc." not in release_tag and signature_status != "signed-valid":
        errors.append("stable physical release evidence requires valid Authenticode")

    installations = evidence.get("installations")
    if not isinstance(installations, list) or len(installations) < 2:
        errors.append("physical evidence must contain native installer evidence for both nodes")
        installations = []
    installer_hashes: set[str] = set()
    selected_profiles: set[str] = set()
    for installation in installations:
        if not isinstance(installation, dict):
            errors.append("physical installation records must be objects")
            continue
        if installation.get("status") != "PASS":
            errors.append("each physical node must pass native setup installation")
        if installation.get("source") != "github-release-installer":
            errors.append("physical nodes must install independently from the GitHub Release")
        if installation.get("repository_cloned") is not False:
            errors.append("physical installation must prove that no repository clone was used")
        if installation.get("installer_filename") != installer_filename:
            errors.append("physical installation filename must match the release installer")
        installer_hash = str(installation.get("installer_sha256", ""))
        if not _is_sha256_identity(installer_hash):
            errors.append("physical installation must record the installer SHA-256")
        else:
            installer_hashes.add(installer_hash)
        if installation.get("release_manifest_sha256") != release_manifest_hash:
            errors.append("physical installation manifest hash must match the GitHub Release")
        if installation.get("authenticode_status") != signature_status:
            errors.append("physical installation Authenticode status must match the release")
        profile = str(installation.get("selected_profile", ""))
        if profile not in {"cpu", "cuda"}:
            errors.append("physical installation must record the selected CPU or CUDA profile")
        else:
            selected_profiles.add(profile)
        product_version = installation.get("product_version")
        record = installation.get("installation_record")
        if not isinstance(record, dict):
            errors.append("physical installation must retain its strict installation record")
            continue
        if (
            record.get("installation_mode") != "native-windows"
            or record.get("product_version") != product_version
            or record.get("selected_backend") != profile
            or record.get("release_manifest_sha256") != release_manifest_hash
        ):
            errors.append("physical installation record does not match its release evidence")
    if installations and len(installer_hashes) != 1:
        errors.append("both physical machines must use the same installer SHA-256")
    if installations and selected_profiles != {"cpu", "cuda"}:
        errors.append("physical gate requires one CUDA profile and one CPU profile")

    pairing = evidence.get("pairing")
    if not isinstance(pairing, dict):
        errors.append("physical evidence must contain pairing evidence")
    elif (
        pairing.get("status") != "consumed"
        or pairing.get("single_use") is not True
        or pairing.get("fingerprint_copied_manually") is not False
    ):
        errors.append(
            "physical pairing must be consumed, single-use, and require no fingerprint copy"
        )

    automation = evidence.get("automatic_configuration")
    required_automation = {
        "backend_selected",
        "memory_selected",
        "control_endpoint_selected",
        "data_endpoint_selected",
        "ports_selected",
    }
    if not isinstance(automation, dict) or any(
        automation.get(name) is not True for name in required_automation
    ):
        errors.append(
            "physical evidence must prove automatic backend, memory, endpoint, and port selection"
        )

    services = evidence.get("services")
    if not isinstance(services, list) or len(services) < 2:
        errors.append("physical evidence must contain persistent service state for both nodes")
    elif any(
        not isinstance(item, dict)
        or item.get("running_after_terminal_close") is not True
        or item.get("reconnected_after_restart") is not True
        for item in services
    ):
        errors.append("both physical node services must persist and reconnect after restart")

    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("physical evidence must contain automatically provisioned stage artifacts")
    elif any(
        not isinstance(item, dict)
        or item.get("verified") is not True
        or len(str(item.get("artifact_id", "")).removeprefix("sha256:")) != 64
        for item in artifacts
    ):
        errors.append("every physical stage artifact must have a verified content identity")

    links = evidence.get("directed_network_links")
    if not isinstance(links, list) or len(links) < 2:
        errors.append("physical evidence must contain both directed node-to-node measurements")
        links = []
    directions = set()
    for link in links:
        if not isinstance(link, dict):
            errors.append("directed network evidence records must be objects")
            continue
        source = str(link.get("source_worker_id", ""))
        destination = str(link.get("destination_worker_id", ""))
        directions.add((source, destination))
        endpoint = str(link.get("destination_endpoint", ""))
        host = endpoint.rsplit(":", 1)[0].strip("[]") if ":" in endpoint else endpoint
        if (
            link.get("measured") is not True
            or link.get("authentication_verified") is not True
            or not host
            or _is_loopback_host(host)
        ):
            errors.append(
                "directed physical links must be authenticated, measured, and non-loopback"
            )
    if directions and not all(
        (destination, source) in directions for source, destination in directions
    ):
        errors.append("physical network evidence must contain both link directions")

    capacity_workers: set[str] = set()
    for name in ("speed_run", "capacity_run"):
        run = evidence.get(name)
        if not isinstance(run, dict):
            errors.append(f"physical evidence must contain {name}")
            continue
        if run.get("status") != "completed" or run.get("token_ids") != run.get(
            "expected_token_ids"
        ):
            errors.append(f"{name} must complete with the exact expected token IDs")
        run_topology = run.get("topology")
        run_assignments = (
            run_topology.get("assignments") if isinstance(run_topology, dict) else None
        )
        if not isinstance(run_assignments, list) or not run_assignments:
            errors.append(f"{name} must retain its deployed topology assignments")
            run_workers: set[str] = set()
        else:
            run_workers = {
                str(item.get("worker_id"))
                for item in run_assignments
                if isinstance(item, dict) and item.get("worker_id")
            }
            if not run_workers or not run_workers <= worker_ids:
                errors.append(f"{name} topology contains an unknown or absent worker")
        if name == "speed_run":
            excluded = run.get("excluded_slow_node_id")
            if excluded is not None and str(excluded) in run_workers:
                errors.append("speed_run claims to exclude a worker that its topology assigns")
        else:
            capacity_workers = run_workers
            capacity_machine_ids = {
                str(worker.get("machine_identity"))
                for worker in workers
                if isinstance(worker, dict)
                and worker.get("worker_id") in capacity_workers
                and worker.get("machine_identity")
            }
            if len(capacity_workers) < 2 or len(capacity_machine_ids) < 2:
                errors.append(
                    "capacity_run must assign work to both distinct physical machine identities"
                )
            included = str(run.get("included_slow_node_id", ""))
            if not included or not any(
                worker_id.startswith(f"{included}/") or worker_id == included
                for worker_id in capacity_workers
            ):
                errors.append("capacity_run does not assign the required laptop node")

    direct_traffic = evidence.get("direct_stage_traffic")
    required_direct_traffic_files: set[str] = set()
    if not isinstance(direct_traffic, list) or not direct_traffic:
        errors.append("physical evidence must retain observed direct-stage traffic")
    else:
        observed_edges: set[tuple[str, str]] = set()
        for observation in direct_traffic:
            if not isinstance(observation, dict):
                errors.append("direct-stage traffic observations must be objects")
                continue
            source = str(observation.get("source_worker_id", ""))
            destination = str(observation.get("destination_worker_id", ""))
            endpoint = str(observation.get("destination_endpoint", ""))
            host = endpoint.rsplit(":", 1)[0].strip("[]") if ":" in endpoint else endpoint
            evidence_file = str(observation.get("evidence_file", ""))
            if (
                observation.get("observed") is not True
                or int(observation.get("bytes_observed", 0)) <= 0
                or source == destination
                or source not in capacity_workers
                or destination not in capacity_workers
                or not host
                or _is_loopback_host(host)
                or not evidence_file
            ):
                errors.append(
                    "direct-stage traffic must be observed between assigned distinct workers "
                    "on a non-loopback endpoint with retained evidence"
                )
                continue
            observed_edges.add((source, destination))
            required_direct_traffic_files.add(evidence_file)
        if capacity_workers and not observed_edges:
            errors.append("capacity topology has no retained cross-worker direct-stage traffic")

    commands = [_command_text(item) for item in evidence.get("commands", [])]
    required_commands = {
        "cluster create": 1,
        "node join": 1,
        "cluster status": 1,
        "run": 2,
        "--mode speed": 1,
        "--mode capacity": 1,
    }
    for fragment, minimum in required_commands.items():
        count = sum(fragment in command for command in commands)
        if count < minimum:
            errors.append(f"physical evidence is missing {minimum} '{fragment}' command(s)")
    forbidden_manual = ("swarm identity", "swarm coordinator", "swarm worker", "swarm model deploy")
    if any(fragment in command for fragment in forbidden_manual for command in commands):
        errors.append("physical product acceptance used a manual low-level provisioning command")

    source_files = evidence.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        errors.append("physical evidence must checksum its source logs and command outputs")
    else:
        missing_direct = sorted(required_direct_traffic_files - set(source_files))
        if missing_direct:
            errors.append(
                "direct-stage traffic evidence files are not checksummed: "
                + ", ".join(missing_direct)
            )
        root = evidence_directory.resolve()
        for relative, expected_hash in source_files.items():
            candidate = (root / str(relative)).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                errors.append(f"physical evidence source escapes its directory: {relative}")
                continue
            try:
                actual_hash = _sha256_bytes(candidate.read_bytes())
            except OSError as exc:
                errors.append(f"cannot read physical evidence source {relative}: {exc}")
                continue
            if expected_hash != f"sha256:{actual_hash}":
                errors.append(f"physical evidence source checksum mismatch: {relative}")

    return (AcceptanceStatus.FAIL, errors) if errors else (AcceptanceStatus.PASS, [])


def _junit_counts(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return tuple(
        sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    )  # type: ignore[return-value]


def _git_value(repository_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _canonical_payload_checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _repeatability_summary_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    return resolved / "summary.json" if resolved.is_dir() else resolved


def validate_repeatability_evidence(
    repository_root: Path,
    evidence_path: Path,
) -> tuple[AcceptanceStatus, list[str], dict[str, Any] | None]:
    """Validate repeatability provenance, completeness, warnings, and cleanup."""

    summary_path = _repeatability_summary_path(evidence_path)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return (
            AcceptanceStatus.NOT_RUN,
            [f"repeatability evidence is missing: {summary_path}"],
            None,
        )
    except (OSError, json.JSONDecodeError) as exc:
        return AcceptanceStatus.NOT_RUN, [f"repeatability evidence is unreadable: {exc}"], None
    if not isinstance(payload, dict):
        return AcceptanceStatus.NOT_RUN, ["repeatability summary is not a JSON object"], None

    stale: list[str] = []
    failures: list[str] = []
    if payload.get("document_type") != "swarm-process-repeatability":
        stale.append("repeatability document type is invalid")
    if payload.get("schema_version") != REPEATABILITY_SCHEMA_VERSION:
        stale.append("repeatability schema version is stale")
    if payload.get("acceptance_schema_version") != ACCEPTANCE_BUNDLE_VERSION:
        stale.append("repeatability acceptance schema version is stale")
    if payload.get("test_command_version") != REPEATABILITY_TEST_COMMAND_VERSION:
        stale.append("repeatability test command version is stale")

    current_status = _git_value(repository_root, "status", "--porcelain")
    current_commit = _git_value(repository_root, "rev-parse", "HEAD")
    if payload.get("git_commit") != current_commit:
        stale.append("repeatability evidence belongs to another git commit")
    if payload.get("git_status") != current_status:
        stale.append("repeatability dirty-tree state does not match the current tree")
    if payload.get("finish_git_status") != current_status:
        stale.append("repeatability tree changed while the runs were executing")
    if payload.get("git_dirty") != (current_status is None or bool(current_status)):
        stale.append("repeatability dirty-tree flag does not match the current tree")
    if payload.get("python_version") != platform.python_version():
        stale.append("repeatability Python version does not match acceptance")
    if payload.get("os") != platform.platform():
        stale.append("repeatability OS does not match acceptance")

    process_script = repository_root / "scripts/run_productization_process_suite.py"
    expected_process_hash = f"sha256:{_sha256_bytes(process_script.read_bytes())}"
    expected_acceptance_hash = f"sha256:{_sha256_bytes(Path(__file__).read_bytes())}"
    if payload.get("process_runner_sha256") != expected_process_hash:
        stale.append("repeatability process runner content hash is stale")
    if payload.get("acceptance_source_sha256") != expected_acceptance_hash:
        stale.append("repeatability acceptance source content hash is stale")

    recorded_checksum = payload.get("evidence_checksum")
    checksummed = dict(payload)
    checksummed.pop("evidence_checksum", None)
    if recorded_checksum != f"sha256:{_canonical_payload_checksum(checksummed)}":
        stale.append("repeatability summary checksum is invalid")

    required = payload.get("required_runs")
    requested = payload.get("requested_runs")
    expected_counts = {
        "full_process_suite": 3,
        "stage_ring_module": 5,
    }
    if required != expected_counts or requested != expected_counts:
        stale.append("repeatability evidence does not contain the required three plus five runs")
    if payload.get("excluded_source_audit_tests") != list(NON_PRODUCT_SOURCE_AUDIT_TESTS):
        stale.append("repeatability source-audit exclusion contract is stale")
    results = payload.get("results")
    expected_names = [f"full-{index}" for index in range(1, 4)] + [
        f"stage-ring-{index}" for index in range(1, 6)
    ]
    if (
        not isinstance(results, list)
        or not all(isinstance(item, dict) for item in results)
        or [item.get("name") for item in results] != expected_names
    ):
        stale.append("repeatability run list is incomplete or out of order")
        results = []
    evidence_root = summary_path.parent
    for result in results:
        name = str(result.get("name", "unknown"))
        status = result.get("status")
        if status in {"FAIL", "TIMEOUT", "WARNING_FAILURE"}:
            failures.append(f"{name} has status {status}")
        elif status != "PASS":
            stale.append(f"{name} has incomplete status {status}")
        if result.get("resource_warning_count") != 0 or result.get("warning_scan", {}).get(
            "matches"
        ):
            failures.append(f"{name} emitted a managed-resource warning")
        for field in (
            "unexpected_terminate_count",
            "unexpected_kill_count",
            "leaked_process_count",
        ):
            if int(result.get(field, 0)):
                failures.append(f"{name} recorded non-zero {field}")
        if result.get("exit_code") != 0:
            failures.append(f"{name} did not exit with code zero")
        test_counts = result.get("test_counts")
        if not isinstance(test_counts, dict) or int(test_counts.get("tests", 0)) <= 0:
            stale.append(f"{name} has no retained test counts")
        elif int(test_counts.get("skipped", 0)):
            stale.append(f"{name} contains skipped tests")
        checksums = result.get("checksums")
        if not isinstance(checksums, dict) or not checksums:
            stale.append(f"{name} has no retained log checksums")
            continue
        for relative, expected in checksums.items():
            candidate = evidence_root / str(relative)
            try:
                actual = f"sha256:{_sha256_bytes(candidate.read_bytes())}"
            except OSError:
                stale.append(f"{name} retained evidence is missing: {relative}")
                continue
            if actual != expected:
                stale.append(f"{name} retained evidence checksum mismatch: {relative}")
    if payload.get("overall_repeatability_status") == "FAIL":
        failures.append("repeatability overall status is FAIL")
    elif payload.get("overall_repeatability_status") != "PASS":
        stale.append("repeatability overall status is incomplete")

    if failures:
        return AcceptanceStatus.FAIL, [*stale, *failures], payload
    if stale:
        return AcceptanceStatus.NOT_RUN, stale, payload
    return AcceptanceStatus.PASS, [], payload


def environment_evidence(repository_root: Path) -> dict[str, Any]:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if "Name" in distribution.metadata
    }
    cuda: dict[str, Any] = {"available": False, "devices": []}
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    # PyTorch can be installed yet unusable when a native runtime dependency is
    # missing or its DLL/shared object is inaccessible.  Environment capture is
    # evidence collection, so retain that exact failure instead of crashing the
    # entire acceptance bundle before its gates can be reported.
    except (ImportError, RuntimeError, OSError) as exc:
        cuda["error"] = f"{type(exc).__name__}: {exc}"
    status = _git_value(repository_root, "status", "--porcelain")
    wheels = [
        {
            "path": str(path.relative_to(repository_root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_bytes(path.read_bytes()),
        }
        for path in sorted((repository_root / "dist").glob("*.whl"))
    ]
    lock_path = repository_root / "uv.lock"
    return {
        "captured_at": _utc_now(),
        "git_commit": _git_value(repository_root, "rev-parse", "HEAD"),
        "git_dirty": status is None or bool(status),
        "git_status": status,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "os": platform.platform(),
        "machine": machine_identity(),
        "cuda": cuda,
        "packages": dict(sorted(packages.items())),
        "package_hashes": {
            "swarm_inference_source_tree_sha256": _bounded_tree_sha256(
                repository_root / "src" / "swarm_inference"
            ),
            "uv_lock_sha256": (
                _sha256_bytes(lock_path.read_bytes()) if lock_path.is_file() else None
            ),
        },
        "wheel_hashes": wheels,
    }


def _terminate_subprocess_tree(process: subprocess.Popen[str]) -> tuple[int, int, int]:
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        descendants = []
    terminate_count = 0
    kill_count = 0
    for child in reversed(descendants):
        with suppress(psutil.Error):
            child.terminate()
            terminate_count += 1
    if process.poll() is None:
        process.terminate()
        terminate_count += 1
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        kill_count += 1
        process.wait(timeout=5)
    _, alive = psutil.wait_procs(descendants, timeout=5)
    for child in alive:
        with suppress(psutil.Error):
            child.kill()
            kill_count += 1
    _, survivors = psutil.wait_procs(alive, timeout=5)
    return terminate_count, kill_count, len(survivors)


def _process_lifecycle_counts(path: Path) -> dict[str, int]:
    totals = {
        "graceful_shutdown_count": 0,
        "unexpected_terminate_count": 0,
        "unexpected_kill_count": 0,
        "leaked_process_count": 0,
    }
    paths = ([path] if path.is_file() else []) + sorted(
        path.parent.glob(f"{path.name}.worker-*.json")
    )
    for lifecycle_path in paths:
        for line in lifecycle_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            for field in totals:
                totals[field] += int(payload.get(field, 0))
    return totals


def _validate_real_gate_payload(gate: str, payload: dict[str, Any]) -> list[str]:
    expected_gate = REAL_MODEL_EVIDENCE_IDS[gate]
    errors: list[str] = []
    if payload.get("gate") != expected_gate:
        errors.append(f"real-model evidence gate must be {expected_gate!r}")
    required = (
        "executed",
        "model_id",
        "model_revision",
        "model_fingerprint",
        "architecture_profile",
        "artifact_format",
        "engine_id",
        "runtime_fingerprint",
        "execution_started_at",
        "execution_finished_at",
        "tokenizer_revision",
        "model_metadata_hash",
        "prompt",
        "prompt_token_ids",
        "generated_token_ids",
        "expected_token_ids",
        "worker_identities",
        "worker_pids",
        "stage_assignments",
        "route_generations",
        "bytes_transferred",
        "critical_path_timings",
        "fallback_count",
        "recovery_events",
        "git_commit",
        "git_dirty",
        "environment",
        "status",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        errors.append("real-model evidence is missing fields: " + ", ".join(missing))
    if payload.get("status") != "PASS":
        errors.append("real-model evidence does not explicitly record PASS")
    if payload.get("executed") is not True:
        errors.append("real-model evidence does not prove an executed run")
    if not str(payload.get("model_id", "")).strip():
        errors.append("real-model evidence has no model ID")
    if not str(payload.get("model_revision", "")).strip():
        errors.append("real-model evidence has no immutable revision")
    profile = payload.get("architecture_profile")
    if not isinstance(profile, dict) or not str(profile.get("architecture_id", "")).strip():
        errors.append("real-model evidence has no inspected architecture profile")
    if not _is_sha256_identity(payload.get("model_fingerprint")):
        errors.append("real-model evidence model fingerprint is not a SHA-256 identity")
    if not _is_sha256_identity(payload.get("runtime_fingerprint")):
        errors.append("real-model evidence runtime fingerprint is not a SHA-256 identity")
    if payload.get("generated_token_ids") != payload.get("expected_token_ids"):
        errors.append("real-model evidence generated token IDs do not match expected IDs")
    if gate == "real_model_baseline" and payload.get("recovery_events"):
        errors.append("baseline evidence contains a recovery event")
    if gate == "real_model_restart_and_replay" and not payload.get("recovery_events"):
        errors.append("recovery evidence contains no recovery event")
    if (
        gate in {"real_model_whole_expert", "real_model_native_microshard"}
        and int(payload.get("fallback_count", -1)) != 0
    ):
        errors.append("forced-remote real-model evidence recorded a fallback")
    return errors


class ProductizationAcceptanceRunner:
    def __init__(self, *, repository_root: Path, output_root: Path) -> None:
        self.repository_root = repository_root.expanduser().resolve()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.bundle = output_root.expanduser().resolve() / f"productization-{timestamp}"
        self.logs = self.bundle / "logs"
        self.temp = self.bundle / "temp"
        self.logs.mkdir(parents=True, exist_ok=False)
        self.temp.mkdir(parents=True)
        self.results: list[GateResult] = []
        self.physical_evidence: dict[str, Any] | None = None
        self.model_evidence_records: list[dict[str, Any]] = []
        self.repeatability_evidence: dict[str, Any] | None = None
        self.repeatability_path: Path | None = None

    def _run_pytest(self, spec: GateSpec, *, category: GateCategory) -> GateResult:
        junit_path = self.temp / f"{spec.name}.junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            *spec.tests,
            "-q",
            "--basetemp",
            str(self.temp / spec.name),
            "--junitxml",
            str(junit_path),
        ]
        environment = dict(os.environ)
        environment["TEMP"] = str(self.temp)
        environment["TMP"] = str(self.temp)
        gate_evidence = self.bundle / "gate-evidence" / spec.name
        gate_evidence.mkdir(parents=True, exist_ok=True)
        environment["SWARM_ACCEPTANCE_GATE_EVIDENCE"] = str(gate_evidence)
        lifecycle_path = self.logs / f"{spec.name}.lifecycle.jsonl"
        environment["SWARM_PROCESS_LIFECYCLE_LOG"] = str(lifecycle_path)
        started = time.monotonic()
        external_terminate_count = 0
        external_kill_count = 0
        external_leak_count = 0
        process = subprocess.Popen(
            command,
            cwd=self.repository_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=spec.timeout_s)
            status = AcceptanceStatus.PASS if process.returncode == 0 else AcceptanceStatus.FAIL
            reason = "pytest command passed" if status == AcceptanceStatus.PASS else "pytest failed"
            tests = failures = errors = skipped = 0
            try:
                tests, failures, errors, skipped = _junit_counts(junit_path)
            except (OSError, ET.ParseError, ValueError) as exc:
                status = AcceptanceStatus.FAIL
                reason = f"cannot verify test execution: {exc}"
            else:
                if tests == 0:
                    status = AcceptanceStatus.FAIL
                    reason = "gate selected no tests"
            if process.returncode == 0:
                if skipped:
                    status = AcceptanceStatus.SKIP
                    reason = f"{skipped} of {tests} required tests skipped"
                elif failures or errors:
                    status = AcceptanceStatus.FAIL
                    reason = f"JUnit reported {failures} failures and {errors} errors"
            exit_code: int | None = process.returncode
        except subprocess.TimeoutExpired:
            (
                external_terminate_count,
                external_kill_count,
                external_leak_count,
            ) = _terminate_subprocess_tree(process)
            stdout, stderr = process.communicate()
            status = AcceptanceStatus.FAIL
            reason = f"external timeout after {spec.timeout_s:.0f}s"
            exit_code = None
            tests = failures = errors = skipped = 0
        duration = time.monotonic() - started
        stdout_path = self.logs / f"{spec.name}.stdout.log"
        stderr_path = self.logs / f"{spec.name}.stderr.log"
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")

        warning_matches = tuple(
            fragment
            for fragment in MANAGED_RESOURCE_WARNINGS
            if fragment.lower() in f"{stdout}\n{stderr}".lower()
        )
        try:
            lifecycle = _process_lifecycle_counts(lifecycle_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            lifecycle = {
                "graceful_shutdown_count": 0,
                "unexpected_terminate_count": 0,
                "unexpected_kill_count": 0,
                "leaked_process_count": 0,
            }
            status = AcceptanceStatus.FAIL
            reason = f"process lifecycle telemetry is invalid: {exc}"
        lifecycle["unexpected_terminate_count"] += external_terminate_count
        lifecycle["unexpected_kill_count"] += external_kill_count
        lifecycle["leaked_process_count"] += external_leak_count
        if warning_matches:
            status = AcceptanceStatus.FAIL
            reason = "managed-resource warning signature was emitted"
        if (
            lifecycle["unexpected_terminate_count"]
            or lifecycle["unexpected_kill_count"]
            or lifecycle["leaked_process_count"]
        ):
            status = AcceptanceStatus.FAIL
            reason = "managed process cleanup required force or leaked a child"

        evidence_records: list[dict[str, Any]] = []
        for evidence_path in sorted(gate_evidence.glob("*.json")):
            try:
                payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                evidence_records.append(
                    {
                        "gate": spec.name,
                        "path": str(evidence_path.relative_to(self.bundle)),
                        "evidence": payload,
                    }
                )
        if category == "real_model" and status == AcceptanceStatus.PASS:
            if len(evidence_records) != 1:
                status = AcceptanceStatus.FAIL
                reason = "real-model gate did not produce exactly one evidence record"
            else:
                evidence_errors = _validate_real_gate_payload(
                    spec.name,
                    evidence_records[0]["evidence"],
                )
                if evidence_errors:
                    status = AcceptanceStatus.FAIL
                    reason = "; ".join(evidence_errors)

        result = GateResult(
            name=spec.name,
            category=category,
            status=status,
            command=command,
            reason=reason,
            duration_s=duration,
            exit_code=exit_code,
            stdout_log=str(stdout_path.relative_to(self.bundle)),
            stderr_log=str(stderr_path.relative_to(self.bundle)),
            resource_warnings=warning_matches,
            graceful_shutdown_count=lifecycle["graceful_shutdown_count"],
            unexpected_terminate_count=lifecycle["unexpected_terminate_count"],
            unexpected_kill_count=lifecycle["unexpected_kill_count"],
            leaked_process_count=lifecycle["leaked_process_count"],
            tests=tests,
            failures=failures,
            errors=errors,
            skipped=skipped,
        )
        self.results.append(result)
        self.model_evidence_records.extend(evidence_records)
        return result

    def run_software(self) -> None:
        for spec in SOFTWARE_GATES:
            self._run_pytest(spec, category="software")

    def record_repeatability_not_run(self, reason: str) -> None:
        self.results.append(
            GateResult(
                name="process_repeatability",
                category="software",
                status=AcceptanceStatus.NOT_RUN,
                command=[],
                reason=reason,
            )
        )

    def consume_repeatability(
        self,
        evidence_path: Path,
        *,
        command: list[str] | None = None,
        stdout_log: str | None = None,
        stderr_log: str | None = None,
        duration_s: float = 0.0,
        exit_code: int | None = None,
    ) -> GateResult:
        status, errors, payload = validate_repeatability_evidence(
            self.repository_root,
            evidence_path,
        )
        summary_path = _repeatability_summary_path(evidence_path)
        if summary_path.is_file():
            source = summary_path.parent
            try:
                source.relative_to(self.bundle)
            except ValueError:
                retained = self.bundle / "repeatability-evidence"
                if retained.exists():
                    shutil.rmtree(retained)
                shutil.copytree(source, retained)
                summary_path = retained / "summary.json"
        self.repeatability_path = summary_path if summary_path.is_file() else None
        self.repeatability_evidence = payload
        result = GateResult(
            name="process_repeatability",
            category="software",
            status=status,
            command=command or [],
            reason=(
                "three full process-suite runs and five stage-ring runs passed"
                if status == AcceptanceStatus.PASS
                else "; ".join(errors)
            ),
            duration_s=duration_s,
            exit_code=exit_code,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            resource_warnings=tuple(
                sorted(
                    {
                        warning
                        for item in (payload or {}).get("results", [])
                        for warning in item.get("warning_scan", {}).get("matches", [])
                    }
                )
            ),
            graceful_shutdown_count=sum(
                int(item.get("graceful_shutdown_count", 0))
                for item in (payload or {}).get("results", [])
            ),
            unexpected_terminate_count=sum(
                int(item.get("unexpected_terminate_count", 0))
                for item in (payload or {}).get("results", [])
            ),
            unexpected_kill_count=sum(
                int(item.get("unexpected_kill_count", 0))
                for item in (payload or {}).get("results", [])
            ),
            leaked_process_count=sum(
                int(item.get("leaked_process_count", 0))
                for item in (payload or {}).get("results", [])
            ),
        )
        self.results.append(result)
        return result

    def run_gate(self, spec: GateSpec, *, category: GateCategory = "software") -> GateResult:
        """Run one gate with the canonical isolation, evidence, and cleanup checks."""

        return self._run_pytest(spec, category=category)

    def run_repeatability(
        self,
        *,
        timeout_s: float,
        full_runs: int = 3,
        stage_runs: int = 5,
    ) -> GateResult:
        if timeout_s <= 0:
            raise ValueError("repeatability timeout must be positive")
        if full_runs <= 0 or stage_runs <= 0:
            raise ValueError("repeatability run counts must be positive")
        output_root = self.bundle / "repeatability-runs"
        output_root.mkdir()
        command = [
            sys.executable,
            str(self.repository_root / "scripts/run_productization_process_suite.py"),
            "--full-runs",
            str(full_runs),
            "--stage-runs",
            str(stage_runs),
            "--timeout-seconds",
            str(timeout_s),
            "--output",
            str(output_root),
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=self.repository_root,
            env={**os.environ, "TEMP": str(self.temp), "TMP": str(self.temp)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        external_timeout = timeout_s * 8 + 120
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=external_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_subprocess_tree(process)
            stdout, stderr = process.communicate()
        duration = time.monotonic() - started
        stdout_path = self.logs / "process_repeatability.stdout.log"
        stderr_path = self.logs / "process_repeatability.stderr.log"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        evidence_directories = sorted(output_root.glob("process-repeatability-*"))
        if timed_out or len(evidence_directories) != 1:
            result = GateResult(
                name="process_repeatability",
                category="software",
                status=AcceptanceStatus.FAIL,
                command=command,
                reason=(
                    f"repeatability runner exceeded {external_timeout:.0f}s"
                    if timed_out
                    else "repeatability runner did not produce exactly one evidence bundle"
                ),
                duration_s=duration,
                exit_code=None if timed_out else process.returncode,
                stdout_log=str(stdout_path.relative_to(self.bundle)),
                stderr_log=str(stderr_path.relative_to(self.bundle)),
            )
            self.results.append(result)
            return result
        return self.consume_repeatability(
            evidence_directories[0],
            command=command,
            stdout_log=str(stdout_path.relative_to(self.bundle)),
            stderr_log=str(stderr_path.relative_to(self.bundle)),
            duration_s=duration,
            exit_code=process.returncode,
        )

    def record_real_not_run(self, reason: str) -> None:
        for spec in REAL_MODEL_GATES:
            self.results.append(
                GateResult(
                    name=spec.name,
                    category="real_model",
                    status=AcceptanceStatus.NOT_RUN,
                    command=[],
                    reason=reason,
                )
            )

    def run_real_model(self) -> None:
        evidence_root = os.environ.get("SWARM_REAL_MODEL_ACCEPTANCE_EVIDENCE")
        if os.environ.get("SWARM_RUN_REAL_MODEL_ACCEPTANCE") != "1":
            reason = "SWARM_RUN_REAL_MODEL_ACCEPTANCE=1 was not set"
        elif not evidence_root:
            reason = "SWARM_REAL_MODEL_ACCEPTANCE_EVIDENCE was not set"
        elif not Path(evidence_root).expanduser().is_dir():
            reason = f"real-model evidence directory is unavailable: {evidence_root}"
        else:
            reason = ""
        if reason:
            for spec in REAL_MODEL_GATES:
                self.results.append(
                    GateResult(
                        name=spec.name,
                        category="real_model",
                        status=AcceptanceStatus.SKIP,
                        command=[],
                        reason=reason,
                    )
                )
            return
        for spec in REAL_MODEL_GATES:
            self._run_pytest(spec, category="real_model")

    def record_physical_not_run(
        self,
        reason: str,
        *,
        gate_name: str = "physical_two_machine",
    ) -> None:
        if gate_name not in PHYSICAL_GATES:
            raise ValueError(f"unknown physical gate {gate_name!r}")
        self.results.append(
            GateResult(
                name=gate_name,
                category="physical",
                status=AcceptanceStatus.NOT_RUN,
                command=[],
                reason=reason,
            )
        )

    def validate_physical(
        self,
        configuration_path: Path,
        *,
        gate_name: str = "physical_two_machine",
    ) -> None:
        if gate_name not in PHYSICAL_GATES:
            raise ValueError(f"unknown physical gate {gate_name!r}")
        try:
            configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors = [f"invalid physical configuration: {exc}"]
            configuration = {}
        else:
            errors = validate_physical_configuration(configuration)
        if errors:
            status = AcceptanceStatus.FAIL
            reason = "; ".join(errors)
        else:
            configured_directory = Path(
                str(configuration["evidence_output_directory"])
            ).expanduser()
            if not configured_directory.is_absolute():
                configured_directory = configuration_path.parent / configured_directory
            status, evidence_errors = validate_physical_evidence(
                configuration,
                configured_directory.resolve(),
            )
            reason = (
                "checksummed evidence proves two distinct physical workers"
                if status == AcceptanceStatus.PASS
                else "; ".join(evidence_errors)
            )
            summary_path = configured_directory.resolve() / "physical-evidence.json"
            if status == AcceptanceStatus.PASS:
                self.physical_evidence = json.loads(summary_path.read_text(encoding="utf-8"))
        self.results.append(
            GateResult(
                name=gate_name,
                category="physical",
                status=status,
                command=[],
                reason=reason,
            )
        )

    def write_bundle(self) -> tuple[Path, OverallStatus]:
        repeatability_results = [
            (index, result)
            for index, result in enumerate(self.results)
            if result.category == "software" and result.name == "process_repeatability"
        ]
        if not repeatability_results:
            self.record_repeatability_not_run("repeatability evidence was never evaluated")
        else:
            index, repeatability_result = repeatability_results[-1]
            if repeatability_result.status == AcceptanceStatus.PASS:
                if self.repeatability_path is None:
                    self.results[index] = replace(
                        repeatability_result,
                        status=AcceptanceStatus.NOT_RUN,
                        reason="validated repeatability evidence is missing",
                    )
                else:
                    status, errors, payload = validate_repeatability_evidence(
                        self.repository_root,
                        self.repeatability_path,
                    )
                    self.repeatability_evidence = payload
                    if status != AcceptanceStatus.PASS:
                        self.results[index] = replace(
                            repeatability_result,
                            status=status,
                            reason="; ".join(errors),
                        )
        overall = aggregate_status(self.results)
        evidence = environment_evidence(self.repository_root)
        summary = {
            "document_type": "swarm-productization-acceptance",
            "format_version": ACCEPTANCE_BUNDLE_VERSION,
            "overall_status": overall.value,
            "created_at": _utc_now(),
            "gate_counts": {
                status.value: sum(result.status == status for result in self.results)
                for status in AcceptanceStatus
            },
            "test_counts": {
                "tests": sum(result.tests for result in self.results),
                "failures": sum(result.failures for result in self.results),
                "errors": sum(result.errors for result in self.results),
                "skipped": sum(result.skipped for result in self.results),
            },
            "definitions": {
                OverallStatus.SOFTWARE_ACCEPTANCE_PASS.value: (
                    "all software gates passed; real-model and physical closure are not claimed"
                ),
                OverallStatus.REAL_MODEL_ACCEPTANCE_PASS.value: (
                    "all software and four architecture-neutral real-model gates passed"
                ),
                OverallStatus.PHYSICAL_ACCEPTANCE_PASS.value: (
                    "software, real-model, and distinct physical-machine gates passed"
                ),
                OverallStatus.INCOMPLETE.value: "one or more required software gates did not run",
                OverallStatus.FAIL.value: "at least one executed gate failed",
            },
        }
        (self.bundle / "acceptance-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.bundle / "gate-results.json").write_text(
            json.dumps(
                [{**asdict(result), "status": result.status.value} for result in self.results],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.bundle / "environment.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        repeatability_relative: str | None = None
        repeatability_checksum: str | None = None
        if self.repeatability_path is not None and self.repeatability_path.is_file():
            repeatability_relative = str(self.repeatability_path.relative_to(self.bundle))
            repeatability_checksum = f"sha256:{_sha256_bytes(self.repeatability_path.read_bytes())}"
        (self.bundle / "repeatability.json").write_text(
            json.dumps(
                {
                    "schema_version": REPEATABILITY_SCHEMA_VERSION,
                    "summary_path": repeatability_relative,
                    "summary_sha256": repeatability_checksum,
                    "evidence": self.repeatability_evidence,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        executed = [record["evidence"] for record in self.model_evidence_records]
        executed_model_ids = sorted(
            {str(item["model_id"]) for item in executed if item.get("model_id")}
        )
        executed_revisions = sorted(
            {str(item["model_revision"]) for item in executed if item.get("model_revision")}
        )
        (self.bundle / "model.json").write_text(
            json.dumps(
                {
                    "model_id": executed_model_ids[0] if len(executed_model_ids) == 1 else None,
                    "model_revision": (
                        executed_revisions[0] if len(executed_revisions) == 1 else None
                    ),
                    "model_ids": executed_model_ids,
                    "model_revisions": executed_revisions,
                    "tokenizer_revision": next(
                        (
                            item.get("tokenizer_revision")
                            for item in executed
                            if item.get("tokenizer_revision")
                        ),
                        None,
                    ),
                    "topology": [item.get("topology") for item in executed if item.get("topology")],
                    "stage_assignments": [
                        assignment
                        for item in executed
                        for assignment in item.get("stage_assignments", [])
                    ],
                    "worker_identities": [
                        identity
                        for item in executed
                        for identity in item.get("worker_identities", [])
                    ],
                    "token_ids": [
                        item.get("generated_token_ids", item.get("token_ids", []))
                        for item in executed
                    ],
                    "timings": [
                        item.get("critical_path_timings", item.get("timings", {}))
                        for item in executed
                    ],
                    "recovery_events": [
                        event for item in executed for event in item.get("recovery_events", [])
                    ],
                    "route_generations": [item.get("route_generations", []) for item in executed],
                    "runs": self.model_evidence_records,
                    "physical_evidence": self.physical_evidence,
                    "note": "populated only by executed real-model or physical gates",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.bundle / "commands.json").write_text(
            json.dumps(
                [{"gate": result.name, "command": result.command} for result in self.results],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest: dict[str, str] = {}
        for path in sorted(self.bundle.rglob("*")):
            if path.is_file() and path.name != "checksums.json":
                manifest[path.relative_to(self.bundle).as_posix()] = _sha256_bytes(
                    path.read_bytes()
                )
        (self.bundle / "checksums.json").write_text(
            json.dumps(
                {
                    "algorithm": "sha256",
                    "format_version": 1,
                    "files": manifest,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self.bundle, overall


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    machine = subparsers.add_parser("machine-identity")
    machine.add_argument("--output", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, default=Path("artifacts/acceptance"))
    repeatability = run.add_mutually_exclusive_group()
    repeatability.add_argument(
        "--run-repeatability",
        "--require-repeatability",
        dest="run_repeatability",
        action="store_true",
    )
    repeatability.add_argument("--repeatability-evidence", type=Path)
    run.add_argument("--repeatability-runs", type=int, default=3)
    run.add_argument("--ring-repeatability-runs", type=int, default=5)
    run.add_argument("--repeatability-timeout-seconds", type=float, default=600)
    run.add_argument("--real-model", action="store_true")
    run.add_argument("--physical-config", type=Path)
    run.add_argument("--linux-x86-physical-config", type=Path)
    run.add_argument("--macos-arm64-physical-config", type=Path)
    run.add_argument("--linux-arm64-physical-config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "machine-identity":
        payload = json.dumps(machine_identity(), indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    if args.command != "run":
        _parser().print_help()
        return 2
    if args.repeatability_runs <= 0 or args.ring_repeatability_runs <= 0:
        _parser().error("repeatability run counts must be positive")
    repository_root = Path(__file__).resolve().parents[3]
    runner = ProductizationAcceptanceRunner(
        repository_root=repository_root,
        output_root=args.output,
    )
    runner.run_software()
    if args.run_repeatability:
        runner.run_repeatability(
            timeout_s=args.repeatability_timeout_seconds,
            full_runs=args.repeatability_runs,
            stage_runs=args.ring_repeatability_runs,
        )
    elif args.repeatability_evidence is not None:
        runner.consume_repeatability(args.repeatability_evidence)
    else:
        runner.record_repeatability_not_run(
            "repeatability evidence was not supplied; use --run-repeatability or "
            "--repeatability-evidence"
        )
    if args.real_model:
        runner.run_real_model()
    else:
        runner.record_real_not_run("--real-model was not requested")
    physical_arguments = (
        ("physical_two_machine", "--physical-config", args.physical_config),
        (
            "physical_linux_x86_64_cpu",
            "--linux-x86-physical-config",
            args.linux_x86_physical_config,
        ),
        (
            "physical_macos_arm64_mps",
            "--macos-arm64-physical-config",
            args.macos_arm64_physical_config,
        ),
        (
            "physical_linux_arm64_cpu",
            "--linux-arm64-physical-config",
            args.linux_arm64_physical_config,
        ),
    )
    for gate_name, option, configuration in physical_arguments:
        if configuration is not None:
            runner.validate_physical(
                configuration.expanduser().resolve(),
                gate_name=gate_name,
            )
        else:
            runner.record_physical_not_run(
                f"{option} was not provided; {PHYSICAL_GATES[gate_name]} was not run",
                gate_name=gate_name,
            )
    bundle, overall = runner.write_bundle()
    print(f"acceptance_bundle={bundle}")
    print(f"overall_status={overall.value}")
    return 1 if overall in {OverallStatus.FAIL, OverallStatus.INCOMPLETE} else 0


__all__ = [
    "ACCEPTANCE_BUNDLE_VERSION",
    "MACHINE_IDENTITY_VERSION",
    "NON_GPU_PRODUCT_TEST_ARGUMENTS",
    "NON_PRODUCT_SOURCE_AUDIT_TESTS",
    "PHYSICAL_EVIDENCE_VERSION",
    "REPEATABILITY_SCHEMA_VERSION",
    "REPEATABILITY_TEST_COMMAND_VERSION",
    "AcceptanceStatus",
    "GateResult",
    "OverallStatus",
    "ProductizationAcceptanceRunner",
    "aggregate_status",
    "environment_evidence",
    "machine_identity",
    "validate_physical_configuration",
    "validate_physical_evidence",
    "validate_repeatability_evidence",
]
