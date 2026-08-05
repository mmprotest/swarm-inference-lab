"""Honest, versioned productization acceptance and evidence generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import platform
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import psutil

ACCEPTANCE_BUNDLE_VERSION = 1
MACHINE_IDENTITY_VERSION = 1
PHYSICAL_EVIDENCE_VERSION = 1
REAL_MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
REAL_MODEL_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"


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


@dataclass(frozen=True, slots=True)
class GateSpec:
    name: str
    tests: tuple[str, ...]
    timeout_s: float


SOFTWARE_GATES = (
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
        ("tests/integration", "tests/failure", "-m", "not gpu"),
        600,
    ),
)

REAL_MODEL_GATES = (
    GateSpec(
        "two_stage_olmoe",
        (
            "tests/integration/test_product_stage_ring.py::test_exact_olmoe_cuda_restart_and_replay_recovery_uses_third_worker",
        ),
        900,
    ),
    GateSpec(
        "restart_and_replay_olmoe",
        (
            "tests/integration/test_product_stage_ring.py::test_exact_olmoe_cuda_restart_and_replay_recovery_uses_third_worker",
        ),
        900,
    ),
    GateSpec(
        "whole_expert_olmoe",
        (
            "tests/integration/test_product_real_expert_acceptance.py::test_real_olmoe_whole_expert_product_inference",
        ),
        900,
    ),
    GateSpec(
        "native_microshard_olmoe",
        (
            "tests/integration/test_product_real_expert_acceptance.py::test_real_olmoe_native_microshard_product_inference",
        ),
        900,
    ),
)


def aggregate_status(results: list[GateResult]) -> OverallStatus:
    software = [result for result in results if result.category == "software"]
    real = [result for result in results if result.category == "real_model"]
    physical = [result for result in results if result.category == "physical"]
    if any(result.status == AcceptanceStatus.FAIL for result in results):
        return OverallStatus.FAIL
    if not software or any(result.status != AcceptanceStatus.PASS for result in software):
        return OverallStatus.INCOMPLETE
    real_pass = bool(real) and all(result.status == AcceptanceStatus.PASS for result in real)
    physical_pass = bool(physical) and all(
        result.status == AcceptanceStatus.PASS for result in physical
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

    commands = [_command_text(item) for item in evidence.get("commands", [])]
    required_commands = {
        "coordinator": 1,
        "worker": 2,
        "model deploy": 1,
        "submit": 2,
    }
    for fragment, minimum in required_commands.items():
        count = sum(f"swarm {fragment}" in command for command in commands)
        if count < minimum:
            errors.append(f"physical evidence is missing {minimum} 'swarm {fragment}' command(s)")

    source_files = evidence.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        errors.append("physical evidence must checksum its source logs and command outputs")
    else:
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
    except (ImportError, RuntimeError) as exc:
        cuda["error"] = f"{type(exc).__name__}: {exc}"
    status = _git_value(repository_root, "status", "--porcelain")
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
    }


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
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=spec.timeout_s,
            )
            status = AcceptanceStatus.PASS if completed.returncode == 0 else AcceptanceStatus.FAIL
            reason = "pytest command passed" if status == AcceptanceStatus.PASS else "pytest failed"
            if completed.returncode == 0 and category == "real_model":
                try:
                    tests, failures, errors, skipped = _junit_counts(junit_path)
                except (OSError, ET.ParseError, ValueError) as exc:
                    status = AcceptanceStatus.FAIL
                    reason = f"cannot verify real-model test execution: {exc}"
                else:
                    if tests == 0:
                        status = AcceptanceStatus.FAIL
                        reason = "real-model gate selected no tests"
                    elif skipped:
                        status = AcceptanceStatus.SKIP
                        reason = f"{skipped} of {tests} required real-model tests skipped"
                    elif failures or errors:
                        status = AcceptanceStatus.FAIL
                        reason = f"JUnit reported {failures} failures and {errors} errors"
            exit_code: int | None = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            status = AcceptanceStatus.FAIL
            reason = f"external timeout after {spec.timeout_s:.0f}s"
            exit_code = None
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr or ""
            )
        duration = time.monotonic() - started
        stdout_path = self.logs / f"{spec.name}.stdout.log"
        stderr_path = self.logs / f"{spec.name}.stderr.log"
        stdout_path.write_text(str(stdout), encoding="utf-8")
        stderr_path.write_text(str(stderr), encoding="utf-8")
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
        )
        self.results.append(result)
        for evidence_path in sorted(gate_evidence.glob("*.json")):
            try:
                payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                self.model_evidence_records.append(
                    {
                        "gate": spec.name,
                        "path": str(evidence_path.relative_to(self.bundle)),
                        "evidence": payload,
                    }
                )
        return result

    def run_software(self) -> None:
        for spec in SOFTWARE_GATES:
            self._run_pytest(spec, category="software")

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
        model_path = (
            self.repository_root / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
        )
        cuda_available = bool(environment_evidence(self.repository_root)["cuda"]["available"])
        if os.environ.get("SWARM_RUN_PRODUCT_OLMOE_CUDA") != "1":
            reason = "SWARM_RUN_PRODUCT_OLMOE_CUDA=1 was not set"
        elif not model_path.is_dir():
            reason = f"pinned OLMoE snapshot is unavailable: {model_path}"
        elif not cuda_available:
            reason = "CUDA is unavailable"
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

    def record_physical_not_run(self, reason: str) -> None:
        self.results.append(
            GateResult(
                name="physical_two_machine",
                category="physical",
                status=AcceptanceStatus.NOT_RUN,
                command=[],
                reason=reason,
            )
        )

    def validate_physical(self, configuration_path: Path) -> None:
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
                name="physical_two_machine",
                category="physical",
                status=status,
                command=[],
                reason=reason,
            )
        )

    def write_bundle(self) -> tuple[Path, OverallStatus]:
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
            "definitions": {
                OverallStatus.SOFTWARE_ACCEPTANCE_PASS.value: (
                    "all software gates passed; real-model and physical closure are not claimed"
                ),
                OverallStatus.REAL_MODEL_ACCEPTANCE_PASS.value: (
                    "all software and four real OLMoE product gates passed"
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
        executed = [record["evidence"] for record in self.model_evidence_records]
        (self.bundle / "model.json").write_text(
            json.dumps(
                {
                    "model_id": REAL_MODEL_ID,
                    "model_revision": REAL_MODEL_REVISION,
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
                    "token_ids": [item.get("token_ids", []) for item in executed],
                    "timings": [item.get("timings", {}) for item in executed],
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
    run.add_argument("--real-model", action="store_true")
    run.add_argument("--physical-config", type=Path)
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
    repository_root = Path(__file__).resolve().parents[3]
    runner = ProductizationAcceptanceRunner(
        repository_root=repository_root,
        output_root=args.output,
    )
    runner.run_software()
    if args.real_model:
        runner.run_real_model()
    else:
        runner.record_real_not_run("--real-model was not requested")
    if args.physical_config is not None:
        runner.validate_physical(args.physical_config.expanduser().resolve())
    else:
        runner.record_physical_not_run("--physical-config was not provided")
    bundle, overall = runner.write_bundle()
    print(f"acceptance_bundle={bundle}")
    print(f"overall_status={overall.value}")
    return 1 if overall in {OverallStatus.FAIL, OverallStatus.INCOMPLETE} else 0


__all__ = [
    "ACCEPTANCE_BUNDLE_VERSION",
    "MACHINE_IDENTITY_VERSION",
    "PHYSICAL_EVIDENCE_VERSION",
    "AcceptanceStatus",
    "GateResult",
    "OverallStatus",
    "ProductizationAcceptanceRunner",
    "aggregate_status",
    "environment_evidence",
    "machine_identity",
    "validate_physical_configuration",
    "validate_physical_evidence",
]
