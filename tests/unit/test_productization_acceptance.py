from __future__ import annotations

import hashlib
import json
import platform
import runpy
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from swarm_inference.acceptance import productization
from swarm_inference.acceptance.productization import (
    ACCEPTANCE_BUNDLE_VERSION,
    NON_GPU_PRODUCT_TEST_ARGUMENTS,
    NON_PRODUCT_SOURCE_AUDIT_TESTS,
    REAL_MODEL_GATES,
    REPEATABILITY_TEST_COMMAND_VERSION,
    SOFTWARE_GATES,
    AcceptanceStatus,
    GateResult,
    GateSpec,
    OverallStatus,
    ProductizationAcceptanceRunner,
    aggregate_status,
    environment_evidence,
    machine_identity,
    validate_physical_configuration,
    validate_physical_evidence,
    validate_repeatability_evidence,
)


def test_repeatability_producer_reuses_current_acceptance_contract() -> None:
    namespace = runpy.run_path("scripts/run_productization_process_suite.py")

    assert namespace["ACCEPTANCE_BUNDLE_VERSION"] == ACCEPTANCE_BUNDLE_VERSION
    assert namespace["REPEATABILITY_TEST_COMMAND_VERSION"] == REPEATABILITY_TEST_COMMAND_VERSION
    assert tuple(namespace["NON_GPU_PRODUCT_TEST_ARGUMENTS"]) == NON_GPU_PRODUCT_TEST_ARGUMENTS


def test_non_gpu_product_collection_excludes_opt_in_experiment_audits() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *NON_GPU_PRODUCT_TEST_ARGUMENTS,
            "--collect-only",
            "-q",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert "test_cluster_pair_and_join.py" in completed.stdout
    assert all(path not in completed.stdout for path in NON_PRODUCT_SOURCE_AUDIT_TESTS)


def test_acceptance_parser_supports_required_repeatability_command() -> None:
    arguments = productization._parser().parse_args(
        [
            "run",
            "--require-repeatability",
            "--repeatability-runs",
            "3",
            "--ring-repeatability-runs",
            "5",
            "--repeatability-timeout-seconds",
            "600",
        ]
    )

    assert arguments.run_repeatability is True
    assert arguments.repeatability_runs == 3
    assert arguments.ring_repeatability_runs == 5
    assert arguments.repeatability_timeout_seconds == 600


def result(
    name: str,
    category: str,
    status: AcceptanceStatus,
) -> GateResult:
    return GateResult(
        name=name,
        category=category,  # type: ignore[arg-type]
        status=status,
        command=[],
        reason="test",
    )


def software_passes() -> list[GateResult]:
    return [result(spec.name, "software", AcceptanceStatus.PASS) for spec in SOFTWARE_GATES] + [
        result("process_repeatability", "software", AcceptanceStatus.PASS)
    ]


def real_passes() -> list[GateResult]:
    return [result(spec.name, "real_model", AcceptanceStatus.PASS) for spec in REAL_MODEL_GATES]


def test_status_aggregation_never_promotes_skip_or_not_run() -> None:
    software = software_passes()
    real_not_run = [
        result(f"real-{index}", "real_model", AcceptanceStatus.NOT_RUN) for index in range(4)
    ]
    assert aggregate_status(software + real_not_run) == OverallStatus.SOFTWARE_ACCEPTANCE_PASS

    real_with_skip = [
        result("real-pass", "real_model", AcceptanceStatus.PASS),
        result("real-skip", "real_model", AcceptanceStatus.SKIP),
    ]
    assert aggregate_status(software + real_with_skip) == OverallStatus.SOFTWARE_ACCEPTANCE_PASS
    assert (
        aggregate_status([result("software", "software", AcceptanceStatus.SKIP)])
        == OverallStatus.INCOMPLETE
    )
    assert (
        aggregate_status([*software, result("failed", "physical", AcceptanceStatus.FAIL)])
        == OverallStatus.FAIL
    )


def test_status_aggregation_requires_real_before_physical_full_pass() -> None:
    software = software_passes()
    real = real_passes()
    physical = [result("physical_two_machine", "physical", AcceptanceStatus.PASS)]
    assert aggregate_status(software + real) == OverallStatus.REAL_MODEL_ACCEPTANCE_PASS
    assert aggregate_status(software + real + physical) == OverallStatus.PHYSICAL_ACCEPTANCE_PASS
    assert aggregate_status(software + physical) == OverallStatus.SOFTWARE_ACCEPTANCE_PASS


def _physical_configuration() -> dict[str, object]:
    first = machine_identity()
    second = {
        **first,
        "machine_identity": "sha256:" + "1" * 64,
        "host_identity": "sha256:" + "2" * 64,
        "process_namespace_identity": "sha256:" + "3" * 64,
    }
    return {
        "coordinator_host": "coordinator.example",
        "coordinator_endpoint": "coordinator.example:50051",
        "worker_a_host": "worker-a.example",
        "worker_b_host": "worker-b.example",
        "worker_a_identity_path": ".swarm/identities/worker-a.json",
        "worker_b_identity_path": ".swarm/identities/worker-b.json",
        "worker_a_machine_identity": first,
        "worker_b_machine_identity": second,
        "model_revision": "commit",
        "evidence_output_directory": "artifacts/acceptance/physical",
    }


def test_loopback_and_same_host_cannot_satisfy_physical_gate() -> None:
    configuration = _physical_configuration()
    configuration["worker_a_host"] = "127.0.0.1"
    configuration["worker_b_host"] = "localhost"
    errors = validate_physical_configuration(configuration)
    assert sum("loopback" in error for error in errors) >= 2

    configuration = _physical_configuration()
    configuration["worker_b_host"] = configuration["worker_a_host"]
    configuration["worker_b_machine_identity"] = configuration["worker_a_machine_identity"]
    errors = validate_physical_configuration(configuration)
    assert "physical workers resolve to the same declared host" in errors
    assert "physical workers report the same machine identity" in errors
    assert "physical workers report the same process namespace" in errors


def test_physical_configuration_readiness_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses = {
        "coordinator.example": "192.0.2.10",
        "worker-a.example": "192.0.2.11",
        "worker-b.example": "192.0.2.12",
    }

    def resolve(
        host: str, _port: object
    ) -> list[tuple[object, object, object, str, tuple[str, int]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addresses[host], 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    assert validate_physical_configuration(_physical_configuration()) == []


def test_acceptance_bundle_has_provenance_placeholders_and_verified_checksums(
    tmp_path: Path,
) -> None:
    runner = ProductizationAcceptanceRunner(
        repository_root=Path.cwd(),
        output_root=tmp_path,
    )
    runner.results.extend(
        result(spec.name, "software", AcceptanceStatus.PASS) for spec in SOFTWARE_GATES
    )
    repeatability, _ = _write_repeatability_bundle(tmp_path)
    runner.consume_repeatability(repeatability)
    runner.record_real_not_run("not requested")
    runner.record_physical_not_run("not requested")
    bundle, overall = runner.write_bundle()

    assert overall == OverallStatus.SOFTWARE_ACCEPTANCE_PASS
    for name in (
        "acceptance-summary.json",
        "gate-results.json",
        "environment.json",
        "repeatability.json",
        "model.json",
        "commands.json",
        "checksums.json",
    ):
        assert (bundle / name).is_file()
    environment = json.loads((bundle / "environment.json").read_text(encoding="utf-8"))
    assert isinstance(environment["git_dirty"], bool)
    assert environment["python_version"]
    assert environment["os"]
    assert environment["machine"]["machine_identity"].startswith("sha256:")
    model = json.loads((bundle / "model.json").read_text(encoding="utf-8"))
    for field in (
        "model_revision",
        "tokenizer_revision",
        "topology",
        "stage_assignments",
        "token_ids",
        "timings",
        "recovery_events",
        "route_generations",
    ):
        assert field in model
    checksums = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in checksums["files"].items():
        assert hashlib.sha256((bundle / relative).read_bytes()).hexdigest() == expected


def test_dirty_tree_reporting_is_explicit() -> None:
    evidence = environment_evidence(Path.cwd())
    assert isinstance(evidence["git_dirty"], bool)
    assert "git_status" in evidence


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_repeatability_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    evidence = tmp_path / "repeatability"
    evidence.mkdir()
    results: list[dict[str, object]] = []
    names = [f"full-{index}" for index in range(1, 4)] + [
        f"stage-ring-{index}" for index in range(1, 6)
    ]
    for name in names:
        stdout = evidence / f"{name}.stdout.log"
        stderr = evidence / f"{name}.stderr.log"
        stdout.write_text("passed\n", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        results.append(
            {
                "name": name,
                "status": "PASS",
                "exit_code": 0,
                "resource_warning_count": 0,
                "warning_scan": {"status": "PASS", "matches": [], "ignore_list": []},
                "graceful_shutdown_count": 2,
                "unexpected_terminate_count": 0,
                "unexpected_kill_count": 0,
                "leaked_process_count": 0,
                "test_counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
                "checksums": {
                    stdout.name: "sha256:" + hashlib.sha256(stdout.read_bytes()).hexdigest(),
                    stderr.name: "sha256:" + hashlib.sha256(stderr.read_bytes()).hexdigest(),
                },
            }
        )
    git_status = _git_output("status", "--porcelain")
    process_script = Path("scripts/run_productization_process_suite.py")
    acceptance_source = Path("src/swarm_inference/acceptance/productization.py")
    payload: dict[str, object] = {
        "document_type": "swarm-process-repeatability",
        "schema_version": 2,
        "test_command_version": REPEATABILITY_TEST_COMMAND_VERSION,
        "acceptance_schema_version": ACCEPTANCE_BUNDLE_VERSION,
        "excluded_source_audit_tests": list(NON_PRODUCT_SOURCE_AUDIT_TESTS),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "finish_git_status": git_status,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "required_runs": {"full_process_suite": 3, "stage_ring_module": 5},
        "requested_runs": {"full_process_suite": 3, "stage_ring_module": 5},
        "process_runner_sha256": "sha256:"
        + hashlib.sha256(process_script.read_bytes()).hexdigest(),
        "acceptance_source_sha256": "sha256:"
        + hashlib.sha256(acceptance_source.read_bytes()).hexdigest(),
        "results": results,
        "overall_repeatability_status": "PASS",
    }
    _save_repeatability_payload(evidence, payload)
    return evidence, payload


def _save_repeatability_payload(evidence: Path, payload: dict[str, object]) -> None:
    payload.pop("evidence_checksum", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["evidence_checksum"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    (evidence / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_missing_repeatability_evidence_yields_incomplete(tmp_path: Path) -> None:
    runner = ProductizationAcceptanceRunner(repository_root=Path.cwd(), output_root=tmp_path)
    runner.results.extend(
        result(spec.name, "software", AcceptanceStatus.PASS) for spec in SOFTWARE_GATES
    )
    runner.record_repeatability_not_run("missing")
    assert aggregate_status(runner.results) == OverallStatus.INCOMPLETE

    status, errors, payload = validate_repeatability_evidence(Path.cwd(), tmp_path / "missing")
    assert status == AcceptanceStatus.NOT_RUN
    assert errors
    assert payload is None


def test_stale_commit_and_dirty_tree_repeatability_are_rejected(tmp_path: Path) -> None:
    evidence, payload = _write_repeatability_bundle(tmp_path)
    payload["git_commit"] = "0" * 40
    _save_repeatability_payload(evidence, payload)
    status, errors, _ = validate_repeatability_evidence(Path.cwd(), evidence)
    assert status == AcceptanceStatus.NOT_RUN
    assert any("another git commit" in error for error in errors)

    payload["git_commit"] = _git_output("rev-parse", "HEAD")
    payload["git_status"] = "synthetic dirty-tree mismatch"
    _save_repeatability_payload(evidence, payload)
    status, errors, _ = validate_repeatability_evidence(Path.cwd(), evidence)
    assert status == AcceptanceStatus.NOT_RUN
    assert any("dirty-tree state" in error for error in errors)


def test_repeatability_with_skipped_product_test_is_incomplete(tmp_path: Path) -> None:
    evidence, payload = _write_repeatability_bundle(tmp_path)
    results = payload["results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    counts = first["test_counts"]
    assert isinstance(counts, dict)
    counts["skipped"] = 1
    _save_repeatability_payload(evidence, payload)

    status, errors, _ = validate_repeatability_evidence(Path.cwd(), evidence)
    assert status == AcceptanceStatus.NOT_RUN
    assert any("contains skipped tests" in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ({"status": "FAIL", "exit_code": 1}, "has status FAIL"),
        (
            {
                "status": "WARNING_FAILURE",
                "resource_warning_count": 1,
                "warning_scan": {
                    "status": "FAIL",
                    "matches": ["resource_tracker"],
                    "ignore_list": [],
                },
            },
            "managed-resource warning",
        ),
        ({"unexpected_terminate_count": 1}, "unexpected_terminate_count"),
    ],
)
def test_failed_warning_or_forced_repeatability_run_fails_acceptance(
    tmp_path: Path,
    mutation: dict[str, object],
    expected_fragment: str,
) -> None:
    evidence, payload = _write_repeatability_bundle(tmp_path)
    results = payload["results"]
    assert isinstance(results, list)
    first = results[0]
    assert isinstance(first, dict)
    first.update(mutation)
    payload["overall_repeatability_status"] = "FAIL"
    _save_repeatability_payload(evidence, payload)

    status, errors, _ = validate_repeatability_evidence(Path.cwd(), evidence)
    assert status == AcceptanceStatus.FAIL
    assert any(expected_fragment in error for error in errors)


def test_three_plus_five_clean_runs_permit_software_acceptance(tmp_path: Path) -> None:
    evidence, _ = _write_repeatability_bundle(tmp_path)
    status, errors, _ = validate_repeatability_evidence(Path.cwd(), evidence)
    assert status == AcceptanceStatus.PASS
    assert errors == []

    runner = ProductizationAcceptanceRunner(
        repository_root=Path.cwd(), output_root=tmp_path / "acceptance"
    )
    runner.results.extend(
        result(spec.name, "software", AcceptanceStatus.PASS) for spec in SOFTWARE_GATES
    )
    consumed = runner.consume_repeatability(evidence)
    assert consumed.status == AcceptanceStatus.PASS
    runner.record_real_not_run("not requested")
    runner.record_physical_not_run("not requested")
    _bundle, overall = runner.write_bundle()
    assert overall == OverallStatus.SOFTWARE_ACCEPTANCE_PASS


def test_real_model_baseline_and_recovery_are_distinct_gates() -> None:
    gates = {spec.name: spec for spec in REAL_MODEL_GATES}
    baseline = gates["two_stage_olmoe_baseline"]
    recovery = gates["restart_and_replay_olmoe"]
    assert baseline.tests != recovery.tests
    assert "test_exact_two_stage_olmoe_cuda_baseline" in baseline.tests[0]
    assert "restart_and_replay_recovery" in recovery.tests[0]

    source = Path("tests/integration/test_product_stage_ring.py").read_text(encoding="utf-8")
    assert '"failure_injected": False' in source
    assert '"failure_injected": True' in source


def test_real_model_pytest_skip_is_never_classified_as_pass(tmp_path: Path) -> None:
    test_file = tmp_path / "test_skipped_real_gate.py"
    test_file.write_text(
        "import pytest\n\n@pytest.mark.skip(reason='fixture unavailable')\n"
        "def test_real_gate():\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    runner = ProductizationAcceptanceRunner(
        repository_root=tmp_path,
        output_root=tmp_path / "output",
    )
    result = runner._run_pytest(
        GateSpec("skipped_real", (str(test_file),), 30),
        category="real_model",
    )
    assert result.status == AcceptanceStatus.SKIP
    assert "1 of 1" in result.reason


def test_software_pytest_skip_is_never_classified_as_pass(tmp_path: Path) -> None:
    test_file = tmp_path / "test_skipped_software_gate.py"
    test_file.write_text(
        "import pytest\n\n@pytest.mark.skip(reason='fixture unavailable')\n"
        "def test_software_gate():\n    raise AssertionError('must not run')\n",
        encoding="utf-8",
    )
    runner = ProductizationAcceptanceRunner(
        repository_root=tmp_path,
        output_root=tmp_path / "output",
    )
    result = runner._run_pytest(
        GateSpec("skipped_software", (str(test_file),), 30),
        category="software",
    )
    assert result.status == AcceptanceStatus.SKIP
    assert aggregate_status([result]) == OverallStatus.INCOMPLETE


def test_complete_distinct_physical_evidence_can_pass(tmp_path: Path) -> None:
    configuration = _physical_configuration()
    source = tmp_path / "operator.log"
    source.write_text("captured product output\n", encoding="utf-8")
    worker_a_identity = configuration["worker_a_machine_identity"]
    worker_b_identity = configuration["worker_b_machine_identity"]
    assert isinstance(worker_a_identity, dict)
    assert isinstance(worker_b_identity, dict)
    evidence = {
        "document_type": "swarm-physical-two-machine-evidence",
        "format_version": 3,
        "model_revision": "commit",
        "machine_identities": {
            "worker_a": worker_a_identity,
            "worker_b": worker_b_identity,
        },
        "workers": [
            {
                "worker_id": "worker-a",
                "machine_identity": worker_a_identity["machine_identity"],
                "public_key_fingerprint": "a" * 64,
                "control_endpoint": "192.0.2.11:50052",
                "data_plane_endpoint": "192.0.2.11:51052",
            },
            {
                "worker_id": "worker-b",
                "machine_identity": worker_b_identity["machine_identity"],
                "public_key_fingerprint": "b" * 64,
                "control_endpoint": "192.0.2.12:50052",
                "data_plane_endpoint": "192.0.2.12:51052",
            },
        ],
        "worker_identities": {
            "worker_a": {"worker_id": "worker-a", "fingerprint": "a" * 64},
            "worker_b": {"worker_id": "worker-b", "fingerprint": "b" * 64},
        },
        "topology": {
            "assignments": [
                {"stage_id": 0, "worker_id": "worker-a"},
                {"stage_id": 1, "worker_id": "worker-b"},
            ]
        },
        "normal_run": {
            "status": "completed",
            "expected_token_ids": [7, 8],
            "token_ids": [7, 8],
        },
        "recovery_run": {
            "status": "completed",
            "expected_token_ids": [7, 8],
            "token_events": [
                {"token_position": 0, "token_id": 7},
                {"token_position": 1, "token_id": 8},
            ],
            "recovery_events": [{"event": "recovery_completed"}],
            "route_generations": [1, 2],
        },
        "installations": [
            {
                "node_id": "worker-a",
                "status": "PASS",
                "source": "source-wheel",
                "repository_cloned": False,
                "wheel_sha256": "sha256:" + "c" * 64,
            },
            {
                "node_id": "worker-b",
                "status": "PASS",
                "source": "source-wheel",
                "repository_cloned": False,
                "wheel_sha256": "sha256:" + "c" * 64,
            },
        ],
        "pairing": {
            "status": "consumed",
            "single_use": True,
            "fingerprint_copied_manually": False,
        },
        "automatic_configuration": {
            "backend_selected": True,
            "memory_selected": True,
            "control_endpoint_selected": True,
            "data_endpoint_selected": True,
            "ports_selected": True,
        },
        "services": [
            {
                "node_id": "worker-a",
                "running_after_terminal_close": True,
                "reconnected_after_restart": True,
            },
            {
                "node_id": "worker-b",
                "running_after_terminal_close": True,
                "reconnected_after_restart": True,
            },
        ],
        "artifacts": [
            {"artifact_id": "a" * 64, "verified": True},
            {"artifact_id": "b" * 64, "verified": True},
        ],
        "directed_network_links": [
            {
                "source_worker_id": "worker-a",
                "destination_worker_id": "worker-b",
                "destination_endpoint": "192.0.2.12:51052",
                "measured": True,
                "authentication_verified": True,
            },
            {
                "source_worker_id": "worker-b",
                "destination_worker_id": "worker-a",
                "destination_endpoint": "192.0.2.11:51052",
                "measured": True,
                "authentication_verified": True,
            },
        ],
        "speed_run": {
            "status": "completed",
            "expected_token_ids": [7, 8],
            "token_ids": [7, 8],
            "excluded_slow_node_id": "worker-b",
            "topology": {"assignments": [{"stage_id": 0, "worker_id": "worker-a"}]},
        },
        "capacity_run": {
            "status": "completed",
            "expected_token_ids": [7, 8],
            "token_ids": [7, 8],
            "included_slow_node_id": "worker-b",
            "topology": {
                "assignments": [
                    {"stage_id": 0, "worker_id": "worker-a"},
                    {"stage_id": 1, "worker_id": "worker-b"},
                ]
            },
        },
        "direct_stage_traffic": [
            {
                "source_worker_id": "worker-a",
                "destination_worker_id": "worker-b",
                "destination_endpoint": "192.0.2.12:51052",
                "observed": True,
                "bytes_observed": 4096,
                "evidence_file": source.name,
            }
        ],
        "commands": [
            "swarm cluster create --name physical-gate",
            "swarm node join <redacted-pairing-uri>",
            "swarm cluster status --json",
            "swarm run model --mode speed --revision commit",
            "swarm run model --mode capacity --revision commit",
        ],
        "source_files": {source.name: "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (tmp_path / "physical-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    status, errors = validate_physical_evidence(configuration, tmp_path)
    assert status == AcceptanceStatus.PASS
    assert errors == []

    capacity = evidence["capacity_run"]
    assert isinstance(capacity, dict)
    capacity["topology"] = {"assignments": [{"stage_id": 0, "worker_id": "worker-a"}]}
    (tmp_path / "physical-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    status, errors = validate_physical_evidence(configuration, tmp_path)
    assert status == AcceptanceStatus.FAIL
    assert any("both distinct physical machine identities" in error for error in errors)


def test_release_candidate_software_gates_are_explicit() -> None:
    names = {spec.name for spec in SOFTWARE_GATES}
    assert {
        "pairing_json_contract",
        "invitation_secret_file_protection",
        "no_secret_in_machine_output",
        "platform_status_separation",
        "firewall_ownership_isolation",
        "unpaired_wheel_installer_success",
        "confirmation_semantics",
        "recursive_source_dependency_validation",
        "clean_wheel_import_isolation",
        "physical_two_machine_configuration_readiness",
    } <= names
