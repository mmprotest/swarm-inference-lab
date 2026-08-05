from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from swarm_inference.acceptance.productization import (
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
)


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


def test_status_aggregation_never_promotes_skip_or_not_run() -> None:
    software = [result("software", "software", AcceptanceStatus.PASS)]
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
    software = [result("software", "software", AcceptanceStatus.PASS)]
    real = [result(f"real-{index}", "real_model", AcceptanceStatus.PASS) for index in range(4)]
    physical = [result("physical", "physical", AcceptanceStatus.PASS)]
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


def test_distinct_resolvable_physical_configuration_passes_static_validation(
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
    runner.results.append(result("software", "software", AcceptanceStatus.PASS))
    runner.record_real_not_run("not requested")
    runner.record_physical_not_run("not requested")
    bundle, overall = runner.write_bundle()

    assert overall == OverallStatus.SOFTWARE_ACCEPTANCE_PASS
    for name in (
        "acceptance-summary.json",
        "gate-results.json",
        "environment.json",
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
        "format_version": 1,
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
        "commands": [
            "swarm coordinator --config product.yaml",
            "swarm worker --worker-id worker-a",
            "swarm worker --worker-id worker-b",
            "swarm model deploy --plan plan.json",
            "swarm submit --request-id normal",
            "swarm submit --request-id recovery",
        ],
        "source_files": {source.name: "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (tmp_path / "physical-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    status, errors = validate_physical_evidence(configuration, tmp_path)
    assert status == AcceptanceStatus.PASS
    assert errors == []
