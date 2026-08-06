from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from swarm_inference.backends.colibri import dependency
from swarm_inference.backends.colibri.dependency import (
    COLIBRI_SOURCE_REMEDIATION,
    ColibriSourceDependencyError,
    verify_colibri_source_contract,
)


def _source_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    integration = root / "integrations" / "colibri"
    checkout = root / "third_party" / "colibri"
    integration.mkdir(parents=True)
    (checkout / ".git").mkdir(parents=True)
    (checkout / "LICENSE").write_text(
        "Apache License\nVersion 2.0, January 2004\n",
        encoding="utf-8",
    )
    (integration / "dependency.json").write_text(
        json.dumps(
            {
                "repository": dependency.COLIBRI_REPOSITORY,
                "repository_url": dependency.COLIBRI_REPOSITORY_URL,
                "release": dependency.COLIBRI_RELEASE,
                "commit": dependency.COLIBRI_COMMIT,
                "license": dependency.COLIBRI_LICENSE,
                "license_path": "third_party/colibri/LICENSE",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_missing_source_checkout_has_precise_dependency_remediation(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path)
    checkout = root / "third_party" / "colibri"
    (checkout / ".git").rmdir()
    (checkout / "LICENSE").unlink()
    checkout.rmdir()
    with pytest.raises(ColibriSourceDependencyError) as raised:
        verify_colibri_source_contract(root)
    message = str(raised.value)
    assert "pinned Colibri source checkout is missing" in message
    assert COLIBRI_SOURCE_REMEDIATION in message


def test_wrong_colibri_commit_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _source_fixture(tmp_path)
    wrong = "0" * 40
    monkeypatch.setattr(dependency, "_git", lambda *_args, **_kwargs: wrong)
    with pytest.raises(ColibriSourceDependencyError) as raised:
        verify_colibri_source_contract(root)
    assert f"expected {dependency.COLIBRI_COMMIT}, found {wrong}" in str(raised.value)
    assert COLIBRI_SOURCE_REMEDIATION in str(raised.value)


def test_correct_commit_and_apache_license_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _source_fixture(tmp_path)
    monkeypatch.setattr(
        dependency,
        "_git",
        lambda *_args, **_kwargs: dependency.COLIBRI_COMMIT,
    )
    result = verify_colibri_source_contract(root)
    assert result["status"] == "PASS"
    assert result["commit"] == dependency.COLIBRI_COMMIT
    assert len(result["license_sha256"]) == 64


def test_source_ci_jobs_checkout_recursive_submodules_before_preflight() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/productization.yml").read_text(encoding="utf-8")
    )
    for name in ("quality", "platform-product", "python-compatibility", "software-acceptance"):
        steps = workflow["jobs"][name]["steps"]
        checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
        assert checkout.get("with", {}).get("submodules") == "recursive"
        preflight = next(
            index
            for index, step in enumerate(steps)
            if "verify_colibri_source.py" in str(step.get("run", ""))
        )
        source_tests = [
            index
            for index, step in enumerate(steps)
            if "pytest" in str(step.get("run", ""))
            or "run_productization_acceptance.py" in str(step.get("run", ""))
            or "ruff" in str(step.get("run", ""))
            or "mypy" in str(step.get("run", ""))
        ]
        assert source_tests
        assert preflight < min(source_tests)
