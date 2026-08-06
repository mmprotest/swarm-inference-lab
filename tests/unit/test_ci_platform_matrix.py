from __future__ import annotations

from pathlib import Path

import yaml


def test_product_ci_declares_actual_supported_platform_jobs_and_wheel_gates() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/productization.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    matrix = jobs["platform-product"]["strategy"]["matrix"]["include"]
    platforms = {item["platform"]: item for item in matrix}
    assert set(platforms) == {
        "windows-x86_64-cpu",
        "linux-x86_64-cpu",
        "macos-arm64-mps",
        "linux-arm64-cpu",
    }
    assert platforms["macos-arm64-mps"]["runner"] == "macos-14"
    assert platforms["linux-arm64-cpu"]["runner"] == "ubuntu-24.04-arm"
    serialized = Path(".github/workflows/productization.yml").read_text(encoding="utf-8")
    for requirement in (
        "uv build --wheel",
        "test_wheel_install.py",
        "scripts/install.ps1",
        "scripts/install.sh",
        "pytest tests/unit",
        'tests/integration tests/failure -m "not gpu"',
        "upload-artifact",
    ):
        assert requirement in serialized


def test_source_test_jobs_initialize_and_verify_the_pinned_submodule() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/productization.yml").read_text(encoding="utf-8")
    )
    for job_name in (
        "quality",
        "platform-product",
        "python-compatibility",
        "software-acceptance",
    ):
        steps = workflow["jobs"][job_name]["steps"]
        checkout = next(step for step in steps if step.get("uses") == "actions/checkout@v4")
        assert checkout["with"]["submodules"] == "recursive"
        commands = [str(step.get("run", "")) for step in steps]
        assert any("verify_colibri_source.py" in command for command in commands)
