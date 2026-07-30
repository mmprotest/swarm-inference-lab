from __future__ import annotations


def test_first_experiment_launcher_is_non_pruning_and_truthful(
    repository_root,
) -> None:
    script = (repository_root / "scripts" / "run_first_experiment.ps1").read_text(encoding="utf-8")
    lowered = script.lower()
    assert "uv run --no-sync" in lowered
    assert "uv sync" not in lowered
    assert '"sync"' not in lowered
    assert "[switch]$Bootstrap" in script
    assert '[ValidateSet("synthetic", "cpu", "cuda")]' in script
    assert "finally {" in script
    assert "Dependency preservation check" in script
    assert "exit $exitCode" in script
    assert "--backend $Backend" in script
    assert "experiment_001_replica_scaling.yaml" in script


def test_matrix_runner_emits_progress_and_partial_summary(repository_root) -> None:
    source = (
        repository_root / "src" / "swarm_inference" / "experiments" / "loopback_matrix.py"
    ).read_text(encoding="utf-8")
    child = (repository_root / "src" / "swarm_inference" / "experiments" / "loopback.py").read_text(
        encoding="utf-8"
    )
    assert "Starting matrix point" in source
    assert "Completed point" in source
    assert "Child artifact:" in source
    assert '"partial": True' in source
    assert "except BaseException" in source
    assert "Warm-up:" not in child
    assert 'label="Warm-up"' in child
    assert 'label="Measurement"' in child
