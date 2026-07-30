from __future__ import annotations

from pathlib import Path

from swarm_inference.experiments.fanout_lifecycle import (
    REQUIRED_LIFECYCLE_EVENTS,
    LifecycleRecorder,
    pipeline_ready_time_seconds,
    read_lifecycle_events,
    validate_lifecycle_events,
)


def _complete_worker(path: Path, worker_id: str, stage_id: int, start: int) -> None:
    recorder = LifecycleRecorder(
        path=path,
        experiment_id="experiment-003-test",
        worker_id=worker_id,
        stage_id=stage_id,
        origin_monotonic_ns=0,
        process_id=1000 + stage_id,
    )
    for offset, event_name in enumerate(REQUIRED_LIFECYCLE_EVENTS):
        recorder.emit(
            event_name,
            monotonic_ns=start + offset,
            duration_ns=0 if event_name.endswith(("completed", "spawned")) else None,
        )


def test_required_lifecycle_events_order_and_pipeline_readiness(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _complete_worker(first, "worker-0", 0, 100)
    _complete_worker(second, "worker-1", 1, 200)
    rows = read_lifecycle_events([first, second])
    assert validate_lifecycle_events(rows) == []
    expected = max(
        int(row["monotonic_timestamp_ns"]) for row in rows if row["event_name"] == "worker_routable"
    )
    assert (
        pipeline_ready_time_seconds(
            rows,
            experiment_worker_start_origin_ns=50,
            required_worker_ids=["worker-0", "worker-1"],
        )
        == (expected - 50) / 1_000_000_000
    )


def test_missing_and_backward_lifecycle_evidence_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    recorder = LifecycleRecorder(
        path=path,
        experiment_id="experiment-003-test",
        worker_id="failed-worker",
        stage_id=3,
        origin_monotonic_ns=0,
        process_id=333,
    )
    recorder.emit("assignment_created", monotonic_ns=20)
    recorder.emit("process_spawn_started", monotonic_ns=10)
    rows = read_lifecycle_events([path])
    errors = validate_lifecycle_events(rows)
    assert any("missing lifecycle events" in error for error in errors)
    assert any("moved backward" in error for error in errors)
    # Partial evidence remains readable after a failed worker.
    assert {row["event_name"] for row in rows} == {
        "assignment_created",
        "process_spawn_started",
    }


def test_lifecycle_accepts_windows_launcher_and_runtime_process_ids(tmp_path: Path) -> None:
    path = tmp_path / "split-pid.jsonl"
    launcher = LifecycleRecorder(
        path=path,
        experiment_id="experiment-003-test",
        worker_id="worker-0",
        stage_id=0,
        origin_monotonic_ns=0,
        process_id=100,
    )
    runtime = LifecycleRecorder(
        path=path,
        experiment_id="experiment-003-test",
        worker_id="worker-0",
        stage_id=0,
        origin_monotonic_ns=0,
        process_id=200,
    )
    for offset, event_name in enumerate(REQUIRED_LIFECYCLE_EVENTS):
        recorder = runtime if event_name.startswith(("python_", "cuda_")) else launcher
        recorder.emit(event_name, monotonic_ns=offset + 1)
    assert validate_lifecycle_events(read_lifecycle_events([path])) == []
