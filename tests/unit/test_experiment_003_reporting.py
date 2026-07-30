from __future__ import annotations

from pathlib import Path

from swarm_inference.experiments.fanout_reporting import render_report


def test_report_preserves_failed_28_as_valid_measurement_and_labels_emulation(
    tmp_path: Path,
) -> None:
    summary = {
        "experiment_integrity_status": "PASS",
        "overall_status": "PASS",
        "maximum_semantic_worker_count": 28,
        "maximum_runnable_worker_count": 24,
        "maximum_stable_worker_count": 21,
        "single_request_latency_optimal_worker_count": 1,
        "concurrency_4_throughput_optimal_worker_count": 2,
        "file_cache_control": {"controlled": False},
    }
    report = render_report(
        run_dir=tmp_path,
        summary=summary,
        count_rows=[
            {"worker_count": 24, "runnable": True, "stable": False},
            {
                "worker_count": 28,
                "runnable": False,
                "stable": False,
                "failure_reason": "CUDA OOM",
            },
        ],
        acquisition_rows=[
            {
                "worker_count": 14,
                "stage_role": "middle",
                "profile": "gigabit_lan",
                "measurement_class": "emulated-shard-acquisition",
            }
        ],
        economics_rows=[],
        rejoin={"status": "PASS"},
    )
    text = report.read_text(encoding="utf-8")
    assert "Maximum runnable worker count: 24" in text
    assert "CUDA OOM" in text
    assert "emulated-shard-acquisition" in text
