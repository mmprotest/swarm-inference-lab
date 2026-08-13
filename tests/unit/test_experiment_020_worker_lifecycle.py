from __future__ import annotations

import json
import os
import subprocess
import sys


def test_host_agent_launches_eight_explicit_logical_workers(tmp_path) -> None:
    credential = tmp_path / "credential"
    credential.write_bytes(b"e020-test-only-credential-32-byte!")
    environment = os.environ.copy()
    environment["SWARM_RUN_CREDENTIAL_FILE"] = str(credential)
    environment["SWARM_POD_ID"] = "pod-011"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "swarm_inference.experiments.experiment_020.worker_main",
            "host-agent",
            "--workers-per-pod",
            "8",
            "--logical-devices",
            "--self-test",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["host_agent_compute_resource"] is False
    assert receipt["explicit_worker_processes"] == 8
    assert receipt["unique_worker_ids"] == 8
    assert receipt["unique_gpu_bindings"] == 8
    assert receipt["whole_layer_fallback"] is False
    assert receipt["whole_expert_fallback"] is False
