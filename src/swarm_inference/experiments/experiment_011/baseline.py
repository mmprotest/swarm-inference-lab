"""Fresh Experiment 010 baseline reproduction without modifying its runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.colibri_workloads import (
    RealCandidate,
    measure_reference,
    run_network_profile_matrix,
)
from swarm_inference.experiments.experiment_011.discovery import DiscoveredAssets


def run_fresh_expert_rpc_baseline(
    *,
    assets: DiscoveredAssets,
    profiles: Sequence[str],
    output_directory: Path,
    repeats: int = 1,
    timeout_seconds: float = 7200,
) -> list[dict[str, Any]]:
    candidate = RealCandidate(
        name="experiment_011_same_run_expert_rpc",
        bank_paths=tuple(Path(path) for path in assets.worker_bank_paths),
        mode="rpc",
        response_mode="per_expert_exact",
        data_plane="relayed_tcp",
        shard_layout="whole",
        exact_contract=True,
        coordinator_model=Path(assets.native_model_path),
        worker_memory_budget_bytes=512 * 1024 * 1024,
    )
    rows = run_network_profile_matrix(
        candidate=candidate,
        reference_paths=[Path(assets.experiment_010_workload_reference)],
        profiles=list(profiles),
        repeats=repeats,
        engine=Path(assets.native_engine),
        worker_executable=Path(assets.native_expert_worker),
        model_path=Path(assets.native_model_path),
        output_directory=output_directory,
        model_fingerprint=assets.model_fingerprint,
        coordinator_threads=4,
        worker_threads=3,
        timeout_seconds=timeout_seconds,
    )
    return rows


def run_native_local_reference(
    *,
    assets: DiscoveredAssets,
    output_directory: Path,
    repeat: int = 1,
    timeout_seconds: float = 7200,
) -> dict[str, Any]:
    candidate = RealCandidate(name="local_monolithic_reference")
    result = measure_reference(
        run_id=f"experiment-011-local-monolithic-r{repeat}",
        workload="network_profile_real_token_path",
        candidate=candidate,
        engine=Path(assets.native_engine),
        model_path=Path(assets.native_model_path),
        reference_path=Path(assets.experiment_010_workload_reference),
        run_root=output_directory / f"repeat-{repeat}",
        prompt_id="code-01",
        repeat=repeat,
        coordinator_threads=4,
        timeout_seconds=timeout_seconds,
        model_fingerprint=assets.model_fingerprint,
    )
    (output_directory / "local_monolithic_reference.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
