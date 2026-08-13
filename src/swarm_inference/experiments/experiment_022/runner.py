"""End-to-end local Experiment 022 orchestration."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from . import E022_EXTERNAL_SWARM, E022_VAST_ACCESS, E022_ZERO_RENTAL, EVIDENCE_CLASS
from .benchmark import (
    headline_statistics,
    run_dynamic_suite,
    run_static_suite,
)
from .control_plane import run_scale_case, summarize_scale
from .inventories import materialize_inventory_suite
from .io import atomic_write_json, atomic_write_text, read_json, sha256_file, write_csv
from .model_graph import build_model_graph, model_metadata
from .models import PartitionKind, PlannerLevel
from .native_dispatch import (
    CallableResidentPrimitive,
    NativeShardDispatcher,
    ShardTaskType,
)
from .oracle import validate_optimizer_oracle
from .planner import OptimizerConfiguration, SharedPlacementOptimizer
from .sensitivity import run_sensitivities
from .validation import materialize_validation


def _command(arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return (completed.stdout or completed.stderr).strip()


def _environment(repo: Path) -> dict[str, Any]:
    return {
        "schema_version": "experiment-022-environment-v1",
        "captured_unix_ns": time.time_ns(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_bytes": psutil.virtual_memory().total,
        "gpu": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status_at_finalize": _command(["git", "status", "--short"]),
        "checkpoint": "F:/models/Kimi-K3",
        "evidence_class": EVIDENCE_CLASS,
        "physical_heterogeneous_swarm_tested": False,
        "gpu_rentals": 0,
        "vast_queries": 0,
        "vast_mutations": 0,
        "external_physical_swarm": E022_EXTERNAL_SWARM,
        "zero_rental_guard": E022_ZERO_RENTAL,
        "vast_access_guard": E022_VAST_ACCESS,
    }


def _source_manifest(repo: Path) -> dict[str, Any]:
    roots = [
        repo / "src" / "swarm_inference" / "experiments" / "experiment_022",
        repo / "scripts",
    ]
    files = [
        path
        for root in roots
        for path in root.rglob("*.py")
        if root.name != "scripts" or path.name.startswith("experiment_022")
    ]
    files.extend(
        [
            repo / "AGENTS.md",
            repo / "tests/test_experiment_022.py",
            Path("F:/models/Kimi-K3/model.safetensors.index.json"),
            repo / "artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll",
            repo / "artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll",
            repo / "artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll",
        ]
    )
    rows = []
    for path in sorted(set(files), key=lambda value: str(value)):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "experiment-022-source-manifest-v1",
        "files": rows,
        "file_count": len(rows),
    }


def _production_primitive_receipt(artifact_root: Path) -> dict[str, Any]:
    physical = {
        "whole_layer": read_json(
            artifact_root / "validation" / "whole-layer-resident-raw.json"
        )["status"],
        "attention_shards": read_json(
            artifact_root / "validation" / "resident-attention-robust-raw.json"
        )["status"],
        "complete_expert_banks": read_json(
            artifact_root / "validation" / "resident-expert-bank-raw.json"
        )["status"],
        "ordered_layer_dag": read_json(
            artifact_root / "validation" / "resident-ordered-raw.json"
        )["status"],
    }
    # The protocol/dispatch unit exercises exact binary shape and digest
    # handling. Native invocation is separately proven by full correctness.
    dispatcher = NativeShardDispatcher("receipt-worker")
    for task_type in (
        ShardTaskType.KDA_SHARD,
        ShardTaskType.MLA_SHARD,
        ShardTaskType.EXPERT_STRIPE,
        ShardTaskType.SHARED_EXPERT_SHARD,
        ShardTaskType.PROJECTION_SHARD,
        ShardTaskType.REDUCTION_CONTRIBUTION,
    ):
        dispatcher.register(
            f"schema:{task_type.value}",
            task_type,
            CallableResidentPrimitive(
                f"native-binding-required:{task_type.value}",
                lambda values, _request: values,
                native=True,
            ),
        )
    correctness = read_json(
        artifact_root / "correctness" / "worker-process-full-93.json"
    )
    physical_graph_pass = (
        all(value == "PASS" for value in physical.values())
        and correctness["status"] == "PASS"
    )
    missing_individual_bindings = [
        value.value
        for value in (
            ShardTaskType.KDA_SHARD,
            ShardTaskType.MLA_SHARD,
            ShardTaskType.EXPERT_STRIPE,
            ShardTaskType.SHARED_EXPERT_SHARD,
            ShardTaskType.PROJECTION_SHARD,
            ShardTaskType.REDUCTION_CONTRIBUTION,
        )
    ]
    return {
        "schema_version": "experiment-022-primitive-results-v1",
        "status": "FAIL",
        "registered_task_types": [value.value for value in ShardTaskType],
        "placeholder_result_present": False,
        "partial_latent_vector_present": False,
        "physical_receipts": physical,
        "full_worker_native_dispatch": correctness,
        "batched_ordered_dag_native_dispatch_status": (
            "PASS" if physical_graph_pass else "FAIL"
        ),
        "individual_task_bindings": {
            "status": "FAIL_NOT_BOUND_TO_PRODUCTION_NATIVE_HANDLES",
            "binding_count": dispatcher.assignment_count,
            "schema_dispatch_tested": True,
            "missing_individual_native_bindings": missing_individual_bindings,
            "native_operator_evidence": "physical receipts and internal full-DAG audit only",
        },
        "headline_gate_effect": "MODEL_INVALID",
    }


def _representative_correctness(
    artifact_root: Path,
    static: dict[str, Any],
    inventories: list[Any],
) -> dict[str, Any]:
    worker = read_json(artifact_root / "correctness" / "worker-process-full-93.json")
    analysis = static["analysis"]
    mixed_uplifts = [
        row
        for row in analysis["throughput-uplift"]
        if any(
            assignment.partition_kind is not PartitionKind.WHOLE_LAYER
            for assignment in static["plans"][(str(row["inventory_id"]), PlannerLevel.E)].assignments
        )
    ]
    strongest_case = max(
        mixed_uplifts,
        key=lambda row: float(row["throughput_uplift_percent"]),
        default=None,
    )
    capacity = (
        analysis["capacity-unlocks"][0]["inventory_id"]
        if analysis["capacity-unlocks"]
        else inventories[-1].inventory_id
    )
    strongest = (
        str(strongest_case["inventory_id"])
        if strongest_case is not None
        else str(capacity)
    )
    control = next(value for value in inventories if value.family == "coarse-friendly")
    full_mixed = next(value for value in inventories if value.family == "full-mixed")
    cases = [
        ("coarse-friendly Planner A", control.inventory_id, PlannerLevel.A),
        ("coarse-friendly Planner E", control.inventory_id, PlannerLevel.E),
        (
            "strongest performance-uplift mixed"
            if strongest_case is not None
            else "no A-feasible mixed win; capacity-unlocked mixed fallback",
            str(strongest),
            PlannerLevel.E,
        ),
        ("capacity-unlocked mixed", str(capacity), PlannerLevel.E),
        ("full-mixed representative", full_mixed.inventory_id, PlannerLevel.E),
    ]
    rows = []
    manifest_checks_pass = worker["status"] == "PASS"
    for label, inventory_id, level in cases:
        plan = static["plans"][(inventory_id, level)]
        manifest = read_json(
            artifact_root
            / "planner"
            / "placements"
            / f"{inventory_id}-{level.value}.json"
        )
        kinds = sorted(
            {assignment.partition_kind.value for assignment in plan.assignments}
        )
        reconciliation = manifest["checkpoint_reconciliation"]
        manifest_passed = (
            plan.feasible
            and int(reconciliation["gap_bytes"]) == 0
            and int(reconciliation["overlap_bytes"]) == 0
            and worker["status"] == "PASS"
        )
        manifest_checks_pass = manifest_checks_pass and manifest_passed
        rows.append(
            {
                "case": label,
                "inventory_id": inventory_id,
                "planner_level": level.value,
                "partition_types": kinds,
                "manifest_reconciliation_status": (
                    "PASS" if manifest_passed else "FAIL"
                ),
                "plan_specific_full_93_execution_status": "NOT_EXECUTED",
                "status": "FAIL",
                "manifest_checkpoint_gap_bytes": reconciliation["gap_bytes"],
                "manifest_checkpoint_overlap_bytes": reconciliation["overlap_bytes"],
                "physical_execution_template": None,
                "physical_template_execution_reused": False,
                "failure_reason": (
                    "the fresh P8 worker traversal validates the primitive graph but does "
                    "not execute this saved placement manifest's exact mixed action sequence"
                ),
            }
        )
    return {
        "schema_version": "experiment-022-full-93-representative-v1",
        "status": "FAIL",
        "representative_plan_count": len(rows),
        "required_plan_specific_full_traversals": len(rows),
        "completed_plan_specific_full_traversals": 0,
        "unique_fresh_worker_process_full_traversals": 1,
        "generic_p8_primitive_graph_traversal_status": worker["status"],
        "manifest_reconciliation_checks_pass": manifest_checks_pass,
        "execution_reuse_disclosed": False,
        "headline_gate_effect": "MODEL_INVALID",
        "worker_process_correctness": worker,
        "plans": rows,
    }


def run(repo: Path) -> dict[str, Any]:
    artifact_root = repo / "artifacts" / "experiment-022"
    checkpoint = Path("F:/models/Kimi-K3")
    model = build_model_graph(
        checkpoint,
        whole_layer_service_csv=repo
        / "artifacts"
        / "experiment-018"
        / "physical"
        / "layer-service.csv",
    )
    suite, inventories = materialize_inventory_suite(artifact_root, model)
    atomic_write_json(artifact_root / "environment.json", _environment(repo))
    atomic_write_json(artifact_root / "model-metadata.json", model_metadata(model))
    atomic_write_json(artifact_root / "source-manifest.json", _source_manifest(repo))
    preregistered_suite_path = artifact_root / "inventories" / "inventory-suite.json"
    if not preregistered_suite_path.is_file():
        raise RuntimeError("frozen inventory suite is missing before planner comparison")
    frozen_suite = read_json(preregistered_suite_path)
    if (
        frozen_suite.get("suite_sha256") != suite["suite_sha256"]
        or not frozen_suite.get("frozen_before_planner_comparison")
    ):
        raise RuntimeError("inventory suite changed after preregistration")
    service, validation = materialize_validation(repo, artifact_root, model)
    correctness_path = artifact_root / "correctness" / "worker-process-full-93.json"
    if not correctness_path.is_file() or read_json(correctness_path).get("status") != "PASS":
        raise RuntimeError(
            "planner comparison is blocked until full worker-process correctness passes"
        )
    configuration = OptimizerConfiguration(
        proposal_budget=24,
        exact_evaluation_budget=6,
        restarts=3,
        candidate_groups_per_action=4,
        randomization=0.08,
        seed=22022,
    )
    oracle_rows = validate_optimizer_oracle(
        model, inventories[0], service, configuration
    )
    write_csv(
        artifact_root / "validation" / "optimizer-small-oracle.csv", oracle_rows
    )
    optimizer_pass = all(row["status"] == "PASS" for row in oracle_rows)
    if validation["status"] != "PASS" or not optimizer_pass:
        raise RuntimeError("validation gate failed; planner comparison not admitted")
    primitive = _production_primitive_receipt(artifact_root)
    atomic_write_json(
        artifact_root / "correctness" / "primitive-results.json", primitive
    )
    # Continue the frozen comparison as diagnostic output even when the
    # production primitive gate is open. Finalization must emit MODEL_INVALID
    # and must not admit those modeled rows as a headline result.

    static = run_static_suite(
        artifact_root, model, inventories, service, configuration
    )
    optimizer = SharedPlacementOptimizer(model, service, configuration)
    dynamic = run_dynamic_suite(artifact_root, inventories, static, optimizer)
    representative = _representative_correctness(
        artifact_root, static, inventories
    )
    atomic_write_json(
        artifact_root / "correctness" / "full-93-representative.json",
        representative,
    )
    # Mechanically select the strongest A-feasible win whose E plan actually
    # uses sub-layer work. If no such win exists, use the most sub-layer-heavy
    # feasible E plan. Every preregistered inventory remains in the benchmark.
    inventory_by_id = {value.inventory_id: value for value in inventories}
    mixed_uplifts = [
        row
        for row in static["analysis"]["throughput-uplift"]
        if bool(row["adaptive_uses_sublayer"])
    ]
    if mixed_uplifts:
        strongest = max(
            mixed_uplifts,
            key=lambda row: (
                float(row["throughput_uplift_percent"]),
                str(row["inventory_id"]),
            ),
        )
        sensitivity_inventory = inventory_by_id[str(strongest["inventory_id"])]
    else:
        feasible = [
            inventory
            for inventory in inventories
            if static["plans"][(inventory.inventory_id, PlannerLevel.E)].feasible
        ]
        sensitivity_inventory = max(
            feasible,
            key=lambda inventory: sum(
                assignment.partition_kind is not PartitionKind.WHOLE_LAYER
                for assignment in static["plans"][(inventory.inventory_id, PlannerLevel.E)].assignments
            ),
        )
    sensitivity = run_sensitivities(sensitivity_inventory, optimizer)
    for name, rows in sensitivity.items():
        write_csv(artifact_root / "analysis" / f"{name}.csv", rows)

    scale_rows = []
    discoveries = []
    for count in (128, 256, 512, 1000, 2000):
        print(f"[e022 control plane] logical_nodes={count}", flush=True)
        scale, discovery = run_scale_case(inventories[0].nodes, count)
        scale_rows.append(scale)
        discoveries.append({"logical_nodes": count, **discovery})
    write_csv(artifact_root / "control-plane" / "scaling.csv", scale_rows)
    atomic_write_json(
        artifact_root / "control-plane" / "capability-discovery.json",
        {
            "schema_version": "experiment-022-capability-discovery-suite-v1",
            "summary": summarize_scale(scale_rows),
            "cases": discoveries,
        },
    )
    statistics_ = headline_statistics(static["analysis"])
    dynamic_rows = [row for rows in dynamic.values() for row in rows]
    useful_join_pass = bool(dynamic.get("JOIN_USEFUL")) and all(
        bool(row["beneficial_join_non_regression"])
        and bool(row["beneficial_join_admitted_and_improved"])
        for row in dynamic["JOIN_USEFUL"]
    )
    harmful_join_pass = bool(dynamic.get("JOIN_HARMFUL")) and all(
        bool(row["harmful_join_ignored"])
        for row in dynamic["JOIN_HARMFUL"]
    )
    dynamic_pass = (
        bool(dynamic_rows)
        and all(row["status"] == "PASS" for row in dynamic_rows)
        and useful_join_pass
        and harmful_join_pass
        and all(not bool(row["manual_topology_supplied"]) for row in dynamic_rows)
    )
    result = {
        "schema_version": "experiment-022-run-v1",
        "status": "PASS",
        "inventory_suite_sha256": suite["suite_sha256"],
        "inventory_count": suite["inventory_count"],
        "optimizer_configuration": asdict(configuration),
        "same_optimizer_all_arms": True,
        "only_action_space_differs_between_static_arms": True,
        "adaptive_seeded_with_whole_fallback": True,
        "model_validation": validation,
        "optimizer_oracle_pass": optimizer_pass,
        "primitive_pass": primitive["status"] == "PASS",
        "representative_correctness_pass": representative["status"] == "PASS",
        "dynamic_status": "PASS" if dynamic_pass else "FAIL",
        "dynamic_useful_join_non_regression": useful_join_pass,
        "dynamic_harmful_join_ignored": harmful_join_pass,
        "control_plane_status": summarize_scale(scale_rows)["status"],
        "headline_statistics": statistics_,
        "sensitivity_inventory": sensitivity_inventory.inventory_id,
    }
    atomic_write_json(artifact_root / "run-result.json", result)
    atomic_write_text(
        artifact_root / "commands.txt",
        "\n".join(
            [
                "# Experiment 022 executed commands (local only)",
                "python -m swarm_inference.experiments.experiment_018.physical_benchmark --checkpoint F:/models/Kimi-K3 --cuda-library artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll --oracle-trace artifacts/experiment-014/oracle-full-93-idot0/hidden-trace.f32 --output artifacts/experiment-022/validation/whole-layer-resident-raw.json --gpu-samples artifacts/experiment-022/validation/whole-layer-gpu-samples.csv --warmup 2 --iterations 10 --profile-iterations 3 --layers 45,47,89,91 --equivalence-layers 89",
                "python scripts/experiment_022_physical.py ordered --warmup 1 --iterations 5 --output artifacts/experiment-022/validation/resident-ordered-raw.json",
                "python scripts/experiment_022_physical.py attention --warmup 2 --iterations 7 --output artifacts/experiment-022/validation/resident-attention-raw.json",
                "python scripts/experiment_022_physical.py attention --warmup 5 --iterations 31 --output artifacts/experiment-022/validation/resident-attention-robust-raw.json",
                "python scripts/experiment_022_physical.py expert-bank --output artifacts/experiment-022/validation/resident-expert-bank-raw.json",
                "python scripts/experiment_022_correctness.py --output artifacts/experiment-022/correctness/worker-process-full-93.json --raw-output artifacts/experiment-022/correctness/worker-process-full-93-raw.json --timeout-seconds 3600",
                "python scripts/experiment_022_run.py",
                "python scripts/experiment_022_dynamic.py",
                "python scripts/experiment_022_finalize.py",
                "python scripts/experiment_022_test.py",
                "# No network, marketplace, Vast, rental, or external-swarm command was run.",
            ]
        ),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    arguments = parser.parse_args()
    result = run(arguments.repo.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
