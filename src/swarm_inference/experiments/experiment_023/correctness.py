"""Compatibility and exactness gates for Experiment 023."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.completion_inputs import (
    load_frozen_inventories,
)
from swarm_inference.experiments.experiment_022.evaluator import (
    PlacementEvaluator,
    common_endpoint_policy,
)
from swarm_inference.experiments.experiment_022.io import (
    atomic_write_json,
    sha256_file,
    write_csv,
)
from swarm_inference.experiments.experiment_022.manifest_correctness import (
    ManifestK3Runner,
    execute_manifest,
)
from swarm_inference.experiments.experiment_022.model_graph import build_model_graph
from swarm_inference.experiments.experiment_022.service import ResidentServiceModel

from .baseline import clone_placement, load_frozen_placement
from .freeze import FROZEN_CONSTANTS
from .models import E023Plan, NetworkMode
from .repair_validation import validate_repair_inputs_read_only
from .serving_engine import ServingEngine

COMPATIBILITY_REPRESENTATIVES = (
    ("coarse-friendly-01", "A"),
    ("memory-fragmented-03", "B"),
    ("compute-heterogeneous-04", "A"),
    ("network-heterogeneous-01", "D"),
    ("full-mixed-01", "E"),
)
FULL_CORRECTNESS_REPRESENTATIVES = (
    "memory-fragmented-03",
    "compute-heterogeneous-04",
    "network-heterogeneous-01",
    "full-mixed-02",
    "coarse-friendly-01",
)


def _float_equal(left: float, right: float, tolerance: float = 1e-9) -> bool:
    absolute = abs(left - right)
    scale = max(abs(left), abs(right))
    relative = absolute / scale if scale else 0.0
    return absolute <= tolerance or relative <= tolerance


def run_engine_compatibility(repo: Path) -> list[dict[str, Any]]:
    """Require the interval engine's legacy/no-replica path to reproduce E022."""

    root = repo.resolve()
    validate_repair_inputs_read_only(root)
    inventories, _audit = load_frozen_inventories(root)
    inventory_map = {value.inventory_id: value for value in inventories}
    model = build_model_graph(
        Path("F:/models/Kimi-K3"),
        whole_layer_service_csv=root
        / "artifacts/experiment-018/physical/layer-service.csv",
    )
    service = ResidentServiceModel.from_csv(
        root
        / "artifacts/experiment-022/completion/validation/repaired-resident-service.csv"
    )
    rows: list[dict[str, Any]] = []
    placement_root = root / "artifacts/experiment-022/completion/rerun/placements"
    for inventory_id, planner_level in COMPATIBILITY_REPRESENTATIVES:
        inventory = inventory_map[inventory_id]
        endpoint = common_endpoint_policy(model, inventory)
        if endpoint is None:
            raise RuntimeError(f"MODEL_INVALID: endpoint infeasible for {inventory_id}")
        plan = load_frozen_placement(
            placement_root / f"{inventory_id}-{planner_level}.json", inventory
        )
        e022_plan = clone_placement(plan)
        old = PlacementEvaluator(model, inventory, service, endpoint).evaluate(e022_plan)
        new = ServingEngine(
            model,
            inventory,
            service,
            endpoint,
            network_mode=NetworkMode.LEGACY_DIRECTED_LINK,
        ).run_single_pass(E023Plan(inventory_id, "U_STRONG", clone_placement(plan)))
        network_equal = new.network_bytes == old.total_network_bytes
        message_equal = new.messages == old.messages
        event_equal = new.event_count == len(old.records)
        worker_equal = set(new.participating_workers) == set(old.worker_utilization)
        critical_equal = _float_equal(new.critical_path_ms, old.makespan_ms)
        compute_equal = _float_equal(
            new.total_worker_compute_ms, old.total_compute_ms
        )
        network_time_equal = _float_equal(
            new.total_network_ms, old.total_network_ms
        )
        passed = all(
            (
                network_equal,
                message_equal,
                event_equal,
                worker_equal,
                critical_equal,
                compute_equal,
                network_time_equal,
            )
        )
        rows.append(
            {
                "case_id": f"{inventory_id}-{planner_level}",
                "inventory_id": inventory_id,
                "planner_level": planner_level,
                "network_mode": NetworkMode.LEGACY_DIRECTED_LINK.value,
                "concurrency": 1,
                "target_passes": 1,
                "replicas": 0,
                "e022_critical_path_ms": old.makespan_ms,
                "e023_critical_path_ms": new.critical_path_ms,
                "critical_path_absolute_error_ms": abs(
                    new.critical_path_ms - old.makespan_ms
                ),
                "e022_total_worker_compute_ms": old.total_compute_ms,
                "e023_total_worker_compute_ms": new.total_worker_compute_ms,
                "worker_compute_absolute_error_ms": abs(
                    new.total_worker_compute_ms - old.total_compute_ms
                ),
                "e022_total_network_ms": old.total_network_ms,
                "e023_total_network_ms": new.total_network_ms,
                "network_time_absolute_error_ms": abs(
                    new.total_network_ms - old.total_network_ms
                ),
                "e022_network_bytes": old.total_network_bytes,
                "e023_network_bytes": new.network_bytes,
                "e022_messages": old.messages,
                "e023_messages": new.messages,
                "e022_event_count": len(old.records),
                "e023_event_count": new.event_count,
                "e022_participating_workers": len(old.worker_utilization),
                "e023_participating_workers": len(new.participating_workers),
                "network_bytes_equal": network_equal,
                "messages_equal": message_equal,
                "event_count_equal": event_equal,
                "participating_workers_equal": worker_equal,
                "timing_equal_within_1e_9": (
                    critical_equal and compute_equal and network_time_equal
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )
    write_csv(
        root / "artifacts/experiment-023/validation/engine-compatibility.csv",
        rows,
    )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("MODEL_INVALID: E023 legacy engine compatibility failed")
    return rows


class ReplicaAwareManifestK3Runner(ManifestK3Runner):
    """Force exact alternate destinations while preserving logical ownership."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        manifest = args[3] if len(args) >= 4 else kwargs["manifest"]
        self.replica_destinations = {
            (int(row["layer_id"]), int(row["logical_group_id"])): str(
                row["alternate_node_id"]
            )
            for row in manifest.get("replicas", ())
        }
        self.primary_dispatch_count = 0
        self.alternate_dispatch_count = 0
        self.forced_alternate_dispatch_count = 0
        self.replica_dispatch_decisions: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def physical_expert_worker_id(
        self,
        *,
        layer: int,
        logical_group_id: int,
        chunk_index: int,
        logical_worker_id: str,
    ) -> str:
        alternate = self.replica_destinations.get((layer, logical_group_id))
        use_alternate = (
            alternate is not None
            and (layer + logical_group_id + chunk_index) % 2 == 1
        )
        destination = alternate if use_alternate else logical_worker_id
        if use_alternate:
            self.alternate_dispatch_count += 1
            self.forced_alternate_dispatch_count += 1
        else:
            self.primary_dispatch_count += 1
        self.replica_dispatch_decisions.append(
            {
                "layer_id": layer,
                "logical_group_id": logical_group_id,
                "chunk_index": chunk_index,
                "logical_primary_worker_id": logical_worker_id,
                "physical_worker_destination": destination,
                "alternate_available": alternate is not None,
                "use_alternate": use_alternate,
            }
        )
        return destination


def _execute_full_manifest(arguments: dict[str, str]) -> dict[str, Any]:
    manifest_path = Path(arguments["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    holders: list[ReplicaAwareManifestK3Runner] = []

    def factory(*args: Any, **kwargs: Any) -> ReplicaAwareManifestK3Runner:
        runner = ReplicaAwareManifestK3Runner(*args, **kwargs)
        holders.append(runner)
        return runner

    receipt = execute_manifest(
        selection_id=arguments["inventory_id"],
        selection_case=f"E023 final FLEX_POOL {arguments['inventory_id']}",
        manifest_path=manifest_path,
        expected_manifest_sha256=sha256_file(manifest_path),
        checkpoint=Path(arguments["checkpoint"]),
        cuda_library=Path(arguments["cuda_library"]),
        grouped_library=Path(arguments["grouped_library"]),
        shard_library=Path(arguments["shard_library"]),
        oracle_root=Path(arguments["oracle_root"]),
        state_reference_root=Path(arguments["state_reference_root"]),
        runner_factory=factory,
    )
    runner = holders[0]
    replica_count = len(manifest.get("replicas", ()))
    forced = runner.forced_alternate_dispatch_count
    receipt.update(
        {
            "schema_version": "experiment-023-replica-aware-correctness-v1",
            "experiment_id": "023",
            "authoritative_attempt": arguments["primary_attempt"],
            "replica_count_in_plan": replica_count,
            "forced_alternate_dispatch_count": forced,
            "primary_dispatch_count": runner.primary_dispatch_count,
            "alternate_dispatch_count": runner.alternate_dispatch_count,
            "forced_alternate_uses": runner.replica_dispatch_decisions,
            "complete_93_layer_traversal_executed": receipt[
                "complete_93_layer_traversal"
            ],
            "kda_state_reconciliation_exact": receipt[
                "state_reference_validation"
            ]["status"]
            == "PASS",
            "mla_state_reconciliation_exact": receipt[
                "state_reference_validation"
            ]["status"]
            == "PASS",
            "attnres_fingerprint_reconciliation_exact": receipt[
                "state_reference_validation"
            ]["status"]
            == "PASS",
        }
    )
    additional_pass = (
        receipt["status"] == "PASS"
        and receipt["complete_93_layer_traversal"]
        and receipt["complete_tensor_assignment_coverage"]
        and receipt["authenticated_execute_shard"]
        and not receipt["whole_layer_fallback_for_split_layers"]
        and receipt["checkpoint_reads_in_expert_timed_regions"] == 0
        and receipt["route_equality"]
        and receipt["ordered_expert_equality"]
        and receipt["all_state_finite"]
        and receipt["kda_state_reconciliation_exact"]
        and receipt["mla_state_reconciliation_exact"]
        and receipt["attnres_fingerprint_reconciliation_exact"]
        and receipt["hidden_relative_l2_maximum"]
        <= FROZEN_CONSTANTS["whole_expert_relative_l2_gate"]
        and receipt["logit_relative_l2"]
        <= FROZEN_CONSTANTS["whole_expert_relative_l2_gate"]
        and receipt["greedy_token_equality"]
        and (replica_count == 0 or forced > 0)
    )
    receipt["status"] = "PASS" if additional_pass else "FAIL"
    return receipt


def _full_correctness_worker(arguments: dict[str, str], output: str) -> None:
    try:
        receipt = _execute_full_manifest(arguments)
    except BaseException as exc:
        receipt = {
            "schema_version": "experiment-023-replica-aware-correctness-v1",
            "experiment_id": "023",
            "inventory_id": arguments["inventory_id"],
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    atomic_write_json(Path(output), receipt)


def run_full_correctness(
    repo: Path,
    *,
    primary_attempt: str = "deterministic-run-2",
    timeout_seconds: float = 14_400.0,
) -> list[dict[str, Any]]:
    """Execute the five frozen final FLEX_POOL manifests through E022's runner."""

    root = repo.resolve()
    validate_repair_inputs_read_only(root)
    paths = {
        "checkpoint": Path("F:/models/Kimi-K3"),
        "cuda_library": root
        / "artifacts/experiment-016/cuda/coli_cuda-sm120-h016-final.dll",
        "shard_library": root
        / "artifacts/experiment-019/physical/exp019-kda-shard-sm120-v2.dll",
        "grouped_library": root
        / "artifacts/experiment-020/physical/e020-grouped-top16-sm120.dll",
        "oracle_root": root / "artifacts/experiment-014/oracle-full-93-idot0",
        "state_reference_root": root
        / "artifacts/experiment-022/completion/correctness/"
        "state-reference-representative-01",
    }
    if any(not path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise RuntimeError("MODEL_INVALID: correctness input missing: " + ",".join(missing))
    output_root = root / "artifacts/experiment-023/correctness"
    receipts: list[dict[str, Any]] = []
    spawn = mp.get_context("spawn")
    for inventory_id in FULL_CORRECTNESS_REPRESENTATIVES:
        manifest = (
            root
            / "artifacts/experiment-023/attempts"
            / primary_attempt
            / "plans"
            / inventory_id
            / "FLEX_POOL.json"
        )
        if not manifest.is_file():
            raise RuntimeError(
                f"MODEL_INVALID: final FLEX_POOL manifest missing for {inventory_id}"
            )
        output = output_root / f"{inventory_id}.json"
        arguments = {
            "inventory_id": inventory_id,
            "primary_attempt": primary_attempt,
            "manifest": str(manifest.resolve()),
            **{name: str(path.resolve()) for name, path in paths.items()},
        }
        process = spawn.Process(
            target=_full_correctness_worker,
            args=(arguments, str(output.resolve())),
            name=f"e023-{inventory_id}-replica-correctness",
        )
        started = time.perf_counter()
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
            raise TimeoutError(f"MODEL_INVALID: correctness timed out for {inventory_id}")
        if not output.is_file():
            raise RuntimeError(
                f"MODEL_INVALID: correctness receipt missing for {inventory_id}"
            )
        receipt = json.loads(output.read_text(encoding="utf-8"))
        receipt["worker_process"] = {
            "pid": process.pid,
            "exitcode": process.exitcode,
            "spawn_method": "spawn",
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_write_json(output, receipt)
        receipts.append(receipt)
        if receipt.get("status") != "PASS":
            raise RuntimeError(
                f"MODEL_INVALID: full correctness failed for {inventory_id}"
            )
    return receipts


__all__ = [
    "COMPATIBILITY_REPRESENTATIVES",
    "FULL_CORRECTNESS_REPRESENTATIVES",
    "ReplicaAwareManifestK3Runner",
    "run_engine_compatibility",
    "run_full_correctness",
]
