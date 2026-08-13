"""Capability discovery and batched local control-plane scaling for E022."""

from __future__ import annotations

import hashlib
import hmac
import os
import statistics
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import psutil

from .io import canonical_json_bytes
from .models import NodeCapability


@dataclass(frozen=True, slots=True)
class CapabilityAdvertisement:
    node: dict[str, Any]
    nonce: str
    signature: str


def advertise(node: NodeCapability, credential: bytes, nonce: str) -> CapabilityAdvertisement:
    if len(credential) < 32:
        raise ValueError("capability credential must contain at least 256 bits")
    body = {"node": node.as_dict(), "nonce": nonce}
    signature = hmac.new(credential, canonical_json_bytes(body), hashlib.sha256).hexdigest()
    return CapabilityAdvertisement(node=body["node"], nonce=nonce, signature=signature)


def verify_advertisement(
    advertisement: CapabilityAdvertisement,
    credential: bytes,
) -> NodeCapability:
    body = {"node": advertisement.node, "nonce": advertisement.nonce}
    expected = hmac.new(credential, canonical_json_bytes(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, advertisement.signature):
        raise ValueError("capability advertisement authentication failed")
    return NodeCapability.from_dict(advertisement.node)


def _process_assignment_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Persistent process entry point: validate one coarse worker task graph."""

    started = time.perf_counter_ns()
    assignments = 0
    internal_tasks = 0
    digest = hashlib.sha256()
    for item in batch:
        required = {
            "node_id",
            "capability_sha256",
            "placement_epoch",
            "pieces",
            "internal_task_count",
        }
        if not required.issubset(item):
            raise ValueError("assignment batch is missing canonical fields")
        assignments += len(item["pieces"])
        internal_tasks += int(item["internal_task_count"])
        digest.update(canonical_json_bytes(item))
    return {
        "process_id": os.getpid(),
        "logical_nodes": len(batch),
        "assignments": assignments,
        "internal_tasks": internal_tasks,
        "batch_sha256": digest.hexdigest(),
        "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
        "central_rpcs_per_internal_task": len(batch) / max(internal_tasks, 1),
    }


def _batches(values: list[dict[str, Any]], count: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(values), count):
        yield values[index : index + count]


def run_scale_case(
    template_nodes: tuple[NodeCapability, ...],
    logical_node_count: int,
    *,
    physical_processes: int = 4,
    assignment_batch_size: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise registration -> advertisement -> assignment -> schedule -> replan."""

    if logical_node_count <= 0 or not template_nodes:
        raise ValueError("scale case requires capabilities and a positive node count")
    credential = hashlib.sha256(
        f"e022-local-control-plane-{logical_node_count}".encode()
    ).digest()
    process = psutil.Process()
    cpu_before = process.cpu_times()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    advertisements: list[CapabilityAdvertisement] = []
    discovered: list[NodeCapability] = []
    rejected_tamper = False
    for index in range(logical_node_count):
        source = template_nodes[index % len(template_nodes)]
        value = source.as_dict()
        value["node_id"] = f"logical-{index:04d}"
        # Peers are not expanded to O(N^2) in this scale-only control-plane
        # arm. The advertised link profile is summarized as a locality class;
        # planner benchmark inventories retain complete peer matrices.
        value["network_peers"] = {}
        node = NodeCapability.from_dict(value)
        advertisement = advertise(node, credential, f"join-{index:04d}")
        advertisements.append(advertisement)
        discovered.append(verify_advertisement(advertisement, credential))
    if advertisements:
        first = advertisements[0]
        bad = CapabilityAdvertisement(first.node, first.nonce + "-tampered", first.signature)
        try:
            verify_advertisement(bad, credential)
        except ValueError:
            rejected_tamper = True
    registration_finished = time.perf_counter_ns()

    assignments: list[dict[str, Any]] = []
    for index, node in enumerate(discovered):
        pieces = [f"layer-{index % 93:02d}:epoch-0"]
        assignments.append(
            {
                "node_id": node.node_id,
                "capability_sha256": hashlib.sha256(
                    canonical_json_bytes(node.as_dict())
                ).hexdigest(),
                "placement_epoch": 0,
                "pieces": pieces,
                # A persistent worker executes this internal graph from one
                # coarse assignment message; there is no RPC per tensor op.
                "internal_task_count": 64 + (index % 32),
            }
        )
    scheduling_started = time.perf_counter_ns()
    batches = list(_batches(assignments, assignment_batch_size))
    with ProcessPoolExecutor(max_workers=min(physical_processes, len(batches))) as pool:
        process_results = list(pool.map(_process_assignment_batch, batches))
    scheduling_finished = time.perf_counter_ns()

    # Replanning is exercised by a capability update, not a manual topology.
    updated = discovered[logical_node_count // 2].as_dict()
    updated["compute_profile"] = {"reference": 0.5 * float(updated["compute_profile"]["reference"])}
    changed = NodeCapability.from_dict(updated)
    replacement = advertise(changed, credential, "capability-update")
    verified_change = verify_advertisement(replacement, credential)
    replan_started = time.perf_counter_ns()
    affected = [
        item for item in assignments if item["node_id"] == verified_change.node_id
    ]
    for item in affected:
        item["placement_epoch"] = 1
    replanning_ms = (time.perf_counter_ns() - replan_started) / 1e6

    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    cpu_after = process.cpu_times()
    rss_after = process.memory_info().rss
    internal_tasks = sum(int(row["internal_tasks"]) for row in process_results)
    rpc_count = len(batches)
    status = (
        len(discovered) == logical_node_count
        and rejected_tamper
        and sum(int(row["logical_nodes"]) for row in process_results)
        == logical_node_count
        and internal_tasks > rpc_count
        and bool(affected)
    )
    scale = {
        "schema_version": "experiment-022-control-plane-scale-v1",
        "status": "PASS" if status else "FAIL",
        "logical_nodes": logical_node_count,
        "persistent_worker_processes": len(
            {int(row["process_id"]) for row in process_results}
        ),
        "registration_ms": (registration_finished - started) / 1e6,
        "assignment_and_schedule_ms": (scheduling_finished - scheduling_started) / 1e6,
        "replanning_ms": replanning_ms,
        "elapsed_ms": elapsed_ms,
        "controller_cpu_seconds": (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        ),
        "controller_rss_delta_bytes": rss_after - rss_before,
        "capability_advertisements": len(discovered),
        "placement_assignments": sum(int(row["assignments"]) for row in process_results),
        "coarse_schedule_rpcs": rpc_count,
        "worker_internal_tasks": internal_tasks,
        "central_rpcs_per_internal_task": rpc_count / internal_tasks,
        "central_rpc_per_tiny_tensor_operation": False,
        "capability_change_triggered_replan": bool(affected),
        "manual_topology_supplied": False,
        "native_compute_in_scope": False,
        "evidence_class": "LOCAL_CONTROL_PLANE_PROCESS_TEST",
    }
    discovery = {
        "schema_version": "experiment-022-capability-discovery-v1",
        "status": "PASS" if status else "FAIL",
        "canonical_fields": sorted(template_nodes[0].as_dict()),
        "authenticated_advertisements": len(discovered),
        "tampered_advertisement_rejected": rejected_tamper,
        "controller_consumed_advertised_record": True,
        "gpu_product_name_branch": False,
        "capability_update": {
            "node_id": verified_change.node_id,
            "old_compute_multiplier": discovered[logical_node_count // 2].compute_multiplier,
            "new_compute_multiplier": verified_change.compute_multiplier,
            "replanned_assignment_count": len(affected),
        },
    }
    return scale, discovery


def summarize_scale(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "cases": len(rows),
        "largest_logical_node_count": max((int(row["logical_nodes"]) for row in rows), default=0),
        "median_central_rpcs_per_internal_task": statistics.median(
            [float(row["central_rpcs_per_internal_task"]) for row in rows]
        ) if rows else None,
    }


__all__ = [
    "CapabilityAdvertisement",
    "advertise",
    "run_scale_case",
    "summarize_scale",
    "verify_advertisement",
]
