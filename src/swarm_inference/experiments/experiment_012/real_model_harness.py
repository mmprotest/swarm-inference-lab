"""H012-012 real-model vocabulary-head delegation experiment harness."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from swarm_inference.experiments.experiment_012.baseline_harness import (
    WorkerPool,
    WorkerProcess,
    _append_jsonl,
    _sha256_file,
    _source_identity,
    _write_json,
)
from swarm_inference.experiments.experiment_012.delegation_harness import (
    DelegatedNode,
    DelegatedTopology,
    build_delegated_topology,
    install_topology,
    topology_record,
)
from swarm_inference.microworker_protocol import (
    NETWORK_PROFILES,
    LinkProfile,
    PersistentChannel,
    aggregate_digest,
    combine_operation_aggregates,
    make_delegated_request,
    shaped_round_trip,
)
from swarm_inference.model.shard_builder import (
    inspect_native_model,
    model_inspection_payload,
    resolve_model,
)

HYPOTHESIS_ID = "H012-012"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SAFETENSORS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
MICROSHARD_MANIFEST_SHA256 = "c8d0f419f6779019ad4ca2c277fce9c3810d9e457434cc41c49293822a71d14b"
PROMPT = "Paris is the capital of France. Paris is the capital of France. Paris is the capital of"
WORKER_COUNT = 8
BRANCH_FACTOR = 2
AGGREGATION = {
    "mode": "vocabulary_argmax",
    "ordering": "maximum_score_then_lowest_token_id",
}


def _flat_topology(workers: list[WorkerProcess], *, generation: int) -> DelegatedTopology:
    nodes = tuple(
        DelegatedNode(
            worker=worker,
            parent_worker="stage-owner",
            partition_start=worker.worker_index,
            partition_end=worker.worker_index + 1,
            partition_worker_indices=(worker.worker_index,),
            subtree_worker_count=1,
        )
        for worker in sorted(workers, key=lambda item: item.worker_index)
    )
    return DelegatedTopology(
        topology_id="h012-012-flat-real-output-head",
        route_lease_id="h012-012-flat-lease",
        route_generation=generation,
        branch_factor=WORKER_COUNT,
        root_children=nodes,
        nodes=nodes,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


class RealArgmaxRunner:
    """Run one real hidden vector through flat or delegated output-head workers."""

    def __init__(
        self,
        *,
        topology: DelegatedTopology,
        output_directory: Path,
        workload: dict[str, Any],
        operation_deadline_s: float,
        connection_policy: str,
        mode: str,
    ) -> None:
        self.topology = topology
        self.output_directory = output_directory
        self.workload = workload
        self.operation_deadline_s = operation_deadline_s
        self.connection_policy = connection_policy
        self.mode = mode
        self.trace_path = output_directory / "traces" / "root.jsonl"
        self.observation_path = output_directory / "raw" / "measurements.jsonl"
        self.trace_lock = threading.Lock()
        self.channels_lock = threading.Lock()
        self.channels: dict[str, PersistentChannel] = {}

    def _channel(self, node: DelegatedNode) -> PersistentChannel:
        with self.channels_lock:
            channel = self.channels.get(node.worker.worker_id)
            if channel is None:
                channel = PersistentChannel(
                    node.worker.endpoint, "stage-owner", node.worker.worker_id
                )
                self.channels[node.worker.worker_id] = channel
            return channel

    def close(self) -> None:
        with self.channels_lock:
            channels = list(self.channels.values())
            self.channels.clear()
        for channel in channels:
            channel.close()

    def _root_child(
        self,
        *,
        node: DelegatedNode,
        operation_id: str,
        execution_generation: int,
        payload_b64: str,
        payload_bytes: int,
        deadline_unix_ns: int,
        profile: LinkProfile,
    ) -> tuple[DelegatedNode, dict[str, Any], dict[str, Any]]:
        request_id = f"{operation_id}:{node.worker.worker_id}"
        request = make_delegated_request(
            request_id=request_id,
            operation_id=operation_id,
            execution_generation=execution_generation,
            parent_worker="stage-owner",
            worker_id=node.worker.worker_id,
            worker_index=node.worker.worker_index,
            assigned_child_workers=[child.worker.worker_id for child in node.children],
            partition_start=node.partition_start,
            partition_end=node.partition_end,
            partition_worker_indices=list(node.partition_worker_indices),
            subtree_worker_count=node.subtree_worker_count,
            deadline_unix_ns=deadline_unix_ns,
            route_lease_id=self.topology.route_lease_id,
            ordering_key=node.ordering_key,
            payload_bytes=payload_bytes,
            payload_b64=payload_b64,
            trace_id=operation_id,
            parent_span_id="root",
            span_id=f"root-to-{node.worker.worker_id}",
            network_profile=profile,
            dispatch_policy="parallel",
            connection_policy=self.connection_policy,
            route_generation=self.topology.route_generation,
            retry_policy={"max_attempts": 0, "backoff_ms": 0.0},
            aggregation=AGGREGATION,
            workload=self.workload,
        )
        timeout_s = max(0.001, (deadline_unix_ns - time.time_ns()) / 1_000_000_000)
        if self.connection_policy == "persistent":
            response, transport = self._channel(node).round_trip(
                message=request, profile=profile, timeout_s=timeout_s, attempt=0
            )
        else:
            response, transport = shaped_round_trip(
                endpoint=node.worker.endpoint,
                message=request,
                profile=profile,
                timeout_s=timeout_s,
                sender_id="stage-owner",
                receiver_id=node.worker.worker_id,
                attempt=0,
            )
        if (
            response.get("request_id") != request_id
            or response.get("operation_id") != operation_id
            or int(response.get("execution_generation", 0)) != execution_generation
        ):
            raise RuntimeError("root received a stale real-model response")
        aggregate = dict(response["aggregate"])
        if response.get("aggregate_digest") != aggregate_digest(aggregate):
            raise RuntimeError("root received a corrupt real-model aggregate")
        _append_jsonl(
            self.trace_path,
            {
                "event": "root_real_model_round_trip",
                "hypothesis_id": HYPOTHESIS_ID,
                "mode": self.mode,
                "operation_id": operation_id,
                "request_id": request_id,
                "sender_worker_id": "stage-owner",
                "receiver_worker_id": node.worker.worker_id,
                "sender_process_id": os.getpid(),
                "receiver_process_id": node.worker.process_id,
                "receiver_is_leaf": not node.children,
                "aggregation": AGGREGATION["mode"],
                "timestamp_unix_ns": time.time_ns(),
                **transport,
            },
            self.trace_lock,
        )
        return node, response, transport

    def run(
        self,
        *,
        hidden_path: Path,
        reference_token_id: int,
        step_index: int,
        phase: str,
        trial_index: int,
        execution_generation: int,
    ) -> dict[str, Any]:
        hidden_bytes = hidden_path.read_bytes()
        payload_b64 = base64.b64encode(hidden_bytes).decode("ascii")
        operation_id = (
            f"h012-012-{self.mode}-step{step_index:02d}-{phase}-{trial_index}-{uuid4().hex[:8]}"
        )
        deadline_unix_ns = time.time_ns() + int(self.operation_deadline_s * 1_000_000_000)
        profile = NETWORK_PROFILES["same_host_shaped"]
        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.process_time_ns()
        results: list[tuple[DelegatedNode, dict[str, Any], dict[str, Any]]] = []
        errors: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(self.topology.root_children)) as executor:
            futures: dict[Future[Any], DelegatedNode] = {
                executor.submit(
                    self._root_child,
                    node=node,
                    operation_id=operation_id,
                    execution_generation=execution_generation,
                    payload_b64=payload_b64,
                    payload_bytes=len(hidden_bytes),
                    deadline_unix_ns=deadline_unix_ns,
                    profile=profile,
                ): node
                for node in self.topology.root_children
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    results.append(future.result())
                except BaseException as error:
                    errors.append(
                        {
                            "worker_id": node.worker.worker_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
        root_cpu_ns = time.process_time_ns() - cpu_started_ns
        latency_ns = time.perf_counter_ns() - started_ns
        results.sort(key=lambda item: item[0].ordering_key)
        aggregate = (
            combine_operation_aggregates(
                [dict(item[1]["aggregate"]) for item in results], AGGREGATION
            )
            if results and not errors
            else None
        )
        transports = [item[2] for item in results]
        subtree_metrics = [dict(item[1]["subtree_metrics"]) for item in results]
        root_bytes_sent = sum(int(item["request_bytes"]) for item in transports)
        root_bytes_received = sum(int(item["response_bytes"]) for item in transports)
        root_messages_sent = sum(int(item["messages_sent"]) for item in transports)
        root_messages_received = sum(int(item["messages_received"]) for item in transports)
        hierarchy_depth = max((int(item["hierarchy_depth"]) for item in subtree_metrics), default=0)
        reduction_depth = (
            1 + max((int(item["reduction_depth"]) for item in subtree_metrics), default=0)
            if aggregate is not None
            else 0
        )
        selected_token_id = int(aggregate["token_id"]) if aggregate is not None else None
        status = "ok" if not errors and selected_token_id == reference_token_id else "failed"
        if not errors and selected_token_id != reference_token_id:
            errors.append(
                {
                    "error_type": "TokenIdentityError",
                    "error": (
                        f"hierarchy selected {selected_token_id}; reference selected "
                        f"{reference_token_id}"
                    ),
                }
            )
        row = {
            "schema_version": "1.0",
            "experiment_id": "012",
            "cycle_id": HYPOTHESIS_ID,
            "evidence_class": "real_immutable_model_output_head",
            "mode": self.mode,
            "phase": phase,
            "trial_index": trial_index,
            "step_index": step_index,
            "warmup": phase == "warmup",
            "operation_id": operation_id,
            "execution_generation": execution_generation,
            "route_generation": self.topology.route_generation,
            "worker_count": len(self.topology.nodes),
            "branch_factor": self.topology.branch_factor,
            "network_profile": profile.name,
            "connection_policy": self.connection_policy,
            "payload_bytes": len(hidden_bytes),
            "status": status,
            "correctness": status == "ok",
            "reference_token_id": reference_token_id,
            "selected_token_id": selected_token_id,
            "aggregate": aggregate,
            "aggregate_digest": aggregate_digest(aggregate) if aggregate is not None else None,
            "result_published": status == "ok",
            "errors": errors,
            "root_rpc_count": sum(int(item.get("attempt_count", 1)) for item in transports),
            "root_messages_sent": root_messages_sent,
            "root_messages_received": root_messages_received,
            "root_messages_total": root_messages_sent + root_messages_received,
            "root_bytes_sent": root_bytes_sent,
            "root_bytes_received": root_bytes_received,
            "root_bytes_total": root_bytes_sent + root_bytes_received,
            "root_serial_waits": len(results),
            "root_coordinator_waits": 1 if results else 0,
            "root_direct_degree": len(self.topology.root_children),
            "root_leaf_rpc_count": sum(not item[0].children for item in results),
            "root_cpu_ns": root_cpu_ns,
            "stage_owner_model_weight_bytes": 0,
            "fallback_count": 0,
            "worker_to_worker_rpc_count": sum(
                int(item["worker_to_worker_rpc_count"]) for item in subtree_metrics
            ),
            "leaf_rpc_count": sum(int(item["leaf_rpc_count"]) for item in subtree_metrics),
            "total_messages": root_messages_sent
            + root_messages_received
            + sum(int(item["total_messages"]) for item in subtree_metrics),
            "total_bytes": root_bytes_sent
            + root_bytes_received
            + sum(int(item["total_bytes"]) for item in subtree_metrics),
            "hierarchy_depth": hierarchy_depth,
            "reduction_depth": reduction_depth,
            "intermediate_reductions": sum(
                int(item["intermediate_reductions"]) for item in subtree_metrics
            ),
            "root_connection_count": sum(int(item["new_connection_count"]) for item in transports),
            "total_connection_count": sum(int(item["new_connection_count"]) for item in transports)
            + sum(int(item["connection_count"]) for item in subtree_metrics),
            "persistent_reused_root_connections": sum(
                int(bool(item["connection_reused"])) for item in transports
            ),
            "end_to_end_latency_ns": latency_ns,
            "end_to_end_latency_ms": latency_ns / 1_000_000,
            "throughput_ops_s": 1_000_000_000 / latency_ns if latency_ns else 0.0,
            "root_results_received": len(results),
            "root_result_worker_ids": [item[0].worker.worker_id for item in results],
        }
        _append_jsonl(self.observation_path, row)
        return row


def _validate_microshards(microshard_directory: Path) -> dict[str, Any]:
    top_manifest_path = microshard_directory / "manifest.json"
    hashes_path = microshard_directory / "hashes.json"
    if _sha256_file(top_manifest_path) != MICROSHARD_MANIFEST_SHA256:
        raise ValueError("microshard top-level manifest hash differs from H012-012")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    top_manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for rank in range(WORKER_COUNT):
        relative_directory = Path("ranks") / f"rank-{rank:03d}"
        manifest_path = microshard_directory / relative_directory / "shard_manifest.json"
        weight_path = microshard_directory / relative_directory / "weights.safetensors"
        relative_manifest = str(relative_directory / "shard_manifest.json").replace("\\", "/")
        relative_weight = str(relative_directory / "weights.safetensors").replace("\\", "/")
        if _sha256_file(manifest_path) != str(hashes[relative_manifest]):
            raise ValueError(f"rank {rank} manifest hash mismatch")
        if _sha256_file(weight_path) != str(hashes[relative_weight]):
            raise ValueError(f"rank {rank} weight-file hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tensors = [item for item in manifest["tensors"] if item["tensor_name"] == "lm_head.weight"]
        if len(tensors) != 1:
            raise ValueError(f"rank {rank} has no unique lm_head shard")
        tensor = dict(tensors[0])
        records.append(
            {
                "rank": rank,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": hashes[relative_manifest],
                "weight_file": str(weight_path.resolve()),
                "weight_file_sha256": hashes[relative_weight],
                "model_id": manifest["model_id"],
                "model_revision": manifest["model_revision"],
                **tensor,
            }
        )
    ordered = sorted(records, key=lambda item: int(item["shard_start"]))
    cursor = 0
    for record in ordered:
        if int(record["shard_start"]) != cursor:
            raise ValueError("lm_head shard ranges contain a gap or overlap")
        cursor = int(record["shard_end"])
    return {
        "top_manifest": str(top_manifest_path.resolve()),
        "top_manifest_sha256": MICROSHARD_MANIFEST_SHA256,
        "top_manifest_model_id": top_manifest["model_id"],
        "top_manifest_model_revision": top_manifest["model_revision"],
        "rank_count": len(records),
        "coverage_start": 0,
        "coverage_end": cursor,
        "gap_free_non_overlapping": cursor == 151936,
        "records": records,
    }


def _run_reference_process(
    *, output_directory: Path, model_path: Path, max_new_tokens: int
) -> dict[str, Any]:
    reference_script = Path(__file__).with_name("real_model_reference.py").resolve()
    log_path = output_directory / "logs" / "reference-process.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(reference_script),
        "--model-path",
        str(model_path),
        "--model-id",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--prompt",
        PROMPT,
        "--max-new-tokens",
        str(max_new_tokens),
        "--output",
        str((output_directory / "raw" / "reference").resolve()),
    ]
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=900,
        )
    invocation = {
        "command": command,
        "returncode": completed.returncode,
        "log_path": str(log_path.resolve()),
        "completed_unix_ns": time.time_ns(),
    }
    _write_json(output_directory / "raw" / "reference-invocation.json", invocation)
    if completed.returncode != 0:
        raise RuntimeError(f"independent reference exited with code {completed.returncode}")
    return json.loads(
        (output_directory / "raw" / "reference" / "reference.json").read_text(encoding="utf-8")
    )


def _validate_worker_tensors(
    *,
    evidence_directory: Path,
    reference: dict[str, Any],
    operation_steps: dict[str, int],
    atol: float,
    rtol: float,
    minimum_cosine: float,
) -> dict[str, Any]:
    reference_logits = {
        int(step["step_index"]): np.fromfile(step["logits_path"], dtype="<f4")
        for step in reference["steps"]
    }
    comparisons: list[dict[str, Any]] = []
    for metadata_path in sorted(evidence_directory.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        operation_id = str(metadata["operation_id"])
        if operation_id not in operation_steps:
            raise ValueError(f"worker tensor has unknown operation {operation_id}")
        step_index = operation_steps[operation_id]
        actual_path = Path(metadata["tensor_path"])
        actual_bytes = actual_path.read_bytes()
        actual = np.frombuffer(actual_bytes, dtype="<f4")
        reference_slice = reference_logits[step_index][
            int(metadata["token_start"]) : int(metadata["token_end"])
        ]
        difference = np.abs(actual.astype(np.float64) - reference_slice.astype(np.float64))
        denominator = float(np.linalg.norm(actual) * np.linalg.norm(reference_slice))
        cosine = (
            float(np.dot(actual.astype(np.float64), reference_slice.astype(np.float64)))
            / denominator
            if denominator
            else 1.0
        )
        allclose = bool(np.allclose(actual, reference_slice, atol=atol, rtol=rtol))
        finite = bool(np.isfinite(actual).all())
        hash_matches = hashlib.sha256(actual_bytes).hexdigest() == metadata["logits_sha256"]
        comparison = {
            "operation_id": operation_id,
            "step_index": step_index,
            "worker_id": metadata["worker_id"],
            "process_id": metadata["process_id"],
            "rank": metadata["rank"],
            "token_start": metadata["token_start"],
            "token_end": metadata["token_end"],
            "element_count": int(actual.size),
            "finite": finite,
            "hash_matches": hash_matches,
            "allclose": allclose,
            "max_abs_error": float(difference.max(initial=0.0)),
            "mean_abs_error": float(difference.mean()) if difference.size else 0.0,
            "cosine_similarity": cosine,
            "cosine_pass": cosine >= minimum_cosine,
            "tensor_path": str(actual_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        }
        comparison["passed"] = bool(
            finite and hash_matches and allclose and comparison["cosine_pass"]
        )
        comparisons.append(comparison)
    expected_count = len(operation_steps) * WORKER_COUNT
    return {
        "schema_version": "1.0",
        "atol": atol,
        "rtol": rtol,
        "minimum_cosine_similarity": minimum_cosine,
        "expected_comparison_count": expected_count,
        "actual_comparison_count": len(comparisons),
        "all_passed": len(comparisons) == expected_count
        and all(item["passed"] for item in comparisons),
        "maximum_absolute_error": max(
            (float(item["max_abs_error"]) for item in comparisons), default=None
        ),
        "minimum_cosine_observed": min(
            (float(item["cosine_similarity"]) for item in comparisons), default=None
        ),
        "comparisons": comparisons,
    }


def _trace_evidence(
    *, output_directory: Path, workers: list[WorkerProcess], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_operations = {
        str(row["operation_id"]) for row in rows if row["mode"] == "delegated_real_argmax"
    }
    worker_events: list[dict[str, Any]] = []
    copied: list[str] = []
    worker_trace_directory = output_directory / "traces" / "workers"
    worker_trace_directory.mkdir(parents=True, exist_ok=True)
    for worker in workers:
        source = output_directory / "raw" / "processes" / worker.worker_id / "trace.jsonl"
        destination = worker_trace_directory / f"{worker.worker_id}.jsonl"
        if source.exists():
            shutil.copy2(source, destination)
            copied.append(str(destination.resolve()))
            worker_events.extend(_read_jsonl(destination))
    child_edges = [
        event
        for event in worker_events
        if event.get("event") == "worker_child_round_trip"
        and event.get("operation_id") in candidate_operations
    ]
    reductions = [
        event
        for event in worker_events
        if event.get("event") == "delegated_subtree_reduced"
        and event.get("operation_id") in candidate_operations
        and event.get("child_worker_ids")
    ]
    root_events = [
        event
        for event in _read_jsonl(output_directory / "traces" / "root.jsonl")
        if event.get("mode") == "delegated_real_argmax"
    ]
    root_results_by_operation = {
        operation_id: sum(event.get("operation_id") == operation_id for event in root_events)
        for operation_id in candidate_operations
    }
    cross_process = all(
        int(event["sender_process_id"]) != int(event["receiver_process_id"])
        for event in child_edges
    )
    return {
        "candidate_operation_count": len(candidate_operations),
        "worker_child_edge_event_count": len(child_edges),
        "cross_process_worker_edges": bool(child_edges) and cross_process,
        "intermediate_reduction_event_count": len(reductions),
        "root_results_by_operation": root_results_by_operation,
        "one_aggregate_per_root_branch": bool(root_results_by_operation)
        and all(value == BRANCH_FACTOR for value in root_results_by_operation.values()),
        "copied_worker_trace_files": copied,
    }


def run_h012_012(
    *,
    output_directory: Path,
    microshard_directory: Path,
    max_new_tokens: int = 4,
    operation_deadline_s: float = 60.0,
    startup_deadline_s: float = 180.0,
    atol: float = 0.05,
    rtol: float = 0.02,
    minimum_cosine: float = 0.999,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_directory / "hypothesis.json"
    hypothesis = json.loads(hypothesis_path.read_text(encoding="utf-8"))
    if (
        hypothesis.get("hypothesis_id") != HYPOTHESIS_ID
        or hypothesis.get("status") != "predeclared"
        or hypothesis.get("criteria_locked_before_real_argmax_protocol_implementation") is not True
    ):
        raise ValueError("H012-012 hypothesis is not a locked predeclared record")
    _write_json(
        output_directory / "hypothesis-identity.json",
        {
            "path": str(hypothesis_path.resolve()),
            "sha256": _sha256_file(hypothesis_path),
            "bytes": hypothesis_path.stat().st_size,
        },
    )
    repository_root = Path(__file__).resolve().parents[4]
    protocol_script = repository_root / "src" / "swarm_inference" / "microworker_protocol.py"
    reference_script = Path(__file__).with_name("real_model_reference.py")
    source_identity = _source_identity(
        repository_root,
        [Path(__file__).resolve(), reference_script.resolve(), protocol_script.resolve()],
    )
    _write_json(output_directory / "code-revision.json", source_identity)
    environment = {
        "captured_unix_ns": time.time_ns(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "root_process_id": os.getpid(),
        "physical_network_claimed": False,
        "network_evidence": "single-host shaped loopback",
    }
    _write_json(output_directory / "environment.json", environment)
    rows: list[dict[str, Any]] = []
    worker_pool: WorkerPool | None = None
    workers: list[WorkerProcess] = []
    stop_evidence: dict[str, Any] = {}
    fatal_error: dict[str, Any] | None = None
    tensor_validation: dict[str, Any] = {}
    trace_evidence: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    microshard_validation: dict[str, Any] = {}
    resolver_evidence: dict[str, Any] = {}
    flat_installation: dict[str, Any] = {}
    delegated_installation: dict[str, Any] = {}
    flat_topology_record: dict[str, Any] = {}
    delegated_topology_record: dict[str, Any] = {}
    try:
        resolution_started_ns = time.perf_counter_ns()
        resolved = resolve_model(MODEL_ID, revision=MODEL_REVISION, allow_download=True)
        inspection = model_inspection_payload(inspect_native_model(resolved))
        source_weight = resolved.path / "model.safetensors"
        source_hash = _sha256_file(source_weight)
        resolver_evidence = {
            "api": "swarm_inference.model.shard_builder.resolve_model",
            "requested_model_id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_model_id": resolved.model_id,
            "resolved_revision": resolved.revision,
            "resolved_path": str(resolved.path),
            "downloaded": resolved.downloaded,
            "resolution_elapsed_ns": time.perf_counter_ns() - resolution_started_ns,
            "source_safetensors_path": str(source_weight.resolve()),
            "source_safetensors_sha256": source_hash,
            "inspection": inspection,
        }
        if (
            resolved.model_id != MODEL_ID
            or resolved.revision != MODEL_REVISION
            or source_hash != MODEL_SAFETENSORS_SHA256
            or inspection["architecture"] != "Qwen3ForCausalLM"
            or inspection["hidden_size"] != 1024
            or inspection["vocabulary_size"] != 151936
        ):
            raise ValueError("canonical resolver identity differs from H012-012")
        _write_json(output_directory / "model-resolution.json", resolver_evidence)
        microshard_validation = _validate_microshards(microshard_directory)
        _write_json(output_directory / "microshard-validation.json", microshard_validation)
        reference = _run_reference_process(
            output_directory=output_directory,
            model_path=resolved.path,
            max_new_tokens=max_new_tokens,
        )
        runtime_profiles = tuple(
            {
                "name": "real_output_head_rank",
                "compute_delay_ms": 0.0,
                "capacity_score": 1.0,
                "maximum_payload_bytes": 1 << 20,
                "real_model_shard": {
                    "manifest_path": record["manifest_path"],
                    "weight_file": record["weight_file"],
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "tensor_name": "lm_head.weight",
                    "torch_threads": 1,
                    "evidence_directory": str(
                        (
                            output_directory
                            / "raw"
                            / "worker-tensors"
                            / f"worker-{int(record['rank']):06d}"
                        ).resolve()
                    ),
                },
            }
            for record in microshard_validation["records"]
        )
        worker_pool = WorkerPool(
            count=WORKER_COUNT,
            directory=output_directory / "raw" / "processes",
            startup_deadline_s=startup_deadline_s,
            protocol_script=protocol_script,
            runtime_profiles=runtime_profiles,
            include_site_packages=True,
        )
        workers = worker_pool.start()
        _write_json(
            output_directory / "worker-identities.json",
            {
                "workers": [worker.ready for worker in workers],
                "unique_process_ids": len({worker.process_id for worker in workers}),
                "unique_endpoints": len({worker.endpoint for worker in workers}),
                "reference_process_id": reference["process_id"],
                "stage_owner_process_id": os.getpid(),
            },
        )
        workload = {
            "kind": "real_model_lm_head",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tensor_name": "lm_head.weight",
            "hidden_size": 1024,
            "vocabulary_size": 151936,
            "hidden_dtype": "float32-le",
        }
        flat_topology = _flat_topology(workers, generation=1)
        flat_topology_record = topology_record(flat_topology)
        _write_json(output_directory / "topologies" / "flat.json", flat_topology_record)
        flat_installation = install_topology(flat_topology)
        _write_json(output_directory / "topology-installation" / "flat.json", flat_installation)
        flat_runner = RealArgmaxRunner(
            topology=flat_topology,
            output_directory=output_directory,
            workload=workload,
            operation_deadline_s=operation_deadline_s,
            connection_policy="ephemeral",
            mode="flat_real_argmax",
        )
        try:
            for step in reference["steps"]:
                rows.append(
                    flat_runner.run(
                        hidden_path=Path(step["hidden_path"]),
                        reference_token_id=int(step["token_id"]),
                        step_index=int(step["step_index"]),
                        phase="measured",
                        trial_index=0,
                        execution_generation=1,
                    )
                )
        finally:
            flat_runner.close()

        delegated_topology = build_delegated_topology(
            workers,
            branch_factor=BRANCH_FACTOR,
            topology_id="h012-012-delegated-real-output-head",
            route_lease_id="h012-012-delegated-lease",
            route_generation=2,
        )
        delegated_topology_record = topology_record(delegated_topology)
        _write_json(
            output_directory / "topologies" / "delegated-b2.json", delegated_topology_record
        )
        delegated_installation = install_topology(delegated_topology)
        _write_json(
            output_directory / "topology-installation" / "delegated-b2.json",
            delegated_installation,
        )
        delegated_runner = RealArgmaxRunner(
            topology=delegated_topology,
            output_directory=output_directory,
            workload=workload,
            operation_deadline_s=operation_deadline_s,
            connection_policy="persistent",
            mode="delegated_real_argmax",
        )
        try:
            step_zero = reference["steps"][0]
            rows.append(
                delegated_runner.run(
                    hidden_path=Path(step_zero["hidden_path"]),
                    reference_token_id=int(step_zero["token_id"]),
                    step_index=0,
                    phase="warmup",
                    trial_index=0,
                    execution_generation=2,
                )
            )
            for repeat in range(2):
                rows.append(
                    delegated_runner.run(
                        hidden_path=Path(step_zero["hidden_path"]),
                        reference_token_id=int(step_zero["token_id"]),
                        step_index=0,
                        phase="repeat",
                        trial_index=repeat,
                        execution_generation=2,
                    )
                )
            for step in reference["steps"][1:]:
                rows.append(
                    delegated_runner.run(
                        hidden_path=Path(step["hidden_path"]),
                        reference_token_id=int(step["token_id"]),
                        step_index=int(step["step_index"]),
                        phase="measured",
                        trial_index=0,
                        execution_generation=2,
                    )
                )
        finally:
            delegated_runner.close()
        memory_evidence = worker_pool.sample_memory_snapshot()
        _write_json(output_directory / "worker-memory.json", memory_evidence)
    except BaseException as error:
        fatal_error = {
            "error_type": type(error).__name__,
            "error": str(error),
            "timestamp_unix_ns": time.time_ns(),
        }
        _write_json(output_directory / "errors" / "fatal.json", fatal_error)
    finally:
        if worker_pool is not None:
            stop_evidence = worker_pool.stop()
            _write_json(output_directory / "worker-shutdown.json", stop_evidence)

    operation_steps = {str(row["operation_id"]): int(row["step_index"]) for row in rows}
    if reference and operation_steps:
        try:
            tensor_validation = _validate_worker_tensors(
                evidence_directory=output_directory / "raw" / "worker-tensors",
                reference=reference,
                operation_steps=operation_steps,
                atol=atol,
                rtol=rtol,
                minimum_cosine=minimum_cosine,
            )
            _write_json(
                output_directory / "correctness" / "tensor-comparisons.json", tensor_validation
            )
        except BaseException as error:
            tensor_validation = {
                "all_passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(output_directory / "errors" / "tensor-validation.json", tensor_validation)
    if workers:
        trace_evidence = _trace_evidence(
            output_directory=output_directory, workers=workers, rows=rows
        )
        _write_json(output_directory / "trace-evidence.json", trace_evidence)

    flat_rows = [row for row in rows if row["mode"] == "flat_real_argmax"]
    delegated_rows = [row for row in rows if row["mode"] == "delegated_real_argmax"]
    measured_delegated = [row for row in delegated_rows if not row["warmup"]]
    repeats = [row for row in delegated_rows if row["phase"] == "repeat"]
    ready_proofs = [worker.ready.get("model_shard_proof") for worker in workers]
    checks = {
        "resolver_identity": bool(resolver_evidence)
        and resolver_evidence.get("resolved_model_id") == MODEL_ID
        and resolver_evidence.get("resolved_revision") == MODEL_REVISION
        and resolver_evidence.get("source_safetensors_sha256") == MODEL_SAFETENSORS_SHA256,
        "reference_normal_generation": bool(reference)
        and reference.get("manual_matches_generate") is True
        and len(reference.get("manual_token_ids", [])) == max_new_tokens,
        "worker_process_isolation": len(workers) == WORKER_COUNT
        and len({worker.process_id for worker in workers}) == WORKER_COUNT
        and len({worker.endpoint for worker in workers}) == WORKER_COUNT,
        "shard_coverage": bool(microshard_validation)
        and microshard_validation.get("gap_free_non_overlapping") is True
        and microshard_validation.get("rank_count") == WORKER_COUNT,
        "verified_partial_loads": len(ready_proofs) == WORKER_COUNT
        and all(proof and not proof["complete_output_head_loaded"] for proof in ready_proofs)
        and all(proof and not proof["complete_model_loaded"] for proof in ready_proofs)
        and all(
            proof and proof["local_tensor_hash"] == proof["expected_local_tensor_hash"]
            for proof in ready_proofs
        ),
        "flat_token_identity": len(flat_rows) == max_new_tokens
        and all(row["correctness"] for row in flat_rows),
        "delegated_token_identity": len(measured_delegated) == max_new_tokens + 1
        and all(row["correctness"] for row in measured_delegated),
        "local_tensor_correctness": tensor_validation.get("all_passed") is True,
        "deterministic_repetition": len(repeats) == 2
        and repeats[0].get("aggregate") == repeats[1].get("aggregate"),
        "runtime_hierarchy": trace_evidence.get("cross_process_worker_edges") is True
        and trace_evidence.get("intermediate_reduction_event_count", 0) > 0
        and trace_evidence.get("one_aggregate_per_root_branch") is True,
        "delegated_root_bounds": bool(measured_delegated)
        and all(
            row["root_direct_degree"] == BRANCH_FACTOR
            and row["root_rpc_count"] == BRANCH_FACTOR
            and row["root_messages_total"] == 2 * BRANCH_FACTOR
            and row["root_leaf_rpc_count"] == 0
            and row["hierarchy_depth"] == delegated_topology_record.get("hierarchy_depth")
            and row["reduction_depth"] == delegated_topology_record.get("hierarchy_depth")
            for row in measured_delegated
        ),
        "flat_root_bounds": bool(flat_rows)
        and all(
            row["root_direct_degree"] == WORKER_COUNT
            and row["root_rpc_count"] == WORKER_COUNT
            and row["root_messages_total"] == 2 * WORKER_COUNT
            and row["root_leaf_rpc_count"] == WORKER_COUNT
            for row in flat_rows
        ),
        "persistent_reuse_after_warmup": bool(measured_delegated)
        and all(row["total_connection_count"] == 0 for row in measured_delegated),
        "no_fallback_or_root_weights": bool(rows)
        and all(row["fallback_count"] == 0 for row in rows)
        and all(row["stage_owner_model_weight_bytes"] == 0 for row in rows),
        "evidence_retained": bool(trace_evidence.get("copied_worker_trace_files"))
        and (output_directory / "raw" / "reference" / "reference.json").exists()
        and (output_directory / "raw" / "measurements.jsonl").exists(),
    }
    status = "PASS" if fatal_error is None and all(checks.values()) else "FAIL"
    correctness = {
        "status": status,
        "checks": checks,
        "reference_token_ids": reference.get("manual_token_ids"),
        "flat_token_ids": [row.get("selected_token_id") for row in flat_rows],
        "delegated_token_ids": [
            row.get("selected_token_id")
            for row in delegated_rows
            if row["phase"] in {"repeat", "measured"}
            and not (row["phase"] == "repeat" and row["trial_index"] == 1)
        ],
        "tensor_validation_summary": {
            key: tensor_validation.get(key)
            for key in (
                "expected_comparison_count",
                "actual_comparison_count",
                "all_passed",
                "maximum_absolute_error",
                "minimum_cosine_observed",
            )
        },
    }
    _write_json(output_directory / "correctness" / "summary.json", correctness)
    summary = {
        "schema_version": "1.0",
        "experiment_id": "012",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": status,
        "evidence_classification": (
            "single-host real immutable model; independent full-model reference; "
            "independent rank-worker processes; shaped loopback links"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "reference_process_id": reference.get("process_id"),
        "stage_owner_process_id": os.getpid(),
        "worker_process_ids": [worker.process_id for worker in workers],
        "checks": checks,
        "flat_rows": flat_rows,
        "delegated_rows": delegated_rows,
        "tensor_validation": correctness["tensor_validation_summary"],
        "trace_evidence": trace_evidence,
        "worker_shutdown": stop_evidence,
        "fatal_error": fatal_error,
        "completed_unix_ns": time.time_ns(),
    }
    _write_json(output_directory / "summary.json", summary)
    _write_json(
        output_directory / "decision.json",
        {
            "hypothesis_id": HYPOTHESIS_ID,
            "result": status,
            "criteria_evaluation": checks,
            "thesis_implication": (
                "real supported-model output-head work crossed the same delegated hierarchy"
                if status == "PASS"
                else "real-model hierarchical correctness remains unproven"
            ),
            "promotion_implication": (
                "eligible for the separate runtime promotion gate"
                if status == "PASS"
                else "do not promote"
            ),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 012 H012-012")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microshards", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--operation-deadline-s", type=float, default=60.0)
    parser.add_argument("--startup-deadline-s", type=float, default=180.0)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    args = parser.parse_args(argv)
    summary = run_h012_012(
        output_directory=args.output.resolve(),
        microshard_directory=args.microshards.resolve(),
        max_new_tokens=args.max_new_tokens,
        operation_deadline_s=args.operation_deadline_s,
        startup_deadline_s=args.startup_deadline_s,
        atol=args.atol,
        rtol=args.rtol,
        minimum_cosine=args.minimum_cosine,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
