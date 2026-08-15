"""Execute a frozen E022 placement manifest through authenticated K3 workers."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter, defaultdict
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _CudaRuntime,
    _numerical_metrics,
)
from swarm_inference.execution.kimi_k3_graph_runtime import (
    KimiCudaGraphRunner,
    _LayerResources,
    _parse_oracle_routes,
    _pointer_offset,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.physical import (
    ROUTED_EXPERTS,
    _ResidentHandles,
)
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
)
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
    new_run_credential,
)
from swarm_inference.experiments.experiment_022.native_dispatch import (
    CallableResidentPrimitive,
    NativeShardDispatcher,
    ShardRequest,
    ShardTaskType,
    decode_shard_result,
    encode_shard_request,
)
from swarm_inference.experiments.experiment_022.resident_primitives import (
    _sha256_arrays,
    prepare_complete_expert_stripe_banks,
)
from swarm_inference.experiments.experiment_022.resident_replay import (
    ResidentMixedLayerGraph,
)
from swarm_inference.experiments.experiment_022.whole_expert import (
    PreparedWholeExpertGroup,
)

HIDDEN = 7168
LATENT = 3584
LAYERS = 93
RELATIVE_L2_GATE = 2e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer_number(piece: str) -> int | None:
    prefix = "transformer_layer_"
    return int(piece[len(prefix) :]) if piece.startswith(prefix) else None


def _manifest_assignments(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for node in manifest["nodes"]:
        for piece in node["pieces"]:
            layer = _layer_number(str(piece["piece"]))
            if layer is not None:
                grouped[layer].append({"node_id": node["node_id"], **piece})
    if set(grouped) != set(range(LAYERS)):
        missing = sorted(set(range(LAYERS)) - set(grouped))
        extra = sorted(set(grouped) - set(range(LAYERS)))
        raise ValueError(f"manifest layer coverage differs: missing={missing} extra={extra}")
    result: dict[int, dict[str, Any]] = {}
    for layer, pieces in grouped.items():
        kinds = {str(piece["partition_type"]) for piece in pieces}
        degrees = {int(piece.get("degree", 1)) for piece in pieces}
        candidates = {str(piece["candidate_id"]) for piece in pieces}
        if len(kinds) != 1 or len(degrees) != 1 or len(candidates) != 1:
            raise ValueError(f"layer {layer} has inconsistent frozen assignments")
        kind = next(iter(kinds))
        degree = next(iter(degrees))
        expected = 1 if kind == "WHOLE_LAYER" else degree
        supported = {
            "WHOLE_LAYER",
            "WHOLE_EXPERT",
            "EXPERT_SHARD",
            "FULL_MIXED_STRIPE",
        }
        if kind not in supported or len(pieces) != expected:
            raise ValueError(
                f"layer {layer} manifest kind/count unsupported: {kind}/{len(pieces)}"
            )
        # Shards are symmetric in the frozen candidate.  The persisted manifest
        # identifies owners but not stripe ordinals, so coordinator-first then
        # node-id order is the unique deterministic reconstruction used here.
        ordered = sorted(
            pieces,
            key=lambda row: (not bool(row.get("coordinator")), str(row["node_id"])),
        )
        result[layer] = {
            "partition_type": kind,
            "degree": degree,
            "candidate_id": next(iter(candidates)),
            "pieces": [
                {**piece, "shard_index": index}
                for index, piece in enumerate(ordered)
            ],
        }
    return result


def _manifest_execution_workers(
    assignments: dict[int, dict[str, Any]], endpoint_owner: str
) -> set[str]:
    """Return the logical workers that own work in the executed task graph."""

    workers = {
        str(piece["node_id"])
        for assignment in assignments.values()
        for piece in assignment["pieces"]
    }
    workers.add(endpoint_owner)
    return workers


def _sha256_array(value: np.ndarray) -> str:
    source = np.ascontiguousarray(value)
    return "sha256:" + hashlib.sha256(source.tobytes()).hexdigest()


class _SharedResidentExpertStripe:
    """One exact native stripe backed by a process-shared CUDA runtime."""

    native = True
    native_primitive = "e020_kimi_grouped_top16"

    def __init__(
        self,
        runtime: _CudaRuntime,
        grouped: GroupedTop16Runtime,
        loader: DirectShardLoader,
        resident: _ResidentHandles,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int,
    ) -> None:
        self.runtime = runtime
        self.grouped = grouped
        self.loader = loader
        self.resident = resident
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.invocation_count = 0
        self.input = runtime.allocate(max_rows * LATENT * 4)
        self.route_weights = runtime.allocate(max_rows * 16 * 4)
        self.output = runtime.allocate(max_rows * LATENT * 4)
        self.resident_bytes = resident.runtime_bytes + max_rows * (LATENT * 2 + 16) * 4
        self.last_execution: dict[str, Any] = {}

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        if (
            request.layer != self.layer
            or request.degree != self.degree
            or request.shard_index != self.shard_index
        ):
            raise ValueError("expert request does not match its resident assignment")
        if source.shape != (rows, LATENT + 32) or rows > self.max_rows:
            raise ValueError("expert input must pack latent, route IDs, and weights")
        activation = np.ascontiguousarray(source[:, :LATENT])
        routes = np.rint(source[:, LATENT : LATENT + 16]).astype(np.int32)
        weights = np.ascontiguousarray(source[:, LATENT + 16 :])
        if any(
            int(value) not in self.resident.handles for value in routes.reshape(-1)
        ):
            raise ValueError("expert route is absent from the complete resident bank")
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copy_started = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, activation)
        self.runtime.upload_activation(self.route_weights, weights)
        input_copy_ms = (time.perf_counter_ns() - copy_started) / 1e6
        cuda_ms = self.grouped.execute(
            self.resident,
            routes,
            self.output,
            self.input,
            self.route_weights,
        )
        copy_started = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, LATENT))
        output_copy_ms = (time.perf_counter_ns() - copy_started) / 1e6
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        self.invocation_count += 1
        self.last_execution = {
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "host_ms": max(0.0, wall_ms - cuda_ms),
            "input_copy_ms": input_copy_ms,
            "output_copy_ms": output_copy_ms,
            "physical_launches": self.grouped.physical_launches,
            "state_mutated": False,
            "checkpoint_reads_in_timed_region": len(self.loader.audit) - reads_before,
            "whole_layer_fallback": False,
            "route_ids_sha256": _sha256_array(routes),
            "route_weights_sha256": _sha256_array(weights),
            "ordered_route_ids": routes.tolist(),
        }
        return output

    def close(self) -> None:
        self.runtime.free(self.output)
        self.runtime.free(self.route_weights)
        self.runtime.free(self.input)


def _persistent_expert_worker(
    connection: Any,
    credential: bytes,
    *,
    checkpoint: str,
    cuda_library: str,
    grouped_library: str,
) -> None:
    """Serve all manifest expert layers from one persistent worker process.

    Logical workers retain distinct dispatchers and resident handle banks.  The
    local one-GPU validation multiplexes those logical workers in one OS
    process so CUDA context creation and checkpoint catalog parsing are startup
    costs paid once, not once per task.
    """

    runtime: _CudaRuntime | None = None
    grouped: GroupedTop16Runtime | None = None
    try:
        catalog = CheckpointCatalog(Path(checkpoint))
        loader = DirectShardLoader(catalog)
        runtime = _CudaRuntime(Path(cuda_library), 0)
        runtime.set_telemetry("minimal")
        runtime.set_fused_gate_up(True)
        grouped = GroupedTop16Runtime(Path(grouped_library))
        connection.send(
            {
                "status": "READY",
                "pid": os.getpid(),
                "checkpoint_index_sha256": catalog.index_sha256,
            }
        )
        while True:
            message = connection.recv()
            if message.get("command") == "CLOSE":
                connection.send({"status": "CLOSED", "pid": os.getpid()})
                break
            if message.get("command") != "EXECUTE_EXPERT_LAYER":
                raise ValueError("persistent expert worker received an unknown command")
            layer = int(message["layer"])
            degree = int(message["degree"])
            rows = int(message["rows"])
            pieces = list(message["pieces"])
            partition_type = str(message.get("partition_type", "EXPERT_SHARD"))
            if partition_type == "WHOLE_EXPERT":
                responses: list[dict[str, Any]] = []
                startup_rows: list[dict[str, Any]] = []
                ownership: list[tuple[int, int]] = []
                timed_reads = 0
                try:
                    for piece in pieces:
                        shard = int(piece["shard_index"])
                        primitive = PreparedWholeExpertGroup(
                            Path(checkpoint),
                            Path(cuda_library),
                            Path(grouped_library),
                            layer=layer,
                            degree=degree,
                            shard_index=shard,
                            max_rows=rows,
                            # This group is nested inside the persistent expert
                            # worker's outer CUDA runtime.  Free group-owned
                            # handles without shutting down that shared runtime;
                            # the worker performs the one final shutdown.
                            shutdown_runtime_on_close=False,
                        )
                        dispatcher = NativeShardDispatcher(str(piece["worker_id"]))
                        try:
                            dispatcher.register(
                                str(piece["assignment_id"]),
                                ShardTaskType.WHOLE_EXPERT_GROUP,
                                primitive,
                            )
                            protocol_started = time.perf_counter_ns()
                            frame = decode_frame(piece["encoded_frame"], credential)
                            result_frame = dispatcher.execute_frame(frame)
                            encoded_result = encode_frame(result_frame, credential)
                            protocol_wall_ms = (
                                time.perf_counter_ns() - protocol_started
                            ) / 1e6
                            last_execution = dict(primitive.last_execution)
                            last_execution["route_ids_sha256"] = last_execution[
                                "input_route_ids_sha256"
                            ]
                            last_execution["route_weights_sha256"] = last_execution[
                                "input_route_weights_sha256"
                            ]
                            timed_reads += int(
                                last_execution["checkpoint_reads_in_timed_region"]
                            )
                            ownership.append(
                                (primitive.expert_start, primitive.expert_stop)
                            )
                            startup_rows.append(
                                {
                                    "worker_id": piece["worker_id"],
                                    **primitive.startup,
                                }
                            )
                            responses.append(
                                {
                                    "worker_id": piece["worker_id"],
                                    "assignment_id": piece["assignment_id"],
                                    "shard_index": shard,
                                    "encoded_result": encoded_result,
                                    "protocol_wall_ms": protocol_wall_ms,
                                    "resident_bytes": primitive.resident_bytes,
                                    "last_execution": last_execution,
                                    "dispatcher_audit": dispatcher.audit,
                                    "complete_expert_bank_resident": False,
                                    "complete_assigned_expert_group": True,
                                    "expert_id_start": primitive.expert_start,
                                    "expert_id_stop_exclusive": primitive.expert_stop,
                                }
                            )
                        finally:
                            primitive.close()
                    ordered = sorted(ownership)
                    complete_ownership = (
                        len(ordered) == degree
                        and ordered[0][0] == 0
                        and ordered[-1][1] == ROUTED_EXPERTS
                        and all(
                            left[1] == right[0]
                            for left, right in pairwise(ordered)
                        )
                    )
                    if not complete_ownership:
                        raise RuntimeError(
                            "whole-expert workers did not exactly cover the expert bank"
                        )
                    connection.send(
                        {
                            "status": "PASS",
                            "pid": os.getpid(),
                            "layer": layer,
                            "degree": degree,
                            "partition_type": partition_type,
                            "startup": {
                                "wall_ms": sum(
                                    float(value["wall_ms"])
                                    for value in startup_rows
                                ),
                                "checkpoint_reads": sum(
                                    int(value["checkpoint_reads"])
                                    for value in startup_rows
                                ),
                                "checkpoint_read_strategy": (
                                    "disjoint complete-expert groups prepared before "
                                    "their authenticated execution"
                                ),
                                "worker_startups": startup_rows,
                                "complete_expert_ownership": complete_ownership,
                            },
                            "checkpoint_reads_in_timed_region": timed_reads,
                            "responses": responses,
                        }
                    )
                except BaseException as exc:
                    connection.send(
                        {
                            "status": "FAIL",
                            "pid": os.getpid(),
                            "layer": layer,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                continue
            if partition_type != "EXPERT_SHARD":
                raise ValueError(
                    f"persistent expert worker does not support {partition_type}"
                )
            banks: tuple[_ResidentHandles, ...] = ()
            primitives: list[_SharedResidentExpertStripe] = []
            dispatchers: list[NativeShardDispatcher] = []
            try:
                startup_reads_before = len(loader.audit)
                startup_started = time.perf_counter_ns()
                banks = prepare_complete_expert_stripe_banks(
                    runtime,
                    loader,
                    layer=layer,
                    degree=degree,
                    worker_prefix=f"manifest.layer-{layer:02d}.p{degree}",
                )
                startup_wall_ms = (time.perf_counter_ns() - startup_started) / 1e6
                startup_reads = len(loader.audit) - startup_reads_before
                if len(banks) != degree or any(
                    len(bank.handles) != ROUTED_EXPERTS for bank in banks
                ):
                    raise RuntimeError("manifest worker did not prepare complete banks")
                for piece in pieces:
                    shard = int(piece["shard_index"])
                    primitive = _SharedResidentExpertStripe(
                        runtime,
                        grouped,
                        loader,
                        banks[shard],
                        layer=layer,
                        degree=degree,
                        shard_index=shard,
                        max_rows=rows,
                    )
                    dispatcher = NativeShardDispatcher(str(piece["worker_id"]))
                    dispatcher.register(
                        str(piece["assignment_id"]),
                        ShardTaskType.EXPERT_STRIPE,
                        primitive,
                    )
                    primitives.append(primitive)
                    dispatchers.append(dispatcher)
                timed_reads_before = len(loader.audit)
                responses: list[dict[str, Any]] = []
                for piece, primitive, dispatcher in zip(
                    pieces, primitives, dispatchers, strict=True
                ):
                    protocol_started = time.perf_counter_ns()
                    frame = decode_frame(piece["encoded_frame"], credential)
                    result_frame = dispatcher.execute_frame(frame)
                    encoded_result = encode_frame(result_frame, credential)
                    protocol_wall_ms = (time.perf_counter_ns() - protocol_started) / 1e6
                    responses.append(
                        {
                            "worker_id": piece["worker_id"],
                            "assignment_id": piece["assignment_id"],
                            "shard_index": piece["shard_index"],
                            "encoded_result": encoded_result,
                            "protocol_wall_ms": protocol_wall_ms,
                            "resident_bytes": primitive.resident_bytes,
                            "last_execution": primitive.last_execution,
                            "dispatcher_audit": dispatcher.audit,
                            "complete_expert_bank_resident": (
                                len(primitive.resident.handles) == ROUTED_EXPERTS
                            ),
                        }
                    )
                timed_reads = len(loader.audit) - timed_reads_before
                connection.send(
                    {
                        "status": "PASS",
                        "pid": os.getpid(),
                        "layer": layer,
                        "degree": degree,
                        "startup": {
                            "wall_ms": startup_wall_ms,
                            "checkpoint_reads": startup_reads,
                            "checkpoint_read_strategy": (
                                "one immutable tensor read per expert followed by exact "
                                "MXFP4 axis slicing into distinct resident worker handles"
                            ),
                            "complete_expert_banks": degree,
                            "experts_per_bank": ROUTED_EXPERTS,
                        },
                        "checkpoint_reads_in_timed_region": timed_reads,
                        "responses": responses,
                    }
                )
            except BaseException as exc:
                connection.send(
                    {
                        "status": "FAIL",
                        "pid": os.getpid(),
                        "layer": layer,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                for primitive in reversed(primitives):
                    primitive.close()
                for bank in reversed(banks):
                    bank.close()
    except EOFError:
        pass
    except BaseException as exc:
        with suppress(BaseException):
            connection.send(
                {
                    "status": "FAIL",
                    "pid": os.getpid(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    finally:
        if grouped is not None:
            grouped.close()
        if runtime is not None:
            runtime.close()
        connection.close()


def _endpoint_owners(manifest: dict[str, Any]) -> list[str]:
    return sorted(
        str(node["node_id"])
        for node in manifest["nodes"]
        if any(
            piece["partition_type"] == "IDENTICAL_ENDPOINT_POLICY"
            for piece in node["pieces"]
        )
    )


class ManifestK3Runner(KimiCudaGraphRunner):
    """Canonical K3 graph with compute dispatched by the frozen manifest."""

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        grouped_library: Path,
        manifest: dict[str, Any],
        *,
        shard_library: Path | None = None,
        state_trace_root: Path | None = None,
        state_reference_root: Path | None = None,
    ) -> None:
        super().__init__(checkpoint, cuda_library)
        self.checkpoint = checkpoint
        self.cuda_library = cuda_library
        self.grouped_library = grouped_library
        self.shard_library = shard_library
        self.manifest = manifest
        self.assignments = _manifest_assignments(manifest)
        self.endpoint_owners = _endpoint_owners(manifest)
        if not self.endpoint_owners:
            raise ValueError("manifest has no endpoint owner")
        self.credential = new_run_credential()
        self.dispatchers: dict[str, NativeShardDispatcher] = {}
        self.layer_dispatch: dict[int, tuple[str, ShardTaskType, str]] = {}
        self._pending_layer: dict[int, dict[str, Any]] = {}
        self._pending_endpoint: dict[str, Any] = {}
        self.dispatch_receipts: list[dict[str, Any]] = []
        self.expert_worker_receipts: list[dict[str, Any]] = []
        self.full_mixed_worker_receipts: list[dict[str, Any]] = []
        self._prepared_full_mixed: dict[int, ResidentMixedLayerGraph] = {}
        self.maximum_actual_resident_bytes = 0
        self.logical_workers_instantiated: set[str] = set()
        self.state_trace_root = (
            state_trace_root.resolve() if state_trace_root is not None else None
        )
        self.state_reference_root = (
            state_reference_root.resolve()
            if state_reference_root is not None
            else None
        )
        self.state_trace_rows: list[dict[str, Any]] = []
        self.state_reference_receipts: list[dict[str, Any]] = []
        self.state_reference_metadata: dict[str, Any] | None = None
        if self.state_trace_root is not None:
            self.state_trace_root.mkdir(parents=True, exist_ok=True)
            if any(self.state_trace_root.iterdir()):
                raise FileExistsError(
                    f"state trace destination is not empty: {self.state_trace_root}"
                )
        if self.state_reference_root is not None:
            metadata_path = self.state_reference_root / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("status") != "PASS"
                or int(metadata.get("layer_count", -1)) != LAYERS
            ):
                raise ValueError("state reference trace is not a complete PASS")
            self.state_reference_metadata = metadata
        self._expert_connection: Any | None = None
        self._expert_process: Any | None = None
        self.expert_worker_process_startups = 0
        self.expert_worker_process_audit: dict[str, Any] = {
            "required": False,
            "successful_startups": 0,
            "ready_status": "NOT_APPLICABLE",
            "close_status": "NOT_APPLICABLE",
            "pid": None,
            "exitcode": None,
        }
        self._register_manifest()

    @staticmethod
    def _state_trace_filename(layer: int, name: str) -> str:
        return f"layer-{layer:02d}-{name}.npy"

    def _record_state_trace(
        self,
        *,
        layer: int,
        attention_type: str,
        attnres: np.ndarray,
    ) -> None:
        if self.state_trace_root is None:
            return
        components: list[dict[str, Any]] = []
        for name, values in sorted(self.states[layer].items()):
            source = np.ascontiguousarray(values, dtype=np.float32)
            filename = self._state_trace_filename(layer, name)
            np.save(self.state_trace_root / filename, source, allow_pickle=False)
            components.append(
                {
                    "name": name,
                    "path": filename,
                    "shape": list(source.shape),
                    "dtype": source.dtype.str,
                    "sha256": _sha256_array(source),
                    "bytes": int(source.nbytes),
                }
            )
        attnres_source = np.ascontiguousarray(attnres, dtype=np.float32)
        attnres_filename = self._state_trace_filename(layer, "attnres")
        np.save(
            self.state_trace_root / attnres_filename,
            attnres_source,
            allow_pickle=False,
        )
        self.state_trace_rows.append(
            {
                "layer": layer,
                "attention_type": attention_type,
                "components": components,
                "attnres": {
                    "path": attnres_filename,
                    "shape": list(attnres_source.shape),
                    "dtype": attnres_source.dtype.str,
                    "sha256": _sha256_array(attnres_source),
                    "bytes": int(attnres_source.nbytes),
                },
            }
        )

    def _compare_state_reference(
        self,
        *,
        layer: int,
        attention_type: str,
        attnres: np.ndarray,
    ) -> None:
        if self.state_reference_root is None:
            return
        assert self.state_reference_metadata is not None
        rows = {
            int(row["layer"]): row
            for row in self.state_reference_metadata["layers"]
        }
        if layer not in rows:
            raise ValueError(f"state reference omitted layer {layer}")
        reference_row = rows[layer]
        if reference_row["attention_type"] != attention_type:
            raise ValueError("state reference attention type differs")
        reference_components = {
            str(row["name"]): row for row in reference_row["components"]
        }
        if set(reference_components) != set(self.states[layer]):
            raise ValueError("state reference component set differs")
        component_receipts: list[dict[str, Any]] = []
        squared_delta = 0.0
        squared_reference = 0.0
        for name, actual_values in sorted(self.states[layer].items()):
            component = reference_components[name]
            reference_values = np.load(
                self.state_reference_root / component["path"],
                allow_pickle=False,
            )
            actual = np.ascontiguousarray(actual_values, dtype=np.float32)
            metrics = _numerical_metrics(reference_values, actual)
            delta = actual.astype(np.float64) - reference_values.astype(np.float64)
            squared_delta += float(np.dot(delta.ravel(), delta.ravel()))
            reference64 = reference_values.astype(np.float64)
            squared_reference += float(
                np.dot(reference64.ravel(), reference64.ravel())
            )
            component_receipts.append(
                {
                    "name": name,
                    "reference_sha256": component["sha256"],
                    "actual_sha256": _sha256_array(actual),
                    "fingerprint_equal": (
                        component["sha256"] == _sha256_array(actual)
                    ),
                    "metrics": metrics,
                }
            )
        attention_relative_l2 = math.sqrt(squared_delta) / (
            math.sqrt(squared_reference) if squared_reference else 1.0
        )
        attnres_reference = np.load(
            self.state_reference_root / reference_row["attnres"]["path"],
            allow_pickle=False,
        )
        attnres_actual = np.ascontiguousarray(attnres, dtype=np.float32)
        attnres_metrics = _numerical_metrics(attnres_reference, attnres_actual)
        passed = (
            attention_relative_l2 <= RELATIVE_L2_GATE
            and float(attnres_metrics["relative_l2_error"])
            <= RELATIVE_L2_GATE
            and all(
                receipt["metrics"]["reference_finite"]
                and receipt["metrics"]["actual_finite"]
                and float(receipt["metrics"]["relative_l2_error"])
                <= RELATIVE_L2_GATE
                for receipt in component_receipts
            )
            and attnres_metrics["reference_finite"]
            and attnres_metrics["actual_finite"]
        )
        self.state_reference_receipts.append(
            {
                "layer": layer,
                "attention_type": attention_type,
                "status": "PASS" if passed else "FAIL",
                "relative_l2_gate": RELATIVE_L2_GATE,
                "attention_state_relative_l2": attention_relative_l2,
                "attention_state_fingerprint_equal": all(
                    row["fingerprint_equal"] for row in component_receipts
                ),
                "components": component_receipts,
                "attnres_reference_sha256": reference_row["attnres"]["sha256"],
                "attnres_actual_sha256": _sha256_array(attnres_actual),
                "attnres_fingerprint_equal": (
                    reference_row["attnres"]["sha256"]
                    == _sha256_array(attnres_actual)
                ),
                "attnres_metrics": attnres_metrics,
            }
        )

    def finalize_state_trace(
        self,
        *,
        selection_id: str,
        manifest_sha256: str,
    ) -> dict[str, Any] | None:
        if self.state_trace_root is None:
            return None
        complete = len(self.state_trace_rows) == LAYERS and {
            int(row["layer"]) for row in self.state_trace_rows
        } == set(range(LAYERS))
        metadata = {
            "schema_version": "experiment-022-completion-state-trace-v1",
            "status": "PASS" if complete else "FAIL",
            "selection_id": selection_id,
            "manifest_sha256": manifest_sha256,
            "layer_count": len(self.state_trace_rows),
            "layers": sorted(self.state_trace_rows, key=lambda row: row["layer"]),
            "semantic_boundary": (
                "whole-layer physical reference arrays for numerical recurrent-state "
                "and AttnRes comparison; fingerprints remain separately visible"
            ),
        }
        metadata_path = self.state_trace_root / "metadata.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
        if not complete:
            raise RuntimeError("state trace did not cover all 93 layers")
        return {
            "path": str(metadata_path),
            "sha256": _sha256_file(metadata_path),
            "layer_count": len(self.state_trace_rows),
        }

    def state_reference_summary(self) -> dict[str, Any]:
        if self.state_reference_root is None:
            return {
                "status": "NOT_APPLICABLE",
                "reference_path": None,
                "layer_count": 0,
            }
        metadata_path = self.state_reference_root / "metadata.json"
        rows = self.state_reference_receipts
        attention_max = max(
            float(row["attention_state_relative_l2"]) for row in rows
        )
        attnres_max = max(
            float(row["attnres_metrics"]["relative_l2_error"]) for row in rows
        )
        passed = (
            len(rows) == LAYERS
            and {int(row["layer"]) for row in rows} == set(range(LAYERS))
            and all(row["status"] == "PASS" for row in rows)
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "reference_path": str(metadata_path),
            "reference_metadata_sha256": _sha256_file(metadata_path),
            "reference_selection_id": self.state_reference_metadata[
                "selection_id"
            ],
            "reference_manifest_sha256": self.state_reference_metadata[
                "manifest_sha256"
            ],
            "relative_l2_gate": RELATIVE_L2_GATE,
            "layer_count": len(rows),
            "kda_layer_count": sum(
                row["attention_type"] == "KDA" for row in rows
            ),
            "mla_layer_count": sum(
                row["attention_type"] == "Gated_MLA" for row in rows
            ),
            "attention_state_relative_l2_maximum": attention_max,
            "attnres_relative_l2_maximum": attnres_max,
            "attention_state_fingerprint_equal_layers": sum(
                row["attention_state_fingerprint_equal"] for row in rows
            ),
            "attnres_fingerprint_equal_layers": sum(
                row["attnres_fingerprint_equal"] for row in rows
            ),
            "layers": rows,
        }

    @staticmethod
    def _attnres_list(
        block_residuals: np.ndarray, block_count: int
    ) -> list[np.ndarray]:
        return [
            np.ascontiguousarray(block_residuals[:, index], dtype=np.float32)
            for index in range(block_count)
        ]

    def _prepare_full_mixed_layer(
        self,
        *,
        layer: int,
        values: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
    ) -> dict[str, Any]:
        """Prepare the real resident mixed DAG before its authenticated hot call."""

        if self.shard_library is None:
            raise ValueError("FULL_MIXED_STRIPE requires the native shard library")
        if layer in self._prepared_full_mixed:
            raise RuntimeError("full-mixed layer was prepared more than once")
        assignment = self.assignments[layer]
        started = time.perf_counter_ns()
        graph = ResidentMixedLayerGraph(
            self.checkpoint,
            self.cuda_library,
            self.shard_library,
            self.grouped_library,
            degree=int(assignment["degree"]),
            capture_state_arrays=True,
            # All per-layer graphs share the outer worker's process-global
            # native CUDA runtime.  Free their handles without shutting that
            # runtime down between transformer layers.
            shutdown_runtime_on_close=False,
        )
        try:
            graph.prepare_full_expert_banks(layer)
            graph.execute_layer_rows(
                layer,
                values,
                self._attnres_list(block_residuals, block_count),
            )
            checkpoint_reads = len(graph.loader.audit)
            graph.runtime.begin_replay()
            state_reset_ms = graph.runtime.reset_persistent_states()
            graph.loader.begin_replay()
            graph.quantizer.begin_replay()
            graph.worker_operations.clear()
            graph.reduction_records.clear()
            self._prepared_full_mixed[layer] = graph
            return {
                "wall_ms": (time.perf_counter_ns() - started) / 1e6,
                "checkpoint_reads": checkpoint_reads,
                "weight_uploads": graph.runtime.weight_uploads,
                "buffer_allocations": graph.runtime.allocations,
                "state_reset_ms": state_reset_ms,
                "resident_expert_banks": len(graph._full_expert_banks),
                "experts_per_bank": (
                    min(
                        len(bank.handles)
                        for bank in graph._full_expert_banks.values()
                    )
                    if graph._full_expert_banks
                    else 0
                ),
            }
        except BaseException:
            graph.close()
            raise

    def _execute_full_mixed_layer(
        self,
        *,
        layer: int,
        values: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
        positions: list[int],
        startup: dict[str, Any],
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Execute one prepared exact full-mixed placement and expose its state."""

        graph = self._prepared_full_mixed[layer]
        reads_before = len(graph.loader.audit)
        output, residuals, record = graph.execute_layer_rows(
            layer,
            values,
            self._attnres_list(block_residuals, block_count),
        )
        timed_reads = len(graph.loader.audit) - reads_before
        if timed_reads:
            raise RuntimeError("full-mixed hot call read checkpoint bytes")
        attention = dict(record["attention"])
        state_arrays = attention.pop("_state_arrays", None)
        if not isinstance(state_arrays, dict) or not state_arrays:
            raise RuntimeError("full-mixed attention omitted its physical state arrays")
        self.states[layer] = {
            str(name): np.ascontiguousarray(array, dtype=np.float32)
            for name, array in state_arrays.items()
        }
        is_snapshot = layer % self.config.residual_block == 0
        next_block_count = block_count + int(is_snapshot)
        if is_snapshot:
            block_residuals[:, block_count] = values
        if len(residuals) != next_block_count:
            raise RuntimeError("full-mixed AttnRes progression differs")
        route_record = record.get("routes")
        routes: list[dict[str, Any]] = []
        if route_record is not None:
            route_ids = list(route_record["selected_expert_ids"])
            route_weights = list(route_record["selected_weights"])
            routes = [
                {
                    "position": int(position),
                    "selected_expert_ids": [int(value) for value in ids],
                    "selected_weights": [float(value) for value in weights],
                }
                for position, ids, weights in zip(
                    positions, route_ids, route_weights, strict=True
                )
            ]
        assignment = self.assignments[layer]
        pieces = assignment["pieces"]
        for piece in pieces:
            self.logical_workers_instantiated.add(str(piece["node_id"]))
        operation_rows: list[dict[str, Any]] = []
        for operation in graph.worker_operations:
            row = dict(operation)
            stripe = row.get("stripe_index")
            piece_index = int(stripe) if stripe is not None else 0
            piece_index = min(piece_index, len(pieces) - 1)
            row["manifest_worker_id"] = str(pieces[piece_index]["node_id"])
            operation_rows.append(row)
        self.full_mixed_worker_receipts.append(
            {
                "layer": layer,
                "candidate_id": assignment["candidate_id"],
                "partition_type": assignment["partition_type"],
                "degree": assignment["degree"],
                "manifest_workers": [str(piece["node_id"]) for piece in pieces],
                "startup": startup,
                "checkpoint_reads_in_timed_region": timed_reads,
                "whole_layer_fallback": False,
                "native_operation_count": len(operation_rows),
                "native_operations": operation_rows,
                "reduction_records": [
                    dict(value) for value in graph.reduction_records
                ],
                "state_output": attention["state_output"],
            }
        )
        return output, next_block_count, {
            "layer": layer,
            "attention_type": attention["attention_type"],
            "mlp_type": "moe",
            "positions": positions,
            "input_fingerprint": _array_fingerprint(values),
            "output_fingerprint": _array_fingerprint(output),
            "state_input_fingerprint": None,
            "state_output": attention["state_output"],
            "routes": routes,
            "timing": {
                "load_ms": 0.0,
                "execute_ms": sum(
                    float(value) for value in graph.last_phase_timings.values()
                ),
            },
            "memory": {"resident_weight_bytes": 0},
            "backend_identity": (
                "resident_production_native_full_mixed_shard_dag_no_whole_layer_fallback"
            ),
            "full_mixed_native_operations": operation_rows,
        }

    def _dispatcher(self, worker_id: str) -> NativeShardDispatcher:
        if worker_id not in self.dispatchers:
            self.dispatchers[worker_id] = NativeShardDispatcher(worker_id)
        return self.dispatchers[worker_id]

    def _register_manifest(self) -> None:
        for layer, assignment in self.assignments.items():
            coordinator = str(assignment["pieces"][0]["node_id"])
            kind = str(assignment["partition_type"])
            task_type = (
                ShardTaskType.WHOLE_LAYER
                if kind == "WHOLE_LAYER"
                else ShardTaskType.ORDERED_LAYER_DAG
            )
            assignment_id = str(assignment["candidate_id"])

            def operation(
                values: np.ndarray,
                request: ShardRequest,
                *,
                expected_layer: int = layer,
            ) -> np.ndarray:
                if request.layer != expected_layer:
                    raise ValueError("layer request reached the wrong resident binding")
                pending = self._pending_layer[expected_layer]
                expected_kind = str(
                    self.assignments[expected_layer]["partition_type"]
                )
                if expected_kind == "FULL_MIXED_STRIPE":
                    output, block_count, evidence = self._execute_full_mixed_layer(
                        layer=expected_layer,
                        values=values,
                        block_residuals=pending["block_residuals"],
                        block_count=pending["block_count"],
                        positions=pending["positions"],
                        startup=pending["full_mixed_startup"],
                    )
                else:
                    output, block_count, evidence = super(
                        ManifestK3Runner, self
                    ).execute_layer(
                        expected_layer,
                        values,
                        pending["block_residuals"],
                        pending["block_count"],
                        pending["positions"],
                        maximum_context=pending["maximum_context"],
                    )
                pending["result"] = (block_count, evidence)
                return output

            native_name = {
                "WHOLE_LAYER": "KimiCudaGraphRunner.resident_whole_layer",
                "WHOLE_EXPERT": "ManifestK3Runner.whole_expert_layer_dag",
                "EXPERT_SHARD": "ManifestK3Runner.expert_sharded_layer_dag",
                "FULL_MIXED_STRIPE": (
                    "ResidentMixedLayerGraph.production_native_full_mixed_dag"
                ),
            }[kind]
            self._dispatcher(coordinator).register(
                assignment_id,
                task_type,
                CallableResidentPrimitive(native_name, operation),
            )
            self.layer_dispatch[layer] = (coordinator, task_type, assignment_id)

        endpoint_owner = self.endpoint_owners[0]

        def endpoint_operation(values: np.ndarray, _request: ShardRequest) -> np.ndarray:
            pending = self._pending_endpoint
            final_hidden, logits, evidence = super(
                ManifestK3Runner, self
            ).execute_final_head(
                values,
                pending["block_residuals"],
                pending["block_count"],
            )
            pending["result"] = (final_hidden, logits, evidence)
            return logits

        self._dispatcher(endpoint_owner).register(
            "kimi-k3-identical-endpoint-policy",
            ShardTaskType.ENDPOINT,
            CallableResidentPrimitive(
                "KimiCudaGraphRunner.resident_final_hidden_logits_argmax",
                endpoint_operation,
            ),
        )

    def _authenticated_execute(
        self,
        *,
        dispatcher: NativeShardDispatcher,
        request: ShardRequest,
        values: np.ndarray,
        request_id: str,
    ) -> tuple[Any, np.ndarray, float]:
        frame = Frame(
            MessageType.EXECUTE_SHARD,
            request_id,
            request.rows,
            dispatcher.worker_id,
            request.state_id,
            encode_shard_request(request, values),
        )
        started = time.perf_counter_ns()
        authenticated = decode_frame(encode_frame(frame, self.credential), self.credential)
        result_frame = dispatcher.execute_frame(authenticated)
        authenticated_result = decode_frame(
            encode_frame(result_frame, self.credential), self.credential
        )
        result, output = decode_shard_result(authenticated_result.payload)
        protocol_wall_ms = (time.perf_counter_ns() - started) / 1e6
        self.logical_workers_instantiated.add(dispatcher.worker_id)
        return result, output, protocol_wall_ms

    def _ensure_expert_worker(self) -> None:
        if self._expert_process is not None:
            if not self._expert_process.is_alive():
                raise RuntimeError("persistent expert worker exited unexpectedly")
            return
        self.expert_worker_process_audit.update(
            {
                "required": True,
                "ready_status": "STARTING",
                "close_status": "PENDING",
            }
        )
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_persistent_expert_worker,
            kwargs={
                "connection": child,
                "credential": self.credential,
                "checkpoint": str(self.checkpoint),
                "cuda_library": str(self.cuda_library),
                "grouped_library": str(self.grouped_library),
            },
            name="e022-persistent-manifest-expert-worker",
        )
        process.start()
        child.close()
        if not parent.poll(300):
            process.terminate()
            process.join(timeout=30)
            self.expert_worker_process_audit.update(
                {
                    "ready_status": "FAIL",
                    "pid": process.pid,
                    "exitcode": process.exitcode,
                    "error": "persistent expert worker did not become ready",
                }
            )
            raise TimeoutError("persistent expert worker did not become ready")
        ready = parent.recv()
        if ready.get("status") != "READY":
            process.terminate()
            process.join(timeout=30)
            self.expert_worker_process_audit.update(
                {
                    "ready_status": "FAIL",
                    "pid": process.pid,
                    "exitcode": process.exitcode,
                    "error": f"persistent expert worker startup failed: {ready}",
                }
            )
            raise RuntimeError(f"persistent expert worker startup failed: {ready}")
        self._expert_connection = parent
        self._expert_process = process
        self.expert_worker_process_startups += 1
        self.expert_worker_process_audit.update(
            {
                "successful_startups": self.expert_worker_process_startups,
                "ready_status": "PASS",
                "pid": int(ready["pid"]),
                "checkpoint_index_sha256": ready["checkpoint_index_sha256"],
            }
        )

    def _expert_layer_process(
        self,
        *,
        layer: int,
        assignment: dict[str, Any],
        packed: np.ndarray,
    ) -> tuple[list[tuple[Any, np.ndarray, float, dict[str, Any]]], float]:
        self._ensure_expert_worker()
        if self._expert_connection is None or self._expert_process is None:
            raise RuntimeError("persistent expert worker was not initialized")
        degree = int(assignment["degree"])
        partition_type = str(assignment["partition_type"])
        if partition_type == "EXPERT_SHARD":
            task_type = ShardTaskType.EXPERT_STRIPE
            assignment_stem = "expert-stripe"
        elif partition_type == "WHOLE_EXPERT":
            task_type = ShardTaskType.WHOLE_EXPERT_GROUP
            assignment_stem = "whole-expert-group"
        else:
            raise ValueError(
                f"expert process cannot execute partition {partition_type}"
            )
        pieces: list[dict[str, Any]] = []
        for piece in assignment["pieces"]:
            shard_index = int(piece["shard_index"])
            worker_id = str(piece["node_id"])
            assignment_id = (
                f"{assignment['candidate_id']}:{assignment_stem}-{shard_index:02d}"
            )
            request = ShardRequest(
                assignment_id,
                task_type,
                layer,
                shard_index,
                degree,
                packed.shape[0],
                tuple(int(value) for value in packed.shape),
                state_id=(
                    f"manifest:{self.manifest['inventory_id']}:layer-{layer}:experts"
                ),
            )
            frame = Frame(
                MessageType.EXECUTE_SHARD,
                f"manifest-layer-{layer:02d}-expert-{shard_index:02d}",
                packed.shape[0],
                worker_id,
                request.state_id,
                encode_shard_request(request, packed),
            )
            pieces.append(
                {
                    "worker_id": worker_id,
                    "assignment_id": assignment_id,
                    "shard_index": shard_index,
                    "encoded_frame": encode_frame(frame, self.credential),
                }
            )
        started = time.perf_counter_ns()
        self._expert_connection.send(
            {
                "command": "EXECUTE_EXPERT_LAYER",
                "layer": layer,
                "degree": degree,
                "rows": packed.shape[0],
                "partition_type": partition_type,
                "pieces": pieces,
            }
        )
        if not self._expert_connection.poll(7200):
            self._expert_process.terminate()
            self._expert_process.join(timeout=30)
            raise TimeoutError(f"persistent expert layer {layer} timed out")
        response = self._expert_connection.recv()
        layer_wall_ms = (time.perf_counter_ns() - started) / 1e6
        if response.get("status") != "PASS":
            raise RuntimeError(f"persistent expert worker failed: {response}")
        if int(response.get("checkpoint_reads_in_timed_region", -1)) != 0:
            raise RuntimeError("persistent expert worker read checkpoint data while timed")
        response_rows = list(response.get("responses", ()))
        if len(response_rows) != degree:
            raise RuntimeError("persistent expert worker omitted assignment responses")
        by_shard = {int(row["shard_index"]): row for row in response_rows}
        outputs: list[tuple[Any, np.ndarray, float, dict[str, Any]]] = []
        for piece in pieces:
            shard_index = int(piece["shard_index"])
            row = by_shard[shard_index]
            result_frame = decode_frame(row["encoded_result"], self.credential)
            result, output = decode_shard_result(result_frame.payload)
            audit = {
                **row,
                "status": "PASS",
                "pid": response["pid"],
                "startup": response["startup"],
                "worker_process_persistent": True,
                "worker_process_exitcode": None,
                "worker_process_layer_wall_ms": layer_wall_ms,
                "checkpoint_reads_in_timed_region": response[
                    "checkpoint_reads_in_timed_region"
                ],
            }
            outputs.append((result, output, float(row["protocol_wall_ms"]), audit))
            self.logical_workers_instantiated.add(str(piece["worker_id"]))
        return outputs, layer_wall_ms

    def _close_expert_worker(self) -> None:
        connection = self._expert_connection
        process = self._expert_process
        self._expert_connection = None
        self._expert_process = None
        if connection is None or process is None:
            if self.expert_worker_process_audit["required"]:
                self.expert_worker_process_audit.update(
                    {
                        "close_status": "FAIL",
                        "error": "required persistent expert worker was absent at close",
                    }
                )
            return
        try:
            if process.is_alive():
                connection.send({"command": "CLOSE"})
                if connection.poll(60):
                    closed = connection.recv()
                    if closed.get("status") != "CLOSED":
                        raise RuntimeError(f"persistent worker close failed: {closed}")
                    self.expert_worker_process_audit["close_frame"] = closed
                else:
                    raise TimeoutError("persistent expert worker omitted close receipt")
                process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
                raise TimeoutError("persistent expert worker did not close")
            self.expert_worker_process_audit["exitcode"] = process.exitcode
            if process.exitcode != 0:
                raise RuntimeError(
                    f"persistent expert worker exited with {process.exitcode}"
                )
            self.expert_worker_process_audit["close_status"] = "PASS"
            for receipt in self.expert_worker_receipts:
                receipt["worker_process_exitcode"] = process.exitcode
        except BaseException as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
            self.expert_worker_process_audit.update(
                {
                    "close_status": "FAIL",
                    "exitcode": process.exitcode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            connection.close()

    def close(self) -> None:
        try:
            self._close_expert_worker()
        finally:
            for graph in self._prepared_full_mixed.values():
                graph.close()
            self._prepared_full_mixed.clear()
            super().close()

    def execute_layer(
        self,
        layer: int,
        hidden_rows: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
        positions: list[int],
        *,
        maximum_context: int,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        worker, task_type, assignment_id = self.layer_dispatch[layer]
        assignment = self.assignments[layer]
        values = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        full_mixed_startup = None
        if assignment["partition_type"] == "FULL_MIXED_STRIPE":
            full_mixed_startup = self._prepare_full_mixed_layer(
                layer=layer,
                values=values,
                block_residuals=block_residuals,
                block_count=block_count,
            )
        self._pending_layer[layer] = {
            "block_residuals": block_residuals,
            "block_count": block_count,
            "positions": positions,
            "maximum_context": maximum_context,
            "full_mixed_startup": full_mixed_startup,
        }
        request = ShardRequest(
            assignment_id,
            task_type,
            layer,
            0,
            int(assignment["degree"]),
            values.shape[0],
            tuple(int(value) for value in values.shape),
            state_id=f"manifest:{self.manifest['inventory_id']}:layer-{layer}",
        )
        try:
            result, output, protocol_wall_ms = self._authenticated_execute(
                dispatcher=self._dispatcher(worker),
                request=request,
                values=values,
                request_id=f"manifest-layer-{layer:02d}",
            )
            next_count, evidence = self._pending_layer.pop(layer)["result"]
        finally:
            graph = self._prepared_full_mixed.pop(layer, None)
            if graph is not None:
                graph.close()
        evidence["manifest_assignment"] = assignment
        evidence["authenticated_execute_shard"] = True
        evidence["native_primitive"] = result.native_primitive
        evidence["worker_protocol_wall_ms"] = protocol_wall_ms
        evidence["attnres_state_fingerprint"] = _array_fingerprint(
            np.ascontiguousarray(block_residuals[:, :next_count])
        )
        attnres_state = np.ascontiguousarray(
            block_residuals[:, :next_count], dtype=np.float32
        )
        self._record_state_trace(
            layer=layer,
            attention_type=str(evidence["attention_type"]),
            attnres=attnres_state,
        )
        self._compare_state_reference(
            layer=layer,
            attention_type=str(evidence["attention_type"]),
            attnres=attnres_state,
        )
        self.dispatch_receipts.append(
            {
                "layer": layer,
                "worker_id": worker,
                "assignment_id": assignment_id,
                "task_type": task_type.value,
                "native_primitive": result.native_primitive,
                "worker_native_wall_ms": result.wall_ms,
                "worker_protocol_wall_ms": protocol_wall_ms,
                "output_sha256": result.output_sha256,
            }
        )
        return output, int(next_count), evidence

    def _execute_sparse_mlp_rows(
        self,
        resources: _LayerResources,
        weights: dict[str, Any],
        mlp_inputs: ctypes.c_void_p,
        prefix_rows: ctypes.c_void_p,
        routes: list[dict[str, Any]],
        *,
        layer: int,
    ) -> None:
        assignment = self.assignments[layer]
        if assignment["partition_type"] == "WHOLE_LAYER":
            return super()._execute_sparse_mlp_rows(
                resources,
                weights,
                mlp_inputs,
                prefix_rows,
                routes,
                layer=layer,
            )
        if assignment["partition_type"] not in {"EXPERT_SHARD", "WHOLE_EXPERT"}:
            raise RuntimeError("manifest runner reached an unsupported partition")
        config = self.config
        rows = len(routes)
        latent_device = resources.allocate(rows * config.latent)
        for row in range(rows):
            self.runtime.execute_dense(
                weights["latent_down"],
                _pointer_offset(latent_device, row * config.latent),
                _pointer_offset(mlp_inputs, row * config.hidden),
                1,
            )
        self.runtime.synchronize()
        latent = self.runtime.download_activation(latent_device, (rows, config.latent))
        route_ids = np.asarray(
            [row["selected_expert_ids"] for row in routes], dtype=np.float32
        )
        route_weights = np.asarray(
            [row["selected_weights"] for row in routes], dtype=np.float32
        )
        packed = np.ascontiguousarray(
            np.concatenate([latent, route_ids, route_weights], axis=1),
            dtype=np.float32,
        )
        partials: list[np.ndarray] = []
        executions, layer_worker_wall_ms = self._expert_layer_process(
            layer=layer,
            assignment=assignment,
            packed=packed,
        )
        if assignment["partition_type"] == "WHOLE_EXPERT":
            expected_route_hash = _sha256_arrays(
                (np.rint(route_ids).astype(np.int32),)
            )
            expected_weight_hash = _sha256_arrays(
                (np.ascontiguousarray(route_weights, dtype=np.float32),)
            )
        else:
            expected_route_hash = _sha256_array(
                np.rint(route_ids).astype(np.int32)
            )
            expected_weight_hash = _sha256_array(
                np.ascontiguousarray(route_weights, dtype=np.float32)
            )
        for piece, execution in zip(
            assignment["pieces"], executions, strict=True
        ):
            shard_index = int(piece["shard_index"])
            worker_id = str(piece["node_id"])
            result, partial, protocol_wall_ms, audit = execution
            route_hash_equal = (
                audit["last_execution"].get("route_ids_sha256")
                == expected_route_hash
            )
            route_weight_hash_equal = (
                audit["last_execution"].get("route_weights_sha256")
                == expected_weight_hash
            )
            if not route_hash_equal or not route_weight_hash_equal:
                raise RuntimeError("expert worker changed ordered route metadata")
            partials.append(partial)
            self.maximum_actual_resident_bytes = max(
                self.maximum_actual_resident_bytes, int(audit["resident_bytes"])
            )
            self.expert_worker_receipts.append(
                {
                    "layer": layer,
                    "candidate_id": assignment["candidate_id"],
                    "partition_type": assignment["partition_type"],
                    "worker_id": worker_id,
                    "shard_index": shard_index,
                    "degree": assignment["degree"],
                    "native_primitive": result.native_primitive,
                    "output_sha256": result.output_sha256,
                    "worker_native_wall_ms": result.wall_ms,
                    "worker_protocol_wall_ms": protocol_wall_ms,
                    "worker_layer_process_wall_ms": layer_worker_wall_ms,
                    "resident_bytes": audit["resident_bytes"],
                    "startup": audit["startup"],
                    "checkpoint_reads_in_timed_region": audit["last_execution"][
                        "checkpoint_reads_in_timed_region"
                    ],
                    "route_ids_sha256": audit["last_execution"][
                        "route_ids_sha256"
                    ],
                    "route_weights_sha256": audit["last_execution"][
                        "route_weights_sha256"
                    ],
                    "route_ids_equal": route_hash_equal,
                    "route_weights_equal": route_weight_hash_equal,
                    "whole_layer_fallback": False,
                    "complete_expert_bank_resident": audit[
                        "complete_expert_bank_resident"
                    ],
                    "complete_assigned_expert_group": bool(
                        audit.get("complete_assigned_expert_group", False)
                    ),
                    "expert_id_start": audit.get("expert_id_start"),
                    "expert_id_stop_exclusive": audit.get(
                        "expert_id_stop_exclusive"
                    ),
                    "worker_process_pid": audit["pid"],
                    "worker_process_persistent": audit[
                        "worker_process_persistent"
                    ],
                    "worker_process_exitcode": audit["worker_process_exitcode"],
                    "dispatcher_audit": audit["dispatcher_audit"],
                }
            )

        partial_devices = [resources.upload_data(partial) for partial in partials]
        reduced = resources.allocate(rows * config.latent)
        for row in range(rows):
            destination = _pointer_offset(reduced, row * config.latent)
            self.runtime.execute_copy(
                destination,
                _pointer_offset(partial_devices[0], row * config.latent),
                config.latent,
            )
            for source in partial_devices[1:]:
                self.runtime.execute_add(
                    destination,
                    _pointer_offset(source, row * config.latent),
                    config.latent,
                )
        self.runtime.execute_rmsnorm(
            reduced,
            reduced,
            weights["routed_norm"],
            batch=rows,
            dimension=config.latent,
            epsilon=config.epsilon,
        )
        routed_output = resources.allocate(rows * config.hidden)
        shared_output = resources.allocate(rows * config.hidden)
        self.runtime.execute_dense(
            weights["latent_up"], routed_output, reduced, rows
        )
        self.runtime.execute_resident(
            weights["shared_mlp"], shared_output, mlp_inputs, rows
        )
        self.runtime.execute_add(
            routed_output, shared_output, rows * config.hidden
        )
        for row in range(rows):
            self.runtime.execute_add(
                _pointer_offset(prefix_rows, row * config.hidden),
                _pointer_offset(routed_output, row * config.hidden),
                config.hidden,
            )
        self.operation_counts["MXFP4_routed_expert"] += rows * config.topk
        self.operation_counts["MoE_reduction"] += rows
        self.operation_counts["shared_experts"] += rows
        self.operation_counts["dense_projection"] += rows * 5

    def execute_final_head(
        self,
        hidden_rows: np.ndarray,
        block_residuals: np.ndarray,
        block_count: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        values = np.ascontiguousarray(hidden_rows, dtype=np.float32)
        self._pending_endpoint = {
            "block_residuals": block_residuals,
            "block_count": block_count,
        }
        worker = self.endpoint_owners[0]
        request = ShardRequest(
            "kimi-k3-identical-endpoint-policy",
            ShardTaskType.ENDPOINT,
            LAYERS,
            0,
            1,
            values.shape[0],
            tuple(int(value) for value in values.shape),
            state_id=f"manifest:{self.manifest['inventory_id']}:endpoint",
        )
        result, _logit_output, protocol_wall_ms = self._authenticated_execute(
            dispatcher=self._dispatcher(worker),
            request=request,
            values=values,
            request_id="manifest-endpoint",
        )
        final_hidden, logits, evidence = self._pending_endpoint.pop("result")
        evidence["manifest_endpoint_owners"] = self.endpoint_owners
        evidence["executed_endpoint_owner"] = worker
        evidence["authenticated_execute_shard"] = True
        evidence["native_primitive"] = result.native_primitive
        evidence["worker_protocol_wall_ms"] = protocol_wall_ms
        self.dispatch_receipts.append(
            {
                "layer": LAYERS,
                "worker_id": worker,
                "assignment_id": request.assignment_id,
                "task_type": ShardTaskType.ENDPOINT.value,
                "native_primitive": result.native_primitive,
                "worker_native_wall_ms": result.wall_ms,
                "worker_protocol_wall_ms": protocol_wall_ms,
                "output_sha256": result.output_sha256,
            }
        )
        return final_hidden, logits, evidence


def execute_manifest(
    *,
    selection_id: str,
    selection_case: str,
    manifest_path: Path,
    expected_manifest_sha256: str,
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    shard_library: Path | None,
    oracle_root: Path,
    state_trace_root: Path | None = None,
    state_reference_root: Path | None = None,
) -> dict[str, Any]:
    actual_hash = _sha256_file(manifest_path)
    if actual_hash != expected_manifest_sha256:
        raise ValueError("frozen representative manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = _manifest_assignments(manifest)
    trace = np.memmap(
        oracle_root / "hidden-trace.f32",
        mode="r",
        dtype="<f4",
        shape=(3 * (LAYERS + 1), HIDDEN),
    )
    logits = np.memmap(
        oracle_root / "prefill-logits.f32",
        mode="r",
        dtype="<f4",
        shape=(2, 163840),
    )
    routes = _parse_oracle_routes(oracle_root / "routes.txt")
    started = time.perf_counter_ns()
    runner = ManifestK3Runner(
        checkpoint,
        cuda_library,
        grouped_library,
        manifest,
        shard_library=shard_library,
        state_trace_root=state_trace_root,
        state_reference_root=state_reference_root,
    )
    try:
        result = runner.execute_pass(
            [163584],
            [0],
            layer_limit=LAYERS,
            maximum_context=256,
            oracle_trace=trace,
            oracle_layer_count=LAYERS,
            oracle_routes=routes,
            oracle_logits=logits,
            progress_label=f"e022-{selection_id}",
        )
    except BaseException:
        with suppress(BaseException):
            runner.close()
        raise

    close_error: str | None = None
    try:
        runner.close()
    except BaseException as exc:
        close_error = f"{type(exc).__name__}: {exc}"
    state_trace_receipt = (
        runner.finalize_state_trace(
            selection_id=selection_id,
            manifest_sha256=actual_hash,
        )
        if close_error is None
        else None
    )
    state_reference_validation = runner.state_reference_summary()

    layers = result["layers"]
    state_rows = [
        {
            "layer": int(row["layer"]),
            "attention_type": row["attention_type"],
            "state_fingerprint": row["state_output"]["fingerprint"],
            "state_bytes": row["state_output"]["bytes"],
            "state_finite": row["state_output"]["finite"],
            "attnres_state_fingerprint": row["attnres_state_fingerprint"],
        }
        for row in layers
    ]
    hidden_errors = [
        float(row["oracle_correctness"]["relative_l2_error"]) for row in layers
    ]
    logit_l2 = float(result["head"]["oracle_logits"]["relative_l2_error"])
    greedy_reference = int(np.argmax(logits[0]))
    greedy_actual = int(result["sampled_token_id"])
    split_assignments = {
        layer: assignment
        for layer, assignment in assignments.items()
        if assignment["partition_type"] in {"EXPERT_SHARD", "WHOLE_EXPERT"}
    }
    expected_expert_dispatches = sum(
        int(assignment["degree"]) for assignment in split_assignments.values()
    )
    expected_expert_by_layer = Counter(
        {
            int(layer): int(assignment["degree"])
            for layer, assignment in split_assignments.items()
        }
    )
    actual_expert_by_layer = Counter(
        int(row["layer"]) for row in runner.expert_worker_receipts
    )
    all_route_hashes_exact = all(
        row.get("route_ids_equal") is True
        and row.get("route_weights_equal") is True
        for row in runner.expert_worker_receipts
    )
    stripe_bank_complete = all(
        bool(row.get("complete_expert_bank_resident"))
        for row in runner.expert_worker_receipts
        if row.get("partition_type") == "EXPERT_SHARD"
    )
    whole_expert_ownership_complete = True
    for split_layer, assignment in split_assignments.items():
        if assignment["partition_type"] != "WHOLE_EXPERT":
            continue
        rows = sorted(
            (
                int(row["expert_id_start"]),
                int(row["expert_id_stop_exclusive"]),
            )
            for row in runner.expert_worker_receipts
            if int(row["layer"]) == int(split_layer)
            and row.get("partition_type") == "WHOLE_EXPERT"
            and row.get("complete_assigned_expert_group") is True
        )
        whole_expert_ownership_complete = whole_expert_ownership_complete and (
            len(rows) == int(assignment["degree"])
            and rows[0][0] == 0
            and rows[-1][1] == ROUTED_EXPERTS
            and all(
                left[1] == right[0]
                for left, right in pairwise(rows)
            )
        )
    complete_bank = (
        len(runner.expert_worker_receipts) == expected_expert_dispatches
        and stripe_bank_complete
        and whole_expert_ownership_complete
    )
    timed_checkpoint_reads = sum(
        int(row.get("checkpoint_reads_in_timed_region", -1))
        for row in runner.expert_worker_receipts
    )
    timed_checkpoint_reads += sum(
        int(row.get("checkpoint_reads_in_timed_region", -1))
        for row in runner.full_mixed_worker_receipts
    )

    outer_dispatch_audits = [
        row
        for dispatcher in runner.dispatchers.values()
        for row in dispatcher.audit
    ]
    outer_keys = Counter(
        (
            str(row["worker_id"]),
            str(row["assignment_id"]),
            str(row["task_type"]),
            str(row["native_primitive"]),
        )
        for row in outer_dispatch_audits
    )
    receipt_keys = Counter(
        (
            str(row["worker_id"]),
            str(row["assignment_id"]),
            str(row["task_type"]),
            str(row["native_primitive"]),
        )
        for row in runner.dispatch_receipts
    )
    outer_dispatch_valid = (
        len(outer_dispatch_audits) == LAYERS + 1
        and len(runner.dispatch_receipts) == LAYERS + 1
        and outer_keys == receipt_keys
        and all(
            row.get("whole_layer_fallback") is False
            for row in outer_dispatch_audits
        )
    )
    def expected_expert_binding(row: dict[str, Any]) -> tuple[str, str]:
        if row.get("partition_type") == "WHOLE_EXPERT":
            return (
                f"{row['candidate_id']}:whole-expert-group-{int(row['shard_index']):02d}",
                ShardTaskType.WHOLE_EXPERT_GROUP.value,
            )
        return (
            f"{row['candidate_id']}:expert-stripe-{int(row['shard_index']):02d}",
            ShardTaskType.EXPERT_STRIPE.value,
        )

    expert_dispatch_valid = (
        len(runner.expert_worker_receipts) == expected_expert_dispatches
        and actual_expert_by_layer == expected_expert_by_layer
        and all(
            len(row.get("dispatcher_audit", ())) == 1
            and row["dispatcher_audit"][0].get("worker_id") == row["worker_id"]
            and row["dispatcher_audit"][0].get("assignment_id")
            == expected_expert_binding(row)[0]
            and row["dispatcher_audit"][0].get("task_type")
            == expected_expert_binding(row)[1]
            and row["dispatcher_audit"][0].get("native_primitive")
            == row["native_primitive"]
            and row["dispatcher_audit"][0].get("whole_layer_fallback") is False
            for row in runner.expert_worker_receipts
        )
    )
    expected_full_mixed_layers = {
        int(layer)
        for layer, assignment in assignments.items()
        if assignment["partition_type"] == "FULL_MIXED_STRIPE"
    }
    actual_full_mixed_layers = {
        int(row["layer"]) for row in runner.full_mixed_worker_receipts
    }
    full_mixed_dispatch_valid = (
        actual_full_mixed_layers == expected_full_mixed_layers
        and len(runner.full_mixed_worker_receipts)
        == len(expected_full_mixed_layers)
        and all(
            int(row.get("checkpoint_reads_in_timed_region", -1)) == 0
            and row.get("whole_layer_fallback") is False
            and len(row.get("manifest_workers", ())) == int(row["degree"])
            and {
                "latent_down_projection_stripe",
                "grouped_expert_stripe_bank_top16",
                "latent_up_projection_stripe",
                "shared_expert_stripe",
            }.issubset(
                {
                    str(operation.get("operator"))
                    for operation in row.get("native_operations", ())
                }
            )
            and any(
                str(operation.get("operator", "")).endswith(
                    "_attention_stripe"
                )
                for operation in row.get("native_operations", ())
            )
            and bool(row.get("reduction_records"))
            for row in runner.full_mixed_worker_receipts
        )
    )
    no_whole_layer_fallback = (
        all(
            row.get("whole_layer_fallback") is False
            for row in outer_dispatch_audits
        )
        and all(
            row.get("whole_layer_fallback") is False
            for row in runner.expert_worker_receipts
        )
        and all(
            row.get("whole_layer_fallback") is False
            for row in runner.full_mixed_worker_receipts
        )
    )
    expected_workers = _manifest_execution_workers(
        assignments, runner.endpoint_owners[0]
    )
    instantiated_workers = set(runner.logical_workers_instantiated)
    logical_workers_exact = instantiated_workers == expected_workers
    process_audit = runner.expert_worker_process_audit
    if expected_expert_dispatches:
        process_valid = (
            process_audit.get("required") is True
            and int(process_audit.get("successful_startups", -1)) == 1
            and process_audit.get("ready_status") == "PASS"
            and process_audit.get("close_status") == "PASS"
            and int(process_audit.get("exitcode", -1)) == 0
            and close_error is None
            and {
                int(row["worker_process_pid"])
                for row in runner.expert_worker_receipts
            }
            == {int(process_audit["pid"])}
            and all(
                row.get("worker_process_persistent") is True
                and int(row.get("worker_process_exitcode", -1)) == 0
                for row in runner.expert_worker_receipts
            )
        )
    else:
        process_valid = (
            process_audit.get("required") is False
            and int(process_audit.get("successful_startups", -1)) == 0
            and process_audit.get("ready_status") == "NOT_APPLICABLE"
            and process_audit.get("close_status") == "NOT_APPLICABLE"
            and not runner.expert_worker_receipts
            and close_error is None
        )
    authenticated_execute_shard = (
        outer_dispatch_valid
        and expert_dispatch_valid
        and full_mixed_dispatch_valid
    )
    complete_tensor_coverage = (
        manifest["checkpoint_reconciliation"]["gap_bytes"] == 0
        and manifest["checkpoint_reconciliation"]["overlap_bytes"] == 0
    )
    all_state_finite = all(bool(row["state_finite"]) for row in state_rows)
    passed = (
        complete_tensor_coverage
        and len(assignments) == LAYERS
        and len(layers) == LAYERS
        and result["routing_equality"]
        and all_route_hashes_exact
        and max(hidden_errors) <= RELATIVE_L2_GATE
        and logit_l2 <= RELATIVE_L2_GATE
        and greedy_actual == greedy_reference
        and all_state_finite
        and complete_bank
        and timed_checkpoint_reads == 0
        and authenticated_execute_shard
        and no_whole_layer_fallback
        and logical_workers_exact
        and process_valid
        and (
            state_reference_root is None
            or state_reference_validation["status"] == "PASS"
        )
    )
    partition_counts = Counter(
        str(value["partition_type"]) for value in assignments.values()
    )
    return {
        "schema_version": "experiment-022-completion-manifest-correctness-v2",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "PHYSICAL sequential logical workers on one RTX 5090",
        "selection_id": selection_id,
        "selection_case": selection_case,
        "manifest_id": f"{manifest['inventory_id']}-{manifest['planner_level']}",
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_hash,
        "inventory_id": manifest["inventory_id"],
        "planner_level": manifest["planner_level"],
        "chunk_rows": manifest["chunk_rows"],
        "complete_tensor_assignment_coverage": complete_tensor_coverage,
        "assigned_layers": len(assignments),
        "partition_types_used": dict(sorted(partition_counts.items())),
        "manifest_worker_count": len(manifest["nodes"]),
        "logical_workers_expected": len(expected_workers),
        "logical_worker_ids_expected": sorted(expected_workers),
        "logical_workers_instantiated": len(instantiated_workers),
        "logical_worker_ids_instantiated": sorted(instantiated_workers),
        "logical_worker_instantiation_exact": logical_workers_exact,
        "maximum_manifest_resident_bytes": max(
            int(node["assigned_memory_bytes"]) for node in manifest["nodes"]
        ),
        "maximum_physically_instantiated_shard_resident_bytes": runner.maximum_actual_resident_bytes,
        "complete_93_layer_traversal": len(layers) == LAYERS,
        "route_equality": bool(result["routing_equality"] and all_route_hashes_exact),
        "ordered_expert_equality": bool(
            result["routing_equality"] and all_route_hashes_exact
        ),
        "kda_state_fingerprints": [
            row for row in state_rows if row["attention_type"] == "KDA"
        ],
        "mla_state_fingerprints": [
            row for row in state_rows if row["attention_type"] == "Gated_MLA"
        ],
        "attnres_state_fingerprints": [
            {
                "layer": row["layer"],
                "fingerprint": row["attnres_state_fingerprint"],
            }
            for row in state_rows
        ],
        "all_state_finite": all_state_finite,
        "hidden_relative_l2_maximum": max(hidden_errors),
        "hidden_relative_l2_by_layer": hidden_errors,
        "logit_relative_l2": logit_l2,
        "greedy_token_actual": greedy_actual,
        "greedy_token_reference": greedy_reference,
        "greedy_token_equality": greedy_actual == greedy_reference,
        "authenticated_execute_shard": authenticated_execute_shard,
        "authentication_evidence": {
            "outer_expected_roundtrips": LAYERS + 1,
            "outer_dispatcher_audits": len(outer_dispatch_audits),
            "outer_receipts": len(runner.dispatch_receipts),
            "outer_audits_match_receipts": outer_keys == receipt_keys,
            "expert_expected_roundtrips": expected_expert_dispatches,
            "expert_dispatcher_audits": sum(
                len(row.get("dispatcher_audit", ()))
                for row in runner.expert_worker_receipts
            ),
            "expert_layer_cardinality_exact": (
                actual_expert_by_layer == expected_expert_by_layer
            ),
            "full_mixed_expected_layers": sorted(expected_full_mixed_layers),
            "full_mixed_actual_layers": sorted(actual_full_mixed_layers),
            "full_mixed_dispatch_valid": full_mixed_dispatch_valid,
            "all_ordered_route_hashes_exact": all_route_hashes_exact,
        },
        "whole_layer_fallback_for_split_layers": not no_whole_layer_fallback,
        "full_expert_bank_resident_for_every_split_task": complete_bank,
        "complete_routed_expert_ownership_for_split_tasks": complete_bank,
        "expert_stripe_full_banks_complete": stripe_bank_complete,
        "whole_expert_disjoint_ownership_complete": (
            whole_expert_ownership_complete
        ),
        "checkpoint_reads_in_expert_timed_regions": timed_checkpoint_reads,
        "persistent_expert_worker": process_audit,
        "persistent_expert_worker_gate": process_valid,
        "runner_close_error": close_error,
        "state_trace": state_trace_receipt,
        "state_reference_validation": state_reference_validation,
        "layer_dispatch_receipts": runner.dispatch_receipts,
        "expert_worker_receipts": runner.expert_worker_receipts,
        "full_mixed_worker_receipts": runner.full_mixed_worker_receipts,
        "endpoint": result["head"],
        "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "local_execution_boundary": (
            "logical manifest workers were multiplexed sequentially on one physical RTX 5090; "
            "this is correctness and production-path evidence, not physical swarm throughput"
        ),
    }


def _worker(arguments: dict[str, str], output: str) -> None:
    try:
        receipt = execute_manifest(
            selection_id=arguments["selection_id"],
            selection_case=arguments["selection_case"],
            manifest_path=Path(arguments["manifest"]),
            expected_manifest_sha256=arguments["manifest_sha256"],
            checkpoint=Path(arguments["checkpoint"]),
            cuda_library=Path(arguments["cuda_library"]),
            grouped_library=Path(arguments["grouped_library"]),
            shard_library=(
                Path(arguments["shard_library"])
                if arguments.get("shard_library")
                else None
            ),
            oracle_root=Path(arguments["oracle_root"]),
            state_trace_root=(
                Path(arguments["state_trace_root"])
                if arguments.get("state_trace_root")
                else None
            ),
            state_reference_root=(
                Path(arguments["state_reference_root"])
                if arguments.get("state_reference_root")
                else None
            ),
        )
    except BaseException as exc:
        receipt = {
            "schema_version": "experiment-022-completion-manifest-correctness-v2",
            "status": "FAIL",
            "selection_id": arguments["selection_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-id", required=True)
    parser.add_argument("--selection-case", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--shard-library", type=Path)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-trace-root", type=Path)
    parser.add_argument("--state-reference-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=10800.0)
    args = parser.parse_args()
    mapping = {
        "selection_id": args.selection_id,
        "selection_case": args.selection_case,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": args.manifest_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "cuda_library": str(args.cuda_library.resolve()),
        "grouped_library": str(args.grouped_library.resolve()),
        "shard_library": (
            str(args.shard_library.resolve()) if args.shard_library else ""
        ),
        "oracle_root": str(args.oracle_root.resolve()),
        "state_trace_root": (
            str(args.state_trace_root.resolve()) if args.state_trace_root else ""
        ),
        "state_reference_root": (
            str(args.state_reference_root.resolve())
            if args.state_reference_root
            else ""
        ),
    }
    context = mp.get_context("spawn")
    process = context.Process(
        target=_worker,
        args=(mapping, str(args.output.resolve())),
        name=f"e022-{args.selection_id}-manifest-worker",
    )
    started = time.perf_counter_ns()
    process.start()
    process.join(timeout=args.timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
        raise TimeoutError(f"{args.selection_id} manifest correctness timed out")
    if not args.output.is_file():
        raise RuntimeError("manifest worker did not create a receipt")
    receipt = json.loads(args.output.read_text(encoding="utf-8"))
    receipt["worker_process"] = {
        "pid": process.pid,
        "exitcode": process.exitcode,
        "spawn_method": "spawn",
        "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
    }
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selection_id": args.selection_id,
                "status": receipt["status"],
                "output": str(args.output),
                "wall_seconds": receipt["worker_process"]["wall_seconds"],
            }
        )
    )
    return 0 if receipt["status"] == "PASS" and process.exitcode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
