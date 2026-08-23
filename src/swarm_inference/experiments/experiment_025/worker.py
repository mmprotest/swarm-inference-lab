"""Independent physical E025 worker process and real native dispatch boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import socketserver
import ssl
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_cuda_runtime import _array_fingerprint, _sha256_file
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_020.transport import Frame, MessageType
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_worker import LoadStageRequest

from .constants import (
    LATENT_SIZE,
    MODEL_ID,
    MODEL_REVISION,
    SUB_LAYER_TARGET,
    SUB_LAYER_WORKERS,
    TOP_K,
    TRANSFORMER_LAYERS,
)
from .expert_partition import ExpertPartitionExecutor
from .io import append_jsonl, atomic_write_json, read_json, sha256_file, utc_now
from .wire import (
    Action,
    AuthenticatedConnection,
    pack_payload,
    recv_frame,
    send_frame,
    server_ssl_context,
    unpack_payload,
)

SCHEMA_VERSION = "experiment-025-worker-runtime-v1"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _execution_receipt(value: dict[str, Any]) -> dict[str, Any]:
    """Remove model arrays while retaining attributable numerical evidence."""

    receipt: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.ndarray):
            receipt[f"{key}_fingerprint"] = _array_fingerprint(item)
            receipt[f"{key}_shape"] = [int(dimension) for dimension in item.shape]
            receipt[f"{key}_dtype"] = str(item.dtype)
        else:
            receipt[key] = _json_safe(item)
    return receipt


def _gpu_identity() -> dict[str, Any]:
    gpu_slot = int(os.environ.get("E025_GPU_SLOT", "0"))
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
        f"--id={gpu_slot}",
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {process.stderr.strip()}")
    fields = [part.strip() for part in process.stdout.strip().split(",")]
    if len(fields) != 4:
        raise RuntimeError("nvidia-smi returned an unexpected identity row")
    return {
        "gpu_name": fields[0],
        "gpu_uuid": fields[1],
        "vram_mib": int(fields[2]),
        "driver_version": fields[3],
        "physical_gpu_slot": gpu_slot,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "cuda_runtime": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
    }


def _process_snapshot() -> list[dict[str, Any]]:
    process = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,comm=,args="],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    if process.returncode != 0:
        return [{"status": "UNAVAILABLE", "error": process.stderr[-500:]}]
    rows: list[dict[str, Any]] = []
    for line in process.stdout.splitlines()[:100]:
        fields = line.strip().split(maxsplit=3)
        if len(fields) < 3:
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "command": fields[2],
                "arguments": fields[3] if len(fields) == 4 else "",
            }
        )
    return rows


def _stage_assignment(config: dict[str, Any]) -> StageAssignment:
    worker = config["worker"]
    layer_ids = tuple(int(value) for value in worker["owned_layers"])
    if len(layer_ids) != 1:
        raise ValueError("E025 physical stage worker must own exactly one layer")
    layer = layer_ids[0]
    components = set(str(value) for value in worker.get("owned_components", []))
    return StageAssignment(
        stage_id=layer,
        layer_start=layer,
        layer_end=layer + 1,
        layer_ids=(layer,),
        weight_bytes=int(worker["source_weight_bytes"]),
        estimated_compute_ns=0,
        measured_compute_ns=None,
        kv_cache_bytes_per_token=0,
        peak_temporary_bytes=0,
        activation_bytes=9 * 7168 * 4,
        device="native-cuda:0",
        owns_embeddings="embedding" in components,
        owns_final_norm="final_norm" in components,
        owns_output_projection="lm_head" in components,
    )


class ExpertCollectiveClient:
    """Fan out every retained layer-89 token to all frozen physical workers."""

    def __init__(
        self,
        endpoints: list[dict[str, Any]],
        credential: bytes,
        certificate: Path,
    ) -> None:
        if len(endpoints) != SUB_LAYER_WORKERS:
            raise ValueError("E025 parent requires all four frozen sub-layer workers")
        indices = sorted(int(row["worker_index"]) for row in endpoints)
        if indices != list(range(SUB_LAYER_WORKERS)):
            raise ValueError("E025 expert endpoints do not cover worker indices 0..3")
        self.endpoints = sorted(endpoints, key=lambda row: int(row["worker_index"]))
        self.connections = [
            AuthenticatedConnection(
                str(row["host"]),
                int(row["port"]),
                credential,
                certificate,
                timeout_seconds=float(row.get("timeout_seconds", 120.0)),
            )
            for row in self.endpoints
        ]
        self.pool = ThreadPoolExecutor(max_workers=SUB_LAYER_WORKERS)
        self.generation = 0
        self.records: list[dict[str, Any]] = []

    def _one(
        self,
        endpoint: dict[str, Any],
        connection: AuthenticatedConnection,
        selected: np.ndarray,
        latent: np.ndarray,
        generation: int,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
        worker_id = str(endpoint["worker_id"])
        before = (connection.sent_bytes, connection.received_bytes)
        request_id = f"e025-expert-{generation:08d}-{int(endpoint['worker_index']):02d}"
        frame = Frame(
            MessageType.EXECUTE_SHARD,
            request_id,
            generation,
            worker_id,
            f"layer-{SUB_LAYER_TARGET}:token-generation-{generation}",
            pack_payload(
                Action.EXECUTE_EXPERT_PARTITION,
                {
                    "layer": SUB_LAYER_TARGET,
                    "worker_index": int(endpoint["worker_index"]),
                    "worker_count": SUB_LAYER_WORKERS,
                    "generation": generation,
                    "whole_layer_fallback": False,
                },
                {"selected_expert_ids": selected, "latent_activation": latent},
            ),
        )
        response = connection.request(frame)
        if response.message_type is not MessageType.SHARD_RESULT:
            raise RuntimeError(f"expert worker {worker_id} returned {response.message_type.name}")
        action, metadata, arrays = unpack_payload(response.payload)
        if action is not Action.EXECUTE_EXPERT_PARTITION:
            raise RuntimeError("expert worker returned the wrong action")
        output = np.ascontiguousarray(arrays["expert_rows"], dtype=np.float32)
        if output.shape != (TOP_K, LATENT_SIZE) or not np.isfinite(output).all():
            raise RuntimeError("expert worker returned invalid real expert rows")
        after = (connection.sent_bytes, connection.received_bytes)
        network = {
            "request_wire_bytes": after[0] - before[0],
            "response_wire_bytes": after[1] - before[1],
        }
        return output, metadata, network

    def __call__(
        self,
        selected_expert_ids: np.ndarray,
        latent_activation: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        selected = np.ascontiguousarray(selected_expert_ids, dtype=np.int32).reshape(TOP_K)
        latent = np.ascontiguousarray(latent_activation, dtype=np.float32).reshape(LATENT_SIZE)
        self.generation += 1
        futures = [
            self.pool.submit(
                self._one,
                endpoint,
                connection,
                selected,
                latent,
                self.generation,
            )
            for endpoint, connection in zip(self.endpoints, self.connections, strict=True)
        ]
        results = [future.result() for future in futures]
        combined = np.zeros((TOP_K, LATENT_SIZE), dtype=np.float32)
        owner_counts = np.zeros(TOP_K, dtype=np.int32)
        workers: list[dict[str, Any]] = []
        for endpoint, (output, metadata, network) in zip(
            self.endpoints,
            results,
            strict=True,
        ):
            receipt = dict(metadata["execution"])
            for slot in receipt["owned_slots"]:
                parsed = int(slot)
                owner_counts[parsed] += 1
                if int(selected[parsed]) % SUB_LAYER_WORKERS != int(endpoint["worker_index"]):
                    raise RuntimeError("expert response violates frozen modulo ownership")
            combined += output
            workers.append(
                {
                    "worker_id": endpoint["worker_id"],
                    "worker_index": int(endpoint["worker_index"]),
                    **receipt,
                    **network,
                }
            )
        if not np.all(owner_counts == 1):
            raise RuntimeError(
                "E025 sub-layer collective did not execute every selected expert exactly once"
            )
        record = {
            "generation": self.generation,
            "layer": SUB_LAYER_TARGET,
            "workers_invoked": SUB_LAYER_WORKERS,
            "worker_ids": [str(row["worker_id"]) for row in self.endpoints],
            "selected_expert_ids": selected.tolist(),
            "every_selected_expert_executed_once": True,
            "owner_counts": owner_counts.tolist(),
            "native_expert_calls": sum(int(row["native_expert_calls"]) for row in workers),
            "request_wire_bytes": sum(int(row["request_wire_bytes"]) for row in workers),
            "response_wire_bytes": sum(int(row["response_wire_bytes"]) for row in workers),
            "expert_rows_fingerprint": _array_fingerprint(combined),
            "whole_layer_fallback": False,
            "workers": workers,
        }
        self.records.append(record)
        return combined, record

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        for connection in self.connections:
            connection.close()


class WorkerRuntime:
    """Own exactly one frozen stage or one mandatory expert fragment."""

    def __init__(self, config_path: Path, credential: bytes, certificate: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        if self.config.get("schema_version") != "experiment-025-worker-config-v1":
            raise ValueError("E025 worker config schema is invalid")
        self.worker_id = str(self.config["worker_id"])
        self.role = str(self.config["role"])
        self.snapshot = Path(str(self.config["snapshot_path"])).expanduser().resolve()
        self.cuda_library = Path(str(self.config["cuda_library"])).expanduser().resolve()
        self.telemetry_path = Path(str(self.config["telemetry_path"])).expanduser().resolve()
        self.ready_path = Path(str(self.config["ready_path"])).expanduser().resolve()
        self.credential = credential
        self.gpu = _gpu_identity()
        self.machine_id = str(os.environ.get("E025_MACHINE_ID", "unknown"))
        self.instance_id = str(os.environ.get("E025_INSTANCE_ID", "unknown"))
        self.process_id = os.getpid()
        self.lock = threading.Lock()
        self.network_lock = threading.Lock()
        self.network_frames = 0
        self.network_received_wire_bytes = 0
        self.network_sent_wire_bytes = 0
        self.action_counts: dict[str, int] = {}
        self.closed = False
        self.execution_enabled = True
        bootstrap_receipt_path = Path(
            str(self.config["bootstrap_receipt_path"])
        ).resolve()
        self.bootstrap_receipt = read_json(bootstrap_receipt_path)
        if self.bootstrap_receipt.get("status") != "PASS":
            raise ValueError("E025 worker bootstrap receipt is not passing")
        self.executor: PersistentKimiStageExecutor | ExpertPartitionExecutor
        self.collective: ExpertCollectiveClient | None = None
        if self.role == "SUB_LAYER_WORKER":
            self.executor = ExpertPartitionExecutor(
                self.snapshot,
                self.cuda_library,
                worker_index=int(self.config["worker_index"]),
                worker_count=int(self.config["worker_count"]),
            )
            warm_ids = np.arange(TOP_K, dtype=np.int32)
            _, warm = self.executor.execute(
                warm_ids,
                np.zeros(LATENT_SIZE, dtype=np.float32),
            )
            prepare = {"kind": "real_expert_warmup", "execution": warm}
        else:
            assignment = _stage_assignment(self.config)
            request = LoadStageRequest(
                worker_id=self.worker_id,
                request_id=f"{self.worker_id}-load",
                model_id=MODEL_ID,
                model_revision=MODEL_REVISION,
                tokenizer_revision=MODEL_REVISION,
                topology_id=str(self.config["topology_id"]),
                route_generation=1,
                stage_count=TRANSFORMER_LAYERS,
                assignment=assignment,
                adapter_id="kimi_k3_cuda",
                fast_path_id="colibri-kimi-k3-cuda",
                fast_path_mode="resident",
                fast_path_batch_bucket=1,
                fast_path_context_bucket=int(self.config["maximum_context"]),
                model_content_fingerprint=str(self.config["checkpoint_fingerprint"]),
                native_runtime_library=str(self.cuda_library),
                native_runtime_library_sha256=_sha256_file(self.cuda_library),
                device="native-cuda:0",
                dtype="float32",
                model_path=str(self.snapshot),
            )
            is_parent = self.role == "SUB_LAYER_PARENT"
            self.executor = PersistentKimiStageExecutor(
                request=request,
                checkpoint=self.snapshot,
                cuda_library=self.cuda_library,
                device=0,
                owned_expert_ids=frozenset() if is_parent else None,
            )
            if is_parent:
                self.collective = ExpertCollectiveClient(
                    list(self.config["expert_endpoints"]),
                    credential,
                    certificate,
                )
                prepare = self._prepare_parent()
            else:
                self.executor.prepare_for_ready()
                prepare = _json_safe(self.executor.prepare_warmup)
        self.ready = {
            "schema_version": SCHEMA_VERSION,
            "status": "READY",
            "timestamp": utc_now(),
            "worker_id": self.worker_id,
            "role": self.role,
            "process_id": self.process_id,
            "container_process_snapshot": _process_snapshot(),
            "machine_id": self.machine_id,
            "instance_id": self.instance_id,
            "gpu": self.gpu,
            "snapshot_path": str(self.snapshot),
            "snapshot_activation_sha256": sha256_file(self.snapshot / "activation.json"),
            "cuda_library": str(self.cuda_library),
            "cuda_library_sha256": sha256_file(self.cuda_library),
            "worker_config_sha256": sha256_file(self.config_path),
            "assignment_sha256": str(self.config["assignment_sha256"]),
            "assignment": {
                "owned_layers": self.config["worker"]["owned_layers"],
                "owned_components": self.config["worker"]["owned_components"],
                "owned_expert_count": self.config["worker"]["owned_expert_count"],
                "owned_expert_ids": self.config["worker"]["owned_expert_ids"],
                "tensor_count": self.config["worker"]["tensor_count"],
                "source_weight_bytes": self.config["worker"]["source_weight_bytes"],
                "safetensor_source_files": self.config["worker"][
                    "safetensor_source_files"
                ],
            },
            "checkpoint_fingerprint": str(self.config["checkpoint_fingerprint"]),
            "bootstrap": self.bootstrap_receipt,
            "image_digest": str(os.environ.get("E025_IMAGE_DIGEST", "unknown")),
            "prepare": prepare,
            "executor": (
                _json_safe(self.executor.ready)
                if isinstance(self.executor, ExpertPartitionExecutor)
                else {
                    "native_primitive": "PersistentKimiStageExecutor",
                    "local_routed_expert_count": len(self.executor._expert_ownership),
                }
            ),
            "whole_layer_fallback": False,
            "controller_compute_fallback": False,
        }
        atomic_write_json(self.ready_path, self.ready)
        append_jsonl(self.telemetry_path, {"event": "READY", **self.ready})

    def _prepare_parent(self) -> dict[str, Any]:
        if not isinstance(self.executor, PersistentKimiStageExecutor) or self.collective is None:
            raise RuntimeError("E025 parent preparation has no physical collective")
        session_id = "__e025_parent_prepare__"
        self.executor.open_session(session_id, maximum_context_override=3)
        try:
            boundary = torch.zeros((1, 9, 7168), dtype=torch.float32)
            for position in range(3):
                self.executor.execute_decode_with_external_experts(
                    session_id=session_id,
                    hidden_states=boundary,
                    cache_position_start=position,
                    dispatch=self.collective,
                )
        finally:
            self.executor.close_session(session_id)
        return {
            "kind": "physical_four_worker_sub_layer_warmup",
            "calls": 3,
            "sub_layer_collective_records": self.collective.records[-3:],
            "local_routed_expert_count": len(self.executor._expert_ownership),
            "whole_layer_fallback": False,
        }

    def _response(
        self,
        request: Frame,
        action: Action,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
        *,
        kind: MessageType = MessageType.SHARD_RESULT,
    ) -> Frame:
        return Frame(
            kind,
            request.request_id,
            request.chunk_id,
            self.worker_id,
            request.state_id,
            pack_payload(action, _json_safe(metadata), arrays),
        )

    def process(self, frame: Frame) -> Frame:
        if frame.worker_id != self.worker_id:
            raise ValueError("E025 frame targets a different physical worker")
        action, metadata, arrays = unpack_payload(frame.payload)
        if action is Action.REGISTER:
            return self._response(frame, action, {"ready": self.ready})
        if action is Action.HEALTH:
            health = self._health()
            return self._response(frame, action, health)
        with self.lock:
            if action is Action.OPEN_SESSION:
                if not isinstance(self.executor, PersistentKimiStageExecutor):
                    raise ValueError("expert workers do not own model sessions")
                session_id = str(metadata["session_id"])
                self.executor.open_session(
                    session_id,
                    maximum_context_override=int(metadata["maximum_context"]),
                )
                response = self._response(
                    frame,
                    action,
                    {"session_id": session_id, "opened": True},
                )
            elif action is Action.CLOSE_SESSION:
                if not isinstance(self.executor, PersistentKimiStageExecutor):
                    raise ValueError("expert workers do not own model sessions")
                session_id = str(metadata["session_id"])
                released = self.executor.close_session(session_id)
                response = self._response(
                    frame,
                    action,
                    {"session_id": session_id, "released_kv_bytes": released},
                )
            elif action is Action.EXECUTE_EXPERT_PARTITION:
                if not isinstance(self.executor, ExpertPartitionExecutor):
                    raise ValueError("stage workers cannot execute an expert partition request")
                if not self.execution_enabled:
                    raise RuntimeError(
                        "E025 physical expert fragment is disabled by the negative control"
                    )
                rows, execution = self.executor.execute(
                    arrays["selected_expert_ids"],
                    arrays["latent_activation"],
                )
                response = self._response(
                    frame,
                    action,
                    {
                        "execution": execution,
                        "native_dispatch": True,
                        "cached_output": False,
                        "synthetic_tensor": False,
                    },
                    {"expert_rows": rows},
                )
            elif action is Action.DISABLE_EXECUTION:
                if not isinstance(self.executor, ExpertPartitionExecutor):
                    raise ValueError("only a sub-layer worker can be disabled")
                self.execution_enabled = False
                response = self._response(
                    frame,
                    action,
                    {"worker_id": self.worker_id, "execution_enabled": False},
                )
            elif action is Action.ENABLE_EXECUTION:
                if not isinstance(self.executor, ExpertPartitionExecutor):
                    raise ValueError("only a sub-layer worker can be enabled")
                self.execution_enabled = True
                response = self._response(
                    frame,
                    action,
                    {"worker_id": self.worker_id, "execution_enabled": True},
                )
            elif action is Action.EXECUTE_STAGE:
                response = self._execute_stage(frame, metadata, arrays)
            else:
                raise ValueError(f"E025 worker rejects action {action.value}")
        action_counts = getattr(self, "action_counts", None)
        if action_counts is None:
            action_counts = {}
            self.action_counts = action_counts
        action_counts[action.value] = action_counts.get(action.value, 0) + 1
        append_jsonl(
            self.telemetry_path,
            {
                "event": "REQUEST_COMPLETE",
                "timestamp": utc_now(),
                "worker_id": self.worker_id,
                "request_id": frame.request_id,
                "chunk_id": frame.chunk_id,
                "state_id": frame.state_id,
                "action": action.value,
                "response_sha256": hashlib.sha256(response.payload).hexdigest(),
            },
        )
        return response

    def note_network(self, received: int, sent: int) -> None:
        with self.network_lock:
            self.network_frames += 1
            self.network_received_wire_bytes += int(received)
            self.network_sent_wire_bytes += int(sent)

    def _execute_stage(
        self,
        frame: Frame,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> Frame:
        if not isinstance(self.executor, PersistentKimiStageExecutor):
            raise ValueError("expert fragment cannot execute a complete stage")
        session_id = str(metadata["session_id"])
        position = int(metadata["position"])
        execution_before = len(self.executor.execution_records)
        if self.executor.request.assignment.owns_embeddings:
            token_ids = torch.from_numpy(
                np.ascontiguousarray(arrays["token_ids"], dtype=np.int64)
            )
            result = self.executor.execute_prefill(
                session_id=session_id,
                token_ids=token_ids,
                cache_position_start=position,
            )
        else:
            boundary = torch.from_numpy(
                np.ascontiguousarray(arrays["boundary"], dtype=np.float32)
            )
            if self.role == "SUB_LAYER_PARENT":
                if self.collective is None:
                    raise RuntimeError("E025 sub-layer parent has no physical workers")
                result = self.executor.execute_decode_with_external_experts(
                    session_id=session_id,
                    hidden_states=boundary,
                    cache_position_start=position,
                    dispatch=self.collective,
                )
            else:
                result = self.executor.execute_decode(
                    session_id=session_id,
                    hidden_states=boundary,
                    cache_position_start=position,
                )
        if len(self.executor.execution_records) != execution_before + 1:
            raise RuntimeError("E025 production stage did not emit a fresh execution receipt")
        record = _execution_receipt(self.executor.execution_records[-1])
        output_arrays: dict[str, np.ndarray] = {
            "boundary": np.ascontiguousarray(
                result.stage_boundary_hidden_states.detach().cpu().numpy(),
                dtype=np.float32,
            )
        }
        if result.final_hidden_states is not None:
            output_arrays["final_hidden"] = np.ascontiguousarray(
                result.final_hidden_states.detach().cpu().numpy(), dtype=np.float32
            )
        if result.logits is not None:
            output_arrays["logits"] = np.ascontiguousarray(
                result.logits.detach().cpu().numpy(), dtype=np.float32
            )
        if result.sampled_token_ids is not None:
            output_arrays["sampled_token_ids"] = np.ascontiguousarray(
                result.sampled_token_ids.detach().cpu().numpy(), dtype=np.int64
            )
        return self._response(
            frame,
            Action.EXECUTE_STAGE,
            {
                "execution": record,
                "cache_sequence_length": int(result.cache_sequence_length),
                "compute_ns": int(result.compute_ns),
                "native_dispatch": True,
                "native_primitive": "PersistentKimiStageExecutor",
                "cached_output": False,
                "synthetic_tensor": False,
                "controller_compute_fallback": False,
                "whole_layer_fallback": False,
            },
            output_arrays,
        )

    def _health(self) -> dict[str, Any]:
        if isinstance(self.executor, ExpertPartitionExecutor):
            runtime = self.executor.health()
        else:
            self.executor.runtime.synchronize()
            runtime = {
                "cuda_error_state_ok": self.executor.runtime.error_state_ok(),
                "memory": self.executor.runtime.mem_info(),
                "lifecycle": self.executor.lifecycle_snapshot(),
            }
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "gpu": self.gpu,
            "runtime": _json_safe(runtime),
            "action_counts": dict(sorted(self.action_counts.items())),
            "network": {
                "frames": self.network_frames,
                "received_wire_bytes": self.network_received_wire_bytes,
                "sent_wire_bytes": self.network_sent_wire_bytes,
            },
            "whole_layer_fallback": False,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.collective is not None:
            self.collective.close()
        self.executor.close()
        append_jsonl(
            self.telemetry_path,
            {"event": "CLOSED", "timestamp": utc_now(), "worker_id": self.worker_id},
        )


class _TlsServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[socketserver.BaseRequestHandler],
        context: ssl.SSLContext,
    ) -> None:
        self.ssl_context = context
        super().__init__(address, handler)

    def get_request(self) -> tuple[socket.socket, Any]:
        channel, address = super().get_request()
        return self.ssl_context.wrap_socket(channel, server_side=True), address


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, WorkerServer)
        channel = self.request
        assert isinstance(channel, socket.socket)
        while not server.stopping.is_set():
            try:
                frame, received = recv_frame(channel, server.runtime.credential)
            except (EOFError, ConnectionError, OSError):
                return
            try:
                response = server.runtime.process(frame)
            except BaseException as exc:
                append_jsonl(
                    server.runtime.telemetry_path,
                    {
                        "event": "REQUEST_ERROR",
                        "timestamp": utc_now(),
                        "worker_id": server.runtime.worker_id,
                        "request_id": frame.request_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                response = Frame(
                    MessageType.ERROR,
                    frame.request_id,
                    frame.chunk_id,
                    frame.worker_id,
                    frame.state_id,
                    pack_payload(
                        Action.HEALTH,
                        {"error_type": type(exc).__name__, "error": str(exc)},
                    ),
                )
            sent = send_frame(channel, response, server.runtime.credential)
            server.runtime.note_network(received, sent)
            append_jsonl(
                server.runtime.telemetry_path,
                {
                    "event": "NETWORK_FRAME",
                    "timestamp": utc_now(),
                    "worker_id": server.runtime.worker_id,
                    "request_id": frame.request_id,
                    "received_wire_bytes": received,
                    "sent_wire_bytes": sent,
                },
            )


class WorkerServer(_TlsServer):
    def __init__(
        self,
        address: tuple[str, int],
        runtime: WorkerRuntime,
        context: ssl.SSLContext,
    ) -> None:
        self.runtime = runtime
        self.stopping = threading.Event()
        super().__init__(address, _Handler, context)

    def server_close(self) -> None:
        self.stopping.set()
        try:
            self.runtime.close()
        finally:
            super().server_close()


def run_worker_server(
    config_path: Path,
    credential_path: Path,
    certificate: Path,
    private_key: Path,
    *,
    bind: str,
    port: int,
) -> None:
    credential = credential_path.read_bytes()
    if len(credential) < 32:
        raise ValueError("E025 credential file is invalid")
    runtime = WorkerRuntime(config_path, credential, certificate)
    context = server_ssl_context(certificate, private_key)
    server = WorkerServer((bind, port), runtime, context)
    print(
        json.dumps(
            {
                "status": "READY",
                "worker_id": runtime.worker_id,
                "role": runtime.role,
                "bind": bind,
                "port": server.server_address[1],
                "ready_path": str(runtime.ready_path),
                "credential_logged": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=42525)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    run_worker_server(
        arguments.worker_config,
        arguments.credential_file,
        arguments.certificate,
        arguments.private_key,
        bind=arguments.bind,
        port=arguments.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExpertCollectiveClient",
    "WorkerRuntime",
    "WorkerServer",
    "run_worker_server",
]
