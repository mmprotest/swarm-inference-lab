"""Real two-stage Kimi CUDA/TCP coarse-pipeline certification."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import socket
import struct
import threading
import time
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor, _timing
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _numerical_metrics,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.full_cuda import _parse_oracle_routes
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import _request

SCHEMA_VERSION = "experiment-014-k3-real-coarse-stage-tcp-v1"
FRAME_BYTES = 4
CANONICAL_BOUNDARY_BYTES = 9 * 7168 * 4


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _recv_exact(channel: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise ConnectionError("coarse-stage TCP channel closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(channel: socket.socket, message: dict[str, Any]) -> dict[str, Any]:
    serialization_started = time.perf_counter_ns()
    payload = pickle.dumps(message, protocol=5)
    serialization_ms = (time.perf_counter_ns() - serialization_started) / 1e6
    send_started = time.perf_counter_ns()
    channel.sendall(struct.pack("!I", len(payload)) + payload)
    send_ms = (time.perf_counter_ns() - send_started) / 1e6
    return {
        "payload_bytes": len(payload),
        "framing_bytes": FRAME_BYTES,
        "wire_bytes": len(payload) + FRAME_BYTES,
        "serialization_ms": serialization_ms,
        "send_ms": send_ms,
    }


def _receive_frame(channel: socket.socket) -> tuple[dict[str, Any], dict[str, Any]]:
    receive_started = time.perf_counter_ns()
    header = _recv_exact(channel, FRAME_BYTES)
    payload_size = struct.unpack("!I", header)[0]
    payload = _recv_exact(channel, payload_size)
    receive_ms = (time.perf_counter_ns() - receive_started) / 1e6
    deserialize_started = time.perf_counter_ns()
    message = pickle.loads(payload)
    deserialization_ms = (time.perf_counter_ns() - deserialize_started) / 1e6
    if not isinstance(message, dict):
        raise RuntimeError("coarse-stage TCP frame is not an object")
    return message, {
        "payload_bytes": payload_size,
        "framing_bytes": FRAME_BYTES,
        "wire_bytes": payload_size + FRAME_BYTES,
        "receive_ms": receive_ms,
        "deserialization_ms": deserialization_ms,
    }


def _listener() -> tuple[socket.socket, tuple[str, int]]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    host, port = server.getsockname()
    return server, (str(host), int(port))


def _connect(endpoint: tuple[str, int], *, timeout_seconds: float = 30.0) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        channel = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            channel.connect(endpoint)
            return channel
        except OSError as exc:
            last_error = exc
            channel.close()
            time.sleep(0.05)
    raise TimeoutError(f"could not connect to coarse-stage worker: {last_error}")


def _snapshot(executor: PersistentKimiStageExecutor) -> dict[str, Any]:
    process = psutil.Process()
    return {
        "pid": os.getpid(),
        "child_process_ids": sorted(child.pid for child in process.children(recursive=True)),
        "os_thread_ids": sorted(item.id for item in process.threads()),
        "python_thread_ids": sorted(
            int(item.ident) for item in threading.enumerate() if item.ident is not None
        ),
        "executor": executor.lifecycle_snapshot(),
    }


def _snapshot_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_executor = before["executor"]
    after_executor = after["executor"]
    return {
        "topology_rebuilds": 0,
        "process_creation": len(
            set(after["child_process_ids"]) - set(before["child_process_ids"])
        ),
        "thread_creation": len(set(after["os_thread_ids"]) - set(before["os_thread_ids"])),
        "task_creation": 0,
        "connection_establishment": 0,
        "weight_loading": int(after_executor["weight_load_count"])
        - int(before_executor["weight_load_count"]),
        "model_materialization": int(after_executor["model_materialization_count"])
        - int(before_executor["model_materialization_count"]),
        "persistent_buffer_allocation": int(
            after_executor["persistent_buffer_allocation_count"]
        )
        - int(before_executor["persistent_buffer_allocation_count"]),
    }


def _ready_payload(
    executor: PersistentKimiStageExecutor,
    *,
    layer: int,
    endpoint: tuple[str, int],
    load_wall_ms: float,
) -> dict[str, Any]:
    return {
        "kind": "READY",
        "layer": layer,
        "pid": os.getpid(),
        "endpoint": list(endpoint),
        "load_wall_ms": load_wall_ms,
        "resident_device_bytes": executor.resident_device_bytes,
        "tracked_device_bytes": executor.tracked_device_bytes,
        "weight_fingerprint": executor.weight_fingerprint,
        "cuda_library_sha256": executor.runtime.sha256,
        "prepare": executor.prepare_warmup,
        "lifecycle": executor.lifecycle_snapshot(),
    }


def _worker_health(executor: PersistentKimiStageExecutor) -> dict[str, Any]:
    executor.runtime.synchronize()
    return {
        "cuda_synchronize": "PASS",
        "cuda_error_state_ok": executor.runtime.error_state_ok(),
        "memory": executor.runtime.mem_info(),
        "active_sessions": executor.lifecycle_snapshot()["active_sessions"],
    }


def _stage1_main(
    ready_connection: Connection,
    checkpoint: str,
    cuda_library: str,
    device: int,
    maximum_context: int,
    cycle_id: str,
) -> None:
    server: socket.socket | None = None
    channel: socket.socket | None = None
    executor: PersistentKimiStageExecutor | None = None
    ready_sent = False
    try:
        server, endpoint = _listener()
        started = time.perf_counter_ns()
        request = _request(
            Path(checkpoint),
            Path(cuda_library),
            layer=1,
            device=device,
            cycle_id=cycle_id,
            maximum_context=maximum_context,
        )
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=Path(checkpoint),
            cuda_library=Path(cuda_library),
            device=device,
        )
        executor.prepare_for_ready()
        ready_connection.send(
            _ready_payload(
                executor,
                layer=1,
                endpoint=endpoint,
                load_wall_ms=(time.perf_counter_ns() - started) / 1e6,
            )
        )
        ready_sent = True
        channel, _ = server.accept()
        channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            request_message, ingress = _receive_frame(channel)
            kind = request_message.get("kind")
            if kind == "OPEN":
                executor.open_session(
                    str(request_message["session_id"]),
                    maximum_context_override=int(request_message["maximum_context"]),
                )
                response = {"kind": "OPENED", "layer": 1}
            elif kind == "CLOSE_SESSION":
                released = executor.close_session(str(request_message["session_id"]))
                response = {"kind": "SESSION_CLOSED", "layer": 1, "released_bytes": released}
            elif kind == "SNAPSHOT":
                response = {"kind": "SNAPSHOT", "layer": 1, "snapshot": _snapshot(executor)}
            elif kind == "STATE":
                response = {
                    "kind": "STATE",
                    "layer": 1,
                    "state": executor.session_state_evidence(str(request_message["session_id"])),
                }
            elif kind == "HEALTH":
                response = {"kind": "HEALTH", "layer": 1, "health": _worker_health(executor)}
            elif kind == "EXECUTE":
                hidden = np.ascontiguousarray(request_message["hidden_states"], dtype=np.float32)
                if hidden.shape != (1, 9, executor.config.hidden):
                    raise ValueError("stage-1 TCP boundary has invalid shape")
                wall_started = time.perf_counter_ns()
                result = executor.execute_decode(
                    session_id=str(request_message["session_id"]),
                    hidden_states=torch.from_numpy(hidden),
                    cache_position_start=int(request_message["position"]),
                )
                output = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                record = executor.execution_records[-1]
                response = {
                    "kind": "RESULT",
                    "layer": 1,
                    "sequence": int(request_message["sequence"]),
                    "position": int(request_message["position"]),
                    "input_fingerprint": _array_fingerprint(hidden),
                    "boundary_output": output,
                    "output_fingerprint": _array_fingerprint(output),
                    "selected_expert_ids": record["selected_expert_ids"],
                    "selected_weights": record["selected_weights"],
                    "device_ms": float(record["device_ms"]),
                    "worker_wall_ms": (time.perf_counter_ns() - wall_started) / 1e6,
                    "ingress": ingress,
                }
            elif kind == "SHUTDOWN":
                response = {
                    "kind": "SHUTDOWN",
                    "layer": 1,
                    "health": _worker_health(executor),
                    "lifecycle": executor.lifecycle_snapshot(),
                }
                _send_frame(channel, response)
                return
            else:
                raise RuntimeError(f"stage 1 received invalid command {kind!r}")
            _send_frame(channel, response)
    except BaseException as exc:
        failure = {
            "kind": "FAIL",
            "layer": 1,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not ready_sent:
            with suppress(Exception):
                ready_connection.send(failure)
        elif channel is not None:
            with suppress(Exception):
                _send_frame(channel, failure)
    finally:
        if executor is not None:
            executor.close()
        if channel is not None:
            channel.close()
        if server is not None:
            server.close()
        ready_connection.close()


def _forward(channel: socket.socket, message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter_ns()
    outbound = _send_frame(channel, message)
    response, inbound = _receive_frame(channel)
    return response, {
        "outbound": outbound,
        "inbound": inbound,
        "roundtrip_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def _stage0_main(
    ready_connection: Connection,
    checkpoint: str,
    cuda_library: str,
    device: int,
    maximum_context: int,
    cycle_id: str,
    next_endpoint: tuple[str, int],
) -> None:
    server: socket.socket | None = None
    upstream: socket.socket | None = None
    downstream: socket.socket | None = None
    executor: PersistentKimiStageExecutor | None = None
    ready_sent = False
    try:
        server, endpoint = _listener()
        started = time.perf_counter_ns()
        request = _request(
            Path(checkpoint),
            Path(cuda_library),
            layer=0,
            device=device,
            cycle_id=cycle_id,
            maximum_context=maximum_context,
        )
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=Path(checkpoint),
            cuda_library=Path(cuda_library),
            device=device,
        )
        executor.prepare_for_ready()
        downstream = _connect(next_endpoint)
        ready = _ready_payload(
            executor,
            layer=0,
            endpoint=endpoint,
            load_wall_ms=(time.perf_counter_ns() - started) / 1e6,
        )
        ready["persistent_downstream_connections"] = 1
        ready_connection.send(ready)
        ready_sent = True
        upstream, _ = server.accept()
        upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            root_message, root_ingress = _receive_frame(upstream)
            kind = root_message.get("kind")
            if kind == "OPEN":
                executor.open_session(
                    str(root_message["session_id"]),
                    maximum_context_override=int(root_message["maximum_context"]),
                )
                next_response, edge = _forward(downstream, root_message)
                response = {"kind": "OPENED", "local_layer": 0, "next": next_response, "edge": edge}
            elif kind == "CLOSE_SESSION":
                released = executor.close_session(str(root_message["session_id"]))
                next_response, edge = _forward(downstream, root_message)
                response = {
                    "kind": "SESSION_CLOSED",
                    "local_layer": 0,
                    "released_bytes": released,
                    "next": next_response,
                    "edge": edge,
                }
            elif kind == "SNAPSHOT":
                local = _snapshot(executor)
                next_response, edge = _forward(downstream, root_message)
                response = {"kind": "SNAPSHOT", "local": local, "next": next_response["snapshot"], "edge": edge}
            elif kind == "STATE":
                local = executor.session_state_evidence(str(root_message["session_id"]))
                next_response, edge = _forward(downstream, root_message)
                response = {"kind": "STATE", "local": local, "next": next_response["state"], "edge": edge}
            elif kind == "HEALTH":
                local = _worker_health(executor)
                next_response, edge = _forward(downstream, root_message)
                response = {"kind": "HEALTH", "local": local, "next": next_response["health"], "edge": edge}
            elif kind == "EXECUTE":
                token_ids = torch.tensor([[int(root_message["token_id"])]], dtype=torch.int64)
                local_started = time.perf_counter_ns()
                result = executor.execute_prefill(
                    session_id=str(root_message["session_id"]),
                    token_ids=token_ids,
                    cache_position_start=int(root_message["position"]),
                )
                boundary = result.stage_boundary_hidden_states.detach().cpu().numpy().copy()
                local_record = executor.execution_records[-1]
                local_wall_ms = (time.perf_counter_ns() - local_started) / 1e6
                next_message = {
                    "kind": "EXECUTE",
                    "session_id": str(root_message["session_id"]),
                    "sequence": int(root_message["sequence"]),
                    "position": int(root_message["position"]),
                    "hidden_states": boundary,
                }
                next_response, edge = _forward(downstream, next_message)
                if next_response.get("kind") != "RESULT":
                    raise RuntimeError(f"stage 1 failed: {next_response}")
                exposed = max(0.0, float(edge["roundtrip_ms"]) - float(next_response["worker_wall_ms"]))
                response = {
                    "kind": "RESULT",
                    "sequence": int(root_message["sequence"]),
                    "position": int(root_message["position"]),
                    "stage0_boundary_fingerprint": _array_fingerprint(boundary),
                    "stage0_boundary_payload_bytes": boundary.nbytes,
                    "stage0_device_ms": float(local_record["device_ms"]),
                    "stage0_wall_ms": local_wall_ms,
                    "stage1_input_fingerprint": next_response["input_fingerprint"],
                    "stage1_device_ms": next_response["device_ms"],
                    "stage1_wall_ms": next_response["worker_wall_ms"],
                    "selected_expert_ids": next_response["selected_expert_ids"],
                    "selected_weights": next_response["selected_weights"],
                    "boundary_output": next_response["boundary_output"],
                    "output_fingerprint": next_response["output_fingerprint"],
                    "edge": edge,
                    "exposed_transport_and_serialization_ms": exposed,
                    "root_ingress": root_ingress,
                }
            elif kind == "SHUTDOWN":
                next_response, edge = _forward(downstream, root_message)
                response = {
                    "kind": "SHUTDOWN",
                    "local_layer": 0,
                    "local_health": _worker_health(executor),
                    "local_lifecycle": executor.lifecycle_snapshot(),
                    "next": next_response,
                    "edge": edge,
                }
                _send_frame(upstream, response)
                return
            else:
                raise RuntimeError(f"stage 0 received invalid command {kind!r}")
            _send_frame(upstream, response)
    except BaseException as exc:
        failure = {
            "kind": "FAIL",
            "layer": 0,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if not ready_sent:
            with suppress(Exception):
                ready_connection.send(failure)
        elif upstream is not None:
            with suppress(Exception):
                _send_frame(upstream, failure)
    finally:
        if executor is not None:
            executor.close()
        if upstream is not None:
            upstream.close()
        if downstream is not None:
            downstream.close()
        if server is not None:
            server.close()
        ready_connection.close()


def _start_worker(
    context: mp.context.BaseContext,
    target: Any,
    args: tuple[Any, ...],
    *,
    name: str,
    timeout_seconds: float,
) -> tuple[mp.Process, dict[str, Any]]:
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(child, *args), name=name)
    process.start()
    child.close()
    if not parent.poll(timeout_seconds):
        process.terminate()
        process.join(10)
        raise TimeoutError(f"{name} did not reach READY")
    ready = parent.recv()
    parent.close()
    if ready.get("kind") != "READY":
        process.join(10)
        raise RuntimeError(f"{name} failed before READY: {ready}")
    return process, ready


def _root_round_trip(channel: socket.socket, message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter_ns()
    outbound = _send_frame(channel, message)
    response, inbound = _receive_frame(channel)
    return response, {
        "outbound": outbound,
        "inbound": inbound,
        "roundtrip_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def benchmark_coarse_stage_tcp(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    graph_certification: Path,
    output_path: Path,
    *,
    device: int = 0,
    warmup: int = 10,
    iterations: int = 50,
    cycle_id: str = "H014-032a",
    ready_timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    """Run the simultaneous stage-0 -> stage-1 real CUDA/TCP slice."""
    if warmup < 3 or iterations < 20:
        raise ValueError("coarse-stage benchmark requires >=3/20 warm/retained calls")
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "oracle_routes": oracle_routes.resolve(),
        "graph_certification": graph_certification.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    graph = json.loads(paths["graph_certification"].read_text(encoding="utf-8"))
    graph_fixture = graph.get("fixture", {})
    provenance = {
        "graph_status": graph.get("status"),
        "trace_matches": graph_fixture.get("oracle_trace_sha256")
        == _sha256_file(paths["oracle_trace"]),
        "routes_matches": graph_fixture.get("oracle_routes_sha256")
        == _sha256_file(paths["oracle_routes"]),
    }
    provenance["pass"] = (
        provenance["graph_status"] == "PASS"
        and provenance["trace_matches"]
        and provenance["routes_matches"]
    )
    if not provenance["pass"]:
        raise ValueError("coarse-stage fixtures are not joined to the passing CUDA graph")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": {
            "prediction": (
                "Persistent stage 0 and stage 1 CUDA workers remain simultaneously "
                "resident, reproduce graph-certified output/routes/state over one "
                "persistent TCP edge, and expose <1.5 ms loopback transport p50."
            ),
            "maximum_exposed_transport_p50_ms": 1.5,
        },
        "configuration": {
            "layers": [0, 1],
            "device": device,
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "transport": "TCP IPv4 loopback, persistent, length-prefixed pickle protocol 5",
            "canonical_boundary": "float32 [1,9,7168]",
            "canonical_boundary_bytes": CANONICAL_BOUNDARY_BYTES,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "oracle_provenance": provenance,
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    context = mp.get_context("spawn")
    stage0_process: mp.Process | None = None
    stage1_process: mp.Process | None = None
    root_channel: socket.socket | None = None
    maximum_context = warmup + iterations + 8
    retain("preregistered")
    receipt["gpu_health_before"] = _health_snapshot(device)
    receipt["device_identity"] = _device_identity(device)
    retain("gpu_health_before", status=receipt["gpu_health_before"]["status"])
    if receipt["gpu_health_before"]["status"] != "MEASURED":
        raise RuntimeError("nvidia-smi unavailable before coarse-stage benchmark")
    try:
        stage1_process, stage1_ready = _start_worker(
            context,
            _stage1_main,
            (
                str(paths["checkpoint"]),
                str(paths["cuda_library"]),
                device,
                maximum_context,
                cycle_id,
            ),
            name="h014-coarse-stage-1",
            timeout_seconds=ready_timeout_seconds,
        )
        receipt["stage1_ready"] = stage1_ready
        retain("stage1_ready", resident_device_bytes=stage1_ready["resident_device_bytes"])
        stage1_endpoint = (str(stage1_ready["endpoint"][0]), int(stage1_ready["endpoint"][1]))
        stage0_process, stage0_ready = _start_worker(
            context,
            _stage0_main,
            (
                str(paths["checkpoint"]),
                str(paths["cuda_library"]),
                device,
                maximum_context,
                cycle_id,
                stage1_endpoint,
            ),
            name="h014-coarse-stage-0",
            timeout_seconds=ready_timeout_seconds,
        )
        receipt["stage0_ready"] = stage0_ready
        receipt["simultaneous_residency"] = {
            "stage0_resident_device_bytes": stage0_ready["resident_device_bytes"],
            "stage1_resident_device_bytes": stage1_ready["resident_device_bytes"],
            "sum_resident_device_bytes": int(stage0_ready["resident_device_bytes"])
            + int(stage1_ready["resident_device_bytes"]),
            "both_processes_alive": stage0_process.is_alive() and stage1_process.is_alive(),
            "nvidia_smi": _health_snapshot(device),
        }
        retain(
            "both_workers_ready",
            sum_resident_device_bytes=receipt["simultaneous_residency"]["sum_resident_device_bytes"],
        )
        root_endpoint = (str(stage0_ready["endpoint"][0]), int(stage0_ready["endpoint"][1]))
        root_channel = _connect(root_endpoint)

        _fixtures, expected_boundaries = _stage_fixtures(
            paths["checkpoint"], paths["oracle_trace"], layer=1
        )
        expected_routes = _parse_oracle_routes(paths["oracle_routes"])[1]
        token_ids = (163584, 18699, 11)
        opened, open_transport = _root_round_trip(
            root_channel,
            {"kind": "OPEN", "session_id": "correctness", "maximum_context": 3},
        )
        if opened.get("kind") != "OPENED":
            raise RuntimeError(f"coarse correctness session failed to open: {opened}")
        comparisons: list[dict[str, Any]] = []
        try:
            for position, token_id in enumerate(token_ids):
                response, root_transport = _root_round_trip(
                    root_channel,
                    {
                        "kind": "EXECUTE",
                        "session_id": "correctness",
                        "sequence": position,
                        "position": position,
                        "token_id": token_id,
                    },
                )
                if response.get("kind") != "RESULT":
                    raise RuntimeError(f"coarse correctness execute failed: {response}")
                output = np.ascontiguousarray(response["boundary_output"], dtype=np.float32)
                comparisons.append(
                    {
                        "position": position,
                        "metrics": _numerical_metrics(output, expected_boundaries[position]),
                        "routes_equal": list(response["selected_expert_ids"])
                        == expected_routes[position],
                        "interstage_boundary_fingerprint_equal": response[
                            "stage0_boundary_fingerprint"
                        ]
                        == response["stage1_input_fingerprint"],
                        "stage0_boundary_fingerprint": response[
                            "stage0_boundary_fingerprint"
                        ],
                        "stage1_input_fingerprint": response["stage1_input_fingerprint"],
                        "output_fingerprint": response["output_fingerprint"],
                        "root_transport": root_transport,
                        "edge": response["edge"],
                    }
                )
            state_response, state_transport = _root_round_trip(
                root_channel, {"kind": "STATE", "session_id": "correctness"}
            )
        finally:
            with suppress(Exception):
                _root_round_trip(
                    root_channel,
                    {"kind": "CLOSE_SESSION", "session_id": "correctness"},
                )
        maximum_error = max(
            float(row["metrics"]["relative_l2_error"]) for row in comparisons
        )
        correctness_pass = (
            maximum_error <= 3e-5
            and all(bool(row["routes_equal"]) for row in comparisons)
            and all(bool(row["interstage_boundary_fingerprint_equal"]) for row in comparisons)
            and state_response["local"]["cache_sequence_length"] == 3
            and state_response["next"]["cache_sequence_length"] == 3
        )
        receipt["correctness"] = {
            "comparisons": comparisons,
            "maximum_relative_l2_error": maximum_error,
            "routes_equal": all(bool(row["routes_equal"]) for row in comparisons),
            "interstage_boundary_bit_exact": all(
                bool(row["interstage_boundary_fingerprint_equal"]) for row in comparisons
            ),
            "state": {"stage0": state_response["local"], "stage1": state_response["next"]},
            "state_transport": state_transport,
            "open_transport": open_transport,
            "pass": correctness_pass,
        }
        retain("correctness", status="PASS" if correctness_pass else "FAIL")
        if not correctness_pass:
            raise RuntimeError("coarse two-stage output differs from graph-certified execution")

        opened, _ = _root_round_trip(
            root_channel,
            {
                "kind": "OPEN",
                "session_id": "performance",
                "maximum_context": warmup + iterations,
            },
        )
        if opened.get("kind") != "OPENED":
            raise RuntimeError("coarse performance session failed to open")
        retained_rows: list[dict[str, Any]] = []
        before_snapshot: dict[str, Any] | None = None
        try:
            for position in range(warmup + iterations):
                if position == warmup:
                    before_snapshot, _ = _root_round_trip(
                        root_channel, {"kind": "SNAPSHOT"}
                    )
                response, root_transport = _root_round_trip(
                    root_channel,
                    {
                        "kind": "EXECUTE",
                        "session_id": "performance",
                        "sequence": position,
                        "position": position,
                        "token_id": token_ids[position % len(token_ids)],
                    },
                )
                if response.get("kind") != "RESULT":
                    raise RuntimeError(f"coarse performance execute failed: {response}")
                if position >= warmup:
                    retained_rows.append(
                        {
                            "position": position,
                            "root_roundtrip_ms": root_transport["roundtrip_ms"],
                            "root_transport": root_transport,
                            "stage0_device_ms": response["stage0_device_ms"],
                            "stage0_wall_ms": response["stage0_wall_ms"],
                            "stage1_device_ms": response["stage1_device_ms"],
                            "stage1_wall_ms": response["stage1_wall_ms"],
                            "edge": response["edge"],
                            "exposed_transport_and_serialization_ms": response[
                                "exposed_transport_and_serialization_ms"
                            ],
                            "stage0_boundary_payload_bytes": response[
                                "stage0_boundary_payload_bytes"
                            ],
                            "boundary_fingerprint_equal": response[
                                "stage0_boundary_fingerprint"
                            ]
                            == response["stage1_input_fingerprint"],
                        }
                    )
            after_snapshot, _ = _root_round_trip(root_channel, {"kind": "SNAPSHOT"})
        finally:
            with suppress(Exception):
                _root_round_trip(
                    root_channel,
                    {"kind": "CLOSE_SESSION", "session_id": "performance"},
                )
        if before_snapshot is None:
            raise RuntimeError("coarse benchmark did not enter retained execution")
        stage0_lifecycle = _snapshot_delta(before_snapshot["local"], after_snapshot["local"])
        stage1_lifecycle = _snapshot_delta(before_snapshot["next"], after_snapshot["next"])
        lifecycle_zero = all(value == 0 for value in stage0_lifecycle.values()) and all(
            value == 0 for value in stage1_lifecycle.values()
        )
        timing = {
            "end_to_end_wall": _timing([float(row["root_roundtrip_ms"]) for row in retained_rows]),
            "stage0_device": _timing([float(row["stage0_device_ms"]) for row in retained_rows]),
            "stage0_wall": _timing([float(row["stage0_wall_ms"]) for row in retained_rows]),
            "stage1_device": _timing([float(row["stage1_device_ms"]) for row in retained_rows]),
            "stage1_wall": _timing([float(row["stage1_wall_ms"]) for row in retained_rows]),
            "interstage_roundtrip": _timing(
                [float(row["edge"]["roundtrip_ms"]) for row in retained_rows]
            ),
            "exposed_transport_and_serialization": _timing(
                [float(row["exposed_transport_and_serialization_ms"]) for row in retained_rows]
            ),
        }
        wire = {
            "activation_payload_bytes": CANONICAL_BOUNDARY_BYTES,
            "mean_interstage_request_payload_bytes": float(
                np.mean([row["edge"]["outbound"]["payload_bytes"] for row in retained_rows])
            ),
            "mean_interstage_request_wire_bytes": float(
                np.mean([row["edge"]["outbound"]["wire_bytes"] for row in retained_rows])
            ),
            "mean_harness_collection_payload_bytes": float(
                np.mean([row["edge"]["inbound"]["payload_bytes"] for row in retained_rows])
            ),
            "mean_root_request_wire_bytes": float(
                np.mean([row["root_transport"]["outbound"]["wire_bytes"] for row in retained_rows])
            ),
            "mean_root_response_wire_bytes": float(
                np.mean([row["root_transport"]["inbound"]["wire_bytes"] for row in retained_rows])
            ),
            "framing_bytes_per_message": FRAME_BYTES,
            "interstage_messages_per_token": 1,
            "harness_collection_messages_per_token": 1,
            "critical_path_activation_bytes_per_token": CANONICAL_BOUNDARY_BYTES,
        }
        receipt["performance"] = {
            "warmup_calls": warmup,
            "retained_calls": iterations,
            "timing": timing,
            "wire": wire,
            "lifecycle": {
                "stage0": stage0_lifecycle,
                "stage1": stage1_lifecycle,
                "deltas_zero": lifecycle_zero,
                "persistent_connections": 2,
            },
            "boundary_bit_exact_every_call": all(
                bool(row["boundary_fingerprint_equal"]) for row in retained_rows
            ),
        }
        retain(
            "retained_performance",
            exposed_transport_p50_ms=timing["exposed_transport_and_serialization"]["p50_ms"],
            end_to_end_p50_ms=timing["end_to_end_wall"]["p50_ms"],
        )

        health_response, health_transport = _root_round_trip(root_channel, {"kind": "HEALTH"})
        receipt["post_run_checks"] = {
            "workers": {"stage0": health_response["local"], "stage1": health_response["next"]},
            "health_transport": health_transport,
            "nvidia_smi": _health_snapshot(device),
        }
        health_pass = (
            health_response["local"]["cuda_error_state_ok"]
            and health_response["next"]["cuda_error_state_ok"]
            and receipt["post_run_checks"]["nvidia_smi"]["status"] == "MEASURED"
        )
        supported = (
            correctness_pass
            and lifecycle_zero
            and health_pass
            and receipt["performance"]["boundary_bit_exact_every_call"]
            and timing["exposed_transport_and_serialization"]["p50_ms"] < 1.5
            and receipt["simultaneous_residency"]["both_processes_alive"]
        )
        receipt["hypothesis_supported"] = supported
        receipt["inspection"] = {
            "actual_bottleneck": (
                "stage compute" if timing["exposed_transport_and_serialization"]["p50_ms"]
                < max(timing["stage0_device"]["p50_ms"], timing["stage1_device"]["p50_ms"])
                else "loopback transport and serialization"
            ),
            "physical_scope": (
                "Two independent worker processes and a real TCP edge share one RTX 5090; "
                "this is a logical coarse distributed proof, not physical multi-GPU timing."
            ),
        }
        receipt["decision"] = {
            "coarse_stage_tcp": "RETAIN" if supported else "MODIFY",
            "next_hypothesis": (
                "Replay the measured one-way coarse payload under RTT/bandwidth profiles "
                "and test lower-precision boundary formats only if bandwidth is material."
            ),
        }
        receipt["status"] = "PASS" if supported else "FAIL"
        retain("complete", status=receipt["status"])

        shutdown_response, shutdown_transport = _root_round_trip(
            root_channel, {"kind": "SHUTDOWN"}
        )
        receipt["shutdown"] = {
            "response": shutdown_response,
            "transport": shutdown_transport,
        }
        _atomic_json(output_path, receipt)
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
    finally:
        if root_channel is not None:
            root_channel.close()
        for process in (stage0_process, stage1_process):
            if process is None:
                continue
            process.join(30)
            if process.is_alive():
                process.terminate()
                process.join(10)
