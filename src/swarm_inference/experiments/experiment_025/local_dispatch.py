"""Local physical proof that an E025 TLS frame reaches real K3 CUDA compute."""

from __future__ import annotations

import multiprocessing
import os
import threading
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _numerical_metrics, _sha256_file
from swarm_inference.execution.kimi_k3_graph_runtime import _parse_oracle_routes
from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_020.transport import Frame, MessageType
from swarm_inference.protocol.stage_worker import LoadStageRequest

from .constants import EVIDENCE_CLASS, MODEL_ID, MODEL_REVISION, TRANSFORMER_LAYERS
from .io import atomic_write_json, canonical_sha256, read_json, sha256_file, utc_now
from .wire import (
    Action,
    AuthenticatedConnection,
    pack_payload,
    server_ssl_context,
    unpack_payload,
)
from .worker import WorkerRuntime, WorkerServer, _gpu_identity, _stage_assignment


def _real_worker_process(
    control: Any,
    *,
    checkpoint: str,
    cuda_library: str,
    placement_path: str,
    credential: bytes,
    certificate: str,
    private_key: str,
    telemetry_path: str,
) -> None:
    runtime: WorkerRuntime | None = None
    server: WorkerServer | None = None
    server_thread: threading.Thread | None = None
    try:
        placement = read_json(Path(placement_path))
        worker = next(
            row
            for row in placement["workers"]
            if row["worker_id"] == "e025-stage-001"
        )
        assignment = _stage_assignment({"worker": worker})
        library = Path(cuda_library).resolve()
        request = LoadStageRequest(
            worker_id="e025-stage-001",
            request_id="e025-local-real-native-load",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            topology_id=canonical_sha256(placement["topology"]),
            route_generation=1,
            stage_count=TRANSFORMER_LAYERS,
            assignment=assignment,
            adapter_id="kimi_k3_cuda",
            fast_path_id="colibri-kimi-k3-cuda",
            fast_path_mode="resident",
            fast_path_batch_bucket=1,
            fast_path_context_bucket=3,
            model_content_fingerprint=placement["checkpoint"][
                "checkpoint_fingerprint"
            ],
            native_runtime_library=str(library),
            native_runtime_library_sha256=_sha256_file(library),
            device="native-cuda:0",
            dtype="float32",
            model_path=str(Path(checkpoint).resolve()),
        )
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=Path(checkpoint).resolve(),
            cuda_library=library,
            device=0,
        )
        executor.prepare_for_ready()
        runtime = object.__new__(WorkerRuntime)
        runtime.worker_id = "e025-stage-001"
        runtime.role = "BACKBONE_STAGE"
        runtime.executor = executor
        runtime.collective = None
        runtime.credential = credential
        runtime.telemetry_path = Path(telemetry_path).resolve()
        runtime.lock = threading.Lock()
        runtime.network_lock = threading.Lock()
        runtime.network_frames = 0
        runtime.network_received_wire_bytes = 0
        runtime.network_sent_wire_bytes = 0
        runtime.action_counts = {}
        runtime.execution_enabled = True
        runtime.closed = False
        runtime.process_id = os.getpid()
        runtime.gpu = _gpu_identity()
        runtime.ready = {
            "status": "READY",
            "worker_id": runtime.worker_id,
            "role": runtime.role,
            "process_id": runtime.process_id,
            "gpu": runtime.gpu,
            "cuda_library_sha256": sha256_file(library),
            "checkpoint_fingerprint": placement["checkpoint"][
                "checkpoint_fingerprint"
            ],
            "whole_layer_fallback": False,
            "controller_compute_fallback": False,
        }
        context = server_ssl_context(Path(certificate), Path(private_key))
        server = WorkerServer(("127.0.0.1", 0), runtime, context)
        server_thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            name="e025-local-real-native-worker-server",
        )
        server_thread.start()
        control.send(
            {
                "status": "READY",
                "port": int(server.server_address[1]),
                "worker_process_id": os.getpid(),
                "gpu": runtime.gpu,
                "cuda_library_sha256": sha256_file(library),
            }
        )
        command = control.recv()
        if command != "STOP":
            raise ValueError("E025 local dispatch child received an invalid command")
    except BaseException as exc:
        with suppress(BaseException):
            control.send(
                {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=10.0)
        elif runtime is not None:
            runtime.close()
        control.close()


def run_local_native_dispatch(
    *,
    checkpoint: Path,
    cuda_library: Path,
    placement_path: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    credential_path: Path,
    certificate: Path,
    private_key: Path,
    output_path: Path,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    inputs, expected = _stage_fixtures(
        checkpoint.resolve(), oracle_trace.resolve(), layer=1
    )
    expected_route = _parse_oracle_routes(oracle_routes.resolve())[1][0]
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    telemetry = output_path.with_suffix(".worker.jsonl").resolve()
    process = context.Process(
        target=_real_worker_process,
        kwargs={
            "control": child,
            "checkpoint": str(checkpoint.resolve()),
            "cuda_library": str(cuda_library.resolve()),
            "placement_path": str(placement_path.resolve()),
            "credential": credential_path.read_bytes(),
            "certificate": str(certificate.resolve()),
            "private_key": str(private_key.resolve()),
            "telemetry_path": str(telemetry),
        },
        name="e025-local-real-native-worker",
    )
    process.start()
    child.close()
    ready: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    try:
        if not parent.poll(timeout_seconds):
            raise TimeoutError("E025 local real-native worker did not become READY")
        ready = dict(parent.recv())
        if ready.get("status") != "READY":
            raise RuntimeError(f"E025 local real-native worker failed: {ready}")
        sequence = 0
        with AuthenticatedConnection(
            "127.0.0.1",
            int(ready["port"]),
            credential_path.read_bytes(),
            certificate,
            timeout_seconds=180.0,
        ) as connection:

            def request(
                action: Action,
                metadata: dict[str, Any],
                arrays: dict[str, np.ndarray] | None = None,
            ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, int]]:
                nonlocal sequence
                sequence += 1
                before = (connection.sent_bytes, connection.received_bytes)
                frame = Frame(
                    MessageType.EXECUTE_SHARD,
                    f"e025-local-real-{sequence:04d}",
                    sequence,
                    "e025-stage-001",
                    "e025-local-real-session",
                    pack_payload(action, metadata, arrays),
                )
                response = connection.request(frame)
                if response.message_type is not MessageType.SHARD_RESULT:
                    raise RuntimeError(
                        f"E025 local worker returned {response.message_type.name}"
                    )
                response_action, response_metadata, response_arrays = unpack_payload(
                    response.payload
                )
                if response_action is not action:
                    raise RuntimeError("E025 local worker returned a different action")
                return response_metadata, response_arrays, {
                    "request_wire_bytes": connection.sent_bytes - before[0],
                    "response_wire_bytes": connection.received_bytes - before[1],
                }

            registration, _, registration_wire = request(Action.REGISTER, {})
            opened, _, open_wire = request(
                Action.OPEN_SESSION,
                {"session_id": "e025-local-real-session", "maximum_context": 3},
            )
            execution, arrays, execute_wire = request(
                Action.EXECUTE_STAGE,
                {
                    "session_id": "e025-local-real-session",
                    "position": 0,
                    "layer": 1,
                    "controller_compute_fallback": False,
                },
                {"boundary": np.ascontiguousarray(inputs[0], dtype=np.float32)},
            )
            health, _, health_wire = request(Action.HEALTH, {})
            closed, _, close_wire = request(
                Action.CLOSE_SESSION,
                {"session_id": "e025-local-real-session"},
            )
        actual = np.ascontiguousarray(arrays["boundary"], dtype=np.float32)
        metrics = _numerical_metrics(expected[0], actual)
        observed_route = [
            int(value) for value in execution["execution"]["selected_expert_ids"]
        ]
        rows.append(
            {
                "metrics": metrics,
                "expected_route": expected_route,
                "observed_route": observed_route,
                "route_exact": observed_route == expected_route,
                "execution": execution,
                "health": health,
                "transport": {
                    "registration": registration_wire,
                    "open": open_wire,
                    "execute": execute_wire,
                    "health": health_wire,
                    "close": close_wire,
                },
                "registered_process_id": registration["ready"]["process_id"],
                "session_opened": opened.get("opened") is True,
                "released_kv_bytes": int(closed.get("released_kv_bytes", 0)),
            }
        )
    except BaseException as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if process.is_alive():
            with suppress(BaseException):
                parent.send("STOP")
        process.join(timeout=60.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=15.0)
        parent.close()
    row = rows[0] if rows else {}
    metrics = row.get("metrics", {})
    gates = {
        "independent_worker_process": bool(ready)
        and int(ready.get("worker_process_id", -1)) != os.getpid(),
        "physical_consumer_cuda_gpu": "GEFORCE RTX 5090"
        in str(ready.get("gpu", {}).get("gpu_name", "")).upper(),
        "authenticated_tls_execute_shard_round_trip": bool(rows),
        "production_worker_runtime_dispatch": row.get("execution", {}).get(
            "native_dispatch"
        )
        is True,
        "real_native_k3_output_numerically_correct": bool(metrics)
        and float(metrics.get("relative_l2_error", 1.0)) <= 1e-4
        and float(metrics.get("maximum_absolute_error", 1.0)) <= 1e-3,
        "route_exact": row.get("route_exact") is True,
        "fresh_non_synthetic_result": row.get("execution", {}).get("cached_output")
        is False
        and row.get("execution", {}).get("synthetic_tensor") is False,
        "no_controller_or_whole_layer_fallback": row.get("execution", {}).get(
            "controller_compute_fallback"
        )
        is False
        and row.get("execution", {}).get("whole_layer_fallback") is False,
        "post_execution_cuda_health": row.get("health", {})
        .get("runtime", {})
        .get("cuda_error_state_ok")
        is True,
        "worker_process_exited": process.exitcode == 0,
    }
    payload = {
        "schema_version": "experiment-025-local-real-native-dispatch-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "evidence_class": "PHYSICAL_SINGLE_MACHINE_LOCAL_CONSUMER_GPU_DISPATCH",
        "headline_physical_swarm_claimed": False,
        "headline_evidence_class_target": EVIDENCE_CLASS,
        "ready": ready,
        "rows": rows,
        "worker_telemetry": str(telemetry),
        "worker_telemetry_sha256": sha256_file(telemetry) if telemetry.is_file() else None,
        "failure": failure,
        "worker_exitcode": process.exitcode,
        "gates": gates,
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = ["run_local_native_dispatch"]
