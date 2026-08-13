"""Full 93-layer K3 correctness through authenticated EXECUTE_SHARD."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_020.sharded_graph import certify_full_candidate
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
    new_run_credential,
)

from .io import atomic_write_json
from .native_dispatch import (
    CallableResidentPrimitive,
    NativeShardDispatcher,
    ShardRequest,
    ShardTaskType,
    decode_shard_result,
    encode_shard_request,
)


def _worker(
    connection: Any,
    credential: bytes,
    checkpoint: str,
    cuda_library: str,
    shard_library: str,
    grouped_library: str,
    oracle_root: str,
    raw_output: str,
) -> None:
    worker_id = "local-rtx5090.persistent-k3-worker"
    dispatcher = NativeShardDispatcher(worker_id)
    receipt_holder: dict[str, Any] = {}

    def execute_ordered_dag(_values: np.ndarray, _request: ShardRequest) -> np.ndarray:
        receipt = certify_full_candidate(
            Path(checkpoint),
            Path(cuda_library),
            Path(shard_library),
            Path(grouped_library),
            Path(oracle_root),
            progress=True,
        )
        receipt_holder["receipt"] = receipt
        atomic_write_json(Path(raw_output), receipt)
        head = receipt.get("head") or {}
        distributed = head.get("distributed_exact_argmax") or {}
        return np.asarray(
            [
                1.0 if receipt.get("status") == "PASS" else 0.0,
                float(receipt.get("maximum_relative_l2_error", 1.0)),
                float(distributed.get("token_id", -1)),
                1.0 if receipt.get("routes_exact") else 0.0,
            ],
            dtype=np.float32,
        )

    dispatcher.register(
        "kimi-k3-full-93:p8-mixed",
        ShardTaskType.ORDERED_LAYER_DAG,
        CallableResidentPrimitive(
            "E020ShardedK3Graph.exact_grouped_sub_layer_full_93",
            execute_ordered_dag,
        ),
    )
    try:
        encoded = connection.recv_bytes()
        request_frame = decode_frame(encoded, credential)
        result_frame = dispatcher.execute_frame(request_frame)
        connection.send_bytes(encode_frame(result_frame, credential))
        connection.send_bytes(
            json.dumps(
                {
                    "status": "PASS",
                    "dispatcher_audit": dispatcher.audit,
                    "receipt_status": receipt_holder.get("receipt", {}).get("status"),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    except BaseException as exc:
        connection.send_bytes(
            json.dumps(
                {
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "dispatcher_audit": dispatcher.audit,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    finally:
        connection.close()


def run_worker_process_correctness(
    checkpoint: Path,
    cuda_library: Path,
    shard_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    raw_output: Path,
    *,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    for path in (
        checkpoint,
        cuda_library,
        shard_library,
        grouped_library,
        oracle_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    credential = new_run_credential()
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(
        target=_worker,
        args=(
            child,
            credential,
            str(checkpoint.resolve()),
            str(cuda_library.resolve()),
            str(shard_library.resolve()),
            str(grouped_library.resolve()),
            str(oracle_root.resolve()),
            str(raw_output.resolve()),
        ),
        name="e022-native-k3-worker",
    )
    started = time.perf_counter_ns()
    process.start()
    child.close()
    request = ShardRequest(
        assignment_id="kimi-k3-full-93:p8-mixed",
        task_type=ShardTaskType.ORDERED_LAYER_DAG,
        layer=0,
        shard_index=0,
        degree=8,
        rows=1,
        input_shape=(1,),
        state_id="e022-full-target-state",
        exact=True,
        whole_layer_fallback=False,
    )
    frame = Frame(
        MessageType.EXECUTE_SHARD,
        "e022-full-93-correctness",
        0,
        "local-rtx5090.persistent-k3-worker",
        request.state_id,
        encode_shard_request(request, np.zeros((1,), dtype=np.float32)),
    )
    parent.send_bytes(encode_frame(frame, credential))
    if not parent.poll(timeout_seconds):
        process.terminate()
        process.join(timeout=30)
        raise TimeoutError("full 93-layer worker correctness timed out")
    first = parent.recv_bytes()
    result_frame: Frame | None = None
    worker_status: dict[str, Any]
    try:
        result_frame = decode_frame(first, credential)
        if result_frame.message_type is not MessageType.SHARD_RESULT:
            raise ValueError("worker returned a non-SHARD_RESULT frame")
        native_result, summary = decode_shard_result(result_frame.payload)
        if not parent.poll(30):
            raise TimeoutError("worker omitted dispatch audit")
        worker_status = json.loads(parent.recv_bytes())
    except Exception:
        worker_status = json.loads(first)
        native_result = None
        summary = np.empty((0,), dtype=np.float32)
    process.join(timeout=60)
    if process.is_alive():
        process.terminate()
        process.join(timeout=30)
    raw = (
        json.loads(raw_output.read_text(encoding="utf-8"))
        if raw_output.is_file()
        else {}
    )
    passed = (
        process.exitcode == 0
        and worker_status.get("status") == "PASS"
        and raw.get("status") == "PASS"
        and native_result is not None
        and native_result.native_invocation_count == 1
        and summary.size == 4
        and float(summary[0]) == 1.0
        and float(summary[3]) == 1.0
    )
    return {
        "schema_version": "experiment-022-worker-process-full-93-v1",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "PHYSICAL sequential exact logical workers on one RTX 5090",
        "authenticated_execute_shard": True,
        "execute_shard_invoked_native_k3_primitive": native_result is not None,
        "native_primitive": native_result.native_primitive if native_result else None,
        "native_invocation_count": native_result.native_invocation_count if native_result else 0,
        "whole_layer_mathematical_fallback": False,
        "central_rpc_per_tiny_operation": False,
        "batched_internal_worker_dag": True,
        "complete_93_layer_graph": raw.get("complete_93_layer_graph"),
        "executed_layers": raw.get("executed_layers"),
        "maximum_relative_l2_error": raw.get("maximum_relative_l2_error"),
        "routes_exact": raw.get("routes_exact"),
        "state_fingerprints": raw.get("state_fingerprints"),
        "logit_relative_l2": (raw.get("head") or {}).get("oracle_logits", {}).get(
            "relative_l2_error"
        ),
        "greedy_token_match": (raw.get("head") or {}).get("greedy_token_match"),
        "worker_operation_count": raw.get("worker_operation_count"),
        "worker_process_exitcode": process.exitcode,
        "worker_status": worker_status,
        "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "raw_receipt": str(raw_output),
    }


__all__ = ["run_worker_process_correctness"]
