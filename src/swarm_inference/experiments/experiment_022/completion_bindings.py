"""Physical completion gate for the six production ``EXECUTE_SHARD`` bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime
from swarm_inference.experiments.experiment_019.attention import (
    HEAD_DIMENSION,
    KV_LORA,
    QUERY_LORA,
    QUERY_ROPE,
    DeviceResources,
    _bf16_f32,
    normalized_real_inputs,
)
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.physical import (
    HIDDEN,
    LATENT,
    TOPK,
    real_route_workload,
    striped_latent_down,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
)
from swarm_inference.experiments.experiment_022.native_dispatch import (
    NativeShardDispatcher,
    ShardRequest,
    ShardTaskType,
    decode_shard_result,
    encode_shard_request,
)
from swarm_inference.experiments.experiment_022.resident_primitives import (
    PreparedExpertStripe,
    PreparedKdaShard,
    PreparedLatentDownShard,
    PreparedMlaShard,
    PreparedProjectionShard,
    PreparedReductionContribution,
    PreparedSharedExpertShard,
    _sha256_arrays,
)


def _digest(values: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=np.float64)))
    return float(np.linalg.norm(difference) / max(denominator, 1e-30))


def _state(primitive: Any) -> str:
    method = getattr(primitive, "state_fingerprint", None)
    return str(method()) if callable(method) else ""


def _reset(primitive: Any) -> None:
    method = getattr(primitive, "reset_state", None)
    if callable(method):
        method()


def _binding_case(
    *,
    operation: str,
    worker_id: str,
    assignment_id: str,
    task_type: ShardTaskType,
    primitive: Any,
    values: np.ndarray,
    request: ShardRequest,
) -> dict[str, Any]:
    credential = hashlib.sha256(f"e022-completion:{operation}".encode()).digest()
    try:
        _reset(primitive)
        direct_reads_before = len(primitive.loader.audit)
        direct_started = time.perf_counter_ns()
        direct = primitive(values, request)
        direct_wall_ms = (time.perf_counter_ns() - direct_started) / 1e6
        direct_execution = dict(primitive.last_execution)
        direct_state = _state(primitive)
        direct_reads = len(primitive.loader.audit) - direct_reads_before

        _reset(primitive)
        dispatcher = NativeShardDispatcher(worker_id)
        dispatcher.register(assignment_id, task_type, primitive)
        request_payload = encode_shard_request(request, values)
        request_frame = Frame(
            MessageType.EXECUTE_SHARD,
            f"binding-{operation}",
            request.rows,
            worker_id,
            request.state_id,
            request_payload,
        )
        worker_reads_before = len(primitive.loader.audit)
        protocol_started = time.perf_counter_ns()
        authenticated_request = decode_frame(encode_frame(request_frame, credential), credential)
        response_frame = dispatcher.execute_frame(authenticated_request)
        authenticated_response = decode_frame(
            encode_frame(response_frame, credential), credential
        )
        worker_result, worker = decode_shard_result(authenticated_response.payload)
        worker_protocol_wall_ms = (time.perf_counter_ns() - protocol_started) / 1e6
        worker_state = _state(primitive)
        worker_reads = len(primitive.loader.audit) - worker_reads_before
        worker_execution = dict(primitive.last_execution)
        relative_l2 = _relative_l2(direct, worker)
        output_equal = np.array_equal(direct, worker)
        state_equal = direct_state == worker_state
        fallback_count = sum(
            1 for row in dispatcher.audit if bool(row.get("whole_layer_fallback"))
        )
        checkpoint_reads = direct_reads + worker_reads
        route_applicable = task_type is ShardTaskType.EXPERT_STRIPE
        if route_applicable:
            route_ids = np.rint(values[:, LATENT : LATENT + TOPK]).astype(
                np.int32
            )
            route_weights = np.ascontiguousarray(values[:, LATENT + TOPK :])
            # Primitive digests include shape and dtype as well as payload.  Use
            # the same helper here so equality is derived from the actual input.
            expected_route_ids_sha256 = _sha256_arrays((route_ids,))
            expected_route_weights_sha256 = _sha256_arrays((route_weights,))
            route_equality = (
                direct_execution.get("route_ids_sha256")
                == expected_route_ids_sha256
                and worker_execution.get("route_ids_sha256")
                == expected_route_ids_sha256
                and direct_execution.get("route_weights_sha256")
                == expected_route_weights_sha256
                and worker_execution.get("route_weights_sha256")
                == expected_route_weights_sha256
                and direct_execution.get("ordered_route_ids")
                == route_ids.tolist()
                and worker_execution.get("ordered_route_ids")
                == route_ids.tolist()
            )
            route_basis = "ACTUAL_PACKED_ROUTE_AND_WEIGHT_HASH_COMPARISON"
        else:
            expected_route_ids_sha256 = None
            expected_route_weights_sha256 = None
            route_equality = True
            route_basis = "NOT_APPLICABLE_NO_ROUTE_INPUT"
        status = (
            "PASS"
            if output_equal
            and relative_l2 == 0.0
            and state_equal
            and checkpoint_reads == 0
            and fallback_count == 0
            and dispatcher.assignment_count == 1
            and worker_result.native_primitive == primitive.native_primitive
            and worker_result.native_invocation_count == 2
            and route_equality
            else "FAIL"
        )
        return {
            "operation": operation,
            "task_type": task_type.value,
            "status": status,
            "worker_id": worker_id,
            "assignment_id": assignment_id,
            "layer": request.layer,
            "degree": request.degree,
            "shard_index": request.shard_index,
            "rows": request.rows,
            "input_shape": list(request.input_shape),
            "native_primitive": primitive.native_primitive,
            "production_native_binding": True,
            "authenticated_frame_round_trip": True,
            "direct_result_hash": _digest(direct),
            "worker_result_hash": _digest(worker),
            "relative_l2": relative_l2,
            "output_bitwise_equal": output_equal,
            "route_equality": route_equality,
            "route_equality_basis": route_basis,
            "expected_route_ids_sha256": expected_route_ids_sha256,
            "expected_route_weights_sha256": expected_route_weights_sha256,
            "direct_state_fingerprint": direct_state,
            "worker_state_fingerprint": worker_state,
            "state_equality": state_equal,
            "direct_wall_ms": direct_wall_ms,
            "worker_native_wall_ms": worker_result.wall_ms,
            "worker_native_invocation_count": worker_result.native_invocation_count,
            "worker_protocol_wall_ms": worker_protocol_wall_ms,
            "protocol_overhead_ms": max(
                0.0, worker_protocol_wall_ms - worker_result.wall_ms
            ),
            "direct_execution": direct_execution,
            "worker_execution": worker_execution,
            "resident_bytes": primitive.resident_bytes,
            "startup": primitive.startup,
            "checkpoint_reads_in_timed_region": checkpoint_reads,
            "whole_layer_fallback_count": fallback_count,
            "dispatcher_audit": dispatcher.audit,
        }
    finally:
        primitive.close()


def _real_inputs(
    checkpoint: Path,
    oracle_trace: Path,
    routes_path: Path,
    cuda_library: Path,
    quantizer_library: Path,
    *,
    rows: int,
    layer: int,
    degree: int,
) -> dict[str, np.ndarray]:
    catalog = CheckpointCatalog(checkpoint)
    loader = DirectShardLoader(catalog)
    kda = normalized_real_inputs(catalog, loader, oracle_trace, layer=45, rows=rows)
    mla = normalized_real_inputs(catalog, loader, oracle_trace, layer=47, rows=rows)
    hidden, routes, route_weights, _ = real_route_workload(
        checkpoint,
        oracle_trace,
        routes_path,
        layer=layer,
        rows=rows,
        loader=loader,
    )
    runtime = _CudaRuntime(cuda_library, 0)
    runtime.set_telemetry("minimal")
    quantizer = GpuShardQuantizer(quantizer_library, 0)
    try:
        # Materialize the exact coordinator-owned common projections once.
        # The authenticated KDA/MLA task contract then carries normalized
        # hidden rows plus these common outputs to each native head stripe.
        kda_resources = DeviceResources(runtime)
        try:
            kda_prefix = "language_model.model.layers.45.self_attn"
            f_a_source = loader.reviewed_small(
                f"{kda_prefix}.f_a_proj.weight",
                worker_id="completion.kda-common",
                purpose="completion_kda_common_projection_fixture",
            )
            f_a = kda_resources.tensor(
                runtime.upload_float32(_bf16_f32(f_a_source))
            )
            kda_input = kda_resources.upload(kda)
            decay_low_device = kda_resources.allocate(rows * HEAD_DIMENSION)
            runtime.execute_dense(f_a, decay_low_device, kda_input, rows)
            runtime.synchronize()
            decay_low = runtime.download_activation(
                decay_low_device, (rows, HEAD_DIMENSION)
            )
        finally:
            kda_resources.close()

        mla_resources = DeviceResources(runtime)
        try:
            mla_prefix = "language_model.model.layers.47.self_attn"
            q_a_source = loader.reviewed_small(
                f"{mla_prefix}.q_a_proj.weight",
                worker_id="completion.mla-common",
                purpose="completion_mla_query_common_fixture",
            )
            kv_a_source = loader.reviewed_small(
                f"{mla_prefix}.kv_a_proj_with_mqa.weight",
                worker_id="completion.mla-common",
                purpose="completion_mla_kv_common_fixture",
            )
            q_a = mla_resources.tensor(
                runtime.upload_int8(
                    quantizer.row_int8(q_a_source, owner="completion.mla-common")
                )
            )
            kv_a = mla_resources.tensor(
                runtime.upload_int8(
                    quantizer.row_int8(kv_a_source, owner="completion.mla-common")
                )
            )
            query_norm = mla_resources.upload(
                _bf16_f32(
                    loader.reviewed_small(
                        f"{mla_prefix}.q_a_layernorm.weight",
                        worker_id="completion.mla-common",
                        purpose="completion_mla_query_norm_fixture",
                    )
                )
            )
            mla_input = mla_resources.upload(mla)
            query_low_device = mla_resources.allocate(rows * QUERY_LORA)
            compressed_device = mla_resources.allocate(
                rows * (KV_LORA + QUERY_ROPE)
            )
            runtime.execute_dense(q_a, query_low_device, mla_input, rows)
            runtime.execute_rmsnorm(
                query_low_device,
                query_low_device,
                query_norm,
                batch=rows,
                dimension=QUERY_LORA,
                epsilon=1e-5,
            )
            runtime.execute_dense(kv_a, compressed_device, mla_input, rows)
            runtime.synchronize()
            query_low = runtime.download_activation(
                query_low_device, (rows, QUERY_LORA)
            )
            compressed = runtime.download_activation(
                compressed_device, (rows, KV_LORA + QUERY_ROPE)
            )
        finally:
            mla_resources.close()

        latent, _ = striped_latent_down(
            runtime,
            loader,
            hidden,
            layer=layer,
            degree=degree,
            warmup=0,
            iterations=1,
            quantizer=quantizer,
        )
    finally:
        runtime.close()
    packed_expert = np.ascontiguousarray(
        np.concatenate(
            [latent, routes.astype(np.float32), route_weights.astype(np.float32)], axis=1
        ),
        dtype=np.float32,
    )
    reduction_rng = np.random.default_rng(22022)
    reduction = np.ascontiguousarray(
        reduction_rng.normal(size=(degree, rows, HIDDEN)), dtype=np.float32
    )
    return {
        "kda": np.ascontiguousarray(
            np.concatenate([kda, decay_low], axis=1), dtype=np.float32
        ),
        "mla": np.ascontiguousarray(
            np.concatenate([mla, query_low, compressed], axis=1), dtype=np.float32
        ),
        "expert": packed_expert,
        "shared": np.ascontiguousarray(hidden, dtype=np.float32),
        "latent_down": np.ascontiguousarray(hidden, dtype=np.float32),
        "latent_up": np.ascontiguousarray(latent[:, : LATENT // degree]),
        "reduction": reduction,
    }


def run_binding_gate(
    *,
    checkpoint: Path,
    cuda_library: Path,
    quantizer_library: Path,
    grouped_library: Path,
    oracle_trace: Path,
    routes_path: Path,
    rows: int = 4,
    degree: int = 8,
) -> dict[str, Any]:
    fixtures = _real_inputs(
        checkpoint,
        oracle_trace,
        routes_path,
        cuda_library,
        quantizer_library,
        rows=rows,
        layer=89,
        degree=degree,
    )
    cases: list[dict[str, Any]] = []

    def run(
        operation: str,
        task_type: ShardTaskType,
        layer: int,
        primitive_factory: Callable[[], Any],
        values: np.ndarray,
        *,
        state_id: str = "",
    ) -> None:
        assignment = f"completion.{operation}.layer-{layer}.p{degree}.s0"
        request = ShardRequest(
            assignment,
            task_type,
            layer,
            0,
            degree,
            rows,
            tuple(int(value) for value in values.shape),
            state_id=state_id,
        )
        cases.append(
            _binding_case(
                operation=operation,
                worker_id=f"worker-{operation}",
                assignment_id=assignment,
                task_type=task_type,
                primitive=primitive_factory(),
                values=values,
                request=request,
            )
        )

    run(
        "kda_shard",
        ShardTaskType.KDA_SHARD,
        45,
        lambda: PreparedKdaShard(
            checkpoint,
            cuda_library,
            quantizer_library,
            layer=45,
            degree=degree,
            shard_index=0,
            max_rows=rows,
            precomputed_common=True,
        ),
        fixtures["kda"],
        state_id="kda-completion-state",
    )
    run(
        "mla_shard",
        ShardTaskType.MLA_SHARD,
        47,
        lambda: PreparedMlaShard(
            checkpoint,
            cuda_library,
            quantizer_library,
            layer=47,
            degree=degree,
            shard_index=0,
            max_rows=rows,
            precomputed_common=True,
        ),
        fixtures["mla"],
        state_id="mla-completion-state",
    )
    run(
        "routed_expert_stripe",
        ShardTaskType.EXPERT_STRIPE,
        89,
        lambda: PreparedExpertStripe(
            checkpoint,
            cuda_library,
            grouped_library,
            layer=89,
            degree=degree,
            shard_index=0,
            max_rows=rows,
        ),
        fixtures["expert"],
    )
    run(
        "shared_expert_shard",
        ShardTaskType.SHARED_EXPERT_SHARD,
        89,
        lambda: PreparedSharedExpertShard(
            checkpoint,
            cuda_library,
            quantizer_library,
            layer=89,
            degree=degree,
            shard_index=0,
            max_rows=rows,
        ),
        fixtures["shared"],
    )
    run(
        "projection_shard",
        ShardTaskType.PROJECTION_SHARD,
        89,
        lambda: PreparedProjectionShard(
            checkpoint,
            cuda_library,
            quantizer_library,
            layer=89,
            degree=degree,
            shard_index=0,
            max_rows=rows,
        ),
        fixtures["latent_up"],
    )
    run(
        "reduction_contribution",
        ShardTaskType.REDUCTION_CONTRIBUTION,
        89,
        lambda: PreparedReductionContribution(
            checkpoint,
            cuda_library,
            layer=89,
            degree=degree,
            participants=degree,
            dimension=HIDDEN,
            max_rows=rows,
        ),
        fixtures["reduction"],
    )
    reduction_variants: list[dict[str, Any]] = []
    projection_variants: list[dict[str, Any]] = []
    for name, primitive_factory, values in (
        (
            "latent_down",
            lambda: PreparedLatentDownShard(
                checkpoint,
                cuda_library,
                quantizer_library,
                layer=89,
                degree=degree,
                shard_index=0,
                max_rows=rows,
            ),
            fixtures["latent_down"],
        ),
        (
            "latent_up",
            lambda: PreparedProjectionShard(
                checkpoint,
                cuda_library,
                quantizer_library,
                layer=89,
                degree=degree,
                shard_index=0,
                max_rows=rows,
            ),
            fixtures["latent_up"],
        ),
    ):
        assignment = f"completion.projection.{name}.layer-89.p{degree}.s0"
        request = ShardRequest(
            assignment,
            ShardTaskType.PROJECTION_SHARD,
            89,
            0,
            degree,
            rows,
            tuple(int(value) for value in values.shape),
        )
        projection_variants.append(
            _binding_case(
                operation=f"projection_{name}",
                worker_id=f"worker-projection-{name}",
                assignment_id=assignment,
                task_type=ShardTaskType.PROJECTION_SHARD,
                primitive=primitive_factory(),
                values=values,
                request=request,
            )
        )
    reduction_rng = np.random.default_rng(22023 + rows)
    for name, participants, dimension in (
        ("hidden_p8", degree, HIDDEN),
        ("latent_p8", degree, LATENT),
        ("hidden_p2", 2, HIDDEN),
    ):
        values = np.ascontiguousarray(
            reduction_rng.normal(size=(participants, rows, dimension)),
            dtype=np.float32,
        )
        assignment = f"completion.reduction.{name}.layer-89.rows-{rows}"
        request = ShardRequest(
            assignment,
            ShardTaskType.REDUCTION_CONTRIBUTION,
            89,
            0,
            participants,
            rows,
            tuple(int(value) for value in values.shape),
        )
        reduction_variants.append(
            _binding_case(
                operation=f"reduction_{name}",
                worker_id=f"worker-reduction-{name}",
                assignment_id=assignment,
                task_type=ShardTaskType.REDUCTION_CONTRIBUTION,
                primitive=PreparedReductionContribution(
                    checkpoint,
                    cuda_library,
                    layer=89,
                    degree=participants,
                    participants=participants,
                    dimension=dimension,
                    max_rows=rows,
                ),
                values=values,
                request=request,
            )
        )
    passed = all(row["status"] == "PASS" for row in cases)
    variants_passed = all(row["status"] == "PASS" for row in reduction_variants)
    projection_variants_passed = all(
        row["status"] == "PASS" for row in projection_variants
    )
    return {
        "schema_version": "experiment-022-completion-execute-shard-bindings-v1",
        "evidence_class": "PHYSICAL",
        "status": "PASS"
        if passed
        and variants_passed
        and projection_variants_passed
        and len(cases) == 6
        else "FAIL",
        "rows": rows,
        "degree": degree,
        "operation_count": len(cases),
        "all_six_bound": len(cases) == 6,
        "all_production_native": all(row["production_native_binding"] for row in cases),
        "all_authenticated": all(row["authenticated_frame_round_trip"] for row in cases),
        "checkpoint_reads_in_timed_region": sum(
            int(row["checkpoint_reads_in_timed_region"]) for row in cases
        ),
        "whole_layer_fallback_count": sum(
            int(row["whole_layer_fallback_count"]) for row in cases
        ),
        "cases": cases,
        "projection_service_variants": projection_variants,
        "reduction_service_variants": reduction_variants,
    }


def aggregate_binding_sweep(
    receipts: list[dict[str, Any]], *, output: Path
) -> dict[str, Any]:
    """Freeze the chunk-1/2/4 binding evidence into the required artifact."""

    by_chunk = {int(row["rows"]): row for row in receipts}
    if set(by_chunk) != {1, 2, 4}:
        raise ValueError("binding sweep must contain exact chunks 1, 2, and 4")
    operations = {
        str(case["operation"])
        for receipt in receipts
        for case in receipt.get("cases", [])
    }
    passed = (
        operations
        == {
            "kda_shard",
            "mla_shard",
            "routed_expert_stripe",
            "shared_expert_shard",
            "projection_shard",
            "reduction_contribution",
        }
        and all(row.get("status") == "PASS" for row in receipts)
    )
    value = {
        "schema_version": "experiment-022-completion-execute-shard-bindings-v2",
        "evidence_class": "PHYSICAL",
        "status": "PASS" if passed else "FAIL",
        "chunks_physically_executed": [1, 2, 4],
        "degree": 8,
        "operation_count": len(operations),
        "all_six_bound": len(operations) == 6,
        "all_production_native": all(
            case["production_native_binding"]
            for receipt in receipts
            for case in receipt["cases"]
        ),
        "all_authenticated": all(
            case["authenticated_frame_round_trip"]
            for receipt in receipts
            for case in receipt["cases"]
        ),
        "checkpoint_reads_in_timed_region": sum(
            int(case["checkpoint_reads_in_timed_region"])
            for receipt in receipts
            for case in receipt["cases"]
        ),
        "whole_layer_fallback_count": sum(
            int(case["whole_layer_fallback_count"])
            for receipt in receipts
            for case in receipt["cases"]
        ),
        "operations": sorted(operations),
        "chunk_receipts": [by_chunk[chunk] for chunk in (1, 2, 4)],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _child(arguments: dict[str, str], output: str) -> None:
    path = Path(output)
    try:
        receipt = run_binding_gate(
            checkpoint=Path(arguments["checkpoint"]),
            cuda_library=Path(arguments["cuda_library"]),
            quantizer_library=Path(arguments["quantizer_library"]),
            grouped_library=Path(arguments["grouped_library"]),
            oracle_trace=Path(arguments["oracle_trace"]),
            routes_path=Path(arguments["routes_path"]),
            rows=int(arguments["rows"]),
            degree=int(arguments["degree"]),
        )
    except BaseException as exc:  # preserve a machine-readable failed gate
        receipt = {
            "schema_version": "experiment-022-completion-execute-shard-bindings-v1",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--quantizer-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--oracle-trace", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--degree", type=int, default=8)
    args = parser.parse_args()
    mapping = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cuda_library": str(args.cuda_library.resolve()),
        "quantizer_library": str(args.quantizer_library.resolve()),
        "grouped_library": str(args.grouped_library.resolve()),
        "oracle_trace": str(args.oracle_trace.resolve()),
        "routes_path": str(args.routes.resolve()),
        "rows": str(args.rows),
        "degree": str(args.degree),
    }
    context = mp.get_context("spawn")
    process = context.Process(target=_child, args=(mapping, str(args.output.resolve())))
    process.start()
    process.join()
    if process.exitcode != 0:
        return int(process.exitcode or 1)
    receipt = json.loads(args.output.read_text(encoding="utf-8"))
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
