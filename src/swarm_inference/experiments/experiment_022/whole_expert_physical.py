"""Physical whole-expert capability gate for the E022 completion pass."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime, _numerical_metrics
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
)
from swarm_inference.experiments.experiment_019.physical import (
    LATENT,
    _upload_whole_active_experts,
    real_route_workload,
    striped_latent_down,
)
from swarm_inference.experiments.experiment_019.quantization import GpuShardQuantizer
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
    _execute_grouped,
)
from swarm_inference.experiments.experiment_020.transport import (
    Frame,
    MessageType,
    decode_frame,
    encode_frame,
    new_run_credential,
)
from swarm_inference.experiments.experiment_022.io import atomic_write_json
from swarm_inference.experiments.experiment_022.native_dispatch import (
    NativeShardDispatcher,
    ShardRequest,
    ShardTaskType,
    decode_shard_result,
    encode_shard_request,
)
from swarm_inference.experiments.experiment_022.resident_primitives import (
    PreparedReductionContribution,
    _sha256_arrays,
)
from swarm_inference.experiments.experiment_022.whole_expert import (
    PreparedWholeExpertGroup,
)

CHUNKS = (1, 2, 4)
DEGREE = 8
LAYERS = (
    ("calibration", "KDA", 45),
    ("calibration", "Gated_MLA", 47),
    ("heldout", "KDA", 89),
    ("heldout", "Gated_MLA", 91),
)
RELATIVE_L2_GATE = 2e-6


def _fixture(
    checkpoint: Path,
    cuda_library: Path,
    quantizer_library: Path,
    oracle_root: Path,
    *,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    catalog = CheckpointCatalog(checkpoint)
    loader = DirectShardLoader(catalog)
    hidden, routes, weights, _ = real_route_workload(
        checkpoint,
        oracle_root / "hidden-trace.f32",
        oracle_root / "routes.txt",
        layer=layer,
        rows=max(CHUNKS),
        loader=loader,
    )
    runtime = _CudaRuntime(cuda_library, 0)
    runtime.set_telemetry("minimal")
    quantizer = GpuShardQuantizer(quantizer_library, 0)
    try:
        latent, _ = striped_latent_down(
            runtime,
            loader,
            hidden,
            layer=layer,
            degree=DEGREE,
            warmup=0,
            iterations=1,
            quantizer=quantizer,
        )
    finally:
        runtime.close()
    packed = np.ascontiguousarray(
        np.concatenate(
            [latent, routes.astype(np.float32), weights.astype(np.float32)], axis=1
        ),
        dtype=np.float32,
    )
    return packed, latent, routes, weights


def _reference(
    checkpoint: Path,
    cuda_library: Path,
    grouped_library: Path,
    *,
    layer: int,
    latent: np.ndarray,
    routes: np.ndarray,
    weights: np.ndarray,
) -> dict[int, np.ndarray]:
    runtime = _CudaRuntime(cuda_library, 0)
    runtime.set_telemetry("minimal")
    grouped = GroupedTop16Runtime(grouped_library)
    active = sorted({int(value) for value in routes.reshape(-1)})
    resident = _upload_whole_active_experts(
        runtime,
        checkpoint,
        layer=layer,
        experts=active,
    )
    try:
        return {
            rows: _execute_grouped(
                runtime,
                grouped,
                resident,
                latent[:rows],
                routes[:rows],
                weights[:rows],
                warmup=1,
                iterations=3,
            )[1]
            for rows in CHUNKS
        }
    finally:
        resident.close()
        grouped.close()
        runtime.close()


def _authenticated_call(
    dispatcher: NativeShardDispatcher,
    credential: bytes,
    worker_id: str,
    request: ShardRequest,
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = Frame(
        MessageType.EXECUTE_SHARD,
        f"whole-expert-{request.layer}-{request.shard_index}-{request.rows}",
        request.rows,
        worker_id,
        request.state_id,
        encode_shard_request(request, values),
    )
    started = time.perf_counter_ns()
    incoming = decode_frame(encode_frame(frame, credential), credential)
    response = dispatcher.execute_frame(incoming)
    authenticated = decode_frame(encode_frame(response, credential), credential)
    result, output = decode_shard_result(authenticated.payload)
    protocol_wall_ms = (time.perf_counter_ns() - started) / 1e6
    return output, {
        "native_wall_ms": result.wall_ms,
        "protocol_wall_ms": protocol_wall_ms,
        "protocol_overhead_ms": max(0.0, protocol_wall_ms - result.wall_ms),
        "native_primitive": result.native_primitive,
        "native_invocation_count": result.native_invocation_count,
    }


def _layer_gate(
    checkpoint: Path,
    cuda_library: Path,
    quantizer_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    *,
    split: str,
    layer_type: str,
    layer: int,
) -> dict[str, Any]:
    packed, latent, routes, weights = _fixture(
        checkpoint,
        cuda_library,
        quantizer_library,
        oracle_root,
        layer=layer,
    )
    reference = _reference(
        checkpoint,
        cuda_library,
        grouped_library,
        layer=layer,
        latent=latent,
        routes=routes,
        weights=weights,
    )
    credential = new_run_credential()
    outputs: dict[int, list[np.ndarray]] = {rows: [] for rows in CHUNKS}
    workers: list[dict[str, Any]] = []
    for shard in range(DEGREE):
        worker_id = f"whole-expert.layer-{layer}.p{DEGREE}.worker-{shard:02d}"
        assignment = f"layer-{layer}:WHOLE_EXPERT:p{DEGREE}:s{shard}"
        primitive = PreparedWholeExpertGroup(
            checkpoint,
            cuda_library,
            grouped_library,
            layer=layer,
            degree=DEGREE,
            shard_index=shard,
            max_rows=max(CHUNKS),
        )
        dispatcher = NativeShardDispatcher(worker_id)
        dispatcher.register(assignment, ShardTaskType.WHOLE_EXPERT_GROUP, primitive)
        row_receipts: list[dict[str, Any]] = []
        try:
            for rows in CHUNKS:
                request = ShardRequest(
                    assignment,
                    ShardTaskType.WHOLE_EXPERT_GROUP,
                    layer,
                    shard,
                    DEGREE,
                    rows,
                    tuple(int(value) for value in packed[:rows].shape),
                    state_id=f"whole-expert:{layer}:{shard}",
                )
                samples: list[dict[str, Any]] = []
                output = np.empty((rows, LATENT), dtype=np.float32)
                for iteration in range(4):
                    output, sample = _authenticated_call(
                        dispatcher,
                        credential,
                        worker_id,
                        request,
                        packed[:rows],
                    )
                    sample["cuda_ms"] = float(primitive.last_execution["cuda_ms"])
                    sample["checkpoint_reads_in_timed_region"] = int(
                        primitive.last_execution["checkpoint_reads_in_timed_region"]
                    )
                    sample["input_route_ids_sha256"] = primitive.last_execution[
                        "input_route_ids_sha256"
                    ]
                    sample["input_route_weights_sha256"] = primitive.last_execution[
                        "input_route_weights_sha256"
                    ]
                    sample["route_order_preserved"] = bool(
                        primitive.last_execution["route_order_preserved"]
                    )
                    if iteration:
                        samples.append(sample)
                outputs[rows].append(output)
                expected_route_ids_sha256 = _sha256_arrays(
                    (np.ascontiguousarray(routes[:rows], dtype=np.int32),)
                )
                expected_route_weights_sha256 = _sha256_arrays(
                    (np.ascontiguousarray(weights[:rows], dtype=np.float32),)
                )
                row_receipts.append(
                    {
                        "chunk_rows": rows,
                        "native_wall_p50_ms": statistics.median(
                            float(value["native_wall_ms"]) for value in samples
                        ),
                        "cuda_p50_ms": statistics.median(
                            float(value["cuda_ms"]) for value in samples
                        ),
                        "protocol_overhead_p50_ms": statistics.median(
                            float(value["protocol_overhead_ms"]) for value in samples
                        ),
                        "checkpoint_reads_in_timed_region": sum(
                            int(value["checkpoint_reads_in_timed_region"])
                            for value in samples
                        ),
                        "expected_route_ids_sha256": expected_route_ids_sha256,
                        "expected_route_weights_sha256": expected_route_weights_sha256,
                        "input_route_ids_equal": all(
                            value["input_route_ids_sha256"]
                            == expected_route_ids_sha256
                            for value in samples
                        ),
                        "input_route_weights_equal": all(
                            value["input_route_weights_sha256"]
                            == expected_route_weights_sha256
                            for value in samples
                        ),
                        "route_order_preserved": all(
                            value["route_order_preserved"] for value in samples
                        ),
                        "samples": samples,
                    }
                )
            workers.append(
                {
                    "worker_id": worker_id,
                    "shard_index": shard,
                    "expert_id_start": primitive.expert_start,
                    "expert_id_stop_exclusive": primitive.expert_stop,
                    "owned_expert_count": len(primitive.expert_ids),
                    "resident_bytes": primitive.resident_bytes,
                    "startup": primitive.startup,
                    "rows": row_receipts,
                    "dispatcher_audit": dispatcher.audit,
                }
            )
        finally:
            primitive.close()

    reductions: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    reduction = PreparedReductionContribution(
        checkpoint,
        cuda_library,
        layer=layer,
        degree=DEGREE,
        participants=DEGREE,
        dimension=LATENT,
        max_rows=max(CHUNKS),
    )
    reduction_dispatcher = NativeShardDispatcher(f"whole-expert.layer-{layer}.reducer")
    reduction_dispatcher.register(
        f"layer-{layer}:WHOLE_EXPERT:reduction",
        ShardTaskType.REDUCTION_CONTRIBUTION,
        reduction,
    )
    try:
        for rows in CHUNKS:
            contributions = np.ascontiguousarray(np.stack(outputs[rows]), dtype=np.float32)
            request = ShardRequest(
                f"layer-{layer}:WHOLE_EXPERT:reduction",
                ShardTaskType.REDUCTION_CONTRIBUTION,
                layer,
                0,
                DEGREE,
                rows,
                tuple(int(value) for value in contributions.shape),
            )
            reduced, reduction_receipt = _authenticated_call(
                reduction_dispatcher,
                credential,
                f"whole-expert.layer-{layer}.reducer",
                request,
                contributions,
            )
            reduction_receipt["cuda_ms"] = float(reduction.last_execution["cuda_ms"])
            reductions.append({"chunk_rows": rows, **reduction_receipt})
            metrics = _numerical_metrics(reference[rows], reduced)
            worker_route_receipts = [
                next(
                    value
                    for value in worker["rows"]
                    if value["chunk_rows"] == rows
                )
                for worker in workers
            ]
            correctness.append(
                {
                    "chunk_rows": rows,
                    "relative_l2": float(metrics["relative_l2_error"]),
                    "maximum_absolute_error": float(metrics["maximum_absolute_error"]),
                    "finite": bool(np.isfinite(reduced).all()),
                    "routes_exact": all(
                        row["input_route_ids_equal"]
                        and row["input_route_weights_equal"]
                        for row in worker_route_receipts
                    ),
                    "ordered_expert_ids_equal": all(
                        row["route_order_preserved"]
                        for row in worker_route_receipts
                    ),
                    "each_route_owned_exactly_once": all(
                        sum(
                            int(worker["expert_id_start"] <= int(expert) < worker["expert_id_stop_exclusive"])
                            for worker in workers
                        )
                        == 1
                        for expert in routes[:rows].reshape(-1)
                    ),
                }
            )
    finally:
        reduction.close()

    timed_reads = sum(
        int(row["checkpoint_reads_in_timed_region"])
        for worker in workers
        for row in worker["rows"]
    )
    passed = (
        timed_reads == 0
        and all(
            row["finite"]
            and row["routes_exact"]
            and row["ordered_expert_ids_equal"]
            and row["each_route_owned_exactly_once"]
            and float(row["relative_l2"]) <= RELATIVE_L2_GATE
            for row in correctness
        )
        and all(
            not bool(row.get("whole_layer_fallback"))
            for worker in workers
            for row in worker["dispatcher_audit"]
        )
    )
    services = []
    for rows in CHUNKS:
        condition = [
            next(value for value in worker["rows"] if value["chunk_rows"] == rows)
            for worker in workers
        ]
        services.append(
            {
                "split": split,
                "layer": layer,
                "layer_type": layer_type,
                "operation": "expert_whole_group",
                "degree": DEGREE,
                "rows": rows,
                "native_p50_ms": max(float(value["native_wall_p50_ms"]) for value in condition),
                "protocol_overhead_p50_ms": max(
                    float(value["protocol_overhead_p50_ms"]) for value in condition
                ),
                "correctness_pass": passed,
                "evidence_class": "PHYSICAL RTX 5090 resident whole-expert group",
                "production_native_binding": True,
            }
        )
    return {
        "status": "PASS" if passed else "FAIL",
        "split": split,
        "layer_type": layer_type,
        "layer": layer,
        "degree": DEGREE,
        "chunks": list(CHUNKS),
        "workers": workers,
        "reductions": reductions,
        "correctness": correctness,
        "services": services,
        "timed_checkpoint_reads": timed_reads,
        "network_behavior": {
            "input_per_worker": "rows * (latent float32 + 16 route IDs + 16 route weights)",
            "output_per_worker": "rows * latent float32 contribution",
            "fanout_and_gather_modelled_separately": True,
            "local_reduction": "coli_cuda_copy_add_reduction",
        },
    }


def run(
    *,
    checkpoint: Path,
    cuda_library: Path,
    quantizer_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    completion_root: Path,
) -> dict[str, Any]:
    results = [
        _layer_gate(
            checkpoint,
            cuda_library,
            quantizer_library,
            grouped_library,
            oracle_root,
            split=split,
            layer_type=layer_type,
            layer=layer,
        )
        for split, layer_type, layer in LAYERS
    ]
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    receipt = {
        "schema_version": "experiment-022-completion-whole-expert-v1",
        "status": status,
        "applicability": "APPLICABLE_DISTINCT_PLACEMENT_UNIT",
        "technical_reason": (
            "K3 has 896 independently routed experts; complete expert-ID groups have "
            "distinct ownership and communication from intermediate-dimension stripes"
        ),
        "degree": DEGREE,
        "chunks_physically_validated": list(CHUNKS),
        "calibration_layers": {"KDA": 45, "Gated_MLA": 47},
        "heldout_layers": {"KDA": 89, "Gated_MLA": 91},
        "results": results,
        "services": [service for row in results for service in row["services"]],
        "claim_boundary": (
            "sequential local RTX 5090 physical service/correctness; network transport "
            "remains event-modelled, not physically distributed"
        ),
    }
    atomic_write_json(
        completion_root / "physical" / "whole-expert-services.json", receipt
    )
    atomic_write_json(
        completion_root / "implementation" / "whole-expert-status.json", receipt
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--quantizer-library", type=Path, required=True)
    parser.add_argument("--grouped-library", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        checkpoint=args.checkpoint.resolve(),
        cuda_library=args.cuda_library.resolve(),
        quantizer_library=args.quantizer_library.resolve(),
        grouped_library=args.grouped_library.resolve(),
        oracle_root=args.oracle_root.resolve(),
        completion_root=args.completion_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "degree": result["degree"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
