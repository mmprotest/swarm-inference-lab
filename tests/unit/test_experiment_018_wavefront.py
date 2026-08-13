from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm_inference.experiments.experiment_015.network import NetworkProfile
from swarm_inference.experiments.experiment_018.analysis import (
    parse_oracle_route_weights,
    parse_oracle_routes,
    topology_transport_ms,
)
from swarm_inference.experiments.experiment_018.attnres_benchmark import (
    _score,
    cached_mix,
    reference_mix,
)
from swarm_inference.experiments.experiment_018.dag import k3_dependency_proof
from swarm_inference.experiments.experiment_018.fine_wavefront import FineWavefrontModel
from swarm_inference.experiments.experiment_018.microshard_benchmark import (
    split_native_expert,
)
from swarm_inference.experiments.experiment_018.wavefront import (
    DeterministicEventEngine,
    HierarchicalTaskBatcher,
    ImmutableObjectCache,
    MicrocellServiceProfile,
    PersistentMicrocellWorker,
    StaleObjectError,
    TaskKind,
    WavefrontModel,
    WorkerEnvelope,
    build_fine_expert_tasks,
    partition_rows,
)
from swarm_inference.model.mxfp4 import MXFP4Tensor


def _profiles() -> list[MicrocellServiceProfile]:
    profiles: list[MicrocellServiceProfile] = []
    for cell in range(12):
        compute = {1: 10.0 + cell / 4, 2: 17.0 + cell / 3, 4: 30.0 + cell / 2, 8: 54.0 + cell}
        profiles.append(
            MicrocellServiceProfile(
                microcell_id=cell,
                layer_start=cell * 8,
                layer_end=min(93, (cell + 1) * 8),
                compute_ms_by_rows=compute,
                cuda_ms_by_rows={rows: value - 1 for rows, value in compute.items()},
                host_overhead_ms_by_rows={rows: 1.0 for rows in compute},
                phase_ms_by_rows={rows: {"expert": value * 0.5} for rows, value in compute.items()},
                resident_bytes=1_000_000,
                bottleneck_operator="expert",
            )
        )
    return profiles


def test_measured_row_partition_never_interpolates() -> None:
    assert partition_rows(17, 8) == (8, 8, 1)
    assert partition_rows(13, 4) == (4, 4, 4, 1)
    assert partition_rows(8, 2) == (2, 2, 2, 2)
    assert partition_rows(5, 1) == (1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        partition_rows(17, 3)


def test_inherited_block7_topology_is_not_reopened() -> None:
    assert topology_transport_ms(8) == pytest.approx(165.44697168)


def test_oracle_route_parser_uses_execution_order_not_batch_column(
    tmp_path: Path,
) -> None:
    lines = []
    record = 0
    for _position in range(2):
        for layer in range(1, 93):
            assignments = " ".join(
                f"{expert}:{0.01 * (expert + 1):.4f}" for expert in range(16)
            )
            lines.append(f"{record} 0 {layer} {assignments}")
            record += 1
    path = tmp_path / "routes.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    routes = parse_oracle_routes(path)
    weights = parse_oracle_route_weights(path)
    assert len(routes[89]) == 2
    assert len(weights[89]) == 2
    assert routes[89][0] == routes[89][1] == tuple(range(16))


def test_dependency_proof_covers_required_k3_operator_families() -> None:
    proof = k3_dependency_proof()
    assert len(proof["layer_types"]) == 10
    assert all(row["exact_streaming_legal"] for row in proof["layer_types"])
    assert proof["fixed_topology"]["microcell_count"] == 12
    assert proof["fixed_topology"]["internal_boundaries"] == 81
    assert proof["fixed_topology"]["coarse_boundaries"] == 11


def test_wavefront_uses_independent_resources_and_real_dependencies() -> None:
    result = WavefrontModel(_profiles()).run(
        block_candidates=16,
        maximum_chunk_rows=1,
        cache_enabled=True,
    )
    records = result.event_run.record_map()
    for chunk in range(17):
        for cell in range(12):
            current = records[f"compute:c{cell:02d}:q{chunk:03d}"]
            assert current.resource_id == f"microcell:{cell:02d}"
            if chunk:
                previous = records[f"compute:c{cell:02d}:q{chunk - 1:03d}"]
                assert current.start_ms >= previous.finish_ms
            if cell:
                arrival = records[f"handoff:c{cell - 1:02d}-c{cell:02d}:q{chunk:03d}"]
                assert current.start_ms >= arrival.finish_ms
    assert result.event_run.peak_concurrency > 1
    assert result.max_chunks_in_flight > 1
    assert result.max_chunks_in_flight <= 17
    assert result.event_run.makespan_ms < result.event_run.serial_sum_ms
    assert result.useful_parallelism > 1
    assert result.network.total_attnres_reduction_fraction >= 0.8


def test_cache_seed_is_counted_and_current_representation_is_preserved() -> None:
    model = WavefrontModel(_profiles())
    current = model.run(block_candidates=7, maximum_chunk_rows=2, cache_enabled=False)
    cached = model.run(block_candidates=7, maximum_chunk_rows=2, cache_enabled=True)
    assert current.network.current_total_bytes == cached.network.current_total_bytes
    assert current.network.cached_total_bytes == 0
    assert cached.network.cache_seed_bytes > 0
    assert cached.network.cache_reference_bytes > 0
    assert cached.network.cached_total_bytes == (
        cached.network.internal_boundary_bytes + cached.network.coarse_boundary_bytes
    )
    assert cached.bytes == cached.network.coarse_boundary_bytes


def test_immutable_cache_rejects_stale_or_mutated_versions() -> None:
    cache = ImmutableObjectCache(maximum_bytes=1024)
    value = cache.create(
        object_type="attnres",
        request_id="r",
        model_scope="block-0",
        version=1,
        producer="cell-0",
        consumers=("cell-1",),
        lifetime="request",
        invalidator="request-end",
        payload=b"exact",
    )
    object_id = cache.seed(value)
    cache.validate_versions({object_id: 1})
    assert cache.get(
        object_id,
        consumer="cell-1",
        expected_version=1,
        expected_hash=value.content_hash,
    ) == value
    mutated = cache.create(
        object_type="attnres",
        request_id="r",
        model_scope="block-0",
        version=1,
        producer="cell-0",
        consumers=("cell-1",),
        lifetime="request",
        invalidator="request-end",
        payload=b"changed",
    )
    with pytest.raises(StaleObjectError):
        cache.seed(mutated)
    assert cache.cleanup_request("r") == 1
    assert cache.snapshot()["bytes"] == 0


def test_persistent_worker_duplicate_loss_retry_stale_object_and_restart() -> None:
    attempts: dict[str, int] = {}

    def handler(item: WorkerEnvelope) -> str:
        attempts[item.task_id] = attempts.get(item.task_id, 0) + 1
        if item.payload == "transient" and attempts[item.task_id] == 1:
            raise RuntimeError("injected child failure")
        return f"done:{item.payload}"

    worker = PersistentMicrocellWorker("cell-0", handler)
    worker.start()
    first = WorkerEnvelope("t0", "r", 0, {}, "transient")
    worker.submit(first)
    assert worker.take().status == "FAIL"
    worker.submit(first)
    assert worker.take().status == "PASS"
    worker.submit(first)
    duplicate = worker.take()
    assert duplicate.status == "PASS"
    assert worker.duplicate_count == 1
    worker.submit(WorkerEnvelope("t2", "r", 2, {}, "skipped"))
    assert worker.take().status == "LOST_PREDECESSOR"
    worker.submit(WorkerEnvelope("t1", "r", 1, {"missing": 1}, "stale"))
    assert worker.take().status == "STALE_OBJECT"
    worker.submit(WorkerEnvelope("t1-retry", "r", 1, {}, "retry"))
    assert worker.take().status == "PASS"
    worker.cleanup_request("r")
    worker.close()

    replacement = PersistentMicrocellWorker("cell-0-restarted", handler)
    replacement.start()
    replacement.submit(WorkerEnvelope("restart-0", "r", 0, {}, "restart"))
    assert replacement.take().status == "PASS"
    replacement.close()


def test_fine_expert_fanout_has_stable_hierarchical_reduction() -> None:
    network = NetworkProfile("local", rtt_ms=0.25, bandwidth_gbps=25.0)
    tasks = build_fine_expert_tasks(
        request_id="r",
        block_id="b",
        chunk_id=0,
        microcell_id=4,
        layer=32,
        expert_ids=(7, 11),
        split_degree=4,
        shard_service_ms=2.0,
        partial_payload_bytes=7168 * 4,
        internal_network=network,
    )
    leaves = [task for task in tasks if task.kind == TaskKind.EXPERT]
    reductions = [task for task in tasks if task.kind == TaskKind.REDUCTION]
    assert len(leaves) == 8
    assert len(reductions) == 7
    assert all(len(task.dependencies) == 2 for task in reductions)
    run = DeterministicEventEngine().run(tasks)
    assert run.peak_concurrency == 8
    assert math.isclose(run.makespan_ms, 2.0 + 3 * network.service_ms(7168 * 4))


def test_fine_wavefront_coexecutes_explicit_shards_and_batched_tree_levels() -> None:
    routes = {
        layer: tuple(tuple(range(16)) for _ in range(3)) for layer in range(1, 93)
    }
    expert_components = {
        cell: {1: 4.0, 2: 6.0, 4: 10.0, 8: 18.0} for cell in range(12)
    }
    fine = FineWavefrontModel(
        _profiles(),
        expert_component_ms_by_cell_rows=expert_components,
        shard_service_ms_by_rows={1: 0.2, 2: 0.3, 4: 0.5, 8: 0.8},
        routes_by_layer_position=routes,
        route_weights_by_layer_position={
            layer: tuple(tuple(0.0625 for _ in range(16)) for _ in range(3))
            for layer in range(1, 93)
        },
        split_degree=8,
    )
    result = fine.run(
        block_candidates=4,
        maximum_chunk_rows=4,
        cache_enabled=False,
    )
    records = result.event_run.record_map()
    leaf = records["fine-expert:c00:q000:l01:e000:s00"]
    fanout = records["fine-fanout:c00:q000:l01"]
    reduction = records["fine-reduce:c00:q000:l01:d00"]
    assert leaf.start_ms >= fanout.finish_ms
    assignments = leaf.metadata["route_weight_assignments"]
    assert assignments == [
        (row, 0, 0.0625) for row in range(4)
    ]
    assert reduction.start_ms >= leaf.finish_ms
    assert result.logical_task_count > result.physical_event_count
    assert result.message_count > result.physical_event_count
    assert result.peak_concurrency >= 128
    assert result.speedup_vs_coarse_wavefront > 0


def test_native_mxfp4_microshards_slice_rows_and_matching_down_columns() -> None:
    latent = 64
    intermediate = 128
    gate = MXFP4Tensor(
        packed=np.arange(intermediate * latent // 2, dtype=np.uint8).reshape(
            intermediate, latent // 2
        ),
        scales=np.full((intermediate, latent // 32), 127, dtype=np.uint8),
        input_dimension=latent,
        output_dimension=intermediate,
    )
    up = MXFP4Tensor(
        packed=np.flip(gate.packed, axis=0).copy(),
        scales=gate.scales.copy(),
        input_dimension=latent,
        output_dimension=intermediate,
    )
    down = MXFP4Tensor(
        packed=np.arange(latent * intermediate // 2, dtype=np.uint8).reshape(
            latent, intermediate // 2
        ),
        scales=np.full((latent, intermediate // 32), 126, dtype=np.uint8),
        input_dimension=intermediate,
        output_dimension=latent,
    )
    shards = split_native_expert(SimpleNamespace(gate=gate, up=up, down=down), 4)
    assert len(shards) == 4
    assert all(shard[0].output_dimension == 32 for shard in shards)
    assert all(shard[2].input_dimension == 32 for shard in shards)
    assert sum(sum(tensor.byte_size for tensor in shard) for shard in shards) == (
        gate.byte_size + up.byte_size + down.byte_size
    )
    assert np.array_equal(
        np.concatenate([shard[0].packed for shard in shards]), gate.packed
    )
    assert np.array_equal(
        np.concatenate([shard[2].packed for shard in shards], axis=1), down.packed
    )
    with pytest.raises(ValueError, match="MXFP4 groups"):
        split_native_expert(SimpleNamespace(gate=gate, up=up, down=down), 8)


def test_attnres_future_score_cache_is_algebraically_exact() -> None:
    generator = np.random.default_rng(18)
    completed = generator.normal(size=(4, 64)).astype(np.float32)
    prefix = generator.normal(size=64).astype(np.float32)
    query = generator.normal(size=64).astype(np.float32)
    cached_scores = np.asarray(
        [_score(value, query, 1e-6) for value in completed], dtype=np.float32
    )
    reference, reference_scores = reference_mix(
        prefix, completed, query, epsilon=1e-6
    )
    cached, cached_scores_with_prefix = cached_mix(
        prefix,
        completed,
        query,
        cached_scores,
        epsilon=1e-6,
    )
    assert np.array_equal(reference_scores, cached_scores_with_prefix)
    assert np.array_equal(reference, cached)


@pytest.mark.parametrize("logical_count", [32, 128, 512, 1000, 1152])
def test_hierarchical_control_plane_is_not_one_wait_per_task(logical_count: int) -> None:
    tasks = [
        {
            "microcell": index % 12,
            "operation": "expert",
            "shape": (1, 3584),
            "dtype": "mxfp4",
            "expert": index % 16,
            "weight_shard": index % 32,
            "route_bucket": index % 4,
        }
        for index in range(logical_count)
    ]
    result = HierarchicalTaskBatcher(leaf_batch=32).plan(tasks)
    assert int(result["critical_path_waits"]) < logical_count
    assert int(result["serial_scheduling_decisions"]) < logical_count
    assert int(result["physical_kernel_count"]) <= logical_count


def test_compact_natural_control_plan_preserves_logical_accounting() -> None:
    buckets = {
        str(worker): {
            ("expert", (1, 3584), "mxfp4", worker % 4, shard, 0): (
                10_000,
                1_600_000,
            )
            for shard in range(4)
        }
        for worker in range(12)
    }
    result = HierarchicalTaskBatcher(leaf_batch=32).plan_compact(buckets)
    assert result["logical_task_count"] == 480_000
    assert result["metadata_bytes"] == 76_800_000
    assert result["serial_scheduling_decisions"] == 16
    assert result["critical_path_waits"] == 5
    assert result["task_creation"] == 0
