from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    TensorRecord,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.events import (
    DeterministicMicroworkerEngine,
    MicroworkerTask,
    NetworkProfile,
    assert_headline_trace,
    tree_collective,
)
from swarm_inference.experiments.experiment_019.placement import (
    PlacementSpec,
    assignments_for,
)
from swarm_inference.experiments.experiment_019.simulation import (
    MeasuredShardService,
    SimulationConfiguration,
    simulate,
)


def _write_safetensor(path: Path, name: str, array: np.ndarray) -> None:
    encoded = json.dumps(
        {
            name: {
                "dtype": "U8",
                "shape": list(array.shape),
                "data_offsets": [0, array.nbytes],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + array.tobytes())


def test_direct_loader_reads_only_assigned_rows_and_columns(tmp_path: Path) -> None:
    name = "language_model.model.layers.1.self_attn.q_proj.weight"
    source = np.arange(48, dtype=np.uint8).reshape(6, 8)
    shard = tmp_path / "model.safetensors"
    _write_safetensor(shard, name, source)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": source.nbytes},
                "weight_map": {name: shard.name},
            }
        ),
        encoding="utf-8",
    )
    catalog = CheckpointCatalog(tmp_path)
    loader = DirectShardLoader(catalog)
    rows = loader.load(
        name,
        worker_id="pod-000.worker-00",
        purpose="test_rows",
        axis=0,
        start=1,
        stop=3,
    )
    columns = loader.load(
        name,
        worker_id="pod-000.worker-01",
        purpose="test_columns",
        axis=1,
        start=2,
        stop=5,
    )
    assert np.array_equal(rows, source[1:3])
    assert np.array_equal(columns, source[:, 2:5])
    assert loader.audit[0]["full_source_tensor_materialized"] is False
    assert loader.audit[1]["byte_ranges"]["kind"] == "strided"
    with pytest.raises(ValueError, match="full tensor load forbidden"):
        loader.load(name, worker_id="bad", purpose="forbidden")


def test_native_group_partition_is_complete_and_disjoint() -> None:
    ranges = [balanced_range(3072, 32, index, quantum=32) for index in range(32)]
    assert ranges[0].start == 0
    assert ranges[-1].stop == 3072
    assert all(left.stop == right.start for left, right in zip(ranges, ranges[1:]))
    assert {item.stop - item.start for item in ranges} == {96}


def test_routed_expert_assignment_never_exceeds_quarter() -> None:
    record = TensorRecord(
        name=(
            "language_model.model.layers.89.block_sparse_moe.experts.885."
            "w1.weight_packed"
        ),
        file="model.safetensors",
        dtype="U8",
        shape=(3072, 1792),
        data_offset=4096,
        byte_size=3072 * 1792,
        layer_id=89,
        expert_id=885,
        role="routed_expert",
    )
    spec = PlacementSpec(20, 4, 4, 2)
    assignments = assignments_for(record, spec)
    assert len(assignments) == 4
    assert sum(item.bytes for item in assignments) == record.byte_size
    assert max(item.bytes for item in assignments) / record.byte_size == pytest.approx(0.25)


def _compute(task_id: str, worker: str, dependencies: tuple[str, ...]) -> MicroworkerTask:
    return MicroworkerTask(
        task_id=task_id,
        worker_id=worker,
        pod_id="pod-000",
        request_id="request-0",
        block_id="block-16",
        chunk_id=0,
        layer_id=89,
        operator="expert_stripe",
        shard_id=worker,
        dependency_ids=dependencies,
        input_refs=("latent:0",),
        state_refs=("state:89",),
        service_time_ms=2.0,
    )


def test_event_engine_schedules_only_concrete_microworkers() -> None:
    workers = tuple(f"pod-000.worker-{index:02d}" for index in range(4))
    tasks = [_compute(f"compute-{index}", worker, ()) for index, worker in enumerate(workers)]
    tasks.append(
        tree_collective(
            task_id="reduce",
            participants=workers,
            payload_bytes=14336,
            dependency_ids=tuple(task.task_id for task in tasks),
            profile=NetworkProfile("canonical_local", 0.25, 25.0, 0.02),
            request_id="request-0",
            block_id="block-16",
            chunk_id=0,
            layer_id=89,
            operator="expert_partial_allreduce",
        )
    )
    tasks.append(_compute("after", workers[0], ("reduce",)))
    run = DeterministicMicroworkerEngine().run(tasks)
    assert_headline_trace(run.records)
    assert run.makespan_ms > 4.0
    assert run.total_compute_work_ms == pytest.approx(10.0)
    assert all(
        record.resource_type in {"microworker", "network"} for record in run.records
    )
    reduce = next(record for record in run.records if record.task_id == "reduce")
    assert reduce.collective_participants == workers
    assert all(edge["source_worker_id"] for edge in reduce.communication_edges)
    assert reduce.payload_bytes == 2 * (len(workers) - 1) * 14336


def test_same_worker_cannot_execute_two_tasks_concurrently() -> None:
    worker = "pod-000.worker-00"
    run = DeterministicMicroworkerEngine().run(
        [_compute("a", worker, ()), _compute("b", worker, ())]
    )
    records = {record.task_id: record for record in run.records}
    assert records["b"].start_time >= records["a"].finish_time


def test_full_worker_dag_has_no_aggregate_compute_resource() -> None:
    degree = 4
    service = MeasuredShardService(
        degree=degree,
        rows=4,
        kda_common_ms=0.1,
        kda_worker_ms=(0.2,) * degree,
        mla_common_ms=0.1,
        mla_worker_ms=(0.2,) * degree,
        moe_pre_ms=0.1,
        latent_down_worker_ms=(0.1,) * degree,
        expert_worker_ms=(0.3,) * degree,
        routed_norm_ms=0.01,
        latent_up_shared_worker_ms=(0.2,) * degree,
        dense_worker_ms=(0.2,) * degree,
        layer_finalize_ms=0.01,
        endpoint_worker_ms=(0.2,) * degree,
        embedding_ms=0.01,
        hidden_pair_reduce_ms=0.01,
        latent_pair_reduce_ms=0.01,
        source="synthetic unit-test service",
    )
    local = NetworkProfile("unit-local", 0.25, 25.0, 0.02)
    interpod = NetworkProfile("unit-interpod", 5.0, 10.0, 0.02)
    result, run = simulate(
        service,
        SimulationConfiguration(
            placement=PlacementSpec(20, degree, 8, 4),
            block=7,
            chunk=4,
            local_profile=local,
            inter_pod_profile=interpod,
        ),
    )
    assert result["all_compute_resources_are_microworkers"] is True
    assert all(
        row.resource_type == "microworker"
        for row in run.records
        if row.resource_type != "network"
    )
    assert not hasattr(PlacementSpec(20, degree, 8, 4), "service_time_ms")
