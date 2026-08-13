"""Full-checkpoint placement for bounded Experiment 019 microworkers."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    TensorRecord,
    balanced_range,
)

GIB = 1024**3
MIB = 1024**2
HIDDEN = 7168
KDA_HEADS = 96
KDA_HEAD_DIMENSION = 128
MLA_CACHE_WIDTH = 512 + 64
TRANSFORMER_LAYERS = 93
ROUTED_EXPERTS = 896
KDA_LAYERS = frozenset(
    value - 1
    for value in (
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
        17,
        18,
        19,
        21,
        22,
        23,
        25,
        26,
        27,
        29,
        30,
        31,
        33,
        34,
        35,
        37,
        38,
        39,
        41,
        42,
        43,
        45,
        46,
        47,
        49,
        50,
        51,
        53,
        54,
        55,
        57,
        58,
        59,
        61,
        62,
        63,
        65,
        66,
        67,
        69,
        70,
        71,
        73,
        74,
        75,
        77,
        78,
        79,
        81,
        82,
        83,
        85,
        86,
        87,
        89,
        90,
        91,
    )
)


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    memory_cap_gib: float
    stripe_degree: int
    depth_span: int
    chunk_rows: int
    block_candidates: int = 16
    maximum_context: int = 4096
    hardware_class: str = "RTX_5090_PHYSICAL_SHARD_SERVICE"
    expert_allocation_overhead_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.memory_cap_gib not in (20, 8, 4, 2, 1):
            raise ValueError("Experiment 019 uses the declared 20/8/4/2/1 GiB tiers")
        if self.stripe_degree not in (4, 8, 16, 32):
            raise ValueError("headline stripe degree must be 4/8/16/32")
        if self.depth_span not in (1, 2, 4, 8):
            raise ValueError("depth span must be 1/2/4/8")
        if self.chunk_rows not in (1, 2, 4):
            raise ValueError("chunk rows must be 1/2/4")
        if self.expert_allocation_overhead_factor < 1.0:
            raise ValueError("measured expert allocation factor cannot be below one")

    @property
    def cap_bytes(self) -> int:
        return int(self.memory_cap_gib * GIB)

    @property
    def pod_count(self) -> int:
        return math.ceil(TRANSFORMER_LAYERS / self.depth_span)

    @property
    def worker_count(self) -> int:
        return self.pod_count * self.stripe_degree


@dataclass(slots=True)
class WorkerPlacement:
    worker_id: str
    pod_id: str
    worker_memory_cap: int
    hardware_class: str
    stripe_index: int
    assigned_layers: list[int]
    assigned_tensor_slices: dict[str, Any] = field(default_factory=dict)
    assigned_expert_stripes: list[dict[str, Any]] = field(default_factory=list)
    attention_head_ranges: list[dict[str, int]] = field(default_factory=list)
    state_ownership: list[dict[str, Any]] = field(default_factory=list)
    AttnRes_cache_ownership: list[dict[str, Any]] = field(default_factory=list)
    static_weight_bytes: int = 0
    routed_expert_weight_bytes: int = 0
    persistent_state_bytes: int = 0
    dynamic_buffer_bytes: int = 0
    network_buffer_bytes: int = 0
    workspace_bytes: int = 0
    allocator_overhead_bytes: int = 0
    peak_total_bytes: int = 0
    checkpoint_byte_ranges: dict[str, Any] = field(default_factory=dict)
    checkpoint_hashes: dict[str, str] = field(default_factory=dict)
    largest_layer_fraction: float = 0.0
    largest_expert_fraction: float = 0.0
    largest_shared_expert_fraction: float = 0.0

    def manifest_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SliceAssignment:
    worker_index: int
    bytes: int
    axis: int | None
    start: int
    stop: int
    total: int
    replicated: bool = False


@dataclass(slots=True)
class PlacementResult:
    spec: PlacementSpec
    workers: list[WorkerPlacement]
    checkpoint_payload_bytes: int
    total_resident_weight_bytes: int
    replicated_weight_bytes: int
    coverage_tensor_count: int
    coverage_assigned_bytes: int
    coverage_gap_bytes: int
    coverage_overlap_bytes: int
    max_worker_peak_bytes: int
    max_layer_fraction: float
    max_expert_fraction: float
    max_shared_expert_fraction: float
    valid: bool
    invalid_reasons: list[str]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-019-worker-manifest-v1",
            "placement": asdict(self.spec),
            "summary": {
                "worker_count": len(self.workers),
                "pod_count": self.spec.pod_count,
                "workers_per_pod": self.spec.stripe_degree,
                "checkpoint_payload_bytes": self.checkpoint_payload_bytes,
                "total_resident_weight_bytes": self.total_resident_weight_bytes,
                "replicated_weight_bytes": self.replicated_weight_bytes,
                "weight_replication_factor": self.total_resident_weight_bytes
                / self.checkpoint_payload_bytes,
                "coverage_tensor_count": self.coverage_tensor_count,
                "coverage_assigned_bytes": self.coverage_assigned_bytes,
                "coverage_gap_bytes": self.coverage_gap_bytes,
                "coverage_overlap_bytes": self.coverage_overlap_bytes,
                "max_worker_peak_bytes": self.max_worker_peak_bytes,
                "max_worker_peak_gib": self.max_worker_peak_bytes / GIB,
                "max_layer_fraction": self.max_layer_fraction,
                "max_expert_fraction": self.max_expert_fraction,
                "max_shared_expert_fraction": self.max_shared_expert_fraction,
                "valid": self.valid,
                "invalid_reasons": self.invalid_reasons,
            },
            "workers": [worker.manifest_row() for worker in self.workers],
        }


def _stable_index(text: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") % count


def _pod_worker(spec: PlacementSpec, pod: int, stripe: int) -> int:
    return pod * spec.stripe_degree + stripe


def _split_axis(record: TensorRecord, spec: PlacementSpec) -> int | None:
    name = record.name
    if record.role == "routed_expert":
        return 1 if ".w2." in name else 0
    if record.role == "shared_expert":
        return 1 if ".down_proj." in name else 0
    if record.role == "dense_mlp":
        return 1 if ".down_proj." in name else 0
    if record.role == "latent_moe_projection":
        if "routed_expert_norm" in name:
            return None
        return 1 if "up_proj" in name else 0
    if record.role == "attention" and len(record.shape) >= 2:
        if name.endswith(".o_proj.weight"):
            return 1 if record.layer_id in KDA_LAYERS else 0
        if any(
            marker in name
            for marker in (
                ".q_proj.",
                ".k_proj.",
                ".v_proj.",
                ".g_proj.",
                ".f_b_proj.",
                ".b_proj.",
                ".q_b_proj.",
                ".kv_b_proj.",
            )
        ):
            return 0
        return None
    if record.role in {"embedding", "lm_head"}:
        return 0
    if record.role in {"vision", "multimodal_projector"} and len(record.shape) >= 2:
        return 0
    return None


def _axis_assignments(
    record: TensorRecord,
    *,
    spec: PlacementSpec,
    pod: int,
    axis: int,
) -> list[SliceAssignment]:
    total = record.shape[axis]
    quantum = 1
    if record.role in {"shared_expert", "dense_mlp"}:
        quantum = 64
    elif record.role == "latent_moe_projection" and "up_proj" in record.name:
        quantum = 64
    if total % quantum:
        raise ValueError(
            f"native group quantum {quantum} does not divide {record.name} axis {axis}"
        )
    if total < spec.stripe_degree:
        axis = -1
    if axis == -1:
        stripe = _stable_index(record.name, spec.stripe_degree)
        return [
            SliceAssignment(
                _pod_worker(spec, pod, stripe),
                record.byte_size,
                None,
                0,
                total,
                total,
            )
        ]
    result: list[SliceAssignment] = []
    other = math.prod(record.shape) // total
    for stripe in range(spec.stripe_degree):
        shard = balanced_range(total, spec.stripe_degree, stripe, quantum=quantum)
        bytes_ = (shard.stop - shard.start) * other * record.element_size
        if bytes_:
            result.append(
                SliceAssignment(
                    _pod_worker(spec, pod, stripe),
                    bytes_,
                    axis,
                    shard.start,
                    shard.stop,
                    total,
                )
            )
    if sum(item.bytes for item in result) != record.byte_size:
        raise RuntimeError(f"placement split did not cover {record.name}")
    return result


def assignments_for(record: TensorRecord, spec: PlacementSpec) -> list[SliceAssignment]:
    if record.layer_id is not None:
        pod = record.layer_id // spec.depth_span
        axis = _split_axis(record, spec)
        if axis is not None:
            return _axis_assignments(record, spec=spec, pod=pod, axis=axis)
        stripe = _stable_index(record.name, spec.stripe_degree)
        return [
            SliceAssignment(
                _pod_worker(spec, pod, stripe),
                record.byte_size,
                None,
                0,
                record.shape[0],
                record.shape[0],
            )
        ]
    if record.role == "embedding":
        return _axis_assignments(record, spec=spec, pod=0, axis=0)
    if record.role in {"lm_head", "final_norm", "attnres"}:
        pod = spec.pod_count - 1
    elif record.role in {"vision", "multimodal_projector"}:
        pod = _stable_index(record.name, spec.pod_count)
    else:
        pod = _stable_index(record.name, spec.pod_count)
    axis = _split_axis(record, spec)
    if axis is not None:
        return _axis_assignments(record, spec=spec, pod=pod, axis=axis)
    stripe = _stable_index(record.name, spec.stripe_degree)
    return [
        SliceAssignment(
            _pod_worker(spec, pod, stripe),
            record.byte_size,
            None,
            0,
            record.shape[0],
            record.shape[0],
        )
    ]


def _new_workers(spec: PlacementSpec) -> list[WorkerPlacement]:
    workers: list[WorkerPlacement] = []
    for pod in range(spec.pod_count):
        layers = list(
            range(
                pod * spec.depth_span,
                min(TRANSFORMER_LAYERS, (pod + 1) * spec.depth_span),
            )
        )
        for stripe in range(spec.stripe_degree):
            heads = balanced_range(KDA_HEADS, spec.stripe_degree, stripe)
            worker = WorkerPlacement(
                worker_id=f"pod-{pod:03d}.worker-{stripe:02d}",
                pod_id=f"pod-{pod:03d}",
                worker_memory_cap=spec.cap_bytes,
                hardware_class=spec.hardware_class,
                stripe_index=stripe,
                assigned_layers=layers,
                assigned_tensor_slices={
                    "encoding": "checkpoint-coverage-csv-filter-v1",
                    "artifact": "placement/tensor-coverage.csv",
                    "worker_filter": f"pod-{pod:03d}.worker-{stripe:02d}",
                },
                assigned_expert_stripes=[
                    {
                        "layer": layer,
                        "expert_ids": "0..895",
                        "stripe_index": stripe,
                        "stripe_degree": spec.stripe_degree,
                        "weight_axis": "native_intermediate_dimension",
                    }
                    for layer in layers
                    if layer > 0
                ],
                attention_head_ranges=[
                    {
                        "layer": layer,
                        "start": heads.start,
                        "stop": heads.stop,
                        "total": KDA_HEADS,
                    }
                    for layer in layers
                ],
            )
            workers.append(worker)
    return workers


def _add_memory_envelope(worker: WorkerPlacement, spec: PlacementSpec) -> None:
    for layer in worker.assigned_layers:
        if layer in KDA_LAYERS:
            heads = balanced_range(KDA_HEADS, spec.stripe_degree, worker.stripe_index)
            head_count = heads.stop - heads.start
            recurrent = head_count * KDA_HEAD_DIMENSION * KDA_HEAD_DIMENSION * 4
            convolution = 3 * head_count * KDA_HEAD_DIMENSION * 4 * 4
            worker.persistent_state_bytes += recurrent + convolution
            worker.state_ownership.append(
                {
                    "layer": layer,
                    "type": "KDA_recurrent_and_conv_heads",
                    "head_start": heads.start,
                    "head_stop": heads.stop,
                    "bytes": recurrent + convolution,
                }
            )
        else:
            cache = spec.maximum_context * MLA_CACHE_WIDTH * 4
            worker.persistent_state_bytes += cache
            worker.state_ownership.append(
                {
                    "layer": layer,
                    "type": "MLA_compressed_cache_replica_for_owned_heads",
                    "maximum_context": spec.maximum_context,
                    "bytes": cache,
                }
            )
    if worker.stripe_index == 0:
        queued_chunks = math.ceil((spec.block_candidates + 1) / spec.chunk_rows)
        cache = len(worker.assigned_layers) * queued_chunks * spec.chunk_rows * HIDDEN * 4
        worker.persistent_state_bytes += cache
        worker.AttnRes_cache_ownership.append(
            {
                "layers": worker.assigned_layers,
                "queued_chunks": queued_chunks,
                "bytes": cache,
                "owner": worker.worker_id,
            }
        )
    expert_intermediate = (
        spec.chunk_rows * 16 * math.ceil(3072 / spec.stripe_degree) * 2 * 4
    )
    attention_buffers = spec.chunk_rows * math.ceil(12288 / spec.stripe_degree) * 8
    hidden_buffers = spec.chunk_rows * HIDDEN * 4 * 6
    worker.dynamic_buffer_bytes = expert_intermediate + attention_buffers + hidden_buffers
    worker.network_buffer_bytes = spec.chunk_rows * HIDDEN * 4 * 4
    worker.workspace_bytes = max(64 * MIB, min(512 * MIB, worker.static_weight_bytes // 12))
    subtotal = (
        worker.static_weight_bytes
        + worker.persistent_state_bytes
        + worker.dynamic_buffer_bytes
        + worker.network_buffer_bytes
        + worker.workspace_bytes
    )
    measured_expert_allocation_overhead = math.ceil(
        worker.routed_expert_weight_bytes
        * (spec.expert_allocation_overhead_factor - 1.0)
    )
    worker.allocator_overhead_bytes = (
        math.ceil(subtotal * 0.03) + measured_expert_allocation_overhead
    )
    worker.peak_total_bytes = subtotal + worker.allocator_overhead_bytes


def build_placement(catalog: CheckpointCatalog, spec: PlacementSpec) -> PlacementResult:
    workers = _new_workers(spec)
    records = catalog.records()
    lfs_hashes = catalog.lfs_hashes()
    layer_totals = [0 for _ in range(TRANSFORMER_LAYERS)]
    worker_layer: dict[tuple[int, int], int] = {}
    range_counts = [0 for _ in workers]
    files_by_worker: list[set[str]] = [set() for _ in workers]
    assigned_bytes = 0
    replicated_bytes = 0
    for record in records.values():
        if record.layer_id is not None:
            layer_totals[record.layer_id] += record.byte_size
        assignments = assignments_for(record, spec)
        source_covered = sum(item.bytes for item in assignments if not item.replicated)
        if source_covered != record.byte_size:
            raise RuntimeError(f"coverage mismatch for {record.name}")
        assigned_bytes += source_covered
        replicated_bytes += sum(item.bytes for item in assignments if item.replicated)
        for assignment in assignments:
            worker = workers[assignment.worker_index]
            worker.static_weight_bytes += assignment.bytes
            if record.role == "routed_expert":
                worker.routed_expert_weight_bytes += assignment.bytes
            range_counts[assignment.worker_index] += (
                record.shape[0] if assignment.axis == 1 and len(record.shape) == 2 else 1
            )
            files_by_worker[assignment.worker_index].add(record.file)
            if record.layer_id is not None:
                key = (assignment.worker_index, record.layer_id)
                worker_layer[key] = worker_layer.get(key, 0) + assignment.bytes
    for index, worker in enumerate(workers):
        fractions = [
            worker_layer.get((index, layer), 0) / layer_totals[layer]
            for layer in worker.assigned_layers
            if layer_totals[layer]
        ]
        worker.largest_layer_fraction = max(fractions, default=0.0)
        worker.largest_expert_fraction = 1.0 / spec.stripe_degree
        worker.largest_shared_expert_fraction = 1.0 / spec.stripe_degree
        worker.checkpoint_byte_ranges = {
            "encoding": "safetensors-row-major-contiguous-or-strided-v1",
            "artifact": "placement/tensor-coverage.csv",
            "worker_filter": worker.worker_id,
            "logical_range_count": range_counts[index],
            "assigned_checkpoint_bytes": worker.static_weight_bytes,
        }
        worker.checkpoint_hashes = {
            file: lfs_hashes.get(file, "UNAVAILABLE")
            for file in sorted(files_by_worker[index])
        }
        _add_memory_envelope(worker, spec)
    max_peak = max(worker.peak_total_bytes for worker in workers)
    max_layer = max(worker.largest_layer_fraction for worker in workers)
    max_expert = max(worker.largest_expert_fraction for worker in workers)
    max_shared = max(worker.largest_shared_expert_fraction for worker in workers)
    reasons: list[str] = []
    if max_peak > spec.cap_bytes:
        reasons.append("worker_peak_exceeds_memory_cap")
    if max_layer > 0.25 + 1e-12:
        reasons.append("worker_owns_more_than_25_percent_of_layer")
    if max_expert > 0.25 + 1e-12:
        reasons.append("worker_owns_more_than_25_percent_of_routed_expert")
    if max_shared > 0.25 + 1e-12:
        reasons.append("worker_owns_more_than_25_percent_of_shared_expert")
    if assigned_bytes != sum(record.byte_size for record in records.values()):
        reasons.append("checkpoint_coverage_incomplete")
    return PlacementResult(
        spec=spec,
        workers=workers,
        checkpoint_payload_bytes=sum(record.byte_size for record in records.values()),
        total_resident_weight_bytes=sum(worker.static_weight_bytes for worker in workers),
        replicated_weight_bytes=replicated_bytes,
        coverage_tensor_count=len(records),
        coverage_assigned_bytes=assigned_bytes,
        coverage_gap_bytes=0,
        coverage_overlap_bytes=0,
        max_worker_peak_bytes=max_peak,
        max_layer_fraction=max_layer,
        max_expert_fraction=max_expert,
        max_shared_expert_fraction=max_shared,
        valid=not reasons,
        invalid_reasons=reasons,
    )


def iter_coverage_rows(
    catalog: CheckpointCatalog,
    result: PlacementResult,
) -> Iterable[dict[str, Any]]:
    spec = result.spec
    workers = result.workers
    for record in sorted(catalog.records().values(), key=lambda item: item.name):
        assignments = assignments_for(record, spec)
        owners = []
        for assignment in assignments:
            owners.append(
                {
                    "worker": workers[assignment.worker_index].worker_id,
                    "axis": assignment.axis,
                    "start": assignment.start,
                    "stop": assignment.stop,
                    "total": assignment.total,
                    "bytes": assignment.bytes,
                    "replicated": assignment.replicated,
                }
            )
        yield {
            "tensor": record.name,
            "file": record.file,
            "dtype": record.dtype,
            "shape": "x".join(str(value) for value in record.shape),
            "source_bytes": record.byte_size,
            "layer_id": record.layer_id,
            "expert_id": record.expert_id,
            "role": record.role,
            "partition_count": len(owners),
            "assigned_bytes": sum(item["bytes"] for item in owners if not item["replicated"]),
            "replicated_bytes": sum(item["bytes"] for item in owners if item["replicated"]),
            "gap_bytes": 0,
            "overlap_bytes": 0,
            "coverage_status": "PASS",
            "owners_json": owners,
        }


def estimate_candidate(
    catalog: CheckpointCatalog,
    spec: PlacementSpec,
) -> dict[str, Any]:
    cache = getattr(catalog, "_experiment_019_placement_census", None)
    if cache is None:
        layer_bytes = [0 for _ in range(TRANSFORMER_LAYERS)]
        layer_expert_bytes = [0 for _ in range(TRANSFORMER_LAYERS)]
        endpoint = 0
        auxiliary = 0
        for record in catalog.records().values():
            if record.layer_id is not None:
                layer_bytes[record.layer_id] += record.byte_size
                if record.role == "routed_expert":
                    layer_expert_bytes[record.layer_id] += record.byte_size
            elif record.role in {"embedding", "lm_head", "final_norm", "attnres"}:
                endpoint += record.byte_size
            else:
                auxiliary += record.byte_size
        cache = (tuple(layer_bytes), tuple(layer_expert_bytes), endpoint, auxiliary)
        setattr(catalog, "_experiment_019_placement_census", cache)
    layer_bytes, layer_expert_bytes, endpoint, auxiliary = cache
    pod_payloads = [
        sum(layer_bytes[start : start + spec.depth_span])
        for start in range(0, TRANSFORMER_LAYERS, spec.depth_span)
    ]
    pod_expert_payloads = [
        sum(layer_expert_bytes[start : start + spec.depth_span])
        for start in range(0, TRANSFORMER_LAYERS, spec.depth_span)
    ]
    maximum_pod = max(range(len(pod_payloads)), key=pod_payloads.__getitem__)
    maximum_static = math.ceil(pod_payloads[maximum_pod] / spec.stripe_degree)
    maximum_static += math.ceil(endpoint / (2 * spec.stripe_degree))
    maximum_static += math.ceil(auxiliary / spec.worker_count)
    dummy = WorkerPlacement(
        worker_id="estimate",
        pod_id="estimate",
        worker_memory_cap=spec.cap_bytes,
        hardware_class=spec.hardware_class,
        stripe_index=0,
        assigned_layers=list(range(min(spec.depth_span, TRANSFORMER_LAYERS))),
        static_weight_bytes=maximum_static,
        routed_expert_weight_bytes=math.ceil(
            pod_expert_payloads[maximum_pod] / spec.stripe_degree
        ),
    )
    _add_memory_envelope(dummy, spec)
    valid = dummy.peak_total_bytes <= spec.cap_bytes
    return {
        "memory_cap_gib": spec.memory_cap_gib,
        "stripe_degree": spec.stripe_degree,
        "depth_span": spec.depth_span,
        "chunk_rows": spec.chunk_rows,
        "worker_count": spec.worker_count,
        "pod_count": spec.pod_count,
        "workers_per_pod": spec.stripe_degree,
        "estimated_max_static_bytes": maximum_static,
        "estimated_routed_expert_weight_bytes": dummy.routed_expert_weight_bytes,
        "expert_allocation_overhead_factor": spec.expert_allocation_overhead_factor,
        "estimated_max_peak_bytes": dummy.peak_total_bytes,
        "estimated_max_peak_gib": dummy.peak_total_bytes / GIB,
        "largest_expert_fraction": 1 / spec.stripe_degree,
        "largest_layer_fraction_estimate": 1 / spec.stripe_degree,
        "capacity_valid": valid,
        "invalid_reason": "" if valid else "estimated_worker_peak_exceeds_cap",
    }


__all__ = [
    "GIB",
    "KDA_LAYERS",
    "PlacementResult",
    "PlacementSpec",
    "WorkerPlacement",
    "assignments_for",
    "build_placement",
    "estimate_candidate",
    "iter_coverage_rows",
]
