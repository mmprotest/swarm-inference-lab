"""Production-native resident whole-expert groups for the E022 completion.

Each assignment owns a disjoint contiguous set of complete routed experts.  A
worker receives the canonical ordered route IDs and weights, zeros weights for
non-owned routes, and executes the existing grouped top-16 CUDA primitive.  Its
full-width contribution can therefore be reduced with the other groups without
changing route order or expert semantics.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import _CudaRuntime
from swarm_inference.experiments.experiment_019.checkpoint import (
    CheckpointCatalog,
    DirectShardLoader,
    balanced_range,
)
from swarm_inference.experiments.experiment_019.physical import (
    LATENT,
    ROUTED_EXPERTS,
    TOPK,
    _ResidentHandles,
)
from swarm_inference.experiments.experiment_020.expert_grouped import (
    GroupedTop16Runtime,
)
from swarm_inference.experiments.experiment_022.native_dispatch import ShardRequest
from swarm_inference.experiments.experiment_022.resident_primitives import (
    _PreparedPrimitive,
    _sha256_arrays,
)
from swarm_inference.model.mxfp4 import MXFP4Tensor

INTERMEDIATE = 3072


def _load_whole_expert(
    loader: DirectShardLoader,
    *,
    layer: int,
    expert: int,
    worker_id: str,
) -> tuple[MXFP4Tensor, MXFP4Tensor, MXFP4Tensor]:
    """Load one explicitly assigned complete expert with byte-range receipts."""

    prefix = (
        f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}"
    )

    def complete(name: str, purpose: str) -> np.ndarray:
        record = loader.catalog.record(name)
        return loader.load(
            name,
            worker_id=worker_id,
            purpose=purpose,
            axis=0,
            start=0,
            stop=record.shape[0],
        )

    def matrix(stem: str, input_dimension: int, output_dimension: int) -> MXFP4Tensor:
        packed = complete(
            f"{prefix}.{stem}.weight_packed",
            "resident_whole_expert_packed",
        )
        scales = complete(
            f"{prefix}.{stem}.weight_scale",
            "resident_whole_expert_scales",
        )
        return MXFP4Tensor(
            packed=packed,
            scales=scales,
            input_dimension=input_dimension,
            output_dimension=output_dimension,
        )

    gate = matrix("w1", LATENT, INTERMEDIATE)
    up = matrix("w3", LATENT, INTERMEDIATE)
    down = matrix("w2", INTERMEDIATE, LATENT)
    return gate, up, down


class PreparedWholeExpertGroup(_PreparedPrimitive):
    """One resident group of complete K3 routed experts using grouped CUDA."""

    native_primitive = "e020_kimi_grouped_top16_whole_expert_contribution"

    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        grouped_library: Path,
        *,
        layer: int,
        degree: int,
        shard_index: int,
        max_rows: int = 4,
        device: int = 0,
        shutdown_runtime_on_close: bool = True,
    ) -> None:
        super().__init__()
        if degree not in (2, 4, 8, 16):
            raise ValueError("whole-expert group degree must be 2/4/8/16")
        started = self._begin_startup()
        self.layer = layer
        self.degree = degree
        self.shard_index = shard_index
        self.max_rows = max_rows
        self.shutdown_runtime_on_close = shutdown_runtime_on_close
        self.catalog = CheckpointCatalog(checkpoint)
        self.loader = DirectShardLoader(self.catalog)
        self.runtime = _CudaRuntime(cuda_library, device)
        self.runtime.set_telemetry("minimal")
        self.runtime.set_fused_gate_up(True)
        self.grouped = GroupedTop16Runtime(grouped_library)
        ownership = balanced_range(ROUTED_EXPERTS, degree, shard_index)
        self.expert_start = ownership.start
        self.expert_stop = ownership.stop
        self.expert_ids = tuple(range(ownership.start, ownership.stop))
        if not self.expert_ids:
            raise ValueError("whole-expert assignment owns no experts")
        self.dummy_expert = self.expert_ids[0]
        worker_id = f"whole-expert.layer-{layer}.p{degree}.worker-{shard_index:02d}"
        handles = {}
        runtime_bytes = 0
        try:
            for expert in self.expert_ids:
                tensors = _load_whole_expert(
                    self.loader,
                    layer=layer,
                    expert=expert,
                    worker_id=worker_id,
                )
                triple = tuple(self.runtime.upload(tensor) for tensor in tensors)
                handles[expert] = triple
                runtime_bytes += sum(
                    self.runtime.tensor_bytes(handle) for handle in triple
                )
        except BaseException:
            for triple in handles.values():
                for handle in triple:
                    self.runtime.release_tensor(handle)
            self.grouped.close()
            self.runtime.close(shutdown=self.shutdown_runtime_on_close)
            raise
        self.resident = _ResidentHandles(self.runtime, handles, runtime_bytes)
        self.input = self.runtime.allocate(max_rows * LATENT * 4)
        self.route_weights = self.runtime.allocate(max_rows * TOPK * 4)
        self.output = self.runtime.allocate(max_rows * LATENT * 4)
        self._finish_startup(
            started,
            self.loader,
            persistent_state_bytes=0,
            runtime_weight_bytes=self.resident.runtime_bytes,
            buffer_bytes=max_rows * (LATENT * 2 + TOPK) * 4,
        )

    def state_fingerprint(self) -> str:
        identity = (
            f"stateless-whole-expert:{self.layer}:{self.degree}:"
            f"{self.shard_index}:{self.expert_start}:{self.expert_stop}"
        )
        return "sha256:" + hashlib.sha256(identity.encode("ascii")).hexdigest()

    def __call__(self, values: np.ndarray, request: ShardRequest) -> np.ndarray:
        source = np.ascontiguousarray(values, dtype=np.float32)
        rows = int(source.shape[0])
        expected = LATENT + 2 * TOPK
        if (
            request.layer != self.layer
            or request.degree != self.degree
            or request.shard_index != self.shard_index
        ):
            raise ValueError("whole-expert request does not match resident assignment")
        if source.shape != (rows, expected) or rows > self.max_rows:
            raise ValueError(
                "whole-expert input must pack latent, ordered route IDs, and weights"
            )
        activation = np.ascontiguousarray(source[:, :LATENT])
        routes = np.rint(source[:, LATENT : LATENT + TOPK]).astype(np.int32)
        weights = np.ascontiguousarray(source[:, LATENT + TOPK :])
        owned = (routes >= self.expert_start) & (routes < self.expert_stop)
        local_routes = np.ascontiguousarray(
            np.where(owned, routes, self.dummy_expert), dtype=np.int32
        )
        local_weights = np.ascontiguousarray(
            np.where(owned, weights, 0.0), dtype=np.float32
        )
        reads_before = len(self.loader.audit)
        started = time.perf_counter_ns()
        copied = time.perf_counter_ns()
        self.runtime.upload_activation(self.input, activation)
        self.runtime.upload_activation(self.route_weights, local_weights)
        input_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        cuda_ms = self.grouped.execute(
            self.resident,
            local_routes,
            self.output,
            self.input,
            self.route_weights,
        )
        copied = time.perf_counter_ns()
        output = self.runtime.download_activation(self.output, (rows, LATENT))
        output_copy_ms = (time.perf_counter_ns() - copied) / 1e6
        self._record(
            started=started,
            cuda_ms=cuda_ms,
            input_copy_ms=input_copy_ms,
            output_copy_ms=output_copy_ms,
            reads_before=reads_before,
            loader=self.loader,
            launches=self.grouped.physical_launches,
            state_mutated=False,
        )
        self.last_execution.update(
            {
                "input_route_ids_sha256": _sha256_arrays((routes,)),
                "input_route_weights_sha256": _sha256_arrays((weights,)),
                "local_route_ids_sha256": _sha256_arrays((local_routes,)),
                "local_route_weights_sha256": _sha256_arrays((local_weights,)),
                "ordered_input_route_ids": routes.tolist(),
                "route_order_preserved": (
                    local_routes.shape == routes.shape
                    and local_weights.shape == weights.shape
                ),
                "owned_route_slots": int(np.count_nonzero(owned)),
                "zero_weight_non_owned_slots": int(np.count_nonzero(~owned)),
                "expert_id_start": self.expert_start,
                "expert_id_stop_exclusive": self.expert_stop,
            }
        )
        return output

    def close(self) -> None:
        self.runtime.free(self.output)
        self.runtime.free(self.route_weights)
        self.runtime.free(self.input)
        self.resident.close()
        self.grouped.close()
        self.runtime.close(shutdown=self.shutdown_runtime_on_close)


__all__ = ["PreparedWholeExpertGroup"]
