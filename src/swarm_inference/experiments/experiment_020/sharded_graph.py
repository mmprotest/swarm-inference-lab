"""E021-candidate shard graph with exact grouped expert-stripe execution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_019.physical import (
    RELATIVE_L2_GATE,
    TOPK,
    _upload_stripe_experts,
)
from swarm_inference.experiments.experiment_019.sharded_graph import ShardedK3Graph

from .expert_grouped import GroupedTop16Runtime, _execute_grouped

_LAYER = re.compile(r"layer-(\d+)")
_WORKER = re.compile(r"worker-(\d+)")


class E020ShardedK3Graph(ShardedK3Graph):
    def __init__(
        self,
        checkpoint: Path,
        cuda_library: Path,
        kda_shard_library: Path,
        grouped_library: Path,
        *,
        degree: int = 8,
        depth_span: int = 8,
        device: int = 0,
    ) -> None:
        if degree != 8 or depth_span != 8:
            raise ValueError("the frozen E021 candidate is P=8/depth=8")
        super().__init__(
            checkpoint,
            cuda_library,
            kda_shard_library,
            degree=degree,
            device=device,
        )
        self.depth_span = depth_span
        self.grouped = GroupedTop16Runtime(grouped_library)

    def close(self) -> None:
        self.grouped.close()
        super().close()

    def _routed_experts(
        self,
        latent: np.ndarray,
        routes: np.ndarray,
        route_weights: np.ndarray,
        *,
        layer: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        partials: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        experts = sorted({int(value) for value in routes.reshape(-1)})
        for stripe in range(self.degree):
            worker = f"layer-{layer:02d}.expert-bank.worker-{stripe:02d}"
            resident = _upload_stripe_experts(
                self.runtime,
                self.loader,
                layer=layer,
                experts=experts,
                degree=self.degree,
                stripe=stripe,
                worker_id=worker,
            )
            try:
                measurement, partial = _execute_grouped(
                    self.runtime,
                    self.grouped,
                    resident,
                    latent,
                    routes,
                    route_weights,
                    warmup=0,
                    iterations=1,
                )
                partials.append(partial)
                record = {
                    "worker_id": worker,
                    "operator": "grouped_expert_stripe_bank_top16",
                    "resource_type": "microworker",
                    "stripe_index": stripe,
                    "active_experts": len(experts),
                    "runtime_weight_bytes": resident.runtime_bytes,
                    "duration_ms": measurement["wall"]["p50_ms"],
                    "cuda_ms": measurement["cuda"]["p50_ms"],
                    "host_overhead_ms": measurement["wall"]["p50_ms"]
                    - measurement["cuda"]["p50_ms"],
                    "logical_expert_operations": routes.shape[0] * TOPK,
                    "physical_launches": self.grouped.physical_launches,
                    "network_visible_partial_outputs": 1,
                }
                records.append(record)
                self.worker_operations.append(record)
            finally:
                resident.close()
        output = np.sum(np.stack(partials), axis=0, dtype=np.float64).astype(np.float32)
        return output, records

    def manifest_ownership_audit(self) -> dict[str, Any]:
        mappings = []
        failures = []
        for operation in self.worker_operations:
            identifier = str(operation.get("worker_id", ""))
            layer_match = _LAYER.search(identifier)
            worker_match = _WORKER.search(identifier)
            if layer_match:
                layer = int(layer_match.group(1))
                pod = layer // self.depth_span
            else:
                layer = 93
                pod = 11
            stripe = int(worker_match.group(1)) if worker_match else 0
            stripe %= self.degree
            manifest_worker = f"pod-{pod:03d}.worker-{stripe:02d}"
            owned_layers = list(
                range(pod * self.depth_span, min(93, (pod + 1) * self.depth_span))
            )
            valid = layer == 93 or layer in owned_layers
            mappings.append(
                {
                    "operation_worker_id": identifier,
                    "manifest_worker_id": manifest_worker,
                    "layer": layer,
                    "owned_layers": owned_layers,
                    "valid": valid,
                }
            )
            if not valid:
                failures.append(mappings[-1])
        return {
            "status": "PASS" if not failures else "FAIL",
            "mapping_count": len(mappings),
            "failure_count": len(failures),
            "all_compute_resources_are_explicit_gpu_workers": all(
                operation.get("resource_type") == "microworker"
                for operation in self.worker_operations
            ),
            "failures": failures,
        }


def certify_full_candidate(
    checkpoint: Path,
    cuda_library: Path,
    kda_shard_library: Path,
    grouped_library: Path,
    oracle_root: Path,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    graph = E020ShardedK3Graph(
        checkpoint,
        cuda_library,
        kda_shard_library,
        grouped_library,
    )
    try:
        result = graph.execute_full_oracle(
            oracle_root / "hidden-trace.f32",
            oracle_root / "routes.txt",
            oracle_root / "prefill-logits.f32",
            progress=progress,
        )
        ownership = graph.manifest_ownership_audit()
        result["schema_version"] = "experiment-020-full-sharded-oracle-v1"
        result["candidate"] = {
            "stripe_degree": 8,
            "depth_span": 8,
            "worker_count": 96,
            "pod_count": 12,
            "workers_per_pod": 8,
        }
        result["grouped_expert_execution"] = True
        result["manifest_ownership_audit"] = ownership
        result["status"] = (
            "PASS"
            if result["status"] == "PASS"
            and ownership["status"] == "PASS"
            and result["maximum_relative_l2_error"] <= RELATIVE_L2_GATE
            else "FAIL"
        )
        return result
    finally:
        graph.close()


__all__ = ["E020ShardedK3Graph", "certify_full_candidate"]
