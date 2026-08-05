"""LEGACY_FROZEN coordinator retained only for Experiment 010 reproduction."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from swarm_inference.experiments.experiment_010.expert import ExpertStore, reduce_partials
from swarm_inference.experiments.experiment_010.schemas import (
    DeterminismMode,
    ExpertExecutionMode,
    ExpertExecutionRequest,
    ReductionMode,
    TransportCodec,
)
from swarm_inference.experiments.experiment_010.transport import ExpertTransportClient


@dataclass(frozen=True, slots=True)
class MicroshardOwner:
    worker_id: str
    layer_id: int
    expert_ids: tuple[int, ...]
    hidden_start: int
    hidden_end: int
    logical_intermediate_dimension: int


@dataclass(frozen=True, slots=True)
class LayerDispatchResult:
    output: np.ndarray
    metrics: dict[str, Any]
    worker_responses: tuple[dict[str, Any], ...]


class StableExpertCoordinator:
    """Keep router/request state authoritative while outsourcing expert math."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        quantization_fingerprint: str,
        latent_dimension: int,
        clients: dict[str, ExpertTransportClient],
        whole_ownership: dict[tuple[int, int], str] | None = None,
        microshard_ownership: list[MicroshardOwner] | None = None,
        local_store: ExpertStore | None = None,
        coordinator_id: str = "coordinator",
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.quantization_fingerprint = quantization_fingerprint
        self.latent_dimension = latent_dimension
        self.clients = dict(clients)
        self.whole_ownership = dict(whole_ownership or {})
        self.microshard_ownership = list(microshard_ownership or [])
        self.local_store = local_store
        self.coordinator_id = coordinator_id

    def _request(
        self,
        *,
        request_id: str,
        layer_id: int,
        batch_rows: int,
        expert_ids: list[int],
        routing_weights: list[float],
        mode: ExpertExecutionMode,
        determinism: DeterminismMode,
        codec: TransportCodec,
        reduction: ReductionMode,
        hidden_start: int | None = None,
        hidden_end: int | None = None,
        deadline_ns: int,
    ) -> ExpertExecutionRequest:
        return ExpertExecutionRequest(
            request_id=request_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            quantization_fingerprint=self.quantization_fingerprint,
            layer_id=layer_id,
            batch_rows=batch_rows,
            latent_dimension=self.latent_dimension,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            activations={},
            deadline_ns=deadline_ns,
            execution_mode=mode,
            determinism_mode=determinism,
            compression=codec,
            hidden_start=hidden_start,
            hidden_end=hidden_end,
            reduction_mode=reduction,
        )

    def execute_whole_layer(
        self,
        activation: np.ndarray,
        *,
        layer_id: int,
        expert_ids: list[int],
        routing_weights: list[float],
        coalesced: bool = True,
        determinism: DeterminismMode | str = DeterminismMode.EXACT,
        codec: TransportCodec | str = TransportCodec.RAW_FP32,
        reduction: ReductionMode | str = ReductionMode.FIXED_ORDER_FP32,
        timeout_s: float = 30.0,
        request_id: str | None = None,
    ) -> LayerDispatchResult:
        if len(expert_ids) != len(routing_weights):
            raise ValueError("expert IDs and routing weights differ in length")
        source = np.ascontiguousarray(activation, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != self.latent_dimension:
            raise ValueError("coordinator activation geometry is invalid")
        selected_determinism = DeterminismMode(determinism)
        selected_codec = TransportCodec(codec)
        selected_reduction = ReductionMode(reduction)
        prefix = request_id or str(uuid.uuid4())
        deadline_ns = time.time_ns() + int(timeout_s * 1e9)
        groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for expert_id, weight in zip(expert_ids, routing_weights, strict=True):
            owner = self.whole_ownership.get((layer_id, expert_id), self.coordinator_id)
            groups[owner].append((expert_id, weight))
        started = time.perf_counter_ns()
        partials: list[tuple[str, np.ndarray]] = []
        responses = []
        messages = 0
        activation_transmissions = 0
        for owner in sorted(groups):
            assignments = groups[owner]
            chunks = [assignments] if coalesced else [[item] for item in assignments]
            owner_output = np.zeros_like(source, dtype=np.float32)
            for chunk_index, chunk in enumerate(chunks):
                ids = [item[0] for item in chunk]
                weights = [item[1] for item in chunk]
                request = self._request(
                    request_id=f"{prefix}:{owner}:{chunk_index}",
                    layer_id=layer_id,
                    batch_rows=source.shape[0],
                    expert_ids=ids,
                    routing_weights=weights,
                    mode=ExpertExecutionMode.WHOLE_EXPERT,
                    determinism=selected_determinism,
                    codec=selected_codec,
                    reduction=selected_reduction,
                    deadline_ns=deadline_ns,
                )
                if owner == self.coordinator_id:
                    if self.local_store is None:
                        raise KeyError(f"no local expert store for layer {layer_id} experts {ids}")
                    output, metadata = self.local_store.execute(request, source)
                    responses.append(
                        {"worker_id": owner, "request_id": request.request_id, **metadata}
                    )
                else:
                    if owner not in self.clients:
                        raise KeyError(f"expert owner {owner!r} has no direct data endpoint")
                    response, output, transport = self.clients[owner].execute(request, source)
                    responses.append(
                        {
                            "worker_id": owner,
                            "request_id": response.request_id,
                            "execution_metadata": response.execution_metadata.model_dump(
                                mode="json"
                            ),
                            "transport": transport,
                        }
                    )
                    activation_transmissions += 1
                owner_output += output
                messages += 1
            partials.append((owner, owner_output))
        reduction_started = time.perf_counter_ns()
        result = reduce_partials(partials, mode=selected_reduction)
        reduction_ns = time.perf_counter_ns() - reduction_started
        return LayerDispatchResult(
            output=result,
            metrics={
                "request_id": prefix,
                "layer_id": layer_id,
                "protocol": "coalesced_per_layer" if coalesced else "naive_per_expert",
                "messages_per_layer": messages,
                "activation_transmissions": activation_transmissions,
                "activation_payload_bytes": int(source.nbytes * activation_transmissions),
                "selected_experts": len(expert_ids),
                "destination_workers": len(groups),
                "reduction_ns": reduction_ns,
                "total_ns": time.perf_counter_ns() - started,
            },
            worker_responses=tuple(responses),
        )

    def execute_microshard_layer(
        self,
        activation: np.ndarray,
        *,
        layer_id: int,
        expert_ids: list[int],
        routing_weights: list[float],
        determinism: DeterminismMode | str = DeterminismMode.EXACT,
        codec: TransportCodec | str = TransportCodec.RAW_FP32,
        reduction: ReductionMode | str = ReductionMode.FIXED_ORDER_FP32,
        timeout_s: float = 30.0,
        request_id: str | None = None,
    ) -> LayerDispatchResult:
        source = np.ascontiguousarray(activation, dtype=np.float32)
        selected_experts = set(expert_ids)
        owners = [
            item
            for item in self.microshard_ownership
            if item.layer_id == layer_id and selected_experts <= set(item.expert_ids)
        ]
        if not owners:
            raise KeyError("no microshard ownership covers the routed expert set")
        ranges = sorted((item.hidden_start, item.hidden_end) for item in owners)
        if ranges[0][0] != 0:
            raise ValueError("microshard ownership does not begin at hidden zero")
        if any(left[1] != right[0] for left, right in pairwise(ranges)):
            raise ValueError("microshard ownership has a gap or overlap")
        logical_widths = {item.logical_intermediate_dimension for item in owners}
        if len(logical_widths) != 1 or ranges[-1][1] != next(iter(logical_widths)):
            raise ValueError("microshard ownership does not cover the logical expert width")
        if len({item.worker_id for item in owners}) != len(owners):
            raise ValueError("one worker may own only one coalesced range per layer")
        selected_determinism = DeterminismMode(determinism)
        selected_codec = TransportCodec(codec)
        selected_reduction = ReductionMode(reduction)
        prefix = request_id or str(uuid.uuid4())
        deadline_ns = time.time_ns() + int(timeout_s * 1e9)
        started = time.perf_counter_ns()
        partials = []
        responses = []
        for owner in sorted(owners, key=lambda item: item.worker_id):
            request = self._request(
                request_id=f"{prefix}:{owner.worker_id}",
                layer_id=layer_id,
                batch_rows=source.shape[0],
                expert_ids=expert_ids,
                routing_weights=routing_weights,
                mode=ExpertExecutionMode.MICROSHARD,
                determinism=selected_determinism,
                codec=selected_codec,
                reduction=selected_reduction,
                hidden_start=owner.hidden_start,
                hidden_end=owner.hidden_end,
                deadline_ns=deadline_ns,
            )
            client = self.clients.get(owner.worker_id)
            if client is None:
                raise KeyError(f"microshard owner {owner.worker_id!r} has no endpoint")
            response, output, transport = client.execute(request, source)
            partials.append((owner.worker_id, output))
            responses.append(
                {
                    "worker_id": owner.worker_id,
                    "request_id": response.request_id,
                    "hidden_start": owner.hidden_start,
                    "hidden_end": owner.hidden_end,
                    "execution_metadata": response.execution_metadata.model_dump(mode="json"),
                    "transport": transport,
                }
            )
        reduction_started = time.perf_counter_ns()
        result = reduce_partials(partials, mode=selected_reduction)
        reduction_ns = time.perf_counter_ns() - reduction_started
        return LayerDispatchResult(
            output=result,
            metrics={
                "request_id": prefix,
                "layer_id": layer_id,
                "protocol": "coalesced_layer_microshards",
                "messages_per_layer": len(owners),
                "activation_transmissions": len(owners),
                "activation_payload_bytes": int(source.nbytes * len(owners)),
                "selected_experts": len(expert_ids),
                "shard_workers": len(owners),
                "shard_ranges": [list(item) for item in ranges],
                "reduction_ns": reduction_ns,
                "total_ns": time.perf_counter_ns() - started,
            },
            worker_responses=tuple(responses),
        )


def compare_layer_results(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(candidate, dtype=np.float64)
    difference = observed - expected
    return {
        "exact": bool(np.array_equal(reference, candidate)),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference.ravel()) / max(float(np.linalg.norm(expected.ravel())), 1e-30)
        ),
    }


def dispatch_result_payload(result: LayerDispatchResult) -> dict[str, Any]:
    return {
        "metrics": result.metrics,
        "worker_responses": list(result.worker_responses),
        "output_shape": list(result.output.shape),
        "output_bytes": int(result.output.nbytes),
        "output_sha256": hashlib.sha256(result.output.tobytes()).hexdigest(),
    }
