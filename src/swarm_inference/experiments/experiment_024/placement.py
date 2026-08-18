"""Deterministic corrected E024 commodity placement and reconciliation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_022.io import canonical_sha256

from .commodity_planner import CandidateAdmission, CommodityK3Planner
from .commodity_pool import compute_multiplier, link_definition
from .freeze import (
    COMMODITY_WORKER_MEMORY_BYTES,
    HIDDEN,
    LAYER_ZERO_WHOLE_CANDIDATE_ID,
    P8_REQUIRED_LAYER_IDS,
)
from .geometry import GEOMETRY
from .models import CommodityScenario, StageAArm

MODEL_METADATA_RELATIVE_PATH = Path("artifacts/experiment-022/model-metadata.json")
TRANSIENT_BYTES_PER_ACTIVE_NODE = 4 * HIDDEN * 4


@dataclass(frozen=True, slots=True)
class CommodityNode:
    node_id: str
    node_index: int
    accelerator_memory_bytes: int
    compute_multiplier: float
    reliability: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CommodityNode:
        return cls(
            node_id=str(value["node_id"]),
            node_index=int(value["node_index"]),
            accelerator_memory_bytes=int(value["accelerator_memory_bytes"]),
            compute_multiplier=float(value["compute_multiplier"]),
            reliability=float(value.get("reliability", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class LayerPlacement:
    layer: int
    candidate_id: str
    candidate_type: str
    degree: int
    node_ids: tuple[str, ...]
    resident_memory_bytes: tuple[int, ...]
    checkpoint_bytes: tuple[int, ...]
    coordinator_node_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LayerPlacement:
        return cls(
            layer=int(value["layer"]),
            candidate_id=str(value["candidate_id"]),
            candidate_type=str(value["candidate_type"]),
            degree=int(value["degree"]),
            node_ids=tuple(value["node_ids"]),
            resident_memory_bytes=tuple(
                int(item) for item in value["resident_memory_bytes"]
            ),
            checkpoint_bytes=tuple(int(item) for item in value["checkpoint_bytes"]),
            coordinator_node_id=str(value["coordinator_node_id"]),
        )


@dataclass(frozen=True, slots=True)
class CommodityPlacement:
    scenario: CommodityScenario
    placement_kind: str
    available_node_budget: int
    feasible: bool
    infeasible_reason: str | None
    nodes: tuple[CommodityNode, ...]
    endpoint_memory_by_node: dict[str, int]
    endpoint_checkpoint_bytes_by_node: dict[str, int]
    assignments: tuple[LayerPlacement, ...]
    memory_used_by_node: dict[str, int]
    transient_bytes_by_node: dict[str, int]

    @property
    def active_node_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                node_id
                for node_id, memory in self.memory_used_by_node.items()
                if memory > 0
            )
        )

    @property
    def active_node_count(self) -> int:
        return len(self.active_node_ids)

    @property
    def layer_zero_candidate_id(self) -> str | None:
        return self.assignments[0].candidate_id if self.assignments else None

    @property
    def whole_layer_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            assignment.layer
            for assignment in self.assignments
            if assignment.candidate_type == "WHOLE_LAYER"
        )

    @property
    def p8_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            assignment.layer
            for assignment in self.assignments
            if assignment.degree == 8 and assignment.candidate_type != "WHOLE_LAYER"
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "placement_kind": self.placement_kind,
            "available_node_budget": self.available_node_budget,
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "endpoint_memory_by_node": self.endpoint_memory_by_node,
            "endpoint_checkpoint_bytes_by_node": self.endpoint_checkpoint_bytes_by_node,
            "assignments": [assignment.as_dict() for assignment in self.assignments],
            "memory_used_by_node": self.memory_used_by_node,
            "transient_bytes_by_node": self.transient_bytes_by_node,
        }

    @property
    def placement_sha256(self) -> str:
        return canonical_sha256(self.canonical_payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "experiment-024-commodity-placement-v2",
            **self.canonical_payload(),
            "nodes": [asdict(node) for node in self.nodes],
            "placement_sha256": self.placement_sha256,
            "active_node_ids": list(self.active_node_ids),
            "active_node_count": self.active_node_count,
            "layer_zero_candidate_id": self.layer_zero_candidate_id,
            "whole_layer_layer_ids": list(self.whole_layer_layer_ids),
            "p8_layer_ids": list(self.p8_layer_ids),
            "whole_layer_layer_count": len(self.whole_layer_layer_ids),
            "p8_layer_count": len(self.p8_layer_ids),
            "whole_layer_only_commodity_model_feasible": False,
            "p8_required_whole_layer_incapable_compute_share": 1.0,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CommodityPlacement:
        placement = cls(
            scenario=CommodityScenario(str(value["scenario"])),
            placement_kind=str(value["placement_kind"]),
            available_node_budget=int(value["available_node_budget"]),
            feasible=bool(value["feasible"]),
            infeasible_reason=value.get("infeasible_reason"),
            nodes=tuple(CommodityNode.from_dict(row) for row in value["nodes"]),
            endpoint_memory_by_node={
                str(key): int(item)
                for key, item in value["endpoint_memory_by_node"].items()
            },
            endpoint_checkpoint_bytes_by_node={
                str(key): int(item)
                for key, item in value["endpoint_checkpoint_bytes_by_node"].items()
            },
            assignments=tuple(
                LayerPlacement.from_dict(row) for row in value["assignments"]
            ),
            memory_used_by_node={
                str(key): int(item)
                for key, item in value["memory_used_by_node"].items()
            },
            transient_bytes_by_node={
                str(key): int(item)
                for key, item in value["transient_bytes_by_node"].items()
            },
        )
        if placement.placement_sha256 != value.get(
            "placement_sha256", placement.placement_sha256
        ):
            raise ValueError("serialized commodity placement hash changed")
        if placement.feasible:
            validate_commodity_architecture(placement)
            reconcile_placement_memory(placement)
        return placement

    def as_manifest(self, model_metadata: dict[str, Any]) -> dict[str, Any]:
        pieces: dict[str, list[dict[str, Any]]] = {
            node.node_id: [] for node in self.nodes
        }
        for node_id, memory in self.endpoint_memory_by_node.items():
            pieces[node_id].append(
                {
                    "piece": "common_non_transformer",
                    "partition_type": "IDENTICAL_ENDPOINT_POLICY",
                    "resident_memory_bytes": memory,
                    "checkpoint_bytes": self.endpoint_checkpoint_bytes_by_node[node_id],
                    "cached": False,
                    "network_dependencies": [],
                }
            )
        for assignment in self.assignments:
            for index, node_id in enumerate(assignment.node_ids):
                pieces[node_id].append(
                    {
                        "piece": f"transformer_layer_{assignment.layer:02d}",
                        "candidate_id": assignment.candidate_id,
                        "partition_type": assignment.candidate_type,
                        "degree": assignment.degree,
                        "resident_memory_bytes": assignment.resident_memory_bytes[index],
                        "checkpoint_bytes": assignment.checkpoint_bytes[index],
                        "coordinator": index == 0,
                        "cached": False,
                        "network_dependencies": [
                            other for other in assignment.node_ids if other != node_id
                        ],
                    }
                )
        nodes = [
            {
                "node_id": node.node_id,
                "available_memory_bytes": node.accelerator_memory_bytes,
                "assigned_memory_bytes": self.memory_used_by_node[node.node_id],
                "compute_multiplier": node.compute_multiplier,
                "locality_group": f"commodity-group-{node.node_index % 8:02d}",
                "pieces": pieces[node.node_id],
            }
            for node in self.nodes
        ]
        assigned_checkpoint = sum(self.endpoint_checkpoint_bytes_by_node.values()) + sum(
            sum(assignment.checkpoint_bytes) for assignment in self.assignments
        )
        return {
            "schema_version": "experiment-024-correctness-manifest-v1",
            "inventory_id": (
                f"e024-{self.scenario.value.lower()}-{self.available_node_budget}-"
                f"{self.placement_kind.lower()}"
            ),
            "planner_level": self.placement_kind,
            "feasible": self.feasible,
            "chunk_rows": 1,
            "placement_sha256": self.placement_sha256,
            "architecture": {
                "layer_zero_candidate_id": self.layer_zero_candidate_id,
                "whole_layer_layer_ids": list(self.whole_layer_layer_ids),
                "p8_layer_ids": list(self.p8_layer_ids),
            },
            "checkpoint_reconciliation": {
                "model_checkpoint_bytes": int(model_metadata["checkpoint_payload_bytes"]),
                "assigned_checkpoint_bytes": assigned_checkpoint,
                "gap_bytes": int(model_metadata["checkpoint_payload_bytes"])
                - assigned_checkpoint,
                "overlap_bytes": 0,
            },
            "nodes": nodes,
        }


def commodity_nodes(available_node_budget: int) -> tuple[CommodityNode, ...]:
    if available_node_budget <= 0:
        raise ValueError("commodity node budget must be positive")
    return tuple(
        CommodityNode(
            node_id=f"commodity-{index:03d}",
            node_index=index,
            accelerator_memory_bytes=COMMODITY_WORKER_MEMORY_BYTES,
            compute_multiplier=compute_multiplier(index),
        )
        for index in range(available_node_budget)
    )


def _split_integer(total: int, count: int) -> tuple[int, ...]:
    bounds = [(total * index) // count for index in range(count + 1)]
    return tuple(bounds[index + 1] - bounds[index] for index in range(count))


def _transfer_ms(
    scenario: CommodityScenario,
    source: CommodityNode,
    destination: CommodityNode,
    *,
    payload_bytes: int,
    message_count: int,
) -> float:
    link = link_definition(scenario, source.node_index, destination.node_index)
    transmission_ms = payload_bytes * 8 / (link.bandwidth_gbps * 1_000_000)
    return transmission_ms + message_count * (
        link.latency_ms + link.software_overhead_ms
    )


class CommodityPlacementBuilder:
    """Place endpoint, dense layer 0, then admitted P8 layers in order."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.planner = CommodityK3Planner(self.repo_root)
        self.model_metadata = json.loads(
            (self.repo_root / MODEL_METADATA_RELATIVE_PATH).read_text(encoding="utf-8")
        )

    @staticmethod
    def _remaining(
        node: CommodityNode, used: dict[str, int], active: set[str]
    ) -> int:
        transient = TRANSIENT_BYTES_PER_ACTIVE_NODE if node.node_id in active else 0
        return node.accelerator_memory_bytes - used[node.node_id] - transient

    def _place_candidate(
        self,
        admission: CandidateAdmission,
        *,
        scenario: CommodityScenario,
        arm: StageAArm,
        nodes: tuple[CommodityNode, ...],
        used: dict[str, int],
        active: set[str],
    ) -> tuple[tuple[str, ...], dict[str, int], set[str]] | None:
        if admission.degree != 8:
            raise ValueError("P8 placement received a non-P8 candidate")
        node_by_id = {node.node_id: node for node in nodes}
        geometry = GEOMETRY[arm]
        payload_per_remote = geometry.bytes_per_row * 4 // 7
        messages_per_remote = geometry.messages_per_row // 7
        best: tuple[tuple[float, float, str], tuple[str, ...], dict[str, int], set[str]] | None = None

        for coordinator in nodes:
            coordinator_memory = admission.resident_memory_bytes[0]
            coordinator_extra = (
                TRANSIENT_BYTES_PER_ACTIVE_NODE
                if coordinator.node_id not in active
                else 0
            )
            if used[coordinator.node_id] + coordinator_memory + coordinator_extra > coordinator.accelerator_memory_bytes:
                continue
            trial_used = dict(used)
            trial_active = set(active)
            trial_used[coordinator.node_id] += coordinator_memory
            trial_active.add(coordinator.node_id)
            selected = [coordinator.node_id]
            network_score = 0.0
            compute_score = 2.0 / coordinator.compute_multiplier
            feasible = True
            for memory in admission.resident_memory_bytes[1:]:
                choices: list[tuple[tuple[float, float, float, str], CommodityNode]] = []
                for node in nodes:
                    if node.node_id in selected:
                        continue
                    transient = (
                        TRANSIENT_BYTES_PER_ACTIVE_NODE
                        if node.node_id not in trial_active
                        else 0
                    )
                    projected = trial_used[node.node_id] + memory + transient
                    if projected > node.accelerator_memory_bytes:
                        continue
                    transfer = _transfer_ms(
                        scenario,
                        coordinator,
                        node,
                        payload_bytes=payload_per_remote,
                        message_count=messages_per_remote,
                    )
                    utilization = projected / node.accelerator_memory_bytes
                    score = (
                        transfer + 1.0 / node.compute_multiplier + 12.0 * utilization,
                        utilization,
                        -node.compute_multiplier,
                        node.node_id,
                    )
                    choices.append((score, node))
                if not choices:
                    feasible = False
                    break
                score, chosen = min(choices, key=lambda item: item[0])
                selected.append(chosen.node_id)
                trial_used[chosen.node_id] += memory
                trial_active.add(chosen.node_id)
                network_score = max(network_score, score[0])
                compute_score = max(compute_score, 1.0 / chosen.compute_multiplier)
            if not feasible:
                continue
            maximum_utilization = max(
                (
                    trial_used[node_id]
                    + TRANSIENT_BYTES_PER_ACTIVE_NODE
                )
                / node_by_id[node_id].accelerator_memory_bytes
                for node_id in selected
            )
            candidate_score = (
                network_score + compute_score,
                maximum_utilization,
                "|".join(selected),
            )
            value = (candidate_score, tuple(selected), trial_used, trial_active)
            if best is None or value[0] < best[0]:
                best = value
        if best is None:
            return None
        return best[1], best[2], best[3]

    def build(
        self,
        *,
        scenario: CommodityScenario,
        available_node_budget: int,
        placement_kind: str,
    ) -> CommodityPlacement:
        if placement_kind not in {"CURRENT_PLACEMENT", "D_PLACEMENT"}:
            raise ValueError("unknown E024 placement kind")
        arm = (
            StageAArm.A_CURRENT
            if placement_kind == "CURRENT_PLACEMENT"
            else StageAArm.D_FUSE_OUTPUT
        )
        nodes = commodity_nodes(available_node_budget)
        used = {node.node_id: 0 for node in nodes}
        active: set[str] = set()

        endpoint_nodes = sorted(
            nodes,
            key=lambda node: (-node.compute_multiplier, node.node_id),
        )[: min(4, len(nodes))]
        endpoint_memory_parts = _split_integer(
            int(self.model_metadata["endpoint_resident_bytes"]), len(endpoint_nodes)
        )
        endpoint_checkpoint_parts = _split_integer(
            int(self.model_metadata["endpoint_checkpoint_bytes"]), len(endpoint_nodes)
        )
        endpoint_memory: dict[str, int] = {}
        endpoint_checkpoint: dict[str, int] = {}
        for node, memory, checkpoint in zip(
            endpoint_nodes,
            endpoint_memory_parts,
            endpoint_checkpoint_parts,
            strict=True,
        ):
            if memory + TRANSIENT_BYTES_PER_ACTIVE_NODE > node.accelerator_memory_bytes:
                return self._infeasible(
                    scenario,
                    placement_kind,
                    available_node_budget,
                    nodes,
                    "endpoint state does not fit",
                )
            used[node.node_id] += memory
            active.add(node.node_id)
            endpoint_memory[node.node_id] = memory
            endpoint_checkpoint[node.node_id] = checkpoint

        layer_zero = self.planner.choose_candidate(0)
        if layer_zero.status != "PASS" or layer_zero.candidate_id != LAYER_ZERO_WHOLE_CANDIDATE_ID:
            return self._infeasible(
                scenario,
                placement_kind,
                available_node_budget,
                nodes,
                "layer 0 exact whole candidate unavailable",
                endpoint_memory,
                endpoint_checkpoint,
                used,
                active,
            )
        layer_zero_memory = layer_zero.resident_memory_bytes[0]
        layer_zero_candidates = []
        for node in nodes:
            transient = (
                TRANSIENT_BYTES_PER_ACTIVE_NODE if node.node_id not in active else 0
            )
            projected = used[node.node_id] + layer_zero_memory + transient
            if projected <= node.accelerator_memory_bytes:
                layer_zero_candidates.append(
                    (
                        -node.compute_multiplier,
                        projected / node.accelerator_memory_bytes,
                        node.node_id,
                        node,
                    )
                )
        if not layer_zero_candidates:
            return self._infeasible(
                scenario,
                placement_kind,
                available_node_budget,
                nodes,
                "dense layer 0 does not fit after endpoint reservation",
                endpoint_memory,
                endpoint_checkpoint,
                used,
                active,
            )
        layer_zero_node = min(layer_zero_candidates, key=lambda value: value[:3])[3]
        used[layer_zero_node.node_id] += layer_zero_memory
        active.add(layer_zero_node.node_id)
        assignments = [
            LayerPlacement(
                layer=0,
                candidate_id=layer_zero.candidate_id,
                candidate_type="WHOLE_LAYER",
                degree=1,
                node_ids=(layer_zero_node.node_id,),
                resident_memory_bytes=layer_zero.resident_memory_bytes,
                checkpoint_bytes=layer_zero.checkpoint_bytes,
                coordinator_node_id=layer_zero_node.node_id,
            )
        ]

        for layer in P8_REQUIRED_LAYER_IDS:
            placed = None
            selected_admission = None
            for admission in self.planner.admitted_candidates(layer):
                placed = self._place_candidate(
                    admission,
                    scenario=scenario,
                    arm=arm,
                    nodes=nodes,
                    used=used,
                    active=active,
                )
                if placed is not None:
                    selected_admission = admission
                    break
            if placed is None or selected_admission is None:
                return self._infeasible(
                    scenario,
                    placement_kind,
                    available_node_budget,
                    nodes,
                    f"no memory-feasible admitted P8 placement for layer {layer}",
                    endpoint_memory,
                    endpoint_checkpoint,
                    used,
                    active,
                    assignments,
                )
            node_ids, used, active = placed
            assignments.append(
                LayerPlacement(
                    layer=layer,
                    candidate_id=selected_admission.candidate_id or "",
                    candidate_type=selected_admission.candidate_type or "",
                    degree=selected_admission.degree or 0,
                    node_ids=node_ids,
                    resident_memory_bytes=selected_admission.resident_memory_bytes,
                    checkpoint_bytes=selected_admission.checkpoint_bytes,
                    coordinator_node_id=node_ids[0],
                )
            )

        transient = {
            node.node_id: (
                TRANSIENT_BYTES_PER_ACTIVE_NODE if node.node_id in active else 0
            )
            for node in nodes
        }
        placement = CommodityPlacement(
            scenario=scenario,
            placement_kind=placement_kind,
            available_node_budget=available_node_budget,
            feasible=True,
            infeasible_reason=None,
            nodes=nodes,
            endpoint_memory_by_node=endpoint_memory,
            endpoint_checkpoint_bytes_by_node=endpoint_checkpoint,
            assignments=tuple(assignments),
            memory_used_by_node=used,
            transient_bytes_by_node=transient,
        )
        validate_commodity_architecture(placement)
        reconcile_placement_memory(placement)
        return placement

    @staticmethod
    def _infeasible(
        scenario: CommodityScenario,
        placement_kind: str,
        available_node_budget: int,
        nodes: tuple[CommodityNode, ...],
        reason: str,
        endpoint_memory: dict[str, int] | None = None,
        endpoint_checkpoint: dict[str, int] | None = None,
        used: dict[str, int] | None = None,
        active: set[str] | None = None,
        assignments: list[LayerPlacement] | None = None,
    ) -> CommodityPlacement:
        del active
        memory = used or {node.node_id: 0 for node in nodes}
        transient = {
            node.node_id: (
                TRANSIENT_BYTES_PER_ACTIVE_NODE if memory[node.node_id] else 0
            )
            for node in nodes
        }
        return CommodityPlacement(
            scenario=scenario,
            placement_kind=placement_kind,
            available_node_budget=available_node_budget,
            feasible=False,
            infeasible_reason=reason,
            nodes=nodes,
            endpoint_memory_by_node=endpoint_memory or {},
            endpoint_checkpoint_bytes_by_node=endpoint_checkpoint or {},
            assignments=tuple(assignments or ()),
            memory_used_by_node=memory,
            transient_bytes_by_node=transient,
        )


def validate_commodity_architecture(placement: CommodityPlacement) -> None:
    if not placement.feasible:
        return
    if placement.whole_layer_layer_ids != (0,):
        raise RuntimeError("commodity placement must execute only layer 0 whole")
    if placement.p8_layer_ids != P8_REQUIRED_LAYER_IDS:
        raise RuntimeError("commodity placement must execute layers 1..92 as P8")
    if placement.layer_zero_candidate_id != LAYER_ZERO_WHOLE_CANDIDATE_ID:
        raise RuntimeError("commodity placement changed the exact layer-0 candidate")


def reconcile_placement_memory(placement: CommodityPlacement) -> list[dict[str, Any]]:
    rows = []
    for node in placement.nodes:
        endpoint_bytes = placement.endpoint_memory_by_node.get(node.node_id, 0)
        layer_zero_bytes = sum(
            assignment.resident_memory_bytes[index]
            for assignment in placement.assignments
            for index, assigned_node in enumerate(assignment.node_ids)
            if assigned_node == node.node_id and assignment.layer == 0
        )
        p8_fragment_bytes = sum(
            assignment.resident_memory_bytes[index]
            for assignment in placement.assignments
            for index, assigned_node in enumerate(assignment.node_ids)
            if assigned_node == node.node_id and assignment.layer in P8_REQUIRED_LAYER_IDS
        )
        transient_bytes = placement.transient_bytes_by_node[node.node_id]
        resident = endpoint_bytes + layer_zero_bytes + p8_fragment_bytes
        reconciles = resident == placement.memory_used_by_node[node.node_id]
        within_memory = resident + transient_bytes <= node.accelerator_memory_bytes
        rows.append(
            {
                "scenario": placement.scenario.value,
                "placement_kind": placement.placement_kind,
                "available_node_budget": placement.available_node_budget,
                "node_id": node.node_id,
                "endpoint_bytes": endpoint_bytes,
                "layer_zero_whole_bytes": layer_zero_bytes,
                "p8_fragment_bytes": p8_fragment_bytes,
                "resident_bytes": resident,
                "transient_bytes": transient_bytes,
                "total_resident_plus_transient_bytes": resident + transient_bytes,
                "commodity_memory_bytes": node.accelerator_memory_bytes,
                "reconciles": reconciles,
                "within_memory": within_memory,
            }
        )
        if not reconciles or not within_memory:
            raise RuntimeError(f"memory reconciliation failed for {node.node_id}")
    return rows


__all__ = [
    "TRANSIENT_BYTES_PER_ACTIVE_NODE",
    "CommodityNode",
    "CommodityPlacement",
    "CommodityPlacementBuilder",
    "LayerPlacement",
    "commodity_nodes",
    "reconcile_placement_memory",
    "validate_commodity_architecture",
]
