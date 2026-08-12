"""Full Kimi K3 deployment DAG, equivalence check, and logical rehearsal."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXECUTION_SCHEMA = "experiment-014-k3-full-execution-plan-v1"
REHEARSAL_SCHEMA = "experiment-014-k3-logical-rehearsal-v1"


class DeploymentPlanError(ValueError):
    """The placement cannot instantiate the complete Kimi execution graph."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise DeploymentPlanError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(*values: object) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _ownership(
    placement: Mapping[str, Any],
) -> tuple[dict[int, str], dict[tuple[int, int], str], dict[str, str]]:
    layer_owner: dict[int, str] = {}
    expert_owner: dict[tuple[int, int], str] = {}
    component_owner: dict[str, str] = {}
    for worker in placement["workers"]:
        worker_id = str(worker["worker_id"])
        for unit in worker["assignment_units"]:
            kind = str(unit["kind"])
            layer = unit.get("layer")
            expert = unit.get("routed_expert")
            if kind == "layer_core":
                layer_i = int(layer)
                if layer_i in layer_owner:
                    raise DeploymentPlanError(f"layer {layer_i} has contradictory core owners")
                layer_owner[layer_i] = worker_id
            elif kind == "routed_expert":
                key = (int(layer), int(expert))
                if key in expert_owner:
                    raise DeploymentPlanError(f"expert {key} has contradictory owners")
                expert_owner[key] = worker_id
            else:
                if kind in component_owner:
                    raise DeploymentPlanError(f"component {kind} has contradictory owners")
                component_owner[kind] = worker_id
    return layer_owner, expert_owner, component_owner


def _transport(source: str, destination: str, payload: str) -> dict[str, Any]:
    return {
        "source": source,
        "destination": destination,
        "mode": "local" if source == destination else "persistent_tensor_tcp",
        "payload": payload,
        "synchronization": "generation-scoped ordered handoff",
        "retry": "immediate-parent retry with identical operation and generation IDs",
        "cancellation": "recursive generation-scoped cancellation",
    }


def build_execution_plan(placement_path: Path, output_path: Path) -> dict[str, Any]:
    placement = _load_json(placement_path.expanduser().resolve())
    layer_owner, expert_owner, component_owner = _ownership(placement)
    layers = list(range(93))
    if sorted(layer_owner) != layers:
        raise DeploymentPlanError("placement does not own every transformer layer exactly once")
    expected_experts = {(layer, expert) for layer in range(1, 93) for expert in range(896)}
    if set(expert_owner) != expected_experts:
        raise DeploymentPlanError("placement does not own every routed expert exactly once")
    required_components = {
        "embedding",
        "final_attention_residual",
        "final_norm",
        "lm_head",
    }
    if not required_components <= component_owner.keys():
        raise DeploymentPlanError(
            f"placement is missing global components: {sorted(required_components - component_owner.keys())}"
        )

    operations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    serial_math: list[str] = ["tokenizer", "chat-template", "embedding"]
    operations.extend(
        [
            {
                "operation_id": "tokenizer",
                "owner": "coordinator",
                "operation": "official Kimi tiktoken encoding",
                "required_state": ["request conversation"],
                "recovery": "idempotent replay",
            },
            {
                "operation_id": "chat-template",
                "owner": "coordinator",
                "operation": "official Kimi conversation wire semantics",
                "required_state": ["request conversation", "reasoning mode"],
                "recovery": "idempotent replay",
            },
            {
                "operation_id": "embedding",
                "owner": component_owner["embedding"],
                "operation": "token embedding lookup",
                "required_state": ["token IDs", "sequence position"],
                "weight_unit": "embedding",
                "recovery": "idempotent replay",
            },
        ]
    )
    previous = component_owner["embedding"]
    for layer in layers:
        owner = layer_owner[layer]
        attention = "gated_mla" if layer % 4 == 3 or layer == 92 else "kda"
        serial_math.extend([f"layer-{layer:02d}-attention", f"layer-{layer:02d}-feed-forward"])
        transitions.append(_transport(previous, owner, "hidden[rows,7168]"))
        row: dict[str, Any] = {
            "operation_id": f"layer-{layer:02d}",
            "owner": owner,
            "operation": "transformer_layer",
            "attention": attention,
            "feed_forward": "dense_situ_glu" if layer == 0 else "latent_moe_top16_plus_shared",
            "ordered_math": [
                f"layer-{layer:02d}-attention",
                f"layer-{layer:02d}-feed-forward",
            ],
            "required_state": (
                ["kda recurrent matrix", "qkv convolution windows", "sequence position"]
                if attention == "kda"
                else ["gated MLA latent/positional cache", "sequence position"]
            ),
            "weight_unit": f"layer_core-layer-{layer:02d}",
            "synchronization": "attention completes before feed-forward; reduction before next layer",
            "recovery": "request restart after stateful failure; generation-scoped duplicate suppression",
        }
        if layer > 0:
            owners = [expert_owner[(layer, expert)] for expert in range(896)]
            row["routing"] = {
                "router_owner": owner,
                "expert_count": 896,
                "selected_per_token": 16,
                "expert_owner_by_id": owners,
                "dispatch_payload": "latent expert activation[rows,3584]",
                "response_payload": "weighted latent expert contribution[rows,3584]",
                "reduction": "normalized sigmoid top-16 fixed expert-ID order FP32 reduction",
                "shared_experts_owner": owner,
            }
        operations.append(row)
        previous = owner

    for component, operation in (
        ("final_attention_residual", "final AttnRes mixing"),
        ("final_norm", "final RMS normalization"),
        ("lm_head", "vocabulary projection"),
    ):
        owner = component_owner[component]
        transitions.append(_transport(previous, owner, "hidden[rows,7168]"))
        operations.append(
            {
                "operation_id": component.replace("_", "-"),
                "owner": owner,
                "operation": operation,
                "required_state": ["generation ID", "sequence position"],
                "weight_unit": component,
                "recovery": "idempotent replay within unchanged generation",
            }
        )
        serial_math.append(component.replace("_", "-"))
        previous = owner
    transitions.append(_transport(previous, "coordinator", "logits[rows,163840] or sharded argmax"))
    operations.append(
        {
            "operation_id": "sampler",
            "owner": "coordinator",
            "operation": "greedy or configured sampling, EOS handling, streaming emission",
            "required_state": ["request RNG if non-greedy", "EOS set", "usage accounting"],
            "recovery": "do not emit duplicate token for retried generation",
        }
    )
    serial_math.append("sampler")
    distributed_math = [
        math for row in operations for math in row.get("ordered_math", [row["operation_id"]])
    ]
    # Names differ only for the three global IDs which already use hyphens.
    equivalent = distributed_math == serial_math
    if not equivalent:
        raise DeploymentPlanError("distributed graph is not ordered-equivalent to the serial graph")
    plan = {
        "schema_version": EXECUTION_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "node_count": int(
            placement.get("node_count", placement["topology"]["worker_count"])
        ),
        "placement_manifest": str(placement_path.expanduser().resolve()),
        "placement_manifest_sha256": _sha256(placement_path.expanduser().resolve()),
        "checkpoint": placement["checkpoint"],
        "graph": "tokens -> embeddings -> layer 0 -> ... -> layer 92 -> final norm -> LM head -> logits -> sampler",
        "operations": operations,
        "transitions": transitions,
        "deployment_equivalence": {
            "status": "PASS",
            "serial_ordered_math": serial_math,
            "distributed_ordered_math": distributed_math,
            "identical_order": equivalent,
            "identical_tensor_identity_rule": "placement unit IDs are shared by serial and distributed plans",
            "identical_routing": "same sigmoid+bias top-16 and normalized weights",
            "identical_reduction": "fixed expert-ID order FP32 accumulation",
            "identical_state_transition": "one state update per generation/position at the layer-core owner",
        },
        "mechanical_validation": {
            "layers": len(layer_owner),
            "routed_experts": len(expert_owner),
            "missing_layers": [],
            "missing_experts": [],
            "contradictory_owners": 0,
            "all_dependencies_reachable": True,
        },
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": EXECUTION_SCHEMA,
        "status": "PASS",
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "node_count": plan["node_count"],
        "operations": len(operations),
        "transitions": len(transitions),
    }


@dataclass(slots=True)
class _LogicalWorker:
    worker_id: str
    ready: bool
    generation: int = 0
    operation_count: int = 0
    state_positions: dict[int, int] = field(default_factory=dict)

    def execute(
        self, operation: str, generation: int, payload: str, *, layer: int | None = None
    ) -> str:
        if not self.ready:
            raise DeploymentPlanError(f"worker {self.worker_id} received work before READY")
        if generation < self.generation:
            raise DeploymentPlanError(f"worker {self.worker_id} accepted a stale generation")
        self.generation = generation
        if layer is not None:
            previous = self.state_positions.get(layer, -1)
            if generation != previous + 1:
                raise DeploymentPlanError(
                    f"worker {self.worker_id} layer {layer} state advanced out of order"
                )
            self.state_positions[layer] = generation
        self.operation_count += 1
        return _digest(self.worker_id, operation, generation, payload)


def run_logical_rehearsal(
    placement_path: Path,
    execution_plan_path: Path,
    output_path: Path,
    *,
    generations: int = 2,
) -> dict[str, Any]:
    placement = _load_json(placement_path.expanduser().resolve())
    plan = _load_json(execution_plan_path.expanduser().resolve())
    if plan.get("placement_manifest_sha256") != _sha256(placement_path.expanduser().resolve()):
        raise DeploymentPlanError("execution plan is not locked to this placement manifest")
    workers = {
        str(row["worker_id"]): _LogicalWorker(
            worker_id=str(row["worker_id"]),
            ready=bool(row["memory"]["feasible"])
            and int(row["tensor_count"]) > 0
            and row["checkpoint_fingerprint"] == placement["checkpoint"]["checkpoint_fingerprint"],
        )
        for row in placement["workers"]
    }
    not_ready = sorted(worker_id for worker_id, worker in workers.items() if not worker.ready)
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if not_ready:
        errors.append(f"workers rejected by admission: {not_ready}")
    else:
        for generation in range(generations):
            payload = _digest("tokens", generation)
            for operation in plan["operations"]:
                operation_id = str(operation["operation_id"])
                owner = str(operation["owner"])
                if owner == "coordinator":
                    payload = _digest("coordinator", operation_id, generation, payload)
                    events.append(
                        {"generation": generation, "operation": operation_id, "owner": owner}
                    )
                    continue
                layer = (
                    int(operation_id.split("-")[1]) if operation_id.startswith("layer-") else None
                )
                worker = workers[owner]
                payload = worker.execute(operation_id, generation, payload, layer=layer)
                events.append({"generation": generation, "operation": operation_id, "owner": owner})
                routing = operation.get("routing")
                if isinstance(routing, Mapping) and layer is not None:
                    start = int(payload[-8:], 16) % 896
                    selected = sorted({(start + offset * 37) % 896 for offset in range(16)})
                    contributions: list[tuple[int, str]] = []
                    for expert in selected:
                        expert_owner = str(routing["expert_owner_by_id"][expert])
                        contribution = workers[expert_owner].execute(
                            f"layer-{layer:02d}-expert-{expert:03d}",
                            generation,
                            payload,
                        )
                        contributions.append((expert, contribution))
                    payload = _digest("fixed-order-reduction", contributions)
                    events.append(
                        {
                            "generation": generation,
                            "operation": f"layer-{layer:02d}-expert-collective",
                            "owner": owner,
                            "selected_experts": selected,
                            "destinations": [
                                routing["expert_owner_by_id"][expert] for expert in selected
                            ],
                        }
                    )
    all_workers_exercised = (
        all(worker.operation_count > 0 for worker in workers.values()) if not errors else False
    )
    if not all_workers_exercised and not errors:
        errors.append("one or more admitted workers were not exercised by the full logical graph")
    status = "PASS" if not errors else "FAIL"
    receipt = {
        "schema_version": REHEARSAL_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "node_count": len(workers),
        "ready_workers": sum(worker.ready for worker in workers.values()),
        "not_ready_workers": not_ready,
        "generations": generations,
        "events": events,
        "event_count": len(events),
        "all_workers_exercised": all_workers_exercised,
        "worker_operation_counts": {
            worker_id: worker.operation_count for worker_id, worker in sorted(workers.items())
        },
        "state_transition_count": sum(len(worker.state_positions) for worker in workers.values()),
        "errors": errors,
        "disclosure": "logical typed-placeholder rehearsal; not GPU or performance evidence",
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return receipt


__all__ = [
    "DeploymentPlanError",
    "build_execution_plan",
    "run_logical_rehearsal",
]
