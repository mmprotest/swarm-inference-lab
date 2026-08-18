"""Exact logical A/B/C/D task-graph descriptions.

These descriptions are frozen before any modeled performance result.  They are
also used by the invalid-run audit to preserve the intended experiment without
executing beyond a failed immutable-input gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import GEOMETRY
from .models import StageAArm


@dataclass(frozen=True, slots=True)
class MaterializationStep:
    name: str
    payload: str | None
    direction: str | None


_CURRENT = (
    MaterializationStep("router", None, None),
    MaterializationStep("routed_hidden_and_metadata_fanout", "H+R", "fanout"),
    MaterializationStep("latent_down", None, None),
    MaterializationStep("routed_expert_stripe", None, None),
    MaterializationStep("full_latent_partial_gather", "L", "gather"),
    MaterializationStep("expert_reduction", None, None),
    MaterializationStep("routed_rmsnorm", None, None),
    MaterializationStep("full_normalized_latent_fanout", "L", "fanout"),
    MaterializationStep("latent_up", None, None),
    MaterializationStep("routed_hidden_gather", "H", "gather"),
    MaterializationStep("routed_reduction", None, None),
    MaterializationStep("shared_full_hidden_fanout", "H", "fanout"),
    MaterializationStep("shared_expert_stripe", None, None),
    MaterializationStep("shared_hidden_gather", "H", "gather"),
    MaterializationStep("shared_reduction", None, None),
    MaterializationStep("routed_plus_shared_final_sum", None, None),
)


def materialization_steps(arm: StageAArm) -> tuple[MaterializationStep, ...]:
    if arm is StageAArm.A_CURRENT:
        return _CURRENT
    if arm is StageAArm.B_RETAIN_HIDDEN:
        return (
            MaterializationStep("retained_full_hidden_fanout_once", "H", "fanout"),
            MaterializationStep("router_coordinator_side", None, None),
            MaterializationStep("route_metadata_fanout", "R", "fanout"),
            MaterializationStep("latent_down_waits_for_hidden_and_routes", None, None),
            *_CURRENT[3:11],
            MaterializationStep("shared_consumes_retained_hidden", None, None),
            *_CURRENT[12:],
        )
    if arm is StageAArm.C_SLICE_LATENT:
        return tuple(
            MaterializationStep(
                "local_normalized_latent_slice_fanout" if step.name == "full_normalized_latent_fanout" else step.name,
                "L_SLICE" if step.name == "full_normalized_latent_fanout" else step.payload,
                step.direction,
            )
            for step in materialization_steps(StageAArm.B_RETAIN_HIDDEN)
        )
    if arm is StageAArm.D_FUSE_OUTPUT:
        c_steps = materialization_steps(StageAArm.C_SLICE_LATENT)
        removed = {"routed_hidden_gather", "routed_reduction", "shared_hidden_gather", "shared_reduction", "routed_plus_shared_final_sum"}
        retained = tuple(step for step in c_steps if step.name not in removed)
        return (
            *retained,
            MaterializationStep("worker_local_routed_first_shared_second_fusion", None, None),
            MaterializationStep("combined_hidden_gather", "H", "gather"),
            MaterializationStep("canonical_worker_0_to_7_reduction", None, None),
        )
    raise ValueError(f"unknown Stage A arm: {arm}")


def validate_composer() -> None:
    if GEOMETRY[StageAArm.D_FUSE_OUTPUT].messages_per_row != 35:
        raise RuntimeError("D composer does not match frozen message count")
    retained = materialization_steps(StageAArm.B_RETAIN_HIDDEN)
    latent_down = next(
        index
        for index, step in enumerate(retained)
        if step.name == "latent_down_waits_for_hidden_and_routes"
    )
    route_metadata = next(
        index
        for index, step in enumerate(retained)
        if step.name == "route_metadata_fanout"
    )
    if route_metadata >= latent_down:
        raise RuntimeError("latent-down was scheduled before route metadata")
    if not any(step.payload == "H" for step in retained[:latent_down]):
        raise RuntimeError("latent-down did not receive the complete hidden input")


__all__ = ["MaterializationStep", "materialization_steps", "validate_composer"]
