"""Measured execution-engine registration and competition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionEngine,
    ExecutionPlan,
    ExecutionRequest,
)
from swarm_inference.model.descriptor import ResolvedModelDescriptor


@dataclass(frozen=True, slots=True)
class EngineCompetitionResult:
    selected: ExecutionPlan
    candidates: tuple[ExecutionPlan, ...]
    support: tuple[EngineSupportReport, ...]


class ExecutionEngineRegistry:
    """A registry whose coordinator-facing behavior is independent of engine families."""

    def __init__(self, engines: tuple[ExecutionEngine, ...] = ()) -> None:
        self._engines: dict[str, ExecutionEngine] = {}
        for engine in engines:
            self.register(engine)

    def register(self, engine: ExecutionEngine, *, replace: bool = False) -> None:
        engine_id = engine.engine_id.strip()
        if not engine_id:
            raise ValueError("execution engine ID cannot be empty")
        if engine_id in self._engines and not replace:
            raise ValueError(f"execution engine {engine_id!r} is already registered")
        self._engines[engine_id] = engine

    def unregister(self, engine_id: str) -> ExecutionEngine:
        try:
            return self._engines.pop(engine_id)
        except KeyError as exc:
            raise KeyError(f"execution engine {engine_id!r} is not registered") from exc

    def get(self, engine_id: str) -> ExecutionEngine:
        try:
            return self._engines[engine_id]
        except KeyError as exc:
            raise KeyError(f"execution engine {engine_id!r} is not registered") from exc

    def engines(self) -> tuple[ExecutionEngine, ...]:
        return tuple(self._engines[key] for key in sorted(self._engines))

    def probe_all(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> tuple[EngineSupportReport, ...]:
        reports: list[EngineSupportReport] = []
        for engine in self.engines():
            try:
                capability_probe = getattr(engine, "probe_model_support", None)
                report = (
                    capability_probe(model, cluster)
                    if callable(capability_probe)
                    else engine.probe(model, cluster)
                )
            except Exception as exc:
                report = EngineSupportReport(
                    engine_id=engine.engine_id,
                    status=EngineSupportStatus.BROKEN_RUNTIME,
                    reason=f"capability probe failed: {type(exc).__name__}: {exc}",
                    model_architecture=model.architecture,
                    model_format=model.format,
                )
            if report.engine_id != engine.engine_id:
                raise ValueError("engine support report identifies a different engine")
            reports.append(report)
        return tuple(reports)

    async def compete(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> EngineCompetitionResult:
        reports = self.probe_all(model, cluster)
        by_id = {item.engine_id: item for item in reports}
        if request.requested_engine is not None:
            engine = self.get(request.requested_engine)
            report = by_id[engine.engine_id]
            if not report.supported:
                raise RuntimeError(
                    f"forced engine {engine.engine_id!r} is unavailable: "
                    f"{report.status.value}: {report.reason}"
                )
            selected_engines: tuple[ExecutionEngine, ...] = (engine,)
        else:
            selected_engines = tuple(
                engine for engine in self.engines() if by_id[engine.engine_id].supported
            )
        if not selected_engines:
            detail = "; ".join(
                f"{item.engine_id}={item.status.value}: {item.reason}" for item in reports
            )
            raise RuntimeError(f"no execution engine supports the resolved model; {detail}")
        plan_groups = await asyncio.gather(
            *(engine.candidate_plans(model, cluster, request) for engine in selected_engines)
        )
        raw_plans = tuple(plan for group in plan_groups for plan in group)
        plans = tuple(self._with_network_competition_score(plan) for plan in raw_plans)
        if not plans:
            raise RuntimeError("supported execution engines returned no feasible plan")
        worker_nodes = {worker.worker_id: worker.node_id for worker in cluster.workers}
        non_required_roles = {"idle", "background_replica", "storage_cache", "verification"}
        compatible = []
        for plan in plans:
            participating_nodes = {
                worker_nodes[worker_id]
                for worker_id, role in plan.worker_roles.items()
                if worker_id in worker_nodes and role not in non_required_roles
            }
            if not request.require_distributed or len(participating_nodes) >= 2:
                compatible.append(plan)
        if not compatible:
            raise RuntimeError("forced distributed execution has no feasible multi-worker plan")
        selected = max(
            compatible,
            key=lambda plan: (
                float(plan.engine_parameters["competition_score"]),
                plan.predicted_decode_tokens_s,
                -plan.predicted_ttft_ms,
                plan.engine_id,
                plan.plan_id,
            ),
        )
        # Return only request-compatible plans.  The hierarchical planner must
        # never accidentally re-admit a local candidate after this physical
        # distributed gate has rejected it.
        return EngineCompetitionResult(selected, tuple(compatible), reports)

    @staticmethod
    def _with_network_competition_score(plan: ExecutionPlan) -> ExecutionPlan:
        confidence_multiplier = {
            "measured": 1.0,
            "estimated": 0.96,
            "unmeasured": 0.88,
        }[plan.network_cost_confidence]
        boundary_count = plan.number_of_wan_stage_boundaries
        boundary_multiplier = 0.94**boundary_count if boundary_count is not None else 0.92
        observability_multiplier = (
            1.0
            if plan.predicted_bytes_per_token is not None
            and plan.predicted_messages_per_token is not None
            else 0.94
        )
        competition_score = (
            plan.score * confidence_multiplier * boundary_multiplier * observability_multiplier
        )
        parameters = dict(plan.engine_parameters)
        parameters.update(
            {
                "engine_score": plan.score,
                "competition_score": competition_score,
                "network_competition_factors": {
                    "confidence": plan.network_cost_confidence,
                    "confidence_multiplier": confidence_multiplier,
                    "wan_boundaries": boundary_count,
                    "boundary_multiplier": boundary_multiplier,
                    "communication_observed_or_estimated": (
                        plan.predicted_bytes_per_token is not None
                        and plan.predicted_messages_per_token is not None
                    ),
                    "observability_multiplier": observability_multiplier,
                },
            }
        )
        return plan.model_copy(
            update={
                "engine_parameters": parameters,
                "explanation": (
                    *plan.explanation,
                    "engine competition score includes network confidence, WAN "
                    "boundaries, and communication observability",
                ),
            }
        )


def default_engine_registry() -> ExecutionEngineRegistry:
    """Construct the built-in registry without teaching callers model families."""

    from swarm_inference.engines.colibri import ColibriExecutionEngine
    from swarm_inference.engines.llamacpp_rpc import LlamaCppRpcEngine
    from swarm_inference.engines.native_stage import NativeStageEngine

    return ExecutionEngineRegistry(
        (NativeStageEngine(), LlamaCppRpcEngine(), ColibriExecutionEngine())
    )


__all__ = ["EngineCompetitionResult", "ExecutionEngineRegistry", "default_engine_registry"]
