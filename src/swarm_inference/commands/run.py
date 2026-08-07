"""High-level immutable-model run command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from swarm_inference.cluster.orchestrator import (
    ClusterOrchestrator,
    ClusterRunSummary,
    RunProgress,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.commands._common import emit_document, fail
from swarm_inference.protocol.messages import StreamEventType, SubmitStreamEvent


def _safe_stream_event(event: SubmitStreamEvent) -> dict[str, object]:
    """Return the useful public subset of a stream event.

    Submit events never contain the prompt, authentication material, or pairing
    material.  Keeping the schema explicit also prevents future protocol fields
    from being copied into command output accidentally.
    """

    return {
        "event": "submission",
        "event_type": event.event_type.value,
        "request_id": event.request_id,
        "sequence_number": event.sequence_number,
        "token_position": event.token_position,
        "token_id": event.token_id,
        "decoded_text_fragment": event.decoded_text_fragment,
        "status_detail": event.status_detail,
        "topology_id": event.topology_id,
        "model_revision": event.model_revision,
    }


def _plan_explanation(summary: ClusterRunSummary) -> dict[str, object]:
    """Build the stable, operator-facing preflight explanation."""

    decision = summary.canonical_decision
    if decision is None:
        selected = summary.plan
        candidates: tuple[object, ...] = (summary.plan,)
        rejected_plans: dict[str, tuple[str, ...]] = {}
    else:
        selected = decision.selected
        candidates = decision.candidates
        rejected_plans = decision.rejected_plans

    selected_engine_id = getattr(selected, "engine_id", summary.engine_id)
    selected_explanation = list(getattr(selected, "explanation", ()))
    if summary.requested_engine is not None:
        why_selected = [
            f"explicit engine request {summary.requested_engine!r} passed capability preflight",
            *selected_explanation,
        ]
    else:
        why_selected = [
            "automatic competition selected the highest eligible objective score after "
            "network-cost penalties",
            *selected_explanation,
        ]

    compatible_engines: list[dict[str, object]] = []
    rejected_engines: list[dict[str, object]] = []
    candidate_engine_ids = {getattr(candidate, "engine_id", "unknown") for candidate in candidates}
    for report in summary.engine_support:
        item: dict[str, object] = {
            "engine": report.engine_id,
            "compatibility": report.compatibility,
            "reason": report.reason,
            "adapter": report.adapter_id,
            "required_runtime": report.required_runtime,
            "required_features": list(report.required_features),
            "unsupported_features": list(report.unsupported_features),
            "produced_feasible_plan": report.engine_id in candidate_engine_ids,
            "selected": report.engine_id == selected_engine_id,
        }
        if report.supported:
            compatible_engines.append(item)
        else:
            rejected_engines.append(item)

    worker_roles = dict(getattr(selected, "worker_roles", {}))
    idle_workers = dict(getattr(selected, "idle_workers", {}))
    participating_workers = [
        {"worker_id": worker_id, "role": role}
        for worker_id, role in sorted(worker_roles.items())
        if role not in {"idle", "background_replica", "storage_cache", "verification"}
    ]
    excluded_workers = [
        {"worker_id": worker_id, "reason": reason}
        for worker_id, reason in sorted(idle_workers.items())
    ]

    raw_assignments = list(getattr(selected, "stage_assignments", ()))
    if not raw_assignments and hasattr(summary.plan, "assignments"):
        raw_assignments = [
            {
                "stage_id": assignment.stage_id,
                "worker_id": assignment.worker_id,
                "layer_start": assignment.assignment.layer_start,
                "layer_end": assignment.assignment.layer_end,
                "model_bytes": assignment.assignment.weight_bytes,
                "expected_memory_bytes": assignment.required_memory_bytes,
                "execution_device": assignment.device,
            }
            for assignment in summary.plan.assignments
        ]
    if not raw_assignments:
        tensor_split = getattr(selected, "engine_parameters", {}).get("tensor_split", {})
        if isinstance(tensor_split, dict):
            raw_assignments = [
                {
                    "stage_id": index,
                    "worker_id": worker_id,
                    "ownership": "llama.cpp tensor share",
                    "model_bytes": int(summary.total_model_size_bytes * float(fraction)),
                    "expected_memory_bytes": int(summary.total_model_size_bytes * float(fraction)),
                    "execution_device": "runtime-selected",
                }
                for index, (worker_id, fraction) in enumerate(sorted(tensor_split.items()))
            ]
    stage_ownership = [dict(item) for item in raw_assignments]
    memory_and_device = [
        {
            "worker_id": item.get("worker_id", "unknown"),
            "execution_device": item.get("execution_device", "unknown"),
            "expected_ram_or_vram_bytes": item.get("expected_memory_bytes", "unknown"),
            "model_bytes_owned": item.get("model_bytes", "unknown"),
        }
        for item in stage_ownership
    ]

    topology = getattr(selected, "topology", getattr(summary.plan, "topology_id", "unknown"))
    network = {
        "topology": topology,
        "expected_wan_stage_boundaries": getattr(selected, "number_of_wan_stage_boundaries", None),
        "persistent_connections": getattr(selected, "persistent_connections", None),
        "estimated_bytes_per_token": getattr(selected, "predicted_bytes_per_token", None),
        "estimated_network_operations_per_token": getattr(
            selected, "predicted_messages_per_token", None
        ),
        "estimated_serial_waits_per_token": getattr(
            selected, "predicted_serial_waits_per_token", None
        ),
        "network_cost_confidence": getattr(selected, "network_cost_confidence", "unmeasured"),
        "network_cost_provenance": getattr(selected, "network_cost_provenance", "unmeasured"),
    }
    rejected_candidates = [
        {"plan_id": plan_id, "reasons": list(reasons)}
        for plan_id, reasons in sorted(rejected_plans.items())
    ]
    return {
        "model": {
            "id": summary.model_id,
            "architecture": summary.model_architecture or "unknown",
            "architecture_source": summary.model_architecture_source,
            "format": summary.model_format,
            "revision": summary.model_revision,
            "quantization": summary.quantization or "none",
            "variant": summary.variant or "default",
            "total_model_size_bytes": summary.total_model_size_bytes,
        },
        "selected_engine": {
            "engine": selected_engine_id,
            "requested_explicitly": summary.requested_engine is not None,
            "why_selected": why_selected,
        },
        "compatible_engines": compatible_engines,
        "rejected_engines": rejected_engines,
        "rejected_candidate_plans": rejected_candidates,
        "participating_workers": participating_workers,
        "excluded_workers": excluded_workers,
        "stage_ownership": stage_ownership,
        "worker_memory_and_devices": memory_and_device,
        "network_topology": network,
        "distributed_execution_required": summary.distributed_execution_required,
        "distributed_execution_achieved_by_plan": summary.distributed_execution_achieved,
    }


def run_command(
    model: Annotated[
        str,
        typer.Argument(help="Hugging Face model ID/URL or local model path."),
    ],
    prompt: Annotated[str, typer.Option(help="Prompt; never included in status or logs.")],
    revision: Annotated[
        str | None,
        typer.Option(
            help="Advanced revision override; mutable references are pinned automatically."
        ),
    ] = None,
    tokenizer_revision: Annotated[
        str | None,
        typer.Option(help="Advanced immutable tokenizer identity override."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(help="Advanced model-file variant override."),
    ] = None,
    quantization: Annotated[
        str | None,
        typer.Option("--quant", help="Advanced quantization override."),
    ] = None,
    engine: Annotated[
        str | None,
        typer.Option(help="Force one registered engine; unsupported engines fail closed."),
    ] = None,
    require_distributed: Annotated[
        bool,
        typer.Option(help="Require real required computation on at least two physical hosts."),
    ] = False,
    concurrency: Annotated[
        int,
        typer.Option(min=1, max=4096, help="Expected concurrent request count."),
    ] = 1,
    context_tokens: Annotated[
        int,
        typer.Option(min=1, help="Required context capacity for variant and topology planning."),
    ] = 2048,
    mode: Annotated[
        str,
        typer.Option(help="Planning objective: speed, throughput, capacity, or balanced."),
    ] = "speed",
    dry_run: Annotated[
        bool,
        typer.Option(help="Validate source and plan without artifact or deployment mutation."),
    ] = False,
    explain_plan: Annotated[
        bool,
        typer.Option(help="Include the complete deterministic planning report."),
    ] = False,
    require_node: Annotated[
        list[str] | None,
        typer.Option("--require-node", help="Require a node; repeat for more than one."),
    ] = None,
    exclude_node: Annotated[
        list[str] | None,
        typer.Option("--exclude-node", help="Exclude a node; repeat for more than one."),
    ] = None,
    max_new_tokens: Annotated[
        int,
        typer.Option(min=1, max=4096, help="Bounded maximum generated token count."),
    ] = 16,
    seed: Annotated[int, typer.Option(min=0, help="Deterministic request seed.")] = 1,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit one final machine-readable document."),
    ] = False,
    ndjson: Annotated[
        bool,
        typer.Option(help="Emit progress, tokens, and the final document as JSON lines."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Record explicit preauthorization; run submission itself does not prompt.",
        ),
    ] = False,
    state_root: Annotated[
        Path | None,
        typer.Option(help="Explicit state root for development or acceptance."),
    ] = None,
) -> None:
    """Validate, plan, provision, deploy, and stream through the canonical runtime."""

    if json_output and ndjson:
        fail("arguments", ValueError("--json and --ndjson are mutually exclusive"))
    if mode not in {"speed", "throughput", "capacity", "balanced"}:
        fail("arguments", ValueError("mode must be speed, throughput, capacity, or balanced"))
    objective = cast(Literal["speed", "throughput", "capacity", "balanced"], mode)

    def progress(item: RunProgress) -> None:
        if ndjson:
            emit_document(item.model_dump(mode="json"), json_output=False, ndjson=True)
        elif not json_output:
            typer.echo(f"[{item.stage}] {item.detail}", err=True)

    def stream(item: SubmitStreamEvent) -> None:
        if ndjson:
            emit_document(_safe_stream_event(item), json_output=False, ndjson=True)
        elif not json_output and item.event_type == StreamEventType.TOKEN_GENERATED:
            typer.echo(item.decoded_text_fragment, nl=False)

    state = ClusterStateStore(state_root)
    orchestrator = ClusterOrchestrator(
        state=state,
        progress_sink=progress,
        stream_sink=stream,
    )
    try:
        summary = asyncio.run(
            orchestrator.run(
                model_id=model,
                model_revision=revision,
                tokenizer_revision=tokenizer_revision,
                variant=variant,
                quantization=quantization,
                requested_engine=engine,
                require_distributed=require_distributed,
                concurrency=concurrency,
                max_context_tokens=context_tokens,
                prompt=prompt,
                mode=objective,
                dry_run=dry_run,
                required_node_ids=sorted(set(require_node or [])),
                excluded_node_ids=sorted(set(exclude_node or [])),
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        fail("run", exc)

    payload = summary.model_dump(mode="json")
    payload["confirmation_policy"] = "preauthorized" if yes else "submission-does-not-prompt"
    if explain_plan:
        payload["explain_plan"] = _plan_explanation(summary)
    if not explain_plan:
        plan = cast(dict[str, object], payload["plan"])
        plan.pop("report", None)
    if not json_output and not ndjson:
        plan_topology = getattr(
            summary.plan,
            "topology_id",
            getattr(summary.plan, "topology", "unavailable"),
        )
        typer.echo()
        typer.echo(
            f"run={summary.run_id} status={summary.status} "
            f"topology={summary.topology_id or plan_topology}"
        )
        typer.echo(f"token_ids={json.dumps(summary.output_token_ids)}")
        if explain_plan:
            typer.echo(json.dumps(_plan_explanation(summary), indent=2))
        return
    emit_document(payload, json_output=json_output, ndjson=ndjson)


__all__ = ["_plan_explanation", "run_command"]
