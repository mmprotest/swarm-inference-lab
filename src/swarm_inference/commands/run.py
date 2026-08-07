"""High-level immutable-model run command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from swarm_inference.cluster.orchestrator import (
    ClusterOrchestrator,
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


def run_command(
    model: Annotated[
        str,
        typer.Argument(help="Hugging Face model ID/URL or local model path."),
    ],
    prompt: Annotated[str, typer.Option(help="Prompt; never included in status or logs.")],
    revision: Annotated[
        str | None,
        typer.Option(help="Advanced revision override; mutable references are pinned automatically."),
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
            decision = getattr(summary, "canonical_decision", None)
            report = getattr(summary.plan, "report", None)
            if decision is not None:
                typer.echo(decision.model_dump_json(indent=2))
            elif report is not None:
                typer.echo(report.model_dump_json(indent=2))
            else:
                typer.echo(summary.plan.model_dump_json(indent=2))
        return
    emit_document(payload, json_output=json_output, ndjson=ndjson)


__all__ = ["run_command"]
