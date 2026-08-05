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
    model: Annotated[str, typer.Argument(help="OLMoE model ID or verified local snapshot.")],
    revision: Annotated[
        str,
        typer.Option(help="Mandatory immutable 40- or 64-character model revision."),
    ],
    tokenizer_revision: Annotated[
        str,
        typer.Option(help="Immutable tokenizer commit or sha256:<digest> tokenizer identity."),
    ],
    prompt: Annotated[str, typer.Option(help="Prompt; never included in status or logs.")],
    mode: Annotated[
        str,
        typer.Option(help="Planning objective: speed, capacity, or balanced."),
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
        typer.Option("--yes", help="Acknowledge pre-reviewed deployment actions."),
    ] = False,
    state_root: Annotated[
        Path | None,
        typer.Option(help="Explicit state root for development or acceptance."),
    ] = None,
) -> None:
    """Validate, plan, provision, deploy, and stream through the canonical runtime."""

    del yes  # Deployment is transactional; this flag is for non-interactive policy parity.
    if json_output and ndjson:
        fail("arguments", ValueError("--json and --ndjson are mutually exclusive"))
    if mode not in {"speed", "capacity", "balanced"}:
        fail("arguments", ValueError("mode must be speed, capacity, or balanced"))
    objective = cast(Literal["speed", "capacity", "balanced"], mode)

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
    if not explain_plan:
        plan = cast(dict[str, object], payload["plan"])
        plan.pop("report", None)
    if not json_output and not ndjson:
        typer.echo()
        typer.echo(
            f"run={summary.run_id} status={summary.status} "
            f"topology={summary.topology_id or summary.plan.topology_id}"
        )
        typer.echo(f"token_ids={json.dumps(summary.output_token_ids)}")
        if explain_plan:
            typer.echo(summary.plan.report.model_dump_json(indent=2))
        return
    emit_document(payload, json_output=json_output, ndjson=ndjson)


__all__ = ["run_command"]
