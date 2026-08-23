"""Generate fail-closed E025 reports and static publication evidence."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .io import atomic_write_json, read_json, sha256_file, utc_now

NAVY = "#10243E"
BLUE = "#2C6ECB"
CYAN = "#18A6B8"
ORANGE = "#F28C28"
GREEN = "#218739"
RED = "#B42318"
LIGHT = "#F6F8FB"
GREY = "#5D6B7A"


def _base_figure(*, portrait: bool = False) -> tuple[Any, Any]:
    size = (12, 15) if portrait else (16, 9)
    figure, axis = plt.subplots(figsize=size, dpi=200)
    figure.patch.set_facecolor(LIGHT)
    axis.set_facecolor(LIGHT)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def _save(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def _badge(axis: Any, text: str, *, x: float, y: float, color: str) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": color},
    )


def _hero(summary: dict[str, Any], path: Path) -> None:
    figure, axis = _base_figure(portrait=True)
    axis.text(0.5, 0.91, "KIMI K3", ha="center", fontsize=50, fontweight="bold", color=NAVY)
    axis.text(
        0.5,
        0.83,
        "2.8 TRILLION PARAMETERS",
        ha="center",
        fontsize=28,
        fontweight="bold",
        color=BLUE,
    )
    axis.text(
        0.5,
        0.75,
        "RUNNING ACROSS A\nCONSUMER GPU SWARM",
        ha="center",
        va="center",
        fontsize=31,
        fontweight="bold",
        color=NAVY,
        linespacing=1.25,
    )
    _badge(axis, "PHYSICAL EVIDENCE", x=0.5, y=0.64, color=GREEN)
    metrics = [
        (summary["physical_machine_count"], "PHYSICAL\nMACHINES"),
        (summary["consumer_gpu_count"], "CONSUMER\nGPUs"),
        (summary["qualifying_sub_layer_gpu_count"], "SUB-LAYER\nGPUs"),
    ]
    for index, (value, label) in enumerate(metrics):
        x = 0.18 + index * 0.32
        axis.add_patch(
            FancyBboxPatch(
                (x - 0.125, 0.39),
                0.25,
                0.17,
                boxstyle="round,pad=0.018",
                facecolor="white",
                edgecolor="#D8E0EA",
                linewidth=2,
            )
        )
        axis.text(x, 0.49, str(value), ha="center", fontsize=38, fontweight="bold", color=NAVY)
        axis.text(x, 0.425, label, ha="center", fontsize=15, color=GREY, linespacing=1.2)
    generated = summary["generation"]
    decoded = str(generated["decoded_text"]).strip().replace("\n", " ")
    axis.text(0.5, 0.30, "REAL AUTOREGRESSIVE OUTPUT", ha="center", fontsize=14, color=GREY)
    axis.text(
        0.5,
        0.235,
        f'“{decoded[:150]}”',
        ha="center",
        va="center",
        fontsize=18,
        color=NAVY,
        wrap=True,
    )
    axis.text(
        0.5,
        0.10,
        f"{len(generated['generated_token_ids'])} generated tokens  •  "
        f"{generated['decode_tokens_per_second']:.4f} tok/s  •  greedy decode",
        ha="center",
        fontsize=14,
        color=GREY,
    )
    _save(figure, path)


def _topology(summary: dict[str, Any], path: Path) -> None:
    figure, axis = _base_figure()
    axis.text(0.04, 0.93, "Physical Kimi K3 swarm topology", fontsize=28, fontweight="bold", color=NAVY)
    axis.text(
        0.04,
        0.885,
        "Actual headline workers • controller relays bytes only • no geographic positions implied",
        fontsize=13,
        color=GREY,
    )
    registration = summary["generation"]["registration"]["workers"]
    stages = sorted(
        (
            (int(worker_id.split("-")[2]), worker_id, ready["gpu"]["gpu_name"])
            for worker_id, ready in registration.items()
            if ready.get("role") != "SUB_LAYER_WORKER"
        ),
        key=lambda row: row[0],
    )
    controller_x, controller_y = 0.08, 0.50
    axis.add_patch(plt.Circle((controller_x, controller_y), 0.045, color=BLUE))
    axis.text(controller_x, controller_y, "CTRL", color="white", ha="center", va="center", fontweight="bold")
    columns = 16
    for index, (layer, _worker_id, gpu) in enumerate(stages):
        column = index % columns
        row = index // columns
        x = 0.18 + column * 0.049
        y = 0.79 - row * 0.105
        color = ORANGE if layer == 89 else CYAN
        axis.plot([controller_x + 0.045, x], [controller_y, y], color="#D5DCE5", linewidth=0.35, zorder=0)
        axis.add_patch(plt.Circle((x, y), 0.014, color=color, zorder=2))
        short_gpu = str(gpu).replace("NVIDIA GeForce ", "")
        axis.text(x, y - 0.025, f"L{layer}\n{short_gpu}", ha="center", va="top", fontsize=4.7, color=NAVY)
    fragments = summary["sub_layer_memory_proof"]
    for index, row in enumerate(fragments):
        x = 0.30 + index * 0.16
        y = 0.09
        parent = next(item for item in stages if item[0] == 89)
        parent_index = stages.index(parent)
        px = 0.18 + (parent_index % columns) * 0.049
        py = 0.79 - (parent_index // columns) * 0.105
        axis.plot([px, x], [py, y], color=ORANGE, linewidth=1.6)
        axis.add_patch(plt.Circle((x, y), 0.026, color=ORANGE))
        axis.text(
            x,
            y - 0.04,
            f"{row['worker_id']}\n{str(row['gpu_name']).replace('NVIDIA GeForce ', '')}",
            ha="center",
            va="top",
            fontsize=7,
            color=NAVY,
        )
    axis.text(0.18, 0.03, "93 coarse layer stages", color=CYAN, fontsize=12, fontweight="bold")
    axis.text(0.51, 0.03, "4 mandatory layer-89 fragment workers", color=ORANGE, fontsize=12, fontweight="bold")
    _save(figure, path)


def _sub_layer(summary: dict[str, Any], path: Path) -> None:
    figure, axis = _base_figure()
    axis.text(0.04, 0.92, "Why the small GPUs qualify", fontsize=30, fontweight="bold", color=NAVY)
    axis.text(
        0.04,
        0.865,
        "Each physical worker executed a disjoint fragment of K3 layer 89; the complete layer was not resident elsewhere.",
        fontsize=13,
        color=GREY,
    )
    for index, row in enumerate(summary["sub_layer_memory_proof"]):
        x = 0.04 + index * 0.24
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.20),
                0.215,
                0.57,
                boxstyle="round,pad=0.018",
                facecolor="white",
                edgecolor=ORANGE,
                linewidth=2,
            )
        )
        axis.text(x + 0.1075, 0.70, str(row["gpu_name"]).replace("NVIDIA GeForce ", ""), ha="center", fontsize=18, fontweight="bold", color=NAVY)
        axis.text(x + 0.1075, 0.65, f"{row['physical_vram_gib']:.1f} GiB physical", ha="center", fontsize=12, color=GREY)
        axis.text(x + 0.1075, 0.57, "K3 LAYER 89 FRAGMENT", ha="center", fontsize=12, fontweight="bold", color=ORANGE)
        axis.text(x + 0.025, 0.48, f"Complete layer peak\n{row['complete_layer_peak_gib']:.1f} GiB", fontsize=12, color=RED)
        axis.text(x + 0.025, 0.38, f"Fragment peak\n{row['fragment_peak_gib']:.1f} GiB", fontsize=12, color=GREEN)
        axis.text(
            x + 0.025,
            0.29,
            f"Owns {row['assigned_expert_count']} experts\n"
            f"EXECUTED: {row['retained_headline_execution_count']} tokens",
            fontsize=11,
            color=NAVY,
        )
        _badge(axis, "PHYSICAL WORKER", x=x + 0.1075, y=0.235, color=GREEN)
    _save(figure, path)


def _inference(summary: dict[str, Any], path: Path) -> None:
    generation = summary["generation"]
    figure, axis = _base_figure()
    axis.text(0.04, 0.92, "Physical inference proof", fontsize=30, fontweight="bold", color=NAVY)
    _badge(axis, "REAL KIMI K3 • PHYSICAL CONSUMER SWARM", x=0.31, y=0.84, color=GREEN)
    axis.text(0.06, 0.72, "USER PROMPT", fontsize=12, color=GREY, fontweight="bold")
    axis.text(0.06, 0.66, str(generation["prompt"]), fontsize=20, color=NAVY)
    axis.text(0.06, 0.56, "KIMI K3 RESPONSE", fontsize=12, color=GREY, fontweight="bold")
    axis.add_patch(FancyBboxPatch((0.05, 0.25), 0.90, 0.27, boxstyle="round,pad=0.02", facecolor="white", edgecolor="#D8E0EA"))
    axis.text(0.08, 0.385, str(generation["decoded_text"]), fontsize=19, color=NAVY, va="center", wrap=True)
    axis.text(
        0.06,
        0.13,
        f"{len(generation['generated_token_ids'])} tokens  •  "
        f"TTFT {generation['time_to_first_token_seconds']:.2f}s  •  "
        f"{generation['decode_tokens_per_second']:.4f} tok/s  •  "
        f"{summary['physical_machine_count']} machines",
        fontsize=15,
        color=GREY,
    )
    _save(figure, path)


def _audit(summary: dict[str, Any], path: Path) -> None:
    figure, axis = _base_figure()
    status = summary["status"]
    color = GREEN if status == "PASS" else RED
    axis.text(0.04, 0.92, "E025 audit proof", fontsize=30, fontweight="bold", color=NAVY)
    _badge(axis, status, x=0.87, y=0.91, color=color)
    checks = [
        ("Checkpoint", f"moonshotai/Kimi-K3 @ {summary['model_revision'][:12]}…"),
        ("Image digest", str(summary["image_digest"])[:24] + "…"),
        ("Evidence", summary["evidence_class"]),
        ("Physical machines", str(summary["physical_machine_count"])),
        ("Consumer only", str(summary["pass_gates"]["G3_consumer_only_headline_compute"])),
        ("Byte/tensor coverage", str(summary["pass_gates"]["G5_required_model_coverage"])),
        ("Sub-layer necessity", str(summary["pass_gates"]["G10_sub_layer_necessity"])),
        ("Numerical correctness", str(summary["pass_gates"]["G11_numerical_correctness"])),
        ("Stateful decode", str(summary["pass_gates"]["G12_stateful_autoregressive_correctness"])),
        ("Zero live Vast instances", str(summary["cleanup"]["zero_live_e025_instances"])),
    ]
    for index, (label, value) in enumerate(checks):
        y = 0.80 - index * 0.068
        axis.text(0.06, y, label, fontsize=13, color=GREY, fontweight="bold")
        axis.text(0.36, y, value, fontsize=13, color=NAVY)
    axis.text(0.06, 0.08, "Generated only from retained machine-readable evidence.", fontsize=12, color=GREY)
    _save(figure, path)


def _markdown(summary: dict[str, Any]) -> str:
    generation = summary.get("generation") or {}
    cleanup = summary.get("cleanup") or {}
    costs = summary.get("rental_costs") or {}
    gates = summary.get("pass_gates") or {}
    passed = summary["status"] == "PASS"
    generated = bool(
        gates.get("G1_authoritative_kimi_k3")
        and gates.get("G13_human_readable_generation")
    )
    memory_rows = summary.get("sub_layer_memory_proof") or []
    memory_answer = (
        "The reconciled complete-layer runtime peak exceeded each worker's usable "
        "VRAM, while each measured fragment fit its safety envelope."
        if memory_rows and gates.get("G7_complete_layer_cannot_fit")
        else "That physical memory claim was not established."
    )
    if passed:
        physical_method = """## What physically ran

The controller sent authenticated `EXECUTE_SHARD` frames to independent GPU worker processes. Each of the 93 transformer stages executed the current native Kimi K3 implementation. Layer 89 retained its non-expert work on a parent GPU and dispatched its real routed-expert tensor operations to four disjoint physical consumer-GPU workers. The controller relayed tensors and coordinated greedy sampling; it owned no model tensors and performed no missing model compute.

The headline path used the authoritative `moonshotai/Kimi-K3` checkpoint revision `{model_revision}`. This was a text-only run: the complete required text-generation tensor census was covered, while multimodal-only tensors absent from that authoritative text path were not executed.

## Correctness and falsification controls

The deterministic `Hi` fixture traversed the physical fleet, including a post-prefill decode step. Layer outputs and routes were compared against the trusted retained native oracle. The public generation ran only after that gate passed. Process snapshots, immutable worker identities, native execution receipts, tensor ownership, and transport traces support the controls against API generation, cached output, synthetic tensors, controller fallback, and hidden complete layer-89 execution.

The frozen placement was shown physically invalid when a mandatory fragment worker was disabled during the Stage 2 micro-swarm canary; no replan was allowed in that control.
""".format(model_revision=summary["model_revision"])
    else:
        physical_method = f"""## What was established

E025 did not earn the headline claim. Planned topology or partial physical activity is not reported as a completed swarm. The decisive blocker was: `{summary.get('decisive_failure')}`.

Any retained canary or partial-fleet evidence remains classified at its own stated evidence class. No victory graphics were generated.
"""
    return f"""# Experiment 025: Physical Kimi K3 Consumer Swarm

## Result at a glance

1. **Did real Kimi K3 generate tokens?** {'Yes' if generated else 'No — not proven by the retained headline gates'}.
2. **Physical machines / GPUs:** {summary.get('physical_machine_count', 0)} / {summary.get('consumer_gpu_count', 0)}.
3. **Consumer GPUs only?** {summary.get('pass_gates', {}).get('G3_consumer_only_headline_compute', False)}.
4. **Separate sub-layer workers participated?** {summary.get('qualifying_sub_layer_gpu_count', 0)} qualifying physical workers.
5. **Why could they not hold the full layer?** {memory_answer}
6. **Numerically correct?** {gates.get('G11_numerical_correctness', False)}; stateful decode: {gates.get('G12_stateful_autoregressive_correctness', False)}.
7. **Actual throughput:** {generation.get('decode_tokens_per_second', 0):.6f} output tokens/s.
8. **Every Vast instance destroyed?** {cleanup.get('zero_live_e025_instances', False)}.

**Classification:** `{summary['status']}`  
**Evidence class:** `{summary['evidence_class']}`

{('**Strongest defensible public claim:** ' + summary['strongest_public_claim']) if passed else '**Decisive blocker:** ' + str(summary.get('decisive_failure'))}

{physical_method}

## Performance disclosure

There was deliberately no tokens/s acceptance threshold. The measured headline decode rate was `{generation.get('decode_tokens_per_second', 0):.6f}` tok/s, with `{generation.get('time_to_first_token_seconds', 0):.3f}` seconds to the first generated token. This experiment establishes physical execution, not performance or economic leadership.

The lifecycle-ledger estimate was `${costs.get('estimated_active_cost_usd', 0):.4f}` active rental plus `${costs.get('estimated_storage_cost_usd', 0):.4f}` storage for the headline stage, with `${costs.get('ready_worker_observed_ingress_cost_usd', 0):.4f}` of ingress calculated from READY workers' observed downloaded bytes and their selected offer rates. These are transparent ledger-derived estimates, not a provider invoice.

## Cleanup

The E025 lifecycle manager targeted instance IDs from its append-only ledger and recovered any matching run-labeled instances that were absent from the ledger. A final read-only Vast query found `{len(cleanup.get('surviving_instance_ids', []))}` surviving instances for this run and `{len(cleanup.get('other_live_e025_instance_ids', []))}` other live E025 instances. Cleanup is a mandatory PASS gate.
"""


def _hash_manifest(root: Path, output_path: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != output_path.resolve()
    )
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    payload = {
        "schema_version": "experiment-025-artifact-hashes-v1",
        "generated_at_utc": utc_now(),
        "file_count": len(rows),
        "files": rows,
        "excluded_paths": [
            output_path.relative_to(root).as_posix(),
            "final/render-receipt.json (written after this manifest)",
        ],
    }
    atomic_write_json(output_path, payload)
    return payload


def render_final_artifacts(run_root: Path) -> dict[str, Any]:
    root = run_root.expanduser().resolve()
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    summary = read_json(final / "summary.json")
    audit_path = final / "audit-proof.png"
    _audit(summary, audit_path)
    images = [audit_path]
    if summary.get("status") == "PASS":
        paths = {
            "hero": final / "hero-linkedin.png",
            "topology": final / "swarm-topology.png",
            "sub_layer": final / "sub-layer-proof.png",
            "inference": final / "inference-proof.png",
        }
        _hero(summary, paths["hero"])
        _topology(summary, paths["topology"])
        _sub_layer(summary, paths["sub_layer"])
        _inference(summary, paths["inference"])
        images.extend(paths.values())
    report = _markdown(summary)
    (final / "EXPERIMENT_025_REPORT.md").write_text(report, encoding="utf-8")
    manager = (
        f"# E025 manager summary\n\nStatus: **{summary['status']}**\n\n"
        f"Evidence: `{summary['evidence_class']}`\n\n"
        f"Zero live Vast instances: `{summary['cleanup']['zero_live_e025_instances']}`\n"
    )
    (final / "manager-summary.md").write_text(manager, encoding="utf-8")
    image_html = "".join(
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></figure>'
        for path in images
    )
    static_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>E025 physical Kimi K3 evidence</title><style>
body{{font-family:Inter,Segoe UI,sans-serif;background:#f6f8fb;color:#10243e;margin:0;padding:32px}}
main{{max-width:1400px;margin:auto}}h1{{font-size:42px}}.status{{font-weight:800;color:{GREEN if summary['status']=='PASS' else RED}}}
figure{{margin:28px 0;background:white;padding:12px;border-radius:14px}}img{{width:100%;height:auto}}
code{{background:#e9eef5;padding:2px 5px;border-radius:4px}}</style></head><body><main>
<h1>Experiment 025 evidence</h1><p class="status">{html.escape(summary['status'])}</p>
<p>Evidence class: <code>{html.escape(summary['evidence_class'])}</code></p>{image_html}</main></body></html>"""
    (final / "evidence.html").write_text(static_html, encoding="utf-8")
    hashes = _hash_manifest(root, final / "artifact-hashes.json")
    receipt = {
        "schema_version": "experiment-025-final-render-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "experiment_status": summary["status"],
        "victory_graphics_generated": summary["status"] == "PASS",
        "images": [str(path) for path in images],
        "report": str(final / "EXPERIMENT_025_REPORT.md"),
        "html": str(final / "evidence.html"),
        "artifact_hashes": hashes,
    }
    atomic_write_json(final / "render-receipt.json", receipt)
    return receipt


__all__ = ["render_final_artifacts"]
