"""Render the durable Experiment 019 Markdown report from final receipts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    top = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join((top, separator, *body))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_report(
    root: Path,
    summary: Mapping[str, Any],
    truth: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> str:
    best = summary["best_exact"]
    classification = summary["classification"]
    evidence = summary["evidence_class"]
    truth_rows = [(row["question"], row["answer"]) for row in truth["rows"]]
    tier_rows = _csv(root / "simulation/worker-sweep.csv")
    tier_best: list[dict[str, str]] = []
    for cap in (20, 8, 4, 2, 1):
        candidates = [
            row
            for row in tier_rows
            if row["valid"] == "True" and float(row["memory_cap_gib"]) == cap
        ]
        if candidates:
            tier_best.append(max(candidates, key=lambda row: float(row["exact_tok_s_per_user"])))
    gates = validation["hard_gates"]
    gate_table = _table(
        ("Gate", "Status", "Receipt"),
        [(row["gate"], row["status"], row["evidence"]) for row in gates],
    )
    tier_table = _table(
        ("Cap", "Best tok/s", "Workers", "P", "Depth", "Peak GiB", "Block/chunk"),
        [
            (
                f"{float(row['memory_cap_gib']):g} GiB",
                f"{float(row['exact_tok_s_per_user']):.4f}",
                row["worker_count"],
                row["stripe_degree"],
                row["depth_span"],
                f"{float(row['max_worker_peak_gib']):.3f}",
                f"{row['block']}/{row['chunk']}",
            )
            for row in tier_best
        ],
    )
    network_rows = _csv(root / "network/network-sensitivity.csv")
    network_table = _table(
        ("Local profile", "Inter-pod", "Strategy", "tok/s", "Pass 5?"),
        [
            (
                row["local_profile"],
                row["inter_pod_profile"],
                row["activation_strategy"],
                f"{float(row['exact_tok_s_per_user']):.3f}",
                "YES" if float(row["exact_tok_s_per_user"]) >= 5 else "NO",
            )
            for row in network_rows
        ],
    )
    heterogeneity_rows = _csv(root / "network/heterogeneity-sensitivity.csv")
    heterogeneity_table = _table(
        ("Scenario", "tok/s", "Target ms", "Bottleneck amplification"),
        [
            (
                row["heterogeneity"],
                f"{float(row['exact_tok_s_per_user']):.3f}",
                f"{float(row['target_pass_ms']):.1f}",
                f"{float(row['bottleneck_amplification']):.3f}×",
            )
            for row in heterogeneity_rows
        ],
    )
    failures = summary["important_failures"]
    failure_text = "\n".join(f"- {value}" for value in failures)
    result_lines = [
        f"EXPERIMENT 019: {classification}",
        "",
        "# No-Monolith Kimi K3 Swarm",
        "",
        f"- Best provisional worker-event output: **{best['exact_tok_s_per_user']:.4f} tok/s/user — INADMISSIBLE because Gate 7 failed**",
        f"- Worker memory cap: **{best['memory_cap_gib']:.0f} GiB**",
        f"- Maximum actual accounted worker peak: **{best['max_worker_peak_gib']:.4f} GiB**",
        f"- Total physical workers: **{best['worker_count']}**",
        f"- Workers per locality pod: **{best['stripe_degree']}**",
        f"- Stripe degree / depth span: **{best['stripe_degree']} / {best['depth_span']}**",
        f"- Block / chunk: **{best['block']} / {best['chunk']}**",
        f"- Target pass: **{best['target_pass_ms']:.2f} ms**",
        f"- Full 93-layer sharded correctness: **{summary['full_93_sharded_correctness']}**",
        f"- Bottom-up full-target reconstruction error: **{summary['bottom_up_reconstruction_error_percent']:.2f}%**",
        f"- Maximum layer fraction owned by one headline worker: **{best['max_layer_fraction'] * 100:.3f}%**",
        f"- Maximum routed/shared expert fraction: **{best['max_expert_fraction'] * 100:.3f}%**",
        f"- Evidence class: **{evidence}**",
        "",
        "## Required truth table",
        "",
        _table(("Question", "Answer"), truth_rows),
        "",
        "## 1. Executive result",
        "",
        summary["executive_paragraph"],
        "",
        f"Gate 7 failed by a wide margin. The classification follows the declared gates mechanically: the {best['exact_tok_s_per_user']:.4f} figure is retained as diagnostic scheduler output, not reported as validated Swarm throughput. Gate 10 therefore also fails: a number above five cannot override failed serial reconstruction.",
        "",
        "![Best bounded-worker throughput](../../artifacts/experiment-019/charts/chart-01-swarm-throughput.png)",
        "",
        "The plotted tier frontier is provisional diagnostic output from the explicit-worker scheduler. Because the shared service model fails Gate 7, none of these tier points is an admissible throughput result; invalid placements are still omitted rather than plotted as zero.",
        "",
        "## 2. What Experiment 018 did and did not prove",
        "",
        "Experiment 018 physically proved streamed exact verification chunks, 32-way expert fragments, a 6.7022 tok/s/user independent-resource wavefront, and effective immutable AttnRes transport caching. Its winning compute resources were logical eight-layer microcells holding roughly 122–136 GiB, so it did not prove consumer-sized workers. Experiment 019 uses the E018 result only as historical comparison and as the coarse 2536.46 ms control; no E018 microcell or layer service is used as a headline compute resource.",
        "",
        "## 3. Exact Swarm thesis tested",
        "",
        "The tested claim is whether exact Kimi K3 target inference can be reconstructed entirely from memory-bounded sub-layer workers, with every synchronization and transfer charged, and still exceed 5 tok/s/user. The target remains the zero-draft target oracle: block 16 must finish in at most 3400 ms.",
        "",
        "## 4. Anti-monolith rules",
        "",
        "Pods contain topology only. They have no service-time, model-byte, throughput, or aggregate-GPU field. Compute events accept exactly one `worker_id` and `resource_type=microworker`; the trace assertion rejects `microcell`, `layer`, `stage`, and `aggregate_gpu`. P=2 is retained only as a physical control.",
        "",
        gate_table,
        "",
        "## 5. K3 checkpoint partition",
        "",
        f"The Safetensors census covers **{summary['checkpoint_tensor_count']:,} tensors** and **{summary['checkpoint_payload_bytes']:,} declared payload bytes** across 96 shards. Header offsets, shapes, dtypes, and index ownership agree byte-for-byte. The census includes embeddings, all 93 transformer layers (69 KDA and 24 Gated MLA), every routed expert 0–895 in each MoE layer, shared experts, routers, latent projections, AttnRes tensors, endpoint weights, and multimodal tensors.",
        "",
        "## 6. Worker memory envelopes",
        "",
        "Peak memory is static weights + recurrent/KV/AttnRes state + activations + scratch + reduction/network buffers + CUDA workspace + a measured/accounted allocator margin. Checkpoint bytes alone are never reported as peak memory.",
        "",
        tier_table,
        "",
        "![Memory caps and accounted peaks](../../artifacts/experiment-019/charts/chart-02-worker-memory.png)",
        "",
        "## 7. Complete worker placement",
        "",
        "Each tier has an immutable manifest. The detailed tensor-to-worker byte ranges live in `placement/tensor-coverage.csv`; each worker manifest points to its filter and records its layer range, expert stripe, attention heads, state owners, hashes, and full memory envelope. Total resident required weight bytes are reconciled against the original checkpoint payload.",
        "",
        "![Winning worker placement](../../artifacts/experiment-019/charts/chart-03-worker-placement.png)",
        "",
        "The diagram shows real workers, not an aggregate stage. A pod owns consecutive depth only because its member workers each own the matching stripe across those layers.",
        "",
        "## 8. Direct shard-loading audit",
        "",
        f"The combined physical audit contains **{summary['checkpoint_read_request_count']:,} reads**. Large headline tensors were read through assigned row or strided-column ranges; **{summary['direct_loader_large_full_materializations']}** prohibited large full-tensor materializations were observed. Reviewed whole reads are limited to tensors below 32 MiB and remain charged to an explicit owner.",
        "",
        "## 9. Expert-stripe bank design",
        "",
        "For P workers, worker p owns intermediate stripe p of all 896 experts. A worker receives one latent activation and ordered top-16 route metadata, evaluates all 16 local fragments, applies route weights locally, accumulates locally, and emits one latent partial. The primary arm therefore exposes P outputs, not 16×P RPCs.",
        "",
        "## 10. Expert-stripe physical result",
        "",
        summary["expert_result_paragraph"],
        "",
        "For chunk 2 at P=8, the worker ceiling is 0.8487 ms versus 2.9001 ms for the whole-expert control and 1.9609 ms after canonical-local broadcast/reduction. The negative-control per-expert networking is 16.2597 ms. Network coalescing works; kernel-launch coalescing does not yet.",
        "",
        "![Expert stripe comparison](../../artifacts/experiment-019/charts/chart-04-expert-stripe.png)",
        "",
        "The naive control uses the identical route workload and physical shard compute, but charges 16 network-visible expert outputs per worker. The gap isolates organization, not different routes or weights.",
        "",
        "## 11. KDA stripe design/result",
        "",
        summary["kda_result_paragraph"],
        "",
        "## 12. MLA stripe design/result",
        "",
        summary["mla_result_paragraph"],
        "",
        "## 13. Shared expert / LatentMoE result",
        "",
        "Latent-down is row-striped and followed by an explicit recursive-doubling all-gather. Expert outputs are reduced once in latent space. Latent-up is column-striped; each worker adds its full-hidden routed partial to its colocated intermediate-striped shared-expert partial before one hidden all-reduce. This removes a separate shared-expert monolith and a redundant layer reduction.",
        "",
        "## 14. Endpoint sharding",
        "",
        "Embedding and LM-head vocabulary rows are assigned across endpoint workers. Only the embedding owner of a requested token performs the lookup. Every LM-head worker produces its vocabulary-shard logits and one local candidate; a distributed exact max comparison returns the greedy token.",
        "",
        "## 15. Sharded-layer correctness",
        "",
        summary["layer_correctness_paragraph"],
        "",
        "![Physical sharded-layer DAG](../../artifacts/experiment-019/charts/chart-05-sharded-layer-dag.png)",
        "",
        "## 16. Multi-layer correctness",
        "",
        summary["depth_correctness_paragraph"],
        "",
        "## 17. Full 93-layer sharded correctness",
        "",
        summary["full_correctness_paragraph"],
        "",
        "## 18. Bottom-up timing validation",
        "",
        summary["serial_validation_paragraph"],
        "",
        "No correction multiplier is applied. Calibration-layer shard services are compared with held-out KDA/MLA layers, and the 93-layer serial reconstruction is compared with the historical exact target curve only after it is calculated.",
        "",
        "## 19. Worker-level event model",
        "",
        f"The winning trace contains **{best['total_task_count']:,} events**. It schedules explicit exclusive worker queues, directed link queues, recurrent state readiness, recursive-doubling latent all-gathers, binary-tree reductions, and concrete reduction-compute tasks. Total physical compute work is **{best['total_physical_compute_work_ms']:.1f} ms**; critical-path compute is **{best['critical_path_compute_ms']:.1f} ms** and critical-path network is **{best['network_critical_path_ms']:.1f} ms**.",
        "",
        "## 20. Wavefront over microworkers",
        "",
        "Chunks enter downstream layers as soon as the upstream worker reduction and state dependencies complete. Different pods may process different chunks concurrently, but no worker runs two tasks at once. The result is a scheduled wavefront, never `sum(layer_times)/N`.",
        "",
        "![Wavefront over actual workers](../../artifacts/experiment-019/charts/chart-06-wavefront-workers.png)",
        "",
        "## 21. Worker-memory sweep",
        "",
        tier_table,
        "",
        "![Worker cap versus throughput](../../artifacts/experiment-019/charts/chart-09-worker-cap-vs-throughput.png)",
        "",
        "## 22. Worker-count scaling",
        "",
        f"The winning Tier-A placement reserves **{best['worker_count']} workers**; its mean/p50/p95 utilizations are **{best['average_worker_utilization']:.1%} / {best['p50_worker_utilization']:.1%} / {best['p95_worker_utilization']:.1%}**, with **{best['peak_simultaneous_workers']}** workers active simultaneously at peak. Worker count is reported alongside throughput because capacity fan-out is not free.",
        "",
        "![Worker count versus throughput](../../artifacts/experiment-019/charts/chart-10-worker-count-vs-throughput.png)",
        "",
        "## 23. Network sensitivity",
        "",
        network_table,
        "",
        "![Network sensitivity](../../artifacts/experiment-019/charts/chart-08-network-sensitivity.png)",
        "",
        "## 24. Heterogeneity/stragglers",
        "",
        heterogeneity_table,
        "",
        summary["straggler_paragraph"],
        "",
        "## 25. Monolith tax",
        "",
        f"The provisional worker-event target pass minus the E018 coarse target pass is **{summary['monolith_tax_ms']:.2f} ms**. Because Gate 7 failed, this signed decomposition is diagnostic only and cannot establish a negative monolith tax. It is retained without clipping so the invalid model remains auditable.",
        "",
        "![Monolith tax](../../artifacts/experiment-019/charts/chart-07-monolith-tax.png)",
        "",
        "## 26. Final exact oracle",
        "",
        f"Block {best['block']} / chunk {best['chunk']} yields a **provisional, inadmissible** {best['target_pass_ms']:.2f} ms / {best['exact_tok_s_per_user']:.4f} tok/s/user scheduler output. It is not an exact throughput result because Gate 7 failed. Compute-slowdown sensitivity is reported at 1.0×, 1.5×, 2.0×, 2.5×, and 3.0×; none is labelled RTX 3090.",
        "",
        "## 27. Economics/resource efficiency",
        "",
        f"The provisional candidate reserves **{best['total_resident_gib']:.2f} GiB** of required model weights across the cluster and schedules **{best['worker_seconds_per_accepted_token']:.4f} active worker-seconds per accepted token**. Network traffic is **{best['network_mb_per_output_token']:.3f} MB per output token**. These are diagnostic resource quantities under an invalid timing model, not a cost claim. Capacity reservation and active compute remain separate columns in the economics receipt.",
        "",
        "![Economics and resource efficiency](../../artifacts/experiment-019/charts/chart-11-economics.png)",
        "",
        "## 28. What failed",
        "",
        failure_text,
        "",
        "## 29. What was retained",
        "",
        "The experiment retains exact routing, exact state progression, the E018 chunk wavefront principle, immutable AttnRes transport caching, persistent worker weights/state/buffers, and hierarchical control. It rejects E018 microcells as compute resources and rejects per-expert network RPC as the primary MoE organization.",
        "",
        "## 30. What this actually proves about Swarm Inference",
        "",
        summary["architectural_proof_paragraph"],
        "",
        "![Hard-gate evidence stack](../../artifacts/experiment-019/charts/chart-12-evidence-stack.png)",
        "",
        "## 31. What remains unproven",
        "",
        "No multi-GPU worker pod was physically assembled. Local/inter-pod link performance is shaped by an explicit deterministic model plus measured localhost protocol overhead, not measured on a distributed GPU cluster. RTX 3090 capacity fit is tested by the 20 GiB cap; RTX 3090 throughput is not physically tested. Consumer availability, failure recovery, long-context sustained behavior, and dollar cost of reserved idle capacity remain deployment questions.",
        "",
        "## 32. Recommendation for Experiment 020",
        "",
        summary["experiment_020_recommendation"],
        "",
        "## ARCHITECTURAL SWARM VERDICT",
        "",
        f"**{summary['architectural_verdict']}**",
        "",
        summary["architectural_verdict_paragraph"],
        "",
        "## PHYSICAL SWARM VERDICT",
        "",
        "**NOT YET PHYSICALLY PROVEN**",
        "",
        "A single RTX 5090 physically executed worker shards sequentially for correctness and service measurement. Multiple independent physical compute devices did not execute the distributed path, so architecture-model evidence must not be described as a measured swarm.",
        "",
    ]
    return "\n".join(result_lines)


__all__ = ["render_report"]
