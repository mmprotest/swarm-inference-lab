"""Render the answer-first Experiment 018 Markdown report from validated artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pct(value: Any) -> str:
    return f"{100 * float(value):.1f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    first = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([first, divider, *body])


def render_report(
    *, root: Path, artifact_root: Path, summary: Mapping[str, Any]
) -> str:
    headline = summary["headline"]
    best = summary["best_exact"]
    result_classes = summary["result_classes"]
    baseline_rows = _read_csv(artifact_root / "baseline/oracle-curve.csv")
    stage_rows = _read_csv(artifact_root / "wavefront/stage-balance.csv")
    sweep_rows = _read_csv(artifact_root / "wavefront/sweep.csv")
    micro_rows = _read_csv(artifact_root / "microshards/shard-size.csv")
    shard_equiv = _read_csv(artifact_root / "microshards/equivalence.csv")
    shard_network = _read_csv(artifact_root / "microshards/network-sensitivity.csv")
    control_rows = _read_csv(artifact_root / "control-plane/scaling.csv")
    economics = _read_csv(artifact_root / "economics/results.csv")
    future = _read_csv(artifact_root / "attnres/future-score-results.csv")
    validation = json.loads((artifact_root / "validation.json").read_text(encoding="utf-8"))
    chunk = json.loads((artifact_root / "dependencies/chunk-equivalence.json").read_text(encoding="utf-8"))
    failure = json.loads((artifact_root / "failure-log.json").read_text(encoding="utf-8"))
    audit = json.loads((artifact_root / "repeated-work-audit.json").read_text(encoding="utf-8"))
    sensible_profiles = sorted(
        {
            row["profile"]
            for row in shard_network
            if row.get("latency_sensible", "").lower() == "true"
        }
    )
    geographic_conclusion = (
        "Only " + ", ".join(sensible_profiles) + " beats the measured whole expert at least one tested shape."
        if sensible_profiles
        else "No tested shaped-link profile beats the measured whole expert at these M=1 microshard shapes; compatibility does not establish latency viability."
    )
    oracle = float(best["oracle_tok_s_per_user"])
    if oracle >= 5.0:
        experiment_019 = (
            "Move to a rented low-latency multi-GPU cluster with persistent 8-layer "
            "logical cells, physically overlap the winning chunk schedule, seed the "
            "immutable AttnRes cache, and compare the measured physical event trace "
            "against E018 prediction. Add real shard fanout/reduction only inside a "
            "low-RTT microcell and price those resources separately. The decisive "
            "falsification test is model prediction error on target-pass latency; do "
            "not reopen topology search or broaden into unrelated kernels first."
        )
    elif oracle >= 4.0:
        experiment_019 = (
            f"Attack only measured bottleneck microcell {headline['bottleneck_microcell']} "
            f"({headline['bottleneck_operator']}) while retaining the fixed topology and "
            "wavefront trace. Re-run the identical event model with a fresh physical "
            "service receipt; do not broaden the kernel or topology search."
        )
    elif float(summary["stage_balance_winner"]["max_over_mean"]) >= 1.25:
        experiment_019 = (
            f"Decompose/rebalance measured bottleneck microcell {headline['bottleneck_microcell']} "
            "within the fixed topology boundary, then rerun the same DAG. Do not reopen "
            "arbitrary topology search."
        )
    else:
        experiment_019 = (
            "Treat the wavefront thesis as falsified and do not add scheduler complexity; "
            "the exact dependencies or communication prevented useful overlap."
        )

    baseline_table = _table(
        ["block", "accepted", "historical ms", "fresh-reconciled ms", "deviation", "oracle"],
        [
            (
                row["verification_block"],
                row["accepted_rows"],
                f"{float(row['target_pass_ms']):.1f}",
                f"{float(next(item['reproduced_target_pass_ms'] for item in summary['baseline_reproduction']['rows'] if int(item['verification_block']) == int(row['verification_block']))):.1f}",
                f"{float(next(item['deviation_percent'] for item in summary['baseline_reproduction']['rows'] if int(item['verification_block']) == int(row['verification_block']))):+.2f}%",
                f"{float(row['oracle_tok_s_per_user']):.4f}",
            )
            for row in baseline_rows
        ],
    )
    balance_table = _table(
        ["chunk", "max ms", "mean ms", "median ms", "CV", "max/mean", "slowest", "ceiling rows/s"],
        [
            (
                row["chunk_rows"],
                f"{float(row['max_stage_time_ms']):.2f}",
                f"{float(row['mean_stage_time_ms']):.2f}",
                f"{float(row['median_stage_time_ms']):.2f}",
                f"{float(row['coefficient_of_variation']):.3f}",
                f"{float(row['max_over_mean']):.2f}",
                row["slowest_microcell"],
                f"{float(row['steady_state_ceiling_rows_per_second']):.2f}",
            )
            for row in stage_rows
        ],
    )
    best_by_block = []
    for block in (4, 7, 12, 16):
        row = max(
            (
                item
                for item in sweep_rows
                if item["result_class"] == "C_wavefront_attnres_cache"
                and int(item["block_candidates"]) == block
            ),
            key=lambda item: float(item["oracle_tok_s_per_user"]),
        )
        best_by_block.append(
            (
                block,
                row["chunk_size"],
                f"{float(row['total_ms']):.1f}",
                f"{float(row['oracle_tok_s_per_user']):.4f}",
                f"{float(row['speedup_vs_corresponding_serial']):.2f}x",
                _pct(row["pipeline_efficiency"]),
                _pct(row["median_steady_utilization"]),
                row["max_chunks_in_flight"],
            )
        )
    result_table = _table(
        ["result", "block", "chunk/split", "ms", "tok/s/user", "evidence"],
        [
            (
                "A serial",
                result_classes["A_serial_baseline"]["verification_block"],
                "—",
                f"{float(result_classes['A_serial_baseline']['target_pass_ms']):.1f}",
                f"{float(result_classes['A_serial_baseline']['oracle_tok_s_per_user']):.4f}",
                "historical exact",
            ),
            (
                "B wavefront",
                result_classes["B_wavefront_only"]["block_candidates"],
                result_classes["B_wavefront_only"]["chunk_size"],
                f"{float(result_classes['B_wavefront_only']['total_ms']):.1f}",
                f"{float(result_classes['B_wavefront_only']['oracle_tok_s_per_user']):.4f}",
                "independent-resource model",
            ),
            (
                "C + AttnRes cache",
                result_classes["C_wavefront_attnres_cache"]["block_candidates"],
                result_classes["C_wavefront_attnres_cache"]["chunk_size"],
                f"{float(result_classes['C_wavefront_attnres_cache']['total_ms']):.1f}",
                f"{float(result_classes['C_wavefront_attnres_cache']['oracle_tok_s_per_user']):.4f}",
                "independent-resource + shaped network",
            ),
            (
                "D + microshards",
                result_classes["D_wavefront_fine_microshards"]["block_candidates"],
                f"{result_classes['D_wavefront_fine_microshards']['chunk_size']}/{result_classes['D_wavefront_fine_microshards']['split_degree']}x",
                f"{float(result_classes['D_wavefront_fine_microshards']['total_ms']):.1f}",
                f"{float(result_classes['D_wavefront_fine_microshards']['oracle_tok_s_per_user']):.4f}",
                "additional independent shard resources",
            ),
            (
                "E retained repeated work",
                result_classes["E_wavefront_retained_repeated_work"]["block_candidates"],
                result_classes["E_wavefront_retained_repeated_work"]["chunk_size"],
                f"{float(result_classes['E_wavefront_retained_repeated_work']['total_ms']):.1f}",
                f"{float(result_classes['E_wavefront_retained_repeated_work']['oracle_tok_s_per_user']):.4f}",
                "no unmeasured speedup multiplication",
            ),
            (
                "F complete exact",
                best["block_candidates"],
                best["chunk_size"],
                f"{float(best['total_ms']):.1f}",
                f"{float(best['oracle_tok_s_per_user']):.4f}",
                "fixed-resource-accounted headline",
            ),
        ],
    )
    sweep_table = _table(
        ["block", "winning chunk", "target ms", "tok/s/user", "serial speedup", "efficiency", "median util.", "chunks in flight"],
        best_by_block,
    )
    micro_size_table = _table(
        ["split", "native max MiB", "runtime max MiB", "input B", "output B", "exact"],
        [
            (
                row["split_degree"],
                f"{int(row['native_checkpoint_bytes_per_worker_max']) / (1024**2):.3f}",
                f"{float(row['runtime_mib_per_worker_max']):.3f}",
                row["input_payload_bytes_per_worker"],
                row["output_contribution_bytes_per_worker"],
                next(
                    item["pass"]
                    for item in shard_equiv
                    if item["split_degree"] == row["split_degree"]
                    and item["batch_rows"] == "1"
                ),
            )
            for row in micro_rows
            if int(row["batch_rows"]) == 1
        ],
    )
    network_table = _table(
        ["profile", "RTT", "Gb/s", "8-way ms", "16-way ms", "32-way ms"],
        [
            (
                profile,
                next(row["rtt_ms"] for row in shard_network if row["profile"] == profile),
                next(row["bandwidth_gbps"] for row in shard_network if row["profile"] == profile),
                *(f"{float(next(row['critical_path_ms'] for row in shard_network if row['profile'] == profile and int(row['split_degree']) == degree and int(row['batch_rows']) == 1)):.2f}" for degree in (8, 16, 32)),
            )
            for profile in ("local_microcell", "fast_regional", "regional_wan_like", "adverse_public_wan")
        ],
    )
    control_table = _table(
        ["logical tasks", "physical launches", "coalescing", "coord. p50 ms", "local critical ms", "serial decisions", "critical waits"],
        [
            (
                row["logical_task_count"],
                row["physical_kernel_count"],
                f"{float(row['coalescing_ratio']):.2f}x",
                f"{float(row['coordinator_wall_p50_ms']):.4f}",
                f"{float(row['worker_local_critical_p50_ms']):.4f}",
                row["serial_scheduling_decisions"],
                row["critical_path_waits"],
            )
            for row in control_rows
        ],
    )
    economics_table = _table(
        ["architecture", "tok/s/user", "aggregate tok/s", "GPU-h/M", "USD/M", "vs E017 cost"],
        [
            (
                row["architecture"],
                f"{float(row['tok_s_per_user']):.4f}",
                f"{float(row['aggregate_tok_s']):.2f}",
                f"{float(row['gpu_hours_per_1m_output_tokens']):.2f}",
                f"{float(row['projected_usd_per_1m_output_tokens']):.2f}",
                f"{float(row['change_vs_e017_cost_percent']):+.1f}%",
            )
            for row in economics
        ],
    )
    gates_table = _table(
        ["hard gate", "result"],
        [(key.replace("_", " "), "PASS" if value else "FAIL") for key, value in summary["gates"].items()],
    )
    decision_table = _table(
        ["hypothesis", "implementation", "benchmark", "inspect result", "redesign / decision"],
        [
            (
                "Exact K3 positions can stream across depth",
                "ordered token x depth DAG with carried KDA/conv/MLA state",
                f"{chunk['case_count']} real-K3 monolithic-vs-chunk cases",
                "all exactness gates passed",
                "retain ordered release; forbid overtaking",
            ),
            (
                "Wavefront shortens the serial critical path",
                "deterministic independent-resource event engine",
                "4 blocks x legal chunks, canonical links",
                f"best wavefront-only speedup {float(result_classes['B_wavefront_only']['speedup_vs_corresponding_serial']):.2f}x",
                "retain measured slowest-stage schedule",
            ),
            (
                "Immutable AttnRes objects remove repeated transport",
                "version/hash cache with first-seed accounting",
                "all coarse block/chunk traces",
                f"total reduction {_pct(best['network']['total_attnres_reduction_fraction'])}",
                "retain; reject stale versions",
            ),
            (
                "Completed-block future scores save layer time",
                "exact precompute reference",
                "bit-exact score/output proof plus physical GPU bound",
                f"maximum full-layer bound {100 * max(float(row['full_layer_upper_bound_fraction']) for row in future):.2f}%",
                "reject until a measured faster full layer exists",
            ),
            (
                "Tiny native expert shards preserve the expert",
                "gate/up row + down-column slices and stable weighted sum",
                "real expert degrees 1/2/4/8/16/32",
                f"exact through {validation['microshard']['highest_exact_degree']}-way",
                "retain compatibility; do not infer one-GPU speedup",
            ),
            (
                "Fine shard fanout adds useful parallelism",
                "persistent shard resources + batched stable tree levels",
                "Model D canonical-link event sweep",
                f"best {float(result_classes['D_wavefront_fine_microshards']['oracle_tok_s_per_user']):.4f} tok/s/user",
                "keep separate from fixed-resource headline and economics",
            ),
            (
                "Hierarchical control avoids one wait per shard",
                "worker-local bucket/coalescing summaries",
                "32/128/512/1000/natural-count physical CPU sweep",
                f"serial scaling exponent {float(summary['control_plane']['log_log_scaling_exponent']):.3f}",
                "retain compact persistent-worker hot path",
            ),
        ],
    )
    audit_table = _table(
        ["operation", "exact cache?", "implemented", "upper bound", "measured gain"],
        [
            (
                row["operation"],
                row["exact_cache_possible"],
                row["implemented?"],
                row["estimated_upper_bound"],
                row["measured_gain"],
            )
            for row in audit
        ],
    )

    endpoint_anchor = "artifacts/experiment-014/oracle-full-93/serial-oracle-receipt.json"
    lines = [
        f"EXPERIMENT 018: {summary['outcome']}\n",
        "# Experiment 018: Wavefront Swarm Execution\n",
        f"- Best exact result: **{float(headline['tok_s_per_user']):.4f} tok/s/user**.\n",
        f"- Winning verification block: **{headline['block_candidates']} candidates / {headline['accepted_rows']} accepted rows**.\n",
        f"- Winning chunk size: **{headline['chunk_rows']} row(s)**.\n",
        f"- Target pass: **{float(headline['target_pass_ms']):.1f} ms**, a **{float(headline['speedup_vs_corresponding_serial']):.2f}x** speedup over the corresponding immutable serial baseline.\n",
        f"- Crossed 5 tok/s/user: **{'yes' if headline['five_tok_s_crossed'] else 'no'}**.\n",
        f"- Crossed separate 7.5 tok/s/user breakthrough threshold: **{'yes' if summary['breakthrough'] else 'no'}**.\n",
        f"- Wavefront pipeline efficiency: **{_pct(headline['pipeline_efficiency'])}**; median steady microcell utilization: **{_pct(headline['median_microcell_utilization'])}**.\n",
        f"- Bottleneck: **microcell {headline['bottleneck_microcell']}**, layers {headline['bottleneck_layers'][0]}-{headline['bottleneck_layers'][1]}, dominated by **{headline['bottleneck_operator']}**.\n",
        f"- AttnRes bytes: **{_pct(headline['attnres_total_byte_reduction_fraction'])} total reduction including first seed**, **{100 * float(headline['attnres_steady_state_byte_reduction_fraction']):.2f}% steady-state reduction**.\n",
        f"- Highest physically validated exact expert split: **{headline['highest_exact_microshard_degree_physical']}-way**.\n",
        "- Claim boundary: **VALIDATED INDEPENDENT-RESOURCE MODEL**, driven by physically measured real-K3 service on one RTX 5090 and explicit shaped communication. This is not a physical 12-GPU result and no physical WAN was measured.\n",
        "## 1. Executive result\n",
        f"The central hypothesis {'passed' if summary['outcome'] != 'FAIL' else 'failed'} its non-throughput architecture gates: exact chunks can cross depth boundaries without an all-block barrier, and the event trace yields an exact target critical path of {float(best['total_ms']):.1f} ms. The complete fixed-resource-accounted architecture is Result F (coarse wavefront plus immutable AttnRes transport cache); the fine microshard model remains separate because its additional resource economics have no pinned conversion. The result is {'also above the 7.5 breakthrough threshold' if summary['breakthrough'] else 'not classified as a separate breakthrough beyond the declared thresholds'}.\n",
        gates_table + "\n",
        "Each major arm followed `hypothesis -> implementation -> benchmark -> inspect result -> redesign`:\n",
        decision_table + "\n",
        "![Exact oracle progress](../../artifacts/experiment-018/charts/chart-01-oracle-progress.png)\n",
        "## 2. Primary goal\n",
        "The question was whether schedule-level parallelism—not another isolated faster operator—could reduce one exact Kimi K3 verification critical path from a sum of 12 microcell services toward fill/drain plus the bottleneck service period. The fixed 8-layer, 12-microcell topology was held constant. The primary threshold was 5.0 tok/s/user under the inherited canonical 0.25 ms/25 Gb/s internal and 5 ms/10 Gb/s coarse shaped links.\n",
        "## 3. Baseline reproduction\n",
        f"Fresh unchanged-code layer-89 KDA and layer-91 MLA measurements were reconciled against the historical exact denominator. Maximum curve deviation was {float(summary['baseline_reproduction']['maximum_absolute_deviation_percent']):.2f}%, inside the fixed ±3% gate. The historical curve—not the favorable rerun—remains the denominator. Routes, expert-major output, recurrent state prefix, and geometry were exact in the fresh receipt. Initial dirty state contained only the pre-existing `third_party/colibri` submodule modification.\n",
        baseline_table + "\n",
        "Evidence: [`baseline/oracle-curve.csv`](../../artifacts/experiment-018/baseline/oracle-curve.csv), [`physical/baseline-reproduction.json`](../../artifacts/experiment-018/physical/baseline-reproduction.json), and [`source-manifest.json`](../../artifacts/experiment-018/source-manifest.json).\n",
        "## 4. Dependency analysis\n",
        "The exact DAG has two dependency dimensions: same-chunk depth handoff and same-cell previous-chunk state. KDA recurrence and causal convolution require ordered chunk arrival at each layer; MLA requires its compressed-KV/RoPE cache through the prior position. Routing, experts, shared experts, LatentMoE reduction, residual updates, and endpoint work are row-local once their incoming hidden/state objects exist. Completed AttnRes snapshots are immutable for a request/block/chunk version and may be seeded downstream. Therefore a chunk may leave microcell N after N has updated all local recurrent/cache state for that chunk; it need not wait for the rest of the verification block. Overtaking within a stateful layer is prohibited.\n",
        _table(
            ["operator", "cross-position dependency", "release condition"],
            [
                ("KDA + causal conv", "recurrent matrix + width-4 windows", "state update committed"),
                ("Gated MLA", "compressed KV/RoPE cache", "KV appended and causal result complete"),
                ("routing / experts / LatentMoE", "none beyond incoming row", "stable routed/shared reduction complete"),
                ("AttnRes", "immutable versioned snapshots", "snapshot/reference ready"),
                ("residual / endpoint", "same-row hidden and snapshots", "residual or logits complete"),
            ],
        ) + "\n",
        "The first-class task objects carry request, block, chunk, cell, dependency IDs, input/state references, and versioned output references. Full proof: [`dependencies/k3-wavefront-dag.json`](../../artifacts/experiment-018/dependencies/k3-wavefront-dag.json).\n",
        "## 5. Exact chunk-equivalence proof\n",
        f"The physical proof swept monolithic blocks 4/7/12/16 against chunk sizes 1/2/4/8 where legal on real K3 snapshot-KDA, later KDA, and MLA layers. All {chunk['case_count']} cases passed the predeclared {float(chunk['threshold_fixed_before_results']):.1e} relative-L2 gate; operation order that remained unchanged was bit-identical, and recurrent/conv/MLA state fingerprints matched as recorded. The full-layer calls include routing, routed/shared experts, LatentMoE, residuals, and AttnRes. The inherited exact 93-layer endpoint/logits and greedy-token anchor remains `{endpoint_anchor}`; this experiment did not pretend the partial physical layer sweep was a new full-model generation.\n",
        _table(
            ["attention type", "physical layer", "cases", "all pass", "max output relative L2"],
            [
                (name, value["layer"], value["cases"], value["all_pass"], f"{float(value['maximum_output_relative_l2']):.3e}")
                for name, value in chunk["by_attention_type"].items()
            ],
        ) + "\n",
        "Evidence: [`dependencies/chunk-equivalence.json`](../../artifacts/experiment-018/dependencies/chunk-equivalence.json).\n",
        "## 6. Physical microcell measurements\n",
        "Each of the 12 fixed cells received fresh service inputs from two real checkpoint layers matching its local KDA/MLA composition, across rows 1/2/4/8, with p50/p90/p99 wall, CUDA, host, routing, expert, attention, LatentMoE, AttnRes/residual, state-update and output-preparation phases. The inherited runtime co-measures recurrent/cache state update with attention and AttnRes with residual; these are labeled joint measurements rather than falsely separated. Dense layer 0 and LM-head edges use identified inherited E014 physical p50s. One global factor per row size anchors the sum to the immutable exact serial compute after subtracting inherited topology transport; it does not flatten measured cell imbalance. Repeated checkpoint loading is separately timed and excluded from resident compute service. Because a single full routed layer materializes roughly 18 GiB, several composed resident microcells exceed 32 GiB and require the intended sub-layer distribution; one RTX 5090 was never treated as 12 independent devices.\n",
        "Evidence: [`physical/microcell-service.csv`](../../artifacts/experiment-018/physical/microcell-service.csv), [`physical/layer-service.csv`](../../artifacts/experiment-018/physical/layer-service.csv), and [`physical/gpu-samples.csv`](../../artifacts/experiment-018/physical/gpu-samples.csv).\n",
        "## 7. Stage balance\n",
        balance_table + "\n",
        f"At the winning chunk size, the slowest stage is cell {headline['bottleneck_microcell']}. The steady-state ceiling shown above is rows divided by that measured/anchored maximum service before network. This stage, rather than the mean, drives the event model.\n",
        "![Stage service and utilization](../../artifacts/experiment-018/charts/chart-04-stage-utilization.png)\n",
        "## 8. Wavefront scheduler implementation\n",
        "The deterministic engine schedules from explicit dependencies and independent resource IDs. For `(cell, chunk)`, start is the maximum of previous chunk finish on that cell, same-chunk arrival from the prior cell, local state readiness, and required AttnRes object readiness. Compute, handoff, seed, reference, reduction and control events each occupy an explicit resource and duration; resource contention adds a predecessor to the realized critical path. The engine emits every event, resource utilization, simulation CPU time, and the realized critical predecessor chain. Persistent logical workers have bounded request state, long-lived queues and caches, duplicate/loss/stale-version rejection, deterministic retry/restart and cleanup.\n",
        "## 9. Wavefront results\n",
        result_table + "\n",
        "Result D is not used to inflate Result F. It models extra independently scheduled expert shards and stable tree reductions, while Result F retains fixed-resource accounting and only co-executed effects. No speedups were multiplied after the fact.\n",
        sweep_table + "\n",
        "## 10. Pipeline fill/drain analysis\n",
        f"For the winning configuration, fill was {float(best['fill_ms']):.1f} ms, steady stage period {float(best['steady_stage_period_ms']):.1f} ms, and drain {float(best['drain_ms']):.1f} ms. Pipeline efficiency is defined exactly as balanced pipeline latency from the mean measured stage compute service for each actual chunk, excluding handoff, divided by observed event-DAG makespan including communication and control. The decomposition in `wavefront/sweep.csv` separately records balanced fill/drain, stage-imbalance loss, explicit shaped communication, state barriers and measured control setup. Recurrent state forces chunk order within each cell but does not restore an all-block depth barrier. Median steady utilization was {_pct(best['median_steady_utilization'])} against the 70% target; pipeline efficiency was {_pct(best['pipeline_efficiency'])} against the 65% target. Diagnosis: **{summary['targets']['diagnosis']}**. The best configuration satisfying both utilization targets was block {summary['targets']['best_utilization_compliant_configuration']['block_candidates']} / chunk {summary['targets']['best_utilization_compliant_configuration']['chunk_size']}: {float(summary['targets']['best_utilization_compliant_configuration']['oracle_tok_s_per_user']):.4f} tok/s/user at {_pct(summary['targets']['best_utilization_compliant_configuration']['median_steady_utilization'])} median steady utilization and {_pct(summary['targets']['best_utilization_compliant_configuration']['pipeline_efficiency'])} efficiency. Thus the throughput conclusion does not depend on retaining a utilization-target miss.\n",
        "![Wavefront timeline](../../artifacts/experiment-018/charts/chart-02-wavefront-gantt.png)\n",
        "## 11. Critical-path analysis\n",
        f"Useful parallelism is defined as total compute work divided by compute-equivalent work on the realized critical path; it is **{float(best['useful_parallelism']):.2f}x**. Critical-path fraction is event-DAG makespan divided by the sum of serial component latencies; it is **{float(best['critical_path_fraction']):.3f}**. The winning trace sent {best['messages']} messages ({float(best['messages_per_accepted_token']):.2f} per accepted token) but incurred only {best['serial_waits']} realized critical-path waits ({float(best['serial_waits_per_accepted_token']):.2f} per accepted token): messages are not automatically serialized waits. Critical-path communication was {float(best['critical_path_communication_ms']):.1f} ms and explicitly included.\n",
        "For Chart 03, the immutable corresponding serial target-pass latency is allocated across cells in proportion to the measured/anchored compute work; the wavefront line uses actual event finishes. This preserves the historical denominator while making the depth comparison readable.\n",
        "![Critical path](../../artifacts/experiment-018/charts/chart-03-critical-path.png)\n",
        "Full chain: [`wavefront/critical-path.json`](../../artifacts/experiment-018/wavefront/critical-path.json).\n",
        "## 12. AttnRes immutable-object cache\n",
        f"The cache keys completed snapshots by object type, request/model scope, version and content hash. The producer seeds each downstream path once; later boundary messages carry changing hidden rows plus compact IDs. The complete accounting includes {best['network']['cache_seed_bytes']} seed bytes, {best['network']['cache_reference_bytes']} reference bytes, {best['network']['cache_hits']} hits, {best['network']['cache_misses']} misses and {best['network']['invalidations']} invalidations. Total completed-AttnRes traffic fell {_pct(best['network']['total_attnres_reduction_fraction'])}, and steady repeated traffic fell {100 * float(best['network']['steady_state_attnres_reduction_fraction']):.2f}%; the first seed is not omitted. Output semantics are unchanged because stale versions are rejected.\n",
        "![AttnRes bytes](../../artifacts/experiment-018/charts/chart-05-attnres-bytes.png)\n",
        "## 13. AttnRes future-score cache\n",
        f"For an immutable completed block `v_b`, `depth_query[l] · RMSNorm(v_b)` is independent of future token state, so computing it at block creation is algebraically identical. The reference precomputed {future[0]['future_query_count']} future query rows and kept prefix score, softmax order and accumulation unchanged; cached scores and final AttnRes outputs were bit-identical. The physical GPU experiment measured an upper bound of {100 * max(float(row['full_layer_upper_bound_fraction']) for row in future):.2f}% of a real layer. This is an upper bound, not a cached kernel measurement, so the arm was not retained and contributes 0 ms to Result E/F.\n",
        "## 14. Repeated-work audit\n",
        audit_table + "\n",
        f"The audit rejected mutable route metadata and low-bound work rather than optimizing it for appearance. Only the transport cache and already-persistent static runtime state survive. The optional tiny-M audit reused the pinned E017 real K3 M about 1-8 comparison: native fused MXFP4 was {float(summary['optional_tiny_m_kernel_audit']['fused_speedup']):.2f}x faster than its exact unfused control, but it was already the canonical E016/E017 path, so its incremental E018 gain is zero; no other compatible installed backend existed.\n",
        "![Repeated-work audit](../../artifacts/experiment-018/charts/chart-09-repeated-work.png)\n",
        "## 15. Expert microshard correctness\n",
        f"Real layer-89 expert 885 native MXFP4 tensors were physically sliced at degrees 1/2/4/8/16/32. Gate and up rows and matching down columns were uploaded as independent tensor objects; no degree >1 worker owned the full expert or another shard's activation. Stable FP32 reconstruction and route-weight-before-reduction reconstruction passed the fixed {float(validation['microshard']['threshold']):.1e} relative-L2 gate for all row sizes. The highest validated degree is {validation['microshard']['highest_exact_degree']}; 64-way is incompatible with the native 32-value scale grouping because 3072/64=48 would split groups.\n",
        micro_size_table + "\n",
        "![Microshard size](../../artifacts/experiment-018/charts/chart-06-microshard-size.png)\n",
        "## 16. Expert microshard economics\n",
        f"At 32-way and M=1, the worker holds {summary['microshard_economics']['runtime_weight_bytes_per_worker']} runtime bytes ({summary['microshard_economics']['runtime_weight_bytes_per_worker'] / (1024**2):.3f} MiB), below the 1 MiB target. Input is {summary['microshard_economics']['input_bytes']} bytes, output contribution {summary['microshard_economics']['output_bytes']} bytes, and gate/up intermediates remain local. A routed layer creates 512 logical shard contributions per token, or {summary['microshard_economics']['logical_tasks_per_92_layer_token']} across 92 routed layers before batching. Against the measured M=1 whole expert, the canonical local link leaves {float(summary['microshard_economics']['break_even_worker_latency_ms_raw']):.3f} ms raw compute margin ({float(summary['microshard_economics']['break_even_worker_latency_ms_feasible']):.3f} ms feasible); a non-positive raw value means fanout/reduction latency alone loses. Marketplace price and availability are unpinned, so economic viability is explicitly unproven.\n",
        "## 17. Fine-grain wavefront model\n",
        f"Model D fans actual top-16 routes to persistent `(layer, expert, shard)` resources, applies route weights before a stable expert-ID/shard-ID tree, batches each reduction level, and rejoins non-expert work. Its immutable route fixture contains three real positions for every routed layer and is cycled for longer blocks; it is an exact execution fixture, not a claim about the population route distribution. Its best exact event result was {float(result_classes['D_wavefront_fine_microshards']['oracle_tok_s_per_user']):.4f} tok/s/user at {float(result_classes['D_wavefront_fine_microshards']['total_ms']):.1f} ms. Physical one-GPU shard timings establish service and size only; independent shard overlap is modeled, never claimed as measured. The difference from Model C shows whether fine fanout/reduction communication helps or harms rather than assuming a 32x speedup.\n",
        "## 18. Control-plane scaling\n",
        control_table + "\n",
        f"The measured serial coordinator log-log exponent was {float(summary['control_plane']['log_log_scaling_exponent']):.3f}, below 1. At 1000 tasks there were {summary['control_plane']['thousand_task_critical_waits']} critical waits, not 1000, and {summary['control_plane']['thousand_task_coordinator_p50_ms']:.4f} ms serial coordinator p50. Counts through 1000 use expanded worker-local task fixtures; the natural full count is {summary['control_plane']['natural_logical_task_count']} and uses exact prebucketed persistent-worker counters, because creating every shard object centrally would itself violate the architecture. Fixture construction is measured and reported separately but excluded from planner critical time: in the architecture those counters arise while local routing groups assignments, not through central task creation. The offline event simulator still enumerates every event to make the scientific trace auditable; that simulation CPU time is reported separately and is not asserted to be the runtime control path. Worker-local bucketing coalesces tasks sharing operation, shape, dtype, expert, weight shard and route bucket; logical-task and physical-launch counts are separately recorded. The CPU-only planner has no meaningful compute-idle interval; per-worker idle milliseconds are instead recorded from each wavefront event trace as makespan times one minus stage utilization.\n",
        "![Control-plane scaling](../../artifacts/experiment-018/charts/chart-08-control-plane.png)\n",
        "## 19. Network sensitivity\n",
        network_table + "\n",
        f"RTT dominates the stable reduction depth well before bandwidth does. {geographic_conclusion} Regional and public-WAN-like links are therefore rejected as tensor-level microshard boundaries. The coarse sweep separately covers inherited coarse RTT 1/5/10/20/50/100 ms and bandwidth 0.1/1/10 Gb/s in `wavefront/network-sensitivity.csv`; the headline stays at the inherited 5 ms/10 Gb/s setting.\n",
        "![Microshard network sensitivity](../../artifacts/experiment-018/charts/chart-07-network-sensitivity.png)\n",
        "## 20. Correctness\n",
        "Chunk legality passed real KDA, convolution, MLA, routing, experts, shared experts, LatentMoE, AttnRes and residual boundaries through the physical full-layer runtime. Routes/expert order and recurrent/conv/MLA state fingerprints are in the physical receipt. Microshard and route-weighted stable reductions pass their fixed tolerance. Future-score caching is bit-identical but unretained. The event schedule changes inter-cell timing, not numerical operation order within a stateful layer. Endpoint/logits and greedy-token availability are anchored to the inherited exact full-93 reference; no threshold was weakened after observing results.\n",
        "## 21. Failure/recovery behavior\n",
        f"The first physical sweep stopped after {failure['failures'][0]['inspection']} when an external telemetry sampler violated the CUDA PREPARE quiescence contract. The redesign delayed sampling until READY and resumed by atomic immutable layer key; no incomplete-layer measurement was retained, while five already-complete layer receipts were preserved. A later one-hour shell watchdog expired after layer 83; the same atomic rule resumed the final layers, and one obsolete-path retry failed before loading any state. The first future-score extractor also failed before timing because it looked for materialized score tensors; inspection of the real runtime showed that K3 constructs each score query as residual-norm times residual-projection, which the retained extractor now reproduces from checkpoint factors. Fault tests cover worker task failure, lost and duplicate chunks, stale cache objects, reduction-child failure, scheduler retry, worker restart and failed-request cleanup. These are bounded abstraction tests, not a claim of complete distributed fault tolerance.\n",
        "## 22. Final oracle\n",
        result_table + "\n",
        f"The exact complete headline is **{float(best['oracle_tok_s_per_user']):.4f} tok/s/user**, block {best['block_candidates']}, chunk {best['chunk_size']}, at **{float(best['total_ms']):.1f} ms**. It is a **{float(best['speedup_vs_corresponding_serial']):.2f}x** critical-path reduction over the corresponding serial block. The result class is **{summary['outcome']}** and the claim class remains **VALIDATED INDEPENDENT-RESOURCE MODEL + SHAPED NETWORK**.\n",
        "## 23. Economics\n",
        economics_table + "\n",
        "The repository's inherited $0.15/GPU-hour assumption has no pinned market-snapshot date and is not presented as current pricing. The E018 row uses 44.4863908 retained user slots and 93 paid GPU equivalents, exactly as inherited. Fine microshard resources are excluded from the complete economics because no honest equivalent-price conversion is pinned.\n",
        "![System economics](../../artifacts/experiment-018/charts/chart-10-economics.png)\n",
        "## 24. What failed\n",
        "The future-score arm did not earn retention: exact algebra was proven, but only an upper bound—not a measured faster full layer—was available. Worldwide tensor-level microsharding failed the latency-sensitivity test because reduction RTT compounds by tree depth. Sequential execution of 32 shards on one RTX 5090 did not and could not establish parallel speedup. The physical telemetry design failed once under the PREPARE contract and was corrected before completing the service sweep. Any fine-grain result that consumes additional unpriced independent resources was barred from inflating the fixed-resource Result F.\n",
        "## 25. What was retained\n",
        "Retained mechanisms are the dependency-derived deterministic wavefront, persistent worker/request state, explicit resource contention and communication events, the immutable AttnRes object transport cache, hierarchical local scheduling/coalescing, stable weighted expert reductions, and exact 32-way native microshard compatibility. The historical serial denominator and fixed topology remain unchanged.\n",
        "## 26. Implications for Swarm Inference\n",
        f"Wavefront execution changes the relevant scaling quantity: critical-path fraction fell to {float(best['critical_path_fraction']):.3f} while useful parallelism rose to {float(best['useful_parallelism']):.2f}x. This validates pipelining across independently resident depth resources. Tiny expert workers are mathematically and physically representable below 1 MiB native material at 32-way, but only low-latency local groups are credible and their marketplace economics remain unmeasured. Pipeline validity and tiny-worker economics are therefore separate conclusions.\n",
        "## 27. Exact recommendation for Experiment 019\n",
        experiment_019 + "\n",
        "---\n",
        "**Can wavefront execution make the current Swarm architecture a credible path to >=5 Kimi K3 tok/s/user while remaining compatible with tiny sub-layer workers?**\n",
        f"**{summary['final_answer']}**\n",
        f"The measured evidence is {float(best['oracle_tok_s_per_user']):.4f} tok/s/user at {float(best['total_ms']):.1f} ms for block {best['block_candidates']} / chunk {best['chunk_size']}, a {float(best['speedup_vs_corresponding_serial']):.2f}x reduction over serial, with {_pct(best['pipeline_efficiency'])} pipeline efficiency and {_pct(best['network']['total_attnres_reduction_fraction'])} total AttnRes-byte reduction. All real-K3 chunk cases and physical 32-way expert slices passed exactness gates. The throughput result still assumes independent resident microcells and shaped links rather than a physical multi-GPU run, and the 32-way worker marketplace cost remains unpinned; that is the remaining boundary between architectural credibility and deployed proof.\n",
    ]
    return "\n".join(lines)


__all__ = ["render_report"]
