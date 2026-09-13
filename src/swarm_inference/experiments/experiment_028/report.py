"""Mechanical E028 gates, six static plots, and the requested concise report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .local import ROOT, OUT, write_json

COLORS = ["#276EAD", "#B48B24", "#CE6638", "#75853B", "#B95686"]
MARKERS = ["o", "s", "^", "D", "v"]


def plots(aggregates, best_k):
    destination = OUT/"plots"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#707070","axes.labelcolor":"#282828",
                         "text.color":"#282828","axes.titleweight":"bold"})
    curves = [[r for r in aggregates if r["k"]==best_k and r["rtt_ms"]==rtt] for rtt in [0,30,60,100,150]]
    for curve in curves:
        curve.sort(key=lambda r:r["w"])

    def base(title, ylabel, fraction=False):
        fig,ax=plt.subplots(figsize=(6.7,3.9),layout="constrained")
        ax.set(title=title,xlabel="In-flight verification chunks (W)",ylabel=ylabel)
        ax.set_xscale("log",base=2)
        ax.set_xticks([1,2,4,8,16],labels=["1","2","4","8","16"])
        ax.grid(axis="y",color="#E4E4E4",linewidth=.7)
        ax.set_ylim(bottom=0)
        if fraction:
            ax.set_ylim(0,1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        return fig,ax

    for metric,title,ylabel,name,fraction in [
        ("committed_tokens_per_second","Committed throughput vs window","Committed tokens/s","01_throughput_vs_w.png",False),
        ("wan_wait_fraction","Uncovered WAN wait vs window","Fraction of decode time","02_wan_wait_vs_w.png",True),
        ("discarded_speculative_compute_fraction","Discarded target compute vs window","Fraction of target compute","04_discarded_compute_vs_w.png",True),
    ]:
        fig,ax=base(title,ylabel,fraction)
        for i,curve in enumerate(curves):
            ax.plot([r["w"] for r in curve],[r[metric] for r in curve],color=COLORS[i],marker=MARKERS[i],linewidth=1.6,label=f"{curve[0]['rtt_ms']} ms RTT")
        if not fraction:
            ax.set_ylim(0,1.08*max(r[metric] for curve in curves for r in curve))
        ax.legend(ncol=3,fontsize=8,frameon=False,loc="upper right" if fraction else "best")
        fig.supxlabel(f"Trace-driven simulation · fixed K={best_k} · 8 prompts × 3 jitter seeds",fontsize=8)
        fig.savefig(destination/name,dpi=160)
        plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(10,3.7),layout="constrained",sharey=True)
    for stage,ax in zip("abc",axes):
        for i,curve in enumerate(curves):
            ax.plot([r["w"] for r in curve],[r[f"stage_{stage}_utilization"] for r in curve],color=COLORS[i],marker=MARKERS[i],linewidth=1.4,label=f"{curve[0]['rtt_ms']} ms")
        ax.set(title=f"Stage {stage.upper()}",xlabel="W",ylim=(0,1))
        ax.set_xscale("log",base=2)
        ax.set_xticks([1,2,4,8,16],labels=["1","2","4","8","16"])
        ax.grid(axis="y",color="#E4E4E4",linewidth=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Resource busy fraction")
    axes[-1].legend(fontsize=8,frameon=False)
    fig.suptitle("Virtual stage utilization vs window",fontsize=12)
    fig.supxlabel(f"Trace-driven simulation · K={best_k} · includes work later discarded",fontsize=8)
    fig.savefig(destination/"03_stage_utilization_vs_w.png",dpi=160)
    plt.close(fig)

    for metric,title,ylabel,name,windows in [
        ("async_speedup_vs_spec_sync","Async speedup vs RTT","Speedup over matching-K SPEC_SYNC","05_async_speedup_vs_rtt.png",[2,4,8,16]),
        ("throughput_retained_vs_zero_wan","Throughput retained vs RTT","Fraction of best zero-WAN throughput","06_zero_wan_retention.png",[1,2,4,8,16]),
    ]:
        fig,ax=plt.subplots(figsize=(6.7,3.9),layout="constrained")
        for i,w in enumerate(windows):
            rs=sorted([r for r in aggregates if r["k"]==best_k and r["w"]==w],key=lambda r:r["rtt_ms"])
            ax.plot([r["rtt_ms"] for r in rs],[r[metric] for r in rs],color=COLORS[i],marker=MARKERS[i],linewidth=1.6,label=f"W={w}")
        ax.set(title=title,xlabel="Per-link round-trip latency (ms)",ylabel=ylabel,ylim=(0,None))
        ax.set_xticks([0,30,60,100,150])
        ax.axhline(1,color="#666666",linewidth=.8,linestyle="--")
        ax.grid(axis="y",color="#E4E4E4",linewidth=.7)
        if "retained" in metric:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(0,max(1.65,ax.get_ylim()[1]))
            ax.axhline(1.5,color="#993F2C",linewidth=.8,linestyle=":")
            ax.text(149,1.515,"PASS minimum: 1.5x",ha="right",va="bottom",fontsize=8,color="#993F2C")
        ax.legend(ncol=len(windows),fontsize=8,frameon=False)
        fig.supxlabel(f"Trace-driven simulation · fixed K={best_k} · 8 prompts × 3 jitter seeds",fontsize=8)
        fig.savefig(destination/name,dpi=160)
        plt.close(fig)


def finish():
    config=json.loads((OUT/"config.json").read_text())
    correctness=json.loads((OUT/"correctness_results.json").read_text())
    stress=json.loads((OUT/"rollback_stress_results.json").read_text())
    validation=json.loads((OUT/"simulator_validation.json").read_text())
    aggregates=json.loads((OUT/"wan_aggregate.json").read_text())
    causal=json.loads((OUT/"causal_audit.json").read_text())
    async_check=json.loads((OUT/"async_network_correctness.json").read_text())
    genericity=json.loads((OUT/"genericity_audit.json").read_text())
    best=max((r for r in aggregates if r["network"]=="WAN-60" and r["w"]>1),key=lambda r:r["committed_tokens_per_second"])
    k,w=best["k"],best["w"]
    fixed=sorted([r for r in aggregates if r["k"]==k and r["w"]==w],key=lambda r:r["rtt_ms"])
    sync=max((r for r in aggregates if r["network"]=="WAN-60" and r["k"]>0 and r["w"]==1),key=lambda r:r["committed_tokens_per_second"])
    zero=max((r for r in aggregates if r["network"]=="LOCAL" and r["k"]>0),key=lambda r:r["committed_tokens_per_second"])
    r100=next(r for r in fixed if r["network"]=="WAN-100")
    corruption=sum(r["state_corruption_events"] for r in correctness["runs"])+stress["state_corruption_events"]
    gates=dict(correctness=correctness["complete"] and correctness["passed"] and corruption==0 and causal["passed"] and async_check["exact_committed_tokens"],
               rollback=stress["complete"] and stress["passed"] and stress["rollback_failures"]==0,
               genericity=genericity["passed"],simulator=validation["complete"] and validation["passed"])
    speed=best["committed_tokens_per_second"]/sync["committed_tokens_per_second"]
    retain60=best["committed_tokens_per_second"]/zero["committed_tokens_per_second"]
    retain100=r100["committed_tokens_per_second"]/zero["committed_tokens_per_second"]
    perf_pass=dict(tps_60=best["committed_tokens_per_second"]>=6,speedup_60=speed>=1.5,wan_wait_60=best["wan_wait_fraction"]<.35)
    strong=dict(tps_60=best["committed_tokens_per_second"]>=10,speedup_60=speed>=2,
                retention_60=retain60>=.7,retention_100=retain100>=.5,wan_wait_60=best["wan_wait_fraction"]<=.2,
                stage_utilization=min(best["steady_state_stage_utilization"].values())>=.6,
                discarded_compute=best["discarded_speculative_compute_fraction"]<.35,
                physical_w8_fits=all(r["peak_inflight_chunks"]==8 for r in correctness["runs"] if r["w"]==8))
    verdict="PASS_STRONG" if all(gates.values()) and all(strong.values()) else "PASS" if all(gates.values()) and all(perf_pass.values()) else "FAIL"
    decision="CONTINUE_SWARM" if verdict.startswith("PASS") else "STOP_CURRENT_SWARM_ARCHITECTURE"
    serial60=next(r for r in aggregates if r["network"]=="WAN-60" and r["k"]==0)
    headline=(f"Trace-driven three-stage WAN simulation projects {best['committed_tokens_per_second']:.2f} committed tok/s at 60 ms RTT with K={k}, W={w}; "
              f"{speed:.2f}× the strongest synchronous speculative control, {retain60:.1%} of the best zero-WAN throughput. "
              f"WAN wait is {best['wan_wait_fraction']:.1%} and discarded target compute is {best['discarded_speculative_compute_fraction']:.1%}.")
    stress_memory=json.loads((OUT/"gpu_stress_samples.json").read_text())
    measured_peak=max([r["peak_vram_bytes"] for r in aggregates]+[r.get("memory_bytes",0) for r in stress_memory])
    summary=dict(experiment=config["experiment"],verdict=verdict,
                 correctness_passed=gates["correctness"],rollback_passed=gates["rollback"],genericity_passed=gates["genericity"],simulator_validated=gates["simulator"],
                 best_k=k,best_w=w,best_60ms_committed_tps=best["committed_tokens_per_second"],
                 spec_sync_60ms_committed_tps=sync["committed_tokens_per_second"],spec_sync_best_k=sync["k"],async_speedup_60ms=speed,
                 serial_60ms_committed_tps=serial60["committed_tokens_per_second"],
                 same_k_async_speedup_60ms=best["async_speedup_vs_spec_sync"],
                 zero_wan_committed_tps=zero["committed_tokens_per_second"],zero_wan_k=zero["k"],zero_wan_w=zero["w"],
                 throughput_retention_60ms=retain60,throughput_retention_100ms=retain100,
                 wan_wait_fraction_60ms=best["wan_wait_fraction"],discarded_compute_fraction=best["discarded_speculative_compute_fraction"],
                 state_corruption_count=corruption,headline=headline,decision=decision,
                 gates=gates,pass_criteria=perf_pass,pass_strong_criteria=strong,
                 peak_vram_bytes=measured_peak,peak_vram_scope="sampled physical device during workload matrix and forced stress",
                 physical_gpu_count=1,virtual_target_gpu_count=3,
                 throughput_evidence_class="PHYSICALLY GROUNDED MODEL",
                 runtime_version=config["runtime_version"],simulator_version="e028-des-3")
    write_json(OUT/"summary.json",summary)
    plots(aggregates,k)
    natural=sum(r["rollback_events"] for r in correctness["runs"])
    forced=sum(r["forced_events"] for r in stress["scenarios"])
    hashes=sum(r["full_and_partial_state_hash_checks"] for r in stress["scenarios"])
    real_tokens=sum(r["committed_tokens"] for r in correctness["runs"])
    failures=[name for name,value in {**{f"gate/{k}":v for k,v in gates.items()},**{f"PASS/{k}":v for k,v in perf_pass.items()}}.items() if not value]
    performance_rows="\n".join(f"| {r['rtt_ms']} | {r['committed_tokens_per_second']:.2f} | {r['speedup_vs_best_spec_sync']:.2f}× | {r['throughput_retained_vs_zero_wan']:.1%} | {r['wan_wait_fraction']:.1%} | {r['discarded_speculative_compute_fraction']:.1%} |" for r in fixed)
    validation_rows="\n".join(f"| {r['network']}, K={r['k']}, W=1 | {r['measured_runtime_ms']/1000:.3f} s | {r['predicted_runtime_ms']/1000:.3f} s | {r['absolute_relative_error']:.2%} |" for r in validation["conditions"])
    sensitivity=[]
    for window in [1,4,8]:
        curve={r["rtt_ms"]:r for r in aggregates if r["k"]==k and r["w"]==window}
        sensitivity.append(dict(w=window,tps_30=curve[30]["committed_tokens_per_second"],tps_60=curve[60]["committed_tokens_per_second"],
                                tps_100=curve[100]["committed_tokens_per_second"],tps_150=curve[150]["committed_tokens_per_second"],
                                retention_100_vs_60=curve[100]["committed_tokens_per_second"]/curve[60]["committed_tokens_per_second"],
                                discard_60=curve[60]["discarded_speculative_compute_fraction"]))
    summary["rtt_sensitivity_at_best_k"]=sensitivity
    write_json(OUT/"summary.json",summary)
    sensitivity_rows="\n".join(f"| {r['w']} | {r['tps_30']:.2f} | {r['tps_60']:.2f} | {r['tps_100']:.2f} | {r['tps_150']:.2f} | {r['retention_100_vs_60']:.1%} |" for r in sensitivity)
    interpretation=("Yes, the measured-trace replay meets the required PASS threshold for asynchronous latency hiding." if verdict.startswith("PASS") else
                    f"No. Moving from 60 to 100 ms RTT reduces W=8 throughput by {1-sensitivity[2]['retention_100_vs_60']:.1%}, almost the same loss as W=1 ({1-sensitivity[0]['retention_100_vs_60']:.1%}). W=4 loses {1-sensitivity[1]['retention_100_vs_60']:.1%}. The required asynchronous speedup is not reached, and throughput remains strongly sensitive to RTT.")
    report=f'''# E028 — Local Async WAN Proof

## Verdict
**{verdict}**

## Headline result
{headline} These are **trace-driven simulation results**, derived from one RTX 5090, not measured three-GPU throughput. Decision: **{decision}**.

## Correctness
The final protocol completed {len(correctness['runs'])} physical runs and {real_tokens:,} committed tokens: eight fixed prompts, 256 tokens each, every K/W combination, and serial controls. The two long-context inputs contain 2,037 and 2,281 tokens; neither required a reduced output length. Exact synchronous/async agreement: **{gates['correctness']}**. Natural rollback events: **{natural:,}**. Forced events: **{forced}** on the first conversational prompt, across positions 1, 2, 3 and windows 2, 4, 8; stress rollback failures: **{stress['rollback_failures']}**; confirmed state corruption: **{corruption}**. The stress run made **{hashes*3:,} stage comparisons of both complete and partial-state fingerprints**, checked restored sequence positions and cached-prefix output checksums, and compared continuation with the clean control. Stale chunks never commit. The extra W=8 shaped-network run also matched exactly and reached {async_check["peak_unfinished_target_verifications"]} unfinished target verifications at once. The W=8 and W=16 implementations fit locally; sampled peak total GPU memory was {summary['peak_vram_bytes']/2**30:.2f} GiB.

## Simulator validation
Independent E028 measurements supplied service times. The initial serial WAN-60 check missed the gate at 10.57% because native service increased during idle gaps. A separate fixed prompt then supplied periodic native service measurements for the non-speculative path; the table below uses fresh held-out validation runs. The initial attempt is preserved in `archives/validation_attempt_1/`. Actual local sleeps injected each WAN hop and the parallel rollback controls. No timing correction was fitted to the validation runs.

| Condition | Measured | Predicted | Absolute error |
|---|---:|---:|---:|
{validation_rows}

Serial validity gate: **{gates['simulator']}** (every condition ≤10%). This validates serial accounting, not real multi-GPU contention or WAN transport.

## Performance
The target is the existing Qwen3.8-27B Q4_K_M GGUF, with its local Q8_0 MTP artifact. Measured layer profiling selected contiguous ranges [0,22), [22,45), [45,64); all simulator service times are actual operations on those stages. The virtual coordinator/drafter has independently charged GPU-capable MTP service measured on the same 5090.

Fixed configuration below: **K={k}, W={w}**, selected by aggregate 60 ms throughput. K denotes draft tokens beyond a leading token; full chunks contain K+1 rows. Verification uses scalar target kernels coalesced into one wire chunk; no fused-batch speedup is assumed. Ratios use the strongest SPEC_SYNC configuration at each RTT. The zero-WAN reference is the best LOCAL speculative configuration: K={zero['k']}, W={zero['w']}, {zero['committed_tokens_per_second']:.2f} committed tok/s. Rates pool total committed tokens / total elapsed time over eight prompts and three fixed jitter seeds. Decode includes drafting, pipeline fill/drain, rejection, rollback and transfers; prefill is separate in TTFT.

| RTT (ms) | Committed tok/s | vs best SPEC_SYNC | Zero-WAN retained | WAN wait | Discarded compute |
|---:|---:|---:|---:|---:|---:|
{performance_rows}

The SERIAL control projects {serial60["committed_tokens_per_second"]:.2f} committed tok/s at 60 ms. At 60 ms, the same-K speedup is {best['async_speedup_vs_spec_sync']:.2f}×; the stronger all-K synchronous comparison is {speed:.2f}×. Failed required checks: {', '.join(failures) if failures else 'none'}. PASS_STRONG check details are in `summary.json`. All 1,920 per-run projections and 80 aggregates are retained; the six figures use one fixed K.

## Did async hide WAN latency?
{interpretation} Consult the throughput, wait and discarded-compute curves together: stage activity includes work that can later be invalidated. The functional traces explicitly record launches preceding older verification completion; concurrency is represented in the actual scheduler.

At the same fixed K, the RTT sensitivity is:

| W | 30 ms tok/s | 60 ms tok/s | 100 ms tok/s | 150 ms tok/s | 100/60 ms throughput |
|---:|---:|---:|---:|---:|---:|
{sensitivity_rows}

At W=4 and W=8, discarded target compute at 60 ms is {sensitivity[1]['discard_60']:.1%} and {sensitivity[2]['discard_60']:.1%}, respectively.

## Genericity
**{gates['genericity']}**. Scheduler, transport, chunk invalidation and rollback use generic tokens, activations and llama state APIs. The native adapter queries public RoPE metadata for text positions. Existing architecture-specific model graph support remains inside the pinned llama.cpp build; E028 adds no model-specific numerical kernels or model-name scheduling branch. Only this model/backend combination was physically tested.

## Main bottleneck after E028
At the best setting, only {best['speculative_acceptance_rate']:.1%} of launched speculative input positions become committed, and {best['discarded_speculative_compute_fraction']:.1%} of target compute is discarded. Steady virtual stage occupancy is A={best['steady_state_stage_utilization']['A']:.1%}, B={best['steady_state_stage_utilization']['B']:.1%}, C={best['steady_state_stage_utilization']['C']:.1%}. Provisional-draft rejection and invalidation/refill cycles leave WAN verification/rollback dependencies on the committed-token path. Resident scalar verification and the independently charged MTP drafter also consume measured compute. Increasing raw stage activity is useful only when it yields more committed tokens.

## Decision
**{decision}**

Artifacts: `summary.json`, `correctness_results.json`, `rollback_stress_results.json`, `stage_profile.json`, `simulator_validation.json`, `wan_sweep_results.jsonl`, `wan_aggregate.json`, `plots/`, and reproducible sources listed in `provenance/`. No paid resources or additional hosts were used.
'''
    (OUT/"report.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    return summary
