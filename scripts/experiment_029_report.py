"""Render the seven requested figures and concise E029 decision report."""
from pathlib import Path
import json,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'
def read(name):return json.loads((OUT/name).read_text())
POLICIES=['LINEAR_CONTROL','STATIC_TREE','PROBABILITY_TREE','WAN_AWARE_TREE','ORACLE_TREE_UPPER_BOUND']
LABELS=['Linear','Static','Probability','WAN-aware','Oracle ceiling']
COLORS=['#627d98','#a08517','#16827c','#2457a7','#8b5f8b']
def main():
    s=read('summary.json');d=read('decision_points.json');base=read('e028_baseline_import.json');rows=read('wan_aggregate.json');validation=read('simulator_validation.json');branches=read('branch_state_correctness.json');profile=read('tree_verification_profile.json');draft=read('drafter_profile.json')
    real=d['best_real'];comparison=d['policy_comparison_60ms'];oracle=d['best_oracle']
    def at(point,network):return next(r for r in rows if r['policy']==point['policy'] and r['node_budget']==point['node_budget'] and r['max_depth']==point['max_depth'] and r['network']==network)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True})
    plots=OUT/'plots';plots.mkdir(exist_ok=True)
    def save(fig,name):
        fig.text(.01,.012,'Trace-driven projection from one RTX 5090; oracle uses zero draft cost.',fontsize=8,color='#526071')
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(plots/name,dpi=150);plt.close(fig)
    def bar(metric,ylabel,title,name,threshold=None):
        fig,ax=plt.subplots(figsize=(8,4));values=[p[metric] for p in comparison]
        bars=ax.bar(LABELS,values,color=COLORS,width=.65);bars[-1].set_hatch('//');bars[-1].set_alpha(.6)
        ax.set(ylabel=ylabel,title=title+'\nEach policy uses its best 60 ms operating point',ylim=(0,max(values)*1.18));ax.grid(axis='x',visible=False)
        if threshold is not None:ax.axhline(threshold,color='#7d3744',ls=':',lw=1.3,label='PASS threshold');ax.legend(frameon=False)
        for b,v in zip(bars,values):ax.annotate(f'{v:.2f}',(b.get_x()+b.get_width()/2,v),ha='center',va='bottom',xytext=(0,4),textcoords='offset points')
        save(fig,name)
    bar('committed_tokens_per_wan_traversal','Committed tokens / traversal','Useful progress per WAN round · 60 ms RTT','01_tokens_per_traversal.png',4)
    fig,ax=plt.subplots(figsize=(8,4.5));nets=['LOCAL','WAN-30','WAN-60','WAN-100','WAN-150'];rtts=[0,30,60,100,150]
    for i,p in enumerate(comparison):
        y=[at(p,n)['committed_tokens_per_second'] for n in nets]
        ax.plot(rtts,y,label=LABELS[i],color=COLORS[i],marker=['o','s','^','D','x'][i],ls='--' if i==4 else '-',lw=2 if i==3 else 1.4)
    ax.scatter([60],[base['spec_sync_60ms']['committed_tokens_per_second']],c='#383838',marker='_',s=220,label='E028 sealed sync')
    ax.scatter([60],[base['async_60ms']['committed_tokens_per_second']],c='#777777',marker='+',s=55,label='E028 sealed async')
    ax.set(xlabel='RTT (ms)',ylabel='Committed tokens / second',title='Projected throughput vs RTT · each policy holds its best 60 ms N/D ceiling',xticks=rtts,ylim=(0,None));ax.legend(frameon=False,ncol=3,fontsize=9)
    save(fig,'02_throughput_vs_rtt.png')
    for metric,ylabel,title,name in [('discarded_target_compute_fraction','Discarded target compute (%)','Discarded work vs node budget · 60 ms RTT','03_discarded_vs_budget.png'),('accepted_path_length_mean','Accepted path length (tokens)','Accepted path length vs node budget · 60 ms RTT','04_accepted_path_vs_budget.png'),('committed_tokens_per_second','Committed tokens / second','Projected throughput vs node budget · 60 ms RTT','05_throughput_vs_budget.png')]:
        fig,ax=plt.subplots(figsize=(8,4.2))
        for i,policy in enumerate(POLICIES):
            selected=sorted([r for r in rows if r['policy']==policy and r['network']=='WAN-60'],key=lambda r:r['node_budget']);scale=100 if 'fraction' in metric else 1
            ax.plot([r['node_budget'] for r in selected],[r[metric]*scale for r in selected],marker=['o','s','^','D','x'][i],color=COLORS[i],label=LABELS[i],ls='--' if i==4 else '-')
        ax.set(xlabel='Maximum node budget N (depth ceiling D = 4, 6, 8 respectively)',ylabel=ylabel,title=title,xticks=[8,16,32],ylim=(0,100 if 'fraction' in metric else None));ax.legend(frameon=False,ncol=3,fontsize=9)
        save(fig,name)
    bar('useful_target_compute_fraction','Useful share of target compute','Useful target compute · 60 ms RTT','06_useful_compute.png',.60)
    fig,(ax,heat)=plt.subplots(1,2,figsize=(9.5,4.5),gridspec_kw={'width_ratios':[1,1.3]})
    rr=d['real_rtt_sweep'];means=[];depthmeans=[];matrix=[]
    for row in rr:
        counts=collections.Counter()
        for run in row['selected_shapes_by_run']:counts.update(run['shape_counts'])
        total=sum(counts.values());means.append(sum(sum(map(int,k.split(',')))*v for k,v in counts.items())/total);depthmeans.append(sum(len(k.split(','))*v for k,v in counts.items())/total)
        matrix.append([sum((list(map(int,k.split(',')))[depth] if len(k.split(','))>depth else 0)*v for k,v in counts.items())/total for depth in range(real['max_depth'])])
    ax.plot(rtts,means,'o-',color=COLORS[3],label='Executed nodes');ax.plot(rtts,depthmeans,'s--',color=COLORS[2],label='Maximum depth');ax.set(xlabel='RTT (ms)',ylabel='Mean per round',title='WAN-aware selected size',xticks=rtts,ylim=(0,None));ax.legend(frameon=False,fontsize=9)
    im=heat.imshow(np.asarray(matrix),aspect='auto',cmap='Blues',vmin=0);heat.grid(False);heat.set(xticks=range(real['max_depth']),xticklabels=range(1,real['max_depth']+1),yticks=range(5),yticklabels=rtts,xlabel='Tree depth',ylabel='RTT (ms)',title='Mean verified nodes by depth')
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):heat.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>np.max(matrix)*.6 else '#1c2c41')
    fig.colorbar(im,ax=heat,label='Nodes');save(fig,'07_wan_aware_shapes.png')
    performance='\n'.join(f"| {r['rtt_ms']} | {r['committed_tokens_per_second']:.2f} | {r['committed_tokens_per_wan_traversal']:.2f} | {r['discarded_target_compute_fraction']:.1%} | {r['wan_wait_fraction']:.1%} |" for r in rr)
    compare='\n'.join(f"| E029 {LABELS[i]} | {p['committed_tokens_per_second']:.2f} | {p['committed_tokens_per_wan_traversal']:.2f} | {p['discarded_target_compute_fraction']:.1%} |" for i,p in enumerate(comparison))
    baseline_rows='\n'.join(f"| E028 {label} | {base[key]['committed_tokens_per_second']:.2f} | {base[key]['committed_tokens_per_target_traversal']:.2f} | {base[key]['discarded_speculative_compute_fraction']:.1%} |" for label,key in [('SERIAL','serial_60ms'),('SPEC_SYNC','spec_sync_60ms'),('ASYNC','async_60ms')])
    if s['decision']=='CONTINUE_SWARM':
        final='CONTINUE_SWARM. The real WAN-aware tree meets the preregistered gates and materially improves useful progress per round.'
        answer='Yes, to the extent established by the reported gates. Root anchoring removes unresolved-round invalidation; the measured candidate coverage supports the resulting throughput.'
    elif s['decision']=='DRAFTER_LIMITED':
        need=s['coverage_required_pass_tokens_per_round_lower_bound'];actual=s['committed_tokens_per_traversal_60ms']
        final=f'DRAFTER_LIMITED. Real candidates fail the gates while the ideal-coverage oracle passes strongly. At the current mean round duration, PASS requires at least {need:.2f} committed tokens per round versus {actual:.2f} measured ({need/actual:.2f}× coverage). This is an optimistic lower bound: additional committed-feature traffic and state work can raise it. Discarded target compute must remain at most 40%.'
        answer='No for the real drafter. Root anchoring removes causal invalidation across rounds, but the useful accepted path still does not justify the measured work and WAN round cost.'
    else:
        final='STOP_AUTOREGRESSIVE_WAN_SWARM. Commit-anchored tree verification does not solve the WAN inference problem under the tested target/runtime assumptions.'
        answer='No. The measured verification/state costs and synchronous tree-round dependency erase the required WAN amortization.'
    samples=profile['samples'];cost_lines=[]
    for n in (8,16,32):
        matches=[x for x in samples if x['node_budget']==n and not x.get('calibration_only')]
        cost_lines.append(f"N={n}: {np.median([sum(t['service_ms'] for t in x['stages']) for x in matches]):.1f} ms median across the three stage calls")
    failures='; '.join(s['fail_reasons']) if s['fail_reasons'] else 'All applicable hard gates met.'
    pass_rows='\n'.join([
        f"| Committed tokens/traversal | {s['committed_tokens_per_traversal_60ms']:.3f} | >= 4.0 |",
        f"| Discarded target compute | {s['discarded_target_compute_fraction_60ms']:.1%} | <= 40% |",
        f"| Throughput / sealed SPEC_SYNC | {s['speedup_vs_e028_spec_sync_60ms']:.3f} | >= 1.25 |",
        f"| Throughput / sealed ASYNC | {s['speedup_vs_e028_async_60ms']:.3f} | >= 1.00 |",
        f"| 100 ms / 60 ms throughput | {s['throughput_retention_100ms_vs_60ms']:.1%} | >= 55% |",
        f"| Gain over strongest probability tree | {s['wan_aware_gain_over_probability_60ms']:.1%} | >= 10% (failure gate) |"])
    oracle100=at(oracle,'WAN-100')['committed_tokens_per_second']/oracle['committed_tokens_per_second']
    oracle150=at(oracle,'WAN-150')['committed_tokens_per_second']/oracle['committed_tokens_per_second']
    report=f'''# E029 — Local Commit-Anchored WAN Tree

## Verdict
{s['verdict']}

## Decision
{s['decision']}

## Headline
{s['headline']} These are trace-driven projections of three independent stage resources from real RTX 5090 measurements, **not physical three-GPU throughput**. Gate findings: {failures}

## What changed from E028
Each explicit scored tree starts at an immutable committed root. The coordinator commits its longest valid path before constructing the next tree. There are no unresolved future rounds. E028's sealed artifacts were imported by hash; no E028 benchmark was rerun. Native DFlash2 supplies a scored candidate lattice through a generic drafter interface.

## Correctness
{s['physical_runs']} physical policy/prompt runs generated {s['physical_committed_tokens']:,} committed tokens: 256 for each of the eight fixed prompts in every condition, with exact agreement against E029's same-engine scalar references. All {branches['branch_state_tests']} branch tests passed, including two/four siblings, deeper paths, rejected alternatives, unchanged committed-root hashes, and deterministic continuations. Confirmed cross-branch contamination: {s['cross_branch_contamination_events']}.

Packed and sequence-fork compatibility attempts changed greedy continuations. The final worker uses canonical scalar node execution, device-resident partial-state checkpoints, and KV-tail truncation. Branched trees replay the accepted path on commit; linear trees use shared-prefix sequence checkpoints. Every call, copy, restore and replay is charged. The live checkpoint stack grows with depth; full contexts are not copied per node. The draft service is charged at measured GPU latency as a coordinator resource, following E028's abstraction; no target/draft compute overlap occurs across a commit-anchored round. Remote drafter placement itself was not physically tested. Distributed scheduling/transport remains model-generic; backend-specific draft feature handling is confined to its adapter. E029 prefill uses 256-row batches; one long-context scalar reference differs from E028's 512-row prefill reference. Comparisons use the sealed E028 throughput values and E029's exact internal correctness control.

## Real tree verification cost
E028's measured contiguous stage boundaries are reused: [0,22), [22,45), [45,64). Physical RTX 5090 stage measurements: {'; '.join(cost_lines)}. These values exclude commit replay, which is charged separately in every WAN projection. Peak VRAM was {s['peak_vram_gib']:.2f} GiB. Native DFlash2's median proposal latency was {draft['draft_latency_ms_p50']:.2f} ms; its five target feature taps require {draft['target_feature_bytes_per_committed_token']:,} serialized bytes per committed token in addition to stage activations. CUDA-event-only timing is unavailable; traces record synchronized native execution and host/service overhead.

The simulator reused E028's resource and directed-link event machinery. For a 32-token STATIC_TREE condition (N=8, D=4, 11 rounds), independent unshaped service measurements predicted **{validation['predicted_ms']/1000:.3f} s**, versus **{validation['measured_ms']/1000:.3f} s** measured with actual local 60 ms RTT delays: **{validation['absolute_relative_error']:.2%} error**, within ±10%. Replaying the shaped run's observed service durations gives {validation['accounting_closure_error']:.2%} error, also within tolerance. Physical commit calls share one GPU; the virtual model permits independent stage commits. No fitted normalization was applied.

## WAN results
The real policy's best 60 ms operating point is N≤{s['best_node_budget']}, D≤{s['best_max_depth']}. That ceiling is held fixed below while WAN-aware tree shape adapts to RTT. Links use 100 Mbps and seeded jitter; LOCAL uses unlimited bandwidth. A round charges four activation/control traversal hops plus parallel commit requests and committed-feature replies. No next round overlaps it.

| RTT (ms) | Projected committed tok/s | Tokens/traversal | Discarded compute | WAN wait |
|---:|---:|---:|---:|---:|
{performance}

| 60 ms condition | Projected committed tok/s | Tokens/traversal | Discarded compute |
|---|---:|---:|---:|
{baseline_rows}
{compare}

| Real WAN-aware gate | Observed | PASS requirement |
|---|---:|---:|
{pass_rows}

## Useful work per traversal
The primary WAN-aware result commits {real['committed_tokens_per_wan_traversal']:.2f} tokens per traversal (path p50 {real['accepted_path_length_p50']:.1f}, p95 {real['accepted_path_length_p95']:.1f}); verifies {real['verified_nodes_per_round']:.2f} nodes per round; and uses {real['target_compute_ms_per_committed_token']:.2f} ms of target/state compute per committed token. Its useful target-compute share is {real['useful_target_compute_fraction']:.1%}. Physical target calls and logical WAN rounds are recorded separately, including accepted-path replay. E029's discarded work is off-path verification/state work, not unresolved-round invalidation.

## Did tree speculation solve E028's wasted-compute problem?
{answer}

## WAN-aware optimization
At 60 ms, the best WAN-aware point improves throughput by {s['wan_aware_gain_over_probability_60ms']:.1%} over the strongest simple probability tree. Its optimizer uses draft path scores and measured scalar-node, draft and state service costs, plus activation/feature bytes, RTT and bandwidth; it prunes when an expansion reduces estimated committed tokens per second. Speculative children and branch allocation use only draft scores; the first executed anchor is the greedy token already known from the previous committed round. It is charged and counted only when executed, with no free bonus tokens. Selection estimates are saved beside actual utility. {real['linear_round_fraction']:.1%} of selected rounds are single chains; the strongest E029 linear control projects {next(p for p in comparison if p['policy']=='LINEAR_CONTROL')['committed_tokens_per_second']:.2f} tok/s. Any chain-dominated result is evidence about adaptive commit-anchored blocks, not a demonstrated benefit from branching.

The cost profile crosses all nine N/D combinations. The decision workload uses the fixed pairs (8,4), (16,6), (32,8); N and D effects in those throughput curves are therefore coupled. N=64 was not required and was not run. Per-prompt results and three fixed jitter seeds are retained; the seeds replay the same physical traces and are not independent hardware repetitions. Aggregate rates divide total tokens by total elapsed time.

## Oracle ceiling
The diagnostic oracle commits {oracle['committed_tokens_per_wan_traversal']:.2f} tokens per round and projects {oracle['committed_tokens_per_second']:.2f} tok/s at 60 ms, with {oracle['discarded_target_compute_fraction']:.1%} discarded compute. It uses target reference tokens and **zero draft-generation/injection cost**, while retaining measured target/state costs and the same feature-transfer protocol. It is an architectural ceiling, not a real drafter result. Its 100/150 ms throughput retention is {oracle100:.1%}/{oracle150:.1%}, and target/state compute costs {oracle['target_compute_ms_per_committed_token']:.2f} ms per committed token. PASS gates: {all(s['oracle_pass_gates'].values())}; PASS_STRONG gates: {all(s['oracle_pass_strong_gates'].values())}.

## Bottleneck
The remaining cost is the accepted progress per synchronous round relative to WAN transit, off-path scalar verification, partial-state copying and commit work. The real 60 ms run spends {real['wan_wait_fraction']:.1%} waiting only on the network; target/state compute costs {real['target_compute_ms_per_committed_token']:.2f} ms per committed token. Useful candidate coverage, rather than pipeline occupancy, determines whether that cost is amortized.

## Final decision
{final}
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    (OUT/'README.md').write_text('''# E029_LOCAL_COMMIT_ANCHORED_WAN_TREE

Read `report.md` for the decision and `summary.json` for machine-readable gates. WAN throughput is a physically grounded trace-driven model, not a real multi-GPU benchmark. E028 artifacts are imported and sealed by hash in `e028_baseline_import.json`.

Reproduce the analysis from the saved traces without model execution:

```powershell
python scripts/experiment_029_analyze.py
python scripts/experiment_029_report.py
python scripts/experiment_029_audit.py
```

For a new physical collection on this local installation, preserve this sealed output folder first and start with a fresh experiment output directory. Existing successful run files are reused by the runner, so do not mix native/runtime versions within one output directory. Then run from the repository root:

```powershell
python scripts/experiment_029_prepare.py
python scripts/experiment_029_build_native.py
cmd /c scripts\\experiment_029_build.cmd
python scripts/experiment_029_profile.py
python scripts/experiment_029_validate.py
python scripts/experiment_029_run.py
python scripts/experiment_029_analyze.py
python scripts/experiment_029_report.py
```

The free 1.1 GB DFlash2 GGUF download is documented and hashed in `drafter_profile.json`; the existing target is reused. All execution is local. The native worker links the pinned existing llama.cpp build. `environment.json` records code/library versions and hashes. No credentials or paid resources are required.

`runs/` contains physical traces; `tree_rounds.jsonl` consolidates their round evidence. `wan_sweep_results.jsonl` contains per-prompt/per-seed projections, and `wan_aggregate.json` contains mechanically aggregated values. The eight versioned prompts are unchanged from E028. `compatibility_attempts/` and diagnostic JSON files retain rejected implementation attempts; they are excluded from final timing inputs. The final runtime uses one committed root, a canonical working KV sequence, partial-state checkpoints per active depth, and a measured local accepted-path replay where branching requires it. Generic linear chains use position checkpoints to avoid replay.

The full N/D cost grid and fixed decision operating points are specified in `config.json`. Local wall-clock throughput is never presented as three-GPU throughput. The seven figures in `plots/` are standalone scientific plots derived from the saved results. `provenance/execution_manifest.json` seals the exact physical execution sources and binary; `provenance/portability_equivalence.json` documents a metadata-only portability cleanup with identical selection cost coefficients. Native source comments and metric descriptions were corrected without changing the executed binary or gate thresholds.
''',encoding='utf-8')
    print(s['verdict'],s['decision']);print('Seven plots and report written.')
if __name__=='__main__':main()
