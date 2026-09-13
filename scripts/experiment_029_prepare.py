"""Preregister E029 and import sealed E028 evidence, without running E028."""
from pathlib import Path
import hashlib, json, shutil, datetime, platform, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'
BASE = ROOT / 'experiments/E028_LOCAL_ASYNC_WAN_PROOF'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b=read(BASE/'config.json'); summary=read(BASE/'summary.json'); rows=read(BASE/'wan_aggregate.json')
    wanted=['summary.json','wan_aggregate.json','stage_profile.json','simulator_validation.json','environment.json','prompts.json','config.json']
    seals={f:hashlib.sha256((BASE/f).read_bytes()).hexdigest() for f in wanted}
    sync=max((r for r in rows if r['network']=='WAN-60' and r['condition']=='SPEC_SYNC'),key=lambda r:r['committed_tokens_per_second'])
    asynchronous=max((r for r in rows if r['network']=='WAN-60' and r['condition']=='SPEC_ASYNC'),key=lambda r:r['committed_tokens_per_second'])
    serial=next(r for r in rows if r['network']=='WAN-60' and r['condition']=='SERIAL')
    write(OUT/'e028_baseline_import.json',dict(source=str(BASE.relative_to(ROOT)),sha256=seals,
        imported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary=summary,
        spec_sync_60ms=sync,async_60ms=asynchronous,serial_60ms=serial,
        spec_sync_target_compute_per_committed_token_ms=sync['target_compute_ms']/sync['committed_tokens'],
        rtt_sweep=rows,stage_profile=read(BASE/'stage_profile.json'),rerun=False))
    shutil.copyfile(BASE/'prompts.json',OUT/'prompts.json')
    config=dict(experiment=OUT.name,seed=290913,model_path=b['model_path'],model_sha256=b['model_sha256'],
        llama_commit=b['llama_commit'],draft_model_path='.runtime/experiment-029/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf',
        draft_model_sha256='1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd',
        fallback_draft_model_path=b['draft_model_path'],stage_ranges=[[0,22],[22,45],[45,64]],
        context_tokens=8192,prefill_batch_tokens=256,branch_slots=34,
        node_budgets=[8,16,32],max_depths=[4,6,8],optional_64=False,
        committed_tokens_per_prompt=256,network_profiles=b['network_profiles'],network_seeds=[290913,290914,290915],
        policies=['LINEAR_CONTROL','STATIC_TREE','PROBABILITY_TREE','WAN_AWARE_TREE','ORACLE_TREE_UPPER_BOUND'],
        immutable_root=True,max_uncommitted_rounds=1,
        hypothesis='A commit-anchored scored tree increases useful committed tokens per WAN round without E028 causal invalidation.',
        budget_definition='Maximum executed candidate nodes including the known greedy anchor; all nodes and its compute are charged. D counts executed nodes on a root-to-leaf path. DFlash2 proposes seven further positions after the anchor.',
        measurement_classes={'local':'PHYSICAL single RTX 5090', 'validation':'SHAPED NETWORK on the same physical runtime', 'wan':'PHYSICALLY GROUNDED MODEL: three independent virtual stages'},
        metrics={'discarded_target_compute_fraction':'Measured scalar-node synchronized decode plus attributed fork/restore time for off-path nodes, divided by all target/state compute including accepted-path commit replay. Shared ancestor state costs are allocated to direct children. All costs also enter wall time; CUDA-event-only timing is unavailable.',
            'committed_tokens_per_wan_traversal':'Actual executed accepted nodes / completed tree rounds; no bonus token counted until executed.',
            'wan_wait_fraction':'Fraction of decode wall time occupied only by network waits, with no resource service.',
            'throughput':'Total committed tokens / total decode time, excludes one-time model load and prefill consistently with E028.'},
        preregistered_criteria={'simulator_error_max':0.10,
            'pass_strong':{'commits_min':5,'discard_max':0.30,'speedup_sync_min':1.50,'speedup_async_min':1.25,'retention100_min':0.65,'retention150_min':0.45,'target_compute_ratio_max':1.5,'peak_vram_gib_max':32},
            'pass':{'commits_min':4,'discard_max':0.40,'speedup_sync_min':1.25,'speedup_async_min':1.0,'retention100_min':0.55},
            'fail_any':['correctness failure','genericity failure','simulator validity failure','commits < 3','discard > 0.50','tps60 < E028 spec sync','WAN-aware gain over probability < 10%','oracle fails PASS efficiency/throughput','no PASS criteria satisfied'],
            'decisions':['CONTINUE_SWARM if real WAN-aware passes','DRAFTER_LIMITED if real fails and oracle passes strongly','STOP_AUTOREGRESSIVE_WAN_SWARM if both fail']})
    config.update(
        verification_mode='Canonical DFS scalar nodes with device-resident partial-state checkpoints, KV-tail truncation, and measured accepted-path replay; generic linear-chain checkpoint fast path',
        verification_deviation='Packed multi-sequence and sequence-fork candidates changed long-run greedy continuation. The final verifier uses one canonical working KV sequence and generic partial-state checkpoints; all copies, restores and replay calls are charged.',
        workload_matrix='Scaling profile crosses all 3 N and 3 D. Decision workload uses three preregistered paired operating points (8,4), (16,6), (32,8), all policies and all network profiles; linear uses min(N,D) executed nodes. This avoids a large hyperparameter search.',
        runtime_version='canonical-dfs-device-partial-linear-cow-v2',
        checkpoint_design='At most one partial-state checkpoint per active depth; no full-context duplication per node. Native sequence slots 1..8 identify reusable device checkpoint storage; slot 9 anchors a linear root. All actual node execution uses sequence 0.',
        linear_state_strategy='Shared-prefix sequence checkpoints for a single chain; no full-context or explicit per-position tensor export. Branched trees use canonical DFS partial-state snapshots and measured local accepted-path replay.')
    if not (OUT/'config.json').exists():write(OUT/'config.json',config)
    if not (OUT/'environment.json').exists():
        target=ROOT/config['model_path'];draft=ROOT/config['draft_model_path']
        receipt=draft.with_suffix('.download.json')
        if not target.exists() or not draft.exists() or not receipt.exists():
            raise FileNotFoundError('Keep the existing target and obtain the DFlash2 GGUF/receipt documented in the sealed experiment; no paid service is needed.')
        dll=ROOT/'.runtime/experiment-027/build/bin/llama.dll'
        env=dict(experiment=OUT.name,collected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            platform=platform.platform(),python=sys.version,numpy=np.__version__,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
            llama_commit=config['llama_commit'],simulator_version='e029-tree-des-1',
            target_model=dict(path=config['model_path'],size_bytes=target.stat().st_size,sha256=config['model_sha256'],hash_source='sealed E028 model hash; same local artifact retained'),
            draft_download=read(receipt),llama_dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
            baseline_environment_source=str((BASE/'environment.json').relative_to(ROOT)),paid_resources=False,remote_hosts=False,physical_accelerators=1,virtual_stage_resources=3)
        write(OUT/'environment.json',env)
    (OUT/'plots').mkdir(exist_ok=True)
    if not (OUT/'README.md').exists():
        (OUT/'README.md').write_text('# E029_LOCAL_COMMIT_ANCHORED_WAN_TREE\n\nSee config.json for preregistered gates. Run the E029 scripts to collect physical evidence and generate report.md. The final verifier executes canonical scalar tree nodes, device partial-state checkpoints, KV-tail truncation and measured commit replay. WAN results are trace-driven projections, never physical three-GPU throughput. E028 is imported by hash and is not rerun.\n',encoding='utf-8')
    print(json.dumps({'sealed_sync_tps':sync['committed_tokens_per_second'],'sealed_async_tps':asynchronous['committed_tokens_per_second'],'sealed_sync_compute_ms_per_token':sync['target_compute_ms']/sync['committed_tokens']},indent=2))
if __name__=='__main__':main()
