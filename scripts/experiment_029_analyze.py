"""Reproduce E029 WAN projections and preregistered verdict from physical traces."""
import sys,json,collections,statistics,hashlib,datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from swarm_inference.experiments.experiment_029.local import OUT,ROOT,write_json,token_hash
from swarm_inference.experiments.experiment_029.simulator import TreeReplay,VERSION

def aggregate(rows):
    first=rows[0];a={k:first[k] for k in ('policy','node_budget','max_depth','network','rtt_ms','oracle_zero_draft')}
    sums=['committed_tokens','rounds','elapsed_ms','target_compute_ms','discarded_target_compute_ms','draft_compute_ms','bytes_transferred','verified_nodes','accepted_nodes','rejected_nodes','physical_local_target_calls']
    for k in sums:a[k]=sum(r[k] for r in rows)
    a['runs']=len(rows);a['prompts']=len(set(r['prompt_id'] for r in rows));a['committed_tokens_per_second']=1000*a['committed_tokens']/a['elapsed_ms']
    for k,num,den in [('committed_tokens_per_wan_traversal','committed_tokens','rounds'),('discarded_target_compute_fraction','discarded_target_compute_ms','target_compute_ms'),('draft_compute_fraction','draft_compute_ms','elapsed_ms'),('bytes_per_committed_token','bytes_transferred','committed_tokens'),('bytes_per_tree_round','bytes_transferred','rounds'),('target_compute_ms_per_committed_token','target_compute_ms','committed_tokens'),('verified_nodes_per_round','verified_nodes','rounds')]:a[k]=a[num]/a[den]
    a['useful_target_compute_fraction']=1-a['discarded_target_compute_fraction'];a['rounds_per_256_committed_tokens']=256*a['rounds']/a['committed_tokens']
    depths=sorted({int(d) for row in rows for d in row['tree_coverage_by_depth']})
    a['tree_coverage_by_depth']={str(d):dict(
        verified_nodes=sum(row['tree_coverage_by_depth'].get(str(d),{}).get('verified_nodes',0) for row in rows),
        accepted_nodes=sum(row['tree_coverage_by_depth'].get(str(d),{}).get('accepted_nodes',0) for row in rows),
        accepted_path_coverage=sum(row['tree_coverage_by_depth'].get(str(d),{}).get('accepted_nodes',0) for row in rows)/a['rounds']) for d in depths}
    for k in ('wan_wait_fraction','tree_construction_overhead_fraction'):a[k]=sum(r[k]*r['elapsed_ms'] for r in rows)/a['elapsed_ms']
    a['peak_vram_bytes']=max(r['peak_vram_bytes'] for r in rows)
    # Per-run path quantiles are explicitly retained; the overall mean has the
    # correct round denominator. Do not relabel a mean of p95s as an overall p95.
    a['accepted_path_length_mean']=a['committed_tokens_per_wan_traversal']
    a['per_run_path_quantiles']=[dict(prompt_id=r['prompt_id'],seed=r['seed'],p50=r['accepted_path_length_p50'],p95=r['accepted_path_length_p95']) for r in rows]
    a['selected_shapes_by_run']=[dict(prompt_id=r['prompt_id'],seed=r['seed'],**r['shape_summary']) for r in rows]
    shapes=collections.Counter()
    for r in rows:shapes.update(r['shape_summary']['shape_counts'])
    mode=shapes.most_common(1)[0][0];modal=list(map(int,mode.split(',')))
    a['selected_tree_shape']=modal;a['selected_node_budget']=sum(modal);a['selected_depth']=len(modal)
    a['branching_factor_by_depth']=[modal[i+1]/modal[i] if i+1<len(modal) else 0 for i in range(len(modal))]
    a['linear_round_fraction']=sum(v for k,v in shapes.items() if all(int(x)==1 for x in k.split(',')))/sum(shapes.values())
    a['shape_definition']='Modal actual nodes by depth; configured node/depth maxima are separate fields'
    a['estimated_utility']=sum(r['estimated_utility']*r['rounds'] for r in rows)/a['rounds'];a['actual_utility']=a['committed_tokens_per_second']
    return a

def gates(point,p100,p150,base,*,strong):
    sync=base['spec_sync_60ms']['committed_tokens_per_second'];async_tps=base['async_60ms']['committed_tokens_per_second'];tps=point['committed_tokens_per_second']
    checks=dict(committed_tokens_per_traversal=point['committed_tokens_per_wan_traversal']>=(5 if strong else 4),discarded_compute=point['discarded_target_compute_fraction']<=(.30 if strong else .40),
        throughput_vs_spec_sync=tps/sync>=(1.5 if strong else 1.25),throughput_vs_async=tps/async_tps>=(1.25 if strong else 1),retention_100=p100['committed_tokens_per_second']/tps>=(.65 if strong else .55))
    if strong:checks.update(retention_150=p150['committed_tokens_per_second']/tps>=.45,target_compute=point['target_compute_ms_per_committed_token']<=1.5*base['spec_sync_target_compute_per_committed_token_ms'],vram=point['peak_vram_bytes']<=32*2**30)
    return checks

def main():
    cfg=json.loads((OUT/'config.json').read_text());base=json.loads((OUT/'e028_baseline_import.json').read_text());groups=collections.defaultdict(list);physical=[];path_lengths=collections.defaultdict(list)
    runfiles=sorted((OUT/'runs').glob('*.json'));expected=216
    assert len(runfiles)==expected,(len(runfiles),expected)
    with (OUT/'tree_rounds.jsonl').open('w',encoding='utf-8') as round_file,(OUT/'wan_sweep_results.jsonl').open('w',encoding='utf-8') as sweep:
        for f in runfiles:
            r=json.loads(f.read_text());setting=r['configuration'];assert r['exact_reference_agreement'];assert len(r['committed_tokens'])==256
            reference=json.loads((OUT/'references'/(r['prompt_id']+'.json')).read_text());assert r['committed_tokens']==reference['committed_tokens'];position=len(reference['prompt_tokens']);observed=[]
            for row in r['rounds']:
                assert row['root_position']==position and row['max_uncommitted_rounds']==1
                assert len(row['nodes'])<=setting['node_budget'] and max(n['depth'] for n in row['nodes'])<=setting['max_depth']
                assert all(n['parent_id']<n['node_id'] for n in row['nodes']);assert all(n['verified'] and n['accepted']!=n['rejected'] for n in row['nodes'])
                assert row['committed_tokens']==[row['nodes'][i]['token_id'] for i in row['accepted_path']]
                position+=len(row['committed_tokens']);observed.extend(row['committed_tokens'])
                round_file.write(json.dumps(dict(prompt_id=r['prompt_id'],configuration=setting,source=str(f.relative_to(OUT)),**row),separators=(',',':'))+'\n')
            assert observed==r['committed_tokens']
            physical.append(dict(prompt_id=r['prompt_id'],configuration=setting,committed_tokens=256,token_sha256=r['token_sha256'],exact_agreement=True,rounds=len(r['rounds'])))
            nets=cfg['network_profiles'] if setting['policy']!='WAN_AWARE_TREE' else [n for n in cfg['network_profiles'] if n['name']==setting['construction_network']]
            for net in nets:
                for seed in cfg['network_seeds']:
                    replay=TreeReplay(r['rounds'],net,seed,oracle_zero_draft=setting['policy']=='ORACLE_TREE_UPPER_BOUND');result=replay.execute()
                    for resource in ('A','B','C','draft','cpu'):
                        ops=[o for o in replay.operations if o['resource']==resource]
                        assert all(a['end_ms']<=b['start_ms']+1e-6 for a,b in zip(ops,ops[1:])),resource
                    assert abs(sum(replay.round_times)-result['elapsed_ms'])<1e-5
                    shapes=result.pop('selected_shapes');result['shape_summary']=dict(mean_nodes=float(np.mean([s['nodes'] for s in shapes])),mean_depth=float(np.mean([s['depth'] for s in shapes])),shape_counts=dict(collections.Counter(','.join(map(str,s['nodes_by_depth'])) for s in shapes)))
                    result.update(prompt_id=r['prompt_id'],policy=setting['policy'],node_budget=setting['node_budget'],max_depth=setting['max_depth'],physical_source=str(f.relative_to(OUT)))
                    key=(setting['policy'],setting['node_budget'],setting['max_depth'],net['name'])
                    groups[key].append(result);sweep.write(json.dumps(result,separators=(',',':'))+'\n')
                    path_lengths[key].extend(len(x['accepted_path']) for x in r['rounds'])
            print('REPLAY',f.name,flush=True)
    rows=[]
    for key,items in groups.items():
        a=aggregate(items);a['accepted_path_length_p50']=float(np.percentile(path_lengths[key],50));a['accepted_path_length_p95']=float(np.percentile(path_lengths[key],95));rows.append(a)
    write_json(OUT/'wan_aggregate.json',rows)
    def best(policy,net='WAN-60'):
        return max((r for r in rows if r['policy']==policy and r['network']==net),key=lambda x:x['committed_tokens_per_second'])
    def at(point,net):return next(r for r in rows if r['policy']==point['policy'] and r['node_budget']==point['node_budget'] and r['max_depth']==point['max_depth'] and r['network']==net)
    real=best('WAN_AWARE_TREE');oracle=best('ORACLE_TREE_UPPER_BOUND');prob=best('PROBABILITY_TREE')
    monitor=json.loads((OUT/'vram_monitor.json').read_text()) if (OUT/'vram_monitor.json').exists() else {}
    monitored_peak=max([r['peak_vram_bytes'] for r in rows]+[monitor.get('peak_vram_bytes',0)])
    branches=json.loads((OUT/'branch_state_correctness.json').read_text());validation=json.loads((OUT/'simulator_validation.json').read_text())
    # Genericity is additionally checked mechanically; the native adapter reads
    # generic state/feature APIs. DFlash handling is confined to its adapter.
    forbidden=[]
    for p in (ROOT/'src/swarm_inference/experiments/experiment_029').glob('*.py'):
        for i,line in enumerate(p.read_text().splitlines(),1):
            if ('if ' in line or 'elif ' in line) and ('qwen' in line.lower() or 'deltanet' in line.lower()):forbidden.append(dict(path=str(p.relative_to(ROOT)),line=i))
    correctness=all(r['exact_agreement'] for r in physical);branch_ok=branches['branch_state_failures']==0 and branches['cross_branch_contamination_events']==0;genericity=not forbidden
    common=correctness and branch_ok and genericity and validation['passed']
    strong=gates(real,at(real,'WAN-100'),at(real,'WAN-150'),base,strong=True);passed=gates(real,at(real,'WAN-100'),at(real,'WAN-150'),base,strong=False)
    oracle_strong=gates(oracle,at(oracle,'WAN-100'),at(oracle,'WAN-150'),base,strong=True);oracle_pass=gates(oracle,at(oracle,'WAN-100'),at(oracle,'WAN-150'),base,strong=False)
    gain=real['committed_tokens_per_second']/prob['committed_tokens_per_second']-1
    failures=[]
    if not common:failures.append('A correctness, branch-state, genericity or simulator gate failed')
    if real['committed_tokens_per_wan_traversal']<3:failures.append('Mean committed tokens per traversal < 3')
    if real['discarded_target_compute_fraction']>.5:failures.append('Discarded target compute > 50%')
    if real['committed_tokens_per_second']<base['spec_sync_60ms']['committed_tokens_per_second']:failures.append('60 ms throughput below sealed E028 SPEC_SYNC')
    if gain<.10:failures.append('WAN_AWARE_TREE gain over strongest PROBABILITY_TREE < 10%')
    if not all(oracle_pass.values()):failures.append('Oracle does not meet PASS efficiency/throughput thresholds')
    if common and not failures and all(strong.values()):verdict='PASS_STRONG'
    elif common and not failures and all(passed.values()):verdict='PASS'
    else:verdict='FAIL'
    if verdict in ('PASS_STRONG','PASS'):decision='CONTINUE_SWARM'
    elif common and all(oracle_strong.values()):decision='DRAFTER_LIMITED'
    else:decision='STOP_AUTOREGRESSIVE_WAN_SWARM'
    if verdict=='FAIL' and not failures:
        if not passed['committed_tokens_per_traversal']:failures.append(f"Mean commits/traversal {real['committed_tokens_per_wan_traversal']:.3f} below PASS minimum 4.0")
        if not passed['throughput_vs_spec_sync']:failures.append(f"60 ms throughput {real['committed_tokens_per_second']:.3f} below PASS requirement {1.25*base['spec_sync_60ms']['committed_tokens_per_second']:.3f} tok/s (1.25x sealed SPEC_SYNC)")
        if not passed['throughput_vs_async']:failures.append(f"60 ms throughput below sealed ASYNC {base['async_60ms']['committed_tokens_per_second']:.3f} tok/s")
        if not failures:failures.append('Real WAN-aware policy does not meet all preregistered PASS criteria')
    summary=dict(experiment=cfg['experiment'],verdict=verdict,decision=decision,draft_backend='llama.cpp native DFlash2 Q4_K_M scored lattice',correctness_passed=correctness,branch_state_passed=branch_ok,genericity_passed=genericity,simulator_validated=validation['passed'],best_real_policy='WAN_AWARE_TREE',best_node_budget=real['node_budget'],best_max_depth=real['max_depth'],
        committed_tokens_per_traversal_60ms=real['committed_tokens_per_wan_traversal'],discarded_target_compute_fraction_60ms=real['discarded_target_compute_fraction'],committed_tps_60ms=real['committed_tokens_per_second'],
        speedup_vs_e028_spec_sync_60ms=real['committed_tokens_per_second']/base['spec_sync_60ms']['committed_tokens_per_second'],speedup_vs_e028_async_60ms=real['committed_tokens_per_second']/base['async_60ms']['committed_tokens_per_second'],
        throughput_retention_100ms_vs_60ms=at(real,'WAN-100')['committed_tokens_per_second']/real['committed_tokens_per_second'],throughput_retention_150ms_vs_60ms=at(real,'WAN-150')['committed_tokens_per_second']/real['committed_tokens_per_second'],
        target_compute_per_committed_token_ms=real['target_compute_ms_per_committed_token'],oracle_committed_tokens_per_traversal_60ms=oracle['committed_tokens_per_wan_traversal'],oracle_committed_tps_60ms=oracle['committed_tokens_per_second'],
        cross_branch_contamination_events=branches['cross_branch_contamination_events'],peak_vram_gib=monitored_peak/2**30,best_real_linear_round_fraction=real['linear_round_fraction'],
        wan_aware_gain_over_probability_60ms=gain,wan_wait_fraction_60ms=real['wan_wait_fraction'],pass_strong_gates=strong,pass_gates=passed,oracle_pass_strong_gates=oracle_strong,oracle_pass_gates=oracle_pass,fail_reasons=failures,
        headline=f"{verdict}: real WAN-aware trees project {real['committed_tokens_per_second']:.2f} committed tok/s at 60 ms RTT, {real['committed_tokens_per_wan_traversal']:.2f} tokens/traversal and {real['discarded_target_compute_fraction']:.1%} discarded target compute; oracle ceiling {oracle['committed_tokens_per_second']:.2f} tok/s. {decision}.",
        evidence_class='PHYSICALLY GROUNDED MODEL: three independent virtual stage resources using real single-5090 measurements',physical_runs=len(physical),physical_committed_tokens=sum(r['committed_tokens'] for r in physical),simulator_version=VERSION,
        coverage_required_pass_tokens_per_round_lower_bound=max(4,1.25*base['spec_sync_60ms']['committed_tokens_per_second']*real['elapsed_ms']/real['rounds']/1000),coverage_lower_bound_note='Holds current mean round duration fixed; extra committed-feature payload/state work can raise the required coverage.')
    overall_real=max((r for r in rows if r['policy']!='ORACLE_TREE_UPPER_BOUND' and r['network']=='WAN-60'),key=lambda r:r['committed_tokens_per_second'])
    summary['selection_scope']='Primary result is the strongest WAN_AWARE_TREE at 60 ms; oracle excluded and the selected N/D ceiling is held fixed for retention.'
    summary['highest_non_oracle_tps_policy']=overall_real['policy']
    summary['highest_non_oracle_tps_60ms']=overall_real['committed_tokens_per_second']
    write_json(OUT/'summary.json',summary)
    profile=json.loads((OUT/'tree_verification_profile.json').read_text());samples=profile['samples']
    profile['metric_summary']={}
    for n in (8,16,32):
        points=[x for x in samples if x['node_budget']==n and not x.get('calibration_only')]
        profile['metric_summary'][str(n)]=dict(target_tree_verification_time_ms_median=float(np.median([sum(t['service_ms'] for t in x['stages']) for x in points])),state_fork_time_ms_median=float(np.median([sum(t.get('state_checkpoint_ns',0)/1e6 for t in x['stages']) for x in points])),state_restore_time_ms_median=float(np.median([sum(t.get('state_restore_ns',0)/1e6 for t in x['stages']) for x in points])),activation_bytes_per_tree_mean=float(np.mean([sum(t['request_bytes'] for t in x['stages'])+x['stages'][-1]['response_bytes'] for x in points])),activation_input_or_output_bytes_per_node_per_hidden_boundary=(points[0]['stages'][1]['input_bytes']-8*len(points[0]['nodes']))//len(points[0]['nodes']))
    physical_profile=[x for x in samples if not x.get('calibration_only')]
    profile['verification_time_by_node_budget_ms']={str(n):profile['metric_summary'][str(n)]['target_tree_verification_time_ms_median'] for n in (8,16,32)}
    profile['verification_time_by_depth_ms']={str(d):float(np.median([sum(t['service_ms'] for t in x['stages']) for x in physical_profile if x['max_depth']==d])) for d in (4,6,8)}
    profile['state_commit_time_ms_median']=float(np.median([sum(t['service_ms'] for t in x['commit']) for x in samples]))
    profile['profile_commit_note']='Scaling samples restore the root; accepted-path replay service is independently measured in validation_training.json and charged from each physical workload trace.'
    profile['peak_vram_bytes']=monitored_peak;profile['vram_measurement']='Maximum of run snapshots and whole-GPU 200 ms nvidia-smi sampling';profile['optional_64']='Optional N=64 omitted; required budgets 8,16,32 were tested.'
    profile['partial_state_note']='Backend-defined partial snapshots isolate recurrent state while KV tails are truncated. Device storage may retain allocated depth buffers between rounds; actual total VRAM is sampled.'
    write_json(OUT/'tree_verification_profile.json',profile)
    write_json(OUT/'correctness_results.json',dict(passed=correctness,branch_state_passed=branch_ok,physical_runs=len(physical),committed_tokens=sum(r['committed_tokens'] for r in physical),runs=physical,genericity_passed=genericity,forbidden_model_branches=forbidden,reference='E029 same-engine scalar greedy references, 256 tokens for every prompt',max_uncommitted_rounds=1))
    write_json(OUT/'oracle_results.json',dict(evidence_class='Diagnostic architectural ceiling, NOT real candidate-generator performance',draft_cost='zero generation and injection cost; measured target/state and unchanged feature wire costs charged',best_60ms=oracle,rtt_sweep=[at(oracle,n['name']) for n in cfg['network_profiles']],pass_strong_gates=oracle_strong,pass_gates=oracle_pass))
    write_json(OUT/'decision_points.json',dict(best_real=real,best_oracle=oracle,best_probability=prob,real_rtt_sweep=[at(real,n['name']) for n in cfg['network_profiles']],policy_comparison_60ms=[best(p) for p in cfg['policies']]))
    # Recheck the sealed baselines and preserve an audit of every source used.
    for rel,digest in base['sha256'].items():assert hashlib.sha256((ROOT/base['source']/rel).read_bytes()).hexdigest()==digest,rel
    env=json.loads((OUT/'environment.json').read_text())
    paths=list((ROOT/'src/swarm_inference/experiments/experiment_029').glob('*.py'))+list((ROOT/'native/experiment_029').glob('*'))+list((ROOT/'scripts').glob('experiment_029_*.py'))
    env['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()};env['native_sha256']=hashlib.sha256((ROOT/'.runtime/experiment-029/build/llama-e029-stage.exe').read_bytes()).hexdigest();write_json(OUT/'environment.json',env)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
