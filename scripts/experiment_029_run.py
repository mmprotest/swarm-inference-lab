"""Execute only E029; resume completed prompt/configuration runs."""
import sys,json,time,itertools,hashlib,subprocess,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from swarm_inference.experiments.experiment_029.local import *
from swarm_inference.experiments.experiment_029.tree import *

def configurations(config):
    for n,d in [(8,4),(16,6),(32,8)]:
        for policy in config['policies']:
            nets=config['network_profiles'] if policy=='WAN_AWARE_TREE' else [config['network_profiles'][0]]
            for net in nets:
                yield dict(policy=policy,node_budget=n,max_depth=d,construction_network=net['name'],network=net,
                           id=f'{policy}-n{n}-d{d}-{net["name"]}')

def run(engine,reference,setting,cost,peak,limit=256):
    engine.prefill(reference['prompt_tokens']);start=time.perf_counter();committed=[];rounds=[]
    while len(committed)<limit:
        begin=time.perf_counter();position=engine.position;oracle_run=setting['policy']=='ORACLE_TREE_UPPER_BOUND'
        # The diagnostic oracle alone sees target reference tokens. Real tree
        # constructors receive only the drafter's scored lattice and budget.
        if oracle_run:draft=None;lattice=None
        else:lattice,meta,draft=engine.propose()
        construction_begin=time.perf_counter()
        depth=min(setting['max_depth'],limit-len(committed))
        if oracle_run:
            nodes=oracle(reference['committed_tokens'][len(committed):],depth);utility={}
        else:
            drafter=ScoredLattice(lattice,meta['selector_top_k'],engine.next_token)
            nodes,utility=construct(setting['policy'],drafter,setting['node_budget'],depth,setting['network'],cost)
        construction_ms=(time.perf_counter()-construction_begin)*1000
        greedy,stages=engine.verify(nodes);path,_=engine.accepted_path()
        if not path:raise RuntimeError('known greedy anchor was not accepted')
        selected=[nodes[i]['token_id'] for i in path]
        expected=reference['committed_tokens'][len(committed):len(committed)+len(selected)]
        if selected!=expected:
            write_json(OUT/'correctness_failure.json',dict(prompt=reference['prompt_id'],setting=setting,position=position,selected=selected,expected=expected,nodes=nodes,greedy=greedy,prior_rounds=rounds,prefix_committed_tokens=committed))
            raise RuntimeError('exact committed token mismatch')
        commit,inject=engine.commit(path)
        accepted=set(path)
        for n in nodes:n.update(verified=True,accepted=n['node_id'] in accepted,rejected=n['node_id'] not in accepted,status='accepted' if n['node_id'] in accepted else 'rejected')
        elapsed=(time.perf_counter()-begin)*1000
        service=sum(s['service_ms'] for s in stages+commit)+(draft or {}).get('service_ms',0)+(inject or {}).get('service_ms',0)
        discarded=sum(level['compute_ns']/1e6*(1-sum(i in accepted for i in level['nodes'])/len(level['nodes'])) for s in stages for level in s['levels'])
        # Forks are measured separately and attributed exactly by node in this
        # scalar backend. Discarded branch state setup is discarded compute too.
        discarded+=sum(level['fork_ns']/1e6*(1-sum(i in accepted for i in level['nodes'])/len(level['nodes'])) for s in stages for level in s['levels'])
        row=dict(round_id=len(rounds),root_position=position,immutable_root=True,max_uncommitted_rounds=1,
                 nodes=nodes,shape=shape(nodes),accepted_path=path,committed_tokens=selected,greedy=greedy,
                 draft=draft,stages=stages,commit=commit,inject=inject,tree_construction_ms=construction_ms,
                 physical_elapsed_ms=elapsed,cpu_overhead_ms=max(0,elapsed-service),discarded_compute_ms=discarded,
                 peak_vram_bytes=peak,utility=utility)
        rounds.append(row);committed.extend(selected)
    return dict(prompt_id=reference['prompt_id'],configuration={k:v for k,v in setting.items() if k!='network'},
                evidence_class='PHYSICAL single RTX 5090',physical_elapsed_ms=(time.perf_counter()-start)*1000,
                committed_tokens=committed,token_sha256=token_hash(committed),exact_reference_agreement=committed==reference['committed_tokens'][:limit],rounds=rounds)

def main():
    config=json.loads((OUT/'config.json').read_text());profile=json.loads((OUT/'tree_verification_profile.json').read_text());cost=CostModel(profile)
    references=[json.loads((OUT/'references'/(p['prompt_id']+'.json')).read_text()) for p in json.loads((OUT/'prompts.json').read_text())]
    runs=OUT/'runs';runs.mkdir(exist_ok=True)
    settings=list(configurations(config));completed=[]
    with Workers(tag='workload') as w:
        peak=max(w.vram(),profile['peak_vram_bytes']);e=Engine(w)
        for setting in settings:
            for ref in references:
                name=setting['id']+'-'+ref['prompt_id'];path=runs/(name+'.json')
                if path.exists():
                    existing=json.loads(path.read_text());assert existing['exact_reference_agreement'];completed.append(name);continue
                print('START',name,flush=True)
                result=run(e,ref,setting,cost,peak);peak=max(peak,w.vram())
                for r in result['rounds']:r['peak_vram_bytes']=peak
                write_json(path,result);completed.append(name)
                print('DONE',name,'rounds',len(result['rounds']),'seconds',round(result['physical_elapsed_ms']/1000,2),flush=True)
                write_json(OUT/'progress.json',dict(completed_runs=len(completed),total_runs=len(settings)*8,last_completed=name,peak_vram_bytes=peak))
        draft_samples=[r['draft'] for s in runs.glob('*.json') for r in json.loads(s.read_text())['rounds'] if r['draft']]
        write_json(OUT/'drafter_profile.json',dict(draft_backend='llama.cpp native DFlash2 scored-lattice adapter',draft_model=config['draft_model_path'],download=json.loads((ROOT/config['draft_model_path']).with_suffix('.download.json').read_text()),
            draft_vram_bytes=1079.61*1024**2,draft_vram_note='llama.cpp reported model buffer; context workspace is included in total peak VRAM, not independently isolated',
            draft_latency_ms_mean=float(np.mean([x['service_ms'] for x in draft_samples])),draft_latency_ms_p50=float(np.median([x['service_ms'] for x in draft_samples])),
            target_feature_layers=w.taps,target_feature_bytes_per_committed_token=len(w.taps)*w.clients[0].width*4,
            integration='DFlash shares target embedding/output weights through ctx_other in stage C process; candidates remain a generic scored-lattice interface',
            dflash2_used=True,fallback_used=False,block_size=w.draft_info['block_size'],selector_top_k=w.draft_info['selector_top_k'],probability_transform='conditional selector softmax, temperature 1; not calibrated target acceptance probabilities'))
    print('E029 PHYSICAL WORKLOAD COMPLETE',flush=True)
if __name__=='__main__':main()
