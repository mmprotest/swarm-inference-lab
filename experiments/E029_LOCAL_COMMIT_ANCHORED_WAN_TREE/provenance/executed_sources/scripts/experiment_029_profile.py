import sys,json,time,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from swarm_inference.experiments.experiment_029.local import *
from swarm_inference.experiments.experiment_029.tree import *

def golds():
    b=ROOT/'experiments/E028_LOCAL_ASYNC_WAN_PROOF/traces'
    return [json.loads((b/(p['prompt_id']+'-k0-w1.json')).read_text()) for p in json.loads((OUT/'prompts.json').read_text())]

def main():
    gs=golds();tests=[];samples=[];peak=0
    prior=OUT/'branch_state_correctness.json'
    completed_tests=json.loads(prior.read_text()) if prior.exists() else {}
    reuse_branch_tests=completed_tests.get('branch_state_tests')==16 and completed_tests.get('branch_state_failures')==0
    if reuse_branch_tests:tests=completed_tests['tests']
    with Workers(tag='profile') as w:
        e=Engine(w)
        for g in gs:
            reference_path=OUT/'references'/(g['prompt_id']+'.json')
            if reference_path.exists():
                reference=json.loads(reference_path.read_text());g['committed_tokens']=reference['committed_tokens']
            else:
                e.prefill(g['prompt_tokens']);clean_tokens=[];start=time.perf_counter()
                for step in range(256):
                    clean_tokens.append(e.next_token);e.verify([node(e.next_token)]);e.commit([0])
                reference=dict(prompt_id=g['prompt_id'],prompt_tokens=g['prompt_tokens'],committed_tokens=clean_tokens,token_sha256=token_hash(clean_tokens),elapsed_ms=(time.perf_counter()-start)*1000,control='E029 same-engine scalar greedy; no tree candidates',e028_exact_agreement=clean_tokens==g['committed_tokens'])
                write_json(reference_path,reference);g['committed_tokens']=clean_tokens
                print('E029 scalar reference',g['prompt_id'],reference['e028_exact_agreement'],flush=True)
            if reuse_branch_tests:continue
            # Clean scalar, same engine reference; E028 tokens are only a
            # read-only expected sequence for this correctness diagnostic.
            gold=g['committed_tokens'];e.prefill(g['prompt_tokens'])
            for token in gold[:4]:
                e.verify([node(token)]);e.commit([0])
            clean=[c.fingerprint() for c in w.clients];next_clean=e.next_token
            for siblings in (2,4):
                e.prefill(g['prompt_tokens']);before=[c.fingerprint() for c in w.clients]
                # Wrong siblings execute before and after the surviving branch.
                nodes=[node(gold[0])];survivor=1+(siblings//2)
                for j in range(siblings):nodes.append(node(gold[1] if j==siblings//2 else (gold[1]+j+97)%200000,0,2,len(nodes)))
                nodes.append(node(gold[2],survivor,3,len(nodes)));nodes.append(node(gold[3],len(nodes)-1,4,len(nodes)))
                _,stages=e.verify(nodes);after=[c.fingerprint() for c in w.clients]
                path,_=e.accepted_path();e.commit(path);result=[c.fingerprint() for c in w.clients]
                # Compare the surviving state with identical batching while only
                # rejected sibling tokens change. This distinguishes contamination
                # from scalar-vs-batched floating-point differences.
                e.prefill(g['prompt_tokens'])
                alternate=[dict(n) for n in nodes]
                for z in alternate:
                    if z['depth']==2 and z['node_id']!=survivor:z['token_id']=(z['token_id']+181)%200000
                e.verify(alternate);path2,_=e.accepted_path();e.commit(path2)
                isolated=[c.fingerprint() for c in w.clients]
                continuation=[]
                for token in gold[4:20]:
                    continuation.append(e.next_token);e.verify([node(e.next_token)]);e.commit([0])
                checks=dict(immutable_root=before==after,accepted_path=[nodes[i]['token_id'] for i in path]==gold[:4],
                    full_and_recurrent_state_isolated=result==isolated,continuation_matches=continuation==gold[4:20])
                tests.append(dict(prompt_id=g['prompt_id'],siblings=siblings,multi_depth=True,rejected_sibling_survives=True,checks=checks,root_before=before,root_after=after,branch_commit=result,clean_commit=clean,alternate_sibling_commit=isolated,scalar_vs_batch_hash_equal=result==clean,continuation=continuation))
                print('branch',g['prompt_id'],siblings,checks,flush=True)
                write_json(OUT/'branch_state_correctness.json',dict(branch_state_tests=len(tests),branch_state_failures=sum(not all(t['checks'].values()) for t in tests),cross_branch_contamination_events=sum(not t['checks']['full_and_recurrent_state_isolated'] or not t['checks']['immutable_root'] for t in tests),tests=tests))
                if not all(checks.values()):raise RuntimeError('branch state gate failed')
            peak=max(peak,w.vram())
        # Measured batch-width/depth scaling, fixed shapes and prompts.
        for g in (gs[0],gs[-1]):
            e.prefill(g['prompt_tokens']);gold=g['committed_tokens']
            for N in (8,16,32):
                for D in (4,6,8):
                    for rep in range(2):
                        lattice,meta,draft=e.propose();drafter=ScoredLattice(lattice,meta['selector_top_k'],e.next_token)
                        nodes,_=construct('PROBABILITY_TREE',drafter,N,D,{'rtt_ms':0,'bandwidth_mbps':None})
                        _,stages=e.verify(nodes);path,_=e.accepted_path()
                        # Restore immutable root (no candidate committed) for
                        # independent repetitions of the exact same prefix.
                        commit,inject=e.commit([])
                        samples.append(dict(prompt_id=g['prompt_id'],node_budget=N,max_depth=D,rep=rep,nodes=nodes,stages=stages,draft=draft,commit=commit,inject=inject,accepted_path=path))
                        peak=max(peak,w.vram())
                        print('profile',g['prompt_id'],N,D,rep,'accepted',len(path),'ms',round(sum(s['compute_ms'] for s in stages),2),flush=True)
                        write_json(OUT/'tree_verification_profile.json',dict(samples=samples,peak_vram_bytes=peak,cuda_execution_time_ms=None,cuda_note='Host llama_decode + device synchronization timing; isolated CUDA event duration unavailable.',branch_memory='shared KV prefix and copied recurrent sequence state; no full-context snapshot per node',verification='one batch per depth, real independent sequence IDs',optional_64='not attempted: required 32-node allocation occupies about 29 GiB'))
        # Real committed-feature injection service separate from restored-root samples.
        e.prefill(gs[0]['prompt_tokens'])
        for rep in range(8):
            lattice,meta,draft=e.propose();nodes=oracle(gs[0]['committed_tokens'][rep*4:],4)
            _,stages=e.verify(nodes);path,_=e.accepted_path();commit,inject=e.commit(path)
            samples.append(dict(prompt_id=gs[0]['prompt_id'],node_budget=4,max_depth=4,rep=rep,nodes=nodes,stages=stages,draft=draft,commit=commit,inject=inject,accepted_path=path,calibration_only=True))
        write_json(OUT/'tree_verification_profile.json',dict(samples=samples,peak_vram_bytes=max(peak,w.vram()),cuda_execution_time_ms=None,verification='Canonical DFS with device partial-state checkpoints and KV-tail truncation; accepted path replay charged; linear-chain checkpoint optimization',cuda_note='Host llama_decode + synchronization, not CUDA-event-only measurement',optional_64='not attempted: 32-node allocation occupies about 29 GiB'))
    print('PROFILE COMPLETE',flush=True)
if __name__=='__main__':main()
