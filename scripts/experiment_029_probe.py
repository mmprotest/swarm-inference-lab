import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from swarm_inference.experiments.experiment_029.local import *

def main():
    base=ROOT/'experiments/E028_LOCAL_ASYNC_WAN_PROOF/traces'
    path=next(base.glob('*-k0-w1.json'));gold=json.loads(path.read_text())
    print('trace keys',list(gold),flush=True)
    with Workers(tag='probe') as w:
        print('loaded',w.vram(),w.draft.stats(),flush=True)
        e=Engine(w);e.prefill(gold['prompt_tokens']);print('prefilled',e.position,e.next_token,flush=True)
        lattice,meta,tr=e.propose();print('draft',meta,tr['compute_ms'],lattice[1,:48].tolist(),flush=True)
        ref=gold['committed_tokens'];before=[c.fingerprint() for c in w.clients]
        nodes=[node(ref[0]),node(ref[1],0,2,1),node((ref[1]+99)%200000,0,2,2),node(ref[2],1,3,3)]
        greedy,traces=e.verify(nodes);print('verified',greedy,traces,flush=True)
        after=[c.fingerprint() for c in w.clients];assert before==after
        path,_=e.accepted_path();assert path==[0,1,3],path
        e.commit(path);print('commit',e.position,e.next_token,ref[3],w.vram(),flush=True)
        assert e.next_token==ref[3]
        write_json(OUT/'probe.json',dict(draft=meta,lattice_first_rows=lattice[:2,:272].tolist(),traces=traces,root_hashes_before=before,root_hashes_after=after,path=path,peak_vram_bytes=w.vram()))
if __name__=='__main__':main()
