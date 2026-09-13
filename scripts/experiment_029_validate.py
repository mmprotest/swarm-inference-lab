import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from swarm_inference.experiments.experiment_029.local import *
from swarm_inference.experiments.experiment_029.tree import *
from swarm_inference.experiments.experiment_029.simulator import TreeReplay
from swarm_inference.experiments.experiment_029.shaping import LocalDelay
from experiment_029_run import run

def main():
    reference=json.loads((OUT/'references/conversation-1.json').read_text())
    cost=CostModel(json.loads((OUT/'tree_verification_profile.json').read_text()))
    net=dict(name='VALIDATION-WAN-60',rtt_ms=60,bandwidth_mbps=100,jitter_ms=0)
    setting=dict(policy='STATIC_TREE',node_budget=8,max_depth=4,construction_network='LOCAL',network=dict(name='LOCAL',rtt_ms=0,bandwidth_mbps=None,jitter_ms=0),id='validation')
    with Workers(tag='validation') as w:
        e=Engine(w);peak=w.vram()
        training=run(e,reference,setting,cost,peak,limit=32)
        write_json(OUT/'validation_training.json',training)
        shaper=LocalDelay(net);e.shaper=shaper
        measured=run(e,reference,setting,cost,peak,limit=32);shaper.close();e.shaper=None
        write_json(OUT/'validation_shaped.json',measured)
        predicted=TreeReplay(training['rounds'],net,290913).execute()
        actual=measured['physical_elapsed_ms'];error=abs(predicted['elapsed_ms']/actual-1)
        assert [r['committed_tokens'] for r in training['rounds']]==[r['committed_tokens'] for r in measured['rounds']]
        accounting=copy.deepcopy(measured['rounds'])
        for a,b in zip(accounting,training['rounds']):a['cpu_overhead_ms']=b['cpu_overhead_ms']
        closure=TreeReplay(accounting,net,290913).execute()
        result=dict(passed=error<=.10,network=net,committed_tokens=32,rounds=len(measured['rounds']),predicted_ms=predicted['elapsed_ms'],measured_ms=actual,absolute_relative_error=error,
                    accounting_closure_predicted_ms=closure['elapsed_ms'],accounting_closure_error=abs(closure['elapsed_ms']/actual-1),
                    service_source='Independent preceding unshaped run of the exact same tree task graph; no normalization or fitted wall multiplier',
                    actual_delay='time.sleep using high-resolution Windows timer, 4 traversal hops plus parallel commit control/feature replies; GPU state calls serialized by one physical lock',
                    measured_target_compute_ms=sum(t['compute_ms'] for r in measured['rounds'] for t in r['stages']),
                    training_target_compute_ms=sum(t['compute_ms'] for r in training['rounds'] for t in r['stages']))
        profile=json.loads((OUT/'tree_verification_profile.json').read_text())
        profile['commit_replay_ms_per_token']=float(np.median([sum(t['service_ms'] for t in r['commit'])/len(r['accepted_path']) for r in training['rounds']]))
        profile['commit_replay_source']='validation_training.json: measured branched commit service per accepted token'
        write_json(OUT/'tree_verification_profile.json',profile)
        write_json(OUT/'simulator_validation.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
