"""Tree extension of E028's validated resource/link DES, not a GPU benchmark.

The event queue, serial resource service and independently serialized directed
links are reused unchanged from E028. Only the commit-anchored task graph is new.
No round starts until all commit replies and draft feature injection finish.
"""
from collections import defaultdict,deque
import heapq,itertools,random
import numpy as np
from swarm_inference.experiments.experiment_028.simulator import Replay,union_duration

VERSION='e029-tree-des-1'

class TreeReplay(Replay):
    def __init__(self,rounds,network,seed,oracle_zero_draft=False):
        self.rounds=rounds;self.net=network;self.seed=seed;self.rng=random.Random(seed)
        self.now=0.;self.events=[];self.seq=itertools.count()
        self.queues={k:deque() for k in ['draft','cpu','A','B','C']};self.busy={k:False for k in self.queues}
        self.link_free=defaultdict(float);self.busy_intervals=[];self.network_intervals=[]
        self.operations=[];self.transfers=[];self.bytes=0;self.messages=0;self.index=0;self.commits=[]
        self.oracle_zero_draft=oracle_zero_draft;self.round_times=[];self.done=False
    def begin_round(self):
        if self.index==len(self.rounds):self.done=True;return
        self.current=self.rounds[self.index];self.round_start=self.now
        r=self.current
        draft=0 if self.oracle_zero_draft else (r.get('draft') or {}).get('service_ms',0)
        self.resource('draft',draft,lambda:self.resource('cpu',r['cpu_overhead_ms'],self.ingress),kind='draft')
    def ingress(self):
        self.transfer('coordinator','A',self.current['stages'][0]['request_bytes'],lambda:self.stage(0))
    def stage(self,i):
        s=self.current['stages'][i];name='ABC'[i]
        def after():
            if i<2:self.transfer(name,'ABC'[i+1],self.current['stages'][i+1]['request_bytes'],lambda:self.stage(i+1))
            else:self.transfer(name,'coordinator',s['response_bytes'],self.start_commit)
        self.resource(name,s['service_ms'],after,compute=s['compute_ms'],kind='tree_verification')
    def start_commit(self):
        self.commit_pending=3
        for i in range(3):
            s=self.current['commit'][i];name='ABC'[i]
            def arrived(i=i,s=s,name=name):
                self.resource(name,s['service_ms'],lambda:self.transfer(name,'coordinator',s['response_bytes'],self.committed_stage,kind='committed_features'),compute=s['compute_ms'],kind='state_commit')
            self.transfer('coordinator',name,s['request_bytes'],arrived,kind='commit_control')
    def committed_stage(self):
        self.commit_pending-=1
        if self.commit_pending:return
        # The oracle ceiling has no drafter or feature-injection compute. Its
        # target and wire costs remain fully charged, including feature payload.
        injection=0 if self.oracle_zero_draft else (self.current.get('inject') or {}).get('service_ms',0)
        self.resource('draft',injection,self.finish_round,kind='draft_feature_injection')
    def finish_round(self):
        self.round_times.append(self.now-self.round_start)
        self.commits.extend([self.now]*len(self.current['accepted_path']))
        self.index+=1;self.begin_round()
    def execute(self):
        self.begin_round()
        while self.events:
            self.now,_,fn,args=heapq.heappop(self.events);fn(*args)
        assert self.done and all(not x for x in self.busy.values())
        total=self.now;tokens=len(self.commits);rs=self.rounds
        target=sum(sum(t['compute_ms'] for t in r['stages'])+sum(t['compute_ms'] for t in r['commit']) for r in rs)
        discarded=sum(r['discarded_compute_ms'] for r in rs)
        draft=sum(o['duration_ms'] for o in self.operations if o['resource']=='draft')
        busy=union_duration(self.busy_intervals)
        accepted=[len(r['accepted_path']) for r in rs]
        levels=defaultdict(lambda:[0,0])
        for r in rs:
            accepted_ids=set(r['accepted_path'])
            for n in r['nodes']:
                levels[n['depth']][0]+=1;levels[n['depth']][1]+=n['node_id'] in accepted_ids
        return dict(simulator_version=VERSION,evidence_class='PHYSICALLY GROUNDED MODEL',network=self.net['name'],rtt_ms=self.net['rtt_ms'],seed=self.seed,
            committed_tokens=tokens,rounds=len(rs),elapsed_ms=total,committed_tokens_per_second=1000*tokens/total,
            committed_tokens_per_wan_traversal=tokens/len(rs),accepted_path_length_mean=float(np.mean(accepted)),accepted_path_length_p50=float(np.percentile(accepted,50)),accepted_path_length_p95=float(np.percentile(accepted,95)),
            verified_nodes=sum(len(r['nodes']) for r in rs),verified_nodes_per_round=float(np.mean([len(r['nodes']) for r in rs])),
            target_compute_ms=target,discarded_target_compute_ms=discarded,discarded_target_compute_fraction=discarded/target,useful_target_compute_fraction=1-discarded/target,
            target_compute_ms_per_committed_token=target/tokens,draft_compute_ms=draft,draft_compute_fraction=draft/total,
            tree_construction_overhead_fraction=sum(r['tree_construction_ms'] for r in rs)/total,wan_wait_fraction=(total-busy)/total,
            bytes_transferred=self.bytes,bytes_per_committed_token=self.bytes/tokens,bytes_per_tree_round=self.bytes/len(rs),rounds_per_256_committed_tokens=len(rs)*256/tokens,
            peak_vram_bytes=max(r['peak_vram_bytes'] for r in rs),tree_coverage_by_depth={str(d):{'verified_nodes':v[0],'accepted_nodes':v[1],'coverage':v[1]/len(rs)} for d,v in sorted(levels.items())},
            accepted_nodes=sum(accepted),rejected_nodes=sum(len(r['nodes'])-len(r['accepted_path']) for r in rs),
            logical_wan_traversals=len(rs),physical_local_target_calls=sum(sum(t['physical_decode_calls'] for t in r['stages']+r['commit']) for r in rs),
            oracle_zero_draft=self.oracle_zero_draft,max_uncommitted_rounds=1,
            selected_shapes=[r['shape'] for r in rs],estimated_utility=float(np.mean([r.get('utility',{}).get('estimated_utility',0) for r in rs])),actual_utility=1000*tokens/total)
