"""Generic scored-tree policies. Real policies never receive target logits/tokens."""
from __future__ import annotations
from collections import Counter
from typing import Protocol
import heapq
import numpy as np
from .local import node

class Drafter(Protocol):
    def children(self,parent:dict): ...

class ScoredLattice:
    """Adapter for a position/previous-candidate conditional scored lattice.

    Scores are draft conditional softmax probabilities, not calibrated guarantees.
    Each history has its own tree node even when lattice candidates reconverge.
    """
    def __init__(self,lattice,topk,anchor):self.lattice=lattice;self.topk=topk;self.anchor=anchor
    def children(self,parent):
        row=self.lattice[parent['depth']];k=self.topk;previous=parent.get('candidate_index',0)
        ids=row[:k].astype(int);scores=row[k+previous*k:k+(previous+1)*k].astype(float)
        probs=np.exp(scores-scores.max());probs/=probs.sum()
        return [(int(ids[j]),float(probs[j]),int(j)) for j in np.argsort(-probs,kind='stable')]

def shape(nodes):
    counts=Counter(n['depth'] for n in nodes);children=Counter(n['parent_id'] for n in nodes)
    return dict(nodes=len(nodes),depth=max(counts),nodes_by_depth=[counts[d] for d in range(1,max(counts)+1)],
        branching_factor_by_depth=[sum(children[n['node_id']] for n in nodes if n['depth']==d)/counts[d] for d in range(1,max(counts)+1)])

class CostModel:
    """Selection-only interpolation of measured depth-batch service costs.

    The simulator never uses this interpolation: it replays the selected tree's
    actual complete per-stage measurements. Model error is saved as utility error.
    """
    def __init__(self,profile):
        self.profile=profile;self.curves=[];self.replay_ms_per_token=profile.get('commit_replay_ms_per_token',0)
        self.node_cost={}
        for mode in ('linear','branched'):
            values=[]
            for r in profile['samples']:
                if bool(r['stages'][0].get('linear_checkpoint_mode')) == (mode=='linear'):
                    values.append(sum(t['service_ms'] for t in r['stages'])/len(r['nodes']))
            if values:self.node_cost[mode]=float(np.median(values))
        for stage in range(3):
            pools={}
            for row in profile['samples']:
                for level in row['stages'][stage]['levels']:
                    pools.setdefault(len(level['nodes']),[]).append((level['compute_ns']+level['fork_ns'])/1e6)
            xs=sorted(pools);self.curves.append((xs,[float(np.median(pools[x])) for x in xs]))
        self.draft_ms=float(np.median([x['draft']['service_ms'] for x in profile['samples'] if x.get('draft')]))
        self.commit_ms=float(np.median([sum(t['service_ms'] for t in x['commit']) for x in profile['samples']]))
        self.inject_ms=float(np.median([x['inject']['service_ms'] for x in profile['samples'] if x.get('inject')]))
    def estimate(self,nodes,net):
        counts=Counter(n['depth'] for n in nodes)
        expected=sum(n['branch_probability_or_score'] for n in nodes)
        is_chain=max(Counter(n['parent_id'] for n in nodes).values())<=1
        compute=len(nodes)*self.node_cost.get('linear' if is_chain else 'branched',sum(float(y[0]) for x,y in self.curves))
        if not is_chain:compute+=expected*self.replay_ms_per_token
        # Four traversal hops plus parallel commit request/feature response.
        bw=net['bandwidth_mbps'];network=3*net['rtt_ms']
        if bw:
            network+=(len(nodes)*(2*5120*4+32)+expected*(2*5120*4)+8*144)*8/(bw*1000)
        wall=self.draft_ms+self.inject_ms+self.commit_ms+compute+network
        return dict(expected_commit=expected,estimated_wall_ms=wall,estimated_utility=1000*expected/wall)

def construct(policy,drafter:Drafter,budget,depth,network,cost:CostModel|None=None):
    nodes=[dict(node(drafter.anchor),candidate_index=0)]
    def make(parent,candidate):
        token,p,ci=candidate
        return dict(node(token,parent['node_id'],parent['depth']+1,len(nodes),p,parent['branch_probability_or_score']*p),candidate_index=ci)
    if policy=='LINEAR_CONTROL':
        while len(nodes)<min(depth,budget):nodes.append(make(nodes[-1],drafter.children(nodes[-1])[0]))
    elif policy=='STATIC_TREE':
        # Deterministic breadth-first binary tree below a shared known anchor.
        for parent in nodes:
            if parent['depth']>=depth:continue
            for candidate in drafter.children(parent)[:2]:
                if len(nodes)>=budget:break
                nodes.append(make(parent,candidate))
            if len(nodes)>=budget:break
    else:
        frontier=[];serial=0
        def add(parent):
            nonlocal serial
            if parent['depth']>=depth:return
            for candidate in drafter.children(parent):
                p=parent['branch_probability_or_score']*candidate[1]
                heapq.heappush(frontier,(-p,serial,parent,candidate));serial+=1
        add(nodes[0])
        while frontier and len(nodes)<budget:
            if policy=='PROBABILITY_TREE':
                _,_,parent,candidate=heapq.heappop(frontier);new=make(parent,candidate)
            elif policy=='WAN_AWARE_TREE':
                current=cost.estimate(nodes,network)['estimated_utility']
                chain=max(Counter(n['parent_id'] for n in nodes).values())<=1
                # Within a strategy class every expansion has the same node
                # cost. Evaluate the highest-probability expansion in each
                # class; adding a first sibling also incurs measured replay.
                candidates=[0]
                if chain:
                    tails=[i for i,x in enumerate(frontier) if x[2]['node_id']==nodes[-1]['node_id']]
                    branches=[i for i,x in enumerate(frontier) if x[2]['node_id']!=nodes[-1]['node_id']]
                    candidates=[min(indices,key=lambda i:frontier[i][0]) for indices in (tails,branches) if indices]
                choices=[]
                for index in candidates:
                    _,_,parent,candidate=frontier[index];proposed=make(parent,candidate)
                    choices.append((cost.estimate(nodes+[proposed],network)['estimated_utility'],index,proposed))
                utility,index,new=max(choices,key=lambda x:x[0])
                if utility<=current:break
                frontier.pop(index);heapq.heapify(frontier)
            else:raise ValueError(policy)
            new['node_id']=len(nodes);nodes.append(new);add(new)
    return nodes, cost.estimate(nodes,network) if cost else {}

def oracle(tokens,depth):
    return [node(t,i-1,i+1,i) for i,t in enumerate(tokens[:depth])]
