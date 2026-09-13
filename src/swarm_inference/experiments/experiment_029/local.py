"""Physical single-GPU execution; never a physical three-GPU throughput claim."""
from __future__ import annotations
import json, os, struct, subprocess, time
from pathlib import Path
import numpy as np
from swarm_inference.experiments.experiment_027.protocol import StageClient, REQUEST, Flags
from swarm_inference.experiments.experiment_028.local import response_trace, write_json, token_hash

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'

def unpack(r):
    size=struct.unpack_from('<I',r.payload)[0]
    return json.loads(r.payload[4:4+size]),r.payload[4+size:]

class Client(StageClient):
    def command(self, op, **kw):return self._exchange(op,**kw)
    def stats(self):return json.loads(self.command(6).payload)
    def fingerprint(self,seq=0):return json.loads(self.command(10,arg=seq).payload)
    def call(self,op,payload=b'',**kw):
        r=self.command(op,payload=payload,**kw);meta,data=unpack(r)
        trace=response_trace(r,REQUEST.size+len(payload));trace.update(meta)
        return meta,data,trace

class Workers:
    def __init__(self,config=None,tag='run',slots=None,draft=True):
        self.config=config or json.loads((OUT/'config.json').read_text());self.tag=tag
        self.slots=slots or self.config['branch_slots'];self.use_draft=draft
        self.processes=[];self.clients=[];self.logs=[];self.draft=None
    def __enter__(self):
        try:
            env=os.environ.copy();env['PATH']=str(ROOT/'.runtime/experiment-027/build/bin')+os.pathsep+env.get('PATH','')
            definitions=[(a,b,False) for a,b in self.config['stage_ranges']]

            for i,(a,b,is_draft) in enumerate(definitions):
                log=OUT/'logs'/f'{self.tag}-{i}.log';log.parent.mkdir(exist_ok=True)
                f=log.open('w');self.logs.append(f);port=19929+i
                cmd=[str(ROOT/'.runtime/experiment-029/build/llama-e029-stage.exe'),'--model',str(ROOT/self.config['draft_model_path' if is_draft else 'model_path']),
                     '--port',str(port),'--stage-start',str(a),'--stage-end',str(b),'--n-ctx',str(self.config['context_tokens']),
                     '--n-batch','512','--n-ubatch','512','--checkpoint-slots',str(self.slots),'--n-rs-seq','0','--gpu-layers','999']
                if i==2 and self.use_draft:cmd+=['--draft-model',str(ROOT/self.config['draft_model_path'])]
                p=subprocess.Popen(cmd,stdout=f,stderr=f,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.processes.append(p)
                c=Client('127.0.0.1',port,timeout_s=300)
                deadline=time.time()+240
                while True:
                    if p.poll() is not None:raise RuntimeError(f'worker exited {p.returncode}: {log}')
                    try:
                        info=json.loads(c.command(3).payload);c.width=info['n_embd'];c.final=info['final_stage'];break
                    except OSError:
                        if time.time()>deadline:raise
                        time.sleep(.3)
                if is_draft:self.draft=c
                else:self.clients.append(c)
            if self.use_draft:self.draft=self.clients[-1]
            self.draft_info=self.draft.stats() if self.draft else {}
            self.taps=self.draft_info.get('target_layers',[])
            for c,(a,b) in zip(self.clients,self.config['stage_ranges']):
                c.taps=[x for x in self.taps if a<=x<b]
                c.call(13,np.asarray(c.taps,dtype='<i4').tobytes())
            return self
        except BaseException:
            self.__exit__();raise
    def __exit__(self,*args):
        for c in self.clients:
            try:c.command(5)
            except Exception:pass
            c.close()
        for p in self.processes:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
        for f in self.logs:f.close()
    def vram(self):
        x=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        return int(x.strip().splitlines()[0])*1024**2

class Engine:
    def __init__(self,workers):self.w=workers;self.position=0;self.next_token=None;self.pending=False;self.shaper=None
    def inject(self,features,position):
        if not self.w.draft:return None
        matrix=np.concatenate([features[t] for t in self.w.taps],axis=1).astype('<f4')
        return self.w.draft.call(14,matrix.tobytes(),n_tokens=len(matrix),n_embd=matrix.shape[1],position=position)[2]
    def prefill(self,tokens):
        for c in self.w.clients:c.command(2)
        self.position=0;self.pending=False;traces=[]
        for start in range(0,len(tokens),self.w.config['prefill_batch_tokens']):
            data=np.asarray(tokens[start:start+self.w.config['prefill_batch_tokens']],dtype='<i4');n=len(data);features={}
            for c in self.w.clients:
                inp=data.tobytes();meta,payload,trace=c.call(1,inp,n_tokens=n,n_embd=c.width,position=start,flags=Flags.INPUT_TOKENS if data.ndim==1 and c is self.w.clients[0] else Flags(0))
                traces.append(trace);count=n*(1 if c.final else c.width)
                data=np.frombuffer(payload,dtype='<i4' if c.final else '<f4',count=count).copy()
                if not c.final:data=data.reshape(n,c.width)
                if c.taps:
                    f=np.frombuffer(payload,dtype='<f4',offset=count*4).reshape(n,len(c.taps),c.width)
                    features.update({t:f[:,j,:] for j,t in enumerate(c.taps)})
            self.next_token=int(data[-1]);draft=self.inject(features,start)
            if draft:traces.append(draft)
            self.position+=n
        return traces
    def propose(self):
        assert not self.pending
        block=self.w.draft_info['block_size']
        meta,payload,trace=self.w.draft.call(15,position=self.position,n_tokens=block,arg=int(self.next_token))
        return np.frombuffer(payload,dtype='<f4').reshape(block,meta['width']).copy(),meta,trace
    def verify(self,nodes):
        assert not self.pending,'no unresolved multi-round speculation'
        self.pending=True;self.nodes=nodes;data=np.asarray([n['token_id'] for n in nodes],dtype='<i4');n=len(nodes)
        parents=np.asarray([x['parent_id'] for x in nodes],dtype='<i4');depth=np.asarray([x['depth'] for x in nodes],dtype='<i4');head=parents.tobytes()+depth.tobytes();traces=[]
        for c in self.w.clients:
            if self.shaper:self.shaper.transfer(56+len(head)+data.nbytes)
            _,p,t=c.call(11,head+data.tobytes(),n_tokens=n,n_embd=c.width,position=self.position,flags=Flags.INPUT_TOKENS if c is self.w.clients[0] else Flags(0))
            traces.append(t);data=np.frombuffer(p,dtype='<i4' if c.final else '<f4').copy()
            if not c.final:data=data.reshape(n,c.width)
        if self.shaper:self.shaper.transfer(traces[-1]['response_bytes'])
        self.greedy=data.tolist();return self.greedy,traces
    def accepted_path(self):
        path=[];parent=-1;expected=self.next_token
        while True:
            matches=[n for n in self.nodes if n['parent_id']==parent and n['token_id']==expected]
            if not matches:break
            node=matches[0];parent=node['node_id'];path.append(parent);expected=self.greedy[parent]
        return path,int(expected)
    def commit(self,path):
        assert self.pending
        features={};traces=[];old=self.position
        node=path[-1]+1 if path else 0
        responses=self.shaper.commit(self.w.clients,node) if self.shaper else [c.call(12,arg=node) for c in self.w.clients]
        for c,(meta,p,t) in zip(self.w.clients,responses):
            traces.append(t)
            if c.taps and path:
                f=np.frombuffer(p,dtype='<f4').reshape(len(path),len(c.taps),c.width)
                features.update({tap:f[:,j,:] for j,tap in enumerate(c.taps)})
            assert meta['position']==old+len(path)
        if path:
            self.next_token=int(self.greedy[path[-1]]);self.position+=len(path)
            draft=self.inject(features,old)
        else:draft=None
        self.pending=False
        return traces,draft

def node(token,parent=-1,depth=1,i=0,probability=1.0,score=1.0):
    return dict(node_id=i,parent_id=parent,depth=depth,token_id=int(token),draft_probability_or_score=probability,
                branch_probability_or_score=score,status='proposed',verified=False,accepted=False,rejected=False)
