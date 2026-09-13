"""SYNTHETIC task-graph unit fixtures, never benchmark input or qualification evidence."""
import copy
import unittest

from swarm_inference.experiments.experiment_028.simulator import Measurements, Replay, union_duration


def stage_rows(n=1):
    return [dict(compute_ms=float(t), service_ms=float(t), request_bytes=56+(4 if i==0 else 20)*n,
                 response_bytes=88+(28 if i==2 else 20)*n, input_bytes=(4 if i==0 else 20)*n,
                 output_bytes=(28 if i==2 else 20)*n) for i,t in enumerate([2,3,5])]


def chunk(i,tokens,accepted,*,epoch=0,parent=None,dependency=None):
    return dict(id=i,epoch=epoch,parent=parent,position=i*len(tokens),tokens=tokens,
                launch_dependency=dependency,draft=[],stages=stage_rows(len(tokens)),
                accepted=accepted,rejected=accepted is not None and accepted<len(tokens),
                invalidated=accepted is None,rollback=None)


def run(chunks,k,w,tokens):
    service = sum(s['service_ms'] for c in chunks for s in c['stages']+c['draft'])
    service += sum(s['service_ms'] for c in chunks if c['rollback'] for s in c['rollback']['commands']+c['rollback']['replay'])
    return dict(chunks=chunks,k=k,w=w,committed_tokens=tokens,elapsed_ms=service,
                prefill=dict(target=[stage_rows()],draft=[]),rollback_failures=0,state_corruption_events=0)


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.network=dict(name='TEST',rtt_ms=60,bandwidth_mbps=None,jitter_ms=0)

    def test_serial_closed_form(self):
        r=run([chunk(0,[10],1),chunk(1,[11],1,dependency=0)],0,1,[10,11])
        result=Replay(r,Measurements([r]),self.network,1).execute(include_events=True)
        self.assertAlmostEqual(result['elapsed_ms'],2*(2+3+5+4*30))
        self.assertEqual(result['committed_tokens'],2)
        self.assertEqual(len(result['transfers']),8)
        self.assertAlmostEqual(result['wan_wait_fraction'],240/260)

    def test_independent_stages_overlap_without_self_overlap(self):
        r=run([chunk(0,[10,11],2),chunk(1,[12,13],2,parent=0)],1,2,[10,11,12,13])
        result=Replay(r,Measurements([r]),self.network,1).execute(include_events=True)
        self.assertLess(result['elapsed_ms'],260)
        self.assertEqual(result['peak_inflight_chunks'],2)
        for name in ['A','B','C','draft','cpu']:
            ops=[o for o in result['operations'] if o['resource']==name]
            for a,b in zip(ops,ops[1:]):
                self.assertLessEqual(a['end_ms'],b['start_ms'])

    def test_rejection_cancels_after_control_arrives(self):
        a=chunk(0,[10,99],1)
        a['rollback']=dict(commands=[dict(service_ms=.5+t,compute_ms=t,response_bytes=180) for t in [2,3,5]],
                           replay=[],accepted_replay_rows=1)
        b=chunk(1,[77,88],None,parent=0)
        c=chunk(2,[11,12],2,epoch=1,dependency=0)
        c["position"]=1
        r=run([a,b,c],1,2,[10,11,12])
        result=Replay(r,Measurements([r]),self.network,1).execute(include_events=True)
        self.assertEqual(result['committed_tokens'],3)
        self.assertEqual(result['rollback_count'],1)
        self.assertGreater(result['discarded_speculative_compute_fraction'],0)
        controls={t['dst']:t['arrival_ms'] for t in result['transfers'] if t['kind']=='rollback_control'}
        for op in result['operations']:
            if op['chunk']==1 and op['resource'] in 'ABC':
                self.assertLess(op['start_ms'],controls[op['resource']])
        self.assertEqual([d['chunk'] for d in result['decisions']],[0,2])

    def test_jitter_cannot_reorder_recurrent_state(self):
        r=run([chunk(0,[10,11],2),chunk(1,[12,13],2,parent=0)],1,2,[10,11,12,13])
        net=dict(self.network,jitter_ms=29,bandwidth_mbps=100)
        for seed in range(10):
            result=Replay(r,Measurements([r]),net,seed).execute(include_events=True)
            for name in 'ABC':
                ids=[o['chunk'] for o in result['operations'] if o['resource']==name and o['kind']=='forward']
                self.assertEqual(ids,[0,1])

    def test_seeded_network_is_reproducible(self):
        r=run([chunk(0,[10],1)],0,1,[10])
        net=dict(self.network,jitter_ms=6,bandwidth_mbps=100)
        a=Replay(r,Measurements([r]),net,123).execute(include_events=True)
        b=Replay(copy.deepcopy(r),Measurements([r]),net,123).execute(include_events=True)
        self.assertEqual(a,b)

    def test_periodic_serial_service_is_measured_and_used_in_both_modes(self):
        r=run([chunk(0,[10],1)],0,1,[10])
        stages=[[dict(stage_rows()[i],compute_ms=float(t),service_ms=float(t))] for i,t in enumerate([10,20,30])]
        m=Measurements([r],serial_idle={"profiles":{"TEST":{"stages":stages}}})
        for held_out in [False,True]:
            result=Replay(r,m,self.network,1,held_out=held_out).execute()
            self.assertAlmostEqual(result["elapsed_ms"],10+20+30+4*30)
            self.assertTrue(result["idle_service_profile_applied"])
        row,source=m.stage(r["chunks"][0],0,condition=(1,1),network="TEST")
        self.assertEqual(row["service_ms"],2)
        self.assertEqual(source,"exact_observed_operation")

    def test_missing_measured_shape_fails_closed(self):
        r=run([chunk(0,[10],1)],0,1,[10])
        m=Measurements([r])
        c=chunk(0,[10,11,12],None)
        c['stages']=[]
        with self.assertRaises(ValueError):
            m.stage(c,0)

    def test_interval_union_does_not_double_count_overlap(self):
        self.assertEqual(union_duration([(1,5),(2,8),(9,10)]),8)
        self.assertEqual(union_duration([(1,5),(2,8)],3,6),3)


if __name__=='__main__':
    unittest.main()
