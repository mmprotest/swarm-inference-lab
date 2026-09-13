"""Trace-driven virtual WAN replay, NOT a physical multi-GPU benchmark.

Each target resource serves one operation at a time. Links serialize bytes but
permit propagation overlap. The drafter has its own charged resource. Rejection
cancellation reaches each stage only after a WAN control message; an already
running kernel cannot be canceled. No stage duration is generated synthetically.
"""
from __future__ import annotations

from collections import defaultdict, deque
import heapq
import itertools
import json
import random
from typing import Any

import numpy as np

from .local import OUT, REQUEST, RESPONSE, write_json

VERSION = "e028-des-1"


def union_duration(intervals, start=0., end=float("inf")):
    points = sorted((max(start,a), min(end,b)) for a,b in intervals if b>start and a<end)
    total, stop = 0., start
    for a,b in points:
        total += max(0., b-max(stop,a))
        stop = max(stop,b)
    return total


class Measurements:
    """Frozen empirical fallback for work canceled in the local queue.

    Its actual input need not have executed locally, but the duration is always
    an observed same-stage, same-row-count operation. No interpolation/division
    of full-model latency is used.
    """
    def __init__(self, runs):
        self.samples = defaultdict(list)
        self.draft_samples = []
        self.rollback_samples = defaultdict(list)
        self.state_stats = []
        self.prefill = []
        cpu_samples = defaultdict(list)
        for run in runs:
            service = sum(s["service_ms"] for c in run["chunks"] for s in c["stages"]+c["draft"])
            service += sum(s["service_ms"] for c in run["chunks"] if c["rollback"] for s in c["rollback"]["commands"]+c["rollback"]["replay"])
            actions = len(run["chunks"])+sum(c["accepted"] is not None for c in run["chunks"])
            cpu_samples[run["k"]].append(max(0.,run["elapsed_ms"]-service)/max(1,actions))
            self.state_stats.extend(run.get("state_stats", []))
            self.prefill.append(dict(prompt_id=run.get("prompt_id"), k=run["k"], w=run["w"], **run["prefill"]))
            for chunk in run["chunks"]:
                for stage, row in enumerate(chunk["stages"]):
                    self.samples[(stage, len(chunk["tokens"]))].append(row)
                self.draft_samples.extend(chunk["draft"])
                if chunk["rollback"]:
                    rb = chunk["rollback"]
                    for stage,row in enumerate(rb["commands"]):
                        self.rollback_samples[stage].append(row)
                    for stage,row in enumerate(rb["replay"]):
                        self.samples[(stage, rb["accepted_replay_rows"])].append(row)
        self.cpu_medians = {k:float(np.median(v)) for k,v in cpu_samples.items()}
        self.medians = {key: self.median(rows) for key,rows in self.samples.items()}
        self.rollback_medians = {key: self.median(rows) for key,rows in self.rollback_samples.items()}

    @staticmethod
    def median(rows):
        return {key: float(np.median([r[key] for r in rows])) for key in rows[0]
                if isinstance(rows[0][key], (int,float)) and key not in {"start_ms","end_ms","stage"}}

    def stage(self, chunk, stage, *, held_out=False):
        if not held_out and stage < len(chunk["stages"]):
            return chunk["stages"][stage], "exact_observed_operation"
        key = stage, len(chunk["tokens"])
        if key not in self.medians:
            raise ValueError(f"no measured service for stage/rows {key}")
        return self.medians[key], "empirical_same_stage_same_rows"

    def profile(self):
        stages = []
        for stage in range(3):
            by_rows = {}
            for (s,n),rows in self.samples.items():
                if s == stage:
                    by_rows[str(n)] = dict(sample_count=len(rows), median=self.median(rows),
                                          service_p10_ms=float(np.percentile([r["service_ms"] for r in rows],10)),
                                          service_p90_ms=float(np.percentile([r["service_ms"] for r in rows],90)))
            stages.append(dict(stage="ABC"[stage], verification_by_rows=by_rows,
                               decode_time_ms=by_rows["1"]["median"]["compute_ms"],
                               cuda_execution_time_ms=None,
                               cuda_execution_time_unavailable_reason="Host llama_decode + llama_synchronize timer; no CUDA event instrumentation",
                               serialization_time_ms=None,
                               serialization_unavailable_reason="Native serialize_ns combines state management, output gathering and serialization; composite field preserved"))
        return dict(experiment="E028_LOCAL_ASYNC_WAN_PROOF", evidence_class="PHYSICAL", stages=stages,
                    state_operation_totals=self.state_stats,
                    draft_time_per_token_ms=float(np.median([r["service_ms"] for r in self.draft_samples])),
                    prefill_measurements=self.prefill,
                    note="All services measured with a global GPU lock. Server compute is synchronized host time, not a CUDA kernel event timer. No three-GPU measurement.")


class Replay:
    def __init__(self, run, measurements: Measurements, network, seed, *, held_out=False):
        self.run, self.m, self.net = run, measurements, network
        self.rng, self.held_out = random.Random(seed), held_out
        self.seed = seed
        self.now = 0.
        self.events, self.seq = [], itertools.count()
        self.queues = {name: deque() for name in ["draft","cpu","A","B","C"]}
        self.busy = {name: False for name in self.queues}
        self.link_free = defaultdict(float)
        self.busy_intervals, self.network_intervals = [], []
        self.operations, self.transfers = [], []
        self.chunks = [dict(c, invalidated=False, sim_complete=False, sim_launched=False) for c in run["chunks"]]
        self.next_index, self.inflight = 0, []
        self.filling, self.rolling_back = False, False
        self.stage_epoch = [0,0,0]
        self.stage_position = [self.chunks[0]["position"]]*3
        self.stage_pending = [dict() for _ in range(3)]
        self.stage_active = [False]*3
        self.commits, self.decisions = [], []
        self.committed, self.rollback_count = 0, 0
        self.inflight_points = [(0.,0)]
        self.bytes, self.messages, self.reused_measurements = 0, 0, 0
        self.done = False
        self.phase = "decode"
        # This is measured local non-service wall time, not a fitted correction.
        physical_service = sum(s["service_ms"] for c in run["chunks"] for s in c["stages"])
        physical_service += sum(d["service_ms"] for c in run["chunks"] for d in c["draft"])
        physical_service += sum(s["service_ms"] for c in run["chunks"] if c["rollback"] for s in c["rollback"]["commands"]+c["rollback"]["replay"])
        actions = len(run["chunks"])+sum(c["accepted"] is not None for c in run["chunks"])
        self.cpu_per_action = max(0., run["elapsed_ms"]-physical_service)/max(1,actions)
        if held_out:
            self.cpu_per_action = measurements.cpu_medians[run["k"]]
        self.steady_end = 0.

    def event(self, when, callback, *args):
        heapq.heappush(self.events, (when,next(self.seq),callback,args))

    def resource(self, name, duration, callback, *, chunk=None, compute=None, kind="forward"):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("invalid measured service")
        self.queues[name].append(dict(duration=duration, callback=callback, chunk=chunk, compute=duration if compute is None else compute, kind=kind))
        self.pump(name)

    def pump(self, name):
        if self.busy[name]:
            return
        while self.queues[name]:
            op = self.queues[name].popleft()
            chunk = op["chunk"]
            if name in "ABC" and chunk is not None and op["kind"]=="forward" and chunk["epoch"] < self.stage_epoch["ABC".index(name)]:
                continue
            self.busy[name] = True
            start, end = self.now, self.now+op["duration"]
            row = dict(resource=name, start_ms=start, end_ms=end, duration_ms=op["duration"],
                       compute_ms=op["compute"], kind=op["kind"], chunk=None if chunk is None else chunk["id"])
            self.operations.append(row)
            self.busy_intervals.append((start,end))
            def finish(name=name, op=op):
                self.busy[name] = False
                op["callback"]()
                self.pump(name)
            self.event(end, finish)
            break

    def transfer(self, src, dst, payload, callback, *, kind="activation", chunk=None):
        bandwidth = self.net["bandwidth_mbps"]
        tx = 0. if bandwidth is None else payload*8/(bandwidth*1000)
        jitter = self.rng.uniform(-self.net["jitter_ms"], self.net["jitter_ms"])
        start = max(self.now,self.link_free[(src,dst)])
        arrival = start + tx + max(0., self.net["rtt_ms"]/2+jitter)
        self.link_free[(src,dst)] = start+tx
        self.network_intervals.append((self.now,arrival))
        self.bytes += int(payload)
        self.messages += 1
        self.transfers.append(dict(src=src,dst=dst,bytes=int(payload),queued_ms=self.now,
                                   start_ms=start,arrival_ms=arrival,kind=kind,
                                   chunk=None if chunk is None else chunk["id"]))
        self.event(arrival,callback)

    def draft_next(self):
        if self.next_index >= len(self.chunks):
            self.filling = False
            self.decide()
            return
        chunk = self.chunks[self.next_index]
        self.next_index += 1
        self.filling = True
        traces = chunk["draft"]
        if self.held_out and traces:
            duration = len(traces)*float(np.median([r["service_ms"] for r in self.m.draft_samples]))
        else:
            duration = sum(r["service_ms"] for r in traces)
        def launch():
            chunk["sim_launched"] = True
            chunk["sim_launch_ms"] = self.now
            self.steady_end = self.now
            self.inflight.append(chunk)
            self.inflight_points.append((self.now,len(self.inflight)))
            self.transfer("coordinator","A",REQUEST.size+4*len(chunk["tokens"]),
                          lambda: self.stage(chunk,0),chunk=chunk)
            # Exactly the local scheduler's causal fill/refill policy.
            upcoming = self.chunks[self.next_index] if self.next_index < len(self.chunks) else None
            if len(self.inflight)<self.run["w"] and upcoming is not None and upcoming["epoch"]==chunk["epoch"] and upcoming["launch_dependency"]==chunk["launch_dependency"]:
                self.draft_next()
            else:
                self.filling = False
                self.decide()
        self.resource("draft",duration,lambda: self.resource("cpu",self.cpu_per_action,launch,kind="scheduler"),chunk=chunk,kind="draft")

    def stage(self, chunk, index):
        if chunk["epoch"] < self.stage_epoch[index]:
            return
        self.stage_pending[index][(chunk["epoch"],chunk["position"])] = chunk
        self.stage_ready(index)

    def stage_ready(self,index):
        if self.stage_active[index]:
            return
        key = (self.stage_epoch[index],self.stage_position[index])
        chunk = self.stage_pending[index].pop(key,None)
        if chunk is None:
            return
        self.stage_active[index] = True
        row, provenance = self.m.stage(chunk,index,held_out=self.held_out)
        self.reused_measurements += provenance != "exact_observed_operation"
        def finished():
            self.stage_active[index] = False
            if chunk["epoch"] < self.stage_epoch[index]:
                return
            self.stage_position[index] = chunk["position"]+len(chunk["tokens"])
            if index<2:
                self.transfer("ABC"[index],"ABC"[index+1],row["response_bytes"],
                              lambda: self.stage(chunk,index+1),chunk=chunk)
            else:
                self.transfer("C","coordinator",row["response_bytes"],lambda: self.verified(chunk),chunk=chunk)
            self.stage_ready(index)
        self.resource("ABC"[index],row["service_ms"],finished,chunk=chunk,compute=row["compute_ms"])

    def verified(self, chunk):
        if chunk["invalidated"]:
            return
        chunk["sim_complete"] = True
        if not self.filling and not self.rolling_back:
            self.decide()

    def decide(self):
        if self.done or self.filling or self.rolling_back or not self.inflight:
            return
        head = self.inflight[0]
        if not head["sim_complete"] or head.get("decision_pending"):
            return
        head["decision_pending"] = True
        def commit():
            if head["accepted"] is None:
                raise AssertionError("replay reached an unmeasured verification outcome")
            assert self.inflight.pop(0) is head
            accepted = head["accepted"]
            self.committed += accepted
            self.commits.extend([self.now]*accepted)
            self.inflight_points.append((self.now,len(self.inflight)))
            self.decisions.append(dict(chunk=head["id"],time_ms=self.now,accepted=accepted,rejected=head["rejected"]))
            if head["rejected"]:
                self.rollback(head)
                return
            if self.committed == len(self.run["committed_tokens"]):
                self.done = True
                return
            upcoming = self.chunks[self.next_index] if self.next_index<len(self.chunks) else None
            if upcoming is not None and upcoming["launch_dependency"]==head["id"]:
                self.draft_next()
            else:
                self.decide()
        self.resource("cpu",self.cpu_per_action,commit,kind="scheduler")

    def rollback(self, head):
        self.rollback_count += 1
        self.rolling_back = True
        for c in self.inflight:
            c["invalidated"] = True
        self.inflight.clear()
        self.inflight_points.append((self.now,0))
        rb = head["rollback"]
        if rb is None:
            raise AssertionError("missing measured rollback")
        # Three ordered control exchanges match the implemented synchronous
        # rollback barrier. Future activations are not canceled before delivery.
        def restore(index):
            def received():
                self.stage_epoch[index] = head["epoch"]+1
                self.stage_pending[index].clear()
                row = self.m.rollback_medians[index] if self.held_out else rb["commands"][index]
                def restored():
                    self.stage_position[index] = head["position"]
                    self.stage_active[index] = False
                    self.transfer("ABC"[index],"coordinator",RESPONSE.size,
                                  lambda: restore(index+1) if index<2 else replay_prefix(),kind="rollback_ack")
                self.resource("ABC"[index],row["service_ms"],restored,kind="rollback")
            self.transfer("coordinator","ABC"[index],REQUEST.size,received,kind="rollback_control")
        def replay_prefix():
            if not rb["accepted_replay_rows"]:
                resumed()
                return
            count = rb["accepted_replay_rows"]
            def stage(index):
                row = self.m.medians[(index,count)] if self.held_out else rb["replay"][index]
                def finished():
                    self.stage_position[index] = head["position"]+count
                    self.transfer("ABC"[index],"ABC"[index+1] if index<2 else "coordinator",row["response_bytes"],
                                  lambda: stage(index+1) if index<2 else resumed(),kind="rollback_replay")
                self.resource("ABC"[index],row["service_ms"],finished,compute=row["compute_ms"],kind="rollback_replay")
            self.transfer("coordinator","A",REQUEST.size+4*count,lambda: stage(0),kind="rollback_replay")
        def resumed():
            self.rolling_back = False
            if self.committed == len(self.run["committed_tokens"]):
                self.done = True
            else:
                self.draft_next()
        restore(0)

    def prefill_cost(self):
        """Serial prefill task graph, separate from steady-state decode metric."""
        elapsed = 0.
        target = self.run["prefill"]["target"]
        for batch in target:
            elapsed += sum(row["service_ms"] for row in batch)
            payloads = [batch[0]["request_bytes"]]+[row["response_bytes"] for row in batch]
            for payload in payloads:
                elapsed += self.net["rtt_ms"]/2
                if self.net["bandwidth_mbps"] is not None:
                    elapsed += payload*8/(self.net["bandwidth_mbps"]*1000)
        elapsed += sum(row["service_ms"] for row in self.run["prefill"]["draft"])
        return elapsed

    def execute(self, *, include_events=False):
        self.draft_next()
        while self.events and not self.done:
            when,_,callback,args = heapq.heappop(self.events)
            self.now = when
            callback(*args)
        if not self.done:
            raise AssertionError(f"simulation deadlock: tokens={self.committed}, next={self.next_index}, inflight={[c['id'] for c in self.inflight]}, filling={self.filling}")
        if self.committed != len(self.run["committed_tokens"]):
            raise AssertionError("committed token count changed")
        end = self.now
        target_ops = [o for o in self.operations if o["resource"] in "ABC"]
        # Operations launched before the last commit are charged to completion.
        # A finite generation cannot turn final speculative work into free compute.
        drain_end = max([end]+[o["end_ms"] for o in self.operations])
        end = drain_end
        compute = sum(o["compute_ms"] for o in target_ops)
        discarded = 0.
        for o in target_ops:
            if o["chunk"] is not None and o["kind"]=="forward":
                c = self.chunks[o["chunk"]]
                valid = c["accepted"] or 0
                if c["invalidated"]:
                    valid = 0
                discarded += o["compute_ms"]*(1-valid/len(c["tokens"]))
        stage_util = {name: sum(o["duration_ms"] for o in target_ops if o["resource"]==name)/end for name in "ABC"}
        steady_start = self.commits[0]
        steady_end = max(steady_start,self.steady_end)
        steady_util = {name: union_duration([(o["start_ms"],o["end_ms"]) for o in target_ops if o["resource"]==name],steady_start,steady_end)/max(1e-9,steady_end-steady_start) for name in "ABC"}
        # Uncovered network waiting: intervals with transfers outstanding and no
        # useful resource work. GPU/draft/CPU overlap is not counted as WAN wait.
        network_union = union_duration(self.network_intervals,0,end)
        both_union = union_duration(self.network_intervals+self.busy_intervals,0,end)
        busy_union = union_duration(self.busy_intervals,0,end)
        wait = max(0.,both_union-busy_union)
        weighted_depth = sum((b[0]-a[0])*a[1] for a,b in zip(self.inflight_points, self.inflight_points[1:]+[(end,0)]))/end
        latencies = np.diff([0.]+self.commits)
        proposed = sum(len(c["tokens"])-(1 if c["parent"] is None else 0) for c in self.chunks if c["sim_launched"])
        accepted_draft = sum(max(0,(c["accepted"] or 0)-(1 if c["parent"] is None else 0)) for c in self.chunks if c["sim_launched"] and not c["invalidated"])
        traversals = sum(o["kind"]=="forward" for o in target_ops if o["resource"]=="C")
        row = dict(experiment="E028_LOCAL_ASYNC_WAN_PROOF", simulator_version=VERSION,
                   evidence_class="PHYSICALLY GROUNDED MODEL", physical_gpu_count=1, virtual_target_gpu_count=3,
                   prompt_id=self.run.get("prompt_id","validation"), category=self.run.get("category"),
                   condition="SERIAL" if self.run["k"]==0 else "SPEC_SYNC" if self.run["w"]==1 else "SPEC_ASYNC",
                   k=self.run["k"], w=self.run["w"], network=self.net["name"], rtt_ms=self.net["rtt_ms"], seed=self.seed,
                   committed_tokens=self.committed, elapsed_ms=end, committed_tokens_per_second=self.committed*1000/end,
                   wan_wait_fraction=wait/end, network_outstanding_fraction=network_union/end,
                   **{f"stage_{name.lower()}_utilization":stage_util[name] for name in "ABC"},
                   steady_state_stage_utilization=steady_util,
                   committed_tokens_per_target_traversal=self.committed/max(1,traversals),
                   speculative_acceptance_rate=accepted_draft/max(1,proposed),
                   discarded_speculative_compute_fraction=discarded/max(1e-9,compute),
                   target_compute_ms=compute, discarded_target_compute_ms=discarded,
                   mean_inflight_chunks=weighted_depth, peak_inflight_chunks=max(n for _,n in self.inflight_points),
                   sync_events_per_committed_token=(len(self.decisions)+self.rollback_count*3)/self.committed,
                   bytes_per_committed_token=self.bytes/self.committed, bytes_transferred=self.bytes,
                   messages_per_committed_token=self.messages/self.committed,
                   rollback_count=self.rollback_count, rollback_failure_count=self.run["rollback_failures"],
                   state_corruption_count=self.run["state_corruption_events"],
                   scheduler_overhead_fraction=sum(o["duration_ms"] for o in self.operations if o["resource"]=="cpu")/end,
                   measured_scheduler_gap_ms_per_action=self.cpu_per_action,
                   peak_vram_bytes=self.run.get("peak_vram_bytes"),
                   ttft_ms=self.prefill_cost()+self.commits[0],
                   inter_token_latency_p50_ms=float(np.percentile(latencies,50)),
                   inter_token_latency_p95_ms=float(np.percentile(latencies,95)),
                   empirical_duration_reuses=self.reused_measurements,
                   stage_duration_source="held-out empirical medians" if self.held_out else "exact physical chunk operation; empirical same-stage/same-row fallback for canceled work",
                   headline_interval="decode only, includes pipeline fill/drain, drafting, rejection, rollback, transfers; prefill separate in TTFT")
        if include_events:
            row.update(operations=self.operations,transfers=self.transfers,decisions=self.decisions)
        return row


def load_workload():
    config = json.loads((OUT/"config.json").read_text())
    prompts = json.loads((OUT/"prompts.json").read_text())
    runs = []
    for p in prompts:
        for k,w in [(0,1)]+[(k,w) for k in config["draft_lengths"] for w in config["inflight_windows"]]:
            path = OUT/"traces"/f"{p['prompt_id']}-k{k}-w{w}.json"
            runs.append(json.loads(path.read_text()))
    return config, runs


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["network"],r["k"],r["w"])].append(r)
    result = []
    for (network,k,w),rs in groups.items():
        tokens = sum(r["committed_tokens"] for r in rs)
        elapsed = sum(r["elapsed_ms"] for r in rs)
        compute = sum(r["target_compute_ms"] for r in rs)
        row = dict(network=network,rtt_ms=rs[0]["rtt_ms"],k=k,w=w,condition=rs[0]["condition"],
                   runs=len(rs),committed_tokens=tokens,elapsed_ms=elapsed,
                   committed_tokens_per_second=tokens*1000/elapsed,
                   target_compute_ms=compute,
                   discarded_target_compute_ms=sum(r["discarded_target_compute_ms"] for r in rs),
                   discarded_speculative_compute_fraction=sum(r["discarded_target_compute_ms"] for r in rs)/compute)
        for metric in ["wan_wait_fraction","stage_a_utilization","stage_b_utilization","stage_c_utilization","scheduler_overhead_fraction","mean_inflight_chunks"]:
            row[metric] = sum(r[metric]*r["elapsed_ms"] for r in rs)/elapsed
        for metric in ["bytes_per_committed_token","messages_per_committed_token","sync_events_per_committed_token"]:
            row[metric] = sum(r[metric]*r["committed_tokens"] for r in rs)/tokens
        for metric in ["rollback_count","rollback_failure_count","state_corruption_count"]:
            row[metric] = sum(r[metric] for r in rs)
        row["steady_state_stage_utilization"] = {s:sum(r["steady_state_stage_utilization"][s]*r["elapsed_ms"] for r in rs)/elapsed for s in "ABC"}
        row["peak_inflight_chunks"] = max(r["peak_inflight_chunks"] for r in rs)
        row["peak_vram_bytes"] = max(r["peak_vram_bytes"] or 0 for r in rs)
        result.append(row)
    zero = max(r["committed_tokens_per_second"] for r in result if r["network"]=="LOCAL" and r["k"]>0)
    for r in result:
        if r["k"]:
            sync = next(x for x in result if x["network"]==r["network"] and x["k"]==r["k"] and x["w"]==1)
            best_sync = max(x["committed_tokens_per_second"] for x in result if x["network"]==r["network"] and x["k"]>0 and x["w"]==1)
            r["async_speedup_vs_spec_sync"] = r["committed_tokens_per_second"]/sync["committed_tokens_per_second"]
            r["speedup_vs_best_spec_sync"] = r["committed_tokens_per_second"]/best_sync
            r["throughput_retained_vs_zero_wan"] = r["committed_tokens_per_second"]/zero
    return result


def sweep():
    config,runs = load_workload()
    measurements = Measurements(runs)
    profile = measurements.profile()
    profile["selected_ranges"] = json.loads((OUT/"layer_profile.json").read_text())["selected_ranges"]
    write_json(OUT/"stage_profile.json",profile)
    rows = []
    for net in config["network_profiles"]:
        for run in runs:
            for seed in config["network_seeds"]:
                row = Replay(run,measurements,net,seed).execute()
                rows.append(row)
        print(json.dumps(dict(phase="wan-replay",network=net["name"],runs=len(rows))),flush=True)
    aggs = aggregate(rows)
    lookup = {(r["network"],r["k"],r["w"]):r for r in aggs}
    for r in rows:
        a=lookup[(r["network"],r["k"],r["w"])]
        if r["k"]:
            r["aggregate_async_speedup_vs_spec_sync"] = a["async_speedup_vs_spec_sync"]
            r["aggregate_throughput_retained_vs_zero_wan"] = a["throughput_retained_vs_zero_wan"]
    with (OUT/"wan_sweep_results.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r,allow_nan=False)+"\n")
    write_json(OUT/"wan_aggregate.json",aggs)
    return aggs
