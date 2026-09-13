"""Independent W=1 local injected-latency validation of measured service replay."""
import json

from .local import OUT, Workers, LocalEngine, write_json
from .simulator import Measurements, Replay, load_workload, VERSION


async def validate(config, prompts, ranges):
    _, training_runs = load_workload()
    measured = Measurements(training_runs)
    cases = [(0,30),(0,60),(3,60)]
    rows=[]
    with Workers(config,ranges,tag="validation") as workers:
        ids=workers.clients[0].tokenize(prompts[0]["content"]).tolist()
        for k,rtt in cases:
            net=next(p for p in config["network_profiles"] if p["rtt_ms"]==rtt)
            path=OUT/"traces"/f"validation-k{k}-rtt{rtt}.json"
            if path.exists():
                run=json.loads(path.read_text())
            else:
                engine=LocalEngine(workers,shaped_network=net,seed=config["seed"])
                run=await engine.run(ids,k=k,w=1,count=64)
                write_json(path,run)
            prediction=Replay(run,measured,net,config["seed"],held_out=True).execute(include_events=True)
            error=abs(prediction["elapsed_ms"]-run["elapsed_ms"])/run["elapsed_ms"]
            row=dict(k=k,w=1,network=net["name"],committed_tokens=len(run["committed_tokens"]),
                     measured_runtime_ms=run["elapsed_ms"],predicted_runtime_ms=prediction["elapsed_ms"],
                     absolute_relative_error=error,passed=error<=.10,
                     actual_injected_delays=len(run["shaping"]),
                     requested_delay_ms=sum(t["requested_ms"] for t in run["shaping"]),
                     actual_delay_ms=sum(t["actual_ms"] for t in run["shaping"]),
                     timing_source="independent unshaped E028 workload medians; no fitting to validation wall time",
                     trace=str(path.relative_to(OUT)))
            write_json(OUT/"traces"/f"validation-prediction-k{k}-rtt{rtt}.json",prediction)
            rows.append(row)
            write_json(OUT/"simulator_validation.json",dict(evidence_class="SHAPED NETWORK",simulator_version=VERSION,
                       complete=len(rows)==len(cases),passed=all(r["passed"] for r in rows),
                       threshold_absolute_relative_error=.10,conditions=rows,
                       interpretation="Validates serial causal service/transfer accounting only. Does not validate physical three-GPU contention, transport, or async throughput."))
            print(json.dumps(dict(phase="simulator-validation",**row)),flush=True)
        net=next(p for p in config["network_profiles"] if p["rtt_ms"]==60)
        engine=LocalEngine(workers,shaped_network=net,seed=config["seed"])
        run=await engine.run(ids,k=1,w=8,count=64)
        reference=next(r for r in training_runs if r["prompt_id"]==prompts[0]["prompt_id"] and r["k"]==0)
        incomplete_peak=max(sum(c["epoch"]==launch["epoch"] and c["launched_ms"]<=launch["launched_ms"] and (c["completed_ms"] is None or c["completed_ms"]>launch["launched_ms"]) for c in run["chunks"]) for launch in run["chunks"])
        result=dict(evidence_class="SHAPED NETWORK",physical_gpu_count=1,k=1,w=8,
                    exact_committed_tokens=run["committed_tokens"]==reference["committed_tokens"][:64],
                    peak_inflight_chunks=run["peak_inflight_chunks"],peak_unfinished_target_verifications=incomplete_peak,
                    launches_before_prior_verification_completed=run["launches_before_prior_verification_completed"],
                    rollback_events=run["rollback_events"],state_corruption_events=run["state_corruption_events"])
        write_json(OUT/"traces/async-shaped-w8.json",run)
        write_json(OUT/"async_network_correctness.json",result)
        print(json.dumps(dict(phase="async-network-correctness",**result)),flush=True)
    return rows
