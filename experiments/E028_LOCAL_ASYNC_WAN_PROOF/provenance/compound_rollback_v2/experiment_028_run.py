"""Run E028 only. No cloud operations, artifact downloads or prior suites."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_028.local import ROOT, OUT, Workers, LocalEngine, write_json


def prompts():
    path = OUT / "prompts.json"
    if path.exists():
        return json.loads(path.read_text())
    base = [
        ("conversation-1", "conversational", "I am moving to a new city and working from home. Help me design a realistic first-month routine for meeting people, exercising, and protecting focused work. Explain the tradeoffs warmly and concretely."),
        ("conversation-2", "conversational", "My friend and I disagree about how much planning a holiday needs. Write a thoughtful conversation that helps us negotiate a flexible itinerary and a shared budget. Include practical examples and a fair compromise."),
        ("coding-1", "coding", "Implement a Python bounded LRU cache with get, put, deletion, iteration, and expiration driven by an injected monotonic clock. Include examples and explain the invariants and edge cases."),
        ("coding-2", "coding", "Write a Python streaming CSV aggregator that validates required columns, reports malformed rows, and computes counts and sums per customer without storing the full input. Include a small example, tests, and a discussion of numeric accuracy."),
        ("reasoning-1", "reasoning", "Three machines complete 6, 9, and 12 jobs per hour. The fastest machine is unavailable during hours 2 through 3. Allocate 150 independent identical jobs to minimize the finish time. State assumptions, derive the schedule carefully, and check it."),
        ("reasoning-2", "reasoning", "A community library must choose between extending evening hours and adding a weekend session with the same staffing budget. Design a fair decision method, identify confounders, give a small numerical example, and explain how to test whether the choice helped."),
    ]
    records = "\n".join(f"Log {i:02d}: Team {['Cedar','Maple','Birch','Elm'][i%4]} received {8+i%7} tasks, completed {5+i%6}, and deferred {i%3}. A review was scheduled for day {i+2}. The note says to distinguish new work from carried work and to document every unresolved dependency." for i in range(1, 41))
    policies = "\n".join(f"Section {i:02d}: Equipment group {['Alpha','Beta','Gamma'][i%3]} is inspected every {3+i%5} days. An inspection records owner, location, observed defect, severity, action, and next review. Faulty equipment stays unavailable until an independent inspector signs the repair record. Historical counts are observations, not forecasts." for i in range(1, 41))
    base += [("long-context-1", "long-context", records + "\nUsing these records, explain the difference between throughput and backlog, identify what can and cannot be inferred, and propose a precise next-week operations report. Cite log numbers and avoid assuming deferred work was completed."),
             ("long-context-2", "long-context", policies + "\nCreate an actionable inspection handbook from the policy above. Explain scheduling, evidence retention, escalation, and independent repair verification. Reference specific section numbers and identify any missing information.")]
    rows = [dict(prompt_id=i, category=c, content="<|im_start|>user\n"+text+"<|im_end|>\n<|im_start|>assistant\n") for i,c,text in base]
    write_json(path, rows)
    return rows


def environment(config):
    def run(args, cwd=ROOT):
        return subprocess.check_output(args, cwd=cwd, text=True, errors="replace").strip()
    models = []
    for key in ["model_path", "draft_model_path"]:
        p = ROOT / config[key]
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(16*1024*1024), b""):
                h.update(b)
        models.append(dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest()))
    info = dict(experiment=config["experiment"], collected_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                gpu=run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
                models=models, llama_commit=run(["git", "rev-parse", "HEAD"], ROOT/".runtime/e027-llama.cpp"),
                repository_commit=run(["git", "rev-parse", "HEAD"]),
                evidence_classes=["PHYSICAL", "SHAPED NETWORK", "PHYSICALLY GROUNDED MODEL"],
                physical_gpu_count=1, paid_resources_used=False)
    write_json(OUT/"environment.json", info)
    return info


async def profile(config, prompt_rows):
    # Evaluation callbacks synchronize at existing generic l_out tensor markers.
    # These measurements select boundaries ONLY. They are never simulator times.
    with Workers(config, [(0, 64)], profile=True, draft=False, tag="layer-profile") as workers:
        c = workers.clients[0]
        engine = LocalEngine(workers)
        engine.start = time.perf_counter()
        ids = c.tokenize(prompt_rows[0]["content"]).tolist()
        pending, _, _ = await engine.prefill(ids)
        c.stats()  # discard startup/prefill callback samples
        costs, whole = [], []
        for i in range(20):
            out, tr, _ = await engine.traverse([pending], len(ids)+i)
            pending = int(out.greedy_ids[-1])
            stat = c.stats()
            if i >= 4:
                costs.append(stat["layer_samples"])
                whole.append(tr[0])
        by_layer = [float(np.median([v/1e6 for row in costs for l,v in row if l==layer])) for layer in range(64)]
        if not all(np.isfinite(by_layer)):
            raise RuntimeError("layer callback did not expose every layer")
        head = max(0, float(np.median([t["compute_ms"] for t in whole])) - sum(by_layer))
        best = None
        for a in range(1,63):
            for b in range(a+1,64):
                times = [sum(by_layer[:a]), sum(by_layer[a:b]), sum(by_layer[b:])+head]
                candidate = (max(times), a, b, times)
                if best is None or candidate < best:
                    best = candidate
        _, a, b, balance = best
        result = dict(evidence_class="PHYSICAL", method="synchronized llama evaluation callback on generic l_out markers; used only for boundary selection", layer_decode_ms=by_layer,
                      raw_layer_samples_ns=costs, measured_full_operations=whole,
                      final_head_residual_ms=head, selected_ranges=[[0,a],[a,b],[b,64]],
                      estimated_balance_ms=balance, simulator_uses_these_callback_times=False)
        write_json(OUT/"layer_profile.json", result)
        print(json.dumps(dict(phase="layer-profile", ranges=result["selected_ranges"], balance=balance)), flush=True)
        return result


async def smoke(config, rows, ranges):
    with Workers(config, ranges, tag="smoke") as workers:
        engine = LocalEngine(workers)
        ids = workers.clients[0].tokenize(rows[0]["content"]).tolist()
        reference = await engine.run(ids, k=0, w=1, count=32)
        write_json(OUT/"traces/smoke-serial.json", reference)
        for k,w in [(1,1),(3,1),(3,8)]:
            result = await engine.run(ids, k=k, w=w, count=32, fingerprints=True)
            result["exact"] = reference["committed_tokens"] == result["committed_tokens"]
            write_json(OUT/f"traces/smoke-k{k}-w{w}.json", result)
            print(json.dumps({key: result[key] for key in ["k","w","exact","rollback_events","state_corruption_events","peak_inflight_chunks","launches_before_prior_verification_completed","elapsed_ms"]}), flush=True)
        print(json.dumps(dict(stage_stats=[c.stats() for c in workers.clients])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "profile", "smoke", "collect", "stress", "validate", "sweep"])
    args = parser.parse_args()
    config = json.loads((OUT/"config.json").read_text())
    rows = prompts()
    if args.phase == "prepare":
        environment(config)
    elif args.phase == "profile":
        asyncio.run(profile(config, rows))
    elif args.phase == "sweep":
        from swarm_inference.experiments.experiment_028.simulator import sweep
        sweep()
    else:
        ranges = json.loads((OUT/"layer_profile.json").read_text())["selected_ranges"]
        if args.phase == "smoke":
            asyncio.run(smoke(config, rows, ranges))
        elif args.phase == "validate":
            from swarm_inference.experiments.experiment_028.validation import validate
            asyncio.run(validate(config, rows, ranges))
        else:
            from swarm_inference.experiments.experiment_028.collect import collect, stress
            asyncio.run((collect if args.phase == "collect" else stress)(config, rows, ranges))


if __name__ == "__main__":
    main()
