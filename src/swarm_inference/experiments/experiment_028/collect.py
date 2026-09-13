"""Physical E028 workload collection and forced rollback evidence."""
from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time

import numpy as np

from .local import OUT, Workers, LocalEngine, write_json


class GPURecorder:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            try:
                text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], text=True,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
                values = [float(v.strip()) for v in text.split(",")]
                self.samples.append(dict(unix_seconds=time.time(), memory_bytes=int(values[0]*2**20), utilization_percent=values[1], temperature_c=values[2], power_w=values[3]))
            except Exception as error:
                self.samples.append(dict(error=str(error)))
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)

    def peak(self):
        return max((s.get("memory_bytes", 0) for s in self.samples), default=0)


async def collect(config, prompts, ranges):
    matrix = [(0,1)] + [(k,w) for k in config["draft_lengths"] for w in config["inflight_windows"]]
    results = []
    with GPURecorder() as gpu, Workers(config, ranges, tag="workload") as workers:
        engine = LocalEngine(workers)
        for prompt in prompts:
            ids = workers.clients[0].tokenize(prompt["content"]).tolist()
            if len(ids)+config["committed_tokens_per_prompt"] > config["context_tokens"]:
                raise ValueError("fixed prompt exceeds context; explicit workload revision required")
            for k,w in matrix:
                path = OUT / "traces" / f"{prompt['prompt_id']}-k{k}-w{w}.json"
                if path.exists():
                    run = json.loads(path.read_text())
                else:
                    before = [c.stats() for c in workers.clients]
                    wall_start = time.time()
                    run = await engine.run(ids, k=k, w=w, count=config["committed_tokens_per_prompt"])
                    after = [c.stats() for c in workers.clients]
                    run.update(prompt_id=prompt["prompt_id"], category=prompt["category"],
                               prompt_tokens=ids, peak_vram_bytes=gpu.peak(),
                               started_unix_seconds=wall_start,
                               state_stats=[{key: b[key]-a[key] for key in b if key not in {"layer_samples", "position"}} for a,b in zip(before,after)])
                    write_json(path, run)
                sync_path = OUT / "traces" / f"{prompt['prompt_id']}-k{k}-w1.json"
                sync = json.loads(sync_path.read_text())
                serial = json.loads((OUT/"traces"/f"{prompt['prompt_id']}-k0-w1.json").read_text())
                row = dict(prompt_id=prompt["prompt_id"], k=k, w=w,
                           exact_vs_sync=run["committed_tokens"]==sync["committed_tokens"],
                           exact_vs_serial=run["committed_tokens"]==serial["committed_tokens"],
                           token_sha256=run["token_sha256"], committed_tokens=len(run["committed_tokens"]),
                           rollback_events=run["rollback_events"], rollback_failures=run["rollback_failures"],
                           state_corruption_events=run["state_corruption_events"],
                           peak_inflight_chunks=run["peak_inflight_chunks"],
                           launches_before_prior_verification_completed=run["launches_before_prior_verification_completed"],
                           local_elapsed_ms=run["elapsed_ms"], trace=str(path.relative_to(OUT)))
                results.append(row)
                write_json(OUT/"correctness_results.json", dict(experiment=config["experiment"], evidence_class="PHYSICAL", complete=len(results)==len(matrix)*len(prompts),
                           passed=all(r["exact_vs_sync"] and not r["state_corruption_events"] for r in results),
                           expected_runs=len(matrix)*len(prompts), runs=results))
                print(json.dumps(dict(phase="workload", completed=len(results), total=len(matrix)*len(prompts), **row)), flush=True)
        write_json(OUT/"gpu_samples.json", gpu.samples)
    return results


async def stress(config, prompts, ranges):
    reference_run = json.loads((OUT/"traces"/f"{prompts[0]['prompt_id']}-k0-w1.json").read_text())
    reference = reference_run["committed_tokens"]
    rows = []
    with GPURecorder() as gpu, Workers(config, ranges, tag="stress") as workers:
        engine = LocalEngine(workers)
        ids = reference_run["prompt_tokens"]
        for position, window in config["rollback_stress_scenarios"]:
            path = OUT/"traces"/f"stress-position{position}-w{window}.json"
            target_events = config["rollback_stress_events_per_scenario"]
            count = target_events*position + window*4
            force = dict(position=position, remaining=target_events, reference=reference)
            if path.exists():
                run = json.loads(path.read_text())
            else:
                run = await engine.run(ids, k=3, w=window, count=count, force=force, fingerprints=True)
                run.update(forced_position=position, forced_window=window,
                           force_events_requested=target_events, force_events_remaining=force["remaining"], peak_vram_bytes=gpu.peak())
                write_json(path, run)
            rejected = [c for c in run["chunks"] if c["rejected"]]
            forced = rejected[:target_events]
            hashed = [r for r in run["rollback_checks"] if r["state_hash_equal"] is not None]
            row = dict(position=position, w=window, committed_tokens=count,
                       exact_continuation=run["committed_tokens"]==reference[:count],
                       forced_events=len(forced), forced_positions_correct=all(c["accepted"]==position for c in forced),
                       requested_inflight_reached=run["peak_inflight_chunks"]==window,
                       rollback_events=run["rollback_events"], rollback_failures=run["rollback_failures"],
                       state_corruption_events=run["state_corruption_events"],
                       full_and_partial_state_hash_checks=len(hashed),
                       state_hash_checks_passed=all(r["state_hash_equal"] for r in hashed),
                       remaining_forced_events=run["force_events_remaining"], trace=str(path.relative_to(OUT)))
            rows.append(row)
            result = dict(evidence_class="PHYSICAL", complete=len(rows)==len(config["rollback_stress_scenarios"]),
                          rollback_events=sum(r["rollback_events"] for r in rows),
                          rollback_failures=sum(r["rollback_failures"] for r in rows),
                          state_corruption_events=sum(r["state_corruption_events"] for r in rows),
                          passed=all(r["exact_continuation"] and r["forced_positions_correct"] and r["state_hash_checks_passed"] and r["full_and_partial_state_hash_checks"] >= target_events and r["requested_inflight_reached"] and r["forced_events"] == target_events and not r["remaining_forced_events"] for r in rows), scenarios=rows)
            write_json(OUT/"rollback_stress_results.json", result)
            print(json.dumps(dict(phase="rollback-stress", **row)), flush=True)
        write_json(OUT/"gpu_stress_samples.json", gpu.samples)
