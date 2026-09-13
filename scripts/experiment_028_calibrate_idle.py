"""Independent periodic-service calibration after E028's idle-timing diagnosis.

No whole-run correction is fitted. Only native, measured stage service records
are consumed by the replay. A different fixed prompt supplies calibration.
"""
import asyncio
import json

from swarm_inference.experiments.experiment_028.local import OUT, LocalEngine, Workers, write_json


async def main():
    config=json.loads((OUT/"config.json").read_text())
    prompts=json.loads((OUT/"prompts.json").read_text())
    ranges=json.loads((OUT/"layer_profile.json").read_text())["selected_ranges"]
    result=dict(evidence_class="PHYSICAL",purpose="Measure periodic native service when speculation is disabled; no wall-time fitting",
                calibration_seed=config["idle_calibration_seed"],calibration_prompt_id=prompts[1]["prompt_id"],validation_prompt_id=prompts[0]["prompt_id"],profiles={})
    with Workers(config,ranges,tag="idle-calibration") as workers:
        ids=workers.clients[0].tokenize(prompts[1]["content"]).tolist()
        for net in config["network_profiles"]:
            if not net["rtt_ms"]:
                continue
            engine=LocalEngine(workers,shaped_network=net,seed=config["idle_calibration_seed"])
            run=await engine.run(ids,k=0,w=1,count=32)
            run.update(prompt_id=prompts[1]["prompt_id"],calibration_only=True)
            trace=f"traces/idle-calibration-rtt{net['rtt_ms']}.json"
            write_json(OUT/trace,run)
            result["profiles"][net["name"]]=dict(rtt_ms=net["rtt_ms"],trace=trace,
                stages=[[c["stages"][index] for c in run["chunks"]] for index in range(3)])
            write_json(OUT/"serial_idle_profile.json",result)
            print(json.dumps(dict(phase="idle-calibration",network=net["name"],samples_per_stage=32)),flush=True)


if __name__=="__main__":
    asyncio.run(main())
