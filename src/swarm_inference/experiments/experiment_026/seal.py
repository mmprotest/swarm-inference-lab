"""Freeze E026 configuration and gates before opening sealed prompts."""
import argparse
import json
from pathlib import Path
import subprocess

from .io import digest,file_digest,utc_now,write_once


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("artifacts/experiment-026"))
    p.add_argument("--deployment-id",default="wan-router-services-final-001")
    a=p.parse_args()
    recovery=json.loads((a.root/"recovery/wan-replica-kill-001.json").read_text())
    if recovery["status"]!="PASS":raise RuntimeError("Recovery development gate is not complete")
    model=json.loads((a.root/"acquisition/Qwen3.8-27B-Q4_K_M.gguf.json").read_text())
    sealed=json.loads((a.root/"corpus/sealed.json").read_text())
    prompt=next(row for row in sealed["prompts"] if row["prompt_id"]=="sealed-01-factual")
    package=Path("src/swarm_inference/experiments/experiment_026")
    row={"timestamp":utc_now(),"experiment_id":"E026_Q27_WAN_SWARM_INTEGRATED_PROOF",
         "status":"SEALED_BEFORE_FINAL_PROMPT_EXECUTION","repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         "dirty_tree":subprocess.check_output(["git","status","--porcelain=v1"],text=True),
         "model":{"source":model["model_source"],"revision":model["revision"],"file":model["filename"],
                  "sha256":model["sha256"],"quantization":"Q4_K_M"},
         "llama_commit":"f1b6fbf35cfa010b0a8d6301fdfccbb7f41bd903",
         "final":{"deployment_id":a.deployment_id,"prompt_id":prompt["prompt_id"],
                  "prompt_content_sha256":prompt["content_sha256"],"generated_tokens":512,"n_probs":10},
         "inference":{"speculation":"none","sampling":"greedy","seed":260026,"ctx_checkpoints":0,
                      "tensor_split":[1,1,6],"split_mode":"layer","contiguous_stages":True,
                      "stage_order":["Japan RTX 3080 Ti","South Korea RTX 3060","local RTX 5090"],
                      "peer_transport":"Vast managed SSH proxy","ordered_async_cross_worker_copy":True,
                      "required_exact_cas":True,"pipelined_cache_validation":True,
                      "activation_compression":"off","flash_attention":"on","context":12288,
                      "batch":512,"ubatch":512},
         "recovery":{"policy":"synchronous warm mirror of all RPC state including attention and GDN recurrent buffers",
                     "checkpoint_replay_tokens":0,"development_receipt":"recovery/wan-replica-kill-001.json"},
         "frozen_gates":{"physical_machines_min":3,"preferred_application_rtt_ms":80,
                         "interactive_decode_tok_s":8,"baseline_speedup":1.5,"tokens_per_traversal":2.5,
                         "disk_warm_speedup":3,"recovery_interruption_s":20,"vast_spend_usd":38},
         "decisions":{"native_mtp":"rejected: corpus exactness failure and <=1.295x stable-route speedup",
                      "lossy_activation_compression":"not tested: 20,480-byte decode boundary is latency-dominated",
                      "final_transport":"managed proxy selected over faster direct peer because direct SSH expired mid-run"},
         "orchestration_sha256":{path.name:file_digest(path) for path in sorted(package.glob("*.py"))}}
    row["configuration_sha256"]=digest(row)
    write_once(a.root/"seal/configuration.json",row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
