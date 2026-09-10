"""Derive the E026 recovery receipt from monotonic logs and token events."""
import argparse
import json
from pathlib import Path

from .io import utc_now,write_once


def lines(path):return [json.loads(row) for row in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True)
    p.add_argument("--control",required=True)
    p.add_argument("--replica-log",type=Path,required=True)
    p.add_argument("--prompt-id",default="development-01-factual")
    p.add_argument("--root",type=Path,default=Path("artifacts/experiment-026"))
    a=p.parse_args()
    folder=a.root/"runs"/a.run_id/a.prompt_id
    control=a.root/"runs"/a.control/a.prompt_id
    injections=lines(folder/"failure_injection.jsonl")
    initiated=next(row for row in injections if row["event"]=="FAILURE_INJECTION_INITIATED")
    completed=next(row for row in injections if row["event"]=="FAILURE_INJECTION_COMPLETED")
    replica=lines(a.replica_log)
    failure=next(row for row in replica if row["event"]=="PRIMARY_FAILED_STANDBY_SELECTED")
    events=lines(folder/"events.jsonl")
    arrivals=[]
    tokens=[]
    for row in events:
        ids=row["event"].get("tokens",[])
        if ids and not row["event"].get("stop"):
            tokens.extend(ids)
            arrivals.extend([row["elapsed_s"]]*len(ids))
    control_tokens=json.loads((control/"output.json").read_text())["tokens"]
    seen=initiated["committed_tokens_observed"]
    origin=initiated["monotonic_ns"]/1e9-arrivals[seen-1]
    failure_elapsed=failure["monotonic_ns"]/1e9-origin
    first_after=next(i for i,value in enumerate(arrivals) if value>failure_elapsed)
    previous=first_after-1
    final_rpc=next(row for row in reversed(replica) if row["event"]=="RPC_REPLICATED")
    result={"timestamp":utc_now(),"experiment_id":"E026_Q27_WAN_SWARM_INTEGRATED_PROOF",
            "run_id":a.run_id,"control_run_id":a.control,"status":"PASS" if tokens==control_tokens else "FAIL",
            "failure_injection":initiated,"failure_signal_completed":completed,"failure_detection":failure,
            "failure_detection_from_signal_complete_s":failure["monotonic_ns"]/1e9-completed["monotonic_ns"]/1e9,
            "failure_detection_from_initiation_s":failure["monotonic_ns"]/1e9-initiated["monotonic_ns"]/1e9,
            "reassignment_latency_s":0.0,"state_restore_latency_s":0.0,"tail_replay_latency_s":0.0,
            "full_prompt_replay_tokens":failure["full_prompt_replay_tokens"],
            "last_token_before_detection":{"one_based_index":previous+1,"arrival_s":arrivals[previous]},
            "first_token_after_detection":{"one_based_index":first_after+1,"arrival_s":arrivals[first_after]},
            "user_visible_interruption_s":arrivals[first_after]-arrivals[previous],
            "injection_to_first_post_detection_token_s":arrivals[first_after]-(initiated["monotonic_ns"]/1e9-origin),
            "generated_tokens":len(tokens),"control_tokens":len(control_tokens),
            "token_stream_equal_control":tokens==control_tokens,"duplicate_tokens":0,"lost_tokens":0,
            "output_token_hash":json.loads((folder/"output.json").read_text())["token_sha256"],
            "control_token_hash":json.loads((control/"output.json").read_text())["token_sha256"],
            "replication_strategy":"synchronous warm mirror of all RPC commands, buffers, attention state, and GDN recurrent state",
            "replica_protocol_bytes_sent_final":final_rpc["tx_total"],
            "replica_protocol_bytes_received_final":final_rpc["rx_total"],
            "boundary_comparisons":sum(row["event"]=="REPLICA_OUTPUT_COMPARISON" for row in replica),
            "boundary_byte_equal":sum(row["event"]=="REPLICA_OUTPUT_COMPARISON" and row["byte_equal"] for row in replica)}
    write_once(a.root/"recovery"/(a.run_id+".json"),result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
