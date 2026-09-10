"""Launch direct-source shard acquisition on an owned worker and save receipts."""
import argparse
import json
from pathlib import Path
from .remote import nodes, ssh, upload
from .io import utc_now, write_once


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",required=True)
    p.add_argument("--shard",choices=("a","b"),required=True)
    p.add_argument("--cache",choices=("stage","standby"),default="stage")
    p.add_argument("--run-id",required=True)
    p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    node=next(n for n in nodes() if n["role"]==a.role)
    manifest=a.manifest or Path(f"artifacts/experiment-026/acquisition/remote-stage-{a.shard}-001.json")
    remote_manifest=f"/workspace/e026/{a.run_id}-manifest.json"
    upload(node,manifest,remote_manifest)
    upload(node,"scripts/experiment_026_acquire_shard.py","/workspace/e026/acquire-shard.py")
    command=(f"nohup python3 -u /workspace/e026/acquire-shard.py {remote_manifest} "
             f"/workspace/e026/cache-{a.cache}/rpc > /workspace/e026/{a.run_id}.jsonl 2>&1 < /dev/null & echo $!")
    result=ssh(node,command)
    row={"timestamp":utc_now(),"node":node["id"],"shard":a.shard,"cache":a.cache,"command":command,
         "pid":int(result.stdout.strip()),"manifest":str(manifest),"run_id":a.run_id}
    write_once(Path("artifacts/experiment-026/acquisition")/(a.run_id+"-launch.json"),row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
