"""Collect a fixed whitelist of compact E026 evidence before node teardown."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .io import file_digest,utc_now,write_once
from .remote import nodes,ssh_args


FILES={
 "a":["bootstrap.log","configure.log","compile.log","shard-a-cold-001.jsonl",
      "shard-a-all-upgrade-002.jsonl","wan-services-001-stage.log","wan-services-001-gpu.csv",
      "wan-stage-a-restart-after-kill-001.log",
      *[f"wan-router-services-{name}-routing.jsonl" for name in ("001","002","003","004","005","006","final-001")],
      *[f"wan-router-services-{name}-peer-tunnel.log" for name in ("001","002","003","004","005","006","final-001")]],
 "b":["bootstrap.log","compile.log","shard-b-cold-001.jsonl","shard-b-all-upgrade-002.jsonl",
      "standby-a-cache-001.jsonl","standby-a-cache-verify-002.jsonl",
      "wan-services-001-stage.log","wan-services-001-standby.log","wan-services-001-gpu.csv"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("artifacts/experiment-026/remote-collected-final"))
    p.add_argument("--deployment",type=Path,
                   help="Use the frozen deployment manifest instead of querying the Vast control plane")
    a=p.parse_args()
    root=a.output
    receipt=[]
    if a.deployment:
        deployment=json.loads(a.deployment.read_text(encoding="utf-8"))
        inventory=[deployment["nodes"][role] for role in ("a","b")]
    else:
        inventory=nodes()
    for node in inventory:
        role=node["role"]
        for name in FILES[role]:
            target=root/role/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError(target)
            start=time.perf_counter()
            result=subprocess.run(["scp",*ssh_args(node),"-P",str(node["port"]),
                f"root@{node['host']}:/workspace/e026/{name}",str(target)],capture_output=True,timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            if result.returncode:raise RuntimeError(f"Collection failed for {role}/{name}: {result.stderr.decode(errors='replace')}")
            receipt.append({"role":role,"node_id":node["id"],"remote":"/workspace/e026/"+name,
                            "local":str(target),"size_bytes":target.stat().st_size,
                            "sha256":file_digest(target),"elapsed_s":time.perf_counter()-start})
    row={"timestamp":utc_now(),"files":receipt,"total_bytes":sum(item["size_bytes"] for item in receipt)}
    write_once(root/"receipt.json",row)
    print(json.dumps({"files":len(receipt),"total_bytes":row["total_bytes"]}),flush=True)


if __name__=="__main__":main()
