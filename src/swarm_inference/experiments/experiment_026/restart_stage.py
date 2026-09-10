"""Restart only an intentionally killed E026 RPC stage from its retained cache."""
import argparse
import json
from pathlib import Path

from .io import utc_now,write_once
from .remote import nodes,ssh


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True)
    p.add_argument("--role",choices=("a","b"),required=True)
    p.add_argument("--port",type=int,default=50101)
    p.add_argument("--cache",choices=("stage","standby"),default="stage")
    a=p.parse_args()
    node=next(row for row in nodes() if row["role"]==a.role)
    check=ssh(node,f"pgrep -af '[g]gml-rpc-server.*-p {a.port}'",check=False).stdout.decode().strip()
    if check:raise RuntimeError(f"Refusing duplicate RPC service: {check}")
    log=f"/workspace/e026/{a.run_id}.log"
    command=(f"mkdir -p /workspace/e026/cache-{a.cache}; nohup env E026_TRACE=1 GGML_RPC_NO_RDMA=1 "
             f"LLAMA_CACHE=/workspace/e026/cache-{a.cache} /workspace/e026/source/build/bin/ggml-rpc-server "
             f"-H 127.0.0.1 -p {a.port} -d CUDA0 -c > {log} 2>&1 < /dev/null & echo $!")
    pid=int(ssh(node,command).stdout.strip())
    observed=ssh(node,f"ps -p {pid} -o args=").stdout.decode().strip()
    if "ggml-rpc-server" not in observed:raise RuntimeError("Restarted RPC stage did not remain alive")
    row={"timestamp":utc_now(),"node_id":node["id"],"role":a.role,"pid":pid,"port":a.port,
         "cache":a.cache,"command":command,"observed_command":observed,"log":log,
         "recovery_source":"RETAINED_CONTENT_ADDRESSED_DISK_CACHE"}
    folder=Path("artifacts/experiment-026/services")/a.run_id
    write_once(folder/"deployment.json",row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
