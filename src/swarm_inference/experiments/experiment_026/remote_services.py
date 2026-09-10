"""Start loopback-only stage services, GPU telemetry, and direct SSH tunnels."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .io import write_once, utc_now
from .remote import nodes, ssh, ssh_args


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--role")
    args=parser.parse_args()
    folder=Path("artifacts/experiment-026/services")/args.run_id
    folder.mkdir(parents=True,exist_ok=True)
    for node in nodes():
        if args.role and args.role!=node["role"]:continue
        record=folder/f"{node['id']}.json"
        if record.exists():raise RuntimeError("Service launch already recorded")
        check=ssh(node,"test -x /workspace/e026/source/build/bin/ggml-rpc-server && nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,compute_cap --format=csv && uname -a && lscpu --json && cat /proc/meminfo && cat /proc/net/dev")
        write_once(folder/f"{node['id']}-hardware.json",{"timestamp":utc_now(),"node":node,"output":check.stdout.decode()})
        launches=[]
        for name,port in (("stage",50101),("standby",50102)) if node["role"]=="b" else (("stage",50101),):
            command=(f"mkdir -p /workspace/e026/cache-{name}; "
                     f"nohup env E026_TRACE=1 GGML_RPC_NO_RDMA=1 LLAMA_CACHE=/workspace/e026/cache-{name} "
                     f"/workspace/e026/source/build/bin/ggml-rpc-server -H 127.0.0.1 -p {port} -d CUDA0 -c "
                     f"> /workspace/e026/{args.run_id}-{name}.log 2>&1 < /dev/null & echo $!")
            result=ssh(node,command)
            launches.append({"role":name,"remote_port":port,"pid":int(result.stdout.strip()),"command":command})
        gpu=ssh(node,f"nohup nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw --format=csv -lms 200 > /workspace/e026/{args.run_id}-gpu.csv 2>&1 < /dev/null & echo $!")
        forwards=["-L",f"127.0.0.1:{42641 if node['role']=='a' else 42642}:127.0.0.1:50101"]
        if node["role"]=="b":forwards += ["-L","127.0.0.1:42643:127.0.0.1:50102"]
        command=["ssh",*ssh_args(node),"-N","-T","-o","ExitOnForwardFailure=yes",*forwards,"-p",str(node["port"]),"root@"+node["host"]]
        log=(folder/f"{node['id']}-tunnel.log").open("wb")
        proc=subprocess.Popen(command,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        log.close()
        write_once(record,{"timestamp":utc_now(),"node":node,"launches":launches,"gpu_telemetry_pid":int(gpu.stdout.strip()),
                           "tunnel_pid":proc.pid,"tunnel_command":command})
        print(json.dumps({"node":node["id"],"stages":launches,"tunnel_pid":proc.pid}),flush=True)


if __name__=="__main__":main()
