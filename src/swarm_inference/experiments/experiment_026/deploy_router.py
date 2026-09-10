"""Deploy a concrete two-worker router on A and a local ordered-copy adapter."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import socket
import struct
import time
import sys
import tarfile

from .io import file_digest,utc_now,write_once
from .remote import nodes,ssh,ssh_args,upload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--async-copy",action="store_true")
    parser.add_argument("--cas",action="store_true")
    parser.add_argument("--peer-transport",choices=("direct","vast-proxy"),default="direct")
    args=parser.parse_args()
    inventory={n["role"]:n for n in nodes()}
    a,b=inventory["a"],inventory["b"]
    folder=Path("artifacts/experiment-026/services")/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    archive=Path(".runtime/experiment-026")/(args.run_id+".tar.gz")
    files={}
    with tarfile.open(archive,"x:gz") as stream:
        for name in ("__init__.py","io.py","rpc_replica.py","rpc_router.py","rpc_frontend.py"):
            path=Path("src/swarm_inference/experiments/experiment_026")/name
            stream.add(path,arcname="e026_worker/"+name)
            files[name]=file_digest(path)
    upload(a,archive,f"/workspace/e026/{args.run_id}.tar.gz")
    ssh(a,f"mkdir -p /workspace/e026/{args.run_id} && tar xzf /workspace/e026/{args.run_id}.tar.gz -C /workspace/e026/{args.run_id}")
    peer_node=a
    peer_host=b["host"] if args.peer_transport=="direct" else b["ssh_host"]
    peer_port=b["port"] if args.peer_transport=="direct" else b["ssh_port"]
    prefix=f"[{peer_host}]:{peer_port} "
    lines=[line for line in Path(".keys/e026_known_hosts").read_text().splitlines() if line.startswith(prefix)]
    if len(lines)!=1:raise ValueError("Cannot uniquely pin worker B peer endpoint")
    pin=Path(".runtime/experiment-026")/(args.run_id+"-peer-known-hosts")
    pin.write_text(lines[0]+"\n")
    upload(a,pin,f"/workspace/e026/{args.run_id}-peer-known-hosts")
    peer=("nohup ssh -N -T -i /workspace/e026/peer_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes "
          f"-o StrictHostKeyChecking=yes -o UserKnownHostsFile=/workspace/e026/{args.run_id}-peer-known-hosts "
          "-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "
          f"-L 127.0.0.1:50104:127.0.0.1:50101 -p {peer_port} root@{peer_host} "
          f"> /workspace/e026/{args.run_id}-peer-tunnel.log 2>&1 < /dev/null & echo $!")
    peer_pid=int(ssh(peer_node,peer).stdout.strip())
    router=(f"cd /workspace/e026/{args.run_id}; nohup python3 -m e026_worker.rpc_router --port 50103 "
            f"--workers 127.0.0.1:50101,127.0.0.1:50104 --log /workspace/e026/{args.run_id}-routing.jsonl "
            f"> /workspace/e026/{args.run_id}-router.log 2>&1 < /dev/null & echo $!")
    router_pid=int(ssh(a,router).stdout.strip())
    tunnel=["ssh",*ssh_args(a),"-N","-T","-o","ExitOnForwardFailure=yes","-L","127.0.0.1:42645:127.0.0.1:50103",
            "-p",str(a["port"]),"root@"+a["host"]]
    with (folder/"tunnel.log").open("wb") as log:
        process=subprocess.Popen(tunnel,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    frontend=[sys.executable,"-m","swarm_inference.experiments.experiment_026.rpc_frontend","--port","42646",
              "--router","127.0.0.1:42645","--log",str(folder/"frontend.jsonl")]
    if args.async_copy:frontend.append("--async-copy")
    if args.cas:frontend += ["--cache-manifests","artifacts/experiment-026/acquisition/remote-stage-a-all-002.json,artifacts/experiment-026/acquisition/remote-stage-b-all-002.json"]
    with (folder/"frontend.log").open("wb") as log:
        front=subprocess.Popen(frontend,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    row={"timestamp":utc_now(),"nodes":inventory,"source_files_sha256":files,"source_bundle_sha256":file_digest(archive),
         "peer_tunnel_pid":peer_pid,"peer_tunnel_command":peer,"peer_tunnel_node_role":peer_node["role"],
         "peer_transport":args.peer_transport,"peer_endpoint":{"host":peer_host,"port":peer_port},
         "router_pid":router_pid,"router_command":router,
         "local_tunnel_pid":process.pid,"local_tunnel_command":tunnel,"frontend_pid":front.pid,"frontend_command":frontend,
         "async_copy":args.async_copy,"required_verified_cache":args.cas,"device_ownership":{"0":a["id"],"1":b["id"]}}
    write_once(folder/"deployment.json",row)
    from .rpc_replica import response
    deadline=time.monotonic()+60
    while True:
        try:
            with socket.create_connection(("127.0.0.1",42646),timeout=5) as check:
                check.sendall(struct.pack("<BQ",14,24)+bytes(24))
                if response(check)[0]!=6:raise ValueError("Unexpected protocol")
                check.sendall(struct.pack("<BQ",15,0))
                if struct.unpack("<I",response(check))[0]!=2:raise ValueError("Missing concrete worker")
            break
        except (OSError,EOFError):
            if time.monotonic()>deadline:raise TimeoutError("Router readiness")
            time.sleep(.5)
    write_once(folder/"readiness.json",{"timestamp":utc_now(),"status":"TWO_WORKER_HANDSHAKE_VERIFIED"})
    print(json.dumps({"router_node":a["id"],"worker_nodes":[a["id"],b["id"]],"frontend_pid":front.pid,
                      "async_copy":args.async_copy}),flush=True)


if __name__=="__main__":main()
