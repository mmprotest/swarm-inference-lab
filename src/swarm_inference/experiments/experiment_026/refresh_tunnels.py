"""Refresh expired local SSH forwards without restarting model workers."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from .io import utc_now,write_once
from .remote import nodes,ssh,ssh_args
from .rpc_replica import response
import struct


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True)
    p.add_argument("--role",choices=("a","b"),required=True)
    p.add_argument("--base-services",type=Path,default=Path("artifacts/experiment-026/services/wan-services-001"))
    p.add_argument("--transport",choices=("direct","vast-proxy"),default="direct")
    a=p.parse_args()
    inventory={row["role"]:row for row in nodes()}
    node=inventory[a.role]
    original=json.loads((a.base_services/f"{node['id']}.json").read_text())
    expected={item["pid"]:item for item in original["launches"]}
    observed=ssh(node,"ps -p "+",".join(str(pid) for pid in expected)+" -o pid=,args=").stdout.decode()
    for pid in expected:
        if str(pid) not in observed or "ggml-rpc-server" not in observed:
            raise RuntimeError(f"Recorded RPC worker {pid} is not alive")
    forwards=[(42641,50101)] if a.role=="a" else [(42642,50101),(42643,50102)]
    for local,_ in forwards:
        with socket.socket() as probe:
            try:probe.bind(("127.0.0.1",local))
            except OSError as error:raise RuntimeError(f"Local port {local} is already occupied") from error
    host=node["host"] if a.transport=="direct" else node["ssh_host"]
    port=node["port"] if a.transport=="direct" else node["ssh_port"]
    forward_args=[]
    for local,remote in forwards:forward_args += ["-L",f"127.0.0.1:{local}:127.0.0.1:{remote}"]
    command=["ssh",*ssh_args(node),"-N","-T","-o","ExitOnForwardFailure=yes",*forward_args,
             "-p",str(port),"root@"+host]
    folder=Path("artifacts/experiment-026/services")/a.run_id
    folder.mkdir(parents=True,exist_ok=False)
    with (folder/"tunnel.log").open("wb") as log:
        process=subprocess.Popen(command,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    deadline=time.monotonic()+45
    while True:
        try:
            for local,_ in forwards:
                with socket.create_connection(("127.0.0.1",local),timeout=3) as check:
                    check.sendall(struct.pack("<BQ",14,24)+bytes(24))
                    if response(check)[0]!=6:raise RuntimeError("RPC-v6 handshake failed")
            break
        except (OSError,EOFError):
            if process.poll() is not None:raise RuntimeError("SSH tunnel exited before readiness")
            if time.monotonic()>deadline:raise TimeoutError("RPC tunnel readiness")
            time.sleep(.5)
    row={"timestamp":utc_now(),"role":a.role,"node_id":node["id"],"transport":a.transport,
         "endpoint":{"host":host,"port":port},"pid":process.pid,"command":command,
         "forwards":[{"local":local,"remote":remote} for local,remote in forwards],
         "remote_workers_verified":sorted(expected),"base_services":str(a.base_services)}
    write_once(folder/"deployment.json",row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
