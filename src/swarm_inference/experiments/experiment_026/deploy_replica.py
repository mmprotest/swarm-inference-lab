"""Launch and verify the loopback-only E026 warm-stage replica proxy."""
import argparse
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

from .io import file_digest,utc_now,write_once
from .remote import nodes
from .rpc_replica import response


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True)
    p.add_argument("--port",type=int,default=42644)
    p.add_argument("--primary",default="127.0.0.1:42641")
    p.add_argument("--standby",default="127.0.0.1:42643")
    p.add_argument("--cache-manifest",type=Path,default=Path("artifacts/experiment-026/acquisition/remote-stage-a-all-002.json"))
    a=p.parse_args()
    folder=Path("artifacts/experiment-026/services")/a.run_id
    folder.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-m","swarm_inference.experiments.experiment_026.rpc_replica",
             "--port",str(a.port),"--primary",a.primary,"--standby",a.standby,
             "--log",str(folder/"replica.jsonl"),"--cache-manifest",str(a.cache_manifest)]
    with (folder/"process.log").open("wb") as log:
        process=subprocess.Popen(command,stdout=log,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    deadline=time.monotonic()+30
    while True:
        try:
            with socket.create_connection(("127.0.0.1",a.port),timeout=5) as check:
                check.sendall(struct.pack("<BQ",14,24)+bytes(24))
                if response(check)[0]!=6:raise RuntimeError("Replica RPC-v6 handshake failed")
            break
        except (OSError,EOFError):
            if process.poll() is not None:raise RuntimeError("Replica proxy exited before readiness")
            if time.monotonic()>deadline:raise TimeoutError("Replica readiness")
            time.sleep(.25)
    inventory={row["role"]:{k:row.get(k) for k in ("id","gpu_name","geolocation","host","port")}
               for row in nodes()}
    row={"timestamp":utc_now(),"pid":process.pid,"command":command,"port":a.port,
         "primary":a.primary,"standby":a.standby,"primary_node":inventory["a"],
         "standby_node":inventory["b"],"replication_scope":"ALL_RPC_COMMANDS_AND_MUTABLE_BUFFERS",
         "source_sha256":file_digest(Path(__file__).with_name("rpc_replica.py")),
         "cache_manifest":str(a.cache_manifest),"cache_manifest_sha256":file_digest(a.cache_manifest)}
    write_once(folder/"deployment.json",row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
