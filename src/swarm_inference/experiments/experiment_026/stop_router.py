"""Stop only the processes identified by one recorded router deployment."""
import argparse
import json
from pathlib import Path
import psutil
from .remote import nodes,ssh
from .io import write_once,utc_now


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--deployment",type=Path,required=True)
    a=p.parse_args()
    d=json.loads(a.deployment.read_text())
    stopped=[]
    for key,expected in (("frontend_pid","rpc_frontend"),("local_tunnel_pid","127.0.0.1:42645:127.0.0.1:50103")):
        pid=d[key]
        try:parent=psutil.Process(pid)
        except psutil.NoSuchProcess:continue
        if expected not in " ".join(parent.cmdline()):raise ValueError("PID was reused; refusing termination")
        for process in reversed(parent.children(recursive=True)+[parent]):
            try:process.terminate()
            except psutil.NoSuchProcess:pass
        stopped.append(pid)
    inventory={n["role"]:n for n in nodes()}
    for key,expected,role in (("router_pid","e026_worker.rpc_router","a"),
                              ("peer_tunnel_pid","127.0.0.1:50104:127.0.0.1:50101",d.get("peer_tunnel_node_role","a"))):
        node=inventory[role]
        pid=d[key]
        observed=ssh(node,f"ps -p {pid} -o args=",check=False).stdout.decode()
        if not observed.strip():continue
        if expected not in observed:raise ValueError("Remote PID was reused; refusing termination")
        ssh(node,f"kill -TERM {pid}")
        stopped.append(pid)
    write_once(a.deployment.parent/"stopped.json",{"timestamp":utc_now(),"stopped_pids":stopped,
                                                "model_workers_and_caches_preserved":True})
    print({"stopped":stopped},flush=True)


if __name__=="__main__":main()
