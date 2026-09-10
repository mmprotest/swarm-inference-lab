"""Single-physical-GPU validation of ordered peer routing, before WAN use."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from .io import write_once,utc_now


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id",required=True)
    p.add_argument("--async-copy",action="store_true")
    p.add_argument("--cas",action="store_true")
    a=p.parse_args()
    folder=Path("artifacts/experiment-026/local-rpc")/a.run_id
    folder.mkdir(parents=True,exist_ok=False)
    runtime=Path(".runtime/experiment-026/build/bin").resolve()
    env={**os.environ,"E026_TRACE":"1","GGML_RPC_NO_RDMA":"1"}
    env["PATH"]=str(runtime)+os.pathsep+env.get("PATH","")
    processes=[]
    logs=[]
    def launch(name,command,port,extra_env=None):
        log=(folder/(name+".log")).open("wb")
        logs.append(log)
        proc=subprocess.Popen(command,stdout=log,stderr=log,env={**env,**(extra_env or {})},
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        processes.append(proc)
        write_once(folder/(name+".json"),{"timestamp":utc_now(),"pid":proc.pid,"command":command})
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError(name+" stopped")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=.2):return
            except OSError:time.sleep(.2)
        raise TimeoutError(name)
    try:
        for i,port in enumerate((42661,42662)):
            launch(f"worker-{i}",[str(runtime/"ggml-rpc-server.exe"),"-H","127.0.0.1","-p",str(port),"-d","CUDA0","-c"],port,
                   {"LLAMA_CACHE":str(Path(f".runtime/experiment-026/logical-rpc-cache-{i}").resolve())})
        launch("router",[sys.executable,"-m","swarm_inference.experiments.experiment_026.rpc_router","--port","42663",
                         "--workers","127.0.0.1:42661,127.0.0.1:42662","--log",str(folder/"routing.jsonl")],42663)
        front=[sys.executable,"-m","swarm_inference.experiments.experiment_026.rpc_frontend","--port","42664",
               "--router","127.0.0.1:42663","--log",str(folder/"frontend.jsonl")]
        if a.async_copy:front.append("--async-copy")
        if a.cas:front += ["--cache-manifests","artifacts/experiment-026/acquisition/remote-stage-a-all-002.json,artifacts/experiment-026/acquisition/remote-stage-b-all-002.json"]
        launch("frontend",front,42664)
        command=[sys.executable,"-m","swarm_inference.experiments.experiment_026.benchmark","--run-id",a.run_id,
                 "--runtime",str(runtime),"--rpc","127.0.0.1:42664","--tensor-split","1,1,6",
                 "--topology-class","LOCAL_ROUTED_LOGICAL_WORKERS_ONE_PHYSICAL_GPU",
                 "--prompt-ids","development-01-factual,development-02-code,development-06-retrieval","--n-predict","64"]
        with (folder/"benchmark.log").open("wb") as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=log,timeout=600,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        write_once(folder/"result.json",{"returncode":result.returncode,"evidence_class":"PHYSICAL",
                                        "scope":"ONE_GPU_LOGICAL_ROUTER_VALIDATION","async_copy":a.async_copy,"required_cache":a.cas})
        if result.returncode:raise RuntimeError("Routed local validation failed")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill()
        for log in logs:log.close()


if __name__=="__main__":main()
