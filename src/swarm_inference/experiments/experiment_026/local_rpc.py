"""Same-device RPC development only; never evidence of physical WAN performance."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .io import utc_now, write_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec-type", default="none")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--replica", action="store_true")
    args = parser.parse_args()
    root = Path("artifacts/experiment-026/local-rpc") / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    runtime = Path(".runtime/experiment-026/build/bin").resolve()
    processes, logs = [], []
    env = dict(os.environ)
    env["E026_TRACE"] = "1"
    env["GGML_RPC_NO_RDMA"] = "1"
    env["PATH"] = str(runtime) + os.pathsep + env.get("PATH", "")
    started = time.monotonic()
    try:
        for index, port in enumerate((42631, 42632, 42633) if args.replica else (42631, 42632)):
            cache = Path(".runtime/experiment-026") / f"logical-rpc-cache-{index}"
            cache.mkdir(parents=True, exist_ok=True)
            worker_env = {**env, "LLAMA_CACHE": str(cache.resolve())}
            log = (root / f"worker-{index}.log").open("wb")
            logs.append(log)
            cmd = [str(runtime / "ggml-rpc-server.exe"), "-H", "127.0.0.1", "-p", str(port), "-d", "CUDA0", "-c"]
            proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=worker_env,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append(proc)
            write_once(root / f"worker-{index}.json", {"pid": proc.pid, "command": cmd, "timestamp": utc_now(),
                                                      "scope": "ONE_PHYSICAL_GPU_LOGICAL_WORKERS", "cache": str(cache.resolve())})
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("Logical RPC worker failed")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() - started > 60:
                        raise TimeoutError("RPC worker startup")
                    time.sleep(.2)
        if args.replica:
            log = (root / "replica-proxy.log").open("wb")
            logs.append(log)
            proxy = subprocess.Popen([sys.executable, "-m", "swarm_inference.experiments.experiment_026.rpc_replica",
                                      "--port", "42634", "--primary", "127.0.0.1:42631", "--standby", "127.0.0.1:42633",
                                      "--log", str(root / "replication.jsonl")], stdout=log, stderr=log, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append(proxy)
            time.sleep(1)
        cmd = [sys.executable, "-m", "swarm_inference.experiments.experiment_026.benchmark",
               "--run-id", args.run_id, "--runtime", str(runtime), "--rpc", ("127.0.0.1:42634,127.0.0.1:42632" if args.replica else "127.0.0.1:42631,127.0.0.1:42632"),
               "--tensor-split", ("1,1,2" if args.replica else "1,1,1"), "--prompt-ids", "development-01-factual,development-02-code,development-06-retrieval",
               "--n-predict", str(args.n_predict), "--spec-type", args.spec_type, "--depth", str(args.depth)]
        if args.spec_type != "none":
            cmd += ["--draft", str(Path(".runtime/experiment-026/models/mtp-Qwen3.8-27B-Q8_0.gguf").resolve())]
        result = subprocess.Popen(cmd, env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        processes.append(result)
        killed = False
        while result.poll() is None:
            if args.replica and not killed:
                events = Path("artifacts/experiment-026/runs") / args.run_id / "development-01-factual/events.jsonl"
                if events.exists():
                    count = 0
                    for line in events.read_text().splitlines():
                        try:
                            count += len(json.loads(line)["event"].get("tokens", []))
                        except json.JSONDecodeError:
                            pass
                    if count >= 20:
                        stamp = utc_now()
                        processes[0].kill()
                        processes[0].wait(timeout=10)
                        write_once(root / "kill.json", {"timestamp": stamp, "pid": processes[0].pid, "committed_tokens_observed": count,
                                                        "worker_notice": False, "scope": "LOCAL_PROCESS_FAULT_INJECTION"})
                        killed = True
            if time.monotonic() - started > 900:
                raise TimeoutError("Local distributed trial deadline")
            time.sleep(.02)
        write_once(root / "result.json", {"returncode": result.returncode, "seconds": time.monotonic() - started,
                                         "classification": "LOCAL_RPC_DEVELOPMENT_NOT_WAN", "physical_machines": 1})
        if result.returncode:
            raise RuntimeError("Local RPC benchmark failed")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
