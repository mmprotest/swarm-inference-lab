"""Authenticated SSH/bootstrap helpers for recorded E027 Vast leases only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from ..experiment_026.io import append_event, utc_now
from .vast import ROOT as COST_ROOT, snapshot


ROOT = Path("artifacts/experiment-027/remote")
RUNTIME = Path(".runtime/experiment-027")
IDENTITY = Path(".keys/e026_ed25519")
KNOWN_HOSTS = Path(".keys/e027_known_hosts")


def leases() -> dict[str, dict[str, Any]]:
    folder = COST_ROOT / "leases"
    return {path.stem: json.loads(path.read_text()) for path in folder.glob("*.json")}


def nodes() -> list[dict[str, Any]]:
    owned = leases()
    result: list[dict[str, Any]] = []
    for row in snapshot()["owned_instances"]:
        lease = owned.get(str(row.get("label")))
        if not lease:
            continue
        mapping = ((row.get("ports") or {}).get("22/tcp") or [])
        port = mapping[0].get("HostPort") if mapping else None
        host = row.get("public_ipaddr")
        if not host or not port:
            host, port = row.get("ssh_host"), row.get("ssh_port")
        if not host or not port:
            continue
        result.append({**row, "role": lease["role"], "host": host, "port": int(port)})
    return sorted(result, key=lambda item: item["role"])


def ssh_args(node: dict[str, Any]) -> list[str]:
    if node.get("label") not in leases():
        raise ValueError("unowned E027 SSH endpoint")
    return [
        "-i", str(IDENTITY.resolve()),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}",
        "-o", "ConnectTimeout=12",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
    ]


def ssh(node: dict[str, Any], command: str, *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    begin = time.perf_counter()
    result = subprocess.run(
        ["ssh", *ssh_args(node), "-p", str(node["port"]), f"root@{node['host']}", command],
        capture_output=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    append_event(ROOT / str(node["id"]) / "ssh.jsonl", {
        "timestamp": utc_now(), "command": command, "returncode": result.returncode,
        "elapsed_s": time.perf_counter() - begin,
        "stdout_tail": result.stdout.decode(errors="replace")[-4000:],
        "stderr_tail": result.stderr.decode(errors="replace")[-2000:],
    })
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1600:])
    return result


def upload(node: dict[str, Any], local: Path, remote: str, *, timeout: int = 600) -> None:
    begin = time.perf_counter()
    result = subprocess.run(
        ["scp", *ssh_args(node), "-P", str(node["port"]), str(local), f"root@{node['host']}:{remote}"],
        capture_output=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    append_event(ROOT / str(node["id"]) / "transfers.jsonl", {
        "timestamp": utc_now(), "direction": "UPLOAD", "local": str(local),
        "remote": remote, "size_bytes": local.stat().st_size,
        "elapsed_s": time.perf_counter() - begin, "returncode": result.returncode,
        "stderr_tail": result.stderr.decode(errors="replace")[-1200:],
    })
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1600:])


def wait_ssh(expected: int = 2, timeout_s: int = 600) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    ready: dict[int, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        for node in nodes():
            if node["id"] in ready:
                continue
            try:
                if ssh(node, "true", timeout=20, check=False).returncode == 0:
                    ready[node["id"]] = node
            except subprocess.TimeoutExpired:
                pass
        if len(ready) >= expected:
            return sorted(ready.values(), key=lambda item: item["role"])
        time.sleep(10)
    raise TimeoutError("two E027 SSH workers did not become ready")


def bootstrap_one(node: dict[str, Any]) -> dict[str, Any]:
    ssh(node, "mkdir -p /workspace/e027")
    upload(node, RUNTIME / "remote-source.tar.gz", "/workspace/e027/remote-source.tar.gz")
    upload(node, Path("scripts/experiment_027_remote_bootstrap.sh"), "/workspace/e027/bootstrap.sh")
    command = "cd /workspace/e027 && nohup bash bootstrap.sh > bootstrap.log 2>&1 < /dev/null & echo $!"
    result = ssh(node, command)
    return {"id": node["id"], "role": node["role"], "pid": int(result.stdout.strip())}


def bootstrap() -> None:
    ready = wait_ssh()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(bootstrap_one, ready):
            print(json.dumps(row), flush=True)


def wait_build(timeout_s: int = 1800) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status: list[dict[str, Any]] = []
        for node in nodes():
            result = ssh(
                node,
                "if test -f /workspace/e027/READY; then echo READY; else tail -n 8 /workspace/e027/bootstrap.log; fi",
                timeout=30,
                check=False,
            )
            text = result.stdout.decode(errors="replace")
            status.append({"id": node["id"], "role": node["role"], "ready": "READY" in text, "tail": text[-1500:]})
        print(json.dumps(status), flush=True)
        if len(status) == 2 and all(row["ready"] for row in status):
            return status
        time.sleep(20)
    raise TimeoutError("E027 remote build/model acquisition deadline exceeded")


def start_stage(node: dict[str, Any]) -> dict[str, Any]:
    bounds = {"b": (22, 44), "c": (44, 64)}[node["role"]]
    start, end = bounds
    command = (

        "cd /workspace/e027/source; rm -f /workspace/e027/stage.log; "
        "setsid -f env LD_LIBRARY_PATH=build/bin build/bin/llama-e027-stage "
        "--model /workspace/e027/models/Qwen3.8-27B-Q4_K_M.gguf "
        f"--host 127.0.0.1 --port 19300 --stage-start {start} --stage-end {end} "
        "--n-ctx 4096 --n-batch 512 --n-ubatch 512 --n-rs-seq 0 --gpu-layers 999 --serial-blocks "
        "> /workspace/e027/stage.log 2>&1 < /dev/null; echo launched"
    )
    result = ssh(node, command)
    return {"id": node["id"], "role": node["role"], "status": result.stdout.decode().strip()}

def start_stages() -> None:
    current = nodes()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(start_stage, current):
            print(json.dumps(row), flush=True)


def wait_stage(timeout_s: int = 240) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rows = []
        for node in nodes():
            result = ssh(node, "grep -m1 E027_READY /workspace/e027/stage.log || tail -n 3 /workspace/e027/stage.log", timeout=30, check=False)
            output = result.stdout.decode(errors="replace")
            rows.append({"id": node["id"], "role": node["role"], "ready": "E027_READY" in output, "output": output[-1200:]})
        print(json.dumps(rows), flush=True)
        if len(rows) == 2 and all(row["ready"] for row in rows):
            return
        time.sleep(10)
    raise TimeoutError("E027 stage servers did not become ready")


def start_tunnel(node: dict[str, Any]) -> dict[str, Any]:
    local_port = {"b": 19301, "c": 19302}[node["role"]]
    log_path = RUNTIME / f"tunnel-{node['role']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    process = subprocess.Popen(
        ["ssh", *ssh_args(node), "-p", str(node["port"]), "-N",
         "-o", "ExitOnForwardFailure=yes", "-L", f"{local_port}:127.0.0.1:19300", f"root@{node['host']}"],
        stdout=log,
        stderr=log,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    log.close()
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"E027 tunnel {node['role']} exited early")
    receipt = {"id": node["id"], "role": node["role"], "pid": process.pid, "local_port": local_port}
    (RUNTIME / f"tunnel-{node['role']}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def start_tunnels() -> None:
    for node in nodes():
        print(json.dumps(start_tunnel(node)), flush=True)


def command_all(command: str) -> None:
    for node in nodes():
        result = ssh(node, command, check=False)
        print(json.dumps({"id": node["id"], "role": node["role"], "returncode": result.returncode,
                          "stdout": result.stdout.decode(errors="replace"),
                          "stderr": result.stderr.decode(errors="replace")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("nodes", "bootstrap", "wait-build", "start-stages", "wait-stages", "start-tunnels", "command"))
    parser.add_argument("--command")
    args = parser.parse_args()
    if args.action == "nodes":
        print(json.dumps(nodes(), indent=2))
    elif args.action == "bootstrap":
        bootstrap()
    elif args.action == "wait-build":
        wait_build()
    elif args.action == "start-stages":
        start_stages()
    elif args.action == "wait-stages":
        wait_stage()
    elif args.action == "start-tunnels":
        start_tunnels()
    else:
        command_all(args.command or "true")


if __name__ == "__main__":
    main()
