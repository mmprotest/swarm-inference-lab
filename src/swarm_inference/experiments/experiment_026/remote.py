"""Direct authenticated SSH lifecycle for recorded E026 nodes only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

from .io import append_event, utc_now, write_once
from .vast import snapshot, leases

ROOT = Path("artifacts/experiment-026/remote")


def nodes():
    result = []
    for row in snapshot()["instances"]:
        lease = leases().get(row.get("label"))
        if not lease:
            continue
        ports = row.get("ports") or {}
        mapping = ports.get("22/tcp") or []
        if not mapping:
            continue
        node = {**row, "role": lease["role"], "host": row["public_ipaddr"], "port": int(mapping[0]["HostPort"])}
        result.append(node)
        path = ROOT / str(row["id"]) / "node.json"
        if not path.exists():
            write_once(path, node)
    return result


def ssh_args(node):
    if node["label"] not in leases():
        raise ValueError("Unowned remote endpoint")
    return ["-i", str(Path(".keys/e026_ed25519").resolve()), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=.keys/e026_known_hosts",
            "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"]


def ssh(node, command, timeout=90, check=True):
    start = time.perf_counter()
    result = subprocess.run(["ssh", *ssh_args(node), "-p", str(node["port"]), "root@" + node["host"], command],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    append_event(ROOT / str(node["id"]) / "ssh.jsonl", {"timestamp": utc_now(), "command": command,
                 "returncode": result.returncode, "elapsed_s": time.perf_counter() - start,
                 "stdout": result.stdout.decode(errors="replace"), "stderr": result.stderr.decode(errors="replace")})
    if check and result.returncode:
        raise RuntimeError(f"SSH {node['id']} failed: {result.stderr.decode(errors='replace')[-1200:]}")
    return result


def upload(node, local, remote):
    start = time.perf_counter()
    result = subprocess.run(["scp", *ssh_args(node), "-P", str(node["port"]), str(local),
                             f"root@{node['host']}:{remote}"], capture_output=True, timeout=240,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    append_event(ROOT / str(node["id"]) / "transfers.jsonl", {"timestamp": utc_now(), "direction": "UPLOAD",
                 "local": str(local), "remote": remote, "size_bytes": Path(local).stat().st_size,
                 "elapsed_s": time.perf_counter() - start, "returncode": result.returncode,
                 "stderr": result.stderr.decode(errors="replace")})
    if result.returncode:
        raise RuntimeError("SCP upload failed")


def bootstrap_one(node):
    folder = ROOT / str(node["id"])
    if (folder / "bootstrap-launched.json").exists():
        return {"id": node["id"], "status": "ALREADY_LAUNCHED"}
    ssh(node, "mkdir -p /workspace/e026")
    upload(node, ".runtime/experiment-026/remote-source-001.tar.gz", "/workspace/e026/remote-source-001.tar.gz")
    upload(node, "scripts/experiment_026_remote_bootstrap.sh", "/workspace/e026/bootstrap.sh")
    command = "cd /workspace/e026 && nohup bash bootstrap.sh > bootstrap.log 2>&1 < /dev/null &"
    result = ssh(node, command)
    write_once(folder / "bootstrap-launched.json", {"timestamp": utc_now(), "command": command,
                                                   "returncode": result.returncode})
    return {"id": node["id"], "status": "BUILD_LAUNCHED"}


def bootstrap():
    deadline = time.monotonic() + 900
    ready = []
    while time.monotonic() < deadline:
        current = nodes()
        for node in current:
            if node["id"] in [r["id"] for r in ready]:
                continue
            try:
                if ssh(node, "true", timeout=20, check=False).returncode == 0:
                    ready.append(node)
                    print(json.dumps({"id": node["id"], "status": "SSH_READY", "timestamp": utc_now()}), flush=True)
            except subprocess.TimeoutExpired:
                pass
        if len(ready) >= 2:
            break
        time.sleep(15)
    if len(ready) < 2:
        raise TimeoutError("Two direct SSH workers did not become ready")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(bootstrap_one, ready):
            print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("bootstrap", "build-status", "command"))
    p.add_argument("--role")
    p.add_argument("--command")
    args = p.parse_args()
    if args.action == "bootstrap":
        bootstrap()
        return
    for node in nodes():
        if args.role and args.role != node["role"]:
            continue
        command = args.command if args.action == "command" else "tail -n 12 /workspace/e026/bootstrap.log"
        result = ssh(node, command, check=False)
        print(json.dumps({"id": node["id"], "role": node["role"], "returncode": result.returncode,
                          "stdout": result.stdout.decode(errors="replace"), "stderr": result.stderr.decode(errors="replace")}))


if __name__ == "__main__":
    main()
