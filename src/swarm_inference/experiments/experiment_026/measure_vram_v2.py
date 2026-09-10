"""Health-gated GPU residency measurement for a frozen local configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .io import utc_now, write_once


QUERY = "name,uuid,memory.total,memory.used,driver_version"


def gpu() -> dict:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    name, uuid, total, used, driver = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    return {"name": name, "uuid": uuid, "memory_total_mib": int(total),
            "memory_used_mib": int(used), "driver_version": driver}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text(encoding="utf-8"))
    command = list(configuration["command"])
    command[command.index("--port") + 1] = str(args.port)

    vacancy = socket.socket()
    try:
        vacancy.bind(("127.0.0.1", args.port))
    except OSError as error:
        raise RuntimeError(f"Refusing occupied measurement port {args.port}") from error
    finally:
        vacancy.close()

    args.output.mkdir(parents=True, exist_ok=False)
    before = gpu()
    started = time.perf_counter()
    with (args.output / "server.stdout.log").open("xb") as stdout, (args.output / "server.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            deadline = time.monotonic() + 60
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                        health = json.load(response)
                        if response.status == 200 and health.get("status") == "ok":
                            break
                except (OSError, urllib.error.URLError, json.JSONDecodeError):
                    pass
                if process.poll() is not None:
                    raise RuntimeError("llama-server exited before model readiness")
                if time.monotonic() >= deadline:
                    raise TimeoutError("llama-server model readiness")
                time.sleep(0.25)
            if process.poll() is not None:
                raise RuntimeError("llama-server exited at readiness boundary")
            time.sleep(0.5)
            ready_s = time.perf_counter() - started
            loaded = gpu()
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    after = gpu()
    result = {
        "timestamp": utc_now(),
        "evidence_class": "PHYSICAL_POST_SEAL_RESOURCE_MEASUREMENT",
        "configuration": str(args.configuration),
        "command": command,
        "health": health,
        "ready_seconds": ready_s,
        "before": before,
        "loaded": loaded,
        "after": after,
        "loaded_minus_before_mib": loaded["memory_used_mib"] - before["memory_used_mib"],
        "does_not_modify_sealed_policy": True,
    }
    write_once(args.output / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
