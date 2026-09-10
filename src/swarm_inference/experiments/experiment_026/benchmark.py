"""Real llama.cpp server runs with raw SSE, tokens, timings, and provenance."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import EXPERIMENT_ID
from .io import append_event, digest, file_digest, utc_now, write_once


def inject_failure(plan: dict, folder: Path, seen_tokens: int) -> dict:
    from .remote import nodes,ssh
    node=next((row for row in nodes() if row["id"]==plan["node_id"]),None)
    if node is None:raise RuntimeError("Failure target is not an active owned node")
    pid=int(plan["pid"])
    observed=ssh(node,f"ps -p {pid} -o args=",check=False).stdout.decode().strip()
    if plan["expected_command"] not in observed:
        raise RuntimeError("Failure target PID is absent or reused; refusing kill")
    before={"event":"FAILURE_INJECTION_INITIATED","timestamp":utc_now(),"monotonic_ns":time.perf_counter_ns(),
            "node_id":node["id"],"pid":pid,"observed_command":observed,"committed_tokens_observed":seen_tokens}
    append_event(folder/"failure_injection.jsonl",before)
    start=time.perf_counter()
    ssh(node,f"kill -KILL {pid}")
    after={"event":"FAILURE_INJECTION_COMPLETED","timestamp":utc_now(),"monotonic_ns":time.perf_counter_ns(),
           "node_id":node["id"],"pid":pid,"kill_command_seconds":time.perf_counter()-start,
           "committed_tokens_observed":seen_tokens}
    append_event(folder/"failure_injection.jsonl",after)
    return {"initiated":before,"completed":after}


def post(base: str, endpoint: str, value: dict, timeout: float = 120):
    req = urllib.request.Request(base + endpoint, data=json.dumps(value).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def percentile(values: list[float], q: float):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def run_prompt(base: str, prompt: dict, folder: Path, config: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    messages = [{"role": "user", "content": prompt["content"]}]
    with post(base, "/apply-template", {"messages": messages, "add_generation_prompt": True,
                                       "chat_template_kwargs": {"enable_thinking": False}}) as response:
        rendered = json.load(response)["prompt"]
    with post(base, "/tokenize", {"content": rendered, "add_special": True}) as response:
        prompt_tokens = json.load(response)["tokens"]
    write_once(folder / "prompt.json", {"prompt_id": prompt["prompt_id"], "rendered": rendered,
                                       "tokens": prompt_tokens, "token_sha256": digest(prompt_tokens)})
    payload = {"prompt": prompt_tokens, "n_predict": config.get("n_predict") or prompt["generation_limit"],
               "temperature": 0, "seed": 260026, "repeat_penalty": 1.0,
               "presence_penalty": 0, "frequency_penalty": 0, "cache_prompt": False,
               "stream": True, "return_tokens": True, "timings_per_token": True,
               "n_probs": config.get("n_probs", 0), "id_slot": 0}
    write_once(folder / "request.json", payload)
    t0 = time.perf_counter()
    events, tokens, arrivals = [], [], []
    final = None
    injection = None
    with post(base, "/completion", payload, timeout=config.get("request_timeout", 600)) as response:
        for line in response:
            elapsed = time.perf_counter() - t0
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            record = {"elapsed_s": elapsed, "event": event}
            events.append(record)
            append_event(folder / "events.jsonl", record)
            ids = event.get("tokens", [])
            if ids and not event.get("stop"):
                tokens.extend(ids)
                arrivals.extend([elapsed] * len(ids))
                plan=config.get("failure_plan")
                if plan and injection is None and len(tokens)>=plan["after_tokens"]:
                    injection=inject_failure(plan,folder,len(tokens))
            if event.get("stop"):
                final = event
    duration = time.perf_counter() - t0
    if not final or not tokens:
        raise RuntimeError("Incomplete generation; no terminal event or no token IDs")
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    timings = final.get("timings", {})
    predicted = final.get("tokens_predicted", len(tokens))
    if predicted != len(tokens):
        raise RuntimeError(f"Token accounting mismatch: {predicted} vs {len(tokens)}")
    write_once(folder / "output.json", {"tokens": tokens, "token_sha256": digest(tokens),
                                       "content": "".join(x["event"].get("content", "") for x in events),
                                       "final": final})
    # A speculative block may arrive at once. Preserve these user-visible gaps;
    # throughput uses the complete decode interval, never inverse median gaps.
    decode_seconds = arrivals[-1] - arrivals[0]
    row = {"experiment_id": EXPERIMENT_ID, "timestamp": utc_now(), "run_id": folder.parent.name,
           "configuration_hash": digest(config), "configuration": config,
           "evidence_class": "PHYSICAL", "topology_class": config.get("topology_class", "LOCAL_SINGLE_MACHINE"),
           "prompt_id": prompt["prompt_id"], "prompt_hash": prompt["content_sha256"],
           "context_tokens": len(prompt_tokens), "generated_tokens": len(tokens),
           "output_token_hash": digest(tokens), "ttft_s": arrivals[0], "total_s": duration,
           "decode_tok_s": (len(tokens) - 1) / decode_seconds if decode_seconds > 0 else None,
           "median_tpot_s": statistics.median(gaps) if gaps else None,
           "p95_tpot_s": percentile(gaps, .95), "token_arrivals_s": arrivals,
           "prompt_tok_s": timings.get("prompt_per_second"), "server_timings": timings,
           "target_traversals": None, "wan_traversals": 0 if not config.get("rpc") else None,
           "bytes_sent": None, "bytes_received": None, "speculative_proposals": timings.get("draft_n"),
           "speculative_acceptances": timings.get("draft_n_accepted"),
           "tokens_per_traversal": None, "state_size_bytes": None, "correctness": "PENDING_CONTROL_COMPARISON",
           "vast_cost_usd": 0.0 if not config.get("rpc") else None,"failure_injection":injection}
    write_once(folder / "metrics.json", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiment-026"))
    parser.add_argument("--runtime", type=Path, default=Path(".runtime/experiment-026/b10886"))
    parser.add_argument("--model", type=Path, default=Path(".runtime/experiment-026/models/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec-type", default="none")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--prompt-ids", default="")
    parser.add_argument("--n-predict", type=int)
    parser.add_argument("--n-probs", type=int, default=0)
    parser.add_argument("--port", type=int, default=42626)
    parser.add_argument("--rpc", default="")
    parser.add_argument("--tensor-split", default="")
    parser.add_argument("--topology-class", default="")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--residency", default="LOCAL_DISK_PRESENT_OS_PAGE_CACHE_UNCONTROLLED")
    parser.add_argument("--deployment",type=Path)
    parser.add_argument("--failure-plan",type=Path)
    parser.add_argument("--ctx-checkpoints",type=int)
    parser.add_argument("--split", choices=("development", "sealed"), default="development")
    args = parser.parse_args()
    if args.split == "sealed" and not (args.root / "seal/configuration.json").is_file():
        raise RuntimeError("Final prompts cannot run before configuration sealing")
    run = args.root / "runs" / args.run_id
    run.mkdir(parents=True, exist_ok=False)
    server = args.runtime.resolve() / "llama-server.exe"
    if not server.is_file():
        server = args.runtime.resolve() / "llama-server"
    command = [str(server), "-m", str(args.model.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
               "-ngl", "99", "-c", "12288", "-b", "512", "-ub", "512", "-np", "1", "-fa", "on",
               "--fit", "off", "--metrics", "--spec-type", args.spec_type, "--no-webui"]
    if args.spec_type != "none":
        command += ["--spec-draft-n-max", str(args.depth)]
    if args.ctx_checkpoints is not None:command += ["--ctx-checkpoints",str(args.ctx_checkpoints)]
    if args.draft:
        command += ["-md", str(args.draft.resolve()), "-ngld", "99"]
    if args.rpc:
        command += ["--rpc", args.rpc, "--split-mode", "layer", "--tensor-split", args.tensor_split]
    model_receipt = json.loads((args.root / "acquisition" / f"{args.model.name}.json").read_text())
    if model_receipt["status"] != "PASS" or args.model.stat().st_size != model_receipt["size_bytes"]:
        raise RuntimeError("Missing or invalid exact-model receipt")
    config = {"command": command, "model_sha256": model_receipt["sha256"], "spec_type": args.spec_type,
              "depth": args.depth, "n_predict": args.n_predict, "n_probs": args.n_probs, "rpc": args.rpc,
              "binary_sha256": file_digest(server), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    config["topology_class"] = args.topology_class or ("LOCAL_LOGICAL_RPC" if args.rpc else "LOCAL_SINGLE_MACHINE")
    config["runtime_files_sha256"] = {p.name: file_digest(p) for p in sorted(args.runtime.glob("*.dll"))}
    config["llama_commit"] = subprocess.check_output(["git", "-C", ".runtime/e026-llama.cpp", "rev-parse", "HEAD"], text=True).strip()
    config["llama_diff_sha256"] = digest(subprocess.check_output(["git", "-C", ".runtime/e026-llama.cpp", "diff", "HEAD"], text=True))
    config["dirty_tree"] = subprocess.check_output(["git", "status", "--porcelain=v1"], text=True)
    config["instrumentation_environment"] = {k: os.environ.get(k) for k in ("E026_TRACE", "GGML_RPC_NO_RDMA")}
    config["residency"] = args.residency
    config["deployment"] = json.loads(args.deployment.read_text()) if args.deployment else None
    config["failure_plan"] = json.loads(args.failure_plan.read_text()) if args.failure_plan else None
    config["orchestration_sha256"] = {p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    config["request_timeout"] = 1800
    write_once(run / "configuration.json", config)
    stdout = (run / "server.stdout.log").open("wb")
    stderr = (run / "server.stderr.log").open("wb")
    env = dict(os.environ)
    env["PATH"] = str(args.runtime.resolve()) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    write_once(run / "process.json", {"pid": proc.pid, "timestamp": utc_now(), "command": command})
    base = f"http://127.0.0.1:{args.port}"
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited {proc.returncode}; inspect {run}/server.stderr.log")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.perf_counter() - start > args.startup_timeout:
                raise TimeoutError("Server readiness exceeded configured timeout")
            time.sleep(.2)
        write_once(run / "startup.json", {"timestamp": utc_now(), "ready_seconds": time.perf_counter() - start,
                                          "residency": args.residency})
        prompts = json.loads((args.root / "corpus" / f"{args.split}.json").read_text())["prompts"]
        ids = set(args.prompt_ids.split(",")) if args.prompt_ids else None
        for prompt in prompts:
            if ids and prompt["prompt_id"] not in ids:
                continue
            row = run_prompt(base, prompt, run / prompt["prompt_id"], config)
            append_event(args.root / "runs.jsonl", row)
            print(json.dumps({k: row[k] for k in ("prompt_id", "context_tokens", "generated_tokens", "ttft_s", "decode_tok_s", "server_timings", "output_token_hash")}), flush=True)
    except BaseException as error:
        write_once(run / "failure.json", {"timestamp": utc_now(), "type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        stdout.close()
        stderr.close()


if __name__ == "__main__":
    main()
