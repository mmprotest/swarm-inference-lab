"""Physical complete hybrid-state restart test, never prompt-zero replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zlib
from pathlib import Path

from .io import append_event, digest, file_digest, utc_now, write_once


class Probe:
    def __init__(self, binary: Path, config: Path, folder: Path):
        self.folder = folder
        self.log = (folder / "probe.stderr.log").open("wb")
        env = dict(os.environ)
        env["PATH"] = str(binary.parent.resolve()) + os.pathsep + env.get("PATH", "")
        self.process = subprocess.Popen([str(binary.resolve()), str(config.resolve())],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", env=env,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.ready = self.read()
        write_once(folder / "ready.json", self.ready)

    def read(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Probe terminated; inspect {self.folder}")
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def call(self, request):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self.read()
        append_event(self.folder / "commands.jsonl", {"timestamp": utc_now(), "request": request,
                                                     "response": result, "elapsed_s": time.perf_counter() - start})
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-id", default="development-06-retrieval")
    args = parser.parse_args()
    root = Path("artifacts/experiment-026")
    folder = root / "state" / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    binary = Path(".runtime/experiment-026/build/bin/e026-probe.exe")
    model = Path(".runtime/experiment-026/models/Qwen3.8-27B-Q4_K_M.gguf")
    cfg = {"model": str(model.resolve()), "n_ctx": 12288, "n_batch": 512, "n_ubatch": 512}
    write_once(folder / "config.json", cfg)
    tokens = json.loads((root / "runs/local-target-001" / args.prompt_id / "prompt.json").read_text())["tokens"]
    control_dir = folder / "control"
    control_dir.mkdir()
    proc = Probe(binary, folder / "config.json", control_dir)
    checkpoint = Path(".runtime/experiment-026") / (args.run_id + ".state.bin")
    try:
        out = proc.call({"op": "decode", "tokens": tokens})
        prefix = []
        for _ in range(64):
            prefix.append(out["greedy_token"])
            out = proc.call({"op": "decode", "tokens": [out["greedy_token"]]})
        pending_token = out["greedy_token"]
        save = proc.call({"op": "save", "path": str(checkpoint.resolve())})
        position = save["position"]
        target, target_logits = [], []
        for _ in range(128):
            target.append(out["greedy_token"])
            out = proc.call({"op": "decode", "tokens": [out["greedy_token"]]})
            target_logits.append(out["top_logits"])
        write_once(folder / "control.json", {"prefix": prefix, "pending_token": pending_token,
                                             "continuation": target, "top_logits": target_logits})
        kill_time = utc_now()
        proc.process.kill()
        proc.process.wait(timeout=10)
    finally:
        proc.close()
    recovery_dir = folder / "replacement"
    recovery_dir.mkdir()
    recovery_start = time.perf_counter()
    replacement = Probe(binary, folder / "config.json", recovery_dir)
    try:
        restore = replacement.call({"op": "restore", "path": str(checkpoint.resolve()), "position": position})
        recovered, errors = [], []
        next_token = pending_token
        first_resumed_s = None
        for index in range(128):
            recovered.append(next_token)
            out = replacement.call({"op": "decode", "tokens": [next_token]})
            if first_resumed_s is None:
                first_resumed_s = time.perf_counter() - recovery_start
            if [x["id"] for x in out["top_logits"]] != [x["id"] for x in target_logits[index]]:
                errors.append({"index": index, "top_k_identity": False})
            else:
                delta = max(abs(x["logit"] - y["logit"]) for x, y in zip(out["top_logits"], target_logits[index]))
                if delta:
                    errors.append({"index": index, "max_top_logit_error": delta})
            next_token = out["greedy_token"]
    finally:
        replacement.close()
    raw = checkpoint.read_bytes()
    compression_start = time.perf_counter()
    compressed = zlib.compress(raw, level=1)
    compression_s = time.perf_counter() - compression_start
    decompress_start = time.perf_counter()
    roundtrip = zlib.decompress(compressed)
    decompress_s = time.perf_counter() - decompress_start
    receipt = {"experiment_id": "E026_Q27_WAN_SWARM_INTEGRATED_PROOF", "evidence_class": "PHYSICAL",
               "scope": "LOCAL_SINGLE_MACHINE_PROCESS_REPLACEMENT", "timestamp": utc_now(),
               "prompt_id": args.prompt_id, "prompt_tokens": len(tokens), "prefix_generated_tokens": len(prefix),
               "resumed_tokens": len(recovered), "checkpoint_position": position, "save": save, "restore": restore,
               "kill_timestamp": kill_time, "restart_to_first_resumed_s": first_resumed_s,
               "full_prompt_replay_tokens": 0, "restored_continuation_equal": recovered == target,
               "control_hash": digest(target), "recovered_hash": digest(recovered),
               "numerical_discrepancies": errors, "checkpoint_sha256": file_digest(checkpoint),
               "lossless_zlib_level1_bytes": len(compressed), "compression_s": compression_s,
               "decompression_s": decompress_s, "compression_roundtrip_exact": roundtrip == raw,
               "status": "PASS" if recovered == target and not errors else "FAIL"}
    write_once(folder / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
