"""Bounded parallel HTTP ranges with retained progress and final SHA256 check."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path

from .io import append_event, file_digest, utc_now, write_once
from .preflight import MODEL_FILE, MODEL_REPO, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default=MODEL_FILE)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path("artifacts/experiment-026")
    directory = Path(".runtime/experiment-026/models")
    directory.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "preflight/model-source.json").read_text())
    item = next(x for x in info["siblings"] if x["rfilename"] == args.filename)
    expected, size = item["lfs"]["sha256"], item["size"]
    target = directory / args.filename
    part = directory / (args.filename + ".ranges-part")
    progress = root / "acquisition" / (args.filename + ".ranges.jsonl")
    if target.exists():
        if target.stat().st_size != size or file_digest(target) != expected:
            raise RuntimeError("Existing target failed exact integrity check")
        print("Already verified", flush=True)
        return
    if not progress.exists():
        prefix = 0
        candidates = list((directory / ".cache/huggingface/download").glob(f"*.{expected}.incomplete"))
        if candidates:
            source = candidates[0]
            prefix = source.stat().st_size
            if prefix > size:
                raise RuntimeError("Oversized partial acquisition")
            source.rename(part)
        else:
            with part.open("xb"):
                pass
        append_event(progress, {"event": "PREFIX", "bytes": prefix, "timestamp": utc_now()})
    entries = [json.loads(line) for line in progress.read_text().splitlines()]
    prefix = entries[0]["bytes"]
    done = {(r["start"], r["end"]) for r in entries if r["event"] == "RANGE_OK"}
    chunk_size = 8 * 1024 * 1024
    ranges = [(start, min(size - 1, start + chunk_size - 1)) for start in range(prefix, size, chunk_size)]
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{args.filename}"
    t0 = time.monotonic()

    def fetch(bounds):
        start, end = bounds
        for attempt in range(4):
            started = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Server did not honor exact byte range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Range length mismatch")
                with part.open("r+b", buffering=0) as stream:
                    stream.seek(start)
                    stream.write(data)
                    os.fsync(stream.fileno())
                return {"event": "RANGE_OK", "start": start, "end": end,
                        "bytes": len(data), "seconds": time.monotonic() - started,
                        "timestamp": utc_now(), "attempt": attempt}
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 4))

    pending = [bounds for bounds in ranges if bounds not in done]
    completed_bytes = prefix + sum(end - start + 1 for start, end in done)
    last_print = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, bounds) for bounds in pending]
        for future in concurrent.futures.as_completed(futures):
            receipt = future.result()
            append_event(progress, receipt)
            completed_bytes += receipt["bytes"]
            if time.monotonic() - last_print >= 15 or completed_bytes == size:
                print(json.dumps({"filename": args.filename, "verified_ranges_bytes": completed_bytes,
                                  "total_bytes": size, "elapsed_s": time.monotonic() - t0}), flush=True)
                last_print = time.monotonic()
    actual = file_digest(part)
    if actual != expected or part.stat().st_size != size:
        raise RuntimeError("Complete SHA256/size check failed; preserve partial evidence")
    part.rename(target)
    write_once(root / "acquisition" / (args.filename + ".json"), {
        "experiment_id": "E026_Q27_WAN_SWARM_INTEGRATED_PROOF", "timestamp": utc_now(),
        "model_source": f"https://huggingface.co/{MODEL_REPO}", "revision": MODEL_REVISION,
        "filename": args.filename, "path": str(target.resolve()), "sha256": actual,
        "expected_sha256": expected, "size_bytes": size, "status": "PASS",
        "acquisition_method": "RESUMABLE_HTTP_RANGES", "resumed_prefix_bytes": prefix,
        "seconds_this_invocation": time.monotonic() - t0})
    print(json.dumps({"status": "PASS", "filename": args.filename, "sha256": actual}), flush=True)


if __name__ == "__main__":
    main()
