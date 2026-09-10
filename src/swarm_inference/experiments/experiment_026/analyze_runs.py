"""Mechanical token comparisons and RPC protocol trace summaries."""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
import statistics
from pathlib import Path

from .io import write_once


def summarize_trace(path: Path) -> dict:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"E026_(DECODE|RPC_SEND|RPC_WAIT) (.*)", line)
        if not match:
            continue
        data = dict(re.findall(r"(\w+)=([^ ]+)", match[2]))
        for key in ("kind", "n", "begin_us", "end_us", "ret", "cmd", "bytes", "received_bytes"):
            if key in data:
                data[key] = int(data[key])
        data["event"] = match[1]
        events.append(data)
    decodes = [x for x in events if x["event"] == "DECODE" and x.get("kind") == 0]
    intervals = []
    for left, right in zip(decodes, decodes[1:]):
        if left["n"] != 1 or right["n"] != 1:
            continue
        selected = [x for x in events if left["begin_us"] <= x["begin_us"] < right["begin_us"]]
        waits = [x for x in selected if x["event"] == "RPC_WAIT"]
        sends = [x for x in selected if x["event"] == "RPC_SEND"]
        intervals.append({"begin_us": left["begin_us"], "cycle_us": right["begin_us"] - left["begin_us"],
                          "send_bytes": sum(x["bytes"] for x in sends),
                          "receive_bytes": sum(x["received_bytes"] for x in waits),
                          "request_counts": dict(collections.Counter(x["cmd"] for x in sends)),
                          "response_counts": dict(collections.Counter(x["cmd"] for x in waits)),
                          "sum_response_wait_us_may_overlap": sum(x["end_us"] - x["begin_us"] for x in waits)})
    counts = collections.Counter(x["cmd"] for x in events if x["event"] == "RPC_SEND")
    return {"trace_events": len(events), "target_api_calls": len(decodes),
            "target_tokens_per_api_call": [x["n"] for x in decodes],
            "rpc_request_counts_including_startup": dict(counts),
            "total_sent_protocol_bytes": sum(x.get("bytes", 0) for x in events),
            "total_received_protocol_bytes": sum(x.get("received_bytes", 0) for x in events),
            "steady_single_token_intervals": intervals,
            "median_single_token_cycle_ms": statistics.median(x["cycle_us"] for x in intervals) / 1000 if intervals else None,
            "median_single_token_sent_bytes": statistics.median(x["send_bytes"] for x in intervals) if intervals else None,
            "median_single_token_received_bytes": statistics.median(x["receive_bytes"] for x in intervals) if intervals else None,
            "byte_definition": "RPC protocol bytes, excluding TCP/IP/TLS/SSH framing",
            "timing_definition": "Client process monotonic clock; wait sums are not a critical-path decomposition"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--control", default="local-target-001")
    args = parser.parse_args()
    root = Path("artifacts/experiment-026")
    run = root / "runs" / args.run_id
    rows = []
    for folder in sorted(run.iterdir()):
        if not folder.is_dir() or not (folder / "output.json").exists():
            continue
        out = json.loads((folder / "output.json").read_text())["tokens"]
        ref = json.loads((root / "runs" / args.control / folder.name / "output.json").read_text())["tokens"]
        mismatch = next((i for i, (a, b) in enumerate(zip(ref, out)) if a != b), None)
        metrics = json.loads((folder / "metrics.json").read_text())
        rows.append({"prompt_id": folder.name, "compared_tokens": min(len(ref), len(out)),
                     "prefix_equal": out == ref[:len(out)], "first_mismatch": mismatch,
                     "decode_tok_s": metrics["decode_tok_s"], "timings": metrics["server_timings"]})
    result = {"run_id": args.run_id, "control": args.control, "comparisons": rows,
              "trace": summarize_trace(run / "server.stderr.log")}
    write_once(root / "analysis" / (args.run_id + ".json"), result)
    print(json.dumps({"comparisons": rows, "trace_summary": {k: v for k, v in result["trace"].items()
                                                             if k not in {"steady_single_token_intervals", "target_tokens_per_api_call"}}}), flush=True)


if __name__ == "__main__":
    main()
