"""Turn an interrupted sealed stream into auditable partial-run evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .analyze_runs import summarize_trace
from .benchmark import percentile
from .io import digest, file_digest, write_once


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def probability_rows(rows: list[dict]) -> list[dict]:
    return [row["event"]["completion_probabilities"][0] for row in rows
            if row["event"].get("completion_probabilities") and not row["event"].get("stop")]


def compare_logits(reference_rows: list[dict], observed_rows: list[dict], first_mismatch: int | None) -> dict:
    left = probability_rows(reference_rows)
    right = probability_rows(observed_rows)
    count = min(len(left), len(right), first_mismatch + 1 if first_mismatch is not None else len(right))
    errors=[]
    jaccards=[]
    top1=0
    for a,b in zip(left[:count],right[:count]):
        atop={item["id"]:item["logprob"] for item in a["top_logprobs"]}
        btop={item["id"]:item["logprob"] for item in b["top_logprobs"]}
        common=atop.keys() & btop.keys()
        errors.extend(abs(atop[token]-btop[token]) for token in common)
        jaccards.append(len(common)/len(atop.keys() | btop.keys()))
        top1 += a["top_logprobs"][0]["id"] == b["top_logprobs"][0]["id"]
    detail=None
    if first_mismatch is not None and first_mismatch < count:
        a=left[first_mismatch]
        b=right[first_mismatch]
        detail={
            "zero_based_token_index":first_mismatch,
            "local_selected":{"id":a["id"],"token":a["token"],"logprob":a["logprob"]},
            "wan_selected":{"id":b["id"],"token":b["token"],"logprob":b["logprob"]},
            "local_top2":a["top_logprobs"][:2],
            "wan_top2":b["top_logprobs"][:2],
            "local_top1_margin_logprob":a["top_logprobs"][0]["logprob"]-a["top_logprobs"][1]["logprob"],
            "wan_top1_margin_logprob":b["top_logprobs"][0]["logprob"]-b["top_logprobs"][1]["logprob"],
        }
    return {
        "identical_history_positions":count,
        "top1_agreement":top1/count if count else None,
        "mean_common_top10_logprob_abs_delta":statistics.fmean(errors) if errors else None,
        "max_common_top10_logprob_abs_delta":max(errors) if errors else None,
        "mean_top10_jaccard":statistics.fmean(jaccards) if jaccards else None,
        "first_mismatch_detail":detail,
        "scope":"Only positions whose input history is identical, including the first differing decision",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="final-wan-sealed-001")
    parser.add_argument("--prompt-id", default="sealed-01-factual")
    parser.add_argument("--control", default="final-local-sealed-control-001")
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("artifacts/experiment-026/runs/final-wan-sealed-001/server.stderr.log"),
    )
    parser.add_argument(
        "--tunnel-log",
        type=Path,
        default=Path("artifacts/experiment-026/remote-collected-final-002/a/wan-router-services-final-001-peer-tunnel.log"),
    )
    args = parser.parse_args()

    root = Path("artifacts/experiment-026")
    run = root / "runs" / args.run_id
    folder = run / args.prompt_id
    stream = events(folder / "events.jsonl")
    reference_stream = events(root / "runs" / args.control / args.prompt_id / "events.jsonl")
    tokens = [token for row in stream if not row["event"].get("stop") for token in row["event"].get("tokens", [])]
    arrivals = [row["elapsed_s"] for row in stream if not row["event"].get("stop") for _ in row["event"].get("tokens", [])]
    reference = json.loads(
        (root / "runs" / args.control / args.prompt_id / "output.json").read_text(encoding="utf-8")
    )["tokens"]
    first_mismatch = next((i for i, (left, right) in enumerate(zip(reference, tokens)) if left != right), None)
    gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
    failure = json.loads((run / "failure.json").read_text(encoding="utf-8"))
    trace = summarize_trace(args.trace)
    result = {
        "run_id": args.run_id,
        "prompt_id": args.prompt_id,
        "control": args.control,
        "sealed_request_tokens": json.loads((folder / "request.json").read_text(encoding="utf-8"))["n_predict"],
        "completed_tokens": len(tokens),
        "terminal_event_observed": any(row["event"].get("stop") for row in stream),
        "partial_output_token_hash": digest(tokens),
        "compared_tokens": min(len(reference), len(tokens)),
        "prefix_equal": tokens == reference[: len(tokens)],
        "first_mismatch": first_mismatch,
        "identical_history_logits": compare_logits(reference_stream, stream, first_mismatch),
        "ttft_s": arrivals[0],
        "last_token_arrival_s": arrivals[-1],
        "partial_decode_tok_s": (len(tokens) - 1) / (arrivals[-1] - arrivals[0]),
        "median_tpot_s": statistics.median(gaps),
        "p95_tpot_s": percentile(gaps, 0.95),
        "failure": failure,
        "trace": trace,
        "tunnel_log_sha256": file_digest(args.tunnel_log),
        "tunnel_log": args.tunnel_log.read_text(encoding="utf-8", errors="replace"),
        "classification": "VALID_SEALED_RUN_FAILURE_TRANSPORT_RESET",
    }
    output = root / "analysis" / f"{args.run_id}-partial-logits-trace.json"
    write_once(output, result)
    print(json.dumps({key: result[key] for key in (
        "completed_tokens", "sealed_request_tokens", "partial_output_token_hash", "prefix_equal",
        "first_mismatch", "ttft_s", "partial_decode_tok_s", "median_tpot_s", "p95_tpot_s",
        "terminal_event_observed", "classification")}), flush=True)


if __name__ == "__main__":
    main()
