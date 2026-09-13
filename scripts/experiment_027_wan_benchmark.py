"""Run the sealed E027 target-only and exact n-gram WAN benchmark arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

from swarm_inference.experiments.experiment_027.protocol import StageClient, StagePipeline
from swarm_inference.experiments.experiment_027.mtp import generate_mtp_lane
from swarm_inference.experiments.experiment_027.runner import (
    NativeStageProcess,
    Prompt,
    SpeculativeLane,
    TargetLane,
    generate_ngram_speculative_lane,
    generate_target_lane,
    load_config,
    percentile,
    prompts_from_config,
    write_json,
)


def timing_summary(lanes: Sequence[TargetLane | SpeculativeLane]) -> dict[str, Any]:
    traversals = [traversal for lane in lanes for traversal in lane.traversals]
    elapsed_ns = sum(lane.elapsed_ns for lane in lanes)
    committed = sum(len(lane.tokens) for lane in lanes)
    compute_ns = sum(exchange.compute_ns for item in traversals for exchange in item.exchanges)
    serialization_ns = sum(
        exchange.server_serialize_ns + exchange.client_serialize_ns
        for item in traversals for exchange in item.exchanges
    )
    wan_wait_ns = sum(
        max(0, exchange.round_trip_ns - exchange.compute_ns
            - exchange.server_serialize_ns)
        for item in traversals for exchange in item.exchanges[1:]
    )
    pipeline_gap_ns = sum(
        max(0, item.elapsed_ns - sum(exchange.round_trip_ns + exchange.client_serialize_ns
                                   for exchange in item.exchanges))
        for item in traversals
    )
    local_other_ns = sum(
        max(0, item.exchanges[0].round_trip_ns - item.exchanges[0].compute_ns
            - item.exchanges[0].server_serialize_ns)
        for item in traversals
    )
    draft_other_ns = elapsed_ns - sum(item.elapsed_ns for item in traversals)
    accounted_ns = compute_ns + serialization_ns + wan_wait_ns + pipeline_gap_ns + local_other_ns + draft_other_ns
    bytes_wan = sum(
        exchange.request_bytes + exchange.response_bytes
        for item in traversals for exchange in item.exchanges[1:]
    )
    lane_tok_s = [len(lane.tokens) / (lane.elapsed_ns / 1e9) for lane in lanes]
    tpot_ms = []
    for lane in lanes:
        if isinstance(lane, SpeculativeLane) and lane.commit_counts:
            for ns, n in zip(lane.block_elapsed_ns, lane.commit_counts):
                tpot_ms.extend([ns / n / 1e6] * n)
        else:
            tpot_ms.extend(item.elapsed_ns / 1e6 for item in lane.traversals)
    return {
        "committed_tokens": committed,
        "target_traversals": len(traversals),
        "committed_tokens_per_target_traversal": committed / len(traversals),
        "lane_tok_s": lane_tok_s,
        "median_committed_tok_s": statistics.median(lane_tok_s),
        "median_tpot_ms": statistics.median(tpot_ms),
        "p95_tpot_ms": percentile(tpot_ms, 0.95),
        "tpot_definition": "block wall time divided by committed tokens, token-weighted",
        "time_decomposition": {
            "wall_ms": elapsed_ns / 1e6,
            "compute_ms": compute_ns / 1e6,
            "compute_fraction": compute_ns / elapsed_ns,
            "wan_wait_ms": wan_wait_ns / 1e6,
            "wan_wait_fraction": wan_wait_ns / elapsed_ns,
            "serialization_ms": serialization_ns / 1e6,
            "serialization_fraction": serialization_ns / elapsed_ns,
            "pipeline_gap_ms": pipeline_gap_ns / 1e6,
            "pipeline_gap_fraction": pipeline_gap_ns / elapsed_ns,
            "other_local_ms": local_other_ns / 1e6,
            "other_local_fraction": local_other_ns / elapsed_ns,
            "draft_and_coordinator_ms": draft_other_ns / 1e6,
            "draft_and_coordinator_fraction": draft_other_ns / elapsed_ns,
            "accounting_residual_ms": (elapsed_ns - accounted_ns) / 1e6,
        },
        "stage_compute_median_ms": [
            statistics.median([item.exchanges[index].compute_ns / 1e6 for item in traversals])
            for index in range(3)
        ],
        "stage_round_trip_median_ms": [
            statistics.median([item.exchanges[index].round_trip_ns / 1e6 for item in traversals])
            for index in range(3)
        ],
        "wan_messages_per_committed_token": 4 * len(traversals) / committed,
        "wan_bytes_per_committed_token": bytes_wan / committed,
    }


def raw_lane(lane: TargetLane | SpeculativeLane) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_id": lane.prompt_id,
        "tokens": list(lane.tokens),
        "token_sha256": lane.token_sha256,
        "elapsed_ns": lane.elapsed_ns,
        "traversals": [],
    }
    if isinstance(lane, SpeculativeLane):
        result.update({
            "block_size": lane.block_size,
            "proposed_draft_tokens": lane.proposed_draft_tokens,
            "accepted_draft_tokens": lane.accepted_draft_tokens,
            "commit_counts": lane.commit_counts,
            "block_elapsed_ns": lane.block_elapsed_ns,
        })
    for traversal in lane.traversals:
        result["traversals"].append({
            "position": traversal.position,
            "n_tokens": traversal.n_tokens,
            "elapsed_ns": traversal.elapsed_ns,
            "exchanges": [asdict(exchange) for exchange in traversal.exchanges],
        })
    return result


def ping_samples(clients: Sequence[StageClient], count: int = 7) -> list[dict[str, Any]]:
    rows = []
    for client in clients:
        values = []
        fact = None
        for _ in range(count):
            begin = time.perf_counter_ns()
            fact = client.ping()
            values.append((time.perf_counter_ns() - begin) / 1e6)
        assert fact is not None
        rows.append({"endpoint": client.endpoint, "fact": fact, "rtt_ms": values,
                     "median_rtt_ms": statistics.median(values)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/experiment-027/wan-benchmark.json"))
    parser.add_argument("--draft", choices=("ngram", "mtp"), default="ngram")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--blocks", type=int, nargs="*", default=[2, 4, 8])
    args = parser.parse_args()
    root = Path.cwd()
    config = load_config(Path("configs/experiments/experiment_027_state_local_wan.yaml"))
    if args.tokens is not None: config["output_tokens_per_lane"] = args.tokens
    prompts = prompts_from_config(config)
    executable = (root / ".runtime/experiment-027/build/bin/llama-e027-stage.exe").resolve()
    model = (root / config["model_path"]).resolve()
    logs = root / ".runtime/experiment-027/wan"
    logs.mkdir(parents=True, exist_ok=True)
    local = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19300,
        stage_start=0, stage_end=22, log_path=logs / "stage-a.log",
        n_ctx=int(config["context_tokens"]), n_batch=int(config["batch_tokens"]),
        n_rs_seq=int(config["recurrent_rollback_snapshots"]),
        serial_blocks=bool(config.get("serial_blocks", False)),
    )
    clients = (StageClient("127.0.0.1", 19300), StageClient("127.0.0.1", 19301),
               StageClient("127.0.0.1", 19302))
    pipeline = None
    draft_process = None
    draft_client = None
    try:
        pipeline = StagePipeline(clients)
        if args.draft == "mtp":
            draft_process = NativeStageProcess.start(
                executable=executable,
                model=(root / ".runtime/experiment-026/models/mtp-Qwen3.8-27B-Q8_0.gguf").resolve(),
                host="127.0.0.1", port=19303, stage_start=0, stage_end=64,
                log_path=logs / "mtp.log", mtp=True, n_rs_seq=0,
            )
            draft_client = StageClient("127.0.0.1", 19303)
        endpoints = ping_samples(clients)
        tokenized = {prompt.prompt_id: clients[0].tokenize(prompt.content).tolist()
                     for prompt in prompts}
        generate_target_lane(
            pipeline, prompts[0], tokenized[prompts[0].prompt_id],
            output_tokens=int(config["warmup_tokens"]), top_k=int(config["top_k_validation"]),
        )
        targets = [
            generate_target_lane(
                pipeline, prompt, tokenized[prompt.prompt_id],
                output_tokens=int(config["output_tokens_per_lane"]),
                top_k=int(config["top_k_validation"]),
            )[0]
            for prompt in prompts
        ]
        target_hashes = {lane.prompt_id: lane.token_sha256 for lane in targets}
        arms: dict[str, Any] = {
            "1": {"candidate_source": None, "exact_match": True,
                  "summary": timing_summary(targets),
                  "lanes": [raw_lane(lane) for lane in targets]},
        }
        print(json.dumps({"arm": 1, "summary": arms["1"]["summary"]}), flush=True)
        def speculative(*positional, **keyword):
            if draft_client is not None:
                return generate_mtp_lane(positional[0], draft_client, *positional[1:], **keyword)
            return generate_ngram_speculative_lane(*positional, **keyword)
        for block_size in args.blocks:
            speculative(
                pipeline, prompts[0], tokenized[prompts[0].prompt_id],
                output_tokens=max(block_size, int(config["warmup_tokens"])),
                block_size=block_size, top_k=int(config["top_k_validation"]),
            )
            lanes = [
                speculative(
                    pipeline, prompt, tokenized[prompt.prompt_id],
                    output_tokens=int(config["output_tokens_per_lane"]),
                    block_size=block_size, top_k=int(config["top_k_validation"]),
                )
                for prompt in prompts
            ]
            exact = all(lane.token_sha256 == target_hashes[lane.prompt_id] for lane in lanes)
            proposed = sum(lane.proposed_draft_tokens for lane in lanes)
            accepted = sum(lane.accepted_draft_tokens for lane in lanes)
            arms[str(block_size)] = {
                "candidate_source": args.draft,
                "exact_match": exact,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "acceptance_rate": accepted / proposed if proposed else 0.0,
                "summary": timing_summary(lanes),
                "lanes": [raw_lane(lane) for lane in lanes],
            }
            print(json.dumps({"arm": block_size, "exact": exact,
                              "summary": arms[str(block_size)]["summary"],
                              "acceptance": accepted / proposed if proposed else 0}), flush=True)
            write_json(args.output.with_suffix(".checkpoint.json"), {"endpoints": endpoints, "arms": arms})
        result = {
            "experiment_id": config["experiment_id"],
            "evidence_class": "PHYSICAL",
            "model": "Qwen3.8-27B GGUF Q4_K_M",
            "model_sha256": config["model_sha256"],
            "physical_machines": 3,
            "network": "real WAN via persistent SSH TCP forwarding",
            "layer_ranges": config["layer_ranges"],
            "protocol": "one request and one response per physical stage per target traversal",
            "endpoints": endpoints,
            "arms": arms,
            "all_speculative_arms_exact": all(arms[str(k)]["exact_match"] for k in args.blocks),
        }
        write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "arms": {
            key: {k: value["summary"][k] for k in (
                "median_committed_tok_s", "median_tpot_ms",
                "committed_tokens_per_target_traversal")}
            | {"exact_match": value["exact_match"],
               "acceptance_rate": value.get("acceptance_rate")}
            for key, value in arms.items()}}, indent=2))
        return 0 if result["all_speculative_arms_exact"] else 2
    finally:
        if draft_client is not None: draft_client.close()
        if draft_process is not None: draft_process.stop()
        if pipeline is not None:
            pipeline.close()
        local.stop()


if __name__ == "__main__":
    raise SystemExit(main())
