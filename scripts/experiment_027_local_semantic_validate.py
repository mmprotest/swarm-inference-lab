"""Canonical local semantic gate for E027's full versus staged target path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import StageClient, StagePipeline
from swarm_inference.experiments.experiment_027.runner import (
    NativeStageProcess,
    generate_reference_greedy,
    generate_target_lane,
    load_config,
    prompts_from_config,
    token_hash,
    write_json,
)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/experiments/experiment_027_state_local_wan.yaml"),
    )
    parser.add_argument(
        "--executable", type=Path,
        default=Path(".runtime/experiment-027/build/bin/llama-e027-stage.exe"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/experiment-027/local-semantic-validation.json"),
    )
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    prompts = prompts_from_config(config)
    model = (root / config["model_path"]).resolve()
    executable = (root / args.executable).resolve()
    logs = root / ".runtime" / "experiment-027" / "local-semantic-validation"
    logs.mkdir(parents=True, exist_ok=True)
    top_k = int(config["top_k_validation"])

    reference_process: NativeStageProcess | None = None
    workers: list[NativeStageProcess] = []
    references: dict[str, tuple[list[int], tuple[int, ...], object]] = {}
    try:
        reference_process = NativeStageProcess.start(
            executable=executable, model=model, host="127.0.0.1", port=19269,
            stage_start=0, stage_end=64, log_path=logs / "reference.log",
            n_ctx=int(config["context_tokens"]), n_batch=int(config["batch_tokens"]),
            n_rs_seq=int(config["recurrent_rollback_snapshots"]),
            serial_blocks=bool(config.get("serial_blocks", False)),
        )
        with StageClient("127.0.0.1", 19269) as client:
            n_embd = int(client.ping()["n_embd"])
            for prompt in prompts:
                prompt_tokens = client.tokenize(prompt.content).tolist()
                generated, prefill = generate_reference_greedy(
                    client, prompt_tokens, output_tokens=args.tokens,
                    n_embd=n_embd, top_k=top_k,
                )
                references[prompt.prompt_id] = (prompt_tokens, generated, prefill)
        reference_process.stop()
        reference_process = None

        for index, (start, end) in enumerate(config["layer_ranges"]):
            workers.append(NativeStageProcess.start(
                executable=executable, model=model, host="127.0.0.1", port=19270 + index,
                stage_start=start, stage_end=end, log_path=logs / f"stage-{index}.log",
                n_ctx=int(config["context_tokens"]), n_batch=int(config["batch_tokens"]),
                n_rs_seq=int(config["recurrent_rollback_snapshots"]),
                serial_blocks=bool(config.get("serial_blocks", False)),
            ))
        pipeline = StagePipeline(tuple(
            StageClient("127.0.0.1", 19270 + index) for index in range(3)
        ))
        prompt_results = []
        try:
            for prompt in prompts:
                prompt_tokens, reference_tokens, reference_prefill = references[prompt.prompt_id]
                pipeline.reset()
                distributed_prefill = pipeline.traverse(
                    prompt_tokens, position=0, top_k=top_k,
                    return_nextn=True, return_full_logits=True,
                ).output
                lane, _ = generate_target_lane(
                    pipeline, prompt, prompt_tokens, output_tokens=args.tokens, top_k=top_k,
                )
                assert reference_prefill.full_logits is not None
                assert distributed_prefill.full_logits is not None
                left = reference_prefill.full_logits.astype(np.float64)
                right = distributed_prefill.full_logits.astype(np.float64)
                delta = np.abs(left - right)
                row_top1 = reference_prefill.top_ids[:, 0] == distributed_prefill.top_ids[:, 0]
                topk_sets = [
                    len(set(reference_prefill.top_ids[row]).intersection(distributed_prefill.top_ids[row])) / top_k
                    for row in range(len(prompt_tokens))
                ]
                row_cosine = cosine_rows(left, right)
                relative_l2 = np.linalg.norm(left - right) / np.maximum(
                    np.linalg.norm(left), np.finfo(np.float64).tiny
                )
                semantic_pass = bool(
                    lane.tokens == reference_tokens and row_top1[-1]
                )
                prompt_results.append({
                    "prompt_id": prompt.prompt_id,
                    "prompt_tokens": len(prompt_tokens),
                    "generated_tokens": args.tokens,
                    "reference_token_sha256": token_hash(reference_tokens),
                    "distributed_token_sha256": lane.token_sha256,
                    "greedy_token_match": lane.tokens == reference_tokens,
                    "next_token_top1_match": bool(row_top1[-1]),
                    "teacher_forced_top1_agreement": float(np.mean(row_top1)),
                    "teacher_forced_top1_mismatch_rows": np.flatnonzero(~row_top1).tolist(),
                    "teacher_forced_topk_set_agreement": float(np.mean(topk_sets)),
                    "next_token_topk_set_agreement": float(topk_sets[-1]),
                    "prompt_logits_mean_cosine": float(np.mean(row_cosine)),
                    "next_token_logits_cosine": float(row_cosine[-1]),
                    "prompt_logits_relative_l2": float(relative_l2),
                    "mean_abs_logit_error": float(np.mean(delta)),
                    "max_abs_logit_error": float(np.max(delta)),
                    "semantic_pass": semantic_pass,
                })
        finally:
            pipeline.close()

        result = {
            "evidence_class": "PHYSICAL_SINGLE_MACHINE_DIAGNOSTIC",
            "model": "Qwen3.8-27B Q4_K_M",
            "model_sha256": config["model_sha256"],
            "layer_ranges": config["layer_ranges"],
            "validation_definition": (
                "exact 32-token greedy hashes and consumed next-token top-1; "
                "full teacher-forced logits retained as numerical diagnostics"
            ),
            "prompt_results": prompt_results,
            "correctness_pass": all(row["semantic_pass"] for row in prompt_results),
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["correctness_pass"] else 2
    finally:
        if reference_process is not None:
            reference_process.stop()
        for worker in reversed(workers):
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
