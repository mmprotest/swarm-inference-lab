"""Run the E027 full-model versus three-stage local correctness gate."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/experiment_027_state_local_wan.yaml")
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path(".runtime/experiment-027/build/bin/llama-e027-stage.exe"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiment-027/local-validation.json"),
    )
    parser.add_argument("--tokens", type=int, default=12)
    args = parser.parse_args()

    root = Path.cwd()
    config = load_config(args.config)
    model = (root / config["model_path"]).resolve()
    executable = (root / args.executable).resolve()
    prompt = prompts_from_config(config)[0]
    logs = root / ".runtime" / "experiment-027" / "local-validation"
    logs.mkdir(parents=True, exist_ok=True)

    reference: NativeStageProcess | None = None
    workers: list[NativeStageProcess] = []
    try:
        reference = NativeStageProcess.start(
            executable=executable,
            model=model,
            host="127.0.0.1",
            port=19269,
            stage_start=0,
            stage_end=64,
            log_path=logs / "reference.stdout.log",
            n_ctx=int(config["context_tokens"]),
            n_batch=int(config["batch_tokens"]),
            n_rs_seq=int(config["recurrent_rollback_snapshots"]),
        )
        with StageClient("127.0.0.1", 19269) as reference_client:
            facts = reference_client.ping()
            n_embd = int(facts["n_embd"])
            prompt_tokens = reference_client.tokenize(prompt.content).tolist()
            reference_tokens, reference_prefill = generate_reference_greedy(
                reference_client,
                prompt_tokens,
                output_tokens=args.tokens,
                n_embd=n_embd,
                top_k=int(config["top_k_validation"]),
            )
            control_tokens, control_prefill = generate_reference_greedy(
                reference_client,
                prompt_tokens,
                output_tokens=args.tokens,
                n_embd=n_embd,
                top_k=int(config["top_k_validation"]),
            )
        reference.stop()
        reference = None

        for index, (start, end) in enumerate(config["layer_ranges"]):
            workers.append(
                NativeStageProcess.start(
                    executable=executable,
                    model=model,
                    host="127.0.0.1",
                    port=19270 + index,
                    stage_start=int(start),
                    stage_end=int(end),
                    log_path=logs / f"stage-{index}.stdout.log",
                    n_ctx=int(config["context_tokens"]),
                    n_batch=int(config["batch_tokens"]),
                    n_rs_seq=int(config["recurrent_rollback_snapshots"]),
                )
            )
        clients = tuple(StageClient("127.0.0.1", 19270 + index) for index in range(3))
        pipeline = StagePipeline(clients)  # type: ignore[arg-type]
        try:
            pipeline.reset()
            distributed_prefill = pipeline.traverse(
                prompt_tokens,
                position=0,
                top_k=int(config["top_k_validation"]),
                return_nextn=True,
                return_full_logits=True,
            ).output
            lane, _ = generate_target_lane(
                pipeline,
                prompt,
                prompt_tokens,
                output_tokens=args.tokens,
                top_k=int(config["top_k_validation"]),
            )
        finally:
            pipeline.close()

        assert reference_prefill.full_logits is not None
        assert distributed_prefill.full_logits is not None
        difference = np.abs(
            reference_prefill.full_logits.astype(np.float64)
            - distributed_prefill.full_logits.astype(np.float64)
        )
        assert control_prefill.full_logits is not None
        control_difference = np.abs(
            reference_prefill.full_logits.astype(np.float64)
            - control_prefill.full_logits.astype(np.float64)
        )
        greedy_match = lane.tokens == reference_tokens
        control_greedy_match = control_tokens == reference_tokens
        top_k_match = float(
            np.mean(reference_prefill.top_ids == distributed_prefill.top_ids)
        )
        top_1_match = bool(
            np.array_equal(
                reference_prefill.top_ids[:, 0], distributed_prefill.top_ids[:, 0]
            )
        )
        row_top1 = reference_prefill.top_ids[:, 0] == distributed_prefill.top_ids[:, 0]
        topk_set_agreement = [
            len(set(reference_prefill.top_ids[row]).intersection(distributed_prefill.top_ids[row]))
            / reference_prefill.top_ids.shape[1]
            for row in range(reference_prefill.top_ids.shape[0])
        ]
        max_abs_logit_error = float(np.max(difference))
        mean_abs_logit_error = float(np.mean(difference))
        correctness_pass = bool(
            greedy_match
            and top_1_match
            and top_k_match >= 0.99
            and max_abs_logit_error <= 1e-3
        )
        result = {
            "evidence_class": "PHYSICAL_SINGLE_MACHINE_DIAGNOSTIC",
            "model": "Qwen3.8-27B Q4_K_M",
            "model_sha256": config["model_sha256"],
            "layer_ranges": config["layer_ranges"],
            "prompt_id": prompt.prompt_id,
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": args.tokens,
            "reference_token_sha256": token_hash(reference_tokens),
            "distributed_token_sha256": lane.token_sha256,
            "greedy_token_match": greedy_match,
            "control_repeat_greedy_token_match": control_greedy_match,
            "control_repeat_top1_agreement": float(np.mean(
                reference_prefill.top_ids[:, 0] == control_prefill.top_ids[:, 0]
            )),
            "control_repeat_topk_set_agreement": float(np.mean([
                len(set(reference_prefill.top_ids[row]).intersection(control_prefill.top_ids[row]))
                / reference_prefill.top_ids.shape[1]
                for row in range(reference_prefill.top_ids.shape[0])
            ])),
            "control_repeat_max_abs_logit_error": float(np.max(control_difference)),
            "control_repeat_mean_abs_logit_error": float(np.mean(control_difference)),
            "teacher_forced_top1_match": top_1_match,
            "teacher_forced_top1_agreement": float(np.mean(row_top1)),
            "teacher_forced_top1_mismatch_rows": np.flatnonzero(~row_top1).tolist(),
            "teacher_forced_topk_element_agreement": top_k_match,
            "teacher_forced_topk_set_agreement": float(np.mean(topk_set_agreement)),
            "last_prompt_row_top1_match": bool(row_top1[-1]),
            "last_prompt_row_topk_set_agreement": float(topk_set_agreement[-1]),
            "last_prompt_row_reference_top_ids": reference_prefill.top_ids[-1].tolist(),
            "last_prompt_row_distributed_top_ids": distributed_prefill.top_ids[-1].tolist(),
            "last_prompt_row_max_abs_logit_error": float(np.max(difference[-1])),
            "last_prompt_row_mean_abs_logit_error": float(np.mean(difference[-1])),
            "max_abs_logit_error": max_abs_logit_error,
            "mean_abs_logit_error": mean_abs_logit_error,
            "correctness_pass": correctness_pass,
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if correctness_pass else 2
    finally:
        if reference is not None:
            reference.stop()
        for worker in reversed(workers):
            worker.stop()


if __name__ == "__main__":
    raise SystemExit(main())
