"""Localize E027 error at the layer-44 boundary and final stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import (
    Flags,
    StageClient,
    final_output,
    hidden_output,
)
from swarm_inference.experiments.experiment_027.runner import (
    NativeStageProcess,
    _full_stage_infer,
    load_config,
    prompts_from_config,
    write_json,
)


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    lhs = left.astype(np.float64)
    rhs = right.astype(np.float64)
    delta = np.abs(lhs - rhs)
    cosine = np.einsum("ij,ij->i", lhs, rhs) / (
        np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1)
    )
    return {
        "exact_bytes": left.tobytes() == right.tobytes(),
        "mean_abs_error": float(np.mean(delta)),
        "max_abs_error": float(np.max(delta)),
        "relative_l2": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)),
        "mean_row_cosine": float(np.mean(cosine)),
        "last_row_cosine": float(cosine[-1]),
    }


def main() -> int:
    root = Path.cwd()
    config = load_config(Path("configs/experiments/experiment_027_state_local_wan.yaml"))
    executable = (root / ".runtime/experiment-027/build/bin/llama-e027-stage.exe").resolve()
    model = (root / config["model_path"]).resolve()
    prompt = prompts_from_config(config)[0]
    logs = root / ".runtime" / "experiment-027" / "boundary44-probe"
    logs.mkdir(parents=True, exist_ok=True)

    reference = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19269,
        stage_start=0, stage_end=64, log_path=logs / "reference.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19269) as client:
            tokens = client.tokenize(prompt.content).tolist()
            client.reset()
            response = _full_stage_infer(
                client, tokens, position=0, n_embd=5120, top_k=16,
                return_full_logits=True, return_tap44=True,
            )
            reference_output = final_output(response)
            tap44 = reference_output.tap44
            assert tap44 is not None and reference_output.full_logits is not None
    finally:
        reference.stop()

    stage0 = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19270,
        stage_start=0, stage_end=22, log_path=logs / "stage-0.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    stage1 = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19271,
        stage_start=22, stage_end=44, log_path=logs / "stage-1.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19270) as first, StageClient("127.0.0.1", 19271) as second:
            first.reset()
            second.reset()
            boundary22 = hidden_output(first.infer_tokens(tokens, position=0, n_embd=5120))
            boundary44 = hidden_output(second.infer_hidden(boundary22, position=0))
    finally:
        stage1.stop()
        stage0.stop()

    stage2 = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19272,
        stage_start=44, stage_end=64, log_path=logs / "stage-2.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19272) as final:
            final.reset()
            isolated_response = final.infer_hidden(
                tap44, position=0, top_k=16, return_full_logits=True
            )
            isolated_output = final_output(isolated_response)
            assert isolated_output.full_logits is not None
    finally:
        stage2.stop()

    result = {
        "prompt_id": prompt.prompt_id,
        "tokens": len(tokens),
        "layer_44_boundary": metrics(tap44, boundary44),
        "final_stage_logits_from_reference_boundary": metrics(
            reference_output.full_logits, isolated_output.full_logits
        ),
        "final_stage_top1_agreement": float(np.mean(
            reference_output.top_ids[:, 0] == isolated_output.top_ids[:, 0]
        )),
    }
    write_json(Path("artifacts/experiment-027/boundary-44-probe.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
