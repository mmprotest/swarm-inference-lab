"""Compare the E027 stage-0 boundary against a tapped monolithic graph."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import (
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


def main() -> int:
    root = Path.cwd()
    config = load_config(Path("configs/experiments/experiment_027_state_local_wan.yaml"))
    executable = (root / ".runtime/experiment-027/build/bin/llama-e027-stage.exe").resolve()
    model = (root / config["model_path"]).resolve()
    prompt = prompts_from_config(config)[0]
    logs = root / ".runtime" / "experiment-027" / "boundary-probe"
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
                client, tokens, position=0, n_embd=5120, top_k=1,
                return_tap22=True,
            )
            tap = final_output(response).tap22
            assert tap is not None
    finally:
        reference.stop()

    stage = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19270,
        stage_start=0, stage_end=22, log_path=logs / "stage-0.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19270) as client:
            client.reset()
            boundary = hidden_output(client.infer_tokens(tokens, position=0, n_embd=5120))
    finally:
        stage.stop()

    left = tap.astype(np.float64)
    right = boundary.astype(np.float64)
    delta = np.abs(left - right)
    cosine = np.einsum("ij,ij->i", left, right) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    )
    result = {
        "prompt_id": prompt.prompt_id,
        "tokens": len(tokens),
        "layer_boundary": 22,
        "exact_bytes": tap.tobytes() == boundary.tobytes(),
        "mean_abs_error": float(np.mean(delta)),
        "max_abs_error": float(np.max(delta)),
        "relative_l2": float(np.linalg.norm(left - right) / np.linalg.norm(left)),
        "mean_row_cosine": float(np.mean(cosine)),
        "last_row_cosine": float(cosine[-1]),
    }
    write_json(Path("artifacts/experiment-027/boundary-22-probe.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
