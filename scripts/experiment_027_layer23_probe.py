"""Temporary focused probe for the first layer executed by E027 stage B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import StageClient, final_output, hidden_output
from swarm_inference.experiments.experiment_027.runner import (
    NativeStageProcess,
    _full_stage_infer,
    load_config,
    prompts_from_config,
)


def main() -> int:
    root = Path.cwd()
    config = load_config(Path("configs/experiments/experiment_027_state_local_wan.yaml"))
    executable = (root / ".runtime/experiment-027/build/bin/llama-e027-stage.exe").resolve()
    model = (root / config["model_path"]).resolve()
    logs = root / ".runtime/experiment-027/layer23-probe"
    logs.mkdir(parents=True, exist_ok=True)
    prompt = prompts_from_config(config)[0]

    reference = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19269,
        stage_start=0, stage_end=64, log_path=logs / "reference.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19269) as client:
            tokens = client.tokenize(prompt.content).tolist()
            client.reset()
            reference_output = final_output(_full_stage_infer(
                client, tokens, position=0, n_embd=5120, top_k=1,
                return_tap22=True, return_tap44=True,
            ))
            at23 = reference_output.tap22
            tapped = reference_output.tap44
            assert at23 is not None and tapped is not None
    finally:
        reference.stop()

    second = NativeStageProcess.start(
        executable=executable, model=model, host="127.0.0.1", port=19271,
        stage_start=23, stage_end=24, log_path=logs / "stage-23-only.log",
        n_ctx=512, n_batch=64, n_rs_seq=1,
    )
    try:
        with StageClient("127.0.0.1", 19271) as b:
            b.reset()
            staged = hidden_output(b.infer_hidden(at23, position=0))
    finally:
        second.stop()
    lhs = tapped.astype(np.float64)
    rhs = staged.astype(np.float64)
    delta = np.abs(lhs - rhs)
    result = {
        "tokens": len(tokens),
        "exact_bytes": tapped.tobytes() == staged.tobytes(),
        "mean_abs_error": float(delta.mean()),
        "max_abs_error": float(delta.max()),
        "relative_l2": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
