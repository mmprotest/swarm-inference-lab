"""Measure scalar versus block target logits under identical forced history."""

from pathlib import Path
import numpy as np
from swarm_inference.experiments.experiment_027.protocol import StageClient, StagePipeline
from swarm_inference.experiments.experiment_027.runner import NativeStageProcess, load_config, prompts_from_config, generate_target_lane, write_json


def main():
    root = Path.cwd()
    config = load_config(Path('configs/experiments/experiment_027_state_local_wan.yaml'))
    exe = (root / '.runtime/experiment-027/build/bin/llama-e027-stage.exe').resolve()
    workers = []
    pipe = None
    try:
        for i, (start, end) in enumerate(config['layer_ranges']):
            workers.append(NativeStageProcess.start(executable=exe, model=(root / config['model_path']).resolve(),
                host='127.0.0.1', port=19410+i, stage_start=start, stage_end=end,
                log_path=root / f'.runtime/experiment-027/numeric-{i}.log'))
        pipe = StagePipeline(tuple(StageClient('127.0.0.1', 19410+i) for i in range(3)))
        prompt = prompts_from_config(config)[2]
        tokens = pipe.stages[0].tokenize(prompt.content).tolist()
        target, _ = generate_target_lane(pipe, prompt, tokens, output_tokens=32)
        rows = {}
        for k in (1, 2, 4, 8):
            pipe.reset()
            pipe.traverse(tokens, position=0)
            outputs = []
            for i in range(0, len(target.tokens), k):
                result = pipe.traverse(list(target.tokens[i:i+k]), position=len(tokens)+i,
                                       return_full_logits=True)
                outputs.append(result.output.full_logits)
            matrix = np.concatenate(outputs)
            if k == 1: reference = matrix
            delta = matrix.astype(np.float64)-reference
            top2 = np.argsort(matrix, axis=1)[:, -2:][:, ::-1]
            mismatches = np.flatnonzero(np.argmax(matrix, axis=1) != np.argmax(reference, axis=1))
            row = dict(max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                       relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
                       mismatches=mismatches.tolist(),
                       critical=[dict(row=i, top_ids=top2[i].tolist(),
                                      top_logits=matrix[i,top2[i]].tolist(),
                                      reference_logits=reference[i,top2[i]].tolist())
                                 for i in set([21, *mismatches.tolist()])])
            rows[k] = row
            print(k, row, flush=True)
        write_json(Path('artifacts/experiment-027/numerical-probe.json'), rows)
    finally:
        if pipe: pipe.close()
        for w in reversed(workers): w.stop()


if __name__ == '__main__': main()
