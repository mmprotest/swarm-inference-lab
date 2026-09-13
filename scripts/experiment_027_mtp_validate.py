"""Local debugging only: verify MTP acceptance and scalar-batch exactness."""

import argparse
from pathlib import Path

from swarm_inference.experiments.experiment_027.mtp import generate_mtp_lane
from swarm_inference.experiments.experiment_027.protocol import StageClient, StagePipeline
from swarm_inference.experiments.experiment_027.runner import (
    NativeStageProcess, generate_target_lane, load_config, prompts_from_config, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ubatch', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--prompt', type=int, default=2)
    parser.add_argument('--blocks', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--serial-blocks', action='store_true')
    args = parser.parse_args()
    root = Path.cwd()
    config = load_config(Path('configs/experiments/experiment_027_state_local_wan.yaml'))
    executable = (root / '.runtime/experiment-027/build/bin/llama-e027-stage.exe').resolve()
    logs = root / '.runtime/experiment-027/mtp-local'
    workers = []
    pipeline = None
    draft = None
    try:
        for i, (start, end) in enumerate(config['layer_ranges']):
            workers.append(NativeStageProcess.start(
                executable=executable, model=(root / config['model_path']).resolve(),
                host='127.0.0.1', port=19400+i, stage_start=start, stage_end=end,
                log_path=logs / f'stage-{i}.log', n_ubatch=args.ubatch, serial_blocks=args.serial_blocks,
            ))
        workers.append(NativeStageProcess.start(
            executable=executable,
            model=(root / '.runtime/experiment-026/models/mtp-Qwen3.8-27B-Q8_0.gguf').resolve(),
            host='127.0.0.1', port=19403, stage_start=0, stage_end=64,
            log_path=logs / 'draft.log', mtp=True, n_rs_seq=0,
        ))
        pipeline = StagePipeline(tuple(StageClient('127.0.0.1', 19400+i) for i in range(3)))
        draft = StageClient('127.0.0.1', 19403)
        prompt = prompts_from_config(config)[args.prompt]
        tokens = pipeline.stages[0].tokenize(prompt.content).tolist()
        target, _ = generate_target_lane(pipeline, prompt, tokens, output_tokens=args.tokens)
        print({'target_hash': target.token_sha256, 'tok_s': len(target.tokens)*1e9/target.elapsed_ns}, flush=True)
        rows = []
        for k in args.blocks:
            lane = generate_mtp_lane(pipeline, draft, prompt, tokens, output_tokens=args.tokens, block_size=k)
            row = dict(k=k, exact=lane.tokens == target.tokens, acceptance=lane.acceptance_rate,
                       committed_per_traversal=lane.committed_per_traversal,
                       tok_s=len(lane.tokens)*1e9/lane.elapsed_ns, token_hash=lane.token_sha256,
                       target_tokens=target.tokens, optimized_tokens=lane.tokens)
            rows.append(row)
            print(row, flush=True)
        write_json(Path(f'artifacts/experiment-027/mtp-local-u{args.ubatch}-p{args.prompt}-s{int(args.serial_blocks)}.json'),
                   dict(evidence_class='SINGLE_DEVICE_DEBUG', ubatch=args.ubatch,
                        serial_blocks=args.serial_blocks, rows=rows))
    finally:
        if draft: draft.close()
        if pipeline: pipeline.close()
        for worker in reversed(workers): worker.stop()


if __name__ == '__main__':
    main()
