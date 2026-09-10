"""Teacher-forced serial/block target comparison at native-MTP divergences."""
from pathlib import Path
import json
import numpy as np
from .state_check import Probe
from .io import write_once, file_digest


def main():
    root = Path("artifacts/experiment-026")
    folder = root / "diagnostics/target-batch-numerics-001"
    folder.mkdir(parents=True, exist_ok=False)
    config = {"model": str(Path(".runtime/experiment-026/models/Qwen3.8-27B-Q4_K_M.gguf").resolve()),
              "n_ctx": 12288, "recurrent_snapshots": 3, "embeddings_nextn": True}
    write_once(folder / "config.json", config)
    probe = Probe(Path(".runtime/experiment-026/build/bin/e026-probe.exe"), folder / "config.json", folder)
    rows = []
    try:
        for prompt in sorted((root / "runs/local-target-001").iterdir()):
            if not prompt.is_dir():
                continue
            tokens = json.loads((prompt / "prompt.json").read_text())["tokens"]
            reference = json.loads((prompt / "output.json").read_text())["tokens"]
            mtp = json.loads((root / "runs/local-mtp-k3-001" / prompt.name / "output.json").read_text())["tokens"]
            mismatch = next((i for i, (a, b) in enumerate(zip(reference, mtp)) if a != b), None)
            if mismatch is None:
                continue
            probe.call({"op": "clear"})
            probe.call({"op": "decode", "tokens": tokens[:-4]})
            probe.call({"op": "decode", "tokens": tokens[-4:]})
            for token in reference[:mismatch - 3]:
                probe.call({"op": "decode", "tokens": [token]})
            state = Path(".runtime/experiment-026") / (prompt.name + "-batch-diagnostic.state")
            checkpoint = probe.call({"op": "save", "path": str(state.resolve())})
            tail = reference[mismatch - 3:mismatch]
            for token in tail[:-1]:
                probe.call({"op": "decode", "tokens": [token]})
            serial = folder / (prompt.name + "-serial.f32")
            block = folder / (prompt.name + "-block.f32")
            a = probe.call({"op": "decode", "tokens": tail[-1:], "logits_file": str(serial.resolve())})
            probe.call({"op": "restore", "path": str(state.resolve()), "position": checkpoint["position"]})
            b = probe.call({"op": "decode", "tokens": tail, "all_logits": True, "logits_file": str(block.resolve())})
            x, y = np.fromfile(serial, dtype=np.float32).astype(np.float64), np.fromfile(block, dtype=np.float32).astype(np.float64)
            def nll(values, token):
                maximum = values.max()
                return float(maximum + np.log(np.exp(values - maximum).sum()) - values[token])
            row = {"prompt_id": prompt.name, "first_mtp_mismatch": mismatch,
                   "serial_token": a["greedy_token"], "block_token": b["greedy_token"],
                   "reference_token": reference[mismatch], "mtp_token": mtp[mismatch],
                   "serial_top_logits": a["top_logits"], "block_top_logits": b["top_logits"],
                   "max_abs_logit_error": float(np.max(np.abs(x-y))), "rms_logit_error": float(np.sqrt(np.mean((x-y)**2))),
                   "reference_token_nll_delta": nll(y, reference[mismatch]) - nll(x, reference[mismatch]),
                   "serial_logits_sha256": file_digest(serial), "block_logits_sha256": file_digest(block)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        probe.close()
    write_once(folder / "comparison.json", {"evidence_class": "PHYSICAL", "scope": "LOCAL_TEACHER_FORCED_BATCH_NUMERICS", "rows": rows})


if __name__ == "__main__":
    main()
