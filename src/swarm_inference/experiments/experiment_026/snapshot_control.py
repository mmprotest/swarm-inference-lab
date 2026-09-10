"""Isolate target recurrent-snapshot mode from speculative drafting."""

from __future__ import annotations

import json
from pathlib import Path

from .io import digest, write_once
from .state_check import Probe


def main():
    root = Path("artifacts/experiment-026")
    folder = root / "diagnostics/target-snapshots-3-matched-prefill-003"
    folder.mkdir(parents=True, exist_ok=False)
    config = {"model": str(Path(".runtime/experiment-026/models/Qwen3.8-27B-Q4_K_M.gguf").resolve()),
              "n_ctx": 12288, "recurrent_snapshots": 3, "embeddings_nextn": True}
    from .io import file_digest
    config["probe_sha256"] = file_digest(Path(".runtime/experiment-026/build/bin/e026-probe.exe"))
    write_once(folder / "config.json", config)
    proc = Probe(Path(".runtime/experiment-026/build/bin/e026-probe.exe"), folder / "config.json", folder)
    rows = []
    try:
        for prompt_dir in sorted((root / "runs/local-target-001").iterdir()):
            if not prompt_dir.is_dir():
                continue
            tokens = json.loads((prompt_dir / "prompt.json").read_text())["tokens"]
            reference = json.loads((prompt_dir / "output.json").read_text())["tokens"]
            mtp = json.loads((root / "runs/local-mtp-k3-001" / prompt_dir.name / "output.json").read_text())["tokens"]
            proc.call({"op": "clear"})
            proc.call({"op": "decode", "tokens": tokens[:-4]})
            out = proc.call({"op": "decode", "tokens": tokens[-4:]})
            generated = []
            for _ in range(256):
                generated.append(out["greedy_token"])
                out = proc.call({"op": "decode", "tokens": [out["greedy_token"]]})
            row = {"prompt_id": prompt_dir.name, "tokens": generated, "sha256": digest(generated),
                   "matches_mtp": generated == mtp, "matches_default_reference": generated == reference}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
    finally:
        proc.close()
    write_once(folder / "comparison.json", {"evidence_class": "PHYSICAL", "scope": "LOCAL_TARGET_KERNEL_MODE_ABLATION", "rows": rows})


if __name__ == "__main__":
    main()
