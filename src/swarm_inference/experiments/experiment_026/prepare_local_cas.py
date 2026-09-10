"""Populate only verified development cache entries from the exact local GGUF."""
import hashlib
import json
from pathlib import Path


def main():
    model=Path(".runtime/experiment-026/models/Qwen3.8-27B-Q4_K_M.gguf")
    for device,role in enumerate(("a","b")):
        manifest=json.loads(Path(f"artifacts/experiment-026/acquisition/remote-stage-{role}-all-002.json").read_text())
        cache=Path(f".runtime/experiment-026/logical-rpc-cache-{device}/rpc")
        cache.mkdir(parents=True,exist_ok=True)
        written=0
        with model.open("rb") as source:
            for tensor in manifest["tensors"]:
                path=cache/tensor["rpc_fnv1a"]
                if path.exists():
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream,"sha256").hexdigest()!=tensor["sha256"]:
                            raise ValueError("Existing cache is not the expected tensor; preserving it for inspection")
                    continue
                source.seek(tensor["offset"])
                data=source.read(tensor["size_bytes"])
                if hashlib.sha256(data).hexdigest()!=tensor["sha256"]:raise ValueError("Source range mismatch")
                with path.open("xb") as output:output.write(data)
                written+=len(data)
        print({"role":role,"new_cache_bytes":written},flush=True)


if __name__=="__main__":main()
