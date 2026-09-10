"""Read-only account preflight and pinned local asset acquisition; never rents."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

from . import EXPERIMENT_ID
from .io import append_event, file_digest, utc_now, write_once

MODEL_REPO = "ggml-org/Qwen3.8-27B-GGUF"
MODEL_REVISION = "0669b98607d47046c7c2b3f801011d54a08cfccf"
MODEL_FILE = "Qwen3.8-27B-Q4_K_M.gguf"
RELEASE = "b10886"


def public_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def command(args: list[str]) -> dict:
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def inspect(root: Path) -> None:
    from ..experiment_025.vast_lifecycle import VastClient

    client = VastClient()
    budget = client.user_budget()
    rows = client.show_instances()
    account = {"timestamp": utc_now(), "experiment_id": EXPERIMENT_ID,
               "budget": budget, "hard_ceiling_usd": 38.0, "reserve_usd": 7.0,
               "instances": [{key: row.get(key) for key in
                              ("id", "label", "actual_status", "dph_total")} for row in rows]}
    write_once(root / "preflight" / "account.json", account)
    write_once(root / "cost" / "opening.json", {
        **account, "estimated_e026_spend_usd": 0.0, "created_instance_ids": [],
        "status": "NO_RENTALS", "currency": "USD"})
    probes = [command(["git", "rev-parse", "HEAD"]),
              command(["git", "status", "--porcelain=v1"]),
              command(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,driver_version,compute_cap", "--format=csv"]),
              command(["nvcc", "--version"]),
              command(["git", "-C", ".runtime/e026-llama.cpp", "rev-parse", "HEAD"])]
    write_once(root / "preflight" / "local.json", {"timestamp": utc_now(), "probes": probes})
    model = public_json(f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true")
    write_once(root / "preflight" / "model-source.json", model)
    release = public_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{RELEASE}")
    write_once(root / "preflight" / "llama-release.json", release)
    print(json.dumps({"status": "PREFLIGHT_RECORDED", "budget": budget,
                      "instances": len(rows), "model_revision": model["sha"], "release": RELEASE}), flush=True)


def acquire_model(root: Path, runtime: Path, filename: str) -> None:
    from huggingface_hub import hf_hub_download

    model = json.loads((root / "preflight/model-source.json").read_text())
    item = next(row for row in model["siblings"] if row["rfilename"] == filename)
    start = time.monotonic()
    path = Path(hf_hub_download(repo_id=MODEL_REPO, filename=filename, revision=MODEL_REVISION,
                               local_dir=runtime / "models", token=False))
    download_seconds = time.monotonic() - start
    hash_start = time.monotonic()
    actual = file_digest(path)
    expected = item["lfs"]["sha256"]
    receipt = {"experiment_id": EXPERIMENT_ID, "timestamp": utc_now(),
               "model_source": f"https://huggingface.co/{MODEL_REPO}",
               "revision": MODEL_REVISION, "filename": filename, "path": str(path.resolve()),
               "size_bytes": path.stat().st_size, "sha256": actual, "expected_sha256": expected,
               "download_seconds": download_seconds, "hash_seconds": time.monotonic() - hash_start,
               "status": "PASS" if actual == expected and path.stat().st_size == item["size"] else "FAIL"}
    write_once(root / "acquisition" / f"{filename}.json", receipt)
    print(json.dumps(receipt), flush=True)
    if receipt["status"] != "PASS":
        raise RuntimeError("Exact model integrity check failed")


def acquire_binary(root: Path, runtime: Path) -> None:
    release = json.loads((root / "preflight/llama-release.json").read_text())
    dest = runtime / RELEASE
    dest.mkdir(parents=True, exist_ok=True)
    for name in (f"llama-{RELEASE}-bin-win-cuda-13.3-x64.zip", "cudart-llama-bin-win-cuda-13.3-x64.zip"):
        item = next(row for row in release["assets"] if row["name"] == name)
        archive = runtime / name
        expected = item["digest"].removeprefix("sha256:")
        start = time.monotonic()
        if not archive.exists() or file_digest(archive) != expected:
            urllib.request.urlretrieve(item["browser_download_url"], archive)
        if file_digest(archive) != expected:
            raise RuntimeError(f"Binary archive integrity failure: {name}")
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                if not (dest / info.filename).resolve().is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe archive member")
            stream.extractall(dest)
        append_event(root / "acquisition/binaries.jsonl", {
            "timestamp": utc_now(), "name": name, "sha256": expected,
            "url": item["browser_download_url"], "size_bytes": archive.stat().st_size,
            "seconds": time.monotonic() - start})
    print(json.dumps({"status": "BINARY_READY", "path": str(dest.resolve())}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "model", "binary"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiment-026"))
    parser.add_argument("--runtime", type=Path, default=Path(".runtime/experiment-026"))
    parser.add_argument("--filename", default=MODEL_FILE)
    args = parser.parse_args()
    if args.action == "inspect":
        inspect(args.root)
    elif args.action == "model":
        acquire_model(args.root, args.runtime, args.filename)
    else:
        acquire_binary(args.root, args.runtime)


if __name__ == "__main__":
    main()
