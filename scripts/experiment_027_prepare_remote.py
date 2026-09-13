"""Create the reproducible E027 llama.cpp source bundle for paid workers."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


PINNED_COMMIT = "f1b6fbf35cfa010b0a8d6301fdfccbb7f41bd903"


def main() -> int:
    root = Path.cwd()
    llama = root / ".runtime" / "e027-llama.cpp"
    commit = subprocess.check_output(
        ["git", "-C", str(llama), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"unexpected llama.cpp commit {commit}")
    source = subprocess.check_output(
        ["git", "-C", str(llama), "archive", "--format=tar", "HEAD"]
    )
    patch = subprocess.check_output(
        [
            "git", "-C", str(llama), "diff", "--binary", "HEAD", "--",
            "examples/CMakeLists.txt", "src/llama-model.cpp", "src/models/qwen35.cpp",
        ]
    )
    output = root / ".runtime" / "experiment-027" / "remote-source.tar.gz"
    (root / "native/experiment_027/qwen35-stage-range.patch").write_bytes(patch)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(source)) as archive_in, tarfile.open(
        output, "w:gz"
    ) as archive_out:
        for member in archive_in:
            archive_out.addfile(
                member, archive_in.extractfile(member) if member.isfile() else None
            )
        additions = {
            "e027-stage-range.patch": patch,
            "examples/e027-stage/e027-stage.cpp": (
                root / "native" / "experiment_027" / "e027_stage_server.cpp"
            ).read_bytes(),
            "examples/e027-stage/CMakeLists.txt": (
                root / "native" / "experiment_027" / "CMakeLists.txt"
            ).read_bytes(),
        }
        for name, payload in additions.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive_out.addfile(info, io.BytesIO(payload))
    digest = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    receipt = {
        "experiment_id": "E027_STATE_LOCAL_WAN_STAGE_PROTOCOL",
        "llama_commit": commit,
        "stage_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "bundle_sha256": digest,
        "bundle_bytes": output.stat().st_size,
        "bundle_path": str(output),
    }
    receipt_path = root / "artifacts" / "experiment-027" / "remote-bundle.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
