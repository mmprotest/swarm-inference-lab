from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_025.io import atomic_write_json, utc_now
from swarm_inference.experiments.experiment_025.runpod_planning import (
    IMAGE_DIGEST,
    IMAGE_REFERENCE,
    IMAGE_SIZE_BYTES,
    IMAGE_TAG,
    RUN_ID,
)


def _run(*arguments: str) -> str:
    process = subprocess.run(
        list(arguments),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Docker image verification failed: {process.stderr[-1000:]}")
    return process.stdout.strip()


def _inspect(reference: str) -> dict[str, Any]:
    output = _run(
        "docker",
        "image",
        "inspect",
        reference,
        "--format",
        "{{json .RepoDigests}}|{{json .RepoTags}}|{{.Id}}|{{.Size}}|{{.Architecture}}|{{.Os}}",
    )
    digests, tags, image_id, size, architecture, operating_system = output.split("|", 5)
    return {
        "reference": reference,
        "repo_digests": json.loads(digests),
        "repo_tags": json.loads(tags),
        "image_id": image_id,
        "size_bytes": int(size),
        "architecture": architecture,
        "operating_system": operating_system,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the unchanged E025 image locally without running it"
    )
    parser.add_argument("--verify-pull", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"artifacts/runs/experiment-025-{RUN_ID}/preflight/runpod/"
            "runpod-image-compatibility.json"
        ),
    )
    arguments = parser.parse_args()
    pull_succeeded = False
    if arguments.verify_pull:
        pull_output = _run("docker", "pull", IMAGE_REFERENCE)
        pull_succeeded = (
            f"Digest: {IMAGE_DIGEST}" in pull_output
            and "Status: Image is up to date" in pull_output
        )
    digest = _inspect(IMAGE_REFERENCE)
    tag = _inspect(IMAGE_TAG)
    expected_repo_digest = IMAGE_REFERENCE
    gates = {
        "digest_reference_present": expected_repo_digest in digest["repo_digests"],
        "digest_image_id_exact": digest["image_id"] == IMAGE_DIGEST,
        "tag_and_digest_same_image_id": tag["image_id"] == digest["image_id"],
        "size_exact": digest["size_bytes"] == IMAGE_SIZE_BYTES,
        "linux_amd64": digest["operating_system"] == "linux" and digest["architecture"] == "amd64",
        "registry_pull_by_digest_verified": pull_succeeded if arguments.verify_pull else None,
    }
    receipt: dict[str, Any] = {
        "schema_version": "experiment-025-runpod-image-compatibility-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(value is not False for value in gates.values()) else "FAIL",
        "provider_mutations": [],
        "container_started": False,
        "image_built": False,
        "image_pushed": False,
        "worker_bytes_changed": False,
        "local_rtx_5090_recanary_required": False,
        "rest_v1_image_request": {
            "imageName": IMAGE_TAG,
            "expected_runtime_digest": IMAGE_DIGEST,
            "reason": (
                "RunPod REST v1 documents imageName as an image tag. The unique frozen "
                "tag and digest resolve to the same existing local image. P1 remains the "
                "physical provider pull/start gate."
            ),
        },
        "digest_inspect": digest,
        "tag_inspect": tag,
        "gates": gates,
    }
    atomic_write_json(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
