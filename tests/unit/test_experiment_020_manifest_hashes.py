from __future__ import annotations

import json

from swarm_inference.experiments.experiment_020.model_distribution import (
    hydrate_worker_manifest_hashes,
)


def test_worker_manifest_hashes_are_hydrated_from_pod_bundle(tmp_path) -> None:
    root = tmp_path / "manifests"
    root.mkdir()
    for index in range(96):
        worker_id = f"pod-{index // 8:03d}.worker-{index % 8:02d}"
        (root / f"{worker_id}.json").write_text(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "checkpoint_hashes": {"source.safetensors": "UNAVAILABLE"},
                }
            ),
            encoding="utf-8",
        )
    bundles = {
        "bundles": [
            {
                "files": [
                    {
                        "source_file": "source.safetensors",
                        "sha256": "a" * 64,
                        "worker_consumers": [
                            f"pod-{index // 8:03d}.worker-{index % 8:02d}"
                            for index in range(96)
                        ],
                    }
                ]
            }
        ]
    }

    receipt = hydrate_worker_manifest_hashes(bundles, root)

    assert receipt["status"] == "PASS"
    assert receipt["updated_worker_manifests"] == 96
    assert receipt["placeholder_hashes_remaining"] == 0
    sample = json.loads((root / "pod-000.worker-00.json").read_text(encoding="utf-8"))
    assert sample["checkpoint_hashes"]["source.safetensors"] == "a" * 64
    assert sample["all_checkpoint_hashes_available"] is True
