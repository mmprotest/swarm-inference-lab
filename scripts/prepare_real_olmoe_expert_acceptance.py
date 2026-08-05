"""Prepare canonical whole-expert and native-microshard OLMoE manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from swarm_inference.acceptance.productization import REAL_MODEL_ID, REAL_MODEL_REVISION
from swarm_inference.execution.expert import (
    ExpertWeights,
    safetensors_expert_loader,
    safetensors_expert_ownership_entry,
    slice_expert_weights,
)
from swarm_inference.model.olmoe import inspect_olmoe_partition_metadata
from swarm_inference.model.partition import ModelPartitionMetadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--whole", action="store_true")
    parser.add_argument("--microshards", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def _write_json(path: Path, value: object, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_microshard(path: Path, weights: ExpertWeights, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    np.savez(
        path,
        up=weights.up,
        gate=weights.gate,
        down=weights.down,
        hidden_start=np.asarray(weights.hidden_offset, dtype=np.int64),
        logical_intermediate_dimension=np.asarray(weights.logical_width, dtype=np.int64),
        native_format=np.asarray(weights.native_format),
    )


def _base_manifest(metadata: ModelPartitionMetadata) -> dict[str, object]:
    return {
        "model_id": REAL_MODEL_ID,
        "model_revision": REAL_MODEL_REVISION,
        "model_fingerprint": metadata.model_fingerprint,
        "quantization_fingerprint": metadata.quantization_fingerprint,
        "measured_service_rates": {},
    }


def _prepare_whole(
    *,
    model_path: Path,
    output: Path,
    metadata: ModelPartitionMetadata,
    layer_count: int,
    expert_count: int,
    force: bool,
) -> Path:
    entries = []
    for layer_id in range(layer_count):
        for expert_id in range(expert_count):
            entries.append(
                safetensors_expert_ownership_entry(
                    model_path,
                    layer_id=layer_id,
                    expert_id=expert_id,
                )
            )
            print(f"whole_inventory layer={layer_id} expert={expert_id}", flush=True)
    manifest_path = output / "whole-expert-manifest.json"
    _write_json(
        manifest_path,
        {
            **_base_manifest(metadata),
            "loader_type": "safetensors",
            "model_path": str(model_path),
            "owned_experts": entries,
            "measured_service_rates": {"whole_expert_calls_per_second": 1.0},
        },
        force=force,
    )
    return manifest_path


def _prepare_microshards(
    *,
    model_path: Path,
    output: Path,
    metadata: ModelPartitionMetadata,
    layer_count: int,
    expert_count: int,
    intermediate_size: int,
    force: bool,
) -> list[Path]:
    split = intermediate_size // 2
    if split <= 0 or split >= intermediate_size:
        raise ValueError("native microshard preparation requires an intermediate width >= 2")
    roots = [output / "microshard-0", output / "microshard-1"]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    entries: list[list[dict[str, object]]] = [[], []]
    ownership: list[list[dict[str, object]]] = [[], []]
    loader = safetensors_expert_loader(model_path)
    for layer_id in range(layer_count):
        for expert_id in range(expert_count):
            weights = loader(layer_id, expert_id)
            for shard_id, (hidden_start, hidden_end) in enumerate(
                ((0, split), (split, intermediate_size))
            ):
                shard = slice_expert_weights(
                    weights,
                    hidden_start=hidden_start,
                    hidden_end=hidden_end,
                )
                shard_path = roots[shard_id] / f"layer-{layer_id}-expert-{expert_id}.npz"
                _save_microshard(shard_path, shard, force=force)
                entry = {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "path": str(shard_path.resolve()),
                    "content_hash": shard.content_hash,
                }
                entries[shard_id].append(entry)
                ownership[shard_id].append(
                    {
                        **entry,
                        "hidden_start": hidden_start,
                        "hidden_end": hidden_end,
                        "logical_intermediate_dimension": intermediate_size,
                    }
                )
            print(f"microshards layer={layer_id} expert={expert_id}", flush=True)
    manifests = []
    for shard_id, root in enumerate(roots):
        manifest_path = root / "manifest.json"
        _write_json(
            manifest_path,
            {
                **_base_manifest(metadata),
                "loader_type": "npz",
                "owned_experts": entries[shard_id],
                "owned_microshards": ownership[shard_id],
                "measured_service_rates": {
                    "microshard_calls_per_second": 1.0,
                    "reduction_calls_per_second": 1.0,
                },
            },
            force=force,
        )
        manifests.append(manifest_path)
    return manifests


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.whole and not args.microshards:
        _parser().error("select --whole, --microshards, or both")
    model_path = args.model_path.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = inspect_olmoe_partition_metadata(
        model_path,
        model_revision=REAL_MODEL_REVISION,
        tokenizer_revision=args.tokenizer_revision,
    )
    layer_count = len(metadata.layer_costs)
    if args.whole:
        path = _prepare_whole(
            model_path=model_path,
            output=output,
            metadata=metadata,
            layer_count=layer_count,
            expert_count=metadata.expert_count,
            force=args.force,
        )
        print(f"whole_manifest={path}")
    if args.microshards:
        paths = _prepare_microshards(
            model_path=model_path,
            output=output,
            metadata=metadata,
            layer_count=layer_count,
            expert_count=metadata.expert_count,
            intermediate_size=metadata.expert_intermediate_size,
            force=args.force,
        )
        print("microshard_manifests=" + ";".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
