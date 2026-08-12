"""Pinned Kimi K3 DSpark reference benchmark harness."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_k3_graph_runtime import _CheckpointReader
from swarm_inference.experiments.experiment_015.dspark_reference import K3DSparkReference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bf16_rows(source: np.ndarray, rows: list[int]) -> Any:
    import torch

    bits = np.ascontiguousarray(source[rows], dtype=np.uint16)
    return torch.from_numpy(bits).view(torch.bfloat16)


def _bf16_view(source: np.ndarray) -> Any:
    import torch

    # Safetensor/memmap views can be read-only.  PyTorch warns because its
    # tensor API permits writes even though this reference never mutates the
    # data; an explicit copy makes the ownership contract unambiguous.
    return torch.from_numpy(np.array(source, copy=True)).view(torch.bfloat16)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def benchmark_dspark_reference(
    target_checkpoint: Path,
    draft_checkpoint: Path,
    target_trace: Path,
    output_path: Path,
    *,
    block_sizes: tuple[int, ...] = (1, 2, 3, 5, 7),
    target_layers: tuple[int, ...] = (2, 23, 47, 71, 89),
    context_token_ids: tuple[int, ...] = (163584, 18699, 11),
    anchor_token_id: int = 374,
    mask_token_id: int = 163837,
    cycle_id: str = "H015-001A",
) -> dict[str, Any]:
    """Run real DSpark weights against real certified target hidden states on CPU.

    The result establishes semantic executability and draft candidates. CPU wall
    timing is explicitly excluded from GPU serving capacity evidence.
    """
    import torch

    if tuple(sorted(set(block_sizes))) != block_sizes or any(
        size < 1 or size > 7 for size in block_sizes
    ):
        raise ValueError("DSpark block sizes must be unique, sorted, and inside [1, 7]")
    target_checkpoint = target_checkpoint.expanduser().resolve()
    draft_checkpoint = draft_checkpoint.expanduser().resolve()
    target_trace = target_trace.expanduser().resolve()
    reader = _CheckpointReader(target_checkpoint)
    draft = K3DSparkReference(draft_checkpoint)
    configured_layers = reader.config.layers
    row_width = reader.config.hidden
    trace_values = np.memmap(target_trace, mode="r", dtype="<f4")
    row_count = int(trace_values.size // row_width)
    if row_count % (configured_layers + 1):
        raise ValueError("target trace does not contain complete Kimi forward steps")
    steps = row_count // (configured_layers + 1)
    if steps != len(context_token_ids):
        raise ValueError("target trace steps do not match the deterministic context")
    trace = trace_values.reshape(steps, configured_layers + 1, row_width)
    target_hidden = np.ascontiguousarray(
        np.concatenate([trace[:, layer] for layer in target_layers], axis=1),
        dtype=np.float32,
    )
    embedding_source = reader.array("language_model.model.embed_tokens.weight")
    head_source = reader.array("language_model.lm_head.weight")
    target_head = _bf16_view(head_source)
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        query_ids = [anchor_token_id, *([mask_token_id] * (block_size - 1))]
        query_embeddings = _bf16_rows(embedding_source, query_ids)
        started = time.perf_counter()
        base_logits, details = draft.forward_block(
            target_hidden,
            list(range(steps)),
            query_embeddings,
            list(range(steps, steps + block_size)),
            target_head,
        )
        corrected = draft.apply_markov_bias(base_logits, anchor_token_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        tokens = [int(value) for value in corrected.argmax(dim=-1).tolist()]
        rows.append(
            {
                "block_size": block_size,
                "query_token_ids": query_ids,
                "draft_token_ids": tokens,
                "base_argmax_token_ids": [
                    int(value) for value in base_logits.argmax(dim=-1).tolist()
                ],
                "finite": bool(torch.isfinite(corrected).all()),
                "reference_wall_ms": elapsed_ms,
                "backbone_wall_ms": float(details["wall_ms"]),
                "markov_wall_ms": elapsed_ms - float(details["wall_ms"]),
                "timing_scope": (
                    "CPU BF16 correctness reference; excludes load; not GPU capacity evidence"
                ),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": "experiment-015-k3-dspark-reference-v1",
        "cycle_id": cycle_id,
        "status": "PASS" if all(row["finite"] for row in rows) else "FAIL",
        "evidence_class": None,
        "scientific_result": False,
        "diagnostic_scope": (
            "CPU public-weight correctness diagnostic; outside the four "
            "Experiment 015 scientific evidence classes"
        ),
        "measurement_scope": (
            "real public Kimi K3 DSpark BF16 weights and real Kimi target hidden states; "
            "CPU correctness/reference timing only"
        ),
        "capacity_evidence": False,
        "target": {
            "checkpoint": str(target_checkpoint),
            "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
            "trace": str(target_trace),
            "trace_sha256": _sha256(target_trace),
            "target_layer_ids": list(target_layers),
            "context_token_ids": list(context_token_ids),
            "anchor_token_id": anchor_token_id,
        },
        "draft": {
            "checkpoint": str(draft_checkpoint),
            "revision": "cf6b8244620e7ea4b0651d214f28e89eac75bed6",
            "model_sha256": _sha256(draft_checkpoint / "model.safetensors"),
            "config_sha256": _sha256(draft_checkpoint / "config.json"),
            "model_bytes": (draft_checkpoint / "model.safetensors").stat().st_size,
            "mask_token_id": mask_token_id,
        },
        "runtime": {
            "python": __import__("sys").version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "dtype": "bfloat16",
        },
        "blocks": rows,
        "correctness": {
            "all_finite": all(row["finite"] for row in rows),
            "deterministic_greedy": True,
            "target_distribution_equivalence": "NOT_ESTABLISHED_BY_DRAFT_REFERENCE_ALONE",
        },
    }
    _atomic_json(output_path.resolve(), receipt)
    return receipt
