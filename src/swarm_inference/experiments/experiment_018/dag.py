"""Exact Kimi K3 token/chunk-by-depth dependency description."""

from __future__ import annotations

from typing import Any

from swarm_inference.experiments.experiment_018.wavefront import (
    ATTNRES_SNAPSHOT_LAYERS,
    MICROCELL_COUNT,
    MICROCELL_DEPTH,
)


def k3_dependency_proof() -> dict[str, Any]:
    layer_types = [
        {
            "operator": "KDA recurrent state",
            "same_position_dependencies": ["input RMSNorm", "KDA projections"],
            "previous_position_dependencies": [
                "per-layer 96x128x128 recurrent state"
            ],
            "release_rule": (
                "position t may leave the KDA layer after its recurrent update; "
                "position t+1 must observe that update"
            ),
            "chunk_rule": "chunks must be ordered at every KDA layer",
            "exact_streaming_legal": True,
        },
        {
            "operator": "causal convolution",
            "same_position_dependencies": ["q/k/v projections"],
            "previous_position_dependencies": [
                "three per-layer width-4 convolution windows"
            ],
            "release_rule": "release after the width-4 windows advance for position t",
            "chunk_rule": "preserve contiguous position order and carry windows",
            "exact_streaming_legal": True,
        },
        {
            "operator": "Gated MLA",
            "same_position_dependencies": [
                "query/kv/gate projections",
                "causal attention reduction",
            ],
            "previous_position_dependencies": [
                "per-layer compressed-KV and RoPE cache through t-1"
            ],
            "release_rule": (
                "position t may leave after its KV row is appended and causal context "
                "is computed; t+1 observes that row"
            ),
            "chunk_rule": "chunks must be ordered at every MLA layer",
            "exact_streaming_legal": True,
        },
        {
            "operator": "MoE routing",
            "same_position_dependencies": ["post-attention normalized hidden row"],
            "previous_position_dependencies": [],
            "release_rule": "routes are row-local and may fan out once that row is ready",
            "chunk_rule": "no cross-position route dependency",
            "exact_streaming_legal": True,
        },
        {
            "operator": "routed experts",
            "same_position_dependencies": [
                "route IDs/weights",
                "LatentMoE down projection",
            ],
            "previous_position_dependencies": [],
            "release_rule": "all selected expert contributions for the row must complete",
            "chunk_rule": "rows may batch by expert without changing row identity",
            "exact_streaming_legal": True,
        },
        {
            "operator": "shared experts",
            "same_position_dependencies": ["post-attention normalized hidden row"],
            "previous_position_dependencies": [],
            "release_rule": "shared-expert output is row-local",
            "chunk_rule": "no cross-position dependency",
            "exact_streaming_legal": True,
        },
        {
            "operator": "LatentMoE reductions/projections",
            "same_position_dependencies": [
                "all 16 routed expert outputs in router-slot order",
                "route weights",
                "shared expert output",
            ],
            "previous_position_dependencies": [],
            "release_rule": (
                "stable FP32 reduction, routed norm, latent up projection and shared "
                "addition must finish for the row"
            ),
            "chunk_rule": "expert batching may cross rows; scatter restores row/slot order",
            "exact_streaming_legal": True,
        },
        {
            "operator": "Attention Residuals (AttnRes)",
            "same_position_dependencies": [
                "current prefix",
                "immutable snapshots already produced at layers 0,12,...,84",
                "current layer static depth-query weights",
            ],
            "previous_position_dependencies": [],
            "release_rule": (
                "a snapshot becomes immutable immediately after its producer layer for "
                "that position and may be cached downstream by version/hash"
            ),
            "chunk_rule": "snapshots are request/block/chunk scoped; stale versions reject",
            "exact_streaming_legal": True,
        },
        {
            "operator": "layer/block residual state",
            "same_position_dependencies": [
                "incoming prefix",
                "attention output",
                "MoE output",
            ],
            "previous_position_dependencies": [],
            "release_rule": "the complete nine-row boundary may leave after residual update",
            "chunk_rule": "current hidden changes; completed snapshots remain immutable",
            "exact_streaming_legal": True,
        },
        {
            "operator": "endpoint/logits",
            "same_position_dependencies": [
                "layer-92 output",
                "eight completed AttnRes snapshots",
                "final RMSNorm and LM head",
            ],
            "previous_position_dependencies": [],
            "release_rule": "logits for a position release after the final exact mix/head",
            "chunk_rule": "endpoint rows are independent after their layer-92 boundary",
            "exact_streaming_legal": True,
        },
    ]
    cells = []
    for cell in range(MICROCELL_COUNT):
        start = cell * MICROCELL_DEPTH
        end = min(93, start + MICROCELL_DEPTH)
        cells.append(
            {
                "microcell_id": cell,
                "layer_start": start,
                "layer_end_exclusive": end,
                "same_chunk_dependency": (
                    None if cell == 0 else f"handoff(cell={cell - 1}, same_chunk)"
                ),
                "state_dependency": "same_cell_previous_chunk",
                "attnres_dependencies": [
                    layer for layer in ATTNRES_SNAPSHOT_LAYERS if layer < start
                ],
                "release": f"handoff(cell={cell}, same_chunk)" if cell < 11 else "endpoint",
            }
        )
    return {
        "schema_version": "experiment-018-k3-wavefront-dag-v1",
        "claim": (
            "Exact streaming is legal only with ordered per-layer recurrent/cache state; "
            "there is no all-block barrier between adjacent microcells."
        ),
        "position_order": "strictly increasing within each KDA/MLA layer",
        "microcell_task_start_equation": (
            "max(previous_chunk_finish_same_cell, previous_cell_chunk_arrival, "
            "required_state_ready, required_AttnRes_object_ready)"
        ),
        "task_schemas": {
            "VerificationBlock": ["block_id", "request_id", "positions"],
            "WavefrontChunk": [
                "block_id",
                "chunk_id",
                "start_position",
                "end_position",
            ],
            "MicrocellTask": [
                "request_id",
                "block_id",
                "chunk_id",
                "microcell_id",
                "dependencies",
                "input_refs",
                "state_refs",
                "output_ref",
            ],
            "ExpertTask": [
                "request_id",
                "block_id",
                "chunk_id",
                "microcell_id",
                "layer",
                "expert",
                "shard",
                "route_weight",
                "dependencies",
            ],
            "ReductionTask": ["task_id", "dependency_task_ids", "stable_order"],
            "AttnResObject": [
                "request_id",
                "block_id",
                "chunk_id",
                "snapshot_layer",
                "content_hash",
                "version",
                "producer_microcell",
                "cached_locations",
                "size_bytes",
            ],
        },
        "layer_types": layer_types,
        "fixed_topology": {
            "layers": 93,
            "microcell_depth": 8,
            "microcell_count": 12,
            "internal_boundaries": 81,
            "coarse_boundaries": 11,
            "cells": cells,
        },
        "prohibited_schedule": (
            "position/chunk t+1 may not overtake t inside a stateful KDA or MLA layer"
        ),
        "physical_proof_location": "dependencies/chunk-equivalence.json",
    }


__all__ = ["k3_dependency_proof"]
