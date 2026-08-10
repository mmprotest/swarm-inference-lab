"""Fail-closed semantic support matrix for the official Kimi K3 graph.

The matrix deliberately distinguishes a correctness/reference implementation
from a production GPU implementation.  Finding a C function with the right
name is useful evidence that a semantic path exists, but it is not evidence
that the path has executed real checkpoint weights or that it can run on
``sm_86``.  Those statuses are promoted only by machine-readable receipts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_014.census import _layer_attention_types

SCHEMA_VERSION = "experiment-014-k3-model-support-v1"

_SOURCE_MARKERS = {
    "embedding": "model.embed_tokens.weight",
    "kda": "static void kda_forward",
    "gated_mla": "static void mla_forward",
    "dense_mlp": "static void dense_forward",
    "moe_router_and_reduction": "static void moe_forward",
    "routing_telemetry": "rt_route(li,t,idx,wsel,Kt)",
    "shared_experts": "shared_experts.gate_proj.weight",
    "routed_experts": "block_sparse_moe.experts.%d.%s.weight_%s",
    "attention_residual": "res_mix(",
    "final_norm": "model.norm.weight",
    "lm_head": "lm_head.weight",
    "tokenizer": "tok_load(&T,tp)",
    "chat_template": "chat_build(&T,sysmsg,prompt,think",
    "sampler": "sample_tok(lo,m.c.vocab",
    "eos": "m.c.n_eos",
    "stateful_step": "step_chunk(&m,&t,np+ntok-1,1)",
    "kda_state": "m->kstate[i]=fcalloc",
    "mla_cache": "m->Lc[i]=falloc",
}


class ModelSupportError(ValueError):
    """The support matrix inputs are inconsistent or incomplete."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ModelSupportError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_layers(receipt: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    if not receipt or receipt.get("status") != "PASS":
        return {}
    rows = receipt.get("layers")
    if not isinstance(rows, list):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("layer"), int):
            result[int(row["layer"])] = row
    return result


def build_model_support_matrix(
    checkpoint: Path,
    engine_source: Path,
    output_path: Path,
    *,
    oracle_receipt_path: Path | None = None,
    placement_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Create the 93-layer support matrix and return its compact receipt."""

    root = checkpoint.expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ModelSupportError(f"missing checkpoint config: {config_path}")
    source_path = engine_source.expanduser().resolve()
    if not source_path.is_file():
        raise ModelSupportError(f"missing Kimi engine source: {source_path}")
    config = _load_json(config_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, Mapping):
        raise ModelSupportError("checkpoint config has no text_config object")
    layer_count = int(text_config["num_hidden_layers"])
    if layer_count != 93:
        # The checkpoint remains authoritative; this is a conspicuous warning,
        # not an instruction to force the documented value.
        expected_layer_note = f"checkpoint-authoritative layer count is {layer_count}, not 93"
    else:
        expected_layer_note = "checkpoint-authoritative layer count matches 93"
    attention = _layer_attention_types(config)
    dense_count = int(text_config["first_k_dense_replace"])
    source = source_path.read_text(encoding="utf-8", errors="replace")
    source_evidence = {
        name: {"marker": marker, "present": marker in source}
        for name, marker in _SOURCE_MARKERS.items()
    }
    missing_markers = sorted(name for name, row in source_evidence.items() if not row["present"])

    oracle = (
        _load_json(oracle_receipt_path)
        if oracle_receipt_path and oracle_receipt_path.exists()
        else None
    )
    oracle_layers = _receipt_layers(oracle)
    placement = (
        _load_json(placement_receipt_path)
        if placement_receipt_path and placement_receipt_path.exists()
        else None
    )
    placement_layers = set()
    if placement and placement.get("status") == "PASS":
        coverage = placement.get("coverage", {})
        raw = coverage.get("covered_layers", []) if isinstance(coverage, Mapping) else []
        if isinstance(raw, list):
            placement_layers = {int(value) for value in raw}

    layers: list[dict[str, Any]] = []
    for layer in range(layer_count):
        attention_type = attention[layer]
        is_dense = layer < dense_count
        required_markers = [
            attention_type,
            "attention_residual",
            "dense_mlp" if is_dense else "moe_router_and_reduction",
        ]
        if not is_dense:
            required_markers.extend(["shared_experts", "routed_experts", "routing_telemetry"])
        static_supported = all(source_evidence[name]["present"] for name in required_markers)
        oracle_row = oracle_layers.get(layer)
        correctness = (
            "PASS" if oracle_row is not None and oracle_row.get("status") == "PASS" else "NOT_RUN"
        )
        production_gpu = "BLOCKER"
        layers.append(
            {
                "layer": layer,
                "attention_architecture": attention_type,
                "moe_architecture": "dense_situ_glu"
                if is_dense
                else "latent_moe_896_top16_plus_2_shared",
                "required_state": (
                    ["kda_recurrent_matrix", "qkv_short_convolution_windows", "sequence_position"]
                    if attention_type == "kda"
                    else ["gated_mla_latent_cache", "gated_mla_rope_cache", "sequence_position"]
                ),
                "checkpoint_weight_representation": (
                    "BF16 non-expert weights; native packed MXFP4+UE8M0 routed experts"
                ),
                "reference_kernel_implementation": (
                    "kda_forward" if attention_type == "kda" else "mla_forward"
                ),
                "routing_implementation": "not_applicable" if is_dense else "moe_forward/rt_route",
                "reduction_semantics": (
                    "dense SiTU-GLU residual"
                    if is_dense
                    else "normalized sigmoid top-16 weighted routed sum + shared expert"
                ),
                "positional_state_behaviour": (
                    "causal recurrent KDA state advances once per token"
                    if attention_type == "kda"
                    else "causal MLA cache indexed by absolute sequence position"
                ),
                "cache_state_update_behaviour": (
                    "in-place KDA recurrence and depthwise-convolution window update"
                    if attention_type == "kda"
                    else "append compressed KV and positional cache row"
                ),
                "static_reference_path": "IMPLEMENTED" if static_supported else "MISSING",
                "correctness_test_status": correctness,
                "physical_placement_support": "PASS" if layer in placement_layers else "NOT_PROVEN",
                "production_sm86_execution": production_gpu,
                "oracle_evidence": oracle_row,
            }
        )

    tested = sum(row["correctness_test_status"] == "PASS" for row in layers)
    static_count = sum(row["static_reference_path"] == "IMPLEMENTED" for row in layers)
    placed = sum(row["physical_placement_support"] == "PASS" for row in layers)
    full_oracle = tested == layer_count and oracle is not None and oracle.get("full_graph") is True
    status = "PASS" if full_oracle and not missing_markers and placed == layer_count else "FAIL"
    component_rows = []
    for component in (
        "embedding",
        "final_norm",
        "lm_head",
        "tokenizer",
        "chat_template",
        "sampler",
        "eos",
        "stateful_step",
        "kda_state",
        "mla_cache",
    ):
        component_rows.append(
            {
                "component": component,
                "static_reference_path": (
                    "IMPLEMENTED" if source_evidence[component]["present"] else "MISSING"
                ),
                "correctness_test_status": (
                    "PASS"
                    if oracle and oracle.get("component_status", {}).get(component) == "PASS"
                    else "NOT_RUN"
                ),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "checkpoint": {
            "path": str(root),
            "config_sha256": _sha256(config_path),
            "layer_count_note": expected_layer_note,
        },
        "engine": {
            "source": str(source_path),
            "source_sha256": _sha256(source_path),
            "reference_activation_precision": "FP32",
            "production_activation_target": "MXFP8",
            "full_kimi_cuda_backend": "ABSENT",
            "sm86_status": "BLOCKER: full Kimi engine has CPU/Vulkan paths but no CUDA path",
        },
        "summary": {
            "transformer_layers": layer_count,
            "static_reference_paths": static_count,
            "real_weight_correctness_passed": tested,
            "physical_placement_passed": placed,
            "missing_static_markers": missing_markers,
            "full_graph_oracle_passed": full_oracle,
            "gate_2_93_of_93_tested": tested == layer_count,
        },
        "source_evidence": source_evidence,
        "components": component_rows,
        "layers": layers,
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "summary": payload["summary"],
    }


__all__ = ["ModelSupportError", "build_model_support_matrix"]
