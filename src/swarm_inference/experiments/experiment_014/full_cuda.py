"""Streamed, no-fallback CUDA execution of the complete real Kimi K3 graph.

This module deliberately keeps the experiment's local full-graph runner narrow:
one CUDA context is reused, while one layer's immutable weights are resident at
a time.  Mathematical operations execute through the certified CUDA ABI; host
memory is used only to carry layer boundaries and checkpoint request state when
the full 93-layer model cannot fit on the local GPU at once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_k3_graph_runtime import (
    SCHEMA_VERSION,
    KimiCudaGraphRunner,
    _CheckpointReader,
    _LayerResources,
    _parse_oracle_routes,
    _pointer_offset,
    _quantize_bf16_grouped_int4,
)
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _device_identity,
    _sha256_file,
)

__all__ = (
    "KimiCudaGraphRunner",
    "_CheckpointReader",
    "_LayerResources",
    "_parse_oracle_routes",
    "_pointer_offset",
    "_quantize_bf16_grouped_int4",
    "benchmark_streamed_cuda_graph",
)


def benchmark_streamed_cuda_graph(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace_path: Path,
    oracle_routes_path: Path,
    output_path: Path,
    *,
    oracle_logits_path: Path | None = None,
    layer_limit: int = 4,
    prompt_token_ids: tuple[int, ...] = (163584, 18699),
    decode_token_id: int | None = 11,
    device: int = 0,
    relative_error_gate: float = 3e-3,
    cycle_id: str = "H014-025af",
    oracle_layer_count: int | None = None,
) -> dict[str, Any]:
    """Execute a real Kimi prefix, optionally including one subsequent decode."""

    checkpoint_path = checkpoint.expanduser().resolve()
    library_path = cuda_library.expanduser().resolve()
    trace_path = oracle_trace_path.expanduser().resolve()
    routes_path = oracle_routes_path.expanduser().resolve()
    for source in (checkpoint_path, library_path, trace_path, routes_path):
        if not source.exists():
            raise KimiCudaError(f"missing streamed-graph input: {source}")
    reader_config = json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
    text_config = reader_config["text_config"]
    hidden = int(text_config["hidden_size"])
    layers = int(text_config["num_hidden_layers"])
    reference_layers = oracle_layer_count or layers
    if layer_limit < 1 or layer_limit > layers:
        raise KimiCudaError(f"layer limit must be in [1,{layers}]")
    if reference_layers < layer_limit or reference_layers > layers:
        raise KimiCudaError(
            f"oracle layer count must be in [{layer_limit},{layers}]"
        )
    trace = np.memmap(trace_path, mode="r", dtype="<f4")
    if trace.size % hidden:
        raise KimiCudaError("oracle trace contains a partial hidden row")
    trace = trace.reshape(-1, hidden)
    expected_steps = len(prompt_token_ids) + int(decode_token_id is not None)
    required_rows = expected_steps * (reference_layers + 1)
    if trace.shape[0] < required_rows:
        raise KimiCudaError(
            f"oracle trace has {trace.shape[0]} rows, expected at least {required_rows}"
        )
    oracle_logits = None
    if oracle_logits_path is not None:
        logits_path = oracle_logits_path.expanduser().resolve()
        logits_raw = np.memmap(logits_path, mode="r", dtype="<f4")
        vocab = int(text_config["vocab_size"])
        if logits_raw.size % vocab:
            raise KimiCudaError("oracle logits contain a partial vocabulary row")
        oracle_logits = logits_raw.reshape(-1, vocab)
    expected_routes = _parse_oracle_routes(routes_path)
    started = time.perf_counter_ns()
    runner = KimiCudaGraphRunner(checkpoint_path, library_path, device)
    try:
        prefill = runner.execute_pass(
            list(prompt_token_ids),
            list(range(len(prompt_token_ids))),
            layer_limit=layer_limit,
            maximum_context=len(prompt_token_ids) + int(decode_token_id is not None),
            oracle_trace=trace,
            oracle_layer_count=reference_layers,
            oracle_routes=expected_routes,
            oracle_logits=oracle_logits,
            progress_label="prefill",
        )
        selected_decode = decode_token_id
        if layer_limit == layers:
            sampled = int(prefill["sampled_token_id"])
            if decode_token_id is not None and sampled != decode_token_id:
                raise KimiCudaError(
                    f"full CUDA graph sampled token {sampled}, expected {decode_token_id}"
                )
            selected_decode = sampled
        decode = None
        if selected_decode is not None:
            decode = runner.execute_pass(
                [selected_decode],
                [len(prompt_token_ids)],
                layer_limit=layer_limit,
                maximum_context=len(prompt_token_ids) + 1,
                oracle_trace=trace,
                oracle_layer_count=reference_layers,
                oracle_routes=expected_routes,
                oracle_logits=None,
                progress_label="decode",
            )
        maximum_error = max(
            float(prefill["maximum_layer_relative_l2_error"]),
            float(decode["maximum_layer_relative_l2_error"]) if decode else 0.0,
        )
        routing_equal = bool(prefill["routing_equality"]) and (
            decode is None or bool(decode["routing_equality"])
        )
        required_operations = set(runner.operation_counts)
        covered_operations = {
            name for name, count in runner.operation_counts.items() if count > 0
        }
        expected_operations = (
            required_operations
            if layer_limit == layers
            else required_operations - {"final_norm", "LM_head"}
        )
        status = (
            "PASS"
            if maximum_error <= relative_error_gate
            and routing_equal
            and expected_operations.issubset(covered_operations)
            else "FAIL"
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle_id": cycle_id,
            "status": status,
            "hypothesis": (
                "The certified CUDA primitives compose without a new mathematical kernel "
                "into a real contiguous Kimi graph, preserve exact routing, and stay within "
                f"{relative_error_gate:g} relative L2 of the retained serial graph."
            ),
            "checkpoint": {
                "path": str(checkpoint_path),
                "config_sha256": _sha256_file(checkpoint_path / "config.json"),
                "index_sha256": _sha256_file(
                    checkpoint_path / "model.safetensors.index.json"
                ),
            },
            "backend": {
                "identity": "nvidia_cuda_streamed_kimi_graph",
                "cuda_library": str(library_path),
                "cuda_library_sha256": runner.runtime.sha256,
                "cpu_fallback_allowed": False,
                "binary_min_compute_capability": runner.runtime.binary_min_compute_capability,
                "binary_has_forward_ptx": runner.runtime.binary_has_forward_ptx,
                "capability_negotiation": runner.runtime.capability_negotiation,
                "device": _device_identity(device),
            },
            "fixture": {
                "layer_limit": layer_limit,
                "configured_layers": layers,
                "oracle_layer_count": reference_layers,
                "prompt_token_ids": list(prompt_token_ids),
                "decode_token_id": selected_decode,
                "oracle_trace": str(trace_path),
                "oracle_trace_sha256": _sha256_file(trace_path),
                "oracle_routes": str(routes_path),
                "oracle_routes_sha256": _sha256_file(routes_path),
            },
            "prefill": prefill,
            "decode": decode,
            "correctness": {
                "relative_error_gate": relative_error_gate,
                "maximum_layer_relative_l2_error": maximum_error,
                "routing_equality": routing_equal,
                "stateful_decode_executed": decode is not None,
            },
            "coverage": {
                "layers_executed": layer_limit,
                "configured_layers": layers,
                "operation_counts": runner.operation_counts,
                "covered_operations": sorted(covered_operations),
                "required_for_this_fixture": sorted(expected_operations),
                "no_cpu_mathematical_fallback": True,
            },
            "timing": {
                "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
                "classification": "streamed correctness run; not capacity evidence",
            },
        }
    finally:
        runner.close()
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
    return {
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "layers_executed": layer_limit,
        "maximum_layer_relative_l2_error": payload["correctness"][
            "maximum_layer_relative_l2_error"
        ],
        "routing_equality": payload["correctness"]["routing_equality"],
        "wall_seconds": payload["timing"]["wall_seconds"],
    }


__all__ = ["KimiCudaGraphRunner", "benchmark_streamed_cuda_graph"]
