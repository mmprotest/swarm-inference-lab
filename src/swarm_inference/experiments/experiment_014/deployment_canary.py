"""Experiment 015 physical RTX 3090 Kimi qualification canary.

The fixture builder runs on the development host and distils the immutable
three-position serial oracle into a small, hash-locked bundle.  The physical
runner consumes only four worker-scoped snapshots and the canary-built Linux
ELF.  A failed run always retains a diagnostic receipt and never emits the
production runtime certificate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.execution.kimi_k3_stage import (
    PersistentKimiStageExecutor,
    _boundary_fixtures,
)
from swarm_inference.experiments.experiment_014.cuda import (
    KimiCudaError,
    _array_fingerprint,
    _numerical_metrics,
)
from swarm_inference.experiments.experiment_014.deployment_admission import (
    inspect_local_gpu,
)
from swarm_inference.experiments.experiment_014.full_cuda import _parse_oracle_routes
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.runtime_qualification import (
    BUILD_ARGUMENTS,
    CERTIFICATE_SCHEMA,
    KIMI_OPERATION_CLASSES,
    KIMI_STAGE_ROLES,
    PHYSICAL_CERTIFICATE_GATES,
    validate_linux_runtime_certificate,
    validate_native_source_manifest,
)
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.stage_worker import LoadStageRequest
from swarm_inference.worker.stage_runtime import PersistentStageRuntime

FIXTURE_SCHEMA = "experiment-015-k3-physical-canary-fixtures-v1"
RECEIPT_SCHEMA = "experiment-015-k3-physical-canary-receipt-v1"
ASSIGNED_STAGE_SCHEMA = "experiment-015-k3-assigned-stage-canary-v1"
CANARY_WORKERS = {
    "stage_zero_embedding_dense": 0,
    "kda_moe": 89,
    "gated_mla_moe": 91,
    "final_norm_head_sampling": 92,
}
ROLE_OPERATIONS = {
    "stage_zero_embedding_dense": {
        "embedding",
        "grouped_int4_dense",
        "attnres",
        "kda_stage",
    },
    "kda_moe": {
        "mxfp4_routed_expert",
        "router_top16",
        "shared_expert",
        "moe_reduction",
        "attnres",
        "kda_stage",
    },
    "gated_mla_moe": {
        "mxfp4_routed_expert",
        "router_top16",
        "shared_expert",
        "moe_reduction",
        "attnres",
        "gated_mla_stage",
    },
    "final_norm_head_sampling": {
        "mxfp4_routed_expert",
        "router_top16",
        "shared_expert",
        "moe_reduction",
        "attnres",
        "gated_mla_stage",
        "final_rmsnorm",
        "lm_head",
    },
}
TOKEN_IDS = np.asarray([[163584], [18699], [11]], dtype=np.int64)
EXPECTED_TOKENS = np.asarray([220, 11, 374], dtype=np.int64)
RELATIVE_L2_GATE = 3e-5


class PhysicalCanaryError(RuntimeError):
    """The physical canary input or execution failed closed."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhysicalCanaryError(f"invalid JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise PhysicalCanaryError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def _stack(values: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate(values, axis=0))


def build_physical_canary_fixtures(
    checkpoint: Path,
    oracle_trace: Path,
    oracle_routes: Path,
    oracle_logits: Path,
    placement_path: Path,
    output_npz: Path,
    output_manifest: Path,
    *,
    full_graph_receipt: Path,
    operation_matrix_receipt: Path,
    sm86_receipt: Path,
) -> dict[str, Any]:
    """Distil real Kimi inputs/outputs without copying checkpoint weights."""

    sources = {
        "checkpoint": checkpoint.expanduser().resolve(),
        "oracle_trace": oracle_trace.expanduser().resolve(),
        "oracle_routes": oracle_routes.expanduser().resolve(),
        "oracle_logits": oracle_logits.expanduser().resolve(),
        "placement": placement_path.expanduser().resolve(),
        "full_graph": full_graph_receipt.expanduser().resolve(),
        "operation_matrix": operation_matrix_receipt.expanduser().resolve(),
        "sm86": sm86_receipt.expanduser().resolve(),
    }
    for name, path in sources.items():
        if name == "checkpoint":
            if not path.is_dir():
                raise PhysicalCanaryError(f"checkpoint directory is absent: {path}")
        elif not path.is_file():
            raise PhysicalCanaryError(f"canary source is absent: {path}")
    placement = _read(sources["placement"])
    if placement.get("status") != "PASS" or placement.get("node_count") != 93:
        raise PhysicalCanaryError("physical canary placement is not the passing 93-worker plan")
    for name in ("full_graph", "operation_matrix", "sm86"):
        if _read(sources[name]).get("status") != "PASS":
            raise PhysicalCanaryError(f"reference evidence is not passing: {name}")

    arrays: dict[str, np.ndarray] = {"token_ids": TOKEN_IDS.copy()}
    parsed_routes = _parse_oracle_routes(sources["oracle_routes"])
    for layer in (0, 89, 91):
        inputs, expected = _stage_fixtures(
            sources["checkpoint"], sources["oracle_trace"], layer=layer
        )
        arrays[f"layer_{layer}_input"] = (
            TOKEN_IDS.copy() if layer == 0 else _stack(inputs)
        )
        arrays[f"layer_{layer}_expected_boundary"] = _stack(expected)
        arrays[f"layer_{layer}_expected_routes"] = (
            np.empty((3, 0), dtype=np.int64)
            if layer == 0
            else np.asarray(
                [parsed_routes[layer][position] for position in range(3)],
                dtype=np.int64,
            )
        )
    final_input, expected_layer, expected_final = _boundary_fixtures(
        sources["checkpoint"], sources["oracle_trace"]
    )
    arrays.update(
        {
            "layer_92_input": _stack(final_input),
            "layer_92_expected_layer": _stack(
                [value.reshape(1, -1) for value in expected_layer]
            ),
            "layer_92_expected_final": _stack(
                [value.reshape(1, -1) for value in expected_final]
            ),
            "layer_92_expected_routes": np.asarray(
                [parsed_routes[92][position] for position in range(3)],
                dtype=np.int64,
            ),
            "expected_tokens": EXPECTED_TOKENS.copy(),
            "expected_logits": np.ascontiguousarray(
                np.memmap(
                    sources["oracle_logits"],
                    mode="r",
                    dtype="<f4",
                    shape=(2, 163840),
                )
            ),
        }
    )
    destination = output_npz.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)

    checkpoint_config = sources["checkpoint"] / "config.json"
    checkpoint_index = sources["checkpoint"] / "model.safetensors.index.json"
    manifest = {
        "schema_version": FIXTURE_SCHEMA,
        "status": "PASS",
        "fixture_npz": destination.name,
        "fixture_npz_sha256": _sha256(destination),
        "placement_sha256": _sha256(sources["placement"]),
        "checkpoint": {
            "revision": placement["checkpoint"]["revision"],
            "fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
            "config_sha256": _sha256(checkpoint_config),
            "index_sha256": _sha256(checkpoint_index),
        },
        "oracle_sources": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in (
                ("trace", sources["oracle_trace"]),
                ("routes", sources["oracle_routes"]),
                ("logits", sources["oracle_logits"]),
            )
        },
        "reference_evidence": {
            name: {
                "file": sources[name].name,
                "sha256": _sha256(sources[name]),
                "status": "PASS",
            }
            for name in ("full_graph", "operation_matrix", "sm86")
        },
        "roles": CANARY_WORKERS,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
        "numerical_gate": {"relative_l2_error_at_most": RELATIVE_L2_GATE},
        "physical_execution_status": "NOT_RUN",
    }
    _atomic_json(output_manifest, manifest)
    return {
        "status": "PASS",
        "fixture_npz": str(destination),
        "fixture_npz_sha256": manifest["fixture_npz_sha256"],
        "manifest": str(output_manifest.resolve()),
        "manifest_sha256": _sha256(output_manifest.resolve()),
        "array_count": len(arrays),
        "bytes": destination.stat().st_size,
    }


def _load_fixture_bundle(npz_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _read(manifest_path)
    if manifest.get("schema_version") != FIXTURE_SCHEMA or manifest.get("status") != "PASS":
        raise PhysicalCanaryError("physical canary fixture manifest is not passing")
    if manifest.get("fixture_npz_sha256") != _sha256(npz_path):
        raise PhysicalCanaryError("physical canary NPZ hash differs")
    with np.load(npz_path, allow_pickle=False) as bundle:
        arrays = {name: np.ascontiguousarray(bundle[name]) for name in bundle.files}
    expected = manifest.get("arrays")
    if not isinstance(expected, dict) or sorted(expected) != sorted(arrays):
        raise PhysicalCanaryError("physical canary fixture array allowlist differs")
    for name, value in arrays.items():
        row = expected[name]
        if (
            row.get("shape") != list(value.shape)
            or row.get("dtype") != str(value.dtype)
            or row.get("sha256") != _array_sha256(value)
        ):
            raise PhysicalCanaryError(f"physical canary array differs: {name}")
    return manifest, arrays


def _placement_assignment(worker: dict[str, Any], *, layer: int) -> StageAssignment:
    return StageAssignment(
        stage_id=layer,
        layer_start=layer,
        layer_end=layer + 1,
        layer_ids=(layer,),
        weight_bytes=int(worker["source_weight_bytes"]),
        estimated_compute_ns=max(
            1, round(float(worker["expected_compute"]["projected_rtx3090_wall_p50_ms"]) * 1e6)
        ),
        measured_compute_ns=None,
        kv_cache_bytes_per_token=6400,
        peak_temporary_bytes=512 * 1024**2,
        activation_bytes=258048,
        device="native-cuda:0",
        owns_embeddings=layer == 0,
        owns_final_norm=layer == 92,
        owns_output_projection=layer == 92,
    )


def _lifecycle_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(after[name]) - int(before[name])
        for name in (
            "weight_load_count",
            "model_materialization_count",
            "persistent_buffer_allocation_count",
        )
    }


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise PhysicalCanaryError("physical canary retained no timing values")
    return {
        "minimum_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "maximum_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


async def _run_role(
    *,
    role: str,
    layer: int,
    worker: dict[str, Any],
    snapshot: Path,
    runtime_path: Path,
    runtime_sha256: str,
    placement: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    identity_path = snapshot / "model-identity.json"
    identity = _read(identity_path)
    if identity.get("worker_id") != worker["worker_id"]:
        raise PhysicalCanaryError(f"{role}: snapshot worker identity differs")
    assignment = _placement_assignment(worker, layer=layer)
    runtime = PersistentStageRuntime(
        worker_id=worker["worker_id"],
        device="native-cuda:0",
        dtype="float32",
        memory_limit_bytes=int(worker["memory"]["physical_vram_bytes"]),
        maximum_sessions=32,
        configured_model_path=snapshot,
        configured_model_identity_path=identity_path,
        allow_model_download=False,
    )
    request = LoadStageRequest(
        worker_id=worker["worker_id"],
        request_id=f"experiment-015-canary-{role}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=str(placement["checkpoint"]["revision"]),
        tokenizer_revision=str(identity["tokenizer_revision"]),
        topology_id=f"experiment-015-canary-{role}",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_batch_bucket=8,
        fast_path_context_bucket=8192,
        model_content_fingerprint=str(placement["checkpoint"]["checkpoint_fingerprint"]),
        native_runtime_library=str(runtime_path),
        native_runtime_library_sha256=runtime_sha256,
        device="native-cuda:0",
        dtype="float32",
        model_path=str(snapshot),
        allow_download=False,
        deadline_unix_ns=time.time_ns() + 3_600_000_000_000,
    )
    try:
        response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not response.accepted or not isinstance(executor, PersistentKimiStageExecutor):
            raise PhysicalCanaryError(f"{role}: product loader did not return the Kimi executor")
        prepared = executor.lifecycle_snapshot()
        prepare = prepared.get("prepare_warmup") or {}
        prepare_fixture = prepare.get("stage_fixture") or {}
        prepare_pass = (
            prepared.get("prepare_warmup_count") == 1
            and prepare.get("count") == 1
            and prepare_fixture.get("iterations") == 7
            and prepare_fixture.get("active_sessions_after") == 0
            and prepare_fixture.get("serving_execute_count_restored") is True
            and prepare_fixture.get("research_records_removed") is True
            and prepare_fixture.get("output_finite") is True
            and prepare_fixture.get("cpu_mathematical_fallbacks") == 0
        )
        session_id = f"experiment-015-canary-{role}-serial"
        executor.open_session(session_id, maximum_context_override=3)
        before = executor.lifecycle_snapshot()
        correctness: list[dict[str, Any]] = []
        device_ms: list[float] = []
        try:
            for position in range(3):
                result = (
                    executor.execute_prefill(
                        session_id=session_id,
                        token_ids=torch.from_numpy(arrays["token_ids"][position : position + 1]),
                        cache_position_start=position,
                    )
                    if layer == 0
                    else executor.execute_decode(
                        session_id=session_id,
                        hidden_states=torch.from_numpy(
                            arrays[f"layer_{layer}_input"][position : position + 1]
                        ),
                        cache_position_start=position,
                    )
                )
                record = executor.execution_records[-1]
                device_ms.append(float(record["device_ms"]))
                observed_boundary = (
                    result.stage_boundary_hidden_states.detach().cpu().numpy()
                )
                if layer == 92:
                    layer_metrics = _numerical_metrics(
                        arrays["layer_92_expected_layer"][position], record["layer_output"]
                    )
                    final_metrics = _numerical_metrics(
                        arrays["layer_92_expected_final"][position], record["final_hidden"]
                    )
                    logits_metrics = (
                        _numerical_metrics(
                            arrays["expected_logits"][position], record["logits"]
                        )
                        if position < 2
                        else None
                    )
                    sampled = int(result.sampled_token_ids.item())
                    metrics_pass = (
                        float(layer_metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                        and float(final_metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                        and (
                            logits_metrics is None
                            or float(logits_metrics["relative_l2_error"])
                            <= RELATIVE_L2_GATE
                        )
                        and sampled == int(arrays["expected_tokens"][position])
                    )
                    details: dict[str, Any] = {
                        "layer_metrics": layer_metrics,
                        "final_metrics": final_metrics,
                        "logits_metrics": logits_metrics,
                        "sampled_token_id": sampled,
                        "expected_sampled_token_id": int(arrays["expected_tokens"][position]),
                    }
                else:
                    boundary_metrics = _numerical_metrics(
                        arrays[f"layer_{layer}_expected_boundary"][position : position + 1],
                        observed_boundary,
                    )
                    metrics_pass = (
                        float(boundary_metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                    )
                    details = {"boundary_metrics": boundary_metrics}
                observed_routes = list(record["selected_expert_ids"])
                expected_routes = arrays[f"layer_{layer}_expected_routes"][position].tolist()
                correctness.append(
                    {
                        "position": position,
                        "pass": metrics_pass and observed_routes == expected_routes,
                        "routing_equality": observed_routes == expected_routes,
                        "selected_expert_ids": observed_routes,
                        "expected_expert_ids": expected_routes,
                        "output_fingerprint": _array_fingerprint(observed_boundary),
                        **details,
                    }
                )
            state = executor.session_state_evidence(session_id)
            after = executor.lifecycle_snapshot()
        finally:
            executor.close_session(session_id)
        lifecycle = _lifecycle_delta(before, after)
        role_result: dict[str, Any] = {
            "role": role,
            "layer": layer,
            "worker_id": worker["worker_id"],
            "snapshot": str(snapshot),
            "snapshot_identity_sha256": _sha256(identity_path),
            "product_load_accepted": True,
            "prepare": prepare,
            "prepare_exactly_seven_calls": prepare_pass,
            "correctness": correctness,
            "correctness_pass": all(row["pass"] for row in correctness),
            "state": state,
            "stateful_decode_pass": (
                state["cache_sequence_length"] == 3
                and state["finite"]
                and state["nonzero_prefix"]
                and state["zero_suffix"]
            ),
            "timing": _timing(device_ms),
            "lifecycle_delta": lifecycle,
            "warm_lifecycle_deltas_zero": all(value == 0 for value in lifecycle.values()),
            "resident_device_bytes": int(prepared["resident_device_bytes"]),
            "batch_workspace_bytes": int(prepared["batch_workspace_bytes"]),
            "cpu_mathematical_fallbacks": int(prepared["cpu_mathematical_fallbacks"]),
        }
        if layer == 89:
            batch_ids = tuple(f"experiment-015-canary-b8-{row}" for row in range(8))
            for batch_id in batch_ids:
                executor.open_session(batch_id, maximum_context_override=3)
            batch_before = executor.lifecycle_snapshot()
            try:
                batch_record = executor.execute_decode_batch(
                    session_ids=batch_ids,
                    hidden_states=torch.from_numpy(
                        np.repeat(arrays["layer_89_input"][0:1], 8, axis=0)
                    ),
                    cache_position_start=0,
                    profile_phases=True,
                )
                batch_states = [executor.session_state_evidence(item) for item in batch_ids]
                batch_after = executor.lifecycle_snapshot()
            finally:
                for batch_id in batch_ids:
                    executor.close_session(batch_id)
            batch_comparisons = [
                _numerical_metrics(
                    arrays["layer_89_expected_boundary"][0],
                    batch_record["boundary_output"][row],
                )
                for row in range(8)
            ]
            batch_pass = (
                all(
                    float(row["relative_l2_error"]) <= RELATIVE_L2_GATE
                    for row in batch_comparisons
                )
                and all(
                    route == arrays["layer_89_expected_routes"][0].tolist()
                    for route in batch_record["selected_expert_ids"]
                )
                and bool(batch_record["routing"]["all_selected_experts_executed_once"])
                and len({row["fingerprint"] for row in batch_states}) == 1
                and all(row["cache_sequence_length"] == 1 for row in batch_states)
                and all(
                    value == 0
                    for value in _lifecycle_delta(batch_before, batch_after).values()
                )
            )
            guard_before = executor.lifecycle_snapshot()
            records_before = len(executor.execution_records)
            rejection = ""
            try:
                executor.execute_decode_batch(
                    session_ids=tuple(f"uncertified-{row}" for row in range(9)),
                    hidden_states=torch.from_numpy(
                        np.repeat(arrays["layer_89_input"][0:1], 9, axis=0)
                    ),
                    cache_position_start=0,
                )
            except KimiCudaError as exc:
                rejection = str(exc)
            executor.runtime.synchronize()
            guard_after = executor.lifecycle_snapshot()
            guard_pass = (
                "rejected before CUDA work" in rejection
                and guard_before["execute_count"] == guard_after["execute_count"]
                and guard_before["batch_execute_count"] == guard_after["batch_execute_count"]
                and records_before == len(executor.execution_records)
                and executor.runtime.error_state_ok()
            )
            safe_id = "experiment-015-canary-post-guard-safe"
            executor.open_session(safe_id, maximum_context_override=3)
            try:
                safe_result = executor.execute_decode(
                    session_id=safe_id,
                    hidden_states=torch.from_numpy(arrays["layer_89_input"][0:1]),
                    cache_position_start=0,
                )
                safe_record = executor.execution_records[-1]
                safe_metrics = _numerical_metrics(
                    arrays["layer_89_expected_boundary"][0:1],
                    safe_result.stage_boundary_hidden_states.detach().cpu().numpy(),
                )
                safe_pass = (
                    float(safe_metrics["relative_l2_error"]) <= RELATIVE_L2_GATE
                    and list(safe_record["selected_expert_ids"])
                    == arrays["layer_89_expected_routes"][0].tolist()
                    and executor.runtime.error_state_ok()
                )
            finally:
                executor.close_session(safe_id)
            role_result["production_batch_8"] = {
                "pass": batch_pass,
                "rows": 8,
                "device_ms": float(batch_record["device_ms"]),
                "rows_per_second": 8000.0 / float(batch_record["device_ms"]),
                "maximum_relative_l2_error": max(
                    float(row["relative_l2_error"]) for row in batch_comparisons
                ),
                "phase_device_ms": batch_record["phase_device_ms"],
                "state_isolation": len({row["fingerprint"] for row in batch_states}) == 1,
                "lifecycle_delta": _lifecycle_delta(batch_before, batch_after),
            }
            role_result["over_limit_guard"] = {
                "pass": guard_pass,
                "requested_batch": 9,
                "rejection": rejection,
                "lifecycle_before": guard_before,
                "lifecycle_after": guard_after,
            }
            role_result["post_guard_safe_fixture"] = {
                "pass": safe_pass,
                "metrics": safe_metrics,
            }
        executor.runtime.synchronize()
        role_result["cuda_error_state_clear"] = executor.runtime.error_state_ok()
        role_result["pass"] = (
            role_result["prepare_exactly_seven_calls"]
            and role_result["correctness_pass"]
            and role_result["stateful_decode_pass"]
            and role_result["warm_lifecycle_deltas_zero"]
            and role_result["cpu_mathematical_fallbacks"] == 0
            and role_result["cuda_error_state_clear"]
            and (
                layer != 89
                or (
                    role_result["production_batch_8"]["pass"]
                    and role_result["over_limit_guard"]["pass"]
                    and role_result["post_guard_safe_fixture"]["pass"]
                )
            )
        )
        return role_result
    finally:
        await runtime.close()


async def _run_assigned_stage(
    *,
    worker: dict[str, Any],
    snapshot: Path,
    runtime_path: Path,
    runtime_sha256: str,
    placement: dict[str, Any],
) -> dict[str, Any]:
    """Run a bounded liveness/state/guard canary for one exact assignment."""

    layers = [int(value) for value in worker.get("owned_layers", [])]
    if len(layers) != 1:
        raise PhysicalCanaryError("assigned-stage canary requires exactly one owned layer")
    layer = layers[0]
    identity_path = snapshot / "model-identity.json"
    identity = _read(identity_path)
    if (
        identity.get("worker_id") != worker["worker_id"]
        or identity.get("assignment_sha256") != worker["assignment_sha256"]
        or identity.get("model_content_fingerprint")
        != placement["checkpoint"]["checkpoint_fingerprint"]
    ):
        raise PhysicalCanaryError("assigned-stage snapshot identity differs")
    assignment = _placement_assignment(worker, layer=layer)
    runtime = PersistentStageRuntime(
        worker_id=worker["worker_id"],
        device="native-cuda:0",
        dtype="float32",
        memory_limit_bytes=int(worker["memory"]["physical_vram_bytes"]),
        maximum_sessions=16,
        configured_model_path=snapshot,
        configured_model_identity_path=identity_path,
        allow_model_download=False,
    )
    request = LoadStageRequest(
        worker_id=worker["worker_id"],
        request_id=f"experiment-015-assigned-canary-{worker['worker_id']}-load",
        model_id="moonshotai/Kimi-K3",
        model_revision=str(placement["checkpoint"]["revision"]),
        tokenizer_revision=str(identity["tokenizer_revision"]),
        topology_id=f"experiment-015-assigned-canary-{worker['worker_id']}",
        route_generation=1,
        stage_count=93,
        assignment=assignment,
        adapter_id="kimi_k3_cuda",
        fast_path_id="colibri-kimi-k3-cuda",
        fast_path_mode="resident",
        fast_path_batch_bucket=8,
        fast_path_context_bucket=8192,
        model_content_fingerprint=str(placement["checkpoint"]["checkpoint_fingerprint"]),
        native_runtime_library=str(runtime_path),
        native_runtime_library_sha256=runtime_sha256,
        device="native-cuda:0",
        dtype="float32",
        model_path=str(snapshot),
        allow_download=False,
        deadline_unix_ns=time.time_ns() + 3_600_000_000_000,
    )
    try:
        response = await runtime.load_stage(request)
        executor = runtime.loaded_executor
        if not response.accepted or not isinstance(executor, PersistentKimiStageExecutor):
            raise PhysicalCanaryError("assigned-stage product load was not accepted")
        loaded = executor.lifecycle_snapshot()
        prepare = loaded.get("prepare_warmup") or {}
        fixture = prepare.get("stage_fixture") or {}
        prepare_pass = (
            loaded.get("prepare_warmup_count") == 1
            and prepare.get("count") == 1
            and fixture.get("iterations") == 7
            and fixture.get("active_sessions_after") == 0
            and fixture.get("serving_execute_count_restored") is True
            and fixture.get("research_records_removed") is True
            and fixture.get("output_finite") is True
            and fixture.get("cpu_mathematical_fallbacks") == 0
        )
        hidden = torch.linspace(-0.01, 0.01, 9 * 7168, dtype=torch.float32).reshape(
            1, 9, 7168
        )
        tokens = torch.tensor([[163584], [18699]], dtype=torch.int64)
        session_id = f"experiment-015-assigned-canary-{worker['worker_id']}"
        executor.open_session(session_id, maximum_context_override=3)
        before = executor.lifecycle_snapshot()
        executions: list[dict[str, Any]] = []
        try:
            for position in range(2):
                result = (
                    executor.execute_prefill(
                        session_id=session_id,
                        token_ids=tokens[position : position + 1],
                        cache_position_start=position,
                    )
                    if layer == 0
                    else executor.execute_decode(
                        session_id=session_id,
                        hidden_states=hidden,
                        cache_position_start=position,
                    )
                )
                record = executor.execution_records[-1]
                boundary = result.stage_boundary_hidden_states
                selected = [int(value) for value in record["selected_expert_ids"]]
                expected_route_count = 0 if layer == 0 else 16
                executions.append(
                    {
                        "position": position,
                        "output_shape": list(boundary.shape),
                        "output_finite": bool(torch.isfinite(boundary).all().item()),
                        "selected_expert_count": len(selected),
                        "selected_experts_unique": len(selected) == len(set(selected)),
                        "routed_expert_execution_count": int(
                            record["routed_expert_execution_count"]
                        ),
                        "all_selected_experts_executed_once": bool(
                            record["all_selected_experts_executed_once"]
                        ),
                        "route_count_pass": len(selected) == expected_route_count,
                        "device_ms": float(record["device_ms"]),
                    }
                )
            state = executor.session_state_evidence(session_id)
            after = executor.lifecycle_snapshot()
        finally:
            executor.close_session(session_id)
        lifecycle = _lifecycle_delta(before, after)

        guard = {
            "required": layer != 0,
            "pass": layer == 0,
            "requested_batch": 9 if layer != 0 else None,
            "rejection": None,
        }
        post_guard_safe = True
        if layer != 0:
            guard_ids = tuple(f"assigned-guard-{row}" for row in range(9))
            for guard_id in guard_ids:
                executor.open_session(guard_id, maximum_context_override=2)
            guard_before = executor.lifecycle_snapshot()
            records_before = len(executor.execution_records)
            rejection = ""
            try:
                executor.execute_decode_batch(
                    session_ids=guard_ids,
                    hidden_states=hidden.repeat(9, 1, 1),
                    cache_position_start=0,
                )
            except KimiCudaError as exc:
                rejection = str(exc)
            finally:
                for guard_id in guard_ids:
                    executor.close_session(guard_id)
            executor.runtime.synchronize()
            guard_after = executor.lifecycle_snapshot()
            guard = {
                "required": True,
                "pass": (
                    "rejected before CUDA work" in rejection
                    and guard_before["execute_count"] == guard_after["execute_count"]
                    and guard_before["batch_execute_count"]
                    == guard_after["batch_execute_count"]
                    and records_before == len(executor.execution_records)
                    and executor.runtime.error_state_ok()
                ),
                "requested_batch": 9,
                "rejection": rejection,
            }
            safe_id = "assigned-post-guard-safe"
            executor.open_session(safe_id, maximum_context_override=2)
            try:
                safe = executor.execute_decode(
                    session_id=safe_id,
                    hidden_states=hidden,
                    cache_position_start=0,
                )
                post_guard_safe = bool(
                    torch.isfinite(safe.stage_boundary_hidden_states).all().item()
                    and executor.runtime.error_state_ok()
                )
            finally:
                executor.close_session(safe_id)
        executor.runtime.synchronize()
        gates = {
            "product_load_accepted": True,
            "prepare_exactly_seven_calls": prepare_pass,
            "two_stateful_executions": len(executions) == 2,
            "outputs_finite": all(row["output_finite"] for row in executions),
            "route_cardinality_exact": all(row["route_count_pass"] for row in executions),
            "route_ids_unique": all(row["selected_experts_unique"] for row in executions),
            "selected_experts_executed_once": all(
                row["all_selected_experts_executed_once"] for row in executions
            ),
            "state_sequence_length_two": int(state["cache_sequence_length"]) == 2,
            "state_finite": bool(state["finite"]),
            "warm_lifecycle_deltas_zero": all(value == 0 for value in lifecycle.values()),
            "cpu_mathematical_fallbacks_zero": int(
                loaded["cpu_mathematical_fallbacks"]
            )
            == 0,
            "over_limit_batch_rejected_before_cuda": bool(guard["pass"]),
            "post_guard_safe_fixture": post_guard_safe,
            "cuda_error_state_clear": executor.runtime.error_state_ok(),
        }
        return {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "worker_id": worker["worker_id"],
            "layer": layer,
            "worker_role": worker["worker_role"],
            "assignment_sha256": worker["assignment_sha256"],
            "snapshot_identity_sha256": _sha256(identity_path),
            "prepare": prepare,
            "executions": executions,
            "state": state,
            "lifecycle_delta": lifecycle,
            "guard": guard,
            "acceptance_gates": gates,
        }
    finally:
        await runtime.close()


def run_assigned_stage_canary(
    placement_path: Path,
    native_source_manifest_path: Path,
    runtime_certificate_path: Path,
    runtime_path: Path,
    snapshot_path: Path,
    worker_id: str,
    output_path: Path,
) -> dict[str, Any]:
    """Qualify one worker's exact real stage before it may register."""

    placement_source = placement_path.resolve()
    runtime_source = runtime_path.resolve()
    snapshot = snapshot_path.resolve()
    if not runtime_source.is_file() or not snapshot.is_dir():
        raise PhysicalCanaryError("assigned-stage runtime or snapshot is absent")
    identity = validate_linux_runtime_certificate(
        runtime_certificate_path,
        native_source_manifest_path,
        placement_source,
    )
    if _sha256(runtime_source) != identity["sha256"]:
        raise PhysicalCanaryError("assigned-stage runtime differs from physical certificate")
    placement = _read(placement_source)
    worker = next(
        (row for row in placement["workers"] if row["worker_id"] == worker_id),
        None,
    )
    if worker is None:
        raise PhysicalCanaryError(f"placement has no worker {worker_id!r}")
    hardware_before = inspect_local_gpu(runtime_source)
    if (
        hardware_before.get("hardware_preflight_pass") is not True
        or hardware_before.get("cuda_initialized") is not True
        or hardware_before.get("cuda_error_state_ok") is not True
    ):
        raise PhysicalCanaryError("assigned-stage RTX 3090/CUDA preflight failed")
    result = asyncio.run(
        _run_assigned_stage(
            worker=worker,
            snapshot=snapshot,
            runtime_path=runtime_source,
            runtime_sha256=identity["sha256"],
            placement=placement,
        )
    )
    hardware_after = inspect_local_gpu(runtime_source)
    gates = {
        "physical_certificate_valid": True,
        "runtime_sha256_exact": True,
        "exact_rtx_3090": hardware_after.get("hardware_preflight_pass") is True,
        "cuda_initialized": hardware_after.get("cuda_initialized") is True,
        "cuda_error_state_clear": hardware_after.get("cuda_error_state_ok") is True,
        "assigned_stage_pass": result["status"] == "PASS",
        "nvidia_smi_healthy_after": hardware_after.get("nvidia_smi_status") == "MEASURED",
    }
    receipt = {
        "schema_version": ASSIGNED_STAGE_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "worker_id": worker_id,
        "runtime": {
            "path": str(runtime_source),
            "sha256": identity["sha256"],
            "certificate_sha256": identity["certificate_sha256"],
            "source_manifest_sha256": identity["source_manifest_sha256"],
        },
        "placement_sha256": _sha256(placement_source),
        "hardware_before": hardware_before,
        "assigned_stage": result,
        "hardware_after": hardware_after,
        "acceptance_gates": gates,
    }
    _atomic_json(output_path, receipt)
    return receipt


def _inspect_elf(runtime_path: Path) -> dict[str, Any]:
    magic = runtime_path.read_bytes()[:4] == b"\x7fELF"
    outputs: dict[str, Any] = {"elf_magic": magic}
    commands = {
        "sass": ["cuobjdump", "--list-elf", str(runtime_path)],
        "ptx": ["cuobjdump", "--dump-ptx", str(runtime_path)],
    }
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        outputs[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-4000:],
        }
    outputs["sm86_sass_present"] = (
        outputs["sass"]["returncode"] == 0 and "sm_86" in outputs["sass"]["stdout"]
    )
    ptx_text = outputs["ptx"]["stdout"]
    outputs["compute86_ptx_present"] = (
        outputs["ptx"]["returncode"] == 0
        and (".target sm_86" in ptx_text or "compute_86" in ptx_text)
    )
    return outputs


def _verify_reference_evidence(manifest: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    references = manifest.get("reference_evidence")
    if not isinstance(references, dict) or sorted(references) != [
        "full_graph",
        "operation_matrix",
        "sm86",
    ]:
        raise PhysicalCanaryError("physical canary reference set differs")
    for name, expected in references.items():
        path = evidence_root.resolve() / str(expected["file"])
        document = _read(path)
        passed = (
            _sha256(path) == expected["sha256"]
            and expected.get("status") == "PASS"
            and document.get("status") == "PASS"
        )
        results[name] = {"path": str(path), "sha256": _sha256(path), "pass": passed}
    return results


async def _run_physical_canary_async(
    placement_path: Path,
    source_manifest_path: Path,
    runtime_path: Path,
    fixture_npz: Path,
    fixture_manifest_path: Path,
    evidence_root: Path,
    snapshots: dict[str, Path],
    receipt_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    started = time.time_ns()
    placement_path = placement_path.resolve()
    source_manifest_path = source_manifest_path.resolve()
    runtime_path = runtime_path.resolve()
    placement = _read(placement_path)
    source_manifest = validate_native_source_manifest(
        source_manifest_path,
        placement_path,
        source_directory=source_manifest_path.parent / "src",
    )
    fixture_manifest, arrays = _load_fixture_bundle(
        fixture_npz.resolve(), fixture_manifest_path.resolve()
    )
    if fixture_manifest.get("placement_sha256") != _sha256(placement_path):
        raise PhysicalCanaryError("physical canary fixture targets another placement")
    reference_evidence = _verify_reference_evidence(fixture_manifest, evidence_root)
    if sorted(snapshots) != sorted(CANARY_WORKERS):
        raise PhysicalCanaryError("physical canary requires exactly four named snapshots")
    for role, snapshot in snapshots.items():
        if not snapshot.resolve().is_dir():
            raise PhysicalCanaryError(f"physical canary snapshot is absent: {role}")
    if not runtime_path.is_file():
        raise PhysicalCanaryError("physical canary Linux runtime is absent")

    hardware_only = inspect_local_gpu(None)
    runtime_probe = inspect_local_gpu(runtime_path)
    elf = _inspect_elf(runtime_path)
    runtime_sha = _sha256(runtime_path)
    roles: dict[str, Any] = {}
    workers_by_id = {row["worker_id"]: row for row in placement["workers"]}
    for role, layer in CANARY_WORKERS.items():
        worker_id = f"k3-worker-{layer:03d}"
        roles[role] = await _run_role(
            role=role,
            layer=layer,
            worker=workers_by_id[worker_id],
            snapshot=snapshots[role].resolve(),
            runtime_path=runtime_path,
            runtime_sha256=runtime_sha,
            placement=placement,
            arrays=arrays,
        )
    post_health = inspect_local_gpu(None)
    successful_roles = {name for name, row in roles.items() if row.get("pass") is True}
    observed_operations = sorted(
        set().union(*(ROLE_OPERATIONS[name] for name in successful_roles))
        if successful_roles
        else set()
    )
    all_roles_pass = successful_roles == set(KIMI_STAGE_ROLES)
    all_references_pass = all(row["pass"] for row in reference_evidence.values())
    gates = {
        "exact_rtx_3090": "RTX 3090" in str(hardware_only.get("gpu_name", "")),
        "compute_capability_sm86": hardware_only.get("compute_capability") == "8.6",
        "vram_at_least_24_gib": int(hardware_only.get("vram_total_bytes", 0)) >= 24 * 1024**3,
        "linux_elf_x86_64": (
            sys.platform.startswith("linux")
            and platform.machine().lower() in {"x86_64", "amd64"}
            and elf["elf_magic"]
        ),
        "binary_hash_exact": runtime_probe.get("runtime_sha256") == runtime_sha,
        "sm86_sass_present": bool(elf["sm86_sass_present"]),
        "compute86_ptx_present": bool(elf["compute86_ptx_present"]),
        "all_11_kimi_operation_classes_pass": observed_operations
        == sorted(KIMI_OPERATION_CLASSES),
        "all_four_stage_roles_pass": all_roles_pass,
        "full_graph_correctness_reference_pass": all_references_pass and all_roles_pass,
        "stateful_decode_pass": all(
            row.get("stateful_decode_pass") is True for row in roles.values()
        ),
        "production_batch_8_pass": roles["kda_moe"].get("production_batch_8", {}).get("pass")
        is True,
        "over_limit_batch_rejected_before_cuda": roles["kda_moe"].get(
            "over_limit_guard", {}
        ).get("pass")
        is True,
        "prepare_exactly_seven_calls": all(
            row.get("prepare_exactly_seven_calls") is True for row in roles.values()
        ),
        "warm_lifecycle_deltas_zero": all(
            row.get("warm_lifecycle_deltas_zero") is True for row in roles.values()
        ),
        "cuda_error_state_clear": (
            runtime_probe.get("cuda_error_state_ok") is True
            and all(row.get("cuda_error_state_clear") is True for row in roles.values())
        ),
        "post_canary_safe_fixture_pass": roles["kda_moe"].get(
            "post_guard_safe_fixture", {}
        ).get("pass")
        is True,
        "nvidia_smi_healthy_after": post_health.get("nvidia_smi_status") == "MEASURED",
    }
    if sorted(gates) != sorted(PHYSICAL_CERTIFICATE_GATES):
        raise PhysicalCanaryError("physical canary internal gate set differs from the schema")
    status = "PASS" if all(gates.values()) else "FAIL"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "evidence_kind": "physical_3090",
        "started_unix_ns": started,
        "completed_unix_ns": time.time_ns(),
        "hardware_preflight_before_cuda": hardware_only,
        "runtime_probe": runtime_probe,
        "elf_inspection": elf,
        "roles": roles,
        "observed_operation_classes": observed_operations,
        "reference_evidence": reference_evidence,
        "post_canary_health": post_health,
        "acceptance_gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "certificate_emitted": False,
        "inspection": (
            "The full-graph gate binds retained exact serial-oracle evidence to four "
            "physical checkpoint-aligned role executions; it does not claim that all 93 "
            "layers were physically resident together on one RTX 3090."
        ),
    }
    if status == "PASS":
        certificate = {
            "schema_version": CERTIFICATE_SCHEMA,
            "status": "PASS",
            "evidence_kind": "physical_3090",
            "platform": "linux-x86_64",
            "references": {
                "placement_sha256": _sha256(placement_path),
                "source_manifest_sha256": _sha256(source_manifest_path),
                "source_bundle_sha256": source_manifest["source_bundle_sha256"],
                "windows_reference_sha256": source_manifest["windows_reference_sha256"],
            },
            "build_arguments": list(BUILD_ARGUMENTS),
            "binary": {
                "worker_local_path": str(runtime_path),
                "sha256": runtime_sha,
                "bytes": runtime_path.stat().st_size,
                "elf_magic": True,
            },
            "hardware": {
                "gpu_name": hardware_only["gpu_name"],
                "gpu_uuid": hardware_only["gpu_uuid"],
                "compute_capability": hardware_only["compute_capability"],
                "vram_total_bytes": hardware_only["vram_total_bytes"],
                "driver_version": hardware_only["driver_version"],
            },
            "operation_classes": list(KIMI_OPERATION_CLASSES),
            "stage_roles": list(KIMI_STAGE_ROLES),
            "acceptance_gates": gates,
            "canary_receipt_path": str(receipt_path.resolve()),
        }
        _atomic_json(certificate_path, certificate)
        validated = validate_linux_runtime_certificate(
            certificate_path.resolve(), source_manifest_path, placement_path
        )
        receipt["certificate_emitted"] = True
        receipt["certificate"] = {
            "path": str(certificate_path.resolve()),
            "sha256": _sha256(certificate_path.resolve()),
            "validated_runtime_identity": validated,
        }
    else:
        certificate_path.resolve().unlink(missing_ok=True)
    _atomic_json(receipt_path, receipt)
    return receipt


def run_physical_canary(
    placement_path: Path,
    source_manifest_path: Path,
    runtime_path: Path,
    fixture_npz: Path,
    fixture_manifest_path: Path,
    evidence_root: Path,
    snapshots: dict[str, Path],
    receipt_path: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    """Run the bounded physical canary and emit a certificate only on PASS."""

    return asyncio.run(
        _run_physical_canary_async(
            placement_path,
            source_manifest_path,
            runtime_path,
            fixture_npz,
            fixture_manifest_path,
            evidence_root,
            snapshots,
            receipt_path,
            certificate_path,
        )
    )


__all__ = [
    "ASSIGNED_STAGE_SCHEMA",
    "CANARY_WORKERS",
    "FIXTURE_SCHEMA",
    "PhysicalCanaryError",
    "build_physical_canary_fixtures",
    "run_assigned_stage_canary",
    "run_physical_canary",
]
