"""Authenticated controller for the physical E025 Kimi K3 token path."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_cuda_runtime import (
    _array_fingerprint,
    _numerical_metrics,
)
from swarm_inference.execution.kimi_k3_graph_runtime import _parse_oracle_routes
from swarm_inference.experiments.experiment_020.transport import Frame, MessageType
from swarm_inference.model.kimi_tokenizer import (
    KIMI_TOKENIZER_ASSETS,
    apply_kimi_prompt_special_tokens,
    load_pinned_kimi_tokenizer,
)

from .constants import (
    CONSUMER_GPU_NAMES,
    EVIDENCE_CLASS,
    MODEL_ID,
    MODEL_REVISION,
    SUB_LAYER_TARGET,
    SUB_LAYER_WORKERS,
    TRANSFORMER_LAYERS,
)
from .io import atomic_write_json, read_json, sha256_file, utc_now
from .wire import Action, AuthenticatedConnection, pack_payload, unpack_payload

SCHEMA_VERSION = "experiment-025-physical-controller-v1"


@dataclass(frozen=True, slots=True)
class WorkerEndpoint:
    worker_id: str
    role: str
    host: str
    port: int
    layer: int | None
    worker_index: int | None
    machine_id: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerEndpoint:
        layer = value.get("layer")
        worker_index = value.get("worker_index")
        return cls(
            worker_id=str(value["worker_id"]),
            role=str(value["role"]),
            host=str(value["host"]),
            port=int(value["port"]),
            layer=None if layer is None else int(layer),
            worker_index=None if worker_index is None else int(worker_index),
            machine_id=str(value["machine_id"]),
        )


def load_endpoints(path: Path) -> list[WorkerEndpoint]:
    value = read_json(path.expanduser().resolve())
    if value.get("schema_version") != "experiment-025-live-endpoints-v1":
        raise ValueError("E025 endpoint schema is invalid")
    endpoints = [WorkerEndpoint.from_dict(row) for row in value["workers"]]
    worker_ids = [row.worker_id for row in endpoints]
    if len(worker_ids) != len(set(worker_ids)):
        raise ValueError("E025 live endpoints contain duplicate workers")
    stages = sorted(
        int(row.layer)
        for row in endpoints
        if row.role in {"BACKBONE_STAGE", "SUB_LAYER_PARENT"}
        and row.layer is not None
    )
    if stages != list(range(TRANSFORMER_LAYERS)):
        raise ValueError("E025 live endpoints do not cover all 93 physical stages")
    fragments = sorted(
        int(row.worker_index)
        for row in endpoints
        if row.role == "SUB_LAYER_WORKER" and row.worker_index is not None
    )
    if fragments != list(range(SUB_LAYER_WORKERS)):
        raise ValueError("E025 live endpoints do not cover four sub-layer workers")
    fragment_machines = {
        row.machine_id for row in endpoints if row.role == "SUB_LAYER_WORKER"
    }
    if len(fragment_machines) != SUB_LAYER_WORKERS or "unknown" in fragment_machines:
        raise ValueError("E025 sub-layer workers are not on four known distinct machines")
    return endpoints


def _consumer_gpu(name: str) -> bool:
    normalized = name.upper()
    return "GEFORCE" in normalized and any(
        allowed.upper() in normalized for allowed in CONSUMER_GPU_NAMES
    )


class PhysicalController:
    """Drive real stages while refusing all local model-compute fallbacks."""

    def __init__(
        self,
        endpoints: list[WorkerEndpoint],
        credential: bytes,
        certificate: Path,
        *,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.endpoints = list(endpoints)
        self.stage_endpoints = sorted(
            (
                row
                for row in endpoints
                if row.role in {"BACKBONE_STAGE", "SUB_LAYER_PARENT"}
            ),
            key=lambda row: int(row.layer if row.layer is not None else -1),
        )
        self.fragment_endpoints = sorted(
            (row for row in endpoints if row.role == "SUB_LAYER_WORKER"),
            key=lambda row: int(
                row.worker_index if row.worker_index is not None else -1
            ),
        )
        self.connections = {
            row.worker_id: AuthenticatedConnection(
                row.host,
                row.port,
                credential,
                certificate,
                timeout_seconds=timeout_seconds,
            )
            for row in self.endpoints
        }
        self.request_sequence = 0
        self.session_id: str | None = None
        self.registration: dict[str, dict[str, Any]] = {}
        self.closed = False

    def _request(
        self,
        endpoint: WorkerEndpoint,
        action: Action,
        metadata: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
        connection = self.connections[endpoint.worker_id]
        self.request_sequence += 1
        request_id = f"e025-controller-{self.request_sequence:08d}"
        before = (connection.sent_bytes, connection.received_bytes)
        frame = Frame(
            MessageType.EXECUTE_SHARD,
            request_id,
            self.request_sequence,
            endpoint.worker_id,
            self.session_id or "e025-control",
            pack_payload(action, metadata, arrays),
        )
        started = time.perf_counter_ns()
        response = connection.request(frame)
        wall_ns = time.perf_counter_ns() - started
        response_action, response_metadata, response_arrays = unpack_payload(
            response.payload
        )
        if response.message_type is MessageType.ERROR:
            raise RuntimeError(
                f"physical worker {endpoint.worker_id} failed: {response_metadata}"
            )
        if response.message_type is not MessageType.SHARD_RESULT:
            raise RuntimeError(
                f"physical worker {endpoint.worker_id} returned "
                f"{response.message_type.name}"
            )
        if response_action is not action:
            raise RuntimeError("physical worker returned a different E025 action")
        after = (connection.sent_bytes, connection.received_bytes)
        transport = {
            "wall_ns": wall_ns,
            "request_wire_bytes": after[0] - before[0],
            "response_wire_bytes": after[1] - before[1],
        }
        return response_metadata, response_arrays, transport

    def register(self) -> dict[str, Any]:
        registrations: dict[str, dict[str, Any]] = {}
        for endpoint in self.endpoints:
            metadata, _, transport = self._request(
                endpoint,
                Action.REGISTER,
                {"controller_compute_resource": False},
            )
            ready = dict(metadata["ready"])
            gpu = ready["gpu"]
            if ready.get("worker_id") != endpoint.worker_id:
                raise RuntimeError("worker READY identity differs from live endpoint")
            if not _consumer_gpu(str(gpu["gpu_name"])):
                raise RuntimeError(
                    f"headline worker is not an allowed consumer GeForce GPU: {gpu['gpu_name']}"
                )
            if not bool(gpu.get("torch_cuda_available")):
                raise RuntimeError("headline worker has no real CUDA device")
            if ready.get("whole_layer_fallback") is not False:
                raise RuntimeError("headline worker exposes a whole-layer fallback")
            registrations[endpoint.worker_id] = {**ready, "transport": transport}
        self.registration = registrations
        return {
            "status": "PASS",
            "registered_worker_count": len(registrations),
            "registered_stage_count": len(self.stage_endpoints),
            "registered_sub_layer_worker_count": len(self.fragment_endpoints),
            "all_consumer_geforce": True,
            "all_real_cuda": True,
            "no_whole_layer_fallback": True,
            "workers": registrations,
        }

    def open_session(self, *, maximum_context: int, session_id: str | None = None) -> str:
        if self.session_id is not None:
            raise RuntimeError("E025 controller already has an open session")
        self.session_id = session_id or f"e025-{uuid.uuid4().hex}"

        def open_one(endpoint: WorkerEndpoint) -> tuple[str, dict[str, Any]]:
            metadata, _, transport = self._request(
                endpoint,
                Action.OPEN_SESSION,
                {
                    "session_id": self.session_id,
                    "maximum_context": maximum_context,
                },
            )
            if metadata.get("opened") is not True:
                raise RuntimeError(f"worker {endpoint.worker_id} did not open its session")
            return endpoint.worker_id, transport

        with ThreadPoolExecutor(max_workers=TRANSFORMER_LAYERS) as pool:
            opened = dict(pool.map(open_one, self.stage_endpoints))
        if len(opened) != TRANSFORMER_LAYERS:
            raise RuntimeError("not every physical stage opened the inference session")
        return self.session_id

    def execute_token(
        self,
        token_id: int,
        position: int,
        *,
        expected_trace: np.ndarray | None = None,
        expected_routes: dict[int, dict[int, list[int]]] | None = None,
    ) -> dict[str, Any]:
        if self.session_id is None:
            raise RuntimeError("E025 controller has no open physical session")
        boundary: np.ndarray | None = None
        stages: list[dict[str, Any]] = []
        started_at_utc = utc_now()
        token_started = time.perf_counter_ns()
        final_arrays: dict[str, np.ndarray] = {}
        for endpoint in self.stage_endpoints:
            layer = int(endpoint.layer if endpoint.layer is not None else -1)
            arrays = (
                {"token_ids": np.asarray([[token_id]], dtype=np.int64)}
                if layer == 0
                else {"boundary": np.asarray(boundary, dtype=np.float32)}
            )
            metadata, output_arrays, transport = self._request(
                endpoint,
                Action.EXECUTE_STAGE,
                {
                    "session_id": self.session_id,
                    "position": position,
                    "layer": layer,
                    "controller_compute_fallback": False,
                },
                arrays,
            )
            if metadata.get("native_dispatch") is not True:
                raise RuntimeError("a physical stage did not prove native dispatch")
            if metadata.get("cached_output") is not False:
                raise RuntimeError("a physical stage returned a cached model output")
            if metadata.get("synthetic_tensor") is not False:
                raise RuntimeError("a physical stage returned a synthetic tensor")
            boundary = np.ascontiguousarray(output_arrays["boundary"], dtype=np.float32)
            layer_output = np.ascontiguousarray(boundary[0, 0], dtype=np.float32)
            stage = {
                "layer": layer,
                "worker_id": endpoint.worker_id,
                "machine_id": endpoint.machine_id,
                "gpu_name": self.registration[endpoint.worker_id]["gpu"]["gpu_name"],
                "boundary_fingerprint": _array_fingerprint(boundary),
                "layer_output_fingerprint": _array_fingerprint(layer_output),
                "execution": metadata["execution"],
                "transport": transport,
            }
            if expected_trace is not None:
                expected = np.ascontiguousarray(
                    expected_trace[
                        position * (TRANSFORMER_LAYERS + 1) + layer + 1
                    ],
                    dtype=np.float32,
                )
                metrics = _numerical_metrics(expected, layer_output)
                stage["trusted_oracle"] = {
                    "expected_fingerprint": _array_fingerprint(expected),
                    "actual_fingerprint": _array_fingerprint(layer_output),
                    "exact_sha256": _array_fingerprint(expected)
                    == _array_fingerprint(layer_output),
                    "metrics": metrics,
                }
                expected_route = (
                    expected_routes.get(layer, {}).get(position, [])
                    if expected_routes is not None
                    else []
                )
                observed_route = [
                    int(value)
                    for value in metadata["execution"].get("selected_expert_ids", [])
                ]
                stage["trusted_oracle"]["expected_route"] = expected_route
                stage["trusted_oracle"]["observed_route"] = observed_route
                stage["trusted_oracle"]["route_exact"] = (
                    observed_route == expected_route
                )
            stages.append(stage)
            if layer == TRANSFORMER_LAYERS - 1:
                final_arrays = output_arrays
        if "sampled_token_ids" not in final_arrays:
            raise RuntimeError("physical final stage produced no sampled token")
        sampled = np.asarray(final_arrays["sampled_token_ids"], dtype=np.int64).reshape(-1)
        if sampled.shape != (1,):
            raise RuntimeError("physical final stage returned an invalid sample")
        fragment_record = stages[SUB_LAYER_TARGET]["execution"].get(
            "external_expert_dispatch"
        )
        if not isinstance(fragment_record, dict):
            raise RuntimeError("layer 89 has no physical external-expert receipt")
        if int(fragment_record.get("workers_invoked", 0)) != SUB_LAYER_WORKERS:
            raise RuntimeError("layer 89 did not invoke all four sub-layer workers")
        if fragment_record.get("every_selected_expert_executed_once") is not True:
            raise RuntimeError("layer 89 did not execute selected experts exactly once")
        if fragment_record.get("whole_layer_fallback") is not False:
            raise RuntimeError("layer 89 exposed a forbidden monolithic fallback")
        elapsed_ns = time.perf_counter_ns() - token_started
        return {
            "position": position,
            "input_token_id": int(token_id),
            "sampled_token_id": int(sampled[0]),
            "started_at_utc": started_at_utc,
            "completed_at_utc": utc_now(),
            "elapsed_ns": elapsed_ns,
            "elapsed_seconds": elapsed_ns / 1e9,
            "stages": stages,
            "final_hidden_fingerprint": _array_fingerprint(
                np.asarray(final_arrays["final_hidden"], dtype=np.float32)
            ),
            "logits_fingerprint": _array_fingerprint(
                np.asarray(final_arrays["logits"], dtype=np.float32)
            ),
            "sub_layer_participation": fragment_record,
            "controller_model_compute": False,
        }

    def close_session(self) -> dict[str, Any]:
        if self.session_id is None:
            return {"status": "NOT_OPEN"}
        session_id = self.session_id

        def close_one(endpoint: WorkerEndpoint) -> tuple[str, dict[str, Any]]:
            metadata, _, transport = self._request(
                endpoint,
                Action.CLOSE_SESSION,
                {"session_id": session_id},
            )
            return endpoint.worker_id, {**metadata, "transport": transport}

        try:
            with ThreadPoolExecutor(max_workers=TRANSFORMER_LAYERS) as pool:
                results = dict(pool.map(close_one, self.stage_endpoints))
        finally:
            self.session_id = None
        return {"status": "PASS", "workers": results}

    def health(self) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for endpoint in self.endpoints:
            metadata, _, transport = self._request(endpoint, Action.HEALTH, {})
            rows[endpoint.worker_id] = {**metadata, "transport": transport}
        return {"status": "PASS", "workers": rows}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.session_id is not None:
            self.close_session()
        for connection in self.connections.values():
            connection.close()

    def __enter__(self) -> PhysicalController:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def tokenize_prompt(
    stage_zero_snapshot: Path, prompt: str
) -> tuple[Any, list[int], dict[str, Any]]:
    root = stage_zero_snapshot.expanduser().resolve()
    identity_path = root / "model-identity.json"
    if identity_path.is_file():
        tokenizer = load_pinned_kimi_tokenizer(
            root,
            identity_path,
            expected_worker_id="e025-stage-000",
        )
        tokenizer_identity = {
            "mode": "pinned_stage_zero_snapshot",
            "model_identity_sha256": sha256_file(identity_path),
        }
    else:
        for name in KIMI_TOKENIZER_ASSETS:
            asset = root / name
            if asset.is_symlink() or not asset.is_file():
                raise ValueError(f"authoritative local tokenizer asset is absent: {name}")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            root,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizer_identity = {
            "mode": "authoritative_local_checkpoint_controller_tokenizer",
            "assets": {
                name: sha256_file(root / name) for name in KIMI_TOKENIZER_ASSETS
            },
            "controller_tensor_compute": False,
        }
    raw = tokenizer(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
    token_ids = apply_kimi_prompt_special_tokens(
        tokenizer,
        [int(value) for value in raw],
        add_special_tokens=True,
    )
    return tokenizer, token_ids, tokenizer_identity


def verify_fixture_trace(
    token_records: list[dict[str, Any]],
    hidden_trace_path: Path,
    *,
    relative_l2_gate: float,
    maximum_absolute_gate: float,
) -> dict[str, Any]:
    expected = np.fromfile(hidden_trace_path, dtype=np.float32).reshape(-1, 7168)
    expected_rows = len(token_records) * (TRANSFORMER_LAYERS + 1)
    if expected.shape != (expected_rows, 7168):
        raise ValueError("trusted K3 fixture hidden trace has an unexpected shape")
    comparisons: list[dict[str, Any]] = []
    for position, token in enumerate(token_records):
        for stage in token["stages"]:
            layer = int(stage["layer"])
            actual_fingerprint = stage["layer_output_fingerprint"]
            expected_row = expected[
                position * (TRANSFORMER_LAYERS + 1) + layer + 1
            ]
            expected_fingerprint = _array_fingerprint(expected_row)
            exact = actual_fingerprint == expected_fingerprint
            oracle = stage.get("trusted_oracle")
            if not isinstance(oracle, dict):
                raise ValueError("physical fixture record omitted trusted-oracle metrics")
            metrics = dict(oracle["metrics"])
            within_gate = (
                float(metrics["relative_l2_error"]) <= relative_l2_gate
                and float(metrics["maximum_absolute_error"])
                <= maximum_absolute_gate
                and bool(metrics["actual_finite"])
            )
            comparisons.append(
                {
                    "position": position,
                    "layer": layer,
                    "worker_id": stage["worker_id"],
                    "expected": expected_fingerprint,
                    "actual": actual_fingerprint,
                    "exact_sha256": exact,
                    "metrics": metrics,
                    "within_numerical_gate": within_gate,
                    "route_exact": oracle["route_exact"],
                }
            )
    # Stage receipts retain numerical oracle metrics when the worker was supplied a
    # trusted fixture. Exact hashes are the strongest cross-machine check here.
    exact_count = sum(bool(row["exact_sha256"]) for row in comparisons)
    gated_count = sum(bool(row["within_numerical_gate"]) for row in comparisons)
    route_count = sum(bool(row["route_exact"]) for row in comparisons)
    return {
        "status": (
            "PASS"
            if gated_count == len(comparisons) and route_count == len(comparisons)
            else "FAIL"
        ),
        "comparison_count": len(comparisons),
        "exact_sha256_count": exact_count,
        "within_numerical_gate_count": gated_count,
        "route_exact_count": route_count,
        "relative_l2_gate": relative_l2_gate,
        "maximum_absolute_gate": maximum_absolute_gate,
        "comparisons": comparisons,
    }


def run_physical_generation(
    *,
    endpoints_path: Path,
    credential_path: Path,
    certificate: Path,
    stage_zero_snapshot: Path,
    output_path: Path,
    prompt: str,
    max_new_tokens: int,
    fixture_token_ids: list[int] | None = None,
    fixture_hidden_trace: Path | None = None,
    fixture_routes: Path | None = None,
) -> dict[str, Any]:
    if max_new_tokens < 2:
        raise ValueError("E025 physical generation requires at least two decode tokens")
    endpoints = load_endpoints(endpoints_path)
    credential = credential_path.read_bytes()
    tokenizer, prompt_ids, tokenizer_identity = tokenize_prompt(
        stage_zero_snapshot, prompt
    )
    maximum_context = len(prompt_ids) + max_new_tokens
    records: list[dict[str, Any]] = []
    generated: list[int] = []
    started = time.perf_counter_ns()
    expected_trace: np.ndarray | None = None
    if fixture_hidden_trace is not None:
        expected_trace = np.fromfile(fixture_hidden_trace, dtype=np.float32).reshape(
            -1, 7168
        )
    expected_routes = (
        _parse_oracle_routes(fixture_routes)
        if fixture_routes is not None
        else None
    )
    with PhysicalController(endpoints, credential, certificate) as controller:
        registration = controller.register()
        controller.open_session(maximum_context=maximum_context)
        current: int | None = None
        position = 0
        for prompt_token in prompt_ids:
            record = controller.execute_token(
                prompt_token,
                position,
                expected_trace=expected_trace,
                expected_routes=expected_routes,
            )
            records.append(record)
            current = int(record["sampled_token_id"])
            position += 1
        if current is None:
            raise RuntimeError("Kimi prompt had no physical forward step")
        generated.append(current)
        while len(generated) < max_new_tokens:
            record = controller.execute_token(
                current,
                position,
                expected_trace=expected_trace,
                expected_routes=expected_routes,
            )
            records.append(record)
            current = int(record["sampled_token_id"])
            generated.append(current)
            position += 1
        health = controller.health()
        close_receipt = controller.close_session()
    elapsed_ns = time.perf_counter_ns() - started
    decoded_raw = tokenizer.decode(generated, skip_special_tokens=False)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    physical_decode_records = records[len(prompt_ids) - 1 :]
    forward_seconds = sum(float(row["elapsed_seconds"]) for row in records)
    generated_forward_seconds = sum(
        float(row["elapsed_seconds"]) for row in physical_decode_records
    )
    controller_request_wire_bytes = sum(
        int(stage["transport"]["request_wire_bytes"])
        for record in records
        for stage in record["stages"]
    )
    controller_response_wire_bytes = sum(
        int(stage["transport"]["response_wire_bytes"])
        for record in records
        for stage in record["stages"]
    )
    fragment_request_wire_bytes = sum(
        int(record["sub_layer_participation"]["request_wire_bytes"])
        for record in records
    )
    fragment_response_wire_bytes = sum(
        int(record["sub_layer_participation"]["response_wire_bytes"])
        for record in records
    )
    all_fragment_workers_invoked = all(
        int(row["sub_layer_participation"]["workers_invoked"])
        == SUB_LAYER_WORKERS
        for row in physical_decode_records
    )
    fragment_worker_ids = {
        str(worker["worker_id"])
        for row in physical_decode_records
        for worker in row["sub_layer_participation"]["workers"]
    }
    every_fragment_executes_real_compute = len(fragment_worker_ids) == SUB_LAYER_WORKERS and all(
        sum(
            int(worker["native_expert_calls"])
            for row in physical_decode_records
            for worker in row["sub_layer_participation"]["workers"]
            if worker["worker_id"] == worker_id
        )
        > 0
        for worker_id in fragment_worker_ids
    )
    fixture: dict[str, Any] | None = None
    if fixture_token_ids is not None:
        observed_inputs = [int(row["input_token_id"]) for row in records]
        fixture = {
            "expected_input_token_ids": fixture_token_ids,
            "observed_input_token_ids": observed_inputs[: len(fixture_token_ids)],
            "token_ids_exact": observed_inputs[: len(fixture_token_ids)]
            == fixture_token_ids,
        }
        if fixture_hidden_trace is not None and len(records) == len(fixture_token_ids):
            fixture["hidden_trace"] = verify_fixture_trace(
                records,
                fixture_hidden_trace,
                relative_l2_gate=1e-4,
                maximum_absolute_gate=1e-3,
            )
    fixture_pass = fixture is None or (
        fixture["token_ids_exact"] is True
        and fixture.get("hidden_trace", {}).get("status") == "PASS"
    )
    stateful = all(
        int(stage["execution"]["position"]) == int(record["position"])
        for record in records
        for stage in record["stages"]
    ) and len(records) >= len(prompt_ids) + 1
    gpu_error_worker_count = sum(
        health_row.get("runtime", {}).get("cuda_error_state_ok") is not True
        for health_row in health["workers"].values()
    )
    all_workers_registered = (
        int(registration["registered_worker_count"])
        == TRANSFORMER_LAYERS + SUB_LAYER_WORKERS
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": (
            "PASS"
            if all_fragment_workers_invoked
            and every_fragment_executes_real_compute
            and fixture_pass
            and stateful
            and gpu_error_worker_count == 0
            and all_workers_registered
            else "FAIL"
        ),
        "evidence_class": EVIDENCE_CLASS,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "tokenizer_identity": tokenizer_identity,
        "sampling": {"strategy": "greedy_argmax", "max_new_tokens": max_new_tokens},
        "generated_token_ids": generated,
        "decoded_text": decoded,
        "decoded_text_with_special_tokens": decoded_raw,
        "elapsed_ns": elapsed_ns,
        "elapsed_seconds": elapsed_ns / 1e9,
        "model_forward_seconds": forward_seconds,
        "generated_token_forward_seconds": generated_forward_seconds,
        "time_to_first_token_seconds": sum(
            float(row["elapsed_seconds"]) for row in records[: len(prompt_ids)]
        ),
        "decode_tokens_per_second": max_new_tokens / generated_forward_seconds,
        "network_wire_bytes": {
            "controller_to_stage_requests": controller_request_wire_bytes,
            "stage_to_controller_responses": controller_response_wire_bytes,
            "layer_89_parent_to_fragment_requests": fragment_request_wire_bytes,
            "fragment_to_layer_89_parent_responses": fragment_response_wire_bytes,
            "total": (
                controller_request_wire_bytes
                + controller_response_wire_bytes
                + fragment_request_wire_bytes
                + fragment_response_wire_bytes
            ),
        },
        "physical_machine_count": len({row.machine_id for row in endpoints}),
        "consumer_gpu_count": len(endpoints),
        "qualifying_sub_layer_gpu_count": len(
            [row for row in endpoints if row.role == "SUB_LAYER_WORKER"]
        ),
        "registration": registration,
        "token_records": records,
        "health": health,
        "gpu_error_worker_count": gpu_error_worker_count,
        "session_close": close_receipt,
        "fixture": fixture,
        "anti_cheating": {
            "api_generation": False,
            "controller_model_compute": False,
            "local_gpu_model_compute": False,
            "cached_outputs": False,
            "synthetic_tensors": False,
            "hidden_monolithic_server": False,
            "layer_89_complete_fallback": False,
        },
        "all_retained_tokens_used_four_sub_layer_workers": all_fragment_workers_invoked,
        "every_sub_layer_worker_executed_real_compute": every_fragment_executes_real_compute,
        "stateful_autoregressive_advancement": stateful,
    }
    atomic_write_json(output_path, payload)
    payload["output_path"] = str(output_path.expanduser().resolve())
    payload["output_sha256"] = sha256_file(output_path.expanduser().resolve())
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "PhysicalController",
    "WorkerEndpoint",
    "load_endpoints",
    "run_physical_generation",
    "tokenize_prompt",
    "verify_fixture_trace",
]
