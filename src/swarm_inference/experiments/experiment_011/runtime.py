"""Persistent direct-TCP contiguous-stage ring runtime."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import queue
import select
import socket
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from swarm_inference.experiments.experiment_010.schemas import NetworkShapeProfile
from swarm_inference.experiments.experiment_010.transport import NetworkShaper
from swarm_inference.experiments.experiment_011.model import (
    ContiguousOlmoeStage,
    StageExecutionResult,
)
from swarm_inference.experiments.experiment_011.partition import StageAssignment, StagePlan
from swarm_inference.experiments.experiment_011.protocol import (
    BufferPool,
    MessageSequenceValidator,
    Operation,
    SequenceAllocator,
    StageMessage,
    decode_message,
    encode_message,
    message_wire_identity,
    recv_message,
    send_message,
)
from swarm_inference.experiments.experiment_011.telemetry import (
    TraceContext,
    TraceWriter,
    merge_traces,
    reconstruct_critical_path,
)
from swarm_inference.experiments.experiment_011.tensor_transport import (
    AdaptiveTransportInputs,
    PackedTensor,
    pack_tensor,
    tensor_raw_bytes,
    unpack_tensor,
)

CompressionRequest = Literal["none", "byte_shuffle_fast_codec", "adaptive"]
COORDINATOR_STAGE = -1
CONTROL_SESSION = "__control__"


@dataclass(frozen=True, slots=True)
class FailureInjection:
    kind: str
    stage_id: int
    token_position: int


@dataclass(frozen=True, slots=True)
class StageWorkerConfiguration:
    run_id: str
    model_path: str
    model_revision: str
    tokenizer_revision: str
    topology_id: str
    assignment: StageAssignment
    assignments: tuple[StageAssignment, ...]
    network_profile: dict[str, Any]
    compression_request: CompressionRequest
    trace_path: str
    capture_directory: str | None
    control_endpoint: tuple[str, int]
    data_endpoint: tuple[str, int]
    ring_endpoints: tuple[tuple[str, int], ...]
    timeout_s: float
    publication_queue_size: int
    failure_injection: FailureInjection | None = None


@dataclass(slots=True)
class _WorkerSession:
    session_id: str
    request_id: str
    prompt_length: int
    generated_token_target: int
    generated_tokens: list[int] = field(default_factory=list)
    token_step_started_ns: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageRingResult:
    run_id: str
    session_id: str
    request_id: str
    topology_id: str
    profile_name: str
    compression_request: str
    generated_token_ids: tuple[int, ...]
    generated_tokens: int
    elapsed_seconds: float
    ttft_seconds: float
    inter_token_latencies_seconds: tuple[float, ...]
    throughput_tps: float
    stage_process_ids: tuple[int, ...]
    stage_endpoints: tuple[str, ...]
    ownership: tuple[dict[str, Any], ...]
    trace_paths: tuple[str, ...]
    critical_path: dict[str, Any]
    fallback_used: bool
    valid_for_claims: bool
    errors: tuple[str, ...]
    compression_modes_used: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["generated_token_ids"] = list(self.generated_token_ids)
        value["inter_token_latencies_seconds"] = list(self.inter_token_latencies_seconds)
        value["stage_process_ids"] = list(self.stage_process_ids)
        value["stage_endpoints"] = list(self.stage_endpoints)
        value["ownership"] = list(self.ownership)
        value["trace_paths"] = list(self.trace_paths)
        value["errors"] = list(self.errors)
        value["compression_modes_used"] = list(self.compression_modes_used)
        return value


def _make_listener() -> tuple[socket.socket, tuple[str, int]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    host, port = listener.getsockname()
    return listener, (str(host), int(port))


def _connect(endpoint: tuple[str, int], timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    error: OSError | None = None
    while time.monotonic() < deadline:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.settimeout(timeout_s)
        try:
            connection.connect(endpoint)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return connection
        except OSError as exc:
            error = exc
            connection.close()
            time.sleep(0.05)
    raise TimeoutError(f"could not connect to stage endpoint {endpoint}: {error}")


def _trace_fields(message: StageMessage) -> dict[str, Any]:
    tensor_metadata = message.attributes.get("tensor", {})
    return {
        "session_id": message.session_id,
        "request_id": message.request_id,
        "token_position": message.token_position,
        "stage_id": message.stage_id,
        "source_stage": message.source_stage,
        "destination_stage": message.destination_stage,
        "message_type": message.operation.name,
        "payload_bytes": len(message.payload),
        "tensor_shape": list(message.tensor_shape),
        "tensor_dtype": message.tensor_dtype,
        "compression_mode": message.compression_mode,
        "sequence_number": message.sequence_number,
        "model_revision": message.model_revision,
        "message_id": message_wire_identity(message),
        "tensor_raw_checksum": (
            tensor_metadata.get("raw_checksum") if isinstance(tensor_metadata, dict) else None
        ),
    }


def _control_message(
    *,
    operation: Operation,
    config: StageWorkerConfiguration,
    sequence: int,
    source: int,
    destination: int,
    session_id: str = CONTROL_SESSION,
    request_id: str = "control",
    token_position: int = -1,
    payload: bytes = b"",
    tensor_shape: tuple[int, ...] = (),
    tensor_dtype: str = "none",
    attributes: dict[str, Any] | None = None,
    status: str = "OK",
) -> StageMessage:
    assignment = config.assignments[destination if destination >= 0 else source]
    return StageMessage(
        operation=operation,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
        topology_id=config.topology_id,
        stage_id=destination if destination >= 0 else source,
        layer_start=assignment.layer_start,
        layer_end=assignment.layer_end,
        session_id=session_id,
        request_id=request_id,
        sequence_number=sequence,
        token_position=token_position,
        source_stage=source,
        destination_stage=destination,
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        compression_mode="none",
        payload=payload,
        attributes=attributes or {},
        status=status,
    )


def _send_with_trace(
    connection: socket.socket,
    message: StageMessage,
    *,
    trace: TraceWriter,
    data_plane: str,
    shaper: NetworkShaper | None = None,
    timeout_s: float,
) -> None:
    fields = _trace_fields(message)
    trace.emit("serialization_start", data_plane=data_plane, **fields)
    trace.emit("socket_send_start", data_plane=data_plane, **fields)
    captured: dict[str, Any] = {}

    def callback(_: str, values: dict[str, Any]) -> None:
        captured.update(values)

    encoded = send_message(
        connection,
        message,
        shaper=shaper,
        timeout_s=timeout_s,
        telemetry=callback,
    )
    trace.emit(
        "serialization_end",
        data_plane=data_plane,
        duration_ns=int(captured.get("serialisation_ns", 0)),
        wire_bytes=encoded.wire_bytes,
        checksum=encoded.checksum,
        **fields,
    )
    trace.emit(
        "socket_send_end",
        data_plane=data_plane,
        duration_ns=int(captured.get("socket_ns", 0)),
        shaping_ns=int(captured.get("shaping_ns", 0)),
        wire_bytes=encoded.wire_bytes,
        checksum=encoded.checksum,
        **fields,
    )


def _receive_with_trace(
    connection: socket.socket,
    *,
    trace: TraceWriter,
    pool: BufferPool,
    data_plane: str,
    critical_dependency: bool = False,
    unblocks_event_id: str | None = None,
    dependency_token_position: int | None = None,
) -> StageMessage:
    trace.emit("socket_receive_start", data_plane=data_plane)
    started = time.perf_counter_ns()
    message = recv_message(connection, pool=pool)
    duration_ns = time.perf_counter_ns() - started
    encoded = encode_message(message)
    fields = _trace_fields(message)
    trace.emit(
        "socket_receive_end",
        data_plane=data_plane,
        duration_ns=duration_ns,
        wire_bytes=encoded.wire_bytes,
        critical_dependency=critical_dependency,
        unblocks_event_id=unblocks_event_id,
        dependency_token_position=(
            message.token_position
            if dependency_token_position is None
            else dependency_token_position
        ),
        event_id=f"receive:{message_wire_identity(message)}",
        **fields,
    )
    return message


class _StageWorker:
    def __init__(
        self,
        config: StageWorkerConfiguration,
        control_listener: socket.socket,
        data_listener: socket.socket,
    ) -> None:
        self.config = config
        self.assignment = config.assignment
        self.control_listener = control_listener
        self.data_listener = data_listener
        self.profile = NetworkShapeProfile.model_validate(config.network_profile)
        self.shaper = NetworkShaper(self.profile)
        self.trace = TraceWriter(
            Path(config.trace_path),
            base=TraceContext(
                run_id=config.run_id,
                session_id=CONTROL_SESSION,
                request_id="control",
                token_position=-1,
                stage_id=self.assignment.stage_id,
                source_stage=self.assignment.stage_id,
                destination_stage=self.assignment.stage_id,
                message_type="control",
                model_revision=config.model_revision,
            ),
        )
        self.pool = BufferPool(capacity=4, initial_size=256 * 1024)
        self.control_sequences = SequenceAllocator()
        self.ring_sequences = SequenceAllocator()
        self.control_validator = MessageSequenceValidator()
        self.ring_validator = MessageSequenceValidator()
        self.control_send_lock = threading.Lock()
        self.publication_queue: queue.Queue[tuple[_WorkerSession, int, int] | None] = queue.Queue(
            maxsize=config.publication_queue_size
        )
        self.publication_thread: threading.Thread | None = None
        self.model: ContiguousOlmoeStage | None = None
        self.control: socket.socket | None = None
        self.inbound: socket.socket | None = None
        self.outbound: socket.socket | None = None
        self.sessions: dict[str, _WorkerSession] = {}
        self.running = True

    @property
    def stage_id(self) -> int:
        return self.assignment.stage_id

    def _send_control(
        self,
        operation: Operation,
        *,
        session_id: str = CONTROL_SESSION,
        request_id: str = "control",
        token_position: int = -1,
        payload: bytes = b"",
        tensor_shape: tuple[int, ...] = (),
        tensor_dtype: str = "none",
        attributes: dict[str, Any] | None = None,
        status: str = "OK",
    ) -> None:
        assert self.control is not None
        sequence = self.control_sequences.next(session_id, self.stage_id, COORDINATOR_STAGE)
        message = _control_message(
            operation=operation,
            config=self.config,
            sequence=sequence,
            source=self.stage_id,
            destination=COORDINATOR_STAGE,
            session_id=session_id,
            request_id=request_id,
            token_position=token_position,
            payload=payload,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            attributes=attributes,
            status=status,
        )
        with self.control_send_lock:
            _send_with_trace(
                self.control,
                message,
                trace=self.trace,
                data_plane="publication" if operation == Operation.TOKEN_RESULT else "control",
                timeout_s=self.config.timeout_s,
            )

    def _connect_ring(self) -> None:
        next_stage = (self.stage_id + 1) % len(self.config.assignments)
        self.trace.emit(
            "connection_reconnect",
            destination_stage=next_stage,
            status="START",
            endpoint=f"{self.config.ring_endpoints[next_stage][0]}:{self.config.ring_endpoints[next_stage][1]}",
        )
        self.outbound = _connect(self.config.ring_endpoints[next_stage], self.config.timeout_s)
        inbound, _ = self.data_listener.accept()
        inbound.settimeout(self.config.timeout_s)
        inbound.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.inbound = inbound
        self.trace.emit(
            "connection_reconnect",
            source_stage=(self.stage_id - 1) % len(self.config.assignments),
            destination_stage=self.stage_id,
            status="OK",
        )

    def _publication_loop(self) -> None:
        while True:
            item = self.publication_queue.get()
            try:
                if item is None:
                    return
                session, token_position, token = item
                packed = pack_tensor(
                    torch.tensor([token], dtype=torch.int64), requested_mode="none"
                )
                self._send_control(
                    Operation.TOKEN_RESULT,
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=token_position,
                    payload=packed.payload,
                    tensor_shape=packed.shape,
                    tensor_dtype=packed.dtype,
                    attributes={"tensor": packed.attributes(), "publication_only": True},
                )
                self.trace.emit(
                    "token_publication_to_client",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=token_position,
                    message_type="TOKEN_RESULT",
                    published_token_id=token,
                    data_plane="publication",
                )
                self.trace.emit(
                    "token_publication",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=token_position,
                    message_type="TOKEN_RESULT",
                    published_token_id=token,
                    data_plane="publication",
                )
            finally:
                self.publication_queue.task_done()

    def _pack_activation(self, tensor: torch.Tensor) -> PackedTensor:
        requested = self.config.compression_request
        adaptive_inputs = None
        if requested == "adaptive":
            transfers = max(self.shaper.metrics.transfers, 1)
            adaptive_inputs = AdaptiveTransportInputs(
                bandwidth_bps=self.profile.bandwidth_bps,
                rtt_ms=self.profile.one_way_latency_ms * 2.0,
                queue_delay_ms=self.shaper.metrics.queue_wait_ns / transfers / 1e6,
            )
        self.trace.emit("device_to_host_transfer_start")
        started = time.perf_counter_ns()
        packed = pack_tensor(
            tensor,
            requested_mode=requested,
            adaptive_inputs=adaptive_inputs,
        )
        total_ns = time.perf_counter_ns() - started
        self.trace.emit(
            "device_to_host_transfer_end",
            duration_ns=max(0, total_ns - packed.encode_ns - packed.decode_trial_ns),
            payload_bytes=packed.raw_bytes,
            tensor_shape=list(packed.shape),
            tensor_dtype=packed.dtype,
        )
        self.trace.emit(
            "compression_start",
            compression_mode=packed.compression_mode,
            payload_bytes=packed.raw_bytes,
        )
        self.trace.emit(
            "compression_end",
            duration_ns=packed.encode_ns,
            compression_mode=packed.compression_mode,
            payload_bytes=packed.raw_bytes,
            wire_bytes=packed.encoded_bytes,
            compression_ratio=(
                packed.raw_bytes / packed.encoded_bytes if packed.encoded_bytes else 1.0
            ),
            decision=(
                packed.compression_decision.to_dict()
                if packed.compression_decision is not None
                else None
            ),
        )
        return packed

    def _send_ring(
        self,
        *,
        operation: Operation,
        session: _WorkerSession,
        token_position: int,
        packed: PackedTensor,
        cache_position_start: int,
    ) -> None:
        assert self.outbound is not None
        destination = (self.stage_id + 1) % len(self.config.assignments)
        assignment = self.config.assignments[destination]
        sequence = self.ring_sequences.next(session.session_id, self.stage_id, destination)
        message = StageMessage(
            operation=operation,
            model_revision=self.config.model_revision,
            tokenizer_revision=self.config.tokenizer_revision,
            topology_id=self.config.topology_id,
            stage_id=destination,
            layer_start=assignment.layer_start,
            layer_end=assignment.layer_end,
            session_id=session.session_id,
            request_id=session.request_id,
            sequence_number=sequence,
            token_position=token_position,
            source_stage=self.stage_id,
            destination_stage=destination,
            tensor_shape=packed.shape,
            tensor_dtype=packed.dtype,
            compression_mode=packed.compression_mode,
            payload=packed.payload,
            attributes={
                "tensor": packed.attributes(),
                "cache_position_start": cache_position_start,
                "prompt_length": session.prompt_length,
                "generated_token_target": session.generated_token_target,
            },
        )
        _send_with_trace(
            self.outbound,
            message,
            trace=self.trace,
            data_plane="ring",
            shaper=self.shaper,
            timeout_s=self.config.timeout_s,
        )

    def _capture(
        self,
        session: _WorkerSession,
        token_position: int,
        result: StageExecutionResult,
    ) -> None:
        if self.config.capture_directory is None:
            return
        root = (
            Path(self.config.capture_directory)
            / session.session_id
            / f"token-{token_position:04d}"
            / f"stage-{self.stage_id}"
        )
        root.mkdir(parents=True, exist_ok=True)
        tensors: list[tuple[str, torch.Tensor]] = [
            ("stage_boundary_hidden", result.stage_boundary_hidden_states)
        ]
        if result.final_hidden_states is not None:
            tensors.append(("final_hidden", result.final_hidden_states))
        if result.logits is not None:
            tensors.append(("pre_sampling_logits_fp32", result.logits[:, -1, :].float()))
        for local_id, router in enumerate(result.router_logits):
            global_id = self.assignment.layer_start + local_id
            tensors.append((f"router_layer_{global_id:02d}_fp32", router.float()))
        manifest = []
        for name, tensor in tensors:
            raw = tensor_raw_bytes(tensor)
            path = root / f"{name}.bin"
            path.write_bytes(raw)
            manifest.append(
                {
                    "name": name,
                    "path": str(path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _execute(
        self,
        *,
        session: _WorkerSession,
        operation: Operation,
        token_position: int,
        cache_position_start: int,
        tensor: torch.Tensor,
        is_token_ids: bool,
    ) -> None:
        assert self.model is not None
        injection = self.config.failure_injection
        if (
            injection is not None
            and injection.stage_id == self.stage_id
            and injection.token_position == token_position
            and injection.kind in {"stage_process_termination", "socket_disconnect"}
        ):
            self.trace.emit(
                "worker_failure",
                session_id=session.session_id,
                request_id=session.request_id,
                token_position=token_position,
                status="INJECTED",
                failure_kind=injection.kind,
            )
            if injection.kind == "stage_process_termination":
                os._exit(97)
            assert self.outbound is not None
            self.outbound.shutdown(socket.SHUT_RDWR)
            self.outbound.close()
            raise ConnectionError("injected stage socket disconnect")
        self.trace.emit(
            "stage_queue_entry",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            message_type=operation.name,
        )
        queue_started = time.perf_counter_ns()
        self.trace.emit(
            "stage_queue_exit",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            message_type=operation.name,
            duration_ns=time.perf_counter_ns() - queue_started,
        )
        self.trace.emit(
            "host_to_device_transfer_start",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            tensor_shape=list(tensor.shape),
            tensor_dtype=str(tensor.dtype).replace("torch.", ""),
        )
        transfer_started = time.perf_counter_ns()
        device_tensor = tensor.to(
            device=self.model.device,
            dtype=torch.long if is_token_ids else self.model.dtype,
        )
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        self.trace.emit(
            "host_to_device_transfer_end",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            duration_ns=time.perf_counter_ns() - transfer_started,
            tensor_shape=list(device_tensor.shape),
            tensor_dtype=str(device_tensor.dtype).replace("torch.", ""),
        )
        compute_id = f"compute:{session.session_id}:{token_position}:{self.stage_id}"
        self.trace.emit(
            "cuda_compute_start",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            message_type=operation.name,
            event_id=compute_id,
        )
        result = self.model.execute(
            session_id=session.session_id,
            cache_position_start=cache_position_start,
            input_ids=device_tensor if is_token_ids else None,
            hidden_states=None if is_token_ids else device_tensor,
            capture_router_logits=self.config.capture_directory is not None,
        )
        self.trace.emit(
            "cuda_compute_end",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            message_type=operation.name,
            event_id=compute_id,
            duration_ns=result.compute_ns,
        )
        self.trace.emit(
            "kv_cache_update",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            kv_cache_bytes=self.model.kv_cache_bytes(session.session_id),
            cache_sequence_length=result.cache_sequence_length,
        )
        self._capture(session, token_position, result)
        self.trace.emit(
            "stage_output_availability",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            tensor_shape=list(result.hidden_states.shape),
            tensor_dtype=str(result.hidden_states.dtype).replace("torch.", ""),
        )
        if result.sampled_token_ids is None:
            packed = self._pack_activation(result.hidden_states)
            self._send_ring(
                operation=operation,
                session=session,
                token_position=token_position,
                packed=packed,
                cache_position_start=cache_position_start,
            )
            return
        token = int(result.sampled_token_ids.item())
        self.trace.emit(
            "token_sampling",
            session_id=session.session_id,
            request_id=session.request_id,
            token_position=token_position,
            sampled_token_id=token,
            sampling_method="deterministic_greedy_argmax",
        )
        packed_token = pack_tensor(
            result.sampled_token_ids.detach().to(dtype=torch.int64, device="cpu"),
            requested_mode="none",
        )
        if (
            injection is not None
            and injection.stage_id == self.stage_id
            and injection.token_position == token_position
            and injection.kind == "final_stage_failure_before_token_return"
        ):
            raise RuntimeError("injected final-stage failure before token return")
        self._send_ring(
            operation=Operation.TOKEN_RESULT,
            session=session,
            token_position=token_position,
            packed=packed_token,
            cache_position_start=cache_position_start,
        )

    def _validate_incoming(self, message: StageMessage) -> None:
        if message.model_revision != self.config.model_revision:
            raise ValueError("wrong model revision")
        if message.tokenizer_revision != self.config.tokenizer_revision:
            raise ValueError("wrong tokenizer revision")
        if message.topology_id != self.config.topology_id:
            raise ValueError("wrong stage topology")
        if message.destination_stage != self.stage_id or message.stage_id != self.stage_id:
            raise ValueError("wrong destination stage")
        if (message.layer_start, message.layer_end) != (
            self.assignment.layer_start,
            self.assignment.layer_end,
        ):
            raise ValueError("wrong destination layer ownership")
        if message.session_id not in self.sessions:
            raise ValueError("wrong or closed session")

    def _decode_tensor(self, message: StageMessage) -> torch.Tensor:
        self.trace.emit("input_buffer_acquisition", **_trace_fields(message))
        self.trace.emit("decompression_start", **_trace_fields(message))
        tensor, decode_ns = unpack_tensor(message.payload, message.attributes["tensor"])
        self.trace.emit("decompression_end", duration_ns=decode_ns, **_trace_fields(message))
        self.trace.emit("buffer_release", **_trace_fields(message))
        return tensor

    def _handle_ring(self) -> None:
        assert self.inbound is not None
        # The target compute ID is derivable before the receive completes.
        message = _receive_with_trace(
            self.inbound,
            trace=self.trace,
            pool=self.pool,
            data_plane="ring",
        )
        self._validate_incoming(message)
        self.ring_validator.validate(message)
        session = self.sessions[message.session_id]
        is_token_return = message.operation == Operation.TOKEN_RESULT and self.stage_id == 0
        if is_token_return:
            next_position = message.token_position + 1
            critical = next_position < session.generated_token_target
            # Add a dependency event with the fully resolved link.  The raw
            # socket receive above remains available for transport accounting.
            self.trace.emit(
                "socket_receive_end",
                data_plane="ring_dependency",
                critical_dependency=critical,
                unblocks_event_id=(
                    f"compute:{session.session_id}:{next_position}:0" if critical else None
                ),
                dependency_token_position=message.token_position,
                duration_ns=0,
                event_id=f"dependency:{message_wire_identity(message)}",
                **_trace_fields(message),
            )
            token_tensor = self._decode_tensor(message)
            token = int(token_tensor.item())
            if message.token_position != len(session.generated_tokens):
                raise ValueError("stale or out-of-order generated token position")
            session.generated_tokens.append(token)
            injection = self.config.failure_injection
            if (
                injection is not None
                and injection.stage_id == 0
                and injection.token_position == message.token_position
                and injection.kind == "stage_zero_failure_after_token_acceptance"
            ):
                raise RuntimeError("injected stage-zero failure after token acceptance")
            self.trace.emit(
                "token_return_to_stage_zero",
                session_id=session.session_id,
                request_id=session.request_id,
                token_position=message.token_position,
                source_stage=message.source_stage,
                destination_stage=0,
                message_type="TOKEN_RESULT",
                sampled_token_id=token,
            )
            started = session.token_step_started_ns.pop(message.token_position)
            self.trace.emit(
                "token_step_end",
                session_id=session.session_id,
                request_id=session.request_id,
                token_position=message.token_position,
                duration_ns=time.perf_counter_ns() - started,
                sampled_token_id=token,
            )
            self.trace.emit(
                "token_step_completion",
                session_id=session.session_id,
                request_id=session.request_id,
                token_position=message.token_position,
                duration_ns=time.perf_counter_ns() - started,
                sampled_token_id=token,
            )
            try:
                self.publication_queue.put_nowait((session, message.token_position, token))
            except queue.Full as exc:
                self.trace.emit(
                    "fallback",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=message.token_position,
                    status="ERROR",
                    fallback_used=False,
                    reason="bounded publication queue exhausted",
                )
                self.trace.emit(
                    "retry_or_fallback",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=message.token_position,
                    status="ERROR",
                    retry_used=False,
                    fallback_used=False,
                    reason="bounded publication queue exhausted",
                )
                raise RuntimeError("bounded publication queue exhausted") from exc
            if len(session.generated_tokens) < session.generated_token_target:
                target_position = len(session.generated_tokens)
                session.token_step_started_ns[target_position] = time.perf_counter_ns()
                self.trace.emit(
                    "token_step_start",
                    session_id=session.session_id,
                    request_id=session.request_id,
                    token_position=target_position,
                    message_type="DECODE",
                )
                self._execute(
                    session=session,
                    operation=Operation.DECODE,
                    token_position=target_position,
                    cache_position_start=session.prompt_length + target_position - 1,
                    tensor=torch.tensor([[token]], dtype=torch.int64),
                    is_token_ids=True,
                )
            return

        compute_id = f"compute:{session.session_id}:{message.token_position}:{self.stage_id}"
        self.trace.emit(
            "socket_receive_end",
            data_plane="ring_dependency",
            critical_dependency=True,
            unblocks_event_id=compute_id,
            dependency_token_position=message.token_position,
            duration_ns=0,
            event_id=f"dependency:{message_wire_identity(message)}",
            **_trace_fields(message),
        )
        tensor = self._decode_tensor(message)
        self._execute(
            session=session,
            operation=message.operation,
            token_position=message.token_position,
            cache_position_start=int(message.attributes["cache_position_start"]),
            tensor=tensor,
            is_token_ids=False,
        )

    def _handle_control(self) -> None:
        assert self.control is not None
        message = _receive_with_trace(
            self.control,
            trace=self.trace,
            pool=self.pool,
            data_plane="control",
        )
        self.control_validator.validate(message)
        if message.destination_stage != self.stage_id:
            raise ValueError("control message addressed to the wrong stage")
        if message.model_revision != self.config.model_revision:
            raise ValueError("wrong model revision")
        if message.topology_id != self.config.topology_id:
            raise ValueError("wrong topology")
        if message.operation == Operation.OPEN_SESSION:
            assert self.model is not None
            self.model.open_session(message.session_id)
            session = _WorkerSession(
                session_id=message.session_id,
                request_id=message.request_id,
                prompt_length=int(message.attributes["prompt_length"]),
                generated_token_target=int(message.attributes["generated_token_target"]),
            )
            self.sessions[message.session_id] = session
            self.trace.emit(
                "session_creation",
                session_id=message.session_id,
                request_id=message.request_id,
                stage_id=self.stage_id,
                layer_start=self.assignment.layer_start,
                layer_end=self.assignment.layer_end,
            )
            self._send_control(
                Operation.OPEN_SESSION,
                session_id=message.session_id,
                request_id=message.request_id,
                attributes={"opened": True},
            )
            return
        if message.operation == Operation.PREFILL:
            if self.stage_id != 0:
                raise ValueError("coordinator may start prefill only at stage zero")
            session = self.sessions[message.session_id]
            prompt = self._decode_tensor(message)
            session.token_step_started_ns[0] = time.perf_counter_ns()
            self.trace.emit(
                "token_step_start",
                session_id=session.session_id,
                request_id=session.request_id,
                token_position=0,
                message_type="PREFILL",
            )
            self._execute(
                session=session,
                operation=Operation.PREFILL,
                token_position=0,
                cache_position_start=0,
                tensor=prompt,
                is_token_ids=True,
            )
            return
        if message.operation in {Operation.CLOSE_SESSION, Operation.CANCEL_SESSION}:
            assert self.model is not None
            session = self.sessions.pop(message.session_id)
            if self.stage_id == 0:
                self.publication_queue.join()
            released = (
                self.model.cancel_session(message.session_id)
                if message.operation == Operation.CANCEL_SESSION
                else self.model.close_session(message.session_id)
            )
            self.trace.emit(
                "session_close",
                session_id=session.session_id,
                request_id=session.request_id,
                kv_cache_bytes_released=released,
                cancelled=message.operation == Operation.CANCEL_SESSION,
            )
            self._send_control(
                message.operation,
                session_id=session.session_id,
                request_id=session.request_id,
                attributes={"closed": True, "kv_cache_bytes_released": released},
            )
            if message.attributes.get("shutdown"):
                self.running = False
            return
        if message.operation == Operation.HEALTH:
            self._send_control(
                Operation.HEALTH,
                session_id=message.session_id,
                request_id=message.request_id,
                attributes={"healthy": True, "sessions": sorted(self.sessions)},
            )
            return
        raise ValueError(f"unexpected worker control operation {message.operation.name}")

    def run(self) -> None:
        self.control, _ = self.control_listener.accept()
        self.control.settimeout(self.config.timeout_s)
        self.control.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = _receive_with_trace(
            self.control, trace=self.trace, pool=self.pool, data_plane="control"
        )
        self.control_validator.validate(hello)
        if hello.operation != Operation.HELLO:
            raise ValueError("worker expected HELLO")
        self._send_control(
            Operation.CAPABILITIES,
            attributes={
                "operations": [operation.name for operation in Operation],
                "persistent_full_duplex_tcp": True,
                "pickle_allowed": False,
                "pid": os.getpid(),
                "device": "cuda:0",
            },
        )
        load = _receive_with_trace(
            self.control, trace=self.trace, pool=self.pool, data_plane="control"
        )
        self.control_validator.validate(load)
        if load.operation != Operation.LOAD_STAGE:
            raise ValueError("worker expected LOAD_STAGE")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.cuda.set_device(0)
        load_started = time.perf_counter_ns()
        self.model = ContiguousOlmoeStage(
            model_path=Path(self.config.model_path),
            assignment=self.assignment,
            stage_count=len(self.config.assignments),
        )
        self._connect_ring()
        self._send_control(
            Operation.LOAD_STAGE,
            attributes={
                "loaded": True,
                "load_ns": time.perf_counter_ns() - load_started,
                "ownership": self.model.ownership.to_dict(),
                "pid": os.getpid(),
                "control_endpoint": f"{self.config.control_endpoint[0]}:{self.config.control_endpoint[1]}",
                "data_endpoint": f"{self.config.data_endpoint[0]}:{self.config.data_endpoint[1]}",
                "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(0),
            },
        )
        if self.stage_id == 0:
            self.publication_thread = threading.Thread(
                target=self._publication_loop,
                name="stage-zero-token-publisher",
                daemon=True,
            )
            self.publication_thread.start()
        while self.running:
            assert self.inbound is not None
            readable, _, _ = select.select([self.control, self.inbound], [], [], 1.0)
            for connection in readable:
                if connection is self.control:
                    self._handle_control()
                else:
                    self._handle_ring()
        if self.stage_id == 0:
            self.publication_queue.put(None)
            self.publication_queue.join()
            assert self.publication_thread is not None
            self.publication_thread.join(timeout=5.0)

    def close(self) -> None:
        for connection in (self.inbound, self.outbound, self.control):
            if connection is not None:
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                connection.close()
        self.control_listener.close()
        self.data_listener.close()
        self.trace.close()


def _stage_worker_entry(
    config: StageWorkerConfiguration,
    control_listener: socket.socket,
    data_listener: socket.socket,
) -> None:
    worker = _StageWorker(config, control_listener, data_listener)
    try:
        worker.run()
    except BaseException as exc:
        try:
            worker.trace.emit(
                "worker_failure",
                status="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            if worker.control is not None:
                worker._send_control(
                    Operation.ERROR,
                    attributes={
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                    status="ERROR",
                )
        except BaseException:
            pass
        raise
    finally:
        worker.close()


class StageRingController:
    """Control-plane owner; never executes model layers."""

    def __init__(
        self,
        *,
        run_id: str,
        plan: StagePlan,
        network_profile: NetworkShapeProfile,
        output_directory: Path,
        compression_request: CompressionRequest = "none",
        timeout_s: float = 180.0,
        capture_boundaries: bool = False,
        failure_injection: FailureInjection | None = None,
    ) -> None:
        plan.validate()
        self.run_id = run_id
        self.plan = plan
        self.network_profile = network_profile
        self.output_directory = output_directory.resolve()
        self.compression_request = compression_request
        self.timeout_s = timeout_s
        self.capture_boundaries = capture_boundaries
        self.failure_injection = failure_injection
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_directory / "traces" / "coordinator.ndjson"
        self.trace = TraceWriter(
            self.trace_path,
            base=TraceContext(
                run_id=run_id,
                session_id=CONTROL_SESSION,
                request_id="control",
                token_position=-1,
                stage_id=COORDINATOR_STAGE,
                source_stage=COORDINATOR_STAGE,
                destination_stage=COORDINATOR_STAGE,
                message_type="control",
                model_revision=plan.model_revision,
            ),
        )
        self.control_sequences = SequenceAllocator()
        self.response_validators = [MessageSequenceValidator() for _ in plan.assignments]

    def _send_control(
        self,
        connection: socket.socket,
        config: StageWorkerConfiguration,
        operation: Operation,
        *,
        session_id: str = CONTROL_SESSION,
        request_id: str = "control",
        token_position: int = -1,
        payload: bytes = b"",
        tensor_shape: tuple[int, ...] = (),
        tensor_dtype: str = "none",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        sequence = self.control_sequences.next(
            session_id, COORDINATOR_STAGE, config.assignment.stage_id
        )
        message = _control_message(
            operation=operation,
            config=config,
            sequence=sequence,
            source=COORDINATOR_STAGE,
            destination=config.assignment.stage_id,
            session_id=session_id,
            request_id=request_id,
            token_position=token_position,
            payload=payload,
            tensor_shape=tensor_shape,
            tensor_dtype=tensor_dtype,
            attributes=attributes,
        )
        _send_with_trace(
            connection,
            message,
            trace=self.trace,
            data_plane="control",
            timeout_s=self.timeout_s,
        )

    def _receive_control(
        self,
        connection: socket.socket,
        stage_id: int,
        pool: BufferPool,
    ) -> StageMessage:
        message = _receive_with_trace(connection, trace=self.trace, pool=pool, data_plane="control")
        self.response_validators[stage_id].validate(message)
        if message.source_stage != stage_id or message.destination_stage != COORDINATOR_STAGE:
            raise ValueError("worker response has the wrong control-plane route")
        if message.status != "OK" or message.operation == Operation.ERROR:
            raise RuntimeError(str(message.attributes.get("error", "stage worker error")))
        return message

    def run(
        self,
        *,
        prompt_token_ids: list[int],
        generated_token_count: int,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> StageRingResult:
        if generated_token_count < 1:
            raise ValueError("generated token count must be positive")
        session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        request_id = request_id or f"request-{uuid.uuid4().hex[:12]}"
        control_listeners: list[socket.socket] = []
        data_listeners: list[socket.socket] = []
        control_endpoints: list[tuple[str, int]] = []
        data_endpoints: list[tuple[str, int]] = []
        for _ in self.plan.assignments:
            control_listener, control_endpoint = _make_listener()
            data_listener, data_endpoint = _make_listener()
            control_listeners.append(control_listener)
            data_listeners.append(data_listener)
            control_endpoints.append(control_endpoint)
            data_endpoints.append(data_endpoint)
        capture_directory = (
            str(self.output_directory / "captures") if self.capture_boundaries else None
        )
        configs = tuple(
            StageWorkerConfiguration(
                run_id=self.run_id,
                model_path=self.plan.model_path,
                model_revision=self.plan.model_revision,
                tokenizer_revision=self.plan.tokenizer_revision,
                topology_id=self.plan.topology_id,
                assignment=assignment,
                assignments=self.plan.assignments,
                network_profile=self.network_profile.model_dump(mode="json"),
                compression_request=self.compression_request,
                trace_path=str(
                    self.output_directory / "traces" / f"stage-{assignment.stage_id}.ndjson"
                ),
                capture_directory=capture_directory,
                control_endpoint=control_endpoints[assignment.stage_id],
                data_endpoint=data_endpoints[assignment.stage_id],
                ring_endpoints=tuple(data_endpoints),
                timeout_s=self.timeout_s,
                publication_queue_size=max(64, generated_token_count * 2),
                failure_injection=self.failure_injection,
            )
            for assignment in self.plan.assignments
        )
        context = mp.get_context("spawn")
        processes = [
            context.Process(
                target=_stage_worker_entry,
                args=(config, control_listeners[index], data_listeners[index]),
                name=f"experiment-011-stage-{index}",
            )
            for index, config in enumerate(configs)
        ]
        controls: list[socket.socket] = []
        pools = [BufferPool(capacity=2, initial_size=256 * 1024) for _ in configs]
        ownership: list[dict[str, Any]] = []
        pids: list[int] = []
        errors: list[str] = []
        received_tokens: list[int] = []
        publication_times: list[int] = []
        elapsed_seconds = 0.0
        graceful_shutdown_acknowledged = False
        try:
            for process in processes:
                process.start()
            for listener in control_listeners + data_listeners:
                listener.close()
            controls = [_connect(endpoint, self.timeout_s) for endpoint in control_endpoints]
            for connection in controls:
                connection.settimeout(self.timeout_s)
            for connection, config in zip(controls, configs, strict=True):
                self._send_control(connection, config, Operation.HELLO)
            for stage_id, connection in enumerate(controls):
                response = self._receive_control(connection, stage_id, pools[stage_id])
                if response.operation != Operation.CAPABILITIES:
                    raise RuntimeError("worker did not return CAPABILITIES")
                pids.append(int(response.attributes["pid"]))
            # Issue all load commands before waiting so model shards load in parallel.
            for connection, config in zip(controls, configs, strict=True):
                self._send_control(connection, config, Operation.LOAD_STAGE)
            for stage_id, connection in enumerate(controls):
                response = self._receive_control(connection, stage_id, pools[stage_id])
                if response.operation != Operation.LOAD_STAGE:
                    raise RuntimeError("worker did not acknowledge LOAD_STAGE")
                ownership.append(dict(response.attributes["ownership"]))
            for connection, config in zip(controls, configs, strict=True):
                self._send_control(
                    connection,
                    config,
                    Operation.OPEN_SESSION,
                    session_id=session_id,
                    request_id=request_id,
                    attributes={
                        "prompt_length": len(prompt_token_ids),
                        "generated_token_target": generated_token_count,
                    },
                )
            for stage_id, connection in enumerate(controls):
                response = self._receive_control(connection, stage_id, pools[stage_id])
                if response.operation != Operation.OPEN_SESSION:
                    raise RuntimeError("worker did not acknowledge OPEN_SESSION")
            prompt = pack_tensor(
                torch.tensor([prompt_token_ids], dtype=torch.int64), requested_mode="none"
            )
            measurement_started = time.perf_counter_ns()
            self._send_control(
                controls[0],
                configs[0],
                Operation.PREFILL,
                session_id=session_id,
                request_id=request_id,
                token_position=0,
                payload=prompt.payload,
                tensor_shape=prompt.shape,
                tensor_dtype=prompt.dtype,
                attributes={
                    "tensor": prompt.attributes(),
                    "cache_position_start": 0,
                    "prompt_length": len(prompt_token_ids),
                    "generated_token_target": generated_token_count,
                },
            )
            while len(received_tokens) < generated_token_count:
                blocked_started = time.perf_counter_ns()
                self.trace.emit(
                    "coordinator_blocked_start",
                    session_id=session_id,
                    request_id=request_id,
                    token_position=len(received_tokens),
                    reason="await asynchronous token publication",
                )
                publication = self._receive_control(controls[0], 0, pools[0])
                self.trace.emit(
                    "coordinator_blocked_end",
                    session_id=session_id,
                    request_id=request_id,
                    token_position=len(received_tokens),
                    duration_ns=time.perf_counter_ns() - blocked_started,
                    reason="await asynchronous token publication",
                )
                if publication.operation != Operation.TOKEN_RESULT:
                    raise RuntimeError(
                        f"unexpected stage-zero publication {publication.operation.name}"
                    )
                tensor, _ = unpack_tensor(publication.payload, publication.attributes["tensor"])
                received_tokens.append(int(tensor.item()))
                publication_times.append(time.perf_counter_ns())
            measurement_ended = time.perf_counter_ns()
            elapsed_seconds = (measurement_ended - measurement_started) / 1e9
            for connection, config in zip(controls, configs, strict=True):
                self._send_control(
                    connection,
                    config,
                    Operation.CLOSE_SESSION,
                    session_id=session_id,
                    request_id=request_id,
                    attributes={"shutdown": True},
                )
            for stage_id, connection in enumerate(controls):
                response = self._receive_control(connection, stage_id, pools[stage_id])
                if response.operation != Operation.CLOSE_SESSION:
                    raise RuntimeError("worker did not acknowledge CLOSE_SESSION")
            graceful_shutdown_acknowledged = True
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            self.trace.emit(
                "worker_failure",
                session_id=session_id,
                request_id=request_id,
                status="ERROR",
                error=errors[-1],
                traceback=traceback.format_exc(),
            )
        finally:
            for connection in controls:
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                connection.close()
            for listener in control_listeners + data_listeners:
                with contextlib.suppress(OSError):
                    listener.close()
            # CUDA runtime teardown can linger on Windows after every worker
            # has acknowledged CLOSE_SESSION and released its KV cache.  Give
            # the complete process set one bounded grace period, then reap it;
            # this occurs after the measured session and is not a fallback.
            cleanup_deadline = time.monotonic() + 1.0
            for process in processes:
                process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                    if not graceful_shutdown_acknowledged:
                        errors.append(f"stage process {process.name} required termination")
                elif process.exitcode not in {0, None}:
                    errors.append(f"stage process {process.name} exited {process.exitcode}")
            self.trace.close()
        trace_paths = (
            *(
                str(self.output_directory / "traces" / f"stage-{index}.ndjson")
                for index in range(len(configs))
            ),
            str(self.trace_path),
        )
        events = merge_traces(tuple(Path(path) for path in trace_paths))
        critical_path = reconstruct_critical_path(events, generated_tokens=len(received_tokens))
        ttft = (
            (publication_times[0] - (publication_times[0] - int(elapsed_seconds * 1e9))) / 1e9
            if publication_times and elapsed_seconds
            else 0.0
        )
        # Reconstruct TTFT directly from coordinator events to avoid relying on
        # wall-clock conversion or parent-side bookkeeping after exceptions.
        sends = [
            event
            for event in events
            if event.get("process_id") == os.getpid()
            and event.get("event") == "socket_send_end"
            and event.get("message_type") == "PREFILL"
        ]
        if sends and publication_times:
            ttft = (publication_times[0] - int(sends[-1]["monotonic_ns"])) / 1e9
        itls = tuple((right - left) / 1e9 for left, right in itertools.pairwise(publication_times))
        compression_modes = sorted(
            {
                str(event.get("compression_mode"))
                for event in events
                if event.get("event") == "compression_end"
            }
        )
        result = StageRingResult(
            run_id=self.run_id,
            session_id=session_id,
            request_id=request_id,
            topology_id=self.plan.topology_id,
            profile_name=self.network_profile.name,
            compression_request=self.compression_request,
            generated_token_ids=tuple(received_tokens),
            generated_tokens=len(received_tokens),
            elapsed_seconds=elapsed_seconds,
            ttft_seconds=ttft,
            inter_token_latencies_seconds=itls,
            throughput_tps=(len(received_tokens) / elapsed_seconds if elapsed_seconds else 0.0),
            stage_process_ids=tuple(pids),
            stage_endpoints=tuple(f"{host}:{port}" for host, port in data_endpoints),
            ownership=tuple(ownership),
            trace_paths=trace_paths,
            critical_path=critical_path,
            fallback_used=False,
            valid_for_claims=(
                not errors
                and len(received_tokens) == generated_token_count
                and not critical_path["invalid_dependency_links"]
            ),
            errors=tuple(errors),
            compression_modes_used=tuple(compression_modes),
        )
        result_path = self.output_directory / "stage_ring_result.json"
        result_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result


def decode_frame_bytes(frame: bytes) -> StageMessage:
    """Small public seam used by corruption/recovery smoke tests."""

    return decode_message(frame)
