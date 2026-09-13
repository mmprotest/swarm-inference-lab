"""Binary client and measurement model for the E027 coarse stage protocol."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from itertools import count
from typing import Any

import numpy as np
from numpy.typing import NDArray


WIRE_MAGIC = 0x37323045
WIRE_VERSION = 1
REQUEST = struct.Struct("<IHHQqqIIIIQ")
RESPONSE = struct.Struct("<IHHQIIIIIIiiQQQQQ")
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


class Operation(IntEnum):
    INFER = 1
    RESET = 2
    PING = 3
    TOKENIZE = 4
    SHUTDOWN = 5


class ResponseKind(IntEnum):
    HIDDEN = 1
    FINAL = 2
    TOKENS = 3
    JSON = 4
    EMPTY = 5


class Flags(IntFlag):
    INPUT_TOKENS = 1 << 0
    RETURN_NEXTN = 1 << 1
    RETURN_FULL_LOGITS = 1 << 2
    ADD_SPECIAL_TOKENS = 1 << 3
    PARSE_SPECIAL_TOKENS = 1 << 4
    RETURN_TAP22 = 1 << 5
    RETURN_TAP44 = 1 << 6
    INPUT_MTP = 1 << 7


@dataclass(frozen=True, slots=True)
class StageResponse:
    request_id: int
    kind: ResponseKind
    flags: Flags
    n_tokens: int
    n_embd: int
    n_vocab: int
    top_k: int
    stage_start: int
    stage_end: int
    payload: bytes
    compute_ns: int
    total_ns: int
    deserialize_ns: int
    serialize_ns: int
    round_trip_ns: int
    client_serialize_ns: int

    @property
    def wire_bytes(self) -> int:
        return RESPONSE.size + len(self.payload)


@dataclass(frozen=True, slots=True)
class FinalStageOutput:
    top_ids: NDArray[np.int32]
    top_logits: NDArray[np.float32]
    nextn: NDArray[np.float32] | None
    full_logits: NDArray[np.float32] | None
    tap22: NDArray[np.float32] | None
    tap44: NDArray[np.float32] | None

    @property
    def greedy_ids(self) -> NDArray[np.int32]:
        return self.top_ids[:, 0]


@dataclass(frozen=True, slots=True)
class StageExchange:
    endpoint: str
    stage_start: int
    stage_end: int
    request_bytes: int
    response_bytes: int
    round_trip_ns: int
    compute_ns: int
    server_total_ns: int
    server_deserialize_ns: int
    server_serialize_ns: int
    client_serialize_ns: int

    @property
    def non_compute_ns(self) -> int:
        return max(0, self.round_trip_ns - self.compute_ns)


@dataclass(frozen=True, slots=True)
class StageTraversal:
    position: int
    n_tokens: int
    elapsed_ns: int
    exchanges: tuple[StageExchange, ...]
    output: FinalStageOutput

    @property
    def compute_ns(self) -> int:
        return sum(item.compute_ns for item in self.exchanges)

    @property
    def serialization_ns(self) -> int:
        return sum(
            item.client_serialize_ns
            + item.server_deserialize_ns
            + item.server_serialize_ns
            for item in self.exchanges
        )


class StageProtocolError(RuntimeError):
    pass


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise StageProtocolError("stage connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class StageClient:
    """One ordered, persistent TCP connection to one state-owning stage."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 180.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._ids = count(1)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = sock

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None

    def __enter__(self) -> StageClient:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _exchange(
        self,
        operation: Operation,
        *,
        payload: bytes = b"",
        position: int = 0,
        rewind_position: int = -1,
        n_tokens: int = 0,
        n_embd: int = 0,
        flags: Flags = Flags(0),
        arg: int = 0,
    ) -> StageResponse:
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("E027 payload exceeds the protocol limit")
        self.connect()
        assert self._socket is not None
        request_id = next(self._ids)
        serialize_begin = time.perf_counter_ns()
        header = REQUEST.pack(
            WIRE_MAGIC,
            WIRE_VERSION,
            int(operation),
            request_id,
            position,
            rewind_position,
            n_tokens,
            n_embd,
            int(flags),
            arg,
            len(payload),
        )
        client_serialize_ns = time.perf_counter_ns() - serialize_begin
        with self._lock:
            begin = time.perf_counter_ns()
            self._socket.sendall(header + payload)
            values = RESPONSE.unpack(_recv_exact(self._socket, RESPONSE.size))
            (
                magic,
                version,
                status,
                response_id,
                kind,
                response_flags,
                response_tokens,
                response_embd,
                n_vocab,
                top_k,
                stage_start,
                stage_end,
                payload_bytes,
                compute_ns,
                total_ns,
                deserialize_ns,
                serialize_ns,
            ) = values
            if payload_bytes > MAX_PAYLOAD_BYTES:
                raise StageProtocolError("stage response exceeds the protocol limit")
            response_payload = _recv_exact(self._socket, payload_bytes)
            round_trip_ns = time.perf_counter_ns() - begin
        if magic != WIRE_MAGIC or version != WIRE_VERSION or response_id != request_id:
            raise StageProtocolError("stage returned a mismatched frame")
        if status:
            message = response_payload.decode("utf-8", errors="replace")
            raise StageProtocolError(
                f"stage {self.endpoint} [{stage_start},{stage_end}) failed: {message}"
            )
        return StageResponse(
            request_id=response_id,
            kind=ResponseKind(kind),
            flags=Flags(response_flags),
            n_tokens=response_tokens,
            n_embd=response_embd,
            n_vocab=n_vocab,
            top_k=top_k,
            stage_start=stage_start,
            stage_end=stage_end,
            payload=response_payload,
            compute_ns=compute_ns,
            total_ns=total_ns,
            deserialize_ns=deserialize_ns,
            serialize_ns=serialize_ns,
            round_trip_ns=round_trip_ns,
            client_serialize_ns=client_serialize_ns,
        )

    def ping(self) -> dict[str, Any]:
        response = self._exchange(Operation.PING)
        if response.kind is not ResponseKind.JSON:
            raise StageProtocolError("ping returned the wrong response kind")
        return json.loads(response.payload)

    def reset(self) -> StageResponse:
        return self._exchange(Operation.RESET)

    def shutdown(self) -> StageResponse:
        return self._exchange(Operation.SHUTDOWN)

    def tokenize(
        self, text: str, *, add_special: bool = True, parse_special: bool = True
    ) -> NDArray[np.int32]:
        flags = Flags(0)
        if add_special:
            flags |= Flags.ADD_SPECIAL_TOKENS
        if parse_special:
            flags |= Flags.PARSE_SPECIAL_TOKENS
        response = self._exchange(
            Operation.TOKENIZE, payload=text.encode("utf-8"), flags=flags
        )
        if response.kind is not ResponseKind.TOKENS:
            raise StageProtocolError("tokenize returned the wrong response kind")
        tokens = np.frombuffer(response.payload, dtype="<i4").copy()
        if tokens.size != response.n_tokens:
            raise StageProtocolError("token response length mismatch")
        return tokens

    def infer_tokens(
        self,
        tokens: NDArray[np.integer[Any]] | list[int],
        *,
        position: int,
        n_embd: int,
        rewind_position: int = -1,
    ) -> StageResponse:
        array = np.ascontiguousarray(tokens, dtype="<i4").reshape(-1)
        return self._exchange(
            Operation.INFER,
            payload=array.tobytes(),
            position=position,
            rewind_position=rewind_position,
            n_tokens=int(array.size),
            n_embd=n_embd,
            flags=Flags.INPUT_TOKENS,
        )

    def infer_hidden(
        self,
        hidden: NDArray[np.floating[Any]],
        *,
        position: int,
        rewind_position: int = -1,
        top_k: int = 16,
        return_nextn: bool = False,
        return_full_logits: bool = False,
    ) -> StageResponse:
        array = np.ascontiguousarray(hidden, dtype="<f4")
        if array.ndim != 2:
            raise ValueError("stage hidden input must be a [tokens, embedding] matrix")
        flags = Flags(0)
        if return_nextn:
            flags |= Flags.RETURN_NEXTN
        if return_full_logits:
            flags |= Flags.RETURN_FULL_LOGITS
        return self._exchange(
            Operation.INFER,
            payload=array.tobytes(),
            position=position,
            rewind_position=rewind_position,
            n_tokens=array.shape[0],
            n_embd=array.shape[1],
            flags=flags,
            arg=top_k,
        )

    def infer_mtp(self, tokens: list[int], hidden: NDArray[np.floating[Any]], *,
                  position: int, rewind_position: int = -1) -> FinalStageOutput:
        ids = np.ascontiguousarray(tokens, dtype="<i4")
        rows = np.ascontiguousarray(hidden, dtype="<f4")
        if rows.ndim != 2 or rows.shape[0] != len(ids):
            raise ValueError("MTP requires one hidden row per token")
        return final_output(self._exchange(
            Operation.INFER, payload=ids.tobytes() + rows.tobytes(), position=position,
            rewind_position=rewind_position, n_tokens=len(ids), n_embd=rows.shape[1],
            flags=Flags.INPUT_TOKENS | Flags.INPUT_MTP | Flags.RETURN_NEXTN, arg=1,
        ))


def hidden_output(response: StageResponse) -> NDArray[np.float32]:
    if response.kind is not ResponseKind.HIDDEN:
        raise StageProtocolError("expected a hidden-stage response")
    expected = response.n_tokens * response.n_embd
    array = np.frombuffer(response.payload, dtype="<f4")
    if array.size != expected:
        raise StageProtocolError("hidden-stage payload length mismatch")
    return array.reshape(response.n_tokens, response.n_embd).copy()


def final_output(response: StageResponse) -> FinalStageOutput:
    if response.kind is not ResponseKind.FINAL:
        raise StageProtocolError("expected a final-stage response")
    rows = response.n_tokens
    width = response.top_k
    top_count = rows * width
    offset = 0
    ids_bytes = top_count * np.dtype("<i4").itemsize
    logits_bytes = top_count * np.dtype("<f4").itemsize
    top_ids = np.frombuffer(response.payload, dtype="<i4", count=top_count, offset=offset)
    offset += ids_bytes
    top_logits = np.frombuffer(
        response.payload, dtype="<f4", count=top_count, offset=offset
    )
    offset += logits_bytes
    nextn = None
    if response.flags & Flags.RETURN_NEXTN:
        count_nextn = rows * response.n_embd
        nextn = np.frombuffer(
            response.payload, dtype="<f4", count=count_nextn, offset=offset
        ).reshape(rows, response.n_embd)
        offset += count_nextn * np.dtype("<f4").itemsize
    full_logits = None
    if response.flags & Flags.RETURN_FULL_LOGITS:
        count_logits = rows * response.n_vocab
        full_logits = np.frombuffer(
            response.payload, dtype="<f4", count=count_logits, offset=offset
        ).reshape(rows, response.n_vocab)
        offset += count_logits * np.dtype("<f4").itemsize
    tap22 = None
    if response.flags & Flags.RETURN_TAP22:
        count_tap = rows * response.n_embd
        tap22 = np.frombuffer(
            response.payload, dtype="<f4", count=count_tap, offset=offset
        ).reshape(rows, response.n_embd)
        offset += count_tap * np.dtype("<f4").itemsize
    tap44 = None
    if response.flags & Flags.RETURN_TAP44:
        count_tap = rows * response.n_embd
        tap44 = np.frombuffer(
            response.payload, dtype="<f4", count=count_tap, offset=offset
        ).reshape(rows, response.n_embd)
        offset += count_tap * np.dtype("<f4").itemsize
    if offset != len(response.payload):
        raise StageProtocolError("final-stage payload length mismatch")
    return FinalStageOutput(
        top_ids=top_ids.reshape(rows, width).copy(),
        top_logits=top_logits.reshape(rows, width).copy(),
        nextn=None if nextn is None else nextn.copy(),
        full_logits=None if full_logits is None else full_logits.copy(),
        tap22=None if tap22 is None else tap22.copy(),
        tap44=None if tap44 is None else tap44.copy(),
    )


def _exchange_measurement(client: StageClient, response: StageResponse, request_bytes: int) -> StageExchange:
    return StageExchange(
        endpoint=client.endpoint,
        stage_start=response.stage_start,
        stage_end=response.stage_end,
        request_bytes=request_bytes,
        response_bytes=response.wire_bytes,
        round_trip_ns=response.round_trip_ns,
        compute_ns=response.compute_ns,
        server_total_ns=response.total_ns,
        server_deserialize_ns=response.deserialize_ns,
        server_serialize_ns=response.serialize_ns,
        client_serialize_ns=response.client_serialize_ns,
    )


class StagePipeline:
    """Coordinator relay for three coarse stage calls; no tensor work occurs here."""

    def __init__(self, stages: tuple[StageClient, StageClient, StageClient]) -> None:
        self.stages = stages
        facts = tuple(stage.ping() for stage in stages)
        ranges = tuple((int(item["stage_start"]), int(item["stage_end"])) for item in facts)
        if ranges[0][0] != 0 or any(
            ranges[index][1] != ranges[index + 1][0] for index in range(2)
        ) or not bool(facts[-1]["final_stage"]):
            raise ValueError(f"stages do not form a contiguous final pipeline: {ranges}")
        embedding_sizes = {int(item["n_embd"]) for item in facts}
        if len(embedding_sizes) != 1:
            raise ValueError("stage embedding widths differ")
        self.n_embd = embedding_sizes.pop()

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def traverse(
        self,
        tokens: NDArray[np.integer[Any]] | list[int],
        *,
        position: int,
        rewind_position: int = -1,
        top_k: int = 16,
        return_nextn: bool = False,
        return_full_logits: bool = False,
    ) -> StageTraversal:
        token_array = np.ascontiguousarray(tokens, dtype="<i4").reshape(-1)
        begin = time.perf_counter_ns()
        first = self.stages[0].infer_tokens(
            token_array,
            position=position,
            n_embd=self.n_embd,
            rewind_position=rewind_position,
        )
        first_hidden = hidden_output(first)
        second = self.stages[1].infer_hidden(
            first_hidden,
            position=position,
            rewind_position=rewind_position,
        )
        second_hidden = hidden_output(second)
        third = self.stages[2].infer_hidden(
            second_hidden,
            position=position,
            rewind_position=rewind_position,
            top_k=top_k,
            return_nextn=return_nextn,
            return_full_logits=return_full_logits,
        )
        elapsed = time.perf_counter_ns() - begin
        output = final_output(third)
        exchanges = (
            _exchange_measurement(
                self.stages[0], first, REQUEST.size + token_array.nbytes
            ),
            _exchange_measurement(
                self.stages[1], second, REQUEST.size + first_hidden.nbytes
            ),
            _exchange_measurement(
                self.stages[2], third, REQUEST.size + second_hidden.nbytes
            ),
        )
        return StageTraversal(
            position=position,
            n_tokens=int(token_array.size),
            elapsed_ns=elapsed,
            exchanges=exchanges,
            output=output,
        )

    def close(self) -> None:
        for stage in self.stages:
            stage.close()
