"""Legacy experiment-only Universal Worker ABI with fail-closed remote TLS."""

from __future__ import annotations

import asyncio
import json
import struct
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from swarm_inference.config.models import StrictModel
from swarm_inference.host import format_endpoint
from swarm_inference.protocol.checksums import sha256_file
from swarm_inference.security.tls import (
    TlsClientConfig,
    TlsServerConfig,
    require_tls_for_endpoint,
)
from swarm_inference.worker.abi import (
    BackendAdapter,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerIdentity,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerProtocolVersion,
)

_FRAME_LENGTH = struct.Struct(">I")
_MAX_CONTROL_FRAME_BYTES = 512 * 1024 * 1024


class UniversalWorkerRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    method: Literal[
        "negotiate",
        "identity",
        "capabilities",
        "benchmark",
        "submit",
        "cancel",
        "heartbeat",
        "validate_shard",
        "shutdown",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


class UniversalWorkerResponse(StrictModel):
    request_id: str
    accepted: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


async def _read_frame(reader: asyncio.StreamReader) -> bytes:
    length_raw = await reader.readexactly(_FRAME_LENGTH.size)
    length = _FRAME_LENGTH.unpack(length_raw)[0]
    if length > _MAX_CONTROL_FRAME_BYTES:
        raise ValueError(f"control frame exceeds {_MAX_CONTROL_FRAME_BYTES} bytes")
    return await reader.readexactly(length)


async def _write_frame(writer: asyncio.StreamWriter, payload: bytes) -> None:
    if len(payload) > _MAX_CONTROL_FRAME_BYTES:
        raise ValueError(f"control frame exceeds {_MAX_CONTROL_FRAME_BYTES} bytes")
    writer.write(_FRAME_LENGTH.pack(len(payload)) + payload)
    await writer.drain()


def _encode_model(value: StrictModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class UniversalWorkerServer:
    """Expose one concrete backend without cross-backend fallback."""

    def __init__(
        self,
        *,
        adapter: BackendAdapter,
        identity: WorkerIdentity,
        host: str = "127.0.0.1",
        port: int = 0,
        tls: TlsServerConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        if identity.backend_id != adapter.backend_id:
            raise ValueError("worker identity backend_id must match its adapter")
        self.adapter = adapter
        self.identity = identity
        self.host = host
        self.port = port
        self.tls = tls
        self.allow_plaintext_loopback = allow_plaintext_loopback
        self._server: asyncio.Server | None = None
        self._closed = asyncio.Event()
        self._jobs: dict[str, asyncio.Task[WorkerJobResult]] = {}
        self._job_request_ids: dict[str, str] = {}
        self._started_ns = time.monotonic_ns()
        self._heartbeat_count = 0

    @property
    def endpoint(self) -> tuple[str, int]:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Universal Worker server is not started")
        address = self._server.sockets[0].getsockname()
        return str(address[0]), int(address[1])

    async def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.endpoint
        require_tls_for_endpoint(
            format_endpoint(self.host, self.port),
            tls_configured=self.tls is not None,
            allow_plaintext_loopback=self.allow_plaintext_loopback,
            transport_name="legacy Universal Worker experiment transport",
        )
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
            ssl=self.tls.ssl_context() if self.tls is not None else None,
        )
        return self.endpoint

    async def serve_until_shutdown(self) -> None:
        await self.start()
        await self._closed.wait()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        for task in self._jobs.values():
            task.cancel()
        if self._jobs:
            await asyncio.gather(*self._jobs.values(), return_exceptions=True)
        await self.adapter.shutdown()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            if self.tls is not None:
                tls_object = writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                self.tls.validate_peer_der(peer_der)
            while not reader.at_eof():
                try:
                    raw = await _read_frame(reader)
                except asyncio.IncompleteReadError:
                    break
                request = UniversalWorkerRequest.model_validate_json(raw)
                response = await self._dispatch(request)
                await _write_frame(writer, _encode_model(response))
                if request.method == "shutdown" and response.accepted:
                    break
        except (ValueError, json.JSONDecodeError) as exc:
            response = UniversalWorkerResponse(
                request_id="invalid",
                accepted=False,
                error=f"invalid Universal Worker request: {exc}",
            )
            with suppress(ConnectionError):
                await _write_frame(writer, _encode_model(response))
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    async def _dispatch(self, request: UniversalWorkerRequest) -> UniversalWorkerResponse:
        try:
            payload = await self._dispatch_payload(request.method, request.payload)
            return UniversalWorkerResponse(
                request_id=request.request_id,
                accepted=True,
                payload=payload,
            )
        except Exception as exc:
            return UniversalWorkerResponse(
                request_id=request.request_id,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _dispatch_payload(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "negotiate":
            peer = WorkerProtocolVersion.model_validate(payload)
            agreed = self.identity.protocol_version.negotiate(peer)
            if agreed is None:
                raise ValueError(
                    f"protocol major mismatch: worker={self.identity.protocol_version.major} "
                    f"peer={peer.major}"
                )
            return agreed.model_dump(mode="json")
        if method == "identity":
            return self.identity.model_dump(mode="json")
        if method == "capabilities":
            return self.adapter.capabilities().model_dump(mode="json")
        if method == "benchmark":
            return self.adapter.benchmark_profile().model_dump(mode="json")
        if method == "submit":
            job = WorkerJob.model_validate(payload)
            rejected = self.adapter.admission_result(job)
            if rejected is not None:
                return rejected.model_dump(mode="json")
            task = asyncio.create_task(self.adapter.execute(job), name=f"worker-job-{job.job_id}")
            self._jobs[job.job_id] = task
            self._job_request_ids[job.job_id] = job.request_id
            try:
                result = await task
            except asyncio.CancelledError:
                result = WorkerJobResult(
                    job_id=job.job_id,
                    request_id=job.request_id,
                    status=WorkerJobStatus.CANCELLED,
                    detail="job cancelled",
                )
            except Exception as exc:
                result = WorkerJobResult(
                    job_id=job.job_id,
                    request_id=job.request_id,
                    status=WorkerJobStatus.BACKEND_FAILURE,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._jobs.pop(job.job_id, None)
                self._job_request_ids.pop(job.job_id, None)
            return result.model_dump(mode="json")
        if method == "cancel":
            request_id = str(payload["request_id"])
            cancelled = False
            for job_id, job_request_id in tuple(self._job_request_ids.items()):
                if job_request_id == request_id:
                    cancel_task = self._jobs.get(job_id)
                    if cancel_task is not None:
                        cancel_task.cancel()
                        cancelled = True
            cancelled = await self.adapter.cancel(request_id) or cancelled
            return {"cancelled": cancelled, "request_id": request_id}
        if method == "heartbeat":
            self._heartbeat_count += 1
            return {
                "worker_id": self.identity.worker_id,
                "heartbeat_count": self._heartbeat_count,
                "active_jobs": len(self._jobs),
                "uptime_ms": (time.monotonic_ns() - self._started_ns) / 1_000_000,
            }
        if method == "validate_shard":
            path = Path(str(payload["path"])).expanduser().resolve()
            expected = str(payload["sha256"])
            actual = sha256_file(path)
            return {"valid": actual == expected, "expected": expected, "actual": actual}
        if method == "shutdown":
            asyncio.get_running_loop().call_soon(asyncio.create_task, self.close())
            return {"clean_shutdown": True}
        raise ValueError(f"unsupported Universal Worker method {method!r}")


class UniversalWorkerClient:
    """Reconnectable client; each call uses an independent TCP connection."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float = 60.0,
        tls: TlsClientConfig | None = None,
        allow_plaintext_loopback: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.tls = tls
        require_tls_for_endpoint(
            format_endpoint(host, port),
            tls_configured=tls is not None,
            allow_plaintext_loopback=allow_plaintext_loopback,
            transport_name="legacy Universal Worker experiment transport",
        )

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = UniversalWorkerRequest(method=method, payload=payload or {})
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                self.host,
                self.port,
                ssl=self.tls.ssl_context() if self.tls is not None else None,
                server_hostname=(self.tls.expected_server_name if self.tls is not None else None),
            ),
            timeout=self.timeout_seconds,
        )
        try:
            if self.tls is not None:
                tls_object = writer.get_extra_info("ssl_object")
                peer_der = tls_object.getpeercert(binary_form=True) if tls_object else None
                self.tls.validate_peer_der(peer_der)
            await asyncio.wait_for(
                _write_frame(writer, _encode_model(request)),
                timeout=self.timeout_seconds,
            )
            raw = await asyncio.wait_for(_read_frame(reader), timeout=self.timeout_seconds)
            response = UniversalWorkerResponse.model_validate_json(raw)
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
        if not response.accepted:
            raise RuntimeError(response.error)
        return response.payload

    async def negotiate(self, version: WorkerProtocolVersion) -> WorkerProtocolVersion:
        payload = await self.call("negotiate", version.model_dump(mode="json"))
        return WorkerProtocolVersion.model_validate(payload)

    async def identity(self) -> WorkerIdentity:
        return WorkerIdentity.model_validate(await self.call("identity"))

    async def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities.model_validate(await self.call("capabilities"))

    async def benchmark(self) -> WorkerBenchmarkProfile:
        return WorkerBenchmarkProfile.model_validate(await self.call("benchmark"))

    async def submit(self, job: WorkerJob) -> WorkerJobResult:
        return WorkerJobResult.model_validate(
            await self.call("submit", job.model_dump(mode="json"))
        )

    async def cancel(self, request_id: str) -> bool:
        payload = await self.call("cancel", {"request_id": request_id})
        return bool(payload["cancelled"])

    async def heartbeat(self) -> dict[str, Any]:
        return await self.call("heartbeat")

    async def validate_shard(self, path: Path, sha256: str) -> bool:
        payload = await self.call("validate_shard", {"path": str(path), "sha256": sha256})
        return bool(payload["valid"])

    async def shutdown(self) -> bool:
        payload = await self.call("shutdown")
        return bool(payload["clean_shutdown"])
