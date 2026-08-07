"""Pinned llama.cpp GGUF compatibility engine with managed RPC lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field

from swarm_inference.config.models import StrictModel
from swarm_inference.engines.cost_model import PlanCostInputs, score_costs, stable_plan_id
from swarm_inference.engines.interfaces import (
    ClusterCapabilities,
    Deployment,
    EngineSupportReport,
    EngineSupportStatus,
    ExecutionEngineCapability,
    ExecutionPlan,
    ExecutionRequest,
    InferenceEvent,
    InferenceRequest,
    PhasePlan,
    WorkerExecutionCapability,
)
from swarm_inference.engines.topology import TopologyDomain, summarize_network_path
from swarm_inference.host import format_endpoint
from swarm_inference.model.descriptor import ResolvedModelDescriptor
from swarm_inference.runtime.engine_processes import (
    EngineProcessManager,
    require_private_bind_host,
    sha256_file,
)
from swarm_inference.security.tls import TlsClientConfig, TlsServerConfig
from swarm_inference.transport.tcp_meter import TcpMeteringProxy


def _snapshot_int(snapshot: Mapping[str, int | float | None], key: str) -> int:
    value = snapshot.get(key)
    if value is None:
        return 0
    return int(value)


def _llama_capability(worker: WorkerExecutionCapability) -> ExecutionEngineCapability:
    capability = worker.engine("llamacpp-rpc")
    assert capability is not None
    return capability


class LlamaCppRuntimeManifest(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str
    build_id: str
    platform: str
    server_binary: Path
    server_sha256: str
    rpc_server_binary: Path
    rpc_server_sha256: str
    build_flags: dict[str, bool | str] = Field(default_factory=dict)
    device_support: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ("text-generation",)
    unsupported_features: tuple[str, ...] = ()

    @property
    def rpc_enabled(self) -> bool:
        """Whether the pinned build positively records GGML RPC support."""

        value = next(
            (item for key, item in self.build_flags.items() if key.casefold() == "ggml_rpc"),
            False,
        )
        if isinstance(value, bool):
            return value
        return value.strip().casefold() in {"1", "on", "true", "yes"}

    def verify(self) -> None:
        for path, expected in (
            (self.server_binary, self.server_sha256),
            (self.rpc_server_binary, self.rpc_server_sha256),
        ):
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            if sha256_file(resolved) != expected.removeprefix("sha256:").lower():
                raise RuntimeError(f"pinned llama.cpp binary hash mismatch: {resolved}")


class LlamaCppArchitectureProbe(StrictModel):
    """Evidence that one pinned executable contains a GGUF architecture ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supported_identifiers: tuple[str, ...]
    inspected_binary: Path
    binary_sha256: str
    mechanism: str = "bounded-binary-identifier-scan"


def _contains_identifier(path: Path, identifier: str, *, chunk_size: int = 1 << 20) -> bool:
    """Scan a binary with bounded memory and token-boundary checks."""

    needle = identifier.casefold().encode("ascii")
    if not needle or len(needle) > 128:
        return False
    overlap = len(needle) + 1
    tail = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            data = (tail + block).lower()
            eof = not block
            scan_limit = len(data) if eof else max(0, len(data) - overlap)
            start = 0
            while (index := data.find(needle, start)) >= 0:
                if index >= scan_limit:
                    break
                before = data[index - 1 : index] if index else b""
                after_index = index + len(needle)
                after = data[after_index : after_index + 1]
                identifier_chars = b"abcdefghijklmnopqrstuvwxyz0123456789_"
                if (not before or before[0] not in identifier_chars) and (
                    not after or after[0] not in identifier_chars
                ):
                    return True
                start = index + 1
            if eof:
                break
            tail = data[-overlap:]
    return False


def probe_llamacpp_architectures(
    manifest: LlamaCppRuntimeManifest,
    identifiers: tuple[str, ...],
) -> LlamaCppArchitectureProbe:
    """Prove architecture support from the hash-verified llama-server binary.

    llama.cpp compiles its GGUF architecture dispatch identifiers into the
    model loader.  Scanning the pinned executable is deterministic, bounded,
    and avoids opening a many-gigabyte model merely to discover an unsupported
    architecture.
    """

    manifest.verify()
    binary = manifest.server_binary.expanduser().resolve()
    supported = tuple(
        sorted(
            {
                identifier.casefold()
                for identifier in identifiers
                if _contains_identifier(binary, identifier)
            }
        )
    )
    return LlamaCppArchitectureProbe(
        supported_identifiers=supported,
        inspected_binary=binary,
        binary_sha256="sha256:" + sha256_file(binary),
    )


def load_llamacpp_runtime_manifest(path: Path) -> LlamaCppRuntimeManifest:
    """Load a pinned manifest with binary paths relative to the manifest."""

    resolved = path.expanduser().resolve()
    manifest = LlamaCppRuntimeManifest.model_validate_json(resolved.read_text(encoding="utf-8"))
    manifest = manifest.model_copy(
        update={
            "server_binary": (
                manifest.server_binary
                if manifest.server_binary.is_absolute()
                else resolved.parent / manifest.server_binary
            ).resolve(),
            "rpc_server_binary": (
                manifest.rpc_server_binary
                if manifest.rpc_server_binary.is_absolute()
                else resolved.parent / manifest.rpc_server_binary
            ).resolve(),
        }
    )
    manifest.verify()
    return manifest


class LlamaCppLifecycle(Protocol):
    async def prepare(self, plan: ExecutionPlan) -> Deployment: ...

    def submit(
        self, deployment: Deployment, request: InferenceRequest
    ) -> AsyncIterator[InferenceEvent]: ...

    async def unload(self, deployment: Deployment) -> None: ...


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _wait_http(endpoint: str, process: Any, *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama.cpp exited during startup with {process.returncode}")
        try:
            with urllib.request.urlopen(endpoint + "/health", timeout=2) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"llama.cpp did not become healthy: {last_error}")


def _wait_tcp(host: str, port: int, process: Any, *, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama.cpp RPC server exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise TimeoutError(f"llama.cpp RPC server did not become ready: {last_error}")


async def _http_body_chunks(
    reader: asyncio.StreamReader,
    headers: dict[str, str],
) -> AsyncIterator[bytes]:
    transfer_encoding = headers.get("transfer-encoding", "").casefold()
    if "chunked" in transfer_encoding:
        while True:
            size_line = await reader.readline()
            if not size_line or len(size_line) > 128:
                raise RuntimeError("llama.cpp returned invalid HTTP chunk framing")
            try:
                size = int(size_line.split(b";", 1)[0].strip(), 16)
            except ValueError as exc:
                raise RuntimeError("llama.cpp returned an invalid HTTP chunk size") from exc
            if size < 0 or size > 8 * 1024 * 1024:
                raise RuntimeError("llama.cpp HTTP chunk exceeded the bounded size")
            if size == 0:
                while trailer := await reader.readline():
                    if trailer in {b"\r\n", b"\n"}:
                        break
                return
            payload = await reader.readexactly(size)
            if await reader.readexactly(2) != b"\r\n":
                raise RuntimeError("llama.cpp returned malformed HTTP chunk termination")
            yield payload
        return
    if raw_length := headers.get("content-length"):
        try:
            remaining = int(raw_length)
        except ValueError as exc:
            raise RuntimeError("llama.cpp returned an invalid content length") from exc
        if remaining < 0:
            raise RuntimeError("llama.cpp returned a negative content length")
        while remaining:
            payload = await reader.read(min(remaining, 64 * 1024))
            if not payload:
                raise RuntimeError("llama.cpp closed a bounded response early")
            remaining -= len(payload)
            yield payload
        return
    while payload := await reader.read(64 * 1024):
        yield payload


def _decode_sse_event(data_lines: list[bytes]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    payload = b"\n".join(data_lines)
    data_lines.clear()
    if payload == b"[DONE]":
        return None
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("llama.cpp returned malformed SSE JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("llama.cpp returned a non-object SSE event")
    return value


async def _post_json_sse(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> AsyncIterator[dict[str, Any]]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise RuntimeError("worker-owned llama.cpp endpoint must be loopback HTTP")
    host = parsed.hostname
    if not (host == "localhost" or host.startswith("127.") or host == "::1"):
        raise RuntimeError("worker-owned llama.cpp HTTP endpoint escaped loopback")
    port = parsed.port or 80
    target = (parsed.path.rstrip("/") + "/" + path.lstrip("/")) or "/"
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    host_header = f"[{host}]" if ":" in host else host
    request = (
        f"POST {target} HTTP/1.1\r\n"
        f"Host: {host_header}:{port}\r\n"
        "Accept: text/event-stream\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii") + body
    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_connection(
                host,
                port,
                limit=8 * 1024 * 1024,
            )
            writer.write(request)
            await writer.drain()
            try:
                header_block = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
                raise RuntimeError("llama.cpp returned invalid bounded HTTP headers") from exc
            if len(header_block) > 64 * 1024:
                raise RuntimeError("llama.cpp HTTP headers exceeded the bounded size")
            lines = header_block[:-4].split(b"\r\n")
            try:
                status = int(lines[0].split(b" ", 2)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError("llama.cpp returned an invalid HTTP status") from exc
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if b":" not in line:
                    raise RuntimeError("llama.cpp returned a malformed HTTP header")
                name, value = line.split(b":", 1)
                headers[name.decode("ascii").casefold()] = value.decode("latin-1").strip()
            if not 200 <= status < 300:
                detail = (await reader.read(4096)).decode("utf-8", errors="replace")
                raise RuntimeError(f"llama.cpp completion failed with HTTP {status}: {detail}")
            buffer = bytearray()
            data_lines: list[bytes] = []
            async for chunk in _http_body_chunks(reader, headers):
                buffer.extend(chunk)
                if len(buffer) > 8 * 1024 * 1024 and b"\n" not in buffer:
                    raise RuntimeError("llama.cpp SSE line exceeded the bounded size")
                while (newline := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:newline]).removesuffix(b"\r")
                    del buffer[: newline + 1]
                    if not line:
                        if event := _decode_sse_event(data_lines):
                            yield event
                    elif line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip(b" "))
            if buffer:
                line = bytes(buffer).removesuffix(b"\r")
                if line.startswith(b"data:"):
                    data_lines.append(line[5:].lstrip(b" "))
            if event := _decode_sse_event(data_lines):
                yield event
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f"llama.cpp streaming request failed: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()


class LocalLlamaCppLifecycle:
    """Worker-local owner or tensor-RPC process lifecycle."""

    def __init__(
        self,
        *,
        manifest: LlamaCppRuntimeManifest,
        processes: EngineProcessManager,
        worker_id: str,
        bind_host: str = "127.0.0.1",
        tls_server: TlsServerConfig | None = None,
        tls_client: TlsClientConfig | None = None,
    ) -> None:
        self.manifest = manifest
        self.processes = processes
        self.worker_id = worker_id
        self.bind_host = require_private_bind_host(bind_host)
        self.tls_server = tls_server
        self.tls_client = tls_client
        self._metering_proxies: dict[str, list[TcpMeteringProxy]] = {}
        self.manifest.verify()

    def _meter_snapshot(self, deployment_id: str) -> dict[str, int | float | None]:
        snapshots = [proxy.snapshot() for proxy in self._metering_proxies.get(deployment_id, [])]
        return {
            "bytes_sent": sum(_snapshot_int(item, "bytes_sent") for item in snapshots),
            "bytes_received": sum(_snapshot_int(item, "bytes_received") for item in snapshots),
            "connection_count": sum(_snapshot_int(item, "connection_count") for item in snapshots),
            "transfer_count": sum(_snapshot_int(item, "transfer_count") for item in snapshots),
            "connection_failures": sum(
                _snapshot_int(item, "connection_failures") for item in snapshots
            ),
            "runtime_duration_s": max(
                (float(item["runtime_duration_s"] or 0.0) for item in snapshots),
                default=0.0,
            ),
            "message_count": None,
        }

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        try:
            role = plan.worker_roles[self.worker_id]
        except KeyError as exc:
            raise RuntimeError("llama.cpp plan does not assign this worker") from exc
        deployment_id = f"llamacpp-{uuid4().hex}"
        if role == "tensor_rpc_compute":
            if not self.manifest.rpc_enabled:
                raise RuntimeError("pinned llama.cpp runtime was not built with GGML_RPC")
            engine_host = "127.0.0.1"
            port = _free_port(engine_host)
            engine_endpoint = format_endpoint(engine_host, port)
            rpc_arguments = ("-H", engine_host, "-p", str(port))
            managed = await asyncio.to_thread(
                self.processes.start,
                deployment_id=deployment_id,
                role="tensor-rpc-server",
                executable=self.manifest.rpc_server_binary,
                expected_sha256=self.manifest.rpc_server_sha256,
                arguments=rpc_arguments,
                ready=lambda process: _wait_tcp(engine_host, port, process),
            )
            proxy = TcpMeteringProxy(
                listen_endpoint=format_endpoint(self.bind_host, 0),
                upstream_endpoint=engine_endpoint,
                inbound_tls=self.tls_server,
            )
            try:
                endpoint = await proxy.start()
            except BaseException:
                await asyncio.to_thread(self.processes.stop_deployment, deployment_id)
                raise
            self._metering_proxies[deployment_id] = [proxy]
            return Deployment(
                deployment_id=deployment_id,
                engine_id=plan.engine_id,
                execution_identity=plan.execution_identity,
                plan=plan,
                ready=True,
                endpoints={self.worker_id: endpoint},
                process_ids={self.worker_id: managed.process.pid},
                metadata={
                    "runtime_commit": self.manifest.commit,
                    "rpc_server_sha256": self.manifest.rpc_server_sha256,
                    "role": role,
                    "transport_confidentiality": self.tls_server is not None,
                    "transport_authentication": self.tls_server is not None,
                    "network_metering": self._meter_snapshot(deployment_id),
                    "network_boundary": "swarm-managed-tls-proxy",
                },
            )
        if role in {"idle", "background_replica", "storage_cache", "verification"}:
            raise RuntimeError("llama.cpp cannot prepare an inactive worker role")
        model_paths = [
            Path(item)
            for item in plan.engine_parameters.get("model_paths", [])
            if str(item).lower().endswith(".gguf")
        ]
        if not model_paths or not all(item.is_file() for item in model_paths):
            raise FileNotFoundError("selected GGUF files have not been acquired")
        http_host = "127.0.0.1"
        port = _free_port(http_host)
        endpoint = f"http://{http_host}:{port}"
        arguments: tuple[str, ...] = (
            "--model",
            str(model_paths[0].resolve()),
            "--host",
            http_host,
            "--port",
            str(port),
            "--ctx-size",
            str(plan.engine_parameters.get("context_size", 2048)),
            "--parallel",
            str(plan.engine_parameters.get("parallel", 1)),
        )
        rpc_endpoints = plan.engine_parameters.get("rpc_endpoints", {})
        metering_proxies: list[TcpMeteringProxy] = []
        if rpc_endpoints:
            if not isinstance(rpc_endpoints, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in rpc_endpoints.items()
            ):
                raise ValueError("llama.cpp RPC endpoints must be a worker/address mapping")
            split_values = plan.engine_parameters.get("tensor_split_values", [])
            if not isinstance(split_values, list) or len(split_values) != len(rpc_endpoints) + 1:
                raise ValueError("llama.cpp distributed plan has an invalid tensor split")
            metered_rpc_endpoints: dict[str, str] = {}
            try:
                for worker_id, remote_endpoint in rpc_endpoints.items():
                    proxy = TcpMeteringProxy(
                        listen_endpoint="127.0.0.1:0",
                        upstream_endpoint=remote_endpoint,
                        outbound_tls=self.tls_client,
                    )
                    metered_rpc_endpoints[worker_id] = await proxy.start()
                    metering_proxies.append(proxy)
            except BaseException:
                await asyncio.gather(
                    *(proxy.close() for proxy in metering_proxies),
                    return_exceptions=True,
                )
                raise
            self._metering_proxies[deployment_id] = metering_proxies
            arguments += (
                "--rpc",
                ",".join(str(value) for value in metered_rpc_endpoints.values()),
                "--split-mode",
                "layer",
                "--tensor-split",
                ",".join(f"{float(value):.12g}" for value in split_values),
                "--n-gpu-layers",
                "999",
            )
        try:
            managed = await asyncio.to_thread(
                self.processes.start,
                deployment_id=deployment_id,
                role="model-server",
                executable=self.manifest.server_binary,
                expected_sha256=self.manifest.server_sha256,
                arguments=arguments,
                ready=lambda process: _wait_http(endpoint, process),
            )
        except BaseException:
            self._metering_proxies.pop(deployment_id, None)
            await asyncio.gather(
                *(proxy.close() for proxy in metering_proxies),
                return_exceptions=True,
            )
            raise
        return Deployment(
            deployment_id=deployment_id,
            engine_id=plan.engine_id,
            execution_identity=plan.execution_identity,
            plan=plan,
            ready=True,
            endpoints={self.worker_id: endpoint},
            process_ids={self.worker_id: managed.process.pid},
            metadata={
                "runtime_commit": self.manifest.commit,
                "server_sha256": self.manifest.server_sha256,
                "transport_confidentiality": bool(rpc_endpoints) and self.tls_client is not None,
                "transport_authentication": bool(rpc_endpoints) and self.tls_client is not None,
                "network_metering": self._meter_snapshot(deployment_id),
                "network_boundary": (
                    "swarm-managed-tls-proxy" if rpc_endpoints else "process-local"
                ),
                "rpc_endpoints": dict(rpc_endpoints),
            },
        )

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        endpoint = next(iter(deployment.endpoints.values()))
        yield InferenceEvent(event_type="started", request_id=request.request_id, sequence_number=0)
        meter_before = self._meter_snapshot(deployment.deployment_id)
        request_started = time.monotonic()
        sequence_number = 0
        terminal: dict[str, Any] | None = None
        async for result in _post_json_sse(
            endpoint,
            "/completion",
            {
                "prompt": request.prompt,
                "n_predict": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": 1 if request.temperature == 0 else 40,
                "top_p": 1.0,
                "seed": request.seed,
                "return_tokens": True,
                "n_probs": 1,
                "stream": True,
                "cache_prompt": False,
            },
            3600.0,
        ):
            tokens = result.get("tokens", [])
            if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
                raise RuntimeError("llama.cpp streaming response contained invalid token IDs")
            content = str(result.get("content", ""))
            for index, token in enumerate(tokens):
                sequence_number += 1
                yield InferenceEvent(
                    event_type="token",
                    request_id=request.request_id,
                    sequence_number=sequence_number,
                    token_id=int(token),
                    text=content if index == len(tokens) - 1 else "",
                )
            if result.get("stop") is True:
                terminal = result
        if terminal is None:
            raise RuntimeError("llama.cpp stream ended without a verified terminal event")
        meter_after = self._meter_snapshot(deployment.deployment_id)
        bytes_sent = _snapshot_int(meter_after, "bytes_sent") - _snapshot_int(
            meter_before, "bytes_sent"
        )
        bytes_received = _snapshot_int(meter_after, "bytes_received") - _snapshot_int(
            meter_before, "bytes_received"
        )
        generated_tokens = sequence_number
        yield InferenceEvent(
            event_type="completed",
            request_id=request.request_id,
            sequence_number=sequence_number + 1,
            telemetry={
                "timings": terminal.get("timings", {}),
                "network": {
                    "bytes_sent": bytes_sent,
                    "bytes_received": bytes_received,
                    "connection_count": _snapshot_int(meter_after, "connection_count")
                    - _snapshot_int(meter_before, "connection_count"),
                    "transfer_count": _snapshot_int(meter_after, "transfer_count")
                    - _snapshot_int(meter_before, "transfer_count"),
                    "message_count": None,
                    "runtime_duration_s": time.monotonic() - request_started,
                    "generated_token_count": generated_tokens,
                    "bytes_per_generated_token": (
                        (bytes_sent + bytes_received) / generated_tokens
                        if generated_tokens
                        else None
                    ),
                    "provenance": "byte-transparent Swarm TCP metering proxy",
                },
            },
        )

    async def unload(self, deployment: Deployment) -> None:
        proxies = self._metering_proxies.pop(deployment.deployment_id, [])
        await asyncio.gather(*(proxy.close() for proxy in proxies), return_exceptions=True)
        await asyncio.to_thread(self.processes.stop_deployment, deployment.deployment_id)


def _runtime_identity(
    model: ResolvedModelDescriptor,
    runtime_records: tuple[dict[str, Any], ...],
    execution_parameters: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "model": model.content_fingerprint,
            "gguf": [item.sha256 or item.etag for item in model.files],
            "runtime": runtime_records,
            "execution_parameters": execution_parameters,
            "quantization": model.quantization,
            "tokenizer": model.tokenizer_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_record(
    capability: ExecutionEngineCapability,
    *,
    role: str,
    memory_fraction: float,
) -> dict[str, Any]:
    return {
        "role": role,
        "memory_fraction": memory_fraction,
        "revision": capability.runtime_revision,
        "binary_hashes": capability.binary_hashes,
        "devices": [
            {
                "identity": device.uuid or f"{device.device_id}:{device.name}",
                "type": device.device_type,
                "runtime": device.runtime_version,
                "driver": device.driver_version,
            }
            for device in capability.devices
        ],
    }


class LlamaCppRpcEngine:
    engine_id = "llamacpp-rpc"

    def __init__(self, *, lifecycle: LlamaCppLifecycle | None = None) -> None:
        self.lifecycle = lifecycle
        self._acquired: dict[str, tuple[Path, ...]] = {}

    def bind_acquired_model(
        self, descriptor: ResolvedModelDescriptor, paths: tuple[Path, ...]
    ) -> None:
        if len(paths) != len(descriptor.files):
            raise ValueError("acquired GGUF path count differs from its immutable descriptor")
        gguf_paths = tuple(
            path.resolve()
            for file, path in zip(descriptor.files, paths, strict=True)
            if file.relative_path.lower().endswith(".gguf")
        )
        if not gguf_paths:
            raise ValueError("acquired descriptor contains no GGUF model file")
        self._acquired[descriptor.content_fingerprint] = gguf_paths

    def probe(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        if model.format != "gguf":
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_FORMAT,
                reason="llama.cpp compatibility execution consumes GGUF artifacts",
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with GGML RPC",
            )
        if not (model.architecture_raw or model.architecture):
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                reason=(
                    "GGUF architecture metadata is missing; the runtime cannot prove "
                    "which model loader is required"
                ),
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with an advertised GGUF architecture",
            )
        if model.architecture_source != "gguf.general.architecture":
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                reason=(
                    "exact GGUF general.architecture metadata was not available; "
                    "source-model configuration is insufficient to prove the loader "
                    "required by the selected GGUF artifact"
                ),
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with an exact GGUF architecture loader",
                required_features=("gguf.general.architecture",),
            )
        # Engine compatibility is exact even when the product registry groups
        # several upstream identifiers into one architecture family.
        required_architectures = {(model.architecture_raw or "").strip().casefold()}
        workers = []
        broken = []
        architecture_rejected: list[str] = []
        feature_rejected: dict[str, tuple[str, ...]] = {}
        for worker in cluster.workers_for_engine(self.engine_id):
            capability = worker.engine(self.engine_id)
            assert capability is not None
            if not capability.runtime_revision or not capability.binary_hashes:
                broken.append(worker.worker_id)
            elif not required_architectures.intersection(
                item.casefold() for item in capability.model_architectures
            ):
                architecture_rejected.append(worker.worker_id)
            elif rejected := tuple(
                sorted(set(model.features).intersection(capability.unsupported_features))
            ):
                feature_rejected[worker.worker_id] = rejected
            elif "gguf" in {item.lower() for item in capability.formats}:
                workers.append(worker)
        if not workers:
            if architecture_rejected:
                return EngineSupportReport(
                    engine_id=self.engine_id,
                    status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                    reason=(
                        "the installed pinned llama.cpp runtime did not advertise any "
                        "of the required GGUF loader identifiers: "
                        + ", ".join(sorted(required_architectures))
                    ),
                    model_architecture=model.architecture,
                    model_format=model.format,
                    required_runtime="pinned llama.cpp architecture loader",
                    required_features=tuple(sorted(required_architectures)),
                )
            if feature_rejected:
                unsupported = tuple(
                    sorted({item for values in feature_rejected.values() for item in values})
                )
                return EngineSupportReport(
                    engine_id=self.engine_id,
                    status=EngineSupportStatus.UNSUPPORTED_ARCHITECTURE,
                    reason="the GGUF requires features rejected by the pinned runtime: "
                    + ", ".join(unsupported),
                    model_architecture=model.architecture,
                    model_format=model.format,
                    required_runtime="pinned llama.cpp architecture loader",
                    unsupported_features=unsupported,
                )
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=(
                    EngineSupportStatus.BROKEN_RUNTIME
                    if broken
                    else EngineSupportStatus.MISSING_RUNTIME
                ),
                reason=(
                    "advertised llama.cpp runtime lacks pinned revision/hash evidence"
                    if broken
                    else "no worker advertises a pinned GGUF-compatible llama.cpp runtime"
                ),
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with GGML RPC",
            )
        required = int(model.weight_bytes * 1.05)
        owners = [
            worker
            for worker in workers
            if "critical_path_stage" in set(_llama_capability(worker).roles)
        ]
        if not owners:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.MISSING_DEVICE_CAPABILITY,
                reason="no llama.cpp worker advertises the model-server owner role",
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with a model-server role",
            )
        rpc_workers = [
            worker
            for worker in workers
            if "tensor_rpc_compute" in set(_llama_capability(worker).roles)
        ]
        memory_by_worker = {
            worker.worker_id: sum(
                device.usable_memory_bytes for device in _llama_capability(worker).devices
            )
            for worker in workers
        }
        feasible_ids: set[str] = {
            owner.worker_id for owner in owners if memory_by_worker[owner.worker_id] >= required
        }
        for owner in owners:
            cumulative = memory_by_worker[owner.worker_id]
            selected_ids = {owner.worker_id}
            selected_nodes = {owner.node_id}
            for worker in rpc_workers:
                if worker.node_id in selected_nodes:
                    continue
                cumulative += memory_by_worker[worker.worker_id]
                selected_ids.add(worker.worker_id)
                selected_nodes.add(worker.node_id)
                if cumulative >= required:
                    feasible_ids.update(selected_ids)
                    break
        if not feasible_ids:
            return EngineSupportReport(
                engine_id=self.engine_id,
                status=EngineSupportStatus.INSUFFICIENT_MEMORY,
                reason=(
                    "no valid local or distinct-host llama.cpp RPC topology can admit "
                    "the selected GGUF with runtime headroom"
                ),
                supported_worker_ids=tuple(item.worker_id for item in workers),
                model_architecture=model.architecture,
                model_format=model.format,
                required_runtime="pinned llama.cpp with sufficient aggregate memory",
            )
        proven = sorted(
            required_architectures.intersection(
                identifier.casefold()
                for worker in workers
                for identifier in worker.engine(self.engine_id).model_architectures  # type: ignore[union-attr]
            )
        )
        return EngineSupportReport(
            engine_id=self.engine_id,
            status=EngineSupportStatus.SUPPORTED,
            reason=("pinned llama.cpp runtime supports " + ", ".join(proven) + " GGUF models"),
            supported_worker_ids=tuple(sorted(feasible_ids)),
            model_architecture=model.architecture,
            model_format=model.format,
            required_runtime="pinned llama.cpp with GGML RPC",
            required_features=tuple(proven),
        )

    def probe_model_support(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
    ) -> EngineSupportReport:
        return self.probe(model, cluster)

    async def candidate_plans(
        self,
        model: ResolvedModelDescriptor,
        cluster: ClusterCapabilities,
        request: ExecutionRequest,
    ) -> list[ExecutionPlan]:
        workers = [
            worker
            for worker in cluster.workers_for_engine(self.engine_id)
            if not {worker.worker_id, worker.node_id}.intersection(request.excluded_nodes)
            and (
                not request.requested_nodes
                or bool({worker.worker_id, worker.node_id}.intersection(request.requested_nodes))
            )
        ]
        workers.sort(
            key=lambda worker: (
                -max(
                    (device.measured_decode_tokens_s or 0)
                    for device in _llama_capability(worker).devices
                ),
                worker.queue_depth,
                worker.worker_id,
            )
        )
        owner_workers = [
            worker
            for worker in workers
            if "critical_path_stage" in set(_llama_capability(worker).roles)
        ]
        rpc_workers = [
            worker
            for worker in workers
            if "tensor_rpc_compute" in set(_llama_capability(worker).roles)
        ]
        required = int(model.weight_bytes * 1.05)
        plans: list[ExecutionPlan] = []
        local_rates: dict[str, float] = {}
        for worker in owner_workers:
            capability = worker.engine(self.engine_id)
            assert capability is not None
            memory = sum(device.usable_memory_bytes for device in capability.devices)
            if memory < required:
                continue
            rate = max((device.measured_decode_tokens_s or 1.0) for device in capability.devices)
            rate /= 1 + worker.queue_depth
            local_rates[worker.worker_id] = rate
            roles = {worker.worker_id: "critical_path_stage"}
            identity = _runtime_identity(
                model,
                (
                    _runtime_record(
                        capability,
                        role="critical_path_stage",
                        memory_fraction=1.0,
                    ),
                ),
                {
                    "topology": "local",
                    "context": request.max_context_tokens,
                    "parallel": request.concurrency,
                },
            )
            costs = score_costs(
                PlanCostInputs(
                    measured_decode_tokens_s=max(
                        (device.measured_decode_tokens_s or 1.0) for device in capability.devices
                    ),
                    queue_depth=worker.queue_depth,
                    reliability=worker.reliability,
                    usable_memory_bytes=memory,
                    required_memory_bytes=required,
                    resident_model_bytes=(
                        model.weight_bytes
                        if model.content_fingerprint in worker.resident_model_fingerprints
                        else 0
                    ),
                    concurrency=request.concurrency,
                    request_priority=request.priority,
                    network_latency_ms=0.0,
                    network_jitter_ms=0.0,
                    messages_per_token=0.0,
                    bytes_per_token=0.0,
                    serial_waits_per_token=0.0,
                ),
                objective=request.objective,
            )
            topology = "llamacpp-local-hybrid"
            plan_identity = {
                "model": model.content_fingerprint,
                "execution": identity,
                "topology": topology,
                "worker": worker.worker_id,
                "context": request.max_context_tokens,
                "parallel": request.concurrency,
            }
            plans.append(
                ExecutionPlan(
                    plan_id=stable_plan_id("llamacpp-local", plan_identity),
                    engine_id=self.engine_id,
                    model_fingerprint=model.content_fingerprint,
                    execution_identity=identity,
                    objective=request.objective,
                    topology=topology,
                    worker_roles=roles,
                    idle_workers={
                        item.worker_id: "no positive single-request utility"
                        for item in workers
                        if item.worker_id != worker.worker_id
                    },
                    prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                    decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                    predicted_ttft_ms=costs.predicted_ttft_ms,
                    predicted_decode_tokens_s=costs.predicted_decode_tokens_s,
                    predicted_aggregate_tokens_s=costs.predicted_aggregate_tokens_s,
                    predicted_network_bytes=0,
                    predicted_messages_per_token=0.0,
                    predicted_bytes_per_token=0.0,
                    predicted_serial_waits_per_token=0.0,
                    number_of_wan_stage_boundaries=0,
                    persistent_connections=False,
                    network_cost_confidence="measured",
                    network_cost_provenance="no network boundary",
                    required_memory_bytes=required,
                    score=costs.score,
                    explanation=("one managed llama.cpp process owns the complete GGUF",),
                    engine_parameters={
                        "model_paths": [
                            str(item) for item in self._acquired.get(model.content_fingerprint, ())
                        ],
                        "context_size": request.max_context_tokens,
                        "parallel": request.concurrency,
                        "quantization": model.quantization,
                        "cost_components": costs.components,
                        "unmeasured_inputs": costs.unmeasured_inputs,
                    },
                )
            )
        distributed_candidates: list[tuple[Any, list[Any], int]] = []
        for owner in owner_workers:
            owner_capability = owner.engine(self.engine_id)
            assert owner_capability is not None
            owner_memory = sum(device.usable_memory_bytes for device in owner_capability.devices)
            if owner_memory <= 0:
                continue
            selected = [(owner, owner_capability, owner_memory)]
            selected_nodes = {owner.node_id}
            cumulative = owner_memory
            for worker in rpc_workers:
                if worker.node_id in selected_nodes:
                    continue
                capability = worker.engine(self.engine_id)
                assert capability is not None
                memory = sum(device.usable_memory_bytes for device in capability.devices)
                if memory <= 0:
                    continue
                selected.append((worker, capability, memory))
                selected_nodes.add(worker.node_id)
                cumulative += memory
                if cumulative >= required:
                    break
            if len(selected) >= 2 and cumulative >= required:
                distributed_candidates.append((owner, selected, cumulative))
        for _owner, selected, cumulative in distributed_candidates:
            rates = [
                max((device.measured_decode_tokens_s or 1.0) for device in capability.devices)
                for _, capability, _ in selected
            ]
            links = tuple(selected[0][0].link_to(worker.worker_id) for worker, _, _ in selected[1:])
            network = summarize_network_path(links)
            fractions = {worker.worker_id: memory / cumulative for worker, _, memory in selected}
            identity = _runtime_identity(
                model,
                tuple(
                    _runtime_record(
                        capability,
                        role=("critical_path_stage" if index == 0 else "tensor_rpc_compute"),
                        memory_fraction=fractions[worker.worker_id],
                    )
                    for index, (worker, capability, _memory) in enumerate(selected)
                ),
                {
                    "topology": "rpc-layer-split",
                    "context": request.max_context_tokens,
                    "parallel": request.concurrency,
                },
            )
            parallel_compute_ms = max(
                fraction / max(rate, 1e-9) * 1000
                for fraction, rate in zip(fractions.values(), rates, strict=True)
            )
            raw_distributed_rate = 1000 / max(parallel_compute_ms, 1e-9)
            distributed_rate = (
                1000
                / (
                    1000 / max(raw_distributed_rate, 0.001)
                    + float(network.aggregate_rtt_ms)
                    + float(network.aggregate_jitter_ms or 0.0)
                )
                if network.aggregate_rtt_ms is not None
                else None
            )
            best_local = max(local_rates.values(), default=0.0)
            if (
                request.require_distributed
                or request.objective == "capacity"
                or (distributed_rate is not None and distributed_rate > best_local)
            ):
                roles = {selected[0][0].worker_id: "critical_path_stage"}
                roles.update(
                    {worker.worker_id: "tensor_rpc_compute" for worker, _, _ in selected[1:]}
                )
                costs = score_costs(
                    PlanCostInputs(
                        measured_decode_tokens_s=raw_distributed_rate,
                        queue_depth=max(worker.queue_depth for worker, _, _ in selected),
                        reliability=min(worker.reliability for worker, _, _ in selected),
                        usable_memory_bytes=cumulative,
                        required_memory_bytes=required,
                        network_latency_ms=network.aggregate_rtt_ms,
                        network_jitter_ms=network.aggregate_jitter_ms,
                        messages_per_token=None,
                        bytes_per_token=None,
                        serial_waits_per_token=None,
                        concurrency=request.concurrency,
                        request_priority=request.priority,
                    ),
                    objective=request.objective,
                )
                topology = f"llamacpp-rpc-{len(selected)}-host"
                plan_identity = {
                    "model": model.content_fingerprint,
                    "execution": identity,
                    "topology": topology,
                    "tensor_split": fractions,
                    "context": request.max_context_tokens,
                    "parallel": request.concurrency,
                }
                plans.append(
                    ExecutionPlan(
                        plan_id=stable_plan_id("llamacpp-rpc", plan_identity),
                        engine_id=self.engine_id,
                        model_fingerprint=model.content_fingerprint,
                        execution_identity=identity,
                        objective=request.objective,
                        topology=topology,
                        worker_roles=roles,
                        idle_workers={
                            item.worker_id: "not required for feasible positive-utility tensor placement"
                            for item in workers
                            if item.worker_id not in roles
                        },
                        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
                        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
                        predicted_ttft_ms=costs.predicted_ttft_ms,
                        predicted_decode_tokens_s=costs.predicted_decode_tokens_s,
                        predicted_aggregate_tokens_s=costs.predicted_aggregate_tokens_s,
                        predicted_network_bytes=None,
                        predicted_messages_per_token=None,
                        predicted_bytes_per_token=None,
                        predicted_serial_waits_per_token=None,
                        number_of_wan_stage_boundaries=network.wan_boundaries,
                        persistent_connections=True,
                        network_cost_confidence=network.confidence.value,
                        network_cost_provenance=(
                            "llama.cpp protocol volume unmeasured; link metrics: "
                            + network.provenance
                        ),
                        required_memory_bytes=required,
                        score=costs.score,
                        explanation=(
                            "each selected host owns a non-zero tensor share and performs required compute",
                            "RPC lifecycle is owned by authenticated Swarm workers",
                            "llama.cpp private-protocol operation and byte counts remain "
                            "unmeasured until the metering transport observes execution",
                            "network domains: " + ", ".join(item.value for item in network.domains),
                            *(
                                (
                                    "fine-grained tensor RPC crosses a WAN link and is "
                                    "admitted only for required/capacity execution",
                                )
                                if TopologyDomain.WAN in network.domains
                                else ()
                            ),
                        ),
                        engine_parameters={
                            "tensor_split": fractions,
                            "model_paths": [
                                str(item)
                                for item in self._acquired.get(model.content_fingerprint, ())
                            ],
                            "context_size": request.max_context_tokens,
                            "parallel": request.concurrency,
                            "quantization": model.quantization,
                            "network_estimate_status": "unmeasured-engine-protocol-bytes",
                            "network_links": [item.model_dump(mode="json") for item in links],
                            "topology_domains": [item.value for item in network.domains],
                            "cost_components": costs.components,
                            "unmeasured_inputs": costs.unmeasured_inputs,
                        },
                    )
                )
        return plans

    async def prepare(self, plan: ExecutionPlan) -> Deployment:
        if plan.engine_id != self.engine_id:
            raise ValueError("llama.cpp engine cannot prepare another engine's plan")
        if self.lifecycle is None:
            raise RuntimeError("managed llama.cpp worker lifecycle is unavailable")
        return await self.lifecycle.prepare(plan)

    async def submit(
        self,
        deployment: Deployment,
        request: InferenceRequest,
    ) -> AsyncIterator[InferenceEvent]:
        if self.lifecycle is None:
            raise RuntimeError("managed llama.cpp worker lifecycle is unavailable")
        async for event in self.lifecycle.submit(deployment, request):
            yield event

    async def unload(self, deployment: Deployment) -> None:
        if self.lifecycle is None:
            raise RuntimeError("managed llama.cpp worker lifecycle is unavailable")
        await self.lifecycle.unload(deployment)


__all__ = [
    "LlamaCppArchitectureProbe",
    "LlamaCppLifecycle",
    "LlamaCppRpcEngine",
    "LlamaCppRuntimeManifest",
    "LocalLlamaCppLifecycle",
    "load_llamacpp_runtime_manifest",
    "probe_llamacpp_architectures",
]
