from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from swarm_inference.engines.interfaces import (
    Deployment,
    ExecutionPlan,
    InferenceRequest,
    PhasePlan,
)
from swarm_inference.engines.llamacpp_rpc import (
    LlamaCppRuntimeManifest,
    LocalLlamaCppLifecycle,
)
from swarm_inference.host import split_endpoint


class _Processes:
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.stops: list[str] = []

    def start(self, **kwargs: Any) -> SimpleNamespace:
        self.starts.append(kwargs)
        return SimpleNamespace(process=SimpleNamespace(pid=100 + len(self.starts)))

    def stop_deployment(self, deployment_id: str) -> dict[str, int]:
        self.stops.append(deployment_id)
        return {}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> LlamaCppRuntimeManifest:
    server = tmp_path / "llama-server"
    rpc = tmp_path / "rpc-server"
    server.write_bytes(b"pinned-server")
    rpc.write_bytes(b"pinned-rpc")
    return LlamaCppRuntimeManifest(
        commit="pinned-commit",
        build_id="build-a",
        platform="test",
        server_binary=server,
        server_sha256=_sha(server),
        rpc_server_binary=rpc,
        rpc_server_sha256=_sha(rpc),
        build_flags={"GGML_RPC": True},
        device_support=("CPU", "CUDA"),
    )


def _plan(*, model_path: Path | None = None, rpc_endpoint: str | None = None) -> ExecutionPlan:
    roles = {"owner": "critical_path_stage", "compute": "tensor_rpc_compute"}
    parameters: dict[str, Any] = {
        "model_paths": [str(model_path)] if model_path is not None else [],
        "context_size": 4096,
        "parallel": 2,
        "tensor_split": {"owner": 0.6, "compute": 0.4},
    }
    if rpc_endpoint is not None:
        parameters.update(
            {
                "rpc_endpoints": {"compute": rpc_endpoint},
                "tensor_split_order": ["compute", "owner"],
                "tensor_split_values": [0.4, 0.6],
            }
        )
    return ExecutionPlan(
        plan_id="llama-plan",
        engine_id="llamacpp-rpc",
        model_fingerprint="sha256:" + "1" * 64,
        execution_identity="sha256:" + "2" * 64,
        objective="capacity",
        topology="llamacpp-rpc-2-host",
        worker_roles=roles,
        engine_parameters=parameters,
        prefill_plan=PhasePlan(phase="prefill", worker_roles=roles),
        decode_plan=PhasePlan(phase="decode", worker_roles=roles),
        predicted_ttft_ms=10,
        predicted_decode_tokens_s=5,
        predicted_aggregate_tokens_s=5,
        score=5,
    )


@pytest.mark.asyncio
async def test_llamacpp_workers_start_rpc_before_private_submission_owner(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    compute_processes = _Processes()
    compute = LocalLlamaCppLifecycle(
        manifest=manifest,
        processes=compute_processes,  # type: ignore[arg-type]
        worker_id="compute",
        bind_host="127.0.0.1",
    )
    compute_deployment = await compute.prepare(_plan())
    rpc_endpoint = compute_deployment.endpoints["compute"]
    compute_start = compute_processes.starts[0]
    assert compute_start["executable"] == manifest.rpc_server_binary
    assert compute_start["role"] == "tensor-rpc-server"
    assert compute_start["arguments"][:2] == ("-H", "127.0.0.1")

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test")
    owner_processes = _Processes()
    owner = LocalLlamaCppLifecycle(
        manifest=manifest,
        processes=owner_processes,  # type: ignore[arg-type]
        worker_id="owner",
        bind_host="127.0.0.1",
    )
    owner_deployment = await owner.prepare(_plan(model_path=model, rpc_endpoint=rpc_endpoint))
    owner_start = owner_processes.starts[0]
    arguments = owner_start["arguments"]
    assert owner_start["executable"] == manifest.server_binary
    assert arguments[arguments.index("--host") + 1] == "127.0.0.1"
    metered_endpoint = arguments[arguments.index("--rpc") + 1]
    assert metered_endpoint != rpc_endpoint
    assert split_endpoint(metered_endpoint)[0] == "127.0.0.1"
    assert arguments[arguments.index("--tensor-split") + 1] == "0.4,0.6"
    assert arguments[arguments.index("--n-gpu-layers") + 1] == "999"
    assert owner_deployment.metadata["rpc_endpoints"] == {"compute": rpc_endpoint}
    assert owner_deployment.metadata["network_boundary"] == "swarm-managed-tls-proxy"
    assert owner_deployment.metadata["network_metering"]["message_count"] is None

    await owner.unload(owner_deployment)
    await compute.unload(compute_deployment)
    assert owner_processes.stops == [owner_deployment.deployment_id]
    assert compute_processes.stops == [compute_deployment.deployment_id]


@pytest.mark.asyncio
async def test_llamacpp_rpc_role_rejects_a_build_without_rpc_support(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path).model_copy(update={"build_flags": {"GGML_RPC": False}})
    lifecycle = LocalLlamaCppLifecycle(
        manifest=manifest,
        processes=_Processes(),  # type: ignore[arg-type]
        worker_id="compute",
        bind_host="127.0.0.1",
    )

    with pytest.raises(RuntimeError, match="not built with GGML_RPC"):
        await lifecycle.prepare(_plan())


@pytest.mark.asyncio
async def test_llamacpp_completion_is_forwarded_as_real_sse_tokens(tmp_path: Path) -> None:
    received: dict[str, Any] = {}

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        headers = await reader.readuntil(b"\r\n\r\n")
        content_length = next(
            int(line.split(b":", 1)[1])
            for line in headers.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        )
        received.update(json.loads(await reader.readexactly(content_length)))
        events = (
            b'data: {"content":"a","tokens":[7],"stop":false}\n\n',
            b'data: {"content":"b","tokens":[8],"stop":true,'
            b'"timings":{"predicted_per_second":42}}\n\n',
        )
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        for event in events:
            writer.write(f"{len(event):x}\r\n".encode("ascii") + event + b"\r\n")
            await writer.drain()
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    plan = _plan()
    deployment = Deployment(
        deployment_id="deployment",
        engine_id=plan.engine_id,
        execution_identity=plan.execution_identity,
        plan=plan,
        ready=True,
        endpoints={"owner": f"http://127.0.0.1:{port}"},
    )
    lifecycle = LocalLlamaCppLifecycle(
        manifest=_manifest(tmp_path),
        processes=_Processes(),  # type: ignore[arg-type]
        worker_id="owner",
    )
    try:
        events = [
            event
            async for event in lifecycle.submit(
                deployment,
                InferenceRequest(request_id="request", prompt="hello", max_new_tokens=2),
            )
        ]
    finally:
        server.close()
        await server.wait_closed()

    assert [event.event_type for event in events] == [
        "started",
        "token",
        "token",
        "completed",
    ]
    assert [event.token_id for event in events[1:3]] == [7, 8]
    assert "".join(event.text for event in events) == "ab"
    assert events[-1].telemetry["timings"]["predicted_per_second"] == 42
    network = events[-1].telemetry["network"]
    assert network["bytes_sent"] == 0
    assert network["bytes_received"] == 0
    assert network["connection_count"] == 0
    assert network["transfer_count"] == 0
    assert network["message_count"] is None
    assert network["runtime_duration_s"] >= 0
    assert network["generated_token_count"] == 2
    assert network["bytes_per_generated_token"] == 0.0
    assert network["provenance"] == "byte-transparent Swarm TCP metering proxy"
    assert received["stream"] is True
    assert received["return_tokens"] is True
    assert received["cache_prompt"] is False
