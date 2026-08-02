"""Process-isolated expert worker, lifecycle manager, and fault controls."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import platform
import secrets
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from swarm_inference.experiments.experiment_010.codecs import decode_array, encode_array
from swarm_inference.experiments.experiment_010.expert import (
    ExpertStore,
    npz_expert_loader,
    safetensors_expert_loader,
)
from swarm_inference.experiments.experiment_010.schemas import (
    ExpertExecutionMetadata,
    ExpertExecutionRequest,
    ExpertExecutionResponse,
    FailureType,
    ResultIntegrity,
    TransportCodec,
    WorkerBudget,
    WorkerManifest,
)
from swarm_inference.experiments.experiment_010.wire import (
    ExpertPacket,
    decode_packet,
    decode_request,
    encode_packet,
    encode_response,
    frame_with_length,
    read_length_frame,
)
from swarm_inference.protocol.checksums import sha256_bytes, sha256_file
from swarm_inference.worker.abi import (
    BackendAdapter,
    ResultClassification,
    TensorPayload,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
    WorkerProtocolVersion,
    tensor_payload_from_array,
)
from swarm_inference.worker.universal import UniversalWorkerClient


@dataclass(slots=True)
class WorkerFaultState:
    fault_type: FailureType | None = None
    remaining: int = 0
    fixed_delay_ms: float = 0.0
    random_delay_ms: float = 0.0
    storage_slowdown_ms: float = 0.0

    def consume(self) -> FailureType | None:
        if self.fault_type is None or self.remaining == 0:
            return None
        selected = self.fault_type
        if self.remaining > 0:
            self.remaining -= 1
            if self.remaining == 0:
                self.fault_type = None
        return selected


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class ExpertWorkerRuntime:
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.configuration = configuration
        self.worker_id = str(configuration["worker_id"])
        self.model_id = str(configuration["model_id"])
        self.model_revision = str(configuration["model_revision"])
        self.quantization_fingerprint = str(configuration["quantization_fingerprint"])
        self.model_fingerprint = str(configuration["model_fingerprint"])
        self.bridge_version = str(configuration.get("bridge_version", "experiment-010-v1"))
        self.memory_budget_bytes = int(configuration["memory_budget_bytes"])
        self.storage_directory = Path(str(configuration["storage_directory"])).resolve()
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        self.secret = base64.b64decode(str(configuration["signature_secret"]))
        self.started_ns = time.time_ns()
        self.requests_completed = 0
        self.failures = 0
        self.peak_rss_bytes = 0
        self.fault = WorkerFaultState()
        self.control_endpoint: str | None = None
        owned = {
            (int(item["layer_id"]), int(item["expert_id"]))
            for item in configuration["owned_experts"]
        }
        loader_type = str(configuration["loader_type"])
        if loader_type == "npz":
            files = {
                (int(item["layer_id"]), int(item["expert_id"])): Path(str(item["path"]))
                for item in configuration["owned_experts"]
            }
            loader = npz_expert_loader(files)
        elif loader_type == "safetensors":
            loader = safetensors_expert_loader(Path(str(configuration["model_path"])))
        else:
            raise ValueError(f"unsupported expert loader {loader_type!r}")
        self.store = ExpertStore(
            owned=owned,
            loader=loader,
            residency_budget_bytes=int(configuration["expert_residency_budget_bytes"]),
            cache_budget_bytes=int(configuration["cache_budget_bytes"]),
        )
        self._owned = owned
        self._owned_expert_bytes = sum(
            int(item.get("logical_bytes", 0)) for item in configuration["owned_experts"]
        )
        self._shutdown = asyncio.Event()
        self._responses: dict[str, tuple[ExpertExecutionResponse, np.ndarray]] = {}

    def manifest(self, endpoint: str) -> WorkerManifest:
        process = psutil.Process()
        memory = process.memory_info()
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(memory.rss))
        by_layer: dict[str, list[int]] = {}
        for layer_id, expert_id in sorted(self._owned):
            by_layer.setdefault(str(layer_id), []).append(expert_id)
        hashes = {
            f"{item['layer_id']}:{item['expert_id']}": str(item.get("content_hash", ""))
            for item in self.configuration["owned_experts"]
        }
        return WorkerManifest(
            worker_id=self.worker_id,
            process_id=os.getpid(),
            endpoint=endpoint,
            control_endpoint=self.control_endpoint,
            universal_worker_abi={
                "major": 1,
                "minor": 1,
                "job_role": WorkerJobType.MOE_EXPERT.value,
                "control_plane": "UniversalWorkerServer",
                "data_plane": "SWARMEX1",
            },
            model_id=self.model_id,
            model_revision=self.model_revision,
            quantization_fingerprint=self.quantization_fingerprint,
            model_fingerprint=self.model_fingerprint,
            bridge_version=self.bridge_version,
            owned_experts=by_layer,
            owned_microshards=list(self.configuration.get("owned_microshards", [])),
            tensor_hashes=hashes,
            resident_tensor_bytes=self.store.cache_bytes,
            expert_bytes=self._owned_expert_bytes,
            cache_bytes=self.store.cache_bytes,
            peak_rss_bytes=self.peak_rss_bytes,
            roles=list(self.configuration.get("roles", ["expert"])),
        )

    def _sign(self, result_hash: str) -> str:
        message = f"{self.worker_id}|{self.model_fingerprint}|{result_hash}".encode()
        return "hmac-sha256:" + hmac.new(self.secret, message, hashlib.sha256).hexdigest()

    async def execute(
        self,
        request: ExpertExecutionRequest,
        activation: np.ndarray,
        *,
        bytes_received: int,
        decode_ns: int,
        queue_ns: int,
    ) -> tuple[ExpertExecutionResponse, np.ndarray]:
        if request.request_id in self._responses:
            return self._responses[request.request_id]
        if time.time_ns() > request.deadline_ns:
            raise TimeoutError("expert request deadline elapsed before execution")
        if request.model_id != self.model_id or request.model_revision != self.model_revision:
            raise ValueError("expert request model identity does not match worker")
        if request.quantization_fingerprint != self.quantization_fingerprint:
            raise ValueError("expert request quantization fingerprint does not match worker")
        fault = self.fault.consume()
        if fault == FailureType.FIXED_DELAY and self.fault.fixed_delay_ms:
            await asyncio.sleep(self.fault.fixed_delay_ms / 1000)
        if fault == FailureType.RANDOM_DELAY and self.fault.random_delay_ms:
            digest = hashlib.sha256(request.request_id.encode("utf-8")).digest()
            delay = int.from_bytes(digest[:2], "big") / 0xFFFF * self.fault.random_delay_ms
            await asyncio.sleep(delay / 1000)
        if fault == FailureType.STORAGE_SLOWDOWN and self.fault.storage_slowdown_ms:
            await asyncio.sleep(self.fault.storage_slowdown_ms / 1000)
        if fault == FailureType.WORKER_TERMINATION:
            os._exit(70)
        if fault == FailureType.WORKER_PAUSE:
            await asyncio.sleep(
                self.fault.fixed_delay_ms / 1000 if self.fault.fixed_delay_ms > 0 else 60.0
            )
        if fault == FailureType.CACHE_DROP:
            self.store.drop_cache()
        output, metrics = self.store.execute(request, activation)
        revision = self.model_revision
        if fault == FailureType.ZERO_RESULT:
            output.fill(0)
        elif fault == FailureType.LOWER_PRECISION_RESULT:
            output = output.astype(np.float16).astype(np.float32)
        elif fault == FailureType.BIT_FLIP:
            raw = bytearray(np.ascontiguousarray(output).tobytes())
            raw[-1] ^= 1
            output = np.frombuffer(raw, dtype=np.float32).reshape(output.shape).copy()
        elif fault == FailureType.WRONG_MODEL_REVISION:
            revision = "stale-" + self.model_revision
        elif fault == FailureType.WRONG_EXPERT:
            output = np.roll(output, 1, axis=-1)
        elif fault == FailureType.STALE_RESULT and self._responses:
            return next(iter(self._responses.values()))
        if request.compression != TransportCodec.RAW_FP32:
            transported = encode_array(output, name="result", codec=request.compression)
            output = decode_array(transported.metadata, transported.payload).array
        result_hash = "sha256:" + sha256_bytes(np.ascontiguousarray(output).tobytes())
        response = ExpertExecutionResponse(
            request_id=request.request_id,
            worker_id=self.worker_id,
            model_revision=revision,
            layer_id=request.layer_id,
            result={"codec": request.compression.value},
            execution_metadata=ExpertExecutionMetadata(
                **metrics,
                bytes_received=bytes_received,
                bytes_sent=int(output.nbytes),
                queue_ns=queue_ns,
                transfer_ns=0,
                serialisation_ns=decode_ns,
                backend=str(self.configuration.get("backend", "numpy")),
                device=str(self.configuration.get("device", "cpu")),
                fallback_events=([{"injected_fault": fault.value}] if fault is not None else []),
            ),
            integrity=ResultIntegrity(
                result_hash=result_hash,
                model_fingerprint=self.model_fingerprint,
                worker_signature=self._sign(result_hash),
            ),
        )
        self.requests_completed += 1
        process = psutil.Process()
        self.peak_rss_bytes = max(self.peak_rss_bytes, int(process.memory_info().rss))
        if self.store.cache_bytes > self.memory_budget_bytes:
            raise MemoryError("worker logical memory budget was exceeded")
        result = (response, output)
        self._responses[request.request_id] = result
        if len(self._responses) > 256:
            self._responses.pop(next(iter(self._responses)))
        return result

    def configure_fault(self, payload: dict[str, Any]) -> None:
        selected = payload.get("fault_type")
        self.fault = WorkerFaultState(
            fault_type=FailureType(selected) if selected else None,
            remaining=int(payload.get("remaining", 1)),
            fixed_delay_ms=float(payload.get("fixed_delay_ms", 0)),
            random_delay_ms=float(payload.get("random_delay_ms", 0)),
            storage_slowdown_ms=float(payload.get("storage_slowdown_ms", 0)),
        )


class ExpertUniversalAdapter(BackendAdapter):
    """Expose expert execution through the repository's Universal Worker ABI.

    The Universal Worker endpoint is the authoritative lifecycle and capability
    control plane. The SWARMEX1 endpoint remains the measured activation data
    plane so tensor payloads do not acquire base64/JSON overhead in benchmarks.
    """

    backend_id = "experiment-010-expert"
    supported_jobs = frozenset({WorkerJobType.MOE_EXPERT})

    def __init__(self, runtime: ExpertWorkerRuntime, *, data_endpoint: str) -> None:
        self.runtime = runtime
        self.data_endpoint = data_endpoint
        self._active_request_ids: set[str] = set()
        self._cancelled_request_ids: set[str] = set()

    def capabilities(self) -> WorkerCapabilities:
        memory = psutil.virtual_memory()
        logical = psutil.cpu_count(logical=True) or 1
        physical = psutil.cpu_count(logical=False) or logical
        device = str(self.runtime.configuration.get("device", "cpu"))
        accelerator_type = "cuda" if device.startswith("cuda") else None
        return WorkerCapabilities(
            architecture=platform.machine() or "unknown",
            operating_system=platform.platform(),
            cpu_model=platform.processor() or "unknown",
            physical_cpu_cores=physical,
            logical_cpu_cores=logical,
            cpu_features=[],
            accelerator_type=accelerator_type,
            accelerator_model=None,
            accelerator_memory_bytes=0,
            system_memory_bytes=int(memory.total),
            supported_weight_formats=[
                str(self.runtime.configuration.get("loader_type", "unknown"))
            ],
            supported_activation_dtypes=["float32"],
            supported_cache_dtypes=["float32"],
            supported_collectives=[],
            maximum_weight_bytes=self.runtime.memory_budget_bytes,
            maximum_cache_bytes=int(self.runtime.configuration["cache_budget_bytes"]),
            maximum_batch_size=4096,
            maximum_context_length=1,
            measured_network_upload_bps=0.0,
            measured_network_download_bps=0.0,
            coordinator_latency_ms=0.0,
            backend_features=[
                "moe_expert",
                "expert_manifest",
                "direct_expert_data_plane",
                "shared_memory_data_plane",
                "deterministic_reduction_contract",
            ],
            backend_details={
                "expert_data_endpoint": self.data_endpoint,
                "model_id": self.runtime.model_id,
                "model_revision": self.runtime.model_revision,
                "model_fingerprint": self.runtime.model_fingerprint,
                "quantization_fingerprint": self.runtime.quantization_fingerprint,
                "owned_experts": sorted([list(item) for item in self.runtime._owned]),
                "memory_budget_bytes": self.runtime.memory_budget_bytes,
            },
        )

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return WorkerBenchmarkProfile(
            model_revision=self.runtime.model_revision,
            shard_hash=self.runtime.model_fingerprint,
            expert_calls_per_second=None,
            model_load_seconds=max(0.0, (time.time_ns() - self.runtime.started_ns) / 1e9),
            warmup_seconds=0.0,
        )

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        if not isinstance(job.input_payload, TensorPayload):
            raise TypeError("MOE_EXPERT requires a Universal Worker tensor payload")
        request = ExpertExecutionRequest.model_validate(job.metadata["expert_request"])
        if job.request_id in self._cancelled_request_ids:
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.CANCELLED,
                detail="request was cancelled before expert execution",
            )
        self._active_request_ids.add(job.request_id)
        try:
            activation = np.ascontiguousarray(job.input_payload.to_tensor().array)
            response, output = await self.runtime.execute(
                request,
                activation,
                bytes_received=int(activation.nbytes),
                decode_ns=0,
                queue_ns=0,
            )
            output_payload = tensor_payload_from_array(
                output,
                tensor_id=f"{job.job_id}:expert-result",
                request_id=job.request_id,
                stage_id=request.layer_id,
                token_position=0,
                sequence_length=int(output.shape[0]),
                model_revision=self.runtime.model_revision,
                partition_hash=self.runtime.model_fingerprint,
                route_generation=job.route_generation,
                logical_dtype="float32",
            )
            return WorkerJobResult(
                job_id=job.job_id,
                request_id=job.request_id,
                status=WorkerJobStatus.ACCEPTED,
                output_payload=output_payload,
                metrics={
                    "expert_response": response.model_dump(mode="json"),
                    "control_plane": "UniversalWorkerServer",
                    "data_plane_available": self.data_endpoint,
                },
                classification=(
                    ResultClassification.MEASURED_CUDA
                    if str(self.runtime.configuration.get("device", "cpu")).startswith("cuda")
                    else ResultClassification.MEASURED_X86_CPU
                ),
            )
        finally:
            self._active_request_ids.discard(job.request_id)

    async def cancel(self, request_id: str) -> bool:
        was_active = request_id in self._active_request_ids
        self._cancelled_request_ids.add(request_id)
        return was_active

    async def shutdown(self) -> None:
        self.runtime._shutdown.set()


class ExpertWorkerServer:
    def __init__(self, runtime: ExpertWorkerRuntime, *, host: str, port: int) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self.server: asyncio.Server | None = None

    @property
    def endpoint(self) -> tuple[str, int]:
        if self.server is None or not self.server.sockets:
            raise RuntimeError("expert worker server is not started")
        address = self.server.sockets[0].getsockname()
        return str(address[0]), int(address[1])

    async def start(self) -> tuple[str, int]:
        self.server = await asyncio.start_server(self._connection, self.host, self.port)
        return self.endpoint

    async def close(self) -> None:
        self.runtime._shutdown.set()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted_ns = time.perf_counter_ns()
        try:
            payload = await read_length_frame(reader)
            packet = decode_packet(payload)
            if packet.kind == "request":
                request, activation, decode_ns = decode_request(payload)
                response, output = await self.runtime.execute(
                    request,
                    activation,
                    bytes_received=len(payload),
                    decode_ns=decode_ns,
                    queue_ns=time.perf_counter_ns() - accepted_ns,
                )
                encoded, encode_ns = encode_response(response, output)
                response.execution_metadata.serialisation_ns += encode_ns
                encoded, _ = encode_response(response, output)
                if response.execution_metadata.fallback_events and any(
                    item.get("injected_fault") == FailureType.MALFORMED_RESULT.value
                    for item in response.execution_metadata.fallback_events
                ):
                    encoded = encoded[:-1]
                writer.write(frame_with_length(encoded))
                await writer.drain()
            elif packet.kind == "control":
                await self._control(packet.semantic, writer)
            else:
                raise ValueError("worker accepts request or control frames only")
        except Exception as error:
            self.runtime.failures += 1
            failure = encode_packet(
                ExpertPacket(
                    kind="control",
                    semantic={"ok": False, "error": f"{type(error).__name__}: {error}"},
                    blobs=(),
                )
            )
            with suppress(ConnectionError):
                writer.write(frame_with_length(failure))
                await writer.drain()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()

    async def _control(self, payload: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        command = str(payload.get("command"))
        if command == "manifest":
            endpoint = f"{self.endpoint[0]}:{self.endpoint[1]}"
            result: dict[str, Any] = {
                "ok": True,
                "manifest": self.runtime.manifest(endpoint).model_dump(mode="json"),
            }
        elif command == "heartbeat":
            result = {
                "ok": True,
                "worker_id": self.runtime.worker_id,
                "pid": os.getpid(),
                "requests_completed": self.runtime.requests_completed,
                "failures": self.runtime.failures,
                "uptime_ns": time.time_ns() - self.runtime.started_ns,
            }
        elif command == "configure_fault":
            self.runtime.configure_fault(payload)
            result = {"ok": True}
        elif command == "cache_drop":
            self.runtime.store.drop_cache()
            result = {"ok": True}
        elif command == "shared_memory":
            result = await self._shared_memory(payload)
        elif command == "shutdown":
            result = {"ok": True}
            asyncio.get_running_loop().call_soon(asyncio.create_task, self.close())
        else:
            result = {"ok": False, "error": f"unsupported control command {command!r}"}
        encoded = encode_packet(ExpertPacket(kind="control", semantic=result, blobs=()))
        writer.write(frame_with_length(encoded))
        await writer.drain()

    async def _shared_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = shared_memory.SharedMemory(name=str(payload["name"]), create=False)
        try:
            size = int(payload["size"])
            framed = bytes(source.buf[:size])
        finally:
            source.close()
        request, activation, decode_ns = decode_request(framed)
        response, output = await self.runtime.execute(
            request,
            activation,
            bytes_received=size,
            decode_ns=decode_ns,
            queue_ns=0,
        )
        encoded, encode_ns = encode_response(response, output)
        response.execution_metadata.serialisation_ns += encode_ns
        encoded, _ = encode_response(response, output)
        capacity = int(payload["result_capacity"])
        if len(encoded) > capacity:
            raise ValueError(
                f"encoded response needs {len(encoded)} bytes but result buffer has {capacity}"
            )
        destination = shared_memory.SharedMemory(name=str(payload["result_name"]), create=False)
        try:
            destination.buf[: len(encoded)] = encoded
        finally:
            destination.close()
        return {
            "ok": True,
            "name": str(payload["result_name"]),
            "size": len(encoded),
        }


@dataclass(slots=True)
class WorkerProcess:
    worker_id: str
    process: subprocess.Popen[bytes]
    endpoint: str
    control_endpoint: str
    configuration_path: Path
    ready_path: Path
    stdout_path: Path
    stderr_path: Path
    signature_secret: bytes
    negotiated_protocol: dict[str, Any]
    universal_identity: dict[str, Any]
    universal_capabilities: dict[str, Any]
    initial_heartbeat: dict[str, Any]


class ExpertWorkerManager:
    """Start and clean exact child PIDs while restoring no global state."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workers: dict[str, WorkerProcess] = {}
        self.lifecycle_records: list[dict[str, Any]] = []
        self._lifecycle_by_worker: dict[str, dict[str, Any]] = {}

    def start(
        self,
        *,
        worker_id: str,
        model_id: str,
        model_revision: str,
        quantization_fingerprint: str,
        model_fingerprint: str,
        owned_experts: list[dict[str, Any]],
        budget: WorkerBudget,
        loader_type: str,
        model_path: Path | None = None,
        owned_microshards: list[dict[str, Any]] | None = None,
        timeout_seconds: float = 30.0,
    ) -> WorkerProcess:
        if worker_id in self.workers:
            raise ValueError(f"worker {worker_id} already exists")
        directory = (self.root / worker_id).resolve()
        if self.root not in directory.parents:
            raise ValueError("worker directory escaped manager root")
        directory.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_bytes(32)
        configuration = {
            "worker_id": worker_id,
            "model_id": model_id,
            "model_revision": model_revision,
            "quantization_fingerprint": quantization_fingerprint,
            "model_fingerprint": model_fingerprint,
            "signature_secret": base64.b64encode(secret).decode("ascii"),
            "owned_experts": owned_experts,
            "owned_microshards": owned_microshards or [],
            "memory_budget_bytes": budget.memory_budget_bytes,
            "expert_residency_budget_bytes": budget.expert_residency_budget_bytes,
            "cache_budget_bytes": budget.cache_budget_bytes,
            "loader_type": loader_type,
            "model_path": str(model_path) if model_path else None,
            "backend": budget.backend,
            "device": budget.device,
            "roles": [WorkerJobType.MOE_EXPERT.value],
            "storage_directory": budget.storage_directory,
            "physical_memory_limit": budget.physical_memory_limit,
            "host": "127.0.0.1",
            "port": 0,
        }
        config_path = directory / "worker-config.json"
        ready_path = directory / "ready.json"
        if ready_path.is_file():
            ready_path.unlink()
        config_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        stdout_path, stderr_path = directory / "stdout.log", directory / "stderr.log"
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(budget.thread_count),
                "MKL_NUM_THREADS": str(budget.thread_count),
                "OPENBLAS_NUM_THREADS": str(budget.thread_count),
                "NUMEXPR_NUM_THREADS": str(budget.thread_count),
            }
        )
        repository_root = Path(__file__).resolve().parents[4]
        source_root = str(repository_root / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_root, environment.get("PYTHONPATH", "")) if part
        )
        command = [
            sys.executable,
            "-m",
            "swarm_inference.experiments.experiment_010.process_main",
            "--config",
            str(config_path),
            "--ready",
            str(ready_path),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        started_ns = time.time_ns()
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        stdout.close()
        stderr.close()
        try:
            ps_process = psutil.Process(process.pid)
            ps_process.cpu_affinity(budget.cpu_affinity)
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"worker {worker_id} exited with {process.returncode}")
                if ready_path.is_file():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    control_endpoint = str(ready["control_endpoint"])
                    control_host, separator, raw_control_port = control_endpoint.rpartition(":")
                    if not separator or not control_host:
                        raise ValueError("worker returned an invalid Universal Worker endpoint")

                    async def inspect_universal_worker(
                        host: str = control_host,
                        raw_port: str = raw_control_port,
                        data_endpoint: str = str(ready["endpoint"]),
                        expected_worker_id: str = worker_id,
                    ) -> dict[str, Any]:
                        client = UniversalWorkerClient(
                            host,
                            int(raw_port),
                            timeout_seconds=5.0,
                        )
                        negotiated = await client.negotiate(
                            WorkerProtocolVersion(
                                major=1,
                                minor=1,
                                capabilities={
                                    "jobs",
                                    "cancel",
                                    "heartbeat",
                                    "clean-shutdown",
                                    "moe-expert",
                                    "direct-expert-data-plane",
                                },
                            )
                        )
                        identity = await client.identity()
                        capabilities = await client.capabilities()
                        heartbeat = await client.heartbeat()
                        if identity.worker_id != expected_worker_id:
                            raise ValueError("Universal Worker identity does not match process")
                        if (
                            capabilities.backend_details.get("expert_data_endpoint")
                            != data_endpoint
                        ):
                            raise ValueError(
                                "Universal Worker capability points at the wrong data plane"
                            )
                        return {
                            "negotiated": negotiated.model_dump(mode="json"),
                            "identity": identity.model_dump(mode="json"),
                            "capabilities": capabilities.model_dump(mode="json"),
                            "heartbeat": heartbeat,
                        }

                    universal = asyncio.run(inspect_universal_worker())
                    worker = WorkerProcess(
                        worker_id=worker_id,
                        process=process,
                        endpoint=str(ready["endpoint"]),
                        control_endpoint=control_endpoint,
                        configuration_path=config_path,
                        ready_path=ready_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        signature_secret=secret,
                        negotiated_protocol=universal["negotiated"],
                        universal_identity=universal["identity"],
                        universal_capabilities=universal["capabilities"],
                        initial_heartbeat=universal["heartbeat"],
                    )
                    self.workers[worker_id] = worker
                    lifecycle = {
                        "worker_id": worker_id,
                        "pid": process.pid,
                        "command": command,
                        "environment": {
                            key: environment[key]
                            for key in (
                                "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS",
                            )
                        },
                        "data_endpoint": worker.endpoint,
                        "control_endpoint": worker.control_endpoint,
                        "started_ns": started_ns,
                        "stopped_ns": None,
                        "exit_code": None,
                        "shutdown_via_universal_worker": None,
                        "status": "RUNNING",
                    }
                    self.lifecycle_records.append(lifecycle)
                    self._lifecycle_by_worker[worker_id] = lifecycle
                    return worker
                time.sleep(0.05)
            raise TimeoutError(f"worker {worker_id} did not become ready")
        except Exception:
            self._stop_process(process)
            self.lifecycle_records.append(
                {
                    "worker_id": worker_id,
                    "pid": process.pid,
                    "command": command,
                    "started_ns": started_ns,
                    "stopped_ns": time.time_ns(),
                    "exit_code": process.poll(),
                    "shutdown_via_universal_worker": False,
                    "status": "START_FAILED",
                }
            )
            raise

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        with suppress(psutil.Error):
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            for child in children:
                with suppress(psutil.Error):
                    child.terminate()
            parent.terminate()
            _, alive = psutil.wait_procs([*children, parent], timeout=3)
            for item in alive:
                with suppress(psutil.Error):
                    item.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)

    def stop(self, worker_id: str) -> None:
        worker = self.workers.pop(worker_id, None)
        if worker is not None:
            graceful = False
            if worker.process.poll() is None:
                host, separator, raw_port = worker.control_endpoint.rpartition(":")
                if separator and host:
                    with suppress(Exception):
                        graceful = bool(
                            asyncio.run(
                                UniversalWorkerClient(
                                    host, int(raw_port), timeout_seconds=3.0
                                ).shutdown()
                            )
                        )
                        worker.process.wait(timeout=3)
            self._stop_process(worker.process)
            lifecycle = self._lifecycle_by_worker.get(worker_id)
            if lifecycle is not None:
                lifecycle.update(
                    {
                        "stopped_ns": time.time_ns(),
                        "exit_code": worker.process.poll(),
                        "exit_expected": worker.process.poll() in {0, 70},
                        "shutdown_via_universal_worker": graceful,
                        "termination_mode": (
                            "universal_worker_shutdown" if graceful else "manager_process_fallback"
                        ),
                        "status": "STOPPED",
                    }
                )

    def close(self) -> None:
        for worker_id in list(self.workers):
            self.stop(worker_id)

    def __enter__(self) -> ExpertWorkerManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def verify_worker_signature(
    response: ExpertExecutionResponse,
    result: np.ndarray,
    *,
    secret: bytes,
) -> bool:
    result_hash = "sha256:" + sha256_bytes(np.ascontiguousarray(result).tobytes())
    if not hmac.compare_digest(response.integrity.result_hash, result_hash):
        return False
    message = f"{response.worker_id}|{response.integrity.model_fingerprint}|{result_hash}".encode()
    expected = "hmac-sha256:" + hmac.new(secret, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(response.integrity.worker_signature, expected)


def fixture_ownership_entry(layer_id: int, expert_id: int, path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        logical_bytes = sum(int(archive[name].nbytes) for name in ("up", "gate", "down"))
    return {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "path": str(path.expanduser().resolve()),
        "content_hash": "sha256:" + sha256_file(path),
        "logical_bytes": logical_bytes,
    }
