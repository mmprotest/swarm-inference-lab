"""Explicit GPU-local, same-host, and remote-compatible tensor paths."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.qwen3_runtime import nvtx_range
from swarm_inference.protocol.checksums import sha256_bytes


class TensorPath(StrEnum):
    IN_PROCESS_GPU = "in_process_gpu"
    SAME_HOST_PROCESS = "same_host_process"
    REMOTE_COMPATIBLE = "remote_compatible"


FRAME_MAGIC = b"Q3TF0001"
FRAME_HEADER = struct.Struct(">I")


@dataclass(slots=True)
class TensorPathMetrics:
    profile: str
    path: str
    selected_method: str
    transfers: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    serialised_bytes: int = 0
    serialisation_ms: float = 0.0
    deserialisation_ms: float = 0.0
    explicit_synchronisations: int = 0
    buffer_allocations: int = 0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class PreallocatedTensorTransport:
    def __init__(
        self,
        *,
        torch_module: Any,
        device: Any,
        path: TensorPath | str,
        profile: str,
        nvtx_enabled: bool = False,
    ) -> None:
        self.torch = torch_module
        self.device = torch_module.device(device)
        self.path = TensorPath(path)
        self.profile = profile
        self.nvtx_enabled = nvtx_enabled
        selected_method = {
            TensorPath.IN_PROCESS_GPU: "direct_cuda_tensor_reference",
            TensorPath.SAME_HOST_PROCESS: "pinned_bfloat16_async_staging",
            TensorPath.REMOTE_COMPATIBLE: "pinned_bfloat16_binary_frame",
        }[self.path]
        self.metrics = TensorPathMetrics(
            profile=profile,
            path=self.path.value,
            selected_method=selected_method,
        )
        self._host_buffers: dict[tuple[tuple[int, ...], Any], Any] = {}
        self._device_buffers: dict[tuple[tuple[int, ...], Any], Any] = {}
        self._copy_stream = (
            torch_module.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    def _key(self, tensor: Any) -> tuple[tuple[int, ...], Any]:
        return tuple(int(value) for value in tensor.shape), tensor.dtype

    def _host_buffer(self, tensor: Any) -> Any:
        key = self._key(tensor)
        buffer = self._host_buffers.get(key)
        if buffer is None:
            buffer = self.torch.empty(
                key[0],
                dtype=tensor.dtype,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            )
            self._host_buffers[key] = buffer
            self.metrics.buffer_allocations += 1
        return buffer

    def _device_buffer(self, tensor: Any) -> Any:
        key = self._key(tensor)
        buffer = self._device_buffers.get(key)
        if buffer is None:
            buffer = self.torch.empty(
                key[0],
                dtype=tensor.dtype,
                device=self.device,
            )
            self._device_buffers[key] = buffer
            self.metrics.buffer_allocations += 1
        return buffer

    def transfer(self, tensor: Any) -> Any:
        if tensor.dtype != self.torch.bfloat16:
            raise ValueError(f"stage transport requires BF16, received {tensor.dtype}")
        self.metrics.transfers += 1
        if self.path == TensorPath.IN_PROCESS_GPU:
            if tensor.device != self.device:
                raise ValueError(
                    f"in-process GPU path requires tensor on {self.device}, "
                    f"received {tensor.device}"
                )
            return tensor
        if self.path == TensorPath.REMOTE_COMPATIBLE:
            return self.decode_remote(self.encode_remote(tensor))
        if self._copy_stream is None:
            raise RuntimeError("same-host pinned path requires CUDA")
        host = self._host_buffer(tensor)
        destination = self._device_buffer(tensor)
        byte_count = int(tensor.numel() * tensor.element_size())
        with (
            nvtx_range(
                self.torch,
                "device_to_host",
                enabled=self.nvtx_enabled,
            ),
            self.torch.cuda.stream(self._copy_stream),
        ):
            host.copy_(tensor, non_blocking=True)
            copied = self.torch.cuda.Event()
            copied.record(self._copy_stream)
        with nvtx_range(
            self.torch,
            "host_to_device",
            enabled=self.nvtx_enabled,
        ):
            current = self.torch.cuda.current_stream(self.device)
            current.wait_event(copied)
            destination.copy_(host, non_blocking=True)
            complete = self.torch.cuda.Event()
            complete.record(current)
        complete.synchronize()
        self.metrics.explicit_synchronisations += 1
        self.metrics.device_to_host_bytes += byte_count
        self.metrics.host_to_device_bytes += byte_count
        return destination

    def encode_remote(self, tensor: Any) -> bytes:
        if tensor.dtype != self.torch.bfloat16:
            raise ValueError("remote-compatible tensor frame requires BF16")
        started = time.perf_counter()
        host = tensor
        if tensor.device.type == "cuda":
            if self._copy_stream is None:
                raise RuntimeError("CUDA remote encoding requires a copy stream")
            host = self._host_buffer(tensor)
            with nvtx_range(
                self.torch,
                "device_to_host",
                enabled=self.nvtx_enabled,
            ):
                with self.torch.cuda.stream(self._copy_stream):
                    host.copy_(tensor, non_blocking=True)
                    complete = self.torch.cuda.Event()
                    complete.record(self._copy_stream)
                complete.synchronize()
            self.metrics.explicit_synchronisations += 1
            self.metrics.device_to_host_bytes += int(tensor.numel() * tensor.element_size())
        with nvtx_range(
            self.torch,
            "tensor_encoding",
            enabled=self.nvtx_enabled,
        ):
            raw = host.contiguous().view(self.torch.uint16).numpy().tobytes(order="C")
            header = {
                "profile": self.profile,
                "dtype": "bfloat16",
                "shape": [int(value) for value in host.shape],
                "payload_length": len(raw),
                "checksum": sha256_bytes(raw),
            }
            encoded_header = json.dumps(
                header,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            frame = FRAME_MAGIC + FRAME_HEADER.pack(len(encoded_header)) + encoded_header + raw
        self.metrics.serialised_bytes += len(frame)
        self.metrics.serialisation_ms += (time.perf_counter() - started) * 1000
        return bytes(frame)

    def decode_remote(self, frame: bytes) -> Any:
        import numpy as np

        started = time.perf_counter()
        minimum = len(FRAME_MAGIC) + FRAME_HEADER.size
        if len(frame) < minimum or frame[: len(FRAME_MAGIC)] != FRAME_MAGIC:
            raise IntegrityError("invalid Qwen3 tensor frame magic")
        header_length = FRAME_HEADER.unpack_from(frame, len(FRAME_MAGIC))[0]
        header_start = minimum
        header_end = header_start + header_length
        if header_end > len(frame):
            raise IntegrityError("Qwen3 tensor frame header is truncated")
        with nvtx_range(
            self.torch,
            "tensor_decoding",
            enabled=self.nvtx_enabled,
        ):
            header = json.loads(frame[header_start:header_end].decode("utf-8"))
            raw = frame[header_end:]
            if header.get("dtype") != "bfloat16":
                raise IntegrityError(
                    f"remote tensor frame dtype must be bfloat16, got {header.get('dtype')}"
                )
            if len(raw) != int(header.get("payload_length", -1)):
                raise IntegrityError("remote tensor frame payload length mismatch")
            if sha256_bytes(raw) != header.get("checksum"):
                raise IntegrityError("remote tensor frame checksum mismatch")
            shape = tuple(int(value) for value in header["shape"])
            bits = np.frombuffer(raw, dtype=np.uint16).copy().reshape(shape)
            cpu_tensor = self.torch.from_numpy(bits).view(self.torch.bfloat16)
        if self.device.type == "cuda":
            pinned = self._host_buffer(cpu_tensor)
            pinned.copy_(cpu_tensor)
            destination = self._device_buffer(cpu_tensor)
            with nvtx_range(
                self.torch,
                "host_to_device",
                enabled=self.nvtx_enabled,
            ):
                destination.copy_(pinned, non_blocking=True)
                complete = self.torch.cuda.Event()
                complete.record(self.torch.cuda.current_stream(self.device))
                complete.synchronize()
            self.metrics.explicit_synchronisations += 1
            self.metrics.host_to_device_bytes += len(raw)
            result = destination
        else:
            result = cpu_tensor
        self.metrics.deserialisation_ms += (time.perf_counter() - started) * 1000
        return result
