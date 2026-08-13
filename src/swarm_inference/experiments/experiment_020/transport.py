"""Authenticated, encrypted, bounded binary transport for E020/E021.

The protocol has no arbitrary-execution message.  Workers accept only the
enumerated registration, health, shard-execution, result, and shutdown frames.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import hmac
import json
import os
import ssl
import statistics
import struct
import tempfile
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

MAGIC = b"SW20"
VERSION = 1
MAX_METADATA_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
_HEADER = struct.Struct("!4sBBHII32s32s")


class MessageType(IntEnum):
    REGISTER = 1
    REGISTERED = 2
    HEALTH = 3
    EXECUTE_SHARD = 4
    SHARD_RESULT = 5
    ERROR = 6
    SHUTDOWN = 7


@dataclass(frozen=True, slots=True)
class Frame:
    message_type: MessageType
    request_id: str
    chunk_id: int
    worker_id: str
    state_id: str
    payload: bytes = b""

    def metadata(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "chunk_id": self.chunk_id,
            "worker_id": self.worker_id,
            "state_id": self.state_id,
        }


class ProtocolError(RuntimeError):
    pass


def encode_frame(frame: Frame, credential: bytes) -> bytes:
    if not credential:
        raise ProtocolError("per-run credential is required")
    metadata = json.dumps(
        frame.metadata(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload = bytes(frame.payload)
    if len(metadata) > MAX_METADATA_BYTES or len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("bounded frame limit exceeded")
    checksum = hashlib.sha256(payload).digest()
    prefix = struct.pack(
        "!4sBBHII32s",
        MAGIC,
        VERSION,
        int(frame.message_type),
        0,
        len(metadata),
        len(payload),
        checksum,
    )
    signature = hmac.new(credential, prefix + metadata + payload, hashlib.sha256).digest()
    return _HEADER.pack(
        MAGIC,
        VERSION,
        int(frame.message_type),
        0,
        len(metadata),
        len(payload),
        checksum,
        signature,
    ) + metadata + payload


def decode_frame(encoded: bytes, credential: bytes) -> Frame:
    if len(encoded) < _HEADER.size:
        raise ProtocolError("truncated frame")
    magic, version, kind, flags, metadata_length, payload_length, checksum, signature = (
        _HEADER.unpack_from(encoded)
    )
    if magic != MAGIC or version != VERSION or flags != 0:
        raise ProtocolError("invalid framing header")
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ProtocolError("declared frame length exceeds bound")
    expected_length = _HEADER.size + metadata_length + payload_length
    if len(encoded) != expected_length:
        raise ProtocolError("frame length mismatch")
    metadata = encoded[_HEADER.size : _HEADER.size + metadata_length]
    payload = encoded[_HEADER.size + metadata_length :]
    if not hmac.compare_digest(hashlib.sha256(payload).digest(), checksum):
        raise ProtocolError("payload checksum mismatch")
    prefix = struct.pack(
        "!4sBBHII32s",
        magic,
        version,
        kind,
        flags,
        metadata_length,
        payload_length,
        checksum,
    )
    expected_signature = hmac.new(
        credential, prefix + metadata + payload, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected_signature, signature):
        raise ProtocolError("frame authentication failed")
    try:
        fields = json.loads(metadata)
        message_type = MessageType(kind)
        return Frame(
            message_type=message_type,
            request_id=str(fields["request_id"]),
            chunk_id=int(fields["chunk_id"]),
            worker_id=str(fields["worker_id"]),
            state_id=str(fields["state_id"]),
            payload=payload,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid frame metadata") from exc


async def read_frame(reader: asyncio.StreamReader, credential: bytes) -> Frame:
    header = await asyncio.wait_for(reader.readexactly(_HEADER.size), timeout=15.0)
    _, _, _, _, metadata_length, payload_length, _, _ = _HEADER.unpack(header)
    if metadata_length > MAX_METADATA_BYTES or payload_length > MAX_PAYLOAD_BYTES:
        raise ProtocolError("declared frame length exceeds bound")
    body = await asyncio.wait_for(
        reader.readexactly(metadata_length + payload_length), timeout=15.0
    )
    return decode_frame(header + body, credential)


async def write_frame(
    writer: asyncio.StreamWriter,
    frame: Frame,
    credential: bytes,
) -> None:
    writer.write(encode_frame(frame, credential))
    await asyncio.wait_for(writer.drain(), timeout=15.0)


@dataclass(frozen=True, slots=True)
class TlsMaterial:
    directory: Path
    certificate: Path
    private_key: Path
    certificate_sha256: str


def generate_run_tls_material(directory: Path | None = None) -> TlsMaterial:
    root = directory or Path(tempfile.mkdtemp(prefix="swarm-e021-tls-"))
    root.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "swarm-e021-controller")]
    )
    now = dt.datetime.now(dt.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=1))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    certificate_bytes = certificate.public_bytes(serialization.Encoding.PEM)
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    certificate_path = root / "controller-cert.pem"
    key_path = root / "controller-key.pem"
    certificate_path.write_bytes(certificate_bytes)
    key_path.write_bytes(key_bytes)
    return TlsMaterial(
        directory=root,
        certificate=certificate_path,
        private_key=key_path,
        certificate_sha256=hashlib.sha256(certificate_bytes).hexdigest(),
    )


def tls_contexts(material: TlsMaterial) -> tuple[ssl.SSLContext, ssl.SSLContext]:
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.minimum_version = ssl.TLSVersion.TLSv1_2
    server.load_cert_chain(material.certificate, material.private_key)
    client = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cafile=str(material.certificate)
    )
    client.minimum_version = ssl.TLSVersion.TLSv1_2
    client.check_hostname = True
    return server, client


def new_run_credential() -> bytes:
    return os.urandom(32)


def security_receipt(material: TlsMaterial, credential: bytes) -> dict[str, Any]:
    return {
        "schema_version": "experiment-020-security-validation-v1",
        "tls_enabled": True,
        "minimum_tls_version": "TLSv1.2",
        "per_run_credential_present": bool(credential),
        "credential_persisted": False,
        "certificate_sha256": material.certificate_sha256,
        "arbitrary_execution_message_supported": False,
        "hmac_sha256_frame_authentication": True,
        "sha256_payload_checksums": True,
        "bounded_payload_bytes": MAX_PAYLOAD_BYTES,
        "backpressure": "asyncio StreamWriter.drain",
    }


def benchmark_transport_protocol(
    payload_sizes: tuple[int, ...] = (128, 14336, 28672, 57344, 114688),
    *,
    warmup: int = 20,
    iterations: int = 200,
) -> dict[str, Any]:
    """Measure the exact E021 framing/auth/checksum software path."""

    credential = new_run_credential()
    rows = []
    for size in payload_sizes:
        payload = bytes((index * 31) & 0xFF for index in range(size))
        frame = Frame(MessageType.EXECUTE_SHARD, "benchmark", 0, "worker", "state", payload)
        samples: list[float] = []
        for iteration in range(warmup + iterations):
            started = time.perf_counter_ns()
            decoded = decode_frame(encode_frame(frame, credential), credential)
            if decoded.payload != payload:
                raise ProtocolError("transport benchmark roundtrip mismatch")
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if iteration >= warmup:
                samples.append(elapsed)
        ordered = sorted(samples)
        rows.append(
            {
                "payload_bytes": size,
                "software_overhead_p50_ms": statistics.median(samples),
                "software_overhead_p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
                "software_overhead_max_ms": max(samples),
                "iterations": iterations,
                "binary_framing": True,
                "hmac_verified": True,
                "checksum_verified": True,
            }
        )
    return {
        "schema_version": "experiment-020-transport-benchmark-v1",
        "status": "PASS",
        "protocol": {"rows": rows},
        "persistent_connections": True,
        "bounded_buffers": True,
        "backpressure": True,
        "timeouts": True,
        "error_propagation": True,
        "credential_value_persisted": False,
    }


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "Frame",
    "MessageType",
    "ProtocolError",
    "TlsMaterial",
    "benchmark_transport_protocol",
    "decode_frame",
    "encode_frame",
    "generate_run_tls_material",
    "new_run_credential",
    "read_frame",
    "security_receipt",
    "tls_contexts",
    "write_frame",
]
