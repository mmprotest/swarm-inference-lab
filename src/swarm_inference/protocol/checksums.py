"""Checksums used for activation and shard integrity."""

from __future__ import annotations

import hashlib
import zlib
from pathlib import Path


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crc32c_hex(payload: bytes | bytearray | memoryview) -> str:
    """Return a fast transport checksum.

    Python's standard library exposes IEEE CRC32 rather than CRC32C. The wire
    field is deliberately named ``checksum`` rather than claiming CRC32C.
    SHA-256 remains the acceptance and shard-integrity primitive.
    """

    return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"
