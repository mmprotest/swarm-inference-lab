"""Measured local shard streaming with honest network-profile emulation."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.manifest import hash_shard_directory


class AcquisitionCancelledError(RuntimeError):
    """Raised when a caller cancels an otherwise resumable transfer."""


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    source: str
    destination: str
    profile: str
    measurement_class: str
    shard_bytes: int
    transfer_duration_seconds: float
    verification_duration_seconds: float
    total_acquisition_duration_seconds: float
    effective_throughput_mbps: float
    chunk_count: int
    retry_count: int
    resumed_bytes: int
    latency_ms: float
    bandwidth_mbps: float | None
    expected_hash: str
    actual_hash: str
    atomic_rename: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def acquire_shard_directory(
    *,
    source: str | Path,
    destination: str | Path,
    expected_hash: str,
    profile: str,
    bandwidth_mbps: float | None,
    latency_ms: float,
    chunk_bytes: int = 4 * 1024 * 1024,
    cancel_requested: Callable[[], bool] | None = None,
    allow_resume: bool = True,
    corrupt_chunk_index: int | None = None,
) -> AcquisitionResult:
    """Stream a stage directory, validate every chunk and the final directory hash."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if bandwidth_mbps is not None and bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive when provided")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    source_root = Path(source).expanduser().resolve()
    destination_root = Path(destination).expanduser().resolve()
    if not source_root.is_dir():
        raise IntegrityError(f"shard acquisition source is not a directory: {source_root}")
    if destination_root.exists():
        raise IntegrityError(f"refusing to overwrite existing acquired shard: {destination_root}")
    temporary = destination_root.with_name(destination_root.name + ".partial")
    if temporary.exists() and not allow_resume:
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    source_files = _files(source_root)
    shard_bytes = sum(path.stat().st_size for path in source_files)
    bytes_per_second = bandwidth_mbps * 1_000_000 / 8 if bandwidth_mbps is not None else None
    transfer_started = time.perf_counter()
    if latency_ms:
        time.sleep(latency_ms / 1000)
    chunk_count = 0
    resumed_bytes = 0
    transferred_bytes = 0
    global_chunk_index = 0
    try:
        for source_file in source_files:
            relative = source_file.relative_to(source_root)
            partial_file = temporary / relative
            partial_file.parent.mkdir(parents=True, exist_ok=True)
            source_size = source_file.stat().st_size
            offset = partial_file.stat().st_size if partial_file.exists() and allow_resume else 0
            if offset > source_size:
                partial_file.unlink()
                offset = 0
            resumed_bytes += offset
            mode = "ab" if offset else "wb"
            with source_file.open("rb") as reader, partial_file.open(mode) as writer:
                reader.seek(offset)
                while True:
                    if cancel_requested is not None and cancel_requested():
                        raise AcquisitionCancelledError(
                            f"acquisition cancelled after {transferred_bytes} new bytes"
                        )
                    payload = reader.read(chunk_bytes)
                    if not payload:
                        break
                    expected_chunk_hash = _sha256_bytes(payload)
                    written = (
                        bytes([payload[0] ^ 0x01]) + payload[1:]
                        if corrupt_chunk_index == global_chunk_index and payload
                        else payload
                    )
                    writer.write(written)
                    writer.flush()
                    if _sha256_bytes(written) != expected_chunk_hash:
                        raise IntegrityError(
                            f"chunk checksum mismatch at chunk {global_chunk_index}"
                        )
                    chunk_count += 1
                    global_chunk_index += 1
                    transferred_bytes += len(payload)
                    if bytes_per_second is not None:
                        target_elapsed = transferred_bytes / bytes_per_second
                        actual_elapsed = time.perf_counter() - transfer_started
                        if target_elapsed > actual_elapsed:
                            time.sleep(target_elapsed - actual_elapsed)
            if partial_file.stat().st_size != source_size:
                raise IntegrityError(
                    f"acquired file length mismatch for {relative}: "
                    f"{partial_file.stat().st_size} != {source_size}"
                )
        transfer_duration = time.perf_counter() - transfer_started
        verification_started = time.perf_counter()
        actual_hash = hash_shard_directory(temporary)
        verification_duration = time.perf_counter() - verification_started
        if actual_hash != expected_hash:
            raise IntegrityError(
                f"acquired shard hash mismatch: expected={expected_hash} actual={actual_hash}"
            )
        destination_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination_root)
        total_duration = time.perf_counter() - transfer_started
        effective = (
            transferred_bytes * 8 / transfer_duration / 1_000_000 if transfer_duration > 0 else 0.0
        )
        return AcquisitionResult(
            source=str(source_root),
            destination=str(destination_root),
            profile=profile,
            measurement_class="emulated-shard-acquisition",
            shard_bytes=shard_bytes,
            transfer_duration_seconds=transfer_duration,
            verification_duration_seconds=verification_duration,
            total_acquisition_duration_seconds=total_duration,
            effective_throughput_mbps=effective,
            chunk_count=chunk_count,
            retry_count=1 if resumed_bytes else 0,
            resumed_bytes=resumed_bytes,
            latency_ms=latency_ms,
            bandwidth_mbps=bandwidth_mbps,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            atomic_rename=True,
        )
    except AcquisitionCancelledError:
        # A checksum-valid prefix is intentionally retained for an explicit resume.
        raise
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
