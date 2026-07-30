from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from swarm_inference.exceptions import IntegrityError
from swarm_inference.experiments.fanout_acquisition import (
    AcquisitionCancelledError,
    acquire_shard_directory,
)
from swarm_inference.model.manifest import hash_shard_directory


def _source(tmp_path: Path, size: int = 128 * 1024) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.safetensors").write_bytes(os.urandom(size))
    (source / "stage.json").write_text('{"stage_id":0}\n', encoding="utf-8")
    return source


def test_acquisition_throttling_hash_and_timing_fields(tmp_path: Path) -> None:
    source = _source(tmp_path, 64 * 1024)
    destination = tmp_path / "destination"
    started = time.perf_counter()
    result = acquire_shard_directory(
        source=source,
        destination=destination,
        expected_hash=hash_shard_directory(source),
        profile="test",
        bandwidth_mbps=8,
        latency_ms=10,
        chunk_bytes=16 * 1024,
    )
    elapsed = time.perf_counter() - started
    assert result.measurement_class == "emulated-shard-acquisition"
    assert result.chunk_count >= 4
    assert result.shard_bytes > 0
    assert result.transfer_duration_seconds >= 0.05
    assert elapsed >= result.transfer_duration_seconds
    assert result.verification_duration_seconds >= 0
    assert hash_shard_directory(destination) == result.expected_hash


def test_corrupt_transfer_is_rejected_and_partial_is_cleaned(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "corrupt"
    with pytest.raises(IntegrityError, match="chunk checksum"):
        acquire_shard_directory(
            source=source,
            destination=destination,
            expected_hash=hash_shard_directory(source),
            profile="test",
            bandwidth_mbps=None,
            latency_ms=0,
            chunk_bytes=16 * 1024,
            corrupt_chunk_index=0,
        )
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".partial").exists()


def test_cancelled_transfer_resumes_valid_prefix(tmp_path: Path) -> None:
    source = _source(tmp_path, 128 * 1024)
    destination = tmp_path / "resumed"
    checks = 0

    def cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks > 2

    with pytest.raises(AcquisitionCancelledError):
        acquire_shard_directory(
            source=source,
            destination=destination,
            expected_hash=hash_shard_directory(source),
            profile="test",
            bandwidth_mbps=None,
            latency_ms=0,
            chunk_bytes=16 * 1024,
            cancel_requested=cancel,
        )
    assert destination.with_name(destination.name + ".partial").exists()
    result = acquire_shard_directory(
        source=source,
        destination=destination,
        expected_hash=hash_shard_directory(source),
        profile="test",
        bandwidth_mbps=None,
        latency_ms=0,
        chunk_bytes=16 * 1024,
    )
    assert result.resumed_bytes > 0
    assert result.retry_count == 1
