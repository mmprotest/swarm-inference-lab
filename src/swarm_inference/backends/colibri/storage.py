"""Conservative read-only profiling of each local Colibri storage path."""

from __future__ import annotations

import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _timed_read(path: Path, offsets: list[int], size: int) -> tuple[int, float]:
    total = 0
    started = time.perf_counter()
    with path.open("rb", buffering=0) as handle:
        for offset in offsets:
            handle.seek(offset)
            block = handle.read(size)
            if not block:
                break
            total += len(block)
    return total, time.perf_counter() - started


class ColibriStorageProfiler:
    """Measure warm/buffered reads without pretending the OS cache was flushed."""

    def profile(
        self,
        path: str | Path,
        *,
        expert_read_bytes: int = 4 * 1024 * 1024,
        samples: int = 16,
        maximum_queue_depth: int = 4,
        seed: int = 9009,
    ) -> dict[str, Any]:
        selected = Path(path).expanduser().resolve()
        size = selected.stat().st_size
        if size <= 0:
            raise ValueError("cannot profile an empty storage file")
        block = min(expert_read_bytes, size)
        samples = max(1, min(samples, max(1, size // block)))
        sequential = [min(index * block, size - block) for index in range(samples)]
        generator = random.Random(seed)
        random_offsets = [generator.randrange(0, max(1, size - block + 1)) for _ in range(samples)]
        sequential_bytes, sequential_seconds = _timed_read(selected, sequential, block)
        random_results = [_timed_read(selected, [offset], block) for offset in random_offsets]
        random_bytes = sum(item[0] for item in random_results)
        random_seconds = sum(item[1] for item in random_results)
        parallel: list[dict[str, Any]] = []
        for depth in range(1, max(1, maximum_queue_depth) + 1):
            groups = [random_offsets[index::depth] for index in range(depth)]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=depth) as executor:
                values = list(
                    executor.map(lambda offsets: _timed_read(selected, offsets, block), groups)
                )
            elapsed = time.perf_counter() - started
            transferred = sum(item[0] for item in values)
            parallel.append(
                {
                    "queue_depth": depth,
                    "bytes": transferred,
                    "seconds": elapsed,
                    "bandwidth_bytes_per_second": transferred / elapsed if elapsed else None,
                }
            )
        drive, _ = os.path.splitdrive(str(selected))
        return {
            "path": str(selected),
            "device_id": drive or str(selected.anchor),
            "file_size_bytes": size,
            "expert_read_bytes": block,
            "sample_count": samples,
            "sequential": {
                "bytes": sequential_bytes,
                "seconds": sequential_seconds,
                "bandwidth_bytes_per_second": (
                    sequential_bytes / sequential_seconds if sequential_seconds else None
                ),
            },
            "random_expert_reads": {
                "bytes": random_bytes,
                "seconds": random_seconds,
                "bandwidth_bytes_per_second": (
                    random_bytes / random_seconds if random_seconds else None
                ),
                "median_latency_ms": statistics.median(
                    seconds * 1000 for _, seconds in random_results
                ),
            },
            "parallel_read_scaling": parallel,
            "buffered_reads_supported": True,
            "direct_reads_supported": False,
            "direct_reads_reason": (
                "not requested: Python cannot prove Colibri-aligned Windows direct I/O"
            ),
            "cold_cache_measured": False,
            "cold_cache_reason": "OS page cache was not forcibly evicted",
            "warm_cache_measured": True,
        }
