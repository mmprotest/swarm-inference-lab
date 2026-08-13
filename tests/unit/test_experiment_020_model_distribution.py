from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from swarm_inference.experiments.experiment_020.model_distribution import ShardCache


def test_cache_resumes_verifies_and_deduplicates_concurrent_consumers(tmp_path: Path) -> None:
    content = (b"0123456789abcdef" * 4096) + b"tail"
    digest = hashlib.sha256(content).hexdigest()
    cache = ShardCache(tmp_path / "cache")
    _final, partial, _marker = cache._paths(digest)
    partial.write_bytes(content[:12345])
    calls: list[int] = []

    def fetch(_url: str, start: int, stop: int | None):
        calls.append(start)
        yield content[start:stop]

    paths: list[Path] = []

    def acquire() -> None:
        paths.append(cache.acquire("memory://fixture", digest, len(content), fetcher=fetch))

    threads = [threading.Thread(target=acquire) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == content
    assert calls == [12345]
    assert paths[0].with_name(f"{digest}.complete").is_file()


def test_cache_rejects_hash_mismatch_without_completion_marker(tmp_path: Path) -> None:
    cache = ShardCache(tmp_path / "cache")
    content = b"wrong"
    digest = hashlib.sha256(b"right").hexdigest()

    def fetch(_url: str, start: int, stop: int | None):
        yield content[start:stop]

    try:
        cache.acquire("memory://fixture", digest, len(content), fetcher=fetch)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("hash mismatch was accepted")
    final, _partial, marker = cache._paths(digest)
    assert not final.exists()
    assert not marker.exists()
