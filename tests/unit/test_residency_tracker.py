from __future__ import annotations

from swarm_inference.runtime.residency import (
    MovementRecord,
    ResidencyKind,
    ResidencyTier,
    ResidencyTracker,
)


def test_residency_tracks_physical_bytes_movement_and_cache_effects() -> None:
    tracker = ResidencyTracker(maximum_movement_records=2)
    tracker.put(
        allocation_id="weights",
        worker_id="worker-a",
        model_fingerprint="sha256:model",
        kind=ResidencyKind.MODEL_TENSOR,
        tier=ResidencyTier.VRAM,
        bytes=100,
        now_unix_ns=1,
    )
    tracker.put(
        allocation_id="kv",
        worker_id="worker-a",
        model_fingerprint="sha256:model",
        kind=ResidencyKind.KV_CACHE,
        tier=ResidencyTier.VRAM,
        bytes=20,
        now_unix_ns=2,
    )
    tracker.record_movement(
        MovementRecord(
            movement_id="move-1",
            model_fingerprint="sha256:model",
            source_worker_id=None,
            destination_worker_id="worker-a",
            source_tier=ResidencyTier.STORAGE,
            destination_tier=ResidencyTier.RAM,
            bytes=100,
            elapsed_ns=50,
            timestamp_unix_ns=3,
            reason="model acquisition",
        )
    )
    tracker.record_cache_effect(
        hit=False,
        bytes_loaded=100,
        prefetch_useful_bytes=80,
        prefetch_wasted_bytes=20,
        stall_ns=50,
    )

    snapshot = tracker.snapshot(now_unix_ns=4)

    assert snapshot.bytes_by_worker_tier == {"worker-a:vram": 120}
    assert snapshot.cache.misses == 1
    assert snapshot.cache.prefetch_useful_bytes == 80
    assert snapshot.cache.prefetch_wasted_bytes == 20
    assert snapshot.recent_movements[0].reason == "model acquisition"
