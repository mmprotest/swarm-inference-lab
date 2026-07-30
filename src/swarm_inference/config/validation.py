"""Cross-object validation that is awkward to express in one Pydantic model."""

from __future__ import annotations

from collections.abc import Iterable

from swarm_inference.config.models import (
    Backend,
    ModelManifest,
    StageDefinition,
    StageReplica,
    WorkerCapability,
)
from swarm_inference.exceptions import (
    BackendIncompatibleError,
    InsufficientStageCoverageError,
    MemoryLimitExceededError,
    NoValidRouteError,
)


def validate_stage_coverage(
    stages: Iterable[StageDefinition],
    replicas: Iterable[StageReplica],
) -> None:
    expected = {stage.stage_id for stage in stages}
    available = {
        replica.stage_id
        for replica in replicas
        if replica.health.value == "healthy" and replica.load_status == "loaded"
    }
    missing = sorted(expected - available)
    if missing:
        raise InsufficientStageCoverageError(
            f"no healthy, loaded replica for stage(s): {', '.join(map(str, missing))}"
        )


def validate_route(route: list[StageReplica], stages: list[StageDefinition]) -> None:
    expected = [stage.stage_id for stage in sorted(stages, key=lambda item: item.stage_id)]
    actual = [replica.stage_id for replica in route]
    if actual != expected:
        raise NoValidRouteError(f"route stage order {actual!r} does not cover {expected!r}")


def validate_assignment(
    stage: StageDefinition,
    worker: WorkerCapability,
    manifest: ModelManifest | None = None,
) -> None:
    if stage.required_memory_bytes > worker.effective_memory_bytes:
        raise MemoryLimitExceededError(
            f"stage {stage.stage_id} requires {stage.required_memory_bytes} bytes, "
            f"worker {worker.worker_id} limit is {worker.effective_memory_bytes} bytes"
        )
    if manifest is not None and worker.backend not in manifest.compatible_worker_backends:
        raise BackendIncompatibleError(
            f"worker {worker.worker_id} backend {worker.backend.value} is not compatible "
            f"with model {manifest.model_id}"
        )
    if worker.backend == Backend.TORCH_CUDA and worker.available_vram_bytes <= 0:
        raise BackendIncompatibleError(
            f"worker {worker.worker_id} requested torch-cuda without available VRAM"
        )
