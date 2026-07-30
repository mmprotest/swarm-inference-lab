"""Worker lifecycle independent of transport implementation."""

from __future__ import annotations

from swarm_inference.config.models import (
    ModelManifest,
    QueueConfig,
    StageDefinition,
    SyntheticModelConfig,
    WorkerCapability,
)
from swarm_inference.protocol.messages import (
    ActivationRequest,
    ActivationResult,
    HealthResponse,
)
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.worker.execution import ExecutionEngine
from swarm_inference.worker.metrics import WorkerMetrics
from swarm_inference.worker.shard_manager import ShardManager


class WorkerAgent:
    def __init__(
        self,
        *,
        capability: WorkerCapability,
        identity: WorkerIdentity,
        queue_config: QueueConfig,
    ) -> None:
        if capability.memory_limit_bytes is None:
            raise ValueError("worker capability must declare an enforced memory limit")
        self.capability = capability
        self.identity = identity
        self.shards = ShardManager(memory_limit_bytes=capability.memory_limit_bytes)
        self.metrics = WorkerMetrics(worker_id=capability.worker_id)
        self.execution = ExecutionEngine(
            worker_id=capability.worker_id,
            identity=identity,
            shards=self.shards,
            queue_config=queue_config,
            metrics=self.metrics,
        )

    async def start(self) -> None:
        await self.execution.start()

    async def stop(self) -> None:
        await self.execution.stop()

    def assign_synthetic(
        self,
        *,
        config: SyntheticModelConfig,
        stage: object,
        corrupt: bool = False,
    ) -> None:
        from swarm_inference.config.models import StageDefinition

        if not isinstance(stage, StageDefinition):
            raise TypeError("stage must be StageDefinition")
        self.shards.load_synthetic(config=config, stage=stage, corrupt=corrupt)
        self.capability.current_shard_assignments = sorted(self.shards.modules)

    def assign_qwen3(
        self,
        *,
        config: dict[str, object],
        manifest: ModelManifest,
        stage: StageDefinition,
        shard_path: str,
        shard_hash: str,
        dtype: str | None,
    ) -> None:
        self.shards.load_qwen3(
            config=config,
            manifest=manifest,
            stage=stage,
            shard_path=shard_path,
            expected_hash=shard_hash,
            backend=self.capability.backend,
            dtype_name=dtype,
        )
        self.capability.current_shard_assignments = sorted(self.shards.modules)

    async def execute(self, request: ActivationRequest) -> ActivationResult:
        return await self.execution.submit(request)

    def cancel(self, request_id: str) -> None:
        for module in self.shards.modules.values():
            module.cancel(request_id)

    def health(self) -> HealthResponse:
        return HealthResponse(
            worker_id=self.capability.worker_id,
            healthy=True,
            queue_depth=self.execution.queue_depth,
            loaded_stages=sorted(self.shards.modules),
            detail="ready",
            proof=self.proof(),
        )

    def proof(self) -> dict[str, object]:
        return {
            "capability": self.capability.model_dump(mode="json"),
            "shards": self.shards.proof(),
            "metrics": self.metrics.snapshot(),
        }
