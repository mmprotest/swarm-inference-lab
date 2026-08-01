"""Interface-only declarations for backends without local physical hardware."""

from __future__ import annotations

from swarm_inference.worker.abi import (
    BackendAdapter,
    BackendInterfaceEvidence,
    WorkerBenchmarkProfile,
    WorkerCapabilities,
    WorkerJob,
    WorkerJobResult,
    WorkerJobStatus,
    WorkerJobType,
)


class InterfaceOnlyAdapter(BackendAdapter):
    supported_jobs = frozenset(WorkerJobType)

    def __init__(
        self,
        evidence: BackendInterfaceEvidence,
        capabilities: WorkerCapabilities,
    ) -> None:
        self.evidence = evidence
        self.backend_id = evidence.backend_id
        self._capabilities = capabilities

    def capabilities(self) -> WorkerCapabilities:
        return self._capabilities

    def benchmark_profile(self) -> WorkerBenchmarkProfile:
        return WorkerBenchmarkProfile(model_load_seconds=0.0, warmup_seconds=0.0)

    async def execute(self, job: WorkerJob) -> WorkerJobResult:
        return WorkerJobResult(
            job_id=job.job_id,
            request_id=job.request_id,
            status=WorkerJobStatus.UNSUPPORTED,
            detail=(f"{self.backend_id}: interface_defined; physical_execution_unproven"),
        )

    async def cancel(self, request_id: str) -> bool:
        return False
