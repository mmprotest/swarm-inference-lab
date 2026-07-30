"""Request route lifecycle and rerouting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from swarm_inference.config.models import StageReplica
from swarm_inference.exceptions import NoValidRouteError


@dataclass(slots=True)
class RequestRoute:
    request_id: str
    replicas: list[StageReplica]
    changes: list[dict[str, object]] = field(default_factory=list)

    def worker_for_stage(self, stage_id: int) -> str:
        for replica in self.replicas:
            if replica.stage_id == stage_id:
                return replica.worker_id
        raise NoValidRouteError(
            f"request {self.request_id} has no route entry for stage {stage_id}"
        )

    def replace(
        self,
        *,
        stage_id: int,
        replacement: StageReplica,
        reason: str,
        monotonic_s: float,
    ) -> None:
        for index, existing in enumerate(self.replicas):
            if existing.stage_id == stage_id:
                self.replicas[index] = replacement
                self.changes.append(
                    {
                        "stage_id": stage_id,
                        "from_worker": existing.worker_id,
                        "to_worker": replacement.worker_id,
                        "reason": reason,
                        "monotonic_s": monotonic_s,
                    }
                )
                return
        raise NoValidRouteError(
            f"cannot replace missing stage {stage_id} in request {self.request_id}"
        )
