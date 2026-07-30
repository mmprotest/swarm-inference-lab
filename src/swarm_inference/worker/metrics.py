"""Worker-side operation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(slots=True)
class WorkerMetrics:
    worker_id: str
    operations: int = 0
    failures: int = 0
    bytes_received: int = 0
    bytes_sent: int = 0
    busy_time_s: float = 0.0
    service_times_s: list[float] = field(default_factory=list)
    maximum_queue_depth: int = 0

    def record_success(
        self,
        *,
        received_bytes: int,
        sent_bytes: int,
        service_s: float,
        queue_depth: int,
    ) -> None:
        self.operations += 1
        self.bytes_received += received_bytes
        self.bytes_sent += sent_bytes
        self.busy_time_s += service_s
        self.service_times_s.append(service_s)
        self.maximum_queue_depth = max(self.maximum_queue_depth, queue_depth)

    def snapshot(self) -> dict[str, object]:
        values = np.asarray(self.service_times_s, dtype=np.float64)
        result = asdict(self)
        result.update(
            {
                "mean_service_s": float(values.mean()) if values.size else 0.0,
                "p50_service_s": float(np.percentile(values, 50)) if values.size else 0.0,
                "p95_service_s": float(np.percentile(values, 95)) if values.size else 0.0,
                "p99_service_s": float(np.percentile(values, 99)) if values.size else 0.0,
            }
        )
        result.pop("service_times_s", None)
        return result
