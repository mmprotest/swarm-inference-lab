"""Independent canonical expert worker used only by the H012-013 harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from swarm_inference.execution.expert import ExpertStore, npz_expert_loader
from swarm_inference.security.identity import WorkerIdentity
from swarm_inference.worker.expert_service import ExpertWorkerRuntime, ExpertWorkerServer


class ControlledExpertStore(ExpertStore):
    """Real canonical store with explicit, traceable experiment fault controls."""

    def __init__(
        self,
        *args: Any,
        transient_marker: str = "",
        delay_marker: str = "",
        delay_ms: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transient_marker = transient_marker
        self.delay_marker = delay_marker
        self.delay_ms = delay_ms
        self.transient_failures = 0

    def execute(self, request: Any, activation: Any, down_accumulators: Any = None) -> Any:
        if (
            self.transient_marker
            and self.transient_marker in request.request_id
            and self.transient_failures == 0
        ):
            self.transient_failures += 1
            raise ConnectionError("injected one-shot canonical child failure")
        if self.delay_marker and self.delay_marker in request.request_id and self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        return super().execute(request, activation, down_accumulators)

    def status(self) -> dict[str, Any]:
        return {**super().status(), "injected_transient_failures": self.transient_failures}


async def _run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    worker_id = str(config["worker_id"])
    identity = WorkerIdentity.load(config["identity_path"])
    weights_path = Path(config["weights_path"])
    owned = dict(config["owned_microshard"])
    with __import__("numpy").load(weights_path, allow_pickle=False) as archive:
        tensor_bytes = sum(int(archive[name].nbytes) for name in ("up", "gate", "down"))
    store = ControlledExpertStore(
        owned={(0, 0)},
        loader=npz_expert_loader({(0, 0): weights_path}),
        residency_budget_bytes=tensor_bytes,
        cache_budget_bytes=tensor_bytes,
        transient_marker=str(config.get("transient_marker", "")),
        delay_marker=str(config.get("delay_marker", "")),
        delay_ms=float(config.get("delay_ms", 0.0)),
    )
    runtime = ExpertWorkerRuntime(
        worker_id=worker_id,
        identity=identity,
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        model_fingerprint=str(config["model_fingerprint"]),
        quantization_fingerprint=str(config["quantization_fingerprint"]),
        store=store,
        roles={"expert-microshard", "reducer"},
        owned_microshards=[owned],
        maximum_queue_depth=16,
        maximum_concurrent_requests=1,
        require_authenticated_routes=True,
        trusted_coordinators={
            str(config["coordinator_identity"]): str(config["coordinator_public_key"])
        },
    )
    server = ExpertWorkerServer(runtime, host="127.0.0.1", port=0)
    host, port = await server.start()
    ready = {
        "worker_id": worker_id,
        "process_id": os.getpid(),
        "endpoint": f"{host}:{port}",
        "identity_public_key": identity.public_key_b64,
        "identity_fingerprint": identity.public_key_fingerprint,
        "owned_microshard": owned,
        "resident_tensor_bytes": tensor_bytes,
        "roles": ["expert-microshard", "reducer"],
        "started_unix_ns": time.time_ns(),
    }
    ready_path = Path(config["ready_path"])
    ready_path.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stop_path = Path(config["stop_path"])
    try:
        while not stop_path.exists():
            await asyncio.sleep(0.05)
    finally:
        await server.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.config.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
