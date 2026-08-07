"""Historical exact-byte regression for the Experiment 010 fixture.

Universal product acceptance is architecture-neutral and does not select this module.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import torch

from swarm_inference.config.models import Backend, WorkerRole
from swarm_inference.config.product import ProductCoordinatorConfig
from swarm_inference.coordinator.service import (
    CoordinatorClient,
    CoordinatorCore,
    CoordinatorRpcServer,
)
from swarm_inference.model.product import ProductModelReference
from swarm_inference.protocol.messages import StreamEventType, SubmitRequest
from swarm_inference.protocol.product import ModelDeployRequest, ModelPlanRequest
from swarm_inference.security.identity import public_key_fingerprint
from swarm_inference.worker.service import run_worker

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REAL_MODEL_PATH = REPOSITORY_ROOT / "artifacts" / "models" / "colibri" / "source-b89a7c4bc24f"
REAL_REFERENCE_PATH = (
    REPOSITORY_ROOT
    / "artifacts"
    / "runs"
    / "experiment-010-correction-work"
    / "phase-6"
    / "local-correctness-references"
    / "code-01"
    / "reference.json"
)
REAL_MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"
REAL_MODEL_REVISION = "b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e"
REAL_TOKENIZER_REVISION = "sha256:d1e645ebd850d79567e531a3c103ac575d8e9cf45fa941420afc584b293438ea"
DEFAULT_EXPERT_ROOT = REPOSITORY_ROOT / "artifacts" / "acceptance" / "olmoe-expert-prerequisites"
WHOLE_MANIFEST = Path(
    os.environ.get(
        "SWARM_REAL_WHOLE_EXPERT_MANIFEST",
        str(DEFAULT_EXPERT_ROOT / "whole-expert-manifest.json"),
    )
)
MICROSHARD_MANIFESTS = tuple(
    Path(value)
    for value in os.environ.get(
        "SWARM_REAL_MICROSHARD_MANIFESTS",
        os.pathsep.join(
            (
                str(DEFAULT_EXPERT_ROOT / "microshard-0" / "manifest.json"),
                str(DEFAULT_EXPERT_ROOT / "microshard-1" / "manifest.json"),
            )
        ),
    ).split(os.pathsep)
    if value
)
STAGE_MEMORY_BYTES = 16_000_000_000
EXPERT_MEMORY_BYTES = 16_000_000_000
EXPERT_CACHE_BYTES = 6_000_000_000


@dataclass(frozen=True, slots=True)
class _ExpertWorker:
    worker_id: str
    role: WorkerRole
    manifest: Path
    control_endpoint: str
    data_endpoint: str


def _reserve_endpoints(count: int) -> list[str]:
    listeners: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listeners.append(listener)
        return [f"127.0.0.1:{listener.getsockname()[1]}" for listener in listeners]
    finally:
        for listener in listeners:
            listener.close()


def _manifest_is_real(path: Path, *, microshard: bool) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        value.get("model_id") != REAL_MODEL_ID
        or value.get("model_revision") != REAL_MODEL_REVISION
        or not value.get("owned_experts")
    ):
        return False
    ownership = value.get("owned_microshards", [])
    return bool(ownership) if microshard else value.get("loader_type") == "safetensors"


def _write_gate_evidence(name: str, payload: dict[str, object]) -> None:
    configured = os.environ.get("SWARM_ACCEPTANCE_GATE_EVIDENCE")
    if not configured:
        return
    directory = Path(configured).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evidence_provenance(reference: dict[str, object]) -> dict[str, object]:
    metadata_hasher = hashlib.sha256()
    for relative in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
        path = REAL_MODEL_PATH / relative
        if path.is_file():
            metadata_hasher.update(relative.encode())
            metadata_hasher.update(path.read_bytes())
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "model_metadata_hash": f"sha256:{metadata_hasher.hexdigest()}",
        "prompt": str(reference.get("prompt", reference.get("text", ""))),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status": git_status,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "os": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
        },
    }


async def _wait_for_workers(
    core: CoordinatorCore,
    tasks: list[asyncio.Task[None]],
    expected_worker_ids: set[str],
) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        for task in tasks:
            if task.done() and (error := task.exception()) is not None:
                raise RuntimeError(f"real product worker {task.get_name()} failed") from error
        if expected_worker_ids <= {worker.worker_id for worker in core.registry.workers()}:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(f"real product workers did not register: {sorted(expected_worker_ids)}")


async def _run_real_remote_expert_product_inference(
    tmp_path: Path,
    *,
    policy: Literal["whole-remote", "microshard-remote"],
    manifests: tuple[Path, ...],
) -> None:
    reference = json.loads(REAL_REFERENCE_PATH.read_text(encoding="utf-8"))
    prompt_ids = [int(value) for value in reference["prompt_ids"]]
    expected_tokens = [int(reference["full_ids"][len(prompt_ids)])]
    config = ProductCoordinatorConfig(
        require_trusted_workers=False,
        worker_heartbeat_timeout_s=60,
        control_timeout_s=300,
        request_timeout_s=600,
        event_queue_capacity=64,
        token_ingress_capacity=64,
    )
    core = CoordinatorCore(product_config=config, state_directory=tmp_path / "coordinator")
    assert core.coordinator_identity is not None
    coordinator = CoordinatorRpcServer(core)
    coordinator_port = await coordinator.start("127.0.0.1:0")
    coordinator_endpoint = f"127.0.0.1:{coordinator_port}"
    coordinator_fingerprint = core.coordinator_identity.public_key_fingerprint

    endpoints = _reserve_endpoints(4 + len(manifests) * 2)
    stage_endpoints = endpoints[:4]
    expert_endpoints = endpoints[4:]
    expert_workers = [
        _ExpertWorker(
            worker_id=f"real-{policy}-worker-{index}",
            role=(
                WorkerRole.WHOLE_EXPERT
                if policy == "whole-remote"
                else WorkerRole.EXPERT_MICROSHARD
            ),
            manifest=manifest,
            control_endpoint=expert_endpoints[index * 2],
            data_endpoint=expert_endpoints[index * 2 + 1],
        )
        for index, manifest in enumerate(manifests)
    ]
    stop_event = asyncio.Event()
    worker_tasks = [
        asyncio.create_task(
            run_worker(
                coordinator_endpoint=coordinator_endpoint,
                listen_endpoint=stage_endpoints[stage_id * 2],
                advertised_endpoint=stage_endpoints[stage_id * 2],
                backend=Backend.TORCH_CUDA,
                memory_limit_bytes=STAGE_MEMORY_BYTES,
                identity_path=tmp_path / f"real-stage-worker-{stage_id}.json",
                worker_id=f"real-stage-worker-{stage_id}",
                stage_runtime_enabled=True,
                data_listen_endpoint=stage_endpoints[stage_id * 2 + 1],
                data_advertised_endpoint=stage_endpoints[stage_id * 2 + 1],
                device="cuda:0",
                dtype="bfloat16",
                configured_model_path=REAL_MODEL_PATH,
                stop_event=stop_event,
                max_stage_sessions=2,
                stage_execution_queue_capacity=8,
                token_publication_queue_capacity=8,
                upload_bandwidth_bytes_s=1_000_000_000,
                download_bandwidth_bytes_s=1_000_000_000,
                network_rates_measured=True,
                trusted_coordinator_fingerprint=coordinator_fingerprint,
                worker_roles={WorkerRole.CONTIGUOUS_STAGE},
            ),
            name=f"real-stage-worker-{stage_id}",
        )
        for stage_id in range(2)
    ]
    for worker in expert_workers:
        worker_tasks.append(
            asyncio.create_task(
                run_worker(
                    coordinator_endpoint=coordinator_endpoint,
                    listen_endpoint=worker.control_endpoint,
                    advertised_endpoint=worker.control_endpoint,
                    backend=Backend.TORCH_CPU,
                    memory_limit_bytes=EXPERT_MEMORY_BYTES,
                    identity_path=tmp_path / f"{worker.worker_id}.json",
                    worker_id=worker.worker_id,
                    stop_event=stop_event,
                    upload_bandwidth_bytes_s=1_000_000_000,
                    download_bandwidth_bytes_s=1_000_000_000,
                    network_rates_measured=True,
                    trusted_coordinator_fingerprint=coordinator_fingerprint,
                    worker_roles={worker.role},
                    expert_manifest_path=worker.manifest,
                    expert_data_listen_endpoint=worker.data_endpoint,
                    expert_data_advertised_endpoint=worker.data_endpoint,
                    expert_residency_budget_bytes=EXPERT_CACHE_BYTES,
                    expert_cache_budget_bytes=EXPERT_CACHE_BYTES,
                    expert_queue_capacity=32,
                    expert_max_concurrent_requests=1,
                ),
                name=worker.worker_id,
            )
        )

    client = CoordinatorClient(coordinator_endpoint, timeout_s=600)
    try:
        expected_workers = {
            "real-stage-worker-0",
            "real-stage-worker-1",
            *(worker.worker_id for worker in expert_workers),
        }
        await _wait_for_workers(core, worker_tasks, expected_workers)
        planned = await client.plan_model(
            ModelPlanRequest(
                reference=ProductModelReference(
                    model_id=REAL_MODEL_ID,
                    model_revision=REAL_MODEL_REVISION,
                    tokenizer_revision=REAL_TOKENIZER_REVISION,
                    dtype="bfloat16",
                ),
                stage_count=2,
                partition_method="equal",
                require_distributed=True,
                max_sequence_tokens=len(prompt_ids) + 1,
                expert_policy=policy,
                require_remote_experts=True,
                allow_expert_local_fallback=False,
            )
        )
        placements = [
            placement
            for expert_plan in planned.plan.expert_plans
            for placement in expert_plan.placements
        ]
        assert len(placements) == 16 * 64
        assert all(
            placement.strategy == policy
            and placement.forced_remote
            and not placement.local_fallback_permitted
            for placement in placements
        )
        if policy == "microshard-remote":
            assert all(
                len(placement.worker_ids) == 2
                and all(
                    int(shard["hidden_end"]) - int(shard["hidden_start"])
                    < int(shard["logical_intermediate_dimension"])
                    for shard in placement.microshards
                )
                for placement in placements
            )
        deployed = await client.deploy_model(ModelDeployRequest(plan=planned.plan))
        assert deployed.deployment.ready

        events = []
        async for event in client.submit_stream(
            SubmitRequest(
                request_id=f"real-{policy}-acceptance",
                prompt_token_ids=prompt_ids,
                max_new_tokens=1,
                random_seed=0,
                model_id=REAL_MODEL_ID,
                model_revision=REAL_MODEL_REVISION,
            )
        ):
            events.append(event)
        generated = [
            event for event in events if event.event_type == StreamEventType.TOKEN_GENERATED
        ]
        assert [event.token_id for event in generated] == expected_tokens
        assert events[-1].event_type == StreamEventType.REQUEST_COMPLETED
        assert events[-1].final_token_ids == expected_tokens
        expected_trace = (
            "remote_whole_expert_result_consumed"
            if policy == "whole-remote"
            else "remote_microshard_result_consumed"
        )
        assert generated[0].expert_trace
        assert all(
            item["event"] == expected_trace
            and item["request_bytes"] > 0
            and item["response_bytes"] > 0
            and item["fallback_reason"] is None
            for item in generated[0].expert_trace
        )
        status = await client.status()
        assert status.expert_worker_count == len(expert_workers)
        assert status.expert_bytes_transferred > 0
        assert status.expert_fallbacks == 0
        if policy == "whole-remote":
            assert status.remote_expert_calls > 0
            assert status.remote_microshard_calls == 0
        else:
            assert status.remote_expert_calls == 0
            assert status.remote_microshard_calls > 0
        _write_gate_evidence(
            f"real-{policy}.json",
            {
                "document_type": "swarm-real-model-gate-evidence",
                "format_version": 2,
                "gate": policy,
                "status": "PASS",
                "model_id": REAL_MODEL_ID,
                "model_revision": REAL_MODEL_REVISION,
                "tokenizer_revision": REAL_TOKENIZER_REVISION,
                **_evidence_provenance(reference),
                "prompt_token_ids": prompt_ids,
                "topology": {
                    "topology_id": planned.plan.topology_id,
                    "route_generation": planned.plan.generation,
                    "stage_count": planned.plan.stage_count,
                    "expert_policy": policy,
                },
                "stage_assignments": [
                    assignment.model_dump(mode="json") for assignment in planned.plan.assignments
                ],
                "worker_identities": [
                    {
                        "worker_id": worker.worker_id,
                        "fingerprint": public_key_fingerprint(worker.public_key),
                    }
                    for worker in core.registry.workers()
                ],
                "worker_pids": {
                    worker.worker_id: os.getpid() for worker in core.registry.workers()
                },
                "expert_assignments": [
                    placement.model_dump(mode="json") for placement in placements
                ],
                "generated_token_ids": [event.token_id for event in generated],
                "expected_token_ids": expected_tokens,
                "token_ids": expected_tokens,
                "bytes_transferred": {
                    "expert_bytes_transferred": status.expert_bytes_transferred,
                    "coordinator_activation_bytes": core.runtime_transport_metrics[
                        "coordinator_activation_bytes"
                    ],
                },
                "critical_path_timings": events[-1].timing_metrics,
                "timings": events[-1].timing_metrics,
                "fallback_count": status.expert_fallbacks,
                "recovery_events": [],
                "route_generations": [planned.plan.generation],
                "expert_metrics": generated[0].expert_metrics,
                "expert_trace": generated[0].expert_trace,
                "distributed_ownership": [
                    {
                        "worker_id": worker.worker_id,
                        "manifest": str(worker.manifest),
                        "manifest_sha256": f"sha256:{hashlib.sha256(worker.manifest.read_bytes()).hexdigest()}",
                    }
                    for worker in expert_workers
                ],
            },
        )
    finally:
        stop_event.set()
        try:
            await asyncio.wait_for(
                asyncio.gather(*worker_tasks, return_exceptions=True),
                timeout=60,
            )
        except TimeoutError:
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)
        await client.close()
        await coordinator.stop(grace_s=0)
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SWARM_RUN_LEGACY_OLMOE_CUDA_REGRESSION") != "1"
    or not torch.cuda.is_available()
    or not REAL_MODEL_PATH.is_dir()
    or not REAL_REFERENCE_PATH.is_file()
    or not _manifest_is_real(WHOLE_MANIFEST, microshard=False),
    reason=(
        "real whole-expert acceptance requires CUDA, the pinned OLMoE snapshot, and a "
        "historical manifest prepared by scripts/prepare_experiment_010_olmoe_regression.py"
    ),
)
@pytest.mark.asyncio
async def test_legacy_fixture_whole_expert_regression(tmp_path: Path) -> None:
    await _run_real_remote_expert_product_inference(
        tmp_path,
        policy="whole-remote",
        manifests=(WHOLE_MANIFEST,),
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SWARM_RUN_LEGACY_OLMOE_CUDA_REGRESSION") != "1"
    or not torch.cuda.is_available()
    or not REAL_MODEL_PATH.is_dir()
    or not REAL_REFERENCE_PATH.is_file()
    or len(MICROSHARD_MANIFESTS) != 2
    or not all(_manifest_is_real(path, microshard=True) for path in MICROSHARD_MANIFESTS),
    reason=(
        "real native-microshard acceptance requires CUDA, the pinned OLMoE snapshot, and two "
        "physically sliced manifests prepared by scripts/prepare_experiment_010_olmoe_regression.py"
    ),
)
@pytest.mark.asyncio
async def test_legacy_fixture_native_microshard_regression(tmp_path: Path) -> None:
    await _run_real_remote_expert_product_inference(
        tmp_path,
        policy="microshard-remote",
        manifests=MICROSHARD_MANIFESTS,
    )
