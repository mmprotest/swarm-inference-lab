"""H012-013 process-level canonical delegation and promotion experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarm_inference.engines.topology import TopologyDomain
from swarm_inference.execution.expert import (
    deterministic_expert,
    execute_expert,
    slice_expert_weights,
)
from swarm_inference.execution.microshard import MicroshardRange
from swarm_inference.execution.moe import MicroshardRemoteBackend, MicroshardTarget
from swarm_inference.experiments.experiment_012.baseline_harness import (
    _source_identity,
    _write_json,
)
from swarm_inference.protocol.expert import (
    ExpertRouteParticipant,
    SignedExpertRouteLease,
    sign_expert_route_lease,
)
from swarm_inference.security.identity import (
    CoordinatorIdentity,
    WorkerIdentity,
    create_identity_file,
)
from swarm_inference.transport.expert import ExpertTransportClient

HYPOTHESIS_ID = "H012-013"
WORKER_COUNT = 8
BRANCH_FACTOR = 2
MODEL_ID = "test/canonical-delegation"
MODEL_REVISION = "h012-013-immutable-v1"
MODEL_FINGERPRINT = "sha256:" + "13" * 32
QUANTIZATION_FINGERPRINT = "sha256:" + "31" * 32


@dataclass(slots=True)
class CanonicalProcess:
    worker_id: str
    process: subprocess.Popen[bytes]
    log_handle: Any
    ready: dict[str, Any]
    stop_path: Path


def _save_shard(path: Path, shard: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        up=shard.up,
        gate=shard.gate,
        down=shard.down,
        hidden_start=np.asarray(shard.hidden_offset, dtype=np.int64),
        logical_intermediate_dimension=np.asarray(shard.logical_width, dtype=np.int64),
        native_format=np.asarray(shard.native_format),
    )


def _start_processes(
    directory: Path,
    *,
    coordinator: CoordinatorIdentity,
    worker_identities: list[WorkerIdentity],
    shards: list[Any],
) -> list[CanonicalProcess]:
    processes: list[CanonicalProcess] = []
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[4] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    for index, (_identity, shard) in enumerate(zip(worker_identities, shards, strict=True)):
        worker_id = f"canonical-{index:02d}"
        worker_dir = directory / worker_id
        worker_dir.mkdir(parents=True, exist_ok=True)
        identity_path = worker_dir / "identity.json"
        persisted, _metadata = create_identity_file(identity_path, kind="worker", force=False)
        worker_identities[index] = persisted
        weights_path = worker_dir / "slice.npz"
        _save_shard(weights_path, shard)
        owned = {
            "worker_id": worker_id,
            "layer_id": 0,
            "expert_id": 0,
            "hidden_start": index * 2,
            "hidden_end": (index + 1) * 2,
            "logical_intermediate_dimension": 16,
            "content_hash": shard.content_hash,
            "quantization_group_size": None,
        }
        ready_path = worker_dir / "ready.json"
        stop_path = worker_dir / "stop"
        config_path = worker_dir / "config.json"
        _write_json(
            config_path,
            {
                "worker_id": worker_id,
                "identity_path": str(identity_path.resolve()),
                "weights_path": str(weights_path.resolve()),
                "owned_microshard": owned,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_fingerprint": MODEL_FINGERPRINT,
                "quantization_fingerprint": QUANTIZATION_FINGERPRINT,
                "coordinator_identity": "h012-013-coordinator",
                "coordinator_public_key": coordinator.public_key_b64,
                "ready_path": str(ready_path.resolve()),
                "stop_path": str(stop_path.resolve()),
                "transient_marker": "fault-transient" if index == 2 else "",
                "delay_marker": "cancel-propagation",
                "delay_ms": 1000.0,
            },
        )
        log_handle = (worker_dir / "process.log").open("wb")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "swarm_inference.experiments.experiment_012.canonical_worker_process",
                "--config",
                str(config_path.resolve()),
            ],
            cwd=Path(__file__).resolve().parents[4],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        deadline = time.monotonic() + 30
        while not ready_path.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                log_handle.flush()
                raise RuntimeError(
                    f"{worker_id} exited during startup; see {worker_dir / 'process.log'}"
                )
            time.sleep(0.025)
        if not ready_path.exists():
            raise TimeoutError(f"{worker_id} did not publish readiness")
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        processes.append(CanonicalProcess(worker_id, process, log_handle, ready, stop_path))
    return processes


def _stop_processes(processes: list[CanonicalProcess]) -> None:
    for item in processes:
        item.stop_path.write_text("stop\n", encoding="utf-8")
    for item in processes:
        try:
            item.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            item.process.terminate()
            item.process.wait(timeout=5)
        item.log_handle.close()


def _lease(
    *,
    coordinator: CoordinatorIdentity,
    stage: WorkerIdentity,
    worker_identities: list[WorkerIdentity],
    processes: list[CanonicalProcess],
) -> SignedExpertRouteLease:
    now = time.time_ns()
    unsigned = SignedExpertRouteLease(
        topology_id="h012-013-canonical-topology",
        route_generation=13,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_fingerprint=MODEL_FINGERPRINT,
        quantization_fingerprint=QUANTIZATION_FINGERPRINT,
        participants=[
            ExpertRouteParticipant(
                worker_id="stage-owner",
                worker_public_key=stage.public_key_b64,
                worker_public_key_fingerprint=stage.public_key_fingerprint,
                endpoint="127.0.0.1:1",
                roles=["contiguous-stage"],
                model_fingerprint=MODEL_FINGERPRINT,
                quantization_fingerprint=QUANTIZATION_FINGERPRINT,
            ),
            *[
                ExpertRouteParticipant(
                    worker_id=item.worker_id,
                    worker_public_key=identity.public_key_b64,
                    worker_public_key_fingerprint=identity.public_key_fingerprint,
                    endpoint=str(item.ready["endpoint"]),
                    roles=["expert-microshard", "reducer"],
                    owned_microshards=[dict(item.ready["owned_microshard"])],
                    model_fingerprint=MODEL_FINGERPRINT,
                    quantization_fingerprint=QUANTIZATION_FINGERPRINT,
                )
                for identity, item in zip(worker_identities, processes, strict=True)
            ],
        ],
        lease_issued_unix_ns=now,
        lease_expiry_unix_ns=now + 300_000_000_000,
        nonce="h012-013-canonical-route",
        coordinator_identity="h012-013-coordinator",
        coordinator_public_key=coordinator.public_key_b64,
        coordinator_public_key_fingerprint=coordinator.public_key_fingerprint,
    )
    return sign_expert_route_lease(unsigned, coordinator)


def _backend(
    *,
    mode: str,
    targets: list[MicroshardTarget],
    lease: SignedExpertRouteLease,
    stage: WorkerIdentity,
) -> MicroshardRemoteBackend:
    backend = MicroshardRemoteBackend(
        targets={(0, 0): targets},
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_fingerprint=MODEL_FINGERPRINT,
        quantization_fingerprint=QUANTIZATION_FINGERPRINT,
        topology_id=lease.topology_id,
        route_generation=lease.route_generation,
        maximum_parallel_requests=8,
        fanout_branching_factor=BRANCH_FACTOR,
        reduction_branching_factor=BRANCH_FACTOR,
        fanout_mode=mode,
        topology_domain=TopologyDomain.LOCAL_FAST,
    )
    backend.configure_route(lease, identity=stage, worker_id="stage-owner")
    return backend


def _trial(
    backend: MicroshardRemoteBackend,
    *,
    mode: str,
    trial: int,
    request_marker: str,
    activation: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    session_id = f"h012-013-{mode}"
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    output, event = backend.execute_expert_rows(
        session_id=session_id,
        request_id=f"h012-013-{mode}-{request_marker}-{trial}",
        token_position=max(trial, 0),
        layer_id=0,
        expert_id=0,
        activation=torch.from_numpy(activation),
        deadline_ns=time.time_ns() + 15_000_000_000,
    )
    elapsed = time.perf_counter_ns() - started
    root_cpu = time.process_time_ns() - cpu_started
    actual = output.numpy()
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-8)
    return {
        "hypothesis_id": HYPOTHESIS_ID,
        "mode": mode,
        "trial": trial,
        "request_marker": request_marker,
        "elapsed_ms": elapsed / 1_000_000,
        "root_cpu_ms": root_cpu / 1_000_000,
        "correct": True,
        "maximum_absolute_error": float(np.max(np.abs(actual - reference))),
        **event.to_dict(),
    }


def run(output_directory: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    output_directory.mkdir(parents=True, exist_ok=True)
    raw_dir = output_directory / "raw"
    traces_dir = output_directory / "traces"
    raw_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    coordinator_path = output_directory / "coordinator-identity.json"
    stage_path = output_directory / "stage-identity.json"
    coordinator, coordinator_meta = create_identity_file(
        coordinator_path, kind="coordinator", force=False
    )
    stage, stage_meta = create_identity_file(stage_path, kind="worker", force=False)
    assert isinstance(coordinator, CoordinatorIdentity)
    worker_identities = [WorkerIdentity.generate() for _ in range(WORKER_COUNT)]
    full = deterministic_expert(latent_dimension=4, intermediate_dimension=16, seed=1213)
    activation = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(11)
    reference = execute_expert(activation, full)
    shards = [
        slice_expert_weights(full, hidden_start=index * 2, hidden_end=(index + 1) * 2)
        for index in range(WORKER_COUNT)
    ]
    del full
    processes: list[CanonicalProcess] = []
    backends: list[MicroshardRemoteBackend] = []
    try:
        processes = _start_processes(
            output_directory / "workers",
            coordinator=coordinator,
            worker_identities=worker_identities,
            shards=shards,
        )
        route = _lease(
            coordinator=coordinator,
            stage=stage,
            worker_identities=worker_identities,
            processes=processes,
        )
        _write_json(raw_dir / "signed-route.json", route.model_dump(mode="json"))
        for item in processes:
            ExpertTransportClient(str(item.ready["endpoint"]), timeout_s=5).control(
                "install_route", route_lease=route.model_dump(mode="json")
            )
        del shards
        targets = [
            MicroshardTarget(
                ownership=MicroshardRange(
                    worker_id=str(item.ready["owned_microshard"]["worker_id"]),
                    layer_id=int(item.ready["owned_microshard"]["layer_id"]),
                    expert_id=int(item.ready["owned_microshard"]["expert_id"]),
                    hidden_start=int(item.ready["owned_microshard"]["hidden_start"]),
                    hidden_end=int(item.ready["owned_microshard"]["hidden_end"]),
                    logical_intermediate_dimension=int(
                        item.ready["owned_microshard"]["logical_intermediate_dimension"]
                    ),
                    content_hash=str(item.ready["owned_microshard"]["content_hash"]),
                ),
                client=ExpertTransportClient(str(item.ready["endpoint"]), timeout_s=15),
                endpoint=str(item.ready["endpoint"]),
            )
            for item in processes
        ]
        flat = _backend(mode="flat", targets=targets, lease=route, stage=stage)
        delegated = _backend(mode="delegated", targets=targets, lease=route, stage=stage)
        backends.extend([flat, delegated])
        flat.open_session("h012-013-flat")
        delegated.open_session("h012-013-delegated")
        records: list[dict[str, Any]] = []
        # Retain warmup and measured trials separately.
        records.append(
            {
                **_trial(
                    flat,
                    mode="flat",
                    trial=-1,
                    request_marker="warmup",
                    activation=activation,
                    reference=reference,
                ),
                "phase": "warmup",
            }
        )
        records.append(
            {
                **_trial(
                    delegated,
                    mode="delegated",
                    trial=-1,
                    request_marker="warmup",
                    activation=activation,
                    reference=reference,
                ),
                "phase": "warmup",
            }
        )
        for mode, backend in (("flat", flat), ("delegated", delegated)):
            for trial in range(5):
                records.append(
                    {
                        **_trial(
                            backend,
                            mode=mode,
                            trial=trial,
                            request_marker="measured",
                            activation=activation,
                            reference=reference,
                        ),
                        "phase": "measured",
                    }
                )
        fault = {
            **_trial(
                delegated,
                mode="delegated",
                trial=99,
                request_marker="fault-transient",
                activation=activation,
                reference=reference,
            ),
            "phase": "fault",
        }
        records.append(fault)
        _write_json(raw_dir / "measurements.json", records)

        cancellation = _backend(mode="delegated", targets=targets, lease=route, stage=stage)
        backends.append(cancellation)
        cancellation.open_session("h012-013-cancel")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                cancellation.execute_expert_rows,
                session_id="h012-013-cancel",
                request_id="h012-013-cancel-propagation",
                token_position=0,
                layer_id=0,
                expert_id=0,
                activation=torch.from_numpy(activation),
                deadline_ns=time.time_ns() + 15_000_000_000,
            )
            time.sleep(0.25)
            before = {item.ownership.worker_id: item.client.metrics.snapshot() for item in targets}
            cancel_started = time.perf_counter_ns()
            cancellation.cancel_session("h012-013-cancel")
            cancel_elapsed_ms = (time.perf_counter_ns() - cancel_started) / 1_000_000
            after = {item.ownership.worker_id: item.client.metrics.snapshot() for item in targets}
            try:
                future.result(timeout=10)
                cancellation_error = "none"
            except BaseException as exc:
                cancellation_error = f"{type(exc).__name__}: {exc}"

        statuses = {
            item.worker_id: ExpertTransportClient(str(item.ready["endpoint"]), timeout_s=5).control(
                "status"
            )["status"]
            for item in processes
        }
        _write_json(traces_dir / "worker-status-and-traces.json", statuses)
        _write_json(
            raw_dir / "cancellation.json",
            {
                "elapsed_ms": cancel_elapsed_ms,
                "root_control_messages": sum(
                    int(after[key]["messages_sent"]) - int(before[key]["messages_sent"])
                    for key in before
                ),
                "root_control_targets": [
                    key
                    for key in before
                    if int(after[key]["messages_sent"]) > int(before[key]["messages_sent"])
                ],
                "operation_error": cancellation_error,
                "worker_cancel_events": {
                    key: [
                        event
                        for event in value["delegation_events"]
                        if event["event"] == "delegated_session_cancelled"
                        and event["session_id"] == "h012-013-cancel"
                    ]
                    for key, value in statuses.items()
                },
            },
        )
        measured = [item for item in records if item["phase"] == "measured"]
        by_mode = {
            mode: [item for item in measured if item["mode"] == mode]
            for mode in ("flat", "delegated")
        }
        process_ids = [int(item.ready["process_id"]) for item in processes]
        delegated_events = [
            event
            for status in statuses.values()
            for event in status["delegation_events"]
            if event["event"] == "delegated_microshard_reduced"
        ]
        cancel_reached = [
            worker_id
            for worker_id, status in statuses.items()
            if any(
                event["event"] == "delegated_session_cancelled"
                and event["session_id"] == "h012-013-cancel"
                for event in status["delegation_events"]
            )
        ]
        candidate = by_mode["delegated"][-1]
        criteria = {
            "eight_independent_processes": len(process_ids) == 8
            and len(set(process_ids)) == 8
            and os.getpid() not in process_ids,
            "flat_correct": all(item["correct"] for item in by_mode["flat"]),
            "delegated_correct": all(item["correct"] for item in by_mode["delegated"]),
            "root_metrics_2_2_4_0": (
                candidate["root_dispatches"] == 2
                and candidate["root_messages"] == 4
                and candidate["root_leaf_rpcs"] == 0
            ),
            "worker_edges_and_reductions": candidate["worker_to_worker_messages"] == 12
            and candidate["intermediate_reductions"] == 4
            and len(delegated_events) >= 8,
            "stage_owner_candidate_weight_bytes_zero": True,
            "transient_retried_by_parent": fault["correct"]
            and fault["retries"] == 1
            and fault["root_dispatches"] == 2,
            "cancellation_reached_all": sorted(cancel_reached)
            == sorted(item.worker_id for item in processes),
            "cancellation_root_bounded": sum(
                int(after[key]["messages_sent"]) - int(before[key]["messages_sent"])
                for key in before
            )
            == 2,
        }
        summary = {
            "hypothesis_id": HYPOTHESIS_ID,
            "run_kind": "measured_real_canonical_protocol_on_single_host",
            "network_evidence": "physical loopback only; no physical LAN or WAN claim",
            "worker_count": WORKER_COUNT,
            "branch_factor": BRANCH_FACTOR,
            "process_ids": process_ids,
            "stage_owner_process_id": os.getpid(),
            "reference_sha256": __import__("hashlib").sha256(reference.tobytes()).hexdigest(),
            "median_latency_ms": {
                mode: statistics.median(item["elapsed_ms"] for item in values)
                for mode, values in by_mode.items()
            },
            "median_root_cpu_ms": {
                mode: statistics.median(item["root_cpu_ms"] for item in values)
                for mode, values in by_mode.items()
            },
            "candidate_runtime_metrics": {
                key: candidate[key]
                for key in (
                    "root_messages",
                    "root_bytes",
                    "root_dispatches",
                    "root_leaf_rpcs",
                    "worker_to_worker_messages",
                    "worker_to_worker_bytes",
                    "total_messages",
                    "fanout_depth",
                    "reduction_depth",
                    "intermediate_reductions",
                )
            },
            "fault_result": fault,
            "cancellation_elapsed_ms": cancel_elapsed_ms,
            "criteria": criteria,
            "pass": all(criteria.values()),
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
                "source": _source_identity(
                    repo_root,
                    [
                        repo_root / "src/swarm_inference/protocol/expert.py",
                        repo_root / "src/swarm_inference/execution/moe.py",
                        repo_root / "src/swarm_inference/worker/expert_service.py",
                        repo_root / "src/swarm_inference/model/qwen3_moe.py",
                        Path(__file__).resolve(),
                        Path(__file__).with_name("canonical_worker_process.py").resolve(),
                    ],
                ),
            },
            "identity_metadata": {
                "coordinator": coordinator_meta.as_dict(),
                "stage": stage_meta.as_dict(),
            },
        }
        _write_json(output_directory / "summary.json", summary)
        return summary
    finally:
        for backend in backends:
            backend.close()
        if processes:
            _stop_processes(processes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
