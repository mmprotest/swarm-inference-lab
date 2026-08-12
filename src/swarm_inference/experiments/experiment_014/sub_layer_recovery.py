"""Fail-closed recovery certification for the real Kimi expert collective."""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_014.batch_safety import _health_snapshot
from swarm_inference.experiments.experiment_014.complete_stage_batch import (
    _batch_boundaries,
)
from swarm_inference.experiments.experiment_014.cuda import (
    _array_fingerprint,
    _device_identity,
    _sha256_file,
)
from swarm_inference.experiments.experiment_014.persistent_stages import _stage_fixtures
from swarm_inference.experiments.experiment_014.sub_layer_microwork import (
    _atomic_json,
    _microwork_lifecycle_delta,
    _microwork_process_snapshot,
    _PersistentExpertCollective,
    _request,
    _start_worker,
    _WorkerProxy,
)

SCHEMA_VERSION = "experiment-014-k3-real-expert-recovery-v1"
FAULTS = ("partial", "stale", "timeout", "loss")


class _InjectedDispatcher:
    def __init__(
        self,
        collective: _PersistentExpertCollective,
        proxies: list[_WorkerProxy],
        fault: str,
    ) -> None:
        self.collective = collective
        self.proxies = {proxy.worker_id: proxy for proxy in proxies}
        self.fault = fault
        self.target_worker: int | None = None
        self.handle: Any | None = None
        self.record: dict[str, Any] | None = None

    def __call__(
        self, selected_ids: np.ndarray, latent_activations: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        selected = np.ascontiguousarray(selected_ids, dtype=np.int32)
        contacted = sorted(
            {
                self.collective.owner_by_expert[int(expert)]
                for expert in selected.reshape(-1).tolist()
            }
        )
        self.target_worker = contacted[0]
        if self.fault == "cancel":
            self.handle = self.collective.start_batch(selected, latent_activations)
            self.record = self.collective.cancel_batch(self.handle)
            raise RuntimeError("injected expert generation cancellation")
        test_faults = None
        delay_seconds = 0.3
        if self.fault in {"partial", "stale", "timeout", "duplicate", "loss"}:
            if self.fault == "timeout":
                worker_fault = "delay"
            elif self.fault == "loss":
                worker_fault = "loss_after_cuda"
            else:
                worker_fault = self.fault
            test_faults = {self.target_worker: worker_fault}
        self.handle = self.collective.start_batch(
            selected,
            latent_activations,
            test_faults=test_faults,
            test_delay_seconds=delay_seconds,
        )
        outputs, self.record = self.collective.collect_batch(self.handle)
        return outputs, self.record


def _start_group(
    checkpoint: Path,
    cuda_library: Path,
    *,
    device: int,
    response_timeout_seconds: float,
) -> tuple[list[_WorkerProxy], _PersistentExpertCollective, dict[str, Any]]:
    context = mp.get_context("spawn")
    proxies: list[_WorkerProxy] = []
    try:
        for worker_id in range(4):
            ownership = frozenset(
                expert for expert in range(896) if expert % 4 == worker_id
            )
            proxies.append(
                _start_worker(
                    context,
                    checkpoint=checkpoint,
                    cuda_library=cuda_library,
                    layer=89,
                    device=device,
                    worker_id=worker_id,
                    ownership=ownership,
                    batch_capacity=8,
                )
            )
        collective = _PersistentExpertCollective(
            proxies,
            latent=3584,
            response_timeout_seconds=response_timeout_seconds,
        )
        ready = {
            "workers": [proxy.ready for proxy in proxies],
            "worker_ids": [proxy.worker_id for proxy in proxies],
            "complete_disjoint_ownership": set(collective.owner_by_expert)
            == set(range(896)),
            "tracked_worker_bytes": [
                int(proxy.ready["tracked_worker_bytes"]) for proxy in proxies
            ],
        }
        return proxies, collective, ready
    except BaseException:
        for proxy in reversed(proxies):
            with suppress(Exception):
                proxy.close()
        raise


def _close_group(proxies: list[_WorkerProxy], device: int) -> dict[str, Any]:
    shutdown = []
    for proxy in reversed(proxies):
        with suppress(Exception):
            shutdown.append(proxy.close())
    return {
        "worker_shutdown": shutdown,
        "gpu_health": _health_snapshot(device),
    }


def _expected_from_receipt(reference: dict[str, Any]) -> tuple[dict[int, str], dict[int, list[int]]]:
    correctness = reference["batches"]["1"]["correctness"]
    fingerprints = {
        int(row["position"]): str(row["reference_fingerprint"])
        for row in correctness["comparisons"]
    }
    routes = {
        index: [int(value) for value in row["selected_expert_ids"][0]]
        for index, row in enumerate(correctness["external_records"])
    }
    return fingerprints, routes


def _execute_positions(
    executor: PersistentKimiStageExecutor,
    collective: _PersistentExpertCollective,
    fixtures: list[np.ndarray],
    expected_fingerprints: dict[int, str],
    expected_routes: dict[int, list[int]],
    *,
    session_id: str,
    positions: tuple[int, ...],
    first_dispatcher: Any | None = None,
) -> dict[str, Any]:
    executor.open_session(session_id, maximum_context_override=max(positions) + 2)
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, position in enumerate(positions):
            dispatcher = (
                first_dispatcher
                if ordinal == 0 and first_dispatcher is not None
                else collective.dispatch_batch
            )
            record = executor.execute_decode_batch(
                session_ids=(session_id,),
                hidden_states=_batch_boundaries(
                    fixtures, batch=1, position=position
                ),
                cache_position_start=position,
                external_expert_dispatch=dispatcher,
            )
            output = np.asarray(record["boundary_output"])[0]
            fingerprint = _array_fingerprint(output)
            routes = [int(value) for value in record["selected_expert_ids"][0]]
            rows.append(
                {
                    "position": position,
                    "output_fingerprint": fingerprint,
                    "expected_fingerprint": expected_fingerprints[position],
                    "output_exact": fingerprint == expected_fingerprints[position],
                    "routes": routes,
                    "expected_routes": expected_routes[position],
                    "routes_exact": routes == expected_routes[position],
                    "all_selected_experts_executed_once": bool(
                        record["routing"]["all_selected_experts_executed_once"]
                    ),
                    "external_record": record["external_expert_collective"],
                }
            )
        state = executor.session_state_evidence(session_id)
    finally:
        executor.close_session(session_id)
    passed = all(
        row["output_exact"]
        and row["routes_exact"]
        and row["all_selected_experts_executed_once"]
        for row in rows
    ) and int(state["cache_sequence_length"]) == len(positions)
    return {"positions": rows, "state": state, "pass": passed}


def _execute_expected_failure(
    executor: PersistentKimiStageExecutor,
    fixtures: list[np.ndarray],
    dispatcher: _InjectedDispatcher,
    *,
    session_id: str,
) -> dict[str, Any]:
    executor.open_session(session_id, maximum_context_override=2)
    returned = False
    error: dict[str, Any] | None = None
    try:
        try:
            executor.execute_decode_batch(
                session_ids=(session_id,),
                hidden_states=_batch_boundaries(fixtures, batch=1, position=0),
                cache_position_start=0,
                external_expert_dispatch=dispatcher,
            )
            returned = True
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        state = executor.session_state_evidence(session_id)
    finally:
        executor.cancel_session(session_id)
    handle_state = getattr(dispatcher.handle, "state", None)
    return {
        "fault": dispatcher.fault,
        "target_worker": dispatcher.target_worker,
        "stage_returned_output": returned,
        "error": error,
        "handle_state": handle_state,
        "collective_poisoned_reason": dispatcher.collective._poisoned_reason,
        "session_state_after_failure": state,
        "cache_length_not_advanced": int(state["cache_sequence_length"]) == 0,
        "reduction_output_exposed": returned,
        "pass": (
            not returned
            and error is not None
            and int(state["cache_sequence_length"]) == 0
            and handle_state in {"FAILED", "CANCELLED"}
        ),
    }


def benchmark_sub_layer_recovery(
    checkpoint: Path,
    cuda_library: Path,
    oracle_trace: Path,
    passing_batch_receipt: Path,
    output_path: Path,
    *,
    device: int = 0,
    cycle_id: str = "H014-SUB-009",
) -> dict[str, Any]:
    """Exercise real expert recovery faults and atomically retain every scenario."""
    paths = {
        "checkpoint": checkpoint.resolve(),
        "cuda_library": cuda_library.resolve(),
        "oracle_trace": oracle_trace.resolve(),
        "passing_batch_receipt": passing_batch_receipt.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    reference = json.loads(
        paths["passing_batch_receipt"].read_text(encoding="utf-8")
    )
    if reference.get("status") != "PASS" or not reference.get("execution_pass"):
        raise ValueError("recovery requires a passing distributed batch receipt")
    expected_fingerprints, expected_routes = _expected_from_receipt(reference)
    request = _request(
        paths["checkpoint"],
        paths["cuda_library"],
        layer=89,
        device=device,
        cycle_id=cycle_id,
        maximum_context=3,
    )
    fixtures, _ = _stage_fixtures(
        paths["checkpoint"], paths["oracle_trace"], layer=89
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "RUNNING",
        "hypothesis": (
            "The real persistent expert collective fails closed on timeout, worker "
            "loss, partial, duplicate and stale responses; cancellation drains work, "
            "restored ownership retries exactly and fresh slots remain reusable."
        ),
        "configuration": {
            "layer": 89,
            "batch": 1,
            "workers": 4,
            "experts_per_worker": 224,
            "response_timeout_seconds": 0.1,
            "delay_fault_seconds": 0.3,
        },
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in paths.items()
        },
        "device_identity": _device_identity(device),
        "gpu_health_before": _health_snapshot(device),
        "scenarios": {},
        "groups": [],
        "progress": [],
    }

    def retain(phase: str, **evidence: Any) -> None:
        receipt["progress"].append({"phase": phase, **evidence})
        _atomic_json(output_path, receipt)

    executor: PersistentKimiStageExecutor | None = None
    proxies: list[_WorkerProxy] = []
    collective: _PersistentExpertCollective | None = None
    retain("preregistered")
    try:
        executor = PersistentKimiStageExecutor(
            request=request,
            checkpoint=paths["checkpoint"],
            cuda_library=paths["cuda_library"],
            device=device,
            owned_expert_ids=frozenset(),
        )
        lifecycle_before = _microwork_process_snapshot()
        executor_lifecycle_before = executor.lifecycle_snapshot()

        proxies, collective, ready = _start_group(
            paths["checkpoint"],
            paths["cuda_library"],
            device=device,
            response_timeout_seconds=0.1,
        )
        receipt["groups"].append({"group": 0, "ready": ready})
        retain("initial_group_ready")

        baseline = _execute_positions(
            executor,
            collective,
            fixtures,
            expected_fingerprints,
            expected_routes,
            session_id="h014-sub-009-baseline",
            positions=(0,),
        )
        receipt["scenarios"]["baseline"] = baseline
        retain("baseline_exact", passed=baseline["pass"])

        duplicate_dispatcher = _InjectedDispatcher(
            collective, proxies, "duplicate"
        )
        duplicate = _execute_positions(
            executor,
            collective,
            fixtures,
            expected_fingerprints,
            expected_routes,
            session_id="h014-sub-009-duplicate",
            positions=(0, 1),
            first_dispatcher=duplicate_dispatcher,
        )
        first_external = duplicate["positions"][0]["external_record"]
        duplicate["duplicate_frame_discarded"] = int(
            first_external["duplicate_frames_discarded"]
        ) == 1
        duplicate["fresh_generation_after_duplicate_exact"] = duplicate[
            "positions"
        ][1]["output_exact"]
        duplicate["pass"] = bool(duplicate["pass"]) and bool(
            duplicate["duplicate_frame_discarded"]
        )
        receipt["scenarios"]["duplicate_response"] = duplicate
        retain("duplicate_deduplicated", passed=duplicate["pass"])

        cancel_dispatcher = _InjectedDispatcher(collective, proxies, "cancel")
        cancellation = _execute_expected_failure(
            executor,
            fixtures,
            cancel_dispatcher,
            session_id="h014-sub-009-cancelled",
        )
        cancellation["cancel_record"] = cancel_dispatcher.record
        cancellation["pass"] = bool(cancellation["pass"]) and bool(
            cancel_dispatcher.record
            and cancel_dispatcher.record["reduction_performed"] is False
            and int(cancel_dispatcher.record["tasks_discarded"]) == 16
        )
        receipt["scenarios"]["cancellation"] = cancellation
        retain("cancelled_and_drained", passed=cancellation["pass"])

        fresh_slot = _execute_positions(
            executor,
            collective,
            fixtures,
            expected_fingerprints,
            expected_routes,
            session_id="h014-sub-009-fresh-slot",
            positions=(0,),
        )
        receipt["scenarios"]["fresh_slot_after_cancel"] = fresh_slot
        retain("fresh_slot_reused", passed=fresh_slot["pass"])

        prior_fault: str | None = None
        for fault_index, fault in enumerate(FAULTS):
            if prior_fault is not None:
                proxies, collective, ready = _start_group(
                    paths["checkpoint"],
                    paths["cuda_library"],
                    device=device,
                    response_timeout_seconds=0.1,
                )
                receipt["groups"].append(
                    {"group": fault_index, "restores": prior_fault, "ready": ready}
                )
                retry = _execute_positions(
                    executor,
                    collective,
                    fixtures,
                    expected_fingerprints,
                    expected_routes,
                    session_id=f"h014-sub-009-retry-{prior_fault}",
                    positions=(0,),
                )
                receipt["scenarios"][f"retry_after_{prior_fault}"] = retry
                retain(f"retry_after_{prior_fault}", passed=retry["pass"])
            if collective is None:
                raise RuntimeError("recovery collective is unavailable")
            dispatcher = _InjectedDispatcher(collective, proxies, fault)
            scenario = _execute_expected_failure(
                executor,
                fixtures,
                dispatcher,
                session_id=f"h014-sub-009-{fault}",
            )
            scenario["all_required_workers_present_before_fault"] = len(proxies) == 4
            scenario["reduction_blocked"] = not scenario["stage_returned_output"]
            scenario["pass"] = bool(scenario["pass"]) and bool(
                scenario["reduction_blocked"]
            )
            receipt["scenarios"][fault] = scenario
            retain(f"fault_{fault}_persisted", passed=scenario["pass"])
            shutdown = _close_group(proxies, device)
            receipt["groups"][-1]["shutdown_after_fault"] = shutdown
            proxies = []
            collective = None
            retain(
                f"fault_{fault}_group_closed",
                gpu_health=shutdown["gpu_health"]["status"],
            )
            prior_fault = fault

        proxies, collective, ready = _start_group(
            paths["checkpoint"],
            paths["cuda_library"],
            device=device,
            response_timeout_seconds=0.1,
        )
        receipt["groups"].append({"group": 4, "restores": "loss", "ready": ready})
        final_retry = _execute_positions(
            executor,
            collective,
            fixtures,
            expected_fingerprints,
            expected_routes,
            session_id="h014-sub-009-retry-loss",
            positions=(0,),
        )
        receipt["scenarios"]["retry_after_loss"] = final_retry
        final_worker_health = collective.health()
        receipt["final_worker_health"] = final_worker_health
        retain("retry_after_loss", passed=final_retry["pass"])

        lifecycle_after = _microwork_process_snapshot()
        executor_lifecycle_after = executor.lifecycle_snapshot()
        warm_lifecycle = _microwork_lifecycle_delta(
            lifecycle_before, lifecycle_after
        )
        warm_lifecycle.update(
            {
                "weight_loading": int(executor_lifecycle_after["weight_load_count"])
                - int(executor_lifecycle_before["weight_load_count"]),
                "model_materialization": int(
                    executor_lifecycle_after["model_materialization_count"]
                )
                - int(executor_lifecycle_before["model_materialization_count"]),
                "persistent_buffer_allocation": int(
                    executor_lifecycle_after["persistent_buffer_allocation_count"]
                )
                - int(
                    executor_lifecycle_before["persistent_buffer_allocation_count"]
                ),
            }
        )
        receipt["coordinator_lifecycle_delta"] = warm_lifecycle
        scenarios_pass = all(
            bool(row.get("pass")) for row in receipt["scenarios"].values()
        )
        health_pass = all(
            bool(row["cuda_error_state_ok"]) for row in final_worker_health
        )
        gates = {
            "baseline_exact": bool(baseline["pass"]),
            "timeout_failed_closed": bool(receipt["scenarios"]["timeout"]["pass"]),
            "worker_loss_failed_closed": bool(receipt["scenarios"]["loss"]["pass"]),
            "partial_failed_closed": bool(receipt["scenarios"]["partial"]["pass"]),
            "stale_failed_closed": bool(receipt["scenarios"]["stale"]["pass"]),
            "duplicate_deduplicated": bool(duplicate["pass"]),
            "cancellation_drained": bool(cancellation["pass"]),
            "fresh_slot_exact": bool(fresh_slot["pass"]),
            "all_retries_exact": all(
                bool(receipt["scenarios"][f"retry_after_{fault}"]["pass"])
                for fault in FAULTS
            ),
            "final_worker_cuda_health": health_pass,
            "all_scenarios_pass": scenarios_pass,
        }
        receipt["acceptance_gates"] = gates
        receipt["hypothesis_supported"] = all(gates.values())
        receipt["inspection"] = {
            "actual_bottleneck": (
                "Duplicate and cancellation recover in place, but any poisoned "
                "generation requires replacing the complete expert-ownership group; "
                "worker weight reload dominates recovery service restoration."
            ),
            "recovery_scope": (
                "Real worker processes and real CUDA expert computation; injected "
                "faults alter only response timing/envelopes or process availability."
            ),
        }
        receipt["decision"] = {
            "persistent_collective_recovery": (
                "RETAIN" if receipt["hypothesis_supported"] else "REJECT"
            ),
            "next_hypothesis": (
                "H014-SUB-010 will test whether retained real Kimi decode routes "
                "create persistent static-ownership hotspots large enough to justify "
                "expert reassignment or replication."
            ),
        }
        receipt["gpu_health_after"] = _health_snapshot(device)
        receipt["status"] = (
            "PASS"
            if receipt["hypothesis_supported"]
            and receipt["gpu_health_after"]["status"] == "MEASURED"
            else "FAIL"
        )
        retain("complete", status=receipt["status"])
        return receipt
    except BaseException as exc:
        receipt["status"] = "FAIL"
        receipt["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with suppress(Exception):
            receipt["gpu_health_after_failure"] = _health_snapshot(device)
        _atomic_json(output_path, receipt)
        return receipt
    finally:
        if proxies:
            receipt.setdefault("final_group_shutdown", _close_group(proxies, device))
        if executor is not None:
            executor.close()
        _atomic_json(output_path, receipt)
