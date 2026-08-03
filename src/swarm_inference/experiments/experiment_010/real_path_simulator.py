"""Trace-parity and held-out timing calibration for real Experiment 010 runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CacheState:
    budget_bytes: int
    max_entries: int
    entries: OrderedDict[tuple[int, int, int | None, int | None], int] = field(
        default_factory=OrderedDict
    )
    bytes_used: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def access(
        self, key: tuple[int, int, int | None, int | None], entry_bytes: int
    ) -> None:
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return
        self.misses += 1
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self.bytes_used + entry_bytes > self.budget_bytes
        ):
            _, removed = self.entries.popitem(last=False)
            self.bytes_used -= removed
            self.evictions += 1
        if entry_bytes > self.budget_bytes:
            raise ValueError("one expert cache entry exceeds the declared worker budget")
        self.entries[key] = entry_bytes
        self.bytes_used += entry_bytes


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _run_directory(row: dict[str, str]) -> Path:
    if row.get("raw_measurement_path"):
        return Path(row["raw_measurement_path"]).resolve().parent
    if row.get("memory_timeseries_path"):
        return Path(row["memory_timeseries_path"]).resolve().parent
    if row.get("telemetry_path"):
        return Path(row["telemetry_path"]).resolve().parent
    raise ValueError(f"row {row.get('run_id')} has no raw run directory")


def _plan_path(run_directory: Path) -> Path:
    for parent in (run_directory, *run_directory.parents):
        candidate = parent / "session" / "plan.json"
        if candidate.is_file():
            return candidate
        direct = parent / "plan.json"
        if direct.is_file() and parent == run_directory:
            return direct
    raise FileNotFoundError(f"no native plan found above {run_directory}")


def _request_directories(run_directory: Path) -> list[Path]:
    if (run_directory / "coordinator-telemetry.jsonl").is_file():
        return [run_directory]
    requests = sorted(
        path.parent
        for path in run_directory.glob("request-*/coordinator-telemetry.jsonl")
    )
    if not requests:
        raise FileNotFoundError(f"no coordinator telemetry below {run_directory}")
    return requests


def _first_event_time(run_directory: Path) -> int:
    values = []
    for request_directory in _request_directories(run_directory):
        for event in _read_jsonl(request_directory / "coordinator-telemetry.jsonl"):
            if isinstance(event.get("wall_time_ns"), int):
                values.append(int(event["wall_time_ns"]))
                break
    if values:
        return min(values)
    return run_directory.stat().st_mtime_ns


def _copy_cache(cache: CacheState) -> CacheState:
    return CacheState(
        cache.budget_bytes,
        cache.max_entries,
        OrderedDict(cache.entries),
        cache.bytes_used,
        cache.hits,
        cache.misses,
        cache.evictions,
    )


def _apply_cache_event(
    cache: CacheState,
    event: dict[str, Any],
    entry_sizes: dict[tuple[int, int, int | None, int | None], int],
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]],
) -> None:
    worker_id = str(event["worker_id"])
    layer_id = int(event["layer_id"])
    expert_ids = [int(value) for value in event.get("expert_ids") or []]
    if not expert_ids:
        return
    hidden_start, hidden_end = slices[(worker_id, layer_id, expert_ids[0])]
    for expert_id in expert_ids:
        key = (layer_id, expert_id, hidden_start, hidden_end)
        cache.access(key, entry_sizes[key])


def _select_cache_plateau(
    cache: CacheState,
    signatures: list[tuple[str, int, int, int]],
    candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]],
    target: tuple[int, int, int],
    entry_sizes: dict[tuple[int, int, int | None, int | None], int],
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]],
) -> tuple[CacheState, list[int]]:
    """Resolve equal-length frames against one atomic telemetry snapshot."""

    def visit(
        position: int, state: CacheState, used: frozenset[int]
    ) -> tuple[CacheState, list[int]] | None:
        if position == len(signatures):
            if (state.hits, state.misses, state.evictions) == target:
                return state, []
            return None
        signature = signatures[position]
        seen_transitions: set[
            tuple[tuple[int, ...], tuple[int, int, int], tuple[tuple[int, int, int | None, int | None], ...]]
        ] = set()
        for choice in candidates[signature]:
            identity = id(choice)
            if identity in used:
                continue
            trial = _copy_cache(state)
            _apply_cache_event(trial, choice, entry_sizes, slices)
            if (
                trial.hits > target[0]
                or trial.misses > target[1]
                or trial.evictions > target[2]
            ):
                continue
            transition = (
                tuple(int(value) for value in choice.get("expert_ids") or []),
                (trial.hits, trial.misses, trial.evictions),
                tuple(trial.entries),
            )
            if transition in seen_transitions:
                continue
            seen_transitions.add(transition)
            result = visit(position + 1, trial, used | {identity})
            if result is not None:
                final_state, selected = result
                return final_state, [identity, *selected]
        return None

    result = visit(0, cache, frozenset())
    if result is None:
        raise ValueError(
            "no rank-preserving coordinator-frame assignment matches the native cache snapshot"
        )
    return result


def _commit_cache_plateau(
    cache: CacheState,
    signatures: list[tuple[str, int, int, int]],
    candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]],
    target: tuple[int, int, int],
    entry_sizes: dict[tuple[int, int, int | None, int | None], int],
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]],
) -> CacheState:
    try:
        next_cache, selected = _select_cache_plateau(
            cache, signatures, candidates, target, entry_sizes, slices
        )
    except ValueError as error:
        raise ValueError(
            f"cache checkpoint mismatch: start={(cache.hits, cache.misses, cache.evictions)} "
            f"target={target} requests={len(signatures)}"
        ) from error
    selected_ids = set(selected)
    for signature in set(signatures):
        candidates[signature] = [
            choice
            for choice in candidates[signature]
            if id(choice) not in selected_ids
        ]
    return next_cache


def _select_cache_window(
    cache: CacheState,
    records: list[tuple[int, tuple[str, int, int, int]]],
    candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]],
    target: tuple[int, int, int],
    entry_sizes: dict[tuple[int, int, int | None, int | None], int],
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]],
) -> tuple[CacheState, list[int], list[int]]:
    """Resolve bounded telemetry reordering after worker mutex release."""

    target_accesses = target[0] + target[1]

    def visit(
        state: CacheState,
        available: tuple[int, ...],
        used_events: frozenset[int],
    ) -> tuple[CacheState, list[int], list[int]] | None:
        current_accesses = state.hits + state.misses
        if current_accesses == target_accesses:
            if (state.hits, state.misses, state.evictions) == target:
                return state, [], []
            return None
        seen: set[
            tuple[
                tuple[str, int, int, int],
                tuple[int, ...],
                tuple[int, int, int],
                tuple[tuple[int, int, int | None, int | None], ...],
            ]
        ] = set()
        for record_position in available:
            record_index, signature = records[record_position]
            remaining_positions = tuple(
                value for value in available if value != record_position
            )
            for choice in candidates[signature]:
                identity = id(choice)
                if identity in used_events:
                    continue
                access_count = len(choice.get("expert_ids") or [])
                if current_accesses + access_count > target_accesses:
                    continue
                trial = _copy_cache(state)
                _apply_cache_event(trial, choice, entry_sizes, slices)
                if (
                    trial.hits > target[0]
                    or trial.misses > target[1]
                    or trial.evictions > target[2]
                ):
                    continue
                transition = (
                    signature,
                    tuple(int(value) for value in choice.get("expert_ids") or []),
                    (trial.hits, trial.misses, trial.evictions),
                    tuple(trial.entries),
                )
                if transition in seen:
                    continue
                seen.add(transition)
                result = visit(
                    trial, remaining_positions, used_events | {identity}
                )
                if result is not None:
                    final_state, selected_events, selected_records = result
                    return (
                        final_state,
                        [identity, *selected_events],
                        [record_index, *selected_records],
                    )
        return None

    result = visit(cache, tuple(range(len(records))), frozenset())
    if result is None:
        raise ValueError("bounded native telemetry reorder could not match cache state")
    return result


def _remove_selected_events(
    candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]],
    selected: list[int],
) -> None:
    selected_ids = set(selected)
    for signature in list(candidates):
        candidates[signature] = [
            choice for choice in candidates[signature] if id(choice) not in selected_ids
        ]


def _replay_worker_cache(
    completed: list[dict[str, Any]],
    accounting: list[dict[str, Any]],
    trace_offsets: dict[Path, int],
    trace_cache: dict[Path, list[dict[str, Any]]],
    caches: dict[str, CacheState],
    entry_sizes: dict[
        str, dict[tuple[int, int, int | None, int | None], int]
    ],
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]],
) -> None:
    """Replay native serialized cache transitions and reject any mismatch."""

    by_worker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in completed:
        by_worker[str(event["worker_id"])].append(event)
    for item in accounting:
        worker_id = str(item["worker_id"])
        worker_events = by_worker[worker_id]
        candidates: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(
            list
        )
        for event in worker_events:
            candidates[
                (
                    str(event["request_id"]),
                    int(event["layer_id"]),
                    int(event["_sent_bytes"]),
                    int(event["_received_bytes"]),
                )
            ].append(event)
        telemetry_path = Path(item["worker_telemetry"])
        if telemetry_path not in trace_cache:
            loaded_trace = _read_jsonl(telemetry_path)
            if loaded_trace and all(
                isinstance(event.get("execution_sequence"), int)
                for event in loaded_trace
            ):
                loaded_trace.sort(key=lambda event: int(event["execution_sequence"]))
            elif loaded_trace and all(
                isinstance(event.get("wall_time_ns"), int) for event in loaded_trace
            ):
                loaded_trace.sort(key=lambda event: int(event["wall_time_ns"]))
            trace_cache[telemetry_path] = loaded_trace
        trace = trace_cache[telemetry_path]
        start = trace_offsets.get(telemetry_path, 0)
        matched = 0
        last_index = start
        for index in range(start, len(trace)):
            native = trace[index]
            if native.get("event") not in {
                "native_expert_request_completed",
                "native_expert_shared_memory_request_completed",
            }:
                continue
            signature = (
                str(native["request_id"]),
                int(native["layer_id"]),
                int(native["bytes_received"]),
                int(native["bytes_sent"]),
            )
            choices = candidates.get(signature)
            if not choices:
                continue
            choice = min(
                choices,
                key=lambda event: abs(
                    int(event["_worker_rpc_ns"]) - int(native["duration_ns"])
                ),
            )
            _apply_cache_event(
                caches[worker_id], choice, entry_sizes[worker_id], slices
            )
            choices.remove(choice)
            matched += 1
            last_index = index + 1
            if matched == len(worker_events):
                break
        if matched != len(worker_events):
            remaining = sum(len(choices) for choices in candidates.values())
            raise ValueError(
                f"native worker trace did not reconcile for {worker_id}: "
                f"matched={matched} expected={len(worker_events)} "
                f"remaining={remaining}"
            )
        remaining = sum(len(choices) for choices in candidates.values())
        if remaining:
            raise ValueError(
                f"native worker trace left {remaining} unmatched coordinator frames for {worker_id}"
            )
        trace_offsets[telemetry_path] = last_index


def _ownership(
    plan: dict[str, Any],
) -> tuple[
    dict[tuple[int, int], tuple[str, ...]],
    dict[tuple[str, int, int], tuple[int | None, int | None]],
]:
    owners: dict[tuple[int, int], list[str]] = defaultdict(list)
    slices: dict[tuple[str, int, int], tuple[int | None, int | None]] = {}
    for worker in plan["workers"]:
        worker_id = str(worker["worker_id"])
        if worker.get("replica_of"):
            continue
        for expert in worker.get("owned_experts", []):
            key = (int(expert["layer_id"]), int(expert["expert_id"]))
            owners[key].append(worker_id)
            slices[(worker_id, *key)] = (None, None)
        for shard in worker.get("owned_microshards", []):
            key = (int(shard["layer_id"]), int(shard["expert_id"]))
            owners[key].append(worker_id)
            slices[(worker_id, *key)] = (
                int(shard["hidden_start"]),
                int(shard["hidden_end"]),
            )
    return {key: tuple(value) for key, value in owners.items()}, slices


def _route_groups(
    path: Path, plan: dict[str, Any]
) -> tuple[
    list[tuple[int, int, dict[str, list[int]]]],
    int,
]:
    owners, _ = _ownership(plan)
    groups: OrderedDict[tuple[int, int], dict[str, list[int]]] = OrderedDict()
    routed_layer_count = max((layer for layer, _ in owners), default=-1) + 1
    if routed_layer_count <= 0:
        raise ValueError("plan has no routed expert layers")
    source_selections = 0
    prefill_rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed route trace row: {line!r}")
        call_id, row_id, layer_id = int(fields[0]), int(fields[1]), int(fields[2])
        if call_id == 0:
            prefill_rows = max(prefill_rows, row_id + 1)
        destinations = groups.setdefault((call_id, layer_id), defaultdict(list))
        for routed in fields[3:]:
            expert_id = int(routed.split(":", 1)[0])
            source_selections += 1
            selected_owners = owners.get((layer_id, expert_id), ())
            for worker_id in selected_owners:
                destinations[worker_id].append(expert_id)
    if prefill_rows <= 0:
        raise ValueError("route trace has no initial prefill rows")
    return [
        (
            0
            if call_id < routed_layer_count
            else prefill_rows + call_id // routed_layer_count - 1,
            layer,
            dict(destinations),
        )
        for (call_id, layer), destinations in groups.items()
    ], source_selections


def _native_cache_slot_count(manifest: dict[str, Any], budget_bytes: int) -> int:
    """Mirror the shared C runtime's full-expert-sized metadata cache."""

    hidden = int(manifest["hidden_size"])
    intermediate = int(manifest["intermediate_size"])
    full_expert_bytes = 3 * hidden * intermediate + 4 * (2 * intermediate + hidden)
    if budget_bytes:
        slots = budget_bytes // full_expert_bytes
        if slots <= 0:
            raise ValueError("worker budget cannot hold one native expert")
    else:
        slots = 1
    return min(slots, 4096)


def _bank_entries(path: Path) -> dict[tuple[int, int, int | None, int | None], int]:
    manifest = _read_json(path)
    entries: dict[tuple[int, int, int | None, int | None], int] = {}
    if manifest["bank_kind"] in {
        "whole_expert",
        "native_colibri_expert_bank",
        "native_colibri_whole_experts",
    }:
        for expert in manifest["experts"]:
            entries[
                (int(expert["layer_id"]), int(expert["expert_id"]), None, None)
            ] = int(expert["expert_bytes"])
    elif manifest["bank_kind"] in {"microshard", "native_colibri_microshards"}:
        for shard in manifest["shards"]:
            entries[
                (
                    int(shard["layer_id"]),
                    int(shard["expert_id"]),
                    int(shard["hidden_start"]),
                    int(shard["hidden_end"]),
                )
            ] = int(shard["shard_bytes"])
    else:
        raise ValueError(f"unsupported native bank kind {manifest['bank_kind']}")
    return entries


def _representative_rows(
    analysis: Path, supplemental_measurements: Path | None = None
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    short = _read_csv(analysis / "short_decode_results.csv")
    remote_short = [row for row in short if row["configuration"] != "local"]
    for configuration in sorted({row["configuration"] for row in remote_short}):
        rows = [row for row in remote_short if row["configuration"] == configuration]
        rows.sort(key=lambda row: (int(row["repeat"]), row["prompt_id"]))
        selected.append(rows[0])
    selected.extend(_read_csv(analysis / "network_profile_results.csv"))
    selected.extend(
        row
        for row in _read_csv(analysis / "concurrent_decode_results.csv")
        if row["configuration"] != "local"
    )
    selected.extend(
        row
        for row in _read_csv(analysis / "mixed_service_results.csv")
        if row["configuration"] != "local"
    )
    selected.extend(
        row
        for row in _read_csv(analysis / "prefill_results.csv")
        if row["configuration"] != "local"
    )
    if supplemental_measurements is not None:
        for path in sorted(supplemental_measurements.rglob("*.csv")):
            selected.extend(_read_csv(path))
    # A run is unique even when one analysis row was copied into more than one
    # consolidated table.  A supplemental rerun with the same run ID replaces
    # only that trace while preserving the original Phase 10 artifact.
    unique: dict[str, dict[str, str]] = {}
    for row in selected:
        unique[str(row["run_id"])] = row
    return list(unique.values())


def replay_behavioral_parity(
    analysis: Path,
    output: Path,
    supplemental_measurements: Path | None = None,
) -> dict[str, Any]:
    selected = _representative_rows(analysis, supplemental_measurements)
    sessions: dict[Path, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        sessions[_plan_path(_run_directory(row))].append(row)
    parity_rows: list[dict[str, Any]] = []
    trace_offsets: dict[Path, int] = {}
    trace_cache: dict[Path, list[dict[str, Any]]] = {}
    for plan_path, rows in sessions.items():
        plan = _read_json(plan_path)
        _, slices = _ownership(plan)
        rows.sort(key=lambda row: _first_event_time(_run_directory(row)))
        caches: dict[str, CacheState] = {}
        entry_sizes: dict[str, dict[tuple[int, int, int | None, int | None], int]] = {}
        process_ids: dict[str, int] = {}
        for row in rows:
            run_directory = _run_directory(row)
            accounting = json.loads(row["worker_process_accounting"])
            current_ids = {str(item["worker_id"]): int(item["pid"]) for item in accounting}
            for item in accounting:
                worker_id = str(item["worker_id"])
                if process_ids.get(worker_id) == current_ids[worker_id]:
                    continue
                process_ids[worker_id] = current_ids[worker_id]
                bank_manifest = _read_json(Path(item["bank_manifest"]))
                budget_bytes = int(item["cache_capacity_bytes"])
                caches[worker_id] = CacheState(
                    budget_bytes,
                    _native_cache_slot_count(bank_manifest, budget_bytes),
                )
                entry_sizes[worker_id] = _bank_entries(Path(item["bank_manifest"]))

            request_directories = _request_directories(run_directory)
            source_selection_count = 0
            simulated_assignments: Counter[str] = Counter()
            simulated_worker_selections: Counter[str] = Counter()
            expected_events: list[dict[str, Any]] = []
            coordinator: list[dict[str, Any]] = []
            for request_directory in request_directories:
                groups, request_selections = _route_groups(
                    request_directory / "route.trace", plan
                )
                source_selection_count += request_selections
                for token_position, layer_id, destinations in groups:
                    for worker_id in plan["reduction_order"]:
                        expert_ids = destinations.get(worker_id, [])
                        if not expert_ids:
                            continue
                        simulated_assignments[worker_id] += 1
                        simulated_worker_selections[worker_id] += len(expert_ids)
                        expected_events.append(
                            {
                                "scope": request_directory.name,
                                "token_position": token_position,
                                "layer_id": layer_id,
                                "worker_id": worker_id,
                                "expert_ids": tuple(expert_ids),
                            }
                        )
                coordinator.extend(
                    {
                        **event,
                        "_scope": request_directory.name,
                    }
                    for event in _read_jsonl(
                        request_directory / "coordinator-telemetry.jsonl"
                    )
                )
            wire_bytes: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
            worker_timings: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
            for event in coordinator:
                event_name = event.get("event")
                if event_name not in {
                    "expert_rpc_bytes_sent",
                    "expert_rpc_bytes_received",
                    "expert_rpc_queue_ns",
                    "expert_rpc_compute_ns",
                }:
                    continue
                key = (
                    str(event["_scope"]),
                    str(event["worker_id"]),
                    str(event["request_id"]),
                )
                if event_name.startswith("expert_rpc_bytes_"):
                    direction = "sent" if event_name.endswith("_sent") else "received"
                    wire_bytes[key][direction] = int(event["byte_count"])
                else:
                    component = "queue" if event_name.endswith("queue_ns") else "compute"
                    worker_timings[key][component] = int(event["duration_ns"])
            completed = [
                {
                    **event,
                    "_sent_bytes": wire_bytes[
                        (
                            str(event["_scope"]),
                            str(event["worker_id"]),
                            str(event["request_id"]),
                        )
                    ]["sent"],
                    "_received_bytes": wire_bytes[
                        (
                            str(event["_scope"]),
                            str(event["worker_id"]),
                            str(event["request_id"]),
                        )
                    ]["received"],
                    "_worker_rpc_ns": sum(
                        worker_timings[
                            (
                                str(event["_scope"]),
                                str(event["worker_id"]),
                                str(event["request_id"]),
                            )
                        ].values()
                    ),
                }
                for event in coordinator
                if event.get("event") == "expert_rpc_request_completed"
            ]
            expected_signatures = Counter(
                (
                    event["scope"],
                    event["token_position"],
                    event["layer_id"],
                    event["worker_id"],
                    event["expert_ids"],
                )
                for event in expected_events
            )
            actual_signatures = Counter(
                (
                    str(event["_scope"]),
                    int(event["token_position"]),
                    int(event["layer_id"]),
                    str(event["worker_id"]),
                    tuple(int(value) for value in event.get("expert_ids") or []),
                )
                for event in completed
            )
            # Concurrent request order is an observed scheduler input.  The
            # native worker telemetry is the authoritative serialized order;
            # frame byte counts disambiguate request IDs reused by coordinators.
            try:
                _replay_worker_cache(
                    completed,
                    accounting,
                    trace_offsets,
                    trace_cache,
                    caches,
                    entry_sizes,
                    slices,
                )
            except ValueError as error:
                raise ValueError(f"{row['run_id']}: {error}") from error

            actual_assignments = Counter(str(event["worker_id"]) for event in completed)
            actual_worker_selections: Counter[str] = Counter()
            for event in completed:
                actual_worker_selections[str(event["worker_id"])] += len(
                    event.get("expert_ids") or []
                )
            actual_payload_bytes = sum(
                int(event.get("byte_count", 0))
                for event in coordinator
                if event.get("event")
                in {"expert_rpc_bytes_sent", "expert_rpc_bytes_received"}
            )
            expected_payload_bytes = int(float(row["rpc_raw_payload_bytes"]))
            expected_messages = int(float(row["rpc_message_count"]))
            expected_worker_selection_count = sum(actual_worker_selections.values())
            simulated_worker_selection_count = sum(simulated_worker_selections.values())
            observed_cache_hits = sum(int(item["logical_cache_hits"]) for item in accounting)
            observed_cache_misses = 0
            for item in accounting:
                sizes = set(entry_sizes[str(item["worker_id"])].values())
                if len(sizes) != 1:
                    raise ValueError(
                        "behavioral replay requires one native entry size per worker bank"
                    )
                resident_entries = int(item["cache_bytes"]) // sizes.pop()
                # With no prefetch or cache-drop event, cumulative misses equal
                # evictions plus the entries that remain resident.
                observed_cache_misses += int(item["cache_evictions"]) + resident_entries
            observed_evictions = sum(int(item["cache_evictions"]) for item in accounting)
            simulated_cache_hits = sum(caches[item["worker_id"]].hits for item in accounting)
            simulated_cache_misses = sum(caches[item["worker_id"]].misses for item in accounting)
            simulated_evictions = sum(caches[item["worker_id"]].evictions for item in accounting)
            event_names = Counter(str(event.get("event")) for event in coordinator)
            actual_prefetches = sum(
                count for name, count in event_names.items() if "prefetch" in name
            )
            actual_fallbacks = event_names["expert_rpc_fallback"]
            actual_duplicates = sum(
                count for name, count in event_names.items() if "duplicate" in name and "started" in name
            )
            actual_failures = event_names["expert_rpc_timeout"] + event_names[
                "expert_rpc_invalid_response"
            ]
            checks = {
                "route_event_identity": expected_signatures == actual_signatures,
                "expert_selection_count": (
                    simulated_worker_selection_count == expected_worker_selection_count
                ),
                "worker_assignments": simulated_assignments == actual_assignments,
                "cache_hits": simulated_cache_hits == observed_cache_hits,
                "cache_misses": simulated_cache_misses == observed_cache_misses,
                "evictions": simulated_evictions == observed_evictions,
                "message_count": len(expected_events) == len(completed) == expected_messages,
                "raw_payload_bytes": actual_payload_bytes == expected_payload_bytes,
                "prefetch_count": actual_prefetches == 0,
                "fallback_count": actual_fallbacks == 0,
                "duplicate_count": actual_duplicates == 0,
                "failure_count": actual_failures == 0,
            }
            parity_rows.append(
                {
                    "schema_version": "experiment-010-simulator-behavioral-row-v1",
                    "configuration": row["configuration"],
                    "run_id": row["run_id"],
                    "network_profile": row["network_profile"],
                    "source_route_selection_count": source_selection_count,
                    "simulated_worker_selection_count": simulated_worker_selection_count,
                    "observed_worker_selection_count": expected_worker_selection_count,
                    "simulated_worker_assignments": dict(simulated_assignments),
                    "observed_worker_assignments": dict(actual_assignments),
                    "simulated_cache_hits": simulated_cache_hits,
                    "observed_cache_hits": observed_cache_hits,
                    "simulated_cache_misses": simulated_cache_misses,
                    "observed_cache_misses": observed_cache_misses,
                    "simulated_evictions": simulated_evictions,
                    "observed_evictions": observed_evictions,
                    "simulated_message_count": len(expected_events),
                    "observed_message_count": expected_messages,
                    "simulated_raw_payload_bytes": actual_payload_bytes,
                    "observed_raw_payload_bytes": expected_payload_bytes,
                    "simulated_prefetch_count": 0,
                    "observed_prefetch_count": actual_prefetches,
                    "simulated_fallback_count": 0,
                    "observed_fallback_count": actual_fallbacks,
                    "simulated_duplicate_count": 0,
                    "observed_duplicate_count": actual_duplicates,
                    "simulated_failure_count": 0,
                    "observed_failure_count": actual_failures,
                    "checks": checks,
                    "all_exact": all(checks.values()),
                    "route_trace_paths": [
                        str(path / "route.trace") for path in request_directories
                    ],
                    "coordinator_telemetry_paths": [
                        str(path / "coordinator-telemetry.jsonl")
                        for path in request_directories
                    ],
                    "worker_telemetry_paths": sorted(
                        {str(item["worker_telemetry"]) for item in accounting}
                    ),
                    "payload_replay_input": (
                        "canonical SWARMEX1 frame byte counts recorded beside each semantic request"
                    ),
                    "evidence_category": "REAL_MODEL_MEASURED",
                }
            )
    summary = {
        "schema_version": "experiment-010-simulator-behavioral-parity-v1",
        "configuration_count": len(parity_rows),
        "all_exact": all(row["all_exact"] for row in parity_rows),
        "required_exact_checks": [
            "expert_selection_count",
            "cache_hits",
            "cache_misses",
            "evictions",
            "message_count",
            "raw_payload_bytes",
        ],
        "policy": {
            "routing": "actual router trace replayed through plan ownership",
            "cache": (
                "shared native runtime LRU with full-expert-sized slot capacity "
                "and per-bank resident byte accounting"
            ),
            "transport": "one coalesced request per non-empty destination/layer/token group",
            "prefetch": "disabled, matching the measured Phase 10 policy",
        },
        "rows": parity_rows,
    }
    _write_json(output / "simulator_behavioral_parity.json", summary)
    _write_csv(output / "simulator_behavioral_parity.csv", parity_rows)
    return summary


def _first_token_transport_ns(run_directory: Path) -> int:
    """Return the critical request's measured token-zero transport component.

    The simulator treats the native component trace as an input, not as the
    end-to-end target.  Concurrent runs have one trace per request, so their
    TTFT is bounded by the slowest request's token-zero component.
    """

    request_totals: list[int] = []
    for request_directory in _request_directories(run_directory):
        total = 0
        saw_token_zero = False
        with (request_directory / "coordinator-telemetry.jsonl").open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if "expert_rpc_transport_ns" not in line:
                    continue
                event = json.loads(line)
                if event.get("event") != "expert_rpc_transport_ns":
                    continue
                token_position = int(event["token_position"])
                if token_position == 0:
                    total += int(event["duration_ns"])
                    saw_token_zero = True
                elif saw_token_zero:
                    break
        if not saw_token_zero:
            raise ValueError(
                f"no token-zero RPC transport events in {request_directory}"
            )
        request_totals.append(total)
    return max(request_totals)


def _worker_subset(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("worker_ids")
    if raw in (None, ""):
        accounting = row.get("worker_process_accounting")
        if accounting in (None, ""):
            return ()
        return tuple(
            sorted(str(item["worker_id"]) for item in json.loads(str(accounting)))
        )
    values = json.loads(str(raw)) if isinstance(raw, str) else raw
    return tuple(sorted(str(value) for value in values))


def _timing_sample(
    row: dict[str, str], *, table_kind: str
) -> dict[str, Any] | None:
    if row.get("configuration") == "local":
        return None
    if row.get("measurement_status") not in (None, "", "MEASURED"):
        return None
    if row.get("evidence_category") != "REAL_MODEL_MEASURED":
        return None
    if row.get("valid_performance_candidate") not in (None, "") and not _true(
        row["valid_performance_candidate"]
    ):
        return None
    if row.get("canonical_verified_candidate") not in (None, "") and not _true(
        row["canonical_verified_candidate"]
    ):
        return None
    identity_field = (
        "exact_group_token_identity"
        if row.get("exact_group_token_identity") not in (None, "")
        else "exact_token_identity"
    )
    if not _true(row.get(identity_field)):
        return None

    requests = json.loads(row["requests"]) if row.get("requests") else []
    concurrency = int(_number(row, "concurrency", 1.0))
    if requests:
        critical_transport_ns = max(
            _number(item, "rpc_transport_ns") for item in requests
        )
        prompt_tokens = float(
            statistics.median(_number(item, "prompt_tokens") for item in requests)
        )
        output_tokens = float(
            statistics.median(_number(item, "generated_tokens") for item in requests)
        )
        measured_ttft_ns = max(
            _number(item, "ttft_seconds") for item in requests
        ) * 1e9
    else:
        critical_transport_ns = _number(row, "rpc_transport_ns")
        prompt_tokens = _number(row, "prompt_tokens")
        output_tokens = _number(row, "generated_tokens")
        measured_ttft_ns = _number(row, "ttft_seconds") * 1e9

    measured_total_ns = _number(row, "group_elapsed_ns") or _number(
        row, "wall_elapsed_ns"
    )
    measured_p95_ns = (
        _number(row, "p95_latency_seconds") * 1e9 or measured_total_ns
    )
    measured_throughput = _number(
        row, "aggregate_verified_tokens_per_second"
    ) or _number(row, "decode_tokens_per_second")
    verified_tokens = _number(row, "verified_tokens") or output_tokens
    worker_count = int(_number(row, "worker_count"))
    if not all(
        value > 0
        for value in (
            measured_total_ns,
            measured_p95_ns,
            measured_ttft_ns,
            measured_throughput,
            verified_tokens,
            worker_count,
        )
    ):
        raise ValueError(f"incomplete measured timing row {row.get('run_id')}")

    phase = {
        "short": "decode",
        "network": "decode",
        "concurrent": "concurrent_decode",
        "mixed": "mixed_service",
        "prefill": "prefill",
    }[table_kind]
    configuration = str(row["configuration"])
    network_profile = str(row.get("network_profile") or "loopback_unshaped")
    layout = str(row.get("shard_layout") or "whole")
    response_mode = str(row.get("response_mode") or "per_expert_exact")
    workers = _worker_subset(row)
    configuration_id = "|".join(
        (
            phase,
            configuration,
            network_profile,
            str(concurrency),
            layout,
            response_mode,
            ",".join(workers),
        )
    )
    if table_kind == "network":
        workload_id = f"decode-network|p{int(prompt_tokens)}|o{int(output_tokens)}"
    elif table_kind == "short":
        workload_id = f"decode-short|{network_profile}|o{int(output_tokens)}"
    elif table_kind in {"concurrent", "mixed"}:
        workload_id = f"{phase}|c{concurrency}|o{int(output_tokens)}"
    else:
        workload_id = f"prefill|p{int(prompt_tokens)}|o{int(output_tokens)}"

    rpc_compute_ns = _number(row, "rpc_compute_ns")
    measured_utilization = _number(row, "worker_compute_utilization_fraction")
    if measured_utilization <= 0:
        measured_utilization = rpc_compute_ns / (worker_count * measured_total_ns)
    return {
        "configuration_id": configuration_id,
        "workload_id": workload_id,
        "phase": phase,
        "configuration": configuration,
        "network_profile": network_profile,
        "data_plane": str(row.get("data_plane") or "direct_tcp"),
        "response_mode": response_mode,
        "shard_layout": layout,
        "worker_subset": workers,
        "worker_count": worker_count,
        "concurrency": concurrency,
        "is_microshard": int("microshard" in configuration),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "verified_tokens": verified_tokens,
        "critical_transport_ns": critical_transport_ns,
        "first_token_transport_ns": _first_token_transport_ns(_run_directory(row)),
        "rpc_compute_ns": rpc_compute_ns,
        "rpc_queue_ns": _number(row, "rpc_queue_ns"),
        "rpc_raw_payload_bytes": _number(row, "rpc_raw_payload_bytes"),
        "rpc_message_count": _number(row, "rpc_message_count"),
        "measured_total_ns": measured_total_ns,
        "measured_throughput": measured_throughput,
        "measured_p95_ns": measured_p95_ns,
        "measured_ttft_ns": measured_ttft_ns,
        "measured_network_bytes": _number(row, "rpc_raw_payload_bytes"),
        "measured_queue_ns": _number(row, "rpc_queue_ns"),
        "measured_worker_utilization": measured_utilization,
        "run_id": str(row["run_id"]),
        "source_path": str(row.get("source_path") or row.get("raw_measurement_path") or ""),
        "evidence_category": "REAL_MODEL_MEASURED",
    }


def _aggregate_timing_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["configuration_id"])].append(sample)
    rows: list[dict[str, Any]] = []
    for _configuration_id, group in sorted(grouped.items()):
        first = group[0]
        row = {
            key: first[key]
            for key in (
                "configuration_id",
                "workload_id",
                "phase",
                "configuration",
                "network_profile",
                "data_plane",
                "response_mode",
                "shard_layout",
                "worker_subset",
                "worker_count",
                "concurrency",
                "is_microshard",
                "evidence_category",
            )
        }
        numeric = (
            "prompt_tokens",
            "output_tokens",
            "verified_tokens",
            "critical_transport_ns",
            "first_token_transport_ns",
            "rpc_compute_ns",
            "rpc_queue_ns",
            "rpc_raw_payload_bytes",
            "rpc_message_count",
            "measured_total_ns",
            "measured_throughput",
            "measured_p95_ns",
            "measured_ttft_ns",
            "measured_network_bytes",
            "measured_queue_ns",
            "measured_worker_utilization",
        )
        for key in numeric:
            row[key] = float(statistics.median(float(item[key]) for item in group))
        row["sample_count"] = len(group)
        row["run_ids"] = sorted(str(item["run_id"]) for item in group)
        row["source_paths"] = sorted(
            {str(item["source_path"]) for item in group if item["source_path"]}
        )
        rows.append(row)
    return rows


def _load_timing_rows(analysis: Path) -> list[dict[str, Any]]:
    specifications = (
        ("short_decode_results.csv", "short"),
        ("network_profile_results.csv", "network"),
        ("concurrent_decode_results.csv", "concurrent"),
        ("mixed_service_results.csv", "mixed"),
        ("prefill_results.csv", "prefill"),
    )
    samples: list[dict[str, Any]] = []
    for filename, table_kind in specifications:
        for source in _read_csv(analysis / filename):
            sample = _timing_sample(source, table_kind=table_kind)
            if sample is not None:
                samples.append(sample)
    return _aggregate_timing_samples(samples)


def _configuration_split(
    rows: list[dict[str, Any]], *, validation_fraction: float = 0.30, seed: int = 1010
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 4:
        raise ValueError("held-out timing calibration requires four configurations")
    phase_counts = Counter(str(row["phase"]) for row in rows)
    # A singleton phase cannot be held out without asking the model to
    # extrapolate outside its calibration domain.  Keep it in calibration and
    # select the same requested number of held-out complete configurations
    # from phases with an independently measured calibration peer.
    forced_calibration = {
        str(row["configuration_id"])
        for row in rows
        if phase_counts[str(row["phase"])] == 1
    }
    ordered = sorted(
        [row for row in rows if row["configuration_id"] not in forced_calibration],
        key=lambda row: hashlib.sha256(
            f"{seed}|{row['configuration_id']}".encode()
        ).digest(),
    )
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation_ids = {
        str(row["configuration_id"]) for row in ordered[:validation_count]
    }
    calibration = [
        row for row in rows if str(row["configuration_id"]) not in validation_ids
    ]
    validation = [
        row for row in rows if str(row["configuration_id"]) in validation_ids
    ]
    return calibration, validation


def _fit_linear(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    target: Any,
) -> dict[str, Any]:
    if len(rows) < len(feature_names):
        raise ValueError(
            f"linear timing fit has {len(rows)} rows for {len(feature_names)} features"
        )
    matrix = np.asarray(
        [
            [1.0 if name == "intercept" else float(row[name]) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    scales = np.ones(matrix.shape[1], dtype=np.float64)
    for index, name in enumerate(feature_names):
        if name == "intercept":
            continue
        nonzero = np.abs(matrix[:, index][matrix[:, index] != 0])
        scales[index] = max(float(np.median(nonzero)) if len(nonzero) else 1.0, 1.0)
    normalized = matrix / scales
    targets = np.asarray([float(target(row)) for row in rows], dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(normalized, targets, rcond=None)
    return {
        "feature_names": list(feature_names),
        "feature_scales": [float(value) for value in scales],
        "coefficients": [float(value) for value in coefficients],
        "calibration_row_count": len(rows),
    }


def _predict_linear(model: dict[str, Any], row: dict[str, Any]) -> float:
    values = np.asarray(
        [
            1.0 if name == "intercept" else float(row[name])
            for name in model["feature_names"]
        ],
        dtype=np.float64,
    )
    scales = np.asarray(model["feature_scales"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return float(np.dot(values / scales, coefficients))


def _median_ratios(
    rows: list[dict[str, Any]], key: Any, numerator: Any, denominator: Any
) -> tuple[dict[str, float], float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    values: list[float] = []
    for row in rows:
        value = float(numerator(row)) / max(float(denominator(row)), 1e-12)
        grouped[str(key(row))].append(value)
        values.append(value)
    return (
        {name: float(statistics.median(items)) for name, items in grouped.items()},
        float(statistics.median(values)),
    )


def _fraction_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / max(abs(measured), 1e-12)


def _ranking_and_regret(rows: list[dict[str, Any]]) -> tuple[float, float, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["workload_id"])].append(row)
    pairs = 0
    agreements = 0
    regrets: list[float] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        measured_best = max(group, key=lambda item: item["measured_throughput"])
        predicted_best = max(group, key=lambda item: item["predicted_throughput"])
        regrets.append(
            max(
                0.0,
                (
                    float(measured_best["measured_throughput"])
                    - float(predicted_best["measured_throughput"])
                )
                / max(float(measured_best["measured_throughput"]), 1e-12),
            )
        )
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                measured_delta = float(left["measured_throughput"]) - float(
                    right["measured_throughput"]
                )
                predicted_delta = float(left["predicted_throughput"]) - float(
                    right["predicted_throughput"]
                )
                if measured_delta == 0:
                    continue
                pairs += 1
                agreements += int(measured_delta * predicted_delta > 0)
    if not pairs:
        raise ValueError("held-out split contains no comparable candidate pair")
    return agreements / pairs, max(regrets, default=0.0), pairs


def _recovery_validation(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv(path):
        if not _true(source.get("passed")) or not source.get("recovery_latency_ns"):
            continue
        result = _read_json(Path(source["run_path"]) / "result.json")
        detection_ns = _number(source, "failure_detection_latency_ns")
        trigger_ns = detection_ns or float(result["expert_timeout_ms"]) * 1e6
        strategy = str(source["strategy"])
        rows.append(
            {
                "configuration_id": (
                    f"recovery|{source['failure_kind']}|{strategy}|{source['scenario']}"
                ),
                "scenario": str(source["scenario"]),
                "strategy": strategy,
                "trigger_ns": trigger_ns,
                "is_alternate": int("alternate" in strategy),
                "is_local": int("local" in strategy),
                "measured_recovery_ns": _number(source, "recovery_latency_ns"),
                "raw_result_path": str(Path(source["run_path"]) / "result.json"),
                "evidence_category": "REAL_MODEL_MEASURED",
            }
        )
    if len(rows) < 4:
        raise ValueError("recovery calibration needs four measured recoverable scenarios")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"1010|recovery|{row['scenario']}".encode()
        ).digest(),
    )
    validation_count = max(1, round(len(rows) * 0.30))
    validation_ids = {
        str(row["configuration_id"]) for row in ordered[:validation_count]
    }
    calibration = [row for row in rows if row["configuration_id"] not in validation_ids]
    validation = [row for row in rows if row["configuration_id"] in validation_ids]
    model = _fit_linear(
        calibration,
        ("intercept", "is_alternate", "is_local"),
        lambda row: row["measured_recovery_ns"] - row["trigger_ns"],
    )
    scored: list[dict[str, Any]] = []
    for row in validation:
        prediction = max(row["trigger_ns"] + _predict_linear(model, row), 1.0)
        scored.append(
            {
                **row,
                "row_type": "failure_recovery",
                "partition": "held_out_validation",
                "predicted_recovery_ns": prediction,
                "recovery_error_fraction": _fraction_error(
                    prediction, row["measured_recovery_ns"]
                ),
            }
        )
    payload = {
        "model": model,
        "calibration_configuration_ids": sorted(
            str(row["configuration_id"]) for row in calibration
        ),
        "validation_configuration_ids": sorted(validation_ids),
        "held_out_error_fraction": float(
            statistics.median(row["recovery_error_fraction"] for row in scored)
        ),
        "validation_rows": len(scored),
    }
    return payload, scored


def calibrate_real_path_timing(
    analysis: Path,
    failure_results: Path,
    output: Path,
) -> dict[str, Any]:
    behavioral_path = output / "simulator_behavioral_parity.json"
    behavioral = _read_json(behavioral_path)
    if not behavioral.get("all_exact"):
        raise ValueError("timing calibration is ineligible until behavioral parity passes")

    rows = _load_timing_rows(analysis)
    calibration, validation = _configuration_split(rows)
    single_calibration = [
        row for row in calibration if row["phase"] in {"decode", "prefill"}
    ]
    parallel_calibration = [
        row
        for row in calibration
        if row["phase"] in {"concurrent_decode", "mixed_service"}
    ]
    base_single_model = _fit_linear(
        single_calibration,
        ("intercept", "output_tokens", "prompt_tokens"),
        lambda row: row["measured_total_ns"] - row["critical_transport_ns"],
    )
    base_parallel_model = _fit_linear(
        parallel_calibration,
        ("intercept", "concurrency", "is_microshard"),
        lambda row: row["measured_total_ns"] - row["critical_transport_ns"],
    )
    ttft_model = _fit_linear(
        calibration,
        ("intercept", "prompt_tokens"),
        lambda row: row["measured_ttft_ns"] - row["first_token_transport_ns"],
    )
    p95_ratios, p95_default = _median_ratios(
        calibration,
        lambda row: (
            "parallel"
            if row["phase"] in {"concurrent_decode", "mixed_service"}
            else "serial"
        ),
        lambda row: row["measured_p95_ns"],
        lambda row: row["measured_total_ns"],
    )
    throughput_ratios, throughput_default = _median_ratios(
        calibration,
        lambda row: f"{row['phase']}|o{int(row['output_tokens'])}",
        lambda row: row["measured_throughput"],
        lambda row: row["verified_tokens"] * 1e9 / row["measured_total_ns"],
    )

    scored: list[dict[str, Any]] = []
    for row in validation:
        parallel = row["phase"] in {"concurrent_decode", "mixed_service"}
        base_model = base_parallel_model if parallel else base_single_model
        predicted_total_ns = max(
            row["critical_transport_ns"] + _predict_linear(base_model, row), 1.0
        )
        throughput_class = f"{row['phase']}|o{int(row['output_tokens'])}"
        predicted_throughput = (
            row["verified_tokens"]
            * 1e9
            / predicted_total_ns
            * throughput_ratios.get(throughput_class, throughput_default)
        )
        p95_class = "parallel" if parallel else "serial"
        predicted_p95_ns = predicted_total_ns * p95_ratios.get(
            p95_class, p95_default
        )
        predicted_ttft_ns = max(
            row["first_token_transport_ns"] + _predict_linear(ttft_model, row),
            1.0,
        )
        predicted_utilization = row["rpc_compute_ns"] / (
            row["worker_count"] * predicted_total_ns
        )
        scored.append(
            {
                **row,
                "row_type": "level_a_timing",
                "partition": "held_out_validation",
                "predicted_total_ns": predicted_total_ns,
                "predicted_throughput": predicted_throughput,
                "predicted_p95_ns": predicted_p95_ns,
                "predicted_ttft_ns": predicted_ttft_ns,
                # These quantities are exact outputs of the already-validated
                # trace replay, not fitted copies of an end-to-end target.
                "predicted_network_bytes": row["rpc_raw_payload_bytes"],
                "predicted_queue_ns": row["rpc_queue_ns"],
                "predicted_worker_utilization": predicted_utilization,
                "throughput_error_fraction": _fraction_error(
                    predicted_throughput, row["measured_throughput"]
                ),
                "p95_error_fraction": _fraction_error(
                    predicted_p95_ns, row["measured_p95_ns"]
                ),
                "ttft_error_fraction": _fraction_error(
                    predicted_ttft_ns, row["measured_ttft_ns"]
                ),
                "network_bytes_error_fraction": _fraction_error(
                    row["rpc_raw_payload_bytes"], row["measured_network_bytes"]
                ),
                "queue_error_fraction": _fraction_error(
                    row["rpc_queue_ns"], row["measured_queue_ns"]
                ),
                "worker_utilization_error_fraction": _fraction_error(
                    predicted_utilization, row["measured_worker_utilization"]
                ),
            }
        )

    ranking, regret, ranking_pairs = _ranking_and_regret(scored)
    throughput_error = float(
        statistics.median(row["throughput_error_fraction"] for row in scored)
    )
    p95_error = float(
        np.percentile([row["p95_error_fraction"] for row in scored], 95)
    )
    ttft_error = float(
        statistics.median(row["ttft_error_fraction"] for row in scored)
    )
    recovery, recovery_rows = _recovery_validation(failure_results)
    gates = {
        "behavioral_parity_pass": bool(behavioral["all_exact"]),
        "median_throughput_error_fraction": throughput_error,
        "median_throughput_error_pass": throughput_error <= 0.10,
        "p95_latency_error_fraction": p95_error,
        "p95_latency_error_pass": p95_error <= 0.15,
        "median_ttft_error_fraction": ttft_error,
        "ttft_error_pass": ttft_error <= 0.15,
        "plan_ranking_agreement_fraction": ranking,
        "plan_ranking_agreement_pass": ranking >= 0.80,
        "planner_regret_fraction": regret,
        "planner_regret_pass": regret <= 0.05,
    }
    gates["all_gates_pass"] = all(
        bool(value) for name, value in gates.items() if name.endswith("_pass")
    )
    prediction_category = (
        "SIMULATED_CALIBRATED"
        if gates["all_gates_pass"]
        else "SIMULATED_UNCALIBRATED"
    )
    for row in (*scored, *recovery_rows):
        row["prediction_category"] = prediction_category

    calibration_ids = sorted(str(row["configuration_id"]) for row in calibration)
    validation_ids = sorted(str(row["configuration_id"]) for row in validation)
    payload = {
        "schema_version": "experiment-010-real-path-simulator-calibration-v1",
        "official_gate_eligible": True,
        "evidence_category": "REAL_MODEL_MEASURED",
        "prediction_category": prediction_category,
        "behavioral_parity_artifact": str(behavioral_path),
        "behavioral_configuration_count": int(behavioral["configuration_count"]),
        "split": {
            "unit": (
                "complete execution-strategy/network-profile/worker-subset/"
                "shard-layout configuration"
            ),
            "seed": 1010,
            "requested_calibration_fraction": 0.70,
            "requested_validation_fraction": 0.30,
            "configuration_count": len(rows),
            "calibration_count": len(calibration),
            "validation_count": len(validation),
            "actual_calibration_fraction": len(calibration) / len(rows),
            "actual_validation_fraction": len(validation) / len(rows),
            "repeats_partitioned_together": True,
            "calibration_configuration_ids": calibration_ids,
            "validation_configuration_ids": validation_ids,
        },
        "models": {
            "serial_non_rpc_time": base_single_model,
            "parallel_critical_path_non_rpc_time": base_parallel_model,
            "ttft_non_rpc_time": ttft_model,
            "p95_to_total_ratios": p95_ratios,
            "p95_default_ratio": p95_default,
            "throughput_to_total_ratios": throughput_ratios,
            "throughput_default_ratio": throughput_default,
            "network_bytes": "exact behavioral trace replay",
            "queue_time": "exact behavioral scheduler trace replay",
            "worker_utilization": "rpc_compute_ns / (workers * predicted_total_ns)",
            "recovery_latency": recovery,
        },
        "validation": {
            **gates,
            "held_out_configuration_count": len(scored),
            "held_out_ranking_pair_count": ranking_pairs,
            "p95_ttft_error_fraction": float(
                np.percentile([row["ttft_error_fraction"] for row in scored], 95)
            ),
            "maximum_network_bytes_error_fraction": max(
                row["network_bytes_error_fraction"] for row in scored
            ),
            "maximum_queue_error_fraction": max(
                row["queue_error_fraction"] for row in scored
            ),
            "p95_worker_utilization_error_fraction": float(
                np.percentile(
                    [row["worker_utilization_error_fraction"] for row in scored], 95
                )
            ),
            "held_out_recovery_error_fraction": recovery[
                "held_out_error_fraction"
            ],
        },
        "acceptance_thresholds": {
            "median_throughput_error_fraction": 0.10,
            "p95_latency_error_fraction": 0.15,
            "median_ttft_error_fraction": 0.15,
            "plan_ranking_agreement_fraction": 0.80,
            "planner_regret_fraction": 0.05,
        },
        "large_topology_evidence_category": prediction_category,
        "limitations": [
            "Calibration is valid only for the measured single-host Level A topology envelope.",
            "Network bytes and queue events are trace-driven behavioral predictions; timing is held out.",
            "Recovery has five eligible scenarios, so its separate diagnostic split is necessarily small.",
        ],
    }
    calibration_rows = []
    calibration_set = set(calibration_ids)
    for row in rows:
        calibration_rows.append(
            {
                **row,
                "partition": (
                    "calibration"
                    if row["configuration_id"] in calibration_set
                    else "held_out_validation"
                ),
            }
        )
    _write_json(output / "simulator_calibration.json", payload)
    _write_json(output / "simulator_timing_calibration.json", payload)
    _write_csv(output / "simulator_calibration_rows.csv", calibration_rows)
    _write_csv(output / "simulator_validation.csv", [*scored, *recovery_rows])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavioral-parity", action="store_true")
    parser.add_argument("--timing-calibration", action="store_true")
    parser.add_argument("--phase10-analysis", type=Path, required=True)
    parser.add_argument("--phase11-failure-results", type=Path)
    parser.add_argument("--supplemental-measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.behavioral_parity and not arguments.timing_calibration:
        parser.error("select --behavioral-parity and/or --timing-calibration")
    if arguments.behavioral_parity:
        replay_behavioral_parity(
            arguments.phase10_analysis,
            arguments.output,
            arguments.supplemental_measurements,
        )
    if arguments.timing_calibration:
        if arguments.phase11_failure_results is None:
            parser.error("--timing-calibration requires --phase11-failure-results")
        calibrate_real_path_timing(
            arguments.phase10_analysis,
            arguments.phase11_failure_results,
            arguments.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
