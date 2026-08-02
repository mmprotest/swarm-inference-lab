"""Readers for bridge NDJSON, stock mux telemetry, route traces, and usage history."""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

from swarm_inference.backends.colibri.schemas import BridgeEvent, RouteSelection

_IKU1 = b"IKU1"
_U32 = struct.Struct("<I")
_IKU1_HEADER = struct.Struct("<III")
_ENGINE_NAMES = ("glm_moe_dsa", "inkling", "olmoe", "kimi_k3")


def _fnv1a32(value: str) -> int:
    result = 2166136261
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 16777619) & 0xFFFFFFFF
    return result


class ColibriTelemetryReader:
    """Validate bridge ordering and decode the stock mux snapshots it wraps."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path).expanduser().resolve()

    def read(self, path: str | Path | None = None) -> list[BridgeEvent]:
        selected = Path(path).expanduser().resolve() if path is not None else self.path
        if selected is None:
            raise ValueError("telemetry path is required")
        events: list[BridgeEvent] = []
        last_sequence: dict[int, int] = {}
        with selected.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = BridgeEvent.model_validate_json(line)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid bridge event at {selected}:{line_number}: {exc}"
                    ) from exc
                expected = last_sequence.get(event.engine_pid, -1) + 1
                if event.sequence_number != expected:
                    raise ValueError(
                        f"non-contiguous bridge sequence for PID {event.engine_pid}: "
                        f"expected {expected}, found {event.sequence_number}"
                    )
                last_sequence[event.engine_pid] = event.sequence_number
                events.append(event)
        return events

    @staticmethod
    def summarize(events: Iterable[BridgeEvent]) -> dict[str, Any]:
        """Aggregate only observed bridge values; absent measurements remain absent."""

        rows = list(events)
        event_counts = Counter(event.event_type for event in rows)
        request_counts = Counter(event.request_id for event in rows if event.request_id is not None)
        storage_bytes = sum(
            int(event.payload.get("bytes", event.payload.get("byte_count", 0)))
            for event in rows
            if event.event_type == "storage_read"
        )
        cache_hits = sum(
            int(event.payload.get("count", 1))
            for event in rows
            if event.event_type == "expert_cache_hit"
        )
        cache_misses = sum(
            int(event.payload.get("count", 1))
            for event in rows
            if event.event_type == "expert_cache_miss"
        )
        cache_total = cache_hits + cache_misses
        phase_durations: dict[str, list[float]] = defaultdict(list)
        for event in rows:
            if event.event_type in {
                "prefill_completed",
                "decode_token_completed",
                "request_completed",
                "cpu_compute",
                "gpu_compute",
            }:
                duration = event.payload.get("duration_ns")
                if duration is not None:
                    phase_durations[event.event_type].append(float(duration) / 1_000_000)
        return {
            "event_count": len(rows),
            "event_type_counts": dict(sorted(event_counts.items())),
            "request_event_counts": dict(sorted(request_counts.items())),
            "engine_pids": sorted({event.engine_pid for event in rows}),
            "storage_read_bytes": storage_bytes,
            "expert_cache_hits": cache_hits,
            "expert_cache_misses": cache_misses,
            "expert_cache_hit_rate": cache_hits / cache_total if cache_total else None,
            "phase_duration_ms": {
                name: {
                    "count": len(values),
                    "total": sum(values),
                    "mean": sum(values) / len(values),
                }
                for name, values in sorted(phase_durations.items())
            },
        }

    @staticmethod
    def parse_stock_line(line: str) -> dict[str, Any] | None:
        fields = line.strip().split()
        if not fields:
            return None
        kind = fields[0]
        if kind == "HWINFO" and len(fields) >= 7:
            names = " ".join(fields[6:]).split("|")
            return {
                "kind": kind,
                "cores": int(fields[1]),
                "ram_total_gb": float(fields[2]),
                "ram_available_gb": float(fields[3]),
                "gpu_count": int(fields[4]),
                "vram_total_gb": float(fields[5]),
                "cpu": names[0].strip() if names else "",
                "gpu": names[1].strip() if len(names) > 1 else "",
            }
        if kind == "TIERS" and len(fields) >= 6:
            return {
                "kind": kind,
                "vram_experts": int(fields[1]),
                "ram_experts": int(fields[2]),
                "nvme_experts": int(fields[3]),
                "vram_gb": float(fields[4]),
                "ram_gb": float(fields[5]),
            }
        if kind == "EMAP" and len(fields) == 4:
            return {
                "kind": kind,
                "rows": int(fields[1]),
                "cols": int(fields[2]),
                "entries": ColibriTelemetryReader.decode_emap(
                    int(fields[1]), int(fields[2]), fields[3]
                ),
            }
        if kind == "HITS" and len(fields) == 4:
            return {
                "kind": kind,
                "rows": int(fields[1]),
                "cols": int(fields[2]),
                "bits": ColibriTelemetryReader.decode_hits(
                    int(fields[1]), int(fields[2]), fields[3]
                ),
            }
        if kind == "PROF" and len(fields) >= 10:
            return {
                "kind": kind,
                "wall_s": float(fields[1]),
                "prompt_tokens": int(fields[2]),
                "completion_tokens": int(fields[3]),
                "expert_disk_s": float(fields[4]),
                "expert_wait_s": float(fields[5]),
                "expert_matmul_s": float(fields[6]),
                "attention_s": float(fields[7]),
                "lm_head_s": float(fields[8]),
                "forwards": int(fields[9]),
            }
        if kind == "DONE" and len(fields) >= 8 and fields[2] == "STAT":
            return {
                "kind": kind,
                "request_id": fields[1],
                "completion_tokens": int(fields[3]),
                "tokens_per_second": float(fields[4]),
                "cache_hit_percent": float(fields[5]),
                "rss_gb": float(fields[6]),
                "prompt_tokens": int(fields[7]),
                "length_limited": bool(int(fields[8])) if len(fields) > 8 else None,
            }
        return {"kind": kind, "raw": line.rstrip("\r\n")}

    @staticmethod
    def decode_emap(rows: int, cols: int, encoded: str) -> list[dict[str, Any]]:
        if rows < 0 or cols < 0:
            raise ValueError("EMAP dimensions cannot be negative")
        try:
            data = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("EMAP is not hexadecimal") from exc
        if len(data) != rows * cols:
            raise ValueError("EMAP byte count does not match its dimensions")
        tier_names = {0: "nvme", 1: "ram", 2: "vram", 3: "reserved"}
        return [
            {
                "layer_id": index // cols,
                "expert_id": index % cols,
                "tier": tier_names[value >> 6],
                "heat": value & 0x3F,
            }
            for index, value in enumerate(data)
        ]

    @staticmethod
    def decode_hits(rows: int, cols: int, encoded: str) -> list[dict[str, Any]]:
        try:
            data = bytes.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("HITS bitmap is not hexadecimal") from exc
        required = math.ceil(rows * cols / 8)
        if len(data) != required:
            raise ValueError("HITS bitmap byte count does not match its dimensions")
        return [
            {
                "layer_id": index // cols,
                "expert_id": index % cols,
                "hit": bool(data[index // 8] & (1 << (index % 8))),
            }
            for index in range(rows * cols)
        ]


class ColibriRouteTraceReader:
    """Parse Colibri's real `<call> <row> <layer> <expert>:<gate>` trace."""

    def read(
        self,
        path: str | Path,
        *,
        phase_by_call: dict[int, str] | None = None,
        request_by_call: dict[int, str] | None = None,
        tier_by_expert: dict[tuple[int, int], str] | None = None,
    ) -> list[RouteSelection]:
        selected = Path(path).expanduser().resolve()
        rows: list[RouteSelection] = []
        with selected.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 4:
                    raise ValueError(f"invalid route trace line {selected}:{line_number}")
                try:
                    call_index, row_index, layer_id = map(int, fields[:3])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid route trace coordinates at line {line_number}"
                    ) from exc
                raw_phase = (phase_by_call or {}).get(call_index, "unknown")
                if raw_phase not in {"prefill", "decode", "unknown"}:
                    raise ValueError(f"invalid phase mapping for call {call_index}: {raw_phase}")
                phase = cast(Literal["prefill", "decode", "unknown"], raw_phase)
                for selection in fields[3:]:
                    expert_text, separator, weight_text = selection.partition(":")
                    if not separator:
                        raise ValueError(f"missing routing weight at line {line_number}")
                    try:
                        expert_id = int(expert_text)
                        weight = float(weight_text)
                    except ValueError as exc:
                        raise ValueError(f"invalid route selection at line {line_number}") from exc
                    if not math.isfinite(weight):
                        raise ValueError(f"non-finite routing weight at line {line_number}")
                    rows.append(
                        RouteSelection(
                            call_index=call_index,
                            row_index=row_index,
                            layer_id=layer_id,
                            expert_id=expert_id,
                            routing_weight=weight,
                            phase=phase,
                            token_index=row_index if phase == "prefill" else None,
                            request_id=(request_by_call or {}).get(call_index),
                            execution_tier=(tier_by_expert or {}).get((layer_id, expert_id)),
                        )
                    )
        return rows

    @staticmethod
    def summarize(selections: Iterable[RouteSelection]) -> dict[str, Any]:
        materialized = list(selections)
        activation = Counter((row.phase, row.layer_id, row.expert_id) for row in materialized)
        phase_totals = Counter(row.phase for row in materialized)
        groups: dict[tuple[int, int, int, str], list[int]] = defaultdict(list)
        for row in materialized:
            groups[(row.call_index, row.row_index, row.layer_id, row.phase)].append(row.expert_id)
        coactivation: Counter[tuple[str, int, int, int]] = Counter()
        for (_, _, layer_id, phase), experts in groups.items():
            unique = sorted(set(experts))
            for left_index, left in enumerate(unique):
                for right in unique[left_index + 1 :]:
                    coactivation[(phase, layer_id, left, right)] += 1

        by_token: dict[tuple[int, int, str], dict[int, list[int]]] = defaultdict(dict)
        for (call, row_index, layer, phase), experts in groups.items():
            by_token[(call, row_index, phase)][layer] = experts
        transitions: Counter[tuple[str, int, int, int, int]] = Counter()
        for (_, _, phase), layers in by_token.items():
            ordered = sorted(layers)
            for left_layer, right_layer in pairwise(ordered):
                for left in set(layers[left_layer]):
                    for right in set(layers[right_layer]):
                        transitions[(phase, left_layer, right_layer, left, right)] += 1

        last_seen: dict[tuple[str, int, int], int] = {}
        reuse_count = 0
        for row in sorted(
            materialized, key=lambda item: (item.call_index, item.row_index, item.layer_id)
        ):
            key = (row.phase, row.layer_id, row.expert_id)
            if key in last_seen:
                reuse_count += 1
            last_seen[key] = row.call_index

        return {
            "selection_count": len(materialized),
            "unique_experts": len({(row.layer_id, row.expert_id) for row in materialized}),
            "phase_selection_counts": dict(phase_totals),
            "reuse_selection_count": reuse_count,
            "activation": [
                {
                    "phase": phase,
                    "layer_id": layer,
                    "expert_id": expert,
                    "activation_count": count,
                    "activation_probability": count / phase_totals[phase]
                    if phase_totals[phase]
                    else None,
                }
                for (phase, layer, expert), count in sorted(activation.items())
            ],
            "coactivation": [
                {
                    "phase": phase,
                    "layer_id": layer,
                    "expert_a": left,
                    "expert_b": right,
                    "count": count,
                }
                for (phase, layer, left, right), count in sorted(coactivation.items())
            ],
            "transitions": [
                {
                    "phase": phase,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "source_expert": source_expert,
                    "target_expert": target_expert,
                    "count": count,
                }
                for (
                    phase,
                    source_layer,
                    target_layer,
                    source_expert,
                    target_expert,
                ), count in sorted(transitions.items())
            ],
        }


class ColibriUsageHistoryReader:
    """Read current sparse text histories and legacy Inkling IKU1 histories."""

    def read(
        self,
        path: str | Path,
        *,
        expected_layers: int | None = None,
        expected_experts: int | None = None,
        expected_engine: str | None = None,
        allow_cross_engine: bool = False,
    ) -> dict[str, Any]:
        selected = Path(path).expanduser().resolve()
        with selected.open("rb") as handle:
            prefix = handle.read(4)
        if prefix == _IKU1:
            return self._read_iku1(
                selected,
                expected_layers=expected_layers,
                expected_experts=expected_experts,
                expected_engine=expected_engine,
                allow_cross_engine=allow_cross_engine,
            )
        return self._read_text(
            selected,
            expected_layers=expected_layers,
            expected_experts=expected_experts,
            expected_engine=expected_engine,
            allow_cross_engine=allow_cross_engine,
        )

    @staticmethod
    def _validate_identity(
        *,
        layers: int | None,
        experts: int | None,
        engine_id: int | None,
        expected_layers: int | None,
        expected_experts: int | None,
        expected_engine: str | None,
        allow_cross_engine: bool,
    ) -> None:
        if layers is not None and expected_layers is not None and layers != expected_layers:
            raise ValueError(f"usage history layer count mismatch: {layers} != {expected_layers}")
        if experts is not None and expected_experts is not None and experts != expected_experts:
            raise ValueError(
                f"usage history expert count mismatch: {experts} != {expected_experts}"
            )
        if engine_id is not None and expected_engine is not None:
            expected_id = _fnv1a32(expected_engine)
            if engine_id != expected_id and not allow_cross_engine:
                writer = next((name for name in _ENGINE_NAMES if _fnv1a32(name) == engine_id), None)
                raise ValueError(
                    f"usage history was written by {writer or hex(engine_id)}, not {expected_engine}"
                )

    def _read_text(
        self,
        path: Path,
        *,
        expected_layers: int | None,
        expected_experts: int | None,
        expected_engine: str | None,
        allow_cross_engine: bool,
    ) -> dict[str, Any]:
        layers = experts = engine_id = version = None
        rows: list[dict[str, int]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"invalid usage history line {path}:{line_number}")
            try:
                layer, second, third = map(int, fields)
            except ValueError as exc:
                raise ValueError(f"non-numeric usage history line {path}:{line_number}") from exc
            if layer == -1:
                layers, experts = second, third
            elif layer == -2:
                version, engine_id = second, third
                if version > 1:
                    raise ValueError(f"unsupported usage history format version {version}")
            elif layer >= 0:
                if second < 0 or third < 0:
                    raise ValueError(
                        f"negative expert/count in usage history at line {line_number}"
                    )
                rows.append({"layer_id": layer, "expert_id": second, "activation_count": third})
        self._validate_identity(
            layers=layers,
            experts=experts,
            engine_id=engine_id,
            expected_layers=expected_layers,
            expected_experts=expected_experts,
            expected_engine=expected_engine,
            allow_cross_engine=allow_cross_engine,
        )
        return {
            "format": "sparse_text",
            "format_version": version,
            "engine_id": engine_id,
            "layers": layers,
            "experts": experts,
            "total_activations": sum(row["activation_count"] for row in rows),
            "records": rows,
        }

    def _read_iku1(
        self,
        path: Path,
        *,
        expected_layers: int | None,
        expected_experts: int | None,
        expected_engine: str | None,
        allow_cross_engine: bool,
    ) -> dict[str, Any]:
        payload = path.read_bytes()
        if len(payload) < _IKU1_HEADER.size:
            raise ValueError("truncated IKU1 usage history")
        magic, layers, experts = _IKU1_HEADER.unpack_from(payload)
        if magic.to_bytes(4, "little") != _IKU1:
            raise ValueError("invalid IKU1 magic")
        expected_bytes = _IKU1_HEADER.size + layers * experts * _U32.size
        if len(payload) != expected_bytes:
            raise ValueError("IKU1 usage history length does not match dimensions")
        self._validate_identity(
            layers=layers,
            experts=experts,
            engine_id=_fnv1a32("inkling"),
            expected_layers=expected_layers,
            expected_experts=expected_experts,
            expected_engine=expected_engine,
            allow_cross_engine=allow_cross_engine,
        )
        rows = []
        offset = _IKU1_HEADER.size
        for layer in range(layers):
            for expert in range(experts):
                count = _U32.unpack_from(payload, offset)[0]
                offset += _U32.size
                if count:
                    rows.append({"layer_id": layer, "expert_id": expert, "activation_count": count})
        return {
            "format": "IKU1",
            "format_version": 0,
            "engine_id": _fnv1a32("inkling"),
            "layers": layers,
            "experts": experts,
            "total_activations": sum(row["activation_count"] for row in rows),
            "records": rows,
        }
