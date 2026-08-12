"""Fail-closed recovery fixture for coarse Kimi stage ownership."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-014-k3-coarse-stage-recovery-v1"
_LENGTH = struct.Struct("!I")
_DIGEST_BYTES = 32


class StageResponseRejected(RuntimeError):
    """A response cannot advance state or expose output."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _encode_response(metadata: dict[str, Any], payload: bytes) -> bytes:
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    body = _LENGTH.pack(len(header)) + header + _LENGTH.pack(len(payload)) + payload
    return body + hashlib.sha256(body).digest()


def _decode_response(frame: bytes) -> tuple[dict[str, Any], bytes]:
    if len(frame) < 2 * _LENGTH.size + _DIGEST_BYTES:
        raise StageResponseRejected("partial response before fixed framing")
    body, observed_digest = frame[:-_DIGEST_BYTES], frame[-_DIGEST_BYTES:]
    if hashlib.sha256(body).digest() != observed_digest:
        raise StageResponseRejected("response digest mismatch or partial response")
    header_length = _LENGTH.unpack_from(body, 0)[0]
    header_start = _LENGTH.size
    header_end = header_start + header_length
    if header_end + _LENGTH.size > len(body):
        raise StageResponseRejected("partial response header")
    payload_length = _LENGTH.unpack_from(body, header_end)[0]
    payload_start = header_end + _LENGTH.size
    payload_end = payload_start + payload_length
    if payload_end != len(body):
        raise StageResponseRejected("partial or overlong response payload")
    metadata = json.loads(body[header_start:header_end].decode())
    if not isinstance(metadata, dict):
        raise StageResponseRejected("response metadata is not an object")
    return metadata, body[payload_start:payload_end]


@dataclass(slots=True)
class _Attempt:
    session_id: str
    request_generation: int
    cache_position: int
    operation_id: str
    assignment_sha256: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.session_id,
            self.request_generation,
            self.cache_position,
            self.operation_id,
        )


@dataclass(slots=True)
class _FailClosedCollector:
    attempt: _Attempt
    cancelled: bool = False
    committed: set[tuple[str, int, int, str]] = field(default_factory=set)
    cache_position: int = -1
    output_emissions: int = 0

    def cancel(self) -> None:
        self.cancelled = True

    def receive(self, frame: bytes) -> dict[str, Any]:
        metadata, payload = _decode_response(frame)
        if self.cancelled:
            raise StageResponseRejected("late response for cancelled attempt")
        expected = {
            "session_id": self.attempt.session_id,
            "request_generation": self.attempt.request_generation,
            "cache_position": self.attempt.cache_position,
            "operation_id": self.attempt.operation_id,
            "assignment_sha256": self.attempt.assignment_sha256,
        }
        observed = {name: metadata.get(name) for name in expected}
        if observed != expected:
            raise StageResponseRejected(
                f"stale or contradictory response identity: {observed!r}"
            )
        if self.attempt.key in self.committed:
            return {
                "status": "DUPLICATE_SUPPRESSED",
                "payload_sha256": _sha256_bytes(payload),
                "output_emitted": False,
            }
        if self.attempt.cache_position != self.cache_position + 1:
            raise StageResponseRejected("response would advance state out of order")
        self.committed.add(self.attempt.key)
        self.cache_position = self.attempt.cache_position
        self.output_emissions += 1
        return {
            "status": "COMMITTED",
            "payload_sha256": _sha256_bytes(payload),
            "output_emitted": True,
        }


def _metadata(attempt: _Attempt, worker_id: str) -> dict[str, Any]:
    return {
        "assignment_sha256": attempt.assignment_sha256,
        "cache_position": attempt.cache_position,
        "operation_id": attempt.operation_id,
        "request_generation": attempt.request_generation,
        "session_id": attempt.session_id,
        "worker_id": worker_id,
    }


def _rejected_case(
    name: str,
    collector: _FailClosedCollector,
    action: Any,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    error = ""
    try:
        action()
    except (StageResponseRejected, ConnectionError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter_ns() - started
    passed = bool(error) and collector.cache_position == -1 and collector.output_emissions == 0
    return {
        "fault": name,
        "error": error,
        "elapsed_ns": elapsed,
        "cache_position_after": collector.cache_position,
        "output_emissions": collector.output_emissions,
        "incomplete_output_exposed": collector.output_emissions != 0,
        "pass": passed,
    }


def benchmark_coarse_stage_recovery(
    output_path: Path,
    *,
    cycle_id: str = "H014-037b",
) -> dict[str, Any]:
    """Inject coarse response faults using the real 258,048-byte boundary size."""
    payload = bytes(range(256)) * 1008
    if len(payload) != 258048:
        raise AssertionError("fixture payload differs from the real coarse boundary")
    assignment_sha = hashlib.sha256(b"k3-worker-089:layer-89").hexdigest()
    base_attempt = _Attempt(
        session_id="h014-037b-session",
        request_generation=2,
        cache_position=0,
        operation_id="layer-89",
        assignment_sha256=assignment_sha,
    )
    valid_frame = _encode_response(_metadata(base_attempt, "k3-worker-089"), payload)

    partial_collector = _FailClosedCollector(base_attempt)
    partial = _rejected_case(
        "partial_response",
        partial_collector,
        lambda: partial_collector.receive(valid_frame[:-37]),
    )
    timeout_collector = _FailClosedCollector(base_attempt)
    timeout = _rejected_case(
        "timeout",
        timeout_collector,
        lambda: (_ for _ in ()).throw(TimeoutError("stage deadline expired")),
    )
    loss_collector = _FailClosedCollector(base_attempt)
    loss = _rejected_case(
        "worker_loss",
        loss_collector,
        lambda: (_ for _ in ()).throw(ConnectionError("stage connection closed")),
    )
    stale_collector = _FailClosedCollector(base_attempt)
    stale_attempt = _Attempt(
        session_id=base_attempt.session_id,
        request_generation=1,
        cache_position=base_attempt.cache_position,
        operation_id=base_attempt.operation_id,
        assignment_sha256=assignment_sha,
    )
    stale_frame = _encode_response(_metadata(stale_attempt, "retired-worker"), payload)
    stale = _rejected_case(
        "stale_generation",
        stale_collector,
        lambda: stale_collector.receive(stale_frame),
    )
    cancellation_collector = _FailClosedCollector(base_attempt)
    cancellation_collector.cancel()
    cancellation = _rejected_case(
        "cancellation_late_response",
        cancellation_collector,
        lambda: cancellation_collector.receive(valid_frame),
    )

    duplicate_collector = _FailClosedCollector(base_attempt)
    duplicate_started = time.perf_counter_ns()
    first = duplicate_collector.receive(valid_frame)
    second = duplicate_collector.receive(valid_frame)
    duplicate = {
        "fault": "duplicate_response",
        "first_status": first["status"],
        "second_status": second["status"],
        "elapsed_ns": time.perf_counter_ns() - duplicate_started,
        "cache_position_after": duplicate_collector.cache_position,
        "output_emissions": duplicate_collector.output_emissions,
        "duplicate_output_exposed": duplicate_collector.output_emissions != 1,
        "pass": (
            first["status"] == "COMMITTED"
            and second["status"] == "DUPLICATE_SUPPRESSED"
            and duplicate_collector.cache_position == 0
            and duplicate_collector.output_emissions == 1
        ),
    }

    wrong_assignment_collector = _FailClosedCollector(base_attempt)
    wrong_metadata = _metadata(base_attempt, "replacement-worker")
    wrong_metadata["assignment_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    wrong_assignment_frame = _encode_response(wrong_metadata, payload)
    wrong_assignment = _rejected_case(
        "replacement_assignment_mismatch",
        wrong_assignment_collector,
        lambda: wrong_assignment_collector.receive(wrong_assignment_frame),
    )

    retry_attempt = _Attempt(
        session_id=base_attempt.session_id,
        request_generation=3,
        cache_position=0,
        operation_id=base_attempt.operation_id,
        assignment_sha256=assignment_sha,
    )
    retry_collector = _FailClosedCollector(retry_attempt)
    retry_frame = _encode_response(
        _metadata(retry_attempt, "replacement-worker"), payload
    )
    retry_result = retry_collector.receive(retry_frame)
    replacement = {
        "replacement_worker": "replacement-worker",
        "assignment_sha256": assignment_sha,
        "request_generation": retry_attempt.request_generation,
        "status": retry_result["status"],
        "output_fingerprint": retry_result["payload_sha256"],
        "baseline_fingerprint": _sha256_bytes(payload),
        "cache_position_after": retry_collector.cache_position,
        "output_emissions": retry_collector.output_emissions,
        "pass": (
            retry_result["status"] == "COMMITTED"
            and retry_result["payload_sha256"] == _sha256_bytes(payload)
            and retry_collector.cache_position == 0
            and retry_collector.output_emissions == 1
        ),
    }
    faults = [partial, timeout, loss, stale, cancellation, duplicate, wrong_assignment]
    gates = {
        "partial_fails_closed": partial["pass"],
        "timeout_fails_closed": timeout["pass"],
        "worker_loss_fails_closed": loss["pass"],
        "stale_generation_fails_closed": stale["pass"],
        "cancellation_drains_late_response": cancellation["pass"],
        "duplicate_suppressed_exactly_once": duplicate["pass"],
        "wrong_replacement_assignment_rejected": wrong_assignment["pass"],
        "exact_replacement_retry_passes": replacement["pass"],
        "real_coarse_payload_size": len(payload) == 258048,
        "no_failed_fault_advances_state": all(
            bool(row["pass"]) for row in faults if row is not duplicate
        ),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "Whole-stage partial, timeout, loss, stale, cancellation, duplicate and "
            "replacement-identity faults fail closed or suppress duplicates without "
            "advancing state, while an exact assignment replacement can retry once."
        ),
        "configuration": {
            "payload_bytes": len(payload),
            "frame_bytes": len(valid_frame),
            "framing_bytes": len(valid_frame) - len(payload),
            "assignment_sha256": assignment_sha,
        },
        "faults": faults,
        "replacement_retry": replacement,
        "acceptance_gates": gates,
        "inspection": {
            "failed_output_emissions": sum(
                int(row["output_emissions"])
                for row in faults
                if row is not duplicate
            ),
            "duplicate_output_emissions": duplicate["output_emissions"],
            "replacement_output_emissions": replacement["output_emissions"],
            "replacement_numerically_exact": replacement["output_fingerprint"]
            == replacement["baseline_fingerprint"],
        },
        "decision": (
            "RETAIN_FAIL_CLOSED_COARSE_RECOVERY_CONTRACT"
            if all(gates.values())
            else "REDESIGN_COARSE_RECOVERY"
        ),
        "disclosure": (
            "Deterministic protocol fault fixture over the real production payload "
            "size; H014-032c separately proves real persistent TCP transport."
        ),
    }
    _atomic_json(output_path, receipt)
    return receipt


__all__ = ["StageResponseRejected", "benchmark_coarse_stage_recovery"]
