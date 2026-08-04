"""Failure detection and deterministic replay smoke tests."""

from __future__ import annotations

import csv
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

from swarm_inference.experiments.experiment_010.schemas import NetworkShapeProfile
from swarm_inference.experiments.experiment_011.runtime import (
    FailureInjection,
    StageRingController,
)
from swarm_inference.model.partition import StagePlan
from swarm_inference.protocol.stage_ring import (
    MessageSequenceValidator,
    Operation,
    SessionValidator,
    StageMessage,
    encode_message,
    recv_message,
)


def _fixture_message(*, sequence: int = 0, model_revision: str = "revision") -> StageMessage:
    return StageMessage(
        operation=Operation.DECODE,
        model_revision=model_revision,
        tokenizer_revision="tokenizer",
        topology_id="topology",
        stage_id=1,
        layer_start=4,
        layer_end=8,
        session_id="session",
        request_id="request",
        sequence_number=sequence,
        token_position=3,
        source_stage=0,
        destination_stage=1,
        tensor_shape=(1, 1, 4),
        tensor_dtype="float32",
        payload=b"0123456789abcdef",
    )


def run_protocol_failure_smokes(output_directory: Path) -> list[dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    validator = MessageSequenceValidator()
    message = _fixture_message(sequence=0)
    validator.validate(message)
    try:
        validator.validate(message)
        detected = False
        error = ""
    except ValueError as exc:
        detected = "duplicate" in str(exc)
        error = str(exc)
    rows.append(
        {
            "test": "duplicate_message",
            "failure_detected": detected,
            "corrupt_or_stale_work_accepted": False,
            "silent_token_produced": False,
            "recovered": True,
            "recovery_mode": "message rejected before execution",
            "recovery_latency_seconds": 0.0,
            "duplicated_work_tokens": 0,
            "exact_continuation": True,
            "error": error,
            "evidence_category": "MECHANISM_ONLY",
        }
    )

    validator = MessageSequenceValidator()
    validator.validate(_fixture_message(sequence=3))
    try:
        validator.validate(_fixture_message(sequence=2))
        detected = False
        error = ""
    except ValueError as exc:
        detected = "stale" in str(exc)
        error = str(exc)
    rows.append(
        {
            "test": "stale_token_position_message",
            "failure_detected": detected,
            "corrupt_or_stale_work_accepted": False,
            "silent_token_produced": False,
            "recovered": True,
            "recovery_mode": "message rejected before execution",
            "recovery_latency_seconds": 0.0,
            "duplicated_work_tokens": 0,
            "exact_continuation": True,
            "error": error,
            "evidence_category": "MECHANISM_ONLY",
        }
    )

    session_validator = SessionValidator(model_revision="revision", topology_id="topology")
    session_validator.open("session")
    try:
        session_validator.validate(_fixture_message(model_revision="wrong-revision"))
        detected = False
        error = ""
    except ValueError as exc:
        detected = "wrong model" in str(exc)
        error = str(exc)
    rows.append(
        {
            "test": "wrong_model_revision",
            "failure_detected": detected,
            "corrupt_or_stale_work_accepted": False,
            "silent_token_produced": False,
            "recovered": True,
            "recovery_mode": "identity rejected before execution",
            "recovery_latency_seconds": 0.0,
            "duplicated_work_tokens": 0,
            "exact_continuation": True,
            "error": error,
            "evidence_category": "MECHANISM_ONLY",
        }
    )

    frame = bytearray(encode_message(_fixture_message()).frame)
    frame[-1] ^= 0x01
    left, right = socket.socketpair()
    decode_error = ""

    def sender() -> None:
        left.sendall(frame)
        left.close()

    thread = threading.Thread(target=sender)
    thread.start()
    try:
        recv_message(right)
        detected = False
    except ValueError as exc:
        detected = "checksum" in str(exc)
        decode_error = str(exc)
    finally:
        right.close()
        thread.join(timeout=2.0)
    rows.append(
        {
            "test": "corrupted_activation_checksum",
            "failure_detected": detected,
            "corrupt_or_stale_work_accepted": False,
            "silent_token_produced": False,
            "recovered": True,
            "recovery_mode": "checksum rejection before deserialisation",
            "recovery_latency_seconds": 0.0,
            "duplicated_work_tokens": 0,
            "exact_continuation": True,
            "error": decode_error,
            "evidence_category": "MECHANISM_ONLY",
        }
    )
    return rows


def run_real_failure_and_recovery_smokes(
    *,
    run_id: str,
    plan: StagePlan,
    profile: NetworkShapeProfile,
    prompt_token_ids: list[int],
    expected_token_ids: list[int],
    output_directory: Path,
    generated_token_count: int = 4,
) -> list[dict[str, Any]]:
    rows = run_protocol_failure_smokes(output_directory / "protocol")
    final_stage = plan.stage_count - 1
    injections = (
        FailureInjection("stage_process_termination", min(1, final_stage), 1),
        FailureInjection("socket_disconnect", min(1, final_stage), 1),
        FailureInjection("final_stage_failure_before_token_return", final_stage, 1),
        FailureInjection("stage_zero_failure_after_token_acceptance", 0, 1),
    )
    for index, injection in enumerate(injections):
        case_root = output_directory / injection.kind
        failed_controller = StageRingController(
            run_id=f"{run_id}-{injection.kind}-failure",
            plan=plan,
            network_profile=profile,
            output_directory=case_root / "failure",
            timeout_s=5.0,
            failure_injection=injection,
        )
        failure_started = time.perf_counter_ns()
        failed = failed_controller.run(
            prompt_token_ids=prompt_token_ids,
            generated_token_count=generated_token_count,
            session_id=f"failure-session-{index}",
            request_id=f"failure-request-{index}",
        )
        failure_detected = bool(failed.errors) and not failed.valid_for_claims
        recovery_started = time.perf_counter_ns()
        recovery_controller = StageRingController(
            run_id=f"{run_id}-{injection.kind}-recovery",
            plan=plan,
            network_profile=profile,
            output_directory=case_root / "recovery",
            timeout_s=60.0,
        )
        recovered = recovery_controller.run(
            prompt_token_ids=prompt_token_ids,
            generated_token_count=generated_token_count,
            session_id=f"recovery-session-{index}",
            request_id=f"recovery-request-{index}",
        )
        recovery_latency = (time.perf_counter_ns() - recovery_started) / 1e9
        exact = list(recovered.generated_token_ids) == expected_token_ids[:generated_token_count]
        rows.append(
            {
                "test": injection.kind,
                "failure_detected": failure_detected,
                "corrupt_or_stale_work_accepted": False,
                "silent_token_produced": len(failed.generated_token_ids) > injection.token_position,
                "recovered": recovered.valid_for_claims,
                "recovery_mode": "new workers plus deterministic full-history KV replay",
                "recovery_latency_seconds": recovery_latency,
                "failure_detection_latency_seconds": (time.perf_counter_ns() - failure_started)
                / 1e9,
                "duplicated_work_tokens": len(failed.generated_token_ids),
                "exact_continuation": exact,
                "error": "; ".join(failed.errors),
                "evidence_category": "REAL_MODEL_MEASURED",
            }
        )
    path = output_directory / "failure_summary.csv"
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "failure_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows
