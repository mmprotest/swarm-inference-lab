"""Route and data-envelope integrity helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac

from swarm_inference.exceptions import IntegrityError
from swarm_inference.protocol.messages import DataPlaneEnvelope, FinalResultMessage, RoutePlan
from swarm_inference.security.signatures import canonical_json_bytes


def encode_route_key(key: bytes) -> str:
    if len(key) < 32:
        raise ValueError("route signing keys must be at least 256 bits")
    return base64.b64encode(key).decode("ascii")


def decode_route_key(encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise IntegrityError("route signing key is not valid base64") from exc
    if len(key) < 32:
        raise IntegrityError("route signing key is shorter than 256 bits")
    return key


def _hmac(payload: dict[str, object], key: bytes) -> str:
    return hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def sign_route_plan(plan: RoutePlan, key: bytes) -> RoutePlan:
    payload = plan.model_dump(mode="json", exclude={"signature"})
    return plan.model_copy(update={"signature": _hmac(payload, key)})


def verify_route_plan(plan: RoutePlan, key: bytes) -> None:
    payload = plan.model_dump(mode="json", exclude={"signature"})
    expected = _hmac(payload, key)
    if not hmac.compare_digest(plan.signature, expected):
        raise IntegrityError(f"route {plan.route_id} signature mismatch")


def sign_data_envelope(envelope: DataPlaneEnvelope, key: bytes) -> DataPlaneEnvelope:
    payload = envelope.model_dump(
        mode="json",
        exclude={"signature", "tensor_payload"},
    )
    return envelope.model_copy(update={"signature": _hmac(payload, key)})


def verify_data_envelope(envelope: DataPlaneEnvelope, key: bytes) -> None:
    payload = envelope.model_dump(
        mode="json",
        exclude={"signature", "tensor_payload"},
    )
    expected = _hmac(payload, key)
    if not hmac.compare_digest(envelope.signature, expected):
        raise IntegrityError(f"data message {envelope.message_id} signature mismatch")


def sign_final_result(message: FinalResultMessage, key: bytes) -> FinalResultMessage:
    payload = message.model_dump(
        mode="json",
        exclude={"signature": True, "result": {"tensor_payload"}},
    )
    return message.model_copy(update={"signature": _hmac(payload, key)})


def verify_final_result(message: FinalResultMessage, key: bytes) -> None:
    payload = message.model_dump(
        mode="json",
        exclude={"signature": True, "result": {"tensor_payload"}},
    )
    expected = _hmac(payload, key)
    if not hmac.compare_digest(message.signature, expected):
        raise IntegrityError(f"final result {message.message_id} signature mismatch")
