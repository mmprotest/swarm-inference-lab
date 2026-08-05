"""Route and data-envelope integrity helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import threading
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping

from pydantic import Field, PositiveInt, model_validator

from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.model.partition import StageAssignment
from swarm_inference.protocol.messages import DataPlaneEnvelope, FinalResultMessage, RoutePlan
from swarm_inference.security.identity import WorkerIdentity, public_key_fingerprint
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


class RouteLeaseParticipant(StrictModel):
    """One exact worker/stage identity authorized by a signed route lease."""

    worker_id: str
    worker_public_key: str
    worker_public_key_fingerprint: str
    control_endpoint: str
    data_endpoint: str
    stage_id: int = Field(ge=0)
    assignment: StageAssignment
    device: str
    dtype: str

    @model_validator(mode="after")
    def validate_identity(self) -> RouteLeaseParticipant:
        actual = public_key_fingerprint(self.worker_public_key)
        if not hmac.compare_digest(actual, self.worker_public_key_fingerprint):
            raise ValueError("route participant public-key fingerprint mismatch")
        if self.assignment.stage_id != self.stage_id:
            raise ValueError("route participant assignment does not match its stage ID")
        return self


class SignedRouteLease(StrictModel):
    """Coordinator-authorized, finite-lived product stage topology."""

    topology_id: str
    route_generation: PositiveInt
    model_id: str
    model_revision: str
    tokenizer_revision: str
    adapter_id: str
    dtype: str
    participants: list[RouteLeaseParticipant]
    lease_issued_unix_ns: PositiveInt
    lease_expiry_unix_ns: PositiveInt
    nonce: str
    coordinator_identity: str
    coordinator_public_key: str
    coordinator_public_key_fingerprint: str
    signature: str = ""

    @model_validator(mode="after")
    def validate_lease(self) -> SignedRouteLease:
        if self.lease_expiry_unix_ns <= self.lease_issued_unix_ns:
            raise ValueError("route lease expiry must follow its issue time")
        stages = [participant.stage_id for participant in self.participants]
        if stages != list(range(len(stages))):
            raise ValueError("route lease stages must be ordered and contiguous")
        if len({participant.worker_id for participant in self.participants}) != len(
            self.participants
        ):
            raise ValueError("route lease worker identities must be unique")
        actual = public_key_fingerprint(self.coordinator_public_key)
        if not hmac.compare_digest(actual, self.coordinator_public_key_fingerprint):
            raise ValueError("coordinator public-key fingerprint mismatch")
        return self


class PeerHandshake(StrictModel):
    """Signed proof exchanged before worker-to-worker stage-ring frames."""

    worker_id: str
    public_key_fingerprint: str
    topology_id: str
    route_generation: PositiveInt
    stage_id: int = Field(ge=0)
    peer_stage_id: int = Field(ge=0)
    model_revision: str
    nonce: str
    timestamp_unix_ns: PositiveInt
    route_lease_hash: str
    signature: str = ""


class BoundedNonceCache:
    """Thread-safe, bounded replay cache keyed by signed random nonces."""

    def __init__(self, *, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("nonce cache capacity must be positive")
        self.capacity = capacity
        self._values: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, nonce: str) -> None:
        if not nonce:
            raise IntegrityError("signed nonce is missing")
        with self._lock:
            if nonce in self._values:
                raise IntegrityError("signed nonce was replayed")
            self._values[nonce] = None
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


def _signed_payload(model: StrictModel) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json", exclude={"signature"}))


def sign_route_lease(lease: SignedRouteLease, identity: WorkerIdentity) -> SignedRouteLease:
    """Sign a route lease with the persistent coordinator identity."""

    if lease.coordinator_public_key != identity.public_key_b64:
        raise IntegrityError("route lease coordinator public key does not match signer")
    if lease.coordinator_public_key_fingerprint != identity.public_key_fingerprint:
        raise IntegrityError("route lease coordinator fingerprint does not match signer")
    return lease.model_copy(update={"signature": identity.sign(_signed_payload(lease))})


def _trusted_coordinator_key(
    lease: SignedRouteLease,
    trusted_coordinators: Mapping[str, str] | Collection[str],
) -> str:
    if isinstance(trusted_coordinators, Mapping):
        trusted_key = trusted_coordinators.get(lease.coordinator_identity)
        if trusted_key is None:
            raise IntegrityError("unknown coordinator identity")
        if trusted_key != lease.coordinator_public_key:
            raise IntegrityError("coordinator identity public key mismatch")
        return trusted_key
    trusted = set(trusted_coordinators)
    if (
        lease.coordinator_public_key not in trusted
        and lease.coordinator_public_key_fingerprint not in trusted
    ):
        raise IntegrityError("unknown coordinator identity")
    return lease.coordinator_public_key


def verify_route_lease(
    lease: SignedRouteLease,
    trusted_coordinators: Mapping[str, str] | Collection[str],
    *,
    now_unix_ns: int | None = None,
    future_tolerance_ns: int = 30_000_000_000,
    nonce_cache: BoundedNonceCache | None = None,
) -> None:
    """Verify signature, trust, time bounds, and optional one-time use."""

    if not lease.signature:
        raise IntegrityError("route lease signature is missing")
    if future_tolerance_ns < 0:
        raise ValueError("future timestamp tolerance cannot be negative")
    trusted_key = _trusted_coordinator_key(lease, trusted_coordinators)
    from swarm_inference.security.signatures import verify_signature

    verify_signature(trusted_key, _signed_payload(lease), lease.signature)
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if lease.lease_expiry_unix_ns <= now:
        raise IntegrityError("route lease has expired")
    if lease.lease_issued_unix_ns > now + future_tolerance_ns:
        raise IntegrityError("route lease is future-dated outside tolerance")
    if nonce_cache is not None:
        nonce_cache.add(lease.nonce)


def route_lease_hash(lease: SignedRouteLease) -> str:
    """Hash the complete signed lease used to bind peer handshakes."""

    return hashlib.sha256(canonical_json_bytes(lease.model_dump(mode="json"))).hexdigest()


def sign_peer_handshake(
    handshake: PeerHandshake,
    identity: WorkerIdentity,
) -> PeerHandshake:
    if handshake.public_key_fingerprint != identity.public_key_fingerprint:
        raise IntegrityError("peer handshake fingerprint does not match signer")
    return handshake.model_copy(update={"signature": identity.sign(_signed_payload(handshake))})


def verify_peer_handshake(
    handshake: PeerHandshake,
    lease: SignedRouteLease,
    *,
    expected_worker_id: str,
    expected_stage_id: int,
    expected_peer_stage_id: int,
    now_unix_ns: int | None = None,
    timestamp_tolerance_ns: int = 30_000_000_000,
    nonce_cache: BoundedNonceCache | None = None,
) -> None:
    """Authenticate a peer against an already verified route lease."""

    if not handshake.signature:
        raise IntegrityError("peer handshake signature is missing")
    if timestamp_tolerance_ns < 0:
        raise ValueError("peer timestamp tolerance cannot be negative")
    if (
        handshake.worker_id != expected_worker_id
        or handshake.stage_id != expected_stage_id
        or handshake.peer_stage_id != expected_peer_stage_id
    ):
        raise IntegrityError("peer handshake identity mismatch")
    if (
        handshake.topology_id != lease.topology_id
        or handshake.route_generation != lease.route_generation
        or handshake.model_revision != lease.model_revision
        or handshake.route_lease_hash != route_lease_hash(lease)
    ):
        raise IntegrityError("peer handshake route identity mismatch")
    participant = next(
        (item for item in lease.participants if item.worker_id == expected_worker_id),
        None,
    )
    if participant is None or participant.stage_id != expected_stage_id:
        raise IntegrityError("peer is not authorized by the installed route")
    if not hmac.compare_digest(
        handshake.public_key_fingerprint,
        participant.worker_public_key_fingerprint,
    ):
        raise IntegrityError("peer public-key fingerprint mismatch")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if abs(now - handshake.timestamp_unix_ns) > timestamp_tolerance_ns:
        raise IntegrityError("peer handshake timestamp is stale or future-dated")
    from swarm_inference.security.signatures import verify_signature

    verify_signature(
        participant.worker_public_key,
        _signed_payload(handshake),
        handshake.signature,
    )
    if nonce_cache is not None:
        nonce_cache.add(handshake.nonce)


def verify_worker_route_lease(
    lease: SignedRouteLease,
    trusted_coordinators: Mapping[str, str] | Collection[str],
    *,
    worker_id: str,
    worker_public_key: str,
    control_endpoint: str,
    data_endpoint: str,
    topology_id: str,
    route_generation: int,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    assignment: StageAssignment,
    device: str,
    dtype: str,
    last_route_generation: int | None = None,
    now_unix_ns: int | None = None,
    future_tolerance_ns: int = 30_000_000_000,
    nonce_cache: BoundedNonceCache | None = None,
) -> None:
    """Verify a lease and bind it to this exact worker route installation."""

    verify_route_lease(
        lease,
        trusted_coordinators,
        now_unix_ns=now_unix_ns,
        future_tolerance_ns=future_tolerance_ns,
        nonce_cache=nonce_cache,
    )
    if last_route_generation is not None and route_generation <= last_route_generation:
        raise IntegrityError("stale route generation")
    if (
        lease.topology_id != topology_id
        or lease.route_generation != route_generation
        or lease.model_id != model_id
        or lease.model_revision != model_revision
        or lease.tokenizer_revision != tokenizer_revision
        or lease.dtype != dtype
    ):
        raise IntegrityError("route lease model or topology identity mismatch")
    participant = next((item for item in lease.participants if item.worker_id == worker_id), None)
    if participant is None:
        raise IntegrityError("worker identity is absent from route lease")
    if (
        participant.worker_public_key != worker_public_key
        or participant.worker_public_key_fingerprint != public_key_fingerprint(worker_public_key)
    ):
        raise IntegrityError("worker identity mismatch")
    if (
        participant.control_endpoint != control_endpoint
        or participant.data_endpoint != data_endpoint
    ):
        raise IntegrityError("worker endpoint mismatch")
    if (
        participant.assignment != assignment
        or participant.stage_id != assignment.stage_id
        or participant.device != device
        or participant.dtype != dtype
    ):
        raise IntegrityError("worker route assignment mismatch")


__all__ = [
    "BoundedNonceCache",
    "PeerHandshake",
    "RouteLeaseParticipant",
    "SignedRouteLease",
    "decode_route_key",
    "encode_route_key",
    "route_lease_hash",
    "sign_data_envelope",
    "sign_final_result",
    "sign_peer_handshake",
    "sign_route_lease",
    "sign_route_plan",
    "verify_data_envelope",
    "verify_final_result",
    "verify_peer_handshake",
    "verify_route_lease",
    "verify_route_plan",
    "verify_worker_route_lease",
]
