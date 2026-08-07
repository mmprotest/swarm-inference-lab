"""Single-use, transcript-bound cluster pairing and membership control."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from swarm_inference.cluster.models import (
    SECURE_WAN_SECURITY_CLASSIFICATION,
    ArtifactCacheDocument,
    ClusterAuditEvent,
    ClusterMetadata,
    NodeMembership,
    NodeMetadata,
    NodeRevocation,
    PairingSession,
)
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.exceptions import (
    CompatibilityError,
    IntegrityError,
    PairingError,
    PairingExpiredError,
    PairingRateLimitError,
)
from swarm_inference.protocol.cluster import (
    ClusterNodeStatus,
    ClusterRequestAuthentication,
    ClusterRevokeRequest,
    ClusterRevokeResponse,
    ClusterStatusRequest,
    ClusterStatusResponse,
    NodeLeaveRequest,
    NodeLeaveResponse,
    NodeUpdateRequest,
    NodeUpdateResponse,
    PairingChallenge,
    PairingChallengePayload,
    PairingCompleteRequest,
    PairingCompleteResponse,
    PairingCoordinatorCompletionPayload,
    PairingCreateRequest,
    PairingCreateResponse,
    PairingHello,
    PairingNodeCompletionPayload,
    PairingResult,
    verify_hello_identity,
)
from swarm_inference.protocol.routes import BoundedNonceCache
from swarm_inference.security.identity import (
    CoordinatorIdentity,
    WorkerIdentity,
    public_key_fingerprint,
)
from swarm_inference.security.signatures import canonical_json_bytes, verify_signature
from swarm_inference.security.tls import (
    certificate_sha256,
    issue_node_certificate,
    validate_certificate_binding,
)
from swarm_inference.security.trust_store import WorkerTrustStore

PAIRING_URI_SCHEME: Final = "swarm"
LEGACY_PAIRING_URI_SCHEME: Final = "swarm+pair"
PAIRING_PROTOCOL_LABEL: Final = b"swarm-cluster-pairing-v1"
PAIRING_DEFAULT_TTL_SECONDS: Final = 600
PAIRING_MINIMUM_SECRET_BYTES: Final = 16
PAIRING_DEFAULT_SECRET_BYTES: Final = 32
PAIRING_AES_NONCE_BYTES: Final = 12
PAIRING_NONCE_BYTES: Final = 16
PAIRING_CHALLENGE_BYTES: Final = 32


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise PairingError("pairing message contains invalid base64") from exc


def _url_b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _url_unb64(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise PairingError("pairing URI contains invalid encoded data") from exc


def _x25519_public_key(private_key: X25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _derive_session_key(*, shared_secret: bytes, pairing_secret: bytes, transcript: bytes) -> bytes:
    transcript_hash = hashlib.sha256(transcript).digest()
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=hashlib.sha256(pairing_secret).digest(),
        info=PAIRING_PROTOCOL_LABEL + b"\0" + transcript_hash,
    ).derive(shared_secret)


def _proof(key: bytes, role: bytes, transcript_hash: str) -> str:
    payload = PAIRING_PROTOCOL_LABEL + b"\0" + role + b"\0" + transcript_hash.encode("ascii")
    return _b64(hmac.digest(key, payload, "sha256"))


def _proof_matches(actual: str, expected: str) -> bool:
    try:
        actual_bytes = _unb64(actual)
        expected_bytes = _unb64(expected)
    except PairingError:
        return False
    return hmac.compare_digest(actual_bytes, expected_bytes)


def _signature_payload(
    *,
    role: str,
    transcript_hash: str,
    challenge_or_binding: str,
    secret_proof: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "protocol": PAIRING_PROTOCOL_LABEL.decode("ascii"),
            "role": role,
            "transcript_hash": transcript_hash,
            "challenge_or_binding": challenge_or_binding,
            "secret_proof": secret_proof,
        }
    )


def _aad(transcript: bytes, *, phase: str) -> bytes:
    return PAIRING_PROTOCOL_LABEL + b"\0" + phase.encode("ascii") + b"\0" + transcript


def _encrypt_json(
    key: bytes,
    payload: dict[str, Any],
    *,
    nonce: bytes,
    aad: bytes,
) -> str:
    plaintext = canonical_json_bytes(payload)
    return _b64(AESGCM(key).encrypt(nonce, plaintext, aad))


def _decrypt_json(key: bytes, ciphertext: str, *, nonce: str, aad: bytes) -> dict[str, Any]:
    try:
        plaintext = AESGCM(key).decrypt(_unb64(nonce), _unb64(ciphertext), aad)
        value = json.loads(plaintext)
    except (InvalidTag, UnicodeDecodeError, json.JSONDecodeError, PairingError) as exc:
        raise PairingError("pairing secret or encrypted transcript was rejected") from exc
    if not isinstance(value, dict):
        raise PairingError("decrypted pairing payload is not an object")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class PairingInvitation:
    """Secret-bearing invitation whose repr and redacted form never disclose it."""

    coordinator_endpoint: str
    session_id: str
    pairing_secret: bytes
    coordinator_ephemeral_public_key: bytes
    coordinator_certificate_pem: str | None = None

    def __post_init__(self) -> None:
        if len(self.pairing_secret) < PAIRING_MINIMUM_SECRET_BYTES:
            raise ValueError("pairing secret must contain at least 128 bits")
        if len(self.coordinator_ephemeral_public_key) != 32:
            raise ValueError("coordinator X25519 public key must contain 32 bytes")

    def __repr__(self) -> str:
        return (
            "PairingInvitation(coordinator_endpoint="
            f"{self.coordinator_endpoint!r}, session_id={self.session_id!r}, "
            "pairing_secret=<redacted>, coordinator_ephemeral_public_key=<public>)"
        )

    def uri(self) -> str:
        payload: dict[str, Any] = {
            "key": _url_b64(self.coordinator_ephemeral_public_key),
            "secret": _url_b64(self.pairing_secret),
            "session": self.session_id,
            "version": 1,
        }
        if self.coordinator_certificate_pem is not None:
            payload["ca"] = _url_b64(self.coordinator_certificate_pem.encode("ascii"))
            payload["version"] = 2
        invitation_data = canonical_json_bytes(payload)
        return (
            f"{PAIRING_URI_SCHEME}://{self.coordinator_endpoint}/join/{_url_b64(invitation_data)}"
        )

    def redacted_uri(self) -> str:
        return f"{PAIRING_URI_SCHEME}://{self.coordinator_endpoint}/join/REDACTED"

    @classmethod
    def parse(cls, uri: str) -> PairingInvitation:
        parsed = urlsplit(uri)
        if not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise PairingError("pairing URI has an invalid coordinator endpoint")
        if parsed.scheme == PAIRING_URI_SCHEME:
            if parsed.query or parsed.fragment or not parsed.path.startswith("/join/"):
                raise PairingError("pairing URI has an unsupported scheme or path")
            encoded = parsed.path.removeprefix("/join/")
            if not encoded or "/" in encoded:
                raise PairingError("pairing URI invitation data is malformed")
            try:
                payload = json.loads(_url_unb64(encoded))
            except (UnicodeDecodeError, json.JSONDecodeError, PairingError) as exc:
                raise PairingError("pairing URI invitation data is malformed") from exc
            if not isinstance(payload, dict):
                raise PairingError("pairing URI invitation data is malformed")
            version = payload.get("version")
            expected_fields = (
                {"version", "session", "secret", "key", "ca"}
                if version == 2
                else {"version", "session", "secret", "key"}
            )
            if (
                set(payload) != expected_fields
                or version not in {1, 2}
                or not all(isinstance(payload[name], str) for name in ("session", "secret", "key"))
            ):
                raise PairingError("pairing URI invitation data is malformed")
            session_id = payload["session"]
            secret = _url_unb64(payload["secret"])
            key = _url_unb64(payload["key"])
            certificate_pem = _url_unb64(payload["ca"]).decode("ascii") if version == 2 else None
        elif parsed.scheme == LEGACY_PAIRING_URI_SCHEME and parsed.path == "/join":
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
            if set(query) != {"session", "secret", "key"} or any(
                len(values) != 1 for values in query.values()
            ):
                raise PairingError("pairing URI query is malformed")
            session_id = query["session"][0]
            secret = _url_unb64(query["secret"][0])
            key = _url_unb64(query["key"][0])
            certificate_pem = None
        else:
            raise PairingError("pairing URI has an unsupported scheme or path")
        return cls(
            coordinator_endpoint=parsed.netloc,
            session_id=session_id,
            pairing_secret=secret,
            coordinator_ephemeral_public_key=key,
            coordinator_certificate_pem=certificate_pem,
        )


@dataclass(slots=True)
class _Handshake:
    hello: PairingHello
    hello_hash: str
    transcript: bytes
    transcript_hash: str
    session_key: bytes
    challenge: str


@dataclass(slots=True)
class _LiveSession:
    public: PairingSession
    pairing_secret: bytes
    ephemeral_private_key: X25519PrivateKey
    handshake: _Handshake | None = None


HelloRpc = Callable[[PairingHello], Awaitable[PairingChallenge]]
CompleteRpc = Callable[[PairingCompleteRequest], Awaitable[PairingCompleteResponse]]


class PairingManager:
    """Coordinator-side bounded pairing session and membership authority."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        trust_store: WorkerTrustStore,
        coordinator_identity: CoordinatorIdentity,
        cluster: ClusterMetadata,
        clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
        random_bytes: Callable[[int], bytes] = os.urandom,
        maximum_active_sessions: int = 32,
        maximum_attempts_per_session: int = 5,
        maximum_attempts_per_source_window: int = 20,
        source_window_seconds: float = 60.0,
        maximum_tracked_sources: int = 1024,
        authentication_skew_seconds: float = 60.0,
    ) -> None:
        if (
            min(
                maximum_active_sessions,
                maximum_attempts_per_session,
                maximum_attempts_per_source_window,
                maximum_tracked_sources,
            )
            <= 0
        ):
            raise ValueError("pairing bounds must be positive")
        if source_window_seconds <= 0 or authentication_skew_seconds <= 0:
            raise ValueError("pairing time windows must be positive")
        cluster.verify_identity_binding()
        if cluster.coordinator_fingerprint != coordinator_identity.public_key_fingerprint:
            raise IntegrityError("cluster metadata does not match the pairing coordinator")
        self.state = state
        self.trust_store = trust_store
        self.coordinator_identity = coordinator_identity
        self.cluster = cluster
        self.clock_ns = clock_ns
        self.monotonic = monotonic
        self.random_bytes = random_bytes
        self.maximum_active_sessions = maximum_active_sessions
        self.maximum_attempts_per_session = maximum_attempts_per_session
        self.maximum_attempts_per_source_window = maximum_attempts_per_source_window
        self.source_window_seconds = source_window_seconds
        self.maximum_tracked_sources = maximum_tracked_sources
        self.authentication_skew_ns = int(authentication_skew_seconds * 1e9)
        self._sessions: dict[str, _LiveSession] = {}
        self._source_attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._nonce_cache = BoundedNonceCache(capacity=4096)
        self._lock = asyncio.Lock()
        invalidated = self.state.invalidate_active_pairing_sessions()
        if invalidated:
            self._audit("pairing_invalidated", detail=f"{invalidated} session(s) invalidated")

    def _audit(
        self,
        event_type: str,
        *,
        node_id: str | None = None,
        session_id: str | None = None,
        category: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                timestamp_unix_ns=self.clock_ns(),
                cluster_id=self.cluster.cluster_id,
                node_id=node_id,
                pairing_session_id=session_id,
                category=category,
                detail=detail,
            )
        )

    def _expire_sessions(self) -> None:
        now = self.clock_ns()
        for session_id, live in list(self._sessions.items()):
            if live.public.expires_at_unix_ns > now:
                continue
            live.public = live.public.model_copy(
                update={"state": "expired", "last_rejection_reason": "pairing session expired"}
            )
            self.state.save_pairing_session(live.public)
            self.state.retire_pairing_invitation(session_id)
            self._audit("pairing_expired", session_id=session_id)
            del self._sessions[session_id]

    def _record_source_attempt(self, source_address: str) -> None:
        source = source_address[:256] or "unknown"
        now = self.monotonic()
        values = self._source_attempts.pop(source, deque())
        while values and now - values[0] > self.source_window_seconds:
            values.popleft()
        if len(values) >= self.maximum_attempts_per_source_window:
            self._source_attempts[source] = values
            raise PairingRateLimitError("pairing source rate limit exceeded")
        values.append(now)
        self._source_attempts[source] = values
        while len(self._source_attempts) > self.maximum_tracked_sources:
            self._source_attempts.popitem(last=False)

    def _live_session(self, session_id: str) -> _LiveSession:
        self._expire_sessions()
        live = self._sessions.get(session_id)
        if live is None:
            known = next(
                (
                    item
                    for item in self.state.load_pairing_sessions().sessions
                    if item.session_id == session_id
                ),
                None,
            )
            if known is not None and known.state == "consumed":
                raise PairingError("pairing session was already consumed")
            if known is not None and known.state in {"expired", "invalidated"}:
                raise PairingExpiredError(f"pairing session is {known.state}")
            raise PairingError("pairing session is unknown or unavailable")
        if live.public.state != "active":
            raise PairingError(f"pairing session is {live.public.state}")
        return live

    async def create_session(
        self,
        coordinator_endpoint: str,
        *,
        ttl_seconds: int = PAIRING_DEFAULT_TTL_SECONDS,
    ) -> PairingInvitation:
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise ValueError("pairing TTL must be between 1 and 3600 seconds")
        async with self._lock:
            self._expire_sessions()
            if len(self._sessions) >= self.maximum_active_sessions:
                raise PairingRateLimitError("maximum active pairing sessions reached")
            session_id = _url_b64(self.random_bytes(18))
            while session_id in self._sessions:
                session_id = _url_b64(self.random_bytes(18))
            secret = self.random_bytes(PAIRING_DEFAULT_SECRET_BYTES)
            ephemeral_private = X25519PrivateKey.from_private_bytes(self.random_bytes(32))
            ephemeral_public = _x25519_public_key(ephemeral_private)
            now = self.clock_ns()
            public = PairingSession(
                session_id=session_id,
                coordinator_endpoint=coordinator_endpoint,
                coordinator_ephemeral_public_key=_b64(ephemeral_public),
                created_at_unix_ns=now,
                expires_at_unix_ns=now + int(ttl_seconds * 1e9),
                maximum_attempts=self.maximum_attempts_per_session,
            )
            self.state.save_pairing_session(public)
            self._sessions[session_id] = _LiveSession(
                public=public,
                pairing_secret=secret,
                ephemeral_private_key=ephemeral_private,
            )
            self._audit("pairing_created", session_id=session_id)
            return PairingInvitation(
                coordinator_endpoint=coordinator_endpoint,
                session_id=session_id,
                pairing_secret=secret,
                coordinator_ephemeral_public_key=ephemeral_public,
                coordinator_certificate_pem=self.cluster.coordinator_certificate_pem,
            )

    async def create_authenticated_session(
        self, request: PairingCreateRequest
    ) -> PairingCreateResponse:
        body = {"ttl_seconds": request.ttl_seconds}
        actor = self.verify_authentication(
            request.authentication,
            action="pairing-create",
            body=body,
        )
        if actor.node_id != self.cluster.coordinator_id:
            raise IntegrityError("only the coordinator node may create pairing invitations")
        invitation = await self.create_session(
            self.cluster.coordinator_endpoint,
            ttl_seconds=request.ttl_seconds,
        )
        session = next(
            item
            for item in self.state.load_pairing_sessions().sessions
            if item.session_id == invitation.session_id
        )
        return PairingCreateResponse(
            session_id=invitation.session_id,
            pairing_uri=invitation.uri(),
            redacted_uri=invitation.redacted_uri(),
            expires_at_unix_ns=session.expires_at_unix_ns,
        )

    def _transcript(
        self,
        *,
        live: _LiveSession,
        hello: PairingHello,
        server_nonce: str,
    ) -> bytes:
        return canonical_json_bytes(
            {
                "protocol": PAIRING_PROTOCOL_LABEL.decode("ascii"),
                "coordinator_endpoint": live.public.coordinator_endpoint,
                "session_id": live.public.session_id,
                "expires_at_unix_ns": live.public.expires_at_unix_ns,
                "hello": hello.model_dump(mode="json"),
                "coordinator_public_key": self.coordinator_identity.public_key_b64,
                "coordinator_fingerprint": self.coordinator_identity.public_key_fingerprint,
                "coordinator_ephemeral_public_key": live.public.coordinator_ephemeral_public_key,
                "server_nonce": server_nonce,
            }
        )

    def _verify_compatibility(self, hello: PairingHello) -> None:
        if hello.product_protocol_major != self.cluster.product_protocol_major:
            raise CompatibilityError(
                "product protocol major is incompatible: "
                f"node={hello.product_protocol_major}, "
                f"coordinator={self.cluster.product_protocol_major}"
            )
        compatibility = self.cluster.runtime_compatibility
        if not (
            compatibility.minimum_product_protocol_minor
            <= hello.product_protocol_minor
            <= compatibility.maximum_product_protocol_minor
        ):
            raise CompatibilityError(
                "product protocol minor is outside the supported range "
                f"{compatibility.minimum_product_protocol_minor}.."
                f"{compatibility.maximum_product_protocol_minor}"
            )
        if not set(hello.artifact_format_versions).intersection(
            compatibility.artifact_format_versions
        ):
            raise CompatibilityError("node and coordinator share no artifact format version")

    async def begin(self, hello: PairingHello, *, source_address: str) -> PairingChallenge:
        async with self._lock:
            self._record_source_attempt(source_address)
            live = self._live_session(hello.session_id)
            attempts = live.public.attempts + 1
            if attempts > live.public.maximum_attempts:
                live.public = live.public.model_copy(
                    update={
                        "attempts": attempts,
                        "state": "rejected",
                        "last_rejection_reason": "maximum pairing attempts exceeded",
                    }
                )
                self.state.save_pairing_session(live.public)
                del self._sessions[hello.session_id]
                self._audit(
                    "pairing_rejected",
                    session_id=hello.session_id,
                    category="rate-limit",
                    detail="maximum pairing attempts exceeded",
                )
                raise PairingRateLimitError("maximum pairing attempts exceeded")
            live.public = live.public.model_copy(update={"attempts": attempts})
            self.state.save_pairing_session(live.public)
            try:
                verify_hello_identity(hello)
                self._verify_compatibility(hello)
                node_ephemeral = X25519PublicKey.from_public_bytes(
                    _unb64(hello.node_ephemeral_public_key)
                )
            except (ValueError, IntegrityError, CompatibilityError, PairingError) as exc:
                self._audit(
                    "pairing_rejected",
                    node_id=hello.node_id,
                    session_id=hello.session_id,
                    category="authentication",
                    detail=str(exc),
                )
                raise PairingError(str(exc)) from exc
            server_nonce = _b64(self.random_bytes(PAIRING_NONCE_BYTES))
            transcript = self._transcript(live=live, hello=hello, server_nonce=server_nonce)
            transcript_hash = hashlib.sha256(transcript).hexdigest()
            shared_secret = live.ephemeral_private_key.exchange(node_ephemeral)
            key = _derive_session_key(
                shared_secret=shared_secret,
                pairing_secret=live.pairing_secret,
                transcript=transcript,
            )
            challenge = _b64(self.random_bytes(PAIRING_CHALLENGE_BYTES))
            coordinator_proof = _proof(key, b"coordinator-challenge", transcript_hash)
            signature = self.coordinator_identity.sign(
                _signature_payload(
                    role="coordinator-challenge",
                    transcript_hash=transcript_hash,
                    challenge_or_binding=challenge,
                    secret_proof=coordinator_proof,
                )
            )
            payload = PairingChallengePayload(
                transcript_hash=transcript_hash,
                challenge=challenge,
                coordinator_secret_proof=coordinator_proof,
                coordinator_signature=signature,
            )
            encryption_nonce = self.random_bytes(PAIRING_AES_NONCE_BYTES)
            ciphertext = _encrypt_json(
                key,
                payload.model_dump(mode="json"),
                nonce=encryption_nonce,
                aad=_aad(transcript, phase="challenge"),
            )
            hello_hash = hashlib.sha256(
                canonical_json_bytes(hello.model_dump(mode="json"))
            ).hexdigest()
            live.handshake = _Handshake(
                hello=hello,
                hello_hash=hello_hash,
                transcript=transcript,
                transcript_hash=transcript_hash,
                session_key=key,
                challenge=challenge,
            )
            return PairingChallenge(
                session_id=hello.session_id,
                coordinator_public_key=self.coordinator_identity.public_key_b64,
                coordinator_fingerprint=self.coordinator_identity.public_key_fingerprint,
                coordinator_ephemeral_public_key=live.public.coordinator_ephemeral_public_key,
                server_nonce=server_nonce,
                expires_at_unix_ns=live.public.expires_at_unix_ns,
                encryption_nonce=_b64(encryption_nonce),
                encrypted_payload=ciphertext,
            )

    def _rollback_pairing_records(
        self,
        *,
        node_id: str,
        previous_node: NodeMetadata | None,
        previous_membership: NodeMembership | None,
        trust_preexisting: bool,
        fingerprint: str,
    ) -> None:
        if not trust_preexisting:
            self.trust_store.untrust(fingerprint)
        if previous_node is None:
            self.state.remove_node(node_id)
        else:
            self.state.save_node(previous_node)
        if previous_membership is None:
            self.state.remove_membership(node_id)
        else:
            self.state.save_membership(previous_membership)

    async def complete(
        self,
        request: PairingCompleteRequest,
        *,
        source_address: str,
    ) -> PairingCompleteResponse:
        async with self._lock:
            self._record_source_attempt(source_address)
            live = self._live_session(request.session_id)
            handshake = live.handshake
            if handshake is None:
                raise PairingError("pairing completion has no active challenge")
            if not hmac.compare_digest(request.hello_hash, handshake.hello_hash):
                self._audit(
                    "pairing_rejected",
                    node_id=handshake.hello.node_id,
                    session_id=request.session_id,
                    category="authentication",
                    detail="pairing hello hash mismatch",
                )
                raise PairingError("pairing hello hash mismatch")
            raw = _decrypt_json(
                handshake.session_key,
                request.encrypted_payload,
                nonce=request.encryption_nonce,
                aad=_aad(handshake.transcript, phase="node-completion"),
            )
            try:
                payload = PairingNodeCompletionPayload.model_validate(raw)
            except ValueError as exc:
                raise PairingError("node pairing completion payload is invalid") from exc
            expected_proof = _proof(
                handshake.session_key,
                b"node-completion",
                handshake.transcript_hash,
            )
            if not hmac.compare_digest(payload.transcript_hash, handshake.transcript_hash):
                raise PairingError("node completion transcript does not match")
            if not hmac.compare_digest(payload.challenge, handshake.challenge):
                raise PairingError("node completion challenge does not match")
            if not _proof_matches(payload.node_secret_proof, expected_proof):
                raise PairingError("node pairing-secret proof is invalid")
            try:
                verify_signature(
                    handshake.hello.node_public_key,
                    _signature_payload(
                        role="node-completion",
                        transcript_hash=handshake.transcript_hash,
                        challenge_or_binding=handshake.challenge,
                        secret_proof=payload.node_secret_proof,
                    ),
                    payload.node_signature,
                )
            except IntegrityError as exc:
                raise PairingError("node identity signature is invalid") from exc
            metadata = payload.node_metadata
            metadata.verify_identity_binding()
            if (
                metadata.node_id != handshake.hello.node_id
                or metadata.public_key != handshake.hello.node_public_key
                or metadata.fingerprint != handshake.hello.node_fingerprint
            ):
                raise PairingError("node metadata is not bound to the pairing hello")
            now = self.clock_ns()
            metadata = metadata.model_copy(
                update={
                    "joined_at_unix_ns": now,
                    "last_seen_at_unix_ns": now,
                    "revoked": False,
                    "revoked_at_unix_ns": None,
                    "revocation_reason": None,
                }
            )
            ca_certificate = self.cluster.coordinator_certificate_pem
            if (
                ca_certificate is None
                and self.cluster.security_classification == SECURE_WAN_SECURITY_CLASSIFICATION
            ):
                raise PairingError(
                    "cluster has no pinned TLS certificate; rotate the coordinator "
                    "transport credentials before accepting WAN peers"
                )
            if ca_certificate is not None:
                if metadata.tls_public_key_pem is None:
                    raise PairingError("node did not provide a signed P-256 TLS public key")
                node_certificate = issue_node_certificate(
                    self.coordinator_identity,
                    ca_certificate_pem=ca_certificate,
                    cluster_id=self.cluster.cluster_id,
                    node_public_key_b64=metadata.public_key,
                    node_fingerprint=metadata.fingerprint,
                    node_tls_public_key_pem=metadata.tls_public_key_pem,
                )
                metadata = metadata.model_copy(
                    update={
                        "tls_certificate_pem": node_certificate,
                        "tls_certificate_sha256": certificate_sha256(node_certificate),
                    }
                )
            metadata.verify_identity_binding()
            membership = NodeMembership(
                cluster_id=self.cluster.cluster_id,
                node_id=metadata.node_id,
                node_public_key=metadata.public_key,
                node_fingerprint=metadata.fingerprint,
                coordinator_public_key=self.coordinator_identity.public_key_b64,
                coordinator_fingerprint=self.coordinator_identity.public_key_fingerprint,
                joined_at_unix_ns=now,
                status="active",
            )
            membership.verify_identity_bindings()
            binding = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "cluster": self.cluster.model_dump(mode="json"),
                        "membership": membership.model_dump(mode="json"),
                        "node_metadata": metadata.model_dump(mode="json"),
                    }
                )
            ).hexdigest()
            final_proof = _proof(
                handshake.session_key,
                b"coordinator-completion",
                handshake.transcript_hash,
            )
            final_signature = self.coordinator_identity.sign(
                _signature_payload(
                    role="coordinator-completion",
                    transcript_hash=handshake.transcript_hash,
                    challenge_or_binding=binding,
                    secret_proof=final_proof,
                )
            )
            response_payload = PairingCoordinatorCompletionPayload(
                transcript_hash=handshake.transcript_hash,
                coordinator_secret_proof=final_proof,
                coordinator_signature=final_signature,
                cluster=self.cluster,
                membership=membership,
                node_metadata=metadata,
            )
            response_nonce = self.random_bytes(PAIRING_AES_NONCE_BYTES)
            encrypted_response = _encrypt_json(
                handshake.session_key,
                response_payload.model_dump(mode="json"),
                nonce=response_nonce,
                aad=_aad(handshake.transcript, phase="coordinator-completion"),
            )
            previous_node = self.state.node(metadata.node_id)
            previous_membership = self.state.membership(metadata.node_id)
            trust_preexisting = self.trust_store.contains(metadata.fingerprint)
            consumed = live.public.model_copy(
                update={
                    "state": "consumed",
                    "consumed_at_unix_ns": now,
                    "node_id": metadata.node_id,
                    "last_rejection_reason": None,
                }
            )
            try:
                self.state.save_node(metadata)
                self.state.save_membership(membership)
                self.trust_store.trust(
                    metadata.fingerprint,
                    label=metadata.node_id,
                    notes="secure cluster pairing",
                )
                self.state.save_pairing_session(consumed)
                self.state.retire_pairing_invitation(request.session_id)
            except BaseException:
                live.public = live.public.model_copy(
                    update={
                        "state": "rejected",
                        "last_rejection_reason": "pairing transaction rolled back",
                    }
                )
                self._rollback_pairing_records(
                    node_id=metadata.node_id,
                    previous_node=previous_node,
                    previous_membership=previous_membership,
                    trust_preexisting=trust_preexisting,
                    fingerprint=metadata.fingerprint,
                )
                raise
            live.public = consumed
            del self._sessions[request.session_id]
            self._audit(
                "pairing_consumed",
                node_id=metadata.node_id,
                session_id=request.session_id,
            )
            self._audit("node_joined", node_id=metadata.node_id)
            return PairingCompleteResponse(
                session_id=request.session_id,
                consumed=True,
                encryption_nonce=_b64(response_nonce),
                encrypted_payload=encrypted_response,
            )

    def _authentication_payload(
        self,
        authentication: ClusterRequestAuthentication,
        *,
        action: str,
        body: dict[str, Any],
    ) -> bytes:
        return canonical_json_bytes(
            {
                "protocol": "swarm-cluster-auth-v1",
                "action": action,
                "node_id": authentication.node_id,
                "timestamp_unix_ns": authentication.timestamp_unix_ns,
                "nonce": authentication.nonce,
                "body": body,
            }
        )

    def verify_authentication(
        self,
        authentication: ClusterRequestAuthentication,
        *,
        action: str,
        body: dict[str, Any],
    ) -> NodeMembership:
        membership = self.state.membership(authentication.node_id)
        if membership is None or membership.status != "active":
            raise IntegrityError("requesting node is not an active cluster member")
        if self.state.is_revoked_fingerprint(membership.node_fingerprint):
            raise IntegrityError("requesting node has been revoked")
        if abs(self.clock_ns() - authentication.timestamp_unix_ns) > self.authentication_skew_ns:
            raise IntegrityError("cluster authentication timestamp is outside the allowed window")
        verify_signature(
            membership.node_public_key,
            self._authentication_payload(authentication, action=action, body=body),
            authentication.signature,
        )
        self._nonce_cache.add(authentication.nonce)
        return membership

    async def status(self, request: ClusterStatusRequest) -> ClusterStatusResponse:
        body = {
            "include_artifacts": request.include_artifacts,
            "include_network": request.include_network,
        }
        self.verify_authentication(request.authentication, action="cluster-status", body=body)
        nodes = [
            ClusterNodeStatus(metadata=node, runtime=self.state.load_runtime(node.node_id))
            for node in self.state.load_nodes().nodes
        ]
        measurements = (
            self.state.load_network_measurements().measurements if request.include_network else []
        )
        artifacts = (
            self.state.load_artifact_cache()
            if request.include_artifacts
            else ArtifactCacheDocument()
        )
        return ClusterStatusResponse(
            cluster=self.cluster,
            nodes=nodes,
            network_links=measurements,
            artifact_entries=artifacts.entries,
            artifact_transfers=artifacts.transfers,
            revocations=self.state.load_revocations().revocations,
            generated_at_unix_ns=self.clock_ns(),
        )

    async def revoke(self, request: ClusterRevokeRequest) -> ClusterRevokeResponse:
        body = {"node_id": request.node_id, "reason": request.reason}
        actor = self.verify_authentication(
            request.authentication,
            action="cluster-revoke",
            body=body,
        )
        if actor.node_id != self.cluster.coordinator_id:
            raise IntegrityError("only the coordinator node may revoke membership")
        node = self.state.node(request.node_id)
        membership = self.state.membership(request.node_id)
        if node is None or membership is None:
            return ClusterRevokeResponse(revoked=False)
        existing = [
            item
            for item in self.state.load_revocations().revocations
            if item.node_id == request.node_id
        ]
        if existing:
            return ClusterRevokeResponse(revoked=True, revocation=existing[-1])
        now = self.clock_ns()
        revocation = NodeRevocation(
            revocation_id=uuid4().hex,
            cluster_id=self.cluster.cluster_id,
            node_id=node.node_id,
            node_fingerprint=node.fingerprint,
            revoked_at_unix_ns=now,
            revoked_by_node_id=actor.node_id,
            reason=request.reason.strip() or "revoked by cluster coordinator",
            generation=membership.membership_generation + 1,
        )
        self.state.append_revocation(revocation)
        self.state.save_node(
            node.model_copy(
                update={
                    "revoked": True,
                    "revoked_at_unix_ns": now,
                    "revocation_reason": revocation.reason,
                }
            )
        )
        self.state.save_membership(
            membership.model_copy(
                update={
                    "status": "revoked",
                    "membership_generation": revocation.generation,
                }
            )
        )
        self.trust_store.untrust(node.fingerprint)
        self._audit("node_revoked", node_id=node.node_id, detail=revocation.reason)
        return ClusterRevokeResponse(revoked=True, revocation=revocation)

    async def leave(self, request: NodeLeaveRequest) -> NodeLeaveResponse:
        body = {"node_id": request.node_id}
        membership = self.verify_authentication(
            request.authentication,
            action="node-leave",
            body=body,
        )
        if membership.node_id != request.node_id:
            raise IntegrityError("a node may only leave its own membership")
        self.state.save_membership(membership.model_copy(update={"status": "left"}))
        self.trust_store.untrust(membership.node_fingerprint)
        self._audit("node_left", node_id=membership.node_id)
        return NodeLeaveResponse(left=True, node_id=membership.node_id)

    async def update_node(self, request: NodeUpdateRequest) -> NodeUpdateResponse:
        # The authentication transcript intentionally covers application fields,
        # not the transport schema marker.  Omitting an absent runtime preserves
        # compatibility with pre-agent members that only publish NodeMetadata.
        body: dict[str, Any] = {"metadata": request.metadata.model_dump(mode="json")}
        if request.runtime is not None:
            body["runtime"] = request.runtime.model_dump(mode="json")
        membership = self.verify_authentication(
            request.authentication,
            action="node-update",
            body=body,
        )
        metadata = request.metadata
        metadata.verify_identity_binding()
        if metadata.node_id != membership.node_id:
            raise IntegrityError("a node may only update its own metadata")
        if (
            metadata.public_key != membership.node_public_key
            or metadata.fingerprint != membership.node_fingerprint
        ):
            raise IntegrityError("node update identity does not match active membership")
        if metadata.revoked:
            raise IntegrityError("active member cannot mark itself revoked through node update")
        previous = self.state.node(metadata.node_id)
        endpoint_changed = previous is not None and (
            previous.control_endpoint != metadata.control_endpoint
            or previous.data_endpoint != metadata.data_endpoint
        )
        now = self.clock_ns()
        metadata = metadata.model_copy(
            update={
                "joined_at_unix_ns": membership.joined_at_unix_ns,
                "last_seen_at_unix_ns": now,
                "revoked": False,
                "revoked_at_unix_ns": None,
                "revocation_reason": None,
            }
        )
        if request.runtime is not None:
            if request.runtime.node_id != metadata.node_id:
                raise IntegrityError("node runtime status belongs to a different node")
            if request.runtime.cluster_id != self.cluster.cluster_id:
                raise IntegrityError("node runtime status belongs to a different cluster")
        self.state.save_node(metadata)
        if request.runtime is not None:
            self.state.save_runtime(request.runtime)
        self._audit(
            "endpoint_changed" if endpoint_changed else "node_reconnected",
            node_id=metadata.node_id,
        )
        return NodeUpdateResponse(
            accepted=True,
            node_id=metadata.node_id,
            endpoint_changed=endpoint_changed,
        )


def create_cluster_authentication(
    *,
    identity: WorkerIdentity,
    node_id: str,
    action: str,
    body: dict[str, Any],
    timestamp_unix_ns: int | None = None,
    nonce: str | None = None,
) -> ClusterRequestAuthentication:
    timestamp = timestamp_unix_ns or time.time_ns()
    selected_nonce = nonce or _url_b64(os.urandom(18))
    unsigned = ClusterRequestAuthentication(
        node_id=node_id,
        timestamp_unix_ns=timestamp,
        nonce=selected_nonce,
        signature="pending",
    )
    payload = canonical_json_bytes(
        {
            "protocol": "swarm-cluster-auth-v1",
            "action": action,
            "node_id": node_id,
            "timestamp_unix_ns": timestamp,
            "nonce": selected_nonce,
            "body": body,
        }
    )
    return unsigned.model_copy(update={"signature": identity.sign(payload)})


class PairingClient:
    """Joining-node transcript verifier and atomic coordinator pinning client."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        identity: WorkerIdentity,
        random_bytes: Callable[[int], bytes] = os.urandom,
    ) -> None:
        self.state = state
        self.identity = identity
        self.random_bytes = random_bytes

    def _transcript(
        self,
        invitation: PairingInvitation,
        hello: PairingHello,
        challenge: PairingChallenge,
    ) -> bytes:
        return canonical_json_bytes(
            {
                "protocol": PAIRING_PROTOCOL_LABEL.decode("ascii"),
                "coordinator_endpoint": invitation.coordinator_endpoint,
                "session_id": invitation.session_id,
                "expires_at_unix_ns": challenge.expires_at_unix_ns,
                "hello": hello.model_dump(mode="json"),
                "coordinator_public_key": challenge.coordinator_public_key,
                "coordinator_fingerprint": challenge.coordinator_fingerprint,
                "coordinator_ephemeral_public_key": challenge.coordinator_ephemeral_public_key,
                "server_nonce": challenge.server_nonce,
            }
        )

    async def join(
        self,
        pairing_uri: str,
        *,
        node_metadata: NodeMetadata,
        hello_rpc: HelloRpc,
        complete_rpc: CompleteRpc,
    ) -> PairingResult:
        invitation = PairingInvitation.parse(pairing_uri)
        node_metadata.verify_identity_binding()
        if node_metadata.public_key != self.identity.public_key_b64:
            raise PairingError("node metadata does not match the durable node identity")
        ephemeral_private = X25519PrivateKey.from_private_bytes(self.random_bytes(32))
        hello = PairingHello(
            session_id=invitation.session_id,
            node_id=node_metadata.node_id,
            node_public_key=node_metadata.public_key,
            node_fingerprint=node_metadata.fingerprint,
            node_ephemeral_public_key=_b64(_x25519_public_key(ephemeral_private)),
            client_nonce=_b64(self.random_bytes(PAIRING_NONCE_BYTES)),
            agent_version=node_metadata.agent_version,
            runtime_version=node_metadata.runtime_version,
            build_id=node_metadata.build_id,
            product_protocol_major=node_metadata.product_protocol_major,
            product_protocol_minor=node_metadata.product_protocol_minor,
            artifact_format_versions=node_metadata.artifact_format_versions,
        )
        challenge = await hello_rpc(hello)
        if challenge.session_id != invitation.session_id:
            raise PairingError("coordinator challenge uses a different session")
        if not hmac.compare_digest(
            _unb64(challenge.coordinator_ephemeral_public_key),
            invitation.coordinator_ephemeral_public_key,
        ):
            raise PairingError("coordinator ephemeral key does not match the invitation")
        if (
            public_key_fingerprint(challenge.coordinator_public_key)
            != challenge.coordinator_fingerprint
        ):
            raise PairingError("coordinator identity fingerprint does not match its public key")
        transcript = self._transcript(invitation, hello, challenge)
        transcript_hash = hashlib.sha256(transcript).hexdigest()
        shared_secret = ephemeral_private.exchange(
            X25519PublicKey.from_public_bytes(invitation.coordinator_ephemeral_public_key)
        )
        key = _derive_session_key(
            shared_secret=shared_secret,
            pairing_secret=invitation.pairing_secret,
            transcript=transcript,
        )
        raw_challenge = _decrypt_json(
            key,
            challenge.encrypted_payload,
            nonce=challenge.encryption_nonce,
            aad=_aad(transcript, phase="challenge"),
        )
        try:
            challenge_payload = PairingChallengePayload.model_validate(raw_challenge)
        except ValueError as exc:
            raise PairingError("coordinator challenge payload is invalid") from exc
        expected_coordinator_proof = _proof(key, b"coordinator-challenge", transcript_hash)
        if not hmac.compare_digest(challenge_payload.transcript_hash, transcript_hash):
            raise PairingError("coordinator challenge transcript does not match")
        if not _proof_matches(
            challenge_payload.coordinator_secret_proof,
            expected_coordinator_proof,
        ):
            raise PairingError("coordinator pairing-secret proof is invalid")
        try:
            verify_signature(
                challenge.coordinator_public_key,
                _signature_payload(
                    role="coordinator-challenge",
                    transcript_hash=transcript_hash,
                    challenge_or_binding=challenge_payload.challenge,
                    secret_proof=challenge_payload.coordinator_secret_proof,
                ),
                challenge_payload.coordinator_signature,
            )
        except IntegrityError as exc:
            raise PairingError("coordinator identity signature is invalid") from exc
        node_proof = _proof(key, b"node-completion", transcript_hash)
        node_signature = self.identity.sign(
            _signature_payload(
                role="node-completion",
                transcript_hash=transcript_hash,
                challenge_or_binding=challenge_payload.challenge,
                secret_proof=node_proof,
            )
        )
        completion_payload = PairingNodeCompletionPayload(
            transcript_hash=transcript_hash,
            challenge=challenge_payload.challenge,
            node_secret_proof=node_proof,
            node_signature=node_signature,
            node_metadata=node_metadata,
        )
        completion_nonce = self.random_bytes(PAIRING_AES_NONCE_BYTES)
        request = PairingCompleteRequest(
            session_id=invitation.session_id,
            hello_hash=hashlib.sha256(
                canonical_json_bytes(hello.model_dump(mode="json"))
            ).hexdigest(),
            encryption_nonce=_b64(completion_nonce),
            encrypted_payload=_encrypt_json(
                key,
                completion_payload.model_dump(mode="json"),
                nonce=completion_nonce,
                aad=_aad(transcript, phase="node-completion"),
            ),
        )
        response = await complete_rpc(request)
        if response.session_id != invitation.session_id or not response.consumed:
            raise PairingError("coordinator did not consume the pairing session")
        raw_response = _decrypt_json(
            key,
            response.encrypted_payload,
            nonce=response.encryption_nonce,
            aad=_aad(transcript, phase="coordinator-completion"),
        )
        try:
            result = PairingCoordinatorCompletionPayload.model_validate(raw_response)
        except ValueError as exc:
            raise PairingError("coordinator completion payload is invalid") from exc
        if not hmac.compare_digest(result.transcript_hash, transcript_hash):
            raise PairingError("coordinator completion transcript does not match")
        result.cluster.verify_identity_binding()
        result.membership.verify_identity_bindings()
        if (
            result.cluster.coordinator_public_key != challenge.coordinator_public_key
            or result.cluster.coordinator_fingerprint != challenge.coordinator_fingerprint
        ):
            raise PairingError("cluster metadata changed the authenticated coordinator identity")
        if (
            result.membership.node_id != node_metadata.node_id
            or result.membership.node_fingerprint != node_metadata.fingerprint
        ):
            raise PairingError("coordinator completion returned another node membership")
        issued_metadata = result.node_metadata or node_metadata
        secure_cluster = (
            result.cluster.security_classification == SECURE_WAN_SECURITY_CLASSIFICATION
        )
        if secure_cluster and issued_metadata.tls_certificate_pem is None:
            raise PairingError("coordinator completion did not provision node TLS credentials")
        issued_metadata.verify_identity_binding()
        if (
            issued_metadata.node_id != node_metadata.node_id
            or issued_metadata.public_key != self.identity.public_key_b64
        ):
            raise PairingError("coordinator TLS certificate belongs to another node")
        ca_certificate = result.cluster.coordinator_certificate_pem
        if secure_cluster and ca_certificate is None:
            raise PairingError("cluster completion has no pinned TLS certificate")
        if ca_certificate is not None and issued_metadata.tls_certificate_pem is not None:
            try:
                validate_certificate_binding(
                    issued_metadata.tls_certificate_pem,
                    ca_certificate_pem=ca_certificate,
                    cluster_id=result.cluster.cluster_id,
                    role="worker",
                    expected_identity_fingerprint=self.identity.public_key_fingerprint,
                )
            except IntegrityError as exc:
                raise PairingError("coordinator issued an invalid node TLS certificate") from exc
        binding = hashlib.sha256(
            canonical_json_bytes(
                {
                    "cluster": result.cluster.model_dump(mode="json"),
                    "membership": result.membership.model_dump(mode="json"),
                    "node_metadata": issued_metadata.model_dump(mode="json"),
                }
            )
        ).hexdigest()
        expected_final_proof = _proof(key, b"coordinator-completion", transcript_hash)
        if not _proof_matches(result.coordinator_secret_proof, expected_final_proof):
            raise PairingError("coordinator final pairing-secret proof is invalid")
        try:
            verify_signature(
                challenge.coordinator_public_key,
                _signature_payload(
                    role="coordinator-completion",
                    transcript_hash=transcript_hash,
                    challenge_or_binding=binding,
                    secret_proof=result.coordinator_secret_proof,
                ),
                result.coordinator_signature,
            )
        except IntegrityError as exc:
            raise PairingError("coordinator final identity signature is invalid") from exc
        # Pin only after every encrypted proof, identity binding, and versioned
        # document has validated successfully.
        self.state.save_cluster(result.cluster)
        self.state.save_membership(result.membership)
        if ca_certificate is not None and issued_metadata.tls_certificate_pem is not None:
            self.state.materialize_node_tls(
                self.identity,
                certificate_pem=issued_metadata.tls_certificate_pem,
                ca_certificate_pem=ca_certificate,
            )
        self.state.save_node(issued_metadata)
        return PairingResult(
            cluster=result.cluster,
            membership=result.membership,
            node_metadata=issued_metadata,
        )


__all__ = [
    "LEGACY_PAIRING_URI_SCHEME",
    "PAIRING_DEFAULT_TTL_SECONDS",
    "PAIRING_URI_SCHEME",
    "PairingClient",
    "PairingInvitation",
    "PairingManager",
    "create_cluster_authentication",
]
