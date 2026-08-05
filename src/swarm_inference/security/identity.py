"""Versioned Ed25519 coordinator and worker identities.

The product CLI writes JSON identity documents.  Legacy PEM files remain
readable so existing durable coordinator state and worker installations do not
silently rotate their keys during the productization transition.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Literal, Self, cast
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically

IDENTITY_DOCUMENT_TYPE = "swarm-ed25519-identity"
IDENTITY_FORMAT_VERSION = 1
PUBLIC_KEY_ENCODING = "base64-raw-ed25519"
PRIVATE_KEY_ENCODING = "base64-raw-ed25519-private"

IdentityKind = Literal["coordinator", "worker"]


@dataclass(frozen=True, slots=True)
class IdentityMetadata:
    """Public, non-secret metadata safe for CLI and audit output."""

    identity_kind: IdentityKind
    fingerprint: str
    public_key: str
    public_key_encoding: str
    created_at: str | None
    format_version: int | str
    path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _atomic_private_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and path.exists():
            raise FileExistsError(f"identity already exists: {path}")
        replace_atomically(temporary, path)
        # POSIX honours the mode supplied to os.open.  chmod is also harmless on
        # Windows and removes broad writable bits where the filesystem supports it.
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _decode_json_document(path: Path, payload: bytes) -> dict[str, object]:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid identity document {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise IntegrityError(f"identity document must be a JSON object: {path}")
    return cast(dict[str, object], raw)


def _decode_base64(value: object, *, field: str, expected_length: int) -> bytes:
    if not isinstance(value, str):
        raise IntegrityError(f"identity {field} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise IntegrityError(f"identity {field} is not valid base64") from exc
    if len(decoded) != expected_length:
        raise IntegrityError(
            f"identity {field} must contain exactly {expected_length} decoded bytes"
        )
    return decoded


def _identity_kind(value: object) -> IdentityKind:
    if value == "coordinator":
        return "coordinator"
    if value == "worker":
        return "worker"
    raise IntegrityError(f"unsupported identity kind: {value!r}")


@dataclass(slots=True)
class WorkerIdentity:
    private_key: Ed25519PrivateKey

    identity_kind: ClassVar[IdentityKind] = "worker"

    @classmethod
    def generate(cls) -> Self:
        return cls(private_key=Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a canonical JSON identity or a legacy unencrypted PEM key."""

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"identity file does not exist: {resolved}")
        try:
            payload = resolved.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"cannot read identity file {resolved}: {exc}") from exc
        if payload.lstrip().startswith(b"{"):
            identity, _ = _identity_from_document(
                resolved, payload, expected_kind=cls.identity_kind
            )
            if cls is CoordinatorIdentity and not isinstance(identity, CoordinatorIdentity):
                raise IntegrityError(f"identity kind is not coordinator: {resolved}")
            if cls is WorkerIdentity and isinstance(identity, CoordinatorIdentity):
                raise IntegrityError(f"identity kind is not worker: {resolved}")
            return cast(Self, identity)
        try:
            key = serialization.load_pem_private_key(payload, password=None)
        except (ValueError, TypeError) as exc:
            raise IntegrityError(f"invalid legacy identity file {resolved}: {exc}") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise IntegrityError(f"identity is not Ed25519: {resolved}")
        return cls(private_key=key)

    @classmethod
    def load_or_create(cls, path: str | Path) -> Self:
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file():
            return cls.load(resolved)
        if resolved.suffix.lower() == ".json":
            identity, _ = create_identity_file(
                resolved,
                kind=cls.identity_kind,
                force=False,
            )
            return cast(Self, identity)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        identity = cls.generate()
        encoded = identity.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        try:
            _atomic_private_write(resolved, encoded, overwrite=False)
        except FileExistsError:
            # Another process may have provisioned the key after the initial
            # existence check.  Load it rather than rotating or overwriting it.
            return cls.load(resolved)
        return identity

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    @property
    def public_key_b64(self) -> str:
        return base64.b64encode(self.public_key_bytes).decode("ascii")

    @property
    def public_key_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_fingerprint(self) -> str:
        """Stable SHA-256 fingerprint used in signed identity records."""

        return hashlib.sha256(self.public_key_bytes).hexdigest()

    def sign(self, payload: bytes) -> str:
        return base64.b64encode(self.private_key.sign(payload)).decode("ascii")


class CoordinatorIdentity(WorkerIdentity):
    """Persistent coordinator signing identity with a distinct trust role."""

    identity_kind: ClassVar[IdentityKind] = "coordinator"


def _identity_document(
    identity: WorkerIdentity,
    *,
    kind: IdentityKind,
    created_at: str,
) -> dict[str, object]:
    private_bytes = identity.private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return {
        "document_type": IDENTITY_DOCUMENT_TYPE,
        "format_version": IDENTITY_FORMAT_VERSION,
        "identity_kind": kind,
        "created_at": created_at,
        "public_key": {
            "algorithm": "Ed25519",
            "encoding": PUBLIC_KEY_ENCODING,
            "value": identity.public_key_b64,
        },
        "private_key": {
            "algorithm": "Ed25519",
            "encoding": PRIVATE_KEY_ENCODING,
            "value": base64.b64encode(private_bytes).decode("ascii"),
        },
        "fingerprint": identity.public_key_fingerprint,
    }


def _identity_from_document(
    path: Path,
    payload: bytes,
    *,
    expected_kind: IdentityKind | None = None,
) -> tuple[WorkerIdentity, IdentityMetadata]:
    raw = _decode_json_document(path, payload)
    if raw.get("document_type") != IDENTITY_DOCUMENT_TYPE:
        raise IntegrityError(f"unsupported identity document type: {raw.get('document_type')!r}")
    version = raw.get("format_version")
    if version != IDENTITY_FORMAT_VERSION:
        raise IntegrityError(f"unsupported identity format version: {version!r}")
    typed_kind = _identity_kind(raw.get("identity_kind"))
    if expected_kind is not None and typed_kind != expected_kind:
        raise IntegrityError(
            f"identity kind {typed_kind!r} does not match required kind {expected_kind!r}"
        )
    created_at = raw.get("created_at")
    if not isinstance(created_at, str):
        raise IntegrityError("identity created_at must be an ISO-8601 timestamp")
    try:
        parsed_created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntegrityError("identity created_at is not a valid ISO-8601 timestamp") from exc
    if parsed_created.tzinfo is None:
        raise IntegrityError("identity created_at must include a timezone")
    public_record = raw.get("public_key")
    private_record = raw.get("private_key")
    if not isinstance(public_record, dict) or not isinstance(private_record, dict):
        raise IntegrityError("identity document is missing key records")
    if (
        public_record.get("algorithm") != "Ed25519"
        or public_record.get("encoding") != PUBLIC_KEY_ENCODING
        or private_record.get("algorithm") != "Ed25519"
        or private_record.get("encoding") != PRIVATE_KEY_ENCODING
    ):
        raise IntegrityError("identity document uses an unsupported key encoding")
    public_bytes = _decode_base64(
        public_record.get("value"), field="public key", expected_length=32
    )
    private_bytes = _decode_base64(
        private_record.get("value"), field="private key", expected_length=32
    )
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        Ed25519PublicKey.from_public_bytes(public_bytes)
    except ValueError as exc:
        raise IntegrityError("identity document contains an invalid Ed25519 key") from exc
    identity_type = CoordinatorIdentity if typed_kind == "coordinator" else WorkerIdentity
    identity = identity_type(private_key=private_key)
    if identity.public_key_bytes != public_bytes:
        raise IntegrityError("identity public key does not match its private key")
    fingerprint = raw.get("fingerprint")
    if fingerprint != identity.public_key_fingerprint:
        raise IntegrityError("identity fingerprint does not match its public key")
    metadata = IdentityMetadata(
        identity_kind=typed_kind,
        fingerprint=identity.public_key_fingerprint,
        public_key=identity.public_key_b64,
        public_key_encoding=PUBLIC_KEY_ENCODING,
        created_at=created_at,
        format_version=IDENTITY_FORMAT_VERSION,
        path=str(path),
    )
    return identity, metadata


def create_identity_file(
    path: str | Path,
    *,
    kind: IdentityKind,
    force: bool = False,
) -> tuple[WorkerIdentity, IdentityMetadata]:
    """Create one versioned identity document without disclosing its private key."""

    if kind not in {"coordinator", "worker"}:
        raise ValueError(f"unsupported identity kind {kind!r}")
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"identity already exists: {resolved}")
    identity: WorkerIdentity = (
        CoordinatorIdentity.generate() if kind == "coordinator" else WorkerIdentity.generate()
    )
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    document = _identity_document(identity, kind=kind, created_at=created_at)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_private_write(resolved, payload, overwrite=force)
    metadata = IdentityMetadata(
        identity_kind=kind,
        fingerprint=identity.public_key_fingerprint,
        public_key=identity.public_key_b64,
        public_key_encoding=PUBLIC_KEY_ENCODING,
        created_at=created_at,
        format_version=IDENTITY_FORMAT_VERSION,
        path=str(resolved),
    )
    return identity, metadata


def inspect_identity_file(
    path: str | Path,
    *,
    expected_kind: IdentityKind | None = None,
) -> IdentityMetadata:
    """Validate an identity file and return only its public metadata."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"identity file does not exist: {resolved}")
    payload = resolved.read_bytes()
    if payload.lstrip().startswith(b"{"):
        _, metadata = _identity_from_document(
            resolved,
            payload,
            expected_kind=expected_kind,
        )
        return metadata
    # Legacy PEM has no embedded kind or creation time.  It remains usable but
    # is reported honestly as an unversioned compatibility identity.
    legacy_type = CoordinatorIdentity if expected_kind == "coordinator" else WorkerIdentity
    identity = legacy_type.load(resolved)
    kind = expected_kind or legacy_type.identity_kind
    return IdentityMetadata(
        identity_kind=kind,
        fingerprint=identity.public_key_fingerprint,
        public_key=identity.public_key_b64,
        public_key_encoding="pem-pkcs8-private/base64-raw-ed25519-public",
        created_at=None,
        format_version="legacy-pem-v0",
        path=str(resolved),
    )


def public_key_fingerprint(public_key_b64: str) -> str:
    """Return the canonical fingerprint for a base64 Ed25519 public key."""

    try:
        raw = base64.b64decode(public_key_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise IntegrityError("invalid Ed25519 public key") from exc
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "IDENTITY_DOCUMENT_TYPE",
    "IDENTITY_FORMAT_VERSION",
    "PRIVATE_KEY_ENCODING",
    "PUBLIC_KEY_ENCODING",
    "CoordinatorIdentity",
    "IdentityKind",
    "IdentityMetadata",
    "WorkerIdentity",
    "create_identity_file",
    "inspect_identity_file",
    "public_key_fingerprint",
]
