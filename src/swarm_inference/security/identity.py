"""Ed25519 worker identities."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from swarm_inference.exceptions import IntegrityError


@dataclass(slots=True)
class WorkerIdentity:
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls) -> Self:
        return cls(private_key=Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> Self:
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file():
            try:
                key = serialization.load_pem_private_key(resolved.read_bytes(), password=None)
            except (ValueError, TypeError) as exc:
                raise IntegrityError(f"invalid worker identity file {resolved}: {exc}") from exc
            if not isinstance(key, Ed25519PrivateKey):
                raise IntegrityError(f"worker identity is not Ed25519: {resolved}")
            return cls(private_key=key)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        identity = cls.generate()
        encoded = identity.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(resolved, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
        return identity

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    @property
    def public_key_b64(self) -> str:
        raw = self.public_key_bytes
        return base64.b64encode(raw).decode("ascii")

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
    """Persistent coordinator signing identity.

    Coordinator and worker keys use the same Ed25519 representation, while
    distinct types keep trust configuration and audit output unambiguous.
    """


def public_key_fingerprint(public_key_b64: str) -> str:
    """Return the canonical fingerprint for a base64 Ed25519 public key."""

    try:
        raw = base64.b64decode(public_key_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise IntegrityError("invalid Ed25519 public key") from exc
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "CoordinatorIdentity",
    "WorkerIdentity",
    "public_key_fingerprint",
]
