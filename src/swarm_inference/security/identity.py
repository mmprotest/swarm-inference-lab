"""Ed25519 worker identities."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path

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
    def generate(cls) -> WorkerIdentity:
        return cls(private_key=Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(cls, path: str | Path) -> WorkerIdentity:
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
        raw = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return base64.b64encode(raw).decode("ascii")

    def sign(self, payload: bytes) -> str:
        return base64.b64encode(self.private_key.sign(payload)).decode("ascii")
