"""Canonical signing helpers for registration, heartbeat, and results."""

from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from swarm_inference.exceptions import IntegrityError


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def verify_signature(public_key_b64: str, payload: bytes, signature_b64: str) -> None:
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64, validate=True))
        signature = base64.b64decode(signature_b64, validate=True)
        key.verify(signature, payload)
    except (ValueError, InvalidSignature) as exc:
        raise IntegrityError("invalid Ed25519 signature") from exc
