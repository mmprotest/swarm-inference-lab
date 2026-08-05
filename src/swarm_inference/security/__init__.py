"""Worker identity, signatures, reputation, and audit evidence."""

from .identity import (
    CoordinatorIdentity,
    IdentityMetadata,
    WorkerIdentity,
    create_identity_file,
    inspect_identity_file,
    public_key_fingerprint,
)
from .reputation import ReputationBook
from .signatures import canonical_json_bytes, verify_signature
from .trust_store import WorkerTrustStore, normalize_fingerprint

__all__ = [
    "CoordinatorIdentity",
    "IdentityMetadata",
    "ReputationBook",
    "WorkerIdentity",
    "WorkerTrustStore",
    "canonical_json_bytes",
    "create_identity_file",
    "inspect_identity_file",
    "normalize_fingerprint",
    "public_key_fingerprint",
    "verify_signature",
]
