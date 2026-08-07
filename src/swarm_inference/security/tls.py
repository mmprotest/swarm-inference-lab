"""TLS 1.3 material bound to durable Swarm Ed25519 node identities.

The coordinator identity is the cluster certificate authority. Pairing carries
only a node TLS public key and signed public certificate; each node retains a
separate, independently rotatable P-256 private key locally. Legacy derived
keys remain readable for state migration, and private key material never
crosses the network.
"""

from __future__ import annotations

import base64
import hashlib
import os
import ssl
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import grpc
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

from swarm_inference.exceptions import IntegrityError, TransportError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.host import is_loopback_host, split_endpoint
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity

TLS_CERTIFICATE_LIFETIME_DAYS = 90
TLS_RENEWAL_WINDOW_DAYS = 14
COORDINATOR_TLS_NAME = "coordinator.swarm"
WORKER_TLS_NAME = "worker.swarm"
_ROLE_OID = ObjectIdentifier("1.3.6.1.4.1.57264.1.1")
_CLUSTER_OID = ObjectIdentifier("1.3.6.1.4.1.57264.1.2")
_IDENTITY_OID = ObjectIdentifier("1.3.6.1.4.1.57264.1.3")
_IDENTITY_BINDING_OID = ObjectIdentifier("1.3.6.1.4.1.57264.1.4")
_P256_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
)

CertificateRole = Literal["coordinator", "worker"]


def worker_tls_name(identity_fingerprint: str) -> str:
    """Return an address-independent TLS name pinned to one durable worker."""

    normalized = identity_fingerprint.removeprefix("sha256:").lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("worker TLS identity fingerprint must be SHA-256 hex")
    return f"node-{normalized}.worker.swarm"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _certificate_pem(certificate: x509.Certificate) -> str:
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _unrecognized_extension_bytes(
    certificate: x509.Certificate,
    oid: ObjectIdentifier,
) -> bytes:
    extension = certificate.extensions.get_extension_for_oid(oid).value
    if not isinstance(extension, x509.UnrecognizedExtension):
        raise IntegrityError("TLS certificate has an invalid Swarm identity extension")
    return extension.value


def _identity_tls_private_key(identity: WorkerIdentity) -> ec.EllipticCurvePrivateKey:
    raw = identity.private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    scalar = int.from_bytes(hashlib.sha256(b"swarm-tls-p256-v1\0" + raw).digest(), "big")
    return ec.derive_private_key((scalar % (_P256_ORDER - 1)) + 1, ec.SECP256R1())


def identity_private_key_pem(identity: WorkerIdentity) -> bytes:
    """Return the deterministic legacy TLS key for pre-rotation state migration."""

    return _identity_tls_private_key(identity).private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def identity_tls_public_key_pem(identity: WorkerIdentity) -> str:
    """Return the public key paired with the legacy deterministic TLS key."""

    return (
        _identity_tls_private_key(identity)
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def generate_tls_private_key_pem() -> bytes:
    """Generate an independently rotatable P-256 transport key."""

    return ec.generate_private_key(ec.SECP256R1()).private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def tls_public_key_pem(private_key_pem: bytes) -> str:
    """Return and validate the P-256 public key for local TLS material."""

    try:
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("TLS private key is invalid") from exc
    if not isinstance(private_key, ec.EllipticCurvePrivateKey) or not isinstance(
        private_key.curve, ec.SECP256R1
    ):
        raise IntegrityError("TLS private key must use P-256")
    return (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


def _identity_binding_payload(
    cluster_id: str,
    tls_public_key: ec.EllipticCurvePublicKey,
) -> bytes:
    public_der = tls_public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return b"swarm-tls-identity-binding-v1\0" + cluster_id.encode() + b"\0" + public_der


def create_cluster_ca_certificate(
    identity: CoordinatorIdentity,
    *,
    cluster_id: str,
    now: datetime | None = None,
    lifetime_days: int = TLS_CERTIFICATE_LIFETIME_DAYS,
) -> str:
    if lifetime_days <= 0:
        raise ValueError("TLS certificate lifetime must be positive")
    current = now or _utc_now()
    tls_private_key = _identity_tls_private_key(identity)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"swarm-cluster-{cluster_id}")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(tls_private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(current - timedelta(minutes=5))
        .not_valid_after(current + timedelta(days=lifetime_days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(COORDINATOR_TLS_NAME)]),
            critical=False,
        )
        .add_extension(x509.UnrecognizedExtension(_ROLE_OID, b"coordinator"), False)
        .add_extension(x509.UnrecognizedExtension(_CLUSTER_OID, cluster_id.encode("utf-8")), False)
        .add_extension(
            x509.UnrecognizedExtension(
                _IDENTITY_OID,
                identity.public_key_fingerprint.encode("ascii"),
            ),
            False,
        )
        .add_extension(
            x509.UnrecognizedExtension(
                _IDENTITY_BINDING_OID,
                identity.private_key.sign(
                    _identity_binding_payload(cluster_id, tls_private_key.public_key())
                ),
            ),
            False,
        )
        .sign(tls_private_key, algorithm=hashes.SHA256())
    )
    return _certificate_pem(certificate)


def issue_node_certificate(
    coordinator_identity: CoordinatorIdentity,
    *,
    ca_certificate_pem: str,
    cluster_id: str,
    node_public_key_b64: str,
    node_fingerprint: str,
    node_tls_public_key_pem: str,
    now: datetime | None = None,
    lifetime_days: int = TLS_CERTIFICATE_LIFETIME_DAYS,
) -> str:
    if lifetime_days <= 0:
        raise ValueError("TLS certificate lifetime must be positive")
    ca = x509.load_pem_x509_certificate(ca_certificate_pem.encode("ascii"))
    validate_coordinator_certificate_binding(
        ca_certificate_pem,
        cluster_id=cluster_id,
        coordinator_public_key_b64=coordinator_identity.public_key_b64,
        coordinator_fingerprint=coordinator_identity.public_key_fingerprint,
    )
    try:
        node_identity_public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(node_public_key_b64, validate=True)
        )
    except (ValueError, TypeError) as exc:
        raise IntegrityError("node TLS public key is not valid Ed25519") from exc
    actual_fingerprint = hashlib.sha256(
        node_identity_public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).hexdigest()
    if actual_fingerprint != node_fingerprint:
        raise IntegrityError("node TLS certificate request changed the durable identity")
    try:
        node_tls_public_key = serialization.load_pem_public_key(
            node_tls_public_key_pem.encode("ascii")
        )
    except (TypeError, ValueError) as exc:
        raise IntegrityError("node TLS public key is invalid") from exc
    if not isinstance(node_tls_public_key, ec.EllipticCurvePublicKey) or not isinstance(
        node_tls_public_key.curve, ec.SECP256R1
    ):
        raise IntegrityError("node TLS public key must use P-256")
    current = now or _utc_now()
    subject = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, f"swarm-node-{node_fingerprint[:40]}")]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca.subject)
        .public_key(node_tls_public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(current - timedelta(minutes=5))
        .not_valid_after(current + timedelta(days=lifetime_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(WORKER_TLS_NAME),
                    x509.DNSName(worker_tls_name(node_fingerprint)),
                ]
            ),
            critical=False,
        )
        .add_extension(x509.UnrecognizedExtension(_ROLE_OID, b"worker"), False)
        .add_extension(x509.UnrecognizedExtension(_CLUSTER_OID, cluster_id.encode("utf-8")), False)
        .add_extension(
            x509.UnrecognizedExtension(
                _IDENTITY_OID,
                node_fingerprint.encode("ascii"),
            ),
            False,
        )
        .sign(_identity_tls_private_key(coordinator_identity), algorithm=hashes.SHA256())
    )
    return _certificate_pem(certificate)


def certificate_identity_fingerprint(certificate: x509.Certificate | bytes | str) -> str:
    if isinstance(certificate, str):
        loaded_certificate = x509.load_pem_x509_certificate(certificate.encode("ascii"))
    elif isinstance(certificate, bytes):
        try:
            loaded_certificate = x509.load_pem_x509_certificate(certificate)
        except ValueError:
            loaded_certificate = x509.load_der_x509_certificate(certificate)
    else:
        loaded_certificate = certificate
    try:
        encoded = _unrecognized_extension_bytes(loaded_certificate, _IDENTITY_OID)
        fingerprint = encoded.decode("ascii").lower()
    except (UnicodeError, x509.ExtensionNotFound) as exc:
        raise IntegrityError("TLS certificate lacks a durable identity binding") from exc
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise IntegrityError("TLS certificate identity fingerprint is malformed")
    return fingerprint


def certificate_tls_public_key_pem(certificate_pem: str) -> str:
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    public_key = certificate.public_key()
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        raise IntegrityError("TLS certificate does not contain a supported P-256 public key")
    encoded = public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return encoded.decode("ascii")


def validate_coordinator_certificate_binding(
    certificate_pem: str,
    *,
    cluster_id: str,
    coordinator_public_key_b64: str,
    coordinator_fingerprint: str,
) -> None:
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    if certificate_identity_fingerprint(certificate) != coordinator_fingerprint:
        raise IntegrityError("cluster TLS certificate changed the coordinator identity")
    try:
        certificate_cluster = _unrecognized_extension_bytes(certificate, _CLUSTER_OID)
        binding = _unrecognized_extension_bytes(certificate, _IDENTITY_BINDING_OID)
        identity_public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(coordinator_public_key_b64, validate=True)
        )
        tls_public_key = certificate.public_key()
        if not isinstance(tls_public_key, ec.EllipticCurvePublicKey):
            raise IntegrityError("cluster TLS certificate has an unsupported public key")
        identity_public_key.verify(
            binding,
            _identity_binding_payload(cluster_id, tls_public_key),
        )
    except (InvalidSignature, TypeError, ValueError, x509.ExtensionNotFound) as exc:
        raise IntegrityError("cluster TLS certificate identity binding is invalid") from exc
    if certificate_cluster != cluster_id.encode("utf-8"):
        raise IntegrityError("cluster TLS certificate belongs to another cluster")


def _verify_certificate_signature(
    certificate: x509.Certificate,
    ca: x509.Certificate,
) -> None:
    public_key = ca.public_key()
    if isinstance(public_key, ec.EllipticCurvePublicKey):
        signature_hash_algorithm = certificate.signature_hash_algorithm
        if signature_hash_algorithm is None:
            raise IntegrityError("TLS certificate signature does not identify a hash algorithm")
        public_key.verify(
            certificate.signature,
            certificate.tbs_certificate_bytes,
            ec.ECDSA(signature_hash_algorithm),
        )
        return
    if isinstance(public_key, Ed25519PublicKey):
        public_key.verify(certificate.signature, certificate.tbs_certificate_bytes)
        return
    raise IntegrityError("cluster TLS CA uses an unsupported signature key")


def certificate_sha256(certificate_pem: str) -> str:
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    return hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest()


def validate_certificate_binding(
    certificate_pem: str,
    *,
    ca_certificate_pem: str,
    cluster_id: str,
    role: CertificateRole,
    expected_identity_fingerprint: str | None = None,
    revoked_fingerprints: frozenset[str] = frozenset(),
    now: datetime | None = None,
) -> str:
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    ca = x509.load_pem_x509_certificate(ca_certificate_pem.encode("ascii"))
    current = now or _utc_now()
    if not certificate.not_valid_before_utc <= current <= certificate.not_valid_after_utc:
        raise IntegrityError("TLS certificate is expired or not yet valid")
    if certificate.issuer != ca.subject:
        raise IntegrityError("TLS certificate issuer is not the pinned cluster CA")
    try:
        _verify_certificate_signature(certificate, ca)
    except Exception as exc:
        raise IntegrityError("TLS certificate signature is invalid") from exc
    try:
        certificate_role = _unrecognized_extension_bytes(certificate, _ROLE_OID)
        certificate_cluster = _unrecognized_extension_bytes(certificate, _CLUSTER_OID)
    except x509.ExtensionNotFound as exc:
        raise IntegrityError("TLS certificate lacks Swarm identity extensions") from exc
    if certificate_role != role.encode("ascii"):
        raise IntegrityError("TLS certificate has the wrong Swarm role")
    if certificate_cluster != cluster_id.encode("utf-8"):
        raise IntegrityError("TLS certificate belongs to another cluster")
    fingerprint = certificate_identity_fingerprint(certificate)
    if expected_identity_fingerprint is not None and fingerprint != expected_identity_fingerprint:
        raise IntegrityError("TLS certificate does not match the pinned node identity")
    if fingerprint in revoked_fingerprints:
        raise IntegrityError("TLS certificate identity has been revoked")
    return fingerprint


def certificate_needs_rotation(
    certificate_pem: str,
    *,
    now: datetime | None = None,
    renewal_window_days: int = TLS_RENEWAL_WINDOW_DAYS,
) -> bool:
    certificate = x509.load_pem_x509_certificate(certificate_pem.encode("ascii"))
    return certificate.not_valid_after_utc <= (now or _utc_now()) + timedelta(
        days=renewal_window_days
    )


def _atomic_write(path: Path, payload: bytes, *, private: bool) -> None:
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
        replace_atomically(temporary, path)
        if private:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def materialize_tls_identity(
    *,
    identity: WorkerIdentity,
    certificate_pem: str,
    certificate_path: Path,
    private_key_path: Path,
    expected_fingerprint: str | None = None,
    private_key_pem: bytes | None = None,
) -> None:
    fingerprint = certificate_identity_fingerprint(certificate_pem)
    expected = expected_fingerprint or identity.public_key_fingerprint
    if fingerprint != expected or expected != identity.public_key_fingerprint:
        raise IntegrityError("TLS certificate/private identity binding does not match")
    local_private_key = private_key_pem or identity_private_key_pem(identity)
    if certificate_tls_public_key_pem(certificate_pem) != tls_public_key_pem(local_private_key):
        raise IntegrityError("TLS certificate does not match the local private key")
    _atomic_write(certificate_path, certificate_pem.encode("ascii"), private=False)
    _atomic_write(private_key_path, local_private_key, private=True)


@dataclass(frozen=True, slots=True)
class TlsCertificatePaths:
    certificate: Path
    private_key: Path
    ca_certificate: Path

    def verify_files(self) -> None:
        for path in (self.certificate, self.private_key, self.ca_certificate):
            if not path.is_file():
                raise FileNotFoundError(f"TLS material is unavailable: {path}")


@dataclass(frozen=True, slots=True)
class TlsClientConfig:
    material: TlsCertificatePaths
    expected_server_name: str
    expected_peer_fingerprint: str | None = None

    def grpc_credentials(self) -> grpc.ChannelCredentials:
        self.material.verify_files()
        return grpc.ssl_channel_credentials(
            root_certificates=self.material.ca_certificate.read_bytes(),
            private_key=self.material.private_key.read_bytes(),
            certificate_chain=self.material.certificate.read_bytes(),
        )

    def grpc_options(self) -> tuple[tuple[str, str], ...]:
        return (
            ("grpc.ssl_target_name_override", self.expected_server_name),
            ("grpc.default_authority", self.expected_server_name),
        )

    def ssl_context(self) -> ssl.SSLContext:
        self.material.verify_files()
        context = ssl.create_default_context(
            ssl.Purpose.SERVER_AUTH,
            cafile=str(self.material.ca_certificate),
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=str(self.material.certificate),
            keyfile=str(self.material.private_key),
        )
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context

    def validate_peer_der(self, peer_der: bytes | None) -> str:
        if peer_der is None:
            raise TransportError("TLS server did not present a certificate")
        fingerprint = certificate_identity_fingerprint(peer_der)
        if (
            self.expected_peer_fingerprint is not None
            and fingerprint != self.expected_peer_fingerprint
        ):
            raise TransportError("TLS server certificate does not match the pinned identity")
        return fingerprint


@dataclass(frozen=True, slots=True)
class TlsBootstrapClientConfig:
    """Server-authenticated TLS used only while a node is being paired."""

    ca_certificate: Path
    expected_server_name: str = COORDINATOR_TLS_NAME

    def grpc_credentials(self) -> grpc.ChannelCredentials:
        if not self.ca_certificate.is_file():
            raise FileNotFoundError(
                f"pairing TLS certificate is unavailable: {self.ca_certificate}"
            )
        return grpc.ssl_channel_credentials(root_certificates=self.ca_certificate.read_bytes())

    def grpc_options(self) -> tuple[tuple[str, str], ...]:
        return (
            ("grpc.ssl_target_name_override", self.expected_server_name),
            ("grpc.default_authority", self.expected_server_name),
        )


@dataclass(frozen=True, slots=True)
class TlsServerConfig:
    material: TlsCertificatePaths
    require_client_certificate: bool = True
    allowed_peer_fingerprints: frozenset[str] = frozenset()
    revoked_peer_fingerprints: frozenset[str] = frozenset()

    def grpc_credentials(self) -> grpc.ServerCredentials:
        self.material.verify_files()
        return grpc.ssl_server_credentials(
            ((self.material.private_key.read_bytes(), self.material.certificate.read_bytes()),),
            root_certificates=self.material.ca_certificate.read_bytes(),
            require_client_auth=self.require_client_certificate,
        )

    def ssl_context(self) -> ssl.SSLContext:
        self.material.verify_files()
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        context.load_cert_chain(
            certfile=str(self.material.certificate),
            keyfile=str(self.material.private_key),
        )
        context.load_verify_locations(cafile=str(self.material.ca_certificate))
        context.verify_mode = (
            ssl.CERT_REQUIRED if self.require_client_certificate else ssl.CERT_NONE
        )
        return context

    def validate_peer_der(self, peer_der: bytes | None) -> str | None:
        if peer_der is None:
            if self.require_client_certificate:
                raise TransportError("TLS peer did not present a client certificate")
            return None
        fingerprint = certificate_identity_fingerprint(peer_der)
        if self.allowed_peer_fingerprints and fingerprint not in self.allowed_peer_fingerprints:
            raise TransportError("TLS peer identity is not trusted for this transport")
        if fingerprint in self.revoked_peer_fingerprints:
            raise TransportError("TLS peer identity is revoked")
        return fingerprint


def require_tls_for_endpoint(
    endpoint: str,
    *,
    tls_configured: bool,
    allow_plaintext_loopback: bool,
    transport_name: str,
) -> None:
    host, _ = split_endpoint(endpoint)
    if tls_configured:
        return
    if allow_plaintext_loopback and is_loopback_host(host):
        return
    raise TransportError(
        f"{transport_name} refuses unauthenticated plaintext for non-loopback endpoint "
        f"{endpoint}; configure cluster TLS material"
    )


__all__ = [
    "COORDINATOR_TLS_NAME",
    "TLS_CERTIFICATE_LIFETIME_DAYS",
    "TLS_RENEWAL_WINDOW_DAYS",
    "WORKER_TLS_NAME",
    "TlsBootstrapClientConfig",
    "TlsCertificatePaths",
    "TlsClientConfig",
    "TlsServerConfig",
    "certificate_identity_fingerprint",
    "certificate_needs_rotation",
    "certificate_sha256",
    "certificate_tls_public_key_pem",
    "create_cluster_ca_certificate",
    "generate_tls_private_key_pem",
    "identity_private_key_pem",
    "identity_tls_public_key_pem",
    "issue_node_certificate",
    "materialize_tls_identity",
    "require_tls_for_endpoint",
    "tls_public_key_pem",
    "validate_certificate_binding",
    "validate_coordinator_certificate_binding",
    "worker_tls_name",
]
