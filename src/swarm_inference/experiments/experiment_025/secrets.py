"""Create ephemeral E025 transport material outside retained public artifacts."""

from __future__ import annotations

import base64
import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .io import atomic_write_json, sha256_file, utc_now


def _private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def create_transport_material(directory: Path, run_id: str) -> dict[str, Any]:
    destination = directory.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"refusing to replace E025 private material: {destination}")
    destination.mkdir(parents=True)
    credential_path = destination / "run-credential.bin"
    certificate_path = destination / "server.crt"
    key_path = destination / "server.key"
    _private(credential_path, os.urandom(48))
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"e025-{run_id}")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(private_key, hashes.SHA256())
    )
    _private(
        key_path,
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    public_receipt = {
        "schema_version": "experiment-025-transport-material-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS",
        "run_id": run_id,
        "certificate_sha256": sha256_file(certificate_path),
        "certificate_subject": certificate.subject.rfc4514_string(),
        "certificate_not_before_utc": certificate.not_valid_before_utc.isoformat(),
        "certificate_not_after_utc": certificate.not_valid_after_utc.isoformat(),
        "credential_bytes": credential_path.stat().st_size,
        "private_key_persisted_to_public_artifacts": False,
        "credential_persisted_to_public_artifacts": False,
        "vast_api_key_persisted": False,
    }
    atomic_write_json(destination / "private-material-receipt.json", public_receipt)
    return {
        **public_receipt,
        "directory": str(destination),
        "credential_path": credential_path,
        "certificate_path": certificate_path,
        "private_key_path": key_path,
    }


def load_transport_material(directory: Path, run_id: str) -> dict[str, Any]:
    destination = directory.expanduser().resolve()
    credential_path = destination / "run-credential.bin"
    certificate_path = destination / "server.crt"
    key_path = destination / "server.key"
    receipt_path = destination / "private-material-receipt.json"
    if not all(
        path.is_file()
        for path in (credential_path, certificate_path, key_path, receipt_path)
    ):
        raise ValueError("E025 private transport material is incomplete")
    import json

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("run_id") != run_id or receipt.get("status") != "PASS":
        raise ValueError("E025 private transport receipt has a different run identity")
    if receipt.get("certificate_sha256") != sha256_file(certificate_path):
        raise ValueError("E025 private certificate differs from its public receipt")
    return {
        **receipt,
        "directory": str(destination),
        "credential_path": credential_path,
        "certificate_path": certificate_path,
        "private_key_path": key_path,
    }


def vast_environment_options(
    *,
    material: dict[str, Any],
    worker_id: str,
    run_id: str,
    image_digest: str,
    instance_id: int | None,
    machine_id: int,
    maximum_context: int,
    expert_endpoints: list[dict[str, Any]] | None = None,
) -> str:
    def encoded(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    values = {
        "E025_WORKER_ID": worker_id,
        "E025_RUN_ID": run_id,
        "E025_IMAGE_DIGEST": image_digest,
        "E025_INSTANCE_ID": "pending" if instance_id is None else str(instance_id),
        "E025_MACHINE_ID": str(machine_id),
        "E025_MAXIMUM_CONTEXT": str(maximum_context),
        "E025_RUN_CREDENTIAL_B64": encoded(Path(material["credential_path"])),
        "E025_TLS_CERT_B64": encoded(Path(material["certificate_path"])),
        "E025_TLS_KEY_B64": encoded(Path(material["private_key_path"])),
    }
    if expert_endpoints is not None:
        import json

        values["E025_EXPERT_ENDPOINTS_B64"] = base64.b64encode(
            json.dumps(expert_endpoints, sort_keys=True, separators=(",", ":")).encode()
        ).decode("ascii")
    environment = " ".join(f"-e {key}={value}" for key, value in values.items())
    return f"{environment} -p 42525:42525/tcp"


def vast_environment_options_for_workers(
    *,
    material: dict[str, Any],
    worker_specs: list[dict[str, Any]],
    run_id: str,
    image_digest: str,
    instance_id: int | None,
    machine_id: int,
) -> str:
    """Encode a bounded one-process-per-GPU instance without logging secrets."""

    if not worker_specs or len(worker_specs) > 8:
        raise ValueError("E025 instance must own one to eight physical GPU workers")

    def encoded(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("ascii")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(worker_specs):
        port = int(row.get("port", 42525 + index))
        gpu_slot = int(row.get("gpu_slot", index))
        if port != 42525 + index or gpu_slot != index:
            raise ValueError("E025 grouped worker ports and GPU slots must be contiguous")
        normalized.append(
            {
                "worker_id": str(row["worker_id"]),
                "gpu_slot": gpu_slot,
                "port": port,
                "maximum_context": int(row.get("maximum_context", 64)),
                "expert_endpoints": row.get("expert_endpoints"),
            }
        )
    values = {
        "E025_RUN_ID": run_id,
        "E025_IMAGE_DIGEST": image_digest,
        "E025_INSTANCE_ID": "pending" if instance_id is None else str(instance_id),
        "E025_MACHINE_ID": str(machine_id),
        "E025_RUN_CREDENTIAL_B64": encoded(Path(material["credential_path"])),
        "E025_TLS_CERT_B64": encoded(Path(material["certificate_path"])),
        "E025_TLS_KEY_B64": encoded(Path(material["private_key_path"])),
        "E025_WORKER_SPECS_B64": base64.b64encode(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        ).decode("ascii"),
    }
    environment = " ".join(f"-e {key}={value}" for key, value in values.items())
    ports = " ".join(
        f"-p {int(row['port'])}:{int(row['port'])}/tcp" for row in normalized
    )
    return f"{environment} {ports}"


__all__ = [
    "create_transport_material",
    "load_transport_material",
    "vast_environment_options",
    "vast_environment_options_for_workers",
]
