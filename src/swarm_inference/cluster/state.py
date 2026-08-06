"""Atomic, strict persistence for cluster security and mutable node state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from uuid import uuid4

from pydantic import ValidationError

from swarm_inference.cluster.models import (
    NODE_REGISTRY_SCHEMA_VERSION,
    ArtifactCacheDocument,
    BackendValidationRecord,
    ClusterAuditEvent,
    ClusterMetadata,
    MembershipRegistryDocument,
    NetworkLinkMeasurement,
    NetworkMeasurementRegistryDocument,
    NodeConfiguration,
    NodeMembership,
    NodeMetadata,
    NodeRegistryDocument,
    NodeRevocation,
    NodeRuntimeMetadata,
    PairingSession,
    PairingSessionRegistryDocument,
    RevocationRegistryDocument,
    aggregate_validation_status,
    migrate_legacy_node_metadata,
)
from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.platforms import default_state_directory
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity

DocumentT = TypeVar("DocumentT", bound=StrictModel)
Migration = Callable[[dict[str, Any]], dict[str, Any]]
PairingFileProtection = Literal["posix-0600", "windows-user-acl", "user-scoped-best-effort"]


@dataclass(frozen=True, slots=True)
class ClusterStatePaths:
    root: Path
    security: Path
    runtime: Path
    logs: Path
    artifacts: Path
    downloads: Path
    cluster_metadata: Path
    nodes: Path
    memberships: Path
    revocations: Path
    pairing_sessions: Path
    pairing_invitations: Path
    network_measurements: Path
    artifact_cache: Path
    coordinator_identity: Path
    node_identity: Path
    node_configuration: Path
    audit_log: Path
    node_runtime_directory: Path
    coordinator_runtime_directory: Path

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> ClusterStatePaths:
        resolved = Path(root or default_state_directory()).expanduser().resolve()
        security = resolved / "security"
        runtime = resolved / "runtime"
        logs = resolved / "logs"
        artifacts = resolved / "artifacts"
        downloads = resolved / "downloads"
        return cls(
            root=resolved,
            security=security,
            runtime=runtime,
            logs=logs,
            artifacts=artifacts,
            downloads=downloads,
            cluster_metadata=security / "cluster.json",
            nodes=security / "nodes.json",
            memberships=security / "memberships.json",
            revocations=security / "revocations.json",
            pairing_sessions=runtime / "pairing-sessions.json",
            pairing_invitations=runtime / "pairing-invitations",
            network_measurements=runtime / "network-measurements.json",
            artifact_cache=runtime / "artifact-cache.json",
            coordinator_identity=security / "coordinator-identity.json",
            node_identity=security / "node-identity.json",
            node_configuration=security / "node-configuration.json",
            audit_log=logs / "cluster-audit.jsonl",
            node_runtime_directory=runtime / "nodes",
            coordinator_runtime_directory=runtime / "coordinator",
        )


def _version_zero_to_one(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    migrated["schema_version"] = 1
    migrated.setdefault("document_version", 1)
    return migrated


_MIGRATIONS: dict[int, Migration] = {0: _version_zero_to_one}
_FORBIDDEN_AUDIT_TERMS = (
    "private_key",
    "pairing_secret",
    "session_key",
    "aes_key",
    "raw_proof",
    "prompt",
)


def _version_one_to_two_node_registry(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(raw)
    nodes = migrated.get("nodes", [])
    if not isinstance(nodes, list):
        raise IntegrityError("legacy node registry has a non-list nodes field")
    migrated["nodes"] = [
        migrate_legacy_node_metadata(cast(dict[str, Any], item)) if isinstance(item, dict) else item
        for item in nodes
    ]
    migrated["schema_version"] = NODE_REGISTRY_SCHEMA_VERSION
    migrated["document_version"] = 2
    return migrated


def _migrate(
    raw: dict[str, Any],
    *,
    path: Path,
    model: type[StrictModel],
) -> tuple[dict[str, Any], bool]:
    if "schema_version" not in raw:
        raise IntegrityError(f"state document has no schema_version: {path}")
    value = raw["schema_version"]
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid schema_version in {path}: {value!r}") from exc
    changed = False
    target_version = NODE_REGISTRY_SCHEMA_VERSION if model is NodeRegistryDocument else 1
    while version < target_version:
        migration = (
            _version_one_to_two_node_registry
            if model is NodeRegistryDocument and version == 1
            else _MIGRATIONS.get(version)
        )
        if migration is None:
            raise IntegrityError(f"no migration from state schema {version} for {path}")
        raw = migration(raw)
        version += 1
        changed = True
    if version != target_version:
        raise IntegrityError(f"unsupported future state schema {version} for {path}")
    return raw, changed


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    private: bool,
    overwrite: bool = True,
) -> None:
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
        if overwrite:
            replace_atomically(temporary, path)
        else:
            # A hard-link publish is atomic and refuses an existing destination
            # on both POSIX and NTFS. The temporary contains the complete,
            # fsync'd payload before it becomes visible at the final path.
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite existing file: {path}") from None
            except OSError as exc:
                raise OSError(
                    f"filesystem cannot atomically publish a no-overwrite invitation file: {path}"
                ) from exc
            temporary.unlink()
        if private:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


class ClusterStateStore:
    """One lock-protected owner for versioned cluster documents."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.paths = ClusterStatePaths.from_root(root)
        self.clock_ns = clock_ns
        self._lock = threading.RLock()
        self.initialize_directories()

    def initialize_directories(self) -> None:
        for directory in (
            self.paths.security,
            self.paths.runtime,
            self.paths.logs,
            self.paths.artifacts,
            self.paths.downloads,
            self.paths.node_runtime_directory,
            self.paths.coordinator_runtime_directory,
            self.paths.pairing_invitations,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        # Some Windows filesystems do not implement POSIX mode bits.  The
        # user-scoped root and private file modes remain the safe fallback.
        with suppress(OSError):
            os.chmod(self.paths.security, stat.S_IRWXU)
        with suppress(OSError):
            os.chmod(self.paths.pairing_invitations, stat.S_IRWXU)

    @staticmethod
    def _validate_pairing_session_id(session_id: str) -> str:
        if (
            not session_id
            or len(session_id) > 128
            or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for character in session_id
            )
        ):
            raise ValueError("pairing session ID is not safe for an invitation filename")
        return session_id

    def default_pairing_invitation_path(self, session_id: str) -> Path:
        safe = self._validate_pairing_session_id(session_id)
        return self.paths.pairing_invitations / f"{safe}.uri"

    @staticmethod
    def _apply_pairing_file_protection(
        path: Path,
    ) -> tuple[PairingFileProtection, str | None]:
        if os.name != "nt":
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            return "posix-0600", None
        try:
            identity = subprocess.run(
                ["whoami.exe", "/user", "/fo", "csv", "/nh"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            limitation = f"Windows token SID lookup was unavailable: {type(exc).__name__}"
        else:
            sid_match = re.search(r"S-\d-(?:\d+-)+\d+", identity.stdout)
            if identity.returncode != 0 or sid_match is None:
                limitation = "Windows token SID was unavailable for explicit ACL hardening"
            else:
                try:
                    completed = subprocess.run(
                        [
                            "icacls.exe",
                            str(path),
                            "/inheritance:r",
                            "/grant:r",
                            f"*{sid_match.group(0)}:(F)",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    limitation = f"Windows ACL hardening was unavailable: {type(exc).__name__}"
                else:
                    if completed.returncode == 0:
                        return "windows-user-acl", None
                    limitation = (
                        "Windows ACL hardening failed; the file remains in a user-scoped path"
                    )
        with suppress(OSError):
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        return "user-scoped-best-effort", limitation

    def write_pairing_invitation(
        self,
        *,
        session_id: str,
        pairing_uri: str,
        output_path: Path | None = None,
        force: bool = False,
    ) -> tuple[Path, PairingFileProtection, str | None]:
        """Atomically publish one secret invitation without storing it in metadata."""

        self._validate_pairing_session_id(session_id)
        if "\n" in pairing_uri or "\r" in pairing_uri or "secret=" not in pairing_uri:
            raise ValueError("pairing invitation URI is malformed")
        path = (
            output_path.expanduser().resolve()
            if output_path is not None
            else self.default_pairing_invitation_path(session_id)
        )
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"pairing invitation path is a directory: {path}")
        _atomic_write(path, pairing_uri.encode("utf-8"), private=True, overwrite=force)
        protection, limitation = self._apply_pairing_file_protection(path)
        return path, protection, limitation

    def retire_pairing_invitation(self, session_id: str) -> bool:
        """Remove only the default invitation associated with one session."""

        path = self.default_pairing_invitation_path(session_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def cleanup_expired_pairing_invitations(self) -> list[Path]:
        """Retire known expired/non-active files while preserving active sessions."""

        sessions = {item.session_id: item for item in self.load_pairing_sessions().sessions}
        removed: list[Path] = []
        now = self.clock_ns()
        for path in sorted(self.paths.pairing_invitations.glob("*.uri")):
            session = sessions.get(path.stem)
            if session is None:
                continue
            if session.state == "active" and session.expires_at_unix_ns > now:
                continue
            path.unlink()
            removed.append(path)
        return removed

    def _write_model(self, path: Path, value: StrictModel, *, private: bool) -> None:
        validated = type(value).model_validate(value.model_dump(mode="python"))
        payload = (validated.model_dump_json(indent=2) + "\n").encode("utf-8")
        _atomic_write(path, payload, private=private)

    def _read_model(
        self,
        path: Path,
        model: type[DocumentT],
        *,
        rewrite_migration: bool = True,
        private: bool = True,
    ) -> DocumentT | None:
        if not path.exists():
            return None
        if not path.is_file():
            raise IntegrityError(f"state document is not a file: {path}")
        try:
            raw_value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid state document {path}: {exc}") from exc
        if not isinstance(raw_value, dict):
            raise IntegrityError(f"state document must be a JSON object: {path}")
        migrated, changed = _migrate(
            cast(dict[str, Any], raw_value),
            path=path,
            model=model,
        )
        try:
            parsed = model.model_validate(migrated)
        except ValidationError as exc:
            raise IntegrityError(f"state validation failed for {path}: {exc}") from exc
        if changed and rewrite_migration:
            self._write_model(path, parsed, private=private)
        return parsed

    def save_cluster(self, metadata: ClusterMetadata) -> None:
        metadata.verify_identity_binding()
        with self._lock:
            existing = self.load_cluster()
            if existing is not None and existing.cluster_id != metadata.cluster_id:
                raise IntegrityError("refusing to replace state with a different cluster identity")
            if (
                existing is not None
                and existing.coordinator_fingerprint != metadata.coordinator_fingerprint
            ):
                raise IntegrityError("refusing to rotate the cluster coordinator identity")
            self._write_model(self.paths.cluster_metadata, metadata, private=True)

    def load_cluster(self) -> ClusterMetadata | None:
        with self._lock:
            metadata = self._read_model(self.paths.cluster_metadata, ClusterMetadata)
            if metadata is not None:
                metadata.verify_identity_binding()
            return metadata

    def load_nodes(self) -> NodeRegistryDocument:
        with self._lock:
            document = self._read_model(self.paths.nodes, NodeRegistryDocument)
            return document or NodeRegistryDocument()

    def save_node(self, node: NodeMetadata) -> None:
        node.verify_identity_binding()
        with self._lock:
            document = self.load_nodes()
            existing = next((item for item in document.nodes if item.node_id == node.node_id), None)
            if existing is not None and existing.fingerprint != node.fingerprint:
                raise IntegrityError(f"node identity changed for {node.node_id}")
            nodes = [item for item in document.nodes if item.node_id != node.node_id]
            nodes.append(node)
            updated = document.model_copy(
                update={"nodes": sorted(nodes, key=lambda item: item.node_id)}
            )
            self._write_model(self.paths.nodes, updated, private=True)

    def node(self, node_id: str) -> NodeMetadata | None:
        return next((item for item in self.load_nodes().nodes if item.node_id == node_id), None)

    def record_backend_validation(
        self,
        node_id: str,
        record: BackendValidationRecord,
    ) -> NodeMetadata:
        """Retain a scoped gate result only when its claimed evidence is verifiable."""

        node = self.node(node_id)
        if node is None:
            raise IntegrityError(f"cannot attach validation evidence to unknown node {node_id}")
        if record.platform_architecture.lower() != node.architecture.lower():
            raise IntegrityError("validation evidence architecture does not match the node")
        if "validated" in {record.software_status, record.physical_status}:
            assert record.evidence_path is not None
            assert record.evidence_id is not None
            evidence_path = record.evidence_path.expanduser().resolve()
            if not evidence_path.is_file():
                raise IntegrityError(f"validation evidence is not retained: {evidence_path}")
            if record.evidence_id.startswith("sha256:"):
                digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                if record.evidence_id != f"sha256:{digest}":
                    raise IntegrityError("validation evidence SHA-256 does not match its identity")
        records = [
            item
            for item in node.backend_validations
            if not (
                item.backend == record.backend
                and item.platform_system == record.platform_system
                and item.platform_release == record.platform_release
                and item.platform_architecture.lower() == record.platform_architecture.lower()
            )
        ]
        records.append(record)
        software, physical = aggregate_validation_status(
            records,
            backend=node.selected_backend,
            architecture=node.architecture,
            operating_system=node.operating_system,
        )
        updated = node.model_copy(
            update={
                "backend_validations": records,
                "software_validation_status": software,
                "physical_validation_status": physical,
            }
        )
        self.save_node(updated)
        return updated

    def remove_node(self, node_id: str) -> None:
        """Rollback helper used only before a pairing transaction is committed."""

        with self._lock:
            document = self.load_nodes()
            updated = document.model_copy(
                update={"nodes": [item for item in document.nodes if item.node_id != node_id]}
            )
            self._write_model(self.paths.nodes, updated, private=True)

    def load_memberships(self) -> MembershipRegistryDocument:
        with self._lock:
            document = self._read_model(self.paths.memberships, MembershipRegistryDocument)
            return document or MembershipRegistryDocument()

    def save_membership(self, membership: NodeMembership) -> None:
        membership.verify_identity_bindings()
        with self._lock:
            document = self.load_memberships()
            existing = next(
                (item for item in document.memberships if item.node_id == membership.node_id),
                None,
            )
            if existing is not None and existing.node_fingerprint != membership.node_fingerprint:
                raise IntegrityError(f"membership identity changed for {membership.node_id}")
            values = [item for item in document.memberships if item.node_id != membership.node_id]
            values.append(membership)
            updated = document.model_copy(
                update={"memberships": sorted(values, key=lambda item: item.node_id)}
            )
            self._write_model(self.paths.memberships, updated, private=True)

    def membership(self, node_id: str) -> NodeMembership | None:
        return next(
            (item for item in self.load_memberships().memberships if item.node_id == node_id),
            None,
        )

    def remove_membership(self, node_id: str) -> None:
        """Rollback helper used only before a pairing transaction is committed."""

        with self._lock:
            document = self.load_memberships()
            updated = document.model_copy(
                update={
                    "memberships": [
                        item for item in document.memberships if item.node_id != node_id
                    ]
                }
            )
            self._write_model(self.paths.memberships, updated, private=True)

    def load_revocations(self) -> RevocationRegistryDocument:
        with self._lock:
            document = self._read_model(self.paths.revocations, RevocationRegistryDocument)
            return document or RevocationRegistryDocument()

    def append_revocation(self, revocation: NodeRevocation) -> None:
        with self._lock:
            document = self.load_revocations()
            same = [
                item
                for item in document.revocations
                if item.node_id == revocation.node_id and item.generation == revocation.generation
            ]
            if same:
                if same[0] != revocation:
                    raise IntegrityError("conflicting node revocation generation")
                return
            values = [*document.revocations, revocation]
            values.sort(key=lambda item: (item.revoked_at_unix_ns, item.revocation_id))
            self._write_model(
                self.paths.revocations,
                document.model_copy(update={"revocations": values}),
                private=True,
            )

    def is_revoked_fingerprint(self, fingerprint: str) -> bool:
        normalized = fingerprint.removeprefix("sha256:").lower()
        return any(
            item.node_fingerprint == normalized for item in self.load_revocations().revocations
        )

    def load_pairing_sessions(self) -> PairingSessionRegistryDocument:
        with self._lock:
            document = self._read_model(
                self.paths.pairing_sessions,
                PairingSessionRegistryDocument,
                private=False,
            )
            return document or PairingSessionRegistryDocument()

    def save_pairing_session(self, session: PairingSession) -> None:
        with self._lock:
            document = self.load_pairing_sessions()
            values = [item for item in document.sessions if item.session_id != session.session_id]
            values.append(session)
            values.sort(key=lambda item: (item.created_at_unix_ns, item.session_id))
            self._write_model(
                self.paths.pairing_sessions,
                document.model_copy(update={"sessions": values}),
                private=False,
            )

    def invalidate_active_pairing_sessions(self, *, reason: str = "coordinator restart") -> int:
        with self._lock:
            document = self.load_pairing_sessions()
            count = 0
            values: list[PairingSession] = []
            for session in document.sessions:
                if session.state == "active":
                    count += 1
                    session = session.model_copy(
                        update={"state": "invalidated", "last_rejection_reason": reason}
                    )
                values.append(session)
            if count:
                self._write_model(
                    self.paths.pairing_sessions,
                    document.model_copy(update={"sessions": values}),
                    private=False,
                )
                for session in values:
                    if session.state == "invalidated":
                        self.retire_pairing_invitation(session.session_id)
            return count

    def save_runtime(self, runtime: NodeRuntimeMetadata) -> None:
        path = self.paths.node_runtime_directory / f"{runtime.node_id}.json"
        with self._lock:
            self._write_model(path, runtime, private=False)

    def load_runtime(self, node_id: str) -> NodeRuntimeMetadata | None:
        path = self.paths.node_runtime_directory / f"{node_id}.json"
        with self._lock:
            return self._read_model(path, NodeRuntimeMetadata, private=False)

    def load_network_measurements(self) -> NetworkMeasurementRegistryDocument:
        with self._lock:
            document = self._read_model(
                self.paths.network_measurements,
                NetworkMeasurementRegistryDocument,
                private=False,
            )
            return document or NetworkMeasurementRegistryDocument()

    def save_network_measurement(self, measurement: NetworkLinkMeasurement) -> None:
        with self._lock:
            document = self.load_network_measurements()
            values = [
                item
                for item in document.measurements
                if not (
                    item.source_worker_id == measurement.source_worker_id
                    and item.destination_worker_id == measurement.destination_worker_id
                )
            ]
            values.append(measurement)
            values.sort(key=lambda item: (item.source_worker_id, item.destination_worker_id))
            self._write_model(
                self.paths.network_measurements,
                document.model_copy(update={"measurements": values}),
                private=False,
            )

    def load_artifact_cache(self) -> ArtifactCacheDocument:
        with self._lock:
            document = self._read_model(
                self.paths.artifact_cache,
                ArtifactCacheDocument,
                private=False,
            )
            return document or ArtifactCacheDocument()

    def save_artifact_cache(self, document: ArtifactCacheDocument) -> None:
        with self._lock:
            self._write_model(self.paths.artifact_cache, document, private=False)

    def append_audit(self, event: ClusterAuditEvent) -> None:
        payload = event.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        lowered = encoded.lower()
        if any(term in lowered for term in _FORBIDDEN_AUDIT_TERMS):
            raise ValueError("audit event contains a forbidden sensitive field")
        with self._lock:
            self.paths.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.audit_log.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def adopt_legacy_coordinator_identity(self, legacy_directory: str | Path) -> Path | None:
        """Copy a valid old coordinator identity exactly once; never rotate it."""

        destination = self.paths.coordinator_identity
        if destination.is_file():
            CoordinatorIdentity.load(destination)
            return destination
        legacy = Path(legacy_directory).expanduser().resolve()
        candidates = (
            legacy / "coordinator-identity.json",
            legacy / "coordinator-identity.pem",
        )
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is None:
            return None
        CoordinatorIdentity.load(source)
        _atomic_write(destination, source.read_bytes(), private=True)
        CoordinatorIdentity.load(destination)
        return destination

    def load_or_create_node_identity(self) -> WorkerIdentity:
        return WorkerIdentity.load_or_create(self.paths.node_identity)

    def save_node_configuration(self, configuration: NodeConfiguration) -> None:
        with self._lock:
            self._write_model(self.paths.node_configuration, configuration, private=True)

    def load_node_configuration(self) -> NodeConfiguration | None:
        with self._lock:
            return self._read_model(self.paths.node_configuration, NodeConfiguration)


__all__ = ["ClusterStatePaths", "ClusterStateStore"]
