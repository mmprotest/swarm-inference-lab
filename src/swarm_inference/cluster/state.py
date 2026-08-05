"""Atomic, strict persistence for cluster security and mutable node state."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast
from uuid import uuid4

from pydantic import ValidationError

from swarm_inference.cluster.models import (
    ArtifactCacheDocument,
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
)
from swarm_inference.config.models import StrictModel
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.platforms import default_state_directory
from swarm_inference.security.identity import CoordinatorIdentity, WorkerIdentity

DocumentT = TypeVar("DocumentT", bound=StrictModel)
Migration = Callable[[dict[str, Any]], dict[str, Any]]


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


def _migrate(raw: dict[str, Any], *, path: Path) -> tuple[dict[str, Any], bool]:
    if "schema_version" not in raw:
        raise IntegrityError(f"state document has no schema_version: {path}")
    value = raw["schema_version"]
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"invalid schema_version in {path}: {value!r}") from exc
    changed = False
    while version < 1:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise IntegrityError(f"no migration from state schema {version} for {path}")
        raw = migration(raw)
        version += 1
        changed = True
    if version != 1:
        raise IntegrityError(f"unsupported future state schema {version} for {path}")
    return raw, changed


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
        ):
            directory.mkdir(parents=True, exist_ok=True)
        # Some Windows filesystems do not implement POSIX mode bits.  The
        # user-scoped root and private file modes remain the safe fallback.
        with suppress(OSError):
            os.chmod(self.paths.security, stat.S_IRWXU)

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
        migrated, changed = _migrate(cast(dict[str, Any], raw_value), path=path)
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
