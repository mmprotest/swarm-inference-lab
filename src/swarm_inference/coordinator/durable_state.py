"""Atomic local persistence for product coordinator recovery and inspection."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from swarm_inference.config.models import WorkerCapability
from swarm_inference.exceptions import IntegrityError
from swarm_inference.filesystem import replace_atomically
from swarm_inference.protocol.product import (
    ProductRequestPhase,
    ProductRequestRecoveryState,
)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class DurableCoordinatorState:
    """Documented on-disk state owned by one coordinator identity."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.requests_directory = self.directory / "requests"
        self.workers_directory = self.directory / "workers"
        self.replay_directory = self.directory / "replay"
        self.audit_path = self.directory / "audit.jsonl"
        self.metadata_path = self.directory / "coordinator.json"
        self._lock = threading.RLock()

    @property
    def identity_path(self) -> Path:
        canonical = self.directory / "coordinator-identity.json"
        legacy = self.directory / "coordinator-identity.pem"
        return legacy if legacy.is_file() and not canonical.exists() else canonical

    def save_metadata(self, values: dict[str, Any]) -> None:
        with self._lock:
            _atomic_write(
                self.metadata_path,
                json.dumps(values, indent=2, sort_keys=True) + "\n",
            )

    def load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            return {}
        try:
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid durable coordinator metadata: {exc}") from exc
        if not isinstance(value, dict):
            raise IntegrityError("durable coordinator metadata must be a JSON object")
        return value

    def save_worker(self, capability: WorkerCapability) -> None:
        path = self.workers_directory / f"{capability.worker_id}.json"
        with self._lock:
            if path.is_file():
                try:
                    known = WorkerCapability.model_validate_json(path.read_text(encoding="utf-8"))
                except (OSError, ValidationError) as exc:
                    raise IntegrityError(f"invalid persisted worker identity: {exc}") from exc
                if known.public_key != capability.public_key:
                    raise IntegrityError(
                        f"worker {capability.worker_id} re-registered with a different identity"
                    )
            _atomic_write(path, capability.model_dump_json(indent=2) + "\n")

    def known_workers(self) -> list[WorkerCapability]:
        if not self.workers_directory.is_dir():
            return []
        workers: list[WorkerCapability] = []
        for path in sorted(self.workers_directory.glob("*.json")):
            try:
                workers.append(
                    WorkerCapability.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValidationError) as exc:
                raise IntegrityError(f"invalid persisted worker record {path}: {exc}") from exc
        return workers

    def save_request(self, state: ProductRequestRecoveryState) -> None:
        path = self.requests_directory / f"{state.request_id}.json"
        with self._lock:
            _atomic_write(path, state.model_dump_json(indent=2) + "\n")

    def load_requests(self) -> dict[str, ProductRequestRecoveryState]:
        if not self.requests_directory.is_dir():
            return {}
        loaded: dict[str, ProductRequestRecoveryState] = {}
        for path in sorted(self.requests_directory.glob("*.json")):
            try:
                state = ProductRequestRecoveryState.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                raise IntegrityError(f"invalid persisted request state {path}: {exc}") from exc
            if state.request_id in loaded:
                raise IntegrityError(f"duplicate persisted request {state.request_id}")
            loaded[state.request_id] = state
        return loaded

    def mark_restart_boundaries(self) -> dict[str, ProductRequestRecoveryState]:
        """Never treat pre-restart worker connections or tokens as live."""

        loaded = self.load_requests()
        now = time.time_ns()
        active = {
            ProductRequestPhase.PENDING,
            ProductRequestPhase.RUNNING,
            ProductRequestPhase.RECOVERING,
        }
        for request_id, state in list(loaded.items()):
            replay_error: str | None
            try:
                replay_tokens = self.load_replay_tokens(request_id)
            except IntegrityError as exc:
                replay_tokens = []
                replay_error = str(exc)
            else:
                replay_error = None
            if replay_error is not None or replay_tokens != state.accepted_generated_token_ids:
                updated = state.model_copy(
                    update={
                        "status": ProductRequestPhase.FAILED,
                        "last_error": (
                            replay_error
                            or "accepted token history does not match the durable replay log"
                        ),
                        "updated_unix_ns": now,
                    }
                )
                self.save_request(updated)
                loaded[request_id] = updated
                continue
            if state.status not in active:
                continue
            recoverable = bool(state.prompt_token_ids) and state.sampling_policy == "greedy"
            updated = state.model_copy(
                update={
                    "status": (
                        ProductRequestPhase.RECOVERABLE
                        if recoverable
                        else ProductRequestPhase.FAILED
                    ),
                    "last_error": (
                        "coordinator restarted; workers must re-register before recovery"
                    ),
                    "updated_unix_ns": now,
                }
            )
            self.save_request(updated)
            loaded[request_id] = updated
        return loaded

    def load_replay_tokens(self, request_id: str) -> list[int]:
        """Validate and return the exact durable accepted-token prefix."""

        path = self.replay_directory / f"{request_id}.jsonl"
        if not path.is_file():
            return []
        tokens: list[int] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise IntegrityError(f"cannot read replay log for {request_id}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(
                    f"invalid replay log for {request_id} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise IntegrityError(
                    f"invalid replay log object for {request_id} at line {line_number}"
                )
            position = row.get("token_position")
            token_id = row.get("token_id")
            request_generation = row.get("request_generation")
            route_generation = row.get("route_generation")
            if (
                row.get("event") != "token_accepted"
                or row.get("request_id") != request_id
                or isinstance(position, bool)
                or not isinstance(position, int)
                or position != len(tokens)
                or isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or isinstance(request_generation, bool)
                or not isinstance(request_generation, int)
                or request_generation <= 0
                or isinstance(route_generation, bool)
                or not isinstance(route_generation, int)
                or route_generation <= 0
            ):
                raise IntegrityError(
                    f"invalid replay log record for {request_id} at line {line_number}"
                )
            tokens.append(token_id)
        return tokens

    def append_replay_token(
        self,
        *,
        request_id: str,
        request_generation: int,
        route_generation: int,
        token_position: int,
        token_id: int,
    ) -> None:
        row = {
            "event": "token_accepted",
            "request_id": request_id,
            "request_generation": request_generation,
            "route_generation": route_generation,
            "token_position": token_position,
            "token_id": token_id,
            "recorded_unix_ns": time.time_ns(),
        }
        path = self.replay_directory / f"{request_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

    def append_audit_event(self, event_type: str, **details: Any) -> dict[str, Any]:
        row = {
            "event_type": event_type,
            "timestamp_unix_ns": time.time_ns(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            **details,
        }
        serialized = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        return row


__all__ = ["DurableCoordinatorState"]
