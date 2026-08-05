"""Explicit blue/green node-runtime wheel updates with bounded rollback."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import PositiveInt

from swarm_inference.cluster.models import ClusterAuditEvent, node_id_from_fingerprint
from swarm_inference.cluster.service_manager import ServiceManager
from swarm_inference.cluster.state import ClusterStateStore
from swarm_inference.config.models import StrictModel
from swarm_inference.filesystem import replace_atomically
from swarm_inference.platforms.base import PlatformAdapter, ServiceDefinition


class ActiveRuntimePointer(StrictModel):
    schema_version: Literal[1] = 1
    document_version: Literal[1] = 1
    python_executable: str
    wheel_sha256: str
    committed_at_unix_ns: PositiveInt


class RuntimeUpdateResult(StrictModel):
    schema_version: Literal[1] = 1
    document_version: Literal[1] = 1
    update_id: str
    wheel_sha256: str
    source_wheel: str
    staged_runtime: str
    previous_runtime: str
    status: Literal["staged", "committed", "rolled-back", "failed"]
    detail: str
    started_at_unix_ns: PositiveInt
    completed_at_unix_ns: PositiveInt | None = None


ProcessRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _run_process(argv: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _atomic_model(path: Path, value: StrictModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomically(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_active_runtime_python(state: ClusterStateStore) -> Path | None:
    path = state.paths.runtime / "active-runtime.json"
    if not path.is_file():
        return None
    pointer = ActiveRuntimePointer.model_validate_json(path.read_text(encoding="utf-8"))
    executable = Path(pointer.python_executable).expanduser().resolve()
    if not executable.is_file():
        raise RuntimeError("active runtime pointer references a missing Python executable")
    return executable


class RuntimeUpdater:
    """Stage a wheel in an isolated runtime and atomically switch the user service."""

    def __init__(
        self,
        *,
        state: ClusterStateStore,
        platform: PlatformAdapter,
        services: ServiceManager,
        process_runner: ProcessRunner = _run_process,
        process_timeout_seconds: float = 180.0,
        startup_timeout_seconds: float = 60.0,
    ) -> None:
        if not 0 < process_timeout_seconds <= 600:
            raise ValueError("update process timeout must be in (0, 600] seconds")
        if not 0 < startup_timeout_seconds <= 300:
            raise ValueError("update startup timeout must be in (0, 300] seconds")
        self.state = state
        self.platform = platform
        self.services = services
        self.process_runner = process_runner
        self.process_timeout_seconds = process_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_wheel(path: Path) -> None:
        if path.suffix != ".whl" or not zipfile.is_zipfile(path):
            raise ValueError("source update must be a valid wheel archive")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if not any(name.startswith("swarm_inference/") for name in names):
            raise ValueError("wheel does not contain the swarm_inference package")
        if not any(name.endswith(".dist-info/WHEEL") for name in names):
            raise ValueError("wheel has no distribution metadata")

    def _uv(self) -> str:
        explicit = os.environ.get("UV_EXECUTABLE")
        selected = explicit or shutil.which("uv")
        if selected is None:
            raise RuntimeError(
                "uv is unavailable; reinstall with the platform installer before updating"
            )
        return selected

    def _run_checked(self, argv: Sequence[str], *, stage: str) -> None:
        try:
            result = self.process_runner(argv, self.process_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"update {stage} exceeded its bounded timeout") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"update {stage} failed: {detail[:1000]}")

    def _python_path(self, root: Path) -> Path:
        return root / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")

    async def _wait_started(self, definition: ServiceDefinition, *, after_ns: int) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        node_id = definition.node_id
        while time.monotonic() < deadline:
            runtime = self.state.load_runtime(node_id)
            if (
                runtime is not None
                and runtime.last_refresh_unix_ns >= after_ns
                and runtime.state == "ready"
            ):
                return
            if runtime is not None and runtime.state in {"blocked", "failed"}:
                raise RuntimeError(runtime.reason or "updated agent failed startup")
            await asyncio.sleep(0.25)
        raise TimeoutError("updated node agent did not report ready before the startup timeout")

    def _audit(self, event_type: str, detail: str) -> None:
        cluster = self.state.load_cluster()
        identity = self.state.load_or_create_node_identity()
        self.state.append_audit(
            ClusterAuditEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                timestamp_unix_ns=time.time_ns(),
                cluster_id=cluster.cluster_id if cluster else None,
                node_id=node_id_from_fingerprint(identity.public_key_fingerprint),
                detail=detail,
            )
        )

    async def update(
        self,
        source_wheel: Path,
        current_definition: ServiceDefinition,
    ) -> RuntimeUpdateResult:
        source = source_wheel.expanduser().resolve()
        self._validate_wheel(source)
        digest = self._hash(source)
        update_id = f"update-{uuid4().hex}"
        started = time.time_ns()
        versions = self.state.paths.runtime / "versions"
        target = versions / digest
        staging = versions / f".{digest}.{uuid4().hex}.staging"
        previous = current_definition.executable.resolve()
        result = RuntimeUpdateResult(
            update_id=update_id,
            wheel_sha256=digest,
            source_wheel=str(source),
            staged_runtime=str(target),
            previous_runtime=str(previous),
            status="staged",
            detail="wheel hash and archive structure validated",
            started_at_unix_ns=started,
        )
        status_path = self.state.paths.runtime / "updates" / f"{update_id}.json"
        _atomic_model(status_path, result)
        self._audit("update_staged", f"update {update_id} wheel {digest[:12]} staged")
        try:
            if not target.is_dir():
                staging.parent.mkdir(parents=True, exist_ok=True)
                self._run_checked(
                    [
                        self._uv(),
                        "venv",
                        "--python",
                        sys.executable,
                        "--system-site-packages",
                        str(staging),
                    ],
                    stage="environment creation",
                )
                staged_python = self._python_path(staging)
                self._run_checked(
                    [
                        self._uv(),
                        "pip",
                        "install",
                        "--python",
                        str(staged_python),
                        "--no-deps",
                        "--force-reinstall",
                        str(source),
                    ],
                    stage="wheel installation",
                )
                self._run_checked(
                    [
                        str(staged_python),
                        "-c",
                        (
                            "import swarm_inference, swarm_inference.cli; "
                            "print(swarm_inference.__version__)"
                        ),
                    ],
                    stage="runtime import validation",
                )
                os.replace(staging, target)
            staged_python = self._python_path(target).resolve()
            new_definition = current_definition.model_copy(
                update={"executable": staged_python},
                deep=True,
            )
            await self.services.stop(current_definition)
            startup_marker = time.time_ns()
            installed = await self.services.install(new_definition)
            if not installed.installed:
                raise RuntimeError(f"updated service installation failed: {installed.detail}")
            await self._wait_started(new_definition, after_ns=startup_marker)
            pointer = ActiveRuntimePointer(
                python_executable=str(staged_python),
                wheel_sha256=digest,
                committed_at_unix_ns=time.time_ns(),
            )
            _atomic_model(self.state.paths.runtime / "active-runtime.json", pointer)
            committed = result.model_copy(
                update={
                    "status": "committed",
                    "detail": "updated agent reported ready; active runtime committed",
                    "completed_at_unix_ns": time.time_ns(),
                }
            )
            _atomic_model(status_path, committed)
            self._audit("update_committed", f"update {update_id} committed")
            return committed
        except BaseException as exc:
            try:
                await self.services.stop(
                    current_definition.model_copy(update={"executable": self._python_path(target)})
                )
                rollback = await self.services.install(current_definition)
                if not rollback.installed:
                    raise RuntimeError(rollback.detail)
            except BaseException as rollback_exc:
                failed = result.model_copy(
                    update={
                        "status": "failed",
                        "detail": (
                            f"update failed: {exc}; rollback failed: {rollback_exc}; "
                            "manual service repair is required"
                        ),
                        "completed_at_unix_ns": time.time_ns(),
                    }
                )
                _atomic_model(status_path, failed)
                self._audit("update_rolled_back", failed.detail)
                return failed
            rolled_back = result.model_copy(
                update={
                    "status": "rolled-back",
                    "detail": f"update startup failed and previous runtime was restored: {exc}",
                    "completed_at_unix_ns": time.time_ns(),
                }
            )
            _atomic_model(status_path, rolled_back)
            self._audit("update_rolled_back", rolled_back.detail)
            return rolled_back
        finally:
            if staging.is_dir():
                shutil.rmtree(staging)


__all__ = [
    "ActiveRuntimePointer",
    "RuntimeUpdateResult",
    "RuntimeUpdater",
    "load_active_runtime_python",
]
