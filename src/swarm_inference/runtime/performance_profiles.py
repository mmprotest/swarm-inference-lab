"""Persistent exactness and performance evidence for native fast paths."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import ConfigDict, Field, NonNegativeInt

from swarm_inference.config.models import StrictModel
from swarm_inference.execution.fast_path import FastPathMeasurement


class FastPathProfileKey(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_content_fingerprint: str
    adapter_id: str
    adapter_version: str
    engine_version: str
    fast_path_id: str
    device_uuid: str
    driver_version: str
    runtime_version: str
    dtype: str
    quantization: str
    stage_ownership: str
    batch_bucket: int = Field(ge=1)
    context_bucket: int = Field(ge=1)

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FastPathProfile(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    key: FastPathProfileKey
    measurements: tuple[FastPathMeasurement, ...]
    selected_candidate: str | None = None
    exactness_result: Literal["passed", "failed", "not-run"]
    failure_reason: str | None = None
    timestamp_unix_ns: NonNegativeInt = Field(default_factory=time.time_ns)


class _ProfileDocument(StrictModel):
    schema_version: Literal[1] = 1
    profiles: dict[str, FastPathProfile] = Field(default_factory=dict)


class FastPathProfileStore:
    """Small content-keyed store; partial hardware matches are never reused."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._lock = threading.RLock()

    def _load(self) -> _ProfileDocument:
        if not self.path.is_file():
            return _ProfileDocument()
        try:
            return _ProfileDocument.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"fast-path profile store is invalid: {self.path}") from exc

    def _save(self, document: _ProfileDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def get(self, key: FastPathProfileKey) -> FastPathProfile | None:
        with self._lock:
            profile = self._load().profiles.get(key.fingerprint)
            if profile is None:
                return None
            if profile.key != key:
                raise RuntimeError("fast-path profile fingerprint collision")
            return profile

    def put(self, profile: FastPathProfile) -> None:
        with self._lock:
            document = self._load()
            profiles = dict(document.profiles)
            profiles[profile.key.fingerprint] = profile
            self._save(document.model_copy(update={"profiles": profiles}))

    def records(self) -> tuple[FastPathProfile, ...]:
        with self._lock:
            profiles = self._load().profiles
            return tuple(profiles[key] for key in sorted(profiles))


def profile_key_from_runtime(
    *,
    model_content_fingerprint: str,
    adapter_id: str,
    adapter_version: str,
    engine_version: str,
    fast_path_id: str,
    device_uuid: str | None,
    driver_version: str | None,
    runtime_version: str | None,
    dtype: str,
    quantization: str | None,
    stage_ownership: dict[str, Any] | str,
    batch_bucket: int,
    context_bucket: int,
) -> FastPathProfileKey:
    ownership = (
        stage_ownership
        if isinstance(stage_ownership, str)
        else json.dumps(stage_ownership, sort_keys=True, separators=(",", ":"))
    )
    return FastPathProfileKey(
        model_content_fingerprint=model_content_fingerprint,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        engine_version=engine_version,
        fast_path_id=fast_path_id,
        device_uuid=device_uuid or "unavailable",
        driver_version=driver_version or "unavailable",
        runtime_version=runtime_version or "unavailable",
        dtype=dtype,
        quantization=quantization or "none",
        stage_ownership=ownership,
        batch_bucket=batch_bucket,
        context_bucket=context_bucket,
    )


__all__ = [
    "FastPathProfile",
    "FastPathProfileKey",
    "FastPathProfileStore",
    "profile_key_from_runtime",
]
