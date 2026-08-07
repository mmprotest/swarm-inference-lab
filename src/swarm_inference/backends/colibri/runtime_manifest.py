"""Validated installer/runtime metadata for canonical Colibri execution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROFILE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")
_MODEL_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
_INTEGER_SETTING = re.compile(r"[0-9]{1,6}")

# Experiment 009 reverse-confirmed this bounded policy. Prefetch and other
# experimental knobs deliberately do not enter the product allowlist.
ROUTING_PROFILE_SETTINGS = frozenset({"PILOT", "WIDE", "PILOT_EVICT_GUARD"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_sha256(value: object, *, field: str) -> str:
    normalised = str(value).casefold()
    if normalised.startswith("sha256:"):
        normalised = normalised.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", normalised):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return normalised


def _relative_asset(value: object, *, manifest_path: Path, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must be a manifest-relative path")
    root = manifest_path.parent.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive after the parts check
        raise ValueError(f"{field} escapes the Colibri runtime directory") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _optional_path(value: object, *, manifest_path: Path) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _verify_binaries(
    hashes: dict[str, str],
    *,
    engine_directory: Path,
) -> dict[str, str]:
    if not engine_directory.is_dir():
        raise FileNotFoundError("Colibri engine_directory is unavailable")
    verified: dict[str, str] = {}
    root = engine_directory.resolve()
    for name, supplied_hash in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Colibri binary hash keys must be engine-relative paths")
        candidates = [root / relative]
        if not relative.suffix:
            candidates.append(root / relative.with_suffix(".exe"))
        binary = next((item.resolve() for item in candidates if item.is_file()), None)
        if binary is None:
            raise FileNotFoundError(f"Colibri runtime binary is unavailable: {name}")
        try:
            binary.relative_to(root)
        except ValueError as exc:  # pragma: no cover - defensive after the parts check
            raise ValueError("Colibri binary path escapes engine_directory") from exc
        expected = _normalise_sha256(supplied_hash, field=f"binary_hashes[{name!r}]")
        if _sha256(binary) != expected:
            raise ValueError(f"Colibri runtime binary hash does not match: {name}")
        verified[name] = expected
    return verified


@dataclass(frozen=True, slots=True)
class ColibriRoutingProfile:
    profile_id: str
    adapter_id: str
    model_fingerprint: str
    hot_pin_path: Path
    hot_pin_sha256: str
    settings: dict[str, str]
    exactness_passed: bool
    measured_utility: float
    evidence_fingerprint: str
    content_fingerprint: str

    @property
    def admitted(self) -> bool:
        return self.exactness_passed and self.measured_utility > 0

    @property
    def environment(self) -> dict[str, str]:
        return {**self.settings, "COLI_HOT_PIN_PATH": str(self.hot_pin_path)}


@dataclass(frozen=True, slots=True)
class ColibriRuntimeManifest:
    path: Path
    runtime_revision: str
    binary_hashes: dict[str, str]
    model_families: tuple[str, ...]
    formats: tuple[str, ...]
    fast_paths: tuple[str, ...]
    engine_directory: Path | None
    source_directory: Path | None
    build_manifest: Path | None
    routing_profiles: tuple[ColibriRoutingProfile, ...]

    def routing_profile(self, profile_id: str) -> ColibriRoutingProfile:
        matches = [item for item in self.routing_profiles if item.profile_id == profile_id]
        if len(matches) != 1:
            raise ValueError(f"unknown Colibri routing profile {profile_id!r}")
        return matches[0]


def _routing_profile(raw: object, *, manifest_path: Path) -> ColibriRoutingProfile:
    if not isinstance(raw, dict):
        raise ValueError("Colibri routing profile must be an object")
    required = {
        "profile_id",
        "adapter_id",
        "model_fingerprint",
        "hot_pin_path",
        "hot_pin_sha256",
        "settings",
        "exactness_passed",
        "measured_utility",
        "evidence_fingerprint",
    }
    unknown = set(raw).difference(required)
    missing = required.difference(raw)
    if missing or unknown:
        raise ValueError(
            "Colibri routing profile fields do not match the canonical schema: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    profile_id = str(raw["profile_id"])
    if not _PROFILE_ID.fullmatch(profile_id):
        raise ValueError("Colibri routing profile_id is invalid")
    adapter_id = str(raw["adapter_id"])
    if not adapter_id:
        raise ValueError("Colibri routing profile adapter_id is required")
    model_fingerprint = str(raw["model_fingerprint"]).casefold()
    if not _MODEL_FINGERPRINT.fullmatch(model_fingerprint):
        raise ValueError("Colibri routing profile model_fingerprint is invalid")
    path = _relative_asset(
        raw["hot_pin_path"],
        manifest_path=manifest_path,
        field="routing profile hot_pin_path",
    )
    expected_sha = _normalise_sha256(
        raw["hot_pin_sha256"], field="routing profile hot_pin_sha256"
    )
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise ValueError("Colibri routing profile bitmap hash does not match its manifest")
    supplied_settings = raw["settings"]
    if not isinstance(supplied_settings, dict):
        raise ValueError("Colibri routing profile settings must be an object")
    unsupported = set(supplied_settings).difference(ROUTING_PROFILE_SETTINGS)
    if unsupported:
        raise ValueError(f"Colibri routing profile contains unsupported settings: {sorted(unsupported)}")
    settings = {str(key): str(value) for key, value in supplied_settings.items()}
    if not settings or any(not _INTEGER_SETTING.fullmatch(value) for value in settings.values()):
        raise ValueError("Colibri routing profile settings must be bounded non-negative integers")
    if type(raw["exactness_passed"]) is not bool:
        raise ValueError("Colibri routing profile exactness_passed must be boolean")
    measured_utility = float(raw["measured_utility"])
    if not (-1000 < measured_utility < 1000):
        raise ValueError("Colibri routing profile measured utility is outside the bounded range")
    evidence_fingerprint = str(raw["evidence_fingerprint"])
    if not evidence_fingerprint or len(evidence_fingerprint) > 256:
        raise ValueError("Colibri routing profile evidence_fingerprint is invalid")
    identity = json.dumps(
        {
            "profile_id": profile_id,
            "adapter_id": adapter_id,
            "model_fingerprint": model_fingerprint,
            "hot_pin_sha256": expected_sha,
            "settings": settings,
            "exactness_passed": raw["exactness_passed"],
            "measured_utility": measured_utility,
            "evidence_fingerprint": evidence_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ColibriRoutingProfile(
        profile_id=profile_id,
        adapter_id=adapter_id,
        model_fingerprint=model_fingerprint,
        hot_pin_path=path,
        hot_pin_sha256=expected_sha,
        settings=settings,
        exactness_passed=raw["exactness_passed"],
        measured_utility=measured_utility,
        evidence_fingerprint=evidence_fingerprint,
        content_fingerprint="sha256:"
        + hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    )


def load_colibri_runtime_manifest(path: str | Path) -> ColibriRuntimeManifest:
    """Load runtime facts and fail closed on malformed routing profiles."""

    manifest_path = Path(path).expanduser().resolve()
    raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Colibri runtime manifest must contain an object")
    revision = str(raw["runtime_revision"])
    hashes = {str(key): str(value) for key, value in dict(raw["binary_hashes"]).items()}
    families = tuple(str(item) for item in raw.get("model_families", ()))
    formats = tuple(str(item) for item in raw.get("formats", ("safetensors",)))
    if not revision or not hashes or not families:
        raise ValueError("revision, binary hashes, and model families are required")
    engine_directory = _optional_path(raw.get("engine_directory"), manifest_path=manifest_path)
    if engine_directory is None:
        raise ValueError("Colibri runtime manifest requires engine_directory")
    hashes = _verify_binaries(hashes, engine_directory=engine_directory)
    profiles = tuple(
        _routing_profile(item, manifest_path=manifest_path)
        for item in raw.get("routing_profiles", ())
    )
    identifiers = [item.profile_id for item in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Colibri routing profile IDs must be unique")
    configured_fast_paths = {
        str(item) for item in raw.get("fast_paths", ()) if str(item) != "routing-aware-placement"
    }
    if any(profile.admitted for profile in profiles):
        configured_fast_paths.add("routing-aware-placement")
    fast_paths = tuple(sorted(configured_fast_paths))
    return ColibriRuntimeManifest(
        path=manifest_path,
        runtime_revision=revision,
        binary_hashes=hashes,
        model_families=families,
        formats=formats,
        fast_paths=fast_paths,
        engine_directory=engine_directory,
        source_directory=_optional_path(raw.get("source_directory"), manifest_path=manifest_path),
        build_manifest=_optional_path(raw.get("build_manifest"), manifest_path=manifest_path),
        routing_profiles=profiles,
    )


__all__ = [
    "ROUTING_PROFILE_SETTINGS",
    "ColibriRoutingProfile",
    "ColibriRuntimeManifest",
    "load_colibri_runtime_manifest",
]
