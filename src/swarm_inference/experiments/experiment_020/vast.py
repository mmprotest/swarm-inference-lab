"""Fail-closed Vast.ai command boundary for Experiments 020 and 021.

Experiment 020 is compile-time/read-time locked to read-only behavior.  The
arming predicate for Experiment 021 is implemented here so it can be tested in
E020, but the E020 runner can never cross that boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

E020_RENTAL_FORBIDDEN = "E020_RENTAL_FORBIDDEN"
EXPERIMENT_020_READ_ONLY = True


class VastMode(StrEnum):
    READ_ONLY = "READ_ONLY"
    RENTAL_ENABLED = "RENTAL_ENABLED"


class VastSafetyError(RuntimeError):
    """Raised before subprocess invocation when a command is unsafe."""


class CompletedProcessLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[..., CompletedProcessLike]


_MUTATING_VERBS = frozenset(
    {
        "create",
        "launch",
        "start",
        "stop",
        "destroy",
        "delete",
        "remove",
        "change",
        "update",
        "set",
        "copy",
        "cancel",
        "reboot",
        "recycle",
        "attach",
        "detach",
    }
)
_READ_ONLY_PREFIXES = (
    ("show", "user"),
    ("show", "ssh-keys"),
    ("show", "instances"),
    ("search", "offers"),
)
_FUTURE_RENTAL_PREFIXES = (
    ("create", "instance"),
    ("destroy", "instance"),
    ("stop", "instance"),
)


def _tokens(arguments: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in arguments if str(value).strip())


def _is_help(arguments: tuple[str, ...]) -> bool:
    # Vast's argparse exits before dispatch when --help is the final argument.
    return bool(arguments) and arguments[-1] in {"--help", "-h"}


def _is_read_only(arguments: tuple[str, ...]) -> bool:
    if arguments in {("--help",), ("-h",), ("--version",)} or _is_help(arguments):
        return True
    return any(arguments[: len(prefix)] == prefix for prefix in _READ_ONLY_PREFIXES)


def guard_vast_command(arguments: Sequence[str], *, mode: VastMode) -> tuple[str, ...]:
    """Validate a Vast subcommand without invoking Vast.

    Mutating verbs always use the experiment's required error code.  Unknown
    commands also fail closed; E020 has no permissive passthrough.
    """

    normalized = _tokens(arguments)
    if not normalized:
        raise VastSafetyError("E020_VAST_COMMAND_NOT_ALLOWLISTED: empty command")
    if _is_help(normalized):
        return normalized
    lowered = {token.lower() for token in normalized}
    mutating = bool(lowered & _MUTATING_VERBS)
    if mutating:
        if mode is VastMode.READ_ONLY or EXPERIMENT_020_READ_ONLY:
            raise VastSafetyError(E020_RENTAL_FORBIDDEN)
        if not any(
            normalized[: len(prefix)] == prefix for prefix in _FUTURE_RENTAL_PREFIXES
        ):
            raise VastSafetyError(E020_RENTAL_FORBIDDEN)
        return normalized
    if mode is VastMode.READ_ONLY and not _is_read_only(normalized):
        raise VastSafetyError(
            f"E020_VAST_COMMAND_NOT_ALLOWLISTED: {' '.join(normalized)}"
        )
    if mode is VastMode.RENTAL_ENABLED and EXPERIMENT_020_READ_ONLY:
        raise VastSafetyError(E020_RENTAL_FORBIDDEN)
    if not _is_read_only(normalized):
        raise VastSafetyError(E020_RENTAL_FORBIDDEN)
    return normalized


@dataclass(frozen=True, slots=True)
class VastAuditEntry:
    timestamp_unix_ns: int
    executable: str
    arguments: tuple[str, ...]
    mode: str
    classification: str
    subprocess_invoked: bool
    returncode: int | None


class VastCommandRunner:
    """Audited Vast CLI wrapper with an injectable subprocess boundary."""

    def __init__(
        self,
        executable: Path | str,
        *,
        mode: VastMode = VastMode.READ_ONLY,
        runner: Runner = subprocess.run,
    ) -> None:
        self.executable = str(executable)
        self.mode = mode
        self._runner = runner
        self.audit: list[VastAuditEntry] = []

    def run(
        self,
        arguments: Sequence[str],
        *,
        timeout: float = 60.0,
        check: bool = False,
    ) -> CompletedProcessLike:
        try:
            safe = guard_vast_command(arguments, mode=self.mode)
        except VastSafetyError:
            self.audit.append(
                VastAuditEntry(
                    timestamp_unix_ns=time.time_ns(),
                    executable=self.executable,
                    arguments=_tokens(arguments),
                    mode=self.mode.value,
                    classification="FORBIDDEN",
                    subprocess_invoked=False,
                    returncode=None,
                )
            )
            raise
        completed = self._runner(
            [self.executable, *safe],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        self.audit.append(
            VastAuditEntry(
                timestamp_unix_ns=time.time_ns(),
                executable=self.executable,
                arguments=safe,
                mode=self.mode.value,
                classification="READ_ONLY",
                subprocess_invoked=True,
                returncode=int(completed.returncode),
            )
        )
        return completed

    def write_audit(self, path: Path) -> None:
        existing: list[dict[str, Any]] = []
        if path.exists():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(prior.get("commands"), list):
                    existing = [dict(row) for row in prior["commands"]]
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                # A malformed prior receipt must not prevent the current audit
                # from being persisted; final validation will flag it.
                existing = []
        current = [asdict(row) for row in self.audit]
        commands: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in [*existing, *current]:
            key = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                commands.append(row)
        payload = {
            "schema_version": "experiment-020-vast-safety-audit-v1",
            "experiment": "experiment-020",
            "mode": self.mode.value,
            "experiment_020_read_only": EXPERIMENT_020_READ_ONLY,
            "commands": commands,
            "executed_command_count": sum(
                bool(row.get("subprocess_invoked")) for row in commands
            ),
            "mutating_command_count": sum(
                bool(row.get("subprocess_invoked"))
                and row.get("classification") != "READ_ONLY"
                for row in commands
            ),
            "vast_resource_mutations": 0,
            "gpu_rentals": 0,
        }
        atomic_write_json(path, payload)


def assert_rental_armed(
    *,
    experiment_id: str,
    apply: bool,
    approved_plan: Mapping[str, Any] | None,
    max_budget_usd: float | None,
    environment: Mapping[str, str] | None = None,
    e020_read_only: bool = EXPERIMENT_020_READ_ONLY,
) -> None:
    """Require every independent E021 rental interlock to be satisfied."""

    env = os.environ if environment is None else environment
    plan_ok = bool(
        approved_plan
        and approved_plan.get("approved") is True
        and approved_plan.get("experiment_id") == "experiment-021"
        and approved_plan.get("plan_sha256")
    )
    armed = (
        env.get("SWARM_ALLOW_RENTAL") == "EXPERIMENT_021"
        and apply
        and experiment_id == "experiment-021"
        and plan_ok
        and max_budget_usd is not None
        and 0 < float(max_budget_usd) < float("inf")
    )
    if e020_read_only or not armed:
        raise VastSafetyError(E020_RENTAL_FORBIDDEN)


def redact_account_payload(kind: str, payload: Any) -> dict[str, Any]:
    """Return the minimum non-sensitive receipt for a read-only account call."""

    if kind == "user":
        mapping = payload if isinstance(payload, Mapping) else {}
        return {
            "authenticated": bool(mapping),
            "account_id_present": any(key in mapping for key in ("id", "user_id")),
            "credit_balance_present": any(
                key in mapping for key in ("credit", "balance", "balance_usd")
            ),
            "email_redacted": "email" in mapping,
        }
    if kind == "ssh_keys":
        rows = payload if isinstance(payload, list) else []
        return {
            "ssh_key_present": bool(rows),
            "ssh_key_count": len(rows),
            "key_material_persisted": False,
        }
    if kind == "instances":
        rows = payload if isinstance(payload, list) else []
        return {
            "unrelated_instance_count": len(rows),
            "instance_details_persisted": False,
        }
    raise ValueError(f"unsupported redaction kind: {kind}")


def plan_digest(plan: Mapping[str, Any]) -> str:
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


__all__ = [
    "E020_RENTAL_FORBIDDEN",
    "EXPERIMENT_020_READ_ONLY",
    "VastAuditEntry",
    "VastCommandRunner",
    "VastMode",
    "VastSafetyError",
    "assert_rental_armed",
    "atomic_write_json",
    "guard_vast_command",
    "plan_digest",
    "redact_account_payload",
]
