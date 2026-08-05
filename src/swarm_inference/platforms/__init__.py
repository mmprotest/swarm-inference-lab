"""Single dispatch point for host-specific product operations."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm_inference.platforms.base import CommandRunner, PlatformAdapter


def default_state_directory(
    *,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
    home_directory: Path | None = None,
) -> Path:
    """Return the documented per-user state root without touching the filesystem."""

    selected = (system or sys.platform).lower()
    values = environment if environment is not None else os.environ
    home = home_directory or Path.home()
    if selected.startswith("win"):
        local_app_data = values.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else home / "AppData" / "Local"
        return base / "SwarmInference"
    if selected == "darwin":
        return home / "Library" / "Application Support" / "SwarmInference"
    xdg_state = values.get("XDG_STATE_HOME")
    base = Path(xdg_state) if xdg_state else home / ".local" / "state"
    return base / "swarm-inference"


def get_platform_adapter(
    *,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
    home_directory: Path | None = None,
    command_runner: CommandRunner | None = None,
) -> PlatformAdapter:
    """Construct the one adapter selected at the platform boundary."""

    from swarm_inference.platforms.base import default_command_runner

    selected = (system or sys.platform).lower()
    runner = command_runner or default_command_runner
    if selected.startswith("win"):
        from swarm_inference.platforms.windows import WindowsPlatformAdapter

        return WindowsPlatformAdapter(
            environment=environment,
            home_directory=home_directory,
            command_runner=runner,
        )
    if selected == "darwin" or selected == "macos":
        from swarm_inference.platforms.macos import MacOSPlatformAdapter

        return MacOSPlatformAdapter(
            environment=environment,
            home_directory=home_directory,
            command_runner=runner,
        )
    if selected.startswith("linux"):
        from swarm_inference.platforms.linux import LinuxPlatformAdapter

        return LinuxPlatformAdapter(
            environment=environment,
            home_directory=home_directory,
            command_runner=runner,
        )
    raise RuntimeError(f"unsupported platform {selected!r}; use --foreground only")


__all__ = ["default_state_directory", "get_platform_adapter"]
