from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock

import pytest

from swarm_inference.experiments.experiment_020.vast import (
    E020_RENTAL_FORBIDDEN,
    VastCommandRunner,
    VastSafetyError,
    assert_rental_armed,
)


@pytest.mark.parametrize(
    "arguments",
    [
        ("create", "instance", "123"),
        ("launch", "instance", "123"),
        ("destroy", "instance", "123"),
        ("stop", "instance", "123"),
    ],
)
def test_e020_mutations_are_rejected_before_subprocess(arguments: tuple[str, ...]) -> None:
    boundary = Mock()
    vast = VastCommandRunner("vastai", runner=boundary)

    with pytest.raises(VastSafetyError, match=E020_RENTAL_FORBIDDEN):
        vast.run(arguments)

    boundary.assert_not_called()
    assert vast.audit[-1].subprocess_invoked is False
    assert vast.audit[-1].classification == "FORBIDDEN"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--help",),
        ("--version",),
        ("show", "user", "--raw"),
        ("show", "ssh-keys", "--raw"),
        ("show", "instances", "--raw"),
        ("search", "offers", "gpu_ram>=24", "--raw"),
        ("create", "instance", "--help"),
    ],
)
def test_read_only_allowlist_invokes_mock_boundary(arguments: tuple[str, ...]) -> None:
    boundary = Mock(
        return_value=subprocess.CompletedProcess(
            ["vastai", *arguments], 0, stdout="{}", stderr=""
        )
    )
    vast = VastCommandRunner("vastai", runner=boundary)

    result = vast.run(arguments)

    assert result.returncode == 0
    boundary.assert_called_once()
    assert vast.audit[-1].classification == "READ_ONLY"


def test_e020_hard_lock_wins_even_when_every_e021_arm_is_present() -> None:
    with pytest.raises(VastSafetyError, match=E020_RENTAL_FORBIDDEN):
        assert_rental_armed(
            experiment_id="experiment-021",
            apply=True,
            approved_plan={
                "approved": True,
                "experiment_id": "experiment-021",
                "plan_sha256": "a" * 64,
            },
            max_budget_usd=1000.0,
            environment={"SWARM_ALLOW_RENTAL": "EXPERIMENT_021"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "experiment-020"),
        ("apply", False),
        ("approved_plan", None),
        ("max_budget_usd", None),
        ("environment", {}),
    ],
)
def test_future_rental_arming_fails_closed_when_one_condition_is_missing(
    field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "experiment_id": "experiment-021",
        "apply": True,
        "approved_plan": {
            "approved": True,
            "experiment_id": "experiment-021",
            "plan_sha256": "b" * 64,
        },
        "max_budget_usd": 500.0,
        "environment": {"SWARM_ALLOW_RENTAL": "EXPERIMENT_021"},
        "e020_read_only": False,
    }
    arguments[field] = value
    with pytest.raises(VastSafetyError, match=E020_RENTAL_FORBIDDEN):
        assert_rental_armed(**arguments)  # type: ignore[arg-type]


def test_future_rental_arming_can_only_pass_outside_e020() -> None:
    assert_rental_armed(
        experiment_id="experiment-021",
        apply=True,
        approved_plan={
            "approved": True,
            "experiment_id": "experiment-021",
            "plan_sha256": "c" * 64,
        },
        max_budget_usd=500.0,
        environment={"SWARM_ALLOW_RENTAL": "EXPERIMENT_021"},
        e020_read_only=False,
    )


def test_safety_audit_appends_across_runner_sessions(tmp_path) -> None:
    boundary = Mock(
        return_value=subprocess.CompletedProcess(["vastai"], 0, stdout="{}", stderr="")
    )
    path = tmp_path / "audit.json"
    first = VastCommandRunner("vastai", runner=boundary)
    first.run(("show", "user", "--raw"))
    first.write_audit(path)
    second = VastCommandRunner("vastai", runner=boundary)
    second.run(("show", "instances", "--raw"))
    second.write_audit(path)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["executed_command_count"] == 2
    assert [row["arguments"][1] for row in receipt["commands"]] == ["user", "instances"]
