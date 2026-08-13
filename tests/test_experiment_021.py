from __future__ import annotations

from swarm_inference.experiments.experiment_020.vast import (
    VastMode,
    VastSafetyError,
    guard_vast_command,
)
from swarm_inference.experiments.experiment_021 import (
    CANONICAL_SWARM_DEFINITION,
    E021_VAST_MUTATIONS_ALLOWED,
    E021_ZERO_RENTAL,
)
from swarm_inference.experiments.experiment_021.control_plane import (
    run_control_plane_scale,
)
from swarm_inference.experiments.experiment_021.runtime import (
    run_shard_cache_startup_self_test,
)
from swarm_inference.experiments.experiment_021.simulation import (
    REGIMES,
    independent_profile,
)


def test_canonical_swarm_definition_is_frozen() -> None:
    assert CANONICAL_SWARM_DEFINITION == (
        "A Swarm result is only a Swarm result if the model cannot be executed "
        "by assigning whole layers to the participating workers, and the reported "
        "performance emerges from sub-layer fragments distributed across independent "
        "machines."
    )


def test_e021_is_compile_time_zero_rental() -> None:
    assert E021_ZERO_RENTAL is True
    assert E021_VAST_MUTATIONS_ALLOWED is False
    assert guard_vast_command(
        ("search", "offers", "num_gpus=1"), mode=VastMode.READ_ONLY
    )
    try:
        guard_vast_command(("create", "instance", "1"), mode=VastMode.READ_ONLY)
    except VastSafetyError as exc:
        assert "RENTAL_FORBIDDEN" in str(exc)
    else:
        raise AssertionError("Vast mutation escaped the E021 read-only guard")


def test_every_network_regime_is_an_independent_machine_profile() -> None:
    expected = {
        "A": (0.25, 25.0),
        "B": (1.0, 10.0),
        "C": (5.0, 1.0),
        "D": (20.0, 0.1),
        "E": (50.0, 0.1),
    }
    assert set(REGIMES) == set(expected)
    for name, (rtt_ms, bandwidth_gbps) in expected.items():
        profile = independent_profile(name)
        assert profile.rtt_ms == rtt_ms
        assert profile.bandwidth_gbps == bandwidth_gbps
        assert profile.name.startswith("independent_machine_")
        assert profile.software_overhead_ms > 0


def test_shard_cache_startup_verifies_hash_and_atomic_marker() -> None:
    receipt = run_shard_cache_startup_self_test()
    assert receipt["status"] == "PASS"
    assert receipt["second_startup_cache_hit_without_fetch"] is True
    assert receipt["corrupted_object_rejected"] is True


def test_independent_controller_holds_all_connections_before_dispatch() -> None:
    receipt = run_control_plane_scale(16, connection_batch=4)
    assert receipt["status"] == "PASS"
    assert receipt["machine_count"] == 16
    assert receipt["compute_workers_per_machine"] == 1
    assert receipt["peak_open_connections"] == 16
    assert receipt["messages"] == 80
    assert receipt["per_expert_rpc"] is False
    assert receipt["same_host_collective"] is False
    assert receipt["native_shard_compute_invoked"] is False
