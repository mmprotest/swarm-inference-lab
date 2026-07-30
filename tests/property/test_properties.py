from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from swarm_inference.config.loader import load_experiment_config
from swarm_inference.coordinator.scheduler import predicted_network_capacity
from swarm_inference.simulation.model import partition_layer_ranges
from swarm_inference.simulation.simulator import Simulator


@given(
    layer_count=st.integers(min_value=1, max_value=256),
    data=st.data(),
)
def test_every_layer_is_assigned_exactly_once(layer_count: int, data) -> None:
    stage_count = data.draw(st.integers(min_value=1, max_value=layer_count))
    ranges = partition_layer_ranges(layer_count, stage_count)
    assigned = [layer for start, end in ranges for layer in range(start, end)]
    assert assigned == list(range(layer_count))


@given(
    capacities=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    ),
    index=st.integers(min_value=0, max_value=31),
    increase=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_increasing_stage_capacity_cannot_reduce_predicted_capacity(
    capacities: list[float],
    index: int,
    increase: float,
) -> None:
    selected = index % len(capacities)
    before = predicted_network_capacity(capacities)
    updated = list(capacities)
    updated[selected] += increase
    assert predicted_network_capacity(updated) >= before


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_same_seed_and_configuration_produce_identical_events(
    seed: int,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(repository_root / "configs/experiments/smoke.yaml")
    config.seed = seed
    left = Simulator(config).run()
    right = Simulator(config).run()
    assert left.deterministic_fingerprint() == right.deterministic_fingerprint()
