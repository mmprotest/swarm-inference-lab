from __future__ import annotations

from swarm_inference.acceptance.pre_physical import (
    CHECK_TO_GATE,
    PHYSICAL_NOT_RUN,
    PRE_PHYSICAL_GATE_GROUPS,
)


def test_pre_physical_acceptance_maps_every_required_software_check_to_a_gate() -> None:
    required = {
        "model resolver",
        "Qwen3 MoE architecture recognition",
        "engine compatibility probes",
        "llama.cpp compatibility preflight",
        "native engine registration",
        "Colibri engine registration",
        "llama.cpp RPC engine registration",
        "no experiment imports in canonical runtime",
        "no silent local fallback",
        "distributed plan generation",
        "WAN-aware planner behavior",
        "network telemetry",
        "encrypted WAN control path",
        "encrypted WAN data path",
        "README command examples",
        "packaging",
    }
    groups = {item.name for item in PRE_PHYSICAL_GATE_GROUPS}

    assert {name for name, _gate in CHECK_TO_GATE} == required
    assert {gate for _name, gate in CHECK_TO_GATE} <= groups
    assert all(item.tests for item in PRE_PHYSICAL_GATE_GROUPS)
    assert all("physical" not in " ".join(item.tests) for item in PRE_PHYSICAL_GATE_GROUPS)


def test_pre_physical_acceptance_never_promotes_physical_work_to_pass() -> None:
    assert PHYSICAL_NOT_RUN == (
        ("two physical hosts", "requires operator hardware"),
        ("heterogeneous CPU/GPU inference", "requires operator hardware"),
        ("real Wi-Fi throughput", "requires physical/network execution"),
        ("real WAN throughput", "requires physical/network execution"),
    )
