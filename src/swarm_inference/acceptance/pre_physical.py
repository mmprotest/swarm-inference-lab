"""Software-only gate immediately before operator-owned physical validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from swarm_inference.acceptance.productization import (
    AcceptanceStatus,
    GateSpec,
    ProductizationAcceptanceRunner,
)

PRE_PHYSICAL_GATE_GROUPS: tuple[GateSpec, ...] = (
    GateSpec(
        "model_runtime",
        (
            "tests/unit/test_canonical_model_resolution.py",
            "tests/unit/test_model_architecture_capabilities.py",
            "tests/correctness/test_qwen3_moe_tiny.py",
        ),
        240,
    ),
    GateSpec(
        "engine_planning",
        (
            "tests/unit/test_execution_engine_registry.py",
            "tests/unit/test_installed_engine_manifests.py",
            "tests/unit/test_explain_plan.py",
        ),
        180,
    ),
    GateSpec(
        "architecture_boundary",
        ("tests/unit/test_architecture_boundaries.py",),
        120,
    ),
    GateSpec(
        "network_telemetry",
        (
            "tests/unit/test_tcp_meter.py",
            "tests/unit/test_llamacpp_worker_lifecycle.py",
        ),
        180,
    ),
    GateSpec(
        "secure_wan_transport",
        (
            "tests/integration/test_secure_wan_transport.py",
            "tests/unit/test_network_measurements.py",
        ),
        180,
    ),
    GateSpec(
        "readme_commands",
        ("tests/unit/test_documented_product_commands.py",),
        120,
    ),
    GateSpec(
        "packaging",
        ("tests/integration/test_wheel_install.py",),
        1800,
    ),
)


CHECK_TO_GATE: tuple[tuple[str, str], ...] = (
    ("model resolver", "model_runtime"),
    ("Qwen3 MoE architecture recognition", "model_runtime"),
    ("engine compatibility probes", "model_runtime"),
    ("llama.cpp compatibility preflight", "model_runtime"),
    ("native engine registration", "engine_planning"),
    ("Colibri engine registration", "engine_planning"),
    ("llama.cpp RPC engine registration", "engine_planning"),
    ("no experiment imports in canonical runtime", "architecture_boundary"),
    ("no silent local fallback", "engine_planning"),
    ("distributed plan generation", "engine_planning"),
    ("WAN-aware planner behavior", "model_runtime"),
    ("network telemetry", "network_telemetry"),
    ("encrypted WAN control path", "secure_wan_transport"),
    ("encrypted WAN data path", "secure_wan_transport"),
    ("README command examples", "readme_commands"),
    ("packaging", "packaging"),
)

PHYSICAL_NOT_RUN: tuple[tuple[str, str], ...] = (
    ("two physical hosts", "requires operator hardware"),
    ("heterogeneous CPU/GPU inference", "requires operator hardware"),
    ("real Wi-Fi throughput", "requires physical/network execution"),
    ("real WAN throughput", "requires physical/network execution"),
)


def run_pre_physical_acceptance(
    *,
    repository_root: Path,
    output_root: Path,
) -> int:
    """Run canonical software gates and report physical work honestly."""

    runner = ProductizationAcceptanceRunner(
        repository_root=repository_root,
        output_root=output_root,
    )
    gate_results = {spec.name: runner.run_gate(spec) for spec in PRE_PHYSICAL_GATE_GROUPS}
    software_pass = all(result.status == AcceptanceStatus.PASS for result in gate_results.values())
    checks = []
    for name, gate_name in CHECK_TO_GATE:
        status = gate_results[gate_name].status
        checks.append({"name": name, "status": status.value, "gate": gate_name})
        print(f"{status.value} {name}")
    physical = [
        {"name": name, "status": AcceptanceStatus.NOT_RUN.value, "reason": reason}
        for name, reason in PHYSICAL_NOT_RUN
    ]
    for item in physical:
        print(f"NOT_RUN {item['name']}")
    summary = {
        "document_type": "swarm-pre-physical-acceptance",
        "software_acceptance": "PASS" if software_pass else "FAIL",
        "checks": checks,
        "gate_results": [asdict(item) for item in gate_results.values()],
        "physical_gates": physical,
    }
    summary_path = runner.bundle / "pre-physical-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print(f"SOFTWARE ACCEPTANCE: {'PASS' if software_pass else 'FAIL'}")
    print()
    print("PHYSICAL TWO-MACHINE VALIDATION: NOT_RUN")
    print("Reason: requires operator hardware")
    print()
    print("WAN PERFORMANCE VALIDATION: NOT_RUN")
    print("Reason: requires physical/network execution")
    print(f"acceptance_summary={summary_path}")
    return 0 if software_pass else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pre-physical-acceptance"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    return run_pre_physical_acceptance(
        repository_root=repository_root,
        output_root=args.output,
    )


__all__ = [
    "CHECK_TO_GATE",
    "PHYSICAL_NOT_RUN",
    "PRE_PHYSICAL_GATE_GROUPS",
    "main",
    "run_pre_physical_acceptance",
]
