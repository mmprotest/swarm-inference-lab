"""Fail-closed Experiment 015 node admission and fleet cost guard."""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "experiment-015-k3-admission-v1"
FIXTURE_SCHEMA_VERSION = "experiment-014-k3-admission-fixture-v1"
GIB = 1024**3


class AdmissionRejected(RuntimeError):
    """A node or fleet failed a preregistered admission gate."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdmissionRejected(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _parse_compute_capability(value: object) -> tuple[int, int] | None:
    parts = str(value).strip().split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def inspect_local_gpu(runtime_path: Path | None, *, device: int = 0) -> dict[str, Any]:
    """Inspect hardware first; initialize CUDA only after the 3090 preflight passes."""
    fields = "index,name,uuid,driver_version,compute_cap,memory.total,memory.free"
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode:
        return {
            "nvidia_smi_status": "UNAVAILABLE",
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
            "cuda_initialization_attempted": False,
        }
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        return {
            "nvidia_smi_status": "UNPARSEABLE",
            "stdout": completed.stdout.strip(),
            "cuda_initialization_attempted": False,
        }
    row = dict(zip(names, values, strict=True))
    result: dict[str, Any] = {
        "nvidia_smi_status": "MEASURED",
        "gpu_index": int(row["index"]),
        "gpu_name": row["name"],
        "gpu_uuid": row["uuid"],
        "driver_version": row["driver_version"],
        "compute_capability": row["compute_cap"],
        "vram_total_bytes": int(float(row["memory.total"]) * 1024**2),
        "vram_free_bytes": int(float(row["memory.free"]) * 1024**2),
        "cuda_initialization_attempted": False,
    }
    capability = _parse_compute_capability(result["compute_capability"])
    hardware_preflight = (
        "RTX 3090" in str(result["gpu_name"])
        and capability == (8, 6)
        and int(result["vram_total_bytes"]) >= 24 * GIB
    )
    result["hardware_preflight_pass"] = hardware_preflight
    if not hardware_preflight or runtime_path is None:
        return result
    library_path = runtime_path.resolve()
    result["runtime_path"] = str(library_path)
    result["runtime_sha256"] = _sha256(library_path)
    result["cuda_initialization_attempted"] = True
    library = ctypes.CDLL(str(library_path))
    library.coli_cuda_init.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.coli_cuda_init.restype = ctypes.c_int
    library.coli_cuda_shutdown.argtypes = []
    library.coli_cuda_shutdown.restype = None
    library.coli_cuda_binary_min_compute_capability.argtypes = []
    library.coli_cuda_binary_min_compute_capability.restype = ctypes.c_int
    library.coli_cuda_binary_has_forward_ptx.argtypes = []
    library.coli_cuda_binary_has_forward_ptx.restype = ctypes.c_int
    library.coli_cuda_binary_accepts_compute_capability.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
    ]
    library.coli_cuda_binary_accepts_compute_capability.restype = ctypes.c_int
    library.coli_cuda_kimi_expert_max_certified_batch.argtypes = []
    library.coli_cuda_kimi_expert_max_certified_batch.restype = ctypes.c_int
    library.coli_cuda_error_state_ok.argtypes = [ctypes.c_int]
    library.coli_cuda_error_state_ok.restype = ctypes.c_int
    devices = (ctypes.c_int * 1)(device)
    initialized = library.coli_cuda_init(devices, 1) == 1
    result["cuda_initialized"] = initialized
    if not initialized:
        return result
    try:
        result.update(
            {
                "binary_min_compute_capability": int(
                    library.coli_cuda_binary_min_compute_capability()
                ),
                "binary_has_forward_ptx": bool(
                    library.coli_cuda_binary_has_forward_ptx()
                ),
                "binary_accepts_sm86": bool(
                    library.coli_cuda_binary_accepts_compute_capability(8, 6)
                ),
                "native_max_certified_expert_batch": int(
                    library.coli_cuda_kimi_expert_max_certified_batch()
                ),
                "cuda_error_state_ok": bool(library.coli_cuda_error_state_ok(device)),
            }
        )
    finally:
        library.coli_cuda_shutdown()
    return result


def evaluate_node_admission(
    observation: dict[str, Any],
    requirements: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Evaluate a measured node without performing any additional GPU work."""
    if mode not in {"PRE_CANARY", "FLEET"}:
        raise ValueError("mode must be PRE_CANARY or FLEET")
    network = observation.get("network", {})
    runtime = observation.get("runtime", {})
    expected_runtime_sha = str(requirements.get("runtime_sha256") or "")
    gates = {
        "nvidia_smi_healthy": observation.get("nvidia_smi_status") == "MEASURED",
        "exact_rtx_3090": "RTX 3090" in str(observation.get("gpu_name", "")),
        "compute_capability_sm86": _parse_compute_capability(
            observation.get("compute_capability")
        )
        == (8, 6),
        "vram_at_least_24_gib": int(observation.get("vram_total_bytes", 0))
        >= int(requirements["minimum_vram_bytes"]),
        "disk_capacity": int(observation.get("disk_free_bytes", 0))
        >= int(requirements["minimum_disk_free_bytes"]),
        "coarse_rtt": float(network.get("rtt_ms", float("inf")))
        <= float(requirements["network"]["maximum_rtt_ms"]),
        "coarse_bandwidth": float(network.get("bandwidth_gbps", 0.0))
        >= float(requirements["network"]["minimum_bandwidth_gbps"]),
        "assignment_hash_exact": str(observation.get("assignment_sha256", ""))
        == str(requirements["assignment_sha256"]),
        "checkpoint_revision_exact": str(observation.get("checkpoint_revision", ""))
        == str(requirements["checkpoint_revision"]),
        "package_version_exact": str(observation.get("package_version", ""))
        == str(requirements["package_version"]),
    }
    if mode == "FLEET":
        gates.update(
            {
                "runtime_hash_exact": bool(expected_runtime_sha)
                and str(runtime.get("sha256", "")) == expected_runtime_sha,
                "cuda_initialized": bool(runtime.get("cuda_initialized")),
                "binary_accepts_sm86": bool(runtime.get("binary_accepts_sm86")),
                "compute86_ptx": bool(runtime.get("binary_has_forward_ptx")),
                "native_batch_at_least_eight": int(
                    runtime.get("native_max_certified_expert_batch", 0)
                )
                >= 8,
                "safe_fixture_pass": bool(runtime.get("safe_fixture_pass")),
                "assigned_stage_canary_pass": bool(
                    runtime.get("assigned_stage_canary_pass")
                ),
                "prepare_seven_calls_pass": bool(
                    runtime.get("prepare_seven_calls_pass")
                ),
            }
        )
    status = "PASS" if all(gates.values()) else "REJECT"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "node_admission",
        "mode": mode,
        "status": status,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "cuda_initialization_attempted": bool(
            observation.get("cuda_initialization_attempted", False)
        ),
    }


def write_worker_requirements(
    placement_path: Path,
    distribution_path: Path,
    worker_id: str,
    output_path: Path,
    *,
    package_version: str,
    runtime_sha256: str = "",
) -> dict[str, Any]:
    placement = _read(placement_path.resolve())
    distribution = _read(distribution_path.resolve())
    worker = next(
        (row for row in placement["workers"] if row["worker_id"] == worker_id),
        None,
    )
    download = next(
        (row for row in distribution["workers"] if row["worker_id"] == worker_id),
        None,
    )
    if worker is None or download is None:
        raise AdmissionRejected(f"manifests do not both contain {worker_id!r}")
    if (
        placement.get("status") != "PASS"
        or distribution.get("status") != "PASS"
        or str(download["checkpoint_fingerprint"])
        != str(placement["checkpoint"]["checkpoint_fingerprint"])
    ):
        raise AdmissionRejected("placement/distribution identity is not passing")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "worker_requirements",
        "status": "PASS",
        "worker_id": worker_id,
        "minimum_vram_bytes": int(worker["memory"]["physical_vram_bytes"]),
        "minimum_disk_free_bytes": int(download["temporary_disk_bytes"]) + 32 * GIB,
        "network": placement["network_admission"],
        "assignment_sha256": worker["assignment_sha256"],
        "checkpoint_revision": placement["checkpoint"]["revision"],
        "checkpoint_fingerprint": placement["checkpoint"]["checkpoint_fingerprint"],
        "package_version": package_version,
        "runtime_sha256": runtime_sha256,
        "owned_layers": worker["owned_layers"],
        "worker_role": worker["worker_role"],
        "source_weight_bytes": worker["source_weight_bytes"],
        "cold_download_bytes": download["download_bytes_cold_cache"],
        "placement_manifest_sha256": _sha256(placement_path.resolve()),
        "distribution_manifest_sha256": _sha256(distribution_path.resolve()),
    }
    _atomic_json(output_path, receipt)
    return receipt


def write_local_node_observation(
    output_path: Path,
    *,
    runtime_path: Path | None,
    network_rtt_ms: float,
    network_bandwidth_gbps: float,
    assignment_sha256: str,
    checkpoint_revision: str,
    package_version: str,
    disk_path: Path,
    device: int = 0,
    assigned_stage_canary_path: Path | None = None,
) -> dict[str, Any]:
    observation = inspect_local_gpu(runtime_path, device=device)
    observation.update(
        {
            "network": {
                "rtt_ms": network_rtt_ms,
                "bandwidth_gbps": network_bandwidth_gbps,
                "source": "measured external probe supplied to bootstrap",
            },
            "assignment_sha256": assignment_sha256,
            "checkpoint_revision": checkpoint_revision,
            "package_version": package_version,
            "disk_free_bytes": shutil.disk_usage(disk_path.resolve()).free,
            "disk_path": str(disk_path.resolve()),
        }
    )
    if runtime_path is not None and observation.get("cuda_initialized"):
        observation["runtime"] = {
            "sha256": observation.get("runtime_sha256"),
            "cuda_initialized": observation.get("cuda_initialized"),
            "binary_accepts_sm86": observation.get("binary_accepts_sm86"),
            "binary_has_forward_ptx": observation.get("binary_has_forward_ptx"),
            "native_max_certified_expert_batch": observation.get(
                "native_max_certified_expert_batch"
            ),
            "cuda_error_state_ok": observation.get("cuda_error_state_ok"),
            "safe_fixture_pass": False,
            "assigned_stage_canary_pass": False,
            "prepare_seven_calls_pass": False,
        }
        if assigned_stage_canary_path is not None:
            canary_source = assigned_stage_canary_path.resolve()
            canary = _read(canary_source)
            assigned = canary.get("assigned_stage", {})
            gates = assigned.get("acceptance_gates", {})
            if (
                canary.get("status") != "PASS"
                or canary.get("runtime", {}).get("sha256")
                != observation["runtime"]["sha256"]
                or canary.get("worker_id") is None
            ):
                raise AdmissionRejected("assigned-stage canary identity is not passing")
            observation["runtime"].update(
                {
                    "safe_fixture_pass": bool(gates.get("post_guard_safe_fixture")),
                    "assigned_stage_canary_pass": bool(
                        canary.get("acceptance_gates", {}).get("assigned_stage_pass")
                    ),
                    "prepare_seven_calls_pass": bool(
                        gates.get("prepare_exactly_seven_calls")
                    ),
                    "assigned_stage_canary_sha256": _sha256(canary_source),
                }
            )
    _atomic_json(output_path, observation)
    return observation


def evaluate_fleet_cost_guard(
    fleet: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    expected = int(policy["expected_nodes"])
    connected = int(fleet.get("connected_nodes", 0))
    ready = int(fleet.get("ready_nodes", 0))
    price = float(fleet.get("gpu_price_per_hour_usd", float("inf")))
    hourly = connected * price
    gates = {
        "expected_node_count": int(fleet.get("expected_nodes", -1)) == expected,
        "all_expected_connected": connected == expected,
        "all_expected_ready": ready == expected,
        "no_unhealthy": int(fleet.get("unhealthy_nodes", -1)) == 0,
        "no_unused": int(fleet.get("unused_nodes", -1)) == 0,
        "no_wrong_hardware": int(fleet.get("wrong_hardware_nodes", -1)) == 0,
        "no_wrong_network": int(fleet.get("wrong_network_nodes", -1)) == 0,
        "all_shards_exact": bool(fleet.get("all_shards_exact")),
        "topology_exact": bool(fleet.get("topology_exact")),
        "single_3090_canary_pass": bool(fleet.get("single_3090_canary_pass")),
        "linux_runtime_qualification_pass": bool(
            fleet.get("linux_runtime_qualification_pass")
        ),
        "gpu_price_within_economic_gate": price
        <= float(policy["maximum_gpu_price_per_hour_usd"]),
        "fleet_hourly_cost_within_gate": hourly
        <= float(policy["maximum_fleet_cost_per_hour_usd"]),
    }
    if bool(policy.get("sub_layer_canary_required")):
        gates["sub_layer_canary_pass"] = bool(fleet.get("sub_layer_canary_pass"))
    status = "ALLOW" if all(gates.values()) else "ABORT"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "fleet_cost_guard",
        "status": status,
        "expected_nodes": expected,
        "connected_nodes": connected,
        "ready_nodes": ready,
        "hourly_cost_usd": hourly,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }


def run_admission_fixture(
    output_path: Path,
    *,
    cycle_id: str = "H014-037b",
) -> dict[str, Any]:
    assignment = "a" * 64
    runtime_sha = "b" * 64
    requirements = {
        "minimum_vram_bytes": 24 * GIB,
        "minimum_disk_free_bytes": 32 * GIB,
        "network": {"maximum_rtt_ms": 5.0, "minimum_bandwidth_gbps": 10.0},
        "assignment_sha256": assignment,
        "checkpoint_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "package_version": "0.1.0rc11",
        "runtime_sha256": runtime_sha,
    }
    good_observation = {
        "nvidia_smi_status": "MEASURED",
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "compute_capability": "8.6",
        "vram_total_bytes": 24 * GIB,
        "disk_free_bytes": 64 * GIB,
        "network": {"rtt_ms": 4.9, "bandwidth_gbps": 10.1},
        "assignment_sha256": assignment,
        "checkpoint_revision": requirements["checkpoint_revision"],
        "package_version": requirements["package_version"],
        "cuda_initialization_attempted": True,
        "runtime": {
            "sha256": runtime_sha,
            "cuda_initialized": True,
            "binary_accepts_sm86": True,
            "binary_has_forward_ptx": True,
            "native_max_certified_expert_batch": 16,
            "safe_fixture_pass": True,
            "assigned_stage_canary_pass": True,
            "prepare_seven_calls_pass": True,
        },
    }
    node_pass = evaluate_node_admission(good_observation, requirements, mode="FLEET")
    wrong_gpu_observation = {**good_observation, "gpu_name": "NVIDIA RTX 4090"}
    wrong_gpu_observation["cuda_initialization_attempted"] = False
    wrong_gpu = evaluate_node_admission(
        wrong_gpu_observation, requirements, mode="PRE_CANARY"
    )
    wrong_network_observation = {
        **good_observation,
        "network": {"rtt_ms": 5.1, "bandwidth_gbps": 9.9},
    }
    wrong_network = evaluate_node_admission(
        wrong_network_observation, requirements, mode="PRE_CANARY"
    )
    policy = {
        "expected_nodes": 93,
        "maximum_gpu_price_per_hour_usd": 0.05,
        "maximum_fleet_cost_per_hour_usd": 4.65,
        "sub_layer_canary_required": False,
    }
    base_fleet = {
        "expected_nodes": 93,
        "connected_nodes": 93,
        "ready_nodes": 93,
        "unhealthy_nodes": 0,
        "unused_nodes": 0,
        "wrong_hardware_nodes": 0,
        "wrong_network_nodes": 0,
        "all_shards_exact": True,
        "topology_exact": True,
        "single_3090_canary_pass": True,
        "linux_runtime_qualification_pass": True,
    }
    viable_fleet = evaluate_fleet_cost_guard(
        {**base_fleet, "gpu_price_per_hour_usd": 0.05}, policy
    )
    nominal_fleet = evaluate_fleet_cost_guard(
        {**base_fleet, "gpu_price_per_hour_usd": 0.165}, policy
    )
    missing_worker = evaluate_fleet_cost_guard(
        {
            **base_fleet,
            "connected_nodes": 92,
            "ready_nodes": 92,
            "gpu_price_per_hour_usd": 0.05,
        },
        policy,
    )
    unused_worker = evaluate_fleet_cost_guard(
        {
            **base_fleet,
            "unused_nodes": 1,
            "gpu_price_per_hour_usd": 0.05,
        },
        policy,
    )
    gates = {
        "valid_node_passes": node_pass["status"] == "PASS",
        "wrong_gpu_rejected_before_cuda": wrong_gpu["status"] == "REJECT"
        and not wrong_gpu["cuda_initialization_attempted"],
        "wrong_network_rejected": wrong_network["status"] == "REJECT",
        "viable_price_fleet_allowed": viable_fleet["status"] == "ALLOW",
        "nominal_price_fleet_aborted": nominal_fleet["status"] == "ABORT",
        "missing_worker_fleet_aborted": missing_worker["status"] == "ABORT",
        "unused_worker_fleet_aborted": unused_worker["status"] == "ABORT",
    }
    receipt = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "hypothesis": (
            "Hardware/network mismatches reject before unsafe CUDA, and the full-fleet "
            "guard permits only an exact healthy 93-worker fleet at or below the "
            "measured USD 0.05/GPU-hour economic ceiling."
        ),
        "requirements": requirements,
        "policy": policy,
        "cases": {
            "valid_node": node_pass,
            "wrong_gpu": wrong_gpu,
            "wrong_network": wrong_network,
            "viable_fleet": viable_fleet,
            "nominal_0_165_fleet": nominal_fleet,
            "missing_worker": missing_worker,
            "unused_worker": unused_worker,
        },
        "acceptance_gates": gates,
        "decision": (
            "RETAIN_FAIL_CLOSED_ADMISSION_AND_COST_GUARD"
            if all(gates.values())
            else "REDESIGN_ADMISSION"
        ),
    }
    _atomic_json(output_path, receipt)
    return receipt


def run_node_admission(
    observation_path: Path,
    requirements_path: Path,
    output_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    receipt = evaluate_node_admission(
        _read(observation_path), _read(requirements_path), mode=mode
    )
    _atomic_json(output_path, receipt)
    return receipt


def run_fleet_cost_guard(
    fleet_path: Path, policy_path: Path, output_path: Path
) -> dict[str, Any]:
    try:
        receipt = evaluate_fleet_cost_guard(_read(fleet_path), _read(policy_path))
    except BaseException as exc:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "kind": "fleet_cost_guard",
            "status": "ABORT",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    _atomic_json(output_path, receipt)
    return receipt


__all__ = [
    "AdmissionRejected",
    "evaluate_fleet_cost_guard",
    "evaluate_node_admission",
    "inspect_local_gpu",
    "run_admission_fixture",
    "run_fleet_cost_guard",
    "run_node_admission",
    "write_local_node_observation",
    "write_worker_requirements",
]
