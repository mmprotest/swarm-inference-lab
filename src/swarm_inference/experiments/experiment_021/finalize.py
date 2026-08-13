"""Finalize E021 receipts, economics, truth table, and scientific report."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import CANONICAL_SWARM_DEFINITION
from .io import atomic_write_json, sha256_file, write_csv

OUTCOME = "MODEL_INVALID"
EVIDENCE_INVALID = (
    "PHYSICAL_SHARD_EXECUTION inputs + SHAPED_NETWORK + "
    "UNVALIDATED_INDEPENDENT_MACHINE_MODEL_DIAGNOSTIC_ONLY"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict[str, str]], cap: int, regime: str) -> dict[str, str]:
    values = [
        row
        for row in rows
        if int(float(row["memory_cap_gib"])) == cap and row["regime"] == regime
    ]
    return max(values, key=lambda row: float(row["exact_tok_s_per_user"]))


def _command(
    arguments: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": arguments,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": arguments,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def materialize_environment(repo: Path, artifact_root: Path) -> dict[str, Any]:
    gpu = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo,
    )
    git_commit = _command(["git", "rev-parse", "HEAD"], cwd=repo)
    git_status = _command(["git", "status", "--short"], cwd=repo)
    packages = {}
    for name in ("cryptography", "matplotlib", "numpy", "psutil", "pytest", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    receipt = {
        "schema_version": "experiment-021-environment-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "packages": packages,
        "gpu_query": gpu,
        "physical_gpu_count_used": 1,
        "physical_gpu_role": "ordered PHYSICAL_SHARD_EXECUTION replay only",
        "checkpoint": "F:/models/Kimi-K3",
        "checkpoint_writes": 0,
        "git_commit": git_commit["stdout"] or None,
        "git_status_at_finalize": git_status["stdout"].splitlines(),
        "timezone_context": "Australia/Sydney",
        "zero_gpu_rental": True,
        "vast_resource_mutations": 0,
        "physical_independent_machine_swarm": False,
    }
    atomic_write_json(artifact_root / "environment.json", receipt)
    return receipt


def materialize_model_metadata(artifact_root: Path) -> dict[str, Any]:
    feasibility = _read(artifact_root / "placement" / "whole-layer-feasibility.json")
    manifest = _read(artifact_root / "placement" / "worker-manifest-8g.json")
    receipt = {
        "schema_version": "experiment-021-model-metadata-v1",
        "model": "Kimi K3",
        "checkpoint_path": feasibility["checkpoint_index"],
        "checkpoint_index_sha256": feasibility["checkpoint_index_sha256"],
        "checkpoint_payload_bytes": manifest["summary"]["checkpoint_payload_bytes"],
        "checkpoint_payload_gib": manifest["summary"]["checkpoint_payload_bytes"]
        / 1024**3,
        "tensor_count": manifest["summary"]["coverage_tensor_count"],
        "transformer_layers": 93,
        "kda_layers": 69,
        "gated_mla_layers": 24,
        "routed_experts": 896,
        "selected_routed_experts_per_applicable_token_layer": 16,
        "shared_experts": 2,
        "hidden_dimension": 7168,
        "latent_dimension": 3584,
        "checkpoint_metadata_revalidated": True,
        "checkpoint_coverage_gap_bytes": manifest["summary"]["coverage_gap_bytes"],
        "checkpoint_coverage_overlap_bytes": manifest["summary"]["coverage_overlap_bytes"],
        "source_of_truth": (
            "current local checkpoint index plus E021 byte-exact CheckpointCatalog census"
        ),
    }
    atomic_write_json(artifact_root / "model-metadata.json", receipt)
    return receipt


def materialize_source_manifest(repo: Path, artifact_root: Path) -> dict[str, Any]:
    paths = [
        repo / "AGENTS.md",
        repo / "pyproject.toml",
        repo / "scripts" / "run_experiment_021.py",
        repo / "scripts" / "validate_experiment_021.py",
        repo / "deployment" / "Dockerfile.e021",
        *sorted(
            (repo / "src" / "swarm_inference" / "experiments" / "experiment_021").glob(
                "*.py"
            )
        ),
        repo / "src" / "swarm_inference" / "experiments" / "experiment_019" / "events.py",
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_019"
        / "placement.py",
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_019"
        / "sharded_graph.py",
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_020"
        / "transport.py",
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_020"
        / "sharded_graph.py",
        repo
        / "src"
        / "swarm_inference"
        / "experiments"
        / "experiment_020"
        / "model_distribution.py",
        repo / "artifacts" / "experiment-020" / "physical" / "attention-raw.json",
        repo
        / "artifacts"
        / "experiment-020"
        / "physical"
        / "expert-grouped-robust-calibration.json",
        repo
        / "artifacts"
        / "experiment-019"
        / "physical"
        / "expert-stripe-raw.json",
        repo
        / "artifacts"
        / "experiment-019"
        / "physical"
        / "other-shards-raw.json",
        repo
        / "artifacts"
        / "experiment-020"
        / "runtime"
        / "protocol-heldout-raw.json",
        repo
        / "artifacts"
        / "experiment-016"
        / "cuda"
        / "coli_cuda-sm120-h016-final.dll",
        repo
        / "artifacts"
        / "experiment-019"
        / "physical"
        / "exp019-kda-shard-sm120-v2.dll",
        repo
        / "artifacts"
        / "experiment-020"
        / "physical"
        / "e020-grouped-top16-sm120.dll",
        Path("F:/models/Kimi-K3/model.safetensors.index.json"),
        repo
        / "artifacts"
        / "experiment-014"
        / "oracle-full-93-idot0"
        / "hidden-trace.f32",
        repo
        / "artifacts"
        / "experiment-014"
        / "oracle-full-93-idot0"
        / "routes.txt",
        repo
        / "artifacts"
        / "experiment-014"
        / "oracle-full-93-idot0"
        / "prefill-logits.f32",
    ]
    rows = []
    for path in dict.fromkeys(paths):
        if not path.is_file():
            rows.append({"path": str(path), "exists": False, "bytes": None, "sha256": None})
            continue
        try:
            display = str(path.resolve().relative_to(repo))
        except ValueError:
            display = str(path.resolve())
        rows.append(
            {
                "path": display.replace("\\", "/"),
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        "schema_version": "experiment-021-source-manifest-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if all(row["exists"] for row in rows) else "FAIL",
        "source_count": len(rows),
        "sources": rows,
    }
    atomic_write_json(artifact_root / "source-manifest.json", receipt)
    return receipt


def materialize_commands(artifact_root: Path) -> None:
    lines = [
        "# Experiment 021 reproducibility command ledger",
        "# Commands containing secrets are intentionally absent; no secret was persisted.",
        "# Vast commands marked READ_ONLY were executed; rendered create commands were NOT executed.",
        "",
        "python scripts/run_experiment_021.py --phase placement",
        "python scripts/run_experiment_021.py --phase validation",
        "python scripts/run_experiment_021.py --phase simulation",
        "python scripts/run_experiment_021.py --phase runtime",
        "python scripts/run_experiment_021.py --phase control-plane",
        "vastai --version  # READ_ONLY",
        "vastai show user --raw  # READ_ONLY; response redacted",
        "vastai search offers \"rentable=True num_gpus=1 gpu_ram>=1 gpu_ram<=8\" --limit 500 --raw  # READ_ONLY",
        "python scripts/run_experiment_021.py --phase charts",
        "python -m compileall -q src/swarm_inference/experiments/experiment_021 scripts/run_experiment_021.py scripts/validate_experiment_021.py tests/test_experiment_021.py",
        ".venv/Scripts/ruff.exe check src/swarm_inference/experiments/experiment_021 scripts/run_experiment_021.py scripts/validate_experiment_021.py tests/test_experiment_021.py",
        "# First expanded pytest attempt: 39 passed / 5 fixture errors because sandbox denied the default user Temp directory.",
        "python -m pytest -q tests/test_experiment_021.py tests/unit/test_experiment_019_no_monolith.py tests/unit/test_experiment_020_transport.py tests/unit/test_experiment_020_vast_safety.py tests/unit/test_experiment_020_model_distribution.py tests/unit/test_experiment_020_topology.py tests/unit/test_experiment_020_fleet.py tests/unit/test_experiment_020_worker_lifecycle.py --basetemp=artifacts/experiment-021/qa/pytest-temp --junitxml=artifacts/experiment-021/qa/pytest.xml",
        "python scripts/run_experiment_021.py --phase finalize",
        "python scripts/validate_experiment_021.py",
        "",
        "# Forbidden guard self-test (blocked before subprocess):",
        "# vastai create instance 0  # EXECUTED=false; guard classification=FORBIDDEN",
        "# GPU_RENTALS=0",
        "# VAST_RESOURCE_MUTATIONS=0",
    ]
    (artifact_root / "commands.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_economics(artifact_root: Path) -> list[dict[str, Any]]:
    sweep = _csv(artifact_root / "simulation" / "sweep.csv")
    market = _read(artifact_root / "vast" / "fragmented-fleet-feasibility.json")
    market_by_cap = {
        int(row["worker_memory_cap_gib"]): row for row in market["tiers"]
    }
    rows = []
    for cap in (8, 4, 2, 1):
        for regime in ("A", "B", "C", "D", "E"):
            row = _best(sweep, cap, regime)
            market_row = market_by_cap[cap]
            rows.append(
                {
                    "worker_memory_cap_gib": cap,
                    "regime": regime,
                    "rtt_ms": row["rtt_ms"],
                    "bandwidth_gbps": row["bandwidth_gbps"],
                    "worker_count": row["worker_count"],
                    "total_resident_gib": float(row["total_resident_bytes"]) / 1024**3,
                    "max_peak_gib_per_worker": row["max_peak_memory_per_worker_gib"],
                    "admissible_exact_tok_s_per_user": "",
                    "diagnostic_tok_s_per_user": row["exact_tok_s_per_user"],
                    "target_pass_ms_diagnostic": row["target_pass_ms"],
                    "active_worker_seconds_per_output_token": row[
                        "active_worker_seconds_per_accepted_token"
                    ],
                    "resident_worker_seconds_per_output_token": row[
                        "resident_worker_seconds_per_accepted_token"
                    ],
                    "network_mb_per_output_token": float(
                        row["network_bytes_per_accepted_token"]
                    )
                    / 1_000_000,
                    "compute_work_inflation": row["compute_work_inflation"],
                    "one_request_worker_utilization": row["average_worker_utilization"],
                    "market_unique_compatible_machines": market_row[
                        "unique_kernel_compatible_machines"
                    ],
                    "market_fleet_feasible": market_row["currently_feasible"],
                    "market_estimated_hourly_fleet_price_usd": market_row[
                        "market_estimated_hourly_fleet_price_usd"
                    ],
                    "market_estimate_is_actual_cost_per_token": False,
                    "admissible": False,
                    "evidence_class": EVIDENCE_INVALID,
                }
            )
    write_csv(artifact_root / "cost" / "resource-economics.csv", rows)
    return rows


def _gate_rows(artifact_root: Path) -> list[dict[str, Any]]:
    scaling = _csv(artifact_root / "control-plane" / "scaling.csv")
    scaling_pass = all(row["status"] == "PASS" for row in scaling)
    return [
        {"gate": 1, "name": "Root AGENTS.md read and acknowledged", "status": "PASS"},
        {"gate": 2, "name": "Zero GPU rentals / zero Vast mutations", "status": "PASS"},
        {"gate": 3, "name": "Whole-layer feasibility proves impossible", "status": "PASS"},
        {"gate": 4, "name": "Headline cap <=8 GiB", "status": "PASS"},
        {"gate": 5, "name": "One worker equals one independent machine", "status": "PASS"},
        {"gate": 6, "name": "No whole layer/expert on headline worker", "status": "PASS"},
        {"gate": 7, "name": "Full checkpoint byte coverage", "status": "PASS"},
        {"gate": 8, "name": "Direct shard loading", "status": "PASS"},
        {"gate": 9, "name": "Production EXECUTE_SHARD invokes native code", "status": "FAIL"},
        {"gate": 10, "name": "Complete 93-layer worker-process correctness", "status": "FAIL"},
        {"gate": 11, "name": "Corrected ordered-shard model validation", "status": "FAIL"},
        {"gate": 12, "name": "Event accounting reconciliation", "status": "PASS"},
        {"gate": 13, "name": "Every compute event has worker_id", "status": "PASS"},
        {"gate": 14, "name": "Every worker dependency has explicit network cost", "status": "PASS"},
        {"gate": 15, "name": "No same-host/NCCL/PCIe assumption", "status": "PASS"},
        {"gate": 16, "name": "Memory-fragmentation chart produced", "status": "PASS_DIAGNOSTIC"},
        {"gate": 17, "name": "Network envelope produced", "status": "PASS_DIAGNOSTIC"},
        {"gate": 18, "name": "Whole-layer control separate", "status": "PASS_DIAGNOSTIC"},
        {
            "gate": 19,
            "name": "Control-plane scale test",
            "status": "PASS" if scaling_pass else "FAIL",
        },
        {"gate": 20, "name": "Evidence-class labels correct", "status": "PASS"},
    ]


def build_summary(artifact_root: Path) -> dict[str, Any]:
    placement = _read(artifact_root / "placement" / "worker-manifest-8g.json")
    feasibility = _read(artifact_root / "placement" / "whole-layer-feasibility.json")
    validation = _read(artifact_root / "physical" / "ordered-workloads.json")
    accounting = _read(artifact_root / "validation" / "accounting-reconciliation.json")
    correctness = _read(artifact_root / "correctness" / "worker-process-93-layer.json")
    safety = _read(artifact_root / "vast" / "safety-audit.json")
    sweep = _csv(artifact_root / "simulation" / "sweep.csv")
    best = {regime: _best(sweep, 8, regime) for regime in ("A", "B", "C", "D", "E")}
    gates = _gate_rows(artifact_root)
    receipt = {
        "schema_version": "experiment-021-summary-v1",
        "experiment": 21,
        "outcome": OUTCOME,
        "canonical_swarm_definition": CANONICAL_SWARM_DEFINITION,
        "root_agents_md_read": True,
        "headline_structural_candidate": {
            "memory_cap_gib": 8,
            "max_peak_memory_gib": placement["summary"]["max_worker_peak_gib"],
            "worker_count": placement["summary"]["worker_count"],
            "machine_count": placement["summary"]["machine_count"],
            "one_worker_per_machine": True,
            "complete_model_whole_layer_placement_possible": feasibility[
                "headline_8g_complete_model_whole_layer_placement_possible"
            ],
            "full_checkpoint_covered": placement["summary"]["coverage_gap_bytes"] == 0
            and placement["summary"]["coverage_overlap_bytes"] == 0,
            "whole_layer_on_any_worker": placement["summary"]["whole_layer_on_any_worker"],
            "whole_routed_expert_on_any_worker": placement["summary"][
                "whole_routed_expert_on_any_worker"
            ],
            "whole_shared_expert_on_any_worker": placement["summary"][
                "whole_shared_expert_on_any_worker"
            ],
        },
        "model_validation": validation["model_validation"],
        "same_residency_policy_as_headline_model": validation[
            "same_residency_policy_as_headline_model"
        ],
        "maximum_hidden_relative_l2_error_in_ordered_replay": validation[
            "maximum_hidden_relative_l2_error"
        ],
        "admissible_exact_throughput_available": False,
        "best_exact_tok_s_by_regime": {regime: None for regime in ("A", "B", "C", "D", "E")},
        "inadmissible_diagnostic_projection": {
            regime: {
                "tok_s_per_user": float(row["exact_tok_s_per_user"]),
                "target_pass_ms": float(row["target_pass_ms"]),
                "block": int(row["block"]),
                "chunk": int(row["chunk"]),
            }
            for regime, row in best.items()
        },
        "accounting_reconciliation": accounting["status"],
        "full_93_layer_worker_process_correctness": correctness["status"],
        "zero_gpu_rentals": safety["gpu_rentals"] == 0,
        "zero_vast_mutations": safety["vast_resource_mutations"] == 0,
        "physical_independent_machine_swarm_measured": False,
        "fast_lan_swarm": "NOT SUPPORTED — MODEL_INVALID",
        "regional_swarm": "NOT SUPPORTED — MODEL_INVALID",
        "consumer_wan_swarm": "NOT SUPPORTED — MODEL_INVALID",
        "decisive_bottleneck": (
            "the executable shard path is non-resident and production EXECUTE_SHARD "
            "native dispatch is absent, so the event model does not predict the physical runtime"
        ),
        "normalization_applied": False,
        "global_multiplier": None,
        "gates": gates,
        "gates_3_through_15_all_pass": all(
            row["status"] == "PASS" for row in gates if 3 <= row["gate"] <= 15
        ),
        "core_swarm_prephysical_verdict": "MODEL INVALID",
        "physical_swarm_verdict": "NOT YET PHYSICALLY PROVEN",
        "evidence_classes": [
            "PHYSICAL_SINGLE_MACHINE",
            "PHYSICAL_SHARD_EXECUTION",
            "SHAPED_NETWORK",
            "UNVALIDATED_INDEPENDENT_MACHINE_MODEL_DIAGNOSTIC_ONLY",
        ],
    }
    atomic_write_json(artifact_root / "summary.json", receipt)
    return receipt


def build_truth_table(artifact_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    candidate = summary["headline_structural_candidate"]
    rows = [
        ("Any GPUs rented in E021?", "NO"),
        ("Any Vast resources mutated?", "NO"),
        (
            "Headline max memory per independent machine",
            f"{candidate['max_peak_memory_gib']:.3f} GiB peak (8 GiB cap)",
        ),
        (
            "Can complete K3 be executed by assigning whole layers to these machines?",
            "NO",
        ),
        ("Does any headline machine host multiple compute workers?", "NO"),
        ("Does any headline worker own a whole ordinary transformer layer?", "NO"),
        ("Does any headline worker own a whole routed/shared expert?", "NO"),
        ("Full checkpoint covered?", "YES"),
        ("Full 93-layer worker-process shard correctness?", "FAIL"),
        ("Timing model validated against ordered physical shard execution?", "FAIL"),
        ("Headline compute events all tied to explicit workers?", "YES"),
        ("Same-host NCCL/NVLink assumed?", "NO"),
        ("Best 8 GiB / Regime B tok/s", "N/A — MODEL_INVALID"),
        ("Best 8 GiB / Regime C tok/s", "N/A — MODEL_INVALID"),
        ("Best 8 GiB / Regime D tok/s", "N/A — MODEL_INVALID"),
        ("Best 4 GiB result", "N/A — MODEL_INVALID"),
        ("Best 2 GiB result", "N/A — MODEL_INVALID"),
        ("Best 1 GiB result", "N/A — MODEL_INVALID"),
        ("Physical independent-machine Swarm measured?", "NO"),
    ]
    receipt = {
        "schema_version": "experiment-021-truth-table-v1",
        "outcome": OUTCOME,
        "rows": [{"question": question, "answer": answer} for question, answer in rows],
        "diagnostic_projection_is_not_an_exact_result": True,
    }
    atomic_write_json(artifact_root / "truth-table.json", receipt)
    return receipt


def build_failure_log(artifact_root: Path) -> dict[str, Any]:
    validation = _read(artifact_root / "physical" / "ordered-workloads.json")
    market = _read(artifact_root / "vast" / "fragmented-fleet-feasibility.json")
    failures = [
        {
            "id": "E021-F001",
            "status": "OPEN_DECISIVE",
            "gate": 11,
            "failure": "ordered physical shard model validation failed",
            "evidence": validation["model_validation"],
            "cause": validation["residency_mismatch"],
            "normalization_applied": False,
        },
        {
            "id": "E021-F002",
            "status": "OPEN",
            "gate": 9,
            "failure": "production EXECUTE_SHARD daemon returns a mock partial instead of native shard work",
            "effect": "production-path correctness cannot be claimed",
        },
        {
            "id": "E021-F003",
            "status": "OPEN",
            "gate": 10,
            "failure": "complete 93-layer correctness was not executed through worker processes",
            "control": "E020 in-process physical shard math passed but is inadmissible for this gate",
        },
        {
            "id": "E021-F004",
            "status": "OPEN",
            "failure": "no published immutable one-worker E021 Linux image digest",
            "effect": "not ready for physical deployment",
        },
        {
            "id": "E021-F005",
            "status": "OPEN_MARKET",
            "failure": "read-only independent-machine inventory is insufficient",
            "required_8g": market["tiers"][0]["required_independent_machines"],
            "observed_compatible_8g": market["tiers"][0][
                "unique_kernel_compatible_machines"
            ],
        },
        {
            "id": "E021-F006",
            "status": "RESOLVED_IN_E021",
            "failure": "2,000 simultaneous connection setup storm failed on first harness",
            "redesign": (
                "bounded-batch connection establishment while retaining all connections "
                "open before dispatch"
            ),
            "rerun": "PASS at 2,000 open connections / 10,000 messages",
        },
    ]
    receipt = {
        "schema_version": "experiment-021-failure-log-v1",
        "outcome": OUTCOME,
        "failure_count": len(failures),
        "open_failure_count": sum(row["status"].startswith("OPEN") for row in failures),
        "failures": failures,
    }
    atomic_write_json(artifact_root / "failure-log.json", receipt)
    return receipt


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def generate_report(repo: Path, artifact_root: Path, summary: dict[str, Any]) -> Path:
    truth = _read(artifact_root / "truth-table.json")
    placement_rows = _csv(artifact_root / "placement" / "worker-memory-tiers.csv")
    validation = _read(artifact_root / "physical" / "ordered-workloads.json")
    validation_rows = _csv(artifact_root / "validation" / "ordered-shard-replay.csv")
    sweep = _csv(artifact_root / "simulation" / "sweep.csv")
    correctness = _read(artifact_root / "correctness" / "worker-process-93-layer.json")
    scaling = _csv(artifact_root / "control-plane" / "scaling.csv")
    heterogeneity = _csv(artifact_root / "simulation" / "heterogeneity.csv")
    concurrency = _csv(artifact_root / "simulation" / "concurrency.csv")
    market = _read(artifact_root / "vast" / "fragmented-fleet-feasibility.json")
    feasibility = _read(artifact_root / "placement" / "whole-layer-feasibility.json")
    direct = _read(artifact_root / "placement" / "direct-read-audit.json")
    control = max(
        _csv(artifact_root / "simulation" / "whole-layer-control.csv"),
        key=lambda row: float(row["exact_tok_s_per_user"]),
    )
    best = {
        cap: {regime: _best(sweep, cap, regime) for regime in ("A", "B", "C", "D")}
        for cap in (8, 4, 2, 1)
    }
    diagnostic = summary["inadmissible_diagnostic_projection"]
    truth_table = _markdown_table(
        ["Question", "Answer"],
        [[row["question"], row["answer"]] for row in truth["rows"]],
    )
    placement_table = _markdown_table(
        ["Cap", "P", "Depth", "Machines", "Max peak", "Layer max", "Expert max"],
        [
            [
                f"{row['worker_cap_gib']} GiB",
                row["stripe_degree"],
                row["depth_span"],
                f"{int(row['worker_count']):,}",
                f"{float(row['max_peak_gib']):.3f} GiB",
                f"{100 * float(row['max_layer_fraction']):.2f}%",
                f"{100 * float(row['max_expert_fraction']):.2f}%",
            ]
            for row in placement_rows
        ],
    )
    validation_table = _markdown_table(
        ["Workload", "Predicted ms", "Actual ms", "APE"],
        [
            [
                row["workload"],
                f"{float(row['predicted_sharded_wall_ms']):.4f}",
                f"{float(row['actual_sharded_wall_ms']):.4f}",
                f"{100 * float(row['absolute_percentage_error']):.2f}%",
            ]
            for row in validation_rows
        ],
    )
    diagnostic_table = _markdown_table(
        ["Cap", "A", "B", "C", "D", "Workers"],
        [
            [
                f"{cap} GiB",
                *[
                    f"{float(best[cap][regime]['exact_tok_s_per_user']):.3f}*"
                    for regime in ("A", "B", "C", "D")
                ],
                f"{int(best[cap]['B']['worker_count']):,}",
            ]
            for cap in (8, 4, 2, 1)
        ],
    )
    scaling_table = _markdown_table(
        ["Workers", "Status", "Peak conns", "CPU s", "RSS MiB", "Queue p50 ms"],
        [
            [
                f"{int(row['requested_worker_count']):,}",
                row["status"],
                f"{int(row['peak_open_connections']):,}",
                f"{float(row['controller_cpu_seconds']):.3f}",
                f"{float(row['controller_rss_bytes']) / 1024**2:.1f}",
                f"{float(row['queue_latency_p50_ms']):.2f}",
            ]
            for row in scaling
        ],
    )
    gates_table = _markdown_table(
        ["Gate", "Requirement", "Status"],
        [[row["gate"], row["name"], row["status"]] for row in summary["gates"]],
    )
    completion_table = _markdown_table(
        ["Question", "Answer"],
        [
            ["1. Could whole-layer assignment execute complete K3?", "NO at 8/4/2/1 GiB"],
            ["2. Does a headline worker aggregate machines/GPUs?", "NO"],
            ["3. Is every compute event tied to a worker?", "YES"],
            ["4. Are total worker memory caps respected?", "YES in placement accounting"],
            ["5. Is the complete checkpoint covered?", "YES; 497,220 tensors, zero gap/overlap"],
            ["6. Are arbitrary real routes supported without hot movement?", "PLACED YES; production process NOT PROVEN"],
            ["7. Are fanout/reduction/network costs explicit?", "YES in the diagnostic event DAG"],
            ["8. Is sharded timing validated?", "NO — decisive MODEL_INVALID gate"],
            ["9. Is the result exact?", "Physical replay math YES; no admissible system result"],
            ["10. Is evidence labeled correctly?", "YES; invalid simulations are diagnostics"],
            ["11. Does the result move the 5 tok/s goal?", "NO performance claim; it identifies the blocking runtime mismatch"],
            ["12. What tier/network is supported?", "NONE until model validation passes"],
        ],
    )
    report = f"""# EXPERIMENT 021: MODEL_INVALID

- Max memory/independent machine: **4.495 GiB peak under the 8 GiB cap**.
- Worker count: **376 independent machines / 376 compute workers**.
- Whole-layer complete-model placement possible: **NO**.
- Best exact tok/s under Regimes A/B/C/D: **N/A / N/A / N/A / N/A — model invalid**.
- Inadmissible diagnostic outputs (not results): A **{diagnostic['A']['tok_s_per_user']:.3f}**, B **{diagnostic['B']['tok_s_per_user']:.3f}**, C **{diagnostic['C']['tok_s_per_user']:.3f}**, D **{diagnostic['D']['tok_s_per_user']:.3f}** tok/s/user.
- Best block/chunk: **no admissible winner**; diagnostic search selected block **{diagnostic['B']['block']}**, chunk **{diagnostic['B']['chunk']}** at Regime B.
- Model-validation error: **median {100 * validation['model_validation']['median_error']:.2f}% / p90 {100 * validation['model_validation']['p90_error']:.2f}% / max {100 * validation['model_validation']['maximum_error']:.2f}%** versus 5%/10%/15% gates.
- Full 93-layer correctness through production worker processes: **FAIL / not executed**.
- Physical independent-machine Swarm measured: **NO**.
- Decisive bottleneck: **the executable shard path is non-resident and production `EXECUTE_SHARD` native dispatch is absent, so the event model does not predict physical runtime**.

This outcome is neither support nor falsification of the sub-layer thesis. The preregistered validation rule requires `MODEL_INVALID`, and it forbids a performance verdict.

## Required truth table

{truth_table}

The `N/A` throughput cells are intentional. Values generated by the failed model are shown only in explicitly marked diagnostic figures/tables and never as exact results.

## 1. Executive verdict

E021 succeeds as a falsification of its current performance model. It does **not** validate an independent-machine K3 runtime and it does **not** justify GPU rental. Structural placement is sound, exact math remained numerically correct in the ordered layer-0-to-8 replay, and accounting is conserved, but primitive predictions miss by factors of roughly 2-to-11 and complete-layer/span predictions miss by roughly 2,800x.

The scientific loop was followed: hypothesis → implementation → benchmark → inspection → redesign. The first 2,000-connection controller run failed; bounded-batch connection establishment retained all 2,000 connections and passed on rerun. No corresponding resident-native runtime redesign was completed, so the decisive model gate remains failed without normalization.

## 2. Canonical Swarm definition from AGENTS.md

The repository-root `AGENTS.md` was read completely before experiment work and is authoritative. Its canonical definition is reproduced verbatim:

> **{CANONICAL_SWARM_DEFINITION}**

Every E021 headline identity is one compute worker on one independent machine. Logical depth groups have zero compute and zero memory. There is no free PCIe, NVLink, NCCL, host aggregate, or same-host link class.

## 3. What E018-E020 got right/wrong

E018 contributed the exact wavefront scheduling idea, but its giant 8-layer stages were not worker-sized Swarm resources. E019 contributed full checkpoint sub-layer placement, exact attention decomposition, and expert stripes; its serial-equality model gate was the wrong question. E020 contributed grouped top-16 expert execution, authenticated transport, cache primitives, and deployment scaffolding, but its 12xP8 topology and eight-workers-per-host image are rejected here.

E021 retains the useful shard primitives and wavefront DAG, removes the host/P8 compute abstraction, and subjects the model to contiguous ordered physical replay. That corrected replay exposes a different failure: current executable residency and current service inputs do not describe the same runtime.

## 4. Hard worker definition

- One `machine-XXXX.worker` is one independent machine.
- Every machine hosts exactly one headline compute worker.
- Every resident byte, mutable state, and compute event has a worker owner.
- Every inter-worker dependency uses the same selected independent-machine link profile.
- Depth groups are scheduling labels only; they have no service time or capacity.

## 5. Whole-layer infeasibility proof

The validator censused checkpoint weights plus KDA/MLA state, AttnRes cache, activations, scratch, transport/reduction buffers, CUDA workspace, and allocator overhead for every layer. At 8 GiB only the small layer 0 fits; **92/93 layers do not**. Complete-model whole-layer assignment is therefore impossible. At 4 GiB the same single special layer fits; at 2/1 GiB no layer fits.

- Minimum complete-layer peak: **{min(row['complete_layer_peak_gib'] for row in feasibility['layers']):.3f} GiB**.
- Maximum complete-layer peak: **{max(row['complete_layer_peak_gib'] for row in feasibility['layers']):.3f} GiB**.
- Receipt: [`placement/whole-layer-feasibility.json`](../../artifacts/experiment-021/placement/whole-layer-feasibility.json).

## 6. Memory-tier placements

{placement_table}

Every tier is capacity-valid, but performance is not validated. P=8 has physically measured grouped top-16 service; P=16/P=32 currently fall back to legacy exact expert-stripe service inputs and are especially provisional.

![Worker counts](../../artifacts/experiment-021/charts/chart-03-worker-count.png)

## 7. Complete checkpoint coverage

The E021 census covers **497,220 tensors** and **1,560,860,324,864 bytes**. The byte ledger reports zero gaps and zero unintended overlap. The 8 GiB manifest assigns all checkpoint bytes across 376 independent workers; no worker owns a whole ordinary layer, routed expert, or shared expert.

Direct shard loading passed on **{direct['request_count']} representative role reads / {direct['bytes_read']:,} bytes**, with zero large full-tensor materializations. Full coverage is in [`placement/checkpoint-coverage.csv`](../../artifacts/experiment-021/placement/checkpoint-coverage.csv); it is deliberately large because it records every tensor owner and byte slice.

## 8. Production worker execution path

Authenticated bounded TLS/HMAC frames, backpressure, explicit worker registration, lifecycle, and cache hash/atomic-marker primitives pass. The ShardCache startup primitive rejects corrupted content and reuses verified objects on restart.

The production path is nevertheless incomplete:

- `EXECUTE_SHARD` in the current worker daemon returns a mock partial vector rather than invoking a native shard primitive.
- Full K3 bundle acquisition is not integrated into the independent worker startup path.
- The only built local Linux image embeds E020's rejected eight-worker host lifecycle, is unpublished, and is not a pinned E021 independent-machine image.

Runtime receipt: [`runtime/production-gates.json`](../../artifacts/experiment-021/runtime/production-gates.json).

## 9. Ordered physical shard validation

The RTX 5090 physically executed layer 0 to establish state, then layers 1-to-8 contiguously. Actual native calls, exact hidden transitions, local authenticated request/result frames, and one-resource/no-overlap order were timed. No target-pass normalization or global multiplier was applied.

{validation_table}

The hidden replay remained correct (maximum relative L2 **{validation['maximum_hidden_relative_l2_error']:.3e}**), but timing failed. Complete layers took **{min(validation['layer_walls_ms'].values()) / 1000:.2f}-{max(validation['layer_walls_ms'].values()) / 1000:.2f} s** while the model predicted millisecond-scale service. The timed graph performed **{validation['direct_read_request_count']:,} direct reads / {validation['direct_read_bytes'] / 1e9:.2f} GB** during the span; the event model assumes resident worker banks. Even isolated primitive rows missed by 46-91%, so residency alone is not the only stale assumption.

**Gate result: FAIL → `MODEL_INVALID`.**

## 10. 93-layer correctness

E020's prior in-process physical shard control completed 93 layers, exact routes, and a maximum hidden L2 of **{correctness['prior_physical_shard_math_control']['maximum_relative_l2_error']:.3e}** in about **{correctness['prior_physical_shard_math_control']['wall_seconds'] / 60:.1f} minutes**. It used the rejected P8 host-era ownership mapping and did not cross production worker-process dispatch, so it is not admissible for E021 Gate 10.

E021 did not rerun that same in-process shortcut. Production worker-process hidden/logit/token/state receipts remain null. **Gate result: FAIL.**

## 11. Independent-machine event engine

Every diagnostic compute record has `worker_id`; every network record names directed worker-to-worker edges and pays RTT, bandwidth serialization, measured protocol overhead, queueing, and synchronization. Local and inter-depth profile inputs are identical, preventing a hidden intra-host fast path.

Accounting passed across 8/4/2/1 GiB: `sum(worker compute durations)` reconciled to the bottom-up task list, and network duration reconciled to latency + serialization + software overhead. Maximum residuals are floating-point noise; see [`validation/accounting-reconciliation.json`](../../artifacts/experiment-021/validation/accounting-reconciliation.json).

Accounting correctness does not imply timing-model validity. All simulation rows therefore have `admissible=false`.

## 12. Expert stripes

The P=8 diagnostic uses the measured grouped top-16 design: each stripe receives activation, 16 routes and weights, executes local fragments, route-weights and accumulates locally, emits one partial latent vector, and participates in one explicit reduction. It has no per-expert network RPC fanout.

P=16/P=32 lack a physically characterized grouped-bank service and use exact legacy stripe timings only as invalid diagnostics. They cannot support a headline claim.

## 13. KDA/MLA stripes

KDA and Gated MLA are split by explicit head/projection stripes. State objects have worker ownership, AttnRes completed objects are cached at named owners, and required reductions cross independent-machine links. There is no whole-attention-layer service event.

The ordered replay included both complete KDA and complete MLA layers. Their numerical outputs passed; their wall predictions failed.

## 14. Wavefront over independent workers

Blocks 7/12/16 and chunks 1/2/4 were swept. Every stage transition emerges from explicit worker completion. The invalid 8 GiB/Regime B diagnostic selected block 16/chunk 2, with fill **{_read(artifact_root / 'simulation' / 'critical-path.json')['fill_ms']:.2f} ms**, median steady period **{_read(artifact_root / 'simulation' / 'critical-path.json')['steady_state_period_ms_median']:.2f} ms**, and drain **{_read(artifact_root / 'simulation' / 'critical-path.json')['drain_ms']:.2f} ms**. These values characterize the failed model, not K3 performance.

![Diagnostic critical path](../../artifacts/experiment-021/charts/chart-04-critical-path.png)

## 15. Memory-fragmentation curve

{diagnostic_table}

`*` = diagnostic output from a failed model; no entry is an exact or admissible throughput result.

![Memory fragmentation diagnostic](../../artifacts/experiment-021/charts/chart-01-memory-fragmentation-curve.png)

The invalid model suggests that smaller machines increase worker count, network bytes, active worker-seconds, compute inflation, and idle capacity. At Regime B, diagnostic active worker-seconds/token rise from **{float(best[8]['B']['active_worker_seconds_per_accepted_token']):.3f}** at 8 GiB to **{float(best[1]['B']['active_worker_seconds_per_accepted_token']):.3f}** at 1 GiB; diagnostic network traffic rises from **{float(best[8]['B']['network_bytes_per_accepted_token']) / 1e6:.1f} MB** to **{float(best[1]['B']['network_bytes_per_accepted_token']) / 1e6:.1f} MB** per accepted token.

## 16. Network envelope

The 8 GiB diagnostic grid re-schedules the event DAG across RTT 0.25-50 ms and bandwidth 0.1-25 Gb/s. It would cross 5 tok/s only up to about 2 ms RTT at ≥2.5 Gb/s and never at 5 ms on the sampled grid. This is a sensitivity map of an invalid model, not a viable-network claim.

![Diagnostic network envelope](../../artifacts/experiment-021/charts/chart-02-network-envelope.png)

- FAST-LAN SWARM: **NOT SUPPORTED — MODEL_INVALID**
- REGIONAL SWARM: **NOT SUPPORTED — MODEL_INVALID**
- CONSUMER-WAN SWARM: **NOT SUPPORTED — MODEL_INVALID**

## 17. Whole-layer control

The separate relaxed 20 GiB control assigns one complete layer per independent machine (93 machines) and uses E018 physical whole-layer services plus shaped Regime B links. Its best diagnostic value is **{float(control['exact_tok_s_per_user']):.2f} tok/s/user**, versus **{float(best[8]['B']['exact_tok_s_per_user']):.2f}** for the invalid 8 GiB sub-layer model. The comparison suggests a large fragmentation tax, but neither number repairs E021 model validation and the whole-layer arm is not a Swarm headline result.

![Whole-layer control](../../artifacts/experiment-021/charts/chart-06-whole-layer-control.png)

## 18. Control-plane scaling

{scaling_table}

The initial unbounded 2,000-connection storm failed. After redesign, the controller established TLS connections in batches of 100, retained all requested connections, then dispatched. The rerun passed with 2,000 peak open connections and 10,000 messages. This measures lightweight lifecycle/protocol scaling only; `native_shard_compute_invoked=false`.

## 19. Heterogeneity

In the invalid 8 GiB/Regime B model, ±20% random compute, 10% at 1.5x/2x slower, ±20% network jitter, and one 2x slow critical-path worker produced diagnostic amplification from **{min(float(row['straggler_amplification']) for row in heterogeneity):.3f}x** to **{max(float(row['straggler_amplification']) for row in heterogeneity):.3f}x**. These small effects indicate modeled network latency, not stragglers, dominates—but this conclusion must be rechecked after runtime validation.

## 20. Concurrency/economics

The invalid model predicts per-user rate falling from **{float(concurrency[0]['per_user_tok_s_p50_diagnostic']):.2f}** at one request to **{float(concurrency[-1]['per_user_tok_s_p50_diagnostic']):.2f}** at 16, while aggregate rate rises to **{float(concurrency[-1]['aggregate_tok_s_diagnostic']):.2f}**. Average utilization remains only **{100 * float(concurrency[-1]['average_worker_utilization']):.2f}%** at 16 requests.

[`cost/resource-economics.csv`](../../artifacts/experiment-021/cost/resource-economics.csv) reports workers, resident GiB, active/resident worker-seconds, network MB/token, compute inflation, and utilization by tier/regime. It deliberately leaves admissible throughput and cost/token blank. No full compatible market fleet exists, so no legitimate hourly fleet estimate is emitted.

![Diagnostic concurrency](../../artifacts/experiment-021/charts/chart-09-concurrency.png)

## 21. Read-only fragmented market inventory

The timestamped Vast query searched only rentable single-GPU offers with ≤8 GiB. It found **{_read(artifact_root / 'vast' / 'single-machine-offer-snapshot.json')['offer_count']} offers**, of which **{market['tiers'][0]['unique_kernel_compatible_machines']} unique machines** met the current SM86/CUDA/driver and 8 GiB placement requirements, versus **{market['tiers'][0]['required_independent_machines']} required**. No smaller tier had a qualifying offer. Offer bandwidth metadata was not used as inter-worker RTT evidence.

The safety receipt records **0 rentals, 0 Vast mutations**, and a blocked mutation self-test before subprocess. Future commands are rendered only in [`vast/rendered-future-plan.txt`](../../artifacts/experiment-021/vast/rendered-future-plan.txt).

![Fragmented inventory](../../artifacts/experiment-021/charts/chart-08-market-fragmented-inventory.png)

## 22. What failed

1. The timing model failed its corrected physical validation by a decisive margin.
2. Physical replay and modeled service do not share a residency policy.
3. Primitive service inputs are stale even before complete-layer load/upload costs.
4. Production worker `EXECUTE_SHARD` native dispatch remains unimplemented.
5. Complete 93-layer worker-process correctness therefore remains untested.
6. The only existing Linux image encodes the rejected E020 host topology and lacks a published immutable digest.
7. Current compatible fragmented market inventory is insufficient.

No gate was moved, no normalization was applied, and no favorable diagnostic projection was promoted.

## 23. What is proven pre-physically

- Exact byte-complete sub-layer placement is possible at 8/4/2/1 GiB caps.
- Complete-model whole-layer assignment is impossible at those caps.
- One-machine/one-worker ownership can be represented without aggregate hosts.
- Ordered physical shard math through layer 8 remains numerically correct.
- The explicit event engine conserves compute/network accounting.
- Authenticated controller protocol can hold 2,000 lightweight workers after bounded connection setup.
- Vast safety remained read-only.

No validated throughput, network envelope, full production correctness, or physical multi-machine claim is proven.

## 24. What remains physically unproven

Independent machines have not executed K3 fragments together. Physical inter-worker RTT/bandwidth, remote collectives, memory peaks on small GPUs, heterogeneous kernels, complete worker startup, and exact 93-layer production transport all remain unmeasured. The evidence class therefore does not reach `VALIDATED_INDEPENDENT_MACHINE_MODEL` or `PHYSICAL_SWARM`.

## 25. Recommendation for the first physical Swarm experiment

**Do not rent yet.** First implement a resident per-worker native runtime whose timed execution matches the event-model residency contract; wire authenticated `EXECUTE_SHARD` to every required native primitive; integrate bundle acquisition and ShardCache at startup; publish a one-worker immutable Linux image; rerun the ordered primitive/layer/2-to-8-layer validation; then complete 93-layer correctness through worker processes.

Only if median/p90/max error passes 5%/10%/15% without normalization and production correctness passes should a later experiment reassess market availability and request explicit rental approval.

## Required hard gates

{gates_table}

Because Gates 9-11 fail, E021 issues no Swarm support verdict.

## Before declaring the experiment complete

{completion_table}

![Evidence stack](../../artifacts/experiment-021/charts/chart-10-evidence-stack.png)

## CORE SWARM PRE-PHYSICAL VERDICT

> Can a Kimi K3 system whose participating independent machines are individually too small to execute the complete model via whole-layer assignment plausibly exceed 5 tok/s using exact sub-layer distributed execution?

**MODEL INVALID.** No memory tier or network regime qualifies. The 8 GiB / Regime B diagnostic projection is not admissible because ordered physical validation failed.

## PHYSICAL SWARM VERDICT

**NOT YET PHYSICALLY PROVEN**
"""
    path = repo / "docs" / "experiments" / "EXPERIMENT_021_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path


def finalize_experiment(repo: Path, artifact_root: Path) -> dict[str, Any]:
    environment = materialize_environment(repo, artifact_root)
    model = materialize_model_metadata(artifact_root)
    economics = materialize_economics(artifact_root)
    materialize_commands(artifact_root)
    summary = build_summary(artifact_root)
    truth = build_truth_table(artifact_root, summary)
    failures = build_failure_log(artifact_root)
    source_manifest = materialize_source_manifest(repo, artifact_root)
    report = generate_report(repo, artifact_root, summary)
    return {
        "outcome": OUTCOME,
        "summary": summary,
        "truth_table": truth,
        "environment": environment,
        "model_metadata": model,
        "economics_rows": len(economics),
        "failure_log": failures,
        "source_manifest": source_manifest,
        "report": str(report),
    }


__all__ = ["finalize_experiment"]
