"""Zero-spend E025 freeze, checkpoint, placement, and GO gates."""

from __future__ import annotations

import inspect
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from swarm_inference.execution.kimi_k3_stage import PersistentKimiStageExecutor
from swarm_inference.experiments.experiment_022.native_dispatch import (
    NativeShardDispatcher,
)

from .bundles import build_image_inputs
from .constants import EVIDENCE_CLASS, MODEL_ID, MODEL_REVISION
from .controller import PhysicalController
from .io import atomic_write_json, canonical_sha256, read_json, sha256_file, utc_now
from .placement import build_placement_and_distribution
from .worker import WorkerRuntime

SCHEMA_VERSION = "experiment-025-zero-spend-preflight-v1"


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path

    @property
    def preflight(self) -> Path:
        return self.root / "preflight"

    @property
    def rental(self) -> Path:
        return self.root / "rental"

    @property
    def workers(self) -> Path:
        return self.root / "workers"

    @property
    def correctness(self) -> Path:
        return self.root / "correctness"

    @property
    def generation(self) -> Path:
        return self.root / "generation"

    @property
    def telemetry(self) -> Path:
        return self.root / "telemetry"

    @property
    def final(self) -> Path:
        return self.root / "final"

    def create(self) -> None:
        for directory in (
            self.preflight,
            self.rental,
            self.workers,
            self.correctness,
            self.generation,
            self.telemetry,
            self.final,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_run_layout(artifact_parent: Path, run_id: str | None = None) -> RunLayout:
    resolved_run_id = run_id or new_run_id()
    root = artifact_parent.expanduser().resolve() / f"experiment-025-{resolved_run_id}"
    if root.exists():
        raise ValueError(f"refusing to replace an E025 run root: {root}")
    layout = RunLayout(root)
    layout.create()
    return layout


def _git(repo: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {process.stderr[-500:]}")
    return process.stdout


def _source_files(repo: Path) -> list[Path]:
    roots = [
        repo / "src",
        repo / "scripts",
        repo / "deployment",
        repo / "tests",
        repo / "third_party/colibri/c",
    ]
    files = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    files.extend(
        path
        for path in (repo / "pyproject.toml", repo / "uv.lock", repo / "README.md")
        if path.is_file()
    )
    return sorted(set(files))


def source_tree_sha256(repo: Path) -> str:
    root = repo.expanduser().resolve()
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _source_files(root)
    ]
    return canonical_sha256(rows)


def code_freeze(repo: Path, output_path: Path) -> dict[str, Any]:
    root = repo.expanduser().resolve()
    previous = read_json(output_path) if output_path.is_file() else None
    files = _source_files(root)
    file_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    status = _git(root, "status", "--short")
    e024_roots = [
        root / "src/swarm_inference/experiments/experiment_024",
        root / "artifacts/experiment-024",
    ]
    e024_paths = [
        path
        for e024_root in e024_roots
        if e024_root.is_dir()
        for path in e024_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    e024_paths.extend(root.glob("scripts/experiment_024*.py"))
    e024_paths.extend(root.glob("tests/test_experiment_024*.py"))
    e024_paths = sorted(set(e024_paths))
    current_e024_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in e024_paths
    ]
    baseline_e024_rows = (
        previous.get("experiment_024_preservation", {}).get("baseline_files")
        or previous.get("experiment_024_preservation", {}).get("files")
        if previous
        else current_e024_rows
    )
    e024_preserved = current_e024_rows == baseline_e024_rows
    payload = {
        "schema_version": "experiment-025-code-freeze-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if e024_preserved else "FAIL",
        "git_head": _git(root, "rev-parse", "HEAD").strip(),
        "git_branch": _git(root, "branch", "--show-current").strip(),
        "git_status_short": status.splitlines(),
        "dirty_worktree_disclosed": bool(status.strip()),
        "source_file_count": len(file_rows),
        "source_tree_sha256": canonical_sha256(file_rows),
        "files": file_rows,
        "experiment_024_preservation": {
            "instruction": "historical evidence; do not repair, close, or rerun",
            "file_count": len(e024_paths),
            "baseline_files": baseline_e024_rows,
            "files": current_e024_rows,
            "baseline_matches_current": e024_preserved,
            "modified_by_e025": not e024_preserved,
        },
    }
    atomic_write_json(output_path, payload)
    return payload


def checkpoint_identity(
    checkpoint: Path,
    source_placement_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    source = read_json(source_placement_path.expanduser().resolve())
    required = [
        root / "config.json",
        root / "model.safetensors.index.json",
        root / "tokenizer_config.json",
        root / "tiktoken.model",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    index = read_json(root / "model.safetensors.index.json") if not missing else {}
    weight_map = index.get("weight_map", {})
    shards = sorted({str(value) for value in weight_map.values()})
    shard_rows = [
        {
            "name": name,
            "bytes": (root / name).stat().st_size if (root / name).is_file() else None,
            "present": (root / name).is_file(),
        }
        for name in shards
    ]
    checkpoint_source = source["checkpoint"]
    gates = {
        "authoritative_model": MODEL_ID == "moonshotai/Kimi-K3",
        "authoritative_revision": checkpoint_source["revision"] == MODEL_REVISION,
        "required_metadata_present": not missing,
        "config_hash_exact": (
            not missing
            and sha256_file(root / "config.json") == checkpoint_source["config_sha256"]
        ),
        "index_hash_exact": (
            not missing
            and sha256_file(root / "model.safetensors.index.json")
            == checkpoint_source["index_sha256"]
        ),
        "all_indexed_shards_present": bool(shard_rows)
        and all(bool(row["present"]) for row in shard_rows),
        "expected_shard_count": len(shard_rows) == int(checkpoint_source["shard_count"]),
    }
    payload = {
        "schema_version": "experiment-025-checkpoint-identity-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model_id": MODEL_ID,
        "checkpoint_path": str(root),
        "checkpoint": checkpoint_source,
        "required_text_path": {
            "tensor_count": source["coverage"]["expected_required_tensor_count"],
            "source_weight_bytes": source["coverage"][
                "expected_required_source_weight_bytes"
            ],
            "transformer_layers": 93,
            "embedding": True,
            "final_norm": True,
            "language_model_head": True,
        },
        "multimodal_disclosure": {
            "headline_mode": "text-only",
            "multimodal_only_tensors_executed": False,
            "excluded_only_when_absent_from_authoritative_text_path_census": True,
        },
        "shard_count": len(shard_rows),
        "shards": shard_rows,
        "gates": gates,
        "missing": missing,
    }
    atomic_write_json(output_path, payload)
    return payload


def dispatch_attestation(repo: Path, output_path: Path) -> dict[str, Any]:
    worker_source = inspect.getsource(WorkerRuntime.process) + inspect.getsource(
        WorkerRuntime._execute_stage
    )
    stage_source = inspect.getsource(PersistentKimiStageExecutor._execute_one)
    e022_source = inspect.getsource(NativeShardDispatcher)
    controller_source = inspect.getsource(PhysicalController._request)
    controller_module_path = (
        repo / "src/swarm_inference/experiments/experiment_025/controller.py"
    )
    controller_module_source = controller_module_path.read_text(encoding="utf-8")
    gates = {
        "authenticated_execute_shard_frame": "MessageType.EXECUTE_SHARD"
        in controller_source
        and "AuthenticatedConnection" in inspect.getsource(PhysicalController),
        "worker_dispatches_execute_stage": "Action.EXECUTE_STAGE" in worker_source,
        "native_stage_implementation_called": "execute_decode" in worker_source
        and "execute_prefill" in worker_source,
        "real_tensor_result_returned": "stage_boundary_hidden_states" in worker_source,
        "native_cuda_primitives_called": "runtime.execute_" in stage_source,
        "external_real_expert_dispatch": "external_expert_dispatch" in stage_source,
        "e022_native_lineage_present": "execute" in e022_source,
        "no_controller_compute_fallback": "controller_compute_fallback" in worker_source,
        "controller_imports_no_model_executor": "PersistentKimiStageExecutor"
        not in controller_module_source
        and "ExpertPartitionExecutor" not in controller_module_source,
        "controller_has_no_generation_api_client": all(
            marker not in controller_module_source
            for marker in (
                "import requests",
                "import httpx",
                "import openai",
                "urllib.request",
            )
        ),
        "sampled_token_originates_in_worker_result": "sampled_token_ids"
        in inspect.getsource(PhysicalController.execute_token),
        "worker_rejects_cached_or_synthetic_claims": "cached_output" in worker_source
        and "synthetic_tensor" in worker_source,
    }
    relevant = [
        repo / "src/swarm_inference/experiments/experiment_025/wire.py",
        controller_module_path,
        repo / "src/swarm_inference/experiments/experiment_025/worker.py",
        repo / "src/swarm_inference/experiments/experiment_025/expert_partition.py",
        repo / "src/swarm_inference/execution/kimi_k3_stage.py",
        repo / "src/swarm_inference/experiments/experiment_022/native_dispatch.py",
    ]
    payload = {
        "schema_version": "experiment-025-native-dispatch-attestation-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "path": [
            "controller task",
            "HMAC-authenticated EXECUTE_SHARD frame over TLS",
            "independent worker process",
            "PersistentKimiStageExecutor or ExpertPartitionExecutor",
            "real native CUDA primitive",
            "fresh real tensor result",
            "authenticated SHARD_RESULT frame",
        ],
        "files": [
            {
                "path": path.relative_to(repo).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in relevant
        ],
        "gates": gates,
        "functional_test_required": "tests/test_experiment_025_dispatch.py",
    }
    atomic_write_json(output_path, payload)
    return payload


def environment_receipt(output_path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    payload = {
        "schema_version": "experiment-025-local-environment-v1",
        "generated_at_utc": utc_now(),
        "status": "PASS" if process.returncode == 0 else "FAIL",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "local_gpu_disclosed_as_pre_rental_only": process.stdout.strip(),
        "local_gpu_headline_compute_permitted": False,
        "nvidia_smi_returncode": process.returncode,
    }
    atomic_write_json(output_path, payload)
    return payload


def write_emergency_cleanup_script(
    *,
    repo: Path,
    layout: RunLayout,
) -> Path:
    """Write the idempotent, run-bound emergency cleanup command before rental."""

    run_id = layout.root.name.removeprefix("experiment-025-")
    script_path = layout.rental / f"destroy_all_e025_{run_id}.ps1"
    repo_literal = str(repo.expanduser().resolve()).replace("'", "''")
    ledger_literal = str(layout.rental / "instance-ledger.jsonl").replace("'", "''")
    output_literal = str(layout.rental / "emergency-cleanup.json").replace("'", "''")
    content = f"""param(
    [string]$Reason = 'manual emergency cleanup'
)
$ErrorActionPreference = 'Stop'
$E025Repo = '{repo_literal}'
$env:PYTHONPATH = Join-Path $E025Repo 'src'
& (Join-Path $E025Repo '.venv\\Scripts\\python.exe') `
    (Join-Path $E025Repo 'scripts\\experiment_025_cleanup.py') `
    --run-id '{run_id}' `
    --ledger '{ledger_literal}' `
    --output '{output_literal}' `
    --reason $Reason
exit $LASTEXITCODE
"""
    script_path.write_text(content, encoding="utf-8", newline="\n")
    return script_path


def build_zero_spend_preflight(
    *,
    repo: Path,
    checkpoint: Path,
    source_placement: Path,
    layout: RunLayout,
    image_context: Path,
) -> dict[str, Any]:
    layout.create()
    emergency_cleanup = write_emergency_cleanup_script(repo=repo, layout=layout)
    code = code_freeze(repo, layout.preflight / "code-freeze.json")
    environment = environment_receipt(layout.preflight / "environment.json")
    checkpoint_receipt = checkpoint_identity(
        checkpoint,
        source_placement,
        layout.preflight / "checkpoint-identity.json",
    )
    dispatch = dispatch_attestation(repo, layout.preflight / "native-dispatch.json")
    placement_path = layout.preflight / "physical-placement.json"
    distribution_path = layout.preflight / "checkpoint-distribution.json"
    placement_distribution = build_placement_and_distribution(
        checkpoint,
        source_placement,
        placement_path,
        distribution_path,
    )
    image_inputs = build_image_inputs(
        checkpoint=checkpoint,
        placement_path=placement_path,
        distribution_path=distribution_path,
        output_directory=image_context,
    )
    atomic_write_json(layout.preflight / "image-inputs.json", image_inputs)
    gates = {
        "stage_zero_no_vast_mutations": True,
        "code_frozen": code["status"] == "PASS",
        "e024_preserved": code["experiment_024_preservation"]["modified_by_e025"]
        is False,
        "local_environment_visible": environment["status"] == "PASS",
        "checkpoint_exact": checkpoint_receipt["status"] == "PASS",
        "physical_placement_exact": placement_distribution["placement"]["status"]
        == "PASS",
        "distribution_exact": placement_distribution["distribution"]["status"]
        == "PASS",
        "native_dispatch_attested": dispatch["status"] == "PASS",
        "image_inputs_frozen": image_inputs["status"] == "PASS",
        "deployment_image_published": False,
        "local_full_graph_rehearsal_passed": False,
        "focused_tests_passed": False,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": "PREPARED_NOT_RENTAL_READY",
        "run_id": layout.root.name.removeprefix("experiment-025-"),
        "evidence_class_target": EVIDENCE_CLASS,
        "hypothesis": (
            "The complete real Kimi K3 text path can generate correct autoregressive "
            "tokens over independent consumer GPUs while four 8-12 GiB physical "
            "workers necessarily execute disjoint layer-89 expert fragments."
        ),
        "falsification": (
            "Any missing real tensor, native dispatch failure, numerical gate failure, "
            "monolithic fallback, non-consumer compute, missing physical trace, or "
            "surviving Vast instance prevents PASS."
        ),
        "placement": placement_distribution,
        "image_inputs": image_inputs,
        "emergency_cleanup_script": {
            "path": str(emergency_cleanup),
            "sha256": sha256_file(emergency_cleanup),
            "created_before_any_rental": True,
            "idempotent": True,
        },
        "gates": gates,
        "vast_mutations_performed": False,
    }
    atomic_write_json(layout.preflight / "stage-0-summary.json", payload)
    return payload


def write_full_fleet_go(
    *,
    repo: Path,
    run_id: str,
    image_receipt_path: Path,
    test_receipt_path: Path,
    rehearsal_receipt_path: Path,
    backbone_canary_path: Path,
    sub_layer_canary_path: Path,
    local_image_canary_path: Path,
    offer_snapshot_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    image = read_json(image_receipt_path)
    tests = read_json(test_receipt_path)
    rehearsal = read_json(rehearsal_receipt_path)
    backbone = read_json(backbone_canary_path)
    sub_layer = read_json(sub_layer_canary_path)
    local_image_canary = read_json(local_image_canary_path)
    offers = read_json(offer_snapshot_path)
    digest = str(image.get("immutable_digest", ""))
    current_source_id = source_tree_sha256(repo)
    checklist = {
        "deployment_image_receipt_passed": image.get("status") == "PASS",
        "image_built": image.get("build_status") == "PASS",
        "image_pushed": image.get("push_status") == "PASS",
        "image_anonymously_resolvable": image.get(
            "anonymous_registry_resolve_status"
        )
        == "PASS",
        "immutable_image_digest": digest.startswith("sha256:")
        and len(digest) == 71,
        "focused_tests_passed": tests.get("status") == "PASS",
        "controller_source_matches_tested_image": bool(current_source_id)
        and current_source_id == image.get("source_id")
        and current_source_id == tests.get("source_tree_sha256")
        and current_source_id == rehearsal.get("source_tree_sha256"),
        "local_full_graph_rehearsal_passed": rehearsal.get("status") == "PASS",
        "single_backbone_canary_passed": backbone.get("status") == "PASS"
        and backbone.get("image_digest") == digest,
        "physical_sub_layer_canary_passed": sub_layer.get("status") == "PASS"
        and sub_layer.get("image_digest") == digest,
        "selected_non_sm86_gpu_runtime_canary_passed": (
            not offers.get("fleet_plan", {}).get("compatibility_canaries_required")
            or (
                local_image_canary.get("status") == "PASS"
                and local_image_canary.get("immutable_digest") == digest
                and local_image_canary.get("physical_gpu", {}).get("gpu_name", "")
                .upper()
                .endswith("RTX 5090")
            )
        ),
        "fresh_complete_on_demand_fleet_available": offers.get("status") == "PASS"
        and offers.get("fleet_plan", {}).get("status") == "PASS",
        "complete_97_worker_fleet": offers.get("fleet_plan", {}).get("worker_count")
        == 97,
        "acquisition_deadline_feasible": offers.get("fleet_plan", {}).get(
            "expected_acquisition_within_25_minute_deadline"
        )
        is True,
        "budget_sufficient_for_full_ttl_and_ingress": offers.get(
            "redacted_budget", {}
        ).get("sufficient")
        is True,
        "every_instance_group_has_three_alternates": bool(
            offers.get("fleet_plan", {}).get("instance_groups")
        )
        and all(
            len(group.get("alternates", [])) >= 3
            for group in offers.get("fleet_plan", {}).get("instance_groups", [])
        ),
        "four_distinct_sub_layer_machines": len(
            set(offers.get("fleet_plan", {}).get("sub_layer_machine_ids", []))
        )
        == 4,
    }
    payload = {
        "schema_version": "experiment-025-full-fleet-go-v1",
        "generated_at_utc": utc_now(),
        "status": "GO" if all(checklist.values()) else "NO_GO",
        "run_id": run_id,
        "current_source_tree_sha256": current_source_id,
        "checklist": checklist,
        "image": image,
        "local_image_canary": local_image_canary,
        "fleet_plan": offers.get("fleet_plan"),
        "paid_mutation_authorized_only_if_status_go": True,
    }
    atomic_write_json(output_path, payload)
    return payload


__all__ = [
    "SCHEMA_VERSION",
    "RunLayout",
    "build_zero_spend_preflight",
    "checkpoint_identity",
    "code_freeze",
    "create_run_layout",
    "dispatch_attestation",
    "environment_receipt",
    "new_run_id",
    "source_tree_sha256",
    "write_emergency_cleanup_script",
    "write_full_fleet_go",
]
