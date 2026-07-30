"""End-to-end Experiment 002 orchestration and evidence generation."""

from __future__ import annotations

import asyncio
import html
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import yaml

from swarm_inference.config.real_model import (
    RealExperimentConfig,
    load_real_experiment_config,
)
from swarm_inference.exceptions import IntegrityError
from swarm_inference.experiments.real_model import run_qwen3_experiment_session
from swarm_inference.experiments.real_status import evaluate_experiment_002_status
from swarm_inference.experiments.runner import write_artifact_manifest
from swarm_inference.model.manifest import (
    load_manifest,
    verify_manifest_shards,
)
from swarm_inference.model.reference import run_reference_suite_subprocess
from swarm_inference.model.shard_builder import (
    inspect_qwen3_model,
    model_inspection_payload,
    resolve_model,
    shard_model,
)


@dataclass(frozen=True, slots=True)
class Experiment002Run:
    run_directory: Path
    report_path: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.summary.get("overall_status") == "PASS"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_quality_evidence() -> dict[str, Any]:
    evidence_path = os.environ.get("SWARM_EXPERIMENT_002_QUALITY_EVIDENCE")
    if not evidence_path:
        return {
            "required": False,
            "overall_status": "NOT_RUN",
            "detail": "No launcher-provided quality evidence was supplied.",
            "gates": [],
        }
    path = Path(evidence_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "required": True,
            "overall_status": "FAIL",
            "detail": f"Could not read launcher quality evidence {path}: {exc}",
            "gates": [],
        }
    if not isinstance(payload, dict):
        return {
            "required": True,
            "overall_status": "FAIL",
            "detail": f"Launcher quality evidence is not a JSON object: {path}",
            "gates": [],
        }
    payload["source_path"] = str(path)
    return cast(dict[str, Any], payload)


def _validate_logs(run_dir: Path) -> dict[str, Any]:
    fatal_markers = (
        "Traceback (most recent call last)",
        "OutOfMemoryError",
        "CUDA out of memory",
        "unhandled exception",
    )
    files: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for path in sorted((run_dir / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(run_dir)).replace("\\", "/")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "line_count": len(text.splitlines()),
            }
        )
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(marker.lower() in line.lower() for marker in fatal_markers):
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "text": line[:500],
                    }
                )
    required_names = {
        "coordinator.log",
        "reference.log",
        "worker-000.log",
        "worker-001.log",
        "worker-002.log",
        "worker-003.log",
    }
    observed_names = {Path(item["path"]).name for item in files}
    worker_logs_nonempty = all(
        (run_dir / "logs" / f"worker-{index:03d}.log").is_file()
        and (run_dir / "logs" / f"worker-{index:03d}.log").stat().st_size > 0
        for index in range(4)
    )
    passed = required_names <= observed_names and worker_logs_nonempty and not matches
    return {
        "status": "PASS" if passed else "FAIL",
        "required_logs_present": required_names <= observed_names,
        "worker_logs_nonempty": worker_logs_nonempty,
        "ignored_fatal_exception_count": len(matches),
        "fatal_matches": matches,
        "files": files,
    }


def _git_evidence(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise IntegrityError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_status": status.splitlines(),
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _environment_probe(output: Path, log: Path | None = None) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "swarm_inference.experiments.environment_probe",
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            result.stdout + ("\n" if result.stdout else "") + result.stderr,
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise IntegrityError(f"isolated environment probe failed: {result.stderr or result.stdout}")
    return cast(
        dict[str, Any],
        json.loads(output.read_text(encoding="utf-8")),
    )


def _prompt_with_token_count(
    tokenizer: Any,
    *,
    seed_text: str,
    token_count: int,
) -> tuple[str, list[int]]:
    repeated = seed_text
    ids: list[int] = []
    while len(ids) < token_count:
        repeated = repeated + " " + seed_text
        ids = [int(value) for value in tokenizer(repeated, return_tensors=None)["input_ids"]]
    ids = ids[:token_count]
    return tokenizer.decode(ids, skip_special_tokens=False), ids


def build_prompt_suite(
    *,
    model_path: Path,
    max_new_tokens: int,
    include_prompt_suite: bool,
    include_replay: bool,
) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    if max_new_tokens < 4 and include_replay:
        raise ValueError("cache replay validation requires at least four new tokens")
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_path,
        local_files_only=True,
    )

    def request(
        request_id: str,
        name: str,
        prompt: str,
        phase: str,
        **extra: Any,
    ) -> dict[str, Any]:
        token_ids = [int(value) for value in tokenizer(prompt, return_tensors=None)["input_ids"]]
        return {
            "request_id": request_id,
            "name": name,
            "prompt": prompt,
            "phase": phase,
            "prompt_token_ids": token_ids,
            "max_new_tokens": max_new_tokens,
            **extra,
        }

    prompts = [
        request(
            "prompt-001-factual",
            "factual-completion",
            "The capital of France is",
            "smoke",
        )
    ]
    if include_prompt_suite:
        medium_text, medium_ids = _prompt_with_token_count(
            tokenizer,
            seed_text=(
                "Distributed inference separates a neural network into ordered stages "
                "while preserving deterministic tensor and cache semantics."
            ),
            token_count=128,
        )
        long_text, long_ids = _prompt_with_token_count(
            tokenizer,
            seed_text=(
                "A reliable systems experiment records immutable inputs, process ownership, "
                "transport evidence, numerical correctness, memory use, and cleanup."
            ),
            token_count=512,
        )
        prompts.extend(
            [
                request(
                    "prompt-002-arithmetic",
                    "arithmetic",
                    "Calculate 17 multiplied by 23. The answer is",
                    "suite",
                ),
                request(
                    "prompt-003-code",
                    "code-completion",
                    "def fibonacci(n):",
                    "suite",
                ),
                request(
                    "prompt-004-repeated",
                    "repeated-token-structure",
                    "echo echo echo echo :: :: :: repeat repeat repeat ->",
                    "suite",
                ),
                request(
                    "prompt-005-punctuation",
                    "punctuation-heavy",
                    '{"status":"pending","values":[1,2,3],"nested":{"ok":true}} ->',
                    "suite",
                ),
                {
                    "request_id": "prompt-006-medium",
                    "name": "medium-128-token",
                    "prompt": medium_text,
                    "phase": "suite",
                    "prompt_token_ids": medium_ids,
                    "max_new_tokens": max_new_tokens,
                },
                {
                    "request_id": "prompt-007-long",
                    "name": "long-512-token",
                    "prompt": long_text,
                    "phase": "suite",
                    "prompt_token_ids": long_ids,
                    "max_new_tokens": max_new_tokens,
                },
                request(
                    "prompt-008-concurrent-a",
                    "concurrent-first-request",
                    "Name one primary colour:",
                    "suite",
                    concurrent_group="pair-001",
                ),
                request(
                    "prompt-008-concurrent-b",
                    "concurrent-second-request",
                    "Complete this sequence: 2, 4, 6,",
                    "suite",
                    concurrent_group="pair-001",
                ),
            ]
        )
    if include_replay:
        prompts.append(
            request(
                "prompt-009-cache-replay",
                "cache-replay",
                "The capital of France is",
                "replay",
                cache_replay_stage_id=1,
                cache_replay_after_tokens=4,
            )
        )
    return prompts


def _validate_or_build_shards(
    *,
    config: RealExperimentConfig,
    description: Any,
    repository_root: Path,
    skip_sharding: bool,
) -> tuple[Path, Any, dict[str, Any]]:
    shard_root = (repository_root / config.model.shard_directory).resolve()
    manifest_path = shard_root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        if (
            manifest.model_id != description.model_id
            or manifest.model_revision != description.model_revision
            or len(manifest.stages) != 4
        ):
            raise IntegrityError(
                "existing shard directory does not match the requested immutable "
                "model revision and four-stage layout"
            )
        verify_manifest_shards(manifest, shard_root)
    else:
        if skip_sharding:
            raise IntegrityError(
                f"-SkipSharding requested but no manifest exists at {manifest_path}"
            )
        maximum_stage_bytes = sum(item.bytes for item in description.tensors) - 1
        manifest = shard_model(
            description,
            output=shard_root,
            target_stage_bytes=math.ceil(maximum_stage_bytes / 4),
            maximum_stage_bytes=maximum_stage_bytes,
            stage_count=4,
        )
    validation_path = shard_root / "validation.json"
    if not validation_path.is_file():
        raise IntegrityError(f"shard validation artifact is missing: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    verify_manifest_shards(manifest, shard_root)
    validation["stage_hashes_valid"] = True
    return shard_root, manifest, validation


def _chart(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if values:
        axis.bar(labels, values, color="#4C78A8")
        axis.tick_params(axis="x", rotation=30)
    else:
        axis.text(0.5, 0.5, "No evidence recorded", ha="center", va="center")
        axis.set_xticks([])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_charts(
    run_dir: Path,
    manifest: Any,
    distributed: dict[str, Any],
) -> None:
    charts = run_dir / "charts"
    stage_labels = [f"stage {stage.stage_id}" for stage in manifest.stages]
    _chart(
        charts / "stage_weight_bytes.png",
        title="Stage weight bytes",
        labels=stage_labels,
        values=[float(stage.required_memory_bytes) for stage in manifest.stages],
        ylabel="bytes",
    )
    worker_proofs = distributed.get("worker_load_proofs", [])
    _chart(
        charts / "worker_memory.png",
        title="Peak CUDA allocation by worker",
        labels=[str(item.get("worker_id", "worker")) for item in worker_proofs],
        values=[
            float(item.get("cuda_memory_after_load", {}).get("cuda_peak_allocated_bytes", 0))
            for item in worker_proofs
        ],
        ylabel="bytes",
    )
    prompt_results = distributed.get("prompt_results", [])
    _chart(
        charts / "prefill_latency.png",
        title="Time to first token",
        labels=[str(item.get("name", "prompt")) for item in prompt_results],
        values=[float(item.get("time_to_first_token_s") or 0) * 1000 for item in prompt_results],
        ylabel="milliseconds",
    )
    _chart(
        charts / "decode_latency.png",
        title="Decode latency",
        labels=[str(item.get("name", "prompt")) for item in prompt_results],
        values=[
            max(
                0.0,
                (
                    float(item.get("end_to_end_latency_s") or 0)
                    - float(item.get("time_to_first_token_s") or 0)
                )
                * 1000,
            )
            for item in prompt_results
        ],
        ylabel="milliseconds",
    )
    boundaries = distributed.get("boundary_diagnostics", [])
    _chart(
        charts / "boundary_errors.png",
        title="Maximum absolute boundary error",
        labels=[f"{item.get('request_id')} / s{item.get('stage_id')}" for item in boundaries],
        values=[float(item.get("maximum_absolute_error", 0)) for item in boundaries],
        ylabel="absolute error",
    )
    _chart(
        charts / "activation_bytes.png",
        title="Worker-to-worker activation bytes",
        labels=[str(item.get("name", "prompt")) for item in prompt_results],
        values=[
            float(item.get("transport_metrics", {}).get("worker_to_worker_activation_bytes", 0))
            for item in prompt_results
        ],
        ylabel="bytes",
    )
    cache_by_stage = distributed.get("cache_metrics", {}).get("maximum_cache_bytes_by_stage", {})
    _chart(
        charts / "kv_cache_bytes.png",
        title="Maximum stage-local KV-cache bytes",
        labels=stage_labels,
        values=[float(cache_by_stage.get(str(stage.stage_id), 0)) for stage in manifest.stages],
        ylabel="bytes",
    )


def _render_report_legacy(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    distributed: dict[str, Any],
    fatal_error: dict[str, Any] | None,
) -> Path:
    status_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td class='{value.lower()}'>{value}</td></tr>"
        for key, value in summary.items()
        if key.endswith("_status")
    )
    stages = "".join(
        "<tr>"
        f"<td>{stage['stage_id']}</td>"
        f"<td>[{stage['layer_start']}, {stage['layer_end']})</td>"
        f"<td>{stage['required_memory_bytes']:,}</td>"
        f"<td>{stage.get('tensor_count', len(stage.get('tensor_names', [])))}</td>"
        "</tr>"
        for stage in manifest.get("stages", [])
    )
    prompts = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('name')))}</td>"
        f"<td>{item.get('input_token_count')}</td>"
        f"<td>{item.get('output_token_count')}</td>"
        f"<td>{'PASS' if item.get('token_identity') else 'FAIL'}</td>"
        f"<td>{float(item.get('end_to_end_latency_s') or 0):.3f}</td>"
        "</tr>"
        for item in distributed.get("prompt_results", [])
    )
    error_html = (
        f"<h2>Fatal error</h2><pre>{html.escape(json.dumps(fatal_error, indent=2))}</pre>"
        if fatal_error
        else ""
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Experiment 002 — Real Distributed Qwen3 Inference</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd1d1;padding:.5rem;text-align:left}}
.pass{{color:#117a65;font-weight:700}}.fail{{color:#b03a2e;font-weight:700}}
code,pre{{background:#f4f6f7;padding:.2rem .35rem}}img{{max-width:48%;margin:.5rem}}
</style></head><body>
<h1>Experiment 002 — Real Distributed Qwen3 Inference</h1>
<p><strong>Overall status:</strong> <span class="{summary["overall_status"].lower()}">{summary["overall_status"]}</span></p>
<h2>Acceptance status</h2><table>{status_rows}</table>
<h2>Immutable model and partition</h2>
<p>{html.escape(str(manifest.get("model_id")))} @ <code>{html.escape(str(manifest.get("model_revision")))}</code></p>
<p>Source weights: {int(manifest.get("total_weight_bytes", 0)):,} bytes. Sharded weights:
{int(manifest.get("total_sharded_weight_bytes") or 0):,} bytes. Explicit tied-weight duplication:
{int(manifest.get("duplicated_tensor_bytes", 0)):,} bytes.</p>
<table><tr><th>Stage</th><th>Layers</th><th>Weight bytes</th><th>Tensors</th></tr>{stages}</table>
<h2>Prompt correctness</h2>
<table><tr><th>Prompt</th><th>Input tokens</th><th>Output tokens</th><th>Exact identity</th><th>End-to-end s</th></tr>{prompts}</table>
<h2>Architecture evidence</h2>
<pre>{html.escape(json.dumps(distributed.get("transport_metrics", {}), indent=2))}</pre>
<h2>Limitations</h2>
<p>This is single-host loopback process isolation on one Windows 11 machine and one RTX 5090.
It does not prove multi-machine, LAN, WAN, Raspberry Pi, Kimi K3, additional physical compute,
or single-request speedup from partitioning.</p>
<h2>Charts</h2>
<img src="charts/stage_weight_bytes.png"><img src="charts/worker_memory.png">
<img src="charts/prefill_latency.png"><img src="charts/decode_latency.png">
<img src="charts/boundary_errors.png"><img src="charts/activation_bytes.png">
<img src="charts/kv_cache_bytes.png">
{error_html}
</body></html>"""
    report = run_dir / "report.html"
    report.write_text(document, encoding="utf-8")
    return report


def _render_report(
    *,
    run_dir: Path,
    summary: dict[str, Any],
    environment: dict[str, Any],
    git: dict[str, Any],
    inspection: dict[str, Any],
    manifest: dict[str, Any],
    shard_validation: dict[str, Any],
    reference: dict[str, Any],
    distributed: dict[str, Any],
    quality_gates: dict[str, Any],
    log_validation: dict[str, Any],
    fatal_error: dict[str, Any] | None,
) -> Path:
    """Render the complete reader-facing Experiment 002 evidence report."""

    def esc(value: Any) -> str:
        return html.escape(str(value))

    def as_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def as_float(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    def json_html(value: Any) -> str:
        serialised = json.dumps(value, indent=2, sort_keys=True)
        return f"<pre>{html.escape(serialised)}</pre>"

    resolved: dict[str, Any] = {}
    resolved_path = run_dir / "config.resolved.yaml"
    if resolved_path.is_file():
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            resolved = loaded

    status_rows = "".join(
        f"<tr><th>{esc(key)}</th><td class='{value.lower()}'>{value}</td></tr>"
        for key, value in summary.items()
        if key.endswith("_status")
    )
    gpu = environment.get("gpu", {})
    environment_rows = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>"
        for label, value in (
            ("Platform", environment.get("platform")),
            ("GPU", gpu.get("model")),
            ("Compute capability", gpu.get("compute_capability")),
            ("BF16 supported", gpu.get("bf16_supported")),
            ("Total VRAM bytes", f"{as_int(gpu.get('total_vram_bytes')):,}"),
            ("Available VRAM bytes", f"{as_int(gpu.get('available_vram_bytes')):,}"),
            ("System RAM bytes", f"{as_int(environment.get('system_ram_total_bytes')):,}"),
            (
                "Available system RAM bytes",
                f"{as_int(environment.get('system_ram_available_bytes')):,}",
            ),
            ("Required packages preserved", environment.get("required_packages_preserved")),
        )
    )
    software_rows = "".join(
        f"<tr><th>{esc(label)}</th><td>{esc(environment.get(key))}</td></tr>"
        for label, key in (
            ("Python", "python_version"),
            ("PyTorch", "pytorch_version"),
            ("Transformers", "transformers_version"),
            ("Safetensors", "safetensors_version"),
            ("CUDA runtime", "cuda_runtime_version"),
            ("NVIDIA driver", "nvidia_driver_version"),
        )
    )
    hash_rows: list[str] = []
    for group, key in (
        ("config", "config_hashes"),
        ("tokenizer", "tokenizer_hashes"),
        ("safetensors", "safetensors_hashes"),
    ):
        values = inspection.get(key, {})
        if isinstance(values, dict):
            hash_rows.extend(
                f"<tr><td>{group}</td><td>{esc(name)}</td><td><code>{esc(digest)}</code></td></tr>"
                for name, digest in sorted(values.items())
            )
    stage_rows = "".join(
        "<tr>"
        f"<td>{stage.get('stage_id')}</td>"
        f"<td>[{stage.get('layer_start')}, {stage.get('layer_end')})</td>"
        f"<td>{as_int(stage.get('required_memory_bytes')):,}</td>"
        f"<td>{as_int(stage.get('required_total_memory_bytes')):,}</td>"
        f"<td>{as_int(stage.get('tensor_count'))}</td>"
        f"<td>{'yes' if stage.get('owns_embeddings') else 'no'}</td>"
        f"<td>{'yes' if stage.get('owns_final_norm') else 'no'}</td>"
        f"<td>{'yes' if stage.get('owns_output_head') else 'no'}</td>"
        "</tr>"
        for stage in manifest.get("stages", [])
    )
    shard_hash_rows = "".join(
        f"<tr><td>{esc(stage_id)}</td><td><code>{esc(digest)}</code></td></tr>"
        for stage_id, digest in sorted(manifest.get("shard_hashes", {}).items())
    )
    worker_proofs = distributed.get("worker_load_proofs", [])
    worker_rows = "".join(
        "<tr>"
        f"<td>{esc(proof.get('worker_id'))}</td>"
        f"<td>{as_int(proof.get('process_id'))}</td>"
        f"<td>{as_int(proof.get('stage_id'))}</td>"
        f"<td>{esc(proof.get('decoder_layer_range'))}</td>"
        f"<td>{as_int(proof.get('tensor_count'))}</td>"
        f"<td>{as_int(proof.get('total_loaded_weight_bytes')):,}</td>"
        f"<td>{'no' if proof.get('loaded_complete_source_tensor_set') is False else 'YES'}</td>"
        f"<td>{'PASS' if proof.get('proof_verified') else 'FAIL'}</td>"
        "</tr>"
        for proof in worker_proofs
    )
    memory_rows = "".join(
        "<tr>"
        f"<td>{esc(proof.get('worker_id'))}</td>"
        f"<td>{as_int(proof.get('host_rss_before_load')):,}</td>"
        f"<td>{as_int(proof.get('host_rss_after_load')):,}</td>"
        f"<td>{as_int(proof.get('cuda_memory_after_load', {}).get('cuda_allocated_bytes')):,}</td>"
        f"<td>{as_int(proof.get('peak_cuda_memory_bytes')):,}</td>"
        f"<td>{as_int(proof.get('peak_cuda_reserved_bytes')):,}</td>"
        "</tr>"
        for proof in worker_proofs
    )
    copy_rows = "".join(
        "<tr>"
        f"<td>{esc(proof.get('worker_id'))}</td>"
        f"<td>{as_int(proof.get('stage_id'))}</td>"
        f"<td>{as_int(proof.get('stage_transfer_metrics', {}).get('operation_count'))}</td>"
        f"<td>{as_float(proof.get('stage_transfer_metrics', {}).get('host_to_device_copy_ms')):.3f}</td>"
        f"<td>{as_float(proof.get('stage_transfer_metrics', {}).get('device_to_host_copy_ms')):.3f}</td>"
        f"<td>{as_float(proof.get('stage_transfer_metrics', {}).get('cuda_execution_ms')):.3f}</td>"
        f"<td>{as_int(proof.get('stage_transfer_metrics', {}).get('host_to_device_bytes')):,}</td>"
        f"<td>{as_int(proof.get('stage_transfer_metrics', {}).get('device_to_host_bytes')):,}</td>"
        "</tr>"
        for proof in worker_proofs
    )
    prompt_rows = "".join(
        "<tr>"
        f"<td>{esc(result.get('name'))}</td>"
        f"<td>{as_int(result.get('input_token_count'))}</td>"
        f"<td>{as_int(result.get('output_token_count'))}</td>"
        f"<td class='{'pass' if result.get('token_identity') else 'fail'}'>"
        f"{'PASS' if result.get('token_identity') else 'FAIL'}</td>"
        f"<td><code>{esc(result.get('prompt_token_ids', []))}</code></td>"
        f"<td><code>{esc(result.get('reference_generated_token_ids', []))}</code></td>"
        f"<td><code>{esc(result.get('distributed_generated_token_ids', []))}</code></td>"
        f"<td><pre>{esc(result.get('reference_decoded_text', ''))}</pre></td>"
        f"<td><pre>{esc(result.get('distributed_decoded_text', ''))}</pre></td>"
        "</tr>"
        for result in distributed.get("prompt_results", [])
    )
    boundary_rows = "".join(
        "<tr>"
        f"<td>{esc(result.get('request_id'))}</td>"
        f"<td>{as_int(result.get('stage_id'))}</td>"
        f"<td>{as_float(result.get('maximum_absolute_error')):.8g}</td>"
        f"<td>{as_float(result.get('mean_absolute_error')):.8g}</td>"
        f"<td>{as_float(result.get('maximum_relative_error')):.8g}</td>"
        f"<td>{as_float(result.get('cosine_similarity')):.12f}</td>"
        f"<td>{as_int(result.get('nan_count'))}</td>"
        f"<td>{as_int(result.get('inf_count'))}</td>"
        f"<td class='{'pass' if result.get('within_tolerance') else 'fail'}'>"
        f"{'PASS' if result.get('within_tolerance') else 'FAIL'}</td>"
        "</tr>"
        for result in distributed.get("boundary_diagnostics", [])
    )
    cache = distributed.get("cache_metrics", {})
    cache_maximums = cache.get("maximum_cache_bytes_by_stage", {})
    cache_rows = "".join(
        "<tr>"
        f"<td>{stage.get('stage_id')}</td>"
        f"<td>[{stage.get('layer_start')}, {stage.get('layer_end')})</td>"
        f"<td>{as_int(cache_maximums.get(str(stage.get('stage_id')))):,}</td>"
        "</tr>"
        for stage in manifest.get("stages", [])
    )
    timing_rows = ""
    for result in distributed.get("prompt_results", []):
        metrics = result.get("request_metrics", {})
        per_stage = metrics.get("per_stage", [])
        prefill_ms = sum(
            as_float(item.get("execution_ms"))
            for item in per_stage
            if as_int(item.get("token_position")) == 0
        )
        decode_ms = sum(
            as_float(item.get("execution_ms"))
            for item in per_stage
            if as_int(item.get("token_position")) != 0
        )
        timing_rows += (
            "<tr>"
            f"<td>{esc(result.get('name'))}</td>"
            f"<td>{as_float(result.get('time_to_first_token_s')) * 1000:.3f}</td>"
            f"<td>{prefill_ms:.3f}</td>"
            f"<td>{decode_ms:.3f}</td>"
            f"<td>{as_float(result.get('end_to_end_latency_s')) * 1000:.3f}</td>"
            f"<td>{as_float(metrics.get('transport_s')) * 1000:.3f}</td>"
            "</tr>"
        )
    quality_rows = "".join(
        "<tr>"
        f"<td>{esc(gate.get('name'))}</td>"
        f"<td><code>{esc(gate.get('command'))}</code></td>"
        f"<td class='{str(gate.get('status', 'FAIL')).lower()}'>"
        f"{esc(gate.get('status'))}</td>"
        f"<td>{as_float(gate.get('duration_seconds')):.3f}</td>"
        f"<td><pre>{esc(chr(10).join(gate.get('output', [])))}</pre></td>"
        "</tr>"
        for gate in quality_gates.get("gates", [])
    )
    artifact_paths = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and ".reference-boundaries" not in path.parts
        and path.suffix.lower() != ".pem"
    ]
    artifact_rows = "".join(
        "<tr>"
        f"<td><a href='{esc(str(path.relative_to(run_dir)).replace(chr(92), '/'))}'>"
        f"{esc(str(path.relative_to(run_dir)).replace(chr(92), '/'))}</a></td>"
        f"<td>{path.stat().st_size:,}</td>"
        "</tr>"
        for path in artifact_paths
    )
    artifact_rows += "<tr><td><a href='report.html'>report.html</a></td><td>this document</td></tr>"
    largest_stage = max(
        (as_int(stage.get("required_memory_bytes")) for stage in manifest.get("stages", [])),
        default=0,
    )
    worker_config = resolved.get("workers", {})
    conclusion = (
        "PASS: A real Qwen3 model was split across four process-isolated workers, "
        "real hidden states crossed worker boundaries, stage-local KV caches were used, "
        "and distributed greedy output matched the independent full-model reference exactly."
        if summary.get("overall_status") == "PASS"
        else "FAIL: The real distributed Qwen3 experiment did not satisfy all correctness "
        "and isolation criteria."
    )
    fatal_html = f"<h2>Fatal error</h2>{json_html(fatal_error)}" if fatal_error else ""
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Experiment 002 - Real Distributed Qwen3 Inference</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#17202a}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd1d1;padding:.5rem;text-align:left;vertical-align:top}}
.pass{{color:#117a65;font-weight:700}}.fail{{color:#b03a2e;font-weight:700}}
code,pre{{background:#f4f6f7;padding:.2rem .35rem;white-space:pre-wrap;overflow-wrap:anywhere}}
img{{max-width:48%;margin:.5rem}}details{{margin:.75rem 0}}.compact td,.compact th{{font-size:.88rem}}
</style></head><body>
<h1>Experiment 002 - Real Distributed Qwen3 Inference</h1>
<h2>Executive summary</h2>
<p><strong>Overall status:</strong> <span class="{str(summary.get("overall_status", "FAIL")).lower()}">{esc(summary.get("overall_status"))}</span></p>
<p>{esc(conclusion)}</p>
<p>Execution mode: <code>single-host-loopback-real-model</code>. Four process-isolated
workers shared one RTX 5090. This run tested correctness and isolation, not speedup.</p>
<h2>Acceptance status</h2><table>{status_rows}</table>
<h2>Environment</h2><table>{environment_rows}</table>
<h2>Software versions</h2><table>{software_rows}</table>
<h2>Git state</h2>{json_html(git)}
<h2>Immutable model revision</h2>
<p>{esc(manifest.get("model_id"))} @ <code>{esc(manifest.get("model_revision"))}</code></p>
<p>Local snapshot: <code>{esc(inspection.get("local_snapshot_path"))}</code></p>
<h2>Model-file hash table</h2>
<table><tr><th>Kind</th><th>File</th><th>SHA-256</th></tr>{"".join(hash_rows)}</table>
<h2>Model architecture</h2>
<table>
<tr><th>Architecture</th><td>{esc(inspection.get("architecture"))}</td></tr>
<tr><th>Decoder layers</th><td>{as_int(inspection.get("decoder_layer_count"))}</td></tr>
<tr><th>Hidden / intermediate</th><td>{as_int(inspection.get("hidden_size"))} / {as_int(inspection.get("intermediate_size"))}</td></tr>
<tr><th>Attention / KV heads</th><td>{as_int(inspection.get("attention_head_count"))} / {as_int(inspection.get("key_value_head_count"))}</td></tr>
<tr><th>Vocabulary</th><td>{as_int(inspection.get("vocabulary_size"))}</td></tr>
<tr><th>Maximum positions</th><td>{as_int(inspection.get("maximum_position_embeddings"))}</td></tr>
<tr><th>Weight dtype</th><td>{esc(inspection.get("weight_dtype"))}</td></tr>
<tr><th>Tied embeddings</th><td>{esc(inspection.get("tied_embeddings"))}</td></tr>
</table>
<h2>Model inspection</h2>{json_html(inspection)}
<h2>Partition plan and stage ownership</h2>
<p>Source weights: {as_int(manifest.get("total_weight_bytes")):,} bytes; sharded weights:
{as_int(manifest.get("total_sharded_weight_bytes")):,} bytes; explicit duplication:
{as_int(manifest.get("duplicated_tensor_bytes")):,} bytes.</p>
<table><tr><th>Stage</th><th>Layers</th><th>Weight bytes</th><th>Total estimate</th><th>Tensors</th><th>Embeddings</th><th>Final norm</th><th>LM head</th></tr>{stage_rows}</table>
<p>Tied-weight treatment: <code>{esc(manifest.get("tied_weight_treatment"))}</code>.</p>
<h2>Shard hashes and reconstruction validation</h2>
<table><tr><th>Stage</th><th>SHA-256</th></tr>{shard_hash_rows}</table>{json_html(shard_validation)}
<h2>Logical memory-limit proof</h2>
<table>
<tr><th>Full model weight bytes</th><td>{as_int(manifest.get("total_weight_bytes")):,}</td></tr>
<tr><th>Largest stage weight bytes</th><td>{largest_stage:,}</td></tr>
<tr><th>Worker logical weight limit</th><td>{as_int(worker_config.get("logical_weight_limit_bytes")):,}</td></tr>
<tr><th>Worker logical total-memory limit</th><td>{as_int(worker_config.get("logical_total_memory_limit_bytes")):,}</td></tr>
<tr><th>Full model exceeds each weight limit</th><td>{esc(distributed.get("model_larger_than_each_worker_limit"))}</td></tr>
<tr><th>Full model exceeds each total-memory limit</th><td>{esc(distributed.get("model_larger_than_each_worker_total_limit"))}</td></tr>
</table>
<h2>Worker load proofs</h2>
<table class="compact"><tr><th>Worker</th><th>PID</th><th>Stage</th><th>Layers</th><th>Tensors</th><th>Loaded bytes</th><th>Full set loaded?</th><th>Checksum/signature</th></tr>{worker_rows}</table>
<details><summary>Complete checksummed load proofs</summary>{json_html(worker_proofs)}</details>
<h2>Per-worker CUDA and host memory</h2>
<table><tr><th>Worker</th><th>Host RSS before</th><th>Host RSS after</th><th>CUDA after load</th><th>Peak CUDA allocated</th><th>Peak CUDA reserved</th></tr>{memory_rows}</table>
<h2>CUDA execution and host-staged copy accounting</h2>
<table><tr><th>Worker</th><th>Stage</th><th>Operations</th><th>Host-to-device ms</th><th>Device-to-host ms</th><th>CUDA execution ms</th><th>Host-to-device bytes</th><th>Device-to-host bytes</th></tr>{copy_rows}</table>
<h2>Coordinator load proof</h2>{json_html(distributed.get("coordinator_proof", {}))}
<h2>Direct worker-to-worker data-plane metrics</h2>{json_html(distributed.get("transport_metrics", {}))}
<h2>Independent full-model reference</h2>
<p>PID {as_int(reference.get("process_id"))}; full model loaded:
{esc(reference.get("full_model_loaded"))}; counted as swarm memory:
{esc(reference.get("memory_counted_as_swarm"))}.</p>{json_html(reference.get("environment", {}))}
<h2>Prompt IDs, generated IDs, decoded outputs, and exact token identity</h2>
<table class="compact"><tr><th>Prompt</th><th>Input</th><th>Output</th><th>Identity</th><th>Prompt IDs</th><th>Reference IDs</th><th>Distributed IDs</th><th>Reference output</th><th>Distributed output</th></tr>{prompt_rows}</table>
<h2>Boundary error statistics</h2>
<p>atol={esc(resolved.get("correctness", {}).get("boundary_atol"))};
rtol={esc(resolved.get("correctness", {}).get("boundary_rtol"))};
minimum cosine={esc(resolved.get("correctness", {}).get("minimum_cosine_similarity"))}.</p>
<table class="compact"><tr><th>Request</th><th>Stage</th><th>Max abs</th><th>Mean abs</th><th>Max relative</th><th>Cosine</th><th>NaN</th><th>Inf</th><th>Status</th></tr>{boundary_rows}</table>
<h2>Per-stage KV-cache evidence</h2>
<table><tr><th>Stage</th><th>Owned layers</th><th>Maximum cache bytes</th></tr>{cache_rows}</table>
<p>Total owned layers: {as_int(cache.get("owned_layer_count"))}; active after completion:
{as_int(cache.get("active_cache_count_after_completion"))}; stale cache:
{esc(cache.get("stale_request_cache_remains"))}.</p>
<details><summary>Cache lifecycle records</summary>{json_html(cache.get("histories", []))}</details>
<h2>Cache replay result</h2>{json_html(distributed.get("cache_replay", {}))}
<h2>Timing results</h2>
<table><tr><th>Prompt</th><th>TTFT ms</th><th>Prefill execution ms</th><th>Decode execution ms</th><th>End-to-end ms</th><th>Transport ms</th></tr>{timing_rows}</table>
<h2>Unit, CPU, CUDA, Ruff, mypy, and pytest results</h2>
<p>Quality status: <span class="{str(quality_gates.get("overall_status", "FAIL")).lower()}">{esc(quality_gates.get("overall_status"))}</span>.
CUDA evidence: {esc(quality_gates.get("cuda_integration", {}))}</p>
<table><tr><th>Gate</th><th>Command</th><th>Status</th><th>Seconds</th><th>Output</th></tr>{quality_rows}</table>
<h2>Log and cleanup validation</h2>{json_html(log_validation)}
{json_html(distributed.get("cleanup", {}))}
<h2>Artifact index</h2>
<table><tr><th>Artifact</th><th>Bytes</th></tr>{artifact_rows}</table>
<h2>Known limitations</h2>
<p>This is single-host loopback process isolation on one Windows 11 machine and one
RTX 5090. It does not prove multi-machine, LAN, WAN, Raspberry Pi, Kimi K3, additional
physical compute, or single-request speedup from partitioning.</p>
<h2>Charts</h2>
<img src="charts/stage_weight_bytes.png"><img src="charts/worker_memory.png">
<img src="charts/prefill_latency.png"><img src="charts/decode_latency.png">
<img src="charts/boundary_errors.png"><img src="charts/activation_bytes.png">
<img src="charts/kv_cache_bytes.png">
{fatal_html}
<h2>Conclusion</h2><p><strong>{esc(conclusion)}</strong></p>
</body></html>"""
    report = run_dir / "report.html"
    report.write_text(document, encoding="utf-8")
    return report


def run_experiment_002(
    *,
    config_path: str | Path,
    model_id: str | None = None,
    revision: str | None = None,
    max_new_tokens: int | None = None,
    output_root: str | Path | None = None,
    skip_download: bool = False,
    skip_sharding: bool = False,
    skip_prompt_suite: bool = False,
    skip_replay_test: bool = False,
    keep_workers: bool = False,
) -> Experiment002Run:
    if keep_workers:
        raise ValueError(
            "-KeepWorkers is incompatible with mandatory cleanup evidence in Experiment 002"
        )
    repository_root = Path.cwd().resolve()
    requested_config_path = Path(config_path).expanduser().resolve()
    config = load_real_experiment_config(requested_config_path)
    if model_id is not None:
        config.model.model_id = model_id
    if revision is not None:
        config.model.revision = revision
    if max_new_tokens is not None:
        config.generation.max_new_tokens = max_new_tokens
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (repository_root / "artifacts" / "runs").resolve()
    )
    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-qwen3-real-loopback-" + uuid4().hex[:8]
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "tensors").mkdir()
    (run_dir / "charts").mkdir()
    shutil.copy2(requested_config_path, run_dir / "config.requested.yaml")
    git = _git_evidence(repository_root)
    _write_json(run_dir / "git.json", git)
    environment = _environment_probe(
        run_dir / "environment.json",
        run_dir / "logs" / "environment.log",
    )
    quality_gates = _load_quality_evidence()
    fatal_error: dict[str, Any] | None = None
    inspection: dict[str, Any] = {}
    manifest_payload: dict[str, Any] = {}
    shard_validation: dict[str, Any] = {}
    distributed: dict[str, Any] = {
        "prompt_results": [],
        "worker_load_proofs": [],
        "coordinator_proof": {},
        "transport_metrics": {},
        "cache_metrics": {},
        "boundary_diagnostics": [],
        "cache_replay": {},
    }
    reference: dict[str, Any] = {"results": []}
    manifest: Any | None = None
    boundary_root: Path | None = None
    try:
        resolved = resolve_model(
            config.model.model_id,
            revision=config.model.revision,
            allow_download=not skip_download,
        )
        if config.model.model_id != "Qwen/Qwen3-0.6B" and model_id is None:
            raise IntegrityError(
                "the default Qwen3 model was changed without an explicit -ModelId selection"
            )
        description = inspect_qwen3_model(resolved)
        inspection = model_inspection_payload(description)
        if int(inspection["parameter_count"]) >= 1_000_000_000:
            raise IntegrityError(
                "Experiment 002 only supports an explicitly selected Qwen3 model below "
                "one billion parameters"
            )
        _write_json(run_dir / "model_inspection.json", inspection)
        shard_root, manifest, shard_validation = _validate_or_build_shards(
            config=config,
            description=description,
            repository_root=repository_root,
            skip_sharding=skip_sharding,
        )
        manifest_payload = manifest.model_dump(mode="json")
        _write_json(run_dir / "model_manifest.json", manifest_payload)
        hashes = json.loads((shard_root / "hashes.json").read_text(encoding="utf-8"))
        _write_json(run_dir / "shard_hashes.json", hashes)
        largest_stage = max(stage.required_memory_bytes for stage in manifest.stages)
        logical_weight_limit = (
            config.workers.logical_weight_limit_bytes or largest_stage + 128 * 1024 * 1024
        )
        if not largest_stage < logical_weight_limit < manifest.total_weight_bytes:
            raise IntegrityError(
                "resolved logical weight limit must exceed the largest stage and remain "
                "below the complete model"
            )
        logical_total_limit = (
            config.workers.logical_total_memory_limit_bytes
            or max(
                stage.required_total_memory_bytes or stage.required_memory_bytes
                for stage in manifest.stages
            )
            + 512 * 1024 * 1024
        )
        config.model.revision = resolved.revision
        config.workers.logical_weight_limit_bytes = logical_weight_limit
        config.workers.logical_total_memory_limit_bytes = logical_total_limit
        resolved_config = config.model_dump(mode="json")
        resolved_config["model"]["resolved_local_snapshot_path"] = str(resolved.path)
        resolved_config["model"]["source_weight_bytes"] = manifest.total_weight_bytes
        resolved_config["model"]["largest_stage_weight_bytes"] = largest_stage
        resolved_config["model"]["stage_layer_ranges"] = [
            [stage.layer_start, stage.layer_end] for stage in manifest.stages
        ]
        (run_dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved_config, sort_keys=False),
            encoding="utf-8",
        )
        requests = build_prompt_suite(
            model_path=resolved.path,
            max_new_tokens=config.generation.max_new_tokens,
            include_prompt_suite=not skip_prompt_suite,
            include_replay=not skip_replay_test,
        )
        reference_requests = [
            {
                "request_id": item["request_id"],
                "name": item["name"],
                "prompt": item["prompt"],
                "prompt_token_ids": item["prompt_token_ids"],
                "max_new_tokens": item["max_new_tokens"],
            }
            for item in requests
        ]
        reference = run_reference_suite_subprocess(
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            model_path=resolved.path,
            requests=reference_requests,
            device=config.model.device,
            dtype_name=config.model.dtype,
            stage_layer_ends=[stage.layer_end for stage in manifest.stages],
            output_dir=run_dir,
            log_path=run_dir / "logs" / "reference.log",
        )
        boundary_root = Path(reference["boundary_root"]).resolve()
        architecture_config = json.loads(
            (resolved.path / "config.json").read_text(encoding="utf-8")
        )
        runtime = config.runtime_config(
            model_layer_count=manifest.layer_count,
            model_hidden_size=manifest.hidden_size,
            logical_weight_limit_bytes=logical_weight_limit,
        )
        runtime.model_revision = resolved.revision
        distributed = asyncio.run(
            run_qwen3_experiment_session(
                experiment=runtime,
                manifest=manifest,
                architecture_config=architecture_config,
                shard_root=shard_root,
                model_path=resolved.path,
                output_dir=run_dir,
                requests=requests,
                reference_results=reference["results"],
                dtype=config.model.dtype,
                logical_weight_limit_bytes=logical_weight_limit,
                logical_total_memory_limit_bytes=logical_total_limit,
                boundary_reference_root=boundary_root,
                boundary_atol=config.correctness.boundary_atol,
                boundary_rtol=config.correctness.boundary_rtol,
                minimum_cosine_similarity=(config.correctness.minimum_cosine_similarity),
                worker_start_timeout_s=config.timeouts.worker_start_seconds,
                shutdown_timeout_s=config.timeouts.shutdown_seconds,
            )
        )
    except Exception as exc:
        fatal_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _write_json(run_dir / "fatal_error.json", fatal_error)
    finally:
        for index in range(4):
            worker_log = run_dir / "logs" / f"worker-{index:03d}.log"
            if not worker_log.exists():
                worker_log.write_text(
                    "Worker was not started because the experiment failed earlier.\n",
                    encoding="utf-8",
                )
        coordinator_log = run_dir / "logs" / "coordinator.log"
        if not coordinator_log.exists():
            coordinator_log.write_text(
                (
                    json.dumps(fatal_error, sort_keys=True) + "\n"
                    if fatal_error
                    else "No coordinator events recorded.\n"
                ),
                encoding="utf-8",
            )
        reference_log = run_dir / "logs" / "reference.log"
        if not reference_log.exists():
            reference_log.write_text(
                "Reference process was not started because the experiment failed earlier.\n",
                encoding="utf-8",
            )
        post_environment_path = run_dir / "environment.post.json"
        try:
            post_environment = _environment_probe(post_environment_path)
            environment["post_run"] = post_environment
            environment["required_packages_preserved"] = all(
                post_environment.get(key) == environment.get(key)
                for key in (
                    "pytorch_version",
                    "transformers_version",
                    "safetensors_version",
                )
            )
            _write_json(run_dir / "environment.json", environment)
        except Exception as exc:
            environment["required_packages_preserved"] = False
            environment["post_run_probe_error"] = str(exc)
            _write_json(run_dir / "environment.json", environment)
        if post_environment_path.exists():
            post_environment_path.unlink()

    _write_json(run_dir / "quality_gates.json", quality_gates)
    log_validation = _validate_logs(run_dir)
    distributed["log_validation"] = log_validation
    bundle = {
        "environment": environment,
        "manifest": manifest_payload,
        "shard_validation": shard_validation,
        "reference": reference,
        "distributed": distributed,
        "quality_gates": quality_gates,
    }
    statuses = evaluate_experiment_002_status(bundle)
    prompt_results = distributed.get("prompt_results", [])
    correctness = {
        "require_exact_token_identity": True,
        "all_tokens_identical": bool(
            prompt_results and all(item.get("token_identity") is True for item in prompt_results)
        ),
        "token_mismatches": [
            item["mismatch"] for item in prompt_results if item.get("mismatch") is not None
        ],
        "boundary_checks": distributed.get("boundary_diagnostics", []),
        "all_boundaries_within_tolerance": bool(
            distributed.get("boundary_diagnostics")
            and all(
                item.get("within_tolerance") is True
                for item in distributed.get("boundary_diagnostics", [])
            )
        ),
    }
    _write_json(run_dir / "distributed.json", distributed)
    _write_json(run_dir / "correctness.json", correctness)
    _write_json(
        run_dir / "worker_load_proofs.json",
        distributed.get("worker_load_proofs", []),
    )
    _write_json(
        run_dir / "coordinator_proof.json",
        distributed.get("coordinator_proof", {}),
    )
    _write_json(
        run_dir / "transport_metrics.json",
        distributed.get("transport_metrics", {}),
    )
    _write_json(
        run_dir / "cache_metrics.json",
        distributed.get("cache_metrics", {}),
    )
    _write_jsonl(run_dir / "prompt_results.jsonl", prompt_results)
    _write_json(
        run_dir / "cache_replay.json",
        distributed.get("cache_replay", {}),
    )
    _write_jsonl(run_dir / "events.jsonl", distributed.get("events", []))
    _write_json(
        run_dir / "tensors" / "boundary-diagnostics.json",
        distributed.get("boundary_diagnostics", []),
    )
    _write_json(run_dir / "log_validation.json", log_validation)
    if manifest is not None:
        try:
            _write_charts(run_dir, manifest, distributed)
        except Exception as exc:
            fatal_error = fatal_error or {
                "type": type(exc).__name__,
                "message": f"chart generation failed: {exc}",
            }
    else:
        for name in (
            "stage_weight_bytes.png",
            "worker_memory.png",
            "prefill_latency.png",
            "decode_latency.png",
            "boundary_errors.png",
            "activation_bytes.png",
            "kv_cache_bytes.png",
        ):
            (run_dir / "charts" / name).write_bytes(b"")
    if fatal_error is not None:
        statuses["experiment_integrity_status"] = "FAIL"
        statuses["overall_status"] = "FAIL"
    summary: dict[str, Any] = {
        **statuses,
        "experiment_name": config.name,
        "execution_mode": config.execution_mode,
        "model_id": manifest_payload.get("model_id", config.model.model_id),
        "model_revision": manifest_payload.get("model_revision", config.model.revision),
        "run_directory": str(run_dir),
        "report_path": str(run_dir / "report.html"),
        "fatal_error": fatal_error,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(run_dir / "summary.json", summary)
    report = _render_report(
        run_dir=run_dir,
        summary=summary,
        environment=environment,
        git=git,
        inspection=inspection,
        manifest=manifest_payload,
        shard_validation=shard_validation,
        reference=reference,
        distributed=distributed,
        quality_gates=quality_gates,
        log_validation=log_validation,
        fatal_error=fatal_error,
    )
    if boundary_root is not None and boundary_root.exists() and boundary_root.parent == run_dir:
        shutil.rmtree(boundary_root)
    write_artifact_manifest(run_dir)
    return Experiment002Run(
        run_directory=run_dir,
        report_path=report,
        summary=summary,
    )
