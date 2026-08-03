"""Measured local and native-RPC token-path suites for Experiment 010.

The local patched Colibri binary in ``COLI_SWARM_EXPERT_MODE=local`` is the
only correctness oracle. References contain its exact input and generated token
IDs; the RPC suite replays those files through the same binary and model bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer
import psutil

from swarm_inference.experiments.experiment_010.colibri_expert_bank import (
    verify_bank,
    verify_coordinator_container,
)
from swarm_inference.experiments.experiment_010.colibri_native import (
    COLIBRI_MODEL_REVISION,
    NativeColibriExpertWorker,
    NativeColibriExpertWorkerManager,
    _generated_token_ids,
    _worker_process_accounting,
    whole_expert_ownership_from_banks,
    write_colibri_expert_plan,
)

CORRECTNESS_PROMPTS: tuple[tuple[str, str, str], ...] = (
    ("fact-01", "general_chat", "The capital of France is"),
    ("fact-02", "general_chat", "The largest planet in our solar system is"),
    ("fact-03", "general_chat", "Water freezes at what temperature in Celsius?"),
    ("fact-04", "general_chat", "Name the author of Pride and Prejudice."),
    ("fact-05", "general_chat", "Explain why leaves change colour in autumn."),
    ("fact-06", "general_chat", "What causes ocean tides?"),
    ("fact-07", "general_chat", "Describe the water cycle in four sentences."),
    ("fact-08", "general_chat", "Why does the sky appear blue during the day?"),
    ("fact-09", "general_chat", "Give three practical ways to reduce food waste."),
    ("fact-10", "general_chat", "How can I build a consistent morning routine?"),
    ("code-01", "coding", "Write a Python function that merges two sorted lists."),
    ("code-02", "coding", "Implement binary search in Python and state its complexity."),
    ("code-03", "coding", "Show a Python context manager for timing a code block."),
    ("code-04", "coding", "Write SQL to count orders by customer."),
    ("code-05", "coding", "Explain the difference between a process and a thread."),
    ("code-06", "coding", "Write a Rust function that returns the maximum list value."),
    ("code-07", "coding", "Describe how a hash table resolves collisions."),
    ("code-08", "coding", "Create a JavaScript debounce function."),
    ("code-09", "coding", "Explain why immutable data can simplify concurrent programs."),
    ("code-10", "coding", "Give pseudocode for breadth-first graph traversal."),
    ("math-01", "mathematics_reasoning", "A train travels 180 km in 2.5 hours. Find its average speed."),
    ("math-02", "mathematics_reasoning", "If x squared minus 5x plus 6 equals zero, solve for x."),
    ("math-03", "mathematics_reasoning", "What is 17 percent of 240? Show the calculation."),
    ("math-04", "mathematics_reasoning", "A rectangle is 8 by 13. Find its area and perimeter."),
    ("math-05", "mathematics_reasoning", "Simplify the fraction 84 over 126."),
    ("math-06", "mathematics_reasoning", "A fair coin is tossed three times. What is the probability of two heads?"),
    ("math-07", "mathematics_reasoning", "Find the next two terms: 2, 6, 12, 20, 30."),
    ("math-08", "mathematics_reasoning", "Convert 3.75 hours into hours and minutes."),
    ("math-09", "mathematics_reasoning", "Explain the Pythagorean theorem with a numerical example."),
    ("math-10", "mathematics_reasoning", "A price rises from 80 to 92. What is the percentage increase?"),
    ("multi-01", "multilingual", "Resume en francais les causes de la Revolution industrielle."),
    ("multi-02", "multilingual", "Escribe un ensayo breve sobre la conservacion del agua."),
    ("multi-03", "multilingual", "Erklaere auf Deutsch, warum regelmaessiger Schlaf wichtig ist."),
    ("multi-04", "multilingual", "Scrivi in italiano tre consigli per imparare una lingua."),
    ("multi-05", "multilingual", "Explique em portugues como funciona a fotossintese."),
    ("multi-06", "multilingual", "Geef in het Nederlands een korte uitleg van windenergie."),
    ("multi-07", "multilingual", "Beskriv pa svenska varfor kallkritik ar viktigt."),
    ("multi-08", "multilingual", "Napisz po polsku krotkie podsumowanie obiegu wody."),
    ("multi-09", "multilingual", "Describe en espanol como preparar una reunion eficaz."),
    ("multi-10", "multilingual", "Redige en francais une liste de trois habitudes durables."),
    ("reason-01", "reasoning", "Compare solar and wind power using cost, reliability, and land use."),
    ("reason-02", "reasoning", "A library is crowded after school. Suggest a fair seating policy."),
    ("reason-03", "reasoning", "Explain two benefits and two risks of remote work."),
    ("reason-04", "reasoning", "Design a simple experiment to test whether light affects plant growth."),
    ("reason-05", "reasoning", "Summarize the strongest argument for preserving urban trees."),
    ("reason-06", "reasoning", "List the steps for checking whether a news claim is reliable."),
    ("reason-07", "reasoning", "Propose a weekly study plan for three subjects and explain the tradeoffs."),
    ("reason-08", "reasoning", "Why can an average hide important differences in a dataset?"),
    ("reason-09", "reasoning", "Explain when a queue is a better data structure than a stack."),
    ("reason-10", "reasoning", "Give a concise argument for and against congestion pricing."),
)

_NUMERIC_TRACE_HEADER = struct.Struct("<8sIiiiiQ")
_NUMERIC_TRACE_MAGIC = b"COLNUM1\x00"
_NUMERIC_TRACE_KINDS = {
    1: "post_moe_hidden_state",
    2: "pre_sampling_logits",
    3: "router_weights_exact_fp32",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _numeric_trace_records(path: Path):
    """Yield validated native numeric-trace records without retaining the file."""

    with path.open("rb") as handle:
        record_index = 0
        while True:
            header = handle.read(_NUMERIC_TRACE_HEADER.size)
            if not header:
                return
            if len(header) != _NUMERIC_TRACE_HEADER.size:
                raise ValueError(f"truncated numeric trace header in {path}")
            magic, kind, token_position, layer_id, rows, width, payload_bytes = (
                _NUMERIC_TRACE_HEADER.unpack(header)
            )
            if magic != _NUMERIC_TRACE_MAGIC:
                raise ValueError(f"invalid numeric trace magic in {path}")
            if kind not in _NUMERIC_TRACE_KINDS:
                raise ValueError(f"unsupported numeric trace kind {kind} in {path}")
            expected_bytes = rows * width * 4
            if rows <= 0 or width <= 0 or payload_bytes != expected_bytes:
                raise ValueError(f"invalid numeric trace geometry in {path}")
            payload = handle.read(payload_bytes)
            if len(payload) != payload_bytes:
                raise ValueError(f"truncated numeric trace payload in {path}")
            yield {
                "record_index": record_index,
                "kind": kind,
                "token_position": token_position,
                "layer_id": layer_id,
                "batch_rows": rows,
                "width": width,
                "payload": payload,
            }
            record_index += 1


def compare_colibri_numeric_traces(
    *,
    prompt_id: str,
    local_trace: Path,
    distributed_trace: Path,
    expected_token_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare paired in-engine hidden, logit, and exact router-weight records."""

    rows: list[dict[str, Any]] = []
    kind_counts = {name: 0 for name in _NUMERIC_TRACE_KINDS.values()}
    exact_kind_counts = {name: 0 for name in _NUMERIC_TRACE_KINDS.values()}
    logit_index = 0
    first_divergent_layer: int | None = None
    first_divergent_token: int | None = None
    local_records = _numeric_trace_records(local_trace)
    distributed_records = _numeric_trace_records(distributed_trace)
    for local, remote in itertools.zip_longest(local_records, distributed_records):
        if local is None or remote is None:
            raise ValueError("local and distributed numeric traces have different record counts")
        metadata = ("kind", "token_position", "layer_id", "batch_rows", "width")
        if any(local[name] != remote[name] for name in metadata):
            raise ValueError(
                "local and distributed numeric trace record metadata diverged at "
                f"record {local['record_index']}"
            )
        kind_name = _NUMERIC_TRACE_KINDS[int(local["kind"])]
        kind_counts[kind_name] += 1
        exact = local["payload"] == remote["payload"]
        if exact:
            exact_kind_counts[kind_name] += 1
        local_values = np.frombuffer(local["payload"], dtype="<f4")
        remote_values = np.frombuffer(remote["payload"], dtype="<f4")
        difference = remote_values.astype(np.float64) - local_values.astype(np.float64)
        maximum_absolute_error = float(np.max(np.abs(difference), initial=0.0))
        local_norm = float(np.linalg.norm(local_values.astype(np.float64)))
        relative_l2_error = float(np.linalg.norm(difference) / max(local_norm, 1e-30))
        sampled_token_id: int | None = None
        local_argmax: int | None = None
        distributed_argmax: int | None = None
        sampled_logit_error: float | None = None
        logit_margin: float | None = None
        if local["kind"] == 2:
            sampled_token_id = (
                int(expected_token_ids[logit_index])
                if logit_index < len(expected_token_ids)
                else None
            )
            local_argmax = int(np.argmax(local_values))
            distributed_argmax = int(np.argmax(remote_values))
            if sampled_token_id is not None:
                sampled_logit_error = float(
                    abs(remote_values[sampled_token_id] - local_values[sampled_token_id])
                )
            top_two = np.partition(local_values, -2)[-2:]
            logit_margin = float(top_two.max() - top_two.min())
            logit_index += 1
        if not exact and first_divergent_token is None:
            first_divergent_token = int(local["token_position"])
            first_divergent_layer = (
                int(local["layer_id"]) if int(local["layer_id"]) >= 0 else None
            )
        rows.append(
            {
                "schema_version": "experiment-010-colibri-numeric-boundary-error-v1",
                "evidence_category": "REAL_MODEL_MEASURED",
                "prompt_id": prompt_id,
                "record_index": int(local["record_index"]),
                "record_kind": kind_name,
                "token_position": int(local["token_position"]),
                "layer_id": int(local["layer_id"]),
                "batch_rows": int(local["batch_rows"]),
                "width": int(local["width"]),
                "value_count": int(local_values.size),
                "exact_fp32_identity": exact,
                "maximum_absolute_error": maximum_absolute_error,
                "relative_l2_error": relative_l2_error,
                "sampled_token_id": sampled_token_id,
                "local_argmax": local_argmax,
                "distributed_argmax": distributed_argmax,
                "sampled_logit_error": sampled_logit_error,
                "logit_margin": logit_margin,
                "local_trace_path": str(local_trace),
                "distributed_trace_path": str(distributed_trace),
            }
        )
    summary = {
        "record_count": len(rows),
        "record_counts": kind_counts,
        "exact_record_counts": exact_kind_counts,
        "all_records_exact_fp32": all(row["exact_fp32_identity"] for row in rows),
        "router_weights_exact": (
            kind_counts["router_weights_exact_fp32"] > 0
            and kind_counts["router_weights_exact_fp32"]
            == exact_kind_counts["router_weights_exact_fp32"]
        ),
        "first_divergent_token": first_divergent_token,
        "first_divergent_layer": first_divergent_layer,
        "maximum_hidden_state_absolute_error": max(
            (
                row["maximum_absolute_error"]
                for row in rows
                if row["record_kind"] == "post_moe_hidden_state"
            ),
            default=None,
        ),
        "maximum_logit_absolute_error": max(
            (
                row["maximum_absolute_error"]
                for row in rows
                if row["record_kind"] == "pre_sampling_logits"
            ),
            default=None,
        ),
        "logit_record_count": logit_index,
        "local_trace_sha256": _sha256_file(local_trace),
        "distributed_trace_sha256": _sha256_file(distributed_trace),
    }
    return rows, summary


def _base_environment(
    model_path: Path, *, threads: int, idot: bool = True
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "COLI_SWARM_EXPERT_PLAN",
        "COLI_SWARM_EXPERT_TELEMETRY",
        "COLI_SWARM_BRIDGE_PATH",
        "COLI_HOT_PIN_PATH",
        "COLI_USAGE_PATH",
        "ROUTE_TRACE",
        "COLI_SWARM_NUMERIC_TRACE",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "SNAP": str(model_path),
            "PILOT": "0",
            "HOT": "0",
            "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "FALSE",
            "IDOT": "1" if idot else "0",
        }
    )
    return environment


def _run_engine(
    engine: Path,
    reference: Path,
    environment: dict[str, str],
    *,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], int]:
    started = time.perf_counter_ns()
    completed = subprocess.run(
        [str(engine), "16", "8", str(reference)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return completed, time.perf_counter_ns() - started


def _sample_process(
    process: psutil.Process,
    *,
    prompt_id: str,
    process_role: str,
    worker_id: str,
    expert_bank_bytes: int,
    owned_expert_count: int,
) -> dict[str, Any] | None:
    try:
        full = process.memory_full_info()
        basic = process.memory_info()
        io = process.io_counters()
        return {
            "timestamp_ns": time.time_ns(),
            "prompt_id": prompt_id,
            "process_role": process_role,
            "worker_id": worker_id,
            "pid": process.pid,
            "expert_bank_bytes": expert_bank_bytes,
            "owned_expert_count": owned_expert_count,
            "working_set_bytes": int(full.rss),
            "private_bytes": int(getattr(full, "private", getattr(full, "uss", 0))),
            "commit_size_bytes": int(getattr(full, "pagefile", full.vms)),
            "peak_working_set_bytes": int(getattr(full, "peak_wset", full.rss)),
            "page_fault_count": int(getattr(basic, "num_page_faults", 0)),
            "thread_count": process.num_threads(),
            "cpu_affinity": ";".join(
                str(value)
                for value in (
                    process.cpu_affinity() if hasattr(process, "cpu_affinity") else []
                )
            ),
            "storage_read_bytes": int(io.read_bytes),
            "storage_write_bytes": int(io.write_bytes),
            "system_available_physical_bytes": int(psutil.virtual_memory().available),
            "system_pagefile_used_bytes": int(psutil.swap_memory().used),
        }
    except (psutil.Error, OSError):
        return None


def _run_engine_sampled(
    engine: Path,
    reference: Path,
    environment: dict[str, str],
    *,
    timeout_seconds: float,
    prompt_id: str,
    workers: list[NativeColibriExpertWorker],
) -> tuple[subprocess.CompletedProcess[str], int, list[dict[str, Any]]]:
    """Run one coordinator while sampling every participating process."""

    command = [str(engine), "16", "8", str(reference)]
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    coordinator = psutil.Process(process.pid)
    worker_details: list[tuple[NativeColibriExpertWorker, int, int]] = []
    for worker in workers:
        manifest = json.loads((worker.bank_path / "manifest.json").read_text(encoding="utf-8"))
        worker_details.append(
            (
                worker,
                int(manifest["total_expert_bytes"]),
                int(manifest.get("owned_expert_count", len(manifest.get("owned_experts", [])))),
            )
        )
    samples: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        coordinator_sample = _sample_process(
            coordinator,
            prompt_id=prompt_id,
            process_role="coordinator",
            worker_id="coordinator",
            expert_bank_bytes=0,
            owned_expert_count=0,
        )
        if coordinator_sample:
            samples.append(coordinator_sample)
        for worker, bank_bytes, owned_count in worker_details:
            sample = _sample_process(
                psutil.Process(worker.process.pid),
                prompt_id=prompt_id,
                process_role="expert_worker",
                worker_id=worker.worker_id,
                expert_bank_bytes=bank_bytes,
                owned_expert_count=owned_count,
            )
            if sample:
                samples.append(sample)
        if time.monotonic() >= deadline:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout_seconds, stdout, stderr)
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return completed, time.perf_counter_ns() - started, samples


def _route_ownership_counts(
    route_trace: Path,
    local_experts: set[tuple[int, int]],
) -> tuple[int, int]:
    """Count local and remote selected ranks in Colibri's canonical route trace."""

    local_count = 0
    remote_count = 0
    for line in route_trace.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"malformed Colibri route trace row: {line!r}")
        layer_id = int(fields[2])
        for selected in fields[3:]:
            expert_id = int(selected.split(":", 1)[0])
            if (layer_id, expert_id) in local_experts:
                local_count += 1
            else:
                remote_count += 1
    return local_count, remote_count


def audit_capacity_ownership(
    *,
    coordinator_model: Path,
    bank_paths: list[Path],
) -> dict[str, Any]:
    """Reconcile physical coordinator/worker ownership for Gate 8."""

    coordinator = verify_coordinator_container(coordinator_model)
    config = json.loads((coordinator_model / "config.json").read_text(encoding="utf-8"))
    expected = {
        (layer_id, expert_id)
        for layer_id in range(int(config["num_hidden_layers"]))
        for expert_id in range(int(config["num_experts"]))
    }
    ownership_counts: dict[tuple[int, int], int] = {}
    worker_rows: list[dict[str, Any]] = []
    source_fingerprints: set[str] = set()
    total_bytes = 0
    for path in bank_paths:
        bank = path.expanduser().resolve()
        verified = verify_bank(bank)
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        ownership = json.loads((bank / "ownership.json").read_text(encoding="utf-8"))
        if verified["bank_kind"] != "native_colibri_whole_experts":
            raise ValueError("capacity isolation requires whole-expert worker banks")
        owned = {
            (int(item["layer_id"]), int(item["expert_id"]))
            for item in ownership["owned_experts"]
        }
        if len(owned) != len(ownership["owned_experts"]):
            raise ValueError(f"duplicate expert inside worker bank {verified['worker_id']}")
        for identity in owned:
            ownership_counts[identity] = ownership_counts.get(identity, 0) + 1
        worker_bytes = int(manifest["total_expert_bytes"])
        total_bytes += worker_bytes
        source_fingerprints.add(str(manifest["source_model_fingerprint"]))
        worker_rows.append(
            {
                "worker_id": verified["worker_id"],
                "expert_bank_bytes": worker_bytes,
                "owned_expert_count": len(owned),
            }
        )
    if source_fingerprints != {coordinator["source_model_fingerprint"]}:
        raise ValueError("coordinator and worker banks do not share one source fingerprint")
    observed = set(ownership_counts)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    duplicates = sorted(identity for identity, count in ownership_counts.items() if count != 1)
    for row in worker_rows:
        row["ownership_fraction"] = row["expert_bank_bytes"] / total_bytes if total_bytes else 1.0
        row["under_30_percent"] = row["ownership_fraction"] <= 0.30
    result = {
        "schema_version": "experiment-010-capacity-ownership-audit-v1",
        "source_model_fingerprint": coordinator["source_model_fingerprint"],
        "coordinator": coordinator,
        "expected_expert_count": len(expected),
        "globally_owned_expert_count": len(observed),
        "global_expert_bank_bytes": total_bytes,
        "missing_experts": [list(item) for item in missing],
        "unexpected_experts": [list(item) for item in unexpected],
        "duplicate_ownership": [list(item) for item in duplicates],
        "workers": worker_rows,
    }
    result["valid"] = (
        coordinator["coordinator_owned_routed_expert_count"] == 0
        and coordinator["coordinator_owned_routed_expert_bytes"] == 0
        and not missing
        and not unexpected
        and not duplicates
        and len(worker_rows) >= 4
        and all(row["under_30_percent"] for row in worker_rows)
    )
    return result


def build_local_reference_suite(
    *,
    engine: Path,
    model_path: Path,
    output_directory: Path,
    generated_tokens: int = 32,
    thread_count: int = 2,
    timeout_seconds: float = 300.0,
    prompt_limit: int | None = None,
    required_prompt_count: int = 50,
    idot: bool = True,
) -> dict[str, Any]:
    """Capture exact local Colibri token references for a declared corpus slice."""

    if generated_tokens < 32:
        raise ValueError("official correctness references require at least 32 tokens")
    if required_prompt_count < 1 or required_prompt_count > len(CORRECTNESS_PROMPTS):
        raise ValueError("required prompt count is outside the correctness corpus")
    selected_prompts = CORRECTNESS_PROMPTS[:prompt_limit]
    if len(selected_prompts) < required_prompt_count:
        raise ValueError("prompt limit is smaller than the required prompt count")
    engine = engine.expanduser().resolve()
    model = model_path.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = model / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    results: list[dict[str, Any]] = []
    for prompt_id, workload_group, text in selected_prompts:
        prompt_root = output / prompt_id
        prompt_root.mkdir(parents=True, exist_ok=True)
        reference_path = prompt_root / "reference.json"
        baseline_result_path = prompt_root / "local-result.json"
        if reference_path.is_file() and baseline_result_path.is_file():
            recorded = json.loads(baseline_result_path.read_text(encoding="utf-8"))
            if recorded.get("exact_token_identity") and recorded.get("expected_tokens") == generated_tokens:
                results.append(recorded)
                continue
        prompt_ids = tokenizer.encode(text).ids
        placeholder_path = prompt_root / "capture-reference.json"
        _write_json(
            placeholder_path,
            {"prompt_id": prompt_id, "prompt": text, "prompt_ids": prompt_ids,
             "full_ids": [*prompt_ids, *([0] * generated_tokens)]},
        )
        capture_route = prompt_root / "capture-route.trace"
        capture_numeric_trace = prompt_root / "capture-numeric.trace"
        capture_environment = _base_environment(model, threads=thread_count, idot=idot)
        capture_environment.update(
            {
                "COLI_SWARM_EXPERT_MODE": "local",
                "ROUTE_TRACE": str(capture_route),
                "COLI_SWARM_NUMERIC_TRACE": str(capture_numeric_trace),
                "COLI_USAGE_PATH": str(prompt_root / "capture.coli_usage"),
                "COLI_SWARM_BRIDGE": "1",
                "COLI_SWARM_TELEMETRY": "trace",
                "COLI_SWARM_BRIDGE_PATH": str(prompt_root / "local-bridge.ndjson"),
                "COLI_REQUEST_ID": prompt_id,
                "COLI_MODEL_REVISION": COLIBRI_MODEL_REVISION,
            }
        )
        capture, capture_ns = _run_engine(
            engine, placeholder_path, capture_environment, timeout_seconds=timeout_seconds
        )
        (prompt_root / "capture-stdout.log").write_text(capture.stdout, encoding="utf-8")
        (prompt_root / "capture-stderr.log").write_text(capture.stderr, encoding="utf-8")
        token_ids = _generated_token_ids(capture.stdout)
        if capture.returncode != 0 or len(token_ids) != generated_tokens:
            raise RuntimeError(f"local Colibri oracle capture failed for {prompt_id}")
        _write_json(
            reference_path,
            {"prompt_id": prompt_id, "workload_group": workload_group, "prompt": text,
             "prompt_ids": prompt_ids, "full_ids": [*prompt_ids, *token_ids]},
        )
        (prompt_root / "local-stdout.log").write_text(capture.stdout, encoding="utf-8")
        (prompt_root / "local-stderr.log").write_text(capture.stderr, encoding="utf-8")
        result = {
            "prompt_id": prompt_id,
            "workload_group": workload_group,
            "prompt_token_count": len(prompt_ids),
            "expected_tokens": generated_tokens,
            "actual_token_ids": token_ids,
            "expected_token_ids": token_ids,
            "matching_tokens": generated_tokens,
            "exact_token_identity": capture.returncode == 0 and len(token_ids) == generated_tokens,
            "return_code": capture.returncode,
            "capture_elapsed_ns": capture_ns,
            "baseline_elapsed_ns": capture_ns,
            "route_trace_sha256": _sha256_file(capture_route),
            "reference_path": str(reference_path),
            "route_trace_path": str(capture_route),
            "numeric_trace_sha256": _sha256_file(capture_numeric_trace),
            "numeric_trace_path": str(capture_numeric_trace),
            "evidence_category": "REAL_MODEL_MEASURED",
        }
        _write_json(baseline_result_path, result)
        if not result["exact_token_identity"]:
            raise RuntimeError(f"local Colibri oracle replay diverged for {prompt_id}")
        results.append(result)
    summary = {
        "schema_version": "experiment-010-colibri-local-reference-suite-v1",
        "model_path": str(model),
        "model_revision": COLIBRI_MODEL_REVISION,
        "engine_path": str(engine),
        "engine_sha256": _sha256_file(engine),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "idot": idot,
        "prompt_count": len(results),
        "required_prompt_count": required_prompt_count,
        "generated_tokens_per_prompt": generated_tokens,
        "exact_prompt_count": sum(row["exact_token_identity"] for row in results),
        "complete": len(results) == required_prompt_count
        and all(row["exact_token_identity"] for row in results),
        "results": results,
    }
    _write_json(output / "local-suite.json", summary)
    return summary


def _start_workers(
    manager: NativeColibriExpertWorkerManager,
    bank_paths: list[Path],
    *,
    model_fingerprint: str,
    quantization_fingerprint: str,
    memory_budget_bytes: int,
    thread_count: int,
) -> list[NativeColibriExpertWorker]:
    workers: list[NativeColibriExpertWorker] = []
    for bank_path in bank_paths:
        bank = bank_path.expanduser().resolve()
        manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
        workers.append(
            manager.start(
                worker_id=str(manifest["worker_id"]),
                bank_path=bank,
                model_id="colibri-olmoe",
                model_revision=COLIBRI_MODEL_REVISION,
                quantization_fingerprint=quantization_fingerprint,
                model_fingerprint=model_fingerprint,
                memory_budget_bytes=memory_budget_bytes,
                thread_count=thread_count,
            )
        )
    return workers


def run_rpc_correctness_suite(
    *,
    worker_executable: Path,
    engine: Path,
    model_path: Path,
    reference_directory: Path,
    bank_paths: list[Path],
    output_directory: Path,
    model_fingerprint: str,
    response_mode: str = "per_expert_exact",
    quantization_fingerprint: str = "native-colibri-int8-v1",
    coordinator_thread_count: int = 2,
    worker_thread_count: int = 1,
    memory_budget_bytes: int = 256 * 1024 * 1024,
    timeout_seconds: float = 300.0,
    prompt_limit: int | None = None,
    required_prompt_count: int = 50,
    capacity_isolation: bool = False,
    expert_mode: str = "rpc",
    local_experts: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Run all local references through persistent native expert workers."""

    if response_mode not in {"per_expert_exact", "per_worker_fast"}:
        raise ValueError("unsupported RPC response mode")
    if expert_mode not in {"rpc", "hybrid", "planner"}:
        raise ValueError("unsupported external expert mode")
    local_ownership = list(local_experts or [])
    if expert_mode == "rpc" and local_ownership:
        raise ValueError("rpc mode cannot declare local expert ownership")
    if expert_mode == "hybrid" and not local_ownership:
        raise ValueError("hybrid mode requires explicit local expert ownership")
    if required_prompt_count < 1 or required_prompt_count > len(CORRECTNESS_PROMPTS):
        raise ValueError("required prompt count is outside the correctness corpus")
    selected_prompts = CORRECTNESS_PROMPTS[:prompt_limit]
    if len(selected_prompts) < required_prompt_count:
        raise ValueError("prompt limit is smaller than the required prompt count")
    engine = engine.expanduser().resolve()
    model = model_path.expanduser().resolve()
    references = reference_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_manifests = [
        json.loads((path.expanduser().resolve() / "manifest.json").read_text(encoding="utf-8"))
        for path in bank_paths
    ]
    bank_kinds = {str(manifest.get("bank_kind", "")) for manifest in bank_manifests}
    if bank_kinds == {"native_colibri_microshards"}:
        execution_strategy = "native_microshard"
    elif bank_kinds == {"native_colibri_whole_experts"}:
        execution_strategy = "whole_expert"
    else:
        raise ValueError(f"RPC correctness suite cannot mix bank kinds: {sorted(bank_kinds)}")
    configuration = f"{execution_strategy}_{expert_mode}_{response_mode}"
    capacity_audit: dict[str, Any] | None = None
    if capacity_isolation:
        if execution_strategy != "whole_expert" or response_mode != "per_expert_exact":
            raise ValueError("capacity isolation requires exact whole-expert RPC")
        capacity_audit = audit_capacity_ownership(
            coordinator_model=model,
            bank_paths=bank_paths,
        )
        if not capacity_audit["valid"]:
            raise ValueError("capacity ownership audit did not pass")
        configuration = "capacity_isolated_whole_expert_rpc_per_expert_exact"
    manager = NativeColibriExpertWorkerManager(output / "workers", worker_executable)
    rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    memory_samples: list[dict[str, Any]] = []
    try:
        workers = _start_workers(
            manager,
            bank_paths,
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
            memory_budget_bytes=memory_budget_bytes,
            thread_count=worker_thread_count,
        )
        plan = write_colibri_expert_plan(
            output / "plan.json",
            model_fingerprint=model_fingerprint,
            quantization_fingerprint=quantization_fingerprint,
            phase="decode",
            workers=workers,
            local_experts=local_ownership,
        )
        local_identity_set = {
            (int(row["layer_id"]), int(row["expert_id"]))
            for row in local_ownership
        }
        for prompt_id, workload_group, _ in selected_prompts:
            reference_path = references / prompt_id / "reference.json"
            local_result = json.loads(
                (references / prompt_id / "local-result.json").read_text(encoding="utf-8")
            )
            prompt_output = output / prompt_id
            prompt_output.mkdir(parents=True, exist_ok=True)
            telemetry = prompt_output / "coordinator-telemetry.jsonl"
            route_trace = prompt_output / "route.trace"
            numeric_trace = prompt_output / "numeric.trace"
            bridge = prompt_output / "bridge.ndjson"
            for path in (telemetry, route_trace, numeric_trace, bridge):
                if path.is_file():
                    path.unlink()
            environment = _base_environment(model, threads=coordinator_thread_count)
            environment.update(
                {
                    "COLI_SWARM_EXPERT_MODE": expert_mode,
                    "COLI_SWARM_EXPERT_PLAN": str(plan),
                    "COLI_SWARM_EXPERT_TIMEOUT_MS": str(round(timeout_seconds * 1000)),
                    "COLI_SWARM_EXPERT_FALLBACK": "fail",
                    "COLI_SWARM_EXPERT_RESPONSE_MODE": response_mode,
                    "COLI_SWARM_EXPERT_DETERMINISM": (
                        "exact" if response_mode == "per_expert_exact" else "quality_bounded"
                    ),
                    "COLI_SWARM_EXPERT_TELEMETRY": str(telemetry),
                    "ROUTE_TRACE": str(route_trace),
                    "COLI_SWARM_NUMERIC_TRACE": str(numeric_trace),
                    "COLI_USAGE_PATH": str(prompt_output / "distributed.coli_usage"),
                    "COLI_SWARM_BRIDGE": "1",
                    "COLI_SWARM_TELEMETRY": "trace",
                    "COLI_SWARM_BRIDGE_PATH": str(bridge),
                    "COLI_REQUEST_ID": prompt_id,
                    "COLI_MODEL_REVISION": COLIBRI_MODEL_REVISION,
                }
            )
            if capacity_isolation:
                completed, elapsed_ns, prompt_samples = _run_engine_sampled(
                    engine,
                    reference_path,
                    environment,
                    timeout_seconds=timeout_seconds,
                    prompt_id=prompt_id,
                    workers=workers,
                )
                memory_samples.extend(prompt_samples)
            else:
                completed, elapsed_ns = _run_engine(
                    engine, reference_path, environment, timeout_seconds=timeout_seconds
                )
            (prompt_output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (prompt_output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            actual_tokens = _generated_token_ids(completed.stdout)
            events = (
                [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines() if line]
                if telemetry.is_file()
                else []
            )
            expected_tokens = local_result["expected_token_ids"]
            local_route = Path(local_result["route_trace_path"])
            local_numeric_trace_value = local_result.get("numeric_trace_path")
            if not local_numeric_trace_value:
                raise ValueError(
                    f"local reference {prompt_id} predates required numeric tracing"
                )
            local_numeric_trace = Path(local_numeric_trace_value)
            prompt_boundary_rows, numeric_comparison = compare_colibri_numeric_traces(
                prompt_id=prompt_id,
                local_trace=local_numeric_trace,
                distributed_trace=numeric_trace,
                expected_token_ids=expected_tokens,
            )
            boundary_rows.extend(prompt_boundary_rows)
            local_selected_rank_count, remote_selected_rank_count = (
                _route_ownership_counts(route_trace, local_identity_set)
                if route_trace.is_file()
                else (0, 0)
            )
            row = {
                "prompt_id": prompt_id,
                "workload_group": workload_group,
                "configuration": configuration,
                "response_mode": response_mode,
                "prompt_token_count": local_result["prompt_token_count"],
                "generated_token_count": len(actual_tokens),
                "matching_token_count": sum(
                    a == b for a, b in zip(actual_tokens, expected_tokens, strict=False)
                ),
                "exact_token_identity": completed.returncode == 0 and actual_tokens == expected_tokens,
                "router_trace_identity": route_trace.is_file() and route_trace.read_bytes() == local_route.read_bytes(),
                "router_weight_identity": numeric_comparison["router_weights_exact"],
                "local_route_sha256": local_result["route_trace_sha256"],
                "distributed_route_sha256": _sha256_file(route_trace) if route_trace.is_file() else None,
                "return_code": completed.returncode,
                "elapsed_ns": elapsed_ns,
                "remote_rpc_request_count": sum(
                    event.get("event") == "expert_rpc_request_completed" for event in events
                ),
                "remote_result_consumed_count": sum(
                    event.get("event") == "expert_rpc_request_completed"
                    and event.get("remote_result_consumed") is True
                    for event in events
                ),
                "forbidden_local_expert_load_count": sum(
                    event.get("event") == "forbidden_local_expert_load" for event in events
                ),
                "silent_local_retry_count": sum(
                    event.get("event") == "expert_rpc_fallback" for event in events
                ),
                "local_selected_rank_count": local_selected_rank_count,
                "remote_selected_rank_count": remote_selected_rank_count,
                "first_divergent_token": next(
                    (index for index, (actual, expected) in enumerate(zip(actual_tokens, expected_tokens, strict=False))
                     if actual != expected),
                    None,
                ),
                "first_divergent_layer": numeric_comparison["first_divergent_layer"],
                "numeric_trace_all_records_exact_fp32": numeric_comparison[
                    "all_records_exact_fp32"
                ],
                "numeric_trace_record_count": numeric_comparison["record_count"],
                "hidden_boundary_record_count": numeric_comparison["record_counts"][
                    "post_moe_hidden_state"
                ],
                "logit_record_count": numeric_comparison["logit_record_count"],
                "maximum_hidden_state_absolute_error": numeric_comparison[
                    "maximum_hidden_state_absolute_error"
                ],
                "maximum_logit_absolute_error": numeric_comparison[
                    "maximum_logit_absolute_error"
                ],
                "evidence_category": "REAL_MODEL_MEASURED",
                "telemetry_path": str(telemetry),
                "route_trace_path": str(route_trace),
                "numeric_trace_path": str(numeric_trace),
                "local_numeric_trace_sha256": numeric_comparison["local_trace_sha256"],
                "distributed_numeric_trace_sha256": numeric_comparison[
                    "distributed_trace_sha256"
                ],
            }
            residency_events = [
                event
                for event in events
                if event.get("event") == "coordinator_memory_residency"
            ]
            row.update(
                {
                    "coordinator_owned_routed_expert_bytes": max(
                        (
                            int(event.get("coordinator_owned_routed_expert_bytes", -1))
                            for event in residency_events
                        ),
                        default=-1,
                    ),
                    "coordinator_owned_routed_expert_count": max(
                        (
                            int(event.get("coordinator_owned_routed_expert_count", -1))
                            for event in residency_events
                        ),
                        default=-1,
                    ),
                    "local_expert_runtime_disabled": bool(residency_events)
                    and all(
                        event.get("local_expert_runtime_enabled") is False
                        for event in residency_events
                    ),
                    "capacity_isolation_message_observed": (
                        "capacity-isolated coordinator: local routed-expert runtime disabled"
                        in completed.stderr
                    ),
                }
            )
            _write_json(prompt_output / "result.json", {**row, "actual_token_ids": actual_tokens,
                                                         "expected_token_ids": expected_tokens})
            rows.append(row)
        accounting = [_worker_process_accounting(worker) for worker in workers]
    finally:
        manager.close()
    memory_timeseries_path: str | None = None
    coordinator_accounting: dict[str, Any] | None = None
    if memory_samples:
        timeseries = output / "memory_residency_timeseries.csv"
        with timeseries.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(memory_samples[0]))
            writer.writeheader()
            writer.writerows(memory_samples)
        memory_timeseries_path = str(timeseries)
        coordinator_rows = [
            sample for sample in memory_samples if sample["process_role"] == "coordinator"
        ]
        coordinator_accounting = {
            "process_role": "coordinator",
            "owned_expert_count": 0,
            "expert_bank_bytes": 0,
            "sample_count": len(coordinator_rows),
            "peak_working_set_bytes": max(
                (int(sample["peak_working_set_bytes"]) for sample in coordinator_rows),
                default=0,
            ),
            "peak_private_bytes": max(
                (int(sample["private_bytes"]) for sample in coordinator_rows), default=0
            ),
            "peak_commit_size_bytes": max(
                (int(sample["commit_size_bytes"]) for sample in coordinator_rows),
                default=0,
            ),
            "max_page_fault_count": max(
                (int(sample["page_fault_count"]) for sample in coordinator_rows), default=0
            ),
            "max_thread_count": max(
                (int(sample["thread_count"]) for sample in coordinator_rows), default=0
            ),
            "max_storage_read_bytes": max(
                (int(sample["storage_read_bytes"]) for sample in coordinator_rows),
                default=0,
            ),
        }
    fieldnames = list(rows[0]) if rows else []
    with (output / "colibri_rpc_token_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    boundary_fields = list(boundary_rows[0]) if boundary_rows else []
    with (output / "colibri_rpc_boundary_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=boundary_fields)
        writer.writeheader()
        writer.writerows(boundary_rows)
    summary = {
        "schema_version": "experiment-010-colibri-rpc-correctness-suite-v1",
        "configuration": configuration,
        "expert_mode": expert_mode,
        "execution_strategy": execution_strategy,
        "response_mode": response_mode,
        "model_path": str(model),
        "model_fingerprint": model_fingerprint,
        "model_revision": COLIBRI_MODEL_REVISION,
        "engine_path": str(engine),
        "engine_sha256": _sha256_file(engine),
        "worker_executable": str(worker_executable.expanduser().resolve()),
        "worker_executable_sha256": _sha256_file(worker_executable.expanduser().resolve()),
        "prompt_count": len(rows),
        "generated_tokens_per_prompt": min(
            (row["generated_token_count"] for row in rows), default=0
        ),
        "required_prompt_count": required_prompt_count,
        "exact_prompt_count": sum(row["exact_token_identity"] for row in rows),
        "router_identity_prompt_count": sum(row["router_trace_identity"] for row in rows),
        "router_weight_identity_prompt_count": sum(
            row["router_weight_identity"] for row in rows
        ),
        "numeric_exact_prompt_count": sum(
            row["numeric_trace_all_records_exact_fp32"] for row in rows
        ),
        "hidden_boundary_record_count": sum(
            row["hidden_boundary_record_count"] for row in rows
        ),
        "logit_record_count": sum(row["logit_record_count"] for row in rows),
        "forbidden_local_expert_load_count": sum(
            row["forbidden_local_expert_load_count"] for row in rows
        ),
        "silent_local_retry_count": sum(row["silent_local_retry_count"] for row in rows),
        "remote_rpc_request_count": sum(row["remote_rpc_request_count"] for row in rows),
        "remote_result_consumed_count": sum(row["remote_result_consumed_count"] for row in rows),
        "local_expert_count": len(local_ownership),
        "local_selected_rank_count": sum(row["local_selected_rank_count"] for row in rows),
        "remote_selected_rank_count": sum(row["remote_selected_rank_count"] for row in rows),
        "worker_process_accounting": accounting,
        "coordinator_process_accounting": coordinator_accounting,
        "memory_residency_timeseries": memory_timeseries_path,
        "capacity_isolation": capacity_audit,
        "worker_lifecycle": manager.lifecycle_records,
        "complete": (
            len(rows) == required_prompt_count
            and all(row["generated_token_count"] >= 32 for row in rows)
            and all(
                row["exact_token_identity"]
                and row["router_trace_identity"]
                and row["router_weight_identity"]
                and row["numeric_trace_all_records_exact_fp32"]
                and row["logit_record_count"] == row["generated_token_count"]
                for row in rows
            )
            and not any(row["forbidden_local_expert_load_count"] for row in rows)
            and not any(row["silent_local_retry_count"] for row in rows)
            and (
                expert_mode != "hybrid"
                or (
                    bool(local_ownership)
                    and all(row["local_selected_rank_count"] > 0 for row in rows)
                    and all(row["remote_selected_rank_count"] > 0 for row in rows)
                    and all(row["remote_result_consumed_count"] > 0 for row in rows)
                )
            )
            and (
                not capacity_isolation
                or (
                    capacity_audit is not None
                    and capacity_audit["valid"]
                    and len(rows) >= 10
                    and all(row["generated_token_count"] >= 128 for row in rows)
                    and all(row["coordinator_owned_routed_expert_bytes"] == 0 for row in rows)
                    and all(row["coordinator_owned_routed_expert_count"] == 0 for row in rows)
                    and all(row["local_expert_runtime_disabled"] for row in rows)
                    and all(row["capacity_isolation_message_observed"] for row in rows)
                )
            )
        ),
        "results": rows,
    }
    _write_json(output / "suite-result.json", summary)
    if capacity_isolation:
        _write_json(
            output / "capacity_accounting.json",
            {
                "schema_version": "experiment-010-capacity-accounting-v1",
                "ownership": capacity_audit,
                "coordinator_process_accounting": coordinator_accounting,
                "worker_process_accounting": accounting,
                "memory_residency_timeseries": memory_timeseries_path,
                "prompt_count": len(rows),
                "generated_tokens_per_prompt": summary["generated_tokens_per_prompt"],
                "exact_prompt_count": summary["exact_prompt_count"],
                "complete": summary["complete"],
            },
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-local-references", action="store_true")
    parser.add_argument("--run-rpc-suite", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--bank", type=Path, action="append", default=[])
    parser.add_argument("--local-bank", type=Path, action="append", default=[])
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--model-fingerprint")
    parser.add_argument(
        "--expert-mode",
        choices=("rpc", "hybrid", "planner"),
        default="rpc",
    )
    parser.add_argument(
        "--response-mode",
        choices=("per_expert_exact", "per_worker_fast"),
        default="per_expert_exact",
    )
    parser.add_argument("--generated-tokens", type=int, default=32)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--required-prompts", type=int, default=50)
    parser.add_argument("--capacity-isolation", action="store_true")
    arguments = parser.parse_args()
    if arguments.build_local_references == arguments.run_rpc_suite:
        parser.error("choose exactly one suite action")
    if arguments.build_local_references:
        result = build_local_reference_suite(
            engine=arguments.engine,
            model_path=arguments.model,
            output_directory=arguments.output,
            generated_tokens=arguments.generated_tokens,
            prompt_limit=arguments.prompt_limit,
            required_prompt_count=arguments.required_prompts,
        )
    else:
        if not arguments.worker or not arguments.bank or not arguments.reference_directory or not arguments.model_fingerprint:
            parser.error("the RPC suite requires worker, banks, references, and model fingerprint")
        if arguments.expert_mode == "hybrid" and not arguments.local_bank:
            parser.error("hybrid correctness requires one or more --local-bank values")
        if arguments.expert_mode == "rpc" and arguments.local_bank:
            parser.error("--local-bank is only valid for hybrid or planner correctness")
        result = run_rpc_correctness_suite(
            worker_executable=arguments.worker,
            engine=arguments.engine,
            model_path=arguments.model,
            reference_directory=arguments.reference_directory,
            bank_paths=arguments.bank,
            output_directory=arguments.output,
            model_fingerprint=arguments.model_fingerprint,
            response_mode=arguments.response_mode,
            prompt_limit=arguments.prompt_limit,
            required_prompt_count=arguments.required_prompts,
            capacity_isolation=arguments.capacity_isolation,
            expert_mode=arguments.expert_mode,
            local_experts=(
                whole_expert_ownership_from_banks(arguments.local_bank)
                if arguments.local_bank
                else []
            ),
        )
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
