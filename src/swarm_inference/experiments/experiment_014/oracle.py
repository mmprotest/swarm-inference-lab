"""Runner and validator for the real-weight full-model serial Kimi K3 oracle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import time
from array import array
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA_VERSION = "experiment-014-k3-serial-oracle-v1"
_PROMPT = re.compile(r"\[K3\] prompt: (\d+) tokens")
_DECODE = re.compile(r"\[K3\] decode (\d+) tokens")
_TOKEN = re.compile(r"\[tok (\d+):(?: id (\d+),)?")
_STATE = re.compile(r"^(\d+) (\d+) (kda|mla) (\d+) ([0-9a-f]{16}) (\d+)$")


class SerialOracleError(RuntimeError):
    """The serial oracle could not be run or its evidence was invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(checkpoint: Path) -> dict[str, Any]:
    with (checkpoint / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise SerialOracleError("checkpoint has no text_config")
    return text


def _read_float_row(handle: BinaryIO, elements: int) -> tuple[str, bool, float, float]:
    raw = handle.read(elements * 4)
    if len(raw) != elements * 4:
        raise SerialOracleError("trace ended inside a hidden-state row")
    values = array("f")
    values.frombytes(raw)
    if os.sys.byteorder != "little":
        values.byteswap()
    finite = all(math.isfinite(value) for value in values)
    l2 = math.sqrt(sum(float(value) * float(value) for value in values))
    maximum = max((abs(float(value)) for value in values), default=0.0)
    return hashlib.sha256(raw).hexdigest(), finite, l2, maximum


def inspect_trace(
    trace_path: Path,
    *,
    hidden_size: int,
    layer_count: int,
) -> dict[str, Any]:
    """Validate the C engine trace and fingerprint every layer/forward step."""

    row_bytes = hidden_size * 4
    rows_per_step = layer_count + 1  # transformer outputs + final normalized hidden
    size = trace_path.stat().st_size
    if size == 0 or size % row_bytes:
        raise SerialOracleError(f"trace has {size} bytes, not a positive multiple of {row_bytes}")
    rows = size // row_bytes
    if rows % rows_per_step:
        raise SerialOracleError(
            f"trace has {rows} rows, not complete {rows_per_step}-row forward steps"
        )
    steps = rows // rows_per_step
    per_layer: list[dict[str, Any]] = [
        {"layer": layer, "status": "PASS", "step_fingerprints": []} for layer in range(layer_count)
    ]
    final_rows: list[dict[str, Any]] = []
    all_finite = True
    with trace_path.open("rb") as handle:
        for step in range(steps):
            for layer in range(layer_count):
                digest, finite, l2, maximum = _read_float_row(handle, hidden_size)
                all_finite &= finite
                if not finite:
                    per_layer[layer]["status"] = "FAIL"
                per_layer[layer]["step_fingerprints"].append(
                    {"step": step, "sha256": digest, "finite": finite, "l2": l2, "max_abs": maximum}
                )
            digest, finite, l2, maximum = _read_float_row(handle, hidden_size)
            all_finite &= finite
            final_rows.append(
                {"step": step, "sha256": digest, "finite": finite, "l2": l2, "max_abs": maximum}
            )
    return {
        "trace_bytes": size,
        "trace_sha256": _sha256(trace_path),
        "forward_steps": steps,
        "rows": rows,
        "all_finite": all_finite,
        "layers": per_layer,
        "final_hidden": final_rows,
    }


def inspect_route_trace(
    route_path: Path,
    *,
    layer_count: int,
    first_dense_layer_count: int,
    expert_count: int,
    top_k: int,
) -> dict[str, Any]:
    calls: dict[int, list[tuple[int, int]]] = {}
    selections = 0
    malformed: list[str] = []
    with route_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 3 + top_k:
                malformed.append(f"line {number}: expected {3 + top_k} fields")
                continue
            try:
                call, row, layer = map(int, fields[:3])
                pairs = [
                    (int(item.split(":", 1)[0]), float(item.split(":", 1)[1]))
                    for item in fields[3:]
                ]
            except (ValueError, IndexError):
                malformed.append(f"line {number}: invalid route record")
                continue
            if row < 0 or layer < first_dense_layer_count or layer >= layer_count:
                malformed.append(f"line {number}: invalid row/layer")
            if any(
                expert < 0 or expert >= expert_count or not math.isfinite(gate)
                for expert, gate in pairs
            ):
                malformed.append(f"line {number}: invalid expert/gate")
            if len({expert for expert, _ in pairs}) != top_k:
                malformed.append(f"line {number}: duplicate expert")
            calls.setdefault(call, []).append((row, layer))
            selections += len(pairs)
    expected_layers = set(range(first_dense_layer_count, layer_count))
    complete_calls = 0
    layer_calls: Counter[int] = Counter()
    for records in calls.values():
        unique_layers = {layer for _, layer in records}
        rows = sorted(row for row, _ in records)
        if len(unique_layers) == 1 and rows == list(range(len(records))):
            complete_calls += 1
            layer_calls[records[0][1]] += 1
    no_moe_expected = not expected_layers
    covered_layers = set(layer_calls)
    balanced = len(set(layer_calls.values())) <= 1
    status = (
        "PASS"
        if not malformed
        and (
            (no_moe_expected and not calls)
            or (
                calls
                and complete_calls == len(calls)
                and covered_layers == expected_layers
                and balanced
            )
        )
        else "FAIL"
    )
    return {
        "route_sha256": _sha256(route_path),
        "route_calls": len(calls),
        "complete_route_calls": complete_calls,
        "selection_count": selections,
        "layer_call_counts": dict(sorted(layer_calls.items())),
        "malformed": malformed,
        "status": status,
    }


def inspect_state_trace(
    state_path: Path,
    *,
    layer_count: int,
    kda_layers: set[int],
) -> dict[str, Any]:
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    malformed: list[str] = []
    with state_path.open("r", encoding="ascii") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            match = _STATE.fullmatch(line)
            if match is None:
                malformed.append(f"line {number}: malformed state fingerprint")
                continue
            step, layer, kind, byte_count, digest, finite_count = match.groups()
            step_i, layer_i = int(step), int(layer)
            expected_kind = "kda" if layer_i in kda_layers else "mla"
            if layer_i < 0 or layer_i >= layer_count or kind != expected_kind:
                malformed.append(f"line {number}: inconsistent layer/type")
            by_step.setdefault(step_i, {})[layer_i] = {
                "kind": kind,
                "bytes": int(byte_count),
                "fnv1a64": digest,
                "finite_values": int(finite_count),
            }
            if int(finite_count) * 4 != int(byte_count):
                malformed.append(f"line {number}: state contains non-finite values")
    complete = all(len(rows) == layer_count for rows in by_step.values()) and bool(by_step)
    transitions = 0
    ordered = sorted(by_step)
    for left, right in pairwise(ordered):
        if any(
            by_step[left][layer]["fnv1a64"] != by_step[right][layer]["fnv1a64"]
            for layer in range(layer_count)
        ):
            transitions += 1
    return {
        "state_sha256": _sha256(state_path),
        "fingerprinted_steps": len(by_step),
        "complete_steps": complete,
        "changed_step_transitions": transitions,
        "malformed": malformed,
        "status": "PASS" if complete and not malformed and transitions >= 1 else "FAIL",
    }


def run_serial_oracle(
    checkpoint: Path,
    executable: Path,
    output_directory: Path,
    *,
    prompt: str = "Hi",
    generated_tokens: int = 2,
    layer_limit: int | None = None,
    timeout_seconds: float | None = None,
    k3_idot: int | None = None,
) -> dict[str, Any]:
    """Run the real engine, retain raw evidence, and emit a fail-closed receipt."""

    root = checkpoint.expanduser().resolve()
    binary = executable.expanduser().resolve()
    if not root.is_dir() or not binary.is_file():
        raise SerialOracleError("checkpoint or serial engine executable is missing")
    text = _load_config(root)
    configured_layers = int(text["num_hidden_layers"])
    layers = min(layer_limit or configured_layers, configured_layers)
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "hidden-trace.f32"
    logits_path = output / "prefill-logits.f32"
    routes_path = output / "routes.txt"
    states_path = output / "states.txt"
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    for path in (trace_path, logits_path, routes_path, states_path, stdout_path, stderr_path):
        path.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "COLI_TEMP": "0",
            "K3_CHUNK": "1",
            "K3_TRACE": str(trace_path),
            "K3_LOGITS": str(logits_path),
            "K3_STATE_TRACE": str(states_path),
            "ROUTE_TRACE": str(routes_path),
        }
    )
    if layer_limit is not None:
        environment["K3_LAYERS"] = str(layer_limit)
    if k3_idot is not None:
        if k3_idot not in (0, 1):
            raise SerialOracleError("K3_IDOT must be 0 or 1")
        environment["K3_IDOT"] = str(k3_idot)
    command = [str(binary), str(root), prompt, "--ngen", str(generated_tokens)]
    started = datetime.now(UTC)
    monotonic = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        try:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                check=False,
                text=True,
            )
            return_code: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            return_code = None
            timed_out = True
    duration = time.perf_counter() - monotonic
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    prompt_match = _PROMPT.search(stderr_text)
    decode_match = _DECODE.search(stderr_text)
    prompt_tokens = int(prompt_match.group(1)) if prompt_match else None
    decoded_tokens = int(decode_match.group(1)) if decode_match else None
    token_ids = [int(match.group(2)) for match in _TOKEN.finditer(stderr_text) if match.group(2)]

    validation_errors: list[str] = []
    trace: dict[str, Any] | None = None
    routes: dict[str, Any] | None = None
    states: dict[str, Any] | None = None
    if trace_path.exists():
        try:
            trace = inspect_trace(
                trace_path, hidden_size=int(text["hidden_size"]), layer_count=layers
            )
        except SerialOracleError as error:
            validation_errors.append(str(error))
    else:
        validation_errors.append("hidden trace was not created")
    if routes_path.exists():
        routes = inspect_route_trace(
            routes_path,
            layer_count=layers,
            first_dense_layer_count=int(text["first_k_dense_replace"]),
            expert_count=int(text["num_experts"]),
            top_k=int(text["num_experts_per_token"]),
        )
        if routes["status"] != "PASS":
            validation_errors.append("routing trace failed validation")
    else:
        validation_errors.append("routing trace was not created")
    kda = {int(value) - 1 for value in text["linear_attn_config"]["kda_layers"]}
    if states_path.exists():
        states = inspect_state_trace(states_path, layer_count=layers, kda_layers=kda)
        if states["status"] != "PASS":
            validation_errors.append("state trace failed validation")
    else:
        validation_errors.append("state trace was not created")
    logits_bytes = logits_path.stat().st_size if logits_path.exists() else 0
    expected_logits_multiple = int(text["vocab_size"]) * 4
    logits_structural = logits_bytes > 0 and logits_bytes % expected_logits_multiple == 0
    if not logits_structural:
        validation_errors.append("prefill logits are missing or structurally invalid")
    if return_code != 0:
        validation_errors.append(f"engine return code was {return_code}")
    if timed_out:
        validation_errors.append("engine timed out")
    full_graph = layers == configured_layers
    stateful = bool(trace and prompt_tokens is not None and trace["forward_steps"] > prompt_tokens)
    if not stateful:
        validation_errors.append("no post-prefill stateful decode forward step was proven")
    if token_ids and any(token < 0 or token >= int(text["vocab_size"]) for token in token_ids):
        validation_errors.append("generated token ID outside vocabulary")
    status = "PASS" if full_graph and not validation_errors else "FAIL"
    component_status = {
        "embedding": "PASS" if prompt_tokens and trace else "FAIL",
        "final_norm": "PASS" if trace and trace.get("all_finite") else "FAIL",
        "lm_head": "PASS" if logits_structural else "FAIL",
        "tokenizer": "PASS" if "tokenizer.json loaded" in stderr_text and prompt_tokens else "FAIL",
        "chat_template": "NOT_TESTED",
        "sampler": "PASS" if decoded_tokens and decoded_tokens > 0 else "FAIL",
        "eos": "PASS" if decoded_tokens is not None else "FAIL",
        "stateful_step": "PASS" if stateful else "FAIL",
        "kda_state": "PASS" if states and states["status"] == "PASS" else "FAIL",
        "mla_cache": "PASS" if states and states["status"] == "PASS" else "FAIL",
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "started_at_utc": started.isoformat(),
        "status": status,
        "full_graph": full_graph and status == "PASS",
        "stateful_decode": stateful,
        "command": command,
        "environment_overrides": {
            key: environment[key]
            for key in sorted(environment)
            if key.startswith("K3_") or key == "ROUTE_TRACE" or key == "COLI_TEMP"
        },
        "checkpoint": str(root),
        "executable": str(binary),
        "executable_sha256": _sha256(binary),
        "configured_layers": configured_layers,
        "executed_layers": layers,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "requested_generated_tokens": generated_tokens,
        "decoded_tokens": decoded_tokens,
        "generated_token_ids": token_ids,
        "duration_seconds": duration,
        "return_code": return_code,
        "timed_out": timed_out,
        "validation_errors": validation_errors,
        "component_status": component_status,
        "trace": trace,
        "routes": routes,
        "states": states,
        "logits": {
            "bytes": logits_bytes,
            "sha256": _sha256(logits_path) if logits_path.exists() else None,
            "structurally_valid": logits_structural,
        },
        "layers": trace["layers"] if trace else [],
        "raw_artifacts": {
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "hidden_trace": str(trace_path),
            "prefill_logits": str(logits_path),
            "routes": str(routes_path),
            "states": str(states_path),
        },
        "precision_disclosure": {
            "reference_activations": "FP32",
            "routed_expert_weights": "native checkpoint MXFP4 E2M1/UE8M0",
            "routed_expert_activation_path": (
                "FP32" if k3_idot == 0 else "INT8_APPROXIMATION"
                if k3_idot == 1
                else "ENGINE_DEFAULT_INT8_APPROXIMATION"
            ),
            "non_expert_weights": "load-time Colibri quantization controlled by K3_BITS/K3_MLA_BITS/K3_HEAD_BITS",
            "production_mxfp8_activation_gate": "NOT_CERTIFIED_BY_THIS_ORACLE",
        },
    }
    receipt_path = output / "serial-oracle-receipt.json"
    temporary = receipt_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt


__all__ = [
    "SerialOracleError",
    "inspect_route_trace",
    "inspect_state_trace",
    "inspect_trace",
    "run_serial_oracle",
]
