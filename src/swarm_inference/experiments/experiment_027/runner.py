"""Correctness and lean performance runners for E027."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .protocol import (
    Flags,
    Operation,
    REQUEST,
    FinalStageOutput,
    StageClient,
    StagePipeline,
    StageResponse,
    StageTraversal,
    final_output,
)


MODEL_SHA256 = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    category: str
    content: str


@dataclass(frozen=True, slots=True)
class TargetLane:
    prompt_id: str
    tokens: tuple[int, ...]
    token_sha256: str
    traversals: tuple[StageTraversal, ...]
    elapsed_ns: int


@dataclass(frozen=True, slots=True)
class SpeculativeLane:
    prompt_id: str
    block_size: int
    tokens: tuple[int, ...]
    token_sha256: str
    traversals: tuple[StageTraversal, ...]
    elapsed_ns: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    commit_counts: tuple[int, ...] = ()
    block_elapsed_ns: tuple[int, ...] = ()

    @property
    def committed_per_traversal(self) -> float:
        return len(self.tokens) / len(self.traversals)

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.proposed_draft_tokens


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("E027 configuration must be an object")
    return data


def prompts_from_config(config: dict[str, Any]) -> tuple[Prompt, ...]:
    return tuple(Prompt(**item) for item in config["prompts"])


def token_hash(tokens: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _full_stage_infer(
    client: StageClient,
    tokens: Sequence[int],
    *,
    position: int,
    n_embd: int,
    rewind_position: int = -1,
    top_k: int = 16,
    return_nextn: bool = False,
    return_full_logits: bool = False,
    return_tap22: bool = False,
    return_tap44: bool = False,
) -> StageResponse:
    array = np.ascontiguousarray(tokens, dtype="<i4")
    flags = Flags.INPUT_TOKENS
    if return_nextn:
        flags |= Flags.RETURN_NEXTN
    if return_full_logits:
        flags |= Flags.RETURN_FULL_LOGITS
    if return_tap22:
        flags |= Flags.RETURN_TAP22
    if return_tap44:
        flags |= Flags.RETURN_TAP44
    return client._exchange(  # The reference uses the same audited wire frame.
        Operation.INFER,
        payload=array.tobytes(),
        position=position,
        rewind_position=rewind_position,
        n_tokens=int(array.size),
        n_embd=n_embd,
        flags=flags,
        arg=top_k,
    )


def generate_reference_greedy(
    client: StageClient,
    prompt_tokens: Sequence[int],
    *,
    output_tokens: int,
    n_embd: int,
    top_k: int = 16,
) -> tuple[tuple[int, ...], FinalStageOutput]:
    client.reset()
    prefill_response = _full_stage_infer(
        client,
        prompt_tokens,
        position=0,
        n_embd=n_embd,
        top_k=top_k,
        return_nextn=True,
        return_full_logits=True,
    )
    prefill = final_output(prefill_response)
    pending = int(prefill.greedy_ids[-1])
    generated: list[int] = []
    position = len(prompt_tokens)
    while len(generated) < output_tokens:
        generated.append(pending)
        response = _full_stage_infer(
            client, [pending], position=position, n_embd=n_embd, top_k=top_k
        )
        pending = int(final_output(response).greedy_ids[-1])
        position += 1
    return tuple(generated), prefill


def generate_target_lane(
    pipeline: StagePipeline,
    prompt: Prompt,
    prompt_tokens: Sequence[int],
    *,
    output_tokens: int,
    top_k: int = 16,
    capture_nextn: bool = False,
) -> tuple[TargetLane, FinalStageOutput]:
    pipeline.reset()
    prefill = pipeline.traverse(
        prompt_tokens,
        position=0,
        top_k=top_k,
        return_nextn=capture_nextn,
        return_full_logits=False,
    )
    pending = int(prefill.output.greedy_ids[-1])
    generated: list[int] = []
    traversals: list[StageTraversal] = []
    position = len(prompt_tokens)
    begin = time.perf_counter_ns()
    while len(generated) < output_tokens:
        traversal = pipeline.traverse(
            [pending], position=position, top_k=top_k, return_nextn=capture_nextn
        )
        traversals.append(traversal)
        generated.append(pending)
        pending = int(traversal.output.greedy_ids[-1])
        position += 1
    elapsed = time.perf_counter_ns() - begin
    lane = TargetLane(
        prompt_id=prompt.prompt_id,
        tokens=tuple(generated),
        token_sha256=token_hash(generated),
        traversals=tuple(traversals),
        elapsed_ns=elapsed,
    )
    return lane, prefill.output


def ngram_draft(history_with_pending: Sequence[int], maximum: int) -> list[int]:
    """Deterministic prompt/output n-gram draft; never changes target semantics."""

    if maximum <= 0:
        return []
    history = list(history_with_pending)
    for width in range(min(12, len(history) - 1), 0, -1):
        needle = history[-width:]
        for start in range(len(history) - width - 1, -1, -1):
            if history[start : start + width] != needle:
                continue
            candidate_start = start + width
            available = min(maximum, len(history) - candidate_start)
            if available > 0:
                return history[candidate_start : candidate_start + available]
    return []


def generate_ngram_speculative_lane(
    pipeline: StagePipeline,
    prompt: Prompt,
    prompt_tokens: Sequence[int],
    *,
    output_tokens: int,
    block_size: int,
    top_k: int = 16,
) -> SpeculativeLane:
    if block_size < 2:
        raise ValueError("speculative block size must be at least two")
    pipeline.reset()
    prefill = pipeline.traverse(prompt_tokens, position=0, top_k=top_k)
    pending = int(prefill.output.greedy_ids[-1])
    history = list(prompt_tokens)
    generated: list[int] = []
    traversals: list[StageTraversal] = []
    proposed = 0
    accepted = 0
    commit_counts: list[int] = []
    block_elapsed_ns: list[int] = []
    position = len(prompt_tokens)
    rewind_position = -1
    begin = time.perf_counter_ns()
    while len(generated) < output_tokens:
        block_begin = time.perf_counter_ns()
        draft = ngram_draft(
            history + [pending],
            min(block_size - 1, output_tokens - len(generated) - 1),
        )
        block = [pending, *draft]
        traversal = pipeline.traverse(
            block,
            position=position,
            rewind_position=rewind_position,
            top_k=top_k,
        )
        traversals.append(traversal)
        proposed += len(draft)
        accepted_now = 0
        for index, candidate in enumerate(draft):
            expected = int(traversal.output.greedy_ids[index])
            if candidate != expected:
                break
            accepted_now += 1
        accepted += accepted_now
        committed = [pending, *draft[:accepted_now]]
        generated.extend(committed)
        history.extend(committed)
        pending = int(traversal.output.greedy_ids[accepted_now])
        position += len(committed)
        rewind_position = position if accepted_now < len(draft) else -1
        commit_counts.append(len(committed))
        block_elapsed_ns.append(time.perf_counter_ns() - block_begin)
    elapsed = time.perf_counter_ns() - begin
    generated = generated[:output_tokens]
    return SpeculativeLane(
        prompt_id=prompt.prompt_id,
        block_size=block_size,
        tokens=tuple(generated),
        token_sha256=token_hash(generated),
        traversals=tuple(traversals),
        elapsed_ns=elapsed,
        proposed_draft_tokens=proposed,
        accepted_draft_tokens=accepted,
        commit_counts=tuple(commit_counts),
        block_elapsed_ns=tuple(block_elapsed_ns),
    )


def summarize_target_lanes(lanes: Iterable[TargetLane]) -> dict[str, Any]:
    materialized = tuple(lanes)
    throughputs = [len(lane.tokens) / (lane.elapsed_ns / 1e9) for lane in materialized]
    traversals = [item for lane in materialized for item in lane.traversals]
    tpot_ms = [item.elapsed_ns / 1e6 for item in traversals]
    total_wall_ns = sum(item.elapsed_ns for item in traversals)
    compute_ns = sum(item.compute_ns for item in traversals)
    serialization_ns = sum(item.serialization_ns for item in traversals)
    wan_wait_ns = sum(
        exchange.non_compute_ns
        for item in traversals
        for exchange in item.exchanges[1:]
    )
    wan_bytes = sum(
        exchange.request_bytes + exchange.response_bytes
        for item in traversals
        for exchange in item.exchanges[1:]
    )
    committed = sum(len(lane.tokens) for lane in materialized)
    return {
        "lane_tok_s": throughputs,
        "median_tok_s": statistics.median(throughputs),
        "median_tpot_ms": statistics.median(tpot_ms),
        "p95_tpot_ms": percentile(tpot_ms, 0.95),
        "compute_fraction": compute_ns / total_wall_ns,
        "wan_wait_fraction": min(1.0, wan_wait_ns / total_wall_ns),
        "serialization_fraction": serialization_ns / total_wall_ns,
        "pipeline_idle_fraction": 0.0,
        "wan_messages_per_token": 4 * len(traversals) / committed,
        "wan_bytes_per_token": wan_bytes / committed,
        "stage_compute_ms": [
            statistics.median(
                [item.exchanges[index].compute_ns / 1e6 for item in traversals]
            )
            for index in range(3)
        ],
        "token_sha256": {lane.prompt_id: lane.token_sha256 for lane in materialized},
    }


def summarize_speculative_lanes(lanes: Iterable[SpeculativeLane]) -> dict[str, Any]:
    materialized = tuple(lanes)
    throughputs = [len(lane.tokens) / (lane.elapsed_ns / 1e9) for lane in materialized]
    committed = sum(len(lane.tokens) for lane in materialized)
    traversals = sum(len(lane.traversals) for lane in materialized)
    proposed = sum(lane.proposed_draft_tokens for lane in materialized)
    accepted = sum(lane.accepted_draft_tokens for lane in materialized)
    return {
        "block_size": materialized[0].block_size,
        "lane_tok_s": throughputs,
        "median_tok_s": statistics.median(throughputs),
        "committed_tokens_per_target_traversal": committed / traversals,
        "target_traversals_per_committed_token": traversals / committed,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "token_sha256": {lane.prompt_id: lane.token_sha256 for lane in materialized},
    }


@dataclass(slots=True)
class NativeStageProcess:
    process: subprocess.Popen[bytes]
    log_handle: Any
    log_path: Path
    endpoint: tuple[str, int]

    @classmethod
    def start(
        cls,
        *,
        executable: Path,
        model: Path,
        host: str,
        port: int,
        stage_start: int,
        stage_end: int,
        log_path: Path,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_rs_seq: int = 16,
        n_ubatch: int | None = None,
        mtp: bool = False,
        serial_blocks: bool = False,
    ) -> NativeStageProcess:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("wb")
        environment = os.environ.copy()
        environment["PATH"] = str(executable.parent) + os.pathsep + environment.get("PATH", "")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [
                str(executable),
                "--model", str(model),
                "--host", host,
                "--port", str(port),
                "--stage-start", str(stage_start),
                "--stage-end", str(stage_end),
                "--n-ctx", str(n_ctx),
                "--n-batch", str(n_batch),
                "--n-ubatch", str(n_batch if n_ubatch is None else n_ubatch),
                "--n-rs-seq", str(n_rs_seq),
                "--gpu-layers", "999",
            ] + (["--mtp"] if mtp else []) + (["--serial-blocks"] if serial_blocks else []),
            cwd=executable.parent,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            creationflags=creationflags,
        )
        instance = cls(process, log_handle, log_path, (host, port))
        instance.wait_ready()
        return instance

    def wait_ready(self, timeout_s: float = 180.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_handle.flush()
                detail = self.log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                raise RuntimeError(f"stage process exited {self.process.returncode}:\n{detail}")
            self.log_handle.flush()
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            if "E027_READY" in text:
                return
            time.sleep(0.25)
        raise TimeoutError(f"stage did not become ready: {self.log_path}")

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                with StageClient(*self.endpoint, timeout_s=5) as client:
                    client.shutdown()
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        self.log_handle.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "MODEL_SHA256",
    "NativeStageProcess",
    "Prompt",
    "SpeculativeLane",
    "TargetLane",
    "generate_ngram_speculative_lane",
    "generate_reference_greedy",
    "generate_target_lane",
    "load_config",
    "ngram_draft",
    "prompts_from_config",
    "summarize_speculative_lanes",
    "summarize_target_lanes",
    "token_hash",
    "write_json",
]
