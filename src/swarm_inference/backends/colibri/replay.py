"""Deterministic replay, bounded tuning, and adapter benchmarking for Colibri."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import Field, model_validator

from swarm_inference.backends.colibri.adapters import default_colibri_adapter_registry
from swarm_inference.backends.colibri.model import resolve_model_family
from swarm_inference.backends.colibri.process import ColibriProcess
from swarm_inference.backends.colibri.schemas import ColibriGenerationResult, TuningSample
from swarm_inference.config.models import StrictModel
from swarm_inference.protocol.checksums import sha256_file

_PROMPT_RE = re.compile(r"\[PROMPT_TOKENS\]\s+\d+:\s*([0-9 ]+)")
_TOKENS_RE = re.compile(r"\[TOKENS\]\s+\d+\s+generated:\s*([0-9 ]+)")
_GLM_REPLAY_SPEED_RE = re.compile(
    r"REPLAY decode:\s+\d+\s+tokens.*?\|\s*([0-9.]+)\s+tok/s", re.IGNORECASE
)
_SPEED_RE = re.compile(r"Speed:\s*([0-9.]+)\s*tok/s", re.IGNORECASE)
_HIT_RE = re.compile(r"(?:expert cache hit rate|expert cache hit|expert hit)[: ]+([0-9.]+)%", re.I)
_HIT_COUNTS_RE = re.compile(r"hit=(\d+)\s+miss=(\d+)", re.I)


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


_P95_RE = re.compile(r"(?:latency\s+)?p95\s*[:=]?\s*([0-9.]+)\s*ms", re.I)
_P99_RE = re.compile(r"(?:latency\s+)?p99\s*[:=]?\s*([0-9.]+)\s*ms", re.I)
_C_ENGINE_RE = re.compile(r"(?:GLM C engine|C engine\s*):\s*([0-9 ]+)", re.I)


def _token_hash(prompt_ids: list[int], continuation_ids: list[int]) -> str:
    raw = json.dumps(
        {"prompt_ids": prompt_ids, "continuation_ids": continuation_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class ReplayTokenSequence(StrictModel):
    """Immutable tokens used by every candidate in a fixed-replay comparison."""

    schema_version: Literal["experiment-009-replay-v1"] = "experiment-009-replay-v1"
    model_id: str
    model_revision: str
    tokenizer_hash: str
    prompt_ids: list[int] = Field(min_length=1)
    continuation_ids: list[int] = Field(min_length=1)
    sequence_hash: str = ""
    generated_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def validate_hash(self) -> ReplayTokenSequence:
        expected = _token_hash(self.prompt_ids, self.continuation_ids)
        if self.sequence_hash and self.sequence_hash != expected:
            raise ValueError("fixed-replay token sequence hash does not match its token IDs")
        object.__setattr__(self, "sequence_hash", expected)
        return self

    @property
    def full_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.continuation_ids]

    def engine_reference(self) -> dict[str, list[int]]:
        return {"prompt_ids": self.prompt_ids, "full_ids": self.full_ids}


class ReplayExecution(StrictModel):
    candidate_id: str
    family: str
    command: list[str]
    settings_applied: dict[str, str]
    settings_ignored: list[str] = Field(default_factory=list)
    return_code: int
    timed_out: bool = False
    elapsed_ms: float = Field(ge=0)
    decode_tokens_per_second: float | None = Field(default=None, gt=0)
    time_to_first_token_ms: float | None = Field(default=None, ge=0)
    p95_latency_ms: float | None = Field(default=None, ge=0)
    p99_latency_ms: float | None = Field(default=None, ge=0)
    expert_hit_rate: float | None = Field(default=None, ge=0, le=1)
    expert_cache_hits: int | None = Field(default=None, ge=0)
    expert_cache_misses: int | None = Field(default=None, ge=0)
    storage_read_bytes: int | None = Field(default=None, ge=0)
    storage_read_count: int | None = Field(default=None, ge=0)
    storage_read_duration_ms: float | None = Field(default=None, ge=0)
    cpu_compute_duration_ms: float | None = Field(default=None, ge=0)
    prefetch_useful_bytes: int | None = Field(default=None, ge=0)
    prefetch_wasted_bytes: int | None = Field(default=None, ge=0)
    expert_evictions: int | None = Field(default=None, ge=0)
    bridge_event_count: int = Field(default=0, ge=0)
    input_token_ids: list[int]
    output_token_ids: list[int]
    exact_replay: bool
    stdout: str
    stderr: str


class TuningCandidate(StrictModel):
    candidate_id: str
    settings: dict[str, Any] = Field(default_factory=dict)


class FixedReplayTuningResult(StrictModel):
    schema_version: Literal["experiment-009-fixed-replay-v1"] = "experiment-009-fixed-replay-v1"
    minimum_gain: float = Field(ge=0)
    repeats: int = Field(ge=3)
    baseline_id: str
    selected_candidate_id: str
    accepted: bool
    reverse_confirmed: bool
    confirmed_gain: float = 0.0
    rejection_reason: str | None = None
    replay_sequence_hash: str
    candidates: list[dict[str, Any]]
    reverse_confirmation: dict[str, Any] | None = None


class ColibriReplayRunner:
    """Run exact teacher-forced inputs without changing router or model semantics."""

    def __init__(
        self,
        *,
        engine_directory: str | Path,
        model_path: str | Path,
        model_id: str,
        model_revision: str,
        model_family: str | None = None,
        cap: int = 16,
        quant_bits: int = 8,
        ram_safety_reserve_bytes: int = 8 * 1024**3,
        timeout_seconds: float = 900.0,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.engine_directory = Path(engine_directory).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        self.family = resolve_model_family(config, model_family)
        self.adapter = default_colibri_adapter_registry().get(self.family)
        self.model_id = model_id
        self.model_revision = model_revision
        self.cap = cap
        self.quant_bits = quant_bits
        self.ram_safety_reserve_bytes = ram_safety_reserve_bytes
        self.timeout_seconds = timeout_seconds
        self.environment = dict(environment or {})

    @property
    def engine(self) -> Path:
        basename = self.adapter.engine_basename
        for candidate in (
            self.engine_directory / basename,
            self.engine_directory / f"{basename}.exe",
        ):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"missing Colibri {self.family} engine in {self.engine_directory}")

    def create_calibration(
        self,
        *,
        prompt: str,
        continuation_tokens: int,
        tokenizer_hash: str,
        environment: dict[str, str] | None = None,
    ) -> ReplayTokenSequence:
        """Generate one deterministic continuation and capture its exact IDs."""

        if not self.adapter.supports_text_calibration:
            raise NotImplementedError(
                f"adapter {self.adapter.adapter_id!r} has no text calibration invocation; "
                "supply a ReplayTokenSequence"
            )
        env = self._environment(environment)
        invocation = self.adapter.calibration_invocation(
            engine=self.engine,
            model_path=self.model_path,
            cap=self.cap,
            prompt=prompt,
            continuation_tokens=continuation_tokens,
        )
        env.update(invocation.environment)
        completed = self._run(list(invocation.command), env)
        output = completed.stdout + "\n" + completed.stderr
        if completed.returncode:
            raise RuntimeError(
                f"Colibri calibration failed with exit {completed.returncode}: {output[-4000:]}"
            )
        prompt_match = _PROMPT_RE.search(output)
        token_match = _TOKENS_RE.search(output)
        if prompt_match is None or token_match is None:
            raise RuntimeError(
                "Colibri did not expose calibration token IDs; bridge build required"
            )
        prompt_ids = [int(value) for value in prompt_match.group(1).split()]
        continuation_ids = [int(value) for value in token_match.group(1).split()]
        return ReplayTokenSequence(
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_hash=tokenizer_hash,
            prompt_ids=prompt_ids,
            continuation_ids=continuation_ids,
        )

    def run(
        self,
        replay: ReplayTokenSequence,
        *,
        candidate_id: str,
        settings: dict[str, Any] | None = None,
        supported_settings: set[str] | None = None,
        route_trace_path: str | Path | None = None,
    ) -> ReplayExecution:
        if replay.model_id != self.model_id or replay.model_revision != self.model_revision:
            raise ValueError("replay model identity does not match the Colibri runner")
        invocation_started_ns = time.time_ns()
        bridge_offset = self._bridge_offset()
        supplied = settings or {}
        unsupported = sorted(set(supplied).difference(supported_settings or set(supplied)))
        if unsupported:
            raise ValueError(f"Colibri does not execute requested settings: {unsupported}")
        if psutil.virtual_memory().available < self.ram_safety_reserve_bytes:
            raise MemoryError("available RAM is below the configured Colibri replay safety reserve")
        env = self._environment({key: str(value) for key, value in supplied.items()})
        env["SNAP"] = str(self.model_path)
        env["PROF"] = "1"
        env["COLI_REQUEST_ID"] = candidate_id
        if route_trace_path is not None:
            env["ROUTE_TRACE"] = str(Path(route_trace_path).expanduser().resolve())
        with tempfile.TemporaryDirectory(prefix="swarm-colibri-replay-") as directory:
            reference = Path(directory) / "replay.json"
            reference.write_text(
                json.dumps(replay.engine_reference(), sort_keys=True), encoding="utf-8"
            )
            invocation = self.adapter.replay_invocation(
                engine=self.engine,
                model_path=self.model_path,
                cap=self.cap,
                quant_bits=self.quant_bits,
                reference=reference,
                prompt_ids=tuple(replay.prompt_ids),
                completion_tokens=len(replay.continuation_ids),
                teacher_forced=True,
            )
            env.update(invocation.environment)
            command = list(invocation.command)
            exact_replay = invocation.exact_replay
            started = time.perf_counter()
            try:
                completed = self._run(command, env)
                timed_out = False
                stdout, stderr = completed.stdout, completed.stderr
                return_code = completed.returncode
            except subprocess.TimeoutExpired as error:
                timed_out = True
                stdout = _timeout_text(error.stdout)
                stderr = _timeout_text(error.stderr)
                return_code = -1
            elapsed_ms = (time.perf_counter() - started) * 1000
        combined = stdout + "\n" + stderr
        bridge = self._bridge_metrics(
            offset=bridge_offset,
            request_id=candidate_id,
            invocation_started_ns=invocation_started_ns,
        )
        output_ids = (
            replay.continuation_ids if exact_replay else self._parse_generated_ids(combined)
        )
        speed = self._parse_speed(combined)
        hit = _HIT_RE.search(combined)
        hit_counts = _HIT_COUNTS_RE.search(combined)
        p95 = _P95_RE.search(combined)
        p99 = _P99_RE.search(combined)
        return ReplayExecution(
            candidate_id=candidate_id,
            family=self.family,
            command=command,
            settings_applied={key: str(value) for key, value in supplied.items()},
            return_code=return_code,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
            decode_tokens_per_second=speed,
            time_to_first_token_ms=bridge.get("time_to_first_token_ms"),
            p95_latency_ms=float(p95.group(1)) if p95 else None,
            p99_latency_ms=float(p99.group(1)) if p99 else None,
            expert_hit_rate=float(hit.group(1)) / 100.0 if hit else None,
            expert_cache_hits=int(hit_counts.group(1)) if hit_counts else None,
            expert_cache_misses=int(hit_counts.group(2)) if hit_counts else None,
            storage_read_bytes=bridge.get("storage_read_bytes"),
            storage_read_count=bridge.get("storage_read_count"),
            storage_read_duration_ms=bridge.get("storage_read_duration_ms"),
            cpu_compute_duration_ms=bridge.get("cpu_compute_duration_ms"),
            prefetch_useful_bytes=bridge.get("prefetch_useful_bytes"),
            prefetch_wasted_bytes=bridge.get("prefetch_wasted_bytes"),
            expert_evictions=bridge.get("expert_evictions"),
            bridge_event_count=int(bridge.get("bridge_event_count", 0)),
            input_token_ids=replay.prompt_ids,
            output_token_ids=output_ids,
            exact_replay=exact_replay,
            stdout=stdout,
            stderr=stderr,
        )

    def generate_from_tokens(
        self,
        prompt_ids: list[int],
        *,
        completion_tokens: int,
        candidate_id: str,
        settings: dict[str, Any] | None = None,
        supported_settings: set[str] | None = None,
        route_trace_path: str | Path | None = None,
        invocation_started_ns: int | None = None,
    ) -> ReplayExecution:
        """Run an adapter-declared one-shot generator and return actual token IDs."""

        invocation_started_ns = invocation_started_ns or time.time_ns()
        bridge_offset = self._bridge_offset()
        if self.adapter.launch_mode != "one-shot":
            raise NotImplementedError(
                f"adapter {self.adapter.adapter_id!r} uses its persistent gateway"
            )
        if not prompt_ids or completion_tokens < 1:
            raise ValueError("one-shot generation requires prompt IDs and a positive token count")
        supplied = settings or {}
        unsupported = sorted(set(supplied).difference(supported_settings or set(supplied)))
        if unsupported:
            raise ValueError(f"Colibri does not execute requested settings: {unsupported}")
        if psutil.virtual_memory().available < self.ram_safety_reserve_bytes:
            raise MemoryError("available RAM is below the configured Colibri generation reserve")
        env = self._environment({key: str(value) for key, value in supplied.items()})
        env.update(
            {
                "SNAP": str(self.model_path),
                "PROF": "1",
                "TEMP": "0",
                "NUCLEUS": "1",
                "COLI_REQUEST_ID": candidate_id,
            }
        )
        if route_trace_path is not None:
            env["ROUTE_TRACE"] = str(Path(route_trace_path).expanduser().resolve())
        with tempfile.TemporaryDirectory(prefix="swarm-colibri-generate-") as directory:
            reference = Path(directory) / "generation.json"
            reference.write_text(
                json.dumps(
                    {
                        "prompt_ids": prompt_ids,
                        "full_ids": [*prompt_ids, *([0] * completion_tokens)],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            invocation = self.adapter.replay_invocation(
                engine=self.engine,
                model_path=self.model_path,
                cap=self.cap,
                quant_bits=self.quant_bits,
                reference=reference,
                prompt_ids=tuple(prompt_ids),
                completion_tokens=completion_tokens,
                teacher_forced=False,
            )
            env.update(invocation.environment)
            command = list(invocation.command)
            started = time.perf_counter()
            try:
                completed = self._run(command, env)
                timed_out = False
                stdout, stderr = completed.stdout, completed.stderr
                return_code = completed.returncode
            except subprocess.TimeoutExpired as error:
                timed_out = True
                stdout = _timeout_text(error.stdout)
                stderr = _timeout_text(error.stderr)
                return_code = -1
            elapsed_ms = (time.perf_counter() - started) * 1000
        combined = stdout + "\n" + stderr
        bridge = self._bridge_metrics(
            offset=bridge_offset,
            request_id=candidate_id,
            invocation_started_ns=invocation_started_ns,
        )
        output_ids = self._parse_generated_ids(combined)
        speed = self._parse_speed(combined)
        hit = _HIT_RE.search(combined)
        hit_counts = _HIT_COUNTS_RE.search(combined)
        return ReplayExecution(
            candidate_id=candidate_id,
            family=self.family,
            command=command,
            settings_applied={key: str(value) for key, value in supplied.items()},
            return_code=return_code,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
            decode_tokens_per_second=speed,
            time_to_first_token_ms=bridge.get("time_to_first_token_ms"),
            expert_hit_rate=float(hit.group(1)) / 100.0 if hit else None,
            expert_cache_hits=int(hit_counts.group(1)) if hit_counts else None,
            expert_cache_misses=int(hit_counts.group(2)) if hit_counts else None,
            storage_read_bytes=bridge.get("storage_read_bytes"),
            storage_read_count=bridge.get("storage_read_count"),
            storage_read_duration_ms=bridge.get("storage_read_duration_ms"),
            cpu_compute_duration_ms=bridge.get("cpu_compute_duration_ms"),
            prefetch_useful_bytes=bridge.get("prefetch_useful_bytes"),
            prefetch_wasted_bytes=bridge.get("prefetch_wasted_bytes"),
            expert_evictions=bridge.get("expert_evictions"),
            bridge_event_count=int(bridge.get("bridge_event_count", 0)),
            input_token_ids=prompt_ids,
            output_token_ids=output_ids,
            exact_replay=False,
            stdout=stdout,
            stderr=stderr,
        )

    def _environment(self, overlay: dict[str, str] | None) -> dict[str, str]:
        env = os.environ.copy()
        controlled = {
            "SNAP",
            "COLI_MODEL",
            "COLI_MODEL_ID",
            "COLI_MODEL_REVISION",
            "COLI_SWARM_BRIDGE",
            "COLI_SWARM_BRIDGE_PATH",
            "COLI_SWARM_TELEMETRY",
            "COLI_USAGE_PATH",
            "COLI_HOT_PIN_PATH",
            "PROMPT",
            "TOKENS",
            "REPLAY",
            "REF",
            "REF_FORCE",
            "PPL",
            "ROUTE_TRACE",
            "COLI_REQUEST_ID",
            "OMP_NUM_THREADS",
            "PILOT",
            "HOT",
            "WARMUP",
            "WIDE",
            "SMOOTH",
            "CONF_LIMIT",
            "PILOT_EVICT_GUARD",
            "EXPERT_DROP",
            "AUTOPIN",
            "CAP_RAISE",
            "TEMP",
            "NUCLEUS",
            "TOPK",
            "TOPP",
        }
        for key in controlled:
            env.pop(key, None)
        env.update(self.environment)
        if overlay:
            env.update(overlay)
        return env

    def _bridge_offset(self) -> int:
        value = self.environment.get("COLI_SWARM_BRIDGE_PATH")
        if not value:
            return 0
        path = Path(value)
        return path.stat().st_size if path.is_file() else 0

    def _bridge_metrics(
        self,
        *,
        offset: int,
        request_id: str,
        invocation_started_ns: int,
    ) -> dict[str, Any]:
        value = self.environment.get("COLI_SWARM_BRIDGE_PATH")
        if not value:
            return {"bridge_event_count": 0}
        path = Path(value)
        if not path.is_file():
            return {"bridge_event_count": 0}
        events: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(offset)
            for raw in handle:
                try:
                    event = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if event.get("request_id") == request_id:
                    events.append(event)
        result: dict[str, Any] = {"bridge_event_count": len(events)}
        for event in events:
            raw_payload = event.get("payload")
            payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
            kind = event.get("event_type")
            if kind == "prefill_completed" and "time_to_first_token_ms" not in result:
                timestamp = int(event.get("timestamp_ns", 0))
                if timestamp >= invocation_started_ns:
                    result["time_to_first_token_ms"] = (
                        timestamp - invocation_started_ns
                    ) / 1_000_000
            elif kind == "storage_read":
                result["storage_read_bytes"] = int(payload.get("byte_count", 0))
                result["storage_read_count"] = int(payload.get("read_count", 0))
                result["storage_read_duration_ms"] = int(payload.get("duration_ns", 0)) / 1_000_000
            elif kind == "cpu_compute":
                result["cpu_compute_duration_ms"] = int(payload.get("duration_ns", 0)) / 1_000_000
            elif kind == "expert_prefetch_completed":
                result["prefetch_useful_bytes"] = int(payload.get("useful_bytes", 0))
                result["prefetch_wasted_bytes"] = int(payload.get("wasted_bytes", 0))
            elif kind == "expert_evicted":
                result["expert_evictions"] = int(payload.get("count", 0))
        return result

    def _run(self, command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=self.engine_directory,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )

    @staticmethod
    def _parse_generated_ids(output: str) -> list[int]:
        match = _TOKENS_RE.search(output) or _C_ENGINE_RE.search(output)
        return [] if match is None else [int(value) for value in match.group(1).split()]

    @staticmethod
    def _parse_speed(output: str) -> float | None:
        match = _GLM_REPLAY_SPEED_RE.search(output) or _SPEED_RE.search(output)
        return None if match is None else float(match.group(1))


class ColibriFixedReplayTuner:
    """Conservative bounded search with exact inputs and reverse-order confirmation."""

    def __init__(
        self,
        *,
        repeats: int = 3,
        minimum_gain: float = 0.03,
        maximum_p95_regression: float = 0.05,
    ) -> None:
        if repeats < 3:
            raise ValueError("Experiment 009 requires at least three samples per candidate")
        self.repeats = repeats
        self.minimum_gain = minimum_gain
        self.maximum_p95_regression = maximum_p95_regression

    def tune(
        self,
        *,
        replay: ReplayTokenSequence,
        candidates: Iterable[TuningCandidate],
        measure: Callable[[TuningCandidate, int, Literal["forward", "reverse"]], TuningSample],
        baseline_id: str = "baseline",
    ) -> FixedReplayTuningResult:
        candidate_list = list(candidates)
        if not candidate_list or candidate_list[0].candidate_id != baseline_id:
            raise ValueError("the first fixed-replay candidate must be the baseline")
        if len({candidate.candidate_id for candidate in candidate_list}) != len(candidate_list):
            raise ValueError("candidate IDs must be unique")
        measured = [
            self._measure_candidate(candidate, measure, "forward") for candidate in candidate_list
        ]
        baseline = measured[0]
        eligible = [row for row in measured if self._passes_gates(row, baseline, replay)]
        winner = max(eligible, key=lambda row: row["median_decode_tokens_per_second"])
        lookup = {candidate.candidate_id: candidate for candidate in candidate_list}
        challengers = [row for row in eligible if row["candidate_id"] != baseline_id]
        if not challengers:
            confirmed_baseline = self._measure_candidate(lookup[baseline_id], measure, "reverse")
            return FixedReplayTuningResult(
                minimum_gain=self.minimum_gain,
                repeats=self.repeats,
                baseline_id=baseline_id,
                selected_candidate_id=baseline_id,
                accepted=False,
                reverse_confirmed=self._passes_gates(
                    confirmed_baseline, confirmed_baseline, replay
                ),
                rejection_reason="no_correct_challenger",
                replay_sequence_hash=replay.sequence_hash,
                candidates=measured,
                reverse_confirmation={"baseline": confirmed_baseline},
            )
        challenger = (
            winner
            if winner["candidate_id"] != baseline_id
            else max(challengers, key=lambda row: row["median_decode_tokens_per_second"])
        )
        forward_gain = (
            challenger["median_decode_tokens_per_second"]
            / baseline["median_decode_tokens_per_second"]
            - 1.0
        )
        # Winner first, baseline last: the baseline receives any remaining warm-cache advantage.
        confirmed_winner = self._measure_candidate(
            lookup[challenger["candidate_id"]], measure, "reverse"
        )
        confirmed_baseline = self._measure_candidate(lookup[baseline_id], measure, "reverse")
        confirmed_gain = (
            confirmed_winner["median_decode_tokens_per_second"]
            / confirmed_baseline["median_decode_tokens_per_second"]
            - 1.0
        )
        gates_pass = self._passes_gates(confirmed_winner, confirmed_baseline, replay)
        accepted = (
            gates_pass and forward_gain >= self.minimum_gain and confirmed_gain >= self.minimum_gain
        )
        baseline_confirmed = gates_pass and confirmed_gain < self.minimum_gain
        if accepted:
            reason = None
        elif not gates_pass:
            reason = "reverse_confirmation_correctness_or_tail_gate_failed"
        elif forward_gain < self.minimum_gain:
            reason = "gain_below_minimum_meaningful_threshold"
        else:
            reason = "reverse_confirmed_gain_below_threshold"
        return FixedReplayTuningResult(
            minimum_gain=self.minimum_gain,
            repeats=self.repeats,
            baseline_id=baseline_id,
            selected_candidate_id=challenger["candidate_id"] if accepted else baseline_id,
            accepted=accepted,
            reverse_confirmed=accepted or baseline_confirmed,
            confirmed_gain=confirmed_gain,
            rejection_reason=reason,
            replay_sequence_hash=replay.sequence_hash,
            candidates=measured,
            reverse_confirmation={"winner": confirmed_winner, "baseline": confirmed_baseline},
        )

    def _measure_candidate(
        self,
        candidate: TuningCandidate,
        measure: Callable[[TuningCandidate, int, Literal["forward", "reverse"]], TuningSample],
        order: Literal["forward", "reverse"],
    ) -> dict[str, Any]:
        samples = [measure(candidate, repeat, order) for repeat in range(self.repeats)]
        if any(sample.candidate_id != candidate.candidate_id for sample in samples):
            raise ValueError("measurement returned the wrong candidate ID")
        if any(sample.order != order for sample in samples):
            raise ValueError("measurement returned the wrong execution order")
        reported_p95 = [
            sample.p95_latency_ms for sample in samples if sample.p95_latency_ms is not None
        ]
        latencies = sorted(sample.latency_ms for sample in samples if sample.latency_ms is not None)
        measured_p95 = (
            latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None
        )
        return {
            "candidate_id": candidate.candidate_id,
            "settings": candidate.settings,
            "median_decode_tokens_per_second": statistics.median(
                sample.decode_tokens_per_second for sample in samples
            ),
            "median_p95_latency_ms": (
                statistics.median(reported_p95) if reported_p95 else measured_p95
            ),
            "samples": [sample.model_dump(mode="json") for sample in samples],
        }

    def _passes_gates(
        self,
        row: dict[str, Any],
        baseline: dict[str, Any],
        replay: ReplayTokenSequence,
    ) -> bool:
        for sample in row["samples"]:
            if sample["settings_ignored"]:
                return False
            if sample["input_token_ids"] != replay.prompt_ids:
                return False
            if sample["output_token_ids"] != replay.continuation_ids:
                return False
        baseline_p95 = baseline["median_p95_latency_ms"]
        candidate_p95 = row["median_p95_latency_ms"]
        return not (
            baseline_p95 is not None
            and candidate_p95 is not None
            and candidate_p95 > baseline_p95 * (1.0 + self.maximum_p95_regression)
        )


class ColibriBenchmarkRunner:
    """Measure direct gateway calls against the same process through an adapter call."""

    def compare(
        self,
        *,
        process: ColibriProcess,
        prompt: str,
        max_tokens: int,
        repeats: int,
        adapter_call: Callable[[str], ColibriGenerationResult],
    ) -> list[dict[str, Any]]:
        if repeats < 3:
            raise ValueError("adapter overhead requires at least three measured repeats")
        rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            direct = process.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                request_id=f"direct-{repeat}",
            )
            adapted = adapter_call(f"adapter-{repeat}")
            token_match = (
                direct.token_identity_observed
                and adapted.token_identity_observed
                and direct.input_token_ids == adapted.input_token_ids
                and direct.output_token_ids == adapted.output_token_ids
                and direct.stop_reason == adapted.stop_reason
            )
            rows.extend(
                (
                    self._benchmark_row("direct", repeat, direct, token_match),
                    self._benchmark_row("adapter", repeat, adapted, token_match),
                )
            )
        return rows

    @staticmethod
    def _benchmark_row(
        configuration: str,
        repeat: int,
        result: ColibriGenerationResult,
        token_match: bool,
    ) -> dict[str, Any]:
        return {
            "configuration": configuration,
            "repeat": repeat,
            "request_id": result.request_id,
            "decode_tokens_per_second": result.decode_tokens_per_second,
            "time_to_first_token_ms": result.time_to_first_token_ms,
            "latency_ms": result.elapsed_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "token_identity_observed": result.token_identity_observed,
            "token_identity_match": token_match,
        }


def engine_fingerprint(engine: str | Path) -> dict[str, Any]:
    path = Path(engine).expanduser().resolve()
    return {"path": str(path), "byte_size": path.stat().st_size, "sha256": sha256_file(path)}
