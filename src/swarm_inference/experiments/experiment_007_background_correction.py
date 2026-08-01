"""Fixed-duration, token-timestamped background capacity benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from swarm_inference.backends.http import post_json
from swarm_inference.backends.sglang import _sglang_output_ids
from swarm_inference.experiments.services import HostTelemetry

Lane = Literal["gpu", "cpu"]
TrafficMode = Literal["closed_loop", "open_loop"]
ServingArm = Literal["gpu_only", "gpu_plus_cpu", "cpu_only"]


@dataclass(frozen=True, slots=True)
class MeasurementWindow:
    warmup_start: float
    measurement_start: float
    measurement_end: float
    drain_deadline: float

    @property
    def duration_seconds(self) -> float:
        return self.measurement_end - self.measurement_start

    def classify(self, timestamp: float) -> str:
        if timestamp < self.measurement_start:
            return "before_window"
        if timestamp < self.measurement_end:
            return "inside_window"
        return "after_window"


@dataclass(frozen=True, slots=True)
class WorkloadFixture:
    fixture_id: str
    prompt_token_ids: tuple[int, ...]
    requested_output_tokens: int
    prompt_hash: str


def workload_fixture_hash(fixtures: list[WorkloadFixture]) -> str:
    payload = [
        {
            "fixture_id": item.fixture_id,
            "prompt_token_ids": list(item.prompt_token_ids),
            "requested_output_tokens": item.requested_output_tokens,
            "prompt_hash": item.prompt_hash,
        }
        for item in fixtures
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(slots=True)
class TokenCompletion:
    token_id: int
    completion_monotonic: float
    sequence_index: int


@dataclass(slots=True)
class RequestObservation:
    lane: Lane
    request_id: str
    fixture_id: str
    prompt_hash: str
    prompt_token_ids: list[int]
    requested_output_tokens: int
    admitted_monotonic: float
    started_monotonic: float
    completed_monotonic: float
    queue_delay_ms: float
    success: bool
    error: str | None
    token_events: list[TokenCompletion]
    artifact_hash: str
    artifact_revision: str

    @property
    def token_ids(self) -> list[int]:
        return [item.token_id for item in self.token_events]


@dataclass(slots=True)
class LaneState:
    active_requests: int = 0
    maximum_active_requests: int = 0
    queued_requests: int = 0
    maximum_queued_requests: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def queued(self, delta: int) -> None:
        with self.lock:
            self.queued_requests += delta
            self.maximum_queued_requests = max(self.maximum_queued_requests, self.queued_requests)

    def active(self, delta: int) -> None:
        with self.lock:
            self.active_requests += delta
            self.maximum_active_requests = max(self.maximum_active_requests, self.active_requests)


class TokenEventWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write_request(
        self,
        observation: RequestObservation,
        window: MeasurementWindow,
        *,
        run_point_id: str,
        arm: ServingArm,
        repeat: int,
        traffic_mode: TrafficMode,
    ) -> None:
        with self._lock:
            for event in observation.token_events:
                payload = {
                    "classification": {
                        "gpu_only": "measured_cuda",
                        "gpu_plus_cpu": "measured_mixed_backend",
                        "cpu_only": "measured_x86_cpu",
                    }[arm],
                    "run_point_id": run_point_id,
                    "arm": arm,
                    "repeat": repeat,
                    "traffic_mode": traffic_mode,
                    "lane": observation.lane,
                    "request_id": observation.request_id,
                    "fixture_id": observation.fixture_id,
                    "token_id": event.token_id,
                    "token_sequence_index": event.sequence_index,
                    "token_completion_monotonic": event.completion_monotonic,
                    "window_bucket": window.classify(event.completion_monotonic),
                    "request_completed_successfully": observation.success,
                    "artifact_hash": observation.artifact_hash,
                    "artifact_revision": observation.artifact_revision,
                    "verified": observation.success,
                }
                self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def parse_sse_json_lines(lines: list[bytes]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for raw in lines:
        text = raw.decode("utf-8", errors="strict").strip()
        if not text or text.startswith(":"):
            continue
        if text.startswith("data:"):
            text = text[5:].strip()
        if text == "[DONE]":
            break
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise RuntimeError("streaming backend emitted a non-object JSON event")
        parsed.append(payload)
    return parsed


def _iter_sse_json(
    endpoint: str, path: str, payload: dict[str, Any], timeout_seconds: float
) -> Any:
    request = Request(
        endpoint.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw in response:
                text = raw.decode("utf-8", errors="strict").strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                event = json.loads(text)
                if not isinstance(event, dict):
                    raise RuntimeError("streaming backend emitted non-object JSON")
                yield event
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from streaming backend: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"streaming backend unavailable: {exc}") from exc


def _candidate_token_ids(chunk: dict[str, Any], lane: Lane) -> tuple[list[int], bool]:
    candidates: Any = None
    cumulative = False
    for key in ("output_ids", "token_ids", "tokens"):
        value = chunk.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            candidates = value
            # SGLang streams the cumulative output sequence. llama.cpp's
            # legacy /completion endpoint streams the token(s) produced by
            # the current event, and an event can contain more than one token.
            cumulative = lane == "gpu"
            break
    if candidates is None:
        meta = chunk.get("meta_info")
        if isinstance(meta, dict):
            for key in ("output_ids", "completion_token_ids"):
                value = meta.get(key)
                if isinstance(value, list) and all(isinstance(item, int) for item in value):
                    candidates = value
                    cumulative = True
                    break
    if candidates is None:
        probabilities = chunk.get("completion_probabilities")
        if isinstance(probabilities, list):
            ids = [item.get("id") for item in probabilities if isinstance(item, dict)]
            if ids and all(isinstance(item, int) for item in ids):
                candidates = ids
                cumulative = lane == "gpu"
    if candidates is None:
        token = chunk.get("token")
        if isinstance(token, int):
            candidates = [token]
    if candidates is None:
        raise RuntimeError(f"{lane} streaming event did not expose token IDs")
    return [int(item) for item in candidates], cumulative


def _new_tokens(previous: list[int], candidates: list[int], *, cumulative: bool) -> list[int]:
    if cumulative:
        if len(candidates) < len(previous) or candidates[: len(previous)] != previous:
            raise RuntimeError("cumulative streaming token sequence changed retrospectively")
        return candidates[len(previous) :]
    return candidates


def _stream_request(
    *,
    lane: Lane,
    endpoint: str,
    fixture: WorkloadFixture,
    artifact_hash: str,
    artifact_revision: str,
    admitted_monotonic: float,
    timeout_seconds: float,
    workload_seed: int,
) -> RequestObservation:
    request_id = f"exp007-correction-{lane}-{uuid4().hex}"
    started = time.monotonic()
    if lane == "gpu":
        path = "/generate"
        payload = {
            "input_ids": list(fixture.prompt_token_ids),
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": fixture.requested_output_tokens,
                "ignore_eos": True,
            },
            "stream": True,
            "rid": request_id,
        }
    else:
        path = "/completion"
        payload = {
            "prompt": list(fixture.prompt_token_ids),
            "n_predict": fixture.requested_output_tokens,
            "temperature": 0.0,
            "seed": workload_seed,
            "ignore_eos": True,
            "cache_prompt": True,
            "n_probs": 1,
            "return_tokens": True,
            "stream": True,
            "id_slot": -1,
        }
    token_ids: list[int] = []
    events: list[TokenCompletion] = []
    error: str | None = None
    success = False
    try:
        for chunk in _iter_sse_json(endpoint, path, payload, timeout_seconds):
            candidates, cumulative = _candidate_token_ids(chunk, lane)
            additions = _new_tokens(token_ids, candidates, cumulative=cumulative)
            now = time.monotonic()
            for token_id in additions:
                events.append(
                    TokenCompletion(
                        token_id=token_id,
                        completion_monotonic=now,
                        sequence_index=len(token_ids),
                    )
                )
                token_ids.append(token_id)
            if bool(chunk.get("stop")) or bool(chunk.get("finished")):
                success = True
        success = success or len(token_ids) == fixture.requested_output_tokens
        if len(token_ids) > fixture.requested_output_tokens:
            raise RuntimeError("streaming backend emitted more tokens than requested")
        if not success:
            error = "stream ended without successful request completion evidence"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = time.monotonic()
    return RequestObservation(
        lane=lane,
        request_id=request_id,
        fixture_id=fixture.fixture_id,
        prompt_hash=fixture.prompt_hash,
        prompt_token_ids=list(fixture.prompt_token_ids),
        requested_output_tokens=fixture.requested_output_tokens,
        admitted_monotonic=admitted_monotonic,
        started_monotonic=started,
        completed_monotonic=completed,
        queue_delay_ms=(started - admitted_monotonic) * 1000,
        success=success,
        error=error,
        token_events=events,
        artifact_hash=artifact_hash,
        artifact_revision=artifact_revision,
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def count_window_tokens(
    requests: list[RequestObservation], window: MeasurementWindow
) -> dict[str, int]:
    counts = {"tokens_before_window": 0, "tokens_inside_window": 0, "tokens_after_window": 0}
    for request in requests:
        if not request.success:
            continue
        for token in request.token_events:
            key = {
                "before_window": "tokens_before_window",
                "inside_window": "tokens_inside_window",
                "after_window": "tokens_after_window",
            }[window.classify(token.completion_monotonic)]
            counts[key] += 1
    return counts


def lane_metrics(
    lane: Lane,
    requests: list[RequestObservation],
    window: MeasurementWindow,
    state: LaneState,
) -> dict[str, Any]:
    lane_requests = [item for item in requests if item.lane == lane]
    token_counts = count_window_tokens(lane_requests, window)
    measured_requests = [
        item
        for item in lane_requests
        if item.success
        and item.started_monotonic >= window.measurement_start
        and item.started_monotonic < window.measurement_end
    ]
    completed_inside = [
        item for item in measured_requests if item.completed_monotonic < window.measurement_end
    ]
    ttft = [
        (item.token_events[0].completion_monotonic - item.started_monotonic) * 1000
        for item in measured_requests
        if item.token_events
    ]
    latency = [
        (item.completed_monotonic - item.started_monotonic) * 1000 for item in completed_inside
    ]
    inter_token: list[float] = []
    for item in measured_requests:
        times = [event.completion_monotonic for event in item.token_events]
        inter_token.extend((current - previous) * 1000 for previous, current in pairwise(times))
    request_rates = [
        len(item.token_events) / max(item.completed_monotonic - item.started_monotonic, 1e-12)
        for item in completed_inside
    ]
    duration = window.duration_seconds
    return {
        "lane": lane,
        **token_counts,
        "verified_output_tokens": token_counts["tokens_inside_window"],
        "verified_tokens_per_second": token_counts["tokens_inside_window"] / duration,
        "completed_requests": len(completed_inside),
        "successful_requests_including_drain": sum(item.success for item in lane_requests),
        "failed_requests": sum(not item.success for item in lane_requests),
        "ttft_p50_ms": _percentile(ttft, 0.50),
        "ttft_p95_ms": _percentile(ttft, 0.95),
        "ttft_p99_ms": _percentile(ttft, 0.99),
        "latency_p50_ms": _percentile(latency, 0.50),
        "latency_p95_ms": _percentile(latency, 0.95),
        "latency_p99_ms": _percentile(latency, 0.99),
        "inter_token_latency_p50_ms": _percentile(inter_token, 0.50),
        "inter_token_latency_p95_ms": _percentile(inter_token, 0.95),
        "inter_token_latency_p99_ms": _percentile(inter_token, 0.99),
        "per_request_output_tps_median": statistics.median(request_rates) if request_rates else 0.0,
        "maximum_active_requests": state.maximum_active_requests,
        "maximum_queue_depth": state.maximum_queued_requests,
        "mean_queue_delay_ms": statistics.mean([item.queue_delay_ms for item in measured_requests])
        if measured_requests
        else 0.0,
    }


def combined_throughput(
    gpu_tokens: int, cpu_tokens: int, measurement_window_seconds: float
) -> float:
    if measurement_window_seconds <= 0:
        raise ValueError("measurement window must be positive")
    return (gpu_tokens + cpu_tokens) / measurement_window_seconds


async def _closed_loop_lane(
    *,
    lane: Lane,
    endpoint: str,
    fixtures: list[WorkloadFixture],
    concurrency: int,
    window: MeasurementWindow,
    artifact_hash: str,
    artifact_revision: str,
    state: LaneState,
    on_observation: Callable[[RequestObservation], None],
    workload_seed: int,
) -> list[RequestObservation]:
    observations: list[RequestObservation] = []
    lock = asyncio.Lock()

    async def worker(worker_index: int) -> None:
        sequence = worker_index
        while time.monotonic() < window.measurement_end:
            fixture = fixtures[sequence % len(fixtures)]
            admitted = time.monotonic()
            state.active(1)
            try:
                observation = await asyncio.to_thread(
                    _stream_request,
                    lane=lane,
                    endpoint=endpoint,
                    fixture=fixture,
                    artifact_hash=artifact_hash,
                    artifact_revision=artifact_revision,
                    admitted_monotonic=admitted,
                    timeout_seconds=max(60.0, window.drain_deadline - admitted),
                    workload_seed=workload_seed,
                )
            finally:
                state.active(-1)
            async with lock:
                observations.append(observation)
            on_observation(observation)
            sequence += concurrency
            if not observation.success:
                await asyncio.sleep(0.1)

    tasks = [asyncio.create_task(worker(index)) for index in range(concurrency)]
    await asyncio.gather(*tasks)
    return observations


async def _open_loop_lane(
    *,
    lane: Lane,
    endpoint: str,
    fixtures: list[WorkloadFixture],
    maximum_concurrency: int,
    arrival_rate_rps: float,
    window: MeasurementWindow,
    artifact_hash: str,
    artifact_revision: str,
    state: LaneState,
    on_observation: Callable[[RequestObservation], None],
    workload_seed: int,
) -> list[RequestObservation]:
    if arrival_rate_rps <= 0:
        raise ValueError("open-loop arrival rate must be positive")
    semaphore = asyncio.Semaphore(maximum_concurrency)
    observations: list[RequestObservation] = []
    lock = asyncio.Lock()

    async def submit(index: int, admitted: float) -> None:
        state.queued(1)
        async with semaphore:
            state.queued(-1)
            if time.monotonic() >= window.measurement_end:
                return
            state.active(1)
            try:
                observation = await asyncio.to_thread(
                    _stream_request,
                    lane=lane,
                    endpoint=endpoint,
                    fixture=fixtures[index % len(fixtures)],
                    artifact_hash=artifact_hash,
                    artifact_revision=artifact_revision,
                    admitted_monotonic=admitted,
                    timeout_seconds=max(60.0, window.drain_deadline - admitted),
                    workload_seed=workload_seed,
                )
            finally:
                state.active(-1)
            async with lock:
                observations.append(observation)
            on_observation(observation)

    interval = 1.0 / arrival_rate_rps
    tasks: list[asyncio.Task[None]] = []
    index = 0
    next_arrival = window.warmup_start
    while next_arrival < window.measurement_end:
        delay = next_arrival - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        admitted = time.monotonic()
        tasks.append(asyncio.create_task(submit(index, admitted)))
        index += 1
        next_arrival = window.warmup_start + index * interval
    await asyncio.gather(*tasks)
    return observations


def _telemetry_metrics(
    samples: list[dict[str, float]], window: MeasurementWindow
) -> dict[str, Any]:
    measured = [
        item
        for item in samples
        if window.measurement_start
        <= float(item.get("monotonic_seconds", -1))
        < window.measurement_end
    ]

    def mean(name: str) -> float | None:
        values = [float(item[name]) for item in measured if name in item]
        return statistics.mean(values) if values else None

    def maximum(name: str) -> float | None:
        values = [float(item[name]) for item in measured if name in item]
        return max(values) if values else None

    return {
        "telemetry_samples": len(measured),
        "host_cpu_percent_mean": mean("host_cpu_percent"),
        "host_cpu_attribution_scope": "whole_host_not_per_service",
        "sglang_host_cpu_attribution_status": "unavailable_from_docker_host_telemetry",
        "host_memory_used_bytes_maximum": maximum("host_memory_used_bytes"),
        "gpu_utilisation_percent_mean": mean("gpu_utilisation_percent"),
        "gpu_memory_used_bytes_maximum": maximum("gpu_memory_used_bytes"),
        "gpu_power_watts_mean": mean("gpu_power_watts"),
        "cpu_power_watts_mean": None,
        "cpu_power_measurement_status": "unavailable_on_host",
    }


async def run_serving_window(
    *,
    run_point_id: str,
    arm: ServingArm,
    repeat: int,
    traffic_mode: TrafficMode,
    gpu_endpoint: str,
    cpu_endpoint: str,
    gpu_fixtures: list[WorkloadFixture],
    cpu_fixtures: list[WorkloadFixture],
    gpu_concurrency: int,
    cpu_concurrency: int,
    open_loop_arrival_rate_rps: float | None,
    warmup_seconds: float,
    measurement_seconds: float,
    drain_timeout_seconds: float,
    gpu_artifact_hash: str,
    gpu_revision: str,
    cpu_artifact_hash: str,
    cpu_revision: str,
    event_writer: TokenEventWriter,
    cpu_thread_count: int,
    workload_seed: int,
    telemetry_interval_seconds: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[RequestObservation]]:
    warmup_start = time.monotonic()
    window = MeasurementWindow(
        warmup_start=warmup_start,
        measurement_start=warmup_start + warmup_seconds,
        measurement_end=warmup_start + warmup_seconds + measurement_seconds,
        drain_deadline=(
            warmup_start + warmup_seconds + measurement_seconds + drain_timeout_seconds
        ),
    )
    gpu_state = LaneState()
    cpu_state = LaneState()
    telemetry = HostTelemetry(interval_seconds=telemetry_interval_seconds)
    telemetry.start()

    def persist(observation: RequestObservation) -> None:
        event_writer.write_request(
            observation,
            window,
            run_point_id=run_point_id,
            arm=arm,
            repeat=repeat,
            traffic_mode=traffic_mode,
        )

    gpu_task: asyncio.Task[list[RequestObservation]] | None = None
    cpu_task: asyncio.Task[list[RequestObservation]] | None = None
    if arm in {"gpu_only", "gpu_plus_cpu"}:
        if traffic_mode == "closed_loop":
            gpu_task = asyncio.create_task(
                _closed_loop_lane(
                    lane="gpu",
                    endpoint=gpu_endpoint,
                    fixtures=gpu_fixtures,
                    concurrency=gpu_concurrency,
                    window=window,
                    artifact_hash=gpu_artifact_hash,
                    artifact_revision=gpu_revision,
                    state=gpu_state,
                    on_observation=persist,
                    workload_seed=workload_seed,
                )
            )
        else:
            if open_loop_arrival_rate_rps is None:
                raise ValueError("open-loop traffic requires an arrival rate")
            gpu_task = asyncio.create_task(
                _open_loop_lane(
                    lane="gpu",
                    endpoint=gpu_endpoint,
                    fixtures=gpu_fixtures,
                    maximum_concurrency=gpu_concurrency,
                    arrival_rate_rps=open_loop_arrival_rate_rps,
                    window=window,
                    artifact_hash=gpu_artifact_hash,
                    artifact_revision=gpu_revision,
                    state=gpu_state,
                    on_observation=persist,
                    workload_seed=workload_seed,
                )
            )
    if arm in {"gpu_plus_cpu", "cpu_only"}:
        cpu_task = asyncio.create_task(
            _closed_loop_lane(
                lane="cpu",
                endpoint=cpu_endpoint,
                fixtures=cpu_fixtures,
                concurrency=cpu_concurrency,
                window=window,
                artifact_hash=cpu_artifact_hash,
                artifact_revision=cpu_revision,
                state=cpu_state,
                on_observation=persist,
                workload_seed=workload_seed,
            )
        )
    tasks = [item for item in (gpu_task, cpu_task) if item is not None]
    try:
        lane_results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=warmup_seconds + measurement_seconds + drain_timeout_seconds + 5,
        )
    finally:
        telemetry_samples = telemetry.stop()
    requests = [item for lane_result in lane_results for item in lane_result]
    gpu_metrics = lane_metrics("gpu", requests, window, gpu_state)
    cpu_metrics = lane_metrics("cpu", requests, window, cpu_state)
    telemetry_metrics = _telemetry_metrics(telemetry_samples, window)
    combined_tps = combined_throughput(
        int(gpu_metrics["verified_output_tokens"]),
        int(cpu_metrics["verified_output_tokens"]),
        window.duration_seconds,
    )
    scheduler_overhead = (
        statistics.mean(
            [
                item.queue_delay_ms
                for item in requests
                if item.started_monotonic >= window.measurement_start
            ]
        )
        if requests
        else 0.0
    )
    combined = {
        "classification": {
            "gpu_only": "measured_cuda",
            "gpu_plus_cpu": "measured_mixed_backend",
            "cpu_only": "measured_x86_cpu",
        }[arm],
        "metric_version": "fixed_window_token_accounting_v1",
        "denominator_kind": "shared_fixed_measurement_window",
        "combined_throughput_formula": (
            "(verified_gpu_output_tokens + verified_cpu_output_tokens) / measurement_window_seconds"
        ),
        "run_point_id": run_point_id,
        "arm": arm,
        "repeat": repeat,
        "traffic_mode": traffic_mode,
        "gpu_concurrency": gpu_concurrency,
        "cpu_concurrency": cpu_concurrency,
        "open_loop_arrival_rate_rps": open_loop_arrival_rate_rps,
        "workload_seed": workload_seed,
        "cpu_thread_count": cpu_thread_count,
        "gpu_workload_fixture_hash": workload_fixture_hash(gpu_fixtures),
        "cpu_workload_fixture_hash": workload_fixture_hash(cpu_fixtures),
        "warmup_start_monotonic": window.warmup_start,
        "measurement_start_monotonic": window.measurement_start,
        "measurement_end_monotonic": window.measurement_end,
        "drain_deadline_monotonic": window.drain_deadline,
        "warmup_seconds": warmup_seconds,
        "measurement_window_seconds": window.duration_seconds,
        "drain_timeout_seconds": drain_timeout_seconds,
        "gpu_verified_output_tokens": gpu_metrics["verified_output_tokens"],
        "cpu_verified_output_tokens": cpu_metrics["verified_output_tokens"],
        "gpu_tokens_before_window": gpu_metrics["tokens_before_window"],
        "gpu_tokens_inside_window": gpu_metrics["tokens_inside_window"],
        "gpu_tokens_after_window": gpu_metrics["tokens_after_window"],
        "cpu_tokens_before_window": cpu_metrics["tokens_before_window"],
        "cpu_tokens_inside_window": cpu_metrics["tokens_inside_window"],
        "cpu_tokens_after_window": cpu_metrics["tokens_after_window"],
        "gpu_verified_tps": gpu_metrics["verified_tokens_per_second"],
        "cpu_verified_tps": cpu_metrics["verified_tokens_per_second"],
        "combined_verified_tps": combined_tps,
        "gpu_completed_requests": gpu_metrics["completed_requests"],
        "cpu_completed_requests": cpu_metrics["completed_requests"],
        "gpu_ttft_p50_ms": gpu_metrics["ttft_p50_ms"],
        "gpu_ttft_p95_ms": gpu_metrics["ttft_p95_ms"],
        "gpu_ttft_p99_ms": gpu_metrics["ttft_p99_ms"],
        "gpu_latency_p50_ms": gpu_metrics["latency_p50_ms"],
        "gpu_latency_p95_ms": gpu_metrics["latency_p95_ms"],
        "gpu_latency_p99_ms": gpu_metrics["latency_p99_ms"],
        "gpu_inter_token_latency_p50_ms": gpu_metrics["inter_token_latency_p50_ms"],
        "gpu_inter_token_latency_p95_ms": gpu_metrics["inter_token_latency_p95_ms"],
        "gpu_inter_token_latency_p99_ms": gpu_metrics["inter_token_latency_p99_ms"],
        "cpu_latency_p50_ms": cpu_metrics["latency_p50_ms"],
        "cpu_latency_p95_ms": cpu_metrics["latency_p95_ms"],
        "cpu_latency_p99_ms": cpu_metrics["latency_p99_ms"],
        "gpu_maximum_queue_depth": gpu_metrics["maximum_queue_depth"],
        "cpu_maximum_queue_depth": cpu_metrics["maximum_queue_depth"],
        "gpu_maximum_active_requests": gpu_metrics["maximum_active_requests"],
        "cpu_maximum_active_requests": cpu_metrics["maximum_active_requests"],
        "scheduler_overhead_ms": scheduler_overhead,
        "pcie_interference_measurement_status": "dedicated_pcie_counter_unavailable",
        "post_window_tokens_excluded": (
            int(gpu_metrics["tokens_after_window"]) + int(cpu_metrics["tokens_after_window"])
        ),
        "pre_window_tokens_excluded": (
            int(gpu_metrics["tokens_before_window"]) + int(cpu_metrics["tokens_before_window"])
        ),
        **telemetry_metrics,
    }
    if telemetry_metrics["gpu_power_watts_mean"]:
        combined["verified_tokens_per_joule_partial_gpu_only"] = combined_tps / float(
            telemetry_metrics["gpu_power_watts_mean"]
        )
        combined["total_power_status"] = "partial_cpu_power_unavailable"
    else:
        combined["verified_tokens_per_joule_partial_gpu_only"] = None
        combined["total_power_status"] = "unavailable"
    gpu_row = {**combined, **{f"gpu_{key}": value for key, value in gpu_metrics.items()}}
    cpu_row = {**combined, **{f"cpu_{key}": value for key, value in cpu_metrics.items()}}
    sample_requests = [
        item
        for item in requests
        if item.success and item.token_events and item.started_monotonic >= window.measurement_start
    ][:4]
    return combined, gpu_row, cpu_row, sample_requests


def _reference_token_ids(
    observation: RequestObservation,
    *,
    endpoint: str,
    workload_seed: int,
) -> list[int]:
    if observation.lane == "gpu":
        response = post_json(
            endpoint,
            "/generate",
            {
                "input_ids": observation.prompt_token_ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": observation.requested_output_tokens,
                    "ignore_eos": True,
                },
                "stream": False,
                "rid": f"exp007-correction-audit-{uuid4().hex}",
            },
            600,
        )
        return _sglang_output_ids(response)
    response = post_json(
        endpoint,
        "/completion",
        {
            "prompt": observation.prompt_token_ids,
            "n_predict": observation.requested_output_tokens,
            "temperature": 0.0,
            "seed": workload_seed,
            "ignore_eos": True,
            "cache_prompt": True,
            "n_probs": 1,
            "return_tokens": True,
            "stream": False,
            "id_slot": -1,
        },
        600,
    )
    for key in ("tokens", "token_ids", "output_ids"):
        value = response.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return [int(item) for item in value]
    probabilities = response.get("completion_probabilities")
    if isinstance(probabilities, list):
        values = [item.get("id") for item in probabilities if isinstance(item, dict)]
        if values and all(isinstance(item, int) for item in values):
            return [cast(int, item) for item in values]
    content = response.get("content")
    if isinstance(content, str):
        tokenized = post_json(
            endpoint,
            "/tokenize",
            {"content": content, "add_special": False},
            600,
        )
        tokenized_values = tokenized.get("tokens")
        if isinstance(tokenized_values, list) and all(
            isinstance(item, int) for item in tokenized_values
        ):
            return [int(item) for item in tokenized_values]
    raise RuntimeError(
        "llama.cpp reference response did not expose token IDs or tokenizable content"
    )


def verify_sampled_requests(
    samples: list[RequestObservation],
    *,
    gpu_endpoint: str,
    cpu_endpoint: str,
    workload_seed: int,
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], RequestObservation] = {}
    for sample in samples:
        unique.setdefault((sample.lane, sample.fixture_id), sample)
    rows: list[dict[str, Any]] = []
    for sample in unique.values():
        reference = _reference_token_ids(
            sample,
            endpoint=gpu_endpoint if sample.lane == "gpu" else cpu_endpoint,
            workload_seed=workload_seed,
        )
        identity = sample.token_ids == reference
        rows.append(
            {
                "lane": sample.lane,
                "fixture_id": sample.fixture_id,
                "request_id": sample.request_id,
                "artifact_hash": sample.artifact_hash,
                "artifact_revision": sample.artifact_revision,
                "streamed_token_count": len(sample.token_ids),
                "reference_token_count": len(reference),
                "token_identity": identity,
                "verification_status": "PASS" if identity else "FAIL",
                "quantisation_interpretation": (
                    "pinned_Q4_reference_identity"
                    if sample.lane == "cpu"
                    else "immutable_BF16_target_identity"
                ),
            }
        )
    return rows


def aggregate_fixed_window_results(
    rows: list[dict[str, Any]],
    *,
    minimum_combined_gain_fraction: float,
    maximum_gpu_p95_increase_fraction: float,
    maximum_gpu_throughput_decrease_fraction: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["traffic_mode"],
            int(row["gpu_concurrency"]),
            int(row["cpu_concurrency"]),
            row.get("open_loop_arrival_rate_rps"),
            row["arm"],
        )
        groups.setdefault(key, []).append(row)

    def median(group: list[dict[str, Any]], name: str) -> float:
        return statistics.median(float(item[name]) for item in group)

    aggregate_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in groups.items():
        aggregate_by_key[key] = {
            "traffic_mode": key[0],
            "gpu_concurrency": key[1],
            "cpu_concurrency": key[2],
            "open_loop_arrival_rate_rps": key[3],
            "arm": key[4],
            "repeats": len(group),
            "measurement_window_seconds": median(group, "measurement_window_seconds"),
            "gpu_verified_tps_median": median(group, "gpu_verified_tps"),
            "cpu_verified_tps_median": median(group, "cpu_verified_tps"),
            "combined_verified_tps_median": median(group, "combined_verified_tps"),
            "gpu_p95_latency_ms_median": statistics.median(
                float(item.get("gpu_latency_p95_ms", 0.0)) for item in group
            ),
            "gpu_verified_tokens_median": statistics.median(
                float(item["gpu_verified_output_tokens"]) for item in group
            ),
            "cpu_verified_tokens_median": statistics.median(
                float(item["cpu_verified_output_tokens"]) for item in group
            ),
            "fixed_window_formula_used": all(
                item.get("denominator_kind") == "shared_fixed_measurement_window" for item in group
            ),
        }
    comparisons: list[dict[str, Any]] = []
    for key, paired in aggregate_by_key.items():
        if key[4] != "gpu_plus_cpu":
            continue
        baseline_key = (key[0], key[1], 0, key[3], "gpu_only")
        baseline = aggregate_by_key.get(baseline_key)
        if baseline is None:
            raise RuntimeError(f"fixed-window paired point lacks matched GPU baseline: {key}")
        baseline_tps = float(baseline["gpu_verified_tps_median"])
        paired_gpu_tps = float(paired["gpu_verified_tps_median"])
        baseline_p95 = float(baseline["gpu_p95_latency_ms_median"])
        paired_p95 = float(paired["gpu_p95_latency_ms_median"])
        combined_gain = float(paired["combined_verified_tps_median"]) / max(baseline_tps, 1e-12) - 1
        throughput_change = paired_gpu_tps / max(baseline_tps, 1e-12) - 1
        p95_change = paired_p95 / max(baseline_p95, 1e-12) - 1
        positive = (
            combined_gain >= minimum_combined_gain_fraction
            and p95_change <= maximum_gpu_p95_increase_fraction
            and throughput_change >= -maximum_gpu_throughput_decrease_fraction
            and float(paired["cpu_verified_tps_median"]) > 0
        )
        comparisons.append(
            {
                **paired,
                "baseline_gpu_verified_tps_median": baseline_tps,
                "baseline_gpu_p95_latency_ms_median": baseline_p95,
                "gpu_throughput_change_fraction": throughput_change,
                "gpu_p95_latency_change_fraction": p95_change,
                "combined_gain_fraction": combined_gain,
                "positive_contribution_pass": positive,
                "acceptance_evaluated_over_median_repeats": True,
            }
        )
    return comparisons
