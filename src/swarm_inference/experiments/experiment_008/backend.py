"""Native llama.cpp lifecycle and deterministic streaming client for Experiment 008.

The adapter deliberately discovers capabilities from the selected executable.  It
does not infer dynamic expert caching, routing traces, or overlap support merely
because a request happened to use both CPU and GPU resources.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from swarm_inference.experiments.experiment_008.planning import BackendCapabilities


class BackendProcessError(RuntimeError):
    """Backend lifecycle failure carrying the real child-process exit code."""

    def __init__(self, message: str, *, returncode: int | None) -> None:
        super().__init__(message)
        self.returncode = returncode


def free_local_port(host: str = "127.0.0.1", *, start: int | None = None) -> int:
    """Return an available loopback port, honouring a preferred starting port."""

    if start is not None:
        for port in range(start, min(start + 100, 65_535)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                try:
                    listener.bind((host, port))
                except OSError:
                    continue
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def file_sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class BackendProbe:
    executable: str
    executable_sha256: str
    version_output: str
    help_output: str
    version_exit_code: int
    help_exit_code: int
    capabilities: BackendCapabilities

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = self.capabilities.model_dump(mode="json")
        return payload


def _run_probe(executable: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )


def probe_llama_server(executable: Path) -> BackendProbe:
    resolved = executable.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"llama.cpp server executable does not exist: {resolved}")
    version = _run_probe(resolved, ["--version"])
    help_result = _run_probe(resolved, ["--help"])
    help_text = "\n".join(value for value in (help_result.stdout, help_result.stderr) if value)
    version_text = "\n".join(value for value in (version.stdout, version.stderr) if value)

    def has(flag: str) -> bool:
        return flag in help_text

    limitations: list[str] = []
    if not has("--override-tensor"):
        limitations.append("binary does not expose tensor-buffer override placement")
    if not (has("--cpu-moe") or has("--n-cpu-moe")):
        limitations.append("binary does not expose MoE-specific CPU placement")
    # No released llama.cpp server API currently exports routed expert IDs or a
    # dynamic expert residency/prefetch controller.  These stay false unless an
    # explicitly extended executable advertises the experiment-specific flags.
    routing_trace = has("--exp008-expert-trace")
    dynamic_residency = has("--exp008-expert-cache")
    expert_prefetch = has("--exp008-expert-prefetch")
    overlap_trace = has("--exp008-overlap-trace")
    if not routing_trace:
        limitations.append("server API does not export routed expert IDs")
    if not dynamic_residency:
        limitations.append("server does not expose per-expert dynamic residency")
    if not expert_prefetch:
        limitations.append("server does not expose bounded expert prefetch scheduling")
    if not overlap_trace:
        limitations.append("server does not expose operation-level CPU/GPU overlap traces")
    capabilities = BackendCapabilities(
        conventional_layer_offload=has("--n-gpu-layers") or has("-ngl"),
        tensor_buffer_override=has("--override-tensor"),
        cpu_moe=has("--cpu-moe") or has("--n-cpu-moe"),
        asynchronous_backend_scheduler=has("--parallel") or has("--cont-batching"),
        operation_level_overlap_trace=overlap_trace,
        expert_routing_trace=routing_trace,
        per_expert_dynamic_residency=dynamic_residency,
        expert_prefetch=expert_prefetch,
        separate_process_phase_plans=True,
        in_request_phase_switch=False,
        deterministic_greedy_tokens=True,
        final_logits=False,
        limitations=limitations,
    )
    return BackendProbe(
        executable=str(resolved),
        executable_sha256=file_sha256(resolved),
        version_output=version_text.strip(),
        help_output=help_text,
        version_exit_code=version.returncode,
        help_exit_code=help_result.returncode,
        capabilities=capabilities,
    )


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama.cpp server unavailable at {endpoint}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"llama.cpp {path} returned a non-object JSON response")
    return result


def _iter_sse_json(
    endpoint: str, path: str, payload: dict[str, Any], timeout: float
) -> Iterator[dict[str, Any]]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/" + path.lstrip("/"),
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
                    raise RuntimeError("llama.cpp streaming event was not a JSON object")
                yield event
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama.cpp server unavailable at {endpoint}: {exc}") from exc


def event_token_ids(event: dict[str, Any]) -> list[int]:
    """Extract the non-cumulative token IDs emitted by llama.cpp."""

    for key in ("tokens", "token_ids", "output_ids"):
        value = event.get(key)
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            return [int(item) for item in value]
    probabilities = event.get("completion_probabilities")
    if isinstance(probabilities, list):
        values = [item.get("id") for item in probabilities if isinstance(item, dict)]
        if values and all(isinstance(item, int) for item in values):
            return [int(item) for item in values]
    token = event.get("token")
    if isinstance(token, int):
        return [token]
    return []


@dataclass(slots=True)
class TokenEvent:
    token_id: int
    sequence_index: int
    monotonic_ns: int


@dataclass(slots=True)
class GenerationResult:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    content: str
    token_events: list[TokenEvent]
    admitted_monotonic_ns: int
    started_monotonic_ns: int
    completed_monotonic_ns: int
    timings: dict[str, Any]
    stop_reason: str | None
    success: bool
    error: str | None = None
    raw_final_event: dict[str, Any] = field(default_factory=dict)

    @property
    def time_to_first_token_ms(self) -> float | None:
        if not self.token_events:
            return None
        return (self.token_events[0].monotonic_ns - self.started_monotonic_ns) / 1_000_000

    @property
    def elapsed_ms(self) -> float:
        return (self.completed_monotonic_ns - self.started_monotonic_ns) / 1_000_000

    @property
    def decode_tokens_per_second(self) -> float | None:
        if len(self.token_events) < 2:
            return None
        elapsed = (
            self.token_events[-1].monotonic_ns - self.token_events[0].monotonic_ns
        ) / 1_000_000_000
        return (len(self.token_events) - 1) / elapsed if elapsed > 0 else None

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        return [
            (current.monotonic_ns - previous.monotonic_ns) / 1_000_000
            for previous, current in zip(self.token_events, self.token_events[1:], strict=False)
        ]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "time_to_first_token_ms": self.time_to_first_token_ms,
                "elapsed_ms": self.elapsed_ms,
                "decode_tokens_per_second": self.decode_tokens_per_second,
                "inter_token_latencies_ms": self.inter_token_latencies_ms,
            }
        )
        return payload


class LlamaCppClient:
    def __init__(self, endpoint: str, *, timeout_seconds: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def tokenize(self, text: str, *, add_special: bool = True) -> list[int]:
        result = _post_json(
            self.endpoint,
            "/tokenize",
            {"content": text, "add_special": add_special, "parse_special": True},
            self.timeout_seconds,
        )
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(item, int) for item in tokens):
            raise RuntimeError("llama.cpp /tokenize did not return integer token IDs")
        return [int(item) for item in tokens]

    def detokenize(self, tokens: list[int]) -> str:
        result = _post_json(
            self.endpoint,
            "/detokenize",
            {"tokens": tokens},
            self.timeout_seconds,
        )
        content = result.get("content")
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp /detokenize did not return text")
        return content

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        output_tokens: int,
        seed: int,
        ignore_eos: bool = True,
        admitted_monotonic_ns: int | None = None,
    ) -> GenerationResult:
        admitted = admitted_monotonic_ns or time.perf_counter_ns()
        started = time.perf_counter_ns()
        output: list[int] = []
        events: list[TokenEvent] = []
        content_parts: list[str] = []
        final_event: dict[str, Any] = {}
        try:
            for event in _iter_sse_json(
                self.endpoint,
                "/completion",
                {
                    "prompt": prompt_token_ids,
                    "n_predict": output_tokens,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": seed,
                    "ignore_eos": ignore_eos,
                    "cache_prompt": False,
                    "n_probs": 1,
                    "return_tokens": True,
                    "stream": True,
                    "id_slot": -1,
                },
                self.timeout_seconds,
            ):
                final_event = event
                now = time.perf_counter_ns()
                ids = event_token_ids(event)
                for token_id in ids:
                    events.append(TokenEvent(token_id, len(output), now))
                    output.append(token_id)
                content = event.get("content")
                if isinstance(content, str):
                    content_parts.append(content)
            completed = time.perf_counter_ns()
            success = len(output) == output_tokens or bool(final_event.get("stop"))
            return GenerationResult(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=output,
                content="".join(content_parts),
                token_events=events,
                admitted_monotonic_ns=admitted,
                started_monotonic_ns=started,
                completed_monotonic_ns=completed,
                timings=(
                    final_event.get("timings")
                    if isinstance(final_event.get("timings"), dict)
                    else {}
                ),
                stop_reason=(
                    str(final_event.get("stop_type"))
                    if final_event.get("stop_type") is not None
                    else None
                ),
                success=success,
                error=None
                if success
                else "stream ended without a completion marker or requested tokens",
                raw_final_event=final_event,
            )
        except Exception as exc:
            return GenerationResult(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=output,
                content="".join(content_parts),
                token_events=events,
                admitted_monotonic_ns=admitted,
                started_monotonic_ns=started,
                completed_monotonic_ns=time.perf_counter_ns(),
                timings={},
                stop_reason=None,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                raw_final_event=final_event,
            )


def wait_for_server(
    endpoint: str, *, timeout_seconds: float, process: subprocess.Popen[str]
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BackendProcessError(
                f"llama.cpp server exited during startup with code {process.returncode}",
                returncode=process.returncode,
            )
        for path in ("/health", "/props", "/v1/models"):
            try:
                with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=2) as response:
                    if 200 <= int(response.status) < 500:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"llama.cpp server at {endpoint} was not ready: {last_error}")


@dataclass(slots=True)
class ManagedLlamaServer:
    process: subprocess.Popen[str]
    endpoint: str
    command: list[str]
    stdout_handle: TextIO
    stderr_handle: TextIO
    launched_monotonic_ns: int
    ready_monotonic_ns: int
    keep: bool = False

    @property
    def launch_seconds(self) -> float:
        return (self.ready_monotonic_ns - self.launched_monotonic_ns) / 1_000_000_000

    def close(self) -> int | None:
        if self.keep:
            return self.process.poll()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        code = self.process.returncode
        self.stdout_handle.close()
        self.stderr_handle.close()
        return code

    def __enter__(self) -> ManagedLlamaServer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def launch_llama_server(
    *,
    executable: Path,
    model_path: Path,
    host: str,
    port: int,
    context_size: int,
    parallel: int,
    plan_arguments: list[str],
    logs: Path,
    startup_timeout_seconds: float,
    keep: bool = False,
) -> ManagedLlamaServer:
    """Launch one owned native server process and retain exact command evidence."""

    logs.mkdir(parents=True, exist_ok=True)
    stdout_handle = (logs / "server.stdout.log").open("w", encoding="utf-8")
    stderr_handle = (logs / "server.stderr.log").open("w", encoding="utf-8")
    endpoint = f"http://{host}:{port}"
    command = [
        str(executable.resolve()),
        "--model",
        str(model_path.resolve()),
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        str(parallel),
        "--metrics",
        "--no-webui",
        "--log-verbosity",
        "4",
        *plan_arguments,
    ]
    startup_info: dict[str, Any] | None = None
    if os.name == "nt":
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    launched = time.perf_counter_ns()
    process = subprocess.Popen(
        command,
        cwd=model_path.parent,
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
        startupinfo=startup_info,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    service = ManagedLlamaServer(
        process=process,
        endpoint=endpoint,
        command=command,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        launched_monotonic_ns=launched,
        ready_monotonic_ns=launched,
        keep=keep,
    )
    try:
        wait_for_server(endpoint, timeout_seconds=startup_timeout_seconds, process=process)
    except Exception as exc:
        exit_code = service.close()
        stderr_path = logs / "server.stderr.log"
        stderr_text = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path.is_file()
            else ""
        )
        lowered = stderr_text.lower()
        failure_kind = (
            "OUT_OF_MEMORY"
            if any(
                marker in lowered
                for marker in ("out of memory", "cuda error 2", "cudaerrormemoryallocation")
            )
            else "BACKEND_STARTUP_FAILURE"
        )
        (logs / "server.exit.json").write_text(
            json.dumps(
                {
                    "exit_code": exit_code,
                    "failure_kind": failure_kind,
                    "shutdown_mode": "startup_failure_cleanup",
                    "error": f"{type(exc).__name__}: {exc}",
                    "stderr_tail": stderr_text[-8_000:],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if isinstance(exc, BackendProcessError):
            if failure_kind == "OUT_OF_MEMORY":
                raise BackendProcessError(
                    f"OUT_OF_MEMORY: {exc}", returncode=exc.returncode
                ) from exc
            raise
        raise BackendProcessError(f"{failure_kind}: {exc}", returncode=exit_code) from exc
    service.ready_monotonic_ns = time.perf_counter_ns()
    (logs / "server.command.json").write_text(
        json.dumps(
            {
                "command": command,
                "endpoint": endpoint,
                "pid": process.pid,
                "launch_seconds": service.launch_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return service
