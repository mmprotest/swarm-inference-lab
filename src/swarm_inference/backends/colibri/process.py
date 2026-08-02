"""Lifecycle and HTTP streaming control for a local Colibri engine gateway."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psutil

from swarm_inference.backends.colibri.model import resolve_model_family
from swarm_inference.backends.colibri.schemas import (
    ColibriGenerationResult,
    ColibriMode,
    TelemetryLevel,
)
from swarm_inference.backends.http import get_json, post_json


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, port)) == 0


class ColibriProcess:
    """Own one gateway and its engine child; never mutates the caller environment."""

    def __init__(
        self,
        *,
        engine_directory: str | Path,
        model_path: str | Path,
        model_id: str,
        model_revision: str,
        mode: ColibriMode = ColibriMode.BRIDGE,
        telemetry_level: TelemetryLevel = TelemetryLevel.SUMMARY,
        telemetry_path: str | Path | None = None,
        model_family: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        cap: int | None = None,
        max_tokens: int = 1024,
        kv_slots: int = 1,
        environment: dict[str, str] | None = None,
        log_directory: str | Path | None = None,
        ram_safety_reserve_bytes: int = 8 * 1024**3,
    ) -> None:
        self.engine_directory = Path(engine_directory).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.model_id = model_id
        self.model_revision = model_revision
        config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8"))
        self.model_family = resolve_model_family(config, model_family)
        self.mode = mode
        self.telemetry_level = telemetry_level
        self.host = host
        self.port = port or free_local_port()
        self.cap = cap
        self.max_tokens = max_tokens
        self.kv_slots = kv_slots
        self.extra_environment = dict(environment or {})
        self.log_directory = (
            Path(log_directory).expanduser().resolve()
            if log_directory is not None
            else self.model_path / ".swarm_colibri_logs"
        )
        self.telemetry_path = (
            Path(telemetry_path).expanduser().resolve()
            if telemetry_path is not None
            else self.log_directory / "telemetry.ndjson"
        )
        self.ram_safety_reserve_bytes = ram_safety_reserve_bytes
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout_handle: Any = None
        self.stderr_handle: Any = None
        self.pid_file = self.log_directory / "colibri_gateway.pid.json"
        self._cancellations: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _engine(self) -> Path:
        basename = {
            "glm-5.2": "colibri",
            "inkling": "inkling",
            "kimi-k3": "kimi_k3",
            "olmoe": "olmoe",
        }[self.model_family]
        for name in (basename, f"{basename}.exe"):
            path = self.engine_directory / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"missing Colibri engine for {self.model_family}")

    def _gateway(self) -> Path:
        path = self.engine_directory / "openai_server.py"
        if not path.is_file():
            raise FileNotFoundError(f"Colibri OpenAI gateway is missing: {path}")
        return path

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Every process starts from a controlled Colibri surface. Callers can
        # opt settings back in through ``extra_environment``; a prior run or
        # interactive shell cannot leak placement or quality knobs into it.
        for key in {
            "SNAP",
            "COLI_MODEL",
            "PROMPT",
            "TOKENS",
            "REPLAY",
            "REF",
            "REF_FORCE",
            "PPL",
            "ROUTE_TRACE",
            "COLI_REQUEST_ID",
            "COLI_USAGE_PATH",
            "COLI_HOT_PIN_PATH",
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
        }:
            env.pop(key, None)
        env.update(self.extra_environment)
        env["COLI_MODEL"] = str(self.model_path)
        env["COLI_MODEL_ID"] = self.model_id
        env["COLI_MODEL_REVISION"] = self.model_revision
        if self.mode == ColibriMode.BRIDGE:
            if not (self.engine_directory / "swarm_bridge.py").is_file():
                raise FileNotFoundError("bridge mode requested but swarm_bridge.py is not built")
            env["COLI_SWARM_BRIDGE"] = "1"
            env["COLI_SWARM_BRIDGE_PATH"] = str(self.telemetry_path)
            env["COLI_SWARM_TELEMETRY"] = self.telemetry_level.value
            if self.telemetry_level != TelemetryLevel.OFF:
                # Enables Colibri's existing exact phase counters.  This is
                # instrumentation only and is applied equally to direct and
                # adapted benchmark configurations.
                env.setdefault("PROF", "1")
        else:
            env.pop("COLI_SWARM_BRIDGE", None)
            env.pop("COLI_SWARM_BRIDGE_PATH", None)
            env.pop("COLI_SWARM_TELEMETRY", None)
        return env

    def start(self, *, timeout_seconds: float = 600.0) -> None:
        if self.running:
            return
        if self.model_family == "olmoe":
            raise NotImplementedError(
                "Colibri v1.4.0 OLMoE has no persistent mux server; use ColibriReplayRunner"
            )
        if psutil.virtual_memory().available < self.ram_safety_reserve_bytes:
            raise MemoryError("available RAM is below the configured Colibri safety reserve")
        if _is_listening(self.host, self.port):
            raise RuntimeError(
                f"refusing to start over a stale or occupied endpoint {self.endpoint}"
            )
        self.cleanup_orphaned_process(self.pid_file, expected_model=self.model_path)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.stdout_handle = (self.log_directory / "gateway.stdout.log").open("ab")
        self.stderr_handle = (self.log_directory / "gateway.stderr.log").open("ab")
        arch = {"glm-5.2": "glm", "inkling": "inkling", "kimi-k3": "kimi"}[self.model_family]
        command = [
            sys.executable,
            str(self._gateway()),
            "--model",
            str(self.model_path),
            "--engine",
            str(self._engine()),
            "--arch",
            arch,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--model-id",
            self.model_id,
            "--max-tokens",
            str(self.max_tokens),
            "--kv-slots",
            str(self.kv_slots),
        ]
        if self.cap is not None:
            command.extend(("--cap", str(self.cap)))
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command,
                cwd=self.engine_directory,
                env=self._environment(),
                stdin=subprocess.DEVNULL,
                stdout=self.stdout_handle,
                stderr=self.stderr_handle,
                creationflags=creationflags,
            )
            self.pid_file.write_text(
                json.dumps(
                    {
                        "pid": self.process.pid,
                        "model_path": str(self.model_path),
                        "command": command,
                        "started_ns": time.time_ns(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            deadline = time.monotonic() + timeout_seconds
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Colibri gateway exited with code {self.process.returncode}"
                    )
                try:
                    health = get_json(self.endpoint, "/health", 2.0)
                    if health.get("status") == "ok":
                        return
                except RuntimeError as exc:
                    last_error = exc
                time.sleep(0.1)
            raise TimeoutError(f"Colibri did not become ready: {last_error}")
        except Exception:
            self.shutdown()
            raise

    def health(self, *, timeout_seconds: float = 5.0) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("Colibri process is not running")
        return get_json(self.endpoint, "/health", timeout_seconds)

    def generate(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_id: str,
        timeout_seconds: float = 600.0,
    ) -> ColibriGenerationResult:
        if not self.running:
            raise RuntimeError("Colibri process is not running")
        started = time.perf_counter()
        response = post_json(
            self.endpoint,
            "/v1/completions",
            {
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            },
            timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        choice = response.get("choices", [{}])[0]
        usage = response.get("usage", {})
        raw_bridge = response.get("colibri")
        bridge: dict[str, Any] | None = raw_bridge if isinstance(raw_bridge, dict) else None
        token_identity = bridge is not None
        input_ids = bridge.get("input_token_ids") if bridge is not None else None
        output_ids = bridge.get("token_ids") if bridge is not None else None
        completion = int(usage.get("completion_tokens", 0))
        return ColibriGenerationResult(
            request_id=request_id,
            text=str(choice.get("text", "")),
            input_token_ids=input_ids,
            output_token_ids=output_ids,
            token_identity_observed=token_identity,
            stop_reason=str(choice.get("finish_reason", "unknown")),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=completion,
            elapsed_ms=elapsed_ms,
            time_to_first_token_ms=None,
            decode_tokens_per_second=self._latest_decode_rate(),
            raw_response=response,
        )

    def stream_generate(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_id: str,
        timeout_seconds: float = 600.0,
        on_text: Callable[[str], None] | None = None,
    ) -> ColibriGenerationResult:
        if not self.running:
            raise RuntimeError("Colibri process is not running")
        payload = json.dumps(
            {
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            self.endpoint + "/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        cancellation = threading.Event()
        with self._lock:
            self._cancellations[request_id] = cancellation
        started = time.perf_counter()
        first_text: float | None = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        bridge: dict[str, Any] | None = None
        finish_reason = "unknown"
        raw_events: list[dict[str, Any]] = []
        try:
            try:
                response_context = urlopen(request, timeout=timeout_seconds)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} from Colibri: {detail}") from exc
            except URLError as exc:
                raise RuntimeError(f"Colibri stream unavailable: {exc}") from exc
            with response_context as response:
                for raw_line in response:
                    if cancellation.is_set():
                        raise InterruptedError(f"Colibri request {request_id} was cancelled")
                    line = raw_line.decode("utf-8", errors="strict").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if not isinstance(event, dict):
                        raise RuntimeError("Colibri SSE event is not an object")
                    raw_events.append(event)
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    if isinstance(event.get("colibri"), dict):
                        bridge = event["colibri"]
                    choices = event.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        text = str(choice.get("text", ""))
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            text = str(delta.get("content", ""))
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            finish_reason = str(reason)
                        if text:
                            if first_text is None:
                                first_text = time.perf_counter()
                            chunks.append(text)
                            if on_text is not None:
                                on_text(text)
        finally:
            with self._lock:
                self._cancellations.pop(request_id, None)
        finished = time.perf_counter()
        token_identity = bridge is not None
        input_ids = bridge.get("input_token_ids") if bridge else None
        output_ids = bridge.get("token_ids") if bridge else None
        prompt_tokens = int(usage.get("prompt_tokens", len(input_ids) if input_ids else 0))
        completion_tokens = int(
            usage.get("completion_tokens", len(output_ids) if output_ids else 0)
        )
        return ColibriGenerationResult(
            request_id=request_id,
            text="".join(chunks),
            input_token_ids=input_ids,
            output_token_ids=output_ids,
            token_identity_observed=token_identity,
            stop_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=(finished - started) * 1000,
            time_to_first_token_ms=(first_text - started) * 1000 if first_text else None,
            decode_tokens_per_second=self._latest_decode_rate(),
            raw_response={"events": raw_events, "usage": usage, "colibri": bridge},
        )

    def _latest_decode_rate(self) -> float | None:
        if self.mode != ColibriMode.BRIDGE or not self.telemetry_path.is_file():
            return None
        try:
            lines = self.telemetry_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "request_completed":
                value = event.get("payload", {}).get("tokens_per_second")
                return float(value) if value is not None else None
        return None

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._cancellations.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def shutdown(self, *, timeout_seconds: float = 10.0) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                self._terminate_tree(process.pid)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        if self.pid_file.is_file():
            with suppress(OSError):
                self.pid_file.unlink()
        for handle_name in ("stdout_handle", "stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                with suppress(OSError):
                    handle.close()
                setattr(self, handle_name, None)

    @staticmethod
    def _terminate_tree(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            return
        children = parent.children(recursive=True)
        for process in children:
            with suppress(psutil.Error):
                process.terminate()
        _, alive = psutil.wait_procs(children, timeout=3)
        for process in alive:
            with suppress(psutil.Error):
                process.kill()
        with suppress(psutil.Error):
            parent.kill()

    @staticmethod
    def cleanup_orphaned_process(pid_file: str | Path, *, expected_model: Path) -> bool:
        path = Path(pid_file)
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            pid = int(record["pid"])
            recorded_model = Path(record["model_path"]).resolve()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Colibri PID record: {path}") from error
        if recorded_model != expected_model.resolve():
            raise RuntimeError("refusing to clean a Colibri process for a different model")
        try:
            process = psutil.Process(pid)
            command = " ".join(process.cmdline()).lower()
        except psutil.NoSuchProcess:
            with suppress(OSError):
                path.unlink()
            return False
        if "openai_server.py" not in command or str(expected_model).lower() not in command:
            raise RuntimeError(f"PID {pid} is not the recorded Colibri gateway; refusing cleanup")
        ColibriProcess._terminate_tree(pid)
        with suppress(OSError):
            path.unlink()
        return True

    def __enter__(self) -> ColibriProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
