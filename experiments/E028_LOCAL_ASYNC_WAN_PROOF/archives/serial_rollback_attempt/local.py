"""Generic asynchronous staged verification on one physical device.

The GPU lock deliberately serializes ALL target and draft calls on the one
5090. The queues, provisional chunks and dependencies are real; their wall time
is NOT three-GPU throughput. The simulator uses independent measured resources.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from swarm_inference.experiments.experiment_027.protocol import (
    Flags, Operation, REQUEST, RESPONSE, StageClient, StageResponse,
    final_output, hidden_output,
)

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "experiments/E028_LOCAL_ASYNC_WAN_PROOF"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def token_hash(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def response_trace(r: StageResponse, request_bytes: int) -> dict:
    return dict(compute_ms=r.compute_ns / 1e6, service_ms=r.round_trip_ns / 1e6,
                server_total_ms=r.total_ns / 1e6,
                deserialize_ms=r.deserialize_ns / 1e6,
                serialize_and_state_ms=r.serialize_ns / 1e6,
                client_header_serialize_ms=r.client_serialize_ns / 1e6,
                request_bytes=request_bytes, response_bytes=r.wire_bytes,
                input_bytes=request_bytes - REQUEST.size, output_bytes=len(r.payload))


class Client(StageClient):
    def command(self, operation: int, position=0):
        return self._exchange(operation, position=position)

    def stats(self):
        return json.loads(self.command(6).payload)

    def fingerprint(self):
        return json.loads(self.command(10).payload)

    def forward(self, data, *, position: int, checkpoint=False, mtp=False, prefill=False):
        flags = Flags(0)
        if isinstance(data, tuple):
            ids, rows = data
            ids = np.asarray(ids, dtype="<i4")
            rows = np.ascontiguousarray(rows, dtype="<f4")
            payload = ids.tobytes() + rows.tobytes()
            n, width = len(ids), rows.shape[1]
            flags = Flags.INPUT_TOKENS | Flags.INPUT_MTP | Flags.RETURN_NEXTN
        else:
            data = np.asarray(data)
            if data.ndim == 1:
                payload = np.asarray(data, dtype="<i4").tobytes()
                n, width = len(data), self.width
                flags = Flags.INPUT_TOKENS
            else:
                payload = np.asarray(data, dtype="<f4").tobytes()
                n, width = data.shape
            if self.final:
                flags |= Flags.RETURN_NEXTN
        if checkpoint:
            flags = Flags(int(flags) | (1 << 8))
        if prefill:
            flags = Flags(int(flags) | (1 << 9))
        r = self._exchange(Operation.INFER, payload=payload, position=position,
                           n_tokens=n, n_embd=width, flags=flags, arg=1,
                           rewind_position=-(self.commit_position+2) if not mtp and getattr(self, "commit_position", -1) >= 0 else -1)
        return (final_output(r) if self.final or mtp else hidden_output(r)), response_trace(r, REQUEST.size + len(payload))


class Workers:
    """Start only local child processes; reuse already-built llama libraries."""
    def __init__(self, config, ranges, *, profile=False, draft=True, tag="run"):
        self.config, self.ranges = config, ranges
        self.profile, self.use_draft, self.tag = profile, draft, tag
        self.processes, self.clients, self.logs = [], [], []
        self.draft = None

    def __enter__(self):
        try:
            definitions = [(s, e, False) for s, e in self.ranges]
            if self.use_draft:
                definitions.append((0, self.ranges[-1][1], True))
            for i, (start, end, mtp) in enumerate(definitions):
                port = 19828 + i
                log = OUT / "logs" / f"{self.tag}-{i}.log"
                log.parent.mkdir(exist_ok=True)
                handle = log.open("wb")
                self.logs.append(handle)
                env = os.environ.copy()
                env["PATH"] = str(ROOT / ".runtime/experiment-027/build/bin") + os.pathsep + env.get("PATH", "")
                args = [str(ROOT / ".runtime/experiment-028/build/llama-e028-stage.exe"),
                        "--model", str(ROOT / self.config["draft_model_path" if mtp else "model_path"]),
                        "--host", "127.0.0.1", "--port", str(port),
                        "--stage-start", str(start), "--stage-end", str(end),
                        "--n-ctx", str(self.config["context_tokens"]),
                        "--n-batch", str(self.config["prefill_batch_tokens"]),
                        "--n-ubatch", str(self.config["prefill_batch_tokens"]),
                        "--n-rs-seq", "0", "--gpu-layers", "999"]
                if mtp:
                    args.append("--mtp")
                elif not self.profile:
                    args += ["--serial-blocks", "--checkpoint-slots", str(self.config["checkpoint_slots"])]
                if self.profile:
                    args.append("--profile-layers")
                p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.processes.append(p)
                deadline = time.monotonic() + 240
                while "E027_READY" not in log.read_text(errors="replace"):
                    if p.poll() is not None:
                        raise RuntimeError(log.read_text(errors="replace")[-6000:])
                    if time.monotonic() > deadline:
                        raise TimeoutError(str(log))
                    time.sleep(.2)
                c = Client("127.0.0.1", port)
                facts = c.ping()
                c.width, c.final = facts["n_embd"], facts["final_stage"]
                if mtp:
                    self.draft = c
                else:
                    self.clients.append(c)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for c in [*self.clients, self.draft]:
            if c is not None:
                c.close()
        for p in reversed(self.processes):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=10)
        for h in self.logs:
            h.close()


class SpeculativeDraft(Protocol):
    """Drafts may depend on provisional history; target alone authorizes commits."""
    async def step(self, token: int, hidden: np.ndarray, position: int): ...


@dataclass
class Chunk:
    id: int
    epoch: int
    parent: int | None
    position: int
    tokens: list[int]
    launch_dependency: int | None
    draft: list[dict]
    launched_ms: float
    stages: list[dict] = field(default_factory=list)
    accepted: int | None = None
    rejected: bool = False
    invalidated: bool = False
    completed_ms: float | None = None
    decided_ms: float | None = None
    target_greedy: list[int] = field(default_factory=list)
    rollback: dict | None = None
    inputs: list[Any] = field(default_factory=list, repr=False)
    output: Any = field(default=None, repr=False)
    future: Any = field(default=None, repr=False)

    def record(self):
        return {k: v for k, v in self.__dict__.items() if k not in {"inputs", "output", "future"}}


class LocalEngine:
    def __init__(self, workers: Workers, *, shaped_network=None, seed=280912):
        self.workers = workers
        self.stages, self.draft = workers.clients, workers.draft
        self.width = self.stages[0].width
        self.gpu = asyncio.Lock()
        self.events = []
        self.start = 0
        self.shaped_network = shaped_network
        self.network_rng = random.Random(seed)
        self.shaping = []

    def now(self):
        return (time.perf_counter() - self.start) * 1000

    async def transfer_delay(self, payload_bytes):
        if self.shaped_network is None:
            return
        net = self.shaped_network
        duration = max(0., net["rtt_ms"]/2 + self.network_rng.uniform(-net["jitter_ms"],net["jitter_ms"]))
        if net["bandwidth_mbps"] is not None:
            duration += payload_bytes*8/(net["bandwidth_mbps"]*1000)
        begin = self.now()
        await asyncio.sleep(duration/1000)
        self.shaping.append(dict(bytes=payload_bytes, requested_ms=duration, actual_ms=self.now()-begin))

    async def call(self, client, data, **kwargs):
        async with self.gpu:
            begin = self.now()
            result, trace = await asyncio.to_thread(client.forward, data, **kwargs)
            trace.update(start_ms=begin, end_ms=self.now(), rows=len(data[0]) if isinstance(data, tuple) else len(data))
            return result, trace

    async def traverse(self, tokens, position, *, checkpoint=False, prefill=False, shape=False):
        data, trace, inputs = tokens, [], []
        for index, stage in enumerate(self.stages):
            if shape:
                await self.transfer_delay(REQUEST.size+4*len(data) if index==0 else RESPONSE.size+np.asarray(data).nbytes)
            inputs.append(np.asarray(data).copy())
            data, row = await self.call(stage, data, position=position, checkpoint=checkpoint, prefill=prefill)
            trace.append(row)
        if shape:
            await self.transfer_delay(trace[-1]["response_bytes"])
        return data, trace, inputs

    async def prefill(self, tokens):
        for c in [*self.stages, self.draft]:
            if c is not None:
                await asyncio.to_thread(c.reset)
                c.commit_position = -1
        rows, traces = [], []
        batch = self.workers.config["prefill_batch_tokens"]
        for p in range(0, len(tokens), batch):
            out, tr, _ = await self.traverse(tokens[p:p+batch], p, prefill=True)
            rows.append(out.nextn)
            traces.append(tr)
        h = np.concatenate(rows)
        draft_traces = []
        if self.draft is not None:
            shifted = np.concatenate((np.zeros_like(h[:1]), h[:-1]))
            for p in range(0, len(tokens), batch):
                _, tr = await self.call(self.draft, (tokens[p:p+batch], shifted[p:p+batch]), position=p, mtp=True)
                draft_traces.append(tr)
        return int(out.greedy_ids[-1]), h[-1:], dict(target=traces, draft=draft_traces)

    async def draft_step(self, token, hidden, position, rewind=-1):
        if rewind >= 0:
            # MTP is accessed only through generic token + conditioning-state API.
            async with self.gpu:
                begin = self.now()
                ids = np.asarray([token], dtype="<i4")
                h = np.asarray(hidden, dtype="<f4")
                r = await asyncio.to_thread(self.draft._exchange, Operation.INFER,
                    payload=ids.tobytes()+h.tobytes(), position=position, rewind_position=rewind,
                    n_tokens=1, n_embd=self.width,
                    flags=Flags.INPUT_TOKENS | Flags.INPUT_MTP | Flags.RETURN_NEXTN, arg=1)
                result, tr = final_output(r), response_trace(r, REQUEST.size+ids.nbytes+h.nbytes)
                tr.update(start_ms=begin, end_ms=self.now(), rows=1)
        else:
            result, tr = await self.call(self.draft, ([token], hidden), position=position, mtp=True)
        return int(result.greedy_ids[-1]), result.nextn, tr

    async def run(self, tokens, *, k, w, count=256, force=None, fingerprints=False):
        self.start = time.perf_counter()
        self.events = []
        self.shaping = []
        next_token, next_hidden, prefill = await self.prefill(tokens)
        decode_start = self.now()
        queues = [asyncio.Queue() for _ in self.stages]
        chunks, inflight, generated = [], [], []
        epoch, cursor = 0, len(tokens)
        draft_token, draft_hidden = next_token, next_hidden
        rewind_draft = -1
        canceled_epochs = set()
        peak, overlap_launches, commits = 0, 0, []
        launch_dependency = None
        last_parent = None
        rollback_rows = []
        # Optional checkpoint fingerprints are stress evidence, excluded from timing profiles.
        root_hashes = {}

        async def worker(index):
            while True:
                item = await queues[index].get()
                if item is None:
                    queues[index].task_done()
                    break
                chunk, data = item
                if chunk.invalidated:
                    if not chunk.future.done():
                        chunk.future.set_result(None)
                    queues[index].task_done()
                    continue
                await self.transfer_delay(REQUEST.size+4*len(data) if index==0 else RESPONSE.size+np.asarray(data).nbytes)
                chunk.inputs.append(np.asarray(data).copy())
                try:
                    result, tr = await self.call(self.stages[index], data, position=chunk.position, checkpoint=True)
                except Exception as error:
                    if not chunk.future.done():
                        chunk.future.set_exception(error)
                    queues[index].task_done()
                    continue
                tr["stage"] = index
                chunk.stages.append(tr)
                if index + 1 < len(self.stages):
                    await queues[index+1].put((chunk, result))
                else:
                    await self.transfer_delay(tr["response_bytes"])
                    chunk.output = result
                    chunk.target_greedy = result.greedy_ids.tolist()
                    chunk.completed_ms = self.now()
                    chunk.future.set_result(result)
                queues[index].task_done()

        tasks = [asyncio.create_task(worker(i)) for i in range(len(self.stages))]
        try:
            while len(generated) < count:
                while len(inflight) < w and cursor < len(tokens) + count:
                    block_size = min(k+1 if k else 1, len(tokens)+count-cursor)
                    block, draft_traces = [], []
                    if fingerprints and not inflight:
                        root_hashes[cursor] = [await asyncio.to_thread(c.fingerprint) for c in self.stages]
                    for j in range(block_size):
                        tok = int(draft_token)
                        if force is not None and not inflight and force["remaining"] > 0:
                            absolute = cursor-len(tokens)+j
                            if j <= force["position"]:
                                tok = int(force["reference"][absolute])
                                if j == force["position"]:
                                    tok = (tok + 1) % force["n_vocab"]
                        block.append(tok)
                        if k:
                            draft_token, draft_hidden, tr = await self.draft_step(
                                tok, draft_hidden, cursor+j, rewind=rewind_draft)
                            rewind_draft = -1
                            draft_traces.append(tr)
                    chunk = Chunk(len(chunks), epoch, last_parent, cursor, block,
                                  launch_dependency, draft_traces, self.now())
                    chunk.future = asyncio.get_running_loop().create_future()
                    if any(c.completed_ms is None for c in inflight):
                        overlap_launches += 1
                    chunks.append(chunk)
                    inflight.append(chunk)
                    peak = max(peak, len(inflight))
                    self.events.append(dict(event="launch", time_ms=self.now(), chunk=chunk.id,
                                            position=cursor, depth=len(inflight), epoch=epoch))
                    await queues[0].put((chunk, np.asarray(block, dtype=np.int32)))
                    cursor += len(block)
                    last_parent = chunk.id
                head = inflight[0]
                out = await head.future
                accepted, expected = 0, next_token
                for j, tok in enumerate(head.tokens):
                    if tok != expected:
                        break
                    accepted += 1
                    expected = int(out.greedy_ids[j])
                head.accepted = accepted
                head.rejected = accepted < len(head.tokens)
                head.decided_ms = self.now()
                generated.extend(head.tokens[:accepted])
                if accepted:
                    next_hidden = out.nextn[accepted-1:accepted]
                next_token = expected
                commits.extend([self.now()] * accepted)
                self.events.append(dict(event="commit", time_ms=self.now(), chunk=head.id,
                                        accepted=accepted, position=len(tokens)+len(generated), depth=len(inflight)))
                inflight.pop(0)
                launch_dependency = head.id
                if head.rejected:
                    if force is not None and force["remaining"] > 0:
                        force["remaining"] -= 1
                    self.events.append(dict(event="reject", chunk=head.id, time_ms=self.now(),
                                            accepted=accepted, invalidated=[c.id for c in inflight]))
                    for c in inflight:
                        c.invalidated = True
                    # Drain already running kernels; queued stale work is skipped.
                    for q in queues:
                        await q.join()
                    rb_begin = self.now()
                    commands = []
                    for stage in self.stages:
                        await self.transfer_delay(REQUEST.size)
                        async with self.gpu:
                            r = await asyncio.to_thread(stage.command, 9, head.position)
                        await self.transfer_delay(RESPONSE.size)
                        commands.append(response_trace(r, REQUEST.size))
                    integrity = None
                    if head.position in root_hashes:
                        after = [await asyncio.to_thread(c.fingerprint) for c in self.stages]
                        integrity = after == root_hashes[head.position]
                    replay = []
                    if accepted:
                        restored, replay, _ = await self.traverse(head.tokens[:accepted], head.position, shape=True)
                        if restored.greedy_ids.tolist() != out.greedy_ids[:accepted].tolist():
                            raise AssertionError("rollback replay changed greedy logits")
                        if not np.array_equal(restored.nextn, out.nextn[:accepted]):
                            raise AssertionError("rollback replay changed target hidden state")
                    rb = dict(chunk=head.id, boundary_position=head.position,
                              restored_position=head.position+accepted, accepted_replay_rows=accepted,
                              commands=commands, replay=replay, state_hash_equal=integrity,
                              elapsed_ms=self.now()-rb_begin,
                              before_hashes=root_hashes.get(head.position),
                              after_hashes=after if integrity is not None else None)
                    head.rollback = rb
                    rollback_rows.append(rb)
                    inflight.clear()
                    epoch += 1
                    cursor = len(tokens)+len(generated)
                    draft_token, draft_hidden = next_token, next_hidden
                    rewind_draft = cursor
                    last_parent = None
                else:
                    for stage in self.stages:
                        stage.commit_position = len(tokens)+len(generated)
                if not head.rejected and not inflight:
                    # At W=1, seed the next MTP step from freshly verified target state.
                    # At W>1 existing provisional history is permitted to continue.
                    draft_token, draft_hidden = next_token, next_hidden
                    cursor = len(tokens)+len(generated)
                    rewind_draft = cursor if k else -1
                    last_parent = None
                if not k:
                    draft_token = next_token
            for c in inflight:
                c.invalidated = True
            for q in queues:
                await q.join()
        finally:
            for q in queues:
                await q.put(None)
            await asyncio.gather(*tasks)
        return dict(k=k, w=w, committed_tokens=generated, token_sha256=token_hash(generated),
                    prefill=prefill, prefill_ms=decode_start,
                    elapsed_ms=self.now()-decode_start, total_ms=self.now(),
                    chunks=[c.record() for c in chunks], events=self.events,
                    rollback_events=len(rollback_rows), rollback_failures=0,
                    state_corruption_events=sum(r["state_hash_equal"] is False for r in rollback_rows),
                    rollback_checks=rollback_rows, peak_inflight_chunks=peak,
                    launches_before_prior_verification_completed=overlap_launches,
                    commit_times_ms=commits, shaping=self.shaping, shaped_network=self.shaped_network, evidence_class="PHYSICAL_SINGLE_GPU_CORRECTNESS")
