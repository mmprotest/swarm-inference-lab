"""Pinned RPC-v6 warm replica with explicit pointer translation and failover.

Bind only to loopback. Remote endpoints must be authenticated SSH/TLS tunnels.
This replicates stage computation and all mutable state; it does not fake output.
"""

from __future__ import annotations

import argparse
import json
import socket
import socketserver
import struct
import threading
import time
from pathlib import Path

from .io import append_event, utc_now

TENSOR_SIZE = 296
RESPONSE_COMMANDS = {0, 1, 2, 3, 7, 8, 9, 11, 13, 14, 15}
MAX_FRAME_BYTES = 2 * 1024**3


def exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(size - len(chunks), 1024 * 1024))
        if not chunk:
            raise EOFError("RPC endpoint closed")
        chunks.extend(chunk)
    return bytes(chunks)


def response(sock: socket.socket) -> bytes:
    size = struct.unpack("<Q", exact(sock, 8))[0]
    if size > MAX_FRAME_BYTES:
        raise ValueError("Unbounded RPC response")
    return exact(sock, size)


def endpoint(value: str) -> tuple[str, int]:
    host, port = value.rsplit(":", 1)
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Use an authenticated tunnel terminating on loopback")
    return host, int(port)


class ReplicaConnection:
    def __init__(self, client: socket.socket, primary: tuple, standby: tuple, log, cache_hashes=frozenset()):
        self.client, self.log = client, log
        self.primary = socket.create_connection(primary, timeout=30)
        self.standby = socket.create_connection(standby, timeout=30)
        # Large disk-cache lookups can exceed five seconds during cold GPU load.
        # A killed RPC process closes this tunnel immediately; use a conservative
        # timeout here to avoid misclassifying slow startup as node failure.
        self.primary.settimeout(30)
        self.standby.settimeout(60)
        for stream in (client, self.primary, self.standby):
            stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buffers = {}
        self.failed = False
        self.index = 0
        self.tx = [0, 0]
        self.rx = [0, 0]
        self.cache_hashes=cache_hashes
        self.pending_cache=0

    def enqueue_verified_cache(self,payload,translated):
        standby_packet=struct.pack("<BQ",7,len(translated))+translated
        primary_packet=struct.pack("<BQ",7,len(payload))+payload
        self.standby.sendall(standby_packet)
        self.tx[1]+=len(standby_packet)
        if not self.failed:
            try:
                self.primary.sendall(primary_packet)
                self.tx[0]+=len(primary_packet)
            except (OSError,EOFError) as error:self.fail(error)
        self.pending_cache+=1
        self.log({"event":"REPLICA_CACHE_LOOKUP_ENQUEUED","command_index":self.index,
                  "pending":self.pending_cache,"primary_failed":self.failed})

    def drain_cache(self):
        count=self.pending_cache
        if not count:return
        start=time.perf_counter()
        for _ in range(count):
            primary_reply=None
            if not self.failed:
                try:
                    primary_reply=response(self.primary)
                    self.rx[0]+=8+len(primary_reply)
                except (OSError,EOFError) as error:self.fail(error)
            standby_reply=response(self.standby)
            self.rx[1]+=8+len(standby_reply)
            if standby_reply!=b"\x01" or (primary_reply is not None and primary_reply!=b"\x01"):
                raise RuntimeError("Required verified replica cache tensor is unavailable")
        self.pending_cache=0
        self.log({"event":"REPLICA_CACHE_BATCH_DRAIN","count":count,
                  "seconds":time.perf_counter()-start,"primary_failed":self.failed})

    def fail(self, error: Exception):
        if not self.failed:
            self.failed = True
            self.log({"event": "PRIMARY_FAILED_STANDBY_SELECTED", "command_index": self.index,
                      "error": type(error).__name__, "detail": str(error), "full_prompt_replay_tokens": 0})
            self.primary.close()

    def rewrite_tensor(self, payload: bytearray, offset: int):
        if offset < 0 or offset + TENSOR_SIZE > len(payload):
            raise ValueError("Invalid tensor extent")
        buf = struct.unpack_from("<Q", payload, offset + 12)[0]
        data = struct.unpack_from("<Q", payload, offset + 220)[0]
        if not buf:
            return
        mapping = self.buffers[buf]
        struct.pack_into("<Q", payload, offset + 12, mapping["standby"])
        if data:
            if mapping["primary_base"] is None or mapping["standby_base"] is None:
                raise ValueError("Tensor used before buffer base resolution")
            delta = data - mapping["primary_base"]
            if delta < 0 or delta > mapping["size"]:
                raise ValueError("Tensor data outside registered buffer")
            struct.pack_into("<Q", payload, offset + 220, mapping["standby_base"] + delta)

    def translate(self, cmd: int, data: bytes) -> bytes:
        value = bytearray(data)
        if cmd in {3, 4, 5}:
            buf = struct.unpack_from("<Q", value)[0]
            struct.pack_into("<Q", value, 0, self.buffers[buf]["standby"])
        elif cmd in {6, 7, 8, 12, 17}:
            self.rewrite_tensor(value, 0)
        elif cmd == 9:
            self.rewrite_tensor(value, 0)
            self.rewrite_tensor(value, TENSOR_SIZE)
        elif cmd == 13:
            self.rewrite_tensor(value, 4)
        elif cmd == 10:
            n_nodes = struct.unpack_from("<I", value, 4)[0]
            offset = 8 + 8 * n_nodes
            if offset + 4 > len(value):
                raise ValueError("Invalid graph extent")
            n_tensors = struct.unpack_from("<I", value, offset)[0]
            offset += 4
            if offset + n_tensors * TENSOR_SIZE != len(value):
                raise ValueError("RPC tensor ABI mismatch")
            for i in range(n_tensors):
                self.rewrite_tensor(value, offset + i * TENSOR_SIZE)
        return bytes(value)

    def transact(self, cmd: int, payload: bytes) -> bytes | None:
        start = time.perf_counter()
        translated = self.translate(cmd, payload)
        cache_key=struct.unpack_from("<Q",payload,304)[0] if cmd==7 and len(payload)>=312 else None
        if cache_key in self.cache_hashes:
            self.enqueue_verified_cache(payload,translated)
            self.index+=1
            return b"\x01"
        self.drain_cache()
        standby_packet = struct.pack("<BQ", cmd, len(translated)) + translated
        primary_packet = struct.pack("<BQ", cmd, len(payload)) + payload
        self.standby.sendall(standby_packet)
        self.tx[1] += len(standby_packet)
        if not self.failed:
            try:
                self.primary.sendall(primary_packet)
                self.tx[0] += len(primary_packet)
            except (OSError, EOFError) as error:
                self.fail(error)
        a = b = None
        if cmd in RESPONSE_COMMANDS:
            if not self.failed:
                try:
                    a = response(self.primary)
                    self.rx[0] += 8 + len(a)
                except (OSError, EOFError) as error:
                    self.fail(error)
            b = response(self.standby)
            self.rx[1] += 8 + len(b)
            if a is None:
                a = b
            if cmd == 14:
                if a != b or a[:1] != b"\x06":
                    raise ValueError("Replica protocol versions do not match RPC v6")
            elif cmd == 0:
                p, p_size = struct.unpack("<QQ", a)
                s, s_size = struct.unpack("<QQ", b)
                if not p or not s or p_size != s_size:
                    raise RuntimeError("Replica allocation failed or differs")
                self.buffers[p] = {"standby": s, "size": p_size, "primary_base": None, "standby_base": None}
            elif cmd == 3:
                buf = struct.unpack("<Q", payload)[0]
                mapping = self.buffers[buf]
                mapping["standby_base"] = struct.unpack("<Q", b)[0]
                if not self.failed:
                    mapping["primary_base"] = struct.unpack("<Q", a)[0]
                elif mapping["primary_base"] is None:
                    mapping["primary_base"] = mapping["standby_base"]
                a = struct.pack("<Q", mapping["primary_base"])
            elif cmd == 7:
                a = bytes([int(bool(a[0]) and bool(b[0]))])
            elif cmd == 11:
                af, at = struct.unpack("<QQ", a)
                bf, bt = struct.unpack("<QQ", b)
                a = struct.pack("<QQ", min(af, bf), min(at, bt))
            elif cmd in {1, 2, 9, 13, 15} and a != b:
                raise RuntimeError(f"Replica capability/operation disagreement for command {cmd}")
            elif cmd == 8 and not self.failed:
                self.log({"event": "REPLICA_OUTPUT_COMPARISON", "command_index": self.index,
                          "bytes": len(a), "byte_equal": a == b})
        if cmd == 4:
            del self.buffers[struct.unpack("<Q", payload)[0]]
        self.log({"event": "RPC_REPLICATED", "command_index": self.index, "cmd": cmd,
                  "payload_bytes": len(payload), "response_bytes": len(a) if a is not None else 0,
                  "seconds": time.perf_counter() - start, "primary_failed": self.failed,
                  "tx_total": self.tx, "rx_total": self.rx})
        self.index += 1
        return a

    def run(self):
        try:
            while True:
                header = exact(self.client, 9)
                cmd, size = struct.unpack("<BQ", header)
                if cmd > 17 or size > MAX_FRAME_BYTES:
                    raise ValueError("Unexpected RPC command or size")
                payload = exact(self.client, size)
                result = self.transact(cmd, payload)
                if result is not None:
                    self.client.sendall(struct.pack("<Q", len(result)) + result)
        finally:
            try:self.drain_cache()
            finally:
                self.primary.close()
                self.standby.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--standby", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--cache-manifest",type=Path)
    args = parser.parse_args()
    primary, standby = endpoint(args.primary), endpoint(args.standby)
    manifest=json.loads(args.cache_manifest.read_text()) if args.cache_manifest else {"tensors":[]}
    cache_hashes=frozenset(int(row["rpc_fnv1a"],16) for row in manifest["tensors"])
    log_lock = threading.Lock()
    def log(event):
        with log_lock:
            append_event(args.log, {"timestamp": utc_now(), "monotonic_ns": time.perf_counter_ns(), **event})
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                ReplicaConnection(self.request, primary, standby, log,cache_hashes).run()
            except (OSError, EOFError):
                log({"event": "CONNECTION_CLOSED"})
            except Exception as error:
                log({"event": "REPLICA_ERROR", "type": type(error).__name__, "detail": str(error)})
                raise
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as server:
        log({"event": "REPLICA_PROXY_READY", "listen_port": args.port,
             "primary": args.primary, "standby": args.standby, "tensor_abi_bytes": TENSOR_SIZE,
             "primary_socket_timeout_s":30,"required_cache_hashes":len(cache_hashes)})
        server.serve_forever()


if __name__ == "__main__":
    main()
