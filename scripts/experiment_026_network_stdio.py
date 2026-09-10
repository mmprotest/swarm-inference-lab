"""Persistent application RTT/throughput probe carried by authenticated SSH."""
import os
import struct
import sys

source = sys.stdin.buffer
sink = sys.stdout.buffer


def exact(n):
    chunks = bytearray()
    while len(chunks) < n:
        data = source.read(min(n-len(chunks), 1024*1024))
        if not data:
            raise EOFError()
        chunks.extend(data)
    return bytes(chunks)


sink.write(b"E026READY")
sink.flush()
block = os.urandom(1024*1024)
while True:
    try:
        op = exact(1)
        length = struct.unpack("<Q", exact(8))[0]
        if length > 256*1024*1024:
            raise ValueError("Bounded test exceeded")
        if op in (b"P", b"U"):
            remaining = length
            while remaining:
                n = min(remaining, len(block))
                exact(n)
                remaining -= n
            sink.write(b"A")
        elif op == b"D":
            for offset in range(0, length, len(block)):
                sink.write(block[:min(length-offset,len(block))])
        else:
            break
        sink.flush()
    except EOFError:
        break
