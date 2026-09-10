"""Runs on worker A: measure A-to-B, excluding the coordinator's network leg."""
import json
import os
import statistics
import struct
import subprocess
import sys
import time

host,port=sys.argv[1:3]
command=["ssh","-i","/workspace/e026/peer_ed25519","-o","BatchMode=yes","-o","IdentitiesOnly=yes",
         "-o","StrictHostKeyChecking=yes","-o","UserKnownHostsFile=/workspace/e026/peer_known_hosts",
         "-T","-p",port,"root@"+host,"python3 -u /tmp/e026-network.py"]
proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE)


def exact(n):
    chunks=bytearray()
    while len(chunks)<n:
        block=proc.stdout.read(min(n-len(chunks),1024*1024))
        if not block:raise EOFError("Peer disconnected")
        chunks.extend(block)
    return chunks


try:
    assert exact(9)==b"E026READY"
    samples=[]
    for _ in range(40):
        start=time.perf_counter()
        proc.stdin.write(struct.pack("<cQ",b"P",1024)+b"x"*1024)
        proc.stdin.flush()
        assert exact(1)==b"A"
        samples.append(time.perf_counter()-start)
    rows=[]
    block=os.urandom(1024*1024)
    for op in (b"U",b"D"):
        size=16*1024*1024
        start=time.perf_counter()
        proc.stdin.write(struct.pack("<cQ",op,size))
        if op==b"U":
            for _ in range(16):proc.stdin.write(block)
            proc.stdin.flush()
            assert exact(1)==b"A"
        else:
            proc.stdin.flush()
            exact(size)
        elapsed=time.perf_counter()-start
        rows.append({"direction":"A_TO_B" if op==b"U" else "B_TO_A","bytes":size,"elapsed_s":elapsed,"mbps":size*8/elapsed/1e6})
    print(json.dumps({"source_clock":"WORKER_A_MONOTONIC","rtt_samples_s":samples,
                      "median_rtt_s":statistics.median(samples),"jitter_stddev_s":statistics.pstdev(samples),
                      "throughput":rows,"transport":"DIRECT_PERSISTENT_SSH_A_TO_B","coordinator_relay":False}),flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=10)
