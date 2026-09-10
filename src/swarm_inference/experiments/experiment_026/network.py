"""Measure the actual persistent SSH application transport, without shaping."""
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time

from .io import write_once, utc_now
from .remote import nodes, ssh_args, upload
from .benchmark import percentile


def exact(stream, n):
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(min(n-len(data),1024*1024))
        if not chunk:
            raise EOFError("Network probe ended")
        data.extend(chunk)
    return bytes(data)


def measure(node, run_id):
    upload(node, "scripts/experiment_026_network_stdio.py", "/tmp/e026-network.py")
    folder = Path("artifacts/experiment-026/network") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    stderr = (folder / f"{node['id']}.stderr.log").open("wb")
    command = ["ssh", *ssh_args(node), "-T", "-p", str(node["port"]), "root@"+node["host"], "python3 -u /tmp/e026-network.py"]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        assert exact(proc.stdout,9) == b"E026READY"
        rtt=[]
        for _ in range(40):
            start=time.perf_counter()
            proc.stdin.write(struct.pack("<cQ",b"P",1024)+b"x"*1024)
            proc.stdin.flush()
            assert exact(proc.stdout,1)==b"A"
            rtt.append(time.perf_counter()-start)
        throughput=[]
        block=os.urandom(1024*1024)
        for op in (b"U",b"D"):
            size=16*1024*1024
            start=time.perf_counter()
            proc.stdin.write(struct.pack("<cQ",op,size))
            if op==b"U":
                for _ in range(16): proc.stdin.write(block)
                proc.stdin.flush()
                assert exact(proc.stdout,1)==b"A"
            else:
                proc.stdin.flush()
                exact(proc.stdout,size)
            elapsed=time.perf_counter()-start
            throughput.append({"direction":"LOCAL_TO_REMOTE" if op==b"U" else "REMOTE_TO_LOCAL",
                               "bytes":size,"elapsed_s":elapsed,"mbps":size*8/elapsed/1e6})
        row={"timestamp":utc_now(),"evidence_class":"PHYSICAL", "source":"local-RTX5090",
             "destination":node["id"],"region":node["geolocation"],"transport":"PERSISTENT_DIRECT_SSH_CHANNEL",
             "artificial_delay":False,"rtt_samples_s":rtt,"median_rtt_s":statistics.median(rtt),
             "p95_rtt_s":percentile(rtt,.95),"jitter_stddev_s":statistics.pstdev(rtt),"throughput":throughput,
             "command":command,"ssh_handshake_excluded":True}
        write_once(folder/f"{node['id']}.json",row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("command","rtt_samples_s")}),flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        stderr.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--role")
    args=parser.parse_args()
    for node in nodes():
        if not args.role or args.role==node["role"]:
            measure(node,args.run_id)


if __name__=="__main__":main()
