"""Download only selected GGUF ranges into the RPC cache, with SHA256 checks."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

manifest=json.loads(Path(sys.argv[1]).read_text())
cache=Path(sys.argv[2])
cache.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()


def acquire(tensor):
    path=cache/tensor["rpc_fnv1a"]
    start=time.perf_counter()
    if path.exists() and path.stat().st_size==tensor["size_bytes"]:
        with path.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()==tensor["sha256"]:
                return {"tensor":tensor["tensor"],"status":"CACHE_HIT_VERIFIED","network_bytes":0}
    error=None
    for attempt in range(3):
        try:
            first=tensor["offset"]
            last=first+tensor["size_bytes"]-1
            req=urllib.request.Request(manifest["source"]+f"?e026_range={first}-{last}",headers={"Range":f"bytes={first}-{last}"})
            temporary=path.with_suffix(".partial")
            total=0
            digest=hashlib.sha256()
            with urllib.request.urlopen(req,timeout=90) as response, temporary.open("wb") as output:
                if response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {first}-{last}/"):
                    raise ValueError("Server did not honor exact tensor range")
                while True:
                    block=response.read(4*1024*1024)
                    if not block:break
                    total+=len(block)
                    digest.update(block)
                    output.write(block)
            if total!=tensor["size_bytes"] or digest.hexdigest()!=tensor["sha256"]:
                raise ValueError("Shard integrity failure")
            os.replace(temporary,path)
            return {"tensor":tensor["tensor"],"status":"ACQUIRED_VERIFIED","network_bytes":total,
                    "elapsed_s":time.perf_counter()-start,"sha256":digest.hexdigest(),"cache_file":str(path)}
        except Exception as exc:
            error=repr(exc)
    raise RuntimeError(f"{tensor['tensor']}: {error}")


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    rows=[]
    for row in pool.map(acquire,manifest["tensors"]):
        rows.append(row)
        print(json.dumps(row),flush=True)
print(json.dumps({"event":"SHARD_READY","elapsed_s":time.perf_counter()-started,
                  "network_bytes":sum(x["network_bytes"] for x in rows),"tensor_count":len(rows)}),flush=True)
