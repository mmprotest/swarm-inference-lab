"""Exact GGUF tensor ranges and RPC-v6 cache names; never synthesizes weights."""
import argparse
import ctypes
import hashlib
from pathlib import Path
import sys

from .io import write_once, utc_now
from .preflight import MODEL_FILE, MODEL_REPO, MODEL_REVISION


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--first-layer",type=int,required=True)
    p.add_argument("--end-layer",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--minimum-bytes",type=int,default=10*1024*1024)
    a=p.parse_args()
    sys.path.insert(0,str(Path(".runtime/e026-llama.cpp/gguf-py").resolve()))
    from gguf import GGUFReader
    reader=GGUFReader(str(Path(".runtime/experiment-026/models")/MODEL_FILE))
    dll=ctypes.CDLL(str(Path(".runtime/experiment-026/cache_hash.dll").resolve()))
    dll.e026_fnv.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    dll.e026_fnv.restype=ctypes.c_uint64
    tensors=[]
    for tensor in reader.tensors:
        parts=tensor.name.split(".")
        if len(parts)<3 or parts[0]!="blk" or not a.first_layer<=int(parts[1])<a.end_layer:
            continue
        # Native cache only handles tensors larger than 10 MiB.
        if tensor.n_bytes<=a.minimum_bytes:continue
        data=memoryview(tensor.data)
        item={"tensor":tensor.name,"offset":tensor.data_offset,"size_bytes":tensor.n_bytes,
              "sha256":hashlib.sha256(data).hexdigest(),
              "rpc_fnv1a":f"{dll.e026_fnv(tensor.data.ctypes.data,tensor.n_bytes):016x}"}
        tensors.append(item)
    source=f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
    write_once(a.output,{"timestamp":utc_now(),"source":source,"model_file":MODEL_FILE,
                        "model_revision":MODEL_REVISION,"layer_range":[a.first_layer,a.end_layer],"minimum_bytes":a.minimum_bytes,
                        "tensor_bytes":sum(x["size_bytes"] for x in tensors),"tensors":tensors})
    print({"tensors":len(tensors),"bytes":sum(x["size_bytes"] for x in tensors),"output":str(a.output)})


if __name__=="__main__":main()
