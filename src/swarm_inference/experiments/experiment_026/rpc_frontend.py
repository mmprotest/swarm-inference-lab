"""Ordered async-copy adapter for the explicitly paired E026 device router."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import socketserver
import struct
import time
from .rpc_replica import exact,response,endpoint,MAX_FRAME_BYTES,RESPONSE_COMMANDS
from .rpc_router import copy_size
from .io import append_event,utc_now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--router",required=True)
    parser.add_argument("--async-copy",action="store_true")
    parser.add_argument("--cache-manifests",default="")
    parser.add_argument("--log",type=Path,required=True)
    args=parser.parse_args()
    remote=endpoint(args.router)
    manifests=[json.loads(Path(path).read_text()) for path in args.cache_manifests.split(",")] if args.cache_manifests else []
    hashes=[{int(t["rpc_fnv1a"],16) for t in m["tensors"]} for m in manifests]
    content=[{t["sha256"]:int(t["rpc_fnv1a"],16) for t in m["tensors"]} for m in manifests]
    def log(row):append_event(args.log,{"timestamp":utc_now(),**row})
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            backend=None
            buffers={}
            try:
                first=exact(self.request,9)
                backend=socket.create_connection(remote,timeout=30)
                backend.settimeout(180)
                for sock in (backend,self.request):sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                while True:
                    start=time.perf_counter()
                    cmd,size=struct.unpack("<BQ",first)
                    if size>MAX_FRAME_BYTES:raise ValueError("Unbounded frame")
                    payload=exact(self.request,size)
                    asynchronous=cmd==9 and args.async_copy and copy_size(payload) is not None
                    wire_cmd=18 if asynchronous else cmd
                    wire_payload=payload
                    cache=False
                    if manifests and cmd in {6,7}:
                        device=buffers.get(struct.unpack_from("<Q",payload,12)[0])
                        if device is not None and device<len(manifests):
                            if cmd==7 and struct.unpack_from("<Q",payload,304)[0] in hashes[device]:
                                cache=True
                            elif cmd==6:
                                key=content[device].get(hashlib.sha256(payload[304:]).hexdigest())
                                if key is not None:
                                    wire_payload=payload[:304]+struct.pack("<Q",key)
                                    cache=True
                            if cache:wire_cmd=19
                    backend.sendall(struct.pack("<BQ",wire_cmd,len(wire_payload))+wire_payload)
                    if asynchronous or (cache and cmd==7):reply=b"\x01"
                    elif cache:reply=None
                    else:reply=response(backend) if cmd in RESPONSE_COMMANDS else None
                    if cmd==0 and reply and struct.unpack_from("<Q",reply)[0]:
                        buffers[struct.unpack_from("<Q",reply)[0]]=struct.unpack_from("<I",payload)[0]
                    if cmd==4:buffers.pop(struct.unpack_from("<Q",payload)[0],None)
                    if reply is not None:self.request.sendall(struct.pack("<Q",len(reply))+reply)
                    log({"event":"FRONTEND_COMMAND","cmd":cmd,"async_copy":asynchronous,
                         "required_cache":cache,"request_bytes":size+9,"upstream_bytes":len(wire_payload)+9,
                         "elapsed_s":time.perf_counter()-start})
                    first=exact(self.request,9)
            except EOFError:log({"event":"CONNECTION_CLOSED"})
            except Exception as error:log({"event":"FRONTEND_FAILURE","type":type(error).__name__,"error":str(error)})
            finally:
                if backend:backend.close()
    class Server(socketserver.TCPServer):allow_reuse_address=True
    with Server(("127.0.0.1",args.port),Handler) as server:server.serve_forever()


if __name__=="__main__":main()
