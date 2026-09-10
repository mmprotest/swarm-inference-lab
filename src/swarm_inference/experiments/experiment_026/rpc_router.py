"""RPC-v6 device router: each virtual device maps to one concrete worker.

No aggregate timing is assumed. Cross-worker copies physically read the source
and write the destination over authenticated loopback-terminated SSH tunnels.
Command 18 is an ordered, no-response copy understood only by this router and
its explicitly configured local front end; native workers never receive it.
"""
from __future__ import annotations
import argparse
import json
import socket
import socketserver
import struct
import time
from pathlib import Path

from .rpc_replica import TENSOR_SIZE, RESPONSE_COMMANDS, MAX_FRAME_BYTES, exact, response, endpoint
from .io import append_event, utc_now


def copy_size(payload):
    if len(payload)!=2*TENSOR_SIZE:return None
    source,dest=payload[:TENSOR_SIZE],payload[TENSOR_SIZE:]
    t=struct.unpack_from("<I",source,8)[0]
    width={0:4,1:2}.get(t)
    if width is None or t!=struct.unpack_from("<I",dest,8)[0]:return None
    ne=struct.unpack_from("<4I",source,20)
    if ne!=struct.unpack_from("<4I",dest,20):return None
    size=width
    for i,extent in enumerate(ne):
        if any(struct.unpack_from("<I",tensor,36+4*i)[0]!=size for tensor in (source,dest)):
            return None
        size*=extent
    return size


class RoutedConnection:
    def __init__(self, peers, log):
        self.peers=[socket.create_connection(peer,timeout=30) for peer in peers]
        for peer in self.peers:
            peer.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            peer.settimeout(120)
        self.buffers={}
        self.next_id=0xE02600000000
        self.log=log
        self.tx=[0]*len(peers)
        self.rx=[0]*len(peers)
        self.pending_cache=[0]*len(peers)

    def drain_cache(self,worker):
        count=self.pending_cache[worker]
        if not count:return
        start=time.perf_counter()
        for _ in range(count):
            data=response(self.peers[worker])
            self.rx[worker]+=len(data)+8
            if data!=b"\x01":raise RuntimeError("Required verified cached tensor is unavailable")
        self.pending_cache[worker]=0
        self.log({"event":"CACHE_BATCH_DRAIN","worker":worker,"count":count,
                  "response_bytes":count*9,"elapsed_s":time.perf_counter()-start})

    def defer_cache(self,worker,payload):
        packet=struct.pack("<BQ",7,len(payload))+payload
        self.peers[worker].sendall(packet)
        self.tx[worker]+=len(packet)
        self.pending_cache[worker]+=1
        self.log({"event":"CACHE_LOOKUP_ENQUEUED","worker":worker,"pending":self.pending_cache[worker],
                  "request_bytes":len(packet)})

    def send(self,worker,cmd,payload,reply=None):
        self.drain_cache(worker)
        packet=struct.pack("<BQ",cmd,len(payload))+payload
        start=time.perf_counter()
        self.peers[worker].sendall(packet)
        self.tx[worker]+=len(packet)
        data=response(self.peers[worker]) if (cmd in RESPONSE_COMMANDS if reply is None else reply) else None
        if data is not None:self.rx[worker]+=len(data)+8
        self.log({"event":"WORKER_OPERATION","worker":worker,"cmd":cmd,"request_bytes":len(packet),
                  "response_bytes":0 if data is None else len(data)+8,"elapsed_s":time.perf_counter()-start})
        return data

    def tensor_worker(self,data,offset=0):
        handle=struct.unpack_from("<Q",data,offset+12)[0]
        return self.buffers[handle]["worker"]

    def rewrite(self,data,offset,worker):
        if offset+TENSOR_SIZE>len(data):raise ValueError("Truncated tensor")
        handle=struct.unpack_from("<Q",data,offset+12)[0]
        if not handle:return
        owner=self.buffers[handle]
        if owner["worker"]!=worker:raise ValueError("Graph contains foreign-owned buffer")
        struct.pack_into("<Q",data,offset+12,owner["real"])

    def transact(self,cmd,payload):
        data=bytearray(payload)
        if cmd==14:
            replies=[self.send(i,cmd,payload) for i in range(len(self.peers))]
            if any(item!=replies[0] for item in replies):raise ValueError("Worker protocol capabilities differ")
            if not replies[0] or replies[0][0]!=6:raise ValueError("RPC-v6 required")
            return replies[0]
        if cmd==15:return struct.pack("<I",len(self.peers))
        if cmd in {0,1,2,11,13,10,16}:
            worker=struct.unpack_from("<I",data)[0]
            if worker>=len(self.peers):raise ValueError("Unknown device")
            struct.pack_into("<I",data,0,0)
            if cmd==13:
                for offset in range(4,len(data),TENSOR_SIZE):self.rewrite(data,offset,worker)
            elif cmd==10:
                n_nodes=struct.unpack_from("<I",data,4)[0]
                start=8+8*n_nodes
                count=struct.unpack_from("<I",data,start)[0]
                start+=4
                if start+count*TENSOR_SIZE!=len(data):raise ValueError("Invalid graph extent")
                for offset in range(start,len(data),TENSOR_SIZE):self.rewrite(data,offset,worker)
            result=self.send(worker,cmd,data)
            if cmd==0:
                real,size=struct.unpack("<QQ",result)
                if real==0:return result
                virtual=self.next_id
                self.next_id+=0x100
                self.buffers[virtual]={"worker":worker,"real":real,"size":size}
                return struct.pack("<QQ",virtual,size)
            return result
        if cmd in {3,4,5}:
            handle=struct.unpack_from("<Q",data)[0]
            owner=self.buffers[handle]
            struct.pack_into("<Q",data,0,owner["real"])
            result=self.send(owner["worker"],cmd,data)
            if cmd==4:del self.buffers[handle]
            return result
        if cmd in {6,7,8,12,17,19}:
            worker=self.tensor_worker(data)
            self.rewrite(data,0,worker)
            if cmd==19:
                self.defer_cache(worker,data)
                self.log({"event":"REQUIRED_CACHE_HIT","worker":worker})
                return None
            return self.send(worker,cmd,data)
        if cmd in {9,18}:
            src,dst=self.tensor_worker(data),self.tensor_worker(data,TENSOR_SIZE)
            size=copy_size(data)
            if cmd==18 and size is None:raise ValueError("Unsupported asynchronous copy")
            self.rewrite(data,0,src)
            self.rewrite(data,TENSOR_SIZE,dst)
            if src==dst:
                result=self.send(src,9,data)
                if cmd==18 and result!=b"\x01":raise RuntimeError("Enqueued copy failed")
                return result if cmd==9 else None
            if size is None:return b"\x00"
            values=self.send(src,8,data[:TENSOR_SIZE]+struct.pack("<QQ",0,size))
            if len(values)!=size:raise ValueError("Incomplete peer activation")
            self.send(dst,6,data[TENSOR_SIZE:]+struct.pack("<Q",0)+values)
            self.log({"event":"PHYSICAL_PEER_COPY","source_worker":src,"destination_worker":dst,
                      "bytes":size,"asynchronous_client_ack":cmd==18})
            return b"\x01" if cmd==9 else None
        raise ValueError(f"Unsupported RPC command {cmd}")

    def close(self):
        for worker,peer in enumerate(self.peers):
            try:self.drain_cache(worker)
            finally:peer.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--workers",required=True)
    parser.add_argument("--log",type=Path,required=True)
    args=parser.parse_args()
    peers=[endpoint(value) for value in args.workers.split(",")]
    def log(row):append_event(args.log,{"timestamp":utc_now(),**row})
    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            connection=None
            try:
                self.request.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                # An empty TCP readiness check must not claim worker connections.
                first=exact(self.request,9)
                connection=RoutedConnection(peers,log)
                while True:
                    cmd,size=struct.unpack("<BQ",first)
                    if size>MAX_FRAME_BYTES:raise ValueError("Unbounded RPC frame")
                    result=connection.transact(cmd,exact(self.request,size))
                    if result is not None:self.request.sendall(struct.pack("<Q",len(result))+result)
                    first=exact(self.request,9)
            except EOFError:log({"event":"CONNECTION_CLOSED"})
            except Exception as error:log({"event":"ROUTER_FAILURE","type":type(error).__name__,"error":str(error)})
            finally:
                if connection:connection.close()
    class Server(socketserver.TCPServer):allow_reuse_address=True
    with Server(("127.0.0.1",args.port),Handler) as server:server.serve_forever()


if __name__=="__main__":main()
