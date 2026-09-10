import struct
import socket
import threading
from swarm_inference.experiments.experiment_026.rpc_router import copy_size,RoutedConnection
from swarm_inference.experiments.experiment_026.rpc_replica import TENSOR_SIZE,ReplicaConnection


def tensor(ne=(5120,4,1,1),kind=0):
    data=bytearray(TENSOR_SIZE)
    struct.pack_into("<I",data,8,kind)
    struct.pack_into("<4I",data,20,*ne)
    size=4 if kind==0 else 2
    nb=[]
    for extent in ne:
        nb.append(size)
        size*=extent
    struct.pack_into("<4I",data,36,*nb)
    return data


def test_copy_extent_and_reject_strides():
    a=tensor()
    assert copy_size(a+a)==81920
    b=tensor()
    struct.pack_into("<I",b,40,40960)
    assert copy_size(a+b) is None
    assert copy_size(a+tensor(kind=1)) is None


def test_pointer_ownership_is_checked():
    import pytest
    obj=object.__new__(RoutedConnection)
    obj.buffers={42:{"worker":0,"real":111,"size":81920}}
    a=tensor()
    struct.pack_into("<Q",a,12,42)
    with pytest.raises(ValueError,match="foreign-owned"):
        obj.rewrite(a,0,1)
    obj.rewrite(a,0,0)
    assert struct.unpack_from("<Q",a,12)[0]==111


def test_cache_lookups_pipeline_then_fail_closed_drain():
    client,server=socket.socketpair()
    events=[]
    obj=object.__new__(RoutedConnection)
    obj.peers=[client]
    obj.tx=[0]
    obj.rx=[0]
    obj.pending_cache=[0]
    obj.log=events.append
    def worker():
        for _ in range(2):
            header=server.recv(9)
            cmd,size=struct.unpack("<BQ",header)
            assert cmd==7
            assert server.recv(size)==b"lookup"
        server.sendall(struct.pack("<Q",1)+b"\x01")
        server.sendall(struct.pack("<Q",1)+b"\x01")
    thread=threading.Thread(target=worker)
    thread.start()
    obj.defer_cache(0,b"lookup")
    obj.defer_cache(0,b"lookup")
    assert obj.pending_cache==[2]
    obj.drain_cache(0)
    assert obj.pending_cache==[0]
    assert events[-1]["event"]=="CACHE_BATCH_DRAIN"
    assert events[-1]["count"]==2
    thread.join(timeout=2)
    client.close()
    server.close()


def test_replica_cache_lookup_is_deferred_and_validated_on_both_workers():
    primary,primary_server=socket.socketpair()
    standby,standby_server=socket.socketpair()
    events=[]
    obj=object.__new__(ReplicaConnection)
    obj.client=None
    obj.primary=primary
    obj.standby=standby
    obj.log=events.append
    obj.buffers={}
    obj.failed=False
    obj.index=0
    obj.tx=[0,0]
    obj.rx=[0,0]
    obj.cache_hashes=frozenset({42})
    obj.pending_cache=0
    def worker(stream):
        header=stream.recv(9)
        cmd,size=struct.unpack("<BQ",header)
        assert cmd==7
        assert len(stream.recv(size))==312
        stream.sendall(struct.pack("<Q",1)+b"\x01")
    threads=[threading.Thread(target=worker,args=(stream,)) for stream in (primary_server,standby_server)]
    for thread in threads:thread.start()
    payload=bytearray(312)
    struct.pack_into("<Q",payload,304,42)
    assert obj.transact(7,bytes(payload))==b"\x01"
    assert obj.pending_cache==1
    obj.drain_cache()
    assert obj.pending_cache==0
    assert events[-1]["event"]=="REPLICA_CACHE_BATCH_DRAIN"
    for thread in threads:thread.join(timeout=2)
    for stream in (primary,primary_server,standby,standby_server):stream.close()
