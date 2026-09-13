"""Actual local delays for independent simulator validation."""
import time, threading
from concurrent.futures import ThreadPoolExecutor

class LocalDelay:
    def __init__(self,network):
        assert network['jitter_ms']==0,'validation uses deterministic zero jitter'
        self.network=network;self.gpu=threading.Lock();self.pool=ThreadPoolExecutor(max_workers=3)
    def transfer(self,bytes):
        bw=self.network['bandwidth_mbps'];duration=self.network['rtt_ms']/2+(bytes*8/(bw*1000) if bw else 0)
        time.sleep(duration/1000)
    def commit(self,clients,node):
        def task(c):
            self.transfer(56)
            with self.gpu:response=c.call(12,arg=node)
            self.transfer(response[2]['response_bytes'])
            return response
        return list(self.pool.map(task,clients))
    def close(self):self.pool.shutdown()
