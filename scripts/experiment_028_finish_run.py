"""Continue E028's authorized local run after the active collector exits."""
import argparse
import ctypes
import json
import subprocess
import sys

from swarm_inference.experiments.experiment_028.local import ROOT, OUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--collector-pid",type=int,required=True)
    args=parser.parse_args()
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong]
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,args.collector_pid)
    if handle:
        try:
            while kernel.WaitForSingleObject(handle,5000)==258:
                pass
        finally:
            kernel.CloseHandle(handle)
    result=json.loads((OUT/"correctness_results.json").read_text())
    if not result["complete"]:
        raise RuntimeError("collector exited before completing E028; inspect and resume collection")
    for phase in ["stress","validate","sweep"]:
        subprocess.run([sys.executable,"scripts/experiment_028_run.py",phase],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"scripts/experiment_028_audit.py"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"scripts/experiment_028_run.py","report"],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
