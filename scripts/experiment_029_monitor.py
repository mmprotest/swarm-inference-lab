"""Sample whole-device VRAM every 200 ms while the E029 workload runs."""
from pathlib import Path
import json,subprocess,time
OUT=Path(__file__).resolve().parents[1]/'experiments/E029_LOCAL_COMMIT_ANCHORED_WAN_TREE'
p=subprocess.Popen(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','--loop-ms=200'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
peak=0;count=0;start=time.time();check=0
try:
    with (OUT/'vram_samples.jsonl').open('w') as f:
        for line in p.stdout:
            used=int(line.strip())*1024**2;peak=max(peak,used);count+=1
            f.write(json.dumps(dict(unix_seconds=time.time(),used_bytes=used))+'\n')
            if time.time()-check>3:
                check=time.time();f.flush()
                (OUT/'vram_monitor.json').write_text(json.dumps(dict(peak_vram_bytes=peak,samples=count,sample_period_ms=200,scope='whole GPU including other desktop applications',started_unix_seconds=start),indent=2)+'\n')
                progress=OUT/'progress.json'
                if (OUT/'monitor.stop').exists() or (progress.exists() and json.loads(progress.read_text()).get('completed_runs')==216):break
finally:
    p.terminate();p.wait(timeout=10)
print(json.dumps(dict(peak_vram_bytes=peak,samples=count)))
