"""Scope a remote peer identity to the two owned experiment containers."""
import hashlib
import json
from pathlib import Path
import shlex

from .io import write_once, utc_now
from .remote import nodes, ssh, upload


def main():
    inventory={n["role"]:n for n in nodes()}
    a,b=inventory["a"],inventory["b"]
    ssh(a,"test -f /workspace/e026/peer_ed25519.pub || ssh-keygen -q -t ed25519 -N '' -f /workspace/e026/peer_ed25519")
    public=ssh(a,"cat /workspace/e026/peer_ed25519.pub").stdout.decode().strip()
    if not public.startswith("ssh-ed25519 "):raise ValueError("Unexpected peer key")
    code=("from pathlib import Path; p=Path('/root/.ssh/authorized_keys'); "
          f"key={public!r}; current=p.read_text(); "
          "p.open('a').write(key+'\\n') if key not in current.splitlines() else None; p.chmod(0o600)")
    ssh(b,"python3 -c "+shlex.quote(code))
    prefix=f"[{b['host']}]:{b['port']} "
    lines=[line for line in Path(".keys/e026_known_hosts").read_text().splitlines() if line.startswith(prefix)]
    if len(lines)!=1:raise ValueError("Cannot uniquely pin the already-observed peer host key")
    path=Path(".runtime/experiment-026/peer_known_hosts")
    with path.open("x") as stream:stream.write(lines[0]+"\n")
    upload(a,path,"/workspace/e026/peer_known_hosts")
    upload(a,"scripts/experiment_026_network_peer.py","/workspace/e026/network-peer.py")
    result=ssh(a,f"python3 -u /workspace/e026/network-peer.py {shlex.quote(b['host'])} {b['port']}",timeout=180)
    row={"timestamp":utc_now(),"evidence_class":"PHYSICAL","a":a["id"],"b":b["id"],
         "peer_public_key_sha256":hashlib.sha256(public.encode()).hexdigest(),"private_key_left_worker_a":False,
         "network":json.loads(result.stdout)}
    write_once(Path("artifacts/experiment-026/network/direct-wan-preflight-001/peer-a-b.json"),row)
    print(json.dumps(row),flush=True)


if __name__=="__main__":main()
