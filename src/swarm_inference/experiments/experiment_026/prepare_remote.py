"""Prepare a scoped SSH identity and reproducible source bundle, never rent."""
from pathlib import Path
import json
import subprocess
import tarfile
import io
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .io import write_once, file_digest, utc_now
from .vast import client


def main():
    keys = Path(".keys")
    keys.mkdir(exist_ok=True)
    private = keys / "e026_ed25519"
    public = keys / "e026_ed25519.pub"
    if not private.exists():
        key = Ed25519PrivateKey.generate()
        with private.open("xb") as stream:
            stream.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                                           serialization.NoEncryption()))
        with public.open("xb") as stream:
            stream.write(key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
                         + b" e026-temporary\n")
    # Only the public key is passed to the service. Private key stays local.
    receipt = Path("artifacts/experiment-026/preflight/ssh-registration.json")
    if not receipt.exists():
        result = client()._run(["create", "ssh-key", public.read_text().strip(), "--raw"], timeout_seconds=60)
        write_once(receipt, {"timestamp": utc_now(), "public_key_sha256": file_digest(public),
                             "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("SSH public-key registration failed")
    root = Path(".runtime/e026-llama.cpp")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    patch = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"])
    source = subprocess.check_output(["git", "-C", str(root), "archive", "--format=tar", "HEAD"])
    bundle = Path(".runtime/experiment-026/remote-source-001.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(source)) as archive_in, tarfile.open(bundle, "x:gz") as archive_out:
        for member in archive_in:
            archive_out.addfile(member, archive_in.extractfile(member) if member.isfile() else None)
        info = tarfile.TarInfo("e026-instrumentation.diff")
        info.size = len(patch)
        archive_out.addfile(info, io.BytesIO(patch))
    write_once(Path("artifacts/experiment-026/preflight/remote-bundle-001.json"),
               {"timestamp": utc_now(), "llama_commit": commit, "source_diff_sha256": hashlib.sha256(patch).hexdigest(),
                "bundle_sha256": file_digest(bundle), "bundle_bytes": bundle.stat().st_size})
    print(json.dumps({"bundle": str(bundle), "bytes": bundle.stat().st_size, "ssh_identity_ready": True}))


if __name__ == "__main__":
    main()
