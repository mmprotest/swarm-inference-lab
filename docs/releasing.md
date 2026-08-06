# Releasing Swarm Inference

The Windows release pipeline is tag-driven, immutable, hash-verified, and draft-first. It builds
the existing Python wheel into a native installer; it does not replace the canonical runtime.

## Version and tag policy

`pyproject.toml` is the package version source. Runtime `__version__` comes from installed package
metadata. Supported mappings are:

| Package version | Git tag | GitHub channel |
|---|---|---|
| `0.1.0rc1` | `v0.1.0-rc.1` | prerelease |
| `0.1.0rc2` | `v0.1.0-rc.2` | prerelease |
| `0.1.0rc3` | `v0.1.0-rc.3` | prerelease |
| `0.1.0rc4` | `v0.1.0-rc.4` | prerelease |
| `0.1.0rc5` | `v0.1.0-rc.5` | prerelease |
| `0.1.0rc6` | `v0.1.0-rc.6` | prerelease |
| `0.1.0rc7` | `v0.1.0-rc.7` | prerelease |
| `0.1.0` | `v0.1.0` | stable |

Never overwrite, move, or force-push a release tag. If a version or tag already exists, increment
the release-candidate number. `scripts/verify_release_version.py` rejects a mismatched version,
dirty tree, wrong commit, absent tag, or a tag that does not resolve to the checkout.

## Pinned build inputs

`installer/windows/toolchain.json` pins the managed Python version, uv release/archive/executable
hashes, .NET LTS SDK archive/executable hashes, and Inno Setup installer/compiler hashes and
publisher identity. `scripts/prepare_windows_toolchain.py` verifies every download before any
downloaded executable runs. Runtime profiles are exported from `uv.lock` with the pinned uv and
contain exact versions and artifact hashes.

GitHub Actions uses exact Python 3.11.9 only to orchestrate the build because it is the newest
Python 3.11 artifact published for the Windows runner. The installed product runtime remains the
separately pinned uv-managed Python 3.11.15 recorded in the release manifest.

## Local release-candidate build

From a Windows x86-64 checkout:

```powershell
uv sync --locked --python 3.11 --extra cpu --extra dev
uv run --python 3.11 ruff format --check src tests scripts
uv run --python 3.11 ruff check src tests scripts
uv run --python 3.11 mypy
uv run --python 3.11 pytest tests/unit -q
uv run --python 3.11 pytest tests/installer -q
uv run --python 3.11 pytest tests/integration tests/failure -m "not gpu" -q
uv run --python 3.11 python scripts/prepare_windows_toolchain.py --tools all
uv run --python 3.11 python scripts/build_windows_installer.py
uv run --python 3.11 python scripts/verify_release_payload.py `
  --manifest release/generated/release-manifest.json `
  --payload-dir release/generated `
  --checksums release/generated/SHA256SUMS
```

Then run the three acceptance scripts with the generated setup and lifecycle fixtures. They are
repository validation tools; ordinary users never run them.

## Signing

CI reads these GitHub Actions secrets only at signing time:

```text
WINDOWS_SIGNING_PFX_BASE64
WINDOWS_SIGNING_PASSWORD
WINDOWS_SIGNING_TIMESTAMP_URL
```

Set all three or none. Signing uses SHA-256 and an HTTPS RFC 3161 timestamp. CI verifies the
bootstrapper before embedding it, verifies setup after signing, and records the publisher subject
and validation result. Temporary certificate material is deleted. Certificates, passwords,
tokens, and private keys must never be committed.

An unsigned RC is allowed for private physical testing and is labelled `unsigned prerelease` in
the manifest, title/notes, and acceptance evidence. A stable release fails before publication
when signing is unavailable or invalid. A self-signed certificate is not a public publisher
identity.

## Tag workflow and draft-first publication

After all changes and evidence are ready:

```powershell
git add --all
git commit -m "release: native Windows installer 0.1.0rc7"
git tag -a v0.1.0-rc.7 -m "Swarm Inference 0.1.0rc7"
git push origin HEAD
git push origin v0.1.0-rc.7
```

`.github/workflows/release.yml` checks out that tag recursively and independently repeats source,
static, unit, integration, productization, wheel-isolation, installer, repair, upgrade, rollback,
and uninstall gates. It generates the manifest, SHA256SUMS, CycloneDX SBOM, and provenance
attestation.

The authenticated official `gh` CLI creates a draft against the existing immutable tag and
uploads every checksummed asset. CI re-downloads the entire draft into a fresh directory, verifies
all hashes, and clean-installs the downloaded setup with a CPU override and sanitized PATH. Only
then does it mark an RC as a prerelease. It retrieves and verifies the published assets again and
runs the downloaded setup acceptance again. No third-party release action is required.

Required public assets include:

```text
SwarmInferenceSetup-x64.exe
swarm_inference_lab-<version>-py3-none-any.whl
release-manifest.json
SHA256SUMS
swarm-inference-sbom.json
productization-acceptance.zip
```

The release also publishes the small locked/embedded payload components named by the manifest so
external verification can recompute every identity.

## Rollback and revoking a bad release

Setup keeps the working runtime until the staged replacement passes import, version, doctor, state
compatibility, and service readiness. Any failure restores the old runtime, record, service
definition, and prior running state. Do not move the tag to fix a bad build.

For a bad published release:

1. Mark the GitHub Release as a draft or delete the release entry if policy permits.
2. Leave the Git tag unchanged as an audit record.
3. Publish release notes explaining the revocation.
4. Increment the RC or patch version, commit the fix, and create a new annotated tag.
5. Run the complete workflow again.

If automation cannot safely edit the release after a late failure, leave it draft and do not
claim it as published. The publisher script refuses to overwrite an existing published release.
