# P0 starting state

Captured before product-behaviour changes on 2026-08-09T07:26:18.9263261+10:00.

## Repository identity

- Branch: `main`
- Commit: `cd35c6950722a89b1f91f0fa45031042587f5c13`
- Describe: `v0.1.0-rc.11-6-gcd35c69`
- Initial tracked-worktree status: clean
- Package: `swarm-inference-lab==0.1.0rc11`
- Supported Python metadata: `>=3.11,<3.14`; `.python-version` selects `3.11`
- Host: Windows 11, build 26200, AMD64 (single physical machine)
- Initial active Python: CPython 3.11.9, OpenSSL 3.0.13
- Reproduction Python: CPython 3.13.14, OpenSSL 3.0.21, cryptography 49.0.0

## Locked dependency identity

- `uv.lock` SHA-256: `A48749361FD32174CA5620AD84437F373556CC12262D19A54D65844D27C01085`
- `uv.lock` git blob: `0b420dbecae58fa30f6e06204576d28d0be7ae13`
- `pyproject.toml` SHA-256: `B3171FEB1B7EFE69E3248560F9FA997EAC2800FE743C71AE137D9C7276415870`
- Build backend: `hatchling==1.31.0`
- Runtime dependencies include `transformers>=4.51,<5`.
- The `dev` extra includes `hypothesis>=6.100`, pytest, pytest-asyncio, pytest-cov,
  ruff, mypy, and typing stubs.
- Hardware extras are mutually exclusive: `cpu`, `cuda`, and `mps`.
- The initial checked-in `.venv` was not a trustworthy environment: the package and
  pytest-asyncio were unavailable to its test invocation. A separate environment created with
  pinned uv 0.12.0 and `uv sync --locked --extra cpu --extra dev` resolved 96 packages and
  installed the project, `transformers==4.57.6`, and `hypothesis==6.163.0`.

## Colibri and workflows

- `.gitmodules` maps `third_party/colibri` to `https://github.com/JustVugg/colibri.git`.
- Superproject gitlink pin: `b085b48888a88d9a1c00b151a9979774b72cdbfd`.
- Populated submodule HEAD: `b085b48888a88d9a1c00b151a9979774b72cdbfd`.
- Populated submodule worktree: clean.
- CI workflows: `.github/workflows/productization.yml`, `installer.yml`, and `release.yml`.
- Product CI checks out submodules recursively, verifies Colibri source, runs locked Python
  3.11 platform jobs, and runs Python 3.12/3.13 compatibility jobs.
- Release/installer workflows pin Python 3.11.9 and uv 0.12.0 and include wheel isolation,
  acceptance, manifest/checksum/SBOM/provenance, and Windows installer lifecycle gates.

## Test classifications

Registered markers are `gpu`, `model_download`, `physical`, and `slow`.

- Three CUDA experiment integration files use `gpu + model_download + slow`.
- Two Kimi evidence tests use `gpu`.
- `tests/physical/test_manual_physical.py` is the only `physical` test and explicitly requires
  at least two provisioned physical hosts.
- No separate MPS, WAN, or external-model marker is registered.

## Initial commands and failures

Locked environment setup (workspace-isolated):

```powershell
$env:UV_PROJECT_ENVIRONMENT = '<repo>/.tmp/p0-baseline-env311'
$env:UV_CACHE_DIR = '<repo>/.uv-cache'
build/toolchain/uv-0.12.0/uv.exe sync --locked --python 3.11 --extra cpu --extra dev
```

Promotion ledger reproduction on Python 3.11:

```powershell
.tmp/p0-baseline-env311/Scripts/python.exe -m pytest tests/unit/test_canonical_benchmark_contracts.py::test_promotion_ledger_covers_contract_mechanisms_and_importable_product_owners -q
```

Result: **1 failed**. `swarm_inference.backends.colibri.expert_bank` did not exist.

TLS control run on Python 3.11:

```powershell
.tmp/p0-baseline-env311/Scripts/python.exe -m pytest tests/integration/test_secure_wan_transport.py tests/unit/test_network_measurements.py -q
```

Result: **11 passed**.

TLS reproduction on Python 3.13:

```powershell
.tmp/p0-baseline-env313/Scripts/python.exe -m pytest tests/integration/test_secure_wan_transport.py tests/unit/test_network_measurements.py -q
```

Result: **5 failed, 6 passed**. All five failures were verified TLS handshakes rejected by
OpenSSL 3.0.21 with `CERTIFICATE_VERIFY_FAILED: Missing Authority Key Identifier`.

