# Swarm Inference Lab P0 completion report

Status: **P0 PASS**  
Completed: 2026-08-09 (Australia/Sydney)  
Validation boundary: one Windows 11 physical machine; local processes and loopback networking only.

## Manager summary

On a clean single-machine development environment, the repository builds, installs, imports, compiles, and passes every test that does not inherently require another physical machine, an unavailable operating-system/hardware class, or unavailable heavyweight external-model evidence.

- Authoritative unique Python 3.11 corpus: **1,039 passed, 0 failed, 5 unavailable**. The five unavailable cases are four opt-in heavyweight real-model evidence consumers and one test requiring at least two physical hosts.
- Supported interpreter security/core matrix: Python 3.11, 3.12, and 3.13 all pass. Python 3.13 TLS now passes with verification enabled.
- Promotion owners and evidence: all canonical references resolve and are provenance-anchored; product source has no imports from experiment implementations.
- Colibri: submodule pin populated and verified at `b085b48888a88d9a1c00b151a9979774b72cdbfd`; local CPU/native/CUDA integration tests pass; clean wheel and installer do not depend on the development checkout.
- Release engineering: wheel isolation, Windows installer lifecycle, downgrade protection, rollback, checksums, SBOM, provenance, and manifest/payload verification pass locally. Nothing was published.
- Defects fixed: stale Experiment 009 owner, stale evidence references, incorrect promotion classifications, incomplete X.509 profiles, Colibri availability classification and build hygiene, Qwen3-MoE artifact/profile validation gaps, CUDA device-index normalization, strict process-log handling, locked CI syncs, and an acceptance-suite scope bug.
- Remaining external prerequisites: four heavyweight real-model evidence archives/weights, a second physical host, and non-Windows/MPS hardware. These were not simulated and no physical multi-machine validation is claimed.
- Safe to proceed to P1: **yes, within the stated one-machine boundary**.

## 1. Starting state

The starting state was captured before behavioral edits in [STARTING_STATE.md](STARTING_STATE.md) and [starting_state.json](starting_state.json).

| Item | Initial value |
|---|---|
| Branch / commit | `main` / `cd35c6950722a89b1f91f0fa45031042587f5c13` |
| Git description | `v0.1.0-rc.11-6-gcd35c69` |
| Initial tracked worktree | clean |
| Package version | `0.1.0rc11` |
| Supported Python metadata | `>=3.11,<3.14`; `.python-version` is 3.11 |
| OS | Windows 11, build 26200, AMD64 |
| `uv.lock` | SHA-256 `a48749361fd32174ca5620ad84437f373556cc12262d19a54d65844d27c01085`; Git blob `0b420dbecae58fa30f6e06204576d28d0be7ae13` |
| `pyproject.toml` | SHA-256 `b3171feb1b7efe69e3248560f9fa997eac2800fe743c71ae137d9c7276415870` |
| Colibri pin | `b085b48888a88d9a1c00b151a9979774b72cdbfd`, populated and clean |
| Colibri URL | `https://github.com/JustVugg/colibri.git` |

Dependency groups, markers, CI/release workflows, `.gitmodules`, lockfiles, promotion contracts, and initial commands are enumerated in the starting-state artifacts. `transformers` was already correctly declared as a runtime dependency and `hypothesis` in the development/test group, both locked. Clean `uv sync --locked` environments proved that no declaration change was required.

Initial failures reproduced:

1. Canonical owner resolution failed because Experiment 009 named nonexistent module `swarm_inference.backends.colibri.expert_bank`.
2. On Python 3.13/OpenSSL 3.0.21, the secure-WAN slice produced five failures and six passes; verification rejected generated certificates with `Missing Authority Key Identifier`.
3. Several canonical evidence paths were stale, and one evidence-only Experiment 007 reference was incorrectly treated as a product owner.
4. The supplied audit concern about an absent Colibri checkout did not reproduce in this working copy: the exact gitlink was populated. Reproducibility and clean-checkout CI handling were nevertheless audited and strengthened.

## 2. Defects and exact fixes

### Promotion ledger and provenance

- Experiment 009 `expert_bank` now resolves to the real product implementation, `swarm_inference.backends.colibri.torch_backend.ColibriMoeBackend`; no compatibility shim was created.
- Corrected stale evidence roots for Experiments 001, 002, 003, 006, 007, and 010.
- Reclassified Experiment 007 held-out planner evaluation as evidence-only because it is a validation method, not a product implementation.
- Corrected Experiment 010 hardware provenance from a physical-distribution claim to single-machine isolated whole-expert/microshard workers.
- Added `benchmarks/canonical/evidence_provenance.yaml`, with immutable SHA-256 anchors for Experiments 001–004 and 006–011.
- Strengthened the canonical contract test to resolve modules, classes/functions, files, and YAML; require product owners to originate outside experiment code; verify evidence files and hashes; and AST-scan all non-experiment product source for experiment imports.
- Removed the obsolete Experiment 010 evidence-module allowlist from runtime-line reporting.

Result: every canonical owner and evidence reference resolves, provenance tests pass, and no non-experiment product source imports `swarm_inference.experiments`.

### TLS and Python 3.13

- CA certificates now have critical CA Basic Constraints, CA Key Usage, Subject Key Identifier, and self Authority Key Identifier.
- Node/client leaf certificates now have non-CA constraints, leaf Key Usage, appropriate server/client Extended Key Usage, DNS/IP SANs, Subject Key Identifier, and issuer-derived Authority Key Identifier.
- Certificate issuance validates that the supplied issuer key matches the issuer certificate and bounds leaf validity to the issuer lifetime.
- Existing TLS 1.3 minimum, `CERT_REQUIRED`, hostname verification, trust-chain checks, route identity, and lease authentication remain enabled.
- Added assertions for certificate profiles plus trusted mTLS, untrusted CA, wrong host, forged same-subject issuer, expired chain, plaintext rejection, route identity, and stage/expert/llama.cpp routes.

Result: the secure-WAN slice is 14/14 on each of Python 3.11, 3.12, and 3.13; the route/lease authentication slice is 16/16. Negative peers continue to be rejected.

### Colibri and dependency reproducibility

- CI now performs recursive submodule checkout and all `uv sync` calls use `--locked`; tests assert both properties.
- Runtime classification distinguishes `RUNTIME_UNAVAILABLE` from `MODEL_UNSUPPORTED`.
- Colibri build scripts clean only their direct generated source/bin output and produce an explicit supported target set, preventing stale binaries from masquerading as supported product paths.
- Verified the exact submodule pin, bridge patches, generic native ABI, local CUDA path, and wheel/installer isolation from `third_party/colibri`.
- The built development output contains only the audited Colibri/Inkling/Kimi and Swarm bridge targets. The product wheel contains neither the submodule source nor imports from it.

Colibri evidence:

| Check | Result |
|---|---|
| Gitlink / checkout | `b085b48888a88d9a1c00b151a9979774b72cdbfd`, clean |
| License SHA-256 | `1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6` |
| Build-source identity | `d4002535267968c140cad816a1df22af2d1c15f6ee1ebc54a82461b4d8ebad44` |
| Bridge tests | 26 passed |
| Broader local slice | 43 passed |
| Generic native ABI | 2 passed (`f32`/`bf16`, `int4-g32`) |
| CUDA | RTX 5090, compute 12.0 / `sm_120`; 2 tests passed; max absolute error `7.28e-11` |
| CUDA DLL SHA-256 | `7874cb42d77290c9ec9cc208587dab4816a0d256d1899e41b44213ca43c218ee` |

### Additional release-blocking defects found during the surrounding audit

- Qwen3-MoE adapters now validate expert topology, top-k/intermediate dimensions, complete per-layer expert banks, expert memory/cost metadata, artifact identity, and model fingerprints.
- The canonical Qwen3-MoE correctness test loads the stage and experts from one real content-addressed immutable artifact and derives exact KV-cache bytes from the runtime cache specification.
- Experiment 002 now pins the external model revision and recognizes only the exact bounded benign gRPC `CancelledError` emitted immediately before orderly worker shutdown; any other traceback remains fatal. The real four-process local run passes.
- CUDA device equality normalizes only equivalent CUDA indices (`cuda`/`cuda:0`) while retaining cross-device rejection.
- Product acceptance no longer misclassifies opt-in heavyweight evidence/native-DLL tests as mandatory software repeatability tests. Mandatory repeatability still rejects every skip, error, failure, unexpected termination, forced kill, and leak.
- Locked formatting/type/static checks exposed and removed stale imports and inconsistent source formatting; no architectural redesign was introduced.

## 3. Authoritative one-machine test matrix

Counts below keep the unique corpus separate from deliberately overlapping interpreter and release reruns.

| Matrix slice | Python | Passed | Failed | Skipped/not run | Result |
|---|---:|---:|---:|---:|---|
| Full default suite | 3.11 | 1,031 | 0 | 13 | PASS |
| Locally dischargeable default skips (unique) | 3.11 | 8 | 0 | 0 | PASS |
| **Authoritative unique corpus** | **3.11** | **1,039** | **0** | **5** | **PASS** |
| Unit suite repeat | 3.12 | 914 | 0 | 0 | PASS |
| Integration/failure repeat | 3.12 | 38 | 0 | 9 | PASS |
| Unit suite repeat | 3.13 | 914 | 0 | 0 | PASS |
| Integration/failure repeat | 3.13 | 38 | 0 | 9 | PASS |
| TLS secure-WAN focused (each interpreter) | 3.11/3.12/3.13 | 14 each | 0 | 0 | PASS |
| Route/lease authentication focused | 3.11 | 16 | 0 | 0 | PASS |
| Installer Python contracts | 3.11 | 52 | 0 | 0 | PASS |
| Productization contract | 3.11 | 21 | 0 | 0 | PASS |
| Productization mandatory software gates | 3.11 | 207 | 0 | 0 | PASS |

The 13 default skips consisted of three opt-in real CUDA experiments, three Experiment 007 archive validations, two generic native Colibri-DLL tests, four universal heavyweight real-model evidence consumers, and one physical-host test. The first eight were subsequently executed and passed on this machine. The five remaining unavailable cases are legitimate external prerequisites:

- four opt-in consumers whose real-model baseline, restart/replay, whole-expert, and native-microshard evidence archives/weights are absent;
- one topology validation requiring at least two physical hosts.

No test was weakened, deleted, or mocked to change these outcomes. No second machine, WAN endpoint, MPS host, or non-Windows host was used or claimed.

Python matrix details:

| Python | OpenSSL | `cryptography` | Outcome |
|---|---|---:|---|
| 3.11.9 | 3.0.13 | 49 | core/full and TLS pass |
| 3.12.11 | 3.0.16 | 49 | unit/integration and TLS pass |
| 3.13.14 | 3.0.21 | 49 | unit/integration and TLS pass |

Static and packaging gates: `compileall` passes; Ruff checks and formatting pass over 510 files; MyPy reports zero issues in 335 source files; `git diff --check` passes; clean imports and CLI doctor pass from wheel-only environments on Python 3.11, 3.12, and 3.13.

## 4. Productization and release evidence

Final productization acceptance is `SOFTWARE_ACCEPTANCE_PASS`: 33 software gates and 207 tests passed with zero failures, errors, or skips. Three full-suite repeatability runs each passed 38 tests; five stage-ring runs each passed eight tests. All processes stopped gracefully with no unexpected termination, forced kill, leak, or warning. Eight non-software gates were explicitly `NOT_RUN`: four heavyweight real-model gates and four physical/platform configurations.

| Artifact/check | Result |
|---|---|
| Wheel | `swarm_inference_lab-0.1.0rc11-py3-none-any.whl`, 1,425,262 bytes |
| Wheel SHA-256 | `93046fa27441d7327bb6022011148cef05e1101eb116cb72a3a974579fbd2ced` |
| CPU runtime profile | `733b680a32dc31a5c7796655059291851d41282cded43c6f73faee197033b35e` |
| CUDA runtime profile | `df0411fc046f480838d759508757993fdb168c53b172f09d0e7f9ff2169a0e1d` |
| Windows setup | `SwarmInferenceSetup-x64.exe`, 601,772,732 bytes, unsigned prerelease |
| Setup SHA-256 | `e235378c545ec4ef6d3a001c1b83f7024205ce66efa1034128875c16a4bdb901` |
| Release manifest SHA-256 | `2ec4d283e0cca72299450d1f314244bd5bb6ae391d62cb88f47976ba3e10586d` |
| `SHA256SUMS` SHA-256 | `42ec0b39f79ecb5bcf36985c42e64b91a14ce032c1460255186de9ffcde29215` |
| SPDX SBOM SHA-256 | `b2365d16eb2dd9ec176d348f7a7b5850216d1199b208e32d164d9e1f88d381e6` |
| Acceptance archive SHA-256 | `1e8943eb985a6e314eb7e04bc34f92f61d35b56043de50eb081aab3da92d722f` |
| Manifest and payload | PASS; 13 listed payload files verified |

Windows lifecycle acceptance performed real sanitized-profile install, import/doctor, repair, and uninstall operations. Upgrade `rc11 -> rc12`, downgrade rejection (exit 7), explicit downgrade, intentional broken-`rc13` rejection (exit 7), rollback to `rc12`, durable-state preservation, reinstall, normal uninstall, and explicit purge all pass. Durable machine-readable results are in `installer-clean.json`, `installer-silent.json`, `installer-upgrade.json`, and `installer-uninstall.json`.

The setup is deliberately identified as an unsigned prerelease because this was a local dirty/untagged validation build. It was not published and is not represented as a signed stable release.

## 5. No-OLMoE gate

There is no active Swarm Inference Lab support path, test, documentation, configuration, script, benchmark, example, CI path, or release payload for OLMoE. `tests/unit/test_forbidden_model_support.py` enforces this across active tracked project surfaces without embedding a self-matching literal.

The immutable upstream Colibri submodule pin contains dormant upstream files for additional architectures, including this one. Those third-party historical sources were not altered. Swarm's explicit build target allowlist excludes them, generated output is cleaned before building, the audited output has no corresponding binary, and neither the wheel nor installer packages the submodule. This is an upstream-source boundary, not active product support.

## 6. Files changed

The final tracked content diff before adding this report was 47 modified files, 721 insertions, and 259 deletions, plus three new implementation/provenance files and the P0 evidence artifacts. `git diff --check` was clean. Git porcelain also reports a worktree stat touch for `src/swarm_inference/model/resolver.py`, but its worktree and HEAD blob hashes are both `b9d16fbb9f5ba24a67aad4f70921cbea34dae160` and its content diff is empty. The exact machine-readable file list is in [P0_COMPLETION_REPORT.json](P0_COMPLETION_REPORT.json).

Behavioral changes are confined to:

- promotion contracts/provenance, CI locking, and Colibri build hygiene;
- TLS PKI generation and strict security tests;
- Colibri availability classification and Qwen3-MoE product validation;
- bounded experiment/process diagnostics and product acceptance scoping;
- associated correctness, integration, release, and regression tests.

Other touched Python files contain only locked Ruff formatting, typing cleanup, or imports required by the above changes. There is no unrelated rewrite.

## 7. Reproduction commands

Run from the repository root in PowerShell. The test/build environments used the checked-in lock and did not rely on globally installed Python packages.

```powershell
git submodule update --init --recursive
uv sync --locked --python 3.11 --all-extras --group dev
uv sync --locked --python 3.12 --all-extras --group dev
uv sync --locked --python 3.13 --all-extras --group dev

uv run --locked --python 3.11 python -m compileall -q src
uv run --locked --python 3.11 ruff format --check .
uv run --locked --python 3.11 ruff check .
uv run --locked --python 3.11 mypy src
uv run --locked --python 3.11 pytest -q

uv run --locked --python 3.11 pytest -q tests/integration/test_secure_wan_transport.py
uv run --locked --python 3.12 pytest -q tests/integration/test_secure_wan_transport.py
uv run --locked --python 3.13 pytest -q tests/integration/test_secure_wan_transport.py
uv run --locked --python 3.11 pytest -q tests/unit/test_canonical_benchmark_contracts.py tests/unit/test_forbidden_model_support.py

& integrations/colibri/build.ps1 -ApplyBridgePatches -BuildCuda -PythonPath .tmp/p0-env311-cuda/Scripts/python.exe
uv run --locked --python 3.11 pytest -q tests/unit/test_colibri_source_contract.py tests/unit/test_colibri_support_classification.py

uv build --wheel --out-dir release/generated
uv run --locked --python 3.11 python scripts/test_wheel_install.py --wheel release/generated/swarm_inference_lab-0.1.0rc11-py3-none-any.whl --python 3.11 --output artifacts/p0_baseline/wheel-isolation.json
uv run --locked --python 3.12 python scripts/test_wheel_install.py --wheel release/generated/swarm_inference_lab-0.1.0rc11-py3-none-any.whl --python 3.12 --output artifacts/p0_baseline/wheel-isolation-py312.json
uv run --locked --python 3.13 python scripts/test_wheel_install.py --wheel release/generated/swarm_inference_lab-0.1.0rc11-py3-none-any.whl --python 3.13 --output artifacts/p0_baseline/wheel-isolation-py313.json

uv run --locked --python 3.11 python scripts/run_productization_acceptance.py run --output artifacts/acceptance --run-repeatability
uv run --locked --python 3.11 python scripts/build_windows_installer.py --output-dir release/generated --acceptance-zip release/generated/swarm-inference-acceptance.zip
uv run --locked --python 3.11 python scripts/verify_release_manifest.py release/generated/release-manifest.json --payload-dir release/generated
uv run --locked --python 3.11 python scripts/verify_release_payload.py --manifest release/generated/release-manifest.json --payload-dir release/generated --checksums release/generated/SHA256SUMS

& scripts/test_windows_installer.ps1 -SetupPath release/generated/SwarmInferenceSetup-x64.exe -EvidencePath artifacts/p0_baseline/installer-clean.json
& scripts/test_windows_uninstall.ps1 -SetupPath release/generated/SwarmInferenceSetup-x64.exe -EvidencePath artifacts/p0_baseline/installer-uninstall.json
```

Upgrade/rollback reproduction uses `scripts/test_windows_upgrade.ps1` with the locally generated `rc11`, fixture `rc12`, and intentional broken `rc13` setup paths recorded in `installer-upgrade.json`. No release publication command is part of this validation.

## 8. Final acceptance decision

All P0 gates that are valid on this physical machine pass. Every non-run gate has an explicit external-model, physical-host, operating-system, or hardware prerequisite; no unsupported capability was relabeled as supported. TLS verification was strengthened, not bypassed. Colibri is reproducibly pinned and isolated from packaging. Promotion provenance is machine-validated. The repository is safe to proceed to P1 within the stated boundary.
