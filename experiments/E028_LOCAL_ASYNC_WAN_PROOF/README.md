# E028_LOCAL_ASYNC_WAN_PROOF

Hypothesis, fixed before collecting E028 measurements: multi-token verification
and a window of independent provisional chunks can reduce WAN latency sensitivity
enough to meet the unchanged PASS or PASS_STRONG thresholds in `config.json`.
The null is that causal draft dependencies, rejection, rollback, or network waits
prevent the required benefit.

This experiment uses only the local RTX 5090 and the existing target/MTP GGUFs.
No old experiment is rerun. Existing llama.cpp stage support and wire framing
are reused as implementation dependencies, not as performance evidence.

Evidence is separated into **PHYSICAL** single-GPU stage measurements and runtime
correctness, **SHAPED NETWORK** local delay validation, and **PHYSICALLY GROUNDED
MODEL** trace-driven simulation of three independent virtual GPUs. Local wall
time is never a measurement of physical three-GPU throughput.

K means speculative draft tokens beyond the leading token: a full verification
chunk has K+1 rows. Later chunks can contain entirely provisional tokens.
Every chunk records its parent, position, epoch, candidates, stage operations,
verification, commitment and invalidation. Greedy token identity uses the same
scalar llama decoding kernels at every K/W; logical rows are coalesced on the wire.

Checkpointing uses llama's generic sequence-copy API at chunk boundaries. KV
prefix cells are shared; recurrent state is copied on write by llama. Rejection
restores a boundary checkpoint and replays only the accepted prefix. The entire
KV context is not copied W times. The pinned llama adapter contains existing
architecture support; E028 adds no architecture-specific numerical kernels.

Run instructions and exact evidence provenance are recorded with the completed
report. Acceptance criteria and all eight prompts are versioned here before runs.


Reproduce from the repository root (PowerShell, existing local runtime required):

```powershell
.venv/Scripts/python.exe scripts/experiment_028_prepare.py
scripts/experiment_028_build.cmd
.venv/Scripts/python.exe scripts/experiment_028_run.py prepare
.venv/Scripts/python.exe scripts/experiment_028_run.py profile
.venv/Scripts/python.exe scripts/experiment_028_run.py collect
.venv/Scripts/python.exe scripts/experiment_028_run.py stress
.venv/Scripts/python.exe scripts/experiment_028_calibrate_idle.py
.venv/Scripts/python.exe scripts/experiment_028_run.py validate
.venv/Scripts/python.exe scripts/experiment_028_run.py sweep
.venv/Scripts/python.exe scripts/experiment_028_audit.py
.venv/Scripts/python.exe scripts/experiment_028_run.py report
.venv/Scripts/python.exe scripts/experiment_028_verify_results.py
.venv/Scripts/python.exe -m pytest tests/experiments/test_experiment_028_simulator.py -q
```

Collection resumes completed E028 trace files. For a fresh rerun, preserve the
previous experiment output directory first; do not mix traces from different
runtime versions. See `methods.md` for resource assumptions, metric definitions,
protocol details and unavailable metrics. Native builds use the existing pinned
llama libraries and do not contact a package registry or download a model.
