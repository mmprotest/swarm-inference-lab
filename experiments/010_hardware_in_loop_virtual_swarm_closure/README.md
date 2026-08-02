# Experiment 010: Hardware-in-the-Loop Virtual Swarm Closure

Experiment 010 is the final broad single-workstation experiment. It extends the
existing universal worker ABI, Colibri integration, executable tensor paths,
positive-utility planner, and simulator. It does not create a second inference
stack.

The experiment has four modes:

- `quick`: deterministic protocol and fixture smoke tests; never official.
- `development`: shortened Level A real-model work; never an official verdict.
- `full`: all required model levels, matrices, repeats, failures, held-out
  simulator validation, plots, and an official verdict.
- `frontier`: optional additional evidence from a complete frontier model.

Native Windows quick run:

```powershell
.\experiments\010_hardware_in_loop_virtual_swarm_closure\reproduce.ps1 -Quick
```

Official resumable run:

```powershell
.\experiments\010_hardware_in_loop_virtual_swarm_closure\reproduce.ps1 -Full -Resume
```

Only `-Full` can produce an official verdict. Unsupported hardware or backend
features remain null and fail their gates; they are never replaced with
fixture, emulated, or projected values.

## Reuse map

| Experiment 010 concern | Existing source of truth |
|---|---|
| Worker identity, jobs, lifecycle | Universal Worker ABI and process service |
| Colibri execution and inventory | Experiment 009 backend, probe, telemetry, replay |
| Native quantisation descriptors | Experiment 009 inventory and Experiment 006 expert ABI |
| Direct peer traffic | Persistent peer data-plane implementation |
| Hardware, PCIe, storage sampling | Experiment 008 profiler |
| Utility selection | Experiment 007/008 positive-utility planning |
| Network semantics | Existing network emulator, applied before real socket writes |
| Evidence persistence | Experiment 008/009 atomic bundle convention |
| Failure, integrity, quarantine | Expert-boundary extension; the existing stage `FaultProxy` is structurally incompatible and the reason is recorded in the manifest |

The Experiment 009 fixed-replay tuner remains the token-level source of truth.
Experiment 010 does not feed operator-vector results to that tuner as fabricated
tokens: its execution is deferred until the Colibri hook continues generation,
and this is recorded as part of failed Gate 4.

See `IMPLEMENTATION_MAP.md` and `docs/experiment-010-decisions.md` for the
protocol and architecture decisions.
